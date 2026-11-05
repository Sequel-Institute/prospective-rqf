"""LRA Pathfinder / Path-X sequence classification.

Long Range Arena convention (Tay et al. 2021, as used by S4/S5/LRU): grayscale
``resolution x resolution`` images are raster-scanned row-major into sequences
of length ``resolution**2`` with one real feature per timestep. The binary
label says whether the two dots are connected by a path. LRA uses the
``curv_contour_length_14`` ("hard") difficulty for both Pathfinder
(resolution 32, T=1024) and Path-X (resolution 128, T=16384): 200k examples
split 80/10/10. The split is a seeded random permutation (default seed 42,
the S4-lineage convention) rather than LRA's deterministic TFDS slices.

Data layout is the official ``lra_release`` archive's:
``<root>/lra_release/lra_release/pathfinder<R>/<difficulty>/{imgs,metadata}``
(a bare ``<root>/pathfinder<R>`` is also accepted). The ``metadata/*.npy``
files are plain text despite the extension: one example per line, with the
image subdirectory in field 0, the filename in field 1, and the binary label
in field 3. Empty or undecodable PNGs are skipped with a warning. The archive
also contains a truncated top-level ``lra_release/pathfinder128`` duplicate;
only the nested path above is searched.

Decoded images are cached once as uint8 tensors under
``<pathfinder_dir>/processed_data/<difficulty>``; the ~200k PNGs are decoded
(with Pillow) only when that cache is absent. A cache written by the original
preparation pipeline has the same layout and is reused. Inputs are
standardized with the scalar mean/std of the train split. This z-scoring is
a repository choice (official LRA at most
rescales to [0, 1]; S4's loader maps to [-1, 1]) that the forward-preserving
``w_init`` assumes. Pathfinder images are ~96% black, so path pixels land
several standard deviations above zero.
"""

from __future__ import annotations

import json
import os
import pickle
import shutil
import tempfile
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import TensorDataset

PATHFINDER_CLASS_NAMES = ("disconnected", "connected")
PATHFINDER_DIFFICULTY = "curv_contour_length_14"
_CACHE_COMPLETE = "complete.json"

# Official curv_contour_length_14 release: 200 metadata shards of 1000
# examples each at every resolution (checked against the lra_release archive
# for 32 and 128). Other difficulties ship different, unpinned counts.
_OFFICIAL_METADATA_SHARDS = 200
_OFFICIAL_NUM_EXAMPLES = 200_000


@dataclass(frozen=True, slots=True)
class PathfinderDataSpec:
    """LRA Pathfinder / Path-X splits.

    Args:
        root: Directory holding the extracted LRA release (see module doc).
        resolution: Image side length: 32 (Pathfinder) or 128 (Path-X);
            64 and 256 exist in the release and are accepted too.
        difficulty: Pathfinder difficulty subdirectory; LRA uses
            ``curv_contour_length_14`` for both Pathfinder and Path-X.
        train_split / val_split: Fractions for the seeded random split
            (test gets the remainder; LRA/S4 use 0.8/0.1/0.1).
        split_seed: Seed for the split permutation (S4 uses 42).
        normalize: Standardize with the scalar mean/std of the train split.
        verify_complete: Require the official release's counts for the
            official difficulty (200 metadata shards, 200,000 examples) both
            when decoding and when validating an existing cache, so a
            partially extracted archive cannot silently become a smaller
            benchmark with a permanent "complete" cache. No effect for other
            difficulties; set ``False`` for deliberate subsets such as
            synthetic test data.
        max_samples_per_split: Truncate each split (smoke runs / tests).
    """

    root: str = "./data"
    resolution: int = 32
    difficulty: str = PATHFINDER_DIFFICULTY
    train_split: float = 0.8
    val_split: float = 0.1
    split_seed: int = 42
    normalize: bool = True
    verify_complete: bool = True
    max_samples_per_split: int | None = None

    def __post_init__(self) -> None:
        if self.resolution not in (32, 64, 128, 256):
            raise ValueError(f"resolution must be one of 32/64/128/256, got {self.resolution}")
        if not 0 < self.train_split < 1 or not 0 < self.val_split < 1:
            raise ValueError("train_split and val_split must be in (0, 1)")
        if self.train_split + self.val_split >= 1:
            raise ValueError("train_split + val_split must leave a test remainder")
        if self.max_samples_per_split is not None and self.max_samples_per_split <= 0:
            raise ValueError("max_samples_per_split must be positive when provided")

    @property
    def sequence_length(self) -> int:
        return self.resolution * self.resolution

    def build(self) -> tuple[TensorDataset, TensorDataset, TensorDataset]:
        """Load (or decode and cache) the images; return train/val/test datasets.

        Each dataset yields ``(x, y)`` with ``x`` float32 ``[T, 1]`` and ``y``
        a scalar long label.
        """

        images, labels = self._load_or_decode()
        n = images.shape[0]
        generator = torch.Generator().manual_seed(self.split_seed)
        perm = torch.randperm(n, generator=generator)
        n_train = int(self.train_split * n)
        n_val = int(self.val_split * n)
        splits = {
            "train": perm[:n_train],
            "val": perm[n_train : n_train + n_val],
            "test": perm[n_train + n_val :],
        }
        limit = self.max_samples_per_split
        if limit is not None:
            splits = {name: idx[:limit] for name, idx in splits.items()}

        # uint8 [N, T] -> float32 [n, T, 1] per split, standardized with the
        # scalar mean/std of the train split.
        train_x = images[splits["train"]].float().div_(255.0).unsqueeze_(-1)
        mean = train_x.mean() if self.normalize else torch.tensor(0.0)
        std = train_x.std().clamp_min(1e-5) if self.normalize else torch.tensor(1.0)
        datasets: dict[str, TensorDataset] = {}
        for name, idx in splits.items():
            x = train_x if name == "train" else images[idx].float().div_(255.0).unsqueeze_(-1)
            if self.normalize:
                x = x.sub_(mean).div_(std)
            datasets[name] = TensorDataset(x, labels[idx])
        return datasets["train"], datasets["val"], datasets["test"]

    def pathfinder_dir(self) -> Path:
        """Locate ``pathfinder<resolution>`` under the accepted layouts."""

        root = Path(self.root)
        name = f"pathfinder{self.resolution}"
        candidates = (
            root / "lra_release" / "lra_release" / name,
            root / name,
        )
        for candidate in candidates:
            if (candidate / self.difficulty / "metadata").exists():
                return candidate
        raise FileNotFoundError(
            f"Pathfinder data not found; expected {candidates[0]}/{self.difficulty} "
            f"(or {candidates[1]}/...). Extract the LRA release under root='{self.root}'."
        )

    def _cache_dir(self) -> Path:
        return self.pathfinder_dir() / "processed_data" / self.difficulty

    def _expected_examples(self) -> int | None:
        """Example count to enforce, or ``None`` when completeness is unknowable."""

        if self.verify_complete and self.difficulty == PATHFINDER_DIFFICULTY:
            return _OFFICIAL_NUM_EXAMPLES
        return None

    def _load_or_decode(self) -> tuple[Tensor, Tensor]:
        cache_dir = self._cache_dir()
        cached = self._load_cache(cache_dir, require_marker=True)
        if cached is not None:
            return cached

        cache_parent = cache_dir.parent
        cache_parent.mkdir(parents=True, exist_ok=True)
        lock_path = cache_parent / ".cache.lock"
        while True:
            with _exclusive_lock(lock_path):
                self._reclaim_stale_staging(cache_parent)
                marker_token = self._cache_marker_token(cache_dir)
                if marker_token is None:
                    # Validate and adopt a cache written without a completion
                    # marker. A partial two-file cache is rebuilt from the PNGs.
                    cached = self._load_cache(cache_dir, require_marker=False)
                    if cached is not None:
                        self._write_cache_marker(cache_dir, *cached)
                        return cached

                    images, labels = self._decode_all()
                    if not self._cache_tensors_valid(images, labels):
                        raise RuntimeError(
                            "decoded Pathfinder tensors have invalid shapes or dtypes"
                        )

                    staged = Path(
                        tempfile.mkdtemp(prefix=f".{self.difficulty}-", dir=cache_parent)
                    )
                    try:
                        torch.save(images, staged / "images_u8.pt")
                        torch.save(labels, staged / "labels.pt")
                        self._write_cache_marker(staged, images, labels)

                        # Publish both tensors plus their completion marker as
                        # one directory.
                        if cache_dir.exists():
                            shutil.rmtree(cache_dir)
                        os.replace(staged, cache_dir)
                    finally:
                        if staged.exists():
                            shutil.rmtree(staged)
                    return images, labels

            # Another process published while this one waited. Load the large
            # tensors outside the lock so all waiters can read in parallel. If
            # validation fails, remove only the marker generation this process
            # observed, then retry the repair under the lock.
            cached = self._load_cache(cache_dir, require_marker=True)
            if cached is not None:
                return cached
            with _exclusive_lock(lock_path):
                if self._cache_marker_token(cache_dir) == marker_token:
                    (cache_dir / _CACHE_COMPLETE).unlink(missing_ok=True)

    def _reclaim_stale_staging(self, cache_parent: Path) -> None:
        """Remove staging directories left by killed builders.

        Only called while holding the build lock: every publisher stages under
        the same lock, so any ``.<difficulty>-*`` entry visible here belongs to
        a process that died between ``mkdtemp`` and ``os.replace``.
        """

        for stale in cache_parent.glob(f".{self.difficulty}-*"):
            if stale.is_dir():
                shutil.rmtree(stale, ignore_errors=True)

    def _load_cache(
        self, cache_dir: Path, *, require_marker: bool
    ) -> tuple[Tensor, Tensor] | None:
        images_path = cache_dir / "images_u8.pt"
        labels_path = cache_dir / "labels.pt"
        manifest = self._read_cache_manifest(cache_dir) if require_marker else None
        if require_marker and manifest is None:
            return None
        if not images_path.is_file() or not labels_path.is_file():
            return None
        try:
            images = torch.load(images_path, weights_only=True)
            labels = torch.load(labels_path, weights_only=True)
        except (
            EOFError,
            IndexError,
            KeyError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
            pickle.UnpicklingError,
        ):
            return None
        if not self._cache_tensors_valid(images, labels):
            return None
        if manifest is not None and images.shape[0] != manifest["num_examples"]:
            return None
        return images, labels

    def _read_cache_manifest(self, cache_dir: Path) -> dict[str, object] | None:
        try:
            manifest = json.loads((cache_dir / _CACHE_COMPLETE).read_text(encoding="utf-8"))
            if not isinstance(manifest, dict):
                return None
            num_examples = manifest.get("num_examples")
            images_bytes = manifest.get("images_bytes")
            labels_bytes = manifest.get("labels_bytes")
            if (
                manifest.get("version") != 1
                or manifest.get("resolution") != self.resolution
                or manifest.get("difficulty") != self.difficulty
                or not isinstance(num_examples, int)
                or num_examples <= 0
                or not isinstance(images_bytes, int)
                or images_bytes <= 0
                or not isinstance(labels_bytes, int)
                or labels_bytes <= 0
                or images_bytes != (cache_dir / "images_u8.pt").stat().st_size
                or labels_bytes != (cache_dir / "labels.pt").stat().st_size
            ):
                return None
            return manifest
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return None

    def _cache_marker_token(self, cache_dir: Path) -> tuple[int, int, int, int] | None:
        if self._read_cache_manifest(cache_dir) is None:
            return None
        try:
            stat = (cache_dir / _CACHE_COMPLETE).stat()
        except OSError:
            return None
        return stat.st_dev, stat.st_ino, stat.st_mtime_ns, stat.st_size

    def _cache_tensors_valid(self, images: object, labels: object) -> bool:
        expected = self._expected_examples()
        return (
            isinstance(images, Tensor)
            and images.dtype == torch.uint8
            and images.ndim == 2
            and images.shape[0] > 0
            and images.shape[1] == self.sequence_length
            and (expected is None or images.shape[0] == expected)
            and isinstance(labels, Tensor)
            and labels.dtype == torch.long
            and labels.ndim == 1
            and labels.shape[0] == images.shape[0]
            and bool(torch.all((labels == 0) | (labels == 1)))
        )

    def _write_cache_marker(self, cache_dir: Path, images: Tensor, labels: Tensor) -> None:
        manifest = {
            "version": 1,
            "resolution": self.resolution,
            "difficulty": self.difficulty,
            "num_examples": images.shape[0],
            "images_bytes": (cache_dir / "images_u8.pt").stat().st_size,
            "labels_bytes": (cache_dir / "labels.pt").stat().st_size,
        }
        marker = cache_dir / _CACHE_COMPLETE
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{_CACHE_COMPLETE}-",
            dir=cache_dir,
            delete=False,
        ) as fh:
            json.dump(manifest, fh, sort_keys=True)
            fh.write("\n")
            tmp = Path(fh.name)
        try:
            os.replace(tmp, marker)
        finally:
            tmp.unlink(missing_ok=True)

    def _decode_all(self) -> tuple[Tensor, Tensor]:
        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError("Pillow is required to decode the Pathfinder PNGs") from exc

        base = self.pathfinder_dir() / self.difficulty
        metadata_files = sorted((base / "metadata").glob("*.npy"), key=lambda p: int(p.stem))
        if not metadata_files:
            raise FileNotFoundError(f"no metadata files under {base / 'metadata'}")
        expected = self._expected_examples()
        if expected is not None and len(metadata_files) != _OFFICIAL_METADATA_SHARDS:
            raise FileNotFoundError(
                f"found {len(metadata_files)} metadata shard(s) under "
                f"{base / 'metadata'}, expected {_OFFICIAL_METADATA_SHARDS} for the "
                f"official {self.difficulty} release -- incomplete extraction? "
                "Pass verify_complete=False for a deliberate subset."
            )

        seq_len = self.sequence_length
        chunks: list[Tensor] = []
        labels: list[int] = []
        skipped = 0
        print(f"Decoding Pathfinder{self.resolution} PNGs from {base} (one-time cache build)")
        for metadata_path in metadata_files:
            for line in metadata_path.read_text(encoding="utf-8").splitlines():
                fields = line.split()
                if len(fields) < 4:
                    continue
                image_path = base / fields[0] / fields[1]
                try:
                    with Image.open(image_path) as im:
                        gray = im.convert("L")
                        pixels = torch.frombuffer(bytearray(gray.tobytes()), dtype=torch.uint8)
                except (OSError, SyntaxError):
                    skipped += 1
                    continue
                if pixels.numel() != seq_len:
                    skipped += 1
                    continue
                chunks.append(pixels)
                labels.append(int(fields[3]))
        if skipped:
            warnings.warn(
                f"skipped {skipped} unreadable Pathfinder example(s) under {base}",
                stacklevel=2,
            )
        if not chunks:
            raise FileNotFoundError(f"no decodable Pathfinder images under {base}")
        if expected is not None and len(chunks) != expected:
            raise RuntimeError(
                f"decoded {len(chunks)} Pathfinder example(s) under {base}, expected "
                f"{expected} for the official {self.difficulty} release -- missing "
                "or unreadable PNGs? Pass verify_complete=False for a deliberate subset."
            )
        return torch.stack(chunks), torch.tensor(labels, dtype=torch.long)


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    """Serialize first-use preparation with an advisory lock (works on NFSv4/EFS).

    ``fcntl`` is imported here so importing this module works on non-POSIX
    platforms; only first-use preparation takes the lock.
    """

    import fcntl

    with path.open("a+b") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
