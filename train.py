from __future__ import annotations

import copy
import json
import random
import wave
from collections import defaultdict
from dataclasses import fields, is_dataclass, replace
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from rqf_models.cells.implementations.organics_cell import ORGaNICsCell
from rqf_models.cells.state import ORGaNICsState
from rqf_models.embedding.rqf import PureRQFEncoder, S4RQFEncoder


class SequenceClassifier(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        encoder_dim: int,
        num_classes: int,
        pooling: str,
        hidden_dims: int | None,
        activation: str,
        dropout: float,
        per_step_loss: bool,
    ) -> None:
        super().__init__()
        if pooling not in {"mean", "last"}:
            raise ValueError(f"pooling must be 'mean' or 'last', got {pooling!r}")
        activations = {"gelu": nn.GELU, "relu": nn.ReLU}
        if activation not in activations:
            raise ValueError(f"unsupported classifier activation: {activation!r}")
        self.encoder = encoder
        self.pooling = pooling
        self.per_step_loss = per_step_loss
        if hidden_dims is None:
            self.head = nn.Linear(encoder_dim, num_classes)
        else:
            self.head = nn.Sequential(
                nn.Linear(encoder_dim, hidden_dims),
                activations[activation](),
                nn.Dropout(dropout),
                nn.Linear(hidden_dims, num_classes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        if self.per_step_loss:
            batch, time, dim = features.shape
            logits = self.head(features.reshape(batch * time, dim)).reshape(batch, time, -1)
            if self.training:
                return logits
            return logits.mean(dim=1) if self.pooling == "mean" else logits[:, -1, :]
        pooled = features.mean(dim=1) if self.pooling == "mean" else features[:, -1, :]
        return self.head(pooled)


class OrganicsSequenceClassifier(nn.Module):
    def __init__(
        self,
        cells: list[ORGaNICsCell],
        hidden_width: int,
        num_classes: int,
        use_layer_norm: bool,
        skip_connection: bool,
        recurrent_grad_mode: str,
        pooling: str,
        per_step_loss: bool,
        per_step_readout: str,
    ) -> None:
        super().__init__()
        if recurrent_grad_mode not in {"full", "tbptt1"}:
            raise ValueError(f"unsupported recurrent_grad_mode: {recurrent_grad_mode!r}")
        if pooling not in {"mean", "last"}:
            raise ValueError(f"pooling must be 'mean' or 'last', got {pooling!r}")
        if per_step_readout not in {"instant", "causal_mean_local"}:
            raise ValueError(f"unsupported per_step_readout: {per_step_readout!r}")
        self.cells = nn.ModuleList(cells)
        self.head = nn.Linear(hidden_width, num_classes)
        self.use_layer_norm = use_layer_norm
        self.skip_connection = skip_connection
        self.recurrent_grad_mode = recurrent_grad_mode
        self.pooling = pooling
        self.per_step_loss = per_step_loss
        self.per_step_readout = per_step_readout
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_width) for _ in cells[:-1]]
        ) if use_layer_norm else nn.ModuleList()

    @staticmethod
    def _detach_recurrent_state(state):
        if isinstance(state, ORGaNICsState):
            return ORGaNICsState(
                y=state.y.detach(),
                a=state.a.detach(),
                b0=state.b0.detach(),
                b1=state.b1.detach(),
                z_prev=state.z_prev,
            )
        if isinstance(state, torch.Tensor):
            return state.detach()
        if is_dataclass(state) and not isinstance(state, type):
            return replace(
                state,
                **{
                    field.name: OrganicsSequenceClassifier._detach_recurrent_state(
                        getattr(state, field.name)
                    )
                    for field in fields(state)
                },
            )
        if isinstance(state, tuple):
            return type(state)(
                OrganicsSequenceClassifier._detach_recurrent_state(item) for item in state
            )
        if isinstance(state, list):
            return [OrganicsSequenceClassifier._detach_recurrent_state(item) for item in state]
        if isinstance(state, dict):
            return {
                key: OrganicsSequenceClassifier._detach_recurrent_state(value)
                for key, value in state.items()
            }
        return state

    def _maybe_detach_recurrent_state(self, state):
        if self.recurrent_grad_mode == "tbptt1" and torch.is_grad_enabled():
            return self._detach_recurrent_state(state)
        return state

    def _step_cell(self, cell: ORGaNICsCell, layer_input: torch.Tensor, state):
        if (
            self.recurrent_grad_mode == "tbptt1"
            and torch.is_grad_enabled()
            and cell.prospective_input
        ):
            return cell.step(layer_input, state, detach_z_prev_source=True)
        return cell.step(layer_input, state)

    @staticmethod
    def _causal_mean_local(hidden_seq: torch.Tensor) -> torch.Tensor:
        _, seq_len, _ = hidden_seq.shape
        counts = torch.arange(
            1,
            seq_len + 1,
            device=hidden_seq.device,
            dtype=hidden_seq.dtype,
        ).view(1, seq_len, 1)
        with torch.no_grad():
            prefix_sum_detached = torch.cumsum(hidden_seq.detach(), dim=1)
            prev_sum_detached = prefix_sum_detached - hidden_seq.detach()
        return (prev_sum_detached + hidden_seq) / counts

    def _readout_sequence(self, hidden_seq: torch.Tensor, final_state) -> torch.Tensor:
        if self.per_step_loss:
            if self.training:
                if self.per_step_readout == "causal_mean_local":
                    hidden_seq = self._causal_mean_local(hidden_seq)
                return self.head(hidden_seq)
            logits = self.head(hidden_seq)
            return logits.mean(dim=1) if self.pooling == "mean" else logits[:, -1, :]

        if self.pooling == "mean":
            final_hidden = hidden_seq.mean(dim=1)
        else:
            final_hidden = self.cells[-1].output(final_state)
        return self.head(final_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length, _ = x.shape
        states = [cell.init_state_training(batch_size) for cell in self.cells]
        outputs: list[torch.Tensor] = []
        for cell in self.cells:
            cell.compute_cache()
        with parametrize.cached():
            for t in range(seq_length):
                layer_input = x[:, t, :]
                for layer_idx, cell in enumerate(self.cells):
                    states[layer_idx] = self._maybe_detach_recurrent_state(states[layer_idx])
                    states[layer_idx] = self._step_cell(cell, layer_input, states[layer_idx])
                    layer_output = cell.output(states[layer_idx])
                    if self.skip_connection and layer_output.shape == layer_input.shape:
                        layer_output = layer_output + layer_input
                    if self.use_layer_norm and layer_idx < len(self.cells) - 1:
                        layer_output = self.layer_norms[layer_idx](layer_output)
                    layer_input = layer_output
                outputs.append(layer_input)
        return self._readout_sequence(torch.stack(outputs, dim=1), states[-1])


class SyntheticSpeechCommands(Dataset):
    def __init__(self, size: int, sequence_length: int, feature_dim: int, num_classes: int) -> None:
        self.size = int(size)
        self.sequence_length = int(sequence_length)
        self.feature_dim = int(feature_dim)
        self.num_classes = int(num_classes)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        generator = torch.Generator().manual_seed(index)
        x = torch.randn(self.sequence_length, self.feature_dim, generator=generator)
        y = torch.tensor(index % self.num_classes, dtype=torch.long)
        return x, y


class SpeechCommandsSubset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        classes: list[str],
        variant: str,
        sample_rate: int,
        sequence_length: int,
        mfcc_cfg: DictConfig | None,
        download: bool,
        split_seed: int,
        train_split: float,
        val_split: float,
        mean: torch.Tensor | None = None,
        std: torch.Tensor | None = None,
    ) -> None:
        try:
            import torchaudio
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError("torchaudio is required for Speech Commands training") from exc

        self.torchaudio = torchaudio
        Path(root).mkdir(parents=True, exist_ok=True)
        self.base = torchaudio.datasets.SPEECHCOMMANDS(
            root=str(root),
            download=download,
            subset=None,
        )
        self.classes = list(classes)
        self.class_to_index = {label: i for i, label in enumerate(self.classes)}
        self.variant = variant
        self.sample_rate = int(sample_rate)
        self.sequence_length = int(sequence_length)
        self.mean = mean
        self.std = std
        self.resampler = None
        self.mfcc = None
        if variant == "mfcc":
            if mfcc_cfg is None:
                raise ValueError("MFCC data variant requires data.mfcc config")
            self.mfcc = torchaudio.transforms.MFCC(
                sample_rate=self.sample_rate,
                n_mfcc=int(mfcc_cfg.n_mfcc),
                melkwargs={
                    "n_fft": int(mfcc_cfg.n_fft),
                    "n_mels": int(mfcc_cfg.n_mels),
                },
            )
        elif variant != "raw":
            raise ValueError(f"unsupported data variant: {variant!r}")

        buckets: dict[str, list[int]] = defaultdict(list)
        for i, item in enumerate(self.base._walker):  # noqa: SLF001 - torchaudio public pattern
            label = Path(item).parent.name
            if label in self.class_to_index:
                buckets[label].append(i)
        rng = random.Random(split_seed)
        selected: list[int] = []
        for label in self.classes:
            indices = buckets[label]
            rng.shuffle(indices)
            n_train = int(len(indices) * train_split)
            n_val = int(len(indices) * val_split)
            if split == "train":
                selected.extend(indices[:n_train])
            elif split == "val":
                selected.extend(indices[n_train : n_train + n_val])
            elif split == "test":
                selected.extend(indices[n_train + n_val :])
            else:
                raise ValueError(f"unsupported split: {split!r}")
        self.indices = selected

    def set_normalizer(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.mean = mean
        self.std = std

    def __len__(self) -> int:
        return len(self.indices)

    def _metadata(self, dataset_index: int) -> tuple[Path, str]:
        item = Path(self.base._walker[dataset_index])  # noqa: SLF001 - torchaudio public pattern
        if not item.is_absolute():
            item = Path(self.base._archive) / item  # noqa: SLF001 - torchaudio public pattern
        return item, item.parent.name

    @staticmethod
    def _load_waveform(path: Path) -> tuple[torch.Tensor, int]:
        with wave.open(str(path), "rb") as wav:
            sample_rate = wav.getframerate()
            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            frames = bytearray(wav.readframes(wav.getnframes()))

        if sample_width == 1:
            data = torch.frombuffer(frames, dtype=torch.uint8).clone().float()
            data = (data - 128.0) / 128.0
        elif sample_width == 2:
            data = torch.frombuffer(frames, dtype=torch.int16).clone().float()
            data = data / 32768.0
        elif sample_width == 4:
            data = torch.frombuffer(frames, dtype=torch.int32).clone().float()
            data = data / 2147483648.0
        else:
            raise ValueError(f"unsupported WAV sample width {sample_width} bytes for {path}")
        waveform = data.reshape(-1, channels).transpose(0, 1).contiguous()
        return waveform, sample_rate

    def _prepare_waveform(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != self.sample_rate:
            if self.resampler is None:
                self.resampler = self.torchaudio.transforms.Resample(sample_rate, self.sample_rate)
            waveform = self.resampler(waveform)
        target = self.sample_rate
        if waveform.size(-1) < target:
            waveform = torch.nn.functional.pad(waveform, (0, target - waveform.size(-1)))
        return waveform[..., :target]

    def _features(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        waveform = self._prepare_waveform(waveform, sample_rate)
        if self.variant == "raw":
            x = waveform.squeeze(0).unsqueeze(-1)
        else:
            assert self.mfcc is not None
            x = self.mfcc(waveform).squeeze(0).transpose(0, 1)
            if x.size(0) < self.sequence_length:
                x = torch.nn.functional.pad(x, (0, 0, 0, self.sequence_length - x.size(0)))
            x = x[: self.sequence_length]
        if self.mean is not None and self.std is not None:
            x = (x - self.mean) / self.std.clamp_min(1e-6)
        return x.float()

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, label = self._metadata(self.indices[index])
        waveform, sample_rate = self._load_waveform(path)
        return self._features(waveform, sample_rate), torch.tensor(self.class_to_index[label])


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def as_container(obj: Any) -> Any:
    return OmegaConf.to_container(obj, resolve=True)


def get_model_cfg(cfg: DictConfig, key: str, default: Any = None) -> Any:
    return cfg.model[key] if key in cfg.model else default


def variant_encoder_kwargs(cfg: DictConfig) -> dict[str, Any]:
    kwargs = dict(as_container(cfg.model.encoder_kwargs))
    if cfg.data.variant == "mfcc" and "mfcc_encoder_kwargs" in cfg.model:
        kwargs.update(as_container(cfg.model.mfcc_encoder_kwargs))
    return kwargs


def variant_filter_scales(cfg: DictConfig) -> list[dict[str, Any]] | None:
    if cfg.data.variant == "mfcc" and "mfcc_filter_scales" in cfg.model:
        return as_container(cfg.model.mfcc_filter_scales)
    if "filter_scales" in cfg.model:
        return as_container(cfg.model.filter_scales)
    return None


def validate_cfg(cfg: DictConfig) -> None:
    if cfg.training.per_step_loss and cfg.training.scan_grad_mode != "tbptt1":
        raise ValueError("per_step_loss=true requires training.scan_grad_mode=tbptt1")


def build_encoder(cfg: DictConfig) -> tuple[nn.Module, int]:
    kwargs = variant_encoder_kwargs(cfg)
    input_dims = int(cfg.data.feature_dim)
    output_dims = int(cfg.model.output_dims)
    hidden_width = int(cfg.model.hidden_width)
    depth = int(cfg.model.depth)
    skip_connection = bool(cfg.model.skip_connection)

    if cfg.model.encoder == "pure_rqf":
        encoder = PureRQFEncoder(
            input_dims=input_dims,
            output_dims=output_dims,
            hidden_dims=hidden_width,
            num_layers=depth,
            filter_scales=variant_filter_scales(cfg),
            skip_connection=skip_connection,
            prospective_input=bool(cfg.model.prospective_input),
            prospective_mode=str(cfg.model.prospective_mode),
            prospective_tau=get_model_cfg(cfg, "prospective_tau"),
            scan_grad_mode=str(cfg.training.scan_grad_mode),
            **kwargs,
        )
    elif cfg.model.encoder == "rqf_s4":
        encoder = S4RQFEncoder(
            input_dims=input_dims,
            output_dims=output_dims,
            hidden_dims=hidden_width,
            num_layers=depth,
            filter_scales=variant_filter_scales(cfg),
            skip_connection=skip_connection,
            prospective_input=bool(cfg.model.prospective_input),
            prospective_mode=str(cfg.model.prospective_mode),
            prospective_tau=get_model_cfg(cfg, "prospective_tau"),
            scan_grad_mode=str(cfg.training.scan_grad_mode),
            tbptt_delayed_input_mode=str(
                getattr(cfg.training, "tbptt_delayed_input_mode", "strict")
            ),
            **kwargs,
        )
    else:
        raise ValueError(f"unsupported model.encoder={cfg.model.encoder!r}")
    return encoder, output_dims


def build_organics_model(cfg: DictConfig) -> OrganicsSequenceClassifier:
    hidden_width = int(cfg.model.hidden_width)
    cells = [
        ORGaNICsCell(
            input_size=int(cfg.data.feature_dim) if layer == 0 else hidden_width,
            hidden_size=hidden_width,
            organics_activation=str(cfg.model.cell.organics_activation),
            Wr_constraint=str(cfg.model.cell.Wr_constraint),
            Way_constraint=str(cfg.model.cell.Way_constraint),
            learn_tau=bool(cfg.model.cell.learn_tau),
            dt_tau_max_y=float(cfg.model.cell.dt_tau_max_y),
            dt_tau_max_a=float(cfg.model.cell.dt_tau_max_a),
            dt_tau_max_b=float(cfg.model.cell.dt_tau_max_b),
            sigma=float(cfg.model.cell.sigma),
            n=float(cfg.model.cell.n),
            prospective_input=bool(cfg.model.prospective_input),
        )
        for layer in range(int(cfg.model.depth))
    ]
    return OrganicsSequenceClassifier(
        cells=cells,
        hidden_width=hidden_width,
        num_classes=len(cfg.data.classes),
        use_layer_norm=bool(cfg.model.network.use_layer_norm),
        skip_connection=bool(cfg.model.skip_connection),
        recurrent_grad_mode=str(cfg.training.scan_grad_mode),
        pooling=str(cfg.training.pooling),
        per_step_loss=bool(cfg.training.per_step_loss),
        per_step_readout=str(cfg.model.network.per_step_readout),
    )


def build_model(cfg: DictConfig) -> nn.Module:
    if cfg.model.encoder == "discrete_organics":
        return build_organics_model(cfg)
    encoder, output_dims = build_encoder(cfg)
    return SequenceClassifier(
        encoder=encoder,
        encoder_dim=output_dims,
        num_classes=len(cfg.data.classes),
        pooling=str(cfg.training.pooling),
        hidden_dims=get_model_cfg(cfg, "classifier").hidden_dims,
        activation=str(cfg.model.classifier.activation),
        dropout=float(cfg.model.classifier.dropout),
        per_step_loss=bool(cfg.training.per_step_loss),
    )


def build_optimizer(cfg: DictConfig, model: nn.Module) -> torch.optim.Optimizer:
    rqf_patterns = ("freq_param", "gamma_param", "log_tau", "rqf_fwd.", "rqf_bwd.")
    base_params = []
    rqf_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(pattern in name for pattern in rqf_patterns):
            rqf_params.append(param)
        else:
            base_params.append(param)
    groups = [
        {
            "params": base_params,
            "lr": float(cfg.training.learning_rate),
            "weight_decay": float(cfg.training.weight_decay),
        }
    ]
    if rqf_params:
        groups.append(
            {
                "params": rqf_params,
                "lr": float(cfg.training.learning_rate) * float(cfg.training.rqf_lr_factor),
                "weight_decay": 0.0,
            }
        )
    return torch.optim.AdamW(groups)


def per_step_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    label_smoothing: float,
) -> torch.Tensor:
    batch, time, classes = logits.shape
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(batch * time, classes),
        target[:, None].expand(batch, time).reshape(-1),
        label_smoothing=label_smoothing,
    )
    return loss


def compute_stats(dataset: Dataset, max_samples: int | None) -> tuple[torch.Tensor, torch.Tensor]:
    total = None
    total_sq = None
    count = 0
    limit = len(dataset) if max_samples is None else min(len(dataset), int(max_samples))
    for i in tqdm(range(limit), desc="feature stats", leave=False):
        x, _ = dataset[i]
        if total is None:
            total = torch.zeros(x.size(-1), dtype=torch.float64)
            total_sq = torch.zeros(x.size(-1), dtype=torch.float64)
        x64 = x.double()
        total += x64.sum(dim=0)
        total_sq += (x64 * x64).sum(dim=0)
        count += x.size(0)
    assert total is not None and total_sq is not None
    mean = total / count
    var = (total_sq / count) - mean * mean
    return mean.float(), var.clamp_min(1e-12).sqrt().float()


def build_datasets(cfg: DictConfig) -> tuple[Dataset, Dataset, Dataset]:
    if cfg.data.use_synthetic:
        return (
            SyntheticSpeechCommands(
                cfg.data.synthetic_train_size,
                cfg.data.sequence_length,
                cfg.data.feature_dim,
                len(cfg.data.classes),
            ),
            SyntheticSpeechCommands(
                cfg.data.synthetic_val_size,
                cfg.data.sequence_length,
                cfg.data.feature_dim,
                len(cfg.data.classes),
            ),
            SyntheticSpeechCommands(
                cfg.data.synthetic_test_size,
                cfg.data.sequence_length,
                cfg.data.feature_dim,
                len(cfg.data.classes),
            ),
        )

    common = {
        "root": cfg.data.root,
        "classes": list(cfg.data.classes),
        "variant": str(cfg.data.variant),
        "sample_rate": int(cfg.data.sample_rate),
        "sequence_length": int(cfg.data.sequence_length),
        "mfcc_cfg": cfg.data.get("mfcc"),
        "download": bool(cfg.data.download),
        "split_seed": int(cfg.data.split_seed),
        "train_split": float(cfg.data.train_split),
        "val_split": float(cfg.data.val_split),
    }
    train = SpeechCommandsSubset(split="train", **common)
    val = SpeechCommandsSubset(split="val", **common)
    test = SpeechCommandsSubset(split="test", **common)
    if cfg.data.normalize and cfg.data.compute_train_stats:
        mean, std = compute_stats(train, cfg.data.stats_max_samples)
        for dataset in (train, val, test):
            dataset.set_normalizer(mean, std)
    return train, val, test


def make_loader(dataset: Dataset, cfg: DictConfig, split: str) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(cfg.data.batch_size),
        shuffle=(split == "train"),
        num_workers=int(cfg.num_workers),
        pin_memory=torch.cuda.is_available(),
        drop_last=bool(cfg.data.drop_last_train if split == "train" else cfg.data.drop_last_eval),
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    cfg: DictConfig,
) -> dict[str, float]:
    train = optimizer is not None
    model.train(train)
    loss_sum = 0.0
    correct = 0
    seen = 0
    for x, y in tqdm(loader, desc="train" if train else "eval", leave=False):
        x = x.to(device)
        y = y.to(device)
        with torch.set_grad_enabled(train):
            logits = model(x)
            if train and cfg.training.per_step_loss:
                loss = per_step_cross_entropy(logits, y, float(cfg.training.label_smoothing))
                pred = logits.mean(dim=1).argmax(dim=-1)
            else:
                loss = torch.nn.functional.cross_entropy(
                    logits,
                    y,
                    label_smoothing=float(cfg.training.label_smoothing) if train else 0.0,
                )
                pred = logits.argmax(dim=-1)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.training.gradient_clip_val))
                optimizer.step()
        batch = y.numel()
        loss_sum += float(loss.detach()) * batch
        correct += int((pred == y).sum().item())
        seen += batch
    return {"loss": loss_sum / max(seen, 1), "acc": correct / max(seen, 1)}


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    validate_cfg(cfg)
    seed_everything(int(cfg.seed))

    device = torch.device(
        "cuda"
        if str(cfg.device) == "auto" and torch.cuda.is_available()
        else ("cpu" if str(cfg.device) == "auto" else str(cfg.device))
    )
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out_dir / "resolved_config.yaml")

    train_data, val_data, test_data = build_datasets(cfg)
    train_loader = make_loader(train_data, cfg, "train")
    val_loader = make_loader(val_data, cfg, "val")
    test_loader = make_loader(test_data, cfg, "test")

    model = build_model(cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"model={cfg.model.name} data={cfg.data.variant} params={total_params} trainable={trainable_params}")
    if cfg.dry_run:
        return

    optimizer = build_optimizer(cfg, model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(cfg.training.max_epochs),
        eta_min=float(cfg.training.min_lr),
    )

    ckpt_dir = out_dir / cfg.checkpoint_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val_acc = -1.0
    best_epoch = -1
    best_state = None
    stale = 0
    history = []
    for epoch in range(int(cfg.training.max_epochs)):
        train_metrics = run_epoch(model, train_loader, device, optimizer, cfg)
        val_metrics = run_epoch(model, val_loader, device, None, cfg)
        scheduler.step()
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        print(
            f"epoch={epoch:03d} train_acc={train_metrics['acc']:.4f} "
            f"val_acc={val_metrics['acc']:.4f} val_loss={val_metrics['loss']:.4f}"
        )
        torch.save({"model": model.state_dict(), "cfg": as_container(cfg), "epoch": epoch}, ckpt_dir / "last.pt")
        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            torch.save(
                {"model": best_state, "cfg": as_container(cfg), "epoch": epoch, "val_acc": best_val_acc},
                ckpt_dir / "best_val_acc.pt",
            )
            stale = 0
        else:
            stale += 1
            if int(cfg.training.patience) > 0 and stale >= int(cfg.training.patience):
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = run_epoch(model, test_loader, device, None, cfg)
    summary = {
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "test": test_metrics,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "history": history,
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(
        f"best_epoch={best_epoch} best_val_acc={best_val_acc:.4f} "
        f"test_acc={test_metrics['acc']:.4f}"
    )


if __name__ == "__main__":
    main()
