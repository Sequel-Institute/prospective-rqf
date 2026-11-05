# LRA Path-X

The Long Range Arena Path-X task is available through the existing `train.py`
and `evaluate_checkpoint.py` entry points. Each 128×128 grayscale Pathfinder
image is raster-scanned row-major into a sequence of 16,384 steps with one
input feature. The binary label says whether the image's two dots are
connected by a dashed path. The encoder reads the whole sequence; evaluation
averages the per-timestep logits over time.

## Configuration

`+experiment=pathx` selects the following settings:

| Component | Setting |
| --- | --- |
| Encoder | Six layers of 64 complex RQFs, no residual skips, no input or output projection |
| Drive | Prospective ZOH, horizon tau; dt = 1/16384, tau = 5 dt |
| Initialization | Log-uniform theta in [2π·tau/T, 2π·0.25] = [5/16384, π/2] with T = 2π; critical gamma (Q=1); forward-preserving W |
| Activation | Split ReLU between layers; raw final state |
| Readout | Real/imaginary features → He-normal 256-unit ReLU MLP → near-zero-initialized logits, at every timestep |
| Objective | Cross-entropy at every timestep, full BPTT |
| Parameters | 75,394 real trainable parameters |
| Optimizer | Adam, all learning rates 0.003, epsilon 0.0001, gradient clipping 1 |
| Schedule | Cosine annealing to zero over 300 epochs |
| Regularization | No weight decay, label smoothing, or dropout |
| Data | 200,000 examples (`curv_contour_length_14`), seed-42 random 160,000/20,000/20,000 split, scalar z-score with train-split statistics |
| Budget | Batch 64, 2,500 updates/epoch, 300 epochs = 750,000 updates |
| Evaluation | Validation selects the checkpoint; test accuracy is reported for it |

Setting T = 2π (rather than the 1 s sequence) lowers the init frequency floor,
so the slowest channels decay over the whole sequence. The first complex RQF
weight maps the single input feature directly to 64 filters. A six-layer
128-wide variant uses `model.hidden_width=128 model.output_dims=128`. Expect
validation accuracy to stay near chance for many epochs before it rises.

## Data

Extract the official LRA release (`lra_release.gz` from
`https://storage.googleapis.com/long-range-arena/lra_release.gz`) so the
following path exists under `data.root` (default `data/`):

```
data/lra_release/lra_release/pathfinder128/curv_contour_length_14/{imgs,metadata}
```

A bare `data/pathfinder128/...` layout is also accepted. The archive also
contains a truncated top-level `lra_release/pathfinder128` copy; it is not used.
The first build decodes the 200,000 PNGs once (Pillow) into
`pathfinder128/processed_data/curv_contour_length_14/` (a 3.3 GB uint8 cache);
later builds load the cache. Concurrent jobs share a lock, so only one of them
decodes. By default the build requires all 200 metadata shards and 200,000
examples, so a partial extraction fails instead of producing a smaller
benchmark. The splits are held in memory as float32, and training peaks at
about 17 GB of host RAM. The full recipe uses about 10 GiB of GPU memory at
batch size 64 (about 16 GiB for the 128-wide variant).

## Run locally

After installing the project dependencies, run the small CPU check:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python train.py \
  +experiment=pathx_smoke \
  hydra.run.dir=outputs/pathx_smoke
```

This keeps the 6×64 model and optimizer, but reads eight Pathfinder-32
examples per split (T=1024, `pathfinder32` from the same release) and runs two
updates. It checks execution rather than convergence. To check the full
16,384-step length on a GPU with a few hundred examples:

```bash
.venv/bin/python train.py +experiment=pathx device=cuda \
  data.max_samples_per_split=256 training.max_epochs=2 \
  hydra.run.dir=outputs/pathx_cuda_smoke
```

Both commands save `resolved_config.yaml`, `metrics.json`, and last/best-validation
checkpoints. The full Path-X build still loads the complete uint8 cache.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python evaluate_checkpoint.py \
  --checkpoint outputs/pathx_smoke/checkpoints/best_val_acc.pt \
  --device cpu --split test

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m unittest tests.test_pathfinder -v
```

The unit tests use small synthetic Pathfinder images and need no download.
The complete GPU training recipe is:

```bash
.venv/bin/python train.py +experiment=pathx device=cuda
```
