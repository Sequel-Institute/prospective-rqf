# Selective copying

The task is available through the existing `train.py` and
`evaluate_checkpoint.py` entry points. Data is generated without downloads.
Inputs contain randomly spaced data symbols followed by recall
markers. The encoder processes the entire sequence; the readout, cross-entropy,
and accuracy use only the recall suffix, during both training and evaluation.

## Configuration

`+experiment=selective_copying` selects the following settings:

| Component | Setting |
| --- | --- |
| Encoder | Four layers of 64 complex RQFs, no residual skips |
| Drive | Prospective ZOH, horizon tau; dt=1, tau=5 |
| Initialization | Log-uniform theta, critical gamma (Q=1), forward-preserving W |
| Frequency initialization duration | 4096, retained even in shorter smoke runs |
| Selectors | Raw previous-layer real/imaginary output |
| Controls | Sigmoid gamma, signed write, four frequency groups with multipliers [1/4, 4] |
| Activation | Split ReLU between layers; raw final state |
| Readout | Real/imaginary features → He-normal 256-unit ReLU MLP → logits |
| Parameters | 119,184 real trainable parameters |
| Optimizer | Adam, all learning rates 0.001, epsilon 0.0001, gradient clipping 1 |
| Regularization | No weight decay, scheduler, label smoothing, or dropout |
| Task | Total length 4096, 16 recall tokens, vocabulary 16 |
| Budget | Batch 64, 64,000 fresh examples/epoch, 100 epochs = 100,000 updates |
| Evaluation | Fixed validation/test sets, 1024 sequences each; validation selects the checkpoint |

The first complex RQF weight maps the 16 one-hot inputs directly to 64 filters.
The optional `input_projection=false` and `output_projection=false` settings
avoid additional or unused projections. Existing encoder configurations retain
their projection defaults. Training uses eager execution with seed 0 by default.

The generator keys each sample by seed, split, epoch, and sample index.
Token 0 is blank, tokens 1–14 carry data, and token 15 is the recall marker.
Sixteen distinct positions in the first 4080 timesteps hold data, sampled with
replacement from tokens 1–14. The final 16 inputs are recall markers; the targets
repeat the data in input order. Context targets are `-100`. Training samples
change each epoch; validation and test remain fixed and use separate keys.

## Run locally

After installing the project dependencies, run the small CPU check:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python train.py \
  +experiment=selective_copying_smoke \
  hydra.run.dir=outputs/selective_copying_smoke
```

This keeps the 4×64 model, controllers, and optimizer, with sequence length 128,
batch 4, and eight optimizer updates. It checks execution rather than convergence.
To check the full length of 4096 with a small CPU batch:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python train.py \
  +experiment=selective_copying device=cpu \
  training.max_epochs=1 training.batch_size=1 \
  data.n_train=2 data.n_val=1 data.n_test=1 \
  hydra.run.dir=outputs/selective_copying_full_length_smoke
```

Both commands save `resolved_config.yaml`, `metrics.json`, and last/best-validation
checkpoints. Accuracy is the fraction of correctly recalled **tokens**, not
exact-sequence accuracy. Smoke accuracy is not a convergence result. Metrics
and checkpoints include optimizer-step counts; this trainer's epoch numbers
are zero-based.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python evaluate_checkpoint.py \
  --checkpoint outputs/selective_copying_smoke/checkpoints/best_val_acc.pt \
  --device cpu --split test

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests -v
```

The complete GPU training recipe is:

```bash
.venv/bin/python train.py +experiment=selective_copying device=cuda
```

CPU runs use the PyTorch selective recurrence; CUDA runs use the fused Triton
implementation. See [selective RQF](selective_rqf.md#validation-commands) for
backend validation commands.
