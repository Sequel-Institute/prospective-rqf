# Selective copying

The [local validation record](selective_copying_local_validation.md) includes
the CPU training, learning, checkpoint, and regression-test results.

The task is available through the existing `train.py` and
`evaluate_checkpoint.py` entry points. No downloads or local-learning installation
are needed. Inputs contain randomly spaced data symbols followed by recall
markers. The encoder processes the entire sequence; the readout, cross-entropy,
and accuracy use only the recall suffix, during both training and evaluation.

## Recipe provenance

`+experiment=selective_copying` ports local-learning's
`configs/selective_copying/2026-09-14/factorial_400epochs_lr1e-3/rqf_bptt_4x64_p_write_split_relu_400epochs_lr1e-3.json`,
config hash `899afaa5a82768b9`, submitted at commit
`b09e95ee31f58fdd112285f940df31259683932f`.
Its saved 400-epoch metric history records **100% validation recall-token
accuracy at epoch 308**, with **99.981689453125% test accuracy** at that epoch.
This is the guide's recommended recipe and the highest test-at-best-validation
result among the completed September 14 factorial runs. These are historical
local-learning results, not a fresh full-budget result from this repository.

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
| Budget | Batch 64, 64,000 fresh examples/epoch, 400 epochs = 400,000 updates |
| Evaluation | Fixed validation/test sets, 1024 sequences each; validation selects the checkpoint |

The first complex RQF weight maps the 16 one-hot inputs directly to 64 filters.
The optional `input_projection=false` and `output_projection=false` settings
avoid additional or unused projections. Existing encoder configurations retain
their projection defaults. Parameter initialization uses this repository's RNG
draw ordering; seed 0 does not imply bitwise-identical initialization to
local-learning. This trainer also uses its existing eager execution rather than
local-learning's CUDA-graph training loop.

The generator preserves local-learning's seed/split/epoch/index sample keys.
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

CPU runs exercise the PyTorch selective recurrence. Native CUDA/Triton execution
was also checked on the local A10G outside the device-restricted sandbox; see
the validation record. Full-budget convergence remains untested in this port.
