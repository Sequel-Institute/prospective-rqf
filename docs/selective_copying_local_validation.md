# Local selective copying validation — 2026-09-17

Initial checks ran on CPU with Python 3.12 and the project's pinned PyTorch
2.7.1 (`2.7.1+cpu`). The execution sandbox hid the NVIDIA device nodes; the
initial conclusion that the host lacked CUDA was incorrect. A check outside
the sandbox confirmed an NVIDIA A10G (23,028 MiB), driver 570.172.08, and CUDA
12.8 support. The initial project environment had a CPU-only PyTorch wheel.
All runs used seed 0 and the ported 4×64 prospective recipe with gamma,
write, and four-group frequency selection: 119,184 trainable real parameters.
No full-budget training was performed.

| Check | Budget | Result |
| --- | --- | --- |
| Full-length training smoke | Length 4096, batch 1, two updates; one validation and one test sequence | Forward, backward, updates, evaluation, and checkpoint saving completed; finite train/val/test losses 2.767445 / 2.759101 / 2.773603 |
| Small task smoke | Length 128, batch 4, two epochs, eight updates | Completed; validation loss fell from 2.763564 to 2.747288 |
| Checkpoint evaluation | Reload the small smoke's validation-selected checkpoint with `evaluate_checkpoint.py` | Exactly reproduced saved test loss 2.7653476 and recall-token accuracy 7.03125% |
| Fixed-minibatch learning probe | Four length-128 training sequences, batch 4, 300 updates, unchanged model/optimizer | Loss 2.772325 → 0.089467; final training recall-token accuracy 96.875%; all trainable gradients finite |
| Source-checkpoint compatibility | Map original epoch-308 tensors into the new model; 16 fixed validation sequences at length 4096 | 256/256 recall tokens correct; loss 0.0000025084 |
| Unit/regression suite | `python -m unittest discover -s tests -v` | 51 tests: 33 passed, 18 hardware-dependent skips |

The source-checkpoint check used the existing trained model, not a new training
run. It mapped all 32 source state entries (weights, controllers, frequency-group
buffers, and readout tensors) into the matching architecture for a local
diagnostic. A general checkpoint conversion interface is not provided.
The fixed-minibatch check measures memorization of four training examples, not
held-out performance. Short smoke accuracies remain near chance for these tiny
update budgets.

The six new task tests cover token placement and ordered recall, ignored context,
training resampling and fixed evaluation splits, invalid task sizes, parameter
count and optimizer settings, partial-batch metric weighting, gradients through
every trainable component, and strict checkpoint round trips. Existing selective
reference tests and float64 gradchecks also passed. CUDA/Triton and MPS checks
were skipped in that initial sandboxed CPU run.

## CUDA follow-up

The local A10G was then tested outside the device-restricted sandbox, using
PyTorch `2.7.1+cu128` and Triton `3.3.1`. The first native run exposed a compiler
error in the existing selective kernel: passing a tuple as a `tl.constexpr`
argument and indexing it inside a JIT helper raised
`TypeError("'constexpr' object is not subscriptable")`. The kernel now receives
the eight controller options as individual compile-time scalars. The recurrence
and gradient equations are unchanged.

The GPU smoke completed at the original **sequence length 4096 and batch size
64**, with two optimizer updates, 64 validation sequences, and 64 test sequences.
Train/validation/test losses were 2.769756 / 2.760402 / 2.760755, all finite.
Checkpoint reload reproduced the saved test metrics. This validates native GPU
execution of the recipe, not full-budget convergence.

The complete regression suite then passed on the GPU-enabled host: **51 tests,
47 passed, 4 MPS-only skips**. Native CUDA checks include fused/reference output
and gradient comparisons, all controller variants, full BPTT and TBPTT(1),
chunk boundaries, numerical edge cases, and optimizer updates. The CUDA tests
explicitly reject fallback to the PyTorch reference.

```bash
TRITON_CACHE_DIR=/tmp/prospective-rqf-triton-cache \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python train.py \
  +experiment=selective_copying device=cuda training.max_epochs=1 \
  data.n_train=128 data.n_val=64 data.n_test=64 \
  hydra.run.dir=outputs/selective_copying_cuda_smoke
```

The project virtual environment was changed to CUDA-enabled PyTorch at the same
pinned version. CUDA commands must run in a context with access to the GPU
device nodes; testing only inside the restricted sandbox gives a misleading
picture of host GPU availability.

Before selecting the recipe, the source run's complete finite 400-epoch history
was read. Its last and resume checkpoints both reported epoch 400 and config
hash `899afaa5a82768b9`, with identical finite model tensors and optimizer step
counters of 400,000. The archived terminal status was `SUCCEEDED`. The earliest
maximum-validation epoch was 308, with test recall-token accuracy
99.981689453125%.

Run commands are in [selective_copying.md](selective_copying.md). Local artifacts
are retained under `outputs/` (ignored by Git):

- `selective_copying_full_length_smoke/`: config, metrics, checkpoints, training log.
- `selective_copying_smoke/`: config, metrics, checkpoints, log, reloaded test metrics.
- `selective_copying_learning_probe/`: metrics, config, final checkpoint.
- `selective_copying_source_check/metrics.json`: source-checkpoint diagnostic.
- `selective_copying_validation/`: test log and combined validation record.
- `selective_copying_cuda_smoke/`: GPU config, metrics, checkpoints, and reloaded test metrics.
