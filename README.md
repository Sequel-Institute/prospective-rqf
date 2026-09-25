Patent notice: this repository includes methods that are patent pending. Unless otherwise noted, it is shared for anonymous review of the submitted paper under the CC BY-NC-SA 4.0 License; see `LICENSE`.

# Recursive Quadrature Filter Experiments

This repository contains the training and evaluation code for the Speech
Commands experiments based on Recursive Quadrature Filters (RQFs),
prospective-input coding, and spatial-only backpropagation. It includes the
model code, Hydra configurations, packaged checkpoints, and direct
training/evaluation scripts.

The code includes:

- standalone Pure RQF encoders;
- RQF-S4 encoders;
- S5 baseline encoders;
- discrete ORGaNICs recurrent baselines;
- Hydra configs for raw waveform and MFCC Speech Commands inputs;
- Hydra configs for full BPTT and spatial-only backpropagation;
- [selective copying](docs/selective_copying.md) with a selective RQF training configuration and a CPU smoke configuration;
- a simple PyTorch training script.

The paper tables were produced from five seeds. This review repository defaults to `seed=0` and exposes the same model/configuration axes so each reported configuration can be re-run directly.

## Configuration Terms

`Recursive Quadrature Filter (RQF)` denotes the complex-valued temporal filter
used here. Each scalar RQF is equivalent to a single-channel diagonal state-space
model, with learnable phase-prior frequency and bandwidth parameters.

`model.prospective_input=false` is the instantaneous-input baseline: the
current bottom-up layer input is passed directly to the next recurrent layer.
`model.prospective_input=true` is the prospective-input condition: the same
bottom-up input is advanced with the look-ahead operator
`chi_tau = 1 + tau d/dt`. In code this is implemented as the derived two-tap
input path.

S5 exposes two exact prospective definitions with the shared physical horizon
`tau = c * input_dt` and `c=5` by default. Native learned S5 steps convert this
to the modal impulse coefficient `c * delta[p]`:

- `model=s5` selects **P-S5**, the default `prospective_mode=direct`, which
  applies the prospective operator to the ordinary learned S5 input matrix.
  Set `model.prospective_input=false` for the non-prospective S5 control.
- `model=alpha_p_s5` selects **alphaP-S5**, the named
  `prospective_mode=decay_gain`, which uses
  `diag(-Re(Lambda_eff)) @ W_in` as the continuous-time input matrix.

Both definitions leave the direct `D * u` feedthrough instantaneous and use
the fixed `prospective_initial` boundary convention (`zero` by default).

`training=full_bptt` is standard full backpropagation through time, with
gradients propagated through the full recurrent trajectory. `training=tbptt_local_loss`
implements spatial-only backpropagation: the temporal recurrence is detached at
each step (TBPTT(1)), while the within-step spatial chain through depth remains
live and receives a local per-step loss.

The raw-waveform input is one second of audio at 16 kHz (`T=16000`, one scalar
sample per step). The MFCC input is a lower-rate cepstral representation
(`T=161`, 20 coefficients per step).

## Setup

On Ubuntu, install the standard venv package if needed:

```bash
sudo apt-get install python3.10-venv
```

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

## Smoke Test

This uses synthetic data and only checks that the selected model/config builds and trains for one epoch:

```bash
python train.py data=speech_commands_mfcc data.use_synthetic=true training.max_epochs=1 training.batch_size=4 model=pure_rqf model.depth=2
python train.py data=speech_commands_mfcc data.use_synthetic=true training.max_epochs=1 training.batch_size=4 model=rqf_s4 model.depth=2
python train.py data=speech_commands_mfcc data.use_synthetic=true training.max_epochs=1 training.batch_size=4 model=discrete_organics model.depth=2
```

## Main Training Grids

### Input-dependent Pure RQF

`model=pure_rqf_selective` extends the paper recipe with input-dependent damping,
an optional bounded sigmoid write gate, and optional global/grouped frequency warps. It
supports full BPTT and spatial-only TBPTT(1), with split ReLU, ZOH, log-space
parameters, and fixed tau. The default enables damping selection only.

```bash
python train.py model=pure_rqf_selective data=speech_commands_mfcc device=mps \
  model.selective.write_control=true model.selective.frequency_warp=grouped
python train.py model=pure_rqf_selective training=tbptt_local_loss device=cuda
```

CPU and MPS run the PyTorch reference; CUDA uses fused Triton kernels when
available. Set `ORGANICS_RQF_FBT=0` to force the reference. See
[selective RQF behavior and validation](docs/selective_rqf.md) for all options,
gradient conventions, and CPU/MPS/CUDA test commands.

Pure RQF, raw waveform, full BPTT:

```bash
python train.py -m model=pure_rqf data=speech_commands_raw training=full_bptt \
  model.depth=2,4,6 model.prospective_input=false,true model.skip_connection=false,true seed=0
```

Pure RQF, raw waveform, spatial-only backpropagation / TBPTT(1):

```bash
python train.py -m model=pure_rqf data=speech_commands_raw training=tbptt_local_loss \
  model.depth=2,4,6 model.prospective_input=false,true model.skip_connection=false,true seed=0
```

Pure RQF, MFCC, full BPTT:

```bash
python train.py -m model=pure_rqf data=speech_commands_mfcc training=full_bptt \
  model.depth=2,4,6 model.prospective_input=false,true model.skip_connection=false,true seed=0
```

Pure RQF, MFCC, spatial-only backpropagation / TBPTT(1):

```bash
python train.py -m model=pure_rqf data=speech_commands_mfcc training=tbptt_local_loss \
  model.depth=2,4,6 model.prospective_input=false,true model.skip_connection=false,true seed=0
```

RQF-S4 uses the same axes:

```bash
python train.py -m model=rqf_s4 data=speech_commands_raw training=full_bptt \
  model.depth=2,4,6 model.prospective_input=false,true model.skip_connection=false,true seed=0

python train.py -m model=rqf_s4 data=speech_commands_raw training=tbptt_local_loss \
  model.depth=2,4,6 model.prospective_input=false,true model.skip_connection=false,true seed=0

python train.py -m model=rqf_s4 data=speech_commands_mfcc training=full_bptt \
  model.depth=2,4,6 model.prospective_input=false,true model.skip_connection=false,true seed=0

python train.py -m model=rqf_s4 data=speech_commands_mfcc training=tbptt_local_loss \
  model.depth=2,4,6 model.prospective_input=false,true model.skip_connection=false,true seed=0
```

Discrete ORGaNICs baselines use MFCC inputs:

```bash
python train.py -m model=discrete_organics data=speech_commands_mfcc training=full_bptt \
  model.depth=2,4,6 model.prospective_input=false,true model.skip_connection=false,true \
  training.batch_size=512 seed=0

python train.py -m model=discrete_organics data=speech_commands_mfcc training=tbptt_local_loss \
  model.depth=2,4,6 model.prospective_input=false,true model.skip_connection=false,true \
  training.batch_size=512 seed=0
```

## Outputs

Each run writes:

- `resolved_config.yaml`
- `checkpoints/last.pt`
- `checkpoints/best_val_acc.pt`
- `metrics.json`

Packaged checkpoints are stored under `pretrained/`. Each checkpoint stores the
Hydra overrides needed to reconstruct its model configuration.

## Checkpoint Evaluation

The repository includes seed-0 raw-waveform and MFCC checkpoints for Pure RQF and RQF-S4,
plus seed-0 MFCC/full-BPTT checkpoints for discrete ORGaNICs.
Model sizes are:

```text
S = 2 recurrent layers
M = 4 recurrent layers
L = 6 recurrent layers
```

Coverage:

```text
Pure RQF raw full BPTT:   S/M/L, prospective and non-prospective, residual and no-residual
Pure RQF raw TBPTT(1):    S/M/L, prospective and non-prospective, no-residual
Pure RQF MFCC full BPTT:  S/M/L, prospective and non-prospective, residual and no-residual
Pure RQF MFCC TBPTT(1):   S/M/L, prospective and non-prospective, no-residual
RQF-S4 raw full BPTT:     S/M/L, prospective and non-prospective, residual and no-residual
RQF-S4 raw TBPTT(1):      available M no-residual runs
RQF-S4 MFCC full BPTT:    S/M/L residual, plus available M no-residual runs
RQF-S4 MFCC TBPTT(1):     available M no-residual runs
Discrete ORGaNICs MFCC:   S/M/L full BPTT, prospective and non-prospective, no-residual/no-layer-norm
```

The checkpoint filenames encode model, input representation, training mode,
size, residual flag, prospective flag, and seed:

```text
pretrained/pure_rqf_mfcc_full_M_nores_p_seed0.pt
pretrained/pure_rqf_raw_full_L_res_p_seed0.pt
pretrained/pure_rqf_raw_tbptt_M_nores_p_seed0.pt
pretrained/pure_rqf_mfcc_tbptt_S_nores_np_seed0.pt
pretrained/rqf_s4_raw_full_M_nores_np_seed0.pt
pretrained/rqf_s4_raw_tbptt_M_nores_p_seed0.pt
pretrained/rqf_s4_mfcc_full_L_res_p_seed0.pt
pretrained/rqf_s4_mfcc_tbptt_M_nores_p_seed0.pt
pretrained/discrete_organics_mfcc_full_M_nores_p_seed0.pt
```

Each checkpoint stores its own Hydra overrides. Evaluate one checkpoint on the
Speech Commands test split:

```bash
python evaluate_checkpoint.py \
  --checkpoint pretrained/rqf_s4_mfcc_full_M_res_p_seed0.pt \
  --split test \
  --data-root data \
  --output-json outputs/rqf_s4_mfcc_full_M_res_p_seed0_test_metrics.json
```

Evaluate every packaged checkpoint:

```bash
python evaluate_pretrained.py \
  --split test \
  --data-root data \
  --batch-size 512
```

For a quick loader/model smoke test without downloading Speech Commands:

```bash
python evaluate_checkpoint.py \
  --checkpoint pretrained/rqf_s4_mfcc_full_M_res_p_seed0.pt \
  --synthetic
```
