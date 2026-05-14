Patent notice: this repository includes patent-protected methods and is shared for anonymous review of the submitted paper.

# Recursive Quadrature Filter Experiments

This repository contains the training and evaluation code for the Speech
Commands experiments based on Recursive Quadrature Filters (RQFs), prospective
input, and local TBPTT(1) loss. It includes the model code, Hydra
configurations, packaged checkpoints, and direct training/evaluation scripts.

The code includes:

- standalone Pure RQF encoders;
- RQF-S4 encoders;
- discrete ORGaNICs recurrent baselines;
- Hydra configs for raw waveform and MFCC Speech Commands inputs;
- Hydra configs for full BPTT and TBPTT(1) local per-step loss;
- a simple PyTorch training script.

The paper tables were produced from five seeds. This review repository defaults to `seed=0` and exposes the same model/configuration axes so each reported configuration can be re-run directly.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Smoke Test

This uses synthetic data and only checks that the selected model/config builds and trains for one epoch:

```bash
python train.py data=speech_commands_mfcc data.use_synthetic=true training.max_epochs=1 training.batch_size=4 model=pure_rqf model.depth=2
python train.py data=speech_commands_mfcc data.use_synthetic=true training.max_epochs=1 training.batch_size=4 model=rqf_s4 model.depth=2
python train.py data=speech_commands_mfcc data.use_synthetic=true training.max_epochs=1 training.batch_size=4 model=discrete_organics model.depth=2
```

## Main Training Grids

Pure RQF, raw waveform, full BPTT:

```bash
python train.py -m model=pure_rqf data=speech_commands_raw training=full_bptt \
  model.depth=2,4,6 model.prospective_input=false,true model.skip_connection=false,true seed=0
```

Pure RQF, raw waveform, TBPTT(1) with local per-step loss:

```bash
python train.py -m model=pure_rqf data=speech_commands_raw training=tbptt_local_loss \
  model.depth=2,4,6 model.prospective_input=false,true model.skip_connection=false,true seed=0
```

Pure RQF, MFCC, full BPTT:

```bash
python train.py -m model=pure_rqf data=speech_commands_mfcc training=full_bptt \
  model.depth=2,4,6 model.prospective_input=false,true model.skip_connection=false,true seed=0
```

Pure RQF, MFCC, TBPTT(1) with local per-step loss:

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

Packaged checkpoints are stored under `pretrained/`. Metadata, source run IDs,
source validation/test metrics, and Hydra overrides are listed in:

```text
pretrained/checkpoint_manifest.json
```

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
