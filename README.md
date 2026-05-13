Patent notice: this repository includes patent-protected methods and is shared for anonymous review of the submitted paper.

# Recursive Quadrature Filter Experiments

This repository contains a compact training entrypoint for the Speech Commands experiments based on Recursive Quadrature Filters (RQFs), prospective input, and local TBPTT(1) loss. It is intentionally minimal: it includes the model code, Hydra configurations, and a direct training script, but not the larger infrastructure used to run the full sweep.

The code includes:

- standalone Pure RQF encoders;
- RQF-S4 encoders;
- S4D and S5 full-BPTT baselines;
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
python train.py data.use_synthetic=true training.max_epochs=1 training.batch_size=4 model=pure_rqf model.depth=2
python train.py data.use_synthetic=true training.max_epochs=1 training.batch_size=4 model=rqf_s4 model.depth=2
python train.py data.use_synthetic=true training.max_epochs=1 training.batch_size=4 model=s4d model.depth=2 training=full_bptt
python train.py data.use_synthetic=true training.max_epochs=1 training.batch_size=4 model=s5 model.depth=2 training=full_bptt
```

## Main RQF Grids

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

## S4D and S5 Baselines

The baseline S4D and S5 configs are exposed for full BPTT only:

```bash
python train.py -m model=s4d,s5 data=speech_commands_raw,speech_commands_mfcc training=full_bptt \
  model.depth=2,4,6 model.skip_connection=false,true seed=0
```

## Outputs

Each run writes:

- `resolved_config.yaml`
- `checkpoints/last.pt`
- `checkpoints/best_val_acc.pt`
- `metrics.json`

Additional checkpoints used for the submitted tables can be added under
`pretrained/` using the same format.

## Checkpoint Evaluation

The repository includes one example checkpoint:

```text
pretrained/pure_rqf_mfcc_tbptt_seed0.pt
```

It corresponds to:

```text
model=pure_rqf
data=speech_commands_mfcc
training=tbptt_local_loss
model.depth=4
model.skip_connection=false
model.prospective_input=false
seed=0
```

Evaluate it on the Speech Commands test split:

```bash
python evaluate_checkpoint.py \
  --checkpoint pretrained/pure_rqf_mfcc_tbptt_seed0.pt \
  --split test \
  --data-root data \
  --output-json outputs/pure_rqf_mfcc_tbptt_seed0_test_metrics.json
```

For a quick loader/model smoke test without downloading Speech Commands:

```bash
python evaluate_checkpoint.py \
  --checkpoint pretrained/pure_rqf_mfcc_tbptt_seed0.pt \
  --synthetic
```
