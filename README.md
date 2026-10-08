# Multitask-learning experiments

Run all commands from the repository root. Use Python 3.12. Native models run on CPU; MTT requires a supported GPU.

## Install

```bash
python3.12 -m venv .venv
```

Install PyTorch and torchvision for your hardware using the [official installation instructions](https://pytorch.org/get-started/locally/), with `.venv/bin/python -m pip`. Then install the remaining dependencies:

```bash
.venv/bin/python -m pip install -r requirements.txt
```

## Install data

The data download link is not available yet. Dataset labels and prepared-input paths are listed in `review/datasets.json`. For raw preprocessing, place the complete original sources in the corresponding dataset directory.

## ASSISTments: preprocess and train

With the original sources installed, this generates five folds and the native/MTT training inputs:

```bash
.venv/bin/python review/pipeline.py preprocess --label assistments2009_sb --work-root artifacts/assistments2009-sb-prepared --workers 1
.venv/bin/python review/pipeline.py train --label assistments2009_sb --work-root artifacts/assistments2009-sb-prepared --models all --output-root artifacts/assistments2009-sb-results
```

Use `--models local` for a CPU-only example, or a subset such as `--models local mtt`. `--models all` runs every registered model for that dataset.

## Other datasets: preprocess, prepare and train

Skip `preprocess` if the canonical input is already installed:

```bash
.venv/bin/python review/pipeline.py preprocess --label whas1_preprocessed
.venv/bin/python review/pipeline.py prepare --label whas1_preprocessed --work-root artifacts/whas1-prepared
.venv/bin/python review/pipeline.py train --label whas1_preprocessed --work-root artifacts/whas1-prepared --models all --output-root artifacts/whas1-results
```

Hydraulic inputs must be supplied as canonical CSVs; start with `prepare`. `prepare` automatically reuses the label's registered input bundle. If that ASSISTments or NedBox bundle is missing, it reconstructs the official five folds from the registered raw sources.

HIGGS and SPR cohort reconstruction requires the original source. The pipeline always uses the included official selection map:

```bash
.venv/bin/python review/pipeline.py preprocess --label higgs_50k --input datasets/higgs/train_val_test.h5
.venv/bin/python review/pipeline.py preprocess --label spr_xray_manifest --input 'datasets/spr xray'
```

Then run `prepare` and `train` with the same label. SPR uses existing embeddings, not raw images.

## Outputs and retry

Under the chosen `--output-root`, consolidated outputs are in `experiment_outputs/<label>/`: `metrics.csv`, native `predictions/` and metric tables in `results/`. MTT logs are in `mtt_logs/`.

Use new work/output directories. Repeating a completed training command verifies and reuses the results; interrupted runs require a new output directory.

## Help

```bash
.venv/bin/python review/pipeline.py --help
.venv/bin/python review/pipeline.py preprocess --help
.venv/bin/python review/pipeline.py prepare --help
.venv/bin/python review/pipeline.py train --help
```
