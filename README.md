# Multitask-learning experiments

Run all commands from the repository root. Use Python 3.12. Native models run on CPU; MTT requires a supported GPU. See [DATASETS.md](DATASETS.md) for dataset features and targets.

## Install

```bash
python3.12 -m venv .venv
```

Install PyTorch and torchvision for your hardware using the [official installation instructions](https://pytorch.org/get-started/locally/), with `.venv/bin/python -m pip`. Then install the remaining dependencies:

The MTT launcher currently includes AMD ROCm defaults (`HIP_VISIBLE_DEVICES=0` and `TORCH_BLAS_PREFER_HIPBLASLT=0`). The original system used PyTorch 2.13.0+rocm10.0.0 and torchvision 0.28.0+rocm10.0.0 from AMD's gfx1201 wheel index.

For an NVIDIA GPU, install the PyTorch and torchvision CUDA builds recommended by the official selector for your GPU/driver. PyTorch uses the `torch.cuda` API for both CUDA and ROCm. The MultiTab-Net preflight requires one visible GPU. On NVIDIA, select it with `CUDA_VISIBLE_DEVICES=0` before running the pipeline.

```bash
.venv/bin/python -m pip install -r requirements.txt
```

## Install data

Download the experiment inputs from the anonymized [Google Drive Folder](https://drive.google.com/drive/u/3/folders/1-1jRbhQtV8b_8Dpwamz2SkG_xUxzGaKP). Install the files under the repository's `datasets/` directory, preserving the directory structure and filenames. Dataset labels and prepared-input paths are listed in `reproduction/datasets.json`.
ASSISTments 2019–2020, ASSISTments 2020–2021 and SPR X-ray must be obtained directly from their providers rather than from the Drive folder:

- ASSISTments 2019–2020: [OSF release](https://osf.io/q7zc5/) — [full-school-year ZIP](https://osf.io/download/7sftn/).
- ASSISTments 2020–2021: [OSF release](https://osf.io/7cgav/) — [aug_to_oct ZIP](https://osf.io/download/gny98/).
- SPR X-Ray: [Kaggle dataset](https://www.kaggle.com/datasets/felipekitamura/spr-x-ray-age-and-gender-dataset)

To start from the original source files instead of the prepared inputs, follow [Preprocessing from original sources](DATASETS.md#preprocessing-from-original-sources) at the end of DATASETS.md. That section contains source downloads, installation paths, access terms and preprocessing commands. 

## Prepare and train

`prepare` creates the experiment files in `--work-root`, including MTT data and configuration files, and verifies the export. For datasets supplied as prepared folds, it reuses those folds without repeating their preprocessing or changing their partitions. It does not train models. Use a new work directory.

`train` uses that work directory to fit and evaluate the models selected with `--models`, writing results under `--output-root`.

Use `--models local` for the local model, a subset such as `--models local mtt`, or `--models all` for every registered model for that dataset.

## ASSISTments Datasets

Published prepared bundles contain `fold_1.csv` through `fold_5.csv`. Keep the CSVs together at `datasets/<native dataset directory>/prepared/<label>/`, using the registered input path for that label. Then run:

```bash
.venv/bin/python reproduction/pipeline.py prepare --label assistments2009_sb --work-root artifacts/assistments2009-sb-prepared

.venv/bin/python reproduction/pipeline.py train --label assistments2009_sb --work-root artifacts/assistments2009-sb-prepared --models all --output-root artifacts/assistments2009-sb-results
```

## Other datasets

The pipeline reads each dataset from the `input_location` listed in `reproduction/datasets.json`. The expected preprocessed input is a table containing features and targets, a prepared-fold directory or the selected SPR X-Ray embeddings and accompanying files.

For example, with `datasets/whas1_preprocessed/whas1_preprocessed.csv` installed, run:

```bash
.venv/bin/python reproduction/pipeline.py prepare --label whas1_preprocessed --work-root artifacts/whas1-prepared

.venv/bin/python reproduction/pipeline.py train --label whas1_preprocessed --work-root artifacts/whas1-prepared --models all --output-root artifacts/whas1-results
```

### SPR processed inputs

Download SPR X-Ray dataset directly from its source. Once you have generated the selected-cohort inputs, install `spr_xray_embeddings.npy`, `spr_xray_manifest.csv` and `cohort_selection.csv` in `datasets/spr xray/prepared/spr_xray_manifest/`. Then use label `spr_xray_manifest` with `prepare` and `train`. These stages consume the existing embeddings and do not extract them from images.

## Outputs and retry

Training subprocesses use the Python interpreter that starts the pipeline. After moving or renaming the repository, rerun `prepare` into a new work directory before training: older preparation records and export metadata may contain absolute paths to the previous location. Do not manually rewrite those integrity-bound files.

Under the chosen `--output-root`, consolidated outputs are in `experiment_outputs/<label>/`: `metrics.csv`, native `predictions/` and metric tables in `results/`. MTT logs are in `mtt_logs/`.

Use new work/output directories. Repeating a completed training command verifies and reuses the results; interrupted runs require a new output directory.

## Help

```bash
.venv/bin/python reproduction/pipeline.py --help
.venv/bin/python reproduction/pipeline.py prepare --help
.venv/bin/python reproduction/pipeline.py train --help
```
