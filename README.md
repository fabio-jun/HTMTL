# Multitask-learning experiments

Run all commands from the repository root. Use Python 3.12. Native models run on CPU; MTT requires a supported GPU. See [DATASETS.md](DATASETS.md) for dataset features and targets.

## Install

```bash
python3.12 -m venv .venv
```

Install PyTorch and torchvision for your hardware using the [official installation instructions](https://pytorch.org/get-started/locally/), with `.venv/bin/python -m pip`. Then install the remaining dependencies:

The MTT launcher currently includes AMD ROCm defaults (`HIP_VISIBLE_DEVICES=0` and `TORCH_BLAS_PREFER_HIPBLASLT=0`). The original system used PyTorch 2.13.0+rocm10.0.0 and torchvision 0.28.0+rocm10.0.0 from AMD's gfx1201 wheel index.

For an NVIDIA GPU, install the PyTorch and torchvision **CUDA builds** recommended by the official selector for your GPU/driver instead of ROCm builds. PyTorch uses the `torch.cuda` API for both CUDA and ROCm. The MTT preflight requires one visible GPU; on NVIDIA, select it with `CUDA_VISIBLE_DEVICES=0` before running the pipeline. The ROCm environment settings do not select an NVIDIA GPU.

```bash
.venv/bin/python -m pip install -r requirements.txt
```

Check the installed packages with `.venv/bin/python -m pip check`. For MTT, verify GPU availability with `.venv/bin/python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"`; availability must be `True` and the visible device count must be `1`.

## Install data

Download the experiment inputs from [Google Drive](https://drive.google.com/drive/u/3/folders/1-1jRbhQtV8b_8Dpwamz2SkG_xUxzGaKP). Install the files under the repository's `datasets/` directory, preserving the directory structure and filenames. Dataset labels and prepared-input paths are listed in `review/datasets.json`. ASSISTments five-fold bundles belong at `datasets/<native dataset directory>/prepared/<label>/`. For raw preprocessing, place the complete original sources in the corresponding dataset directory. Follow the source-specific access and usage terms listed below.

**Source-only datasets:** ASSISTments 2019–2020, ASSISTments 2020–2021 and SPR X-ray must be downloaded directly from their original providers, not from the Google Drive folder above. Use the [ASSISTments 2019–2020 OSF release](https://osf.io/q7zc5/), the [ASSISTments 2020–2021 OSF release](https://osf.io/7cgav/) and the [SPR X-ray Kaggle source](https://www.kaggle.com/datasets/felipekitamura/spr-x-ray-age-and-gender-dataset). Each user must obtain access and accept the provider's terms. Source installation paths and preprocessing requirements are listed below.

## ASSISTments Datasets

`prepare` reuses the registered prepared bundle. Use `preprocess --work-root` only when intentionally reconstructing the five folds from the original sources:

Published prepared bundles contain only `fold_1.csv` through `fold_5.csv`; no input `manifest.json` or preprocessing `.joblib` files are required. Keep the CSVs together in a directory named after the dataset label, at its registered input path. Keep the student identifiers and `source_cv_role` columns: they define the existing partitions and are excluded from model features. The reader derives the schema from the CSV headers and validates the folds without refitting imputers or scalers. Legacy bundles with their sidecars remain supported.

```bash
.venv/bin/python review/pipeline.py prepare --label assistments2009_sb --work-root artifacts/assistments2009-sb-prepared
.venv/bin/python review/pipeline.py train --label assistments2009_sb --work-root artifacts/assistments2009-sb-prepared --models all --output-root artifacts/assistments2009-sb-results
```

```bash
.venv/bin/python review/pipeline.py preprocess --label assistments2009_sb --work-root artifacts/assistments2009-sb-reconstructed --workers 1
```

Use `--models local` for a CPU-only example, or a subset such as `--models local mtt`. `--models all` runs every registered model for that dataset.

## Other datasets

The pipeline reads each dataset from the `input_location` listed in `review/datasets.json`. This is the expected processed input: a table containing features and targets, a prepared-fold directory, or the selected SPR embeddings and accompanying files. It is not the original source download.

`preprocess` creates that input from the original sources. `prepare` uses the processed input to prepare the experiment inputs, including MTT export. `train` fits and evaluates the requested models.

### If the processed input is already installed

Skip `preprocess`. For example, if `datasets/whas1_preprocessed/whas1_preprocessed.csv` already exists, run only:

```bash
.venv/bin/python review/pipeline.py prepare --label whas1_preprocessed --work-root artifacts/whas1-prepared
.venv/bin/python review/pipeline.py train --label whas1_preprocessed --work-root artifacts/whas1-prepared --models all --output-root artifacts/whas1-results
```

### If you have only the original source files

For WHAS1, install `whas.dat` or `whas1.csv` in `datasets/whas1_preprocessed/raw/`. Then create the processed input:

```bash
.venv/bin/python review/pipeline.py preprocess --label whas1_preprocessed
```

After this command finishes, run the `prepare` and `train` commands above. `preprocess` refuses to overwrite existing processed outputs, so do not run it when the expected input is already installed.

### Combined reconstruction and preparation

`preprocess --label <label> --work-root <new-directory>` combines source reconstruction with experiment-input preparation and MTT export. Without `--work-root`, `preprocess` runs only the source preprocessor. The work directory stores preparation artifacts, not the original source files. Use a new directory and install the required sources first. Datasets without a registered reconstruction route require their processed input to be installed separately.

For Hydraulic, use `prepare` if the sensor's processed CSV is already installed. Otherwise, install the original sensor files and use `preprocess --label <sensor> --work-root <new-directory>` as described below. For ASSISTments, `prepare` reuses the registered five-fold bundle or reconstructs it from the registered raw sources if the bundle is missing.

### SPR processed inputs

Download SPR X-ray directly from its source. Once you have generated the selected-cohort inputs, install `spr_xray_embeddings.npy`, `spr_xray_manifest.csv` and `cohort_selection.csv` in `datasets/spr xray/prepared/spr_xray_manifest/`. Then use label `spr_xray_manifest` with `prepare` and `train`. These stages consume the existing embeddings and do not extract them from images.

HIGGS and SPR cohort reconstruction requires the original source. The pipeline always uses the included official selection map. For SPR, `--input` accepts either the historical full-embedding directory or a Kaggle dataset root containing `train_age.csv`, `train_gender.csv`, and the literal `kaggle/kaggle/train/` subtree:

```bash
.venv/bin/python review/pipeline.py preprocess --label higgs_50k --input datasets/higgs/train_val_test.h5
.venv/bin/python review/pipeline.py preprocess --label spr_xray_manifest --input '/path/to/spr xray' --output artifacts/spr-raw-selected --device cuda:0 --batch-size 16
```

The raw route needs only those Kaggle images and labels plus the included selection CSV: it derives the five-column manifest, verifies its pinned official hash before model loading, then extracts only selected training PNGs with RAD-DINO. No prebuilt SPR bundle or selected manifest is required. Its identity, schema, labels, and order are frozen; numerical embeddings can differ across Torch or GPU backends and are not asserted bitwise equal to the historical array.

## Optional source-data downloads

Use the prepared packages for experiment reproduction. Source downloads below are optional, may require accepting the provider's terms, and are not redistributed with this repository. Paths are relative to the repository root. A source release may need format conversion before the existing preprocessor can read it; changing an extension alone is not conversion.

### SUPPORT

Source: [Vanderbilt/HBiostat](https://hbiostat.org/data/) — [TSV](https://hbiostat.org/data/repo/support.tsv)

Download `support.tsv` unchanged to `datasets/support_withoutmissing_outputs/support.tsv`, then run `.venv/bin/python review/pipeline.py preprocess --label support_withoutmissing_outputs`. The reader accepts the original tab-delimited format; no manual CSV conversion is needed. It still accepts `support.csv` and gives that existing CSV priority when both files are present. The preprocessing generates `support_withoutmissing_outputs.csv` and the auxiliary `support_withmissing.csv`; existing outputs are protected against overwriting.

### SUPPORT2

Source: [Vanderbilt/HBiostat](https://hbiostat.org/data/) — [CSV ZIP](https://hbiostat.org/data/repo/support2csv.zip)

Extract `support2.csv` to `datasets/support2_withoutmissing_outputs/support2.csv`.

### WHAS1 (481 records, not WHAS100)

Download: [Google Drive](https://drive.google.com/drive/u/3/folders/1-1jRbhQtV8b_8Dpwamz2SkG_xUxzGaKP).

Extract `whas1_preprocessed.zip`, take `whas1.csv` and place it at `datasets/whas1_preprocessed/raw/whas1.csv` to run preprocessing.

### WHAS500

Download: [Google Drive](https://drive.google.com/drive/u/3/folders/1-1jRbhQtV8b_8Dpwamz2SkG_xUxzGaKP).

Extract `whas500_preprocessed.zip`, take `whas500.csv` and place it at `datasets/whas500_preprocessed/raw/whas500.csv` to run preprocessing.

### ASSISTments 2009–2010 SB/NSB

Source: [SB release](https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data/skill-builder-data-2009-2010), [NSB release](https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data/non-skill-builder-data-2009-10)

Install the matching original releases as `datasets/assistments2009-2010/sb.csv` and `nsb.csv`. Original Drive links may require Google access. The corrected/collapsed SB export and the combined Full Data export are not verified substitutes for these benchmark sources.

### ASSISTments Goldstein P(J), 2011

Source: [Official page](https://sites.google.com/site/assistmentsdata/datasets/2011-goldstein-baker-heffernan) — [WPI ZIP](http://web.cs.wpi.edu/~nth/PublicScienceArchive/assistments-p(j)-data.zip)

Extract `PJ Data (Features Added).csv` and `PJ Data (Actions).csv` to `datasets/assistments_paper:2011_goldstein/`. Preserve spaces, parentheses and the colon in the directory name.

### ASSISTments 2012–2013

Source: [Official release](https://sites.google.com/site/assistmentsdata/datasets/2012-13-school-data-with-affect) — [ZIP](https://drive.google.com/file/d/1cU6Ft4R3hLqA7G1rIGArVfelSZvc6RxY/view?usp=sharing)

Extract the CSV with affect predictions and install as `datasets/assistments2012-2013/data.csv`. Preserve the `actions` column. One source serves both SB and NSB.

### ASSISTments Controlled Experiments

Source: [Study supplement](https://sites.google.com/site/las2016data/data/thison) — [ThisOne.csv](https://drive.google.com/file/d/16zGibMINqvQAef_FkP9f78Y4OlON4NrX/view?usp=drive_web)

Save as `datasets/assistments_from_controlled_experiments/ThisOne.csv`. Use the CSV release, not an arbitrary export of the spreadsheet.

### ASSISTments 2017

Source: [Official access page](https://sites.google.com/view/assistmentsdatamining/dataset)

Complete the provider's access form and accept its terms; the download link is sent by email. Install as `datasets/assistments2017/Assistments2017.csv`. No public direct-download URL is supplied here.

### ASSISTments 2019–2020

The [source terms](https://osf.io/download/b7myu/) prohibit redistribution. Each collaborator must download this dataset directly from the original source, not from the Google Drive folder.

Source: [OSF release](https://osf.io/q7zc5/) — [full-school-year ZIP](https://osf.io/download/7sftn/)

Extract the complete archive to `datasets/assistments2019-2020/2019-2020_school_year/`, preserving filenames. The readers need `adets.csv`, `pdets.csv`, `plogs.csv`, `alogs.csv`, and `slogs.csv`; do not substitute the pre/post-shutdown subsets.

### ASSISTments 2020–2021

The [source terms](https://osf.io/download/tpm4b/) prohibit redistribution. Each collaborator must download this dataset directly from the original source, not from the Google Drive folder.

Source: [OSF release](https://osf.io/7cgav/) — [aug_to_oct ZIP](https://osf.io/download/gny98/)

Extract this block to `datasets/assistments2020-2021/aug_to_oct/`, preserving filenames. The readers need `adets.csv`, `pdets.csv`, `plogs.csv`, `alogs.csv`, and `slogs.csv`. Other period blocks are not inputs to this benchmark.

### Auction Verification

Source: [UCI release](https://archive.ics.uci.edu/dataset/713/auction+verification) — [ZIP](https://archive.ics.uci.edu/static/public/713/auction+verification.zip)

Place the release's data CSV at `datasets/auction_preprocessed/data.csv`; retain its original header.

### SPR X-ray

The source prohibits copying and redistribution. Download SPR X-ray directly from its original provider, not from the Google Drive folder.

Source: [Kaggle dataset page](https://www.kaggle.com/datasets/felipekitamura/spr-x-ray-age-and-gender-dataset) / [original competition](https://www.kaggle.com/competitions/spr-x-ray-gender)

Image downloads may require a Kaggle account and accepted competition rules. Raw extraction needs labeled training PNGs in `kaggle/kaggle/train/` plus `train_age.csv` and `train_gender.csv` at the Kaggle dataset root. Obtain SPR data directly from the source; it is not redistributed with this repository.

SUPPORT/SUPPORT2 and the public WHAS book-data files can be downloaded without CITI training. Follow the source's attribution and usage terms. These links identify sources, not a newly validated raw-to-results reproduction route.


### HIGGS source used by this pipeline

Source: [MultiTab](https://github.com/Armanfard-Lab/MultiTab) — [authors' HDF5 collection](https://huggingface.co/datasets/Dimitri-Sinodinos/multitab/tree/main) — [higgs.h5](https://huggingface.co/datasets/Dimitri-Sinodinos/multitab/resolve/main/higgs.h5?download=true)

Download `higgs.h5`, rename it to `train_val_test.h5`, and place it at `datasets/higgs/train_val_test.h5`. This is an upstream prepared HDF5, not the UCI raw CSV; the HIGGS cohort reconstruction command expects this HDF5 and the repository's fixed selection map.

### Hydraulic (17 sensor views)

Source: [UCI release](https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems) — [ZIP](https://archive.ics.uci.edu/static/public/447/condition+monitoring+of+hydraulic+systems.zip)

Extract the sensor `*.txt` files and `profile.txt` into `datasets/hydraulic/data/`. Run `preprocess --label <sensor> --work-root <new-directory>` to extract that sensor's tsfresh features and prepare its folds/MTT inputs. The source files must be installed first, and existing processed CSVs are protected against overwriting. CP is written to `datasets/EPS1/CP.csv`; the other views use `datasets/<sensor>/<sensor>.csv`.

## Outputs and retry

Training subprocesses use the Python interpreter that starts the pipeline. After moving or renaming the repository, rerun `prepare` into a new work directory before training: older preparation records and export metadata may contain absolute paths to the previous location. Do not manually rewrite those integrity-bound files.

Under the chosen `--output-root`, consolidated outputs are in `experiment_outputs/<label>/`: `metrics.csv`, native `predictions/` and metric tables in `results/`. MTT logs are in `mtt_logs/`.

Use new work/output directories. Repeating a completed training command verifies and reuses the results; interrupted runs require a new output directory.

## Help

```bash
.venv/bin/python review/pipeline.py --help
.venv/bin/python review/pipeline.py preprocess --help
.venv/bin/python review/pipeline.py prepare --help
.venv/bin/python review/pipeline.py train --help
```
