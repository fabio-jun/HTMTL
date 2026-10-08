from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_release_root = _Path(__file__).resolve().parents[2]
if str(_release_root / "review") not in _sys.path:
    _sys.path.insert(0, str(_release_root / "review"))
from dataset_registry import configure_imports
configure_imports(_release_root)


import json
import os
import subprocess
from pathlib import Path

import h5py
import yaml

from mtt_shared_registry import (
    LABELS,
    N_FOLDS,
    assert_scope,
    config_root,
    data_root,
    log_root,
)
from mtt_shared_validation import validate_fold_metrics, verify_hparams

FOLDS = tuple(range(1, N_FOLDS + 1))
TOTAL_JOBS = len(LABELS) * N_FOLDS


def build_command(project_root, data_root_path, config_root_path, log_root_path, label, fold):
    project_root = Path(project_root)

    return [
        str(project_root / ".venv/bin/python"),
        str(project_root / "src/methods/MultiTab/main.py"),
        "--data_root", str(data_root_path),
        "--model", "mtt",
        "--dataset", f"{label}/fold_{fold}",
        "--seed", "42",
        "--patience", "5",
        "--cuda", "0",
        "--run_name", f"{label}_fold_{fold}",
        "--log-root", str(log_root_path),
        "--config-root", str(config_root_path),
        "--mtt-protocol", "selection-refit",
    ]


def device_environment():
    environment = os.environ.copy()
    environment["HIP_VISIBLE_DEVICES"] = "0"
    environment["TORCH_BLAS_PREFER_HIPBLASLT"] = "0"
    return environment


def fold_base(log_root_path, label, fold):
    return Path(log_root_path) / f"{label}_fold_{fold}" / f"fold_{fold}"


def load_row_counts(spec, data_path, fold):
    manifest = json.loads((Path(data_path) / spec.label / f"fold_{fold}" / "manifest.json").read_text(encoding="utf-8"))
    return manifest["row_counts"]


def fold_complete(spec, log_root_path, fold, data_path, config_path):
    base = fold_base(log_root_path, spec.label, fold)
    metrics, hparams = base / "metrics.csv", base / "hparams.yaml"
    if not metrics.is_file() or not hparams.is_file():
        return False

    try:
        row_counts = load_row_counts(spec, data_path, fold)
        verify_hparams(spec, hparams, fold, data_root_path=Path(data_path), config_root_path=Path(config_path),
                       row_counts=row_counts)
        validate_fold_metrics(spec, metrics, fold)
    except Exception:
        return False

    return True


def _preflight(spec, data_path, config_path, folds):
    label_data = Path(data_path) / spec.label
    label_config = Path(config_path) / spec.label

    for fold in folds:
        h5_path = label_data / f"fold_{fold}" / "train_val_test.h5"
        dataset_yaml = label_config / f"fold_{fold}" / "dataset.yaml"
        mtt_yaml = label_config / f"fold_{fold}" / "mtt.yaml"

        for path in (h5_path, dataset_yaml, mtt_yaml):
            if path.is_symlink() or not path.is_file():
                raise FileNotFoundError(f"Missing MTT shared-input fold input: {path}")

        with h5py.File(h5_path, "r") as handle:
            if set(handle) != {"train", "val", "test"}:
                raise ValueError(f"Malformed MTT shared-input HDF5: {h5_path}")
            if spec.profile == "higgs50k":
                for role, rows in (("train", 32000), ("val", 8000), ("test", 10000)):
                    if handle[role]["features_numerical"].shape != (rows, 21):
                        raise ValueError("HIGGS role shape mismatch")

        data_config = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))["data"]
        if data_config.get("path") != f"/{spec.label}/fold_{fold}/" or data_config.get("num_features") != spec.num_features:
            raise ValueError(f"{spec.label} fold {fold}: dataset config contract mismatch")
        if data_config.get("categorical_cardinalities") != []:
            raise ValueError(f"{spec.label} fold {fold}: unexpected categorical cardinalities")

        model_config = yaml.safe_load(mtt_yaml.read_text(encoding="utf-8"))
        if model_config["training"].get("optimizer") != "adam" or int(model_config["training"].get("batch_size")) != 32:
            raise ValueError(f"{spec.label} fold {fold}: MTT config contract mismatch")
        if model_config["model"].get("att_type") != spec.att_type:
            raise ValueError(f"{spec.label} fold {fold}: MTT architecture mismatch")


def _write_exit_marker(path, code):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(str(code), encoding="utf-8")
    os.replace(temporary, path)


def runtime_preflight(python, environment):
    code = (
        "import torch,pytorch_lightning,torchmetrics,easydict,rotary_embedding_torch; "
        "assert torch.cuda.is_available(); assert torch.cuda.device_count()==1"
    )
    result = subprocess.run([str(python), "-c", code], capture_output=True, text=True, check=False,
                            timeout=120, env=environment)
    if result.returncode:
        raise RuntimeError("MTT shared-input ROCm runtime preflight failed: " + result.stderr.strip())


def valid_fold_count(specs, log_root_path, data_path, config_path):
    count = 0
    for spec in specs.values():
        for fold in FOLDS:
            if fold_complete(spec, log_root_path, fold, data_path, config_path):
                count += 1
    return count


def run(project_root, labels=None, folds=None, specs=None, runner=subprocess.run,
        runtime_check=runtime_preflight, force=False, data_path=None, config_path=None,
        log_root_path=None, exit_root=None):
    root = Path(project_root)
    specs = assert_scope(root, labels=labels) if specs is None else specs
    labels = list(specs) if labels is None else list(labels)
    unknown = [label for label in labels if label not in specs]
    if unknown:
        raise ValueError(f"Unknown migration label(s): {unknown}")

    selected = list(FOLDS if folds is None else folds)
    if not selected or not set(selected) <= set(FOLDS):
        raise ValueError(f"Invalid fold selection: {selected}")

    data_path = Path(data_path) if data_path is not None else data_root(root)
    config_path = Path(config_path) if config_path is not None else config_root(root)
    log_root_path = Path(log_root_path) if log_root_path is not None else log_root(root)
    exit_root = Path(exit_root) if exit_root is not None else Path(root) / "artifacts/runs/mtt-shared-input/exitcodes"

    for name, path in (("data_path", data_path), ("config_path", config_path), ("log_root", log_root_path)):
        if path.is_symlink():
            raise ValueError(f"MTT shared-input {name} must not be a symlink: {path}")

    for label in labels:
        _preflight(specs[label], data_path, config_path, selected)

    environment = device_environment()
    runtime_check(Path(root) / ".venv/bin/python", environment)

    for label in labels:
        spec = specs[label]
        for fold in selected:
            if not force and fold_complete(spec, log_root_path, fold, data_path, config_path):
                print(f"[{label}] fold {fold}: complete metrics present, resuming (skipping)", flush=True)
                continue

            print(f"[{label}] fold {fold}/{len(FOLDS)}: running selection-refit", flush=True)
            result = runner(
                build_command(root, data_path, config_path, log_root_path, spec.label, fold),
                check=False, cwd=root / "src/methods/MultiTab", env=environment,
            )

            code = int(result.returncode)
            marker = exit_root / f"{label}_fold_{fold}.exitcode"

            if code:
                _write_exit_marker(marker, code)
                print(f"[{label}] fold {fold} failed with exit {code}; failing closed", flush=True)
                _write_exit_marker(exit_root / "batch.exitcode", code)
                if spec.profile == "higgs50k":
                    _write_exit_marker(exit_root / "mtt.exitcode", code)
                return code

            if spec.profile == "higgs50k" and not fold_complete(spec, log_root_path, fold, data_path, config_path):
                _write_exit_marker(exit_root / "mtt.exitcode", 70)
                return 70

            _write_exit_marker(marker, 0)

    completed = valid_fold_count(specs, log_root_path, data_path, config_path)

    if len(labels) == 1 and specs[labels[0]].profile == "higgs50k":
        if not all(fold_complete(specs[labels[0]], log_root_path, fold, data_path, config_path) for fold in selected):
            _write_exit_marker(exit_root / "mtt.exitcode", 70)
            return 70
        _write_exit_marker(exit_root / "mtt.exitcode", 0)
        return 0

    if completed != len(labels) * N_FOLDS:
        print(f"[mtt-shared-input] only {completed}/{len(labels) * N_FOLDS} selected folds are valid; refusing success", flush=True)
        _write_exit_marker(exit_root / "batch.exitcode", 1)
        return 1

    _write_exit_marker(exit_root / "batch.exitcode", 0)
    if any(specs[label].profile == "higgs50k" for label in labels):
        _write_exit_marker(exit_root / "mtt.exitcode", 0)

    print(f"[mtt-shared-input] all {len(labels) * N_FOLDS} selected folds valid; aggregate marker written", flush=True)
    return 0
