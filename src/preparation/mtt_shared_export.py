from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import IterativeImputer
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold, train_test_split
from sklearn.preprocessing import StandardScaler

from fold_preparation import read_prepared_folds
from mtt_shared_registry import (
    N_FOLDS,
    SEED,
    SELECTION_VAL_FRACTION,
    SPR_COHORT_SELECTION_NAME,
    SPR_EMBEDDINGS_NAME,
    SPR_MANIFEST_NAME,
    SPR_ROWS,
    CLASSIC_FAMILY,
    PREPARED_FAMILY,
    SPR_FAMILY,
    assert_scope,
    build_specs,
    config_root,
    data_root,
    mtt_config,
)
from spr_xray_loader import load_spr_xray

ROLES = ("train", "val", "test")
BATCH_MANIFEST = "batch_manifest.json"
BUNDLE_MANIFEST = "bundle_manifest.json"
IMPUTER_NAME = "IterativeImputer(random_state=42, keep_empty_features=True)"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _assert_regular_file(path, description):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{description} must be a regular non-symlink file: {path}")
    return path


def _assert_regular_dir(path, description):
    path = Path(path)
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"{description} must be a regular non-symlink directory: {path}")
    return path


def selection_split(groups, stratify, n, seed=SEED, val_fraction=SELECTION_VAL_FRACTION):
    positions = np.arange(n)
    if groups is not None:
        groups = np.asarray(groups)
        if len(groups) != n:
            raise ValueError("Internal split groups must align with rows")
        splitter = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
        train_pos, val_pos = next(splitter.split(positions, groups=groups))
    else:
        train_pos, val_pos = train_test_split(
            positions, test_size=val_fraction, random_state=seed, stratify=stratify
        )
    train_pos = np.sort(np.asarray(train_pos, dtype=np.int64))
    val_pos = np.sort(np.asarray(val_pos, dtype=np.int64))

    if len(train_pos) == 0 or len(val_pos) == 0:
        raise ValueError("Internal selection split produced an empty partition")
    if np.intersect1d(train_pos, val_pos).size:
        raise ValueError("Internal selection split partitions overlap")

    return train_pos, val_pos


def classic_transform(X_train, X_test):
    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    imputer = None
    if np.isnan(X_train).any():
        imputer = IterativeImputer(random_state=SEED, keep_empty_features=True)
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)

    scaler = StandardScaler().fit(X_train)
    provenance = {
        "kind": "imputed" if imputer is not None else "identity_impute",
        "imputer": IMPUTER_NAME if imputer is not None else None,
        "scaler_mean": [float(value) for value in scaler.mean_],
        "scaler_scale": [float(value) for value in scaler.scale_],
        "fit_population": "fold_outer_train_before_selection_split",
    }

    return scaler.transform(X_train), scaler.transform(X_test), provenance


@dataclass
class FoldBundle:
    label: str
    fold: int
    family: str
    feature_names: tuple
    tasks: list
    role_features: dict
    role_targets: dict
    role_identities: dict
    role_indices: dict
    transform: dict
    selection: dict
    source_hashes: dict
    native_models: tuple
    short_name: str

    @property
    def target_order(self):
        return [source for source, _output, _kind in self.tasks]


class _ClassicSource:
    family = CLASSIC_FAMILY

    def __init__(self, spec, project_root):
        self.spec = spec
        self.path = _assert_regular_file(spec.source, "Classic canonical CSV")

        self.frame = pd.read_csv(self.path)

        self.feature_names = tuple(spec.feature_columns)

        if spec.profile == "higgs50k" and (len(self.frame) != 50000 or sha256_file(self.path) != "bc149ac7c5dcaf4e3d2d0b61bd4cc5f393412e1bd2b595aec249613dc4439a6a"):
            raise ValueError("HIGGS requires the exact official 50k CSV")

        self.targets = {target: self.frame[spec.aliases[target] if spec.profile == "higgs50k" else target].to_numpy(dtype=float) for target in spec.aliases}
        for target, values in self.targets.items():
            if not np.isfinite(values).all():
                raise ValueError(f"{spec.label} target {target} is missing or nonfinite")
        self.X = self.frame[list(self.feature_names)].to_numpy(dtype=float)

        self.groups = self.frame[spec.group_column].to_numpy() if spec.group_column else None
        self.native_models = spec.native_models

    def outer_folds(self):
        if self.groups is not None:
            splitter = GroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
            splits = splitter.split(self.X, groups=self.groups)
        else:
            splitter = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
            splits = splitter.split(self.X)
        return [(np.asarray(train, dtype=np.int64), np.asarray(test, dtype=np.int64)) for train, test in splits]

    def first_binary(self, indices):
        for target, kind in self.spec.target_types.items():
            if kind == "binary":
                return self.targets[target][indices]
        return None

    def identity(self, indices):
        frame = pd.DataFrame({"row_index": np.asarray(indices, dtype=np.int64)})
        if self.spec.profile == "higgs50k":
            for task, target in self.spec.aliases.items():
                frame[target] = self.targets[task][indices]
        return frame

    def transform(self, train_idx, test_idx):
        train, test, provenance = classic_transform(self.X[train_idx], self.X[test_idx])
        if self.spec.profile == "higgs50k":
            provenance["kind"] = "identity" if provenance["imputer"] is None else "imputed"
            provenance["fit_population"] = "fold_outer_train_40000"
        return train, test, provenance

    def source_hashes(self):
        return {str(self.path): sha256_file(self.path)}


class _PreparedSource:
    family = PREPARED_FAMILY

    def __init__(self, spec, project_root):
        self.spec = spec

        self.folds = read_prepared_folds(spec.source)

        self.feature_names = tuple(spec.feature_columns)
        self.metadata = [
            column
            for column in self.folds[0]["manifest"]["schema"]["columns"]
            if column not in spec.feature_columns and column not in spec.target_types
        ]
        self.native_models = spec.native_models

    def outer_folds(self):
        return [
            (entry["train_frame"], entry["validation_frame"])
            for entry in self.folds
        ]

    def first_binary(self, indices):
        return None  # stratify is replaced by manifest student groups

    def identity(self, frame):
        audit = frame[self.metadata].reset_index(drop=True)
        audit.insert(0, "row_index", np.arange(len(frame), dtype=np.int64))
        return audit

    def transform(self, train_frame, test_frame):
        roles = {"train": train_frame, "test": test_frame}
        features = {
            role: frame[list(self.feature_names)].to_numpy(dtype=float) for role, frame in roles.items()
        }
        provenance = {
            "kind": "prepared_materialized",
            "imputer": None,
            "scaler_mean": None,
            "scaler_scale": None,
            "fit_population": "prepared_bundle_train_only_offline",
        }
        return features["train"], features["test"], provenance

    def source_hashes(self):
        manifest = self.spec.source / "manifest.json"
        return {str(manifest): sha256_file(manifest)}


class _SprSource:
    family = SPR_FAMILY

    def __init__(self, spec, project_root):
        self.spec = spec
        source = _assert_regular_dir(spec.source, "SPR source directory")
        self.embeddings_path = _assert_regular_file(source / SPR_EMBEDDINGS_NAME, "SPR embeddings")
        self.manifest_path = _assert_regular_file(source / SPR_MANIFEST_NAME, "SPR manifest")
        self.cohort_path = _assert_regular_file(source / SPR_COHORT_SELECTION_NAME, "SPR cohort selection")

        self.frame = load_spr_xray(str(self.embeddings_path), str(self.manifest_path))
        if len(self.frame) != SPR_ROWS:
            raise ValueError(f"SPR cohort must contain exactly {SPR_ROWS} rows, got {len(self.frame)}")

        self.feature_names = tuple(spec.feature_columns)

        self.targets = {
            target: self.frame[target].to_numpy(dtype=float) for target in spec.aliases
        }
        for target, values in self.targets.items():
            if not np.isfinite(values).all():
                raise ValueError(f"SPR target {target} is missing or nonfinite")
        self.X = self.frame[list(self.feature_names)].to_numpy(dtype=float)

        self.groups = None
        cohort = pd.read_csv(self.cohort_path)
        expected = cohort["legacy_sample_row_index"].to_numpy(dtype=np.int64)
        if len(expected) != SPR_ROWS:
            raise ValueError("SPR cohort selection row count mismatch")
        self.legacy = expected
        self.native_models = spec.native_models

    def outer_folds(self):
        splitter = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        return [(np.asarray(train, dtype=np.int64), np.asarray(test, dtype=np.int64)) for train, test in splitter.split(self.X)]

    def first_binary(self, indices):
        return self.frame["target_binary_gender"].to_numpy()[indices]

    def identity(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        return pd.DataFrame({
            "row_index": indices,
            "legacy_row_index": self.legacy[indices],
        })

    def transform(self, train_idx, test_idx):
        scaler = StandardScaler().fit(self.X[train_idx])
        provenance = {
            "kind": "identity_impute",
            "imputer": None,
            "scaler_mean": [float(value) for value in scaler.mean_],
            "scaler_scale": [float(value) for value in scaler.scale_],
            "fit_population": "fold_outer_train_before_selection_split",
        }
        return scaler.transform(self.X[train_idx]), scaler.transform(self.X[test_idx]), provenance

    def source_hashes(self):
        return {
            str(self.embeddings_path): sha256_file(self.embeddings_path),
            str(self.manifest_path): sha256_file(self.manifest_path),
            str(self.cohort_path): sha256_file(self.cohort_path),
        }


def open_source(spec, project_root, source_factory=None):
    source_factory = source_factory or {CLASSIC_FAMILY: _ClassicSource, PREPARED_FAMILY: _PreparedSource, SPR_FAMILY: _SprSource}
    if spec.family not in source_factory:
        raise ValueError(f"Unsupported migration family: {spec.family}")
    return source_factory[spec.family](spec, project_root)


def _source_roles(spec, source, fold):
    outer = source.outer_folds()
    if len(outer) != N_FOLDS:
        raise ValueError(f"{spec.label} must expose exactly {N_FOLDS} outer folds")
    if spec.family == PREPARED_FAMILY:
        train_frame, test_frame = outer[fold - 1]
        groups = train_frame[spec.group_column].to_numpy()
        train_pos, val_pos = selection_split(groups, None, len(train_frame))
        X_train_t, X_test_t, provenance = source.transform(train_frame, test_frame)
        role_features = {
            "train": X_train_t[train_pos],
            "val": X_train_t[val_pos],
            "test": X_test_t,
        }
        role_targets = {
            "train": {t: train_frame[t].to_numpy(dtype=float)[train_pos] for t in spec.aliases},
            "val": {t: train_frame[t].to_numpy(dtype=float)[val_pos] for t in spec.aliases},
            "test": {t: test_frame[t].to_numpy(dtype=float) for t in spec.aliases},
        }
        role_frames = {
            "train": train_frame.iloc[train_pos],
            "val": train_frame.iloc[val_pos],
            "test": test_frame,
        }
        role_identities = {role: source.identity(frame) for role, frame in role_frames.items()}
        role_indices = {role: np.arange(len(frame), dtype=np.int64) for role, frame in role_frames.items()}
    else:
        train_idx, test_idx = outer[fold - 1]
        stratify = source.first_binary(train_idx)
        if spec.profile == "higgs50k":
            train_pos, val_pos = train_test_split(np.arange(len(train_idx)), test_size=8000, random_state=SEED, stratify=stratify)
        else:
            train_pos, val_pos = selection_split(None, stratify, len(train_idx))
        X_train_t, X_test_t, provenance = source.transform(train_idx, test_idx)
        role_features = {
            "train": X_train_t[train_pos],
            "val": X_train_t[val_pos],
            "test": X_test_t,
        }
        role_targets = {
            "train": {t: source.targets[t][train_idx][train_pos] for t in spec.aliases},
            "val": {t: source.targets[t][train_idx][val_pos] for t in spec.aliases},
            "test": {t: source.targets[t][test_idx] for t in spec.aliases},
        }
        role_identities = {
            "train": source.identity(train_idx[train_pos]),
            "val": source.identity(train_idx[val_pos]),
            "test": source.identity(test_idx),
        }
        role_indices = {
            "train": train_idx[train_pos],
            "val": train_idx[val_pos],
            "test": test_idx,
        }
    uses_groups = bool(spec.group_column) and (
        spec.family == PREPARED_FAMILY or getattr(source, "groups", None) is not None
    )

    tasks = [(source_name, spec.aliases[source_name], spec.target_types[source_name]) for source_name in spec.aliases]
    bundle = FoldBundle(
        label=spec.label,
        fold=fold,
        family=spec.family,
        feature_names=tuple(spec.feature_columns),
        tasks=tasks,
        role_features=role_features,
        role_targets=role_targets,
        role_identities=role_identities,
        role_indices=role_indices,
        transform=provenance,
        selection={
            "method": "GroupShuffleSplit" if uses_groups else "train_test_split",
            "val_fraction": SELECTION_VAL_FRACTION,
            "random_state": SEED,
            "train_count": int(len(train_pos)),
            "val_count": int(len(val_pos)),
        },
        source_hashes=source.source_hashes(),
        native_models=source.native_models,
        short_name=spec.short_name,
    )
    if spec.profile == "higgs50k":
        bundle.selection = {"method": "train_test_split", "test_size": 8000, "random_state": SEED,
                            "stratify": "target_binary_Target", "train_count": 32000, "val_count": 8000}
    return bundle


def build_fold(spec, project_root, fold, source_factory=None):
    if fold not in range(1, N_FOLDS + 1):
        raise ValueError(f"Invalid fold: {fold}")
    source = open_source(spec, project_root, source_factory)
    return _source_roles(spec, source, fold)


def _dataset_config(bundle, num_features):
    return {"data": {
        "format": "h5",
        "name": f"{bundle.label}_fold_{bundle.fold}",
        "short_name": bundle.short_name,
        "path": f"/{bundle.label}/fold_{bundle.fold}/",
        "tasks": bundle.target_order,
        "task_type": {source: kind for source, _output, kind in bundle.tasks},
        "task_out_dim": {source: 1 for source, _output, _kind in bundle.tasks},
        "num_features": int(num_features),
        "categorical_cardinalities": [],
    }}


def _write_fold(folder, bundle):
    folder.mkdir(parents=True, exist_ok=True)
    h5_path = folder / "train_val_test.h5"
    with h5py.File(h5_path, "w") as handle:
        for role in ROLES:
            group = handle.create_group(role)
            group.create_dataset("features_numerical", data=np.asarray(bundle.role_features[role], dtype=np.float32))
            for source_name, _output, _kind in bundle.tasks:
                group.create_dataset(source_name, data=np.asarray(bundle.role_targets[role][source_name], dtype=np.float32))
            bundle.role_identities[role].to_csv(folder / f"{role}_audit.csv", index=False)
    return h5_path


def _fold_manifest(bundle, h5_path, folder, dataset_text, mtt_text, spec=None):
    manifest = {
        "label": bundle.label,
        "fold": bundle.fold,
        "family": bundle.family,
        "seed": SEED,
        "short_name": bundle.short_name,
        "feature_names": list(bundle.feature_names),
        "feature_count": len(bundle.feature_names),
        "task_map": [[source, output, kind] for source, output, kind in bundle.tasks],
        "row_counts": {role: int(len(bundle.role_indices[role])) for role in ROLES},
        "row_indices": {role: bundle.role_indices[role].tolist() for role in ROLES},
        "selection": bundle.selection,
        "transform": bundle.transform,
        "source_hashes": bundle.source_hashes,
        "native_models": list(bundle.native_models),
        "hdf5_sha256": sha256_file(h5_path),
        "audit_sha256": {role: sha256_file(folder / f"{role}_audit.csv") for role in ROLES},
        "dataset_config_sha256": sha256_text(dataset_text),
        "mtt_config_sha256": sha256_text(mtt_text),
    }
    if spec is not None and spec.profile == "higgs50k":
        for key in ("family", "short_name", "native_models", "source_hashes"):
            manifest.pop(key)
        manifest["source_csv"] = str(spec.source)
        manifest["source_csv_sha256"] = sha256_file(spec.source)
        manifest["task_map"] = [[target, task, spec.target_types[task]] for task, target in spec.aliases.items()]
        manifest["mtt_config"] = mtt_config(spec)
        manifest["preprocessing_fit_population"] = "fold_outer_train_40000_rows_before_selection_split"
    return manifest


def _write_label(spec, project_root, data_root_path, config_root_path, force=False, source_factory=None):
    label_data = Path(data_root_path) / spec.label

    label_config = Path(config_root_path) / spec.label
    if not force and (label_data.exists() or label_config.exists()):
        raise FileExistsError(f"Refusing to overwrite MTT shared-input artifacts for {spec.label}")
    Path(data_root_path).mkdir(parents=True, exist_ok=True)
    Path(config_root_path).mkdir(parents=True, exist_ok=True)
    temp_data = Path(tempfile.mkdtemp(prefix=f".tmp_{spec.label}_", dir=data_root_path))
    temp_config = Path(tempfile.mkdtemp(prefix=f".tmp_{spec.label}_cfg_", dir=config_root_path))
    manifests = []

    try:
        for fold in range(1, N_FOLDS + 1):
            bundle = build_fold(spec, project_root, fold, source_factory=source_factory)
            fold_data = temp_data / f"fold_{fold}"
            h5_path = _write_fold(fold_data, bundle)
            fold_config = temp_config / f"fold_{fold}"
            fold_config.mkdir(parents=True)
            dataset_text = yaml.safe_dump(_dataset_config(bundle, spec.num_features), sort_keys=False)
            mtt_text = yaml.safe_dump(mtt_config(spec), sort_keys=False)
            (fold_config / "dataset.yaml").write_text(dataset_text, encoding="utf-8")
            (fold_config / "mtt.yaml").write_text(mtt_text, encoding="utf-8")
            manifest = _fold_manifest(bundle, h5_path, fold_data, dataset_text, mtt_text, spec)
            (fold_data / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            manifests.append(manifest)
            print(f"[{spec.label}] fold {fold}: train={manifest['row_counts']['train']} "
                  f"val={manifest['row_counts']['val']} test={manifest['row_counts']['test']}", flush=True)
        bundle_manifest = {
            "label": spec.label,
            "family": spec.family,
            "feature_names": list(spec.feature_columns),
            "task_map": manifests[0]["task_map"],
            "folds": [manifest["fold"] for manifest in manifests],
            "fold_manifest_sha256": {
                str(manifest["fold"]): sha256_file(temp_data / f"fold_{manifest['fold']}" / "manifest.json")
                for manifest in manifests
            },
            "source_hashes": manifests[0].get("source_hashes", {}),
        }
        if spec.profile == "higgs50k":
            bundle_manifest = {key: bundle_manifest[key] for key in ("label", "folds", "fold_manifest_sha256")}
            bundle_manifest.update(source_csv=str(spec.source), source_csv_sha256=sha256_file(spec.source))
        (temp_data / BUNDLE_MANIFEST).write_text(json.dumps(bundle_manifest, indent=2), encoding="utf-8")
        backups = []
        try:
            for target in (label_data, label_config):
                if target.exists() or target.is_symlink():
                    backup = target.with_name(target.name + f".backup_{os.getpid()}")
                    if backup.exists():
                        shutil.rmtree(backup, ignore_errors=True)
                    os.replace(target, backup)
                    backups.append((target, backup))
            os.replace(temp_data, label_data)
            os.replace(temp_config, label_config)
        except BaseException:
            for target, backup in reversed(backups):
                shutil.rmtree(target, ignore_errors=True)
                if backup.exists():
                    os.replace(backup, target)
            raise
        for _target, backup in backups:
            shutil.rmtree(backup, ignore_errors=True)
    except BaseException:
        shutil.rmtree(label_data, ignore_errors=True)
        shutil.rmtree(label_config, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(temp_data, ignore_errors=True)
        shutil.rmtree(temp_config, ignore_errors=True)
    return label_data


def export_labels(project_root, labels=None, force=False, specs=None, data_path=None, config_path=None):
    root = Path(project_root)
    specs = assert_scope(root, labels=labels) if specs is None else specs
    labels = list(specs) if labels is None else list(labels)
    unknown = [label for label in labels if label not in specs]
    if unknown:
        raise ValueError(f"Unknown migration label(s): {unknown}")
    data_path = Path(data_path) if data_path is not None else data_root(root)
    config_path = Path(config_path) if config_path is not None else config_root(root)
    written = []

    for label in labels:
        spec = specs[label]
        target_data = data_path / spec.label
        if not force and (target_data / BUNDLE_MANIFEST).is_file():
            print(f"[{label}] existing bundle present; skipping", flush=True)
            written.append(label)
            continue
        _write_label(spec, root, data_path, config_path, force=force)
        written.append(label)

    _write_batch_manifest(root, specs, data_path)
    return written


def _write_batch_manifest(project_root, specs, data_path):

    entries = {}
    complete = []
    for label, spec in specs.items():
        bundle_path = data_path / spec.label / BUNDLE_MANIFEST
        if bundle_path.is_file():
            manifest = json.loads(bundle_path.read_text(encoding="utf-8"))
            entries[label] = {
                "family": spec.family,
                "num_features": spec.num_features,
                "tasks": [source for source in spec.aliases],
                "folds": manifest["folds"],
                "bundle_manifest_sha256": sha256_file(bundle_path),
            }
            if manifest["folds"] == list(range(1, N_FOLDS + 1)):
                complete.append(label)

    payload = {
        "labels": list(specs),
        "label_count": len(specs),
        "job_count": len(specs) * N_FOLDS,
        "complete_labels": complete,
        "entries": entries,
    }

    path = data_path / BATCH_MANIFEST
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)
    return path


def build_parser():
    parser = argparse.ArgumentParser(description="Export shared-input MTT fold bundles for the 35-label migration")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--label", action="append", dest="labels")
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    export_labels(args.project_root, labels=args.labels, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
