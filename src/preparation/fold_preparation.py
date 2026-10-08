from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler

from prepared_inputs import read_native_matrices

FOLD_COUNT = 5
FOLD_SEED = 42
ATTEMPT_QUANTILE = 0.999
IQR_MULTIPLIER = 1.5

ROLE_COLUMN = "source_cv_role"
ROLES = ("train", "validation")

MANIFEST_NAME = "manifest.json"
FOLD_FILE_PATTERN = "fold_{fold}.csv"
TRANSFORM_ARTIFACT_PATTERN = "transform_fold_{fold}.joblib"
ALL_NULL_FILL = 0.0


# -------------------------
# Adaptive interaction outlier limits
# -------------------------

@dataclass(frozen=True)
class InteractionLimits:
    attempt_threshold: float
    time_lower: float
    time_upper: float
    unit: str
    attempt_quantile: float = ATTEMPT_QUANTILE
    iqr_multiplier: float = IQR_MULTIPLIER
    time_zero_is_valid: bool = False


def _finite_values(values: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    return numeric[np.isfinite(numeric)]


def fit_attempt_threshold(values: pd.Series, quantile: float = ATTEMPT_QUANTILE) -> float:
    finite = _finite_values(values)
    if finite.size == 0:
        raise ValueError("Attempt threshold population is empty; refusing to fit")
    return float(np.quantile(finite, quantile))


def apply_attempt_mask(values: pd.Series, threshold: float) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    return ~(numeric > threshold)


def fit_time_fences(
    values: pd.Series,
    multiplier: float = IQR_MULTIPLIER,
    zero_is_valid: bool = False,
) -> tuple[float, float]:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    applicable = numeric[np.isfinite(numeric) & (numeric >= 0 if zero_is_valid else numeric > 0)]
    if applicable.size == 0:
        raise ValueError("Time fence population has no valid positive times; refusing to fit")

    logs = np.log1p(applicable)
    q1 = float(np.quantile(logs, 0.25))
    q3 = float(np.quantile(logs, 0.75))
    iqr = q3 - q1
    if iqr == 0.0:
        constant = float(np.quantile(applicable, 0.25))
        return constant, constant

    lower = max(0.0, float(np.expm1(q1 - multiplier * iqr)))
    upper = float(np.expm1(q3 + multiplier * iqr))
    if not (np.isfinite(lower) and np.isfinite(upper) and lower <= upper):
        raise ValueError(f"Learned time fences are malformed: lower={lower} upper={upper}")

    return lower, upper


def apply_time_mask(values: pd.Series, lower: float, upper: float, zero_is_valid: bool = False) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    applicable = np.isfinite(numeric) & (numeric >= 0 if zero_is_valid else numeric > 0)
    in_bounds = (numeric >= lower) & (numeric <= upper)
    missing = np.isnan(numeric)
    return np.where(applicable, in_bounds, missing)


def fit_interaction_limits(
    attempt_values: pd.Series,
    time_values: pd.Series,
    attempt_mask: np.ndarray | None = None,
    unit: str = "seconds",
    attempt_quantile: float = ATTEMPT_QUANTILE,
    iqr_multiplier: float = IQR_MULTIPLIER,
    time_zero_is_valid: bool = False,
) -> InteractionLimits:
    threshold = fit_attempt_threshold(attempt_values, attempt_quantile)
    if attempt_mask is None:
        attempt_mask = apply_attempt_mask(attempt_values, threshold)
    else:
        attempt_mask = np.asarray(attempt_mask, dtype=bool)

    time_surviving = pd.Series(pd.to_numeric(time_values, errors="coerce").to_numpy(dtype=float)[attempt_mask])
    lower, upper = fit_time_fences(time_surviving, iqr_multiplier, zero_is_valid=time_zero_is_valid)

    return InteractionLimits(
        attempt_threshold=threshold,
        time_lower=lower,
        time_upper=upper,
        unit=unit,
        attempt_quantile=attempt_quantile,
        iqr_multiplier=iqr_multiplier,
        time_zero_is_valid=time_zero_is_valid,
    )


def _filter_interactions(frame, limits, attempt_column, time_column, zero_is_valid):
    keep = apply_attempt_mask(frame[attempt_column], limits.attempt_threshold)
    surviving = frame.loc[keep]
    time_keep = apply_time_mask(surviving[time_column], limits.time_lower, limits.time_upper, zero_is_valid=zero_is_valid)
    return surviving.loc[time_keep], int((~keep).sum()), int((~time_keep).sum())


def apply_interaction_limits(frame, limits, attempt_column, time_column):
    filtered, attempt_removed, time_removed = _filter_interactions(frame, limits, attempt_column, time_column, limits.time_zero_is_valid)
    accounting = {"attempt_threshold": limits.attempt_threshold, "attempt_removed": attempt_removed,
                  "time_lower": limits.time_lower, "time_upper": limits.time_upper,
                  "time_removed": time_removed, "retained": int(len(filtered))}
    return filtered.reset_index(drop=True), accounting


def assign_student_folds(students, n_folds: int = FOLD_COUNT, seed: int = FOLD_SEED) -> dict:
    students = sorted(students, key=str)
    if len(students) < n_folds:
        raise ValueError(f"Need at least {n_folds} students for {n_folds} folds, got {len(students)}")

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(len(students))
    fold_map = {}
    for position, index in enumerate(shuffled):
        fold_map[students[index]] = position % n_folds + 1
    return fold_map


def sample_students(students, fraction: float, seed: int = FOLD_SEED, student_count: int | None = None) -> list:
    students = sorted(students, key=str)
    if student_count is not None:
        if isinstance(student_count, bool) or not isinstance(student_count, (int, np.integer)):
            raise ValueError(f"student_count must be an integer or None, got {student_count!r}")
        if not 1 <= student_count <= len(students):
            raise ValueError(f"student_count must be in [1, {len(students)}], got {student_count}")
        keep_count = student_count
    elif fraction >= 1.0:
        return students
    else:
        keep_count = max(1, int(len(students) * fraction))

    rng = np.random.default_rng(seed)
    keep = np.sort(rng.choice(len(students), size=keep_count, replace=False))
    return [students[index] for index in keep]


def fold_role_students(fold_map: dict, fold: int) -> tuple[set, set]:
    validation_students = {student for student, assigned in fold_map.items() if assigned == fold}
    train_students = set(fold_map) - validation_students
    return train_students, validation_students



class InputTransform:
 
    IMPUTER_NAME = "IterativeImputer(random_state=42, keep_empty_features=True)"

    def __init__(self, feature_columns, all_null_columns, imputer, scaler):
        self.feature_columns = list(feature_columns)
        self.all_null_columns = list(all_null_columns)
        self.imputer = imputer
        self.scaler = scaler
        self.train_missing_columns: list = []
        self.last_report: dict = {}

    @classmethod
    def fit(cls, train_features: pd.DataFrame) -> "InputTransform":
        columns = list(train_features.columns)
        all_null = [column for column in columns if not train_features[column].notna().any()]
        train_missing = int(train_features.isna().sum().sum())

        imputer = IterativeImputer(random_state=42, keep_empty_features=True)
        imputed = imputer.fit_transform(train_features.to_numpy(dtype=float))
        scaler = StandardScaler().fit(imputed)

        transform = cls(columns, all_null, imputer, scaler)
        transform.train_missing_columns = [
            column for column in columns if train_features[column].isna().any()
        ]

        transform.last_report = {"missing_filled": train_missing}
        return transform

    def transform(self, features: pd.DataFrame, role: str) -> pd.DataFrame:
        missing = features.isna()
        missing_count = int(missing.sum().sum())

        values = self.imputer.transform(features.to_numpy(dtype=float))
        values = self.scaler.transform(values)

        self.last_report = {
            "role": role,
            "missing_filled": missing_count,
            "validation_only_missing_cells": int(
                sum(int(missing[column].sum()) for column in self.feature_columns if column not in self.train_missing_columns)
            ),
        }

        return pd.DataFrame(values, columns=self.feature_columns, index=features.index)

    def describe(self, train_features: pd.DataFrame) -> dict:
        initial_imputer = self.imputer.initial_imputer_
        return {
            "imputer": self.IMPUTER_NAME,
            "imputer_parameters": {
                "initial_strategy": initial_imputer.strategy,
                "initial_fill": [float(value) for value in initial_imputer.statistics_],
                "max_iter": self.imputer.max_iter,
                "keep_empty_features": bool(self.imputer.keep_empty_features),
                "random_state": int(self.imputer.random_state),
            },
            "scaler": "StandardScaler",
            "all_null_columns": list(self.all_null_columns),
            "all_null_fill": ALL_NULL_FILL,
            "train_missing_filled": int(train_features.isna().sum().sum()),
            "scaler_mean": [float(value) for value in self.scaler.mean_],
            "scaler_scale": [float(value) for value in self.scaler.scale_],
            "constant_features_after_transform": [
                column
                for column, variance in zip(self.feature_columns, self.scaler.var_)
                if variance == 0.0
            ],
        }


BARE_ID_COLUMNS = ("user_id", "selected_teaser_id", "assignment_id", "problem_id")


def discover_target_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column.startswith("target_")]


def discover_feature_columns(frame: pd.DataFrame, target_cols) -> list[str]:

    target_set = set(target_cols)
    return [
        column
        for column in frame.columns
        if column not in target_set
        and column != "split"
        and not column.startswith("source_")
        and column not in BARE_ID_COLUMNS
    ]


def _prepare_feature_frame(adapter, features: pd.DataFrame) -> pd.DataFrame:
    prepare_features = getattr(adapter, "prepare_features", None)
    prepared = features if prepare_features is None else prepare_features(features)
    if not isinstance(prepared, pd.DataFrame):
        raise ValueError("prepare_features must return a pandas DataFrame")
    if len(prepared) != len(features) or not prepared.index.equals(features.index):
        raise ValueError("prepare_features must preserve row count and index")
    if list(prepared.columns) != list(features.columns):
        raise ValueError("prepare_features must preserve feature column names and order")

    try:
        prepared.to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("prepare_features must return numeric-compatible data") from error

    return prepared


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_bundle_file(prepared_dir: Path, path: Path, description: str) -> Path:
    prepared = Path(prepared_dir).resolve(strict=False)
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Prepared {description} must be a regular non-symlink file: {path}")

    resolved = path.resolve(strict=True)
    if resolved.parent != prepared:
        raise ValueError(f"Prepared {description} escapes the bundle directory: {path}")
    return resolved


def fingerprint_paths(paths) -> list[dict]:
    fingerprints = []
    for path in paths:
        if path is None:
            continue
        path = Path(path)
        stat = path.stat()
        fingerprints.append({
            "path": str(path),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": _sha256_file(path),
        })
    return fingerprints


def validate_fold_frames(fold_frames: list[dict], fold_map: dict, fold_accounting: dict) -> None:
    if len(fold_frames) != FOLD_COUNT:
        raise ValueError(f"Expected exactly {FOLD_COUNT} folds, got {len(fold_frames)}")
    schemas = None

    validation_students_seen: set = set()
    for entry in fold_frames:
        fold = entry["fold"]
        frame = entry["frame"]

        roles = set(frame[ROLE_COLUMN].unique())
        if roles != set(ROLES):
            raise ValueError(f"Fold {fold} roles are malformed or incomplete: {sorted(roles)}")

        train = frame.loc[frame[ROLE_COLUMN].eq("train")]
        validation = frame.loc[frame[ROLE_COLUMN].eq("validation")]

        expected_train, expected_validation = fold_role_students(fold_map, fold)
        observed_train = set(train["source_student_key"].astype(str))
        observed_validation = set(validation["source_student_key"].astype(str))
        unexpected_train = observed_train - expected_train
        unexpected_validation = observed_validation - expected_validation
        if unexpected_train or unexpected_validation:
            unexpected = sorted(unexpected_train | unexpected_validation)[:5]
            raise ValueError(f"Fold {fold} contains students outside frozen fold map or expected role: {unexpected}")

        overlap = observed_validation & observed_train
        if overlap:
            raise ValueError(f"Fold {fold} train/validation group overlap: {sorted(map(str, overlap))[:5]}")

        revalidated = observed_validation & validation_students_seen
        if revalidated:
            raise ValueError(f"Students validated in multiple folds: {sorted(map(str, revalidated))[:5]}")
        validation_students_seen |= observed_validation

        accounting = fold_accounting[fold]
        expected_id_sets = {
            "retained_train_student_ids": observed_train,
            "retained_validation_student_ids": observed_validation,
            "removed_train_student_ids": expected_train - observed_train,
            "removed_validation_student_ids": expected_validation - observed_validation,
        }
        for key, expected in expected_id_sets.items():
            if set(map(str, accounting[key])) != expected:
                raise ValueError(f"Fold {fold} accounting {key} does not match retained identities")
        if accounting["removed_train_students"] < 0 or accounting["removed_validation_students"] < 0:
            raise ValueError(f"Fold {fold} has invalid negative student-removal accounting")

        schema = list(frame.columns)
        if schemas is None:
            schemas = schema
        elif schema != schemas:
            raise ValueError(f"Fold {fold} schema diverges from fold 1")

        target_cols = discover_target_columns(frame)
        if not target_cols:
            raise ValueError(f"Fold {fold} has no target columns")
        if not np.isfinite(frame[target_cols].to_numpy(dtype=float)).all():
            raise ValueError(f"Fold {fold} contains non-finite target values")


def _fold_column_order(frame: pd.DataFrame) -> list[str]:
    target_cols = discover_target_columns(frame)
    feature_cols = discover_feature_columns(frame, target_cols)

    feature_set = set(feature_cols)

    target_set = set(target_cols)
    metadata = [
        column
        for column in frame.columns
        if column not in feature_set and column not in target_set and column != ROLE_COLUMN
    ]
    return metadata + [ROLE_COLUMN] + feature_cols + target_cols


def validate_prepared_label(label: str) -> str:
    if (
        not isinstance(label, str)
        or not label
        or label in (".", "..")
        or "\0" in label
        or label != Path(label).name
        or "/" in label
        or "\\" in label
    ):
        raise ValueError(f"Prepared label is not a safe single path component: {label!r}")
    return label


def prepared_target_dir(prepared_root, label: str) -> Path:
    label = validate_prepared_label(label)

    root = Path(prepared_root).resolve(strict=False)
    target = (root / label).resolve(strict=False)
    if target.parent != root:
        raise ValueError(f"Prepared target {target} escapes prepared root {root}")
    return target


def _write_atomic(prepared_dir: Path, fold_frames: list[dict], manifest: dict, transforms: dict) -> None:
    prepared_dir = Path(prepared_dir).resolve(strict=False)
    prepared_root = prepared_dir.parent
    if prepared_target_dir(prepared_root, prepared_dir.name) != prepared_dir:
        raise ValueError(f"Prepared target is not a direct child of its root: {prepared_dir}")

    prepared_root.mkdir(parents=True, exist_ok=True)
    if prepared_dir.exists():
        raise FileExistsError(f"Prepared directory already exists: {prepared_dir}")

    temporary_dir = prepared_root / f".tmp_{prepared_dir.name}_{os.getpid()}"
    shutil.rmtree(temporary_dir, ignore_errors=True)

    try:
        temporary_dir.mkdir(parents=True)
        fold_integrity = {}
        for entry in fold_frames:
            csv_path = temporary_dir / FOLD_FILE_PATTERN.format(fold=entry["fold"])
            entry["frame"].to_csv(csv_path, index=False)
            fold_integrity[csv_path.name] = {"sha256": _sha256_file(csv_path), "rows": int(len(entry["frame"]))}

        transform_artifacts = {}
        for fold, transform in sorted(transforms.items()):
            artifact_path = temporary_dir / TRANSFORM_ARTIFACT_PATTERN.format(fold=fold)
            joblib.dump(transform, artifact_path)
            transform_artifacts[str(fold)] = {
                "file": artifact_path.name,
                "sha256": _sha256_file(artifact_path),
                "format": "joblib",
                "sklearn_version": sklearn.__version__,
                "feature_columns": list(transform.feature_columns),
            }

        manifest = {
            **manifest,
            "fold_file_integrity": fold_integrity,
            "transform_artifacts": transform_artifacts,
        }

        (temporary_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, sort_keys=False), encoding="utf-8")
        os.replace(temporary_dir, prepared_dir)
    finally:
        shutil.rmtree(temporary_dir, ignore_errors=True)


def load_prepared_transform(prepared_dir: Path, fold: int) -> "InputTransform":
    prepared_dir = Path(prepared_dir)
    manifest = read_manifest(prepared_dir)
    if "split" in manifest.get("schema", {}).get("columns", []):
        raise ValueError("Prepared manifests forbid the legacy split column")

    artifacts = manifest.get("transform_artifacts", {})
    artifact = artifacts.get(str(fold))
    if not artifact:
        raise ValueError(f"Manifest records no transform artifact for fold {fold}")
    expected_name = TRANSFORM_ARTIFACT_PATTERN.format(fold=fold)
    if artifact.get("file") != expected_name or Path(artifact.get("file", "")).name != expected_name:
        raise ValueError(f"Transform artifact path is unsafe for fold {fold}")

    path = _require_bundle_file(prepared_dir, prepared_dir / expected_name, f"transform for fold {fold}")
    if _sha256_file(path) != artifact["sha256"]:
        raise ValueError(f"Transform artifact for fold {fold} does not match the manifest integrity hash")

    transform = joblib.load(path)
    if list(transform.feature_columns) != list(manifest["schema"]["feature_columns"]):
        raise ValueError(f"Transform artifact for fold {fold} does not match the manifest schema")
    return transform


def read_manifest(prepared_dir: Path) -> dict:
    path = Path(prepared_dir) / MANIFEST_NAME

    path = _require_bundle_file(Path(prepared_dir), path, "manifest")
    return json.loads(path.read_text(encoding="utf-8"))


def read_prepared_folds(prepared_dir: Path) -> list[dict]:
    prepared_dir = Path(prepared_dir)
    manifest = read_manifest(prepared_dir)
    if manifest.get("protocol") == "native_materialized":
        return read_native_matrices(prepared_dir)

    n_folds = int(manifest["n_folds"])
    if n_folds != FOLD_COUNT:
        raise ValueError(
            f"Prepared artifacts must contain exactly {FOLD_COUNT} folds; manifest declares {n_folds}"
        )

    expected_files = {FOLD_FILE_PATTERN.format(fold=fold) for fold in range(1, n_folds + 1)}
    present_csvs = {path.name for path in prepared_dir.glob("*.csv")}
    if present_csvs != expected_files:
        raise ValueError(
            f"Prepared fold files do not match manifest; missing={sorted(expected_files - present_csvs)} "
            f"orphan={sorted(present_csvs - expected_files)}"
        )

    schema = manifest["schema"]
    declared_columns = list(schema["columns"])
    declared_features = list(schema["feature_columns"])
    declared_targets = list(schema["target_columns"])
    declared_metadata = list(schema["metadata_columns"])
    student_column = manifest.get("student_column")
    if not student_column or student_column not in declared_columns:
        raise ValueError(f"Manifest student_column {student_column!r} is missing from the declared schema")

    raw_fold_map = manifest["student_fold_map"]
    fold_map = {}
    for student, fold in raw_fold_map.items():
        if isinstance(fold, bool) or not isinstance(fold, int) or fold not in range(1, FOLD_COUNT + 1):
            raise ValueError(f"student_fold_map assigns {student!r} to invalid fold {fold!r}")
        fold_map[str(student)] = fold
    if not fold_map:
        raise ValueError("Manifest student_fold_map is empty")

    integrity = manifest["fold_file_integrity"]
    if set(integrity) != expected_files:
        raise ValueError("Manifest fold_file_integrity does not cover exactly the declared fold files")

    expected_artifact_names = {TRANSFORM_ARTIFACT_PATTERN.format(fold=fold) for fold in range(1, FOLD_COUNT + 1)}
    present_artifacts = {path.name for path in prepared_dir.glob("*.joblib")}

    artifacts = manifest.get("transform_artifacts", {})
    if present_artifacts != expected_artifact_names or set(artifacts) != {str(fold) for fold in range(1, FOLD_COUNT + 1)}:
        raise ValueError("Prepared transform artifacts are missing, orphaned or incomplete")

    feature_set = set(declared_features)
    if len(feature_set) != len(declared_features) or not declared_features:
        raise ValueError("Declared feature columns are empty or duplicated")

    target_set = set(declared_targets)
    if not declared_targets:
        raise ValueError("Declared target columns are empty")
    if feature_set & target_set:
        raise ValueError(f"Declared feature columns leak target columns: {sorted(feature_set & target_set)}")

    identity_columns = {ROLE_COLUMN, student_column} | set(BARE_ID_COLUMNS)
    if feature_set & identity_columns or any(column.startswith("source_") for column in declared_features):
        raise ValueError("Declared feature columns leak role/student identity columns")
    if "split" in declared_columns or "split" in declared_metadata:
        raise ValueError("Prepared bundles forbid the legacy split column")

    for fold in range(1, FOLD_COUNT + 1):
        artifact = artifacts[str(fold)]
        expected_name = TRANSFORM_ARTIFACT_PATTERN.format(fold=fold)
        if artifact.get("file") != expected_name or Path(artifact.get("file", "")).name != expected_name:
            raise ValueError(f"Transform artifact metadata is unsafe for fold {fold}")
        if artifact.get("format") != "joblib" or artifact.get("sklearn_version") != sklearn.__version__:
            raise ValueError(f"Transform artifact metadata is incoherent for fold {fold}")
        if artifact.get("feature_columns") != declared_features:
            raise ValueError(f"Transform artifact schema is incoherent for fold {fold}")
        artifact_path = _require_bundle_file(
            prepared_dir, prepared_dir / expected_name, f"transform for fold {fold}"
        )
        if _sha256_file(artifact_path) != artifact.get("sha256"):
            raise ValueError(f"Transform artifact for fold {fold} does not match the manifest integrity hash")

    folds = []

    validation_students_seen: set = set()

    for fold in range(1, n_folds + 1):
        path = _require_bundle_file(
            prepared_dir,
            prepared_dir / FOLD_FILE_PATTERN.format(fold=fold),
            f"CSV for fold {fold}",
        )
        if _sha256_file(path) != integrity[path.name]["sha256"]:
            raise ValueError(f"Fold {fold} CSV does not match the manifest integrity hash")

        frame = pd.read_csv(path, dtype={student_column: "string", ROLE_COLUMN: "string"}, keep_default_na=False)
        if int(integrity[path.name].get("rows", -1)) != len(frame):
            raise ValueError(f"Fold {fold} CSV row count does not match the manifest integrity record")
        if list(frame.columns) != declared_columns:
            raise ValueError(f"Fold {fold} schema does not match the manifest")

        target_cols = [column for column in frame.columns if column.startswith("target_")]
        if target_cols != declared_targets:
            raise ValueError(f"Fold {fold} target partition does not match independently derived targets")
        derived_features = discover_feature_columns(frame, target_cols)
        if derived_features != declared_features:
            raise ValueError(f"Fold {fold} feature partition does not match independently derived features")
        derived_metadata = [
            column
            for column in frame.columns
            if column not in target_set and column not in feature_set
        ]
        if derived_metadata != declared_metadata:
            raise ValueError(f"Fold {fold} metadata partition does not match independently derived metadata")

        roles = set(frame[ROLE_COLUMN].unique())
        if roles != set(ROLES):
            raise ValueError(f"Fold {fold} roles are malformed or incomplete: {sorted(roles)}")

        train = frame.loc[frame[ROLE_COLUMN].eq("train")].reset_index(drop=True)
        validation = frame.loc[frame[ROLE_COLUMN].eq("validation")].reset_index(drop=True)

        try:
            X_train = train[declared_features].to_numpy(dtype=float)
            X_validation = validation[declared_features].to_numpy(dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Fold {fold} features are not numeric: {error}") from error
        if not (np.isfinite(X_train).all() and np.isfinite(X_validation).all()):
            raise ValueError(f"Fold {fold} contains non-finite or missing feature values")

        try:
            y_train = train[declared_targets].to_numpy(dtype=float)
            y_validation = validation[declared_targets].to_numpy(dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Fold {fold} targets are not numeric: {error}") from error
        if not (np.isfinite(y_train).all() and np.isfinite(y_validation).all()):
            raise ValueError(f"Fold {fold} contains missing or non-finite target values")

        observed_train_students = set(train[student_column].astype(str))
        observed_validation_students = set(validation[student_column].astype(str))
        unknown = (observed_train_students | observed_validation_students) - set(fold_map)
        if unknown:
            raise ValueError(
                f"Fold {fold} contains students absent from student_fold_map: {sorted(unknown)[:5]}"
            )

        expected_validation_students = {student for student, assigned in fold_map.items() if assigned == fold}
        expected_train_students = set(fold_map) - expected_validation_students
        misplaced_train = observed_train_students - expected_train_students
        if misplaced_train:
            raise ValueError(
                f"Fold {fold} train role contains students not assigned as train: {sorted(misplaced_train)[:5]}"
            )
        misplaced_validation = observed_validation_students - expected_validation_students
        if misplaced_validation:
            raise ValueError(
                f"Fold {fold} validation role contains students not assigned as validation: "
                f"{sorted(misplaced_validation)[:5]}"
            )

        overlap = observed_train_students & observed_validation_students
        if overlap:
            raise ValueError(f"Fold {fold} train/validation student overlap: {sorted(overlap)[:5]}")

        revalidated = observed_validation_students & validation_students_seen
        if revalidated:
            raise ValueError(f"Students validated in multiple folds: {sorted(revalidated)[:5]}")
        validation_students_seen |= observed_validation_students

        accounting = manifest["fold_accounting"].get(str(fold))
        if accounting is None:
            raise ValueError(f"Manifest fold_accounting is missing fold {fold}")
        expected_counts = {
            "assigned_train_students": len(expected_train_students),
            "assigned_validation_students": len(expected_validation_students),
            "retained_train_students": len(observed_train_students),
            "retained_validation_students": len(observed_validation_students),
            "retained_train_rows": int(len(train)),
            "retained_validation_rows": int(len(validation)),
            "removed_train_students": len(expected_train_students) - len(observed_train_students),
            "removed_validation_students": len(expected_validation_students) - len(observed_validation_students),
        }
        for key, expected in expected_counts.items():
            if key not in accounting or int(accounting[key]) != expected:
                observed = accounting.get(key, "<missing>")
                raise ValueError(
                    f"Fold {fold} manifest accounting {key}={observed} does not match observed {expected}"
                )

        expected_ids = {
            "retained_train_student_ids": observed_train_students,
            "retained_validation_student_ids": observed_validation_students,
            "removed_train_student_ids": expected_train_students - observed_train_students,
            "removed_validation_student_ids": expected_validation_students - observed_validation_students,
        }
        for key, expected in expected_ids.items():
            if key not in accounting or set(map(str, accounting[key])) != expected:
                raise ValueError(f"Fold {fold} manifest {key} does not match observed student identities")

        folds.append({"fold": fold, "train_frame": train, "validation_frame": validation, "manifest": manifest})

    expected_retained_validation = {
        student
        for student, fold in fold_map.items()
        if student not in set(map(str, manifest["fold_accounting"][str(fold)]["removed_validation_student_ids"]))
    }
    if validation_students_seen != expected_retained_validation:
        raise ValueError("Mapped students are not validated exactly once or explicitly removed in their assigned fold")

    return folds


def make_interaction_filter(limits, attempt_column, time_column, time_zero_is_valid=None):
    zero_is_valid = limits.time_zero_is_valid if time_zero_is_valid is None else time_zero_is_valid

    def interaction_filter(frame):
        filtered, _, _ = _filter_interactions(frame, limits, attempt_column, time_column, zero_is_valid)
        return filtered.reset_index(drop=True)

    return interaction_filter


def _interaction_accounting(frame, limits, attempt_column, time_column, time_zero_is_valid):
    filtered, attempt_removed, time_removed = _filter_interactions(frame, limits, attempt_column, time_column, time_zero_is_valid)
    return {"attempt_threshold": limits.attempt_threshold, "attempt_removed": attempt_removed,
            "time_lower": limits.time_lower, "time_upper": limits.time_upper, "time_removed": time_removed,
            "retained_interactions": int(len(filtered)), "observed_interactions": int(len(frame))}


def validate_preparation_parameters(
    n_folds: int,
    seed: int,
    fraction: float,
    attempt_quantile: float,
    iqr_multiplier: float,
    student_count: int | None = None,
) -> None:
    if isinstance(n_folds, bool) or not isinstance(n_folds, (int, np.integer)) or n_folds != FOLD_COUNT:
        raise ValueError(f"n_folds must be the literal integer {FOLD_COUNT}, got {n_folds!r}")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError(f"seed must be an integer, got {seed}")

    try:
        fraction = float(fraction)
        attempt_quantile = float(attempt_quantile)
        iqr_multiplier = float(iqr_multiplier)
    except (TypeError, ValueError) as error:
        raise ValueError("fraction, attempt_quantile and iqr_multiplier must be numeric") from error
    if not np.isfinite(fraction) or not (0.0 < fraction <= 1.0):
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    if not np.isfinite(attempt_quantile) or not (0.0 < attempt_quantile <= 1.0):
        raise ValueError(f"attempt_quantile must be in (0, 1], got {attempt_quantile}")
    if not np.isfinite(iqr_multiplier) or iqr_multiplier <= 0.0:
        raise ValueError(f"iqr_multiplier must be positive, got {iqr_multiplier}")
    if student_count is not None and (isinstance(student_count, bool) or not isinstance(student_count, (int, np.integer)) or student_count <= 0):
        raise ValueError(f"student_count must be a positive integer or None, got {student_count!r}")


def prepare_dataset_folds(
    adapter,
    prepared_root,
    fraction: float = 1.0,
    n_folds: int = FOLD_COUNT,
    seed: int = FOLD_SEED,
    attempt_quantile: float = ATTEMPT_QUANTILE,
    iqr_multiplier: float = IQR_MULTIPLIER,
    workers: int = 1,
    student_count: int | None = None,
    log=print,
) -> Path:
    validate_preparation_parameters(n_folds, seed, fraction, attempt_quantile, iqr_multiplier, student_count)
    if isinstance(workers, bool) or not isinstance(workers, (int, np.integer)) or workers <= 0:
        raise ValueError(f"workers must be a positive integer, got {workers!r}")

    adapter.workers = int(workers)
    prepared_root = Path(prepared_root)
    target_dir = prepared_target_dir(prepared_root, adapter.label)
    if target_dir.exists():
        raise FileExistsError(f"Prepared directory already exists: {target_dir}")

    context = adapter.load()
    provisional_students = adapter.provisional_students(context)
    if not provisional_students:
        raise ValueError("Provisional cohort is empty")

    students = sample_students(provisional_students, fraction, seed=seed, student_count=student_count)
    fold_map = {str(student): fold for student, fold in assign_student_folds(students, n_folds=n_folds, seed=seed).items()}
    log(f"[{adapter.label}] cohort={len(students)} folds={n_folds} fraction={fraction} student_count={student_count}")

    artifact_mode = adapter.adapter_kind == "artifact"
    interactions = None if artifact_mode else adapter.interactions(context)
    interaction_student_column = adapter.interaction_student_column
    student_column = adapter.student_column
    time_zero_is_valid = bool(getattr(adapter, "time_zero_is_valid", False))

    fold_frames = []
    fold_accounting = {}
    transform_provenance = {}
    transforms = {}

    for fold in range(1, n_folds + 1):
        train_students, validation_students = fold_role_students(fold_map, fold)
        fold_accounting[fold] = {
            "assigned_train_students": len(train_students),
            "assigned_validation_students": len(validation_students),
        }

        limits = None
        if not artifact_mode:
            train_interactions = interactions.loc[interactions[interaction_student_column].astype(str).isin(map(str, train_students))]
            validation_interactions = interactions.loc[interactions[interaction_student_column].astype(str).isin(map(str, validation_students))]
            if train_interactions.empty:
                raise ValueError(f"Fold {fold} has no training interactions to fit limits on")

            limits = fit_interaction_limits(
                train_interactions[adapter.attempt_column],
                train_interactions[adapter.time_column],
                unit=adapter.time_unit,
                attempt_quantile=attempt_quantile,
                iqr_multiplier=iqr_multiplier,
                time_zero_is_valid=time_zero_is_valid,
            )

            filter_fn = make_interaction_filter(
                limits, adapter.attempt_column, adapter.time_column,
                time_zero_is_valid=time_zero_is_valid,
            )

            fold_accounting[fold]["train_interactions"] = _interaction_accounting(
                train_interactions, limits, adapter.attempt_column, adapter.time_column, time_zero_is_valid
            )
            fold_accounting[fold]["validation_interactions"] = _interaction_accounting(
                validation_interactions, limits, adapter.attempt_column, adapter.time_column, time_zero_is_valid
            )
        else:
            filter_fn = None

        train_rows = adapter.build_rows(context, train_students, filter_fn)
        validation_rows = adapter.build_rows(context, validation_students, filter_fn)
        if train_rows.empty or validation_rows.empty:
            raise ValueError(f"Fold {fold} produced empty train or validation rows")

        train_rows = train_rows.copy()
        validation_rows = validation_rows.copy()
        if "split" in train_rows.columns or "split" in validation_rows.columns:
            raise ValueError("Prepared folds forbid the legacy split column")
        train_rows[ROLE_COLUMN] = "train"
        validation_rows[ROLE_COLUMN] = "validation"

        target_cols = discover_target_columns(train_rows)
        feature_cols = discover_feature_columns(train_rows, target_cols)
        if not feature_cols or not target_cols:
            raise ValueError(f"Fold {fold} feature/target discovery failed")

        train_input = _prepare_feature_frame(adapter, train_rows[feature_cols])
        validation_input = _prepare_feature_frame(adapter, validation_rows[feature_cols])
        transform = InputTransform.fit(train_input)
        train_features = transform.transform(train_input, "train")
        validation_features = transform.transform(validation_input, "validation")

        transforms[fold] = transform
        transform_provenance[fold] = {
            **transform.describe(train_input),
            "fixed_log1p_columns": list(getattr(adapter, "log1p_columns", ())),
            "validation_missing_filled": transform.last_report["missing_filled"],
            "validation_only_missing_cells": transform.last_report["validation_only_missing_cells"],
        }

        train_output = pd.concat(
            [
                train_rows.drop(columns=feature_cols).reset_index(drop=True),
                train_features.reset_index(drop=True),
            ],
            axis=1,
        )
        validation_output = pd.concat(
            [
                validation_rows.drop(columns=feature_cols).reset_index(drop=True),
                validation_features.reset_index(drop=True),
            ],
            axis=1,
        )
        frame = pd.concat([train_output, validation_output], ignore_index=True)
        frame = frame[_fold_column_order(train_rows)]
        frame["source_student_key"] = frame[student_column].astype(str)

        retained_train = set(train_rows[student_column].astype(str))
        retained_validation = set(validation_rows[student_column].astype(str))
        fold_accounting[fold].update({
            "retained_train_students": len(retained_train),
            "retained_validation_students": len(retained_validation),
            "retained_train_rows": int(len(train_rows)),
            "retained_validation_rows": int(len(validation_rows)),
            "removed_train_students": len(train_students) - len(retained_train),
            "removed_validation_students": len(validation_students) - len(retained_validation),
            "retained_train_student_ids": sorted(retained_train),
            "retained_validation_student_ids": sorted(retained_validation),
            "removed_train_student_ids": sorted(set(map(str, train_students)) - retained_train),
            "removed_validation_student_ids": sorted(set(map(str, validation_students)) - retained_validation),
        })

        fold_frames.append({"fold": fold, "frame": frame})
        log(f"[{adapter.label}] fold {fold}: train_rows={len(train_rows)} validation_rows={len(validation_rows)}")

    validate_fold_frames(fold_frames, fold_map, fold_accounting)
    for entry in fold_frames:
        entry["frame"] = entry["frame"].drop(columns=["source_student_key"])

    prepared_dir = target_dir
    manifest = {
        "label": adapter.label,
        "adapter_kind": adapter.adapter_kind,
        "n_folds": n_folds,
        "adapter_filter_note": adapter.filter_description,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            "n_folds": n_folds,
            "seed": seed,
            "fraction": fraction,
            "student_count": None if student_count is None else int(student_count),
            "attempt_quantile": attempt_quantile,
            "iqr_multiplier": iqr_multiplier,
            "time_unit": None if artifact_mode else adapter.time_unit,
            "attempt_column": None if artifact_mode else adapter.attempt_column,
            "time_column": None if artifact_mode else adapter.time_column,
            "time_zero_is_valid": None if artifact_mode else time_zero_is_valid,
            "workers": int(workers),
        },
        "source_fingerprints": fingerprint_paths(adapter.source_paths()),
        "student_fold_map": {str(student): int(fold) for student, fold in fold_map.items()},
        "student_column": student_column,
        "fold_accounting": fold_accounting,
        "transform_provenance": transform_provenance,
        "schema": _manifest_schema(fold_frames[0]["frame"]),
    }

    _write_atomic(prepared_dir, fold_frames, manifest, transforms)
    log(f"[{adapter.label}] wrote {prepared_dir}")

    return prepared_dir


class BasePreparedAdapter:
    label: str
    adapter_kind = "interaction"
    student_column: str
    interaction_student_column: str
    attempt_column: str | None
    time_column: str | None
    time_unit: str | None
    filter_description: str
    time_zero_is_valid = False

    def source_paths(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

    def provisional_students(self, context):
        return context["provisional_students"]

    def interactions(self, context):
        raise NotImplementedError

    def build_rows(self, context, students, interaction_filter):
        raise NotImplementedError


def _manifest_schema(frame: pd.DataFrame) -> dict:
    target_cols = discover_target_columns(frame)
    feature_cols = discover_feature_columns(frame, target_cols)

    feature_set = set(feature_cols)

    target_set = set(target_cols)
    metadata = [column for column in frame.columns if column not in feature_set and column not in target_set]
    return {
        "columns": list(frame.columns),
        "metadata_columns": metadata,
        "feature_columns": feature_cols,
        "target_columns": target_cols,
        "role_column": ROLE_COLUMN,
    }
