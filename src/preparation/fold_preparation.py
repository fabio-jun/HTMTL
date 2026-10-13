from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler

from prepared_inputs import read_native_matrices, native_digest

FOLD_COUNT = 5
FOLD_SEED = 42
ATTEMPT_QUANTILE = 0.999
IQR_MULTIPLIER = 1.5

ROLE_COLUMN = "source_cv_role"
ROLES = ("train", "validation")

MANIFEST_NAME = "manifest.json"
FOLD_FILE_PATTERN = "fold_{fold}.csv"


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

    logs = np.log1p(applicable)
    q1 = float(np.quantile(logs, 0.25))
    q3 = float(np.quantile(logs, 0.75))
    iqr = q3 - q1
    if iqr == 0.0:
        constant = float(np.quantile(applicable, 0.25))
        return constant, constant

    lower = max(0.0, float(np.expm1(q1 - multiplier * iqr)))
    upper = float(np.expm1(q3 + multiplier * iqr))

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


def assign_student_folds(students, n_folds: int = FOLD_COUNT, seed: int = FOLD_SEED) -> dict:
    students = sorted(students, key=str)

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(len(students))
    fold_map = {}
    for position, index in enumerate(shuffled):
        fold_map[students[index]] = position % n_folds + 1
    return fold_map


def sample_students(students, fraction: float, seed: int = FOLD_SEED, student_count: int | None = None) -> list:
    students = sorted(students, key=str)
    if student_count is not None:
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
    def __init__(self, feature_columns, imputer, scaler):
        self.feature_columns = list(feature_columns)
        self.imputer = imputer
        self.scaler = scaler

    @classmethod
    def fit(cls, train_features: pd.DataFrame) -> "InputTransform":
        imputer = IterativeImputer(random_state=42, keep_empty_features=True)
        imputed = imputer.fit_transform(train_features.to_numpy(dtype=float))
        scaler = StandardScaler().fit(imputed)
        return cls(train_features.columns, imputer, scaler)

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        values = self.imputer.transform(features.to_numpy(dtype=float))
        values = self.scaler.transform(values)
        return pd.DataFrame(values, columns=self.feature_columns, index=features.index)


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
    return features if prepare_features is None else prepare_features(features)


def _require_bundle_file(prepared_dir: Path, path: Path, description: str) -> Path:
    prepared = Path(prepared_dir).resolve(strict=False)
    path = Path(path)
    resolved = path.resolve(strict=True)
    if resolved.parent != prepared:
        raise ValueError(f"Prepared {description} escapes the bundle directory: {path}")
    return resolved


def validate_fold_frames(fold_frames: list[dict], fold_map: dict) -> None:
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

        revalidated = observed_validation & validation_students_seen
        if revalidated:
            raise ValueError(f"Students validated in multiple folds: {sorted(map(str, revalidated))[:5]}")
        validation_students_seen |= observed_validation

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


def _write_atomic(prepared_dir: Path, fold_frames: list[dict]) -> None:
    prepared_dir = Path(prepared_dir).resolve(strict=False)
    prepared_root = prepared_dir.parent
    prepared_root.mkdir(parents=True, exist_ok=True)
    if prepared_dir.exists():
        raise FileExistsError(f"Prepared directory already exists: {prepared_dir}")
    temporary_dir = prepared_root / f".tmp_{prepared_dir.name}_{os.getpid()}"
    shutil.rmtree(temporary_dir, ignore_errors=True)
    try:
        temporary_dir.mkdir(parents=True)
        for entry in fold_frames:
            entry["frame"].to_csv(temporary_dir / FOLD_FILE_PATTERN.format(fold=entry["fold"]), index=False)
        os.replace(temporary_dir, prepared_dir)
    finally:
        shutil.rmtree(temporary_dir, ignore_errors=True)


def read_manifest(prepared_dir: Path) -> dict:
    path = Path(prepared_dir) / MANIFEST_NAME

    path = _require_bundle_file(Path(prepared_dir), path, "manifest")
    return json.loads(path.read_text(encoding="utf-8"))


def prepared_source_files(prepared_dir):
    prepared_dir = Path(prepared_dir)
    if (prepared_dir / MANIFEST_NAME).exists():
        return [prepared_dir / MANIFEST_NAME]
    return [prepared_dir / FOLD_FILE_PATTERN.format(fold=fold) for fold in range(1, FOLD_COUNT + 1)]


def read_prepared_metadata(prepared_dir):
    import csv

    prepared_dir = Path(prepared_dir)
    legacy = read_manifest(prepared_dir) if (prepared_dir / MANIFEST_NAME).exists() else {}
    if legacy.get('protocol') == 'native_materialized':
        return legacy

    if prepared_dir.is_symlink() or any(parent.is_symlink() for parent in prepared_dir.parents):
        raise ValueError('Prepared bundle must not use symlinks')

    expected = {FOLD_FILE_PATTERN.format(fold=fold) for fold in range(1, FOLD_COUNT + 1)}
    if {path.name for path in prepared_dir.glob('*.csv')} != expected:
        raise ValueError('Prepared bundle must contain exactly five fold CSVs')

    path = _require_bundle_file(prepared_dir, prepared_dir / FOLD_FILE_PATTERN.format(fold=1), 'CSV for fold 1')
    with path.open(newline='') as handle:
        columns = next(csv.reader(handle), [])
    if not columns or len(columns) != len(set(columns)) or 'split' in columns:
        raise ValueError('Prepared CSV header is empty, duplicated or contains legacy split')

    student_column = next((name for name in ('source_user_id', 'source_student_id', 'user_id') if name in columns), None)
    if student_column is None or ROLE_COLUMN not in columns:
        raise ValueError('Prepared CSV requires student identity and source_cv_role')

    schema = _manifest_schema(pd.DataFrame(columns=pd.Index(columns)))
    if not schema['feature_columns'] or not schema['target_columns']:
        raise ValueError('Prepared CSV requires features and targets')

    label = validate_prepared_label(prepared_dir.name)
    if legacy and (legacy.get('schema') != schema or legacy.get('student_column') != student_column):
        raise ValueError('Prepared manifest schema differs from independently derived CSV schema')
    if legacy and (legacy.get('label') != label or legacy.get('n_folds') != FOLD_COUNT):
        raise ValueError('Prepared manifest label/fold count mismatch')

    return {**legacy, 'protocol': 'csv_prepared', 'label': label,
            'n_folds': FOLD_COUNT, 'student_column': student_column, 'schema': schema}


def read_csv_prepared_folds(prepared_dir):
    import csv

    prepared_dir = Path(prepared_dir)
    metadata = read_prepared_metadata(prepared_dir)
    schema = metadata['schema']
    student_column = metadata['student_column']

    integrity = metadata.get('fold_file_integrity')
    expected_files = {FOLD_FILE_PATTERN.format(fold=fold) for fold in range(1, FOLD_COUNT + 1)}
    if integrity is not None and set(integrity) != expected_files:
        raise ValueError('Prepared manifest integrity must cover five fold CSVs')

    frozen = metadata.get('student_fold_map')
    if frozen is not None:
        if not frozen or any(isinstance(fold, bool) or not isinstance(fold, int) or fold not in range(1, FOLD_COUNT + 1)
                             for fold in frozen.values()):
            raise ValueError('Prepared frozen student map is empty or invalid')
        frozen = {str(student): fold for student, fold in frozen.items()}

    folds, validated, observed = [], {}, set()
    for fold in range(1, FOLD_COUNT + 1):
        path = _require_bundle_file(prepared_dir, prepared_dir / FOLD_FILE_PATTERN.format(fold=fold), f'CSV for fold {fold}')
        if integrity is not None and native_digest(path) != integrity[path.name].get('sha256'):
            raise ValueError(f'Fold {fold} CSV integrity mismatch')

        with path.open(newline='') as handle:
            columns = next(csv.reader(handle), [])
        if columns != schema['columns']:
            raise ValueError(f'Fold {fold} schema differs from fold 1')

        frame = pd.read_csv(path, dtype={student_column: 'string', ROLE_COLUMN: 'string'}, keep_default_na=False)
        if integrity is not None and len(frame) != integrity[path.name].get('rows'):
            raise ValueError(f'Fold {fold} CSV integrity row count mismatch')
        if frame.empty or frame[student_column].str.strip().eq('').any():
            raise ValueError(f'Fold {fold} has empty rows or student identifiers')
        if set(frame[ROLE_COLUMN].unique()) != set(ROLES):
            raise ValueError(f'Fold {fold} requires train and validation roles')

        try:
            values = frame[schema['feature_columns'] + schema['target_columns']].to_numpy(dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError(f'Fold {fold} features/targets must be numeric') from error
        if not np.isfinite(values).all():
            raise ValueError(f'Fold {fold} features/targets must be finite')

        train = frame.loc[frame[ROLE_COLUMN].eq('train')].reset_index(drop=True)
        validation = frame.loc[frame[ROLE_COLUMN].eq('validation')].reset_index(drop=True)
        train_students = set(train[student_column])
        validation_students = set(validation[student_column])

        if frozen is not None:
            expected_validation = {student for student, assigned in frozen.items() if assigned == fold}
            if train_students - (set(frozen) - expected_validation) or validation_students - expected_validation:
                raise ValueError(f'Fold {fold} student assignment differs from frozen manifest map')
        if train_students & validation_students:
            raise ValueError(f'Fold {fold} train/validation student overlap')
        if validation_students & set(validated):
            raise ValueError('Students validated in multiple folds')

        validated.update({student: fold for student in validation_students})
        observed |= train_students | validation_students
        folds.append({'fold': fold, 'train_frame': train, 'validation_frame': validation, 'manifest': metadata})

    if frozen is not None:
        accounting = metadata.get('fold_accounting', {})
        expected = {student for student, fold in frozen.items()
                    if student not in set(map(str, accounting.get(str(fold), {}).get('removed_validation_student_ids', [])))}
        if set(validated) != expected:
            raise ValueError('Frozen students must be validated once or explicitly removed')

    metadata.setdefault('student_fold_map', validated)
    metadata['observed_student_count'] = len(observed)
    return folds


def read_prepared_folds(prepared_dir: Path) -> list[dict]:
    prepared_dir = Path(prepared_dir)
    if (prepared_dir / MANIFEST_NAME).exists():
        manifest = read_manifest(prepared_dir)
        if manifest.get("protocol") == "native_materialized":
            return read_native_matrices(prepared_dir)
    return read_csv_prepared_folds(prepared_dir)


def make_interaction_filter(limits, attempt_column, time_column, time_zero_is_valid=None):
    zero_is_valid = limits.time_zero_is_valid if time_zero_is_valid is None else time_zero_is_valid

    def interaction_filter(frame):
        filtered, _, _ = _filter_interactions(frame, limits, attempt_column, time_column, zero_is_valid)
        return filtered.reset_index(drop=True)

    return interaction_filter


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
    target_dir = prepared_target_dir(Path(prepared_root), adapter.label)
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
    student_column = adapter.student_column
    time_zero_is_valid = bool(getattr(adapter, "time_zero_is_valid", False))

    fold_frames = []
    for fold in range(1, n_folds + 1):
        train_students, validation_students = fold_role_students(fold_map, fold)
        filter_fn = None
        if not artifact_mode:
            train_interactions = interactions.loc[interactions[adapter.interaction_student_column].astype(str).isin(map(str, train_students))]
            if train_interactions.empty:
                raise ValueError(f"Fold {fold} has no training interactions to fit limits on")
            limits = fit_interaction_limits(
                train_interactions[adapter.attempt_column], train_interactions[adapter.time_column],
                unit=adapter.time_unit, attempt_quantile=attempt_quantile,
                iqr_multiplier=iqr_multiplier, time_zero_is_valid=time_zero_is_valid,
            )
            filter_fn = make_interaction_filter(
                limits, adapter.attempt_column, adapter.time_column, time_zero_is_valid=time_zero_is_valid,
            )

        train_rows = adapter.build_rows(context, train_students, filter_fn)
        validation_rows = adapter.build_rows(context, validation_students, filter_fn)
        if train_rows.empty or validation_rows.empty:
            raise ValueError(f"Fold {fold} produced empty train or validation rows")

        train_rows, validation_rows = train_rows.copy(), validation_rows.copy()
        if "split" in train_rows.columns or "split" in validation_rows.columns:
            raise ValueError("Prepared folds forbid the legacy split column")
        train_rows[ROLE_COLUMN], validation_rows[ROLE_COLUMN] = "train", "validation"
        target_cols = discover_target_columns(train_rows)
        feature_cols = discover_feature_columns(train_rows, target_cols)
        if not feature_cols or not target_cols:
            raise ValueError(f"Fold {fold} feature/target discovery failed")

        train_input = _prepare_feature_frame(adapter, train_rows[feature_cols])
        validation_input = _prepare_feature_frame(adapter, validation_rows[feature_cols])
        transform = InputTransform.fit(train_input)
        train_features = transform.transform(train_input)
        validation_features = transform.transform(validation_input)

        train_output = pd.concat([train_rows.drop(columns=feature_cols).reset_index(drop=True),
                                  train_features.reset_index(drop=True)], axis=1)
        validation_output = pd.concat([validation_rows.drop(columns=feature_cols).reset_index(drop=True),
                                       validation_features.reset_index(drop=True)], axis=1)
        frame = pd.concat([train_output, validation_output], ignore_index=True)
        frame = frame[_fold_column_order(train_rows)]
        frame["source_student_key"] = frame[student_column].astype(str)
        fold_frames.append({"fold": fold, "frame": frame})
        log(f"[{adapter.label}] fold {fold}: train_rows={len(train_rows)} validation_rows={len(validation_rows)}")

    validate_fold_frames(fold_frames, fold_map)
    for entry in fold_frames:
        entry["frame"] = entry["frame"].drop(columns=["source_student_key"])

    _write_atomic(target_dir, fold_frames)
    log(f"[{adapter.label}] wrote {target_dir}")
    return target_dir


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
