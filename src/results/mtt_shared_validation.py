from __future__ import annotations
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from mtt_shared_registry import N_FOLDS

EASYDICT_TAG = "tag:yaml.org,2002:python/object/new:easydict.EasyDict"


class _LightningHparamsLoader(yaml.SafeLoader):
    pass


def _construct_easydict(loader, node):
    if not isinstance(node, yaml.MappingNode):
        raise yaml.constructor.ConstructorError(
            None, None, "EasyDict hparams node must be a mapping", node.start_mark
        )

    mapping = loader.construct_mapping(node, deep=True)
    items = mapping.get("dictitems")
    if not isinstance(items, dict):
        raise yaml.constructor.ConstructorError(
            None, None, "EasyDict hparams node missing dictitems mapping", node.start_mark
        )

    return items


def load_hparams(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing MTT-CV hparams: {path}")
    return yaml.load(path.read_text(encoding="utf-8"), Loader=_LightningHparamsLoader)

_LightningHparamsLoader.add_constructor(EASYDICT_TAG, _construct_easydict)


MODEL_NAME = "mtt"
COLUMNS = ["dataset", "target", "task_type", "metric", "model", "fold", "value"]

METRICS_BY_TYPE = {
    "binary": ("AUC", "Accuracy", "F1"),
    "regression": ("MAE", "RMSE", "R2"),
}

SOURCE_METRICS = {
    "BinaryAUROC": ("AUC", "binary"),
    "BinaryAccuracy": ("Accuracy", "binary"),
    "BinaryF1Score": ("F1", "binary"),
    "MeanAbsoluteError": ("MAE", "regression"),
    "MeanSquaredError": ("RMSE", "regression"),
    "R2Score": ("R2", "regression"),
}


def fold_rows(spec):
    return sum(len(METRICS_BY_TYPE[kind]) for kind in spec.target_types.values())


def suffix_rows(spec):
    return fold_rows(spec) * N_FOLDS


def expected_metric_keys(spec, folds=range(1, N_FOLDS + 1)):
    keys = set()
    for source, output, kind in [(s, spec.aliases[s], spec.target_types[s]) for s in spec.aliases]:
        for metric in METRICS_BY_TYPE[kind]:
            for fold in folds:
                keys.add((spec.result_label, output, kind, metric, MODEL_NAME, int(fold)))
    return keys


def frame_metric_keys(frame):
    return {
        (row.dataset, row.target, row.task_type, row.metric, row.model, int(row.fold))
        for row in frame.itertuples(index=False)
    }


def validate_metric_frame(spec, frame, expected_rows=None):
    if expected_rows is not None and len(frame) != expected_rows:
        raise ValueError(f"{spec.label}: expected {expected_rows} metric rows, got {len(frame)}")
    if frame["dataset"].nunique() != 1 or set(frame["dataset"]) != {spec.result_label}:
        raise ValueError(f"{spec.label}: metric dataset column mismatch")

    keys = frame_metric_keys(frame)
    expected = expected_metric_keys(spec)
    if keys != expected:
        missing, extra = expected - keys, keys - expected
        raise ValueError(f"{spec.label}: metric key mismatch missing={sorted(missing)[:4]} extra={sorted(extra)[:4]}")

    values = pd.to_numeric(frame["value"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{spec.label}: nonfinite metric value")

    return frame


def _require_int(value, key):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"MTT hparams mismatch for {key}: {value!r}")
    return value


def expected_dataset_data(spec, fold):
    return {
        "format": "h5",
        "name": f"{spec.label}_fold_{fold}",
        "short_name": spec.short_name,
        "path": f"/{spec.label}/fold_{fold}/",
        "tasks": [source for source in spec.aliases],
        "task_type": {source: spec.target_types[source] for source in spec.aliases},
        "task_out_dim": {source: 1 for source in spec.aliases},
        "num_features": spec.num_features,
        "categorical_cardinalities": [],
    }


def verify_hparams(spec, path, fold, data_root_path=None, config_root_path=None, row_counts=None):
    data = load_hparams(path)
    identity = {'dataset': f'{spec.label}/fold_{fold}', 'run_name': f'{spec.label}_fold_{fold}'}
    if not isinstance(data, dict) or any(data.get(key) != value for key, value in identity.items()):
        raise ValueError('MTT logs belong to another dataset or fold')

    for key, expected in {'data_root': data_root_path, 'config_root': config_root_path}.items():
        recorded = data.get(key)
        if recorded is not None and expected is not None and Path(recorded).resolve() != Path(expected).resolve():
            raise ValueError(f'MTT logs reference another {key}')

    if spec.profile == 'higgs50k':
        if data.get('mtt_protocol') != 'selection-refit':
            raise ValueError('HIGGS must use selection-refit')
        if row_counts is None:
            row_counts = {'train': 32000, 'val': 8000, 'test': 10000}
        if row_counts != {'train': 32000, 'val': 8000, 'test': 10000}:
            raise ValueError('HIGGS row counts mismatch')

    best_epoch = _require_int(data.get('mtt_selection_best_epoch'), 'mtt_selection_best_epoch')
    refit_epochs = _require_int(data.get('mtt_refit_epochs'), 'mtt_refit_epochs')
    epochs = _require_int(data['training']['epochs'], 'epochs')
    if not 0 <= best_epoch < epochs or refit_epochs != best_epoch + 1:
        raise ValueError('MTT selection/refit epoch accounting mismatch')

    if row_counts is not None:
        expected_counts = {'mtt_n_train': row_counts['train'], 'mtt_n_val': row_counts['val'],
                           'mtt_n_test': row_counts['test'], 'mtt_n_refit': row_counts['train'] + row_counts['val']}
        if any(_require_int(data.get(key), key) != value for key, value in expected_counts.items()):
            raise ValueError('MTT logged populations differ from the prepared folds')
    return data


def validate_fold_metrics(spec, path, fold):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing fold metrics: {path}")

    table = pd.read_csv(path)
    test_columns = [column for column in table.columns if column.startswith("test_")]
    if not test_columns:
        raise ValueError(f"{path}: missing test metrics")

    completed = table.loc[table[test_columns].notna().any(axis=1)]
    if len(completed) != 1:
        raise ValueError(f"{path}: expected exactly one completed test row, found {len(completed)}")

    row = completed.iloc[0]
    records, seen = [], set()

    for column in test_columns:
        value = row[column]
        if pd.isna(value):
            continue

        matched = False
        for suffix, (metric, kind) in SOURCE_METRICS.items():
            marker = "_" + suffix
            if not column.endswith(marker):
                continue

            source = column[5:-len(marker)]
            if source not in spec.aliases or spec.target_types[source] != kind:
                raise ValueError(f"{path}: unknown or mistyped source target {source!r}")

            number = float(value)
            if not math.isfinite(number):
                raise ValueError(f"{path}: nonfinite source value {column}")
            if suffix == "MeanSquaredError":
                if number < 0:
                    raise ValueError(f"{path}: negative MSE")
                number = math.sqrt(number)

            key = (spec.aliases[source], kind, metric)
            if key in seen:
                raise ValueError(f"{path}: duplicate target/metric {key}")

            seen.add(key)
            records.append({
                "dataset": spec.result_label, "target": spec.aliases[source], "task_type": kind,
                "metric": metric, "model": MODEL_NAME, "fold": int(fold), "value": number,
            })
            matched = True
            break

        if not matched and not _allowed_diagnostic(spec, column):
            raise ValueError(f"{path}: unknown filled test diagnostic {column}")

    expected = {
        (spec.aliases[source], kind, metric)
        for source in spec.aliases
        for kind in (spec.target_types[source],)
        for metric in METRICS_BY_TYPE[kind]
    }
    if seen != expected or len(records) != fold_rows(spec):
        raise ValueError(f"{path}: incomplete or extra test metrics")

    return records


def _allowed_diagnostic(spec, column):
    if column == "test_total_loss":
        return True
    return any(
        column in (f"test_{source}_loss", f"test_{source}_ExplainedVariance")
        for source in spec.aliases
    )
