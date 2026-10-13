import math
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from dataset_registry import checked_path, regular_tree


class MetricsCollector:

    COLUMNS = ["dataset","target","task_type","metric","model","fold","value",]

    def __init__(self, target_cols, output_types):
        if len(target_cols) != len(output_types):
            raise ValueError("Target_cols and output_types must have the same length")

        self.target_cols = list(target_cols)
        self.output_types = list(output_types)
        self._records: list[dict[str, object]] = []

    def add_fold(self, model_name, fold, y_original, predictions):
        metrics = compute_fold_metrics(y_original, predictions,self.target_cols,self.output_types,)

        for (task_type, metric, target), value in metrics.items():
            self._records.append(
                {
                    "target": target,
                    "task_type": task_type,
                    "metric": metric,
                    "model": str(model_name),
                    "fold": int(fold),
                    "value": value,
                }
            )

    def to_frame(self, dataset_name: str) -> pd.DataFrame:
        records = pd.DataFrame(self._records)
        if records.empty:
            return pd.DataFrame(columns=self.COLUMNS)

        records.insert(0, "dataset", dataset_name)
        target_order = {}
        for index, target in enumerate(self.target_cols):
            target_order[target] = index

        records["_target_order"] = records["target"].map(target_order)
        records = records.sort_values(["fold", "model", "_target_order", "metric"], kind="stable").drop(columns="_target_order")

        return records[self.COLUMNS].reset_index(drop=True)

    def write(self, dataset_name: str, output_root) -> Path:
        dataset_dir = Path(output_root) / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)

        metrics_path = dataset_dir / "metrics.csv"
        records = self.to_frame(dataset_name)
        records.to_csv(metrics_path, index=False)

        write_metric_tables(records, Path(output_root) / "results")

        return metrics_path


def _fold_label(fold):
    fold_as_float = float(fold)
    if fold_as_float.is_integer():
        return str(int(fold_as_float))
    return str(fold)


def write_metric_tables(fold_records, output_dir) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if fold_records.empty:
        return []

    paths = []
    for (dataset, metric), metric_rows in fold_records.groupby(
        ["dataset", "metric"], sort=True
    ):
        models = sorted(metric_rows["model"].dropna().unique())
        target_order = metric_rows["target"].drop_duplicates().tolist()
        rows = []
        row_names = []

        for target in target_order:
            target_rows = metric_rows[metric_rows["target"] == target]
            folds = sorted(target_rows["fold"].dropna().unique())
            for fold in folds:
                fold_values = (
                    target_rows[target_rows["fold"] == fold].groupby("model", sort=False)["value"].mean().reindex(models))
                rows.append(fold_values)
                row_names.append(f"{target}_fold{_fold_label(fold)}")

            average_values = (
                target_rows.groupby("model", sort=False)["value"].mean().reindex(models)
            )
            rows.append(average_values)
            row_names.append(f"{target}_avg")

        table = pd.DataFrame(rows, index=row_names, columns=models)
        path = output_dir / f"{dataset}_{metric}.csv"
        table.to_csv(path)
        paths.append(path)

    return paths


def normalize_predictions(predictions, target_cols) -> np.ndarray:
    if isinstance(predictions, Mapping):
        missing = [target for target in target_cols if target not in predictions]
        if missing:
            raise ValueError(f"Missing prediction target(s): {', '.join(missing)}")

        columns = []
        for target in target_cols:
            column = np.asarray(predictions[target])
            if column.ndim == 2 and column.shape[1] == 1:
                column = column[:, 0]
            if column.ndim != 1:
                raise ValueError(
                    f"Predictions for '{target}' must contain one value per sample; "
                    f"received shape {column.shape}"
                )
            columns.append(column)

        array = np.column_stack(columns)
    elif isinstance(predictions, (pd.Series, pd.DataFrame)):
        array = predictions.to_numpy()
    else:
        array = np.asarray(predictions)

    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2 or array.shape[1] != len(target_cols):
        raise ValueError(
            f"Expected predictions with {len(target_cols)} target column(s); "
            f"received shape {array.shape}"
        )

    try:
        return array.astype(float, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("Predictions must be numeric") from exc


def compute_fold_metrics(y_original, y_pred, target_cols, output_types,) -> dict[tuple[str, str, str], float]:
    if len(target_cols) != len(output_types):
        raise ValueError("target_cols and output_types must have the same length")

    truth = normalize_predictions(y_original, target_cols)
    predictions = normalize_predictions(y_pred, target_cols)
    if truth.shape != predictions.shape:
        raise ValueError(
            f"Truth and prediction shapes differ: {truth.shape} != {predictions.shape}"
        )

    results: dict[tuple[str, str, str], float] = {}
    for index, (target, output_type) in enumerate(zip(target_cols, output_types)):
        target_true = truth[:, index]
        target_pred = predictions[:, index]

        if output_type == "binary":
            labels = (target_pred >= 0.5).astype(int)
            auc = (
                float(roc_auc_score(target_true, target_pred))
                if np.unique(target_true).size > 1
                else math.nan
            )
            results[("binary", "AUC", target)] = auc

            results[("binary", "Accuracy", target)] = float(accuracy_score(target_true, labels))

            results[("binary", "F1", target)] = float(
                f1_score(target_true, labels, zero_division=0)
            )
        elif output_type == "regression":
            results[("regression", "MAE", target)] = float(
                mean_absolute_error(target_true, target_pred)
            )
            results[("regression", "RMSE", target)] = math.sqrt(
                float(mean_squared_error(target_true, target_pred))
            )
            results[("regression", "R2", target)] = float(
                r2_score(target_true, target_pred)
            )
        else:
            raise ValueError(f"Unsupported output type '{output_type}' for '{target}'")

    return results


def metric_frame(item, frame, models):
    if list(frame.columns) != MetricsCollector.COLUMNS:
        raise ValueError('Metric schema mismatch')

    folds = pd.to_numeric(frame['fold'], errors='coerce').to_numpy(dtype=float)
    values = pd.to_numeric(frame['value'], errors='coerce').to_numpy(dtype=float)
    if not np.isfinite(values).all() or not np.isfinite(folds).all() or (folds != np.floor(folds)).any():
        raise ValueError('Invalid scores or fold identifiers')

    keys = list(zip(frame['dataset'], frame['target'], frame['task_type'], frame['metric'], frame['model'], folds.astype(int)))
    kinds = {'binary': ('AUC', 'Accuracy', 'F1'), 'regression': ('MAE', 'RMSE', 'R2')}
    expected = {(item['label'], target, kind, metric, model, fold)
                for target, kind in item['target_types'].items() for metric in kinds[kind]
                for model in models for fold in range(1, 6)}

    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError('Incomplete, duplicate or unexpected metric coverage')

    return frame


def expected_results(backend, item, source, work, output, models):
    native = [model for model in models if model != 'mtt']
    frames, prediction_paths = [], []

    if native:
        aliases = backend.native_aliases(item)
        directory = output / 'native' / item['label']
        regular_tree(directory)
        frame = pd.read_csv(directory / 'metrics.csv', float_precision='round_trip')
        if set(frame['target']) != set(aliases):
            raise ValueError('Native score target schema differs from the prepared contract')

        frame['target'] = frame['target'].map(aliases)
        metric_frame(item, frame, native)

        if item['family'] in {'assistments', 'nedbox'}:
            truth_by_fold = backend.native_truth(item, source)
        else:
            folds = backend.module('prepared_inputs').read_native_matrices(work / 'native' / item['label'])
            truth_by_fold = {fold['fold']: (fold['validation_frame'][list(item['target_types'])], fold['validation_frame']['source_row_index'].to_numpy()) for fold in folds}

        predictions = directory / 'predictions'
        expected = {f'fold_{fold}_{model}.csv' for fold in range(1, 6) for model in native}
        if {path.name for path in predictions.iterdir()} != expected:
            raise ValueError('Native prediction file coverage mismatch')

        for fold in range(1, 6):
            truth, indices = truth_by_fold[fold]
            for model in native:
                path = predictions / f'fold_{fold}_{model}.csv'
                predicted = pd.read_csv(path, float_precision='round_trip')
                targets = list(predicted.columns[1:])
                if list(predicted.columns)[0] != 'row_index' or not np.array_equal(predicted['row_index'], indices):
                    raise ValueError('Native prediction row identity mismatch')
                if len(targets) != len(aliases) or set(targets) != set(aliases):
                    raise ValueError('Native prediction target schema mismatch')

                values = compute_fold_metrics(truth[targets], predicted[targets], targets,
                                              [item['target_types'][aliases[target]] for target in targets])
                observed = frame.loc[(frame['fold'] == fold) & (frame['model'] == model)]
                recorded = {(row.task_type, row.metric, row.target): row.value for row in observed.itertuples()}
                if any(not np.isclose(value, recorded[(kind, metric, aliases[target])], rtol=1e-12, atol=1e-12)
                       for (kind, metric, target), value in values.items()):
                    raise ValueError('Native scores do not reproduce prediction files')

                prediction_paths.append(path)

        frames.append(frame)

    if 'mtt' in models:
        regular_tree(checked_path(backend.root, output / 'mtt_logs'))
        data, configs = work / 'mtt/data', work / 'mtt/configs'
        spec = backend.specs(item, source)[item['label']]
        frame = backend.module('mtt_shared_stage').convert_label(spec, output / 'mtt_logs', data, configs)
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(['fold', 'model', 'target', 'metric'], kind='stable').reset_index(drop=True)
    metric_frame(item, combined, models)

    return combined, prediction_paths


def finish_results(backend, item, source, work, output, models):
    frame, predictions = expected_results(backend, item, source, work, output, models)
    target = output / 'experiment_outputs' / item['label']
    if target.exists():
        raise FileExistsError('Consolidated result already exists')

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.results-', dir=output))

    try:
        frame.to_csv(staging / 'metrics.csv', index=False)

        write_metric_tables(frame, staging / 'results')

        if predictions:
            (staging / 'predictions').mkdir()
            for path in predictions:
                shutil.copyfile(path, staging / 'predictions' / path.name)

        staging.replace(target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def verify_results(backend, item, source, work, output, models):
    expected, predictions = expected_results(backend, item, source, work, output, models)
    target = output / 'experiment_outputs' / item['label']
    regular_tree(target)
    actual = pd.read_csv(target / 'metrics.csv', float_precision='round_trip')
    metric_frame(item, actual, models)
    if not actual.equals(expected):
        raise ValueError('Consolidated scores diverge from the native/MTT sources')

    expected_predictions = {path.name for path in predictions}
    present = {path.name for path in (target / 'predictions').iterdir()} if (target / 'predictions').exists() else set()
    if present != expected_predictions:
        raise ValueError('Consolidated prediction coverage mismatch')

    for path in predictions:
        if (target / 'predictions' / path.name).read_bytes() != path.read_bytes():
            raise ValueError('Consolidated predictions diverge from native files')

    with tempfile.TemporaryDirectory(prefix='.readback-', dir=output) as temporary:
        paths = write_metric_tables(expected, Path(temporary))
        if {path.name for path in (target / 'results').iterdir()} != {path.name for path in paths}:
            raise ValueError('Consolidated metric-table coverage mismatch')
        for path in paths:
            if (target / 'results' / path.name).read_bytes() != path.read_bytes():
                raise ValueError('Consolidated metric tables diverge from scores')
