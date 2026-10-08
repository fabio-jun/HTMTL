from pathlib import Path

import pandas as pd

from mtt_shared_registry import N_FOLDS
from mtt_shared_run import fold_base, load_row_counts
from mtt_shared_validation import COLUMNS, suffix_rows, validate_fold_metrics, validate_metric_frame, verify_hparams


def convert_label(spec, log_root_path, data_path, config_path):
    records, folds = [], []

    for fold in range(1, N_FOLDS + 1):
        base = fold_base(log_root_path, spec.label, fold)
        metrics, hparams = base / "metrics.csv", base / "hparams.yaml"

        if not metrics.is_file() or not hparams.is_file():
            raise FileNotFoundError(f"{spec.label} fold {fold}: missing metrics or hparams")

        row_counts = load_row_counts(spec, data_path, fold)
        verify_hparams(spec, hparams, fold, data_root_path=Path(data_path),
                       config_root_path=Path(config_path), row_counts=row_counts)

        records.extend(validate_fold_metrics(spec, metrics, fold))
        folds.append(fold)

    frame = pd.DataFrame(records, columns=COLUMNS)
    validate_metric_frame(spec, frame, expected_rows=suffix_rows(spec))

    order = {target: index for index, target in enumerate(spec.aliases.values())}
    frame["_target_order"] = frame["target"].map(order)
    frame = frame.sort_values(["fold", "model", "_target_order", "metric"], kind="stable").drop(columns="_target_order")

    return frame[COLUMNS].reset_index(drop=True)
