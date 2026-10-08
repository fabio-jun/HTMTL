import sys as _sys
from pathlib import Path as _Path
_release_root = _Path(__file__).resolve().parents[2]
if str(_release_root / "review") not in _sys.path:
    _sys.path.insert(0, str(_release_root / "review"))
from dataset_registry import configure_imports
configure_imports(_release_root)


import sys
import os
import argparse
import shutil
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Custom models
from experiment_metrics import MetricsCollector, normalize_predictions
from fold_preparation import read_prepared_folds, validate_prepared_label as _validate_prepared_label
import json
from dataset_registry import checked_path, entry, regular_tree, separate_output, write_json

# -------------------------
# Global parameters
# -------------------------
RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = None
RF_RANDOM_STATE = 42
N_CHAINS = "all"
N_FOLDS = 5
N_JOBS = int(os.environ.get("MTL_N_JOBS") or -1)

# -------------------------
# Model mapping
# -------------------------
short_to_long = {
    "hu":  "hmtrf_uniform",
    "hun": "hmtrf_uniform_norm",
    "hrn": "hmtrf_random_norm",
    "loc": "local",
    "ecc": "ecc_scikit",
    "rfstd": "RFStandard",
    #"dte": "dte",
    "mlp": "mlp"
}
all_models = list(short_to_long.values())


# -------------------------
# Infer output types
# -------------------------
def infer_output_types(df, target_cols):
    types = []
    for col in target_cols:
        if "binary" in col:
            types.append("binary")
        elif "multiclass" in col:
            types.append("classification")
        else:
            types.append("regression")
    return types

def prepare_dataset_output(dataset_name, output_root):
    dataset_dir = Path(output_root) / dataset_name
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    (dataset_dir / "predictions").mkdir(parents=True)
    return dataset_dir

BARE_ID_COLUMNS = ("user_id", "selected_teaser_id", "assignment_id", "problem_id")

def save_and_evaluate(
    predictions,
    target_cols,
    dataset_name,
    method_name,
    fold,
    test_indices,
    y_test,
    collector,
    output_root,
):
    normalized = normalize_predictions(predictions, target_cols)
    if len(test_indices) != normalized.shape[0]:
        raise ValueError(
            f"Prediction count for {method_name} fold {fold} does not match test indices"
        )

    prediction_frame = pd.DataFrame(normalized, columns=target_cols)
    prediction_frame.insert(0, "row_index", np.asarray(test_indices))
    prediction_filename = f"fold_{fold}_{method_name}.csv"
    filename = Path(output_root) / dataset_name / "predictions" / prediction_filename
    filename.parent.mkdir(parents=True, exist_ok=True)
    prediction_frame.to_csv(filename, index=False)
    collector.add_fold(method_name, fold, y_test, normalized)
    print(f"Saved predictions to {filename}")
    return filename


def build_parser():
    parser = argparse.ArgumentParser(description="Train a complete native/MTT panel from prepared inputs; run prepare first.")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--label", help="Registered dataset for the complete panel, including MTT")
    inputs.add_argument("--prepared-dir", help="Legacy low-level native-only fold bundle")
    parser.add_argument("--work-root", help="Completed prepare output for --label")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--output-root", required=True)
    return parser


def validate_prepared_label(dataset_name):
    return _validate_prepared_label(dataset_name)


def validate_prepared_output_location(dataset_name, prepared_dir, output_root):
    validate_prepared_label(dataset_name)
    prepared = Path(prepared_dir).expanduser().resolve(strict=False)
    root = Path(output_root).expanduser().resolve(strict=False)
    target = (root / dataset_name).resolve(strict=False)
    if target == root or root not in target.parents:
        raise ValueError(f"Prepared output {target} escapes the explicit output root {root}")
    if target == prepared or prepared in target.parents or target in prepared.parents:
        raise ValueError(f"Prepared output {target} collides with prepared inputs {prepared}")

def train_and_evaluate_models(
    selected_models,
    X_train,
    y_train,
    X_test,
    y_test,
    test_indices,
    fold,
    output_types,
    target_cols,
    dataset_name,
    collector,
    output_root,
):
    # -------------------------
    # HMTRF Uniform no heuristic normalization
    # -------------------------
    load_native_models()
    if "hmtrf_uniform" in selected_models:
        model = HMTRF(output_types=output_types, n_estimators=RF_N_ESTIMATORS, max_depth=RF_MAX_DEPTH, weights_scheme="uniform", heuristic_normalization=False, n_jobs=N_JOBS, progress_label="hmtrf_uniform")
        model.fit(X_train, y_train)
        save_and_evaluate(model.predict_proba(X_test), target_cols, dataset_name, "hmtrf_uniform", fold, test_indices, y_test, collector, output_root)

    # -------------------------
    # HMTRF Uniform with heuristic normalization
    # -------------------------
    if "hmtrf_uniform_norm" in selected_models:
        model = HMTRF(output_types=output_types, n_estimators=RF_N_ESTIMATORS, max_depth=RF_MAX_DEPTH, weights_scheme="uniform", heuristic_normalization=True, n_jobs=N_JOBS, progress_label="hmtrf_uniform_norm")
        model.fit(X_train, y_train)
        save_and_evaluate(model.predict_proba(X_test), target_cols, dataset_name, "hmtrf_uniform_norm", fold, test_indices, y_test, collector, output_root)

    # -------------------------
    # HMTRF Random with heuristic normalization
    # -------------------------
    if "hmtrf_random_norm" in selected_models:
        model = HMTRF(output_types=output_types, n_estimators=RF_N_ESTIMATORS, max_depth=RF_MAX_DEPTH, weights_scheme="random", heuristic_normalization=True, n_jobs=N_JOBS, progress_label="hmtrf_random_norm")
        model.fit(X_train, y_train)
        save_and_evaluate(model.predict_proba(X_test), target_cols, dataset_name, "hmtrf_random_norm", fold, test_indices, y_test, collector, output_root)

    # -------------------------
    # Local
    # -------------------------
    if "local" in selected_models:
        model = Local(
            base_regressor=RandomForestRegressor(n_estimators=RF_N_ESTIMATORS, max_depth=RF_MAX_DEPTH, n_jobs=N_JOBS),
            base_classifier=RandomForestClassifier(n_estimators=RF_N_ESTIMATORS, max_depth=RF_MAX_DEPTH, n_jobs=N_JOBS),
            output_types=output_types,
        )
        model.fit(X_train, y_train)
        save_and_evaluate(model.predict_proba(X_test), target_cols, dataset_name, "local", fold, test_indices, y_test, collector, output_root)

    # -------------------------
    # ECC
    # -------------------------
    if "ecc_scikit" in selected_models:
        model = ECCScikit(
            RandomForestRegressor(n_estimators=RF_N_ESTIMATORS, max_depth=RF_MAX_DEPTH, n_jobs=N_JOBS),
            RandomForestClassifier(n_estimators=RF_N_ESTIMATORS, max_depth=RF_MAX_DEPTH, n_jobs=N_JOBS),
            output_types=output_types,
        )
        model.fit(X_train, y_train)
        save_and_evaluate(model.predict_proba(X_test), target_cols, dataset_name, "ecc", fold, test_indices, y_test, collector, output_root)

    # -------------------------
    # RF Standard
    # -------------------------
    if "RFStandard" in selected_models:
        model = RFStandard(output_types=output_types, n_estimators=RF_N_ESTIMATORS, max_depth=RF_MAX_DEPTH, n_jobs=N_JOBS)
        model.fit(X_train, y_train)
        save_and_evaluate(model.predict(X_test), target_cols, dataset_name, "rfstd", fold, test_indices, y_test, collector, output_root)

    # -------------------------
    # MLP
    # -------------------------
    if "mlp" in selected_models:
        print("Training MLP...")
        mlp = MultiTaskMLP(output_types=output_types)
        mlp.fit(X_train, y_train)

        preds = mlp.predict_proba(X_test) if hasattr(mlp, "predict_proba") else mlp.predict(X_test)
        if output_root is not None and isinstance(preds, dict):
            preds = np.column_stack(list(preds.values()))

        save_and_evaluate(preds, target_cols, dataset_name, "mlp", fold, test_indices, y_test, collector, output_root)


# -------------------------
# Prepared-fold input mode
# -------------------------
def run_prepared_experiments(args, selected_models):
    folds = read_prepared_folds(args.prepared_dir)
    manifest = folds[0]["manifest"]
    schema = manifest["schema"]
    dataset_name = manifest["label"]
    validate_prepared_output_location(dataset_name, args.prepared_dir, args.output_root)
    load_native_models()
    target_cols = list(schema["target_columns"])
    feature_cols = list(schema["feature_columns"])
    output_types = infer_output_types(None, target_cols)

    print("Prepared label:", dataset_name)
    print("Targets:", target_cols)
    print("Types:", output_types)
    print("Models:", selected_models)
    print("Features:", feature_cols)

    prepare_dataset_output(dataset_name, args.output_root)
    cv_collector = MetricsCollector(target_cols, output_types)

    for entry in folds:
        fold = entry["fold"]
        train_frame = entry["train_frame"]
        validation_frame = entry["validation_frame"]
        print(f"\n=== Prepared fold {fold} (train rows={len(train_frame)}, validation rows={len(validation_frame)}) ===")

        X_train = train_frame[feature_cols].to_numpy(dtype=float)
        y_train = train_frame[target_cols].to_numpy(dtype=float)
        X_test = validation_frame[feature_cols].to_numpy(dtype=float)
        y_test = validation_frame[target_cols].to_numpy(dtype=float)
        test_indices = validation_frame["source_row_index"].to_numpy() if manifest.get("protocol") == "native_materialized" else np.arange(len(validation_frame))

        train_and_evaluate_models(
            selected_models,
            X_train,
            y_train,
            X_test,
            y_test,
            test_indices,
            fold,
            output_types,
            target_cols,
            dataset_name,
            cv_collector,
            args.output_root,
        )

    metrics_path = cv_collector.write(dataset_name, args.output_root)
    print(f"Saved metrics to {metrics_path}")


# -------------------------
# Main
# -------------------------
def load_native_models():
    global HMTRF, Local, ECCScikit, RFStandard, MultiTaskMLP
    if all(name in globals() for name in ("HMTRF", "Local", "ECCScikit", "RFStandard", "MultiTaskMLP")):
        return
    from HMTRF.HMTRF import HMTRF
    from Local.Local import Local
    from ECCScikit.ECCScikit import ECCScikit
    from RFStandard.RFStandard import RFStandard
    from MLP.MultiTaskMLP import MultiTaskMLP


def run_panel(root, label, models, output_root, work_root=None, backend=None):
    root = checked_path(Path.cwd(), root)
    item = entry(root, label)
    if not models or len(set(models)) != len(models) or (models != ['all'] and set(models) - set(item['models'])):
        raise ValueError('Select only official models, without duplicates')
    
    selected = list(item['models']) if models == ['all'] else [model for model in item['models'] if model in models]

    if work_root is None:
        raise FileNotFoundError('Specify --work-root from a completed prepare stage')
    
    work, output = checked_path(root, work_root), checked_path(root, output_root)
    regular_tree(work)
    record_path = work / 'prepared.json'

    if not record_path.is_file():
        raise FileNotFoundError('Preparation is partial; a completed prepare stage is required')
    prepared = json.loads(record_path.read_text())

    if prepared.get('label') != label or prepared.get('preparation') != item.get('preparation'):
        raise ValueError('Prepared recipe differs from the selected official cohort')
    
    source = checked_path(root, prepared['source'])
    regular_tree(source)
    separate_output(root, output, [source, work])

    if backend is None:
        from pipeline import Backend
        backend = Backend(root)
    backend.verify_inputs(item, source, work, selected)

    if hasattr(backend, 'training_ready'):
        backend.training_ready(selected)
    record = {'label': label, 'models': selected, 'source': str(source), 'work_root': str(work)}

    if output.exists():
        regular_tree(output)
        if not (output / 'complete.json').is_file() or not (output / 'run.json').is_file():
            raise FileExistsError(f'Partial output is preserved; select a fresh output root: {output}')
        if json.loads((output / 'run.json').read_text()) != record or json.loads((output / 'complete.json').read_text()) != record:
            raise ValueError('Existing completed output belongs to another requested run')
        backend.verify_results(item, source, work, output, selected)
        return 0
    
    output.mkdir(parents=True)
    write_json(output / 'run.json', record)
    native = [model for model in selected if model != 'mtt']
    
    if native:
        native_source = source if item['family'] in {'assistments', 'nedbox'} else work / 'native' / label
        code = backend.run_native(item, native_source, output, native)
        if code:
            write_json(output / 'failed.json', {'stage': 'native', 'exitcode': code})
            return code
    if 'mtt' in selected:
        code = backend.run_mtt(item, source, work, output)
        if code:
            write_json(output / 'failed.json', {'stage': 'mtt', 'exitcode': code})
            return code
    backend.finish_results(item, source, work, output, selected)
    backend.verify_results(item, source, work, output, selected)
    write_json(output / 'complete.json', record)
    return 0


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.label is not None:
        if args.work_root is None:
            parser.error("--label requires --work-root from a completed prepare stage")
        aliases = {"hu": "hmtrf_uniform", "hun": "hmtrf_uniform_norm",
                   "hrn": "hmtrf_random_norm", "loc": "local", "ecc_scikit": "ecc", "RFStandard": "rfstd"}
        models = [aliases.get(model, model) for model in args.models]
        return run_panel(Path(__file__).resolve().parents[2], args.label, models, args.output_root, args.work_root)
    if args.work_root is not None or "mtt" in args.models:
        parser.error("MTT/full-panel execution requires --label and --work-root; --prepared-dir is native-only")
    selected_models = all_models if args.models == ["all"] else [short_to_long.get(model, model) for model in args.models]
    invalid = sorted(set(selected_models).difference(all_models))
    if invalid:
        parser.error(f"invalid model name(s): {', '.join(invalid)}")
    run_prepared_experiments(args, selected_models)


if __name__ == "__main__":
    raise SystemExit(main())
