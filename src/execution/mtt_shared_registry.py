from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_release_root = _Path(__file__).resolve().parents[2]
if str(_release_root / "review") not in _sys.path:
    _sys.path.insert(0, str(_release_root / "review"))
from dataset_registry import configure_imports
configure_imports(_release_root)


from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from fold_preparation import discover_feature_columns, read_manifest
from dataset_registry import inventory, entry

_RECORDS = inventory(Path(__file__).resolve().parents[2])
PREPARED_LABELS = tuple(label for label, item in _RECORDS.items() if item["family"] == "assistments")
CLASSIC_PATHS = {label: item["input_location"] for label, item in _RECORDS.items() if item["family"] == "traditional"}

N_FOLDS = 5
SEED = 42
SELECTION_VAL_FRACTION = 0.2

RUN_ROOT_RELATIVE = "artifacts/runs/mtt-shared-input"
PREPARED_ROOT_RELATIVE = "prepared_folds_assistments"
NEDBOX_PREPARED_ROOT_RELATIVE = "artifacts/prepared"

SPR_SOURCE_RELATIVE = "artifacts/runs/spr-4013/source"
SPR_EMBEDDINGS_NAME = "spr_xray_embeddings.npy"
SPR_MANIFEST_NAME = "spr_xray_manifest.csv"
SPR_COHORT_SELECTION_NAME = "cohort_selection.csv"

SPR_ROWS = 4013
SPR_FEATURES = 768
BARE_ID_COLUMNS = ("user_id", "selected_teaser_id", "assignment_id", "problem_id")

MTT_ROW_COL_ARCH = {
    "att_type": "row_col", "n_blocks": 1, "n_heads": 4, "embed_dim": 16,
    "ff_hid_dim": 256, "tower_hid_dims": [64], "mask_mode_if": "TxT",
    "multi_token": True, "rope": True,
}
MTT_COL_ARCH = {**MTT_ROW_COL_ARCH, "att_type": "col"}

MTT_TRAINING = {
    "epochs": 100, "lr": 0.001, "optimizer": "adam", "batch_size": 32,
    "weight_decay": 0.00001, "ff_dropout": 0.0, "att_dropout": 0.0,
}

CLASSIC_FAMILY = "classic"
PREPARED_FAMILY = "prepared"
SPR_FAMILY = "spr"
HIGGS_LABELS = ("higgs_50k", "higgs_50k_mtt_cv", "higgs")

FAMILY_ORDER = (
    *PREPARED_LABELS,
    *CLASSIC_PATHS,
    "spr_xray_manifest",
    "nedbox",
    "higgs_50k",
)
LABELS = tuple(FAMILY_ORDER)
CLASSIC_LABELS = tuple(CLASSIC_PATHS)


@dataclass(frozen=True)
class DatasetSpec:
    label: str
    family: str
    source: Path
    aliases: dict
    target_types: dict
    feature_columns: tuple
    num_features: int
    group_column: str | None
    att_type: str
    native_models: tuple
    short_name: str
    public_label: str | None = None
    profile: str | None = None

    @property
    def result_label(self):
        return self.public_label or self.label


def family_of(label):
    if label == "higgs_50k":
        return CLASSIC_FAMILY
    if label in PREPARED_LABELS:
        return PREPARED_FAMILY
    if label in CLASSIC_PATHS:
        return CLASSIC_FAMILY
    if label == "spr_xray_manifest":
        return SPR_FAMILY
    if label == "nedbox":
        return PREPARED_FAMILY
    raise ValueError(f"Unknown MTT migration label: {label!r}")


def _target_types(aliases, metadata, label):
    if set(aliases.values()) != set(metadata["target_types"]):
        raise ValueError(f"{label} aliases differ from configured targets")

    return {source: metadata["target_types"][output] for source, output in aliases.items()}


def _run_root(project_root):
    return Path(project_root) / RUN_ROOT_RELATIVE


def data_root(project_root):
    return _run_root(project_root) / "data"


def config_root(project_root):
    return _run_root(project_root) / "configs"


def log_root(project_root):
    return _run_root(project_root) / "logs"


def _native_models(metadata):
    models = tuple(model for model in metadata["models"] if model != "mtt")
    if not models:
        raise ValueError("No native models configured")
    return models


def _classic_spec(project_root, label):
    path = Path(project_root) / CLASSIC_PATHS[label]
    frame = pd.read_csv(path, nrows=0)
    targets = [column for column in frame.columns if column.startswith("target_")]
    aliases = entry(project_root, label)["target_aliases"]
    if set(aliases) != set(targets):
        raise ValueError(f"{label} classic alias keys do not match source targets")

    features = discover_feature_columns(frame, targets)
    group_column = next(
        (column for column in ("source_user_id", "source_student_id", "user_id") if column in frame.columns),
        None,
    )

    live = entry(project_root, label)

    return DatasetSpec(
        label=label,
        family=CLASSIC_FAMILY,
        source=path,
        aliases=dict(aliases),
        target_types=_target_types(aliases, live, label),
        feature_columns=tuple(features),
        num_features=len(features),
        group_column=group_column,
        att_type=MTT_ROW_COL_ARCH["att_type"],
        native_models=_native_models(live),
        short_name=label,
    )


def _prepared_spec(project_root, label, root_relative, manifest_label=None, source=None):
    if source is None:
        source = Path(project_root) / entry(project_root, label)["input_location"]

    source = Path(source)
    manifest = read_manifest(source)
    schema = manifest["schema"]
    targets = list(schema["target_columns"])
    features = list(schema["feature_columns"])
    aliases = entry(project_root, label)["target_aliases"]
    if set(aliases) != set(targets):
        raise ValueError(f"{label} configured alias keys differ from prepared targets")

    student_column = manifest["student_column"]
    if student_column not in schema["columns"]:
        raise ValueError(f"{label} prepared student column missing from schema")

    live = entry(project_root, label)

    return DatasetSpec(
        label=manifest_label or label,
        family=PREPARED_FAMILY,
        source=source,
        aliases=dict(aliases),
        target_types=_target_types(aliases, live, label),
        feature_columns=tuple(features),
        num_features=len(features),
        group_column=student_column,
        att_type=MTT_ROW_COL_ARCH["att_type"],
        native_models=_native_models(live),
        short_name=manifest.get("label", label),
    )


def _spr_spec(project_root):
    source = Path(project_root) / entry(project_root, "spr_xray_manifest")["input_location"]
    aliases = entry(project_root, "spr_xray_manifest")["target_aliases"]
    features = tuple(f"emb_{index}" for index in range(SPR_FEATURES))
    live = entry(project_root, "spr_xray_manifest")

    return DatasetSpec(
        label="spr_xray_manifest",
        family=SPR_FAMILY,
        source=source,
        aliases=aliases,
        target_types=_target_types(aliases, live, "spr_xray_manifest"),
        feature_columns=features,
        num_features=SPR_FEATURES,
        group_column=None,
        att_type=MTT_ROW_COL_ARCH["att_type"],
        native_models=_native_models(live),
        short_name="spr_xray_mtt_row_col",
    )


def build_specs(project_root, labels=None, sources=None):
    root = Path(project_root)
    requested = LABELS if labels is None else tuple(labels)
    if not requested or set(requested) - set(LABELS) or len(set(requested)) != len(requested):
        raise ValueError(f"Invalid MTT label scope: {requested!r}")

    specs = {}
    sources = {} if sources is None else sources

    for label in PREPARED_LABELS:
        if label not in requested:
            continue
        specs[label] = _prepared_spec(root, label, PREPARED_ROOT_RELATIVE, source=sources.get(label))

    if "nedbox" in requested:
        specs["nedbox"] = _prepared_spec(root, "nedbox", NEDBOX_PREPARED_ROOT_RELATIVE, source=sources.get("nedbox"))

    for label in CLASSIC_PATHS:
        if label in requested:
            specs[label] = _classic_spec(root, label)

    if "spr_xray_manifest" in requested:
        specs["spr_xray_manifest"] = _spr_spec(root)

    if "higgs_50k" in requested:
        item = entry(root, "higgs_50k")
        path = Path(sources.get("higgs_50k", root / item["input_location"]))
        frame = pd.read_csv(path, nrows=0)
        targets = list(item["target_aliases"])
        aliases = {target.removeprefix("target_binary_").removeprefix("target_regression_"): target for target in targets}
        features = tuple(discover_feature_columns(frame, targets))
        if len(features) != 21 or list(frame.filter(like="target_").columns) != targets:
            raise ValueError("HIGGS official feature/target schema mismatch")
        specs["higgs_50k"] = DatasetSpec("higgs_50k_mtt_cv", CLASSIC_FAMILY, path, aliases,
            {task: item["target_types"][target] for task, target in aliases.items()}, features, 21,
            None, "row_col", _native_models(item), "hig50k", "higgs_50k", "higgs50k")

    return {label: specs[label] for label in requested}


def assert_scope(project_root, specs=None, labels=None):
    labels = LABELS if labels is None else tuple(labels)
    if not labels or set(labels) - set(LABELS) or len(set(labels)) != len(labels):
        raise ValueError(f"Invalid MTT label scope: {labels!r}")
    specs = build_specs(project_root, labels) if specs is None else specs
    if tuple(specs) != labels:
        raise ValueError(f"MTT spec scope drifted: {tuple(specs)!r}")
    return specs


def mtt_config(spec):
    arch = MTT_COL_ARCH if spec.att_type == "col" else MTT_ROW_COL_ARCH
    model = {"type": "mt", "name": "mtt", **{key: value for key, value in arch.items()}}
    if spec.profile == "higgs50k":
        model = {key: model[key] for key in ("type", "name", "n_blocks", "n_heads", "embed_dim", "ff_hid_dim", "tower_hid_dims", "mask_mode_if", "att_type", "multi_token", "rope")}

    return {"model": model, "training": dict(MTT_TRAINING)}
