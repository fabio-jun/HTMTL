from pathlib import Path
import importlib.util
import sys
import numpy as np
import pandas as pd

from fold_preparation import BasePreparedAdapter
from dataset_registry import _load_dataset_module

VARIANT_DIR = Path(__file__).resolve().parent
INPUT_PATH = VARIANT_DIR / "data.csv"
SKILL_BUILDER_OUTPUT_PATH = VARIANT_DIR / "sb_preprocessed.csv"
NON_SKILL_BUILDER_OUTPUT_PATH = VARIANT_DIR / "nsb_preprocessed.csv"

AFFECT_COLUMNS = {
    "Average_confidence(FRUSTRATED)": "predicted_frustration_mean",
    "Average_confidence(CONFUSED)": "predicted_confusion_mean",
    "Average_confidence(CONCENTRATING)": "predicted_concentration_mean",
    "Average_confidence(BORED)": "predicted_boredom_mean",
}
SKILL_BUILDER_TYPE = "MasterySection"
NON_SKILL_BUILDER_TYPES = (
    "LinearSection", "RandomChildOrderSection", "RandomIterateSection", "PlacementsSection", "ChooseConditionSection",
    "NumericLimitSection",
)
REQUIRED_SOURCE_COLUMNS = (
    "problem_log_id", "user_id", "assignment_id", "problem_id", "start_time", "end_time", "original", "correct",
    "attempt_count", "hint_count", "bottom_hint", "first_action", "problem_type", "type", "tutor_mode", "skill_id",
    *AFFECT_COLUMNS,
)


def load_data(file_path: Path = INPUT_PATH) -> pd.DataFrame:
    return pd.read_csv(file_path, usecols=list(REQUIRED_SOURCE_COLUMNS), encoding="latin-1", low_memory=False, memory_map=True)


def split_regimes(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    values = df["type"]
    skill_builder_mask = values.eq(SKILL_BUILDER_TYPE).fillna(False)
    non_skill_builder_mask = values.isin(NON_SKILL_BUILDER_TYPES).fillna(False)

    skill_builder_rows = df.loc[skill_builder_mask].reset_index(drop=True)
    non_skill_builder_rows = df.loc[non_skill_builder_mask].reset_index(drop=True)


    return {
        "skill_builder": skill_builder_rows,
        "non_skill_builder": non_skill_builder_rows,
    }


def _load_branch_module(module_name: str):
    qualified_name = f"_assistments2012_2013_{module_name}"
    spec = importlib.util.spec_from_file_location(qualified_name, VARIANT_DIR / f"{module_name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified_name] = module
    spec.loader.exec_module(module)
    return module


def build_report(df: pd.DataFrame, split: dict[str, pd.DataFrame], skill_builder: pd.DataFrame, non_skill_builder: pd.DataFrame) -> dict:
    skill_builder_problem_log_ids = split["skill_builder"]["problem_log_id"]
    non_skill_builder_problem_log_ids = split["non_skill_builder"]["problem_log_id"]
    combined_problem_log_ids = np.concatenate([
        skill_builder_problem_log_ids.to_numpy(),
        non_skill_builder_problem_log_ids.to_numpy(),
    ])
    report = {
        "source_row_count": int(len(df)),
        "skill_builder_source_row_count": int(len(split["skill_builder"])),
        "non_skill_builder_source_row_count": int(len(split["non_skill_builder"])),
        "problem_log_id_coverage": bool(len(combined_problem_log_ids) == len(df)),
        "problem_log_id_exclusive": bool(len(np.unique(combined_problem_log_ids)) == len(combined_problem_log_ids)),
    }
    for suffix, frame in (
        ("skill_builder", skill_builder),
        ("non_skill_builder", non_skill_builder),
    ):
        report.update({f"{suffix}_{key}": value for key, value in frame.attrs.items()})
    return report


def preprocess_dataframe(df: pd.DataFrame) -> dict[str, pd.DataFrame | dict]:
    split = split_regimes(df)
    skill_builder_module = _load_branch_module("preprocess_sb")
    non_skill_builder_module = _load_branch_module("preprocess_nsb")
    skill_builder_frame = skill_builder_module.preprocess_dataframe(split["skill_builder"])
    non_skill_builder_frame = non_skill_builder_module.preprocess_dataframe(split["non_skill_builder"])
    report = build_report(df, split, skill_builder_frame, non_skill_builder_frame)
    return {
        "skill_builder": skill_builder_frame,
        "non_skill_builder": non_skill_builder_frame,
        "report": report,
    }


def write_preprocessed_dataset(
    input_path: Path = INPUT_PATH,
    skill_builder_output_path: Path = SKILL_BUILDER_OUTPUT_PATH,
    non_skill_builder_output_path: Path = NON_SKILL_BUILDER_OUTPUT_PATH,
) -> dict[str, pd.DataFrame | dict]:
    input_path = Path(input_path)
    skill_builder_output_path = Path(skill_builder_output_path)
    non_skill_builder_output_path = Path(non_skill_builder_output_path)
    output_paths = (skill_builder_output_path, non_skill_builder_output_path)
    if skill_builder_output_path.resolve() == non_skill_builder_output_path.resolve():
        raise ValueError("Skill Builder and Non-Skill Builder output paths must differ")
    if input_path.resolve() in {path.resolve() for path in output_paths} or (
        input_path.exists()
        and any(
            path.exists() and input_path.samefile(path)
            for path in output_paths
        )
    ):
        raise ValueError("Input and output paths must differ to protect the raw source")
    result = preprocess_dataframe(load_data(input_path))
    result["skill_builder"].to_csv(skill_builder_output_path, index=False)
    result["non_skill_builder"].to_csv(non_skill_builder_output_path, index=False)
    return result


DATASETS_ROOT = Path(__file__).resolve().parents[1]
class _Base2012Adapter(BasePreparedAdapter):
    branch_file = "preprocess_sb.py"
    branch_module_name = "_prepared_assistments2012_sb"
    regime = "skill_builder"
    student_column = "source_user_id"
    interaction_student_column = "user_id"
    attempt_column = "attempt_count"
    time_column = "overlap_time_seconds"
    time_unit = "seconds"

    def __init__(self, split_module=None, branch_module=None, input_path: Path | None = None):
        self._split_module = split_module
        self._branch_module = branch_module
        self.input_path = input_path

    @property
    def split_module(self):
        if self._split_module is None:
            self._split_module = _load_dataset_module(
                "_prepared_assistments2012_split_regimes",
                DATASETS_ROOT / "assistments2012-2013" / "preprocess_split_regimes.py",
            )
        return self._split_module

    @property
    def branch_module(self):
        if self._branch_module is None:
            self._branch_module = _load_dataset_module(
                self.branch_module_name,
                DATASETS_ROOT / "assistments2012-2013" / self.branch_file,
            )
        return self._branch_module

    def source_paths(self):
        return [self.input_path or self.split_module.INPUT_PATH]

    def load(self):
        raw = self.split_module.load_data(self.input_path or self.split_module.INPUT_PATH)
        split = self.split_module.split_regimes(raw)
        frame = split[self.regime]
        artifact = self.branch_module.preprocess_dataframe(frame)
        return {
            "frame": frame,
            "artifact": artifact,
            "provisional_students": sorted(artifact["source_user_id"].astype(str).unique().tolist()),
        }

    def interactions(self, context):
        return self._fixed_validity(context["frame"])

    def build_rows(self, context, students, interaction_filter):
        subset = context["frame"].loc[context["frame"]["user_id"].astype(str).isin(map(str, students))]
        return self._preprocess(subset, interaction_filter)


def main():
    write_preprocessed_dataset()


if __name__ == "__main__":
    main()
