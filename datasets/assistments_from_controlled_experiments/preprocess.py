import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd


from fold_preparation import BasePreparedAdapter
from dataset_registry import _load_dataset_module

INPUT_PATH = Path(__file__).resolve().parent / "ThisOne.csv"
OUTPUT_PATH = Path(__file__).resolve().parent / "assistments_controlled_experiments_preprocessed.csv"

MIN_PRIOR_ASSIGNMENT_COUNT = 3

NUMERIC_HISTORY_FIELDS = (
    "Prior Problem Count", "Prior Percent Correct", "Prior Assignments Assigned", "Prior Assignment Count",
    "Prior Percent Completion", "Prior Class Percent Completion", "Z-Scored Mastery Speed", "Prior Homework Assigned",
    "Prior Homework Count", "Prior Homework Percent Completion", "Prior Class Homework Percent Completion",
    "Z-Scored HW Mastery Speed",
)
NUMERIC_HISTORY_COLUMNS = (
    "prior_problem_count", "prior_percent_correct", "prior_assignments_assigned", "prior_assignment_count",
    "prior_percent_completion", "prior_class_percent_completion", "z_scored_mastery_speed", "prior_homework_assigned",
    "prior_homework_count", "prior_homework_percent_completion", "prior_class_homework_percent_completion",
    "z_scored_hw_mastery_speed",
)
GRADE_COLUMNS = (
    "class_grade_5", "class_grade_6", "class_grade_7", "class_grade_8", "class_grade_9", "class_grade_10",
    "class_grade_11", "class_grade_12", "class_grade_freshman", "class_grade_sophomore", "class_grade_junior",
    "class_grade_senior", "class_grade_unknown",
)
GENDER_COLUMNS = (
    "guessed_gender_female", "guessed_gender_male", "guessed_gender_unknown",
)
STATE_COLUMNS = (
    "state_5", "state_7", "state_9", "state_10", "state_20", "state_22", "state_23", "state_30", "state_31", "state_33",
    "state_34", "state_36", "state_39", "state_44", "state_48", "state_53",
)
PROBLEM_SET_COLUMNS = (
    "problem_set_226210", "problem_set_237447", "problem_set_241501", "problem_set_241622", "problem_set_243393",
    "problem_set_246482", "problem_set_246627", "problem_set_246647", "problem_set_250476", "problem_set_255116",
    "problem_set_256017", "problem_set_256027", "problem_set_259379", "problem_set_263015", "problem_set_263052",
    "problem_set_263057", "problem_set_263109", "problem_set_263115", "problem_set_293151", "problem_set_303899",
    "problem_set_377658", "problem_set_377938",
)
FEATURE_COLUMNS = (
    NUMERIC_HISTORY_COLUMNS
    + GRADE_COLUMNS
    + GENDER_COLUMNS
    + STATE_COLUMNS
    + PROBLEM_SET_COLUMNS
)
METADATA_COLUMNS = ("source_user_id", "source_problem_set_id")
TARGET_COLUMNS = ("target_binary_complete", "target_regression_problems")
OUTPUT_COLUMNS = METADATA_COLUMNS + FEATURE_COLUMNS + TARGET_COLUMNS

GRADE_TO_COLUMN = {
    "5": "class_grade_5", "6": "class_grade_6", "7": "class_grade_7", "8": "class_grade_8", "9": "class_grade_9",
    "10": "class_grade_10", "11": "class_grade_11", "12": "class_grade_12", "Freshmen": "class_grade_freshman",
    "Sophomore": "class_grade_sophomore", "Junior": "class_grade_junior", "Senior": "class_grade_senior",
    "N/A": "class_grade_unknown",
}
GENDER_TO_COLUMN = {
    "Female": "guessed_gender_female", "Male": "guessed_gender_male", "Uknown": "guessed_gender_unknown",
}
STATE_TO_COLUMN = {
    "5": "state_5", "7": "state_7", "9": "state_9", "10": "state_10", "20": "state_20", "22": "state_22",
    "23": "state_23", "30": "state_30", "31": "state_31", "33": "state_33", "34": "state_34", "36": "state_36",
    "39": "state_39", "44": "state_44", "48": "state_48", "53": "state_53",
}
PROBLEM_SET_TO_COLUMN = {
    "226210": "problem_set_226210", "237447": "problem_set_237447", "241501": "problem_set_241501",
    "241622": "problem_set_241622", "243393": "problem_set_243393", "246482": "problem_set_246482",
    "246627": "problem_set_246627", "246647": "problem_set_246647", "250476": "problem_set_250476",
    "255116": "problem_set_255116", "256017": "problem_set_256017", "256027": "problem_set_256027",
    "259379": "problem_set_259379", "263015": "problem_set_263015", "263052": "problem_set_263052",
    "263057": "problem_set_263057", "263109": "problem_set_263109", "263115": "problem_set_263115",
    "293151": "problem_set_293151", "303899": "problem_set_303899", "377658": "problem_set_377658",
    "377938": "problem_set_377938",
}


def load_valid_rows(file_path: Path = INPUT_PATH) -> tuple[pd.DataFrame, dict[str, int]]:
    with Path(file_path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        expected_field_count = len(header)
        rows = []
        malformed = {"short": 0, "overlong": 0}
        for row in reader:
            if len(row) < expected_field_count:
                malformed["short"] += 1
            elif len(row) > expected_field_count:
                malformed["overlong"] += 1
            else:
                rows.append(row)
    return pd.DataFrame(rows, columns=header), malformed


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _clean_quoted(value) -> str:
    return str(value).strip().strip('"').strip()


def _map_category(series: pd.Series, mapping: dict, label: str) -> pd.Series:
    cleaned = series.map(_clean_quoted)
    mapped = cleaned.map(mapping)
    return mapped


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:

    prior_assignment_count = _numeric(df["Prior Assignment Count"])
    df = df.loc[np.isfinite(prior_assignment_count.to_numpy(dtype=float)) & (prior_assignment_count >= MIN_PRIOR_ASSIGNMENT_COUNT)]

    grade_columns = _map_category(df["Class Grade"], GRADE_TO_COLUMN, "Class Grade")
    gender_columns = _map_category(df["Guessed Gender"], GENDER_TO_COLUMN, "Guessed Gender")
    state_columns = _map_category(df["State ID"], STATE_TO_COLUMN, "State ID")
    problem_set_columns = _map_category(df["problem_set"], PROBLEM_SET_TO_COLUMN, "problem_set")

    user_id = _numeric(df["User ID"])
    problem_set_id = _numeric(df["problem_set"])

    result = pd.DataFrame(index=df.index)
    result["source_user_id"] = user_id.astype(int)
    result["source_problem_set_id"] = problem_set_id.astype(int)

    for source, column in zip(NUMERIC_HISTORY_FIELDS, NUMERIC_HISTORY_COLUMNS):
        result[column] = _numeric(df[source])

    for column in GRADE_COLUMNS:
        result[column] = grade_columns.eq(column).astype(int)
    for column in GENDER_COLUMNS:
        result[column] = gender_columns.eq(column).astype(int)
    for column in STATE_COLUMNS:
        result[column] = state_columns.eq(column).astype(int)
    for column in PROBLEM_SET_COLUMNS:
        result[column] = problem_set_columns.eq(column).astype(int)

    complete = _numeric(df["complete"])
    problem_count = _numeric(df["ProblemCount"])
    result["target_binary_complete"] = complete.astype(int)
    result["target_regression_problems"] = problem_count.astype(int)


    result = result.sort_values(["source_user_id", "source_problem_set_id"], kind="mergesort").reset_index(drop=True)
    result = result.reindex(columns=OUTPUT_COLUMNS)
    return result


def write_preprocessed_dataset(input_path: Path = INPUT_PATH, output_path: Path = OUTPUT_PATH) -> pd.DataFrame:
    frame, malformed = load_valid_rows(input_path)
    result = preprocess_dataframe(frame)
    result.to_csv(output_path, index=False)
    result.attrs["malformed_rows"] = malformed
    return result


DATASETS_ROOT = Path(__file__).resolve().parents[1]
class _ControlledExperimentsAdapter(BasePreparedAdapter):
    label = "assistments_controlled_experiments"
    adapter_kind = "artifact"
    student_column = "source_user_id"
    interaction_student_column = "source_user_id"
    attempt_column = None
    time_column = None
    time_unit = None
    filter_description = (
        "not_applicable: student/problem-set rows have no raw attempt or duration measures; "
        "ProblemCount is not an attempt count and stays untouched"
    )

    def __init__(self, module=None, input_path: Path | None = None):
        self._module = module
        self.input_path = input_path

    @property
    def module(self):
        if self._module is None:
            self._module = _load_dataset_module(
                "_prepared_assistments_controlled",
                DATASETS_ROOT / "assistments_from_controlled_experiments" / "preprocess.py",
            )
        return self._module

    def source_paths(self):
        return [self.input_path or self.module.INPUT_PATH]

    def load(self):
        frame, _malformed = self.module.load_valid_rows(self.input_path or self.module.INPUT_PATH)
        artifact = self.module.preprocess_dataframe(frame)
        return {
            "artifact": artifact,
            "provisional_students": sorted(artifact["source_user_id"].astype(str).unique().tolist()),
        }

    def interactions(self, context):
        raise ValueError("Artifact-mode adapter has no interaction pool")

    def build_rows(self, context, students, interaction_filter):
        if interaction_filter is not None:
            raise ValueError("Artifact-mode adapter must not receive an interaction filter")
        artifact = context["artifact"]
        return artifact.loc[artifact["source_user_id"].astype(str).isin(map(str, students))].copy()


def create_adapter():
    return _ControlledExperimentsAdapter()


def main():
    write_preprocessed_dataset()


if __name__ == "__main__":
    main()
