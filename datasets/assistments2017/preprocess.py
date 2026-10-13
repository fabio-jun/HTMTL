import csv
from pathlib import Path
import numpy as np
import pandas as pd
import sys

from fold_preparation import BasePreparedAdapter
from dataset_registry import _load_dataset_module

INPUT_PATH = Path(__file__).resolve().parent / "Assistments2017.csv"
OUTPUT_PATH = Path(__file__).resolve().parent / "assistments2017_preprocessed.csv"
DIRECT_FEATURE_SOURCES = {
    "AveKnow": "AveKnow",
    "AveCarelessness": "AveCarelessness",
    "AveCorrect": "AveCorrect",
    "NumActions": "NumActions",
    "AveResBored": "predicted_boredom_mean",
    "AveResEngcon": "predicted_concentration_mean",
    "AveResConf": "predicted_confusion_mean",
    "AveResFrust": "predicted_frustration_mean",
    "AveResOfftask": "predicted_off_task_mean",
    "AveResGaming": "predicted_gaming_mean",
}
DIRECT_FEATURE_COLUMNS = tuple(DIRECT_FEATURE_SOURCES.values())
PROBLEMTYPE_GROUPS = ("textfield", "radio", "algebra_field", "algebra", "popup_menu", "checkbox", "other", "unknown")
PROBLEMTYPE_RATE_COLUMNS = tuple(f"problemType_{group}_rate" for group in PROBLEMTYPE_GROUPS)
EXTRACTED_FEATURE_COLUMNS = (
    "skill_distinct_count", "problemId_distinct_count", "assignmentId_distinct_count", "assistmentId_distinct_count",
    "timeTaken_mean_seconds", "timeTaken_median_seconds", "active_days", "usage_span_days", "original_rate",
    "hint_rate", "scaffold_rate", "frIsHelpRequest_rate", "stlHintUsed_rate", "endsWithScaffolding_rate",
    "endsWithAutoScaffolding_rate", "frWorkingInSchool_rate", "responseIsFillIn_rate", "responseIsChosen_rate",
    "Ln_mean",
) + PROBLEMTYPE_RATE_COLUMNS
USAGE_YEAR_MAP = {
    "2004-2005": 0, "2005-2006": 1,
}
PRE_MCAS_CUTOFFS = {
    0: pd.Timestamp("2005-05-13", tz="America/New_York").timestamp(),
    1: pd.Timestamp("2006-05-13", tz="America/New_York").timestamp(),
}
FEATURE_COLUMNS = ("usage_year_2005_2006",) + DIRECT_FEATURE_COLUMNS + EXTRACTED_FEATURE_COLUMNS
TARGET_COLUMNS = ("target_regression_mcas", "target_binary_enrolled")
OUTPUT_COLUMNS = ("source_student_id",) + FEATURE_COLUMNS + TARGET_COLUMNS
DIRECT_CHECK_COLUMNS = (
    "SY ASSISTments Usage",
) + tuple(DIRECT_FEATURE_SOURCES) + ("MCAS", "Enrolled", "Selective", "isSTEM")
PROBLEMTYPE_ALIASES = {
    "textfieldquestion": "textfield", "interfacetextfieldquestion": "textfield",
    "interfacetextfieldquestion1": "textfield", "radioquestion": "radio", "interfaceradioquestion": "radio",
    "interfaceradioquestion1": "radio", "algebrafieldquestion": "algebra_field",
    "algebrafieldquestion1": "algebra_field", "algebra": "algebra", "popupmenuquestion": "popup_menu",
    "interfacepopupmenuquestion1": "popup_menu", "checkboxquestion": "checkbox", "other": "other",
    "noprobtype": "unknown", "0": "unknown", "1": "unknown",
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


def normalize_targets(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized["MCAS"] = _numeric(normalized["MCAS"]).replace(-999, np.nan)
    normalized["Enrolled"] = _numeric(normalized["Enrolled"])
    return normalized


def filter_target_cohort(df: pd.DataFrame) -> pd.DataFrame:
    valid_mcas = df["MCAS"].notna() & np.isfinite(df["MCAS"])
    valid_enrolled = df["Enrolled"].isin([0, 1])
    return df.loc[valid_mcas & valid_enrolled].copy()


def _constant_value(values: pd.Series, student_id, column):
    return values.iloc[0]


def map_usage_year(value) -> int:
    key = str(value).strip()
    return USAGE_YEAR_MAP[key]


def collect_student_constants(df: pd.DataFrame) -> dict:
    constants = {}
    for student_id, group in df.groupby("studentId", sort=False):
        constants[student_id] = {
            column: _constant_value(group[column], student_id, column)
            for column in DIRECT_CHECK_COLUMNS
        }
    return constants


def filter_pre_mcas_actions(df: pd.DataFrame, constants_by_student: dict) -> pd.DataFrame:
    usage_years = df["studentId"].map(lambda student_id: map_usage_year(constants_by_student[student_id]["SY ASSISTments Usage"]))
    cutoffs = usage_years.map(PRE_MCAS_CUTOFFS)
    return df.loc[_numeric(df["startTime"]) < cutoffs].copy()


def normalize_problem_type(value) -> str:
    if pd.isna(value) or str(value).strip() == "":
        return "unknown"
    return PROBLEMTYPE_ALIASES.get(str(value).strip().lower(), "unknown")


def _valid_binary_mean(group: pd.DataFrame, column: str, student_id) -> float:
    values = _numeric(group[column])
    values = values[values.isin([0, 1])]
    return float(values.mean())


def _finite_numeric_values(group: pd.DataFrame, column: str, student_id) -> pd.Series:
    values = _numeric(group[column])
    values = values[np.isfinite(values)]
    return values


def _student_record(student_id, group: pd.DataFrame, constants: dict) -> dict:
    record = {"source_student_id": student_id}
    record["usage_year_2005_2006"] = map_usage_year(constants["SY ASSISTments Usage"])
    for source_column, output_column in DIRECT_FEATURE_SOURCES.items():
        record[output_column] = float(constants[source_column])
    record["target_regression_mcas"] = float(constants["MCAS"])
    record["target_binary_enrolled"] = int(float(constants["Enrolled"]))
    record.update({
        "skill_distinct_count": int(group["skill"].replace("", np.nan).nunique(dropna=True)),
        "problemId_distinct_count": int(group["problemId"].replace("", np.nan).nunique(dropna=True)),
        "assignmentId_distinct_count": int(group["assignmentId"].replace("", np.nan).nunique(dropna=True)),
        "assistmentId_distinct_count": int(group["assistmentId"].replace("", np.nan).nunique(dropna=True)),
    })
    time_values = _finite_numeric_values(group, "timeTaken", student_id)
    record["timeTaken_mean_seconds"] = float(time_values.mean())
    record["timeTaken_median_seconds"] = float(time_values.median())
    starts = _finite_numeric_values(group, "startTime", student_id)
    ends = _finite_numeric_values(group, "endTime", student_id)
    record["active_days"] = int(pd.to_datetime(starts, unit="s", utc=True).dt.date.nunique())
    record["usage_span_days"] = float((ends.max() - starts.min()) / 86_400.0)
    for column in ("original", "hint", "scaffold", "frIsHelpRequest", "stlHintUsed", "endsWithScaffolding", "endsWithAutoScaffolding", "frWorkingInSchool", "responseIsFillIn", "responseIsChosen"):
        record[f"{column}_rate"] = _valid_binary_mean(group, column, student_id)
    record["Ln_mean"] = float(_finite_numeric_values(group, "Ln", student_id).mean())
    normalized = group["problemType"].map(normalize_problem_type)
    for group_name in PROBLEMTYPE_GROUPS:
        record[f"problemType_{group_name}_rate"] = float(normalized.eq(group_name).mean())
    return record


def _sort_key(value):
    return (type(value).__name__, str(value))


def build_student_instances(working: pd.DataFrame, constants_by_student: dict,) -> pd.DataFrame:
    records = []
    for student_id, group in working.groupby("studentId", sort=False):
        record = _student_record(student_id, group, constants_by_student[student_id],)
        records.append(record)
    return pd.DataFrame.from_records(records)


def sort_student_instances(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values("source_student_id", key=lambda values: values.map(_sort_key), kind="mergesort",).reset_index(drop=True)


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    normalized = normalize_targets(df)
    constants_by_student = collect_student_constants(normalized)
    cohort = filter_target_cohort(normalized)
    result = build_student_instances(filter_pre_mcas_actions(cohort, constants_by_student), constants_by_student)
    result = sort_student_instances(result)
    result = result.reindex(columns=OUTPUT_COLUMNS)
    return result


def write_preprocessed_dataset(input_path: Path = INPUT_PATH, output_path: Path = OUTPUT_PATH) -> pd.DataFrame:
    frame, malformed = load_valid_rows(input_path)
    result = preprocess_dataframe(frame)
    result.to_csv(output_path, index=False)
    result.attrs["malformed_rows"] = malformed
    return result


DATASETS_ROOT = Path(__file__).resolve().parents[1]
class _2017Adapter(BasePreparedAdapter):
    label = "assistments2017"
    adapter_kind = "artifact"
    student_column = "source_student_id"
    interaction_student_column = "source_student_id"
    attempt_column = None
    time_column = None
    time_unit = None
    filter_description = "not_applicable: 2017 has no raw attempt/duration interaction filtering in scope; shared artifact path"

    def __init__(self, module=None, input_path: Path | None = None):
        self._module = module
        self.input_path = input_path

    @property
    def module(self):
        if self._module is None:
            self._module = _load_dataset_module(
                "_prepared_assistments2017",
                DATASETS_ROOT / "assistments2017" / "preprocess.py",
            )
        return self._module

    def source_paths(self):
        return [self.input_path or self.module.INPUT_PATH]

    def load(self):
        frame, _malformed = self.module.load_valid_rows(self.input_path or self.module.INPUT_PATH)
        artifact = self.module.preprocess_dataframe(frame)
        return {
            "artifact": artifact,
            "provisional_students": sorted(artifact["source_student_id"].astype(str).unique().tolist()),
        }

    def interactions(self, context):
        raise ValueError("Artifact-mode adapter has no interaction pool")

    def build_rows(self, context, students, interaction_filter):
        if interaction_filter is not None:
            raise ValueError("Artifact-mode adapter must not receive an interaction filter")
        artifact = context["artifact"]
        return artifact.loc[artifact["source_student_id"].astype(str).isin(map(str, students))].copy()


def create_adapter():
    return _2017Adapter()


def main():
    write_preprocessed_dataset()


if __name__ == "__main__":
    main()
