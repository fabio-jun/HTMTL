from pathlib import Path
import numpy as np
import pandas as pd
import sys

from fold_preparation import BasePreparedAdapter
from dataset_registry import _load_dataset_module

INPUT_PATH = Path(__file__).resolve().parent / "nsb.csv"
OUTPUT_PATH = Path(__file__).resolve().parent / "nsb_preprocessed.csv"

MILLISECONDS_PER_SECOND = 1_000.0
MIN_INTERACTIONS = 5
ANSWER_TYPES = ("algebra", "choose_1", "choose_n", "fill_in_1")
TUTOR_MODES = ("tutor", "test", "pre", "post")
PROBLEM_SET_TYPES = {
    "LinearSection": "linear", "RandomIterateSection": "random_iterate",
    "RandomChildOrderSection": "random_child_order", "NumericLimitSection": "numeric_limit",
    "ChooseConditionSection": "choose_condition",
}
LAST_INTERACTION_COLUMNS = (
    "last_interaction_original", "last_interaction_tutor_mode_tutor", "last_interaction_tutor_mode_test",
    "last_interaction_tutor_mode_pre", "last_interaction_tutor_mode_post", "last_interaction_answer_type_algebra",
    "last_interaction_answer_type_choose_1", "last_interaction_answer_type_choose_n",
    "last_interaction_answer_type_fill_in_1", "last_interaction_answer_type_other",
    "last_interaction_problem_set_type_linear", "last_interaction_problem_set_type_random_iterate",
    "last_interaction_problem_set_type_random_child_order", "last_interaction_problem_set_type_numeric_limit",
    "last_interaction_problem_set_type_choose_condition",
)
METADATA_COLUMNS = ("source_user_id", "source_order_id")
BASE_FEATURE_COLUMNS = (
    "interactions", "repeated_problem_rate", "scaffolding_problem_rate", "original_problem_count", "correct_rate",
    "hint_count_mean", "hint_count_total", "hint_use_rate", "bottom_hint_rate", "first_action_attempt_rate",
    "first_action_hint_rate", "first_action_scaffolding_rate", *LAST_INTERACTION_COLUMNS, "attempt_count_mean",
    "attempt_count_std", "attempt_count_total", "attempt_count_p90", "overlap_time_mean_seconds",
    "overlap_time_std_seconds", "overlap_time_total_seconds", "overlap_time_p90_seconds", "answer_type_algebra_rate",
    "answer_type_choose_1_rate", "answer_type_choose_n_rate", "answer_type_fill_in_1_rate",
    "answer_type_algebra_correct_rate", "answer_type_algebra_attempt_count_mean",
    "answer_type_algebra_overlap_time_mean_seconds", "answer_type_algebra_overlap_time_total_seconds",
    "answer_type_choose_1_correct_rate", "answer_type_choose_1_attempt_count_mean",
    "answer_type_choose_1_overlap_time_mean_seconds", "answer_type_choose_1_overlap_time_total_seconds",
    "answer_type_choose_n_correct_rate", "answer_type_choose_n_attempt_count_mean",
    "answer_type_choose_n_overlap_time_mean_seconds", "answer_type_choose_n_overlap_time_total_seconds",
    "answer_type_fill_in_1_correct_rate", "answer_type_fill_in_1_attempt_count_mean",
    "answer_type_fill_in_1_overlap_time_mean_seconds", "answer_type_fill_in_1_overlap_time_total_seconds",
    "previous_problem_correct", "previous_problem_attempt_count", "previous_problem_overlap_time_seconds",
)
TARGET_COLUMNS = (
    "target_binary_correct", "target_regression_overlap_time", "target_regression_attempts",
)
BASE_OUTPUT_COLUMNS = METADATA_COLUMNS + BASE_FEATURE_COLUMNS + TARGET_COLUMNS
ADDITIONAL_FEATURE_COLUMNS = (
    "recent_5_correct_rate", "recent_5_attempt_count_mean", "recent_5_overlap_time_mean_seconds",
    "recent_5_hint_use_rate", "selected_skill_prior_interactions", "selected_skill_correct_rate",
    "selected_skill_attempt_count_mean", "selected_skill_overlap_time_mean_seconds", "selected_skill_hint_use_rate",
    "attempt_count_median", "attempt_count_p75", "attempt_count_iqr", "attempt_count_above_p90_rate",
    "overlap_time_median_seconds", "overlap_time_p75_seconds", "overlap_time_iqr_seconds",
    "overlap_time_above_p90_rate", "recent_5_observation_count", "recent_5_attempt_count_median",
    "recent_5_attempt_count_p90", "recent_5_overlap_time_median_seconds", "recent_5_overlap_time_p90_seconds",
    "selected_skill_has_history", "recent_10_correct_trend", "recent_10_log1p_attempt_count_trend",
    "recent_10_log1p_overlap_time_seconds_trend", "recent_10_hint_use_trend",
)
FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + ADDITIONAL_FEATURE_COLUMNS
OUTPUT_COLUMNS = METADATA_COLUMNS + FEATURE_COLUMNS + TARGET_COLUMNS


def load_data(file_path: Path = INPUT_PATH) -> pd.DataFrame:
    df = pd.read_csv(file_path, encoding="latin-1", low_memory=False,memory_map=True)
    return df.drop(columns=["Unnamed: 0"], errors="ignore")


def sort_interactions(df: pd.DataFrame) -> pd.DataFrame:
    sorted_df = (df.sort_values(["user_id", "order_id"], kind="mergesort").reset_index(drop=True))
    remaining_columns = [column for column in sorted_df.columns if column not in ("user_id", "order_id")]

    return sorted_df[["user_id", "order_id"] + remaining_columns]


def filter_by_interaction_count(df: pd.DataFrame, minimum: int = MIN_INTERACTIONS) -> pd.DataFrame:
    interaction_count = df.groupby("user_id")["order_id"].transform("count")
    return df.loc[interaction_count.ge(minimum)].reset_index(drop=True)


def filter_interactions(df: pd.DataFrame) -> pd.DataFrame:
    inconsistent_correct_attempt = (df["correct"].eq(1) & df["attempt_count"].gt(1))
    valid = (
        df["answer_type"].ne("open_response")
        & df["correct"].isin([0, 1])
        & df["attempt_count"].gt(0)
        & df["overlap_time"].notna()
        & df["overlap_time"].gt(0)
        & ~inconsistent_correct_attempt
    )
    filtered = df.loc[valid].copy().reset_index(drop=True)
    return filtered


def split_history_and_targets(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = sort_interactions(df)
    target_indices = ordered.groupby("user_id", sort=False).tail(1).index
    targets = ordered.loc[target_indices].reset_index(drop=True)
    histories = ordered.drop(index=target_indices).reset_index(drop=True)
    return histories, targets


def prepare_student_data(df: pd.DataFrame, interaction_filter=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    filtered_interactions = filter_interactions(df)
    if interaction_filter is not None:
        filtered_interactions = interaction_filter(filtered_interactions)
    clean = filter_by_interaction_count(filtered_interactions)
    histories, targets = split_history_and_targets(clean)

    return histories, targets


def normalize_category(value) -> str:
    if pd.isna(value):
        return "unknown"
    value = "".join(character.lower() if character.isalnum() else "_" for character in str(value).strip())
    return "_".join(part for part in value.split("_") if part) or "unknown"


def numeric_summary(history: pd.DataFrame, source_column: str, mean_name: str, std_name: str, divisor: float = 1.0) -> dict[str, float]:
    values = (history[source_column].dropna().astype(float) / divisor)
    if values.empty:
        return {mean_name: np.nan, std_name: np.nan}
    return {mean_name: float(values.mean()), std_name: float(values.std(ddof=0))}


def summarize_history(history: pd.DataFrame) -> dict[str, float]:
    record = {
        "interactions": int(len(history)),
        "repeated_problem_rate": float(1 - history["problem_id"].nunique() / len(history)),
        "scaffolding_problem_rate": float(history["original"].eq(0).sum() / len(history)),
        "original_problem_count": int(history["original"].eq(1).sum()),
        "correct_rate": float(history["correct"].sum() / len(history)),
        "hint_count_mean": float(history["hint_count"].mean()),
        "hint_count_total": float(history["hint_count"].sum()),
        "hint_use_rate": float(history["hint_count"].gt(0).fillna(False).mean()),
        "bottom_hint_rate": float(history["bottom_hint"].eq(1).fillna(False).sum() / len(history)),
        "first_action_attempt_rate": float(history["first_action"].eq(0).fillna(False).sum() / len(history)),
        "first_action_hint_rate": float(history["first_action"].eq(1).fillna(False).sum() / len(history)),
        "first_action_scaffolding_rate": float(history["first_action"].eq(2).fillna(False).sum() / len(history)),
    }

    record.update(numeric_summary(history, "attempt_count", "attempt_count_mean", "attempt_count_std"))
    record["attempt_count_total"] = float(history["attempt_count"].sum())
    record["attempt_count_p90"] = float(history["attempt_count"].quantile(0.90))
    record.update(numeric_summary(history, "overlap_time", "overlap_time_mean_seconds", "overlap_time_std_seconds", MILLISECONDS_PER_SECOND))
    record["overlap_time_total_seconds"] = float(history["overlap_time"].sum() / MILLISECONDS_PER_SECOND)
    record["overlap_time_p90_seconds"] = float(history["overlap_time"].quantile(0.90) / MILLISECONDS_PER_SECOND)

    ordered = history.sort_values("order_id", kind="mergesort")
    previous = ordered.iloc[-1]
    record["previous_problem_correct"] = int(previous["correct"])
    record["previous_problem_attempt_count"] = int(previous["attempt_count"])
    record["previous_problem_overlap_time_seconds"] = float(previous["overlap_time"] / MILLISECONDS_PER_SECOND)

    return record


def summarize_answer_types(history: pd.DataFrame, active_levels: tuple[str, ...]) -> dict[str, float]:
    record = {}
    values = history["answer_type"].map(normalize_category)
    values = values.where(values.isin(active_levels), "unknown")
    for level in ANSWER_TYPES:
        level_rows = history.loc[values.eq(level)]
        record[f"answer_type_{level}_rate"] = float(len(level_rows) / len(history))
        if level_rows.empty:
            correct_rate = 0.0
            attempt_count_mean = 0.0
            overlap_time_mean_seconds = 0.0
            overlap_time_total_seconds = 0.0
        else:
            correct_rate = float(level_rows["correct"].sum() / len(level_rows))
            attempt_count_mean = float(level_rows["attempt_count"].mean())
            overlap_time_mean_seconds = float(level_rows["overlap_time"].mean() / MILLISECONDS_PER_SECOND)
            overlap_time_total_seconds = float(level_rows["overlap_time"].sum() / MILLISECONDS_PER_SECOND)
        record[f"answer_type_{level}_correct_rate"] = correct_rate
        record[f"answer_type_{level}_attempt_count_mean"] = attempt_count_mean
        record[f"answer_type_{level}_overlap_time_mean_seconds"] = overlap_time_mean_seconds
        record[f"answer_type_{level}_overlap_time_total_seconds"] = overlap_time_total_seconds

    return record


def encode_last_interaction(target) -> dict[str, float]:
    original = target["original"]
    record = {"last_interaction_original": int(original)}

    tutor_mode = str(target["tutor_mode"]).strip()
    record.update({f"last_interaction_tutor_mode_{mode}": int(tutor_mode == mode) for mode in TUTOR_MODES})

    answer_type = str(target["answer_type"]).strip()
    record.update({f"last_interaction_answer_type_{level}": int(answer_type == level) for level in ANSWER_TYPES})
    record["last_interaction_answer_type_other"] = int(answer_type not in ANSWER_TYPES)

    section_type = str(target["type"]).strip()
    record.update({f"last_interaction_problem_set_type_{name}": int(section_type == raw) for raw, name in PROBLEM_SET_TYPES.items()})

    return record


def build_base_student_instances(histories: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    records = []
    targets_by_user = targets.set_index("user_id")
    for user_id, history in histories.groupby("user_id", sort=True):
        history = history.sort_values("order_id", kind="mergesort")
        target = targets_by_user.loc[user_id]
        record = {
            "source_user_id": user_id,
            "source_order_id": target["order_id"],
        }
        record.update(summarize_history(history))
        record.update(summarize_answer_types(history, ANSWER_TYPES))
        record.update(encode_last_interaction(target))
        record.update({
            "target_binary_correct": int(target["correct"]),
            "target_regression_overlap_time": float(target["overlap_time"] / MILLISECONDS_PER_SECOND),
            "target_regression_attempts": float(target["attempt_count"]),
        })
        records.append(record)

    return pd.DataFrame.from_records(records).reindex(columns=list(BASE_OUTPUT_COLUMNS))


def parse_skill_ids(value):
    if pd.isna(value):
        return set()
    return {token.strip() for token in str(value).split(",") if token.strip()}


def _ols_slope(values):
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    x_mean = x.mean()
    y_mean = values.mean()
    denominator = ((x - x_mean) ** 2).sum()
    if denominator == 0:
        return 0.0
    return float(((x - x_mean) * (values - y_mean)).sum() / denominator)


def summarize_additional_features(history, target):
    ordered = history.sort_values("order_id", kind="mergesort")
    record = {}

    attempt_counts = ordered["attempt_count"].astype(float)
    overlap_times = ordered["overlap_time"].astype(float)

    attempt_p25 = float(attempt_counts.quantile(0.25))
    attempt_p75 = float(attempt_counts.quantile(0.75))
    attempt_p90 = float(attempt_counts.quantile(0.90))
    record["attempt_count_median"] = float(attempt_counts.median())
    record["attempt_count_p75"] = attempt_p75
    record["attempt_count_iqr"] = attempt_p75 - attempt_p25
    record["attempt_count_above_p90_rate"] = float(attempt_counts.gt(attempt_p90).sum() / len(ordered))

    overlap_p25 = float(overlap_times.quantile(0.25))
    overlap_p75 = float(overlap_times.quantile(0.75))
    overlap_p90 = float(overlap_times.quantile(0.90))
    record["overlap_time_median_seconds"] = float(overlap_times.median() / MILLISECONDS_PER_SECOND)
    record["overlap_time_p75_seconds"] = overlap_p75 / MILLISECONDS_PER_SECOND
    record["overlap_time_iqr_seconds"] = (overlap_p75 - overlap_p25) / MILLISECONDS_PER_SECOND
    record["overlap_time_above_p90_rate"] = float(overlap_times.gt(overlap_p90).sum() / len(ordered))

    for window in (5,):
        recent = ordered.tail(window)
        count = len(recent)
        record[f"recent_{window}_observation_count"] = count
        record[f"recent_{window}_correct_rate"] = float(recent["correct"].mean())
        record[f"recent_{window}_attempt_count_mean"] = float(recent["attempt_count"].mean())
        record[f"recent_{window}_overlap_time_mean_seconds"] = float(recent["overlap_time"].mean() / MILLISECONDS_PER_SECOND)
        record[f"recent_{window}_hint_use_rate"] = float(recent["hint_count"].gt(0).fillna(False).mean())
        record[f"recent_{window}_attempt_count_median"] = float(recent["attempt_count"].median())
        record[f"recent_{window}_attempt_count_p90"] = float(recent["attempt_count"].quantile(0.90))
        record[f"recent_{window}_overlap_time_median_seconds"] = float(recent["overlap_time"].median() / MILLISECONDS_PER_SECOND)
        record[f"recent_{window}_overlap_time_p90_seconds"] = float(recent["overlap_time"].quantile(0.90) / MILLISECONDS_PER_SECOND)

    target_skill_ids = parse_skill_ids(target.get("skill_id"))
    matched = ordered.loc[ordered["skill_id"].map(lambda value: bool(parse_skill_ids(value) & target_skill_ids))] if target_skill_ids else ordered.iloc[0:0]
    matched_count = int(len(matched))
    record["selected_skill_prior_interactions"] = matched_count
    record["selected_skill_has_history"] = int(matched_count > 0)
    if matched.empty:
        record.update({
            "selected_skill_correct_rate": 0.0,
            "selected_skill_attempt_count_mean": 0.0,
            "selected_skill_overlap_time_mean_seconds": 0.0,
            "selected_skill_hint_use_rate": 0.0,
        })
    else:
        record.update({
            "selected_skill_correct_rate": float(matched["correct"].mean()),
            "selected_skill_attempt_count_mean": float(matched["attempt_count"].mean()),
            "selected_skill_overlap_time_mean_seconds": float(matched["overlap_time"].mean() / MILLISECONDS_PER_SECOND),
            "selected_skill_hint_use_rate": float(matched["hint_count"].gt(0).fillna(False).mean()),
        })

    recent_10 = ordered.tail(10)
    record["recent_10_correct_trend"] = _ols_slope(recent_10["correct"].to_numpy())
    record["recent_10_log1p_attempt_count_trend"] = _ols_slope(np.log1p(recent_10["attempt_count"].astype(float).to_numpy()))
    record["recent_10_log1p_overlap_time_seconds_trend"] = _ols_slope(np.log1p(recent_10["overlap_time"].astype(float).to_numpy() / MILLISECONDS_PER_SECOND))
    record["recent_10_hint_use_trend"] = _ols_slope(recent_10["hint_count"].gt(0).fillna(False).astype(float).to_numpy())

    return record


def build_student_instances(histories, targets):
    result = build_base_student_instances(histories, targets)
    targets_by_user = targets.set_index("user_id")
    additional_by_user = {}
    for user_id, history in histories.groupby("user_id", sort=True):
        target = targets_by_user.loc[user_id]
        additional_by_user[user_id] = summarize_additional_features(history, target)
    for column in ADDITIONAL_FEATURE_COLUMNS:
        result[column] = result["source_user_id"].map(lambda user_id: additional_by_user[user_id][column])
    return result.reindex(columns=list(OUTPUT_COLUMNS))


def preprocess_dataframe(df, interaction_filter=None):
    histories, targets = prepare_student_data(df, interaction_filter)
    result = build_student_instances(histories, targets)
    result = result.sort_values("source_user_id", kind="mergesort").reset_index(drop=True)
    return result


def write_preprocessed_dataset(input_path=INPUT_PATH, output_path=OUTPUT_PATH):
    result = preprocess_dataframe(load_data(input_path))
    result.to_csv(output_path, index=False)
    return result


DATASETS_ROOT = Path(__file__).resolve().parents[1]
class _2009NSBAdapter(BasePreparedAdapter):
    label = "assistments2009_nsb"
    student_column = "source_user_id"
    interaction_student_column = "user_id"
    attempt_column = "attempt_count"
    time_column = "overlap_time"
    time_unit = "milliseconds"
    filter_description = "raw attempt_count P99.9 and log1p-IQR fences on overlap_time (milliseconds)"

    def __init__(self, module=None, input_path: Path | None = None):
        self._module = module
        self.input_path = input_path

    @property
    def module(self):
        if self._module is None:
            self._module = _load_dataset_module(
                "_prepared_assistments2009_nsb",
                DATASETS_ROOT / "assistments2009-2010" / "preprocess_nsb.py",
            )
        return self._module

    def source_paths(self):
        return [self.input_path or self.module.INPUT_PATH]

    def load(self):
        raw = self.module.load_data(self.input_path or self.module.INPUT_PATH)
        artifact = self.module.preprocess_dataframe(raw)
        return {
            "raw": raw,
            "artifact": artifact,
            "provisional_students": sorted(artifact["source_user_id"].astype(str).unique().tolist()),
        }

    def interactions(self, context):
        return self.module.filter_interactions(context["raw"])

    def build_rows(self, context, students, interaction_filter):
        subset = context["raw"].loc[context["raw"]["user_id"].astype(str).isin(map(str, students))]
        return self.module.preprocess_dataframe(subset, interaction_filter=interaction_filter)


def create_adapter():
    return _2009NSBAdapter()


def main():
    write_preprocessed_dataset()


if __name__ == "__main__":
    main()
