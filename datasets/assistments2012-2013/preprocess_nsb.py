import numpy as np
import pandas as pd
import sys
from pathlib import Path

from fold_preparation import BasePreparedAdapter
from dataset_registry import _load_dataset_module

MIN_INTERACTIONS = 5
AFFECT_COLUMNS = {
    "Average_confidence(FRUSTRATED)": "predicted_frustration_mean",
    "Average_confidence(CONFUSED)": "predicted_confusion_mean",
    "Average_confidence(CONCENTRATING)": "predicted_concentration_mean",
    "Average_confidence(BORED)": "predicted_boredom_mean",
}
ANSWER_TYPES = ("algebra", "choose_1", "choose_n", "fill_in_1")
TUTOR_MODES = ("tutor", "test", "pre", "post")
PROBLEM_SET_TYPES = {
    "LinearSection": "linear", "RandomIterateSection": "random_iterate",
    "RandomChildOrderSection": "random_child_order", "NumericLimitSection": "numeric_limit",
    "ChooseConditionSection": "choose_condition",
}
NON_SKILL_BUILDER_TYPES = (
    "LinearSection", "RandomChildOrderSection", "RandomIterateSection", "PlacementsSection", "ChooseConditionSection",
    "NumericLimitSection",
)
METADATA_COLUMNS = ("source_user_id", "source_order_id")
LAST_INTERACTION_COLUMNS = (
    "last_interaction_original", *(f"last_interaction_tutor_mode_{mode}" for mode in TUTOR_MODES),
    *(f"last_interaction_answer_type_{level}" for level in ANSWER_TYPES), "last_interaction_answer_type_other",
    *(f"last_interaction_problem_set_type_{name}" for name in PROBLEM_SET_TYPES.values()),
)
BASE_FEATURE_COLUMNS = (
    "interactions", "repeated_problem_rate", "scaffolding_problem_rate", "original_problem_count", "correct_rate",
    "hint_count_mean", "hint_count_total", "hint_use_rate", "bottom_hint_rate", "first_action_attempt_rate",
    "first_action_hint_rate", "first_action_scaffolding_rate", *LAST_INTERACTION_COLUMNS, "attempt_count_mean",
    "attempt_count_std", "attempt_count_total", "attempt_count_p90", "overlap_time_mean_seconds",
    "overlap_time_std_seconds", "overlap_time_total_seconds", "overlap_time_p90_seconds",
) + tuple(
    f"answer_type_{level}_{suffix}"
    for level in ANSWER_TYPES
    for suffix in ("rate",)
) + tuple(
    f"answer_type_{level}_{suffix}"
    for level in ANSWER_TYPES
    for suffix in ("correct_rate", "attempt_count_mean", "overlap_time_mean_seconds", "overlap_time_total_seconds")
) + (
    "previous_problem_correct", "previous_problem_attempt_count", "previous_problem_overlap_time_seconds",
)
ENRICHED_FEATURE_COLUMNS = (
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
FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + ENRICHED_FEATURE_COLUMNS + tuple(AFFECT_COLUMNS.values())
TARGET_COLUMNS = (
    "target_binary_correct", "target_regression_overlap_time", "target_regression_attempts",
)
OUTPUT_COLUMNS = METADATA_COLUMNS + FEATURE_COLUMNS + TARGET_COLUMNS
REQUIRED_SOURCE_COLUMNS = (
    "user_id", "problem_log_id", "problem_id", "start_time", "end_time", "problem_type", "original", "correct",
    "hint_count", "bottom_hint", "first_action", "attempt_count", "type", "tutor_mode", "skill_id", *AFFECT_COLUMNS,
)


def load_data(file_path):
    return pd.read_csv(file_path, usecols=list(REQUIRED_SOURCE_COLUMNS), encoding="latin-1", low_memory=False, memory_map=True)


def filter_non_skill_builder_rows(df: pd.DataFrame) -> pd.DataFrame:
    filtered_by_type = df.loc[df["type"].isin(NON_SKILL_BUILDER_TYPES)].reset_index(drop=True)
    return filtered_by_type


def normalize_category(value) -> str:
    if pd.isna(value):
        return "unknown"
    normalized = "".join(character.lower() if character.isalnum() else "_" for character in str(value).strip())
    return "_".join(part for part in normalized.split("_") if part) or "unknown"


def filter_interactions(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    frame["start_time"] = pd.to_datetime(frame["start_time"], format="mixed", errors="coerce")
    frame["end_time"] = pd.to_datetime(frame["end_time"], format="mixed", errors="coerce")
    durations = (frame["end_time"] - frame["start_time"]).dt.total_seconds()
    normalized_types = frame["problem_type"].map(normalize_category)
    inconsistent_correct_attempt = frame["correct"].eq(1) & frame["attempt_count"].ne(1)
    valid = (
        frame["correct"].isin([0, 1])
        & frame["attempt_count"].gt(0)
        & np.isfinite(durations)
        & durations.gt(0)
        & normalized_types.ne("open_response")
        & ~inconsistent_correct_attempt
        & frame["start_time"].notna()
        & frame["end_time"].notna()
    )
    filtered = frame.loc[valid].copy().reset_index(drop=True)
    filtered["overlap_time_seconds"] = durations.loc[valid].to_numpy()
    return filtered


def filter_by_interaction_count(df: pd.DataFrame, minimum: int = MIN_INTERACTIONS) -> pd.DataFrame:
    interaction_count = df.groupby("user_id")["problem_log_id"].transform("count")
    return df.loc[interaction_count.ge(minimum)].reset_index(drop=True)


def sort_interactions(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["user_id", "start_time", "problem_log_id"], kind="mergesort").reset_index(drop=True)


def split_history_and_targets(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = sort_interactions(df)
    target_indices = ordered.groupby("user_id", sort=False).tail(1).index
    targets = ordered.loc[target_indices].reset_index(drop=True)
    histories = ordered.drop(index=target_indices).reset_index(drop=True)
    return histories, targets


def prepare_student_data(df: pd.DataFrame, interaction_filter=None) -> tuple:
    df = filter_non_skill_builder_rows(df)
    filtered = filter_interactions(df)
    if interaction_filter is not None:
        filtered = interaction_filter(filtered)
    clean = filter_by_interaction_count(filtered)
    histories, targets = split_history_and_targets(clean)

    return histories, targets, len(filtered), int(clean["user_id"].nunique())


def numeric_summary(history: pd.DataFrame, source_column: str, mean_name: str, std_name: str) -> dict[str, float]:
    values = history[source_column].astype(float)
    return {mean_name: float(values.mean()), std_name: float(values.std(ddof=0))}


def summarize_history(history: pd.DataFrame) -> dict[str, float]:
    interactions = len(history)
    record = {
        "interactions": int(interactions),
        "repeated_problem_rate": float(1 - history["problem_id"].nunique() / interactions),
        "scaffolding_problem_rate": float(history["original"].eq(0).sum() / interactions),
        "original_problem_count": int(history["original"].eq(1).sum()),
        "correct_rate": float(history["correct"].mean()),
        "hint_count_mean": float(history["hint_count"].astype(float).mean()),
        "hint_count_total": float(history["hint_count"].astype(float).sum()),
        "hint_use_rate": float(history["hint_count"].astype(float).gt(0).fillna(False).mean()),
        "bottom_hint_rate": float(history["bottom_hint"].eq(1).mean()),
        "first_action_attempt_rate": float(history["first_action"].eq(0).mean()),
        "first_action_hint_rate": float(history["first_action"].eq(1).mean()),
        "first_action_scaffolding_rate": float(history["first_action"].eq(2).mean()),
    }
    record.update(numeric_summary(history, "attempt_count", "attempt_count_mean", "attempt_count_std"))
    record["attempt_count_total"] = float(history["attempt_count"].sum())
    record["attempt_count_p90"] = float(history["attempt_count"].quantile(0.90))
    record.update(numeric_summary(history, "overlap_time_seconds", "overlap_time_mean_seconds", "overlap_time_std_seconds"))
    record["overlap_time_total_seconds"] = float(history["overlap_time_seconds"].sum())
    record["overlap_time_p90_seconds"] = float(history["overlap_time_seconds"].quantile(0.90))
    for source_column, output_column in AFFECT_COLUMNS.items():
        record[output_column] = float(history[source_column].astype(float).mean())

    ordered = history.sort_values(["start_time", "problem_log_id"], kind="mergesort")
    previous = ordered.iloc[-1]
    record["previous_problem_correct"] = int(previous["correct"])
    record["previous_problem_attempt_count"] = int(previous["attempt_count"])
    record["previous_problem_overlap_time_seconds"] = float(previous["overlap_time_seconds"])
    return record


def summarize_answer_types(history: pd.DataFrame, active_levels: tuple[str, ...]) -> dict[str, float]:
    record = {}
    normalized_types = history["problem_type"].map(normalize_category)
    normalized_types = normalized_types.where(normalized_types.isin(active_levels), "unknown")
    for level in ANSWER_TYPES:
        level_rows = history.loc[normalized_types.eq(level)]
        record[f"answer_type_{level}_rate"] = float(len(level_rows) / len(history))
        if level_rows.empty:
            correct_rate = 0.0
            attempt_count_mean = 0.0
            overlap_time_mean_seconds = 0.0
            overlap_time_total_seconds = 0.0
        else:
            correct_rate = float(level_rows["correct"].mean())
            attempt_count_mean = float(level_rows["attempt_count"].mean())
            overlap_time_mean_seconds = float(level_rows["overlap_time_seconds"].mean())
            overlap_time_total_seconds = float(level_rows["overlap_time_seconds"].sum())
        record[f"answer_type_{level}_correct_rate"] = correct_rate
        record[f"answer_type_{level}_attempt_count_mean"] = attempt_count_mean
        record[f"answer_type_{level}_overlap_time_mean_seconds"] = overlap_time_mean_seconds
        record[f"answer_type_{level}_overlap_time_total_seconds"] = overlap_time_total_seconds
    return record


def encode_last_interaction(target: pd.Series) -> dict[str, float]:
    original = target["original"]
    record = {"last_interaction_original": int(original)}

    tutor_mode = str(target["tutor_mode"]).strip()
    record.update({f"last_interaction_tutor_mode_{mode}": int(tutor_mode == mode) for mode in TUTOR_MODES})

    problem_type = target["problem_type"]
    if pd.isna(problem_type) or str(problem_type).strip() not in ANSWER_TYPES:
        record.update({f"last_interaction_answer_type_{level}": 0 for level in ANSWER_TYPES})
        record["last_interaction_answer_type_other"] = 1
    else:
        problem_type_value = str(problem_type).strip()
        record.update({f"last_interaction_answer_type_{level}": int(problem_type_value == level) for level in ANSWER_TYPES})
        record["last_interaction_answer_type_other"] = 0

    problem_set_type = target["type"]
    problem_set_type = "" if pd.isna(problem_set_type) else str(problem_set_type).strip()
    record.update({
        f"last_interaction_problem_set_type_{name}": int(problem_set_type == raw)
        for raw, name in PROBLEM_SET_TYPES.items()
    })
    return record


def _ols_slope(values) -> float:
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return 0.0
    positions = np.arange(len(values), dtype=float)
    centered_positions = positions - positions.mean()
    denominator = float((centered_positions ** 2).sum())
    return 0.0 if denominator == 0 else float(centered_positions.dot(values - values.mean()) / denominator)


def parse_skill_ids(value):
    if pd.isna(value):
        return set()
    return {token.strip() for token in str(value).split(",") if token.strip()}


def summarize_additional_features(history: pd.DataFrame, target: pd.Series) -> dict[str, float]:
    ordered = history.sort_values(["start_time", "problem_log_id"], kind="mergesort")
    record = {}
    attempt_counts = ordered["attempt_count"].astype(float)
    overlap_times = ordered["overlap_time_seconds"].astype(float)

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
    record["overlap_time_median_seconds"] = float(overlap_times.median())
    record["overlap_time_p75_seconds"] = overlap_p75
    record["overlap_time_iqr_seconds"] = overlap_p75 - overlap_p25
    record["overlap_time_above_p90_rate"] = float(overlap_times.gt(overlap_p90).sum() / len(ordered))

    for window in (5,):
        recent = ordered.tail(window)
        record[f"recent_{window}_observation_count"] = int(len(recent))
        record[f"recent_{window}_correct_rate"] = float(recent["correct"].mean())
        record[f"recent_{window}_attempt_count_mean"] = float(recent["attempt_count"].mean())
        record[f"recent_{window}_overlap_time_mean_seconds"] = float(recent["overlap_time_seconds"].mean())
        record[f"recent_{window}_hint_use_rate"] = float(recent["hint_count"].astype(float).gt(0).fillna(False).mean())
        record[f"recent_{window}_attempt_count_median"] = float(recent["attempt_count"].median())
        record[f"recent_{window}_attempt_count_p90"] = float(recent["attempt_count"].quantile(0.90))
        record[f"recent_{window}_overlap_time_median_seconds"] = float(recent["overlap_time_seconds"].median())
        record[f"recent_{window}_overlap_time_p90_seconds"] = float(recent["overlap_time_seconds"].quantile(0.90))

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
            "selected_skill_overlap_time_mean_seconds": float(matched["overlap_time_seconds"].mean()),
            "selected_skill_hint_use_rate": float(matched["hint_count"].astype(float).gt(0).fillna(False).mean()),
        })

    recent_10 = ordered.tail(10)
    record["recent_10_correct_trend"] = _ols_slope(recent_10["correct"].to_numpy())
    record["recent_10_log1p_attempt_count_trend"] = _ols_slope(np.log1p(recent_10["attempt_count"].astype(float).to_numpy()))
    record["recent_10_log1p_overlap_time_seconds_trend"] = _ols_slope(
        np.log1p(recent_10["overlap_time_seconds"].astype(float).to_numpy())
    )
    record["recent_10_hint_use_trend"] = _ols_slope(
        recent_10["hint_count"].astype(float).gt(0).fillna(False).astype(float).to_numpy()
    )
    return record


def build_student_instances(histories: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    records = []
    targets_by_user = targets.set_index("user_id")
    for user_id, history in histories.groupby("user_id", sort=True):
        history = history.sort_values(["start_time", "problem_log_id"], kind="mergesort").reset_index(drop=True)
        target = targets_by_user.loc[user_id]
        record = {
            "source_user_id": user_id,
            "source_order_id": int(target["problem_log_id"]),
        }
        summary = summarize_history(history)
        record.update(summary)
        record.update(summarize_answer_types(history, ANSWER_TYPES))
        record.update(encode_last_interaction(target))
        record.update(summarize_additional_features(history, target))
        record.update({
            "target_binary_correct": int(target["correct"]),
            "target_regression_overlap_time": float(target["overlap_time_seconds"]),
            "target_regression_attempts": float(target["attempt_count"]),
        })
        records.append(record)
    return pd.DataFrame.from_records(records).reindex(columns=list(OUTPUT_COLUMNS))


def preprocess_dataframe(df: pd.DataFrame, interaction_filter=None) -> pd.DataFrame:
    (
        histories,
        targets,
        filtered_interaction_count,
        provisional_user_count,
    ) = prepare_student_data(df, interaction_filter)
    result = build_student_instances(histories, targets)
    result = result.sort_values("source_user_id", kind="mergesort").reset_index(drop=True)
    result.attrs.update({
        "source_row_count": int(len(df)),
        "filtered_interaction_count": filtered_interaction_count,
        "provisional_user_count": provisional_user_count,
        "final_student_count": int(len(result)),
    })
    return result

DATASETS_ROOT = Path(__file__).resolve().parents[1]
_Base2012Adapter = _load_dataset_module('_prepared_assistments2012_split_regimes', DATASETS_ROOT / 'assistments2012-2013/preprocess_split_regimes.py')._Base2012Adapter
class _2012NSBAdapter(_Base2012Adapter):
    label = "assistments2012_nsb"
    branch_file = "preprocess_nsb.py"
    branch_module_name = "_prepared_assistments2012_nsb"
    regime = "non_skill_builder"
    student_column = "source_user_id"
    filter_description = "raw attempt_count P99.9 and log1p-IQR fences on overlap_time_seconds (seconds)"

    def _fixed_validity(self, frame):
        return self.branch_module.filter_interactions(self.branch_module.filter_non_skill_builder_rows(frame))

    def _preprocess(self, subset, interaction_filter):
        return self.branch_module.preprocess_dataframe(subset, interaction_filter=interaction_filter)


def create_adapter():
    return _2012NSBAdapter()
