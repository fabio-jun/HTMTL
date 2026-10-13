import sys
from pathlib import Path

import numpy as np
import pandas as pd


from fold_preparation import BasePreparedAdapter
from dataset_registry import _load_dataset_module

DIR_PATH = Path(__file__).resolve().parent
DATASETS_ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = DIR_PATH / "PJ Data (Features Added).csv"
ACTIONS_PATH = DIR_PATH / "PJ Data (Actions).csv"
OUTPUT_PATH = DIR_PATH / "pj_student_preprocessed.csv"

MIN_HISTORY = 4

INTERACTION_KEY = ("userId", "assignmentId", "assistmentId", "problemId")
INTERACTION_COLUMNS = (
    "skill", "problemId", "userId", "assignmentId", "assistmentId", "startTime", "endTime", "timeTaken", "correct",
    "original", "hint", "hintCount", "scaffold", "bottomHint", "attemptCount", "problemType", "frWorkingInSchool",
)
ACTION_COLUMNS = (
    "userId", "assignmentId", "assistmentId", "problemId", "hintCount", "bottomHint", "attemptCount",
)
PROBLEM_TYPE_MAP = {
    "algebra": "problem_type_algebra", "choose_1": "problem_type_choose_one", "fill_in_1": "problem_type_fill_in",
}
ANSWER_TYPES = ("algebra", "choose_1", "fill_in_1")

ID_COLUMNS = ("user_id", "assignment_id", "problem_id")
METADATA_COLUMNS = ()
TARGET_COLUMNS = (
    "target_binary_first_correct", "target_regression_first_response_time",
)
FEATURE_COLUMNS = (
    "interactions", "repeated_problem_rate", "scaffolding_problem_rate", "original_problem_count", "correct_rate",
    "hint_count_mean", "hint_count_total", "hint_use_rate", "bottom_hint_rate", "first_action_attempt_rate",
    "first_action_hint_rate", "first_action_scaffolding_rate", "last_interaction_original", "during_school",
    "problem_type_algebra", "problem_type_choose_one", "problem_type_fill_in", "attempt_count_mean",
    "attempt_count_std", "attempt_count_total", "attempt_count_p90", "first_response_time_mean_seconds",
    "first_response_time_std_seconds", "first_response_time_total_seconds", "first_response_time_p90_seconds",
    "answer_type_algebra_rate", "answer_type_algebra_correct_rate", "answer_type_algebra_attempt_count_mean",
    "answer_type_algebra_first_response_time_mean_seconds", "answer_type_algebra_first_response_time_total_seconds",
    "answer_type_choose_1_rate", "answer_type_choose_1_correct_rate", "answer_type_choose_1_attempt_count_mean",
    "answer_type_choose_1_first_response_time_mean_seconds", "answer_type_choose_1_first_response_time_total_seconds",
    "answer_type_fill_in_1_rate", "answer_type_fill_in_1_correct_rate", "answer_type_fill_in_1_attempt_count_mean",
    "answer_type_fill_in_1_first_response_time_mean_seconds", "answer_type_fill_in_1_first_response_time_total_seconds",
    "previous_problem_correct", "previous_problem_attempt_count", "previous_problem_first_response_time_seconds",
    "recent_5_correct_rate", "recent_5_attempt_count_mean", "recent_5_first_response_time_mean_seconds",
    "recent_5_hint_use_rate", "selected_skill_prior_interactions", "selected_skill_correct_rate",
    "selected_skill_attempt_count_mean", "selected_skill_first_response_time_mean_seconds",
    "selected_skill_hint_use_rate", "attempt_count_median", "attempt_count_p75", "attempt_count_iqr",
    "attempt_count_above_p90_rate", "first_response_time_median_seconds", "first_response_time_p75_seconds",
    "first_response_time_iqr_seconds", "first_response_time_above_p90_rate", "recent_5_observation_count",
    "recent_5_attempt_count_median", "recent_5_attempt_count_p90", "recent_5_first_response_time_median_seconds",
    "recent_5_first_response_time_p90_seconds", "selected_skill_has_history", "recent_10_correct_trend",
    "recent_10_log1p_attempt_count_trend", "recent_10_log1p_first_response_time_seconds_trend",
    "recent_10_hint_use_trend",
)
OUTPUT_COLUMNS = ID_COLUMNS + METADATA_COLUMNS + FEATURE_COLUMNS + TARGET_COLUMNS


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _clean_problem_type(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.strip('"').str.strip().fillna("")


def load_features(file_path=FEATURES_PATH) -> pd.DataFrame:
    return pd.read_csv(file_path, usecols=list(INTERACTION_COLUMNS))


def load_actions(file_path=ACTIONS_PATH) -> pd.DataFrame:
    return pd.read_csv(file_path, usecols=list(ACTION_COLUMNS))


def normalize_interactions(features_df: pd.DataFrame) -> pd.DataFrame:
    working = features_df.copy()

    cleaned_type = _clean_problem_type(working["problemType"])
    working = working[cleaned_type.ne("open_response")].copy()

    time_taken = _numeric(working["timeTaken"])
    working = working[np.isfinite(time_taken) & (time_taken > 0)].copy()

    cleaned_type = _clean_problem_type(working["problemType"])

    for source in INTERACTION_KEY:
        series = _numeric(working[source])
        working[source] = series.astype("int64")

    for column in ("correct", "original", "frWorkingInSchool", "hint", "scaffold"):
        series = _numeric(working[column])
        working[column] = series.astype("int64")

    for column in ("startTime", "endTime"):
        series = _numeric(working[column])
        working[column] = series

    working["_problem_type"] = cleaned_type
    working["_time_taken"] = _numeric(working["timeTaken"])
    return working.reset_index(drop=True)


def aggregate_actions(actions_df: pd.DataFrame) -> pd.DataFrame:
    working = actions_df.copy()

    for source in INTERACTION_KEY:
        series = _numeric(working[source])
        working[source] = series.astype("int64")

    for column in ("attemptCount", "hintCount"):
        series = _numeric(working[column])
        working[column] = series

    bottom_hint = _numeric(working["bottomHint"])
    working["bottomHint"] = bottom_hint

    agg = (
        working.groupby(list(INTERACTION_KEY), as_index=False)
        .agg(
            attempt_count=("attemptCount", "max"),
            hint_count=("hintCount", "max"),
            bottom_hint_used=("bottomHint", "max"),
        )
    )
    return agg


def join_action_aggregates(features_df: pd.DataFrame, actions_df: pd.DataFrame) -> pd.DataFrame:
    working = normalize_interactions(features_df)
    agg = aggregate_actions(actions_df)
    merged = working.merge(agg, on=list(INTERACTION_KEY), how="left", indicator=True)
    merged = merged.drop(columns=["_merge"])
    return merged


def freeze_targets(interactions: pd.DataFrame) -> pd.DataFrame:
    users = np.sort(interactions["userId"].unique())
    records = []
    for user in users:
        user_rows = interactions.loc[interactions["userId"] == user]
        max_start = user_rows["startTime"].max()
        frozen = user_rows.loc[user_rows["startTime"] == max_start]
        records.append(frozen.iloc[0])
    return pd.DataFrame(records)


def build_histories(interactions: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    target_start = targets.set_index("userId")["startTime"]
    interactions = interactions.copy()
    interactions["_target_start"] = interactions["userId"].map(target_start)
    history = interactions.loc[interactions["endTime"] < interactions["_target_start"]].copy()
    history = history.drop(columns=["_target_start"]).reset_index(drop=True)
    counts = history.groupby("userId").size()
    eligible = counts[counts >= MIN_HISTORY].index
    return history.loc[history["userId"].isin(eligible)].reset_index(drop=True)


def _completion_order(history: pd.DataFrame) -> pd.DataFrame:
    return history.sort_values(
        ["endTime", "startTime", "assignmentId", "assistmentId", "problemId"],
        kind="mergesort",
    ).reset_index(drop=True)


def _ols_slope(values) -> float:
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


def summarize_history(history: pd.DataFrame, target) -> dict:
    n = len(history)
    ordered = _completion_order(history)
    record = {
        "interactions": int(n),
        "repeated_problem_rate": float(1 - history["problemId"].nunique() / n),
        "scaffolding_problem_rate": float(history["original"].eq(0).sum() / n),
        "original_problem_count": int(history["original"].eq(1).sum()),
        "correct_rate": float(history["correct"].sum() / n),
        "hint_count_mean": float(history["hint_count"].mean()),
        "hint_count_total": float(history["hint_count"].sum()),
        "hint_use_rate": float(history["hint_count"].gt(0).mean()),
        "bottom_hint_rate": float(history["bottom_hint_used"].eq(1).sum() / n),
        "first_action_attempt_rate": float((history["hint"].eq(0) & history["scaffold"].eq(0)).sum() / n),
        "first_action_hint_rate": float(history["hint"].eq(1).sum() / n),
        "first_action_scaffolding_rate": float(history["scaffold"].eq(1).sum() / n),
    }

    attempt = history["attempt_count"].astype(float)
    time = history["_time_taken"].astype(float)
    record["attempt_count_mean"] = float(attempt.mean())
    record["attempt_count_std"] = float(attempt.std(ddof=0))
    record["attempt_count_total"] = float(attempt.sum())
    record["attempt_count_p90"] = float(attempt.quantile(0.90))
    record["first_response_time_mean_seconds"] = float(time.mean())
    record["first_response_time_std_seconds"] = float(time.std(ddof=0))
    record["first_response_time_total_seconds"] = float(time.sum())
    record["first_response_time_p90_seconds"] = float(time.quantile(0.90))

    ptype = history["_problem_type"]
    for level in ANSWER_TYPES:
        level_rows = history.loc[ptype.eq(level)]
        record[f"answer_type_{level}_rate"] = float(len(level_rows) / n)
        if level_rows.empty:
            record[f"answer_type_{level}_correct_rate"] = 0.0
            record[f"answer_type_{level}_attempt_count_mean"] = 0.0
            record[f"answer_type_{level}_first_response_time_mean_seconds"] = 0.0
            record[f"answer_type_{level}_first_response_time_total_seconds"] = 0.0
        else:
            record[f"answer_type_{level}_correct_rate"] = float(level_rows["correct"].sum() / len(level_rows))
            record[f"answer_type_{level}_attempt_count_mean"] = float(level_rows["attempt_count"].mean())
            record[f"answer_type_{level}_first_response_time_mean_seconds"] = float(level_rows["_time_taken"].mean())
            record[f"answer_type_{level}_first_response_time_total_seconds"] = float(level_rows["_time_taken"].sum())

    previous = ordered.iloc[-1]
    record["previous_problem_correct"] = int(previous["correct"])
    record["previous_problem_attempt_count"] = int(previous["attempt_count"])
    record["previous_problem_first_response_time_seconds"] = float(previous["_time_taken"])

    attempt_p25 = float(attempt.quantile(0.25))
    attempt_p75 = float(attempt.quantile(0.75))
    attempt_p90 = float(attempt.quantile(0.90))
    time_p25 = float(time.quantile(0.25))
    time_p75 = float(time.quantile(0.75))
    time_p90 = float(time.quantile(0.90))
    record["attempt_count_median"] = float(attempt.median())
    record["attempt_count_p75"] = attempt_p75
    record["attempt_count_iqr"] = attempt_p75 - attempt_p25
    record["attempt_count_above_p90_rate"] = float(attempt.gt(attempt_p90).sum() / n)
    record["first_response_time_median_seconds"] = float(time.median())
    record["first_response_time_p75_seconds"] = time_p75
    record["first_response_time_iqr_seconds"] = time_p75 - time_p25
    record["first_response_time_above_p90_rate"] = float(time.gt(time_p90).sum() / n)

    recent_5 = ordered.tail(5)
    count_5 = len(recent_5)
    record["recent_5_observation_count"] = count_5
    record["recent_5_correct_rate"] = float(recent_5["correct"].mean())
    record["recent_5_attempt_count_mean"] = float(recent_5["attempt_count"].mean())
    record["recent_5_first_response_time_mean_seconds"] = float(recent_5["_time_taken"].mean())
    record["recent_5_hint_use_rate"] = float(recent_5["hint_count"].gt(0).mean())
    record["recent_5_attempt_count_median"] = float(recent_5["attempt_count"].median())
    record["recent_5_attempt_count_p90"] = float(recent_5["attempt_count"].quantile(0.90))
    record["recent_5_first_response_time_median_seconds"] = float(recent_5["_time_taken"].median())
    record["recent_5_first_response_time_p90_seconds"] = float(recent_5["_time_taken"].quantile(0.90))

    target_skill = str(target["skill"]).strip()
    matched = history.loc[history["skill"].astype("string").str.strip().eq(target_skill)]
    matched_count = int(len(matched))
    record["selected_skill_prior_interactions"] = matched_count
    record["selected_skill_has_history"] = int(matched_count > 0)
    if matched.empty:
        record.update({
            "selected_skill_correct_rate": 0.0,
            "selected_skill_attempt_count_mean": 0.0,
            "selected_skill_first_response_time_mean_seconds": 0.0,
            "selected_skill_hint_use_rate": 0.0,
        })
    else:
        record.update({
            "selected_skill_correct_rate": float(matched["correct"].mean()),
            "selected_skill_attempt_count_mean": float(matched["attempt_count"].mean()),
            "selected_skill_first_response_time_mean_seconds": float(matched["_time_taken"].mean()),
            "selected_skill_hint_use_rate": float(matched["hint_count"].gt(0).mean()),
        })

    recent_10 = ordered.tail(10)
    record["recent_10_correct_trend"] = _ols_slope(recent_10["correct"].to_numpy())
    record["recent_10_log1p_attempt_count_trend"] = _ols_slope(
        np.log1p(recent_10["attempt_count"].astype(float).to_numpy())
    )
    record["recent_10_log1p_first_response_time_seconds_trend"] = _ols_slope(
        np.log1p(recent_10["_time_taken"].astype(float).to_numpy())
    )
    record["recent_10_hint_use_trend"] = _ols_slope(
        recent_10["hint_count"].gt(0).astype(float).to_numpy()
    )
    return record


def build_student_rows(interactions: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    targets_by_user = targets.set_index("userId")
    history = build_histories(interactions, targets)
    records = []
    for user_id, group in history.groupby("userId", sort=True):
        target = targets_by_user.loc[user_id]
        record = {
            "user_id": int(user_id),
            "assignment_id": int(target["assignmentId"]),
            "problem_id": int(target["problemId"]),
            "last_interaction_original": int(target["original"]),
            "during_school": int(target["frWorkingInSchool"]),
        }
        ptype = str(target["_problem_type"]).strip()
        for source_label, feature in PROBLEM_TYPE_MAP.items():
            record[feature] = int(ptype == source_label)
        record.update(summarize_history(group, target))
        record["target_binary_first_correct"] = int(target["correct"])
        record["target_regression_first_response_time"] = float(target["_time_taken"])
        records.append(record)
    frame = pd.DataFrame.from_records(records)
    for column in list(OUTPUT_COLUMNS):
        if column not in frame.columns:
            frame[column] = pd.Series(dtype="float64")
    return frame.reindex(columns=list(OUTPUT_COLUMNS))


def preprocess_dataframes(features_df, actions_df, interaction_filter=None):
    interactions = join_action_aggregates(features_df, actions_df)
    if interaction_filter is not None:
        interactions = interaction_filter(interactions)
    targets = freeze_targets(interactions)
    result = build_student_rows(interactions, targets)
    result = result.sort_values("user_id", kind="mergesort").reset_index(drop=True)
    result = result.reindex(columns=list(OUTPUT_COLUMNS))
    return result


def write_preprocessed_dataset(features_path=FEATURES_PATH, actions_path=ACTIONS_PATH, output_path=OUTPUT_PATH):
    result = preprocess_dataframes(load_features(features_path), load_actions(actions_path))
    result.to_csv(output_path, index=False)
    return result


class _PJAdapter(BasePreparedAdapter):
    label = "assistments_paper_2011"
    student_column = "user_id"
    interaction_student_column = "userId"
    attempt_column = "attempt_count"
    time_column = "_time_taken"
    time_unit = "seconds"
    filter_description = (
        "reconstructed attempt_count P99.9 and log1p-IQR fences on timeTaken (seconds), "
        "applied to joined problem interactions before freeze_targets/build_student_rows"
    )

    def __init__(self, module=None, features_path: Path | None = None, actions_path: Path | None = None):
        self._module = module
        self.features_path = features_path
        self.actions_path = actions_path

    @property
    def module(self):
        if self._module is None:
            self._module = _load_dataset_module(
                "_prepared_assistments_paper_2011",
                DATASETS_ROOT / "assistments_paper:2011_goldstein" / "preprocess.py",
            )
        return self._module

    def source_paths(self):
        return [
            self.features_path or self.module.FEATURES_PATH,
            self.actions_path or self.module.ACTIONS_PATH,
        ]

    def load(self):
        features = self.module.load_features(self.features_path or self.module.FEATURES_PATH)
        actions = self.module.load_actions(self.actions_path or self.module.ACTIONS_PATH)
        artifact = self.module.preprocess_dataframes(features, actions)
        return {
            "features": features,
            "actions": actions,
            "artifact": artifact,
            "provisional_students": sorted(artifact["user_id"].astype(str).unique().tolist()),
        }

    def interactions(self, context):
        return self.module.join_action_aggregates(context["features"], context["actions"])

    def build_rows(self, context, students, interaction_filter):
        features = context["features"]
        actions = context["actions"]
        features_subset = features.loc[features["userId"].astype(str).isin(map(str, students))]
        actions_subset = actions.loc[actions["userId"].astype(str).isin(map(str, students))]
        return self.module.preprocess_dataframes(features_subset, actions_subset, interaction_filter=interaction_filter)


def create_adapter():
    return _PJAdapter()


def main():
    result = write_preprocessed_dataset()
    print(f"student rows: {len(result)}")
    print(f"features: {len(FEATURE_COLUMNS)} targets: {len(TARGET_COLUMNS)} columns: {len(OUTPUT_COLUMNS)}")


if __name__ == "__main__":
    main()
