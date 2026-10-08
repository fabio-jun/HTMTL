from pathlib import Path
import multiprocessing
import os
import numpy as np
import pandas as pd
import sys

from fold_preparation import BasePreparedAdapter
from dataset_registry import _load_dataset_module

INPUT_PATH = Path(__file__).resolve().parent / "sb.csv"
OUTPUT_PATH = Path(__file__).resolve().parent / "sb_preprocessed.csv"
MILLISECONDS_PER_SECOND = 1_000.0
METADATA_COLUMNS = ("source_user_id", "source_assignment_id", "source_start_order_id")
FEATURE_COLUMNS = (
    "original_problem_count", "ended_assignment_count", "scaffolding_problem_rate", "interactions",
    "repeated_problem_rate", "correct_rate", "attempt_count_mean", "attempt_count_std", "attempt_count_total",
    "attempt_count_p90", "hint_count_mean", "hint_count_total", "hint_use_rate", "bottom_hint_rate",
    "first_action_attempt_rate", "first_action_hint_rate", "first_action_scaffolding_rate", "overlap_time_mean_seconds",
    "overlap_time_std_seconds", "overlap_time_total_seconds", "overlap_time_p90_seconds", "answer_type_algebra_rate",
    "answer_type_algebra_correct_rate", "answer_type_algebra_overlap_time_mean_seconds",
    "answer_type_algebra_overlap_time_total_seconds", "answer_type_choose_1_rate", "answer_type_choose_1_correct_rate",
    "answer_type_choose_1_overlap_time_mean_seconds", "answer_type_choose_1_overlap_time_total_seconds",
    "answer_type_choose_n_rate", "answer_type_choose_n_correct_rate", "answer_type_choose_n_overlap_time_mean_seconds",
    "answer_type_choose_n_overlap_time_total_seconds", "answer_type_fill_in_1_rate",
    "answer_type_fill_in_1_correct_rate", "answer_type_fill_in_1_overlap_time_mean_seconds",
    "answer_type_fill_in_1_overlap_time_total_seconds", "mastery_rate", "assignment_problem_count_mean",
    "assignment_problem_count_std", "assignment_problem_count_p90", "assignment_time_mean", "assignment_time_std",
    "assignment_time_p90", "previous_assignment_mastered", "previous_assignment_problem_count",
    "previous_assignment_time", "selected_assignment_answer_type_algebra_present",
    "selected_assignment_answer_type_choose_1_present", "selected_assignment_answer_type_choose_n_present",
    "selected_assignment_answer_type_fill_in_1_present", "attempt_count_median", "attempt_count_p75",
    "attempt_count_iqr", "attempt_count_above_p90_rate", "overlap_time_median_seconds", "overlap_time_p75_seconds",
    "overlap_time_iqr_seconds", "overlap_time_above_p90_rate", "recent_5_observation_count", "recent_5_correct_rate",
    "recent_5_attempt_count_mean", "recent_5_overlap_time_mean_seconds", "recent_5_hint_use_rate",
    "recent_5_attempt_count_median", "recent_5_attempt_count_p90", "recent_5_overlap_time_median_seconds",
    "recent_5_overlap_time_p90_seconds", "selected_skill_prior_interactions", "selected_skill_has_history",
    "selected_skill_correct_rate", "selected_skill_attempt_count_mean", "selected_skill_overlap_time_mean_seconds",
    "selected_skill_hint_use_rate", "recent_10_correct_trend", "recent_10_log1p_attempt_count_trend",
    "recent_10_log1p_overlap_time_seconds_trend", "recent_10_hint_use_trend",
)
TARGET_COLUMNS = (
    "target_binary_mastered", "target_regression_problems", "target_regression_work_time",
)
OUTPUT_COLUMNS = METADATA_COLUMNS + FEATURE_COLUMNS + TARGET_COLUMNS
PARALLEL_MIN_CANDIDATES = 2_000
_WORKER_SOURCE_ROWS = None
_WORKER_LIFECYCLE = None


def filter_source_rows(df: pd.DataFrame) -> pd.DataFrame:
    keep = (
        df["answer_type"].ne("open_response")
        & df["correct"].isin([0, 1])
        & ~(df["correct"].eq(1) & df["attempt_count"].gt(1))
        & ~df["attempt_count"].lt(0)
    )
    return df.loc[keep].reset_index(drop=True)


def build_assignment_lifecycle(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (user_id, assignment_id), group in df.groupby(["user_id", "assignment_id"], sort=False):
        ordered = group.sort_values("order_id", kind="mergesort")
        original = ordered.loc[ordered["original"].eq(1)]
        original_correct = original["correct"].eq(1).to_numpy()
        records.append({
            "user_id": user_id,
            "assignment_id": assignment_id,
            "first_order_id": ordered["order_id"].min(),
            "last_order_id": ordered["order_id"].max(),
            "mastered": int(len(original_correct) >= 3 and bool((original_correct[:-2] & original_correct[1:-1] & original_correct[2:]).any())),
            "problem_count": int(ordered["problem_id"].nunique()),
            "assignment_overlap_time_total_seconds": float(ordered["overlap_time"].sum() / MILLISECONDS_PER_SECOND),
            "duration_valid": bool(np.isfinite(ordered["overlap_time"].astype(float)).all() and ordered["overlap_time"].gt(0).all()),
        })
    return pd.DataFrame.from_records(records)


def select_eligible_assignments(lifecycle: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for _, group in lifecycle.groupby("user_id", sort=False):
        ordered = group.sort_values(["first_order_id", "assignment_id"], kind="mergesort").copy()
        ended_order_ids = np.sort(group["last_order_id"].to_numpy())
        ordered["prior_ended_assignment_count"] = np.searchsorted(ended_order_ids, ordered["first_order_id"].to_numpy(), side="left")
        pieces.append(ordered.loc[ordered["prior_ended_assignment_count"].ge(3)])
    return pd.concat(pieces, ignore_index=True) if pieces else lifecycle.iloc[0:0].copy()


def select_final_assignment_per_student(lifecycle: pd.DataFrame) -> pd.DataFrame:
    ordered = lifecycle.sort_values(["user_id", "first_order_id", "assignment_id"], kind="mergesort")
    return ordered.drop_duplicates(subset=["user_id"], keep="last").reset_index(drop=True)


def select_final_eligible_assignments(lifecycle: pd.DataFrame) -> pd.DataFrame:
    final_assignments = select_final_assignment_per_student(lifecycle)
    eligible = select_eligible_assignments(lifecycle)
    return final_assignments.merge(
        eligible[["user_id", "assignment_id"]], on=["user_id", "assignment_id"], how="inner"
    )


def filter_duration_complete_candidates(lifecycle: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    keep = []
    for row in candidates.itertuples(index=False):
        ended = lifecycle.loc[lifecycle["user_id"].eq(row.user_id) & lifecycle["last_order_id"].lt(row.first_order_id)]
        keep.append(bool(row.duration_valid and ended["duration_valid"].all()))
    return candidates.loc[keep].reset_index(drop=True)


def load_data(file_path: Path = INPUT_PATH) -> pd.DataFrame:
    return pd.read_csv(file_path, encoding="latin-1", low_memory=False, memory_map=True).drop(columns=["Unnamed: 0"], errors="ignore")


def _sample_std(values: pd.Series) -> float:
    return 0.0 if len(values) == 1 else float(values.std(ddof=1))


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


def summarize_ported_features(main_history, user_rows, target):
    ordered = main_history.sort_values("order_id", kind="mergesort")
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

    target_rows = user_rows.loc[user_rows["assignment_id"].eq(target["assignment_id"])]
    target_skill_ids = set()
    for value in target_rows["skill_id"]:
        target_skill_ids |= parse_skill_ids(value)
    if target_skill_ids:
        matched = ordered.loc[ordered["skill_id"].map(lambda value: bool(parse_skill_ids(value) & target_skill_ids))]
    else:
        matched = ordered.iloc[0:0]
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


def summarize_candidate(df: pd.DataFrame, lifecycle: pd.DataFrame, target: pd.Series) -> dict[str, float]:
    user_rows = df.loc[df["user_id"].eq(target["user_id"])]
    ended = lifecycle.loc[
        lifecycle["user_id"].eq(target["user_id"]) & lifecycle["last_order_id"].lt(target["first_order_id"])
    ].sort_values(["last_order_id", "assignment_id"], kind="mergesort")
    event_history = user_rows.loc[user_rows["assignment_id"].isin(ended["assignment_id"])]
    main_history = event_history.loc[event_history["original"].eq(1)]
    previous = ended.iloc[-1]
    attempts = main_history["attempt_count"].astype(float)
    hints = main_history["hint_count"].astype(float)
    times = event_history["overlap_time"].astype(float) / MILLISECONDS_PER_SECOND
    problem_counts = ended["problem_count"].astype(float)
    assignment_times = ended["assignment_overlap_time_total_seconds"].astype(float)
    record = {
        "original_problem_count": int(len(main_history)),
        "ended_assignment_count": int(len(ended)),
        "scaffolding_problem_rate": float(event_history["original"].eq(0).sum() / len(event_history)),
        "interactions": int(len(event_history)),
        "repeated_problem_rate": float(1 - event_history["problem_id"].nunique() / len(event_history)),
        "correct_rate": float(main_history["correct"].eq(1).sum() / len(main_history)),
        "attempt_count_mean": float(attempts.mean()),
        "attempt_count_std": _sample_std(attempts),
        "attempt_count_total": float(attempts.sum()),
        "attempt_count_p90": float(attempts.quantile(0.90)),
        "hint_count_mean": float(hints.mean()),
        "hint_count_total": float(hints.sum()),
        "hint_use_rate": float(hints.gt(0).sum() / len(main_history)),
        "bottom_hint_rate": float(main_history["bottom_hint"].eq(1).sum() / len(main_history)),
        "first_action_attempt_rate": float(main_history["first_action"].eq(0).sum() / len(main_history)),
        "first_action_hint_rate": float(main_history["first_action"].eq(1).sum() / len(main_history)),
        "first_action_scaffolding_rate": float(main_history["first_action"].eq(2).sum() / len(main_history)),
        "overlap_time_mean_seconds": float(times.mean()),
        "overlap_time_std_seconds": _sample_std(times),
        "overlap_time_total_seconds": float(times.sum()),
        "overlap_time_p90_seconds": float(times.quantile(0.90)),
        "mastery_rate": float(ended["mastered"].mean()),
        "assignment_problem_count_mean": float(problem_counts.mean()),
        "assignment_problem_count_std": _sample_std(problem_counts),
        "assignment_problem_count_p90": float(problem_counts.quantile(0.90)),
        "assignment_time_mean": float(assignment_times.mean()),
        "assignment_time_std": _sample_std(assignment_times),
        "assignment_time_p90": float(assignment_times.quantile(0.90)),
        "previous_assignment_mastered": int(previous["mastered"]),
        "previous_assignment_problem_count": int(previous["problem_count"]),
        "previous_assignment_time": float(previous["assignment_overlap_time_total_seconds"]),
    }
    answer_types = main_history["answer_type"]
    for level in ("algebra", "choose_1", "choose_n", "fill_in_1"):
        matching = main_history.loc[answer_types.eq(level)]
        record[f"answer_type_{level}_rate"] = float(len(matching) / len(main_history))
        record[f"answer_type_{level}_correct_rate"] = 0.0 if matching.empty else float(matching["correct"].eq(1).sum() / len(matching))
        record[f"answer_type_{level}_overlap_time_mean_seconds"] = 0.0 if matching.empty else float((matching["overlap_time"].astype(float) / MILLISECONDS_PER_SECOND).mean())
        record[f"answer_type_{level}_overlap_time_total_seconds"] = 0.0 if matching.empty else float((matching["overlap_time"].astype(float) / MILLISECONDS_PER_SECOND).sum())
    target_answer_types = user_rows.loc[user_rows["assignment_id"].eq(target["assignment_id"]), "answer_type"]
    for level in ("algebra", "choose_1", "choose_n", "fill_in_1"):
        record[f"selected_assignment_answer_type_{level}_present"] = int(target_answer_types.eq(level).any())
    record.update(summarize_ported_features(main_history, user_rows, target))
    return record


def build_candidate_instance(df: pd.DataFrame, lifecycle: pd.DataFrame, target: pd.Series) -> dict[str, float]:
    record = {
        "source_user_id": target["user_id"],
        "source_assignment_id": target["assignment_id"],
        "source_start_order_id": target["first_order_id"],
    }
    record.update(summarize_candidate(df, lifecycle, target))
    record.update({
        "target_binary_mastered": int(target["mastered"]),
        "target_regression_problems": int(target["problem_count"]),
        "target_regression_work_time": float(target["assignment_overlap_time_total_seconds"]),
    })
    return {column: record[column] for column in OUTPUT_COLUMNS}


def _build_candidate_worker(args):
    user_id, target_fields = args
    target = pd.Series(target_fields)
    return build_candidate_instance(_WORKER_SOURCE_ROWS[user_id], _WORKER_LIFECYCLE[user_id], target)


def preprocess_dataframe(df: pd.DataFrame, interaction_filter=None) -> pd.DataFrame:
    global _WORKER_SOURCE_ROWS, _WORKER_LIFECYCLE
    fixed_filtered_source = filter_source_rows(df)
    if interaction_filter is not None:
        fixed_filtered_source = interaction_filter(fixed_filtered_source)
    final_lifecycle = build_assignment_lifecycle(fixed_filtered_source)
    final_candidates = select_final_eligible_assignments(final_lifecycle)
    duration_complete_candidates = filter_duration_complete_candidates(final_lifecycle, final_candidates)
    duration_complete_candidates = duration_complete_candidates.sort_values(
        ["user_id", "first_order_id", "assignment_id"], kind="mergesort"
    )
    final_source_rows_by_user = {key: value for key, value in fixed_filtered_source.groupby("user_id", sort=False)}
    final_lifecycle_by_user = {key: value for key, value in final_lifecycle.groupby("user_id", sort=False)}
    candidate_args = [
        (target.user_id, dict(target._asdict()))
        for target in duration_complete_candidates.itertuples(index=False)
    ]
    if len(candidate_args) >= PARALLEL_MIN_CANDIDATES:
        _WORKER_SOURCE_ROWS = final_source_rows_by_user
        _WORKER_LIFECYCLE = final_lifecycle_by_user
        processes = min(os.cpu_count() or 1, 16)
        try:
            with multiprocessing.Pool(processes=processes) as pool:
                output_records = pool.map(_build_candidate_worker, candidate_args, chunksize=100)
        finally:
            _WORKER_SOURCE_ROWS = None
            _WORKER_LIFECYCLE = None
    else:
        output_records = []
        for target in duration_complete_candidates.itertuples(index=False):
            target_series = pd.Series(target._asdict())
            output_records.append(build_candidate_instance(final_source_rows_by_user[target.user_id], final_lifecycle_by_user[target.user_id], target_series))
    result = pd.DataFrame.from_records(output_records, columns=OUTPUT_COLUMNS)
    result.attrs.update({
        "source_row_count": int(len(df)),
        "filtered_source_row_count": int(len(fixed_filtered_source)),
        "removed_source_row_count": int(len(df) - len(fixed_filtered_source)),
        "physical_assignment_count": int(len(final_lifecycle)),
        "provisional_target_count": int(len(final_candidates)),
        "provisional_student_count": int(final_candidates["user_id"].nunique()),
        "duration_valid_candidate_count": int(len(duration_complete_candidates)),
        "duration_invalid_candidate_count": int(len(final_candidates) - len(duration_complete_candidates)),
        "final_candidate_count": int(len(result)),
    })
    return result


def write_preprocessed_dataset(input_path: Path = INPUT_PATH, output_path: Path = OUTPUT_PATH) -> pd.DataFrame:
    result = preprocess_dataframe(load_data(input_path))
    result.to_csv(output_path, index=False)
    return result


DATASETS_ROOT = Path(__file__).resolve().parents[1]
class _2009SBAdapter(BasePreparedAdapter):
    label = "assistments2009_sb"
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
                "_prepared_assistments2009_sb",
                DATASETS_ROOT / "assistments2009-2010" / "preprocess_sb.py",
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
        return self.module.filter_source_rows(context["raw"])

    def build_rows(self, context, students, interaction_filter):
        subset = context["raw"].loc[context["raw"]["user_id"].astype(str).isin(map(str, students))]
        return self.module.preprocess_dataframe(subset, interaction_filter=interaction_filter)


def create_adapter():
    return _2009SBAdapter()


def main():
    write_preprocessed_dataset()


if __name__ == "__main__":
    main()
