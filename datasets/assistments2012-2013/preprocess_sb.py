import numpy as np
import pandas as pd
import sys
from pathlib import Path

from fold_preparation import BasePreparedAdapter
from dataset_registry import _load_dataset_module

MIN_PRIOR_ASSIGNMENTS = 3
AFFECT_COLUMNS = {
    "Average_confidence(FRUSTRATED)": "predicted_frustration_mean",
    "Average_confidence(CONFUSED)": "predicted_confusion_mean",
    "Average_confidence(CONCENTRATING)": "predicted_concentration_mean",
    "Average_confidence(BORED)": "predicted_boredom_mean",
}
ANSWER_LEVELS = ("algebra", "choose_1", "choose_n", "fill_in_1")
METADATA_COLUMNS = ("source_user_id", "source_assignment_id", "source_start_problem_log_id")
SHARED_FEATURE_COLUMNS = (
    "original_problem_count", "ended_assignment_count", "scaffolding_problem_rate", "interactions",
    "repeated_problem_rate", "correct_rate", "attempt_count_mean", "attempt_count_std", "attempt_count_total",
    "attempt_count_p90", "hint_count_mean", "hint_count_total", "hint_use_rate", "bottom_hint_rate",
    "first_action_attempt_rate", "first_action_hint_rate", "first_action_scaffolding_rate", "overlap_time_mean_seconds",
    "overlap_time_std_seconds", "overlap_time_total_seconds", "overlap_time_p90_seconds",
) + tuple(
    f"answer_type_{level}_{suffix}"
    for level in ANSWER_LEVELS
    for suffix in (
    "rate", "correct_rate", "overlap_time_mean_seconds", "overlap_time_total_seconds",
)
) + (
    "mastery_rate", "assignment_problem_count_mean", "assignment_problem_count_std", "assignment_problem_count_p90",
    "assignment_time_mean", "assignment_time_std", "assignment_time_p90", "previous_assignment_mastered",
    "previous_assignment_problem_count", "previous_assignment_time", "selected_assignment_answer_type_algebra_present",
    "selected_assignment_answer_type_choose_1_present", "selected_assignment_answer_type_choose_n_present",
    "selected_assignment_answer_type_fill_in_1_present",
)
ENRICHED_FEATURE_COLUMNS = (
    "attempt_count_median", "attempt_count_p75", "attempt_count_iqr", "attempt_count_above_p90_rate",
    "overlap_time_median_seconds", "overlap_time_p75_seconds", "overlap_time_iqr_seconds",
    "overlap_time_above_p90_rate", "recent_5_observation_count", "recent_5_correct_rate", "recent_5_attempt_count_mean",
    "recent_5_overlap_time_mean_seconds", "recent_5_hint_use_rate", "recent_5_attempt_count_median",
    "recent_5_attempt_count_p90", "recent_5_overlap_time_median_seconds", "recent_5_overlap_time_p90_seconds",
    "selected_skill_prior_interactions", "selected_skill_has_history", "selected_skill_correct_rate",
    "selected_skill_attempt_count_mean", "selected_skill_overlap_time_mean_seconds", "selected_skill_hint_use_rate",
    "recent_10_correct_trend", "recent_10_log1p_attempt_count_trend", "recent_10_log1p_overlap_time_seconds_trend",
    "recent_10_hint_use_trend",
)
FEATURE_COLUMNS = SHARED_FEATURE_COLUMNS + ENRICHED_FEATURE_COLUMNS + tuple(AFFECT_COLUMNS.values())
TARGET_COLUMNS = (
    "target_binary_mastered", "target_regression_problems", "target_regression_work_time",
)
OUTPUT_COLUMNS = METADATA_COLUMNS + FEATURE_COLUMNS + TARGET_COLUMNS
REQUIRED_SOURCE_COLUMNS = (
    "user_id", "assignment_id", "problem_log_id", "problem_id", "start_time", "end_time", "original", "correct",
    "attempt_count", "hint_count", "bottom_hint", "first_action", "problem_type", "skill_id", "type", *AFFECT_COLUMNS,
)


def load_data(file_path):
    return pd.read_csv(file_path, usecols=list(REQUIRED_SOURCE_COLUMNS), encoding="latin-1", low_memory=False, memory_map=True)


def normalize_category(value) -> str:
    if pd.isna(value):
        return "unknown"
    normalized = "".join(character.lower() if character.isalnum() else "_" for character in str(value).strip())
    return "_".join(part for part in normalized.split("_") if part) or "unknown"


def filter_source_rows(df: pd.DataFrame) -> pd.DataFrame:
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


def build_assignment_lifecycle(df: pd.DataFrame) -> pd.DataFrame:
    key = ["user_id", "assignment_id"]
    grouped = df.groupby(key, sort=False)
    first_start_time = grouped["start_time"].min()
    last_end_time = grouped["end_time"].max()
    problem_count = grouped["problem_id"].nunique()
    ordered = df.sort_values(key + ["start_time", "problem_log_id"], kind="mergesort")
    source_start_problem_log_id = ordered.groupby(key, sort=False)["problem_log_id"].first().astype("int64")
    duration_seconds = (last_end_time - first_start_time).dt.total_seconds()
    work_time = grouped["overlap_time_seconds"].sum()

    original_rows = df.loc[df["original"].eq(1)]

    if original_rows.empty:
        mastered = pd.Series(0, index=first_start_time.index, dtype="int64")
    else:
        original_rows = original_rows.sort_values(key + ["start_time", "problem_log_id"], kind="mergesort")
        original_grouped = original_rows.groupby(key, sort=False)
        run_of_three = (
            original_rows["correct"].eq(1)
            & original_grouped["correct"].shift(1).eq(1)
            & original_grouped["correct"].shift(2).eq(1)
        )
        mastered = run_of_three.groupby([original_rows[column] for column in key], sort=False).any()

    result = pd.DataFrame({
        "first_start_time": first_start_time,
        "last_end_time": last_end_time,
        "source_start_problem_log_id": source_start_problem_log_id,
        "mastered": mastered.reindex(first_start_time.index).fillna(0).astype("int64"),
        "problem_count": problem_count,
        "assignment_duration_seconds": duration_seconds,
        "assignment_work_time": work_time,
    })
    work_time_values = result["assignment_work_time"].to_numpy(dtype=float)
    result["duration_valid"] = np.isfinite(work_time_values) & (work_time_values > 0)
    return result.reset_index()


def select_eligible_assignments(lifecycle: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for _, group in lifecycle.groupby("user_id", sort=False):
        ended_times = np.sort(group["last_end_time"].to_numpy())
        started = group["first_start_time"].to_numpy()
        counts = np.searchsorted(ended_times, started, side="left")
        pieces.append(group.loc[counts >= MIN_PRIOR_ASSIGNMENTS])
    return pd.concat(pieces, ignore_index=True) if pieces else lifecycle.iloc[0:0].copy()


def select_final_assignment_per_student(candidates: pd.DataFrame) -> pd.DataFrame:
    ordered = candidates.sort_values(
        ["user_id", "first_start_time", "source_start_problem_log_id", "assignment_id"], kind="mergesort"
    )
    return ordered.drop_duplicates(subset=["user_id"], keep="last").reset_index(drop=True)


def select_final_eligible_assignments(lifecycle: pd.DataFrame, eligible_assignments: pd.DataFrame | None = None) -> pd.DataFrame:
    physical_final = select_final_assignment_per_student(lifecycle)
    if eligible_assignments is None:
        eligible_assignments = select_eligible_assignments(lifecycle)
    eligible_keys = eligible_assignments.loc[:, ["user_id", "assignment_id"]].drop_duplicates()
    return physical_final.merge(eligible_keys, on=["user_id", "assignment_id"], how="inner")


def filter_duration_complete_candidates(lifecycle: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    keep_indices = []
    for _, group in candidates.groupby("user_id", sort=False):
        user_id = group["user_id"].iloc[0]
        lc = lifecycle.loc[lifecycle["user_id"].eq(user_id)].sort_values(
            ["last_end_time", "assignment_id"], kind="mergesort"
        )
        end_times = lc["last_end_time"].to_numpy()
        valid_flags = lc["duration_valid"].to_numpy(dtype=bool)
        prefix_valid = np.logical_and.accumulate(valid_flags)
        start_times = group["first_start_time"].to_numpy()
        counts = np.searchsorted(end_times, start_times, side="left")
        ended_ok = np.where(counts > 0, prefix_valid[np.maximum(counts - 1, 0)], True)
        target_ok = group["duration_valid"].to_numpy(dtype=bool)
        keep_indices.extend(group.index[(ended_ok & target_ok)].tolist())
    return candidates.loc[keep_indices].reset_index(drop=True)


def _sample_std(values: pd.Series) -> float:
    if len(values) < 2:
        return 0.0
    return float(values.std(ddof=1))


def _fraction(numerator, denominator) -> float:
    return 0.0 if denominator == 0 else float(numerator / denominator)


def _ols_slope(values) -> float:
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return 0.0
    positions = np.arange(len(values), dtype=float)
    centered_positions = positions - positions.mean()
    denominator = float((centered_positions ** 2).sum())
    return 0.0 if denominator == 0 else float(centered_positions.dot(values - values.mean()) / denominator)


def summarize_candidate(
    event_history: pd.DataFrame,
    main_history: pd.DataFrame,
    ended: pd.DataFrame,
    target_rows: pd.DataFrame,
) -> dict[str, float]:
    ended = ended.sort_values(["last_end_time", "assignment_id"], kind="mergesort")
    attempts = main_history["attempt_count"].astype(float)
    times = event_history["overlap_time_seconds"].astype(float)
    problem_counts = ended["problem_count"].astype(float)
    assignment_work_times = ended["assignment_work_time"].astype(float)
    main_total = len(main_history)
    event_total = len(event_history)
    record = {
        "original_problem_count": int(main_total),
        "ended_assignment_count": int(len(ended)),
        "scaffolding_problem_rate": _fraction(event_history["original"].eq(0).sum(), event_total),
        "interactions": int(event_total),
        "repeated_problem_rate": float(1 - event_history["problem_id"].nunique() / event_total),
        "correct_rate": _fraction(main_history["correct"].eq(1).sum(), main_total),
        "attempt_count_mean": 0.0 if attempts.empty else float(attempts.mean()),
        "attempt_count_std": _sample_std(attempts),
        "attempt_count_total": float(attempts.sum()),
        "attempt_count_p90": 0.0 if attempts.empty else float(attempts.quantile(0.90)),
        "hint_count_mean": 0.0 if main_total == 0 else float(main_history["hint_count"].astype(float).mean()),
        "hint_count_total": float(main_history["hint_count"].astype(float).sum()),
        "hint_use_rate": 0.0 if main_total == 0 else float(main_history["hint_count"].astype(float).gt(0).fillna(False).mean()),
        "bottom_hint_rate": 0.0 if main_total == 0 else float(main_history["bottom_hint"].eq(1).mean()),
        "first_action_attempt_rate": 0.0 if main_total == 0 else float(main_history["first_action"].eq(0).mean()),
        "first_action_hint_rate": 0.0 if main_total == 0 else float(main_history["first_action"].eq(1).mean()),
        "first_action_scaffolding_rate": 0.0 if main_total == 0 else float(main_history["first_action"].eq(2).mean()),
        "overlap_time_mean_seconds": float(times.mean()),
        "overlap_time_std_seconds": _sample_std(times),
        "overlap_time_total_seconds": float(times.sum()),
        "overlap_time_p90_seconds": float(times.quantile(0.90)),
    }

    normalized_types = main_history["problem_type"].map(normalize_category)

    for level in ANSWER_LEVELS:
        matching = main_history.loc[normalized_types.eq(level)]
        record[f"answer_type_{level}_rate"] = _fraction(len(matching), main_total)

        if matching.empty:
            record[f"answer_type_{level}_correct_rate"] = 0.0
            record[f"answer_type_{level}_overlap_time_mean_seconds"] = 0.0
            record[f"answer_type_{level}_overlap_time_total_seconds"] = 0.0
        else:
            record[f"answer_type_{level}_correct_rate"] = float(matching["correct"].eq(1).mean())
            record[f"answer_type_{level}_overlap_time_mean_seconds"] = float(matching["overlap_time_seconds"].mean())
            record[f"answer_type_{level}_overlap_time_total_seconds"] = float(matching["overlap_time_seconds"].sum())

    for source_column, output_column in AFFECT_COLUMNS.items():
        record[output_column] = float(event_history[source_column].astype(float).mean())

    record["mastery_rate"] = float(ended["mastered"].mean())
    record["assignment_problem_count_mean"] = float(problem_counts.mean())
    record["assignment_problem_count_std"] = _sample_std(problem_counts)
    record["assignment_problem_count_p90"] = float(problem_counts.quantile(0.90))
    record["assignment_time_mean"] = float(assignment_work_times.mean())
    record["assignment_time_std"] = _sample_std(assignment_work_times)
    record["assignment_time_p90"] = float(assignment_work_times.quantile(0.90))
    previous = ended.iloc[-1]
    record["previous_assignment_mastered"] = int(previous["mastered"])
    record["previous_assignment_problem_count"] = int(previous["problem_count"])
    record["previous_assignment_time"] = float(previous["assignment_work_time"])
    normalized_target_types = target_rows["problem_type"].map(normalize_category)

    for level in ANSWER_LEVELS:
        record[f"selected_assignment_answer_type_{level}_present"] = int(normalized_target_types.eq(level).any())

    return record


def summarize_additional_features(
    history_all: pd.DataFrame,
    history_main: pd.DataFrame,
    summary: dict[str, float],
) -> dict[str, float]:
    ordered_main = history_main.sort_values(["start_time", "problem_log_id"], kind="mergesort")
    ordered_all = history_all.sort_values(["start_time", "problem_log_id"], kind="mergesort")
    record = {}
    attempts = ordered_main["attempt_count"].astype(float)
    times = ordered_all["overlap_time_seconds"].astype(float)

    if attempts.empty:
        record.update({
            "attempt_count_median": 0.0,
            "attempt_count_p75": 0.0,
            "attempt_count_iqr": 0.0,
            "attempt_count_above_p90_rate": 0.0,
        })

    else:
        attempt_p25 = float(attempts.quantile(0.25))
        attempt_p75 = float(attempts.quantile(0.75))
        attempt_p90 = float(attempts.quantile(0.90))
        record["attempt_count_median"] = float(attempts.median())
        record["attempt_count_p75"] = attempt_p75
        record["attempt_count_iqr"] = attempt_p75 - attempt_p25
        record["attempt_count_above_p90_rate"] = float(attempts.gt(attempt_p90).sum() / len(attempts))

    time_p25 = float(times.quantile(0.25))
    time_p75 = float(times.quantile(0.75))
    time_p90 = float(times.quantile(0.90))
    record["overlap_time_median_seconds"] = float(times.median())
    record["overlap_time_p75_seconds"] = time_p75
    record["overlap_time_iqr_seconds"] = time_p75 - time_p25
    record["overlap_time_above_p90_rate"] = float(times.gt(time_p90).sum() / len(times))

    for window in (5,):
        recent_main = ordered_main.tail(window)
        recent_all = ordered_all.tail(window)
        record[f"recent_{window}_observation_count"] = int(len(recent_main))
        record[f"recent_{window}_correct_rate"] = 0.0 if recent_main.empty else float(recent_main["correct"].mean())
        record[f"recent_{window}_attempt_count_mean"] = 0.0 if recent_main.empty else float(recent_main["attempt_count"].mean())
        record[f"recent_{window}_overlap_time_mean_seconds"] = float(recent_all["overlap_time_seconds"].mean())
        record[f"recent_{window}_hint_use_rate"] = 0.0 if recent_main.empty else float(recent_main["hint_count"].astype(float).gt(0).fillna(False).mean())
        record[f"recent_{window}_attempt_count_median"] = 0.0 if recent_main.empty else float(recent_main["attempt_count"].median())
        record[f"recent_{window}_attempt_count_p90"] = 0.0 if recent_main.empty else float(recent_main["attempt_count"].quantile(0.90))
        record[f"recent_{window}_overlap_time_median_seconds"] = float(recent_all["overlap_time_seconds"].median())
        record[f"recent_{window}_overlap_time_p90_seconds"] = float(recent_all["overlap_time_seconds"].quantile(0.90))

    recent_main = ordered_main.tail(10)
    recent_all = ordered_all.tail(10)
    record["recent_10_correct_trend"] = _ols_slope(recent_main["correct"].to_numpy())
    record["recent_10_log1p_attempt_count_trend"] = _ols_slope(np.log1p(recent_main["attempt_count"].astype(float).to_numpy()))
    record["recent_10_log1p_overlap_time_seconds_trend"] = _ols_slope(
        np.log1p(recent_all["overlap_time_seconds"].astype(float).to_numpy())
    )
    record["recent_10_hint_use_trend"] = _ols_slope(
        recent_main["hint_count"].astype(float).gt(0).fillna(False).astype(float).to_numpy()
    )
    return record


def _problem_type_codes(problem_type: pd.Series) -> np.ndarray:
    unique_codes = {}
    for value in problem_type.dropna().unique():
        normalized = normalize_category(value)
        unique_codes[value] = (
            ANSWER_LEVELS.index(normalized) if normalized in ANSWER_LEVELS else -1
        )
    return problem_type.map(unique_codes).fillna(-1).to_numpy().astype("int64")


def _top_window_mask(chron_ranks: np.ndarray, window: int) -> np.ndarray:
    count = len(chron_ranks)
    if count <= window:
        return np.ones(count, dtype=bool)
    threshold = np.partition(chron_ranks, count - window)[count - window]
    return chron_ranks >= threshold


def build_user_instance_records(
    user_rows: pd.DataFrame,
    user_lifecycle: pd.DataFrame,
    user_candidates: pd.DataFrame,
) -> list[dict]:
    user_rows = user_rows.sort_values(["start_time", "problem_log_id"], kind="mergesort").reset_index(drop=True)
    ended_lc = user_lifecycle.sort_values(["last_end_time", "assignment_id"], kind="mergesort").reset_index(drop=True)
    ended_assignment_ids = ended_lc["assignment_id"].to_numpy()
    last_end_times = ended_lc["last_end_time"].to_numpy()
    rank_of_assignment = pd.Series(np.arange(len(ended_assignment_ids)), index=ended_assignment_ids)
    assignment_rank = user_rows["assignment_id"].map(rank_of_assignment).to_numpy()
    rows_by_assignment = {
        assignment_id: indices
        for assignment_id, indices in user_rows.groupby("assignment_id", sort=False).groups.items()
    }

    original = user_rows["original"].to_numpy(dtype="int64")
    correct = user_rows["correct"].to_numpy(dtype="float64")
    attempt_count = user_rows["attempt_count"].to_numpy(dtype="float64")
    hint_count = user_rows["hint_count"].to_numpy(dtype="float64")
    bottom_hint = user_rows["bottom_hint"].to_numpy(dtype="float64")
    first_action = user_rows["first_action"].to_numpy(dtype="float64")
    overlap_time = user_rows["overlap_time_seconds"].to_numpy(dtype="float64")
    affect_values = user_rows[list(AFFECT_COLUMNS)].to_numpy(dtype="float64")
    skill_values = user_rows["skill_id"]
    problem_type_codes = _problem_type_codes(user_rows["problem_type"])
    chron_ranks = np.arange(len(user_rows))
    mastered_values = ended_lc["mastered"].to_numpy(dtype="float64")
    problem_count_values = ended_lc["problem_count"].to_numpy(dtype="float64")
    work_time_values = ended_lc["assignment_work_time"].to_numpy(dtype="float64")

    records = []
    for candidate in user_candidates.itertuples(index=False, name="Candidate"):
        ended_count = int(np.searchsorted(last_end_times, np.datetime64(candidate.first_start_time), side="left"))
        event_mask = assignment_rank < ended_count
        event_indices = np.flatnonzero(event_mask)
        main_mask = event_mask & (original == 1)
        main_indices = np.flatnonzero(main_mask)
        event_size = int(event_indices.size)
        main_size = int(main_indices.size)

        record = {
            "source_user_id": candidate.user_id,
            "source_assignment_id": candidate.assignment_id,
            "source_start_problem_log_id": int(candidate.source_start_problem_log_id),
        }

        record["original_problem_count"] = main_size
        record["ended_assignment_count"] = ended_count
        record["scaffolding_problem_rate"] = (
            0.0 if event_size == 0 else float(np.count_nonzero(original[event_indices] == 0) / event_size)
        )
        record["interactions"] = event_size
        record["repeated_problem_rate"] = (
            0.0 if event_size == 0 else float(1 - len(np.unique(user_rows["problem_id"].iloc[event_indices])) / event_size)
        )
        correct_main = correct[main_indices]
        record["correct_rate"] = (
            0.0 if main_size == 0 else float(np.count_nonzero(correct_main == 1) / main_size)
        )

        attempt_main = attempt_count[main_indices]
        record["attempt_count_mean"] = 0.0 if main_size == 0 else float(attempt_main.mean())
        record["attempt_count_std"] = 0.0 if main_size < 2 else float(np.std(attempt_main, ddof=1))
        record["attempt_count_total"] = float(attempt_main.sum())
        record["attempt_count_p90"] = 0.0 if main_size == 0 else float(np.quantile(attempt_main, 0.90))
        hint_main = hint_count[main_indices]
        record["hint_count_mean"] = 0.0 if main_size == 0 else float(hint_main.mean())
        record["hint_count_total"] = float(hint_main.sum())
        record["hint_use_rate"] = (
            0.0 if main_size == 0 else float(np.count_nonzero(hint_main > 0) / main_size)
        )

        bottom_main = bottom_hint[main_indices]
        record["bottom_hint_rate"] = (
            0.0 if main_size == 0 else float(np.count_nonzero(bottom_main == 1) / main_size)
        )

        first_main = first_action[main_indices]
        record["first_action_attempt_rate"] = (
            0.0 if main_size == 0 else float(np.count_nonzero(first_main == 0) / main_size)
        )

        record["first_action_hint_rate"] = (
            0.0 if main_size == 0 else float(np.count_nonzero(first_main == 1) / main_size)
        )

        record["first_action_scaffolding_rate"] = (
            0.0 if main_size == 0 else float(np.count_nonzero(first_main == 2) / main_size)
        )

        overlap_event = overlap_time[event_indices]
        record["overlap_time_mean_seconds"] = float(overlap_event.mean())
        record["overlap_time_std_seconds"] = 0.0 if event_size < 2 else float(np.std(overlap_event, ddof=1))
        record["overlap_time_total_seconds"] = float(overlap_event.sum())
        record["overlap_time_p90_seconds"] = float(np.quantile(overlap_event, 0.90))

        for level_index, level in enumerate(ANSWER_LEVELS):
            level_mask = main_mask & (problem_type_codes == level_index)
            level_size = int(np.count_nonzero(level_mask))
            record[f"answer_type_{level}_rate"] = (
                0.0 if main_size == 0 else float(level_size / main_size)
            )
            if level_size == 0:
                record[f"answer_type_{level}_correct_rate"] = 0.0
                record[f"answer_type_{level}_overlap_time_mean_seconds"] = 0.0
                record[f"answer_type_{level}_overlap_time_total_seconds"] = 0.0
            else:
                level_overlap = overlap_time[level_mask]
                record[f"answer_type_{level}_correct_rate"] = float(
                    np.count_nonzero(correct[level_mask] == 1) / level_size
                )
                record[f"answer_type_{level}_overlap_time_mean_seconds"] = float(level_overlap.mean())
                record[f"answer_type_{level}_overlap_time_total_seconds"] = float(level_overlap.sum())
        affect_means = affect_values[event_indices].mean(axis=0)

        for (source_column, column), value in zip(AFFECT_COLUMNS.items(), affect_means):
            record[column] = float(value)
        mastered_slice = mastered_values[:ended_count]
        problem_count_slice = problem_count_values[:ended_count]
        work_time_slice = work_time_values[:ended_count]
        record["mastery_rate"] = float(mastered_slice.mean())
        record["assignment_problem_count_mean"] = float(problem_count_slice.mean())
        record["assignment_problem_count_std"] = (
            0.0 if ended_count < 2 else float(np.std(problem_count_slice, ddof=1))
        )
        record["assignment_problem_count_p90"] = float(np.quantile(problem_count_slice, 0.90))
        record["assignment_time_mean"] = float(work_time_slice.mean())
        record["assignment_time_std"] = (
            0.0 if ended_count < 2 else float(np.std(work_time_slice, ddof=1))
        )
        record["assignment_time_p90"] = float(np.quantile(work_time_slice, 0.90))
        previous_index = ended_count - 1
        record["previous_assignment_mastered"] = int(mastered_values[previous_index])
        record["previous_assignment_problem_count"] = int(problem_count_values[previous_index])
        record["previous_assignment_time"] = float(work_time_values[previous_index])
        target_codes = problem_type_codes[rows_by_assignment[candidate.assignment_id]]

        for level_index, level in enumerate(ANSWER_LEVELS):
            record[f"selected_assignment_answer_type_{level}_present"] = int(
                np.any(target_codes == level_index)
            )

        if main_size == 0:
            record.update({
                "attempt_count_median": 0.0,
                "attempt_count_p75": 0.0,
                "attempt_count_iqr": 0.0,
                "attempt_count_above_p90_rate": 0.0,
            })
            
        else:
            attempt_p25 = float(np.quantile(attempt_main, 0.25))
            attempt_p75 = float(np.quantile(attempt_main, 0.75))
            attempt_p90 = float(np.quantile(attempt_main, 0.90))
            record["attempt_count_median"] = float(np.median(attempt_main))
            record["attempt_count_p75"] = attempt_p75
            record["attempt_count_iqr"] = attempt_p75 - attempt_p25
            record["attempt_count_above_p90_rate"] = float(np.count_nonzero(attempt_main > attempt_p90) / main_size)
        time_p25 = float(np.quantile(overlap_event, 0.25))
        time_p75 = float(np.quantile(overlap_event, 0.75))
        time_p90 = float(np.quantile(overlap_event, 0.90))
        record["overlap_time_median_seconds"] = float(np.median(overlap_event))
        record["overlap_time_p75_seconds"] = time_p75
        record["overlap_time_iqr_seconds"] = time_p75 - time_p25
        record["overlap_time_above_p90_rate"] = float(np.count_nonzero(overlap_event > time_p90) / event_size)

        main_chron = chron_ranks[main_indices]
        event_chron = chron_ranks[event_indices]
        recent_5_main = main_indices[_top_window_mask(main_chron, 5)]
        recent_10_main = main_indices[_top_window_mask(main_chron, 10)]
        recent_5_event = event_indices[_top_window_mask(event_chron, 5)]
        recent_10_event = event_indices[_top_window_mask(event_chron, 10)]
        record["recent_5_observation_count"] = int(recent_5_main.size)
        record["recent_5_correct_rate"] = 0.0 if main_size == 0 else float(correct[recent_5_main].mean())
        record["recent_5_attempt_count_mean"] = 0.0 if main_size == 0 else float(attempt_count[recent_5_main].mean())
        record["recent_5_overlap_time_mean_seconds"] = 0.0 if event_size == 0 else float(overlap_time[recent_5_event].mean())
        record["recent_5_hint_use_rate"] = (
            0.0 if main_size == 0 else float(np.count_nonzero(hint_count[recent_5_main] > 0) / recent_5_main.size)
        )
        record["recent_5_attempt_count_median"] = 0.0 if main_size == 0 else float(np.median(attempt_count[recent_5_main]))
        record["recent_5_attempt_count_p90"] = 0.0 if main_size == 0 else float(np.quantile(attempt_count[recent_5_main], 0.90))
        record["recent_5_overlap_time_median_seconds"] = 0.0 if event_size == 0 else float(np.median(overlap_time[recent_5_event]))
        record["recent_5_overlap_time_p90_seconds"] = 0.0 if event_size == 0 else float(np.quantile(overlap_time[recent_5_event], 0.90))
        target_skill_ids = set(skill_values.iloc[rows_by_assignment[candidate.assignment_id]].dropna().astype(str))
        matched = main_indices[np.asarray(skill_values.iloc[main_indices].notna() & skill_values.iloc[main_indices].astype(str).isin(target_skill_ids))] if target_skill_ids else np.array([], dtype=int)
        record["selected_skill_prior_interactions"] = int(matched.size)
        record["selected_skill_has_history"] = int(matched.size > 0)

        if matched.size == 0:
            record.update({
                "selected_skill_correct_rate": 0.0,
                "selected_skill_attempt_count_mean": 0.0,
                "selected_skill_overlap_time_mean_seconds": 0.0,
                "selected_skill_hint_use_rate": 0.0,
            })
        else:
            record.update({
                "selected_skill_correct_rate": float(correct[matched].mean()),
                "selected_skill_attempt_count_mean": float(attempt_count[matched].mean()),
                "selected_skill_overlap_time_mean_seconds": float(overlap_time[matched].mean()),
                "selected_skill_hint_use_rate": float(np.count_nonzero(hint_count[matched] > 0) / matched.size),
            })
        record["recent_10_correct_trend"] = _ols_slope(correct[recent_10_main])
        record["recent_10_log1p_attempt_count_trend"] = _ols_slope(np.log1p(attempt_count[recent_10_main]))
        record["recent_10_log1p_overlap_time_seconds_trend"] = _ols_slope(np.log1p(overlap_time[recent_10_event]))
        record["recent_10_hint_use_trend"] = _ols_slope((hint_count[recent_10_main] > 0).astype(float))
        record["target_binary_mastered"] = int(candidate.mastered)
        record["target_regression_problems"] = int(candidate.problem_count)
        record["target_regression_work_time"] = float(candidate.assignment_work_time)
        records.append(record)
    return records


def preprocess_dataframe(df: pd.DataFrame, interaction_filter=None) -> pd.DataFrame:
    fixed_filtered_source = filter_source_rows(df)

    if interaction_filter is not None:
        fixed_filtered_source = interaction_filter(fixed_filtered_source)
    final_lifecycle = build_assignment_lifecycle(fixed_filtered_source)
    final_candidates = select_final_eligible_assignments(final_lifecycle)
    duration_complete_candidates = filter_duration_complete_candidates(final_lifecycle, final_candidates)
    duration_complete_candidates = duration_complete_candidates.sort_values(
        ["user_id", "first_start_time", "source_start_problem_log_id", "assignment_id"], kind="mergesort"
    )
    final_source_rows_by_user = {key: value for key, value in fixed_filtered_source.groupby("user_id", sort=False)}
    final_lifecycle_by_user = {key: value for key, value in final_lifecycle.groupby("user_id", sort=False)}
    output_records = []

    for user_id, user_candidates in duration_complete_candidates.groupby("user_id", sort=False):
        output_records.extend(build_user_instance_records(
            final_source_rows_by_user[user_id],
            final_lifecycle_by_user[user_id],
            user_candidates,
        ))
        
    result = pd.DataFrame(output_records, columns=OUTPUT_COLUMNS)
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

DATASETS_ROOT = Path(__file__).resolve().parents[1]
_Base2012Adapter = _load_dataset_module('_prepared_assistments2012_split_regimes', DATASETS_ROOT / 'assistments2012-2013/preprocess_split_regimes.py')._Base2012Adapter
class _2012SBAdapter(_Base2012Adapter):
    label = "assistments2012_sb"
    filter_description = "raw attempt_count P99.9 and log1p-IQR fences on overlap_time_seconds (seconds)"

    def _fixed_validity(self, frame):
        return self.branch_module.filter_source_rows(frame)

    def _preprocess(self, subset, interaction_filter):
        return self.branch_module.preprocess_dataframe(subset, interaction_filter=interaction_filter)


def create_adapter():
    return _2012SBAdapter()
