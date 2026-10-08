import numpy as np
import pandas as pd
import sys
from pathlib import Path

from fold_preparation import BasePreparedAdapter
from dataset_registry import _load_dataset_module

MIN_PRIOR_ASSIGNMENTS = 3
PROBLEM_TYPES = (
    "algebraic_expression", "number", "ungraded_open_response", "multiple_choice", "exact_match_case_sensitive",
    "exact_match_ignore_case", "check_all_that_apply", "exact_fraction", "numeric_expression", "ordering", "unknown",
)
METADATA_COLUMNS = (
    "source_student_id", "source_assignment_log_id", "source_assignment_id",
)
FEATURE_COLUMNS = (
    "interactions", "repeated_problem_rate", "assignment_count", "active_day_count", "history_span_days",
    "days_since_previous_interaction", "correct_rate", "attempt_count_mean", "attempt_count_std", "attempt_count_total",
    "attempt_count_p90", "time_on_task_mean_seconds", "time_on_task_std_seconds", "time_on_task_total_seconds",
    "time_on_task_p90_seconds", "time_on_task_observed_rate", "answer_given_rate",
    *(f"problem_type_{name}_rate" for name in PROBLEM_TYPES), "hint_tutoring_available_rate",
    "explanation_tutoring_available_rate", "scaffold_tutoring_available_rate", "mastery_rate",
    "assignment_problem_count_mean", "assignment_problem_count_std", "assignment_problem_count_p90",
    "previous_assignment_mastered", "previous_assignment_problem_count", "attempt_count_median", "attempt_count_p75",
    "attempt_count_iqr", "attempt_count_above_p90_rate", "time_on_task_median_seconds", "time_on_task_p75_seconds",
    "time_on_task_iqr_seconds", "time_on_task_above_p90_rate", "recent_5_observation_count", "recent_5_correct_rate",
    "recent_5_attempt_count_mean", "recent_5_time_on_task_mean_seconds", "recent_5_attempt_count_median",
    "recent_5_attempt_count_p90", "recent_5_time_on_task_median_seconds", "recent_5_time_on_task_p90_seconds",
    "recent_10_correct_trend", "recent_10_log1p_attempt_count_trend", "recent_10_log1p_time_on_task_seconds_trend",
    "assignment_time_on_task_mean_seconds", "assignment_time_on_task_std_seconds",
    "assignment_time_on_task_p90_seconds", "previous_assignment_time_on_task_seconds",
    "assignment_time_on_task_observed_rate",
)
TARGET_COLUMNS = (
    "target_binary_mastered", "target_regression_problems", "target_regression_time_on_task",
)
OUTPUT_COLUMNS = METADATA_COLUMNS + FEATURE_COLUMNS + TARGET_COLUMNS


def _normalized_problem_type(value) -> str:
    if pd.isna(value):
        return "unknown"
    normalized = "".join(character.lower() if character.isalnum() else "_" for character in str(value).strip())
    normalized = "_".join(part for part in normalized.split("_") if part)
    return normalized if normalized in PROBLEM_TYPES else "unknown"


def _tutoring_available(frame: pd.DataFrame, name: str) -> pd.Series:
    column = f"{name}_tutoring_available"
    if column in frame:
        return pd.to_numeric(frame[column], errors="coerce").fillna(0).gt(0).astype("int8")
    return frame.get("tutoring_types", pd.Series("", index=frame.index)).fillna("").astype(str).str.contains(name, case=False, regex=False).astype("int8")


def _contexts(plogs, adets, alogs, pdets, lifecycle):
    assignments = adets.loc[:, ["assignment_id", "assignment_type"]].drop_duplicates("assignment_id")
    sb_assignment_ids = assignments.loc[assignments["assignment_type"].eq("skill_builder"), "assignment_id"]
    assignment_logs = alogs.merge(assignments, on="assignment_id", how="inner")
    assignment_logs = assignment_logs.loc[assignment_logs["assignment_type"].eq("skill_builder")].copy()
    assignment_logs["start_time"] = pd.to_datetime(assignment_logs["start_time"], utc=True, errors="coerce")
    assignment_logs["assignment_completed"] = pd.to_numeric(assignment_logs["assignment_completed"], errors="coerce")
    if "time_on_task" in assignment_logs.columns:
        assignment_logs["time_on_task"] = pd.to_numeric(assignment_logs["time_on_task"], errors="coerce")
    else:
        assignment_logs["time_on_task"] = np.nan
    lifecycle = lifecycle.loc[:, ["log_id", "final_action_timestamp", "invalid_timestamp_count"]].drop_duplicates("log_id")
    lifecycle["final_action_timestamp"] = pd.to_datetime(lifecycle["final_action_timestamp"], utc=True, errors="coerce")
    lifecycle["invalid_timestamp_count"] = pd.to_numeric(lifecycle["invalid_timestamp_count"], errors="coerce").fillna(1)
    problems = pdets.loc[:, ["problem_id", "problem_type"] + [
        column for column in ("hint_tutoring_available", "explanation_tutoring_available", "scaffold_tutoring_available") if column in pdets
    ]].drop_duplicates("problem_id")
    sb_plogs = plogs.loc[plogs["assignment_id"].isin(sb_assignment_ids)].merge(problems, on="problem_id", how="left")
    sb_plogs["start_time"] = pd.to_datetime(sb_plogs["start_time"], utc=True, errors="coerce")
    sb_plogs["problem_type"] = sb_plogs["problem_type"].map(_normalized_problem_type)
    for name in ("hint", "explanation", "scaffold"):
        sb_plogs[f"{name}_tutoring_available"] = _tutoring_available(sb_plogs, name)
    return assignment_logs, sb_plogs, lifecycle


def _clean_behavior(plogs: pd.DataFrame) -> pd.DataFrame:
    frame = plogs.copy()
    for column in ("correct", "attempt_count", "time_on_task"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    valid = frame["correct"].isin([0, 1]) & frame["attempt_count"].gt(0) & ~(
        frame["correct"].eq(1) & frame["attempt_count"].ne(1)
    )
    return frame.loc[valid].copy()


def _candidates(alogs: pd.DataFrame, lifecycle: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    joined = alogs.merge(lifecycle, on="log_id", how="left")
    times = pd.to_numeric(alogs["time_on_task"], errors="coerce")
    targets = alogs.loc[np.isfinite(times) & times.gt(0)].sort_values(["student_id", "start_time", "log_id"], kind="mergesort").groupby("student_id", sort=False).tail(1)
    histories = {}
    candidates = []
    for target in targets.itertuples(index=False):
        if target.assignment_completed not in (0, 1):
            continue
        prior = joined.loc[
            joined["student_id"].eq(target.student_id)
            & joined["start_time"].lt(target.start_time)
            & joined["assignment_completed"].isin([0, 1])
            & joined["final_action_timestamp"].notna()
            & joined["invalid_timestamp_count"].eq(0)
            & joined["final_action_timestamp"].lt(target.start_time)
        ].sort_values(["final_action_timestamp", "log_id"], kind="mergesort")
        if len(prior) >= MIN_PRIOR_ASSIGNMENTS:
            candidates.append(target)
            histories[target.student_id] = prior
    return pd.DataFrame(candidates), histories


def _fraction(numerator, denominator):
    return 0.0 if denominator == 0 else float(numerator / denominator)


def _robust_summaries(values):
    values = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return 0.0, 0.0, 0.0, 0.0
    series = pd.Series(values)
    median = float(series.quantile(.5))
    p75 = float(series.quantile(.75))
    p25 = float(series.quantile(.25))
    p90 = float(series.quantile(.9))
    return median, p75, p75 - p25, float((values > p90).mean())


def _quantile_or_zero(values, probability):
    values = pd.to_numeric(values, errors="coerce")
    if len(values) == 0:
        return 0.0
    return float(values.quantile(probability))


def _ols_slope(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2:
        return 0.0
    x_centered = x - x.mean()
    denominator = float(np.sum(x_centered ** 2))
    if denominator == 0.0:
        return 0.0
    return float(np.sum(x_centered * (y - y.mean())) / denominator)


def _summarize(target, prior, clean_plogs, raw_counts):
    prior_log_ids = prior["log_id"]
    history = clean_plogs.loc[clean_plogs["log_id"].isin(prior_log_ids)].sort_values(["start_time", "log_id", "problem_id"], kind="mergesort")
    interactions = len(history)
    observed_time = history["time_on_task"].dropna().astype(float)
    attempts = history["attempt_count"].astype(float)
    counts = prior["log_id"].map(raw_counts).fillna(0).astype(float)
    attempt_median, attempt_p75, attempt_iqr, attempt_above_p90_rate = _robust_summaries(attempts)
    time_median, time_p75, time_iqr, time_above_p90_rate = _robust_summaries(observed_time)
    recent_5 = history.tail(min(5, interactions))
    recent_5_count = len(recent_5)
    recent_5_attempts = recent_5["attempt_count"].astype(float)
    recent_5_times = recent_5["time_on_task"].dropna().astype(float)
    recent_10 = history.tail(min(10, interactions))
    recent_10_count = len(recent_10)
    positions = np.arange(recent_10_count, dtype=float)
    recent_10_times = pd.to_numeric(recent_10["time_on_task"], errors="coerce").to_numpy(dtype=float)
    finite_time_mask = np.isfinite(recent_10_times)
    prior_times = pd.to_numeric(prior["time_on_task"], errors="coerce").to_numpy(dtype=float)
    observed_prior_times = prior_times[np.isfinite(prior_times)]
    record = {
        "source_student_id": target.student_id,
        "source_assignment_log_id": target.log_id,
        "source_assignment_id": target.assignment_id,
        "interactions": interactions,
        "repeated_problem_rate": _fraction(interactions - history["problem_id"].nunique(), interactions),
        "assignment_count": len(prior),
        "active_day_count": 0 if history.empty else history["start_time"].dt.date.nunique(),
        "history_span_days": 0.0 if history.empty else float((history["start_time"].iloc[-1] - history["start_time"].iloc[0]).total_seconds() / 86400),
        "days_since_previous_interaction": 0.0 if history.empty else float((target.start_time - history["start_time"].iloc[-1]).total_seconds() / 86400),
        "correct_rate": _fraction(history["correct"].eq(1).sum(), interactions),
        "attempt_count_mean": 0.0 if attempts.empty else float(attempts.mean()),
        "attempt_count_std": 0.0 if len(attempts) < 2 else float(attempts.std(ddof=1)),
        "attempt_count_total": float(attempts.sum()),
        "attempt_count_p90": 0.0 if attempts.empty else float(attempts.quantile(.9)),
        "time_on_task_mean_seconds": 0.0 if observed_time.empty else float(observed_time.mean()),
        "time_on_task_std_seconds": 0.0 if len(observed_time) < 2 else float(observed_time.std(ddof=1)),
        "time_on_task_total_seconds": float(observed_time.sum()),
        "time_on_task_p90_seconds": 0.0 if observed_time.empty else float(observed_time.quantile(.9)),
        "time_on_task_observed_rate": _fraction(len(observed_time), interactions),
        "answer_given_rate": _fraction(pd.to_numeric(history["answer_given"], errors="coerce").fillna(0).eq(1).sum(), interactions),
        "mastery_rate": float(prior["assignment_completed"].mean()),
        "assignment_problem_count_mean": float(counts.mean()),
        "assignment_problem_count_std": 0.0 if len(counts) < 2 else float(counts.std(ddof=1)),
        "assignment_problem_count_p90": float(counts.quantile(.9)),
        "previous_assignment_mastered": int(prior.iloc[-1]["assignment_completed"]),
        "previous_assignment_problem_count": float(counts.iloc[-1]),
        "attempt_count_median": attempt_median,
        "attempt_count_p75": attempt_p75,
        "attempt_count_iqr": attempt_iqr,
        "attempt_count_above_p90_rate": attempt_above_p90_rate,
        "time_on_task_median_seconds": time_median,
        "time_on_task_p75_seconds": time_p75,
        "time_on_task_iqr_seconds": time_iqr,
        "time_on_task_above_p90_rate": time_above_p90_rate,
        "recent_5_observation_count": int(recent_5_count),
        "recent_5_correct_rate": _fraction(recent_5["correct"].eq(1).sum(), recent_5_count),
        "recent_5_attempt_count_mean": 0.0 if recent_5_count == 0 else float(recent_5_attempts.mean()),
        "recent_5_time_on_task_mean_seconds": 0.0 if len(recent_5_times) == 0 else float(recent_5_times.mean()),
        "recent_5_attempt_count_median": _quantile_or_zero(recent_5_attempts, .5),
        "recent_5_attempt_count_p90": _quantile_or_zero(recent_5_attempts, .9),
        "recent_5_time_on_task_median_seconds": _quantile_or_zero(recent_5_times, .5),
        "recent_5_time_on_task_p90_seconds": _quantile_or_zero(recent_5_times, .9),
        "recent_10_correct_trend": _ols_slope(positions, recent_10["correct"].astype(float).to_numpy()),
        "recent_10_log1p_attempt_count_trend": _ols_slope(positions, np.log1p(recent_10["attempt_count"].astype(float).to_numpy())),
        "recent_10_log1p_time_on_task_seconds_trend": _ols_slope(positions[finite_time_mask], np.log1p(recent_10_times[finite_time_mask])),
        "assignment_time_on_task_mean_seconds": 0.0 if len(observed_prior_times) == 0 else float(observed_prior_times.mean()),
        "assignment_time_on_task_std_seconds": 0.0 if len(observed_prior_times) < 2 else float(np.std(observed_prior_times, ddof=1)),
        "assignment_time_on_task_p90_seconds": _quantile_or_zero(pd.Series(observed_prior_times), .9),
        "previous_assignment_time_on_task_seconds": 0.0 if not np.isfinite(prior_times[-1]) else float(prior_times[-1]),
        "assignment_time_on_task_observed_rate": _fraction(len(observed_prior_times), len(prior)),
        "target_binary_mastered": int(target.assignment_completed),
        "target_regression_problems": float(raw_counts.get(target.log_id, 0)),
        "target_regression_time_on_task": float(target.time_on_task),
    }
    for problem_type in PROBLEM_TYPES:
        record[f"problem_type_{problem_type}_rate"] = _fraction(history["problem_type"].eq(problem_type).sum(), interactions)
    for column in ("hint_tutoring_available", "explanation_tutoring_available", "scaffold_tutoring_available"):
        record[f"{column}_rate"] = _fraction(history[column].sum(), interactions)
    return record


def preprocess_dataframe(plogs, adets, alogs, pdets, lifecycle, interaction_filter=None):
    assignment_logs, sb_plogs, lifecycle = _contexts(plogs, adets, alogs, pdets, lifecycle)
    candidates, histories = _candidates(assignment_logs, lifecycle)
    raw_counts = sb_plogs.groupby("log_id").size()
    clean_plogs = _clean_behavior(sb_plogs)
    if interaction_filter is not None:
        clean_plogs = interaction_filter(clean_plogs)
    records = [
        _summarize(target, histories[target.student_id], clean_plogs, raw_counts)
        for target in candidates.itertuples(index=False)
    ]
    result = pd.DataFrame.from_records(records, columns=OUTPUT_COLUMNS).sort_values("source_student_id", kind="mergesort").reset_index(drop=True)
    return result

DATASETS_ROOT = Path(__file__).resolve().parents[1]
_Base2019NSBAdapter = _load_dataset_module('_prepared_assistments2019_nsb', DATASETS_ROOT / 'assistments2019-2020/preprocess_nsb.py')._Base2019NSBAdapter
class _Base2019SBAdapter(_Base2019NSBAdapter):
    label = "assistments2019_sb"
    branch_file = "preprocess_sb.py"
    branch_module_name = "_prepared_assistments2019_sb"
    time_zero_is_valid = True
    filter_description = (
        "raw attempt_count P99.9 and log1p-IQR fences on plogs time_on_task (seconds) inside the "
        "behavioral pool only; alogs assignment times, raw_counts and persisted mastery untouched"
    )

    def __init__(self, split_module=None, branch_module=None, data_dir: Path | None = None,
                 plogs=None, adets=None, pdets=None, alogs=None, lifecycle=None,
                 lifecycle_cache_path: Path | None = None, workers: int = 1):
        super().__init__(split_module, branch_module, data_dir, plogs, adets, pdets, workers)
        self._alogs = alogs
        self._lifecycle = lifecycle
        self.lifecycle_cache_path = lifecycle_cache_path

    def source_paths(self):
        paths = super().source_paths()
        paths.append(self.data_path / "alogs.csv")
        paths.append(self.data_path / "slogs.csv")
        paths.append(self._lifecycle_cache_file())
        return paths

    def _lifecycle_cache_file(self):
        return Path(self.lifecycle_cache_path) if self.lifecycle_cache_path is not None else self.split_module.LIFECYCLE_CACHE_PATH

    def _read_sources(self):
        if self._plogs is not None:
            return (*super()._read_sources(), self._alogs, self._lifecycle)
        cache_file = self._lifecycle_cache_file()
        if not cache_file.exists():
            raise FileNotFoundError(
                f"Lifecycle cache {cache_file} is missing for {self.label}"
            )
        alogs = self._read_csv(self.data_path / "alogs.csv", (
            "log_id", "student_id", "assignment_id", "start_time", "assignment_completed", "time_on_task",
        ))
        lifecycle = self._load_lifecycle_cache(self.data_path / "slogs.csv", cache_file)
        plogs, adets, pdets = super()._read_sources()
        return plogs, adets, pdets, alogs, lifecycle

    def load(self):
        plogs, adets, pdets, alogs, lifecycle = self._read_sources()
        artifact = self.branch_module.preprocess_dataframe(plogs, adets, alogs, pdets, lifecycle)
        return {
            "plogs": plogs,
            "adets": adets,
            "pdets": pdets,
            "alogs": alogs,
            "lifecycle": lifecycle,
            "artifact": artifact,
            "provisional_students": sorted(artifact["source_student_id"].astype(str).unique().tolist()),
        }

    def interactions(self, context):
        branch = self.branch_module
        _assignment_logs, sb_plogs, _lifecycle = branch._contexts(
            context["plogs"], context["adets"], context["alogs"], context["pdets"], context["lifecycle"]
        )
        return branch._clean_behavior(sb_plogs)

    def build_rows(self, context, students, interaction_filter):
        students = {str(student) for student in map(str, students)}
        plogs = context["plogs"].loc[context["plogs"]["student_id"].astype(str).isin(students)]
        alogs = context["alogs"].loc[context["alogs"]["student_id"].astype(str).isin(students)]
        return self.branch_module.preprocess_dataframe(
            plogs, context["adets"], alogs, context["pdets"], context["lifecycle"],
            interaction_filter=interaction_filter,
        )


def create_adapter():
    return _Base2019SBAdapter()
