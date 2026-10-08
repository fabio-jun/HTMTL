import numpy as np
import pandas as pd
import sys
from pathlib import Path

from fold_preparation import BasePreparedAdapter
from dataset_registry import _load_dataset_module

MIN_INTERACTIONS = 5
PROBLEM_TYPES = (
    "algebraic_expression", "number", "ungraded_open_response", "multiple_choice", "exact_match_case_sensitive",
    "exact_match_ignore_case", "check_all_that_apply", "exact_fraction", "numeric_expression", "ordering", "unknown",
)
METADATA_COLUMNS = (
    "source_student_id", "source_assignment_log_id", "source_assignment_id", "source_problem_id",
)
FEATURE_COLUMNS = (
    "interactions", "repeated_problem_rate", "assignment_count", "skill_builder_assignment_rate",
    "problem_set_assignment_rate", "active_day_count", "history_span_days", "days_since_previous_interaction",
    "correct_rate", "attempt_count_mean", "attempt_count_std", "attempt_count_total", "attempt_count_p90",
    "time_on_task_mean_seconds", "time_on_task_std_seconds", "time_on_task_total_seconds", "time_on_task_p90_seconds",
    "answer_given_rate", *(f"problem_type_{name}_rate" for name in PROBLEM_TYPES), "hint_tutoring_available_rate",
    "explanation_tutoring_available_rate", "scaffold_tutoring_available_rate", "attempt_count_median",
    "attempt_count_p75", "attempt_count_iqr", "attempt_count_above_p90_rate", "time_on_task_median_seconds",
    "time_on_task_p75_seconds", "time_on_task_iqr_seconds", "time_on_task_above_p90_rate", "recent_5_observation_count",
    "recent_5_correct_rate", "recent_5_attempt_count_mean", "recent_5_time_on_task_mean_seconds",
    "recent_5_attempt_count_median", "recent_5_attempt_count_p90", "recent_5_time_on_task_median_seconds",
    "recent_5_time_on_task_p90_seconds", "recent_10_correct_trend", "recent_10_log1p_attempt_count_trend",
    "recent_10_log1p_time_on_task_seconds_trend", "previous_problem_correct", "previous_problem_attempt_count",
    "previous_problem_time_on_task_seconds",
)
TARGET_COLUMNS = (
    "target_binary_correct", "target_regression_time_on_task", "target_regression_attempts",
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


def _source_context(plogs: pd.DataFrame, adets: pd.DataFrame, pdets: pd.DataFrame) -> pd.DataFrame:
    assignments = adets.loc[:, ["assignment_id", "assignment_type"]].drop_duplicates("assignment_id")
    problem_columns = ["problem_id", "problem_type"] + [
        column for column in (
            "hint_tutoring_available", "explanation_tutoring_available", "scaffold_tutoring_available"
        ) if column in pdets
    ]
    problems = pdets.loc[:, problem_columns].drop_duplicates("problem_id")
    frame = plogs.merge(assignments, on="assignment_id", how="inner")
    frame = frame.merge(problems, on="problem_id", how="left")
    frame["start_time"] = pd.to_datetime(frame["start_time"], utc=True, errors="coerce")
    frame["problem_type"] = frame["problem_type"].map(_normalized_problem_type)
    for name in ("hint", "explanation", "scaffold"):
        frame[f"{name}_tutoring_available"] = _tutoring_available(frame, name)
    return frame


def _clean(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = ("correct", "time_on_task", "attempt_count")
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    valid = (
        frame["start_time"].notna() & frame["correct"].isin([0, 1])
        & frame["time_on_task"].gt(0) & np.isfinite(frame["time_on_task"])
        & frame["attempt_count"].gt(0) & np.isfinite(frame["attempt_count"])
        & ~(frame["correct"].eq(1) & frame["attempt_count"].ne(1))
    )
    return frame.loc[valid].copy()


def _targets_and_eligible(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = frame.sort_values(["student_id", "start_time", "log_id", "problem_id"], kind="mergesort").reset_index(drop=True)
    target_pool = ordered.loc[ordered["assignment_type"].eq("problem_set")]
    targets = target_pool.groupby("student_id", sort=False).tail(1).copy()
    target_times = targets.set_index("student_id")["start_time"]
    with_targets = ordered.join(target_times.rename("target_start_time"), on="student_id", how="inner")
    history_count = with_targets.loc[with_targets["start_time"].lt(with_targets["target_start_time"])].reset_index(drop=True)
    eligible_ids = history_count.groupby("student_id").size().loc[lambda counts: counts.ge(MIN_INTERACTIONS - 1)].index
    return ordered.loc[ordered["student_id"].isin(eligible_ids)].copy(), targets.loc[targets["student_id"].isin(eligible_ids)].copy()


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


def _summaries(histories: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    records = []
    target_by_student = targets.set_index("student_id")
    for student_id, history in histories.groupby("student_id", sort=True):
        history = history.sort_values(["start_time", "log_id", "problem_id"], kind="mergesort")
        target = target_by_student.loc[student_id]
        interactions = len(history)
        attempts = history["attempt_count"].astype(float)
        times = history["time_on_task"].astype(float)
        assignment_types = history.drop_duplicates("assignment_id")["assignment_type"]
        attempt_median, attempt_p75, attempt_iqr, attempt_above_p90_rate = _robust_summaries(attempts)
        time_median, time_p75, time_iqr, time_above_p90_rate = _robust_summaries(times)
        recent_5 = history.tail(min(5, interactions))
        recent_5_count = len(recent_5)
        recent_5_attempts = recent_5["attempt_count"].astype(float)
        recent_5_times = recent_5["time_on_task"].astype(float)
        recent_10 = history.tail(min(10, interactions))
        recent_10_count = len(recent_10)
        positions = np.arange(recent_10_count, dtype=float)
        recent_10_times = recent_10["time_on_task"].to_numpy(dtype=float)
        finite_time_mask = np.isfinite(recent_10_times)
        previous = history.iloc[-1]
        record = {
            "source_student_id": student_id,
            "source_assignment_log_id": target["log_id"],
            "source_assignment_id": target["assignment_id"],
            "source_problem_id": target["problem_id"],
            "interactions": interactions,
            "repeated_problem_rate": float(1 - history["problem_id"].nunique() / interactions),
            "assignment_count": int(len(assignment_types)),
            "skill_builder_assignment_rate": float(assignment_types.eq("skill_builder").mean()),
            "problem_set_assignment_rate": float(assignment_types.eq("problem_set").mean()),
            "active_day_count": int(history["start_time"].dt.date.nunique()),
            "history_span_days": float((history["start_time"].iloc[-1] - history["start_time"].iloc[0]).total_seconds() / 86400),
            "days_since_previous_interaction": float((target["start_time"] - history["start_time"].iloc[-1]).total_seconds() / 86400),
            "correct_rate": float(history["correct"].mean()),
            "attempt_count_mean": float(attempts.mean()),
            "attempt_count_std": 0.0 if interactions < 2 else float(attempts.std(ddof=1)),
            "attempt_count_total": float(attempts.sum()),
            "attempt_count_p90": float(attempts.quantile(.9)),
            "time_on_task_mean_seconds": float(times.mean()),
            "time_on_task_std_seconds": 0.0 if interactions < 2 else float(times.std(ddof=1)),
            "time_on_task_total_seconds": float(times.sum()),
            "time_on_task_p90_seconds": float(times.quantile(.9)),
            "answer_given_rate": float(pd.to_numeric(history["answer_given"], errors="coerce").fillna(0).eq(1).mean()),
            "attempt_count_median": attempt_median,
            "attempt_count_p75": attempt_p75,
            "attempt_count_iqr": attempt_iqr,
            "attempt_count_above_p90_rate": attempt_above_p90_rate,
            "time_on_task_median_seconds": time_median,
            "time_on_task_p75_seconds": time_p75,
            "time_on_task_iqr_seconds": time_iqr,
            "time_on_task_above_p90_rate": time_above_p90_rate,
            "recent_5_observation_count": int(recent_5_count),
            "recent_5_correct_rate": 0.0 if recent_5_count == 0 else float(recent_5["correct"].mean()),
            "recent_5_attempt_count_mean": 0.0 if recent_5_count == 0 else float(recent_5_attempts.mean()),
            "recent_5_time_on_task_mean_seconds": 0.0 if recent_5_count == 0 else float(recent_5_times.mean()),
            "recent_5_attempt_count_median": _quantile_or_zero(recent_5_attempts, .5),
            "recent_5_attempt_count_p90": _quantile_or_zero(recent_5_attempts, .9),
            "recent_5_time_on_task_median_seconds": _quantile_or_zero(recent_5_times, .5),
            "recent_5_time_on_task_p90_seconds": _quantile_or_zero(recent_5_times, .9),
            "recent_10_correct_trend": _ols_slope(positions, recent_10["correct"].astype(float).to_numpy()),
            "recent_10_log1p_attempt_count_trend": _ols_slope(positions, np.log1p(recent_10["attempt_count"].astype(float).to_numpy())),
            "recent_10_log1p_time_on_task_seconds_trend": _ols_slope(positions[finite_time_mask], np.log1p(recent_10_times[finite_time_mask])),
            "previous_problem_correct": int(previous["correct"]),
            "previous_problem_attempt_count": float(previous["attempt_count"]),
            "previous_problem_time_on_task_seconds": float(previous["time_on_task"]),
            "target_binary_correct": int(target["correct"]),
            "target_regression_time_on_task": float(target["time_on_task"]),
            "target_regression_attempts": float(target["attempt_count"]),
        }
        for problem_type in PROBLEM_TYPES:
            record[f"problem_type_{problem_type}_rate"] = float(history["problem_type"].eq(problem_type).mean())
        for column in ("hint_tutoring_available", "explanation_tutoring_available", "scaffold_tutoring_available"):
            record[f"{column}_rate"] = float(history[column].mean())
        records.append(record)
    return pd.DataFrame.from_records(records, columns=OUTPUT_COLUMNS)


def preprocess_dataframe(plogs: pd.DataFrame, adets: pd.DataFrame, pdets: pd.DataFrame, interaction_filter=None) -> pd.DataFrame:
    clean = _clean(_source_context(plogs, adets, pdets))
    if interaction_filter is not None:
        clean = interaction_filter(clean)
    final, targets = _targets_and_eligible(clean)
    target_times = targets.set_index("student_id")["start_time"]
    histories = final.join(target_times.rename("target_start_time"), on="student_id", how="inner")
    histories = histories.loc[histories["start_time"].lt(histories["target_start_time"])].drop(columns="target_start_time").reset_index(drop=True)
    result = _summaries(histories, targets).sort_values("source_student_id", kind="mergesort").reset_index(drop=True)
    return result


DATASETS_ROOT = Path(__file__).resolve().parents[1]
class _Base2019NSBAdapter(BasePreparedAdapter):
    label = "assistments2019_nsb"
    dataset_dir = "assistments2019-2020"
    split_module_name = "_prepared_assistments2019_split_regimes"
    branch_file = "preprocess_nsb.py"
    branch_module_name = "_prepared_assistments2019_nsb"
    student_column = "source_student_id"
    interaction_student_column = "student_id"
    attempt_column = "attempt_count"
    time_column = "time_on_task"
    time_unit = "seconds"
    filter_description = (
        "raw attempt_count P99.9 and log1p-IQR fences on time_on_task (seconds), applied to the "
        "cleaned behavioral pool before target selection and histories"
    )

    def __init__(self, split_module=None, branch_module=None, data_dir: Path | None = None,
                 plogs=None, adets=None, pdets=None, workers: int = 1):
        self._split_module = split_module
        self._branch_module = branch_module
        self.data_dir = data_dir
        self._plogs = plogs
        self._adets = adets
        self._pdets = pdets
        if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
            raise ValueError(f"workers must be a positive integer, got {workers!r}")
        self.workers = workers

    @property
    def split_module(self):
        if self._split_module is None:
            self._split_module = _load_dataset_module(
                self.split_module_name,
                DATASETS_ROOT / self.dataset_dir / "preprocess_split_regimes.py",
            )
        return self._split_module

    @property
    def branch_module(self):
        if self._branch_module is None:
            self._branch_module = _load_dataset_module(
                self.branch_module_name,
                DATASETS_ROOT / self.dataset_dir / self.branch_file,
            )
        return self._branch_module

    @property
    def data_path(self):
        return Path(self.data_dir) if self.data_dir is not None else self.split_module.DATA_DIR

    def source_paths(self):
        return [
            self.data_path / "adets.csv",
            self.data_path / "pdets.csv",
            self.data_path / "plogs.csv",
        ]

    def _read_csv(self, path, columns):
        return self.split_module._read_csv(path, columns)

    def _load_lifecycle_cache(self, slogs_path, cache_path):
        return self.split_module.load_lifecycle_cache(slogs_path, cache_path)

    def _read_sources(self):
        if self._plogs is not None:
            return self._plogs, self._adets, self._pdets
        data_dir = self.data_path
        adets = self._read_csv(data_dir / "adets.csv", ("assignment_id", "assignment_type"))
        pdets = self._read_csv(data_dir / "pdets.csv", ("problem_id", "problem_type", "tutoring_types"))
        plogs = self._read_csv(data_dir / "plogs.csv", (
            "log_id", "student_id", "assignment_id", "problem_id", "start_time", "time_on_task",
            "answer_before_tutoring", "fraction_of_hints_used", "attempt_count", "answer_given",
            "problem_completed", "correct",
        ))
        return plogs, adets, pdets

    def load(self):
        plogs, adets, pdets = self._read_sources()
        artifact = self.branch_module.preprocess_dataframe(plogs, adets, pdets)
        return {
            "plogs": plogs,
            "adets": adets,
            "pdets": pdets,
            "artifact": artifact,
            "provisional_students": sorted(artifact["source_student_id"].astype(str).unique().tolist()),
        }

    def interactions(self, context):
        branch = self.branch_module
        return branch._clean(branch._source_context(context["plogs"], context["adets"], context["pdets"]))

    def build_rows(self, context, students, interaction_filter):
        subset = context["plogs"].loc[
            context["plogs"]["student_id"].astype(str).isin(map(str, students))
        ]
        return self.branch_module.preprocess_dataframe(
            subset, context["adets"], context["pdets"], interaction_filter=interaction_filter
        )


def create_adapter():
    return _Base2019NSBAdapter()
