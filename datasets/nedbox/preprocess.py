#!/usr/bin/env python3
import ast
from pathlib import Path
import numpy as np
import pandas as pd
import sys

from fold_preparation import BasePreparedAdapter
from dataset_registry import _load_dataset_module

ROOT = Path(__file__).resolve().parent
OUTPUT_PATH = ROOT / "nedbox_preprocessed.csv"
FEATURE_COLUMNS = [
    "teasers", "sessions", "publications", "exercises", "answer_rate", "incomplete_rate", "no_response_rate",
    "score_ratio", "score_teaser_mean", "score_teaser_std", "score_last", "duration_total", "duration_median",
    "duration_p90", "session_duration_mean", "session_duration_last", "gap_mean", "gap_median", "gap_p90", "gap_last",
    "exercise_type_count", "theme_count", "video_rate", "context_known_rate", "difficulty_alpha_rate",
    "difficulty_n1_rate", "difficulty_n2_rate", "difficulty_n3_rate", "skill_kijken_rate", "skill_luisteren_rate",
    "skill_lezen_rate", "skill_woordenschat_rate", "skill_grammatica_rate", "skill_schrijven_rate",
    "skill_spreken_rate", "selected_is_video", "selected_theme_count", "selected_publications", "selected_exercises",
    "selected_max_score", "selected_difficulty_alpha_rate", "selected_difficulty_n1_rate",
    "selected_difficulty_n2_rate", "selected_difficulty_n3_rate", "selected_skill_kijken_rate",
    "selected_skill_luisteren_rate", "selected_skill_lezen_rate", "selected_skill_woordenschat_rate",
    "selected_skill_grammatica_rate", "selected_skill_schrijven_rate", "selected_skill_spreken_rate",
]
TARGET_COLUMNS = ["target_binary_dropout", "target_regression_score"]
LOG1P_COLUMNS = [
    "teasers", "sessions", "publications", "exercises", "duration_total", "duration_median", "duration_p90",
    "session_duration_mean", "session_duration_last", "gap_mean", "gap_median", "gap_p90", "gap_last",
    "exercise_type_count", "theme_count", "selected_theme_count", "selected_publications", "selected_exercises",
    "selected_max_score",
]
DIFFICULTIES = ("alpha", "n1", "n2", "n3")
SKILLS = ("kijken", "luisteren", "lezen", "woordenschat", "grammatica", "schrijven", "spreken")


def _themes(value):
    if not isinstance(value, str):
        return []
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return []
    return parsed if isinstance(parsed, (list, tuple, set)) else []


def _rate(values, value):
    return float((values == value).mean()) if len(values) else np.nan


def _answered(values):
    return values.map(lambda value: isinstance(value, (bool, np.bool_)) and bool(value))


def _catalog(exercises, publications, teasers):
    catalog = exercises.merge(publications, on="publication_id", how="left")
    catalog = catalog.merge(teasers[["teaser_id", "is_video", "themes"]], on="teaser_id", how="left")
    for column in ("difficulty_level", "language_skill"):
        catalog[column] = catalog[column].astype("string").str.strip().str.lower()
    return catalog


def _deduplicate(interactions):
    frame = interactions.rename(columns={"session_nmuber": "session_number"}).drop_duplicates().copy()
    return frame.reset_index(drop=True), len(interactions) - len(frame)


def _latest_user_exercise(frame):
    subordinate = [column for column in frame.columns if column not in {"user_id", "exercise_id", "session_number"}]
    frame = frame.sort_values(["user_id", "exercise_id", "session_number", *subordinate], kind="mergesort")
    frame = frame.drop_duplicates(["user_id", "exercise_id"], keep="last")
    return frame.reset_index(drop=True), None


def _completion(group, catalog_counts):
    answered = group.loc[_answered(group["is_answered"]), ["difficulty_level", "exercise_id"]].drop_duplicates()
    started = answered.groupby("difficulty_level")["exercise_id"].nunique()
    incomplete = any(0 < count < catalog_counts.get(difficulty, 0) for difficulty, count in started.items())
    return bool(incomplete), int(len(started))


def _selected_context(teaser_id, catalog, teasers):
    rows = catalog.loc[catalog["teaser_id"] == teaser_id]
    teaser = teasers.loc[teasers["teaser_id"] == teaser_id].iloc[0]
    total = len(rows)

    result = {
        "selected_is_video": float(bool(teaser["is_video"])) if pd.notna(teaser["is_video"]) else 0.0,
        "selected_theme_count": float(len(_themes(teaser["themes"]))),
        "selected_publications": float(rows["publication_id"].nunique()),
        "selected_exercises": float(total),
        "selected_max_score": float(pd.to_numeric(rows["max_score"], errors="coerce").fillna(0).sum()),
    }
    for difficulty in DIFFICULTIES:
        result[f"selected_difficulty_{difficulty}_rate"] = _rate(rows["difficulty_level"], difficulty)
    for skill in SKILLS:
        result[f"selected_skill_{skill}_rate"] = _rate(rows["language_skill"], skill)
    return result


def _history_features(history, catalog_counts):
    result = {column: np.nan for column in FEATURE_COLUMNS[:35]}
    for column in ("teasers", "sessions", "publications", "exercises", "duration_total", "exercise_type_count", "theme_count"):
        result[column] = 0.0

    if history.empty:
        return result

    known = history.loc[history["teaser_id"].notna()].copy()
    sessions = history.sort_values(["session_number", "session_id"], kind="mergesort").drop_duplicates("session_id")
    known_sessions = known.drop_duplicates("session_id")

    result.update({
        "teasers": float(known["teaser_id"].nunique()), "sessions": float(sessions["session_id"].nunique()),
        "publications": float(history["publication_id"].nunique()), "exercises": float(history["exercise_id"].nunique()),
        "answer_rate": float(_answered(history["is_answered"]).mean()),
        "duration_total": float(history["duration"].sum()), "duration_median": float(history["duration"].median()),
        "duration_p90": float(history["duration"].quantile(.9)), "exercise_type_count": float(history["type"].nunique()),
    })

    session_duration = history.groupby("session_id", sort=False)["duration"].sum()
    result["session_duration_mean"] = float(session_duration.mean())
    last_session = sessions.iloc[-1]["session_id"]
    result["session_duration_last"] = float(session_duration.loc[last_session])

    gaps = sessions.loc[sessions["session_number"] != 1, "session_time_gap"]
    result.update({"gap_mean": gaps.mean(), "gap_median": gaps.median(), "gap_p90": gaps.quantile(.9), "gap_last": gaps.iloc[-1] if len(gaps) else np.nan})

    themes = set()
    for value in known.drop_duplicates("teaser_id")["themes"]:
        themes.update(_themes(value))
    result["theme_count"] = float(len(themes))
    result["video_rate"] = float(known.drop_duplicates("teaser_id")["is_video"].eq(True).mean()) if len(known) else np.nan
    result["context_known_rate"] = float(len(known_sessions) / len(sessions)) if len(sessions) else np.nan

    incomplete_count = 0
    started_count = 0
    for teaser_id, group in known.groupby("teaser_id", sort=False):
        incomplete, started = _completion(group, catalog_counts.get(teaser_id, {}))
        incomplete_count += int(incomplete)
        started_count += int(started > 0)

    result["incomplete_rate"] = incomplete_count / started_count if started_count else np.nan
    result["no_response_rate"] = float(known.groupby("teaser_id")["is_answered"].apply(lambda values: not _answered(values).any()).mean()) if len(known) else np.nan

    responses = history.loc[_answered(history["is_answered"]) & (history["max_score"] > 0)]
    result["score_ratio"] = float(responses["score"].sum() / responses["max_score"].sum()) if len(responses) else np.nan

    known_responses = known.loc[_answered(known["is_answered"]) & (known["max_score"] > 0)]
    teaser_starts = known.groupby("teaser_id")["session_number"].min().to_dict()
    ratios = []
    for teaser_id, group in known_responses.groupby("teaser_id", sort=False):
        ratios.append((teaser_starts[teaser_id], str(teaser_id), group["score"].sum() / group["max_score"].sum()))
    if ratios:
        ratios.sort(key=lambda value: (value[0], value[1]))
        values = [value[2] for value in ratios]
        result["score_teaser_mean"] = float(np.mean(values)); result["score_teaser_std"] = float(np.std(values, ddof=0)); result["score_last"] = float(values[-1])
    else:
        result["score_teaser_mean"] = result["score_teaser_std"] = result["score_last"] = np.nan

    votes = known_sessions
    for difficulty in DIFFICULTIES:
        result[f"difficulty_{difficulty}_rate"] = _rate(votes["difficulty_level"], difficulty)

    for skill in SKILLS:
        result[f"skill_{skill}_rate"] = _rate(votes["language_skill"], skill)

    return result


def preprocess_dataframes(interactions, exercises, publications, teasers):
    exact_stream, exact_duplicates_removed = _deduplicate(interactions)
    catalog = _catalog(exercises, publications, teasers)
    joined = exact_stream.merge(catalog, on="exercise_id", how="left", suffixes=("", "_catalog"))
    catalog_counts = {teaser: group.groupby("difficulty_level")["exercise_id"].nunique().to_dict() for teaser, group in catalog.dropna(subset=["teaser_id"]).groupby("teaser_id")}

    candidates = joined.dropna(subset=["teaser_id"])
    starts = candidates.groupby(["user_id", "teaser_id"], as_index=False)["session_number"].min()
    selected = starts.sort_values(["user_id", "session_number", "teaser_id"], kind="mergesort").drop_duplicates("user_id", keep="last")

    records = []
    no_answer = no_denominator = 0

    for selected_row in selected.itertuples(index=False):
        user_id, teaser_id, boundary = selected_row.user_id, selected_row.teaser_id, selected_row.session_number
        target = candidates.loc[(candidates["user_id"] == user_id) & (candidates["teaser_id"] == teaser_id)]
        target, _ = _latest_user_exercise(target)
        answered = _answered(target["is_answered"])

        if not answered.any():
            no_answer += 1; continue

        responses = target.loc[answered & (target["max_score"] > 0)]
        if responses.empty or responses["max_score"].sum() <= 0:
            no_denominator += 1; continue

        incomplete, _ = _completion(target, catalog_counts[teaser_id])

        history = joined.loc[(joined["user_id"] == user_id) & (joined["session_number"] < boundary)]
        history, _ = _latest_user_exercise(history)

        record = {"user_id": user_id, "selected_teaser_id": teaser_id}
        record.update(_history_features(history, catalog_counts))
        record.update(_selected_context(teaser_id, catalog, teasers))
        record["target_binary_dropout"] = int(incomplete)
        record["target_regression_score"] = float(responses["score"].sum() / responses["max_score"].sum())
        records.append(record)

    frame = pd.DataFrame(records, columns=["user_id", "selected_teaser_id", *FEATURE_COLUMNS, *TARGET_COLUMNS])

    accounting = {
        "input_interactions": int(len(interactions)), "exact_duplicates_removed": int(exact_duplicates_removed),
        "exact_deduplicated_interactions": int(len(exact_stream)), "missing_teaser_interactions": int(joined["teaser_id"].isna().sum()),
        "selected_users": int(len(selected)), "excluded_no_answer": no_answer, "excluded_no_positive_denominator": no_denominator,
        "final_rows": int(len(frame)), "dropout_1": int(frame["target_binary_dropout"].sum()) if len(frame) else 0,
        "dropout_0": int((frame["target_binary_dropout"] == 0).sum()) if len(frame) else 0,
        "zero_prior_interactions": int((frame["sessions"] == 0).sum()) if len(frame) else 0,
        "zero_identifiable_prior_teasers": int((frame["teasers"] == 0).sum()) if len(frame) else 0,
    }

    return frame, accounting


def load_sources(root=ROOT):
    root = Path(root)
    return tuple(pd.read_csv(root / name) for name in ("interactions.csv", "exercises.csv", "publications.csv", "teasers.csv"))


def write_artifact():
    frame, _ = preprocess_dataframes(*load_sources())
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT_PATH, index=False)
    return frame


DATASETS_ROOT = Path(__file__).resolve().parents[1]
NEDBOX_DIR = Path(__file__).resolve().parent


def _module():
    return sys.modules[__name__]


class NedBoxArtifactAdapter:
    label = "nedbox"
    adapter_kind = "artifact"
    student_column = "user_id"
    interaction_student_column = "user_id"
    attempt_column = time_column = time_unit = None
    filter_description = "canonical NedBox artifact only; no outlier filtering"

    def __init__(self, artifact_path=NEDBOX_DIR / "nedbox_preprocessed.csv"):
        self.artifact_path = Path(artifact_path)
        self.module = _module()
        self.log1p_columns = self.module.LOG1P_COLUMNS

    def source_paths(self):
        return [self.artifact_path]

    def load(self):
        if self.artifact_path.is_symlink() or not self.artifact_path.is_file():
            raise ValueError(f"NedBox canonical CSV must be a regular non-symlink file: {self.artifact_path}")
        frame = pd.read_csv(self.artifact_path)
        columns = ["user_id", "selected_teaser_id", *self.module.FEATURE_COLUMNS, *self.module.TARGET_COLUMNS]
        frame = frame[columns]
        return {"artifact": frame, "provisional_students": sorted(frame["user_id"].astype(str))}

    def provisional_students(self, context):
        return context["provisional_students"]

    def interactions(self, context):
        raise ValueError("NedBox artifact mode has no interactions")

    def build_rows(self, context, students, interaction_filter):
        if interaction_filter is not None:
            raise ValueError("NedBox artifact mode forbids interaction filters")
        return context["artifact"].loc[context["artifact"]["user_id"].astype(str).isin(map(str, students))].copy()

    def prepare_features(self, frame):
        output = frame.copy()
        output[self.log1p_columns] = np.log1p(output[self.log1p_columns])
        return output


def create_adapter():
    return NedBoxArtifactAdapter()


def main():
    frame = write_artifact()
    print(f"rows={len(frame)}")


if __name__ == "__main__":
    main()
