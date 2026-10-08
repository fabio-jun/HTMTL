import argparse
import os
from pathlib import Path

import duckdb
import importlib.util
import pandas as pd
import sys


VARIANT_DIR = Path(__file__).resolve().parent
PERIOD = "aug_to_oct"
DATA_DIR = VARIANT_DIR / PERIOD
NON_SKILL_BUILDER_OUTPUT_PATH = VARIANT_DIR / f"{PERIOD}_nsb_preprocessed.csv"
SKILL_BUILDER_OUTPUT_PATH = VARIANT_DIR / f"{PERIOD}_sb_preprocessed.csv"
LIFECYCLE_CACHE_PATH = VARIANT_DIR / f"{PERIOD}_lifecycle_cache.duckdb"


def _source_fingerprint(source_path: Path) -> tuple[str, int, int]:
    source_path = Path(source_path).resolve()
    stat = source_path.stat()
    return str(source_path), stat.st_size, stat.st_mtime_ns


def _resolve_workers(workers):
    if workers is None:
        return os.cpu_count() or 1
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be an integer greater than or equal to 1")
    return workers


def load_lifecycle_cache(source_path: Path, cache_path: Path, rebuild: bool = False, workers: int = 1) -> pd.DataFrame:
    source_path, source_size, source_mtime_ns = _source_fingerprint(source_path)
    cache_path = Path(cache_path)
    cache_existed = cache_path.exists()
    connection = duckdb.connect(str(cache_path))
    connection.execute(f"SET threads = {workers}")
    connection.execute("SET enable_progress_bar = false")
    try:
        has_metadata = connection.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_name = 'lifecycle_cache_metadata'"
        ).fetchone()[0]
        if has_metadata and not rebuild:
            metadata = connection.execute(
                "SELECT source_path, source_size, source_mtime_ns FROM lifecycle_cache_metadata"
            ).fetchone()
            if metadata != (source_path, source_size, source_mtime_ns):
                raise ValueError("Lifecycle cache is stale; rerun with rebuild=True")
            return connection.execute("SELECT * FROM lifecycle ORDER BY log_id").fetchdf()
        if not has_metadata and cache_existed and not rebuild:
            raise ValueError("Lifecycle cache metadata is missing; rerun with rebuild=True")
        connection.execute("DROP TABLE IF EXISTS lifecycle")
        connection.execute("DROP TABLE IF EXISTS lifecycle_cache_metadata")
        connection.execute(
            """
            CREATE TABLE lifecycle AS
            SELECT
                log_id,
                max(parsed_timestamp) FILTER (WHERE parsed_timestamp IS NOT NULL) AS final_action_timestamp,
                count(*) FILTER (WHERE parsed_timestamp IS NULL)::BIGINT AS invalid_timestamp_count
            FROM (
                SELECT
                    try_cast(log_id AS BIGINT) AS log_id,
                    try_cast(timestamp AS TIMESTAMPTZ) AS parsed_timestamp
                FROM read_csv_auto(?, all_varchar = true)
            )
            WHERE log_id IS NOT NULL
            GROUP BY log_id
            """,
            [source_path],
        )
        connection.execute(
            "CREATE TABLE lifecycle_cache_metadata (source_path VARCHAR, source_size BIGINT, source_mtime_ns BIGINT)"
        )
        connection.execute(
            "INSERT INTO lifecycle_cache_metadata VALUES (?, ?, ?)",
            [source_path, source_size, source_mtime_ns],
        )
        return connection.execute("SELECT * FROM lifecycle ORDER BY log_id").fetchdf()
    finally:
        connection.close()


def split_regimes(plogs: pd.DataFrame, adets: pd.DataFrame) -> dict[str, pd.DataFrame]:
    assignment_types = adets.loc[:, ["assignment_id", "assignment_type"]].drop_duplicates("assignment_id")
    joined = plogs.merge(assignment_types, on="assignment_id", how="left")
    non_skill_builder = joined.loc[joined["assignment_type"].eq("problem_set")].reset_index(drop=True)
    skill_builder = joined.loc[joined["assignment_type"].eq("skill_builder")].reset_index(drop=True)
    return {"non_skill_builder": non_skill_builder, "skill_builder": skill_builder}


def _load_branch_module(name: str):
    qualified_name = f"_assistments2020_2021_{name}"
    spec = importlib.util.spec_from_file_location(qualified_name, VARIANT_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[qualified_name] = module
    spec.loader.exec_module(module)
    return module


def _read_csv(path: Path, columns: tuple[str, ...], workers: int) -> pd.DataFrame:
    connection = duckdb.connect()
    try:
        connection.execute(f"SET threads = {workers}")
        connection.execute("SET enable_progress_bar = false")
        projection = ", ".join(columns)
        return connection.execute(f"SELECT {projection} FROM read_csv_auto(?)", [str(path)]).fetchdf()
    finally:
        connection.close()


def write_preprocessed_datasets(
    data_dir: Path = DATA_DIR,
    non_skill_builder_output_path: Path = NON_SKILL_BUILDER_OUTPUT_PATH,
    skill_builder_output_path: Path = SKILL_BUILDER_OUTPUT_PATH,
    lifecycle_cache_path: Path = LIFECYCLE_CACHE_PATH,
    rebuild_lifecycle_cache: bool = False,
    workers: int = None,
) -> dict[str, pd.DataFrame]:
    workers = _resolve_workers(workers)
    data_dir = Path(data_dir)
    non_skill_builder_output_path = Path(non_skill_builder_output_path)
    skill_builder_output_path = Path(skill_builder_output_path)
    if non_skill_builder_output_path.resolve() == skill_builder_output_path.resolve():
        raise ValueError("NSB and SB output paths must differ")
    source_paths = {path.resolve() for path in data_dir.glob("*.csv")}
    if {non_skill_builder_output_path.resolve(), skill_builder_output_path.resolve()} & source_paths:
        raise ValueError("Output paths must not overwrite raw sources")
    print(f"[assistments2020-2021] period={PERIOD} workers={workers}", flush=True)
    print("[assistments2020-2021] reading source tables", flush=True)
    adets = _read_csv(data_dir / "adets.csv", ("assignment_id", "assignment_type"), workers)
    pdets = _read_csv(data_dir / "pdets.csv", ("problem_id", "problem_type", "tutoring_types"), workers)
    alogs = _read_csv(data_dir / "alogs.csv", (
        "log_id", "student_id", "assignment_id", "start_time", "assignment_completed",
        "time_on_task",
    ), workers)
    plogs = _read_csv(data_dir / "plogs.csv", (
        "log_id", "student_id", "assignment_id", "problem_id", "start_time", "time_on_task",
        "answer_before_tutoring", "fraction_of_hints_used", "attempt_count", "answer_given",
        "problem_completed", "correct",
    ), workers)
    print("[assistments2020-2021] loading lifecycle cache", flush=True)
    lifecycle = load_lifecycle_cache(data_dir / "slogs.csv", lifecycle_cache_path, rebuild_lifecycle_cache, workers)
    print("[assistments2020-2021] non-skill-builder branch", flush=True)
    nsb = _load_branch_module("preprocess_nsb").preprocess_dataframe(plogs, adets, pdets, workers=workers)
    print("[assistments2020-2021] skill-builder branch", flush=True)
    sb = _load_branch_module("preprocess_sb").preprocess_dataframe(plogs, adets, alogs, pdets, lifecycle, workers=workers)
    print("[assistments2020-2021] writing outputs", flush=True)
    nsb.to_csv(non_skill_builder_output_path, index=False)
    sb.to_csv(skill_builder_output_path, index=False)
    return {"non_skill_builder": nsb, "skill_builder": sb}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild-lifecycle-cache", action="store_true")
    parser.add_argument("--workers", type=int, default=None)
    arguments = parser.parse_args()
    write_preprocessed_datasets(rebuild_lifecycle_cache=arguments.rebuild_lifecycle_cache, workers=arguments.workers)


if __name__ == "__main__":
    main()
