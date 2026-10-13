import argparse
import tempfile
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
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    fingerprint = _source_fingerprint(source_path)
    source_path, source_size, source_mtime_ns = fingerprint
    cache_path = Path(cache_path)
    if cache_path.is_symlink() or cache_path.resolve() == Path(source_path):
        raise ValueError("Lifecycle cache must not overwrite its source or use a symlink")
    if cache_path.exists() and not rebuild:
        connection = None
        try:
            connection = duckdb.connect(str(cache_path), read_only=True)
            metadata = connection.execute(
                "SELECT source_path, source_size, source_mtime_ns FROM lifecycle_cache_metadata"
            ).fetchone()
            if metadata == fingerprint:
                return connection.execute("SELECT * FROM lifecycle ORDER BY log_id").fetchdf()
        except duckdb.Error:
            pass
        finally:
            if connection is not None:
                connection.close()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Building lifecycle cache from {source_path} (workers={workers})", flush=True)
    descriptor, name = tempfile.mkstemp(prefix=".lifecycle-", suffix=".duckdb", dir=cache_path.parent)
    os.close(descriptor)
    temporary = Path(name)
    temporary.unlink()
    connection = None
    try:
        connection = duckdb.connect(str(temporary))
        connection.execute(f"SET threads = {workers}")
        connection.execute("SET memory_limit = '4GB'")
        connection.execute("SET enable_progress_bar = false")
        connection.execute('\n            CREATE TABLE lifecycle AS\n            SELECT\n                log_id,\n                max(parsed_timestamp) FILTER (WHERE parsed_timestamp IS NOT NULL) AS final_action_timestamp,\n                count(*) FILTER (WHERE parsed_timestamp IS NULL)::BIGINT AS invalid_timestamp_count\n            FROM (\n                SELECT\n                    try_cast(log_id AS BIGINT) AS log_id,\n                    try_cast(timestamp AS TIMESTAMPTZ) AS parsed_timestamp\n                FROM read_csv_auto(?, all_varchar = true)\n            )\n            WHERE log_id IS NOT NULL\n            GROUP BY log_id\n            ', [source_path])
        connection.execute(
            "CREATE TABLE lifecycle_cache_metadata (source_path VARCHAR, source_size BIGINT, source_mtime_ns BIGINT)"
        )
        connection.execute("INSERT INTO lifecycle_cache_metadata VALUES (?, ?, ?)", list(fingerprint))
        frame = connection.execute("SELECT * FROM lifecycle ORDER BY log_id").fetchdf()
        if _source_fingerprint(Path(source_path)) != fingerprint:
            raise ValueError("Lifecycle source changed while constructing cache")
        connection.close()
        connection = None
        os.replace(temporary, cache_path)
        print(f"Lifecycle cache ready: {len(frame)} log IDs", flush=True)
        return frame
    finally:
        if connection is not None:
            connection.close()
        temporary.unlink(missing_ok=True)
        Path(str(temporary) + ".wal").unlink(missing_ok=True)


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
