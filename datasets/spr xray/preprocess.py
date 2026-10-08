import argparse
import hashlib
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd

SOURCE_HASHES = {
    "spr_xray_embeddings.npy": "5d434126ca0b47506358b1d03e93b854b2b1adc0e00b263825e0054ae7a6d5b1",
    "spr_xray_manifest.csv": "132e1c9b7ac5dbbb01ffa999b9aa3fccee0a4cad9d91afc3a947601b8b2c9849",
    "cohort_selection.csv": "a28fce5a2cd41df82711f77dcbd728a13f79a0585efdbd6aeb46f04c1f673eb4",
}
OUTPUT_HASHES = {
    "spr_xray_embeddings.npy": "f54560bd3c39305f3f444c855c1a1f10702415ddab8ced1708cc41582d34e585",
    "spr_xray_manifest.csv": "70234bc8fc5164f91a4302e74a48f385494a7a2921dcfab31f914c834c6af12c",
    "cohort_selection.csv": SOURCE_HASHES["cohort_selection.csv"],
}


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def preprocess(input_path, output_path, selection_path, source_hashes=SOURCE_HASHES,
               output_hashes=OUTPUT_HASHES, expected_rows=4013):
    input_path, output_path, selection_path = map(Path, (input_path, output_path, selection_path))
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"Refusing existing SPR cohort: {output_path}")
    inputs = {name: input_path / name for name in ("spr_xray_embeddings.npy", "spr_xray_manifest.csv")}
    inputs["cohort_selection.csv"] = selection_path
    if output_path.resolve() == input_path.resolve() or input_path.resolve().is_relative_to(output_path.resolve()):
        raise ValueError("SPR output must not replace its source")
    for name, path in inputs.items():
        if path.is_symlink() or not path.is_file() or digest(path) != source_hashes[name]:
            raise ValueError(f"Install the official full SPR inputs and selection map: {name}")
    selection = pd.read_csv(selection_path)
    if len(selection) != expected_rows or not np.array_equal(selection["subset_row"], np.arange(expected_rows)):
        raise ValueError("SPR selection map differs from the official cohort order")
    rows = selection["source_embedding_row"].to_numpy()
    embeddings = np.load(inputs["spr_xray_embeddings.npy"], mmap_mode="r", allow_pickle=False)
    manifest = pd.read_csv(inputs["spr_xray_manifest.csv"], float_precision="round_trip")
    chosen = manifest.set_index("embedding_row").loc[rows].reset_index()
    if not np.array_equal(chosen["source_image_id"], selection["source_image_id"]):
        raise ValueError("SPR image identities differ from the official selection map")
    chosen["embedding_row"] = np.arange(expected_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".spr-cohort-", dir=output_path.parent))
    try:
        np.save(staging / "spr_xray_embeddings.npy", np.asarray(embeddings[rows], dtype=np.float32), allow_pickle=False)
        chosen.to_csv(staging / "spr_xray_manifest.csv", index=False)
        shutil.copyfile(selection_path, staging / "cohort_selection.csv")
        for name, expected in output_hashes.items():
            if digest(staging / name) != expected:
                raise ValueError(f"Reconstructed SPR cohort does not match the official artifact: {name}")
        if output_path.exists() or output_path.is_symlink():
            raise FileExistsError(f"Refusing existing SPR cohort: {output_path}")
        staging.rename(output_path)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print(f"SPR: official {expected_rows}-image cohort written to {output_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Reconstruct the exact paired SPR cohort from the full frozen embeddings and saved image selection; no new sampling or embedding extraction.")
    parser.add_argument("--input", type=Path, default=Path(__file__).parent)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[2] / "artifacts/runs/spr-4013/source")
    parser.add_argument("--selection", type=Path, default=Path(__file__).parent / "cohort_selection.csv")
    args = parser.parse_args()
    preprocess(args.input, args.output, args.selection)


if __name__ == "__main__":
    main()
