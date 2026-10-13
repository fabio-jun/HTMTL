import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import time

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
MANIFEST_COLUMNS = ["embedding_row", "source_image_id", "relative_image_path", "target_age", "target_gender"]
EXPECTED_ROWS = 4013


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def _regular(path, description):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"SPR {description} must be a regular file: {path}")
    return path


def _integer(values, name):
    values = pd.to_numeric(values, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
        raise ValueError(f"SPR {name} must be finite integral values")
    return values.astype(np.int64)


def _labels(dataset_root):
    root = Path(dataset_root)
    age_path, gender_path = (_regular(root / name, name) for name in ("train_age.csv", "train_gender.csv"))
    age, gender = pd.read_csv(age_path), pd.read_csv(gender_path)
    if set(("imageId", "age")) - set(age) or set(("imageId", "gender")) - set(gender):
        raise ValueError("SPR labels require imageId/age and imageId/gender columns")
    age_ids, gender_ids = _integer(age["imageId"], "image IDs"), _integer(gender["imageId"], "image IDs")
    if len(age_ids) != len(set(age_ids)) or len(gender_ids) != len(set(gender_ids)) or set(age_ids) != set(gender_ids):
        raise ValueError("SPR age and gender labels must be one-to-one over the same image IDs")
    ages = pd.to_numeric(age["age"], errors="raise").to_numpy(dtype=float)
    genders = _integer(gender["gender"], "gender labels")
    if not np.isfinite(ages).all() or not set(genders).issubset({0, 1}):
        raise ValueError("SPR ages must be finite and genders must be binary")
    age_frame = pd.DataFrame({"source_image_id": age_ids, "target_age": ages})
    gender_frame = pd.DataFrame({"source_image_id": gender_ids, "target_gender": genders})
    return age_frame.merge(gender_frame, on="source_image_id", validate="one_to_one")


def _selection(selection_path, selection_hash, expected_rows):
    selection_path = _regular(selection_path, "cohort selection")
    if selection_hash is not None and digest(selection_path) != selection_hash:
        raise ValueError("SPR selection map differs from the official frozen selection")
    selection = pd.read_csv(selection_path)
    expected = ["subset_row", "legacy_sample_row_index", "source_embedding_row", "source_image_id"]
    if list(selection.columns) != expected or len(selection) != expected_rows:
        raise ValueError("SPR selection map schema or row count differs from the official cohort")
    if not np.array_equal(_integer(selection["subset_row"], "subset rows"), np.arange(expected_rows)):
        raise ValueError("SPR selection map differs from the official cohort order")
    ids = _integer(selection["source_image_id"], "selected image IDs")
    if len(ids) != len(set(ids)):
        raise ValueError("SPR selection map contains duplicate image IDs")
    selection["source_image_id"] = ids
    return selection


def _selected_manifest(dataset_root, selection_path, selection_hash, expected_rows, manifest_hash=None):
    selection = _selection(selection_path, selection_hash, expected_rows)
    selected_ids = selection["source_image_id"].to_numpy()
    labels = _labels(dataset_root).set_index("source_image_id")
    if not set(selected_ids).issubset(labels.index):
        raise ValueError("SPR source labels are missing selected image IDs")
    chosen = labels.loc[selected_ids].reset_index()
    chosen.insert(0, "embedding_row", np.arange(expected_rows, dtype=np.int64))
    chosen.insert(2, "relative_image_path", [f"kaggle/kaggle/train/{image_id:06d}.png" for image_id in selected_ids])
    manifest = chosen.loc[:, MANIFEST_COLUMNS]
    expected_hash = OUTPUT_HASHES["spr_xray_manifest.csv"] if manifest_hash is None and expected_rows == EXPECTED_ROWS else manifest_hash
    if expected_hash is not None and hashlib.sha256(manifest.to_csv(index=False).encode("utf-8")).hexdigest() != expected_hash:
        raise ValueError("SPR source labels differ from the frozen selected target/schema contract")
    return manifest, selection


def _extractor():
    path = Path(__file__).with_name("extract_embeddings.py")
    spec = importlib.util.spec_from_file_location("_spr_extract_embeddings", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load SPR extractor: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def preprocess_raw(dataset_root, output_path, selection_path, *, device="cuda:0", batch_size=16,
                   encoder=None, selection_hash=SOURCE_HASHES["cohort_selection.csv"], expected_rows=EXPECTED_ROWS,
                   manifest_hash=None):
    dataset_root, output_path, selection_path = map(Path, (dataset_root, output_path, selection_path))
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"Refusing existing SPR cohort: {output_path}")
    if output_path.resolve() == dataset_root.resolve() or dataset_root.resolve().is_relative_to(output_path.resolve()) or selection_path.resolve().is_relative_to(output_path.resolve()):
        raise ValueError("SPR output must not replace its source or selection map")
    manifest, selection = _selected_manifest(dataset_root, selection_path, selection_hash, expected_rows, manifest_hash)
    image_paths = [Path(dataset_root) / path for path in manifest["relative_image_path"]]
    for path in image_paths:
        _regular(path, "selected training image")
    started = time.time()
    if encoder is None:
        extractor = _extractor()
        extractor.require_cuda(device)
        processor, model = extractor.load_processor_and_model(device)
        embeddings = extractor.extract_embeddings(processor, model, image_paths, batch_size, device)
        processor_info = extractor.processor_contract(processor)
        versions = extractor.package_versions()
    else:
        embeddings = encoder(image_paths)
        processor_info, versions = {"class": "injected-test-encoder"}, {}
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.shape != (expected_rows, 768) or not np.isfinite(embeddings).all():
        raise ValueError("SPR encoder must return finite FP32-compatible 768-coordinate embeddings")
    _publish_raw(output_path, embeddings, manifest, selection_path, selection, dataset_root, processor_info, versions,
                 batch_size, device, time.time() - started)


def _publish_raw(output_path, embeddings, manifest, selection_path, selection, dataset_root, processor, versions, batch_size, device, elapsed):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".spr-cohort-", dir=output_path.parent))
    try:
        np.save(staging / "spr_xray_embeddings.npy", embeddings, allow_pickle=False)
        manifest.to_csv(staging / "spr_xray_manifest.csv", index=False)
        shutil.copyfile(selection_path, staging / "cohort_selection.csv")
        artifacts = {name: digest(staging / name) for name in ("spr_xray_embeddings.npy", "spr_xray_manifest.csv", "cohort_selection.csv")}
        metadata = {"route": "kaggle_raw_selected_train", "model": {"id": "microsoft/rad-dino", "revision": "110cbc18d5133582e320b43d53bf5c44e410c936"},
                    "representation": "last_hidden_state[:, 0, :]", "embedding": {"shape": list(embeddings.shape), "dtype": str(embeddings.dtype)},
                    "processor": processor, "package_versions": versions, "batch_size": batch_size, "device": device,
                    "sources": {name: digest(Path(dataset_root) / name) for name in ("train_age.csv", "train_gender.csv")},
                    "selection_sha256": digest(selection_path), "selected_rows": len(selection), "artifact_sha256": artifacts,
                    "elapsed_seconds": elapsed}
        (staging / "spr_xray_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        if output_path.exists() or output_path.is_symlink():
            raise FileExistsError(f"Refusing existing SPR cohort: {output_path}")
        staging.rename(output_path)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def preprocess(input_path, output_path, selection_path, source_hashes=SOURCE_HASHES, output_hashes=OUTPUT_HASHES, expected_rows=EXPECTED_ROWS):
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
    selection = _selection(selection_path, source_hashes["cohort_selection.csv"], expected_rows)
    rows = _integer(selection["source_embedding_row"], "source embedding rows")
    embeddings = np.load(inputs["spr_xray_embeddings.npy"], mmap_mode="r", allow_pickle=False)
    manifest = pd.read_csv(inputs["spr_xray_manifest.csv"], float_precision="round_trip")
    chosen = manifest.set_index("embedding_row").loc[rows].reset_index()
    if not np.array_equal(chosen["source_image_id"].to_numpy(), selection["source_image_id"].to_numpy()):
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


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build the frozen SPR cohort from pinned full embeddings or Kaggle training images.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--raw", action="store_true", help="Treat --input as the Kaggle dataset root with kaggle/kaggle/train.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args(argv)
    if args.raw:
        preprocess_raw(args.input, args.output, args.selection, device=args.device, batch_size=args.batch_size)
    else:
        preprocess(args.input, args.output, args.selection)


if __name__ == "__main__":
    main()
