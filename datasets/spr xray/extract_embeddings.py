import argparse
import hashlib
import io
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

DATA_DIR = Path(__file__).resolve().parent
AGE_CSV_PATH = DATA_DIR / "train_age.csv"
GENDER_CSV_PATH = DATA_DIR / "train_gender.csv"
EMBEDDINGS_PATH = DATA_DIR / "spr_xray_embeddings.npy"
MANIFEST_PATH = DATA_DIR / "spr_xray_manifest.csv"
METADATA_PATH = DATA_DIR / "spr_xray_metadata.json"

MODEL_ID = "microsoft/rad-dino"
MODEL_REVISION = "110cbc18d5133582e320b43d53bf5c44e410c936"
EMBEDDING_SIZE = 768
TEST_SIZE = 0.25
SPLIT_SEED = 42
AGE_BINS = ((18, 30), (30, 40), (40, 50), (50, 60), (60, 70), (70, 80), (80, 90))
MANIFEST_COLUMNS = (
    "embedding_row", "source_image_id", "relative_image_path", "split", "target_age", "target_gender",
)


def relative_image_path(image_id):
    return f"kaggle/kaggle/train/{int(image_id):06d}.png"


def sha256_file(file_path):
    digest = hashlib.sha256()
    with Path(file_path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_labeled_frame(age_csv_path, gender_csv_path):
    age = pd.read_csv(age_csv_path)
    gender = pd.read_csv(gender_csv_path)
    merged = age.merge(gender, on="imageId", how="inner")
    merged["source_image_id"] = pd.to_numeric(merged["imageId"], errors="raise").astype(int)
    merged["target_age"] = pd.to_numeric(merged["age"], errors="raise").astype(float)
    merged["target_gender"] = pd.to_numeric(merged["gender"], errors="raise").astype(int)
    merged = merged.sort_values("source_image_id").reset_index(drop=True)
    merged["embedding_row"] = np.arange(len(merged), dtype=np.int64)
    merged["relative_image_path"] = merged["source_image_id"].map(relative_image_path)
    return merged.loc[:, ["source_image_id", "target_age", "target_gender", "relative_image_path", "embedding_row"]]


def age_bin(age):
    for low, high in AGE_BINS:
        if low <= age < high:
            return f"[{low},{high})"


def build_manifest(age_csv_path, gender_csv_path):
    frame = load_labeled_frame(age_csv_path, gender_csv_path)
    strata = frame["target_gender"].astype(str) + ":" + frame["target_age"].map(age_bin)
    train_index, test_index = train_test_split(
        frame.index.to_numpy(),
        test_size=TEST_SIZE,
        random_state=SPLIT_SEED,
        shuffle=True,
        stratify=strata.to_numpy(),
    )
    frame["split"] = np.where(frame.index.isin(train_index), "train", "test")
    return frame.reindex(columns=list(MANIFEST_COLUMNS))


def split_counts(manifest):
    return manifest["split"].value_counts().to_dict()


def extract_embeddings(processor, model, image_paths, batch_size, device):
    image_paths = [Path(p) for p in image_paths]
    embeddings = np.empty((len(image_paths), EMBEDDING_SIZE), dtype=np.float32)
    with torch.inference_mode():
        for start in tqdm(range(0, len(image_paths), batch_size)):
            batch_paths = image_paths[start:start + batch_size]
            images = []
            for path in batch_paths:
                with Image.open(path) as img:
                    img.load()
                    images.append(img)
            inputs = processor(images=images, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items() if isinstance(value, torch.Tensor)}
            outputs = model(**inputs)
            cls_vectors = outputs.last_hidden_state[:, 0, :]
            embeddings[start:start + batch_size] = cls_vectors.float().cpu().numpy()
    return embeddings


def _plain_size(value):
    if value is None or isinstance(value, (int, str)):
        return value
    if isinstance(value, dict):
        return {key: item for key, item in value.items()}
    return dict(value)


def processor_contract(processor):
    config = getattr(processor, "image_processor", None)
    if config is None:
        config = getattr(processor, "config", None)
    if config is None:
        config = processor
    return {
        "class": type(processor).__name__,
        "do_convert_rgb": getattr(config, "do_convert_rgb", None),
        "size": _plain_size(getattr(config, "size", None)),
        "resample": getattr(config, "resample", None),
        "crop_size": _plain_size(getattr(config, "crop_size", None)),
        "do_center_crop": getattr(config, "do_center_crop", None),
        "do_rescale": getattr(config, "do_rescale", None),
        "rescale_factor": getattr(config, "rescale_factor", None),
        "image_mean": list(getattr(config, "image_mean", None) or []),
        "image_std": list(getattr(config, "image_std", None) or []),
    }


def require_cuda(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"device {device!r} requires CUDA/HIP but torch.cuda.is_available() is False"
        )


def package_versions():
    import importlib.metadata
    versions = {}
    for name in ("torch", "torchvision", "transformers", "huggingface_hub", "numpy", "pandas", "scikit-learn"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def serialize_embeddings(embeddings):
    buffer = io.BytesIO()
    np.save(buffer, embeddings)
    return buffer.getvalue()


def serialize_manifest(manifest):
    return manifest.to_csv(index=False).encode("utf-8")


def serialize_metadata(metadata):
    return json.dumps(metadata, indent=2, sort_keys=True).encode("utf-8")


def _atomic_write_bytes(target_path, payload):
    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, tmp_path = tempfile.mkstemp(dir=str(target.parent), prefix=".tmp-", suffix=target.suffix)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
        os.replace(tmp_path, target)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def build_metadata(manifest, embeddings, batch_size, processor, source_checksums, started_at, elapsed_seconds):
    return {
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
        "representation": "last_hidden_state[:, 0, :]",
        "embedding": {"shape": list(embeddings.shape), "dtype": str(embeddings.dtype)},
        "processor": processor_contract(processor),
        "split": {
            "test_size": TEST_SIZE,
            "seed": SPLIT_SEED,
            "shuffle": True,
            "strata": "joint: target_gender + fixed target_age bins",
            "age_bins": [[low, high] for low, high in AGE_BINS],
            "counts": split_counts(manifest),
        },
        "batch_size": batch_size,
        "package_versions": package_versions(),
        "checksums": {"sources": source_checksums, "artifacts": {}},
        "timing": {"started_at": started_at, "elapsed_seconds": elapsed_seconds},
    }


def write_artifacts(embeddings, manifest, metadata, embeddings_path=EMBEDDINGS_PATH, manifest_path=MANIFEST_PATH, metadata_path=METADATA_PATH):
    _atomic_write_bytes(embeddings_path, serialize_embeddings(embeddings))
    _atomic_write_bytes(manifest_path, serialize_manifest(manifest))
    final_metadata = dict(metadata)
    final_metadata["checksums"] = dict(metadata.get("checksums", {}))
    final_metadata["checksums"]["artifacts"] = {
        "embeddings.npy": sha256_file(embeddings_path),
        "manifest.csv": sha256_file(manifest_path),
    }
    _atomic_write_bytes(metadata_path, serialize_metadata(final_metadata))
    return final_metadata


def load_processor_and_model(device):
    from transformers import AutoImageProcessor, AutoModel
    processor = AutoImageProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = AutoModel.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = model.to(device)
    model.eval()
    return processor, model


def build_parser():
    parser = argparse.ArgumentParser(description="Extract frozen RAD-DINO CLS embeddings for SPR X-ray")
    parser.add_argument("--batch-size", type=int, default=16, help="fixed batch size for the extraction run")
    parser.add_argument("--device", default="cuda:0", help="explicit torch device (default cuda:0)")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    require_cuda(args.device)
    device = args.device
    processor, model = load_processor_and_model(device)
    manifest = build_manifest(AGE_CSV_PATH, GENDER_CSV_PATH)
    image_paths = [DATA_DIR / rel for rel in manifest["relative_image_path"]]
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    started_epoch = time.time()
    embeddings = extract_embeddings(processor, model, image_paths, args.batch_size, device)
    elapsed_seconds = time.time() - started_epoch
    source_checksums = {
        "train_age.csv": sha256_file(AGE_CSV_PATH),
        "train_gender.csv": sha256_file(GENDER_CSV_PATH),
    }
    metadata = build_metadata(
        manifest, embeddings, args.batch_size, processor, source_checksums, started_at, elapsed_seconds
    )
    final_metadata = write_artifacts(embeddings, manifest, metadata)
    print(
        f"wrote {EMBEDDINGS_PATH.name}, {MANIFEST_PATH.name}, {METADATA_PATH.name} "
        f"({len(manifest)} rows, batch {args.batch_size}, {elapsed_seconds:.1f}s); "
        f"split {split_counts(manifest)}"
    )


if __name__ == "__main__":
    main()