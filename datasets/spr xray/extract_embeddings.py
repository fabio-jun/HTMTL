import argparse
import importlib.metadata
import importlib.util
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

MODEL_ID = "microsoft/rad-dino"
MODEL_REVISION = "110cbc18d5133582e320b43d53bf5c44e410c936"
EMBEDDING_SIZE = 768


def require_cuda(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"device {device!r} requires CUDA/HIP but torch.cuda.is_available() is False")


def _plain_size(value):
    if value is None or isinstance(value, (int, str)):
        return value
    return dict(value)


def processor_contract(processor):
    config = getattr(processor, "image_processor", None) or getattr(processor, "config", None) or processor
    return {"class": type(processor).__name__, "do_convert_rgb": getattr(config, "do_convert_rgb", None),
            "size": _plain_size(getattr(config, "size", None)), "resample": getattr(config, "resample", None),
            "crop_size": _plain_size(getattr(config, "crop_size", None)), "do_center_crop": getattr(config, "do_center_crop", None),
            "do_rescale": getattr(config, "do_rescale", None), "rescale_factor": getattr(config, "rescale_factor", None),
            "image_mean": list(getattr(config, "image_mean", None) or []), "image_std": list(getattr(config, "image_std", None) or [])}


def package_versions():
    versions = {}
    for name in ("torch", "torchvision", "transformers", "huggingface_hub", "numpy", "pandas"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def load_processor_and_model(device):
    from transformers import AutoImageProcessor, AutoModel
    processor = AutoImageProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = AutoModel.from_pretrained(MODEL_ID, revision=MODEL_REVISION).to(device)
    model.eval()
    return processor, model


def extract_embeddings(processor, model, image_paths, batch_size, device):
    if batch_size < 1:
        raise ValueError("batch size must be positive")
    image_paths = [Path(path) for path in image_paths]
    for path in image_paths:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"SPR selected training image must be a regular file: {path}")
    embeddings = np.empty((len(image_paths), EMBEDDING_SIZE), dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for start in tqdm(range(0, len(image_paths), batch_size)):
            batch_paths = image_paths[start:start + batch_size]
            images = []
            for path in batch_paths:
                with Image.open(path) as image:
                    image.load()
                    images.append(image.copy())
            inputs = processor(images=images, return_tensors="pt")
            inputs = {name: value.to(device) for name, value in inputs.items() if isinstance(value, torch.Tensor)}
            cls_vectors = model(**inputs).last_hidden_state[:, 0, :]
            embeddings[start:start + len(batch_paths)] = cls_vectors.float().cpu().numpy()
    return embeddings


def build_parser():
    parser = argparse.ArgumentParser(description="Extract the frozen RAD-DINO SPR selected cohort from Kaggle training images")
    parser.add_argument("--input", type=Path, required=True, help="Kaggle dataset root containing labels and kaggle/kaggle/train")
    parser.add_argument("--output", type=Path, required=True, help="new selected-cohort output directory")
    parser.add_argument("--selection", type=Path, required=True, help="official frozen cohort selection CSV")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    spec = importlib.util.spec_from_file_location("_spr_cohort_preprocess", Path(__file__).with_name("preprocess.py"))
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load SPR cohort preprocessor")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.preprocess_raw(args.input, args.output, args.selection, device=args.device, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
