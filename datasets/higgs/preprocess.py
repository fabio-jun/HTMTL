import argparse
import hashlib
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROLES = ("train", "val", "test")
ROLE_COUNTS = {"train": 45000, "val": 2500, "test": 2500}
SOURCE_SHA256 = "8322f9ba7f406f73632e40a70db95e2fcc8dbf15ca3f037769b5c34ae339fe8d"
INDICES_SHA256 = "b75e20e57e5cf07db3ddbf3a8fa51c51cdfa7eb3243d19b80aebaa58fd2ab4fc"
OUTPUT_SHA256 = "bc149ac7c5dcaf4e3d2d0b61bd4cc5f393412e1bd2b595aec249613dc4439a6a"
FEATURE_COLUMNS = ("cat_0", "cat_1", "cat_2", "cat_3", *[f"num_{index}" for index in range(17)])
TASKS = ("Target", "m_bb", "m_jj", "m_jjj", "m_jlv", "m_lv", "m_wbb", "m_wwbb")
TARGET_COLUMNS = ("target_binary_Target", *[f"target_regression_{task}" for task in TASKS[1:]])


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def preprocess(input_path, output_path, selection_path, chunksize=100000,
               source_sha256=SOURCE_SHA256, indices_sha256=INDICES_SHA256,
               output_sha256=OUTPUT_SHA256, role_counts=ROLE_COUNTS):
    input_path, output_path, selection_path = map(Path, (input_path, output_path, selection_path))

    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"Refusing existing output: {output_path}")
    
    if any(path.is_symlink() or not path.is_file() for path in (input_path, selection_path)):
        raise ValueError("Install the original MultiTab HDF5 and official cohort indices as regular files")
    
    if digest(input_path) != source_sha256 or digest(selection_path) != indices_sha256:
        raise ValueError("HIGGS source or cohort indices differ from the official inputs")
    
    with np.load(selection_path, allow_pickle=False) as selection, h5py.File(input_path, "r") as source:
        if set(selection.files) != set(ROLES):
            raise ValueError("HIGGS cohort indices must contain train, val and test")
        frames = []

        for role in ROLES:
            indices = selection[role]

            if indices.dtype.kind not in "iu" or len(indices) != role_counts[role] or not np.all(indices[1:] > indices[:-1]):
                raise ValueError(f"Invalid official HIGGS indices for {role}")
            group = source[role]

            if np.any(indices < 0) or np.any(indices >= len(group["Target"])):
                raise ValueError(f"HIGGS index outside source role {role}")
            pieces = []

            for start in range(0, len(group["Target"]), chunksize):
                selected = indices[(indices >= start) & (indices < start + chunksize)] - start
                if not len(selected):
                    continue
                cats = group["features_categorical"][start:start + chunksize][selected]
                nums = group["features_numerical"][start:start + chunksize][selected]
                frame = pd.DataFrame(cats, columns=FEATURE_COLUMNS[:4])

                for column, values in zip(FEATURE_COLUMNS[4:], nums.T, strict=True):
                    frame[column] = values

                for task, target in zip(TASKS, TARGET_COLUMNS, strict=True):
                    frame[target] = group[task][start:start + chunksize][selected]
                frame["source_split_role"] = role

                pieces.append(frame)
            frames.append(pd.concat(pieces, ignore_index=True))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".partial")

    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"Refusing existing partial output: {temporary}")
    handle = temporary.open("x", newline="")

    try:
        with handle:
            pd.concat(frames, ignore_index=True).to_csv(handle, index=False)
        if digest(temporary) != output_sha256:
            raise ValueError("Reconstructed HIGGS CSV does not match the official 50k artifact")
        output_path.hardlink_to(temporary)
    finally:
        temporary.unlink(missing_ok=True)

    print(f"HIGGS: 50k cohort written to {output_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Reconstruct the exact official HIGGS 50k cohort from the original MultiTab HDF5 and saved indices; no new random sample or transforms.")
    parser.add_argument("--input", type=Path, default=Path(__file__).parent / "train_val_test.h5")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "higgs_50k.csv")
    parser.add_argument("--selection", type=Path, default=Path(__file__).parent / "higgs_50k.indices.npz")
    args = parser.parse_args()
    preprocess(args.input, args.output, args.selection)


if __name__ == "__main__":
    main()
