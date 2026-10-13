import numpy as np
import pandas as pd


N_FEATURES = 768


def load_spr_xray(embeddings_path, manifest_path):
    manifest = pd.read_csv(manifest_path)
    embeddings = np.load(embeddings_path, mmap_mode="r")

    rows = manifest["embedding_row"].to_numpy()
    feature_array = np.asarray(embeddings[rows], dtype="float32")

    frame = pd.DataFrame(
        feature_array,
        columns=[f"emb_{i}" for i in range(feature_array.shape[1])],
    )

    frame["target_age"] = manifest["target_age"].to_numpy(dtype=float)
    frame["target_binary_gender"] = manifest["target_gender"].to_numpy(dtype=int)

    return frame
