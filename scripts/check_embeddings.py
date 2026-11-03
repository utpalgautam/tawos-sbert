import os
import numpy as np

BASE_DIR = os.path.expanduser("~/tawos-sbert")
EMBED_DIR = f"{BASE_DIR}/data/processed/embeddings"

for split in ["train", "validation", "test"]:
    path = f"{EMBED_DIR}/{split}_embeddings.npz"

    with np.load(path, allow_pickle=False) as data:
        X = data["X"]
        y = data["y"]
        project_ids = data["project_id"]

        print(f"\n{split.upper()}")
        print("Embeddings:", X.shape)
        print("Labels:", y.shape)
        print("Projects:", np.unique(project_ids))
        print("NaN values:", np.isnan(X).sum())

        assert X.ndim == 2 and X.shape[1] == 384
        assert len(X) == len(y) == len(project_ids)
        assert np.isfinite(X).all()
        assert np.isfinite(y).all()
        assert set(np.unique(project_ids)).issubset(
            {1, 13, 18, 21, 29, 30, 34}
        )

print("\nAll embedding checks passed.")
