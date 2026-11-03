import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    mean_absolute_error,
    median_absolute_error,
    mean_squared_error,
)

BASE = Path.home() / "tawos-sbert"
EMBED_DIR = BASE / "data/processed/embeddings"
CV_DIR = BASE / "outputs/cross_validation"
PROJECT_IDS = [1, 13, 18, 21, 29, 30, 34]
PROJECT_TO_IDX = {pid: i for i, pid in enumerate(PROJECT_IDS)}

def load_npz(path):
    with np.load(path, allow_pickle=True) as data:
        return {k: data[k].copy() for k in data.files}

train = load_npz(EMBED_DIR / "train_embeddings.npz")
val = load_npz(EMBED_DIR / "validation_embeddings.npz")

X = np.concatenate([train["X"], val["X"]]).astype(np.float32)
y = np.concatenate([train["y"], val["y"]]).astype(np.float32)
pids = np.concatenate([
    train["project_id"], val["project_id"]
]).astype(np.int64)
groups = np.concatenate([
    train["split_group"], val["split_group"]
]).astype(str)
pidx = np.array([PROJECT_TO_IDX[int(p)] for p in pids])

class UnifiedStoryPointCNN(nn.Module):
    def __init__(self, embedding_dim=384, num_projects=7):
        super().__init__()
        self.text_cnn = nn.Sequential(
            nn.Conv1d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
        )
        self.project_embedding = nn.Embedding(num_projects, 8)
        self.regressor = nn.Sequential(
            nn.Linear(136, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, embeddings, project_indices):
        text_features = self.text_cnn(embeddings.unsqueeze(1))
        project_features = self.project_embedding(project_indices)
        return self.regressor(
            torch.cat([text_features, project_features], dim=1)
        ).squeeze(1)

results = []
cv = GroupKFold(n_splits=5)

for fold, (_, val_idx) in enumerate(
    cv.split(X, y, groups=groups), start=1
):
    checkpoint_path = CV_DIR / f"fold_{fold}" / "best_model.pt"
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )

    model = UnifiedStoryPointCNN()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        xb = torch.from_numpy(X[val_idx])
        pb = torch.from_numpy(pidx[val_idx])
        predictions = model(xb, pb).numpy()

    actual = y[val_idx]
    results.append({
        "fold": fold,
        "best_epoch": checkpoint["best_epoch"],
        "MAE": mean_absolute_error(actual, predictions),
        "MdAE": median_absolute_error(actual, predictions),
        "RMSE": np.sqrt(mean_squared_error(actual, predictions)),
    })

df = pd.DataFrame(results)
df.to_csv(CV_DIR / "cv_metrics_with_mdae.csv", index=False)

print("\nFIVE-FOLD CROSS-VALIDATION METRICS")
print(df.to_string(index=False))

print("\nMEAN AND STANDARD DEVIATION ACROSS FOLDS")
for metric in ["MAE", "MdAE", "RMSE"]:
    print(
        f"{metric}: {df[metric].mean():.4f} "
        f"+/- {df[metric].std(ddof=1):.4f}"
    )

print(f"\nSaved: {CV_DIR / 'cv_metrics_with_mdae.csv'}")
print("Held-out test set was not used.")
