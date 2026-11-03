import copy
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---------------- Configuration ----------------
SEED = 42
N_SPLITS = 5
BATCH_SIZE = 64
MAX_EPOCHS = 100
PATIENCE = 10
LEARNING_RATE = 1e-3
MIN_DELTA = 1e-4

BASE = Path.home() / "tawos-sbert"
EMBED_DIR = BASE / "data" / "processed" / "embeddings"
OUTPUT_DIR = BASE / "outputs" / "cross_validation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PROJECT_IDS = [1, 13, 18, 21, 29, 30, 34]
PROJECT_TO_IDX = {pid: i for i, pid in enumerate(PROJECT_IDS)}

# ---------------- Reproducibility ----------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

device = torch.device("cpu")
print(f"Device: {device}")

# ---------------- Load development embeddings ----------------
def load_npz(path):
    # These local embedding files contain object arrays for string metadata.
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key].copy() for key in data.files}

train_data = load_npz(EMBED_DIR / "train_embeddings.npz")
val_data = load_npz(EMBED_DIR / "validation_embeddings.npz")

# Combine only train and validation. Do NOT load test embeddings.
X = np.concatenate([train_data["X"], val_data["X"]]).astype(np.float32)
y = np.concatenate([train_data["y"], val_data["y"]]).astype(np.float32)
project_ids = np.concatenate([
    train_data["project_id"], val_data["project_id"]
]).astype(np.int64)
groups = np.concatenate([
    train_data["split_group"], val_data["split_group"]
]).astype(str)

project_idx = np.array(
    [PROJECT_TO_IDX[int(pid)] for pid in project_ids],
    dtype=np.int64
)

if not (
    len(X) == len(y) == len(project_idx) == len(groups)
):
    raise ValueError("Embedding arrays have inconsistent lengths.")

if len(np.unique(groups)) < N_SPLITS:
    raise ValueError("Not enough unique text groups for five-fold CV.")

print(f"Development examples: {len(y):,}")
print(f"Embedding dimension: {X.shape[1]}")
print(f"Unique text groups: {len(np.unique(groups)):,}")
print(f"Story-point range: {y.min():g} to {y.max():g}")
print(f"Project IDs: {sorted(np.unique(project_ids).tolist())}")

# ---------------- Model ----------------
class UnifiedStoryPointCNN(nn.Module):
    def __init__(self, embedding_dim=384, num_projects=7):
        super().__init__()
        self.text_cnn = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
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
        combined = torch.cat(
            [text_features, project_features], dim=1
        )
        return self.regressor(combined).squeeze(1)

def make_loader(indices, shuffle, seed):
    dataset = TensorDataset(
        torch.from_numpy(X[indices]),
        torch.from_numpy(project_idx[indices]),
        torch.from_numpy(y[indices]),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        generator=generator,
    )

def evaluate(model, indices):
    model.eval()
    predictions = []
    actual = []
    loader = make_loader(indices, shuffle=False, seed=SEED)

    with torch.no_grad():
        for xb, pb, yb in loader:
            xb, pb = xb.to(device), pb.to(device)
            pred = model(xb, pb)
            predictions.extend(pred.cpu().numpy())
            actual.extend(yb.numpy())

    actual = np.asarray(actual)
    predictions = np.asarray(predictions)
    mae = mean_absolute_error(actual, predictions)
    rmse = np.sqrt(mean_squared_error(actual, predictions))
    return float(mae), float(rmse)

# ---------------- Five-fold grouped CV ----------------
cv = GroupKFold(n_splits=N_SPLITS)
fold_results = []
all_histories = []

for fold, (train_idx, val_idx) in enumerate(
    cv.split(X, y, groups=groups), start=1
):
    fold_seed = SEED + fold
    set_seed(fold_seed)

    train_groups = set(groups[train_idx])
    val_groups = set(groups[val_idx])
    overlap = train_groups.intersection(val_groups)

    if overlap:
        raise RuntimeError(
            f"Group leakage detected in fold {fold}: {len(overlap)} groups"
        )

    model = UnifiedStoryPointCNN(
        embedding_dim=X.shape[1],
        num_projects=len(PROJECT_IDS),
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE
    )
    criterion = nn.MSELoss()
    train_loader = make_loader(
        train_idx, shuffle=True, seed=fold_seed
    )

    best_mae = float("inf")
    best_rmse = float("inf")
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0
    history = []

    print(
        f"\n{'=' * 60}\n"
        f"FOLD {fold}/{N_SPLITS} | "
        f"train={len(train_idx):,}, validation={len(val_idx):,}\n"
        f"{'=' * 60}"
    )

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        batch_losses = []

        for xb, pb, yb in train_loader:
            xb, pb, yb = xb.to(device), pb.to(device), yb.to(device)

            optimizer.zero_grad()
            predictions = model(xb, pb)
            loss = criterion(predictions, yb)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        val_mae, val_rmse = evaluate(model, val_idx)
        train_mse = float(np.mean(batch_losses))

        row = {
            "fold": fold,
            "epoch": epoch,
            "train_mse": train_mse,
            "validation_mae": val_mae,
            "validation_rmse": val_rmse,
        }
        history.append(row)
        all_histories.append(row)

        improved = val_mae < best_mae - MIN_DELTA

        if improved:
            best_mae = val_mae
            best_rmse = val_rmse
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch == 1 or improved or epoch % 5 == 0:
            print(
                f"Epoch {epoch:03d} | "
                f"train MSE={train_mse:.4f} | "
                f"val MAE={val_mae:.4f} | "
                f"val RMSE={val_rmse:.4f}"
                + (" | best" if improved else "")
            )

        if epochs_without_improvement >= PATIENCE:
            print(f"Early stopping at epoch {epoch}.")
            break

    fold_dir = OUTPUT_DIR / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    if best_state is None:
        raise RuntimeError(f"No best checkpoint recorded for fold {fold}.")

    torch.save(
        {
            "model_state_dict": best_state,
            "best_epoch": best_epoch,
            "best_validation_mae": best_mae,
            "best_validation_rmse": best_rmse,
            "project_ids": PROJECT_IDS,
            "embedding_dim": X.shape[1],
            "fold": fold,
            "seed": fold_seed,
        },
        fold_dir / "best_model.pt",
    )
    pd.DataFrame(history).to_csv(
        fold_dir / "training_history.csv", index=False
    )

    fold_result = {
        "fold": fold,
        "train_size": len(train_idx),
        "validation_size": len(val_idx),
        "best_epoch": best_epoch,
        "best_validation_mae": best_mae,
        "best_validation_rmse": best_rmse,
    }
    fold_results.append(fold_result)

    print(
        f"FOLD {fold} RESULT | best epoch={best_epoch} | "
        f"MAE={best_mae:.4f} | RMSE={best_rmse:.4f}"
    )

# ---------------- Save summary ----------------
results_df = pd.DataFrame(fold_results)
results_df.to_csv(OUTPUT_DIR / "cv_fold_results.csv", index=False)
pd.DataFrame(all_histories).to_csv(
    OUTPUT_DIR / "cv_all_training_histories.csv", index=False
)

summary = {
    "n_splits": N_SPLITS,
    "early_stopping_patience": PATIENCE,
    "development_examples": int(len(y)),
    "grouped_by": "split_group",
    "test_set_used": False,
    "fold_mae_mean": float(results_df["best_validation_mae"].mean()),
    "fold_mae_std": float(results_df["best_validation_mae"].std(ddof=1)),
    "fold_rmse_mean": float(results_df["best_validation_rmse"].mean()),
    "fold_rmse_std": float(results_df["best_validation_rmse"].std(ddof=1)),
    "best_epochs": results_df["best_epoch"].tolist(),
    "median_best_epoch": int(
        np.median(results_df["best_epoch"])
    ),
    "project_id_to_index": {
        str(pid): idx for pid, idx in PROJECT_TO_IDX.items()
    },
}

with open(OUTPUT_DIR / "cv_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 60)
print("FIVE-FOLD CROSS-VALIDATION COMPLETE")
print("=" * 60)
print(results_df.to_string(index=False))
print(
    f"\nValidation MAE:  "
    f"{summary['fold_mae_mean']:.4f} "
    f"+/- {summary['fold_mae_std']:.4f}"
)
print(
    f"Validation RMSE: "
    f"{summary['fold_rmse_mean']:.4f} "
    f"+/- {summary['fold_rmse_std']:.4f}"
)
print(f"Best epochs: {summary['best_epochs']}")
print(f"Median best epoch: {summary['median_best_epoch']}")
print(f"Results saved in: {OUTPUT_DIR}")
print("The held-out test set was not loaded or evaluated.")
