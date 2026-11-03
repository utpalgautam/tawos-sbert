import os
import json
import copy
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------
BASE_DIR = os.path.expanduser("~/tawos-sbert")
EMBED_DIR = f"{BASE_DIR}/data/processed/embeddings"
OUTPUT_DIR = f"{BASE_DIR}/outputs/unified_model"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = 42
BATCH_SIZE = 64
MAX_EPOCHS = 100
PATIENCE = 10
LEARNING_RATE = 1e-3

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# CPU is a safe default for an 8 GB MacBook Air.
DEVICE = torch.device("cpu")
print("Training device:", DEVICE)
print("PyTorch:", torch.__version__)

# Consistent project-to-index mapping
PROJECT_IDS = [1, 13, 18, 21, 29, 30, 34]
PROJECT_TO_IDX = {pid: i for i, pid in enumerate(PROJECT_IDS)}

# --------------------------------------------------
# 2. Load saved SBERT embeddings
# --------------------------------------------------
def load_split(name):
    path = f"{EMBED_DIR}/{name}_embeddings.npz"

    with np.load(path, allow_pickle=False) as data:
        X = data["X"].astype(np.float32)
        y = data["y"].astype(np.float32)
        project_ids = data["project_id"].astype(np.int64)

    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError(f"Non-finite values found in {name}")

    unknown = set(np.unique(project_ids)) - set(PROJECT_TO_IDX)
    if unknown:
        raise ValueError(f"Unexpected project IDs: {unknown}")

    project_idx = np.array(
        [PROJECT_TO_IDX[int(pid)] for pid in project_ids],
        dtype=np.int64
    )

    return X, project_idx, y, project_ids

X_train, p_train, y_train, ids_train = load_split("train")
X_val, p_val, y_val, ids_val = load_split("validation")

print("Training embeddings:", X_train.shape)
print("Validation embeddings:", X_val.shape)
print("Projects:", PROJECT_IDS)

# --------------------------------------------------
# 3. Create DataLoaders
# --------------------------------------------------
def make_loader(X, project_idx, y, shuffle):
    dataset = TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(project_idx),
        torch.from_numpy(y)
    )
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0
    )

train_loader = make_loader(X_train, p_train, y_train, True)
val_loader = make_loader(X_val, p_val, y_val, False)

# --------------------------------------------------
# 4. Unified SBERT + CNN model
# --------------------------------------------------
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
            nn.Flatten()
        )

        self.project_embedding = nn.Embedding(num_projects, 8)

        self.regressor = nn.Sequential(
            nn.Linear(128 + 8, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, X, project_idx):
        # X shape: batch x 384
        text_features = self.text_cnn(X.unsqueeze(1))
        project_features = self.project_embedding(project_idx)

        combined = torch.cat(
            [text_features, project_features], dim=1
        )
        return self.regressor(combined).squeeze(1)

model = UnifiedStoryPointCNN().to(DEVICE)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=LEARNING_RATE
)

# --------------------------------------------------
# 5. Training and validation
# --------------------------------------------------
def evaluate(loader):
    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        for X, project_idx, y in loader:
            X = X.to(DEVICE)
            project_idx = project_idx.to(DEVICE)

            pred = model(X, project_idx)

            predictions.extend(pred.cpu().numpy())
            actuals.extend(y.numpy())

    predictions = np.asarray(predictions)
    actuals = np.asarray(actuals)

    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    return mae, rmse

best_val_mae = float("inf")
best_epoch = 0
best_state = None
epochs_without_improvement = 0
history = []

for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    batch_losses = []

    for X, project_idx, y in train_loader:
        X = X.to(DEVICE)
        project_idx = project_idx.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()
        predictions = model(X, project_idx)
        loss = criterion(predictions, y)
        loss.backward()
        optimizer.step()

        batch_losses.append(loss.item())

    train_loss = float(np.mean(batch_losses))
    val_mae, val_rmse = evaluate(val_loader)

    history.append({
        "epoch": epoch,
        "train_mse": train_loss,
        "validation_mae": val_mae,
        "validation_rmse": val_rmse
    })

    print(
        f"Epoch {epoch:03d} | "
        f"Train MSE: {train_loss:.4f} | "
        f"Val MAE: {val_mae:.4f} | "
        f"Val RMSE: {val_rmse:.4f}"
    )

    if val_mae < best_val_mae:
        best_val_mae = val_mae
        best_epoch = epoch
        best_state = copy.deepcopy(model.state_dict())
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= PATIENCE:
        print(f"Early stopping at epoch {epoch}.")
        break

# --------------------------------------------------
# 6. Save best model and training history
# --------------------------------------------------
if best_state is None:
    raise RuntimeError("Training did not produce a model checkpoint.")

model.load_state_dict(best_state)

model_path = f"{OUTPUT_DIR}/unified_sbert_cnn.pt"
torch.save(model.state_dict(), model_path)

history_path = f"{OUTPUT_DIR}/training_history.csv"
pd.DataFrame(history).to_csv(history_path, index=False)

metadata = {
    "sbert_model": "sentence-transformers/all-MiniLM-L6-v2",
    "embedding_dim": 384,
    "project_ids": PROJECT_IDS,
    "project_to_idx": {
        str(k): v for k, v in PROJECT_TO_IDX.items()
    },
    "best_epoch": best_epoch,
    "best_validation_mae": best_val_mae,
    "training_rows": len(y_train),
    "validation_rows": len(y_val),
    "seed": SEED
}

with open(f"{OUTPUT_DIR}/model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("\nTraining complete.")
print("Best epoch:", best_epoch)
print("Best validation MAE:", round(best_val_mae, 4))
print("Model saved:", model_path)
print("History saved:", history_path)
print("Metadata saved:", f"{OUTPUT_DIR}/model_metadata.json")
print("The test set has not been loaded or evaluated.")
