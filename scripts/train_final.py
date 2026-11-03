import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --------------------------------------------------
# Configuration
# --------------------------------------------------
SEED = 42
EPOCHS = 4
BATCH_SIZE = 64
LEARNING_RATE = 1e-3

PROJECTS = {
    1: "XD",
    13: "TISTUD",
    18: "JRASERVER",
    21: "CONFCLOUD",
    29: "DURACLOUD",
    30: "COMPASS",
    34: "MDL",
}

PROJECT_IDS = sorted(PROJECTS)
PROJECT_TO_INDEX = {pid: i for i, pid in enumerate(PROJECT_IDS)}

ROOT = Path(".")
EMBED_DIR = ROOT / "data" / "processed" / "embeddings"
OUT_DIR = ROOT / "outputs" / "final_model"
OUT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cpu")
print(f"Device: {device}")
print(f"Final training epochs: {EPOCHS}")


# --------------------------------------------------
# Load embeddings
# --------------------------------------------------
def load_embeddings(filename):
    path = EMBED_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing embeddings file: {path}")

    with np.load(path, allow_pickle=True) as data:
        X = data["X"].astype(np.float32)
        y = data["y"].astype(np.float32).reshape(-1)
        raw_project_ids = data["project_id"].astype(int)
        issue_keys = data["issue_key"].astype(str)

    project_indices = np.array(
        [PROJECT_TO_INDEX[int(pid)] for pid in raw_project_ids],
        dtype=np.int64,
    )

    return X, y, project_indices, raw_project_ids, issue_keys


# Use training + validation as the complete development set.
X_train, y_train, p_train, _, _ = load_embeddings("train_embeddings.npz")
X_val, y_val, p_val, _, _ = load_embeddings("validation_embeddings.npz")

X_dev = np.concatenate([X_train, X_val])
y_dev = np.concatenate([y_train, y_val])
p_dev = np.concatenate([p_train, p_val])

print(f"Development samples: {len(y_dev)}")
print(f"Embedding dimensions: {X_dev.shape[1]}")


# --------------------------------------------------
# Model architecture: same unified SBERT + CNN
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
            nn.Flatten(),
        )

        self.project_embedding = nn.Embedding(num_projects, 8)

        self.regressor = nn.Sequential(
            nn.Linear(136, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x, project_idx):
        text_features = self.text_cnn(x.unsqueeze(1))
        project_features = self.project_embedding(project_idx)
        combined = torch.cat([text_features, project_features], dim=1)
        return self.regressor(combined).squeeze(1)


model = UnifiedStoryPointCNN(
    embedding_dim=X_dev.shape[1],
    num_projects=len(PROJECT_IDS),
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()

train_dataset = TensorDataset(
    torch.from_numpy(X_dev),
    torch.from_numpy(p_dev),
    torch.from_numpy(y_dev),
)

generator = torch.Generator().manual_seed(SEED)
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    generator=generator,
)

# --------------------------------------------------
# Train final model for the CV-selected epoch count
# --------------------------------------------------
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for X_batch, p_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        p_batch = p_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        predictions = model(X_batch, p_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y_batch)
        total_samples += len(y_batch)

    average_loss = total_loss / total_samples
    print(
        f"Epoch {epoch + 1}/{EPOCHS} "
        f"- training MSE: {average_loss:.4f}"
    )

# Save the finalized model before opening the test data.
model_path = OUT_DIR / "final_sbert_cnn.pt"
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "project_ids": PROJECT_IDS,
        "project_to_index": PROJECT_TO_INDEX,
        "epochs": EPOCHS,
        "seed": SEED,
        "embedding_dim": X_dev.shape[1],
    },
    model_path,
)
print(f"\nSaved final model: {model_path}")


# --------------------------------------------------
# Final evaluation: load the held-out test set now
# --------------------------------------------------
X_test, y_test, p_test, raw_p_test, issue_keys = load_embeddings(
    "test_embeddings.npz"
)

model.eval()
with torch.no_grad():
    predictions = model(
        torch.from_numpy(X_test).to(device),
        torch.from_numpy(p_test).to(device),
    ).cpu().numpy()

mae = mean_absolute_error(y_test, predictions)
mdae = float(np.median(np.abs(y_test - predictions)))
rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))

print("\nFINAL HELD-OUT TEST RESULTS")
print(f"Test samples: {len(y_test)}")
print(f"MAE:  {mae:.4f}")
print(f"MdAE: {mdae:.4f}")
print(f"RMSE: {rmse:.4f}")

metrics = {
    "test_samples": int(len(y_test)),
    "MAE": float(mae),
    "MdAE": mdae,
    "RMSE": rmse,
    "epochs": EPOCHS,
    "seed": SEED,
}

with open(OUT_DIR / "test_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# Per-project results
project_rows = []
for project_id in PROJECT_IDS:
    mask = raw_p_test == project_id
    if mask.sum() == 0:
        continue

    actual = y_test[mask]
    predicted = predictions[mask]

    project_rows.append({
        "project_id": project_id,
        "project_name": PROJECTS[project_id],
        "n_test": int(mask.sum()),
        "MAE": float(mean_absolute_error(actual, predicted)),
        "MdAE": float(np.median(np.abs(actual - predicted))),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
    })

per_project = pd.DataFrame(project_rows)
per_project.to_csv(OUT_DIR / "test_metrics_by_project.csv", index=False)

print("\nTEST RESULTS BY PROJECT")
print(per_project.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# Save individual predictions for analysis and plots.
prediction_df = pd.DataFrame({
    "issue_key": issue_keys,
    "project_id": raw_p_test,
    "project_name": [PROJECTS[int(pid)] for pid in raw_p_test],
    "actual_story_points": y_test,
    "predicted_story_points": predictions,
    "absolute_error": np.abs(y_test - predictions),
})
prediction_df.to_csv(OUT_DIR / "test_predictions.csv", index=False)

print(f"\nSaved final outputs to: {OUT_DIR}")
