import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset



# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "outputs" / "simran"

PROJECTS = [
    "BAM",
    "CLI",
    "CLOV",
    "EVG",
    "INDY",
    "JSWSERVER",
    "NEXUS",
    "TIDOC",
]

SEED = 42
EMBEDDING_DIM = 384

DEFAULT_EPOCHS = 100
DEFAULT_PATIENCE = 10
DEFAULT_BATCH_SIZE = 64
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_WEIGHT_DECAY = 0.0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


class StoryPointCNN(nn.Module):
    """Three-layer Conv1D regression head for frozen 384-D SBERT embeddings.

    Architecture follows COMMON_PIPELINE.md:
    1 -> 128 -> 64 -> 32 convolution channels, kernel 3, padding 1,
    ReLU + max-pooling after the first two convolutions, adaptive average
    pooling, dropout 0.2, and a single linear regression output.
    """

    def __init__(self, embedding_dim=384, dropout=0.2):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.features = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(output_size=1),
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(32, 1),
        )

    def forward(self, embeddings):
        if embeddings.ndim != 2 or embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Expected embeddings with shape (batch, {self.embedding_dim}); "
                f"received {tuple(embeddings.shape)}."
            )
        x = embeddings.unsqueeze(1)
        x = self.features(x)
        return self.regressor(x).squeeze(-1)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed: int) -> None:
    """Set random seeds for reproducible experiments."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, y_pred):
    """
    Calculate MAE and median absolute error.

    Negative story-point predictions are clipped to zero for
    evaluation because story points cannot be negative.
    """

    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have equal lengths.")

    if len(y_true) == 0:
        raise ValueError("Cannot calculate metrics on an empty dataset.")

    if not np.isfinite(y_true).all():
        raise ValueError("Ground-truth labels contain non-finite values.")

    if not np.isfinite(y_pred).all():
        raise ValueError("Predictions contain non-finite values.")

    y_pred = np.maximum(y_pred, 0.0)
    absolute_errors = np.abs(y_true - y_pred)

    return {
        "mae": float(np.mean(absolute_errors)),
        "mdae": float(np.median(absolute_errors)),
    }


# ============================================================
# LOAD AND VALIDATE PROJECT DATA
# ============================================================

def load_project_data(project):
    """
    Load embeddings, embedding index, labels, and split manifest.

    All tables are joined using issue_key and project_id.
    Embeddings are retrieved using their explicit row_index.
    """

    project_dir = DATA_DIR / project
    output_project_dir = OUTPUT_DIR / project

    labeled_path = project_dir / "labeled_clean.csv"
    embeddings_path = project_dir / "emb_labeled.npy"
    index_path = project_dir / "emb_labeled_index.csv"
    manifest_path = output_project_dir / "split_manifest.csv"

    required_files = [
        labeled_path,
        embeddings_path,
        index_path,
        manifest_path,
    ]

    missing_files = [
        str(path) for path in required_files if not path.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            f"{project}: missing required files:\n"
            + "\n".join(missing_files)
        )

    labels_df = pd.read_csv(labeled_path)
    index_df = pd.read_csv(index_path)
    manifest_df = pd.read_csv(manifest_path)
    embeddings = np.load(embeddings_path, allow_pickle=False)

    # Required columns
    required_label_columns = {
        "issue_key",
        "project_id",
        "story_point",
    }

    required_index_columns = {
        "issue_key",
        "project_id",
        "row_index",
    }

    required_manifest_columns = {
        "issue_key",
        "project_id",
        "split",
        "fold",
        "duplicate_group",
    }

    if not required_label_columns.issubset(labels_df.columns):
        raise ValueError(
            f"{project}: labeled_clean.csv is missing columns: "
            f"{required_label_columns - set(labels_df.columns)}"
        )

    if not required_index_columns.issubset(index_df.columns):
        raise ValueError(
            f"{project}: emb_labeled_index.csv is missing columns: "
            f"{required_index_columns - set(index_df.columns)}"
        )

    if not required_manifest_columns.issubset(manifest_df.columns):
        raise ValueError(
            f"{project}: split_manifest.csv is missing columns: "
            f"{required_manifest_columns - set(manifest_df.columns)}"
        )

    # Validate embedding dimensions.
    if embeddings.ndim != 2:
        raise ValueError(
            f"{project}: expected 2D embeddings, got {embeddings.shape}."
        )

    if embeddings.shape[1] != EMBEDDING_DIM:
        raise ValueError(
            f"{project}: expected {EMBEDDING_DIM}-dimensional embeddings, "
            f"got {embeddings.shape[1]}."
        )

    if len(index_df) != len(embeddings):
        raise ValueError(
            f"{project}: embedding count ({len(embeddings)}) does not "
            f"match index count ({len(index_df)})."
        )

    if not np.isfinite(embeddings).all():
        raise ValueError(f"{project}: embeddings contain NaN or infinity.")

    # row_index must point to the original embedding order.
    index_df["row_index"] = pd.to_numeric(
        index_df["row_index"], errors="raise"
    ).astype(int)

    expected_indices = np.arange(len(embeddings))

    if not np.array_equal(
        np.sort(index_df["row_index"].to_numpy()),
        expected_indices,
    ):
        raise ValueError(
            f"{project}: row_index must contain every embedding index "
            "exactly once."
        )

    # Prevent accidental duplicate-key joins.
    join_keys = ["issue_key", "project_id"]

    for name, frame in [
        ("labels", labels_df),
        ("embedding index", index_df),
        ("split manifest", manifest_df),
    ]:
        if frame[join_keys].isna().any().any():
            raise ValueError(f"{project}: {name} contains missing join keys.")

        if frame.duplicated(join_keys).any():
            raise ValueError(
                f"{project}: {name} contains duplicate issue/project keys."
            )

    # Keep only issues with actual labels.
    labels_df = labels_df.loc[
        labels_df["story_point"].notna()
    ].copy()

    labels_df["story_point"] = pd.to_numeric(
        labels_df["story_point"], errors="raise"
    )

    if not np.isfinite(labels_df["story_point"].to_numpy()).all():
        raise ValueError(f"{project}: story_point contains invalid values.")

    if (labels_df["story_point"] < 0).any():
        raise ValueError(
            f"{project}: negative ground-truth story points were found."
        )

    # Join labels to embedding indices.
    merged = labels_df.merge(
        index_df,
        on=join_keys,
        how="outer",
        validate="one_to_one",
        indicator="_embedding_match",
    )

    unmatched = merged["_embedding_match"] != "both"

    if unmatched.any():
        examples = merged.loc[
            unmatched, join_keys + ["_embedding_match"]
        ].head(10)

        raise ValueError(
            f"{project}: labels and embedding index do not match.\n"
            f"{examples.to_string(index=False)}"
        )

    merged = merged.drop(columns="_embedding_match")

    # Join the split manifest.
    merged = merged.merge(
        manifest_df,
        on=join_keys,
        how="outer",
        validate="one_to_one",
        indicator="_split_match",
        suffixes=("", "_manifest"),
    )

    unmatched_splits = merged["_split_match"] != "both"

    if unmatched_splits.any():
        examples = merged.loc[
            unmatched_splits,
            join_keys + ["_split_match"],
        ].head(10)

        raise ValueError(
            f"{project}: labeled issues and split manifest do not match.\n"
            f"{examples.to_string(index=False)}"
        )

    merged = merged.drop(columns="_split_match")

    # Every issue must be assigned to development or test.
    valid_splits = {"development", "test"}

    if not set(merged["split"].dropna().unique()).issubset(valid_splits):
        raise ValueError(
            f"{project}: unexpected split labels: "
            f"{merged['split'].unique().tolist()}"
        )

    if merged["split"].isna().any():
        raise ValueError(f"{project}: some issues have no split assignment.")

    development = merged["split"] == "development"
    test = merged["split"] == "test"

    if development.sum() == 0 or test.sum() == 0:
        raise ValueError(
            f"{project}: development or test set is empty."
        )

    # Every development issue must have a fold from 1 through 5.
    dev_folds = pd.to_numeric(
        merged.loc[development, "fold"], errors="coerce"
    )

    if dev_folds.isna().any():
        raise ValueError(
            f"{project}: development issues have missing fold assignments."
        )

    if not set(dev_folds.astype(int).unique()).issubset(
        {1, 2, 3, 4, 5}
    ):
        raise ValueError(
            f"{project}: development folds must be numbered 1 through 5."
        )

    merged["fold"] = pd.to_numeric(
        merged["fold"], errors="coerce"
    )

    # Ensure duplicate groups do not leak across split/fold boundaries.
    duplicate_rows = merged.loc[
        merged["duplicate_group"].notna()
    ]

    for group, group_df in duplicate_rows.groupby("duplicate_group"):
        assignments = group_df[["split", "fold"]].drop_duplicates()

        if len(assignments) != 1:
            raise ValueError(
                f"{project}: duplicate group {group!r} crosses "
                "split/fold boundaries."
            )

    # Restore the correct embedding row for each issue.
    row_indices = merged["row_index"].to_numpy(dtype=int)
    X = embeddings[row_indices].astype(np.float32, copy=False)

    y = merged["story_point"].to_numpy(dtype=np.float32)

    if len(X) != len(y):
        raise ValueError(f"{project}: embedding/label count mismatch.")

    print(f"\nProject: {project}")
    print(f"  Total labeled issues: {len(merged)}")
    print(f"  Development issues:   {int(development.sum())}")
    print(f"  Test issues:          {int(test.sum())}")
    print(f"  Embedding shape:      {X.shape}")
    print(
        "  Development fold sizes:",
        merged.loc[development, "fold"]
        .astype(int)
        .value_counts()
        .sort_index()
        .to_dict(),
    )

    return merged.reset_index(drop=True), X, y


# ============================================================
# MODEL TRAINING
# ============================================================

def train_one_model(
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    epochs=DEFAULT_EPOCHS,
    patience=DEFAULT_PATIENCE,
    batch_size=DEFAULT_BATCH_SIZE,
    learning_rate=DEFAULT_LEARNING_RATE,
    weight_decay=DEFAULT_WEIGHT_DECAY,
    seed=SEED,
):
    """Train a fresh CNN using original story-point labels.

    When validation data is supplied, keep the checkpoint with the lowest
    validation MAE and stop after `patience` consecutive non-improving epochs.
    Without validation data, train for the preselected CV-derived epoch count;
    the locked test set is never used for early stopping.
    """
    set_seed(seed)

    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32).reshape(-1)
    if len(X_train) != len(y_train) or len(y_train) == 0:
        raise ValueError("Training embeddings and labels must be nonempty and aligned.")
    if not np.isfinite(X_train).all() or not np.isfinite(y_train).all():
        raise ValueError("Training data contains NaN or infinite values.")

    train_dataset = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        drop_last=False,
    )

    model = StoryPointCNN(embedding_dim=EMBEDDING_DIM, dropout=0.2).to(DEVICE)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    has_validation = X_val is not None and y_val is not None
    best_state = None
    best_val_mae = float("inf")
    best_epoch = epochs
    epochs_without_improvement = 0
    training_start = time.time()

    if has_validation:
        X_val = np.asarray(X_val, dtype=np.float32)
        y_val = np.asarray(y_val, dtype=np.float32).reshape(-1)
        if len(X_val) != len(y_val) or len(y_val) == 0:
            raise ValueError("Validation embeddings and labels must be nonempty and aligned.")
        if not np.isfinite(X_val).all() or not np.isfinite(y_val).all():
            raise ValueError("Validation data contains NaN or infinite values.")
        val_tensor = torch.from_numpy(X_val).to(DEVICE)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        examples_seen = 0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            if not torch.isfinite(loss):
                raise RuntimeError(
                    "Non-finite MSE loss encountered. Check label distribution, "
                    "especially extreme story-point values."
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            batch_count = len(batch_X)
            epoch_loss += float(loss.item()) * batch_count
            examples_seen += batch_count

        average_loss = epoch_loss / max(examples_seen, 1)

        # Final development fit has no validation set; train for the duration
        # selected from the median best epoch across CV folds.
        if not has_validation:
            if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
                print(f"    Epoch {epoch:03d}/{epochs} | train_MSE={average_loss:.4f}")
            continue

        model.eval()
        with torch.no_grad():
            val_predictions = model(val_tensor).detach().cpu().numpy()
        val_metrics = calculate_metrics(y_val, val_predictions)
        val_mae = val_metrics["mae"]

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"    Epoch {epoch:03d}/{epochs} | train_MSE={average_loss:.4f} | "
                f"val_MAE={val_mae:.4f} | best_val_MAE={best_val_mae:.4f} | "
                f"patience={epochs_without_improvement}/{patience}"
            )

        if epochs_without_improvement >= patience:
            print(
                f"    Early stopping at epoch {epoch}; best epoch was {best_epoch}."
            )
            break

    if has_validation:
        if best_state is None:
            raise RuntimeError("No valid validation checkpoint was saved.")
        model.load_state_dict(best_state)

    return {
        "model": model,
        "best_epoch": best_epoch,
        "best_val_mae": best_val_mae if has_validation else None,
        "training_seconds": time.time() - training_start,
    }

def predict_raw(model, X):
    """Predict story points in the original numeric scale."""
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(X), 512):
            batch = torch.from_numpy(
                np.asarray(X[start:start + 512], dtype=np.float32)
            ).to(DEVICE)
            predictions.append(model(batch).detach().cpu().numpy())
    if not predictions:
        return np.asarray([], dtype=np.float64)
    return np.concatenate(predictions).astype(np.float64, copy=False)


# ============================================================
# FIVE-FOLD CROSS-VALIDATION
# ============================================================

def run_cross_validation(
    project,
    merged,
    X,
    y,
    epochs,
    patience,
    batch_size,
    learning_rate,
    weight_decay,
):
    """
    Run five-fold CV using development issues only.

    In fold k:
        - Training = development issues in folds other than k.
        - Validation = development issues in fold k.
        - Test issues are never used.
    """

    dev_mask = merged["split"].to_numpy() == "development"
    dev_indices = np.flatnonzero(dev_mask)

    fold_results = []
    best_epochs = []

    for fold in range(1, 6):
        print(f"\n  [{project}] Cross-validation fold {fold}/5")

        fold_values = merged["fold"].to_numpy()

        val_indices = dev_indices[
            fold_values[dev_indices] == fold
        ]

        train_indices = dev_indices[
            fold_values[dev_indices] != fold
        ]

        if len(train_indices) == 0 or len(val_indices) == 0:
            raise ValueError(
                f"{project}: fold {fold} has an empty training "
                "or validation partition."
            )

        X_train = X[train_indices]
        y_train = y[train_indices]

        X_val = X[val_indices]
        y_val = y[val_indices]

        # Baseline: predict the median of this fold's training labels.
        baseline_value = float(np.median(y_train))

        baseline_predictions = np.full(
            len(y_val), baseline_value, dtype=np.float64
        )

        baseline_metrics = calculate_metrics(
            y_val, baseline_predictions
        )

        result = train_one_model(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            epochs=epochs,
            patience=patience,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            seed=SEED + fold,
        )

        model = result["model"]

        val_predictions = predict_raw(model, X_val)

        model_metrics = calculate_metrics(
            y_val, val_predictions
        )

        fold_record = {
            "project": project,
            "fold": fold,
            "n_train": len(train_indices),
            "n_validation": len(val_indices),
            "best_epoch": result["best_epoch"],
            "training_seconds": result["training_seconds"],
            "model_mae": model_metrics["mae"],
            "model_mdae": model_metrics["mdae"],
            "baseline_value": baseline_value,
            "baseline_mae": baseline_metrics["mae"],
            "baseline_mdae": baseline_metrics["mdae"],
        }

        fold_results.append(fold_record)
        best_epochs.append(result["best_epoch"])

        print(
            f"    Fold {fold} results: "
            f"model MAE={model_metrics['mae']:.4f}, "
            f"model MdAE={model_metrics['mdae']:.4f}, "
            f"baseline MAE={baseline_metrics['mae']:.4f}, "
            f"best epoch={result['best_epoch']}"
        )

        # Free memory between folds.
        del result
        del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    fold_df = pd.DataFrame(fold_results)

    metric_columns = [
        "model_mae",
        "model_mdae",
        "baseline_mae",
        "baseline_mdae",
    ]

    summary = {
        "project": project,
        "n_folds": len(fold_results),
        "n_development": int(dev_mask.sum()),
        "best_epochs": best_epochs,
        "final_training_epochs": max(
            1, int(round(float(np.median(best_epochs))))
        ),
    }

    for column in metric_columns:
        summary[f"{column}_mean"] = float(fold_df[column].mean())
        summary[f"{column}_std"] = float(fold_df[column].std(ddof=1))

    return fold_df, summary


# ============================================================
# FINAL MODEL AND LOCKED TEST EVALUATION
# ============================================================

def train_final_and_evaluate_test(
    project,
    merged,
    X,
    y,
    final_epochs,
    batch_size,
    learning_rate,
    weight_decay,
    output_project_dir,
):
    """
    Train a fresh model on all development data and evaluate it
    once on the locked test set.
    """

    split_values = merged["split"].to_numpy()

    dev_indices = np.flatnonzero(split_values == "development")
    test_indices = np.flatnonzero(split_values == "test")

    X_dev = X[dev_indices]
    y_dev = y[dev_indices]

    X_test = X[test_indices]
    y_test = y[test_indices]

    # Train a fresh model on all development data.
    # The held-out test labels are not used for training or selection.
    result = train_one_model(
        X_train=X_dev,
        y_train=y_dev,
        X_val=None,
        y_val=None,
        epochs=final_epochs,
        patience=DEFAULT_PATIENCE,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        seed=SEED + 1000,
    )

    model = result["model"]

    # Save the final checkpoint and target-scaling parameters.
    checkpoint_path = output_project_dir / "final_model.pt"

    torch.save(
        {
            "project": project,
            "model_state_dict": model.state_dict(),
            "embedding_dim": EMBEDDING_DIM,
            "conv_channels": (128, 64, 32),
            "kernel_size": 3,
            "dropout": 0.2,
            "training_epochs": final_epochs,
            "seed": SEED + 1000,
        },
        checkpoint_path,
    )

    # Test predictions are generated only after model selection
    # and final development-set training are complete.
    test_predictions_raw = predict_raw(model, X_test)

    test_predictions = np.maximum(test_predictions_raw, 0.0)

    model_metrics = calculate_metrics(y_test, test_predictions)

    # Test baseline uses only development labels.
    baseline_value = float(np.median(y_dev))

    baseline_predictions = np.full(
        len(y_test), baseline_value, dtype=np.float64
    )

    baseline_metrics = calculate_metrics(
        y_test, baseline_predictions
    )

    test_rows = merged.iloc[test_indices]

    prediction_df = pd.DataFrame(
        {
            "issue_key": test_rows["issue_key"].to_numpy(),
            "project_id": test_rows["project_id"].to_numpy(),
            "actual_story_point": y_test,
            "predicted_story_point_raw": test_predictions_raw,
            "predicted_story_point": test_predictions,
            "absolute_error": np.abs(y_test - test_predictions),
            "baseline_prediction": baseline_value,
            "baseline_absolute_error": np.abs(
                y_test - baseline_predictions
            ),
        }
    )

    prediction_df.to_csv(
        output_project_dir / "test_predictions.csv",
        index=False,
    )

    metrics = {
        "project": project,
        "test_size": len(test_indices),
        "development_size": len(dev_indices),
        "final_training_epochs": final_epochs,
        "model": model_metrics,
        "median_baseline": baseline_metrics,
        "model_beats_baseline_on_mae": (
            model_metrics["mae"] < baseline_metrics["mae"]
        ),
        "checkpoint": str(checkpoint_path),
    }

    with open(
        output_project_dir / "test_metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metrics, file, indent=2)

    print(f"\n  [{project}] FINAL LOCKED TEST RESULTS")
    print(f"    Test issues:       {len(test_indices)}")
    print(f"    CNN MAE:           {model_metrics['mae']:.4f}")
    print(f"    CNN MdAE:          {model_metrics['mdae']:.4f}")
    print(f"    Baseline MAE:      {baseline_metrics['mae']:.4f}")
    print(f"    Baseline MdAE:     {baseline_metrics['mdae']:.4f}")
    print(
        "    CNN beats baseline on MAE:",
        metrics["model_beats_baseline_on_mae"],
    )

    return metrics


# ============================================================
# PROCESS ONE PROJECT
# ============================================================

def run_project(
    project,
    epochs,
    patience,
    batch_size,
    learning_rate,
    weight_decay,
):
    output_project_dir = OUTPUT_DIR / project
    output_project_dir.mkdir(parents=True, exist_ok=True)

    merged, X, y = load_project_data(project)

    print(f"\n{'=' * 65}")
    print(f"TRAINING PROJECT: {project}")
    print(f"Device: {DEVICE}")
    print(f"{'=' * 65}")

    # 1. Five-fold cross-validation on development data.
    fold_df, cv_summary = run_cross_validation(
        project=project,
        merged=merged,
        X=X,
        y=y,
        epochs=epochs,
        patience=patience,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )

    fold_df.to_csv(
        output_project_dir / "cv_fold_metrics.csv",
        index=False,
    )

    with open(
        output_project_dir / "cv_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(cv_summary, file, indent=2)

    # Save the configuration and cross-validation results.
    config = {
        "project": project,
        "seed": SEED,
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "embedding_dim": EMBEDDING_DIM,
        "model": "StoryPointCNN",
        "conv_channels": [128, 64, 32],
        "kernel_size": 3,
        "dropout": 0.2,
        "loss": "L1Loss",
        "optimizer": "Adam",
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "max_epochs": epochs,
        "early_stopping_patience": patience,
        "cross_validation_folds": 5,
        "selected_final_epochs": cv_summary["final_training_epochs"],
        "target_scaling": "none; original numeric labels",
        "evaluation_metrics": ["MAE", "MdAE"],
        "device": str(DEVICE),
    }

    with open(
        output_project_dir / "best_config.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(config, file, indent=2)

    # 2. Train the final model on all development data.
    # 3. Evaluate the locked test set once.
    final_metrics = train_final_and_evaluate_test(
        project=project,
        merged=merged,
        X=X,
        y=y,
        final_epochs=cv_summary["final_training_epochs"],
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        output_project_dir=output_project_dir,
    )

    run_metadata = {
        "project": project,
        "seed": SEED,
        "device": str(DEVICE),
        "python_version": __import__("platform").python_version(),
        "torch_version": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "n_labeled": len(merged),
        "n_development": int(
            (merged["split"] == "development").sum()
        ),
        "n_test": int(
            (merged["split"] == "test").sum()
        ),
        "embedding_shape": list(X.shape),
        "cross_validation": cv_summary,
        "final_test_metrics": final_metrics,
    }

    with open(
        output_project_dir / "run_metadata.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(run_metadata, file, indent=2)

    print(f"\nFinished project: {project}")
    print(f"Results saved in: {output_project_dir}")

    return {
        "project": project,
        "cv_model_mae": cv_summary["model_mae_mean"],
        "cv_model_mdae": cv_summary["model_mdae_mean"],
        "cv_baseline_mae": cv_summary["baseline_mae_mean"],
        "test_model_mae": final_metrics["model"]["mae"],
        "test_model_mdae": final_metrics["model"]["mdae"],
        "test_baseline_mae": final_metrics["median_baseline"]["mae"],
        "test_baseline_mdae": final_metrics["median_baseline"]["mdae"],
        "test_size": final_metrics["test_size"],
    }


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train SBERT + CNN story-point regression models "
            "using five-fold cross-validation."
        )
    )

    parser.add_argument(
        "--project",
        choices=PROJECTS,
        default=None,
        help="Run one project only. Omit to run all eight projects.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Maximum epochs per CV fold.",
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=DEFAULT_PATIENCE,
        help="Early-stopping patience based on validation MAE.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Training batch size.",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help="Adam learning rate.",
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=DEFAULT_WEIGHT_DECAY,
        help="Adam weight decay (default 0.0).",
    )

    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be at least 1.")

    if args.patience < 1:
        parser.error("--patience must be at least 1.")

    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1.")

    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive.")

    if args.weight_decay < 0:
        parser.error("--weight-decay cannot be negative.")

    set_seed(SEED)

    projects_to_run = [args.project] if args.project else PROJECTS

    print("=" * 65)
    print("TAWOS SBERT + CNN STORY-POINT REGRESSION")
    print("=" * 65)
    print(f"Projects: {projects_to_run}")
    print(f"Device: {DEVICE}")
    print(f"Maximum epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Weight decay: {args.weight_decay}")
    print(f"Early-stopping patience: {args.patience}")
    print("Architecture: Conv1D channels 128 -> 64 -> 32; dropout=0.2")
    print("Loss/optimizer: L1Loss (MAE) / Adam")

    all_results = []

    for project in projects_to_run:
        try:
            result = run_project(
                project=project,
                epochs=args.epochs,
                patience=args.patience,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
            )

            all_results.append(result)

        except Exception as error:
            print(f"\nERROR while processing {project}: {error}")
            raise

    if all_results:
        summary_df = pd.DataFrame(all_results)

        summary_path = OUTPUT_DIR / "all_projects_training_summary.csv"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(summary_path, index=False)

        print("\n" + "=" * 65)
        print("ALL PROJECTS SUMMARY")
        print("=" * 65)
        print(summary_df.to_string(index=False))
        print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
