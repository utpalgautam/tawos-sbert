#!/usr/bin/env python3
"""
Final shared SBERT-CNN pipeline for Problem 5.

- Uses cached, frozen SBERT embeddings.
- Trains one shared CNN across all assigned projects.
- Uses the existing development/test split and five development CV folds.
- Uses a grouped inner validation split for early stopping in each outer CV fold.
- Selects final training epochs from the median best epoch across CV folds.
- Fits the final model on all development rows and evaluates the held-out test set once.
- Saves overall and per-project metrics, predictions, checkpoints, histories, and metadata.
"""
from __future__ import annotations

import copy
import json
import random
import statistics
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import GroupShuffleSplit
import yaml

BASE_DIR = Path(__file__).resolve().parents[1]


class SharedStoryPointCNN(nn.Module):
    """Shared CNN over SBERT vectors, with a learned project-ID embedding."""

    def __init__(
        self,
        embedding_dim: int = 384,
        project_count: int = 8,
        project_embedding_dim: int = 8,
        channels=(128, 64, 32),
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.features = nn.Sequential(
            nn.Conv1d(1, channels[0], kernel_size, padding=padding),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(channels[0], channels[1], kernel_size, padding=padding),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(channels[1], channels[2], kernel_size, padding=padding),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.project_embedding = nn.Embedding(project_count, project_embedding_dim)
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(channels[2] + project_embedding_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x, project_index):
        z = self.features(x.unsqueeze(1)).squeeze(-1)
        p = self.project_embedding(project_index)
        return self.regressor(torch.cat([z, p], dim=1)).squeeze(1)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    err = y_true - y_pred
    ae = np.abs(err)
    return {
        "n": int(len(y_true)),
        "MAE": float(np.mean(ae)),
        "MdAE": float(np.median(ae)),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "R2": float(1 - np.sum(err ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
        if len(y_true) > 1 and np.sum((y_true - np.mean(y_true)) ** 2) > 0
        else float("nan"),
    }


def make_loader(X, P, y=None, batch_size=64, shuffle=False, device=None):
    # Copy memory-mapped/read-only arrays before creating writable tensors.
    xt = torch.from_numpy(np.array(X, dtype=np.float32, order="C", copy=True))
    pt = torch.from_numpy(np.array(P, dtype=np.int64, order="C", copy=True))
    if y is None:
        ds = TensorDataset(xt, pt)
    else:
        yt = torch.from_numpy(np.array(y, dtype=np.float32, order="C", copy=True))
        ds = TensorDataset(xt, pt, yt)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=(device is not None and device.type == "cuda"),
    )


def predict(model, X, P, device, batch_size=512):
    model.eval()
    out = []
    loader = make_loader(X, P, batch_size=batch_size, shuffle=False, device=device)
    with torch.no_grad():
        for batch in loader:
            xb, pb = batch
            xb = xb.to(device, non_blocking=(device.type == "cuda"))
            pb = pb.to(device, non_blocking=(device.type == "cuda"))
            out.append(model(xb, pb).cpu().numpy())
    return np.concatenate(out) if out else np.array([], dtype=np.float32)


def fit_model(
    X_train, P_train, y_train, X_val, P_val, y_val, *,
    project_count, channels, kernel_size, dropout, lr, batch_size,
    max_epochs, patience, seed, device, use_early_stopping=True,
):
    set_seed(seed)
    model = SharedStoryPointCNN(
        embedding_dim=X_train.shape[1],
        project_count=project_count,
        channels=channels,
        kernel_size=kernel_size,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss()
    train_loader = make_loader(
        X_train, P_train, y_train, batch_size=batch_size, shuffle=True, device=device
    )
    best_state = None
    best_val_mae = float("inf")
    best_epoch = max_epochs
    stale = 0
    history = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        for xb, pb, yb in train_loader:
            xb = xb.to(device, non_blocking=(device.type == "cuda"))
            pb = pb.to(device, non_blocking=(device.type == "cuda"))
            yb = yb.to(device, non_blocking=(device.type == "cuda"))
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb, pb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(yb)
            seen += len(yb)

        train_loss = total_loss / max(seen, 1)
        if X_val is not None and y_val is not None:
            val_pred = predict(model, X_val, P_val, device)
            val_mae = metrics(y_val, val_pred)["MAE"]
        else:
            val_mae = float("nan")
        history.append({
            "epoch": epoch,
            "train_mae_loss": train_loss,
            "validation_mae": val_mae,
        })

        if use_early_stopping:
            if val_mae < best_val_mae - 1e-8:
                best_val_mae = val_mae
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
            if epoch == 1 or epoch % 10 == 0:
                print(
                    f"  epoch={epoch:03d} train_MAE_loss={train_loss:.4f} "
                    f"val_MAE={val_mae:.4f} best={best_val_mae:.4f}@{best_epoch}"
                )
            if stale >= patience:
                print(f"  early stopping at epoch {epoch}; best epoch={best_epoch}")
                break

    if use_early_stopping:
        if best_state is None:
            raise RuntimeError("Early stopping did not record a best model state.")
        model.load_state_dict(best_state)
    return model, best_epoch, pd.DataFrame(history)


def main():
    with open(BASE_DIR / "configs/problem5.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg["experiment"]["random_seed"])
    set_seed(seed)
    data_cfg = cfg["data"]
    cnn_cfg = cfg["cnn"]
    train_cfg = cfg["training"]
    out_cfg = cfg["output"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = BASE_DIR / out_cfg["results_dir"] / "cnn" / "overall"
    model_dir = BASE_DIR / out_cfg["models_dir"] / "cnn"
    log_dir = BASE_DIR / out_cfg["logs_dir"] / "cnn"
    for d in (output_dir, model_dir, log_dir):
        d.mkdir(parents=True, exist_ok=True)

    metadata_path = BASE_DIR / data_cfg["embeddings_dir"] / "embedding_metadata.csv"
    embedding_path = BASE_DIR / data_cfg["embeddings_dir"] / "sbert_embeddings.npy"
    manifest_path = BASE_DIR / data_cfg["splits_dir"] / "split_manifest.csv"
    folds_path = BASE_DIR / data_cfg["splits_dir"] / "development_cv_folds.csv"

    meta = pd.read_csv(metadata_path)
    emb = np.load(embedding_path, mmap_mode="r")
    manifest = pd.read_csv(manifest_path)
    folds = pd.read_csv(folds_path)

    key = data_cfg["issue_key_column"]
    project_col = data_cfg["project_column"]
    target_col = data_cfg["target_column"]
    expected_projects = list(data_cfg["assigned_project_ids"])

    if len(meta) != len(emb):
        raise ValueError("Embedding metadata and embedding matrix lengths differ.")
    if meta[key].duplicated().any():
        raise ValueError("Duplicate issue keys in embedding metadata.")
    if not np.isfinite(emb).all():
        raise ValueError("Embedding matrix contains non-finite values.")

    meta = meta.merge(
        manifest[[key, "project_id", "story_point", "split"]],
        on=key, how="left", validate="one_to_one", suffixes=("", "_manifest"),
    )
    if meta["split"].isna().any():
        raise ValueError("Some embeddings do not have split assignments.")
    for c in ("project_id_manifest", "story_point_manifest"):
        if c in meta.columns:
            base = c.replace("_manifest", "")
            if not np.array_equal(meta[base].to_numpy(), meta[c].to_numpy()):
                raise ValueError(f"Metadata mismatch for {base} against split manifest.")
            meta = meta.drop(columns=[c])

    meta = meta.merge(
        folds[[key, "cv_fold"]], on=key, how="left", validate="one_to_one"
    )
    dev_mask = meta["split"].eq("development").to_numpy()
    test_mask = meta["split"].eq("test").to_numpy()
    if meta.loc[test_mask, "cv_fold"].notna().any():
        raise ValueError("Test rows must not have CV fold assignments.")
    if meta.loc[dev_mask, "cv_fold"].isna().any():
        raise ValueError("Some development rows have no CV fold.")
    meta["cv_fold"] = pd.to_numeric(meta["cv_fold"], errors="coerce")
    meta.loc[dev_mask, "cv_fold"] = meta.loc[dev_mask, "cv_fold"].astype(int)

    # Use stable categorical indices for the project's learned embedding.
    project_to_index = {int(pid): i for i, pid in enumerate(expected_projects)}
    meta[project_col] = pd.to_numeric(meta[project_col], errors="raise").astype(int)
    if sorted(meta[project_col].unique().tolist()) != sorted(expected_projects):
        raise ValueError("Observed projects differ from assigned_project_ids.")
    meta["project_index"] = meta[project_col].map(project_to_index).astype(int)

    X_all = np.asarray(emb, dtype=np.float32)
    y_all = meta[target_col].to_numpy(dtype=np.float32)
    if not np.isfinite(y_all).all():
        raise ValueError("Targets contain non-finite values.")

    dev_idx = np.flatnonzero(dev_mask)
    test_idx = np.flatnonzero(test_mask)
    dev = meta.iloc[dev_idx].reset_index(drop=True)
    X_dev = X_all[dev_idx]
    y_dev = y_all[dev_idx]
    p_dev = dev["project_index"].to_numpy(dtype=np.int64)
    # Group identical text within each project, matching CV-fold creation.
    text_hash = dev["input_text_sha256"].fillna(dev[key].astype(str)).astype(str)
    groups = (
        dev[project_col].astype(str) + ":" + text_hash
    ).to_numpy()

    n_folds = int(cfg["experiment"]["validation_folds"])
    folds_found = sorted(dev["cv_fold"].astype(int).unique().tolist())
    if folds_found != list(range(n_folds)):
        raise ValueError(f"Expected CV folds 0..{n_folds-1}; found {folds_found}")

    channels = tuple(int(v) for v in cnn_cfg["conv_channels"])
    kernel_size = int(cnn_cfg["kernel_size"])
    dropout = float(cnn_cfg["dropout"])
    max_epochs = int(train_cfg["max_epochs"])
    patience = int(train_cfg["early_stopping_patience"])
    batch_size = int(train_cfg["batch_size"])
    lr = float(train_cfg["learning_rate"])
    project_count = len(expected_projects)

    print(f"Device: {device}")
    print(f"All labeled rows: {len(meta)} | development: {len(dev_idx)} | held-out test: {len(test_idx)}")
    print(f"Projects: {expected_projects}")
    print(f"Embedding shape: {X_all.shape}; project embedding dim: 8")
    print("Model: shared three-layer CNN + learned project-ID embedding")
    print("Loss: MAE; test set is not used for selection.")

    # Outer CV: early stopping uses a grouped inner split from outer-training rows.
    cv_pred_records = []
    cv_fold_records = []
    cv_project_records = []
    best_epochs = []
    all_histories = []

    for fold in range(n_folds):
        print(f"\n{'=' * 64}\nOUTER CV FOLD {fold}\n{'=' * 64}")
        outer_val_mask = dev["cv_fold"].astype(int).to_numpy() == fold
        outer_train_pos = np.flatnonzero(~outer_val_mask)
        outer_val_pos = np.flatnonzero(outer_val_mask)

        inner_splitter = GroupShuffleSplit(n_splits=1, test_size=0.12, random_state=seed + fold)
        inner_train_rel, inner_val_rel = next(
            inner_splitter.split(
                X_dev[outer_train_pos],
                y_dev[outer_train_pos],
                groups=groups[outer_train_pos],
            )
        )
        inner_train_pos = outer_train_pos[inner_train_rel]
        inner_val_pos = outer_train_pos[inner_val_rel]

        # Guard against identical-text leakage in the inner split.
        if set(groups[inner_train_pos]).intersection(set(groups[inner_val_pos])):
            raise ValueError(f"Duplicate-text leakage in inner validation split for fold {fold}")

        model, best_epoch, history = fit_model(
            X_dev[inner_train_pos], p_dev[inner_train_pos], y_dev[inner_train_pos],
            X_dev[inner_val_pos], p_dev[inner_val_pos], y_dev[inner_val_pos],
            project_count=project_count, channels=channels, kernel_size=kernel_size,
            dropout=dropout, lr=lr, batch_size=batch_size, max_epochs=max_epochs,
            patience=patience, seed=seed + fold, device=device,
        )
        best_epochs.append(best_epoch)
        history.insert(0, "outer_fold", fold)
        all_histories.append(history)

        # Retrain on all outer-training rows for the selected number of epochs.
        # The outer-validation fold remains completely unseen during training.
        model, _, refit_history = fit_model(
            X_dev[outer_train_pos], p_dev[outer_train_pos], y_dev[outer_train_pos],
            None, None, None,
            project_count=project_count, channels=channels, kernel_size=kernel_size,
            dropout=dropout, lr=lr, batch_size=batch_size,
            max_epochs=best_epoch, patience=patience,
            seed=seed + 1000 + fold, device=device,
            use_early_stopping=False,
        )

        pred = np.maximum(0.0, predict(
            model, X_dev[outer_val_pos], p_dev[outer_val_pos], device
        ))
        actual = y_dev[outer_val_pos]
        val_df = dev.iloc[outer_val_pos].copy()
        fold_metric = metrics(actual, pred)
        cv_fold_records.append({
            "fold": fold, "model": "shared_sbert_cnn", "best_epoch": best_epoch,
            "outer_train_rows": len(outer_train_pos), "inner_train_rows": len(inner_train_pos),
            "inner_validation_rows": len(inner_val_pos), "outer_validation_rows": len(outer_val_pos),
            **fold_metric,
        })
        print(f"Outer fold {fold}: MAE={fold_metric['MAE']:.4f}, MdAE={fold_metric['MdAE']:.4f}, RMSE={fold_metric['RMSE']:.4f}")

        # Fold-specific baselines are fit on outer training only.
        global_med = float(np.median(y_dev[outer_train_pos]))
        train_df = dev.iloc[outer_train_pos]
        project_medians = train_df.groupby(project_col)[target_col].median()
        base_global = np.full(len(outer_val_pos), global_med, dtype=np.float32)
        base_project = val_df[project_col].map(project_medians).to_numpy(dtype=np.float32)
        if not np.isfinite(base_project).all():
            raise ValueError("A validation project has no outer-training median.")

        for model_name, pred_values in [
            ("shared_sbert_cnn", pred),
            ("global_median_baseline", base_global),
            ("project_median_baseline", base_project),
        ]:
            for i, row in enumerate(val_df.itertuples(index=False)):
                cv_pred_records.append({
                    "issue_key": str(getattr(row, key)),
                    "project_id": int(getattr(row, project_col)),
                    "actual_story_point": float(actual[i]),
                    "predicted_story_point": float(pred_values[i]),
                    "absolute_error": float(abs(actual[i] - pred_values[i])),
                    "fold": fold,
                    "model": model_name,
                    "split": "development_oof",
                })
            for pid in expected_projects:
                mask = val_df[project_col].to_numpy() == pid
                if mask.any():
                    cv_project_records.append({
                        "fold": fold, "project_id": pid, "model": model_name,
                        **metrics(actual[mask], pred_values[mask]),
                    })

    cv_predictions = pd.DataFrame(cv_pred_records)
    cv_predictions.to_csv(output_dir / "cv_oof_predictions.csv", index=False)
    pd.DataFrame(cv_fold_records).to_csv(output_dir / "cv_fold_metrics.csv", index=False)
    pd.DataFrame(cv_project_records).to_csv(output_dir / "cv_fold_project_metrics.csv", index=False)
    pd.concat(all_histories, ignore_index=True).to_csv(log_dir / "cv_training_history.csv", index=False)

    cv_overall_records = []
    for name, frame in cv_predictions.groupby("model"):
        cv_overall_records.append({"model": name, **metrics(frame["actual_story_point"], frame["predicted_story_point"])})
    pd.DataFrame(cv_overall_records).to_csv(output_dir / "cv_overall_metrics.csv", index=False)

    cv_project_summary = []
    for (pid, name), frame in cv_predictions.groupby(["project_id", "model"]):
        cv_project_summary.append({"project_id": int(pid), "model": name, **metrics(frame["actual_story_point"], frame["predicted_story_point"])})
    pd.DataFrame(cv_project_summary).to_csv(output_dir / "cv_project_metrics.csv", index=False)

    # Choose fixed final epoch count from development-only inner-validation selections.
    final_epochs = max(1, int(round(float(statistics.median(best_epochs)))))
    print(f"\nCV best epochs: {best_epochs}; final fixed epoch count (median): {final_epochs}")

    # Final fit on all development rows; no test data used for training or epoch selection.
    set_seed(seed)
    final_model = SharedStoryPointCNN(
        embedding_dim=X_dev.shape[1], project_count=project_count, channels=channels,
        kernel_size=kernel_size, dropout=dropout,
    ).to(device)
    final_optimizer = torch.optim.Adam(final_model.parameters(), lr=lr)
    criterion = nn.L1Loss()
    train_loader = make_loader(X_dev, p_dev, y_dev, batch_size=batch_size, shuffle=True, device=device)
    final_history = []
    for epoch in range(1, final_epochs + 1):
        final_model.train()
        total = 0.0
        seen = 0
        for xb, pb, yb in train_loader:
            xb = xb.to(device, non_blocking=(device.type == "cuda"))
            pb = pb.to(device, non_blocking=(device.type == "cuda"))
            yb = yb.to(device, non_blocking=(device.type == "cuda"))
            final_optimizer.zero_grad(set_to_none=True)
            pred_batch = final_model(xb, pb)
            loss = criterion(pred_batch, yb)
            loss.backward()
            final_optimizer.step()
            total += float(loss.item()) * len(yb)
            seen += len(yb)
        final_history.append({"epoch": epoch, "train_mae_loss": total / max(seen, 1)})
        if epoch == 1 or epoch % 10 == 0 or epoch == final_epochs:
            print(f"Final fit epoch {epoch}/{final_epochs}: train MAE loss={total / max(seen, 1):.4f}")

    pd.DataFrame(final_history).to_csv(log_dir / "final_training_history.csv", index=False)

    # Evaluate the held-out test set exactly once after final training.
    X_test = X_all[test_idx]
    y_test = y_all[test_idx]
    test_df = meta.iloc[test_idx].copy()
    p_test = test_df["project_index"].to_numpy(dtype=np.int64)
    test_pred = np.maximum(0.0, predict(final_model, X_test, p_test, device))
    test_baseline_global = np.full(len(test_idx), float(np.median(y_dev)), dtype=np.float32)
    dev_project_medians = dev.groupby(project_col)[target_col].median()
    test_baseline_project = test_df[project_col].map(dev_project_medians).to_numpy(dtype=np.float32)

    test_prediction_records = []
    for model_name, pred_values in [
        ("shared_sbert_cnn", test_pred),
        ("global_median_baseline", test_baseline_global),
        ("project_median_baseline", test_baseline_project),
    ]:
        for i, (_, row) in enumerate(test_df.iterrows()):
            test_prediction_records.append({
                "issue_key": str(row[key]),
                "project_id": int(row[project_col]),
                "actual_story_point": float(y_test[i]),
                "predicted_story_point": float(pred_values[i]),
                "absolute_error": float(abs(y_test[i] - pred_values[i])),
                "model": model_name,
                "split": "test",
            })
    test_predictions = pd.DataFrame(test_prediction_records)
    test_predictions.to_csv(output_dir / "test_predictions.csv", index=False)

    test_overall = []
    for name, frame in test_predictions.groupby("model"):
        test_overall.append({"model": name, **metrics(frame["actual_story_point"], frame["predicted_story_point"])})
    pd.DataFrame(test_overall).to_csv(output_dir / "test_overall_metrics.csv", index=False)

    test_project = []
    for (pid, name), frame in test_predictions.groupby(["project_id", "model"]):
        test_project.append({"project_id": int(pid), "model": name, **metrics(frame["actual_story_point"], frame["predicted_story_point"])})
    pd.DataFrame(test_project).to_csv(output_dir / "test_project_metrics.csv", index=False)

    # Equal-project averages of per-project metrics.
    for split_name, frame in [("cv", cv_project_summary), ("test", test_project)]:
        df = pd.DataFrame(frame)
        if not df.empty:
            equal = df.groupby("model")[["MAE", "MdAE", "RMSE", "R2"]].mean().reset_index()
            equal.insert(0, "split", split_name)
            equal.to_csv(output_dir / f"{split_name}_equal_project_metrics.csv", index=False)

    checkpoint = {
        "model_state_dict": final_model.cpu().state_dict(),
        "model_class": "SharedStoryPointCNN",
        "embedding_dim": int(X_dev.shape[1]),
        "project_ids": expected_projects,
        "project_to_index": project_to_index,
        "project_embedding_dim": 8,
        "channels": list(channels),
        "kernel_size": kernel_size,
        "dropout": dropout,
        "final_epochs": final_epochs,
        "seed": seed,
        "sbert_checkpoint": cfg["sbert"]["checkpoint"],
        "target_transform": "none",
        "prediction_clipping": "min 0",
    }
    torch.save(checkpoint, model_dir / "model.pt")
    final_model.to(device)

    summary = {
        "experiment_name": "shared_sbert_cnn",
        "device": str(device),
        "seed": seed,
        "model": "One shared three-layer CNN with learned project-ID embedding",
        "project_ids": expected_projects,
        "project_to_index": project_to_index,
        "embedding_model": cfg["sbert"]["checkpoint"],
        "embedding_dim": int(X_dev.shape[1]),
        "embedding_frozen": True,
        "training_loss": "MAE / L1Loss",
        "optimizer": "Adam",
        "learning_rate": lr,
        "batch_size": batch_size,
        "max_cv_epochs": max_epochs,
        "early_stopping_patience": patience,
        "cv_best_epochs": best_epochs,
        "final_epochs": final_epochs,
        "development_rows": int(len(dev_idx)),
        "test_rows": int(len(test_idx)),
        "cv_folds": n_folds,
        "test_evaluated_after_final_fit_only": True,
        "outputs": {
            "cv_oof_predictions": str((output_dir / "cv_oof_predictions.csv").relative_to(BASE_DIR)),
            "cv_overall_metrics": str((output_dir / "cv_overall_metrics.csv").relative_to(BASE_DIR)),
            "cv_project_metrics": str((output_dir / "cv_project_metrics.csv").relative_to(BASE_DIR)),
            "test_predictions": str((output_dir / "test_predictions.csv").relative_to(BASE_DIR)),
            "test_overall_metrics": str((output_dir / "test_overall_metrics.csv").relative_to(BASE_DIR)),
            "test_project_metrics": str((output_dir / "test_project_metrics.csv").relative_to(BASE_DIR)),
            "checkpoint": str((model_dir / "model.pt").relative_to(BASE_DIR)),
        },
    }
    with open(output_dir / "experiment_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 64}\nFINAL HELD-OUT TEST RESULTS\n{'=' * 64}")
    print(pd.DataFrame(test_overall).to_string(index=False))
    print("\nPer-project test metrics:")
    print(pd.DataFrame(test_project).sort_values(["project_id", "model"]).to_string(index=False))
    print(f"\nSaved final pipeline outputs to: {output_dir}")
    print(f"Saved model checkpoint to: {model_dir / 'model.pt'}")


if __name__ == "__main__":
    main()
