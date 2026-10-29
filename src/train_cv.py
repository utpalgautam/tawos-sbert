#!/usr/bin/env python3
"""
Train Cross-Validation and Final Model Fitting Module.
Implements Steps 6 & 7 of COMMON_PIPELINE.md:
  - 5-fold cross-validation inside development data
  - Early stopping on validation MAE (patience=10)
  - Saves cv_fold_metrics.csv and cv_summary.csv
  - Predeclared training-duration rule: median best epoch across folds
  - Refits fresh model on 100% development data
  - Saves final_model.pt and best_config.json
"""

import argparse
import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, median_absolute_error
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.model import SBERT_CNN

def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for x_b, y_b in dataloader:
        x_b, y_b = x_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        preds = model(x_b)
        loss = criterion(preds, y_b)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_b)
    return total_loss / len(dataloader.dataset)

def eval_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for x_b, y_b in dataloader:
            x_b = x_b.to(device)
            preds = model(x_b)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(y_b.numpy().tolist())
    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    mae = float(mean_absolute_error(y_true, y_pred))
    mdae = float(median_absolute_error(y_true, y_pred))
    return mae, mdae, y_pred

def run_cv_and_final_fit(
    project_key: str,
    project_id: int,
    manifest_path: Path,
    embeddings_path: Path,
    output_dir: Path,
    seed: int = 42,
    lr: float = 0.001,
    batch_size: int = 64,
    max_epochs: int = 100,
    patience: int = 10
):
    set_seed(seed)
    device = get_device()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load manifest and cached embeddings
    manifest = pd.read_csv(manifest_path)
    cache = torch.load(embeddings_path, map_location="cpu")
    
    # Map issue_key to index in cache
    key_to_idx = {k: i for i, k in enumerate(cache["issue_keys"])}
    
    # Filter to development set
    dev_manifest = manifest[manifest["split"] == "development"].copy().reset_index(drop=True)
    dev_indices = [key_to_idx[k] for k in dev_manifest["issue_key"]]
    
    X_dev = cache["embeddings"][dev_indices]
    y_dev = cache["targets"][dev_indices]
    folds_dev = dev_manifest["fold"].astype(int).values
    
    print(f"[{project_key}] Starting 5-Fold CV on {len(X_dev)} development issues (device={device})...")
    
    fold_metrics = []
    n_folds = 5
    
    for f in range(1, n_folds + 1):
        val_mask = (folds_dev == f)
        train_mask = ~val_mask
        
        X_tr, y_tr = X_dev[train_mask], y_dev[train_mask]
        X_va, y_va = X_dev[val_mask], y_dev[val_mask]
        
        train_ds = TensorDataset(X_tr, y_tr)
        val_ds = TensorDataset(X_va, y_va)
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        
        # Fresh model
        set_seed(seed + f)
        model = SBERT_CNN(in_features=384, dropout=0.2).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        best_val_mae = float("inf")
        best_val_mdae = float("inf")
        best_epoch = 1
        epochs_no_improve = 0
        best_weights = None
        
        for epoch in range(1, max_epochs + 1):
            train_epoch(model, train_loader, optimizer, criterion, device)
            val_mae, val_mdae, _ = eval_model(model, val_loader, device)
            
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_val_mdae = val_mdae
                best_epoch = epoch
                best_weights = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    break
                    
        print(f"  Fold {f}: Best Epoch={best_epoch}, Val MAE={best_val_mae:.4f}, Val MdAE={best_val_mdae:.4f}")
        fold_metrics.append({
            "project_id": project_id,
            "fold": f,
            "mae": best_val_mae,
            "mdae": best_val_mdae,
            "best_epoch": best_epoch,
            "train_count": int(len(X_tr)),
            "validation_count": int(len(X_va)),
            "seed": seed
        })
        
    df_fold_metrics = pd.DataFrame(fold_metrics)
    df_fold_metrics.to_csv(output_dir / "cv_fold_metrics.csv", index=False)
    
    # CV Summary
    cv_summary = {
        "project_id": project_id,
        "mean_mae": float(df_fold_metrics["mae"].mean()),
        "std_mae": float(df_fold_metrics["mae"].std()),
        "mean_mdae": float(df_fold_metrics["mdae"].mean()),
        "std_mdae": float(df_fold_metrics["mdae"].std()),
        "median_best_epoch": int(np.median(df_fold_metrics["best_epoch"]))
    }
    pd.DataFrame([cv_summary]).to_csv(output_dir / "cv_summary.csv", index=False)
    print(f"[{project_key}] CV Summary: Mean MAE={cv_summary['mean_mae']:.4f} (±{cv_summary['std_mae']:.4f}), "
          f"Mean MdAE={cv_summary['mean_mdae']:.4f}, Median Best Epoch={cv_summary['median_best_epoch']}")
          
    # Final model refitting on 100% development data
    final_epochs = max(1, cv_summary["median_best_epoch"])
    print(f"[{project_key}] Refitting final model on all {len(X_dev)} dev examples for {final_epochs} epochs...")
    set_seed(seed)
    final_model = SBERT_CNN(in_features=384, dropout=0.2).to(device)
    optimizer = torch.optim.Adam(final_model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    full_dev_ds = TensorDataset(X_dev, y_dev)
    full_dev_loader = DataLoader(full_dev_ds, batch_size=batch_size, shuffle=True)
    
    for epoch in range(1, final_epochs + 1):
        train_epoch(final_model, full_dev_loader, optimizer, criterion, device)
        
    # Save final model
    model_path = output_dir / "final_model.pt"
    torch.save(final_model.state_dict(), model_path)
    print(f"[{project_key}] Saved final model checkpoint to {model_path}")
    
    # Save best config
    best_config = {
        "project_key": project_key,
        "project_id": project_id,
        "sbert_checkpoint": "sentence-transformers/all-MiniLM-L6-v2",
        "embedding_dim": 384,
        "cnn_architecture": {
            "conv_channels": [128, 64, 32],
            "kernel_size": 3,
            "padding": 1,
            "dropout": 0.2
        },
        "training": {
            "optimizer": "Adam",
            "learning_rate": lr,
            "batch_size": batch_size,
            "loss": "MSE",
            "median_best_epoch": final_epochs,
            "random_seed": seed
        }
    }
    with open(output_dir / "best_config.json", "w") as f:
        json.dump(best_config, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Run CV and fit final CNN model")
    parser.add_argument("--output_base", default="outputs")
    parser.add_argument("--allocation_path", default="project_allocation.csv")
    parser.add_argument("--member", default="devanarayanan")
    parser.add_argument("--project_key", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args()
    
    alloc_df = pd.read_csv(args.allocation_path)
    if args.member:
        alloc_df = alloc_df[alloc_df["assigned_member"] == args.member]
    if args.project_key:
        alloc_df = alloc_df[alloc_df["project_key"] == args.project_key]
        
    for _, row in alloc_df.iterrows():
        p_key = str(row["project_key"])
        p_id = int(row["project_id"])
        p_out_dir = Path(args.output_base) / args.member / p_key
        manifest_path = p_out_dir / "split_manifest.csv"
        emb_path = p_out_dir / "embeddings_labeled.pt"
        
        if not manifest_path.exists() or not emb_path.exists():
            print(f"Skipping {p_key}: Missing {manifest_path} or {emb_path}")
            continue
            
        run_cv_and_final_fit(
            project_key=p_key,
            project_id=p_id,
            manifest_path=manifest_path,
            embeddings_path=emb_path,
            output_dir=p_out_dir,
            seed=args.seed,
            lr=args.lr,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            patience=args.patience
        )

if __name__ == "__main__":
    main()
