#!/usr/bin/env python3
"""
Evaluate Module for TAWOS SBERT + CNN Pipeline.
Implements Step 8 (Evaluation and baseline) of COMMON_PIPELINE.md:
  - Evaluates final_model.pt on the locked held-out test set
  - Computes test MAE and MdAE
  - Computes training-median baseline on the test set
  - Saves:
      - test_predictions.csv (issue_key, actual, prediction, absolute_error)
      - test_metrics.json
      - run_metadata.json
"""

import argparse
import json
import platform
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, median_absolute_error
import torch

from src.model import SBERT_CNN

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def evaluate_project(
    project_key: str,
    project_id: int,
    manifest_path: Path,
    embeddings_path: Path,
    model_path: Path,
    output_dir: Path,
    seed: int = 42
):
    device = get_device()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load manifest and cached embeddings
    manifest = pd.read_csv(manifest_path)
    cache = torch.load(embeddings_path, map_location="cpu")
    key_to_idx = {k: i for i, k in enumerate(cache["issue_keys"])}
    
    # Development targets (for training median baseline)
    dev_keys = manifest[manifest["split"] == "development"]["issue_key"].tolist()
    dev_indices = [key_to_idx[k] for k in dev_keys]
    y_dev = cache["targets"][dev_indices].numpy()
    baseline_median = float(np.median(y_dev))
    
    # Test set
    test_manifest = manifest[manifest["split"] == "test"].copy().reset_index(drop=True)
    test_indices = [key_to_idx[k] for k in test_manifest["issue_key"]]
    
    X_test = cache["embeddings"][test_indices]
    y_test = cache["targets"][test_indices].numpy()
    
    # Load final model
    model = SBERT_CNN(in_features=384, dropout=0.2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    with torch.no_grad():
        preds = model(X_test.to(device)).cpu().numpy()
        
    abs_errors = np.abs(y_test - preds)
    test_mae = float(mean_absolute_error(y_test, preds))
    test_mdae = float(median_absolute_error(y_test, preds))
    
    baseline_preds = np.full_like(y_test, baseline_median)
    base_mae = float(mean_absolute_error(y_test, baseline_preds))
    base_mdae = float(median_absolute_error(y_test, baseline_preds))
    
    print(f"[{project_key}] Test Evaluation ({len(y_test)} examples):")
    print(f"  Model MAE: {test_mae:.4f} | Baseline MAE: {base_mae:.4f}")
    print(f"  Model MdAE: {test_mdae:.4f} | Baseline MdAE: {base_mdae:.4f}")
    
    # Save test_predictions.csv
    df_preds = pd.DataFrame({
        "issue_key": test_manifest["issue_key"],
        "actual": y_test,
        "prediction": preds,
        "absolute_error": abs_errors
    })
    df_preds.to_csv(output_dir / "test_predictions.csv", index=False)
    
    # Save test_metrics.json
    test_metrics = {
        "project_id": project_id,
        "project_key": project_key,
        "mae": test_mae,
        "mdae": test_mdae,
        "test_count": int(len(y_test)),
        "baseline_mae": base_mae,
        "baseline_mdae": base_mdae,
        "training_median_value": baseline_median,
        "encoder_checkpoint": "sentence-transformers/all-MiniLM-L6-v2",
        "model_version": "v1.0_sbert_cnn",
        "seed": seed
    }
    with open(output_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)
        
    # Save run_metadata.json
    import sklearn
    run_meta = {
        "project_key": project_key,
        "project_id": project_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "sklearn_version": sklearn.__version__,
        "device": str(device),
        "random_seed": seed,
        "encoder": "sentence-transformers/all-MiniLM-L6-v2",
        "model_architecture": "3-layer CNN (128-64-32)",
        "split_ratio": "80/20 dev/test",
        "validation_strategy": "5-fold CV on dev"
    }
    with open(output_dir / "run_metadata.json", "w") as f:
        json.dump(run_meta, f, indent=2)
        
    return test_metrics

def main():
    parser = argparse.ArgumentParser(description="Evaluate final model on locked test set")
    parser.add_argument("--output_base", default="outputs")
    parser.add_argument("--allocation_path", default="project_allocation.csv")
    parser.add_argument("--member", default="devanarayanan")
    parser.add_argument("--project_key", default=None)
    parser.add_argument("--seed", type=int, default=42)
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
        model_path = p_out_dir / "final_model.pt"
        
        if not manifest_path.exists() or not emb_path.exists() or not model_path.exists():
            print(f"Skipping {p_key}: Missing required artifacts.")
            continue
            
        evaluate_project(
            project_key=p_key,
            project_id=p_id,
            manifest_path=manifest_path,
            embeddings_path=emb_path,
            model_path=model_path,
            output_dir=p_out_dir,
            seed=args.seed
        )

if __name__ == "__main__":
    main()
