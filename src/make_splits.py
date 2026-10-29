#!/usr/bin/env python3
"""
Make Splits Module for TAWOS SBERT + CNN Pipeline.
Implements Step 4 (Required Problem 5 Split) of COMMON_PIPELINE.md:
  - 80% development / 20% locked held-out test split
  - 5-fold cross-validation inside development data
  - Group-aware splitting by duplicate_group so exact duplicates never leak across splits or folds
  - Outputs split_manifest.csv:
      issue_key,project_id,split,fold,duplicate_group
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, KFold

def create_splits(df_labeled: pd.DataFrame, random_seed: int = 42, n_folds: int = 5, test_fraction: float = 0.20):
    df = df_labeled.copy().reset_index(drop=True)
    
    # Check if duplicate_group exists
    if "duplicate_group" not in df.columns:
        # Create groups on input_text or issue_key
        df["duplicate_group"] = np.arange(len(df))
        
    groups = df["duplicate_group"].values
    
    # 80/20 Dev/Test Split
    # Use GroupShuffleSplit to ensure duplicate text does not cross dev and test
    gss = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=random_seed)
    train_idx, test_idx = next(gss.split(df, groups=groups))
    
    df["split"] = ""
    df.loc[train_idx, "split"] = "development"
    df.loc[test_idx, "split"] = "test"
    
    # 5-fold inside development
    df["fold"] = ""
    dev_mask = df["split"] == "development"
    dev_df = df[dev_mask].copy().reset_index()
    dev_groups = dev_df["duplicate_group"].values
    
    unique_groups = len(np.unique(dev_groups))
    if unique_groups >= n_folds:
        # We can use GroupKFold
        gkf = GroupKFold(n_splits=n_folds)
        for fold_num, (_, val_idx) in enumerate(gkf.split(dev_df, groups=dev_groups), start=1):
            original_indices = dev_df.loc[val_idx, "index"].values
            df.loc[original_indices, "fold"] = str(fold_num)
    else:
        # Fallback to standard KFold if groups are fewer than folds
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
        for fold_num, (_, val_idx) in enumerate(kf.split(dev_df), start=1):
            original_indices = dev_df.loc[val_idx, "index"].values
            df.loc[original_indices, "fold"] = str(fold_num)
            
    manifest_cols = ["issue_key", "project_id", "split", "fold", "duplicate_group"]
    manifest = df[manifest_cols].copy()
    return manifest

def main():
    parser = argparse.ArgumentParser(description="Create train/val/test splits manifest")
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--output_base", default="outputs")
    parser.add_argument("--allocation_path", default="project_allocation.csv")
    parser.add_argument("--member", default="devanarayanan")
    parser.add_argument("--project_key", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--test_fraction", type=float, default=0.20)
    args = parser.parse_args()
    
    alloc_df = pd.read_csv(args.allocation_path)
    if args.member:
        alloc_df = alloc_df[alloc_df["assigned_member"] == args.member]
    if args.project_key:
        alloc_df = alloc_df[alloc_df["project_key"] == args.project_key]
        
    for _, row in alloc_df.iterrows():
        p_key = str(row["project_key"])
        p_id = int(row["project_id"])
        labeled_clean_path = Path(args.processed_dir) / p_key / "labeled_clean.csv"
        if not labeled_clean_path.exists():
            print(f"Skipping {p_key}: {labeled_clean_path} does not exist. Run preprocess.py first.")
            continue
            
        df_labeled = pd.read_csv(labeled_clean_path)
        manifest = create_splits(df_labeled, random_seed=args.seed, n_folds=args.folds, test_fraction=args.test_fraction)
        
        # Save to outputs/<member>/<project_key>/split_manifest.csv
        out_dir = Path(args.output_base) / args.member / p_key
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(out_dir / "split_manifest.csv", index=False)
        
        # Also copy to data/processed/<project_key>/split_manifest.csv for reference
        manifest.to_csv(Path(args.processed_dir) / p_key / "split_manifest.csv", index=False)
        
        dev_count = (manifest["split"] == "development").sum()
        test_count = (manifest["split"] == "test").sum()
        print(f"[{p_key}] Split created: {dev_count} dev, {test_count} test (seed={args.seed}) -> {out_dir / 'split_manifest.csv'}")

if __name__ == "__main__":
    main()
