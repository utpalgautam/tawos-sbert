#!/usr/bin/env python3
"""
Full Pipeline Runner for TAWOS SBERT + CNN within-project experiments.
Executes the full pipeline for assigned projects:
  1. Profile raw project data
  2. Preprocess text, apply exclusions, group duplicates
  3. Generate 80/20 train/test split manifest & 5-fold CV
  4. Precompute frozen SBERT embeddings (sentence-transformers/all-MiniLM-L6-v2)
  5. 5-fold CV training with early stopping, determine median best epoch, refit final model on full dev
  6. Evaluate on locked test set, compute training-median baseline, log metrics
  7. Run inference on unlabeled issues
  8. Generate markdown summary report
"""

import argparse
import json
import time
from pathlib import Path
import pandas as pd
import numpy as np

from src.inspect_data import profile_project
from src.preprocess import preprocess_project
from src.make_splits import create_splits
from src.embed_sbert import embed_project_data
from src.train_cv import run_cv_and_final_fit
from src.evaluate import evaluate_project
from src.predict_unlabeled import predict_unlabeled_project

def generate_report(project_key: str, project_id: int, out_dir: Path, proc_dir: Path):
    """Generate a clean markdown report for the project matching Section 12."""
    profile_path = out_dir / "project_profile.json"
    test_metrics_path = out_dir / "test_metrics.json"
    cv_summary_path = out_dir / "cv_summary.csv"
    cv_fold_path = out_dir / "cv_fold_metrics.csv"
    exclusion_path = proc_dir / "exclusion_log.csv"
    review_flags_path = proc_dir / "review_flags.csv"
    
    profile = {}
    if profile_path.exists():
        with open(profile_path) as f:
            profile = json.load(f)
            
    test_metrics = {}
    if test_metrics_path.exists():
        with open(test_metrics_path) as f:
            test_metrics = json.load(f)
            
    cv_summary = {}
    if cv_summary_path.exists():
        cv_summary = pd.read_csv(cv_summary_path).iloc[0].to_dict()
        
    df_cv_folds = pd.read_csv(cv_fold_path) if cv_fold_path.exists() else pd.DataFrame()
    df_exclusions = pd.read_csv(exclusion_path) if exclusion_path.exists() else pd.DataFrame()
    df_flags = pd.read_csv(review_flags_path) if review_flags_path.exists() else pd.DataFrame()
    
    report_content = f"""# Project Report: {project_key} (ID: {project_id})
**Experiment:** Problem 5 — SBERT + 3-Layer CNN (Within-Project, Random Split)  
**Assigned Member:** devanarayanan  
**Generated At:** {time.strftime("%Y-%m-%d %H:%M:%S")}

---

## 1. Project Profile & Data Summary
- **Total Issues:** {profile.get("total_issues", "N/A")}
- **Labeled Issues (Raw):** {profile.get("labeled_issues", "N/A")}
- **Unlabeled Issues (Raw):** {profile.get("unlabeled_issues", "N/A")}
- **Excluded Issues:** {len(df_exclusions)}
- **Review Flags Logged:** {len(df_flags)}
- **Story Point Statistics (Raw Labeled):**
  - Min: {profile.get('story_point_stats', {}).get('min', 'N/A')}
  - 25th percentile: {profile.get('story_point_stats', {}).get('p25', 'N/A')}
  - Median: {profile.get('story_point_stats', {}).get('median', 'N/A')}
  - 75th percentile: {profile.get('story_point_stats', {}).get('p75', 'N/A')}
  - Mean: {profile.get('story_point_stats', {}).get('mean', 'N/A')}
  - Max: {profile.get('story_point_stats', {}).get('max', 'N/A')}

---

## 2. Cross-Validation Results (5 Folds on Development Set)
- **Mean Validation MAE:** {cv_summary.get('mean_mae', 'N/A'):.4f} (± {cv_summary.get('std_mae', 0.0):.4f})
- **Mean Validation MdAE:** {cv_summary.get('mean_mdae', 'N/A'):.4f} (± {cv_summary.get('std_mdae', 0.0):.4f})
- **Median Best Epoch:** {cv_summary.get('median_best_epoch', 'N/A')}

### Fold Details:
| Fold | Best Epoch | Train Count | Val Count | Val MAE | Val MdAE |
|------|------------|-------------|-----------|---------|----------|
"""
    for _, r in df_cv_folds.iterrows():
        report_content += f"| {int(r['fold'])} | {int(r['best_epoch'])} | {int(r['train_count'])} | {int(r['validation_count'])} | {r['mae']:.4f} | {r['mdae']:.4f} |\n"
        
    report_content += f"""
---

## 3. Held-Out Test Evaluation (20% Locked Test Set)
- **Test Sample Count:** {test_metrics.get("test_count", "N/A")}
- **CNN Model MAE:** {test_metrics.get("mae", "N/A"):.4f}
- **Baseline MAE (Training Median):** {test_metrics.get("baseline_mae", "N/A"):.4f}
- **CNN Model MdAE:** {test_metrics.get("mdae", "N/A"):.4f}
- **Baseline MdAE (Training Median):** {test_metrics.get("baseline_mdae", "N/A"):.4f}
- **Training Median Value:** {test_metrics.get("training_median_value", "N/A")}

---

## 4. Notes and Limitations
1. In accordance with the project guidelines, the SBERT encoder (`sentence-transformers/all-MiniLM-L6-v2`) remained frozen.
2. The CNN regression head was trained on the full development set for exactly the median best epoch determined during 5-fold cross-validation.
3. Test data was never used for early stopping or hyperparameter tuning.
4. Duplicate text was isolated to avoid data leakage between splits.
"""
    with open(out_dir / "report.md", "w") as f:
        f.write(report_content)

def run_pipeline_for_project(
    project_key: str,
    project_id: int,
    df_raw: pd.DataFrame,
    base_proc_dir: Path,
    base_out_dir: Path,
    member: str = "devanarayanan",
    seed: int = 42,
    batch_size: int = 64
):
    print(f"\n{'='*70}\n[START] Pipeline for {project_key} (ID: {project_id})\n{'='*70}")
    t0 = time.time()
    
    p_proc_dir = base_proc_dir / project_key
    p_out_dir = base_out_dir / member / project_key
    p_proc_dir.mkdir(parents=True, exist_ok=True)
    p_out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Profile
    print(f"\n--- Step 1: Profiling {project_key} ---")
    df_p = df_raw[df_raw["project_id"] == project_id].copy()
    profile_project(df_p, project_key, project_id, out_dir=p_out_dir)
    
    # 2. Preprocess
    print(f"\n--- Step 2: Preprocessing {project_key} ---")
    preprocess_project(df_p, project_key, project_id, out_dir=p_proc_dir)
    # Also copy flags and exclusions to outputs directory
    for fname in ["review_flags.csv", "exclusion_log.csv"]:
        if (p_proc_dir / fname).exists():
            import shutil
            shutil.copy(p_proc_dir / fname, p_out_dir / fname)
            
    # 3. Splits
    print(f"\n--- Step 3: Making Splits for {project_key} ---")
    labeled_clean_df = pd.read_csv(p_proc_dir / "labeled_clean.csv")
    manifest = create_splits(labeled_clean_df, random_seed=seed, n_folds=5, test_fraction=0.20)
    manifest.to_csv(p_out_dir / "split_manifest.csv", index=False)
    manifest.to_csv(p_proc_dir / "split_manifest.csv", index=False)
    
    # 4. SBERT Embeddings
    print(f"\n--- Step 4: SBERT Embeddings for {project_key} ---")
    embed_project_data(project_key, project_id, p_proc_dir, p_out_dir, batch_size=batch_size)
    
    # 5. Train CV & Fit Final Model
    print(f"\n--- Step 5: 5-Fold Cross Validation & Final Refit for {project_key} ---")
    run_cv_and_final_fit(
        project_key=project_key,
        project_id=project_id,
        manifest_path=p_out_dir / "split_manifest.csv",
        embeddings_path=p_out_dir / "embeddings_labeled.pt",
        output_dir=p_out_dir,
        seed=seed,
        batch_size=batch_size
    )
    
    # 6. Evaluate on Test Set
    print(f"\n--- Step 6: Test Set Evaluation for {project_key} ---")
    evaluate_project(
        project_key=project_key,
        project_id=project_id,
        manifest_path=p_out_dir / "split_manifest.csv",
        embeddings_path=p_out_dir / "embeddings_labeled.pt",
        model_path=p_out_dir / "final_model.pt",
        output_dir=p_out_dir,
        seed=seed
    )
    
    # 7. Predict on Unlabeled Issues
    print(f"\n--- Step 7: Unlabeled Prediction for {project_key} ---")
    predict_unlabeled_project(
        project_key=project_key,
        project_id=project_id,
        processed_dir=p_proc_dir,
        output_dir=p_out_dir,
        batch_size=batch_size
    )
    
    # 8. Report
    print(f"\n--- Step 8: Generating Report for {project_key} ---")
    generate_report(project_key, project_id, p_out_dir, p_proc_dir)
    
    elapsed = time.time() - t0
    print(f"\n[DONE] Pipeline for {project_key} completed in {elapsed:.1f}s!\n{'='*70}\n")

def main():
    parser = argparse.ArgumentParser(description="End-to-end TAWOS Problem 5 Pipeline Runner")
    parser.add_argument("--raw_path", default="data/raw/tawos.csv")
    parser.add_argument("--allocation_path", default="project_allocation.csv")
    parser.add_argument("--member", default="devanarayanan")
    parser.add_argument("--project_key", default=None, help="Run only specific project_key")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    
    alloc_df = pd.read_csv(args.allocation_path)
    if args.member:
        alloc_df = alloc_df[alloc_df["assigned_member"] == args.member]
    if args.project_key:
        alloc_df = alloc_df[alloc_df["project_key"] == args.project_key]
        
    print(f"Loading raw dataset from {args.raw_path}...")
    df_raw = pd.read_csv(args.raw_path)
    print(f"Loaded {len(df_raw):,} rows from raw data.")
    
    base_proc_dir = Path("data/processed")
    base_out_dir = Path("outputs")
    
    for _, row in alloc_df.iterrows():
        p_id = int(row["project_id"])
        p_key = str(row["project_key"])
        run_pipeline_for_project(
            project_key=p_key,
            project_id=p_id,
            df_raw=df_raw,
            base_proc_dir=base_proc_dir,
            base_out_dir=base_out_dir,
            member=args.member,
            seed=args.seed,
            batch_size=args.batch_size
        )

if __name__ == "__main__":
    main()
