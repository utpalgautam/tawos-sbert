#!/usr/bin/env python3
"""
Inspect Data Module for TAWOS SBERT + CNN Pipeline.
Implements Step 1 (Project-level profiling) of COMMON_PIPELINE.md:
  - Filter using project_id
  - Count total, labeled, and unlabeled rows
  - Check missing/blank titles and descriptions
  - Inspect issue types and story-point distribution
  - Flag zero, fractional, extreme (>100), and non-numeric labels
  - Check duplicate issue keys, repeated text, conflicting labels
  - Save profile before making exclusions
"""

import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np

def profile_project(df_project: pd.DataFrame, project_key: str, project_id: int, out_dir: Path = None):
    p = df_project.copy()
    total_issues = len(p)
    labeled_mask = p["story_point"].notna()
    labeled = p[labeled_mask].copy()
    unlabeled_count = int((~labeled_mask).sum())
    labeled_count = int(labeled_mask.sum())
    
    missing_titles = int(p["title"].isna().sum())
    missing_descriptions = int(p["description_text"].isna().sum())
    blank_titles = int((p["title"].fillna("").astype(str).str.strip() == "").sum())
    blank_descriptions = int((p["description_text"].fillna("").astype(str).str.strip() == "").sum())
    both_blank = int((
        (p["title"].fillna("").astype(str).str.strip() == "") & 
        (p["description_text"].fillna("").astype(str).str.strip() == "")
    ).sum())
    
    # Story-point label analysis
    sp_series = pd.to_numeric(labeled["story_point"], errors="coerce")
    non_numeric_count = int(sp_series.isna().sum())
    valid_sp = sp_series.dropna()
    
    zero_count = int((valid_sp == 0).sum())
    fractional_count = int((valid_sp.apply(lambda x: x != int(x))).sum())
    extreme_count = int((valid_sp > 100).sum())
    negative_count = int((valid_sp < 0).sum())
    
    # Stats on valid story points
    sp_stats = {}
    if not valid_sp.empty:
        sp_stats = {
            "count": int(len(valid_sp)),
            "mean": float(valid_sp.mean()),
            "std": float(valid_sp.std()) if len(valid_sp) > 1 else 0.0,
            "min": float(valid_sp.min()),
            "p25": float(valid_sp.quantile(0.25)),
            "median": float(valid_sp.median()),
            "p75": float(valid_sp.quantile(0.75)),
            "p90": float(valid_sp.quantile(0.90)),
            "p95": float(valid_sp.quantile(0.95)),
            "p99": float(valid_sp.quantile(0.99)),
            "max": float(valid_sp.max()),
            "distinct_values": int(valid_sp.nunique())
        }
        
    # Duplicates in project
    dup_issue_keys = int(p["issue_key"].duplicated().sum())
    
    # Clean text to evaluate duplicate content
    norm_title = p["title"].fillna("").astype(str).str.strip().str.lower()
    norm_desc = p["description_text"].fillna("").astype(str).str.strip().str.lower()
    full_text = norm_title + " ||| " + norm_desc
    dup_full_text = int(full_text.duplicated().sum())
    
    # Repeated text with conflicting labels in labeled set
    labeled_full_text = full_text[labeled_mask]
    labeled_with_text = pd.DataFrame({
        "text": labeled_full_text,
        "sp": valid_sp
    })
    grouped_labels = labeled_with_text.groupby("text")["sp"].nunique()
    conflicting_text_count = int((grouped_labels > 1).sum())
    
    issue_type_dist = labeled["issue_type"].value_counts(dropna=False).to_dict()
    issue_type_dist = {str(k): int(v) for k, v in issue_type_dist.items()}
    
    profile = {
        "project_key": project_key,
        "project_id": project_id,
        "total_issues": total_issues,
        "labeled_issues": labeled_count,
        "unlabeled_issues": unlabeled_count,
        "missing_titles": missing_titles,
        "blank_titles": blank_titles,
        "missing_descriptions": missing_descriptions,
        "blank_descriptions": blank_descriptions,
        "both_text_blank": both_blank,
        "non_numeric_labels": non_numeric_count,
        "zero_labels": zero_count,
        "fractional_labels": fractional_count,
        "extreme_labels_gt_100": extreme_count,
        "negative_labels": negative_count,
        "duplicate_issue_keys": dup_issue_keys,
        "duplicate_full_text": dup_full_text,
        "conflicting_labels_same_text": conflicting_text_count,
        "story_point_stats": sp_stats,
        "issue_type_distribution": issue_type_dist
    }
    
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "project_profile.json", "w") as f:
            json.dump(profile, f, indent=2)
            
    return profile

def main():
    parser = argparse.ArgumentParser(description="Profile projects in TAWOS dataset")
    parser.add_argument("--raw_path", default="data/raw/tawos.csv")
    parser.add_argument("--allocation_path", default="project_allocation.csv")
    parser.add_argument("--member", default="devanarayanan")
    parser.add_argument("--project_key", default=None)
    parser.add_argument("--output_base", default="outputs")
    args = parser.parse_args()
    
    raw_path = Path(args.raw_path)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data not found at {raw_path}")
        
    alloc_df = pd.read_csv(args.allocation_path)
    if args.member:
        alloc_df = alloc_df[alloc_df["assigned_member"] == args.member]
    if args.project_key:
        alloc_df = alloc_df[alloc_df["project_key"] == args.project_key]
        
    print(f"Reading raw data from {raw_path}...")
    df_raw = pd.read_csv(raw_path)
    
    all_profiles = []
    for _, row in alloc_df.iterrows():
        p_id = int(row["project_id"])
        p_key = str(row["project_key"])
        p_name = str(row["project_name"])
        print(f"\nProfiling {p_key} ({p_name}, ID: {p_id})...")
        p_df = df_raw[df_raw["project_id"] == p_id].copy()
        out_dir = Path(args.output_base) / args.member / p_key
        prof = profile_project(p_df, p_key, p_id, out_dir)
        all_profiles.append(prof)
        print(f"  Total: {prof['total_issues']} | Labeled: {prof['labeled_issues']} | Unlabeled: {prof['unlabeled_issues']}")
        if prof['story_point_stats']:
            s = prof['story_point_stats']
            print(f"  SP: min={s['min']}, median={s['median']}, mean={s['mean']:.2f}, max={s['max']}")
            
    # Save combined profiling report
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame([{
        "project_id": p["project_id"],
        "project_key": p["project_key"],
        "total_issues": p["total_issues"],
        "labeled_issues": p["labeled_issues"],
        "unlabeled_issues": p["unlabeled_issues"],
        "zero_labels": p["zero_labels"],
        "fractional_labels": p["fractional_labels"],
        "extreme_labels": p["extreme_labels_gt_100"],
        "median_sp": p["story_point_stats"].get("median", np.nan),
        "mean_sp": p["story_point_stats"].get("mean", np.nan),
    } for p in all_profiles])
    summary_df.to_csv(reports_dir / f"profile_{args.member}.csv", index=False)
    print(f"\nSaved profile summary to {reports_dir / f'profile_{args.member}.csv'}")

if __name__ == "__main__":
    main()
