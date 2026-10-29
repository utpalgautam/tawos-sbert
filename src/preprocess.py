#!/usr/bin/env python3
"""
Preprocess module for TAWOS SBERT + CNN Pipeline.
Implements Steps 2 & 3 of COMMON_PIPELINE.md:
  - Text cleaning and common input construction:
      Title: <clean title>
      Description: <clean description>
  - Data-cleaning policy:
      - Preserve raw CSV
      - Strip surrounding extraction quotes if present
      - Keep title and description columns
      - Numeric story points conversion
      - Document all exclusions in exclusion_log.csv
      - Record review flags in review_flags.csv
      - Group duplicate input texts with duplicate_group
      - Save labeled_clean.csv and unlabeled.csv per project
"""

import argparse
import os
from pathlib import Path
import pandas as pd
import numpy as np
import yaml

def clean_text(value):
    """Clean individual text field as specified in Section 5."""
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1]
    return " ".join(text.split())

def build_input_text(title, description):
    """Build standardized model input text as specified in Section 5."""
    c_title = clean_text(title)
    c_desc = clean_text(description)
    if c_title and c_desc:
        return f"Title: {c_title}\nDescription: {c_desc}"
    if c_title:
        return f"Title: {c_title}"
    if c_desc:
        return f"Description: {c_desc}"
    return ""

def preprocess_project(df_project: pd.DataFrame, project_key: str, project_id: int, out_dir: Path):
    """
    Preprocess data for a single project.
    Generates:
      - labeled_clean.csv
      - unlabeled.csv
      - review_flags.csv
      - exclusion_log.csv
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Clean text fields for all rows
    df = df_project.copy()
    df["title_clean"] = df["title"].apply(clean_text)
    df["desc_clean"] = df["description_text"].apply(clean_text)
    df["input_text"] = [
        build_input_text(t, d) for t, d in zip(df["title"], df["description_text"])
    ]
    
    # 2. Separate labeled vs unlabeled
    labeled_mask = df["story_point"].notna()
    df_labeled = df[labeled_mask].copy()
    df_unlabeled = df[~labeled_mask].copy()
    
    exclusion_records = []
    review_flags = []
    
    # Check duplicate issue_keys in labeled
    dup_keys = df_labeled[df_labeled.duplicated(subset=["issue_key"], keep=False)]
    for idx, row in dup_keys.iterrows():
        review_flags.append({
            "issue_key": row["issue_key"],
            "project_id": project_id,
            "flag_reason": "duplicate_issue_key",
            "details": f"Repeated issue_key {row['issue_key']}"
        })
    
    # Deduplicate issue_keys, keeping first, logging excluded
    first_key_mask = ~df_labeled.duplicated(subset=["issue_key"], keep="first")
    for idx, row in df_labeled[~first_key_mask].iterrows():
        exclusion_records.append({
            "issue_key": row["issue_key"],
            "project_id": project_id,
            "original_label": row["story_point"],
            "reason": "duplicate_issue_key",
            "decision": "excluded_keep_first_occurrence"
        })
    df_labeled = df_labeled[first_key_mask].copy()
    
    # Check numeric story point conversion
    df_labeled["sp_numeric"] = pd.to_numeric(df_labeled["story_point"], errors="coerce")
    non_numeric_mask = df_labeled["sp_numeric"].isna()
    for idx, row in df_labeled[non_numeric_mask].iterrows():
        exclusion_records.append({
            "issue_key": row["issue_key"],
            "project_id": project_id,
            "original_label": row["story_point"],
            "reason": "non_numeric_story_point",
            "decision": "excluded_cannot_train_regression"
        })
    df_labeled = df_labeled[~non_numeric_mask].copy()
    
    # Check empty text (both title and description empty)
    empty_text_mask = df_labeled["input_text"] == ""
    for idx, row in df_labeled[empty_text_mask].iterrows():
        exclusion_records.append({
            "issue_key": row["issue_key"],
            "project_id": project_id,
            "original_label": row["story_point"],
            "reason": "empty_title_and_description",
            "decision": "excluded_no_text_available"
        })
    df_labeled = df_labeled[~empty_text_mask].copy()
    
    # Check review flags on remaining labeled issues
    for idx, row in df_labeled.iterrows():
        sp = row["sp_numeric"]
        ik = row["issue_key"]
        if sp == 0:
            review_flags.append({
                "issue_key": ik,
                "project_id": project_id,
                "flag_reason": "zero_story_point",
                "details": "Label is 0"
            })
        elif sp != int(sp):
            review_flags.append({
                "issue_key": ik,
                "project_id": project_id,
                "flag_reason": "fractional_story_point",
                "details": f"Label is fractional: {sp}"
            })
        if sp > 100:
            review_flags.append({
                "issue_key": ik,
                "project_id": project_id,
                "flag_reason": "extreme_story_point",
                "details": f"Label exceeds 100: {sp}"
            })
        if row["title_clean"] == "":
            review_flags.append({
                "issue_key": ik,
                "project_id": project_id,
                "flag_reason": "description_only",
                "details": "Title is blank, description only"
            })
        elif row["desc_clean"] == "":
            review_flags.append({
                "issue_key": ik,
                "project_id": project_id,
                "flag_reason": "title_only",
                "details": "Description is blank, title only"
            })
            
    # Assign duplicate_group based on identical input_text within the project
    # Exact duplicate text within project must share the same duplicate_group ID
    text_to_group = {}
    group_id = 1
    dup_groups = []
    for text in df_labeled["input_text"]:
        if text not in text_to_group:
            text_to_group[text] = group_id
            group_id += 1
        dup_groups.append(text_to_group[text])
    df_labeled["duplicate_group"] = dup_groups
    
    # Check duplicate text with conflicting labels
    grouped = df_labeled.groupby("duplicate_group")["sp_numeric"].nunique()
    conflicting_groups = set(grouped[grouped > 1].index)
    if conflicting_groups:
        for idx, row in df_labeled[df_labeled["duplicate_group"].isin(conflicting_groups)].iterrows():
            review_flags.append({
                "issue_key": row["issue_key"],
                "project_id": project_id,
                "flag_reason": "duplicate_text_conflicting_labels",
                "details": f"duplicate_group {row['duplicate_group']} has conflicting story_points"
            })

    # Prepare final columns for labeled_clean
    df_labeled["story_point"] = df_labeled["sp_numeric"]
    keep_cols_labeled = [
        "issue_id", "issue_key", "project_id", "title", "description_text",
        "input_text", "story_point", "issue_type", "priority", "status",
        "duplicate_group"
    ]
    # Keep whichever exist in df
    keep_cols_labeled = [c for c in keep_cols_labeled if c in df_labeled.columns]
    labeled_clean = df_labeled[keep_cols_labeled].copy()
    
    # Prepare unlabeled
    keep_cols_unlabeled = [
        "issue_id", "issue_key", "project_id", "title", "description_text",
        "input_text", "issue_type", "priority", "status"
    ]
    keep_cols_unlabeled = [c for c in keep_cols_unlabeled if c in df_unlabeled.columns]
    unlabeled_clean = df_unlabeled[keep_cols_unlabeled].copy()
    
    # Save CSVs
    labeled_clean.to_csv(out_dir / "labeled_clean.csv", index=False)
    unlabeled_clean.to_csv(out_dir / "unlabeled.csv", index=False)
    
    df_exclusions = pd.DataFrame(exclusion_records)
    if df_exclusions.empty:
        df_exclusions = pd.DataFrame(columns=["issue_key", "project_id", "original_label", "reason", "decision"])
    df_exclusions.to_csv(out_dir / "exclusion_log.csv", index=False)
    
    df_flags = pd.DataFrame(review_flags)
    if df_flags.empty:
        df_flags = pd.DataFrame(columns=["issue_key", "project_id", "flag_reason", "details"])
    df_flags.to_csv(out_dir / "review_flags.csv", index=False)
    
    print(f"[{project_key}] Preprocessed: {len(labeled_clean)} labeled clean, {len(unlabeled_clean)} unlabeled, "
          f"{len(df_exclusions)} excluded, {len(df_flags)} review flags.")
    return {
        "labeled_clean_count": len(labeled_clean),
        "unlabeled_count": len(unlabeled_clean),
        "excluded_count": len(df_exclusions),
        "review_flags_count": len(df_flags)
    }

def main():
    parser = argparse.ArgumentParser(description="Preprocess project data for TAWOS SBERT pipeline")
    parser.add_argument("--config", default="configs/problem5_sbert_cnn.yaml")
    parser.add_argument("--raw_path", default="data/raw/tawos.csv")
    parser.add_argument("--allocation_path", default="project_allocation.csv")
    parser.add_argument("--member", default="devanarayanan")
    parser.add_argument("--project_key", default=None, help="Specific project_key to process (optional)")
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
    
    for _, row in alloc_df.iterrows():
        p_id = int(row["project_id"])
        p_key = str(row["project_key"])
        print(f"\nProcessing project {p_key} (ID: {p_id})...")
        p_df = df_raw[df_raw["project_id"] == p_id].copy()
        out_dir = Path("data/processed") / p_key
        preprocess_project(p_df, p_key, p_id, out_dir)

if __name__ == "__main__":
    main()
