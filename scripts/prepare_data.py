#!/usr/bin/env python3

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


BASE_DIR = Path(__file__).resolve().parents[1]


def clean_text(value):
    """Convert a value to clean, single-line text."""
    if pd.isna(value):
        return ""

    text = str(value).strip()

    # Remove one pair of wrapping quotes if present.
    if (
        len(text) >= 2
        and text[0] == text[-1]
        and text[0] in "\"'"
    ):
        text = text[1:-1].strip()

    return " ".join(text.split())


def build_input_text(title, description):
    """Build the model input using issue text only."""
    title = clean_text(title)
    description = clean_text(description)

    if title and description:
        return f"Title: {title}\nDescription: {description}"
    if title:
        return f"Title: {title}"
    if description:
        return f"Description: {description}"
    return ""


def main():
    config_path = BASE_DIR / "configs" / "problem5.yaml"

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    data_cfg = config["data"]

    raw_path = BASE_DIR / data_cfg["path"]
    processed_dir = BASE_DIR / data_cfg["processed_dir"]
    processed_dir.mkdir(parents=True, exist_ok=True)

    project_col = data_cfg["project_column"]
    issue_key_col = data_cfg["issue_key_column"]
    title_col = data_cfg["title_column"]
    description_col = data_cfg["description_column"]
    target_col = data_cfg["target_column"]

    assigned_projects = set(
        int(x) for x in data_cfg["assigned_project_ids"]
    )
    excluded_issue_keys = set(
        str(x) for x in data_cfg.get("exclude_issue_keys", [])
    )

    required_columns = [
        "issue_id",
        issue_key_col,
        project_col,
        title_col,
        description_col,
        target_col,
    ]

    print(f"Reading raw dataset: {raw_path}")
    print("Loading required columns...")

    df = pd.read_csv(
        raw_path,
        usecols=required_columns,
        low_memory=False,
    )

    raw_rows_read = len(df)

    # Filter to Utpal's assigned projects before preparing the dataset.
    df[project_col] = pd.to_numeric(
        df[project_col], errors="coerce"
    )

    df = df[df[project_col].isin(assigned_projects)].copy()
    assigned_project_rows = len(df)

    # Keep original target values available for auditing.
    original_target = df[target_col].copy()
    numeric_target = pd.to_numeric(original_target, errors="coerce")

    invalid_target = original_target.notna() & numeric_target.isna()
    missing_target = original_target.isna()

    finite_target = (
        numeric_target.notna()
        & np.isfinite(numeric_target.fillna(0))
    )

    nonfinite_target = (
        numeric_target.notna() & ~finite_target
    )

    invalid_or_nonfinite = invalid_target | nonfinite_target

    df[target_col] = numeric_target

    # Prepare the issue text.
    df["title_clean"] = df[title_col].map(clean_text)
    df["description_clean"] = df[description_col].map(clean_text)

    df["input_text"] = [
        build_input_text(title, description)
        for title, description in zip(
            df["title_clean"], df["description_clean"]
        )
    ]

    empty_text = df["input_text"].eq("")

    # A row can be used for supervised learning only if it has
    # a finite numeric target and non-empty input text.
    labeled_mask = (
        ~missing_target
        & ~invalid_or_nonfinite
        & ~empty_text
    )

    unlabeled_mask = missing_target & ~empty_text

    labeled_all = df.loc[labeled_mask].copy()
    unlabeled = df.loc[unlabeled_mask].copy()

    # Apply the documented target transformation.
    fractional_policy = data_cfg.get(
        "fractional_target_policy", "ceil"
    )

    fractional_mask = (
        labeled_all[target_col] % 1 != 0
    )

    fractional_count_before = int(fractional_mask.sum())

    if fractional_policy == "ceil":
        labeled_all[target_col] = np.ceil(
            labeled_all[target_col]
        )
    else:
        raise ValueError(
            f"Unsupported fractional_target_policy: "
            f"{fractional_policy}"
        )

    # Exclude explicitly reviewed issue keys from the primary dataset.
    labeled_issue_keys = (
        labeled_all[issue_key_col].astype(str)
    )
    excluded_mask = labeled_issue_keys.isin(excluded_issue_keys)

    explicitly_excluded = labeled_all.loc[excluded_mask].copy()
    labeled = labeled_all.loc[~excluded_mask].copy()

    # Build the audit report from assigned projects.
    audit_rows = []

    for idx, row in df.iterrows():
        reasons = []
        target = numeric_target.loc[idx]

        if invalid_target.loc[idx]:
            reasons.append("non_numeric_story_point")

        if nonfinite_target.loc[idx]:
            reasons.append("non_finite_story_point")

        if pd.notna(target) and np.isfinite(target):
            if target < 0:
                reasons.append("negative_story_point")

            if target == 0:
                reasons.append("zero_story_point_retained")

            if target != int(target):
                reasons.append("fractional_story_point_rounded_up")

            if target > 100:
                reasons.append("story_point_above_100_review")

        if empty_text.loc[idx]:
            reasons.append("empty_issue_text")

        key = str(row[issue_key_col])

        if key in excluded_issue_keys:
            reasons.append("explicitly_excluded_by_configuration")

        if reasons:
            audit_rows.append({
                issue_key_col: row[issue_key_col],
                project_col: row[project_col],
                "original_story_point": target,
                "reasons": ";".join(reasons),
            })

    audit_df = pd.DataFrame(
        audit_rows,
        columns=[
            issue_key_col,
            project_col,
            "original_story_point",
            "reasons",
        ],
    )

    # Flag repeated text for leakage review; do not automatically
    # remove repeated inputs at this stage.
    duplicate_mask = labeled["input_text"].duplicated(
        keep=False
    )

    duplicate_text = labeled.loc[
        duplicate_mask,
        [
            issue_key_col,
            project_col,
            target_col,
            "input_text",
        ],
    ].copy()

    # Save to the existing filenames.
    labeled_path = processed_dir / "labeled_clean.csv"
    unlabeled_path = processed_dir / "unlabeled.csv"
    audit_path = processed_dir / "data_quality_audit.csv"
    duplicates_path = processed_dir / "repeated_input_text.csv"
    report_path = processed_dir / "data_quality_report.json"

    labeled.to_csv(labeled_path, index=False)
    unlabeled.to_csv(unlabeled_path, index=False)
    audit_df.to_csv(audit_path, index=False)
    duplicate_text.to_csv(duplicates_path, index=False)

    # Calculate final target statistics after transformation and
    # explicit exclusions.
    targets = labeled[target_col].astype(float)

    project_summary = (
        labeled.groupby(project_col)
        .agg(
            labeled_issues=(issue_key_col, "size"),
            unique_issue_keys=(issue_key_col, "nunique"),
        )
        .reset_index()
        .sort_values(project_col)
    )

    project_summary_path = (
        processed_dir / "assigned_project_summary.csv"
    )
    project_summary.to_csv(project_summary_path, index=False)

    report = {
        "raw_path": str(raw_path),
        "rows_read_from_raw_dataset": int(raw_rows_read),
        "assigned_project_ids": sorted(assigned_projects),
        "rows_in_assigned_projects": int(assigned_project_rows),
        "labeled_rows_before_explicit_exclusions": int(
            len(labeled_all)
        ),
        "explicitly_excluded_labeled_rows": int(
            len(explicitly_excluded)
        ),
        "final_labeled_rows": int(len(labeled)),
        "unlabeled_rows_with_text_in_assigned_projects": int(
            len(unlabeled)
        ),
        "missing_story_point_rows_in_assigned_projects": int(
            missing_target.sum()
        ),
        "invalid_or_nonfinite_story_point_rows": int(
            invalid_or_nonfinite.sum()
        ),
        "empty_text_rows_in_assigned_projects": int(
            empty_text.sum()
        ),
        "fractional_targets_rounded_up": fractional_count_before,
        "zero_targets_retained": int((targets == 0).sum()),
        "distinct_projects_in_final_dataset": int(
            labeled[project_col].nunique()
        ),
        "excluded_issue_keys_found": sorted(
            set(explicitly_excluded[issue_key_col].astype(str))
        ),
        "story_point_summary_after_transformation": {
            "minimum": float(targets.min()),
            "median": float(targets.median()),
            "maximum": float(targets.max()),
            "zero_count": int((targets == 0).sum()),
            "negative_count": int((targets < 0).sum()),
            "fractional_count_remaining": int(
                (targets % 1 != 0).sum()
            ),
            "above_100_count": int((targets > 100).sum()),
        },
        "project_counts": [
            {
                "project_id": int(row[project_col]),
                "labeled_issues": int(row["labeled_issues"]),
            }
            for _, row in project_summary.iterrows()
        ],
        "input_text_column": "input_text",
        "embedding_checkpoint": config["sbert"]["checkpoint"],
        "note": (
            "Only assigned projects are included. Fractional labels "
            "are rounded upward; zeros are retained. Explicitly "
            "excluded issues are listed in the configuration. "
            "Repeated inputs are flagged, not removed."
        ),
    }

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, allow_nan=False)

    print("\n=== ASSIGNED-PROJECT PREPROCESSING COMPLETE ===")
    print(json.dumps(report, indent=2))

    print("\nSaved files:")
    for path in [
        labeled_path,
        unlabeled_path,
        audit_path,
        duplicates_path,
        project_summary_path,
        report_path,
    ]:
        print(path)


if __name__ == "__main__":
    main()
