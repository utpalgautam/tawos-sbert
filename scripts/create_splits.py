#!/usr/bin/env python3

from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import GroupShuffleSplit


BASE_DIR = Path(__file__).resolve().parents[1]


def main():
    with open(BASE_DIR / "configs/problem5.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    exp = config["experiment"]
    data_cfg = config["data"]

    input_path = BASE_DIR / data_cfg["processed_dir"] / "labeled_clean.csv"
    splits_dir = BASE_DIR / data_cfg["splits_dir"]
    splits_dir.mkdir(parents=True, exist_ok=True)

    project_col = data_cfg["project_column"]
    issue_key_col = data_cfg["issue_key_column"]
    target_col = data_cfg["target_column"]

    development_fraction = float(exp["development_fraction"])
    test_fraction = float(exp["test_fraction"])
    seed = int(exp["random_seed"])

    if abs(development_fraction + test_fraction - 1.0) > 1e-9:
        raise ValueError("Development and test fractions must sum to 1.")

    df = pd.read_csv(input_path, low_memory=False)

    required = {
        project_col,
        issue_key_col,
        target_col,
        "input_text",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if df[issue_key_col].duplicated().any():
        duplicates = df.loc[
            df[issue_key_col].duplicated(keep=False), issue_key_col
        ].tolist()
        raise ValueError(
            "Duplicate issue keys found. Resolve them before splitting. "
            f"Examples: {duplicates[:10]}"
        )

    if df["input_text"].fillna("").str.strip().eq("").any():
        raise ValueError("Empty input_text found in labeled dataset.")

    df["split"] = ""
    split_summaries = []

    # Group identical text within each project so that repeated inputs
    # cannot leak across development and test partitions.
    for project_id, project_df in df.groupby(project_col, sort=True):
        group_labels = project_df["input_text"].fillna("").astype(str)

        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=test_fraction,
            random_state=seed,
        )

        dev_positions, test_positions = next(
            splitter.split(
                project_df,
                groups=group_labels,
            )
        )

        dev_indices = project_df.iloc[dev_positions].index
        test_indices = project_df.iloc[test_positions].index

        df.loc[dev_indices, "split"] = "development"
        df.loc[test_indices, "split"] = "test"

        dev_text = set(df.loc[dev_indices, "input_text"].astype(str))
        test_text = set(df.loc[test_indices, "input_text"].astype(str))

        overlap = dev_text.intersection(test_text)
        if overlap:
            raise RuntimeError(
                f"Text leakage detected in project {project_id}."
            )

        split_summaries.append({
            "project_id": int(project_id),
            "total_rows": int(len(project_df)),
            "development_rows": int(len(dev_indices)),
            "test_rows": int(len(test_indices)),
            "development_fraction_actual": len(dev_indices) / len(project_df),
            "test_fraction_actual": len(test_indices) / len(project_df),
        })

    if df["split"].eq("").any():
        raise RuntimeError("Some rows were not assigned to a split.")

    manifest_columns = [
        issue_key_col,
        project_col,
        target_col,
        "split",
    ]

    manifest_path = splits_dir / "split_manifest.csv"
    df[manifest_columns].to_csv(manifest_path, index=False)

    summary = pd.DataFrame(split_summaries)
    summary_path = splits_dir / "split_summary.csv"
    summary.to_csv(summary_path, index=False)

    dev_df = df[df["split"] == "development"]
    test_df = df[df["split"] == "test"]

    # Verify no repeated text crosses partitions within a project.
    dev_pairs = set(zip(
        dev_df[project_col],
        dev_df["input_text"].astype(str),
    ))
    test_pairs = set(zip(
        test_df[project_col],
        test_df["input_text"].astype(str),
    ))

    if dev_pairs.intersection(test_pairs):
        raise RuntimeError("Cross-partition duplicate text detected.")

    print("\n=== SPLIT CREATION COMPLETE ===")
    print("Input rows:", len(df))
    print("Development rows:", len(dev_df))
    print("Test rows:", len(test_df))
    print("Actual development fraction:", round(len(dev_df) / len(df), 4))
    print("Actual test fraction:", round(len(test_df) / len(df), 4))
    print("\nPer-project summary:")
    print(summary.to_string(index=False))
    print("\nSaved:")
    print(manifest_path)
    print(summary_path)


if __name__ == "__main__":
    main()
