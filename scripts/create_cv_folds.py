#!/usr/bin/env python3

from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import GroupKFold


BASE_DIR = Path(__file__).resolve().parents[1]


def main():
    with open(BASE_DIR / "configs/problem5.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_cfg = config["data"]
    n_splits = int(config["experiment"]["validation_folds"])

    processed_dir = BASE_DIR / data_cfg["processed_dir"]
    splits_dir = BASE_DIR / data_cfg["splits_dir"]

    df = pd.read_csv(
        processed_dir / "labeled_clean.csv",
        low_memory=False,
    )
    manifest = pd.read_csv(
        splits_dir / "split_manifest.csv",
        low_memory=False,
    )

    issue_key = data_cfg["issue_key_column"]
    project_col = data_cfg["project_column"]

    # Join the split assignment back to the model input text.
    df = df.merge(
        manifest[[issue_key, "split"]],
        on=issue_key,
        how="left",
        validate="one_to_one",
    )

    development = df[df["split"] == "development"].copy()

    if development["split"].isna().any():
        raise ValueError("Some development rows have no split assignment.")

    development["cv_fold"] = -1

    # Assign folds separately within each project.
    for project_id, project_df in development.groupby(
        project_col, sort=True
    ):
        unique_groups = project_df["input_text"].nunique()

        if unique_groups < n_splits:
            raise ValueError(
                f"Project {project_id} has only {unique_groups} "
                f"unique inputs; cannot make {n_splits} folds."
            )

        splitter = GroupKFold(n_splits=n_splits)

        for fold, (_, validation_positions) in enumerate(
            splitter.split(
                project_df,
                groups=project_df["input_text"].astype(str),
            ),
            start=0,
        ):
            validation_indices = project_df.iloc[
                validation_positions
            ].index
            development.loc[
                validation_indices, "cv_fold"
            ] = fold

    if (development["cv_fold"] < 0).any():
        raise RuntimeError("Some development rows were not assigned a fold.")

    # Verify identical text within a project occurs in only one fold.
    group_fold_counts = development.groupby(
        [project_col, "input_text"]
    )["cv_fold"].nunique()

    if (group_fold_counts > 1).any():
        raise RuntimeError("Duplicate text crosses CV folds.")

    output_columns = [
        issue_key,
        project_col,
        data_cfg["target_column"],
        "cv_fold",
    ]

    output_path = splits_dir / "development_cv_folds.csv"
    development[output_columns].to_csv(output_path, index=False)

    print("Development rows:", len(development))
    print("Number of folds:", n_splits)
    print("\nRows per fold:")
    print(
        development.groupby("cv_fold")
        .size()
        .sort_index()
        .to_string()
    )
    print("\nPer-project fold counts:")
    print(
        development.groupby([project_col, "cv_fold"])
        .size()
        .unstack(fill_value=0)
        .to_string()
    )
    print("\nSaved:", output_path)


if __name__ == "__main__":
    main()
