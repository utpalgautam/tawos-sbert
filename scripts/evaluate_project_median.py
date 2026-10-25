#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

BASE_DIR = Path(__file__).resolve().parents[1]


def metrics(actual, predicted):
    errors = np.abs(
        np.asarray(actual, dtype=float)
        - np.asarray(predicted, dtype=float)
    )
    return {
        "MAE": errors.mean(),
        "MdAE": np.median(errors),
    }


def main():
    with open(
        BASE_DIR / "configs/problem5.yaml",
        encoding="utf-8",
    ) as f:
        config = yaml.safe_load(f)

    data_cfg = config["data"]

    issue_col = data_cfg["issue_key_column"]
    project_col = data_cfg["project_column"]
    target_col = data_cfg["target_column"]

    processed_dir = BASE_DIR / data_cfg["processed_dir"]
    splits_dir = BASE_DIR / data_cfg["splits_dir"]
    results_dir = BASE_DIR / config["output"]["results_dir"]

    data = pd.read_csv(
        processed_dir / "labeled_clean.csv",
        low_memory=False,
    )

    manifest = pd.read_csv(
        splits_dir / "split_manifest.csv"
    )

    folds = pd.read_csv(
        splits_dir / "development_cv_folds.csv"
    )

    data = data.merge(
        manifest[[issue_col, "split"]],
        on=issue_col,
        how="left",
        validate="one_to_one",
    )

    data = data.merge(
        folds[[issue_col, "cv_fold"]],
        on=issue_col,
        how="left",
        validate="one_to_one",
    )

    dev = data[data["split"] == "development"].copy()
    dev["cv_fold"] = dev["cv_fold"].astype(int)

    predictions = []

    for fold in sorted(dev["cv_fold"].unique()):
        train = dev[dev["cv_fold"] != fold]
        validation = dev[dev["cv_fold"] == fold]

        for project_id, val_project in validation.groupby(project_col):
            train_project = train[
                train[project_col] == project_id
            ]

            if train_project.empty:
                raise ValueError(
                    f"No training rows for project {project_id}"
                )

            project_median = float(
                train_project[target_col].median()
            )

            for _, row in val_project.iterrows():
                predictions.append({
                    "issue_key": row[issue_col],
                    "project_id": project_id,
                    "actual_story_point": float(row[target_col]),
                    "predicted_story_point": project_median,
                    "fold": fold,
                    "model": "project_median_baseline",
                })

    pred = pd.DataFrame(predictions)

    output_path = results_dir / "cv_project_median_predictions.csv"
    pred.to_csv(output_path, index=False)

    print("PROJECT-SPECIFIC MEDIAN BASELINE")
    print("=" * 65)

    overall = metrics(
        pred["actual_story_point"],
        pred["predicted_story_point"],
    )

    print(
        f"Overall OOF MAE:  {overall['MAE']:.4f}\n"
        f"Overall OOF MdAE: {overall['MdAE']:.4f}"
    )

    print("\nPer-project results:")

    for project_id, group in pred.groupby("project_id"):
        result = metrics(
            group["actual_story_point"],
            group["predicted_story_point"],
        )

        print(
            f"Project {project_id:>2} | "
            f"n={len(group):>5} | "
            f"median predictions={group['predicted_story_point'].nunique():>2} "
            f"distinct | "
            f"MAE={result['MAE']:.4f} | "
            f"MdAE={result['MdAE']:.4f}"
        )

    print("\nComparison with existing CNN:")
    cnn = pd.read_csv(results_dir / "cv_predictions.csv")
    cnn = cnn[cnn["model"] == "sbert_cnn"]

    cnn_result = metrics(
        cnn["actual_story_point"],
        cnn["predicted_story_point"],
    )

    print(
        f"CNN OOF MAE:  {cnn_result['MAE']:.4f}\n"
        f"CNN OOF MdAE: {cnn_result['MdAE']:.4f}"
    )

    print("\nSaved:", output_path)


if __name__ == "__main__":
    main()