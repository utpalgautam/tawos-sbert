#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

BASE_DIR = Path(__file__).resolve().parents[1]


def main():
    with open(
        BASE_DIR / "configs/problem5.yaml",
        encoding="utf-8",
    ) as f:
        config = yaml.safe_load(f)

    data_cfg = config["data"]

    data_path = (
        BASE_DIR
        / data_cfg["processed_dir"]
        / "labeled_clean.csv"
    )

    df = pd.read_csv(data_path, low_memory=False)

    project_col = data_cfg["project_column"]
    target_col = data_cfg["target_column"]

    print("OVERALL STORY-POINT DISTRIBUTION")
    print("=" * 75)

    print(df[target_col].describe(
        percentiles=[0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    ).to_string())

    print("\nMost frequent story-point values:")
    print(
        df[target_col]
        .value_counts()
        .head(20)
        .rename_axis("story_point")
        .to_string()
    )

    print("\nPER-PROJECT TARGET DISTRIBUTIONS")
    print("=" * 110)

    rows = []

    for project_id, group in df.groupby(project_col, sort=True):
        y = group[target_col].astype(float)

        frequencies = y.value_counts(normalize=True)
        top_value = y.mode().iloc[0]

        rows.append({
            "project_id": project_id,
            "n": len(y),
            "unique_targets": y.nunique(),
            "mean": y.mean(),
            "median": y.median(),
            "std": y.std(),
            "q75": y.quantile(0.75),
            "q90": y.quantile(0.90),
            "q95": y.quantile(0.95),
            "q99": y.quantile(0.99),
            "max": y.max(),
            "zero_fraction": (y == 0).mean(),
            "mode": top_value,
            "mode_fraction": frequencies.iloc[0],
        })

    summary = pd.DataFrame(rows)

    print(summary.round(4).to_string(index=False))

    output_path = (
        BASE_DIR
        / config["output"]["results_dir"]
        / "target_distribution_by_project.csv"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)

    print("\nSaved:", output_path)


if __name__ == "__main__":
    main()