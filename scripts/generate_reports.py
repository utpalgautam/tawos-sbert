#!/usr/bin/env python3
"""Generate overall and per-project reports from shared CNN training outputs."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

BASE = Path(__file__).resolve().parents[1]
OVERALL = BASE / "results/cnn/overall"
PROJECTS_OUT = BASE / "results/cnn/projects"
LOGS = BASE / "logs/cnn"

PROJECT_NAMES = {
    2: "DATACASS",
    6: "MXNET",
    10: "DAEMON",
    16: "CWD",
    17: "FE",
    28: "DM",
    31: "CXX",
    32: "JAVA",
}

CNN = "shared_sbert_cnn"
BASELINES = ["global_median_baseline", "project_median_baseline"]


def save_plot(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def calculate_metrics(frame):
    y = frame["actual_story_point"].to_numpy(dtype=float)
    pred = frame["predicted_story_point"].to_numpy(dtype=float)
    err = y - pred
    denominator = np.sum((y - np.mean(y)) ** 2)

    return {
        "n": len(frame),
        "MAE": float(np.mean(np.abs(err))) if len(err) else np.nan,
        "MdAE": float(np.median(np.abs(err))) if len(err) else np.nan,
        "RMSE": float(np.sqrt(np.mean(err ** 2))) if len(err) else np.nan,
        "R2": (
            float(1 - np.sum(err ** 2) / denominator)
            if len(err) > 1 and denominator > 0
            else np.nan
        ),
    }


def plot_actual_predicted(frame, path, title):
    if frame.empty:
        return

    plt.figure(figsize=(7, 6))
    plt.scatter(
        frame["actual_story_point"],
        frame["predicted_story_point"],
        alpha=0.45,
        s=18,
    )
    values = np.concatenate([
        frame["actual_story_point"].to_numpy(dtype=float),
        frame["predicted_story_point"].to_numpy(dtype=float),
    ])
    low, high = float(np.nanmin(values)), float(np.nanmax(values))
    plt.plot([low, high], [low, high], linestyle="--")
    plt.xlabel("Actual story points")
    plt.ylabel("Predicted story points")
    plt.title(title)
    save_plot(path)


def plot_error_distribution(frame, path, title):
    if frame.empty:
        return

    errors = (
        frame["actual_story_point"].to_numpy(dtype=float)
        - frame["predicted_story_point"].to_numpy(dtype=float)
    )
    plt.figure(figsize=(8, 5))
    plt.hist(errors, bins=40)
    plt.axvline(0, linestyle="--")
    plt.xlabel("Actual − predicted story points")
    plt.ylabel("Number of issues")
    plt.title(title)
    save_plot(path)


def plot_model_comparison(metrics_frame, path, title, metric="MAE"):
    if metrics_frame.empty or metric not in metrics_frame.columns:
        return

    frame = metrics_frame.dropna(subset=[metric])
    if frame.empty:
        return

    plt.figure(figsize=(9, 5))
    plt.bar(frame["model"], frame[metric])
    plt.ylabel(metric)
    plt.title(title)
    plt.xticks(rotation=20, ha="right")
    save_plot(path)


def read_csv(name):
    path = OVERALL / name
    if not path.exists():
        raise FileNotFoundError(
            f"Required training output is missing: {path}\n"
            "Run scripts/train_cnn.py before generating reports."
        )
    return pd.read_csv(path)


def main():
    with open(BASE / "configs/problem5.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    assigned_ids = [int(x) for x in config["data"]["assigned_project_ids"]]
    if set(assigned_ids) != set(PROJECT_NAMES):
        raise ValueError(
            f"Configured project IDs {assigned_ids} do not match "
            f"the reporting map {sorted(PROJECT_NAMES)}."
        )

    OVERALL.mkdir(parents=True, exist_ok=True)
    PROJECTS_OUT.mkdir(parents=True, exist_ok=True)

    cv = read_csv("cv_oof_predictions.csv")
    test = read_csv("test_predictions.csv")

    required = {
        "issue_key",
        "project_id",
        "actual_story_point",
        "predicted_story_point",
        "model",
    }
    for label, frame in [("CV predictions", cv), ("Test predictions", test)]:
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{label} missing columns: {sorted(missing)}")

    cv["project_id"] = cv["project_id"].astype(int)
    test["project_id"] = test["project_id"].astype(int)

    unexpected = (
        set(cv["project_id"].unique()) | set(test["project_id"].unique())
    ) - set(assigned_ids)
    if unexpected:
        raise ValueError(f"Unexpected project IDs in predictions: {unexpected}")

    # Save a compact, consistent overall metrics table.
    overall_rows = []
    for split_name, frame in [("cv_oof", cv), ("test", test)]:
        for model, subset in frame.groupby("model"):
            overall_rows.append({
                "split": split_name,
                "model": model,
                **calculate_metrics(subset),
            })

    overall_metrics = pd.DataFrame(overall_rows)
    overall_metrics.to_csv(OVERALL / "overall_metrics.csv", index=False)

    test_metrics = overall_metrics[overall_metrics["split"] == "test"].copy()
    test_metrics.to_csv(OVERALL / "baseline_comparison.csv", index=False)

    cv_metrics = overall_metrics[overall_metrics["split"] == "cv_oof"].copy()
    cv_metrics.to_csv(OVERALL / "cv_metrics.csv", index=False)

    # Generate overall prediction plots using the CNN predictions only.
    cv_cnn = cv[cv["model"] == CNN].copy()
    test_cnn = test[test["model"] == CNN].copy()

    plot_actual_predicted(
        test_cnn,
        OVERALL / "graphs" / "test_actual_vs_predicted.png",
        "Shared CNN: held-out test actual vs predicted",
    )
    plot_error_distribution(
        test_cnn,
        OVERALL / "graphs" / "test_error_distribution.png",
        "Shared CNN: held-out test error distribution",
    )
    plot_actual_predicted(
        cv_cnn,
        OVERALL / "graphs" / "cv_actual_vs_predicted.png",
        "Shared CNN: cross-validation out-of-fold predictions",
    )
    plot_error_distribution(
        cv_cnn,
        OVERALL / "graphs" / "cv_error_distribution.png",
        "Shared CNN: cross-validation error distribution",
    )
    plot_model_comparison(
        test_metrics,
        OVERALL / "graphs" / "test_baseline_comparison_mae.png",
        "Held-out test MAE: CNN vs median baselines",
    )

    # Plot final fit history, if available.
    history_path = LOGS / "final_training_history.csv"
    if history_path.exists():
        history = pd.read_csv(history_path)
        if {"epoch", "train_mae_loss"}.issubset(history.columns):
            plt.figure(figsize=(8, 5))
            plt.plot(history["epoch"], history["train_mae_loss"])
            plt.xlabel("Epoch")
            plt.ylabel("Training MAE loss")
            plt.title("Final shared CNN training history")
            save_plot(OVERALL / "graphs" / "final_training_history.png")

    # Calculate per-project CV and test metrics for every available model.
    project_metric_rows = []
    for split_name, frame in [("cv_oof", cv), ("test", test)]:
        for (pid, model), subset in frame.groupby(["project_id", "model"]):
            project_metric_rows.append({
                "split": split_name,
                "project_id": int(pid),
                "project_key": PROJECT_NAMES[int(pid)],
                "model": model,
                **calculate_metrics(subset),
            })

    project_metrics = pd.DataFrame(project_metric_rows)
    project_metrics.to_csv(
        OVERALL / "all_project_metrics.csv",
        index=False,
    )

    # Per-project folders: predictions, metrics, and plots.
    for pid in assigned_ids:
        project_key = PROJECT_NAMES[pid]
        folder = PROJECTS_OUT / project_key
        graphs = folder / "graphs"
        graphs.mkdir(parents=True, exist_ok=True)

        project_cv = cv[cv["project_id"] == pid].copy()
        project_test = test[test["project_id"] == pid].copy()

        project_cv.to_csv(folder / "cv_predictions.csv", index=False)
        project_test.to_csv(folder / "test_predictions.csv", index=False)

        rows = []
        for split_name, frame in [
            ("cv_oof", project_cv),
            ("test", project_test),
        ]:
            for model, subset in frame.groupby("model"):
                rows.append({
                    "split": split_name,
                    "project_id": pid,
                    "project_key": project_key,
                    "model": model,
                    **calculate_metrics(subset),
                })

        metrics_frame = pd.DataFrame(rows)
        metrics_frame.to_csv(folder / "metrics.csv", index=False)

        cnn_test = project_test[project_test["model"] == CNN]
        cnn_cv = project_cv[project_cv["model"] == CNN]

        plot_actual_predicted(
            cnn_test,
            graphs / "test_actual_vs_predicted.png",
            f"{project_key}: test actual vs predicted",
        )
        plot_error_distribution(
            cnn_test,
            graphs / "test_error_distribution.png",
            f"{project_key}: test error distribution",
        )
        plot_actual_predicted(
            cnn_cv,
            graphs / "cv_actual_vs_predicted.png",
            f"{project_key}: CV actual vs predicted",
        )

        test_project_metrics = metrics_frame[
            metrics_frame["split"] == "test"
        ]
        plot_model_comparison(
            test_project_metrics,
            graphs / "test_baseline_comparison_mae.png",
            f"{project_key}: test MAE comparison",
        )

    # One overview comparing CNN test MAE across the eight projects.
    cnn_project_test = project_metrics[
        (project_metrics["split"] == "test")
        & (project_metrics["model"] == CNN)
    ].copy()

    if not cnn_project_test.empty:
        cnn_project_test = cnn_project_test.sort_values("project_id")
        plt.figure(figsize=(9, 5))
        plt.bar(cnn_project_test["project_key"], cnn_project_test["MAE"])
        plt.ylabel("Test MAE")
        plt.title("Shared CNN test MAE by assigned project")
        plt.xticks(rotation=25, ha="right")
        save_plot(OVERALL / "graphs" / "test_mae_by_project.png")

    print("\nReport generation complete.")
    print(f"Overall results: {OVERALL}")
    print(f"Project results: {PROJECTS_OUT}")
    print("\nHeld-out test metrics:")
    print(test_metrics.to_string(index=False))
    print("\nShared CNN test metrics by project:")
    print(
        cnn_project_test[
            ["project_id", "project_key", "n", "MAE", "MdAE", "RMSE", "R2"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
