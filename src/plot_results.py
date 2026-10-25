"""
TAWOS SBERT + CNN plotting script.

Expected layout:
    outputs/
      BAM/test_predictions.csv
      CLI/test_predictions.csv
      CLOV/test_predictions.csv
      EVG/test_predictions.csv
      INDY/test_predictions.csv
      JSWSERVER/test_predictions.csv
      NEXUS/test_predictions.csv
      TIDOC/test_predictions.csv

Individual plots: outputs/<PROJECT>/graphs/
Combined plots:   outputs/graphs/

Run from repository root:
    python plot_results_corrected.py

Install dependencies:
    python -m pip install pandas numpy matplotlib
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECTS = ["BAM", "CLI", "CLOV", "EVG", "INDY", "JSWSERVER", "NEXUS", "TIDOC"]

# Works when this script is in the repository root or in its src/ folder.
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name.lower() == "src" else SCRIPT_DIR
OUTPUTS_DIR = ROOT / "outputs"

# Existing prediction and metrics files are under outputs/simran/<PROJECT>/
RUN_DIR = OUTPUTS_DIR / "simran"

# Combined graphs for all eight projects go directly into outputs/graphs/
ALL_GRAPHS_DIR = OUTPUTS_DIR / "graphs"
# Earlier MSE-loss model test MAE results. Confirm both experiments used the same
# held-out test splits before interpreting this comparison.
MSE_LOSS_TEST_MAE = {
    "BAM": 1.347851,
    "CLI": 2.204109,
    "CLOV": 5.328741,
    "EVG": 0.817479,
    "INDY": 1.483852,
    "JSWSERVER": 2.785962,
    "NEXUS": 1.134509,
    "TIDOC": 3.151451,
}


def find_column(df, candidates):
    """Case-insensitive column finder."""
    cols = {str(c).strip().lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in cols:
            return cols[candidate.lower()]
    return None


def read_metrics_json(path):
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        warnings.warn(f"Could not read {path}: {exc}")
        return {}


def json_metric(data, section_candidates, metric_candidates):
    """Find metrics in common nested or flat JSON formats."""
    if not isinstance(data, dict):
        return np.nan
    for section in section_candidates:
        block = data.get(section)
        if isinstance(block, dict):
            for key in metric_candidates:
                if key in block:
                    try:
                        return float(block[key])
                    except (TypeError, ValueError):
                        pass
    for key in metric_candidates:
        if key in data:
            try:
                return float(data[key])
            except (TypeError, ValueError):
                pass
    return np.nan


def read_predictions(project):
    path = RUN_DIR / project / "test_predictions.csv"
    if not path.exists():
        warnings.warn(f"Missing predictions CSV: {path}")
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        warnings.warn(f"Could not read {path}: {exc}")
        return None


def get_prediction_arrays(df, project):
    if df is None or df.empty:
        return None

    actual_col = find_column(
        df, ["actual_story_point", "actual_story_points", "actual", "y_true"]
    )
    pred_col = find_column(
        df, [
            "predicted_story_point", "predict_story_point",
            "predicted_story_points", "prediction", "y_pred"
        ]
    )
    if actual_col is None or pred_col is None:
        warnings.warn(
            f"{project}: couldn't find actual/prediction columns. "
            f"CSV columns: {df.columns.tolist()}"
        )
        return None

    actual = pd.to_numeric(df[actual_col], errors="coerce")
    predicted = pd.to_numeric(df[pred_col], errors="coerce")
    valid = actual.notna() & predicted.notna()
    actual = actual[valid].to_numpy(dtype=float)
    predicted = predicted[valid].to_numpy(dtype=float)

    if not len(actual):
        warnings.warn(f"{project}: no valid numeric prediction rows.")
        return None
    errors = np.abs(actual - predicted)
    return actual, predicted, errors, df.loc[valid].copy()


def baseline_errors_from_df(df, actual_column=None):
    """Return baseline absolute errors if the CSV includes them."""
    error_col = find_column(
        df, [
            "baseline_absolute_error", "baseline_abs_error",
            "median_baseline_absolute_error", "baseline_error"
        ]
    )
    if error_col is not None:
        values = pd.to_numeric(df[error_col], errors="coerce").dropna()
        if len(values):
            return values.to_numpy(dtype=float)

    baseline_col = find_column(
        df, [
            "baseline_prediction", "baseline_predicted_story_point",
            "median_baseline_prediction", "train_median_prediction"
        ]
    )
    actual_col = actual_column or find_column(
        df, ["actual_story_point", "actual_story_points", "actual", "y_true"]
    )
    if baseline_col is not None and actual_col is not None:
        actual = pd.to_numeric(df[actual_col], errors="coerce")
        baseline = pd.to_numeric(df[baseline_col], errors="coerce")
        valid = actual.notna() & baseline.notna()
        if valid.any():
            return np.abs(
                actual[valid].to_numpy(dtype=float)
                - baseline[valid].to_numpy(dtype=float)
            )
    return None


def save_plot(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def create_individual_graphs(project, df):
    """Save per-project graphs under outputs/<PROJECT>/graphs/."""
    arrays = get_prediction_arrays(df, project)
    if arrays is None:
        return

    actual, predicted, errors, valid_df = arrays
    graph_dir = RUN_DIR / project / "graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)

    # 1. Actual vs predicted scatter plot.
    plt.figure(figsize=(7.5, 6.2))
    plt.scatter(actual, predicted, alpha=0.65, edgecolors="none")
    low = min(float(actual.min()), float(predicted.min()))
    high = max(float(actual.max()), float(predicted.max()))
    if low == high:
        high = low + 1.0
    plt.plot([low, high], [low, high], "--", linewidth=1.5,
             label="Perfect prediction (y = x)")
    plt.xlabel("Actual story points")
    plt.ylabel("Predicted story points")
    plt.title(f"{project}: Actual vs. Predicted Story Points")
    plt.legend()
    plt.grid(True, alpha=0.25)
    save_plot(graph_dir / "actual_vs_predicted.png")

    # 2. Absolute error histogram.
    plt.figure(figsize=(7.5, 5.2))
    plt.hist(errors, bins="auto", edgecolor="black", alpha=0.8)
    plt.axvline(errors.mean(), linestyle="--", linewidth=1.6,
                label=f"MAE = {errors.mean():.3f}")
    plt.axvline(np.median(errors), linestyle=":", linewidth=1.8,
                label=f"MdAE = {np.median(errors):.3f}")
    plt.xlabel("Absolute error |actual − predicted|")
    plt.ylabel("Number of test issues")
    plt.title(f"{project}: Absolute Prediction Error Distribution")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.25)
    save_plot(graph_dir / "absolute_error_distribution.png")

    # 3. CNN vs baseline absolute-error box plot, if baseline data is available.
    baseline_errors = baseline_errors_from_df(
        valid_df,
        actual_column=find_column(
            valid_df, ["actual_story_point", "actual_story_points", "actual", "y_true"]
        ),
    )
    if baseline_errors is not None and len(baseline_errors):
        plt.figure(figsize=(7.5, 5.2))
        # Matplotlib's labels parameter works across a broad range of versions.
        plt.boxplot(
            [baseline_errors, errors],
            tick_labels=["Train-median baseline", "CNN"],
            showfliers=True,
        )
        plt.ylabel("Absolute error")
        plt.title(f"{project}: Absolute Errors vs. Baseline")
        plt.grid(True, axis="y", alpha=0.25)
        save_plot(graph_dir / "absolute_error_boxplot_vs_baseline.png")


def create_summary(project, df, metrics):
    arrays = get_prediction_arrays(df, project)
    if arrays is not None:
        actual, predicted, errors, _ = arrays
        cnn_mae = float(np.mean(errors))
        cnn_mdae = float(np.median(errors))
        test_size = int(len(errors))
    else:
        cnn_mae = json_metric(
            metrics, ["cnn", "model", "test", "test_metrics"],
            ["mae", "test_mae", "MAE"]
        )
        cnn_mdae = json_metric(
            metrics, ["cnn", "model", "test", "test_metrics"],
            ["mdae", "median_absolute_error", "test_mdae", "MdAE"]
        )
        n = json_metric(metrics, ["test"], ["test_size", "n_test", "n"])
        test_size = int(n) if np.isfinite(n) else 0

    baseline_mae = json_metric(
        metrics,
        ["baseline", "median_baseline", "train_median_baseline", "baseline_metrics"],
        ["mae", "test_mae", "MAE"],
    )
    baseline_mdae = json_metric(
        metrics,
        ["baseline", "median_baseline", "train_median_baseline", "baseline_metrics"],
        ["mdae", "median_absolute_error", "test_mdae", "MdAE"],
    )

    if df is not None:
        be = baseline_errors_from_df(df)
        if be is not None and len(be):
            if not np.isfinite(baseline_mae):
                baseline_mae = float(np.mean(be))
            if not np.isfinite(baseline_mdae):
                baseline_mdae = float(np.median(be))

    # These are fallback values from the user's reported current L1Loss run.
    # They are used only if the output metrics do not provide baseline metrics.
    reported = {
        "BAM": (1.094340, 1.000000),
        "CLI": (1.986667, 2.000000),
        "CLOV": (3.733766, 1.000000),
        "EVG": (0.812963, 1.000000),
        "INDY": (1.397059, 1.000000),
        "JSWSERVER": (2.200000, 2.000000),
        "NEXUS": (0.991599, 0.500000),
        "TIDOC": (2.980695, 2.000000),
    }
    if project in reported:
        if not np.isfinite(baseline_mae):
            baseline_mae = reported[project][0]
        if not np.isfinite(baseline_mdae):
            baseline_mdae = reported[project][1]

    return {
        "project": project,
        "test_size": test_size,
        "cnn_mae": cnn_mae,
        "baseline_mae": baseline_mae,
        "cnn_mdae": cnn_mdae,
        "baseline_mdae": baseline_mdae,
    }


def create_all_project_graphs(summary_rows, cv_frames):
    """Save combined comparisons directly inside outputs/graphs/."""
    ALL_GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(ALL_GRAPHS_DIR / "all_projects_test_summary.csv", index=False)

    x = np.arange(len(summary))
    width = 0.36

    # 1. Test MAE: CNN vs baseline.
    if summary["cnn_mae"].notna().any() and summary["baseline_mae"].notna().any():
        plt.figure(figsize=(11, 6))
        plt.bar(x - width / 2, summary["baseline_mae"], width,
                label="Train-median baseline")
        plt.bar(x + width / 2, summary["cnn_mae"], width,
                label="CNN (L1Loss)")
        plt.xticks(x, summary["project"], rotation=25, ha="right")
        plt.ylabel("Test MAE (lower is better)")
        plt.title("Test MAE: CNN vs. Train-Median Baseline")
        plt.legend()
        plt.grid(True, axis="y", alpha=0.25)
        save_plot(ALL_GRAPHS_DIR / "test_mae_comparison.png")

        valid = (
            summary["cnn_mae"].notna()
            & summary["baseline_mae"].notna()
            & (summary["baseline_mae"] != 0)
        )
        if valid.any():
            subset = summary.loc[valid]
            pct = (
                (subset["cnn_mae"] - subset["baseline_mae"])
                / subset["baseline_mae"] * 100
            )
            plt.figure(figsize=(10, 6))
            plt.barh(subset["project"], pct)
            plt.axvline(0, color="black", linewidth=1)
            plt.xlabel("(CNN MAE − baseline MAE) / baseline MAE (%)")
            plt.title("CNN Test MAE Difference Relative to Baseline")
            plt.grid(True, axis="x", alpha=0.25)
            save_plot(ALL_GRAPHS_DIR / "mae_percentage_difference.png")
    else:
        warnings.warn("Insufficient metrics for the test MAE comparison.")

    # 2. Test MdAE: CNN vs baseline.
    if summary["cnn_mdae"].notna().any() and summary["baseline_mdae"].notna().any():
        plt.figure(figsize=(11, 6))
        plt.bar(x - width / 2, summary["baseline_mdae"], width,
                label="Train-median baseline")
        plt.bar(x + width / 2, summary["cnn_mdae"], width,
                label="CNN (L1Loss)")
        plt.xticks(x, summary["project"], rotation=25, ha="right")
        plt.ylabel("Test MdAE (lower is better)")
        plt.title("Test MdAE: CNN vs. Train-Median Baseline")
        plt.legend()
        plt.grid(True, axis="y", alpha=0.25)
        save_plot(ALL_GRAPHS_DIR / "test_mdae_comparison.png")
    else:
        warnings.warn("Insufficient metrics for the test MdAE comparison.")

    # 3. Earlier MSE-loss CNN vs current L1Loss CNN test MAE.
    if set(summary["project"]) == set(PROJECTS):
        ordered = summary.set_index("project").reindex(PROJECTS)
        mse = [MSE_LOSS_TEST_MAE[p] for p in PROJECTS]
        l1 = ordered["cnn_mae"].to_numpy(dtype=float)
        if np.isfinite(l1).any():
            positions = np.arange(len(PROJECTS))
            plt.figure(figsize=(11, 6))
            plt.bar(positions - width / 2, mse, width, label="CNN trained with MSELoss")
            plt.bar(positions + width / 2, l1, width, label="CNN trained with L1Loss")
            plt.xticks(positions, PROJECTS, rotation=25, ha="right")
            plt.ylabel("Test MAE (lower is better)")
            plt.title("Test MAE: MSE-Loss CNN vs. MAE-Loss CNN")
            plt.legend()
            plt.grid(True, axis="y", alpha=0.25)
            save_plot(ALL_GRAPHS_DIR / "mse_vs_mae_loss_comparison.png")

    # 4. Number of held-out test issues.
    plt.figure(figsize=(10, 5.5))
    plt.bar(summary["project"], summary["test_size"])
    plt.ylabel("Number of test issues")
    plt.title("Held-Out Test Set Size by Project")
    plt.grid(True, axis="y", alpha=0.25)
    save_plot(ALL_GRAPHS_DIR / "test_set_sizes.png")

    # 5. Cross-validation fold MAE, if fold-level data exists.
    usable = []
    for project, frame in cv_frames.items():
        if frame is None or frame.empty:
            continue
        mae_col = find_column(
            frame,
            ["model_mae", "cv_model_mae", "mae", "val_mae", "validation_mae"],
        )
        fold_col = find_column(frame, ["fold", "fold_id", "fold_number"])
        if mae_col is None:
            warnings.warn(
                f"{project}: cannot identify validation MAE in CV columns "
                f"{frame.columns.tolist()}"
            )
            continue
        fold_values = (
            frame[fold_col].astype(str).to_numpy()
            if fold_col is not None
            else np.arange(1, len(frame) + 1).astype(str)
        )
        usable.append(pd.DataFrame({
            "project": project,
            "fold": fold_values,
            "cv_mae": pd.to_numeric(frame[mae_col], errors="coerce"),
        }))

    if usable:
        cv = pd.concat(usable, ignore_index=True).dropna(subset=["cv_mae"])
        cv.to_csv(ALL_GRAPHS_DIR / "cross_validation_fold_mae.csv", index=False)
        plt.figure(figsize=(11, 6))
        for project in PROJECTS:
            part = cv[cv["project"] == project]
            if not part.empty:
                plt.plot(part["fold"], part["cv_mae"], marker="o", label=project)
        plt.xlabel("Cross-validation fold")
        plt.ylabel("Validation MAE (lower is better)")
        plt.title("Cross-Validation MAE by Project and Fold")
        plt.legend(ncol=2)
        plt.grid(True, alpha=0.25)
        save_plot(ALL_GRAPHS_DIR / "cross_validation_mae.png")

        cv.groupby("project")["cv_mae"].agg(
            cv_mean_mae="mean", cv_std_mae="std"
        ).reset_index().to_csv(
            ALL_GRAPHS_DIR / "cross_validation_mae_summary.csv", index=False
        )
    else:
        print(
            "NOTE: Cross-validation graph skipped: no readable fold-level CSVs found. "
            "The script checks outputs/<PROJECT>/cv_fold_metrics.csv."
        )


def main():
    if not OUTPUTS_DIR.exists():
        raise FileNotFoundError(
            f"Output directory not found: {OUTPUTS_DIR}\n"
            "Place this script in your repository root (or src/)."
        )

    summaries = []
    cv_frames = {}

    for project in PROJECTS:
        project_dir = RUN_DIR / project
        df = read_predictions(project)
        metrics = read_metrics_json(project_dir / "test_metrics.json")
        create_individual_graphs(project, df)
        summaries.append(create_summary(project, df, metrics))

        cv_path = project_dir / "cv_fold_metrics.csv"
        if cv_path.exists():
            try:
                cv_frames[project] = pd.read_csv(cv_path)
            except Exception as exc:
                warnings.warn(f"Could not read {cv_path}: {exc}")
                cv_frames[project] = None
        else:
            cv_frames[project] = None

    create_all_project_graphs(summaries, cv_frames)

    print("\nGraph generation finished.")
    print(f"Individual project graphs: {OUTPUTS_DIR}\\<PROJECT>\\graphs\\")
    print(f"Combined graphs for all projects: {ALL_GRAPHS_DIR}\\")


if __name__ == "__main__":
    main()
