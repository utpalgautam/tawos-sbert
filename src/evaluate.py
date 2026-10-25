"""MAE / MdAE and training-median baseline."""
import numpy as np
from sklearn.metrics import mean_absolute_error, median_absolute_error


def metrics(y_true, y_pred):
    return {"mae": float(mean_absolute_error(y_true, y_pred)),
            "mdae": float(median_absolute_error(y_true, y_pred))}


def baseline_metrics(y_train, y_eval):
    """Predict the TRAINING median for every evaluation row."""
    pred = np.full(len(y_eval), np.median(y_train))
    return metrics(y_eval, pred)


def make_plots(test_df, cv_fold_df, y_all, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(test_df["actual"], test_df["predicted"], s=8, alpha=.5)
    lim = [0, max(test_df["actual"].max(), test_df["predicted"].max())]
    ax.plot(lim, lim, "r--"); ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
    ax.set_title("Test: actual vs predicted")
    fig.savefig(out_dir / "plot_actual_vs_pred.png", dpi=120, bbox_inches="tight"); plt.close(fig)
    fig, ax = plt.subplots(); ax.hist(test_df["absolute_error"], bins=40)
    ax.set_title("Test absolute error")
    fig.savefig(out_dir / "plot_abs_error.png", dpi=120, bbox_inches="tight"); plt.close(fig)
    fig, ax = plt.subplots(); ax.hist(y_all, bins=60); ax.set_yscale("log")
    ax.set_title("Label distribution (log count)")
    fig.savefig(out_dir / "plot_label_dist.png", dpi=120, bbox_inches="tight"); plt.close(fig)
    fig, ax = plt.subplots(); ax.bar(cv_fold_df["fold"], cv_fold_df["mae"])
    ax.set_xlabel("Fold"); ax.set_ylabel("Validation MAE"); ax.set_title("Validation MAE by fold")
    fig.savefig(out_dir / "plot_cv_mae.png", dpi=120, bbox_inches="tight"); plt.close(fig)
