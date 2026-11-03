
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

out = Path("results/plots")
out.mkdir(parents=True, exist_ok=True)

# Figure 1: Validation MAE by fold
folds = ["Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5"]
fold_mae = [3.403448, 3.890747, 3.352985, 3.511915, 3.240796]
mean_mae = 3.4800

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(folds, fold_mae)
ax.axhline(mean_mae, linestyle="--",
           label=f"Mean MAE = {mean_mae:.4f}")
ax.set_ylabel("MAE")
ax.set_title("Five-Fold Cross-Validation MAE")
ax.legend()
fig.tight_layout()
fig.savefig(out / "validation_mae_by_fold.png", dpi=300)
plt.close(fig)

# Figure 2: Validation versus test metrics
metrics = ["MAE", "MdAE"]
validation = [3.4800, 1.9556]
test = [5.2193, 2.4027]
x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(7, 5))
ax.bar(x - width / 2, validation, width, label="CV mean")
ax.bar(x + width / 2, test, width, label="Held-out test")
ax.set_xticks(x, metrics)
ax.set_ylabel("Error (lower is better)")
ax.set_title("Validation vs. Test Performance")
ax.legend()
fig.tight_layout()
fig.savefig(out / "validation_vs_test_metrics.png", dpi=300)
plt.close(fig)

# Figure 3: Test metrics by project
projects = [
    "XD", "TISTUD", "JRASERVER", "CONFCLOUD",
    "DURACLOUD", "COMPASS", "MDL"
]
mae = [6.0888, 2.8011, 3.1321, 1.4245, 2.1358, 1.8509, 12.2563]
mdae = [2.7367, 1.9026, 2.2831, 1.4083, 2.3490, 1.4897, 8.0456]
x = np.arange(len(projects))

fig, ax = plt.subplots(figsize=(11, 6))
ax.bar(x - width / 2, mae, width, label="MAE")
ax.bar(x + width / 2, mdae, width, label="MdAE")
ax.set_xticks(x, projects, rotation=25, ha="right")
ax.set_ylabel("Error (lower is better)")
ax.set_title("Held-Out Test Performance by Project")
ax.legend()
fig.tight_layout()
fig.savefig(out / "test_metrics_by_project.png", dpi=300)
plt.close(fig)

print(f"Saved three plots to {out.resolve()}")

