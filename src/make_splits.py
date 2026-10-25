"""Step 4: 80/20 development/test split + 5 development folds, per project.

Rows with identical normalized input text share a duplicate_group, and a group
is never divided across test/development or across folds.
Manifest columns: issue_key,project_id,split,fold,duplicate_group
(test rows have fold = 0; development folds are 1..5).
"""
import numpy as np
import pandas as pd
from .common import normalize_for_dup


def make_splits(labeled, cfg, out_path):
    seed = cfg["experiment"]["random_seed"]
    k = cfg["experiment"]["validation_folds"]
    test_frac = cfg["experiment"]["test_fraction"]
    n = len(labeled)
    if n < cfg["experiment"]["min_labeled_issues"]:
        raise ValueError(f"Only {n} eligible labeled issues; below min_labeled_issues="
                         f"{cfg['experiment']['min_labeled_issues']}. Consult the coordinator "
                         "before changing the required design.")
    rng = np.random.default_rng(seed)
    norm = labeled["input_text"].map(normalize_for_dup)
    gid = norm.astype("category").cat.codes.to_numpy()
    groups = pd.Series(np.arange(n)).groupby(gid).apply(list).to_list()

    # test set: shuffle groups, add while the target size is not exceeded
    order = rng.permutation(len(groups))
    target = int(round(test_frac * n))
    test_idx, cnt = [], 0
    for gi in order:
        if cnt + len(groups[gi]) <= target:
            test_idx += groups[gi]
            cnt += len(groups[gi])
    is_test = np.zeros(n, dtype=bool)
    is_test[test_idx] = True

    # development folds: largest groups first, each into the currently smallest fold
    dev_groups = [g for g in groups if not is_test[g[0]]]
    sizes = np.array([len(g) for g in dev_groups])
    tiebreak = rng.random(len(dev_groups))
    fold_of = np.zeros(n, dtype=int)
    fold_sizes = [0] * k
    for gi in np.lexsort((tiebreak, -sizes)):
        f = int(np.argmin(fold_sizes))
        fold_of[dev_groups[gi]] = f + 1
        fold_sizes[f] += len(dev_groups[gi])
    if min(fold_sizes) < 2:
        raise ValueError(f"A development fold has too few rows: {fold_sizes}. Consult the coordinator.")

    m = pd.DataFrame({
        "issue_key": labeled["issue_key"].to_numpy(),
        "project_id": labeled["project_id"].to_numpy(),
        "split": np.where(is_test, "test", "development"),
        "fold": np.where(is_test, 0, fold_of),
        "duplicate_group": gid,
    })
    m.to_csv(out_path, index=False)
    print(f"Split: development={int((~is_test).sum())} test={int(is_test.sum())} fold sizes={fold_sizes}")
    return m
