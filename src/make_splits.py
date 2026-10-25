from pathlib import Path
import numpy as np
import pandas as pd

ALLOC = Path("project_allocation.csv")
MEMBER = "simran"
PROC = Path("data/processed")
OUT = Path("outputs") / MEMBER
SEED = 42
TEST_FRACTION = 0.20
N_FOLDS = 5

alloc = pd.read_csv(ALLOC)
mine = alloc[alloc["assigned_member"].str.lower() == MEMBER]

for _, row in mine.iterrows():
    key, pid = row["project_key"], int(row["project_id"])
    d = pd.read_csv(PROC / key / "labeled_clean.csv")

    # rows with no duplicate group become their own group
    d["duplicate_group"] = d["duplicate_group"].fillna("")
    d["grp"] = np.where(d["duplicate_group"] == "",
                        "S_" + d["issue_key"].astype(str), d["duplicate_group"])

    sizes = d.groupby("grp").size()
    rng = np.random.default_rng(SEED)

    # 1) locked test set: shuffle groups, fill until ~20% of rows
    order = rng.permutation(sizes.index.to_numpy())
    target = round(TEST_FRACTION * len(d))
    test_groups, count = set(), 0
    for g in order:
        if count >= target:
            break
        test_groups.add(g)
        count += sizes[g]

    # 2) five folds on development groups, balanced by row count
    dev_groups = [g for g in order if g not in test_groups]
    dev_groups.sort(key=lambda g: -sizes[g])  # stable: ties stay in shuffled order
    fold_of, fold_count = {}, [0] * N_FOLDS
    for g in dev_groups:
        f = int(np.argmin(fold_count))
        fold_of[g] = f + 1
        fold_count[f] += sizes[g]

    d["split"] = np.where(d["grp"].isin(test_groups), "test", "development")
    d["fold"] = d["grp"].map(fold_of)
    d["fold"] = d["fold"].astype("Int64")

    # leakage check: no duplicate group on both sides
    both = set(d[d.split == "test"]["grp"]) & set(d[d.split == "development"]["grp"])
    assert not both, f"{key}: duplicate groups leak across test/dev"

    path = OUT / key
    path.mkdir(parents=True, exist_ok=True)
    d[["issue_key", "project_id", "split", "fold", "duplicate_group"]] \
        .to_csv(path / "split_manifest.csv", index=False)

    dev_n = int((d.split == "development").sum())
    print(f"\n{key}: total={len(d)} development={dev_n} test={int((d.split=='test').sum())}")
    print("  fold sizes:", d[d.split == "development"]["fold"].value_counts().sort_index().to_dict())

print("\nSaved split manifests under", OUT)