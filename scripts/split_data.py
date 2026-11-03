
import os
import hashlib
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

BASE_DIR = os.path.expanduser("~/tawos-sbert")
INPUT_PATH = f"{BASE_DIR}/data/processed/all_projects_labeled.csv"
SPLIT_DIR = f"{BASE_DIR}/data/processed/splits"

os.makedirs(SPLIT_DIR, exist_ok=True)

PROJECTS = {
    1: "XD",
    13: "TISTUD",
    18: "JRASERVER",
    21: "CONFCLOUD",
    29: "DURACLOUD",
    30: "COMPASS",
    34: "MDL",
}

SEED = 42

# Load labeled data
data = pd.read_csv(INPUT_PATH, low_memory=False)

required = ["text", "story_point", "project_id"]
missing = [c for c in required if c not in data.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

data["project_id"] = pd.to_numeric(
    data["project_id"], errors="coerce"
)
data["story_point"] = pd.to_numeric(
    data["story_point"], errors="coerce"
)

# Strictly labeled issues from assigned projects only
data = data[
    data["story_point"].notna()
    & data["project_id"].isin(PROJECTS)
].copy()

data = data.reset_index(drop=True)

# Stable grouping of identical normalized text
normalized = (
    data["text"].fillna("").astype(str)
    .str.lower()
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

data["split_group"] = normalized.map(
    lambda s: hashlib.sha256(s.encode("utf-8")).hexdigest()
)

# First split: 80% development, 20% test
outer = GroupShuffleSplit(
    n_splits=1, test_size=0.20, random_state=SEED
)
dev_idx, test_idx = next(
    outer.split(data, groups=data["split_group"])
)

development = data.iloc[dev_idx].copy()
test = data.iloc[test_idx].copy()

# Second split: 80% train, 20% validation of development
inner = GroupShuffleSplit(
    n_splits=1, test_size=0.20, random_state=SEED
)
train_idx, val_idx = next(
    inner.split(
        development,
        groups=development["split_group"]
    )
)

train = development.iloc[train_idx].copy()
validation = development.iloc[val_idx].copy()

# Check that no duplicate-text group crosses splits
train_groups = set(train["split_group"])
val_groups = set(validation["split_group"])
test_groups = set(test["split_group"])

assert train_groups.isdisjoint(val_groups)
assert train_groups.isdisjoint(test_groups)
assert val_groups.isdisjoint(test_groups)

# Verify all rows are labeled and belong to assigned projects
for name, part in [
    ("train", train),
    ("validation", validation),
    ("test", test),
]:
    assert part["story_point"].notna().all()
    assert part["project_id"].isin(PROJECTS).all()

    path = f"{SPLIT_DIR}/{name}.csv"
    part.to_csv(path, index=False)
    print(f"{name}: {len(part)} issues saved to {path}")

print("\nProject counts by split:")
for name, part in [
    ("TRAIN", train),
    ("VALIDATION", validation),
    ("TEST", test),
]:
    print(f"\n{name}")
    print(
        part.groupby(["project_id", "project_name"])
        .size()
        .rename("issue_count")
        .reset_index()
        .to_string(index=False)
    )

print("\nTotal:", len(train) + len(validation) + len(test))
print("Leakage check passed.")

