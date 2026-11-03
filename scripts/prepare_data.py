import os
import pandas as pd

# Paths
BASE_DIR = os.path.expanduser("~/tawos-sbert")
DATA_PATH = os.path.join(BASE_DIR, "data/raw/tawos.csv")
PROCESSED_DIR = os.path.join(BASE_DIR, "data/processed")

os.makedirs(PROCESSED_DIR, exist_ok=True)

# The seven assigned projects
PROJECTS = {
    1: "XD",
    13: "TISTUD",
    18: "JRASERVER",
    21: "CONFCLOUD",
    29: "DURACLOUD",
    30: "COMPASS",
    34: "MDL",
}

# Verify the input file
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found at {DATA_PATH}\n"
        "Copy tawos.csv into ~/tawos-sbert/data/raw/"
    )

print("Loading original CSV...")
df = pd.read_csv(DATA_PATH, low_memory=False)

required_columns = [
    "issue_key",
    "title",
    "description_text",
    "story_point",
    "project_id",
]

missing = [c for c in required_columns if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

print("Original dataset shape:", df.shape)

# Keep the original DataFrame unchanged
data = df.copy()

# Convert IDs and labels to numeric values
data["project_id"] = pd.to_numeric(
    data["project_id"], errors="coerce"
)
data["story_point"] = pd.to_numeric(
    data["story_point"], errors="coerce"
)

# Keep only the seven assigned projects
data = data[data["project_id"].isin(PROJECTS)].copy()

print("Issues from the seven projects:", len(data))

# Keep labeled issues only
data = data[data["story_point"].notna()].copy()

print("Labeled issues before text cleaning:", len(data))

# Clean missing title and description values
for col in ["title", "description_text"]:
    data[col] = (
        data[col]
        .fillna("")
        .astype(str)
        .str.strip()
    )

# Combine title and description
data["text"] = (
    data["title"] + ". " + data["description_text"]
).str.strip()

# Remove leading/trailing spaces and separator punctuation
data["text"] = data["text"].str.strip(" .\t\n\r")

# Exclude issues with no usable text
data = data[data["text"].ne("")].copy()

# Add readable project names
data["project_name"] = data["project_id"].map(PROJECTS)

# Create a stable group ID for duplicate-text leakage checks
import hashlib

normalized_text = (
    data["text"]
    .str.lower()
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

data["split_group"] = normalized_text.map(
    lambda s: hashlib.sha256(s.encode("utf-8")).hexdigest()
)

# Save the combined labeled dataset
output_path = os.path.join(
    PROCESSED_DIR, "all_projects_labeled.csv"
)
data.to_csv(output_path, index=False)

# Report results
print("\nLabeled issues per project:")
summary = (
    data.groupby(["project_id", "project_name"])
    .agg(
        issue_count=("story_point", "size"),
        unique_story_points=("story_point", "nunique"),
    )
    .reset_index()
    .sort_values("project_id")
)
print(summary.to_string(index=False))

print("\nFinal dataset shape:", data.shape)
print("Missing labels:", data["story_point"].isna().sum())
print("Empty texts:", data["text"].str.strip().eq("").sum())
print("Saved to:", output_path)

assert data["story_point"].notna().all()
assert data["project_id"].isin(PROJECTS).all()
assert data["text"].str.strip().ne("").all()

print("\nSUCCESS: labeled-only dataset is ready.")
