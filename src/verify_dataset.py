from pathlib import Path
import pandas as pd

path = Path("data/raw/tawos.csv")
df = pd.read_csv(path)

print("Rows:", len(df), "| Columns:", len(df.columns))
print("Projects:", df["project_id"].nunique())
print("Labeled:", int(df["story_point"].notna().sum()))
print("Unlabeled:", int(df["story_point"].isna().sum()))
print("Columns:", df.columns.tolist())

required = {"issue_key", "project_id", "title", "description_text", "story_point"}
missing = required - set(df.columns)
print("Missing expected columns:", sorted(missing) if missing else "none")