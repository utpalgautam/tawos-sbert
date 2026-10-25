"""Step 0: verify dataset path/schema/counts. Run: python -m src.inspect_data"""
import pandas as pd
from .common import load_config, REPO_ROOT


def main():
    cfg = load_config()
    path = REPO_ROOT / cfg["data"]["path"]
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path, low_memory=False)
    print("Path:", path)
    print("Rows:", len(df), "Columns:", len(df.columns))
    print("Projects:", df["project_id"].nunique())
    print("Labeled:", int(df["story_point"].notna().sum()))
    print("Unlabeled:", int(df["story_point"].isna().sum()))
    print("Columns:", df.columns.tolist())
    need = {"issue_key", "project_id", "title", "description_text", "story_point"}
    miss = need - set(df.columns)
    if miss:
        print("WARNING - missing expected columns:", sorted(miss))
    g = (df.groupby("project_id")
           .agg(total=("issue_key", "size"), labeled=("story_point", lambda s: s.notna().sum()),
                key=("issue_key", lambda s: s.dropna().astype(str).str.split("-").str[0].mode().iloc[0]))
           .reset_index())
    print(g.to_string(index=False))


if __name__ == "__main__":
    main()
