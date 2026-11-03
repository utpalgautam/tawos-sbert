import pandas as pd
from pathlib import Path

targets = {
    "XD-2602",
    "XD-92",
    "MDL-46658",
    "MDL-40975",
    "MDL-40063",
    "MDL-42102",
    "MDL-12403",
    "MDL-39851",
}

files = [
    Path("data/raw/tawos.csv"),
    Path("data/processed/all_projects_labeled.csv"),
    Path("data/processed/splits/train.csv"),
    Path("data/processed/splits/validation.csv"),
    Path("data/processed/splits/test.csv"),
]

for path in files:
    if not path.exists():
        continue

    print(f"\n{'=' * 70}\nFILE: {path}\n{'=' * 70}")

    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as exc:
        print(f"Could not read file: {exc}")
        continue

    print("Columns:", df.columns.tolist())

    # Find likely issue-key columns.
    key_cols = [
        c for c in df.columns
        if any(term in c.lower() for term in ("issue_key", "issuekey", "issue_id", "key"))
    ]

    matched = pd.Series(False, index=df.index)
    for col in key_cols:
        matched |= df[col].astype(str).isin(targets)

    if not matched.any():
        print("No exact matches in likely issue-key columns.")
        continue

    rows = df.loc[matched]
    relevant_cols = [
        c for c in df.columns
        if any(term in c.lower() for term in (
            "key", "story", "point", "project", "summary",
            "title", "description", "text", "estimate"
        ))
    ]

    print(rows[relevant_cols].to_string(index=False, max_colwidth=250))
