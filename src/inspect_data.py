import json
from pathlib import Path
import pandas as pd

RAW = Path("data/raw/tawos.csv")
ALLOC = Path("project_allocation.csv")
MEMBER = "simran"
OUT = Path("data/processed")

alloc = pd.read_csv(ALLOC)
mine = alloc[alloc["assigned_member"].str.lower() == MEMBER]
print("My projects:", mine["project_key"].tolist())

cols = ["issue_id", "issue_key", "project_id", "title",
        "description_text", "issue_type", "story_point"]
df = pd.read_csv(RAW, usecols=cols)
df = df[df["project_id"].isin(mine["project_id"])].copy()

def blank(s):
    return s.isna() | s.astype(str).str.strip().eq("")

summary_rows = []
for _, row in mine.iterrows():
    pid, key = int(row["project_id"]), row["project_key"]
    p = df[df["project_id"] == pid].copy()
    sp = pd.to_numeric(p["story_point"], errors="coerce")
    non_numeric = int((p["story_point"].notna() & sp.isna()).sum())
    lab = p[sp.notna()].copy()
    lab["sp"] = sp[sp.notna()]

    norm_title = lab["title"].fillna("").str.lower().str.split().str.join(" ")
    conflict = lab.groupby(norm_title)["sp"].agg(["size", "nunique"])
    conflict_groups = int(((conflict["size"] > 1) & (conflict["nunique"] > 1)).sum())

    profile = {
        "project_id": pid,
        "project_key": key,
        "total_issues": len(p),
        "labeled_issues": len(lab),
        "expected_labeled": int(row["labeled_issues"]),
        "unlabeled_issues": int(sp.isna().sum()),
        "non_numeric_labels": non_numeric,
        "missing_title_labeled": int(blank(lab["title"]).sum()),
        "missing_desc_labeled": int(blank(lab["description_text"]).sum()),
        "both_blank_labeled": int((blank(lab["title"]) & blank(lab["description_text"])).sum()),
        "zero_labels": int((lab["sp"] == 0).sum()),
        "fractional_labels": int((lab["sp"] % 1 != 0).sum()),
        "labels_gt_100": int((lab["sp"] > 100).sum()),
        "duplicate_issue_keys": int(p["issue_key"].duplicated().sum()),
        "same_title_conflicting_label_groups": conflict_groups,
        "sp_min": float(lab["sp"].min()),
        "sp_median": float(lab["sp"].median()),
        "sp_p90": float(lab["sp"].quantile(0.9)),
        "sp_p99": float(lab["sp"].quantile(0.99)),
        "sp_max": float(lab["sp"].max()),
        "issue_types": lab["issue_type"].value_counts(dropna=False).to_dict(),
    }

    d = OUT / key
    d.mkdir(parents=True, exist_ok=True)
    (d / "project_profile.json").write_text(json.dumps(profile, indent=2))
    summary_rows.append({k: v for k, v in profile.items() if k != "issue_types"})
    print(f"{key}: total={len(p)}, labeled={len(lab)} (expected {row['labeled_issues']})")

pd.DataFrame(summary_rows).to_csv("reports/simran_profile_summary.csv", index=False)
print("Saved reports/simran_profile_summary.csv")