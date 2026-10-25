import pandas as pd
from pathlib import Path

keys = ["NEXUS", "CLI", "TIDOC", "BAM", "CLOV", "JSWSERVER", "INDY", "EVG"]
rows = []
for k in keys:
    d = pd.read_csv(Path("data/processed") / k / "labeled_clean.csv")
    top = d.nlargest(3, "story_point")
    for _, r in top.iterrows():
        rows.append({
            "project": k, "issue_key": r["issue_key"],
            "story_point": r["story_point"], "issue_type": r["issue_type"],
            "title": r["title_clean"],
            "description": str(r["description_clean"])[:300],
        })
        print(f"\n[{k}] {r['issue_key']}  SP={r['story_point']}  type={r['issue_type']}")
        print("  Title:", r["title_clean"])
        print("  Desc :", str(r["description_clean"])[:300])

pd.DataFrame(rows).to_csv("reports/simran_extreme_review.csv", index=False)
print("\nSaved reports/simran_extreme_review.csv")