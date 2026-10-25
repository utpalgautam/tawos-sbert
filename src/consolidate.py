"""Combine every test_metrics.json under outputs/ into one table."""
import json
import pandas as pd
from .common import REPO_ROOT

rows = []
for p in (REPO_ROOT / "outputs").glob("*/*/test_metrics.json"):
    d = json.load(open(p))
    d["member"], d["project_key"] = p.parts[-3], p.parts[-2]
    rows.append(d)
df = pd.DataFrame(rows).sort_values("project_id")
df.to_csv(REPO_ROOT / "outputs" / "problem5_all_projects.csv", index=False)
print(df.to_string(index=False))
