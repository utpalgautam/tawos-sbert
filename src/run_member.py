"""Run all projects assigned to one member: python -m src.run_member --member neha"""
import argparse
import subprocess
import sys
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--member", required=True)
ap.add_argument("--allocation", default="project_allocation.csv")
ap.add_argument("--max-epochs", type=int, default=None)
a = ap.parse_args()

alloc = pd.read_csv(a.allocation)
mine = alloc[alloc["assigned_member"].str.lower() == a.member.lower()].sort_values("labeled_issues")
if mine.empty:
    sys.exit(f"No projects found for member '{a.member}'")

failed = []
for _, r in mine.iterrows():
    cmd = [sys.executable, "-m", "src.run_problem5", "--project-id", str(r.project_id), "--member", a.member.lower()]
    if a.max_epochs:
        cmd += ["--max-epochs", str(a.max_epochs)]
    print(f"\n##### {r.project_key} (id {r.project_id}, {r.labeled_issues} labeled) #####", flush=True)
    if subprocess.run(cmd).returncode != 0:
        failed.append(r.project_key)

print("\nFinished. Failed projects:", failed or "none")