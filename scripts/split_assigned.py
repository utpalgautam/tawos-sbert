#!/usr/bin/env python3
"""
Step 4: Generate 80/20 Train/Test splits and 5-fold CV manifest.
Operates on data/processed/<project_key>/labeled_clean.csv.
Ensures duplicate_group never leaks across development and test,
and never leaks across CV folds.
Saves split_manifest.csv in:
  - outputs/devanarayanan/<project_key>/split_manifest.csv
  - data/processed/<project_key>/split_manifest.csv
"""

import csv
from pathlib import Path
import random
import sys

csv.field_size_limit(sys.maxsize)

ASSIGNED_PROJECTS = ["MESOS", "APSTUD", "JSWCLOUD", "FAB", "IS", "APIKIT", "MULE", "JRACLOUD"]

def make_project_splits(project_key: str, seed: int = 42, test_frac: float = 0.20, n_folds: int = 5):
    proc_dir = Path("data/processed") / project_key
    out_dir = Path("outputs/devanarayanan") / project_key
    labeled_file = proc_dir / "labeled_clean.csv"

    if not labeled_file.exists():
        print(f"Skipping {project_key}: {labeled_file} does not exist.")
        return

    with open(labeled_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print(f"[{project_key}] labeled_clean.csv is empty.")
        return

    # Group rows by duplicate_group
    group_map = {}
    for r in rows:
        gid = r.get("duplicate_group", r["issue_key"])
        group_map.setdefault(gid, []).append(r)

    unique_groups = list(group_map.keys())
    
    # Deterministic shuffle of groups
    rng = random.Random(seed)
    rng.shuffle(unique_groups)

    total_issues = len(rows)
    target_test_count = int(round(total_issues * test_frac))

    # Accumulate groups into test set until target_test_count is approximately met
    test_groups = set()
    current_test_count = 0

    for g in unique_groups:
        count = len(group_map[g])
        if current_test_count + count <= target_test_count or current_test_count == 0:
            test_groups.add(g)
            current_test_count += count
        else:
            # Check if adding exceeds by less than stopping
            if abs((current_test_count + count) - target_test_count) < abs(current_test_count - target_test_count):
                test_groups.add(g)
                current_test_count += count
                break
            break

    dev_groups = [g for g in unique_groups if g not in test_groups]
    # In case test_groups took everything or dev_groups is empty
    if not dev_groups:
        dev_groups = [unique_groups[0]]
        test_groups.remove(unique_groups[0])

    # Assign 5 folds to dev_groups
    # Sort groups by size descending and greedily assign to fold with minimum items to ensure balanced folds
    dev_group_counts = sorted([(g, len(group_map[g])) for g in dev_groups], key=lambda x: -x[1])
    fold_assignments = {}
    fold_totals = {f: 0 for f in range(1, n_folds + 1)}

    for g, cnt in dev_group_counts:
        # Pick fold with minimum items
        best_fold = min(fold_totals, key=lambda f: fold_totals[f])
        fold_assignments[g] = best_fold
        fold_totals[best_fold] += cnt

    # Build manifest
    manifest_rows = []
    for r in rows:
        gid = r.get("duplicate_group", r["issue_key"])
        if gid in test_groups:
            split = "test"
            fold = ""
        else:
            split = "development"
            fold = fold_assignments[gid]

        manifest_rows.append({
            "issue_key": r["issue_key"],
            "project_id": r["project_id"],
            "split": split,
            "fold": fold,
            "duplicate_group": gid
        })

    # Save manifest
    fieldnames = ["issue_key", "project_id", "split", "fold", "duplicate_group"]
    for dest in [out_dir / "split_manifest.csv", proc_dir / "split_manifest.csv"]:
        with open(dest, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(manifest_rows)

    dev_n = sum(1 for m in manifest_rows if m["split"] == "development")
    test_n = sum(1 for m in manifest_rows if m["split"] == "test")
    print(f"[{project_key}] Splits created: {dev_n} development, {test_n} test ({test_n/total_issues*100:.1f}%) | Folds: {fold_totals}")

def main():
    for pkey in ASSIGNED_PROJECTS:
        make_project_splits(pkey, seed=42, test_frac=0.20, n_folds=5)

if __name__ == "__main__":
    main()
