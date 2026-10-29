#!/usr/bin/env python3
"""
Fast, robust profiling and preprocessing for assigned projects of devanarayanan.
Runs using standard library to avoid waiting for heavy dependencies.
Implements:
  - Step 1: Profiling & project_profile.json
  - Step 2: Data cleaning, exclusions log, review flags, duplicate grouping
  - Step 3: Standard input text construction
  - Saves labeled_clean.csv, unlabeled.csv, review_flags.csv, exclusion_log.csv, project_profile.json
"""

import csv
import json
import math
import os
from pathlib import Path
import sys
import time

# Increase CSV field size limit for massive descriptions in TAWOS
csv.field_size_limit(sys.maxsize)

ASSIGNED_PROJECTS = {
    4: ("MESOS", "Apache Mesos"),
    8: ("APSTUD", "Aptana Studio"),
    19: ("JSWCLOUD", "Atlassian Software Cloud"),
    24: ("FAB", "Hyperledger Fabric"),
    26: ("IS", "Hyperledger Indy SDK"),
    35: ("APIKIT", "Mule APIkit"),
    36: ("MULE", "Mule"),
    44: ("JRACLOUD", "Atlassian Jira Cloud"),
}

def clean_text(value):
    if value is None:
        return ""
    text = str(value).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1]
    return " ".join(text.split())

def build_input_text(title, description):
    c_title = clean_text(title)
    c_desc = clean_text(description)
    if c_title and c_desc:
        return f"Title: {c_title}\nDescription: {c_desc}"
    if c_title:
        return f"Title: {c_title}"
    if c_desc:
        return f"Description: {c_desc}"
    return ""

def compute_percentiles(values):
    if not values:
        return {}
    s = sorted(values)
    n = len(s)
    def pct(p):
        k = (n - 1) * (p / 100.0)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return float(s[int(k)])
        return float(s[f] * (c - k) + s[c] * (k - f))
    mean = float(sum(s) / n)
    variance = sum((x - mean) ** 2 for x in s) / (n - 1) if n > 1 else 0.0
    return {
        "count": n,
        "mean": mean,
        "std": math.sqrt(variance),
        "min": float(s[0]),
        "p25": pct(25),
        "median": pct(50),
        "p75": pct(75),
        "p90": pct(90),
        "p95": pct(95),
        "p99": pct(99),
        "max": float(s[-1]),
        "distinct_values": len(set(s))
    }

def main():
    raw_path = Path("data/raw/tawos.csv")
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data not found at {raw_path}")

    base_proc = Path("data/processed")
    base_out = Path("outputs/devanarayanan")
    base_proc.mkdir(parents=True, exist_ok=True)
    base_out.mkdir(parents=True, exist_ok=True)
    Path("reports").mkdir(parents=True, exist_ok=True)

    print(f"Reading raw issues from {raw_path}...")
    t0 = time.time()

    # Store raw rows per assigned project_id
    project_rows = {pid: [] for pid in ASSIGNED_PROJECTS}

    with open(raw_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                pid = int(row["project_id"])
                if pid in ASSIGNED_PROJECTS:
                    project_rows[pid].append(row)
            except (ValueError, KeyError):
                pass

    print(f"Extraction of assigned projects completed in {time.time() - t0:.2f}s.")

    profile_summary_list = []

    for pid, (pkey, pname) in ASSIGNED_PROJECTS.items():
        rows = project_rows[pid]
        p_proc_dir = base_proc / pkey
        p_out_dir = base_out / pkey
        p_proc_dir.mkdir(parents=True, exist_ok=True)
        p_out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nProcessing {pkey} ({pname}, ID: {pid}) - {len(rows)} issues...")

        # Profiling stats
        total_count = len(rows)
        raw_labeled = []
        raw_unlabeled = []
        missing_titles = 0
        blank_titles = 0
        missing_descriptions = 0
        blank_descriptions = 0
        both_blank = 0
        issue_types = {}
        all_issue_keys = set()
        dup_issue_keys_count = 0

        for r in rows:
            ik = r.get("issue_key", "")
            if ik in all_issue_keys:
                dup_issue_keys_count += 1
            else:
                all_issue_keys.add(ik)

            t = r.get("title", "")
            d = r.get("description_text", "")
            if t is None: missing_titles += 1
            if d is None: missing_descriptions += 1
            c_t = clean_text(t)
            c_d = clean_text(d)
            if not c_t: blank_titles += 1
            if not c_d: blank_descriptions += 1
            if not c_t and not c_d: both_blank += 1

            itype = r.get("issue_type", "Unknown")
            issue_types[itype] = issue_types.get(itype, 0) + 1

            sp_raw = r.get("story_point", "")
            if sp_raw is not None and str(sp_raw).strip() != "":
                raw_labeled.append(r)
            else:
                raw_unlabeled.append(r)

        # Parse numeric labels for profiling
        numeric_labels = []
        non_numeric_count = 0
        zero_count = 0
        fractional_count = 0
        extreme_count = 0
        negative_count = 0

        for r in raw_labeled:
            sp_val = r["story_point"]
            try:
                val = float(sp_val)
                numeric_labels.append(val)
                if val == 0: zero_count += 1
                elif val != int(val): fractional_count += 1
                if val > 100: extreme_count += 1
                if val < 0: negative_count += 1
            except ValueError:
                non_numeric_count += 1

        sp_stats = compute_percentiles(numeric_labels)

        # Profile dict
        profile = {
            "project_key": pkey,
            "project_id": pid,
            "project_name": pname,
            "total_issues": total_count,
            "labeled_issues": len(raw_labeled),
            "unlabeled_issues": len(raw_unlabeled),
            "missing_titles": missing_titles,
            "blank_titles": blank_titles,
            "missing_descriptions": missing_descriptions,
            "blank_descriptions": blank_descriptions,
            "both_text_blank": both_blank,
            "non_numeric_labels": non_numeric_count,
            "zero_labels": zero_count,
            "fractional_labels": fractional_count,
            "extreme_labels_gt_100": extreme_count,
            "negative_labels": negative_count,
            "duplicate_issue_keys": dup_issue_keys_count,
            "story_point_stats": sp_stats,
            "issue_type_distribution": issue_types
        }

        with open(p_out_dir / "project_profile.json", "w") as f:
            json.dump(profile, f, indent=2)

        profile_summary_list.append({
            "project_id": pid,
            "project_key": pkey,
            "project_name": pname,
            "total_issues": total_count,
            "labeled_issues": len(raw_labeled),
            "unlabeled_issues": len(raw_unlabeled),
            "zero_labels": zero_count,
            "fractional_labels": fractional_count,
            "extreme_labels": extreme_count,
            "median_sp": sp_stats.get("median", ""),
            "mean_sp": f"{sp_stats.get('mean', 0.0):.2f}" if sp_stats else ""
        })

        # Preprocessing & Data Cleaning
        exclusions = []
        review_flags = []
        labeled_clean = []
        seen_keys_in_labeled = set()

        for r in raw_labeled:
            ik = r.get("issue_key", "")
            orig_sp = r.get("story_point", "")

            # Check duplicate issue_key
            if ik in seen_keys_in_labeled:
                exclusions.append({
                    "issue_key": ik,
                    "project_id": pid,
                    "original_label": orig_sp,
                    "reason": "duplicate_issue_key",
                    "decision": "excluded_keep_first_occurrence"
                })
                review_flags.append({
                    "issue_key": ik,
                    "project_id": pid,
                    "flag_reason": "duplicate_issue_key",
                    "details": f"Duplicate issue key {ik}"
                })
                continue
            seen_keys_in_labeled.add(ik)

            # Check numeric story point
            try:
                num_sp = float(orig_sp)
            except ValueError:
                exclusions.append({
                    "issue_key": ik,
                    "project_id": pid,
                    "original_label": orig_sp,
                    "reason": "non_numeric_story_point",
                    "decision": "excluded_cannot_train_regression"
                })
                continue

            # Build standardized text
            inp_text = build_input_text(r.get("title", ""), r.get("description_text", ""))
            if not inp_text:
                exclusions.append({
                    "issue_key": ik,
                    "project_id": pid,
                    "original_label": orig_sp,
                    "reason": "empty_title_and_description",
                    "decision": "excluded_no_text_available"
                })
                continue

            # Review flags for valid issues
            if num_sp == 0:
                review_flags.append({
                    "issue_key": ik,
                    "project_id": pid,
                    "flag_reason": "zero_story_point",
                    "details": "Label is 0"
                })
            elif num_sp != int(num_sp):
                review_flags.append({
                    "issue_key": ik,
                    "project_id": pid,
                    "flag_reason": "fractional_story_point",
                    "details": f"Fractional label: {num_sp}"
                })
            if num_sp > 100:
                review_flags.append({
                    "issue_key": ik,
                    "project_id": pid,
                    "flag_reason": "extreme_story_point",
                    "details": f"Label > 100: {num_sp}"
                })

            c_t = clean_text(r.get("title", ""))
            c_d = clean_text(r.get("description_text", ""))
            if not c_t:
                review_flags.append({
                    "issue_key": ik,
                    "project_id": pid,
                    "flag_reason": "description_only",
                    "details": "Title is blank, description only"
                })
            elif not c_d:
                review_flags.append({
                    "issue_key": ik,
                    "project_id": pid,
                    "flag_reason": "title_only",
                    "details": "Description is blank, title only"
                })

            labeled_clean.append({
                "issue_id": r.get("issue_id", ""),
                "jira_id": r.get("jira_id", ""),
                "issue_key": ik,
                "project_id": pid,
                "title": r.get("title", ""),
                "description_text": r.get("description_text", ""),
                "input_text": inp_text,
                "story_point": num_sp,
                "issue_type": r.get("issue_type", ""),
                "priority": r.get("priority", ""),
                "status": r.get("status", "")
            })

        # Assign duplicate_group to identical input_text
        text_to_group = {}
        gid = 1
        for item in labeled_clean:
            t = item["input_text"]
            if t not in text_to_group:
                text_to_group[t] = gid
                gid += 1
            item["duplicate_group"] = text_to_group[t]

        # Check conflicting labels in identical text groups
        group_labels = {}
        for item in labeled_clean:
            g = item["duplicate_group"]
            group_labels.setdefault(g, set()).add(item["story_point"])
        for item in labeled_clean:
            g = item["duplicate_group"]
            if len(group_labels[g]) > 1:
                review_flags.append({
                    "issue_key": item["issue_key"],
                    "project_id": pid,
                    "flag_reason": "duplicate_text_conflicting_labels",
                    "details": f"duplicate_group {g} has conflicting labels {sorted(list(group_labels[g]))}"
                })

        # Save labeled_clean.csv
        labeled_fieldnames = [
            "issue_id", "jira_id", "issue_key", "project_id", "title", "description_text",
            "input_text", "story_point", "issue_type", "priority", "status", "duplicate_group"
        ]
        with open(p_proc_dir / "labeled_clean.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=labeled_fieldnames)
            writer.writeheader()
            writer.writerows(labeled_clean)

        # Preprocess unlabeled
        unlabeled_clean = []
        for r in raw_unlabeled:
            inp_text = build_input_text(r.get("title", ""), r.get("description_text", ""))
            unlabeled_clean.append({
                "issue_id": r.get("issue_id", ""),
                "jira_id": r.get("jira_id", ""),
                "issue_key": r.get("issue_key", ""),
                "project_id": pid,
                "title": r.get("title", ""),
                "description_text": r.get("description_text", ""),
                "input_text": inp_text,
                "issue_type": r.get("issue_type", ""),
                "priority": r.get("priority", ""),
                "status": r.get("status", "")
            })

        unlabeled_fieldnames = [
            "issue_id", "jira_id", "issue_key", "project_id", "title", "description_text",
            "input_text", "issue_type", "priority", "status"
        ]
        with open(p_proc_dir / "unlabeled.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=unlabeled_fieldnames)
            writer.writeheader()
            writer.writerows(unlabeled_clean)

        # Save exclusion_log.csv
        excl_fieldnames = ["issue_key", "project_id", "original_label", "reason", "decision"]
        for dest in [p_proc_dir / "exclusion_log.csv", p_out_dir / "exclusion_log.csv"]:
            with open(dest, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=excl_fieldnames)
                writer.writeheader()
                writer.writerows(exclusions)

        # Save review_flags.csv
        flags_fieldnames = ["issue_key", "project_id", "flag_reason", "details"]
        for dest in [p_proc_dir / "review_flags.csv", p_out_dir / "review_flags.csv"]:
            with open(dest, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=flags_fieldnames)
                writer.writeheader()
                writer.writerows(review_flags)

        print(f"  -> {len(labeled_clean)} clean labeled, {len(unlabeled_clean)} unlabeled, "
              f"{len(exclusions)} exclusions, {len(review_flags)} review flags.")

    # Save overall profile report
    with open("reports/profile_devanarayanan.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "project_id", "project_key", "project_name", "total_issues", "labeled_issues",
            "unlabeled_issues", "zero_labels", "fractional_labels", "extreme_labels", "median_sp", "mean_sp"
        ])
        writer.writeheader()
        writer.writerows(profile_summary_list)

    print("\n[SUCCESS] Completed profiling and preprocessing for all 8 assigned projects!")
    print("Report saved to reports/profile_devanarayanan.csv")

if __name__ == "__main__":
    main()
