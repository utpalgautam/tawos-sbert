"""Step 1-2: profile, flag, clean one project. Raw CSV is never modified.

Writes to data/processed/<project_key>/:
  project_profile.json, review_flags.csv, exclusion_log.csv,
  labeled_clean.csv, unlabeled.csv
Automatic exclusions are limited to rows that cannot be used at all
(non-numeric label, empty text, repeated issue_key). Zero, fractional and
extreme labels are FLAGGED, not removed. Extra exclusions only come from
data/approved_exclusions.csv (issue_key,reason), maintained by the coordinator.
"""
import json
import re
import numpy as np
import pandas as pd
from .common import (build_input_text, text_source, normalize_for_dup,
                     project_key_from_issue_keys, load_project_rows,
                     processed_dir, REPO_ROOT)


def preprocess_project(cfg, project_id):
    p = load_project_rows(cfg, project_id)
    project_key = project_key_from_issue_keys(p["issue_key"])
    out = processed_dir(project_key)

    p["input_text"] = [build_input_text(t, d) for t, d in zip(p["title"], p["description"])]
    p["text_source"] = [text_source(t, d) for t, d in zip(p["title"], p["description"])]
    p["story_point"] = pd.to_numeric(p["story_point_raw"], errors="coerce")
    p["label_missing_in_source"] = p["story_point_raw"].isna()
    p["label_non_numeric"] = p["story_point_raw"].notna() & p["story_point"].isna()
    p["norm_text"] = p["input_text"].map(normalize_for_dup)

    # ---------- review flags (no deletion) ----------
    labeled_mask = ~p["label_missing_in_source"]
    flags = []

    def add(mask, reason):
        for _, r in p[mask].iterrows():
            flags.append({"issue_key": r["issue_key"], "project_id": project_id,
                          "story_point": r["story_point_raw"], "reason": reason})

    title_clean = p["title"].fillna("").astype(str).str.strip()
    add(labeled_mask & (title_clean.str.len() < 10), "title_shorter_than_10")
    add(labeled_mask & title_clean.map(lambda t: bool(re.search(r"(.)\1{5,}", t))), "title_repeated_char_6plus")
    add(labeled_mask & (p["text_source"] == "title-only"), "description_missing")
    add(labeled_mask & (p["text_source"] == "none"), "no_text")
    add(p["label_non_numeric"], "non_numeric_label")
    add(labeled_mask & (p["story_point"] == 0), "zero_label")
    add(labeled_mask & p["story_point"].notna() & (p["story_point"] % 1 != 0), "fractional_label")
    add(labeled_mask & (p["story_point"] > 100), "label_gt_100")
    add(labeled_mask & (p["story_point"] < 0), "negative_label")
    add(p.duplicated("issue_key", keep=False), "duplicate_issue_key")
    lab = p[labeled_mask & (p["norm_text"] != "")]
    conf = lab.groupby("norm_text")["story_point"].nunique()
    conf_texts = set(conf[conf > 1].index)
    add(labeled_mask & p["norm_text"].isin(conf_texts), "same_text_conflicting_labels")
    flags_df = pd.DataFrame(flags, columns=["issue_key", "project_id", "story_point", "reason"])
    flags_df.to_csv(out / "review_flags.csv", index=False)

    # ---------- exclusions ----------
    excl = []

    def exclude(mask, reason):
        for _, r in p[mask].iterrows():
            excl.append({"issue_key": r["issue_key"], "project_id": project_id,
                         "original_label": r["story_point_raw"], "reason": reason,
                         "decision": "excluded"})

    dup_extra = p.duplicated("issue_key", keep="first")
    exclude(dup_extra, "repeated_issue_key_kept_first")
    exclude(~dup_extra & (p["text_source"] == "none"), "empty_title_and_description")
    exclude(~dup_extra & p["label_non_numeric"], "non_numeric_label")
    approved_keys = set()
    ap_cfg = cfg["data"].get("approved_exclusions")
    if ap_cfg and (REPO_ROOT / ap_cfg).is_file():
        ap = pd.read_csv(REPO_ROOT / ap_cfg)
        for _, r in ap.iterrows():
            if r["issue_key"] in set(p["issue_key"]):
                approved_keys.add(r["issue_key"])
                orig = p.loc[p["issue_key"] == r["issue_key"], "story_point_raw"].iloc[0]
                excl.append({"issue_key": r["issue_key"], "project_id": project_id,
                             "original_label": orig,
                             "reason": f"coordinator_approved: {r.get('reason', '')}",
                             "decision": "excluded"})
    excl_df = pd.DataFrame(excl, columns=["issue_key", "project_id", "original_label", "reason", "decision"])
    excl_df.to_csv(out / "exclusion_log.csv", index=False)

    # ---------- output tables ----------
    keep = (~dup_extra) & (p["text_source"] != "none") & (~p["issue_key"].isin(approved_keys))
    cols = ["issue_key", "project_id", "title", "description", "issue_type",
            "input_text", "text_source", "story_point"]
    labeled_clean = p[keep & p["story_point"].notna()][cols]
    unlabeled = p[keep & p["label_missing_in_source"]][cols].drop(columns=["story_point"])
    labeled_clean.to_csv(out / "labeled_clean.csv", index=False)
    unlabeled.to_csv(out / "unlabeled.csv", index=False)

    sp = p.loc[labeled_mask, "story_point"].dropna()
    profile = {
        "project_id": int(project_id), "project_key": project_key,
        "total_rows": int(len(p)),
        "labeled_rows_raw": int(labeled_mask.sum()),
        "unlabeled_rows_raw": int(p["label_missing_in_source"].sum()),
        "missing_title": int(p["title"].isna().sum()),
        "missing_description": int(p["description"].isna().sum()),
        "non_numeric_labels": int(p["label_non_numeric"].sum()),
        "label_stats": sp.describe().to_dict() if len(sp) else {},
        "zero_labels": int((sp == 0).sum()),
        "fractional_labels": int((sp % 1 != 0).sum()),
        "labels_gt_100": int((sp > 100).sum()),
        "issue_type_counts": p.loc[labeled_mask, "issue_type"].value_counts(dropna=False).to_dict(),
        "flagged_rows_distinct": int(flags_df["issue_key"].nunique()) if len(flags_df) else 0,
        "excluded_rows": int(len(excl_df)),
        "labeled_clean_rows": int(len(labeled_clean)),
        "unlabeled_rows": int(len(unlabeled)),
        "preprocessing_version": cfg["experiment"]["preprocessing_version"],
    }
    with open(out / "project_profile.json", "w") as f:
        json.dump(profile, f, indent=2, default=str)
    print(json.dumps({k: v for k, v in profile.items() if k not in ("label_stats", "issue_type_counts")}, indent=2))
    return project_key, out
