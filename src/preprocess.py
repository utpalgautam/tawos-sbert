import re
from pathlib import Path
import numpy as np
import pandas as pd
from text_utils import clean_text, build_input_text

RAW = Path("data/raw/tawos.csv")
ALLOC = Path("project_allocation.csv")
MEMBER = "simran"
OUT = Path("data/processed")
REPEAT_RE = re.compile(r"(.)\1{5,}")

# Documented manual exclusions (pending coordinator approval).
MANUAL_EXCLUSIONS = {
    "JSWSERVER-10638": "non-informative gibberish text and implausible label (688, ~34x next-largest in project)",
}

alloc = pd.read_csv(ALLOC)
mine = alloc[alloc["assigned_member"].str.lower() == MEMBER]

cols = ["issue_key", "project_id", "title", "description_text",
        "issue_type", "story_point"]
df = pd.read_csv(RAW, usecols=cols)
df = df[df["project_id"].isin(mine["project_id"])].copy()

df["title_clean"] = df["title"].map(clean_text)
df["description_clean"] = df["description_text"].map(clean_text)
df["input_text"] = [build_input_text(t, d)
                    for t, d in zip(df["title"], df["description_text"])]
has_t = df["title_clean"] != ""
has_d = df["description_clean"] != ""
df["text_source"] = np.select(
    [has_t & has_d, has_t, has_d],
    ["title+description", "title-only", "description-only"], "none")
df["sp_num"] = pd.to_numeric(df["story_point"], errors="coerce")

summary = []
for _, row in mine.iterrows():
    pid, key = int(row["project_id"]), row["project_key"]
    p = df[df["project_id"] == pid].copy()
    unl = p[p["story_point"].isna()].copy()
    lab = p[p["story_point"].notna()].copy()

    # exact-duplicate text groups (within project), used later for splitting
    m = lab["input_text"] != ""
    lab["duplicate_group"] = ""
    codes = pd.factorize(lab.loc[m, "input_text"].str.lower())[0]
    lab.loc[m, "duplicate_group"] = ["G%05d" % c for c in codes]
    lab["dup_group_size"] = 0
    lab["dup_group_nlabels"] = 0
    g = lab.loc[m].groupby("duplicate_group")
    lab.loc[m, "dup_group_size"] = g["issue_key"].transform("size")
    lab.loc[m, "dup_group_nlabels"] = g["sp_num"].transform("nunique")

    flags = {
        "blank_text": lab["input_text"] == "",
        "non_numeric_label": lab["sp_num"].isna(),
        "short_title": lab["title_clean"].str.len() < 10,
        "repeated_char_title": lab["title_clean"].map(lambda s: bool(REPEAT_RE.search(s))),
        "zero_label": lab["sp_num"] == 0,
        "fractional_label": lab["sp_num"].notna() & (lab["sp_num"] % 1 != 0),
        "label_gt_100": lab["sp_num"] > 100,
        "duplicate_issue_key": lab["issue_key"].duplicated(keep=False),
        "repeated_text": lab["dup_group_size"] > 1,
        "repeated_text_conflicting_label":
            (lab["dup_group_size"] > 1) & (lab["dup_group_nlabels"] > 1),
    }

    reasons = pd.Series("", index=lab.index)
    for name, mask in flags.items():
        reasons = reasons.where(~mask, reasons + name + ";")
    reasons = reasons.str.rstrip(";")

    # Exclusions: no usable text, non-numeric label, or documented manual exclusion.
    excl = (flags["blank_text"] | flags["non_numeric_label"]
            | lab["issue_key"].isin(MANUAL_EXCLUSIONS.keys()))

    d = OUT / key
    d.mkdir(parents=True, exist_ok=True)

    manual = lab["issue_key"].map(MANUAL_EXCLUSIONS).fillna("")
    log = lab[excl].copy()
    log["original_label"] = log["story_point"]
    log["reason"] = (reasons[excl] + ";" + manual[excl]).str.strip(";")
    log["decision"] = "excluded"
    log[["issue_key", "project_id", "original_label", "reason", "decision"]] \
        .to_csv(d / "exclusion_log.csv", index=False)

    rf = lab[reasons != ""].copy()
    rf["flags"] = reasons[reasons != ""]
    rf["decision"] = np.where(excl[reasons != ""], "excluded", "kept-pending-review")
    rf[["issue_key", "project_id", "issue_type", "title", "story_point",
        "flags", "decision"]].to_csv(d / "review_flags.csv", index=False)

    kept = lab[~excl].copy()
    kept["story_point"] = kept["sp_num"]
    kept[["issue_key", "project_id", "issue_type", "title_clean",
          "description_clean", "input_text", "text_source", "story_point",
          "duplicate_group", "dup_group_size"]] \
        .to_csv(d / "labeled_clean.csv", index=False)

    unl[["issue_key", "project_id", "issue_type", "title_clean",
         "description_clean", "input_text", "text_source"]] \
        .to_csv(d / "unlabeled.csv", index=False)

    s = {"project_key": key, "labeled_raw": len(lab), "excluded": int(excl.sum()),
         "labeled_clean": len(kept), "unlabeled": len(unl)}
    s.update({f"flag_{k}": int(v.sum()) for k, v in flags.items()})
    summary.append(s)

    print(f"\n{key}: raw={len(lab)} excluded={int(excl.sum())} "
          f"clean={len(kept)} unlabeled={len(unl)}")
    print("  text_source:", kept["text_source"].value_counts().to_dict())
    print("  top labels:", lab.nlargest(5, "sp_num")[["issue_key", "sp_num"]]
          .values.tolist())

pd.DataFrame(summary).to_csv("reports/simran_cleaning_summary.csv", index=False)
print("\nSaved reports/simran_cleaning_summary.csv")