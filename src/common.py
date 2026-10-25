"""Shared helpers: config loading, text cleaning, project paths."""
from pathlib import Path
import re
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_config(path="configs/problem5_sbert_cnn.yaml"):
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    with open(p) as f:
        return yaml.safe_load(f)


def clean_text(value):
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1]
    return " ".join(text.split())


def build_input_text(title, description):
    """Common model input (Section 5 of COMMON_PIPELINE.md)."""
    title = clean_text(title)
    description = clean_text(description)
    if title and description:
        return f"Title: {title}\nDescription: {description}"
    if title:
        return f"Title: {title}"
    if description:
        return f"Description: {description}"
    return ""


def text_source(title, description):
    t, d = bool(clean_text(title)), bool(clean_text(description))
    if t and d:
        return "title+description"
    if t:
        return "title-only"
    if d:
        return "description-only"
    return "none"


def normalize_for_dup(text):
    return re.sub(r"\s+", " ", text.lower()).strip()


def project_key_from_issue_keys(keys):
    s = pd.Series(keys).dropna().astype(str).str.split("-").str[0]
    return s.mode().iloc[0] if not s.empty else "UNKNOWN"


def load_project_rows(cfg, project_id):
    """Read only the needed columns from the raw CSV and filter one project."""
    d = cfg["data"]
    cols = [d["issue_key_column"], d["project_column"], d["title_column"],
            d["description_column"], d["target_column"], d["issue_type_column"]]
    path = Path(d["path"])
    if not path.is_absolute():
        path = REPO_ROOT / path
    df = pd.read_csv(path, usecols=cols, low_memory=False)
    p = df[df[d["project_column"]] == project_id].copy()
    if p.empty:
        raise ValueError(f"No rows for project_id={project_id}")
    p = p.rename(columns={
        d["issue_key_column"]: "issue_key", d["project_column"]: "project_id",
        d["title_column"]: "title", d["description_column"]: "description",
        d["target_column"]: "story_point_raw", d["issue_type_column"]: "issue_type"})
    return p.reset_index(drop=True)


def processed_dir(project_key):
    d = REPO_ROOT / "data" / "processed" / project_key
    d.mkdir(parents=True, exist_ok=True)
    return d


def output_dir(member, project_key):
    d = REPO_ROOT / "outputs" / member / project_key
    d.mkdir(parents=True, exist_ok=True)
    return d
