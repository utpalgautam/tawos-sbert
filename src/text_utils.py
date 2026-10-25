import pandas as pd

PREPROCESSING_VERSION = "v1"

def clean_text(value):
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1]
    return " ".join(text.split())

def build_input_text(title, description):
    title = clean_text(title)
    description = clean_text(description)
    if title and description:
        return f"Title: {title}\nDescription: {description}"
    if title:
        return f"Title: {title}"
    if description:
        return f"Description: {description}"
    return ""