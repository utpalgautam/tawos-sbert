import pandas as pd
import numpy as np
from pathlib import Path

path = Path("outputs/final_model/test_predictions.csv")
df = pd.read_csv(path)

df["signed_error"] = (
    df["predicted_story_points"] - df["actual_story_points"]
)
df["squared_error"] = df["signed_error"] ** 2

print("\n=== TARGET AND PREDICTION DISTRIBUTIONS ===")
print(df[["actual_story_points", "predicted_story_points"]].describe().to_string())

print("\n=== ACTUAL STORY-POINT VALUES ===")
print(df["actual_story_points"].value_counts().sort_index().to_string())

print("\n=== 20 LARGEST ABSOLUTE ERRORS ===")
cols = [
    "issue_key", "project_name", "actual_story_points",
    "predicted_story_points", "signed_error", "absolute_error"
]
print(
    df.sort_values("absolute_error", ascending=False)[cols]
      .head(20).to_string(index=False)
)

print("\n=== LARGEST ACTUAL STORY-POINT VALUES ===")
print(
    df.sort_values("actual_story_points", ascending=False)[cols]
      .head(15).to_string(index=False)
)

print("\n=== PER-PROJECT TARGET RANGES ===")
summary = df.groupby("project_name").agg(
    n=("actual_story_points", "size"),
    actual_min=("actual_story_points", "min"),
    actual_median=("actual_story_points", "median"),
    actual_max=("actual_story_points", "max"),
    predicted_min=("predicted_story_points", "min"),
    predicted_max=("predicted_story_points", "max"),
    mean_absolute_error=("absolute_error", "mean"),
)
print(summary.to_string(float_format=lambda x: f"{x:.3f}"))

print("\n=== NEGATIVE PREDICTIONS ===")
negative = df[df["predicted_story_points"] < 0]
print(f"Count: {len(negative)} of {len(df)}")
if len(negative):
    print(negative[cols].head(20).to_string(index=False))

print("\n=== ERROR CONTRIBUTION BY PROJECT ===")
contrib = df.groupby("project_name").agg(
    squared_error_sum=("squared_error", "sum"),
    n=("squared_error", "size"),
)
contrib["share_of_total_squared_error_pct"] = (
    100 * contrib["squared_error_sum"] / df["squared_error"].sum()
)
print(
    contrib.sort_values("squared_error_sum", ascending=False)
           .to_string(float_format=lambda x: f"{x:.2f}")
)
