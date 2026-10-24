# TAWOS Problem 5 --- Common SBERT + CNN Pipeline

**Purpose:** Implement Problem 5 from the NLP project guidelines: SBERT
sentence embeddings for within-project story-point estimation using
random splitting.

**Raw dataset:** `tawos-sbert/data/raw/tawos.csv`\
**Members:** Utpal, Simran, Neha, Devanarayanan, Animesh\
**Primary model:** Fixed pretrained SBERT sentence embeddings +
three-layer CNN regression head\
**Required metrics:** MAE and Median Absolute Error (MdAE)

> **Critical interpretation:** Each assigned project must have its own
> within-project experiment. For example, DM's primary Problem 5 model
> is trained and tested on DM issues only. A member can automate several
> projects, but must not combine projects for the primary within-project
> result. Cross-project, temporal, BERT-word-embedding, and BERT+SBERT
> experiments are separate configurations in the assignment.

The guideline describes SBERT within-project random splitting for
Problem 5. It also specifies an 80:20 development/test split, five-fold
validation inside development data, a CNN prediction component as an
allowed option, and MAE/MdAE reporting. See the assignment's Problem 5
and common experimental requirements. fileciteturn6file0L91-L103
fileciteturn6file0L126-L149 fileciteturn6file0L246-L262

------------------------------------------------------------------------

## 1. Dataset verification

The earlier extraction summary reported 458,232 rows, 65,427 non-missing
story-point labels, 392,805 missing labels, 39 projects, and 23 columns.
These are initial counts; recalculate them from `data/raw/tawos.csv`. Do
not assume it is identical to the earlier `issues_extracted.csv` until
checked.

Run from the repository root:

``` bash
cd ~/tawos-sbert

python - <<'PY'
from pathlib import Path
import pandas as pd

path = Path("data/raw/tawos.csv")
if not path.exists():
    raise FileNotFoundError(f"Dataset not found: {path.resolve()}")

df = pd.read_csv(path)
print("Path:", path.resolve())
print("Rows:", len(df), "Columns:", len(df.columns))
print("Projects:", df["project_id"].nunique())
print("Labeled:", int(df["story_point"].notna().sum()))
print("Unlabeled:", int(df["story_point"].isna().sum()))
print("Columns:", df.columns.tolist())

required = {"issue_key", "project_id", "title", "description_text", "story_point"}
missing = required - set(df.columns)
if missing:
    print("WARNING — missing expected columns:", sorted(missing))
PY
```

If the file path or column names differ, stop and agree on the correct
schema. Do not silently substitute another CSV.

## 2. One common repository/output structure

``` text
tawos-sbert/
├── data/raw/tawos.csv
├── data/processed/<project_key>/
│   ├── labeled_clean.csv
│   ├── unlabeled.csv
│   ├── review_flags.csv
│   └── exclusion_log.csv
├── configs/problem5_sbert_cnn.yaml
├── src/
│   ├── inspect_data.py
│   ├── preprocess.py
│   ├── make_splits.py
│   ├── embed_sbert.py
│   ├── model.py
│   ├── train_cv.py
│   ├── evaluate.py
│   └── predict_unlabeled.py
└── outputs/<member>/<project_key>/
    ├── split_manifest.csv
    ├── cv_fold_metrics.csv
    ├── cv_summary.csv
    ├── best_config.json
    ├── final_model.pt
    ├── test_predictions.csv
    ├── test_metrics.json
    ├── unlabeled_predictions.csv
    └── run_metadata.json
```

The filenames are the proposed shared convention. Keep the raw file
unchanged and save every derived artifact separately.

## 3. Step 1 --- Project-level profiling

For each project in the allocation CSV:

1.  Filter using `project_id`.
2.  Count total, labeled, and unlabeled rows.
3.  Check missing/blank titles and descriptions.
4.  Inspect issue types and story-point distribution.
5.  Flag zero, fractional, extreme, and non-numeric labels.
6.  Check duplicate issue keys, repeated text, and conflicting labels.
7.  Save a profile before making exclusions.

Example:

``` python
import pandas as pd

df = pd.read_csv("data/raw/tawos.csv")
project_id = 28  # Replace with assigned project ID.
p = df[df["project_id"] == project_id].copy()
labeled = p[p["story_point"].notna()].copy()

print("Total:", len(p))
print("Labeled:", len(labeled))
print("Unlabeled:", p["story_point"].isna().sum())
print("Missing titles:", p["title"].isna().sum())
print("Missing descriptions:", p["description_text"].isna().sum())
print(labeled["issue_type"].value_counts(dropna=False))
print(pd.to_numeric(labeled["story_point"], errors="coerce").describe())
```

Do not delete rows during profiling.

## 4. Step 2 --- Common data-cleaning policy

All members must apply the same rules.

### Text cleaning

-   Preserve the raw CSV.
-   Handle missing values before converting text to strings.
-   Trim leading/trailing whitespace and normalize repeated
    whitespace/newlines.
-   Remove a surrounding quote pair only if it is demonstrably an
    extraction wrapper.
-   Preserve meaningful punctuation, code, file paths, identifiers, and
    technical terms.
-   Do not apply stemming, lemmatization, or stop-word removal by
    default.
-   Keep title and description columns available separately.
-   Use the common input construction in Section 5.

### Missing text

-   If title and description both contain useful text, use both.
-   If only one is available, use that field and record it as title-only
    or description-only.
-   If both are empty, flag the issue; exclude from text-based training
    unless the coordinator approves a different rule.
-   Do not discard a row simply because its description is missing if
    its title is informative.

### Labels

-   Convert story points to numeric and report any conversion failures.
-   Do not automatically remove zero or fractional values.
-   Do not cap all labels above 100.
-   Review extreme values in context: issue key, title, description,
    issue type, and project.
-   Exclude a record only for a documented reason.
-   Record each exclusion with `issue_key`, `project_id`, original
    label, reason, and decision.

### Duplicates

-   Check duplicate `issue_key` values within each project.
-   Repeated titles do not automatically mean duplicate issues.
-   Similar issues across projects may be legitimate; do not remove them
    just because they match.
-   Investigate same-project repeated text with conflicting labels.
-   Group exact duplicate text within a project before splitting, so
    copies cannot land in both development and test sets.
-   Record duplicate groups in the split manifest when applicable.

### Leakage

Do not feed `story_point`, time-spent, total-effort, resolution-time,
final status/resolution, issue key, or post-estimation change flags into
the model. These are targets, identifiers, outcome-related metadata, or
potential leakage. The primary input should be issue text available for
estimation.

The source's `description_changed_after_estimation` flag was zero for
all rows in an earlier audit. That does not prove descriptions were
historically unchanged; document this limitation.

Save two separate outputs: - `labeled_clean.csv`: eligible labeled
issues. - `unlabeled.csv`: issues whose original story point is missing.

Never write model predictions back into the original `story_point`
column.

## 5. Step 3 --- Standard model input

Use this exact input template for all members and all splits:

``` text
Title: <clean title>
Description: <clean description>
```

If one field is blank, use the available field. If both are blank,
flag/exclude according to the shared rule. Do not include project ID,
issue key, story-point label, resolution, or time/effort fields in the
input.

Example helper:

``` python
import pandas as pd

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
```

Use the same function in training, validation, testing, and unlabeled
inference.

## 6. Step 4 --- Required Problem 5 split

For each project independently:

1.  Use eligible labeled data after agreed cleaning.
2.  Reserve **20% as a held-out final test set**.
3.  Use the other **80% as development data**.
4.  Divide the development set into five folds.
5.  For each fold, train on four folds and validate on the fifth.
6.  Rotate so every fold serves as validation once.
7.  Select settings using development folds only.
8.  Refit a fresh model on the full development set using the selected
    configuration/training-duration rule.
9.  Evaluate once on the locked test set.

Use a documented random seed. Save a split manifest with at least:

``` text
issue_key,project_id,split,fold,duplicate_group
```

`split` should identify `development` or `test`; development rows also
have fold IDs 1--5. The test rows must not be used for early stopping,
model selection, or hyperparameter tuning.

For small projects, five-fold estimates may be unstable or a fold may
contain too few examples. Check sample counts first. Report the
limitation and consult the coordinator/instructor before changing the
required design. Do not silently use a different split.

## 7. Step 5 --- SBERT sentence embeddings

Problem 5 uses **sentence-level SBERT embeddings**. Do not use BERT
token-level embeddings or BERT+SBERT fusion in this experiment; those
are separate assignment problems.

### Proposed common encoder

`sentence-transformers/all-MiniLM-L6-v2`

This is a proposed team choice, not a model specified by the assignment.
The coordinator must approve and freeze the exact checkpoint/revision
for all members.

### Embedding procedure

1.  Encode the common input text using the same Sentence Transformers
    checkpoint.
2.  Use the same encoding settings across projects and splits.
3.  Run the frozen encoder in evaluation mode.
4.  Verify the embedding dimension and check for NaN/Inf.
5.  Cache embeddings and retain a row mapping to issue key, project ID,
    split, and preprocessing version.
6.  Do not fine-tune SBERT on test examples or test labels.

For `all-MiniLM-L6-v2`, the expected dimension is 384; verify this in
code. Include the encoder checkpoint, version/revision, embedding
dimension, and preprocessing version in metadata. If the encoder is kept
frozen, cached embeddings can be reused for CNN experiments.

## 8. Step 6 --- Common three-layer CNN regression architecture

The guideline permits a three-layer CNN on top of extracted
representations but does not mandate specific channel sizes. The
following is the **proposed shared implementation**; freeze it before
full runs.

### Input

SBERT produces a vector of shape `(batch_size, 384)`. Reshape it to
`(B, 1, 384)` for Conv1D. This CNN operates over the dimensions of the
sentence vector; it is not a token-level CNN.

  Layer                      Configuration                         Output shape
  -------------------------- ------------------------------------- -----------------
  Input                      Reshape SBERT vector                  `(B, 1, 384)`
  Conv1D 1                   1→128 channels, kernel=3, padding=1   `(B, 128, 384)`
  ReLU + MaxPool             pool size=2                           `(B, 128, 192)`
  Conv1D 2                   128→64, kernel=3, padding=1           `(B, 64, 192)`
  ReLU + MaxPool             pool size=2                           `(B, 64, 96)`
  Conv1D 3                   64→32, kernel=3, padding=1            `(B, 32, 96)`
  ReLU                       activation                            `(B, 32, 96)`
  Adaptive average pool      output length=1                       `(B, 32, 1)`
  Flatten                    ---                                   `(B, 32)`
  Dropout                    p=0.2                                 `(B, 32)`
  Linear regression output   32→1                                  `(B, 1)`

### Proposed initial training settings

  Setting                  Value
  ------------------------ ------------------------------------------
  SBERT checkpoint         `sentence-transformers/all-MiniLM-L6-v2`
  SBERT fine-tuning        Frozen for baseline
  CNN                      Three Conv1D layers above
  Loss                     MSE
  Optimizer                Adam
  CNN learning rate        `1e-3`
  Batch size               64 initially; reduce if memory requires
  Maximum epochs           100
  Early stopping           Patience 10 on validation MAE
  Dropout                  0.2
  Random seed              42
  Selection                Lowest mean validation MAE across folds
  Required final metrics   MAE and MdAE

These are team-selected initial values, not exact requirements from the
PDF. The coordinator must freeze the configuration. Document any
approved change.

Story-point labels are skewed and include extreme values. For the
baseline, train on original numeric labels and do not silently cap them.
If a log transform or robust loss is explored later, treat it as a
separate experiment, fit transformations using training data only,
invert predictions to the original scale, and compare using identical
splits.

## 9. Step 7 --- Cross-validation and final fitting

For each of the five development folds:

1.  Train on four folds.
2.  Validate on the remaining fold.
3.  Use validation MAE for early stopping/checkpoint selection.
4.  Save fold MAE, MdAE, best epoch, and sample counts.
5.  Repeat for all folds.

Report each fold's metrics and mean/standard deviation where feasible.

After configuration selection: 1. Initialize a fresh CNN. 2. Train on
all development data. 3. Use a predeclared training-duration rule based
on CV (e.g. median best epoch across folds). 4. Do not use the test set
for early stopping. 5. Evaluate once on the held-out test set.

Never present CV validation metrics as final test metrics.

## 10. Step 8 --- Evaluation and baseline

Required: - **MAE:** mean absolute difference between actual and
predicted story points. - **MdAE:** median absolute difference between
actual and predicted story points.

Use one implementation consistently, e.g. scikit-learn's
`mean_absolute_error` and `median_absolute_error`.

Report separately: 1. Fold-level MAE and MdAE. 2. Mean and standard
deviation across folds where feasible. 3. Final held-out test MAE and
MdAE. 4. Number of test examples. 5. Baseline metrics.

Use a simple training-median baseline: predict the median label from the
relevant training portion for every validation/test row. Never calculate
the baseline median from validation or test labels.

RMSE and R² may be supplementary, but do not replace MAE or MdAE.

Recommended plots: - Actual versus predicted story points on test
data. - Absolute-error distribution. - Label distribution. - Validation
MAE across folds.

## 11. Step 9 --- Unlabeled inference

After the model is selected and evaluated for a project:

1.  Load that project's unlabeled rows.
2.  Apply the same text construction and cleaning.
3.  Encode with the same frozen SBERT checkpoint.
4.  Run the project's final CNN model.
5.  Save a separate prediction file.

Suggested columns:

``` text
issue_key,project_id,title,predicted_story_point,encoder_checkpoint,model_version
```

Predictions are estimates, not verified historical labels. Keep original
missing labels missing. Do not invent a confidence score unless a
defined uncertainty method is implemented and documented.

## 12. Required deliverables for every project

-   `project_profile.csv` or `.json`: raw/labeled/unlabeled counts.
-   `review_flags.csv`: flagged rows and reasons.
-   `exclusion_log.csv`: excluded rows and justifications.
-   `labeled_clean.csv`: eligible labeled examples.
-   `unlabeled.csv`: missing-label examples for inference.
-   `split_manifest.csv`: development/test/fold assignment.
-   `embedding_metadata.json`: encoder, dimension, preprocessing
    version.
-   `cv_fold_metrics.csv`: per-fold metrics.
-   `cv_summary.csv`: mean and standard deviation.
-   `best_config.json`: selected configuration.
-   `final_model.pt`: final CNN checkpoint.
-   `test_predictions.csv`: issue key, actual, prediction, absolute
    error.
-   `test_metrics.json`: MAE, MdAE, sample count, baseline.
-   `unlabeled_predictions.csv`: inferred values for missing-label
    issues.
-   `run_metadata.json`: seeds, package versions, dataset/model
    versions.
-   A short report describing cleaning, decisions, results, and
    limitations.

Suggested metric schemas:

``` text
project_id,fold,mae,mdae,best_epoch,train_count,validation_count,seed
```

``` text
project_id,mae,mdae,test_count,baseline_mae,baseline_mdae,encoder_checkpoint,model_version,seed
```

## 13. Shared configuration file

The coordinator should maintain one versioned configuration, for
example:

``` yaml
experiment:
  name: problem5_sbert_cnn_within_project_random
  random_seed: 42
  project_setting: within_project
  split_strategy: random
  development_fraction: 0.80
  test_fraction: 0.20
  validation_folds: 5

data:
  path: data/raw/tawos.csv
  project_column: project_id
  issue_key_column: issue_key
  title_column: title
  description_column: description_text
  target_column: story_point

sbert:
  checkpoint: sentence-transformers/all-MiniLM-L6-v2
  fine_tune: false
  expected_embedding_dim: 384

cnn:
  conv_channels: [128, 64, 32]
  kernel_size: 3
  padding: 1
  dropout: 0.2
  output_dim: 1

training:
  loss: mse
  optimizer: adam
  learning_rate: 0.001
  batch_size: 64
  max_epochs: 100
  early_stopping_patience: 10
  early_stopping_metric: validation_mae

evaluation:
  required_metrics: [mae, mdae]
  baseline: training_median
```

Freeze the configuration and implementation before the full runs. Save
package versions and all seeds. Colab can be used for GPU training, but
checkpoints and results must be copied to persistent storage
(e.g. Drive).

## 14. Milestones

1.  **Verify and profile:** confirm the correct CSV, schema, and project
    counts.
2.  **Audit:** flag text problems, duplicates, label anomalies, and
    missing values.
3.  **Agree cleaning rules:** coordinator approves exclusions before
    full cleaning.
4.  **Prepare data:** save labeled/unlabeled files and split manifests.
5.  **Smoke test:** run one small end-to-end experiment before launching
    all projects.
6.  **Cross-validation:** complete five-fold validation on development
    data.
7.  **Final test:** refit using development data and evaluate the locked
    test set.
8.  **Inference:** predict for unlabeled issues.
9.  **Consolidate:** combine project metrics and document limitations.

## 15. Final checklist

-   [ ] Correct dataset path and schema verified.
-   [ ] Experiment is Problem 5: SBERT + within-project + random split.
-   [ ] Each project is modeled and evaluated independently.
-   [ ] Raw CSV unchanged.
-   [ ] Every exclusion documented.
-   [ ] Zero/fractional/extreme labels handled by agreed rules.
-   [ ] Same input-text function used in every split.
-   [ ] Same approved SBERT checkpoint used by all members.
-   [ ] Three-layer CNN matches the shared configuration.
-   [ ] 80% development / 20% locked test split created.
-   [ ] Five-fold validation performed inside development data.
-   [ ] Test data never used for tuning or early stopping.
-   [ ] MAE and MdAE reported for folds and final test.
-   [ ] Training-median baseline reported.
-   [ ] Predictions and metadata saved.
-   [ ] Unlabeled predictions stored separately.
-   [ ] Code/configuration/results committed to the team repository.

## Scope boundary

This document covers **Problem 5 only**. BERT word embeddings, combined
BERT+SBERT representations, cross-project evaluation, and temporal
splits belong to separate problem configurations in the assignment. Keep
those experiments separate and do not mix their results with the primary
Problem 5 results.

**Common rule:** one versioned pipeline, one approved configuration, one
project-specific experiment at a time, reproducible splits, and real
measured results only.
