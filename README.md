# TAWOS Issue Dataset README

## 1. Overview

This dataset was extracted from the TAWOS SQL database dump and is
intended for research on software issue story-point estimation using
issue text and machine-learning models such as SBERT + CNN.

The extracted CSV combines issues from **39 projects** into one file. It
includes issues both with and without a recorded story-point value.

### Dataset at a glance

  Item                                                                    Value
  ------------------------------------------- ---------------------------------
  Source SQL dump                                                   `TAWOS.sql`
  Approximate source dump size                                           4.1 GB
  Extracted CSV                                 `data/raw/issues_extracted.csv`
  Total extracted issues                                            **458,232**
  Issues with a recorded story-point value                           **65,427**
  Issues with no recorded story-point value                         **392,805**
  Projects represented                                                   **39**
  Columns in the extracted CSV                                           **23**
  SQL `INSERT` statements processed                                       1,004
  Malformed rows reported by extraction                                       0
  SQL parsing errors reported by extraction                                   0

**Important:** "Unlabeled" means the extracted `story_point` value is
missing. It does not establish why the value is missing in the original
source. Treat these records as missing-label examples unless the source
documentation establishes a more specific meaning.

## 2. Files and provenance

The extraction was performed locally without starting a MySQL or MariaDB
server. A Python script used SQLGlot to parse the SQL dump and extract
rows from the `Issue` table into CSV.


The CSV is the extracted dataset. It is not yet the final cleaned or
model-ready dataset. Keep the original extracted CSV unchanged and
create separate processed files for all cleaning, filtering, splitting,
and model-generated predictions.

## 3. CSV schema: all 23 columns

The extracted CSV has the following columns in this order.

  -------------------------------------------------------------------------------------------
                            \# Column                                   Description /
                                                                        interpretation
  ---------------------------- ---------------------------------------- ---------------------
                             1 `issue_id`                               Issue row ID from the
                                                                        source data.

                             2 `jira_id`                                Jira-related
                                                                        identifier from the
                                                                        source data.

                             3 `issue_key`                              Human-readable issue
                                                                        key, such as
                                                                        `XD-3768`.

                             4 `title`                                  Issue title/summary.

                             5 `description`                            Original description
                                                                        field extracted from
                                                                        the source.

                             6 `description_text`                       Description text
                                                                        field extracted from
                                                                        the source; often the
                                                                        most convenient text
                                                                        field to inspect for
                                                                        modelling. Confirm
                                                                        its semantics before
                                                                        assuming it is
                                                                        equivalent to
                                                                        `description`.

                             7 `issue_type`                             Issue type, for
                                                                        example Bug,
                                                                        Improvement, Epic, or
                                                                        Sub-task where
                                                                        present.

                             8 `priority`                               Priority field from
                                                                        the source.

                             9 `status`                                 Issue status from the
                                                                        source.

                            10 `resolution`                             Resolution field from
                                                                        the source.

                            11 `creation_date`                          Issue creation
                                                                        date/time field.

                            12 `estimation_date`                        Estimation date/time
                                                                        field.

                            13 `resolution_date`                        Resolution date/time
                                                                        field.

                            14 `story_point`                            Story-point target.
                                                                        Missing values are
                                                                        counted as unlabeled
                                                                        issues.

                            15 `timespent`                              Time-spent field from
                                                                        the source.

                            16 `in_progress_minutes`                    In-progress duration,
                                                                        in minutes as named
                                                                        by the source field.

                            17 `total_effort_minutes`                   Total-effort
                                                                        duration, in minutes
                                                                        as named by the
                                                                        source field.

                            18 `resolution_time_minutes`                Resolution-time
                                                                        duration, in minutes
                                                                        as named by the
                                                                        source field.

                            19 `title_changed_after_estimation`         Source flag
                                                                        indicating whether
                                                                        the title changed
                                                                        after estimation.

                            20 `description_changed_after_estimation`   Source flag
                                                                        indicating whether
                                                                        the description
                                                                        changed after
                                                                        estimation.

                            21 `story_point_changed_after_estimation`   Source flag
                                                                        indicating whether
                                                                        the story-point value
                                                                        changed after
                                                                        estimation.

                            22 `project_id`                             Project identifier;
                                                                        used to associate
                                                                        each issue with a
                                                                        project.

                            23 `sprint_id`                              Sprint identifier,
                                                                        where present.
  -------------------------------------------------------------------------------------------

### Notes about interpretation

-   The names above describe the extracted columns, not a guarantee that
    every source field is complete or semantically reliable.
-   The `description_changed_after_estimation` flag was `0` for all
    458,232 extracted rows in the analysis performed so far. This should
    be investigated; do not assume it proves that every description
    remained unchanged.
-   A flag value of `0` does not independently prove that an issue's
    text or label is suitable for training.
-   Text values may include surrounding quotation marks or other
    extraction artifacts. Inspect and clean text in a separate processed
    dataset; do not overwrite the raw CSV.
-   Missing `story_point` values should remain missing in the source
    data. Do not fill them with model predictions in the `story_point`
    column.

## 4. Project inventory (sorted by `project_id`)

The table below lists all 39 projects identified in the extracted
dataset, sorted by project ID. "Labeled issues" is the count of issues
with a non-missing `story_point` value from the initial analysis.

    Project ID Project key   Project title/name                  Labeled issues
  ------------ ------------- --------------------------------- ----------------
             1 XD            Spring XD                                    3,705
             2 DATACASS      Spring DataCass                                243
             3 NEXUS         Sonatype Nexus                               1,845
             4 MESOS         Apache Mesos                                 3,272
             5 USERGRID      Apache Usergrid                                487
             6 MXNET         Apache MXNet                                   209
             7 ALOY          Alloy Framework                                315
             8 APSTUD        Aptana Studio                                  890
             9 CLI           Command-Line Interface                         374
            10 DAEMON        Appcelerator Daemon                            242
            11 TIDOC         Titanium Mobile Platform                     1,297
            12 TIMOB         The Titanium SDK                             4,665
            13 TISTUD        Appcelerator Studio                          3,406
            14 BAM           Atlassian Bamboo                               528
            15 CLOV          Atlassian Clover                               387
            16 CWD           Atlassian Crowd                                214
            17 FE            Atlassian FishEye                              240
            18 JRASERVER     Atlassian Jira Server                          380
            19 JSWCLOUD      Atlassian Software Cloud                       318
            20 JSWSERVER     Atlassian Software Server                      351
            21 CONFCLOUD     Atlassian Confluence Cloud                     352
            22 CONFSERVER    Atlassian Confluence Server                    662
            23 BE            Hyperledger Blockchain Explorer                373
            24 FAB           Hyperledger Fabric                             636
            25 INDY          Hyperledger Indy Node                          681
            26 IS            Hyperledger Indy SDK                           720
            27 STL           Apache Sawtooth                                966
            28 DM            LSST Data Management                        20,664
            29 DURACLOUD     Lyrasis DuraCloud                              666
            30 COMPASS       MongoDB Compass                                499
            31 CXX           The MongoDB C++ Driver                         224
            32 JAVA          The MongoDB Java Driver                        238
            33 SERVER        MongoDB Core Server                            784
            34 MDL           Moodle                                       1,594
            35 APIKIT        Mule APIkit                                    473
            36 MULE          Mule                                         4,170
            42 DNN           DotNetNuke Platform                          2,594
            43 EVG           The MongoDB Engineering                      5,402
            44 JRACLOUD      Atlassian Jira Cloud                           361
               **Total**     **39 projects**                         **65,427**

Project IDs are not consecutive; IDs 37--41 do not appear in this
extracted project inventory. The project names above reflect the
metadata used in the extraction analysis and may be shortened for
readability.

### Per-project total, labeled, and unlabeled issue counts

The labeled counts above are available from the completed dataset
analysis. The exact total and unlabeled counts for each project were not
included in that summary, so they are deliberately not guessed here.

Run the following command in the project directory to calculate the
exact counts from the CSV. It prints a project table sorted by
`project_id` and saves it as `reports/project_issue_counts.csv`.

``` bash
cd ~/tawos-sbert

mkdir -p reports

python - <<'PY'
import pandas as pd

df = pd.read_csv("data/raw/tawos.csv")

summary = (
    df.groupby("project_id", dropna=False)
      .agg(
          total_issues=("issue_id", "size"),
          labeled_issues=("story_point", lambda s: s.notna().sum()),
          unlabeled_issues=("story_point", lambda s: s.isna().sum()),
          project_key=("issue_key", lambda s: (
              s.dropna().astype(str).str.split("-").str[0].mode().iloc[0]
              if not s.dropna().empty else ""
          )),
      )
      .reset_index()
      .sort_values("project_id")
)

summary["labeled_percent"] = (
    100 * summary["labeled_issues"] / summary["total_issues"]
).round(2)

print(summary.to_string(index=False))
print("\\nGrand totals:")
print("Total issues:", len(df))
print("Labeled issues:", df["story_point"].notna().sum())
print("Unlabeled issues:", df["story_point"].isna().sum())
print("Unique projects:", df["project_id"].nunique())

summary.to_csv("reports/project_issue_counts.csv", index=False)
print("\\nSaved: reports/project_issue_counts.csv")
PY
```

## 5. Labeled and unlabeled issues

### Labeled issues: 65,427

These rows have a non-missing `story_point` value. They are the initial
candidate pool for supervised story-point prediction.

Before training, the team must still review data quality, define
inclusion/exclusion rules, and create train/validation/test splits.
Having a label does not automatically mean an issue is suitable for
training.

### Unlabeled issues: 392,805

These rows have a missing `story_point` value.

Potential uses include:

-   Predict story points after a model has been trained and evaluated.
-   Compare text characteristics between labeled and unlabeled issues.
-   Explore text similarity and project-specific vocabulary.
-   Investigate semi-supervised learning as a later experiment.

They cannot directly be used as ordinary ground-truth examples for
supervised regression unless labels are obtained separately.
Model-generated estimates must be stored separately from actual source
labels.

## 6. Initial story-point label profile

The initial analysis of the 65,427 labeled issues found:

  Statistic                        Value
  ----------------------------- --------
  Labeled issues                  65,427
  Distinct story-point values        367
  Median                               3
  25th percentile                      1
  75th percentile                      5
  90th percentile                      8
  95th percentile                     14
  99th percentile                     50
  Zero-point labels                2,416
  Fractional labels                3,337
  Labels greater than 100            141
  Negative labels                      0

The mean was approximately 413.57 and was heavily affected by extreme
values. The largest observed value was 26,710,114, associated with
`DATACASS-228`. Other very large values included `TIMOB-18052` (6,765),
`JRACLOUD-62509` (3,030), and `XD-2602` (2,567).

**Do not automatically cap all high values, delete all fractional
estimates, or remove all zero-point estimates.** Some large values may
be valid for certain projects or issue types, and zero/fractional values
may reflect legitimate conventions. Review and document any exclusions.

## 7. Initial data-quality findings

These findings come from the first audits and are intended to guide
further review; they are not final exclusion decisions.

-   70 labeled issues had titles shorter than 10 characters in the
    initial title-quality check.
-   Four titles were flagged for a character repeated six or more times.
-   2,131 rows appeared in groups with repeated normalized title and
    story-point combinations.
-   The broader audit flagged 8,312 labeled rows for one or more review
    reasons.
-   1,069 rows had titles repeated within the same project.
-   2,544 rows had titles repeated across projects.
-   459 rows were in same-project repeated-title groups with different
    story-point labels.
-   All 458,232 rows had `description_changed_after_estimation = 0` in
    the extracted data.

These counts can overlap. A row may be flagged for several reasons, so
the counts must not be added together to calculate the number of
problematic issues.

Repeated text across projects is not automatically invalid. The same or
similar issue can appear in related projects. For model evaluation,
however, duplicates within a project should be considered when creating
splits to avoid overly optimistic test results.

## 8. Extraction and integrity notes

The extraction completed with the following reported results:

-   1,004 `INSERT` statements processed.
-   458,232 issue rows extracted.
-   0 malformed rows reported.
-   0 SQL parsing errors reported.
-   Extraction time approximately 6.08 minutes.

These results indicate that the extraction script reported no parsing
failures or malformed rows. They do not independently prove that every
field has the correct semantic interpretation or that the source dataset
contains no quality issues.

The source SQL `Issue` table had 30 columns; the extracted CSV contains
the 23 selected fields documented above.

------------------------------------------------------------------------

**Status:** Extraction and initial metadata analysis are complete.
Per-project total/labeled/unlabeled counts, final cleaning decisions,
train/validation/test splits, and model training remain follow-up tasks.
