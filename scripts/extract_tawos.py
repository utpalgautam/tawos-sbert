#!/usr/bin/env python3

import csv
import sys
import time
from pathlib import Path

from sqlglot import parse_one, exp

BASE_DIR = Path.home() / "tawos-sbert"
SQL_FILE = Path.home() / "Downloads" / "21308124" / "TAWOS.sql"
OUTPUT_DIR = BASE_DIR / "data" / "raw"
OUTPUT_FILE = OUTPUT_DIR / "issues_extracted.csv"
ERROR_FILE = BASE_DIR / "logs" / "extraction_errors.log"

FIELDS = {
    "issue_id": 0,
    "jira_id": 1,
    "issue_key": 2,
    "title": 4,
    "description": 5,
    "description_text": 6,
    "issue_type": 8,
    "priority": 9,
    "status": 10,
    "resolution": 11,
    "creation_date": 12,
    "estimation_date": 13,
    "resolution_date": 14,
    "story_point": 16,
    "timespent": 17,
    "in_progress_minutes": 18,
    "total_effort_minutes": 19,
    "resolution_time_minutes": 20,
    "title_changed_after_estimation": 21,
    "description_changed_after_estimation": 22,
    "story_point_changed_after_estimation": 23,
    "project_id": 28,
    "sprint_id": 29,
}

COLUMNS = list(FIELDS.keys())
EXPECTED_COLUMNS = 30


def convert(node):
    if isinstance(node, exp.Null):
        return None
    if isinstance(node, exp.Literal):
        return node.to_py()
    if isinstance(node, exp.Boolean):
        return bool(node.this)
    return node.sql(dialect="mysql")


def main():
    if not SQL_FILE.is_file():
        sys.exit(f"SQL dump not found: {SQL_FILE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ERROR_FILE.parent.mkdir(parents=True, exist_ok=True)

    temporary = OUTPUT_FILE.with_suffix(".csv.tmp")
    started = time.time()
    statements = written = missing_points = bad_rows = parse_errors = 0

    print(f"Input:  {SQL_FILE}")
    print(f"Output: {OUTPUT_FILE}")
    print("Processing SQL dump. This may take a while.")

    with SQL_FILE.open("r", encoding="utf-8", errors="replace") as source, \
         temporary.open("w", newline="", encoding="utf-8") as output, \
         ERROR_FILE.open("w", encoding="utf-8") as error_log:

        writer = csv.DictWriter(output, fieldnames=COLUMNS)
        writer.writeheader()

        for line_number, line in enumerate(source, start=1):
            if not line.startswith("INSERT INTO `Issue`"):
                continue

            statements += 1

            try:
                statement = parse_one(line, read="mysql")
                rows = statement.expression.expressions

                for tuple_number, row in enumerate(rows, start=1):
                    values = row.expressions

                    if len(values) != EXPECTED_COLUMNS:
                        bad_rows += 1
                        error_log.write(
                            f"Line {line_number}, tuple {tuple_number}: "
                            f"expected 30 columns, got {len(values)}\n"
                        )
                        continue

                    try:
                        record = {
                            name: convert(values[index])
                            for name, index in FIELDS.items()
                        }
                        writer.writerow(record)
                        written += 1

                        if record["story_point"] is None:
                            missing_points += 1

                    except Exception as exc:
                        bad_rows += 1
                        error_log.write(
                            f"Line {line_number}, tuple {tuple_number}: "
                            f"{type(exc).__name__}: {exc}\n"
                        )

            except Exception as exc:
                parse_errors += 1
                error_log.write(
                    f"Line {line_number}: {type(exc).__name__}: {exc}\n"
                )

            if statements % 100 == 0:
                elapsed = (time.time() - started) / 60
                print(
                    f"\rStatements: {statements:,} | "
                    f"Rows: {written:,} | Elapsed: {elapsed:.1f} min",
                    end="",
                    flush=True,
                )

    if written == 0:
        temporary.unlink(missing_ok=True)
        sys.exit("No rows extracted. Check the error log.")

    temporary.replace(OUTPUT_FILE)

    print("\n\nExtraction finished.")
    print(f"INSERT statements: {statements:,}")
    print(f"Rows extracted: {written:,}")
    print(f"Rows missing story points: {missing_points:,}")
    print(f"Malformed rows: {bad_rows:,}")
    print(f"SQL parsing errors: {parse_errors:,}")
    print(f"Elapsed: {(time.time() - started) / 60:.2f} minutes")
    print(f"CSV: {OUTPUT_FILE}")
    print(f"Error log: {ERROR_FILE}")


if __name__ == "__main__":
    main()
