#!/usr/bin/env bash
# Runner script for Problem 5 SBERT + CNN within-project experiments
# Assigned member: devanarayanan

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." >/dev/null 2>&1 && pwd )"
cd "$DIR"

echo "Activating virtual environment..."
source .venv/bin/activate

echo "Running full pipeline for devanarayanan assigned projects..."
python -m src.run_pipeline --member devanarayanan

echo "All assigned projects completed successfully!"
