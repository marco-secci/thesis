#!/usr/bin/env bash
# Runs the whole pipeline: installs the locked environment with uv, then hands
# over to scripts/run_pipeline.py. Options are passed through, e.g.
#   ./run_pipeline.sh --list
#   ./run_pipeline.sh --from forecast
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed. Install it with:"
  echo "    curl -LsSf https://astral.sh/uv/install.sh | sh"
  echo "then open a new terminal and run this script again."
  exit 1
fi

echo "Setting up the Python environment (uv sync)..."
uv sync --frozen --extra dev --quiet
exec uv run --frozen python scripts/run_pipeline.py "$@"
