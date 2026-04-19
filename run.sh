#!/usr/bin/env bash
# Convenience launcher for the dashboard.
# First run will create a virtualenv and install dependencies.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

if [[ ! -d .venv ]]; then
  echo "Creating virtual environment..."
  python3 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --upgrade pip >/dev/null
python -m pip install -r requirements.txt

if [[ ! -f .env ]]; then
  echo "WARNING: .env not found. Copy .env.example to .env and fill in your Cliniko API key."
  cp .env.example .env
fi

exec streamlit run dashboard/app.py
