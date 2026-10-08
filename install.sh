#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_ROOT="${NLLB_RUNTIME_ROOT:-$HOME/models/nllb_en_ml_runtime}"
VENV_PATH="${VENV_PATH:-$RUNTIME_ROOT/.venv}"

if [[ ! -d "$VENV_PATH" ]]; then
  echo "Virtual environment not found: $VENV_PATH" >&2
  echo "Create it first or set VENV_PATH." >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$VENV_PATH/bin/activate"

python -m pip install --upgrade pip
python -m pip install -r "$APP_DIR/requirements.txt"

if [[ ! -f "$APP_DIR/.env" ]]; then
  cp "$APP_DIR/.env.example" "$APP_DIR/.env"
fi

echo "Installation complete."
echo "Configuration: $APP_DIR/.env"
echo "Start command: $APP_DIR/run.sh"
