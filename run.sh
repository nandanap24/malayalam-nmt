#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_ROOT="${NLLB_RUNTIME_ROOT:-/mnt/d/aug_data/models/nllb_en_ml_runtime}"
VENV_PATH="${VENV_PATH:-$HOME/nllb_fastapi_env}"

if [[ ! -d "$VENV_PATH" ]]; then
  echo "Virtual environment not found: $VENV_PATH" >&2
  exit 1
fi

if [[ ! -f "$APP_DIR/.env" ]]; then
  cp "$APP_DIR/.env.example" "$APP_DIR/.env"
  echo "Created $APP_DIR/.env"
fi

source "$VENV_PATH/bin/activate"

cd "$APP_DIR"

exec uvicorn app:app \
  --host 0.0.0.0 \
  --port 8000