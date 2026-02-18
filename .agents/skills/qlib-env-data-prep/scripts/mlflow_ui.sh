#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$SCRIPT_DIR"
while [ "$ROOT_DIR" != "/" ] && [ ! -f "$ROOT_DIR/pyproject.toml" ]; do
  ROOT_DIR="$(dirname "$ROOT_DIR")"
done

if [ ! -f "$ROOT_DIR/pyproject.toml" ]; then
  echo "Cannot locate project root (pyproject.toml not found)" >&2
  exit 1
fi

PORT="${1:-5000}"
if [ "${PORT}" = "--help" ] || [ "${PORT}" = "-h" ]; then
  echo "Usage: uv run .agents/skills/qlib-env-data-prep/scripts/mlflow_ui.sh [port]"
  exit 0
fi
HOST="${MLFLOW_HOST:-127.0.0.1}"
BACKEND_URI="file://${ROOT_DIR}/mlruns"

# macOS + gunicorn fork compatibility for mlflow ui
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES

cd "${ROOT_DIR}"
echo "Starting MLflow UI at http://${HOST}:${PORT}"
exec uv run mlflow ui --backend-store-uri "${BACKEND_URI}" --host "${HOST}" --port "${PORT}"
