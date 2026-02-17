#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PORT="${1:-5000}"
HOST="${MLFLOW_HOST:-127.0.0.1}"
BACKEND_URI="file://${ROOT_DIR}/mlruns"

# macOS + gunicorn fork compatibility for mlflow ui
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES

cd "${ROOT_DIR}"
echo "Starting MLflow UI at http://${HOST}:${PORT}"
exec uv run mlflow ui --backend-store-uri "${BACKEND_URI}" --host "${HOST}" --port "${PORT}"
