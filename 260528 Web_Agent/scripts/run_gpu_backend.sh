#!/usr/bin/env bash
# Web Agent 백엔드 — 풀 GPU 모드 (기본 정책과 동일, 명시적 기동용)
set -euo pipefail
cd "$(dirname "$0")/.."
VENV_DIR="${VENV_DIR:-.venv-webagent}"
PORT="${PORT:-8800}"
if [ ! -d "${VENV_DIR}" ]; then
  echo "ERROR: ${VENV_DIR} 없음"
  exit 1
fi
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"
export WEBAGENT_USE_GPU=1
export WEBAGENT_USE_GPU_RT=1
export TF_CPP_MIN_LOG_LEVEL=1
echo "==> GPU 백엔드 (port ${PORT})"
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>/dev/null || true
exec uvicorn backend.app:app --host 0.0.0.0 --port "${PORT}" --workers 1
