#!/usr/bin/env bash
# ============================================================================
# run_h100_backend.sh — H100에서 Web Agent 백엔드를 풀 GPU 모드로 기동
#
# 사용:
#   bash scripts/run_h100_backend.sh           # foreground, 포트 8000
#   PORT=8001 bash scripts/run_h100_backend.sh # 다른 포트
#
# 환경변수:
#   PORT              기본 8000
#   VENV_DIR          기본 .venv-h100
#   WEBAGENT_USE_GPU  강제 1 (TF/sionna GPU)
#   WEBAGENT_USE_GPU_RT  강제 1 (mitsuba CUDA variant — H100은 sm_90이라 drjit 1.3.1 PTX 8.5로 OK)
# ============================================================================

set -euo pipefail

cd "$(dirname "$0")/.."
VENV_DIR="${VENV_DIR:-.venv-h100}"
PORT="${PORT:-8000}"

if [ ! -d "${VENV_DIR}" ]; then
  echo "ERROR: ${VENV_DIR} 없음. 먼저 bash scripts/setup_h100.sh 실행"
  exit 1
fi

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

export WEBAGENT_USE_GPU=1
export WEBAGENT_USE_GPU_RT=1   # H100 sm_90은 drjit 1.3.1 PTX 8.5 정상 지원
export TF_CPP_MIN_LOG_LEVEL=1
export TF_FORCE_GPU_ALLOW_GROWTH=true

echo "==> H100 GPU 모드 백엔드 기동 (port ${PORT})"
echo "    WEBAGENT_USE_GPU=${WEBAGENT_USE_GPU}  WEBAGENT_USE_GPU_RT=${WEBAGENT_USE_GPU_RT}"
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader || true

exec uvicorn backend.app:app --host 0.0.0.0 --port "${PORT}" --workers 1
