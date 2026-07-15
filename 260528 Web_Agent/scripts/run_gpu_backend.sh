#!/usr/bin/env bash
# Web Agent 백엔드 — 풀 GPU 모드 (canonical venv = /opt/venvs/webagent)
#
# 2026-07-09 수정: 기본 venv 를 .venv-webagent → /opt/venvs/webagent 로 변경.
#   .venv-webagent 는 부분/빈 venv 라 open3d 등이 없어 지면격자/TX스냅이 실패한다.
#   (원하면 VENV_DIR 로 override 가능하되, 기본은 올바른 venv 로 강제)
set -euo pipefail
cd "$(dirname "$0")/.."

VENV_DIR="${VENV_DIR:-/opt/venvs/webagent}"
PORT="${PORT:-8800}"
PYBIN="${VENV_DIR}/bin/python"

if [ ! -x "${PYBIN}" ]; then
  echo "ERROR: venv python 없음: ${PYBIN}"
  exit 1
fi

# 프리플라이트: 필수 모듈이 없으면(=잘못된 venv) 조기 차단
if ! "${PYBIN}" -c "import open3d, sionna, trimesh" >/dev/null 2>&1; then
  echo "ERROR: ${VENV_DIR} 에 필수 모듈(open3d/sionna/trimesh)이 없습니다."
  echo "       올바른 venv(/opt/venvs/webagent)인지 확인하세요."
  exit 1
fi

export WEBAGENT_USE_GPU=1
export WEBAGENT_USE_GPU_RT=1
export TF_CPP_MIN_LOG_LEVEL=1
export TF_FORCE_GPU_ALLOW_GROWTH=true   # TF 가 VRAM 전체를 미리 잡지 않도록

echo "==> GPU 백엔드 기동 (venv=${VENV_DIR}, port ${PORT})"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
exec "${VENV_DIR}/bin/uvicorn" backend.app:app --host 0.0.0.0 --port "${PORT}" --workers 1
