#!/usr/bin/env bash
# ============================================================================
# run_planB_venv.sh — Plan B: Docker 없이 호스트 venv 로 Web_Agent 기동
# ----------------------------------------------------------------------------
# Docker 빌드가 실패하거나 컨테이너를 쓰지 않을 때, 지금까지처럼 호스트의
# .venv-webagent 로 백엔드/프런트엔드를 띄운다. (현 상태 유지 방식)
#
# 사용:
#   bash run_planB_venv.sh         # 백엔드(8800) + 프런트(5173) 모두 백그라운드 기동
#   bash run_planB_venv.sh stop    # 기동한 프로세스 종료
#
# 접속: http://165.132.192.78:5173
# 로그: planB_backend.log / planB_frontend.log
# ============================================================================
set -uo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
WEB="${ROOT}/260528 Web_Agent"
BACK_LOG="${ROOT}/planB_backend.log"
FRONT_LOG="${ROOT}/planB_frontend.log"
BACK_PID="${ROOT}/planB_backend.pid"
FRONT_PID="${ROOT}/planB_frontend.pid"

stop() {
  for P in "${BACK_PID}" "${FRONT_PID}"; do
    if [ -f "${P}" ]; then
      kill "$(cat "${P}")" 2>/dev/null && echo "stopped $(cat "${P}")"
      rm -f "${P}"
    fi
  done
}

if [ "${1:-}" = "stop" ]; then
  stop
  exit 0
fi

# 백엔드 (포트 8800, GPU) — run_gpu_backend.sh 가 .venv-webagent 활성화
echo "[planB] 백엔드 기동 (8800) ..."
nohup bash "${WEB}/scripts/run_gpu_backend.sh" > "${BACK_LOG}" 2>&1 &
echo $! > "${BACK_PID}"

# 프런트엔드 (포트 5173) — 백엔드 8800 으로 프록시
echo "[planB] 프런트엔드 기동 (5173) ..."
cd "${WEB}/frontend"
nohup env VITE_API_TARGET=http://localhost:8800 npm run dev > "${FRONT_LOG}" 2>&1 &
echo $! > "${FRONT_PID}"

echo "[planB] 기동 완료. 접속: http://165.132.192.78:5173"
echo "[planB] 로그: ${BACK_LOG} / ${FRONT_LOG}"
echo "[planB] 종료: bash run_planB_venv.sh stop"
