#!/usr/bin/env bash
# ============================================================================
# webagent_ctl.sh — RTWA 백엔드/프론트 통합 관리 (컨테이너 WebAgent_park_server78 내부 실행)
#
# 목적: venv/포트/좀비 프로세스로 인한 불안정을 없애기 위한 '단일 진입점'.
#   - 항상 올바른 venv(/opt/venvs/webagent)로 기동 (open3d/sionna/tf 포함)
#   - 기동 전 필수 모듈 프리플라이트 → 잘못된 venv 조기 차단
#   - 시작 시 기존/좀비 프로세스 정리, 시작 후 헬스체크
#   - VRAM 독점 방지 env(TF_FORCE_GPU_ALLOW_GROWTH) 자동 설정
#
# 사용:
#   bash scripts/webagent_ctl.sh status
#   bash scripts/webagent_ctl.sh preflight
#   bash scripts/webagent_ctl.sh start   [backend|frontend|all]   # 기본 all
#   bash scripts/webagent_ctl.sh stop    [backend|frontend|all]
#   bash scripts/webagent_ctl.sh restart [backend|frontend|all]
#   bash scripts/webagent_ctl.sh health
#
# 포트(컨테이너 내부): backend 8800, frontend 5173
#   (호스트 매핑: 8900→8800, 5273→5173 — 브라우저는 호스트:5273 접속)
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

VENV="${VENV:-/opt/venvs/webagent}"
PYBIN="${VENV}/bin/python"
BACK_PORT="${BACK_PORT:-8800}"
FRONT_PORT="${FRONT_PORT:-5173}"
LOGDIR="${ROOT}/logs"
mkdir -p "${LOGDIR}"

back_pids() { pgrep -f "uvicorn backend.app:app" 2>/dev/null || true; }
front_pids() { pgrep -f "node.*vite" 2>/dev/null || true; }

preflight() {
  echo "[preflight] venv=${VENV}"
  if [ ! -x "${PYBIN}" ]; then echo "  ✗ venv python 없음: ${PYBIN}"; return 1; fi
  "${PYBIN}" - <<'PY'
import importlib.util, sys
crit = ["open3d","sionna","tensorflow","trimesh","mitsuba","numpy","fastapi","uvicorn"]
for m in crit:
    print(f"  {'OK ' if importlib.util.find_spec(m) else 'MISSING'} {m}")
hard = [m for m in ("open3d","sionna","trimesh") if importlib.util.find_spec(m) is None]
print("  python:", sys.executable)
sys.exit(1 if hard else 0)
PY
}

health() {
  curl -s -m 12 "http://localhost:${BACK_PORT}/api/health" \
    | "${PYBIN}" -m json.tool 2>/dev/null \
    || echo "  (health 응답 없음 — 기동 중이거나 실패)"
}

stop_backend() {
  local p; p="$(back_pids)"
  if [ -n "${p}" ]; then
    echo "[stop] backend: ${p}"; kill ${p} 2>/dev/null; sleep 3
    p="$(back_pids)"; [ -n "${p}" ] && { echo "  강제종료: ${p}"; kill -9 ${p} 2>/dev/null; }
  else echo "[stop] backend 실행 중 아님"; fi
}

start_backend() {
  if ! preflight; then echo "[start] ✗ preflight 실패 — 기동 중단(잘못된 venv 가능)"; return 1; fi
  stop_backend
  local log="${LOGDIR}/uvicorn_$(date +%y%m%d_%H%M%S).log"
  echo "[start] backend (venv=${VENV}, port ${BACK_PORT}) → ${log}"
  ( cd "${ROOT}" && \
    WEBAGENT_USE_GPU=1 WEBAGENT_USE_GPU_RT=1 TF_CPP_MIN_LOG_LEVEL=1 TF_FORCE_GPU_ALLOW_GROWTH=true \
    nohup "${VENV}/bin/uvicorn" backend.app:app --host 0.0.0.0 --port "${BACK_PORT}" --workers 1 \
    > "${log}" 2>&1 & )
  echo "  기동 대기(최대 ~25s)…"
  for i in $(seq 1 25); do
    sleep 1
    if curl -s -m 3 "http://localhost:${BACK_PORT}/api/health" >/dev/null 2>&1; then break; fi
  done
  health
}

stop_frontend() {
  local p; p="$(front_pids)"
  if [ -n "${p}" ]; then echo "[stop] frontend: ${p}"; kill ${p} 2>/dev/null; else echo "[stop] frontend 실행 중 아님"; fi
}

start_frontend() {
  stop_frontend
  local log="${LOGDIR}/vite_$(date +%y%m%d_%H%M%S).log"
  echo "[start] frontend (port ${FRONT_PORT}) → ${log}"
  ( cd "${ROOT}/frontend" && \
    VITE_API_TARGET="http://localhost:${BACK_PORT}" nohup npm run dev > "${log}" 2>&1 & )
  sleep 3
  echo "  frontend 기동 요청 완료 (브라우저: 호스트:5273)"
}

status() {
  echo "===== RTWA 상태 ====="
  echo "[venv] ${VENV}  (python: ${PYBIN})"
  echo "[backend pids] $(back_pids | tr '\n' ' ')"
  echo "[frontend pids] $(front_pids | tr '\n' ' ')"
  echo "[ports]"; (ss -ltnp 2>/dev/null | grep -E ":${BACK_PORT}|:${FRONT_PORT}") || echo "  (없음)"
  echo "[run_one_job(진행중 잡)]"; (pgrep -af "run_one_job" 2>/dev/null) || echo "  (없음)"
  echo "[health]"; health
}

CMD="${1:-status}"; TARGET="${2:-all}"
case "${CMD}" in
  preflight) preflight ;;
  health)    health ;;
  status)    status ;;
  start)     [ "${TARGET}" = frontend ] || start_backend; [ "${TARGET}" = backend ] || start_frontend ;;
  stop)      [ "${TARGET}" = frontend ] || stop_backend;  [ "${TARGET}" = backend ] || stop_frontend ;;
  restart)   [ "${TARGET}" = frontend ] || start_backend; [ "${TARGET}" = backend ] || start_frontend ;;
  *) echo "usage: $0 {start|stop|restart|status|health|preflight} [backend|frontend|all]"; exit 2 ;;
esac
