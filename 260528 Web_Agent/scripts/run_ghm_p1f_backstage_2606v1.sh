#!/usr/bin/env bash
# P1F Marginal CCM — 2×GPU 백그라운드 재개 (미완료 RX만)
#
# 사용:
#   ./scripts/run_ghm_p1f_backstage_2606v1.sh start
#   ./scripts/run_ghm_p1f_backstage_2606v1.sh status
#   ./scripts/run_ghm_p1f_backstage_2606v1.sh stop
#   ./scripts/run_ghm_p1f_backstage_2606v1.sh logs
#
# 실행 환경: dclserver78, .venv-webagent, H100×2

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SESSION="${SESSION_DIR:-sessions/ghm_full5071_260604_161504}"
NPZ="${GHM_NPZ:-/home/dclserver78/twin_minji/260604 GHM_test/channel_data_260531_GHM_Twin_v0_1.npz}"
PIDFILE="${ROOT}/logs/ghm_p1f_backstage.pid"
LOG="${ROOT}/logs/ghm_p1f_backstage.log"

cmd_start() {
  if [[ -f "$PIDFILE" ]]; then
    local oldpid
    oldpid="$(cat "$PIDFILE")"
    if kill -0 "$oldpid" 2>/dev/null; then
      echo "[backstage] 이미 실행 중 PID=$oldpid"
      echo "  로그: $LOG"
      exit 0
    fi
    rm -f "$PIDFILE"
  fi

  # 동일 세션의 다른 run_ghm (단일 GPU) 중복 방지
  local others
  others="$(pgrep -f "run_ghm_rt_to_ccm.*${SESSION}" 2>/dev/null || true)"
  if [[ -n "$others" ]]; then
    echo "[backstage] 기존 run_ghm 종료: $others"
    kill $others 2>/dev/null || true
    sleep 3
    kill -9 $others 2>/dev/null || true
  fi

  source .venv-webagent/bin/activate
  export WEBAGENT_USE_GPU=1
  export WEBAGENT_P1F_NUM_GPUS="${WEBAGENT_P1F_NUM_GPUS:-2}"
  export TF_CPP_MIN_LOG_LEVEL=1

  echo "[backstage] P1F 2×GPU 시작 (미완료 RX만)"
  echo "  session : $SESSION"
  echo "  log     : $LOG"
  echo "  worker  : $ROOT/$SESSION/.p1f_dualgpu/worker_gpu*.log"
  echo "  GPUs    : $WEBAGENT_P1F_NUM_GPUS"

  : > "$LOG"

  nohup python scripts/run_ghm_rt_to_ccm_2606v1.py \
    --from-stage post-p1b \
    --session-dir "$SESSION" \
    --npz "$NPZ" \
    --rx-subset 0 \
    --no-with-derived \
    >> "$LOG" 2>&1 &

  local pid=$!
  echo "$pid" > "$PIDFILE"
  echo "[backstage] PID=$pid (nohup)"
  sleep 5
  if kill -0 "$pid" 2>/dev/null; then
    tail -12 "$LOG" 2>/dev/null || true
  else
    echo "[backstage] 시작 실패 — 로그 확인: $LOG"
    rm -f "$PIDFILE"
    exit 1
  fi
}

cmd_status() {
  if [[ -f "$PIDFILE" ]]; then
    local pid
    pid="$(cat "$PIDFILE")"
    if kill -0 "$pid" 2>/dev/null; then
      echo "[backstage] 실행 중 PID=$pid"
    else
      echo "[backstage] PID 파일 있으나 프로세스 없음 (완료 또는 비정상 종료)"
    fi
  else
    echo "[backstage] PID 파일 없음"
  fi

  python3 << 'PY'
import re
from pathlib import Path
import numpy as np

root = Path("/home/dclserver78/twin_minji/260528 Web_Agent")
session = root / "sessions/ghm_full5071_260604_161504"
p1b = list((session / "P1B_Valid_Results").glob("*_Valid_RXs.npz"))[0]
with np.load(p1b, allow_pickle=True) as d:
    exp = int(d["tau"].shape[0])
    if "rx_indices" in d.files:
        rx_idx = set(int(x) for x in np.asarray(d["rx_indices"]).reshape(-1))
    else:
        rx_idx = set(range(1, exp + 1))

p1f = session / "P1F_Marginal_CCM_Results"
pat = re.compile(r"RX(\d+)_Marginal_CCM")
done = {int(pat.search(f.name).group(1)) for f in p1f.glob("*_Marginal_CCM.npz") if pat.search(f.name)}
rem = len(rx_idx - done)
print(f"P1F 진행: {len(done)}/{exp} ({100*len(done)/exp:.1f}%)  미완료 {rem}")
PY

  echo "로그: $LOG"
  WDIR="$ROOT/$SESSION/.p1f_dualgpu"
  if [[ -d "$WDIR" ]]; then
    for f in "$WDIR"/worker_gpu*.log; do
      [[ -f "$f" ]] && echo "--- $(basename "$f") (tail 3) ---" && tail -3 "$f" 2>/dev/null
    done
  fi
  pgrep -af "p1f_dual_gpu_worker" 2>/dev/null || true
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null || true
}

cmd_stop() {
  if [[ -f "$PIDFILE" ]]; then
    local pid
    pid="$(cat "$PIDFILE")"
    if kill -0 "$pid" 2>/dev/null; then
      echo "[backstage] 종료 PID=$pid"
      kill "$pid" 2>/dev/null || true
      sleep 2
      kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$PIDFILE"
  fi
  pkill -f "p1f_gpu[01]" 2>/dev/null || true
  echo "[backstage] stop 완료"
}

cmd_logs() {
  tail -n "${1:-40}" "$LOG"
}

case "${1:-start}" in
  start)  cmd_start ;;
  status) cmd_status ;;
  stop)   cmd_stop ;;
  logs)   cmd_logs "${2:-40}" ;;
  *)
    echo "Usage: $0 {start|status|stop|logs [N]}"
    exit 1
    ;;
esac
