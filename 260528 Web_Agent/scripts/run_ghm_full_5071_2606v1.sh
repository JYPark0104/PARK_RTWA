#!/usr/bin/env bash
# GHM Twin — UE 5071 전체 + RSRP + PADP + Marginal CCM (R_BS, R_UE)
# 예상: P1A 수십 분~수 시간, P1F 수 시간 (H100 GPU 권장)
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv-webagent/bin/activate
export WEBAGENT_USE_GPU=1
export WEBAGENT_USE_GPU_RT=1
export TF_CPP_MIN_LOG_LEVEL=1
LOG="logs/ghm_full5071_$(date +%y%m%d_%H%M%S).log"
echo "로그: $LOG"
exec python scripts/run_ghm_rt_to_ccm_2606v1.py \
  --rx-subset 0 \
  --max-depth 5 \
  --bs-panel 4 4 --bs-grid 2 2 \
  --ue-panel 2 2 \
  --with-derived \
  2>&1 | tee "$LOG"
