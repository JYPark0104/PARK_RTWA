#!/usr/bin/env bash
# ============================================================================
# tunnel_from_5090.sh — 현재 5090(dclcom61)에서 H100 백엔드로 SSH 터널
#
# 동작: 로컬 :8000 → H100:8000
# 그러면 dclcom61에서 띄운 프론트엔드(:5173)가 그대로 H100 백엔드로 연결됨.
#
# 사용:
#   H100_HOST=user@h100 bash scripts/tunnel_from_5090.sh
#
# 환경변수:
#   H100_HOST       필수 (예: user@h100.example.com)
#   LOCAL_PORT      기본 8000
#   REMOTE_PORT     기본 8000
# ============================================================================

set -euo pipefail

: "${H100_HOST:?H100_HOST 필요 (예: user@h100.example.com)}"
LOCAL_PORT="${LOCAL_PORT:-8000}"
REMOTE_PORT="${REMOTE_PORT:-8000}"

echo "==> SSH tunnel: localhost:${LOCAL_PORT} → ${H100_HOST}:${REMOTE_PORT}"
echo "   (Ctrl+C로 종료)"
exec ssh -N -L "${LOCAL_PORT}:localhost:${REMOTE_PORT}" "${H100_HOST}"
