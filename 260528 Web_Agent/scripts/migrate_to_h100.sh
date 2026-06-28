#!/usr/bin/env bash
# ============================================================================
# migrate_to_h100.sh — 현재 5090 서버(dclcom61) → H100 서버로 Web Agent 이주
#
# 사용:
#   H100_HOST=user@h100.example.com \
#   H100_PATH=/home/user/Web_Agent \
#   ./scripts/migrate_to_h100.sh
#
# 동작:
#   1) 코드/씬 rsync (.venv-*, sessions, node_modules 제외)
#   2) Map_Mesh.obj 등 자주 쓰는 외부 씬 동기화 (선택)
#   3) H100측 setup_h100.sh 자동 실행 (선택)
#
# 환경변수:
#   H100_HOST     필수. ssh-style host (e.g. user@host)
#   H100_PATH     필수. H100 측 Web_Agent 절대 경로
#   H100_SCENES   선택. H100 측 추가 씬 디렉토리 (예: /home/user/scenes)
#   AUTO_SETUP    선택. 1이면 setup_h100.sh를 ssh로 자동 실행 (기본 0)
#   SYNC_SESSIONS 선택. 1이면 sessions/ 도 동기화 (기본 0 — 결과물 다시 생성 권장)
# ============================================================================

set -euo pipefail

: "${H100_HOST:?H100_HOST 환경변수 필요 (예: user@h100.lab.example.com)}"
: "${H100_PATH:?H100_PATH 환경변수 필요 (예: /home/user/Web_Agent)}"

SRC_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SRC_DIR"

echo "==> Web Agent 이주: ${SRC_DIR}"
echo "    → ${H100_HOST}:${H100_PATH}"

# 1) 코드 + 문서 + 스크립트
RSYNC_EXCLUDES=(
  --exclude='.venv*'
  --exclude='__pycache__'
  --exclude='*.pyc'
  --exclude='node_modules'
  --exclude='dist'
  --exclude='build'
  --exclude='*.npz.tmp'
)

if [ "${SYNC_SESSIONS:-0}" != "1" ]; then
  RSYNC_EXCLUDES+=(--exclude='sessions/*')
fi

ssh "${H100_HOST}" "mkdir -p '${H100_PATH}'"

echo "==> [1/3] 코드 동기화 (rsync)"
rsync -avz --delete --progress \
  "${RSYNC_EXCLUDES[@]}" \
  "${SRC_DIR}/" "${H100_HOST}:${H100_PATH}/"

# 2) 외부 씬 (Map_Mesh.obj 등)
if [ -n "${H100_SCENES:-}" ]; then
  echo "==> [2/3] 외부 씬 디렉토리 동기화"
  ssh "${H100_HOST}" "mkdir -p '${H100_SCENES}'"
  for f in \
    "${HOME}/twin_minji/260512 RT Result Data/Map_Mesh.obj" \
    "${HOME}/twin_minji/260512 RT Result Data/Map_Mesh.ply" \
  ; do
    if [ -f "${f}" ]; then
      echo "    sync: $(basename "${f}")"
      rsync -avz --progress "${f}" "${H100_HOST}:${H100_SCENES}/"
    fi
  done
else
  echo "==> [2/3] 외부 씬 동기화 건너뜀 (H100_SCENES 미설정)"
fi

# 3) H100 측 setup 자동 실행 (선택)
if [ "${AUTO_SETUP:-0}" = "1" ]; then
  echo "==> [3/3] H100 측 setup_h100.sh 실행"
  ssh "${H100_HOST}" "cd '${H100_PATH}' && bash scripts/setup_h100.sh"
else
  echo "==> [3/3] setup 자동 실행 건너뜀"
  echo
  echo "다음 단계 (H100에서 직접):"
  echo "  ssh ${H100_HOST}"
  echo "  cd ${H100_PATH}"
  echo "  bash scripts/setup_h100.sh        # venv 생성 + 의존성 설치 (10~15분)"
  echo "  bash scripts/run_h100_backend.sh  # GPU 모드 백엔드 실행"
fi

echo
echo "✔ 이주 완료."
