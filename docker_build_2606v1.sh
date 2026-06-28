#!/usr/bin/env bash
# ============================================================================
# docker_build_2606v1.sh — twin_minji_verPARK 컨테이너 이미지 빌드 (백그라운드용)
# ----------------------------------------------------------------------------
# 사용:
#   bash docker_build_2606v1.sh
# 로그:
#   docker_build_<타임스탬프>.log  (tail -f 로 진행 확인)
# 이미지 태그:
#   twin-rt-webagent:latest
# ============================================================================
set -uo pipefail

cd "$(dirname "$0")"

TS="$(date +%Y%m%d_%H%M%S)"
LOG="docker_build_${TS}.log"
IMAGE="twin-rt-webagent:latest"

{
  echo "==================================================================="
  echo "[build] 시작: $(date '+%F %T')"
  echo "[build] 이미지: ${IMAGE}"
  echo "[build] 컨텍스트: $(pwd)"
  echo "==================================================================="

  # 진행률을 로그로 잘 남기기 위해 plain progress 사용
  DOCKER_BUILDKIT=1 docker build \
      --progress=plain \
      -t "${IMAGE}" \
      -f Dockerfile \
      . 
  RC=$?

  echo "==================================================================="
  if [ "${RC}" -eq 0 ]; then
    echo "[build] ✅ 성공: $(date '+%F %T')"
    echo "[build] 이미지 확인:"
    docker images "${IMAGE}"
  else
    echo "[build] ❌ 실패 (exit ${RC}): $(date '+%F %T')"
    echo "[build] → Plan B(컨테이너 없이 venv 직접 실행)로 진행 권장. run_planB_venv.sh 참고."
  fi
  echo "==================================================================="
  echo "EXIT_CODE=${RC}"
} > "${LOG}" 2>&1

exit "${RC}"
