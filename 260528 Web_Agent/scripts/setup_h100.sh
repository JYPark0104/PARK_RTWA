#!/usr/bin/env bash
# ============================================================================
# setup_h100.sh — H100 서버에서 한 번만 실행 (venv + 의존성)
#
# 가정:
#   - Python 3.10 설치되어 있음 (`python3.10 --version`)
#   - CUDA 드라이버 12.x+ (nvidia-smi 확인)
#   - H100 (sm_90, Hopper) — TF 2.21 stable native 지원
#
# 사용:
#   bash scripts/setup_h100.sh
#
# 결과:
#   - .venv-h100/ 생성
#   - tensorflow[and-cuda] 2.21.0, sionna 1.2.2, mitsuba, drjit 설치
#   - 빠른 GPU smoke test 실행
# ============================================================================

set -euo pipefail

PY="${PYTHON:-python3.10}"
VENV_DIR="${VENV_DIR:-.venv-h100}"

cd "$(dirname "$0")/.."
ROOT="$(pwd)"

echo "==> setup_h100.sh in ${ROOT}"
echo "    PY=${PY}   VENV_DIR=${VENV_DIR}"

# 0) 사전 점검
${PY} --version
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "WARN: nvidia-smi 미발견. GPU 드라이버 확인하세요."
else
  nvidia-smi | head -n 20
fi

# 1) venv
if [ ! -d "${VENV_DIR}" ]; then
  echo "==> [1/4] venv 생성: ${VENV_DIR}"
  ${PY} -m venv "${VENV_DIR}"
else
  echo "==> [1/4] venv 이미 존재"
fi

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip wheel setuptools

# 2) 필수 ML 패키지
echo "==> [2/4] tensorflow[and-cuda] + sionna 설치 (몇 분 소요)"
pip install "tensorflow[and-cuda]==2.21.0"
pip install "sionna==1.2.2"
pip install "mitsuba==3.8.0" "drjit==1.3.1"

# 3) Web Agent 백엔드 의존성
echo "==> [3/4] 백엔드 의존성 설치"
if [ -f backend/requirements.txt ]; then
  pip install -r backend/requirements.txt
else
  pip install fastapi 'uvicorn[standard]' python-multipart websockets pydantic \
      trimesh[easy] numpy scipy matplotlib pandas pillow tqdm
fi

# 4) GPU smoke test
echo "==> [4/4] GPU smoke test"
python - <<'PY'
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
import tensorflow as tf
print("TF:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
print("GPUs:", gpus)
if gpus:
    for g in gpus:
        d = tf.config.experimental.get_device_details(g)
        print("  ", g.name, d.get('device_name'), "CC", d.get('compute_capability'))
    a = tf.random.normal((4096, 4096))
    b = tf.random.normal((4096, 4096))
    import time; t0 = time.time(); _ = (a @ b).numpy(); print(f"matmul 4096^2 GPU: {time.time()-t0:.3f}s")
else:
    print("WARN: GPU 미감지. driver/CUDA 확인 필요")

# sionna paths.cir 빠른 동작 확인
import sionna.rt as rt
scene = rt.load_scene(rt.scene.simple_street_canyon)
scene.tx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
scene.rx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
scene.add(rt.Transmitter(name="tx", position=[0,0,30], orientation=[0,0,0]))
scene.add(rt.Receiver(name="rx", position=[40,0,2], orientation=[0,0,0]))
solver = rt.PathSolver()
paths = solver(scene=scene, max_depth=3, samples_per_src=100_000, seed=1)
a, _ = paths.cir(normalize_delays=False, out_type='numpy')
print("paths.cir OK:", a.shape, a.dtype)
PY

echo
echo "✔ setup 완료. 이제 백엔드 실행:"
echo "  bash scripts/run_h100_backend.sh"
