"""
m1_env_agent.py
============
환경 초기화 에이전트.

역할:
  - GPU 환경 변수 설정
  - TensorFlow, Sionna, Mitsuba 등 필요한 라이브러리 임포트
  - Mitsuba 렌더링 variant 설정
  - 불필요한 경고/로그 억제 (빨간 글씨 최소화)

반환:
  True (성공) / RuntimeError raise (실패)
"""

import os
import sys
import time


def _suppress_warnings():
    """
    치명적이지 않은 경고/로그를 억제한다.
    - TensorFlow: WARNING 이하 로그 숨김
    - Mitsuba/DrJit: stderr 경고 억제
    - Python warnings 필터
    """
    import warnings
    warnings.filterwarnings("ignore")

    # TensorFlow 로그 레벨: ERROR만 출력 (WARNING, INFO 숨김)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["ABSL_MIN_LOG_LEVEL"] = "3"

    # Mitsuba/DrJit stderr 경고 억제
    os.environ["DRJIT_NO_VERBOSE"] = "1"


def run() -> bool:
    """
    Ray Tracing Agent 환경을 초기화한다.

    Returns:
        True: 모든 라이브러리 임포트 및 환경 설정 성공

    Raises:
        RuntimeError: 필수 라이브러리 임포트 실패 시
    """
    print("=" * 50)
    print("🚀 [Env_Agent] 환경 초기화 시작")
    print("=" * 50)

    start_total = time.time()

    # ── 1. GPU 환경 변수 설정 ──────────────────────────────
    print(f"[{time.strftime('%H:%M:%S')}] 환경 변수 설정 중...")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["OMP_NUM_THREADS"] = "16"
    os.environ["DRJIT_THREAD_COUNT"] = "16"

    # 경고 억제 (치명적이지 않은 빨간 글씨 제거)
    _suppress_warnings()
    print("   ✅ 환경 변수 설정 완료")

    # ── 2. NumPy & Matplotlib ──────────────────────────────
    try:
        s = time.time()
        import numpy as np          # noqa: F401
        import matplotlib           # noqa: F401
        import matplotlib.pyplot    # noqa: F401
        matplotlib.use('Agg')       # 헤드리스 환경 (터미널)에서 GUI 없이 PNG 저장
        print(f"[{time.strftime('%H:%M:%S')}] NumPy & Matplotlib 로드 완료 ({time.time()-s:.2f}s)")
    except Exception as e:
        msg = f"❌ NumPy/Matplotlib 로드 실패: {e}"
        print(msg)
        raise RuntimeError(msg) from e

    # ── 3. TensorFlow ──────────────────────────────────────
    try:
        print(f"[{time.strftime('%H:%M:%S')}] TensorFlow 로드 시작...")
        s = time.time()

        # TF 로그를 stderr가 아닌 /dev/null로 리다이렉트
        import logging
        logging.getLogger('tensorflow').setLevel(logging.ERROR)

        import tensorflow as tf     # noqa: F401
        tf.get_logger().setLevel('ERROR')

        gpus = tf.config.list_physical_devices('GPU')
        print(f"   ✅ TF 로드 완료 ({time.time()-s:.2f}s) - 검출된 GPU: {len(gpus)}개")
    except Exception as e:
        msg = f"❌ TensorFlow 로드 실패: {e}"
        print(msg)
        raise RuntimeError(msg) from e

    # ── 4. Sionna & Mitsuba ────────────────────────────────
    try:
        print(f"[{time.strftime('%H:%M:%S')}] Sionna & Mitsuba 로드 시작...")
        s = time.time()

        import mitsuba as mi
        mi.set_variant('cuda_ad_mono_polarized')

        import sionna               # noqa: F401
        from sionna.rt import (     # noqa: F401
            load_scene, PathSolver, Camera, LambertianPattern, PlanarArray
        )
        print(f"   ✅ Sionna RT 로드 완료 ({time.time()-s:.2f}s)")
    except Exception as e:
        msg = f"❌ Sionna/Mitsuba 로드 실패: {e}"
        print(msg)
        raise RuntimeError(msg) from e

    print("\n" + "=" * 50)
    print(f"🎊 [Env_Agent] 초기화 완료 (총 소요 시간: {time.time()-start_total:.2f}s)")
    print("=" * 50)

    return True
