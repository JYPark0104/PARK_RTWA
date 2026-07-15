"""machine.py — 실행 머신/하드웨어 지문 (예측의 필수 매칭 키).

같은 config라도 dclcom45 / dclcom55 / RTX 5090 / H100 / CPU 모드에 따라 실행
시간이 몇 배씩 차이난다(사용자 확정 Q1). 따라서 run 이력의 **필수 매칭 키**로
`machine_id + gpu_mode + gpu_name` 을 남긴다.

지문 구성
--------
- machine_id : 호스트명 (예: dclcom55). 서버 = 머신 1:1 가정.
- gpu_mode   : "gpu" | "cpu"  (runtime_env 정책 결과 우선, 없으면 env 폴백)
- gpu_name   : 첫 GPU 장치명 (예: "NVIDIA H100 80GB HBM3"), CPU면 None
- n_gpu      : 활성 GPU 장수

실행 환경: Python 3.10 / 표준 라이브러리
"""

from __future__ import annotations

import os
import socket
import subprocess
from functools import lru_cache
from typing import Any


def _gpu_summary() -> dict[str, Any]:
    """runtime_env 요약을 best-effort로 가져온다(초기화 전이면 빈 dict)."""
    try:
        from ..jobs.runtime_env import get_runtime_summary

        return get_runtime_summary() or {}
    except Exception:
        return {}


@lru_cache(maxsize=1)
def _nvidia_gpu() -> tuple[str | None, int]:
    """nvidia-smi 로 (첫 GPU 이름, 장수) 조회. runtime_env 초기화 여부와 무관하게
    일관된 gpu_name 을 얻기 위한 폴백 (예측/기록 간 machine 키 분열 방지)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True, timeout=6,
        )
        names = [ln.strip() for ln in out.strip().splitlines() if ln.strip()]
        return (names[0] if names else None, len(names))
    except Exception:
        return (None, 0)


@lru_cache(maxsize=1)
def _hostname() -> str:
    try:
        return socket.gethostname() or "unknown-host"
    except Exception:
        return "unknown-host"


def machine_fingerprint() -> dict[str, Any]:
    """현재 실행 머신의 지문 dict.

    gpu_name/장수는 runtime_env 요약을 우선 쓰되, 초기화 전이면 nvidia-smi 로 폴백해
    항상 일관되게 채운다. (예측은 uvicorn, 기록은 잡 subprocess 에서 일어나 gpu_name 이
    None↔'H100 NVL' 로 갈려 machine 키가 분열되던 버그 방지 — 2026-07-10)
    """
    summary = _gpu_summary()
    gpus = summary.get("gpus") or []
    gpu_name = None
    n_gpu = 0
    if gpus:
        gpu_name = gpus[0].get("device_name") or gpus[0].get("name")
        n_gpu = len(gpus)
    else:
        # runtime_env 미초기화(예: uvicorn 프로세스) → nvidia-smi 폴백으로 일관성 확보
        gpu_name, n_gpu = _nvidia_gpu()

    if os.environ.get("CUDA_VISIBLE_DEVICES") == "-1":
        gpu_mode = "cpu"
    elif gpus or n_gpu > 0:
        gpu_mode = "gpu"
    else:
        gpu_mode = "gpu" if os.environ.get("WEBAGENT_USE_GPU", "1") == "1" else "cpu"

    return {
        "machine_id": _hostname(),
        "gpu_mode": gpu_mode,
        "gpu_name": gpu_name,
        "n_gpu": n_gpu,
    }


def machine_key(fp: dict[str, Any] | None = None) -> str:
    """표시/기록용 문자열 키. (매칭은 machine_id+gpu_mode 필드로 직접 하므로 gpu_name 은
    키에서 제외 — 같은 머신이 gpu_name None/실명으로 갈리는 분열을 막는다.)"""
    fp = fp or machine_fingerprint()
    return f"{fp.get('machine_id')}|{fp.get('gpu_mode')}"
