"""Runtime 환경 자동 설정 헬퍼.

# 실행 환경
# - Python    : 3.10
# - GPU       : H100(sm_90) 등 — 기본 풀 GPU (TF + Mitsuba CUDA RT)
# - TF        : 2.21.0 (CUDA 12.5 bundled via nvidia-* pip wheels)
# - Sionna    : 1.2.2

주요 기능
---------
- :func:`apply_webagent_gpu_env`: import 전 GPU/Mitsuba variant 정책 적용 (기본 ON)
- :func:`init_runtime`: TF GPU memory growth 설정
- :func:`get_runtime_summary`: 디버깅용 환경 요약
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Any

log = logging.getLogger("webagent.runtime")

_INITIALIZED = False
_SUMMARY: dict[str, Any] = {}
_GPU_POLICY: dict[str, Any] = {}

# sm_120(Blackwell) stable wheel은 RT GPU 불안 — 자동 CPU 폴백 (override 가능)
_BLACKWELL_CC_THRESHOLD = 12.0


def _nvidia_max_compute_capability() -> float | None:
    """nvidia-smi로 최대 compute capability 조회. 실패 시 None."""

    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            text=True,
            timeout=8,
        )
        caps: list[float] = []
        for line in out.strip().splitlines():
            parts = line.strip().split(".")
            if len(parts) >= 2:
                caps.append(float(f"{parts[0]}.{parts[1]}"))
        return max(caps) if caps else None
    except Exception:
        return None


def resolve_gpu_policy() -> dict[str, Any]:
    """Web Agent GPU 사용 정책.

    기본: GPU ON + Mitsuba CUDA RT ON (H100/A100 등).

    Overrides
    ---------
    WEBAGENT_FORCE_CPU=1       : 강제 CPU
    WEBAGENT_USE_GPU=0         : TF GPU 끔
    WEBAGENT_USE_GPU_RT=0      : Mitsuba LLVM (TF만 GPU)
    WEBAGENT_ALLOW_BLACKWELL_GPU=1 : sm_120에서도 GPU 시도 (실험, SEGV 위험)
    """

    if os.environ.get("WEBAGENT_FORCE_CPU", "0") == "1":
        return {
            "use_gpu": False,
            "use_gpu_rt": False,
            "reason": "WEBAGENT_FORCE_CPU=1",
            "max_compute_capability": _nvidia_max_compute_capability(),
        }

    explicit_off = os.environ.get("WEBAGENT_USE_GPU") == "0"
    if explicit_off:
        return {
            "use_gpu": False,
            "use_gpu_rt": False,
            "reason": "WEBAGENT_USE_GPU=0",
            "max_compute_capability": _nvidia_max_compute_capability(),
        }

    max_cc = _nvidia_max_compute_capability()
    use_gpu = os.environ.get("WEBAGENT_USE_GPU", "1") == "1"
    use_gpu_rt = os.environ.get("WEBAGENT_USE_GPU_RT", "1") == "1"

    if max_cc is not None and max_cc >= _BLACKWELL_CC_THRESHOLD:
        if os.environ.get("WEBAGENT_ALLOW_BLACKWELL_GPU", "0") != "1":
            return {
                "use_gpu": False,
                "use_gpu_rt": False,
                "reason": (
                    f"Blackwell sm_{int(max_cc * 10)} — stable TF/drjit RT GPU 비권장. "
                    "실험 시 WEBAGENT_ALLOW_BLACKWELL_GPU=1"
                ),
                "max_compute_capability": max_cc,
            }

    if max_cc is None and use_gpu:
        reason = "WEBAGENT_USE_GPU 기본 ON (nvidia-smi 미확인)"
    elif max_cc is not None:
        reason = f"GPU 기본 ON (max CC={max_cc})"
    else:
        reason = "GPU 없음 — CPU"
        use_gpu = False
        use_gpu_rt = False

    if not use_gpu:
        use_gpu_rt = False

    return {
        "use_gpu": use_gpu,
        "use_gpu_rt": use_gpu_rt and use_gpu,
        "reason": reason,
        "max_compute_capability": max_cc,
    }


def apply_webagent_gpu_env() -> dict[str, Any]:
    """``backend`` 패키지 import 직후 1회 호출. TF/Mitsuba import 전 환경 확정."""

    global _GPU_POLICY
    if _GPU_POLICY:
        return dict(_GPU_POLICY)

    os.environ["WEBAGENT_MANAGED_RUNTIME"] = "1"
    policy = resolve_gpu_policy()
    _GPU_POLICY = policy

    os.environ["WEBAGENT_USE_GPU"] = "1" if policy["use_gpu"] else "0"
    os.environ["WEBAGENT_USE_GPU_RT"] = "1" if policy["use_gpu_rt"] else "0"

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

    cache_dir = Path.home() / ".nv" / "ComputeCache"
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("CUDA_CACHE_PATH", str(cache_dir))
        os.environ.setdefault("CUDA_CACHE_MAXSIZE", str(2 * 1024**3))
    except Exception:
        pass

    if not policy["use_gpu"]:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        _add_nvidia_libs_to_ld_path()

    configure_mitsuba_variant()
    return dict(policy)


def configure_mitsuba_variant() -> str | None:
    """정책에 맞는 Mitsuba variant 설정. 이미 cuda면 유지."""

    try:
        import mitsuba as mi
    except Exception:
        return None

    variants = mi.variants()
    try:
        current = mi.variant()
    except Exception:
        current = None

    want_cuda = os.environ.get("WEBAGENT_USE_GPU_RT", "0") == "1"
    if want_cuda and "cuda_ad_mono_polarized" in variants:
        if current != "cuda_ad_mono_polarized":
            mi.set_variant("cuda_ad_mono_polarized")
        return "cuda_ad_mono_polarized"
    if "llvm_ad_mono_polarized" in variants and current != "llvm_ad_mono_polarized":
        if not current or "cuda" not in str(current):
            mi.set_variant("llvm_ad_mono_polarized")
        return mi.variant()
    return current


def _add_nvidia_libs_to_ld_path() -> str | None:
    try:
        import nvidia  # noqa: F401
    except Exception:
        return None
    import nvidia as _nvidia

    nv_root = Path(_nvidia.__file__).parent
    lib_dirs = sorted({str(p.parent) for p in nv_root.rglob("lib/*.so*")})
    if not lib_dirs:
        return None
    sep = ":"
    current = os.environ.get("LD_LIBRARY_PATH", "")
    system_cuda_dirs = [
        d
        for d in ("/usr/lib/x86_64-linux-gnu", "/lib/x86_64-linux-gnu")
        if Path(d, "libcuda.so.1").exists()
    ]
    new_parts = [
        d
        for d in (system_cuda_dirs + lib_dirs)
        if d not in current.split(sep)
    ]
    if not new_parts:
        return current
    os.environ["LD_LIBRARY_PATH"] = (
        sep.join(new_parts) + (sep + current if current else "")
    )
    return os.environ["LD_LIBRARY_PATH"]


def _set_env_defaults() -> None:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    cache_dir = Path.home() / ".nv" / "ComputeCache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("CUDA_CACHE_PATH", str(cache_dir))
    os.environ.setdefault("CUDA_CACHE_MAXSIZE", str(2 * 1024**3))
    _add_nvidia_libs_to_ld_path()


def _try_init_gpu(tf, retries: int = 2):
    import time

    last_seen: list = []
    for _ in range(retries + 1):
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            return gpus
        last_seen = gpus
        time.sleep(0.5)
    return last_seen


def init_runtime(
    prefer_gpu: bool | None = None,
    gpu_index: int | list[int] | None = None,
    mixed_precision: bool = False,
) -> dict[str, Any]:
    global _INITIALIZED, _SUMMARY
    if _INITIALIZED:
        return dict(_SUMMARY)

    if not _GPU_POLICY:
        apply_webagent_gpu_env()

    if prefer_gpu is None:
        prefer_gpu = os.environ.get("WEBAGENT_USE_GPU", "1") == "1"

    _set_env_defaults()

    if not prefer_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    elif gpu_index is not None:
        if isinstance(gpu_index, int):
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in gpu_index)
    elif os.environ.get("CUDA_VISIBLE_DEVICES") == "-1":
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)

    import tensorflow as tf

    try:
        import mitsuba as mi

        summary_variant = mi.variant()
    except Exception:
        summary_variant = None

    summary: dict[str, Any] = {
        "tf_version": tf.__version__,
        "tf_built_with_cuda": tf.test.is_built_with_cuda(),
        "tf_cuda_version": tf.sysconfig.get_build_info().get("cuda_version"),
        "tf_cudnn_version": tf.sysconfig.get_build_info().get("cudnn_version"),
        "prefer_gpu": prefer_gpu,
        "use_gpu_rt": os.environ.get("WEBAGENT_USE_GPU_RT") == "1",
        "mitsuba_variant": summary_variant,
        "gpu_policy": dict(_GPU_POLICY),
        "gpu_index": gpu_index,
        "gpus": [],
        "mixed_precision": False,
    }

    gpus = _try_init_gpu(tf) if prefer_gpu else []
    if gpus:
        for g in gpus:
            try:
                tf.config.experimental.set_memory_growth(g, True)
            except RuntimeError as exc:
                log.debug("set_memory_growth failed for %s: %s", g.name, exc)
            details = {}
            try:
                details = tf.config.experimental.get_device_details(g)
            except Exception:
                pass
            summary["gpus"].append(
                {
                    "name": g.name,
                    "device_name": details.get("device_name"),
                    "compute_capability": details.get("compute_capability"),
                }
            )
        if mixed_precision:
            try:
                from tensorflow.keras import mixed_precision as mp

                mp.set_global_policy("mixed_float16")
                summary["mixed_precision"] = True
            except Exception as exc:
                log.warning("Failed to enable mixed precision: %s", exc)

    try:
        import sionna

        summary["sionna_version"] = sionna.__version__
    except Exception:
        summary["sionna_version"] = None

    _SUMMARY = summary
    _INITIALIZED = True

    if summary["gpus"]:
        names = ", ".join(
            f"{i}:{g['device_name']}" for i, g in enumerate(summary["gpus"])
        )
        log.info(
            "GPU 활성화: %d장 (%s) | mitsuba=%s | TF %s",
            len(summary["gpus"]),
            names,
            summary.get("mitsuba_variant"),
            summary["tf_version"],
        )
    else:
        log.warning(
            "GPU 미사용 — %s",
            _GPU_POLICY.get("reason", "unknown"),
        )

    return dict(summary)


def get_runtime_summary() -> dict[str, Any]:
    return dict(_SUMMARY)


def get_gpu_policy() -> dict[str, Any]:
    return dict(_GPU_POLICY)


def select_gpu(idx: int | list[int]) -> None:
    if isinstance(idx, int):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(idx)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in idx)
