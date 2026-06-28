"""25* 파이프라인 어댑터 모음.

각 P1* 모듈은 25* 폴더의 원본 (또는 forked_25x의 fork)을 import 하여
Config override + Pipeline.execute_main() 패턴으로 호출.

``coverage_map_adapter``는 ``import mitsuba`` 부수효과로 drjit-cuda 컨텍스트를 잡아
다른 stage의 TF GPU 초기화(cuInit)를 방해할 수 있으므로 **lazy** 로 노출한다.
"""

from .base_adapter import BaseAdapter, StageMeta, STAGE_REGISTRY  # noqa: F401


def __getattr__(name):
    if name in {"AntennaSimpleCfg", "CoverageMapConfig", "TXPlacement", "run_coverage_map"}:
        from . import coverage_map_adapter as _mod

        return getattr(_mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
