"""p1_adapters.py — P1A~P1Q 13개 thin wrapper.

각 wrapper는 `BaseAdapter`를 사용해 session_dir + config_overrides만 전달.
실제 stage 정의는 base_adapter.STAGE_REGISTRY 에 있다.

사용 예
-------
from backend.adapters.p1_adapters import run_p1a, run_p1f
run_p1a(session_dir, scene_xml_path="...", rt_overrides={"MAX_DEPTH": 8, ...})
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base_adapter import BaseAdapter, run_stages


# ---------------------------------------------------------------------------
# 개별 wrapper (각 stage별 의미 있는 키워드 정리용)
# ---------------------------------------------------------------------------
def run_p1a(
    session_dir: Path,
    scene_xml_path: str,
    area_configs: dict | None = None,
    area_indices: list[int] | None = None,
    frequency_ghz_list: list[float] | None = None,
    rt_overrides: dict | None = None,
) -> dict:
    """P1A Ray Tracing 실행.

    Parameters
    ----------
    scene_xml_path : str
        세션의 scene.xml 절대 경로 (P1A fork의 WEB_OVERRIDE_SCENE_PATH).
    area_configs : dict | None
        25* AREA_CONFIGS 형식 dict. 예: {"area_1": {"tx_positions": [...], "rx_placement": {...}, ...}}
    area_indices : list[int]
        예: [1]
    frequency_ghz_list : list[float]
        예: [7.5]
    rt_overrides : dict | None
        MAX_DEPTH, RT_SEED, PATHSOLVER_*, NUM_INTERESTING_PATHS 등 P1A_Config 키.
    """

    overrides: dict[str, Any] = {
        "WEB_OVERRIDE_SCENE_PATH": str(scene_xml_path),
        "ENABLE_SCENE_PREVIEW": False,  # 서버 headless
    }
    if area_configs is not None:
        overrides["AREA_CONFIGS"] = area_configs
    if area_indices is not None:
        overrides["AREA_INDICES"] = list(area_indices)
    if frequency_ghz_list is not None:
        overrides["FREQUENCY_CONFIGS"] = list(frequency_ghz_list)
    if rt_overrides:
        overrides.update(rt_overrides)
    return BaseAdapter("P1A").run(Path(session_dir), config_overrides=overrides)


def run_p1b(session_dir: Path, overrides: dict | None = None) -> dict:
    """P1B Valid RX filter."""

    return BaseAdapter("P1B").run(Path(session_dir), config_overrides=overrides or {})


def run_p1c(session_dir: Path, overrides: dict | None = None) -> dict:
    """P1C AE OFDM channel."""

    return BaseAdapter("P1C").run(Path(session_dir), config_overrides=overrides or {})


def run_p1d(session_dir: Path, overrides: dict | None = None) -> dict:
    """P1D Separability."""

    return BaseAdapter("P1D").run(Path(session_dir), config_overrides=overrides or {})


def run_p1f(session_dir: Path, overrides: dict | None = None) -> dict:
    """P1F Marginal CCM (R_BS, R_UE).

    PanelArray 관련 overrides가 필요 — antenna_resolver.simple_to_advanced 결과를
    Config 필드(BS_NUM_ROWS_PER_PANEL 등)에 매핑해서 전달.
    """

    return BaseAdapter("P1F").run(Path(session_dir), config_overrides=overrides or {})


def run_p1g(session_dir: Path, overrides: dict | None = None) -> dict:
    """P1G Coupling Matrix Ω."""

    return BaseAdapter("P1G").run(Path(session_dir), config_overrides=overrides or {})


def run_p1h(session_dir: Path, overrides: dict | None = None) -> dict:
    """P1H Mean Channel H̄."""

    return BaseAdapter("P1H").run(Path(session_dir), config_overrides=overrides or {})


def run_p1i(session_dir: Path, overrides: dict | None = None) -> dict:
    """P1I Weichselberger chunk (joint H matrix)."""

    return BaseAdapter("P1I").run(Path(session_dir), config_overrides=overrides or {})


def run_p1j(session_dir: Path, overrides: dict | None = None) -> dict:
    """P1J SU-MIMO Capacity (Wen2011)."""

    return BaseAdapter("P1J").run(Path(session_dir), config_overrides=overrides or {})


def run_p1l(session_dir: Path, overrides: dict | None = None) -> dict:
    """P1L Greedy Beam Management."""

    return BaseAdapter("P1L").run(Path(session_dir), config_overrides=overrides or {})


def run_p1m(session_dir: Path, overrides: dict | None = None) -> dict:
    """P1M QIE UE Clustering."""

    return BaseAdapter("P1M").run(Path(session_dir), config_overrides=overrides or {})


def run_p1n(session_dir: Path, overrides: dict | None = None) -> dict:
    """P1N PADP-DFT Alignment."""

    return BaseAdapter("P1N").run(Path(session_dir), config_overrides=overrides or {})


def run_p1o(session_dir: Path, overrides: dict | None = None) -> dict:
    """P1O Uplink Layer/TRX Beams."""

    return BaseAdapter("P1O").run(Path(session_dir), config_overrides=overrides or {})


def run_p1p(session_dir: Path, overrides: dict | None = None) -> dict:
    """P1P SWOMP Beam Management."""

    return BaseAdapter("P1P").run(Path(session_dir), config_overrides=overrides or {})


def run_p1q(session_dir: Path, overrides: dict | None = None) -> dict:
    """P1Q Beam Pattern Plot."""

    return BaseAdapter("P1Q").run(Path(session_dir), config_overrides=overrides or {})


# ---------------------------------------------------------------------------
# stage_id → callable lookup (BaseAdapter direct도 가능하지만 wrapper 권장 시그니처)
# ---------------------------------------------------------------------------
STAGE_RUNNERS = {
    "P1A": run_p1a,
    "P1B": run_p1b,
    "P1C": run_p1c,
    "P1D": run_p1d,
    "P1F": run_p1f,
    "P1G": run_p1g,
    "P1H": run_p1h,
    "P1I": run_p1i,
    "P1J": run_p1j,
    "P1L": run_p1l,
    "P1M": run_p1m,
    "P1N": run_p1n,
    "P1O": run_p1o,
    "P1P": run_p1p,
    "P1Q": run_p1q,
}


def run_stage(stage_id: str, session_dir: Path, **kwargs) -> dict:
    """stage_id로 wrapper 자동 디스패치."""

    runner = STAGE_RUNNERS.get(stage_id)
    if runner is None:
        raise KeyError(f"Unknown stage_id: {stage_id}")
    return runner(Path(session_dir), **kwargs)


__all__ = [
    "STAGE_RUNNERS",
    "run_p1a", "run_p1b", "run_p1c", "run_p1d",
    "run_p1f", "run_p1g", "run_p1h", "run_p1i",
    "run_p1j", "run_p1l", "run_p1m", "run_p1n",
    "run_p1o", "run_p1p", "run_p1q",
    "run_stage", "run_stages",
]
