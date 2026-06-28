"""metric_catalog.py — 18개 결과 metric을 25* stage 의존성으로 매핑.

플랜 4.6 카탈로그 구현:
- 사용자가 체크한 metric set → 실행해야 할 25* stage set
- coverage_map은 별도 특수 경로 (RadioMapSolver, RX 불필요)
- 도출 metric (pdp/padp/rsrp/ray_stats)은 derived/ 모듈로 처리
- 위상정렬로 실행 순서 보장

사용 예:
    from backend.metric_catalog import resolve_metrics, requires_rx
    plan = resolve_metrics(["rsrp", "swomp_beams"])
    # plan == {
    #     "stages_ordered": ["P1A", "P1B", "P1F", "P1G", "P1H", "P1I", "P1P"],
    #     "derived": ["rsrp"],
    #     "coverage_map": False,
    # }
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


# ---------------------------------------------------------------------------
# 25* stage 의존성 그래프 (X의 prerequisites = DEPENDENCY[X])
# ---------------------------------------------------------------------------
DEPENDENCY: dict[str, set[str]] = {
    "P1A": set(),                              # Ray Tracing 본체
    "P1B": {"P1A"},                            # Valid RX 필터
    "P1C": {"P1A", "P1B"},                     # AE OFDM 채널 (옵션)
    "P1D": {"P1A", "P1B"},                     # Separability
    "P1F": {"P1A", "P1B"},                     # Marginal CCM
    "P1G": {"P1A", "P1B", "P1F"},              # Coupling Matrix
    "P1H": {"P1A", "P1B"},                     # Mean channel
    "P1I": {"P1A", "P1B", "P1F", "P1G", "P1H"},  # Joint H matrix
    "P1J": {"P1A", "P1B", "P1F", "P1G", "P1H", "P1I"},  # Capacity calc
    "P1L": {"P1A", "P1B", "P1F", "P1G", "P1H", "P1I", "P1J"},  # Greedy
    "P1M": {"P1A", "P1B", "P1F", "P1G", "P1H", "P1I", "P1J", "P1L"},  # QIE
    "P1N": {"P1A", "P1B", "P1L"},              # PADP-DFT alignment
    "P1O": {"P1A", "P1B", "P1F", "P1G", "P1H", "P1I", "P1J"},  # Uplink layer/TRX
    "P1P": {"P1A", "P1B", "P1F", "P1G", "P1H", "P1I"},  # SWOMP
    "P1Q": {"P1A", "P1B", "P1F", "P1G", "P1H", "P1I", "P1P"},  # Beam pattern viz
}


# ---------------------------------------------------------------------------
# Metric → 25* stage 매핑 (플랜 4.6 표)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MetricSpec:
    """단일 metric의 메타데이터."""

    metric_id: str
    label: str
    description: str
    stages: tuple[str, ...]           # 25* 단계 (P1A 등)
    derived: bool = False             # backend/derived/derive_*.py로 추가 계산
    coverage_map: bool = False        # Sionna RadioMapSolver 특수 경로
    needs_rx: bool = True             # RX placement 요구 여부
    output_kinds: tuple[str, ...] = ("npz",)


METRIC_CATALOG: dict[str, MetricSpec] = {
    "coverage_map": MetricSpec(
        metric_id="coverage_map",
        label="Coverage Map (TX 기반 광역)",
        description="Sionna RadioMapSolver로 광역 path gain heatmap. RX 불필요.",
        stages=(),
        coverage_map=True,
        needs_rx=False,
        output_kinds=("npz", "png"),
    ),
    "ray_dump": MetricSpec(
        metric_id="ray_dump",
        label="Ray Data 덤프",
        description="P1A 결과를 P1B 필터링까지 거친 raw ray NPZ (tau/power/theta/phi).",
        stages=("P1A", "P1B"),
        output_kinds=("npz",),
    ),
    "pdp": MetricSpec(
        metric_id="pdp",
        label="PDP (Power Delay Profile)",
        description="RX별 (tau, power) → 100 bin histogram + interpolation.",
        stages=("P1A", "P1B"),
        derived=True,
        output_kinds=("npz", "png"),
    ),
    "padp": MetricSpec(
        metric_id="padp",
        label="PADP (Power Angular Delay Profile)",
        description="RX별 (tau × azimuth × zenith, power) 3D heatmap (100×72×36 bin).",
        stages=("P1A", "P1B"),
        derived=True,
        output_kinds=("npz", "png"),
    ),
    "rsrp": MetricSpec(
        metric_id="rsrp",
        label="RSRP map",
        description="RX별 다중 ray power 합 (dB). 2D scatter map + CSV.",
        stages=("P1A", "P1B"),
        derived=True,
        output_kinds=("csv", "png"),
    ),
    "ray_stats": MetricSpec(
        metric_id="ray_stats",
        label="Ray Stats (DS / K / AS)",
        description="Delay spread, Rician K-factor, Angular spread (ASA/ASD).",
        stages=("P1A", "P1B"),
        derived=True,
        output_kinds=("csv",),
    ),
    "ae_ofdm_channel": MetricSpec(
        metric_id="ae_ofdm_channel",
        label="AE OFDM 채널",
        description="안테나 element OFDM 채널 (P1C).",
        stages=("P1A", "P1B", "P1C"),
        output_kinds=("npz",),
    ),
    "marginal_ccm": MetricSpec(
        metric_id="marginal_ccm",
        label="Marginal CCM (R_BS, R_UE)",
        description="P1F: 시간 axis 평균 covariance.",
        stages=("P1A", "P1B", "P1F"),
        output_kinds=("npz",),
    ),
    "coupling_matrix": MetricSpec(
        metric_id="coupling_matrix",
        label="Coupling Matrix Ω",
        description="P1G: U_tx · Ω · U_rx 분해.",
        stages=("P1A", "P1B", "P1F", "P1G"),
        output_kinds=("npz",),
    ),
    "mean_channel": MetricSpec(
        metric_id="mean_channel",
        label="Mean Channel H̄",
        description="P1H: 시간 평균 채널.",
        stages=("P1A", "P1B", "P1H"),
        output_kinds=("npz",),
    ),
    "separability": MetricSpec(
        metric_id="separability",
        label="Separability ε_d, ε_U",
        description="P1D: Weichselberger 가정 검증.",
        stages=("P1A", "P1B", "P1D"),
        output_kinds=("csv",),
    ),
    "su_capacity": MetricSpec(
        metric_id="su_capacity",
        label="SU-MIMO Capacity",
        description="P1J: Wen2011 알고리즘으로 단일 사용자 용량.",
        stages=("P1A", "P1B", "P1F", "P1G", "P1H", "P1I", "P1J"),
        output_kinds=("npz", "csv"),
    ),
    "greedy_beams": MetricSpec(
        metric_id="greedy_beams",
        label="Greedy Beams",
        description="P1L: 그리디 빔 인덱스 + 단계별 용량.",
        stages=("P1A", "P1B", "P1F", "P1G", "P1H", "P1I", "P1J", "P1L"),
        output_kinds=("csv",),
    ),
    "swomp_beams": MetricSpec(
        metric_id="swomp_beams",
        label="SWOMP Beams",
        description="P1P: SWOMP 알고리즘 빔 + perturbation.",
        stages=("P1A", "P1B", "P1F", "P1G", "P1H", "P1I", "P1P"),
        output_kinds=("csv",),
    ),
    "uplink_beams": MetricSpec(
        metric_id="uplink_beams",
        label="Uplink Layer/TRX Beams",
        description="P1O: PADP 기반 uplink 빔.",
        stages=("P1A", "P1B", "P1F", "P1G", "P1H", "P1I", "P1J", "P1O"),
        output_kinds=("csv",),
    ),
    "qie_clustering": MetricSpec(
        metric_id="qie_clustering",
        label="UE Clustering (QIE)",
        description="P1M: Quality-Indexed Ensemble UE clustering.",
        stages=("P1A", "P1B", "P1F", "P1G", "P1H", "P1I", "P1J", "P1L", "P1M"),
        output_kinds=("csv", "png"),
    ),
    "padp_dft_alignment": MetricSpec(
        metric_id="padp_dft_alignment",
        label="PADP-DFT Alignment",
        description="P1N: PADP와 DFT codebook의 각도 정렬도.",
        stages=("P1A", "P1B", "P1L", "P1N"),
        output_kinds=("csv", "png"),
    ),
    "beam_pattern": MetricSpec(
        metric_id="beam_pattern",
        label="Beam Pattern (P1Q)",
        description="P1Q: SWOMP 결과 빔 패턴 시각화.",
        stages=("P1A", "P1B", "P1F", "P1G", "P1H", "P1I", "P1P", "P1Q"),
        output_kinds=("png",),
    ),
    "eigenbeam_pattern": MetricSpec(
        metric_id="eigenbeam_pattern",
        label="Eigenbeam Pattern (P1Q)",
        description="P1Q eigenbeam 모드: R_BS/R_UE 고유벡터 패턴.",
        stages=("P1A", "P1B", "P1F", "P1G", "P1H", "P1I", "P1P", "P1Q"),
        output_kinds=("png",),
    ),
}

assert len(METRIC_CATALOG) == 19, "Plan declared 18+coverage_map; reconcile counts"
# (coverage_map 포함 19개 entry; 단 사용자 노출 metric 카운트는 18로 표기됨)

ALL_METRIC_IDS: tuple[str, ...] = tuple(METRIC_CATALOG.keys())


# ---------------------------------------------------------------------------
# 의존성 해소 (위상정렬)
# ---------------------------------------------------------------------------
@dataclass
class ResolvedPlan:
    """metric_catalog.resolve_metrics() 결과."""

    stages_ordered: list[str]          # P1* 25* 단계 위상정렬
    derived: list[str]                 # 도출 metric (pdp/padp/rsrp/ray_stats)
    coverage_map: bool                 # Coverage Map 어댑터 호출 여부
    needs_rx: bool                     # 하나라도 RX 요구하면 True
    selected: list[str]                # 사용자가 선택한 metric (소문자 정렬)
    unknown: list[str]                 # 카탈로그에 없는 metric (경고용)

    def to_dict(self) -> dict:
        return {
            "stages_ordered": list(self.stages_ordered),
            "derived": list(self.derived),
            "coverage_map": self.coverage_map,
            "needs_rx": self.needs_rx,
            "selected": list(self.selected),
            "unknown": list(self.unknown),
        }


def _topological_sort(stages: set[str]) -> list[str]:
    """DEPENDENCY 그래프에 따라 stages를 위상정렬."""

    # Kahn's algorithm restricted to `stages`
    in_degree: dict[str, int] = {}
    for s in stages:
        in_degree[s] = sum(1 for prereq in DEPENDENCY.get(s, set()) if prereq in stages)

    queue = [s for s in stages if in_degree[s] == 0]
    queue.sort()  # 안정적 순서 (알파벳)
    result: list[str] = []
    while queue:
        node = queue.pop(0)
        result.append(node)
        # 이 node를 prereq로 갖는 다른 stage들의 in_degree를 감소
        for other in stages:
            if node in DEPENDENCY.get(other, set()):
                in_degree[other] -= 1
                if in_degree[other] == 0:
                    queue.append(other)
        queue.sort()

    if len(result) != len(stages):
        raise RuntimeError(
            f"Cycle or missing dependency in stages {stages} -> {result}"
        )
    return result


def resolve_metrics(selected_metrics: Iterable[str]) -> ResolvedPlan:
    """선택된 metric → 실행 plan으로 변환.

    Parameters
    ----------
    selected_metrics : Iterable[str]
        사용자가 체크한 metric_id 리스트.

    Returns
    -------
    ResolvedPlan
        stages_ordered, derived, coverage_map 등 실행 plan.
    """

    selected = sorted({m for m in selected_metrics})
    unknown = [m for m in selected if m not in METRIC_CATALOG]
    known = [m for m in selected if m in METRIC_CATALOG]

    required_stages: set[str] = set()
    derived: list[str] = []
    coverage_map = False
    needs_rx = False

    for m in known:
        spec = METRIC_CATALOG[m]
        # stages는 직접 의존성 + 그래프 prereq을 함께 수집
        for s in spec.stages:
            required_stages.add(s)
            required_stages.update(DEPENDENCY.get(s, set()))
        if spec.derived:
            derived.append(m)
        if spec.coverage_map:
            coverage_map = True
        if spec.needs_rx:
            needs_rx = True

    stages_ordered = _topological_sort(required_stages)

    return ResolvedPlan(
        stages_ordered=stages_ordered,
        derived=sorted(set(derived)),
        coverage_map=coverage_map,
        needs_rx=needs_rx,
        selected=known,
        unknown=unknown,
    )


def requires_rx(selected_metrics: Iterable[str]) -> bool:
    """RX placement이 필요한지 빠르게 검사 (UI 가드용)."""

    for m in selected_metrics:
        spec = METRIC_CATALOG.get(m)
        if spec is None:
            continue
        if spec.needs_rx:
            return True
    return False


# ---------------------------------------------------------------------------
# 프리셋 (플랜 4.10 표) — 7종
# ---------------------------------------------------------------------------
PRESETS: dict[str, dict] = {
    "coverage_only": {
        "label": "⓪ Coverage Map only",
        "description": "TX만 두고 광역 path gain heatmap. RX 불필요.",
        "metrics": ["coverage_map"],
    },
    "rt_only": {
        "label": "① RT only",
        "description": "Ray Data 덤프만 (P1A → P1B).",
        "metrics": ["ray_dump"],
    },
    "rt_ccm": {
        "label": "② RT + CCM",
        "description": "Ray + Marginal CCM + Coupling Matrix + Mean Channel.",
        "metrics": ["ray_dump", "marginal_ccm", "coupling_matrix", "mean_channel"],
    },
    "rt_ccm_swomp": {
        "label": "③ RT + CCM + SWOMP",
        "description": "위 + SU-MIMO Capacity + SWOMP Beams.",
        "metrics": [
            "ray_dump", "marginal_ccm", "coupling_matrix", "mean_channel",
            "su_capacity", "swomp_beams",
        ],
    },
    "rt_ccm_greedy": {
        "label": "④ RT + CCM + Greedy",
        "description": "위 + SU-MIMO Capacity + Greedy Beams.",
        "metrics": [
            "ray_dump", "marginal_ccm", "coupling_matrix", "mean_channel",
            "su_capacity", "greedy_beams",
        ],
    },
    "ray_viz": {
        "label": "⑤ Ray Visualization",
        "description": "PDP + PADP + RSRP + Ray Stats (도출 metric 4종).",
        "metrics": ["pdp", "padp", "rsrp", "ray_stats"],
    },
    "bm_full": {
        "label": "⑥ BM Full",
        "description": "Greedy + SWOMP + Uplink + QIE + Beam Pattern + Eigenbeam Pattern.",
        "metrics": [
            "ray_dump", "marginal_ccm", "coupling_matrix", "mean_channel",
            "su_capacity",
            "greedy_beams", "swomp_beams", "uplink_beams",
            "qie_clustering", "beam_pattern", "eigenbeam_pattern",
        ],
    },
}


def list_user_visible_metrics() -> list[dict]:
    """프론트엔드 ResultMetricsPage용 카탈로그 export."""

    return [
        {
            "id": spec.metric_id,
            "label": spec.label,
            "description": spec.description,
            "stages": list(spec.stages),
            "derived": spec.derived,
            "coverage_map": spec.coverage_map,
            "needs_rx": spec.needs_rx,
            "output_kinds": list(spec.output_kinds),
        }
        for spec in METRIC_CATALOG.values()
    ]


if __name__ == "__main__":
    import json

    print("--- Catalog ---")
    print(f"Total entries: {len(METRIC_CATALOG)} (= 18 user metrics + coverage_map)")
    for spec in METRIC_CATALOG.values():
        flags = []
        if spec.derived: flags.append("derived")
        if spec.coverage_map: flags.append("coverage")
        if not spec.needs_rx: flags.append("no-rx")
        print(f"  {spec.metric_id:20s}  stages={list(spec.stages)}  {flags}")

    print("\n--- Resolve examples ---")
    for sel in [
        ["rsrp"],
        ["rsrp", "swomp_beams"],
        ["coverage_map"],
        ["coverage_map", "rsrp"],
        ["beam_pattern", "qie_clustering"],
        ["padp_dft_alignment"],
    ]:
        plan = resolve_metrics(sel)
        print(f"  selected={sel}")
        print(f"    -> {json.dumps(plan.to_dict())}")

    print("\n--- Presets ---")
    for k, p in PRESETS.items():
        print(f"  [{k}] {p['label']}: {p['metrics']}")
