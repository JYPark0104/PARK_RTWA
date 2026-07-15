"""schemas.py — FastAPI ↔ React 사이 Pydantic 스키마.

Plan 4.4 / 5.x 요구사항을 그대로 구현.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# TX / RX placement
# ---------------------------------------------------------------------------
Coord3 = tuple[float, float, float]


class TXConfig(BaseModel):
    """단일 TX."""

    position: Coord3
    orientation: Coord3 = (0.0, 0.0, 0.0)
    name: str = "tx"


class RXGridConfig(BaseModel):
    """자동 RX 그리드 (4 method 중 하나)."""

    method: Literal["grid", "explicit", "radial", "street", "ground_grid", "facade"] = "grid"

    # grid
    x_start: float | None = None
    x_stop: float | None = None
    x_num: int | None = None
    y_start: float | None = None
    y_stop: float | None = None
    y_num: int | None = None
    z_values: list[float] = Field(default_factory=lambda: [1.5])

    # ground_grid (PARK_2 방식: 경계 bbox → grid_n×grid_n 후보 → 하향 레이캐스팅으로
    #              지면 hit 위치만, 지면 위 rx_height 로 RX 배치)
    grid_n: int | None = None              # 한 축당 후보 수 (총 grid_n×grid_n)
    margin: float | None = None            # bbox 가장자리 제거 비율 (0~0.5)
    rx_height: float | None = None         # 지면으로부터 RX 높이 (m)
    raycasting_z: float | None = None      # 레이 시작 높이 (None=자동: bbox 최고점+10)
    max_height: float | None = None        # 지면고도+rx_height 가 이 값 초과면 RX 제거 (None=제한없음)
    spacing: float | None = None           # 격자 간격(m, 정사각). 지정 시 grid_n 대신 이 간격으로 배치

    # facade (O2I, 건물 벽면): z=k 평면과 건물 수직면의 교선을 따라 RX 배치.
    #   벽 바깥 법선 방향으로 epsilon(m) 이격. host 건물/재질/법선/높이층을 메타로 기록.
    #   (재질 무관: 모든 건물 오브젝트의 수직면 대상)
    z_min: float | None = None            # 최저 높이 [m]
    z_max: float | None = None            # 최대 높이 [m]
    z_distance: float | None = None       # 높이 간격 [m] (z_min, z_min+z_distance, ...)
    facade_spacing: float | None = None   # 컨투어를 따라 RX 간격 [m]
    facade_epsilon: float | None = None   # 벽 바깥 이격 거리 [m] (표면 self-occlusion 방지)
    facade_max_normal_z: float | None = None  # |face normal_z|>이 값이면 수평면(지붕/지면)으로 배제
    # facade XY 경계(bounding region): 4개 모두 주어지면 그 사각형 안 벽면만, None 이면 맵 전체
    facade_x_min: float | None = None
    facade_x_max: float | None = None
    facade_y_min: float | None = None
    facade_y_max: float | None = None

    # explicit
    x_coords: list[float] | None = None
    y_coords: list[float] | None = None

    # radial
    center_xy: tuple[float, float] | None = None
    radii_m: list[float] | None = None
    angles_start: float | None = None
    angles_stop: float | None = None
    angles_num: int | None = None

    # street
    path_points: list[tuple[float, float]] | None = None
    num_points: int | None = None


class RXClickConfig(BaseModel):
    """수동 마우스 클릭 배치 (snap-to-surface)."""

    positions: list[Coord3] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Antenna
# ---------------------------------------------------------------------------
class AntennaSimple(BaseModel):
    """일반 모드: rows × cols만."""

    bs_rows: int = 32
    bs_cols: int = 32
    ue_rows: int = 4
    ue_cols: int = 4


class PanelDict(BaseModel):
    """고급 모드: PanelArray dict."""

    num_rows_per_panel: int
    num_cols_per_panel: int
    num_rows: int
    num_cols: int
    vertical_spacing_panel: float
    horizontal_spacing_panel: float
    vertical_spacing_element: float
    horizontal_spacing_element: float
    pattern: str = "tr38901"
    polarization: str = "V"


class AntennaConfig(BaseModel):
    """일반/고급 모드 dual."""

    mode: Literal["simple", "advanced"] = "simple"
    simple: AntennaSimple = AntennaSimple()
    bs_panel: PanelDict | None = None
    ue_panel: PanelDict | None = None


# ---------------------------------------------------------------------------
# RT options
# ---------------------------------------------------------------------------
class CoverageMapOpts(BaseModel):
    """Coverage Map (RadioMapSolver) 옵션."""

    enabled: bool = False
    cell_size_x: float = 1.0
    cell_size_y: float = 1.0
    height_m: float = 1.5
    samples_per_tx: float = 1e8
    max_depth: int = 5
    specular_reflection: bool = True
    diffuse_reflection: bool = True
    refraction: bool = True
    center_xy: tuple[float, float] | None = None
    size_xy: tuple[float, float] | None = None


class RTConfig(BaseModel):
    """RT 옵션 — simple (4 필드) + advanced (전체 PathSolver/RayGen)."""

    mode: Literal["simple", "advanced"] = "simple"

    # --- RT 엔진 선택 (P1A 표준 / PARK_2 batch RT / Intg 통합) ---
    #   intg: batch 코어로 RT → 단일 multi-TX superset NPZ + reshaper(P1A/channel) 뷰 생성.
    engine: Literal["p1a", "batch", "intg"] = "p1a"
    batch_size: int = 50              # batch/intg 엔진 전용: 배치당 RX 개수
    random_batch: bool = False        # batch 엔진: RX 순서를 무작위 셔플해 배치 구성 (intg 는 항상 ON)
    random_batch_seed: int | None = None  # 재현용 시드(None=매번 무작위)

    frequency_ghz: float = 7.5
    max_depth: int = 5
    seed: int = 41
    num_interesting_paths: int = 20
    max_rays_per_pair: int = 400

    pathsolver_los: bool = True
    pathsolver_specular_reflection: bool = True
    pathsolver_diffuse_reflection: bool = True
    pathsolver_refraction: bool = True
    pathsolver_synthetic_array: bool = False

    itu_scattering_coeff: float = 0.2
    itu_xpd_coeff: float = 0.5
    # 재질별 산란계수 오버라이드 {재질명: S}. 비어있으면 전역 itu_scattering_coeff 사용.
    #   3.TX/RX 'Material Properties' 섹션에서 설정. (2026-07-06)
    material_scattering: dict[str, float] = {}

    # --- TX 배치 ---
    tx_ground_offset_m: float = 2.0   # 클릭 지점 지면으로부터 TX 이격 높이 (m)

    # --- 고급 옵션 (PARK_2 config.yaml 이식) ---
    #   PathSolver 추가 플래그
    pathsolver_diffraction: bool = False
    pathsolver_edge_diffraction: bool = False
    pathsolver_diffraction_lit_region: bool = True
    num_samples: int = 100000          # samples_per_src
    max_num_paths: int = 10000         # max_num_paths_per_src
    # POWER_OFFSET [dB]: RSRP/경로전력(dBm) = 10log10(|a|^2) + power_offset.
    #   Sionna cir() |a|^2 은 TX power 미포함 순수 채널이득 → 이 값이 유효 송신전력 기준[dBm].
    #   기본 30 dBm(=1W). batch/intg 엔진(m5 후처리)에서 사용. (2026-07-02 추가)
    power_offset: float = 30.0
    #   RadioMaterial
    relative_permittivity: float = 5.24
    conductivity: float = 0.0462
    material_thickness: float = 0.1
    scattering_pattern: Literal["lambertian", "directive", "backscattering"] = "lambertian"
    directive_alpha_r: int = 10
    backscattering_alpha_r: int = 20
    backscattering_alpha_i: int = 30
    backscattering_lambda: float = 0.7
    #   안테나 패턴/편파 (rows×cols 는 antenna 섹션에서)
    tx_pattern: str = "iso"
    tx_polarization: str = "V"
    rx_pattern: str = "dipole"
    rx_polarization: str = "V"

    coverage_map: CoverageMapOpts = CoverageMapOpts()


# ---------------------------------------------------------------------------
# Metric selection
# ---------------------------------------------------------------------------
class MetricSelection(BaseModel):
    """체크된 metric 목록."""

    metrics: list[str] = Field(default_factory=list)
    preset: str | None = None  # 사용자가 누른 프리셋 id (옵션)


# ---------------------------------------------------------------------------
# Session create / update
# ---------------------------------------------------------------------------
class SessionCreateRequest(BaseModel):
    scene_name: str
    antenna: AntennaConfig
    metrics: MetricSelection
    notes: str = ""
    user_id: str = ""
    user_name: str = ""


class UserCreateRequest(BaseModel):
    name: str


class UserRenameRequest(BaseModel):
    name: str


class SessionLabelUpdate(BaseModel):
    label: str


class SessionInfo(BaseModel):
    uuid: str
    label: str
    created_at_kst: str
    scene_name: str
    display_name: str = ""
    bs_rows: int
    bs_cols: int
    ue_rows: int
    ue_cols: int
    metrics: list[str]
    notes: str = ""


# ---------------------------------------------------------------------------
# Job / progress
# ---------------------------------------------------------------------------
class JobSubmitRequest(BaseModel):
    """전체 잡 제출 페이로드."""

    session_uuid: str
    tx_list: list[TXConfig]
    rx_grid: RXGridConfig | None = None
    rx_clicks: RXClickConfig | None = None
    antenna: AntennaConfig
    rt: RTConfig
    metrics: MetricSelection


class JobStatus(BaseModel):
    job_id: str
    session_uuid: str
    state: Literal["pending", "running", "succeeded", "failed", "cancelled"] = "pending"
    progress: float = 0.0       # 0.0 ~ 1.0
    current_stage: str = ""
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    stages_done: list[str] = Field(default_factory=list)
    output_paths: dict[str, str] = Field(default_factory=dict)


class ProgressEvent(BaseModel):
    """WebSocket으로 푸시되는 단일 이벤트."""

    job_id: str
    kind: Literal["log", "stage_start", "stage_end", "progress", "done", "error"]
    message: str = ""
    stage: str = ""
    progress: float | None = None
    timestamp: str = ""
