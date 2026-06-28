"""
default_config.py
=================
Ray Tracing Agent 파이프라인 전체에서 공유되는 데이터 클래스 및 기본 파라미터 정의.

설정 우선순위:
  configs/config.yaml (최우선) > CLI 인수 > 이 파일의 dataclass 기본값

데이터 흐름:
  RT_Config  →  [Scene_Agent]  →  [RT_Agent]  →  RawRayData
                                                      ↓
                                              [PostProcess_Agent]
                                                      ↓
                                              SimulationResult  (Output 1 → Channel Agent)
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np
import os
import yaml


# ============================================================
# VisualizationConfig: 시각화 전용 파라미터 (USDA export)
# ============================================================
@dataclass
class VisualizationConfig:
    max_rays_per_rx:  int   = 8    # Blender 시각화용 RX당 최대 Ray 수
    ray_width:        float = 1.5  # Ray 선 굵기 (USDA BasisCurves width)
    tx_radius:        float = 2.0  # TX 구체 반지름
    rx_radius_target: float = 1.8  # Target RX 구체 반지름
    rx_radius_other:  float = 1.2  # 일반 RX 구체 반지름

    # --- 격자 그림 배경: 실제 Twin Map 상공(top-down) 뷰 오버레이 ---
    map_overlay:        bool  = True       # rx_positions/hitmap/rsrp 배경에 맵 외곽선 표시
    map_overlay_z_min:  Optional[float] = None  # 지면 제거 높이. None(=auto)이면 PLY에서 자동 추정
    map_overlay_alpha:  float = 0.5        # 외곽선 투명도
    map_overlay_color:  str   = "#444444"  # 외곽선 색


# ============================================================
# RT_Config: 파이프라인 전체 파라미터
# ============================================================
@dataclass
class RT_Config:
    """
    Ray Tracing Agent 파이프라인 전체에서 공유되는 파라미터 설정 객체.
    configs/config.yaml이 있으면 그 값을 우선 사용합니다.
    """

    # --- 맵 설정 ---
    map_xml: str = "(4) 2DGS-planar.xml"
    map_ply: str = "(4) 2DGS_2048_qem100k+planar(angle8)_synch.ply"
    map_title: str = "4. 2DGS (100k+Planar)"
    # 지면 레이캐스팅용 전체 PLY 목록 (Geo-Radio: 재질별로 쪼개진 모든 PLY).
    # 비어있으면 map_ply 단일을 사용. (2026-06-20 추가)
    map_ply_all: List[str] = field(default_factory=list)

    # --- RT 모드 ---
    rt_mode: str = "multi_rx"   # "multi_rx" | "single_rx" | "batch_rx"
    single_rx_position: Tuple[float, float, float] = None  # single_rx 모드 전용
    batch_size: int = 50  # batch_rx 모드: 배치당 RX 개수

    # --- TX 설정 (Multi-TX 지원) ---
    # tx_positions: 여러 TX 위치 목록 [[x, y, z], ...]
    # tx_position : "현재 처리 중인 TX" (per-TX 루프에서 매 반복마다 설정됨).
    #               기존 단일 TX용 함수들과의 호환을 위해 유지. 기본값=tx_positions[0]
    tx_positions: List[List[float]] = field(
        default_factory=lambda: [[-12.517, 14.894, 41.0]]
    )
    tx_position: Tuple[float, float, float] = (-12.517, 14.894, 41.0)

    # --- RX 격자 설정 ---
    grid_n: int = 20
    rx_height: float = 1.5
    raycasting_z: float = 200.0
    margin: float = 0.15   # bounding box 가장자리 제거 비율

    # --- PathSolver 파라미터 ---
    num_samples: int = 500000
    max_depth: int = 7
    max_num_paths: int = 10000
    seed: int = 41
    synthetic_array: bool = True
    los: bool = True
    specular_reflection: bool = True
    diffuse_reflection: bool = True
    refraction: bool = True
    diffraction: bool = False
    edge_diffraction: bool = False
    diffraction_lit_region: bool = True

    # --- RadioMaterial 파라미터 ---
    relative_permittivity: float = 5.24
    conductivity: float = 0.0462
    thickness: float = 0.1
    scattering_coefficient: float = 0.9
    xpd_coefficient: float = 0.0
    scattering_pattern: str = "lambertian"   # "lambertian" | "directive" | "backscattering"
    directive_alpha_r: int = 10
    backscattering_alpha_r: int = 20
    backscattering_alpha_i: int = 30
    backscattering_lambda: float = 0.7

    # --- 안테나 파라미터 ---
    tx_pattern: str = "iso"
    tx_polarization: str = "V"
    rx_pattern: str = "dipole"
    rx_polarization: str = "V"

    # --- 통신 파라미터 ---
    frequency: float = 7e9
    # 안테나 포트 수 = 행 × 열 × (편파당 포트 수). 공분산 행렬 크기를 결정한다.
    num_tx_ant: int = 1
    num_rx_ant: int = 1
    # PlanarArray 행/열 (실제 Sionna 씬 안테나 배열을 구성할 때 사용).
    #   예) 4×4 평면배열 → num_tx_rows=4, num_tx_cols=4
    num_tx_rows: int = 1
    num_tx_cols: int = 1
    num_rx_rows: int = 1
    num_rx_cols: int = 1

    # --- 후처리 파라미터 ---
    threshold_watt: float = 1e-20

    # --- 분석 설정 ---
    target_tx_index: int = 0   # 상세 분석(PDP/PADP/cov) 대상 TX 인덱스 (0-based)
    target_rx_index: int = 0   # 상세 분석 대상 RX 인덱스 (0-based)

    @property
    def num_tx(self) -> int:
        return len(self.tx_positions) if self.tx_positions else 1

    # --- 출력 설정 ---
    output_dir: str = "output"
    visualize_ray_tracing: bool = False

    # --- 시각화 설정 ---
    viz: VisualizationConfig = field(default_factory=VisualizationConfig)

    # --- 내부 상태 (Config_Agent가 채움) ---
    rx_positions: List[List[float]] = field(default_factory=list)

    # --- 출력 하위 디렉토리 (output_dir 기준으로 자동 파생) ---
    @property
    def output_dir_channel(self) -> str:
        """Output 1 (Channel Agent용 channel_data_*.npz) 저장 경로."""
        return os.path.join(self.output_dir, "output_channel_data")

    @property
    def output_dir_viz(self) -> str:
        """Output 2 (PNG 차트 / OBJ / USD / USDA / 렌더) 저장 경로."""
        return os.path.join(self.output_dir, "output_USDA_and_check")


# ============================================================
# YAML 로더: configs/config.yaml → RT_Config
# ============================================================
def load_config(yaml_path: str = None) -> RT_Config:
    """
    configs/config.yaml을 읽어 RT_Config 객체를 반환한다.

    Args:
        yaml_path: config.yaml 경로 (None이면 자동 탐색)

    Returns:
        RT_Config 객체
    """
    if yaml_path is None:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        candidates = [
            os.path.join(base, "configs", "config.yaml"),
            os.path.join(base, "config.yaml"),
            "configs/config.yaml",
            "config.yaml",
        ]
        for c in candidates:
            if os.path.exists(c):
                yaml_path = c
                break

    if yaml_path is None or not os.path.exists(yaml_path):
        print("   ℹ️  config.yaml 없음 → dataclass 기본값 사용")
        return RT_Config()

    print(f"   📄 config.yaml 로드: {yaml_path}")
    with open(yaml_path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)

    # ── {FILENAME} 플레이스홀더 치환 ───────────────────────────
    # 최상위 FILENAME 값을 정의하면, config 전체의 모든 문자열에 들어있는
    # "{FILENAME}" 를 그 값으로 자동 치환한다.
    # → 맵을 바꿀 때 FILENAME 한 곳만 수정하면 map.xml/ply/title, output.dir 등이
    #   모두 함께 갱신된다.
    filename = y.get("FILENAME") if isinstance(y, dict) else None
    if filename:
        filename = str(filename)

        def _subst_filename(obj):
            if isinstance(obj, str):
                return obj.replace("{FILENAME}", filename)
            if isinstance(obj, dict):
                return {k: _subst_filename(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_subst_filename(v) for v in obj]
            return obj

        y = _subst_filename(y)
        print(f"   🔤 FILENAME = '{filename}' → 경로/이름 자동 치환 적용")

    cfg = RT_Config()

    # rt_mode
    if "rt_mode" in y:
        cfg.rt_mode = str(y["rt_mode"])

    # batch_size
    if "batch_size" in y:
        cfg.batch_size = int(y["batch_size"])

    # single_rx_position
    srx = y.get("single_rx_position", {})
    if all(k in srx for k in ("x", "y", "z")):
        cfg.single_rx_position = (float(srx["x"]), float(srx["y"]), float(srx["z"]))

    # map
    m = y.get("map", {})
    if "xml"   in m: cfg.map_xml   = m["xml"]
    if "ply"   in m: cfg.map_ply   = m["ply"]
    if "title" in m: cfg.map_title = m["title"]

    # tx (Multi-TX 지원)
    #   우선순위: tx_positions(리스트) > tx(리스트) > tx(단일 dict, 레거시)
    def _parse_tx_list(raw):
        """raw → [[x,y,z], ...] 정규화. 실패 시 None."""
        if raw is None:
            return None
        # 단일 dict {x,y,z} (레거시)
        if isinstance(raw, dict) and all(k in raw for k in ("x", "y", "z")):
            return [[float(raw["x"]), float(raw["y"]), float(raw["z"])]]
        # 리스트 형태
        if isinstance(raw, (list, tuple)) and len(raw) > 0:
            out = []
            for item in raw:
                if isinstance(item, dict) and all(k in item for k in ("x", "y", "z")):
                    out.append([float(item["x"]), float(item["y"]), float(item["z"])])
                elif isinstance(item, (list, tuple)) and len(item) >= 3:
                    out.append([float(item[0]), float(item[1]), float(item[2])])
            return out if out else None
        return None

    tx_list = _parse_tx_list(y.get("tx_positions", None))
    if tx_list is None:
        tx_list = _parse_tx_list(y.get("tx", None))
    if tx_list:
        cfg.tx_positions = tx_list
        cfg.tx_position = tuple(tx_list[0])

    # rx
    r = y.get("rx", {})
    if "grid_n"       in r: cfg.grid_n       = int(r["grid_n"])
    if "height"       in r: cfg.rx_height    = float(r["height"])
    if "raycasting_z" in r: cfg.raycasting_z = float(r["raycasting_z"])
    if "margin"       in r: cfg.margin       = float(r["margin"])

    # ray_tracing (PathSolver)
    rt = y.get("ray_tracing", {})
    if "num_samples"            in rt: cfg.num_samples            = int(rt["num_samples"])
    if "max_depth"              in rt: cfg.max_depth              = int(rt["max_depth"])
    if "max_num_paths"          in rt: cfg.max_num_paths          = int(rt["max_num_paths"])
    if "seed"                   in rt: cfg.seed                   = int(rt["seed"])
    if "synthetic_array"        in rt: cfg.synthetic_array        = bool(rt["synthetic_array"])
    if "los"                    in rt: cfg.los                    = bool(rt["los"])
    if "specular_reflection"    in rt: cfg.specular_reflection    = bool(rt["specular_reflection"])
    if "diffuse_reflection"     in rt: cfg.diffuse_reflection     = bool(rt["diffuse_reflection"])
    if "refraction"             in rt: cfg.refraction             = bool(rt["refraction"])
    if "diffraction"            in rt: cfg.diffraction            = bool(rt["diffraction"])
    if "edge_diffraction"       in rt: cfg.edge_diffraction       = bool(rt["edge_diffraction"])
    if "diffraction_lit_region" in rt: cfg.diffraction_lit_region = bool(rt["diffraction_lit_region"])

    # material (RadioMaterial)
    mat = y.get("material", {})
    if "relative_permittivity"  in mat: cfg.relative_permittivity  = float(mat["relative_permittivity"])
    if "conductivity"           in mat: cfg.conductivity           = float(mat["conductivity"])
    if "thickness"              in mat: cfg.thickness              = float(mat["thickness"])
    if "scattering_coefficient" in mat: cfg.scattering_coefficient = float(mat["scattering_coefficient"])
    if "xpd_coefficient"        in mat: cfg.xpd_coefficient        = float(mat["xpd_coefficient"])
    if "scattering_pattern"     in mat: cfg.scattering_pattern     = str(mat["scattering_pattern"])
    if "directive_alpha_r"      in mat: cfg.directive_alpha_r      = int(mat["directive_alpha_r"])
    if "backscattering_alpha_r" in mat: cfg.backscattering_alpha_r = int(mat["backscattering_alpha_r"])
    if "backscattering_alpha_i" in mat: cfg.backscattering_alpha_i = int(mat["backscattering_alpha_i"])
    if "backscattering_lambda"  in mat: cfg.backscattering_lambda  = float(mat["backscattering_lambda"])

    # antenna (PlanarArray)
    #   안테나 포트 수 = 행 × 열 × 편파당 포트 수.
    #   "V"/"H" → 단일 편파(×1), "VH"/"cross"/"dual" → 이중 편파(×2).
    def _pol_factor(pol: str) -> int:
        return 2 if str(pol).strip().lower() in ("vh", "cross", "dual") else 1

    ant = y.get("antenna", {})
    tx_ant = ant.get("tx", {})
    rx_ant = ant.get("rx", {})
    if "pattern"      in tx_ant: cfg.tx_pattern      = str(tx_ant["pattern"])
    if "polarization" in tx_ant: cfg.tx_polarization = str(tx_ant["polarization"])
    if "num_rows"     in tx_ant: cfg.num_tx_rows     = int(tx_ant["num_rows"])
    if "num_cols"     in tx_ant: cfg.num_tx_cols     = int(tx_ant["num_cols"])
    if "pattern"      in rx_ant: cfg.rx_pattern      = str(rx_ant["pattern"])
    if "polarization" in rx_ant: cfg.rx_polarization = str(rx_ant["polarization"])
    if "num_rows"     in rx_ant: cfg.num_rx_rows     = int(rx_ant["num_rows"])
    if "num_cols"     in rx_ant: cfg.num_rx_cols     = int(rx_ant["num_cols"])
    # 실제 안테나 포트 수 (CIR / 공분산 차원과 일치)
    cfg.num_tx_ant = cfg.num_tx_rows * cfg.num_tx_cols * _pol_factor(cfg.tx_polarization)
    cfg.num_rx_ant = cfg.num_rx_rows * cfg.num_rx_cols * _pol_factor(cfg.rx_polarization)

    # comm
    c = y.get("comm", {})
    if "frequency"  in c: cfg.frequency  = float(c["frequency"])

    # postprocess
    pp = y.get("postprocess", {})
    if "threshold_watt" in pp: cfg.threshold_watt = float(pp["threshold_watt"])

    # analysis
    a = y.get("analysis", {})
    if "target_tx_index" in a: cfg.target_tx_index = int(a["target_tx_index"])
    if "target_rx_index" in a: cfg.target_rx_index = int(a["target_rx_index"])

    # output
    o = y.get("output", {})
    if "dir"                   in o: cfg.output_dir            = str(o["dir"])
    if "visualize_ray_tracing" in o: cfg.visualize_ray_tracing = bool(o["visualize_ray_tracing"])

    # visualization
    v = y.get("visualization", {})
    if "max_rays_per_rx"  in v: cfg.viz.max_rays_per_rx  = int(v["max_rays_per_rx"])
    if "ray_width"        in v: cfg.viz.ray_width         = float(v["ray_width"])
    if "tx_radius"        in v: cfg.viz.tx_radius         = float(v["tx_radius"])
    if "rx_radius_target" in v: cfg.viz.rx_radius_target  = float(v["rx_radius_target"])
    if "rx_radius_other"  in v: cfg.viz.rx_radius_other   = float(v["rx_radius_other"])
    if "map_overlay"       in v: cfg.viz.map_overlay       = bool(v["map_overlay"])
    if "map_overlay_z_min" in v:
        _zv = v["map_overlay_z_min"]
        # "auto" / null 이면 자동 추정(None), 숫자면 수동 지정
        if _zv is None or (isinstance(_zv, str) and _zv.strip().lower() == "auto"):
            cfg.viz.map_overlay_z_min = None
        else:
            cfg.viz.map_overlay_z_min = float(_zv)
    if "map_overlay_alpha" in v: cfg.viz.map_overlay_alpha = float(v["map_overlay_alpha"])
    if "map_overlay_color" in v: cfg.viz.map_overlay_color = str(v["map_overlay_color"])

    return cfg



# ============================================================
# RawRayData: PathSolver 연산 직후의 원시 경로 데이터
# ============================================================
@dataclass
class RawRayData:
    """
    RT_Agent가 PathSolver 연산 후 반환하는 원시 데이터.

    Fields:
      a     : 채널 계수 (complex, shape depends on Sionna version)
      tau   : 경로 지연 (s)
      phi_r : 수신 방위각 (rad)
      phi_t : 송신 방위각 (rad)
      los   : RX별 LoS(직시) 경로 존재 여부 (bool, shape (num_rx,))
              — paths.interactions가 모두 NONE(0)인 유효 경로가 있으면 True
    """
    a: np.ndarray = field(default_factory=lambda: np.array([]))
    tau: np.ndarray = field(default_factory=lambda: np.array([]))
    phi_r: np.ndarray = field(default_factory=lambda: np.array([]))
    phi_t: np.ndarray = field(default_factory=lambda: np.array([]))
    los: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))


# ============================================================
# SimulationResult: 후처리 결과 (Output 1 — Channel Agent용)
# ============================================================
@dataclass
class SimulationResult:
    """
    PostProcess_Agent가 반환하는 후처리 결과.
    Channel Agent로 전달되는 수학적 통신 데이터 (Output 1).

    Fields:
      tau            : 유효 경로 지연 (ns)
      power_dbm      : 유효 경로별 수신 전력 (dBm)
      aoa_azimuth    : 유효 경로별 수신 방위각 / AoA (degree)
      aod_azimuth    : 유효 경로별 송신 방위각 / AoD (degree)
      total_rsrp_dbm : 총 RSRP (dBm) — 모든 유효 경로 전력의 선형 합산
      R_TX           : TX 공간 공분산 행렬 (num_tx_ant × num_tx_ant, complex)
      R_RX           : RX 공간 공분산 행렬 (num_rx_ant × num_rx_ant, complex)
    """
    tau: np.ndarray = field(default_factory=lambda: np.array([]))
    power_dbm: np.ndarray = field(default_factory=lambda: np.array([]))
    aoa_azimuth: np.ndarray = field(default_factory=lambda: np.array([]))
    aod_azimuth: np.ndarray = field(default_factory=lambda: np.array([]))
    total_rsrp_dbm: float = -np.inf
    R_TX: np.ndarray = field(default_factory=lambda: np.array([]))
    R_RX: np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def is_dead_zone(self) -> bool:
        """유효 경로가 없는 Dead Zone 여부"""
        return len(self.tau) == 0
