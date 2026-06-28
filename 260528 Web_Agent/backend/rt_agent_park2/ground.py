"""
ground.py — 지면 레이캐스팅 기반 RX 격자 생성 / TX 지면 스냅
=============================================================
PARK_2 RT Agent 의 RX 배치 방식을 이식한 모듈.

원본 대응:
  - m2_config_agent.run()            → compute_ground_rx_grid()
      (PLY bbox → grid_n×grid_n 후보 → 하늘에서 ↓ 레이캐스팅 → 지면 hit만 유효)
  - RT_utils.get_adaptive_rx_positions() → ground_z_at()/_height 적용
      (각 (x,y)에서 지면 고도를 읽어 +rx_height)
  - TX 지면 스냅 (+2m, 조정 가능)    → snap_tx_to_ground()

실행 환경: Python 3.10 / open3d 0.19 / numpy 2.2 (컨테이너 venv-webagent)
서버: dclserver78 (H100). open3d RaycastingScene 사용 (numpy 2.x 호환 확인됨).

A안(표준 NPZ 유지) 정책:
  여기서 계산한 좌표는 P1A 에 "explicit" 좌표로 전달되어, 표준 Ray NPZ 스키마를
  그대로 유지한다. (P1A fork / metric 파이프라인 무수정)
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# open3d 는 모듈 로드 시점에 import 하지 않는다.
#   (백엔드 startup 타이밍/임포트 순서에 따라 일시적으로 실패하면 그 결과가
#    모듈 전역에 캐시되어 이후 영구히 "미설치"로 굳는 문제가 있었음.)
#   → 실제 사용 시점(GroundRaycaster.__init__)에 지연 import 한다.
def _import_o3d():
    import open3d as o3d  # noqa: PLC0415
    return o3d


def _normalize_plys(ply_path) -> List[Path]:
    """단일 경로(str/Path) 또는 경로 시퀀스를 Path 리스트로 정규화."""
    if isinstance(ply_path, (str, Path)):
        return [Path(ply_path)]
    return [Path(p) for p in ply_path]


# ---------------------------------------------------------------------------
# Raycaster (PLY 1개 이상을 하나의 장면으로 빌드 후 캐시)
# ---------------------------------------------------------------------------
class GroundRaycaster:
    """PLY 지형 메시(들)에 대한 하향(下向) 레이캐스팅 헬퍼.

    open3d RaycastingScene 을 사용한다. 하나 이상의 PLY 를 모두 같은 장면에 넣어
    (Geo-Radio Env. Twin 처럼 재질별로 여러 .ply 로 쪼개진 경우에도) 전체 지면을
    하나의 메시처럼 취급해 임의의 (x, y) 지면 고도 z 를 질의한다.
    """

    def __init__(self, ply_path: "str | Path | List[str | Path]"):
        try:
            o3d = _import_o3d()
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                f"open3d import 실패 — RX 지면 격자/TX 스냅에 필요합니다: {exc!r}"
            )
        paths = _normalize_plys(ply_path)
        self.ply_paths = [str(p) for p in paths]
        existing = [p for p in paths if Path(p).exists()]
        if not existing:
            raise FileNotFoundError(f"PLY 파일을 찾을 수 없습니다: {self.ply_paths}")
        self._o3d = o3d  # 다른 메서드에서 재사용 (모듈 전역 import 제거됨)

        self._scene = o3d.t.geometry.RaycastingScene()
        mins: list[np.ndarray] = []
        maxs: list[np.ndarray] = []
        total_tri = 0
        for p in existing:
            mesh_legacy = o3d.io.read_triangle_mesh(str(p))
            if len(mesh_legacy.triangles) == 0:
                continue  # 빈 메시는 건너뜀 (다른 PLY 로 계속)
            mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh_legacy)
            self._scene.add_triangles(mesh_t)
            total_tri += len(mesh_legacy.triangles)
            bbox = mesh_legacy.get_axis_aligned_bounding_box()
            mins.append(np.asarray(bbox.min_bound, dtype=float))
            maxs.append(np.asarray(bbox.max_bound, dtype=float))

        if total_tri == 0:
            raise ValueError(f"PLY 들에 삼각형이 없습니다(빈 메시): {self.ply_paths}")

        self.bbox_min = np.min(np.stack(mins, axis=0), axis=0)  # (3,)
        self.bbox_max = np.max(np.stack(maxs, axis=0), axis=0)  # (3,)

    # -- 단일 (x,y) 지면 고도 -------------------------------------------------
    def ground_z(self, x: float, y: float, cast_z: Optional[float] = None) -> Optional[float]:
        """(x, y) 에서 지면 고도 z 를 반환. 맵 바깥(hit 없음)이면 None.

        cast_z: 레이를 쏘는 시작 높이. None 이면 bbox 최고점 + 10m.
        """
        if cast_z is None:
            cast_z = float(self.bbox_max[2]) + 10.0
        o3d = self._o3d
        ray = o3d.core.Tensor(
            [[float(x), float(y), float(cast_z), 0.0, 0.0, -1.0]],
            dtype=o3d.core.Dtype.Float32,
        )
        hit = self._scene.cast_rays(ray)["t_hit"].item()
        if np.isinf(hit):
            return None
        return float(cast_z - hit)

    # -- 배치 지면 고도 (N개 (x,y) 한 번에) ----------------------------------
    def _ground_z_batch(self, xy: np.ndarray, cast_z: float) -> np.ndarray:
        """xy: (N,2) → ground_z (N,). hit 없으면 NaN. 단일 cast_rays 호출(벡터화)."""
        o3d = self._o3d
        n = xy.shape[0]
        if n == 0:
            return np.empty((0,), dtype=np.float64)
        rays = np.zeros((n, 6), dtype=np.float32)
        rays[:, 0] = xy[:, 0]
        rays[:, 1] = xy[:, 1]
        rays[:, 2] = cast_z
        rays[:, 5] = -1.0  # 방향 (0,0,-1)
        ans = self._scene.cast_rays(o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32))
        t_hit = ans["t_hit"].numpy().reshape(-1)
        gz = cast_z - t_hit
        gz[np.isinf(t_hit)] = np.nan
        return gz

    # -- 격자 RX (지면 위 rx_height) -----------------------------------------
    def rx_grid(
        self,
        grid_n: int = 20,
        margin: float = 0.0,
        rx_height: float = 1.5,
        raycasting_z: Optional[float] = None,
        max_height: Optional[float] = None,
        region: Optional[Tuple[float, float, float, float]] = None,
        spacing: Optional[float] = None,
    ) -> List[List[float]]:
        """bbox 기반 후보 격자 → 지면 hit 위치만 (x, y, ground_z+rx_height).

        격자 생성 방식 (2가지):
          - 밀도(개수): spacing 이 None 이면 한 축당 grid_n 개로 균등분할 (linspace).
          - 간격(미터): spacing(>0) 지정 시 x·y 동일 간격(정사각)으로 arange 격자. (2026-06-24)
        max_height: 최종 RX z(=지면고도+rx_height) 가 이 값 이하인 것만 남김.
        region: (x_min,x_max,y_min,y_max) 지정 시 그 직사각형 안에서만 후보 생성/raycasting.
        """
        # 격자 extent: region 지정 시 그 사각형(단, bbox 로 클립), 아니면 전체 bbox
        if region is not None:
            rx0, rx1, ry0, ry1 = (float(v) for v in region)
            x_min, x_max = min(rx0, rx1), max(rx0, rx1)
            y_min, y_max = min(ry0, ry1), max(ry0, ry1)
            x_min = max(x_min, float(self.bbox_min[0])); x_max = min(x_max, float(self.bbox_max[0]))
            y_min = max(y_min, float(self.bbox_min[1])); y_max = min(y_max, float(self.bbox_max[1]))
            if x_max <= x_min or y_max <= y_min:  # 잘못된 영역이면 전체 bbox 폴백
                x_min, y_min = float(self.bbox_min[0]), float(self.bbox_min[1])
                x_max, y_max = float(self.bbox_max[0]), float(self.bbox_max[1])
        else:
            x_min, y_min = float(self.bbox_min[0]), float(self.bbox_min[1])
            x_max, y_max = float(self.bbox_max[0]), float(self.bbox_max[1])

        # 안전 캡: 실수로 큰 값(예: 500 → 25만 후보) 입력 시 폭주/멈춤 방지
        grid_n = max(2, min(200, int(grid_n)))

        if margin and margin > 0:
            xr, yr = (x_max - x_min), (y_max - y_min)
            x_min += xr * margin
            x_max -= xr * margin
            y_min += yr * margin
            y_max -= yr * margin

        cast_z = raycasting_z if raycasting_z is not None else float(self.bbox_max[2]) + 10.0

        if spacing is not None and float(spacing) > 0:
            # 간격(미터) 방식: x·y 동일 간격(정사각). 후보 개수 상한(축당 200, 총 40,000)으로 폭주 방지.
            step = float(spacing)
            MAX_PER_AXIS = 200
            span_x = max(x_max - x_min, 0.0)
            span_y = max(y_max - y_min, 0.0)
            min_step = max(span_x, span_y) / MAX_PER_AXIS
            if step < min_step:  # 간격이 너무 작아 후보 폭발 → 안전 간격으로 자동 보정
                step = min_step
            xs = np.arange(x_min, x_max + 1e-9, step)
            ys = np.arange(y_min, y_max + 1e-9, step)
            if xs.size == 0:
                xs = np.array([x_min])
            if ys.size == 0:
                ys = np.array([y_min])
        else:
            xs = np.linspace(x_min, x_max, int(grid_n))
            ys = np.linspace(y_min, y_max, int(grid_n))
        gx, gy = np.meshgrid(xs, ys)
        xy = np.column_stack([gx.reshape(-1), gy.reshape(-1)])

        gz = self._ground_z_batch(xy, cast_z)
        hit = ~np.isnan(gz)
        out = np.column_stack([xy[hit, 0], xy[hit, 1], gz[hit] + rx_height])
        if max_height is not None:
            out = out[out[:, 2] <= float(max_height)]
        return [[float(a), float(b), float(c)] for a, b, c in out]

    # -- (x,y) 리스트를 지면 높이 기준으로 필터 -------------------------------
    def filter_xy_by_height(
        self, xy_list, max_height: float, rx_height: float = 1.5,
        raycasting_z: Optional[float] = None,
    ) -> List[List[float]]:
        """주어진 (x,y) 중 지면고도+rx_height 가 max_height 이하인 것만 반환 (2D)."""
        xy = np.asarray(xy_list, dtype=np.float64).reshape(-1, 2)
        if xy.shape[0] == 0:
            return []
        cast_z = raycasting_z if raycasting_z is not None else float(self.bbox_max[2]) + 10.0
        gz = self._ground_z_batch(xy, cast_z)
        z = gz + rx_height
        # 맵 바깥(NaN) 은 보수적으로 유지 (지면 고도 모름) — 필요 시 제거 가능
        keep = np.isnan(z) | (z <= float(max_height))
        return [[float(a), float(b)] for a, b in xy[keep]]

    # -- 명시적 (x,y) 리스트의 3D 좌표 (지면 위 height) -----------------------
    def adaptive_positions(
        self, xy_list, rx_height: float = 1.5, raycasting_z: Optional[float] = None
    ) -> List[List[float]]:
        """주어진 (x, y) 리스트를 지면 고도 + rx_height 로 올린 3D 좌표로 변환.

        RT_utils.get_adaptive_rx_positions 와 동일 (맵 바깥은 z=rx_height 폴백).
        """
        cast_z = raycasting_z if raycasting_z is not None else float(self.bbox_max[2]) + 10.0
        xy = np.asarray(xy_list, dtype=np.float64).reshape(-1, 2)
        gz = self._ground_z_batch(xy, cast_z)
        gz = np.where(np.isnan(gz), 0.0, gz)
        out = np.column_stack([xy[:, 0], xy[:, 1], gz + rx_height])
        return [[float(a), float(b), float(c)] for a, b, c in out]


@functools.lru_cache(maxsize=8)
def _cached_raycaster(key: tuple) -> GroundRaycaster:
    """(path, mtime) 쌍들의 튜플을 키로 Raycaster 를 캐시. 파일이 바뀌면 재생성."""
    paths = [k[0] for k in key]
    return GroundRaycaster(paths)


def _get_raycaster(ply_path: "str | Path | List[str | Path]") -> GroundRaycaster:
    paths = _normalize_plys(ply_path)
    key = tuple(sorted(
        (str(p), (p.stat().st_mtime if p.exists() else 0.0)) for p in paths
    ))
    return _cached_raycaster(key)


# ---------------------------------------------------------------------------
# 공개 함수 (백엔드 API / pipeline 에서 호출)
# ---------------------------------------------------------------------------
def compute_ground_rx_grid(
    ply_path: str | Path,
    grid_n: int = 20,
    margin: float = 0.0,
    rx_height: float = 1.5,
    raycasting_z: Optional[float] = None,
    max_height: Optional[float] = None,
    region: Optional[Tuple[float, float, float, float]] = None,
    spacing: Optional[float] = None,
) -> List[List[float]]:
    """지면 격자 RX 좌표 [[x,y,z], ...] 생성 (지면 위 rx_height).

    spacing(>0) 지정 시 x·y 동일 간격(정사각, 미터) 격자, 아니면 grid_n 균등분할.
    max_height: 지정 시 최종 RX z 가 이 값 이하인 것만 (건물 옥상 RX 제거).
    region: (x_min,x_max,y_min,y_max) 지정 시 그 직사각형 안에서만 후보 생성/raycasting.
    """
    return _get_raycaster(ply_path).rx_grid(
        grid_n=grid_n, margin=margin, rx_height=rx_height,
        raycasting_z=raycasting_z, max_height=max_height, region=region,
        spacing=spacing,
    )


def filter_xy_by_ground_height(
    ply_path: str | Path,
    xy_list,
    max_height: float,
    rx_height: float = 1.5,
    raycasting_z: Optional[float] = None,
) -> List[List[float]]:
    """임의 RX (x,y) 중 지면고도+rx_height 가 max_height 이하인 것만 반환 (모든 배치 방식 공용)."""
    return _get_raycaster(ply_path).filter_xy_by_height(
        xy_list, max_height=max_height, rx_height=rx_height, raycasting_z=raycasting_z
    )


def ground_z_at(
    ply_path: str | Path, x: float, y: float, raycasting_z: Optional[float] = None
) -> Optional[float]:
    """(x, y) 의 지면 고도 z (맵 바깥이면 None)."""
    return _get_raycaster(ply_path).ground_z(x, y, cast_z=raycasting_z)


def snap_tx_to_ground(
    ply_path: str | Path,
    x: float,
    y: float,
    offset_m: float = 2.0,
    raycasting_z: Optional[float] = None,
) -> dict:
    """클릭한 (x, y) 를 지면 + offset_m 위치로 스냅한 '진짜 TX 위치' 계산.

    Returns
    -------
    dict: {
        "clicked": [x, y, <ground_z or None>],   # 클릭 원점(연한 마커용)
        "ground_z": float | None,                # 지면 고도
        "tx": [x, y, ground_z + offset_m],       # 진짜 TX 위치(빨간 마커용)
        "offset_m": offset_m,
        "in_bounds": bool,                        # 지면 hit 여부
    }
    맵 바깥(hit 없음)이면 ground_z=None, tx z 는 offset_m 로 폴백.
    """
    gz = ground_z_at(ply_path, x, y, raycasting_z=raycasting_z)
    in_bounds = gz is not None
    base = gz if in_bounds else 0.0
    return {
        "clicked": [float(x), float(y), (float(gz) if in_bounds else None)],
        "ground_z": (float(gz) if in_bounds else None),
        "tx": [float(x), float(y), float(base + offset_m)],
        "offset_m": float(offset_m),
        "in_bounds": bool(in_bounds),
    }
