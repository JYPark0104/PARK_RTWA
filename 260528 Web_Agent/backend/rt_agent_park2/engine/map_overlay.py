"""
map_overlay.py
==============
PLY 메쉬의 상공(Top-down) 뷰를 matplotlib 축 위에 배경으로 그려주는 유틸리티.

격자 기반 그림(rx_positions / hitmap / rsrp_heatmap)이 단순 xy 산점도라
"여기가 맵의 어디인지" 알기 어려운 문제를 해결하기 위해, 실제 Twin Map PLY의
삼각형들을 xy 평면으로 투영한 건물 외곽선을 배경 LineCollection으로 깔아준다.

핵심 함수:
  - load_ply_top_edges(ply_path, z_min, ...) → [E, 2, 2] xy 세그먼트
  - add_building_overlay(ax, ply_path, ...)   → ax에 외곽선 그리기

특징:
  - open3d 의존 없음 (numpy struct 파싱) → Scene_Agent 외 환경에서도 동작
  - 대형 PLY(수백만 정점) 대응: numpy 벡터화 파싱 + 엣지 결과 디스크 캐싱
"""

from __future__ import annotations

import hashlib
import os
from typing import List, Optional, Tuple

import numpy as np


DEFAULT_EDGE_COLOR = "#444444"   # 밝은 배경 위에서 잘 보이는 진회색

# PLY 타입 → (struct 코드, 바이트 수, numpy base dtype)
_PLY_TYPE = {
    "float": ("f", 4, "f4"), "float32": ("f", 4, "f4"),
    "double": ("d", 8, "f8"), "float64": ("d", 8, "f8"),
    "uchar": ("B", 1, "u1"), "uint8": ("B", 1, "u1"),
    "char": ("b", 1, "i1"), "int8": ("b", 1, "i1"),
    "ushort": ("H", 2, "u2"), "uint16": ("H", 2, "u2"),
    "short": ("h", 2, "i2"), "int16": ("h", 2, "i2"),
    "uint": ("I", 4, "u4"), "uint32": ("I", 4, "u4"),
    "int": ("i", 4, "i4"), "int32": ("i", 4, "i4"),
}


def _read_ply_vertices_triangles(ply_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """PLY를 파싱해 (vertices [V,3] float64, triangles [F,3] int64)를 반환한다.

    binary_little/big_endian 및 ascii 포맷을 지원한다. 정점은 numpy 구조화 dtype으로
    한 번에 읽고, 면은 모두 삼각형이라고 가정한 빠른 경로를 먼저 시도한 뒤
    실패하면 가변 길이 루프로 폴백한다.
    """
    import struct

    with open(ply_path, "rb") as f:
        header_lines = []
        while True:
            line = f.readline()
            header_lines.append(line)
            if line.strip() == b"end_header" or line == b"":
                break
        body = f.read()

    fmt = None
    n_vert = 0
    n_face = 0
    vprops: List[Tuple[str, str]] = []   # (type, name)
    fprop_count_type = "uchar"
    fprop_index_type = "uint"
    in_vert = False
    in_face = False

    for raw in header_lines:
        line = raw.decode("ascii", errors="replace").strip()
        if line.startswith("format"):
            fmt = line.split()[1]
        elif line.startswith("element vertex"):
            n_vert = int(line.split()[-1]); in_vert = True; in_face = False
        elif line.startswith("element face"):
            n_face = int(line.split()[-1]); in_vert = False; in_face = True
        elif line.startswith("property"):
            parts = line.split()
            if in_vert:
                vprops.append((parts[1], parts[2]))
            elif in_face and len(parts) >= 5 and parts[1] == "list":
                fprop_count_type = parts[2]
                fprop_index_type = parts[3]

    if fmt is None:
        raise RuntimeError(f"PLY format 파싱 실패: {ply_path}")

    # ── ASCII 포맷 ────────────────────────────────────────────
    if fmt == "ascii":
        text = body.decode("ascii", errors="replace").splitlines()
        V = np.asarray(
            [list(map(float, ln.split()[:3])) for ln in text[:n_vert]],
            dtype=np.float64,
        )
        triangles: List[List[int]] = []
        for ln in text[n_vert:n_vert + n_face]:
            parts = ln.split()
            if not parts:
                continue
            c = int(parts[0])
            idxs = [int(p) for p in parts[1:1 + c]]
            if c == 3:
                triangles.append(idxs)
            elif c > 3:
                for j in range(1, c - 1):
                    triangles.append([idxs[0], idxs[j], idxs[j + 1]])
        F = np.asarray(triangles, dtype=np.int64) if triangles else np.zeros((0, 3), np.int64)
        return V, F

    if not fmt.startswith("binary"):
        raise RuntimeError(f"지원하지 않는 PLY format: {fmt}")

    endian_np = "<" if "little" in fmt else ">"
    endian_st = "<" if "little" in fmt else ">"

    # ── 정점: numpy 구조화 dtype으로 일괄 읽기 ────────────────
    vfields = [(name, endian_np + _PLY_TYPE[t][2]) for (t, name) in vprops]
    vdtype = np.dtype(vfields)
    varr = np.frombuffer(body, dtype=vdtype, count=n_vert)
    V = np.stack([varr["x"], varr["y"], varr["z"]], axis=1).astype(np.float64)
    off = n_vert * vdtype.itemsize

    # ── 면: 전부 삼각형이라 가정한 빠른 경로 ──────────────────
    cbase = _PLY_TYPE[fprop_count_type][2]
    ibase = _PLY_TYPE[fprop_index_type][2]
    try:
        fdtype = np.dtype([("c", endian_np + cbase), ("v", endian_np + ibase, (3,))])
        if off + n_face * fdtype.itemsize <= len(body):
            farr = np.frombuffer(body, dtype=fdtype, count=n_face, offset=off)
            if n_face == 0 or np.all(farr["c"] == 3):
                return V, farr["v"].astype(np.int64)
    except Exception:
        pass

    # ── 폴백: 가변 길이 면 루프 ───────────────────────────────
    ct_struct = struct.Struct(endian_st + _PLY_TYPE[fprop_count_type][0])
    ct_size = _PLY_TYPE[fprop_count_type][1]
    idx_code = _PLY_TYPE[fprop_index_type][0]
    idx_size = _PLY_TYPE[fprop_index_type][1]
    triangles = []
    cur = off
    for _ in range(n_face):
        (count,) = ct_struct.unpack_from(body, cur)
        cur += ct_size
        idxs = struct.unpack_from(endian_st + idx_code * count, body, cur)
        cur += idx_size * count
        if count == 3:
            triangles.append(list(idxs))
        elif count > 3:
            for j in range(1, count - 1):
                triangles.append([idxs[0], idxs[j], idxs[j + 1]])
    F = np.asarray(triangles, dtype=np.int64) if triangles else np.zeros((0, 3), np.int64)
    return V, F


def _auto_ground_z(z: np.ndarray) -> float:
    """PLY z 분포에서 지면(바닥) 높이를 자동 추정해 '이 값 이상만 건물'로 쓸 임계값을 반환한다.

    좌표계(절대 높이)에 무관하게 동작하도록 최저점 기준 상대 오프셋을 쓴다:
      threshold = z_min + max(0.5m, 5% × (z_max - z_min))
    즉 전체 높이 범위의 하위 약 5%(또는 최소 0.5m)를 지면으로 보고 제거한다.
    평평한 메쉬(높이 차 거의 0)면 전부 유지한다.
    """
    zmin = float(np.min(z))
    zmax = float(np.max(z))
    rng = zmax - zmin
    if rng <= 1e-6:
        return zmin - 1.0  # 평면 → 전부 유지
    return zmin + max(0.5, 0.05 * rng)


def load_ply_top_edges(
    ply_path: str,
    z_min: Optional[float] = None,
    max_edges: int = 80000,
    cache_dir: Optional[str] = None,
) -> np.ndarray:
    """PLY에서 상공 뷰 건물 외곽선을 이루는 xy 세그먼트 목록을 추출한다.

    절차:
      1. 정점/삼각형 파싱
      2. (z_min=None이면) z 분포에서 지면 높이를 자동 추정
      3. 세 정점이 모두 z >= z_min 인 삼각형만 유지 (지면 평면 제거)
      4. 각 삼각형의 3개 엣지를 (x1,y1,x2,y2)로 방출 후 중복 제거
      5. max_edges 이하로 서브샘플링

    Args:
        z_min: 지면 제거 높이 임계값. None이면 PLY에서 자동 추정(권장).

    Returns:
        segments: [E, 2, 2] (matplotlib LineCollection 입력 형식)
    """
    cache_path = None
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        try:
            mtime = os.path.getmtime(ply_path)
        except OSError:
            mtime = 0
        z_key = "auto" if z_min is None else f"{z_min:.4f}"
        key = hashlib.sha256(
            f"{os.path.abspath(ply_path)}|z={z_key}|m={max_edges}|t={mtime}".encode()
        ).hexdigest()[:16]
        cache_path = os.path.join(cache_dir, f"ply_edges_{key}.npy")
        if os.path.isfile(cache_path):
            try:
                return np.load(cache_path)
            except Exception:
                pass

    V, F = _read_ply_vertices_triangles(ply_path)
    if F.shape[0] == 0:
        return np.zeros((0, 2, 2), dtype=np.float64)

    z = V[:, 2]
    if z_min is None:
        z_min = _auto_ground_z(z)
    mask = (z[F[:, 0]] >= z_min) & (z[F[:, 1]] >= z_min) & (z[F[:, 2]] >= z_min)
    Fk = F[mask]
    if Fk.shape[0] == 0:
        # 폴백: z_min 위에 닿는 삼각형이라도 사용
        mask = (z[F[:, 0]] >= z_min) | (z[F[:, 1]] >= z_min) | (z[F[:, 2]] >= z_min)
        Fk = F[mask]
        if Fk.shape[0] == 0:
            # 그래도 없으면 전체 메쉬 외곽 (z 필터 무시)
            Fk = F

    edges_pairs = np.concatenate([
        Fk[:, [0, 1]], Fk[:, [1, 2]], Fk[:, [2, 0]],
    ], axis=0)
    edges_pairs.sort(axis=1)
    edges_pairs = np.unique(edges_pairs, axis=0)

    if edges_pairs.shape[0] > max_edges:
        step = max(1, edges_pairs.shape[0] // max_edges)
        edges_pairs = edges_pairs[::step]

    p1 = V[edges_pairs[:, 0], :2]
    p2 = V[edges_pairs[:, 1], :2]
    segments = np.stack([p1, p2], axis=1).astype(np.float64)

    if cache_path is not None:
        try:
            np.save(cache_path, segments)
        except Exception:
            pass
    return segments


def add_building_overlay(
    ax,
    ply_path: str,
    z_min: Optional[float] = None,
    color: str = DEFAULT_EDGE_COLOR,
    linewidth: float = 0.4,
    alpha: float = 0.5,
    zorder: int = 1,
    cache_dir: Optional[str] = None,
    max_edges: int = 80000,
) -> int:
    """기존 matplotlib 축 위에 PLY 상공 뷰 외곽선을 그린다. 그린 세그먼트 수 반환."""
    import matplotlib.collections as mc

    segs = load_ply_top_edges(
        ply_path=ply_path, z_min=z_min, max_edges=max_edges, cache_dir=cache_dir,
    )
    if segs.shape[0] == 0:
        return 0
    lc = mc.LineCollection(segs, colors=color, linewidths=linewidth,
                           alpha=alpha, zorder=zorder)
    ax.add_collection(lc)
    return int(segs.shape[0])
