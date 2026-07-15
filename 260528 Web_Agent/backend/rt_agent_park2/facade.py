"""
facade.py — 건물 외벽(수직면) RX 배치 모듈 (O2I penetration 관찰용)
==================================================================
RX 배치 방식 중 하나인 'facade'(=O2I, 건물벽면)를 구현한다.
`ground.py`(지면 격자 RX)와 대칭되는 위치의 모듈로, 모듈화를 유지한다.

아이디어(사용자 제안):
  "옆에서 케첩 소스통을 눌러 지면과 평행하게 쭉 뿌리듯", 일정 높이 z=k 평면과
  각 건물 메시의 '교선(cross-section contour)'을 따라 RX 를 배치한다.

핵심 규칙(검토 확정):
  1. RX 는 벽면 '위'가 아니라 바깥 법선 방향으로 epsilon(m) 만큼 띄운다.
     (표면 위 RX 는 self-occlusion 으로 RT 가 깨짐. O2I 는 '외벽 입사 실외 필드'가 필요)
  2. 바깥 방향은 그 세그먼트를 만든 삼각형의 face normal(수평 성분)로 판정.
  3. 지붕/지면 등 수평면은 제외 — 수평 법선 성분이 큰(=수직 벽) 면만 사용.
     (수평면은 face normal 의 수평 성분이 0 에 가까워 자동 배제됨)
  4. 높이 레이어: z_min, z_min+z_distance, z_min+2*z_distance, ... (<= z_max).
     (예: z_min=1, z_distance=2 → z=1,3,5,...)  글로벌 평면 방식(건물별 clamp 없음).
  5. 밀도: 컨투어를 따라 spacing(m) 간격으로 샘플.
  6. 각 RX 에 host 건물 오브젝트명 / 재질 / 벽 바깥 법선 / 높이층 을 메타로 기록
     → 이후 penetration(입사각·재질 의존) 계산에 사용.
  * 재질 무관: 모든 건물 오브젝트의 수직면을 대상으로 한다(재질 필터 없음).

실행 환경: Python 3.10 / trimesh 4.12 / numpy 2.x (컨테이너 .venv-webagent)
서버: dclserver78 (H100).

A안(표준 NPZ 유지) 정책:
  여기서 계산한 좌표는 P1A 에 "points"(explicit 쌍좌표)로 전달되어 표준 Ray NPZ
  스키마를 그대로 유지한다. 메타데이터는 별도 사이드카(json/npz)로 저장한다.
"""

from __future__ import annotations

import unicodedata as _ud
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import trimesh
from trimesh.intersections import mesh_plane


# ---------------------------------------------------------------------------
# 메시 로드 캐시 (같은 세션에서 파라미터만 바뀔 때 PLY 재로드 방지 → 카운트/미리보기 가속)
#   key = (절대경로, mtime_ns) → trimesh.Trimesh. mtime 이 바뀌면 자동 무효화.
# ---------------------------------------------------------------------------
_MESH_CACHE: dict = {}
_MESH_CACHE_MAX = 4096


def _load_mesh_cached(ply_path: Path):
    """PLY 를 로드하되 (경로, mtime) 기준으로 캐시. 실패 시 None."""
    try:
        p = Path(ply_path)
        key = (str(p.resolve()), p.stat().st_mtime_ns)
    except Exception:
        key = None
    if key is not None and key in _MESH_CACHE:
        return _MESH_CACHE[key]
    try:
        mesh = trimesh.load(ply_path, force="mesh", process=False)
    except Exception:
        return None
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        return None
    if key is not None:
        if len(_MESH_CACHE) >= _MESH_CACHE_MAX:
            _MESH_CACHE.clear()  # 단순 방어(세션 씬 수 대비 충분히 큼)
        _MESH_CACHE[key] = mesh
    return mesh


def _xy_overlaps(mesh_bounds: np.ndarray,
                 bounds: Optional[Tuple[float, float, float, float]]) -> bool:
    """메시 XY bbox 가 선택 사각형(x_min,x_max,y_min,y_max)과 겹치는지. bounds=None → 항상 True."""
    if bounds is None:
        return True
    x_min, x_max, y_min, y_max = bounds
    mx0, my0 = float(mesh_bounds[0][0]), float(mesh_bounds[0][1])
    mx1, my1 = float(mesh_bounds[1][0]), float(mesh_bounds[1][1])
    return not (mx1 < x_min or mx0 > x_max or my1 < y_min or my0 > y_max)


# ---------------------------------------------------------------------------
# scene.xml 파싱 (shape → ply, material, object 이름)
# ---------------------------------------------------------------------------
_MATERIAL_SUFFIXES = ("-itu_concrete", "-itu_ceiling_board", "-itu_glass")


def _nfc(s: str) -> str:
    return _ud.normalize("NFC", s)


def _object_name_from_ply(ply_basename: str) -> str:
    """PLY basename 에서 재질 접미사/확장자를 떼어 건물 오브젝트 이름 추정."""
    stem = ply_basename[:-4] if ply_basename.lower().endswith(".ply") else ply_basename
    for suf in _MATERIAL_SUFFIXES:
        if stem.endswith(suf):
            return stem[: -len(suf)]
    return stem


def _parse_scene_shapes(scene_xml: Path) -> List[dict]:
    """scene.xml → [{ply_path, material, object_name}, ...].

    각 <shape> 의 <string name="filename"> 와 <ref id="mat-itu_*"> 를 읽는다.
    (mat- 접두어를 제거해 'itu_concrete' 형태의 재질명으로 저장)
    """
    scene_xml = Path(scene_xml)
    base = scene_xml.parent
    tree = ET.parse(scene_xml)
    root = tree.getroot()

    # meshes 실제 파일 색인 (NFC basename → 경로) : 한글 NFD/NFC 불일치 방지
    ply_index: dict[str, Path] = {}
    for hit in base.rglob("*.ply"):
        if hit.is_file():
            ply_index.setdefault(_nfc(hit.name), hit)

    shapes: List[dict] = []
    for shape in root.findall(".//shape"):
        fn = None
        for s in shape.findall("string"):
            if s.get("name") == "filename":
                fn = s.get("value")
                break
        if not fn:
            continue
        mat = None
        ref = shape.find("ref")
        if ref is not None and ref.get("id", "").startswith("mat-"):
            mat = ref.get("id")[4:]  # "mat-itu_concrete" → "itu_concrete"
        basename = Path(fn.replace("\\", "/")).name
        ply_path = base / fn
        if not ply_path.exists():
            ply_path = ply_index.get(_nfc(basename), ply_path)
        shapes.append({
            "ply_path": ply_path,
            "material": mat or "unknown",
            "object_name": _object_name_from_ply(basename),
        })
    return shapes


# ---------------------------------------------------------------------------
# 높이 레이어 / 세그먼트 샘플링
# ---------------------------------------------------------------------------
def _z_layers(z_min: float, z_max: float, z_distance: float) -> np.ndarray:
    """z=z_min, z_min+z_distance, ... (<= z_max). z_distance<=0 이면 z_min 단일."""
    if z_distance is None or float(z_distance) <= 0:
        return np.array([float(z_min)], dtype=float)
    if z_max < z_min:
        z_min, z_max = z_max, z_min
    n = int(np.floor((float(z_max) - float(z_min)) / float(z_distance) + 1e-9)) + 1
    n = max(1, n)
    return float(z_min) + float(z_distance) * np.arange(n, dtype=float)


def _sample_segment(p1: np.ndarray, p2: np.ndarray, spacing: float) -> np.ndarray:
    """세그먼트 [p1,p2] 를 spacing 간격 중점 샘플. (n,3) 반환. 최소 1점."""
    seg = p2 - p1
    length = float(np.linalg.norm(seg))
    n = max(1, int(np.floor(length / max(spacing, 1e-6))))
    ts = (np.arange(n) + 0.5) / n           # 중점 샘플 → 인접 세그먼트 코너 중복 완화
    return p1[None, :] + ts[:, None] * seg[None, :]


# ---------------------------------------------------------------------------
# 핵심: 단일 메시의 한 높이층 슬라이스 → RX 후보 + 메타
# ---------------------------------------------------------------------------
def _slice_mesh_layer(
    mesh: trimesh.Trimesh,
    z: float,
    spacing: float,
    epsilon: float,
    max_normal_z: float,
    bounds: Optional[Tuple[float, float, float, float]] = None,
) -> Tuple[List[List[float]], List[np.ndarray]]:
    """z 평면과 mesh 의 교선을 따라 (수직면만) RX 점 + 바깥 수평법선 리스트 생성.

    bounds=(x_min,x_max,y_min,y_max) 가 주어지면, ε 이격 '전'의 컨투어 점 (x,y) 가
    사각형 안인 샘플만 유지한다(= bounding region 으로 딱 클립). None 이면 전체.
    """
    try:
        res = mesh_plane(
            mesh, plane_normal=[0.0, 0.0, 1.0], plane_origin=[0.0, 0.0, float(z)],
            return_faces=True,
        )
    except Exception:
        return [], []
    if isinstance(res, tuple):
        lines, faces = res
    else:  # return_faces 미동작 방어
        lines, faces = res, None
    if lines is None or len(lines) == 0:
        return [], []

    face_normals = mesh.face_normals
    pts: List[List[float]] = []
    normals: List[np.ndarray] = []
    for i, seg in enumerate(lines):
        p1 = np.asarray(seg[0], dtype=float)
        p2 = np.asarray(seg[1], dtype=float)
        # 바깥 법선(수평 성분): 세그먼트 원본 삼각형의 face normal 사용
        if faces is not None and i < len(faces):
            nrm = np.asarray(face_normals[int(faces[i])], dtype=float)
        else:
            nrm = np.array([0.0, 0.0, 0.0])
        # 수직 벽 필터: |nz| 가 크면(수평면=지붕/지면) 제외
        if abs(float(nrm[2])) > float(max_normal_z):
            continue
        h = np.array([nrm[0], nrm[1], 0.0], dtype=float)
        hn = float(np.linalg.norm(h))
        if hn < 1e-6:
            continue  # 수평 성분 없음 → 벽 아님
        h = h / hn
        for sp in _sample_segment(p1, p2, spacing):
            # 경계 클립: ε 적용 '전'(벽면 컨투어 점) 기준으로 사각형 안만 유지
            if bounds is not None and not (
                bounds[0] <= float(sp[0]) <= bounds[1]
                and bounds[2] <= float(sp[1]) <= bounds[3]
            ):
                continue
            out = sp + epsilon * h        # 바깥으로 epsilon 이격
            pts.append([float(out[0]), float(out[1]), float(z)])
            normals.append(h.copy())
    return pts, normals


# ---------------------------------------------------------------------------
# 공개 함수
# ---------------------------------------------------------------------------
def compute_facade_rx(
    scene_xml: str | Path,
    z_min: float = 1.0,
    z_max: float = 30.0,
    z_distance: float = 3.0,
    spacing: float = 5.0,
    epsilon: float = 0.3,
    max_normal_z: float = 0.5,
    max_points: int = 200_000,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
) -> dict:
    """건물 외벽(수직면) RX 배치를 계산한다.

    Parameters
    ----------
    scene_xml : Mitsuba scene.xml 경로 (shape/재질 참조 포함)
    z_min, z_max, z_distance : 높이 레이어 (z_min, z_min+z_distance, ... <= z_max)
    spacing  : 컨투어를 따라 RX 간격 [m]
    epsilon  : 벽 바깥 법선 방향 이격 거리 [m] (표면 self-occlusion 방지)
    max_normal_z : 이 값보다 |face normal_z| 가 크면 수평면으로 보고 제외 (지붕/지면 배제)
    max_points : 안전 상한 (초과 시 균등 서브샘플)

    Returns
    -------
    dict: {
      "points": [[x,y,z], ...],                       # P1A 'points' 로 전달
      "meta":  [{rx_index, host_object, material,
                 normal:[nx,ny,nz], z_layer}, ...],    # penetration 용 메타([6])
      "z_layers": [...],
      "count": int,
      "params": {...},
    }
    """
    scene_xml = Path(scene_xml)
    shapes = _parse_scene_shapes(scene_xml)
    z_list = _z_layers(z_min, z_max, z_distance)

    # XY 경계(bounding region): 4개 모두 주어질 때만 활성. 하나라도 None 이면 전체(맵 전역).
    bounds: Optional[Tuple[float, float, float, float]] = None
    if None not in (x_min, x_max, y_min, y_max):
        bx0, bx1 = sorted((float(x_min), float(x_max)))
        by0, by1 = sorted((float(y_min), float(y_max)))
        bounds = (bx0, bx1, by0, by1)

    points: List[List[float]] = []
    meta: List[dict] = []

    for sh in shapes:
        ply_path = sh["ply_path"]
        if not Path(ply_path).exists():
            continue
        mesh = _load_mesh_cached(ply_path)
        if mesh is None:
            continue
        # 선택 사각형과 XY bbox 가 안 겹치는 덩어리는 통째로 스킵 (슬라이싱 비용 절감)
        if not _xy_overlaps(mesh.bounds, bounds):
            continue
        # 이 오브젝트가 z 범위에 전혀 없으면 스킵 (전역 평면이지만 빈 슬라이스 회피)
        zmin_mesh, zmax_mesh = float(mesh.bounds[0][2]), float(mesh.bounds[1][2])
        for z in z_list:
            if z < zmin_mesh - 1e-6 or z > zmax_mesh + 1e-6:
                continue
            pts, normals = _slice_mesh_layer(
                mesh, float(z), spacing=spacing, epsilon=epsilon,
                max_normal_z=max_normal_z, bounds=bounds,
            )
            for p, nrm in zip(pts, normals):
                points.append(p)
                meta.append({
                    "host_object": sh["object_name"],
                    "material": sh["material"],
                    "normal": [float(nrm[0]), float(nrm[1]), float(nrm[2])],
                    "z_layer": float(z),
                })

    # 실제 전체 개수(캡 적용 前). 'count' 는 항상 이 값을 보고한다(표시가 상한에 붙는 오해 방지).
    total_count = len(points)

    # 안전 상한: 반환/렌더용 좌표 배열만 균등 서브샘플. max_points=None 이면 캡 없음(=실제 잡).
    if max_points and total_count > max_points:
        idx = np.linspace(0, total_count - 1, max_points).astype(int)
        points = [points[i] for i in idx]
        meta = [meta[i] for i in idx]

    # rx_index 태깅 (P1A add_receivers 순서 = points 순서 = rx1..rxN)
    for i, m in enumerate(meta):
        m["rx_index"] = i

    return {
        "points": points,
        "meta": meta,
        "z_layers": [float(z) for z in z_list],
        "count": total_count,           # 실제 총 RX 개수(캡 무관)
        "rendered": len(points),        # 반환된 좌표 수(렌더/페이로드 상한 적용 후)
        "params": {
            "z_min": float(z_min), "z_max": float(z_max), "z_distance": float(z_distance),
            "spacing": float(spacing), "epsilon": float(epsilon),
            "max_normal_z": float(max_normal_z),
            "bounds": list(bounds) if bounds is not None else None,
        },
    }


def save_facade_metadata(out_dir: str | Path, result: dict, timestamp: str) -> dict:
    """facade RX 메타데이터를 사이드카(json + npz)로 저장. (덮어쓰기 금지: timestamp 사용)

    Returns: {"json": path, "npz": path}
    """
    import json

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"facade_rx_meta_{timestamp}.json"
    npz_path = out_dir / f"facade_rx_meta_{timestamp}.npz"

    payload = {
        "params": result["params"],
        "z_layers": result["z_layers"],
        "count": result["count"],
        "rx": result["meta"],
        "points": result["points"],
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    meta = result["meta"]
    if meta:
        np.savez_compressed(
            npz_path,
            positions=np.asarray(result["points"], dtype=np.float32),
            normals=np.asarray([m["normal"] for m in meta], dtype=np.float32),
            z_layer=np.asarray([m["z_layer"] for m in meta], dtype=np.float32),
            host_object=np.asarray([m["host_object"] for m in meta], dtype=object),
            material=np.asarray([m["material"] for m in meta], dtype=object),
            rx_index=np.asarray([m["rx_index"] for m in meta], dtype=np.int32),
        )
    return {"json": str(json_path), "npz": str(npz_path)}


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Facade(O2I wall) RX placement self-test")
    ap.add_argument("scene_xml", help="Mitsuba scene.xml 경로")
    ap.add_argument("--z_min", type=float, default=1.0)
    ap.add_argument("--z_max", type=float, default=30.0)
    ap.add_argument("--z_distance", type=float, default=3.0)
    ap.add_argument("--spacing", type=float, default=5.0)
    ap.add_argument("--epsilon", type=float, default=0.3)
    args = ap.parse_args()

    r = compute_facade_rx(
        args.scene_xml, z_min=args.z_min, z_max=args.z_max,
        z_distance=args.z_distance, spacing=args.spacing, epsilon=args.epsilon,
    )
    print(json.dumps({
        "count": r["count"],
        "z_layers": r["z_layers"],
        "materials": sorted({m["material"] for m in r["meta"]}),
        "n_objects": len(sorted({m["host_object"] for m in r["meta"]})),
        "sample": r["meta"][:3],
        "sample_points": r["points"][:3],
    }, ensure_ascii=False, indent=2))
