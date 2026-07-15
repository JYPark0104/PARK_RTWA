"""
facade_recover.py — 이미 완료된 RT 결과에서 facade RX 메타데이터 사후 복구
==========================================================================
facade(O2I) RX 를 배치해 RT 를 돌렸지만, 메타 사이드카(host 건물/재질/법선)가
저장되지 않은 세션을 위해, **저장된 RX 좌표**로부터 각 RX 의 host 벽면을 찾아
메타데이터를 복구한다. (RT/GPU 재실행 불필요 — 순수 기하 lookup)

원리:
  - facade RX 는 벽 바깥으로 epsilon 만큼 떨어진 3D 점.
  - 씬의 모든 건물 수직면과 최근접(nearest surface) 질의로 host 삼각형을 찾고,
    그 삼각형의 오브젝트/재질/face normal(수평 성분=바깥 법선)을 기록한다.

입력: 세션 디렉토리 (channel_data_*.npz 의 rx_positions + scene/scene.xml)
출력: {세션}/P1A_RT_Results/facade_rx_meta_recovered_<ts>.json / .npz
      (기존 결과 파일은 건드리지 않음)

실행 환경: Python 3.10 / trimesh 4.12 / numpy 2.x (컨테이너 .venv-webagent)
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Optional

import numpy as np
import trimesh

from .facade import _parse_scene_shapes  # scene.xml → (ply, material, object)


def _find_channel_npz(session_dir: Path) -> Optional[Path]:
    """rx_positions 를 담은 결과 NPZ 를 탐색 (channel_data 우선, 없으면 다른 npz)."""
    session_dir = Path(session_dir)
    cands = sorted(session_dir.rglob("channel_data_*.npz"))
    if cands:
        return cands[-1]
    # 폴백: rx_positions 키를 가진 아무 npz
    for p in sorted(session_dir.rglob("*.npz")):
        try:
            with np.load(p, allow_pickle=True) as d:
                if "rx_positions" in d.files:
                    return p
        except Exception:
            continue
    return None


def _build_combined_wall_mesh(shapes, max_normal_z: float = 0.5):
    """모든 건물 오브젝트의 수직면만 모아 하나의 trimesh 로 결합.

    Returns: (combined_mesh, face_object[list], face_material[list])
      face_object[i], face_material[i] 는 결합 메시의 i번째 face 의 출처.
    """
    all_v = []
    all_f = []
    face_object: list[str] = []
    face_material: list[str] = []
    v_offset = 0
    for sh in shapes:
        p = Path(sh["ply_path"])
        if not p.exists():
            continue
        try:
            m = trimesh.load(p, force="mesh", process=False)
        except Exception:
            continue
        if not isinstance(m, trimesh.Trimesh) or len(m.faces) == 0:
            continue
        nz = np.abs(m.face_normals[:, 2])
        keep = nz <= float(max_normal_z)   # 수직 벽만
        if not np.any(keep):
            continue
        faces = m.faces[keep]
        all_v.append(m.vertices)
        all_f.append(faces + v_offset)
        v_offset += len(m.vertices)
        face_object.extend([sh["object_name"]] * len(faces))
        face_material.extend([sh["material"]] * len(faces))

    if not all_f:
        raise ValueError("수직 벽면을 가진 건물 메시를 찾지 못했습니다.")
    verts = np.vstack(all_v)
    facs = np.vstack(all_f)
    combined = trimesh.Trimesh(vertices=verts, faces=facs, process=False)
    return combined, face_object, face_material


def recover_facade_metadata(
    session_dir: str | Path,
    scene_xml: Optional[str | Path] = None,
    max_normal_z: float = 0.5,
    max_dist_warn: float = 3.0,
) -> dict:
    """완료된 세션의 RX 좌표 → host 건물/재질/법선 메타 복구 + 사이드카 저장.

    Returns: {"json":..., "npz":..., "count":N, "unmatched":M, "materials":[...]}
    """
    session_dir = Path(session_dir)
    if scene_xml is None:
        scene_xml = session_dir / "scene" / "scene.xml"
    scene_xml = Path(scene_xml)
    if not scene_xml.exists():
        raise FileNotFoundError(f"scene.xml 없음: {scene_xml}")

    npz_path = _find_channel_npz(session_dir)
    if npz_path is None:
        raise FileNotFoundError("rx_positions 를 가진 결과 NPZ 를 찾지 못했습니다.")
    with np.load(npz_path, allow_pickle=True) as d:
        if "rx_positions" not in d.files:
            raise KeyError(f"{npz_path.name} 에 rx_positions 키가 없습니다.")
        rx = np.asarray(d["rx_positions"], dtype=np.float64).reshape(-1, 3)

    shapes = _parse_scene_shapes(scene_xml)
    combined, face_object, face_material = _build_combined_wall_mesh(shapes, max_normal_z)
    face_object = np.asarray(face_object, dtype=object)
    face_material = np.asarray(face_material, dtype=object)

    # 각 RX 에서 최근접 벽면 삼각형 찾기 (한 번의 벡터화 질의)
    pq = trimesh.proximity.ProximityQuery(combined)
    _closest, dist, tri_id = pq.on_surface(rx)
    tri_id = np.asarray(tri_id, dtype=np.int64)
    dist = np.asarray(dist, dtype=np.float64)

    normals = combined.face_normals[tri_id]                 # (N,3)
    # 바깥 법선: 수평 성분만 정규화
    h = normals.copy(); h[:, 2] = 0.0
    hn = np.linalg.norm(h, axis=1, keepdims=True)
    hn[hn < 1e-9] = 1.0
    h = h / hn

    host = face_object[tri_id]
    mat = face_material[tri_id]
    unmatched = int(np.sum(dist > max_dist_warn))

    meta = []
    for i in range(len(rx)):
        meta.append({
            "rx_index": int(i),
            "host_object": str(host[i]),
            "material": str(mat[i]),
            "normal": [float(h[i, 0]), float(h[i, 1]), 0.0],
            "z_layer": float(round(rx[i, 2], 3)),
            "wall_dist": float(round(dist[i], 4)),
        })

    ts = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    out_dir = session_dir / "P1A_RT_Results"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"facade_rx_meta_recovered_{ts}.json"
    npz_out = out_dir / f"facade_rx_meta_recovered_{ts}.npz"

    json_path.write_text(json.dumps({
        "source_npz": str(npz_path),
        "count": len(rx),
        "unmatched_over_%.1fm" % max_dist_warn: unmatched,
        "materials": sorted(set(map(str, mat.tolist()))),
        "rx": meta,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    np.savez_compressed(
        npz_out,
        positions=rx.astype(np.float32),
        normals=h.astype(np.float32),
        z_layer=rx[:, 2].astype(np.float32),
        host_object=host.astype(object),
        material=mat.astype(object),
        wall_dist=dist.astype(np.float32),
        rx_index=np.arange(len(rx), dtype=np.int32),
    )
    return {
        "json": str(json_path), "npz": str(npz_out),
        "count": len(rx), "unmatched": unmatched,
        "materials": sorted(set(map(str, mat.tolist()))),
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="완료된 세션의 facade RX 메타데이터 복구")
    ap.add_argument("session_dir")
    ap.add_argument("--scene_xml", default=None)
    args = ap.parse_args()
    r = recover_facade_metadata(args.session_dir, scene_xml=args.scene_xml)
    print(json.dumps(r, ensure_ascii=False, indent=2))
