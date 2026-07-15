"""
mesh_merge.py
=============
씬의 수많은 PLY(수천 개)를 '재질별 비인덱스 삼각형 배열(float32)' 로 병합하여
3D 뷰어가 재질당 1개 요청으로 로드하도록 만든다.

문제: 기존 뷰어는 PLY 를 파일당 1 HTTP 요청으로 로드 → 1625개면 요청 폭주 →
      dev 프록시/백엔드가 못 버텨 타임아웃·실패 누적.
해결: 서버에서 재질별로 삼각형(positions only)을 병합해 `meshes_merged/<mat>.f32`(binary)로
      캐시하고, manifest 로 목록을 제공. 프론트는 재질 수(보통 2~3)만큼만 fetch.

포맷:
  - <mat>.f32 : little-endian float32, 길이 = n_tri*9 (삼각형당 3정점 × xyz).
  - manifest  : {"materials":[{"name","rel","n_tri","bytes"}], "n_tri_total", "built_at"}

작성: 2026-07-06
"""
from __future__ import annotations

import json
import struct
import time
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np


def _parse_ply_triangles(path: Path) -> np.ndarray | None:
    """binary_little_endian PLY → (n_tri,3,3) float32 삼각형 soup. 실패 시 None."""
    try:
        d = path.read_bytes()
        he = d.find(b"end_header")
        if he < 0:
            return None
        header = d[:he].decode("ascii", "ignore")
        body_off = d.find(b"\n", he) + 1
        if "binary_little_endian" not in header:
            return None  # ascii/big-endian 은 미지원(호출측에서 trimesh fallback)
        nv = nf = 0
        vprops: list[str] = []
        section = None
        for ln in header.splitlines():
            t = ln.split()
            if not t:
                continue
            if t[0] == "element":
                if t[1] == "vertex":
                    nv = int(t[2]); section = "v"
                elif t[1] == "face":
                    nf = int(t[2]); section = "f"
                else:
                    section = None
            elif t[0] == "property" and section == "v":
                vprops.append(t[-1])
        ncol = len(vprops)
        if ncol < 3 or nv <= 0:
            return None
        varr = np.frombuffer(d, dtype="<f4", count=nv * ncol, offset=body_off).reshape(nv, ncol)
        xyz = varr[:, :3]
        foff = body_off + nv * ncol * 4
        fbuf = d[foff:]
        if nf <= 0 or len(fbuf) == 0:
            return None
        # 빠른 경로: 모든 face 가 삼각형(uchar=3, record=13B) 이라고 가정하고 검증
        if len(fbuf) == nf * 13:
            rec = np.frombuffer(fbuf, dtype=np.uint8).reshape(nf, 13)
            if np.all(rec[:, 0] == 3):
                tri = rec[:, 1:].copy().view("<i4").reshape(nf, 3)
                return xyz[tri].astype("<f4", copy=False)
        # 일반 경로: 가변 다각형(팬 삼각분할)
        tris = []
        pos = 0
        mv = memoryview(fbuf)
        for _ in range(nf):
            if pos >= len(fbuf):
                break
            n = fbuf[pos]; pos += 1
            inds = struct.unpack_from("<%di" % n, mv, pos)
            pos += 4 * n
            for k in range(1, n - 1):
                tris.append((inds[0], inds[k], inds[k + 1]))
        if not tris:
            return None
        tri = np.asarray(tris, dtype=np.int64)
        return xyz[tri].astype("<f4", copy=False)
    except Exception:
        return None


def _ply_material_map(scene_xml: Path) -> dict[str, str]:
    """scene.xml → {ply_basename: material_name}. shape 의 filename ↔ ref(mat-*) 매핑."""
    out: dict[str, str] = {}
    try:
        root = ET.fromstring(scene_xml.read_text(encoding="utf-8", errors="ignore"))
    except ET.ParseError:
        return out
    for shape in root.findall("shape"):
        fn = None
        for s in shape.findall("string"):
            if s.get("name") == "filename":
                fn = s.get("value")
        ref = shape.find("ref")
        mat = ref.get("id", "")[4:] if (ref is not None and ref.get("id", "").startswith("mat-")) else "default"
        if fn:
            base = Path(fn.replace("\\", "/")).name
            out[base] = mat
    return out


def _needs_rebuild(merged_dir: Path, scene_xml: Path, meshes_dir: Path) -> bool:
    manifest = merged_dir / "manifest.json"
    if not manifest.exists():
        return True
    try:
        mt = manifest.stat().st_mtime
    except OSError:
        return True
    # 소스(scene.xml 또는 meshes 폴더)가 manifest 보다 새로우면 재빌드
    newest = scene_xml.stat().st_mtime
    try:
        newest = max(newest, meshes_dir.stat().st_mtime)
    except OSError:
        pass
    return newest > mt


def build_merged_geometry(scene_dir: str | Path, force: bool = False) -> dict:
    """재질별 병합 geometry 를 빌드(캐시)하고 manifest(dict) 반환."""
    scene_dir = Path(scene_dir)
    scene_xml = scene_dir / "scene.xml"
    meshes_dir = scene_dir / "meshes"
    merged_dir = scene_dir / "meshes_merged"
    manifest_path = merged_dir / "manifest.json"

    if not force and not _needs_rebuild(merged_dir, scene_xml, meshes_dir):
        try:
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            pass  # 캐시 손상 → 재빌드

    if not scene_xml.exists() or not meshes_dir.exists():
        raise FileNotFoundError("scene.xml 또는 meshes 폴더가 없습니다")

    mat_map = _ply_material_map(scene_xml)
    # 재질별 삼각형 누적
    by_mat: dict[str, list[np.ndarray]] = {}
    n_fail = 0
    for ply in sorted(meshes_dir.glob("*.ply")):
        mat = mat_map.get(ply.name, "default")
        tris = _parse_ply_triangles(ply)
        if tris is None:
            # 미지원 포맷 → trimesh fallback (드묾)
            try:
                import trimesh
                m = trimesh.load(str(ply), process=False)
                v = np.asarray(m.vertices, dtype="<f4")
                f = np.asarray(m.faces, dtype=np.int64)
                tris = v[f].astype("<f4", copy=False)
            except Exception:
                n_fail += 1
                continue
        by_mat.setdefault(mat, []).append(tris.reshape(-1, 9))

    merged_dir.mkdir(parents=True, exist_ok=True)
    # 기존 .f32 정리
    for old in merged_dir.glob("*.f32"):
        try: old.unlink()
        except OSError: pass

    materials = []
    n_tri_total = 0
    for mat, chunks in sorted(by_mat.items()):
        arr = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 9), "<f4")
        arr = np.ascontiguousarray(arr, dtype="<f4")
        rel = f"{mat}.f32"
        (merged_dir / rel).write_bytes(arr.tobytes())
        n_tri = int(arr.shape[0])
        n_tri_total += n_tri
        materials.append({"name": mat, "rel": rel, "n_tri": n_tri, "bytes": int(arr.nbytes)})

    manifest = {
        "materials": materials,
        "n_tri_total": n_tri_total,
        "n_fail": n_fail,
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    return manifest
