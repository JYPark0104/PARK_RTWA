"""
m7_export_agent.py
===============
Export 에이전트 — Output 2 (Blender/렌더링 툴용 3D 파일 export).

역할:
  - RT_scene_*.usda : TX + 모든 RX 구체 + 모든 RX의 Ray 경로 (USD 포맷, Python으로 직접 생성)
  - Map_Mesh.obj    : PLY → OBJ 변환 (Open3D 사용, 맵 메쉬만)
  - Map_Mesh_RT.obj : TX/RX 마커 + Ray 경로 (OBJ 포맷)

노트북 2-1.2DGS-RT_only2DGSplannr.ipynb의 Phase 5 코드를 기반으로 구현.
Sionna API(scene.export, paths.export)에 의존하지 않고
paths.vertices, paths.valid, paths.sources, paths.targets를 직접 읽어서
Python 문자열로 USDA를 생성 → Sionna 버전에 무관하게 동작.

USDA 구조:
  RayTracingScene/
    Mat_TX, Mat_RX_Target, Mat_RX_Other, Mat_Ray_Target, Mat_Ray_Other  (Materials)
    TX_Main                    (빨간 구체)
    RX{i}_Group/               (각 RX 그룹)
      RX_Marker                (구체: target=초록, other=파란)
      Ray_0000, Ray_0001, ...  (BasisCurves: target=노란, other=주황)
"""

import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.default_config import RT_Config


# ============================================================
# USDA 생성 헬퍼 함수들
# ============================================================

def _usda_material_block() -> str:
    """UsdPreviewSurface 기반 Material 5종 정의 블록"""
    def mat(name, r, g, b, emissive=False):
        emit = (f"\n            color3f inputs:emissiveColor = "
                f"({r*0.3:.3f}, {g*0.3:.3f}, {b*0.3:.3f})") if emissive else ""
        return f'''
    def Material "{name}" {{
        token outputs:surface.connect = </RayTracingScene/{name}/PBR.outputs:surface>
        def Shader "PBR" {{
            uniform token info:id = "UsdPreviewSurface"
            color3f inputs:diffuseColor = ({r}, {g}, {b}){emit}
            float inputs:roughness = 0.4
            float inputs:metallic  = 0.0
            token outputs:surface
        }}
    }}'''
    return (
        mat("Mat_TX",         1.0, 0.0, 0.0, emissive=True)  +
        mat("Mat_RX_Target",  0.0, 1.0, 0.0, emissive=True)  +
        mat("Mat_RX_Other",   0.0, 0.4, 1.0, emissive=False) +
        mat("Mat_Ray_Target", 1.0, 0.8, 0.0, emissive=True)  +
        mat("Mat_Ray_Other",  1.0, 0.4, 0.0, emissive=False) +
        mat("Mat_Ray_LOS",    0.1, 0.5, 1.0, emissive=True)   # LoS 레이 = 파랑
    )


def _power_to_width(power_w: float, p_min_db: float, p_max_db: float,
                    w_min: float = 0.1, w_max: float = 1.2) -> float:
    """경로 전력(W) → 선 굵기. dB 변환 후 [p_min_db, p_max_db] 를 [w_min, w_max] 로 선형 매핑."""
    if power_w <= 0:
        return w_min
    p_db = 10.0 * np.log10(power_w + 1e-30)
    if p_max_db <= p_min_db:
        return (w_min + w_max) * 0.5
    frac = (p_db - p_min_db) / (p_max_db - p_min_db)
    frac = max(0.0, min(1.0, frac))
    return w_min + frac * (w_max - w_min)


def _ray_power_db_range(per_tx_rays: list):
    """모든 TX·RX·ray 의 power(W) → dB 최소/최대 (굵기 정규화용)."""
    dbs = []
    for rays_by_rx in per_tx_rays:
        for rays in rays_by_rx.values():
            for r in rays:
                pw = r.get("power", 0.0) if isinstance(r, dict) else 0.0
                if pw > 0:
                    dbs.append(10.0 * np.log10(pw + 1e-30))
    if not dbs:
        return (-1.0, 0.0)
    return (float(min(dbs)), float(max(dbs)))


def _usda_sphere(name: str, position, radius: float, mat_name: str) -> str:
    """구체 오브젝트 (TX/RX 마커용)"""
    x, y, z = float(position[0]), float(position[1]), float(position[2])
    return f"""
    def Xform "{name}" {{
        double3 xformOp:translate = ({x}, {y}, {z})
        uniform token[] xformOpOrder = ["xformOp:translate"]
        def Sphere "shape" {{
            double radius = {radius}
            rel material:binding = </RayTracingScene/{mat_name}>
        }}
    }}
"""


def _build_usda(paths, tx_pos, rx_pos_3d_list: list,
                target_rx_index: int, map_title: str,
                max_rays_per_rx: int = 8,
                ray_width: float = 1.5,
                tx_radius: float = 2.0,
                rx_radius_target: float = 1.8,
                rx_radius_other: float = 1.2) -> tuple:
    """
    paths 객체에서 직접 데이터를 읽어 USDA 문자열을 생성한다.

    Sionna 1.2.1 shape:
      vertices : (max_depth, num_rx, num_tx, num_paths, 3)
      valid    : (num_rx, num_tx, num_paths)
      sources  : (3, num_tx)   → TX 위치
      targets  : (3, num_rx)   → RX 위치

    Returns:
        (usda_content: str, total_ray_count: int)
    """
    vertices_np = np.array(paths.vertices)  # (max_depth, num_rx, num_tx, num_paths, 3)
    valid_mask  = np.array(paths.valid)     # (num_rx, num_tx, num_paths)
    sources_np  = np.array(paths.sources)  # (3, num_tx)
    targets_np  = np.array(paths.targets)  # (3, num_rx)

    print(f"   [진단] vertices shape: {vertices_np.shape}")
    print(f"   [진단] valid_mask shape: {valid_mask.shape}")
    print(f"   [진단] valid_mask sum: {int(np.sum(valid_mask))} (전체 valid path 수)")

    num_rx    = len(rx_pos_3d_list)
    num_paths = valid_mask.shape[-1]   # 마지막 차원이 num_paths
    max_depth = vertices_np.shape[0]
    tx_idx    = 0  # TX는 1개

    # ── USDA 헤더 ─────────────────────────────────────────────
    usda = f'''#usda 1.0
(
    defaultPrim = "RayTracingScene"
    metersPerUnit = 1.0
    upAxis = "Y"
)

def Xform "RayTracingScene" {{
'''
    usda += _usda_material_block()
    usda += _usda_sphere("TX_Main", tx_pos, radius=tx_radius, mat_name="Mat_TX")

    # ── 각 RX 그룹 ────────────────────────────────────────────
    total_ray_count = 0
    rx_with_rays = 0

    for rx_idx in range(num_rx):
        is_target = (rx_idx == target_rx_index)
        ray_mat   = "Mat_Ray_Target" if is_target else "Mat_Ray_Other"
        rx_mat    = "Mat_RX_Target"  if is_target else "Mat_RX_Other"
        radius    = rx_radius_target if is_target else rx_radius_other

        # TX/RX 위치: sources[:, tx_idx], targets[:, rx_idx]
        tx_start = sources_np[:, tx_idx]   # (3,)
        rx_end   = targets_np[:, rx_idx]   # (3,)

        # 유효 경로 수집 — valid한 경로만 모아서 max_rays_per_rx개 저장
        rx_curves = []
        valid_paths = []

        for path_idx in range(num_paths):
            if not valid_mask[rx_idx, tx_idx, path_idx]:
                continue
            full_path = [tx_start.tolist()]
            for depth_idx in range(max_depth):
                pt = vertices_np[depth_idx, rx_idx, tx_idx, path_idx]
                if np.any(pt != 0):
                    full_path.append(pt.tolist())
            full_path.append(rx_end.tolist())
            if len(full_path) >= 2:
                valid_paths.append(full_path)

        for i, full_path in enumerate(valid_paths[:max_rays_per_rx]):
            rx_curves.append((f"Ray_{i:04d}", full_path))
        rx_ray_count = len(rx_curves)

        # RX 그룹 Xform
        usda += f'\n    def Xform "RX{rx_idx}_Group" {{\n'

        rx_pos_3d = rx_pos_3d_list[rx_idx]
        x, y, z = float(rx_pos_3d[0]), float(rx_pos_3d[1]), float(rx_pos_3d[2])
        usda += f'''        def Xform "RX_Marker" {{
            double3 xformOp:translate = ({x}, {y}, {z})
            uniform token[] xformOpOrder = ["xformOp:translate"]
            def Sphere "shape" {{
                double radius = {radius}
                rel material:binding = </RayTracingScene/{rx_mat}>
            }}
        }}
'''
        for curve_name, pts in rx_curves:
            pts_str    = ", ".join([f"({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f})" for p in pts])
            widths_str = ", ".join([str(ray_width)] * len(pts))  # config.yaml ray_width
            usda += f'''        def BasisCurves "{curve_name}" {{
            uniform token type = "linear"
            uniform token wrap = "nonperiodic"
            int[] curveVertexCounts = [{len(pts)}]
            point3f[] points = [{pts_str}]
            float[] widths = [{widths_str}]
            uniform token widthsInterpolation = "vertex"
            rel material:binding = </RayTracingScene/{ray_mat}>
        }}
'''
        usda += '    }\n'
        total_ray_count += rx_ray_count
        if rx_ray_count > 0:
            rx_with_rays += 1

    usda += "}\n"
    print(f"   진단: Ray 있는 RX {rx_with_rays}/{num_rx}개, 총 Ray {total_ray_count}개 (RX당 평균 {total_ray_count/max(rx_with_rays,1):.1f}개)")
    return usda, total_ray_count


# ============================================================
# Batch USDA 생성: 배치별 paths를 합쳐서 하나의 USDA 생성
# ============================================================

def _build_usda_batch(batch_paths_list, tx_pos, all_rx_pos_3d: list,
                      target_rx_index: int, map_title: str,
                      max_rays_per_rx: int = 20,
                      ray_width: float = 1.5,
                      tx_radius: float = 2.0,
                      rx_radius_target: float = 1.8,
                      rx_radius_other: float = 1.2) -> tuple:
    """
    배치별 paths 결과를 합쳐서 하나의 USDA를 생성한다.

    Args:
        batch_paths_list: [(paths, batch_rx_3d, start_idx, end_idx), ...]
        all_rx_pos_3d: 전체 RX 3D 좌표 목록
        나머지: _build_usda와 동일

    Returns:
        (usda_content: str, total_ray_count: int)
    """
    num_rx = len(all_rx_pos_3d)

    # USDA 헤더
    usda = f'''#usda 1.0
(
    defaultPrim = "RayTracingScene"
    metersPerUnit = 1.0
    upAxis = "Y"
)

def Xform "RayTracingScene" {{
'''
    usda += _usda_material_block()
    usda += _usda_sphere("TX_Main", tx_pos, radius=tx_radius, mat_name="Mat_TX")

    total_ray_count = 0
    rx_with_rays = 0

    # 각 배치에서 해당 RX의 Ray 추출
    for batch_paths, batch_rx_3d, start_idx, end_idx in batch_paths_list:
        vertices_np = np.array(batch_paths.vertices)
        valid_mask  = np.array(batch_paths.valid)
        sources_np  = np.array(batch_paths.sources)
        targets_np  = np.array(batch_paths.targets)

        batch_num_rx = end_idx - start_idx
        num_paths    = valid_mask.shape[-1]
        max_depth    = vertices_np.shape[0]
        tx_idx       = 0

        for local_rx_idx in range(batch_num_rx):
            global_rx_idx = start_idx + local_rx_idx
            is_target = (global_rx_idx == target_rx_index)
            ray_mat   = "Mat_Ray_Target" if is_target else "Mat_Ray_Other"
            rx_mat    = "Mat_RX_Target"  if is_target else "Mat_RX_Other"
            radius    = rx_radius_target if is_target else rx_radius_other

            tx_start = sources_np[:, tx_idx]
            rx_end   = targets_np[:, local_rx_idx]

            # 유효 경로 수집
            rx_curves = []
            valid_paths = []

            for path_idx in range(num_paths):
                if not valid_mask[local_rx_idx, tx_idx, path_idx]:
                    continue
                full_path = [tx_start.tolist()]
                for depth_idx in range(max_depth):
                    pt = vertices_np[depth_idx, local_rx_idx, tx_idx, path_idx]
                    if np.any(pt != 0):
                        full_path.append(pt.tolist())
                full_path.append(rx_end.tolist())
                if len(full_path) >= 2:
                    valid_paths.append(full_path)

            for i, full_path in enumerate(valid_paths[:max_rays_per_rx]):
                rx_curves.append((f"Ray_{i:04d}", full_path))
            rx_ray_count = len(rx_curves)

            # RX 그룹 Xform
            usda += f'\n    def Xform "RX{global_rx_idx}_Group" {{\n'

            rx_pos = all_rx_pos_3d[global_rx_idx]
            x, y, z = float(rx_pos[0]), float(rx_pos[1]), float(rx_pos[2])
            usda += f'''        def Xform "RX_Marker" {{
            double3 xformOp:translate = ({x}, {y}, {z})
            uniform token[] xformOpOrder = ["xformOp:translate"]
            def Sphere "shape" {{
                double radius = {radius}
                rel material:binding = </RayTracingScene/{rx_mat}>
            }}
        }}
'''
            for curve_name, pts in rx_curves:
                pts_str    = ", ".join([f"({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f})" for p in pts])
                widths_str = ", ".join([str(ray_width)] * len(pts))
                usda += f'''        def BasisCurves "{curve_name}" {{
            uniform token type = "linear"
            uniform token wrap = "nonperiodic"
            int[] curveVertexCounts = [{len(pts)}]
            point3f[] points = [{pts_str}]
            float[] widths = [{widths_str}]
            uniform token widthsInterpolation = "vertex"
            rel material:binding = </RayTracingScene/{ray_mat}>
        }}
'''
            usda += '    }\n'
            total_ray_count += rx_ray_count
            if rx_ray_count > 0:
                rx_with_rays += 1

    usda += "}\n"
    print(f"   진단 (Batch): Ray 있는 RX {rx_with_rays}/{num_rx}개, 총 Ray {total_ray_count}개 (RX당 평균 {total_ray_count/max(rx_with_rays,1):.1f}개)")
    return usda, total_ray_count


# ============================================================
# OBJ export (TX/RX 마커 + Ray 경로)
# ============================================================

def _export_all_rx_to_obj(paths, tx_pos, rx_pos_3d_list: list,
                           target_rx_index: int, filename: str):
    """모든 RX의 Ray 경로를 하나의 .obj 파일로 export한다.

    Sionna 1.2.1 shape:
      vertices : (max_depth, num_rx, num_tx, num_paths, 3)
      valid    : (num_rx, num_tx, num_paths)
      sources  : (3, num_tx)
      targets  : (3, num_rx)
    """
    base_name = os.path.splitext(filename)[0]
    mtl_filename = base_name + ".mtl"
    mtl_name = os.path.basename(mtl_filename)

    with open(mtl_filename, 'w') as m:
        m.write("# Material definitions\n")
        m.write("newmtl MAT_TX\nKd 1.0 0.0 0.0\n\n")
        m.write("newmtl MAT_RX\nKd 0.0 1.0 0.0\n\n")
        m.write("newmtl MAT_RX_TARGET\nKd 1.0 0.5 0.0\n\n")
        m.write("newmtl MAT_PATH\nKd 0.0 0.6 1.0\n\n")

    try:
        vertices_np = np.array(paths.vertices)  # (max_depth, num_rx, num_tx, num_paths, 3)
        valid_mask  = np.array(paths.valid)     # (num_rx, num_tx, num_paths)
        sources_np  = np.array(paths.sources)  # (3, num_tx)
        targets_np  = np.array(paths.targets)  # (3, num_rx)
        num_rx    = len(rx_pos_3d_list)
        num_paths = valid_mask.shape[2]
        max_depth = vertices_np.shape[0]
        tx_idx    = 0
    except Exception as e:
        print(f"   Paths 데이터 추출 실패: {e}")
        return

    with open(filename, 'w') as f:
        f.write(f"mtllib {mtl_name}\n")
        v_idx = 1
        size  = 1.5

        # TX 마커 (빨간 박스)
        f.write("g TX_Marker\nusemtl MAT_TX\n")
        tx = tx_pos
        tx_cube = [
            [tx[0]-size, tx[1]-size, tx[2]-size], [tx[0]+size, tx[1]-size, tx[2]-size],
            [tx[0]+size, tx[1]+size, tx[2]-size], [tx[0]-size, tx[1]+size, tx[2]-size],
            [tx[0]-size, tx[1]-size, tx[2]+size], [tx[0]+size, tx[1]-size, tx[2]+size],
            [tx[0]+size, tx[1]+size, tx[2]+size], [tx[0]-size, tx[1]+size, tx[2]+size]
        ]
        for v in tx_cube:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in [[0,1,2,3],[4,5,6,7],[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7]]:
            f.write(f"f {v_idx+face[0]} {v_idx+face[1]} {v_idx+face[2]} {v_idx+face[3]}\n")
        v_idx += 8

        # 모든 RX 마커
        for rx_i, rx_pos in enumerate(rx_pos_3d_list):
            mat = "MAT_RX_TARGET" if rx_i == target_rx_index else "MAT_RX"
            f.write(f"g RX_Marker_{rx_i}\nusemtl {mat}\n")
            rx = rx_pos
            rx_cube = [
                [rx[0]-size, rx[1]-size, rx[2]-size], [rx[0]+size, rx[1]-size, rx[2]-size],
                [rx[0]+size, rx[1]+size, rx[2]-size], [rx[0]-size, rx[1]+size, rx[2]-size],
                [rx[0]-size, rx[1]-size, rx[2]+size], [rx[0]+size, rx[1]-size, rx[2]+size],
                [rx[0]+size, rx[1]+size, rx[2]+size], [rx[0]-size, rx[1]+size, rx[2]+size]
            ]
            for v in rx_cube:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for face in [[0,1,2,3],[4,5,6,7],[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7]]:
                f.write(f"f {v_idx+face[0]} {v_idx+face[1]} {v_idx+face[2]} {v_idx+face[3]}\n")
            v_idx += 8

        # 각 RX별 Ray 경로
        for rx_i in range(num_rx):
            tx_start = sources_np[:, tx_idx]   # (3,)
            rx_end   = targets_np[:, rx_i]     # (3,)

            f.write(f"g Ray_Paths_RX{rx_i}\nusemtl MAT_PATH\n")

            for path_idx in range(num_paths):
                if not valid_mask[rx_i, tx_idx, path_idx]:
                    continue

                path_points = [tx_start.tolist()]
                for depth_idx in range(max_depth):
                    pt = vertices_np[depth_idx, rx_i, tx_idx, path_idx]
                    if np.any(pt != 0):
                        if not np.allclose(pt, path_points[-1]):
                            path_points.append(pt.tolist())
                if not np.allclose(rx_end, path_points[-1]):
                    path_points.append(rx_end.tolist())

                if len(path_points) >= 2:
                    indices = []
                    for v in path_points:
                        f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
                        indices.append(v_idx)
                        v_idx += 1
                    f.write("l " + " ".join(map(str, indices)) + "\n")

    print(f"   ✅ Map_Mesh_RT.obj 저장 완료: {filename}")
    print(f"      TX: 1개, RX: {len(rx_pos_3d_list)}개 (target: RX_{target_rx_index})")


# ============================================================
# Multi-TX: Ray 폴리라인 추출 (Paths → 경량 좌표 목록)
# ============================================================

def _compute_path_power(paths, num_rx: int):
    """paths.cir() 로부터 (num_rx, num_paths) 경로별 안테나-평균 전력(W) 추정.

    실패 시 None 반환(굵기 균일 폴백). synthetic_array=True 형태를 우선 처리.
    """
    try:
        a, _tau = paths.cir()
        a_np = np.asarray(a)
        if a_np.ndim >= 6 and a_np.shape[0] == 2:
            a_c = a_np[0].astype(np.float64) + 1j * a_np[1].astype(np.float64)
            a_c = a_c[:, :, 0, :, :, 0]               # (num_rx, R, T, num_paths)
            return np.mean(np.abs(a_c) ** 2, axis=(1, 2))
        if np.iscomplexobj(a_np) and a_np.ndim >= 5:
            a_c = a_np[:, :, 0, :, :, 0] if a_np.ndim >= 6 else a_np
            return np.mean(np.abs(a_c) ** 2, axis=tuple(range(1, a_c.ndim - 1)))
    except Exception:
        return None
    return None


def extract_rays_single(paths, num_rx: int, max_rays_per_rx: int):
    """단일 TX paths에서 RX별 Ray (폴리라인 + power + LoS) 와 valid 경로 수를 추출한다.

    Returns:
        (rays_by_rx: dict[int -> list[dict]], ray_counts: np.ndarray(num_rx,))
        ray dict = {"pts": [[x,y,z],...], "power": float(W), "los": bool}
        LoS = 반사점이 없는 직접 경로(폴리라인 점 2개).
    """
    vertices_np = np.array(paths.vertices)   # (max_depth, num_rx, num_tx, num_paths, 3)
    valid_mask  = np.array(paths.valid)      # (num_rx, num_tx, num_paths)
    sources_np  = np.array(paths.sources)    # (3, num_tx)
    targets_np  = np.array(paths.targets)    # (3, num_rx)
    pow_path    = _compute_path_power(paths, num_rx)   # (num_rx, num_paths) | None

    num_paths = valid_mask.shape[-1]
    max_depth = vertices_np.shape[0]
    tx_idx = 0

    rays_by_rx = {}
    ray_counts = np.zeros(num_rx, dtype=int)

    for rx_idx in range(min(num_rx, valid_mask.shape[0])):
        tx_start = sources_np[:, tx_idx]
        rx_end   = targets_np[:, rx_idx]
        rays = []
        cnt = 0
        for p in range(num_paths):
            if not valid_mask[rx_idx, tx_idx, p]:
                continue
            cnt += 1
            if len(rays) >= max_rays_per_rx:
                continue
            full = [tx_start.tolist()]
            for d in range(max_depth):
                pt = vertices_np[d, rx_idx, tx_idx, p]
                if np.any(pt != 0):
                    full.append(pt.tolist())
            full.append(rx_end.tolist())
            if len(full) >= 2:
                pw = float(pow_path[rx_idx, p]) if pow_path is not None else 0.0
                rays.append({"pts": full, "power": pw, "los": len(full) == 2})
        ray_counts[rx_idx] = cnt
        if rays:
            rays_by_rx[rx_idx] = rays
    return rays_by_rx, ray_counts


def extract_rays_batch(batch_paths_list, num_rx: int, max_rays_per_rx: int):
    """배치 paths 목록에서 RX별 Ray (폴리라인 + power + LoS) 와 valid 경로 수를 추출.

    batch_paths_list 항목은 두 형태를 모두 허용:
      - (batch_paths, batch_rx_3d, start_idx, end_idx)   # 순차 배치 (기존)
      - (batch_paths, batch_rx_3d, global_indices)        # 무작위/임의 배치: 원래 RX 인덱스 리스트
    어느 경우든 g(=원래 RX 인덱스)에 결과를 기록하므로 출력은 항상 원래 인덱스 정렬을 따른다.
    """
    rays_by_rx = {}
    ray_counts = np.zeros(num_rx, dtype=int)

    for entry in batch_paths_list:
        if len(entry) == 4:
            batch_paths, batch_rx_3d, start_idx, end_idx = entry
            global_indices = list(range(start_idx, end_idx))
        else:
            batch_paths, batch_rx_3d, global_indices = entry
        vertices_np = np.array(batch_paths.vertices)
        valid_mask  = np.array(batch_paths.valid)
        sources_np  = np.array(batch_paths.sources)
        targets_np  = np.array(batch_paths.targets)
        num_paths = valid_mask.shape[-1]
        max_depth = vertices_np.shape[0]
        tx_idx = 0
        n_local = len(global_indices)
        pow_path = _compute_path_power(batch_paths, n_local)

        for local in range(n_local):
            g = int(global_indices[local])
            tx_start = sources_np[:, tx_idx]
            rx_end   = targets_np[:, local]
            rays = []
            cnt = 0
            for p in range(num_paths):
                if not valid_mask[local, tx_idx, p]:
                    continue
                cnt += 1
                if len(rays) >= max_rays_per_rx:
                    continue
                full = [tx_start.tolist()]
                for d in range(max_depth):
                    pt = vertices_np[d, local, tx_idx, p]
                    if np.any(pt != 0):
                        full.append(pt.tolist())
                full.append(rx_end.tolist())
                if len(full) >= 2:
                    pw = float(pow_path[local, p]) if pow_path is not None else 0.0
                    rays.append({"pts": full, "power": pw, "los": len(full) == 2})
            ray_counts[g] = cnt
            if rays:
                rays_by_rx[g] = rays
    return rays_by_rx, ray_counts


# ============================================================
# Multi-TX: 모든 TX의 Ray를 하나의 USDA / OBJ 로 합치기
# ============================================================

def _build_usda_multi_tx(per_tx_rays: list, tx_positions: list, rx_pos_3d: list,
                         target_tx_index: int, target_rx_index: int,
                         ray_width: float = 1.5, tx_radius: float = 2.0,
                         rx_radius_target: float = 1.8,
                         rx_radius_other: float = 1.2) -> tuple:
    """여러 TX의 Ray 폴리라인을 하나의 USDA 문자열로 합친다.

    구조:
      TX{t}_Main             (TX 구체, target TX는 빨강 강조)
      RX{i}_Marker           (RX 구체, 1회만 — target RX 초록 강조)
      TX{t}_Rays/            (각 TX의 Ray BasisCurves, target TX 노랑 / 그 외 주황)
    """
    usda = '''#usda 1.0
(
    defaultPrim = "RayTracingScene"
    metersPerUnit = 1.0
    upAxis = "Y"
)

def Xform "RayTracingScene" {
'''
    usda += _usda_material_block()

    # ── TX 구체들 ──
    for t, tp in enumerate(tx_positions):
        usda += _usda_sphere(f"TX{t}_Main", tp, radius=tx_radius, mat_name="Mat_TX")

    # ── RX 마커 (1회만) ──
    for rx_idx, rx_pos in enumerate(rx_pos_3d):
        is_target_rx = (rx_idx == target_rx_index)
        rx_mat = "Mat_RX_Target" if is_target_rx else "Mat_RX_Other"
        radius = rx_radius_target if is_target_rx else rx_radius_other
        usda += _usda_sphere(f"RX{rx_idx}_Marker", rx_pos, radius=radius, mat_name=rx_mat)

    # ── TX별 Ray ── (LoS=파랑 Mat_Ray_LOS / NLoS=타겟TX 노랑·그외 주황, 굵기는 power 비례)
    p_min_db, p_max_db = _ray_power_db_range(per_tx_rays)
    total_ray_count = 0
    for t, rays_by_rx in enumerate(per_tx_rays):
        nlos_mat = "Mat_Ray_Target" if t == target_tx_index else "Mat_Ray_Other"
        usda += f'\n    def Xform "TX{t}_Rays" {{\n'
        for rx_idx, rays in rays_by_rx.items():
            for i, ray in enumerate(rays):
                pts = ray["pts"] if isinstance(ray, dict) else ray
                is_los = bool(ray.get("los", len(pts) == 2)) if isinstance(ray, dict) else (len(pts) == 2)
                power = ray.get("power", 0.0) if isinstance(ray, dict) else 0.0
                ray_mat = "Mat_Ray_LOS" if is_los else nlos_mat
                w = _power_to_width(power, p_min_db, p_max_db)
                pts_str    = ", ".join([f"({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f})" for p in pts])
                widths_str = ", ".join([f"{w:.4f}"] * len(pts))
                usda += f'''        def BasisCurves "R{rx_idx}_{i:04d}" {{
            uniform token type = "linear"
            uniform token wrap = "nonperiodic"
            int[] curveVertexCounts = [{len(pts)}]
            point3f[] points = [{pts_str}]
            float[] widths = [{widths_str}]
            uniform token widthsInterpolation = "vertex"
            rel material:binding = </RayTracingScene/{ray_mat}>
        }}
'''
                total_ray_count += 1
        usda += '    }\n'

    usda += "}\n"
    print(f"   진단 (Multi-TX): TX {len(tx_positions)}대, RX {len(rx_pos_3d)}개, "
          f"총 Ray {total_ray_count}개")
    return usda, total_ray_count


def _export_multi_tx_obj(per_tx_rays: list, tx_positions: list, rx_pos_3d: list,
                         target_rx_index: int, filename: str):
    """여러 TX의 Ray 폴리라인 + TX/RX 마커를 하나의 .obj 로 export한다."""
    base_name = os.path.splitext(filename)[0]
    mtl_filename = base_name + ".mtl"
    mtl_name = os.path.basename(mtl_filename)

    with open(mtl_filename, 'w') as m:
        m.write("# Material definitions\n")
        m.write("newmtl MAT_TX\nKd 1.0 0.0 0.0\n\n")
        m.write("newmtl MAT_RX\nKd 0.0 1.0 0.0\n\n")
        m.write("newmtl MAT_RX_TARGET\nKd 1.0 0.5 0.0\n\n")
        m.write("newmtl MAT_PATH\nKd 0.0 0.6 1.0\n\n")

    size = 1.5

    def _cube(c):
        return [
            [c[0]-size, c[1]-size, c[2]-size], [c[0]+size, c[1]-size, c[2]-size],
            [c[0]+size, c[1]+size, c[2]-size], [c[0]-size, c[1]+size, c[2]-size],
            [c[0]-size, c[1]-size, c[2]+size], [c[0]+size, c[1]-size, c[2]+size],
            [c[0]+size, c[1]+size, c[2]+size], [c[0]-size, c[1]+size, c[2]+size],
        ]

    faces = [[0,1,2,3],[4,5,6,7],[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7]]

    with open(filename, 'w') as f:
        f.write(f"mtllib {mtl_name}\n")
        v_idx = 1

        # TX 마커들
        for t, tp in enumerate(tx_positions):
            f.write(f"g TX{t}_Marker\nusemtl MAT_TX\n")
            for v in _cube(tp):
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for face in faces:
                f.write(f"f {v_idx+face[0]} {v_idx+face[1]} {v_idx+face[2]} {v_idx+face[3]}\n")
            v_idx += 8

        # RX 마커들 (1회)
        for rx_i, rx_pos in enumerate(rx_pos_3d):
            mat = "MAT_RX_TARGET" if rx_i == target_rx_index else "MAT_RX"
            f.write(f"g RX_Marker_{rx_i}\nusemtl {mat}\n")
            for v in _cube(rx_pos):
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for face in faces:
                f.write(f"f {v_idx+face[0]} {v_idx+face[1]} {v_idx+face[2]} {v_idx+face[3]}\n")
            v_idx += 8

        # TX별 Ray 경로
        for t, rays_by_rx in enumerate(per_tx_rays):
            f.write(f"g TX{t}_Ray_Paths\nusemtl MAT_PATH\n")
            for rx_idx, rays in rays_by_rx.items():
                for ray in rays:
                    pts = ray["pts"] if isinstance(ray, dict) else ray
                    indices = []
                    for v in pts:
                        f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
                        indices.append(v_idx)
                        v_idx += 1
                    f.write("l " + " ".join(map(str, indices)) + "\n")

    print(f"   ✅ Map_Mesh_RT.obj 저장 완료: {filename}")
    print(f"      TX: {len(tx_positions)}대, RX: {len(rx_pos_3d)}개 (target RX_{target_rx_index})")


def run_multi(per_tx_rays: list, config: RT_Config, rx_pos_3d: list,
              tx_positions: list = None) -> list:
    """Multi-TX export: 모든 TX의 Ray를 합친 USDA/OBJ + 맵 메쉬를 생성한다.

    Args:
        per_tx_rays  : list[num_tx] of rays_by_rx dict (extract_rays_* 결과)
        config       : RT_Config
        rx_pos_3d    : 3D RX 좌표 목록
        tx_positions : [[x,y,z], ...] (None이면 config.tx_positions)
    """
    import open3d as o3d

    print("=" * 60)
    print("📦 [Export_Agent] Multi-TX 3D 파일 Export 시작 (Output 2)")
    print("=" * 60)

    start = time.time()
    os.makedirs(config.output_dir_viz, exist_ok=True)
    exported_files = []
    if tx_positions is None:
        tx_positions = config.tx_positions
    target_tx = min(config.target_tx_index, len(tx_positions) - 1)

    safe_title = "".join(c if c.isalnum() or c in "-_" else "_" for c in config.map_title)

    # ── 1. 모든 TX Ray를 합친 USDA ─────────────────────────────
    try:
        print(f"\n📁 USDA export 중 (TX {len(tx_positions)}대 + {len(rx_pos_3d)}개 RX + Ray)...")
        usda_path = os.path.join(config.output_dir_viz, f"RT_scene_{safe_title}.usda")
        usda_content, total_rays = _build_usda_multi_tx(
            per_tx_rays=per_tx_rays,
            tx_positions=tx_positions,
            rx_pos_3d=rx_pos_3d,
            target_tx_index=target_tx,
            target_rx_index=config.target_rx_index,
            ray_width=config.viz.ray_width,
            tx_radius=config.viz.tx_radius,
            rx_radius_target=config.viz.rx_radius_target,
            rx_radius_other=config.viz.rx_radius_other,
        )
        with open(usda_path, 'w') as f:
            f.write(usda_content)
        exported_files.append(usda_path)
        print(f"   ✅ USDA 저장 완료: {usda_path} (Ray {total_rays}개)")
    except Exception as e:
        print(f"   USDA export 실패: {e}")
        import traceback
        traceback.print_exc()

    # ── 2. Map_Mesh.usd / .obj (PLY → 변환, 맵 메쉬) ───────────
    try:
        print(f"\n📁 Map_Mesh export 중 (PLY → OBJ/USDA)...")
        mesh = o3d.io.read_triangle_mesh(config.map_ply)
        mesh.compute_vertex_normals()
        map_obj_path = os.path.join(config.output_dir_viz, "Map_Mesh.obj")
        map_usd_path = os.path.join(config.output_dir_viz, "Map_Mesh.usda")
        o3d.io.write_triangle_mesh(map_obj_path, mesh)

        verts = np.asarray(mesh.vertices)
        tris  = np.asarray(mesh.triangles)
        pts_str = ", ".join([f"({v[0]:.4f}, {v[1]:.4f}, {v[2]:.4f})" for v in verts])
        counts_str = ", ".join(["3"] * len(tris))
        indices_str = ", ".join([f"{t[0]}, {t[1]}, {t[2]}" for t in tris])
        map_usda = f'''#usda 1.0
(
    defaultPrim = "Map_Mesh"
    metersPerUnit = 1.0
    upAxis = "Y"
)

def Mesh "Map_Mesh" {{
    point3f[] points = [{pts_str}]
    int[] faceVertexCounts = [{counts_str}]
    int[] faceVertexIndices = [{indices_str}]
}}
'''
        with open(map_usd_path, 'w') as f:
            f.write(map_usda)
        exported_files.append(map_usd_path)
        exported_files.append(map_obj_path)
        print(f"   ✅ Map_Mesh.usda / .obj 저장 (정점 {len(verts)}, 삼각형 {len(tris)})")
    except Exception as e:
        print(f"   Map_Mesh export 실패: {e}")

    # ── 3. Map_Mesh_RT.obj (모든 TX/RX 마커 + Ray) ────────────
    try:
        print(f"\n📁 Map_Mesh_RT.obj export 중 (모든 TX Ray 합침)...")
        rt_obj_path = os.path.join(config.output_dir_viz, "Map_Mesh_RT.obj")
        _export_multi_tx_obj(
            per_tx_rays=per_tx_rays,
            tx_positions=tx_positions,
            rx_pos_3d=rx_pos_3d,
            target_rx_index=config.target_rx_index,
            filename=rt_obj_path,
        )
        for fpath in [rt_obj_path, rt_obj_path.replace(".obj", ".mtl")]:
            if os.path.exists(fpath):
                exported_files.append(fpath)
    except Exception as e:
        print(f"   Map_Mesh_RT.obj export 실패: {e}")

    print(f"\n✅ [Export_Agent] 완료 — {len(exported_files)}개 파일 ({time.time()-start:.2f}s)")
    for f in exported_files:
        print(f"   📄 {f}")
    return exported_files


# ============================================================
# 메인 run() 함수 (단일 TX 레거시 — 호환용)
# ============================================================

def run(scene, paths, config: RT_Config, rx_pos_3d: list = None) -> list:
    """
    Ray Tracing 결과를 Blender 등 외부 렌더링 툴용 파일로 export한다.

    Args:
        scene     : Sionna Scene 객체
        paths     : Sionna Paths 객체 (RT_Agent 출력)
        config    : RT_Config
        rx_pos_3d : 3D RX 좌표 목록 (없으면 config에서 재계산)

    Returns:
        list[str]: 생성된 파일 경로 목록
    """
    from RT_utils import get_adaptive_rx_positions
    import open3d as o3d

    print("=" * 60)
    print("📦 [Export_Agent] 3D 파일 Export 시작 (Output 2)")
    print("=" * 60)

    start = time.time()
    os.makedirs(config.output_dir_viz, exist_ok=True)
    exported_files = []

    # 3D RX 좌표 확보
    if rx_pos_3d is None:
        rx_pos_3d = get_adaptive_rx_positions(
            getattr(config, "map_ply_all", None) or config.map_ply, config.rx_positions, config.rx_height
        )

    # ── 1. USDA export (Python으로 직접 생성) ─────────────────
    try:
        print(f"\n📁 USDA export 중 (TX + {len(rx_pos_3d)}개 RX + Ray 경로)...")
        safe_title = "".join(
            c if c.isalnum() or c in "-_" else "_" for c in config.map_title
        )
        usda_path = os.path.join(config.output_dir_viz, f"RT_scene_{safe_title}.usda")

        # Batch 모드: 배치별 paths를 합쳐서 USDA 생성
        if hasattr(config, '_batch_paths_list') and config._batch_paths_list:
            usda_content, total_rays = _build_usda_batch(
                batch_paths_list=config._batch_paths_list,
                tx_pos=config.tx_position,
                all_rx_pos_3d=rx_pos_3d,
                target_rx_index=config.target_rx_index,
                map_title=config.map_title,
                max_rays_per_rx=config.viz.max_rays_per_rx,
                ray_width=config.viz.ray_width,
                tx_radius=config.viz.tx_radius,
                rx_radius_target=config.viz.rx_radius_target,
                rx_radius_other=config.viz.rx_radius_other,
            )
        else:
            usda_content, total_rays = _build_usda(
                paths=paths,
                tx_pos=config.tx_position,
                rx_pos_3d_list=rx_pos_3d,
                target_rx_index=config.target_rx_index,
                map_title=config.map_title,
                max_rays_per_rx=config.viz.max_rays_per_rx,
                ray_width=config.viz.ray_width,
                tx_radius=config.viz.tx_radius,
                rx_radius_target=config.viz.rx_radius_target,
                rx_radius_other=config.viz.rx_radius_other,
            )

        with open(usda_path, 'w') as f:
            f.write(usda_content)

        exported_files.append(usda_path)
        print(f"   ✅ USDA 저장 완료: {usda_path}")
        print(f"      TX: 1개, RX: {len(rx_pos_3d)}개, Ray 경로: {total_rays}개")
        print(f"      Blender: File → Import → USD")
    except Exception as e:
        print(f"   USDA export 실패: {e}")
        import traceback
        traceback.print_exc()

    # ── 2. Map_Mesh.usd (PLY → USD, 맵 메쉬) ────────────────
    try:
        print(f"\n📁 Map_Mesh.usd export 중 (PLY → USD 변환)...")
        mesh = o3d.io.read_triangle_mesh(config.map_ply)
        mesh.compute_vertex_normals()

        # OBJ로 임시 변환 후 USD로 저장 (Open3D는 USD 직접 지원 안 함)
        # 대신 OBJ + USD 둘 다 출력
        map_obj_path = os.path.join(config.output_dir_viz, "Map_Mesh.obj")
        map_usd_path = os.path.join(config.output_dir_viz, "Map_Mesh.usda")
        o3d.io.write_triangle_mesh(map_obj_path, mesh)

        # USDA로 메시 직접 생성
        verts = np.asarray(mesh.vertices)
        tris  = np.asarray(mesh.triangles)

        pts_str = ", ".join([f"({v[0]:.4f}, {v[1]:.4f}, {v[2]:.4f})" for v in verts])
        counts_str = ", ".join(["3"] * len(tris))
        indices_str = ", ".join([f"{t[0]}, {t[1]}, {t[2]}" for t in tris])

        map_usda = f'''#usda 1.0
(
    defaultPrim = "Map_Mesh"
    metersPerUnit = 1.0
    upAxis = "Y"
)

def Mesh "Map_Mesh" {{
    point3f[] points = [{pts_str}]
    int[] faceVertexCounts = [{counts_str}]
    int[] faceVertexIndices = [{indices_str}]
}}
'''
        with open(map_usd_path, 'w') as f:
            f.write(map_usda)

        exported_files.append(map_usd_path)
        exported_files.append(map_obj_path)
        print(f"   ✅ Map_Mesh.usda 저장 완료: {map_usd_path}")
        print(f"   ✅ Map_Mesh.obj 저장 완료: {map_obj_path}")
        print(f"      정점: {len(verts)}개, 삼각형: {len(tris)}개")
        print(f"      Omniverse: File → Open → Map_Mesh.usda")
        print(f"      Blender: File → Import → USD 또는 OBJ")
    except Exception as e:
        print(f"   Map_Mesh export 실패: {e}")

    # ── 3. Map_Mesh_RT.obj (TX/RX 마커 + Ray 경로) ────────────
    try:
        print(f"\n📁 Map_Mesh_RT.obj export 중 (TX/RX 마커 + Ray 경로)...")
        rt_obj_path = os.path.join(config.output_dir_viz, "Map_Mesh_RT.obj")
        _export_all_rx_to_obj(
            paths=paths,
            tx_pos=config.tx_position,
            rx_pos_3d_list=rx_pos_3d,
            target_rx_index=config.target_rx_index,
            filename=rt_obj_path
        )
        for fpath in [rt_obj_path, rt_obj_path.replace(".obj", ".mtl")]:
            if os.path.exists(fpath):
                exported_files.append(fpath)
    except Exception as e:
        print(f"   Map_Mesh_RT.obj export 실패: {e}")

    print(f"\n✅ [Export_Agent] 완료 — {len(exported_files)}개 파일 export ({time.time()-start:.2f}s)")
    for f in exported_files:
        print(f"   📄 {f}")
    print("\n   📋 Blender 임포트 순서:")
    print("      1. File → Import → Wavefront (.obj) → Map_Mesh.obj  (맵 메쉬)")
    print("      2. File → Import → USD → RT_scene_*.usda             (TX/RX/Ray)")
    print("      두 레이어가 같은 좌표계에 겹쳐서 표시됩니다!")

    return exported_files
