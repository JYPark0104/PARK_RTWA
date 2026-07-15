# RT_utils.py
import os
import open3d as o3d
import numpy as np
import tensorflow as tf
from sionna.rt import Transmitter, Receiver, PlanarArray
from scipy.spatial.distance import jensenshannon
import scipy.linalg as la

def get_adaptive_rx_positions(mesh_filename, rx_xy_list, rx_height=1.5, verbose=True):
    """
    RX 좌표 리스트를 3D (X, Y, Z) 로 반환한다.

    - 항목이 2D (X, Y): 3D 맵의 지형 고도를 읽어 지면 + rx_height 로 Z 를 계산한다(기존 동작).
    - 항목이 3D (X, Y, Z): Z 를 그대로 보존한다(예: O2I 벽면 RX). 지면 스냅을 하지 않는다.
      2D/3D 항목을 섞어 넣어도 각 항목별로 처리된다.

    mesh_filename: 단일 PLY 경로(str/Path) 또는 경로 리스트.
        Geo-Radio Env. Twin 은 재질별로 여러 PLY 로 쪼개지므로, 전부를 하나의
        RaycastingScene 에 넣어야 MA(재질부여) 영역의 지면도 인식된다.
        (단일 mesh 만 쓰면 그 영역은 ground hit 실패 → z 폴백 → RT dead zone)

    verbose=False 이면 진행 로그를 출력하지 않습니다 (batch 모드 진행바 보호).
    """
    _p = print if verbose else (lambda *a, **k: None)

    # 단일 경로 / 경로 리스트 정규화
    if isinstance(mesh_filename, (str, bytes)) or hasattr(mesh_filename, "__fspath__"):
        mesh_files = [os.fspath(mesh_filename)]
    else:
        mesh_files = [os.fspath(m) for m in mesh_filename]
    _p(f"🔍 [{len(mesh_files)}개 PLY] 지형 분석 및 레이캐스팅 시작...")

    ray_scene = o3d.t.geometry.RaycastingScene()
    z_max = -np.inf
    n_tri_total = 0
    for mf in mesh_files:
        mesh_legacy = o3d.io.read_triangle_mesh(mf)
        if len(mesh_legacy.triangles) == 0:
            continue
        mesh_tensor = o3d.t.geometry.TriangleMesh.from_legacy(mesh_legacy)
        ray_scene.add_triangles(mesh_tensor)
        n_tri_total += len(mesh_legacy.triangles)
        bb = mesh_legacy.get_axis_aligned_bounding_box()
        z_max = max(z_max, float(np.asarray(bb.max_bound)[2]))

    # 레이 시작 높이: 씬 최고점 + 10m (예전 z=50.0 하드코딩은 고층(>50m) 씬에서 오류)
    cast_z = (z_max + 10.0) if np.isfinite(z_max) else 1000.0

    rx_pos_3d_list = []
    for i, pt in enumerate(rx_xy_list):
        # 3D 좌표가 명시된 경우(예: O2I 벽면 RX)는 z 를 그대로 보존한다(지면 스냅 안 함).
        # 2D (x, y) 입력은 기존과 동일하게 지면 고도 + rx_height 로 재계산한다(하위호환).
        if len(pt) >= 3:
            rx_pos_3d_list.append([float(pt[0]), float(pt[1]), float(pt[2])])
            continue
        x, y = float(pt[0]), float(pt[1])
        ray = o3d.core.Tensor([[x, y, float(cast_z), 0, 0, -1]],
                              dtype=o3d.core.Dtype.Float32)
        ans = ray_scene.cast_rays(ray)
        hit_distance = ans['t_hit'].item()

        if np.isinf(hit_distance):
            _p(f"  ⚠️ RX {i+1}: ({x}, {y}) 위치는 맵 바깥입니다! (Z=0으로 강제 할당)")
            ground_z = 0.0
        else:
            ground_z = cast_z - hit_distance

        final_z = ground_z + rx_height
        rx_pos_3d_list.append([x, y, final_z])

    _p(f"✅ {len(rx_pos_3d_list)}개의 Adaptive RX 좌표 추출 완료! (tri={n_tri_total}, cast_z={cast_z:.1f})")
    return rx_pos_3d_list

def setup_multi_rx_scene(scene, tx_pos, rx_pos_list, num_tx_ant=4, num_rx_ant=4, verbose=True, tx_orientation=None):
    """
    Sionna Scene 객체에 1개의 TX와 다수의 RX를 배치하고 안테나를 설정합니다.

    verbose=False 이면 진행 로그를 출력하지 않습니다 (batch 모드 진행바 보호).
    """
    for tx_name in list(scene.transmitters.keys()):
        scene.remove(tx_name)
        
    for rx_name in list(scene.receivers.keys()):
        scene.remove(rx_name)
    
    # TX 방향(orientation): Sionna Euler 각 (α,β,γ)[rad]. 지향성 패턴에서만 효과(iso 무관).
    if tx_orientation is not None:
        tx = Transmitter(name="tx_main", position=tx_pos, orientation=tuple(tx_orientation))
    else:
        tx = Transmitter(name="tx_main", position=tx_pos)
    scene.add(tx)
    
    for i, pos in enumerate(rx_pos_list):
        rx = Receiver(name=f"rx_{i}", position=pos)
        scene.add(rx)
        
    scene.tx_array = PlanarArray(num_rows=1, num_cols=num_tx_ant, pattern="iso", polarization="V")
    scene.rx_array = PlanarArray(num_rows=1, num_cols=num_rx_ant, pattern="iso", polarization="V")
    
    if verbose:
        print(f"✅ Sionna Scene 세팅 완료: TX 1개, RX {len(rx_pos_list)}개 | 안테나: TX({num_tx_ant}개), RX({num_rx_ant}개)")
    return scene

# --------------------- Compute Loss Functions ---------------------
def compute_rsrp_loss(rsrp_true, rsrp_pred):
    """(1) RSRP Loss (Euclidean Distance / RMSE)"""
    rsrp_true = np.asarray(rsrp_true)
    rsrp_pred = np.asarray(rsrp_pred)
    loss = np.sqrt(np.mean((rsrp_true - rsrp_pred)**2))
    return loss

def compute_padp_js_loss(p_true, p_pred):
    """(2) PADP/PDP Loss (Jensen-Shannon Divergence)"""
    p_true = np.asarray(p_true).flatten()
    p_pred = np.asarray(p_pred).flatten()
    
    sum_true = np.sum(p_true)
    sum_pred = np.sum(p_pred)
    
    if sum_true == 0 and sum_pred == 0:
        return 0.0 
    elif sum_true == 0 or sum_pred == 0:
        return 1.0 
        
    p_true_norm = p_true / sum_true
    p_pred_norm = p_pred / sum_pred
    
    js_distance = jensenshannon(p_true_norm, p_pred_norm, base=2)
    return js_distance ** 2

def compute_cov_log_euclidean_loss(R_true, R_pred, eps=1e-10):
    """(3) Covariance Matrix Loss (Log-Euclidean Distance)"""
    R_true = np.asarray(R_true)
    R_pred = np.asarray(R_pred)
    
    def safe_logm(A):
        eigvals, eigvecs = np.linalg.eigh(A)
        eigvals = np.clip(eigvals, eps, None)
        log_eigvals = np.diag(np.log(eigvals))
        return eigvecs @ log_eigvals @ eigvecs.conj().T

    log_R_true = safe_logm(R_true)
    log_R_pred = safe_logm(R_pred)
    loss = np.linalg.norm(log_R_true - log_R_pred, ord='fro')
    
    return loss

# --------------------- Visualization / Export ---------------------
def export_blender_usd_layers(scene, paths=None, output_dir="."):
    """
    [Layer 1] 기본 맵과 [Layer 2] Ray Tracing 경로를 
    각각 분리된 USD 파일로 추출합니다.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    layer1_filename = os.path.join(output_dir, "Layer1_Map.usd")
    scene.export(layer1_filename)
    print(f"✅ [Layer 1] 기본 맵 추출 완료: {layer1_filename}")
    
    if paths is not None:
        layer2_filename = os.path.join(output_dir, "Layer2_RayPaths.usd")
        paths.export(layer2_filename)
        print(f"✅ [Layer 2] 레이 트레이싱 경로 추출 완료: {layer2_filename}")
    else:
        print("⚠️ 전달된 paths 객체가 없어 Layer 2 추출을 건너뜁니다.")

def export_paths_to_obj(paths, tx_pos, rx_pos, filename):
    """
    Sionna의 Paths 객체를 분석하여 TX/RX 마커가 포함된 
    Blender용 .obj 및 .mtl 파일을 생성합니다.
    """
    base_name = os.path.splitext(filename)[0]
    mtl_filename = base_name + ".mtl"
    mtl_name = os.path.basename(mtl_filename)

    with open(mtl_filename, 'w') as m:
        m.write("# Material definitions\n")
        m.write("newmtl MAT_TX\nKd 1.0 0.0 0.0\n\n")
        m.write("newmtl MAT_RX\nKd 0.0 1.0 0.0\n\n")
        m.write("newmtl MAT_PATH\nKd 0.0 0.6 1.0\n\n")

    vertices = np.array(paths.vertices)
    flat_vertices = vertices.reshape(-1, vertices.shape[-2], 3)
    a, tau = paths.cir()
    a_np = np.array(a)
    
    if a_np.ndim == 7:
        pow_all = np.abs(a_np[0, 0, 0, 0, 0, :, 0])**2
    elif a_np.ndim >= 5:
        pow_all = np.abs(a_np[0, 0, 0, 0, 0, :])**2
    else:
        pow_all = np.abs(a_np)**2
    pow_flat = pow_all.flatten()
    
    with open(filename, 'w') as f:
        f.write(f"mtllib {mtl_name}\n")
        v_idx = 1
        
        f.write("g TX_Marker\nusemtl MAT_TX\n")
        size = 1.5
        tx_cube = [
            [tx_pos[0]-size, tx_pos[1]-size, tx_pos[2]-size], [tx_pos[0]+size, tx_pos[1]-size, tx_pos[2]-size],
            [tx_pos[0]+size, tx_pos[1]+size, tx_pos[2]-size], [tx_pos[0]-size, tx_pos[1]+size, tx_pos[2]-size],
            [tx_pos[0]-size, tx_pos[1]-size, tx_pos[2]+size], [tx_pos[0]+size, tx_pos[1]-size, tx_pos[2]+size],
            [tx_pos[0]+size, tx_pos[1]+size, tx_pos[2]+size], [tx_pos[0]-size, tx_pos[1]+size, tx_pos[2]+size]
        ]
        for v in tx_cube: f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in [[0,1,2,3], [4,5,6,7], [0,1,5,4], [1,2,6,5], [2,3,7,6], [3,0,4,7]]:
            f.write(f"f {v_idx+face[0]} {v_idx+face[1]} {v_idx+face[2]} {v_idx+face[3]}\n")
        v_idx += 8

        f.write("g RX_Marker\nusemtl MAT_RX\n")
        rx_cube = [
            [rx_pos[0]-size, rx_pos[1]-size, rx_pos[2]-size], [rx_pos[0]+size, rx_pos[1]-size, rx_pos[2]-size],
            [rx_pos[0]+size, rx_pos[1]+size, rx_pos[2]-size], [rx_pos[0]-size, rx_pos[1]+size, rx_pos[2]-size],
            [rx_pos[0]-size, rx_pos[1]-size, rx_pos[2]+size], [rx_pos[0]+size, rx_pos[1]-size, rx_pos[2]+size],
            [rx_pos[0]+size, rx_pos[1]+size, rx_pos[2]+size], [rx_pos[0]-size, rx_pos[1]+size, rx_pos[2]+size]
        ]
        for v in rx_cube: f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in [[0,1,2,3], [4,5,6,7], [0,1,5,4], [1,2,6,5], [2,3,7,6], [3,0,4,7]]:
            f.write(f"f {v_idx+face[0]} {v_idx+face[1]} {v_idx+face[2]} {v_idx+face[3]}\n")
        v_idx += 8

        f.write("g Ray_Paths\nusemtl MAT_PATH\n")
        for p_pow, path_verts in zip(pow_flat, flat_vertices):
            if p_pow > 1e-15:
                path_points = [tx_pos]
                for v in path_verts:
                    if not (v[0] == 0.0 and v[1] == 0.0 and v[2] == 0.0):
                        if len(path_points) == 0 or not np.allclose(v, path_points[-1]):
                            path_points.append(v)
                if not np.allclose(rx_pos, path_points[-1]): path_points.append(rx_pos)
                
                indices = []
                for v in path_points:
                    f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
                    indices.append(v_idx)
                    v_idx += 1
                f.write("l " + " ".join(map(str, indices)) + "\n")
                    
    print(f"✅ 시각화 파일 생성 완료: {filename}")
