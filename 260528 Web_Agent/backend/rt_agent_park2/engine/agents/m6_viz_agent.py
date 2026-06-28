"""
viz_agent.py
============
시각화 에이전트 — Output 2 (결과 확인 및 렌더링 데이터).

역할:
  - PDP 스템 차트
  - PADP 버블 차트 (2D)
  - 3D PADP 산점도
  - 공분산 행렬 히트맵 + EVD
  - RX 위치 시각화 (격자 인덱스 확인용)
  - scene.render() — Ray Tracing 3D 이미지
  - scene.preview() — 인터랙티브 3D 뷰어

모든 차트는 output_dir에 PNG로 저장됨.
"""

import os
import sys
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.default_config import RT_Config, SimulationResult


def _add_map_overlay(ax, config: RT_Config) -> None:
    """격자 그림 배경에 실제 Twin Map 상공(top-down) 뷰 외곽선을 그린다.

    config.viz.map_overlay 가 True이고 map_ply 가 존재할 때만 동작하며,
    실패해도 그림 생성 자체는 막지 않는다(경고 후 진행).
    """
    viz = config.viz
    if not getattr(viz, "map_overlay", False):
        return
    ply_path = config.map_ply
    if not ply_path or not os.path.isfile(ply_path):
        return
    try:
        from map_overlay import add_building_overlay
        cache_dir = os.path.join(config.output_dir, ".overlay_cache")
        # z_min=None 이면 PLY z 분포에서 지면을 자동 추정 (수동 설정 불필요)
        z_min = getattr(viz, "map_overlay_z_min", None)
        add_building_overlay(
            ax,
            ply_path=ply_path,
            z_min=z_min,
            color=str(getattr(viz, "map_overlay_color", "#444444")),
            alpha=float(getattr(viz, "map_overlay_alpha", 0.5)),
            linewidth=0.8,
            zorder=10,   # 점(zorder 2~5)보다 높게 → 선이 점 위에 그려짐
            cache_dir=cache_dir,
        )
    except Exception as e:
        print(f"   ⚠️  맵 오버레이 건너뜀: {e}")


def _draw_tx_markers(ax, config: RT_Config, tx_markers=None,
                     target_tx_index: int = None, annotate: bool = True,
                     tx_index_offset: int = 0) -> None:
    """plot 위에 TX 마커(별)를 그린다.

    tx_markers     : 표시할 TX 위치 목록. None이면 config.tx_position 단일 표시.
    target_tx_index: tx_markers 기준 로컬 인덱스. 해당 TX를 빨강으로 강조.
    tx_index_offset: 레이블에 붙일 전역 TX 인덱스 오프셋.
                     (예: TX별 단일 맵 호출 시 offset=t 로 넘기면 "TX{t}"로 표시됨)
    """
    if tx_markers is None:
        tx_markers = [config.tx_position]
    for ti, tp in enumerate(tx_markers):
        global_idx = ti + tx_index_offset
        is_target = (target_tx_index is not None and ti == target_tx_index)
        # TX 마커는 지도 외곽선(zorder=10)보다 위에 그린다 → 항상 보이도록
        ax.scatter(
            tp[0], tp[1],
            c=('red' if is_target else '#ff7f0e'),
            s=(320 if is_target else 200),
            marker='*', zorder=20,
            edgecolors=('darkred' if is_target else 'black'),
            label=('Target TX' if is_target else ('TX' if ti == 0 else None)),
        )
        if annotate:
            ax.annotate(f"TX{global_idx}", (tp[0], tp[1]), textcoords="offset points",
                        xytext=(6, 6), fontsize=9, color='red', fontweight='bold',
                        zorder=21)


def plot_rx_positions(config: RT_Config, rx_pos_3d: list, output_dir: str,
                      tx_markers=None) -> str:
    """격자 RX의 인덱스 번호와 위치를 2D 맵 위에 표시한다 (모든 TX 마커 포함)."""
    fig, ax = plt.subplots(figsize=(12, 10))

    _add_map_overlay(ax, config)

    xs = [p[0] for p in rx_pos_3d]
    ys = [p[1] for p in rx_pos_3d]

    ax.scatter(xs, ys, c='steelblue', s=30, alpha=0.7, zorder=2, label='RX')

    if config.target_rx_index < len(rx_pos_3d):
        tp = rx_pos_3d[config.target_rx_index]
        ax.scatter(tp[0], tp[1], c='orange', s=150, zorder=4,
                   label=f'Target RX (idx={config.target_rx_index})', edgecolors='black')

    step = max(1, len(rx_pos_3d) // 50)
    for i, (x, y) in enumerate(zip(xs, ys)):
        if i % step == 0 or i == config.target_rx_index:
            ax.annotate(str(i), (x, y), textcoords="offset points",
                        xytext=(3, 3), fontsize=6, color='gray')

    if tx_markers is None:
        tx_markers = config.tx_positions
    _draw_tx_markers(ax, config, tx_markers=tx_markers,
                     target_tx_index=config.target_tx_index)

    ax.set_xlabel("X [m]", fontsize=12)
    ax.set_ylabel("Y [m]", fontsize=12)
    ax.set_title(f"RX Grid Positions — {config.map_title}\n"
                 f"(총 {len(rx_pos_3d)}개, grid_n={config.grid_n})", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_aspect('equal')

    filepath = os.path.join(output_dir, "rx_positions.png")
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   📊 RX 위치 시각화 저장: {filepath}")
    return filepath


def plot_pdp(result: SimulationResult, config: RT_Config, output_dir: str) -> str:
    """Power Delay Profile 스템 차트"""
    fig, ax = plt.subplots(figsize=(10, 5))

    if result.is_dead_zone:
        ax.text(0.5, 0.5, 'No Signal (Dead Zone)',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=16, color='gray')
    else:
        y_min = np.min(result.power_dbm) - 10
        markerline, stemlines, baseline = ax.stem(
            result.tau, result.power_dbm,
            linefmt='b-', markerfmt='bx', basefmt=' ', bottom=y_min
        )
        plt.setp(markerline, alpha=0.8, markersize=7)
        plt.setp(stemlines, alpha=0.7, linewidth=1.5)
        ax.set_ylim([y_min, np.max(result.power_dbm) + 10])
        ax.set_xlim([-5, np.max(result.tau) + 5 if len(result.tau) > 0 else 100])

    ax.set_xlabel("Delay [ns]", fontsize=12)
    ax.set_ylabel("Received Power [dBm]", fontsize=12)
    ax.set_title(f"PDP — {config.map_title}\n"
                 f"(Total RSRP: {result.total_rsrp_dbm:.1f} dBm, "
                 f"RX idx={config.target_rx_index})", fontsize=13)
    ax.grid(True, linestyle='--', alpha=0.6)

    filepath = os.path.join(output_dir, "pdp.png")
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   📊 PDP 차트 저장: {filepath}")
    return filepath


def plot_padp_bubble(result: SimulationResult, config: RT_Config, output_dir: str) -> str:
    """Power-Angle-Delay Profile 버블 차트"""
    fig, ax = plt.subplots(figsize=(10, 6))

    if result.is_dead_zone:
        ax.text(0.5, 0.5, 'No Signal (Dead Zone)',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=16, color='gray')
    else:
        vmin, vmax = np.min(result.power_dbm), np.max(result.power_dbm)
        bubble_sizes = np.interp(result.power_dbm, (vmin, vmax), (10, 300))
        sc = ax.scatter(
            result.aoa_azimuth, result.tau,
            s=bubble_sizes, c=result.power_dbm,
            cmap='plasma', vmin=vmin, vmax=vmax,
            alpha=0.7, edgecolors='white', linewidth=0.5
        )
        plt.colorbar(sc, ax=ax, label="Received Power [dBm]")
        ax.set_xlim([-180, 180])
        ax.set_ylim([-5, np.max(result.tau) + 5])
        ax.set_xticks(np.arange(-180, 181, 60))

    ax.set_xlabel("Azimuth AoA [degree]", fontsize=12)
    ax.set_ylabel("Delay [ns]", fontsize=12)
    ax.set_title(f"PADP Bubble Chart — {config.map_title}\n"
                 f"(Total RSRP: {result.total_rsrp_dbm:.1f} dBm)", fontsize=13)
    ax.grid(True, linestyle='--', alpha=0.5)

    filepath = os.path.join(output_dir, "padp_bubble.png")
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   📊 PADP 버블 차트 저장: {filepath}")
    return filepath


def plot_padp_3d(result: SimulationResult, config: RT_Config, output_dir: str) -> str:
    """3D Power-Angle-Delay Profile 산점도"""
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    if result.is_dead_zone:
        ax.text2D(0.5, 0.5, 'No Signal (Dead Zone)',
                  ha='center', va='center', transform=ax.transAxes,
                  fontsize=16, color='gray')
    else:
        vmin, vmax = np.min(result.power_dbm), np.max(result.power_dbm)
        sc = ax.scatter(
            result.aoa_azimuth, result.tau, result.power_dbm,
            c=result.power_dbm, cmap='plasma', vmin=vmin, vmax=vmax,
            s=40, alpha=0.9, edgecolors='w', linewidth=0.5
        )
        for x, y, z in zip(result.aoa_azimuth, result.tau, result.power_dbm):
            ax.plot([x, x], [y, y], [vmin, z], color='gray', alpha=0.4, linewidth=0.8)

        ax.set_xlim([-180, 180])
        ax.set_ylim([-5, np.max(result.tau) + 5])
        ax.set_zlim([vmin, vmax])
        ax.set_xticks(np.arange(-180, 181, 90))
        ax.view_init(elev=20, azim=-50)

        cbar_ax = fig.add_axes([0.92, 0.2, 0.02, 0.6])
        fig.colorbar(sc, cax=cbar_ax, label="Received Power [dBm]")

    ax.set_xlabel("Azimuth AoA [°]", labelpad=10)
    ax.set_ylabel("Delay [ns]", labelpad=10)
    ax.set_zlabel("Power [dBm]", labelpad=10)
    ax.set_title(f"3D PADP — {config.map_title}\n"
                 f"(Total RSRP: {result.total_rsrp_dbm:.1f} dBm)", fontsize=13)

    filepath = os.path.join(output_dir, "padp_3d.png")
    plt.subplots_adjust(right=0.9)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   📊 3D PADP 저장: {filepath}")
    return filepath


def plot_covariance(result: SimulationResult, config: RT_Config, output_dir: str) -> str:
    """R_TX, R_RX 공분산 행렬 히트맵 및 고유값 분해"""
    if result.is_dead_zone or result.R_TX.size == 0:
        print("   ⚠️  Dead Zone — 공분산 행렬 시각화 건너뜀")
        return ""

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, R, label, cmap in [
        (axes[0], result.R_TX, "R_TX", 'Blues'),
        (axes[1], result.R_RX, "R_RX", 'Reds')
    ]:
        mag = np.abs(R)
        cax = ax.imshow(mag, cmap=cmap)
        ax.set_title(f"Magnitude of ${label}$  ({mag.shape[0]}×{mag.shape[1]})",
                     fontsize=14, fontweight='bold', pad=12)
        fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
        # 작은 행렬(≤8)일 때만 눈금/셀 값 텍스트를 표기 (16×16 등은 과밀해서 생략)
        if mag.shape[0] <= 8:
            ax.set_xticks(range(mag.shape[1]))
            ax.set_yticks(range(mag.shape[0]))
            for (i, j), val in np.ndenumerate(mag):
                color = 'white' if val > np.max(mag) / 2 else 'black'
                ax.text(j, i, f"{val:.1e}", ha='center', va='center', color=color, fontsize=9)
        else:
            ax.set_xlabel("Antenna port index")
            ax.set_ylabel("Antenna port index")

    plt.suptitle(f"Spatial Correlation Matrix — {config.map_title}",
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    filepath = os.path.join(output_dir, "covariance_heatmap.png")
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   📊 공분산 히트맵 저장: {filepath}")

    w_tx, _ = np.linalg.eigh(result.R_TX)
    w_rx, _ = np.linalg.eigh(result.R_RX)
    print(f"   ✨ TX Eigenvalues: {np.sort(np.real(w_tx))[::-1]}")
    print(f"   ✨ RX Eigenvalues: {np.sort(np.real(w_rx))[::-1]}")

    return filepath


def render_scene(scene, config: RT_Config, rx_pos_3d: list, output_dir: str) -> str:
    """타겟 RX 단독으로 씬을 재구성하고 scene.render()로 3D 이미지를 렌더링한다."""
    from sionna.rt import Camera, PathSolver, PlanarArray, Receiver
    from RT_utils import get_adaptive_rx_positions, setup_multi_rx_scene

    print("   📸 scene.render() 렌더링 중...")

    try:
        for rx_name in list(scene.receivers.keys()):
            scene.remove(rx_name)

        scene.tx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
        scene.rx_array = PlanarArray(num_rows=1, num_cols=1, pattern="dipole", polarization="cross")

        target_pos = rx_pos_3d[config.target_rx_index]
        rx_vis = Receiver(name="rx_vis", position=target_pos)
        scene.add(rx_vis)
        list(scene.transmitters.values())[0]
        rx_vis.look_at(list(scene.transmitters.values())[0])

        render_solver = PathSolver()
        paths_vis = render_solver(
            scene=scene, max_depth=config.max_depth,
            los=True, specular_reflection=True,
            diffuse_reflection=config.diffuse_reflection,
            samples_per_src=1000, max_num_paths_per_src=10000,
            refraction=True, seed=config.seed
        )

        my_cam = Camera(position=(100, -200, 250), look_at=(0, 0, 0))
        try:
            img = scene.render(camera=my_cam, paths=paths_vis,
                               resolution=[640, 480], num_samples=16, return_bitmap=True)
        except TypeError:
            img = scene.render(camera=my_cam, resolution=[640, 480],
                               num_samples=16, return_bitmap=True)

        img_np = np.array(img)
        filepath = os.path.join(output_dir, "rt_render.png")
        plt.figure(figsize=(8, 6))
        plt.imshow(img_np)
        plt.axis('off')
        plt.title(f"Ray Tracing Render — {config.map_title}\n"
                  f"(Target RX idx={config.target_rx_index})", fontsize=12)
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   📊 렌더링 이미지 저장: {filepath}")

        scene.remove("rx_vis")
        rx_pos_3d_all = get_adaptive_rx_positions(
            getattr(config, "map_ply_all", None) or config.map_ply, config.rx_positions, config.rx_height
        )
        setup_multi_rx_scene(scene, config.tx_position, rx_pos_3d_all,
                             config.num_tx_ant, config.num_rx_ant)
        return filepath

    except Exception as e:
        print(f"   ⚠️  렌더링 실패: {e}")
        return ""


def _build_rsrp_los_matrices(per_tx_all_results: list, num_rx: int):
    """per_tx_all_results → (rsrp_mat, los_mat) (num_tx, num_rx)."""
    num_tx = len(per_tx_all_results)
    rsrp_mat = np.full((num_tx, num_rx), -np.inf, dtype=np.float64)
    los_mat  = np.zeros((num_tx, num_rx), dtype=bool)
    for t, all_results in enumerate(per_tx_all_results):
        for i, r in enumerate(all_results):
            if i >= num_rx:
                break
            rsrp_mat[t, i] = r["total_rsrp_dbm"]
            los_mat[t, i]  = bool(r.get("los", False))
    return rsrp_mat, los_mat


def _combine_rsrp_linear(rsrp_mat: np.ndarray) -> np.ndarray:
    """여러 TX의 RSRP(dBm)를 linear(Watt)에서 합산 후 다시 dBm으로 변환한다.

    Dead Zone(-inf)은 0 W로 취급. 모든 TX가 Dead면 -inf 유지.
    """
    finite = np.isfinite(rsrp_mat)
    watt = np.zeros_like(rsrp_mat, dtype=np.float64)
    watt[finite] = 10.0 ** ((rsrp_mat[finite] - 30.0) / 10.0)  # dBm → W
    total_watt = watt.sum(axis=0)                               # (num_rx,)
    out = np.full(total_watt.shape, -np.inf, dtype=np.float64)
    pos = total_watt > 0
    out[pos] = 10.0 * np.log10(total_watt[pos]) + 30.0
    return out


def run(result: SimulationResult, scene, config: RT_Config, rx_pos_3d: list,
        per_tx_all_results: list = None, tx_positions: list = None) -> list:
    """모든 시각화를 수행하고 저장된 파일 경로 목록을 반환한다 (Multi-TX 대응).

    Args:
        result            : target (TX, RX)의 SimulationResult (PDP/PADP/cov용)
        scene             : Sionna Scene
        config            : RT_Config (config.tx_position은 target TX로 설정된 상태 권장)
        rx_pos_3d         : 3D RX 좌표 목록
        per_tx_all_results: list[num_tx] of (list[num_rx] of per-RX dict)
        tx_positions      : [[x,y,z], ...] (None이면 config.tx_positions)
    """
    print("=" * 60)
    print("📊 [Viz_Agent] 시각화 시작 (Output 2, Multi-TX)")
    print("=" * 60)

    start = time.time()
    os.makedirs(config.output_dir_viz, exist_ok=True)
    saved_files = []
    out_dir = config.output_dir_viz

    if tx_positions is None:
        tx_positions = config.tx_positions
    num_tx = len(tx_positions)
    target_tx = min(config.target_tx_index, num_tx - 1)

    # ── 1. RX 격자(모든 TX 마커) + target (TX,RX) 상세 분석 ──────
    for fn in [
        lambda: plot_rx_positions(config, rx_pos_3d, out_dir, tx_markers=tx_positions),
        lambda: plot_pdp(result, config, out_dir),
        lambda: plot_padp_bubble(result, config, out_dir),
        lambda: plot_padp_3d(result, config, out_dir),
        lambda: plot_covariance(result, config, out_dir),
    ]:
        f = fn()
        if f:
            saved_files.append(f)

    # ── 2. TX별 RSRP heatmap / LoS map + all-TX 집계 ────────────
    if per_tx_all_results:
        try:
            num_rx = len(rx_pos_3d)
            rsrp_mat, los_mat = _build_rsrp_los_matrices(per_tx_all_results, num_rx)

            # 2-1. TX별 개별 맵
            for t in range(num_tx):
                tx_lbl = (f"TX{t} @ ({tx_positions[t][0]:.1f}, "
                          f"{tx_positions[t][1]:.1f}, {tx_positions[t][2]:.1f})")
                dead_t = ~np.isfinite(rsrp_mat[t])
                f = plot_rsrp_heatmap(
                    rsrp_mat[t], config, rx_pos_3d, out_dir,
                    filename=f"rsrp_heatmap_TX{t}.png",
                    tx_markers=[tx_positions[t]], target_tx_index=None,
                    title_label=f"RSRP by {tx_lbl}",
                    tx_index_offset=t,
                )
                if f:
                    saved_files.append(f)
                f = plot_los_map(
                    los_mat[t], config, rx_pos_3d, out_dir, dead_mask=dead_t,
                    filename=f"los_map_TX{t}.png",
                    tx_markers=[tx_positions[t]], target_tx_index=None,
                    title_label=f"LoS/NLoS by {tx_lbl}",
                    tx_index_offset=t,
                )
                if f:
                    saved_files.append(f)

            # 2-2. all-TX 집계 (TX가 2대 이상일 때만 의미 있음)
            if num_tx >= 2:
                # RSRP: linear(Watt) 합산 → dB
                rsrp_all_tx = _combine_rsrp_linear(rsrp_mat)
                dead_all = ~np.isfinite(rsrp_all_tx)
                f = plot_rsrp_heatmap(
                    rsrp_all_tx, config, rx_pos_3d, out_dir,
                    filename="rsrp_heatmap_allTX.png",
                    tx_markers=tx_positions, target_tx_index=target_tx,
                    title_label=f"All-TX combined RSRP ({num_tx} TXs, linear sum -> dB)",
                )
                if f:
                    saved_files.append(f)

                # LoS: 합집합 (어떤 TX라도 LoS면 LoS)
                los_union = los_mat.any(axis=0)
                f = plot_los_map(
                    los_union, config, rx_pos_3d, out_dir, dead_mask=dead_all,
                    filename="los_map_allTX.png",
                    tx_markers=tx_positions, target_tx_index=target_tx,
                    title_label=f"All-TX LoS union (LoS if any of {num_tx} TXs is LoS)",
                    los_label="LoS (>=1 TX)", nlos_label="NLoS (all TX)",
                )
                if f:
                    saved_files.append(f)
        except Exception as e:
            print(f"   ⚠️  RSRP/LoS 맵 실패 (계속 진행): {e}")
            import traceback
            traceback.print_exc()
    else:
        print("   ℹ️  RSRP/LoS 맵 건너뜀 (per_tx_all_results 없음)")

    if config.visualize_ray_tracing:
        f = render_scene(scene, config, rx_pos_3d, out_dir)
        if f:
            saved_files.append(f)
    else:
        print("   ℹ️  visualize_ray_tracing=False — scene.render() 건너뜀")

    print(f"\n✅ [Viz_Agent] 완료 — {len(saved_files)}개 파일 저장 ({time.time()-start:.2f}s)")
    return saved_files


def plot_ray_hitmap(paths, config: RT_Config, rx_pos_3d: list,
                   output_dir: str, max_rays_per_rx: int = None) -> str:
    """
    RX별 유효 Ray 개수를 2D 히트맵으로 시각화한다.

    Args:
        paths          : Sionna Paths 객체 (RT_Agent 출력)
        config         : RT_Config
        rx_pos_3d      : 3D RX 좌표 목록
        output_dir     : 저장 디렉토리
        max_rays_per_rx: 시각화 상한선 표시용 (None이면 데이터용 무제한)

    Returns:
        저장된 PNG 파일 경로
    """
    import numpy as np

    # valid_mask: (num_rx, num_tx, num_paths)
    valid_mask = np.array(paths.valid)
    num_rx = len(rx_pos_3d)

    # RX별 유효 경로 수 계산
    ray_counts = np.zeros(num_rx, dtype=int)
    for rx_idx in range(min(num_rx, valid_mask.shape[0])):
        ray_counts[rx_idx] = int(np.sum(valid_mask[rx_idx]))

    xs = np.array([p[0] for p in rx_pos_3d])
    ys = np.array([p[1] for p in rx_pos_3d])

    # 제목 및 파일명 결정
    if max_rays_per_rx is None:
        title_suffix = "Data (unlimited)"
        filename = "hitmap_data.png"
        cmap = "YlOrRd"
    else:
        title_suffix = f"Visual (max {max_rays_per_rx} per RX)"
        filename = "hitmap_visual.png"
        cmap = "YlGnBu"
        # 시각화용은 max_rays_per_rx로 클리핑
        ray_counts = np.clip(ray_counts, 0, max_rays_per_rx)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # ── 왼쪽: 산점도 히트맵 ──────────────────────────────────
    ax = axes[0]
    _add_map_overlay(ax, config)
    sc = ax.scatter(
        xs, ys,
        c=ray_counts,
        cmap=cmap,
        s=30,
        alpha=0.85,
        edgecolors='none',
        vmin=0,
        vmax=max(ray_counts.max(), 1),
        zorder=3
    )
    plt.colorbar(sc, ax=ax, label="Valid Ray Count")

    # TX 표시
    tx = config.tx_position
    ax.scatter(tx[0], tx[1], c='red', s=300, marker='*',
               zorder=5, label='TX', edgecolors='darkred')
    ax.annotate('TX', (tx[0], tx[1]),
                textcoords="offset points", xytext=(6, 6),
                fontsize=9, color='red', fontweight='bold')

    # Dead Zone (ray=0) 강조
    dead_mask = ray_counts == 0
    if dead_mask.any():
        ax.scatter(xs[dead_mask], ys[dead_mask],
                   c='lightgray', s=15, alpha=0.4,
                   marker='x', label=f'Dead Zone ({dead_mask.sum()})')

    ax.set_xlabel("X [m]", fontsize=11)
    ax.set_ylabel("Y [m]", fontsize=11)
    ax.set_title(f"Valid Rays per RX — {title_suffix}", fontsize=12)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(fontsize=9)

    # ── 오른쪽: 히스토그램 ───────────────────────────────────
    ax2 = axes[1]
    max_count = max(ray_counts.max(), 1)
    bins = min(50, max_count + 1)
    n, bin_edges, patches = ax2.hist(
        ray_counts, bins=bins, range=(0, max_count),
        color='steelblue', edgecolor='white', linewidth=0.5, alpha=0.85
    )

    # Dead Zone 막대 강조
    if bin_edges[0] == 0 and len(patches) > 0:
        patches[0].set_facecolor('lightcoral')
        patches[0].set_label(f'Dead Zone: {dead_mask.sum()}')

    ax2.set_xlabel("Valid Ray Count", fontsize=11)
    ax2.set_ylabel("Number of RX", fontsize=11)
    ax2.set_title("Ray Count Distribution", fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.4, axis='y')

    # 통계 텍스트
    valid_rx = ray_counts[ray_counts > 0]
    stats_text = (
        f"Total RX: {num_rx}\n"
        f"Valid RX: {len(valid_rx)} ({100*len(valid_rx)/num_rx:.1f}%)\n"
        f"Dead Zone: {dead_mask.sum()} ({100*dead_mask.sum()/num_rx:.1f}%)\n"
        f"Total Rays: {ray_counts.sum():,}\n"
        f"Mean (valid RX): {valid_rx.mean():.1f}\n"
        f"Max: {ray_counts.max()}"
    )
    ax2.text(0.97, 0.97, stats_text,
             transform=ax2.transAxes,
             fontsize=9, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.suptitle(
        f"Ray Hitmap — {config.map_title}\n"
        f"TX: ({tx[0]:.1f}, {tx[1]:.1f}, {tx[2]:.1f})",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()

    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   📊 Ray Hitmap 저장: {filepath}")
    return filepath


def _plot_ray_hitmap_from_counts(ray_counts, config: RT_Config, rx_pos_3d: list,
                                  output_dir: str, title_suffix: str,
                                  filename: str, cmap: str, max_clip=None) -> str:
    """
    ray_counts 배열로 직접 히트맵을 그린다 (batch 모드용).
    """
    import numpy as np

    num_rx = len(rx_pos_3d)
    xs = np.array([p[0] for p in rx_pos_3d])
    ys = np.array([p[1] for p in rx_pos_3d])
    dead_mask = ray_counts == 0

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # 왼쪽: 산점도
    ax = axes[0]
    _add_map_overlay(ax, config)
    sc = ax.scatter(xs, ys, c=ray_counts, cmap=cmap, s=30, alpha=0.85,
                    edgecolors='none', vmin=0, vmax=max(ray_counts.max(), 1),
                    zorder=3)
    plt.colorbar(sc, ax=ax, label="Valid Ray Count")

    tx = config.tx_position
    ax.scatter(tx[0], tx[1], c='red', s=300, marker='*', zorder=5,
               label='TX', edgecolors='darkred')
    if dead_mask.any():
        ax.scatter(xs[dead_mask], ys[dead_mask], c='lightgray', s=15, alpha=0.4,
                   marker='x', label=f'Dead Zone ({dead_mask.sum()})')

    ax.set_xlabel("X [m]", fontsize=11)
    ax.set_ylabel("Y [m]", fontsize=11)
    ax.set_title(f"Valid Rays per RX — {title_suffix}", fontsize=12)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(fontsize=9)

    # 오른쪽: 히스토그램
    ax2 = axes[1]
    max_count = max(ray_counts.max(), 1)
    bins = min(50, max_count + 1)
    n, bin_edges, patches = ax2.hist(ray_counts, bins=bins, range=(0, max_count),
                                      color='steelblue', edgecolor='white', linewidth=0.5, alpha=0.85)
    if bin_edges[0] == 0 and len(patches) > 0:
        patches[0].set_facecolor('lightcoral')

    ax2.set_xlabel("Valid Ray Count", fontsize=11)
    ax2.set_ylabel("Number of RX", fontsize=11)
    ax2.set_title("Ray Count Distribution", fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.4, axis='y')

    valid_rx = ray_counts[ray_counts > 0]
    stats_text = (
        f"Total RX: {num_rx}\n"
        f"Valid RX: {len(valid_rx)} ({100*len(valid_rx)/num_rx:.1f}%)\n"
        f"Dead Zone: {dead_mask.sum()} ({100*dead_mask.sum()/num_rx:.1f}%)\n"
        f"Total Rays: {ray_counts.sum():,}\n"
        f"Mean (valid RX): {valid_rx.mean():.1f}\n"
        f"Max: {ray_counts.max()}"
    )
    ax2.text(0.97, 0.97, stats_text, transform=ax2.transAxes,
             fontsize=9, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.suptitle(f"Ray Hitmap — {config.map_title}\nTX: ({tx[0]:.1f}, {tx[1]:.1f}, {tx[2]:.1f})",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()

    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   📊 Ray Hitmap 저장: {filepath}")
    return filepath


def plot_rsrp_heatmap(rsrp_values, config: RT_Config, rx_pos_3d: list,
                      output_dir: str, filename: str = "rsrp_heatmap.png",
                      tx_markers=None, target_tx_index: int = None,
                      title_label: str = None, tx_index_offset: int = 0) -> str:
    """각 RX 격자의 총 RSRP(dBm) 값을 2D 히트맵으로 시각화한다.

    Args:
        rsrp_values    : shape (num_rx,) 각 RX의 total_rsrp_dbm (Dead Zone은 -inf)
        config         : RT_Config
        rx_pos_3d      : 3D RX 좌표 목록 (rsrp_values와 인덱스 일치)
        output_dir     : 저장 디렉토리
        filename       : 저장 파일명
        tx_markers     : 표시할 TX 목록 [[x,y,z],...] (None이면 config.tx_position 단일)
        target_tx_index: 강조할 TX 인덱스 (tx_markers 기준)
        title_label    : suptitle 보조 라벨 (예: "TX0", "All-TX 합산")

    Returns:
        저장된 PNG 파일 경로 (실패 시 "")
    """
    import numpy as np

    rsrp = np.asarray(rsrp_values, dtype=np.float64).ravel()
    num_rx = min(len(rx_pos_3d), rsrp.shape[0])
    rsrp = rsrp[:num_rx]
    xs = np.array([rx_pos_3d[i][0] for i in range(num_rx)])
    ys = np.array([rx_pos_3d[i][1] for i in range(num_rx)])

    # Dead Zone (-inf 또는 비유한값) 분리
    finite_mask = np.isfinite(rsrp)
    dead_mask = ~finite_mask

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # ── 왼쪽: RSRP 산점도 히트맵 ──────────────────────────────
    ax = axes[0]
    _add_map_overlay(ax, config)

    if finite_mask.any():
        vmin = float(np.percentile(rsrp[finite_mask], 2))
        vmax = float(np.percentile(rsrp[finite_mask], 98))
        if vmax <= vmin:
            vmin, vmax = float(rsrp[finite_mask].min()), float(rsrp[finite_mask].max() + 1e-6)
        sc = ax.scatter(
            xs[finite_mask], ys[finite_mask], c=rsrp[finite_mask],
            cmap="jet", s=30, alpha=0.9, edgecolors='none',
            vmin=vmin, vmax=vmax, zorder=3,
        )
        plt.colorbar(sc, ax=ax, label="RSRP [dBm]")

    # Dead Zone 강조
    if dead_mask.any():
        ax.scatter(xs[dead_mask], ys[dead_mask], c='lightgray', s=15,
                   alpha=0.5, marker='x', zorder=2,
                   label=f'Dead Zone ({int(dead_mask.sum())})')

    # TX 표시 (단일 또는 다중)
    _draw_tx_markers(ax, config, tx_markers=tx_markers,
                     target_tx_index=target_tx_index,
                     tx_index_offset=tx_index_offset)

    ax.set_xlabel("X [m]", fontsize=11)
    ax.set_ylabel("Y [m]", fontsize=11)
    ax.set_title("RSRP per RX", fontsize=12)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(fontsize=9)

    # ── 오른쪽: RSRP 히스토그램 ───────────────────────────────
    ax2 = axes[1]
    if finite_mask.any():
        ax2.hist(rsrp[finite_mask], bins=40, color='steelblue',
                 edgecolor='white', linewidth=0.5, alpha=0.85)
        ax2.set_xlabel("RSRP [dBm]", fontsize=11)
        ax2.set_ylabel("Number of RX", fontsize=11)
        ax2.set_title("RSRP Distribution", fontsize=12)
        ax2.grid(True, linestyle='--', alpha=0.4, axis='y')

        valid = rsrp[finite_mask]
        stats_text = (
            f"Total RX: {num_rx}\n"
            f"Valid RX: {len(valid)} ({100*len(valid)/num_rx:.1f}%)\n"
            f"Dead Zone: {int(dead_mask.sum())} ({100*dead_mask.sum()/num_rx:.1f}%)\n"
            f"Mean: {valid.mean():.1f} dBm\n"
            f"Max: {valid.max():.1f} dBm\n"
            f"Min: {valid.min():.1f} dBm"
        )
        ax2.text(0.97, 0.97, stats_text, transform=ax2.transAxes,
                 fontsize=9, verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    else:
        ax2.text(0.5, 0.5, "모든 RX가 Dead Zone", transform=ax2.transAxes,
                 ha='center', va='center', fontsize=12, color='gray')

    if title_label is None:
        _t = config.tx_position
        title_label = f"TX: ({_t[0]:.1f}, {_t[1]:.1f}, {_t[2]:.1f})"
    plt.suptitle(
        f"RSRP Heatmap — {config.map_title}\n{title_label}",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()

    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   📊 RSRP Heatmap 저장: {filepath}")
    return filepath


def plot_los_map(los_values, config: RT_Config, rx_pos_3d: list,
                 output_dir: str, dead_mask=None,
                 filename: str = "los_map.png",
                 tx_markers=None, target_tx_index: int = None,
                 title_label: str = None,
                 los_label: str = "LoS", nlos_label: str = "NLoS",
                 tx_index_offset: int = 0) -> str:
    """각 RX 격자의 LoS/NLoS 여부를 2D 맵으로 시각화한다.

    Args:
        los_values     : shape (num_rx,) bool — RX별 LoS(직시) 경로 존재 여부
        config         : RT_Config
        rx_pos_3d      : 3D RX 좌표 목록 (los_values와 인덱스 일치)
        dead_mask      : shape (num_rx,) bool — 유효 경로가 전혀 없는 Dead Zone (선택)
        output_dir     : 저장 디렉토리
        filename       : 저장 파일명
        tx_markers     : 표시할 TX 목록 (None이면 config.tx_position 단일)
        target_tx_index: 강조할 TX 인덱스
        title_label    : suptitle 보조 라벨
        los_label/nlos_label : 범례 라벨 (all-TX 합집합 시 의미 구분용)

    Returns:
        저장된 PNG 파일 경로
    """
    import numpy as np

    los = np.asarray(los_values, dtype=bool).ravel()
    num_rx = min(len(rx_pos_3d), los.shape[0])
    los = los[:num_rx]
    xs = np.array([rx_pos_3d[i][0] for i in range(num_rx)])
    ys = np.array([rx_pos_3d[i][1] for i in range(num_rx)])

    if dead_mask is not None:
        dead = np.asarray(dead_mask, dtype=bool).ravel()[:num_rx]
    else:
        dead = np.zeros(num_rx, dtype=bool)

    los_mask  = los & (~dead)
    nlos_mask = (~los) & (~dead)   # 경로는 있으나 LoS는 없음 (반사/회절로만 도달)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # ── 왼쪽: LoS/NLoS 맵 ─────────────────────────────────────
    ax = axes[0]
    _add_map_overlay(ax, config)

    if nlos_mask.any():
        ax.scatter(xs[nlos_mask], ys[nlos_mask], c='#d62728', s=28, alpha=0.85,
                   edgecolors='none', zorder=3, label=f'{nlos_label} ({int(nlos_mask.sum())})')
    if los_mask.any():
        ax.scatter(xs[los_mask], ys[los_mask], c='#1f77b4', s=28, alpha=0.9,
                   edgecolors='none', zorder=4, label=f'{los_label} ({int(los_mask.sum())})')
    if dead.any():
        ax.scatter(xs[dead], ys[dead], c='lightgray', s=14, alpha=0.5,
                   marker='x', zorder=2, label=f'Dead Zone ({int(dead.sum())})')

    _draw_tx_markers(ax, config, tx_markers=tx_markers,
                     target_tx_index=target_tx_index,
                     tx_index_offset=tx_index_offset)

    ax.set_xlabel("X [m]", fontsize=11)
    ax.set_ylabel("Y [m]", fontsize=11)
    ax.set_title("LoS / NLoS per RX", fontsize=12)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(fontsize=9, loc='upper right')

    # ── 오른쪽: 카테고리 막대 ─────────────────────────────────
    ax2 = axes[1]
    cats = [los_label, nlos_label, 'Dead Zone']
    counts = [int(los_mask.sum()), int(nlos_mask.sum()), int(dead.sum())]
    colors = ['#1f77b4', '#d62728', 'lightgray']
    bars = ax2.bar(cats, counts, color=colors, edgecolor='black', linewidth=0.6)
    for b, c in zip(bars, counts):
        pct = 100 * c / num_rx if num_rx else 0
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height(),
                 f"{c}\n({pct:.1f}%)", ha='center', va='bottom', fontsize=10)
    ax2.set_ylabel("Number of RX", fontsize=11)
    ax2.set_title(f"LoS Coverage (Total RX: {num_rx})", fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.4, axis='y')

    if title_label is None:
        _t = config.tx_position
        title_label = f"TX: ({_t[0]:.1f}, {_t[1]:.1f}, {_t[2]:.1f})"
    plt.suptitle(
        f"LoS / NLoS Map — {config.map_title}\n{title_label}",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()

    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   📊 LoS/NLoS Map 저장: {filepath}")
    return filepath
