"""
m2_config_agent.py
===============
파라미터 설정 및 격자 RX 생성 에이전트.

역할:
  - RT_Config 객체 생성 (파라미터 기본값 + 외부 오버라이드)
  - PLY 파일의 bounding box 기반으로 grid_n × grid_n 격자 후보 생성
  - Open3D 레이캐스팅으로 맵 바깥 위치 필터링 → 유효 RX (x, y) 목록 생성

반환:
  RT_Config (rx_positions 필드가 채워진 상태)
"""

import time
import numpy as np
import open3d as o3d
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.default_config import RT_Config


def run(
    map_ply: str,
    map_xml: str,
    map_title: str = "Map",
    tx_position: tuple = (-12.517, 14.894, 41.0),
    **kwargs
) -> RT_Config:
    """
    RT_Config를 생성하고 PLY bounding box 기반으로 격자 RX 좌표를 생성한다.

    Args:
        map_ply    : PLY 파일 경로 (bounding box 추출 및 레이캐스팅에 사용)
        map_xml    : XML 씬 파일 경로
        map_title  : 맵 이름 (로그 출력용)
        tx_position: TX 위치 (x, y, z)
        **kwargs   : RT_Config 필드 오버라이드 (grid_n, rx_height, num_samples 등)

    Returns:
        RT_Config: rx_positions 필드가 채워진 설정 객체
    """
    print("=" * 60)
    print("⚙️  [Config_Agent] 파라미터 설정 및 격자 RX 생성 시작")
    print("=" * 60)

    start = time.time()

    # ── 1. RT_Config 생성 (기본값 + 오버라이드) ───────────────
    config = RT_Config(
        map_xml=map_xml,
        map_ply=map_ply,
        map_title=map_title,
        tx_position=tx_position,
    )

    # kwargs로 전달된 파라미터로 오버라이드
    allowed_fields = {f.name for f in config.__dataclass_fields__.values()}
    for key, value in kwargs.items():
        if key in allowed_fields:
            setattr(config, key, value)
        else:
            print(f"   ⚠️  알 수 없는 파라미터 무시: {key}")

    # ── 2. PLY 로드 및 bounding box 추출 ─────────────────────
    print(f"\n🔍 [{map_ply}] bounding box 분석 중...")
    mesh_legacy = o3d.io.read_triangle_mesh(map_ply)
    mesh_tensor = o3d.t.geometry.TriangleMesh.from_legacy(mesh_legacy)
    ray_scene = o3d.t.geometry.RaycastingScene()
    ray_scene.add_triangles(mesh_tensor)

    bbox = mesh_legacy.get_axis_aligned_bounding_box()
    x_min, y_min, _ = np.asarray(bbox.min_bound)
    x_max, y_max, _ = np.asarray(bbox.max_bound)
    print(f"   X 범위: [{x_min:.1f}, {x_max:.1f}]")
    print(f"   Y 범위: [{y_min:.1f}, {y_max:.1f}]")

    # ── 3. 격자 후보 생성 (margin 적용: 가장자리 제거) ────────
    margin = getattr(config, 'margin', 0.0)
    if margin > 0:
        x_range = x_max - x_min
        y_range = y_max - y_min
        x_min += x_range * margin
        x_max -= x_range * margin
        y_min += y_range * margin
        y_max -= y_range * margin
        print(f"   Margin {margin*100:.0f}% 적용 → X: [{x_min:.1f}, {x_max:.1f}], Y: [{y_min:.1f}, {y_max:.1f}]")

    xs = np.linspace(x_min, x_max, config.grid_n)
    ys = np.linspace(y_min, y_max, config.grid_n)
    candidates = [[float(x), float(y)] for y in ys for x in xs]
    print(f"\n   격자 후보: {len(candidates)}개 ({config.grid_n}×{config.grid_n})")

    # ── 4. 레이캐스팅으로 유효 지면 위치 필터링 ──────────────
    valid_positions = []
    for x, y in candidates:
        ray = o3d.core.Tensor(
            [[x, y, config.raycasting_z, 0, 0, -1]],
            dtype=o3d.core.Dtype.Float32
        )
        ans = ray_scene.cast_rays(ray)
        hit_dist = ans['t_hit'].item()
        if not np.isinf(hit_dist):   # 맵에 hit → 유효한 지면
            valid_positions.append([x, y])

    config.rx_positions = valid_positions

    print(f"✅ 격자 후보 {len(candidates)}개 중 유효 RX {len(valid_positions)}개 선택")
    print(f"📍 타겟 맵: {map_title}")
    print(f"📍 TX 개수: {len(config.tx_positions)}대")
    for _ti, _tp in enumerate(config.tx_positions):
        print(f"      TX{_ti}: ({_tp[0]:.2f}, {_tp[1]:.2f}, {_tp[2]:.2f})")
    print(f"📍 총 RX 수: {len(valid_positions)}개")
    print(f"📍 타겟 (TX, RX) 인덱스: ({config.target_tx_index}, {config.target_rx_index})")
    print(f"\n✅ [Config_Agent] 완료 ({time.time()-start:.2f}s)")

    return config
