"""
scene_agent.py
==============
씬 로드 및 TX/RX 환경 세팅 에이전트.

역할:
  - Sionna load_scene()으로 XML 씬 파일 로드
  - 라디오 재질 산란 계수 설정
  - RT_utils.get_adaptive_rx_positions()로 3D RX 좌표 계산
  - RT_utils.setup_multi_rx_scene()으로 TX/RX 배치
  - 안테나 설정 (TX: iso, RX: dipole)
  - 모든 RX가 TX를 바라보도록 설정

반환:
  (scene, rx_pos_3d): Sionna Scene 객체, 3D RX 좌표 목록
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.default_config import RT_Config


def run(config: RT_Config):
    """
    Sionna 씬을 로드하고 TX/RX 환경을 세팅한다.

    Args:
        config: RT_Config 객체 (map_xml, map_ply, rx_positions 등 포함)

    Returns:
        tuple: (scene, rx_pos_3d)
            - scene    : Sionna Scene 객체
            - rx_pos_3d: 3D RX 좌표 목록 [(x, y, z), ...]

    Raises:
        FileNotFoundError: XML 파일이 존재하지 않을 경우
    """
    from sionna.rt import load_scene, PlanarArray
    from RT_utils import get_adaptive_rx_positions, setup_multi_rx_scene

    print("=" * 60)
    print("🛠️  [Scene_Agent] 씬 로드 및 TX/RX 환경 세팅 시작")
    print("=" * 60)

    start = time.time()

    # ── 1. XML 파일 존재 확인 ─────────────────────────────────
    if not os.path.exists(config.map_xml):
        msg = f"❌ XML 파일을 찾을 수 없습니다: {config.map_xml}"
        print(msg)
        raise FileNotFoundError(msg)

    # ── 2. 씬 로드 ────────────────────────────────────────────
    print(f"\n🔄 씬 로드 중: {config.map_title}")

    # Mitsuba PLY 색상 경고 억제 (치명적이지 않은 빨간 글씨 제거)
    import logging
    logging.getLogger('mitsuba').setLevel(logging.ERROR)

    scene = load_scene(config.map_xml)

    # Mitsuba가 ANSI 빨간색 코드(\033[31m)를 출력하고 리셋을 안 하는 버그 보정
    # → 강제로 ANSI 리셋 코드 출력하여 이후 글씨를 흰색으로 복구
    print("\033[0m", end="", flush=True)
    scene.frequency = config.frequency
    print(f"   ✅ 씬 로드 완료 (주파수: {config.frequency/1e9:.1f} GHz)")

    # ── 3. 라디오 재질 설정 ───────────────────────────────────
    from sionna.rt import LambertianPattern, DirectivePattern, BackscatteringPattern

    mat = list(scene.radio_materials.values())[0]
    mat.scattering_coefficient = config.scattering_coefficient
    mat.xpd_coefficient        = config.xpd_coefficient

    # 산란 패턴 선택
    pattern_name = config.scattering_pattern.lower()
    if pattern_name == "directive":
        mat.scattering_pattern = DirectivePattern(alpha_r=config.directive_alpha_r)
    elif pattern_name == "backscattering":
        mat.scattering_pattern = BackscatteringPattern(
            alpha_r=config.backscattering_alpha_r,
            alpha_i=config.backscattering_alpha_i,
            lambda_=config.backscattering_lambda
        )
    else:
        mat.scattering_pattern = LambertianPattern()

    print(f"   ✅ 라디오 재질 설정 완료 "
          f"(S={config.scattering_coefficient}, Kx={config.xpd_coefficient}, "
          f"pattern={config.scattering_pattern})")

    # ── 4. 3D RX 좌표 계산 (지형 고도 적응형) ─────────────────
    print(f"\n🔍 3D RX 좌표 계산 중 ({len(config.rx_positions)}개 위치)...")
    rx_pos_3d = get_adaptive_rx_positions(
        getattr(config, "map_ply_all", None) or config.map_ply,
        config.rx_positions,
        rx_height=config.rx_height
    )

    # ── 5. TX/RX 씬 배치 ─────────────────────────────────────
    setup_multi_rx_scene(
        scene,
        config.tx_position,
        rx_pos_3d,
        num_tx_ant=config.num_tx_ant,
        num_rx_ant=config.num_rx_ant
    )

    # ── 6. 안테나 재설정 (config.yaml 기반, 평면배열 행×열) ──────
    scene.tx_array = PlanarArray(
        num_rows=config.num_tx_rows, num_cols=config.num_tx_cols,
        pattern=config.tx_pattern,
        polarization=config.tx_polarization
    )
    scene.rx_array = PlanarArray(
        num_rows=config.num_rx_rows, num_cols=config.num_rx_cols,
        pattern=config.rx_pattern,
        polarization=config.rx_polarization
    )
    print(f"   ✅ 안테나 설정 완료 "
          f"(TX: {config.num_tx_rows}×{config.num_tx_cols} {config.tx_pattern}/{config.tx_polarization} "
          f"→ {config.num_tx_ant}포트, "
          f"RX: {config.num_rx_rows}×{config.num_rx_cols} {config.rx_pattern}/{config.rx_polarization} "
          f"→ {config.num_rx_ant}포트)")

    # ── 7. 모든 RX가 TX를 바라보도록 설정 ────────────────────
    current_tx = list(scene.transmitters.values())[0]
    for rx in scene.receivers.values():
        rx.look_at(current_tx)
    print("   ✅ 모든 RX → TX look_at 설정 완료")

    print(f"\n✅ [Scene_Agent] 완료 (TX: {len(scene.transmitters)}개, RX: {len(scene.receivers)}개, {time.time()-start:.2f}s)")

    return scene, rx_pos_3d
