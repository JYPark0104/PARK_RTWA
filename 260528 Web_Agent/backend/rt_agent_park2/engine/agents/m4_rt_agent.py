"""
m4_rt_agent.py
===========
Ray Tracing 연산 에이전트.

역할:
  - Sionna PathSolver로 Ray Tracing 연산 수행
  - CIR (채널 계수 a, 지연 tau) 추출
  - 수신 방위각 phi_r, 송신 방위각 phi_t 추출

반환:
  (RawRayData, paths): 원시 경로 데이터, Sionna Paths 객체
"""

import time
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.default_config import RT_Config, RawRayData


def run(scene, config: RT_Config, quiet: bool = False):
    """
    Sionna PathSolver로 Ray Tracing 연산을 수행하고 원시 데이터를 추출한다.

    Args:
        scene : Sionna Scene 객체 (Scene_Agent 출력)
        config: RT_Config 객체
        quiet : True이면 진행 로그 출력을 생략한다 (batch 모드에서 진행바와 충돌 방지).

    Returns:
        tuple: (RawRayData, paths)
            - RawRayData: 원시 경로 데이터 (a, tau, phi_r, phi_t)
            - paths     : Sionna Paths 객체 (Export_Agent에서 사용)
    """
    from sionna.rt import PathSolver

    # quiet=True 이면 아무것도 출력하지 않는 print 대체 함수
    _p = (lambda *a, **k: None) if quiet else print

    _p("=" * 60)
    _p("🚀 [RT_Agent] Ray Tracing 연산 시작")
    _p("=" * 60)

    start = time.time()

    # ── 1. PathSolver 인스턴스화 및 연산 ─────────────────────
    _p(f"\n⚡ 연산 가동 중 (H100 파워🔥)")
    _p(f"   num_samples={config.num_samples:,}, max_depth={config.max_depth}, "
       f"diffuse_reflection={config.diffuse_reflection}")

    solver = PathSolver()
    paths = solver(
        scene=scene,
        max_depth=config.max_depth,
        los=config.los,
        specular_reflection=config.specular_reflection,
        diffuse_reflection=config.diffuse_reflection,
        samples_per_src=config.num_samples,
        max_num_paths_per_src=config.max_num_paths,
        refraction=config.refraction,
        diffraction=config.diffraction,
        edge_diffraction=config.edge_diffraction,
        diffraction_lit_region=config.diffraction_lit_region,
        synthetic_array=config.synthetic_array,
        seed=config.seed
    )

    elapsed = time.time() - start
    _p(f"   -> 연산 완료! ({elapsed:.2f}초 소요)")

    # valid path 수 진단
    try:
        valid_np = np.array(paths.valid)  # (num_rx, num_tx, num_paths)
        total_valid = int(np.sum(valid_np))
        num_rx = valid_np.shape[0]
        _p(f"   📊 전체 valid path: {total_valid}개 (RX당 평균 {total_valid/max(num_rx,1):.1f}개)")
    except Exception:
        pass

    # ── 2. CIR 추출 (채널 계수 a, 지연 tau) ──────────────────
    a, tau = paths.cir()
    a_np = np.array(a)
    tau_np = np.array(tau)

    # ── 3. 방위각 추출 ────────────────────────────────────────
    phi_r_np = np.array(paths.phi_r)
    phi_t_np = np.array(paths.phi_t)

    # ── 3-1. RX별 LoS(직시) 경로 존재 여부 계산 ────────────────
    #   LoS 경로 = 유효(valid) AND 모든 depth의 interaction이 NONE(0)
    #   interactions: (max_depth, num_rx, num_tx, num_paths)
    #   valid       : (num_rx, num_tx, num_paths)
    try:
        valid_arr = np.array(paths.valid)
        inter_arr = np.array(paths.interactions)
        no_interaction = np.all(inter_arr == 0, axis=0)          # (num_rx, num_tx, num_paths)
        los_path_mask = np.asarray(valid_arr, dtype=bool) & no_interaction
        # TX/경로 축에 대해 OR → RX별 LoS 존재 여부
        los_per_rx = los_path_mask.any(axis=tuple(range(1, los_path_mask.ndim)))
        los_per_rx = np.asarray(los_per_rx, dtype=bool)
        n_los = int(los_per_rx.sum())
        _p(f"   📡 LoS 보유 RX: {n_los}개 / {los_per_rx.shape[0]}개")
    except Exception as e:
        _p(f"   ⚠️  LoS 판별 실패 (빈 배열로 처리): {e}")
        los_per_rx = np.array([], dtype=bool)

    # ── 4. 타겟 RX 유효 경로 수 출력 ─────────────────────────
    num_rx = len(config.rx_positions)
    if num_rx > 0 and tau_np.size > 0:
        try:
            pow_all = (a_np[0].reshape(num_rx, -1) ** 2
                       + a_np[1].reshape(num_rx, -1) ** 2)
            target_pow = pow_all[config.target_rx_index]
            valid_count = int(np.sum(target_pow > config.threshold_watt))
            _p(f"   📊 RX {config.target_rx_index + 1}의 유효 경로: {valid_count}개")
        except Exception:
            _p(f"   📊 전체 경로 수: {tau_np.shape[-1] if tau_np.ndim > 0 else 'N/A'}")

    raw_data = RawRayData(
        a=a_np,
        tau=tau_np,
        phi_r=phi_r_np,
        phi_t=phi_t_np,
        los=los_per_rx
    )

    _p(f"\n✅ [RT_Agent] 완료 ({elapsed:.2f}s)")

    return raw_data, paths
