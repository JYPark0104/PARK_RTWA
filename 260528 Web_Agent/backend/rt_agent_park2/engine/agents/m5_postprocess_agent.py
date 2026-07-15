"""
m5_postprocess_agent.py
====================
후처리 에이전트 — Output 1 (Channel Agent용 수학적 통신 데이터) 생성.

역할:
  - 모든 RX에 대해 유효 경로 필터링, RSRP, 공분산 행렬 계산
  - target_rx_index에 해당하는 RX의 결과를 SimulationResult로 반환 (시각화용)
  - 전체 RX 결과를 .npz 파일로 저장 (Channel Agent용 Output 1)

Output 1 .npz 구조:
  - rsrp_all        : shape (num_rx,)         — 각 RX의 총 RSRP (dBm)
  - tau_rx{i}       : shape (num_valid_paths,) — RX i의 유효 경로 지연 (ns)
  - power_rx{i}     : shape (num_valid_paths,) — RX i의 경로별 수신 전력 (dBm)
  - aoa_rx{i}       : shape (num_valid_paths,) — RX i의 방위각 / AoA (degree)
  - aod_rx{i}       : shape (num_valid_paths,) — RX i의 송신 방위각 / AoD (degree)
  - R_TX_rx{i}      : shape (num_tx_ant, num_tx_ant) — RX i의 TX 공분산 행렬
  - R_RX_rx{i}      : shape (num_rx_ant, num_rx_ant) — RX i의 RX 공분산 행렬
  - rx_positions    : shape (num_rx, 2)        — 각 RX의 (x, y) 좌표
  - target_rx_index : scalar                   — 분석 대상 RX 인덱스

batch_rx 모드 사용 흐름:
  1. 배치 루프마다 run_batch(raw_data, batch_rx_positions, config) 호출
     → 내부적으로 _accumulated_results 리스트에 결과를 누적
  2. 루프 종료 후 finalize_batch(config) 호출
     → 누적 결과를 run._all_results 에 저장하고 SimulationResult 반환
  3. save_output1() 으로 전체 결과를 .npz 저장
"""

import time
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.default_config import RT_Config, RawRayData, SimulationResult


def _extract_mimo_cir(raw_data: RawRayData, num_rx: int):
    """
    RawRayData를 안테나 차원이 보존된 형태로 변환한다.

    Sionna `paths.cir()`의 a 텐서(np.array 변환 후)는
        (2, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time)
    형태이며 dim0 = [실수부, 허수부] 이다. (synthetic_array=True 기준)
    이 함수는 단일 TX·단일 time-step을 가정하여 다음을 추출한다.

    Returns (실패 시 None):
        a_c       : (num_rx, num_rx_ant, num_tx_ant, num_paths) complex128
        pow_path  : (num_rx, num_paths) — 경로별 '안테나 평균' 수신 전력 (W)
        tau_all   : (num_rx, num_paths) — 경로 지연 (ns)
        phi_r_all : (num_rx, num_paths) — 수신 방위각 (rad)
        phi_t_all : (num_rx, num_paths) — 송신 방위각 (rad)
    """
    a_np     = np.asarray(raw_data.a)
    tau_np   = np.asarray(raw_data.tau)
    phi_r_np = np.asarray(raw_data.phi_r)
    phi_t_np = np.asarray(raw_data.phi_t)

    # ── 복소 CIR 복원 (a_np[0]=실수부, a_np[1]=허수부) ──────────
    if a_np.ndim >= 6 and a_np.shape[0] == 2:
        a_c = a_np[0].astype(np.float64) + 1j * a_np[1].astype(np.float64)
        # a_c: (num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time)
        # 단일 TX(축2)·단일 time-step(마지막 축) 제거
        a_c = a_c[:, :, 0, :, :, 0]          # (num_rx, R, T, num_paths)
    elif a_np.ndim == 2 and a_np.shape[0] == 2:
        # 매우 예외적 형태 — 단일 안테나로 간주
        a_c = (a_np[0] + 1j * a_np[1]).reshape(num_rx, 1, 1, -1)
    else:
        # 복소 dtype이 그대로 들어온 경우 등 — 단일 안테나 폴백
        a_c = np.asarray(a_np, dtype=np.complex128).reshape(num_rx, 1, 1, -1)

    num_paths = a_c.shape[-1]

    # 경로별 안테나-평균 전력 (단일 안테나일 때 |a|² 과 동일 → 하위호환)
    pow_path = np.mean(np.abs(a_c) ** 2, axis=(1, 2))   # (num_rx, num_paths)

    # tau / phi: (num_rx, num_tx, num_paths) → 단일 TX 축 제거
    def _squeeze_tx(arr):
        arr = np.asarray(arr)
        if arr.ndim == 3:
            return arr[:, 0, :]
        return arr.reshape(num_rx, -1)

    tau_all   = _squeeze_tx(tau_np) * 1e9    # s → ns
    phi_r_all = _squeeze_tx(phi_r_np)
    phi_t_all = _squeeze_tx(phi_t_np)

    # [Intg] 고도각(zenith) + 경로별 LoS (있을 때만; batch 단독 모드면 None)
    theta_r_np = np.asarray(getattr(raw_data, "theta_r", np.array([])))
    theta_t_np = np.asarray(getattr(raw_data, "theta_t", np.array([])))
    theta_r_all = _squeeze_tx(theta_r_np) if theta_r_np.size else None
    theta_t_all = _squeeze_tx(theta_t_np) if theta_t_np.size else None
    path_los_np = np.asarray(getattr(raw_data, "path_los", np.array([])), dtype=bool)
    path_los_all = path_los_np if path_los_np.size else None

    return a_c, pow_path, tau_all, phi_r_all, phi_t_all, theta_r_all, theta_t_all, path_los_all


def _process_single_rx(
    rx_idx: int,
    a_c: np.ndarray,
    pow_path: np.ndarray,
    tau_all: np.ndarray,
    phi_r_all: np.ndarray,
    phi_t_all: np.ndarray,
    config: RT_Config,
    theta_r_all: np.ndarray | None = None,
    theta_t_all: np.ndarray | None = None,
    path_los_all: np.ndarray | None = None,
) -> dict:
    """
    단일 RX에 대해 유효 경로 필터링, RSRP, 공간 공분산 행렬을 계산한다.

    공분산은 Sionna가 계산한 '실제 안테나별 복소 채널계수'(a_c)로부터
    경로별 외적을 합산하여 구한다 (이상화된 ULA 스티어링 벡터를 쓰지 않음).
        A_p = a_c[rx, :, :, p]  (num_rx_ant × num_tx_ant)
        R_RX = Σ_p A_p A_pᴴ   (num_rx_ant × num_rx_ant)
        R_TX = Σ_p A_pᴴ A_p   (num_tx_ant × num_tx_ant)

    Returns:
        dict: tau, power_dbm, aoa_azimuth, aod_azimuth, total_rsrp_dbm, R_TX, R_RX
              (Dead Zone이면 빈 배열과 -inf RSRP)
    """
    num_rx_ant = a_c.shape[1]
    num_tx_ant = a_c.shape[2]

    target_pow   = pow_path[rx_idx]
    target_tau   = tau_all[rx_idx]
    target_phi_r = phi_r_all[rx_idx]
    target_phi_t = phi_t_all[rx_idx]

    valid_mask = target_pow > config.threshold_watt
    val_pow    = target_pow[valid_mask]
    val_tau    = target_tau[valid_mask]
    val_phi_r  = target_phi_r[valid_mask]
    val_phi_t  = target_phi_t[valid_mask]

    # [Intg] 고도각 + 경로별 LoS (있을 때만)
    n_path = target_tau.shape[0]
    if theta_r_all is not None:
        vtr = np.asarray(theta_r_all[rx_idx]).reshape(-1)[:n_path][valid_mask]
        vtt = np.asarray(theta_t_all[rx_idx]).reshape(-1)[:n_path][valid_mask]
    else:
        vtr = vtt = None
    if path_los_all is not None:
        vlos = np.asarray(path_los_all[rx_idx], dtype=bool).reshape(-1)[:n_path][valid_mask]
    else:
        vlos = None

    if val_pow.size == 0:
        return {
            "tau": np.array([]),
            "power_dbm": np.array([]),
            "aoa_azimuth": np.array([]),
            "aod_azimuth": np.array([]),
            "theta_r_deg": np.array([]),
            "theta_t_deg": np.array([]),
            "los_flag": np.array([], dtype=np.int32),
            "total_rsrp_dbm": -np.inf,
            "R_TX": np.zeros((num_tx_ant, num_tx_ant), dtype=np.complex128),
            "R_RX": np.zeros((num_rx_ant, num_rx_ant), dtype=np.complex128),
        }

    rsrp_dbm = 10 * np.log10(np.sum(val_pow) + 1e-15) + config.power_offset

    # 유효 경로의 안테나별 복소 채널 (R, T, n_valid)
    A = a_c[rx_idx][:, :, valid_mask]
    # 경로 합산 공간 공분산
    R_RX = np.einsum('ikp,jkp->ij', A, np.conj(A))   # (R, R)
    R_TX = np.einsum('kip,kjp->ij', np.conj(A), A)   # (T, T)

    n_valid = int(val_pow.size)
    return {
        "tau": val_tau,
        "power_dbm": 10 * np.log10(val_pow + 1e-15) + config.power_offset,
        "aoa_azimuth": val_phi_r * (180 / np.pi),
        "aod_azimuth": val_phi_t * (180 / np.pi),
        # [Intg] 고도각(deg) + 경로별 LoS(int). theta 없으면 0 으로 채움.
        "theta_r_deg": (vtr * (180 / np.pi)) if vtr is not None else np.zeros(n_valid),
        "theta_t_deg": (vtt * (180 / np.pi)) if vtt is not None else np.zeros(n_valid),
        "los_flag": (vlos.astype(np.int32) if vlos is not None else np.zeros(n_valid, np.int32)),
        "total_rsrp_dbm": rsrp_dbm,
        "R_TX": R_TX,
        "R_RX": R_RX,
    }


def run(raw_data: RawRayData, config: RT_Config) -> SimulationResult:
    """
    모든 RX에 대해 후처리를 수행하고,
    target_rx_index에 해당하는 RX의 결과를 SimulationResult로 반환한다.

    Args:
        raw_data: RawRayData (RT_Agent 출력)
        config  : RT_Config 객체

    Returns:
        SimulationResult: target_rx_index RX의 결과 (시각화용)
        (전체 RX 결과는 save_output1()으로 별도 저장)
    """
    print("=" * 60)
    print("🧮 [PostProcess_Agent] 전체 RX RSRP 및 공분산 행렬 계산 시작")
    print("=" * 60)

    start = time.time()

    num_rx = len(config.rx_positions)

    # ── 안테나 차원이 보존된 CIR 추출 ─────────────────────────
    try:
        a_c, pow_all, tau_all, phi_r_all, phi_t_all, theta_r_all, theta_t_all, path_los_all = _extract_mimo_cir(raw_data, num_rx)
    except Exception as e:
        print(f"   데이터 슬라이싱 실패: {e}")
        print("   → 전체 Dead Zone으로 처리합니다.")
        return SimulationResult()

    # RX별 LoS 정보 (rt_agent가 계산, 없으면 전부 False)
    los_arr = np.asarray(getattr(raw_data, "los", np.array([])), dtype=bool)

    # ── 모든 RX 처리 ──────────────────────────────────────────
    all_results = []
    dead_zone_count = 0
    for i in range(num_rx):
        res = _process_single_rx(i, a_c, pow_all, tau_all, phi_r_all, phi_t_all, config,
                                 theta_r_all, theta_t_all, path_los_all)
        res["los"] = bool(los_arr[i]) if i < los_arr.shape[0] else False
        all_results.append(res)
        if res["total_rsrp_dbm"] == -np.inf:
            dead_zone_count += 1

    valid_count = num_rx - dead_zone_count
    rsrp_values = [r["total_rsrp_dbm"] for r in all_results if r["total_rsrp_dbm"] != -np.inf]
    if rsrp_values:
        print(f"   📊 전체 {num_rx}개 RX 처리 완료")
        print(f"      유효 RX: {valid_count}개 | Dead Zone: {dead_zone_count}개")
        print(f"      RSRP 범위: {min(rsrp_values):.1f} ~ {max(rsrp_values):.1f} dBm")
    else:
        print(f"   모든 RX가 Dead Zone입니다.")

    # ── target_rx_index 결과 출력 ─────────────────────────────
    target_res = all_results[config.target_rx_index]
    if target_res["total_rsrp_dbm"] != -np.inf:
        print(f"   📍 Target RX {config.target_rx_index}: "
              f"유효 경로 {len(target_res['tau'])}개 | "
              f"RSRP: {target_res['total_rsrp_dbm']:.2f} dBm")
    else:
        print(f"   📍 Target RX {config.target_rx_index}: Dead Zone")

    # ── 전체 결과를 인스턴스 변수로 보관 (save_output1에서 사용) ──
    run._all_results = all_results

    print(f"\n✅ [PostProcess_Agent] 완료 ({time.time()-start:.2f}s)")

    # target RX 결과를 SimulationResult로 반환 (시각화용)
    r = target_res
    return SimulationResult(
        tau=r["tau"],
        power_dbm=r["power_dbm"],
        aoa_azimuth=r["aoa_azimuth"],
        aod_azimuth=r["aod_azimuth"],
        total_rsrp_dbm=r["total_rsrp_dbm"],
        R_TX=r["R_TX"],
        R_RX=r["R_RX"],
    )


# ============================================================
# Batch 모드 전용 함수
# ============================================================

# 배치 누적 버퍼 (모듈 레벨 상태)
_accumulated_results: list = []


def reset_batch():
    """배치 누적 버퍼를 초기화한다. 배치 루프 시작 전에 호출."""
    global _accumulated_results
    _accumulated_results = []


def run_batch(raw_data: RawRayData, batch_rx_positions: list,
              config: RT_Config, batch_idx: int = -1, quiet: bool = False) -> dict:
    """
    단일 배치의 raw_data를 후처리하여 누적 버퍼에 append한다.

    Args:
        raw_data           : 해당 배치의 RawRayData (RT_Agent 출력)
        batch_rx_positions : 해당 배치의 RX (x, y) 좌표 목록
        config             : RT_Config (threshold_watt, num_tx_ant 등 참조)
        batch_idx          : 로그 출력용 배치 번호 (0-based)
        quiet              : True이면 배치 로그를 출력하지 않는다 (진행바와 충돌 방지).

    Returns:
        dict: 진행바 표시용 통계
              {num_rx, valid, dead, rsrp_min, rsrp_max, accumulated}
    """
    global _accumulated_results

    num_rx = len(batch_rx_positions)

    # RX별 LoS 정보 (해당 배치 기준, 없으면 전부 False)
    los_arr = np.asarray(getattr(raw_data, "los", np.array([])), dtype=bool)

    try:
        a_c, pow_all, tau_all, phi_r_all, phi_t_all, theta_r_all, theta_t_all, path_los_all = _extract_mimo_cir(raw_data, num_rx)
    except Exception as e:
        # 슬라이싱 실패는 quiet 여부와 무관하게 항상 알림 (비정상 상황)
        print(f"\n   ⚠️  [PostProcess] 배치 {batch_idx} 슬라이싱 실패: {e} → Dead Zone으로 채움")
        for i in range(num_rx):
            _accumulated_results.append({
                "tau": np.array([]), "power_dbm": np.array([]),
                "aoa_azimuth": np.array([]), "aod_azimuth": np.array([]),
                "total_rsrp_dbm": -np.inf,
                "R_TX": np.zeros((config.num_tx_ant, config.num_tx_ant), dtype=np.complex128),
                "R_RX": np.zeros((config.num_rx_ant, config.num_rx_ant), dtype=np.complex128),
                "los": bool(los_arr[i]) if i < los_arr.shape[0] else False,
            })
        return {"num_rx": num_rx, "valid": 0, "dead": num_rx,
                "rsrp_min": None, "rsrp_max": None,
                "accumulated": len(_accumulated_results)}

    dead_count = 0
    for i in range(num_rx):
        res = _process_single_rx(i, a_c, pow_all, tau_all, phi_r_all, phi_t_all, config,
                                 theta_r_all, theta_t_all, path_los_all)
        res["los"] = bool(los_arr[i]) if i < los_arr.shape[0] else False
        _accumulated_results.append(res)
        if res["total_rsrp_dbm"] == -np.inf:
            dead_count += 1

    valid_count = num_rx - dead_count
    rsrp_vals = [r["total_rsrp_dbm"] for r in _accumulated_results[-num_rx:]
                 if r["total_rsrp_dbm"] != -np.inf]
    if not quiet:
        rsrp_range = (f"{min(rsrp_vals):.1f} ~ {max(rsrp_vals):.1f} dBm"
                      if rsrp_vals else "N/A")
        print(f"      [PostProcess] 배치 {batch_idx}: {num_rx}개 RX 처리 "
              f"(유효 {valid_count} / Dead {dead_count}) | RSRP {rsrp_range} "
              f"| 누적 {len(_accumulated_results)}개")

    return {
        "num_rx": num_rx,
        "valid": valid_count,
        "dead": dead_count,
        "rsrp_min": (min(rsrp_vals) if rsrp_vals else None),
        "rsrp_max": (max(rsrp_vals) if rsrp_vals else None),
        "accumulated": len(_accumulated_results),
    }


def finalize_batch(config: RT_Config) -> SimulationResult:
    """
    배치 루프 종료 후 누적 결과를 확정하고 target RX의 SimulationResult를 반환한다.

    run._all_results 에 전체 결과를 저장하므로 save_output1()을 그대로 사용 가능.

    Args:
        config: RT_Config (target_rx_index 참조)

    Returns:
        SimulationResult: target_rx_index RX의 결과 (시각화용)
    """
    global _accumulated_results

    all_results = list(_accumulated_results)
    num_rx = len(all_results)

    if num_rx == 0:
        print("   ⚠️  [PostProcess] 누적 결과 없음 → 빈 SimulationResult 반환")
        return SimulationResult()

    # run._all_results 에 저장 (save_output1이 참조)
    run._all_results = all_results

    rsrp_vals = [r["total_rsrp_dbm"] for r in all_results if r["total_rsrp_dbm"] != -np.inf]
    dead_count = num_rx - len(rsrp_vals)

    print(f"\n   📊 [PostProcess] 전체 배치 누적 완료: {num_rx}개 RX")
    print(f"      유효 RX: {len(rsrp_vals)}개 | Dead Zone: {dead_count}개")
    if rsrp_vals:
        print(f"      RSRP 범위: {min(rsrp_vals):.1f} ~ {max(rsrp_vals):.1f} dBm")

    # target RX 결과
    target_idx = min(config.target_rx_index, num_rx - 1)
    target_res = all_results[target_idx]
    if target_res["total_rsrp_dbm"] != -np.inf:
        print(f"   📍 Target RX {target_idx}: "
              f"유효 경로 {len(target_res['tau'])}개 | "
              f"RSRP: {target_res['total_rsrp_dbm']:.2f} dBm")
    else:
        print(f"   📍 Target RX {target_idx}: Dead Zone")

    r = target_res
    return SimulationResult(
        tau=r["tau"],
        power_dbm=r["power_dbm"],
        aoa_azimuth=r["aoa_azimuth"],
        aod_azimuth=r["aod_azimuth"],
        total_rsrp_dbm=r["total_rsrp_dbm"],
        R_TX=r["R_TX"],
        R_RX=r["R_RX"],
    )


def save_output1(result: SimulationResult, output_dir: str,
                 map_title: str = "map", config: RT_Config = None,
                 rx_positions_3d: list = None) -> str:
    """
    모든 RX의 결과를 .npz 파일로 저장한다 (Output 1 — Channel Agent용).

    저장 구조:
      rsrp_all     : (num_rx,) — 각 RX의 총 RSRP (dBm), Dead Zone은 -inf
      los_all      : (num_rx,) — 각 RX의 LoS(직시) 경로 존재 여부 (bool)
      tau_rx{i}    : 유효 경로 지연 (ns)
      power_rx{i}  : 경로별 수신 전력 (dBm)
      aoa_rx{i}    : 방위각 / AoA (degree)
      aod_rx{i}    : 송신 방위각 / AoD (degree)
      R_TX_rx{i}   : TX 공분산 행렬
      R_RX_rx{i}   : RX 공분산 행렬
      rx_positions : (num_rx, 3) — 각 RX의 (x, y, z) (rx_positions_3d 미제공 시 (num_rx,2) 폴백)
      target_rx_index : 분석 대상 RX 인덱스
    """
    os.makedirs(output_dir, exist_ok=True)
    safe_title = "".join(c if c.isalnum() or c in "-_" else "_" for c in map_title)
    filepath = os.path.join(output_dir, f"channel_data_{safe_title}.npz")

    # run()에서 저장해둔 전체 결과 사용
    all_results = getattr(run, '_all_results', None)

    if all_results is None:
        # fallback: target RX 결과만 저장
        np.savez(
            filepath,
            tau=result.tau,
            power_dbm=result.power_dbm,
            aoa_azimuth=result.aoa_azimuth,
            aod_azimuth=result.aod_azimuth,
            total_rsrp_dbm=np.array([result.total_rsrp_dbm]),
            R_TX=result.R_TX,
            R_RX=result.R_RX
        )
    else:
        num_rx = len(all_results)
        save_dict = {}

        # 전체 RSRP 배열
        save_dict["rsrp_all"] = np.array([r["total_rsrp_dbm"] for r in all_results])

        # 전체 LoS 배열 (RX별 직시 경로 존재 여부, bool)
        save_dict["los_all"] = np.array([bool(r.get("los", False)) for r in all_results], dtype=bool)

        # 각 RX별 데이터
        for i, r in enumerate(all_results):
            save_dict[f"tau_rx{i}"] = r["tau"]
            save_dict[f"power_rx{i}"] = r["power_dbm"]
            save_dict[f"aoa_rx{i}"] = r["aoa_azimuth"]
            save_dict[f"aod_rx{i}"] = r["aod_azimuth"]
            save_dict[f"R_TX_rx{i}"] = r["R_TX"]
            save_dict[f"R_RX_rx{i}"] = r["R_RX"]

        # 메타데이터
        if config is not None:
            if rx_positions_3d is not None and len(rx_positions_3d) > 0:
                save_dict["rx_positions"] = np.array(rx_positions_3d, dtype=np.float64)  # (num_rx, 3)
            else:
                save_dict["rx_positions"] = np.array(config.rx_positions)  # (num_rx, 2) 폴백
            save_dict["target_rx_index"] = np.array([config.target_rx_index])

        np.savez(filepath, **save_dict)

    print(f"   💾 [Output 1] Channel Agent용 데이터 저장: {filepath}")
    if all_results:
        print(f"      포함된 RX 수: {len(all_results)}개")
    return filepath


def save_output1_multi(per_tx_all_results: list, output_dir: str,
                       map_title: str = "map", config: RT_Config = None,
                       rx_positions_3d: list = None,
                       rx_valid_mask=None) -> str:
    """
    Multi-TX 결과를 하나의 .npz로 저장한다 (Output 1 — Channel Agent용).

    data 인덱싱: [index_TX, index_RX, {통신 데이터}]

    Args:
        per_tx_all_results : list[num_tx] of (list[num_rx] of per-RX dict)
                             각 dict = {tau, power_dbm, aoa_azimuth, aod_azimuth,
                                        total_rsrp_dbm, R_TX, R_RX, los}
        output_dir         : 저장 디렉토리
        map_title          : 맵 이름 (파일명용)
        config             : RT_Config (tx_positions/rx_positions/target 인덱스)

    저장 구조:
      rsrp_all          : (num_tx, num_rx)        — 각 (TX, RX)의 총 RSRP (dBm), Dead=-inf
      los_all           : (num_tx, num_rx) bool   — 각 (TX, RX)의 LoS 경로 존재 여부
      tau_tx{t}_rx{i}   : RX i의 유효 경로 지연 (ns)   [TX t]
      power_tx{t}_rx{i} : 경로별 수신 전력 (dBm)        [TX t]
      aoa_tx{t}_rx{i}   : 방위각 / AoA (degree)               [TX t]
      aod_tx{t}_rx{i}   : 송신 방위각 / AoD (degree)          [TX t]
      R_TX_tx{t}_rx{i}  : TX 공분산 행렬               [TX t]
      R_RX_tx{t}_rx{i}  : RX 공분산 행렬               [TX t]
      tx_positions      : (num_tx, 3)              — 각 TX의 (x, y, z)
      rx_positions      : (num_rx, 3)              — 각 RX의 (x, y, z)  ← 지면고도+rx_height
                          (rx_positions_3d 미제공 시에만 (num_rx, 2) (x, y) 폴백)
      target_tx_index   : scalar
      target_rx_index   : scalar
    """
    os.makedirs(output_dir, exist_ok=True)
    safe_title = "".join(c if c.isalnum() or c in "-_" else "_" for c in map_title)
    filepath = os.path.join(output_dir, f"channel_data_{safe_title}.npz")

    num_tx = len(per_tx_all_results)
    num_rx = max((len(r) for r in per_tx_all_results), default=0)

    save_dict = {}

    rsrp_all = np.full((num_tx, num_rx), -np.inf, dtype=np.float64)
    los_all = np.zeros((num_tx, num_rx), dtype=bool)

    for t, all_results in enumerate(per_tx_all_results):
        for i, r in enumerate(all_results):
            rsrp_all[t, i] = r["total_rsrp_dbm"]
            los_all[t, i] = bool(r.get("los", False))
            save_dict[f"tau_tx{t}_rx{i}"]   = r["tau"]
            save_dict[f"power_tx{t}_rx{i}"] = r["power_dbm"]
            save_dict[f"aoa_tx{t}_rx{i}"]   = r["aoa_azimuth"]
            save_dict[f"aod_tx{t}_rx{i}"]   = r["aod_azimuth"]
            save_dict[f"R_TX_tx{t}_rx{i}"]  = r["R_TX"]
            save_dict[f"R_RX_tx{t}_rx{i}"]  = r["R_RX"]

    save_dict["rsrp_all"] = rsrp_all
    save_dict["los_all"]  = los_all

    # 비파괴 RX 유효 마스크 (0=valid 1=dead 2=rt_fail). intg 엔진은 dead/rt_fail 구분 마스크를
    # 전달; batch 엔진은 미전달 → 모든 TX RSRP=-inf 인 RX 만 dead 로 자동 산출.
    if rx_valid_mask is not None:
        save_dict["rx_valid_mask"] = np.asarray(rx_valid_mask).astype(np.int8).reshape(-1)
    else:
        _dead = np.all(~np.isfinite(rsrp_all), axis=0)   # (num_rx,) 모든 TX dead
        save_dict["rx_valid_mask"] = np.where(_dead, 1, 0).astype(np.int8)

    if config is not None:
        save_dict["tx_positions"]    = np.array(config.tx_positions, dtype=np.float64)
        # RX 좌표: 3D(x,y,z) 우선 저장. rx_positions_3d(지면고도+rx_height, 시뮬에 실제 사용한 좌표)이
        # 전달되면 그대로, 없으면 config.rx_positions(2D x,y)로 폴백. (2026-06-20: 높이 z 저장 추가)
        if rx_positions_3d is not None and len(rx_positions_3d) > 0:
            save_dict["rx_positions"] = np.array(rx_positions_3d, dtype=np.float64)   # (num_rx, 3)
        else:
            save_dict["rx_positions"] = np.array(config.rx_positions, dtype=np.float64)  # (num_rx, 2) 폴백
        save_dict["target_tx_index"] = np.array([config.target_tx_index])
        save_dict["target_rx_index"] = np.array([config.target_rx_index])
        # 안테나 포트 수 메타데이터 (공분산 행렬 차원)
        save_dict["num_tx_ant"] = np.array([config.num_tx_ant])
        save_dict["num_rx_ant"] = np.array([config.num_rx_ant])

    # MIMO 공분산(예: 16×16)을 RX마다 저장하면 용량이 커지므로 압축 저장한다.
    np.savez_compressed(filepath, **save_dict)

    _nt = config.num_tx_ant if config else "?"
    _nr = config.num_rx_ant if config else "?"
    print(f"   💾 [Output 1] Multi-TX 채널 데이터 저장: {filepath}")
    print(f"      텐서 인덱싱: [TX={num_tx}, RX={num_rx}, {{통신 데이터}}] "
          f"| 공분산 R_TX({_nt}×{_nt}), R_RX({_nr}×{_nr})")
    return filepath


def save_output1_multi_streaming(per_tx_all_results: list, output_dir: str,
                                 map_title: str = "map", config: RT_Config = None,
                                 rx_positions_3d: list = None,
                                 rx_valid_mask=None) -> str:
    """save_output1_multi 와 '동일한' channel_data_*.npz 를 만들되, save_dict 를 통째로
    메모리에 안 들고 각 배열을 zip 에 '하나씩' 흘려 쓴다 (대용량 RX OOM/과다메모리 방지, 2026-07-14).
    출력 키/shape/dtype/값은 save_output1_multi 와 동일."""
    import zipfile
    from numpy.lib import format as _npy_fmt

    os.makedirs(output_dir, exist_ok=True)
    safe_title = "".join(c if c.isalnum() or c in "-_" else "_" for c in map_title)
    filepath = os.path.join(output_dir, f"channel_data_{safe_title}.npz")

    num_tx = len(per_tx_all_results)
    num_rx = max((len(r) for r in per_tx_all_results), default=0)

    def _write(zf, key, arr):
        arr = np.asarray(arr)
        if arr.ndim > 0 and not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)
        with zf.open(key + ".npy", "w", force_zip64=True) as f:
            _npy_fmt.write_array(f, arr, allow_pickle=False)

    rsrp_all = np.full((num_tx, num_rx), -np.inf, dtype=np.float64)
    los_all = np.zeros((num_tx, num_rx), dtype=bool)

    tmp = filepath + ".tmp"
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        # per-(t,i) 배열: 하나씩 기록 후 해제 (14M 개 배열을 dict 에 안 쌓음)
        for t, all_results in enumerate(per_tx_all_results):
            for i, r in enumerate(all_results):
                rsrp_all[t, i] = r["total_rsrp_dbm"]
                los_all[t, i] = bool(r.get("los", False))
                _write(zf, f"tau_tx{t}_rx{i}",   r["tau"])
                _write(zf, f"power_tx{t}_rx{i}", r["power_dbm"])
                _write(zf, f"aoa_tx{t}_rx{i}",   r["aoa_azimuth"])
                _write(zf, f"aod_tx{t}_rx{i}",   r["aod_azimuth"])
                _write(zf, f"R_TX_tx{t}_rx{i}",  r["R_TX"])
                _write(zf, f"R_RX_tx{t}_rx{i}",  r["R_RX"])

        _write(zf, "rsrp_all", rsrp_all)
        _write(zf, "los_all", los_all)

        if rx_valid_mask is not None:
            _write(zf, "rx_valid_mask", np.asarray(rx_valid_mask).astype(np.int8).reshape(-1))
        else:
            _dead = np.all(~np.isfinite(rsrp_all), axis=0)
            _write(zf, "rx_valid_mask", np.where(_dead, 1, 0).astype(np.int8))

        if config is not None:
            _write(zf, "tx_positions", np.array(config.tx_positions, dtype=np.float64))
            if rx_positions_3d is not None and len(rx_positions_3d) > 0:
                _write(zf, "rx_positions", np.array(rx_positions_3d, dtype=np.float64))
            else:
                _write(zf, "rx_positions", np.array(config.rx_positions, dtype=np.float64))
            _write(zf, "target_tx_index", np.array([config.target_tx_index]))
            _write(zf, "target_rx_index", np.array([config.target_rx_index]))
            _write(zf, "num_tx_ant", np.array([config.num_tx_ant]))
            _write(zf, "num_rx_ant", np.array([config.num_rx_ant]))

    os.replace(tmp, filepath)
    return filepath
