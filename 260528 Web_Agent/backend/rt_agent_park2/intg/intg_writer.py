"""intg_writer.py — m5 후처리 결과(per_tx_all_results) → 단일 superset NPZ 빌더.

batch 코어(랜덤배치/공분산)가 만든 per-(TX,RX) 결과 dict 들을 받아:
  - path-level (batch 단위: tau[ns], power[dBm], 각도[deg]) 를 그대로 스택
  - 각 path 를 TR38.901 RayGen(raygen_tr38901) 으로 ray 확장 → ray-level (P1A 단위: tau[s], power[linear])
  - R_TX/R_RX 공분산(유일 출처) + 집계 + rx_valid_mask(비파괴) 스택
→ superset dict 반환 (np.savez_compressed 로 저장).

단위 규약 (reshaper 가 native 로 받도록):
  - path_* : ns / dBm / deg  (to_batch_channel = 거의 항등)
  - ray-level: tau[s], power[linear], 각도[deg]  (to_p1a = P1A 항등)

결과 dict 키 (m5 _process_single_rx 출력): tau(ns), power_dbm, aoa_azimuth(deg),
aod_azimuth(deg), theta_r_deg, theta_t_deg, los_flag, total_rsrp_dbm, R_TX, R_RX, los
"""

from __future__ import annotations

import numpy as np

from .raygen_tr38901 import expand_paths_to_rays, select_top_by_power
from . import superset_schema as S


def _dbm_to_linear(dbm: np.ndarray) -> np.ndarray:
    return np.power(10.0, (np.asarray(dbm, dtype=np.float64) - 30.0) / 10.0)


def build_superset(
    per_tx_all_results: list,
    tx_positions,
    rx_positions_3d,
    frequency_hz: float,
    num_tx_ant: int,
    num_rx_ant: int,
    target_tx_index: int = 0,
    target_rx_index: int = 0,
    rng_seed: int | None = 0,
    max_rays_cap: int | None = None,
    tau_min_threshold: float = S.TAU_MIN_THRESHOLD,
) -> dict:
    """per_tx_all_results(list[T] of list[R] of result dict) → superset dict.

    per_tx_all_results 는 '원래 RX 인덱스' 순서로 정렬돼 있어야 함 (batch_runner 가 복원).
    """
    T = len(per_tx_all_results)
    R = max((len(rt) for rt in per_tx_all_results), default=0)

    # ── 1) 각 (t,i) ray 확장 + path/ray 길이 파악 ───────────────
    ray_cache: dict = {}
    path_counts = np.zeros((T, R), dtype=np.int32)
    ray_counts = np.zeros((T, R), dtype=np.int32)
    for t in range(T):
        results = per_tx_all_results[t]
        for i in range(R):
            r = results[i] if i < len(results) else None
            tau_ns = np.asarray(r["tau"]).reshape(-1) if r else np.array([])
            k = tau_ns.size
            path_counts[t, i] = k
            if k == 0:
                ray_cache[(t, i)] = None
                continue
            # path-level (배열)
            pw_dbm = np.asarray(r["power_dbm"]).reshape(-1)
            phi_r = np.asarray(r["aoa_azimuth"]).reshape(-1)
            phi_t = np.asarray(r["aod_azimuth"]).reshape(-1)
            th_r = np.asarray(r.get("theta_r_deg", np.zeros(k))).reshape(-1)
            th_t = np.asarray(r.get("theta_t_deg", np.zeros(k))).reshape(-1)
            los_f = np.asarray(r.get("los_flag", np.zeros(k, np.int32))).reshape(-1)
            # ray 확장: tau[s], power[linear] 로 변환해 입력
            rng = np.random.default_rng(None if rng_seed is None else int(rng_seed) + t * R + i)
            rays = expand_paths_to_rays(
                path_tau=tau_ns / 1e9, path_power=_dbm_to_linear(pw_dbm),
                path_phi_r=phi_r, path_phi_t=phi_t, path_theta_r=th_r, path_theta_t=th_t,
                path_los_flags=los_f, carrier_frequency=frequency_hz, rng=rng,
            )
            if max_rays_cap is not None:
                rays = select_top_by_power(rays, int(max_rays_cap))
            ray_cache[(t, i)] = rays
            ray_counts[t, i] = len(rays["tau"])

    K = int(path_counts.max()) if path_counts.size else 0
    P = int(ray_counts.max()) if ray_counts.size else 0
    K = max(K, 1); P = max(P, 1)  # 0 방지

    # ── 2) 스택 배열 할당 (동적 K/P) ───────────────────────────
    def zf(shape): return np.zeros(shape, dtype=np.float32)
    sup = {
        "path_tau": zf((T, R, K)), "path_power": zf((T, R, K)),
        "path_phi_r": zf((T, R, K)), "path_phi_t": zf((T, R, K)),
        "path_theta_r": zf((T, R, K)), "path_theta_t": zf((T, R, K)),
        "path_los_flag": np.zeros((T, R, K), np.int32), "path_counts": path_counts,
        "tau": zf((T, R, P)), "power": zf((T, R, P)),
        "phi_r_deg": zf((T, R, P)), "phi_t_deg": zf((T, R, P)),
        "theta_r_deg": zf((T, R, P)), "theta_t_deg": zf((T, R, P)),
        "source_path_idx": np.full((T, R, P), S.PAD_SOURCE_PATH_IDX, np.int32),
        "los_nlos_flag": np.full((T, R, P), S.PAD_LOS_FLAG, np.int32),
        "counts": ray_counts,
        "R_TX": np.zeros((T, R, num_tx_ant, num_tx_ant), np.complex128),
        "R_RX": np.zeros((T, R, num_rx_ant, num_rx_ant), np.complex128),
        "rsrp_all": np.full((T, R), -np.inf, np.float64),
        "los_all": np.zeros((T, R), bool),
    }

    # ── 3) 채우기 ──────────────────────────────────────────────
    for t in range(T):
        results = per_tx_all_results[t]
        for i in range(R):
            r = results[i] if i < len(results) else None
            if r is None:
                continue
            sup["rsrp_all"][t, i] = r.get("total_rsrp_dbm", -np.inf)
            sup["los_all"][t, i] = bool(r.get("los", False))
            R_TX = np.asarray(r.get("R_TX"))
            R_RX = np.asarray(r.get("R_RX"))
            if R_TX.shape == (num_tx_ant, num_tx_ant):
                sup["R_TX"][t, i] = R_TX
            if R_RX.shape == (num_rx_ant, num_rx_ant):
                sup["R_RX"][t, i] = R_RX
            k = int(path_counts[t, i])
            if k > 0:
                sup["path_tau"][t, i, :k] = np.asarray(r["tau"]).reshape(-1)[:k]
                sup["path_power"][t, i, :k] = np.asarray(r["power_dbm"]).reshape(-1)[:k]
                sup["path_phi_r"][t, i, :k] = np.asarray(r["aoa_azimuth"]).reshape(-1)[:k]
                sup["path_phi_t"][t, i, :k] = np.asarray(r["aod_azimuth"]).reshape(-1)[:k]
                sup["path_theta_r"][t, i, :k] = np.asarray(r.get("theta_r_deg", np.zeros(k))).reshape(-1)[:k]
                sup["path_theta_t"][t, i, :k] = np.asarray(r.get("theta_t_deg", np.zeros(k))).reshape(-1)[:k]
                sup["path_los_flag"][t, i, :k] = np.asarray(r.get("los_flag", np.zeros(k, np.int32))).reshape(-1)[:k]
            rays = ray_cache.get((t, i))
            if rays is not None:
                p = len(rays["tau"])
                for key in ("tau", "power", "phi_r_deg", "phi_t_deg", "theta_r_deg", "theta_t_deg"):
                    sup[key][t, i, :p] = rays[key]
                sup["source_path_idx"][t, i, :p] = rays["source_path_idx"]
                sup["los_nlos_flag"][t, i, :p] = rays["los_nlos_flag"]

    # ── 4) rx_valid_mask (비파괴: 0=valid 1=dead 2=rt_fail) ─────
    mask = np.full(R, S.RX_VALID, np.int8)
    for i in range(R):
        any_path = bool((path_counts[:, i] > 0).any())
        if not any_path:
            mask[i] = S.RX_DEAD
            continue
        # rt_fail: 어느 TX 든 음수지연 존재 (path_tau 는 ns 단위; 패딩 0 은 안전).
        # 물리적으로 경로 지연은 양수여야 하므로, 의미있는 음수(<-0.1ns)는 RT 아티팩트로 본다.
        rt_fail = False
        for t in range(T):
            k = int(path_counts[t, i])
            if k > 0 and np.any(sup["path_tau"][t, i, :k] < tau_min_threshold):
                rt_fail = True
                break
        mask[i] = S.RX_RT_FAIL if rt_fail else S.RX_VALID

    sup["rx_valid_mask"] = mask
    sup["tx_positions"] = np.asarray(tx_positions, np.float64).reshape(T, 3)
    sup["rx_positions"] = np.asarray(rx_positions_3d, np.float64).reshape(R, 3)
    sup["num_tx_ant"] = np.array([num_tx_ant], np.int64)
    sup["num_rx_ant"] = np.array([num_rx_ant], np.int64)
    sup["target_tx_index"] = np.array([target_tx_index], np.int64)
    sup["target_rx_index"] = np.array([target_rx_index], np.int64)
    sup["frequency_ghz"] = np.float64(frequency_hz / 1e9)
    sup["max_paths"] = np.array([K], np.int64)
    sup["max_rays"] = np.array([P], np.int64)
    return sup
