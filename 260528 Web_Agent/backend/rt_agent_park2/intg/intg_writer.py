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
    cov_max_bytes: float = 6e9,
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
    # 공분산 크기 가드: 대형 안테나(예: 32×32=1024p)면 (T,R,At,At) 단일 배열이 수십~수백 GB →
    # OOM 방지를 위해 cov_max_bytes 초과 시 superset 에서 공분산을 생략한다(비파괴: channel_data
    # per-pair 에는 그대로 남고, P1C/D 는 ray 로부터 공분산을 재계산하므로 하류 영향 없음).
    _tx_cov_bytes = T * R * num_tx_ant * num_tx_ant * 16
    _rx_cov_bytes = T * R * num_rx_ant * num_rx_ant * 16
    include_tx_cov = _tx_cov_bytes <= cov_max_bytes
    include_rx_cov = _rx_cov_bytes <= cov_max_bytes

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
        "rsrp_all": np.full((T, R), -np.inf, np.float64),
        "los_all": np.zeros((T, R), bool),
    }
    if include_tx_cov:
        sup["R_TX"] = np.zeros((T, R, num_tx_ant, num_tx_ant), np.complex128)
    if include_rx_cov:
        sup["R_RX"] = np.zeros((T, R, num_rx_ant, num_rx_ant), np.complex128)
    sup["cov_omitted"] = np.array(
        [0 if include_tx_cov else 1, 0 if include_rx_cov else 1], np.int8)  # [R_TX생략, R_RX생략]

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
            if include_tx_cov and R_TX.shape == (num_tx_ant, num_tx_ant):
                sup["R_TX"][t, i] = R_TX
            if include_rx_cov and R_RX.shape == (num_rx_ant, num_rx_ant):
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
    #   판정(P1B 규약): dead=유효경로 0 / rt_fail=음수지연(path_tau<-0.1ns) 또는
    #   ray_efficiency(=power>0 ray 비율) < RAY_EFFICIENCY_THRESHOLD.
    mask = np.full(R, S.RX_VALID, np.int8)
    for i in range(R):
        any_path = bool((path_counts[:, i] > 0).any())
        if not any_path:
            mask[i] = S.RX_DEAD
            continue
        rt_fail = False
        for t in range(T):
            k = int(path_counts[t, i])
            if k == 0:
                continue
            # (a) 음수지연 (path_tau 는 ns; 패딩 0 은 안전)
            if np.any(sup["path_tau"][t, i, :k] < tau_min_threshold):
                rt_fail = True
                break
            # (b) ray_efficiency: 유효 ray(power>0) / 전체 ray 수 < 임계 → 실패
            p = int(ray_counts[t, i])
            if p > 0:
                valid_rays = int(np.count_nonzero(sup["power"][t, i, :p] > 0))
                if (valid_rays / p) < S.RAY_EFFICIENCY_THRESHOLD:
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


# ═════════════════════════════════════════════════════════════════════════
# 스트리밍 저장 (대용량 RX OOM 방지) — 2026-07-14
#   문제: build_superset 는 7개의 큰 path_* [T,R,K] 배열을 한 dict 에 통째로 올려
#         (130k RX 면 수백 GB) OOM-kill 로 장시간 연산이 날아간다.
#   해결: '작은 배열 dict(build_superset_small)' 와 '큰 path 배열'을 분리하여,
#         큰 배열은 .npz 에 '한 개씩' 흘려 쓴다(peak = 큰 배열 1개 ≈ T·R·K·4B).
#   출력: build_superset()+np.savez_compressed 와 키/shape/dtype/값이 동일.
#
#   ⚠️ build_superset() 와 로직 동기화 유지 필수(pass-1/small/mask 동일).
# ═════════════════════════════════════════════════════════════════════════

# (superset path_* 키, 결과 dict 의 원본 필드, dtype)
_PATH_SPECS = [
    ("path_tau",      "tau",         np.float32),
    ("path_power",    "power_dbm",   np.float32),
    ("path_phi_r",    "aoa_azimuth", np.float32),
    ("path_phi_t",    "aod_azimuth", np.float32),
    ("path_theta_r",  "theta_r_deg", np.float32),
    ("path_theta_t",  "theta_t_deg", np.float32),
    ("path_los_flag", "los_flag",    np.int32),
]
_PATH_DEFAULTABLE = {"theta_r_deg", "theta_t_deg", "los_flag"}  # 없으면 0 으로


def build_superset_small(per_tx_all_results, tx_positions, rx_positions_3d,
                         frequency_hz, num_tx_ant, num_rx_ant,
                         target_tx_index=0, target_rx_index=0, rng_seed=0,
                         max_rays_cap=None, tau_min_threshold=S.TAU_MIN_THRESHOLD,
                         cov_max_bytes=6e9):
    """build_superset 의 '작은 부분'만 계산 (7개 큰 path_* 배열 제외).
    반환 dict 로 P1A 뷰 생성·rx_valid_mask·로그가 가능하다."""
    T = len(per_tx_all_results)
    R = max((len(rt) for rt in per_tx_all_results), default=0)
    ray_cache: dict = {}
    path_counts = np.zeros((T, R), np.int32)
    ray_counts = np.zeros((T, R), np.int32)
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
            pw_dbm = np.asarray(r["power_dbm"]).reshape(-1)
            phi_r = np.asarray(r["aoa_azimuth"]).reshape(-1)
            phi_t = np.asarray(r["aod_azimuth"]).reshape(-1)
            th_r = np.asarray(r.get("theta_r_deg", np.zeros(k))).reshape(-1)
            th_t = np.asarray(r.get("theta_t_deg", np.zeros(k))).reshape(-1)
            los_f = np.asarray(r.get("los_flag", np.zeros(k, np.int32))).reshape(-1)
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

    K = max(int(path_counts.max()) if path_counts.size else 0, 1)
    P = max(int(ray_counts.max()) if ray_counts.size else 0, 1)
    include_tx_cov = (T * R * num_tx_ant * num_tx_ant * 16) <= cov_max_bytes
    include_rx_cov = (T * R * num_rx_ant * num_rx_ant * 16) <= cov_max_bytes

    def zf(shape): return np.zeros(shape, dtype=np.float32)
    small = {
        "path_counts": path_counts,
        "tau": zf((T, R, P)), "power": zf((T, R, P)),
        "phi_r_deg": zf((T, R, P)), "phi_t_deg": zf((T, R, P)),
        "theta_r_deg": zf((T, R, P)), "theta_t_deg": zf((T, R, P)),
        "source_path_idx": np.full((T, R, P), S.PAD_SOURCE_PATH_IDX, np.int32),
        "los_nlos_flag": np.full((T, R, P), S.PAD_LOS_FLAG, np.int32),
        "counts": ray_counts,
        "rsrp_all": np.full((T, R), -np.inf, np.float64),
        "los_all": np.zeros((T, R), bool),
    }
    if include_tx_cov:
        small["R_TX"] = np.zeros((T, R, num_tx_ant, num_tx_ant), np.complex128)
    if include_rx_cov:
        small["R_RX"] = np.zeros((T, R, num_rx_ant, num_rx_ant), np.complex128)
    small["cov_omitted"] = np.array(
        [0 if include_tx_cov else 1, 0 if include_rx_cov else 1], np.int8)

    for t in range(T):
        results = per_tx_all_results[t]
        for i in range(R):
            r = results[i] if i < len(results) else None
            if r is None:
                continue
            small["rsrp_all"][t, i] = r.get("total_rsrp_dbm", -np.inf)
            small["los_all"][t, i] = bool(r.get("los", False))
            R_TX = np.asarray(r.get("R_TX")); R_RX = np.asarray(r.get("R_RX"))
            if include_tx_cov and R_TX.shape == (num_tx_ant, num_tx_ant):
                small["R_TX"][t, i] = R_TX
            if include_rx_cov and R_RX.shape == (num_rx_ant, num_rx_ant):
                small["R_RX"][t, i] = R_RX
            rays = ray_cache.get((t, i))
            if rays is not None:
                p = len(rays["tau"])
                for key in ("tau", "power", "phi_r_deg", "phi_t_deg", "theta_r_deg", "theta_t_deg"):
                    small[key][t, i, :p] = rays[key]
                small["source_path_idx"][t, i, :p] = rays["source_path_idx"]
                small["los_nlos_flag"][t, i, :p] = rays["los_nlos_flag"]

    # rx_valid_mask: path_tau 대신 원본 r["tau"] 로 계산(값 동일) → 큰 배열 불필요
    mask = np.full(R, S.RX_VALID, np.int8)
    for i in range(R):
        if not bool((path_counts[:, i] > 0).any()):
            mask[i] = S.RX_DEAD
            continue
        rt_fail = False
        for t in range(T):
            k = int(path_counts[t, i])
            if k == 0:
                continue
            r = per_tx_all_results[t][i] if i < len(per_tx_all_results[t]) else None
            tau_ns = np.asarray(r["tau"]).reshape(-1)[:k] if r else np.array([])
            if tau_ns.size and np.any(tau_ns < tau_min_threshold):
                rt_fail = True; break
            p = int(ray_counts[t, i])
            if p > 0:
                valid_rays = int(np.count_nonzero(small["power"][t, i, :p] > 0))
                if (valid_rays / p) < S.RAY_EFFICIENCY_THRESHOLD:
                    rt_fail = True; break
        mask[i] = S.RX_RT_FAIL if rt_fail else S.RX_VALID

    small["rx_valid_mask"] = mask
    small["tx_positions"] = np.asarray(tx_positions, np.float64).reshape(T, 3)
    small["rx_positions"] = np.asarray(rx_positions_3d, np.float64).reshape(R, 3)
    small["num_tx_ant"] = np.array([num_tx_ant], np.int64)
    small["num_rx_ant"] = np.array([num_rx_ant], np.int64)
    small["target_tx_index"] = np.array([target_tx_index], np.int64)
    small["target_rx_index"] = np.array([target_rx_index], np.int64)
    small["frequency_ghz"] = np.float64(frequency_hz / 1e9)
    small["max_paths"] = np.array([K], np.int64)
    small["max_rays"] = np.array([P], np.int64)
    return small


def stream_write_superset(out_path, small, per_tx_all_results):
    """small(build_superset_small 결과) + 7개 큰 path 배열을 .npz 로 '한 개씩' 흘려 쓴다.
    peak 메모리 = 큰 배열 1개(≈ T·R·K·4B). np.load 로 읽으면 build_superset+savez 와 동일."""
    import os as _os
    import zipfile
    from numpy.lib import format as _npy_fmt

    path_counts = np.asarray(small["path_counts"])
    T, R = path_counts.shape
    K = int(np.asarray(small["max_paths"]).reshape(-1)[0])

    def _write(zf, key, arr):
        arr = np.asarray(arr)
        if arr.ndim > 0 and not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)   # 0-d 스칼라는 shape () 유지(ascontiguous 는 1-d 로 승격됨)
        with zf.open(key + ".npy", "w", force_zip64=True) as f:
            _npy_fmt.write_array(f, arr, allow_pickle=False)

    tmp = str(out_path) + ".tmp"
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        for k, v in small.items():          # 작은/ray-level 키
            _write(zf, k, v)
        for supkey, rfield, dt in _PATH_SPECS:   # 큰 path_* 키: 1개씩 build→write→free
            big = np.zeros((T, R, K), dt)
            for t in range(T):
                results = per_tx_all_results[t]
                for i in range(R):
                    kk = int(path_counts[t, i])
                    if kk == 0:
                        continue
                    r = results[i] if i < len(results) else None
                    if r is None:
                        continue
                    if rfield in _PATH_DEFAULTABLE:
                        src = r.get(rfield, np.zeros(kk, dt))
                    else:
                        src = r[rfield]
                    big[t, i, :kk] = np.asarray(src).reshape(-1)[:kk]
            _write(zf, supkey, big)
            del big
    _os.replace(tmp, out_path)
