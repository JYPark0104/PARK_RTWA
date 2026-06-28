"""to_p1a.py — superset NPZ → P1A 호환 NPZ (AreaX_{freq}GHz_Rays_ALL_RXs.npz).

P1B/C/D 가 기대하는 정확한 포맷으로 변환한다 → **P1B/C/D 무수정**.

P1A 포맷 (P1B 가 읽는 키/shape):
    rx_indices       (R,)                      int64   (1-based 라벨)
    area_index       ()                        int64
    frequency_ghz    ()                        float64
    num_rx           ()                        int64
    tau              (R,1,1,1,1,P)             float32
    power            (R,1,1,1,1,P)             float32
    theta_r_deg      (R,1,1,1,1,P)             float32
    theta_t_deg      (R,1,1,1,1,P)             float32
    phi_r_deg        (R,1,1,1,1,P)             float32
    phi_t_deg        (R,1,1,1,1,P)             float32
    source_path_idx  (R,1,1,1,1,P)             int32
    los_nlos_flag    (R,1,1,1,1,P)             int32
    counts           (R,1,1,1)                 int32

통합 NPZ 는 멀티 TX 라, P1A(단일 TX/파일) 로 줄 때는 TX 인덱스 하나를 골라 변환한다.
(P1A/P1B 가 area 당 단일 TX 를 가정하므로, 여기서 tx_index 로 슬라이스.)
"""

from __future__ import annotations

import numpy as np

# P1A ray-level 키: superset 키명과 동일 (phi_r_deg, theta_r_deg, ...)
_RAY_FLOAT_KEYS = ["tau", "power", "theta_r_deg", "theta_t_deg", "phi_r_deg", "phi_t_deg"]
_RAY_INT_KEYS = ["source_path_idx", "los_nlos_flag"]


def superset_to_p1a(superset: dict, tx_index: int = 0, area_index: int = 1) -> dict:
    """superset dict → P1A 호환 dict (np.savez 로 저장하면 됨).

    Parameters
    ----------
    superset : dict   superset NPZ 를 np.load 후 dict 화한 것 (또는 writer 가 만든 dict)
    tx_index : int    어느 TX 를 P1A 단일-TX 파일로 뽑을지
    area_index : int  P1A area_index 라벨
    """
    t = int(tx_index)
    rsrp_all = np.asarray(superset["rsrp_all"])
    num_rx = int(rsrp_all.shape[1])

    out: dict = {}
    out["rx_indices"] = np.arange(1, num_rx + 1, dtype=np.int64)  # 1-based 라벨 (갭 유지)
    out["area_index"] = np.int64(area_index)
    out["frequency_ghz"] = np.float64(np.asarray(superset.get("frequency_ghz", 0.0)).item()
                                      if "frequency_ghz" in superset else 0.0)
    out["num_rx"] = np.int64(num_rx)

    # ray-level: superset[key] (T,R,P) → [t] (R,P) → (R,1,1,1,1,P)
    for key in _RAY_FLOAT_KEYS:
        arr = np.asarray(superset[key])[t]            # (R,P)
        out[key] = arr.reshape(num_rx, 1, 1, 1, 1, -1).astype(np.float32)
    for key in _RAY_INT_KEYS:
        arr = np.asarray(superset[key])[t]            # (R,P)
        out[key] = arr.reshape(num_rx, 1, 1, 1, 1, -1).astype(np.int32)

    # counts: (T,R) → [t] (R,) → (R,1,1,1)
    counts = np.asarray(superset["counts"])[t].reshape(num_rx, 1, 1, 1).astype(np.int32)
    out["counts"] = counts

    # 비파괴 RX 유효 마스크 (0=valid 1=dead 2=rt_fail) — P1C/D 가 valid RX 만 공분산 계산하도록.
    # (P1A 표준 키는 아니지만 additive 라 P1B 의 키 기반 필터에 영향 없음. 인덱스 갭 유지.)
    if "rx_valid_mask" in superset:
        out["rx_valid_mask"] = np.asarray(superset["rx_valid_mask"]).astype(np.int8)
    return out
