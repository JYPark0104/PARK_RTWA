"""to_batch_channel.py — superset NPZ → batch channel_data 포맷 (channel_data_*.npz).

RX Inspector / Scenario / 시각화가 기대하는 정확한 키로 변환 → **batch 후처리 무수정**.

channel_data 포맷 (소비자가 읽는 키):
    rsrp_all          (T,R)
    los_all           (T,R) bool
    tx_positions      (T,3)
    rx_positions      (R,3)
    num_tx_ant, num_rx_ant, target_tx_index, target_rx_index   (1,)
    per (t,i):
      tau_tx{t}_rx{i}   (path_count,)   ← path-level (가변길이)
      power_tx{t}_rx{i} (path_count,)
      aoa_tx{t}_rx{i}   (path_count,)   = path_phi_r
      aod_tx{t}_rx{i}   (path_count,)   = path_phi_t
      R_TX_tx{t}_rx{i}  (At,At)
      R_RX_tx{t}_rx{i}  (Ar,Ar)

batch 뷰는 ray-level 이 아니라 **path-level** 을 쓰므로 superset 의 path_* 를 사용한다.
path_counts 로 가변길이(유효 경로 수)만큼 잘라낸다.
"""

from __future__ import annotations

import numpy as np


def superset_to_batch_channel(superset: dict) -> dict:
    """superset dict → channel_data 호환 dict (np.savez_compressed 로 저장)."""
    rsrp_all = np.asarray(superset["rsrp_all"])
    num_tx, num_rx = rsrp_all.shape

    path_tau = np.asarray(superset["path_tau"])      # (T,R,K)
    path_power = np.asarray(superset["path_power"])
    path_phi_r = np.asarray(superset["path_phi_r"])
    path_phi_t = np.asarray(superset["path_phi_t"])
    path_counts = np.asarray(superset["path_counts"])  # (T,R)
    R_TX = superset.get("R_TX")
    R_RX = superset.get("R_RX")
    has_cov = R_TX is not None and R_RX is not None
    if has_cov:
        R_TX = np.asarray(R_TX)
        R_RX = np.asarray(R_RX)

    out: dict = {
        "rsrp_all": rsrp_all,
        "los_all": np.asarray(superset["los_all"]),
        "tx_positions": np.asarray(superset["tx_positions"]),
        "rx_positions": np.asarray(superset["rx_positions"]),
    }
    # 비파괴 RX 유효 마스크 (0=valid 1=dead 2=rt_fail) — 모든 출력에 공통 기록(인덱스 갭 유지)
    if "rx_valid_mask" in superset:
        out["rx_valid_mask"] = np.asarray(superset["rx_valid_mask"]).astype(np.int8)
    for k in ("num_tx_ant", "num_rx_ant", "target_tx_index", "target_rx_index"):
        if k in superset:
            out[k] = np.asarray(superset[k]).reshape(-1).astype(np.int64)

    for t in range(num_tx):
        for i in range(num_rx):
            n = int(path_counts[t, i])
            out[f"tau_tx{t}_rx{i}"] = path_tau[t, i, :n].astype(np.float32)
            out[f"power_tx{t}_rx{i}"] = path_power[t, i, :n].astype(np.float64)
            out[f"aoa_tx{t}_rx{i}"] = path_phi_r[t, i, :n].astype(np.float32)
            out[f"aod_tx{t}_rx{i}"] = path_phi_t[t, i, :n].astype(np.float32)
            if has_cov:
                out[f"R_TX_tx{t}_rx{i}"] = R_TX[t, i]
                out[f"R_RX_tx{t}_rx{i}"] = R_RX[t, i]
    return out
