"""derived/common.py — 도출 metric 공용 유틸.

- P1B NPZ 로드 (`Area{X}_{freq}GHz_Rays_ALL_RXs.npz` 또는 P1B 필터 결과).
- KST 타임스탬프
- 결과 파일 경로 생성

P1A/P1B NPZ 키 (251218 표준):
- tau, power, theta_r_deg, theta_t_deg, phi_r_deg, phi_t_deg : shape (num_rx, 1, num_tx, 1, max_rays)
- source_path_idx, los_nlos_flag, counts
- area_index, frequency_ghz, num_rx, rx_indices
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Any

import numpy as np


KEY_TAU = "tau"
KEY_POWER = "power"
KEY_THETA_R = "theta_r_deg"
KEY_THETA_T = "theta_t_deg"
KEY_PHI_R = "phi_r_deg"
KEY_PHI_T = "phi_t_deg"
KEY_COUNTS = "counts"
KEY_SRC_PATH = "source_path_idx"
KEY_LOS_NLOS = "los_nlos_flag"

RAY_PARAM_KEYS = (
    KEY_TAU, KEY_POWER, KEY_THETA_R, KEY_THETA_T, KEY_PHI_R, KEY_PHI_T,
)


def kst_timestamp() -> str:
    """KST 기준 YYMMDD_HHMMSS."""

    kst = _dt.timezone(_dt.timedelta(hours=9))
    return _dt.datetime.now(kst).strftime("%y%m%d_%H%M%S")


def load_p1b_npz(npz_path: Path) -> dict[str, Any]:
    """P1A/P1B Rays NPZ 로드 → numpy dict.

    Returns
    -------
    dict
        - rays: dict of arrays (tau, power, theta_*, phi_*, counts, ...)
        - meta: {"area_index", "frequency_ghz", "num_rx", "rx_indices"}
        - shape: tuple (num_rx, 1, num_tx, 1, max_rays) inferred from tau
    """

    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ not found: {npz_path}")

    raw = dict(np.load(npz_path, allow_pickle=False))

    rays = {k: raw[k] for k in raw if k in RAY_PARAM_KEYS or k in (KEY_COUNTS, KEY_SRC_PATH, KEY_LOS_NLOS)}
    meta = {
        "area_index": int(raw.get("area_index", np.array(-1)).item()) if "area_index" in raw else -1,
        "frequency_ghz": float(raw.get("frequency_ghz", np.array(0.0)).item()) if "frequency_ghz" in raw else 0.0,
        "num_rx": int(raw.get("num_rx", np.array(0)).item()) if "num_rx" in raw else int(rays[KEY_TAU].shape[0]),
        "rx_indices": raw["rx_indices"].tolist() if "rx_indices" in raw else list(range(rays[KEY_TAU].shape[0])),
    }
    return {"rays": rays, "meta": meta, "shape": tuple(rays[KEY_TAU].shape)}


def iter_rays_per_rx(rays: dict[str, np.ndarray]):
    """RX 단위로 (tau, power, theta_r, phi_r, theta_t, phi_t)를 yield.

    P1A NPZ는 [num_rx, batch=1, num_tx=1, cluster=1, num_rays] shape.
    """

    tau = rays[KEY_TAU]                # (num_rx, 1, 1, 1, max_rays)
    power = rays[KEY_POWER]
    theta_r = rays[KEY_THETA_R]
    theta_t = rays[KEY_THETA_T]
    phi_r = rays[KEY_PHI_R]
    phi_t = rays[KEY_PHI_T]
    counts = rays.get(KEY_COUNTS, None)

    num_rx = tau.shape[0]
    for rx_idx in range(num_rx):
        # squeeze 모든 형식적 dim
        def _sq(arr):
            return arr[rx_idx].reshape(-1)  # (max_rays,)

        tau_r = _sq(tau)
        power_r = _sq(power)
        # 0인 ray는 zero-padding이므로 필터.
        if counts is not None:
            cnt = int(counts[rx_idx].reshape(-1)[0]) if counts[rx_idx].size > 0 else (power_r > 0).sum()
            cnt = max(cnt, 1)
            cnt = min(cnt, tau_r.size)
            mask = np.zeros_like(tau_r, dtype=bool)
            mask[:cnt] = True
        else:
            mask = power_r > 0

        yield {
            "rx_idx": rx_idx,
            "mask": mask,
            "tau": tau_r,
            "power": power_r,
            "theta_r": _sq(theta_r),
            "phi_r": _sq(phi_r),
            "theta_t": _sq(theta_t),
            "phi_t": _sq(phi_t),
        }
