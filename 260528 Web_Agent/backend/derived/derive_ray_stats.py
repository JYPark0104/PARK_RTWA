"""derive_ray_stats.py — Delay spread / K-factor / Angular spread.

Plan 결정사항:
- Delay spread = RMS τ (power-weighted)
- K-factor = LoS power / NLoS power 비 (los_nlos_flag 사용)
- Angular spread = ASA / ASD (circular standard deviation, power-weighted)

CSV 컬럼:
    rx_index, delay_spread_ns, k_factor_db, asa_deg, asd_deg, zsa_deg, zsd_deg
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .common import KEY_LOS_NLOS, iter_rays_per_rx, kst_timestamp, load_p1b_npz


def _circular_std_deg(angles_deg: np.ndarray, weights: np.ndarray) -> float:
    """Power-weighted circular standard deviation in degrees (azimuth용)."""

    if angles_deg.size == 0 or weights.sum() <= 0:
        return float("nan")
    rad = np.deg2rad(angles_deg)
    w = weights / weights.sum()
    sin_mean = np.sum(w * np.sin(rad))
    cos_mean = np.sum(w * np.cos(rad))
    R = np.sqrt(sin_mean ** 2 + cos_mean ** 2)
    R = min(max(R, 1e-12), 1.0)
    return float(np.rad2deg(np.sqrt(-2.0 * np.log(R))))


def _weighted_std_deg(angles_deg: np.ndarray, weights: np.ndarray) -> float:
    """Power-weighted linear standard deviation (zenith용)."""

    if angles_deg.size == 0 or weights.sum() <= 0:
        return float("nan")
    w = weights / weights.sum()
    m = float(np.sum(w * angles_deg))
    return float(np.sqrt(np.sum(w * (angles_deg - m) ** 2)))


def _rms_tau_ns(tau_s: np.ndarray, power: np.ndarray) -> float:
    """Power-weighted RMS delay spread in nanoseconds."""

    if tau_s.size == 0 or power.sum() <= 0:
        return float("nan")
    w = power / power.sum()
    mean_tau = float(np.sum(w * tau_s))
    rms = float(np.sqrt(np.sum(w * (tau_s - mean_tau) ** 2)))
    return rms * 1e9


def derive_ray_stats(
    npz_path: Path,
    out_dir: Path,
    label: str | None = None,
) -> dict:
    """RX별 ray 통계 CSV."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_p1b_npz(Path(npz_path))
    rays = data["rays"]
    meta = data["meta"]
    los_nlos = rays.get(KEY_LOS_NLOS)

    ts = kst_timestamp()
    lbl = f"_{label}" if label else ""
    csv_path = out_dir / f"RayStats{lbl}_{ts}.csv"

    rows: list[dict] = []
    for entry in iter_rays_per_rx(rays):
        rx_idx = entry["rx_idx"]
        tau = entry["tau"][entry["mask"]]
        power = entry["power"][entry["mask"]]
        phi_r = entry["phi_r"][entry["mask"]]
        phi_t = entry["phi_t"][entry["mask"]]
        theta_r = entry["theta_r"][entry["mask"]]
        theta_t = entry["theta_t"][entry["mask"]]

        ds_ns = _rms_tau_ns(tau, power)
        asa = _circular_std_deg(phi_r, power)
        asd = _circular_std_deg(phi_t, power)
        zsa = _weighted_std_deg(theta_r, power)
        zsd = _weighted_std_deg(theta_t, power)

        k_db = float("nan")
        if los_nlos is not None:
            flags = los_nlos[rx_idx].reshape(-1)[: entry["mask"].size][entry["mask"]]
            if flags.size == power.size and flags.size > 0:
                los_p = float(power[flags == 1].sum())
                nlos_p = float(power[flags == 0].sum())
                if nlos_p > 0:
                    k_db = 10.0 * np.log10(max(los_p, 1e-30) / nlos_p)

        rows.append({
            "rx_index": rx_idx + 1,
            "delay_spread_ns": ds_ns,
            "k_factor_db": k_db,
            "asa_deg": asa,
            "asd_deg": asd,
            "zsa_deg": zsa,
            "zsd_deg": zsd,
        })

    with csv_path.open("w", encoding="utf-8") as f:
        f.write("rx_index,delay_spread_ns,k_factor_db,asa_deg,asd_deg,zsa_deg,zsd_deg\n")
        for r in rows:
            f.write(
                f"{r['rx_index']},{r['delay_spread_ns']:.4f},{r['k_factor_db']:.4f},"
                f"{r['asa_deg']:.4f},{r['asd_deg']:.4f},{r['zsa_deg']:.4f},{r['zsd_deg']:.4f}\n"
            )

    return {
        "csv_path": str(csv_path),
        "summary": {
            "num_rx": len(rows),
            "mean_delay_spread_ns": float(np.nanmean([r["delay_spread_ns"] for r in rows])) if rows else float("nan"),
            "mean_k_factor_db": float(np.nanmean([r["k_factor_db"] for r in rows])) if rows else float("nan"),
            "frequency_ghz": meta["frequency_ghz"],
        },
    }
