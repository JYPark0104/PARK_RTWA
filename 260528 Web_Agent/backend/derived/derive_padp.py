"""derive_padp.py — Power Angular Delay Profile (RX별).

Plan 결정사항:
- (tau, phi, theta, power) → 100 × 72 × 36 bin (4D)
- heatmap PNG는 τ-φ 평면 (theta 축 적분)
- NPZ는 4D 원본 + (tau_bins, phi_bins, theta_bins) 보존
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .common import iter_rays_per_rx, kst_timestamp, load_p1b_npz


N_TAU_BINS = 100
N_PHI_BINS = 72
N_THETA_BINS = 36


def derive_padp(
    npz_path: Path,
    out_dir: Path,
    n_tau: int = N_TAU_BINS,
    n_phi: int = N_PHI_BINS,
    n_theta: int = N_THETA_BINS,
    label: str | None = None,
    max_rx_per_plot: int = 4,
) -> dict:
    """RX별 PADP 4D NPZ + τ-φ 평면 PNG."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_p1b_npz(Path(npz_path))
    rays = data["rays"]
    meta = data["meta"]

    num_rx = data["shape"][0]
    padp = np.zeros((num_rx, n_tau, n_phi, n_theta), dtype=np.float32)
    tau_bins_per_rx = np.zeros((num_rx, n_tau + 1), dtype=np.float32)

    phi_bins = np.linspace(-180.0, 180.0, n_phi + 1)
    theta_bins = np.linspace(0.0, 180.0, n_theta + 1)

    for entry in iter_rays_per_rx(rays):
        rx_idx = entry["rx_idx"]
        tau = entry["tau"][entry["mask"]]
        phi = entry["phi_r"][entry["mask"]]
        theta = entry["theta_r"][entry["mask"]]
        power = entry["power"][entry["mask"]]
        if tau.size == 0:
            continue
        tmin, tmax = float(np.min(tau)), float(np.max(tau))
        if tmax <= tmin:
            tmax = tmin + 1e-9
        tau_bin_edges = np.linspace(tmin, tmax, n_tau + 1)
        tau_bins_per_rx[rx_idx] = tau_bin_edges.astype(np.float32)

        H, _ = np.histogramdd(
            np.stack([tau, phi, theta], axis=-1),
            bins=(tau_bin_edges, phi_bins, theta_bins),
            weights=power,
        )
        padp[rx_idx] = H.astype(np.float32)

    ts = kst_timestamp()
    lbl = f"_{label}" if label else ""
    npz_path_out = out_dir / f"PADP{lbl}_{ts}.npz"
    png_path = out_dir / f"PADP{lbl}_{ts}.png"

    np.savez_compressed(
        npz_path_out,
        padp=padp,
        tau_bins=tau_bins_per_rx,
        phi_bins=phi_bins.astype(np.float32),
        theta_bins=theta_bins.astype(np.float32),
        n_tau=np.int32(n_tau),
        n_phi=np.int32(n_phi),
        n_theta=np.int32(n_theta),
        frequency_ghz=np.float32(meta["frequency_ghz"]),
        area_index=np.int32(meta["area_index"]),
        rx_indices=np.asarray(meta["rx_indices"], dtype=np.int32),
    )

    _plot_padp_tau_phi(png_path, padp, tau_bins_per_rx, phi_bins, meta, max_rx_per_plot)

    return {
        "npz_path": str(npz_path_out),
        "png_path": str(png_path),
        "summary": {
            "num_rx": int(num_rx),
            "shape": [int(n_tau), int(n_phi), int(n_theta)],
            "frequency_ghz": meta["frequency_ghz"],
        },
    }


def _plot_padp_tau_phi(
    out_path: Path,
    padp: np.ndarray,
    tau_bins_per_rx: np.ndarray,
    phi_bins: np.ndarray,
    meta: dict,
    max_rx_per_plot: int,
) -> None:
    """theta 축 적분한 τ-φ heatmap."""

    import math

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_show = min(max_rx_per_plot, padp.shape[0])
    n_cols = min(2, n_show)
    n_rows = max(1, math.ceil(n_show / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), dpi=110, squeeze=False)
    for i in range(n_show):
        ax = axes[i // n_cols][i % n_cols]
        tau_phi = np.sum(padp[i], axis=-1)  # (tau, phi)
        # 정규화 dB
        max_v = tau_phi.max()
        if max_v > 0:
            tau_phi_db = 10 * np.log10(np.maximum(tau_phi / max_v, 1e-6))
        else:
            tau_phi_db = np.full_like(tau_phi, -60.0)
        tau_centers = (tau_bins_per_rx[i, :-1] + tau_bins_per_rx[i, 1:]) / 2.0
        phi_centers = (phi_bins[:-1] + phi_bins[1:]) / 2.0
        # imshow: vertical = tau, horizontal = phi
        im = ax.imshow(
            tau_phi_db.T,
            aspect="auto",
            origin="lower",
            extent=(tau_centers[0] * 1e9, tau_centers[-1] * 1e9, phi_centers[0], phi_centers[-1]),
            cmap="viridis",
            vmin=-60,
            vmax=0,
        )
        ax.set_title(f"RX {i+1}", fontsize=10)
        ax.set_xlabel("delay [ns]", fontsize=9)
        ax.set_ylabel("azimuth [deg]", fontsize=9)
        cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
        cbar.set_label("power [dB, norm]", fontsize=9)
    for j in range(n_show, n_rows * n_cols):
        axes[j // n_cols][j % n_cols].axis("off")

    fig.suptitle(
        f"PADP τ-φ @ {meta.get('frequency_ghz', 0):.2f} GHz (showing {n_show}/{padp.shape[0]} RX)",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
