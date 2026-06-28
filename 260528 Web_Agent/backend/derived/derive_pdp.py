"""derive_pdp.py — Power Delay Profile (RX별).

Plan 결정사항:
- 각 RX의 (tau, power)를 100개 bin으로 histogram (delay-domain power binning)
- linear interpolation으로 부드럽게 표현
- NPZ + PNG (모든 RX subplot 또는 첫 RX 대표)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .common import iter_rays_per_rx, kst_timestamp, load_p1b_npz


def derive_pdp(
    npz_path: Path,
    out_dir: Path,
    n_bins: int = 100,
    label: str | None = None,
    max_rx_per_plot: int = 16,
) -> dict:
    """RX별 PDP 산출.

    Parameters
    ----------
    npz_path : Path
        P1B Rays NPZ.
    out_dir : Path
        출력 디렉토리.
    n_bins : int
        delay 축 bin 개수 (default 100).
    label : str | None
        파일 이름 추가 라벨.
    max_rx_per_plot : int
        한 PNG에 그릴 최대 RX 개수 (subplot 그리드).
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_p1b_npz(Path(npz_path))
    rays = data["rays"]
    meta = data["meta"]

    num_rx = data["shape"][0]
    pdp_array = np.zeros((num_rx, n_bins), dtype=np.float32)
    tau_bins_array = np.zeros((num_rx, n_bins + 1), dtype=np.float32)
    tau_centers = np.zeros((num_rx, n_bins), dtype=np.float32)

    for entry in iter_rays_per_rx(rays):
        rx_idx = entry["rx_idx"]
        tau = entry["tau"][entry["mask"]]
        power = entry["power"][entry["mask"]]
        if tau.size == 0:
            continue
        tau_min = float(np.min(tau))
        tau_max = float(np.max(tau))
        if tau_max <= tau_min:
            tau_max = tau_min + 1e-9
        bins = np.linspace(tau_min, tau_max, n_bins + 1)
        # power-weighted histogram (총합 power)
        hist, edges = np.histogram(tau, bins=bins, weights=power)
        pdp_array[rx_idx, :] = hist.astype(np.float32)
        tau_bins_array[rx_idx, :] = edges.astype(np.float32)
        tau_centers[rx_idx, :] = ((edges[:-1] + edges[1:]) / 2.0).astype(np.float32)

    ts = kst_timestamp()
    lbl = f"_{label}" if label else ""
    npz_path_out = out_dir / f"PDP{lbl}_{ts}.npz"
    png_path = out_dir / f"PDP{lbl}_{ts}.png"

    np.savez_compressed(
        npz_path_out,
        pdp=pdp_array,
        tau_bins=tau_bins_array,
        tau_centers=tau_centers,
        n_bins=np.int32(n_bins),
        frequency_ghz=np.float32(meta["frequency_ghz"]),
        area_index=np.int32(meta["area_index"]),
        rx_indices=np.asarray(meta["rx_indices"], dtype=np.int32),
    )

    _plot_pdp_grid(png_path, tau_centers, pdp_array, meta, max_rx_per_plot)

    return {
        "npz_path": str(npz_path_out),
        "png_path": str(png_path),
        "summary": {
            "num_rx": int(num_rx),
            "n_bins": int(n_bins),
            "frequency_ghz": meta["frequency_ghz"],
        },
    }


def _plot_pdp_grid(
    out_path: Path,
    tau_centers: np.ndarray,
    pdp_array: np.ndarray,
    meta: dict,
    max_rx_per_plot: int,
) -> None:
    """첫 N개의 RX에 대해 subplot 그리드로 PDP plot."""

    import math

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_show = min(max_rx_per_plot, pdp_array.shape[0])
    n_cols = min(4, n_show)
    n_rows = max(1, math.ceil(n_show / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), dpi=110, squeeze=False)
    for i in range(n_show):
        ax = axes[i // n_cols][i % n_cols]
        pdp = pdp_array[i]
        if pdp.max() > 0:
            pdp_db = 10 * np.log10(np.maximum(pdp / pdp.max(), 1e-6))
        else:
            pdp_db = np.full_like(pdp, -60.0)
        ax.plot(tau_centers[i] * 1e9, pdp_db, color="C0", linewidth=1.2)
        ax.set_title(f"RX {i+1}", fontsize=10)
        ax.set_xlabel("delay [ns]", fontsize=9)
        ax.set_ylabel("power [dB, norm]", fontsize=9)
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.set_ylim(-60, 5)
    # 비어 있는 subplot 숨김
    for j in range(n_show, n_rows * n_cols):
        axes[j // n_cols][j % n_cols].axis("off")

    fig.suptitle(
        f"PDP @ {meta.get('frequency_ghz', 0):.2f} GHz (showing {n_show}/{pdp_array.shape[0]} RX)",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
