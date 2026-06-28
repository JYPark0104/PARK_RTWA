"""GHM Twin 스타일 RSRP 대시보드 (RX scatter + 건물 외곽 + TX + 히스토그램).

참고 출력: ``RSRP Heatmap — 260531_GHM Twin_v0.1`` (5071 RX, 3 TX linear 합 → dBm).

실행 환경 (2026-06-04):
  Python 3.10, dclserver78, numpy / scipy / matplotlib / trimesh
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_DEAD_RSRP_DBM = -120.0


def combined_rsrp_dbm(rsrp_all: np.ndarray) -> np.ndarray:
    """3 TX ``rsrp_all`` (3, N) [dBm] → All-TX linear 합 RSRP [dBm]."""

    r = np.asarray(rsrp_all, dtype=np.float64)
    if r.ndim == 1:
        return r
    lin = np.where(np.isinf(r), 0.0, 10.0 ** (r / 10.0))
    return 10.0 * np.log10(np.sum(lin, axis=0) + 1e-30)


def dead_zone_mask(rsrp_all: np.ndarray) -> np.ndarray:
    """Dead zone: 3 TX 모두 ``-inf`` (GHM npz 기준 94개)."""

    r = np.asarray(rsrp_all, dtype=np.float64)
    if r.ndim == 1:
        return np.isinf(r)
    return np.all(np.isinf(r), axis=0)


def load_ghm_channel_rsrp(channel_npz: Path) -> dict:
    """GHM ``channel_data_*.npz`` 에서 RX/TX/RSRP 로드."""

    with np.load(channel_npz, allow_pickle=True) as d:
        if "rsrp_all" not in d or "rx_positions" not in d:
            raise KeyError(f"{channel_npz.name}: rsrp_all / rx_positions 필요")
        rsrp_all = np.asarray(d["rsrp_all"], dtype=np.float64)
        rx_raw = np.asarray(d["rx_positions"], dtype=np.float64)
        tx = np.asarray(d["tx_positions"], dtype=np.float64) if "tx_positions" in d else None

    rsrp_dbm = combined_rsrp_dbm(rsrp_all)
    dead = dead_zone_mask(rsrp_all)
    # rx_positions 가 3D(x,y,z)면 그대로, 2D(x,y)면 기본 높이(1.5m)를 붙여 3D 화.
    rx_raw = rx_raw.reshape(len(rx_raw), -1)
    if rx_raw.shape[1] >= 3:
        rx_positions = rx_raw[:, :3].astype(np.float64)
    else:
        z = np.full(len(rx_raw), 1.5, dtype=np.float64)
        rx_positions = np.column_stack([rx_raw[:, :2], z])
    tx_positions = None if tx is None else [[float(p[0]), float(p[1]), float(p[2])] for p in tx]

    return {
        "rsrp_dbm": rsrp_dbm,
        "rx_positions": rx_positions,
        "tx_positions": tx_positions,
        "dead_mask": dead,
        "num_rx": int(rsrp_dbm.shape[0]),
    }


def _draw_building_outlines(ax, mesh_ply: Path, linewidth: float = 0.35) -> None:
    """PLY top-view 삼각형 외곽선 (검은 윤곽)."""

    import trimesh
    from matplotlib.collections import LineCollection

    mesh = trimesh.load(mesh_ply, process=False)
    if not hasattr(mesh, "vertices") or not hasattr(mesh, "faces"):
        return
    v = np.asarray(mesh.vertices, dtype=np.float64)
    f = np.asarray(mesh.faces, dtype=np.int64)
    segs = []
    for tri in f:
        for i in range(3):
            a, b = tri[i], tri[(i + 1) % 3]
            segs.append(v[[a, b], :2])
    lc = LineCollection(
        segs,
        colors="black",
        linewidths=linewidth,
        alpha=0.85,
        zorder=1,
    )
    ax.add_collection(lc)


def save_ghm_rsrp_dashboard(
    out_path: Path,
    rx_positions: np.ndarray,
    rsrp_dbm: np.ndarray,
    tx_positions: list | None,
    dead_mask: np.ndarray | None = None,
    scene_title: str = "GHM Twin",
    mesh_ply_path: Path | None = None,
    vmin_dbm: float = -110.0,
    vmax_dbm: float = -60.0,
) -> dict:
    """GHM 참고 이미지와 동일 레이아웃 PNG 저장."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    rx_positions = np.asarray(rx_positions, dtype=np.float64)
    rsrp_dbm = np.asarray(rsrp_dbm, dtype=np.float64).reshape(-1)
    n = rsrp_dbm.shape[0]

    if dead_mask is None:
        dead_mask = rsrp_dbm <= _DEAD_RSRP_DBM
    else:
        dead_mask = np.asarray(dead_mask, dtype=bool).reshape(-1)

    valid_mask = ~dead_mask
    valid_rsrp = rsrp_dbm[valid_mask]

    fig, (ax_map, ax_hist) = plt.subplots(
        1, 2, figsize=(16, 7), dpi=120,
        gridspec_kw={"width_ratios": [1.35, 1.0]},
    )

    if mesh_ply_path and Path(mesh_ply_path).exists():
        _draw_building_outlines(ax_map, Path(mesh_ply_path))

    norm = Normalize(vmin=vmin_dbm, vmax=vmax_dbm)
    if np.any(valid_mask):
        sc = ax_map.scatter(
            rx_positions[valid_mask, 0],
            rx_positions[valid_mask, 1],
            c=rsrp_dbm[valid_mask],
            cmap="jet",
            norm=norm,
            s=14,
            linewidths=0,
            zorder=3,
        )
        cbar = fig.colorbar(sc, ax=ax_map, fraction=0.035, pad=0.02)
        cbar.set_label("RSRP [dBm]", fontsize=10)

    if np.any(dead_mask):
        ax_map.scatter(
            rx_positions[dead_mask, 0],
            rx_positions[dead_mask, 1],
            facecolors="white",
            edgecolors="black",
            linewidths=0.4,
            s=18,
            zorder=4,
            label=f"Dead Zone ({int(dead_mask.sum())})",
        )

    if tx_positions:
        for i, p in enumerate(tx_positions):
            ax_map.scatter(
                float(p[0]), float(p[1]),
                marker="*",
                s=380,
                c="red",
                edgecolors="black",
                linewidths=0.8,
                zorder=6,
                label="Target TX" if i == 0 else None,
            )
            ax_map.annotate(
                f"TX{i}",
                (float(p[0]), float(p[1])),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=9,
                fontweight="bold",
                color="darkred",
                zorder=7,
            )

    ax_map.set_xlabel("x [m]", fontsize=11)
    ax_map.set_ylabel("y [m]", fontsize=11)
    ax_map.set_title("RSRP per RX", fontsize=12)
    ax_map.set_aspect("equal", adjustable="box")
    ax_map.grid(True, linestyle=":", alpha=0.35)
    ax_map.legend(loc="upper left", fontsize=9, framealpha=0.9)

    fig.suptitle(
        f"RSRP Heatmap — {scene_title}\n"
        "All-TX combined RSRP (3 TXs, linear sum -> dB)",
        fontsize=13,
        y=1.02,
    )

    if valid_rsrp.size:
        ax_hist.hist(valid_rsrp, bins=50, color="steelblue", edgecolor="white", linewidth=0.4)
    ax_hist.set_xlabel("RSRP [dBm]", fontsize=11)
    ax_hist.set_ylabel("Number of RX", fontsize=11)
    ax_hist.set_title("RSRP Distribution", fontsize=12)
    ax_hist.grid(True, linestyle=":", alpha=0.35)

    n_dead = int(dead_mask.sum())
    n_valid = int(valid_mask.sum())
    stats = (
        f"Total RX: {n}\n"
        f"Valid RX: {n_valid} ({100.0 * n_valid / max(n, 1):.1f}%)\n"
        f"Dead Zone: {n_dead} ({100.0 * n_dead / max(n, 1):.1f}%)\n"
    )
    if valid_rsrp.size:
        stats += (
            f"Mean: {valid_rsrp.mean():.1f} dBm\n"
            f"Max: {valid_rsrp.max():.1f} dBm\n"
            f"Min: {valid_rsrp.min():.1f} dBm"
        )
    ax_hist.text(
        0.97, 0.97, stats,
        transform=ax_hist.transAxes,
        ha="right", va="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.85),
    )

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    return {
        "num_rx": n,
        "num_valid": n_valid,
        "num_dead": n_dead,
        "mean_dbm": float(valid_rsrp.mean()) if valid_rsrp.size else None,
        "min_dbm": float(valid_rsrp.min()) if valid_rsrp.size else None,
        "max_dbm": float(valid_rsrp.max()) if valid_rsrp.size else None,
    }
