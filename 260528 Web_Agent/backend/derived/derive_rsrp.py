"""derive_rsrp.py — RX별 다중 ray power 합 (dB) 및 2D scatter map.

Plan 결정사항:
- RSRP = 10 * log10(sum(power) over rays for each RX)
- 출력: CSV (RX index, x, y, z, RSRP_dB) + 2D map PNG

호출:
    from backend.derived.derive_rsrp import derive_rsrp
    res = derive_rsrp(npz_path, rx_positions, session_dir)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .common import iter_rays_per_rx, kst_timestamp, load_p1b_npz


_RSRP_FLOOR_DB = -300.0


def _align_rx_positions(
    rx_positions: np.ndarray,
    rsrp_db: np.ndarray,
    meta: dict,
) -> np.ndarray:
    """RSRP 행 수와 좌표 행 수가 다를 때 NPZ ``rx_indices``(1-based)로 매칭."""

    rx_positions = np.asarray(rx_positions, dtype=np.float32)
    n_rsrp = int(rsrp_db.shape[0])
    if rx_positions.shape[0] == n_rsrp:
        return rx_positions

    idx = np.asarray(meta.get("rx_indices") or [], dtype=np.int64).reshape(-1)
    if idx.size != n_rsrp:
        n = min(rx_positions.shape[0], n_rsrp)
        return rx_positions[:n]

    if idx.min() >= 1:
        idx = idx - 1
    if idx.max() >= rx_positions.shape[0]:
        raise ValueError(
            f"rx_indices max {idx.max()} >= rx_positions rows {rx_positions.shape[0]}"
        )
    return rx_positions[idx]


def derive_rsrp(
    npz_path: Path,
    rx_positions: np.ndarray,
    out_dir: Path,
    label: str | None = None,
    tx_positions: list | np.ndarray | None = None,
    cell_size_m: float = 5.0,
    ghm_channel_npz: Path | str | None = None,
    mesh_ply_path: Path | str | None = None,
    scene_title: str | None = None,
    dashboard: bool = False,
) -> dict:
    """RX별 RSRP 산출.

    Parameters
    ----------
    npz_path : Path
        P1A 또는 P1B의 `*_Rays_ALL_RXs.npz` 경로.
    rx_positions : np.ndarray
        shape (num_rx, 3). 결과 CSV와 map plot에 사용.
    out_dir : Path
        출력 디렉토리. 일반적으로 `{session_dir}/Derived_RSRP_Results`.
    label : str | None
        파일 이름에 들어갈 추가 라벨.

    Returns
    -------
    dict
        {"csv_path", "png_path", "rsrp_db", "summary"}
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_p1b_npz(Path(npz_path))
    rays = data["rays"]
    meta = data["meta"]

    rsrp_list: list[float] = []
    for entry in iter_rays_per_rx(rays):
        valid_power = entry["power"][entry["mask"]]
        s = float(np.sum(valid_power))
        if s <= 0:
            rsrp_list.append(_RSRP_FLOOR_DB)
        else:
            rsrp_list.append(10.0 * np.log10(s))
    rsrp_db = np.asarray(rsrp_list, dtype=np.float32)
    rx_positions = _align_rx_positions(rx_positions, rsrp_db, meta)
    rx_labels = np.asarray(meta.get("rx_indices") or np.arange(1, len(rsrp_db) + 1)).reshape(-1)

    ts = kst_timestamp()
    lbl = f"_{label}" if label else ""
    csv_path = out_dir / f"RSRP{lbl}_{ts}.csv"
    png_path = out_dir / f"RSRP{lbl}_{ts}.png"

    with csv_path.open("w", encoding="utf-8") as f:
        f.write("rx_index,x,y,z,rsrp_dB\n")
        for label_id, (pos, db) in enumerate(zip(rx_positions, rsrp_db)):
            rx_id = int(rx_labels[label_id]) if label_id < len(rx_labels) else label_id + 1
            f.write(f"{rx_id},{pos[0]:.6f},{pos[1]:.6f},{pos[2]:.6f},{db:.4f}\n")

    plot_summary: dict
    if ghm_channel_npz and Path(ghm_channel_npz).exists():
        from .plot_ghm_rsrp_dashboard import load_ghm_channel_rsrp, save_ghm_rsrp_dashboard

        ch = load_ghm_channel_rsrp(Path(ghm_channel_npz))
        title = scene_title or Path(ghm_channel_npz).stem
        plot_tx = ch["tx_positions"] or tx_positions
        plot_summary = save_ghm_rsrp_dashboard(
            png_path,
            ch["rx_positions"],
            ch["rsrp_dbm"],
            plot_tx,
            ch["dead_mask"],
            scene_title=title,
            mesh_ply_path=Path(mesh_ply_path) if mesh_ply_path else None,
        )
    elif dashboard:
        # PARK_2 RSRP 대시보드 레이아웃 (RX scatter + 건물 외곽 + TX + 히스토그램).
        # P1A NPZ 의 RX별 RSRP 와 실제 좌표/씬 PLY 로부터 직접 렌더.
        from .plot_ghm_rsrp_dashboard import save_ghm_rsrp_dashboard

        dead = rsrp_db <= _RSRP_FLOOR_DB + 1
        valid = rsrp_db[~dead]
        if valid.size:
            vmin = float(np.percentile(valid, 5))
            vmax = float(np.percentile(valid, 95))
            if vmax - vmin < 1.0:
                vmin, vmax = float(valid.min()) - 1.0, float(valid.max()) + 1.0
        else:
            vmin, vmax = -120.0, -60.0
        title = scene_title or f"Area {meta.get('area_index','-')} @ {meta.get('frequency_ghz',0):.2f} GHz"
        plot_summary = save_ghm_rsrp_dashboard(
            png_path,
            rx_positions,
            rsrp_db,
            list(tx_positions) if tx_positions is not None else None,
            dead_mask=dead,
            scene_title=title,
            mesh_ply_path=Path(mesh_ply_path) if mesh_ply_path else None,
            vmin_dbm=vmin,
            vmax_dbm=vmax,
        )
    else:
        _save_rsrp_map(
            png_path,
            rx_positions,
            rsrp_db,
            meta,
            map_style="heatmap",
            cell_size_m=cell_size_m,
            tx_positions=tx_positions,
        )
        valid = rsrp_db[rsrp_db > _RSRP_FLOOR_DB + 1]
        plot_summary = {
            "num_rx": int(rsrp_db.shape[0]),
            "num_valid": int(valid.size),
            "min_db": float(np.min(valid)) if valid.size else _RSRP_FLOOR_DB,
            "max_db": float(np.max(valid)) if valid.size else _RSRP_FLOOR_DB,
            "median_db": float(np.median(valid)) if valid.size else _RSRP_FLOOR_DB,
        }

    summary = {
        **plot_summary,
        "frequency_ghz": meta["frequency_ghz"],
        "area_index": meta["area_index"],
        "plot_style": "ghm_dashboard" if (ghm_channel_npz or dashboard) else "grid_heatmap",
    }
    return {
        "csv_path": str(csv_path),
        "png_path": str(png_path),
        "rsrp_db": rsrp_db,
        "summary": summary,
    }


def _rsrp_grid_from_samples(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    cell_size_m: float = 5.0,
) -> tuple[np.ndarray, tuple[float, float, float, float], int, int]:
    """산점 RSRP → 2D 격자 (linear 보간). Coverage Map 과 유사한 heatmap 용."""

    from scipy.interpolate import griddata

    valid = z > _RSRP_FLOOR_DB + 1
    x, y, z = x[valid], y[valid], z[valid]
    if x.size < 3:
        raise ValueError("RSRP heatmap: valid sample < 3")

    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())
    pad = cell_size_m * 0.5
    xmin -= pad
    xmax += pad
    ymin -= pad
    ymax += pad

    nx = max(32, int(np.ceil((xmax - xmin) / cell_size_m)))
    ny = max(32, int(np.ceil((ymax - ymin) / cell_size_m)))
    xi = np.linspace(xmin, xmax, nx)
    yi = np.linspace(ymin, ymax, ny)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = griddata((x, y), z, (Xi, Yi), method="linear")
    extent = (xmin, xmax, ymin, ymax)
    return Zi.astype(np.float32), extent, nx, ny


def _save_rsrp_map(
    out_path: Path,
    rx_positions: np.ndarray,
    rsrp_db: np.ndarray,
    meta: dict,
    map_style: str = "heatmap",
    cell_size_m: float = 5.0,
    tx_positions: list | None = None,
) -> None:
    """RSRP 2D map PNG (기본: heatmap imshow, optional scatter)."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = rx_positions[:, 0]
    y = rx_positions[:, 1]
    freq = meta.get("frequency_ghz", 0)
    area = meta.get("area_index", "-")

    if map_style == "scatter":
        _save_rsrp_scatter(out_path, rx_positions, rsrp_db, meta)
        return

    grid, extent, nx, ny = _rsrp_grid_from_samples(x, y, rsrp_db, cell_size_m=cell_size_m)
    valid = grid[~np.isnan(grid)]
    vmin = float(np.percentile(valid, 2)) if valid.size else -160.0
    vmax = float(np.percentile(valid, 98)) if valid.size else -90.0

    fig, ax = plt.subplots(figsize=(10, 8), dpi=120)
    im = ax.imshow(
        grid,
        origin="lower",
        extent=extent,
        cmap="viridis",
        aspect="equal",
        vmin=vmin,
        vmax=vmax,
        interpolation="bilinear",
    )
    cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("RSRP [dB]", fontsize=11)

    if tx_positions is not None and len(tx_positions) > 0:
        for i, p in enumerate(tx_positions):
            px, py = float(p[0]), float(p[1])
            pz = float(p[2]) if len(p) > 2 else 0.0
            ax.scatter(
                px, py, marker="^", s=220, c="red", edgecolor="white",
                linewidth=2.0, zorder=6,
            )
            ax.annotate(
                f"BS{i + 1}\n({px:.0f},{py:.0f},{pz:.0f}m)",
                (px, py),
                xytext=(8, 8),
                textcoords="offset points",
                fontsize=9,
                fontweight="bold",
                color="white",
                bbox=dict(boxstyle="round,pad=0.25", fc="red", ec="white", alpha=0.85),
                zorder=7,
            )

    ax.set_xlabel("x [m]", fontsize=11)
    ax.set_ylabel("y [m]", fontsize=11)
    ax.set_title(
        f"RSRP heatmap @ {freq:.2f} GHz (Area {area}, {nx}×{ny} cells)",
        fontsize=12,
    )
    ax.grid(True, alpha=0.2, linestyle="--", color="white")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    if map_style == "both":
        scatter_path = out_path.with_name(out_path.stem + "_scatter" + out_path.suffix)
        _save_rsrp_scatter(scatter_path, rx_positions, rsrp_db, meta)


def _save_rsrp_scatter(
    out_path: Path,
    rx_positions: np.ndarray,
    rsrp_db: np.ndarray,
    meta: dict,
) -> None:
    """RX scatter (참고용)."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 7), dpi=120)
    valid_mask = rsrp_db > _RSRP_FLOOR_DB + 1
    sc = ax.scatter(
        rx_positions[valid_mask, 0],
        rx_positions[valid_mask, 1],
        c=rsrp_db[valid_mask],
        cmap="viridis",
        s=40,
        edgecolor="black",
        linewidth=0.3,
    )
    cbar = plt.colorbar(sc, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("RSRP [dB]", fontsize=11)
    ax.set_xlabel("x [m]", fontsize=11)
    ax.set_ylabel("y [m]", fontsize=11)
    ax.set_title(
        f"RSRP scatter @ {meta.get('frequency_ghz', 0):.2f} GHz (Area {meta.get('area_index', '-')})",
        fontsize=12,
    )
    ax.grid(True, alpha=0.25, linestyle="--")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
