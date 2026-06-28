"""rx_inspector.py — batch RT channel_data 에서 RX 단위 PADP/RSRP/Covariance 시각화.

7.RT Results 의 인터랙티브 RX 인스펙터용. 지도에서 RX 를 클릭하면 해당 (TX, RX) 의
PADP / PDP(RSRP) / 공분산 행렬(R_RX, R_TX) 그림을 on-demand 로 생성한다.

입력: batch RT 산출물 channel_data_*.npz (m5.save_output1_multi)
  키: rsrp_all (num_tx,num_rx), tx_positions, rx_positions,
      tau_tx{t}_rx{i}, power_tx{t}_rx{i}, aoa_tx{t}_rx{i}, aod_tx{t}_rx{i},
      R_TX_tx{t}_rx{i}, R_RX_tx{t}_rx{i}
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def find_channel_npz(session_dir: Path) -> Path | None:
    base = Path(session_dir) / "Batch_RT_Results" / "output_channel_data"
    if not base.exists():
        return None
    cands = sorted(base.glob("channel_data_*.npz"))
    return cands[-1] if cands else None


def _safe(d, key, default=None):
    return d[key] if key in getattr(d, "files", d) else default


def inspect_rx(npz_path: Path, rx_idx: int, tx_index: int, out_dir: Path) -> dict:
    """(tx_index, rx_idx) 의 PADP/PDP/Covariance PNG 생성 + RSRP 반환."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    d = np.load(npz_path, allow_pickle=False)   # 지연 로드 (배열 11만개+ 환경: dict() 금지)

    t, i = int(tx_index), int(rx_idx)
    pre = f"tx{t}_rx{i}"
    tau = _safe(d, f"tau_{pre}")          # ns
    powd = _safe(d, f"power_{pre}")       # dBm
    aoa = _safe(d, f"aoa_{pre}")          # deg
    R_RX = _safe(d, f"R_RX_{pre}")
    R_TX = _safe(d, f"R_TX_{pre}")
    rsrp_all = _safe(d, "rsrp_all")
    rsrp = float(rsrp_all[t, i]) if rsrp_all is not None and t < rsrp_all.shape[0] and i < rsrp_all.shape[1] else float("-inf")
    rx_valid_mask = _safe(d, "rx_valid_mask")
    valid_code = (int(np.asarray(rx_valid_mask).reshape(-1)[i])
                  if rx_valid_mask is not None and i < np.asarray(rx_valid_mask).reshape(-1).size else None)

    has_paths = tau is not None and np.asarray(tau).size > 0
    tag = f"rx{i}_tx{t}"
    padp_png = out_dir / f"{tag}_padp.png"
    pdp_png = out_dir / f"{tag}_pdp.png"
    cov_png = out_dir / f"{tag}_cov.png"

    # ── PADP (3D stem: x=delay, y=AoA, z=정규화 power, 줄기는 바닥 z=0 → power) ──
    fig = plt.figure(figsize=(8, 6), dpi=110)
    ax = fig.add_subplot(111, projection="3d")
    if has_paths:
        tau_a = np.asarray(tau, dtype=float).reshape(-1)
        pw_dbm = np.asarray(powd, dtype=float).reshape(-1)
        ao = np.asarray(aoa, dtype=float).reshape(-1) if aoa is not None else np.zeros_like(tau_a)
        # dBm → 선형(W) → 정규화 power (합=1)
        p_lin = 10.0 ** ((pw_dbm - 30.0) / 10.0)
        s = p_lin.sum()
        p_norm = p_lin / s if s > 0 else p_lin
        cmap = plt.get_cmap("viridis")
        cmax = p_norm.max() if p_norm.size else 1.0
        for x, y, z in zip(tau_a, ao, p_norm):
            col = cmap(float(z / cmax) if cmax > 0 else 0.0)
            ax.plot([x, x], [y, y], [0.0, z], color=col, linewidth=1.0, alpha=0.85)  # 줄기 (바닥에서 위로)
            ax.scatter([x], [y], [z], color=col, s=22, depthshade=True)               # 머리
        ax.set_xlabel("delay [ns]"); ax.set_ylabel("AoA [deg]"); ax.set_zlabel("norm. power")
        ax.set_zlim(0, max(cmax * 1.1, 1e-6))
        ax.view_init(elev=22, azim=-60)
    else:
        ax.text2D(0.5, 0.5, "Dead Zone (no valid path)", ha="center", va="center", transform=ax.transAxes)
    ax.set_title(f"PADP (3D) — RX{i} / TX{t}  (paths={int(np.asarray(tau).size) if has_paths else 0})")
    fig.tight_layout(); fig.savefig(padp_png); plt.close(fig)

    # ── PDP (delay vs power stem) — dBm 음수라도 콩나물이 '바닥에서 위로' 올라오게 ──
    fig, ax = plt.subplots(figsize=(7, 4), dpi=110)
    if has_paths:
        tau_a = np.asarray(tau, dtype=float).reshape(-1)
        pw = np.asarray(powd, dtype=float).reshape(-1)
        order = np.argsort(tau_a)
        # 바닥(floor)을 가장 약한 경로보다 살짝 아래로 잡아 stem 을 floor→값 으로 세운다.
        floor = float(np.min(pw)) - 5.0
        top = float(np.max(pw)) + 3.0
        ax.stem(tau_a[order], pw[order], basefmt=" ", bottom=floor)
        ax.set_ylim(floor, top)
        ax.set_xlabel("Delay τ [ns]"); ax.set_ylabel("Path power [dBm]")
    else:
        ax.text(0.5, 0.5, "Dead Zone", ha="center", va="center", transform=ax.transAxes)
    rsrp_txt = f"{rsrp:.2f} dBm" if np.isfinite(rsrp) else "Dead"
    ax.set_title(f"PDP — RX{i} / TX{t}   |   total RSRP = {rsrp_txt}")
    ax.grid(True, alpha=0.3, ls="--")
    fig.tight_layout(); fig.savefig(pdp_png); plt.close(fig)

    # ── Covariance heatmaps (|R_RX|, |R_TX|) ────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=110)
    for ax, R, name in [(axes[0], R_RX, "R_RX (RX cov)"), (axes[1], R_TX, "R_TX (TX cov)")]:
        if R is not None and np.asarray(R).size > 0:
            M = np.abs(np.asarray(R))
            im = ax.imshow(M, cmap="viridis", aspect="equal")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(f"|{name}|  {M.shape[0]}×{M.shape[1]}")
        else:
            ax.text(0.5, 0.5, f"{name}: N/A", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(name)
    fig.suptitle(f"Spatial Covariance — RX{i} / TX{t}", y=1.02)
    fig.tight_layout(); fig.savefig(cov_png, bbox_inches="tight"); plt.close(fig)

    return {
        "rx_idx": i, "tx_index": t,
        "rsrp_dbm": (rsrp if np.isfinite(rsrp) else None),
        "num_paths": int(np.asarray(tau).size) if has_paths else 0,
        "valid_code": valid_code,
        "padp_png": str(padp_png),
        "pdp_png": str(pdp_png),
        "cov_png": str(cov_png),
    }
