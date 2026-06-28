"""derive_los_map.py — RX별 LoS/NLoS 분류 및 all-TX LoS/NLoS 2D map.

요구사항 (9) Review: PARK_2 RT Agent 의 los_map_allTX 와 동일 취지의
"전체 RX LoS/NLoS 맵" 을 표준 P1A/P1B NPZ 로부터 생성한다.

분류 규칙 (251218 표준 NPZ):
- los_nlos_flag: ray 단위 int32, LoS=1 / NLoS=0 (P1A fork 1542행 기준).
- RX 분류: 유효 ray 중 하나라도 flag==1 → LoS RX, 유효 ray 가 있으나 모두 0 → NLoS RX,
           유효 ray 가 전혀 없음 → Dead (무신호).

출력: CSV(rx_index,x,y,z,class) + 2D map PNG (LoS=초록 / NLoS=주황 / Dead=회색).

실행 환경: Python 3.10 / numpy 2.2 / matplotlib 3.10 (컨테이너 venv-webagent)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .common import KEY_LOS_NLOS, kst_timestamp, load_p1b_npz


# RX 분류 코드
_CLS_DEAD = -1
_CLS_NLOS = 0
_CLS_LOS = 1
_CLS_LABEL = {_CLS_DEAD: "DEAD", _CLS_NLOS: "NLOS", _CLS_LOS: "LOS"}


def _classify_per_rx(rays: dict, num_rx: int) -> np.ndarray:
    """RX별 LoS/NLoS/Dead 분류 배열 (num_rx,) 반환.

    los_nlos_flag, power 를 사용. power>0 인 ray 만 유효로 본다.
    """
    flag = rays.get(KEY_LOS_NLOS, None)
    power = rays.get("power", None)
    out = np.full(num_rx, _CLS_DEAD, dtype=np.int8)
    if flag is None or power is None:
        return out

    flag = np.asarray(flag)
    power = np.asarray(power)
    for rx in range(num_rx):
        p = power[rx].reshape(-1)
        fl = flag[rx].reshape(-1)
        valid = p > 0
        if not np.any(valid):
            out[rx] = _CLS_DEAD
        elif np.any(fl[valid] == 1):
            out[rx] = _CLS_LOS
        else:
            out[rx] = _CLS_NLOS
    return out


def _align_rx_positions(rx_positions: np.ndarray, n: int, meta: dict) -> np.ndarray:
    """RSRP 모듈과 동일한 정렬: 좌표 행 수 != 분류 수이면 rx_indices(1-based)로 매칭."""
    rx_positions = np.asarray(rx_positions, dtype=np.float32)
    if rx_positions.shape[0] == n:
        return rx_positions
    idx = np.asarray(meta.get("rx_indices") or [], dtype=np.int64).reshape(-1)
    if idx.size != n:
        m = min(rx_positions.shape[0], n)
        return rx_positions[:m]
    if idx.min() >= 1:
        idx = idx - 1
    if idx.max() >= rx_positions.shape[0]:
        m = min(rx_positions.shape[0], n)
        return rx_positions[:m]
    return rx_positions[idx]


def derive_los_map(
    npz_path: Path,
    rx_positions: np.ndarray,
    out_dir: Path,
    label: str | None = None,
    tx_positions: list | np.ndarray | None = None,
    mesh_ply_path: Path | str | None = None,
) -> dict:
    """all-TX LoS/NLoS 맵 생성.

    Returns
    -------
    dict: {"csv_path", "png_path", "classes", "summary"}
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_p1b_npz(Path(npz_path))
    rays = data["rays"]
    meta = data["meta"]
    num_rx = data["shape"][0]

    classes = _classify_per_rx(rays, num_rx)
    rx_positions = _align_rx_positions(rx_positions, num_rx, meta)
    rx_labels = np.asarray(meta.get("rx_indices") or np.arange(1, num_rx + 1)).reshape(-1)

    ts = kst_timestamp()
    lbl = f"_{label}" if label else ""
    csv_path = out_dir / f"LOS_NLOS{lbl}_{ts}.csv"
    png_path = out_dir / f"LOS_NLOS{lbl}_{ts}.png"

    n = min(len(rx_positions), len(classes))
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("rx_index,x,y,z,class\n")
        for i in range(n):
            rx_id = int(rx_labels[i]) if i < len(rx_labels) else i + 1
            pos = rx_positions[i]
            f.write(f"{rx_id},{pos[0]:.6f},{pos[1]:.6f},{pos[2]:.6f},{_CLS_LABEL[int(classes[i])]}\n")

    _save_los_map(png_path, rx_positions[:n], classes[:n], meta, tx_positions, mesh_ply_path)

    n_los = int(np.sum(classes == _CLS_LOS))
    n_nlos = int(np.sum(classes == _CLS_NLOS))
    n_dead = int(np.sum(classes == _CLS_DEAD))
    summary = {
        "num_rx": int(num_rx),
        "num_los": n_los,
        "num_nlos": n_nlos,
        "num_dead": n_dead,
        "los_ratio": round(n_los / max(num_rx, 1), 4),
        "frequency_ghz": meta.get("frequency_ghz", 0.0),
        "area_index": meta.get("area_index", -1),
    }
    return {
        "csv_path": str(csv_path),
        "png_path": str(png_path),
        "classes": classes,
        "summary": summary,
    }


def _save_los_map(
    out_path: Path,
    rx_positions: np.ndarray,
    classes: np.ndarray,
    meta: dict,
    tx_positions: list | None = None,
    mesh_ply_path: Path | str | None = None,
) -> None:
    """LoS/NLoS/Dead scatter 맵 PNG (LoS=초록, NLoS=주황, Dead=회색)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    freq = meta.get("frequency_ghz", 0)
    area = meta.get("area_index", "-")

    fig, ax = plt.subplots(figsize=(10, 8), dpi=120)

    # (옵션) PLY 건물 외곽선 오버레이
    if mesh_ply_path is not None:
        try:
            _overlay_mesh_outline(ax, Path(mesh_ply_path))
        except Exception:
            pass

    color_map = {_CLS_LOS: "#2ca02c", _CLS_NLOS: "#ff7f0e", _CLS_DEAD: "#bbbbbb"}
    size_map = {_CLS_LOS: 26, _CLS_NLOS: 22, _CLS_DEAD: 10}
    # Dead 를 먼저(아래), NLoS, LoS 순으로 그려 LoS 가 위에 보이게
    for cls in (_CLS_DEAD, _CLS_NLOS, _CLS_LOS):
        m = classes == cls
        if not np.any(m):
            continue
        ax.scatter(
            rx_positions[m, 0], rx_positions[m, 1],
            c=color_map[cls], s=size_map[cls],
            edgecolor="black", linewidth=0.2, zorder=3 + (cls + 1),
            label=_CLS_LABEL[cls],
        )

    if tx_positions is not None and len(tx_positions) > 0:
        for i, p in enumerate(tx_positions):
            px, py = float(p[0]), float(p[1])
            ax.scatter(px, py, marker="^", s=240, c="red", edgecolor="white", linewidth=2.0, zorder=10)
            ax.annotate(
                f"BS{i + 1}", (px, py), xytext=(8, 8), textcoords="offset points",
                fontsize=9, fontweight="bold", color="white",
                bbox=dict(boxstyle="round,pad=0.25", fc="red", ec="white", alpha=0.85), zorder=11,
            )

    n_los = int(np.sum(classes == _CLS_LOS))
    n_nlos = int(np.sum(classes == _CLS_NLOS))
    n_dead = int(np.sum(classes == _CLS_DEAD))

    legend_elems = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ca02c", markersize=9, label=f"LoS ({n_los})"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#ff7f0e", markersize=9, label=f"NLoS ({n_nlos})"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#bbbbbb", markersize=8, label=f"Dead ({n_dead})"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="red", markersize=11, label="BS (TX)"),
    ]
    ax.legend(handles=legend_elems, loc="upper right", fontsize=9, framealpha=0.9)

    ax.set_xlabel("x [m]", fontsize=11)
    ax.set_ylabel("y [m]", fontsize=11)
    ax.set_title(f"LoS / NLoS map @ {freq:.2f} GHz (Area {area}, all TX)", fontsize=12)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.25, linestyle="--")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _overlay_mesh_outline(ax, ply_path: Path) -> None:
    """PLY 메시의 top-down 외곽(투영 삼각형 에지)을 옅게 표시 (배경 참고용)."""
    try:
        import open3d as o3d
    except Exception:
        return
    if not ply_path.exists():
        return
    mesh = o3d.io.read_triangle_mesh(str(ply_path))
    v = np.asarray(mesh.vertices)
    if v.size == 0:
        return
    # 단순 외곽: x-y 산점을 옅은 회색으로 (건물 윤곽 느낌)
    ax.scatter(v[:, 0], v[:, 1], s=0.2, c="#dddddd", alpha=0.25, zorder=1, linewidths=0)
