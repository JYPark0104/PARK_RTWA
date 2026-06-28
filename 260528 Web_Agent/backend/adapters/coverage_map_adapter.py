"""coverage_map_adapter.py — Sionna RadioMapSolver 기반 광역 path-gain heatmap.

25\* 외 신규 모듈. RX 배치가 필요 없는 특수 경로.

실행 환경:
- Python 3.10
- 서버: dclcom61 (.venv-webagent)
- 의존성: sionna>=1.0, tensorflow>=2.18, numpy, matplotlib

호출 흐름:
    from backend.adapters.coverage_map_adapter import run_coverage_map
    result = run_coverage_map(session_dir, scene_xml_path, tx_list, antenna_cfg, cov_opts)
"""

from __future__ import annotations

import datetime as _dt
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence


# Mitsuba variant: backend.runtime_env 정책 유지 (GPU RT 시 cuda)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np  # noqa: E402

from ..jobs.runtime_env import configure_mitsuba_variant  # noqa: E402

import mitsuba as _mi  # noqa: E402

configure_mitsuba_variant()

import tensorflow as _tf  # noqa: E402
from sionna.rt import (  # noqa: E402
    load_scene,
    PlanarArray,
    RadioMapSolver,
    Transmitter,
)


# ---------------------------------------------------------------------------
# 입력 데이터 클래스 (Pydantic 모델과 1:1 매핑되는 dataclass)
# ---------------------------------------------------------------------------
@dataclass
class TXPlacement:
    """단일 TX 배치 정보."""

    position: tuple[float, float, float]
    orientation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    name: str = "tx"


@dataclass
class AntennaSimpleCfg:
    """RadioMapSolver는 PlanarArray만 필요. rows x cols만 보면 됨."""

    bs_rows: int = 1
    bs_cols: int = 1
    pattern: str = "iso"
    polarization: str = "V"
    vertical_spacing: float = 0.5
    horizontal_spacing: float = 0.5


@dataclass
class CoverageMapConfig:
    """Plan 4.7 표 그대로."""

    frequency_ghz: float = 7.5
    cell_size: tuple[float, float] = (1.0, 1.0)
    height_m: float = 1.5
    center: Optional[tuple[float, float, float]] = None   # None → AABB 중심
    size: Optional[tuple[float, float]] = None            # None → AABB XY
    samples_per_tx: int = int(1e8)
    max_depth: int = 5
    specular_reflection: bool = True
    diffuse_reflection: bool = True
    refraction: bool = True
    seed: int = 41


# ---------------------------------------------------------------------------
# 메인 함수
# ---------------------------------------------------------------------------
def _kst_now() -> str:
    """KST 기준 YYMMDDhhmm 타임스탬프."""

    kst = _dt.timezone(_dt.timedelta(hours=9))
    return _dt.datetime.now(kst).strftime("%y%m%d_%H%M%S")


def _resolve_area(
    scene_xml_path: Path,
    cfg: CoverageMapConfig,
) -> tuple[tuple[float, float, float], tuple[float, float]]:
    """center/size가 None이면 scene_info.json 의 AABB를 사용해 자동 계산.

    scene_builder.py가 생성한 scene/scene_info.json을 같은 폴더에서 찾아 사용한다.
    """

    info_path = scene_xml_path.parent / "scene_info.json"
    if cfg.center is None or cfg.size is None:
        if not info_path.exists():
            raise FileNotFoundError(
                f"AABB auto-detection requires {info_path}; provide explicit center/size or run scene_builder first."
            )
        info = json.loads(info_path.read_text(encoding="utf-8"))
        mn, mx = info["aabb_min"], info["aabb_max"]
        cx, cy = (mn[0] + mx[0]) / 2.0, (mn[1] + mx[1]) / 2.0
        sx, sy = mx[0] - mn[0], mx[1] - mn[1]
    else:
        cx, cy = cfg.center[:2]
        sx, sy = cfg.size

    center = (float(cx), float(cy), float(cfg.height_m))
    size = (float(sx), float(sy))
    return center, size


def _save_heatmap_png(
    out_path: Path,
    path_gain_db: np.ndarray,
    cell_size: Sequence[float],
    center: Sequence[float],
    size: Sequence[float],
    tx_positions: Sequence[Sequence[float]],
    frequency_ghz: float,
) -> None:
    """Coverage Map heatmap PNG 저장. AGENTS.md 그래프 체크 규칙 준수."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 7), dpi=120)

    # path_gain_db shape: (num_tx, ny, nx) 또는 (ny, nx)
    if path_gain_db.ndim == 3:
        combined = 10 * np.log10(np.sum(10 ** (path_gain_db / 10.0), axis=0) + 1e-30)
    else:
        combined = path_gain_db

    cx, cy, _ = center
    sx, sy = size
    extent = (cx - sx / 2.0, cx + sx / 2.0, cy - sy / 2.0, cy + sy / 2.0)

    im = ax.imshow(
        combined,
        origin="lower",
        extent=extent,
        cmap="viridis",
        aspect="equal",
    )
    cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("Path Gain [dB]", fontsize=11)

    for i, p in enumerate(tx_positions):
        ax.scatter(p[0], p[1], marker="^", s=140, c="red", edgecolor="white",
                   linewidth=1.5, label="TX" if i == 0 else None, zorder=5)

    ax.set_xlabel("x [m]", fontsize=11)
    ax.set_ylabel("y [m]", fontsize=11)
    ax.set_title(
        f"Coverage Map @ {frequency_ghz:.2f} GHz (z={center[2]:.2f} m, "
        f"cell={cell_size[0]:.2f}×{cell_size[1]:.2f} m)",
        fontsize=12,
    )
    if tx_positions:
        ax.legend(loc="upper right", fontsize=10, framealpha=0.85)
    ax.grid(True, alpha=0.25, linestyle="--")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def run_coverage_map(
    session_dir: Path,
    scene_xml_path: Path,
    tx_list: Iterable[TXPlacement],
    antenna_cfg: AntennaSimpleCfg,
    cov_cfg: CoverageMapConfig,
) -> dict:
    """Sionna RadioMapSolver를 호출해서 path-gain heatmap을 NPZ + PNG로 저장.

    Returns
    -------
    dict : {
        "npz_path": str,
        "png_path": str,
        "summary": {
            "n_tx": int,
            "n_cells": (int, int),
            "min_db": float,
            "max_db": float,
            "median_db": float,
        }
    }
    """

    session_dir = Path(session_dir)
    scene_xml_path = Path(scene_xml_path)
    tx_list = list(tx_list)
    if not tx_list:
        raise ValueError("coverage_map needs at least one TX placement.")

    _tf.random.set_seed(cov_cfg.seed)
    np.random.seed(cov_cfg.seed)

    scene = load_scene(str(scene_xml_path), merge_shapes=False)
    scene.frequency = cov_cfg.frequency_ghz * 1e9

    array = PlanarArray(
        num_rows=antenna_cfg.bs_rows,
        num_cols=antenna_cfg.bs_cols,
        vertical_spacing=antenna_cfg.vertical_spacing,
        horizontal_spacing=antenna_cfg.horizontal_spacing,
        pattern=antenna_cfg.pattern,
        polarization=antenna_cfg.polarization,
    )
    scene.tx_array = array
    scene.rx_array = array  # RadioMapSolver는 rx_array 도 요구

    for i, tx in enumerate(tx_list):
        scene.add(Transmitter(
            f"{tx.name}_{i+1}" if tx.name == "tx" else tx.name,
            tuple(tx.position),
            orientation=tuple(tx.orientation),
        ))

    center, size = _resolve_area(scene_xml_path, cov_cfg)

    solver = RadioMapSolver()
    rm = solver(
        scene=scene,
        max_depth=cov_cfg.max_depth,
        cell_size=tuple(cov_cfg.cell_size),
        samples_per_tx=int(cov_cfg.samples_per_tx),
        specular_reflection=cov_cfg.specular_reflection,
        diffuse_reflection=cov_cfg.diffuse_reflection,
        refraction=cov_cfg.refraction,
        center=center,
        size=size,
        orientation=(0.0, 0.0, 0.0),
        seed=cov_cfg.seed,
    )

    # rm.path_gain 은 shape (num_tx, num_cells_y, num_cells_x). TF or NumPy 가능
    path_gain = rm.path_gain
    if hasattr(path_gain, "numpy"):
        path_gain = path_gain.numpy()
    path_gain_db = 10.0 * np.log10(np.maximum(path_gain, 1e-30))

    out_dir = session_dir / "Coverage_Map_Results"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = _kst_now()
    npz_path = out_dir / f"CoverageMap_{ts}.npz"
    png_path = out_dir / f"CoverageMap_{ts}.png"

    tx_positions = [list(tx.position) for tx in tx_list]
    np.savez_compressed(
        npz_path,
        path_gain=path_gain.astype(np.float32),
        path_gain_db=path_gain_db.astype(np.float32),
        cell_size=np.asarray(cov_cfg.cell_size, dtype=np.float32),
        center=np.asarray(center, dtype=np.float32),
        size=np.asarray(size, dtype=np.float32),
        height_m=np.float32(cov_cfg.height_m),
        frequency_ghz=np.float32(cov_cfg.frequency_ghz),
        tx_positions=np.asarray(tx_positions, dtype=np.float32),
        max_depth=np.int32(cov_cfg.max_depth),
    )

    _save_heatmap_png(
        png_path,
        path_gain_db,
        cell_size=cov_cfg.cell_size,
        center=center,
        size=size,
        tx_positions=tx_positions,
        frequency_ghz=cov_cfg.frequency_ghz,
    )

    valid = path_gain_db[np.isfinite(path_gain_db) & (path_gain_db > -250)]
    summary = {
        "n_tx": len(tx_list),
        "n_cells": list(path_gain_db.shape[-2:]),
        "min_db": float(np.min(valid)) if valid.size else float("nan"),
        "max_db": float(np.max(valid)) if valid.size else float("nan"),
        "median_db": float(np.median(valid)) if valid.size else float("nan"),
    }
    return {
        "npz_path": str(npz_path),
        "png_path": str(png_path),
        "summary": summary,
    }


if __name__ == "__main__":
    import argparse, sys

    p = argparse.ArgumentParser(description="Coverage Map adapter CLI test")
    p.add_argument("--scene", required=True, help="scene.xml 경로")
    p.add_argument("--out", required=True, help="세션 출력 디렉토리")
    p.add_argument("--tx", nargs="+", required=True,
                   help="TX 좌표 'x,y,z' (여러 개 가능)")
    p.add_argument("--freq", type=float, default=7.5)
    p.add_argument("--bs-rows", type=int, default=1)
    p.add_argument("--bs-cols", type=int, default=1)
    p.add_argument("--cell", type=float, default=2.0,
                   help="cell_size in meters (square)")
    p.add_argument("--samples", type=float, default=1e6,
                   help="samples_per_tx (lower for quick test)")
    args = p.parse_args()

    tx_list = []
    for s in args.tx:
        xyz = tuple(float(v) for v in s.split(","))
        tx_list.append(TXPlacement(position=xyz))

    cov_cfg = CoverageMapConfig(
        frequency_ghz=args.freq,
        cell_size=(args.cell, args.cell),
        samples_per_tx=int(args.samples),
    )
    ant_cfg = AntennaSimpleCfg(bs_rows=args.bs_rows, bs_cols=args.bs_cols)

    out = run_coverage_map(
        session_dir=Path(args.out),
        scene_xml_path=Path(args.scene),
        tx_list=tx_list,
        antenna_cfg=ant_cfg,
        cov_cfg=cov_cfg,
    )
    print(json.dumps(out, indent=2))
