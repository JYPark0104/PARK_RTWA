# -*- coding: utf-8 -*-
"""
P1U_UE_Fingerprint_Localization_2605v1.py
==========================================
UE Fingerprint Localization Heatmap (single BS, SISO)
======================================================

목적
----
이미 RT가 완료된 channel_data_GHMTwin2_cutting.npz (1 TX × 2,492 RX, SISO)에서
하나의 query RX의 채널 핑거프린트와 위치가 알려진 다른 RX들의 핑거프린트를
5가지 메트릭으로 비교하여 query 위치에 대한 likelihood heatmap을 산출합니다.

heatmap 은 Map_Mesh.obj 의 top-view 위에 overlay 됩니다.

데이터 한계 (2026-05-14 기준 확인 사항)
---------------------------------------
* rx_positions.shape = (42, 2) 이지만 채널 데이터는 2,492 개의 RX가 존재합니다.
  → 위치가 알려진 42 개 RX (rx0 ~ rx41) 만 fingerprinting DB로 사용합니다.
* 42 개 RX 는 y=81.756 m 의 1차원 line 위에 x ∈ [-62.347, 81.376] 으로 분포.
  → heatmap 은 1D line 색상 + 보간 2D 로 동시에 표시.
* R_TX, R_RX 는 (1, 1) → SISO. SPD-manifold 기반 metric (Chordal, AIRM 등)
  은 의미가 없으므로 본 스크립트에서 제외하고, PDP / AoA 기반 metric 만 사용.
* TX 위치 / 주파수 등 메타가 npz 에 없음 → P1A_RT_to_Rays_2509v6.py 의 Jonggak
  기본값 (TX = (-51.561, -21.794, 19.0) m, fc = 7.5 GHz) 을 그대로 가정.

사용 메트릭 (5종)
-----------------
[A] PDP 기반 (1D: tau-power 히스토그램)
   1. Pearson correlation       — 1 - corr  ⇒  거리화
   2. Cosine similarity         — 1 - cos   ⇒  거리화
   3. Wasserstein-1 distance    — Earth Mover's Distance, scipy.stats.wasserstein_distance
   4. Jensen-Shannon divergence — scipy.spatial.distance.jensenshannon
[B] Scalar 기반
   5. RMS delay spread 차이     — |τ_rms(p) − τ_rms(q)|  (단순 스칼라 비교)

heatmap 산출 흐름
-----------------
1. rx0 (= target_rx_index) 를 query 로 두고 rx1 ~ rx41 (41 개) 을 DB 로 사용.
2. 각 metric 별로 d_i = dist(query_pdp, db_i_pdp) 계산 → softmax(-d/τ) 로 likelihood.
3. 1D line 위 41 점에 likelihood 색상 표시.
4. Map_Mesh.obj 를 top-view (x-y projection) 로 시각화 (vertex sub-sampling).
5. TX, GT(query) 위치 마커 추가, metric 별 추정 위치(argmax) 와 GT 거리 출력.

산출물 (P1U_UE_Fingerprint_Results/ 자동 생성)
---------------------------------------------
* P1U_heatmap_<metric>_<TIMESTAMP>.png    : metric 별 heatmap
* P1U_summary_<TIMESTAMP>.csv             : metric × RX 별 distance / likelihood
* P1U_localization_error_<TIMESTAMP>.csv  : metric 별 추정 위치 vs GT 거리
* P1U_metric_comparison_<TIMESTAMP>.png   : 5 metric 비교 plot

실행 환경 (가정)
----------------
* Python  : 3.10 ~ 3.12
* 서버    : 로컬 Mac (혹은 dclcom45/55 같은 공용 서버)
* 라이브러리:
    - numpy        >= 1.24
    - scipy        >= 1.10
    - matplotlib   >= 3.7
    - tqdm         (선택)

실행 방법
---------
$ python P1U_UE_Fingerprint_Localization_2605v1.py \
       --npz "/path/to/channel_data_GHMTwin2_cutting.npz" \
       --obj "/path/to/Map_Mesh.obj" \
       --query 0 \
       --tx-x -51.561 --tx-y -21.794 --tx-z 19.0 \
       --fc-ghz 7.5

옵션 모두 생략 시 본 파일 기본값 (`DEFAULT_*`) 을 사용합니다.

작성: 2026-05-14
규칙 준수: CLAUDE.md / AGENTS.md
"""

from __future__ import annotations

# ============================================================
# 표준 라이브러리
# ============================================================
import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ============================================================
# 외부 라이브러리
# ============================================================
import numpy as np

try:
    from scipy.spatial.distance import jensenshannon as _scipy_jsd
    from scipy.stats import wasserstein_distance as _scipy_wd
    _HAVE_SCIPY = True
except Exception:  # noqa: BLE001
    _HAVE_SCIPY = False

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib import cm

matplotlib.rcParams["font.family"] = ["DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


# ============================================================
# Default config (CLI 인자 미지정 시 사용)
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent

DEFAULT_NPZ = WORKSPACE_ROOT / "260512 RT Result Data" / "channel_data_GHMTwin2_cutting.npz"
DEFAULT_OBJ = WORKSPACE_ROOT / "260512 RT Result Data" / "Map_Mesh.obj"

DEFAULT_TX_XYZ = (-51.561, -21.794, 19.0)
DEFAULT_FC_GHZ = 7.5
DEFAULT_QUERY_RX = 0          # target_rx_index 와 일치
DEFAULT_PDP_BINS = 128
DEFAULT_PDP_CLIP_NS = 2000.0  # tau 클립 (ns) — outlier 억제
DEFAULT_OBJ_VERTEX_SAMPLE = 30000  # top-view 시각화용 vertex sub-sampling
DEFAULT_RESULT_DIR = SCRIPT_DIR / "P1U_UE_Fingerprint_Results"
DEFAULT_HEATMAP_GRID = 256    # 2D 보간 격자 해상도
DEFAULT_TEMPERATURE = None    # softmax temperature (None ⇒ metric 별 자동)


# ============================================================
# NPZ 로더
# ============================================================
@dataclass
class ChannelData:
    """RT 결과 NPZ 의 메타와 핑거프린팅에 필요한 필드만 정리한 컨테이너."""
    rx_positions_xy: np.ndarray   # (N_pos, 2) — 알려진 RX 위치 (현재 42 개)
    rsrp_all: np.ndarray          # (N_total,) — 전체 RX 의 RSRP (dBm). dead = -inf
    target_rx_index: int          # query 로 사용할 RX 인덱스
    n_total_rx: int               # 전체 RX 데이터 수 (= 2492)
    tau_list: List[np.ndarray]    # 위치가 있는 RX (N_pos 개) 의 tau (ns)
    power_list: List[np.ndarray]  # 위치가 있는 RX (N_pos 개) 의 power (linear, watts? — 그대로 사용)
    aoa_list: List[np.ndarray]    # 위치가 있는 RX (N_pos 개) 의 AoA (deg)


def load_channel_data(npz_path: Path) -> ChannelData:
    """채널 NPZ 로드. rx_positions 의 행 수 (= N_pos) 만큼 RX 데이터를 가져온다."""
    if not npz_path.exists():
        sys.exit(f"[ERROR] NPZ 파일이 없습니다: {npz_path}")
    print(f"[load] {npz_path}  ({npz_path.stat().st_size/1e6:.1f} MB)")

    data = np.load(npz_path, allow_pickle=True)
    rx_xy = np.asarray(data["rx_positions"], dtype=np.float64)
    rsrp_all = np.asarray(data["rsrp_all"], dtype=np.float64)
    target_idx = int(np.asarray(data["target_rx_index"]).ravel()[0])

    n_pos = rx_xy.shape[0]
    n_total = rsrp_all.shape[0]
    print(f"  N_pos (위치 있음): {n_pos}, N_total (채널 데이터): {n_total}")
    if n_total < n_pos:
        sys.exit(f"[ERROR] 채널 데이터 ({n_total}) 가 위치 ({n_pos}) 보다 적습니다.")

    tau_list, power_list, aoa_list = [], [], []
    for i in range(n_pos):
        tau_list.append(np.asarray(data[f"tau_rx{i}"], dtype=np.float64))
        power_list.append(np.asarray(data[f"power_rx{i}"], dtype=np.float64))
        aoa_list.append(np.asarray(data[f"aoa_rx{i}"], dtype=np.float64))

    return ChannelData(
        rx_positions_xy=rx_xy,
        rsrp_all=rsrp_all,
        target_rx_index=target_idx,
        n_total_rx=n_total,
        tau_list=tau_list,
        power_list=power_list,
        aoa_list=aoa_list,
    )


# ============================================================
# PDP 빌더
# ============================================================
class PDPBuilder:
    """tau (ns), power (linear) → 정규화된 1D PDP 히스토그램.

    * 모든 RX 가 같은 bin edge 를 사용하도록 전역 [tau_min, tau_max] 를 결정한 뒤
      각 RX 의 PDP 를 동일 격자에 투영한다.
    * power 가 비어있는 (path 0 개) RX 는 균등 분포로 둔다 (모든 metric 에서 비슷한
      거리를 갖도록).
    """

    def __init__(self, n_bins: int = 128, tau_clip_ns: float = 2000.0):
        self.n_bins = n_bins
        self.tau_clip_ns = tau_clip_ns
        self.bin_edges: Optional[np.ndarray] = None
        self.bin_centers: Optional[np.ndarray] = None

    @staticmethod
    def _power_to_linear(power: np.ndarray) -> np.ndarray:
        """power 단위 자동 감지: 모두 음수이면 dBm 으로 보고 linear 변환."""
        if power.size == 0:
            return power
        if np.all(power <= 0) and float(np.min(power)) < -10.0:
            return np.power(10.0, power / 10.0)
        return power

    def fit(self, tau_list: List[np.ndarray]) -> None:
        # tau == 0 path 는 invalid 로 보고 제외 후 통계 결정
        valid_tau_list = [t[t != 0] for t in tau_list if t.size > 0]
        all_tau = (np.concatenate(valid_tau_list)
                   if valid_tau_list else np.array([0.0, 1.0]))
        q99 = float(np.quantile(all_tau, 0.99)) if all_tau.size > 0 else 0.0
        # tau 단위 자동 추정:
        #   q99 < 1e-3   → sec  → ns 변환 (×1e9)
        #   1e-3 ≤ q99 < 1e3 → 이미 ns 라고 보고 그대로 사용
        #   1e3 ≤ q99    → 거리(m) 가능성 → ns 변환 (×1/c×1e9 = ÷0.3)
        if q99 > 0 and q99 < 1e-3:
            self._scale = 1e9
            unit = "sec → ns"
        elif q99 >= 1e3:
            self._scale = 1.0 / 0.299792458    # m → ns  (c = 0.3 m/ns)
            unit = "m → ns"
        else:
            self._scale = 1.0
            unit = "ns (그대로)"
        all_tau_ns = np.clip(all_tau * self._scale, 0.0, self.tau_clip_ns)
        tau_min = 0.0
        tau_max = float(np.max(all_tau_ns)) if all_tau_ns.size > 0 else 1.0
        if tau_max <= tau_min:
            tau_max = tau_min + 1.0
        self.bin_edges = np.linspace(tau_min, tau_max + 1e-9, self.n_bins + 1)
        self.bin_centers = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])
        print(f"  [PDP] tau q99={q99:.3e} → unit={unit}, "
              f"bin range [{tau_min:.2f}, {tau_max:.2f}] ns, n_bins={self.n_bins}")

    def build(self, tau: np.ndarray, power: np.ndarray) -> np.ndarray:
        if self.bin_edges is None:
            raise RuntimeError("PDPBuilder.fit() 먼저 호출")
        if tau.size == 0 or power.size == 0:
            return np.full(self.n_bins, 1.0 / self.n_bins, dtype=np.float64)
        # invalid path 제거: tau == 0 또는 power == 0
        mask = (tau != 0) & (power != 0) & np.isfinite(tau) & np.isfinite(power)
        if not np.any(mask):
            return np.full(self.n_bins, 1.0 / self.n_bins, dtype=np.float64)
        tau = tau[mask]
        power = power[mask]
        # dBm → linear 자동 변환
        power_lin = self._power_to_linear(power)
        tau_ns = np.clip(tau * self._scale, 0.0, self.tau_clip_ns)
        pdp, _ = np.histogram(tau_ns, bins=self.bin_edges, weights=power_lin)
        s = pdp.sum()
        if s <= 0:
            return np.full(self.n_bins, 1.0 / self.n_bins, dtype=np.float64)
        return pdp / s


# ============================================================
# Metric Lib
# ============================================================
class MetricLib:
    """모든 metric 은 '거리' (작을수록 유사) 형태로 통일."""

    @staticmethod
    def pearson_dist(p: np.ndarray, q: np.ndarray) -> float:
        if p.std() == 0 or q.std() == 0:
            return 1.0
        r = float(np.corrcoef(p, q)[0, 1])
        if not np.isfinite(r):
            return 1.0
        return 1.0 - r  # [0, 2]

    @staticmethod
    def cosine_dist(p: np.ndarray, q: np.ndarray) -> float:
        npn = np.linalg.norm(p)
        nqn = np.linalg.norm(q)
        if npn == 0 or nqn == 0:
            return 1.0
        c = float(np.dot(p, q) / (npn * nqn))
        return 1.0 - c

    @staticmethod
    def wasserstein_dist(p: np.ndarray, q: np.ndarray, support: np.ndarray) -> float:
        if _HAVE_SCIPY:
            return float(_scipy_wd(support, support, u_weights=p, v_weights=q))
        cdf_p = np.cumsum(p)
        cdf_q = np.cumsum(q)
        return float(np.sum(np.abs(cdf_p - cdf_q)) * (support[1] - support[0]))

    @staticmethod
    def jsd_dist(p: np.ndarray, q: np.ndarray) -> float:
        if _HAVE_SCIPY:
            d = float(_scipy_jsd(p + 1e-12, q + 1e-12, base=2))
            return d
        m = 0.5 * (p + q) + 1e-12
        kl_pm = np.sum(p * np.log2((p + 1e-12) / m))
        kl_qm = np.sum(q * np.log2((q + 1e-12) / m))
        return float(0.5 * (kl_pm + kl_qm))

    @staticmethod
    def tau_rms_diff(p: np.ndarray, q: np.ndarray, support: np.ndarray) -> float:
        def rms(weights, support):
            s = weights.sum()
            if s <= 0:
                return 0.0
            mu = float(np.sum(support * weights) / s)
            var = float(np.sum(((support - mu) ** 2) * weights) / s)
            return float(np.sqrt(max(var, 0.0)))
        return abs(rms(p, support) - rms(q, support))


METRIC_NAMES = ["pearson", "cosine", "wasserstein", "jsd", "tau_rms"]
METRIC_LABELS = {
    "pearson":     "PDP Pearson  (1 - r)",
    "cosine":      "PDP Cosine   (1 - cos)",
    "wasserstein": "PDP Wasserstein (ns)",
    "jsd":         "PDP Jensen-Shannon (bits)",
    "tau_rms":     "RMS delay diff (ns)",
}


def compute_all_metrics(
    pdp_query: np.ndarray,
    pdp_db: np.ndarray,
    support: np.ndarray,
) -> Dict[str, float]:
    return {
        "pearson":     MetricLib.pearson_dist(pdp_query, pdp_db),
        "cosine":      MetricLib.cosine_dist(pdp_query, pdp_db),
        "wasserstein": MetricLib.wasserstein_dist(pdp_query, pdp_db, support),
        "jsd":         MetricLib.jsd_dist(pdp_query, pdp_db),
        "tau_rms":     MetricLib.tau_rms_diff(pdp_query, pdp_db, support),
    }


# ============================================================
# Heatmap (likelihood) 빌더
# ============================================================
def softmax_likelihood(distances: np.ndarray, temperature: Optional[float]) -> np.ndarray:
    """거리 → likelihood. temperature=None 이면 자동 (median 거리).

    * 거리값의 스케일이 metric 마다 매우 달라 (예: pearson∈[0,2], wasserstein∈[0,1000ns])
      자동 temperature = median(distances) 사용으로 metric-agnostic 비교 가능.
    """
    d = np.asarray(distances, dtype=np.float64)
    finite = d[np.isfinite(d)]
    if finite.size == 0:
        return np.full_like(d, 1.0 / d.size)
    if temperature is None:
        med = float(np.median(finite))
        tau = max(med, 1e-9)
    else:
        tau = max(float(temperature), 1e-9)
    z = -d / tau
    z = z - np.nanmax(z)  # numerical stability
    w = np.exp(z)
    w[~np.isfinite(w)] = 0.0
    s = w.sum()
    return w / s if s > 0 else np.full_like(w, 1.0 / w.size)


# ============================================================
# OBJ Reader (top-view 시각화용 vertex 일부만 읽기)
# ============================================================
def load_obj_vertices(obj_path: Path, n_sample: int = 30000) -> np.ndarray:
    """OBJ 의 'v ' 줄에서 (x, y, z) 만 추출하여 sub-sampling."""
    if not obj_path.exists():
        print(f"  [obj] WARNING: {obj_path} 없음 → mesh overlay skip.")
        return np.empty((0, 3), dtype=np.float64)

    print(f"[obj] reading {obj_path.name} ({obj_path.stat().st_size/1e6:.1f} MB) ...")
    t0 = time.time()
    xs, ys, zs = [], [], []
    with open(obj_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                # format: v x y z [r g b]
                xs.append(float(parts[1]))
                ys.append(float(parts[2]))
                zs.append(float(parts[3]))
    verts = np.column_stack([xs, ys, zs])
    print(f"  [obj] {len(xs):,} vertices read in {time.time()-t0:.1f}s")
    if n_sample is not None and verts.shape[0] > n_sample:
        idx = np.random.RandomState(2605).choice(verts.shape[0], n_sample, replace=False)
        verts = verts[idx]
        print(f"  [obj] sub-sampled to {verts.shape[0]:,}")
    return verts


# ============================================================
# Visualization
# ============================================================
def draw_topview_with_heatmap(
    obj_verts: np.ndarray,
    rx_xy: np.ndarray,
    likelihood: np.ndarray,
    query_xy: Optional[np.ndarray],
    tx_xyz: Tuple[float, float, float],
    metric_name: str,
    metric_value_query_to_db: np.ndarray,
    out_path: Path,
    interpolate_2d: bool = True,
    grid_size: int = DEFAULT_HEATMAP_GRID,
) -> None:
    """1 figure 안에 (1) Map_Mesh top-view (2) likelihood overlay 그림."""
    fig, ax = plt.subplots(figsize=(11, 9), constrained_layout=True)

    # (1) Map_Mesh top-view (z 를 색으로) — 옅게
    if obj_verts.shape[0] > 0:
        sc = ax.scatter(
            obj_verts[:, 0], obj_verts[:, 1],
            c=obj_verts[:, 2],
            cmap="Greys", s=0.4, alpha=0.35,
            rasterized=True,
            label=f"Map_Mesh.obj (z)",
        )
        cb = plt.colorbar(sc, ax=ax, fraction=0.025, pad=0.02, location="right")
        cb.set_label("z (m)  — building height", rotation=90, fontsize=9)

    # (2) bbox 결정
    if obj_verts.shape[0] > 0:
        xmin, xmax = obj_verts[:, 0].min(), obj_verts[:, 0].max()
        ymin, ymax = obj_verts[:, 1].min(), obj_verts[:, 1].max()
    else:
        xmin, ymin = rx_xy.min(0)
        xmax, ymax = rx_xy.max(0)
        margin = 30.0
        xmin -= margin; xmax += margin; ymin -= margin; ymax += margin

    # (3) 2D 보간 heatmap (선택, 1D line 만 있을 때는 가로 stripe 처럼 보임)
    if interpolate_2d and rx_xy.shape[0] >= 3:
        from scipy.interpolate import griddata  # noqa: WPS433
        gx = np.linspace(xmin, xmax, grid_size)
        gy = np.linspace(ymin, ymax, grid_size)
        GX, GY = np.meshgrid(gx, gy)
        try:
            grid_val = griddata(
                rx_xy, likelihood, (GX, GY), method="linear", fill_value=np.nan
            )
            grid_val_n = griddata(
                rx_xy, likelihood, (GX, GY), method="nearest"
            )
            grid_val[np.isnan(grid_val)] = grid_val_n[np.isnan(grid_val)]
            im = ax.imshow(
                grid_val,
                origin="lower",
                extent=[xmin, xmax, ymin, ymax],
                cmap="viridis",
                alpha=0.55,
                zorder=2,
            )
            cb2 = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.06, location="right")
            cb2.set_label("Likelihood (softmax)", rotation=90, fontsize=9)
        except Exception as e:  # noqa: BLE001
            print(f"  [viz] 2D interp 실패 ({e}); skip.")

    # (4) RX 점 — 더 진하게
    norm_l = Normalize(vmin=likelihood.min(), vmax=likelihood.max())
    ax.scatter(
        rx_xy[:, 0], rx_xy[:, 1],
        c=likelihood, cmap="viridis",
        norm=norm_l,
        edgecolor="black", linewidth=0.6,
        s=70, zorder=4,
        label=f"DB RX (n={rx_xy.shape[0]})",
    )

    # (5) Query (GT) 마커
    if query_xy is not None:
        ax.scatter(
            [query_xy[0]], [query_xy[1]],
            marker="*", s=420,
            c="red", edgecolor="black", linewidth=1.2,
            zorder=6, label="Query / GT (rx0)",
        )

    # (6) argmax 추정 위치 마커
    if rx_xy.shape[0] > 0:
        i_hat = int(np.argmax(likelihood))
        ax.scatter(
            [rx_xy[i_hat, 0]], [rx_xy[i_hat, 1]],
            marker="X", s=260,
            c="lime", edgecolor="black", linewidth=1.2,
            zorder=5, label=f"argmax → rx{i_hat}",
        )
        if query_xy is not None:
            err = float(np.linalg.norm(rx_xy[i_hat] - query_xy))
        else:
            err = float("nan")
    else:
        err = float("nan")

    # (7) TX 마커
    ax.scatter(
        [tx_xyz[0]], [tx_xyz[1]],
        marker="^", s=320,
        c="orange", edgecolor="black", linewidth=1.2,
        zorder=6, label=f"TX (BS) z={tx_xyz[2]:.1f}m",
    )

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(
        f"P1U  UE Fingerprint Localization Heatmap  —  metric: {METRIC_LABELS[metric_name]}\n"
        f"localization error (argmax) = {err:.2f} m"
    )
    ax.legend(loc="lower left", framealpha=0.85, fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.3)

    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [save] {out_path.name}")


def draw_metric_comparison(
    rx_xy: np.ndarray,
    distances_per_metric: Dict[str, np.ndarray],
    likelihoods_per_metric: Dict[str, np.ndarray],
    query_xy: Optional[np.ndarray],
    out_path: Path,
) -> None:
    """5 metric × 2 row (distance, likelihood) 비교."""
    n_m = len(METRIC_NAMES)
    fig, axes = plt.subplots(2, n_m, figsize=(4.0 * n_m, 7.5), constrained_layout=True)
    for col, m in enumerate(METRIC_NAMES):
        d = distances_per_metric[m]
        lik = likelihoods_per_metric[m]

        ax_d = axes[0, col]
        if query_xy is not None:
            geo_d = np.linalg.norm(rx_xy - query_xy[None, :], axis=1)
            ax_d.scatter(geo_d, d, s=14, alpha=0.7)
            ax_d.set_xlabel("|geodesic distance to query|  (m)")
        else:
            ax_d.plot(d, marker=".", linestyle="-")
            ax_d.set_xlabel("DB rx index")
        ax_d.set_ylabel(METRIC_LABELS[m])
        ax_d.set_title(m)
        ax_d.grid(True, linestyle=":", alpha=0.4)

        ax_l = axes[1, col]
        ax_l.bar(np.arange(rx_xy.shape[0]), lik, color="C0", alpha=0.85)
        ax_l.set_xlabel("DB rx index")
        ax_l.set_ylabel("likelihood")
        ax_l.grid(True, linestyle=":", alpha=0.4)

    fig.suptitle("P1U  Metric Comparison  (top: metric vs geo dist  /  bottom: likelihood)",
                 fontsize=12)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [save] {out_path.name}")


# ============================================================
# Main pipeline
# ============================================================
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--obj", type=Path, default=DEFAULT_OBJ)
    parser.add_argument("--query", type=int, default=DEFAULT_QUERY_RX,
                        help="Query RX index (0..N_pos-1).")
    parser.add_argument("--tx-x", type=float, default=DEFAULT_TX_XYZ[0])
    parser.add_argument("--tx-y", type=float, default=DEFAULT_TX_XYZ[1])
    parser.add_argument("--tx-z", type=float, default=DEFAULT_TX_XYZ[2])
    parser.add_argument("--fc-ghz", type=float, default=DEFAULT_FC_GHZ)
    parser.add_argument("--n-bins", type=int, default=DEFAULT_PDP_BINS)
    parser.add_argument("--obj-sample", type=int, default=DEFAULT_OBJ_VERTEX_SAMPLE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--no-2d-interp", action="store_true",
                        help="2D 보간 끄고 점만 표시 (1D line 데이터에서 더 명확).")
    parser.add_argument("--temperature", type=float, default=None,
                        help="softmax temperature (None=median 자동).")
    args = parser.parse_args(argv)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"  P1U_UE_Fingerprint_Localization v1  (TS={ts})")
    print("=" * 70)
    print(f"  npz   : {args.npz}")
    print(f"  obj   : {args.obj}")
    print(f"  TX    : ({args.tx_x:.3f}, {args.tx_y:.3f}, {args.tx_z:.3f}) m")
    print(f"  fc    : {args.fc_ghz} GHz")
    print(f"  query : rx{args.query}")
    print(f"  out   : {out_dir}")
    print()

    # 1) NPZ
    ch = load_channel_data(args.npz)
    n_pos = ch.rx_positions_xy.shape[0]
    if not (0 <= args.query < n_pos):
        sys.exit(f"[ERROR] query rx{args.query} 가 위치 있는 RX 범위 [0, {n_pos-1}] 밖입니다.")
    query_xy = ch.rx_positions_xy[args.query]
    print(f"  query rx{args.query} 위치: ({query_xy[0]:.3f}, {query_xy[1]:.3f})")
    print(f"  target_rx_index (npz)    : {ch.target_rx_index}")
    if args.query != ch.target_rx_index:
        print(f"  [warn] CLI query ({args.query}) ≠ npz target_rx_index ({ch.target_rx_index})")

    # 2) PDP build
    pdp_builder = PDPBuilder(n_bins=args.n_bins, tau_clip_ns=DEFAULT_PDP_CLIP_NS)
    pdp_builder.fit(ch.tau_list)
    pdps = np.stack([pdp_builder.build(ch.tau_list[i], ch.power_list[i])
                     for i in range(n_pos)])
    print(f"  PDPs shape: {pdps.shape}  (n_pos × n_bins)")

    # 3) metric: query vs all DB
    db_indices = [i for i in range(n_pos) if i != args.query]
    distances_per_metric: Dict[str, np.ndarray] = {m: np.zeros(len(db_indices)) for m in METRIC_NAMES}
    pdp_q = pdps[args.query]
    support = pdp_builder.bin_centers
    for k, j in enumerate(db_indices):
        m_dict = compute_all_metrics(pdp_q, pdps[j], support)
        for m in METRIC_NAMES:
            distances_per_metric[m][k] = m_dict[m]

    # 4) likelihood (softmax)
    likelihoods_per_metric = {
        m: softmax_likelihood(distances_per_metric[m], args.temperature)
        for m in METRIC_NAMES
    }

    # 5) load obj (top-view)
    obj_verts = load_obj_vertices(args.obj, n_sample=args.obj_sample)

    # 6) per-metric heatmap 저장
    db_xy = ch.rx_positions_xy[db_indices]
    for m in METRIC_NAMES:
        out_png = out_dir / f"P1U_heatmap_{m}_{ts}.png"
        draw_topview_with_heatmap(
            obj_verts=obj_verts,
            rx_xy=db_xy,
            likelihood=likelihoods_per_metric[m],
            query_xy=query_xy,
            tx_xyz=(args.tx_x, args.tx_y, args.tx_z),
            metric_name=m,
            metric_value_query_to_db=distances_per_metric[m],
            out_path=out_png,
            interpolate_2d=not args.no_2d_interp,
        )

    # 7) metric 비교 plot
    cmp_png = out_dir / f"P1U_metric_comparison_{ts}.png"
    draw_metric_comparison(
        rx_xy=db_xy,
        distances_per_metric=distances_per_metric,
        likelihoods_per_metric=likelihoods_per_metric,
        query_xy=query_xy,
        out_path=cmp_png,
    )

    # 8) summary CSV
    sum_csv = out_dir / f"P1U_summary_{ts}.csv"
    with open(sum_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["db_rx_idx", "x", "y"]
                   + [f"d_{m}" for m in METRIC_NAMES]
                   + [f"L_{m}" for m in METRIC_NAMES])
        for k, j in enumerate(db_indices):
            row = [j, ch.rx_positions_xy[j, 0], ch.rx_positions_xy[j, 1]]
            row += [distances_per_metric[m][k] for m in METRIC_NAMES]
            row += [likelihoods_per_metric[m][k] for m in METRIC_NAMES]
            w.writerow(row)
    print(f"  [save] {sum_csv.name}")

    # 9) localization error
    err_csv = out_dir / f"P1U_localization_error_{ts}.csv"
    with open(err_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "argmax_rx_idx", "argmax_x", "argmax_y",
                    "gt_x", "gt_y", "error_m"])
        for m in METRIC_NAMES:
            i_hat = int(np.argmax(likelihoods_per_metric[m]))
            j_hat = db_indices[i_hat]
            xy_hat = ch.rx_positions_xy[j_hat]
            err = float(np.linalg.norm(xy_hat - query_xy))
            w.writerow([m, j_hat, xy_hat[0], xy_hat[1],
                        query_xy[0], query_xy[1], err])
            print(f"  [{m:<11}] argmax → rx{j_hat:>3}  err = {err:7.2f} m")
    print(f"  [save] {err_csv.name}")

    # 10) log
    log_path = SCRIPT_DIR / "P1U_UE_Fingerprint_Localization_2605v1.log"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(
            f"{ts} - P1U_UE_Fingerprint_Localization_2605v1.py - "
            f"실행 완료 (npz={args.npz.name}, obj={args.obj.name}, "
            f"query=rx{args.query}, n_pos={n_pos}, fc={args.fc_ghz}GHz)\n"
        )
    print(f"  [save] log appended: {log_path.name}")
    print("\n[done]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
