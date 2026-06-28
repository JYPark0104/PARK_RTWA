# -*- coding: utf-8 -*-
"""
P1U_UE_Fingerprint_Localization_2605v2.py
==========================================
UE Fingerprint Localization Heatmap (Area-1 grid, dense 2D, PDP + AoA/AoD)
==========================================================================

목적
----
P1B 필터를 거쳐 저장된 NLoS-only 채널 데이터
`Area1_7.5GHz_Rays_Valid_RXs.npz` (1,037 RX, max 400 paths each, with AoA+AoD)
와 P1A 의 Area 1 grid 정의 (x: linspace(-136.138, 58.862, 40),
y: linspace(-117.667, 77.333, 40), z: 1.5) 를 결합하여,
단일 query RX 의 위치를 fingerprinting 으로 추정하고 dense 2D heatmap 을
산출합니다.

v1 (channel_data_GHMTwin2_cutting.npz, 42 위치, SISO PDP only) 대비 개선점
--------------------------------------------------------------------------
* RX 1,037 개 모두에 대해 (x, y, z) 환원 → **2D dense heatmap**
* Map_Mesh.obj 와 동일한 scene → 시각화 정합
* Path 별 AoA (theta_r, phi_r) + AoD (theta_t, phi_t) 모두 활용
* PDP 5 metric + Angular 3 metric = 총 8 metric 비교
* counts[i] 또는 tau != 0 mask 로 zero-padded slot 자동 제외

데이터 layout (확인 완료)
-------------------------
* shape : (1037, 1, 1, 1, 1, 400) → SISO, max_paths=400, zero-padded
* counts[i] : i 번째 RX 의 valid (nonzero) path 개수
* rx_indices : 원본 1600 grid 의 flat 인덱스 (40×40×1, 행=y 외부 / 열=x 내부)
* los_nlos_flag : NLoS only 필터링 후라 모두 0 (= NLoS, not padding)
* zero-padding 위치 : 일반적으로 앞 부분 (tau == 0 으로 식별 안전)

8 가지 metric
-------------
[A] PDP 기반 (v1 과 동일)
   1. Pearson correlation       (1 - r)
   2. Cosine similarity         (1 - cos)
   3. Wasserstein-1 distance    over tau axis (ns)
   4. Jensen-Shannon divergence (bits)
   5. RMS delay spread diff     (ns)
[B] Angular spectrum 기반 (신규)
   6. AoA cosine                — power-weighted theta_r×phi_r 2D 히스토그램 cosine
   7. AoD cosine                — power-weighted theta_t×phi_t 2D 히스토그램 cosine
   8. Joint AoA+AoD cosine      — AoA + AoD 평탄화 후 가중 평균 cosine

추정/평가
---------
* likelihood = softmax(-d / median(d))
* argmax 위치 vs query GT 위치 거리로 metric 평가

산출물 (P1U_v2_Results/ 자동 생성)
----------------------------------
* P1U_v2_heatmap_<metric>_<TS>.png   (8 장)
* P1U_v2_metric_comparison_<TS>.png  (8 metric 비교)
* P1U_v2_summary_<TS>.csv
* P1U_v2_localization_error_<TS>.csv

실행 환경 (가정)
----------------
* Python  : 3.10 ~ 3.12
* 서버    : 로컬 Mac (혹은 dclcom45/55 같은 공용 서버)
* 라이브러리:
    - numpy        >= 1.24
    - scipy        >= 1.10
    - matplotlib   >= 3.7

실행 방법
---------
$ python P1U_UE_Fingerprint_Localization_2605v2.py            # 기본값 (query=중앙 근처)
$ python P1U_UE_Fingerprint_Localization_2605v2.py --query 500
$ python P1U_UE_Fingerprint_Localization_2605v2.py \
       --npz ".../Area1_7.5GHz_Rays_Valid_RXs.npz" \
       --obj ".../Map_Mesh.obj" --query 500

작성 : 2026-05-14
규칙 준수 : CLAUDE.md / AGENTS.md
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
    from scipy.interpolate import griddata as _scipy_griddata
    _HAVE_SCIPY = True
except Exception:  # noqa: BLE001
    _HAVE_SCIPY = False

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

matplotlib.rcParams["font.family"] = ["DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


# ============================================================
# Default config
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent

DEFAULT_NPZ = (WORKSPACE_ROOT
               / "251020 BM QIE (Weichselberger)"
               / "P1B_Valid_Results"
               / "Area1_7.5GHz_Rays_Valid_RXs.npz")
DEFAULT_OBJ = (WORKSPACE_ROOT
               / "260512 RT Result Data"
               / "Map_Mesh.obj")

# Area 1 grid (P1A_RT_to_Rays_2509v6.py 의 AREA_CONFIGS['area_1'] 와 동일)
AREA1_X_PARAMS = {"start": -136.138, "stop": 58.862, "num": 40}
AREA1_Y_PARAMS = {"start": -117.667, "stop":  77.333, "num": 40}
AREA1_Z_VAL = 1.5
AREA1_TX_XYZ = (-51.561, -21.794, 19.0)
AREA1_FC_GHZ = 7.5

DEFAULT_QUERY_RX = 500           # 1,037 중 임의 (중반)
DEFAULT_PDP_BINS = 128
DEFAULT_PDP_CLIP_NS = 2000.0
DEFAULT_OBJ_VERTEX_SAMPLE = 30000
DEFAULT_RESULT_DIR = SCRIPT_DIR / "P1U_v2_Results"
DEFAULT_HEATMAP_GRID = 256
DEFAULT_TEMPERATURE = None

# Angular spectrum 빈
ANG_THETA_BINS = 18      # elevation 0..180  (10 deg step)
ANG_PHI_BINS = 36        # azimuth -180..180 (10 deg step)
ANG_THETA_RANGE = (0.0, 180.0)
ANG_PHI_RANGE = (-180.0, 180.0)


# ============================================================
# Area 1 NPZ 로더 (v2)
# ============================================================
@dataclass
class Area1Channel:
    """Area1_7.5GHz_Rays_Valid_RXs.npz + grid 환원 후 컨테이너."""
    rx_indices: np.ndarray          # (N,) 원본 grid 의 flat idx
    rx_xyz: np.ndarray              # (N, 3) 환원된 (x, y, z)
    counts: np.ndarray              # (N,) valid path 개수
    tau_padded: np.ndarray          # (N, 400) sec
    power_padded: np.ndarray        # (N, 400) linear
    theta_r_padded: np.ndarray      # (N, 400) deg
    phi_r_padded: np.ndarray        # (N, 400) deg
    theta_t_padded: np.ndarray      # (N, 400) deg
    phi_t_padded: np.ndarray        # (N, 400) deg
    los_nlos_padded: np.ndarray     # (N, 400) int
    frequency_ghz: float
    tx_xyz: Tuple[float, float, float]


def _flat_idx_to_xyz(
    flat_idx: int,
    n_x: int,
    n_y: int,
    rx_x: np.ndarray,
    rx_y: np.ndarray,
    z_val: float,
) -> Tuple[float, float, float]:
    """P1A 의 grid 생성 순서 (z 외부, y 중간, x 내부) 기반 환원.

    rx_positions = [
        [x, y, z]
        for z in rx_z_coords       # outer (1 개라 z idx=0)
        for y in rx_y_coords       # middle
        for x in rx_x_coords       # inner
    ]
    → flat = (iz * n_y + iy) * n_x + ix
    """
    ix = flat_idx % n_x
    iy = (flat_idx // n_x) % n_y
    return float(rx_x[ix]), float(rx_y[iy]), float(z_val)


def load_area1_npz(npz_path: Path,
                   tx_xyz: Tuple[float, float, float] = AREA1_TX_XYZ
                   ) -> Area1Channel:
    if not npz_path.exists():
        sys.exit(f"[ERROR] NPZ 파일이 없습니다: {npz_path}")
    print(f"[load] {npz_path}  ({npz_path.stat().st_size/1e6:.1f} MB)")

    data = np.load(npz_path, allow_pickle=True)

    rx_indices = np.asarray(data["rx_indices"], dtype=np.int64)
    counts_raw = np.asarray(data["counts"]).reshape(-1).astype(np.int64)
    n_rx = rx_indices.shape[0]

    def _squeeze(a):
        a = np.asarray(a)
        return a.reshape(n_rx, -1)
    tau   = _squeeze(data["tau"])
    pwr   = _squeeze(data["power"])
    th_r  = _squeeze(data["theta_r_deg"])
    ph_r  = _squeeze(data["phi_r_deg"])
    th_t  = _squeeze(data["theta_t_deg"])
    ph_t  = _squeeze(data["phi_t_deg"])
    los   = _squeeze(data["los_nlos_flag"])
    freq  = float(np.asarray(data.get("frequency_ghz", AREA1_FC_GHZ)).reshape(-1)[0])
    n_paths = tau.shape[1]
    print(f"  N_rx={n_rx}, max_paths={n_paths}, fc={freq} GHz")

    # rx_indices → (x, y, z)
    n_x = AREA1_X_PARAMS["num"]
    n_y = AREA1_Y_PARAMS["num"]
    rx_x = np.linspace(AREA1_X_PARAMS["start"], AREA1_X_PARAMS["stop"], n_x)
    rx_y = np.linspace(AREA1_Y_PARAMS["start"], AREA1_Y_PARAMS["stop"], n_y)
    xyz = np.zeros((n_rx, 3), dtype=np.float64)
    for i, fi in enumerate(rx_indices):
        xyz[i] = _flat_idx_to_xyz(int(fi), n_x, n_y, rx_x, rx_y, AREA1_Z_VAL)
    print(f"  grid: {n_x} × {n_y} × 1 = 1600 (filtered → {n_rx})")
    print(f"  bbox: x[{rx_x[0]:.2f}, {rx_x[-1]:.2f}], "
          f"y[{rx_y[0]:.2f}, {rx_y[-1]:.2f}], z={AREA1_Z_VAL}")

    return Area1Channel(
        rx_indices=rx_indices,
        rx_xyz=xyz,
        counts=counts_raw,
        tau_padded=tau.astype(np.float64),
        power_padded=pwr.astype(np.float64),
        theta_r_padded=th_r.astype(np.float64),
        phi_r_padded=ph_r.astype(np.float64),
        theta_t_padded=th_t.astype(np.float64),
        phi_t_padded=ph_t.astype(np.float64),
        los_nlos_padded=los.astype(np.int64),
        frequency_ghz=freq,
        tx_xyz=tx_xyz,
    )


def valid_paths_for_rx(ch: Area1Channel, idx: int) -> Dict[str, np.ndarray]:
    """tau != 0 mask 로 valid path 만 추출."""
    mask = ch.tau_padded[idx] != 0.0
    return {
        "tau":     ch.tau_padded[idx][mask],
        "power":   ch.power_padded[idx][mask],
        "theta_r": ch.theta_r_padded[idx][mask],
        "phi_r":   ch.phi_r_padded[idx][mask],
        "theta_t": ch.theta_t_padded[idx][mask],
        "phi_t":   ch.phi_t_padded[idx][mask],
    }


# ============================================================
# Builders
# ============================================================
class PDPBuilder:
    def __init__(self, n_bins: int = 128, tau_clip_ns: float = 2000.0):
        self.n_bins = n_bins
        self.tau_clip_ns = tau_clip_ns
        self.bin_edges: Optional[np.ndarray] = None
        self.bin_centers: Optional[np.ndarray] = None
        self._scale = 1.0

    @staticmethod
    def _power_to_linear(power: np.ndarray) -> np.ndarray:
        if power.size == 0:
            return power
        if np.all(power <= 0) and float(np.min(power)) < -10.0:
            return np.power(10.0, power / 10.0)
        return power

    def fit(self, all_tau_lists: List[np.ndarray]) -> None:
        valid_tau_list = [t[t != 0] for t in all_tau_lists if t.size > 0]
        all_tau = (np.concatenate(valid_tau_list)
                   if valid_tau_list else np.array([0.0, 1.0]))
        q99 = float(np.quantile(all_tau, 0.99)) if all_tau.size > 0 else 0.0
        if q99 > 0 and q99 < 1e-3:
            self._scale = 1e9
            unit = "sec → ns"
        elif q99 >= 1e3:
            self._scale = 1.0 / 0.299792458
            unit = "m → ns"
        else:
            self._scale = 1.0
            unit = "ns (그대로)"
        all_tau_ns = np.clip(all_tau * self._scale, 0.0, self.tau_clip_ns)
        tau_min, tau_max = 0.0, float(all_tau_ns.max() if all_tau_ns.size > 0 else 1.0)
        self.bin_edges = np.linspace(tau_min, tau_max + 1e-9, self.n_bins + 1)
        self.bin_centers = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])
        print(f"  [PDP] tau q99={q99:.3e} → unit={unit}, "
              f"bin range [{tau_min:.2f}, {tau_max:.2f}] ns, n_bins={self.n_bins}")

    def build(self, tau: np.ndarray, power: np.ndarray) -> np.ndarray:
        if tau.size == 0 or power.size == 0:
            return np.full(self.n_bins, 1.0 / self.n_bins, dtype=np.float64)
        mask = (tau != 0) & (power != 0) & np.isfinite(tau) & np.isfinite(power)
        if not np.any(mask):
            return np.full(self.n_bins, 1.0 / self.n_bins, dtype=np.float64)
        tau = tau[mask]; power = power[mask]
        power_lin = self._power_to_linear(power)
        tau_ns = np.clip(tau * self._scale, 0.0, self.tau_clip_ns)
        pdp, _ = np.histogram(tau_ns, bins=self.bin_edges, weights=power_lin)
        s = pdp.sum()
        return pdp / s if s > 0 else np.full(self.n_bins, 1.0 / self.n_bins)


class AngularSpectrumBuilder:
    """power-weighted 2D (theta, phi) histogram → flat L1-normalized vector."""

    def __init__(self,
                 n_theta: int = ANG_THETA_BINS,
                 n_phi: int = ANG_PHI_BINS,
                 theta_range: Tuple[float, float] = ANG_THETA_RANGE,
                 phi_range: Tuple[float, float] = ANG_PHI_RANGE):
        self.n_theta = n_theta
        self.n_phi = n_phi
        self.theta_edges = np.linspace(theta_range[0], theta_range[1], n_theta + 1)
        self.phi_edges = np.linspace(phi_range[0], phi_range[1], n_phi + 1)

    @property
    def n_bins_total(self) -> int:
        return self.n_theta * self.n_phi

    @staticmethod
    def _power_to_linear(power: np.ndarray) -> np.ndarray:
        if power.size == 0:
            return power
        if np.all(power <= 0) and float(np.min(power)) < -10.0:
            return np.power(10.0, power / 10.0)
        return power

    def build(self, theta_deg: np.ndarray, phi_deg: np.ndarray,
              power: np.ndarray) -> np.ndarray:
        if theta_deg.size == 0:
            return np.full(self.n_bins_total, 1.0 / self.n_bins_total)
        mask = (power != 0) & np.isfinite(power) & np.isfinite(theta_deg) & np.isfinite(phi_deg)
        if not np.any(mask):
            return np.full(self.n_bins_total, 1.0 / self.n_bins_total)
        theta_v = theta_deg[mask]; phi_v = phi_deg[mask]
        power_lin = self._power_to_linear(power[mask])
        theta_clip = np.clip(theta_v,
                             self.theta_edges[0] + 1e-6,
                             self.theta_edges[-1] - 1e-6)
        phi_clip = np.clip(phi_v,
                           self.phi_edges[0] + 1e-6,
                           self.phi_edges[-1] - 1e-6)
        H, _, _ = np.histogram2d(theta_clip, phi_clip,
                                 bins=[self.theta_edges, self.phi_edges],
                                 weights=power_lin)
        flat = H.ravel()
        s = flat.sum()
        return flat / s if s > 0 else np.full(self.n_bins_total, 1.0 / self.n_bins_total)


# ============================================================
# Metrics
# ============================================================
class MetricLib:
    @staticmethod
    def pearson_dist(p, q):
        if p.std() == 0 or q.std() == 0:
            return 1.0
        r = float(np.corrcoef(p, q)[0, 1])
        return 1.0 if not np.isfinite(r) else (1.0 - r)

    @staticmethod
    def cosine_dist(p, q):
        npn = np.linalg.norm(p); nqn = np.linalg.norm(q)
        if npn == 0 or nqn == 0:
            return 1.0
        return 1.0 - float(np.dot(p, q) / (npn * nqn))

    @staticmethod
    def wasserstein_dist(p, q, support):
        if _HAVE_SCIPY:
            return float(_scipy_wd(support, support, u_weights=p, v_weights=q))
        return float(np.sum(np.abs(np.cumsum(p) - np.cumsum(q))) * (support[1] - support[0]))

    @staticmethod
    def jsd_dist(p, q):
        if _HAVE_SCIPY:
            return float(_scipy_jsd(p + 1e-12, q + 1e-12, base=2))
        m = 0.5 * (p + q) + 1e-12
        kl_pm = np.sum(p * np.log2((p + 1e-12) / m))
        kl_qm = np.sum(q * np.log2((q + 1e-12) / m))
        return float(0.5 * (kl_pm + kl_qm))

    @staticmethod
    def tau_rms_diff(p, q, support):
        def rms(w, s):
            sw = w.sum()
            if sw <= 0: return 0.0
            mu = float(np.sum(s * w) / sw)
            var = float(np.sum(((s - mu) ** 2) * w) / sw)
            return float(np.sqrt(max(var, 0.0)))
        return abs(rms(p, support) - rms(q, support))


METRIC_NAMES = [
    "pdp_pearson", "pdp_cosine", "pdp_wasserstein", "pdp_jsd", "tau_rms",
    "aoa_cosine", "aod_cosine", "joint_aoa_aod_cosine",
]
METRIC_LABELS = {
    "pdp_pearson":          "PDP Pearson  (1 - r)",
    "pdp_cosine":           "PDP Cosine   (1 - cos)",
    "pdp_wasserstein":      "PDP Wasserstein (ns)",
    "pdp_jsd":              "PDP JSD (bits)",
    "tau_rms":              "RMS delay diff (ns)",
    "aoa_cosine":           "AoA spectrum cosine (1 - cos)",
    "aod_cosine":           "AoD spectrum cosine (1 - cos)",
    "joint_aoa_aod_cosine": "Joint AoA+AoD cosine (1 - cos)",
}


def compute_all_metrics(pdp_q, pdp_d, support_pdp,
                        aoa_q, aoa_d, aod_q, aod_d) -> Dict[str, float]:
    joint_q = np.concatenate([aoa_q, aod_q])
    joint_d = np.concatenate([aoa_d, aod_d])
    return {
        "pdp_pearson":          MetricLib.pearson_dist(pdp_q, pdp_d),
        "pdp_cosine":           MetricLib.cosine_dist(pdp_q, pdp_d),
        "pdp_wasserstein":      MetricLib.wasserstein_dist(pdp_q, pdp_d, support_pdp),
        "pdp_jsd":              MetricLib.jsd_dist(pdp_q, pdp_d),
        "tau_rms":              MetricLib.tau_rms_diff(pdp_q, pdp_d, support_pdp),
        "aoa_cosine":           MetricLib.cosine_dist(aoa_q, aoa_d),
        "aod_cosine":           MetricLib.cosine_dist(aod_q, aod_d),
        "joint_aoa_aod_cosine": MetricLib.cosine_dist(joint_q, joint_d),
    }


# ============================================================
# softmax
# ============================================================
def softmax_likelihood(distances: np.ndarray, temperature: Optional[float]) -> np.ndarray:
    d = np.asarray(distances, dtype=np.float64)
    finite = d[np.isfinite(d)]
    if finite.size == 0:
        return np.full_like(d, 1.0 / d.size)
    tau = max(float(np.median(finite)) if temperature is None else float(temperature), 1e-9)
    z = -d / tau
    z = z - np.nanmax(z)
    w = np.exp(z)
    w[~np.isfinite(w)] = 0.0
    s = w.sum()
    return w / s if s > 0 else np.full_like(w, 1.0 / w.size)


# ============================================================
# OBJ vertex 로더
# ============================================================
def load_obj_vertices(obj_path: Path, n_sample: int = 30000) -> np.ndarray:
    if not obj_path.exists():
        print(f"  [obj] WARNING: {obj_path} 없음 → mesh overlay skip.")
        return np.empty((0, 3), dtype=np.float64)
    print(f"[obj] {obj_path.name} ({obj_path.stat().st_size/1e6:.1f} MB) 읽는 중...")
    t0 = time.time()
    xs, ys, zs = [], [], []
    with open(obj_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                p = line.split()
                xs.append(float(p[1])); ys.append(float(p[2])); zs.append(float(p[3]))
    verts = np.column_stack([xs, ys, zs])
    print(f"  [obj] {len(xs):,} vertices in {time.time()-t0:.1f}s")
    if n_sample is not None and verts.shape[0] > n_sample:
        idx = np.random.RandomState(2605).choice(verts.shape[0], n_sample, replace=False)
        verts = verts[idx]
        print(f"  [obj] sub-sample → {verts.shape[0]:,}")
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
    out_path: Path,
    grid_size: int = DEFAULT_HEATMAP_GRID,
) -> float:
    fig, ax = plt.subplots(figsize=(11.5, 9.5), constrained_layout=True)

    # (1) Map_Mesh top-view
    if obj_verts.shape[0] > 0:
        sc = ax.scatter(obj_verts[:, 0], obj_verts[:, 1],
                        c=obj_verts[:, 2], cmap="Greys",
                        s=0.3, alpha=0.30, rasterized=True)
        cb = plt.colorbar(sc, ax=ax, fraction=0.025, pad=0.02, location="right")
        cb.set_label("z (m) — building height", rotation=90, fontsize=9)
        xmin, xmax = obj_verts[:, 0].min(), obj_verts[:, 0].max()
        ymin, ymax = obj_verts[:, 1].min(), obj_verts[:, 1].max()
    else:
        xmin, ymin = rx_xy.min(0); xmax, ymax = rx_xy.max(0)
        m = 30.0; xmin -= m; xmax += m; ymin -= m; ymax += m

    # (2) 2D 보간 heatmap
    if _HAVE_SCIPY and rx_xy.shape[0] >= 4:
        gx = np.linspace(xmin, xmax, grid_size)
        gy = np.linspace(ymin, ymax, grid_size)
        GX, GY = np.meshgrid(gx, gy)
        try:
            grid_val = _scipy_griddata(rx_xy, likelihood, (GX, GY),
                                       method="cubic", fill_value=np.nan)
            grid_nn = _scipy_griddata(rx_xy, likelihood, (GX, GY), method="nearest")
            grid_val[np.isnan(grid_val)] = grid_nn[np.isnan(grid_val)]
            im = ax.imshow(grid_val, origin="lower",
                           extent=[xmin, xmax, ymin, ymax],
                           cmap="viridis", alpha=0.55, zorder=2)
            cb2 = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.06, location="right")
            cb2.set_label("Likelihood (softmax)", rotation=90, fontsize=9)
        except Exception as e:  # noqa: BLE001
            print(f"  [viz] 2D interp 실패 ({e}); skip.")

    # (3) RX dots
    norm_l = Normalize(vmin=likelihood.min(), vmax=likelihood.max())
    ax.scatter(rx_xy[:, 0], rx_xy[:, 1],
               c=likelihood, cmap="viridis", norm=norm_l,
               edgecolor="black", linewidth=0.4, s=22, zorder=4)

    # (4) Query GT
    if query_xy is not None:
        ax.scatter([query_xy[0]], [query_xy[1]], marker="*", s=420,
                   c="red", edgecolor="black", linewidth=1.2,
                   zorder=6, label="Query / GT")

    # (5) argmax
    err = float("nan")
    if rx_xy.shape[0] > 0:
        i_hat = int(np.argmax(likelihood))
        ax.scatter([rx_xy[i_hat, 0]], [rx_xy[i_hat, 1]],
                   marker="X", s=240, c="lime",
                   edgecolor="black", linewidth=1.2, zorder=5,
                   label=f"argmax (rx#{i_hat})")
        if query_xy is not None:
            err = float(np.linalg.norm(rx_xy[i_hat] - query_xy))

    # (6) TX
    ax.scatter([tx_xyz[0]], [tx_xyz[1]], marker="^", s=320,
               c="orange", edgecolor="black", linewidth=1.2, zorder=6,
               label=f"TX (BS) z={tx_xyz[2]:.1f}m")

    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"P1U_v2  metric: {METRIC_LABELS[metric_name]}"
                 f"\nlocalization error (argmax) = {err:.2f} m")
    ax.legend(loc="lower left", framealpha=0.85, fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.3)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [save] {out_path.name}  err={err:.2f} m")
    return err


def draw_metric_comparison(
    rx_xy: np.ndarray,
    distances_per_metric: Dict[str, np.ndarray],
    likelihoods_per_metric: Dict[str, np.ndarray],
    query_xy: Optional[np.ndarray],
    out_path: Path,
) -> None:
    n_m = len(METRIC_NAMES)
    n_col = 4
    n_row = (n_m + n_col - 1) // n_col
    fig, axes = plt.subplots(n_row * 2, n_col,
                             figsize=(4.0 * n_col, 3.6 * n_row * 2),
                             constrained_layout=True)
    axes = axes.reshape(n_row * 2, n_col)
    if query_xy is not None:
        geo_d = np.linalg.norm(rx_xy - query_xy[None, :], axis=1)
    else:
        geo_d = np.arange(rx_xy.shape[0])
    for k, m in enumerate(METRIC_NAMES):
        r = (k // n_col) * 2
        c = k % n_col
        ax_d = axes[r, c]
        ax_l = axes[r + 1, c]
        ax_d.scatter(geo_d, distances_per_metric[m], s=8, alpha=0.5)
        ax_d.set_title(m); ax_d.set_xlabel("|geo dist to query| (m)")
        ax_d.set_ylabel(METRIC_LABELS[m]); ax_d.grid(True, linestyle=":", alpha=0.4)
        ax_l.scatter(geo_d, likelihoods_per_metric[m], s=8, alpha=0.5, color="C2")
        ax_l.set_xlabel("|geo dist to query| (m)"); ax_l.set_ylabel("likelihood")
        ax_l.grid(True, linestyle=":", alpha=0.4)
    # 빈 칸 끄기
    for k in range(len(METRIC_NAMES), n_row * n_col):
        r = (k // n_col) * 2; c = k % n_col
        axes[r, c].axis("off"); axes[r + 1, c].axis("off")
    fig.suptitle("P1U_v2  Metric Comparison "
                 "(top: distance vs geo  /  bottom: likelihood vs geo)", fontsize=12)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [save] {out_path.name}")


# ============================================================
# Main
# ============================================================
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--obj", type=Path, default=DEFAULT_OBJ)
    parser.add_argument("--query", type=int, default=DEFAULT_QUERY_RX)
    parser.add_argument("--tx-x", type=float, default=AREA1_TX_XYZ[0])
    parser.add_argument("--tx-y", type=float, default=AREA1_TX_XYZ[1])
    parser.add_argument("--tx-z", type=float, default=AREA1_TX_XYZ[2])
    parser.add_argument("--n-bins", type=int, default=DEFAULT_PDP_BINS)
    parser.add_argument("--obj-sample", type=int, default=DEFAULT_OBJ_VERTEX_SAMPLE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--temperature", type=float, default=None)
    args = parser.parse_args(argv)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"  P1U_UE_Fingerprint_Localization v2  (TS={ts})")
    print("=" * 72)
    print(f"  npz   : {args.npz}")
    print(f"  obj   : {args.obj}")
    print(f"  TX    : ({args.tx_x:.3f}, {args.tx_y:.3f}, {args.tx_z:.3f}) m")
    print(f"  query : rx#{args.query}")
    print(f"  out   : {out_dir}")
    print()

    # 1) load
    ch = load_area1_npz(args.npz, tx_xyz=(args.tx_x, args.tx_y, args.tx_z))
    n_rx = ch.rx_xyz.shape[0]
    if not (0 <= args.query < n_rx):
        sys.exit(f"[ERROR] query {args.query} out of range [0, {n_rx - 1}]")
    query_xyz = ch.rx_xyz[args.query]
    print(f"  query rx#{args.query} → ({query_xyz[0]:.2f}, "
          f"{query_xyz[1]:.2f}, {query_xyz[2]:.2f}) m,  "
          f"counts={ch.counts[args.query]}")

    # 2) build PDP / angular spectra
    print("\n[fp] building fingerprints ...")
    pdp_b = PDPBuilder(n_bins=args.n_bins, tau_clip_ns=DEFAULT_PDP_CLIP_NS)
    all_tau_valid = [valid_paths_for_rx(ch, i)["tau"] for i in range(n_rx)]
    pdp_b.fit(all_tau_valid)
    aoa_b = AngularSpectrumBuilder()
    aod_b = AngularSpectrumBuilder()

    pdps = np.zeros((n_rx, pdp_b.n_bins))
    aoas = np.zeros((n_rx, aoa_b.n_bins_total))
    aods = np.zeros((n_rx, aod_b.n_bins_total))
    for i in range(n_rx):
        d = valid_paths_for_rx(ch, i)
        pdps[i] = pdp_b.build(d["tau"], d["power"])
        aoas[i] = aoa_b.build(d["theta_r"], d["phi_r"], d["power"])
        aods[i] = aod_b.build(d["theta_t"], d["phi_t"], d["power"])
    print(f"  PDPs:  {pdps.shape}")
    print(f"  AoA spectra: {aoas.shape}  (theta {aoa_b.n_theta} × phi {aoa_b.n_phi})")
    print(f"  AoD spectra: {aods.shape}")

    # 3) metric: query vs all DB
    db_indices = [i for i in range(n_rx) if i != args.query]
    distances_per_metric: Dict[str, np.ndarray] = {
        m: np.zeros(len(db_indices)) for m in METRIC_NAMES
    }
    pdp_q = pdps[args.query]; aoa_q = aoas[args.query]; aod_q = aods[args.query]
    support = pdp_b.bin_centers
    print("\n[metric] computing ...")
    t0 = time.time()
    for k, j in enumerate(db_indices):
        m_dict = compute_all_metrics(
            pdp_q, pdps[j], support,
            aoa_q, aoas[j], aod_q, aods[j],
        )
        for m in METRIC_NAMES:
            distances_per_metric[m][k] = m_dict[m]
    print(f"  done in {time.time()-t0:.1f}s")

    # 4) likelihood
    likelihoods_per_metric = {
        m: softmax_likelihood(distances_per_metric[m], args.temperature)
        for m in METRIC_NAMES
    }

    # 5) obj
    obj_verts = load_obj_vertices(args.obj, n_sample=args.obj_sample)

    # 6) per-metric heatmap
    db_xy = ch.rx_xyz[db_indices][:, :2]
    print("\n[viz] heatmaps ...")
    err_per_metric: Dict[str, float] = {}
    for m in METRIC_NAMES:
        out_png = out_dir / f"P1U_v2_heatmap_{m}_{ts}.png"
        err_per_metric[m] = draw_topview_with_heatmap(
            obj_verts=obj_verts,
            rx_xy=db_xy,
            likelihood=likelihoods_per_metric[m],
            query_xy=query_xyz[:2],
            tx_xyz=(args.tx_x, args.tx_y, args.tx_z),
            metric_name=m,
            out_path=out_png,
        )

    # 7) comparison
    cmp_png = out_dir / f"P1U_v2_metric_comparison_{ts}.png"
    draw_metric_comparison(
        rx_xy=db_xy,
        distances_per_metric=distances_per_metric,
        likelihoods_per_metric=likelihoods_per_metric,
        query_xy=query_xyz[:2],
        out_path=cmp_png,
    )

    # 8) summary CSV
    sum_csv = out_dir / f"P1U_v2_summary_{ts}.csv"
    with open(sum_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["db_rx_idx", "rx_indices_orig", "x", "y", "z"]
                   + [f"d_{m}" for m in METRIC_NAMES]
                   + [f"L_{m}" for m in METRIC_NAMES])
        for k, j in enumerate(db_indices):
            xyz = ch.rx_xyz[j]
            row = [j, int(ch.rx_indices[j]), xyz[0], xyz[1], xyz[2]]
            row += [distances_per_metric[m][k] for m in METRIC_NAMES]
            row += [likelihoods_per_metric[m][k] for m in METRIC_NAMES]
            w.writerow(row)
    print(f"  [save] {sum_csv.name}")

    # 9) localization error CSV
    err_csv = out_dir / f"P1U_v2_localization_error_{ts}.csv"
    with open(err_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "argmax_db_idx", "argmax_x", "argmax_y", "argmax_z",
                    "gt_x", "gt_y", "gt_z", "error_m"])
        print()
        for m in METRIC_NAMES:
            i_hat = int(np.argmax(likelihoods_per_metric[m]))
            j_hat = db_indices[i_hat]
            xyz = ch.rx_xyz[j_hat]
            err = float(np.linalg.norm(xyz - query_xyz))
            w.writerow([m, j_hat, xyz[0], xyz[1], xyz[2],
                        query_xyz[0], query_xyz[1], query_xyz[2], err])
            print(f"  [{m:<22}] argmax → rx#{j_hat:>4}  err = {err:7.2f} m")
    print(f"  [save] {err_csv.name}")

    # 10) log append
    log_path = SCRIPT_DIR / "P1U_UE_Fingerprint_Localization_2605v2.log"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(
            f"{ts} - P1U_UE_Fingerprint_Localization_2605v2.py - "
            f"실행 완료 (npz={args.npz.name}, query=rx#{args.query}, "
            f"n_rx={n_rx}, fc={ch.frequency_ghz}GHz)\n"
        )
    print(f"  [save] log appended: {log_path.name}")
    print("\n[done]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
