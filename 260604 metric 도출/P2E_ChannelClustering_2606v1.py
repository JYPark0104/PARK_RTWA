# -*- coding: utf-8 -*-
"""
P2E_ChannelClustering_2606v1.py
===============================
[ 목적 ]
P2D 의 채널 거리(Bures-Cov + 1D-Wasserstein-PDP, trace=1 정규화, ρ≈0.60)를 기반으로
6G Digital Twin user grouping(채널이 유사한 UE 묶기) 을 수행한다.
커스텀 메트릭이므로 **사전계산 거리행렬(N×N)** 을 받는 군집화를 사용:
  - K-medoids (자체 구현; K-means 의 메트릭 일반화)
  - Spectral clustering (유사도 그래프)
  - Agglomerative (precomputed, average linkage)
Silhouette(precomputed)로 최적 k 자동 선택 후 물리지도(x,y)에 색칠.

[ 채널 거리 행렬 (벡터화) ]
  - Bures-shape : D_b(i,j)=Σ(√a_i−√a_j)² = 2 − 2·(√a_i·√a_j)  (BS별 합, a=APS/Tr)
  - PDP-Wass    : Σ_k |CDF_i−CDF_j|  (BS별 cityblock, CDF=cumsum(정규화 PDP))
  - D = γ·minmax(Bures) + δ·minmax(Wass),  γ:δ = 0.77:0.23 (P2D 최적)

----------------------------------------------------------------------
실행 환경
  - Python 3.10.12 / dclcom61 / numpy·scipy·scikit-learn·matplotlib
  - 데이터 : 260512 RT Result Data/channel_data_260531_GHM_Twin_v0_1.npz (TX3, SISO)
----------------------------------------------------------------------
"""

import os
import csv
import datetime

import numpy as np
from scipy.spatial.distance import cdist

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from sklearn.cluster import SpectralClustering, AgglomerativeClustering
from sklearn.metrics import silhouette_score

for _fp in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]:
    if os.path.exists(_fp):
        font_manager.fontManager.addfont(_fp)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=_fp).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

NPZ = "/home/dclcom61/twin_minji/260512 RT Result Data/channel_data_260531_GHM_Twin_v0_1.npz"
OUTDIR = "/home/dclcom61/twin_minji/260604 metric 도출/P2E_ChannelClustering_Results"
N_BS = 3
N_PDP = 32
N_APS = 48
MIN_PATHS = 10
N_NODE = 1000
GAMMA, DELTA = 0.77, 0.23     # P2D 최적 비중 (정규화 항)
K_RANGE = list(range(2, 11))
SEED = 0
EPS = 1e-30

os.makedirs(OUTDIR, exist_ok=True)
STAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def lin(p_db):
    return 10.0 ** (np.asarray(p_db, dtype=np.float64) / 10.0)


def load_build():
    d = np.load(NPZ, allow_pickle=True)
    rxp = np.asarray(d["rx_positions"], float)
    txp = np.asarray(d["tx_positions"], float)
    N = rxp.shape[0]
    npaths = np.array([[len(d[f"tau_tx{T}_rx{i}"]) for T in range(N_BS)] for i in range(N)])
    ok = np.where(np.all(npaths >= MIN_PATHS, axis=1))[0]
    rng = np.random.default_rng(SEED)
    sub = np.sort(rng.choice(ok, min(N_NODE, len(ok)), replace=False))
    n = len(sub)
    coords = rxp[sub]

    aoa_edges = np.linspace(-180, 180, N_APS + 1)
    tmax = np.zeros(N_BS)
    for T in range(N_BS):
        allt = np.concatenate([np.asarray(d[f"tau_tx{T}_rx{i}"], float) for i in sub])
        tmax[T] = float(np.quantile(allt, 0.999))

    APSn = np.zeros((n, N_BS, N_APS))
    CDF = np.zeros((n, N_BS, N_PDP))
    for k, i in enumerate(sub):
        for T in range(N_BS):
            p = lin(d[f"power_tx{T}_rx{i}"])
            a = np.asarray(d[f"aoa_tx{T}_rx{i}"], float)
            t = np.asarray(d[f"tau_tx{T}_rx{i}"], float)
            aps = np.zeros(N_APS); pdp = np.zeros(N_PDP)
            if p.size and p.sum() > 0:
                ba = np.clip(np.digitize(a, aoa_edges) - 1, 0, N_APS - 1)
                np.add.at(aps, ba, p)
                tau_edges = np.linspace(0, tmax[T], N_PDP + 1)
                bt = np.clip(np.digitize(np.clip(t, 0, tmax[T] * (1 - 1e-9)), tau_edges) - 1,
                             0, N_PDP - 1)
                np.add.at(pdp, bt, p)
            APSn[k, T] = aps / (aps.sum() + EPS)        # trace=1
            CDF[k, T] = np.cumsum(pdp / (pdp.sum() + EPS))
    return coords, txp, APSn, CDF


def channel_distance_matrix(APSn, CDF):
    n = APSn.shape[0]
    B = np.zeros((n, n)); W = np.zeros((n, n))
    for T in range(N_BS):
        S = np.sqrt(APSn[:, T, :])              # (n,48)
        G = S @ S.T
        B += np.clip(2.0 - 2.0 * G, 0, None)    # Bures-shape (Hellinger)
        W += cdist(CDF[:, T, :], CDF[:, T, :], metric="cityblock")
    iu = np.triu_indices(n, 1)
    def mm(M):
        v = M[iu]; lo, hi = v.min(), v.max()
        return (M - lo) / (hi - lo + 1e-12)
    D = GAMMA * mm(B) + DELTA * mm(W)
    D = (D + D.T) / 2.0
    np.fill_diagonal(D, 0.0)
    return D


def kmedoids(D, k, seed=0, n_init=8, max_iter=120):
    """사전계산 거리행렬 기반 K-medoids(PAM 근사, Voronoi 반복)."""
    n = D.shape[0]
    rng = np.random.default_rng(seed)
    best_labels, best_cost = None, np.inf
    for _ in range(n_init):
        med = rng.choice(n, k, replace=False)
        for _ in range(max_iter):
            labels = np.argmin(D[:, med], axis=1)
            new = med.copy()
            for c in range(k):
                mem = np.where(labels == c)[0]
                if len(mem) == 0:
                    continue
                new[c] = mem[np.argmin(D[np.ix_(mem, mem)].sum(1))]
            if np.array_equal(new, med):
                break
            med = new
        cost = D[np.arange(n), med[labels]].sum()
        if cost < best_cost:
            best_cost, best_labels = cost, labels
    return best_labels


def cluster_all(D, k):
    out = {}
    out["K-medoids"] = kmedoids(D, k, seed=SEED)
    sigma = np.median(D[np.triu_indices(D.shape[0], 1)]) + 1e-9
    sim = np.exp(-(D ** 2) / (2 * sigma ** 2))
    out["Spectral"] = SpectralClustering(n_clusters=k, affinity="precomputed",
                                         assign_labels="discretize",
                                         random_state=SEED).fit_predict(sim)
    out["Agglomerative"] = AgglomerativeClustering(n_clusters=k, metric="precomputed",
                                                   linkage="average").fit_predict(D)
    return out


def main():
    print("[load/build] ...")
    coords, txp, APSn, CDF = load_build()
    n = coords.shape[0]
    print(f"[build] n={n}, BS={N_BS}")
    D = channel_distance_matrix(APSn, CDF)
    print(f"[dist] D {D.shape}, mean={D[np.triu_indices(n,1)].mean():.3f}")

    methods = ["K-medoids", "Spectral", "Agglomerative"]
    sil = {m: [] for m in methods}
    labels_by_k = {m: {} for m in methods}
    for k in K_RANGE:
        cl = cluster_all(D, k)
        for m in methods:
            lb = cl[m]
            s = silhouette_score(D, lb, metric="precomputed") if len(set(lb)) > 1 else -1
            sil[m].append(s); labels_by_k[m][k] = lb
        print(f"  k={k}  " + "  ".join(f"{m}={sil[m][-1]:+.3f}" for m in methods))

    best_k = {m: K_RANGE[int(np.argmax(sil[m]))] for m in methods}
    print("\n=== 최적 k (silhouette) ===")
    for m in methods:
        print(f"  {m:14s}: k={best_k[m]}  silhouette={max(sil[m]):+.3f}")

    # ---------------------------------------------------------------- 시각화
    fig, ax = plt.subplots(2, 2, figsize=(14.5, 12))
    cols = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

    # (0,0) silhouette vs k
    for m in methods:
        ax[0, 0].plot(K_RANGE, sil[m], "-o", lw=2, label=m)
    ax[0, 0].set_xlabel("클러스터 수 k", fontsize=11)
    ax[0, 0].set_ylabel("Silhouette score (precomputed)", fontsize=11)
    ax[0, 0].set_title("(a) k 선택 — Silhouette vs k", fontsize=12)
    ax[0, 0].grid(alpha=0.3); ax[0, 0].legend(fontsize=10)

    # 나머지 3패널: 각 방법 best k 물리지도
    panel = [(0, 1), (1, 0), (1, 1)]
    for (pos, m) in zip(panel, methods):
        k = best_k[m]; lb = labels_by_k[m][k]
        for c in range(k):
            mm = lb == c
            ax[pos].scatter(coords[mm, 0], coords[mm, 1], s=14, color=cols[c % 10],
                            alpha=0.8, edgecolors="none", label=f"C{c}")
        ax[pos].scatter(txp[:, 0], txp[:, 1], marker="*", s=320, c="black",
                        edgecolors="white", linewidths=1.2, zorder=5, label="BS")
        ax[pos].set_title(f"({'bcd'[panel.index(pos)]}) {m}  (k={k}, "
                          f"silhouette={max(sil[m]):+.3f})", fontsize=12)
        ax[pos].set_xlabel("x (m)", fontsize=10); ax[pos].set_ylabel("y (m)", fontsize=10)
        ax[pos].set_aspect("equal", "datalim"); ax[pos].grid(alpha=0.3)
        ax[pos].legend(fontsize=7, ncol=2, loc="best")

    fig.suptitle("6G Digital Twin 채널-거리 기반 User Grouping  "
                 "(Bures-Cov + PDP-Wasserstein, trace=1, GHM 3-BS, n=%d)" % n, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fpng = os.path.join(OUTDIR, f"P2E_clustering_{STAMP}.png")
    fig.savefig(fpng, dpi=140)
    print(f"[save] {fpng}")

    fcsv = os.path.join(OUTDIR, f"P2E_clustering_{STAMP}.csv")
    with open(fcsv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "k"] + [f"sil_k{k}" for k in K_RANGE] + ["best_k", "best_sil"])
        for m in methods:
            w.writerow([m, ""] + [f"{s:.4f}" for s in sil[m]]
                       + [best_k[m], f"{max(sil[m]):.4f}"])
    print(f"[save] {fcsv}")


if __name__ == "__main__":
    main()
