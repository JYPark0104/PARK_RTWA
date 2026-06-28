# -*- coding: utf-8 -*-
"""
P2F_AngularSectorization_2606v1.py
==================================
[ 목적 ]
P2E 클러스터가 연속 채널 장이라 흐릿했던 문제를, **방위(AoA) 섹터화**로 가장 뚜렷하게
만든다. AoA 가 지배 신호이므로 채널의 '지배 방위각(APS 원형평균)' 으로 묶으면 깔끔한
부채꼴 섹터가 형성된다. 또한 채널-방위 섹터가 실제 기하 방위와 일치하는지 검증해
"채널이 위치(섹터)를 복원함" 을 보인다.

[ 방법 ]
  - BS별 지배 방위각 : θ_T = atan2( Σ aps_k sinφ_k , Σ aps_k cosφ_k ) (APS 원형평균)
    집중도 R_T = |Σ aps_k e^{jφ_k}| ∈ [0,1] (낮으면 방위 불명확).
  - 군집 특징 : [R_T cosθ_T, R_T sinθ_T] (T=0,1,2) → 6차원, K-means.
  - 최적 k : silhouette(euclidean) k=3~10.
  - 검증 : 채널 θ(BS0) vs 기하 방위 β0=atan2(y-y0,x-x0) 정렬(원형상관),
           채널 군집 vs 기하 방위 섹터 ARI.

----------------------------------------------------------------------
실행 환경 : Python 3.10.12 / dclcom61 / numpy·scikit-learn·matplotlib
데이터    : 260512 RT Result Data/channel_data_260531_GHM_Twin_v0_1.npz (TX3, SISO)
----------------------------------------------------------------------
"""

import os
import csv
import datetime

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment

for _fp in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]:
    if os.path.exists(_fp):
        font_manager.fontManager.addfont(_fp)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=_fp).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

NPZ = "/home/dclcom61/twin_minji/260512 RT Result Data/channel_data_260531_GHM_Twin_v0_1.npz"
OUTDIR = "/home/dclcom61/twin_minji/260604 metric 도출/P2F_AngularSectorization_Results"
N_BS = 3
N_APS = 48
MIN_PATHS = 10
N_NODE = 1500
K_RANGE = list(range(3, 11))
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
    coords = rxp[sub]

    centers = np.deg2rad(-180 + 360 * (np.arange(N_APS) + 0.5) / N_APS)
    cos_c, sin_c = np.cos(centers), np.sin(centers)
    aoa_edges = np.linspace(-180, 180, N_APS + 1)

    n = len(sub)
    theta = np.zeros((n, N_BS))      # 지배 방위각(rad)
    conc = np.zeros((n, N_BS))       # 집중도 R
    for k, i in enumerate(sub):
        for T in range(N_BS):
            p = lin(d[f"power_tx{T}_rx{i}"])
            a = np.asarray(d[f"aoa_tx{T}_rx{i}"], float)
            aps = np.zeros(N_APS)
            if p.size and p.sum() > 0:
                ba = np.clip(np.digitize(a, aoa_edges) - 1, 0, N_APS - 1)
                np.add.at(aps, ba, p)
                aps /= aps.sum()
            C = (aps * cos_c).sum(); S = (aps * sin_c).sum()
            theta[k, T] = np.arctan2(S, C)
            conc[k, T] = np.hypot(C, S)
    return coords, txp, theta, conc


def main():
    print("[load/build] ...")
    coords, txp, theta, conc = load_build()
    n = coords.shape[0]
    print(f"[build] n={n}, BS={N_BS}")

    # 군집 특징 : [R cosθ, R sinθ] x 3 BS
    feat = np.zeros((n, 2 * N_BS))
    for T in range(N_BS):
        feat[:, 2 * T] = conc[:, T] * np.cos(theta[:, T])
        feat[:, 2 * T + 1] = conc[:, T] * np.sin(theta[:, T])

    sil = []
    labs = {}
    for k in K_RANGE:
        lb = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit_predict(feat)
        s = silhouette_score(feat, lb)
        sil.append(s); labs[k] = lb
        print(f"  k={k}  silhouette={s:+.3f}")
    best_k = K_RANGE[int(np.argmax(sil))]
    lb_ch = labs[best_k]
    print(f"[best] k={best_k}, silhouette={max(sil):+.3f}")

    # 기하 방위 군집 : 채널과 동일 방식(3-BS 방위 특징 + K-means) 공정 비교
    beta0 = np.arctan2(coords[:, 1] - txp[0, 1], coords[:, 0] - txp[0, 0])
    geo_feat = np.zeros((n, 2 * N_BS))
    for T in range(N_BS):
        bT = np.arctan2(coords[:, 1] - txp[T, 1], coords[:, 0] - txp[T, 0])
        geo_feat[:, 2 * T] = np.cos(bT)
        geo_feat[:, 2 * T + 1] = np.sin(bT)
    geo_sect = KMeans(n_clusters=best_k, n_init=10, random_state=SEED).fit_predict(geo_feat)
    ari = adjusted_rand_score(lb_ch, geo_sect)

    # 색 맞추기 : 채널 군집 라벨 → 기하 섹터에 최대 겹침으로 Hungarian 재정렬
    conf = np.zeros((best_k, best_k), dtype=int)
    for a, b in zip(lb_ch, geo_sect):
        conf[a, b] += 1
    row, col = linear_sum_assignment(-conf)          # 채널라벨 row → 기하라벨 col
    remap = {int(r): int(c) for r, c in zip(row, col)}
    lb_ch_m = np.array([remap[int(l)] for l in lb_ch])
    match_rate = float(np.mean(lb_ch_m == geo_sect)) * 100.0
    print(f"[match] Hungarian 라벨 정렬 후 색 일치율 = {match_rate:.1f}%")
    # 채널 방위(BS0) vs 기하 방위 정렬
    th0 = theta[:, 0]
    def circ_corr(a, b):   # Jammalamadaka 원형상관
        a = a - np.arctan2(np.sin(a).mean(), np.cos(a).mean())
        b = b - np.arctan2(np.sin(b).mean(), np.cos(b).mean())
        return float((np.sin(a) * np.sin(b)).sum() /
                     (np.sqrt((np.sin(a) ** 2).sum() * (np.sin(b) ** 2).sum()) + 1e-12))
    rcc = circ_corr(th0, beta0)
    # 정렬도 : (θ-β) 합벡터 길이 R_align ∈[0,1] (일정 오프셋으로 추종하면 높음)
    diff = th0 - beta0
    R_align = float(np.hypot(np.cos(diff).mean(), np.sin(diff).mean()))
    offset = float(np.rad2deg(np.arctan2(np.sin(diff).mean(), np.cos(diff).mean())))
    print(f"[verify] ARI={ari:+.3f}, 원형상관={rcc:+.3f}, "
          f"정렬도 R_align={R_align:.3f} (오프셋 {offset:+.0f}°)")

    # ---------------------------------------------------------------- 시각화
    cols = plt.cm.tab10(np.linspace(0, 1, 10))
    fig, ax = plt.subplots(2, 2, figsize=(14.5, 12))

    # (a) 채널 AoA K-means 섹터 물리지도 (색은 (b)에 맞춰 Hungarian 정렬)
    for c in range(best_k):
        m = lb_ch_m == c
        ax[0, 0].scatter(coords[m, 0], coords[m, 1], s=16, color=cols[c % 10],
                         alpha=0.85, edgecolors="none")
    ax[0, 0].scatter(txp[:, 0], txp[:, 1], marker="*", s=340, c="black",
                     edgecolors="white", linewidths=1.3, zorder=5)
    ax[0, 0].set_title(f"(a) 채널 방위(AoA) 섹터  k={best_k} "
                       f"(silhouette={max(sil):+.3f}, 색 일치율 {match_rate:.0f}%)",
                       fontsize=12)
    ax[0, 0].set_xlabel("x (m)"); ax[0, 0].set_ylabel("y (m)")
    ax[0, 0].set_aspect("equal", "datalim"); ax[0, 0].grid(alpha=0.3)

    # (b) 기하 방위 섹터(BS0)
    for c in range(best_k):
        m = geo_sect == c
        ax[0, 1].scatter(coords[m, 0], coords[m, 1], s=16, color=cols[c % 10],
                         alpha=0.85, edgecolors="none")
    ax[0, 1].scatter(txp[:, 0], txp[:, 1], marker="*", s=340, c="black",
                     edgecolors="white", linewidths=1.3, zorder=5)
    ax[0, 1].set_title(f"(b) 기하 방위 군집 (3-BS, 동일 K-means k={best_k}) — 공정 참조",
                       fontsize=12)
    ax[0, 1].set_xlabel("x (m)"); ax[0, 1].set_ylabel("y (m)")
    ax[0, 1].set_aspect("equal", "datalim"); ax[0, 1].grid(alpha=0.3)

    # (c) 정렬 검증 scatter
    ax[1, 0].scatter(np.rad2deg(beta0), np.rad2deg(th0), s=10, alpha=0.35,
                     c="#1f4e79", edgecolors="none")
    ax[1, 0].set_xlabel("기하 방위 β0 (deg, BS0→RX)")
    ax[1, 0].set_ylabel("채널 지배 방위 θ (deg, BS0)")
    ax[1, 0].set_title(f"(c) 채널 방위 vs 기하 방위  정렬도 R={R_align:.2f} "
                       f"(오프셋 {offset:+.0f}°), 원형상관={rcc:+.2f}", fontsize=12)
    ax[1, 0].grid(alpha=0.3)

    # (d) silhouette vs k
    ax[1, 1].plot(K_RANGE, sil, "-o", lw=2, color="#1b9e77")
    ax[1, 1].axvline(best_k, ls="--", color="#d62728", label=f"best k={best_k}")
    ax[1, 1].set_xlabel("클러스터 수 k"); ax[1, 1].set_ylabel("Silhouette (AoA 특징)")
    ax[1, 1].set_title(f"(d) k 선택 — Silhouette\nARI={ari:+.3f}, 색 일치율={match_rate:.0f}%",
                       fontsize=12)
    ax[1, 1].grid(alpha=0.3); ax[1, 1].legend(fontsize=10)

    fig.suptitle("6G Digital Twin 방위(AoA) 섹터화 — 채널 방위 군집이 지리 섹터를 복원  "
                 f"(GHM 3-BS, n={n})", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fpng = os.path.join(OUTDIR, f"P2F_sectorization_{STAMP}.png")
    fig.savefig(fpng, dpi=140)
    print(f"[save] {fpng}")

    fcsv = os.path.join(OUTDIR, f"P2F_sectorization_{STAMP}.csv")
    with open(fcsv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["k"] + list(K_RANGE))
        w.writerow(["silhouette"] + [f"{s:.4f}" for s in sil])
        w.writerow(["best_k", best_k]); w.writerow(["best_silhouette", f"{max(sil):.4f}"])
        w.writerow(["ARI_channel_vs_geo", f"{ari:.4f}"])
        w.writerow(["circ_corr_theta_vs_beta", f"{rcc:.4f}"])
        w.writerow(["R_align_theta_minus_beta", f"{R_align:.4f}"])
        w.writerow(["offset_deg", f"{offset:.1f}"])
        w.writerow(["color_match_rate_pct", f"{match_rate:.1f}"])
    print(f"[save] {fcsv}")


if __name__ == "__main__":
    main()
