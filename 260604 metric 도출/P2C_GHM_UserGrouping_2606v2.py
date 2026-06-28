# -*- coding: utf-8 -*-
"""
P2C_GHM_UserGrouping_2606v2.py
==============================
v1 결론(HYBRID-L1 메트릭으로 거친 그룹핑은 가능하나 세밀 그룹핑은 ARI 하락)을
보완하기 위해, '방위(섹터) 기반 특징'을 추가하여 세밀 그룹핑 성능을 끌어올린다.

[ 핵심 아이디어 ]
  - 다중경로 환경에서 지배적 신호는 AoA(방위). 사용자별로 각 TX 에서 본
    '전력가중 평균 방위각(원형평균)' 을 구하면, 3개 BS 의 방위 삼각형이
    사용자의 물리 위치를 강하게 규정한다(다변측위 = 방위 교차).
  - 따라서 (θ0, θ1, θ2) 방위 벡터로 군집화하면 v1 의 분포형 메트릭보다
    공간 정합(ARI) 이 개선될 것으로 기대.

[ 비교 군집화 방식 (모두 정답=물리좌표 KMeans 와 ARI/NMI 비교) ]
  (M0) HYBRID-L1  : v1 메트릭 거리행렬(Spectral)               -- 베이스라인
  (M1) Sector-AoA : 3 TX 원형평균 방위각의 원형거리 L2 (Spectral)
  (M2) Sector-KM  : [cosθ,sinθ]×3 = (n,6) 특징에 KMeans         -- 자연 섹터 분할
  (M3) Sector+Range KM : 위 방위특징 + 표준화 RSRP(n,3) 결합 KMeans -- 방위+거리

[ 평가 ]
  - ARI / NMI (vs 물리 KMeans 정답), K=2..10 스윕.
  - random 라벨 베이스라인(≈0) 동시 표기.
  - 대표 K 의 공간 산점도(정답 + M0/M1/M3) 로 공간 정합 육안 확인.

----------------------------------------------------------------------
실행 환경
  - Python   : 3.10.12
  - 실행 서버 : dclcom61 (deepgadget)
  - 라이브러리: numpy / scipy / scikit-learn(KMeans,Spectral,Agglomerative) / matplotlib (CPU)
  - 연산규모  : N(샘플) 900, 거리행렬 900x900, K 스윕 2..10
  - 입력      : channel_data_260531_GHM_Twin_v0_1.npz (TX 3, RX 5071, SISO)
변경 사유(2026-06-04)
  - v1 대비 '방위(섹터) 기반' 특징(M1~M3) 추가: 세밀 그룹핑(큰 K) ARI 개선 목적.
----------------------------------------------------------------------
"""

import os
import csv
import datetime

import numpy as np
from scipy.stats import rankdata

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# 한글 폰트 (Noto Sans CJK KR)
for _fp in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]:
    if os.path.exists(_fp):
        font_manager.fontManager.addfont(_fp)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=_fp).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

# ---------------------------------------------------------------- 설정
NPZ = "/home/dclcom61/twin_minji/260512 RT Result Data/channel_data_260531_GHM_Twin_v0_1.npz"
OUTDIR = "/home/dclcom61/twin_minji/260604 metric 도출/P2C_GHM_UserGrouping_Results"
N_TX = 3
N_SAMPLE = 900
MIN_PATHS = 10
N_AOA = 48
PRIMARY_TX = 0
W_STEP = 0.05
K_LIST = list(range(2, 11))
K_SHOW = 6
SEED = 0

os.makedirs(OUTDIR, exist_ok=True)
STAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


# ---------------------------------------------------------------- 공통(P2B/v1 재사용)
def lin(p_db):
    return 10.0 ** (np.asarray(p_db, dtype=np.float64) / 10.0)


def load():
    d = np.load(NPZ, allow_pickle=True)
    rxp = np.asarray(d["rx_positions"], dtype=np.float64)
    txp = np.asarray(d["tx_positions"], dtype=np.float64)
    N = rxp.shape[0]
    npaths = np.array([
        [len(d[f"tau_tx{T}_rx{i}"]) for T in range(N_TX)] for i in range(N)
    ])
    return d, rxp, txp, N, npaths


def stratified_sample(rxp, txp, npaths, n_want):
    ok = np.all(npaths >= MIN_PATHS, axis=1)
    idx = np.where(ok)[0]
    rad = np.sqrt(((rxp[idx] - txp[PRIMARY_TX, :2]) ** 2).sum(1))
    rng = np.random.default_rng(SEED)
    bins = np.quantile(rad, np.linspace(0, 1, 11))
    per = max(1, n_want // 10)
    picks = []
    for b in range(10):
        m = (rad >= bins[b]) & (rad <= bins[b + 1] if b == 9 else rad < bins[b + 1])
        cand = idx[m]
        if len(cand) == 0:
            continue
        picks.append(rng.choice(cand, min(per, len(cand)), replace=False))
    return np.sort(np.concatenate(picks))


def build_features(d, sub):
    """
    샘플별:
      - APS  (N_TX, n, N_AOA)  : 분포형 메트릭(M0)용
      - rsrp_multi (n, N_TX)   : 거리(range) 보강(M3)용
      - ang  (n, N_TX)         : 전력가중 평균 방위각(원형평균, rad) -- 섹터 특징(M1~M3)
    """
    n = len(sub)
    aoa_edges = np.linspace(-180, 180, N_AOA + 1)
    APS = np.zeros((N_TX, n, N_AOA))
    rsrp_multi = np.zeros((n, N_TX))
    ang = np.zeros((n, N_TX))
    for k, i in enumerate(sub):
        for T in range(N_TX):
            p = lin(d[f"power_tx{T}_rx{i}"])
            a = np.asarray(d[f"aoa_tx{T}_rx{i}"], float)
            if p.size:
                ba = np.clip(np.digitize(a, aoa_edges) - 1, 0, N_AOA - 1)
                np.add.at(APS[T, k], ba, p)
                ar = np.deg2rad(a)
                # 전력가중 원형평균 방위각
                ang[k, T] = np.arctan2((p * np.sin(ar)).sum(),
                                       (p * np.cos(ar)).sum())
            rsrp_multi[k, T] = 10 * np.log10(p.sum() + 1e-30)
    APS /= APS.sum(-1, keepdims=True) + 1e-30
    return APS, rsrp_multi, ang


def wass1d(D):
    c = np.cumsum(D, axis=-1)
    return np.abs(c[:, None, :] - c[None, :, :]).sum(-1)


def mm(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-12)


def fast_spearman_vs(ref_rank, x):
    rx = rankdata(x)
    rx = rx - rx.mean()
    denom = np.sqrt((rx * rx).sum() * (ref_rank * ref_rank).sum())
    return float((rx * ref_rank).sum() / (denom + 1e-30))


def simplex_grid(step):
    g = []
    k = int(round(1.0 / step))
    for a in range(k + 1):
        for b in range(k + 1 - a):
            c = k - a - b
            g.append((a / k, b / k, c / k))
    return np.array(g)


# ---------------------------------------------------------------- 거리행렬 / 특징
def build_hybrid_distance(APS, rsrp_multi, pos):
    """v1 과 동일한 HYBRID(L1) 거리행렬(full n×n) — 베이스라인(M0)."""
    n = pos.shape[0]
    iu = np.triu_indices(n, 1)
    pf = np.sqrt(((pos[:, None] - pos[None]) ** 2).sum(-1))[iu]
    pf_rank = rankdata(pf); pf_rank = pf_rank - pf_rank.mean()
    Daos = [mm(wass1d(APS[T])[iu]) for T in range(N_TX)]
    D0, D1, D2 = Daos
    best_w = (-9.0, None)
    for (w0, w1, w2) in simplex_grid(W_STEP):
        r = fast_spearman_vs(pf_rank, w0 * D0 + w1 * D1 + w2 * D2)
        if r > best_w[0]:
            best_w = (r, (w0, w1, w2))
    w = best_w[1]
    Dm_aoa = mm(w[0] * D0 + w[1] * D1 + w[2] * D2)
    D_rsrp = mm(np.sqrt(((rsrp_multi[:, None, :] - rsrp_multi[None, :, :]) ** 2).sum(-1))[iu])
    best_a = (-9.0, 0.0)
    for a in np.linspace(0, 1, 101):
        r = fast_spearman_vs(pf_rank, a * Dm_aoa + (1 - a) * D_rsrp)
        if r > best_a[0]:
            best_a = (r, a)
    Dm_hyb = mm(best_a[1] * Dm_aoa + (1 - best_a[1]) * D_rsrp)
    Dfull = np.zeros((n, n)); Dfull[iu] = Dm_hyb; Dfull = Dfull + Dfull.T
    return Dfull, dict(w=w, alpha=best_a[1], rho=best_a[0])


def build_sector_distance(ang):
    """3 TX 원형평균 방위각의 원형거리 L2 (full n×n, rad). — M1 용."""
    n = ang.shape[0]
    Dsec = np.zeros((n, n))
    for T in range(N_TX):
        A = ang[:, T]
        diff = A[:, None] - A[None, :]
        dcirc = np.abs((diff + np.pi) % (2 * np.pi) - np.pi)  # [0, pi]
        Dsec += dcirc ** 2
    return np.sqrt(Dsec)


def sector_features(ang):
    """방위각 → [cosθ, sinθ]×3 = (n,6). 원형성 보존 KMeans 특징. — M2/M3 용."""
    return np.concatenate([np.cos(ang), np.sin(ang)], axis=1)


def zscore(x):
    return (x - x.mean(0)) / (x.std(0) + 1e-12)


# ---------------------------------------------------------------- 군집화 헬퍼
def spectral_from_dist(Dfull, K):
    sigma = np.median(Dfull[Dfull > 0]) + 1e-12
    A = np.exp(-(Dfull ** 2) / (2 * sigma ** 2))
    return SpectralClustering(n_clusters=K, affinity="precomputed",
                              assign_labels="kmeans",
                              random_state=SEED).fit_predict(A)


def kmeans_feat(F, K):
    return KMeans(n_clusters=K, n_init=10, random_state=SEED).fit_predict(F)


def main():
    print("[load] GHM Twin npz ...")
    d, rxp, txp, N, npaths = load()
    sub = stratified_sample(rxp, txp, npaths, N_SAMPLE)
    n = len(sub)
    pos = rxp[sub]
    print(f"[sample] n={n}")

    APS, rsrp_multi, ang = build_features(d, sub)

    print("[metric] M0 HYBRID-L1 거리행렬 ...")
    D_hyb, hinfo = build_hybrid_distance(APS, rsrp_multi, pos)
    print(f"  HYBRID-L1: w={hinfo['w']}, alpha={hinfo['alpha']:.2f}, rho={hinfo['rho']:.3f}")

    print("[metric] M1 Sector 원형거리 행렬 ...")
    D_sec = build_sector_distance(ang)

    F_sec = sector_features(ang)                         # (n,6)  M2
    F_secrange = np.concatenate(                          # (n,6+3) M3
        [zscore(F_sec), zscore(rsrp_multi)], axis=1)

    # ---- K 스윕 ----
    print("[sweep] K=2..10 군집화/평가 ...")
    methods = ["M0_HYBRID", "M1_Sector", "M2_SectorKM", "M3_Sector+Range"]
    curves = {m: {"ari": [], "nmi": []} for m in methods}
    rand_ari = []
    rows = []
    store = {}
    rng = np.random.default_rng(SEED)

    for K in K_LIST:
        labels_gt = KMeans(n_clusters=K, n_init=10, random_state=SEED).fit_predict(pos)
        lab = {
            "M0_HYBRID":       spectral_from_dist(D_hyb, K),
            "M1_Sector":       spectral_from_dist(D_sec, K),
            "M2_SectorKM":     kmeans_feat(F_sec, K),
            "M3_Sector+Range": kmeans_feat(F_secrange, K),
        }
        rlab = rng.integers(0, K, size=n)
        rand_ari.append(adjusted_rand_score(labels_gt, rlab))
        for m in methods:
            ari = adjusted_rand_score(labels_gt, lab[m])
            nmi = normalized_mutual_info_score(labels_gt, lab[m])
            curves[m]["ari"].append(ari)
            curves[m]["nmi"].append(nmi)
            rows.append([K, m, f"{ari:.4f}", f"{nmi:.4f}"])
        if K == K_SHOW:
            store = dict(gt=labels_gt, **lab)
        print(f"  K={K:2d} | " + " | ".join(
            f"{m.split('_')[0]}:ARI={curves[m]['ari'][-1]:+.3f}" for m in methods)
            + f" | rand={rand_ari[-1]:+.3f}")

    # ---------------------------------------------------------------- 콘솔 요약
    print("\n=== v2 그룹핑 비교 요약 (GHM Twin) ===")
    for m in methods:
        a = curves[m]["ari"]
        print(f"  {m:16s} ARI: max={max(a):.3f}(K={K_LIST[int(np.argmax(a))]}) "
              f"mean={np.mean(a):.3f}")
    print(f"  random ARI mean: {np.mean(rand_ari):+.3f}")

    # ---------------------------------------------------------------- 시각화
    fig = plt.figure(figsize=(18, 9.5))
    gs = fig.add_gridspec(2, 4, hspace=0.34, wspace=0.30)
    fig.suptitle(
        f"GHM Twin | User Grouping v2: 방위(섹터) 특징 추가 (n={n})  "
        f"[정답=물리좌표 KMeans]", fontsize=14)

    mcolor = {"M0_HYBRID": "#9e9e9e", "M1_Sector": "#1565c0",
              "M2_SectorKM": "#2e7d32", "M3_Sector+Range": "#c62828"}

    # (0,0) ARI 곡선
    axA = fig.add_subplot(gs[0, 0])
    for m in methods:
        axA.plot(K_LIST, curves[m]["ari"], "-o", ms=4, color=mcolor[m], label=m)
    axA.plot(K_LIST, rand_ari, "--", color="k", lw=0.8, label="random")
    axA.axhline(0, color="k", lw=0.5)
    axA.set_title("ARI vs K", fontsize=10)
    axA.set_xlabel("K (군집 수)", fontsize=9)
    axA.set_ylabel("Adjusted Rand Index", fontsize=9)
    axA.grid(alpha=0.3); axA.legend(fontsize=7.5, loc="best")

    # (0,1) NMI 곡선
    axN = fig.add_subplot(gs[0, 1])
    for m in methods:
        axN.plot(K_LIST, curves[m]["nmi"], "-o", ms=4, color=mcolor[m], label=m)
    axN.set_title("NMI vs K", fontsize=10)
    axN.set_xlabel("K (군집 수)", fontsize=9)
    axN.set_ylabel("Normalized Mutual Info", fontsize=9)
    axN.grid(alpha=0.3); axN.legend(fontsize=7.5, loc="best")

    # (0,2) 방위 특징 산점도: θ(TX0) vs θ(TX1), 색=K_SHOW M3 라벨
    axP = fig.add_subplot(gs[0, 2])
    scP = axP.scatter(np.rad2deg(ang[:, 0]), np.rad2deg(ang[:, 1]),
                      c=store["M3_Sector+Range"], cmap="tab10", s=12)
    axP.set_title(f"방위 특징공간 θ0-θ1 (M3 라벨, K={K_SHOW})", fontsize=10)
    axP.set_xlabel("θ TX0 [deg]", fontsize=9)
    axP.set_ylabel("θ TX1 [deg]", fontsize=9)
    axP.grid(alpha=0.3); axP.tick_params(labelsize=7)

    # (0,3) best-method 요약 막대 (각 방식 max ARI)
    axB = fig.add_subplot(gs[0, 3])
    maxari = [max(curves[m]["ari"]) for m in methods]
    bars = axB.bar(range(len(methods)), maxari,
                   color=[mcolor[m] for m in methods])
    axB.set_xticks(range(len(methods)))
    axB.set_xticklabels([m.replace("_", "\n") for m in methods], fontsize=7)
    axB.set_ylabel("max ARI (over K)", fontsize=9)
    axB.set_title("방식별 최고 ARI", fontsize=10)
    axB.grid(axis="y", alpha=0.3)
    for b, v in zip(bars, maxari):
        axB.text(b.get_x() + b.get_width() / 2, v + 0.005,
                 f"{v:.3f}", ha="center", va="bottom", fontsize=7.5)

    # (1,0~3) 공간 산점도: 정답 + M0/M1/M3
    panel = [("정답 그룹 (물리 KMeans)", "gt"),
             ("M0 HYBRID-L1", "M0_HYBRID"),
             ("M1 Sector-AoA", "M1_Sector"),
             ("M3 Sector+Range", "M3_Sector+Range")]
    for j, (ttl, key) in enumerate(panel):
        ax = fig.add_subplot(gs[1, j])
        ax.scatter(pos[:, 0], pos[:, 1], c=store[key], cmap="tab10", s=12)
        if key == "gt":
            sub_ttl = f"{ttl}\n(K={K_SHOW})"
        else:
            ari_here = curves[key]["ari"][K_LIST.index(K_SHOW)]
            sub_ttl = f"{ttl}\n(물리좌표, K={K_SHOW}, ARI={ari_here:.3f})"
        ax.set_title(sub_ttl, fontsize=10)
        ax.set_xlabel("x [m]", fontsize=9); ax.set_ylabel("y [m]", fontsize=9)
        ax.set_aspect("equal", "datalim"); ax.tick_params(labelsize=7)

    fpng = os.path.join(OUTDIR, f"P2C_v2_Sector_{STAMP}.png")
    fig.savefig(fpng, dpi=130, bbox_inches="tight")
    print(f"[save] {fpng}")

    # ---------------------------------------------------------------- CSV
    fcsv = os.path.join(OUTDIR, f"P2C_v2_Sector_{STAMP}.csv")
    with open(fcsv, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["K", "method", "ARI", "NMI"])
        wr.writerows(rows)
        wr.writerow([])
        wr.writerow(["random_ARI_mean", f"{np.mean(rand_ari):.4f}", "", ""])
        for m in methods:
            wr.writerow([f"{m}_maxARI", f"{max(curves[m]['ari']):.4f}",
                         f"K={K_LIST[int(np.argmax(curves[m]['ari']))]}", ""])
    print(f"[save] {fcsv}")


if __name__ == "__main__":
    main()
