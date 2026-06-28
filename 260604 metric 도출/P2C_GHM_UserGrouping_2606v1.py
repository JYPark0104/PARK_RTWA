# -*- coding: utf-8 -*-
"""
P2C_GHM_UserGrouping_2606v1.py
==============================
P2B 에서 도출한 채널 메트릭(특히 HYBRID(L1) = a*AoA_L1 + (1-a)*RSRP)이
'K-clustering(User Grouping)' 에 실제로 쓸 수 있는지를 정량 지표로 검증한다.

[ 핵심 질문 ]
  - 메트릭 거리행렬로 사용자를 군집화하면, 그 군집이 '물리 공간상의 그룹'과
    얼마나 일치하는가? (ρ(상관) 만으로는 군집 품질을 알 수 없으므로 직접 측정)

[ 방법 ]
  1) P2B 와 동일 파이프라인으로 샘플/특징/메트릭(HYBRID-L1) 거리행렬 D 생성.
  2) 정답(GT) 그룹  : 물리좌표 (x,y) 에 KMeans(K) → labels_gt.
  3) 메트릭 그룹    : 거리행렬 D 에
       (a) Agglomerative(average linkage, metric='precomputed')
       (b) Spectral(affinity=exp(-D^2/2σ^2), precomputed)
       (c) KMeans on MDS(2D) 임베딩
     세 방식으로 군집화 → labels_metric.
  4) 평가지표:
       - ARI / NMI : labels_metric vs labels_gt (외부 정합도, 0=무작위 1=완전일치)
       - Silhouette(precomputed D) : 메트릭 공간 내적 군집 응집도
       - 무작위 라벨 ARI : 베이스라인(≈0) 비교
  5) K = 2..10 스윕 곡선 + 대표 K 의 공간 산점도(정답/메트릭 군집 색칠) 시각화.

[ 해석 가이드 ]
  - ARI ≳ 0.3 이면 "거리-비례 그룹핑이 의미 있게 동작",
    ARI ~ 0 이면 "메트릭으로는 자연 그룹을 못 찾음(연속 분포)".
  - Silhouette > 0 이면 메트릭 공간에 군집 구조가 존재.

----------------------------------------------------------------------
실행 환경
  - Python   : 3.10.12
  - 실행 서버 : dclcom61 (deepgadget)
  - 라이브러리: numpy / scipy / scikit-learn(KMeans,Spectral,Agglo,MDS) / matplotlib (CPU)
  - 연산규모  : N(샘플) 900, 거리행렬 900x900, K 스윕 2..10
  - 입력      : channel_data_260531_GHM_Twin_v0_1.npz (TX 3, RX 5071, SISO)
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

from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.manifold import MDS
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)

# 한글 폰트 (Noto Sans CJK KR) : 라벨/제목 글리프 깨짐(tofu) 방지
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
N_TAU = 32
N_AOA = 48
PRIMARY_TX = 0
W_STEP = 0.05            # AoA 가중치 단체격자 간격
K_LIST = list(range(2, 11))   # 군집 수 스윕
K_SHOW = 6              # 공간 산점도로 보여줄 대표 K
SEED = 0

os.makedirs(OUTDIR, exist_ok=True)
STAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


# ---------------------------------------------------------------- 공통(P2B 재사용)
def lin(p_db):
    """dB(m) -> 선형 전력."""
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
    """대표 TX 거리로 10 구간 층화 샘플. 3 TX 모두 유효경로 보유 조건."""
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
    """샘플별 TX 3개 APS / 3개 RSRP((n,3)) 생성."""
    n = len(sub)
    aoa_edges = np.linspace(-180, 180, N_AOA + 1)
    APS = np.zeros((N_TX, n, N_AOA))
    rsrp_multi = np.zeros((n, N_TX))
    for k, i in enumerate(sub):
        for T in range(N_TX):
            p = lin(d[f"power_tx{T}_rx{i}"])
            a = np.asarray(d[f"aoa_tx{T}_rx{i}"], float)
            if p.size:
                ba = np.clip(np.digitize(a, aoa_edges) - 1, 0, N_AOA - 1)
                np.add.at(APS[T, k], ba, p)
            rsrp_multi[k, T] = 10 * np.log10(p.sum() + 1e-30)
    APS /= APS.sum(-1, keepdims=True) + 1e-30
    return APS, rsrp_multi


def wass1d(D):
    """행 히스토그램들의 1D Wasserstein = CDF L1. 반환 (n,n)."""
    c = np.cumsum(D, axis=-1)
    return np.abs(c[:, None, :] - c[None, :, :]).sum(-1)


def mm(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-12)


def fast_spearman_vs(ref_rank, x):
    """ref(고정 rank) 대비 x 의 Spearman = rank-Pearson."""
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


# ---------------------------------------------------------------- 메트릭 거리행렬
def build_hybrid_distance(APS, rsrp_multi, pos):
    """
    P2B 와 동일한 HYBRID(L1) 메트릭 거리행렬(full n×n, [0,1] 정규화)을 생성한다.
      - AoA_L1 : 단체격자에서 ρ(물리거리) 최대화하는 w0,w1,w2 가중합
      - RSRP   : (n,3) 벡터 L2 쌍거리
      - HYBRID : α·AoA_L1 + (1-α)·RSRP, α 자동탐색
    반환 : Dfull(n,n), info(dict)
    """
    n = pos.shape[0]
    iu = np.triu_indices(n, 1)
    pf = np.sqrt(((pos[:, None] - pos[None]) ** 2).sum(-1))[iu]
    pf_rank = rankdata(pf); pf_rank = pf_rank - pf_rank.mean()

    Daos = [mm(wass1d(APS[T])[iu]) for T in range(N_TX)]
    D0, D1, D2 = Daos

    # AoA-L1 가중합 탐색
    grid = simplex_grid(W_STEP)
    best_w = (-9.0, None)
    for (w0, w1, w2) in grid:
        r = fast_spearman_vs(pf_rank, w0 * D0 + w1 * D1 + w2 * D2)
        if r > best_w[0]:
            best_w = (r, (w0, w1, w2))
    w = best_w[1]
    Dm_aoa = mm(w[0] * D0 + w[1] * D1 + w[2] * D2)

    # RSRP (n,3) L2 쌍거리
    D_rsrp = mm(np.sqrt(((rsrp_multi[:, None, :] - rsrp_multi[None, :, :]) ** 2).sum(-1))[iu])

    # HYBRID α 탐색
    best_a = (-9.0, 0.0)
    for a in np.linspace(0, 1, 101):
        r = fast_spearman_vs(pf_rank, a * Dm_aoa + (1 - a) * D_rsrp)
        if r > best_a[0]:
            best_a = (r, a)
    alpha = best_a[1]
    Dm_hyb = mm(alpha * Dm_aoa + (1 - alpha) * D_rsrp)

    # full 대칭 행렬 복원 (대각=0)
    Dfull = np.zeros((n, n))
    Dfull[iu] = Dm_hyb
    Dfull = Dfull + Dfull.T
    info = dict(w=w, alpha=alpha, rho_hybrid=best_a[0], rho_aoa=best_w[0])
    return Dfull, info


# ---------------------------------------------------------------- 군집화/평가
def cluster_metric(Dfull, K, mds_xy):
    """메트릭 거리행렬 D 로 3가지 군집화 → 라벨 dict 반환."""
    labels = {}
    # (a) Agglomerative average linkage (precomputed)
    labels["agglo"] = AgglomerativeClustering(
        n_clusters=K, metric="precomputed", linkage="average"
    ).fit_predict(Dfull)
    # (b) Spectral (affinity from distance)
    sigma = np.median(Dfull[Dfull > 0]) + 1e-12
    A = np.exp(-(Dfull ** 2) / (2 * sigma ** 2))
    labels["spectral"] = SpectralClustering(
        n_clusters=K, affinity="precomputed", assign_labels="kmeans",
        random_state=SEED
    ).fit_predict(A)
    # (c) KMeans on MDS embedding
    labels["kmeans_mds"] = KMeans(
        n_clusters=K, n_init=10, random_state=SEED
    ).fit_predict(mds_xy)
    return labels


def evaluate(Dfull, labels_metric, labels_gt):
    """ARI/NMI(vs GT) + Silhouette(메트릭 공간) 산출."""
    out = {}
    for name, lab in labels_metric.items():
        ari = adjusted_rand_score(labels_gt, lab)
        nmi = normalized_mutual_info_score(labels_gt, lab)
        try:
            sil = silhouette_score(Dfull, lab, metric="precomputed")
        except Exception:
            sil = float("nan")
        out[name] = dict(ari=ari, nmi=nmi, sil=sil)
    return out


def main():
    print("[load] GHM Twin npz ...")
    d, rxp, txp, N, npaths = load()
    sub = stratified_sample(rxp, txp, npaths, N_SAMPLE)
    n = len(sub)
    pos = rxp[sub]
    rad0 = np.sqrt(((pos - txp[PRIMARY_TX, :2]) ** 2).sum(1))
    print(f"[sample] n={n}, dist[{rad0.min():.1f},{rad0.max():.1f}]m")

    APS, rsrp_multi = build_features(d, sub)
    print("[metric] HYBRID(L1) 거리행렬 생성 ...")
    Dfull, info = build_hybrid_distance(APS, rsrp_multi, pos)
    print(f"  w={info['w']}, alpha={info['alpha']:.2f}, "
          f"rho_hybrid={info['rho_hybrid']:.3f}")

    # 메트릭 거리행렬 2D MDS 임베딩 (KMeans 입력 + 시각화 공용)
    print("[mds] 메트릭 거리행렬 2D MDS 임베딩 ...")
    mds_xy = MDS(n_components=2, dissimilarity="precomputed",
                 random_state=SEED, normalized_stress="auto").fit_transform(Dfull)

    rng = np.random.default_rng(SEED)

    # ---- K 스윕 ----
    print("[sweep] K=2..10 군집화/평가 ...")
    rows = []
    curves = {m: {"ari": [], "nmi": [], "sil": []}
              for m in ["agglo", "spectral", "kmeans_mds"]}
    rand_ari = []
    store_labels = {}  # K_SHOW 시각화용
    for K in K_LIST:
        labels_gt = KMeans(n_clusters=K, n_init=10,
                           random_state=SEED).fit_predict(pos)
        labels_metric = cluster_metric(Dfull, K, mds_xy)
        res = evaluate(Dfull, labels_metric, labels_gt)
        # 무작위 라벨 베이스라인
        rlab = rng.integers(0, K, size=n)
        rand_ari.append(adjusted_rand_score(labels_gt, rlab))
        for m in curves:
            curves[m]["ari"].append(res[m]["ari"])
            curves[m]["nmi"].append(res[m]["nmi"])
            curves[m]["sil"].append(res[m]["sil"])
            rows.append([K, m, f"{res[m]['ari']:.4f}",
                         f"{res[m]['nmi']:.4f}", f"{res[m]['sil']:.4f}"])
        if K == K_SHOW:
            store_labels = dict(gt=labels_gt, **labels_metric)
        print(f"  K={K:2d} | "
              + " | ".join(f"{m}:ARI={res[m]['ari']:+.3f}" for m in curves)
              + f" | rand={rand_ari[-1]:+.3f}")

    # ---------------------------------------------------------------- 콘솔 요약
    best = max(rows, key=lambda r: float(r[2]))
    print("\n=== 그룹핑 가능성 요약 (GHM Twin, HYBRID-L1) ===")
    print(f"  최고 ARI : {best[2]} (K={best[0]}, {best[1]})")
    print(f"  무작위 ARI 평균 : {np.mean(rand_ari):+.3f}")

    # ---------------------------------------------------------------- 시각화
    fig = plt.figure(figsize=(18, 9.5))
    gs = fig.add_gridspec(2, 4, hspace=0.32, wspace=0.30)
    fig.suptitle(
        f"GHM Twin | User Grouping 검증 (n={n}, HYBRID-L1 "
        f"α={info['alpha']:.2f}, ρ={info['rho_hybrid']:.3f})",
        fontsize=14)

    cmap = "tab10"
    mnames = ["agglo", "spectral", "kmeans_mds"]
    mcolor = {"agglo": "#2e7d32", "spectral": "#1565c0", "kmeans_mds": "#c62828"}

    # (0,0) ARI 곡선
    axA = fig.add_subplot(gs[0, 0])
    for m in mnames:
        axA.plot(K_LIST, curves[m]["ari"], "-o", ms=4, color=mcolor[m], label=m)
    axA.plot(K_LIST, rand_ari, "--", color="gray", label="random")
    axA.axhline(0, color="k", lw=0.5)
    axA.set_title("ARI vs K (정답=물리좌표 군집)", fontsize=10)
    axA.set_xlabel("K (군집 수)", fontsize=9)
    axA.set_ylabel("Adjusted Rand Index", fontsize=9)
    axA.grid(alpha=0.3); axA.legend(fontsize=8, loc="best")

    # (0,1) NMI 곡선
    axN = fig.add_subplot(gs[0, 1])
    for m in mnames:
        axN.plot(K_LIST, curves[m]["nmi"], "-o", ms=4, color=mcolor[m], label=m)
    axN.set_title("NMI vs K", fontsize=10)
    axN.set_xlabel("K (군집 수)", fontsize=9)
    axN.set_ylabel("Normalized Mutual Info", fontsize=9)
    axN.grid(alpha=0.3); axN.legend(fontsize=8, loc="best")

    # (0,2) Silhouette 곡선
    axS = fig.add_subplot(gs[0, 2])
    for m in mnames:
        axS.plot(K_LIST, curves[m]["sil"], "-o", ms=4, color=mcolor[m], label=m)
    axS.axhline(0, color="k", lw=0.5)
    axS.set_title("Silhouette vs K (메트릭 공간)", fontsize=10)
    axS.set_xlabel("K (군집 수)", fontsize=9)
    axS.set_ylabel("Silhouette score", fontsize=9)
    axS.grid(alpha=0.3); axS.legend(fontsize=8, loc="best")

    # (0,3) 메트릭 MDS 임베딩 (정답 거리 색) — 군집 구조 육안 확인
    axM = fig.add_subplot(gs[0, 3])
    scM = axM.scatter(mds_xy[:, 0], mds_xy[:, 1], c=rad0, cmap="viridis", s=10)
    axM.set_title("메트릭 MDS 임베딩 (색=TX0 거리)", fontsize=10)
    axM.set_xlabel("MDS-1", fontsize=9); axM.set_ylabel("MDS-2", fontsize=9)
    axM.tick_params(labelsize=7)
    plt.colorbar(scM, ax=axM, fraction=0.046, pad=0.04, label="dist [m]")

    # (1,0) 정답(물리) 군집
    axG = fig.add_subplot(gs[1, 0])
    axG.scatter(pos[:, 0], pos[:, 1], c=store_labels["gt"], cmap=cmap, s=12)
    axG.set_title(f"정답 그룹 (물리좌표 KMeans, K={K_SHOW})", fontsize=10)
    axG.set_xlabel("x [m]", fontsize=9); axG.set_ylabel("y [m]", fontsize=9)
    axG.set_aspect("equal", "datalim"); axG.tick_params(labelsize=7)

    # (1,1~3) 메트릭 군집을 '물리좌표' 위에 색칠 → 공간 정합도 육안 확인
    titles = {"agglo": "Agglomerative", "spectral": "Spectral",
              "kmeans_mds": "KMeans(MDS)"}
    for j, m in enumerate(mnames):
        ax = fig.add_subplot(gs[1, 1 + j])
        ax.scatter(pos[:, 0], pos[:, 1], c=store_labels[m], cmap=cmap, s=12)
        # 해당 K 의 ARI 표기
        ari_here = curves[m]["ari"][K_LIST.index(K_SHOW)]
        ax.set_title(f"메트릭 군집: {titles[m]}\n(물리좌표, K={K_SHOW}, "
                     f"ARI={ari_here:.3f})", fontsize=10)
        ax.set_xlabel("x [m]", fontsize=9); ax.set_ylabel("y [m]", fontsize=9)
        ax.set_aspect("equal", "datalim"); ax.tick_params(labelsize=7)

    fpng = os.path.join(OUTDIR, f"P2C_UserGrouping_{STAMP}.png")
    fig.savefig(fpng, dpi=130, bbox_inches="tight")
    print(f"[save] {fpng}")

    # ---------------------------------------------------------------- CSV
    fcsv = os.path.join(OUTDIR, f"P2C_UserGrouping_{STAMP}.csv")
    with open(fcsv, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["K", "method", "ARI", "NMI", "silhouette"])
        wr.writerows(rows)
        wr.writerow([])
        wr.writerow(["random_ARI_mean", f"{np.mean(rand_ari):.4f}", "", "", ""])
        wr.writerow(["hybrid_alpha", f"{info['alpha']:.2f}",
                     "rho_hybrid", f"{info['rho_hybrid']:.4f}", ""])
    print(f"[save] {fcsv}")


if __name__ == "__main__":
    main()
