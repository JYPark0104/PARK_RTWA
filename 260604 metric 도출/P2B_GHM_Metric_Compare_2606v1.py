# -*- coding: utf-8 -*-
"""
P2B_GHM_Metric_Compare_2606v1.py
================================
신규 RT 데이터( "260604 GHM_test" / channel_data_260531_GHM_Twin_v0_1.npz )에 대해
'다중 기지국(Multi-BS) AoA 결합'을 두 가지 방식으로 동일 샘플·동일 전처리에서
공정 비교한다.

  (L1) 스칼라 가중합  : w0*D0 + w1*D1 + w2*D2  (단체(simplex) 가중치 자동탐색)
  (L2) 벡터 유클리드  : sqrt(D0^2 + D1^2 + D2^2)  (3 TX 동일가중)

배경 : 이전 실행본(114308, 가중합)과 최신본(115151, L2)이 서로 다른 코드에서
나왔고, 문서 결론과 달리 가중합(rho=0.539)이 L2(rho=0.505)보다 높았다.
원인을 한 그림에서 검증하기 위해 두 방식을 같은 파이프라인으로 재계산한다.

[ 데이터 특성 ]
  - RX 5071, TX 3(다중 기지국). per-(TX,RX): tau[ns], power[dB(m)], aoa[deg].
  - SISO(R=1) -> 공간 공분산 불가. 공간정보는 AoA(APS). power 는 dB -> 선형변환.

----------------------------------------------------------------------
실행 환경
  - Python   : 3.10.12
  - 실행 서버 : dclcom61 (deepgadget)
  - 라이브러리: numpy / scipy / scikit-learn(TSNE) / matplotlib (CPU)
  - 연산규모  : N(샘플) 900, 쌍방향 N*(N-1)/2, 가중치 단체격자 탐색
----------------------------------------------------------------------
"""

import os
import csv
import datetime

import numpy as np
from scipy.stats import spearmanr, rankdata

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.manifold import TSNE

# 한글 폰트 설정 (Noto Sans CJK KR) : 라벨/제목 글리프 깨짐(tofu) 방지
for _fp in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]:
    if os.path.exists(_fp):
        font_manager.fontManager.addfont(_fp)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=_fp).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

# ---------------------------------------------------------------- 설정
NPZ = "/home/dclcom61/twin_minji/260512 RT Result Data/channel_data_260531_GHM_Twin_v0_1.npz"
OUTDIR = "/home/dclcom61/twin_minji/260604 metric 도출/P2B_GHM_Metric_Results"
N_TX = 3
N_SAMPLE = 900
MIN_PATHS = 10
N_TAU = 32
N_AOA = 48
PRIMARY_TX = 0
W_STEP = 0.05          # 가중치 단체격자 간격
SEED = 0

os.makedirs(OUTDIR, exist_ok=True)
STAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


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
    """샘플별 TX 3개 APS / 3개 RSRP((n,3)), 대표 TX PDP 생성."""
    n = len(sub)
    aoa_edges = np.linspace(-180, 180, N_AOA + 1)
    allt = np.concatenate([np.asarray(d[f"tau_tx{PRIMARY_TX}_rx{i}"], float) for i in sub])
    tmax = float(np.quantile(allt, 0.999))
    tau_edges = np.linspace(0, tmax, N_TAU + 1)

    APS = np.zeros((N_TX, n, N_AOA))
    PDP = np.zeros((n, N_TAU))
    rsrp_multi = np.zeros((n, N_TX))

    for k, i in enumerate(sub):
        for T in range(N_TX):
            p = lin(d[f"power_tx{T}_rx{i}"])
            a = np.asarray(d[f"aoa_tx{T}_rx{i}"], float)
            if p.size:
                ba = np.clip(np.digitize(a, aoa_edges) - 1, 0, N_AOA - 1)
                np.add.at(APS[T, k], ba, p)
            rsrp_multi[k, T] = 10 * np.log10(p.sum() + 1e-30)
        t = np.asarray(d[f"tau_tx{PRIMARY_TX}_rx{i}"], float)
        p0 = lin(d[f"power_tx{PRIMARY_TX}_rx{i}"])
        if p0.size:
            bt = np.clip(np.digitize(np.clip(t, 0, tmax * (1 - 1e-9)), tau_edges) - 1,
                         0, N_TAU - 1)
            np.add.at(PDP[k], bt, p0)

    APS /= APS.sum(-1, keepdims=True) + 1e-30
    PDP /= PDP.sum(-1, keepdims=True) + 1e-30
    return APS, PDP, rsrp_multi


def wass1d(D):
    """행 히스토그램들의 1D Wasserstein = CDF L1. 반환 (n,n)."""
    c = np.cumsum(D, axis=-1)
    return np.abs(c[:, None, :] - c[None, :, :]).sum(-1)


def mm(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-12)


def fast_spearman_vs(ref_rank, x):
    """ref(고정 rank) 대비 x 의 Spearman = rank-Pearson. (반복탐색 가속)"""
    rx = rankdata(x)
    rx = rx - rx.mean()
    denom = np.sqrt((rx * rx).sum() * (ref_rank * ref_rank).sum())
    return float((rx * ref_rank).sum() / (denom + 1e-30))


def simplex_grid(step):
    """w0+w1+w2=1, wi>=0 격자."""
    g = []
    k = int(round(1.0 / step))
    for a in range(k + 1):
        for b in range(k + 1 - a):
            c = k - a - b
            g.append((a / k, b / k, c / k))
    return np.array(g)


def main():
    print("[load] GHM Twin npz ...")
    d, rxp, txp, N, npaths = load()
    print(f"  RX={N}, TX={N_TX}\n  tx_positions=\n{txp}")

    sub = stratified_sample(rxp, txp, npaths, N_SAMPLE)
    n = len(sub)
    pos = rxp[sub]
    rad0 = np.sqrt(((pos - txp[PRIMARY_TX, :2]) ** 2).sum(1))
    print(f"[sample] n={n}, dist[{rad0.min():.1f},{rad0.max():.1f}]m")

    APS, PDP, rsrp_multi = build_features(d, sub)

    iu = np.triu_indices(n, 1)
    pf = np.sqrt(((pos[:, None] - pos[None]) ** 2).sum(-1))[iu]
    # 고정 ref rank (가속용)
    pf_rank = rankdata(pf); pf_rank = pf_rank - pf_rank.mean()

    # ---- 단일 BS 참고 ----
    Dp = mm(wass1d(PDP)[iu])
    Daos = [mm(wass1d(APS[T])[iu]) for T in range(N_TX)]
    Da = Daos[PRIMARY_TX]
    rho_pdp = spearmanr(Dp, pf).correlation
    rho_aoa = spearmanr(Da, pf).correlation

    # ============================================================
    # (L1) 스칼라 가중합 — 단체격자에서 rho 최대화
    # ============================================================
    grid = simplex_grid(W_STEP)
    D0, D1, D2 = Daos
    best_w = (-9.0, None)
    for (w0, w1, w2) in grid:
        cand = w0 * D0 + w1 * D1 + w2 * D2
        r = fast_spearman_vs(pf_rank, cand)
        if r > best_w[0]:
            best_w = (r, (w0, w1, w2))
    w = best_w[1]
    Dm_aoa_L1 = mm(w[0] * D0 + w[1] * D1 + w[2] * D2)
    rho_aoa_L1 = best_w[0]

    # ============================================================
    # (L2) 벡터 유클리드 — 3 TX 동일가중
    # ============================================================
    Dm_aoa_L2 = mm(np.sqrt(D0 ** 2 + D1 ** 2 + D2 ** 2))
    rho_aoa_L2 = spearmanr(Dm_aoa_L2, pf).correlation

    # ---- 다변측위 RSRP (n,3) 벡터 유클리드 거리 ----
    D_rsrp = mm(np.sqrt(((rsrp_multi[:, None, :] - rsrp_multi[None, :, :]) ** 2).sum(-1))[iu])
    rho_rsrp = spearmanr(D_rsrp, pf).correlation

    # ---- HYBRID(L1): AoA-L1 + RSRP, alpha 탐색 ----
    bh1 = (-9.0, 0.0)
    for a in np.linspace(0, 1, 101):
        r = fast_spearman_vs(pf_rank, a * Dm_aoa_L1 + (1 - a) * D_rsrp)
        if r > bh1[0]:
            bh1 = (r, a)
    Dm_hyb_L1 = mm(bh1[1] * Dm_aoa_L1 + (1 - bh1[1]) * D_rsrp)

    # ---- HYBRID(L2): AoA-L2 + RSRP, alpha 탐색 ----
    bh2 = (-9.0, 0.0)
    for a in np.linspace(0, 1, 101):
        r = fast_spearman_vs(pf_rank, a * Dm_aoa_L2 + (1 - a) * D_rsrp)
        if r > bh2[0]:
            bh2 = (r, a)
    Dm_hyb_L2 = mm(bh2[1] * Dm_aoa_L2 + (1 - bh2[1]) * D_rsrp)

    # ---------------------------------------------------------------- 콘솔 요약
    print("\n=== Spearman rho vs 물리거리 (GHM Twin, n=%d) ===" % n)
    print("  --- 단일 BS(참고) ---")
    print(f"  PDP-Wass             : {rho_pdp:+.3f}")
    print(f"  AoA-Wass (TX{PRIMARY_TX})        : {rho_aoa:+.3f}")
    print("  --- Multi-BS AoA 결합 (공정 비교) ---")
    print(f"  (L1) 가중합          : {rho_aoa_L1:+.3f}  w=({w[0]:.2f},{w[1]:.2f},{w[2]:.2f})")
    print(f"  (L2) 벡터 유클리드   : {rho_aoa_L2:+.3f}")
    print("  --- 다변측위 RSRP / HYBRID ---")
    print(f"  RSRP (n,3) L2        : {rho_rsrp:+.3f}")
    print(f"  HYBRID(L1)+RSRP      : {bh1[0]:+.3f}  alpha_AoA={bh1[1]:.2f}")
    print(f"  HYBRID(L2)+RSRP      : {bh2[0]:+.3f}  alpha_AoA={bh2[1]:.2f}")

    # ---------------------------------------------------------------- 시각화
    def scat(ax, x, y, title):
        ax.scatter(x, y, s=4, alpha=0.08, c="#1f4e79", edgecolors="none")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Physical distance diff", fontsize=8)
        ax.set_ylabel("Metric distance", fontsize=8)
        ax.tick_params(labelsize=7)

    def tsne_map(D):
        full = np.zeros((n, n)); full[iu] = D; full = full + full.T
        return TSNE(n_components=2, metric="precomputed", init="random",
                    perplexity=30, random_state=SEED).fit_transform(full)

    fig, ax = plt.subplots(2, 4, figsize=(20, 9.8))
    fig.suptitle(
        f"GHM Twin | Multi-BS AoA 결합 공정비교 (n={n}) | "
        f"L1(가중합) rho={rho_aoa_L1:.3f}  vs  L2(유클리드) rho={rho_aoa_L2:.3f}",
        fontsize=14)

    # Row1: scatter
    scat(ax[0, 0], pf, pf, "Ground Truth (phys vs phys)")
    scat(ax[0, 1], pf, Dm_aoa_L1,
         f"Multi-BS AoA (L1 가중합)  rho={rho_aoa_L1:.3f}")
    scat(ax[0, 2], pf, Dm_aoa_L2,
         f"Multi-BS AoA (L2 유클리드)  rho={rho_aoa_L2:.3f}")

    # (0,3) : rho 막대 비교
    labels = ["PDP\n(single)", "AoA\nTX0", "AoA\nL1", "AoA\nL2",
              "RSRP\nL2", "HYB\nL1", "HYB\nL2"]
    vals = [rho_pdp, rho_aoa, rho_aoa_L1, rho_aoa_L2, rho_rsrp, bh1[0], bh2[0]]
    colors = ["#9e9e9e", "#9e9e9e", "#2e7d32", "#c62828",
              "#9e9e9e", "#1b5e20", "#b71c1c"]
    bars = ax[0, 3].bar(range(len(vals)), vals, color=colors)
    ax[0, 3].set_xticks(range(len(vals)))
    ax[0, 3].set_xticklabels(labels, fontsize=7)
    ax[0, 3].set_ylabel("Spearman rho", fontsize=8)
    ax[0, 3].set_title("메트릭별 rho 비교", fontsize=10)
    ax[0, 3].axhline(0, color="k", lw=0.6)
    ax[0, 3].grid(axis="y", alpha=0.3)
    for b, v in zip(bars, vals):
        ax[0, 3].text(b.get_x() + b.get_width() / 2, v + 0.008,
                      f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    # Row2: HYBRID scatter 2종 + t-SNE 2종(우세 방식 + GT)
    scat(ax[1, 0], pf, Dm_hyb_L1,
         f"HYBRID L1+RSRP  rho={bh1[0]:.3f} (aA={bh1[1]:.2f})")
    scat(ax[1, 1], pf, Dm_hyb_L2,
         f"HYBRID L2+RSRP  rho={bh2[0]:.3f} (aA={bh2[1]:.2f})")

    print("[viz] t-SNE 2종 계산 ...")
    XY_gt = pos - pos.mean(0)
    XY_best = tsne_map(Dm_aoa_L1 if rho_aoa_L1 >= rho_aoa_L2 else Dm_aoa_L2)
    best_name = "L1 가중합" if rho_aoa_L1 >= rho_aoa_L2 else "L2 유클리드"
    for j, (ttl, XY) in enumerate([("Ground Truth (x,y)", XY_gt),
                                   (f"우세방식 t-SNE ({best_name})", XY_best)]):
        sc = ax[1, 2 + j].scatter(XY[:, 0], XY[:, 1], c=rad0, cmap="viridis", s=10)
        ax[1, 2 + j].set_title(ttl, fontsize=10)
        ax[1, 2 + j].set_aspect("equal", "datalim")
        ax[1, 2 + j].tick_params(labelsize=7)
        plt.colorbar(sc, ax=ax[1, 2 + j], fraction=0.046, pad=0.04, label="dist [m]")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fpng = os.path.join(OUTDIR, f"P2B_GHM_compare_{STAMP}.png")
    fig.savefig(fpng, dpi=130)
    print(f"[save] {fpng}")

    # ---------------------------------------------------------------- CSV
    fcsv = os.path.join(OUTDIR, f"P2B_GHM_compare_{STAMP}.csv")
    with open(fcsv, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["metric", "spearman_rho", "param"])
        wr.writerow(["PDP-Wass_(single)", f"{rho_pdp:.4f}", ""])
        wr.writerow([f"AoA-Wass_TX{PRIMARY_TX}_(single)", f"{rho_aoa:.4f}", ""])
        wr.writerow(["MultiBS_AoA_L1_weighted", f"{rho_aoa_L1:.4f}",
                     f"w={w[0]:.2f},{w[1]:.2f},{w[2]:.2f}"])
        wr.writerow(["MultiBS_AoA_L2_euclid", f"{rho_aoa_L2:.4f}", ""])
        wr.writerow(["MultiBS_RSRP_L2", f"{rho_rsrp:.4f}", ""])
        wr.writerow(["HYBRID_L1+RSRP", f"{bh1[0]:.4f}", f"alpha_AoA={bh1[1]:.2f}"])
        wr.writerow(["HYBRID_L2+RSRP", f"{bh2[0]:.4f}", f"alpha_AoA={bh2[1]:.2f}"])
        wr.writerow(["WINNER", best_name,
                     f"L1={rho_aoa_L1:.4f} vs L2={rho_aoa_L2:.4f}"])
    print(f"[save] {fcsv}")


if __name__ == "__main__":
    main()
