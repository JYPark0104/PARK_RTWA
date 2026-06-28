# -*- coding: utf-8 -*-
"""
P2B_GHM_Metric_Test_2606v1.py
=============================
신규 RT 데이터( "260604 GHM_test" / channel_data_260531_GHM_Twin_v0_1.npz )를
기반으로, 기존 P2A 하이브리드 메트릭 실험보다 '더 나은 scatter(거리-메트릭 선형
상관) 형태'를 얻을 수 있는지 검증한다.

[ 데이터 특성 (탐색으로 확인) ]
  - RX 5071개, TX 3개(다중 기지국), rx_positions(5071,2)/tx_positions(3,3) 포함.
  - per-(TX,RX): tau[ns], power[dB(m)], aoa[deg, -180~180]. 경로 수 가변(최대 1070).
  - R_TX / R_RX 는 (1,1) = SISO -> 공간 공분산(BW)은 불가. 공간 정보는 AoA(APS)로.
  - power 는 dB 스케일 -> 반드시 선형( 10**(p/10) ) 변환 후 히스토그램 누적.

[ 비교 메트릭 ]
  (T) PDP-Wasserstein  : 지연 PDP(32 bin) 의 1D Wasserstein(CDF-L1).  -> 거리(range)
  (A) AoA-Wasserstein  : 방위각 APS(48 bin) 의 1D Wasserstein(CDF-L1). -> 방위(bearing)
  (H) 하이브리드(단일BS): alpha*AoA + (1-alpha)*PDP, alpha 자동탐색.
  (M) 다중 BS 결합     : 3개 TX 각각의 AoA-Wasserstein 을 가중합 -> 다변측위 효과.

[ 결론(요약) ]
  단일 BS 기준 AoA-Wasserstein ~0.47 (기존 데이터 AoD-W ~0.43 대비 소폭 개선),
  3개 BS 결합 시 ~0.55 로 향상. 한계 요인은 다중경로 환경에서 'range(거리)' 관측성이
  약하다는 점(PDP/RSRP 의 거리 상관 0.12~0.17). 방위(AoA)가 지배적 신호.

----------------------------------------------------------------------
실행 환경
  - Python   : 3.10.12
  - 실행 서버 : dclcom61 (deepgadget)
  - 라이브러리: numpy / scipy / scikit-learn(TSNE) / matplotlib
  - 연산규모  : N(샘플) 기본 900, 쌍방향 N*(N-1)/2 (CPU numpy 벡터화)
----------------------------------------------------------------------
"""

import os
import datetime

import numpy as np
from scipy.stats import spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# ---------------------------------------------------------------- 설정
NPZ = "/home/dclcom61/twin_minji/260512 RT Result Data/channel_data_260531_GHM_Twin_v0_1.npz"
OUTDIR = "/home/dclcom61/twin_minji/260604 metric 도출/P2B_GHM_Metric_Results"
N_TX = 3
N_SAMPLE = 900          # 거리 층화 샘플 수
MIN_PATHS = 10          # 유효 경로 최소 개수
N_TAU = 32              # PDP bin
N_AOA = 48              # APS bin
PRIMARY_TX = 0          # 단일 BS 대표 TX
SEED = 0

os.makedirs(OUTDIR, exist_ok=True)
STAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def lin(p_db):
    """dB(m) -> 선형 전력."""
    return 10.0 ** (np.asarray(p_db, dtype=np.float64) / 10.0)


def load():
    d = np.load(NPZ, allow_pickle=True)
    rxp = np.asarray(d["rx_positions"], dtype=np.float64)   # (N,2)
    txp = np.asarray(d["tx_positions"], dtype=np.float64)   # (3,3)
    N = rxp.shape[0]
    npaths = np.array([
        [len(d[f"tau_tx{T}_rx{i}"]) for T in range(N_TX)] for i in range(N)
    ])  # (N,3)
    return d, rxp, txp, N, npaths


def stratified_sample(rxp, txp, npaths, n_want):
    """대표 TX 로부터의 거리로 10 구간 층화 샘플. 3개 TX 모두 유효경로 보유 조건."""
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
    sub = np.concatenate(picks)
    return np.sort(sub)


def build_features(d, sub, txp):
    """샘플별 TX 3개 APS / 3개 RSRP(다변측위용 (n,3)), 대표 TX PDP·mean-delay 생성."""
    n = len(sub)
    aoa_edges = np.linspace(-180, 180, N_AOA + 1)

    # 대표 TX 의 tau 스케일 기준
    allt = np.concatenate([np.asarray(d[f"tau_tx{PRIMARY_TX}_rx{i}"], float) for i in sub])
    tmax = float(np.quantile(allt, 0.999))
    tau_edges = np.linspace(0, tmax, N_TAU + 1)

    APS = np.zeros((N_TX, n, N_AOA))
    PDP = np.zeros((n, N_TAU))
    rsrp_multi = np.zeros((n, N_TX))      # (n,3) : 기지국별 RSRP 벡터 (다변측위)
    mdelay = np.zeros(n)

    for k, i in enumerate(sub):
        for T in range(N_TX):
            p = lin(d[f"power_tx{T}_rx{i}"])
            a = np.asarray(d[f"aoa_tx{T}_rx{i}"], float)
            if p.size:
                ba = np.clip(np.digitize(a, aoa_edges) - 1, 0, N_AOA - 1)
                np.add.at(APS[T, k], ba, p)
            # 기지국 T 에서 본 RSRP (경로 없으면 노이즈 바닥)
            rsrp_multi[k, T] = 10 * np.log10(p.sum() + 1e-30)
        # 대표 TX 의 PDP / mean-delay
        t = np.asarray(d[f"tau_tx{PRIMARY_TX}_rx{i}"], float)
        p0 = lin(d[f"power_tx{PRIMARY_TX}_rx{i}"])
        if p0.size:
            bt = np.clip(np.digitize(np.clip(t, 0, tmax * (1 - 1e-9)), tau_edges) - 1, 0, N_TAU - 1)
            np.add.at(PDP[k], bt, p0)
            mdelay[k] = (t * p0).sum() / (p0.sum() + 1e-30)

    APS /= APS.sum(-1, keepdims=True) + 1e-30
    PDP /= PDP.sum(-1, keepdims=True) + 1e-30
    return APS, PDP, rsrp_multi, mdelay


def wass1d(D):
    """행 정규분포(히스토그램) 들의 1D Wasserstein = CDF L1. 반환 (n,n)."""
    c = np.cumsum(D, axis=-1)
    return np.abs(c[:, None, :] - c[None, :, :]).sum(-1)


def mm(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-12)


def main():
    print("[load] GHM Twin npz ...")
    d, rxp, txp, N, npaths = load()
    print(f"  RX={N}, TX={N_TX}, tx_positions=\n{txp}")

    sub = stratified_sample(rxp, txp, npaths, N_SAMPLE)
    n = len(sub)
    pos = rxp[sub]
    rad0 = np.sqrt(((pos - txp[PRIMARY_TX, :2]) ** 2).sum(1))
    print(f"[sample] 층화샘플 n={n}, dist[{rad0.min():.1f},{rad0.max():.1f}]m")

    APS, PDP, rsrp_multi, mdelay = build_features(d, sub, txp)

    iu = np.triu_indices(n, 1)
    pf = np.sqrt(((pos[:, None] - pos[None]) ** 2).sum(-1))[iu]

    # ---- 참고용 단일 BS 메트릭 ----
    Dp = mm(wass1d(PDP)[iu])
    Daos = [mm(wass1d(APS[T])[iu]) for T in range(N_TX)]
    Da = Daos[PRIMARY_TX]
    rho_pdp = spearmanr(Dp, pf).correlation
    rho_aoa = spearmanr(Da, pf).correlation

    # ============================================================
    # 다변측위(Multi-BS) 메트릭 : 직교 위상정보를 L2(유클리디안)로 결합
    # ============================================================
    # (1) 다중 BS AoA 결합 — 3개 기지국 AoA-Wass 거리의 L2 norm
    Dm_aoa_L2 = mm(np.sqrt(Daos[0] ** 2 + Daos[1] ** 2 + Daos[2] ** 2))

    # (2) 다중 BS RSRP 결합 — (n,3) RSRP 벡터 간 쌍방향 유클리디안 거리
    D_rsrp_multi = np.sqrt(((rsrp_multi[:, None, :] - rsrp_multi[None, :, :]) ** 2).sum(-1))[iu]
    D_rsrp_multi = mm(D_rsrp_multi)

    rho_aoa_L2 = spearmanr(Dm_aoa_L2, pf).correlation
    rho_rsrp_L2 = spearmanr(D_rsrp_multi, pf).correlation

    # (3) 궁극의 하이브리드 — AoA(L2) + RSRP(L2) alpha 선형결합 자동탐색
    best_h = (-9.0, 0.0)
    for a in np.linspace(0, 1, 101):
        cand = a * Dm_aoa_L2 + (1 - a) * D_rsrp_multi
        r = spearmanr(cand, pf).correlation
        if r > best_h[0]:
            best_h = (r, a)
    Dm_hybrid = mm(best_h[1] * Dm_aoa_L2 + (1 - best_h[1]) * D_rsrp_multi)

    print("\n=== Spearman rho vs 물리거리 (GHM Twin, n=%d) ===" % n)
    print("  --- 참고(단일 BS) ---")
    print(f"  PDP-Wass            : {rho_pdp:+.3f}")
    print(f"  AoA-Wass (TX{PRIMARY_TX})       : {rho_aoa:+.3f}")
    print("  --- 다변측위(Multi-BS, L2 결합) ---")
    print(f"  1) Multi-BS AoA (L2)   : {rho_aoa_L2:+.3f}")
    print(f"  2) Multi-BS RSRP (L2)  : {rho_rsrp_L2:+.3f}")
    print(f"  3) Multi-BS HYBRID     : {best_h[0]:+.3f}  (alpha_AoA={best_h[1]:.2f}, "
          f"beta_RSRP={1-best_h[1]:.2f})")

    # ---------------------------------------------------------------- 시각화
    def scat(ax, x, y, title):
        ax.scatter(x, y, s=4, alpha=0.08, c="#1f4e79", edgecolors="none")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Physical distance diff")
        ax.set_ylabel("Metric distance")

    def tsne_map(D):
        Dn = D.copy()
        full = np.zeros((n, n)); full[iu] = Dn; full = full + full.T
        return TSNE(n_components=2, metric="precomputed", init="random",
                    perplexity=30, random_state=SEED).fit_transform(full)

    fig, ax = plt.subplots(2, 4, figsize=(20, 9.5))
    fig.suptitle(
        f"GHM Twin Multi-BS (L2) | n={n}  |  AoA-L2 rho={rho_aoa_L2:.3f}, "
        f"RSRP-L2 rho={rho_rsrp_L2:.3f}, HYBRID rho={best_h[0]:.3f} "
        f"(aAoA={best_h[1]:.2f})",
        fontsize=13)

    # Row1: 상관 scatter (다변측위 L2 메트릭 3종)
    scat(ax[0, 0], pf, pf, "Ground Truth (phys vs phys)")
    scat(ax[0, 1], pf, Dm_aoa_L2, f"Multi-BS AoA (L2)  rho={rho_aoa_L2:.3f}")
    scat(ax[0, 2], pf, D_rsrp_multi, f"Multi-BS RSRP (L2)  rho={rho_rsrp_L2:.3f}")
    scat(ax[0, 3], pf, Dm_hybrid, f"Multi-BS HYBRID  rho={best_h[0]:.3f}")

    # Row2: t-SNE 지형 복원 (색 = 대표 TX 로부터 거리)
    print("[viz] t-SNE 3종 계산 ...")
    maps = {
        "Ground Truth (x,y)": pos - pos.mean(0),
        "Multi-BS AoA (L2) t-SNE": tsne_map(Dm_aoa_L2),
        "Multi-BS RSRP (L2) t-SNE": tsne_map(D_rsrp_multi),
        "Multi-BS HYBRID t-SNE": tsne_map(Dm_hybrid),
    }
    for j, (ttl, XY) in enumerate(maps.items()):
        sc = ax[1, j].scatter(XY[:, 0], XY[:, 1], c=rad0, cmap="viridis", s=10)
        ax[1, j].set_title(ttl, fontsize=10)
        ax[1, j].set_aspect("equal", "datalim")
        plt.colorbar(sc, ax=ax[1, j], fraction=0.046, pad=0.04, label="dist [m]")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fpng = os.path.join(OUTDIR, f"P2B_GHM_2x4_{STAMP}.png")
    fig.savefig(fpng, dpi=130)
    print(f"[save] {fpng}")

    # CSV 요약
    import csv
    fcsv = os.path.join(OUTDIR, f"P2B_GHM_corr_{STAMP}.csv")
    with open(fcsv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "spearman_rho", "param"])
        w.writerow(["PDP-Wass_(single)", f"{rho_pdp:.4f}", ""])
        w.writerow([f"AoA-Wass_TX{PRIMARY_TX}_(single)", f"{rho_aoa:.4f}", ""])
        w.writerow(["MultiBS_AoA_L2", f"{rho_aoa_L2:.4f}", ""])
        w.writerow(["MultiBS_RSRP_L2", f"{rho_rsrp_L2:.4f}", ""])
        w.writerow(["MultiBS_HYBRID", f"{best_h[0]:.4f}",
                    f"alpha_AoA={best_h[1]:.2f}"])
    print(f"[save] {fcsv}")


if __name__ == "__main__":
    main()
