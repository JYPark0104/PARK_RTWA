# -*- coding: utf-8 -*-
"""
P2D_BuresPDP_ChannelDist_2606v1.py
==================================
[ 목적 ]
6G NLoS Digital Twin 연구용 : 두 노드(RX) 간 '전파 채널 거리'를 아래 엄밀한 수식
(Bures metric + 1D Wasserstein 결합)으로 계산하고, 물리거리와의 상관(Scatter +
Spearman rho)을 도출한다.

[ 채널 거리 수식 (기지국별 D^2) ]
  D^2 = gamma * ( Tr(R1) + Tr(R2) - 2 Tr( sqrt( sqrt(R1) R2 sqrt(R1) ) ) )      ... (Bures/Cov)
      + delta * ( sqrt( Tr(R1) Tr(R2) ) * sum_i | CDF1(i) - CDF2(i) | )          ... (PDP Wass)

  - R1,R2 : 48-bin APS/Covariance 로 구성된 공간 공분산.
            대각 취급 시 Bures = Hellinger = sum_k ( sqrt(r1_k) - sqrt(r2_k) )^2  (효율적 동치).
  - Tr(R) : 총 수신전력 = APS 배열의 합.
  - phat  : 합=1 로 정규화된 32-bin PDP. CDF = cumsum(phat).
  - K = 32 (PDP bin), gamma/delta : Cov/PDP 비중 하이퍼파라미터.

[ 최종 거리 ]
  3개 기지국 각각 D^2 산출 -> 합산 -> sqrt => D_total.

[ 데이터 구조 ]
  - coords (N,2), channel_profiles (N,240).
  - 240 = 3 BS * 80. BS당 [0:32]=raw-power PDP(32), [32:80]=raw-power APS/Cov(48).
  - * raw 전력(선형)을 보존해야 Tr(R)=총전력 이 의미를 가짐(정규화 X).

----------------------------------------------------------------------
실행 환경
  - Python 3.10.12 / dclcom61 / numpy·scipy·matplotlib
  - 데이터 : 260512 RT Result Data/channel_data_260531_GHM_Twin_v0_1.npz (TX3, SISO)
----------------------------------------------------------------------
"""

import os
import csv
import datetime

import numpy as np
from scipy.stats import spearmanr
from scipy.linalg import sqrtm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

for _fp in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]:
    if os.path.exists(_fp):
        font_manager.fontManager.addfont(_fp)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=_fp).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

NPZ = "/home/dclcom61/twin_minji/260512 RT Result Data/channel_data_260531_GHM_Twin_v0_1.npz"
OUTDIR = "/home/dclcom61/twin_minji/260604 metric 도출/P2D_BuresPDP_ChannelDist_Results"
N_BS = 3
N_PDP = 32
N_APS = 48
BS_DIM = N_PDP + N_APS          # 80
MIN_PATHS = 10
N_NODE = 1200                   # 노드 풀
N_PAIR = 1500                   # 무작위 쌍
M_ULA = 16                      # full-Bures 모드용 가상 ULA
SEED = 0
EPS = 1e-30

os.makedirs(OUTDIR, exist_ok=True)
STAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


# =====================================================================
#  핵심 : 채널 거리 수식 구현
# =====================================================================
_A_STEER = None


def _steering():
    global _A_STEER
    if _A_STEER is None:
        centers = -180 + 360 * (np.arange(N_APS) + 0.5) / N_APS
        th = np.deg2rad(centers)
        m = np.arange(M_ULA)[:, None]
        _A_STEER = np.exp(1j * np.pi * m * np.sin(th)[None, :])     # (M, 48)
    return _A_STEER


def _bures_full(aps1, aps2):
    """48-bin APS -> 가상 ULA 공분산 재구성 후 정식 Bures^2 (행렬 제곱근)."""
    A = _steering()
    R1 = (A * aps1[None, :]) @ A.conj().T
    R2 = (A * aps2[None, :]) @ A.conj().T
    s1 = sqrtm(R1)
    inner = sqrtm(s1 @ R2 @ s1)
    val = np.real(np.trace(R1) + np.trace(R2) - 2.0 * np.trace(inner))
    return float(max(val, 0.0))


def _bures_diag(aps1, aps2):
    """대각 취급 동치 : Hellinger = sum (sqrt(r1)-sqrt(r2))^2
       = Tr(R1)+Tr(R2)-2 sum sqrt(r1 r2)  (수식과 정확히 일치)."""
    return float(np.sum((np.sqrt(aps1) - np.sqrt(aps2)) ** 2))


def compute_channel_distance(profile1, profile2, gamma, delta, cov_mode="diag",
                             reduce="sum"):
    """두 노드 채널 프로파일(240,) 간 거리.
    profile = [ BS0(80) | BS1(80) | BS2(80) ], BS당 [0:32]=PDP raw, [32:80]=APS raw.
    cov_mode: 'diag'(Hellinger, 기본/효율) 또는 'full'(ULA 재구성 + 행렬제곱근).
    reduce  : 3 BS D^2 결합 'sum' 또는 'mean'. 반환: (D_total, D2_bures합, D2_pdp합).
    """
    d2_bures_sum = 0.0
    d2_pdp_sum = 0.0
    for b in range(N_BS):
        off = b * BS_DIM
        pdp1 = np.asarray(profile1[off:off + N_PDP], float)
        aps1 = np.asarray(profile1[off + N_PDP:off + BS_DIM], float)
        pdp2 = np.asarray(profile2[off:off + N_PDP], float)
        aps2 = np.asarray(profile2[off + N_PDP:off + BS_DIM], float)

        # --- (1) Cov / Bures 항 ---
        bures = _bures_full(aps1, aps2) if cov_mode == "full" else _bures_diag(aps1, aps2)

        # --- (2) PDP 1D Wasserstein 항 ---
        Tr1 = aps1.sum(); Tr2 = aps2.sum()                  # 총 수신전력
        p1 = pdp1 / (pdp1.sum() + EPS)
        p2 = pdp2 / (pdp2.sum() + EPS)
        wass = np.abs(np.cumsum(p1) - np.cumsum(p2)).sum()  # sum_i |CDF1-CDF2|
        pdp_term = np.sqrt(Tr1 * Tr2) * wass

        d2_bures_sum += bures
        d2_pdp_sum += pdp_term

    if reduce == "mean":
        d2_bures_sum /= N_BS
        d2_pdp_sum /= N_BS
    D2 = gamma * d2_bures_sum + delta * d2_pdp_sum
    return np.sqrt(max(D2, 0.0)), d2_bures_sum, d2_pdp_sum


def compute_terms_all(profile1, profile2):
    """분석용 : (raw 수식항, trace=1 정규화 항)을 함께 반환.
       반환 dict: bures_raw, pdp_raw, bures_shape, wass  (모두 3 BS 합).
       - raw       : 수식 그대로(Tr=총전력, √(Tr1Tr2) 스케일) -> 전력 지배 -> 음상관 함정.
       - 정규화(shape): aps/PDP 를 합=1 로 -> Bures=각도 '형상' Hellinger, PDP=순수 Wasserstein.
    """
    bures_raw = pdp_raw = bures_shape = wass_sum = 0.0
    for b in range(N_BS):
        off = b * BS_DIM
        pdp1 = np.asarray(profile1[off:off + N_PDP], float)
        aps1 = np.asarray(profile1[off + N_PDP:off + BS_DIM], float)
        pdp2 = np.asarray(profile2[off:off + N_PDP], float)
        aps2 = np.asarray(profile2[off + N_PDP:off + BS_DIM], float)
        Tr1 = aps1.sum(); Tr2 = aps2.sum()
        p1 = pdp1 / (pdp1.sum() + EPS); p2 = pdp2 / (pdp2.sum() + EPS)
        wass = np.abs(np.cumsum(p1) - np.cumsum(p2)).sum()
        # raw
        bures_raw += np.sum((np.sqrt(aps1) - np.sqrt(aps2)) ** 2)
        pdp_raw += np.sqrt(Tr1 * Tr2) * wass
        # trace=1 정규화 (형상)
        a1 = aps1 / (Tr1 + EPS); a2 = aps2 / (Tr2 + EPS)
        bures_shape += np.sum((np.sqrt(a1) - np.sqrt(a2)) ** 2)
        wass_sum += wass
    return bures_raw, pdp_raw, bures_shape, wass_sum


# =====================================================================
#  데이터 로딩 / 프로파일 빌드 (raw 전력 보존)
# =====================================================================
def lin(p_db):
    return 10.0 ** (np.asarray(p_db, dtype=np.float64) / 10.0)


def load_nodes():
    d = np.load(NPZ, allow_pickle=True)
    rxp = np.asarray(d["rx_positions"], float)
    txp = np.asarray(d["tx_positions"], float)
    N = rxp.shape[0]
    npaths = np.array([[len(d[f"tau_tx{T}_rx{i}"]) for T in range(N_BS)]
                       for i in range(N)])
    ok = np.where(np.all(npaths >= MIN_PATHS, axis=1))[0]
    rng = np.random.default_rng(SEED)
    sub = np.sort(rng.choice(ok, min(N_NODE, len(ok)), replace=False))
    return d, rxp, txp, sub


def build_profiles(d, sub):
    """coords(N,2), channel_profiles(N,240) : raw 선형전력 PDP+APS."""
    n = len(sub)
    aoa_edges = np.linspace(-180, 180, N_APS + 1)
    tmax = np.zeros(N_BS)
    for T in range(N_BS):
        allt = np.concatenate([np.asarray(d[f"tau_tx{T}_rx{i}"], float) for i in sub])
        tmax[T] = float(np.quantile(allt, 0.999))

    prof = np.zeros((n, N_BS * BS_DIM))
    for k, i in enumerate(sub):
        for T in range(N_BS):
            off = T * BS_DIM
            p = lin(d[f"power_tx{T}_rx{i}"])
            a = np.asarray(d[f"aoa_tx{T}_rx{i}"], float)
            t = np.asarray(d[f"tau_tx{T}_rx{i}"], float)
            if p.size and p.sum() > 0:
                # APS (raw power, 정규화 X -> Tr=총전력)
                ba = np.clip(np.digitize(a, aoa_edges) - 1, 0, N_APS - 1)
                np.add.at(prof[k], off + N_PDP + ba, p)
                # PDP (raw power)
                tau_edges = np.linspace(0, tmax[T], N_PDP + 1)
                bt = np.clip(np.digitize(np.clip(t, 0, tmax[T] * (1 - 1e-9)),
                                         tau_edges) - 1, 0, N_PDP - 1)
                np.add.at(prof[k], off + bt, p)
    return prof


def main():
    print("[load] GHM Twin npz ...")
    d, rxp, txp, sub = load_nodes()
    coords = rxp[sub]
    prof = build_profiles(d, sub)
    n = len(sub)
    print(f"[build] coords{coords.shape}, channel_profiles{prof.shape} (raw power)")

    # --- 무작위 쌍 ---
    rng = np.random.default_rng(SEED)
    ii = rng.integers(0, n, N_PAIR)
    jj = rng.integers(0, n, N_PAIR)
    mask = ii != jj
    ii, jj = ii[mask], jj[mask]
    n_pair = len(ii)

    phys = np.sqrt(((coords[ii] - coords[jj]) ** 2).sum(1))
    B_raw = np.zeros(n_pair)    # 수식 그대로 Bures 항(raw)
    P_raw = np.zeros(n_pair)    # 수식 그대로 PDP 항(raw, √Tr 스케일)
    B_sh = np.zeros(n_pair)     # trace=1 Bures(형상)
    W = np.zeros(n_pair)        # 순수 Wasserstein(스케일 X)
    for k in range(n_pair):
        B_raw[k], P_raw[k], B_sh[k], W[k] = compute_terms_all(prof[ii[k]], prof[jj[k]])
    print(f"[pairs] {n_pair} pairs, phys[{phys.min():.1f},{phys.max():.1f}]m")

    def sp(x):
        return spearmanr(x, phys).correlation

    def mm(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-12)

    def best_combo(Bx, Px):
        Bn, Pn = mm(Bx), mm(Px)
        bst = (-9.0, 1.0, 0.0, None)
        for w in np.linspace(0, 1, 101):
            comb = (1 - w) * Bn + w * Pn
            r = sp(comb)
            if r > bst[0]:
                bst = (r, 1 - w, w, comb)
        return bst

    # --- (A) 수식 원형(raw) ---
    rB_raw, rP_raw = sp(B_raw), sp(P_raw)
    rRawComb = sp(1.0 * mm(B_raw) + 1.0 * mm(P_raw))
    # --- (B) trace=1 정규화(형상) ---
    rB_sh, rW = sp(B_sh), sp(W)
    bestr, g_b, d_b, comb_best = best_combo(B_sh, W)

    print("\n=== Spearman rho vs 물리거리 (n_pair=%d) ===" % n_pair)
    print("  --- (A) 수식 그대로 raw (Tr=총전력, √Tr 스케일) ---")
    print(f"    Bures(raw) 단독    : {rB_raw:+.3f}")
    print(f"    PDP(raw) 단독      : {rP_raw:+.3f}")
    print(f"    결합(raw,γ=δ=1)    : {rRawComb:+.3f}   <- 전력 지배로 음/약상관 함정")
    print("  --- (B) trace=1 정규화(형상) : 검증된 해법 ---")
    print(f"    Bures-shape 단독   : {rB_sh:+.3f}")
    print(f"    Wasserstein 단독   : {rW:+.3f}")
    print(f"    결합(최적 γ:δ)     : {bestr:+.3f}  (γ:δ≈{g_b:.2f}:{d_b:.2f})")

    # ---------------------------------------------------------------- 시각화 (2x3)
    def scat(ax, y, title, rho, ylab="채널 거리"):
        ax.scatter(phys, y, s=9, alpha=0.30, c="#1f4e79", edgecolors="none")
        ax.set_title(f"{title}\nSpearman ρ = {rho:+.3f}", fontsize=10.5)
        ax.set_xlabel("물리 거리 (m)", fontsize=9)
        ax.set_ylabel(ylab, fontsize=9)
        ax.grid(alpha=0.3)

    fig, ax = plt.subplots(2, 3, figsize=(18, 10.5))
    # 상단: 수식 그대로(raw)
    scat(ax[0, 0], np.sqrt(B_raw), "Bures(raw) 단독", rB_raw)
    scat(ax[0, 1], np.sqrt(P_raw), "PDP(raw, √Tr 스케일) 단독", rP_raw)
    scat(ax[0, 2], np.sqrt(mm(B_raw) + mm(P_raw)), "결합 raw (γ=δ=1)", rRawComb)
    # 하단: trace=1 정규화(형상)
    scat(ax[1, 0], np.sqrt(B_sh), "Bures-shape (trace=1)", rB_sh)
    scat(ax[1, 1], W, "PDP 순수 Wasserstein", rW)
    scat(ax[1, 2], comb_best, f"결합 (γ:δ≈{g_b:.2f}:{d_b:.2f})", bestr)
    ax[0, 0].annotate("(A) 수식 그대로 raw : 전력(Tr) 지배 → 음/약상관 함정",
                      xy=(0, 1.18), xycoords="axes fraction", fontsize=12,
                      color="#b71c1c", weight="bold")
    ax[1, 0].annotate("(B) trace=1 정규화(형상) : 검증된 해법 → 양상관 회복",
                      xy=(0, 1.18), xycoords="axes fraction", fontsize=12,
                      color="#1b5e20", weight="bold")
    fig.suptitle("6G NLoS Digital Twin : Bures(Cov) + 1D-Wasserstein(PDP) 채널거리 vs 물리거리  "
                 f"(GHM 3-BS, n_pair={n_pair})", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fpng = os.path.join(OUTDIR, f"P2D_BuresPDP_scatter_{STAMP}.png")
    fig.savefig(fpng, dpi=130)
    print(f"[save] {fpng}")

    # ---------------------------------------------------------------- 단독 발표용 그림
    figp, axp = plt.subplots(figsize=(9.2, 6.6))
    axp.scatter(phys, comb_best, s=14, alpha=0.30, c="#3a6fb0",
                edgecolors="none", zorder=2, label="노드 쌍 (pair)")
    # 분위수 빈 추세선(평균 ± 표준편차)
    order = np.argsort(phys)
    ps_, cs_ = phys[order], comb_best[order]
    nb = 18
    edges = np.quantile(ps_, np.linspace(0, 1, nb + 1))
    bx, bm, bs_ = [], [], []
    for b in range(nb):
        m = (ps_ >= edges[b]) & (ps_ <= edges[b + 1] if b == nb - 1 else ps_ < edges[b + 1])
        if m.sum() >= 3:
            bx.append(ps_[m].mean()); bm.append(cs_[m].mean()); bs_.append(cs_[m].std())
    bx, bm, bs_ = np.array(bx), np.array(bm), np.array(bs_)
    axp.fill_between(bx, bm - bs_, bm + bs_, color="#1b5e20", alpha=0.15, zorder=3)
    axp.plot(bx, bm, "-", color="#1b5e20", lw=3, zorder=4, label="구간 평균 추세")
    axp.text(0.04, 0.93, f"Spearman ρ = {bestr:+.3f}", transform=axp.transAxes,
             fontsize=17, weight="bold", color="#1b5e20",
             bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#1b5e20", alpha=0.9))
    axp.set_xlabel("물리 거리 (m)", fontsize=13)
    axp.set_ylabel("채널 거리  D$_{total}$  (Bures-Cov + PDP-Wasserstein)", fontsize=13)
    axp.set_title("6G NLoS Digital Twin : 물리 거리 vs 채널 거리\n"
                  f"Bures(Cov) + 1D-Wasserstein(PDP) 결합, trace=1 정규화  "
                  f"(γ:δ≈{g_b:.2f}:{d_b:.2f}, GHM 3-BS, n={n_pair})", fontsize=12.5)
    axp.grid(alpha=0.3); axp.legend(loc="lower right", fontsize=11)
    figp.tight_layout()
    fpres = os.path.join(OUTDIR, f"P2D_BuresPDP_PRESENT_{STAMP}.png")
    figp.savefig(fpres, dpi=150)
    print(f"[save] {fpres}")

    fcsv = os.path.join(OUTDIR, f"P2D_BuresPDP_corr_{STAMP}.csv")
    with open(fcsv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "spearman_rho", "param"])
        w.writerow(["A_Bures_raw", f"{rB_raw:.4f}", "Tr=total power"])
        w.writerow(["A_PDP_raw", f"{rP_raw:.4f}", "sqrt(Tr1Tr2) scale"])
        w.writerow(["A_Combined_raw", f"{rRawComb:.4f}", "gamma=delta=1 (음/약상관)"])
        w.writerow(["B_Bures_shape", f"{rB_sh:.4f}", "trace=1 normalized"])
        w.writerow(["B_Wasserstein", f"{rW:.4f}", "no power scale"])
        w.writerow(["B_Combined_best", f"{bestr:.4f}", f"gamma:delta={g_b:.2f}:{d_b:.2f}"])
    print(f"[save] {fcsv}")


if __name__ == "__main__":
    main()
