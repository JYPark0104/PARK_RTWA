# -*- coding: utf-8 -*-
"""
P2G_PairTable_2606v1.py
=======================
[단계 3] 쌍 단위 테이블(X, y) 생성 + 진단 산점도/포화 분석.

흐름
  거리행렬(NxN) -> 상삼각(i<j) 추출 -> X=[d_RSRP,d_PDP,d_Cov], y=d_phys.
  M = N(N-1)/2 행. 이 표가 단계 4 least-square(NNLS)의 입력.

추가 산출 (요청)
  1) 세 항 각각 vs d_phys 산점도 3장 (+ 구간별 중앙/10-90% 밴드 오버레이)
  2) d_Cov 포화 분석:
     - d_Cov 가 어느 d_phys 까지 선형이고 어디서 √2(≈1.414) 포화되는지 (유효 거리)
     - d_Cov < 1.40 (비포화) 쌍이 전체의 몇 %, 어떤 d_phys 구간인지

물리거리
  d_phys(i,j) = ||pos_i - pos_j||_2  (UE-UE Euclidean, BS 거리 아님) -> 학습 타깃 y

----------------------------------------------------------------------
실행 환경 : Python 3.10.12 / numpy 2.2.6 / scipy 1.15.3 / matplotlib
서버 dclserver78 (이 단계는 경량 -> CPU/numpy)
입력 : P2G_DistTerms_Results/P2G_DistMat_*.npz
----------------------------------------------------------------------
"""

import os
import glob
import datetime

import numpy as np
from scipy.stats import spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

# 한글 폰트 (Noto Sans CJK KR) 등록 -> 라벨 깨짐 방지
for _fp in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",):
    if os.path.exists(_fp):
        fm.fontManager.addfont(_fp)
        plt.rcParams["font.family"] = fm.FontProperties(fname=_fp).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DM_DIR   = os.path.join(BASE_DIR, "P2G_DistTerms_Results")
OUT_DIR  = os.path.join(BASE_DIR, "P2G_PairTable_Results")

SAT_THRESH = 1.40            # d_Cov 포화 임계 (요청)
SQRT2 = float(np.sqrt(2.0))
N_BIN = 40                   # d_phys 구간 수
SEED = 42


def latest_distmat():
    fs = sorted(glob.glob(os.path.join(DM_DIR, "P2G_DistMat_*.npz")))
    if not fs:
        raise FileNotFoundError("거리행렬 npz 없음. 먼저 단계2 실행.")
    return fs[-1]


def binned_stats(x, y, edges):
    """구간별 중앙/10/90 퍼센타일 (NaN 안전)."""
    idx = np.clip(np.digitize(x, edges) - 1, 0, len(edges) - 2)
    med = np.full(len(edges) - 1, np.nan)
    p10 = np.full_like(med, np.nan)
    p90 = np.full_like(med, np.nan)
    cnt = np.zeros(len(edges) - 1, dtype=int)
    for b in range(len(edges) - 1):
        m = idx == b
        cnt[b] = m.sum()
        if m.any():
            med[b] = np.median(y[m])
            p10[b], p90[b] = np.percentile(y[m], [10, 90])
    ctr = 0.5 * (edges[:-1] + edges[1:])
    return ctr, med, p10, p90, cnt


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    rng = np.random.default_rng(SEED)
    log_f = open(os.path.join(OUT_DIR, f"P2G_pairtable_report_{stamp}.log"),
                 "w", encoding="utf-8")

    def log(*a):
        m = " ".join(str(x) for x in a); print(m); log_f.write(m + "\n"); log_f.flush()

    try:
        dmf = latest_distmat()
        d = np.load(dmf, allow_pickle=True)
        D_rsrp, D_pdp, D_cov = d["D_rsrp"], d["D_pdp"], d["D_cov"]
        pos = d["pos"]
        N = pos.shape[0]
        log(f"[load] {os.path.basename(dmf)}  N={N}, rank={int(d['rank'])}")

        # d_phys (UE-UE) NxN
        diff = pos[:, None, :] - pos[None, :, :]
        D_phys = np.sqrt((diff ** 2).sum(-1))

        # 상삼각 i<j
        iu = np.triu_indices(N, 1)
        y = D_phys[iu].astype(np.float64)                 # d_phys [m]
        x_rsrp = D_rsrp[iu].astype(np.float64)            # dB
        x_pdp = D_pdp[iu].astype(np.float64)              # ns
        x_cov = D_cov[iu].astype(np.float64)              # 무차원
        X = np.stack([x_rsrp, x_pdp, x_cov], axis=1)
        M = X.shape[0]
        log(f"[pair] M=N(N-1)/2={M} 쌍, X{X.shape}, y{y.shape}")
        log(f"[unit] y=d_phys[m], X=[d_RSRP(dB), d_PDP(ns), d_Cov(-)]")

        # 각 항 Spearman vs d_phys (단조 상관)
        for nm, xx in [("d_RSRP", x_rsrp), ("d_PDP", x_pdp), ("d_Cov", x_cov)]:
            rho = spearmanr(xx, y).statistic
            log(f"[corr] Spearman({nm}, d_phys) = {rho:.4f}")

        # ===== d_Cov 포화 분석 =====
        log("\n" + "=" * 70)
        log(f"[d_Cov 포화 분석]  √2={SQRT2:.4f}, 임계={SAT_THRESH}")
        nonsat = x_cov < SAT_THRESH
        frac = nonsat.mean() * 100
        log(f"  d_Cov < {SAT_THRESH} (비포화) 쌍: {nonsat.sum()} / {M} = {frac:.2f}%")
        if nonsat.any():
            yp = y[nonsat]
            log(f"  비포화 쌍의 d_phys[m] min/med/max = "
                f"{yp.min():.2f}/{np.median(yp):.2f}/{yp.max():.2f}")
            for thr in (10, 20, 30, 50, 100):
                in_thr = (yp <= thr).mean() * 100
                log(f"    비포화 쌍 중 d_phys<={thr}m 비율 = {in_thr:.1f}%")
        # 거리 구간별 d_Cov 중앙값으로 선형/포화 경계 추정
        edges = np.linspace(0, y.max(), N_BIN + 1)
        ctr, med_cov, p10c, p90c, cnt = binned_stats(y, x_cov, edges)
        # '유효(선형) 범위': 중앙 d_Cov 곡선이 plateau(원거리 평탄값)의 99%에
        #   처음 도달하는 거리. plateau 는 데이터에서 추정(상위 50% 거리 구간 중앙).
        #   (고정 √2 기준은 실제 plateau≈1.38~1.39 라 거의 교차하지 않아 부적합)
        far = ctr > 0.5 * y.max()
        plateau = float(np.nanmedian(med_cov[far]))
        sat_lvl = 0.99 * plateau
        cross = np.where(med_cov >= sat_lvl)[0]
        eff = edges[cross[0]] if cross.size else y.max()  # 포화 시작 거리[m]
        log(f"  거리 구간별 중앙 d_Cov: 0m={med_cov[0]:.3f}, "
            f"plateau≈{plateau:.3f}(√2={SQRT2:.3f}), "
            f"0.99·plateau={sat_lvl:.3f} 최초 도달 ~ {eff:.1f}m")
        log(f"  => 유효 선형 구간: d_phys ≲ {eff:.0f}m, 그 이상은 plateau 포화(분해능 거의 없음)")
        # 구간별 비포화 비율
        log("  d_phys 구간별 비포화(d_Cov<1.40) 비율:")
        for lo, hi in [(0, 10), (10, 20), (20, 40), (40, 80), (80, 1e9)]:
            m = (y >= lo) & (y < hi)
            if m.any():
                log(f"    [{lo:>3}-{hi if hi<1e9 else 'max':>4}m] "
                    f"비포화 {100*nonsat[m].mean():5.1f}%  (쌍수 {m.sum()})")

        # ===== 산점도 3장 + 포화 분해 =====
        sidx = rng.choice(M, size=min(120000, M), replace=False)  # 렌더용 서브샘플
        fig, ax = plt.subplots(2, 2, figsize=(15, 12))
        terms = [("d_RSRP", x_rsrp, "d_RSRP [dB]"),
                 ("d_PDP", x_pdp, "d_PDP [ns]"),
                 ("d_Cov", x_cov, "d_Cov [-]")]
        for k, (nm, xx, ylab) in enumerate(terms):
            a = ax[k // 2, k % 2]
            a.scatter(y[sidx], xx[sidx], s=2, alpha=0.04, color="navy",
                      edgecolors="none", rasterized=True)
            ctr2, med, p10, p90, _ = binned_stats(y, xx, edges)
            a.plot(ctr2, med, "-o", color="crimson", ms=3, lw=1.6,
                   label="구간 중앙값")
            a.fill_between(ctr2, p10, p90, color="orange", alpha=0.25,
                           label="10-90%")
            a.set_xlabel("d_phys [m] (UE-UE)")
            a.set_ylabel(ylab)
            a.set_title(f"{nm} vs d_phys  (Spearman={spearmanr(xx,y).statistic:.3f})")
            a.grid(True, ls=":", alpha=0.4)
            a.legend(loc="best", fontsize=9)
            if nm == "d_Cov":
                a.axhline(SQRT2, color="green", ls="--", lw=1, label="√2")
                a.axhline(SAT_THRESH, color="purple", ls=":", lw=1)
                a.legend(loc="best", fontsize=9)

        # 4번째: d_Cov 포화 분해 (구간별 비포화 비율 + 누적 d_phys 분포)
        a = ax[1, 1]
        ctr3, med_c, _, _, cnt3 = binned_stats(y, x_cov, edges)
        frac_bin = np.array([
            (x_cov[(np.clip(np.digitize(y, edges)-1, 0, N_BIN-1)) == b] < SAT_THRESH).mean()
            if cnt3[b] > 0 else np.nan for b in range(N_BIN)])
        a.bar(ctr3, frac_bin * 100, width=(edges[1]-edges[0])*0.9,
              color="teal", alpha=0.6, label="비포화(d_Cov<1.40) 비율")
        a.set_xlabel("d_phys [m]")
        a.set_ylabel("비포화 쌍 비율 [%]", color="teal")
        a.tick_params(axis="y", labelcolor="teal")
        a.set_title(f"d_Cov 비포화 비율 vs d_phys  (전체 비포화 {frac:.1f}%)")
        a.grid(True, ls=":", alpha=0.4)
        a2 = a.twinx()
        a2.plot(ctr3, med_c, "-s", color="crimson", ms=3, label="구간 중앙 d_Cov")
        a2.axhline(SQRT2, color="green", ls="--", lw=1)
        a2.axhline(SAT_THRESH, color="purple", ls=":", lw=1)
        a2.set_ylabel("중앙 d_Cov [-]", color="crimson")
        a2.tick_params(axis="y", labelcolor="crimson")
        a2.axhline(plateau, color="gray", ls=":", lw=1)
        a.axvline(eff, color="black", ls="-.", lw=1.2)
        a.annotate(f"포화 시작 ~{eff:.0f}m\n(이후 d_Cov≈{plateau:.2f})",
                   xy=(eff, 50), xytext=(eff + 20, 70), fontsize=9,
                   arrowprops=dict(arrowstyle="->"))

        fig.suptitle("[P2G 단계3] 거리 항 vs 물리거리(UE-UE) + d_Cov 포화 분석",
                     fontsize=14, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        png = os.path.join(OUT_DIR, f"P2G_terms_vs_dphys_{stamp}.png")
        fig.savefig(png, dpi=150)
        plt.close(fig)
        log(f"\n[viz] saved -> {png}")

        # ===== 쌍 테이블 저장 =====
        out = os.path.join(OUT_DIR, f"P2G_PairTable_{stamp}.npz")
        np.savez_compressed(
            out, X=X.astype(np.float32), y=y.astype(np.float32),
            pair_i=iu[0].astype(np.int32), pair_j=iu[1].astype(np.int32),
            col_names=np.array(["d_RSRP_dB", "d_PDP_ns", "d_Cov"]),
            distmat=os.path.basename(dmf))
        log(f"[save] pair table -> {out} ({os.path.getsize(out)/1e6:.1f} MB)")
        log(f"[done] M={M} 쌍, 단계4(NNLS) 입력 준비 완료")
    finally:
        log_f.close()


if __name__ == "__main__":
    main()
