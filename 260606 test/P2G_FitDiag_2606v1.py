# -*- coding: utf-8 -*-
"""
P2G_FitDiag_2606v1.py
=====================
[단계 4 진단] R²<0 의 원인 분해: origin-through 제약 vs 물리 한계.

요청 점검 3종
  (1) intercept 허용 적합 비교
      - origin-through NNLS (b=0, w>=0)         [기존]
      - intercept + 비음수   (b free, w>=0)      scipy.optimize.lsq_linear
      - intercept + 자유(OLS) (b free, w free)    참고용 np.linalg.lstsq
      => intercept 넣어 R²가 확 오르면 '원점 통과' 가정이 범인.
  (2) 근거리(d_phys<40m) 한정 R²  + 거리 구간별 R² (0-40 / 40-80 / 80-max / full)
      => 메트릭이 유효한 거리 범위 확인.
  (3) RMS 스케일 후 세 열(d_RSRP/d_PDP/d_Cov) 분포 전체 점검
      => RSRP/PDP가 d_Cov에 눌렸는지(스케일 문제) 판정.

R² 정의 (기울기-1, 1:1선 기준): R²=1-Σ(y-pred)²/Σ(y-ȳ)².
----------------------------------------------------------------------
실행 환경 : Python 3.10.12 / numpy 2.2.6 / scipy 1.15.3 / matplotlib
서버 dclserver78 (경량 -> CPU/numpy)
입력 : P2G_PairTable_Results/P2G_PairTable_*.npz
----------------------------------------------------------------------
"""

import os
import glob
import datetime

import numpy as np
from scipy.optimize import nnls, lsq_linear

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

for _fp in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",):
    if os.path.exists(_fp):
        fm.fontManager.addfont(_fp)
        plt.rcParams["font.family"] = fm.FontProperties(fname=_fp).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PT_DIR   = os.path.join(BASE_DIR, "P2G_PairTable_Results")
OUT_DIR  = os.path.join(BASE_DIR, "P2G_FitDiag_Results")
COLS = ["d_RSRP", "d_PDP", "d_Cov"]
NEAR = 40.0


def latest_pairtable():
    fs = sorted(glob.glob(os.path.join(PT_DIR, "P2G_PairTable_*.npz")))
    if not fs:
        raise FileNotFoundError("쌍 테이블 npz 없음.")
    return fs[-1]


def r2_mse(y, pred):
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot, ss_res / len(y)


def fit(X, y, mode):
    """mode: 'ot'(origin-through nnls) | 'int_nn'(intercept+nonneg) | 'int_ols'."""
    s = np.sqrt(np.mean(X ** 2, axis=0))
    Xn = X / s
    if mode == "ot":
        wn, _ = nnls(Xn, y)
        w, b = wn / s, 0.0
        pred = X @ w
    elif mode == "int_nn":
        A = np.column_stack([Xn, np.ones(len(y))])
        lb = np.r_[np.zeros(X.shape[1]), -np.inf]
        ub = np.r_[np.full(X.shape[1], np.inf), np.inf]
        res = lsq_linear(A, y, bounds=(lb, ub), max_iter=200)
        wn, b = res.x[:-1], float(res.x[-1])
        w = wn / s
        pred = X @ w + b
    elif mode == "int_ols":
        A = np.column_stack([Xn, np.ones(len(y))])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        wn, b = coef[:-1], float(coef[-1])
        w = wn / s
        pred = X @ w + b
    r2, mse = r2_mse(y, pred)
    return w, b, r2, mse, pred


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_f = open(os.path.join(OUT_DIR, f"P2G_fitdiag_report_{stamp}.log"),
                 "w", encoding="utf-8")

    def log(*a):
        m = " ".join(str(x) for x in a); print(m); log_f.write(m + "\n"); log_f.flush()

    try:
        ptf = latest_pairtable()
        d = np.load(ptf, allow_pickle=True)
        X = d["X"].astype(np.float64)
        y = d["y"].astype(np.float64)
        M = len(y)
        log(f"[load] {os.path.basename(ptf)}  X{X.shape}  d_phys 평균={y.mean():.2f}m")

        # ===== (3) RMS 스케일 후 세 열 분포 =====
        s = np.sqrt(np.mean(X ** 2, axis=0))
        Xn = X / s
        qs = [0, 1, 5, 25, 50, 75, 95, 99, 100]
        log("\n[(3) RMS 정규화 후 열 분포]  (scale-only, 중심화 없음)")
        log(f"  RMS scale s = {dict(zip(COLS, np.round(s,3)))}")
        log(f"  {'col':<8} " + " ".join(f"{q}%" for q in qs) + "   mean  std  std/mean")
        for k in range(3):
            v = Xn[:, k]
            pc = np.percentile(v, qs)
            log(f"  {COLS[k]:<8} " + " ".join(f"{p:.3f}" for p in pc) +
                f"   {v.mean():.3f} {v.std():.3f} {v.std()/v.mean():.3f}")
        log("  => std/mean 작을수록 변별력 낮음(상수에 가까움).")

        # ===== (1) intercept 비교 (full) =====
        log("\n[(1) intercept 비교 — full 3항]  (R²=기울기-1 기준)")
        log(f"  {'mode':<22}{'b[m]':>9}{'w_R':>9}{'w_P':>9}{'w_C':>9}"
            f"{'R²':>9}{'RMSE[m]':>9}")
        modes = [("origin-through(NNLS)", "ot"),
                 ("intercept+비음수", "int_nn"),
                 ("intercept+자유(OLS)", "int_ols")]
        fits = {}
        for nm, mc in modes:
            w, b, r2, mse, pred = fit(X, y, mc)
            fits[mc] = (w, b, r2, mse, pred)
            log(f"  {nm:<22}{b:>9.2f}{w[0]:>9.3f}{w[1]:>9.4f}{w[2]:>9.3f}"
                f"{r2:>9.4f}{np.sqrt(mse):>9.2f}")

        # RSRP+PDP (d_Cov 제외) intercept 비교 — d_Cov 의사절편 효과 분리
        log("\n[(1b) RSRP+PDP 만 (d_Cov 제외) intercept 비교]")
        XP = X[:, :2]
        for nm, mc in modes:
            w, b, r2, mse, _ = fit(XP, y, mc)
            log(f"  {nm:<22}{b:>9.2f}{w[0]:>9.3f}{w[1]:>9.4f}{'-':>9}"
                f"{r2:>9.4f}{np.sqrt(mse):>9.2f}")

        # ===== (2) 거리 구간별 R² =====
        log("\n[(2) 거리 구간별 R²]  (전역 적합 weight로 예측 후 구간별 R² 재계산)")
        bins = [(0, NEAR), (NEAR, 80), (80, 1e9)]
        blab = [f"<{NEAR:.0f}m", "40-80m", "80m+"]
        for mc, nm in [("ot", "origin-through"), ("int_nn", "intercept+비음수")]:
            w, b, _, _, pred = fits[mc]
            log(f"  [{nm}] 전역 R²={r2_mse(y,pred)[0]:.4f}")
            for (lo, hi), lab in zip(bins, blab):
                m = (y >= lo) & (y < hi)
                r2b, _ = r2_mse(y[m], pred[m])
                log(f"     {lab:<7} n={m.sum():>7}  R²={r2b:>8.4f}")

        # 근거리 '재적합'(그 구간만으로 fit) — 유효성 상한
        log("\n[(2b) 근거리(<40m) 쌍만으로 재적합한 상한 R²]")
        near = y < NEAR
        for mc, nm in [("ot", "origin-through"), ("int_nn", "intercept+비음수")]:
            w, b, r2, mse, _ = fit(X[near], y[near], mc)
            log(f"  [{nm}] n={near.sum()}  b={b:.2f}  "
                f"w=({w[0]:.3f},{w[1]:.4f},{w[2]:.3f})  R²={r2:.4f} RMSE={np.sqrt(mse):.2f}m")

        # ===== 그래프 =====
        rng = np.random.default_rng(0)
        sidx = rng.choice(M, size=min(80000, M), replace=False)
        fig, ax = plt.subplots(1, 3, figsize=(18, 5.6))
        ymax = y.max()
        for j, (mc, nm) in enumerate([("ot", "origin-through(NNLS)"),
                                      ("int_nn", "intercept+비음수")]):
            w, b, r2, mse, pred = fits[mc]
            a = ax[j]
            a.scatter(y[sidx], pred[sidx], s=2, alpha=0.04, color="navy",
                      edgecolors="none", rasterized=True)
            a.plot([0, ymax], [0, ymax], "r--", lw=1.5, label="1:1 (기울기1)")
            a.set_xlabel("d_phys [m]"); a.set_ylabel("d_metric [m]")
            a.set_title(f"{nm}\nR²={r2:.3f}, RMSE={np.sqrt(mse):.1f}m, b={b:.1f}m")
            a.set_xlim(0, ymax); a.set_ylim(0, ymax)
            a.grid(True, ls=":", alpha=0.4); a.legend(loc="upper left", fontsize=9)

        # 거리 구간별 R² 막대 (두 모드)
        a = ax[2]
        labels = blab + ["full"]
        xpos = np.arange(len(labels))
        for jj, (mc, nm, col) in enumerate(
                [("ot", "origin-through", "steelblue"),
                 ("int_nn", "intercept", "indianred")]):
            _, _, _, _, pred = fits[mc]
            r2s = []
            for (lo, hi) in bins:
                m = (y >= lo) & (y < hi)
                r2s.append(r2_mse(y[m], pred[m])[0])
            r2s.append(r2_mse(y, pred)[0])
            a.bar(xpos + (jj - 0.5) * 0.4, r2s, width=0.4, color=col, label=nm)
            for x, v in zip(xpos + (jj - 0.5) * 0.4, r2s):
                a.text(x, v, f"{v:.2f}", ha="center",
                       va="bottom" if v >= 0 else "top", fontsize=8)
        a.axhline(0, color="k", lw=0.8)
        a.set_xticks(xpos); a.set_xticklabels(labels)
        a.set_ylabel("R² (기울기-1)"); a.set_title("거리 구간별 R²")
        a.grid(True, ls=":", alpha=0.4); a.legend(loc="best", fontsize=9)

        fig.suptitle("[P2G 단계4 진단] origin-through vs intercept / 거리 구간별 유효성",
                     fontsize=13, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        png = os.path.join(OUT_DIR, f"P2G_fitdiag_{stamp}.png")
        fig.savefig(png, dpi=150); plt.close(fig)
        log(f"\n[viz] saved -> {png}")
        log("[done] 진단 완료")
    finally:
        log_f.close()


if __name__ == "__main__":
    main()
