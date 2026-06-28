# -*- coding: utf-8 -*-
"""
P2G_WeightFit_2606v1.py
=======================
[단계 4] NNLS 가중치 학습.

목표
  d_metric(i,j) = w_R·d_RSRP + w_P·d_PDP + w_C·d_Cov  ≈  d_phys(i,j)[m]
  를 origin-through(절편 없음), w>=0 제약으로 적합.

핵심 절차
  1) scale-only(RMS) 정규화 : X_norm[:,k] = X[:,k] / s_k,  s_k=sqrt(mean(X_k^2))
     - 중심화(평균 제거) 금지 -> origin-through 보존
  2) scipy.optimize.nnls(X_norm, y) -> w_norm (>=0)
  3) 원스케일 역변환 : w_k = w_norm_k / s_k
     - 단위: w_R[m/dB], w_P[m/ns], w_C[m]

추가 산출 (요청)
  (a) ablation 5종: RSRP-only / PDP-only / Cov-only / RSRP+PDP / full
      각각 weight, R²(기울기-1 기준; pred=Xw 를 y와 직접 비교), MSE
  (b) 각 항 기여도  w_k · mean(X_k)  [평균 m]
  (c) full 에서 w_C≈0 이면 d_phys<40m 쌍만으로 재적합해 d_Cov 근거리 기여 확인
  (d) RMS 스케일 후 d_Cov 열 분포 점검
  + 음수 weight 여부 / NNLS 수렴(residual) 로그

R² 정의 (기울기-1 기준)
  pred = X @ w (= d_metric). 1:1 선 위에 놓이길 원함 -> y 와 직접 비교:
  R² = 1 - SS_res/SS_tot,  SS_res=Σ(y-pred)²,  SS_tot=Σ(y-ȳ)²
----------------------------------------------------------------------
실행 환경 : Python 3.10.12 / numpy 2.2.6 / scipy 1.15.3 / matplotlib
서버 dclserver78 (경량 -> CPU/numpy)
입력 : P2G_PairTable_Results/P2G_PairTable_*.npz
----------------------------------------------------------------------
"""

import os
import glob
import csv
import datetime

import numpy as np
from scipy.optimize import nnls

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
OUT_DIR  = os.path.join(BASE_DIR, "P2G_WeightFit_Results")

COLS = ["d_RSRP", "d_PDP", "d_Cov"]
UNITS = ["m/dB", "m/ns", "m"]
NEAR_THRESH = 40.0   # (c) 근거리 재적합 기준 [m]
WC_NEAR_ZERO = 1e-6  # w_C 가 '0 근처'로 볼 임계 (역스케일 후)


def latest_pairtable():
    fs = sorted(glob.glob(os.path.join(PT_DIR, "P2G_PairTable_*.npz")))
    if not fs:
        raise FileNotFoundError("쌍 테이블 npz 없음. 먼저 단계3 실행.")
    return fs[-1]


def fit_nnls(Xsub, y, scales):
    """scale-only(RMS) 정규화 후 NNLS. 원스케일 weight·진단 반환."""
    Xn = Xsub / scales                      # 열별 RMS 정규화 (중심화 X)
    w_norm, rnorm = nnls(Xn, y)             # w_norm >= 0
    w = w_norm / scales                     # 원스케일 역변환
    pred = Xsub @ w
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot              # 기울기-1 기준 (pred vs y 직접)
    mse = ss_res / len(y)
    return w, r2, mse, float(rnorm), w_norm


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_f = open(os.path.join(OUT_DIR, f"P2G_weightfit_report_{stamp}.log"),
                 "w", encoding="utf-8")

    def log(*a):
        m = " ".join(str(x) for x in a); print(m); log_f.write(m + "\n"); log_f.flush()

    try:
        ptf = latest_pairtable()
        d = np.load(ptf, allow_pickle=True)
        X = d["X"].astype(np.float64)       # (M,3) [dB, ns, -]
        y = d["y"].astype(np.float64)       # (M,) [m]
        M = X.shape[0]
        log(f"[load] {os.path.basename(ptf)}  X{X.shape}, y{y.shape}")
        log(f"[unit] X=[d_RSRP(dB), d_PDP(ns), d_Cov(-)], y=d_phys(m)")

        # 전역 RMS 스케일 (scale-only)
        scales = np.sqrt(np.mean(X ** 2, axis=0))
        log(f"[scale] RMS s = {dict(zip(COLS, np.round(scales,4)))}")

        # ===== (d) RMS 스케일 후 d_Cov 열 분포 =====
        Xn = X / scales
        qs = [0, 1, 5, 25, 50, 75, 95, 99, 100]
        cov_n = Xn[:, 2]
        pct = np.percentile(cov_n, qs)
        log("\n[(d) RMS 정규화 후 d_Cov 열 분포]")
        log("  percentile " + " ".join(f"{q}%={v:.4f}" for q, v in zip(qs, pct)))
        log(f"  mean={cov_n.mean():.4f} std={cov_n.std():.4f} "
            f"(std/mean={cov_n.std()/cov_n.mean():.3f})  "
            f"-> 분산 작으면 NNLS에서 d_Cov 식별력 낮음")

        # ===== (a) ablation 5종 =====
        ablations = {
            "RSRP-only": [0],
            "PDP-only":  [1],
            "Cov-only":  [2],
            "RSRP+PDP":  [0, 1],
            "full":      [0, 1, 2],
        }
        rows = []
        full_res = None
        log("\n[(a) Ablation 5종]  (R²=기울기-1 기준, w=원스케일)")
        log(f"  {'model':<10} {'w_R[m/dB]':>11} {'w_P[m/ns]':>11} "
            f"{'w_C[m]':>9} {'R²':>8} {'MSE':>10} {'RMSE[m]':>9}")
        for name, cols in ablations.items():
            w_sub, r2, mse, rnorm, wn = fit_nnls(
                X[:, cols], y, scales[cols])
            w_full = np.zeros(3)
            for c, wv in zip(cols, w_sub):
                w_full[c] = wv
            rows.append((name, *w_full, r2, mse, np.sqrt(mse), rnorm))
            log(f"  {name:<10} {w_full[0]:>11.4f} {w_full[1]:>11.4f} "
                f"{w_full[2]:>9.4f} {r2:>8.4f} {mse:>10.2f} {np.sqrt(mse):>9.3f}")
            if name == "full":
                full_res = (w_full, r2, mse)

        # 음수 weight 점검 (NNLS는 >=0이지만 검증)
        neg = any(any(np.array(r[1:4]) < -1e-12) for r in rows)
        log(f"\n[check] 음수 weight 존재? {neg} (NNLS 제약상 정상=False)")

        # ===== (b) 항 기여도 (full) =====
        w_full, r2_full, mse_full = full_res
        meanX = X.mean(axis=0)
        contrib = w_full * meanX
        log("\n[(b) 항 기여도 = w_k · mean(X_k)] (full, 평균 m)")
        tot = contrib.sum()
        for k in range(3):
            log(f"  {COLS[k]:<8} w={w_full[k]:.4f} {UNITS[k]:<5} "
                f"× mean={meanX[k]:.4f} = {contrib[k]:8.3f} m "
                f"({100*contrib[k]/tot:5.1f}%)")
        log(f"  합계 d_metric 평균 ≈ {tot:.3f} m  (실제 d_phys 평균 {y.mean():.3f} m)")

        # ===== (c) w_C≈0 이면 근거리 재적합 =====
        log(f"\n[(c) full w_C={w_full[2]:.6g}]")
        near_res = None
        if w_full[2] <= WC_NEAR_ZERO:
            near = y < NEAR_THRESH
            log(f"  w_C≈0 -> d_phys<{NEAR_THRESH:.0f}m 쌍({near.sum()}개)만으로 재적합")
            sc_near = np.sqrt(np.mean(X[near] ** 2, axis=0))
            w_n, r2_n, mse_n, _, _ = fit_nnls(X[near], y[near], sc_near)
            near_res = (w_n, r2_n, mse_n, int(near.sum()))
            log(f"  [근거리 full] w_R={w_n[0]:.4f} w_P={w_n[1]:.4f} "
                f"w_C={w_n[2]:.4f} | R²={r2_n:.4f} MSE={mse_n:.2f}")
            log(f"  -> d_Cov 근거리 기여 w_C={w_n[2]:.4f} m "
                f"({'유의미' if w_n[2] > WC_NEAR_ZERO else '여전히 0'})")
        else:
            log(f"  w_C>0 -> 전역 적합에서 d_Cov 식별됨 (근거리 재적합 생략)")

        # ===== 저장 : CSV + NPZ =====
        csv_path = os.path.join(OUT_DIR, f"P2G_weights_ablation_{stamp}.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            wcsv = csv.writer(f)
            wcsv.writerow(["model", "w_R[m/dB]", "w_P[m/ns]", "w_C[m]",
                           "R2", "MSE", "RMSE[m]", "nnls_resid_norm"])
            for r in rows:
                wcsv.writerow([r[0]] + [f"{v:.6g}" for v in r[1:]])
        log(f"\n[save] ablation table -> {csv_path}")

        np.savez_compressed(
            os.path.join(OUT_DIR, f"P2G_weights_{stamp}.npz"),
            w_full=w_full, r2_full=r2_full, mse_full=mse_full,
            scales=scales, col_names=np.array(COLS), units=np.array(UNITS),
            ablation_names=np.array(list(ablations.keys())),
            ablation_w=np.array([r[1:4] for r in rows]),
            ablation_r2=np.array([r[4] for r in rows]),
            ablation_mse=np.array([r[5] for r in rows]),
            pairtable=os.path.basename(ptf))

        # ===== 진단 그래프 (ablation R²/RMSE + 기여도) =====
        fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
        names = [r[0] for r in rows]
        r2s = [r[4] for r in rows]
        rmses = [r[6] for r in rows]
        xpos = np.arange(len(names))
        b1 = ax[0].bar(xpos - 0.2, r2s, width=0.4, color="steelblue", label="R²")
        ax[0].set_ylabel("R² (기울기-1 기준)", color="steelblue")
        ax[0].tick_params(axis="y", labelcolor="steelblue")
        ax[0].set_xticks(xpos); ax[0].set_xticklabels(names, rotation=20)
        ax[0].set_title("Ablation: R² / RMSE")
        ax[0].grid(True, ls=":", alpha=0.4)
        for x, v in zip(xpos - 0.2, r2s):
            ax[0].text(x, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        ax0b = ax[0].twinx()
        ax0b.bar(xpos + 0.2, rmses, width=0.4, color="indianred", label="RMSE[m]")
        ax0b.set_ylabel("RMSE [m]", color="indianred")
        ax0b.tick_params(axis="y", labelcolor="indianred")
        for x, v in zip(xpos + 0.2, rmses):
            ax0b.text(x, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)

        ax[1].bar(COLS, contrib, color=["#4c72b0", "#dd8452", "#55a868"])
        ax[1].set_ylabel("기여도 w_k·mean(X_k) [m]")
        ax[1].set_ylim(0, max(contrib) * 1.18)
        ax[1].set_title(f"full 모델 항 기여도 (합≈{tot:.1f}m, d_phys평균 {y.mean():.1f}m)")
        ax[1].grid(True, ls=":", alpha=0.4)
        for i, v in enumerate(contrib):
            ax[1].text(i, v, f"{v:.2f}m\n({100*v/tot:.0f}%)",
                       ha="center", va="bottom", fontsize=9)
        fig.suptitle("[P2G 단계4] NNLS 가중치 적합 — ablation & 기여도",
                     fontsize=13, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        png = os.path.join(OUT_DIR, f"P2G_weightfit_{stamp}.png")
        fig.savefig(png, dpi=150)
        plt.close(fig)
        log(f"[viz] saved -> {png}")
        log("\n[done] 단계4 완료 — full weight·ablation·기여도 산출")
    finally:
        log_f.close()


if __name__ == "__main__":
    main()
