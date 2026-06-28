# -*- coding: utf-8 -*-
"""
P2G_Retarget_2606v1.py
======================
[타깃 재설정] 단일-BS 채널 메트릭(d_RSRP/d_PDP/d_Cov_BS)이 'UE-UE 거리'가 아니라
무엇을 잘 맞추는지 재평가.

가설
  단일 BS 채널은 그 BS 기준의 (거리, 각도) 정보를 담는다. 그래서 UE-UE 유클리드
  거리(d_phys)보다 BS-기준 양:
    - d_range  = |range_i - range_j|     (각 UE의 BS까지 거리 차)   [m]
    - d_angle  = BS에서 본 방위각 차       [deg]
  를 더 잘 설명할 수 있다.

타깃
  T_phys  = ||pos_i - pos_j||                (기존 UE-UE)
  T_range = |r_i - r_j|,  r=||pos-BS||
  T_angle = wrap(|az_i - az_j|),  az=atan2(dy,dx)  (0~180deg)

비교
  기존 3특징(d_RSRP,d_PDP,d_Cov_BS)을 각 타깃에 OLS(절편) 적합 -> R²/corr.
  + 단일 특징 corr (특히 d_RSRP vs d_range: 경로손실∝거리 기대).
----------------------------------------------------------------------
실행 환경 : Python 3.10.12 / numpy 2.2.6 (경량 CPU)
입력 : P2G_DistMat_*.npz(D_rsrp/D_pdp/D_cov/pos), BS_XY 상수
----------------------------------------------------------------------
"""

import os
import glob
import datetime

import numpy as np

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
DM_DIR   = os.path.join(BASE_DIR, "P2G_DistTerms_Results")
OUT_DIR  = os.path.join(BASE_DIR, "P2G_Retarget_Results")
BS_XY = np.array([-51.561, -21.794])     # Area1 BS (빌더와 동일)
NAMES = ["d_RSRP", "d_PDP", "d_Cov_BS"]


def latest(d, pat):
    return sorted(glob.glob(os.path.join(d, pat)))[-1]


def r2_corr(y, pred):
    ss = np.sum((y - pred) ** 2)
    return 1 - ss / np.sum((y - y.mean()) ** 2), float(np.corrcoef(pred, y)[0, 1])


def fit_ols(Xc, y):
    s = np.sqrt(np.mean(Xc ** 2, axis=0))
    A = np.column_stack([Xc / s, np.ones(len(y))])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return r2_corr(y, A @ coef)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_f = open(os.path.join(OUT_DIR, f"P2G_retarget_report_{stamp}.log"),
                 "w", encoding="utf-8")

    def log(*a):
        m = " ".join(str(x) for x in a); print(m); log_f.write(m + "\n"); log_f.flush()

    try:
        dm = np.load(latest(DM_DIR, "P2G_DistMat_*.npz"), allow_pickle=True)
        D_rsrp, D_pdp, D_cov, pos = dm["D_rsrp"], dm["D_pdp"], dm["D_cov"], dm["pos"]
        N = pos.shape[0]
        iu = np.triu_indices(N, 1)

        # BS 기준 거리/방위각
        rel = pos - BS_XY
        rng = np.sqrt((rel ** 2).sum(1))                  # range[m]
        az = np.degrees(np.arctan2(rel[:, 1], rel[:, 0])) # [-180,180]
        log(f"[geom] BS={BS_XY}, range[m] min/med/max="
            f"{rng.min():.1f}/{np.median(rng):.1f}/{rng.max():.1f}")

        # 타깃들 (상삼각)
        T_phys = np.sqrt(((pos[:, None] - pos[None]) ** 2).sum(-1))[iu]
        T_range = np.abs(rng[:, None] - rng[None])[iu]
        dazm = np.abs(az[:, None] - az[None])
        dazm = np.minimum(dazm, 360 - dazm)               # 0~180
        T_angle = dazm[iu]
        targets = {"T_phys(UE-UE)": T_phys, "T_range(BS거리차)": T_range,
                   "T_angle(BS방위차)": T_angle}

        X = np.stack([D_rsrp[iu], D_pdp[iu], D_cov[iu]], axis=1).astype(np.float64)

        # 단일 특징 corr (타깃별)
        log("\n[단일 특징 corr (타깃별)]")
        log(f"  {'feature':<10}" + "".join(f"{t:>20}" for t in targets))
        for k in range(3):
            row = "".join(f"{np.corrcoef(X[:,k],t)[0,1]:>20.4f}" for t in targets.values())
            log(f"  {NAMES[k]:<10}{row}")

        # 3특징 OLS 적합 (타깃별)
        log("\n[3특징 OLS(절편) 적합: R² (corr)]")
        res = {}
        for tname, y in targets.items():
            r2, r = fit_ols(X, y)
            res[tname] = (r2, r)
            log(f"  {tname:<20} R²={r2:.4f}  corr={r:.4f}")

        # 그래프: 타깃별 R² + (best 타깃) 단일특징 corr
        fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
        tn = list(targets.keys())
        r2s = [res[t][0] for t in tn]
        ax[0].bar(range(len(tn)), r2s, color=["#4c72b0", "#dd8452", "#55a868"])
        ax[0].set_xticks(range(len(tn))); ax[0].set_xticklabels(tn, fontsize=9, rotation=10)
        ax[0].set_ylabel("R² (3특징 OLS)")
        ax[0].set_title("타깃별 메트릭 설명력")
        ax[0].grid(True, ls=":", alpha=0.4)
        for i, v in enumerate(r2s):
            ax[0].text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=10)

        # corr 히트맵 (특징 x 타깃)
        C = np.array([[np.corrcoef(X[:, k], t)[0, 1] for t in targets.values()]
                      for k in range(3)])
        im = ax[1].imshow(C, cmap="RdBu_r", vmin=-0.6, vmax=0.6, aspect="auto")
        ax[1].set_xticks(range(3)); ax[1].set_xticklabels(tn, fontsize=8, rotation=12)
        ax[1].set_yticks(range(3)); ax[1].set_yticklabels(NAMES, fontsize=9)
        for i in range(3):
            for j in range(3):
                ax[1].text(j, i, f"{C[i,j]:.3f}", ha="center", va="center",
                           fontsize=9, color="black")
        ax[1].set_title("특징×타깃 corr")
        fig.colorbar(im, ax=ax[1], fraction=0.046)
        fig.suptitle("[P2G 타깃 재설정] 단일-BS 메트릭은 무엇을 맞추나",
                     fontsize=13, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        png = os.path.join(OUT_DIR, f"P2G_retarget_{stamp}.png")
        fig.savefig(png, dpi=150); plt.close(fig)
        log(f"\n[viz] saved -> {png}")
        log("[done] 타깃 재설정 분석 완료")
    finally:
        log_f.close()


if __name__ == "__main__":
    main()
