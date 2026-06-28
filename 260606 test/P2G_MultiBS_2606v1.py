# -*- coding: utf-8 -*-
"""
P2G_MultiBS_2606v1.py
=====================
[multi-BS 검증] BS 1개 -> 3개로 늘리면 UE-UE 물리거리 복원이 얼마나 좋아지나.

배경
  단일 BS(Area1)에서는 어떤 채널 특징을 더해도 d_phys 복원 R²≈0.14 천장.
  원인 = 방향/거리 모호성. multi-BS(삼각측량)면 풀린다는 가설을 직접 검증.

데이터
  260512 RT Result Data/channel_data_260531_GHM_Twin_v0_1.npz
    rsrp_all (3, 5071)  : BS 3개 x RX 5071, RSRP[dB]
    rx_positions (5071,2), tx_positions(3,3)
  (R_TX/R_RX 는 (1,1) SISO -> 공분산 항 없음. RSRP 핑거프린트만 사용.)

방법
  쌍 특징: f_b(i,j) = |rsrp_b,i - rsrp_b,j|  (BS b 별)
  타깃: d_phys = ||rx_i - rx_j||
  OLS(절편) 적합: 1-BS / 2-BS / 3-BS 조합별 R²/corr 비교.
  + 단일 BS 평균(3개 각각 1-BS) vs 3-BS 종합.
----------------------------------------------------------------------
실행 환경 : Python 3.10.12 / numpy 2.2.6 (경량 CPU)
입력 : ../260512 RT Result Data/channel_data_260531_GHM_Twin_v0_1.npz
----------------------------------------------------------------------
"""

import os
import datetime
import itertools

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
WS_ROOT  = os.path.dirname(BASE_DIR)
CH_NPZ   = os.path.join(WS_ROOT, "260512 RT Result Data",
                        "channel_data_260531_GHM_Twin_v0_1.npz")
OUT_DIR  = os.path.join(BASE_DIR, "P2G_MultiBS_Results")
N_SUB = 3000      # RX 서브샘플 (쌍수 ~4.5M)
SEED = 42


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
    log_f = open(os.path.join(OUT_DIR, f"P2G_multibs_report_{stamp}.log"),
                 "w", encoding="utf-8")

    def log(*a):
        m = " ".join(str(x) for x in a); print(m); log_f.write(m + "\n"); log_f.flush()

    try:
        d = np.load(CH_NPZ, allow_pickle=True)
        rsrp = np.asarray(d["rsrp_all"], dtype=np.float64)     # (3, Nall)
        rxp = np.asarray(d["rx_positions"], dtype=np.float64)  # (Nall,2)
        txp = np.asarray(d["tx_positions"], dtype=np.float64)
        nbs = rsrp.shape[0]
        log(f"[load] BS={nbs}, RX={rxp.shape[0]}, BS위치=\n{txp}")

        # 유효 RX (모든 BS RSRP 유한)
        good = np.all(np.isfinite(rsrp), axis=0)
        idx_all = np.where(good)[0]
        rng = np.random.default_rng(SEED)
        idx = np.sort(rng.choice(idx_all, size=min(N_SUB, len(idx_all)), replace=False))
        R = rsrp[:, idx]                 # (3, n)
        P = rxp[idx]                     # (n,2)
        n = len(idx)
        log(f"[sub] 유효 RX={len(idx_all)}, 서브샘플 n={n}, 쌍={n*(n-1)//2}")

        iu = np.triu_indices(n, 1)
        y = np.sqrt(((P[:, None] - P[None]) ** 2).sum(-1))[iu]   # d_phys
        log(f"[phys] d_phys[m] min/med/max={y.min():.1f}/{np.median(y):.1f}/{y.max():.1f}")

        # 쌍 특징: BS별 |Δrsrp|
        F = np.stack([np.abs(R[b][:, None] - R[b][None])[iu] for b in range(nbs)],
                     axis=1)            # (M, 3)
        for b in range(nbs):
            log(f"  BS{b}: |Δrsrp| corr(d_phys)={np.corrcoef(F[:,b],y)[0,1]:.4f}")

        # 조합별 OLS
        log("\n[조합별 OLS(절편) R²/corr]  (d_phys 복원)")
        combos = []
        for b in range(nbs):
            combos.append((f"BS{b} only", [b]))
        for c in itertools.combinations(range(nbs), 2):
            combos.append(("BS" + "+".join(map(str, c)), list(c)))
        combos.append(("BS0+1+2 (3-BS)", list(range(nbs))))
        res = {}
        for lab, cols in combos:
            r2, r = fit_ols(F[:, cols], y)
            res[lab] = (r2, r)
            log(f"  {lab:<16} R²={r2:.4f}  corr={r:.4f}")

        best1 = max(res[f"BS{b} only"][0] for b in range(nbs))
        full = res["BS0+1+2 (3-BS)"][0]
        log(f"\n[핵심] (RSRP) 단일 BS 최고 R²={best1:.4f} -> 3-BS R²={full:.4f} "
            f"({full-best1:+.4f}, x{full/max(best1,1e-9):.1f})")

        # ===== 이상적 거리(ToA로 얻을 수 있는 참 BS거리) 특징 비교 =====
        # multi-BS '개념'의 상한: RSRP 대신 정확한 BS거리(=ToA*c)를 쓰면?
        rng_true = np.sqrt(((P[:, None, :] - txp[None, :, :2]) ** 2).sum(-1))  # (n,3) 2D
        Fr = np.stack([np.abs(rng_true[:, b][:, None] - rng_true[:, b][None])[iu]
                       for b in range(nbs)], axis=1)
        log("\n[이상적 참-BS거리(|Δrange_b|, ToA 상한) 조합별 R²]")
        ideal = {}
        for b in range(nbs):
            r2, r = fit_ols(Fr[:, [b]], y); ideal[f"range BS{b}"] = r2
            log(f"  range BS{b} only   R²={r2:.4f} corr={r:.4f}")
        r2_3r, r_3r = fit_ols(Fr, y)
        ideal["range 3-BS"] = r2_3r
        log(f"  range 3-BS         R²={r2_3r:.4f} corr={r_3r:.4f}")
        log(f"\n[대비] RSRP 3-BS R²={full:.4f}  vs  이상적 거리 3-BS(pairwise) R²={r2_3r:.4f}")

        # ===== 정식 삼각측량: 참 거리 -> 위치추정 -> 쌍거리 (메트릭 형태 한계 분리) =====
        a = txp[:, :2]                                   # (3,2) BS 2D
        M = 2.0 * (a[1:] - a[0])                         # (2,2)
        Minv = np.linalg.inv(M)
        an2 = (a ** 2).sum(1)                            # |a_b|^2
        # c_b = |a_b|^2-|a_0|^2 - (r_b^2 - r_0^2)
        cc = (an2[1:] - an2[0])[None, :] - (rng_true[:, 1:] ** 2 - rng_true[:, [0]] ** 2)
        pos_est = (Minv @ cc.T).T                        # (n,2)
        d_tri = np.sqrt(((pos_est[:, None] - pos_est[None]) ** 2).sum(-1))[iu]
        r2_tri, r_tri = r2_corr(y, d_tri)
        log(f"[삼각측량] 참거리->위치추정->쌍거리 R²={r2_tri:.4f} corr={r_tri:.4f} "
            f"(위치추정 RMSE={np.sqrt(np.mean((pos_est-P)**2)):.2e}m)")
        log("\n[결론]")
        log("  - RSRP pairwise(3-BS) R²=%.3f : RSRP는 NLoS shadowing으로 거리추정 noisy." % full)
        log("  - 이상적거리 pairwise(3-BS) R²=%.3f : 정보가 완벽해도 '가중합 pairwise' 형태는 한계." % r2_3r)
        log("  - 삼각측량(위치추정후 거리) R²=%.3f : 같은 정보를 '위치 복원'에 쓰면 거의 완벽." % r2_tri)
        log("  => 지렛대 두 개: (1) 정확한 거리량(ToA), (2) pairwise 가중합이 아닌 '위치 추정' 모델.")

        # 그래프
        fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
        labs = [c[0] for c in combos] + ["[이상]range 3-BS", "[정식]삼각측량"]
        r2s = [res[l][0] for l in [c[0] for c in combos]] + [r2_3r, r2_tri]
        cols = (["#4c72b0"] * nbs + ["#dd8452"] * (len(combos) - nbs - 1)
                + ["#55a868", "#c44e52", "#8172b3"])
        ax[0].bar(range(len(labs)), r2s, color=cols)
        ax[0].set_xticks(range(len(labs))); ax[0].set_xticklabels(labs, rotation=25, fontsize=8)
        ax[0].set_ylabel("R² (d_phys 복원)")
        ax[0].set_title("RSRP pairwise(녹) vs 이상거리 pairwise(적) vs 삼각측량(보라)")
        ax[0].grid(True, ls=":", alpha=0.4)
        for i, v in enumerate(r2s):
            ax[0].text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

        # 3-BS 1:1 산점도
        s = np.sqrt(np.mean(F ** 2, axis=0))
        A = np.column_stack([F / s, np.ones(len(y))])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        pred = A @ coef
        sidx = rng.choice(len(y), size=min(80000, len(y)), replace=False)
        ymax = np.percentile(y, 99)
        ax[1].scatter(y[sidx], pred[sidx], s=2, alpha=0.04, color="navy",
                      edgecolors="none", rasterized=True)
        ax[1].plot([0, ymax], [0, ymax], "r--", lw=1.5, label="1:1")
        ax[1].set_xlim(0, ymax); ax[1].set_ylim(0, ymax)
        ax[1].set_xlabel("d_phys [m]"); ax[1].set_ylabel("d_metric [m]")
        ax[1].set_title(f"3-BS 적합: R²={full:.3f}, corr={res['BS0+1+2 (3-BS)'][1]:.3f}")
        ax[1].grid(True, ls=":", alpha=0.4); ax[1].legend(loc="upper left")
        fig.suptitle("[P2G multi-BS] BS 1개 vs 3개 — UE-UE 거리 복원",
                     fontsize=13, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        png = os.path.join(OUT_DIR, f"P2G_multibs_{stamp}.png")
        fig.savefig(png, dpi=150); plt.close(fig)
        log(f"\n[viz] saved -> {png}")
        log("[done] multi-BS 검증 완료")
    finally:
        log_f.close()


if __name__ == "__main__":
    main()
