# -*- coding: utf-8 -*-
"""
P2G_AddUECov_2606v1.py
======================
[ver0.2 점검] UE-side 공분산(R_UE, 16x16)을 4번째 거리 항으로 추가하면
메트릭-물리거리 상관(R²)이 얼마나 오르는지 직접 측정.

배경
  - ver0.1(d_RSRP/d_PDP/d_Cov[BS-side]) full R²≈0.144 (corr 0.38). 천장이 낮음.
  - R_UE(16x16)는 UE 안테나 수신 공간 공분산 = 도래각 분포 정보.
    BS-side 와 독립적 신호일 수 있어 추가 시 corr 상승 기대.

d_Cov_UE 정의 (BS-side d_Cov 와 동일 절차)
  Ã = A/tr(A) + εI,  d_BW^2 = tr(Ã_i)+tr(Ã_j) - 2||B_i^H B_j||_*,
  d_Cov_UE = sqrt(max(d_BW^2, 0)).   16x16 풀랭크(r=16).

비교
  - 조합별 OLS(절편 허용) R²/RMSE/corr
  - full3 (RSRP+PDP+Cov_BS) vs full4 (+Cov_UE)
  - 각 항 한계 기여(빼면 R² 감소량)
----------------------------------------------------------------------
실행 환경 : Python 3.10.12 / torch 2.12.0+cu130 / numpy 2.2.6 / scipy 1.15.3
서버 dclserver78, NVIDIA H100 NVL 95GB x 2 (GPU 필요 -> full-permission 실행)
입력 : P1F_Marginal_CCM_Results/*.npz (R_UE), P2G_DistMat_*.npz(rx_idx,pos),
       P2G_PairTable_*.npz (기존 X,y)
----------------------------------------------------------------------
"""

import os
import re
import glob
import datetime

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

from P2G_gpu_utils_2606v1 import get_devices, batched_topr_eigh_factor, pairwise_nuclear

for _fp in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",):
    if os.path.exists(_fp):
        fm.fontManager.addfont(_fp)
        plt.rcParams["font.family"] = fm.FontProperties(fname=_fp).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WS_ROOT  = os.path.dirname(BASE_DIR)
P1F_DIR  = os.path.join(WS_ROOT, "251009_CCM_Collection (8 GB)",
                        "P1F_Marginal_CCM_Results")
DM_DIR   = os.path.join(BASE_DIR, "P2G_DistTerms_Results")
PT_DIR   = os.path.join(BASE_DIR, "P2G_PairTable_Results")
OUT_DIR  = os.path.join(BASE_DIR, "P2G_AddUECov_Results")
EPS = 1e-12
N_R = 16


def latest(d, pat):
    fs = sorted(glob.glob(os.path.join(d, pat)))
    if not fs:
        raise FileNotFoundError(pat)
    return fs[-1]


def r2_rmse_corr(y, pred):
    ss = np.sum((y - pred) ** 2)
    r2 = 1 - ss / np.sum((y - y.mean()) ** 2)
    return r2, np.sqrt(ss / len(y)), float(np.corrcoef(pred, y)[0, 1])


def fit_ols(Xc, y):
    """scale-only RMS 정규화 + 절편 허용 OLS (선형 전역 최적)."""
    s = np.sqrt(np.mean(Xc ** 2, axis=0))
    A = np.column_stack([Xc / s, np.ones(len(y))])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    return r2_rmse_corr(y, pred)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_f = open(os.path.join(OUT_DIR, f"P2G_adduecov_report_{stamp}.log"),
                 "w", encoding="utf-8")

    def log(*a):
        m = " ".join(str(x) for x in a); print(m); log_f.write(m + "\n"); log_f.flush()

    try:
        dm = np.load(latest(DM_DIR, "P2G_DistMat_*.npz"), allow_pickle=True)
        rx_idx = np.asarray(dm["rx_idx"], dtype=np.int64)      # (N,) DistMat 행순서
        N = len(rx_idx)
        pt = np.load(latest(PT_DIR, "P2G_PairTable_*.npz"), allow_pickle=True)
        X3 = pt["X"].astype(np.float64)                        # (M,3) 기존
        y = pt["y"].astype(np.float64)
        log(f"[load] N={N}, 기존 X{X3.shape}, y평균={y.mean():.2f}m")

        # rx -> P1F 파일 매핑 (빌더와 동일 규칙)
        files = glob.glob(os.path.join(P1F_DIR, "*.npz"))
        rx_file = {}
        for f in files:
            m = re.search(r"RX(\d+)_", os.path.basename(f))
            if m:
                rx_file[int(m.group(1))] = f
        miss = [int(r) for r in rx_idx if int(r) not in rx_file]
        if miss:
            raise RuntimeError(f"P1F 매핑 누락 {len(miss)}개: {miss[:5]}")

        # R_UE 형태 확인
        e0 = np.load(rx_file[int(rx_idx[0])], allow_pickle=True)
        rue0 = np.asarray(e0["R_UE"])
        log(f"[verify] R_UE shape={rue0.shape} (n_r={N_R} 일치={rue0.shape[0]==N_R}), "
            f"dtype={rue0.dtype}")

        # R_UE 저랭크(=풀랭크 r=16) 인수 + BW fidelity
        devices = get_devices()
        log(f"[env] devices={[str(x) for x in devices]}")

        def load_fn(i):
            e = np.load(rx_file[int(rx_idx[i])], allow_pickle=True)
            return np.asarray(e["R_UE"]).astype(np.complex64)

        B, tr, tail = batched_topr_eigh_factor(
            load_fn, N, N_R, N_R, EPS, devices, chunk=256, log=log)
        log(f"[ue] 인수 완료 tail mean={tail.mean():.2e} (풀랭크라 ~0)")
        F = pairwise_nuclear(B, devices, sub=128, rblk=64, log=log)
        dbw2 = tr[:, None] + tr[None, :] - 2.0 * F
        D_cov_ue = np.sqrt(np.clip(dbw2, 0.0, None))
        log(f"[ue] D_cov_ue NxN: 대칭 max|D-Dt|={np.abs(D_cov_ue-D_cov_ue.T).max():.2e}, "
            f"자기거리 max={np.abs(np.diag(D_cov_ue)).max():.2e}")

        iu = np.triu_indices(N, 1)
        x_cov_ue = D_cov_ue[iu].astype(np.float64)
        log(f"[ue] d_Cov_UE 분포 min/med/max="
            f"{x_cov_ue.min():.4f}/{np.median(x_cov_ue):.4f}/{x_cov_ue.max():.4f}, "
            f"std/mean={x_cov_ue.std()/x_cov_ue.mean():.3f}, "
            f"corr(d_phys)={np.corrcoef(x_cov_ue,y)[0,1]:.4f}")

        # 4항 결합
        X4 = np.column_stack([X3, x_cov_ue])     # [RSRP, PDP, Cov_BS, Cov_UE]
        names = ["d_RSRP", "d_PDP", "d_Cov_BS", "d_Cov_UE"]

        # ===== 조합별 비교 (절편 허용 OLS) =====
        log("\n[조합별 OLS(절편) R²/RMSE/corr]")
        combos = {
            "RSRP+PDP+Cov_BS (ver0.1)": [0, 1, 2],
            "Cov_UE only":             [3],
            "Cov_BS+Cov_UE":           [2, 3],
            "RSRP+PDP+Cov_UE":         [0, 1, 3],
            "FULL4 (+Cov_UE)":         [0, 1, 2, 3],
        }
        res = {}
        for lab, cols in combos.items():
            r2, rmse, r = fit_ols(X4[:, cols], y)
            res[lab] = (r2, rmse, r)
            log(f"  {lab:<26} R²={r2:.4f}  RMSE={rmse:6.2f}m  corr={r:.4f}")

        r2_03 = res["RSRP+PDP+Cov_BS (ver0.1)"][0]
        r2_04 = res["FULL4 (+Cov_UE)"][0]
        log(f"\n[핵심] ver0.1(3항) R²={r2_03:.4f} -> FULL4 R²={r2_04:.4f} "
            f"({r2_04-r2_03:+.4f}, corr {res['RSRP+PDP+Cov_BS (ver0.1)'][2]:.3f}"
            f"->{res['FULL4 (+Cov_UE)'][2]:.3f})")

        # 각 항 한계 기여 (FULL4 에서 제거)
        log("\n[FULL4 한계 기여: 각 항 제거 시 R² 감소]")
        for drop in range(4):
            keep = [i for i in range(4) if i != drop]
            r2d, _, _ = fit_ols(X4[:, keep], y)
            log(f"  {names[drop]} 제거 -> R²={r2d:.4f}  (FULL4 {r2_04:.4f} 대비 {r2d-r2_04:+.4f})")

        # ===== 그래프 =====
        fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
        labs = list(combos.keys())
        r2s = [res[l][0] for l in labs]
        ax[0].barh(range(len(labs)), r2s, color="steelblue")
        ax[0].set_yticks(range(len(labs)))
        ax[0].set_yticklabels(labs, fontsize=9)
        ax[0].set_xlabel("R² (기울기-1, 절편 허용)")
        ax[0].set_title("조합별 R²  (UE-side 추가 효과)")
        ax[0].grid(True, ls=":", alpha=0.4)
        for i, v in enumerate(r2s):
            ax[0].text(v, i, f" {v:.3f}", va="center", fontsize=9)

        # FULL4 1:1 산점도
        s = np.sqrt(np.mean(X4 ** 2, axis=0))
        A = np.column_stack([X4 / s, np.ones(len(y))])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        pred = A @ coef
        rng = np.random.default_rng(0)
        sidx = rng.choice(len(y), size=min(80000, len(y)), replace=False)
        ymax = y.max()
        ax[1].scatter(y[sidx], pred[sidx], s=2, alpha=0.04, color="navy",
                      edgecolors="none", rasterized=True)
        ax[1].plot([0, ymax], [0, ymax], "r--", lw=1.5, label="1:1")
        ax[1].set_xlim(0, ymax); ax[1].set_ylim(0, ymax)
        ax[1].set_xlabel("d_phys [m]"); ax[1].set_ylabel("d_metric [m]")
        ax[1].set_title(f"FULL4 적합: R²={res['FULL4 (+Cov_UE)'][0]:.3f}, "
                        f"corr={res['FULL4 (+Cov_UE)'][2]:.3f}")
        ax[1].grid(True, ls=":", alpha=0.4); ax[1].legend(loc="upper left")
        fig.suptitle("[P2G ver0.2] UE-side 공분산(R_UE) 추가 효과",
                     fontsize=13, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        png = os.path.join(OUT_DIR, f"P2G_adduecov_{stamp}.png")
        fig.savefig(png, dpi=150); plt.close(fig)
        log(f"\n[viz] saved -> {png}")

        np.savez_compressed(
            os.path.join(OUT_DIR, f"P2G_DcovUE_{stamp}.npz"),
            D_cov_ue=D_cov_ue.astype(np.float32), rx_idx=rx_idx,
            x_cov_ue=x_cov_ue.astype(np.float32))
        log("[done] UE-side 추가 효과 측정 완료")
    finally:
        log_f.close()


if __name__ == "__main__":
    main()
