# -*- coding: utf-8 -*-
"""
P2G_DistTerms_2606v1.py
=======================
[단계 2] 거리 항 3함수 + 단위테스트, 그리고 점 테이블 -> 3개 거리행렬(NxN) 산출.

거리 항 정의 (단위/가정)
  1) d_RSRP(i,j) [dB, L1] : |rsrp_i - rsrp_j|.  rsrp=10log10(Σpower) [dB].
  2) d_PDP(i,j)  [ns, 1-Wasserstein] : 지연분포의 1D W1.
       - 생산용: 공통 미세 격자(L bin) CDF-L1.  W1 = Σ_k|CDF_i-CDF_j|·Δτ.
         (point-mass 직접 W1 은 537k 쌍에 비현실적 -> 미세 격자로 근사,
          binning 오차는 scipy 점질량 W1 과 단위테스트로 bound)
       - 정규화: p = power/Σpower (유효경로 power>0 만, 총전력 제거 -> RSRP 항 담당)
       - 공통 τ축: 모든 점이 [0, global tau_max] 동일 격자 (점별 tau_max 금지)
       - 동적범위 큼(span med 423ns vs max 4226ns): clip 하면 잘린 질량이 W1 을
         크게 왜곡(scipy 대비 rel>100%) -> clip 안함(TAU_QUANTILE=1.0) + 균일 미세격자
       - 단위 ns 로 유지(s 면 ~1e-7 로 너무 작아 정밀도 손실 우려). weight 는 m/ns.
  3) d_Cov(i,j)  [무차원, Bures-Wasserstein] : 저랭크 인수 기반.
       d_BW^2 = tr(Ã_i)+tr(Ã_j) - 2||B_i^H B_j||_*,  d_Cov=sqrt(max(.,0)).
       (||B_i^H B_j||_* = full Bures fidelity 임은 P2G_BW_Approx_Check 검증,
        r=96 충분성은 P2G_dCov_RankCheck 검증)

GPU 정책 : 모든 거리행렬을 모든 GPU(H100x2)에 샤딩. (P2G_gpu_utils 사용)
  - 샌드박스는 GPU 접근을 막으므로 full-permission 실행 필요.

출력
  P2G_DistTerms_Results/P2G_DistMat_{stamp}.npz  (D_rsrp, D_pdp, D_cov, 메타)
  P2G_DistTerms_Results/P2G_distterms_report_{stamp}.log  (단위테스트 결과)

----------------------------------------------------------------------
실행 환경 : Python 3.10.12 / torch 2.12.0+cu130 / numpy 2.2.6 / scipy 1.15.3
서버 dclserver78, NVIDIA H100 NVL 95GB x 2
입력 : P2G_PointTable_Results/P2G_PointTable_*.npz
----------------------------------------------------------------------
"""

import os
import glob
import time
import datetime
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from scipy.stats import wasserstein_distance

from P2G_gpu_utils_2606v1 import get_devices, warmup_linalg, pairwise_nuclear

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PT_DIR   = os.path.join(BASE_DIR, "P2G_PointTable_Results")
OUT_DIR  = os.path.join(BASE_DIR, "P2G_DistTerms_Results")

N_PDP_GRID   = 1024      # PDP 공통 격자 bin 수 (미세화: Δτ≈4ns)
TAU_QUANTILE = 1.0       # tau_max clip 분위수
                         # 1.0=clip 안함. clip 시 잘린 질량이 W1 을 크게 왜곡하여
                         # (점질량 scipy 대비 rel>100%) 비활성화. 전체 tau_max 사용.
EPS          = 1e-12
SEED         = 42


def latest_point_table():
    fs = sorted(glob.glob(os.path.join(PT_DIR, "P2G_PointTable_*.npz")))
    if not fs:
        raise FileNotFoundError("점 테이블 npz 없음. 먼저 빌더 실행.")
    return fs[-1]


# ============================================================
# 1) d_RSRP : |Δ| (GPU broadcast)
# ============================================================
def d_rsrp_matrix(rsrp, devices):
    x = torch.from_numpy(np.asarray(rsrp, dtype=np.float64))
    d0 = devices[0]
    xd = x.to(d0)
    D = (xd[:, None] - xd[None, :]).abs().cpu().numpy()
    return D


# ============================================================
# 2) d_PDP : 미세 격자 CDF-L1 (GPU, row-shard + j-chunk)
# ============================================================
def build_pdp_cdf(tau, pwr, L, tau_max):
    """행별 weighted 히스토그램(합=1) -> CDF (N,L). 공통 [0,tau_max] 균일격자.
    tau, tau_max 단위는 ns 로 들어옴 -> dtau[ns], W1 도 ns."""
    n = tau.shape[0]
    edges = np.linspace(0.0, tau_max, L + 1)
    dtau = tau_max / L
    tc = np.clip(tau, 0, tau_max * (1 - 1e-9))
    b = np.clip(np.digitize(tc, edges) - 1, 0, L - 1)
    w = np.where(pwr > 0, pwr, 0.0).astype(np.float64)   # 유효경로만
    H = np.zeros((n, L))
    rows = np.repeat(np.arange(n), tau.shape[1])
    np.add.at(H, (rows, b.ravel()), w.ravel())
    # 정규화: power 가 선형 절대값(~1e-10~1e-20)이라 분모에 EPS 더하면 오염됨.
    # 모든 점은 유효경로>0 -> 합>0. 0행만 1로 보호.
    s = H.sum(1, keepdims=True)
    H = H / np.where(s > 0, s, 1.0)
    cdf = np.cumsum(H, axis=1)
    return cdf.astype(np.float32), float(dtau)


def d_pdp_matrix(cdf, dtau, devices, sub=256):
    N = cdf.shape[0]
    warmup = None  # not needed (no lazy linalg)
    cdf_t = torch.from_numpy(cdf)
    cd = {d: cdf_t.to(d) for d in devices}
    D = torch.zeros((N, N), dtype=torch.float64)
    row_splits = np.array_split(np.arange(N), len(devices))
    t0 = time.time()

    def worker(dev, rows):
        if len(rows) == 0:
            return
        r0, r1 = int(rows[0]), int(rows[-1]) + 1
        ci = cd[dev][r0:r1]
        out = torch.empty((r1 - r0, N), device=dev, dtype=torch.float32)
        for j0 in range(0, N, sub):
            j1 = min(N, j0 + sub)
            diff = (ci[:, None, :] - cd[dev][j0:j1][None, :, :]).abs().sum(-1)
            out[:, j0:j1] = diff * dtau
        D[r0:r1] = out.double().cpu()
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)

    with ThreadPoolExecutor(max_workers=len(devices)) as ex:
        list(ex.map(lambda da: worker(*da), zip(devices, row_splits)))
    print(f"[d_PDP] ({N}x{N}) on {len(devices)}GPU, {time.time()-t0:.1f}s")
    return D.numpy()


# ============================================================
# 3) d_Cov : 저랭크 BW (gpu_utils.pairwise_nuclear)
# ============================================================
def d_cov_matrix(B, tr, devices, sub=64):
    F = pairwise_nuclear(B, devices, sub=sub)
    dbw2 = tr[:, None] + tr[None, :] - 2.0 * F
    D = np.sqrt(np.clip(dbw2, 0.0, None))
    return D


# ============================================================
# 단위테스트
# ============================================================
def unittest_rsrp(rsrp, devices, log=print):
    """d_RSRP: 단위 dB, 대칭성, 자기거리=0, 비음수."""
    n = min(200, rsrp.shape[0])
    D = d_rsrp_matrix(rsrp[:n], devices)
    log(f"[unittest d_RSRP] (부분 {n}x{n}) 대칭 max|D-Dt|={np.abs(D-D.T).max():.2e}, "
        f"자기거리 max={np.abs(np.diag(D)).max():.2e}, min={D.min():.2e}, "
        f"NaN/inf={(~np.isfinite(D)).sum()} (단위 dB)")


def unittest_pdp(tau, pwr, cdf, dtau, n_pairs=400, log=print):
    """격자 CDF-L1 W1 vs scipy 점질량 W1(정확) 비교. W1 크기 분위수별 상대오차.
    tau 단위 ns -> W1, 오차 모두 ns."""
    rng = np.random.default_rng(SEED)
    N = tau.shape[0]
    cdf64 = cdf.astype(np.float64)
    we, wg = [], []
    for _ in range(n_pairs):
        i, j = rng.integers(0, N, 2)
        if i == j:
            continue
        vi, vj = pwr[i] > 0, pwr[j] > 0
        wg.append(np.abs(cdf64[i] - cdf64[j]).sum() * dtau)
        we.append(wasserstein_distance(tau[i][vi], tau[j][vj],
                                       pwr[i][vi], pwr[j][vj]))
    we, wg = np.array(we), np.array(wg)
    ok = we > 1e-9
    we, wg = we[ok], wg[ok]
    rel = np.abs(wg - we) / we
    ab = np.abs(wg - we)
    log(f"[unittest d_PDP] grid(L={cdf.shape[1]}, Δτ={dtau:.3f}ns) "
        f"vs scipy 점질량, {len(we)}쌍 (단위 ns)")
    log(f"   W1_exact[ns] min/med/max = {we.min():.2f}/{np.median(we):.2f}/{we.max():.2f}")
    log(f"   전체 rel med/p95/max = {np.median(rel):.3e}/"
        f"{np.percentile(rel,95):.3e}/{rel.max():.3e}  abs[ns] max={ab.max():.3f}")
    # W1 크기 분위수별(가까운 쌍=작은 W1 에서 격자 오차 최대일 수 있음)
    q = np.quantile(we, [0.0, 0.1, 0.25, 0.5, 1.0])
    labels = ["하위10%(가까움)", "10-25%", "25-50%", "50-100%"]
    for k in range(4):
        m = (we >= q[k]) & (we <= q[k + 1] if k == 3 else we < q[k + 1])
        if m.any():
            log(f"   [{labels[k]:14s}] W1<={q[k+1]:.1f}ns: "
                f"rel med={np.median(rel[m]):.3e}, max={rel[m].max():.3e}")
    return rel


def unittest_dcov(B, tr, devices, log=print):
    """d_Cov: 대칭성/비음수/자기거리(저랭크라 ε,tail 로 0 아님)/클램프."""
    n = min(64, B.shape[0])
    D = d_cov_matrix(B[:n], tr[:n], devices, sub=64)
    sym = np.abs(D - D.T).max()
    diag = np.abs(np.diag(D)).max()
    log(f"[unittest d_Cov] (부분 {n}x{n}) r={B.shape[2]} 대칭 max|D-Dt|={sym:.3e}, "
        f"자기거리 max={diag:.3e}(=√(2·tail), 저랭크 floor), "
        f"min={D.min():.3e}(클램프>=0 {D.min()>=-1e-12}), NaN/inf={(~np.isfinite(D)).sum()}")


# ============================================================
# main
# ============================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    devices = get_devices()
    warmup_linalg(devices)
    log_f = open(os.path.join(OUT_DIR, f"P2G_distterms_report_{stamp}.log"),
                 "w", encoding="utf-8")

    def log(*a):
        m = " ".join(str(x) for x in a); print(m); log_f.write(m + "\n"); log_f.flush()

    try:
        ptf = latest_point_table()
        d = np.load(ptf, allow_pickle=True)
        rsrp = d["rsrp"].astype(np.float64)
        tau, pwr = d["tau_raw"], d["pwr_raw"]
        B, tr = d["cov_factor"], d["cov_trace"].astype(np.float64)
        N = rsrp.shape[0]
        log(f"[env] devices={[str(x) for x in devices]}")
        log(f"[load] {os.path.basename(ptf)} N={N}, B{B.shape}, r={B.shape[2]}")

        # PDP 격자 준비 (ns 단위, 공통 [0, global tau_max] 균일격자)
        tau_ns = tau.astype(np.float64) * 1e9
        v = pwr > 0
        tau_max = float(np.quantile(tau_ns[v], TAU_QUANTILE))   # q1.0 = 전체 max
        spans = np.array([np.ptp(tau_ns[i][v[i]]) for i in range(N)])
        cdf, dtau = build_pdp_cdf(tau_ns, pwr, N_PDP_GRID, tau_max)
        log(f"[d_PDP] grid L={N_PDP_GRID}, tau_max(q{TAU_QUANTILE})={tau_max:.1f}ns, "
            f"Δτ={dtau:.3f}ns | 분해능 근거: 점별 지연확산 med={np.median(spans):.1f}ns "
            f"-> 확산/Δτ≈{np.median(spans)/dtau:.0f}bin (가까운쌍 분해 충분)")

        # 단위테스트 (세 함수 공통: NaN/inf, 대칭, 비음수)
        unittest_rsrp(rsrp, devices, log=log)
        unittest_pdp(tau_ns, pwr, cdf, dtau, log=log)
        unittest_dcov(B, tr, devices, log=log)

        # 3개 거리행렬
        t0 = time.time()
        D_rsrp = d_rsrp_matrix(rsrp, devices)
        D_pdp = d_pdp_matrix(cdf, dtau, devices)
        D_cov = d_cov_matrix(B, tr, devices)
        log(f"[mat] 3개 거리행렬 완료, 총 {time.time()-t0:.1f}s")
        for nm, D in [("d_RSRP[dB]", D_rsrp), ("d_PDP[ns]", D_pdp),
                      ("d_Cov[-]", D_cov)]:
            iu = np.triu_indices(N, 1)
            v = D[iu]
            log(f"   {nm:12s} off-diag min/med/max = {v.min():.4e}/"
                f"{np.median(v):.4e}/{v.max():.4e}")

        out = os.path.join(OUT_DIR, f"P2G_DistMat_{stamp}.npz")
        np.savez_compressed(out, D_rsrp=D_rsrp, D_pdp=D_pdp, D_cov=D_cov,
                            rx_idx=d["rx_idx"], pos=d["pos"],
                            unit_rsrp="dB", unit_pdp="ns", unit_cov="dimensionless",
                            n_pdp_grid=N_PDP_GRID, tau_max_ns=tau_max, dtau_ns=dtau,
                            rank=B.shape[2], point_table=os.path.basename(ptf))
        log(f"[save] {out} ({os.path.getsize(out)/1e6:.1f} MB)")
    finally:
        log_f.close()


if __name__ == "__main__":
    main()
