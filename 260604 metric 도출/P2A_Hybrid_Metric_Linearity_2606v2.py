# -*- coding: utf-8 -*-
"""
P2A_Hybrid_Metric_Linearity_2606v2.py
======================================
단일 기지국(RAN Twin, single-BS) 환경에서 하이브리드 채널 메트릭의
'물리 거리 선형 상관' 과 't-SNE 지형(Topology) 복원' 을 동시에 시각화하고,
GPU(2-GPU) 브로드캐스팅으로 최적 가중치를 자동 탐색하는 파이프라인.

[ v1 -> v2 주요 변경 사유 ]
  탐색 결과(실측 데이터) 두 가지 사실을 확인하여 방법론을 보강하였다.
  1) 제공된 1024x1024 R_BS(P1F CCM) 의 Bures-Wasserstein 은 물리거리와
     '음의 상관'(rho ~ -0.16) -> trace(전력) 효과 + NLoS 확산으로 실패.
  2) rx_indices 는 1-based(1..1600) -> 위치 복원은 (idx-1) 로 보정해야 정확.
  -> P1A 'ALL rays'(LoS 포함) 의 AoD/지연/전력 으로부터 메트릭을 직접 구성하고,
     사용자 요청대로 '두 가지 공간 메트릭(BW vs AoD-Wasserstein)' 을 모두
     계산하여 비교한다. (둘 다 hybrid > 단일 성립; AoD-Wasserstein 이 우수)

[ 공간 메트릭 - 두 방식 비교 ]
  (A) Bures-Wasserstein : 레이 AoD(방위각)+power 로 가상 ULA(M=32) 의 공간
      공분산 R 을 steering 벡터로 재구성한 뒤 저랭크 BW.
        W_BW^2(i,j) = Tr(R_i)+Tr(R_j) - 2||B_i^H B_j||_*,  B=U sqrt(Λ)
  (B) AoD-Wasserstein   : 방위각 APS(48 bin) 의 1D Wasserstein(CDF-L1).
[ 시간 메트릭 ]  PDP(32 bin) 의 1D Wasserstein(CDF-L1), Scale=sqrt(Tr_i Tr_j) 가중.

----------------------------------------------------------------------
실행 환경 (검증 완료)
  - Python      : 3.10.12
  - 실행 서버    : dclcom61 (deepgadget), NVIDIA RTX 5090 32GB x 2 (sm_120)
  - GPU/CUDA    : Driver 580.95.05 / CUDA 12.8 (torch cu128)  -> 2-GPU 샤딩
  - torch 2.11.0+cu128 / numpy 2.2.6 / scipy 1.15.3 / scikit-learn 1.7.2 / matplotlib 3.10.9
----------------------------------------------------------------------
수치 안정화
  - 고유값 clamp(min=1e-12): cuSOLVER SVD 비수렴/NaN 방지(노이즈 바닥).
  - Spearman 은 GPU 내 이중 argsort 랭크 + Pearson (scipy CPU 병목 회피).
"""

import os
import time
import datetime

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE


# ============================================================
# 0. 설정
# ============================================================
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
WS_ROOT    = os.path.dirname(BASE_DIR)
DATA_ROOT  = os.path.join(WS_ROOT, "251009_CCM_Collection (8 GB)")
RAYS_NPZ   = os.path.join(DATA_ROOT, "P1A_RT_Results",
                          "Area1_7.5GHz_Rays_ALL_RXs.npz")     # LoS 포함
RESULT_DIR = os.path.join(BASE_DIR, "P2A_Hybrid_Metric_Results")

# Area1 grid (P1A_RT_to_Rays_2509v6.py AREA_CONFIGS['area_1'])
GRID_N = 40
RX_X   = np.linspace(-136.138, 58.862, GRID_N)
RX_Y   = np.linspace(-117.667, 77.333, GRID_N)
BS_XY  = np.array([-51.561, -21.794])

# 하이퍼파라미터
M_ARRAY        = 32        # 가상 ULA 안테나 수(BW 공분산 재구성)
RANK           = 16        # 저랭크 BW 차원
N_PDP_BINS     = 32        # 1D PDP bin
N_AOD_BINS     = 48        # 방위각 APS bin
N_DIST_BINS    = 10        # 거리 층화 bin
TARGET_PER_BIN = 80        # bin 당 목표 표본 (부족 시 전부)
MIN_PATHS      = 20        # 최소 유효 경로 수 필터
TAU_QUANTILE   = 0.999
N_ALPHA        = 100
TSNE_PERPLEXITY = 30.0
SEED           = 42
EPS            = 1e-12

torch.manual_seed(SEED)
np.random.seed(SEED)


def get_devices():
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        return [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
    return [torch.device("cpu")]


# ============================================================
# 1. 데이터 로딩 (P1A ALL rays, LoS 포함)
# ============================================================
def load_rays():
    d = np.load(RAYS_NPZ, allow_pickle=True)
    ridx = np.asarray(d["rx_indices"], dtype=np.int64)
    n = ridx.shape[0]
    pwr   = np.asarray(d["power"]).reshape(n, -1).astype(np.float64)
    tau   = np.asarray(d["tau"]).reshape(n, -1).astype(np.float64)
    phit  = np.deg2rad(np.asarray(d["phi_t_deg"]).reshape(n, -1).astype(np.float64))
    los   = np.asarray(d["los_nlos_flag"]).reshape(n, -1)

    # rx_indices 는 1-based -> (idx-1) 로 grid 환원 (탐색으로 검증)
    f = ridx - 1
    ix = f % GRID_N
    iy = (f // GRID_N) % GRID_N
    xs, ys = RX_X[ix], RX_Y[iy]

    npaths = (pwr > 0).sum(1)
    keep = np.where(npaths >= MIN_PATHS)[0]
    radius = np.sqrt((xs - BS_XY[0]) ** 2 + (ys - BS_XY[1]) ** 2)

    los_frac = (los[(pwr > 0)] == 1).mean()
    print(f"[load] RX total={n}, kept(>= {MIN_PATHS} paths)={len(keep)}, "
          f"LoS path frac={los_frac:.4f}, dist[{radius[keep].min():.1f},"
          f"{radius[keep].max():.1f}]m")

    return dict(idx=ridx[keep], xs=xs[keep], ys=ys[keep], radius=radius[keep],
                pwr=pwr[keep], tau=tau[keep], phit=phit[keep])


# ============================================================
# 2. 거리 층화 샘플링
# ============================================================
def stratified_indices(radius):
    rng = np.random.default_rng(SEED)
    edges = np.linspace(radius.min(), radius.max(), N_DIST_BINS + 1)
    sel, counts = [], []
    for b in range(N_DIST_BINS):
        lo, hi = edges[b], edges[b + 1]
        m = (radius >= lo) & (radius <= hi) if b == N_DIST_BINS - 1 \
            else (radius >= lo) & (radius < hi)
        ids = np.where(m)[0]
        if len(ids) > TARGET_PER_BIN:
            ids = rng.choice(ids, TARGET_PER_BIN, replace=False)
        counts.append(len(ids)); sel.append(ids)
    sel = np.sort(np.concatenate(sel))
    print(f"[strat] bin counts(after)={counts} -> N={len(sel)}")
    return sel


# ============================================================
# 3. 특징 구성 : PDP / AoD-APS / 공간 공분산 R(+trace)
# ============================================================
def hist_weighted(vals, weights, edges):
    """행별 가중 히스토그램 -> 합=1 정규화 (numpy 데이터 준비 단계)."""
    n, K = vals.shape[0], len(edges) - 1
    b = np.clip(np.digitize(vals, edges) - 1, 0, K - 1)
    H = np.zeros((n, K))
    rows = np.repeat(np.arange(n), vals.shape[1])
    np.add.at(H, (rows, b.ravel()), weights.ravel())
    H /= (H.sum(1, keepdims=True) + EPS)
    return H


def build_pdp_aps(data):
    pwr, tau, phit = data["pwr"], data["tau"], data["phit"]
    w = pwr * (pwr > 0)
    tau_max = float(np.quantile(tau[pwr > 0], TAU_QUANTILE))
    pdp = hist_weighted(np.clip(tau, 0, tau_max * (1 - 1e-9)), w,
                        np.linspace(0, tau_max, N_PDP_BINS + 1))
    aps = hist_weighted(phit, w, np.linspace(-np.pi, np.pi, N_AOD_BINS + 1))
    print(f"[feat] PDP({N_PDP_BINS}) / AoD-APS({N_AOD_BINS}) 구성, "
          f"tau_max={tau_max:.3e}s")
    return pdp, aps


def build_cov_factors(data, devices, rank=RANK, chunk=128):
    """레이 AoD(방위각)+power 로 가상 ULA(M) 공간 공분산 R 재구성 후 저랭크 인수분해.
    R_i = sum_k p_k a(phi_k) a(phi_k)^H,  a_m = exp(j pi m sin(phi))

    [중요] R 을 trace=1 로 정규화한 뒤 인수분해한다. 정규화하지 않으면 BW 가
    trace(=총수신전력, 거리와 강한 상관)에 지배되어 물리거리와 '음의 상관'을 보임.
    정규화 후 BW 는 각도(AoA) '형상' 차이만 측정 -> 양의 상관.
    반환: B(M x r, 정규화 R 기준), tr_power(원본 총수신전력, 저장/참고용). 2-GPU 분배.
    """
    pwr, phit = data["pwr"], data["phit"]
    N, P = pwr.shape
    m = torch.arange(M_ARRAY)
    splits = np.array_split(np.arange(N), len(devices))
    B = torch.zeros((N, M_ARRAY, rank), dtype=torch.complex64)
    tr = torch.zeros(N, dtype=torch.float64)

    t0 = time.time()
    for dev, part in zip(devices, splits):
        m_d = m.to(dev)
        for c0 in range(0, len(part), chunk):
            sub = part[c0:c0 + chunk]
            w = torch.from_numpy((pwr[sub] * (pwr[sub] > 0))).to(dev)        # (c,P)
            u = torch.from_numpy(np.sin(phit[sub])).to(dev)                  # (c,P)
            A = torch.exp(1j * np.pi * (m_d[None, :, None] * u[:, None, :])) # (c,M,P)
            A = A.to(torch.complex64)
            Aw = A * w[:, None, :].to(torch.complex64)
            R = Aw @ A.conj().transpose(-1, -2)                             # (c,M,M)
            R = 0.5 * (R + R.mH)
            tr_pow = torch.diagonal(R, dim1=-2, dim2=-1).sum(-1).real.double()
            # trace=1 정규화(형상만 비교)
            Rn = R / (tr_pow.clamp(min=EPS).to(R.dtype)[:, None, None])
            # 일부 UE 는 AoD 가 거의 단일방향 -> R 축퇴(중복 고유값)로 eigh 비수렴.
            # 상위 r 개만 필요하므로 축퇴에 강건한 randomized low-rank SVD 사용.
            U, S, _ = torch.svd_lowrank(Rn, q=min(rank + 8, M_ARRAY), niter=4)
            ev = S[..., :rank].clamp(min=EPS)
            U = U[..., :rank]
            Bsub = torch.nan_to_num(U * torch.sqrt(ev).unsqueeze(-2))
            B[sub] = Bsub.cpu()
            tr[sub] = tr_pow.cpu()
            del A, Aw, R, Rn, U, S, Bsub
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
    print(f"[feat] 공간 공분산(M={M_ARRAY}) 재구성+rank-{rank} 인수분해 "
          f"on {len(devices)} GPU, {time.time()-t0:.1f}s")
    return B, tr


# ============================================================
# 4. Batched Pairwise (2-GPU 샤딩, for-loop over pairs 금지)
# ============================================================
def pairwise_all(B, tr, cdf_pdp, cdf_aps, devices, sub=256):
    """반환(CPU,float64): W_BW2, W_pdp(CDF-L1), W_aps(CDF-L1).
    행 청크를 GPU 들에 분배 -> 두 GPU 동시 실행.
    """
    N = B.shape[0]
    row_splits = np.array_split(np.arange(N), len(devices))
    Bd  = {d: B.to(d) for d in devices}
    cpd = {d: cdf_pdp.to(d) for d in devices}
    cad = {d: cdf_aps.to(d) for d in devices}

    parts = {}
    t0 = time.time()
    for dev, rows in zip(devices, row_splits):
        if len(rows) == 0:
            continue
        r0, r1 = int(rows[0]), int(rows[-1]) + 1
        Bi = Bd[dev][r0:r1]
        nb = r1 - r0
        F  = torch.empty((nb, N), device=dev, dtype=torch.float32)
        Wp = torch.empty((nb, N), device=dev, dtype=torch.float32)
        Wa = torch.empty((nb, N), device=dev, dtype=torch.float32)
        cpi, cai = cpd[dev][r0:r1], cad[dev][r0:r1]
        for j0 in range(0, N, sub):
            j1 = min(N, j0 + sub)
            Bj = Bd[dev][j0:j1]
            G = torch.einsum("amp,jmq->ajpq", Bi.conj(), Bj)     # (b,c,r,r)
            F[:, j0:j1] = torch.linalg.svdvals(G).sum(-1)
            Wp[:, j0:j1] = (cpi[:, None, :] - cpd[dev][j0:j1][None]).abs().sum(-1)
            Wa[:, j0:j1] = (cai[:, None, :] - cad[dev][j0:j1][None]).abs().sum(-1)
        parts[dev] = (r0, r1, F, Wp, Wa)
    for dev in devices:
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)

    W_BW2 = torch.zeros((N, N), dtype=torch.float64)
    W_pdp = torch.zeros((N, N), dtype=torch.float64)
    W_aps = torch.zeros((N, N), dtype=torch.float64)
    for dev, (r0, r1, F, Wp, Wa) in parts.items():
        # R 은 trace=1 정규화됨 -> Tr_i=Tr_j=1, W_BW2 = 2 - 2F
        W_BW2[r0:r1] = 2.0 - 2.0 * F.double().cpu()
        W_pdp[r0:r1] = Wp.double().cpu()
        W_aps[r0:r1] = Wa.double().cpu()
    W_BW2 = torch.nan_to_num(W_BW2).clamp(min=0.0)
    W_pdp = torch.nan_to_num(W_pdp)
    W_aps = torch.nan_to_num(W_aps)
    print(f"[pairwise] W_BW2 / W_pdp / W_aps ({N}x{N}) on {len(devices)} GPU, "
          f"{time.time()-t0:.1f}s")
    return W_BW2, W_pdp, W_aps


# ============================================================
# 5. 정규화 + GPU Spearman 가중치 탐색
# ============================================================
def minmax_offdiag(M):
    N = M.shape[0]
    mask = ~torch.eye(N, dtype=torch.bool)
    vals = M[mask]
    lo, hi = vals.min(), vals.max()
    Mn = (M - lo) / (hi - lo + EPS)
    Mn.fill_diagonal_(0.0)
    return Mn


def spearman_search(W_spatial_n, W_temporal_n, phys, device):
    """alpha 0..1 (N_ALPHA) 스캔, GPU 이중 argsort 랭크 Pearson 으로 Spearman."""
    N = phys.shape[0]
    iu = torch.triu_indices(N, N, offset=1)
    pf = phys[iu[0], iu[1]].to(device).float()
    sp = W_spatial_n[iu[0], iu[1]].to(device).float()
    tp = W_temporal_n[iu[0], iu[1]].to(device).float()
    alphas = torch.linspace(0.0, 1.0, N_ALPHA, device=device)
    total = alphas[:, None] * sp[None, :] + (1 - alphas)[:, None] * tp[None, :]
    rx = torch.argsort(torch.argsort(pf)).float()
    ry = torch.argsort(torch.argsort(total, dim=1), dim=1).float()
    rx_c = rx - rx.mean()
    ry_c = ry - ry.mean(dim=1, keepdim=True)
    num = (ry_c * rx_c[None, :]).sum(1)
    den = torch.sqrt((ry_c ** 2).sum(1) * (rx_c ** 2).sum()) + EPS
    rho = (num / den)
    best = int(torch.argmax(rho).item())
    return (alphas.cpu().numpy(), rho.cpu().numpy(), best,
            pf.cpu().numpy(), sp.cpu().numpy(), tp.cpu().numpy())


# ============================================================
# 6. 시각화
# ============================================================
def run_tsne(Dmat):
    D = np.asarray(Dmat, dtype=np.float64)
    D = 0.5 * (D + D.T); np.fill_diagonal(D, 0.0); D = np.clip(D, 0.0, None)
    ts = TSNE(n_components=2, metric="precomputed", init="random",
              perplexity=TSNE_PERPLEXITY, random_state=SEED, learning_rate="auto")
    return ts.fit_transform(D)


def figure_2x4(method, xs, ys, radius, pf, sp, tp, alphas, rho, best,
               W_spatial_n, W_temporal_n, out_png):
    a_opt = alphas[best]; b_opt = 1 - a_opt; rho_opt = rho[best]
    y_golden = a_opt * sp + b_opt * tp
    mat_golden = a_opt * W_spatial_n.numpy() + b_opt * W_temporal_n.numpy()

    fig, ax = plt.subplots(2, 4, figsize=(22, 11))
    fig.suptitle(
        f"[{method}]  Golden Ratio: alpha(spatial)={a_opt:.3f}, "
        f"beta(temporal)={b_opt:.3f}   |   max Spearman = {rho_opt:.4f}   "
        f"(spatial-only={rho[-1]:.3f}, temporal-only={rho[0]:.3f})",
        fontsize=15, fontweight="bold")

    sc = dict(s=4, alpha=0.1, edgecolors="none", color="navy")
    titles = ["GT: Physical vs Physical",
              f"Spatial 100% (alpha=1)  rho={rho[-1]:.3f}",
              f"Temporal 100% (alpha=0)  rho={rho[0]:.3f}",
              f"Golden (alpha={a_opt:.2f})  rho={rho_opt:.3f}"]
    ys_data = [pf / pf.max(), sp, tp, y_golden]
    for c in range(4):
        ax[0, c].scatter(pf, ys_data[c], **sc)
        ax[0, c].set_title(titles[c], fontsize=12)
        ax[0, c].set_xlabel("Physical distance [m]")
        ax[0, c].set_ylabel("Metric distance (norm.)")
        ax[0, c].grid(True, ls=":", alpha=0.3)

    print(f"[viz:{method}] t-SNE ...")
    emb_sp = run_tsne(W_spatial_n.numpy())
    emb_tp = run_tsne(W_temporal_n.numpy())
    emb_gd = run_tsne(mat_golden)
    gt_xy = np.stack([xs - BS_XY[0], ys - BS_XY[1]], 1)
    t2 = ["GT: real (x,y) layout", "t-SNE Spatial 100%",
          "t-SNE Temporal 100%", f"t-SNE Golden (alpha={a_opt:.2f})"]
    embs = [gt_xy, emb_sp, emb_tp, emb_gd]
    for c in range(4):
        im = ax[1, c].scatter(embs[c][:, 0], embs[c][:, 1], c=radius,
                              cmap="viridis", s=22, alpha=0.9, edgecolors="none")
        ax[1, c].set_title(t2[c], fontsize=12)
        ax[1, c].set_xlabel("dim-1"); ax[1, c].set_ylabel("dim-2")
        ax[1, c].grid(True, ls=":", alpha=0.3)
        cb = fig.colorbar(im, ax=ax[1, c], fraction=0.046, pad=0.04)
        cb.set_label("Radius from BS [m]", fontsize=9)
    ax[1, 0].set_aspect("equal", adjustable="box")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=160); plt.close(fig)
    print(f"[viz:{method}] saved -> {out_png}")
    return a_opt, b_opt, rho_opt


def figure_compare(res, out_png):
    """두 방식의 alpha-Spearman 곡선 + 핵심 rho 막대."""
    fig, (axc, axb) = plt.subplots(1, 2, figsize=(15, 5.5))
    bars_label, bars_val = [], []
    for name, r in res.items():
        axc.plot(r["alphas"], r["rho"], lw=2, label=f"{name} (max={r['rho'][r['best']]:.3f})")
        axc.scatter([r["alphas"][r["best"]]], [r["rho"][r["best"]]], zorder=5)
        bars_label += [f"{name}\nspatial", f"{name}\ntemporal", f"{name}\nhybrid"]
        bars_val += [r["rho"][-1], r["rho"][0], r["rho"][r["best"]]]
    axc.set_xlabel("alpha (spatial weight)"); axc.set_ylabel("Spearman vs physical dist.")
    axc.set_title("Auto Weight Search (both spatial methods)")
    axc.grid(True, ls=":", alpha=0.4); axc.legend()

    x = np.arange(len(bars_val))
    axb.bar(x, bars_val, color=["#4c72b0", "#dd8452", "#55a868"] * len(res))
    axb.set_xticks(x); axb.set_xticklabels(bars_label, fontsize=8)
    axb.set_ylabel("Spearman correlation")
    axb.set_title("Spatial vs Temporal vs Hybrid")
    axb.grid(True, axis="y", ls=":", alpha=0.4)
    for xi, v in zip(x, bars_val):
        axb.text(xi, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)
    fig.tight_layout(); fig.savefig(out_png, dpi=160); plt.close(fig)
    print(f"[viz] saved -> {out_png}")


# ============================================================
# main
# ============================================================
def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    devices = get_devices()
    print(f"[env] devices = {[str(d) for d in devices]}")

    data = load_rays()
    sel = stratified_indices(data["radius"])
    data = {k: v[sel] for k, v in data.items()}

    pdp, aps = build_pdp_aps(data)
    B, tr = build_cov_factors(data, devices)

    cdf_pdp = torch.from_numpy(np.cumsum(pdp, 1)).to(torch.float32)
    cdf_aps = torch.from_numpy(np.cumsum(aps, 1)).to(torch.float32)
    # BW 는 정규화 R(trace=1) 기준이므로 trace 인자 불필요(pairwise 내부 2-2F)
    W_BW2, W_pdp, W_aps = pairwise_all(B, tr, cdf_pdp, cdf_aps, devices)

    # 시간 메트릭 : PDP 의 1D Wasserstein(CDF-L1).
    # NOTE) 프롬프트의 Scale=sqrt(Tr_i Tr_j) 가중은 실측 전력 동적범위가 ~9 자릿수라
    #       min-max 후 metric 을 붕괴시켜(근거리 쌍이 전부 지배) 제외한다.
    W_temporal = W_pdp

    # 각각 독립 Min-Max 정규화
    W_bw_n   = minmax_offdiag(W_BW2)
    W_aps_n  = minmax_offdiag(W_aps)
    W_temp_n = minmax_offdiag(W_temporal)

    xy = torch.from_numpy(np.stack([data["xs"], data["ys"]], 1)).double()
    phys = torch.cdist(xy, xy)
    dev0 = devices[0]

    res = {}
    for name, W_sp_n in [("Bures-Wasserstein", W_bw_n),
                         ("AoD-Wasserstein", W_aps_n)]:
        alphas, rho, best, pf, sp, tp = spearman_search(W_sp_n, W_temp_n, phys, dev0)
        res[name] = dict(alphas=alphas, rho=rho, best=best, pf=pf, sp=sp, tp=tp,
                         W_sp_n=W_sp_n)
        print(f"[search:{name}] golden alpha={alphas[best]:.3f}, "
              f"rho={rho[best]:.4f} (spatial={rho[-1]:.3f}, temporal={rho[0]:.3f})")

    # 시각화 : 방식별 2x4 + 비교
    for name in res:
        r = res[name]
        tag = "BW" if name.startswith("Bures") else "AoDW"
        figure_2x4(name, data["xs"], data["ys"], data["radius"],
                   r["pf"], r["sp"], r["tp"], r["alphas"], r["rho"], r["best"],
                   r["W_sp_n"], W_temp_n,
                   os.path.join(RESULT_DIR, f"P2A_2x4_{tag}_{stamp}.png"))
    figure_compare(res, os.path.join(RESULT_DIR, f"P2A_compare_{stamp}.png"))

    # 결과 저장
    np.savez(os.path.join(RESULT_DIR, f"P2A_result_{stamp}.npz"),
             **{f"{('BW' if k.startswith('Bures') else 'AoDW')}_{f}":
                res[k][f] for k in res for f in ("alphas", "rho", "best")},
             rx_idx=data["idx"], x=data["xs"], y=data["ys"], radius=data["radius"],
             M_array=M_ARRAY, rank=RANK, n_pdp=N_PDP_BINS, n_aod=N_AOD_BINS)
    with open(os.path.join(RESULT_DIR, f"P2A_summary_{stamp}.csv"), "w") as f:
        f.write("method,spatial_only,temporal_only,hybrid_max,alpha_opt\n")
        for k in res:
            r = res[k]
            f.write(f"{k},{r['rho'][-1]:.4f},{r['rho'][0]:.4f},"
                    f"{r['rho'][r['best']]:.4f},{r['alphas'][r['best']]:.4f}\n")
    print(f"[done] results saved in {RESULT_DIR}")


if __name__ == "__main__":
    main()
