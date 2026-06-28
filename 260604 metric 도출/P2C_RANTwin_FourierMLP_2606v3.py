# -*- coding: utf-8 -*-
"""
P2C_RANTwin_FourierMLP_2606v3.py
================================
[ 목적 / v2 -> v3 변경 사유 ]
v1~v2 에서 일반 MLP 가 위치보간(IDW)에 패배 -> 원인은 **저주파 편향(Spectral Bias)**:
좌표->채널의 급격한(고주파) NLoS 변화를 평범한 MLP 가 못 잡음. v3 는 NeRF 식
**고주파 위치 인코딩(Fourier Features)** 을 도입해 위치-only 입력만으로 보간을
능가하는지 검증한다.

[ 과제 정의 (사용자 스펙) ]
  - 입력 : coords (N,2) 정규화 좌표
  - 출력 : channel_profiles (N,240) = 3 TX * (PDP 32 + AoA 48)
  - 데이터 희소 : 10~20 % 관측(train), 나머지 held-out(test).

[ 비교 ]
  (0) 단순보간 IDW(위치)        회색 점선  (baseline, 학습X)
  (1) Plain MLP(위치)           파랑       (ablation: PE 없음 -> 저주파편향)
  (2) Fourier MLP(위치, NeRF)   녹색       (제안 아키텍처)
  floor = 전역평균(공간정보無), skill=(S-floor)/(1-floor).

[ 평가 채널유사도 (v1~v2 동일) ]
  - BS별 PDP/APS -> 가상 ULA(M=16) Cov 재구성(trace=1), Bures-Wasserstein.
  - S = 0.5*(1-W1_PDP) + 0.5*(1-BW_Cov/2), 3 BS·held-out 평균.

----------------------------------------------------------------------
실행 환경
  - Python 3.10.12 / dclcom61 / numpy·torch(2.11+cu128)·matplotlib
  - 데이터 : 260512 RT Result Data/channel_data_260531_GHM_Twin_v0_1.npz (TX3, SISO)
----------------------------------------------------------------------
"""

import os
import csv
import datetime

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

import torch
import torch.nn as nn
import torch.nn.functional as F

for _fp in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]:
    if os.path.exists(_fp):
        font_manager.fontManager.addfont(_fp)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=_fp).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

NPZ = "/home/dclcom61/twin_minji/260512 RT Result Data/channel_data_260531_GHM_Twin_v0_1.npz"
OUTDIR = "/home/dclcom61/twin_minji/260604 metric 도출/P2C_RANTwin_Interpolator_Results"
N_TX = 3
N_SAMPLE = 900
MIN_PATHS = 10
N_TAU = 32
N_AOA = 48
M_ULA = 16
RATIOS = [0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
SEEDS = [0, 1, 2, 3]
SCORE_CAP = 400
SEED0 = 0

# --- Fourier MLP 하이퍼파라미터 ---
PE_L = 10               # 위치인코딩 주파수 개수 L
HIDDEN = 256
N_LAYERS = 4            # 얕은 구조(과적합 방지)
EPOCHS = 800
LR = 2e-3
WD = 1e-4               # weight decay (희소 데이터 과적합 방지)
W_MSE, W_WASS, W_L1 = 1.0, 0.5, 0.2   # custom loss 가중

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(OUTDIR, exist_ok=True)
STAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


# =====================================================================
#  요청 아키텍처 : Fourier Feature (NeRF-style) MLP
# =====================================================================
class PositionalEncoding(nn.Module):
    """입력 (x,y) -> sin/cos 고주파 대역으로 확장. (NeRF γ(p))
    γ(p) = [p, sin(2^0 π p), cos(2^0 π p), ..., sin(2^{L-1} π p), cos(2^{L-1} π p)]
    """

    def __init__(self, num_frequencies: int = PE_L, include_input: bool = True):
        super().__init__()
        self.L = num_frequencies
        self.include_input = include_input
        freqs = (2.0 ** torch.arange(num_frequencies)) * np.pi   # (L,)
        self.register_buffer("freqs", freqs)

    def out_dim(self, in_dim: int = 2) -> int:
        base = in_dim if self.include_input else 0
        return base + in_dim * 2 * self.L

    def forward(self, x):                       # x: (N, in_dim)
        out = [x] if self.include_input else []
        for f in self.freqs:
            out.append(torch.sin(f * x))
            out.append(torch.cos(f * x))
        return torch.cat(out, dim=-1)


class ChannelInterpolatorMLP(nn.Module):
    """PE -> 얕은 MLP(GELU) -> 240차원. Softplus 로 비음수화 후 블록별 L1 정규화.
    출력은 (PDP (N,3,32), APS (N,3,48)) 확률분포."""

    def __init__(self, num_frequencies=PE_L, hidden=HIDDEN, n_layers=N_LAYERS,
                 n_tx=N_TX, n_tau=N_TAU, n_aoa=N_AOA, use_pe=True):
        super().__init__()
        self.use_pe = use_pe
        self.n_tx, self.n_tau, self.n_aoa = n_tx, n_tau, n_aoa
        self.pe = PositionalEncoding(num_frequencies) if use_pe else None
        in_dim = self.pe.out_dim(2) if use_pe else 2

        layers = [nn.Linear(in_dim, hidden), nn.GELU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.GELU()]
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, n_tx * (n_tau + n_aoa))

    def forward(self, coords):
        h = self.pe(coords) if self.use_pe else coords
        h = self.backbone(h)
        raw = F.softplus(self.head(h))           # (N, 240) 비음수
        pdp = raw[:, :self.n_tx * self.n_tau].reshape(-1, self.n_tx, self.n_tau)
        aps = raw[:, self.n_tx * self.n_tau:].reshape(-1, self.n_tx, self.n_aoa)
        pdp = pdp / (pdp.sum(-1, keepdim=True) + 1e-9)   # 확률분포 정규화
        aps = aps / (aps.sum(-1, keepdim=True) + 1e-9)
        return pdp, aps


# =====================================================================
#  Custom Loss : MSE + 1D Wasserstein(CDF-L1) + L1
# =====================================================================
def wasserstein1d_torch(p, q):
    """행 확률분포의 1D Wasserstein = |ΔCDF|_1. p,q: (...,B) -> (...,)"""
    return (torch.cumsum(p, -1) - torch.cumsum(q, -1)).abs().sum(-1)


def channel_loss(pred_pdp, pred_aps, tgt_pdp, tgt_aps,
                 w_mse=W_MSE, w_wass=W_WASS, w_l1=W_L1):
    mse = F.mse_loss(pred_pdp, tgt_pdp) + F.mse_loss(pred_aps, tgt_aps)
    wass = wasserstein1d_torch(pred_pdp, tgt_pdp).mean() \
        + wasserstein1d_torch(pred_aps, tgt_aps).mean()
    l1 = (pred_pdp - tgt_pdp).abs().mean() + (pred_aps - tgt_aps).abs().mean()
    return w_mse * mse + w_wass * wass + w_l1 * l1


# =====================================================================
#  데이터 / 메트릭 (v1~v2 동일)
# =====================================================================
def lin(p_db):
    return 10.0 ** (np.asarray(p_db, dtype=np.float64) / 10.0)


def load():
    d = np.load(NPZ, allow_pickle=True)
    rxp = np.asarray(d["rx_positions"], dtype=np.float64)
    txp = np.asarray(d["tx_positions"], dtype=np.float64)
    N = rxp.shape[0]
    npaths = np.array([[len(d[f"tau_tx{T}_rx{i}"]) for T in range(N_TX)]
                       for i in range(N)])
    return d, rxp, txp, N, npaths


def stratified_sample(rxp, txp, npaths, n_want):
    ok = np.all(npaths >= MIN_PATHS, axis=1)
    idx = np.where(ok)[0]
    rad = np.sqrt(((rxp[idx] - txp[0, :2]) ** 2).sum(1))
    rng = np.random.default_rng(SEED0)
    bins = np.quantile(rad, np.linspace(0, 1, 11))
    per = max(1, n_want // 10)
    picks = []
    for b in range(10):
        m = (rad >= bins[b]) & (rad <= bins[b + 1] if b == 9 else rad < bins[b + 1])
        cand = idx[m]
        if len(cand):
            picks.append(rng.choice(cand, min(per, len(cand)), replace=False))
    return np.sort(np.concatenate(picks))


def build_features(d, sub):
    n = len(sub)
    aoa_edges = np.linspace(-180, 180, N_AOA + 1)
    PDP = np.zeros((n, N_TX, N_TAU))
    APS = np.zeros((n, N_TX, N_AOA))
    tmax = np.zeros(N_TX)
    for T in range(N_TX):
        allt = np.concatenate([np.asarray(d[f"tau_tx{T}_rx{i}"], float) for i in sub])
        tmax[T] = float(np.quantile(allt, 0.999))
    for k, i in enumerate(sub):
        for T in range(N_TX):
            p = lin(d[f"power_tx{T}_rx{i}"])
            a = np.asarray(d[f"aoa_tx{T}_rx{i}"], float)
            t = np.asarray(d[f"tau_tx{T}_rx{i}"], float)
            if p.size and p.sum() > 0:
                ba = np.clip(np.digitize(a, aoa_edges) - 1, 0, N_AOA - 1)
                np.add.at(APS[k, T], ba, p)
                tau_edges = np.linspace(0, tmax[T], N_TAU + 1)
                bt = np.clip(np.digitize(np.clip(t, 0, tmax[T] * (1 - 1e-9)),
                                         tau_edges) - 1, 0, N_TAU - 1)
                np.add.at(PDP[k, T], bt, p)
    PDP /= PDP.sum(-1, keepdims=True) + 1e-30
    APS /= APS.sum(-1, keepdims=True) + 1e-30
    return PDP, APS


def steering_matrix():
    centers = -180 + 360 * (np.arange(N_AOA) + 0.5) / N_AOA
    th = np.deg2rad(centers)
    m = np.arange(M_ULA)[:, None]
    return np.exp(1j * np.pi * m * np.sin(th)[None, :])


A_STEER = steering_matrix()


def aps_to_cov(aps):
    R = (A_STEER * aps[None, :]) @ A_STEER.conj().T
    return R / (np.real(np.trace(R)) + 1e-30)


def bw2_trace1(Ri, Rj):
    wi, Ui = np.linalg.eigh(Ri); wi = np.clip(wi, 0, None)
    Rh = (Ui * np.sqrt(wi)) @ Ui.conj().T
    Mi = Rh @ Rj @ Rh
    ev = np.clip(np.linalg.eigvalsh((Mi + Mi.conj().T) / 2), 0, None)
    return float(np.clip(2.0 - 2.0 * np.sqrt(ev).sum(), 0, 2))


def pdp_w1(p, q):
    return float(np.abs(np.cumsum(p) - np.cumsum(q)).sum() / N_TAU)


def idw_pos(pos_o, feat_o, pos_q, power=2.0, k=20):
    d = np.sqrt(((pos_q[:, None] - pos_o[None]) ** 2).sum(-1))
    kk = min(k, pos_o.shape[0])
    nn = np.argpartition(d, kk - 1, axis=1)[:, :kk]
    out = np.zeros((pos_q.shape[0], feat_o.shape[1]))
    for r in range(pos_q.shape[0]):
        dd = d[r, nn[r]]; w = 1.0 / (dd ** power + 1e-9); w /= w.sum()
        out[r] = w @ feat_o[nn[r]]
    return out


def train_predict_mlp(pos_o, pdp_o, aps_o, pos_q, use_pe, bounds):
    """좌표를 bounds 로 [-1,1] 정규화 후 학습/추론. 반환 (nq,3,32),(nq,3,48)."""
    lo, hi = bounds
    def norm(P):
        return (2 * (P - lo) / (hi - lo + 1e-9) - 1.0).astype(np.float32)
    Xo = torch.tensor(norm(pos_o), device=DEVICE)
    Xq = torch.tensor(norm(pos_q), device=DEVICE)
    Tp = torch.tensor(pdp_o, dtype=torch.float32, device=DEVICE)
    Ta = torch.tensor(aps_o, dtype=torch.float32, device=DEVICE)

    net = ChannelInterpolatorMLP(use_pe=use_pe).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WD)
    net.train()
    for _ in range(EPOCHS):
        opt.zero_grad()
        pp, pa = net(Xo)
        loss = channel_loss(pp, pa, Tp, Ta)
        loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        qp, qa = net(Xq)
    return qp.cpu().numpy(), qa.cpu().numpy()


def score(pdp_hat, aps_hat, pdp_t, aps_t, qsel):
    ps, cs = [], []
    for r in qsel:
        for T in range(N_TX):
            ps.append(1.0 - pdp_w1(pdp_hat[r, T], pdp_t[r, T]))
            cs.append(1.0 - bw2_trace1(aps_to_cov(aps_hat[r, T]),
                                       aps_to_cov(aps_t[r, T])) / 2.0)
    pdp_sim = float(np.clip(np.mean(ps), 0, 1))
    cov_sim = float(np.clip(np.mean(cs), 0, 1))
    return pdp_sim, cov_sim, 0.5 * (pdp_sim + cov_sim)


def main():
    print(f"[load] device={DEVICE}  PE_L={PE_L} hidden={HIDDEN} layers={N_LAYERS}")
    d, rxp, txp, N, npaths = load()
    sub = stratified_sample(rxp, txp, npaths, N_SAMPLE)
    n = len(sub); pos = rxp[sub]
    PDP, APS = build_features(d, sub)
    pdp_flat = PDP.reshape(n, -1); aps_flat = APS.reshape(n, -1)
    bounds = (pos.min(0), pos.max(0))
    print(f"[sample] n={n}, 3-BS, target_dim={N_TX*(N_TAU+N_AOA)}")

    methods = ["단순보간 IDW(위치)", "Plain MLP(위치)", "Fourier MLP(위치,NeRF)"]
    res = {m: {p: [] for p in RATIOS} for m in methods}
    floor_res = {p: [] for p in RATIOS}

    for p in RATIOS:
        for s in SEEDS:
            rng = np.random.default_rng(1000 * s + int(p * 100))
            perm = rng.permutation(n)
            no = max(8, int(round(p * n)))
            obs, qry = perm[:no], perm[no:]
            if len(qry) == 0:
                continue
            nq = len(qry)
            qsel = qry if nq <= SCORE_CAP else rng.choice(qry, SCORE_CAP, replace=False)
            qmap = {g: l for l, g in enumerate(qry)}
            qloc = np.array([qmap[g] for g in qsel])
            pos_o, pos_q = pos[obs], pos[qry]
            pdp_t, aps_t = PDP[qry], APS[qry]

            pdp_m = PDP[obs].mean(0); pdp_m /= pdp_m.sum(-1, keepdims=True) + 1e-30
            aps_m = APS[obs].mean(0); aps_m /= aps_m.sum(-1, keepdims=True) + 1e-30
            floor_res[p].append(score(np.broadcast_to(pdp_m, (nq, N_TX, N_TAU)),
                                      np.broadcast_to(aps_m, (nq, N_TX, N_AOA)),
                                      pdp_t, aps_t, qloc))

            # (0) IDW
            ph = idw_pos(pos_o, pdp_flat[obs], pos_q).reshape(-1, N_TX, N_TAU)
            ah = idw_pos(pos_o, aps_flat[obs], pos_q).reshape(-1, N_TX, N_AOA)
            ph = np.clip(ph, 0, None); ph /= ph.sum(-1, keepdims=True) + 1e-30
            ah = np.clip(ah, 0, None); ah /= ah.sum(-1, keepdims=True) + 1e-30
            res[methods[0]][p].append(score(ph, ah, pdp_t, aps_t, qloc))

            # (1) Plain MLP (PE 없음)
            ph, ah = train_predict_mlp(pos_o, PDP[obs], APS[obs], pos_q,
                                       use_pe=False, bounds=bounds)
            res[methods[1]][p].append(score(ph, ah, pdp_t, aps_t, qloc))

            # (2) Fourier MLP
            ph, ah = train_predict_mlp(pos_o, PDP[obs], APS[obs], pos_q,
                                       use_pe=True, bounds=bounds)
            res[methods[2]][p].append(score(ph, ah, pdp_t, aps_t, qloc))

            print(f"  p={p:.2f} s={s}  floor={floor_res[p][-1][2]:.3f}  "
                  f"IDW={res[methods[0]][p][-1][2]:.3f}  "
                  f"Plain={res[methods[1]][p][-1][2]:.3f}  "
                  f"Fourier={res[methods[2]][p][-1][2]:.3f}")

    # ---------------- 시각화 ----------------
    colors = {methods[0]: "#9e9e9e", methods[1]: "#3a6fb0", methods[2]: "#1b9e77"}
    styles = {methods[0]: "--", methods[1]: "-", methods[2]: "-"}
    xpct = np.array(RATIOS) * 100
    floor_mean = np.array([np.mean([v[2] for v in floor_res[p]]) for p in RATIOS])

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(17, 6.4))
    for m in methods:
        means = np.array([np.mean([v[2] for v in res[m][p]]) for p in RATIOS])
        for p in RATIOS:
            ys = [v[2] for v in res[m][p]]
            ax.scatter([p * 100] * len(ys), ys, s=20, color=colors[m],
                       alpha=0.30, edgecolors="none", zorder=2)
        ax.plot(xpct, means, styles[m], color=colors[m], lw=2.6, label=m, zorder=3)
    ax.plot(xpct, floor_mean, ":", color="#444444", lw=1.8, label="floor(전역평균)")
    ax.set_xlabel("관측 비율 (%)  →  관측 풍부", fontsize=12)
    ax.set_ylabel("추론 vs True 유사도 (1=완전일치)", fontsize=12)
    ax.set_title("(a) 원 유사도 S", fontsize=12)
    ax.set_ylim(0.5, 1.0); ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=10, framealpha=0.9)

    for m in methods:
        sk_mean = []
        for ip, p in enumerate(RATIOS):
            fl = floor_mean[ip]
            sk = [(v[2] - fl) / (1 - fl + 1e-9) for v in res[m][p]]
            sk_mean.append(np.mean(sk))
            ax2.scatter([p * 100] * len(sk), sk, s=20, color=colors[m],
                        alpha=0.30, edgecolors="none", zorder=2)
        ax2.plot(xpct, sk_mean, styles[m], color=colors[m], lw=2.6, label=m, zorder=3)
    ax2.axhline(0, color="#444444", ls=":", lw=1.8, label="floor 기준")
    ax2.set_xlabel("관측 비율 (%)  →  관측 풍부", fontsize=12)
    ax2.set_ylabel("Skill score (S−floor)/(1−floor)", fontsize=12)
    ax2.set_title("(b) Skill score  (Fourier 위치인코딩의 효과)", fontsize=12)
    ax2.grid(alpha=0.3)
    ax2.legend(loc="lower right", fontsize=10, framealpha=0.9)

    fig.suptitle("RAN Twin v3 : Fourier(NeRF) MLP vs Plain MLP vs 위치보간  "
                 "(채널=PDP+재구성Cov, GHM 3-BS)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fpng = os.path.join(OUTDIR, f"P2C_v3_fourier_curve_{STAMP}.png")
    fig.savefig(fpng, dpi=140)
    print(f"[save] {fpng}")

    fcsv = os.path.join(OUTDIR, f"P2C_v3_fourier_{STAMP}.csv")
    with open(fcsv, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["method", "obs_ratio", "seed_run", "pdp_sim", "cov_sim", "S", "skill"])
        for ip, p in enumerate(RATIOS):
            fl = floor_mean[ip]
            for j, (ps, cs, S) in enumerate(floor_res[p]):
                wr.writerow(["floor", f"{p:.2f}", j, f"{ps:.4f}", f"{cs:.4f}", f"{S:.4f}", "0"])
            for m in methods:
                for j, (ps, cs, S) in enumerate(res[m][p]):
                    wr.writerow([m, f"{p:.2f}", j, f"{ps:.4f}", f"{cs:.4f}",
                                 f"{S:.4f}", f"{(S-fl)/(1-fl+1e-9):.4f}"])
    print(f"[save] {fcsv}")

    print("\n=== 관측비율별 평균 S | skill ===")
    for ip, p in enumerate(RATIOS):
        fl = floor_mean[ip]
        Srow = "  ".join(f"{np.mean([v[2] for v in res[m][p]]):.3f}" for m in methods)
        skrow = "  ".join(
            f"{np.mean([(v[2]-fl)/(1-fl+1e-9) for v in res[m][p]]):+.3f}" for m in methods)
        print(f"{p*100:4.0f}%  floor={fl:.3f}  S=[{Srow}]  skill=[{skrow}]")


if __name__ == "__main__":
    main()
