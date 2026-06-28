# -*- coding: utf-8 -*-
"""
P2C_RANTwin_Interpolator_2606v3.py
==================================
[ v2 -> v3 변경 사유 ]
v2 의 'Fourier feature + KL + 표본↑' MLP(귀납적 위치→분포 회귀)는 GP-Kriging 보다
오히려 낮았다(skill 0.25 vs 0.62). 원인은 하이퍼파라미터가 아니라 **귀납 vs 전이**
구조 차이:
  - GP/IDW(전이적) : 질의 주변 '실제 관측 채널' 을 직접 혼합 -> NLoS 국소 디테일 활용.
  - 위치->분포 회귀 MLP(귀납적) : 정보를 가중치에 압축 -> 큰 국소분산 앞에서
    매끄러운 평균(floor 근처)으로 붕괴. FF scale↑ 면 고주파 암기로 held-out 악화.

[ v3 해결책 : Deep-Kernel Attention (학습형 전이적 보간) ]
  관측점들에 대한 '학습된 어텐션 가중치' 로 채널을 혼합한다(=학습하는 IDW/GP).
    - 임베딩 g: FourierFeature(x,y) -> MLP -> e차원. 학습형 온도 τ.
    - 물리거리 항(스케일 학습)을 logits 에 더해 최소 IDW 커널 보장.
    - 어텐션 w(q,j)=softmax_j(-‖g(xq)-g(xj)‖²/τ - ‖xq-xj‖²/ℓ²).
    - 예측 = Σ_j w(q,j)·Y_obs[j]  (관측 분포들의 볼록결합 -> 분포 정합 자동).
  학습(전이적) : 관측셋 LOO(자기 제외) 어텐션으로 관측채널 복원 KL 최소화
    -> 데이터 누수 없이 '관측 분포로 일반화되는 커널' 학습. held-out 에 그대로 적용.

[ 목적/프로토콜/메트릭은 v1·v2 와 동일 ] (관측비율 sweep, 시드4, S=PDP+Cov)

----------------------------------------------------------------------
실행 환경 : Python 3.10.12 / dclcom61 / numpy·scipy·torch(2.11+cu128)·matplotlib
데이터    : 260512 RT Result Data/channel_data_260531_GHM_Twin_v0_1.npz (TX3, SISO)
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

for _fp in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]:
    if os.path.exists(_fp):
        font_manager.fontManager.addfont(_fp)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=_fp).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

# ---------------------------------------------------------------- 설정
NPZ = "/home/dclcom61/twin_minji/260512 RT Result Data/channel_data_260531_GHM_Twin_v0_1.npz"
OUTDIR = "/home/dclcom61/twin_minji/260604 metric 도출/P2C_RANTwin_Interpolator_Results"
N_TX = 3
N_SAMPLE = 1800
MIN_PATHS = 10
N_TAU = 32
N_AOA = 48
M_ULA = 16
RATIOS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
SEEDS = [0, 1, 2, 3]
SCORE_CAP = 400
SEED0 = 0

# Deep-Kernel Attention 하이퍼파라미터
FF_NUM = 64
FF_SCALE = 1.0
EMB_DIM = 32
DK_EPOCHS = 700
DK_LR = 5e-3
DK_WD = 1e-5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(OUTDIR, exist_ok=True)
STAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


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
            if p.size:
                ba = np.clip(np.digitize(a, aoa_edges) - 1, 0, N_AOA - 1)
                np.add.at(APS[k, T], ba, p)
                tau_edges = np.linspace(0, tmax[T], N_TAU + 1)
                bt = np.clip(np.digitize(np.clip(t, 0, tmax[T] * (1 - 1e-9)),
                                         tau_edges) - 1, 0, N_TAU - 1)
                np.add.at(PDP[k, T], bt, p)
    PDP /= PDP.sum(-1, keepdims=True) + 1e-30
    APS /= APS.sum(-1, keepdims=True) + 1e-30
    return PDP, APS


# ---------------------------------------------------------------- 공분산 / 메트릭
def steering_matrix():
    centers = -180 + 360 * (np.arange(N_AOA) + 0.5) / N_AOA
    th = np.deg2rad(centers)
    m = np.arange(M_ULA)[:, None]
    return np.exp(1j * np.pi * m * np.sin(th)[None, :])


A_STEER = steering_matrix()


def aps_to_cov(aps):
    R = (A_STEER * aps[None, :]) @ A_STEER.conj().T
    tr = np.real(np.trace(R))
    return R / (tr + 1e-30)


def bw2_trace1(Ri, Rj):
    wi, Ui = np.linalg.eigh(Ri)
    wi = np.clip(wi, 0, None)
    Rh = (Ui * np.sqrt(wi)) @ Ui.conj().T
    Minner = Rh @ Rj @ Rh
    ev = np.linalg.eigvalsh((Minner + Minner.conj().T) / 2)
    ev = np.clip(ev, 0, None)
    return float(np.clip(2.0 - 2.0 * np.sqrt(ev).sum(), 0, 2))


def pdp_w1(p, q):
    c = np.cumsum(p) - np.cumsum(q)
    return float(np.abs(c).sum() / N_TAU)


# ---------------------------------------------------------------- 보간기 (baseline)
def idw_predict(pos_o, feat_o, pos_q, power=2.0, k=20):
    d = np.sqrt(((pos_q[:, None] - pos_o[None]) ** 2).sum(-1))
    kk = min(k, pos_o.shape[0])
    nn = np.argpartition(d, kk - 1, axis=1)[:, :kk]
    out = np.zeros((pos_q.shape[0], feat_o.shape[1]))
    for r in range(pos_q.shape[0]):
        dd = d[r, nn[r]]
        w = 1.0 / (dd ** power + 1e-9)
        w /= w.sum()
        out[r] = w @ feat_o[nn[r]]
    return out


def fit_corr_length(pos_o, aps_o):
    no = pos_o.shape[0]
    m = min(no, 250)
    ridx = np.random.default_rng(0).choice(no, m, replace=False)
    P = pos_o[ridx]; F = aps_o[ridx]
    d = np.sqrt(((P[:, None] - P[None]) ** 2).sum(-1))
    Fn = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-30)
    S = Fn @ Fn.T
    iu = np.triu_indices(m, 1)
    dd, ss = d[iu], np.clip(S[iu], 1e-6, 1 - 1e-6)
    y = -np.log(ss)
    slope = (dd * y).sum() / ((dd * dd).sum() + 1e-30)
    ell = 1.0 / (slope + 1e-9)
    return float(np.clip(ell, 20.0, 2000.0))


def kriging_predict(pos_o, feat_o, pos_q, ell):
    no = pos_o.shape[0]
    Do = np.sqrt(((pos_o[:, None] - pos_o[None]) ** 2).sum(-1))
    C = np.exp(-Do / ell) + 1e-6 * np.eye(no)
    Cinv = np.linalg.inv(C)
    dq = np.sqrt(((pos_q[:, None] - pos_o[None]) ** 2).sum(-1))
    c0 = np.exp(-dq / ell)
    W = c0 @ Cinv
    W = np.clip(W, 0, None)
    W /= W.sum(1, keepdims=True) + 1e-30
    return W @ feat_o


# ---------------------------------------------------------------- v3: Deep-Kernel Attention
class DeepKernel(nn.Module):
    """위치 -> 임베딩 g(x). 어텐션 logits 에 물리거리 항도 학습 스케일로 결합."""
    def __init__(self, ff_num=FF_NUM, ff_scale=FF_SCALE, emb=EMB_DIM):
        super().__init__()
        B = torch.randn(2, ff_num) * ff_scale       # 고정 Fourier 주파수
        self.register_buffer("B", B)
        self.net = nn.Sequential(
            nn.Linear(2 * ff_num, 128), nn.GELU(),
            nn.Linear(128, 128), nn.GELU(),
            nn.Linear(128, emb))
        self.log_tau = nn.Parameter(torch.zeros(1))      # 임베딩거리 온도
        self.log_ell = nn.Parameter(torch.zeros(1))      # 물리거리 길이스케일(정규화좌표)

    def embed(self, Xn):
        proj = 2 * np.pi * (Xn @ self.B)
        ff = torch.cat([torch.sin(proj), torch.cos(proj)], dim=1)
        return self.net(ff)

    def logits(self, Xa, Xb):
        """질의 Xa(na,2정규화), 관측 Xb(nb,2) -> attention logits (na,nb)."""
        ea, eb = self.embed(Xa), self.embed(Xb)
        d_emb = torch.cdist(ea, eb) ** 2
        d_phys = torch.cdist(Xa, Xb) ** 2
        tau = torch.exp(self.log_tau) + 1e-4
        ell2 = torch.exp(2 * self.log_ell) + 1e-4
        return -(d_emb / tau) - (d_phys / ell2)


def deepkernel_predict(pos_o, pdp_o, aps_o, pos_q, seed=0):
    """관측셋 LOO 어텐션으로 커널 학습 -> 질의에 적용. 출력=관측분포 볼록결합."""
    torch.manual_seed(seed); np.random.seed(seed)
    mu, sd = pos_o.mean(0), pos_o.std(0) + 1e-9
    Xo = torch.tensor((pos_o - mu) / sd, dtype=torch.float32, device=DEVICE)
    Xq = torch.tensor((pos_q - mu) / sd, dtype=torch.float32, device=DEVICE)
    no = Xo.shape[0]

    Yp = torch.tensor(pdp_o.reshape(no, -1), dtype=torch.float32, device=DEVICE)  # (no,96)
    Ya = torch.tensor(aps_o.reshape(no, -1), dtype=torch.float32, device=DEVICE)  # (no,144)
    Yp3 = Yp.reshape(no, N_TX, N_TAU); Ya3 = Ya.reshape(no, N_TX, N_AOA)

    net = DeepKernel().to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=DK_LR, weight_decay=DK_WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=DK_EPOCHS)

    eye_mask = torch.eye(no, device=DEVICE, dtype=torch.bool)
    for _ in range(DK_EPOCHS):
        opt.zero_grad()
        L = net.logits(Xo, Xo)
        L = L.masked_fill(eye_mask, float("-inf"))      # LOO: 자기 제외
        W = torch.softmax(L, dim=1)                     # (no,no)
        ph = (W @ Yp).reshape(no, N_TX, N_TAU).clamp_min(1e-9)
        ah = (W @ Ya).reshape(no, N_TX, N_AOA).clamp_min(1e-9)
        # KL(target || pred), 블록 평균
        klp = (Yp3 * ((Yp3 + 1e-9).log() - ph.log())).sum(-1).mean()
        kla = (Ya3 * ((Ya3 + 1e-9).log() - ah.log())).sum(-1).mean()
        (klp + kla).backward(); opt.step(); sched.step()

    net.eval()
    with torch.no_grad():
        Wq = torch.softmax(net.logits(Xq, Xo), dim=1)   # (nq,no)
        ph = (Wq @ Yp).reshape(-1, N_TX, N_TAU)
        ah = (Wq @ Ya).reshape(-1, N_TX, N_AOA)
    ph = ph / (ph.sum(-1, keepdim=True) + 1e-30)
    ah = ah / (ah.sum(-1, keepdim=True) + 1e-30)
    return ph.cpu().numpy(), ah.cpu().numpy()


# ---------------------------------------------------------------- 채점
def score(pdp_hat, aps_hat, pdp_t, aps_t, qsel):
    ps, cs = [], []
    for r in qsel:
        for T in range(N_TX):
            ps.append(1.0 - pdp_w1(pdp_hat[r, T], pdp_t[r, T]))
            Rh = aps_to_cov(aps_hat[r, T]); Rt = aps_to_cov(aps_t[r, T])
            cs.append(1.0 - bw2_trace1(Rh, Rt) / 2.0)
    pdp_sim = float(np.clip(np.mean(ps), 0, 1))
    cov_sim = float(np.clip(np.mean(cs), 0, 1))
    return pdp_sim, cov_sim, 0.5 * (pdp_sim + cov_sim)


def main():
    print(f"[load] GHM Twin npz ... device={DEVICE}")
    d, rxp, txp, N, npaths = load()
    sub = stratified_sample(rxp, txp, npaths, N_SAMPLE)
    n = len(sub)
    pos = rxp[sub]
    print(f"[sample] n={n}, TX={N_TX}, ULA M={M_ULA}, FF={FF_NUM}/scale={FF_SCALE}, emb={EMB_DIM}")

    PDP, APS = build_features(d, sub)
    pdp_flat = PDP.reshape(n, -1)
    aps_flat = APS.reshape(n, -1)

    methods = ["IDW(단순보간)", "GP-Kriging(RAN Twin)", "DeepKernel(RAN Twin)"]
    res = {m: {p: [] for p in RATIOS} for m in methods}
    floor_res = {p: [] for p in RATIOS}

    for p in RATIOS:
        for s in SEEDS:
            rng = np.random.default_rng(1000 * s + int(p * 100))
            perm = rng.permutation(n)
            no = max(5, int(round(p * n)))
            obs, qry = perm[:no], perm[no:]
            if len(qry) == 0:
                continue
            qsel = qry if len(qry) <= SCORE_CAP else \
                rng.choice(qry, SCORE_CAP, replace=False)
            qmap = {g: l for l, g in enumerate(qry)}
            qloc = np.array([qmap[g] for g in qsel])

            pos_o, pos_q = pos[obs], pos[qry]
            pdp_t = PDP[qry]; aps_t = APS[qry]
            nq = len(qry)

            pdp_m = PDP[obs].mean(0); pdp_m /= pdp_m.sum(-1, keepdims=True) + 1e-30
            aps_m = APS[obs].mean(0); aps_m /= aps_m.sum(-1, keepdims=True) + 1e-30
            ph_f = np.broadcast_to(pdp_m, (nq, N_TX, N_TAU))
            ah_f = np.broadcast_to(aps_m, (nq, N_TX, N_AOA))
            floor_res[p].append(score(ph_f, ah_f, pdp_t, aps_t, qloc))

            ph = idw_predict(pos_o, pdp_flat[obs], pos_q).reshape(-1, N_TX, N_TAU)
            ah = idw_predict(pos_o, aps_flat[obs], pos_q).reshape(-1, N_TX, N_AOA)
            ah = np.clip(ah, 0, None); ah /= ah.sum(-1, keepdims=True) + 1e-30
            ph = np.clip(ph, 0, None); ph /= ph.sum(-1, keepdims=True) + 1e-30
            res[methods[0]][p].append(score(ph, ah, pdp_t, aps_t, qloc))

            ell = fit_corr_length(pos_o, aps_flat[obs])
            ph = kriging_predict(pos_o, pdp_flat[obs], pos_q, ell).reshape(-1, N_TX, N_TAU)
            ah = kriging_predict(pos_o, aps_flat[obs], pos_q, ell).reshape(-1, N_TX, N_AOA)
            ah = np.clip(ah, 0, None); ah /= ah.sum(-1, keepdims=True) + 1e-30
            ph = np.clip(ph, 0, None); ph /= ph.sum(-1, keepdims=True) + 1e-30
            res[methods[1]][p].append(score(ph, ah, pdp_t, aps_t, qloc))

            ph, ah = deepkernel_predict(pos_o, PDP[obs], APS[obs], pos_q, seed=s)
            res[methods[2]][p].append(score(ph, ah, pdp_t, aps_t, qloc))

            print(f"  p={p:.1f} seed={s}  ell={ell:6.1f}  "
                  f"IDW={res[methods[0]][p][-1][2]:.3f}  "
                  f"GP={res[methods[1]][p][-1][2]:.3f}  "
                  f"DK={res[methods[2]][p][-1][2]:.3f}")

    # ---------------------------------------------------------------- 시각화
    colors = {"IDW(단순보간)": "#9e9e9e",
              "GP-Kriging(RAN Twin)": "#1b9e77",
              "DeepKernel(RAN Twin)": "#d62728"}
    styles = {"IDW(단순보간)": "--",
              "GP-Kriging(RAN Twin)": "-",
              "DeepKernel(RAN Twin)": "-"}

    xpct = np.array(RATIOS) * 100
    floor_mean = np.array([np.mean([v[2] for v in floor_res[p]]) for p in RATIOS])

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(17, 6.4))
    for m in methods:
        means = np.array([np.mean([v[2] for v in res[m][p]]) for p in RATIOS])
        for p in RATIOS:
            ys = [v[2] for v in res[m][p]]
            ax.scatter([p * 100] * len(ys), ys, s=22, color=colors[m],
                       alpha=0.35, edgecolors="none", zorder=2)
        ax.plot(xpct, means, styles[m], color=colors[m], lw=2.6, label=m, zorder=3)
    ax.plot(xpct, floor_mean, ":", color="#444444", lw=1.8,
            label="floor(전역평균, 공간정보 無)", zorder=1)
    ax.set_xlabel("관측 비율 (%)  →  관측 풍부", fontsize=12)
    ax.set_ylabel("추론 vs True 유사도  (1 = 완전 일치)", fontsize=12)
    ax.set_title("(a) 원 유사도 S", fontsize=12)
    ax.set_ylim(0.5, 1.0)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=10, framealpha=0.9)

    for m in methods:
        sk_mean = []
        for ip, p in enumerate(RATIOS):
            fl = floor_mean[ip]
            sk = [(v[2] - fl) / (1 - fl + 1e-9) for v in res[m][p]]
            sk_mean.append(np.mean(sk))
            ax2.scatter([p * 100] * len(sk), sk, s=22, color=colors[m],
                        alpha=0.35, edgecolors="none", zorder=2)
        ax2.plot(xpct, sk_mean, styles[m], color=colors[m], lw=2.6, label=m, zorder=3)
    ax2.axhline(0, color="#444444", ls=":", lw=1.8, label="floor 기준(skill=0)")
    ax2.set_xlabel("관측 비율 (%)  →  관측 풍부", fontsize=12)
    ax2.set_ylabel("Skill score  (S−floor)/(1−floor)", fontsize=12)
    ax2.set_title("(b) Skill score  (바닥 대비 정규화)", fontsize=12)
    ax2.grid(alpha=0.3)
    ax2.legend(loc="lower right", fontsize=10, framealpha=0.9)

    fig.suptitle("RAN Twin 보간 v3 (DeepKernel Attention=학습형 전이적 보간) : "
                 "부분관측 학습 → 미관측 추론  (GHM Twin 3-BS)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fpng = os.path.join(OUTDIR, f"P2C_interp_curve_v3_{STAMP}.png")
    fig.savefig(fpng, dpi=140)
    print(f"[save] {fpng}")

    fcsv = os.path.join(OUTDIR, f"P2C_interp_v3_{STAMP}.csv")
    with open(fcsv, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["method", "obs_ratio", "seed_run", "pdp_sim", "cov_sim", "S", "skill"])
        for ip, p in enumerate(RATIOS):
            fl = floor_mean[ip]
            for j, (ps, cs, S) in enumerate(floor_res[p]):
                wr.writerow(["floor(전역평균)", f"{p:.2f}", j, f"{ps:.4f}",
                             f"{cs:.4f}", f"{S:.4f}", "0.0000"])
            for m in methods:
                for j, (ps, cs, S) in enumerate(res[m][p]):
                    sk = (S - fl) / (1 - fl + 1e-9)
                    wr.writerow([m, f"{p:.2f}", j, f"{ps:.4f}", f"{cs:.4f}",
                                 f"{S:.4f}", f"{sk:.4f}"])
    print(f"[save] {fcsv}")

    print("\n=== 관측비율별 평균 S | skill score ===")
    for ip, p in enumerate(RATIOS):
        fl = floor_mean[ip]
        Srow = "  ".join(f"{np.mean([v[2] for v in res[m][p]]):.3f}" for m in methods)
        skrow = "  ".join(
            f"{np.mean([(v[2]-fl)/(1-fl+1e-9) for v in res[m][p]]):+.3f}"
            for m in methods)
        print(f"{p*100:4.0f}%  floor={fl:.3f}  S=[{Srow}]  skill=[{skrow}]")


if __name__ == "__main__":
    main()
