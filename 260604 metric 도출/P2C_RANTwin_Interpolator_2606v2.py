# -*- coding: utf-8 -*-
"""
P2C_RANTwin_Interpolator_2606v2.py
==================================
[ v1 -> v2 변경 사유 ]
v1 은 보간기에 '위치(x,y)'만 줘서 학습(GP/MLP)과 단순보간(IDW)의 격차가 작았다
(둘 다 거리가중 국소평균). v2 는 디지털트윈의 '실측 후 업데이트' 개념을 도입한다:
질의 RX 에서 **싸게 측정 가능한 값(다중 BS RSRP, 평균지연)** 을 학습모델 입력에
추가하여, 비싼 전체 채널(PDP + 재구성 Cov)을 추론한다. 단순보간(baseline)은
위치만 쓰므로 이 측정정보를 활용 못 함 -> 학습 우위가 커진다.

[ 평가 프로토콜 (train/held-out) : v1 과 동일 ]
  - 관측 RX = 학습용, 미관측 RX = 정답(test).
  - 단, '위치'와 '싼 측정값(RSRP/지연)'은 미관측 RX 에서도 알 수 있다고 가정
    (UE 가 셀에 접속하면 RSRP/타이밍은 즉시 측정됨). '비싼 전체 채널'만 추론 대상.

[ 채널 표현 / 유사도 : v1 과 동일 ]
  - BS별 PDP(32) + APS(48) -> 가상 ULA(M=16) Cov 재구성(trace=1).
  - S = 0.5*(1-W1_PDP) + 0.5*(1-BW_Cov/2), 3 BS·held-out 평균.
  - skill = (S - floor)/(1 - floor), floor=전역평균(공간/측정정보 無).

[ 비교 ]
  (0) 단순보간(학습X) : 위치-only IDW                      회색 점선
  (1) RAN Twin kNN    : (위치+RSRP+지연) 표준화공간 kNN가중  녹색 실선
  (2) RAN Twin MLP    : (위치+RSRP+지연) -> 채널분포 회귀     파랑 실선

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
RATIOS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
SEEDS = [0, 1, 2, 3]
SCORE_CAP = 400
SEED0 = 0
KNN_K = 12

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


def build_features(d, sub, pos):
    """반환: PDP(n,3,32), APS(n,3,48), CTX(n,8)=[x,y,rsrp0..2,mdelay0..2]."""
    n = len(sub)
    aoa_edges = np.linspace(-180, 180, N_AOA + 1)
    PDP = np.zeros((n, N_TX, N_TAU))
    APS = np.zeros((n, N_TX, N_AOA))
    rsrp = np.zeros((n, N_TX))
    mdel = np.zeros((n, N_TX))

    tmax = np.zeros(N_TX)
    for T in range(N_TX):
        allt = np.concatenate([np.asarray(d[f"tau_tx{T}_rx{i}"], float) for i in sub])
        tmax[T] = float(np.quantile(allt, 0.999))

    for k, i in enumerate(sub):
        for T in range(N_TX):
            p = lin(d[f"power_tx{T}_rx{i}"])
            a = np.asarray(d[f"aoa_tx{T}_rx{i}"], float)
            t = np.asarray(d[f"tau_tx{T}_rx{i}"], float)
            rsrp[k, T] = 10 * np.log10(p.sum() + 1e-30)
            if p.size and p.sum() > 0:
                mdel[k, T] = (t * p).sum() / p.sum()
                ba = np.clip(np.digitize(a, aoa_edges) - 1, 0, N_AOA - 1)
                np.add.at(APS[k, T], ba, p)
                tau_edges = np.linspace(0, tmax[T], N_TAU + 1)
                bt = np.clip(np.digitize(np.clip(t, 0, tmax[T] * (1 - 1e-9)),
                                         tau_edges) - 1, 0, N_TAU - 1)
                np.add.at(PDP[k, T], bt, p)
    PDP /= PDP.sum(-1, keepdims=True) + 1e-30
    APS /= APS.sum(-1, keepdims=True) + 1e-30
    CTX = np.concatenate([pos, rsrp, mdel], axis=1)     # (n,8)
    return PDP, APS, CTX


# ---------------- 공분산 / 메트릭 (v1 동일) ----------------
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


# ---------------- 보간기 ----------------
def idw_pos(pos_o, feat_o, pos_q, power=2.0, k=20):
    """단순보간(학습X): 위치-only IDW."""
    d = np.sqrt(((pos_q[:, None] - pos_o[None]) ** 2).sum(-1))
    kk = min(k, pos_o.shape[0])
    nn = np.argpartition(d, kk - 1, axis=1)[:, :kk]
    out = np.zeros((pos_q.shape[0], feat_o.shape[1]))
    for r in range(pos_q.shape[0]):
        dd = d[r, nn[r]]
        w = 1.0 / (dd ** power + 1e-9); w /= w.sum()
        out[r] = w @ feat_o[nn[r]]
    return out


def knn_ctx(ctx_o, feat_o, ctx_q, k=KNN_K):
    """RAN Twin kNN: 표준화 context(위치+실측) 공간에서 거리가중 결합."""
    d = np.sqrt(((ctx_q[:, None] - ctx_o[None]) ** 2).sum(-1))
    kk = min(k, ctx_o.shape[0])
    nn = np.argpartition(d, kk - 1, axis=1)[:, :kk]
    out = np.zeros((ctx_q.shape[0], feat_o.shape[1]))
    for r in range(ctx_q.shape[0]):
        dd = d[r, nn[r]]
        sig = np.median(dd) + 1e-6
        w = np.exp(-(dd / sig) ** 2); w /= w.sum() + 1e-30
        out[r] = w @ feat_o[nn[r]]
    return out


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, out_dim))

    def forward(self, x):
        return self.net(x)


def mlp_ctx(ctx_o, pdp_o, aps_o, ctx_q, epochs=400):
    """RAN Twin MLP: context(8) -> (PDP 3블록 + APS 3블록), 블록 softmax."""
    mu, sd = ctx_o.mean(0), ctx_o.std(0) + 1e-9
    Xo = torch.tensor((ctx_o - mu) / sd, dtype=torch.float32, device=DEVICE)
    Xq = torch.tensor((ctx_q - mu) / sd, dtype=torch.float32, device=DEVICE)
    Yp3 = torch.tensor(pdp_o, dtype=torch.float32, device=DEVICE)
    Ya3 = torch.tensor(aps_o, dtype=torch.float32, device=DEVICE)
    out_dim = N_TX * N_TAU + N_TX * N_AOA
    net = MLP(ctx_o.shape[1], out_dim).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=8e-3, weight_decay=1e-5)

    def split(logits):
        a = logits[:, :N_TX * N_TAU].reshape(-1, N_TX, N_TAU)
        b = logits[:, N_TX * N_TAU:].reshape(-1, N_TX, N_AOA)
        return torch.softmax(a, -1), torch.softmax(b, -1)

    for _ in range(epochs):
        opt.zero_grad()
        pa, pb = split(net(Xo))
        loss = ((pa - Yp3) ** 2).mean() + ((pb - Ya3) ** 2).mean()
        loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        qa, qb = split(net(Xq))
    return qa.cpu().numpy(), qb.cpu().numpy()


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
    PDP, APS, CTX = build_features(d, sub, pos)
    pdp_flat = PDP.reshape(n, -1); aps_flat = APS.reshape(n, -1)
    print(f"[sample] n={n}, 3-BS, ctx_dim={CTX.shape[1]} (pos2+rsrp3+mdelay3)")

    methods = ["단순보간 IDW(위치만)", "RAN Twin kNN(위치+실측)", "RAN Twin MLP(위치+실측)"]
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
            # context 표준화(관측 통계 기준)
            cmu, csd = CTX[obs].mean(0), CTX[obs].std(0) + 1e-9
            ctx_o = (CTX[obs] - cmu) / csd
            ctx_q = (CTX[qry] - cmu) / csd
            pdp_t, aps_t = PDP[qry], APS[qry]

            # floor: 전역평균
            pdp_m = PDP[obs].mean(0); pdp_m /= pdp_m.sum(-1, keepdims=True) + 1e-30
            aps_m = APS[obs].mean(0); aps_m /= aps_m.sum(-1, keepdims=True) + 1e-30
            floor_res[p].append(score(np.broadcast_to(pdp_m, (nq, N_TX, N_TAU)),
                                      np.broadcast_to(aps_m, (nq, N_TX, N_AOA)),
                                      pdp_t, aps_t, qloc))

            # (0) 단순보간 IDW (위치만)
            ph = idw_pos(pos_o, pdp_flat[obs], pos_q).reshape(-1, N_TX, N_TAU)
            ah = idw_pos(pos_o, aps_flat[obs], pos_q).reshape(-1, N_TX, N_AOA)
            ph = np.clip(ph, 0, None); ph /= ph.sum(-1, keepdims=True) + 1e-30
            ah = np.clip(ah, 0, None); ah /= ah.sum(-1, keepdims=True) + 1e-30
            res[methods[0]][p].append(score(ph, ah, pdp_t, aps_t, qloc))

            # (1) RAN Twin kNN (위치+실측 context)
            ph = knn_ctx(ctx_o, pdp_flat[obs], ctx_q).reshape(-1, N_TX, N_TAU)
            ah = knn_ctx(ctx_o, aps_flat[obs], ctx_q).reshape(-1, N_TX, N_AOA)
            ph = np.clip(ph, 0, None); ph /= ph.sum(-1, keepdims=True) + 1e-30
            ah = np.clip(ah, 0, None); ah /= ah.sum(-1, keepdims=True) + 1e-30
            res[methods[1]][p].append(score(ph, ah, pdp_t, aps_t, qloc))

            # (2) RAN Twin MLP (위치+실측 context)
            ph, ah = mlp_ctx(CTX[obs], PDP[obs], APS[obs], CTX[qry])
            res[methods[2]][p].append(score(ph, ah, pdp_t, aps_t, qloc))

            print(f"  p={p:.1f} s={s}  floor={floor_res[p][-1][2]:.3f}  "
                  f"IDW={res[methods[0]][p][-1][2]:.3f}  "
                  f"kNN={res[methods[1]][p][-1][2]:.3f}  "
                  f"MLP={res[methods[2]][p][-1][2]:.3f}")

    # ---------------- 시각화 ----------------
    colors = {methods[0]: "#9e9e9e", methods[1]: "#1b9e77", methods[2]: "#3a6fb0"}
    styles = {methods[0]: "--", methods[1]: "-", methods[2]: "-"}
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
            label="floor(전역평균)", zorder=1)
    ax.set_xlabel("관측 비율 (%)  →  관측 풍부", fontsize=12)
    ax.set_ylabel("추론 vs True 유사도  (1 = 완전 일치)", fontsize=12)
    ax.set_title("(a) 원 유사도 S", fontsize=12)
    ax.set_ylim(0.5, 1.0); ax.grid(alpha=0.3)
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
    ax2.axhline(0, color="#444444", ls=":", lw=1.8, label="floor 기준")
    ax2.set_xlabel("관측 비율 (%)  →  관측 풍부", fontsize=12)
    ax2.set_ylabel("Skill score  (S−floor)/(1−floor)", fontsize=12)
    ax2.set_title("(b) Skill score  (실측 활용 학습의 우위)", fontsize=12)
    ax2.grid(alpha=0.3)
    ax2.legend(loc="lower right", fontsize=10, framealpha=0.9)

    fig.suptitle("RAN Twin v2 : 위치+실측(RSRP·지연) 학습 vs 위치-only 단순보간  "
                 "(채널=PDP+재구성Cov, GHM 3-BS)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fpng = os.path.join(OUTDIR, f"P2C_v2_interp_curve_{STAMP}.png")
    fig.savefig(fpng, dpi=140)
    print(f"[save] {fpng}")

    fcsv = os.path.join(OUTDIR, f"P2C_v2_interp_{STAMP}.csv")
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

    print("\n=== 관측비율별 평균 S | skill ===")
    for ip, p in enumerate(RATIOS):
        fl = floor_mean[ip]
        Srow = "  ".join(f"{np.mean([v[2] for v in res[m][p]]):.3f}" for m in methods)
        skrow = "  ".join(
            f"{np.mean([(v[2]-fl)/(1-fl+1e-9) for v in res[m][p]]):+.3f}" for m in methods)
        print(f"{p*100:4.0f}%  floor={fl:.3f}  S=[{Srow}]  skill=[{skrow}]")


if __name__ == "__main__":
    main()
