# -*- coding: utf-8 -*-
"""
P2C_RANTwin_Interpolator_2606v1.py
==================================
[ 목적 ]
P2B 에서 입증한 '메트릭 거리 <-> 물리거리 단조상관'(채널이 공간적으로 상관)을
근거로, **부분 관측(observed) 으로 학습 -> 미관측(held-out=정답) 채널을 추론**하는
보간기(interpolator)를 만들고, 관측비율(observation ratio)에 따른 추론 정확도를
도출한다. (사용자 목표 그림: x=관측비율%, y=추론 vs True 유사도)

[ 평가 프로토콜 (train/held-out) ]
  - 관측 RX(observed) = 학습용. 미관측 RX(held-out) = 정답(test).
  - 보간기는 관측 RX 의 (위치, 채널)로 학습 -> 미관측 RX 위치에서 채널 추론.
  - 추론값 vs True 의 유사도(PDP + Cov)를 held-out 평균 -> 곡선 1점.
  - 관측비율 10~90 % sweep, 시드 4회 반복 -> 점(scatter)+평균선.

[ 채널 표현 (그림의 'PADP + Covariance') ]
  - BS(TX) 별 : PDP(32 bin, 시간) + APS(48 bin, 방위).
  - SISO(R=1) 이므로 공간 공분산은 APS 로부터 '가상 ULA(M=16)' 재구성 (trace=1).
    -> 두 공분산 비교는 Bures-Wasserstein(BW). (P2A 방식과 동일 철학)

[ 보간기 3종 ]
  (0) 단순 보간(학습 없음) : IDW (거리 역수^2 가중)  -- baseline(회색 점선)
  (1) RAN Twin / GP-Kriging: 거리-유사도 곡선에서 상관거리 ell 학습 -> 크리깅 가중
  (2) RAN Twin / MLP       : 위치(x,y) -> 채널분포 회귀(블록 softmax)

  핵심 연결고리: GP 의 커널 C(d)=exp(-d/ell) 의 ell 을 '관측셋에서 측정한
  채널유사도-거리 감쇠'에 적합(fit) -> 우리가 도출한 상관결과가 보간 커널이 됨.

----------------------------------------------------------------------
실행 환경
  - Python 3.10.12 / dclcom61 / numpy·scipy·torch(2.11+cu128, MLP)·matplotlib
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

# 한글 폰트
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
N_SAMPLE = 900          # 공간 보간용 RX 샘플 수
MIN_PATHS = 10
N_TAU = 32              # PDP bin
N_AOA = 48              # APS bin
M_ULA = 16              # 가상 ULA 안테나 수 (공분산 차원)
RATIOS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
SEEDS = [0, 1, 2, 3]
SCORE_CAP = 400         # held-out 채점 최대 개수(속도)
SEED0 = 0

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
    """반환: PDP(n,3,N_TAU), APS(n,3,N_AOA) (행 정규화된 확률분포)."""
    n = len(sub)
    aoa_edges = np.linspace(-180, 180, N_AOA + 1)
    PDP = np.zeros((n, N_TX, N_TAU))
    APS = np.zeros((n, N_TX, N_AOA))

    # TX별 tau 스케일
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
    """가상 반파장 ULA 조향행렬 A (M, N_AOA), 빈 중심각 기준."""
    centers = -180 + 360 * (np.arange(N_AOA) + 0.5) / N_AOA   # deg
    th = np.deg2rad(centers)
    m = np.arange(M_ULA)[:, None]
    return np.exp(1j * np.pi * m * np.sin(th)[None, :])        # (M, B)


A_STEER = steering_matrix()


def aps_to_cov(aps):
    """APS(확률분포) -> trace=1 정규화 가상 ULA 공분산 (M,M)."""
    R = (A_STEER * aps[None, :]) @ A_STEER.conj().T              # (M,M)
    tr = np.real(np.trace(R))
    return R / (tr + 1e-30)


def bw2_trace1(Ri, Rj):
    """trace=1 두 공분산의 Bures-Wasserstein^2 = 2 - 2*sum sqrt(eig(Ri^.5 Rj Ri^.5))."""
    wi, Ui = np.linalg.eigh(Ri)
    wi = np.clip(wi, 0, None)
    Rh = (Ui * np.sqrt(wi)) @ Ui.conj().T
    Minner = Rh @ Rj @ Rh
    ev = np.linalg.eigvalsh((Minner + Minner.conj().T) / 2)
    ev = np.clip(ev, 0, None)
    return float(np.clip(2.0 - 2.0 * np.sqrt(ev).sum(), 0, 2))


def pdp_w1(p, q):
    """PDP(정규분포) 1D Wasserstein, bin축 [0,1] 정규화 -> [0,1]."""
    c = np.cumsum(p) - np.cumsum(q)
    return float(np.abs(c).sum() / N_TAU)


# ---------------------------------------------------------------- 보간기
def idw_predict(pos_o, feat_o, pos_q, power=2.0, k=20):
    """단순 IDW: 가까운 k개 관측의 거리역수^power 가중평균. feat_o (no,F)."""
    d = np.sqrt(((pos_q[:, None] - pos_o[None]) ** 2).sum(-1))   # (nq,no)
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
    """관측셋에서 '채널 유사도(APS 코사인) vs 거리' 감쇠를 적합 -> 상관거리 ell.
    S(d)=exp(-d/ell). 우리가 도출한 거리-유사도 관계를 커널로 변환."""
    no = pos_o.shape[0]
    m = min(no, 250)
    ridx = np.random.default_rng(0).choice(no, m, replace=False)
    P = pos_o[ridx]; F = aps_o[ridx]
    d = np.sqrt(((P[:, None] - P[None]) ** 2).sum(-1))
    Fn = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-30)
    S = Fn @ Fn.T                       # 코사인 유사도 [0,1]
    iu = np.triu_indices(m, 1)
    dd, ss = d[iu], np.clip(S[iu], 1e-6, 1 - 1e-6)
    # log S = -d/ell  -> 선형회귀 기울기
    y = -np.log(ss)
    slope = (dd * y).sum() / ((dd * dd).sum() + 1e-30)
    ell = 1.0 / (slope + 1e-9)
    return float(np.clip(ell, 20.0, 2000.0))


def kriging_predict(pos_o, feat_o, pos_q, ell):
    """단순 크리깅(상관커널 exp(-d/ell)) 가중. 음수가중 clip 후 재정규화."""
    no = pos_o.shape[0]
    Do = np.sqrt(((pos_o[:, None] - pos_o[None]) ** 2).sum(-1))
    C = np.exp(-Do / ell) + 1e-6 * np.eye(no)
    Cinv = np.linalg.inv(C)
    dq = np.sqrt(((pos_q[:, None] - pos_o[None]) ** 2).sum(-1))   # (nq,no)
    c0 = np.exp(-dq / ell)
    W = c0 @ Cinv                                                 # (nq,no)
    W = np.clip(W, 0, None)
    W /= W.sum(1, keepdims=True) + 1e-30
    return W @ feat_o


class MLP(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, out_dim))

    def forward(self, x):
        return self.net(x)


def mlp_predict(pos_o, pdp_o, aps_o, pos_q, epochs=300):
    """위치->(PDP 3블록 + APS 3블록) 회귀. 블록별 softmax 로 분포 유지."""
    mu, sd = pos_o.mean(0), pos_o.std(0) + 1e-9
    Xo = torch.tensor((pos_o - mu) / sd, dtype=torch.float32, device=DEVICE)
    Xq = torch.tensor((pos_q - mu) / sd, dtype=torch.float32, device=DEVICE)
    # 타깃: (n, 3*N_TAU + 3*N_AOA)
    Yp = torch.tensor(pdp_o.reshape(pos_o.shape[0], -1), dtype=torch.float32, device=DEVICE)
    Ya = torch.tensor(aps_o.reshape(pos_o.shape[0], -1), dtype=torch.float32, device=DEVICE)
    out_dim = N_TX * N_TAU + N_TX * N_AOA
    net = MLP(out_dim).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)

    def split(logits):
        a = logits[:, :N_TX * N_TAU].reshape(-1, N_TX, N_TAU)
        b = logits[:, N_TX * N_TAU:].reshape(-1, N_TX, N_AOA)
        return torch.softmax(a, -1), torch.softmax(b, -1)

    Yp3 = Yp.reshape(-1, N_TX, N_TAU); Ya3 = Ya.reshape(-1, N_TX, N_AOA)
    for _ in range(epochs):
        opt.zero_grad()
        pa, pb = split(net(Xo))
        loss = ((pa - Yp3) ** 2).mean() + ((pb - Ya3) ** 2).mean()
        loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        qa, qb = split(net(Xq))
    return qa.cpu().numpy(), qb.cpu().numpy()    # (nq,3,32),(nq,3,48)


# ---------------------------------------------------------------- 채점
def score(pdp_hat, aps_hat, pdp_t, aps_t, qsel):
    """held-out(qsel) 평균 : pdp_sim, cov_sim, S. 입력 (nq,3,·)."""
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
    print(f"[sample] n={n}, TX={N_TX}, ULA M={M_ULA}")

    PDP, APS = build_features(d, sub)     # (n,3,32),(n,3,48)
    pdp_flat = PDP.reshape(n, -1)
    aps_flat = APS.reshape(n, -1)

    methods = ["IDW(단순보간)", "GP-Kriging(RAN Twin)", "MLP(RAN Twin)"]
    res = {m: {p: [] for p in RATIOS} for m in methods}
    # floor(전역평균, 공간정보 無) : skill score 기준선
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
            # 채점 인덱스를 query 로컬 인덱스로 매핑
            qmap = {g: l for l, g in enumerate(qry)}
            qloc = np.array([qmap[g] for g in qsel])

            pos_o, pos_q = pos[obs], pos[qry]
            pdp_t = PDP[qry]; aps_t = APS[qry]
            nq = len(qry)

            # (floor) 전역평균 예측 : 모든 미관측에 관측 평균 채널을 그대로 -> 공간정보 無
            pdp_m = PDP[obs].mean(0); pdp_m /= pdp_m.sum(-1, keepdims=True) + 1e-30
            aps_m = APS[obs].mean(0); aps_m /= aps_m.sum(-1, keepdims=True) + 1e-30
            ph_f = np.broadcast_to(pdp_m, (nq, N_TX, N_TAU))
            ah_f = np.broadcast_to(aps_m, (nq, N_TX, N_AOA))
            floor_res[p].append(score(ph_f, ah_f, pdp_t, aps_t, qloc))

            # (0) IDW
            ph = idw_predict(pos_o, pdp_flat[obs], pos_q).reshape(-1, N_TX, N_TAU)
            ah = idw_predict(pos_o, aps_flat[obs], pos_q).reshape(-1, N_TX, N_AOA)
            ah = np.clip(ah, 0, None); ah /= ah.sum(-1, keepdims=True) + 1e-30
            ph = np.clip(ph, 0, None); ph /= ph.sum(-1, keepdims=True) + 1e-30
            res[methods[0]][p].append(score(ph, ah, pdp_t, aps_t, qloc))

            # (1) GP-Kriging (ell 학습)
            ell = fit_corr_length(pos_o, aps_flat[obs])
            ph = kriging_predict(pos_o, pdp_flat[obs], pos_q, ell).reshape(-1, N_TX, N_TAU)
            ah = kriging_predict(pos_o, aps_flat[obs], pos_q, ell).reshape(-1, N_TX, N_AOA)
            ah = np.clip(ah, 0, None); ah /= ah.sum(-1, keepdims=True) + 1e-30
            ph = np.clip(ph, 0, None); ph /= ph.sum(-1, keepdims=True) + 1e-30
            res[methods[1]][p].append(score(ph, ah, pdp_t, aps_t, qloc))

            # (2) MLP
            ph, ah = mlp_predict(pos_o, PDP[obs], APS[obs], pos_q)
            res[methods[2]][p].append(score(ph, ah, pdp_t, aps_t, qloc))

            print(f"  p={p:.1f} seed={s}  ell={ell:6.1f}  "
                  f"IDW={res[methods[0]][p][-1][2]:.3f}  "
                  f"GP={res[methods[1]][p][-1][2]:.3f}  "
                  f"MLP={res[methods[2]][p][-1][2]:.3f}")

    # ---------------------------------------------------------------- 시각화
    colors = {"IDW(단순보간)": "#9e9e9e",
              "GP-Kriging(RAN Twin)": "#1b9e77",
              "MLP(RAN Twin)": "#3a6fb0"}
    styles = {"IDW(단순보간)": "--",
              "GP-Kriging(RAN Twin)": "-",
              "MLP(RAN Twin)": "-"}

    xpct = np.array(RATIOS) * 100
    floor_mean = np.array([np.mean([v[2] for v in floor_res[p]]) for p in RATIOS])

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(17, 6.4))

    # ---- (좌) 원 유사도 S ----
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
    ax.set_title("(a) 원 유사도 S  (바닥 ~0.8 포화)", fontsize=12)
    ax.set_ylim(0.5, 1.0)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=10, framealpha=0.9)

    # ---- (우) Skill score = (S - floor)/(1 - floor) ----
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
    ax2.set_title("(b) Skill score  (바닥 대비 정규화 → 격차 가시화)", fontsize=12)
    ax2.grid(alpha=0.3)
    ax2.legend(loc="lower right", fontsize=10, framealpha=0.9)

    fig.suptitle("RAN Twin 보간 : 부분관측 학습 → 미관측 채널 추론  "
                 "(채널 = PDP + 재구성 Covariance, GHM Twin 3-BS)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fpng = os.path.join(OUTDIR, f"P2C_interp_curve_{STAMP}.png")
    fig.savefig(fpng, dpi=140)
    print(f"[save] {fpng}")

    # ---------------------------------------------------------------- CSV
    fcsv = os.path.join(OUTDIR, f"P2C_interp_{STAMP}.csv")
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
    print("ratio   floor  " + "  ".join(f"{m.split('(')[0]:>11s}" for m in methods))
    for ip, p in enumerate(RATIOS):
        fl = floor_mean[ip]
        Srow = "  ".join(f"{np.mean([v[2] for v in res[m][p]]):.3f}" for m in methods)
        skrow = "  ".join(
            f"{np.mean([(v[2]-fl)/(1-fl+1e-9) for v in res[m][p]]):+.3f}"
            for m in methods)
        print(f"{p*100:4.0f}%  {fl:.3f}  S=[{Srow}]  skill=[{skrow}]")


if __name__ == "__main__":
    main()
