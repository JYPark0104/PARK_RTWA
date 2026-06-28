# -*- coding: utf-8 -*-
"""
P2G_dCov_RankCheck_2606v1.py
============================
[단계 2 사전검증 / d_Cov 단위테스트]
저랭크 r 이 d_Cov(Bures-Wasserstein) 에 주는 '실제 오차'를 측정한다.

핵심 논리
  '꼬리 에너지 3.3%' != 'd_Cov 오차 3.3%'. BW 의 교차항 ||S_i S_j||_* 에서 작은
  고유값은 sqrt(λ)로 더 작아지고 정렬도 약해 기여가 미미하다. 따라서 에너지 비율이
  아니라 d_Cov 자체의 상대오차로 r 충분성을 판정한다.

방법
  1. 점 테이블에서 꼬리(cov_tail) 큰 점 K개(rows) + 참조점 J개(cols) 선택.
  2. 그 점들의 원본 R_BS(1024x1024)를 다시 로드 -> full eigh (r=M) 로 정확한 인수.
  3. d_Cov 를 (a) full  (b) r=32(점 테이블 cov_factor) 두 방식으로 계산.
     d_BW^2(i,j) = tr_i + tr_j - 2||B_i^H B_j||_*,  d_Cov=sqrt(max(.,0))
     (||B_i^H B_j||_* = full Bures fidelity 임은 P2G_BW_Approx_Check 에서 검증)
  4. d_Cov 상대오차 통계 -> 판정.

판정 기준
  rel(d_Cov) max < 1e-3  -> r=32 확정(진행). 그 이상이면 r 상향 재빌드 권장.

----------------------------------------------------------------------
실행 환경
  - Python 3.10.12 / torch 2.12.0+cu130 / numpy 2.2.6
  - 서버 dclserver78, NVIDIA H100 NVL 95GB x 2  (계산은 모든 GPU 유틸 사용)
입력(읽기 전용)
  - 점 테이블 npz : P2G_PointTable_Results/P2G_PointTable_*.npz
  - 원본 R_BS    : 251009_CCM_Collection (8 GB)/P1F_Marginal_CCM_Results/*.npz
----------------------------------------------------------------------
"""

import os
import re
import glob
import datetime

import numpy as np
import torch

from P2G_gpu_utils_2606v1 import (get_devices, warmup_linalg,
                                  batched_topr_eigh_factor)

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
WS_ROOT   = os.path.dirname(BASE_DIR)
DATA_ROOT = os.path.join(WS_ROOT, "251009_CCM_Collection (8 GB)")
P1F_DIR   = os.path.join(DATA_ROOT, "P1F_Marginal_CCM_Results")
PT_DIR    = os.path.join(BASE_DIR, "P2G_PointTable_Results")

K_ROWS  = 5         # 꼬리 큰 점(행) 수
J_COLS  = 40        # 참조점(열) 수
EPS     = 1e-6
SEED    = 42
PASS_REL = 1e-3     # d_Cov 상대오차 통과 기준


def latest_point_table():
    fs = sorted(glob.glob(os.path.join(PT_DIR, "P2G_PointTable_*.npz")))
    if not fs:
        raise FileNotFoundError("점 테이블 npz 없음. 먼저 빌더 실행.")
    return fs[-1]


def fidelity_block(B_rows, B_cols, device):
    """F(i,j) = ||B_i^H B_j||_*  블록 (nr x nc). B_*: (n,M,r) complex tensor."""
    nr = B_rows.shape[0]
    Bc = B_cols.to(device)
    F = torch.empty((nr, B_cols.shape[0]), dtype=torch.float64, device=device)
    for i in range(nr):
        Bi = B_rows[i].to(device)                     # (M, r_i)
        G = torch.einsum("mr,jms->jrs", Bi.conj(), Bc)  # (nc, r_i, r)
        F[i] = torch.linalg.svdvals(G).sum(-1).double()
    return F.cpu().numpy()


def main():
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    devices = get_devices()
    dev0 = devices[0]
    warmup_linalg(devices)
    rng = np.random.default_rng(SEED)

    ptf = latest_point_table()
    d = np.load(ptf, allow_pickle=True)
    cov32 = d["cov_factor"]            # (N, M, 32)
    tr = d["cov_trace"]               # (N,)
    tail = d["cov_tail"]
    rx_idx = d["rx_idx"]
    N, M, r32 = cov32.shape
    print(f"[env] devices={[str(x) for x in devices]}")
    print(f"[load] {os.path.basename(ptf)}  N={N}, M={M}, r32={r32}")
    print(f"[tail] mean={tail.mean():.3e}, max={tail.max():.3e} "
          f"(점 {int(np.argmax(tail))})")

    # 행: 꼬리 큰 K개 / 열: 참조 J개(꼬리 큰 점 포함 + 랜덤)
    rows = np.argsort(tail)[::-1][:K_ROWS]
    rand = rng.choice(N, size=min(J_COLS, N), replace=False)
    cols = np.unique(np.concatenate([rows, rand]))
    sample = np.unique(np.concatenate([rows, cols]))
    print(f"[sel] rows(꼬리top{K_ROWS})={rows.tolist()}")
    print(f"[sel] cols={len(cols)}개, full eigh 대상 sample={len(sample)}개")
    print(f"[sel] rows tail={tail[rows].round(4).tolist()}")

    # 원본 R_BS 재로드 -> full eigh (r=M)  (모든 GPU 유틸)
    rx_file = {}
    for f in glob.glob(os.path.join(P1F_DIR, "*.npz")):
        m = re.search(r"RX(\d+)_", os.path.basename(f))
        if m:
            rx_file[int(m.group(1))] = f

    def load_fn(k):
        e = np.load(rx_file[int(rx_idx[sample[k]])], allow_pickle=True)
        return np.asarray(e["R_BS"]).astype(np.complex64)

    print(f"[full] sample {len(sample)}개 원본 R_BS full eigh(r={M}) ...")
    Bfull_s, tr_s, tail_s = batched_topr_eigh_factor(
        load_fn, len(sample), M, M, EPS, devices, chunk=16, log=print)
    pos_in_sample = {int(s): k for k, s in enumerate(sample)}

    # 행/열 인덱스를 sample 로컬로
    row_loc = np.array([pos_in_sample[int(i)] for i in rows])
    col_loc = np.array([pos_in_sample[int(j)] for j in cols])

    Bfull_t = torch.from_numpy(Bfull_s)          # (n_s, M, M) 내림차순
    tr_rows = tr[rows][:, None]
    tr_cols = tr[cols][None, :]

    # 기준: full-rank d_Cov
    F_full = fidelity_block(Bfull_t[row_loc], Bfull_t[col_loc], dev0)
    dcov_full = np.sqrt(np.clip(tr_rows + tr_cols - 2.0 * F_full, 0, None))
    same = (rows[:, None] == cols[None, :])
    valid = (~same) & (dcov_full > 1e-6)
    print("\n" + "=" * 78)
    print(f"[기준] d_Cov_full[min/med/max] = {dcov_full[valid].min():.4f}/"
          f"{np.median(dcov_full[valid]):.4f}/{dcov_full[valid].max():.4f} "
          f"(유효 {valid.sum()}쌍)")

    # 교차검증: 점테이블 cov32 로 만든 r=32 와, full 절사 r=32 가 일치하는지
    B32_t = torch.from_numpy(cov32[sample])
    F_32pt = fidelity_block(B32_t[row_loc], B32_t[col_loc], dev0)
    dcov_32pt = np.sqrt(np.clip(tr_rows + tr_cols - 2.0 * F_32pt, 0, None))
    F_32tr = fidelity_block(Bfull_t[row_loc][:, :, :r32],
                            Bfull_t[col_loc][:, :, :r32], dev0)
    dcov_32tr = np.sqrt(np.clip(tr_rows + tr_cols - 2.0 * F_32tr, 0, None))
    xcheck = np.abs(dcov_32pt - dcov_32tr)[valid].max()
    print(f"[교차검증] 점테이블 r32 vs full절사 r32 d_Cov 최대차={xcheck:.3e} "
          f"({'일치' if xcheck < 1e-4 else '불일치(부호/위상 주의)'})")

    # r 스윕: full 절사로 수렴 곡선
    r_list = [r for r in [16, 32, 48, 64, 96, 128, 192, 256, 384, 512] if r <= M]
    print("-" * 78)
    print(f"[r 스윕] d_Cov 상대오차 (vs full-rank)   PASS_REL={PASS_REL}")
    print(f"{'r':>6} {'mem(MB)':>9} {'rel_med':>11} {'rel_max':>11} "
          f"{'rel_p95':>11} {'verdict':>8}")
    sweep = []
    for r in r_list:
        Fr = fidelity_block(Bfull_t[row_loc][:, :, :r],
                            Bfull_t[col_loc][:, :, :r], dev0)
        dcov_r = np.sqrt(np.clip(tr_rows + tr_cols - 2.0 * Fr, 0, None))
        rel = np.abs(dcov_r - dcov_full)[valid] / dcov_full[valid]
        mem = N * M * r * 8 / 1e6
        vv = "PASS" if rel.max() < PASS_REL else ("OK<1%" if rel.max() < 1e-2 else "")
        print(f"{r:>6} {mem:>9.0f} {np.median(rel):>11.3e} {rel.max():>11.3e} "
              f"{np.percentile(rel,95):>11.3e} {vv:>8}")
        sweep.append((r, mem, float(np.median(rel)), float(rel.max()),
                      float(np.percentile(rel, 95))))
    print("=" * 78)

    # 권고 r: rel_max < 1e-2 만족하는 최소 r (없으면 최대)
    cand = [s for s in sweep if s[3] < 1e-2]
    rec = cand[0][0] if cand else sweep[-1][0]
    cand_strict = [s for s in sweep if s[3] < PASS_REL]
    rec_strict = cand_strict[0][0] if cand_strict else None
    print(f"[권고] rel_max<1e-2 최소 r = {rec}, "
          f"rel_max<{PASS_REL} 최소 r = {rec_strict}")

    os.makedirs(PT_DIR, exist_ok=True)
    rep = os.path.join(PT_DIR, f"P2G_dCov_rankcheck_{stamp}.log")
    with open(rep, "w", encoding="utf-8") as fp:
        fp.write(f"point_table={os.path.basename(ptf)}\nN={N} M={M}\n")
        fp.write(f"rows(tail top{K_ROWS})={rows.tolist()} tail={tail[rows].tolist()}\n")
        fp.write(f"valid_pairs={int(valid.sum())}\n")
        fp.write(f"dCov_full min/med/max={dcov_full[valid].min():.4f}/"
                 f"{np.median(dcov_full[valid]):.4f}/{dcov_full[valid].max():.4f}\n")
        fp.write(f"xcheck_pt_vs_trunc_r32={xcheck:.3e}\n")
        fp.write("r,mem_MB,rel_med,rel_max,rel_p95\n")
        for s in sweep:
            fp.write(f"{s[0]},{s[1]:.0f},{s[2]:.3e},{s[3]:.3e},{s[4]:.3e}\n")
        fp.write(f"recommend_rel_max<1e-2: r={rec}\n")
        fp.write(f"recommend_rel_max<{PASS_REL}: r={rec_strict}\n")
    print(f"[save] {rep}")


if __name__ == "__main__":
    main()
