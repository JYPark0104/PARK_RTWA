# -*- coding: utf-8 -*-
"""
P2G_BW_Approx_Check_2606v1.py
=============================
[단계 2-0] 저랭크 Bures-Wasserstein(BW) 근사식 수치 검증 스니펫.

목적
  메트릭의 공분산 항 d_Cov 를 계산하기 전에, "저랭크 인수분해 + nuclear-norm"
  근사가 full Bures 와 정말 일치하는지 작은 행렬(실데이터 R_UE, 16x16)로 먼저
  확인한다. 이게 맞아야 1024x1024 R_BS 로 확장하는 나머지가 의미를 가진다.

비교하는 세 가지 fidelity 항  F(A,B) = tr[(A^{1/2} B A^{1/2})^{1/2}]
  (1) F_full : full Bures. A^{1/2} B A^{1/2} 의 대칭 제곱근 trace (eigh 기반)
  (2) F_sym  : ||A^{1/2} B^{1/2}||_*   (각 행렬의 '대칭 제곱근' 곱의 nuclear norm)
               -> 이론상 (1)과 정확히 동일해야 함 (정답 형태)
  (3) F_fac  : ||B_i^H B_j||_* ,  B_i = U_i Λ_i^{1/2}  (P2A 방식)
               -> full-rank 에서 (2)와 같은지(=unitary 불변), 저랭크 절사 시
                  근사가 단조 수렴하는지 확인

판정
  - full-rank 상대오차 |F_x - F_full| / F_full < 1e-6 통과
  - 저랭크(r<M) 절사 시 r 증가에 따라 F_fac -> F_full 단조 수렴

안정화 (계획서 단계 2)
  Ã = A/tr(A) + εI  (ε≈1e-6),  d_BW^2 = tr(Ã_i)+tr(Ã_j) - 2 F

----------------------------------------------------------------------
실행 환경
  - Python   : 3.10.12
  - 실행 서버 : dclserver78 (twin_minji workspace)
  - 주요 라이브러리 : numpy 2.2.6 / scipy 1.15.3
입력 데이터(읽기 전용)
  - 251009_CCM_Collection (8 GB)/P1F_Marginal_CCM_Results/*.npz  (키: R_UE 16x16)
----------------------------------------------------------------------
"""

import os
import glob

import numpy as np

EPS = 1e-6                      # Ã = A/tr(A) + εI 안정화
RTOL_PASS = 1e-6               # full-rank 통과 기준
N_SAMPLE_RX = 8               # 검증에 쓸 RX(점) 표본 수
SEED = 42

WS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P1F_DIR = os.path.join(WS_ROOT, "251009_CCM_Collection (8 GB)",
                       "P1F_Marginal_CCM_Results")


# ============================================================
# 공통 헬퍼 (Hermitian PSD 전용, eigh 기반으로 수치 안정)
# ============================================================
def hermitize(A):
    """수치 오차로 깨진 Hermitian 대칭성 복원."""
    return 0.5 * (A + A.conj().T)


def stabilize(A):
    """Ã = A/tr(A) + εI. trace 정규화 + 대각 안정화."""
    A = hermitize(A)
    tr = np.trace(A).real
    return A / tr + EPS * np.eye(A.shape[0], dtype=A.dtype)


def psd_eigh(A):
    """Hermitian A 의 고유분해. 고유값은 음수 클램프(노이즈 바닥)."""
    w, U = np.linalg.eigh(hermitize(A))
    w = np.clip(w.real, 0.0, None)
    return w, U


def psd_sqrt(A):
    """대칭(주) 제곱근 A^{1/2} = U Λ^{1/2} U^H."""
    w, U = psd_eigh(A)
    return (U * np.sqrt(w)) @ U.conj().T


def nuclear_norm(M):
    """nuclear norm = 특이값 합."""
    return np.linalg.svd(M, compute_uv=False).sum()


# ============================================================
# 세 가지 fidelity
# ============================================================
def fidelity_full(A, B):
    """(1) F_full = tr[(A^{1/2} B A^{1/2})^{1/2}]  (eigh 기반)."""
    Ah = psd_sqrt(A)
    inner = hermitize(Ah @ B @ Ah)
    w, _ = psd_eigh(inner)
    return np.sqrt(w).sum()


def fidelity_symprod(A, B):
    """(2) F_sym = ||A^{1/2} B^{1/2}||_*  (정답 형태)."""
    return nuclear_norm(psd_sqrt(A) @ psd_sqrt(B))


def factor_topr(A, r):
    """B = U_r Λ_r^{1/2} (상위 r 고유쌍). r=None 이면 full-rank."""
    w, U = psd_eigh(A)
    if r is not None:
        idx = np.argsort(w)[::-1][:r]      # 큰 고유값 상위 r
        w, U = w[idx], U[:, idx]
    return U * np.sqrt(w)                   # (M, r)


def fidelity_factor(A, B, r=None):
    """(3) F_fac = ||B_i^H B_j||_* ,  B_i=U_iΛ_i^{1/2} (P2A 방식)."""
    Bi = factor_topr(A, r)
    Bj = factor_topr(B, r)
    return nuclear_norm(Bi.conj().T @ Bj)


# ============================================================
# 데이터 로드 (R_UE 16x16 표본)
# ============================================================
def load_r_ue_samples(n):
    files = sorted(glob.glob(os.path.join(P1F_DIR, "*.npz")))
    if not files:
        raise FileNotFoundError(f"P1F npz 없음: {P1F_DIR}")
    rng = np.random.default_rng(SEED)
    pick = rng.choice(len(files), size=min(n, len(files)), replace=False)
    mats, names = [], []
    for i in sorted(pick):
        d = np.load(files[i], allow_pickle=True)
        R = np.asarray(d["R_UE"]).astype(np.complex128)
        mats.append(R)
        names.append(os.path.basename(files[i]))
    print(f"[load] R_UE {mats[0].shape} x {len(mats)}개 표본")
    return mats, names


# ============================================================
# main
# ============================================================
def main():
    mats, names = load_r_ue_samples(N_SAMPLE_RX)
    M = mats[0].shape[0]
    As = [stabilize(R) for R in mats]
    ranks = [2, 4, 8, M // 2, M]          # 저랭크 -> full-rank 수렴 확인용
    ranks = sorted(set([r for r in ranks if 1 <= r <= M]))

    print("\n" + "=" * 96)
    print(f"[2-0] Bures fidelity 3-way 대조 (M={M}, ε={EPS}, "
          f"pass<|rel|<{RTOL_PASS})")
    print("=" * 96)
    header = (f"{'pair':>9} | {'F_full':>12} {'F_sym(2)':>12} "
              f"{'F_fac full(3)':>14} | {'rel_sym':>10} {'rel_fac':>10}")
    print(header)
    print("-" * len(header))

    max_rel_sym = 0.0
    max_rel_fac_full = 0.0
    pairs = []
    for a in range(len(As)):
        for b in range(a + 1, len(As)):
            pairs.append((a, b))

    for (a, b) in pairs:
        A, B = As[a], As[b]
        f_full = fidelity_full(A, B)
        f_sym = fidelity_symprod(A, B)
        f_fac = fidelity_factor(A, B, r=None)     # full-rank
        rel_sym = abs(f_sym - f_full) / f_full
        rel_fac = abs(f_fac - f_full) / f_full
        max_rel_sym = max(max_rel_sym, rel_sym)
        max_rel_fac_full = max(max_rel_fac_full, rel_fac)
        print(f"{a:>4}-{b:<4} | {f_full:>12.6f} {f_sym:>12.6f} "
              f"{f_fac:>14.6f} | {rel_sym:>10.2e} {rel_fac:>10.2e}")

    print("-" * len(header))
    print(f"[full-rank] max rel(F_sym vs F_full)      = {max_rel_sym:.3e}  "
          f"-> {'PASS' if max_rel_sym < RTOL_PASS else 'FAIL'}")
    print(f"[full-rank] max rel(F_fac vs F_full)      = {max_rel_fac_full:.3e}  "
          f"-> {'PASS' if max_rel_fac_full < RTOL_PASS else 'FAIL'}")

    # 저랭크 절사 수렴 (대표 쌍 1개)
    a, b = pairs[0]
    A, B = As[a], As[b]
    f_full = fidelity_full(A, B)
    print("\n" + "=" * 60)
    print(f"[저랭크 절사 수렴]  pair {a}-{b},  F_full={f_full:.6f}")
    print(f"{'rank r':>8} {'F_fac(r)':>12} {'rel_err':>12}")
    print("-" * 34)
    for r in ranks:
        fr = fidelity_factor(A, B, r=r)
        print(f"{r:>8} {fr:>12.6f} {abs(fr - f_full) / f_full:>12.2e}")

    # d_BW^2 도 한 번 확인 (trace + fidelity 결합)
    print("\n" + "=" * 60)
    print("[d_BW^2 = tr(Ãi)+tr(Ãj) - 2F]  (대표 쌍)")
    trA, trB = np.trace(A).real, np.trace(B).real
    dbw2 = trA + trB - 2 * f_full
    print(f"  tr(Ãi)={trA:.6f}, tr(Ãj)={trB:.6f}, "
          f"d_BW^2={dbw2:.6e}, d_Cov={np.sqrt(max(dbw2, 0)):.6e}")

    # 결론
    print("\n" + "=" * 96)
    if max_rel_sym < RTOL_PASS:
        print("[결론] F_sym(=||Ã^½ B̃^½||_*) 은 full Bures 와 일치 -> 정답 형태로 채택")
    else:
        print("[결론] F_sym 불일치(!) -> psd_sqrt/안정화 점검 필요")
    if max_rel_fac_full < RTOL_PASS:
        print("[결론] F_fac(=||Bᴴ B||_*) 도 full-rank 에서 일치 "
              "-> U 가 정방 unitary 라 nuclear norm 불변. P2A 방식 유효")
    else:
        print("[결론] F_fac 는 full-rank 에서도 불일치 -> 대칭제곱근 곱 형태로 확정")
    print("=" * 96)


if __name__ == "__main__":
    main()
