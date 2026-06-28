#!/usr/bin/env python3
"""
MATLAB 방식 Vectorization 검증 스크립트

수학적 표준 (MATLAB):
- vec(A): Column-major vectorization
- kron(A, B): Column-major 기준 Kronecker product
- R_AE = R_BS ⊗ R_UE (BS가 먼저)

검증 항목:
1. R_KM_sam 정확도
2. eps_d (고유값 오차)
3. eps_U (고유벡터 오차)
"""

import numpy as np
import tensorflow as tf
import scipy.linalg as la

print("="*80)
print(" MATLAB 방식 Vectorization 검증")
print("="*80)

# 작은 차원으로 테스트 (일반적인 행렬)
n_BS = 2  # TX
n_UE = 3  # RX
n_AE = n_BS * n_UE  # 6

print(f"\n설정: n_BS={n_BS}, n_UE={n_UE}, n_AE={n_AE}")

# ============================================================================
# Ground Truth 생성 (Exponential Correlation Model)
# ============================================================================
print("\n" + "="*80)
print(" 1. Ground Truth 생성 (ECM)")
print("="*80)

def generate_ecm(n, alpha):
    """Exponential Correlation Matrix (Hermitian)"""
    R = np.zeros((n, n), dtype=np.complex64)
    for i in range(n):
        for j in range(n):
            if i < j:
                R[i, j] = alpha ** (j - i)
            elif i > j:
                R[i, j] = np.conj(alpha ** (i - j))
            else:
                R[i, j] = 1.0
    return R

# ECM 생성
alpha_BS = 0.6 + 0.2j
alpha_UE = 0.7 + 0.1j

R_BS_tru = generate_ecm(n_BS, alpha_BS)
R_UE_tru = generate_ecm(n_UE, alpha_UE)

# Trace 정규화
R_BS_tru = R_BS_tru / np.trace(R_BS_tru) * n_BS
R_UE_tru = R_UE_tru / np.trace(R_UE_tru) * n_UE

print("\nR_BS_tru (Ground Truth):")
print(R_BS_tru)
print(f"Trace(R_BS): {np.trace(R_BS_tru)}")

print("\nR_UE_tru (Ground Truth):")
print(R_UE_tru)
print(f"Trace(R_UE): {np.trace(R_UE_tru)}")

# ============================================================================
# MATLAB 방식: kron(R_BS, R_UE)
# ============================================================================
print("\n" + "="*80)
print(" 2. Ground Truth R_AE = kron(R_BS, R_UE) [MATLAB]")
print("="*80)

# NumPy kron: MATLAB과 동일 (Column-major 기준)
R_AE_tru_numpy = np.kron(R_BS_tru, R_UE_tru)

# TensorFlow LinearOperatorKronecker: 검증 필요
R_BS_tru_tf = tf.constant(R_BS_tru)
R_UE_tru_tf = tf.constant(R_UE_tru)

R_AE_tru_tf = tf.linalg.LinearOperatorKronecker([
    tf.linalg.LinearOperatorFullMatrix(R_BS_tru_tf),
    tf.linalg.LinearOperatorFullMatrix(R_UE_tru_tf)
]).to_dense()

print(f"\nR_AE shape: {R_AE_tru_numpy.shape}")
print(f"Trace(R_AE): {np.trace(R_AE_tru_numpy)}")

# TF vs NumPy 비교
diff_tf_numpy = np.max(np.abs(R_AE_tru_tf.numpy() - R_AE_tru_numpy))
print(f"\n|TF - NumPy| max: {diff_tf_numpy:.2e}")
print(f"TF LinearOperatorKronecker == NumPy kron: {diff_tf_numpy < 1e-6}")

# Ground Truth 고유값 분해
lambda_AE_tru, U_AE_tru = np.linalg.eigh(R_AE_tru_numpy)
idx_AE_desc = np.argsort(np.real(lambda_AE_tru))[::-1]
lambda_AE_tru = np.real(lambda_AE_tru[idx_AE_desc])
U_AE_tru = U_AE_tru[:, idx_AE_desc]

lambda_BS_tru, U_BS_tru = np.linalg.eigh(R_BS_tru)
idx_BS_desc = np.argsort(np.real(lambda_BS_tru))[::-1]
lambda_BS_tru = np.real(lambda_BS_tru[idx_BS_desc])
U_BS_tru = U_BS_tru[:, idx_BS_desc]

lambda_UE_tru, U_UE_tru = np.linalg.eigh(R_UE_tru)
idx_UE_desc = np.argsort(np.real(lambda_UE_tru))[::-1]
lambda_UE_tru = np.real(lambda_UE_tru[idx_UE_desc])
U_UE_tru = U_UE_tru[:, idx_UE_desc]

print(f"\nλ_AE (desc): {lambda_AE_tru}")
print(f"λ_BS (desc): {lambda_BS_tru}")
print(f"λ_UE (desc): {lambda_UE_tru}")

# ============================================================================
# 크로네커 근사 방법 1: 현재 P1D (Row-major reshape)
# ============================================================================
print("\n" + "="*80)
print(" 3. 방법 1: 현재 P1D (Row-major reshape)")
print("="*80)

lambda_BS_tf = tf.constant(lambda_BS_tru, dtype=tf.float32)
lambda_UE_tf = tf.constant(lambda_UE_tru, dtype=tf.float32)
U_BS_tf = tf.constant(U_BS_tru, dtype=tf.complex64)
U_UE_tf = tf.constant(U_UE_tru, dtype=tf.complex64)

# 고유값 크로네커 곱 (Row-major)
lambda_outer = tf.tensordot(lambda_BS_tf, lambda_UE_tf, axes=0)  # [n_BS, n_UE]
lambda_KM_row = tf.reshape(lambda_outer, [-1])  # Row-major flatten

# 고유벡터 크로네커 곱 (Row-major)
U_outer_row = tf.einsum('ij,kl->ikjl', U_BS_tf, U_UE_tf)  # [n_BS, n_UE, n_BS, n_UE]
U_KM_row = tf.reshape(U_outer_row, [n_AE, n_AE])  # Row-major flatten

# R_KM 재구성
R_KM_row = U_KM_row @ tf.linalg.diag(tf.cast(lambda_KM_row, tf.complex64)) @ tf.linalg.adjoint(U_KM_row)

print(f"\nλ_KM (row-major): {lambda_KM_row.numpy()}")

# 정렬
idx_KM_row_desc = tf.argsort(lambda_KM_row, direction='DESCENDING')
lambda_KM_row_desc = tf.gather(lambda_KM_row, idx_KM_row_desc).numpy()
U_KM_row_desc = tf.gather(U_KM_row, idx_KM_row_desc, axis=1).numpy()

print(f"λ_KM_desc (row): {lambda_KM_row_desc}")

# R_KM 오차
R_KM_err_row = np.linalg.norm(R_AE_tru_numpy - R_KM_row.numpy(), 'fro') / np.linalg.norm(R_AE_tru_numpy, 'fro')
print(f"\nR_KM 상대 오차 (Frobenius): {R_KM_err_row:.6f}")

# ============================================================================
# 크로네커 근사 방법 2: MATLAB 방식 (Column-major)
# ============================================================================
print("\n" + "="*80)
print(" 4. 방법 2: MATLAB 방식 (Column-major transpose)")
print("="*80)

# 고유값 크로네커 곱 (Column-major)
lambda_outer_T = tf.transpose(lambda_outer)  # [n_UE, n_BS]
lambda_KM_col = tf.reshape(lambda_outer_T, [-1])  # Column-major flatten

# 고유벡터 크로네커 곱 (Column-major)
# einsum: 'ij,kl->ikjl' → [n_BS, n_UE, n_BS, n_UE]
# Column-major: [n_BS, n_BS, n_UE, n_UE] 순서로 변경
U_outer_col = tf.transpose(U_outer_row, [0, 2, 1, 3])  # [n_BS, n_BS, n_UE, n_UE]
U_KM_col = tf.reshape(U_outer_col, [n_AE, n_AE])

# R_KM 재구성
R_KM_col = U_KM_col @ tf.linalg.diag(tf.cast(lambda_KM_col, tf.complex64)) @ tf.linalg.adjoint(U_KM_col)

print(f"\nλ_KM (col-major): {lambda_KM_col.numpy()}")

# 정렬
idx_KM_col_desc = tf.argsort(lambda_KM_col, direction='DESCENDING')
lambda_KM_col_desc = tf.gather(lambda_KM_col, idx_KM_col_desc).numpy()
U_KM_col_desc = tf.gather(U_KM_col, idx_KM_col_desc, axis=1).numpy()

print(f"λ_KM_desc (col): {lambda_KM_col_desc}")

# R_KM 오차
R_KM_err_col = np.linalg.norm(R_AE_tru_numpy - R_KM_col.numpy(), 'fro') / np.linalg.norm(R_AE_tru_numpy, 'fro')
print(f"\nR_KM 상대 오차 (Frobenius): {R_KM_err_col:.6f}")

# ============================================================================
# 크로네커 근사 방법 3: 직접 kron 사용 (참고)
# ============================================================================
print("\n" + "="*80)
print(" 5. 방법 3: 직접 NumPy kron (참고)")
print("="*80)

# NumPy kron으로 직접 계산
R_BS_approx = U_BS_tru @ np.diag(lambda_BS_tru) @ U_BS_tru.conj().T
R_UE_approx = U_UE_tru @ np.diag(lambda_UE_tru) @ U_UE_tru.conj().T
R_KM_numpy = np.kron(R_BS_approx, R_UE_approx)

R_KM_err_numpy = np.linalg.norm(R_AE_tru_numpy - R_KM_numpy, 'fro') / np.linalg.norm(R_AE_tru_numpy, 'fro')
print(f"\nR_KM 상대 오차 (NumPy kron): {R_KM_err_numpy:.6f}")

# ============================================================================
# eps_d 계산
# ============================================================================
print("\n" + "="*80)
print(" 6. eps_d 계산 (고유값 오차)")
print("="*80)

def compute_eps_d(lambda_ref, lambda_approx):
    return np.linalg.norm(lambda_ref - lambda_approx) / np.linalg.norm(lambda_ref)

eps_d_row = compute_eps_d(lambda_AE_tru, lambda_KM_row_desc)
eps_d_col = compute_eps_d(lambda_AE_tru, lambda_KM_col_desc)

print(f"\neps_d (Row-major):    {eps_d_row:.6f}")
print(f"eps_d (Column-major): {eps_d_col:.6f}")

# ============================================================================
# eps_U 계산
# ============================================================================
print("\n" + "="*80)
print(" 7. eps_U 계산 (고유벡터 오차, Row-max)")
print("="*80)

def compute_eps_U(U_ref, U_approx, lambda_ref):
    # Inner product matrix
    inner_prod_mat = np.abs(U_ref.conj().T @ U_approx)
    
    # Row-max
    max_per_row = np.max(inner_prod_mat, axis=1)
    
    # Weighted by eigenvalues
    weights = lambda_ref / np.sum(lambda_ref)
    weighted_avg = np.sum(weights * max_per_row)
    
    eps_U = 1.0 - weighted_avg
    return eps_U, inner_prod_mat

eps_U_row, inner_row = compute_eps_U(U_AE_tru, U_KM_row_desc, lambda_AE_tru)
eps_U_col, inner_col = compute_eps_U(U_AE_tru, U_KM_col_desc, lambda_AE_tru)

print(f"\neps_U (Row-major):    {eps_U_row:.6f}")
print(f"eps_U (Column-major): {eps_U_col:.6f}")

print("\n대각 내적 (일치도):")
print(f"Row-major:    {np.diag(inner_row)}")
print(f"Column-major: {np.diag(inner_col)}")

# ============================================================================
# 종합 결과
# ============================================================================
print("\n" + "="*80)
print(" 종합 결과")
print("="*80)

print("\nR_KM 프로베니우스 오차:")
print(f"  Row-major:    {R_KM_err_row:.6f}")
print(f"  Column-major: {R_KM_err_col:.6f}")
print(f"  NumPy kron:   {R_KM_err_numpy:.6f}")

print("\neps_d (고유값 오차):")
print(f"  Row-major:    {eps_d_row:.6f}")
print(f"  Column-major: {eps_d_col:.6f}")

print("\neps_U (고유벡터 오차):")
print(f"  Row-major:    {eps_U_row:.6f}")
print(f"  Column-major: {eps_U_col:.6f}")

print("\n" + "="*80)
print(" 결론")
print("="*80)

if R_KM_err_col < R_KM_err_row:
    print("\n✓ Column-major가 더 정확합니다!")
    print("  → P1D 코드를 Column-major로 수정 필요")
elif R_KM_err_row < R_KM_err_col:
    print("\n✓ Row-major가 더 정확합니다!")
    print("  → 현재 P1D 구현이 올바름")
else:
    print("\n✓ 두 방법이 동일합니다.")
    print("  → 다른 문제 원인 탐색 필요")

print(f"\nLinearOperatorKronecker는 {'Column-major' if diff_tf_numpy < 1e-6 else 'Row-major'} 방식입니다.")

# ============================================================================
# 상세 분석: 고유벡터 내적 행렬
# ============================================================================
print("\n" + "="*80)
print(" 상세 분석: 고유벡터 내적 행렬")
print("="*80)

print("\n|⟨U_AE, U_KM_row⟩|:")
print(inner_row)

print("\n|⟨U_AE, U_KM_col⟩|:")
print(inner_col)

print("\n" + "="*80)



