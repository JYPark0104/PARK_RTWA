#!/usr/bin/env python3
"""
EVD 크로네커 관계식 검증

목적: R_KM = R_BS ⊗ R_UE 에서 EVD 관계식들이 성립하는지 검증
- U_KM = U_BS ⊗ U_UE
- D_KM = D_BS ⊗ D_UE  
- diag(D_KM) = diag(D_BS) ⊗ diag(D_UE)

검증 방법: 실제 공분산 행렬들로 EVD 수행 후 크로네커 관계식 확인
"""

import numpy as np
import tensorflow as tf

# ============================================================================
# 검증용 TensorFlow MATLAB 표준 메서드들 (기존 검증 완료)
# ============================================================================

@tf.function(jit_compile=True)
def vec_mat_py_tf(H_tf):
    """MATLAB vec(H) 구현 (TensorFlow Only, JIT 최적화)"""
    if len(H_tf.shape) == 2:
        return tf.reshape(tf.transpose(H_tf), [-1])
    elif len(H_tf.shape) == 3:
        return tf.reshape(tf.transpose(H_tf, perm=[0, 2, 1]), [tf.shape(H_tf)[0], -1])
    else:
        raise ValueError(f"Unsupported shape: {H_tf.shape}")

@tf.function(jit_compile=True)
def kron_mat_py_tf(A_tf, B_tf):
    """MATLAB kron(A,B) 구현 (TensorFlow Only - einsum & reshape, JIT 최적화)"""
    # einsum으로 4D 텐서 생성: A[i,j] * B[k,l] -> [i,k,j,l]  
    kron_4d = tf.einsum('ij,kl->ikjl', A_tf, B_tf)
    # reshape으로 2D 행렬로 변환 (Column-Major vec 일관성)
    return tf.reshape(kron_4d, [tf.shape(A_tf)[0] * tf.shape(B_tf)[0], 
                               tf.shape(A_tf)[1] * tf.shape(B_tf)[1]])

@tf.function(jit_compile=True)
def compute_kron_eigvalue_both(lambda_A_desc, lambda_B_desc):
    """크로네커 고유값 계산: diag(D_KM) = diag(D_BS) ⊗ diag(D_UE)
    
    반환: _kron_ordered와 _desc_sorted 두 정렬 옵션
    """
    lambda_outer = tf.tensordot(lambda_A_desc, lambda_B_desc, axes=0)
    lambda_kron_ordered = tf.reshape(lambda_outer, [-1])
    lambda_desc_sorted = tf.sort(lambda_kron_ordered, direction='DESCENDING')
    
    return {
        'eigvalue_kron_ordered': lambda_kron_ordered,
        'eigvalue_desc_sorted': lambda_desc_sorted
    }

@tf.function(jit_compile=True)
def compute_kron_eigvector_both(lambda_A_desc, lambda_B_desc, U_A_desc, U_B_desc):
    """크로네커 고유벡터 계산: U_KM = U_BS ⊗ U_UE
    
    반환: _kron_ordered와 _desc_sorted 두 정렬 옵션
    - desc_sorted: 고유값 크기 기준 내림차순 정렬
    - kron_ordered: 크로네커 곱 원본 순서 유지
    """
    # 고유값 크로네커 곱 (정렬 인덱스 추적용)
    lambda_outer = tf.tensordot(lambda_A_desc, lambda_B_desc, axes=0)
    lambda_kron_ordered = tf.reshape(lambda_outer, [-1])
    
    # 고유벡터 크로네커 곱 (원본 순서)
    U_kron_4d = tf.einsum('ij,kl->ikjl', U_A_desc, U_B_desc)
    U_kron_ordered = tf.reshape(U_kron_4d, 
                                [tf.shape(U_A_desc)[0] * tf.shape(U_B_desc)[0], 
                                 tf.shape(U_A_desc)[1] * tf.shape(U_B_desc)[1]])
    
    # 옵션 1: 고유값 크기 기준 내림차순 정렬
    lambda_desc_sorted = tf.sort(lambda_kron_ordered, direction='DESCENDING')
    sort_indices = tf.argsort(lambda_kron_ordered, direction='DESCENDING')
    U_desc_sorted = tf.gather(U_kron_ordered, sort_indices, axis=1)
    
    return {
        'eigvalue_desc_sorted': lambda_desc_sorted,
        'eigvector_desc_sorted': U_desc_sorted,
        'eigvalue_kron_ordered': lambda_kron_ordered, 
        'eigvector_kron_ordered': U_kron_ordered
    }


@tf.function(jit_compile=True)
def compute_otimes_desc_eigval_eigvec(lambda_A, lambda_B, U_A, U_B):
    """크로네커 곱셈 및 내림차순 정렬 고유값 및 고유벡터 계산
    
    반환:
    - desc_sorted: 고유값 크기 기준 내림차순 정렬
    """
    # 고유값 크로네커 곱 (정렬 인덱스 추적용)
    lambda_outer = tf.tensordot(lambda_A, lambda_B, axes=0)
    lambda_kron_ordered = tf.reshape(lambda_outer, [-1])
    
    # 고유벡터 크로네커 곱 (원본 순서)
    U_kron_4d = tf.einsum('ij,kl->ikjl', U_A, U_B)
    U_kron_ordered = tf.reshape(U_kron_4d, 
                                [tf.shape(U_A)[0] * tf.shape(U_B)[0], 
                                 tf.shape(U_A)[1] * tf.shape(U_B)[1]])
    
    # 옵션 1: 고유값 크기 기준 내림차순 정렬
    lambda_desc_sorted = tf.sort(lambda_kron_ordered, direction='DESCENDING')
    sort_indices = tf.argsort(lambda_kron_ordered, direction='DESCENDING')
    U_desc_sorted = tf.gather(U_kron_ordered, sort_indices, axis=1)
    
    return {
        'eigvalue_desc_sorted': lambda_desc_sorted,
        'eigvector_desc_sorted': U_desc_sorted,
    }

# ============================================================================
# 실제 공분산 행렬 생성기 (P1E 스타일)
# ============================================================================

def generate_exponential_correlation_matrix(n_ant, alpha):
    """Exponential Correlation Model: R[i,j] = α^|i-j|"""
    indices = np.arange(n_ant)
    i_indices, j_indices = np.meshgrid(indices, indices, indexing='ij')
    R = alpha ** np.abs(i_indices - j_indices)
    
    # Hermitian 보장 (복소수 α의 경우)
    if np.iscomplexobj(alpha):
        upper_tri_mask = np.triu(np.ones_like(R), k=1)
        lower_tri_mask = np.tril(np.ones_like(R), k=-1)
        R = R * upper_tri_mask + np.conj(R.T) * lower_tri_mask + np.diag(np.diag(R))
    
    return R.astype(np.complex64)

def generate_test_covariance_matrices():
    """테스트용 공분산 행렬들 생성"""
    
    # P1D/P1E 실제 규모 파라미터
    n_t, n_r = 1024, 16  # P1D/P1E 실제 안테나 수
    
    # 복소수 상관성: 크기와 위상으로 정의
    alpha_BS_mag = 0.3      # 크기
    alpha_BS_phase = np.pi/6  # 위상 30도
    alpha_BS = alpha_BS_mag * np.exp(1j * alpha_BS_phase)
    
    alpha_UE_mag = 0.2      # 크기  
    alpha_UE_phase = np.pi/12  # 위상 15도
    alpha_UE = alpha_UE_mag * np.exp(1j * alpha_UE_phase)
    
    print(f"공분산 행렬 생성: n_t={n_t}, n_r={n_r}")
    print(f"상관성 파라미터:")
    print(f"  α_BS = {alpha_BS_mag:.3f} × exp(j×{alpha_BS_phase*180/np.pi:.1f}°) = {alpha_BS:.3f}")
    print(f"  α_UE = {alpha_UE_mag:.3f} × exp(j×{alpha_UE_phase*180/np.pi:.1f}°) = {alpha_UE:.3f}")
    
    # 기본 상관 행렬들
    R_BS = generate_exponential_correlation_matrix(n_t, alpha_BS)  # [n_t, n_t]
    R_UE = generate_exponential_correlation_matrix(n_r, alpha_UE)  # [n_r, n_r]
    
    print(f"행렬 크기: R_BS {R_BS.shape}, R_UE {R_UE.shape}")
    
    return {
        'R_BS': R_BS,
        'R_UE': R_UE, 
        'n_t': n_t,
        'n_r': n_r
    }

# ============================================================================
# EVD 수행 및 정렬
# ============================================================================

def perform_evd_sorted(R_tf):
    """EVD 수행 후 고유값 내림차순 정렬"""
    eigenvalues, eigenvectors = tf.linalg.eigh(R_tf)
    
    # Hermitian 행렬의 고유값은 실수이므로 실수 부분만 추출
    eigenvalues_real = tf.math.real(eigenvalues)
    
    # 내림차순 정렬
    idx = tf.argsort(eigenvalues_real, direction='DESCENDING')
    lambda_sorted = tf.gather(eigenvalues_real, idx)
    U_sorted = tf.gather(eigenvectors, idx, axis=1)
    
    return lambda_sorted, U_sorted

# ============================================================================
# EVD 크로네커 관계식 검증
# ============================================================================

def test_evd_kronecker_error_analysis():
    """EVD 크로네커 관계식 오차 분석
    
    가정: R_KM = R_BS ⊗ R_UE (이렇게 만듦)
    검증: 3개 관계식의 Frobenius Norm 오차 (두 정렬 옵션 비교)
    - U_err = ||U_KM - U_BS ⊗ U_UE||_F (desc_sorted vs kron_ordered)
    - D_err = ||D_KM - D_BS ⊗ D_UE||_F (desc_sorted vs kron_ordered) 
    - λ_err = ||diag(D_KM) - diag(D_BS) ⊗ diag(D_UE)||_F (desc_sorted vs kron_ordered)
    
    목적: 크로네커 곱 정렬 방식이 EVD 관계식 정확도에 미치는 영향 분석
    """
    
    print("="*80)
    print(" EVD 크로네커 관계식 오차 분석")
    print("="*80)
    print()
    
    # 1. 기본 공분산 행렬 생성
    print("--- 1. 기본 공분산 행렬 생성 ---")
    cov_data = generate_test_covariance_matrices()
    
    R_BS = tf.constant(cov_data['R_BS'])      # [n_t, n_t]
    R_UE = tf.constant(cov_data['R_UE'])      # [n_r, n_r]
    
    print("✓ R_BS, R_UE 생성 완료")
    
    # 2. 가정: R_KM = R_BS ⊗ R_UE 로 정의
    print("\n--- 2. 가정: R_KM = R_BS ⊗ R_UE ---")
    R_KM = kron_mat_py_tf(R_BS, R_UE)  # 가정으로 만듦
    
    print(f"✓ R_KM 정의 완료: {R_KM.shape}")
    
    # 3. 각 행렬의 EVD 수행
    print("\n--- 3. EVD 수행 ---")
    lambda_BS, U_BS = perform_evd_sorted(R_BS)  # [n_t], [n_t, n_t]
    lambda_UE, U_UE = perform_evd_sorted(R_UE)  # [n_r], [n_r, n_r]  
    lambda_KM, U_KM = perform_evd_sorted(R_KM)  # [n_r*n_t], [n_r*n_t, n_r*n_t]
    
    print(f"✓ EVD 완료")
    
    # 4. 크로네커 관계식 계산 (통합 메서드 적용)
    print("\n--- 4. 크로네커 관계식 계산 (통합 메서드) ---")
    
    # 통합 메서드로 desc_sorted 계산
    kron_desc_result = compute_otimes_desc_eigval_eigvec(lambda_BS, lambda_UE, U_BS, U_UE)
    lambda_KM_theory_desc = kron_desc_result['eigvalue_desc_sorted']
    U_KM_theory_desc = kron_desc_result['eigvector_desc_sorted']
    print(f"✓ 크로네커 고유값/고유벡터 계산 완료 (desc_sorted)")
    
    # 비교용 kron_ordered (기존 방식 유지)
    kron_eigenvalue_result = compute_kron_eigvalue_both(lambda_BS, lambda_UE)
    lambda_KM_theory_kron = kron_eigenvalue_result['eigvalue_kron_ordered']
    
    kron_eigenvector_result = compute_kron_eigvector_both(lambda_BS, lambda_UE, U_BS, U_UE)
    U_KM_theory_kron = kron_eigenvector_result['eigvector_kron_ordered']
    print(f"✓ 비교용 kron_ordered 계산 완료")
    
    # 4-3. 대각행렬
    D_KM_actual_desc = tf.linalg.diag(lambda_KM)  # 이미 정렬됨
    D_KM_theory_desc = tf.linalg.diag(lambda_KM_theory_desc)
    D_KM_theory_kron = tf.linalg.diag(lambda_KM_theory_kron)
    print(f"✓ 크로네커 대각행렬 계산 완료")
    
    # 5. 단계별 오차 추적 (정방향)
    print("\n--- 5. 단계별 오차 추적 (정방향) ---")
    
    # 체크포인트 1: EVD 재구성 검증
    print("\n[체크포인트 1] EVD 재구성 검증")
    R_KM_reconstructed = tf.linalg.matmul(
        tf.linalg.matmul(U_KM, tf.linalg.diag(tf.cast(lambda_KM, tf.complex64))), 
        U_KM, adjoint_b=True
    )
    reconstruct_error = tf.math.real(tf.norm(R_KM - R_KM_reconstructed))
    reconstruct_rel = reconstruct_error / tf.math.real(tf.norm(R_KM))
    
    print(f"EVD 재구성 오차: ||R_KM - U_KM*D_KM*U_KM^H||_F = {reconstruct_error.numpy():.2e}")
    print(f"EVD 재구성 상대오차: {reconstruct_rel.numpy():.2e}")
    print(f"EVD 품질: {'양호' if reconstruct_error.numpy() < 1e-10 else 'EVD 자체에 문제'}")
    
    # 체크포인트 2: 크로네커 곱 자체 검증 (중복 계산으로 확인)
    print("\n[체크포인트 2] 크로네커 곱 일관성")
    R_KM_check = kron_mat_py_tf(R_BS, R_UE)
    kron_consistency = tf.math.real(tf.norm(R_KM - R_KM_check))
    print(f"크로네커 곱 일관성 오차: {kron_consistency.numpy():.2e}")
    print(f"크로네커 곱: {'일관성 유지' if kron_consistency.numpy() < 1e-12 else '구현 문제'}")
    
    # 체크포인트 3: 크로네커 고유값 분석  
    print("\n[체크포인트 3] 크로네커 고유값 분석")
    
    # 실제 EVD 결과 (이미 내림차순 정렬됨)
    lambda_KM_actual = lambda_KM
    
    # 크로네커 이론값과 비교
    lambda_error_fro_desc = tf.norm(lambda_KM_actual - lambda_KM_theory_desc)
    lambda_error_fro_kron = tf.norm(lambda_KM_actual - lambda_KM_theory_kron)
    lambda_norm_fro = tf.norm(lambda_KM_actual)
    lambda_error_rel_desc = lambda_error_fro_desc / lambda_norm_fro
    lambda_error_rel_kron = lambda_error_fro_kron / lambda_norm_fro
    
    print(f"크로네커 고유값 오차 (desc_sorted): {lambda_error_fro_desc.numpy():.2e}")
    print(f"크로네커 고유값 상대오차 (desc_sorted): {lambda_error_rel_desc.numpy():.2e}")
    print(f"고유값 정확도 (desc_sorted): {'양호' if lambda_error_fro_desc.numpy() < 1e-8 else '오차 있음'}")
    print(f"크로네커 고유값 오차 (kron_ordered): {lambda_error_fro_kron.numpy():.2e}")
    print(f"크로네커 고유값 상대오차 (kron_ordered): {lambda_error_rel_kron.numpy():.2e}")
    print(f"고유값 정확도 (kron_ordered): {'양호' if lambda_error_fro_kron.numpy() < 1e-8 else '오차 있음'}")  
    
    # 체크포인트 4: 크로네커 고유벡터 정렬 옵션 분석
    print("\n[체크포인트 4] 크로네커 고유벡터 정렬 옵션 분석")
    
    # 4-1. desc_sorted: 고유값 기준 내림차순 정렬 분석
    print("\n[desc_sorted] 고유값 기준 내림차순 정렬")
    U_KM_desc_sorted = U_KM_theory_desc
    inner_products_desc = tf.abs(tf.linalg.matmul(U_KM, U_KM_desc_sorted, adjoint_a=True))
    max_align_desc = tf.reduce_max(inner_products_desc)
    min_align_desc = tf.reduce_min(tf.reduce_max(inner_products_desc, axis=1))
    U_error_desc = tf.math.real(tf.norm(U_KM - U_KM_desc_sorted))
    
    # 유니터리 성질 확인: 각 행의 내적 제곱합 = 1
    unitary_check_desc = tf.reduce_min(tf.reduce_sum(tf.square(inner_products_desc), axis=1))
    
    print(f"  최대 매칭 내적: {max_align_desc.numpy():.6f}")
    print(f"  최소 매칭 내적: {min_align_desc.numpy():.6f}")  
    print(f"  직접 오차: {U_error_desc.numpy():.2e}")
    print(f"  내적 제곱 합산 최소값: {unitary_check_desc.numpy():.6f} (1에 가까워야 함)")
    
    # 4-2. kron_ordered: 크로네커 원본 순서 분석  
    print("\n[kron_ordered] 크로네커 원본 순서")
    U_KM_kron_ordered = U_KM_theory_kron
    inner_products_kron = tf.abs(tf.linalg.matmul(U_KM, U_KM_kron_ordered, adjoint_a=True))
    max_align_kron = tf.reduce_max(inner_products_kron)
    min_align_kron = tf.reduce_min(tf.reduce_max(inner_products_kron, axis=1))
    U_error_kron = tf.math.real(tf.norm(U_KM - U_KM_kron_ordered))
    
    # 유니터리 성질 확인: 각 행의 내적 제곱합 = 1
    unitary_check_kron = tf.reduce_min(tf.reduce_sum(tf.square(inner_products_kron), axis=1))
    
    print(f"  최대 내적: {max_align_kron.numpy():.6f}")
    print(f"  최소 매칭 내적: {min_align_kron.numpy():.6f}")
    print(f"  직접 오차: {U_error_kron.numpy():.2e}")
    print(f"  유니터리 성질: {unitary_check_kron.numpy():.6f} (1에 가까워야 함)")
    print(f"  원본 순서: {'우수함' if min_align_kron.numpy() > min_align_desc.numpy() else 'desc_sorted가 더 적합'}")
    
    # 4-3. eps_U 계산 (P1D 스타일 고유값 가중평균)
    print("\n[eps_U 계산] P1D 스타일 고유값 가중평균")
    
    # desc_sorted의 eps_U 계산 (이론값 가중치 사용)
    weights_by_eigvals_desc = lambda_KM_theory_desc / tf.reduce_sum(lambda_KM_theory_desc)
    U_inner_prod_max_per_row_desc = tf.reduce_max(inner_products_desc, axis=1)  # 각 행의 최대 내적
    weighted_inner_prod_desc = tf.reduce_sum(weights_by_eigvals_desc * U_inner_prod_max_per_row_desc)
    eps_U_desc = 1.0 - weighted_inner_prod_desc
    
    # kron_ordered의 eps_U 계산 (이론값 가중치 사용)
    weights_by_eigvals_kron = lambda_KM_theory_kron / tf.reduce_sum(lambda_KM_theory_kron)
    U_inner_prod_max_per_row_kron = tf.reduce_max(inner_products_kron, axis=1)  # 각 행의 최대 내적
    weighted_inner_prod_kron = tf.reduce_sum(weights_by_eigvals_kron * U_inner_prod_max_per_row_kron)
    eps_U_kron = 1.0 - weighted_inner_prod_kron
    
    print(f"  eps_U (desc_sorted): {eps_U_desc.numpy():.6f}")
    print(f"  eps_U (kron_ordered): {eps_U_kron.numpy():.6f}")
    print(f"  가중 내적 (desc_sorted): {weighted_inner_prod_desc.numpy():.6f}")
    print(f"  가중 내적 (kron_ordered): {weighted_inner_prod_kron.numpy():.6f}")
    
    # 4-4. 최적 옵션 결정
    better_option = "desc_sorted" if min_align_desc.numpy() > min_align_kron.numpy() else "kron_ordered"
    print(f"\n최적 정렬 방법: {better_option} (부공간 매칭 기준)")
    
    # 후속 분석에서 사용할 값들 (더 나은 옵션 선택)
    if min_align_desc.numpy() > min_align_kron.numpy():
        U_integrated_error = U_error_desc
        min_alignment = min_align_desc
        max_alignment = max_align_desc
    else:
        U_integrated_error = U_error_kron
        min_alignment = min_align_kron
        max_alignment = max_align_kron
    
    # 상대 오차 계산
    U_integrated_rel = U_integrated_error / tf.math.real(tf.norm(U_KM))
    
    # 체크포인트 5: 크로네커 상위 고유값 개별 분석
    print("\n[체크포인트 5] 크로네커 상위 고유값 개별 분석")
    n_show = min(5, len(lambda_KM_actual))
    
    print("크로네커 상위 고유값 비교:")
    for i in range(n_show):
        actual = lambda_KM_actual[i].numpy()
        theory = lambda_KM_theory_desc[i].numpy()
        error = abs(actual - theory)
        print(f"  λ[{i}]: 실제={actual:.6f}, 이론={theory:.6f}, 오차={error:.2e}")
        
    # 통합 정렬 종합 진단
    print("\n[통합 정렬 진단 결과]")
    if reconstruct_error.numpy() > 1e-10:
        print("WARNING: EVD 자체에 수치적 문제 있음 (통합 정렬로 해결 불가)")
    elif kron_consistency.numpy() > 1e-12:
        print("WARNING: 크로네커 곱 구현에 문제 있음")
    elif lambda_error_fro_desc.numpy() > 1e-8:
        print("WARNING: 통합 정렬에도 고유값 오차 잔존")
    elif min_alignment.numpy() < 0.99:
        print("WARNING: 통합 정렬에도 고유벡터 부공간 불일치")
    else:
        print("PASS: 통합 정렬로 EVD 크로네커 관계식 정확도 개선")
    
    # 통합 정렬 오차 변수들 (호환성)
    U_error_fro = U_integrated_error
    U_error_rel = U_integrated_rel
    D_error_fro = lambda_error_fro_desc  # 고유값과 대각행렬은 동일
    D_error_rel = lambda_error_rel_desc
    
    return {
        'U_error_fro': U_error_fro.numpy(),
        'U_error_rel': U_error_rel.numpy(), 
        'lambda_error_fro': lambda_error_fro_desc.numpy(),
        'lambda_error_rel': lambda_error_rel_desc.numpy(),
        'D_error_fro': D_error_fro.numpy(),
        'D_error_rel': D_error_rel.numpy(),
        # 통합 정렬 진단 정보
        'evd_reconstruct_error': reconstruct_error.numpy(),
        'kron_consistency_error': kron_consistency.numpy(),
        'integrated_sort_lambda_error': lambda_error_fro_desc.numpy(),
        'integrated_sort_U_error': U_integrated_error.numpy(),
        'min_subspace_alignment_integrated': min_alignment.numpy(),
        'max_subspace_alignment_integrated': max_alignment.numpy(),
        'matrix_dimensions': {'n_t': 1024, 'n_r': 16, 'n_KM': 16384}
    }

# ============================================================================
# 메인 실행
# ============================================================================

if __name__ == "__main__":
    results = test_evd_kronecker_error_analysis()
    
    print("\n" + "="*80)
    print(" 최종 오차 요약")
    print("="*80)
    print(f"고유벡터 오차:   ||U_KM - U_BS⊗U_UE||_F = {results['U_error_fro']:.2e} ({results['U_error_rel']:.2e} 상대)")
    print(f"고유값 오차:     ||λ_KM - λ_BS⊗λ_UE||_F = {results['lambda_error_fro']:.2e} ({results['lambda_error_rel']:.2e} 상대)")
    print(f"대각행렬 오차:   ||D_KM - D_BS⊗D_UE||_F = {results['D_error_fro']:.2e} ({results['D_error_rel']:.2e} 상대)")
    
    # 오차 판정
    threshold = 1e-10
    all_good = all(results[key] < threshold for key in ['U_error_fro', 'lambda_error_fro', 'D_error_fro'])
    
    print(f"\n결과: {'모든 관계식이 수치적으로 성립' if all_good else '일부 관계식에서 큰 오차 발견'}")
    print("="*80)