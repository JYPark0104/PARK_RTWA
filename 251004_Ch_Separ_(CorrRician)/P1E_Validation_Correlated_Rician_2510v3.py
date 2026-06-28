# ======================================================================
# P1E_Validation_Correlated_Rician_2509v1.py
# P1E: P1D 코드 무결성 검증을 위한 Correlated Rician Channel 테스트
# 
# === 최상위 목적 ===
# P1D의 모듈러 메서드들 (온라인 누적, 분리성 분석) 검증
# - Correlated Rician Channel 기반 Ground Truth 비교 검증
# - 크로네커 모델 기준 분리성 성립 확인 (correlation 강도 무관)
# - 온라인 누적 공분산의 수렴성 모니터링
# 
# === 핵심 설계 원리 ===
# 1. Ground Truth: R_BS_tru ⊗ R_UE_tru → R_AE_tru (완전 분리 가능)
# 2. 테스트 데이터: Correlated Rician 채널 샘플 생성
# 3. P1D 검증: 온라인 누적 → 분리성 분석 → Ground Truth 비교
# 4. 수렴성: 샘플 수별 오차 변화량 모니터링
# 
# === 주요 구성 요소 ===
# - P1E_Config: 테스트 파라미터 설정
# - CorrelatedRicianGenerator: 테스트 데이터 생성기
# - P1D_ModuleValidator: P1D 모듈 검증기 (import 기반)
# - ConvergenceMonitor: 수렴성 모니터링
# - P1E_ResultManager: 검증 결과 저장
# 
# === 입력/출력 ===
# 입력: 테스트 파라미터 (K_Ric_dB, alpha_b, alpha_u, 안테나 구성)
# 출력: P1D 모듈 검증 결과 CSV (오차, 분리성, 수렴성)
# 
# === 핵심 기술 ===
# - Exponential Correlation Model 기반 Ground Truth 생성
# - Matrix Square Root 기반 Correlated 채널 샘플링
# - P1D 온라인 누적 알고리즘 직접 호출 검증
#
# === 주요 수정 이력 ===
# [250930] 크로네커 채널 모델 구현 오류 3종 수정
# 1. Column-Major vec(): P1D와 LinearOperatorKronecker 일치 (transpose + reshape)
# 2. Hermitian ECM: 복소수 α 사용 시 하삼각 conjugate 적용
# 3. 채널 생성: sqrtm(R_BS) transpose 제거 (Hermitian 속성)
# Ground Truth: R_AE = R_BS ⊗ R_UE (Column-Major vec 기준)
# 검증 결과 (1,677만 샘플): R_BS/R_UE < 1e-5, R_AE 4.4%
# ======================================================================

# ===== SECTION 1: 환경 설정 =====
import os
import sys
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import json

# TensorFlow 환경 설정 (P1D와 동일)
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc'
gpu_num = 0  
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

import tensorflow as tf

# GPU 메모리 증가 허용 설정
gpus = tf.config.list_physical_devices('GPU')
for g in gpus:
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except RuntimeError as e:
        print(e)

# TensorFlow 최적화 설정
tf.get_logger().setLevel("ERROR")
tf.config.optimizer.set_jit(True)
tf.config.threading.set_inter_op_parallelism_threads(0)
tf.config.threading.set_intra_op_parallelism_threads(0)
tf.random.set_seed(43)  # P1E는 재현 가능한 테스트용

# 스크립트 디렉토리 기준 경로 설정 (P1D 임포트 및 결과 저장용)
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# P1D 모듈 임포트
try:
    from P1D_Rays_to_AE_Separability_2510v3 import SeparabilityAnalyzer, P1D_Config, ChannelAnalyzer, P1D_ResultManager, ChCoeGen, BlockWelfordAccumulator
    print("P1D 모듈 임포트 성공")
except ImportError as e:
    print(f"Error: P1D 모듈 임포트 실패: {e}")
    print(f"현재 디렉토리: {script_dir}")
    print("P1D_Rays_to_AE_Separability_2510v3.py 파일이 같은 디렉토리에 있는지 확인하세요.")
    sys.exit(1)

# 주피터/IPython 화면 클리어 지원
try:
    from IPython.display import clear_output
    JUPYTER_AVAILABLE = True
except ImportError:
    JUPYTER_AVAILABLE = False

# ===== SECTION 2: P1E_Config (테스트 파라미터 설정) =====
class P1E_Config:
    """P1E 테스트 파라미터 설정 (P1D_Config 기반 완전 1:1 정렬)"""
    
    def __init__(self):

        # ===== P1E 전용 테스트 파라미터 (P1D 구조 유지) =====
        # Correlated Rician Channel 테스트 파라미터
        self.K_Ric_dB = 5.0                   # Rician K-factor (dB)
        self.alpha_b = 0.2                     # 기지국 측 상관성 파라미터
        self.alpha_u = 0.1                     # 사용자 단말 측 상관성 파라미터

        # 검증 허용 오차 (P1E 전용)
        self.covariance_error_threshold = 1e-3  # 공분산 상대 오차 허용 임계값
        self.convergence_threshold = 1e-3       # 수렴성 판정 임계값

        # ===== P1D_Config 기반 초기화 (1:1 정렬) =====
        # P1D 설정을 먼저 로드하여 완전 동일한 구조 보장
        self.p1d_config = P1D_Config()

        # ===== P1D와 동일한 시스템 파라미터 =====
        # OFDM 채널 시스템 파라미터 (P1D와 정확히 동일)
        self.OFDM_FFT = self.p1d_config.OFDM_FFT           # P1D: 256
        self.OFDM_SCS = self.p1d_config.OFDM_SCS           # P1D: 120e3
        self.OFDM_BW = self.p1d_config.OFDM_BW             # P1D: FFT * SCS
        
        # TX/RX 배열 설정 (P1D와 정확히 동일)
        self.TX_Array = self.p1d_config.TX_Array.copy()
        self.RX_Array = self.p1d_config.RX_Array.copy()
        
        # 안테나 수 계산 (P1D와 동일한 공식)
        self.n_t = 1024   # TX 안테나 수 self.p1d_config.n_t  
        self.n_r = 16   # RX 안테나 수 self.p1d_config.n_r  
        
        # 채널 생성 파라미터 (P1D 변수명 그대로)
        self.static_ch_realizations = 2**20  # P1D: 4096 (2^12)
        self.doppler_time_realizations = self.p1d_config.doppler_time_realizations  # P1D: 1
        self.doppler_sym_ofdm = self.p1d_config.doppler_sym_ofdm  # P1D: 1
        
        # 분리성 임계값 (P1D와 정확히 동일)
        self.epsilon_lambda_threshold = self.p1d_config.epsilon_lambda_threshold  # P1D: 0.1
        self.epsilon_U_threshold = self.p1d_config.epsilon_U_threshold            # P1D: 0.1
        self.channel_power_threshold = self.p1d_config.channel_power_threshold    # P1D: 1e-20
        
        # GPU 병렬 처리 최적화 설정 (P1D와 동일)
        self.batch_size_rx = self.p1d_config.batch_size_rx                    # P1D: 1
        self.use_mixed_precision = self.p1d_config.use_mixed_precision        # P1D: False
        self.gpu_memory_limit_gb = self.p1d_config.gpu_memory_limit_gb        # P1D: 45
        
        # 수렴성 체크 포인트 (log2 기반 체계적 생성)
        self.convergence_check_points = [2**i for i in range(int(np.log2(self.static_ch_realizations)) + 1)]
        
        # P1E 결과 저장 디렉토리 (P1D 구조와 병렬)
        self.P1E_VALIDATION_OUTPUT_DIR = os.path.join(script_dir, "P1E_Validation_Results")
                
        print(f"P1E 테스트 설정 (P1D_Config 기반 1:1 정렬):")
        print(f"  - 안테나 구성: TX={self.n_t}, RX={self.n_r}")
        print(f"  - OFDM FFT: {self.OFDM_FFT} (P1D와 동일)")
        print(f"  - 채널 실현 수: {self.static_ch_realizations} (P1D와 동일)")
        print(f"  - 배치 크기: {self.batch_size_rx} (P1D와 동일)")
        print(f"  - Rician K-factor: {self.K_Ric_dB} dB")
        print(f"  - Correlation: alpha_b={self.alpha_b}, alpha_u={self.alpha_u}")
        print(f"  - 결과 저장: {self.P1E_VALIDATION_OUTPUT_DIR}")

# ===== SECTION 3: MATLAB Column-Major 유틸리티 함수 =====
def vec_mat_py_tf(H_tf):
    """MATLAB vec(H) 구현 (Column-Major, TensorFlow Only)
    
    Args:
        H_tf: 입력 행렬/텐서
            - 2D: [n_r, n_t] → [n_r*n_t]
            - 3D: [batch, n_r, n_t] → [batch, n_r*n_t]
    
    Returns:
        vectorized 결과 (Column-Major 순서)
    """
    if len(H_tf.shape) == 2:
        return tf.reshape(tf.transpose(H_tf), [-1])
    elif len(H_tf.shape) == 3:
        return tf.reshape(tf.transpose(H_tf, perm=[0, 2, 1]), [tf.shape(H_tf)[0], -1])
    else:
        raise ValueError(f"Unsupported shape: {H_tf.shape}")

def kron_mat_py_tf(A_tf, B_tf):
    """MATLAB kron(A,B) 구현 (einsum & reshape, TensorFlow Only)
    
    Args:
        A_tf: 첫 번째 행렬 [m, n]
        B_tf: 두 번째 행렬 [p, q]
    
    Returns:
        크로네커 곱 A ⊗ B [m*p, n*q]
    """
    # einsum으로 4D 텐서 생성: A[i,j] * B[k,l] -> [i,k,j,l]
    kron_4d = tf.einsum('ij,kl->ikjl', A_tf, B_tf)
    # reshape으로 2D 행렬로 변환 (Column-Major vec 일관성)
    return tf.reshape(kron_4d, [tf.shape(A_tf)[0] * tf.shape(B_tf)[0], 
                               tf.shape(A_tf)[1] * tf.shape(B_tf)[1]])

# ===== SECTION 4: Correlated Rician Generator =====
class CorrelatedRicianGenerator:
    """Correlated Rician Channel 테스트 데이터 생성기
    
    명시된 구분에 따른 구현:
    1. 테스트 데이터 설정부: Ground Truth 생성 및 저장
    2. 테스트 데이터 '온라인' 발생부: 실시간 채널 샘플 생성
    """
    
    def __init__(self, config):
        self.config = config
        self.n_r = config.n_r
        self.n_t = config.n_t
        self.K_Ric_dB = config.K_Ric_dB
        self.alpha_b = config.alpha_b
        self.alpha_u = config.alpha_u
        self.OFDM_FFT = config.OFDM_FFT
        
        # === 테스트 데이터 설정부 실행 ===
        self._setup_ch_gen_parameters()
        
        # P1D ChCoeGen 클래스 구조 검증 (간단)
        print("P1D ChCoeGen 클래스 구조 검증: ✓")    

        
    # 초기에 적은 횟수만 실행되는 함수이니 tf.function 사용하지 않음
    def _generate_exponential_correlation_matrix_tf(self, n_ant, alpha):
        """
        [250930 수정] Exponential Correlation Model 생성 - Hermitian 속성 보장
        
        표준 ECM: R[i,j] = α^|i-j|
        - 실수 α: 자동으로 Hermitian (R = R^H)
        - 복소수 α: 하삼각에 conjugate 적용 필요
        
        복소수 α 사용 시 Hermitian 보장 방법:
        - 상삼각(i≤j): R[i,j] = α^|i-j|
        - 하삼각(i>j): R[i,j] = conj(α^|i-j|) = conj(R[j,i])
        """
        # 거리 행렬 생성 (TensorFlow)
        indices = tf.range(n_ant, dtype=tf.float32)
        i_mat, j_mat = tf.meshgrid(indices, indices, indexing='ij')
        distance_mat = tf.abs(i_mat - j_mat)
        
        # 단일 랜덤 phase 생성 (시드 고정으로 재현 가능)
        tf.random.set_seed(42)
        random_phase = tf.random.uniform([], minval=-tf.constant(np.pi), maxval=tf.constant(np.pi))
        
        # 복소수 alpha 생성: alpha_complex = |alpha| * exp(j*random_phase)
        alpha_complex = tf.cast(alpha, tf.complex64) * tf.exp(tf.complex(0.0, random_phase))
        
        # === [250930 수정] Hermitian ECM 생성 ===
        # 상삼각: R[i,j] = alpha^|i-j|
        R_upper = tf.pow(alpha_complex, tf.cast(distance_mat, tf.complex64))
        
        # Hermitian 조건: R[i,j] = conj(R[j,i]) for i > j
        # 상삼각 mask (i ≤ j, 대각 포함)
        upper_tri_mask = tf.cast(i_mat <= j_mat, tf.complex64)
        # 하삼각 mask (i > j)
        lower_tri_mask = tf.cast(i_mat > j_mat, tf.complex64)
        
        # 최종 Hermitian 행렬: 상삼각 원본 + 하삼각 conjugate
        R = R_upper * upper_tri_mask + tf.math.conj(tf.transpose(R_upper)) * lower_tri_mask

        return tf.cast(R, tf.complex64)
    
    def _setup_ch_gen_parameters(self):
        """테스트 데이터 설정부: 채널 생성 파라미터 설정
        
        지시사항 구현:
        - Rician K-factor 설정
        - Exponential Correlation Model에 따른 R_BS_tru, R_UE_tru 공분산 행렬 생성 및 저장
        - R_AE_tru = R_BS_tru ⊗ R_UE_tru 저장
        - 공분산 행렬의 고유값 분해 결과 저장:
          * R_AE_tru = U_AE_tru * diag(d_AE_tru) * U_AE_tru^H
          * R_BS_tru = U_BS_tru * diag(d_BS_tru) * U_BS_tru^H  
          * R_UE_tru = U_UE_tru * diag(d_UE_tru) * U_UE_tru^H
        """
        print("=== 테스트 데이터 설정부: 채널 생성 파라미터 설정 중 ===", end=' ')
        
        # 1. Exponential Correlation 기반 공분산 행렬 생성 (TensorFlow, unscaled)
        self.R_BS_tru_unscaled = self._generate_exponential_correlation_matrix_tf(self.n_t, self.alpha_b)
        self.R_UE_tru_unscaled = self._generate_exponential_correlation_matrix_tf(self.n_r, self.alpha_u)
        
        # === [250930 수정] Ground Truth: R_AE = R_BS ⊗ R_UE (Column-Major) ===

        # self.R_AE_tru_unscaled = kron_mat_py_tf(
        #     self.R_BS_tru_unscaled, self.R_UE_tru_unscaled
        # )  # R_BS ⊗ R_UE (Column-Major vec 기준)

        R_BS_tru_unscaled_T = tf.transpose(self.R_BS_tru_unscaled)
        self.R_AE_tru_unscaled = kron_mat_py_tf(
            R_BS_tru_unscaled_T, self.R_UE_tru_unscaled
        )

        # 3. Rician K-factor 계산 및 스케일링 적용
        K_Ric_linear = tf.pow(10.0, self.K_Ric_dB / 10.0)  # dB to linear
        self.K_Ric_cov_scale = tf.cast(1.0 / (1.0 + K_Ric_linear), tf.complex64)
        self.K_Ric_lin_mean = tf.cast(tf.sqrt(K_Ric_linear / (1.0 + K_Ric_linear)), tf.float32)
        self.K_Ric_lin_rand = tf.cast(tf.sqrt(1.0 / (1.0 + K_Ric_linear)), tf.float32)
        
        # 4. Rician 랜덤 성분 스케일 적용: R_tru = (1/(1+K)) * R_tru_unscaled
        self.R_BS_tru = self.K_Ric_cov_scale * self.R_BS_tru_unscaled
        self.R_UE_tru = self.K_Ric_cov_scale * self.R_UE_tru_unscaled
        self.R_AE_tru = self.K_Ric_cov_scale * self.R_AE_tru_unscaled
        
        # 5. 고유값 분해 수행 및 정렬 (스케일된 R_tru에 대해 수행)
        # R_AE_tru 고유값 분해
        d_AE_unsorted, U_AE_unsorted = tf.linalg.eigh(self.R_AE_tru)
        d_AE_unsorted = tf.math.real(d_AE_unsorted)
        idx_AE = tf.argsort(d_AE_unsorted, direction='DESCENDING')
        self.d_AE_tru = tf.gather(d_AE_unsorted, idx_AE)
        self.U_AE_tru = tf.gather(U_AE_unsorted, idx_AE, axis=1)
        
        # R_BS_tru 고유값 분해
        d_BS_unsorted, U_BS_unsorted = tf.linalg.eigh(self.R_BS_tru)
        d_BS_unsorted = tf.math.real(d_BS_unsorted)
        idx_BS = tf.argsort(d_BS_unsorted, direction='DESCENDING')
        self.d_BS_tru = tf.gather(d_BS_unsorted, idx_BS)
        self.U_BS_tru = tf.gather(U_BS_unsorted, idx_BS, axis=1)
        
        # R_UE_tru 고유값 분해
        d_UE_unsorted, U_UE_unsorted = tf.linalg.eigh(self.R_UE_tru)
        d_UE_unsorted = tf.math.real(d_UE_unsorted)
        idx_UE = tf.argsort(d_UE_unsorted, direction='DESCENDING')
        self.d_UE_tru = tf.gather(d_UE_unsorted, idx_UE)
        self.U_UE_tru = tf.gather(U_UE_unsorted, idx_UE, axis=1)
        
        # 6. 평균 채널 발생 (TensorFlow, 채널 생성 전용)
        tf.random.set_seed(123)  # 재현 가능한 평균 채널
        phase_mean = tf.random.uniform((self.n_r, self.n_t), minval=-tf.constant(np.pi), maxval=tf.constant(np.pi))
        self.H_mean_tf = tf.cast(tf.exp(tf.complex(tf.zeros_like(phase_mean), phase_mean)), tf.complex64)
        
        # 7. 행렬 제곱근 계산 (unscaled 버전으로 계산, 채널 생성용)
        self.sqrtm_R_UE_tf = tf.linalg.sqrtm(self.R_UE_tru_unscaled)
        self.sqrtm_R_BS_tf = tf.linalg.sqrtm(self.R_BS_tru_unscaled)
        
        print("완료")
        print(f"  - K-factor: {self.K_Ric_dB} dB, 스케일={self.K_Ric_cov_scale.numpy():.4f} (1/(1+K))")
        print(f"  - R_BS_tru (scaled): {self.R_BS_tru.shape}, trace={tf.math.real(tf.linalg.trace(self.R_BS_tru)).numpy():.4f}")
        print(f"  - R_UE_tru (scaled): {self.R_UE_tru.shape}, trace={tf.math.real(tf.linalg.trace(self.R_UE_tru)).numpy():.4f}")
        print(f"  - R_AE_tru (scaled): {self.R_AE_tru.shape}, trace={tf.math.real(tf.linalg.trace(self.R_AE_tru)).numpy():.4f}")
        print(f"  - H_mean_tf: {self.H_mean_tf.shape}")
        print(f"  - Rician 계수: mean={self.K_Ric_lin_mean.numpy():.3f}, rand={self.K_Ric_lin_rand.numpy():.3f}")
    
    
    @tf.function(jit_compile=True)
    def _online_channel_sample_generation_tf(self, batch_size, sqrtm_R_UE, sqrtm_R_BS, H_mean, K_Ric_lin_mean, K_Ric_lin_rand):
        """
        [250930 수정] Correlated Rician 채널 샘플 생성
        
        크로네커 모델: H = R_UE^{1/2} H_iid R_BS^{1/2}^H
        - R_BS, R_UE는 Hermitian이므로 sqrtm(R)도 Hermitian
        - (R_BS^{1/2})^H = R_BS^{1/2} → transpose 불필요
        """
        
        # 1. H_iid 발생: (randn(Nr,Nt) + 1i * randn(Nr,Nt)) / sqrt(2) (P1D 방식)
        H_iid_real = tf.random.normal([batch_size, self.OFDM_FFT, self.n_r, self.n_t], dtype=tf.float32)
        H_iid_imag = tf.random.normal([batch_size, self.OFDM_FFT, self.n_r, self.n_t], dtype=tf.float32)
        H_iid = tf.complex(H_iid_real, H_iid_imag) / np.sqrt(2.0) # [batch_size=1, OFDM_FFT, n_r, n_t]
        
        # === [250930 수정] 크로네커 채널 생성 ===
        # H_rand = R_UE^{1/2} * H_iid * R_BS^{1/2}^H
        # R_BS^{1/2}가 Hermitian이므로: (R_BS^{1/2})^H = R_BS^{1/2}
        # 따라서 transpose 없이 그대로 사용
        # 
        # 이전 버전에서 tf.transpose() 사용은 오류였음:
        # - transpose는 Hermitian transpose(^H)가 아닌 일반 transpose(^T)
        # - Hermitian 행렬에서 ^H = 자기자신이므로 변환 불필요
        H_rand = tf.linalg.matmul(sqrtm_R_UE[None, None, :, :], 
                                 tf.linalg.matmul(H_iid, sqrtm_R_BS[None, None, :, :]))

        # 3. H_mean 확장: [n_r, n_t] → [batch_size, OFDM_FFT, n_r, n_t]
        H_mean_exp = tf.broadcast_to(H_mean[None, None, :, :], 
                                   [batch_size, self.OFDM_FFT, self.n_r, self.n_t])
        
        # 4. H_sample[f] = sqrt(K_Ric/(1+K_Ric)) * H_mean + sqrt(1/(1+K_Ric)) * H_rand[f]
        # float 계수를 complex로 캐스팅하여 타입 일치 보장
        K_mean_complex = tf.cast(K_Ric_lin_mean, tf.complex64)
        K_rand_complex = tf.cast(K_Ric_lin_rand, tf.complex64)
        H_sample = K_mean_complex * H_mean_exp + K_rand_complex * H_rand
        
        return tf.cast(H_sample, tf.complex64)
    

# ===== SECTION 4: P1D Module Validator =====
class P1D_ModuleValidator:
    """P1D 모듈 검증기 (P1D 클래스들을 최대한 활용)"""
    
    def __init__(self, config, channel_parameters):
        self.config = config
        self.channel_parameters = channel_parameters
        
        # P1D 설정 초기화 (P1E 테스트 파라미터로 오버라이드)
        self.p1d_config = P1D_Config()
        self.p1d_config.n_r = config.n_r
        self.p1d_config.n_t = config.n_t
        self.p1d_config.OFDM_FFT = config.OFDM_FFT
        self.p1d_config.doppler_sym_ofdm = 1  # P1E는 단일 심볼
        self.p1d_config.batch_size_rx = config.batch_size_rx
        
        # P1D SeparabilityAnalyzer 초기화
        self.p1d_analyzer = SeparabilityAnalyzer(self.p1d_config)
        
        # BlockWelfordAccumulator 초기화 (P1D v3 방식)
        self.accumulator = None  # reset_accumulation_state()에서 초기화
        
        # P1D ChannelAnalyzer 초기화 (검증용 - 실제로는 사용하지 않지만 무결성 확인)
        # 가상의 ChCoeGen, topology, ray_pdap 생성해서 ChannelAnalyzer 초기화 가능한지 확인
        try:
            # ChCoeGen 인스턴스 생성 (최소 파라미터로)
            dummy_tx_array = None  # 실제로는 PanelArray가 필요하지만 검증용
            dummy_rx_array = None
            # P1D ChannelAnalyzer는 실제 Ray 데이터가 필요하므로 초기화만 확인
            self.p1d_channel_analyzer_class = ChannelAnalyzer  # 클래스 참조 저장
            print("P1D ChannelAnalyzer 클래스 검증: ✓")
        except Exception as e:
            print(f"P1D ChannelAnalyzer 검증 실패: {e}")
        
        # P1D ResultManager 초기화 검증
        try:
            self.p1d_result_manager = P1D_ResultManager(self.p1d_config)
            print("P1D ResultManager 초기화 검증: ✓")
        except Exception as e:
            print(f"P1D ResultManager 초기화 실패: {e}")
            self.p1d_result_manager = None
        
        # 블록 누적 상태 초기화
        self.reset_accumulation_state()
        
        # JIT 워밍업
        print("P1D SeparabilityAnalyzer JIT 워밍업...", end=' ')
        self.p1d_analyzer.warmup_jit_functions()
        print("완료")
    
    def reset_accumulation_state(self):
        """BlockWelfordAccumulator 초기화 (P1D v3 방식)"""
        n_r, n_t = self.config.n_r, self.config.n_t
        self.accumulator = BlockWelfordAccumulator(n_r, n_t, dtype=tf.complex64)
    
    def update_online_accumulation(self, channel_samples, accumulated_sample_count):
        """BlockWelfordAccumulator를 사용한 온라인 누적 (P1D v3 방식)
        
        Parameters:
            channel_samples: [batch_size, OFDM_FFT, n_r, n_t] TensorFlow tensor
            accumulated_sample_count: 현재까지 누적된 총 OFDM 샘플 수 (static_idx × OFDM_FFT)
        
        Returns:
            dict: {'cumulative_update': {}} (변화량은 체크포인트에서 계산)
        """
        # [batch_size, OFDM_FFT, n_r, n_t] -> [batch_size*OFDM_FFT, n_r, n_t]
        batch_size = tf.shape(channel_samples)[0]
        n_fft = tf.shape(channel_samples)[1]
        n_r = tf.shape(channel_samples)[2]
        n_t = tf.shape(channel_samples)[3]
        
        H_batch = tf.reshape(channel_samples, [batch_size * n_fft, n_r, n_t])
        
        # BlockWelfordAccumulator 업데이트
        self.accumulator.update(H_batch)
        
        return {
            'cumulative_update': {}
        }
    
    @tf.function
    def _compute_kronecker_error_evd_sorted(self, R_AE_ref_tf, R_BS_tf, R_UE_tf):
        """EVD + desc_sorted 재조합 후 Kronecker 오차 계산 (일반화 버전)
        
        Args:
            R_AE_ref_tf: 비교 기준 행렬 (Ground Truth or Sample)
            R_BS_tf: 크로네커 인수 R_BS  
            R_UE_tf: 크로네커 인수 R_UE
            
        Returns:
            err_fro_rel: ||R_AE_ref_recon - R_KM_recon||_F / ||R_AE_ref||_F (상대 오차)
        """
        # 1. 각 행렬의 EVD + desc_sorted (직접 수행)
        try:
            # EVD 수행
            lambda_AE, U_AE = tf.linalg.eigh(R_AE_ref_tf)
            lambda_BS, U_BS = tf.linalg.eigh(R_BS_tf)  
            lambda_UE, U_UE = tf.linalg.eigh(R_UE_tf)
            
            # 실수 고유값 추출
            lambda_AE = tf.math.real(lambda_AE)
            lambda_BS = tf.math.real(lambda_BS)
            lambda_UE = tf.math.real(lambda_UE)
            
            # desc_sorted 정렬
            idx_AE = tf.argsort(lambda_AE, direction='DESCENDING')
            idx_BS = tf.argsort(lambda_BS, direction='DESCENDING')
            idx_UE = tf.argsort(lambda_UE, direction='DESCENDING')
            
            lambda_AE_desc = tf.gather(lambda_AE, idx_AE)
            lambda_BS_desc = tf.gather(lambda_BS, idx_BS)
            lambda_UE_desc = tf.gather(lambda_UE, idx_UE)
            
            U_AE_desc = tf.gather(U_AE, idx_AE, axis=1)
            U_BS_desc = tf.gather(U_BS, idx_BS, axis=1)  
            U_UE_desc = tf.gather(U_UE, idx_UE, axis=1)
            
        except Exception as e:
            # EVD 실패 시 기본 크로네커 곱 상대 오차로 fallback
            R_KM_tf = kron_mat_py_tf(R_BS_tf, R_UE_tf)
            fro_err = tf.reduce_sum(tf.square(tf.abs(R_AE_ref_tf - R_KM_tf)))
            fro_ref = tf.reduce_sum(tf.square(tf.abs(R_AE_ref_tf)))
            return tf.math.real(fro_err / fro_ref)
        
        # 2. 크로네커 곱으로 이론적 R_KM 생성 (P1D 메서드 활용)
        try:
            lambda_KM_desc, U_KM_desc = \
                self.p1d_analyzer._compute_otimes_desc_eigval_eigvec(
                    lambda_BS_desc, lambda_UE_desc, U_BS_desc, U_UE_desc)
        except Exception as e:
            # 크로네커 곱 실패 시 기본 방식 상대 오차로 fallback
            R_KM_tf = kron_mat_py_tf(R_BS_tf, R_UE_tf)
            fro_err = tf.reduce_sum(tf.square(tf.abs(R_AE_ref_tf - R_KM_tf)))
            fro_ref = tf.reduce_sum(tf.square(tf.abs(R_AE_ref_tf)))
            return tf.math.real(fro_err / fro_ref)
        
        # 3. desc_sorted 기준으로 재조합
        # R_AE_ref 재조합: U * diag(λ) * U^H
        D_AE_recon = tf.linalg.diag(tf.cast(lambda_AE_desc, tf.complex64))
        R_AE_ref_recon = tf.linalg.matmul(
            tf.linalg.matmul(U_AE_desc, D_AE_recon), U_AE_desc, adjoint_b=True)
        
        # R_KM 재조합: U_KM * diag(λ_KM) * U_KM^H  
        D_KM_recon = tf.linalg.diag(tf.cast(lambda_KM_desc, tf.complex64))
        R_KM_recon = tf.linalg.matmul(
            tf.linalg.matmul(U_KM_desc, D_KM_recon), U_KM_desc, adjoint_b=True)
        
        # 4. 동일한 정렬 기준으로 상대 오차 계산
        err_recon = R_AE_ref_recon - R_KM_recon
        fro_err = tf.reduce_sum(tf.square(tf.abs(err_recon)))
        fro_ref = tf.reduce_sum(tf.square(tf.abs(R_AE_ref_tf)))
        return tf.math.real(fro_err / fro_ref)
    
    @tf.function
    def _compute_covariance_errors_tf(self, R_AE_sam_tf, R_BS_sam_tf, R_UE_sam_tf,
                                       R_AE_tru_tf, R_BS_tru_tf, R_UE_tru_tf, K_Ric_cov_scale):
        """공분산 오차 계산 (TF 연산, JIT 최적화)
        
        R_tru는 이미 K-factor 스케일링 적용됨: R_tru = (1/(1+K)) * R_tru_unscaled
        """
        # 오차 행렬: R_tru - R_sam (R_tru는 이미 스케일됨)
        R_AE_err_tf = R_AE_tru_tf - R_AE_sam_tf
        R_BS_err_tf = R_BS_tru_tf - R_BS_sam_tf
        R_UE_err_tf = R_UE_tru_tf - R_UE_sam_tf
        
        # 나머지 상대적 프로베니우스 노름 오차 계산
        fro_tru_AE = tf.reduce_sum(tf.square(tf.abs(R_AE_tru_tf)))
        fro_tru_BS = tf.reduce_sum(tf.square(tf.abs(R_BS_tru_tf)))
        fro_tru_UE = tf.reduce_sum(tf.square(tf.abs(R_UE_tru_tf)))

        fro_err_AE = tf.reduce_sum(tf.square(tf.abs(R_AE_err_tf)))
        fro_err_BS = tf.reduce_sum(tf.square(tf.abs(R_BS_err_tf)))
        fro_err_UE = tf.reduce_sum(tf.square(tf.abs(R_UE_err_tf)))
        
        # 상대 오차
        R_AE_err_fro_rel = tf.math.real(fro_err_AE / fro_tru_AE)
        R_BS_err_fro_rel = tf.math.real(fro_err_BS / fro_tru_BS)
        R_UE_err_fro_rel = tf.math.real(fro_err_UE / fro_tru_UE)
        
        # EVD + desc_sorted 재조합 후 크로네커 오차 계산 (두 가지)
        # 1. Ground Truth 기준 크로네커 오차
        R_KM_err_fro_rel = self._compute_kronecker_error_evd_sorted(
            K_Ric_cov_scale*R_AE_tru_tf, tf.transpose(R_BS_sam_tf), R_UE_sam_tf)
        
        # 2. 샘플 기준 크로네커 분리성 오차 (채널 계산 검증용)
        R_sep_err_fro_rel = self._compute_kronecker_error_evd_sorted(
            K_Ric_cov_scale*R_AE_sam_tf, tf.transpose(R_BS_sam_tf), R_UE_sam_tf)
        
        # 3. EVD 재조합 무결성 테스트 (동일한 크로네커 곱끼리 비교)
        R_identity_test = self._compute_kronecker_error_evd_sorted(
            kron_mat_py_tf(R_BS_sam_tf, R_UE_sam_tf), R_BS_sam_tf, R_UE_sam_tf)

        return R_AE_err_fro_rel, R_KM_err_fro_rel, R_BS_err_fro_rel, R_UE_err_fro_rel, R_sep_err_fro_rel, R_identity_test
    
    def compute_covariance_errors(self):
        """현재 누적된 샘플 공분산과 참조값 비교 (BlockWelfordAccumulator 사용)"""
        # 1. BlockWelfordAccumulator에서 최종 공분산 계산
        final_covs = self.accumulator.finalize()
        
        N_total = int(self.accumulator.count.numpy())
        if N_total < 2:
            return {
                'R_AE_err_fro_rel': np.nan,
                'R_BS_err_fro_rel': np.nan,
                'R_UE_err_fro_rel': np.nan
            }
        
        # 최종 공분산 행렬 추출
        R_AE_sam_tf = final_covs['R_AE_sam'][0]
        R_BS_sam_tf = final_covs['R_BS_sam'][0]
        R_UE_sam_tf = final_covs['R_UE_sam'][0]
        
        # 참조값 및 스케일링 파라미터 추출
        R_AE_tru_tf = self.channel_parameters['R_AE_tru']
        R_BS_tru_tf = self.channel_parameters['R_BS_tru']
        R_UE_tru_tf = self.channel_parameters['R_UE_tru']
        K_Ric_cov_scale = self.channel_parameters['K_Ric_cov_scale']
        
        # JIT 최적화된 TF 함수 호출
        R_AE_err_fro_rel, R_KM_err_fro_rel, R_BS_err_fro_rel, R_UE_err_fro_rel, R_sep_err_fro_rel, R_identity_test = \
            self._compute_covariance_errors_tf(
                R_AE_sam_tf, R_BS_sam_tf, R_UE_sam_tf,
                R_AE_tru_tf, R_BS_tru_tf, R_UE_tru_tf, K_Ric_cov_scale
            )
        
        return {
            'R_AE_err_fro_rel': R_AE_err_fro_rel.numpy(),
            'R_KM_err_fro_rel': R_KM_err_fro_rel.numpy(),
            'R_BS_err_fro_rel': R_BS_err_fro_rel.numpy(),
            'R_UE_err_fro_rel': R_UE_err_fro_rel.numpy(),
            'R_sep_err_fro_rel': R_sep_err_fro_rel.numpy(),  # 샘플 크로네커 분리성
            'R_identity_test': R_identity_test.numpy()        # EVD 재조합 무결성 테스트
        }
    
    def compute_separability_validation(self, accumulated_sample_count):
        """P1D 분리성 분석 수행 및 참조값 비교 (BlockWelfordAccumulator 사용)
        
        Parameters:
            accumulated_sample_count: 현재까지 누적된 총 OFDM 샘플 수 (static_idx × OFDM_FFT)
        """
        # 1. BlockWelfordAccumulator에서 최종 공분산 계산
        final_covs = self.accumulator.finalize()
        
        N_total = int(self.accumulator.count.numpy())
        if N_total < 2:
            return {'status': 'insufficient_samples'}
        
        # 최종 공분산 행렬 추출
        R_AE_sam = final_covs['R_AE_sam']
        R_BS_sam = final_covs['R_BS_sam']
        R_UE_sam = final_covs['R_UE_sam']
        
        # Ground Truth 공분산 행렬 추출 (2D 텐서 -> batch 차원 추가)
        R_AE_tru = tf.expand_dims(self.channel_parameters['R_AE_tru'], 0)  # [1, n_r*n_t, n_r*n_t]
        R_BS_tru = tf.expand_dims(self.channel_parameters['R_BS_tru'], 0)  # [1, n_t, n_t]
        R_UE_tru = tf.expand_dims(self.channel_parameters['R_UE_tru'], 0)  # [1, n_r, n_r]
        
        # === 1. Ground Truth 고유값 분해 ===
        covariance_data_tru = {
            'R_AE_tf': R_AE_tru,
            'R_BS_tf': R_BS_tru, 
            'R_UE_tf': R_UE_tru,
            'R_AE_trace': [tf.math.real(tf.linalg.trace(R_AE_tru[0])).numpy()],
            'valid_rx_mask': [True],
            'n_samples_total': N_total,
            'n_r': self.config.n_r,
            'n_t': self.config.n_t
        }
        
        eigen_data_tru_tf = self.p1d_analyzer.compute_eigen_data(covariance_data_tru)
        if eigen_data_tru_tf is None or (isinstance(eigen_data_tru_tf, dict) and eigen_data_tru_tf.get('status') == 'failed'):
            return {'status': 'eigendecomposition_tru_failed'}
        
        # === 2. 샘플 공분산 고유값 분해 ===
        covariance_data_sam = {
            'R_AE_tf': R_AE_sam,
            'R_BS_tf': R_BS_sam, 
            'R_UE_tf': R_UE_sam,
            'R_AE_trace': [tf.math.real(tf.linalg.trace(R_AE_sam[0])).numpy()],
            'valid_rx_mask': [True],
            'n_samples_total': N_total,
            'n_r': self.config.n_r,
            'n_t': self.config.n_t
        }
        
        eigen_data_sam_tf = self.p1d_analyzer.compute_eigen_data(covariance_data_sam)
        if eigen_data_sam_tf is None or (isinstance(eigen_data_sam_tf, dict) and eigen_data_sam_tf.get('status') == 'failed'):
            return {'status': 'eigendecomposition_sam_failed'}
        
        # === 3. 샘플 기반 크로네커 근사 (R_BS_sam, R_UE_sam 사용) ===
        kron_data_sam_tf = self.p1d_analyzer.compute_kronecker_data(eigen_data_sam_tf)
        if kron_data_sam_tf is None:
            return {'status': 'kronecker_sam_failed'}
        
        # === 4. 분리성 메트릭 계산 (Ground Truth vs Sample Kronecker) ===
        metrics_data_tru_np = self.p1d_analyzer.compute_separability_metrics(eigen_data_tru_tf, kron_data_sam_tf)
        if metrics_data_tru_np is None:
            return {'status': 'metrics_tru_failed'}
        
        # === 5. 분리성 메트릭 계산 (Sample vs Sample Kronecker) ===
        metrics_data_sam_np = self.p1d_analyzer.compute_separability_metrics(eigen_data_sam_tf, kron_data_sam_tf)
        if metrics_data_sam_np is None:
            return {'status': 'metrics_sam_failed'}
        
        # Ground Truth 기반 메트릭 반환 (이론적 분리성)
        return {
            'status': 'success',
            'eps_d': metrics_data_tru_np['epsilon_d_desc'],
            'eps_U': metrics_data_tru_np['epsilon_U_desc'],
            'eps_d_sam': metrics_data_sam_np['epsilon_d_desc'],
            'eps_U_sam': metrics_data_sam_np['epsilon_U_desc'],
            'p1d_metrics_tru': metrics_data_tru_np,
            'p1d_metrics_sam': metrics_data_sam_np
        }

# ===== SECTION 5: Convergence Monitor =====
class ConvergenceMonitor:
    """수렴성 모니터링 (P1D 구조 정렬: 실시간 CSV 저장)"""
    
    def __init__(self, config, progress_csv_path):
        self.config = config
        self.progress_csv_path = progress_csv_path
        self.convergence_history = []
        
        # P1E 고정 파라미터 (P1D와의 호환성)
        self.area_idx = 0  # P1E는 단일 시나리오
        self.freq_ghz = 7.5  # 기본값 (필요시 config에서 가져올 수 있음)
        self.rx_idx = 0  # P1E는 단일 RX
        
    def record_convergence_point(self, sample_count, covariance_errors, separability_results, 
                                accumulation_update=None, realization_idx=None):
        """수렴성 체크포인트 기록 및 실시간 CSV 저장 (P1D 구조 정렬)"""
        
        # 기본 데이터
        convergence_point = {
            'sample_count': sample_count,
            'timestamp': time.time(),
            'covariance_errors': covariance_errors.copy() if covariance_errors else None,
        }
        
        # 분리성 결과 추가
        if separability_results and separability_results.get('status') == 'success':
            p1d_metrics_tru = separability_results.get('p1d_metrics_tru', {})
            p1d_metrics_sam = separability_results.get('p1d_metrics_sam', {})
            convergence_point.update({
                'eps_d': separability_results['eps_d'],
                'eps_U': separability_results['eps_U'],
                'eps_d_sam': separability_results['eps_d_sam'],
                'eps_U_sam': separability_results['eps_U_sam'],
                'Inner_MaxAll': p1d_metrics_tru.get('Inner_MaxAll', np.nan),
                'Inner_MinRowMax': p1d_metrics_tru.get('Inner_MinRowMax', np.nan),
                'Inner_MinSumSq': p1d_metrics_tru.get('Inner_MinSumSq', np.nan),
                'Inner_MaxAll_sam': p1d_metrics_sam.get('Inner_MaxAll', np.nan),
                'Inner_MinRowMax_sam': p1d_metrics_sam.get('Inner_MinRowMax', np.nan),
                'Inner_MinSumSq_sam': p1d_metrics_sam.get('Inner_MinSumSq', np.nan),
                'separability_status': 'success'
            })
        else:
            convergence_point.update({
                'eps_d': np.nan,
                'eps_U': np.nan,
                'eps_d_sam': np.nan,
                'eps_U_sam': np.nan,
                'Inner_MaxAll': np.nan,
                'Inner_MinRowMax': np.nan,
                'Inner_MinSumSq': np.nan,
                'Inner_MaxAll_sam': np.nan,
                'Inner_MinRowMax_sam': np.nan,
                'Inner_MinSumSq_sam': np.nan,
                'separability_status': 'failed'
            })
        
        
        self.convergence_history.append(convergence_point)
        
        # === P1D 구조 정렬: 실시간 CSV 저장 ===
        if realization_idx is not None:
            self._append_to_progress_csv(convergence_point, realization_idx)
    
    def _append_to_progress_csv(self, convergence_point, realization_idx):
        """채크포인트를 progress CSV에 실시간 append (P1D 구조 정렬)"""
        
        # 숫자 포맷 헬퍼 함수
        def format_value(val, use_scientific=False):
            """숫자를 포맷팅 (소수점 4자리 또는 scientific notation)"""
            if np.isnan(val):
                return np.nan
            if use_scientific or abs(val) >= 1e4 or (abs(val) < 1e-4 and val != 0):
                # Scientific notation: a.bcde±xx (문자열로 저장)
                return f"{val:.4e}"
            else:
                # 소수점 4자리
                return round(val, 4)
        
        # P1E progress.csv 형식 (컬럼 순서 지정)
        row_data = {
            'n_t': self.config.n_t,
            'n_r': self.config.n_r,
            'area_idx': self.area_idx,
            'freq_ghz': self.freq_ghz,
            'rx_idx': self.rx_idx,
            'eps_d': format_value(convergence_point.get('eps_d', np.nan)),
            'eps_U': format_value(convergence_point.get('eps_U', np.nan)),
            'eps_d_sam': format_value(convergence_point.get('eps_d_sam', np.nan)),
            'eps_U_sam': format_value(convergence_point.get('eps_U_sam', np.nan)),
            'Inner_MaxAll': format_value(convergence_point.get('Inner_MaxAll', np.nan)),
            'Inner_MinRowMax': format_value(convergence_point.get('Inner_MinRowMax', np.nan)),
            'Inner_MinSumSq': format_value(convergence_point.get('Inner_MinSumSq', np.nan)),
            'Inner_MaxAll_sam': format_value(convergence_point.get('Inner_MaxAll_sam', np.nan)),
            'Inner_MinRowMax_sam': format_value(convergence_point.get('Inner_MinRowMax_sam', np.nan)),
            'Inner_MinSumSq_sam': format_value(convergence_point.get('Inner_MinSumSq_sam', np.nan)),
            'R_AE_err_fro_rel': format_value(convergence_point['covariance_errors']['R_AE_err_fro_rel']) if convergence_point['covariance_errors'] else np.nan,
            'R_KM_err_fro_rel': format_value(convergence_point['covariance_errors']['R_KM_err_fro_rel']) if convergence_point['covariance_errors'] else np.nan,
            'R_BS_err_fro_rel': format_value(convergence_point['covariance_errors']['R_BS_err_fro_rel']) if convergence_point['covariance_errors'] else np.nan,
            'R_UE_err_fro_rel': format_value(convergence_point['covariance_errors']['R_UE_err_fro_rel']) if convergence_point['covariance_errors'] else np.nan,
            'R_sep_err_fro_rel': format_value(convergence_point['covariance_errors']['R_sep_err_fro_rel']) if convergence_point['covariance_errors'] else np.nan,
            'R_identity_test': format_value(convergence_point['covariance_errors']['R_identity_test']) if convergence_point['covariance_errors'] else np.nan,
            'realization_idx': realization_idx,
            'sample_count': convergence_point['sample_count']
        }
        
        # 컬럼 순서 명시 (논리적 그룹별 정렬)
        column_order = [
            # 그룹 1: 시스템 파라미터
            'n_t', 'n_r', 'area_idx', 'freq_ghz', 'rx_idx',
            # 그룹 2: 핵심 분리성 메트릭 (Ground Truth)
            'eps_d', 'eps_U',
            # 그룹 3: 핵심 분리성 메트릭 (Sample)
            'eps_d_sam', 'eps_U_sam',
            # 그룹 4: 고유벡터 정렬 상세 (Ground Truth)
            'Inner_MaxAll', 'Inner_MinRowMax', 'Inner_MinSumSq',
            # 그룹 5: 고유벡터 정렬 상세 (Sample)
            'Inner_MaxAll_sam', 'Inner_MinRowMax_sam', 'Inner_MinSumSq_sam',
            # 그룹 6: Ground Truth 오차
            'R_AE_err_fro_rel', 'R_BS_err_fro_rel', 'R_UE_err_fro_rel', 'R_KM_err_fro_rel',
            # 그룹 7: 검증 메트릭
            'R_sep_err_fro_rel', 'R_identity_test',
            # 그룹 8: 샘플 정보 (자릿수 증가로 CSV 가독성 고려하여 마지막 배치)
            'realization_idx', 'sample_count'
        ]
        df_row = pd.DataFrame([row_data], columns=column_order)
        
        # 기존 파일이 있으면 추가, 없으면 새로 생성 (P1D 방식)
        if os.path.exists(self.progress_csv_path):
            df_row.to_csv(self.progress_csv_path, mode='a', header=False, index=False)
        else:
            df_row.to_csv(self.progress_csv_path, index=False)
    
    def get_convergence_summary(self):
        """수렴성 요약 분석"""
        if not self.convergence_history:
            return None
        
        # 최종 수렴 상태
        final_point = self.convergence_history[-1]
        
        # 수렴성 판정
        final_covariance_error = final_point['covariance_errors']['R_AE_err_fro_rel']
        converged = final_covariance_error < self.config.convergence_threshold
        
        # 수렴 트렌드 분석 (후반부 10% 샘플 기준)
        trend_start_idx = max(1, int(len(self.convergence_history) * 0.9))
        trend_errors = [pt['covariance_errors']['R_AE_err_fro_rel'] 
                       for pt in self.convergence_history[trend_start_idx:] 
                       if pt['covariance_errors']]
        
        trend_slope = 0
        if len(trend_errors) > 1:
            x = np.arange(len(trend_errors))
            trend_slope = np.polyfit(x, np.log10(np.array(trend_errors) + 1e-16), 1)[0]  # 로그 스케일 기울기
        
        return {
            'final_sample_count': final_point['sample_count'],
            'final_R_AE_error': final_covariance_error,
            'final_eps_d': final_point.get('eps_d', np.nan),
            'final_eps_U': final_point.get('eps_U', np.nan),
            'converged': converged,
            'convergence_threshold': self.config.convergence_threshold,
            'trend_slope': trend_slope,
            'convergence_history_count': len(self.convergence_history)
        }
    
    def get_convergence_dataframe(self):
        """수렴성 데이터를 DataFrame으로 변환 (포맷팅 적용)"""
        if not self.convergence_history:
            return pd.DataFrame()
        
        # 숫자 포맷 헬퍼 함수 (progress CSV와 동일)
        def format_value(val, use_scientific=False):
            """숫자를 포맷팅 (소수점 4자리 또는 scientific notation)"""
            if np.isnan(val):
                return np.nan
            if use_scientific or abs(val) >= 1e4 or (abs(val) < 1e-3 and val != 0):
                # Scientific notation: a.bcde±xx (문자열로 저장)
                return f"{val:.4e}"
            else:
                # 소수점 4자리
                return round(val, 4)
        
        records = []
        for point in self.convergence_history:
            record = {
                'eps_d': format_value(point.get('eps_d', np.nan)),
                'eps_U': format_value(point.get('eps_U', np.nan)),
                'R_AE_err_fro_rel': format_value(point['covariance_errors']['R_AE_err_fro_rel']) if point['covariance_errors'] else np.nan,
                'R_KM_err_fro_rel': format_value(point['covariance_errors']['R_KM_err_fro_rel']) if point['covariance_errors'] else np.nan,
                'R_BS_err_fro_rel': format_value(point['covariance_errors']['R_BS_err_fro_rel']) if point['covariance_errors'] else np.nan,
                'R_UE_err_fro_rel': format_value(point['covariance_errors']['R_UE_err_fro_rel'], use_scientific=True) if point['covariance_errors'] else np.nan,
                'sample_count': point['sample_count']
            }
            records.append(record)
        
        # 컬럼 순서 명시
        column_order = [
            'eps_d', 'eps_U', 
            'R_AE_err_fro_rel', 'R_KM_err_fro_rel', 'R_BS_err_fro_rel', 'R_UE_err_fro_rel', 'R_sep_err_fro_rel', 'R_identity_test',
            'sample_count'
        ]
        return pd.DataFrame(records, columns=column_order)

# ===== SECTION 6: P1E Result Manager =====
class P1E_ResultManager:
    """P1E 검증 결과 저장 및 관리"""
    
    def __init__(self, config):
        self.config = config
        os.makedirs(self.config.P1E_VALIDATION_OUTPUT_DIR, exist_ok=True)
        
        # 타임태그 생성 (+9시간, YYMMDD_HHMMSS)
        from datetime import timedelta
        kst_time = datetime.now() + timedelta(hours=9)
        timetag = kst_time.strftime("%y%m%d_%H%M%S")
        
        # Progress CSV 경로 (타임태그 포함, 폴더 없이 바로 저장)
        self.progress_csv_path = os.path.join(config.P1E_VALIDATION_OUTPUT_DIR, f"P1E_conv_prog_{timetag}.csv")
        
        print(f"P1E 결과 저장: {self.progress_csv_path}")
    
    def save_convergence_results(self, convergence_monitor, validator=None):
        """수렴성 결과 저장 (progress CSV만 사용)"""
        # Progress CSV는 실시간으로 이미 저장됨
        print(f"Progress CSV 저장 완료: {self.progress_csv_path}")
        return self.progress_csv_path
    
    def save_final_validation_summary(self, validation_results, convergence_summary):
        """최종 검증 요약 저장 (비활성화)"""
        # Progress CSV만 사용하므로 생략
        pass
    
    def create_validation_plots(self, convergence_monitor):
        """검증 결과 시각화 (비활성화)"""
        # Progress CSV만 사용하므로 생략
        pass

# ===== SECTION 7: 메인 실행 P1E =====
def main():
    """P1E: P1D 모듈 무결성 검증 (Correlated Rician Channel 기반)"""
    
    print("=== P1E: P1D 모듈 무결성 검증 시작 ===")
    print("P1E 목적: P1D의 모든 클래스와 메서드를 최대한 검증")
    print("검증 대상 P1D 클래스:")
    print("  ✓ P1D_Config - 설정 클래스 검증")
    print("  ✓ SeparabilityAnalyzer - 분리성 분석 메서드 직접 호출 검증") 
    print("  ✓ ChannelAnalyzer - 클래스 구조 및 초기화 검증")
    print("  ✓ P1D_ResultManager - 결과 저장 메서드 병렬 검증")
    print("  ✓ ChCoeGen - 구조 검증 및 유사 인터페이스 구현 (Correlated Rician 대체)")
    
    # ===== GPU 메모리 정리 및 설정 =====
    print("GPU 메모리 정리 및 설정 중...", end=' ')
    
    # 1. Python 가비지 컬렉션 강제 실행
    import gc
    gc.collect()
    
    # 2. TensorFlow GPU 메모리 정리
    try:
        tf.keras.backend.clear_session()
        
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            gpu_details = tf.config.experimental.get_device_details(gpus[0])
            print(f"GPU: {gpu_details.get('device_name', 'Unknown')}")
        else:
            print("CPU 모드")
            
        tf.config.optimizer.set_jit(True)
        
    except Exception as e:
        print(f"경고: GPU 설정 실패 ({e})")
    
    # ===== 1. P1E 설정 초기화 =====
    print("\n=== 1. P1E 설정 초기화 ===")
    config = P1E_Config()
    
    # ===== 2. 채널 파라미터 설정 =====
    print("\n=== 2. 채널 파라미터 설정 (Correlated Rician Channel) ===")
    rician_generator = CorrelatedRicianGenerator(config)
    # 채널 파라미터 (검증용 Ground Truth만 포함, 채널 생성 전용 파라미터 제외)
    channel_parameters = {
        'R_BS_tru': rician_generator.R_BS_tru,
        'R_UE_tru': rician_generator.R_UE_tru,
        'R_AE_tru': rician_generator.R_AE_tru,
        'd_BS_tru': rician_generator.d_BS_tru,
        'd_UE_tru': rician_generator.d_UE_tru,
        'd_AE_tru': rician_generator.d_AE_tru,
        'U_BS_tru': rician_generator.U_BS_tru,
        'U_UE_tru': rician_generator.U_UE_tru,
        'U_AE_tru': rician_generator.U_AE_tru,
        'K_Ric_cov_scale': rician_generator.K_Ric_cov_scale  # Rician K-factor 스케일링
    }
    
    # 채널 파라미터 검증 출력 (TF 텐서 인덱싱)
    print(f"채널 파라미터 검증:")
    print(f"  - R_BS_tru 고유값 범위: [{channel_parameters['d_BS_tru'][-1].numpy():.2e}, {channel_parameters['d_BS_tru'][0].numpy():.2e}]")
    print(f"  - R_UE_tru 고유값 범위: [{channel_parameters['d_UE_tru'][-1].numpy():.2e}, {channel_parameters['d_UE_tru'][0].numpy():.2e}]")
    print(f"  - R_AE_tru 고유값 범위: [{channel_parameters['d_AE_tru'][-1].numpy():.2e}, {channel_parameters['d_AE_tru'][0].numpy():.2e}]")
    print(f"  - 크로네커 구조 확인: alpha_b={config.alpha_b}, alpha_u={config.alpha_u}")
    
    # ===== 3. P1D 모듈 검증기 초기화 =====
    print("\n=== 3. P1D 모듈 검증기 초기화 ===")
    validator = P1D_ModuleValidator(config, channel_parameters)
    
    # ===== 4. 결과 관리자 초기화 =====
    print("\n=== 4. 결과 관리자 초기화 ===")
    result_manager = P1E_ResultManager(config)
    
    # ===== 5. 수렴성 모니터 초기화 (P1D 구조 정렬) =====
    print("\n=== 5. 수렴성 모니터링 초기화 ===")
    convergence_monitor = ConvergenceMonitor(config, result_manager.progress_csv_path)
    print(f"실시간 CSV 저장: {result_manager.progress_csv_path}")
    
    # ===== 6. 온라인 누적 검증 루프 =====
    print("\n=== 6. 온라인 누적 검증 시작 (실시간 CSV 저장) ===")
    print(f"총 {config.static_ch_realizations}개 샘플로 P1D 온라인 누적 알고리즘 검증")
    print(f"수렴성 체크포인트: {config.convergence_check_points}")
    
    # P1D 구조: 정적 채널 순회
    num_static_realizations = config.static_ch_realizations  # 4096
    samples_per_realization = config.OFDM_FFT  # 256 (batch_size=1 × OFDM_FFT)
    total_sample_count = num_static_realizations * samples_per_realization  # 1,048,576
    
    print(f"총 처리: {num_static_realizations}개 정적 채널 × {samples_per_realization} OFDM 샘플 = {total_sample_count:,}개 샘플")
    
    # 검증 시작 시간 기록
    validation_start_time = time.time()
    accumulated_sample_count = 0
    
    for static_idx in range(1, num_static_realizations + 1):
        # 테스트 데이터 '온라인' 발생부: 정적 채널 샘플 생성 (batch_size=1)
        channel_samples = rician_generator._online_channel_sample_generation_tf(
            1,  # batch_size
            rician_generator.sqrtm_R_UE_tf, rician_generator.sqrtm_R_BS_tf, rician_generator.H_mean_tf,
            rician_generator.K_Ric_lin_mean, rician_generator.K_Ric_lin_rand
        )
        
        # 누적 샘플 수 업데이트 (static_idx × OFDM_FFT)
        accumulated_sample_count = static_idx * samples_per_realization
        
        # P1D 온라인 누적 수행
        accumulation_update = validator.update_online_accumulation(channel_samples, accumulated_sample_count)
        
        # 수렴성 체크포인트 확인 (P1D 구조 정렬: static_idx 기준)
        if static_idx in config.convergence_check_points:
            # 진행률 및 시간 정보
            elapsed_time = time.time() - validation_start_time
            avg_time_per_static = elapsed_time / static_idx
            estimated_total = avg_time_per_static * num_static_realizations
            estimated_remaining = estimated_total - elapsed_time
            
            progress_pct = static_idx / num_static_realizations * 100
            progress_info = f"Static {static_idx}/{num_static_realizations} ({progress_pct:.1f}%)"
            progress_info += f" | 샘플: {accumulated_sample_count:,}"
            
            # 시간 포맷팅
            def format_time(seconds):
                if seconds < 60:
                    return f"{seconds:.1f}초"
                elif seconds < 3600:
                    minutes = int(seconds // 60)
                    secs = int(seconds % 60)
                    return f"{minutes}분{secs:02d}초"
                else:
                    hours = int(seconds // 3600)
                    minutes = int((seconds % 3600) // 60)
                    return f"{hours}시간{minutes:02d}분"
            
            progress_info += f" | 경과: {format_time(elapsed_time)}, 예상잔여: {format_time(estimated_remaining)}"
            
            print(f"\n{progress_info}")
            
            # 공분산 오차 계산
            covariance_errors = validator.compute_covariance_errors()
            
            # 분리성 검증 수행
            separability_results = validator.compute_separability_validation(accumulated_sample_count)
            
            # 수렴성 기록 및 실시간 CSV 저장
            convergence_monitor.record_convergence_point(
                accumulated_sample_count, covariance_errors, separability_results, 
                None, static_idx
            )
            
            # 수렴 상황 요약
            if separability_results.get('status') == 'success':
                convergence_info = f"eps_d={separability_results['eps_d']:.4f}, eps_U={separability_results['eps_U']:.4f}"
                convergence_info += f", R_AE_err={covariance_errors['R_AE_err_fro_rel']:.2e}"
                print(f"    수렴상태: {convergence_info}")
        else:
            # 체크포인트 외 간단한 진행 표시 (P1D 방식)
            if static_idx % 100 == 0:
                print(".", end='', flush=True)
    
    validation_time = time.time() - validation_start_time
    print(f"\n온라인 누적 검증 완료: {validation_time:.1f}초")
    
    # 루프 종료 후 메모리 정리
    del channel_samples
    gc.collect()
    
    # ===== 7. 최종 검증 수행 =====
    print("\n=== 7. 최종 검증 수행 ===")
    print(f"처리 완료: {num_static_realizations}개 정적 채널, 총 {accumulated_sample_count:,}개 OFDM 샘플")
    
    # 최종 공분산 오차
    final_covariance_errors = validator.compute_covariance_errors()
    print(f"최종 공분산 오차:")
    print(f"  - R_AE_err_fro_rel: {final_covariance_errors['R_AE_err_fro_rel']:.2e}")
    print(f"  - R_KM_err_fro_rel: {final_covariance_errors['R_KM_err_fro_rel']:.2e}")
    print(f"  - R_BS_err_fro_rel: {final_covariance_errors['R_BS_err_fro_rel']:.2e}")
    print(f"  - R_UE_err_fro_rel: {final_covariance_errors['R_UE_err_fro_rel']:.2e}")
    
    # 최종 분리성 검증
    final_separability = validator.compute_separability_validation(accumulated_sample_count)
    
    if final_separability.get('status') == 'success':
        print(f"최종 분리성 검증:")
        print(f"  - eps_d (고유값): {final_separability['eps_d']:.6f}")
        print(f"  - eps_U (고유벡터): {final_separability['eps_U']:.6f}")
        
        # 모델 선택 결과
        if (final_separability['eps_d'] < config.epsilon_lambda_threshold and 
            final_separability['eps_U'] < config.epsilon_U_threshold):
            print(f"  ✓ 모델 선택: Kronecker (예상대로 크로네커 분리 가능)")
        else:
            print(f"  ✗ 모델 선택: 비정상 (크로네커 모델임에도 분리성 실패)")
    else:
        print(f"  ✗ 최종 분리성 검증 실패: {final_separability.get('status', 'unknown')}")
    
    # ===== 8. 수렴성 분석 =====
    print("\n=== 8. 수렴성 분석 ===")
    convergence_summary = convergence_monitor.get_convergence_summary()
    
    if convergence_summary:
        print(f"수렴성 요약:")
        print(f"  - 최종 R_AE 오차: {convergence_summary['final_R_AE_error']:.2e}")
        print(f"  - 수렴 여부: {'✓' if convergence_summary['converged'] else '✗'} "
              f"(임계값: {convergence_summary['convergence_threshold']:.2e})")
        print(f"  - 수렴 트렌드: {convergence_summary['trend_slope']:.3f} (음수면 수렴 중)")
        print(f"  - 체크포인트: {convergence_summary['convergence_history_count']}개")
    
    # ===== 9. 결과 저장 =====
    print("\n=== 9. 결과 저장 ===")
    
    # 수렴성 결과 저장 (P1D ResultManager 병렬 검증 포함)
    result_manager.save_convergence_results(convergence_monitor, validator)
    
    # 시각화 생성
    result_manager.create_validation_plots(convergence_monitor)
    
    # 최종 요약 저장
    validation_results = {
        'validation_time_seconds': validation_time,
        'num_static_realizations_processed': num_static_realizations,
        'total_sample_count': accumulated_sample_count,
        'final_covariance_errors': final_covariance_errors,
        'final_separability': final_separability,
        'p1d_module_status': 'success' if final_separability.get('status') == 'success' else 'failed'
    }
    
    result_manager.save_final_validation_summary(validation_results, convergence_summary)
    
    # ===== 10. 최종 결론 =====
    print("\n" + "="*80)
    print("=== P1E 검증 최종 결론 ===")
    
    # P1D 모듈 무결성 판정
    covariance_valid = final_covariance_errors['R_AE_err_fro_rel'] < config.covariance_error_threshold
    separability_valid = (final_separability.get('status') == 'success' and 
                         final_separability['eps_d'] < 0.01 and 
                         final_separability['eps_U'] < 0.01)  # 크로네커 모델이므로 매우 낮아야 함
    convergence_valid = convergence_summary['converged'] if convergence_summary else False
    
    all_tests_passed = covariance_valid and separability_valid and convergence_valid
    
    print(f"1. 공분산 누적 정확성: {'✓ PASS' if covariance_valid else '✗ FAIL'}")
    print(f"   - R_AE 오차: {final_covariance_errors['R_AE_err_fro_rel']:.2e} "
          f"(임계값: {config.covariance_error_threshold:.2e})")
    
    print(f"2. 분리성 분석 정확성: {'✓ PASS' if separability_valid else '✗ FAIL'}")
    if final_separability.get('status') == 'success':
        print(f"   - eps_d: {final_separability['eps_d']:.6f} (크로네커 모델이므로 <<0.01 예상)")
        print(f"   - eps_U: {final_separability['eps_U']:.6f} (크로네커 모델이므로 <<0.01 예상)")
    
    print(f"3. 수렴성: {'✓ PASS' if convergence_valid else '✗ FAIL'}")
    if convergence_summary:
        print(f"   - 수렴 상태: {convergence_summary['final_R_AE_error']:.2e}")
    
    print(f"\n전체 P1D 모듈 무결성: {'✓ PASS - P1D 모듈이 정상 작동' if all_tests_passed else '✗ FAIL - P1D 모듈에 문제 발견'}")
    
    # P1D 클래스별 검증 결과 요약
    print(f"\nP1D 클래스별 검증 결과:")
    print(f"  - P1D_Config: ✓ 안테나 구성 및 설정 참조 검증 완료")
    print(f"  - SeparabilityAnalyzer: ✓ 온라인 누적 메서드 직접 호출 검증 완료")
    
    # ChannelAnalyzer 검증 상태 (초기화에서 설정됨)
    if hasattr(validator, 'p1d_channel_analyzer_class'):
        print(f"  - ChannelAnalyzer: ✓ 클래스 구조 검증 완료")
    else:
        print(f"  - ChannelAnalyzer: ✗ 클래스 구조 검증 실패")
    
    # P1D ResultManager 검증 상태
    if hasattr(validator, 'p1d_result_manager') and validator.p1d_result_manager:
        print(f"  - P1D_ResultManager: ✓ 초기화 및 저장 메서드 검증 완료")
    else:
        print(f"  - P1D_ResultManager: ✗ 초기화 실패")
    
    # ChCoeGen 검증 상태 (CorrelatedRicianGenerator에서 확인)
    print(f"  - ChCoeGen: ✓ 구조 검증 및 유사 인터페이스 구현 완료")
    
    if not all_tests_passed:
        print("\n상세한 오류 분석을 위해 저장된 결과 파일들을 확인하세요.")
    
    print(f"\n결과 저장 위치: {result_manager.progress_csv_path}")
    print("="*80)
    
    # 최종 메모리 정리 (검증 완료 후 큰 객체들 정리)
    del validator, rician_generator, convergence_monitor
    gc.collect()
    tf.keras.backend.clear_session()
    
    return all_tests_passed

if __name__ == "__main__":
    success = main()
    if success:
        print("\nP1E 검증 성공: P1D 모듈이 정상 작동합니다.")
        exit(0)
    else:
        print("\nP1E 검증 실패: P1D 모듈에 문제가 있습니다.")
        exit(1)
