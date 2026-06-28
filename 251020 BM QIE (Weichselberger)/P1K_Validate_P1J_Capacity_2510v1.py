# ======================================================================
# P1K_Validate_P1J_Capacity_2510v1.py
# P1K: P1J 점근적 용량 검증 (Wen2011 vs Monte Carlo)
# 
# === 최상위 목적 ===
# P1J 점근적 최적화 결과를 실제 채널 샘플 기반 Monte Carlo 평균과 비교 검증
# - 점근적 용량: Wen2011 식 (7) I(P*) with fixed-point (γ,ψ)
# - Monte Carlo 평균: E[log det(I + H P* H^H)] ≈ (1/n) Σ log det(I + H_i P* H_i^H)
# 
# === 입력/출력 ===
# 입력 1: P1J 결과 (Area{area}_{freq}GHz_UE{ue}_Capacity.npz)
#   - P_opt: [n_ue, n_ue] 최적 입력 공분산
#   - kappa, power_total, metadata
# 
# 입력 2: P1I 청크 (Area{area}_{freq}GHz_Weichsel_Chunk_{idx}_UE{start}-{end}.npz)
#   - U_BS, U_UE: 고유벡터
#   - Omega: 커플링 행렬
#   - H_mean: LoS 성분
# 
# 출력: Area{area}_{freq}GHz_Validation_{timestamp}.csv
#   - ue, I_asym, C_mc, C_std, rel_err, n_samp, kappa, ch_model, time_s
#
# === 주요 수정 이력 ===
# [251016] P1K 신규 작성: P1J 점근적 용량 검증 (Wen2011 vs Monte Carlo)
#
# ======================================================================

# ===== SECTION 1: 환경 설정 =====
import os
import time
from datetime import datetime, timedelta
import numpy as np
import glob
import re
import csv
from pathlib import Path

# TensorFlow 환경 설정
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc'
gpu_num = 0
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

# TensorFlow 코어 임포트
import tensorflow as tf

# GPU 메모리 설정
gpus = tf.config.list_physical_devices('GPU')
for g in gpus:
    try:
        tf.config.experimental.set_memory_growth(g, True)
        print(f"GPU 메모리 설정: 동적 증가 모드")
    except RuntimeError as e:
        print(f"GPU 설정 경고: {e}")

# TensorFlow 최적화 설정
tf.get_logger().setLevel("ERROR")
tf.config.optimizer.set_jit(True)
tf.random.set_seed(42)
np.random.seed(42)

# ===== SECTION 2: P1K_Config =====
class P1K_Config:
    """P1K 검증 설정"""
    
    def __init__(self):
        # 스크립트의 디렉토리를 기준으로 절대 경로 설정
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # P1J 용량 결과 입력
        self.P1J_INPUT_DIR = os.path.join(script_dir, "P1J_Capacity_Results")
        self.P1J_FILE_PATTERN = "Area{area}_{freq}GHz_UE{ue}_Capacity.npz"
        
        # P1I 청크 입력 (채널 파라미터)
        self.P1I_INPUT_DIR = os.path.join(script_dir, "P1I_Weichsel_Chunk_Results")
        self.P1I_CHUNK_PATTERN = "Area{area}_{freq}GHz_Weichsel_Chunk_{idx}_UE{start}-{end}.npz"
        
        # P1K 검증 결과 출력
        self.P1K_OUTPUT_DIR = os.path.join(script_dir, "P1K_Validation_Results")
        os.makedirs(self.P1K_OUTPUT_DIR, exist_ok=True)
        
        # 필터링 설정
        self.target_areas = [1]  # 처리할 area 목록 (None이면 전체)
        self.target_freqs = None  # 처리할 주파수 목록 (None이면 전체)
        
        # 검증 설정
        self.n_samples_monte_carlo = 1000  # Monte Carlo 샘플 수
        self.batch_size_channel = 1000      # 메모리 효율을 위한 배치 크기
        
        # 점근적 용량 계산 설정 (고정점 계산)
        self.eps_inner = 1e-4
        self.max_iter_inner = 200
        self.regularization = 1e-20
        
        # SNR 및 전력 제약 설정 (P1J와 동일)
        self.SNR_dB = 10.0
        self.SNR_linear = 10.0**(self.SNR_dB / 10.0)
        
        # P1J 결과 스캔
        self.detect_p1j_results()
    
    def detect_p1j_results(self):
        """P1J 결과 파일 스캔"""
        
        scan_pattern = f"{self.P1J_INPUT_DIR}/*.npz"
        p1j_files = glob.glob(scan_pattern)
        
        # 파일명 파싱용 정규식
        pattern = r'Area(\d+)_(.+)GHz_UE(\d+)_Capacity\.npz'
        
        p1j_candidates = []
        
        for filepath in p1j_files:
            filename = os.path.basename(filepath)
            match = re.match(pattern, filename)
            if match:
                area = int(match.group(1))
                freq = float(match.group(2))
                ue = int(match.group(3))
                
                # 필터링 적용
                if self.target_areas is not None and area not in self.target_areas:
                    continue
                if self.target_freqs is not None and freq not in self.target_freqs:
                    continue
                
                p1j_candidates.append((area, freq, ue, filepath))
        
        # 정렬 (area, freq, ue 순)
        p1j_candidates.sort(key=lambda x: (x[0], x[1], x[2]))
        
        self.p1j_results = p1j_candidates
        
        if p1j_candidates:
            area_groups = {}
            for area, freq, ue, _ in p1j_candidates:
                key = f"Area{area}_{freq}GHz"
                area_groups[key] = area_groups.get(key, 0) + 1
            
            print(f"P1J 결과 파일 스캔: 총 UE {len(p1j_candidates)}개 | Area별 상세:")
            for area_freq, count in sorted(area_groups.items()):
                print(f"  - {area_freq}: {count}개 UE")
        else:
            print("경고: P1J 결과 파일이 없습니다.")

# ===== SECTION 3: P1K_CapacityValidator =====
class P1K_CapacityValidator:
    """점근적 용량 vs Monte Carlo 평균 용량 검증"""
    
    def __init__(self, config: P1K_Config):
        self.config = config
        self.p1i_chunk_cache = {}  # 청크 파일 캐시
    
    def load_p1j_result(self, filepath: str) -> dict:
        """P1J 결과 로딩
        
        Returns:
            dict: {
                'P_opt': [n_ue, n_ue],
                'sum_rate_opt': float,
                'kappa': float,
                'power_total': float,
                'metadata': dict
            }
        """
        data = np.load(filepath, allow_pickle=True)
        
        result = {
            'P_opt': data['P_opt'],
            'sum_rate_opt': float(data['sum_rate_opt']),
            'kappa': float(data.get('kappa', 0.0)) if data.get('kappa') is not None else 0.0,
            'power_total': float(data.get('power_total', 0.0)) if data.get('power_total') is not None else 0.0,
            'metadata': data['metadata'].item()
        }
        
        return result
    
    def find_p1i_chunk_for_ue(self, area: int, freq: float, ue: int) -> tuple:
        """UE가 포함된 P1I 청크 파일 찾기 (파일명 파싱만)
        
        Returns:
            (chunk_filepath, estimated_idx) or (None, None)
        """
        # 청크 파일 스캔
        scan_pattern = f"{self.config.P1I_INPUT_DIR}/Area{area}_{freq}GHz_Weichsel_Chunk_*.npz"
        chunk_files = glob.glob(scan_pattern)
        
        pattern = r'Area(\d+)_(.+)GHz_Weichsel_Chunk_(\d+)_UE(\d+)-(\d+)\.npz'
        for filepath in chunk_files:
            filename = os.path.basename(filepath)
            match = re.match(pattern, filename)
            if match:
                ue_start = int(match.group(4))
                ue_end = int(match.group(5))
                
                # 파일명 범위로만 판단 (로딩 안 함)
                if ue_start <= ue <= ue_end:
                    # 추정 인덱스: 파일 내에서의 상대 위치
                    estimated_idx = ue - ue_start
                    return filepath, estimated_idx
        
        return None, None
    
    
    def load_p1i_channel_params(self, area: int, freq: float, ue: int) -> dict:
        """P1I 청크에서 채널 파라미터 로딩
        
        Returns:
            dict: {
                'U_bs': [n_bs, n_bs],
                'U_ue': [n_ue, n_ue],
                'Omega': [n_bs, n_ue],
                'H_mean': [n_bs, n_ue],
                'channel_model': str,
                'has_coupling': bool
            }
        """
        chunk_filepath, estimated_idx = self.find_p1i_chunk_for_ue(area, freq, ue)
        
        if chunk_filepath is None:
            raise FileNotFoundError(f"P1I 청크를 찾을 수 없습니다: Area{area}_{freq}GHz UE{ue}")
        
        # 청크 데이터 로딩 (캐시 활용)
        if chunk_filepath not in self.p1i_chunk_cache:
            data = np.load(chunk_filepath, allow_pickle=True)
            self.p1i_chunk_cache[chunk_filepath] = data
        else:
            data = self.p1i_chunk_cache[chunk_filepath]
        
        # 정확한 UE 인덱스 확인
        ue_indices = data['ue_indices'].tolist()
        if ue not in ue_indices:
            raise ValueError(f"UE{ue}가 청크에 없습니다: {chunk_filepath}")
        idx_in_chunk = ue_indices.index(ue)
        
        # UE 데이터 추출 (DL 기준)
        U_bs = data['P1G_U_BS'][idx_in_chunk]  # [n_bs, n_bs]
        U_ue = data['P1G_U_UE'][idx_in_chunk]  # [n_ue, n_ue]
        Omega_dl = data['P1G_Omega'][idx_in_chunk]  # [n_ue, n_bs] DL
        H_mean_dl = data['P1H_H_mean'][idx_in_chunk]  # [n_ue, n_bs] DL
        has_coupling = bool(data['P1G_has_coupling'][idx_in_chunk])
        channel_model = data['enhanced_metadata'][idx_in_chunk]['channel_model']
        
        # DL → UL 전환 (P1J와 동일)
        Omega_ul = Omega_dl.T  # [n_bs, n_ue]
        H_mean_ul = H_mean_dl.conj().T  # [n_bs, n_ue] Hermitian transpose
        
        # Wen2011 채널 정규화 적용 (P1J와 동일)
        Omega_norm, H_mean_norm, kappa = self._normalize_channel(
            Omega_ul, H_mean_ul, U_bs.shape[0], U_ue.shape[0]
        )
        
        return {
            'U_bs': U_bs,
            'U_ue': U_ue,
            'Omega': Omega_norm,
            'H_mean': H_mean_norm,
            'kappa_computed': kappa,
            'channel_model': channel_model,
            'has_coupling': has_coupling
        }
        
    @tf.function
    def _normalize_channel_tf(self, Omega_tf: tf.Tensor, H_mean_tf: tf.Tensor, 
                              rho_target: float, N_M: float) -> tuple:
        """Wen2011 식 (4) 채널 정규화 (TensorFlow)
        
        ||Ω||_1/(N·M) = ρ/(κ+1), tr(H̄H̄^H)/(N·M) = κρ/(κ+1)
        
        Returns:
            (Omega_norm, H_mean_norm, kappa)
        """
        epsilon = 1e-30
        threshold = 1e-10
        
        # 1. 현재 채널 전력 계산
        P_nlos_current = tf.reduce_sum(Omega_tf)  # ||Ω||_1
        P_los_current = tf.reduce_sum(tf.square(tf.abs(H_mean_tf)))  # tr(H̄H̄^H)
        
        # 2. 케이스 구분
        is_nlos_only = P_los_current < threshold * P_nlos_current
        is_los_only = P_nlos_current < threshold * P_los_current
        
        # 3. 조건부 계산
        def nlos_only():
            kappa = 0.0
            Omega_norm = (N_M * rho_target / (P_nlos_current + epsilon)) * Omega_tf
            H_mean_norm = tf.zeros_like(H_mean_tf)
            return Omega_norm, H_mean_norm, kappa
        
        def los_only():
            kappa = float('inf')
            Omega_norm = tf.zeros_like(Omega_tf)
            scale = tf.sqrt(N_M * rho_target / (P_los_current + epsilon))
            H_mean_norm = tf.cast(scale, tf.complex64) * H_mean_tf
            return Omega_norm, H_mean_norm, kappa
        
        def mixed():
            kappa = P_los_current / P_nlos_current
            P_nlos_target = N_M * (1.0 / (kappa + 1.0)) * rho_target
            P_los_target = N_M * (kappa / (kappa + 1.0)) * rho_target
            
            Omega_norm = (P_nlos_target / P_nlos_current) * Omega_tf
            scale = tf.sqrt(P_los_target / P_los_current)
            H_mean_norm = tf.cast(scale, tf.complex64) * H_mean_tf
            return Omega_norm, H_mean_norm, kappa
        
        # TensorFlow 조건 분기
        Omega_norm, H_mean_norm, kappa = tf.cond(
            is_nlos_only,
            nlos_only,
            lambda: tf.cond(is_los_only, los_only, mixed)
        )
        
        return Omega_norm, H_mean_norm, kappa
    
    def _normalize_channel(self, Omega: np.ndarray, H_mean: np.ndarray, 
                           n_bs: int, n_ue: int) -> tuple:
        """채널 정규화 wrapper (NumPy → TensorFlow → NumPy)"""
        Omega_tf = tf.cast(Omega, tf.float32)
        H_mean_tf = tf.cast(H_mean, tf.complex64)
        
        rho_target = float(self.config.SNR_linear)
        N_M = float(n_bs * n_ue)
        
        Omega_norm_tf, H_mean_norm_tf, kappa = self._normalize_channel_tf(
            Omega_tf, H_mean_tf, rho_target, N_M
        )
        
        return Omega_norm_tf.numpy(), H_mean_norm_tf.numpy(), float(kappa)
    
    @tf.function
    def _compute_fixed_point(self, P: tf.Tensor, 
                             U_bs: tf.Tensor, 
                             U_ue: tf.Tensor, 
                             Omega: tf.Tensor, 
                             H_mean: tf.Tensor,
                             eps_inner: float,
                             reg: float,
                             max_iter: int) -> tuple:
        """고정점 계산 (최적 수렴점 선택)
        
        argmin(diff/iter): 초기 행운이 아닌 후반부 수렴성 기반 선택
        
        Returns:
            (gamma, psi): [n_bs], [n_ue]
        """
        n_bs = tf.shape(U_bs)[0]
        n_ue = tf.shape(U_ue)[0]
        
        # 초기화
        gamma = tf.ones(n_bs, dtype=tf.float32)
        psi = tf.ones(n_ue, dtype=tf.float32)
        
        # regularization을 complex64로 캐스팅
        reg_complex = tf.cast(reg, tf.complex64)
        
        # 모든 iteration의 (gamma, psi)와 diff 저장
        gamma_history = tf.TensorArray(tf.float32, size=max_iter, dynamic_size=False)
        psi_history = tf.TensorArray(tf.float32, size=max_iter, dynamic_size=False)
        diff_history = tf.TensorArray(tf.float32, size=max_iter, dynamic_size=False)
        
        # 고정 반복
        for i in tf.range(max_iter):
            gamma_prev = tf.identity(gamma)
            psi_prev = tf.identity(psi)
            
            # (10) T, R 계산
            Omega_T_gamma = tf.matmul(tf.transpose(Omega), tf.expand_dims(gamma, -1))
            T = U_ue @ tf.linalg.diag(tf.cast(tf.squeeze(Omega_T_gamma), tf.complex64)) @ tf.linalg.adjoint(U_ue)
            
            Omega_psi = tf.matmul(Omega, tf.expand_dims(psi, -1))
            R = U_bs @ tf.linalg.diag(tf.cast(tf.squeeze(Omega_psi), tf.complex64)) @ tf.linalg.adjoint(U_bs)
            
            # (8) Xi 계산
            I_R = tf.eye(n_bs, dtype=tf.complex64) + R + reg_complex * tf.eye(n_bs, dtype=tf.complex64)
            H_mean_H = tf.linalg.adjoint(H_mean)
            Xi = T + H_mean_H @ tf.linalg.solve(I_R, H_mean)
            
            # (11a) gamma 업데이트: gamma_n = u_{bs,n}^H (I+R)^{-1} u_{bs,n}
            # solve(I_R, U_bs)가 inv(I_R) @ U_bs보다 수치적으로 안정
            I_R_inv_U_bs = tf.linalg.solve(I_R, U_bs)
            gamma = tf.math.real(
                tf.reduce_sum(tf.math.conj(U_bs) * I_R_inv_U_bs, axis=0)
            )
            
            # (11b) psi 업데이트: psi_m = u_{ue,m}^H P (I+Ξ P)^{-1} u_{ue,m}
            # solve가 inv보다 수치적으로 안정
            I_Xi_P = tf.eye(n_ue, dtype=tf.complex64) + Xi @ P + reg_complex * tf.eye(n_ue, dtype=tf.complex64)
            I_Xi_P_inv_U_ue = tf.linalg.solve(I_Xi_P, U_ue)
            psi = tf.math.real(
                tf.reduce_sum(tf.math.conj(U_ue) * (P @ I_Xi_P_inv_U_ue), axis=0)
            )
            
            # 수렴도 계산 및 저장
            gamma_diff = tf.norm(gamma - gamma_prev)
            psi_diff = tf.norm(psi - psi_prev)
            total_diff = gamma_diff + psi_diff
            
            gamma_history = gamma_history.write(i, gamma)
            psi_history = psi_history.write(i, psi)
            diff_history = diff_history.write(i, total_diff)
        
        # argmin(diff/iter): 수렴 효율성 기반 선택
        all_diffs = diff_history.stack()
        iter_indices = tf.cast(tf.range(1, max_iter + 1), tf.float32)
        diff_per_iter = all_diffs / iter_indices
        best_idx = tf.argmin(diff_per_iter, output_type=tf.int32)
        
        # 최적 (gamma, psi) 추출
        all_gammas = gamma_history.stack()
        all_psis = psi_history.stack()
        
        best_gamma = all_gammas[best_idx]
        best_psi = all_psis[best_idx]
        
        return best_gamma, best_psi
    
    @tf.function
    def _generate_channel_batch_tf(self, U_bs: tf.Tensor, U_ue: tf.Tensor,
                                    Omega_sqrt: tf.Tensor, H_mean: tf.Tensor,
                                    batch_size: int, n_bs: int, n_ue: int) -> tf.Tensor:
        """채널 샘플 배치 생성 (TensorFlow)
        
        Returns:
            H_batch: [batch_size, n_bs, n_ue]
        """
        # W ~ CN(0,1): [batch_size, n_bs, n_ue]
        sqrt_2 = tf.sqrt(tf.constant(2.0, dtype=tf.float32))
        W_real = tf.random.normal([batch_size, n_bs, n_ue], dtype=tf.float32) / sqrt_2
        W_imag = tf.random.normal([batch_size, n_bs, n_ue], dtype=tf.float32) / sqrt_2
        W = tf.complex(W_real, W_imag)
        
        # Element-wise product: sqrt(Omega) ⊙ W
        G_W = Omega_sqrt[None, :, :] * W  # broadcasting
        
        # H = U_bs @ G_W @ U_ue^H + H_mean
        # 배치 행렬 곱: [batch, n_bs, n_bs] @ [batch, n_bs, n_ue] @ [n_ue, n_ue]^H
        U_ue_H = tf.linalg.adjoint(U_ue)
        
        # einsum으로 배치 계산
        U_bs_G_W = tf.einsum('ij,bjk->bik', U_bs, G_W)  # [batch, n_bs, n_ue]
        H_batch = tf.einsum('bij,kj->bik', U_bs_G_W, tf.math.conj(U_ue)) + H_mean[None, :, :]
        
        return H_batch
    
    @tf.function
    def _compute_capacity_batch(self, H_batch: tf.Tensor, P: tf.Tensor) -> tf.Tensor:
        """배치 용량 계산 (TensorFlow)
        
        Args:
            H_batch: [batch_size, n_bs, n_ue]
            P: [n_ue, n_ue]
        
        Returns:
            capacities: [batch_size] natural log
        """
        # H @ P @ H^H for all samples: [batch_size, n_bs, n_bs]
        H_P = tf.einsum('bij,jk->bik', H_batch, P)  # [batch_size, n_bs, n_ue]
        H_conj = tf.math.conj(H_batch)
        H_P_HH = tf.einsum('bij,bkj->bik', H_P, H_conj)  # [batch_size, n_bs, n_bs]
        
        # I + H P H^H
        n_bs = tf.shape(H_batch)[1]
        I_batch = tf.eye(n_bs, dtype=tf.complex64)
        I_H_P_HH = I_batch[None, :, :] + H_P_HH
        
        # log det for all samples
        logdets = tf.math.real(tf.linalg.slogdet(I_H_P_HH)[1])
        
        return logdets
    
    def compute_asymptotic_capacity(self, P_opt: np.ndarray, 
                                     U_bs: np.ndarray, 
                                     U_ue: np.ndarray,
                                     Omega: np.ndarray, 
                                     H_mean: np.ndarray) -> float:
        """점근적 용량 계산 (Wen2011 식 7)
        
        I(P) = log det(I + Xi P) + log det(I + R) - gamma^T Omega psi
        
        Returns:
            I_asymptotic: 점근적 용량 (bits/Hz/sec)
        """
        # TensorFlow 변환
        P = tf.cast(P_opt, tf.complex64)
        U_bs_tf = tf.cast(U_bs, tf.complex64)
        U_ue_tf = tf.cast(U_ue, tf.complex64)
        Omega_tf = tf.cast(Omega, tf.float32)
        H_mean_tf = tf.cast(H_mean, tf.complex64)
        
        n_bs = U_bs.shape[0]
        n_ue = U_ue.shape[0]
        
        # 고정점 계산 (반복 횟수 증가)
        gamma, psi = self._compute_fixed_point(
            P, U_bs_tf, U_ue_tf, Omega_tf, H_mean_tf,
            float(self.config.eps_inner),
            float(self.config.regularization),
            int(self.config.max_iter_inner)
        )
        
        # (10) T, R 계산
        Omega_T_gamma = tf.matmul(tf.transpose(Omega_tf), tf.expand_dims(gamma, -1))
        T = U_ue_tf @ tf.linalg.diag(tf.cast(tf.squeeze(Omega_T_gamma), tf.complex64)) @ tf.linalg.adjoint(U_ue_tf)
        
        Omega_psi = tf.matmul(Omega_tf, tf.expand_dims(psi, -1))
        R = U_bs_tf @ tf.linalg.diag(tf.cast(tf.squeeze(Omega_psi), tf.complex64)) @ tf.linalg.adjoint(U_bs_tf)
        
        # (8) Xi 계산
        I_R = tf.eye(n_bs, dtype=tf.complex64) + R
        H_mean_H = tf.linalg.adjoint(H_mean_tf)
        Xi = T + H_mean_H @ tf.linalg.solve(I_R, H_mean_tf)
        
        # (7) I(P) 계산
        I_Xi_P = tf.eye(n_ue, dtype=tf.complex64) + Xi @ P
        
        logdet1 = tf.math.real(tf.linalg.slogdet(I_Xi_P)[1])
        logdet2 = tf.math.real(tf.linalg.slogdet(I_R)[1])
        coupling_term = tf.reduce_sum(gamma * tf.squeeze(Omega_psi))
        
        I_asymptotic_nat = logdet1 + logdet2 - coupling_term
        
        # Natural log → bits/Hz/sec
        I_asymptotic_bits = float(I_asymptotic_nat.numpy()) / np.log(2)
        
        return I_asymptotic_bits
    
    def generate_weichselberger_samples(self, U_bs: np.ndarray, 
                                        U_ue: np.ndarray,
                                        Omega: np.ndarray, 
                                        H_mean: np.ndarray,
                                        n_samples: int) -> list:
        """Weichselberger 채널 샘플 생성 (배치 처리, TensorFlow 최적화)
        
        H = U_bs @ (sqrt(Omega) ⊙ W) @ U_ue^H + H_mean
        W ~ CN(0,1)
        
        Returns:
            H_samples: list of [n_bs, n_ue] tensors
        """
        U_bs_tf = tf.cast(U_bs, tf.complex64)
        U_ue_tf = tf.cast(U_ue, tf.complex64)
        Omega_sqrt = tf.cast(np.sqrt(Omega), tf.complex64)
        H_mean_tf = tf.cast(H_mean, tf.complex64)
        
        n_bs, n_ue = U_bs.shape[0], U_ue.shape[0]
        batch_size = self.config.batch_size_channel
        
        H_samples = []
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        for b in range(n_batches):
            current_batch_size = min(batch_size, n_samples - b * batch_size)
            
            # 배치 생성 (@tf.function 최적화)
            H_batch = self._generate_channel_batch_tf(
                U_bs_tf, U_ue_tf, Omega_sqrt, H_mean_tf,
                current_batch_size, n_bs, n_ue
            )
            
            # 리스트로 분해 (Monte Carlo에서 필요)
            for i in range(current_batch_size):
                H_samples.append(H_batch[i])
        
        return H_samples
    
    def compute_monte_carlo_capacity(self, H_samples: list, P_opt: np.ndarray) -> tuple:
        """Monte Carlo 평균 용량 계산 (배치 처리)
        
        C_mc = E[log det(I + H P H^H)]
             ≈ (1/n) Σ log det(I + H_i P H_i^H)
        
        Returns:
            (C_mean, C_std): (float, float) in bits/Hz/sec
        """
        P = tf.cast(P_opt, tf.complex64)
        
        # 배치 크기로 나누어 처리 (메모리 효율)
        batch_size = 500
        n_samples = len(H_samples)
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        all_capacities = []
        
        for b in range(n_batches):
            start_idx = b * batch_size
            end_idx = min((b + 1) * batch_size, n_samples)
            
            # 배치 텐서 구성: [batch_size, n_bs, n_ue]
            H_batch = tf.stack(H_samples[start_idx:end_idx], axis=0)
            
            # 배치 용량 계산
            capacities_batch = self._compute_capacity_batch(H_batch, P)
            all_capacities.append(capacities_batch.numpy())
        
        # 모든 배치 결합
        capacities = np.concatenate(all_capacities)
        
        # Natural log → bits/Hz/sec
        capacities_bits = capacities / np.log(2)
        C_mean = np.mean(capacities_bits)
        C_std = np.std(capacities_bits)
        
        return C_mean, C_std
    
    def validate_single_ue(self, area: int, freq: float, ue: int, 
                           p1j_filepath: str) -> dict:
        """단일 UE 검증 전체 파이프라인
        
        Returns:
            dict: {
                'area': int,
                'freq': float,
                'ue_idx': int,
                'I_p1j': float (P1J 원본),
                'I_asymptotic': float (P1K 재계산),
                'C_monte_carlo': float,
                'C_std': float,
                'rel_error_p1j': float (P1J vs MC),
                'rel_error_p1k': float (P1K vs MC),
                'n_samples': int,
                'kappa': float,
                'channel_model': str,
                'elapsed_time': float,
                'success': bool,
                'error_msg': str (if failed)
            }
        """
        start_time = time.time()
        
        try:
            # 1. P1J 결과 로딩
            p1j_result = self.load_p1j_result(p1j_filepath)
            P_opt = p1j_result['P_opt']
            kappa = p1j_result['kappa']
            I_p1j = p1j_result['sum_rate_opt']  # P1J가 계산한 점근적 용량
            
            # 2. P1I 채널 파라미터 로딩
            channel_params = self.load_p1i_channel_params(area, freq, ue)
            U_bs = channel_params['U_bs']
            U_ue = channel_params['U_ue']
            Omega = channel_params['Omega']
            H_mean = channel_params['H_mean']
            channel_model = channel_params['channel_model']
            
            # 3. 점근적 용량 계산 (P1K 재계산)
            I_asymptotic = self.compute_asymptotic_capacity(P_opt, U_bs, U_ue, Omega, H_mean)
            
            # 4. 채널 샘플 생성
            H_samples = self.generate_weichselberger_samples(
                U_bs, U_ue, Omega, H_mean, 
                self.config.n_samples_monte_carlo
            )
            
            # 5. Monte Carlo 평균 용량 계산
            C_monte_carlo, C_std = self.compute_monte_carlo_capacity(H_samples, P_opt)
            
            # 6. 상대 오차 계산 (Monte Carlo 기준)
            if C_monte_carlo != 0:
                rel_error_p1j = abs(I_p1j - C_monte_carlo) / abs(C_monte_carlo)
                rel_error_p1k = abs(I_asymptotic - C_monte_carlo) / abs(C_monte_carlo)
            else:
                rel_error_p1j = 0.0
                rel_error_p1k = 0.0
            
            elapsed_time = time.time() - start_time
            
            return {
                'area': area,
                'freq': freq,
            'ue_idx': ue,
                'I_p1j': I_p1j,
                'I_asymptotic': I_asymptotic,
                'C_monte_carlo': C_monte_carlo,
                'C_std': C_std,
                'rel_error_p1j': rel_error_p1j,
                'rel_error_p1k': rel_error_p1k,
                'n_samples': self.config.n_samples_monte_carlo,
                'kappa': kappa,
            'channel_model': channel_model,
                'elapsed_time': elapsed_time,
                'success': True
            }
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            return {
                'area': area,
                'freq': freq,
                'ue_idx': ue,
                'success': False,
                'error_msg': str(e),
                'elapsed_time': elapsed_time
            }

# ===== SECTION 4: P1K_ResultManager =====
class P1K_ResultManager:
    """검증 결과 CSV 저장 관리"""
    
    def __init__(self, config: P1K_Config):
        self.config = config
        self._csv_timestamp = None
        self._csv_filepath = None
    
    def _init_csv(self, area: int, freq: float):
        """CSV 파일 초기화"""
        if self._csv_timestamp is None:
            utc_plus_9 = datetime.utcnow() + timedelta(hours=9)
            self._csv_timestamp = utc_plus_9.strftime('%Y%m%d_%H%M%S')
        
        csv_filename = f"Area{area}_{freq}GHz_Validation_{self._csv_timestamp}.csv"
        self._csv_filepath = os.path.join(self.config.P1K_OUTPUT_DIR, csv_filename)
    
    def append_validation_result(self, result: dict):
        """검증 결과 CSV 추가
        
        CSV 헤더:
        ue, I_asym, C_mc, C_std, rel_err(%), n_samp, kappa, ch_model, time_s
        """
        if not result['success']:
            # 실패한 경우는 기록하지 않음
            return
        
        # CSV 파일 초기화 (최초 호출 시)
        if self._csv_filepath is None:
            self._init_csv(result['area'], result['freq'])
        
        # 헤더 정의
        headers = [
            'ue', 'I_p1j', 'I_asym', 'C_mc', 'C_std', 
            'err_p1j(%)', 'err_p1k(%)', 'n_samp', 'kappa', 'ch_model', 'time_s'
        ]
        
        # 채널 모델 약어
        ch_model = result['channel_model']
        if 'Rician' in ch_model and 'Weichsel' in ch_model:
            ch_abbr = 'RicW'
        elif 'Rayleigh' in ch_model and 'Weichsel' in ch_model:
            ch_abbr = 'RayW'
        elif 'Rician' in ch_model and 'Kron' in ch_model:
            ch_abbr = 'RicK'
        elif 'Rayleigh' in ch_model and 'Kron' in ch_model:
            ch_abbr = 'RayK'
        else:
            ch_abbr = 'Other'
        
        # 데이터 행
        row = [
            f"{result['ue_idx']:4d}",
            f"{result['I_p1j']:8.4f}",
            f"{result['I_asymptotic']:8.4f}",
            f"{result['C_monte_carlo']:8.4f}",
            f"{result['C_std']:6.4f}",
            f"{result['rel_error_p1j']*100:6.2f}",
            f"{result['rel_error_p1k']*100:6.2f}",
            f"{result['n_samples']:6d}",
            f"{result['kappa']:.2e}",
            ch_abbr,
            f"{result['elapsed_time']:6.2f}"
        ]
        
        # CSV 파일 존재 여부 확인
        file_exists = os.path.exists(self._csv_filepath)
        
        # CSV에 append
        with open(self._csv_filepath, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(headers)
            writer.writerow(row)
    
# ===== SECTION 5: main =====
def main():
    """P1K: P1J 점근적 용량 검증 메인 실행"""
    
    print("=" * 80)
    print("P1K: P1J 점근적 용량 검증 (Wen2011 vs Monte Carlo)")
    print("=" * 80)
    print()
    
    # 설정 초기화
    config = P1K_Config()
    
    if not config.p1j_results:
        print("경고: 처리할 P1J 결과가 없습니다.")
        return
    
    # 검증기 및 결과 관리자 초기화
    validator = P1K_CapacityValidator(config)
    result_mgr = P1K_ResultManager(config)
    
    total_ues = len(config.p1j_results)
    
    print(f"\n검증 계획:")
    print(f"  - 총 UE: {total_ues}개")
    print(f"  - Monte Carlo 샘플: {config.n_samples_monte_carlo}개/UE")
    print(f"  - 배치 크기: {config.batch_size_channel}개")
    print()
    
    # 검증 시작
    overall_start = time.time()
    success_count = 0
    fail_count = 0
    total_time = 0.0
    
    for idx, (area, freq, ue, p1j_filepath) in enumerate(config.p1j_results, 1):
        print(f"\n{'='*80}")
        print(f"[{idx}/{total_ues}] Area{area}_{freq}GHz UE{ue} 검증 중")
        print(f"{'='*80}")
        
        # 검증 실행
        result = validator.validate_single_ue(area, freq, ue, p1j_filepath)
        
        if result['success']:
            # 성공
            result_mgr.append_validation_result(result)
            
            success_count += 1
            total_time += result['elapsed_time']
            avg_time = total_time / success_count
                
            remaining_ues, eta_sec, eta_min, eta_hr = total_ues - idx, avg_time * (total_ues - idx), avg_time * (total_ues - idx) / 60, avg_time * (total_ues - idx) / 3600 # 평균 시간 및 남은 시간 계산
                
            eta_str = f"{eta_hr:.1f}h" if eta_hr >= 1 else f"{eta_min:.1f}m" if eta_min >= 1 else f"{eta_sec:.0f}s" # ETA 포맷
                
            # 결과 출력
            print(f"✓ 검증 성공:")
            print(f"  - P1J (원본):      I_p1j  = {result['I_p1j']:.4f} (오차 vs MC: {result['rel_error_p1j']*100:.2f}%)")
            print(f"  - P1K (재계산):    I_asym = {result['I_asymptotic']:.4f} (오차 vs MC: {result['rel_error_p1k']*100:.2f}%)")
            print(f"  - Monte Carlo:     C_mc   = {result['C_monte_carlo']:.4f} ± {result['C_std']:.4f}")
            print(f"  - 처리 시간:       {result['elapsed_time']:.2f}s (평균 {avg_time:.2f}s, ETA {eta_str})")
        else: # 실패
            fail_count += 1
            print(f"✗ 검증 실패: {result['error_msg']}")
    
    # 완료 메시지
    total_elapsed = time.time() - overall_start
    print(f"\n{'='*80}")
    print(f"=== P1K 검증 완료 ===")
    print(f"{'='*80}")
    print(f"총 {total_ues}개 UE 검증")
    print(f"  - 성공: {success_count}개")
    print(f"  - 실패: {fail_count}개")
    print(f"총 소요 시간: {total_elapsed/60:.1f}분")
    print(f"저장 위치: {config.P1K_OUTPUT_DIR}")
    print()

if __name__ == "__main__":
    main()
