# ======================================================================
# P1L_BeamMgmt_SU_MIMO_2510v3.py
# P1L: Weichselberger SU-MIMO 업링크 빔 관리 (tf.function JIT)
# 
# === 최상위 목적 ===
# P1I/P1J 결과 기반 업링크 빔 세트 최적화
# - Stage 1: UE + BS 빔 선택 + P 최적화
# - Stage 2: BS 빔 재선택 (UE + P 고정)
# 
# === 알고리즘 구조 ===
# Stage 1: UE + BS 빔 선택 + P 최적화
#   1-1: UE 빔 평가 (W_bs=I[1024], P=I)
#   1-2: UE 빔 선택 (W_bs=I[1024], P=I)
#   1-3a: BS 빔 Greedy (UE 고정, 0→64개, P=I)
#   1-3b: P 최적화 (UE + BS 고정)
# 
# Stage 2: BS 빔 재선택 (UE + P 고정, 0→64개)
# 
# === 입력/출력 ===
# 입력 1: P1I 청크 (U_bs, U_ue, Omega, H_mean)
# 입력 2: P1J 결과 (C_AE만 사용, P_opt는 미사용)
# 출력: Area{area}_{freq}GHz_SU_BM_{timestamp}.csv
#   Stage 1: ue, stage, step, detail, C_beam, loss_pct
#   Stage 2: ue, stage, L_bs, added_beam, C_beam, delta_C, improvement_ratio
#
# === 주요 수정 이력 ===
# [251017] P1L v2 신규 작성: 2단계 Greedy 알고리즘 (P1L_2510v2.tex)
# [251018] 빔 중복 허용 옵션 추가 (allow_beam_reuse_ue, allow_beam_reuse_bs)
# [251019] 재시작 옵션 추가 (resume_from_existing)
# [251019] GPU 최적화: 빔 벡터화 + UE 배치 처리 (batch_size_ue=25)
#          - 중복 허용 고정 (조건문 제거)
#          - Stage 1-1: 16개 빔 동시 평가
#          - Stage 1-2: Greedy 후보 벡터화
#          - Stage 2: 64개 후보 벡터화
#          - UE 배치 처리 (25개 동시)
# [251019] Stage 1-3 분리 + tf.function 최적화
#          - Stage 1-1/1-2: W_bs=I (1024 AE 직접 사용)
#          - Stage 1-3a: BS 빔 Greedy (UE 고정, P=I)
#          - Stage 1-3b: P 최적화 (UE + BS 고정)
#          - Stage 2: BS 빔 재선택 (UE + P 고정)
#          - 통합 메서드: _compute_capacity_for_multiple_beamformings_tf
#          - tf.function 데코레이터로 JIT 컴파일 최적화
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
import shutil
import math
from pathlib import Path
from itertools import combinations

# TensorFlow 환경 설정
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc'
gpu_num = 0
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

# GPU 메모리 설정
gpus = tf.config.list_physical_devices('GPU')
for g in gpus:
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except RuntimeError as e:
        pass

tf.get_logger().setLevel("ERROR")
tf.config.optimizer.set_jit(True)
tf.config.threading.set_inter_op_parallelism_threads(0) # 연산 간 병렬성: 모든 GPU 코어 사용
tf.config.threading.set_intra_op_parallelism_threads(0) # 연산 내 병렬성: 모든 GPU 코어 사용
tf.random.set_seed(42)
np.random.seed(42)

# ======================================================================
# 코드 구조: 의존성 기반 순서
# ======================================================================
# LEVEL 0: 독립적 기본 클래스
#   - P1L_Config: 설정 관리
#   - DFTCodebook: DFT 코드북 생성
#   - BeamformingMatrix: 빔포밍 행렬 구성
#   - BeamDomainTransform: 빔 도메인 파라미터 변환
#
# LEVEL 1: Config 의존 클래스
#   - DataLoader: P1I/P1J 데이터 로딩 (← P1L_Config)
#   - BeamDomainCapacity: 빔 도메인 용량 계산 (← P1L_Config)
#   - P1L_ResultManager: CSV 결과 저장 (← P1L_Config)
#
# LEVEL 2: 복합 의존 클래스
#   - GreedyBeamSelector: Greedy 빔 선택 (← 모든 Level 0, 1 클래스)
#
# LEVEL 3: 최상위 실행
#   - main: 전체 파이프라인 실행 (← 모든 클래스)
# ======================================================================

# ===== LEVEL 0: 독립적 기본 클래스 =====

# ----- P1L_Config -----
class P1L_Config:
    """P1L 빔 관리 설정"""
    
    def __init__(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 입력 데이터 경로
        self.P1I_INPUT_DIR = os.path.join(script_dir, "P1I_Weichsel_Chunk_Results")
        self.P1J_INPUT_DIR = os.path.join(script_dir, "P1J_Capacity_Results")
        
        # 출력 결과 경로
        self.P1L_OUTPUT_DIR = os.path.join(script_dir, "P1L_BeamMgmt_Results")
        os.makedirs(self.P1L_OUTPUT_DIR, exist_ok=True)
        
        # 필터링 설정
        self.target_areas = [1]
        self.target_freqs = [7.5]
        
        # 안테나 및 레이어 설정
        self.n_bs_layers = 64  # BS 레이어 수
        self.n_ue_layers = 4   # UE 레이어 수
        
        # DFT 코드북 설정 (2D 오버샘플링)
        self.bs_ant_per_dim = 4        # BS: 4×4 안테나 어레이
        self.bs_oversample = 2          # BS: 오버샘플링 계수 (4×2=8)
        self.ue_ant_per_dim = 2        # UE: 2×2 안테나 어레이
        self.ue_oversample = 2          # UE: 오버샘플링 계수 (2×2=4)
        
        # 자동 계산
        self.n_ant_per_bs_layer = self.bs_ant_per_dim ** 2  # 16
        self.n_ant_per_ue_layer = self.ue_ant_per_dim ** 2  # 4
        self.n_cb_bs = (self.bs_ant_per_dim * self.bs_oversample) ** 2  # 64
        self.n_cb_ue = (self.ue_ant_per_dim * self.ue_oversample) ** 2  # 16
        self.n_bs = self.n_ant_per_bs_layer * self.n_bs_layers  # 1024
        self.n_ue = self.n_ant_per_ue_layer * self.n_ue_layers  # 16
        
        # Beam Selection (Static tf.function 방식)
        self.n_ue_greedy_starts = 4     # UE greedy 시작점 개수
        self.n_ue_beams_static = 4      # UE 빔 고정 개수
        self.n_bs_beams_static = 6      # BS 빔 고정 개수
        
        # 용량 계산 설정 (Wen2011 최적화)
        self.eps_inner = 1e-4
        self.max_iter_inner = 50
        self.max_iter_outer = 50
        self.eps_outer = 0.05
        self.regularization = 1e-20
        
        # SNR 설정
        self.SNR_dB = 10.0
        self.SNR_linear = 10.0**(self.SNR_dB / 10.0)
        
        # 재시작 설정
        self.resume_from_existing = False  # True: 기존 결과 이어서, False: 새로 시작 
        
        # UE 스캔
        self.detect_ue_list()
    
    def detect_ue_list(self):
        """처리 대상 UE 스캔 (P1I & P1J 교집합)"""
        
        # P1I 청크 스캔
        p1i_ues = set()
        for area in self.target_areas:
            for freq in self.target_freqs:
                scan_pattern = f"{self.P1I_INPUT_DIR}/Area{area}_{freq}GHz_Weichsel_Chunk_*.npz"
                chunk_files = glob.glob(scan_pattern)
                
                for filepath in chunk_files:
                    data = np.load(filepath, allow_pickle=True)
                    ue_indices = data['ue_indices'].tolist()
                    for ue in ue_indices:
                        p1i_ues.add((area, freq, ue))
        
        # P1J 결과 스캔
        p1j_ues = set()
        pattern = r'Area(\d+)_(.+)GHz_UE(\d+)_Capacity\.npz'
        p1j_files = glob.glob(f"{self.P1J_INPUT_DIR}/*.npz")
        
        for filepath in p1j_files:
            filename = os.path.basename(filepath)
            match = re.match(pattern, filename)
            if match:
                area = int(match.group(1))
                freq = float(match.group(2))
                ue = int(match.group(3))
                
                if area in self.target_areas and freq in self.target_freqs:
                    p1j_ues.add((area, freq, ue, filepath))
        
        # 교집합
        p1i_keys = {(a, f, u) for a, f, u in p1i_ues}
        self.ue_list = []
        
        for area, freq, ue, p1j_path in sorted(p1j_ues):
            if (area, freq, ue) in p1i_keys:
                self.ue_list.append((area, freq, ue, p1j_path))
        
        if self.ue_list:
            print(f"처리 대상 UE: {len(self.ue_list)}개")
            area_groups = {}
            for area, freq, ue, _ in self.ue_list:
                key = f"Area{area}_{freq}GHz"
                area_groups[key] = area_groups.get(key, 0) + 1
            for area_freq, count in sorted(area_groups.items()):
                print(f"  - {area_freq}: {count}개 UE")
        else:
            print("경고: 처리 대상 UE가 없습니다.")

# ----- DFTCodebook -----
class DFTCodebook:
    """DFT 코드북 생성 (TensorFlow)"""
    
    @staticmethod
    def generate_1d_dft_tf(N: int, K: int) -> tf.Tensor:
        """1D DFT 코드북 (TensorFlow)
        
        [F_{N,K}]_{i,j} = (1/√N) exp(-j 2π ij / (NK))
        
        Args:
            N: 안테나 수 per 차원
            K: 오버샘플링 계수
        
        Returns:
            F: [N, NK] tf.complex64
        """
        i = tf.cast(tf.range(N)[:, None], tf.float32)
        j = tf.cast(tf.range(N * K)[None, :], tf.float32)
        pi = tf.constant(3.141592653589793, dtype=tf.float32)
        phase = -2.0 * pi * i * j / float(N * K)
        sqrt_N = tf.sqrt(tf.constant(float(N), dtype=tf.float32))
        sqrt_N_complex = tf.cast(sqrt_N, tf.complex64)
        return tf.exp(tf.complex(0.0, phase)) / sqrt_N_complex
    
    @staticmethod
    def generate_2d_dft_codebook_tf(ant_per_dim: int, oversample: int) -> tf.Tensor:
        """2D DFT 코드북 생성 (TensorFlow)
        
        F = F_{N,K} ⊗ F_{N,K} ∈ C^{N²×(NK)²}
        
        Args:
            ant_per_dim: 1차원당 안테나 수 (N)
            oversample: 오버샘플링 계수 (K)
        
        Returns:
            F: [ant_per_dim², (ant_per_dim*oversample)²] tf.complex64
        
        Example:
            BS: ant_per_dim=4, oversample=2 → F: [16, 64]
            UE: ant_per_dim=2, oversample=2 → F: [4, 16]
        """
        F_1d = DFTCodebook.generate_1d_dft_tf(ant_per_dim, oversample)
        
        # Kronecker product: F ⊗ F
        n_ant = ant_per_dim
        n_beams = ant_per_dim * oversample
        
        # tf.einsum: F_ij * F_kl = F_ikjl → reshape → [N²,(NK)²]
        F_kron = tf.einsum('ij,kl->ikjl', F_1d, F_1d)
        F_kron = tf.reshape(F_kron, [n_ant**2, n_beams**2])
        
        return F_kron

# ----- BeamformingMatrix -----
class BeamformingMatrix:
    """빔포밍 행렬 구성 (TensorFlow)"""
    
    @staticmethod
    @tf.function
    def construct_beamforming_blockdiag_tf(F: tf.Tensor, 
                                          beam_indices: tf.Tensor,
                                          n_total_layers: int) -> tf.Tensor:
        """Block-diagonal 빔포밍 행렬 (TensorFlow, JIT 최적화)
        
        W = blkdiag(w_1, w_2, ..., w_L) ∈ C^{N_total×L}
        
        Args:
            F: [n_ant_per_layer, n_codebook] DFT 코드북 (tf.complex64)
            beam_indices: [n_active] 선택된 빔 인덱스 (tf.int32)
            n_total_layers: 총 레이어 수 (int)
        
        Returns:
            W: [n_total_ant, n_total_layers] tf.complex64
            
        Example:
            BS: F=[16,64], beam_indices=[0,1,5], n_total_layers=64
                → W=[1024, 64] (레이어 0,1,5만 활성화)
            
            UE: F=[4,16], beam_indices=[3,7,11,15], n_total_layers=4
                → W=[16, 4]
        """
        n_ant_per_layer = tf.shape(F)[0]
        n_total_ant = n_ant_per_layer * n_total_layers
        n_active = tf.shape(beam_indices)[0]
        
        # 초기화
        W = tf.zeros([n_total_ant, n_total_layers], dtype=tf.complex64)
        
        # 빔 할당 방식 결정
        is_sparse = n_active < n_total_layers
        
        def sparse_update():
            """BS: sparse activation (선택된 레이어만)"""
            # 인덱스 생성 (벡터화)
            i_grid = tf.range(n_active)[:, None]  # [n_active, 1]
            j_grid = tf.range(n_ant_per_layer)[None, :]  # [1, n_ant_per_layer]
            
            layer_indices = tf.gather(beam_indices, i_grid)  # [n_active, 1]
            row_starts = layer_indices * n_ant_per_layer
            row_indices = row_starts + j_grid  # [n_active, n_ant_per_layer]
            col_indices = tf.tile(layer_indices, [1, n_ant_per_layer])  # [n_active, n_ant_per_layer]
            
            # 업데이트 인덱스와 값
            indices = tf.stack([
                tf.reshape(row_indices, [-1]),
                tf.reshape(col_indices, [-1])
            ], axis=1)  # [n_active*n_ant_per_layer, 2]
            
            beam_vecs = tf.gather(F, beam_indices, axis=1)  # [n_ant_per_layer, n_active]
            updates = tf.reshape(tf.transpose(beam_vecs), [-1])  # [n_active*n_ant_per_layer]
            
            return tf.tensor_scatter_nd_update(W, indices, updates)
        
        def dense_update():
            """UE: dense activation (모든 레이어)"""
            # 인덱스 생성 (벡터화)
            i_grid = tf.range(n_active)[:, None]  # [n_active, 1]
            j_grid = tf.range(n_ant_per_layer)[None, :]  # [1, n_ant_per_layer]
            
            row_starts = i_grid * n_ant_per_layer
            row_indices = row_starts + j_grid  # [n_active, n_ant_per_layer]
            col_indices = tf.tile(i_grid, [1, n_ant_per_layer])  # [n_active, n_ant_per_layer]
            
            # 업데이트 인덱스와 값
            indices = tf.stack([
                tf.reshape(row_indices, [-1]),
                tf.reshape(col_indices, [-1])
            ], axis=1)  # [n_active*n_ant_per_layer, 2]
            
            beam_vecs = tf.gather(F, beam_indices, axis=1)  # [n_ant_per_layer, n_active]
            updates = tf.reshape(tf.transpose(beam_vecs), [-1])  # [n_active*n_ant_per_layer]
            
            return tf.tensor_scatter_nd_update(W, indices, updates)
        
        # tf.cond로 분기
        W = tf.cond(is_sparse, sparse_update, dense_update)
        
        return W
    
    @staticmethod
    @tf.function
    def construct_beamforming_blockdiag_tf_batch(F: tf.Tensor,
                                                 beam_indices_batch: tf.Tensor,
                                                 n_total_layers: int) -> tf.Tensor:
        """Block-diagonal 빔포밍 행렬 (배치 처리)
        
        W = blkdiag(w_1, w_2, ..., w_L) ∈ C^{N_total×L} for each batch
        
        Args:
            F: [n_ant_per_layer, n_codebook] DFT 코드북 (tf.complex64)
            beam_indices_batch: [batch_size, n_active] 선택된 빔 인덱스들 (tf.int32)
            n_total_layers: 총 레이어 수 (int)
        
        Returns:
            W_batch: [batch_size, n_total_ant, n_total_layers] tf.complex64
            
        Example:
            BS: F=[16,64], beam_indices_batch=[64,3], n_total_layers=64
                → W_batch=[64, 1024, 64]
        """
        batch_size = tf.shape(beam_indices_batch)[0]
        n_active = tf.shape(beam_indices_batch)[1]
        n_ant_per_layer = tf.shape(F)[0]
        n_total_ant = n_ant_per_layer * n_total_layers
        
        # 초기화 (3D)
        W_batch = tf.zeros([batch_size, n_total_ant, n_total_layers], dtype=tf.complex64)
        
        # Sparse update (BS 케이스)
        # 1. 인덱스 계산 (broadcasting)
        batch_idx = tf.range(batch_size, dtype=tf.int32)[:, None, None]  # [batch, 1, 1]
        ant_idx = tf.range(n_ant_per_layer, dtype=tf.int32)[None, None, :]  # [1, 1, n_ant]
        
        # layer_indices: [batch, n_active, 1]
        layer_indices = tf.expand_dims(beam_indices_batch, axis=2)
        row_starts = layer_indices * n_ant_per_layer
        
        # row_indices: [batch, n_active, n_ant]
        row_indices = row_starts + tf.cast(ant_idx, tf.int32)
        
        # col_indices: [batch, n_active, n_ant]
        col_indices = tf.tile(layer_indices, [1, 1, n_ant_per_layer])
        
        # batch_indices: [batch, n_active, n_ant]
        batch_indices = tf.tile(batch_idx, [1, n_active, n_ant_per_layer])
        
        # 3D indices: [N, 3] where N = batch*n_active*n_ant
        indices = tf.stack([
            tf.reshape(batch_indices, [-1]),
            tf.reshape(row_indices, [-1]),
            tf.reshape(col_indices, [-1])
        ], axis=1)
        
        # 2. 업데이트 값 계산
        # tf.gather for each batch
        def gather_beams(indices):
            return tf.gather(F, indices, axis=1)  # [n_ant, n_active]
        
        beam_vecs_batch = tf.map_fn(gather_beams, beam_indices_batch, dtype=tf.complex64)
        # beam_vecs_batch: [batch, n_ant, n_active]
        
        # Reshape for scatter: [batch, n_active, n_ant]
        beam_vecs_transposed = tf.transpose(beam_vecs_batch, [0, 2, 1])
        updates = tf.reshape(beam_vecs_transposed, [-1])
        
        # 3. 단일 scatter 연산
        W_batch = tf.tensor_scatter_nd_update(W_batch, indices, updates)
        
        return W_batch

# ----- BeamDomainTransform -----
class BeamDomainTransform:
    """빔 도메인 Weichselberger 파라미터 변환 (TensorFlow)"""
    
    @staticmethod
    @tf.function
    def transform_mean_channel(H_mean: tf.Tensor, W_bs: tf.Tensor, 
                               W_ue: tf.Tensor) -> tf.Tensor:
        """평균 채널 변환
        
        H̄_beam = W_bs^H H̄ W_ue
        
        Args:
            H_mean: [n_bs, n_ue] complex64
            W_bs: [n_bs, B_bs] complex64
            W_ue: [n_ue, B_ue] complex64
            
        Returns:
            H_mean_beam: [B_bs, B_ue] complex64
        """
        return tf.linalg.adjoint(W_bs) @ H_mean @ W_ue
    
    @staticmethod
    @tf.function
    def transform_bs_covariance(U_bs: tf.Tensor, U_ue: tf.Tensor,
                                Omega: tf.Tensor, W_bs: tf.Tensor,
                                W_ue: tf.Tensor) -> tuple:
        """BS 수신 공분산 변환
        
        Args:
            U_bs: [n_bs, n_bs] complex64
            U_ue: [n_ue, n_ue] complex64
            Omega: [n_bs, n_ue] float32
            W_bs: [n_bs, B_bs] complex64
            W_ue: [n_ue, B_ue] complex64
        
        Returns:
            (R_bs_beam, U_bs_beam, Lambda_bs_beam)
        """
        T_ue = tf.linalg.adjoint(U_ue) @ W_ue
        v_ue = tf.reduce_sum(tf.abs(T_ue)**2, axis=1)
        d_bs = tf.linalg.matvec(Omega, v_ue)
        
        U_bs_W_bs = tf.linalg.adjoint(U_bs) @ W_bs
        d_bs_complex = tf.cast(d_bs, tf.complex64)
        R_bs_beam = tf.linalg.adjoint(U_bs_W_bs) @ tf.linalg.diag(d_bs_complex) @ U_bs_W_bs
        
        Lambda_bs_beam, U_bs_beam = tf.linalg.eigh(R_bs_beam)
        Lambda_bs_beam = tf.math.real(Lambda_bs_beam)
        Lambda_bs_beam = tf.maximum(Lambda_bs_beam, 0.0)
        
        return R_bs_beam, U_bs_beam, Lambda_bs_beam
    
    @staticmethod
    @tf.function
    def transform_ue_covariance(U_bs: tf.Tensor, U_ue: tf.Tensor,
                                Omega: tf.Tensor, W_bs: tf.Tensor,
                                W_ue: tf.Tensor) -> tuple:
        """UE 송신 공분산 변환
        
        Args:
            U_bs: [n_bs, n_bs] complex64
            U_ue: [n_ue, n_ue] complex64
            Omega: [n_bs, n_ue] float32
            W_bs: [n_bs, B_bs] complex64
            W_ue: [n_ue, B_ue] complex64
        
        Returns:
            (R_ue_beam, U_ue_beam, Lambda_ue_beam)
        """
        T_bs = tf.linalg.adjoint(U_bs) @ W_bs
        v_bs = tf.reduce_sum(tf.abs(T_bs)**2, axis=1)
        d_ue = tf.linalg.matvec(Omega, v_bs, transpose_a=True)
        
        U_ue_W_ue = tf.linalg.adjoint(U_ue) @ W_ue
        d_ue_complex = tf.cast(d_ue, tf.complex64)
        R_ue_beam = tf.linalg.adjoint(U_ue_W_ue) @ tf.linalg.diag(d_ue_complex) @ U_ue_W_ue
        
        Lambda_ue_beam, U_ue_beam = tf.linalg.eigh(R_ue_beam)
        Lambda_ue_beam = tf.math.real(Lambda_ue_beam)
        Lambda_ue_beam = tf.maximum(Lambda_ue_beam, 0.0)
        
        return R_ue_beam, U_ue_beam, Lambda_ue_beam
    
    @staticmethod
    @tf.function
    def transform_coupling_matrix(U_bs: tf.Tensor, U_ue: tf.Tensor,
                                  Omega: tf.Tensor, W_bs: tf.Tensor,
                                  W_ue: tf.Tensor, U_bs_beam: tf.Tensor,
                                  U_ue_beam: tf.Tensor) -> tf.Tensor:
        """커플링 행렬 변환
        
        Omega_beam = |V_bs|^2^T Omega |V_ue|^2
        
        Args:
            U_bs: [n_bs, n_bs] complex64
            U_ue: [n_ue, n_ue] complex64
            Omega: [n_bs, n_ue] float32
            W_bs: [n_bs, B_bs] complex64
            W_ue: [n_ue, B_ue] complex64
            U_bs_beam: [B_bs, B_bs] complex64
            U_ue_beam: [B_ue, B_ue] complex64
            
        Returns:
            Omega_beam: [B_bs, B_ue] float32
        """
        V_bs = tf.linalg.adjoint(U_bs) @ W_bs @ U_bs_beam
        V_ue = tf.linalg.adjoint(U_ue) @ W_ue @ U_ue_beam
        
        V_bs_abs2 = tf.abs(V_bs)**2
        V_ue_abs2 = tf.abs(V_ue)**2
        
        return tf.transpose(V_bs_abs2) @ Omega @ V_ue_abs2

# ===== LEVEL 1: Config 의존 클래스 =====

# ----- DataLoader -----
class DataLoader:
    """P1I/P1J 데이터 로딩"""
    
    def __init__(self, config: P1L_Config):
        self.config = config
        self.p1i_cache = {}
    
    def load_p1j_result(self, filepath: str) -> dict:
        """P1J 결과 로딩 (C_AE만 사용)
        
        Returns:
            {'C_AE': float, 'kappa': float, 'metadata': dict}
        """
        data = np.load(filepath, allow_pickle=True)
        
        return {
            'C_AE': float(data['sum_rate_opt']),
            'kappa': float(data.get('kappa', 0.0)) if data.get('kappa') is not None else 0.0,
            'metadata': data['metadata'].item()
        }
        
    def load_p1i_channel_params(self, area: int, freq: float, ue: int) -> dict:
        """P1I 청크에서 채널 파라미터 로딩 (업링크 전환, TensorFlow 변환)
        
        Returns:
            {
                'U_bs': tf.Tensor [complex64],
                'U_ue': tf.Tensor [complex64],
                'Omega': tf.Tensor [float32],
                'H_mean': tf.Tensor [complex64],
                'kappa': float,
                'channel_model': str
            }
        """
        # 청크 파일 찾기
        scan_pattern = f"{self.config.P1I_INPUT_DIR}/Area{area}_{freq}GHz_Weichsel_Chunk_*.npz"
        chunk_files = glob.glob(scan_pattern)
        
        pattern = r'Area(\d+)_(.+)GHz_Weichsel_Chunk_(\d+)_UE(\d+)-(\d+)\.npz'
        chunk_filepath = None
        
        for filepath in chunk_files:
            filename = os.path.basename(filepath)
            match = re.match(pattern, filename)
            if match:
                ue_start = int(match.group(4))
                ue_end = int(match.group(5))
                if ue_start <= ue <= ue_end:
                    chunk_filepath = filepath
                    break
        
        if chunk_filepath is None:
            raise FileNotFoundError(f"P1I 청크를 찾을 수 없습니다: Area{area}_{freq}GHz UE{ue}")
        
        # 캐시 활용
        if chunk_filepath not in self.p1i_cache:
            data = np.load(chunk_filepath, allow_pickle=True)
            self.p1i_cache[chunk_filepath] = data
        else:
            data = self.p1i_cache[chunk_filepath]
        
        # UE 인덱스 확인
        ue_indices = data['ue_indices'].tolist()
        if ue not in ue_indices:
            raise ValueError(f"UE{ue}가 청크에 없습니다: {chunk_filepath}")
        idx_in_chunk = ue_indices.index(ue)
        
        # DL 데이터 추출
        U_bs_np = data['P1G_U_BS'][idx_in_chunk]
        U_ue_np = data['P1G_U_UE'][idx_in_chunk]
        Omega_dl_np = data['P1G_Omega'][idx_in_chunk]
        H_mean_dl_np = data['P1H_H_mean'][idx_in_chunk]
        channel_model = data['enhanced_metadata'][idx_in_chunk]['channel_model']
        
        # DL → UL 전환
        Omega_ul_np = Omega_dl_np.T
        H_mean_ul_np = H_mean_dl_np.conj().T
        
        # Wen2011 정규화
        Omega_norm_np, H_mean_norm_np, kappa = self._normalize_channel(
            Omega_ul_np, H_mean_ul_np, U_bs_np.shape[0], U_ue_np.shape[0]
        )
        
        # TensorFlow 변환
        U_bs_tf = tf.constant(U_bs_np, dtype=tf.complex64)
        U_ue_tf = tf.constant(U_ue_np, dtype=tf.complex64)
        Omega_tf = tf.constant(Omega_norm_np, dtype=tf.float32)
        H_mean_tf = tf.constant(H_mean_norm_np, dtype=tf.complex64)
        
        return {
            'U_bs': U_bs_tf,
            'U_ue': U_ue_tf,
            'Omega': Omega_tf,
            'H_mean': H_mean_tf,
            'kappa': kappa,
            'channel_model': channel_model
        }
    
    def _normalize_channel(self, Omega: np.ndarray, H_mean: np.ndarray,
                          n_bs: int, n_ue: int) -> tuple:
        """Wen2011 채널 정규화"""
        rho_target = self.config.SNR_linear
        N_M = n_bs * n_ue
        epsilon = 1e-30
        threshold = 1e-10
        
        P_nlos = np.sum(Omega)
        P_los = np.sum(np.abs(H_mean)**2)
        
        if P_los < threshold * P_nlos:
            kappa = 0.0
            Omega_norm = (N_M * rho_target / (P_nlos + epsilon)) * Omega
            H_mean_norm = np.zeros_like(H_mean)
        elif P_nlos < threshold * P_los:
            kappa = float('inf')
            Omega_norm = np.zeros_like(Omega)
            scale = np.sqrt(N_M * rho_target / (P_los + epsilon))
            H_mean_norm = scale * H_mean
        else:
            kappa = P_los / P_nlos
            P_nlos_target = N_M * (1.0 / (kappa + 1.0)) * rho_target
            P_los_target = N_M * (kappa / (kappa + 1.0)) * rho_target
            
            Omega_norm = (P_nlos_target / P_nlos) * Omega
            scale = np.sqrt(P_los_target / P_los)
            H_mean_norm = scale * H_mean
        
        return Omega_norm, H_mean_norm, kappa
    
    def load_p1i_batch(self, ue_list: list) -> list:
        """여러 UE의 채널 파라미터 로딩 (리스트 반환)
        
        Args:
            ue_list: [(area, freq, ue), ...]
            
        Returns:
            [
                {'U_bs': tf.Tensor, 'U_ue': ..., 'Omega': ..., ...},
                {'U_bs': tf.Tensor, 'U_ue': ..., 'Omega': ..., ...},
                ...
            ]
        """
        batch_data = []
        for area, freq, ue in ue_list:
            data = self.load_p1i_channel_params(area, freq, ue)
            batch_data.append(data)
        return batch_data
    
    def load_p1j_batch(self, p1j_paths: list) -> list:
        """여러 UE의 P1J 결과 로딩
        
        Args:
            p1j_paths: [filepath1, filepath2, ...]
            
        Returns:
            [{'C_AE': float, 'kappa': float, ...}, ...]
        """
        return [self.load_p1j_result(path) for path in p1j_paths]

# ----- BeamDomainCapacity -----
class BeamDomainCapacity:
    """빔 도메인 채널 용량 계산 (P1J 로직 기반)
    
    메서드 의존성 순서:
    Level 0 (독립): _find_inverse_water_level_tf, _compute_Xi_for_water_filling, _fixed_point_loop_tf
    Level 1: _water_filling_p1j (→ _find_inverse_water_level_tf)
    Level 2: wen2011_optimize (→ Level 0, 1 메서드들) → (P_opt, C_opt) 반환
    Level 3: _compute_beam_capacity_with_P (임의의 P로 용량 계산)
    """
    
    def __init__(self, config: P1L_Config):
        self.config = config
    
    # ========== Level 0: 독립 메서드 (tf.function) ==========
    
    @tf.function
    def _find_inverse_water_level_tf(self, eigvals: tf.Tensor, valid_mask: tf.Tensor, 
                                    power_total: tf.Tensor) -> tf.Tensor:
        """Water-filling level ν = 1/μ 계산 (P1J 동일)
        
        Args:
            eigvals: [n_ue] 고유값 (유효하지 않은 값은 0)
            valid_mask: [n_ue] 유효한 고유값 mask
            power_total: scalar
        
        Returns:
            nu: ν = 1/μ
        """
        # 유효한 고유값만 추출
        valid_eigvals = tf.boolean_mask(eigvals, valid_mask)
        inv_valid_eigvals = 1.0 / valid_eigvals
        
        # ν 탐색 범위
        nu_min = 0.0
        nu_max = power_total + tf.reduce_max(inv_valid_eigvals)
        
        # Bisection loop
        def cond(nu_min, nu_max, i):
            return tf.logical_and(i < 100, tf.abs(nu_max - nu_min) >= 1e-10)
        
        def body(nu_min, nu_max, i):
            nu_mid = (nu_min + nu_max) / 2.0
            allocated_power = tf.reduce_sum(tf.maximum(nu_mid - inv_valid_eigvals, 0.0))
            
            nu_min_new = tf.cond(allocated_power < power_total,
                               lambda: nu_mid,
                               lambda: nu_min)
            nu_max_new = tf.cond(allocated_power < power_total,
                               lambda: nu_max,
                               lambda: nu_mid)
            
            return nu_min_new, nu_max_new, i + 1
        
        nu_min_final, nu_max_final, _ = tf.while_loop(cond, body, [nu_min, nu_max, 0])
        
        return (nu_min_final + nu_max_final) / 2.0
    
    @tf.function
    def _compute_Xi_for_water_filling(self, gamma: tf.Tensor, psi: tf.Tensor,
                                     U_bs: tf.Tensor, U_ue: tf.Tensor, 
                                     Omega: tf.Tensor, H_mean: tf.Tensor,
                                     reg_complex: tf.Tensor, n_bs: tf.Tensor, n_ue: tf.Tensor) -> tf.Tensor:
        """γ, ψ로부터 Ξ 계산 (P1J 동일)"""
        # T, R 계산
        Omega_T_gamma = tf.matmul(tf.transpose(Omega), tf.expand_dims(gamma, -1))
        T = U_ue @ tf.linalg.diag(tf.cast(tf.squeeze(Omega_T_gamma), tf.complex64)) @ tf.linalg.adjoint(U_ue)
        
        Omega_psi = tf.matmul(Omega, tf.expand_dims(psi, -1))
        R = U_bs @ tf.linalg.diag(tf.cast(tf.squeeze(Omega_psi), tf.complex64)) @ tf.linalg.adjoint(U_bs)
        
        # Ξ 계산
        I_R = tf.eye(n_bs, dtype=tf.complex64) + R + reg_complex * tf.eye(n_bs, dtype=tf.complex64)
        H_mean_H = tf.linalg.adjoint(H_mean)
        Xi = T + H_mean_H @ tf.linalg.solve(I_R, H_mean)
        
        return Xi
    
    @tf.function
    def _fixed_point_loop_tf(self, P: tf.Tensor, U_bs: tf.Tensor, U_ue: tf.Tensor,
                            Omega: tf.Tensor, H_mean: tf.Tensor, max_iter_inner: int,
                            reg_complex: tf.Tensor, n_bs: tf.Tensor, n_ue: tf.Tensor) -> tuple:
        """고정점 반복 (P1J 완전 동일 복사)
        
        Args:
            P: [n_ue, n_ue] UE 입력 공분산
            
        Returns:
            (best_gamma, best_psi, best_idx)
        """
        # 초기화
        gamma = tf.ones(n_bs, dtype=tf.float32)
        psi = tf.ones(n_ue, dtype=tf.float32)
        
        # History 저장
        gamma_history = tf.TensorArray(tf.float32, size=max_iter_inner, dynamic_size=False)
        psi_history = tf.TensorArray(tf.float32, size=max_iter_inner, dynamic_size=False)
        diff_history = tf.TensorArray(tf.float32, size=max_iter_inner, dynamic_size=False)
        
        # 고정 반복
        for i in tf.range(max_iter_inner):
            gamma_prev = tf.identity(gamma)
            psi_prev = tf.identity(psi)
            
            # abs() 클리핑: 음수 방지
            gamma = tf.abs(gamma)
            psi = tf.abs(psi)
            
            # (10) T, R 계산
            Omega_T_gamma = tf.matmul(tf.transpose(Omega), tf.expand_dims(gamma, -1))
            T = U_ue @ tf.linalg.diag(tf.cast(tf.squeeze(Omega_T_gamma), tf.complex64)) @ tf.linalg.adjoint(U_ue)
            
            Omega_psi = tf.matmul(Omega, tf.expand_dims(psi, -1))
            R = U_bs @ tf.linalg.diag(tf.cast(tf.squeeze(Omega_psi), tf.complex64)) @ tf.linalg.adjoint(U_bs)
            
            # (8) Ξ 계산
            I_R = tf.eye(n_bs, dtype=tf.complex64) + R + reg_complex * tf.eye(n_bs, dtype=tf.complex64)
            H_mean_H = tf.linalg.adjoint(H_mean)
            Xi = T + H_mean_H @ tf.linalg.solve(I_R, H_mean)
            
            # (11a) γ 업데이트
            I_R_inv_U_bs = tf.linalg.solve(I_R, U_bs)
            gamma = tf.math.real(
                tf.reduce_sum(tf.math.conj(U_bs) * I_R_inv_U_bs, axis=0)
            )
            
            # (11b) ψ 업데이트
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
        iter_indices = tf.cast(tf.range(1, max_iter_inner + 1), tf.float32)
        diff_per_iter = all_diffs / iter_indices
        best_idx = tf.argmin(diff_per_iter, output_type=tf.int32)
        
        # 최적 (gamma, psi) 추출
        all_gammas = gamma_history.stack()
        all_psis = psi_history.stack()
        
        best_gamma = all_gammas[best_idx]
        best_psi = all_psis[best_idx]
        
        return best_gamma, best_psi, best_idx
    
    # ========== Level 1: Level 0 의존 (tf.function) ==========
    
    @tf.function
    def _water_filling_p1j(self, Xi: tf.Tensor, power_total: tf.Tensor, n_ue: tf.Tensor) -> tf.Tensor:
        """Water-filling (P1J 완전 동일 복사)
        
        Args:
            Xi: [n_ue, n_ue] 등가 채널
            power_total: scalar total power
            n_ue: n_ue dimension
        
        Returns:
            P: [n_ue, n_ue] complex64
        """
        # Eigenvalue decomposition
        eigvals, eigvecs = tf.linalg.eigh(Xi)
        eigvals_positive = tf.maximum(tf.math.real(eigvals), 0.0)
        
        # 극소 고유값 필터링: 1/λ overflow 방지
        valid_mask = eigvals_positive > 1e-30
        
        # 채널이 거의 0인 경우
        n_valid = tf.reduce_sum(tf.cast(valid_mask, tf.float32))
        if n_valid == 0:
            return tf.zeros_like(Xi, dtype=tf.complex64)
        
        # Water-filling level ν = 1/μ 계산
        valid_eigvals = eigvals_positive * tf.cast(valid_mask, tf.float32)
        nu = self._find_inverse_water_level_tf(valid_eigvals, valid_mask, power_total)
        
        # p_i = (ν - 1/λ_i)^+
        inv_eigvals = tf.where(valid_mask, 1.0 / eigvals_positive, 0.0)
        Lambda_P_raw = tf.maximum(nu - inv_eigvals, 0.0)
        
        # 유효하지 않은 고유값에는 전력 할당 금지
        Lambda_P = Lambda_P_raw * tf.cast(valid_mask, tf.float32)
        
        # P = U_Ξ Λ_P U_Ξ^H
        P = eigvecs @ tf.linalg.diag(tf.cast(Lambda_P, tf.complex64)) @ tf.linalg.adjoint(eigvecs)
        
        return P
    
    # ========== Level 2: 외부 호출용 (Wen2011 최적화) ==========
    
    @tf.function
    def wen2011_optimize_tf(self, U_bs_beam: tf.Tensor, U_ue_beam: tf.Tensor,
                           Omega_beam: tf.Tensor, H_mean_beam: tf.Tensor) -> tuple:
        """Wen2011 최적화 (JIT 컴파일)
        
        Outer loop + Inner fixed point + Water-filling → P_opt, C_opt 반환
        
        Args:
            U_bs_beam: [B_bs, B_bs] complex64
            U_ue_beam: [B_ue, B_ue] complex64
            Omega_beam: [B_bs, B_ue] float32
            H_mean_beam: [B_bs, B_ue] complex64
        
        Returns:
            (C_opt, P_opt): C_opt float32, P_opt [B_ue, B_ue] complex64
        """
        n_bs = tf.shape(U_bs_beam)[0]
        n_ue = tf.shape(U_ue_beam)[0]
        power_budget = tf.constant(float(self.config.n_ue_layers), dtype=tf.float32)
        
        max_iter_outer = self.config.max_iter_outer
        max_iter_inner = self.config.max_iter_inner
        reg = tf.constant(self.config.regularization, dtype=tf.float32)
        reg_complex = tf.cast(reg, tf.complex64)
        
        # 초기 P
        P_init = tf.eye(self.config.n_ue_layers, dtype=tf.complex64)
        
        # tf.while_loop body
        def body(k, P_prev):
            # Step A: 고정점 계산 (inner loop)
            gamma, psi, _ = self._fixed_point_loop_tf(
                P_prev, U_bs_beam, U_ue_beam, Omega_beam, H_mean_beam, max_iter_inner, reg_complex, n_bs, n_ue
            )
            
            # Step B: Xi 계산
            Xi = self._compute_Xi_for_water_filling(
                gamma, psi, U_bs_beam, U_ue_beam, Omega_beam, H_mean_beam, reg_complex, n_bs, n_ue
            )
            
            # Step C: Water-filling
            P_new = self._water_filling_p1j(Xi, power_budget, n_ue)
            
            return k + 1, P_new
        
        # tf.while_loop condition
        def cond(k, P_prev):
            return k < max_iter_outer
        
        # Outer loop (JIT compiled)
        _, P_opt = tf.while_loop(
            cond, body,
            [0, P_init],
            maximum_iterations=max_iter_outer
        )
        
        # 최종 P_opt로 capacity 계산
        gamma_final, psi_final, _ = self._fixed_point_loop_tf(
            P_opt, U_bs_beam, U_ue_beam, Omega_beam, H_mean_beam, max_iter_inner, reg_complex, n_bs, n_ue
        )
        
        # T, R, Xi 재계산
        Omega_T_gamma = tf.matmul(tf.transpose(Omega_beam), tf.expand_dims(gamma_final, -1))
        T = U_ue_beam @ tf.linalg.diag(tf.cast(tf.squeeze(Omega_T_gamma), tf.complex64)) @ tf.linalg.adjoint(U_ue_beam)
        
        Omega_psi = tf.matmul(Omega_beam, tf.expand_dims(psi_final, -1))
        R = U_bs_beam @ tf.linalg.diag(tf.cast(tf.squeeze(Omega_psi), tf.complex64)) @ tf.linalg.adjoint(U_bs_beam)
        
        I_R = tf.eye(n_bs, dtype=tf.complex64) + R
        H_mean_H = tf.linalg.adjoint(H_mean_beam)
        Xi_final = T + H_mean_H @ tf.linalg.solve(I_R, H_mean_beam)
        
        # I(P) 계산
        I_Xi_P = tf.eye(self.config.n_ue_layers, dtype=tf.complex64) + Xi_final @ P_opt
        
        logdet1 = tf.math.real(tf.linalg.slogdet(I_Xi_P)[1])
        logdet2 = tf.math.real(tf.linalg.slogdet(I_R)[1])
        coupling_term = tf.reduce_sum(gamma_final * tf.squeeze(Omega_psi))
        
        I_nat = logdet1 + logdet2 - coupling_term
        log2 = tf.math.log(2.0)
        C_opt_tf = I_nat / log2
        
        return C_opt_tf, P_opt
    
    def wen2011_optimize(self, U_bs_beam: tf.Tensor, U_ue_beam: tf.Tensor,
                        Omega_beam: tf.Tensor, H_mean_beam: tf.Tensor) -> tuple:
        """Wen2011 최적화 wrapper (호환성)
        
        Args:
            U_bs_beam: [B_bs, B_bs] complex64
            U_ue_beam: [B_ue, B_ue] complex64
            Omega_beam: [B_bs, B_ue] float32
            H_mean_beam: [B_bs, B_ue] complex64
        
        Returns:
            (P_opt, C_opt): P_opt [B_ue, B_ue] complex64, C_opt float (bits/Hz/sec)
        """
        C_opt_tf, P_opt = self.wen2011_optimize_tf(
            U_bs_beam, U_ue_beam, Omega_beam, H_mean_beam
        )
        
        C_opt = float(C_opt_tf.numpy())
        
        return P_opt, C_opt
    
    # ========== Level 3: 범용 용량 계산 ==========
    
    @tf.function
    def _compute_beam_capacity_with_P_tf(self, U_bs_beam: tf.Tensor,
                                         U_ue_beam: tf.Tensor, Omega_beam: tf.Tensor,
                                         H_mean_beam: tf.Tensor, P: tf.Tensor) -> tf.Tensor:
        """빔 도메인 점근적 용량 계산 (임의의 P, TensorFlow)
        
        I(P) = log|I+ΞP| + log|I+R| - γ^T Ω ψ
        
        Args:
            U_bs_beam: [B_bs, B_bs] complex64
            U_ue_beam: [B_ue, B_ue] complex64
            Omega_beam: [B_bs, B_ue] float32
            H_mean_beam: [B_bs, B_ue] complex64
            P: [B_ue, B_ue] complex64 (입력 공분산)
        
        Returns:
            capacity: scalar float32 (bits/Hz/sec)
        """
        n_bs = tf.shape(U_bs_beam)[0]
        n_ue = tf.shape(U_ue_beam)[0]
        
        max_iter_inner = self.config.max_iter_inner
        reg_complex = tf.cast(tf.constant(self.config.regularization, dtype=tf.float32), tf.complex64)
        
        gamma, psi, _ = self._fixed_point_loop_tf(
            P, U_bs_beam, U_ue_beam, Omega_beam, H_mean_beam, max_iter_inner, reg_complex, n_bs, n_ue
        )
        
        Omega_T_gamma = tf.matmul(tf.transpose(Omega_beam), tf.expand_dims(gamma, -1))
        T = U_ue_beam @ tf.linalg.diag(tf.cast(tf.squeeze(Omega_T_gamma), tf.complex64)) @ tf.linalg.adjoint(U_ue_beam)
        
        Omega_psi = tf.matmul(Omega_beam, tf.expand_dims(psi, -1))
        R = U_bs_beam @ tf.linalg.diag(tf.cast(tf.squeeze(Omega_psi), tf.complex64)) @ tf.linalg.adjoint(U_bs_beam)
        
        I_R = tf.eye(n_bs, dtype=tf.complex64) + R
        H_mean_H = tf.linalg.adjoint(H_mean_beam)
        Xi = T + H_mean_H @ tf.linalg.solve(I_R, H_mean_beam)
        
        I_Xi_P = tf.eye(self.config.n_ue_layers, dtype=tf.complex64) + Xi @ P
        
        logdet1 = tf.math.real(tf.linalg.slogdet(I_Xi_P)[1])
        logdet2 = tf.math.real(tf.linalg.slogdet(I_R)[1])
        coupling_term = tf.reduce_sum(gamma * tf.squeeze(Omega_psi))
        
        I_nat = logdet1 + logdet2 - coupling_term
        
        return I_nat / tf.cast(tf.math.log(2.0), tf.float32)
    

# ----- P1L_ResultManager -----
class P1L_ResultManager:
    """빔 관리 결과 CSV 저장"""
    
    def __init__(self, config: P1L_Config):
        self.config = config
        self._csv_timestamp = None
        self._csv_path = None
    
    def _init_csv(self, area: int, freq: float):
        """CSV 파일 초기화 (재시작 시 기존 파일 계속 사용)"""
        if self._csv_path is not None:
            # 이미 설정된 경우 (재시작) 그대로 사용
            return
        
        if self._csv_timestamp is None:
            utc_plus_9 = datetime.utcnow() + timedelta(hours=9)
            self._csv_timestamp = utc_plus_9.strftime('%Y%m%d_%H%M%S')
        
        csv_filename = f"Area{area}_{freq}GHz_SU_BM_{self._csv_timestamp}.csv"
        self._csv_path = os.path.join(self.config.P1L_OUTPUT_DIR, csv_filename)
    
    def get_processed_ues(self, area: int, freq: float) -> set:
        """CSV에서 Stage 1 완료된 UE 목록 반환 (기존 파일 복사 후 이어서 작성)
        
        Returns:
            set: Stage 1 final 저장된 UE 번호 집합
        """
        # 최신 CSV 파일 찾기
        pattern = os.path.join(self.config.P1L_OUTPUT_DIR, f"Area{area}_{freq}GHz_SU_BM_*.csv")
        csv_files = glob.glob(pattern)
        
        if not csv_files:
            return set()
        
        # 가장 최근 파일 선택
        latest_csv = max(csv_files, key=os.path.getmtime)
        
        # CSV에서 처리된 UE 읽기
        processed_ues = set()
        try:
            with open(latest_csv, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Stage 1 final 단계가 기록된 UE만
                    if row.get('stage') == 'S1' and row.get('step') == 'final':
                        ue_num = int(row['ue'].strip())
                        processed_ues.add(ue_num)
            
            if processed_ues:
                # 새 타임스탬프 생성
                utc_plus_9 = datetime.utcnow() + timedelta(hours=9)
                self._csv_timestamp = utc_plus_9.strftime('%Y%m%d_%H%M%S')
                
                # 새 파일명으로 복사
                csv_filename = f"Area{area}_{freq}GHz_SU_BM_{self._csv_timestamp}.csv"
                new_csv_path = os.path.join(self.config.P1L_OUTPUT_DIR, csv_filename)
                
                shutil.copy2(latest_csv, new_csv_path)
                
                self._csv_path = new_csv_path
                
                print(f"기존 결과 발견: {len(processed_ues)}개 UE 이미 처리됨 (재시작)")
                print(f"  기존 파일: {os.path.basename(latest_csv)}")
                print(f"  새 파일: {os.path.basename(new_csv_path)}")
        
        except Exception as e:
            print(f"경고: 기존 결과 파일 읽기 실패 ({e}), 처음부터 시작")
            return set()
        
        return processed_ues
    
    def save_stage1_single_beam(self, area: int, freq: float, ue: int,
                                ell: int, C_beam: float, C_AE: float):
        """Stage 1-1: 단일 빔 평가 결과 (저장 비활성화)"""
        # 단일 빔 결과는 CSV에 저장하지 않음 (너무 많고 불필요)
        pass
    
    def save_stage1_threshold(self, area: int, freq: float, ue: int,
                             Beams_UE_filtered: list, alpha_ue: float, C_AE: float):
        """Stage 1-2: Threshold 결과 (저장 비활성화)"""
        # Threshold 단계는 제거되어 저장하지 않음
        pass
    
    def save_stage1_combination(self, area: int, freq: float, ue: int,
                                Beams_UE: list, C_beam: float, C_AE: float,
                                is_best: bool = False):
        """Stage 1-3: 조합 탐색 결과 (최종만 저장)"""
        if not is_best:
            return
        
        if self._csv_path is None:
            self._init_csv(area, freq)
        
        headers = ['ue', 'stage', 'step', 'C_ref', 'C_now', 'loss_pct', 'beams', 'S2_improv', 'Lambda_bs_ratios', 'Lambda_ue_ratios']
        
        file_exists = os.path.exists(self._csv_path)
        
        with open(self._csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(headers)
            
            loss_pct = (C_AE - C_beam) / C_AE * 100
            Beams_UE_str = ','.join(map(str, Beams_UE))
            
            row = [
                f"{ue:4d}",
                "S1",
                "combi",
                f"{C_AE:8.4f}",
                f"{C_beam:8.4f}",
                f"{loss_pct:6.2f}",
                Beams_UE_str,
                "",
                "",
                ""
            ]
            writer.writerow(row)
    
    def save_stage1_final(self, area: int, freq: float, ue: int,
                         Beams_UE_opt: list, C_S1_ref: float, C_AE: float,
                         Lambda_bs_ratios_str: str, Lambda_ue_ratios_str: str):
        """Stage 1-4: Wen2011 최적화 후 최종 결과 (eigenvalue ratios 포함)"""
        if self._csv_path is None:
            self._init_csv(area, freq)
        
        with open(self._csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            
            loss_pct = (C_AE - C_S1_ref) / C_AE * 100
            Beams_UE_opt_str = ','.join(map(str, Beams_UE_opt))
            
            row = [
                f"{ue:4d}",
                "S1",
                "final",
                f"{C_AE:8.4f}",
                f"{C_S1_ref:8.4f}",
                f"{loss_pct:6.2f}",
                Beams_UE_opt_str,
                "",
                Lambda_bs_ratios_str,
                Lambda_ue_ratios_str
            ]
            writer.writerow(row)
    
    def save_stage2_iteration(self, area: int, freq: float, ue: int,
                             L_bs: int, added_beam: int, Beams_BS_greedy: list,
                             C_beam: float, delta_C: float, 
                             improvement_ratio: float, C_S1_ref: float):
        """Stage 2: BS 빔 추가 반복"""
        if self._csv_path is None:
            self._init_csv(area, freq)
        
        with open(self._csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            
            loss_pct = (C_S1_ref - C_beam) / C_S1_ref * 100
            Beams_BS_str = ','.join(map(str, Beams_BS_greedy))
            
            # improvement_ratio가 inf인 경우 빈 문자열로 출력
            if math.isinf(improvement_ratio):
                S2_improv_str = ""
            else:
                S2_improv_str = f"{improvement_ratio:8.4f}"
            
            row = [
                f"{ue:4d}",
                "S2",
                "greed",
                f"{C_S1_ref:8.4f}",
                f"{C_beam:8.4f}",
                f"{loss_pct:6.2f}",
                Beams_BS_str,
                S2_improv_str,
                "",
                ""
            ]
            writer.writerow(row)
    
    def save_ue_result_static(self, area: int, freq: float, ue: int,
                             ue_beams: str, bs_beams: str,
                             C_AE: float, C_ue_history: str, C_S1: float,
                             C_bs_history: str, C_S2: float,
                             Lambda_ue_ratios: str, Lambda_bs_ratios: str):
        """새로운 형식: UE별 종합 결과 저장 (config-driven static)
        
        CSV Format:
        ue,ue_beams,bs_beams,C_AE,C_ue_history,C_S1,C_bs_history,C_S2,Lambda_ue_ratios,Lambda_bs_ratios
        """
        if self._csv_path is None:
            self._init_csv(area, freq)
        
        headers = ['ue', 'ue_beams', 'bs_beams', 'C_AE', 
                  'C_ue_history', 'C_S1', 'C_bs_history', 'C_S2',
                  'Lambda_ue_ratios', 'Lambda_bs_ratios']
        
        file_exists = os.path.exists(self._csv_path)
        
        with open(self._csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(headers)
            
            row = [
                ue,
                ue_beams,
                bs_beams,
                f"{C_AE:.2f}",
                C_ue_history,
                f"{C_S1:.2f}",
                C_bs_history,
                f"{C_S2:.2f}",
                Lambda_ue_ratios,
                Lambda_bs_ratios
            ]
            writer.writerow(row)

# ===== LEVEL 2: 복합 의존 클래스 =====

# ----- GreedyBeamSelector -----
class GreedyBeamSelector:
    """Greedy 빔 선택 알고리즘 (P1L_2510v2.tex Section 7)"""
    
    def __init__(self, config: P1L_Config):
        self.config = config
        self.capacity_calc = BeamDomainCapacity(config)
        
        # DFT 코드북 생성 (TensorFlow, 한 번만 생성)
        self.F_bs = DFTCodebook.generate_2d_dft_codebook_tf(
            config.bs_ant_per_dim, config.bs_oversample
        )  # [16, 64] tf.complex64
        self.F_ue = DFTCodebook.generate_2d_dft_codebook_tf(
            config.ue_ant_per_dim, config.ue_oversample
        )  # [4, 16] tf.complex64
    
    @tf.function
    def select_ue_beams_stage1(self, U_bs: tf.Tensor, U_ue: tf.Tensor,
                               Omega: tf.Tensor, H_mean: tf.Tensor,
                               C_AE: tf.Tensor) -> tuple:
        """Stage 1: UE + BS 빔 선택 + P 최적화 (전체 tf.function, config-driven static)
        
        1-1: UE 빔 평가 (W_bs=I, 1024 AE)
        1-2: UE 빔 선택 (W_bs=I, 1024 AE)
        1-3a: BS 빔 Greedy (UE 고정, config.n_bs_beams_static개, P=I)
        1-3b: P 최적화 (UE + BS 고정)
        
        Args:
            U_bs: [n_bs, n_bs] complex64
            U_ue: [n_ue, n_ue] complex64
            Omega: [n_bs, n_ue] float32
            H_mean: [n_bs, n_ue] complex64
            C_AE: scalar tensor (AE 도메인 기준 용량)
        
        Returns:
            tuple: (Beams_UE_opt, Beams_BS_S1, P_S1_opt, C_S1,
                   C_ue_history, Lambda_ue_ratios, Lambda_bs_ratios)
        """
        tf.print("  Stage 1: UE + BS 빔 선택 + P 최적화")
        
        # [1-1] 단일 빔 벡터 평가
        tf.print("    [1-1] 단일 빔 평가 (UE=", self.config.n_cb_ue, "개 동시, W_bs=I, 1024 AE)")
        
        # W_bs = Identity (1024 AE 직접 사용)
        n_bs_total = self.config.n_bs
        W_bs_identity = tf.eye(n_bs_total, dtype=tf.complex64)
        P_eye = tf.eye(self.config.n_ue_layers, dtype=tf.complex64)
        
        # 16개 UE 빔포밍 행렬 미리 생성
        W_ue_list = []
        for ell in range(self.config.n_cb_ue):
            L_sel_single = tf.constant([ell] * self.config.n_ue_layers, dtype=tf.int32)
            W_ue = BeamformingMatrix.construct_beamforming_blockdiag_tf(
                self.F_ue, L_sel_single, self.config.n_ue_layers
            )
            W_ue_list.append(W_ue)
        
        W_ue_stack = tf.stack(W_ue_list, axis=0)  # [16, 16, 4]
        
        # 16개 빔 동시 평가
        C_beams = self._compute_capacity_for_multiple_beamformings_tf(
            U_bs, U_ue, Omega, H_mean, W_bs_identity, W_ue_stack, P_eye
        )  # [16] float32
        
        # [1-1] TensorFlow 연산으로 정렬 (내림차순)
        sorted_indices = tf.argsort(C_beams, direction='DESCENDING')  # [16] 정렬된 인덱스
        tf.print("      상위8 빔 평가 완료")
        
        # [1-2] UE 빔 선택 (static)
        all_beams = list(range(self.config.n_cb_ue))
        Beams_UE_opt, C_ue_history = self._ue_search_greedy_static(
            U_bs, U_ue, Omega, H_mean, W_bs_identity, P_eye, 
            all_beams, sorted_indices
        )
        
        # [1-3a] BS 빔 Greedy Addition (UE 고정, P=I, static)
        # Beams_UE_opt는 tensor list이므로 tf.stack() 사용
        Beams_UE_opt_stacked = tf.stack(Beams_UE_opt)  # [n_ue_beams]
        W_ue_opt = BeamformingMatrix.construct_beamforming_blockdiag_tf(
            self.F_ue, Beams_UE_opt_stacked, self.config.n_ue_layers
        )
        
        Beams_BS_S1, C_bs_history = self._bs_greedy_addition_static(
            U_bs, U_ue, Omega, H_mean, W_ue_opt, P_eye
        )
        
        # [1-3b] P 최적화 (UE + BS 빔 고정)
        tf.print("    [1-3b] P 최적화")
        
        # Beams_BS_S1도 tensor list이므로 tf.stack() 사용
        Beams_BS_S1_stacked = tf.stack(Beams_BS_S1)  # [n_bs_beams]
        W_bs_S1 = BeamformingMatrix.construct_beamforming_blockdiag_tf(
            self.F_bs, Beams_BS_S1_stacked, self.config.n_bs_layers
        )
        
        # 빔 도메인 변환
        H_mean_beam = BeamDomainTransform.transform_mean_channel(H_mean, W_bs_S1, W_ue_opt)
        _, U_bs_beam, Lambda_bs_beam = BeamDomainTransform.transform_bs_covariance(
            U_bs, U_ue, Omega, W_bs_S1, W_ue_opt
        )
        _, U_ue_beam, Lambda_ue_beam = BeamDomainTransform.transform_ue_covariance(
            U_bs, U_ue, Omega, W_bs_S1, W_ue_opt
        )
        Omega_beam = BeamDomainTransform.transform_coupling_matrix(
            U_bs, U_ue, Omega, W_bs_S1, W_ue_opt, U_bs_beam, U_ue_beam
        )
        
        # P 최적화 (tf.function 직접 호출)
        C_S1, P_S1_opt = self.capacity_calc.wen2011_optimize_tf(
            U_bs_beam, U_ue_beam, Omega_beam, H_mean_beam
        )
        
        # Lambda eigenvalue ratios (descending order)
        mean_bs = tf.reduce_mean(Lambda_bs_beam)
        mean_ue = tf.reduce_mean(Lambda_ue_beam)
        Lambda_bs_ratios = tf.sort(Lambda_bs_beam / mean_bs, direction='DESCENDING')
        Lambda_ue_ratios = tf.sort(Lambda_ue_beam / mean_ue, direction='DESCENDING')
        
        tf.print("  Stage 1 완료: UE=", self.config.n_ue_beams_static, ", BS=", self.config.n_bs_beams_static, ", C_S1=", C_S1)
        
        return (Beams_UE_opt, Beams_BS_S1, P_S1_opt, C_S1,
                C_ue_history, Lambda_ue_ratios, Lambda_bs_ratios)
    
    @tf.function
    def select_bs_beams_stage2_static(self, U_bs: tf.Tensor, U_ue: tf.Tensor,
                                      Omega: tf.Tensor, H_mean: tf.Tensor,
                                      Beams_UE_fixed, P_fixed: tf.Tensor):
        """Stage 2: BS 빔 재선택 (단일 tf.function, config-driven static)
        
        Args:
            U_bs: [n_bs, n_bs] complex64
            U_ue: [n_ue, n_ue] complex64
            Omega: [n_bs, n_ue] float32
            H_mean: [n_bs, n_ue] complex64
            Beams_UE_fixed: Python list of UE beam indices (traced as constant)
            P_fixed: [n_ue_layers, n_ue_layers] optimized from Stage 1
        
        Returns:
            (Beams_BS_S2, C_bs_history)
        """
        tf.print("  Stage 2: BS 빔 재선택 (UE + P 고정)")
        
        # UE 빔포밍 고정
        W_ue_fixed = BeamformingMatrix.construct_beamforming_blockdiag_tf(
            self.F_ue, tf.constant(Beams_UE_fixed, dtype=tf.int32), 
            self.config.n_ue_layers
        )
        
        # BS 빔 static 추가
        Beams_BS_S2, C_bs_history = self._bs_greedy_addition_static(
            U_bs, U_ue, Omega, H_mean, W_ue_fixed, P_fixed
        )
        
        tf.print("  Stage 2 완료: BS=", self.config.n_bs_beams_static, 
                 ", C_S2=", C_bs_history[-1])
        
        return Beams_BS_S2, C_bs_history
    
    def _ue_search_greedy_static(self, U_bs: tf.Tensor, U_ue: tf.Tensor,
                                 Omega: tf.Tensor, H_mean: tf.Tensor,
                                 W_bs: tf.Tensor, P_eye: tf.Tensor,
                                 Beams_UE_filtered: list, 
                                 sorted_indices: tf.Tensor):
        """Stage 1-2: UE 빔 고정 선택 (TF ops only, config-driven static)
        
        Args:
            Beams_UE_filtered: 사용 가능한 모든 UE 빔 (16개 전체)
            sorted_indices: 정렬된 빔 인덱스 (tensor, 내림차순)
        
        Returns:
            (Beams_best, C_ue_history)
            Beams_best: Python list of best beam indices
            C_ue_history: [n_ue_beams_static] capacity tensor
        """
        n_ue_beams = self.config.n_ue_beams_static
        
        # 시작 빔 선택 (상위 N개, TensorFlow 연산)
        top_n_indices = sorted_indices[:self.config.n_ue_greedy_starts]
        
        tf.print("      [1-2] UE 빔 고정 선택:", n_ue_beams, "개")
        
        candidates = list(Beams_UE_filtered)
        candidates_tf = tf.constant(candidates, dtype=tf.int32)
        
        # 첫 번째 시작점으로 초기화
        start_beam_0 = top_n_indices[0]
        selected_0 = [start_beam_0]
        C_history_list_0 = []
        
        # 첫 번째 빔 용량
        trial_tf = tf.fill([n_ue_beams], start_beam_0)
        W_ue = BeamformingMatrix.construct_beamforming_blockdiag_tf(
            self.F_ue, trial_tf, self.config.n_ue_layers
        )
        C_first = self._compute_beam_capacity_tf(
            U_bs, U_ue, Omega, H_mean, W_bs, W_ue, P_eye
        )
        C_history_list_0.append(C_first)
        
        # Greedy expansion (첫 번째 시작점)
        for layer in range(1, n_ue_beams):
            W_ue_stack_list = []
            for cand in candidates:
                selected_stacked = tf.stack(selected_0)
                cand_tensor = tf.constant([cand], dtype=tf.int32)
                trial_concat = tf.concat([selected_stacked, cand_tensor], axis=0)
                
                current_len = len(selected_0) + 1
                pad_len = n_ue_beams - current_len
                if pad_len > 0:
                    last_elem = trial_concat[-1]
                    padding = tf.fill([pad_len], last_elem)
                    trial_padded = tf.concat([trial_concat, padding], axis=0)
                else:
                    trial_padded = trial_concat
                
                W_ue = BeamformingMatrix.construct_beamforming_blockdiag_tf(
                    self.F_ue, trial_padded, self.config.n_ue_layers
                )
                W_ue_stack_list.append(W_ue)
            
            W_ue_stack = tf.stack(W_ue_stack_list, axis=0)
            C_trials = self._compute_capacity_for_multiple_beamformings_tf(
                U_bs, U_ue, Omega, H_mean, W_bs, W_ue_stack, P_eye
            )
            
            best_idx = tf.argmax(C_trials, output_type=tf.int32)
            best_beam_tf = tf.gather(candidates_tf, best_idx)
            C_best = C_trials[best_idx]
            
            selected_0.append(best_beam_tf)
            C_history_list_0.append(C_best)
        
        Beams_global_best = selected_0
        C_global_best_history = tf.stack(C_history_list_0)
        C_global_best = C_global_best_history[-1]
        tf.print("        Start 1:", Beams_global_best, ", C=", C_global_best)
        
        # 나머지 시작점들 (비교 후 업데이트)
        for start_idx in range(1, self.config.n_ue_greedy_starts):
            start_beam = top_n_indices[start_idx]
            selected = [start_beam]
            C_history_list = []
            
            # 첫 번째 빔 용량
            trial_tf = tf.fill([n_ue_beams], start_beam)
            W_ue = BeamformingMatrix.construct_beamforming_blockdiag_tf(
                self.F_ue, trial_tf, self.config.n_ue_layers
            )
            C_first = self._compute_beam_capacity_tf(
                U_bs, U_ue, Omega, H_mean, W_bs, W_ue, P_eye
            )
            C_history_list.append(C_first)
            
            # Greedy expansion
            for layer in range(1, n_ue_beams):
                W_ue_stack_list = []
                for cand in candidates:
                    selected_stacked = tf.stack(selected)
                    cand_tensor = tf.constant([cand], dtype=tf.int32)
                    trial_concat = tf.concat([selected_stacked, cand_tensor], axis=0)
                    
                    current_len = len(selected) + 1
                    pad_len = n_ue_beams - current_len
                    if pad_len > 0:
                        last_elem = trial_concat[-1]
                        padding = tf.fill([pad_len], last_elem)
                        trial_padded = tf.concat([trial_concat, padding], axis=0)
                    else:
                        trial_padded = trial_concat
                    
                    W_ue = BeamformingMatrix.construct_beamforming_blockdiag_tf(
                        self.F_ue, trial_padded, self.config.n_ue_layers
                    )
                    W_ue_stack_list.append(W_ue)
                
                W_ue_stack = tf.stack(W_ue_stack_list, axis=0)
                C_trials = self._compute_capacity_for_multiple_beamformings_tf(
                    U_bs, U_ue, Omega, H_mean, W_bs, W_ue_stack, P_eye
                )
                
                best_idx = tf.argmax(C_trials, output_type=tf.int32)
                best_beam_tf = tf.gather(candidates_tf, best_idx)
                C_best = C_trials[best_idx]
                
                selected.append(best_beam_tf)
                C_history_list.append(C_best)
            
            C_history = tf.stack(C_history_list)
            C_final = C_history[-1]
            
            # 더 나은 결과이면 업데이트
            if C_final > C_global_best:
                C_global_best = C_final
                Beams_global_best = selected
                C_global_best_history = C_history
            
            tf.print("        Start", start_idx + 1, ":", selected, ", C=", C_final)
        
        tf.print("        → 최적:", Beams_global_best, ", C=", C_global_best)
        
        return Beams_global_best, C_global_best_history
    
    @tf.function
    def _compute_beam_capacity_tf(self, U_bs: tf.Tensor, U_ue: tf.Tensor,
                                   Omega: tf.Tensor, H_mean: tf.Tensor,
                                   W_bs: tf.Tensor, W_ue: tf.Tensor,
                                   P: tf.Tensor) -> tf.Tensor:
        """빔 도메인 용량 계산 (tf.function, JIT)
        
        Args:
            U_bs: [n_bs, n_bs] complex64
            U_ue: [n_ue, n_ue] complex64
            Omega: [n_bs, n_ue] float32
            H_mean: [n_bs, n_ue] complex64
            W_bs: [n_bs, B_bs] complex64
            W_ue: [n_ue, B_ue] complex64
            P: [B_ue, B_ue] complex64
            
        Returns:
            capacity: scalar float32
        """
        H_mean_beam = BeamDomainTransform.transform_mean_channel(H_mean, W_bs, W_ue)
        
        _, U_bs_beam, _ = BeamDomainTransform.transform_bs_covariance(
            U_bs, U_ue, Omega, W_bs, W_ue
        )
        
        _, U_ue_beam, _ = BeamDomainTransform.transform_ue_covariance(
            U_bs, U_ue, Omega, W_bs, W_ue
        )
        
        Omega_beam = BeamDomainTransform.transform_coupling_matrix(
            U_bs, U_ue, Omega, W_bs, W_ue, U_bs_beam, U_ue_beam
        )
        
        capacity = self.capacity_calc._compute_beam_capacity_with_P_tf(
            U_bs_beam, U_ue_beam, Omega_beam, H_mean_beam, P
        )
        
        return capacity
    
    @tf.function
    def _compute_capacity_for_multiple_beamformings_tf(self, U_bs: tf.Tensor, U_ue: tf.Tensor,
                                                       Omega: tf.Tensor, H_mean: tf.Tensor,
                                                       W_bs_or_stack: tf.Tensor,
                                                       W_ue_or_stack: tf.Tensor,
                                                       P: tf.Tensor) -> tf.Tensor:
        """여러 빔포밍 조합에 대한 용량 계산 (tf.function, JIT)
        
        정적 for loop이므로 tf.function으로 JIT 컴파일 가능
        
        Args:
            W_bs_or_stack: [n_bs, B_bs] 또는 [n_beams, n_bs, B_bs]
            W_ue_or_stack: [n_ue, B_ue] 또는 [n_beams, n_ue, B_ue]
            P: [B_ue, B_ue] complex64
            
        Returns:
            capacities: [n_beams] float32
        """
        W_bs_is_stack = len(W_bs_or_stack.shape) == 3
        W_ue_is_stack = len(W_ue_or_stack.shape) == 3
        
        if not W_bs_is_stack and not W_ue_is_stack:
            C = self._compute_beam_capacity_tf(
                U_bs, U_ue, Omega, H_mean, W_bs_or_stack, W_ue_or_stack, P
            )
            return tf.expand_dims(C, 0)
        
        n_beams = tf.shape(W_bs_or_stack)[0] if W_bs_is_stack else tf.shape(W_ue_or_stack)[0]
        
        capacities_array = tf.TensorArray(dtype=tf.float32, size=n_beams, dynamic_size=False)
        
        for i in tf.range(n_beams):
            W_bs = W_bs_or_stack[i] if W_bs_is_stack else W_bs_or_stack
            W_ue = W_ue_or_stack[i] if W_ue_is_stack else W_ue_or_stack
            
            C = self._compute_beam_capacity_tf(U_bs, U_ue, Omega, H_mean, W_bs, W_ue, P)
            capacities_array = capacities_array.write(i, C)
        
        return capacities_array.stack()
    
    @tf.function
    def _evaluate_bs_candidates_tf(self, U_bs: tf.Tensor, U_ue: tf.Tensor,
                                    Omega: tf.Tensor, H_mean: tf.Tensor,
                                    Beams_BS_current: tf.Tensor,
                                    W_ue_fixed: tf.Tensor, P: tf.Tensor) -> tf.Tensor:
        """64개 BS 빔 후보 동시 평가 (tf.function, JIT)
        
        Args:
            Beams_BS_current: [L_bs] 현재 선택된 BS 빔
            W_ue_fixed: [n_ue, n_ue_layers] 고정된 UE 빔포밍
            P: [n_ue_layers, n_ue_layers] 입력 공분산
            
        Returns:
            C_trials: [64] 각 후보의 용량
        """
        n_cb_bs = 64
        L_bs = tf.shape(Beams_BS_current)[0]
        
        candidates = tf.range(n_cb_bs, dtype=tf.int32)
        
        # Cast to int32 for dtype consistency
        Beams_BS_tiled = tf.tile(tf.expand_dims(Beams_BS_current, 0), [n_cb_bs, 1])
        Beams_BS_tiled = tf.cast(Beams_BS_tiled, tf.int32)
        Beams_BS_trials = tf.concat([Beams_BS_tiled, tf.expand_dims(candidates, 1)], axis=1)
        
        # Vectorized batch construction (single operation)
        W_bs_stack = BeamformingMatrix.construct_beamforming_blockdiag_tf_batch(
            self.F_bs, Beams_BS_trials, self.config.n_bs_layers
        )
        
        C_trials = self._compute_capacity_for_multiple_beamformings_tf(
            U_bs, U_ue, Omega, H_mean, W_bs_stack, W_ue_fixed, P
        )
        
        return C_trials
    
    def _bs_greedy_addition_static(self, U_bs: tf.Tensor, U_ue: tf.Tensor,
                                    Omega: tf.Tensor, H_mean: tf.Tensor,
                                    W_ue_fixed: tf.Tensor, P: tf.Tensor):
        """BS 빔 고정 추가 (TF ops only, config-driven static)
        
        Args:
            W_ue_fixed: [n_ue, n_ue_layers] 고정된 UE 빔포밍
            P: [n_ue_layers, n_ue_layers] 입력 공분산
            
        Returns:
            (Beams_BS_list, C_bs_history)
            Beams_BS_list: Python list of tensors (length = n_bs_beams_static)
            C_bs_history: [n_bs_beams_static] capacity tensor
        """
        n_bs_add = self.config.n_bs_beams_static
        Beams_BS_list = []
        C_bs_history_list = []
        
        tf.print("      [1-3a] BS 빔 고정 추가:", n_bs_add, "개")
        
        for add_step in range(n_bs_add):
            # Beams_BS_list가 tensor list이므로 tf.stack() 사용
            if Beams_BS_list:
                Beams_BS_tf = tf.stack(Beams_BS_list)
            else:
                Beams_BS_tf = tf.constant([], dtype=tf.int32)
            
            C_trials = self._evaluate_bs_candidates_tf(
                U_bs, U_ue, Omega, H_mean, Beams_BS_tf, W_ue_fixed, P
            )
            
            best_idx = tf.argmax(C_trials, output_type=tf.int32)
            C_current = C_trials[best_idx]
            
            Beams_BS_list.append(best_idx)
            C_bs_history_list.append(C_current)
            
            tf.print("        L_bs=", add_step + 1, ": beam=", best_idx, ", C=", C_current)
        
        C_bs_history = tf.stack(C_bs_history_list)
        return Beams_BS_list, C_bs_history
    


# ===== LEVEL 3: 최상위 실행 =====
def main():
    """P1L: 빔 관리 메인 실행"""
    
    print("=" * 80)
    print("P1L v3: Weichselberger SU-MIMO 업링크 빔 관리 (tf.function JIT)")
    print("=" * 80)
    print()
    
    # 설정 초기화
    config = P1L_Config()
    
    if not config.ue_list:
        print("경고: 처리 대상 UE가 없습니다.")
        return
    
    # 모듈 초기화
    data_loader = DataLoader(config)
    beam_selector = GreedyBeamSelector(config)
    result_mgr = P1L_ResultManager(config)
    
    total_ues = len(config.ue_list)
    
    print(f"빔 관리 계획:")
    print(f"  - 총 UE: {total_ues}개")
    print(f"  - Stage 1: UE + BS 빔 선택 + P 최적화")
    print(f"    - 1-1/1-2: UE 빔 선택 (W_bs=I, Greedy, {config.n_ue_greedy_starts}개 시작점)")
    print(f"    - 1-3a: BS 빔 선택 (UE 고정, P=I, Greedy 0→64개)")
    print(f"    - 1-3b: P 최적화 (UE + BS 고정)")
    print(f"  - Stage 2: BS 빔 재선택 (UE + P 고정, 0→64개)")
    print(f"  - 재시작 모드: {'기존 결과 이어서' if config.resume_from_existing else '새로 시작'}")
    print()
    
    overall_start = time.time()
    
    # 이미 처리된 UE 확인 (재시작 지원)
    if config.resume_from_existing and config.ue_list:
        area, freq, _, _ = config.ue_list[0]
        processed_ues = result_mgr.get_processed_ues(area, freq)
    else:
        processed_ues = set()
    
    # 처리되지 않은 UE만 필터링
    pending_ues = [(area, freq, ue, p1j_path) for area, freq, ue, p1j_path in config.ue_list 
                   if ue not in processed_ues]
    
    skipped_count = len(config.ue_list) - len(pending_ues)
    if skipped_count > 0:
        print(f"이미 처리된 UE {skipped_count}개 스킵")
    
    stage1_success = skipped_count
    stage1_total_time = 0.0
    stage2_success = skipped_count
    stage2_total_time = 0.0
    
    processed_count = skipped_count
    
    # 개별 UE 처리
    for ue_idx, (area, freq, ue, p1j_path) in enumerate(pending_ues, 1):
        print(f"\n{'='*80}")
        print(f"UE {ue_idx}/{len(pending_ues)}: Area{area}_{freq}GHz_UE{ue}")
        print(f"{'='*80}")
        
        try:
            # 데이터 로딩
            p1j_data = data_loader.load_p1j_result(p1j_path)
            channel_data = data_loader.load_p1i_channel_params(area, freq, ue)
            
            C_AE = p1j_data['C_AE']
            C_AE_tf = tf.constant(C_AE, dtype=tf.float32)
            
            # ========== Stage 1: UE + BS 빔 선택 + P 최적화 ==========
            stage1_start = time.time()
            
            (Beams_UE_opt, Beams_BS_S1, P_S1_opt, C_S1,
             C_ue_history, Lambda_ue_ratios, Lambda_bs_ratios) = beam_selector.select_ue_beams_stage1(
                channel_data['U_bs'], channel_data['U_ue'], 
                channel_data['Omega'], channel_data['H_mean'], C_AE_tf
            )
            
            stage1_elapsed = time.time() - stage1_start
            stage1_success += 1
            stage1_total_time += stage1_elapsed
            
            # Convert tensors to Python for display/CSV
            C_S1_val = float(C_S1.numpy())
            loss_pct = (C_AE - C_S1_val) / C_AE * 100
            
            # Convert UE beam indices and histories to strings
            ue_beams_str = ','.join([str(int(b.numpy() if hasattr(b, 'numpy') else b)) for b in Beams_UE_opt])
            C_ue_history_str = ','.join([f"{float(c.numpy()):.2f}" for c in C_ue_history])
            Lambda_ue_str = ','.join([f"{float(r.numpy()):.2f}" for r in Lambda_ue_ratios])
            Lambda_bs_str = ','.join([f"{float(r.numpy()):.2f}" for r in Lambda_bs_ratios])
            
            print(f"  Stage 1 완료: UE={config.n_ue_beams_static}개, BS={config.n_bs_beams_static}개, "
                  f"Loss={loss_pct:.1f}%")
            
            # ========== Stage 2: BS 빔 재선택 (static, @tf.function) ==========
            stage2_start = time.time()
            
            # Convert UE beams to Python list for @tf.function
            Beams_UE_list = [int(b.numpy() if hasattr(b, 'numpy') else b) for b in Beams_UE_opt]
            
            Beams_BS_S2, C_bs_S2_history = beam_selector.select_bs_beams_stage2_static(
                channel_data['U_bs'], channel_data['U_ue'],
                channel_data['Omega'], channel_data['H_mean'],
                Beams_UE_list, P_S1_opt
            )
            
            C_S2_val = float(C_bs_S2_history[-1].numpy())
            
            # Convert Stage 2 BS beams and history to strings (최종 결과)
            bs_beams_str = ','.join([str(int(b.numpy() if hasattr(b, 'numpy') else b)) for b in Beams_BS_S2])
            C_bs_S2_history_str = ','.join([f"{float(c.numpy()):.2f}" for c in C_bs_S2_history])
            
            stage2_elapsed = time.time() - stage2_start
            stage2_success += 1
            stage2_total_time += stage2_elapsed
            
            print(f"  Stage 2 완료: BS={config.n_bs_beams_static}개, C={C_S2_val:.2f}")
            
            # 종합 결과 저장 (새로운 형식)
            result_mgr.save_ue_result_static(
                area, freq, ue,
                ue_beams=ue_beams_str,
                bs_beams=bs_beams_str,
                C_AE=C_AE,
                C_ue_history=C_ue_history_str,
                C_S1=C_S1_val,
                C_bs_history=C_bs_S2_history_str,
                C_S2=C_S2_val,
                Lambda_ue_ratios=Lambda_ue_str,
                Lambda_bs_ratios=Lambda_bs_str
            )
            
            processed_count += 1
            
            # 진행 상황
            avg1 = stage1_total_time / (stage1_success - skipped_count) if stage1_success > skipped_count else 0
            avg2 = stage2_total_time / (stage2_success - skipped_count) if stage2_success > skipped_count else 0
            remaining = total_ues - processed_count
            eta_sec = (avg1 + avg2) * remaining
            eta_str = f"{eta_sec/3600:.1f}h" if eta_sec >= 3600 else f"{eta_sec/60:.1f}m" if eta_sec >= 60 else f"{eta_sec:.0f}s"
            
            print(f"  완료: Stage1={stage1_elapsed:.1f}s, Stage2={stage2_elapsed:.1f}s")
            print(f"  진행률: {processed_count}/{total_ues} UEs, 평균={(avg1+avg2):.1f}s/UE, ETA={eta_str}")
            
        except Exception as e:
            print(f"UE 처리 실패: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # 완료 메시지
    total_elapsed = time.time() - overall_start
    print(f"\n{'='*80}")
    print(f"P1L v3 완료 (tf.function JIT 최적화)")
    print(f"{'='*80}")
    print(f"총 {total_ues}개 UE 처리")
    print(f"  - Stage 1 성공: {stage1_success}개 (UE + BS 빔 + P 최적화)")
    print(f"  - Stage 2 성공: {stage2_success}개 (BS 빔 재선택)")
    print(f"총 소요 시간: {total_elapsed/60:.1f}분")
    if stage1_success > skipped_count:
        print(f"  - Stage 1 평균: {stage1_total_time/(stage1_success-skipped_count):.1f}s/UE")
    if stage2_success > skipped_count:
        print(f"  - Stage 2 평균: {stage2_total_time/(stage2_success-skipped_count):.1f}s/UE")
    if (stage1_success + stage2_success) > 2*skipped_count:
        avg_per_ue = (stage1_total_time + stage2_total_time) / (stage1_success - skipped_count)
        print(f"  - 전체 평균: {avg_per_ue:.1f}s/UE")
    print(f"저장 위치: {config.P1L_OUTPUT_DIR}")
    print()

if __name__ == "__main__":
    main()

