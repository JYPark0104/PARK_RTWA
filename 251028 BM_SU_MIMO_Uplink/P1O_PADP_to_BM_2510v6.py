#!/usr/bin/env python3
# ======================================================================
# P1O_PADP_to_BM_2510v6.py
# P1O v6 Phase 1: Outer Loop Dynamic Convergence (eps_outer)
# 
# === Purpose ===
# Uplink beam management with optimized Stage 3/4 pattern search strategy
# - Stage 1: UE TRX selection (4 TRX, TRX-level)
# - Stage 2: BS TRX cyclic pattern (256 TRX, pattern length 32, TRX-level)
# - Stage 3: BS Layer best pattern selection (search length 1-8, return best overall)
# - Stage 4: BS Layer minimal greedy add (using Stage 3 unique beams only)
#
# === Architecture ===
# Stage 1,2 (TRX-level):
#   BS: 256 TRX × 4 AE/TRX = 1024 AE, 16-beam codebook per TRX (2×2 subarray)
#   UE: 4 TRX × 4 AE/TRX = 16 AE, 16-beam codebook per TRX (2×2 subarray)
#   Cyclic Pattern: pattern length 4 (UE), pattern length 32 (BS)
# Stage 3,4 (Layer-level):
#   BS: 64 layers × 16 AE/layer = 1024 AE, 64-beam codebook per layer (4×4 subarray)
#   UE: Stage 1 결과 고정 (4 TRX)
#   Cyclic Pattern: pattern length up to 8 (Stage 3)
#
# === Codebook ===
# Stage 1,2: 2×2 subarray, 16-beam codebook (F_2,2,2,2)
# Stage 3,4: 4×4 subarray, 64-beam codebook (F_4,2,4,2)
#
# === Link Direction ===
# UPLINK: UE transmit, BS receive
#
# === Input ===
# P1I: Weichselberger parameters (U_bs, U_ue, Omega, H_mean)
# P1J: C_AE (reference capacity)
#
# === Output ===
# CSV: ue,stage,C_TRX_UE,C_TRX_BS,C_TRX,C_ABM_hist,ue_beams,bs_pattern,bs_beams_abm
#
# ======================================================================

# ===== SECTION 1: Environment Setup =====
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
import argparse

# TensorFlow environment
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
gpu_num = 0
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

# GPU memory setup
gpus = tf.config.list_physical_devices('GPU')
for g in gpus:
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except RuntimeError as e:
        pass

tf.get_logger().setLevel("ERROR")
tf.config.optimizer.set_jit(True)
tf.config.threading.set_inter_op_parallelism_threads(0)
tf.config.threading.set_intra_op_parallelism_threads(0)
tf.random.set_seed(42)
np.random.seed(42)

# ======================================================================
# Code Organization: Dependency-Based Levels
# ======================================================================
# LEVEL 0: Independent Base
#   - P1O_Config: Configuration management
#   - CoordinateTransformer: GCS↔LCS conversion (TF)
#   - DFTCodebook: TRX-level DFT (2×2 subarray)
#   - BeamformingMatrix: TRX-level block-diagonal
#   - BeamDomainTransform: Beam domain Weichselberger transform
#
# LEVEL 1: Config-Dependent
#   - DataLoader: P1I/P1J data loading
#   - BeamDomainCapacity: Wen2011 capacity calculation
#   - P1O_ResultManager: CSV result management
#
# LEVEL 2: Beam Selection
#   - TRXBeamSelector: Stage 1/2/3 beam selection (@tf.function, cyclic pattern)
#   - BSLayerMinimalGreedyAdd: Stage 4 minimal greedy layer addition
#
# LEVEL 3: Main Execution
#   - main: Pipeline orchestration
# ======================================================================

# ===== LEVEL 0: Independent Base Classes =====

# ----- P1O_Config -----
class P1O_Config:
    """P1O v4 Uplink beam management configuration
    
    Stages:
    - Stage 1: UE TRX selection (pattern length 4, TRX-level)
    - Stage 2: BS TRX cyclic pattern (pattern length 32, TRX-level)
    - Stage 3: BS Layer cyclic pattern (pattern length up to 8, Layer-level)
    - Stage 4: BS Layer minimal greedy add (Layer-level)
    """
    
    def __init__(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Input data paths
        self.P1I_INPUT_DIR = os.path.join(script_dir, "P1I_Weichsel_Chunk_Results")
        self.P1J_INPUT_DIR = os.path.join(script_dir, "P1J_Capacity_Results")
        
        # Output results path
        self.P1O_OUTPUT_DIR = os.path.join(script_dir, "P1O_Project")
        os.makedirs(self.P1O_OUTPUT_DIR, exist_ok=True)
        
        # Filtering settings
        self.target_areas = [1]
        self.target_freqs = [7.5]
        
        # Hardware parameters
        self.n_bs_trx = 256  # BS TRX count
        self.n_ue_trx = 4    # UE TRX count
        self.n_ae_per_trx = 4  # AE per TRX
        
        # Auto-calculated
        self.n_bs = self.n_bs_trx * self.n_ae_per_trx  # 1024 AE
        self.n_ue = self.n_ue_trx * self.n_ae_per_trx  # 16 AE
        self.n_bs_layers = 64  # BS layers (64 layers × 16 AE/layer)
        self.n_ue_layers = 4   # UE layers (= n_ue_trx, 1 TRX/layer)
        
        # Stage 1,2: TRX-level DFT codebook (2×2 subarray)
        self.trx_ant_per_dim = 2  # 2×2 subarray per TRX
        self.trx_oversample = 2   # 2× oversample → 16 beams
        self.n_cb_per_trx = 16    # Beams per TRX
        
        # Stage 3: Layer-level DFT codebook (4×4 subarray)
        self.layer_ant_per_dim = 4  # 4×4 subarray per layer
        self.layer_oversample = 2   # 2× oversample → 64 beams
        self.n_cb_per_layer = 64    # Beams per layer (F_4,2,4,2)
        
        # Stage 1,2: TRX-level cyclic pattern
        self.n_ue_pattern_beams = 4   # UE pattern length (= n_ue_trx)
        self.n_bs_pattern_beams = 32  # BS pattern length: 1/8 of 256 TRX
        
        # Stage 3: Layer-level cyclic pattern
        self.n_layer_pattern_max = 8  # Max pattern length: 1/8 of 64 layers
        
        # Stage 4: Minimal greedy layer addition
        self.enable_minimal_greedy_add = True   # Core feature: minimal layers with performance guarantee
        self.alpha_target_stage4 = 0.95  # Performance target: C_ABM / C_Layer >= 90%
        
        # Antenna orientations (from P1N)
        self.BS_orientation = {
            "azimuth_deg": 246,
            "downtilt_deg": 3,
            "roll_deg": 0
        }
        self.UE_orientation = {
            "azimuth_deg": 0,
            "elevation_deg": 0,
            "roll_deg": 0
        }
        
        # Capacity calculation settings (Wen2011)
        # Outer loop (P optimization)
        self.outer_max_iter = 50
        self.outer_eps = 0.05
        
        # Inner loop (gamma, psi fixed-point)
        self.inner_max_iter = 50
        
        # Regularization
        self.regularization = 1e-20
        
        # SNR settings
        self.SNR_dB = 10.0
        self.SNR_linear = 10.0**(self.SNR_dB / 10.0)
        
        # Resume settings
        self.resume_from_existing = False
        
        # Test mode settings
        self.test_mode = False
        self.test_ue_indices = {
            1: [1, 216, 438],
            2: [1, 12, 16],
            3: [1, 17, 96],
            4: [1, 32, 186],
            5: [1, 26, 66],
            6: [1, 31, 271]
        }
        
        # Visualization settings
        self.enable_polar_plots = False
        self.polar_output_dir = os.path.join(self.P1O_OUTPUT_DIR, "Polar_Plots")
        os.makedirs(self.polar_output_dir, exist_ok=True)
        
        # P1A input path (for PADP)
        self.P1A_INPUT_DIR = os.path.join(script_dir, "P1A_RT_Results")
        
        # Detect UE list
        self.detect_ue_list()
    
    def detect_ue_list(self):
        """Scan target UEs (P1I & P1J intersection)"""
        
        # P1I chunk scan
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
        
        # P1J result scan
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
        
        # Intersection
        p1i_keys = {(a, f, u) for a, f, u in p1i_ues}
        self.ue_list = []
        
        for area, freq, ue, p1j_path in sorted(p1j_ues):
            if (area, freq, ue) in p1i_keys:
                self.ue_list.append((area, freq, ue, p1j_path))
        
        if self.ue_list:
            print(f"Target UEs: {len(self.ue_list)}")
            area_groups = {}
            for area, freq, ue, _ in self.ue_list:
                key = f"Area{area}_{freq}GHz"
                area_groups[key] = area_groups.get(key, 0) + 1
            for area_freq, count in sorted(area_groups.items()):
                print(f"  - {area_freq}: {count} UEs")
        else:
            print("Warning: No target UEs found")

# ----- CoordinateTransformer -----
class CoordinateTransformer:
    """GCS↔LCS coordinate transformation (TensorFlow, @tf.function)"""
    
    @staticmethod
    @tf.function
    def rotation_matrix_tf(alpha_rad, beta_rad, gamma_rad):
        """3GPP TS 38.901 rotation matrix
        
        Args:
            alpha_rad: Azimuth [rad]
            beta_rad: Downtilt [rad]
            gamma_rad: Roll [rad]
        
        Returns:
            R: [3, 3] rotation matrix
        """
        ca, sa = tf.cos(alpha_rad), tf.sin(alpha_rad)
        cb, sb = tf.cos(beta_rad), tf.sin(beta_rad)
        cc, sc = tf.cos(gamma_rad), tf.sin(gamma_rad)
        
        R = tf.stack([
            tf.stack([ca*cb, ca*sb*sc - sa*cc, ca*sb*cc + sa*sc]),
            tf.stack([sa*cb, sa*sb*sc + ca*cc, sa*sb*cc - ca*sc]),
            tf.stack([-sb,   cb*sc,            cb*cc])
        ])
        
        return R
    
    @staticmethod
    @tf.function
    def gcs_to_lcs_tf(theta_gcs_rad, phi_gcs_rad, alpha_rad, beta_rad, gamma_rad):
        """GCS → LCS conversion
        
        Args:
            theta_gcs_rad: Zenith angle in GCS [rad]
            phi_gcs_rad: Azimuth angle in GCS [rad]
            alpha_rad, beta_rad, gamma_rad: Orientation [rad]
        
        Returns:
            (theta_lcs_rad, phi_lcs_rad)
        """
        # Unit sphere vector in GCS
        st, ct = tf.sin(theta_gcs_rad), tf.cos(theta_gcs_rad)
        sp, cp = tf.sin(phi_gcs_rad), tf.cos(phi_gcs_rad)
        rho_gcs = tf.stack([st * cp, st * sp, ct])
        
        # Rotate to LCS
        R = CoordinateTransformer.rotation_matrix_tf(alpha_rad, beta_rad, gamma_rad)
        R_inv = tf.transpose(R)
        rho_lcs = tf.linalg.matvec(R_inv, rho_gcs)
        
        # Normalize
        rho_lcs_norm = tf.norm(rho_lcs)
        rho_lcs = rho_lcs / (rho_lcs_norm + 1e-10)
        
        # Convert back to spherical
        theta_lcs_rad = tf.acos(tf.clip_by_value(rho_lcs[2], -1.0, 1.0))
        phi_lcs_rad = tf.atan2(rho_lcs[1], rho_lcs[0])
        
        return theta_lcs_rad, phi_lcs_rad
    
    @staticmethod
    @tf.function
    def beam_index_to_angle_lcs_tf(j, N, K, side):
        """Beam index → LCS angles
        
        Args:
            j: Beam index [0, (NK)²-1]
            N: Antennas per dimension
            K: Oversample factor
            side: 0=tx (BS), 1=rx (UE) for downlink
        
        Returns:
            (theta_rad, phi_rad)
        """
        G = N * K
        
        # Decompose j → (iy, iz): iz-major order
        iz = j // G
        iy = j % G
        
        # Centered q indices
        q_y = iy - G // 2
        q_z = iz - G // 2
        
        # α from q (side-dependent)
        qy_overG = tf.cast(q_y, tf.float32) / tf.cast(G, tf.float32)
        qz_overG = tf.cast(q_z, tf.float32) / tf.cast(G, tf.float32)
        
        ay = tf.where(side == 0, -qy_overG, qy_overG)
        az = tf.where(side == 0, -qz_overG, qz_overG)
        
        # Physical parameters (2×2 subarray)
        sign_y, sign_z = 1.0, -1.0
        dy_lam, dz_lam = 0.5, 0.5
        
        # Inverse transform to angles
        cz = az / (sign_z * dz_lam)
        cz = tf.clip_by_value(cz, -1.0, 1.0)
        theta = tf.acos(cz)
        
        sin_theta = tf.sin(theta)
        sin_phi = tf.where(
            sin_theta > 1e-7,
            ay / (sign_y * dy_lam * sin_theta),
            0.0
        )
        sin_phi = tf.clip_by_value(sin_phi, -1.0, 1.0)
        phi = tf.asin(sin_phi)
        
        return theta, phi
    
    @staticmethod
    @tf.function
    def angular_distance_lcs_tf(theta1_rad, phi1_rad, theta2_rad, phi2_rad):
        """Great circle distance
        
        Returns:
            Angular distance [degrees]
        """
        cos_dist = (tf.sin(theta1_rad) * tf.sin(theta2_rad) * tf.cos(phi1_rad - phi2_rad) +
                    tf.cos(theta1_rad) * tf.cos(theta2_rad))
        cos_dist = tf.clip_by_value(cos_dist, -1.0, 1.0)
        
        dist_rad = tf.acos(cos_dist)
        return dist_rad * 180.0 / np.pi

# ----- DFTCodebook -----
class DFTCodebook:
    """TRX-level DFT codebook (2×2 subarray, 16 beams)"""
    
    @staticmethod
    @tf.function
    def generate_1d_dft_tf(N: int, K: int) -> tf.Tensor:
        """1D DFT codebook
        
        [F_{N,K}]_{i,j} = (1/√N) exp(-j 2π ij / (NK))
        
        Args:
            N: Antennas per dimension
            K: Oversample factor
        
        Returns:
            F: [N, NK] complex64
        """
        i = tf.cast(tf.range(N)[:, None], tf.float32)
        j = tf.cast(tf.range(N * K)[None, :], tf.float32)
        pi = tf.constant(np.pi, dtype=tf.float32)
        phase = -2.0 * pi * i * j / float(N * K)
        sqrt_N = tf.sqrt(tf.constant(float(N), dtype=tf.float32))
        sqrt_N_complex = tf.cast(sqrt_N, tf.complex64)
        return tf.exp(tf.complex(0.0, phase)) / sqrt_N_complex
    
    @staticmethod
    @tf.function
    def generate_2d_dft_codebook_tf(ant_per_dim: int, oversample: int) -> tf.Tensor:
        """2D DFT codebook
        
        F = F_{N,K} ⊗ F_{N,K}
        
        Args:
            ant_per_dim: 1D antennas (N)
            oversample: Oversample factor (K)
        
        Returns:
            F: [ant_per_dim², (ant_per_dim*oversample)²] complex64
        
        Example:
            TRX: (2, 2) → [4, 16]
        """
        F_1d = DFTCodebook.generate_1d_dft_tf(ant_per_dim, oversample)
        
        # Kronecker product
        n_ant = ant_per_dim
        n_beams = ant_per_dim * oversample
        
        F_kron = tf.einsum('ij,kl->ikjl', F_1d, F_1d)
        F_kron = tf.reshape(F_kron, [n_ant**2, n_beams**2])
        
        return F_kron

# ----- BeamformingMatrix -----
class BeamformingMatrix:
    """TRX-level block-diagonal beamforming"""
    
    @staticmethod
    @tf.function
    def construct_beamforming_blockdiag_tf(F: tf.Tensor, 
                                          beam_indices: tf.Tensor,  # Permutation with repetition
                                          n_total_trx: int) -> tf.Tensor:
        """TRX-level block-diagonal beamforming matrix
        
        W = blkdiag(w_1, w_2, ..., w_L)
        
        중복 순열(Permutation with Repetition) - 핵심 개념:
        - beam_indices에 중복 허용 (같은 빔을 여러 TRX에서 재사용)
        - 각 TRX가 독립적으로 빔 선택
        - 예: beam_indices = [8, 8, 10]
          → TRX 0: 빔 8
          → TRX 1: 빔 8 (재사용)
          → TRX 2: 빔 10
        
        Args:
            F: [n_ae_per_trx, n_cb_per_trx] DFT codebook (complex64)
            beam_indices: [n_selected] selected beam indices (중복 순열)
            n_total_trx: Total TRX count (int)
        
        Returns:
            W: [n_total_ae, n_selected] complex64
            
        Example:
            BS: F=[4,16], beam_indices=[0,1,5], n_total_trx=256
                → W=[1024, 3]
            UE: F=[4,16], beam_indices=[3,7,11,15], n_total_trx=4
                → W=[16, 4]
        """
        n_ae_per_trx = tf.shape(F)[0]
        n_total_ae = n_ae_per_trx * n_total_trx
        n_selected = tf.shape(beam_indices)[0]
        
        # Initialize
        W = tf.zeros([n_total_ae, n_selected], dtype=tf.complex64)
        
        # Sparse or dense update
        is_sparse = n_selected < n_total_trx
        
        def sparse_update():
            """Sparse activation (selected TRX only)
            
            P1O TRX 단위: TRX 0, 1, 2, ...에 순차적으로 빔 할당
            - beam_indices = [8, 8, 8] → TRX 0, 1, 2에 빔 8, 8, 8 할당
            - TRX 인덱스는 순차적 (0, 1, 2, ...), 빔 인덱스는 중복 순열
            """
            i_grid = tf.range(n_selected, dtype=tf.int32)[:, None]  # TRX indices: [0, 1, 2, ...]
            j_grid = tf.range(n_ae_per_trx, dtype=tf.int32)[None, :]
            
            # TRX 인덱스는 순차적 (i_grid 자체가 TRX 인덱스)
            trx_indices = i_grid  # [0, 1, 2, ...] (not beam_indices!)
            row_starts = trx_indices * n_ae_per_trx
            row_indices = row_starts + j_grid
            col_indices = tf.tile(i_grid, [1, n_ae_per_trx])
            
            indices = tf.stack([
                tf.reshape(row_indices, [-1]),
                tf.reshape(col_indices, [-1])
            ], axis=1)
            
            # 빔 벡터는 beam_indices로부터 가져옴 (중복 허용)
            beam_vecs = tf.gather(F, beam_indices, axis=1)
            updates = tf.reshape(tf.transpose(beam_vecs), [-1])
            
            return tf.tensor_scatter_nd_update(W, indices, updates)
        
        def dense_update():
            """Dense activation (all TRX)"""
            i_grid = tf.range(n_selected, dtype=tf.int32)[:, None]
            j_grid = tf.range(n_ae_per_trx, dtype=tf.int32)[None, :]
            
            row_starts = i_grid * n_ae_per_trx
            row_indices = row_starts + j_grid
            col_indices = tf.tile(i_grid, [1, n_ae_per_trx])
            
            indices = tf.stack([
                tf.reshape(row_indices, [-1]),
                tf.reshape(col_indices, [-1])
            ], axis=1)
            
            beam_vecs = tf.gather(F, beam_indices, axis=1)
            updates = tf.reshape(tf.transpose(beam_vecs), [-1])
            
            return tf.tensor_scatter_nd_update(W, indices, updates)
        
        W = tf.cond(is_sparse, sparse_update, dense_update)
        
        return W
    
    @staticmethod
    @tf.function
    def construct_beamforming_blockdiag_tf_batch(F: tf.Tensor,
                                                 beam_indices_batch: tf.Tensor,
                                                 n_total_trx: int) -> tf.Tensor:
        """Batch beamforming matrices
        
        Args:
            F: [n_ae_per_trx, n_cb_per_trx]
            beam_indices_batch: [batch_size, n_selected] (중복 순열)
            n_total_trx: Total TRX count
        
        Returns:
            W_batch: [batch_size, n_total_ae, n_selected]
        """
        batch_size = tf.shape(beam_indices_batch)[0]
        n_selected = tf.shape(beam_indices_batch)[1]
        n_ae_per_trx = tf.shape(F)[0]
        n_total_ae = n_ae_per_trx * n_total_trx
        
        W_batch = tf.zeros([batch_size, n_total_ae, n_selected], dtype=tf.complex64)
        
        # Sparse update (BS case)
        batch_idx = tf.range(batch_size, dtype=tf.int32)[:, None, None]
        ae_idx = tf.range(n_ae_per_trx, dtype=tf.int32)[None, None, :]
        
        trx_indices = tf.expand_dims(beam_indices_batch, axis=2)
        row_starts = trx_indices * n_ae_per_trx
        row_indices = row_starts + tf.cast(ae_idx, tf.int32)
        col_indices = tf.tile(trx_indices, [1, 1, n_ae_per_trx])
        batch_indices = tf.tile(batch_idx, [1, n_selected, n_ae_per_trx])
        
        indices = tf.stack([
            tf.reshape(batch_indices, [-1]),
            tf.reshape(row_indices, [-1]),
            tf.reshape(col_indices, [-1])
        ], axis=1)
        
        def gather_beams(indices):
            return tf.gather(F, indices, axis=1)
        
        beam_vecs_batch = tf.map_fn(gather_beams, beam_indices_batch, dtype=tf.complex64)
        beam_vecs_transposed = tf.transpose(beam_vecs_batch, [0, 2, 1])
        updates = tf.reshape(beam_vecs_transposed, [-1])
        
        W_batch = tf.tensor_scatter_nd_update(W_batch, indices, updates)
        
        return W_batch

# ----- BeamDomainTransform -----
class BeamDomainTransform:
    """Beam domain transformation for Weichselberger channel model (UPLINK)
    
    === Convention ===
    UPLINK (P1O v3):
    - Omega_ul: [n_bs, n_ue] (transposed from P1I)
    - H_mean_ul: [n_bs, n_ue]
    - Channel: H_UL = U_BS @ (Omega_ul^{1/2} ⊙ H_iid^H) @ U_UE^H
    - TX side: UE (transmit)
    - RX side: BS (receive)
    
    === Beam Domain Transform (UPLINK) ===
    H_mean_beam = W_BS^H @ H_mean_ul @ W_UE: [B_bs, B_ue]
    d_BS = Omega_ul @ v_UE: [n_bs] where v_UE = Σ_k |T_UE(k,:)|^2
    d_UE = Omega_ul^T @ v_BS: [n_ue] where v_BS = Σ_k |T_BS(k,:)|^2
    Omega_beam_ul = |V_BS|^2^T @ Omega_ul @ |V_UE|^2: [B_bs, B_ue]
    """
    
    @staticmethod
    @tf.function
    def transform_mean_channel(H_mean: tf.Tensor, W_bs: tf.Tensor, 
                               W_ue: tf.Tensor) -> tf.Tensor:
        """Mean channel transform for UPLINK
        
        Formula:
            H_mean_beam = W_BS^H @ H_mean_ul @ W_UE
        
        Args:
            H_mean: [n_bs, n_ue] complex64 UPLINK
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
        """BS covariance transform for UPLINK
        
        Formula:
            d_BS(i) = Σ_j Omega_ul(i,j) · |T_UE(j,·)|^2
                    = Omega_ul @ v_UE
        
        Args:
            U_bs: [n_bs, n_bs] complex64
            U_ue: [n_ue, n_ue] complex64
            Omega: [n_bs, n_ue] float32 UPLINK
            W_bs: [n_bs, B_bs] complex64
            W_ue: [n_ue, B_ue] complex64
        
        Returns:
            R_bs_beam: [B_bs, B_bs] complex64
            U_bs_beam: [B_bs, B_bs] complex64
            Lambda_bs_beam: [B_bs] float32
        """
        # Step 1: Compute v_UE = Σ_k |T_UE(k,:)|^2
        T_ue = tf.linalg.adjoint(U_ue) @ W_ue  # [n_ue, B_ue]
        v_ue = tf.reduce_sum(tf.abs(T_ue)**2, axis=1)  # [n_ue]
        
        # Step 2: d_BS = Omega_ul @ v_UE (UL: Omega_ul is [n_bs, n_ue])
        d_bs = tf.linalg.matvec(Omega, v_ue)  # [n_bs]
        
        # Step 3: Project to beam domain
        U_bs_W_bs = tf.linalg.adjoint(U_bs) @ W_bs  # [n_bs, B_bs]
        d_bs_complex = tf.cast(d_bs, tf.complex64)
        R_bs_beam = tf.linalg.adjoint(U_bs_W_bs) @ tf.linalg.diag(d_bs_complex) @ U_bs_W_bs
        
        # Step 4: Eigenvalue decomposition
        Lambda_bs_beam, U_bs_beam = tf.linalg.eigh(R_bs_beam)
        Lambda_bs_beam = tf.math.real(Lambda_bs_beam)
        Lambda_bs_beam = tf.maximum(Lambda_bs_beam, 0.0)
        
        return R_bs_beam, U_bs_beam, Lambda_bs_beam
    
    @staticmethod
    @tf.function
    def transform_ue_covariance(U_bs: tf.Tensor, U_ue: tf.Tensor,
                                Omega: tf.Tensor, W_bs: tf.Tensor,
                                W_ue: tf.Tensor) -> tuple:
        """UE covariance transform for UPLINK
        
        Formula:
            d_UE(j) = Σ_i Omega_ul(i,j) · |T_BS(i,·)|^2
                    = Omega_ul^T @ v_BS
        
        Args:
            U_bs: [n_bs, n_bs] complex64
            U_ue: [n_ue, n_ue] complex64
            Omega: [n_bs, n_ue] float32 UPLINK
            W_bs: [n_bs, B_bs] complex64
            W_ue: [n_ue, B_ue] complex64
        
        Returns:
            R_ue_beam: [B_ue, B_ue] complex64
            U_ue_beam: [B_ue, B_ue] complex64
            Lambda_ue_beam: [B_ue] float32
        """
        # Step 1: Compute v_BS = Σ_k |T_BS(k,:)|^2
        T_bs = tf.linalg.adjoint(U_bs) @ W_bs  # [n_bs, B_bs]
        v_bs = tf.reduce_sum(tf.abs(T_bs)**2, axis=1)  # [n_bs]
        
        # Step 2: d_UE = Omega_ul^T @ v_BS (UL: Omega_ul is [n_bs, n_ue])
        d_ue = tf.linalg.matvec(Omega, v_bs, transpose_a=True)  # [n_ue]
        
        # Step 3: Project to beam domain
        U_ue_W_ue = tf.linalg.adjoint(U_ue) @ W_ue  # [n_ue, B_ue]
        d_ue_complex = tf.cast(d_ue, tf.complex64)
        R_ue_beam = tf.linalg.adjoint(U_ue_W_ue) @ tf.linalg.diag(d_ue_complex) @ U_ue_W_ue
        
        # Step 4: Eigenvalue decomposition
        Lambda_ue_beam, U_ue_beam = tf.linalg.eigh(R_ue_beam)
        Lambda_ue_beam = tf.math.real(Lambda_ue_beam)
        Lambda_ue_beam = tf.maximum(Lambda_ue_beam, 0.0)
        
        return R_ue_beam, U_ue_beam, Lambda_ue_beam
    
    @staticmethod
    @tf.function
    def transform_coupling_matrix(U_bs: tf.Tensor, U_ue: tf.Tensor,
                                  Omega_ul: tf.Tensor, W_bs: tf.Tensor,
                                  W_ue: tf.Tensor, U_bs_beam: tf.Tensor,
                                  U_ue_beam: tf.Tensor) -> tf.Tensor:
        """Coupling matrix transform for UPLINK
        
        Formula:
            Omega_beam_ul = |V_BS|^2^T @ Omega_ul @ |V_UE|^2  → [B_bs, B_ue]
        
        Args:
            U_bs: [n_bs, n_bs] complex64
            U_ue: [n_ue, n_ue] complex64
            Omega_ul: [n_bs, n_ue] float32 UPLINK
            W_bs: [n_bs, B_bs] complex64
            W_ue: [n_ue, B_ue] complex64
            U_bs_beam: [B_bs, B_bs] complex64
            U_ue_beam: [B_ue, B_ue] complex64
            
        Returns:
            Omega_beam_ul: [B_bs, B_ue] float32 UPLINK
        """
        # Step 1: Compute V matrices
        V_bs = tf.linalg.adjoint(U_bs) @ W_bs @ U_bs_beam  # [n_bs, B_bs]
        V_ue = tf.linalg.adjoint(U_ue) @ W_ue @ U_ue_beam  # [n_ue, B_ue]
        
        # Step 2: Element-wise square magnitude
        V_bs_abs2 = tf.abs(V_bs)**2  # [n_bs, B_bs]
        V_ue_abs2 = tf.abs(V_ue)**2  # [n_ue, B_ue]
        
        # Step 3: UPLINK formula
        # Omega_beam_ul = |V_BS|^2^T @ Omega_ul @ |V_UE|^2
        # [B_bs, n_bs] @ [n_bs, n_ue] @ [n_ue, B_ue] = [B_bs, B_ue]
        return tf.transpose(V_bs_abs2) @ Omega_ul @ V_ue_abs2

# ===== LEVEL 1: Config-Dependent Classes =====

# NOTE: DataLoader and BeamDomainCapacity copied from P1L (work for TRX-level too)

# ----- DataLoader -----
class DataLoader:
    """P1I/P1J data loading"""
    
    def __init__(self, config: P1O_Config):
        self.config = config
        self.p1i_cache = {}
    
    def load_p1j_result(self, filepath: str) -> dict:
        """Load P1J result (C_AE)
        
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
        """Load P1I channel parameters and convert to UPLINK
        
        P1I Convention:
            - Omega_dl: [n_ue, n_bs] (downlink, stored in P1I)
            - H_mean_dl: [n_ue, n_bs] (downlink, stored in P1I)
        
        UPLINK Conversion:
            - Omega_ul = Omega_dl^T: [n_bs, n_ue]
            - H_mean_ul = H_mean_dl^T: [n_bs, n_ue]
        
        Returns:
            {
                'U_bs': tf.Tensor [n_bs, n_bs] complex64,
                'U_ue': tf.Tensor [n_ue, n_ue] complex64,
                'Omega_ul': tf.Tensor [n_bs, n_ue] float32 (UPLINK),
                'H_mean_ul': tf.Tensor [n_bs, n_ue] complex64 (UPLINK),
                'kappa': float,
                'channel_model': str
            }
        """
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
            raise FileNotFoundError(f"P1I chunk not found: Area{area}_{freq}GHz UE{ue}")
        
        if chunk_filepath not in self.p1i_cache:
            data = np.load(chunk_filepath, allow_pickle=True)
            self.p1i_cache[chunk_filepath] = data
        else:
            data = self.p1i_cache[chunk_filepath]
        
        ue_indices = data['ue_indices'].tolist()
        if ue not in ue_indices:
            raise ValueError(f"UE{ue} not in chunk: {chunk_filepath}")
        idx_in_chunk = ue_indices.index(ue)
        
        U_bs_np = data['P1G_U_BS'][idx_in_chunk]
        U_ue_np = data['P1G_U_UE'][idx_in_chunk]
        Omega_dl_np = data['P1G_Omega'][idx_in_chunk]  # [n_ue, n_bs] DOWNLINK
        H_mean_dl_np = data['P1H_H_mean'][idx_in_chunk]  # [n_ue, n_bs] DOWNLINK
        channel_model = data['enhanced_metadata'][idx_in_chunk]['channel_model']
        
        # Uplink conversion: transpose DL → UL
        Omega_ul_np = Omega_dl_np.T  # [n_bs, n_ue]
        H_mean_ul_np = H_mean_dl_np.T  # [n_bs, n_ue]
        
        # Normalize (Uplink 기준)
        Omega_ul_norm_np, H_mean_ul_norm_np, kappa = self._normalize_channel(
            Omega_ul_np, H_mean_ul_np, U_bs_np.shape[0], U_ue_np.shape[0]
        )
        
        # Convert to TensorFlow tensors
        U_bs_tf = tf.constant(U_bs_np, dtype=tf.complex64)
        U_ue_tf = tf.constant(U_ue_np, dtype=tf.complex64)
        Omega_ul_tf = tf.constant(Omega_ul_norm_np, dtype=tf.float32)  # [n_bs, n_ue] UPLINK
        H_mean_ul_tf = tf.constant(H_mean_ul_norm_np, dtype=tf.complex64)  # [n_bs, n_ue] UPLINK
        
        return {
            'U_bs': U_bs_tf,
            'U_ue': U_ue_tf,
            'Omega_ul': Omega_ul_tf,
            'H_mean_ul': H_mean_ul_tf,
            'kappa': kappa,
            'channel_model': channel_model
        }
    
    def _normalize_channel(self, Omega: np.ndarray, H_mean: np.ndarray,
                          n_bs: int, n_ue: int) -> tuple:
        """Wen2011 channel normalization"""
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

# ----- P1O_PolarPlotter -----
class P1O_PolarPlotter:
    """PADP-Beam comparison polar plots for downlink"""
    
    def __init__(self, config: P1O_Config):
        self.config = config
        self._p1a_cache = None  # npz cache
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        self.plt = plt
    
    def load_p1a_ray_data(self, area: int, freq: float, ue: int) -> dict:
        """Load P1A ray data from npz (verified approach from test_PADP_Cluster_Polar.py)
        
        Args:
            area: Area index
            freq: Frequency (GHz)
            ue: UE index (1-based, matches rx_indices in npz)
        
        Returns:
            {
                'theta_t_deg': np.ndarray [n_rays],  # BS departure (DL TX)
                'phi_t_deg': np.ndarray [n_rays],    # BS departure azimuth
                'theta_r_deg': np.ndarray [n_rays],  # UE arrival (DL RX)
                'phi_r_deg': np.ndarray [n_rays],    # UE arrival azimuth
                'power': np.ndarray [n_rays],        # Ray power
                'source_path_idx': np.ndarray [n_rays]  # Cluster ID
            }
            or None if not found
        """
        try:
            # Load or use cached npz
            if self._p1a_cache is None:
                npz_path = os.path.join(
                    self.config.P1A_INPUT_DIR,
                    f"Area{area}_{freq}GHz_Rays_ALL_RXs.npz"
                )
                self._p1a_cache = np.load(npz_path)
            
            rays = self._p1a_cache
            
            # Filter rays for this UE
            mask = rays['rx_indices'] == ue
            if not np.any(mask):
                print(f"Warning: No rays found for UE {ue} in npz")
                return None
            
            # Extract and flatten ray data
            ray_data = {
                'theta_t_deg': rays['theta_t_deg'][mask].ravel(),
                'phi_t_deg': rays['phi_t_deg'][mask].ravel(),
                'theta_r_deg': rays['theta_r_deg'][mask].ravel(),
                'phi_r_deg': rays['phi_r_deg'][mask].ravel(),
                'power': rays['power'][mask].ravel(),
                'source_path_idx': rays['source_path_idx'][mask].ravel()
            }
            
            return ray_data
            
        except FileNotFoundError as e:
            print(f"Warning: P1A npz file not found: {e}")
            return None
        except KeyError as e:
            print(f"Warning: Missing key in P1A npz: {e}")
            return None
    
    def rotation_matrix_numpy(self, alpha_rad: float, beta_rad: float, 
                             gamma_rad: float) -> np.ndarray:
        """3GPP TS 38.901 rotation matrix
        
        Args:
            alpha_rad: Azimuth [rad]
            beta_rad: Downtilt [rad]
            gamma_rad: Roll [rad]
        
        Returns:
            R: [3, 3] rotation matrix
        """
        ca, sa = np.cos(alpha_rad), np.sin(alpha_rad)
        cb, sb = np.cos(beta_rad), np.sin(beta_rad)
        cc, sc = np.cos(gamma_rad), np.sin(gamma_rad)
        
        R = np.array([
            [ca*cb, ca*sb*sc - sa*cc, ca*sb*cc + sa*sc],
            [sa*cb, sa*sb*sc + ca*cc, sa*sb*cc - ca*sc],
            [-sb,   cb*sc,            cb*cc]
        ])
        return R
    
    def gcs_to_lcs_numpy(self, theta_gcs_rad: float, phi_gcs_rad: float,
                        orientation: dict) -> tuple:
        """GCS → LCS conversion
        
        Args:
            theta_gcs_rad: Zenith angle in GCS [rad]
            phi_gcs_rad: Azimuth angle in GCS [rad]
            orientation: {'azimuth_deg', 'downtilt_deg'/'elevation_deg', 'roll_deg'}
        
        Returns:
            (theta_lcs_rad, phi_lcs_rad)
        """
        if 'downtilt_deg' in orientation:
            alpha = np.radians(orientation['azimuth_deg'])
            beta = np.radians(orientation['downtilt_deg'])
        elif 'elevation_deg' in orientation:
            alpha = np.radians(orientation['azimuth_deg'])
            beta = np.radians(orientation['elevation_deg'])
        else:
            alpha = 0.0
            beta = 0.0
        gamma = np.radians(orientation.get('roll_deg', 0))
        
        # Unit sphere vector in GCS
        st, ct = np.sin(theta_gcs_rad), np.cos(theta_gcs_rad)
        sp, cp = np.sin(phi_gcs_rad), np.cos(phi_gcs_rad)
        rho_gcs = np.array([st * cp, st * sp, ct])
        
        # Rotate to LCS
        R = self.rotation_matrix_numpy(alpha, beta, gamma)
        R_inv = R.T
        rho_lcs = R_inv @ rho_gcs
        
        # Normalize
        rho_lcs_norm = np.linalg.norm(rho_lcs)
        if rho_lcs_norm < 1e-10:
            return (0.0, 0.0)
        
        rho_lcs = rho_lcs / rho_lcs_norm
        
        # Convert back to spherical
        theta_lcs_rad = np.arccos(np.clip(rho_lcs[2], -1.0, 1.0))
        phi_lcs_rad = np.arctan2(rho_lcs[1], rho_lcs[0])
        
        return theta_lcs_rad, phi_lcs_rad
    
    def beam_index_to_angle_lcs_numpy(self, j: int, N: int, K: int, 
                                     side: str) -> tuple:
        """Beam index → LCS angles
        
        Args:
            j: Beam index [0, (NK)²-1]
            N: Antennas per dimension
            K: Oversample factor
            side: 'tx' or 'rx'
                - Downlink (P1O): BS='tx', UE='rx'
                - Uplink (P1L): BS='rx', UE='tx'
        
        Returns:
            (theta_rad, phi_rad): Zenith, azimuth in LCS
        """
        G = N * K
        
        # Decompose j → (iy, iz): iz-major order
        iz = j // G
        iy = j % G
        
        # Centered q indices
        q_y = iy - G // 2
        q_z = iz - G // 2
        
        # α from q (side-dependent)
        qy_overG = q_y / float(G)
        qz_overG = q_z / float(G)
        
        if side == 'tx':
            ay = +qy_overG
            az = +qz_overG
        elif side == 'rx':
            ay = -qy_overG
            az = -qz_overG
        else:
            raise ValueError(f"side must be 'tx' or 'rx', got {side}")
        
        # Physical parameters (2×2 subarray)
        sign_y, sign_z = 1.0, -1.0
        dy_lam, dz_lam = 0.5, 0.5
        
        # Inverse transform to angles
        cz = az / (sign_z * dz_lam)
        cz = np.clip(cz, -1.0, 1.0)
        theta = np.arccos(cz)
        
        sin_theta = np.sin(theta)
        if sin_theta > 1e-7:
            sin_phi = ay / (sign_y * dy_lam * sin_theta)
            sin_phi = np.clip(sin_phi, -1.0, 1.0)
            phi = np.arcsin(sin_phi)
        else:
            phi = 0.0
        
        return theta, phi
    
    def _group_rays_by_cluster_side(self, ray_data: dict, side: str) -> dict:
        """Group rays by source_path_idx for specified side
        
        Args:
            ray_data: dict with ray data from load_p1a_ray_data
            side: 'tx' (BS) or 'rx' (UE)
        
        Returns:
            dict: {cluster_id: {phi_deg, theta_deg, power, n_rays}}
        """
        if 'source_path_idx' not in ray_data:
            return {}
        
        path_indices = np.unique(ray_data['source_path_idx'])
        clusters = {}
        
        # Select angles based on side
        if side == 'tx':
            phi_key = 'phi_t_deg'
            theta_key = 'theta_t_deg'
        else:  # side == 'rx'
            phi_key = 'phi_r_deg'
            theta_key = 'theta_r_deg'
        
        for path_idx in path_indices:
            mask = ray_data['source_path_idx'] == path_idx
            clusters[int(path_idx)] = {
                'phi_deg': ray_data[phi_key][mask],
                'theta_deg': ray_data[theta_key][mask],
                'power': ray_data['power'][mask],
                'n_rays': np.sum(mask)
            }
        
        return clusters
    
    def _compute_cluster_stats(self, cluster: dict, side: str) -> dict:
        """Compute cluster statistics (from test_PADP_Cluster_Polar.py)
        
        Args:
            cluster: dict with {phi_deg, theta_deg, power, n_rays}
            side: 'tx' (BS) or 'rx' (UE)
        
        Returns:
            dict: {
                'power_dB': float,
                'phi_lcs_mean': float (rad),
                'theta_lcs_mean': float (rad),
                'phi_spread': float (deg),
                'theta_spread': float (deg),
                'n_rays': int
            }
        """
        power = cluster['power']
        total_power = np.sum(power)
        
        # Select orientation based on side
        if side == 'tx':
            orientation = self.config.BS_orientation
        else:  # side == 'rx'
            orientation = self.config.UE_orientation
        
        # Transform to LCS
        theta_lcs_list = []
        phi_lcs_list = []
        
        for theta_deg, phi_deg in zip(cluster['theta_deg'], cluster['phi_deg']):
            theta_rad = np.radians(theta_deg)
            phi_rad = np.radians(phi_deg)
            theta_lcs, phi_lcs = self.gcs_to_lcs_numpy(theta_rad, phi_rad, orientation)
            theta_lcs_list.append(theta_lcs)
            phi_lcs_list.append(phi_lcs)
        
        theta_lcs = np.array(theta_lcs_list)
        phi_lcs = np.array(phi_lcs_list)
        
        # Circular mean
        theta_lcs_mean = self._compute_circular_mean(theta_lcs, power)
        phi_lcs_mean = self._compute_circular_mean(phi_lcs, power)
        
        # Angle spreads (min-max)
        theta_spread = self._compute_angle_spread_minmax(np.degrees(theta_lcs), power)
        phi_spread = self._compute_angle_spread_minmax(np.degrees(phi_lcs), power)
        
        return {
            'power_dB': 10 * np.log10(total_power) if total_power > 0 else -np.inf,
            'total_power': total_power,
            'phi_lcs_mean': phi_lcs_mean,
            'theta_lcs_mean': theta_lcs_mean,
            'phi_spread': phi_spread,
            'theta_spread': theta_spread,
            'n_rays': cluster['n_rays']
        }
    
    def _compute_circular_mean(self, angles_rad: np.ndarray, weights: np.ndarray) -> float:
        """Power-weighted circular mean (handles ±π wrapping)
        
        Args:
            angles_rad: [n] angles in radians
            weights: [n] weights (e.g., power)
        
        Returns:
            mean_angle_rad: circular mean in radians [-π, π]
        """
        if len(angles_rad) == 0:
            return 0.0
        
        total_weight = np.sum(weights)
        if total_weight == 0:
            return 0.0
        
        # Convert to unit vectors and compute weighted average
        x = np.sum(weights * np.cos(angles_rad)) / total_weight
        y = np.sum(weights * np.sin(angles_rad)) / total_weight
        
        # Circular mean via atan2
        mean_angle_rad = np.arctan2(y, x)
        
        return mean_angle_rad
    
    def _compute_angle_spread_minmax(self, angles_deg: np.ndarray, powers: np.ndarray) -> float:
        """Compute min-max spread using circular angular distance
        
        Args:
            angles_deg: [n] angles in degrees
            powers: [n] powers (linear)
        
        Returns:
            spread: angular spread in degrees (min-max range)
        """
        if len(angles_deg) == 0:
            return 0.0
        
        if len(angles_deg) < 2:
            return 0.0
        
        angles_rad = np.radians(angles_deg)
        
        # Circular mean
        mean_angle_rad = self._compute_circular_mean(angles_rad, powers)
        
        # Compute circular angular distances from mean
        angular_dists = np.abs(np.degrees(np.arctan2(
            np.sin(angles_rad - mean_angle_rad),
            np.cos(angles_rad - mean_angle_rad)
        )))
        
        # Spread = 2 * max distance from mean
        spread = 2 * np.max(angular_dists)
        
        return max(spread, 0.0)
    
    def plot_ue_polar(self, area: int, freq: float, ue: int, 
                     bs_beams: list, ue_beams: list):
        """Generate PADP cluster polar plot with P1O selected beams (integrated from test_PADP_Cluster_Polar.py)
        
        Args:
            area: Area index
            freq: Frequency [GHz]
            ue: UE index
            bs_beams: BS beam indices (list or np.ndarray)
            ue_beams: UE beam indices (list or np.ndarray)
        """
        ray_data = self.load_p1a_ray_data(area, freq, ue)
        if ray_data is None:
            return
        
        # Convert to list if numpy array
        if isinstance(bs_beams, np.ndarray):
            bs_beams = bs_beams.tolist()
        if isinstance(ue_beams, np.ndarray):
            ue_beams = ue_beams.tolist()
        
        # Group rays by cluster
        bs_clusters = self._group_rays_by_cluster_side(ray_data, 'tx')
        ue_clusters = self._group_rays_by_cluster_side(ray_data, 'rx')
        
        # Compute cluster statistics
        bs_cluster_stats = {}
        for cid, cluster in bs_clusters.items():
            bs_cluster_stats[cid] = self._compute_cluster_stats(cluster, 'tx')
        
        ue_cluster_stats = {}
        for cid, cluster in ue_clusters.items():
            ue_cluster_stats[cid] = self._compute_cluster_stats(cluster, 'rx')
        
        # Create figure
        fig, (ax_bs, ax_ue) = self.plt.subplots(1, 2, subplot_kw=dict(projection='polar'), 
                                                figsize=(16, 8))
        
        # Plot BS side
        self._plot_cluster_side(ax_bs, bs_cluster_stats, f'UE {ue} - BS TX (LCS)')
        self._overlay_selected_beams(ax_bs, bs_beams, 'tx')
        
        # Plot UE side
        self._plot_cluster_side(ax_ue, ue_cluster_stats, f'UE {ue} - UE RX (LCS)')
        self._overlay_selected_beams(ax_ue, ue_beams, 'rx')
        
        # Overall title
        fig.suptitle(
            f'UE {ue}: PADP Cluster Analysis + P1O Selected Beams',
            fontsize=16,
            y=0.98
        )
        
        self.plt.tight_layout()
        save_path = os.path.join(self.config.polar_output_dir, 
                                f'Area{area}_{freq}GHz_UE{ue}_polar.png')
        self.plt.savefig(save_path, dpi=150, bbox_inches='tight')
        self.plt.close()
        print(f"    Polar plot saved: {os.path.basename(save_path)}")
    
    def _plot_cluster_side(self, ax, cluster_stats: dict, title: str):
        """Plot cluster polar bars (verified from test_PADP_Cluster_Polar.py)
        
        Args:
            ax: matplotlib polar axis
            cluster_stats: dict {cluster_id: stats}
            title: plot title
        """
        # Sort clusters by power
        clusters_sorted = sorted(
            cluster_stats.items(),
            key=lambda x: x[1]['total_power'],
            reverse=True
        )
        
        # Power range for colorbar
        powers_dB = [stats['power_dB'] for _, stats in clusters_sorted 
                     if np.isfinite(stats['power_dB'])]
        
        if len(powers_dB) == 0:
            ax.text(0.5, 0.5, 'No valid clusters',
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title(title, fontsize=12, pad=15)
            return
        
        # Fixed colorbar range for consistency
        vmin = -170
        vmax = -80
        norm = self.plt.matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = self.plt.cm.turbo
        
        # Plot each cluster as polar bar
        for cluster_id, stats in clusters_sorted:
            if not np.isfinite(stats['power_dB']):
                continue
            
            phi_center = stats['phi_lcs_mean']  # radians
            phi_spread = np.radians(stats['phi_spread'])
            
            theta_center = stats['theta_lcs_mean']  # radians
            elevation_center = 90 - np.degrees(theta_center)
            elevation_spread = stats['theta_spread']
            
            # Bottom of the bar (radial position)
            bottom = max(0, elevation_center - elevation_spread / 2)
            
            # Draw polar bar
            ax.bar(
                x=phi_center,
                height=elevation_spread,
                width=phi_spread,
                bottom=bottom,
                color=cmap(norm(stats['power_dB'])),
                alpha=0.8,
                edgecolor='none'
            )
        
        # Colorbar
        sm = self.plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = self.plt.colorbar(sm, ax=ax, pad=0.1, fraction=0.046)
        cbar.set_label('Cluster Power (dB)', fontsize=10)
        
        # Polar plot settings
        ax.set_theta_direction(1)  # clockwise
        ax.set_theta_zero_location('N')
        ax.set_ylim(-60, 105)  # Extended range to cover all codebook beams
        ax.set_facecolor('#f5f5f5')
        ax.set_title(f'{title}\n{len(clusters_sorted)} clusters', fontsize=12, pad=15)
        ax.set_ylabel('Elevation (deg)', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    def _overlay_selected_beams(self, ax, beam_indices: list, side: str):
        """Overlay P1O selected beams on polar plot
        
        Args:
            ax: matplotlib polar axis
            beam_indices: list of beam indices
            side: 'tx' (BS) or 'rx' (UE)
        """
        if len(beam_indices) == 0:
            return
        
        beam_angles = [self.beam_index_to_angle_lcs_numpy(
            int(b), 2, 2, side=side) for b in beam_indices]
        
        marker = 'o' if side == 'tx' else 's'
        color = 'blue' if side == 'tx' else 'green'
        label = 'P1O BS beams' if side == 'tx' else 'P1O UE beams'
        
        for i, (theta_lcs, phi_lcs) in enumerate(beam_angles):
            elevation = 90.0 - np.degrees(theta_lcs)
            ax.plot(phi_lcs, elevation, marker, markersize=12, 
                   markerfacecolor='none', markeredgecolor=color, 
                   markeredgewidth=2.5, zorder=10,
                   label=label if i == 0 else '')
        
        ax.legend(loc='upper right', fontsize=9)
    
# ----- BeamDomainCapacity -----
class BeamDomainCapacity:
    """Beam domain capacity calculation (Wen2011 Algorithm 1, UPLINK)
    
    History:
    - 2025-10-22: Restored from P1L (Uplink)
    - 2025-10-22: Converted to Downlink (v2)
    - 2025-10-24: Converted back to Uplink (v3)
    
    Implementation:
    - Full Wen2011 Algorithm 1 (Outer + Inner loop + Water-filling)
    - UPLINK: P: [B_ue, B_ue] UE transmit covariance
    - Mathematically proven optimal power allocation
    
    UPLINK Convention:
    - P: [B_ue, B_ue] UE transmit covariance
    - T: [B_ue, B_ue] TX side (UE)
    - R: [B_bs, B_bs] RX side (BS)
    - Xi: [B_ue, B_ue] effective channel
    - gamma: [B_bs] BS receive SNR
    - psi: [B_ue] UE transmit power
    
    5 methods:
      1. _find_inverse_water_level_tf: Bisection for water-level (unchanged)
      2. _compute_Xi_for_water_filling: Xi computation (UL)
      3. _fixed_point_loop_tf: Inner loop (UL)
      4. _water_filling_p1j: Water-filling (UL)
      5. wen2011_optimize_tf: Outer loop (UL)
    """
    
    def __init__(self, config: P1O_Config):
        self.config = config
    
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
        # 유효한 고유값만 추출 (Section 5.6 동적 beam 수 지원)
        valid_indices = tf.where(valid_mask)[:, 0]
        valid_eigvals = tf.gather(eigvals, valid_indices)
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
        """γ, ψ로부터 Ξ 계산 (UPLINK)"""
        # T, R 계산
        # (UPLINK) T: TX side (UE), R: RX side (BS)
        # T = U_ue @ diag(Omega^T @ gamma) @ U_ue^H  [B_ue, B_ue]
        Omega_T_gamma = tf.matmul(tf.transpose(Omega), tf.expand_dims(gamma, -1))
        d_ue = tf.reshape(Omega_T_gamma, [-1])
        T = U_ue @ tf.linalg.diag(tf.cast(d_ue, tf.complex64)) @ tf.linalg.adjoint(U_ue)
        
        # R = U_bs @ diag(Omega @ psi) @ U_bs^H  [B_bs, B_bs]
        Omega_psi = tf.matmul(Omega, tf.expand_dims(psi, -1))
        d_bs = tf.reshape(Omega_psi, [-1])
        R = U_bs @ tf.linalg.diag(tf.cast(d_bs, tf.complex64)) @ tf.linalg.adjoint(U_bs)
        
        # Ξ 계산 (UPLINK)
        I_R = tf.eye(n_bs, dtype=tf.complex64) + R + reg_complex * tf.eye(n_bs, dtype=tf.complex64)
        H_mean_H = tf.linalg.adjoint(H_mean)
        Xi = T + H_mean_H @ tf.linalg.solve(I_R, H_mean)
        
        return Xi
    
    @tf.function
    def _fixed_point_loop_tf(self, P: tf.Tensor, U_bs: tf.Tensor, U_ue: tf.Tensor,
                            Omega: tf.Tensor, H_mean: tf.Tensor, max_iter_inner: int,
                            reg_complex: tf.Tensor, n_bs: tf.Tensor, n_ue: tf.Tensor) -> tuple:
        """고정점 반복 (UPLINK, 고정 반복 + argmin)
        
        Args:
            P: [n_ue, n_ue] UE transmit covariance (UPLINK)
            Omega: [n_bs, n_ue] UPLINK
            H_mean: [n_bs, n_ue] UPLINK
            
        Returns:
            (best_gamma, best_psi, Xi_best, Omega_psi_best, best_idx)
                gamma: [n_bs] BS receive SNR
                psi: [n_ue] UE transmit power
                Xi: [n_ue, n_ue] effective channel
                Omega_psi: [n_bs, 1] coupling term
                best_idx: optimal iteration index
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
            
            # T, R 계산
            Omega_T_gamma = tf.matmul(tf.transpose(Omega), tf.expand_dims(gamma, -1))
            d_ue = tf.reshape(Omega_T_gamma, [-1])
            T = U_ue @ tf.linalg.diag(tf.cast(d_ue, tf.complex64)) @ tf.linalg.adjoint(U_ue)
            
            Omega_psi = tf.matmul(Omega, tf.expand_dims(psi, -1))
            d_bs = tf.reshape(Omega_psi, [-1])
            R = U_bs @ tf.linalg.diag(tf.cast(d_bs, tf.complex64)) @ tf.linalg.adjoint(U_bs)
            
            # Xi 계산
            I_R = tf.eye(n_bs, dtype=tf.complex64) + R + reg_complex * tf.eye(n_bs, dtype=tf.complex64)
            H_mean_H = tf.linalg.adjoint(H_mean)
            Xi = T + H_mean_H @ tf.linalg.solve(I_R, H_mean)
            
            # γ 업데이트
            I_R_inv_U_bs = tf.linalg.solve(I_R, U_bs)
            gamma = tf.math.real(
                tf.reduce_sum(tf.math.conj(U_bs) * I_R_inv_U_bs, axis=0)
            )
            
            # ψ 업데이트
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
        
        # 최적 지점의 Xi, Omega_psi 계산
        Omega_T_gamma_best = tf.matmul(tf.transpose(Omega), tf.expand_dims(best_gamma, -1))
        d_ue_best = tf.reshape(Omega_T_gamma_best, [-1])
        T_best = U_ue @ tf.linalg.diag(tf.cast(d_ue_best, tf.complex64)) @ tf.linalg.adjoint(U_ue)
        
        Omega_psi_best = tf.matmul(Omega, tf.expand_dims(best_psi, -1))
        d_bs_best = tf.reshape(Omega_psi_best, [-1])
        R_best = U_bs @ tf.linalg.diag(tf.cast(d_bs_best, tf.complex64)) @ tf.linalg.adjoint(U_bs)
        
        I_R_best = tf.eye(n_bs, dtype=tf.complex64) + R_best
        H_mean_H = tf.linalg.adjoint(H_mean)
        Xi_best = T_best + H_mean_H @ tf.linalg.solve(I_R_best, H_mean)
        
        return best_gamma, best_psi, Xi_best, Omega_psi_best, best_idx
    
    # ========== Level 1: Level 0 의존 (tf.function) ==========
    
    @tf.function
    def _water_filling_p1j(self, Xi: tf.Tensor, power_total: tf.Tensor, n_tx: tf.Tensor) -> tf.Tensor:
        """Water-filling (UPLINK)
        
        Args:
            Xi: [n_ue, n_ue] effective channel (UPLINK)
            power_total: scalar total power
            n_tx: transmit dimension (n_ue for UPLINK)
        
        Returns:
            P: [n_ue, n_ue] UE transmit covariance (UPLINK)
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
        """Wen2011 최적화 (UPLINK) with dynamic convergence
        
        Outer loop + Inner fixed point + Water-filling → P_opt, C_opt 반환
        
        Args:
            U_bs_beam: [B_bs, B_bs] complex64
            U_ue_beam: [B_ue, B_ue] complex64
            Omega_beam: [B_bs, B_ue] float32 (UPLINK)
            H_mean_beam: [B_bs, B_ue] complex64 (UPLINK)
        
        Returns:
            (C_opt, P_opt, k_final): 
                C_opt float32, 
                P_opt [B_ue, B_ue] complex64 (UPLINK),
                k_final int32 (actual iterations)
        """
        n_bs = tf.shape(U_bs_beam)[0]
        n_ue = tf.shape(U_ue_beam)[0]
        # UPLINK: power_budget는 UE TRX 기준 (빔 개수와 무관하게 고정)
        P_UE_TRX = tf.cast(self.config.n_ue_trx, dtype=tf.float32)
        power_budget = P_UE_TRX
        
        max_iter_outer = self.config.outer_max_iter
        max_iter_inner = self.config.inner_max_iter
        eps_outer = tf.constant(self.config.outer_eps, dtype=tf.float32)
        reg = tf.constant(self.config.regularization, dtype=tf.float32)
        reg_complex = tf.cast(reg, tf.complex64)
        
        # 초기 P (UPLINK: [B_ue, B_ue])
        P_init = tf.eye(n_ue, dtype=tf.complex64)
        
        # tf.while_loop body
        def body(k, P_prev, P_current):
            # Step A: 고정점 계산 (Xi 포함)
            gamma, psi, Xi, Omega_psi, k_inner = self._fixed_point_loop_tf(
                P_current, U_bs_beam, U_ue_beam, Omega_beam, H_mean_beam, max_iter_inner, reg_complex, n_bs, n_ue
            )
            
            # Step B: Water-filling
            P_new = self._water_filling_p1j(Xi, power_budget, n_ue)
            
            # P_current → P_prev, P_new → P_current
            return k + 1, P_current, P_new
        
        # tf.while_loop condition
        def cond(k, P_prev, P_current):
            # 최대 반복
            iter_check = k < max_iter_outer
            
            # 최소 2회 실행 보장
            must_continue = tf.less(k, 2)
            
            # k >= 2: 수렴도 체크 (P는 항상 non-zero, regularization 불필요)
            # tf.linalg.norm()은 복소수 입력에 대해 복소수를 반환하므로 float32로 캐스팅 필요
            diff = tf.cast(tf.linalg.norm(P_current - P_prev), tf.float32)
            norm_prev = tf.cast(tf.linalg.norm(P_prev), tf.float32)
            relative_change = diff / norm_prev
            convergence_check = tf.greater(relative_change, eps_outer)
            
            should_continue = tf.logical_or(must_continue, convergence_check)
            return tf.logical_and(iter_check, should_continue)
        
        # Outer loop (JIT compiled, Stage 3 동적 beam 수 지원)
        k_final, _, P_opt = tf.while_loop(
            cond, body,
            [0, P_init, P_init],
            maximum_iterations=max_iter_outer,
            shape_invariants=[
                tf.TensorShape([]),
                tf.TensorShape([None, None]),
                tf.TensorShape([None, None])
            ]
        )
        
        # 최종 P_opt로 capacity 계산 (Xi, Omega_psi 포함)
        gamma_final, psi_final, Xi_final, Omega_psi, _ = self._fixed_point_loop_tf(
            P_opt, U_bs_beam, U_ue_beam, Omega_beam, H_mean_beam, max_iter_inner, reg_complex, n_bs, n_ue
        )
        
        # I(P) 계산 (UPLINK)
        I_Xi_P = tf.eye(n_ue, dtype=tf.complex64) + Xi_final @ P_opt
        
        # R 재계산 (logdet2용)
        d_bs_final = tf.reshape(Omega_psi, [-1])
        R = U_bs_beam @ tf.linalg.diag(tf.cast(d_bs_final, tf.complex64)) @ tf.linalg.adjoint(U_bs_beam)
        I_R = tf.eye(n_bs, dtype=tf.complex64) + R
        
        logdet1 = tf.math.real(tf.linalg.slogdet(I_Xi_P)[1])
        logdet2 = tf.math.real(tf.linalg.slogdet(I_R)[1])
        coupling_term = tf.reduce_sum(gamma_final * tf.squeeze(Omega_psi))
        
        I_nat = logdet1 + logdet2 - coupling_term
        log2 = tf.math.log(2.0)
        C_opt_tf = I_nat / log2
        
        return C_opt_tf, P_opt, k_final
    
# ===== LEVEL 3: Beam Selection =====

# ----- BSLayerMinimalGreedyAdd -----
class BSLayerMinimalGreedyAdd:
    """Stage 4: Minimal Greedy Layer Addition (CORE FEATURE)
    
    Layer 0개에서 시작, 하나씩 추가하여 최소 Layer로 성능 달성 (beam index 중복 순열 허용)
    - 목표: C_ABM / C_Layer >= alpha_target_stage4 (디폴트 90%)
    - 방법: Layer 하나씩 추가, beam index 중복 허용 (64-beam codebook), 매번 Wen2011 재최적화
    - 출력: 최소 Layer 개수 + 최적 전력 + 달성 용량
    - Codebook: 4×4 subarray, 2× oversample → 64-beam codebook per layer (F_4,2,4,2)
    """
    
    def __init__(self, config: P1O_Config):
        self.config = config
        self.capacity_calc = BeamDomainCapacity(config)
        
        # Layer-level DFT codebook (4×4 subarray, 2× oversample)
        self.codebook_bs_layer = DFTCodebook.generate_2d_dft_codebook_tf(
            self.config.layer_ant_per_dim,  # 4
            self.config.layer_oversample     # 2
        )  # [16, 64]
    
    def minimal_greedy_add_bs_layers(self, U_bs, U_ue, Omega, H_mean, W_ue_fixed,
                                      stage3_pattern, init_beams_bs, C_Layer):
        """Stage 4: Minimal greedy layer addition (Uplink, Eager mode)
        
        Args:
            U_bs: [n_bs, n_bs] BS covariance eigenvectors
            U_ue: [n_ue, n_ue] UE covariance eigenvectors
            Omega: [n_bs, n_ue] Coupling matrix (UPLINK)
            H_mean: [n_bs, n_ue] Mean channel matrix (UPLINK)
            W_ue_fixed: [n_ue, n_ue_beams] UE beamforming matrix (Stage 1 결과, 고정)
            stage3_pattern: [k] Stage 3 selected pattern (TF tensor)
            init_beams_bs: [n_init] 초기화 빔 (empty for v5, TF tensor)
            C_Layer: scalar, Stage 3 결과 (기준 용량, TF tensor)
        
        Returns:
            {
                'beams_abm': [N_selected] BS beam sequence (64-beam indices, 중복 순열),
                'C_ABM': scalar 달성 용량,
                'P_ABM': [N_selected, N_selected] 최적 송신 공분산,
                'N_BS_layers': int 활성화된 Layer 개수
            }
        """
        alpha_target = self.config.alpha_target_stage4
        C_threshold = alpha_target * C_Layer
        
        # Extract unique beams from Stage 3 pattern
        unique_beams = tf.unique(stage3_pattern)[0]
        B_bs_candidates = unique_beams
        
        print(f"    Stage 4 candidates: {len(unique_beams.numpy())} unique beams from Stage 3 pattern")
        print(f"    Unique beams: {unique_beams.numpy()}")
        
        # 1. 초기화: 0 layers (start from scratch)
        beams_selected = init_beams_bs.numpy().tolist()
        t = len(beams_selected)
        C_history = []  # Track capacity at each layer addition
        
        print(f"    Initial beams: {t}")
        
        # 2. Greedy Add with Early Stop
        while t < self.config.n_bs_layers:
            best_beam = None
            best_C = -float('inf')
            best_P = None
            
            # 각 후보 빔 테스트 (중복 순열, Stage 3 unique beams only)
            for b in B_bs_candidates.numpy():
                # 빔 추가 (중복 순열)
                beams_selected_rep = beams_selected + [int(b)]
                beams_selected_rep_tf = tf.constant(beams_selected_rep, dtype=tf.int32)
                
                # 빔포밍 행렬 구성 (Layer-level, 64 layers)
                W_bs_temp = BeamformingMatrix.construct_beamforming_blockdiag_tf(
                    self.codebook_bs_layer, beams_selected_rep_tf, self.config.n_bs_layers
                )
                
                # 빔 도메인 변환
                H_mean_beam = BeamDomainTransform.transform_mean_channel(
                    H_mean, W_bs_temp, W_ue_fixed
                )
                _, U_bs_beam, _ = BeamDomainTransform.transform_bs_covariance(
                    U_bs, U_ue, Omega, W_bs_temp, W_ue_fixed
                )
                _, U_ue_beam, _ = BeamDomainTransform.transform_ue_covariance(
                    U_bs, U_ue, Omega, W_bs_temp, W_ue_fixed
                )
                Omega_beam = BeamDomainTransform.transform_coupling_matrix(
                    U_bs, U_ue, Omega, W_bs_temp, W_ue_fixed, U_bs_beam, U_ue_beam
                )
                
                # Wen2011 최적화 (전력 재최적화)
                C_temp, P_temp, _ = self.capacity_calc.wen2011_optimize_tf(
                    U_bs_beam, U_ue_beam, Omega_beam, H_mean_beam
                )
                
                # 최선 빔 선택
                C_temp_scalar = float(C_temp.numpy())
                if C_temp_scalar > best_C:
                    best_C = C_temp_scalar
                    best_beam = int(b)
                    best_P = P_temp
            
            # 최선 빔 추가
            if best_beam is None:
                print(f"    WARNING: No valid beam found to add")
                break
            
            beams_selected.append(best_beam)
            C_history.append(best_C)
            t += 1
            
            print(f"    Added beam {best_beam}, C = {best_C:.2f}, N_layers = {t}")
            
            # Early Stop: 성능 기준 달성
            if best_C >= float(C_threshold.numpy()):
                print(f"    ✓ Performance target achieved: {best_C/float(C_Layer.numpy())*100:.1f}%")
                break
        
        # 3. 최종 결과 (중복 순열 시퀀스, 64-beam indices)
        beams_abm = tf.constant(beams_selected, dtype=tf.int32)
        
        # 최종 빔포밍 행렬 및 용량 계산 (Layer-level, 64 layers)
        W_bs_abm = BeamformingMatrix.construct_beamforming_blockdiag_tf(
            self.codebook_bs_layer, beams_abm, self.config.n_bs_layers
        )
        
        H_mean_beam = BeamDomainTransform.transform_mean_channel(
            H_mean, W_bs_abm, W_ue_fixed
        )
        _, U_bs_beam, _ = BeamDomainTransform.transform_bs_covariance(
            U_bs, U_ue, Omega, W_bs_abm, W_ue_fixed
        )
        _, U_ue_beam, _ = BeamDomainTransform.transform_ue_covariance(
            U_bs, U_ue, Omega, W_bs_abm, W_ue_fixed
        )
        Omega_beam = BeamDomainTransform.transform_coupling_matrix(
            U_bs, U_ue, Omega, W_bs_abm, W_ue_fixed, U_bs_beam, U_ue_beam
        )
        
        C_ABM, P_ABM, _ = self.capacity_calc.wen2011_optimize_tf(
            U_bs_beam, U_ue_beam, Omega_beam, H_mean_beam
        )
        
        return {
            'beams_abm': beams_abm,
            'C_ABM': C_ABM,
            'P_ABM': P_ABM,
            'N_BS_layers': len(beams_selected),
            'C_ABM_hist': C_history
        }

# ===== LEVEL 1: Config-Dependent Classes =====

# ----- P1O_ResultManager -----
class P1O_ResultManager:
    """CSV result management"""
    
    def __init__(self, config: P1O_Config):
        self.config = config
        self._csv_timestamp = None
        self._csv_path = None
    
    def _init_csv(self, area: int, freq: float):
        """Initialize CSV file"""
        if self._csv_path is not None:
            return
        
        if self._csv_timestamp is None:
            utc_plus_9 = datetime.utcnow() + timedelta(hours=9)
            self._csv_timestamp = utc_plus_9.strftime('%m%d_%H%M')
        
        # Add partition suffix if in partition mode
        partition_suffix = f"_p{self.config.partition_id}" if hasattr(self.config, 'partition_id') and self.config.partition_id is not None else ""
        csv_filename = f"A{area}_{freq}GHz_P1O_{self._csv_timestamp}{partition_suffix}.csv"
        self._csv_path = os.path.join(self.config.P1O_OUTPUT_DIR, csv_filename)
    
    def save_stage_result(self, area: int, freq: float, ue: int, stage: str,
                          C_TRX_UE: float = None,
                          C_TRX_BS: float = None,
                          C_TRX: float = None,
                          C_Layer_no_power: float = None,
                          C_Layer: float = None,
                          C_ABM_hist_list: list = None,
                          lambda_bs_beam_str: str = "",
                          lambda_ue_beam_str: str = "",
                          ue_beams_str: str = "",
                          bs_pattern_str: str = "",
                          bs_layer_pattern_str: str = "",
                          bs_beams_abm_str: str = ""):
        """Save stage result with unified CSV format
        
        Args:
            stage: "Stage1", "Stage2", "PowerOpt", "Stage3", "Stage3_PowerOpt", or "Stage4"
            C_TRX_UE: Stage 1 UE capacity (None if not yet computed)
            C_TRX_BS: Stage 2 BS capacity before power opt (None if not yet computed)
            C_TRX: Capacity after power optimization (Stage 2, None if not yet computed)
            C_Layer_no_power: Stage 3 Layer capacity before power opt (None if not yet computed)
            C_Layer: Stage 3 Layer capacity after power opt (None if not yet computed)
            C_ABM_hist_list: List of capacities during Stage 4 (None if not yet computed)
            ue_beams_str: Comma-separated UE beam indices
            bs_pattern_str: Comma-separated BS TRX pattern (pattern length 32)
            bs_layer_pattern_str: Comma-separated BS Layer pattern (pattern length up to 8)
            bs_beams_abm_str: Comma-separated selected BS beams (with repetition)
        """
        if self._csv_path is None:
            self._init_csv(area, freq)
        
        headers = ['ue', 'stage', 'C_TRX_UE', 'C_TRX_BS', 'C_TRX', 'C_Layer_no_power', 'C_Layer', 
                   'C_ABM_hist', 'lambda_bs_beam', 'lambda_ue_beam', 
                   'ue_beams', 'bs_pattern', 'bs_layer_pattern', 'bs_beams_abm']
        
        file_exists = os.path.exists(self._csv_path)
        
        # Format C_ABM_hist as semicolon-separated string
        if C_ABM_hist_list is not None:
            C_ABM_hist_str = ';'.join(f"{c:.2f}" for c in C_ABM_hist_list)
        else:
            C_ABM_hist_str = ""
        
        with open(self._csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(headers)
            
            row = [
                ue,
                stage,
                f"{C_TRX_UE:.2f}" if C_TRX_UE is not None else "",
                f"{C_TRX_BS:.2f}" if C_TRX_BS is not None else "",
                f"{C_TRX:.2f}" if C_TRX is not None else "",
                f"{C_Layer_no_power:.2f}" if C_Layer_no_power is not None else "",
                f"{C_Layer:.2f}" if C_Layer is not None else "",
                C_ABM_hist_str,
                lambda_bs_beam_str,
                lambda_ue_beam_str,
                ue_beams_str,
                bs_pattern_str,
                bs_layer_pattern_str,
                bs_beams_abm_str
            ]
            writer.writerow(row)

# ===== LEVEL 2: Beam Selection Classes =====

# ----- TRXBeamSelector -----
class TRXBeamSelector:
    """TRX-level and Layer-level beam selection with @tf.function optimization
    
    Stages:
    - Stage 1: UE TRX selection (pattern length 4)
    - Stage 2: BS TRX selection (pattern length 32)
    - Stage 3: BS Layer cyclic pattern (pattern length up to 8)
    
    Codebooks:
    - TRX-level: 16-beam codebook per TRX (2×2 subarray)
    - Layer-level: 64-beam codebook per layer (4×4 subarray)
    """
    
    def __init__(self, config: P1O_Config):
        self.config = config
        self.capacity_calc = BeamDomainCapacity(config)
        
        # DFT codebooks (TRX-level: 2×2 subarray → 16-beam codebook)
        self.F_bs = DFTCodebook.generate_2d_dft_codebook_tf(2, 2)  # [4, 16]
        self.F_ue = DFTCodebook.generate_2d_dft_codebook_tf(2, 2)  # [4, 16]
    
    # Level 0: Single capacity computation
    @tf.function
    def _compute_beam_capacity_tf(self, U_bs, U_ue, Omega, H_mean, 
                                   W_bs, W_ue, P):
        """Compute capacity for single beamforming pair (UPLINK)
        
        Args:
            U_bs, U_ue, Omega, H_mean: Weichselberger params (UPLINK)
            W_bs: [n_bs, B_bs] complex64
            W_ue: [n_ue, B_ue] complex64
            P: [B_ue, B_ue] complex64 (power matrix, unused - recomputed in Wen2011)
        
        Returns:
            C: Capacity [bits/s/Hz]
        """
        # Transform to beam domain
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
        
        # Wen2011 optimization with full algorithm
        C_opt, P_opt, k_final = self.capacity_calc.wen2011_optimize_tf(
            U_bs_beam, U_ue_beam, Omega_beam, H_mean_beam
        )
        
        return C_opt, P_opt, k_final
    
    # Level 1: Vectorized capacity computation
    @tf.function
    def _compute_capacity_for_multiple_beamformings_tf(self, U_bs, U_ue, Omega, H_mean,
                                                        W_fixed, beam_indices_batch, 
                                                        n_total_trx, is_tx_side):
        """Vectorized capacity for multiple candidates
        
        Args:
            W_fixed: Fixed beamforming matrix (UE or BS)
            beam_indices_batch: [batch_size, n_selected] candidate beam indices
            n_total_trx: Total TRX count (256 for BS, 4 for UE)
            is_tx_side: True if selecting BS beams, False if selecting UE beams
        
        Returns:
            capacities: [batch_size] capacities
        """
        batch_size = tf.shape(beam_indices_batch)[0]
        
        # Build W matrices
        if is_tx_side:
            # Selecting BS beams, W_fixed is UE
            F = self.F_bs
            W_ue_fixed = W_fixed
            
            def compute_single(i):
                beam_indices = beam_indices_batch[i]
                W_bs = BeamformingMatrix.construct_beamforming_blockdiag_tf(
                    F, beam_indices, n_total_trx
                )
                
                B_bs = tf.shape(W_bs)[1]
                P = tf.eye(B_bs, dtype=tf.complex64)
                
                return self._compute_beam_capacity_tf(
                    U_bs, U_ue, Omega, H_mean, W_bs, W_ue_fixed, P
                )
        # else:            
        else:
            # Selecting UE beams, W_fixed is BS
            F = self.F_ue
            W_bs_fixed = W_fixed
                    
            def compute_single(i):
                beam_indices = beam_indices_batch[i]
                W_ue = BeamformingMatrix.construct_beamforming_blockdiag_tf(
                    F, beam_indices, n_total_trx
                )
                
                # Stage 1 (UE selection): P = I_{B_ue} (UPLINK: UE transmit, beam domain)
                P = tf.eye(tf.shape(W_ue)[1], dtype=tf.complex64)
                
                return self._compute_beam_capacity_tf(
                    U_bs, U_ue, Omega, H_mean, W_bs_fixed, W_ue, P
                )
        
        # Map over batch
        capacities = tf.map_fn(
            compute_single,
            tf.range(batch_size, dtype=tf.int32),
            fn_output_signature=tf.TensorSpec(shape=(), dtype=tf.float32)
        )
        
        return capacities
    
    # Cyclic Pattern Methods (v2)
    @tf.function
    def _expand_pattern_to_full_beams(self, pattern_k, n_total_trx):
        """Cyclically repeat k-length pattern to fill n_total_trx
        
        Args:
            pattern_k: [k] beam indices
            n_total_trx: Target TRX count (4 for UE, 256 for BS)
            
        Returns:
            full_beams: [n_total_trx] cyclically repeated pattern
        """
        k = tf.shape(pattern_k)[0]
        
        # Calculate repetitions needed: ceil(n_total_trx/k)
        n_reps = tf.cast(tf.math.ceil(tf.cast(n_total_trx, tf.float32) / tf.cast(k, tf.float32)), tf.int32)
        
        # Tile and slice to exact length
        full_beams = tf.tile(pattern_k, [n_reps])[:n_total_trx]
        
        return full_beams
    
    def _select_beams_cyclic_pattern(self, U_bs, U_ue, Omega, H_mean,
                                      W_fixed, n_pattern, n_total_trx,
                                      is_tx_side, stage_name):
        """Unified cyclic pattern beam selection (eager mode)
        
        Algorithm:
        - Start with empty pattern []
        - For k=1 to n_pattern (pattern length):
            - Test pattern_prev + [b] for all beam indices b in 0-15
            - Each pattern cyclically repeated across n_total_trx
            - Select beam index b that maximizes capacity
        - Return final pattern of length n_pattern
        
        Args:
            U_bs, U_ue, Omega, H_mean: Weichselberger params
            W_fixed: Fixed beamforming (other side)
            n_pattern: Pattern length (4 for UE, 32 for BS)
            n_total_trx: Total TRX count (4 for UE, 256 for BS)
            is_tx_side: True for BS (TX), False for UE (RX)
            stage_name: "Stage 1" or "Stage 2" for print
        
        Returns:
            (beam_pattern, C_final): 
            - beam_pattern: [n_pattern] beam indices in pattern
            - C_final: Achieved capacity
        """
        n_candidates = self.config.n_cb_per_trx  # 16 (all beams 0-15)
        
        # Select codebook based on side
        F = self.F_bs if is_tx_side else self.F_ue
        
        # Pattern storage (Python list instead of TensorArray)
        pattern = []
        
        print(f"  {stage_name}: Cyclic pattern beam selection (pattern length {n_pattern})")
        
        # Adaptive print interval
        print_interval = 1 if n_pattern <= 4 else 4
        
        # Outer loop: for loop instead of tf.while_loop
        C_final = 0.0
        C_prev = 0.0
        early_stopped = False
        for k in range(n_pattern):
            # Current pattern (Python list → tensor)
            current_pattern = tf.constant(pattern, dtype=tf.int32) if pattern else tf.constant([], dtype=tf.int32)
            
            best_beam = -1
            best_capacity = -1e10
            best_P = None
            best_k_final = 0
            
            # Inner loop: for loop instead of tf.while_loop
            for b in range(n_candidates):
                # Create test pattern
                b_tensor = tf.constant([b], dtype=tf.int32)
                if len(pattern) > 0:
                    test_pattern = tf.concat([current_pattern, b_tensor], axis=0)
                else:
                    test_pattern = b_tensor
                
                # Call JIT-compiled helper (still @tf.function)
                full_beams = self._expand_pattern_to_full_beams(test_pattern, n_total_trx)
                
                # Build beamforming matrix
                W_selected = BeamformingMatrix.construct_beamforming_blockdiag_tf(
                    F, full_beams, n_total_trx
                )
                
                # Compute capacity (side-dependent)
                if is_tx_side:
                    # BS side: W_bs is selected, W_ue is fixed
                    B_bs = W_selected.shape[1]
                    P = tf.eye(B_bs, dtype=tf.complex64)
                    C, P_result, k_final = self._compute_beam_capacity_tf(
                        U_bs, U_ue, Omega, H_mean, W_selected, W_fixed, P
                    )
                else:
                    # UE side: W_ue is selected, W_bs is fixed (identity)
                    P = tf.eye(W_fixed.shape[1], dtype=tf.complex64)
                    C, P_result, k_final = self._compute_beam_capacity_tf(
                        U_bs, U_ue, Omega, H_mean, W_fixed, W_selected, P
                    )
                
                # Update best (Python if/else instead of tf.cond)
                C_val = float(C.numpy())
                if C_val > best_capacity:
                    best_beam = b
                    best_capacity = C_val
                    best_P = P_result
                    best_k_final = k_final
            
            # Add best beam to pattern
            pattern.append(best_beam)
            C_final = best_capacity
            
            # Early stop check: < 1% improvement
            if k > 0:
                improvement_ratio = (C_final - C_prev) / C_prev if C_prev > 0 else 1.0
                if improvement_ratio < 0.01:
                    print(f"     Early stop: improvement {improvement_ratio*100:.2f}% < 1%")
                    pattern.pop()
                    C_final = C_prev
                    early_stopped = True
                    break
            
            C_prev = C_final
            
            # Progress print (Python if/else instead of tf.cond)
            if (k+1) % print_interval == 0 or (k+1) == n_pattern:
                start_idx = max(0, k+1-print_interval)
                recent = pattern[start_idx:k+1]
                tr_P = float(tf.math.real(tf.linalg.trace(best_P)).numpy())
                print(f"     {stage_name} Pattern {k+1}: {recent} | C = {best_capacity:.2f} bps/Hz, tr(P)={tr_P:.1f}, iter_P={int(best_k_final)+1}")
        
        # Early stop summary
        if early_stopped:
            print(f"  {stage_name}: Stopped at {len(pattern)} patterns (early stop)")
        else:
            print(f"  {stage_name}: Completed {len(pattern)} patterns (full search)")
        
        # Return as tensors
        beam_pattern = tf.constant(pattern, dtype=tf.int32)
        C_final_tf = tf.constant(C_final, dtype=tf.float32)
        
        return beam_pattern, C_final_tf
    
    def select_ue_beams_cyclic_pattern(self, U_bs, U_ue, Omega, H_mean):
        """Stage 1: UE beam selection (4-pattern cyclic, eager mode)
        
        Uses config.n_ue_pattern_beams (4) for pattern length
        Uses config.n_ue_trx (4) for total TRX count
        """
        W_bs_identity = tf.eye(self.config.n_bs, dtype=tf.complex64)
        
        return self._select_beams_cyclic_pattern(
            U_bs, U_ue, Omega, H_mean,
            W_bs_identity,
            n_pattern=self.config.n_ue_pattern_beams,   # 4
            n_total_trx=self.config.n_ue_trx,           # 4
            is_tx_side=False,
            stage_name="Stage 1"
        )
    
    def select_bs_beams_cyclic_pattern(self, U_bs, U_ue, Omega, H_mean, W_ue_fixed):
        """Stage 2: BS beam selection (32-pattern cyclic, eager mode)
        
        Uses config.n_bs_pattern_beams (32) for pattern length
        Uses config.n_bs_trx (256) for total TRX count
        """
        return self._select_beams_cyclic_pattern(
            U_bs, U_ue, Omega, H_mean,
            W_ue_fixed,
            n_pattern=self.config.n_bs_pattern_beams,  # 32
            n_total_trx=self.config.n_bs_trx,          # 256
            is_tx_side=True,
            stage_name="Stage 2"
        )
    
    def select_bs_layer_beams_cyclic_pattern(self, U_bs, U_ue, Omega, H_mean, W_ue_fixed):
        """Stage 3: BS Layer best pattern selection (eager mode)
        
        Algorithm:
        - Pattern length k = 1 to 4: always search
        - Pattern length k = 5 to 8: early stop if improvement < 1%
        - Return best overall pattern across all searched lengths
        
        Args:
            U_bs, U_ue, Omega, H_mean: Weichselberger params (UPLINK)
            W_ue_fixed: [n_ue, B_ue] UE beamforming (Stage 1 result, fixed)
        
        Returns:
            (pattern_best, C_best, best_k): 
            - pattern_best: [k_best] best pattern indices (0-63)
            - C_best: Best achieved capacity across all lengths
            - best_k: Optimal pattern length
        """
        # Layer-level 64-beam codebook
        F_layer = DFTCodebook.generate_2d_dft_codebook_tf(
            self.config.layer_ant_per_dim, self.config.layer_oversample
        )
        
        n_candidates = 64  # All beam indices 0-63
        n_total_layers = self.config.n_bs_layers  # 64
        max_pattern_length = self.config.n_layer_pattern_max  # 8
        
        # Track current pattern being built
        current_pattern = []
        C_prev = 0.0
        early_stop_threshold = 0.01  # 1% improvement threshold
        
        # Track best overall pattern
        best_overall_pattern = []
        best_overall_C = -float('inf')
        best_k = 0
        
        print(f"  Stage 3: Layer best pattern search (length 1-{max_pattern_length})")
        
        for k in range(1, max_pattern_length + 1):
            # Greedy search for best beam at length k
            best_beam_k = -1
            best_C_k = -1e10
            
            for b in range(n_candidates):
                # Test pattern: current_pattern + [b]
                test_pattern = current_pattern + [b]
                test_pattern_tf = tf.constant(test_pattern, dtype=tf.int32)
                
                # Expand to 64 layers (cyclic repetition)
                full_beams = self._expand_pattern_to_full_beams(test_pattern_tf, n_total_layers)
                
                # Build W_bs (Layer-level)
                W_bs_test = BeamformingMatrix.construct_beamforming_blockdiag_tf(
                    F_layer, full_beams, n_total_layers
                )
                
                # Compute capacity
                H_mean_beam = BeamDomainTransform.transform_mean_channel(
                    H_mean, W_bs_test, W_ue_fixed
                )
                _, U_bs_beam, _ = BeamDomainTransform.transform_bs_covariance(
                    U_bs, U_ue, Omega, W_bs_test, W_ue_fixed
                )
                _, U_ue_beam, _ = BeamDomainTransform.transform_ue_covariance(
                    U_bs, U_ue, Omega, W_bs_test, W_ue_fixed
                )
                Omega_beam = BeamDomainTransform.transform_coupling_matrix(
                    U_bs, U_ue, Omega, W_bs_test, W_ue_fixed, U_bs_beam, U_ue_beam
                )
                
                C_test, _, _ = self.capacity_calc.wen2011_optimize_tf(
                    U_bs_beam, U_ue_beam, Omega_beam, H_mean_beam
                )
                C_test_val = float(C_test.numpy())
                
                if C_test_val > best_C_k:
                    best_C_k = C_test_val
                    best_beam_k = b
            
            # Update current pattern
            current_pattern.append(best_beam_k)
            
            print(f"    Pattern length {k}: added beam {best_beam_k}, C = {best_C_k:.2f}")
            
            # Update best overall if this length is better
            if best_C_k > best_overall_C:
                best_overall_C = best_C_k
                best_overall_pattern = current_pattern.copy()
                best_k = k
                print(f"      → New best overall (length {best_k})")
            
            # Early stop: only apply from length 5 onwards
            if k >= 5 and k > 1:
                improvement = (best_C_k - C_prev) / (C_prev + 1e-10)
                if improvement < early_stop_threshold:
                    print(f"    Early stop at length {k}: improvement {improvement*100:.2f}% < 1%")
                    break
            
            C_prev = best_C_k
        
        print(f"  Stage 3 complete: best pattern length = {best_k}, C = {best_overall_C:.2f}")
        
        return (tf.constant(best_overall_pattern, dtype=tf.int32), 
                tf.constant(best_overall_C, dtype=tf.float32),
                best_k)

# ===== LEVEL 3: Main Execution =====

def main():
    """P1O pipeline orchestration"""
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='P1O v5: Optimized Stage 3/4 Pattern Search')
    parser.add_argument('--partition', type=int, default=None, help='Partition index for parallel execution')
    parser.add_argument('--ue-indices', type=str, default=None, help='Comma-separated UE indices to process')
    args = parser.parse_args()
    
    print("="*80)
    print("P1O: Cyclic Pattern TRX-Level Beam Management")
    print("="*80)
    
    config = P1O_Config()
    config.partition_id = args.partition  # Store partition ID for CSV naming
    
    # Test mode filtering
    if config.test_mode:
        test_ues_flat = set()
        for beam_cnt, indices in config.test_ue_indices.items():
            area = indices[0]
            ues = indices[1:]
            for u in ues:
                test_ues_flat.add((area, u))
        
        config.ue_list = [
            (a, f, u, p) for a, f, u, p in config.ue_list 
            if (a, u) in test_ues_flat
        ]
        print(f"Test mode: {len(config.ue_list)} UEs selected")
    
    # Partition mode: auto-split UE list or use explicit UE indices
    if args.partition is not None:
        if args.ue_indices is not None:
            # Explicit UE indices mode (backward compatibility)
            ue_indices = [int(u) for u in args.ue_indices.split(',')]
            config.ue_list = [
                (a, f, u, p) for a, f, u, p in config.ue_list 
                if u in ue_indices
            ]
            print(f"Partition {args.partition}: Processing {len(config.ue_list)} UEs: {ue_indices}")
        else:
            # Auto-split mode: divide UE list into partitions
            total_ues = len(config.ue_list)
            n_partitions = 6  # Fixed to 6 partitions
            
            # Calculate partition ranges
            ues_per_partition = total_ues // n_partitions
            remainder = total_ues % n_partitions
            
            # Assign extra UEs to first 'remainder' partitions
            if args.partition < remainder:
                start_idx = args.partition * (ues_per_partition + 1)
                end_idx = start_idx + ues_per_partition + 1
            else:
                start_idx = remainder * (ues_per_partition + 1) + (args.partition - remainder) * ues_per_partition
                end_idx = start_idx + ues_per_partition
            
            # Extract partition UE list
            config.ue_list = config.ue_list[start_idx:end_idx]
            ue_ids = [u for _, _, u, _ in config.ue_list]
            
            print(f"Partition {args.partition}/{n_partitions}: Processing {len(config.ue_list)} UEs")
            print(f"  UE indices: {ue_ids[:5]}{'...' if len(ue_ids) > 5 else ''}")
            print(f"  Range: [{start_idx}:{end_idx}] of {total_ues} total UEs")
    
    if len(config.ue_list) == 0:
        print("No UEs to process. Exiting.")
        return
    
    # Module instantiation
    data_loader = DataLoader(config)
    trx_selector = TRXBeamSelector(config)
    result_mgr = P1O_ResultManager(config)
    
    # Polar plot plotter (if enabled)
    if config.enable_polar_plots:
        polar_plotter = P1O_PolarPlotter(config)
    
    print(f"\nProcessing {len(config.ue_list)} UEs...")
    print()
    
    # Time tracking
    overall_start = time.time()
    ue_times = []
    
    for idx, (area, freq, ue, p1j_path) in enumerate(config.ue_list):
        ue_start = time.time()
        print(f"[{idx+1}/{len(config.ue_list)}] Area{area}_{freq}GHz UE{ue}")
        
        try:
            # No Stage 0 in v2 (PADP filtering removed)
            # UE beam selection: all 16 beams available as candidates
            B_ue_tf = tf.constant(list(range(16)), dtype=tf.int32)
            init_beams_ue_tf = tf.constant([], dtype=tf.int32)
            
            # Load channel params
            print("  Loading channel parameters...")
            p1j_data = data_loader.load_p1j_result(p1j_path)
            channel_data = data_loader.load_p1i_channel_params(area, freq, ue)
            
            # Recalculate C_AE: identity beamforming on both sides (UPLINK)
            print("  Calculating C_AE baseline (identity beamforming)...")
            W_bs_identity = tf.eye(config.n_bs, dtype=tf.complex64)
            W_ue_identity = tf.eye(config.n_ue, dtype=tf.complex64)
            
            H_mean_ae = BeamDomainTransform.transform_mean_channel(
                channel_data['H_mean_ul'], W_bs_identity, W_ue_identity
            )
            _, U_bs_ae, _ = BeamDomainTransform.transform_bs_covariance(
                channel_data['U_bs'], channel_data['U_ue'],
                channel_data['Omega_ul'], W_bs_identity, W_ue_identity
            )
            _, U_ue_ae, _ = BeamDomainTransform.transform_ue_covariance(
                channel_data['U_bs'], channel_data['U_ue'],
                channel_data['Omega_ul'], W_bs_identity, W_ue_identity
            )
            Omega_ae = BeamDomainTransform.transform_coupling_matrix(
                channel_data['U_bs'], channel_data['U_ue'],
                channel_data['Omega_ul'], W_bs_identity, W_ue_identity,
                U_bs_ae, U_ue_ae
            )
            
            C_AE_tf, P_ae, k_final_ae = trx_selector.capacity_calc.wen2011_optimize_tf(
                U_bs_ae, U_ue_ae, Omega_ae, H_mean_ae
            )
            C_AE = float(C_AE_tf.numpy())
            tr_P = float(tf.math.real(tf.linalg.trace(P_ae)).numpy())
            print(f"    C_AE = {C_AE:.2f} bps/Hz (tr(P)={tr_P:.1f}, iter_P={int(k_final_ae.numpy())+1})")
            
            # Initialize variables for progressive CSV logging
            ue_beams_str = ""
            bs_pattern_str = ""
            bs_layer_pattern_str = ""
            
            # ===== Stage 1: UE TRX Selection (UPLINK) =====
            s1_start = time.time()
            print("  Stage 1: UE TRX selection (cyclic pattern)...")
            pattern_4, C_UE = trx_selector.select_ue_beams_cyclic_pattern(
                channel_data['U_bs'], channel_data['U_ue'], 
                channel_data['Omega_ul'], channel_data['H_mean_ul']
            )
            # Expand to 4 TRX (no expansion needed, pattern already 4-length)
            Beams_UE_opt = pattern_4
            
            ue_beams_compact = ','.join(str(int(b)) for b in Beams_UE_opt.numpy())
            print(f"    UE_beams=[{ue_beams_compact}] | C_UE={float(C_UE.numpy()):.2f} bps/Hz | {time.time() - s1_start:.1f}s")
            
            # Save Stage 1 result
            ue_beams_str = ','.join(str(int(b)) for b in Beams_UE_opt.numpy())
            result_mgr.save_stage_result(
                area, freq, ue, "Stage1",
                C_TRX_UE=float(C_UE.numpy()),
                ue_beams_str=ue_beams_str
            )
            
            # ===== Stage 2: BS Cyclic Pattern Selection (UPLINK) =====
            s2_start = time.time()
            W_ue_fixed = BeamformingMatrix.construct_beamforming_blockdiag_tf(
                trx_selector.F_ue, Beams_UE_opt, config.n_ue_trx
            )
            
            pattern_32, C_BS = trx_selector.select_bs_beams_cyclic_pattern(
                channel_data['U_bs'], channel_data['U_ue'],
                channel_data['Omega_ul'], channel_data['H_mean_ul'],
                W_ue_fixed
            )
            
            print(f"    C_BS = {float(C_BS.numpy()):.2f} bps/Hz (before power opt)")
            
            # Expand pattern to full 256 beams (중복 순열: 32 pattern × 8 repetitions)
            Beams_BS_full_rep = trx_selector._expand_pattern_to_full_beams(pattern_32, config.n_bs_trx)
            
            # Save Stage 2 result
            bs_pattern_str = ','.join(str(int(b)) for b in pattern_32.numpy())
            result_mgr.save_stage_result(
                area, freq, ue, "Stage2",
                C_TRX_UE=float(C_UE.numpy()),
                C_TRX_BS=float(C_BS.numpy()),
                ue_beams_str=ue_beams_str,
                bs_pattern_str=bs_pattern_str
            )
            
            # Power optimization (UPLINK)
            print("  Power optimization...")
            W_bs_opt = BeamformingMatrix.construct_beamforming_blockdiag_tf(
                trx_selector.F_bs, Beams_BS_full_rep, config.n_bs_trx
            )
            
            H_mean_beam = BeamDomainTransform.transform_mean_channel(
                channel_data['H_mean_ul'], W_bs_opt, W_ue_fixed
            )
            _, U_bs_beam, Lambda_bs_beam = BeamDomainTransform.transform_bs_covariance(
                channel_data['U_bs'], channel_data['U_ue'],
                channel_data['Omega_ul'], W_bs_opt, W_ue_fixed
            )
            _, U_ue_beam, Lambda_ue_beam = BeamDomainTransform.transform_ue_covariance(
                channel_data['U_bs'], channel_data['U_ue'],
                channel_data['Omega_ul'], W_bs_opt, W_ue_fixed
            )
            Omega_beam = BeamDomainTransform.transform_coupling_matrix(
                channel_data['U_bs'], channel_data['U_ue'],
                channel_data['Omega_ul'], W_bs_opt, W_ue_fixed,
                U_bs_beam, U_ue_beam
            )
            
            C_TRX, P_trx, k_final_trx = trx_selector.capacity_calc.wen2011_optimize_tf(
                U_bs_beam, U_ue_beam, Omega_beam, H_mean_beam
            )
            
            # Lambda analysis (TRX-level, NO regularization)
            lambda_bs_norm = Lambda_bs_beam.numpy() / Lambda_bs_beam.numpy().mean()
            lambda_bs_str = ','.join(f"{v:.2f}" for v in sorted(lambda_bs_norm[lambda_bs_norm >= 0.01], reverse=True))
            
            lambda_ue_norm = Lambda_ue_beam.numpy() / Lambda_ue_beam.numpy().mean()
            lambda_ue_str = ','.join(f"{v:.2f}" for v in sorted(lambda_ue_norm[lambda_ue_norm >= 0], reverse=True))
            print(f"    Lambda_bs (norm≥0.01): {lambda_bs_str}")
            print(f"    Lambda_ue (norm≥0.00): {lambda_ue_str}")

            s2_elapsed = time.time() - s2_start
            tr_P = float(tf.math.real(tf.linalg.trace(P_trx)).numpy())
            bs_pattern_compact = ','.join(str(int(b)) for b in pattern_32.numpy())
            print(f"    BS_pattern=[{bs_pattern_compact}] (len={len(pattern_32.numpy())}) | "
                  f"C_TRX={float(C_TRX.numpy()):.2f} bps/Hz ({float(C_TRX.numpy())/C_AE*100:.1f}%) | "
                  f"tr(P)={tr_P:.1f} iter_P={int(k_final_trx.numpy())+1} | {s2_elapsed:.1f}s")
            
            # Save Power Opt result with Lambda
            result_mgr.save_stage_result(
                area, freq, ue, "PowerOpt",
                C_TRX_UE=float(C_UE.numpy()),
                C_TRX_BS=float(C_BS.numpy()),
                C_TRX=float(C_TRX.numpy()),
                lambda_bs_beam_str=lambda_bs_str,
                lambda_ue_beam_str=lambda_ue_str,
                ue_beams_str=ue_beams_str,
                bs_pattern_str=bs_pattern_str
            )
            
            # ===== Stage 3: BS Layer Best Pattern Selection (UPLINK, 64-beam codebook) =====
            s3_start = time.time()
            print("  Stage 3: BS Layer best pattern selection...")
            
            pattern_layer, C_Layer_no_power, best_k = trx_selector.select_bs_layer_beams_cyclic_pattern(
                channel_data['U_bs'], channel_data['U_ue'],
                channel_data['Omega_ul'], channel_data['H_mean_ul'],
                W_ue_fixed
            )
            
            layer_pattern_compact = ','.join(str(int(b)) for b in pattern_layer.numpy())
            print(f"    Layer_pattern=[{layer_pattern_compact}] (len={best_k}, no power opt) | C={float(C_Layer_no_power.numpy()):.2f} bps/Hz")
            
            # Expand pattern to full 64 layers (cyclic repetition)
            Beams_BS_layer_full = trx_selector._expand_pattern_to_full_beams(
                pattern_layer, config.n_bs_layers
            )
            
            # Save Stage 3 result (before power opt)
            bs_layer_pattern_str = ','.join(str(int(b)) for b in pattern_layer.numpy())
            result_mgr.save_stage_result(
                area, freq, ue, "Stage3",
                C_TRX_UE=float(C_UE.numpy()),
                C_TRX_BS=float(C_BS.numpy()),
                C_TRX=float(C_TRX.numpy()),
                C_Layer_no_power=float(C_Layer_no_power.numpy()),
                ue_beams_str=ue_beams_str,
                bs_pattern_str=bs_pattern_str,
                bs_layer_pattern_str=bs_layer_pattern_str
            )
            
            # Power optimization (Layer-level)
            print("  Power optimization (Layer-level)...")
            
            # Generate Layer-level codebook
            F_layer = DFTCodebook.generate_2d_dft_codebook_tf(
                config.layer_ant_per_dim, config.layer_oversample
            )
            
            W_bs_layer = BeamformingMatrix.construct_beamforming_blockdiag_tf(
                F_layer, Beams_BS_layer_full, config.n_bs_layers
            )
            
            H_mean_beam = BeamDomainTransform.transform_mean_channel(
                channel_data['H_mean_ul'], W_bs_layer, W_ue_fixed
            )
            _, U_bs_beam, Lambda_bs_beam = BeamDomainTransform.transform_bs_covariance(
                channel_data['U_bs'], channel_data['U_ue'],
                channel_data['Omega_ul'], W_bs_layer, W_ue_fixed
            )
            _, U_ue_beam, Lambda_ue_beam = BeamDomainTransform.transform_ue_covariance(
                channel_data['U_bs'], channel_data['U_ue'],
                channel_data['Omega_ul'], W_bs_layer, W_ue_fixed
            )
            Omega_beam = BeamDomainTransform.transform_coupling_matrix(
                channel_data['U_bs'], channel_data['U_ue'],
                channel_data['Omega_ul'], W_bs_layer, W_ue_fixed,
                U_bs_beam, U_ue_beam
            )
            
            C_Layer, P_layer, k_final_layer = trx_selector.capacity_calc.wen2011_optimize_tf(
                U_bs_beam, U_ue_beam, Omega_beam, H_mean_beam
            )
            
            # Lambda analysis (inline, NO regularization)
            lambda_bs_norm = Lambda_bs_beam.numpy() / Lambda_bs_beam.numpy().mean()
            lambda_bs_str = ','.join(f"{v:.2f}" for v in sorted(lambda_bs_norm[lambda_bs_norm >= 0.01], reverse=True))
            
            lambda_ue_norm = Lambda_ue_beam.numpy() / Lambda_ue_beam.numpy().mean()
            lambda_ue_str = ','.join(f"{v:.2f}" for v in sorted(lambda_ue_norm[lambda_ue_norm >= 0.00], reverse=True))
            
            print(f"    Lambda_bs (norm≥0.01): {lambda_bs_str}")
            print(f"    Lambda_ue (norm≥0.00): {lambda_ue_str}")

            tr_P = float(tf.math.real(tf.linalg.trace(P_layer)).numpy())
            layer_pattern_compact = ','.join(str(int(b)) for b in pattern_layer.numpy())
            print(f"    Layer_pattern=[{layer_pattern_compact}] (len={best_k}) | "
                  f"C_Layer={float(C_Layer.numpy()):.2f} bps/Hz ({float(C_Layer.numpy())/C_AE*100:.1f}%) | "
                  f"tr(P)={tr_P:.1f} iter_P={int(k_final_layer.numpy())+1} | {time.time()-s3_start:.1f}s")
            
            # Update Stage 3 result (with power opt and Lambda)
            result_mgr.save_stage_result(
                area, freq, ue, "Stage3_PowerOpt",
                C_TRX_UE=float(C_UE.numpy()),
                C_TRX_BS=float(C_BS.numpy()),
                C_TRX=float(C_TRX.numpy()),
                C_Layer=float(C_Layer.numpy()),
                lambda_bs_beam_str=lambda_bs_str,
                lambda_ue_beam_str=lambda_ue_str,
                ue_beams_str=ue_beams_str,
                bs_pattern_str=bs_pattern_str,
                bs_layer_pattern_str=bs_layer_pattern_str
            )
            
            # ===== Stage 4: Minimal Greedy Layer Addition (UPLINK, Stage 3 unique beams) =====
            if config.enable_minimal_greedy_add:
                s4_start = time.time()
                print("  Stage 4: Minimal greedy layer addition (target: 90% of C_Layer)...")
                
                layer_add = BSLayerMinimalGreedyAdd(config)
                
                # Use Stage 3 pattern (unique beams extracted inside method), start from 0 layers
                init_beams_bs_tf = tf.constant([], dtype=tf.int32)
                
                result_abm = layer_add.minimal_greedy_add_bs_layers(
                    channel_data['U_bs'], channel_data['U_ue'],
                    channel_data['Omega_ul'], channel_data['H_mean_ul'],
                    W_ue_fixed, pattern_layer, init_beams_bs_tf, C_Layer
                )
                
                C_ABM = float(result_abm['C_ABM'].numpy())
                N_BS_layers = result_abm['N_BS_layers']
                
                # Unique beams 추출
                abm_beams = result_abm['beams_abm'].numpy()
                abm_unique = sorted(set(abm_beams))
                abm_unique_compact = ','.join(str(int(b)) for b in abm_unique)
                
                s4_elapsed = time.time() - s4_start
                print(f"    ABM_unique=[{abm_unique_compact}] (N={len(abm_unique)}, total={N_BS_layers}) | "
                      f"C_ABM={C_ABM:.2f} bps/Hz ({C_ABM/float(C_Layer.numpy())*100:.1f}%) | "
                      f"{s4_elapsed:.1f}s")
                
                # Save Stage4 result (NO Lambda)
                bs_beams_abm_str = ','.join(str(int(b)) for b in result_abm['beams_abm'].numpy())
                C_ABM_hist = result_abm['C_ABM_hist']
                result_mgr.save_stage_result(
                    area, freq, ue, "Stage4",
                    C_TRX_UE=float(C_UE.numpy()),
                    C_TRX_BS=float(C_BS.numpy()),
                    C_TRX=float(C_TRX.numpy()),
                    C_Layer=float(C_Layer.numpy()),
                    C_ABM_hist_list=C_ABM_hist,
                    ue_beams_str=ue_beams_str,
                    bs_pattern_str=bs_pattern_str,
                    bs_layer_pattern_str=bs_layer_pattern_str,
                    bs_beams_abm_str=bs_beams_abm_str
                )
                
                # Generate polar plot (if enabled)
                if config.enable_polar_plots:
                    print("  Generating polar plot...")
                    polar_plotter.plot_ue_polar(
                        area, freq, ue,
                        bs_beams=result_abm['beams_abm'].numpy(),
                        ue_beams=Beams_UE_opt.numpy()
                    )
            
            # UE processing complete - calculate ETA
            ue_elapsed = time.time() - ue_start
            ue_times.append(ue_elapsed)
            
            avg_time_per_ue = sum(ue_times) / len(ue_times)
            remaining_ues = len(config.ue_list) - (idx + 1)
            eta_seconds = avg_time_per_ue * remaining_ues
            eta_minutes = eta_seconds / 60
            
            print(f"  UE elapsed: {ue_elapsed:.1f}s | Avg: {avg_time_per_ue:.1f}s/UE | ETA: {eta_minutes:.1f}min ({remaining_ues} UEs left)")
            print("  ✓ Complete")
            print()
            
        except Exception as e:
            print(f"  ERROR: {e}")
            print()
            continue
    
    # Overall completion
    overall_elapsed = time.time() - overall_start
    print("="*80)
    print(f"P1O Processing Complete - Total elapsed: {overall_elapsed/60:.1f} minutes")
    print("="*80)

if __name__ == "__main__":
    main()
