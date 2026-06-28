#!/usr/bin/env python3
# ======================================================================
# P1P_BM_SWOMP_2511v3.py
# P1P: Bidirectional Wen2011 + SWOMP Beam Selection
# 
# === Purpose ===
# Bidirectional beam management using Wen2011 optimal covariance + SWOMP beam selection
# - Stage 1: Uplink UE beam selection (SWOMP + Capacity Greedy, UL parameters)
# - Stage 2: Downlink BS beam selection (SWOMP + Capacity Greedy, DL parameters)
#
# === Architecture ===
# Stage 1 (Uplink, UE TX):
#   UE: 4 TRX × 4 AE/TRX = 16 AE, 16-beam codebook per TRX (2×2 subarray)
#   BS: 1024 AE (no beamforming optimization at RX)
#   Method: Wen2011 → EVD → SWOMP (r_mode shared candidates) → Greedy (4 beams)
#
# Stage 2 (Downlink, BS TX):
#   BS: 64 layers × 16 AE/layer = 1024 AE, 64-beam codebook per layer (4×4 subarray)
#   UE: Stage 1 result fixed (4 TRX)
#   Method: Beam domain transform → Wen2011 → EVD → SWOMP (r_mode shared candidates) → Greedy (64 beams) → capacity history
#
# === Codebook ===
# Stage 1: 2×2 subarray, 16-beam codebook (F_2,2,2,2)
# Stage 2: 4×4 subarray, 64-beam codebook (F_4,2,4,2)
#
# === SWOMP Algorithm ===
# Output: [r_mode] shared candidate pool (all layers use same candidates)
# Key optimization: TensorFlow graph execution + TensorArray (minimize CPU-GPU sync)
# Complexity: r_mode × N_Codebook
#   - UE: 4 × 16 = 64 evaluations
#   - BS: 4 × 64 = 256 evaluations
#
# === Naming Convention ===
# 
# 1. Global Rules (모든 코드):
#    a) Eigenvectors/Matrices (U, R, Lambda, V, W):
#       - Use _bs / _ue suffix (physical entity)
#       - Example: U_bs, U_ue, U_ue_beam, U_bs_beam_n, V_bs_n, W_ue
#    
#    b) Channel Parameters (Omega, H_mean):
#       - Use _ul / _dl suffix (link direction)
#       - Example: Omega_ul, H_mean_ul, Omega_beam_dl, H_mean_beam_dl_n
#
# 2. Class-specific Rules:
#    
#    a) BeamDomainCapacity:
#       - Internal methods use _rx / _tx (abstracted role)
#       - _fixed_point_loop_tf(U_rx, U_tx, n_rx, n_tx, d_rx, d_tx, ...)
#       - wen2011_optimize_tf(U_rx_beam, U_tx_beam, ...)
#       - Caller responsibility: pass (U_bs, U_ue) or (U_ue, U_bs) in correct order
#       - UL example: wen2011_optimize_tf(U_bs, U_ue, Omega_ul, H_mean_ul)
#       - DL example: wen2011_optimize_tf(U_ue_beam, U_bs, Omega_dl, H_mean_dl)
#    
#    b) BeamDomainTransform:
#       - Uses _bs / _ue (physical entity)
#       - Methods: transform_covariance_dual(U_bs, U_ue, link_direction, target_side)
#       - Returns: U_bs_beam or U_ue_beam (no _ul/_dl suffix)
#    
#    c) Stage1_UE_BeamSelector / Stage2_BS_BeamSelector:
#       - Uses _bs / _ue (physical entity)
#       - Channel params use _ul / _dl (link direction)
#       - select_ue_beams(U_bs, U_ue, Omega_ul, H_mean_ul)
#       - select_bs_beams(U_bs, U_ue, Omega_ul, H_mean_ul, W_ue)
#    
#    d) DFTCodebook, BeamformingMatrix, TwoStageBeamSelector:
#       - Generic, no specific _bs/_ue distinction
#       - Use descriptive names (F_codebook, W_test, beam_indices)
#
# 3. Summary:
#    - Eigenvectors: _bs / _ue everywhere (except BeamDomainCapacity internals)
#    - Channels: _ul / _dl everywhere
#    - BeamDomainCapacity only: _rx / _tx internally for abstraction
#
# === Input ===
# P1I: Weichselberger parameters (U_bs, U_ue, Omega, H_mean)
#
# === Output ===
# C_AE: Full digital capacity (UL, W_UE=I, W_BS=I, Wen2011)
# C_UE: Stage 1 UE beam selection result (UL)
# C_BS_hist: Stage 2 BS beam selection capacity history (DL)
# CSV: ue,C_AE,C_UE,C_BS_hist,lambda_UE,lambda_BS,ue_beams,bs_beams
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
import argparse
from pathlib import Path

# TensorFlow environment
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
gpu_num = 0
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

# GPU memory setup (4GB limit)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Set memory limit to 4GB (4096 MB)
        tf.config.experimental.set_memory_growth(gpus[0], False)
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]
        )
        print(f"GPU memory limit set to 4GB")
    except RuntimeError as e:
        # Fallback to memory growth if configuration fails
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
            print(f"GPU memory growth enabled (fallback)")
        except RuntimeError as e2:
            print(f"GPU memory configuration failed: {e2}")

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
#   - P1P_Config: Configuration management
#   - DFTCodebook: DFT codebook (TRX/Layer level)
#   - BeamformingMatrix: Block-diagonal beamforming
#   - BeamDomainTransform: Beam domain Weichselberger transform (UPLINK)
#
# LEVEL 1: Config-Dependent
#   - DataLoader: P1I data loading
#   - BeamDomainCapacity: Wen2011 capacity calculation (UPLINK)
#   - P1P_ResultManager: CSV result management
#
# LEVEL 2: Beam Selection
#   - SWOMPBeamSelector: SWOMP algorithm (weighted matching)
#   - Stage1_UE_BeamSelector: Stage 1 UE beam selection (UL Wen2011 + SWOMP)
#   - Stage2_BS_BeamSelector: Stage 2 BS beam selection (DL Wen2011 + SWOMP + DL capacity history)
#
# LEVEL 3: Main Execution
#   - main: Pipeline orchestration
# ======================================================================

# ===== LEVEL 0: Independent Base Classes =====

# ----- P1P_Config -----
class P1P_Config:
    """P1P bidirectional beam management configuration
    
    Stages:
    - Stage 1: Uplink UE TRX beam selection (r_UE=4 beams, Wen2011 + SWOMP)
    - Stage 2: Downlink BS Layer beam selection (64 layers, Wen2011 + SWOMP + capacity history)
    """
    
    def __init__(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Input data paths
        self.P1I_INPUT_DIR = os.path.join(script_dir, "P1I_Weichsel_Chunk_Results")
        
        # Output results path
        self.P1P_OUTPUT_DIR = os.path.join(script_dir, "P1P_Project")
        os.makedirs(self.P1P_OUTPUT_DIR, exist_ok=True)
        
        # Filtering settings
        self.target_areas = [1]
        self.target_freqs = [7.5]
        
        # UE hardware (N_UE,*)
        self.n_ue_trx = 4
        self.n_ue_trx_ae = 4
        self.n_ue_ae = self.n_ue_trx * self.n_ue_trx_ae
        self.n_ue_trx_row = 2
        self.n_ue_trx_col = 2
        
        # BS hardware (N_BS,*)
        self.n_bs_layer = 64
        self.n_bs_layer_ae = 16
        self.n_bs_ae = self.n_bs_layer * self.n_bs_layer_ae
        self.n_bs_layer_row = 4
        self.n_bs_layer_col = 4
        
        # Codebook (N_*,Beams)
        self.n_oversample = 2
        self.n_ue_trx_beams = (self.n_oversample * self.n_ue_trx_row)**2
        self.n_bs_layer_beams = (self.n_oversample * self.n_bs_layer_row)**2
        
        # Selection (r_*)
        self.r_mode = self.n_ue_trx  # Number of modes for Two-Stage selection (MIMO rank)
        self.r_bs_cand_max = 32  # BS SWOMP max candidates (adaptive stopping)
        self.r_bs_residual_threshold = 0.05  # Stop if ||R||/||V|| < threshold
        
        # Capacity calculation settings (Wen2011)
        self.outer_eps = 0.1 # 0.05
        self.outer_max_iter = 50
        self.inner_max_iter = 100
        self.regularization = 1e-20
        
        # SNR settings
        self.SNR_dB = 10.0
        self.SNR_linear = 10.0**(self.SNR_dB / 10.0)
        
        # Test mode settings
        self.test_mode = False
        self.test_ue_list = [57, 18, 68, 112, 678]  # UE indices (Area-agnostic, unused when test_mode=False)
        
        # Partition settings (for parallel execution)
        self.partition_id = None
        
        # Detect UE list
        self.detect_ue_list()
    
    def detect_ue_list(self):
        """Scan target UEs from P1I"""
        
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
        
        # Sort UE list
        self.ue_list = sorted(p1i_ues)
        
        # Test mode filtering
        if self.test_mode:
            self.ue_list = [(a, f, u) for a, f, u in self.ue_list if u in self.test_ue_list]
        
        if self.ue_list:
            print(f"Target UEs: {len(self.ue_list)}")
            area_groups = {}
            for area, freq, ue in self.ue_list:
                key = f"Area{area}_{freq}GHz"
                area_groups[key] = area_groups.get(key, 0) + 1
            for area_freq, count in sorted(area_groups.items()):
                print(f"  - {area_freq}: {count} UEs")
        else:
            print("Warning: No target UEs found")

# ----- DFTCodebook -----
class DFTCodebook:
    """DFT Codebook generation
    
    Naming Convention:
        Uses generic naming (no _bs/_ue distinction).
    """
    
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
        """2D DFT codebook F = F_{N,K} ⊗ F_{N,K}
        
        Args:
            ant_per_dim: 1D antennas (N)
            oversample: Oversample factor (K)
        
        Returns:
            F: [ant_per_dim², (ant_per_dim*oversample)²] complex64
        """
        F_1d = DFTCodebook.generate_1d_dft_tf(ant_per_dim, oversample)
        n_ant = ant_per_dim
        n_beams = ant_per_dim * oversample
        F_kron = tf.einsum('ij,kl->ikjl', F_1d, F_1d)
        F_kron = tf.reshape(F_kron, [n_ant**2, n_beams**2])
        return F_kron

# ----- BeamformingMatrix -----
class BeamformingMatrix:
    """Beamforming matrix construction
    
    Naming Convention:
        Uses generic naming (no _bs/_ue distinction).
    """
    
    @staticmethod
    @tf.function
    def construct_beamforming_blockdiag_tf(F: tf.Tensor, 
                                          beam_indices: tf.Tensor,
                                          n_total_trx: int) -> tf.Tensor:
        """Block-diagonal beamforming W = blkdiag(w_1, ..., w_L)
        
        Args:
            F: [n_ae_per_trx, n_cb_per_trx] DFT codebook (complex64)
            beam_indices: [n_selected] selected beam indices
            n_total_trx: Total TRX count (int)
        
        Returns:
            W: [n_total_ae, n_selected] complex64
        """
        n_ae_per_trx = tf.shape(F)[0]
        n_total_ae = n_ae_per_trx * n_total_trx
        n_selected = tf.shape(beam_indices)[0]
        W = tf.zeros([n_total_ae, n_selected], dtype=tf.complex64)
        
        # Build block-diagonal W
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
        W = tf.tensor_scatter_nd_update(W, indices, updates)
        
        return W

# ----- BeamDomainTransform -----
class BeamDomainTransform:
    """Beam domain channel transformations
    
    Naming Convention:
        Uses _bs / _ue suffix (physical entity).
        Channel parameters use _ul / _dl (link direction).
    """
    
    @staticmethod
    @tf.function
    def transform_mean_channel(H_mean: tf.Tensor, W_bs: tf.Tensor, 
                               W_ue: tf.Tensor) -> tf.Tensor:
        """Mean channel transform H_mean_beam = W_BS^H @ H_mean @ W_UE
        
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
    def transform_covariance_dual(U_bs: tf.Tensor, U_ue: tf.Tensor,
                                  Omega: tf.Tensor, W_bs: tf.Tensor, W_ue: tf.Tensor,
                                  link_direction: str, target_side: str) -> tuple:
        """Covariance transform (UL/DL dual support)
        
        Args:
            U_bs: [n_bs, n_bs] complex64
            U_ue: [n_ue, n_ue] complex64
            Omega: [n_bs, n_ue] for UL, [n_ue, n_bs] for DL
            W_bs, W_ue: beamforming matrices
            link_direction: "UL" or "DL"
            target_side: "BS" or "UE" (which side to compute covariance)
        
        UL (BS RX, UE TX):
            - target_side="UE" → UE TX covariance
            - target_side="BS" → BS RX covariance
        DL (UE RX, BS TX):
            - target_side="BS" → BS TX covariance
            - target_side="UE" → UE RX covariance
        
        Returns:
            R_beam: [B_target, B_target] covariance
            U_beam: [B_target, B_target] eigenvectors
            Lambda_beam: [B_target] eigenvalues
        """
        if link_direction == "UL":
            if target_side == "UE":
                # UL UE TX covariance
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
            else:  # "BS"
                # UL BS RX covariance
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
        else:  # "DL"
            # Omega는 [n_ue, n_bs] 순서 (DL convention)
            if target_side == "BS":
                # DL BS TX covariance
                T_ue = tf.linalg.adjoint(U_ue) @ W_ue
                v_ue = tf.reduce_sum(tf.abs(T_ue)**2, axis=1)
                d_bs = tf.linalg.matvec(Omega, v_ue, transpose_a=True)
                U_bs_W_bs = tf.linalg.adjoint(U_bs) @ W_bs
                d_bs_complex = tf.cast(d_bs, tf.complex64)
                R_bs_beam = tf.linalg.adjoint(U_bs_W_bs) @ tf.linalg.diag(d_bs_complex) @ U_bs_W_bs
                Lambda_bs_beam, U_bs_beam = tf.linalg.eigh(R_bs_beam)
                Lambda_bs_beam = tf.math.real(Lambda_bs_beam)
                Lambda_bs_beam = tf.maximum(Lambda_bs_beam, 0.0)
                return R_bs_beam, U_bs_beam, Lambda_bs_beam
            else:  # "UE"
                # DL UE RX covariance
                T_bs = tf.linalg.adjoint(U_bs) @ W_bs
                v_bs = tf.reduce_sum(tf.abs(T_bs)**2, axis=1)
                d_ue = tf.linalg.matvec(Omega, v_bs)
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
                                  Omega_ul: tf.Tensor, W_bs: tf.Tensor,
                                  W_ue: tf.Tensor, U_bs_beam: tf.Tensor,
                                  U_ue_beam: tf.Tensor) -> tf.Tensor:
        """Coupling matrix transform Omega_beam_ul = |V_BS|^2^T @ Omega_ul @ |V_UE|^2
        
        Args:
            U_bs: [n_bs, n_bs] complex64
            U_ue: [n_ue, n_ue] complex64
            Omega_ul: [n_bs, n_ue] float32 UPLINK
            W_bs: [n_bs, B_bs] complex64
            W_ue: [n_ue, B_ue] complex64
            U_bs_beam: [B_bs, B_bs] complex64
            U_ue_beam: [B_ue, B_ue] complex64
            
        Returns:
            Omega_beam_ul: [B_bs, B_ue] float32
        """
        V_bs = tf.linalg.adjoint(U_bs) @ W_bs @ U_bs_beam
        V_ue = tf.linalg.adjoint(U_ue) @ W_ue @ U_ue_beam
        V_bs_abs2 = tf.abs(V_bs)**2
        V_ue_abs2 = tf.abs(V_ue)**2
        return tf.transpose(V_bs_abs2) @ Omega_ul @ V_ue_abs2

# ===== LEVEL 1: Config-Dependent Classes =====

# ----- DataLoader -----
class DataLoader:
    """P1I Weichselberger parameter loader
    
    Returns:
        U_bs, U_ue (physical entity)
        Omega_ul, H_mean_ul (UL parameters from P1I)
    """
    
    def __init__(self, config: P1P_Config):
        self.config = config
        self.p1i_cache = {}
    
    def load_p1i_channel_params(self, area: int, freq: float, ue: int) -> dict:
        """Load P1I channel parameters and convert to UPLINK
        
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
        Omega_dl_np = data['P1G_Omega'][idx_in_chunk]
        H_mean_dl_np = data['P1H_H_mean'][idx_in_chunk]
        channel_model = data['enhanced_metadata'][idx_in_chunk]['channel_model']
        
        # Uplink conversion: transpose DL → UL
        Omega_ul_np = Omega_dl_np.T
        H_mean_ul_np = H_mean_dl_np.T
        
        # Normalize
        Omega_ul_norm_np, H_mean_ul_norm_np, kappa = self._normalize_channel(
            Omega_ul_np, H_mean_ul_np, U_bs_np.shape[0], U_ue_np.shape[0]
        )
        
        # Convert to TensorFlow tensors
        U_bs_tf = tf.constant(U_bs_np, dtype=tf.complex64)
        U_ue_tf = tf.constant(U_ue_np, dtype=tf.complex64)
        Omega_ul_tf = tf.constant(Omega_ul_norm_np, dtype=tf.float32)
        H_mean_ul_tf = tf.constant(H_mean_ul_norm_np, dtype=tf.complex64)
        
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

# ----- BeamDomainCapacity -----
class BeamDomainCapacity:
    """Beam domain capacity calculation (Wen2011 Algorithm 1)
    
    Naming Convention:
        Methods in this class use _rx/_tx suffix (abstracted RX/TX role).
        Caller should pass physical variables (U_bs, U_ue) in correct order.
    """
    
    def __init__(self, config: P1P_Config):
        self.config = config
        self.wen2011_call_count = 0
    
    def _find_inverse_water_level_tf(self, eigvals: tf.Tensor, valid_mask: tf.Tensor, 
                                    power_total: tf.Tensor) -> tf.Tensor:
        """Water-filling level ν = 1/μ"""
        valid_indices = tf.where(valid_mask)[:, 0]
        valid_eigvals = tf.gather(eigvals, valid_indices)
        inv_valid_eigvals = 1.0 / valid_eigvals
        nu_min = 0.0
        nu_max = power_total + tf.reduce_max(inv_valid_eigvals)
        
        def cond(nu_min, nu_max, i):
            return tf.logical_and(i < 100, tf.abs(nu_max - nu_min) >= 1e-10)
        
        def body(nu_min, nu_max, i):
            nu_mid = (nu_min + nu_max) / 2.0
            allocated_power = tf.reduce_sum(tf.maximum(nu_mid - inv_valid_eigvals, 0.0))
            nu_min_new = tf.cond(allocated_power < power_total, lambda: nu_mid, lambda: nu_min)
            nu_max_new = tf.cond(allocated_power < power_total, lambda: nu_max, lambda: nu_mid)
            return nu_min_new, nu_max_new, i + 1
        
        nu_min_final, nu_max_final, _ = tf.while_loop(cond, body, [nu_min, nu_max, 0])
        return (nu_min_final + nu_max_final) / 2.0
    
    @tf.function
    def _fixed_point_loop_tf(self, P: tf.Tensor, U_rx: tf.Tensor, U_tx: tf.Tensor,
                            Omega: tf.Tensor, H_mean: tf.Tensor, max_iter_inner: int,
                            reg_complex: tf.Tensor, n_rx: tf.Tensor, n_tx: tf.Tensor) -> tuple:
        """Fixed-point loop"""
        gamma = tf.ones(n_rx, dtype=tf.float32)
        psi = tf.ones(n_tx, dtype=tf.float32)
        gamma_history = tf.TensorArray(tf.float32, size=max_iter_inner, dynamic_size=False)
        psi_history = tf.TensorArray(tf.float32, size=max_iter_inner, dynamic_size=False)
        diff_history = tf.TensorArray(tf.float32, size=max_iter_inner, dynamic_size=False)
        
        for i in tf.range(max_iter_inner):
            gamma_prev = tf.identity(gamma)
            psi_prev = tf.identity(psi)
            gamma = tf.abs(gamma)
            psi = tf.abs(psi)
            
            Omega_T_gamma = tf.matmul(tf.transpose(Omega), tf.expand_dims(gamma, -1))
            d_tx = tf.reshape(Omega_T_gamma, [-1])
            T = U_tx @ tf.linalg.diag(tf.cast(d_tx, tf.complex64)) @ tf.linalg.adjoint(U_tx)
            
            Omega_psi = tf.matmul(Omega, tf.expand_dims(psi, -1))
            d_rx = tf.reshape(Omega_psi, [-1])
            R = U_rx @ tf.linalg.diag(tf.cast(d_rx, tf.complex64)) @ tf.linalg.adjoint(U_rx)
            
            I_R = tf.eye(n_rx, dtype=tf.complex64) + R + reg_complex * tf.eye(n_rx, dtype=tf.complex64)
            H_mean_H = tf.linalg.adjoint(H_mean)
            Xi = T + H_mean_H @ tf.linalg.solve(I_R, H_mean)
            
            I_R_inv_U_rx = tf.linalg.solve(I_R, U_rx)
            gamma = tf.math.real(tf.reduce_sum(tf.math.conj(U_rx) * I_R_inv_U_rx, axis=0))
            
            I_Xi_P = tf.eye(n_tx, dtype=tf.complex64) + Xi @ P + reg_complex * tf.eye(n_tx, dtype=tf.complex64)
            I_Xi_P_inv_U_tx = tf.linalg.solve(I_Xi_P, U_tx)
            psi = tf.math.real(tf.reduce_sum(tf.math.conj(U_tx) * (P @ I_Xi_P_inv_U_tx), axis=0))
            
            gamma_diff = tf.norm(gamma - gamma_prev)
            psi_diff = tf.norm(psi - psi_prev)
            total_diff = gamma_diff + psi_diff
            
            gamma_history = gamma_history.write(i, gamma)
            psi_history = psi_history.write(i, psi)
            diff_history = diff_history.write(i, total_diff)
        
        all_diffs = diff_history.stack()
        iter_indices = tf.cast(tf.range(1, max_iter_inner + 1), tf.float32)
        diff_per_iter = all_diffs / iter_indices
        best_idx = tf.argmin(diff_per_iter, output_type=tf.int32)
        
        all_gammas = gamma_history.stack()
        all_psis = psi_history.stack()
        best_gamma = all_gammas[best_idx]
        best_psi = all_psis[best_idx]
        
        Omega_T_gamma_best = tf.matmul(tf.transpose(Omega), tf.expand_dims(best_gamma, -1))
        d_tx_best = tf.reshape(Omega_T_gamma_best, [-1])
        T_best = U_tx @ tf.linalg.diag(tf.cast(d_tx_best, tf.complex64)) @ tf.linalg.adjoint(U_tx)
        
        Omega_psi_best = tf.matmul(Omega, tf.expand_dims(best_psi, -1))
        d_rx_best = tf.reshape(Omega_psi_best, [-1])
        R_best = U_rx @ tf.linalg.diag(tf.cast(d_rx_best, tf.complex64)) @ tf.linalg.adjoint(U_rx)
        
        I_R_best = tf.eye(n_rx, dtype=tf.complex64) + R_best
        H_mean_H = tf.linalg.adjoint(H_mean)
        Xi_best = T_best + H_mean_H @ tf.linalg.solve(I_R_best, H_mean)
        
        return best_gamma, best_psi, Xi_best, Omega_psi_best, best_idx
    
    @tf.function
    def _water_filling_p1j(self, Xi: tf.Tensor, power_total: tf.Tensor, 
                           n_tx: tf.Tensor, min_rank: tf.Tensor) -> tuple:
        """Water-filling - Rank-aware
        
        Args:
            Xi: [n_tx, n_tx]
            power_total: float
            n_tx: int
            min_rank: int, min(n_rx, n_tx)
        
        Returns:
            P: [n_tx, n_tx] power allocation matrix
            Lambda_P_r: [min_rank] allocated power eigenvalues
        """
        eigvals, eigvecs = tf.linalg.eigh(Xi)
        eigvals_real = tf.math.real(eigvals)
        
        # 상위 min_rank개만 선택
        sorted_indices = tf.argsort(eigvals_real, direction='DESCENDING')
        top_r_indices = sorted_indices[:min_rank]
        eigvals_top_r = tf.gather(eigvals_real, top_r_indices)
        eigvecs_top_r = tf.gather(eigvecs, top_r_indices, axis=1)  # [n_tx, min_rank]
        
        # min_rank개 고유값에 대해 water-filling
        eigvals_positive = tf.maximum(eigvals_top_r, 0.0)
        valid_mask = eigvals_positive > 1e-30
        n_valid = tf.reduce_sum(tf.cast(valid_mask, tf.float32))
        if n_valid == 0:
            return tf.zeros_like(Xi, dtype=tf.complex64), tf.zeros(min_rank, dtype=tf.float32)
        
        valid_eigvals = eigvals_positive * tf.cast(valid_mask, tf.float32)
        nu = self._find_inverse_water_level_tf(valid_eigvals, valid_mask, power_total)
        inv_eigvals = tf.where(valid_mask, 1.0 / eigvals_positive, 0.0)
        Lambda_P_raw = tf.maximum(nu - inv_eigvals, 0.0)
        Lambda_P_r = Lambda_P_raw * tf.cast(valid_mask, tf.float32)  # [min_rank]
        
        # [n_tx, n_tx]로 복원: P = V_r @ P_r @ V_r^H
        P_r = tf.linalg.diag(tf.cast(Lambda_P_r, tf.complex64))
        P = eigvecs_top_r @ P_r @ tf.linalg.adjoint(eigvecs_top_r)
        return P, Lambda_P_r
    
    @tf.function
    def wen2011_optimize_tf(self, U_rx_beam: tf.Tensor, U_tx_beam: tf.Tensor,
                           Omega_beam: tf.Tensor, H_mean_beam: tf.Tensor) -> tuple:
        """Wen2011 ergodic capacity optimization
        
        Note: Parameters use _rx/_tx naming (abstracted roles).
              Caller should pass physical variables (U_bs, U_ue) in correct order:
              - UL: wen2011_optimize_tf(U_bs, U_ue, Omega_ul, H_mean_ul)
              - DL: wen2011_optimize_tf(U_ue, U_bs, Omega_dl, H_mean_dl)
        
        Args:
            U_rx_beam: [n_rx, n_rx] RX side eigenvectors
            U_tx_beam: [n_tx, n_tx] TX side eigenvectors
            Omega_beam: [n_rx, n_tx] coupling matrix [RX, TX] order
            H_mean_beam: [n_rx, n_tx] mean channel [RX, TX] order
        
        Returns:
            (C_opt, P_opt, k_final, eps_final)
                C_opt: Ergodic capacity
                P_opt: [n_tx, n_tx] Optimal TX covariance
                k_final: Iteration count
                eps_final: Final relative change (convergence metric)
        """
        n_rx = tf.shape(U_rx_beam)[0]
        n_tx = tf.shape(U_tx_beam)[0]
        min_rank = tf.minimum(n_rx, n_tx)
        
        P_UE_TRX = tf.cast(self.config.n_ue_trx, dtype=tf.float32)
        power_budget = P_UE_TRX
        
        max_iter_outer = self.config.outer_max_iter
        max_iter_inner = self.config.inner_max_iter
        eps_outer = tf.constant(self.config.outer_eps, dtype=tf.float32)
        reg = tf.constant(self.config.regularization, dtype=tf.float32)
        reg_complex = tf.cast(reg, tf.complex64)
        P_init = tf.eye(n_tx, dtype=tf.complex64)
        Lambda_P_r_init = tf.ones(min_rank, dtype=tf.float32)
        
        def body(k, P_prev, P_current, Lambda_P_r_prev, converged):
            gamma, psi, Xi, Omega_psi, k_inner = self._fixed_point_loop_tf(
                P_current, U_rx_beam, U_tx_beam, Omega_beam, H_mean_beam, max_iter_inner, reg_complex, n_rx, n_tx
            )
            P_new, Lambda_P_r = self._water_filling_p1j(Xi, power_budget, n_tx, min_rank)
            
            # 수렴도 계산
            diff_lambda = tf.linalg.norm(Lambda_P_r - Lambda_P_r_prev)
            norm_lambda_prev = tf.linalg.norm(Lambda_P_r_prev)
            eps = diff_lambda / (norm_lambda_prev + 1e-10)
            
            # 수렴 판정 (즉시)
            has_converged = tf.logical_and(eps < eps_outer, k >= 2)
            
            # 100번마다 출력
            def _print_progress():
                tf.print("        Wen2011 outer iter", k + 1, "eps:", eps, "Lambda_P_r:", Lambda_P_r)
                return 0
            
            def _no_print():
                return 0
            
            tf.cond(
                tf.equal(tf.math.floormod(k + 1, 100), 0),
                _print_progress,
                _no_print
            )
            
            return k + 1, P_current, P_new, Lambda_P_r, has_converged
        
        def cond(k, P_prev, P_current, Lambda_P_r_prev, converged):
            return tf.logical_and(k < max_iter_outer, tf.logical_not(converged))
        
        k_final, _, P_opt, _, converged_final = tf.while_loop(
            cond, body, 
            [0, P_init, P_init, Lambda_P_r_init, tf.constant(False)],
            maximum_iterations=max_iter_outer,
            shape_invariants=[
                tf.TensorShape([]),
                tf.TensorShape([None, None]),
                tf.TensorShape([None, None]),
                tf.TensorShape([None]),
                tf.TensorShape([])
            ]
        )
        
        gamma_final, psi_final, Xi_final, Omega_psi, _ = self._fixed_point_loop_tf(
            P_opt, U_rx_beam, U_tx_beam, Omega_beam, H_mean_beam, max_iter_inner, reg_complex, n_rx, n_tx
        )
        
        I_Xi_P = tf.eye(n_tx, dtype=tf.complex64) + Xi_final @ P_opt
        d_rx_final = tf.reshape(Omega_psi, [-1])
        R = U_rx_beam @ tf.linalg.diag(tf.cast(d_rx_final, tf.complex64)) @ tf.linalg.adjoint(U_rx_beam)
        I_R = tf.eye(n_rx, dtype=tf.complex64) + R
        
        logdet1 = tf.math.real(tf.linalg.slogdet(I_Xi_P)[1])
        logdet2 = tf.math.real(tf.linalg.slogdet(I_R)[1])
        coupling_term = tf.reduce_sum(gamma_final * tf.squeeze(Omega_psi))
        I_nat = logdet1 + logdet2 - coupling_term
        log2 = tf.math.log(2.0)
        C_opt_tf = I_nat / log2
        
        eps_final = tf.constant(0.0, dtype=tf.float32)
        return C_opt_tf, P_opt, k_final, eps_final

# ===== LEVEL 2: Beam Selection =====

# ----- TwoStageBeamSelector -----
class TwoStageBeamSelector:
    """SWOMP Beam Selection (v3 tex Algorithm 1)
    
    Naming Convention:
        Generic (no _bs/_ue distinction).
        Works with V, Lambda, F_codebook, beam_indices.
    """
    
    @staticmethod
    @tf.function
    def _construct_uniform_beam_matrix_tf(F_codebook: tf.Tensor, 
                                          beam_idx: tf.Tensor,
                                          N_Layer: int) -> tf.Tensor:
        """W[j·1_N] = blkdiag(f[j], ..., f[j]) (tex Line 752)
        
        Args:
            F_codebook: [N_Layer_AE, N_Codebook] complex64
            beam_idx: scalar int32
            N_Layer: int
        
        Returns:
            W_j: [N_Layer*N_Layer_AE, N_Layer] complex64
        """
        beam_indices = tf.fill([N_Layer], beam_idx)
        W_j = BeamformingMatrix.construct_beamforming_blockdiag_tf(
            F_codebook, beam_indices, N_Layer)
        return W_j
    
    @staticmethod
    @tf.function
    def _weighted_correlation_tf(W_j: tf.Tensor, R: tf.Tensor, 
                                 D: tf.Tensor) -> tf.Tensor:
        """ρ = ||W^H R D||²_F (tex Line 753)
        
        Args:
            W_j: [n_AE, N_Layer] complex64
            R: [n_AE, r_mode] complex64
            D: [r_mode, r_mode] complex64
        
        Returns:
            rho: scalar float32
        """
        temp = tf.linalg.adjoint(W_j) @ R @ D
        return tf.reduce_sum(tf.abs(temp)**2)
    
    @tf.function
    def _swomp_iteration_tf(self, R: tf.Tensor, D: tf.Tensor,
                           F_codebook: tf.Tensor, N_Layer: int) -> tuple:
        """Single SWOMP iteration (tex Line 751-759)
        
        Args:
            R: [n_AE, r_mode] residual
            D: [r_mode, r_mode] weight matrix
            F_codebook: [N_Layer_AE, N_Codebook]
            N_Layer: int
        
        Returns:
            j_best: int32 scalar
            R_next: [n_AE, r_mode] updated residual
            rho_max: float32 scalar
        """
        N_Codebook = tf.shape(F_codebook)[1]
        
        # Vectorized correlation for all codebooks
        def compute_rho(j):
            W_j = self._construct_uniform_beam_matrix_tf(F_codebook, j, N_Layer)
            return self._weighted_correlation_tf(W_j, R, D)
        
        # Parallel evaluation (tex Line 751-754)
        rho_values = tf.map_fn(
            compute_rho,
            tf.range(N_Codebook),
            dtype=tf.float32,
            parallel_iterations=16
        )
        
        # Select best (tex Line 755)
        j_best = tf.argmax(rho_values, output_type=tf.int32)
        rho_max = rho_values[j_best]
        
        # Residual update (tex Line 758-759)
        W_best = self._construct_uniform_beam_matrix_tf(F_codebook, j_best, N_Layer)
        
        # Digital beamforming: B = W^† R
        # W is block-diagonal with normalized DFT vectors → W^H W ≈ I
        # Therefore W^† ≈ W^H (more efficient than pinv)
        B = tf.linalg.adjoint(W_best) @ R
        R_next = R - W_best @ B
        
        return j_best, R_next, rho_max
    
    def generate_candidates(self, V: tf.Tensor, Lambda: tf.Tensor, 
                           F_codebook: tf.Tensor, N_Layer: int, 
                           N_Layer_AE: int, r_mode: int,
                           n_cand_max: int = None,
                           residual_threshold: float = None) -> tf.Tensor:
        """SWOMP candidate generation (tex Algorithm 1, Line 747-760)
        
        TensorFlow optimization: TensorArray + deferred .numpy()
        
        Args:
            V: [N_Layer*N_Layer_AE, r_mode] eigenvectors
            Lambda: [r_mode] eigenvalues
            F_codebook: [N_Layer_AE, N_Codebook]
            N_Layer, N_Layer_AE, r_mode: int
            n_cand_max: Max candidates (default: r_mode)
            residual_threshold: Stop if ||R||/||V|| < threshold (adaptive)
        
        Returns:
            candidates: [n_actual] int32 tensor (n_actual <= n_cand_max)
        
        Weighting Strategy:
            - n_cand_max == r_mode: eigenvalue weighting D = diag(√λ)
              (standard SWOMP, all iterations use eigenvalue weights)
            - n_cand_max != r_mode: uniform weighting D = diag(1)
              (extended search, all iterations use uniform weights)
        
        Adaptive Stopping (if residual_threshold provided and n_cand_max != r_mode):
            - Stop if ||R||/||V|| < threshold after iter r_mode
            - Only applies to extended search mode
        """
        if n_cand_max is None:
            n_cand_max = r_mode
        
        adaptive_mode = (n_cand_max != r_mode and residual_threshold is not None)
        
        # Weighting strategy: eigenvalue vs uniform
        # - n_cand_max == r_mode: eigenvalue weighting (standard SWOMP)
        # - n_cand_max != r_mode: uniform weighting (extended search)

        Lambda_weight = tf.cast(Lambda, tf.float32)
        
        # Initialize
        R = tf.identity(V)
        D = tf.linalg.diag(tf.sqrt(Lambda_weight))
        D = tf.cast(D, tf.complex64)
        
        # Initial norm for relative residual
        V_norm = tf.cast(tf.linalg.norm(V), tf.float32)
        
        # TensorArray for GPU-only accumulation (dynamic size for adaptive)
        candidates_ta = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
        rho_ta = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        norm_ta = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        
        # SWOMP iterations (tex Line 750-760)
        n_actual = n_cand_max
        for i in range(n_cand_max):
            j_best, R_next, rho_max = self._swomp_iteration_tf(R, D, F_codebook, N_Layer)
            residual_norm = tf.cast(tf.linalg.norm(R_next), tf.float32)
            residual_norm_rel = residual_norm / V_norm
            
            # Store in TensorArray (no CPU sync)
            candidates_ta = candidates_ta.write(i, j_best)
            rho_ta = rho_ta.write(i, rho_max)
            norm_ta = norm_ta.write(i, residual_norm_rel)
            
            R = R_next
            
            # Adaptive stopping (only in uniform weighting phase)
            if adaptive_mode and i >= r_mode - 1:
                if residual_norm_rel.numpy() < residual_threshold:
                    n_actual = i + 1
                    break
        
        # Stack to tensors
        candidates = candidates_ta.stack()
        rho_values = rho_ta.stack()
        norm_values = norm_ta.stack()
        
        # Convert to numpy and make unique (preserve order)
        candidates_np = candidates.numpy()[:n_actual]
        rho_np = rho_values.numpy()[:n_actual]
        norm_np = norm_values.numpy()[:n_actual]
        
        # Make candidates unique (preserve first occurrence order)
        unique_candidates, unique_indices = np.unique(candidates_np, return_index=True)
        unique_candidates = unique_candidates[np.argsort(unique_indices)]  # Preserve original order
        
        # Logging (single CPU sync at end)
        if adaptive_mode:
            print(f"      SWOMP: r_mode={r_mode}, n_cand={n_actual}/{n_cand_max} (adaptive), N_Codebook={F_codebook.shape[1]}")
        else:
            print(f"      SWOMP: r_mode={r_mode}, n_cand={n_actual}, N_Codebook={F_codebook.shape[1]}")
        
        for i in range(n_actual):
            print(f"        iter[{i+1:2d}/{n_actual:2d}]: j={candidates_np[i]:2d}, "
                  f"ρ={rho_np[i]:.2e}, ||R||/||V||={norm_np[i]:.4f}")
        
        # Early stopping message
        if adaptive_mode and n_actual < n_cand_max:
            print(f"      Early stop: ||R||/||V||={norm_np[n_actual-1]:.4f} < {residual_threshold}")
        
        # Format candidates with padding (show unique)
        cand_str = '[' + ', '.join(f'{c:2d}' for c in unique_candidates) + ']'
        print(f"      Shared candidates (unique): {cand_str}")
        
        return tf.constant(unique_candidates, dtype=tf.int32)
    
    def select_with_capacity_greedy(self, candidates: tf.Tensor, 
                                    F_codebook: tf.Tensor, 
                                    N_Layer: int, n_total_trx: int,
                                    capacity_evaluator_fn,
                                    record_history: bool,
                                    filter_top_r_mode: bool = False,
                                    r_mode_filter: int = None) -> tuple:
        """Capacity-based Greedy selection (tex Line 762-774)
        
        TensorFlow optimization: TensorArray + deferred logging
        
        Args:
            candidates: [n_cand] shared pool
            F_codebook: [N_Layer_AE, N_Codebook]
            capacity_evaluator_fn: Callable[[W], float]
            record_history: bool
            filter_top_r_mode: If True, filter to top r_mode after Layer 1
            r_mode_filter: r_mode value for filtering (required if filter_top_r_mode=True)
        
        Returns:
            beam_indices: [N_Layer] int32 tensor
            capacity_history: list[float] or None
        """
        n_cand = tf.shape(candidates)[0].numpy()
        candidates_np = candidates.numpy()
        
        # TensorArray for accumulation
        selected_beams_ta = tf.TensorArray(dtype=tf.int32, size=N_Layer)
        if record_history:
            capacity_ta = tf.TensorArray(dtype=tf.float32, size=N_Layer)
        
        # Filtered candidates (initially all candidates)
        active_candidates = candidates_np
        n_active = n_cand
        
        # Layer-by-layer selection
        for i in range(N_Layer):
            # Current beam selection
            if i == 0:
                current_beams = tf.constant([], dtype=tf.int32)
            else:
                current_beams = selected_beams_ta.gather(tf.range(i))
            
            C_best = -np.inf
            best_beam = None
            C_list_ta = tf.TensorArray(dtype=tf.float32, size=n_active)
            
            # Evaluate active candidates (tex Line 766-772)
            for j, w in enumerate(active_candidates):
                test_beams = tf.concat([current_beams, [w]], axis=0)
                W_test = BeamformingMatrix.construct_beamforming_blockdiag_tf(
                    F_codebook, test_beams, n_total_trx)
                
                C = capacity_evaluator_fn(W_test)
                C_list_ta = C_list_ta.write(j, C)
                
                if C > C_best:
                    C_best = C
                    best_beam = w
            
            # Store selection
            selected_beams_ta = selected_beams_ta.write(i, best_beam)
            if record_history:
                capacity_ta = capacity_ta.write(i, C_best)
            
            # Logging (per layer, unavoidable for monitoring)
            C_list = C_list_ta.stack().numpy()
            
            # Filter after Layer 1 if requested
            if i == 0 and filter_top_r_mode and r_mode_filter is not None:
                # Sort by capacity (descending)
                sorted_indices = np.argsort(-C_list)  # Descending
                # Select top r_mode candidates (already unique from generate_candidates)
                top_r_indices = sorted_indices[:r_mode_filter]
                active_candidates = active_candidates[top_r_indices]
                n_active = r_mode_filter
                
                # Logging: filtered candidates (sorted by capacity, descending)
                C_list_sorted = C_list[sorted_indices[:r_mode_filter]]  # Top r_mode capacities
                filtered_str = ', '.join(f"b{active_candidates[j]:02d}:{C_list_sorted[j]:5.2f}" 
                                        for j in range(r_mode_filter))
                print(f"      Layer[ 1/{N_Layer}]: {{{filtered_str}}} → b{best_beam:02d}, C={C_best:5.2f}")
                print(f"      Filtered to top {r_mode_filter} candidates: {[f'b{b:02d}' for b in active_candidates]}")
            else:
                # Show only Top-5 for readability when many candidates
                top_k = 5
                if n_active > top_k:
                    sorted_indices = np.argsort(-C_list)  # Descending
                    top_indices = sorted_indices[:top_k]
                    top_strs = [f"b{active_candidates[j]:02d}:{C_list[j]:5.2f}" for j in top_indices]
                    cand_str = ', '.join(top_strs) + f", +{n_active-top_k} more"
                else:
                    cand_str = ', '.join(f"b{active_candidates[j]:02d}:{C_list[j]:5.2f}" 
                                        for j in range(n_active))
                
                print(f"      Layer[{i+1:2d}/{N_Layer}]: {{{cand_str}}} → b{best_beam:02d}, C={C_best:5.2f}")
        
        # Final output
        beam_indices = selected_beams_ta.stack()
        capacity_history = capacity_ta.stack().numpy().tolist() if record_history else None
        
        return beam_indices, capacity_history

# ----- Stage1_UE_BeamSelector -----
class Stage1_UE_BeamSelector:
    """Stage 1: Uplink UE TRX beam selection (v2 tex Section 6)
    
    Naming Convention:
        Uses _bs / _ue (physical entity).
        Channel params use _ul (uplink).
        select_ue_beams(U_bs, U_ue, Omega_ul, H_mean_ul)
    """
    
    def __init__(self, config: P1P_Config):
        self.config = config
        self.capacity_calc = BeamDomainCapacity(config)
        self.two_stage_selector = TwoStageBeamSelector()
        
        # Generate UE TRX codebook
        self.F_ue_trx = DFTCodebook.generate_2d_dft_codebook_tf(
            self.config.n_ue_trx_row, 
            self.config.n_oversample
        )
    
    @tf.function
    def _evaluate_ue_capacity_tf(self, W_ue_test: tf.Tensor, U_bs: tf.Tensor, 
                                  U_ue: tf.Tensor, Omega_ul: tf.Tensor, 
                                  H_mean_ul: tf.Tensor) -> tf.Tensor:
        """UL capacity evaluation with beam domain transform + Wen2011
        
        Args:
            W_ue_test: [n_ue, n_beams] UE beamforming matrix
            U_bs: [n_bs, n_bs] BS eigenvectors
            U_ue: [n_ue, n_ue] UE eigenvectors
            Omega_ul: [n_bs, n_ue] coupling matrix UPLINK
            H_mean_ul: [n_bs, n_ue] mean channel UPLINK
        
        Returns:
            C: Capacity (tf.Tensor scalar)
        """
        # UL beam domain transform
        _, U_ue_beam, _ = BeamDomainTransform.transform_covariance_dual(
            U_bs, U_ue, Omega_ul,
            tf.eye(self.config.n_bs_ae, dtype=tf.complex64), W_ue_test,
            link_direction="UL", target_side="UE")
        
        _, U_bs_beam, _ = BeamDomainTransform.transform_covariance_dual(
            U_bs, U_ue, Omega_ul,
            tf.eye(self.config.n_bs_ae, dtype=tf.complex64), W_ue_test,
            link_direction="UL", target_side="BS")
        
        Omega_beam_ul = BeamDomainTransform.transform_coupling_matrix(
            U_bs, U_ue, Omega_ul,
            tf.eye(self.config.n_bs_ae, dtype=tf.complex64), W_ue_test,
            U_bs_beam, U_ue_beam)
        
        H_mean_beam_ul = BeamDomainTransform.transform_mean_channel(
            H_mean_ul,
            tf.eye(self.config.n_bs_ae, dtype=tf.complex64), W_ue_test)
        
        # Wen2011 capacity (UL: BS RX, UE TX)
        C, _, _, _ = self.capacity_calc.wen2011_optimize_tf(
            U_bs_beam, U_ue_beam, Omega_beam_ul, H_mean_beam_ul)
        return C
    
    def select_ue_beams(self, U_bs: tf.Tensor, U_ue: tf.Tensor, 
                        Omega_ul: tf.Tensor, H_mean_ul: tf.Tensor) -> dict:
        """Stage 1: Uplink UE TRX beam selection (v2 tex Section 6)
        
        TX/RX role: UE transmit [16] (optimize), BS receive [1024]
        Parameters: Omega_UL, H_mean_UL (P1I input)
        
        Args:
            U_bs: [1024, 1024] BS eigenvectors
            U_ue: [16, 16] UE eigenvectors
            Omega_ul: [1024, 16] coupling matrix [RX, TX] order, UPLINK
            H_mean_ul: [1024, 16] mean channel [RX, TX] order, UPLINK
        
        Returns:
            beam_indices: [4] selected UE TRX beam indices
            W_ue: [16, 4] UE beamforming matrix
            C_ae: float, initial capacity (no beamforming)
            C_ue: float, final capacity (with beamforming)
            P_ue: [16, 16] UE transmit covariance
            Lambda_ue: [4] top eigenvalues
        """
        print("  Stage 1: UE beam selection (UPLINK)")
        
        # Step 1: Initial Wen2011 (no beamforming, UL) = C_AE
        # UL: BS RX [1024], UE TX [16]
        C_ae, P_ue, k_final, eps_final = self.capacity_calc.wen2011_optimize_tf(
            U_bs, U_ue, Omega_ul, H_mean_ul
        )
        tr_p = float(tf.math.real(tf.linalg.trace(P_ue)).numpy())
        print(f"    Wen2011 (UL, no BF): C_AE={C_ae.numpy():5.2f} bps/Hz, tr(P)={tr_p:.2f}, k={k_final.numpy()}, eps={eps_final.numpy():.2f}")
        
        # Step 2: Eigenvalue decomposition
        eigvals, eigvecs = tf.linalg.eigh(P_ue)
        eigvals = tf.math.real(eigvals)
        
        # Sort descending
        sorted_indices = tf.argsort(eigvals, direction='DESCENDING')
        eigvals_sorted = tf.gather(eigvals, sorted_indices)
        eigvecs_sorted = tf.gather(eigvecs, sorted_indices, axis=1)
        
        # Extract top r_mode eigenvectors
        Lambda_ue = eigvals_sorted[:self.config.r_mode]
        V_ue = eigvecs_sorted[:, :self.config.r_mode]  # [n_ue, r_mode]
        
        # Format eigenvalues (no normalization)
        eigvals_8 = eigvals_sorted[:8].numpy()
        eig_str = ', '.join(f"{v:.2f}" for v in eigvals_8)
        eig_sum = eigvals_8.sum()
        print(f"    EVD: top 8 eigenvalues (sum={eig_sum:.2f}): {eig_str}")
        
        # Step 3: SWOMP + Greedy
        print(f"    SWOMP Stage 1: Generate shared candidates")
        V_ue_ae = U_ue @ V_ue  # AE-space eigenvectors [16, 4]
        
        candidates = self.two_stage_selector.generate_candidates(
            V_ue_ae, Lambda_ue, self.F_ue_trx,
            self.config.n_ue_trx, self.config.n_ue_trx_ae, self.config.r_mode)
        
        print(f"    SWOMP Stage 2: Greedy capacity-based selection")
        beam_indices, _ = self.two_stage_selector.select_with_capacity_greedy(
            candidates, self.F_ue_trx,
            self.config.n_ue_trx, self.config.n_ue_trx,
            lambda W: float(self._evaluate_ue_capacity_tf(W, U_bs, U_ue, Omega_ul, H_mean_ul).numpy()),
            record_history=False)
        
        print(f"    Selected UE beams: {beam_indices.numpy()}")
        
        # Step 4: Construct beamforming matrix
        W_ue = BeamformingMatrix.construct_beamforming_blockdiag_tf(
            self.F_ue_trx, beam_indices, self.config.n_ue_trx
        )
        
        # Step 5: Recalculate capacity with selected beams (UL with W_UE, W_BS=I)
        _, U_ue_beam, _ = BeamDomainTransform.transform_covariance_dual(
            U_bs, U_ue, Omega_ul,
            tf.eye(self.config.n_bs_ae, dtype=tf.complex64), W_ue,
            link_direction="UL", target_side="UE"
        )
        Omega_beam_ul = BeamDomainTransform.transform_coupling_matrix(
            U_bs, U_ue, Omega_ul,
            tf.eye(self.config.n_bs_ae, dtype=tf.complex64), W_ue,
            tf.eye(self.config.n_bs_ae, dtype=tf.complex64), U_ue_beam
        )
        H_mean_beam_ul = BeamDomainTransform.transform_mean_channel(
            H_mean_ul, 
            tf.eye(self.config.n_bs_ae, dtype=tf.complex64), 
            W_ue
        )
        
        C_ue_beam, P_ue_beam, k_ue, eps_ue = self.capacity_calc.wen2011_optimize_tf(
            U_bs, U_ue_beam, Omega_beam_ul, H_mean_beam_ul
        )
        tr_p_beam = float(tf.math.real(tf.linalg.trace(P_ue_beam)).numpy())
        print(f"    Wen2011 (UL, with W_UE): C={C_ue_beam.numpy():5.2f} bps/Hz, tr(P)={tr_p_beam:.2f}, k={k_ue.numpy()}, eps={eps_ue.numpy():.2f}")
        
        # P_ue_beam eigenvalues
        eigvals_ue_beam = tf.math.real(tf.linalg.eigvalsh(P_ue_beam))
        eigvals_ue_beam_sorted = tf.sort(eigvals_ue_beam, direction='DESCENDING')
        eigvals_ue_beam_8 = eigvals_ue_beam_sorted[:8].numpy()
        eig_str_ue = ', '.join(f"{v:.2f}" for v in eigvals_ue_beam_8)
        eig_sum_ue = eigvals_ue_beam_8.sum()
        print(f"    P_UE eigenvalues (sum={eig_sum_ue:.2f}): {eig_str_ue}")
        
        return {
            'beam_indices': beam_indices,
            'W_ue': W_ue,
            'C_ae': float(C_ae.numpy()),
            'C_ue': float(C_ue_beam.numpy()),
            'P_ue': P_ue,
            'Lambda_ue': Lambda_ue
        }

# ----- Stage2_BS_BeamSelector -----
class Stage2_BS_BeamSelector:
    """Stage 2: Downlink BS Layer beam selection (v2 tex Section 7)
    
    Naming Convention:
        Uses _bs / _ue (physical entity).
        Channel params use _ul/_dl (creates DL from UL internally).
        select_bs_beams(U_bs, U_ue, Omega_ul, H_mean_ul, W_ue)
    """
    
    def __init__(self, config: P1P_Config):
        self.config = config
        self.capacity_calc = BeamDomainCapacity(config)
        self.two_stage_selector = TwoStageBeamSelector()
        
        # Generate BS Layer codebook
        self.F_bs_layer = DFTCodebook.generate_2d_dft_codebook_tf(
            self.config.n_bs_layer_row, 
            self.config.n_oversample
        )
    
    @tf.function
    def _evaluate_bs_capacity_tf(self, W_bs_test: tf.Tensor, U_bs: tf.Tensor, 
                                  U_ue: tf.Tensor, Omega_dl: tf.Tensor, 
                                  H_mean_dl: tf.Tensor, W_ue: tf.Tensor) -> tf.Tensor:
        """DL capacity evaluation with beam domain transform + Wen2011
        
        Args:
            W_bs_test: [n_bs, n_beams] BS beamforming matrix
            U_bs: [n_bs, n_bs] BS eigenvectors
            U_ue: [n_ue, n_ue] UE eigenvectors
            Omega_dl: [n_ue, n_bs] coupling matrix DOWNLINK
            H_mean_dl: [n_ue, n_bs] mean channel DOWNLINK
            W_ue: [n_ue, 4] UE beamforming (fixed from Stage 1)
        
        Returns:
            C: Capacity (tf.Tensor scalar)
        """
        # H_mean_beam_dl_n: [4, n] (RX, TX) order
        H_mean_beam_dl_n = tf.linalg.adjoint(W_ue) @ H_mean_dl @ W_bs_test
        
        # BS TX covariance (DL, n beams)
        _, U_bs_beam_n, _ = BeamDomainTransform.transform_covariance_dual(
            U_bs, U_ue, Omega_dl, W_bs_test, W_ue,
            link_direction="DL", target_side="BS")
        
        # UE RX covariance (DL, 4 beams fixed)
        _, U_ue_beam, _ = BeamDomainTransform.transform_covariance_dual(
            U_bs, U_ue, Omega_dl, W_bs_test, W_ue,
            link_direction="DL", target_side="UE")
        
        # Coupling matrix (DL)
        V_bs_n = tf.linalg.adjoint(U_bs) @ W_bs_test @ U_bs_beam_n
        V_ue = tf.linalg.adjoint(U_ue) @ W_ue @ U_ue_beam
        V_bs_abs2_n = tf.abs(V_bs_n)**2
        V_ue_abs2 = tf.abs(V_ue)**2
        
        # Omega_beam_dl_n: [4, n] (RX, TX) order
        Omega_beam_dl_n = tf.transpose(V_ue_abs2) @ Omega_dl @ V_bs_abs2_n
        
        # Wen2011 capacity (DL: UE RX, BS TX)
        C, _, _, _ = self.capacity_calc.wen2011_optimize_tf(
            U_ue_beam, U_bs_beam_n, Omega_beam_dl_n, H_mean_beam_dl_n)
        return C
    
    def select_bs_beams(self, U_bs: tf.Tensor, U_ue: tf.Tensor, 
                        Omega_ul: tf.Tensor, H_mean_ul: tf.Tensor,
                        W_ue: tf.Tensor) -> dict:
        """Stage 2: Downlink BS Layer beam selection (v2 tex Section 7)
        
        TX/RX: BS transmit [1024], UE receive [4] (W_ue fixed)
        Parameters: Omega_DL = Omega_UL^T, H_mean_DL = H_mean_UL^H (v2 tex Line 1035)
        
        Args:
            U_bs: [1024, 1024] BS eigenvectors
            U_ue: [16, 16] UE eigenvectors (full AE)
            Omega_ul: [1024, 16] coupling matrix UPLINK
            H_mean_ul: [1024, 16] mean channel UPLINK
            W_ue: [16, 4] UE beamforming from Stage 1 (fixed)
        
        Returns:
            beam_indices: [64] BS Layer beam order
            C_bs_hist: list[64] DL capacity for n=1..64 BS beams
            Lambda_bs: [64] top eigenvalues
        
        Process:
            1. DL beam domain transform: apply W_ue, reduce UE dim 16→4
            2. DL Wen2011: optimize BS transmit covariance P_BS [1024, 1024]
            3. EVD: extract top 64 eigenvectors
            4. Two-Stage: determine BS beam order + DL capacity history
        """
        print("  Stage 2: BS beam selection (DOWNLINK)")
        
        # DL parameters (v2 tex Algorithm 2, Line 1035)
        Omega_dl = tf.transpose(Omega_ul)  # [n_ue, n_bs]
        H_mean_dl = tf.linalg.adjoint(H_mean_ul)  # [n_ue, n_bs]
        
        # Step 1: Full BS capacity using DL direct transform (W_BS = I, full 1024 AE)
        # This uses the same DL direct transform as Greedy selection (_evaluate_bs_capacity_tf)
        W_bs_eye = tf.eye(self.config.n_bs_ae, dtype=tf.complex64)
        
        # Mean channel transform: [4, 1024] (RX, TX) order
        H_mean_beam_dl = tf.linalg.adjoint(W_ue) @ H_mean_dl @ W_bs_eye
        
        # BS TX covariance (DL, full 1024 beams)
        _, U_bs_beam_eye, _ = BeamDomainTransform.transform_covariance_dual(
            U_bs, U_ue, Omega_dl, W_bs_eye, W_ue,
            link_direction="DL", target_side="BS")
        
        # UE RX covariance (DL, 4 beams fixed)
        _, U_ue_beam, _ = BeamDomainTransform.transform_covariance_dual(
            U_bs, U_ue, Omega_dl, W_bs_eye, W_ue,
            link_direction="DL", target_side="UE")
        
        # Coupling matrix (DL)
        V_bs_eye = tf.linalg.adjoint(U_bs) @ W_bs_eye @ U_bs_beam_eye
        V_ue = tf.linalg.adjoint(U_ue) @ W_ue @ U_ue_beam
        V_bs_abs2_eye = tf.abs(V_bs_eye)**2
        V_ue_abs2 = tf.abs(V_ue)**2
        
        # Omega_beam_dl: [4, 1024] (RX, TX) order
        Omega_beam_dl = tf.transpose(V_ue_abs2) @ Omega_dl @ V_bs_abs2_eye
        
        # Step 2: Wen2011 (DL BS TX optimization)
        # DL: UE RX [4], BS TX [1024]
        C_bs_full, P_bs, k_final, eps_final = self.capacity_calc.wen2011_optimize_tf(
            U_ue_beam, U_bs_beam_eye, Omega_beam_dl, H_mean_beam_dl
        )
        tr_p = float(tf.math.real(tf.linalg.trace(P_bs)).numpy())
        print(f"    Wen2011 (DL, full BS): C={C_bs_full.numpy():5.2f} bps/Hz, tr(P)={tr_p:.2f}, k={k_final.numpy()}, eps={eps_final.numpy():.2f}")
        
        # Step 3: Eigendecomposition
        eigvals, eigvecs = tf.linalg.eigh(P_bs)
        eigvals = tf.math.real(eigvals)
        sorted_indices = tf.argsort(eigvals, direction='DESCENDING')
        eigvals_sorted = tf.gather(eigvals, sorted_indices)
        eigvecs_sorted = tf.gather(eigvecs, sorted_indices, axis=1)
        
        Lambda_bs = eigvals_sorted[:self.config.n_bs_layer]
        V_bs = eigvecs_sorted[:, :self.config.r_mode]  # Top r_mode for Two-Stage
        V_bs_ae = U_bs @ V_bs  # AE-space
        
        eigvals_8 = eigvals_sorted[:8].numpy()
        eig_str = ', '.join(f"{v:.2f}" for v in eigvals_8)
        eig_sum_bs = eigvals_8.sum()
        print(f"    P_BS eigenvalues (sum={eig_sum_bs:.2f}): {eig_str}")
        
        # Step 4: SWOMP + Greedy
        print(f"    SWOMP Stage 1: Generate shared candidates")
        candidates = self.two_stage_selector.generate_candidates(
            V_bs_ae, Lambda_bs[:self.config.r_mode], self.F_bs_layer,
            self.config.n_bs_layer, self.config.n_bs_layer_ae, self.config.r_mode,
            n_cand_max=self.config.r_bs_cand_max,
            residual_threshold=self.config.r_bs_residual_threshold)
        
        print(f"    SWOMP Stage 2: Greedy capacity-based selection")
        beam_indices, C_bs_hist = self.two_stage_selector.select_with_capacity_greedy(
            candidates, self.F_bs_layer,
            self.config.n_bs_layer, self.config.n_bs_layer,
            lambda W: float(self._evaluate_bs_capacity_tf(W, U_bs, U_ue, Omega_dl, H_mean_dl, W_ue).numpy()),
            record_history=True,
            filter_top_r_mode=True,
            r_mode_filter=self.config.r_mode)
        
        # Beam usage statistics
        beams_np = beam_indices.numpy()
        unique_beams, counts = np.unique(beams_np, return_counts=True)
        sorted_idx = np.argsort(-counts)
        top_beams = unique_beams[sorted_idx][:16]
        top_counts = counts[sorted_idx][:16]
        beam_freq_str = ', '.join(f"b{b}({c})" for b, c in zip(top_beams, top_counts))
        print(f"    Beam usage: {beam_freq_str}")
        print(f"    Final capacity: {C_bs_hist[-1]:.2f} bps/Hz")
        
        return {
            'beam_indices': beam_indices,
            'C_bs_hist': C_bs_hist,
            'Lambda_bs': Lambda_bs
        }

# ----- P1P_ResultManager -----
class P1P_ResultManager:
    """Result manager for CSV output
    
    No specific naming convention.
    """
    
    def __init__(self, config: P1P_Config):
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
        
        if self.config.partition_id is not None:
            csv_filename = f"A{area}_{freq}GHz_P1P_{self._csv_timestamp}_p{self.config.partition_id}.csv"
        else:
            csv_filename = f"A{area}_{freq}GHz_P1P_{self._csv_timestamp}.csv"
        self._csv_path = os.path.join(self.config.P1P_OUTPUT_DIR, csv_filename)
    
    def save_result(self, area: int, freq: float, ue: int, 
                    C_AE: float, C_UE: float,
                    lambda_UE: list, lambda_BS: list,
                    ue_beams: list, bs_beams: list,
                    C_BS_hist: list):
        """Save P1P result to CSV (comma-separated format)
        
        Args:
            area, freq, ue: UE identifier
            C_AE: Full digital capacity (reference)
            C_UE: Stage 1 UE capacity
            lambda_UE: UE eigenvalues (r_ue values)
            lambda_BS: BS eigenvalues (n_bs_layers values)
            ue_beams: UE beam indices (r_ue values)
            bs_beams: BS beam order (n_bs_layers values)
            C_BS_hist: Stage 2 capacity history (n_bs_layers values)
        
        CSV format: "1,2,3,..." (comma-separated)
        """
        if self._csv_path is None:
            self._init_csv(area, freq)
        
        headers = ['ue', 'C_AE', 'C_UE', 'C_BS_hist', 'lambda_UE', 'lambda_BS', 
                   'ue_beams', 'bs_beams']
        
        file_exists = os.path.exists(self._csv_path)
        
        # Format as comma-separated strings
        lambda_UE_str = ','.join(f"{v:.4f}" for v in lambda_UE)
        lambda_BS_str = ','.join(f"{v:.4f}" for v in lambda_BS)
        ue_beams_str = ','.join(str(b) for b in ue_beams)
        bs_beams_str = ','.join(str(b) for b in bs_beams)
        C_BS_hist_str = ','.join(f"{c:.2f}" for c in C_BS_hist)
        
        with open(self._csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(headers)
            
            row = [
                ue,
                f"{C_AE:.2f}",
                f"{C_UE:.2f}",
                C_BS_hist_str,
                lambda_UE_str,
                lambda_BS_str,
                ue_beams_str,
                bs_beams_str
            ]
            writer.writerow(row)
        
        print(f"  Saved to: {self._csv_path}")

# ===== LEVEL 3: Main Execution =====

def main():
    """P1P main pipeline"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='P1P: Bidirectional Wen2011 + SWOMP Beam Selection')
    parser.add_argument('--partition', type=int, default=None, help='Partition index for parallel execution')
    parser.add_argument('--ue-indices', type=str, default=None, help='Comma-separated UE indices to process')
    args = parser.parse_args()
    
    print("=" * 80)
    print("P1P: Bidirectional Wen2011 + SWOMP Beam Selection")
    print("=" * 80)
    
    # Initialize
    config = P1P_Config()
    
    # Store partition ID in config
    config.partition_id = args.partition
    
    # Partition mode: auto-split UE list or use explicit UE indices
    if args.partition is not None:
        if args.ue_indices is not None:
            # Explicit UE indices mode (backward compatibility)
            ue_indices = [int(u) for u in args.ue_indices.split(',')]
            config.ue_list = [(a, f, u) for a, f, u in config.ue_list if u in ue_indices]
            print(f"Partition {args.partition}: Processing {len(config.ue_list)} UEs: {ue_indices}")
        else:
            # Auto-split mode: divide UE list into partitions
            total_ues = len(config.ue_list)
            n_partitions = 8  # Fixed to 8 partitions (L40S 48GB GPU, 4GB per process)
            
            # Calculate partition ranges (same logic as P1O)
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
            ue_ids = [u for _, _, u in config.ue_list]
            
            print(f"Partition {args.partition}/{n_partitions}: Processing {len(config.ue_list)} UEs")
            print(f"  UE indices: {ue_ids[:5]}{'...' if len(ue_ids) > 5 else ''}")
            print(f"  Range: [{start_idx}:{end_idx}] of {total_ues} total UEs")
    
    if len(config.ue_list) == 0:
        print("No UEs to process. Exiting.")
        return
    
    data_loader = DataLoader(config)
    stage1_selector = Stage1_UE_BeamSelector(config)
    stage2_selector = Stage2_BS_BeamSelector(config)
    result_manager = P1P_ResultManager(config)
    
    print(f"\nConfiguration:")
    print(f"  UE: {config.n_ue_trx} TRX, {config.n_ue_ae} AE")
    print(f"  BS: {config.n_bs_layer} Layers, {config.n_bs_ae} AE")
    print(f"  Stage 1: Select {config.r_mode} UE beams from {config.n_ue_trx_beams}-beam codebook")
    print(f"  Stage 2: Select {config.n_bs_layer} BS beams from {config.n_bs_layer_beams}-beam codebook")
    print(f"  Total UEs: {len(config.ue_list)}")
    
    # Process each UE
    for idx, (area, freq, ue) in enumerate(config.ue_list):
        print(f"\n[{idx+1}/{len(config.ue_list)}] Area{area}_{freq}GHz UE{ue}")
        
        try:
            # Load P1I channel parameters
            p1i_params = data_loader.load_p1i_channel_params(area, freq, ue)
            U_bs = p1i_params['U_bs']
            U_ue = p1i_params['U_ue']
            Omega_ul = p1i_params['Omega_ul']
            H_mean_ul = p1i_params['H_mean_ul']
            print(f"  P1I: channel_model={p1i_params['channel_model']}, kappa={p1i_params['kappa']:.2e}")
            
            # Stage 1: UE beam selection (includes C_AE calculation)
            stage1_result = stage1_selector.select_ue_beams(U_bs, U_ue, Omega_ul, H_mean_ul)
            C_AE = stage1_result['C_ae']
            C_UE = stage1_result['C_ue']
            W_ue = stage1_result['W_ue']
            ue_beams = stage1_result['beam_indices'].numpy().tolist()
            lambda_UE = stage1_result['Lambda_ue'].numpy().tolist()
            
            # Stage 2: BS beam selection
            stage2_result = stage2_selector.select_bs_beams(U_bs, U_ue, Omega_ul, H_mean_ul, W_ue)
            bs_beams = stage2_result['beam_indices'].numpy().tolist()
            C_BS_hist = stage2_result['C_bs_hist']
            lambda_BS = stage2_result['Lambda_bs'].numpy().tolist()
            
            # Save result
            result_manager.save_result(
                area, freq, ue, C_AE, C_UE,
                lambda_UE, lambda_BS,
                ue_beams, bs_beams, C_BS_hist
            )
            
            print(f"  Summary: C_AE(UL)={C_AE:5.2f} bps/Hz, C_UE(UL)={C_UE:5.2f} bps/Hz, C_BS(DL)={C_BS_hist[-1]:5.2f} bps/Hz")
            
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'=' * 80}")
    print("P1P pipeline completed")
    print(f"{'=' * 80}")

if __name__ == "__main__":
    main()


