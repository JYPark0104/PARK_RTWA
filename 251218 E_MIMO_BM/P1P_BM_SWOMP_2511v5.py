#!/usr/bin/env python3
# ======================================================================
# P1P_BM_SWOMP_2511v5.py
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
#   Method: Wen2011 → EVD → SWOMP (min_layer shared candidates) → Greedy (4 beams)
#
# Stage 2 (Downlink, BS TX):
#   BS: 64 layers × 16 AE/layer = 1024 AE, 64-beam codebook per layer (4×4 subarray)
#   UE: Stage 1 result fixed (4 TRX)
#   Method: Beam domain transform → Wen2011 → EVD → SWOMP (min_layer shared candidates) → Greedy (64 beams) → capacity history
#
# === Codebook ===
# Stage 1: 2×2 subarray, 16-beam codebook (F_2,2,2,2)
# Stage 2: 4×4 subarray, 64-beam codebook (F_4,2,4,2)
#
# === SWOMP Algorithm ===
# Output: [min_layer] shared candidate pool (all layers use same candidates)
# Key optimization: TensorFlow graph execution + TensorArray (minimize CPU-GPU sync)
# Complexity: min_layer × N_Codebook
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
import shutil
from pathlib import Path

# TensorFlow environment
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
gpu_num = 0
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

# GPU memory setup (2GB limit)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Set memory limit to 2GB (2048 MB)
        tf.config.experimental.set_memory_growth(gpus[0], False)
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)]
        )
        print(f"GPU memory limit set to 2GB")
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
    - Stage 1: Uplink UE TRX beam selection (r_UE=n_ue_layer beams, Wen2011 + SWOMP)
    - Stage 2: Downlink BS Layer beam selection (64 layers, Wen2011 + SWOMP + capacity history)
    
    Modes:
    - test_mode: Process only test_ue_list (for debugging)
    - enable_perturbation: Stage 2 perturbation test (8 layers, 11 trials)
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
        self.n_ue_layer = 4  # UE Layer (primary, 1:1 with TRX in this architecture)
        self.n_ue_trx = self.n_ue_layer  # UE TRX count (derived, 1:1 mapping)
        self.n_ue_layer_ae = 4  # AE per layer
        self.n_ue_ae = self.n_ue_layer * self.n_ue_layer_ae
        self.n_ue_layer_row = 2  # Layer subarray configuration
        self.n_ue_layer_col = 2
        
        # BS hardware (N_BS,*)
        self.n_bs_layer = 64
        self.n_bs_layer_ae = 16
        self.n_bs_ae = self.n_bs_layer * self.n_bs_layer_ae
        self.n_bs_layer_row = 4
        self.n_bs_layer_col = 4
        
        # Codebook (N_*,Beams)
        self.n_oversample = 2
        self.n_ue_layer_beams = (self.n_oversample * self.n_ue_layer_row)**2
        self.n_bs_layer_beams = (self.n_oversample * self.n_bs_layer_row)**2
        
        # Selection (r_*)
        self.min_layer = min(self.n_ue_layer, self.n_bs_layer)  # MIMO rank & max eigenvalue size (=4, fixed)
        self.r_bs_cand_max = 64  # BS SWOMP max candidates (adaptive stopping)
        self.r_bs_residual_threshold = 0.001  # Stop if ||R||/||V|| < threshold
        
        # Capacity calculation settings (Wen2011)
        self.outer_eps = 0.1 # 0.05
        self.outer_max_iter = 50
        self.inner_max_iter = 100
        self.regularization = 1e-20
        
        # Noise power (100 MHz bandwidth)
        self.N0_dBm_per_Hz = -174.0  # Thermal noise PSD at 290K
        self.BW_Hz = 100e6  # 100 MHz
        self.N_dBm = self.N0_dBm_per_Hz + 10 * np.log10(self.BW_Hz)  # -94 dBm
        self.N_mW = 10**(self.N_dBm / 10)  # 3.981e-10 mW

        # Implementation losses (dB)
        self.L_tx_bs_dB = 10.0  # BS TX loss (OBO + Feeder)
        self.L_rx_ue_dB = 10.0  # UE RX loss (NF + Feeder)

        self.L_tx_ue_dB = 6.0   # UE TX loss (OBO + Feeder)
        self.L_rx_bs_dB = 8.0   # BS RX loss (NF + Feeder)

        # UE transmit power (Uplink)
        self.P_ue_total_dBm = 24.0  # 24 dBm (250 mW)
        self.P_ue_total_mW = 10**(self.P_ue_total_dBm / 10)

        # BS transmit power (Downlink)
        self.P_bs_total_dBm = 49.0  # 49 dBm (79.4 W)
        self.P_bs_total_mW = 10**(self.P_bs_total_dBm / 10)

        # Uplink effective TxSNR (UE TX → BS RX)
        P_ue_eff_dBm = self.P_ue_total_dBm - self.L_tx_ue_dB
        P_ue_eff_mW = 10**(P_ue_eff_dBm / 10)
        N_ul_eff_dBm = self.N_dBm + self.L_rx_bs_dB
        N_ul_eff_mW = 10**(N_ul_eff_dBm / 10)
        self.TxSNR_ue = P_ue_eff_mW / N_ul_eff_mW
        self.TxSNR_ue_layer = self.TxSNR_ue / self.n_ue_layer

        # Downlink effective TxSNR (BS TX → UE RX)
        P_bs_eff_dBm = self.P_bs_total_dBm - self.L_tx_bs_dB
        P_bs_eff_mW = 10**(P_bs_eff_dBm / 10)
        N_dl_eff_dBm = self.N_dBm + self.L_rx_ue_dB
        N_dl_eff_mW = 10**(N_dl_eff_dBm / 10)
        self.TxSNR_bs = P_bs_eff_mW / N_dl_eff_mW
        self.TxSNR_bs_layer = self.TxSNR_bs / self.n_bs_layer

        print(f"  Effective UL TxSNR: {10 * np.log10(self.TxSNR_ue):.1f} dB")
        print(f"  Effective DL TxSNR: {10 * np.log10(self.TxSNR_bs):.1f} dB")
        
        # Test mode settings
        self.test_mode = False
        # self.test_ue_list = [57, 18, 68, 112, 678]  # UE indices (Area-agnostic, unused when test_mode=False)
        # self.test_ue_list = [33, 172, 175, 189, 214, 251, 256, 332, 358, 360, 389, 400, 409, 416, 431, 434, 453, 487, 528, 566, 574, 654, 668, 709, 722, 726, 732, 741, 757, 789, 840, 869, 888, 914, 932, 1065, 1089, 1097, 1145, 1167, 1174, 1221, 1224, 1348, 1359, 1360, 1426, 1435, 1453, 1588]  # UE indices (Area-agnostic, unused when test_mode=False)
        self.test_ue_list = [29, 69, 97, 133, 212, 217, 227, 228, 231, 237, 255, 258, 273, 290, 334, 337, 372, 390, 455, 467, 470, 484, 497, 521, 522, 527, 542, 552, 565, 567, 578, 581, 583, 587, 590, 610, 618, 620, 631, 633, 643, 644, 645, 664, 690, 697, 721, 723, 731, 742, 758, 767, 797, 811, 825, 874, 875, 895, 903, 939, 950, 953, 969, 972, 973, 974, 980, 1016, 1055, 1063, 1064, 1130, 1176, 1182, 1215, 1263, 1265, 1268, 1316, 1319, 1343, 1346, 1367, 1376, 1379, 1407, 1408, 1414, 1420, 1432, 1438, 1448, 1455, 1467, 1468, 1480, 1505, 1560, 1568, 1574]  # UE indices (Area-agnostic, unused when test_mode=False)
        
        # Perturbation test settings (Stage 2 only)
        self.enable_perturbation = False
        self.perturbation_alpha = 0.95
        self.perturbation_n_trials = 10
        self.perturbation_n_layers = 8
        
        # SWOMP control
        self.enable_swomp_stage1 = True     # Stage 1 (UE) SWOMP
        self.enable_swomp_stage2 = True    # Stage 2 (BS) SWOMP
        
        # Partition settings (for parallel execution)
        self.partition_id = None
        
        # Resume settings
        self.enable_resume = True  # Resume 모드 활성화
        self.resume_from_dir = "logs_20251115_065837"  # Resume 소스 디렉토리
        
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
                                          n_total_layer: int) -> tf.Tensor:
        """Block-diagonal beamforming W = blkdiag(w_1, ..., w_L)
        
        Args:
            F: [n_ae_per_layer, n_cb_per_layer] DFT codebook (complex64)
            beam_indices: [n_selected] selected beam indices
            n_total_layer: Total layer count (int)
        
        Returns:
            W: [n_total_ae, n_selected] complex64
        """
        n_ae_per_layer = tf.shape(F)[0]
        n_total_ae = n_ae_per_layer * n_total_layer
        n_selected = tf.shape(beam_indices)[0]
        W = tf.zeros([n_total_ae, n_selected], dtype=tf.complex64)
        
        # Build block-diagonal W
        i_grid = tf.range(n_selected, dtype=tf.int32)[:, None]
        j_grid = tf.range(n_ae_per_layer, dtype=tf.int32)[None, :]
        row_starts = i_grid * n_ae_per_layer
        row_indices = row_starts + j_grid
        col_indices = tf.tile(i_grid, [1, n_ae_per_layer])
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
                'Omega_ul': tf.Tensor [n_bs, n_ue] float32 (UPLINK, normalized),
                'H_mean_ul': tf.Tensor [n_bs, n_ue] complex64 (UPLINK, normalized),
                'pl_scalar': tf.Tensor [] float32,
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
        
        # Extract PL and normalize
        Omega_ul_norm_np, H_mean_ul_norm_np, pl_scalar = self._extract_pl_and_normalize(
            Omega_ul_np, H_mean_ul_np, U_bs_np.shape[0], U_ue_np.shape[0]
        )
        
        # Convert to TensorFlow tensors
        U_bs_tf = tf.constant(U_bs_np, dtype=tf.complex64)
        U_ue_tf = tf.constant(U_ue_np, dtype=tf.complex64)
        Omega_ul_tf = tf.constant(Omega_ul_norm_np, dtype=tf.float32)
        H_mean_ul_tf = tf.constant(H_mean_ul_norm_np, dtype=tf.complex64)
        pl_scalar_tf = tf.constant(pl_scalar, dtype=tf.float32)
        
        return {
            'U_bs': U_bs_tf,
            'U_ue': U_ue_tf,
            'Omega_ul': Omega_ul_tf,
            'H_mean_ul': H_mean_ul_tf,
            'pl_scalar': pl_scalar_tf,
            'channel_model': channel_model
        }
    
    def _extract_pl_and_normalize(self, Omega: np.ndarray, H_mean: np.ndarray,
                                  n_bs: int, n_ue: int) -> tuple:
        """Extract PL scalar and normalize channel to 0dB
        
        Returns:
            Omega_norm, H_mean_norm, pl_scalar
        """
        N_M = n_bs * n_ue
        epsilon = 1e-30
        
        P_nlos = np.sum(Omega)
        P_los = np.sum(np.abs(H_mean)**2)
        P_total = P_nlos + P_los
        
        # PL scalar = original power / N_M
        pl_scalar = P_total / (N_M + epsilon)
        
        # Normalize to 0dB (rho_target = 1.0)
        scale = N_M / (P_total + epsilon)
        Omega_norm = scale * Omega
        H_mean_norm = np.sqrt(scale) * H_mean
        
        return Omega_norm, H_mean_norm, float(pl_scalar)
    

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
                           Omega_beam: tf.Tensor, H_mean_beam: tf.Tensor,
                           power_budget: tf.Tensor) -> tuple:
        """Wen2011 ergodic capacity optimization
        
        Note: Parameters use _rx/_tx naming (abstracted roles).
              Caller should pass physical variables (U_bs, U_ue) in correct order:
              - UL: wen2011_optimize_tf(U_bs, U_ue, Omega_ul, H_mean_ul, power_budget)
              - DL: wen2011_optimize_tf(U_ue, U_bs, Omega_dl, H_mean_dl, power_budget)
        
        Args:
            U_rx_beam: [n_rx, n_rx] RX side eigenvectors
            U_tx_beam: [n_tx, n_tx] TX side eigenvectors
            Omega_beam: [n_rx, n_tx] coupling matrix [RX, TX] order
            H_mean_beam: [n_rx, n_tx] mean channel [RX, TX] order
            power_budget: TX power budget = n_layers × TxSNR_layer (scalar float32)
        
        Returns:
            (C_opt, P_opt, Lambda_P_r_final, k_final, eps_final)
                C_opt: Ergodic capacity
                P_opt: [n_tx, n_tx] Optimal TX covariance
                Lambda_P_r_final: [min_rank] Converged power eigenvalues
                k_final: Iteration count
                eps_final: Final relative change (convergence metric)
        """
        n_rx = tf.shape(U_rx_beam)[0]
        n_tx = tf.shape(U_tx_beam)[0]
        min_rank = tf.minimum(n_rx, n_tx)
        
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
        
        k_final, _, P_opt, Lambda_P_r_final, converged_final = tf.while_loop(
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
        return C_opt_tf, P_opt, Lambda_P_r_final, k_final, eps_final

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
            R: [n_AE, min_layer] complex64
            D: [min_layer, min_layer] complex64
        
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
            R: [n_AE, min_layer] residual
            D: [min_layer, min_layer] weight matrix
            F_codebook: [N_Layer_AE, N_Codebook]
            N_Layer: int
        
        Returns:
            j_best: int32 scalar
            R_next: [n_AE, min_layer] updated residual
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
                           N_Layer_AE: int, min_layer: int,
                           n_cand_max: int = None,
                           residual_threshold: float = None) -> tf.Tensor:
        """SWOMP candidate generation (tex Algorithm 1, Line 747-760)
        
        TensorFlow optimization: TensorArray + deferred .numpy()
        
        Args:
            V: [N_Layer*N_Layer_AE, min_layer] eigenvectors
            Lambda: [min_layer] eigenvalues
            F_codebook: [N_Layer_AE, N_Codebook]
            N_Layer, N_Layer_AE, min_layer: int
            n_cand_max: Max candidates (default: min_layer)
            residual_threshold: Stop if ||R||/||V|| < threshold (adaptive)
        
        Returns:
            candidates: [n_actual] int32 tensor (n_actual <= n_cand_max)
        
        Weighting Strategy:
            - eigenvalue weighting D = diag(√λ)
              (standard SWOMP, all iterations use eigenvalue weights)
        
        Adaptive Stopping (if residual_threshold provided and n_cand_max != min_layer):
            - Stop if ||R||/||V|| < threshold after iter min_layer
            - Only applies to extended search mode
        """
        if n_cand_max is None:
            n_cand_max = min_layer
        
        adaptive_mode = (n_cand_max != min_layer and residual_threshold is not None)
        
        # Weighting strategy: eigenvalue weighting (standard SWOMP)

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
            if adaptive_mode and i >= min_layer - 1:
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
            print(f"      SWOMP: min_layer={min_layer}, n_cand={n_actual}/{n_cand_max} (adaptive), N_Codebook={F_codebook.shape[1]}")
        else:
            print(f"      SWOMP: min_layer={min_layer}, n_cand={n_actual}, N_Codebook={F_codebook.shape[1]}")
        
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
    
    @tf.function
    def select_with_capacity_greedy(self, candidates: tf.Tensor, 
                                    F_codebook: tf.Tensor, 
                                    N_Layer: int, n_total_layer: int,
                                    capacity_evaluator_fn,
                                    record_history: bool,
                                    filter_top_min_layer: bool = False,
                                    min_layer_filter: int = None,
                                    n_cand_max: int = 32,
                                    min_layer: int = 4) -> tuple:
        """Capacity-based Greedy selection with eigenvalue tracking
        
        TensorFlow optimization: Full graph compilation with tf.while_loop
        
        Args:
            candidates: [n_actual] actual SWOMP candidates
            F_codebook: [N_Layer_AE, N_Codebook]
            N_Layer: Number of layers to select (int)
            n_total_layer: Total layer count (int)
            capacity_evaluator_fn: Callable[[W], tuple[tf.Tensor, tf.Tensor]]
                Returns: (capacity[], eigenvalues[min_layer] padded, actual size=min_rank)
            record_history: bool
            filter_top_min_layer: If True, filter to top min_layer after Layer 1
            min_layer_filter: min_layer value for filtering
            n_cand_max: Maximum candidate size for padding (default: 32)
            min_layer: Maximum eigenvalue size for padding (fixed, default: 4)
        
        Returns:
            beam_indices: [N_Layer] int32 tensor
            capacity_history: [N_Layer] float32 tensor (or None)
            eigenvalue_history: [N_Layer, min_layer] float32 tensor (0-padded)
        """
        # Set default for min_layer_filter if None
        if min_layer_filter is None:
            min_layer_filter = min_layer
        
        n_actual = tf.shape(candidates)[0]
        n_cand_max_tf = tf.constant(n_cand_max, dtype=tf.int32)
        
        # 고정 크기로 패딩
        padding_size = n_cand_max_tf - n_actual
        active_candidates = tf.concat([
            candidates,
            tf.fill([padding_size], -1)
        ], axis=0)
        n_active = n_actual
        
        # TensorArray 초기화
        selected_beams_ta = tf.TensorArray(dtype=tf.int32, size=N_Layer, dynamic_size=False)
        capacity_ta = tf.TensorArray(dtype=tf.float32, size=N_Layer, dynamic_size=False)
        eigenvalue_ta = tf.TensorArray(dtype=tf.float32, size=N_Layer, 
                                        dynamic_size=False, element_shape=[min_layer])
        
        # 레이어 루프
        def layer_body(i, beams_ta, cap_ta, eig_ta, active_cands, n_act):
            # 현재 빔 선택
            current_beams = tf.cond(
                tf.equal(i, 0),
                lambda: tf.constant([], dtype=tf.int32),
                lambda: beams_ta.gather(tf.range(i))
            )
            
            # 후보 평가 루프
            def cand_body(j, C_best, best_beam, best_lambda, C_list_ta):
                w = active_cands[j]
                is_valid = w >= 0
                
                def evaluate_candidate():
                    test_beams = tf.concat([current_beams, [w]], axis=0)
                    W_test = BeamformingMatrix.construct_beamforming_blockdiag_tf(
                        F_codebook, test_beams, n_total_layer)
                    return capacity_evaluator_fn(W_test)
                
                def skip_candidate():
                    return tf.constant(-1e9, dtype=tf.float32), tf.zeros([min_layer], dtype=tf.float32)
                
                C, Lambda = tf.cond(is_valid, evaluate_candidate, skip_candidate)
                C_list_ta = C_list_ta.write(j, C)
                
                should_update = tf.logical_and(is_valid, C > C_best)
                return (j + 1,
                        tf.where(should_update, C, C_best),
                        tf.where(should_update, w, best_beam),
                        tf.where(should_update, Lambda, best_lambda),
                        C_list_ta)
            
            _, C_best, best_beam, best_lambda, C_list_ta = tf.while_loop(
                lambda j, *_: j < n_cand_max_tf,
                cand_body,
                [0, tf.constant(-1e9, dtype=tf.float32), 
                 tf.constant(-1, dtype=tf.int32), 
                 tf.zeros([min_layer], dtype=tf.float32),
                 tf.TensorArray(dtype=tf.float32, size=n_cand_max, dynamic_size=False)]
            )
            
            # 저장
            beams_ta = beams_ta.write(i, best_beam)
            cap_ta = cap_ta.write(i, C_best)
            eig_ta = eig_ta.write(i, best_lambda)
            
            # 필터링
            C_list = C_list_ta.stack()
            def apply_filter():
                min_layer_filter_tf = tf.constant(min_layer_filter, dtype=tf.int32)
                sorted_indices = tf.argsort(C_list, direction='DESCENDING')
                top_indices = sorted_indices[:min_layer_filter]
                top_candidates = tf.gather(active_cands, top_indices)
                filter_padding_size = n_cand_max_tf - min_layer_filter_tf
                filtered = tf.concat([
                    top_candidates,
                    tf.fill([filter_padding_size], -1)
                ], axis=0)
                return filtered, min_layer_filter_tf
            
            def no_filter():
                return active_cands, n_act
            
            active_cands, n_act = tf.cond(
                tf.logical_and(tf.equal(i, 0), filter_top_min_layer),
                apply_filter,
                no_filter
            )
            
            # 로깅 (scientific notation for lambda only)
            i_str = tf.strings.as_string(i + 1)
            N_str = tf.strings.as_string(N_Layer)
            beam_str = tf.strings.as_string(best_beam)
            C_str = tf.strings.as_string(C_best, precision=2)  # 일반 포맷
            lambda_strs = tf.map_fn(
                lambda x: tf.strings.as_string(x, precision=2, scientific=True),  # scientific
                best_lambda,
                fn_output_signature=tf.string
            )
            lambda_str = tf.strings.reduce_join(lambda_strs, separator=", ")
            
            msg = tf.strings.join([
                "      Layer[", i_str, "/", N_str, "]: → b", beam_str,
                ", C=", C_str, ", λ=", lambda_str
            ], separator="")
            tf.print(msg)
            
            return i + 1, beams_ta, cap_ta, eig_ta, active_cands, n_act
        
        _, beams_ta, cap_ta, eig_ta, _, _ = tf.while_loop(
            lambda i, *_: i < N_Layer,
            layer_body,
            [0, selected_beams_ta, capacity_ta, eigenvalue_ta, active_candidates, n_active],
            shape_invariants=[
                tf.TensorShape([]),      # i
                tf.TensorShape(None),    # beams_ta
                tf.TensorShape(None),    # cap_ta
                tf.TensorShape(None),    # eig_ta
                tf.TensorShape([None]),  # active_candidates: 1D, 크기 변동 허용
                tf.TensorShape([])       # n_active
            ]
        )
        
        beam_indices = beams_ta.stack()
        capacity_history = cap_ta.stack() if record_history else None
        eigenvalue_history = eig_ta.stack()
        
        return beam_indices, capacity_history, eigenvalue_history

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
        
        # Generate UE Layer codebook
        self.F_ue_layer = DFTCodebook.generate_2d_dft_codebook_tf(
            self.config.n_ue_layer_row, 
            self.config.n_oversample
        )
    
    @tf.function
    def _evaluate_ue_capacity_tf(self, W_ue_test: tf.Tensor, U_bs: tf.Tensor, 
                                  U_ue: tf.Tensor, Omega_ul: tf.Tensor, 
                                  H_mean_ul: tf.Tensor, min_layer: int,
                                  power_budget_base: tf.Tensor) -> tuple:
        """UL capacity evaluation with eigenvalues (single Wen2011 call)
        
        SRP: UE 용량 + 고유값 평가 (중복 계산 방지)
        
        Args:
            W_ue_test: [n_ue, n_beams] UE beamforming matrix
            U_bs: [n_bs, n_bs] BS eigenvectors
            U_ue: [n_ue, n_ue] UE eigenvectors
            Omega_ul: [n_bs, n_ue] coupling matrix UPLINK
            H_mean_ul: [n_bs, n_ue] mean channel UPLINK
            min_layer: Maximum eigenvalue size for padding (fixed)
        
        Returns:
            tuple: (C, Lambda_P_r_padded)
                C: [] scalar capacity tensor
                Lambda_P_r_padded: [min_layer] eigenvalues tensor (0-padded from min_rank)
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
        
        # Dynamic power budget
        n_beams = tf.shape(W_ue_test)[1]
        # power_budget_base는 '수신 타겟 SNR' (총 n_ue_layer개 레이어 기준)
        power_per_layer = power_budget_base / tf.cast(self.config.n_ue_layer, tf.float32)
        power_budget = tf.cast(n_beams, tf.float32) * power_per_layer
        
        # Wen2011 capacity (UL: BS RX, UE TX) - SINGLE CALL
        C, P_opt, Lambda_P_r, _, _ = self.capacity_calc.wen2011_optimize_tf(
            U_bs_beam, U_ue_beam, Omega_beam_ul, H_mean_beam_ul, power_budget)
        
        # min_layer(고정) 크기로 패딩: min_rank → min_layer
        # min_rank = min(n_rx, n_tx): 실제 고유값 개수 (동적, 빔 선택 초기 1~4)
        # min_layer = 4: 패딩 타겟 크기 (고정)
        min_rank = tf.shape(Lambda_P_r)[0]
        min_layer_tf = tf.constant(min_layer, dtype=tf.int32)
        Lambda_P_r_padded = tf.concat([
            Lambda_P_r,
            tf.zeros(min_layer_tf - min_rank, dtype=tf.float32)
        ], axis=0)
        
        return C, Lambda_P_r_padded
    
    def select_ue_beams(self, U_bs: tf.Tensor, U_ue: tf.Tensor, 
                        Omega_ul: tf.Tensor, H_mean_ul: tf.Tensor,
                        power_budget_ul: tf.Tensor) -> dict:
        """Stage 1: Uplink UE TRX beam selection (v2 tex Section 6)
        
        TX/RX role: UE transmit [16] (optimize), BS receive [1024]
        Parameters: Omega_UL, H_mean_UL (P1I input)
        
        Args:
            U_bs: [1024, 1024] BS eigenvectors
            U_ue: [16, 16] UE eigenvectors
            Omega_ul: [1024, 16] coupling matrix [RX, TX] order, UPLINK
            H_mean_ul: [1024, 16] mean channel [RX, TX] order, UPLINK
        
        Returns:
            beam_indices: [n_ue_layer] selected UE TRX beam indices
            W_ue: [16, n_ue_layer] UE beamforming matrix
            C_ae: float, initial capacity (no beamforming)
            C_ue: float, final capacity (with beamforming)
            P_ue: [16, 16] UE transmit covariance
            Lambda_ue: [min_layer] top eigenvalues (0-padded, actual size=min_rank)
        """
        print("  Stage 1: UE beam selection (UPLINK)")
        
        # Step 1: Initial Wen2011 (no beamforming, UL) = C_AE
        # UL: BS RX [1024], UE TX [16]
        C_ae, P_ue, Lambda_P_ue, k_final, eps_final = self.capacity_calc.wen2011_optimize_tf(
            U_bs, U_ue, Omega_ul, H_mean_ul, power_budget_ul
        )
        tr_p = float(tf.math.real(tf.linalg.trace(P_ue)).numpy())
        print(f"    Wen2011 (UL, no BF): C_AE={C_ae.numpy():5.2f} bps/Hz, tr(P)={tr_p:.2e}, k={k_final.numpy()}, eps={eps_final.numpy():.2f}")
        
        # Step 2: Eigenvalue decomposition
        eigvals, eigvecs = tf.linalg.eigh(P_ue)
        eigvals = tf.math.real(eigvals)
        
        # Sort descending
        sorted_indices = tf.argsort(eigvals, direction='DESCENDING')
        eigvals_sorted = tf.gather(eigvals, sorted_indices)
        eigvecs_sorted = tf.gather(eigvecs, sorted_indices, axis=1)
        
        # Extract top min_layer eigenvectors
        Lambda_ue = eigvals_sorted[:self.config.min_layer]
        V_ue = eigvecs_sorted[:, :self.config.min_layer]  # [n_ue, min_layer]
        
        # Format eigenvalues (no normalization)
        eigvals_8 = eigvals_sorted[:8].numpy()
        eig_str = ', '.join(f"{v:.2e}" for v in eigvals_8)
        eig_sum = eigvals_8.sum()
        print(f"    EVD: top 8 eigenvalues (sum={eig_sum:.2e}): {eig_str}")
        
        # Step 3: SWOMP + Greedy
        if self.config.enable_swomp_stage1:
            print(f"    SWOMP Stage 1: Generate shared candidates")
            V_ue_ae = U_ue @ V_ue  # AE-space eigenvectors [16, min_layer]
            candidates = self.two_stage_selector.generate_candidates(
                V_ue_ae, Lambda_ue, self.F_ue_layer,
                self.config.n_ue_layer, self.config.n_ue_layer_ae, self.config.min_layer)
        else:
            print(f"    SWOMP Stage 1: Disabled (Full codebook)")
            candidates = tf.range(0, tf.shape(self.F_ue_layer)[1], dtype=tf.int32)
        
        print(f"    SWOMP Stage 2: Greedy capacity-based selection")
        beam_indices, _, _ = self.two_stage_selector.select_with_capacity_greedy(
            candidates, self.F_ue_layer,
            self.config.n_ue_layer, self.config.n_ue_layer,
            lambda W: self._evaluate_ue_capacity_tf(W, U_bs, U_ue, Omega_ul, H_mean_ul, 
                                                     self.config.min_layer, power_budget_ul),
            record_history=False,
            n_cand_max=self.config.n_ue_layer_beams,
            min_layer=self.config.min_layer)
        # eigenvalue_history는 사용 안함 (Stage1은 CSV 저장 없음)
        
        print(f"    Selected UE beams: {beam_indices.numpy()}")
        
        # Step 4: Construct beamforming matrix
        W_ue = BeamformingMatrix.construct_beamforming_blockdiag_tf(
            self.F_ue_layer, beam_indices, self.config.n_ue_layer
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
        
        # Final capacity
        C_ue_beam, P_ue_beam, Lambda_P_ue_beam, k_ue, eps_ue = self.capacity_calc.wen2011_optimize_tf(
            U_bs, U_ue_beam, Omega_beam_ul, H_mean_beam_ul, power_budget_ul
        )
        tr_p_beam = float(tf.math.real(tf.linalg.trace(P_ue_beam)).numpy())
        print(f"    Wen2011 (UL, with W_UE): C={C_ue_beam.numpy():5.2f} bps/Hz, tr(P)={tr_p_beam:.2e}, k={k_ue.numpy()}, eps={eps_ue.numpy():.2f}")
        
        # P_ue_beam eigenvalues
        eigvals_ue_beam = tf.math.real(tf.linalg.eigvalsh(P_ue_beam))
        eigvals_ue_beam_sorted = tf.sort(eigvals_ue_beam, direction='DESCENDING')
        eigvals_ue_beam_8 = eigvals_ue_beam_sorted[:8].numpy()
        eig_str_ue = ', '.join(f"{v:.2e}" for v in eigvals_ue_beam_8)
        eig_sum_ue = eigvals_ue_beam_8.sum()
        print(f"    P_UE eigenvalues (sum={eig_sum_ue:.2e}): {eig_str_ue}")
        
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
                                  H_mean_dl: tf.Tensor, W_ue: tf.Tensor, 
                                  min_layer: int, power_budget_base: tf.Tensor) -> tuple:
        """DL capacity evaluation with eigenvalues (single Wen2011 call)
        
        SRP: BS 용량 + 고유값 평가 (중복 계산 방지)
        
        Args:
            W_bs_test: [n_bs, n_beams] BS beamforming matrix
            U_bs: [n_bs, n_bs] BS eigenvectors
            U_ue: [n_ue, n_ue] UE eigenvectors
            Omega_dl: [n_ue, n_bs] coupling matrix DOWNLINK
            H_mean_dl: [n_ue, n_bs] mean channel DOWNLINK
            W_ue: [n_ue, n_ue_layer] UE beamforming (fixed from Stage 1)
            min_layer: Maximum eigenvalue size for padding (fixed)
        
        Returns:
            tuple: (C, Lambda_P_r_padded)
                C: [] scalar capacity tensor
                Lambda_P_r_padded: [min_layer] eigenvalues tensor (0-padded from min_rank)
        """
        # H_mean_beam_dl_n: [n_ue_layer, n] (RX, TX) order
        H_mean_beam_dl_n = tf.linalg.adjoint(W_ue) @ H_mean_dl @ W_bs_test
        
        # BS TX covariance (DL, n beams)
        _, U_bs_beam_n, _ = BeamDomainTransform.transform_covariance_dual(
            U_bs, U_ue, Omega_dl, W_bs_test, W_ue,
            link_direction="DL", target_side="BS")
        
        # UE RX covariance (DL, n_ue_layer beams fixed)
        _, U_ue_beam, _ = BeamDomainTransform.transform_covariance_dual(
            U_bs, U_ue, Omega_dl, W_bs_test, W_ue,
            link_direction="DL", target_side="UE")
        
        # Coupling matrix (DL)
        V_bs_n = tf.linalg.adjoint(U_bs) @ W_bs_test @ U_bs_beam_n
        V_ue = tf.linalg.adjoint(U_ue) @ W_ue @ U_ue_beam
        V_bs_abs2_n = tf.abs(V_bs_n)**2
        V_ue_abs2 = tf.abs(V_ue)**2
        
        # Omega_beam_dl_n: [n_ue_layer, n] (RX, TX) order
        Omega_beam_dl_n = tf.transpose(V_ue_abs2) @ Omega_dl @ V_bs_abs2_n
        
        # Dynamic power budget
        n_beams = tf.shape(W_bs_test)[1]
        # power_budget_base는 '수신 타겟 SNR' (총 n_bs_layer개 레이어 기준)
        power_per_layer = power_budget_base / tf.cast(self.config.n_bs_layer, tf.float32)
        power_budget = tf.cast(n_beams, tf.float32) * power_per_layer
        
        # Wen2011 capacity (DL: UE RX, BS TX) - SINGLE CALL
        C, P_opt, Lambda_P_r, _, _ = self.capacity_calc.wen2011_optimize_tf(
            U_ue_beam, U_bs_beam_n, Omega_beam_dl_n, H_mean_beam_dl_n, power_budget)
        
        # min_layer(고정) 크기로 패딩: min_rank → min_layer
        # min_rank = min(n_rx, n_tx): 실제 고유값 개수 (동적, 빔 선택 초기 1~4)
        # min_layer = 4: 패딩 타겟 크기 (고정)
        min_rank = tf.shape(Lambda_P_r)[0]
        min_layer_tf = tf.constant(min_layer, dtype=tf.int32)
        Lambda_P_r_padded = tf.concat([
            Lambda_P_r,
            tf.zeros(min_layer_tf - min_rank, dtype=tf.float32)
        ], axis=0)
        
        return C, Lambda_P_r_padded
    
    @staticmethod
    def _apply_perturbation_to_eigenvectors(V: tf.Tensor, alpha: float, 
                                            seed: int) -> tf.Tensor:
        """Apply perturbation to eigenvectors
        
        SRP: 단일 책임 - 고유벡터 perturbation 변환
        
        V_pert = alpha*V + (1-alpha)*N, normalize columns
        
        Args:
            V: [n, r] complex64 eigenvectors
            alpha: Perturbation coefficient (0.95)
            seed: Random seed
        
        Returns:
            V_pert: [n, r] complex64 perturbed eigenvectors
        """
        n = tf.shape(V)[0]
        r = tf.shape(V)[1]
        
        # Complex Gaussian N ~ CN(0, I)
        N_real = tf.random.normal([n, r], seed=seed, dtype=tf.float32)
        N_imag = tf.random.normal([n, r], seed=seed+1000, dtype=tf.float32)
        sqrt_2_complex = tf.complex(tf.sqrt(2.0), 0.0)
        N = tf.complex(N_real, N_imag) / sqrt_2_complex
        
        # Perturbation
        alpha_complex = tf.cast(alpha, tf.complex64)
        beta_complex = tf.cast(1.0 - alpha, tf.complex64)
        V_pert = alpha_complex * V + beta_complex * N
        
        # Normalize columns (each column to unit norm)
        norms = tf.sqrt(tf.reduce_sum(tf.abs(V_pert)**2, axis=0, keepdims=True))
        V_pert = V_pert / tf.complex(norms + 1e-10, 0.0)
        
        return V_pert
    
    def _run_single_trial(self, candidates: tf.Tensor, n_layers: int,
                          U_bs: tf.Tensor, U_ue: tf.Tensor, 
                          Omega_dl: tf.Tensor, H_mean_dl: tf.Tensor,
                          W_ue: tf.Tensor, power_budget_dl: tf.Tensor) -> dict:
        """Run single trial: greedy selection (eigenvalues included)
        
        SRP: 단일 책임 - 하나의 trial 실행
        
        Args:
            candidates: [n_cand] SWOMP candidates
            n_layers: Number of layers to select
            ... (channel parameters)
        
        Returns:
            dict: {
                'beam_indices': [n_layers],
                'C_hist': [n_layers],
                'lambda_per_layer': [[1~min_layer], ...] variable length per layer
            }
        """
        # Greedy selection (eigenvalues included in return)
        beam_indices, C_hist, eigenvalue_history = self.two_stage_selector.select_with_capacity_greedy(
            candidates, self.F_bs_layer, 
            n_layers,                    # N_Layer: 8 (pert) or 64 (normal)
            self.config.n_bs_layer,      # n_total_layer: 64 (고정)
            lambda W: self._evaluate_bs_capacity_tf(W, U_bs, U_ue, Omega_dl, H_mean_dl, W_ue,
                                                     self.config.min_layer, power_budget_dl),
            record_history=True, 
            filter_top_min_layer=True,  # 항상 활성화
            min_layer_filter=self.config.min_layer,  # 항상 min_layer 사용
            n_cand_max=self.config.r_bs_cand_max,
            min_layer=self.config.min_layer)
        
        # 0 패딩 제거
        lambda_per_layer = [
            [float(v) for v in eigs.numpy() if v > 1e-6]
            for eigs in eigenvalue_history
        ]
        
        return {
            'beam_indices': beam_indices,
            'C_hist': C_hist.numpy().tolist(),
            'lambda_per_layer': lambda_per_layer
        }
    
    def _print_perturbation_comparison(self, results: list) -> None:
        """Print perturbation test comparison table
        
        SRP: 단일 책임 - 결과 출력
        
        Args:
            results: List of trial results
        """
        n_layers = len(results[0]['C_hist'])
        
        print(f"\n    {'='*65}")
        print(f"    Perturbation Test ({len(results)} trials)")
        print(f"    {'='*65}")
        print(f"      Layer | Baseline | Perturbed (min/mean/max) | Delta")
        print(f"      {'-'*65}")
        
        for i in range(n_layers):
            baseline_C = results[0]['C_hist'][i]
            pert_Cs = [r['C_hist'][i] for r in results[1:]]
            pert_min = min(pert_Cs)
            pert_mean = np.mean(pert_Cs)
            pert_max = max(pert_Cs)
            delta = pert_mean - baseline_C
            
            print(f"        {i+1:2d}  | {baseline_C:6.2f}   | "
                  f"{pert_min:6.2f}/{pert_mean:6.2f}/{pert_max:6.2f} | {delta:+6.2f}")
        
        print(f"    {'='*65}\n")
    
    def _run_perturbation_test(self, U_bs: tf.Tensor, U_ue: tf.Tensor, 
                               Omega_dl: tf.Tensor, H_mean_dl: tf.Tensor,
                               W_ue: tf.Tensor, V_bs: tf.Tensor, 
                               Lambda_bs: tf.Tensor, 
                               candidates_baseline: tf.Tensor,
                               power_budget_dl: tf.Tensor) -> dict:
        """Run perturbation test: orchestrate multiple trials
        
        SRP: 단일 책임 - perturbation 테스트 흐름 관리
        
        Args:
            ... (channel parameters)
            V_bs: [n_bs, min_layer] original eigenvectors for perturbation
            Lambda_bs: [n_bs_layer] eigenvalues
            candidates_baseline: [n_cand] baseline SWOMP candidates
        
        Returns:
            dict: Baseline result with lambda_BS_per_layer
        """
        n_test_layers = self.config.perturbation_n_layers
        n_trials = self.config.perturbation_n_trials
        alpha = self.config.perturbation_alpha
        
        results = []
        
        # Trial 0: Baseline
        print(f"    === Trial 0 (Baseline) ===")
        baseline = self._run_single_trial(
            candidates_baseline, n_test_layers, 
            U_bs, U_ue, Omega_dl, H_mean_dl, W_ue, power_budget_dl)
        results.append(baseline)
        
        # Trials 1~N: Perturbed
        for trial_idx in range(1, n_trials + 1):
            print(f"    === Trial {trial_idx} (α={alpha}) ===")
            
            # Perturb eigenvectors
            V_bs_pert = self._apply_perturbation_to_eigenvectors(
                V_bs, alpha, seed=1000 * trial_idx)
            V_bs_ae_pert = U_bs @ V_bs_pert
            
            # Generate perturbed candidates
            candidates_pert = self.two_stage_selector.generate_candidates(
                V_bs_ae_pert, Lambda_bs[:self.config.min_layer], 
                self.F_bs_layer, self.config.n_bs_layer, 
                self.config.n_bs_layer_ae, self.config.min_layer)
            
            # Run trial
            pert_result = self._run_single_trial(
                candidates_pert, n_test_layers, 
                U_bs, U_ue, Omega_dl, H_mean_dl, W_ue, power_budget_dl)
            results.append(pert_result)
        
        # Print comparison
        self._print_perturbation_comparison(results)
        
        # Return all trial results
        return {
            'all_trials': results,  # 모든 trial 결과 리스트
            'Lambda_bs': Lambda_bs
        }
    
    def select_bs_beams(self, U_bs: tf.Tensor, U_ue: tf.Tensor, 
                        Omega_ul: tf.Tensor, H_mean_ul: tf.Tensor,
                        W_ue: tf.Tensor, power_budget_dl: tf.Tensor) -> dict:
        """Stage 2: Downlink BS Layer beam selection (v2 tex Section 7)
        
        TX/RX: BS transmit [n_bs_ae], UE receive [n_ue_layer] (W_ue fixed)
        Parameters: Omega_DL = Omega_UL^T, H_mean_DL = H_mean_UL^H (v2 tex Line 1035)
        
        Args:
            U_bs: [1024, 1024] BS eigenvectors
            U_ue: [16, 16] UE eigenvectors (full AE)
            Omega_ul: [1024, 16] coupling matrix UPLINK
            H_mean_ul: [1024, 16] mean channel UPLINK
            W_ue: [16, 4] UE beamforming from Stage 1 (fixed)
        
        Returns:
            dict: {
                'beam_indices': [N] BS Layer beam order
                'C_bs_hist': list[N] DL capacity history
                'Lambda_bs': [64] top eigenvalues
                'lambda_BS_per_layer': [[1~4], [1~4], ...] eigenvalues per layer (variable length)
            }
            where N = perturbation_n_layers (8) if enable_perturbation=True,
                  else n_bs_layer (64)
        
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
        
        # Omega_beam_dl: [n_ue_layer, n_bs_ae] (RX, TX) order
        Omega_beam_dl = tf.transpose(V_ue_abs2) @ Omega_dl @ V_bs_abs2_eye
        
        # Step 2: Wen2011 (DL BS TX optimization)
        # DL: UE RX [n_ue_layer], BS TX [n_bs_ae]
        C_bs_full, P_bs, Lambda_P_bs, k_final, eps_final = self.capacity_calc.wen2011_optimize_tf(
            U_ue_beam, U_bs_beam_eye, Omega_beam_dl, H_mean_beam_dl, power_budget_dl
        )
        tr_p = float(tf.math.real(tf.linalg.trace(P_bs)).numpy())
        print(f"    Wen2011 (DL, full BS): C={C_bs_full.numpy():5.2f} bps/Hz, tr(P)={tr_p:.2e}, k={k_final.numpy()}, eps={eps_final.numpy():.2f}")
        
        # Step 3: Eigendecomposition
        eigvals, eigvecs = tf.linalg.eigh(P_bs)
        eigvals = tf.math.real(eigvals)
        sorted_indices = tf.argsort(eigvals, direction='DESCENDING')
        eigvals_sorted = tf.gather(eigvals, sorted_indices)
        eigvecs_sorted = tf.gather(eigvecs, sorted_indices, axis=1)
        
        Lambda_bs = eigvals_sorted[:self.config.n_bs_layer]
        V_bs = eigvecs_sorted[:, :self.config.min_layer]  # Top min_layer for Two-Stage
        V_bs_ae = U_bs @ V_bs  # AE-space
        
        eigvals_8 = eigvals_sorted[:8].numpy()
        eig_str = ', '.join(f"{v:.2e}" for v in eigvals_8)
        eig_sum_bs = eigvals_8.sum()
        print(f"    P_BS eigenvalues (sum={eig_sum_bs:.2e}): {eig_str}")
        
        # Step 4: SWOMP + Greedy
        if self.config.enable_swomp_stage2:
            print(f"    SWOMP Stage 1: Generate shared candidates")
            candidates = self.two_stage_selector.generate_candidates(
                V_bs_ae, Lambda_bs[:self.config.min_layer], self.F_bs_layer,
                self.config.n_bs_layer, self.config.n_bs_layer_ae, self.config.min_layer,
                n_cand_max=self.config.r_bs_cand_max,
                residual_threshold=self.config.r_bs_residual_threshold)
        else:
            print(f"    SWOMP Stage 1: Disabled (Full codebook)")
            candidates = tf.range(0, tf.shape(self.F_bs_layer)[1], dtype=tf.int32)
        
        if self.config.enable_perturbation:
            # Perturbation test mode (8 layers)
            return self._run_perturbation_test(
                U_bs, U_ue, Omega_dl, H_mean_dl, W_ue, 
                V_bs, Lambda_bs, candidates, power_budget_dl)
        else:
            # Normal mode (64 layers)
            print(f"    SWOMP Stage 2: Greedy capacity-based selection")
            result = self._run_single_trial(
                candidates, self.config.n_bs_layer,
                U_bs, U_ue, Omega_dl, H_mean_dl, W_ue, power_budget_dl)
            
            # Print beam usage statistics
            beams_np = result['beam_indices'].numpy()
            unique_beams, counts = np.unique(beams_np, return_counts=True)
            sorted_idx = np.argsort(-counts)
            top_beams = unique_beams[sorted_idx][:16]
            top_counts = counts[sorted_idx][:16]
            beam_freq_str = ', '.join(f"b{b}({c})" for b, c in zip(top_beams, top_counts))
            print(f"    Beam usage: {beam_freq_str}")
            print(f"    Final capacity: {result['C_hist'][-1]:.2f} bps/Hz")
            
            return {
                'beam_indices': result['beam_indices'],
                'C_bs_hist': result['C_hist'],
                'Lambda_bs': Lambda_bs,
                'lambda_BS_per_layer': result['lambda_per_layer']
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
        self._processed_ues = set()  # Resume 모드에서 처리 완료된 UE 목록
        
        # Resume 모드 초기화
        if config.enable_resume and config.resume_from_dir:
            self._init_resume()
    
    def _init_csv(self, area: int, freq: float):
        """Initialize CSV file"""
        if self._csv_path is not None:
            return
        
        if self._csv_timestamp is None:
            utc_plus_9 = datetime.utcnow() + timedelta(hours=9)
            self._csv_timestamp = utc_plus_9.strftime('%m%d_%H%M')
        
        # Mode suffix
        mode_suffix = "_PERT" if self.config.enable_perturbation else ""
        
        # Frequency formatting: 7.5 → 7_5
        freq_str = str(freq).replace('.', '_')
        
        if self.config.partition_id is not None:
            csv_filename = f"A{area}_{freq_str}GHz_P1P{mode_suffix}_{self._csv_timestamp}_p{self.config.partition_id}.csv"
        else:
            csv_filename = f"A{area}_{freq_str}GHz_P1P{mode_suffix}_{self._csv_timestamp}.csv"
        self._csv_path = os.path.join(self.config.P1P_OUTPUT_DIR, csv_filename)
    
    def save_result(self, area: int, freq: float, ue: int, 
                    C_AE: float, C_UE: float,
                    lambda_UE: list, lambda_BS_per_layer: list,
                    ue_beams: list, bs_beams: list,
                    C_BS_hist: list,
                    trial_id: int = None,
                    lambda_BS: list = None):
        """Save P1P result to CSV (comma-separated format)
        
        Args:
            area, freq, ue: UE identifier
            C_AE: Full digital capacity (reference)
            C_UE: Stage 1 UE capacity
            lambda_UE: UE eigenvalues (r_ue values)
            lambda_BS_per_layer: BS eigenvalues per layer (variable length, 1~min_rank per layer)
            ue_beams: UE beam indices (r_ue values)
            bs_beams: BS beam order (n_bs_layers values)
            C_BS_hist: Stage 2 capacity history (n_bs_layers values)
            trial_id: Trial ID for perturbation mode (None=normal, 0~N=perturbation)
            lambda_BS: BS eigenvalues [64] for perturbation mode (optional)
        
        CSV format: "1,2,3,..." (comma-separated), scientific notation for eigenvalues
        """
        if self._csv_path is None:
            self._init_csv(area, freq)
        
        # Perturbation 모드일 때만 trial_id, lambda_BS 컬럼 추가
        if self.config.enable_perturbation:
            headers = ['ue', 'trial_id', 'C_AE', 'C_UE', 'C_BS_hist', 'lambda_UE', 
                       'lambda_BS', 'ue_beams', 'lambda_BS_per_layer', 'bs_beams']
        else:
            headers = ['ue', 'C_AE', 'C_UE', 'C_BS_hist', 'lambda_UE', 'ue_beams', 
                       'lambda_BS_per_layer', 'bs_beams']
        
        file_exists = os.path.exists(self._csv_path)
        
        # Format: scientific notation for eigenvalues
        lambda_UE_str = ','.join(f"{v:.2e}" for v in lambda_UE)
        lambda_BS_per_layer_str = '; '.join(
            ','.join(f"{v:.2e}" for v in layer_eigs)
            for layer_eigs in lambda_BS_per_layer
        )
        ue_beams_str = ','.join(str(b) for b in ue_beams)
        bs_beams_str = ','.join(str(b) for b in bs_beams)
        C_BS_hist_str = ','.join(f"{c:.2f}" for c in C_BS_hist)
        
        # Lambda_BS formatting (perturbation mode only)
        lambda_BS_str = None
        if self.config.enable_perturbation and lambda_BS is not None:
            lambda_BS_str = ','.join(f"{v:.2e}" for v in lambda_BS)
        
        with open(self._csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(headers)
            
            # Perturbation 모드일 때 trial_id, lambda_BS 포함
            if self.config.enable_perturbation:
                row = [
                    ue,
                    trial_id if trial_id is not None else 0,
                    f"{C_AE:.2f}",
                    f"{C_UE:.2f}",
                    C_BS_hist_str,
                    lambda_UE_str,
                    lambda_BS_str if lambda_BS_str is not None else '',
                    ue_beams_str,
                    lambda_BS_per_layer_str,
                    bs_beams_str
                ]
            else:
                row = [
                    ue,
                    f"{C_AE:.2f}",
                    f"{C_UE:.2f}",
                    C_BS_hist_str,
                    lambda_UE_str,
                    ue_beams_str,
                    lambda_BS_per_layer_str,
                    bs_beams_str
                ]
            writer.writerow(row)
        
        print(f"  Saved to: {self._csv_path}")
    
    def _init_resume(self):
        """Resume 모드 초기화: 기존 CSV 찾기, 처리된 UE 추출, 새 타임스탬프로 복사
        
        작업 순서:
        1. resume_from_dir에서 현재 파티션의 기존 CSV 파일 찾기
        2. CSV에서 처리 완료된 UE 목록 추출 → self._processed_ues
        3. 새 타임스탬프 생성
        4. 기존 CSV 내용을 새 타임스탬프로 복사
        5. self._csv_path 설정 (이후 save_result()에서 append 모드 사용)
        
        에러 처리:
        - resume_from_dir 없으면 경고, 새 실행으로 전환
        - 파티션 번호 불일치 시 경고
        - CSV 없으면 경고, 빈 processed_ues로 시작
        """
        if self.config.partition_id is None:
            print("WARNING: Resume mode requires partition_id. Continuing with new execution...")
            return
        
        # Step 1: Resume 디렉토리 경로 검증
        resume_dir_path = os.path.join(self.config.P1P_OUTPUT_DIR, self.config.resume_from_dir)
        if not os.path.exists(resume_dir_path):
            print(f"WARNING: Resume directory not found: {self.config.resume_from_dir}")
            print("Continuing with new execution...")
            return
        
        # Step 2: 기존 CSV 파일 찾기 (target_areas, target_freqs 기준)
        # 패턴: A{area}_{freq}_P1P_*_p{partition}.csv
        mode_suffix = "_PERT" if self.config.enable_perturbation else ""
        existing_csv_path = None
        
        for area in self.config.target_areas:
            for freq in self.config.target_freqs:
                freq_str = str(freq).replace('.', '_')
                pattern = os.path.join(
                    self.config.P1P_OUTPUT_DIR,
                    f"A{area}_{freq_str}GHz_P1P{mode_suffix}_*_p{self.config.partition_id}.csv"
                )
                matching_files = glob.glob(pattern)
                
                if matching_files:
                    # 최신 파일 선택 (수정 시간 기준)
                    existing_csv_path = max(matching_files, key=os.path.getmtime)
                    break
            
            if existing_csv_path:
                break
        
        if existing_csv_path is None:
            print(f"WARNING: No existing CSV found for partition {self.config.partition_id}")
            print("Starting fresh for this partition...")
            return
        
        # Step 3: CSV에서 처리 완료된 UE 목록 추출
        try:
            with open(existing_csv_path, 'r') as f:
                reader = csv.DictReader(f)
                self._processed_ues = {int(row['ue']) for row in reader}
            
            print(f"Resume: Found {len(self._processed_ues)} processed UEs in {os.path.basename(existing_csv_path)}")
        except Exception as e:
            print(f"WARNING: Failed to read existing CSV: {e}")
            print("Starting fresh for this partition...")
            return
        
        # Step 4: 새 타임스탬프 생성
        utc_plus_9 = datetime.utcnow() + timedelta(hours=9)
        self._csv_timestamp = utc_plus_9.strftime('%m%d_%H%M')
        
        # Step 5: 새 CSV 파일 경로 생성 및 기존 CSV 복사
        area = self.config.target_areas[0]
        freq = self.config.target_freqs[0]
        freq_str = str(freq).replace('.', '_')
        
        csv_filename = f"A{area}_{freq_str}GHz_P1P{mode_suffix}_{self._csv_timestamp}_p{self.config.partition_id}.csv"
        new_csv_path = os.path.join(self.config.P1P_OUTPUT_DIR, csv_filename)
        
        # 기존 CSV 내용을 새 CSV로 복사
        try:
            shutil.copy2(existing_csv_path, new_csv_path)
            self._csv_path = new_csv_path
            print(f"Resume: Copied existing CSV to {os.path.basename(new_csv_path)}")
        except Exception as e:
            print(f"WARNING: Failed to copy CSV: {e}")
            print("Starting fresh for this partition...")
            self._processed_ues = set()
            return
    
    def filter_resume_ue_list(self, ue_list: list) -> tuple:
        """Resume 모드에서 UE 리스트 필터링 및 통계 반환
        
        Args:
            ue_list: 원본 UE 리스트 [(area, freq, ue), ...]
            
        Returns:
            tuple: (filtered_ue_list, skipped_count)
                - filtered_ue_list: 미처리 UE만 포함한 리스트
                - skipped_count: 건너뛴 UE 수
        """
        if not self.config.enable_resume or not self.config.resume_from_dir:
            return ue_list, 0
        
        original_count = len(ue_list)
        filtered_list = [(a, f, u) for a, f, u in ue_list 
                         if u not in self._processed_ues]
        skipped_count = original_count - len(filtered_list)
        
        return filtered_list, skipped_count

# ===== LEVEL 3: Main Execution =====

def main():
    """P1P main pipeline"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='P1P: Bidirectional Wen2011 + SWOMP Beam Selection')
    parser.add_argument('--partition', type=int, default=None, help='Partition index for parallel execution')
    parser.add_argument('--ue-indices', type=str, default=None, help='Comma-separated UE indices to process')
    parser.add_argument('--resume-from', type=str, default=None,
                        help='Resume directory name (e.g., logs_20251115_065837). '
                             'Overrides config.resume_from_dir if specified.')
    args = parser.parse_args()
    
    print("=" * 80)
    print("P1P: Bidirectional Wen2011 + SWOMP Beam Selection")
    print("=" * 80)
    
    # Initialize
    config = P1P_Config()
    
    # Store partition ID in config
    config.partition_id = args.partition
    
    # Resume 옵션: CLI 인자가 Config 설정을 override
    if args.resume_from:
        config.enable_resume = True
        config.resume_from_dir = args.resume_from
        print(f"Resume mode enabled via CLI: {args.resume_from}")
    elif config.enable_resume and config.resume_from_dir:
        print(f"Resume mode enabled via Config: {config.resume_from_dir}")
    
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
            n_partitions = 16  # Fixed to 16 partitions (2GB per process, 10GB reserved)
            
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
    
    # Timing for ETA
    start_time = time.time()
    start_dt = datetime.now()
    print(f"\nExecution started: {start_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # DataLoader, Selectors 초기화
    data_loader = DataLoader(config)
    stage1_selector = Stage1_UE_BeamSelector(config)
    stage2_selector = Stage2_BS_BeamSelector(config)
    
    # ResultManager 초기화 (Resume 로직 자동 실행)
    result_manager = P1P_ResultManager(config)
    
    # Resume 모드: UE 리스트 필터링
    config.ue_list, skipped_count = result_manager.filter_resume_ue_list(config.ue_list)
    
    if config.enable_resume and config.resume_from_dir and skipped_count > 0:
        print(f"\nResume Mode Summary:")
        print(f"  Source directory: {config.resume_from_dir}")
        print(f"  Partition {config.partition_id}: {skipped_count} UEs already processed")
        print(f"  Remaining: {len(config.ue_list)} UEs to process")
        print(f"  CSV: Results will be appended to new timestamped file")
    
    if len(config.ue_list) == 0:
        if config.enable_resume:
            print("All UEs already processed in this partition. Exiting.")
        else:
            print("No UEs to process. Exiting.")
        return
    
    print(f"\nConfiguration:")
    print(f"  UE: {config.n_ue_trx} TRX, {config.n_ue_ae} AE")
    print(f"  BS: {config.n_bs_layer} Layers, {config.n_bs_ae} AE")
    print(f"  Stage 1: Select {config.min_layer} UE beams from {config.n_ue_layer_beams}-beam codebook")
    print(f"  Stage 2: Select {config.n_bs_layer} BS beams from {config.n_bs_layer_beams}-beam codebook")
    print(f"  Total UEs: {len(config.ue_list)}")
    
    # Perturbation mode warning
    if config.enable_perturbation:
        print(f"\n{'='*80}")
        print(f"WARNING: Perturbation Test Mode Enabled")
        print(f"{'='*80}")
        print(f"  Stage 2 configuration:")
        print(f"    - Testing only FIRST {config.perturbation_n_layers} layers (not full {config.n_bs_layer})")
        print(f"    - Running {config.perturbation_n_trials + 1} trials:")
        print(f"      * 1 baseline (original eigenvectors)")
        print(f"      * {config.perturbation_n_trials} perturbed (α={config.perturbation_alpha})")
        print(f"    - Results saved with '_PERT' suffix")
        print(f"  Purpose: Validate SWOMP candidate selection quality")
        print(f"{'='*80}\n")
    
    # Test mode warning
    if config.test_mode:
        print(f"\n{'='*80}")
        print(f"WARNING: Test Mode Enabled")
        print(f"{'='*80}")
        print(f"  Processing only {len(config.test_ue_list)} UEs: {config.test_ue_list}")
        print(f"  Actual UEs to process: {len(config.ue_list)}")
        print(f"{'='*80}\n")
    
    # Process each UE
    for idx, (area, freq, ue) in enumerate(config.ue_list):
        print(f"\n[{idx+1}/{len(config.ue_list)}] Area{area}_{freq}GHz UE{ue}")
        
        # ETA calculation (매 10 UE마다 출력)
        if idx > 0 and (idx % 10 == 0 or idx == len(config.ue_list) - 1):
            elapsed = time.time() - start_time
            avg_time_per_ue = elapsed / (idx + 1)
            remaining_ues = len(config.ue_list) - (idx + 1)
            eta_seconds = remaining_ues * avg_time_per_ue
            eta_dt = datetime.now() + timedelta(seconds=eta_seconds)
            print(f"  [{elapsed/60:.1f} min elapsed, {avg_time_per_ue/60:.1f} min/UE, "
                  f"ETA: {eta_dt.strftime('%m-%d %H:%M')}]")
        
        try:
            # Load P1I (normalized channel + pl_scalar_tf)
            p1i_params = data_loader.load_p1i_channel_params(area, freq, ue)
            U_bs = p1i_params['U_bs']
            U_ue = p1i_params['U_ue']
            Omega_ul_norm = p1i_params['Omega_ul']
            H_mean_ul_norm = p1i_params['H_mean_ul']
            pl_scalar_tf = p1i_params['pl_scalar']
            print(f"  P1I: channel_model={p1i_params['channel_model']}")
            
            # Rx Target SNR (power_budget)
            TxSNR_ue_tf = tf.constant(config.TxSNR_ue, dtype=tf.float32)
            TxSNR_bs_tf = tf.constant(config.TxSNR_bs, dtype=tf.float32)
            power_budget_ul = TxSNR_ue_tf * pl_scalar_tf
            power_budget_dl = TxSNR_bs_tf * pl_scalar_tf
            
            # Logging (convert to numpy for display)
            pl_scalar_np = pl_scalar_tf.numpy()
            pl_db = 10 * np.log10(pl_scalar_np)
            rx_snr_ul_db = 10 * np.log10(power_budget_ul.numpy())
            rx_snr_dl_db = 10 * np.log10(power_budget_dl.numpy())
            print(f"  PL: {pl_db:.1f} dB")
            print(f"  Rx Target SNR (UL): {rx_snr_ul_db:.1f} dB")
            print(f"  Rx Target SNR (DL): {rx_snr_dl_db:.1f} dB")
            
            # Stage 1: UE beam selection
            stage1_result = stage1_selector.select_ue_beams(
                U_bs, U_ue, Omega_ul_norm, H_mean_ul_norm, power_budget_ul)
            C_AE = stage1_result['C_ae']
            C_UE = stage1_result['C_ue']
            W_ue = stage1_result['W_ue']
            ue_beams = stage1_result['beam_indices'].numpy().tolist()
            lambda_UE = stage1_result['Lambda_ue'].numpy().tolist()
            
            # Stage 2: BS beam selection
            stage2_result = stage2_selector.select_bs_beams(
                U_bs, U_ue, Omega_ul_norm, H_mean_ul_norm, W_ue, power_budget_dl)
            
            # Save result
            if config.enable_perturbation:
                # Perturbation mode: save all trials
                all_trials = stage2_result['all_trials']
                Lambda_bs = stage2_result['Lambda_bs'].numpy().tolist()
                
                for trial_idx, trial_result in enumerate(all_trials):
                    bs_beams = trial_result['beam_indices'].numpy().tolist()
                    C_BS_hist = trial_result['C_hist']
                    lambda_BS_per_layer = trial_result['lambda_per_layer']
                    
                    result_manager.save_result(
                        area, freq, ue, C_AE, C_UE,
                        lambda_UE, lambda_BS_per_layer,
                        ue_beams, bs_beams, C_BS_hist,
                        trial_id=trial_idx,
                        lambda_BS=Lambda_bs
                    )
            else:
                # Normal mode: save single result
                bs_beams = stage2_result['beam_indices'].numpy().tolist()
                C_BS_hist = stage2_result['C_bs_hist']
                lambda_BS_per_layer = stage2_result['lambda_BS_per_layer']
                
                result_manager.save_result(
                    area, freq, ue, C_AE, C_UE,
                    lambda_UE, lambda_BS_per_layer,
                    ue_beams, bs_beams, C_BS_hist
                )
            
            # Summary 출력
            if config.enable_perturbation:
                all_trials = stage2_result['all_trials']
                baseline_C = all_trials[0]['C_hist'][-1]
                print(f"  Summary ({config.perturbation_n_layers} layers, {len(all_trials)} trials): "
                      f"C_AE(UL)={C_AE:5.2f} bps/Hz, C_UE(UL)={C_UE:5.2f} bps/Hz, "
                      f"C_BS(DL, baseline)={baseline_C:5.2f} bps/Hz")
            else:
                C_BS_hist = stage2_result['C_bs_hist']
                print(f"  Summary: C_AE(UL)={C_AE:5.2f} bps/Hz, C_UE(UL)={C_UE:5.2f} bps/Hz, "
                      f"C_BS(DL)={C_BS_hist[-1]:5.2f} bps/Hz")
            
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    total_elapsed = time.time() - start_time
    end_dt = datetime.now()
    avg_time = total_elapsed / len(config.ue_list)
    
    print(f"\n{'=' * 80}")
    print("P1P pipeline completed")
    print(f"{'=' * 80}")
    print(f"Start: {start_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End: {end_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total: {total_elapsed/3600:.1f} hours ({total_elapsed/60:.0f} minutes)")
    print(f"Average: {avg_time/60:.1f} minutes per UE")
    print(f"Processed: {len(config.ue_list)} UEs")
    print(f"{'=' * 80}")

if __name__ == "__main__":
    main()


