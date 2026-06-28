#!/usr/bin/env python3
"""
P1O_Replot_Polar_v2.py

Multi-layer beam pattern polar plot with SOLID principles
- Each layer-beam combination plotted in separate subplot (26 subplots for UE 16)
- Last subplot shows total (sum of all layer-beams)
- Detailed per-subplot verification info output

=== Core Principles (NEVER FORGET) ===

1. P1O Capacity Optimization (가장 중요!)
   - Low capacity loss (0.34%) → Good PADP alignment
   - P1O selects beams that align with PADP direction
   - Peak direction ≈ PADP direction (verified fact)

2. P1O Layer Structure (blkdiag) - CRITICAL!
   - 64 layers, each layer = SPECIFIC 16 AE positions in 1024 AE array
   - Layer 63 = AE [1008:1024] = Panel (7,7) → physical position (y: 17.5~19.0λ, z: 17.5~19.0λ)
   - Layer 0 = AE [0:16] = Panel (0,0) → physical position (y: 0~1.5λ, z: 0~1.5λ)
   - NOT independent 4x4 subarrays - they are embedded in full 1024 AE array

3. Beam Pattern Calculation (MUST use actual positions!)
   - DFT codeword w: non-centered [0,1,2,3] for 4x4 DFT
   - Test steering vector a: MUST use Layer's actual 1024 AE physical positions
   - Example: Layer 63 AE 1008 at (17.5λ, 17.5λ), NOT at (0, 0)
   - Verified in verify_physical_array.py: Layer 63 gain=14.75 at PADP (Rank 2/64)

4. Reference Implementation
   - verify_physical_array.py: Ground truth verification
   - Line 85-109: generate_1024_steering_vector() - uses actual AE positions
   - Line 155-193: calculate_layer_gains_at_padp() - extracts layer's 16 AE from 1024
   - Result: Layer 63 has high gain (14.75) at PADP, confirming P1O selection

5. CSV Parsing (DO NOT FORGET!)
   - CSV bs_beams_abm: "63,63,63,56,56,..." (sequence)
   - Meaning: Layer 0→Beam 63, Layer 1→Beam 63, Layer 5→Beam 56, ...
   - P1O selection is CORRECT (considers physical AE positions)

6. Systematic Debugging Principles
   - Complexity reduction: Simple individual plots > Complex overlay
   - Verification before summarization: Validate each subplot before total
   - Clear information: Each subplot has verification metrics

7. SOLID Design Principles
   - SRP (Single Responsibility): Calculation / Analysis / Visualization separated
   - OCP (Open/Closed): Easy to extend with new metrics
   - ISP (Interface Segregation): Methods take only needed parameters
   - DIP (Dependency Inversion): High-level orchestrator delegates to specific methods

=== AI Context Management ===

CRITICAL: AI context resets frequently
- NEVER make judgments with incomplete context
- ALWAYS restore context by reading reference code FIRST
- Reference files: verify_physical_array.py (ground truth)
- When confused, re-read reference files instead of guessing

=== Change Log ===

- 2025-10-27: Absolute naming clarity (AI hallucination prevention)
  - "Layer" is NOT BS-exclusive (both BS and UE have layers in general)
  - Current implementation: BS uses layer-level (4×4), UE uses TRX-level (2×2)
  - ALL methods/variables now have explicit bs_ or ue_ prefix
  
  BS (Layer-level):
  - BSLayerDFTCalculator (renamed from LayerDFTCalculator)
  - bs_layer_get_dft_codeword, bs_layer_beam_index_to_angle_lcs
  - BSPhysicalArrayBeamCalculator: bs_calculate_layer_beam_pattern, bs_extract_main_lobe
  - Variables: bs_layer_idx, bs_beam_idx, bs_ae_*, bs_pattern
  
  UE (TRX-level):
  - UETRXDFTCalculator (explicit UE TRX)
  - ue_trx_get_dft_codeword, ue_trx_beam_index_to_angle_lcs
  - UEPhysicalArrayBeamCalculator: ue_calculate_trx_beam_pattern, ue_extract_main_lobe
  - Variables: ue_trx_idx, ue_beam_idx, ue_ae_*, ue_pattern

- 2025-10-27: UE physical array implementation (BS-equivalent architecture)
  - UE TRX-level patterns use physical AE positions (4 TRXs × 4 AE)
  - Architecture: 4×4 single panel, 0.5λ spacing, sequential TRX indexing

- 2025-10-27: DFT codebook unified with P1O (ABSOLUTE REFERENCE)
  - P1O_PADP_to_BM_2510v6.py = 절대 기준
  - Replaced meshgrid-based DFT with P1O Kronecker product
  - beam_index_to_angle_lcs() replaced with P1O method (iz-major order)
  - Ensures 100% consistency with P1O (atol=1e-10)
  - Affected: P1O_DFTCodebook (new), BSLayerDFTCalculator, UETRXDFTCalculator, UEPhysicalArrayBeamCalculator
  - Contour levels (absolute gain):
    - BS: [1, 4] @ peak=16 (16ant = -12dB, -6dB)
    - UE: [0.5, 2] @ peak=4 (4ant = -9dB, -3dB)
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from collections import Counter
import csv
import glob
import warnings
warnings.filterwarnings('ignore')
from scipy.ndimage import label as ndimage_label
import tensorflow as tf

# ===== Configuration =====
class ReplotConfig:
    """Configuration for P1O polar plot regeneration"""
    
    def __init__(self):
        # Paths
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_dir = script_dir
        
        # Input: CSV results
        self.P1O_OUTPUT_DIR = os.path.join(self.project_dir, "Reference_Data")
        self.csv_pattern = "Area*_P1O_BM_*_p*.csv"
        
        # Input: P1A ray data
        self.P1A_DIR = os.path.join(self.project_dir, "P1A_Data")
        
        # Output
        self.output_dir = os.path.join(self.project_dir, "Polar_Plots_Replot")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Visualization options
        self.main_lobe_only = False
        
        # BS orientation
        self.BS_orientation = {
            "azimuth_deg": 246,
            "downtilt_deg": 3,
            "roll_deg": 0
        }
        
        # UE orientation
        self.UE_orientation = {
            "azimuth_deg": 0,
            "elevation_deg": 0,
            "roll_deg": 0
        }


# ===== CSV Parser =====
class CSVResultParser:
    """Parse P1O CSV results"""
    
    def __init__(self, config):
        self.config = config
    
    def load_all_stage4_results(self):
        """Load Stage 4 results
        
        Returns:
            list: [(area, freq, ue, ue_beams, bs_beams)]
        """
        csv_pattern = os.path.join(self.config.P1O_OUTPUT_DIR, self.config.csv_pattern)
        csv_files = glob.glob(csv_pattern)
        
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found: {csv_pattern}")
        
        print(f"Found {len(csv_files)} CSV files\n")
        
        results = []
        for csv_file in csv_files:
            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row['stage'] == 'Stage4':
                        ue = int(row['ue'])
                        
                        ue_beams_str = row['ue_beams'].strip()
                        bs_beams_str = row['bs_beams_abm'].strip()
                        
                        if ue_beams_str and bs_beams_str:
                            # UE beams (cyclic repeat to 4)
                            ue_beams_list = [int(b) for b in ue_beams_str.split(',')]
                            ue_beams = []
                            idx = 0
                            while len(ue_beams) < 4:
                                ue_beams.append(ue_beams_list[idx % len(ue_beams_list)])
                                idx += 1
                            
                            # BS beams: Layer 0,1,2,... sequence
                            bs_beam_indices = [int(b) for b in bs_beams_str.split(',')]
                            bs_beams = [(layer_idx, beam_idx) 
                                       for layer_idx, beam_idx in enumerate(bs_beam_indices)]
                            
                            results.append((1, 7.5, ue, ue_beams, bs_beams))
        
        print(f"Loaded {len(results)} Stage 4 results")
        return results


# ===== Coordinate Transformation =====
def rotation_matrix(alpha_rad, beta_rad, gamma_rad):
    """3GPP TS 38.901 rotation matrix"""
    ca, sa = np.cos(alpha_rad), np.sin(alpha_rad)
    cb, sb = np.cos(beta_rad), np.sin(beta_rad)
    cc, sc = np.cos(gamma_rad), np.sin(gamma_rad)
    
    R = np.array([
        [ca*cb, ca*sb*sc - sa*cc, ca*sb*cc + sa*sc],
        [sa*cb, sa*sb*sc + ca*cc, sa*sb*cc - ca*sc],
        [-sb,   cb*sc,            cb*cc]
    ])
    return R


def gcs_to_lcs(theta_gcs_rad, phi_gcs_rad, orientation_deg):
    """Convert GCS angles to LCS angles"""
    alpha = np.radians(orientation_deg['azimuth_deg'])
    beta = np.radians(orientation_deg.get('downtilt_deg', 
                      orientation_deg.get('elevation_deg', 0)))
    gamma = np.radians(orientation_deg.get('roll_deg', 0))
    
    st, ct = np.sin(theta_gcs_rad), np.cos(theta_gcs_rad)
    sp, cp = np.sin(phi_gcs_rad), np.cos(phi_gcs_rad)
    rho_gcs = np.array([st * cp, st * sp, ct])
    
    R = rotation_matrix(alpha, beta, gamma)
    R_inv = R.T
    rho_lcs = R_inv @ rho_gcs
    
    rho_lcs_norm = np.linalg.norm(rho_lcs)
    if rho_lcs_norm < 1e-10:
        return (0.0, 0.0)
    
    rho_lcs = rho_lcs / rho_lcs_norm
    
    theta_lcs = np.arccos(np.clip(rho_lcs[2], -1.0, 1.0))
    phi_lcs = np.arctan2(rho_lcs[1], rho_lcs[0])
    
    return theta_lcs, phi_lcs


# ===== Cluster Analyzer =====
class ClusterAnalyzer:
    """Analyze ray clusters from P1A data"""
    
    def __init__(self, config):
        self.config = config
        self._rays_cache = {}
    
    def load_ray_data(self, area, freq, ue_idx):
        """Load P1A ray data for single UE"""
        npz_key = (area, freq)
        
        if npz_key not in self._rays_cache:
            npz_path = os.path.join(
                self.config.P1A_DIR,
                f"Area{area}_{freq}GHz_Rays_ALL_RXs.npz"
            )
            
            if not os.path.exists(npz_path):
                raise FileNotFoundError(f"P1A data not found: {npz_path}")
            
            self._rays_cache[npz_key] = np.load(npz_path)
        
        rays = self._rays_cache[npz_key]
        
        mask = rays['rx_indices'] == ue_idx
        if not np.any(mask):
            raise ValueError(f"No rays found for UE {ue_idx}")
        
        ray_data = {
            'phi_t_deg': rays['phi_t_deg'][mask].ravel(),
            'theta_t_deg': rays['theta_t_deg'][mask].ravel(),
            'phi_r_deg': rays['phi_r_deg'][mask].ravel(),
            'theta_r_deg': rays['theta_r_deg'][mask].ravel(),
            'power': rays['power'][mask].ravel(),
            'source_path_idx': rays['source_path_idx'][mask].ravel()
        }
        
        return ray_data
    
    def group_by_path_idx(self, ray_data):
        """Group rays by source_path_idx"""
        path_indices = np.unique(ray_data['source_path_idx'])
        clusters = {}
        
        for path_idx in path_indices:
            mask = ray_data['source_path_idx'] == path_idx
            
            clusters[int(path_idx)] = {
                'phi_t_deg': ray_data['phi_t_deg'][mask],
                'theta_t_deg': ray_data['theta_t_deg'][mask],
                'phi_r_deg': ray_data['phi_r_deg'][mask],
                'theta_r_deg': ray_data['theta_r_deg'][mask],
                'power': ray_data['power'][mask],
                'n_rays': np.sum(mask)
            }
        
        return clusters
    
    def compute_cluster_stats(self, cluster, side='tx'):
        """Compute cluster statistics in LCS"""
        power = cluster['power']
        total_power = np.sum(power)
        
        if side == 'tx':
            theta_gcs_deg = cluster['theta_t_deg']
            phi_gcs_deg = cluster['phi_t_deg']
            orientation = self.config.BS_orientation
        else:
            theta_gcs_deg = cluster['theta_r_deg']
            phi_gcs_deg = cluster['phi_r_deg']
            orientation = self.config.UE_orientation
        
        theta_lcs_list = []
        phi_lcs_list = []
        
        for theta_deg, phi_deg in zip(theta_gcs_deg, phi_gcs_deg):
            theta_rad = np.radians(theta_deg)
            phi_rad = np.radians(phi_deg)
            theta_lcs, phi_lcs = gcs_to_lcs(theta_rad, phi_rad, orientation)
            theta_lcs_list.append(theta_lcs)
            phi_lcs_list.append(phi_lcs)
        
        theta_lcs = np.array(theta_lcs_list)
        phi_lcs = np.array(phi_lcs_list)
        
        theta_lcs_mean = self._compute_circular_mean(theta_lcs, power)
        phi_lcs_mean = self._compute_circular_mean(phi_lcs, power)
        
        return {
            'power_dB': 10 * np.log10(total_power) if total_power > 0 else -np.inf,
            'total_power': total_power,
            'phi_lcs_mean': phi_lcs_mean,
            'theta_lcs_mean': theta_lcs_mean,
            'n_rays': cluster['n_rays']
        }
    
    def _compute_circular_mean(self, angles_rad, weights):
        """Power-weighted circular mean"""
        if len(angles_rad) == 0:
            return 0.0
        
        total_weight = np.sum(weights)
        if total_weight == 0:
            return 0.0
        
        x = np.sum(weights * np.cos(angles_rad)) / total_weight
        y = np.sum(weights * np.sin(angles_rad)) / total_weight
        
        return np.arctan2(y, x)
    

# ===== P1O DFT Codebook (Absolute Reference) =====
class P1O_DFTCodebook:
    """P1O 절대 기준 DFT 코드북 (Kronecker product)
    
    Source: P1O_PADP_to_BM_2510v6.py Line 389-441
    
    P1O_PADP_to_BM_2510v6.py의 DFT 생성 방식을 NumPy로 정확히 복제.
    P1O가 절대 기준이며, 이 클래스는 P1O와 100% 일치를 보장.
    """
    
    @staticmethod
    def generate_1d_dft(N: int, K: int) -> np.ndarray:
        """1D DFT codebook (P1O Line 395-413)
        
        [F_{N,K}]_{i,j} = (1/√N) exp(-j 2π ij / (NK))
        
        Args:
            N: Antennas per dimension
            K: Oversample factor
        
        Returns:
            F: [N, NK] complex128
        """
        i = np.arange(N)[:, None].astype(np.float64)
        j = np.arange(N * K)[None, :].astype(np.float64)
        phase = -2.0 * np.pi * i * j / float(N * K)
        sqrt_N = np.sqrt(float(N))
        return np.exp(1j * phase) / sqrt_N
    
    @staticmethod
    def generate_2d_dft_codebook(N: int, K: int) -> np.ndarray:
        """2D DFT codebook via Kronecker product (P1O Line 417-441)
        
        F = F_{N,K} ⊗ F_{N,K}
        
        Args:
            N: Antennas per dimension
            K: Oversample factor
        
        Returns:
            F: [N², (NK)²] complex128
            
        Examples:
            TRX (2×2): N=2, K=2 → [4, 16]
            Layer (4×4): N=4, K=2 → [16, 64]
        """
        F_1d = P1O_DFTCodebook.generate_1d_dft(N, K)
        
        # Kronecker product: P1O 방식 그대로
        n_ant = N
        n_beams = N * K
        
        # einsum('ij,kl->ikjl') → reshape
        F_kron = np.einsum('ij,kl->ikjl', F_1d, F_1d)
        F_kron = F_kron.reshape(n_ant**2, n_beams**2)
        
        return F_kron
    
    @staticmethod
    def get_codeword(F_codebook: np.ndarray, beam_idx: int) -> np.ndarray:
        """Extract single codeword from codebook
        
        Args:
            F_codebook: [n_ant, n_beams] full codebook
            beam_idx: Beam index (0-based)
        
        Returns:
            w: [n_ant] complex codeword
        """
        return F_codebook[:, beam_idx]
    
    @staticmethod
    def beam_index_to_angle_lcs(j: int, N: int, K: int, side: str) -> tuple:
        """Beam index → LCS angles (P1O Line 325-372)
        
        P1O 절대 기준 구현
        
        Args:
            j: Beam index [0, (NK)²-1]
            N: Antennas per dimension
            K: Oversample factor
            side: 'tx' (BS) or 'rx' (UE) for downlink
                - Downlink: BS='tx', UE='rx'
        
        Returns:
            (theta_rad, phi_rad): Zenith, azimuth in LCS
        """
        G = N * K
        
        # P1O Line 340-342: iz-major order
        iz = j // G
        iy = j % G
        
        # Centered q indices
        q_y = iy - G // 2
        q_z = iz - G // 2
        
        # α from q (side-dependent)
        qy_overG = float(q_y) / float(G)
        qz_overG = float(q_z) / float(G)
        
        # P1O Line 349-352: side-dependent sign
        if side == 'tx':  # BS (side=0 in P1O)
            ay = -qy_overG
            az = -qz_overG
        elif side == 'rx':  # UE (side=1 in P1O)
            ay = +qy_overG
            az = +qz_overG
        else:
            raise ValueError(f"side must be 'tx' or 'rx', got {side}")
        
        # Physical parameters (P1O Line 354-356)
        sign_y, sign_z = 1.0, -1.0
        dy_lam, dz_lam = 0.5, 0.5
        
        # Inverse transform to angles (P1O Line 358-371)
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


# ===== Layer DFT Calculator =====
class BSLayerDFTCalculator:
    """BS Layer-level 4×4 DFT codewords (P1O method)
    
    BS only: 64 layers, each with 4×4 subarray, 64 DFT beams
    P1O_PADP_to_BM_2510v6.py Stage 3,4 layer-level DFT
    """
    
    def __init__(self):
        self.N = 4
        self.K = 2
        self.G = self.N * self.K  # 8
        self.num_antennas = self.N * self.N  # 16
        
        # Pre-generate full codebook (P1O 방식)
        self.F_codebook = P1O_DFTCodebook.generate_2d_dft_codebook(self.N, self.K)
        # Shape: [16, 64]
    
    def bs_layer_get_dft_codeword(self, beam_idx: int) -> np.ndarray:
        """BS layer DFT codeword from P1O codebook
        
        Args:
            beam_idx: BS beam index [0, 63]
        
        Returns:
            w: [16] complex codeword for BS 4×4 layer
        """
        return P1O_DFTCodebook.get_codeword(self.F_codebook, beam_idx)
    
    def bs_layer_beam_index_to_angle_lcs(self, j: int) -> tuple:
        """BS layer beam index → LCS angles (P1O method)
        
        Args:
            j: BS beam index [0, 63]
        
        Returns:
            (theta_rad, phi_rad)
        """
        # BS is tx side in downlink
        return P1O_DFTCodebook.beam_index_to_angle_lcs(j, self.N, self.K, side='tx')


# ===== UE TRX DFT Calculator =====
class UETRXDFTCalculator:
    """UE TRX-level 2×2 DFT codewords (P1O method)
    
    UE only: 4 TRXs, each with 2×2 subarray, 16 DFT beams per TRX
    P1O_PADP_to_BM_2510v6.py Stage 1,2 TRX-level DFT
    """
    
    def __init__(self):
        self.N = 2  # 2×2 subarray per UE TRX
        self.K = 2  # Oversample factor
        self.G = self.N * self.K  # 4
        self.num_antennas = self.N * self.N  # 4
        
        # Pre-generate full UE TRX codebook (P1O method)
        self.F_codebook = P1O_DFTCodebook.generate_2d_dft_codebook(self.N, self.K)
        # Shape: [4, 16]
    
    def ue_trx_get_dft_codeword(self, ue_beam_idx: int) -> np.ndarray:
        """UE TRX DFT codeword from P1O codebook
        
        Args:
            ue_beam_idx: UE beam index [0, 15]
        
        Returns:
            w: [4] complex codeword for UE 2×2 TRX array
        """
        return P1O_DFTCodebook.get_codeword(self.F_codebook, ue_beam_idx)
    
    def ue_trx_beam_index_to_angle_lcs(self, ue_beam_idx: int) -> tuple:
        """UE TRX beam index → LCS angles (P1O method)
        
        Args:
            ue_beam_idx: UE beam index [0, 15]
        
        Returns:
            (theta_rad, phi_rad)
        """
        # UE is rx side in downlink
        return P1O_DFTCodebook.beam_index_to_angle_lcs(ue_beam_idx, self.N, self.K, side='rx')


# ===== BS Physical Array Beam Calculator =====
class BSPhysicalArrayBeamCalculator:
    """BS Layer-level beam pattern with physical 1024 AE array (TensorFlow)"""
    
    def __init__(self):
        # Array configuration
        self.n_panels_per_dim = 8
        self.n_elements_per_panel_per_dim = 4
        self.total_ae = 1024
        
        # Spacing (wavelengths)
        self.panel_spacing = 2.5
        self.element_spacing = 0.5
        
        # DFT codebook
        self.N = 4
        self.G = 8
        
        # BS Layer DFT calculator
        self.bs_layer_dft_calc = BSLayerDFTCalculator()
        
        # Angle grid (2° resolution)
        self.az_grid = np.linspace(-np.pi, np.pi, 181)
        self.el_grid = np.linspace(-90, 90, 91)
        
        # Pre-compute 1024 BS AE coordinates
        print("  Pre-computing 1024 BS AE coordinates...")
        bs_ae_y_list = []
        bs_ae_z_list = []
        
        for bs_ae_idx in range(1024):
            bs_p_row, bs_p_col, bs_e_row, bs_e_col = self._bs_ae_index_to_physical_coords(bs_ae_idx)
            bs_y, bs_z = self._bs_get_ae_physical_position(bs_p_row, bs_p_col, bs_e_row, bs_e_col)
            bs_ae_y_list.append(bs_y)
            bs_ae_z_list.append(bs_z)
        
        # TF constants for BS AE positions
        self.bs_ae_y_positions = tf.constant(bs_ae_y_list, dtype=tf.float32)
        self.bs_ae_z_positions = tf.constant(bs_ae_z_list, dtype=tf.float32)
        
        # Angle grids as TF constants
        az_mesh, el_mesh = np.meshgrid(self.az_grid, self.el_grid)
        theta_mesh = np.radians(90 - el_mesh)
        
        self.theta_grid = tf.constant(theta_mesh, dtype=tf.float32)
        self.phi_grid = tf.constant(az_mesh, dtype=tf.float32)
        
        print("  ✓ 1024 BS AE coordinates cached")
    
    def _bs_ae_index_to_physical_coords(self, bs_ae_idx: int) -> tuple:
        """BS AE index → (panel_row, panel_col, elem_row, elem_col)
        
        BS 1024 AE: 8×8 panels, each panel = 4×4 elements
        
        Args:
            bs_ae_idx: BS AE index [0, 1023]
        """
        panel_major = bs_ae_idx // 16
        elem_in_panel = bs_ae_idx % 16
        
        panel_row = panel_major // 8
        panel_col = panel_major % 8
        
        elem_row = elem_in_panel // 4
        elem_col = elem_in_panel % 4
        
        return panel_row, panel_col, elem_row, elem_col
    
    def _bs_get_ae_physical_position(self, bs_panel_row: int, bs_panel_col: int,
                                      bs_elem_row: int, bs_elem_col: int) -> tuple:
        """BS AE physical position (non-centered wavelengths)
        
        BS panel spacing: 2.5λ, element spacing: 0.5λ
        
        Args:
            bs_panel_row: Panel row [0, 7]
            bs_panel_col: Panel col [0, 7]
            bs_elem_row: Element row in panel [0, 3]
            bs_elem_col: Element col in panel [0, 3]
        
        Returns:
            (bs_y, bs_z): BS AE position in wavelengths
        """
        y_panel = bs_panel_col * self.panel_spacing
        z_panel = bs_panel_row * self.panel_spacing
        
        y_elem = bs_elem_col * self.element_spacing
        z_elem = bs_elem_row * self.element_spacing
        
        y_total = y_panel + y_elem
        z_total = z_panel + z_elem
        
        return y_total, z_total
    
    @tf.function(jit_compile=True, reduce_retracing=True)
    def _bs_calculate_layer_beam_pattern_core(self, w_bs_layer, bs_layer_ae_y, bs_layer_ae_z):
        """BS layer beam pattern using actual 1024 AE physical positions
        
        Args:
            w_bs_layer: [16] complex BS layer codeword
            bs_layer_ae_y: [16] Y positions of BS layer's AE
            bs_layer_ae_z: [16] Z positions of BS layer's AE
        
        Returns:
            pattern: [91, 181] BS layer beam gain pattern
        """
        k = 2.0 * np.pi
        
        sin_theta = tf.sin(self.theta_grid)
        sin_phi = tf.sin(self.phi_grid)
        cos_phi = tf.cos(self.phi_grid)
        
        phase_y_coeff = k * sin_theta * sin_phi
        phase_z_coeff = k * sin_theta * cos_phi
        
        phase_y_coeff_exp = tf.expand_dims(phase_y_coeff, axis=-1)
        phase_z_coeff_exp = tf.expand_dims(phase_z_coeff, axis=-1)
        
        bs_layer_ae_y_exp = tf.reshape(bs_layer_ae_y, [1, 1, 16])
        bs_layer_ae_z_exp = tf.reshape(bs_layer_ae_z, [1, 1, 16])
        
        phase_test = phase_y_coeff_exp * bs_layer_ae_y_exp + phase_z_coeff_exp * bs_layer_ae_z_exp
        
        a_test = tf.exp(tf.complex(tf.constant(0.0), -phase_test))
        
        w_bs_layer_exp = tf.reshape(w_bs_layer, [1, 1, 16])
        
        inner_prod = tf.reduce_sum(tf.math.conj(a_test) * w_bs_layer_exp, axis=-1)
        gain = tf.abs(inner_prod) ** 2
        
        return tf.cast(gain, tf.float32)
    
    def bs_calculate_layer_beam_pattern(self, bs_layer_idx: int, bs_beam_idx: int):
        """Calculate BS layer beam pattern using actual 1024 AE positions
        
        Args:
            bs_layer_idx: BS layer index [0, 63]
            bs_beam_idx: BS beam index [0, 63]
        
        Returns:
            pattern: [91, 181] TF tensor
        """
        w_bs_np = self.bs_layer_dft_calc.bs_layer_get_dft_codeword(bs_beam_idx)
        w_bs_tf = tf.constant(w_bs_np, dtype=tf.complex64)
        
        bs_ae_start = bs_layer_idx * 16
        bs_ae_end = bs_ae_start + 16
        bs_layer_ae_y = self.bs_ae_y_positions[bs_ae_start:bs_ae_end]
        bs_layer_ae_z = self.bs_ae_z_positions[bs_ae_start:bs_ae_end]
        
        pattern = self._bs_calculate_layer_beam_pattern_core(w_bs_tf, bs_layer_ae_y, bs_layer_ae_z)
        
        # Hemisphere restriction
        theta_center, phi_center = self.bs_layer_dft_calc.bs_layer_beam_index_to_angle_lcs(bs_beam_idx)
        elev_center = 90.0 - np.degrees(theta_center)
        
        el_grid = 90.0 - np.degrees(self.theta_grid.numpy())
        
        if elev_center > 0:
            mask = tf.cast(el_grid >= 0, tf.float32)
        elif elev_center < 0:
            mask = tf.cast(el_grid <= 0, tf.float32)
        else:
            mask = tf.ones_like(pattern, dtype=tf.float32)
        
        pattern_masked = pattern * mask
        
        # Normalize
        w_bs_norm_sq = tf.constant(np.linalg.norm(w_bs_np)**2, dtype=tf.float32)
        pattern_normalized = pattern_masked / w_bs_norm_sq
        
        return pattern_normalized
    
    def bs_extract_main_lobe(self, bs_pattern_tf):
        """Extract BS main lobe region
        
        Args:
            bs_pattern_tf: [91, 181] BS pattern tensor
        
        Returns:
            bs_main_lobe_pattern: [91, 181] BS main lobe only
        """
        bs_pattern = bs_pattern_tf.numpy() if hasattr(bs_pattern_tf, 'numpy') else bs_pattern_tf
        
        # Simple threshold-based extraction
        threshold = 0.5 * np.max(bs_pattern)
        mask = bs_pattern >= threshold
        
        labeled, n_components = ndimage_label(mask)
        
        if n_components == 0:
            return bs_pattern
        
        # Find peak
        peak_idx = np.unravel_index(np.argmax(bs_pattern), bs_pattern.shape)
        center_label = labeled[peak_idx]
        
        if center_label == 0:
            return bs_pattern
        
        main_lobe_mask = (labeled == center_label)
        
        return bs_pattern * main_lobe_mask


# ===== UE Physical Array Beam Calculator =====
class UEPhysicalArrayBeamCalculator:
    """UE TRX-level beam pattern with physical 16 AE array (TensorFlow)
    
    UE Structure:
    - 4 TRXs, each TRX = 4 AE (2×2 subarray)
    - Physical array: 4×4 single panel = 16 AE total
    - Element spacing: 0.5λ
    - Sequential indexing: TRX 0=[0:4], TRX 1=[4:8], TRX 2=[8:12], TRX 3=[12:16]
    
    TRX Codebook → PhyAE Codebook:
      1. UE TRX j → AE indices [j*4 : (j+1)*4]
      2. DFT codeword (4) + 16 AE positions → beam pattern
    
    Architecture matches BSPhysicalArrayBeamCalculator:
      - Pre-computed AE coordinates as tf.constant
      - Vectorized processing [N_el, N_az, 16]
      - @tf.function JIT compilation
      - Explicit ue_ prefix in ALL methods and variables
    """
    
    def __init__(self):
        # UE array configuration
        self.ue_n_elements_per_dim = 4  # 4×4 panel
        self.ue_total_ae = 16
        self.ue_n_trxs = 4
        
        # UE spacing (wavelengths)
        self.ue_element_spacing = 0.5
        
        # UE TRX DFT calculator (2×2, K=2)
        self.ue_trx_dft_calc = UETRXDFTCalculator()
        
        # Angle grid (2° resolution)
        self.az_grid = np.linspace(-np.pi, np.pi, 181)
        self.el_grid = np.linspace(-90, 90, 91)
        
        # Pre-compute all 16 UE AE physical coordinates
        print("  Pre-computing UE 16 AE coordinates...")
        ue_ae_y_list = []
        ue_ae_z_list = []
        
        for ue_ae_idx in range(16):
            ue_row, ue_col = self._ue_ae_index_to_physical_coords(ue_ae_idx)
            ue_y, ue_z = self._ue_get_ae_physical_position(ue_row, ue_col)
            ue_ae_y_list.append(ue_y)
            ue_ae_z_list.append(ue_z)
        
        # TF constants for UE AE positions
        self.ue_ae_y_positions = tf.constant(ue_ae_y_list, dtype=tf.float32)
        self.ue_ae_z_positions = tf.constant(ue_ae_z_list, dtype=tf.float32)
        
        # Angle grids as TF constants
        az_mesh, el_mesh = np.meshgrid(self.az_grid, self.el_grid)
        theta_mesh = np.radians(90 - el_mesh)
        
        self.theta_grid = tf.constant(theta_mesh, dtype=tf.float32)
        self.phi_grid = tf.constant(az_mesh, dtype=tf.float32)
        
        print("  ✓ UE 16 AE coordinates cached (P1O codebook)")
    
    def _ue_ae_index_to_physical_coords(self, ue_ae_idx: int) -> tuple:
        """UE AE index → (row, col) in 4×4 panel
        
        UE sequential indexing with 2×2 TRX blocks:
        TRX 0 = AE [0:4], TRX 1 = AE [4:8], TRX 2 = AE [8:12], TRX 3 = AE [12:16]
        
        Layout:
        Row/Col    0    1    2    3
        0         [0   1]  [4   5]    TRX 0: [0,1,2,3]
        1         [2   3]  [6   7]    TRX 1: [4,5,6,7]
        2         [8   9]  [12 13]    TRX 2: [8,9,10,11]
        3         [10 11]  [14 15]    TRX 3: [12,13,14,15]
        
        Args:
            ue_ae_idx: UE AE index [0, 15]
        
        Returns:
            (ue_row, ue_col): Position in 4×4 panel
        """
        ue_trx_idx = ue_ae_idx // 4
        ue_ae_in_trx = ue_ae_idx % 4
        
        ue_trx_row = ue_trx_idx // 2
        ue_trx_col = ue_trx_idx % 2
        
        ue_ae_row_in_trx = ue_ae_in_trx // 2
        ue_ae_col_in_trx = ue_ae_in_trx % 2
        
        ue_row = ue_trx_row * 2 + ue_ae_row_in_trx
        ue_col = ue_trx_col * 2 + ue_ae_col_in_trx
        
        return ue_row, ue_col
    
    def _ue_get_ae_physical_position(self, ue_row: int, ue_col: int) -> tuple:
        """UE AE physical position (non-centered wavelengths)
        
        UE element spacing: 0.5λ (no panel spacing, single 4×4 panel)
        
        Args:
            ue_row: Row in 4×4 panel [0, 3]
            ue_col: Col in 4×4 panel [0, 3]
        
        Returns:
            (ue_y, ue_z): UE AE position in wavelengths
        """
        ue_y = ue_col * self.ue_element_spacing
        ue_z = ue_row * self.ue_element_spacing
        return ue_y, ue_z
    
    @tf.function(jit_compile=True, reduce_retracing=True)
    def _ue_calculate_trx_beam_pattern_core(self, w_ue_trx, ue_trx_ae_y, ue_trx_ae_z):
        """UE TRX beam pattern using actual 16 AE physical positions
        
        Args:
            w_ue_trx: [4] complex UE TRX codeword
            ue_trx_ae_y: [4] Y positions of UE TRX's AE
            ue_trx_ae_z: [4] Z positions of UE TRX's AE
        
        Returns:
            pattern: [91, 181] UE TRX beam gain pattern
        """
        k = 2.0 * np.pi
        
        sin_theta = tf.sin(self.theta_grid)
        sin_phi = tf.sin(self.phi_grid)
        cos_phi = tf.cos(self.phi_grid)
        
        phase_y_coeff = k * sin_theta * sin_phi
        phase_z_coeff = k * sin_theta * cos_phi
        
        phase_y_coeff_exp = tf.expand_dims(phase_y_coeff, axis=-1)
        phase_z_coeff_exp = tf.expand_dims(phase_z_coeff, axis=-1)
        
        ue_trx_ae_y_exp = tf.reshape(ue_trx_ae_y, [1, 1, 4])
        ue_trx_ae_z_exp = tf.reshape(ue_trx_ae_z, [1, 1, 4])
        
        phase = phase_y_coeff_exp * ue_trx_ae_y_exp + phase_z_coeff_exp * ue_trx_ae_z_exp
        
        a_ue = tf.exp(tf.complex(tf.constant(0.0), -phase))
        
        w_ue_trx_exp = tf.reshape(w_ue_trx, [1, 1, 4])
        
        inner_prod = tf.reduce_sum(tf.math.conj(a_ue) * w_ue_trx_exp, axis=-1)
        gain = tf.abs(inner_prod) ** 2
        
        return tf.cast(gain, tf.float32)
    
    def ue_calculate_trx_beam_pattern(self, ue_trx_idx: int, ue_beam_idx: int):
        """Calculate UE TRX beam pattern using actual 16 AE positions
        
        Args:
            ue_trx_idx: UE TRX index [0, 3]
            ue_beam_idx: UE beam index [0, 15]
        
        Returns:
            pattern: [91, 181] TF tensor
        """
        w_ue_np = self.ue_trx_dft_calc.ue_trx_get_dft_codeword(ue_beam_idx)
        w_ue_tf = tf.constant(w_ue_np, dtype=tf.complex64)
        
        ue_ae_start = ue_trx_idx * 4
        ue_ae_end = ue_ae_start + 4
        ue_trx_ae_y = self.ue_ae_y_positions[ue_ae_start:ue_ae_end]
        ue_trx_ae_z = self.ue_ae_z_positions[ue_ae_start:ue_ae_end]
        
        pattern = self._ue_calculate_trx_beam_pattern_core(w_ue_tf, ue_trx_ae_y, ue_trx_ae_z)
        
        # Hemisphere restriction
        theta_center, phi_center = self.ue_trx_dft_calc.ue_trx_beam_index_to_angle_lcs(ue_beam_idx)
        elev_center = 90.0 - np.degrees(theta_center)
        
        el_grid = 90.0 - np.degrees(self.theta_grid.numpy())
        
        if elev_center > 0:
            mask = tf.cast(el_grid >= 0, tf.float32)
        elif elev_center < 0:
            mask = tf.cast(el_grid <= 0, tf.float32)
        else:
            mask = tf.ones_like(pattern, dtype=tf.float32)
        
        pattern_masked = pattern * mask
        
        # Normalize
        w_ue_norm_sq = tf.constant(np.linalg.norm(w_ue_np)**2, dtype=tf.float32)
        pattern_normalized = pattern_masked / w_ue_norm_sq
        
        return pattern_normalized
    
    def ue_extract_main_lobe(self, ue_pattern_tf):
        """Extract UE main lobe region
        
        Args:
            ue_pattern_tf: [91, 181] UE pattern tensor
        
        Returns:
            ue_main_lobe_pattern: [91, 181] UE main lobe only
        """
        ue_pattern = ue_pattern_tf.numpy() if hasattr(ue_pattern_tf, 'numpy') else ue_pattern_tf
        
        threshold = 0.5 * np.max(ue_pattern)
        mask = ue_pattern >= threshold
        
        labeled, n_components = ndimage_label(mask)
        
        if n_components == 0:
            return ue_pattern
        
        peak_idx = np.unravel_index(np.argmax(ue_pattern), ue_pattern.shape)
        center_label = labeled[peak_idx]
        
        if center_label == 0:
            return ue_pattern
        
        main_lobe_mask = (labeled == center_label)
        
        return ue_pattern * main_lobe_mask


# ===== Polar Plotter (SOLID Principles) =====
class PolarPlotter:
    """Polar plot generator with SOLID principles
    
    SOLID Design:
    - SRP: Calculation / Analysis / Visualization separated
    - OCP: Easy to extend with new metrics
    - ISP: Methods take only needed parameters
    - DIP: High-level orchestrator delegates to specific methods
    """
    
    def __init__(self, config):
        self.config = config
        self.bs_phy_calc = None  # BS calculator
        self.ue_phy_calc = None  # UE calculator
    
    # === Pattern Calculation (Single Responsibility) ===
    
    def bs_calculate_layer_beam_pattern(self, bs_layer_idx: int, bs_beam_idx: int):
        """Calculate BS layer beam pattern
        
        Args:
            bs_layer_idx: BS layer index [0, 63]
            bs_beam_idx: BS beam index [0, 63]
        
        Returns:
            bs_pattern: [91, 181] BS beam pattern
        """
        if self.bs_phy_calc is None:
            self.bs_phy_calc = BSPhysicalArrayBeamCalculator()
        
        return self.bs_phy_calc.bs_calculate_layer_beam_pattern(bs_layer_idx, bs_beam_idx)
    
    def bs_calculate_total_pattern(self, bs_layer_beams: list) -> np.ndarray:
        """Calculate total BS pattern from layer beams
        
        Args:
            bs_layer_beams: List of (bs_layer_idx, bs_beam_idx) tuples
        
        Returns:
            bs_total_pattern: [91, 181] summed BS pattern
        """
        if self.bs_phy_calc is None:
            self.bs_phy_calc = BSPhysicalArrayBeamCalculator()
        
        bs_patterns = []
        for bs_layer_idx, bs_beam_idx in bs_layer_beams:
            bs_pattern = self.bs_phy_calc.bs_calculate_layer_beam_pattern(bs_layer_idx, bs_beam_idx)
            bs_patterns.append(bs_pattern.numpy() if hasattr(bs_pattern, 'numpy') else bs_pattern)
        
        bs_total_pattern = np.sum(bs_patterns, axis=0)
        return bs_total_pattern
    
    def ue_calculate_trx_beam_pattern(self, ue_trx_idx: int, ue_beam_idx: int):
        """Calculate UE TRX beam pattern
        
        Args:
            ue_trx_idx: UE TRX index [0, 3]
            ue_beam_idx: UE beam index [0, 15]
        
        Returns:
            ue_pattern: [91, 181] UE beam pattern
        """
        if self.ue_phy_calc is None:
            self.ue_phy_calc = UEPhysicalArrayBeamCalculator()
        
        return self.ue_phy_calc.ue_calculate_trx_beam_pattern(ue_trx_idx, ue_beam_idx)
    
    def ue_calculate_total_pattern(self, ue_trx_beams: list) -> np.ndarray:
        """Calculate total UE pattern from TRX beams
        
        Args:
            ue_trx_beams: List of (ue_trx_idx, ue_beam_idx) tuples
        
        Returns:
            ue_total_pattern: [91, 181] summed UE pattern
        """
        if self.ue_phy_calc is None:
            self.ue_phy_calc = UEPhysicalArrayBeamCalculator()
        
        ue_patterns = []
        for ue_trx_idx, ue_beam_idx in ue_trx_beams:
            ue_pattern = self.ue_phy_calc.ue_calculate_trx_beam_pattern(ue_trx_idx, ue_beam_idx)
            ue_patterns.append(ue_pattern.numpy() if hasattr(ue_pattern, 'numpy') else ue_pattern)
        
        ue_total_pattern = np.sum(ue_patterns, axis=0)
        return ue_total_pattern
    
    # === Pattern Analysis (Single Responsibility) ===
    
    def extract_pattern_info(self, pattern, padp_elev, padp_az, side='bs'):
        """Extract verification info from pattern
        
        Args:
            side: 'bs' or 'ue'
        """
        if side == 'bs':
            if self.bs_phy_calc is None:
                self.bs_phy_calc = BSPhysicalArrayBeamCalculator()
            calc = self.bs_phy_calc
        else:  # 'ue'
            if self.ue_phy_calc is None:
                self.ue_phy_calc = UEPhysicalArrayBeamCalculator()
            calc = self.ue_phy_calc
            
        # Grids
        el_grid = 90.0 - np.degrees(calc.theta_grid.numpy())
        az_grid = np.degrees(calc.phi_grid.numpy())
        
        pattern_np = pattern.numpy() if hasattr(pattern, 'numpy') else pattern
        
        # Find peak
        peak_idx = np.unravel_index(np.argmax(pattern_np), pattern_np.shape)
        peak_elev = el_grid[peak_idx]
        peak_az = az_grid[peak_idx]
        
        # Gain at PADP
        padp_el_idx = np.argmin(np.abs(el_grid[:, 0] - padp_elev))
        padp_az_idx = np.argmin(np.abs(az_grid[0, :] - padp_az))
        padp_gain = float(pattern_np[padp_el_idx, padp_az_idx])
        
        # Angular distance
        elev_diff = peak_elev - padp_elev
        az_diff = peak_az - padp_az
        angular_distance = np.sqrt(elev_diff**2 + (az_diff * np.cos(np.radians(padp_elev)))**2)
        
        return {
            'peak_elev': float(peak_elev),
            'peak_az': float(peak_az),
            'padp_gain': float(padp_gain),
            'angular_distance': float(angular_distance)
        }
    
    # === Visualization (Single Responsibility) ===
    
    def _create_vertical_polar_subplots(self, n_plots):
        """Create n polar subplots in vertical layout"""
        fig, axes = plt.subplots(n_plots, 1, 
                                 figsize=(16, 4 * n_plots),
                                 subplot_kw={'projection': 'polar'})
        
        if n_plots == 1:
            axes = [axes]
        
        return fig, axes
    
    def _plot_padp_wedge(self, ax, cluster):
        """Plot PADP cluster as wedge (polar bar)
        
        Args:
            ax: Polar axis
            cluster: dict with 'elev', 'az', 'elev_spread', 'az_spread', 'power_dB'
        """
        # Extract parameters
        phi_center = np.radians(cluster['az'])
        phi_spread = np.radians(cluster['az_spread'])
        
        elev_center = cluster['elev']
        elev_spread = cluster['elev_spread']
        
        # Bottom (elevation range)
        bottom = max(0, elev_center - elev_spread / 2)
        
        # Color by power (turbo colormap, fixed range)
        vmin, vmax = -170, -80
        power_dB = cluster.get('power_dB', -120)
        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.cm.turbo
        
        # Plot wedge
        ax.bar(x=phi_center, height=elev_spread, width=phi_spread, bottom=bottom, color=cmap(norm(power_dB)), alpha=0.8, edgecolor='none')
    
    def _plot_pattern_on_axis(self, ax, pattern, title, padp_data, 
                             main_lobe_only=True, color='blue', linewidth=2.0, side='bs'):
        """Plot pattern on given polar axis
        
        Args:
            side: 'bs' or 'ue'
        """
        if side == 'bs':
            if self.bs_phy_calc is None:
                self.bs_phy_calc = BSPhysicalArrayBeamCalculator()
            calc = self.bs_phy_calc
        else:  # 'ue'
            if self.ue_phy_calc is None:
                self.ue_phy_calc = UEPhysicalArrayBeamCalculator()
            calc = self.ue_phy_calc
        
        # Extract contour
        if main_lobe_only:
            if side == 'bs':
                pattern_plot = calc.bs_extract_main_lobe(pattern)  # BS method
            else:  # 'ue'
                pattern_plot = calc.ue_extract_main_lobe(pattern)  # UE method
        else:
            pattern_plot = pattern
        
        # Grids
        el_grid = 90.0 - np.degrees(calc.theta_grid.numpy())
        az_grid = calc.phi_grid.numpy()
        
        pattern_np = pattern_plot.numpy() if hasattr(pattern_plot, 'numpy') else pattern_plot
        
        # Plot contour (side-dependent absolute levels)
        if side == 'bs':
            # BS: 4x4 array → peak gain = 16
            # levels: 1 (-12dB), 4 (-6dB)
            levels = [0.5, 4.0]
        else:  # 'ue'
            # UE: 2x2 array → peak gain = 4
            # levels: 0.5 (-9dB), 2 (-3dB)
            levels = [0.5, 2.0]
        
        ax.contour(az_grid, el_grid, pattern_np,
                   levels=levels,
                   colors=color,
                   linewidths=linewidth)
        
        # Polar plot settings (match v1)
        ax.set_ylim(-60, 105)
        ax.set_theta_direction(1)
        ax.set_theta_zero_location('N')
        ax.set_facecolor('#f5f5f5')
        ax.set_xlabel('Azimuth (deg)', fontsize=8)
        ax.set_ylabel('Elevation (deg)', fontsize=8)
        ax.grid(True, alpha=0.3)
    
        # PADP overlay - 모든 clusters 순회 (v1과 동일)
        if padp_data:
            for cluster in padp_data:
                self._plot_padp_wedge(ax, cluster)
            
            # Colorbar (v1과 동일)
            vmin, vmax = -170, -80
            norm = Normalize(vmin=vmin, vmax=vmax)
            cmap = plt.cm.turbo
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, pad=0.1, fraction=0.046)
            cbar.set_label('Cluster Power (dB)', fontsize=10)
        
        # Title
        ax.set_title(title, fontsize=10, fontweight='bold')
    
    def _create_2column_subplots(self, n_bs, n_ue):
        """Create 2-column polar subplots (SRP: subplot creation)
        
        Layout:
            Row 1: [BS Total] [UE Total]
            Row 2+: [BS individual] [UE individual]
        
        Args:
            n_bs: Number of BS layer-beams
            n_ue: Number of UE beams
        
        Returns:
            fig, axes_bs, axes_ue
        """
        n_rows = max(n_bs, n_ue) + 1
        fig = plt.figure(figsize=(16, 4 * n_rows))
        
        axes_bs = []
        axes_ue = []
        
        for row in range(n_rows):
            ax_bs = plt.subplot(n_rows, 2, 2*row + 1, projection='polar')
            axes_bs.append(ax_bs)
            
            ax_ue = plt.subplot(n_rows, 2, 2*row + 2, projection='polar')
            axes_ue.append(ax_ue)
        
        return fig, axes_bs, axes_ue
    
    def plot_2column_layout(self, ue_name, bs_layer_beams, ue_trx_beams, 
                            padp_bs_data, padp_ue_data, main_lobe_only=True):
        """Plot 2-column layout: BS (left) + UE (right)
        
        SRP: Orchestration only
        
        Args:
            ue_name: UE identifier
            bs_layer_beams: List of (bs_layer_idx, bs_beam_idx) tuples
            ue_trx_beams: List of (ue_trx_idx, ue_beam_idx) tuples
            padp_bs_data: BS PADP cluster data
            padp_ue_data: UE PADP cluster data
            main_lobe_only: Extract main lobe only
        """
        n_bs = len(bs_layer_beams)
        n_ue = len(ue_trx_beams)
        
        print(f"\n{'='*60}")
        print(f"2-Column Layout: {ue_name}")
        print(f"{'='*60}\n")
        
        # Extract PADP info
        if padp_bs_data:
            dom_bs = padp_bs_data[0]
            padp_bs_elev = dom_bs['elev']
            padp_bs_az = dom_bs['az']
        else:
            padp_bs_elev, padp_bs_az = 0.0, 0.0
        
        if padp_ue_data:
            dom_ue = padp_ue_data[0]
            padp_ue_elev = dom_ue['elev']
            padp_ue_az = dom_ue['az']
        else:
            padp_ue_elev, padp_ue_az = 0.0, 0.0
        
        # Phase 1: Calculate all BS patterns
        print(f"Calculating BS patterns ({n_bs} layer-beams)...")
        bs_patterns = []
        bs_infos = []
        
        for idx, (bs_layer_idx, bs_beam_idx) in enumerate(bs_layer_beams):
            bs_pattern = self.bs_calculate_layer_beam_pattern(bs_layer_idx, bs_beam_idx)
            bs_patterns.append(bs_pattern)
            
            info = self.extract_pattern_info(bs_pattern, padp_bs_elev, padp_bs_az, side='bs')
            bs_infos.append(info)
            
            print(f"  [{idx+1}/{n_bs}] Layer {bs_layer_idx} Beam {bs_beam_idx}")
        
        bs_total = self.bs_calculate_total_pattern(bs_layer_beams)
        bs_total_info = self.extract_pattern_info(bs_total, padp_bs_elev, padp_bs_az, side='bs')
        
        print(f"  [Total] BS: {n_bs} layer-beams")
        
        # Phase 2: Calculate all UE patterns
        print(f"\nCalculating UE patterns ({n_ue} TRX-beams)...")
        ue_patterns = []
        ue_infos = []
        
        for idx, (ue_trx_idx, ue_beam_idx) in enumerate(ue_trx_beams):
            ue_pattern = self.ue_calculate_trx_beam_pattern(ue_trx_idx, ue_beam_idx)
            ue_patterns.append(ue_pattern)
            
            info = self.extract_pattern_info(ue_pattern, padp_ue_elev, padp_ue_az, side='ue')
            ue_infos.append(info)
            
            print(f"  [{idx+1}/{n_ue}] TRX {ue_trx_idx} Beam {ue_beam_idx}")
        
        ue_total = self.ue_calculate_total_pattern(ue_trx_beams)
        ue_total_info = self.extract_pattern_info(ue_total, padp_ue_elev, padp_ue_az, side='ue')
        
        print(f"  [Total] UE: {n_ue} TRX-beams\n")
        
        # Phase 3: Create subplots
        fig, axes_bs, axes_ue = self._create_2column_subplots(n_bs, n_ue)
        
        # Phase 4: Plot Row 1 (Total)
        print(f"Plotting...")
        self._plot_pattern_on_axis(
            axes_bs[0], bs_total, 
            f'BS Total P1O ({n_bs} layers)',
            padp_bs_data, main_lobe_only, 
            color='red', linewidth=3.0, side='bs'
        )
        
        self._plot_pattern_on_axis(
            axes_ue[0], ue_total,
            f'UE Total P1O ({n_ue} beams)',
            padp_ue_data, main_lobe_only,
            color='red', linewidth=3.0, side='ue'
        )
        
        # Phase 5: Plot Row 2+ (Individuals)
        for idx, (bs_layer_idx, bs_beam_idx) in enumerate(bs_layer_beams):
            self._plot_pattern_on_axis(
                axes_bs[idx+1], bs_patterns[idx],
                f'Layer {bs_layer_idx} Beam {bs_beam_idx}',
                padp_bs_data, main_lobe_only,
                color='blue', linewidth=2.0, side='bs'
            )
        
        for idx, (ue_trx_idx, ue_beam_idx) in enumerate(ue_trx_beams):
            self._plot_pattern_on_axis(
                axes_ue[idx+1], ue_patterns[idx],
                f'TRX {ue_trx_idx} Beam {ue_beam_idx}',
                padp_ue_data, main_lobe_only,
                color='green', linewidth=2.0, side='ue'
            )
        
        # Phase 6: Hide unused axes
        for i in range(n_bs+1, len(axes_bs)):
            axes_bs[i].axis('off')
        
        for i in range(n_ue+1, len(axes_ue)):
            axes_ue[i].axis('off')
        
        # Phase 7: Layout & Save
        fig.suptitle(f'{ue_name} Beam Patterns (BS + UE)', 
                    fontsize=14, fontweight='bold', y=0.995)
        
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        
        output_path = os.path.join(self.config.output_dir, f'{ue_name}_bs_ue_patterns.png')
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"\n✓ Saved: {ue_name}_bs_ue_patterns.png\n")
    
    # === Orchestrator ===
    
    def plot_individual_beams(self, ue_name, dft_bs_tx_beams, padp_data, main_lobe_only=True):
        """Plot 26 subplots with detailed per-subplot verification info"""
        n_layer_beams = len(dft_bs_tx_beams)
        
        # PADP info
        dom_cluster = padp_data[0]
        padp_elev = dom_cluster['elev']
        padp_az = dom_cluster['az']
        
        print(f"\n{'='*60}")
        print(f"Layer-by-Layer Analysis:")
        print(f"{'='*60}\n")
        
        # Phase 1: Calculate & Analyze all patterns
        patterns = []
        infos = []
        
        for idx, (layer_idx, beam_idx) in enumerate(dft_bs_tx_beams):
            pattern = self.calculate_layer_beam_pattern(layer_idx, beam_idx)
            patterns.append(pattern)
            
            info = self.extract_pattern_info(pattern, padp_elev, padp_az)
            infos.append(info)
            
            print(f"[{idx+1}/{n_layer_beams}] Layer {layer_idx} + Beam {beam_idx}")
            print(f"  Peak direction: elev={info['peak_elev']:.1f}°, az={info['peak_az']:.1f}°")
            print(f"  Gain at PADP: {info['padp_gain']:.2f}")
            print(f"  Angular distance to PADP: {info['angular_distance']:.1f}°\n")
        
        # Phase 2: Calculate & Analyze total pattern
        print(f"{'='*60}")
        total_pattern = self.calculate_total_pattern(dft_bs_tx_beams)
        total_info = self.extract_pattern_info(total_pattern, padp_elev, padp_az)
        
        print(f"[{n_layer_beams+1}/{n_layer_beams+1}] Total P1O ({n_layer_beams} layers)")
        print(f"  Peak direction: elev={total_info['peak_elev']:.1f}°, az={total_info['peak_az']:.1f}°")
        print(f"  Gain at PADP: {total_info['padp_gain']:.2f}")
        print(f"  Angular distance to PADP: {total_info['angular_distance']:.1f}°")
        print(f"{'='*60}\n")
        
        # Phase 3: Create plots
        n_plots = n_layer_beams + 1
        fig, axes = self._create_vertical_polar_subplots(n_plots)
        
        # Plot individual layer-beams
        for idx, (layer_idx, beam_idx) in enumerate(dft_bs_tx_beams):
            title = f'Layer {layer_idx} + Beam {beam_idx}'
            self._plot_pattern_on_axis(axes[idx], patterns[idx], title, 
                                       padp_data, main_lobe_only, 
                                       color='blue', linewidth=2.0)
        
        # Plot total
        ax_total = axes[-1]
        title_total = f'Total P1O ({n_layer_beams} layers)'
        self._plot_pattern_on_axis(ax_total, total_pattern, title_total, 
                                   padp_data, main_lobe_only, 
                                   color='red', linewidth=3.0)
        
        # Overall title
        fig.suptitle(f'{ue_name} BS TX Beam Patterns (Layer-by-Layer)', 
                    fontsize=14, fontweight='bold', y=0.995)
        
        # Layout adjustment (v1과 동일)
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        
        # Save
        output_path = os.path.join(self.config.output_dir, f'{ue_name}_bs_patterns.png')
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)


# ===== Main Execution =====
def main():
    """Run P1O polar plot regeneration"""
    config = ReplotConfig()
    
    print("="*80)
    print("P1O Polar Plot Regeneration (v2 - Multi-Layer SOLID)")
    print("="*80)
    print(f"Output: {config.output_dir}")
    print()
    
    # Initialize modules
    analyzer = ClusterAnalyzer(config)
    plotter = PolarPlotter(config)
    
    # Parse CSV results
    parser = CSVResultParser(config)
    results = parser.load_all_stage4_results()
    
    if not results:
        print("No Stage 4 results found. Exiting.")
        return
    
    print(f"\nProcessing {len(results)} UEs...\n")
    
    for i, (area, freq, ue, ue_beams, bs_beams) in enumerate(results, 1):
        print(f"{'#'*70}")
        print(f"{'#'*70}")
        print(f"[{i}/{len(results)}] UE {ue}")
        print(f"{'#'*70}")
        print(f"{'#'*70}\n")
        
        try:
            # Load ray data
            ray_data = analyzer.load_ray_data(area, freq, ue)
            
            # Group by cluster
            clusters = analyzer.group_by_path_idx(ray_data)
            
            # PADP clusters (BS TX)
            padp_bs_tx_clusters = {}
            for cid, cluster in clusters.items():
                padp_bs_tx_clusters[cid] = analyzer.compute_cluster_stats(cluster, side='tx')
            
            # PADP clusters (UE RX)
            padp_ue_rx_clusters = {}
            for cid, cluster in clusters.items():
                padp_ue_rx_clusters[cid] = analyzer.compute_cluster_stats(cluster, side='rx')
            
            # Summary
            beam_counts = Counter([beam_idx for _, beam_idx in bs_beams])
            total_layers = len(bs_beams)
            
            print(f"P1O Selected: {total_layers} layer-beam pairs")
            for beam_idx in sorted(beam_counts.keys()):
                print(f"  Beam {beam_idx}: {beam_counts[beam_idx]} layers")
            
            # Prepare BS PADP data
            clusters_sorted_bs = sorted(
                padp_bs_tx_clusters.items(),
                key=lambda x: x[1]['total_power'],
                reverse=True
            )
            
            padp_data_bs = []
            for cluster_id, stats in clusters_sorted_bs:
                if not np.isfinite(stats.get('power_dB', -np.inf)):
                    continue
                padp_data_bs.append({
                    'cluster_id': cluster_id,
                    'elev': 90.0 - np.degrees(stats['theta_lcs_mean']),
                    'az': np.degrees(stats['phi_lcs_mean']),
                    'elev_spread': stats.get('theta_spread', 10.0),
                    'az_spread': np.degrees(stats.get('phi_spread', np.radians(15.0))),
                    'power_dB': stats['power_dB']
                })
            
            # Prepare UE PADP data
            clusters_sorted_ue = sorted(
                padp_ue_rx_clusters.items(),
                key=lambda x: x[1]['total_power'],
                reverse=True
            )
            
            padp_data_ue = []
            for cluster_id, stats in clusters_sorted_ue:
                if not np.isfinite(stats.get('power_dB', -np.inf)):
                    continue
                padp_data_ue.append({
                    'cluster_id': cluster_id,
                    'elev': 90.0 - np.degrees(stats['theta_lcs_mean']),
                    'az': np.degrees(stats['phi_lcs_mean']),
                    'elev_spread': stats.get('theta_spread', 10.0),
                    'az_spread': np.degrees(stats.get('phi_spread', np.radians(15.0))),
                    'power_dB': stats['power_dB']
                })
            
            print(f"\nPADP BS Clusters: {len(padp_data_bs)} valid clusters")
            print(f"PADP UE Clusters: {len(padp_data_ue)} valid clusters")
            
            if padp_data_bs:
                dom_bs = padp_data_bs[0]
                print(f"  Dominant BS: Cluster {dom_bs['cluster_id']}")
                print(f"    Direction: elev={dom_bs['elev']:.1f}°, az={dom_bs['az']:.1f}°")
                print(f"    Power: {dom_bs['power_dB']:.1f} dB")
            
            if padp_data_ue:
                dom_ue = padp_data_ue[0]
                print(f"  Dominant UE: Cluster {dom_ue['cluster_id']}")
                print(f"    Direction: elev={dom_ue['elev']:.1f}°, az={dom_ue['az']:.1f}°")
                print(f"    Power: {dom_ue['power_dB']:.1f} dB")
            
            # Convert UE beams to TRX-level format (assume TRX 0 for all beams)
            ue_trx_beams = [(0, beam_idx) for beam_idx in ue_beams]
            
            # Generate 2-column polar plot
            if padp_data_bs and padp_data_ue:
                ue_name = f"Area{area}_{freq}GHz_UE{ue}"
                plotter.plot_2column_layout(
                    ue_name=ue_name,
                    bs_layer_beams=bs_beams,
                    ue_trx_beams=ue_trx_beams,
                    padp_bs_data=padp_data_bs,
                    padp_ue_data=padp_data_ue,
                    main_lobe_only=config.main_lobe_only
                )
            else:
                print(f"WARNING: Missing PADP clusters (BS: {len(padp_data_bs)}, UE: {len(padp_data_ue)})\n")
            
        except Exception as e:
            print(f"ERROR processing UE {ue}: {e}\n")
            import traceback
            traceback.print_exc()
            continue
        
        print("\n" + "="*80)
    print("Processing complete!")
    print("="*80)


if __name__ == "__main__":
    main()
