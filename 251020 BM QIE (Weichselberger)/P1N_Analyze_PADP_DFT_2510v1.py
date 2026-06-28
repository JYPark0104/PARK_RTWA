#!/usr/bin/env python3
"""
P1N: PADP-DFT Codebook Relationship Analysis

Unified script for analyzing relationship between P1A ray tracing PADP 
(Power-Angle-Delay Profile) and P1L DFT codebook beam selection.

Analysis: Area 1, 7.5 GHz, 1033 UEs
Output: P1N_PADP_DFT_Analysis_Results/
"""

# ===== SECTION 1: Environment Setup =====
import os
import sys
import time
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

# ===== SECTION 2: Configuration =====
class P1N_Config:
    """Centralized configuration for P1N analysis"""
    
    def __init__(self):
        # Base paths
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.results_dir = os.path.join(self.script_dir, "P1N_PADP_DFT_Analysis_Results")
        
        # Input paths
        self.P1A_DIR = os.path.join(self.script_dir, "P1A_RT_Results")
        self.P1L_DIR = os.path.join(self.script_dir, "P1L_BeamMgmt_Results")
        self.P1L_CSV = "Area1_7.5GHz_SU_BM_20251019_215413.csv"  # Current in-progress (v3 format)
        
        # Output paths
        self.docs_dir = os.path.join(self.results_dir, "00_Documentation")
        self.ref_dir = os.path.join(self.results_dir, "01_Reference_Data")
        self.analysis_dir = os.path.join(self.results_dir, "02_Analysis_Results")
        self.viz_dir = os.path.join(self.results_dir, "03_Visualizations")
        self.polar_dir = os.path.join(self.viz_dir, "per_ue_polar")
        self.stats_dir = os.path.join(self.results_dir, "04_Statistical_Summary")
        
        # System parameters
        self.area = 1
        self.freq = 7.5
        
        # DFT codebook parameters
        self.N_BS, self.K_BS = 4, 2  # BS: 4x4 UPA, oversample 2 → 64 beams
        self.N_UE, self.K_UE = 2, 2  # UE: 2x2 UPA, oversample 2 → 16 beams
        
        # Analysis parameters
        self.sigma_b = 15.0  # Gaussian kernel width (degrees)
        self.top_k = 10  # Top-K rays for alignment
        self.theta_tol = 15.0  # Alignment tolerance (degrees)
        
        # Antenna orientations (from P1F)
        self.TX_Orientation = {
            "azimuth_deg": 246,
            "downtilt_deg": 3,
            "roll_deg": 0
        }
        self.RX_Orientation = {
            "azimuth_deg": 0,
            "elevation_deg": 0,
            "roll_deg": 0
        }
        
        # Visualization parameters
        self.n_polar_plots = 10  # Number of representative UE polar plots
        
        # Ensure directories exist
        for d in [self.analysis_dir, self.viz_dir, self.polar_dir, self.stats_dir]:
            os.makedirs(d, exist_ok=True)
    
    def print_config(self):
        """Print configuration summary"""
        print(f"P1N Configuration:")
        print(f"  Area: {self.area}, Frequency: {self.freq} GHz")
        print(f"  BS Codebook: {self.N_BS}x{self.N_BS} UPA, K={self.K_BS} → {(self.N_BS*self.K_BS)**2} beams")
        print(f"  UE Codebook: {self.N_UE}x{self.N_UE} UPA, K={self.K_UE} → {(self.N_UE*self.K_UE)**2} beams")
        print(f"  Results: {self.results_dir}")

# ===== SECTION 3: Utility Functions =====
def rotation_matrix(alpha_rad, beta_rad, gamma_rad):
    """
    3GPP TS 38.901 rotation matrix (Sec 7.1-4)
    
    Args:
        alpha_rad: Azimuth (bearing) [rad]
        beta_rad: Downtilt (elevation tilt) [rad]
        gamma_rad: Roll [rad]
    
    Returns:
        R: 3x3 rotation matrix
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

def gcs_to_lcs(theta_gcs_rad, phi_gcs_rad, orientation_deg):
    """
    Convert GCS angles to LCS angles using antenna orientation
    
    Args:
        theta_gcs_rad: Zenith angle in GCS [rad]
        phi_gcs_rad: Azimuth angle in GCS [rad]
        orientation_deg: {'azimuth_deg', 'downtilt_deg'/'elevation_deg', 'roll_deg'}
    
    Returns:
        (theta_lcs_rad, phi_lcs_rad): Angles in LCS [rad]
    """
    # Extract orientation angles
    if 'downtilt_deg' in orientation_deg:
        alpha = np.radians(orientation_deg['azimuth_deg'])
        beta = np.radians(orientation_deg['downtilt_deg'])
    elif 'elevation_deg' in orientation_deg:
        alpha = np.radians(orientation_deg['azimuth_deg'])
        beta = np.radians(orientation_deg['elevation_deg'])
    else:
        alpha = 0.0
        beta = 0.0
    gamma = np.radians(orientation_deg.get('roll_deg', 0))
    
    # Unit sphere vector in GCS
    st, ct = np.sin(theta_gcs_rad), np.cos(theta_gcs_rad)
    sp, cp = np.sin(phi_gcs_rad), np.cos(phi_gcs_rad)
    rho_gcs = np.array([st * cp, st * sp, ct])
    
    # Rotate to LCS (inverse rotation)
    R = rotation_matrix(alpha, beta, gamma)
    R_inv = R.T  # Orthogonal matrix: R^-1 = R^T
    rho_lcs = R_inv @ rho_gcs
    
    # Convert back to spherical
    rho_lcs_norm = np.linalg.norm(rho_lcs)
    if rho_lcs_norm < 1e-10:
        return (0.0, 0.0)
    
    rho_lcs = rho_lcs / rho_lcs_norm
    theta_lcs_rad = np.arccos(np.clip(rho_lcs[2], -1.0, 1.0))
    phi_lcs_rad = np.arctan2(rho_lcs[1], rho_lcs[0])
    
    return (theta_lcs_rad, phi_lcs_rad)

def beam_index_to_angle(j, N, K):
    """
    DFT beam index → (θ_az, θ_el) angles
    
    Original P1N formula (validated for P1L uplink scenario)
    
    Formula: sin(θ) = j / (NK)
    
    Args:
        j: Beam index [0, (NK)²-1]
        N: Antennas per dimension
        K: Oversample factor
    
    Returns:
        (theta_az_deg, theta_el_deg)
    """
    NK = N * K
    j_az = j % NK
    j_el = j // NK
    
    sin_theta_az = j_az / NK
    sin_theta_el = j_el / NK
    
    # Handle numerical precision
    sin_theta_az = np.clip(sin_theta_az, -1.0, 1.0)
    sin_theta_el = np.clip(sin_theta_el, -1.0, 1.0)
    
    theta_az_deg = np.degrees(np.arcsin(sin_theta_az))
    theta_el_deg = np.degrees(np.arcsin(sin_theta_el))
    
    return (theta_az_deg, theta_el_deg)

def angular_distance_spherical(theta1, phi1, theta2, phi2):
    """
    Great circle distance between two points on unit sphere
    
    Args:
        theta1, phi1: Zenith, azimuth of point 1 (degrees)
        theta2, phi2: Zenith, azimuth of point 2 (degrees)
    
    Returns:
        Angular distance (degrees)
    """
    theta1_rad = np.radians(theta1)
    phi1_rad = np.radians(phi1)
    theta2_rad = np.radians(theta2)
    phi2_rad = np.radians(phi2)
    
    cos_dist = (np.sin(theta1_rad) * np.sin(theta2_rad) * np.cos(phi1_rad - phi2_rad) +
                np.cos(theta1_rad) * np.cos(theta2_rad))
    cos_dist = np.clip(cos_dist, -1.0, 1.0)
    
    dist_rad = np.arccos(cos_dist)
    return np.degrees(dist_rad)

def compute_weighted_mean(values, weights):
    """Power-weighted mean"""
    w_sum = np.sum(weights)
    if w_sum == 0:
        return np.nan
    return np.sum(values * weights) / w_sum

def compute_weighted_std(values, weights, mean=None):
    """Power-weighted standard deviation"""
    w_sum = np.sum(weights)
    if w_sum == 0:
        return np.nan
    if mean is None:
        mean = compute_weighted_mean(values, weights)
    variance = np.sum(weights * (values - mean)**2) / w_sum
    return np.sqrt(variance)

# ===== SECTION 4: Data Loading =====
class RayDataLoader:
    """Load P1A ray data and P1L beam selection results"""
    
    def __init__(self, config):
        self.config = config
        self._rays_cache = None  # Cache for NPZ file
    
    def _load_rays_npz(self):
        """Load unified rays NPZ file (cached)"""
        if self._rays_cache is None:
            npz_path = os.path.join(self.config.P1A_DIR, 
                                   f"Area{self.config.area}_{self.config.freq}GHz_Rays_ALL_RXs.npz")
            self._rays_cache = np.load(npz_path)
        return self._rays_cache
    
    def load_p1l_csv(self):
        """
        Load P1L CSV (v3 format: one row per UE with ue_beams, bs_beams, C_AE, C_S2)
        
        Returns:
            DataFrame with columns: ue, ue_beams, bs_beams, C_AE, C_S2
        """
        csv_path = os.path.join(self.config.P1L_DIR, self.config.P1L_CSV)
        df = pd.read_csv(csv_path)
        
        print(f"  Loaded {len(df)} UEs from {os.path.basename(csv_path)}")
        print(f"  Columns: {', '.join(df.columns)}")
        
        return df
    
    def load_p1a_ray_data(self, ue_idx):
        """
        Load P1A ray data for single UE from unified NPZ file
        
        Args:
            ue_idx: UE index (RX index, 1-based)
        
        Returns:
            dict with keys: phi_t_deg, theta_t_deg, phi_r_deg, theta_r_deg, power, los_nlos_flag
        """
        rays = self._load_rays_npz()
        
        # Filter rays for this RX
        mask = rays['rx_indices'] == ue_idx
        if not np.any(mask):
            raise ValueError(f"No rays found for UE {ue_idx}")
        
        ray_data = {
            'phi_t_deg': rays['phi_t_deg'][mask].ravel(),
            'theta_t_deg': rays['theta_t_deg'][mask].ravel(),
            'phi_r_deg': rays['phi_r_deg'][mask].ravel(),
            'theta_r_deg': rays['theta_r_deg'][mask].ravel(),
            'power': rays['power'][mask].ravel(),
            'los_nlos_flag': rays['los_nlos_flag'][mask].ravel()
        }
        
        return ray_data
    
    def parse_beam_string(self, beam_str):
        """Parse beam indices from string format"""
        if pd.isna(beam_str) or beam_str == '':
            return []
        # Assume comma-separated: "4,4,4,3"
        return [int(b) for b in str(beam_str).split(',')]

# ===== SECTION 5: Alignment Analysis =====
class AlignmentAnalyzer:
    """Compute beam-ray alignment metrics"""
    
    def __init__(self, config):
        self.config = config
    
    def compute_beam_angles(self, beam_indices, N, K):
        """Convert beam indices to angles"""
        return [beam_index_to_angle(j, N, K) for j in beam_indices]
    
    def compute_alignment_gaussian(self, ray_angles, beam_angles, power):
        """
        Gaussian kernel alignment: ρ = Σ P_i · max_j exp(-Δθ²/(2σ²)) / Σ P_i
        
        Args:
            ray_angles: [(θ, φ), ...] in degrees
            beam_angles: [(θ, φ), ...] in degrees
            power: [P_1, P_2, ...]
        
        Returns:
            rho_align: Alignment score [0, 1]
        """
        P_total = np.sum(power)
        if P_total == 0 or len(beam_angles) == 0:
            return 0.0
        
        rho_sum = 0.0
        for i, (theta_r, phi_r) in enumerate(ray_angles):
            max_kernel = 0.0
            for (theta_b, phi_b) in beam_angles:
                delta_theta = angular_distance_spherical(theta_r, phi_r, theta_b, phi_b)
                kernel_val = np.exp(-delta_theta**2 / (2 * self.config.sigma_b**2))
                max_kernel = np.maximum(max_kernel, kernel_val)
            rho_sum += float(power[i]) * float(max_kernel)
        
        return rho_sum / P_total
    
    def compute_top_k_alignment(self, ray_angles, beam_angles, power, K=10):
        """
        Top-K ray alignment: fraction of top-K rays within θ_tol of any beam
        
        Returns:
            Score [0, 1]
        """
        if len(ray_angles) == 0 or len(beam_angles) == 0:
            return 0.0
        
        # Sort rays by power
        sorted_idx = np.argsort(power)[::-1]
        top_k_idx = sorted_idx[:min(K, len(sorted_idx))]
        
        aligned_count = 0
        for idx in top_k_idx:
            theta_r, phi_r = ray_angles[idx]
            distances = [angular_distance_spherical(theta_r, phi_r, theta_b, phi_b) 
                        for theta_b, phi_b in beam_angles]
            min_dist = float(np.min(distances))
            if min_dist < self.config.theta_tol:
                aligned_count += 1
        
        return aligned_count / len(top_k_idx)
    
    def compute_los_alignment(self, los_ray_angle, beam_angles):
        """
        Check if LoS ray aligns with any beam
        
        Returns:
            1 if aligned, 0 otherwise
        """
        if los_ray_angle is None or len(beam_angles) == 0:
            return 0
        
        theta_los, phi_los = los_ray_angle
        distances = [angular_distance_spherical(theta_los, phi_los, theta_b, phi_b) 
                    for theta_b, phi_b in beam_angles]
        min_dist = float(np.min(distances))
        
        return 1 if min_dist < self.config.theta_tol else 0

# ===== SECTION 6: Angular Spread Analysis =====
class AngularSpreadAnalyzer:
    """Compute angular spread metrics"""
    
    def __init__(self, config):
        self.config = config
    
    def compute_spreads_per_ue(self, ray_data):
        """
        Compute angular spreads for BS and UE sides
        
        Returns:
            dict with spread metrics
        """
        phi_t = ray_data['phi_t_deg']
        theta_t = ray_data['theta_t_deg']
        phi_r = ray_data['phi_r_deg']
        theta_r = ray_data['theta_r_deg']
        power = ray_data['power']
        los_flag = ray_data['los_nlos_flag']
        
        # All rays
        phi_t_mean = compute_weighted_mean(phi_t, power)
        theta_t_mean = compute_weighted_mean(theta_t, power)
        phi_r_mean = compute_weighted_mean(phi_r, power)
        theta_r_mean = compute_weighted_mean(theta_r, power)
        
        sigma_phi_t = compute_weighted_std(phi_t, power, phi_t_mean)
        sigma_theta_t = compute_weighted_std(theta_t, power, theta_t_mean)
        sigma_phi_r = compute_weighted_std(phi_r, power, phi_r_mean)
        sigma_theta_r = compute_weighted_std(theta_r, power, theta_r_mean)
        
        # NLoS only
        nlos_mask = np.array(los_flag != 1, dtype=bool)
        if np.any(nlos_mask):
            phi_t_nlos_mean = compute_weighted_mean(phi_t[nlos_mask], power[nlos_mask])
            theta_t_nlos_mean = compute_weighted_mean(theta_t[nlos_mask], power[nlos_mask])
            phi_r_nlos_mean = compute_weighted_mean(phi_r[nlos_mask], power[nlos_mask])
            theta_r_nlos_mean = compute_weighted_mean(theta_r[nlos_mask], power[nlos_mask])
            
            sigma_phi_t_nlos = compute_weighted_std(phi_t[nlos_mask], power[nlos_mask], phi_t_nlos_mean)
            sigma_theta_t_nlos = compute_weighted_std(theta_t[nlos_mask], power[nlos_mask], theta_t_nlos_mean)
            sigma_phi_r_nlos = compute_weighted_std(phi_r[nlos_mask], power[nlos_mask], phi_r_nlos_mean)
            sigma_theta_r_nlos = compute_weighted_std(theta_r[nlos_mask], power[nlos_mask], theta_r_nlos_mean)
        else:
            sigma_phi_t_nlos = sigma_theta_t_nlos = sigma_phi_r_nlos = sigma_theta_r_nlos = np.nan
        
        return {
            'phi_t_mean': phi_t_mean,
            'theta_t_mean': theta_t_mean,
            'phi_r_mean': phi_r_mean,
            'theta_r_mean': theta_r_mean,
            'sigma_phi_t': sigma_phi_t,
            'sigma_theta_t': sigma_theta_t,
            'sigma_phi_r': sigma_phi_r,
            'sigma_theta_r': sigma_theta_r,
            'sigma_phi_t_nlos': sigma_phi_t_nlos,
            'sigma_theta_t_nlos': sigma_theta_t_nlos,
            'sigma_phi_r_nlos': sigma_phi_r_nlos,
            'sigma_theta_r_nlos': sigma_theta_r_nlos
        }

# ===== SECTION 7: Rician K-factor =====
class RicianKFactorCalculator:
    """Compute Rician K-factor and classify UE type"""
    
    def compute_k_factor(self, power, los_nlos_flag):
        """
        Compute K-factor: K_dB = 10*log10(P_LoS / P_NLoS)
        
        Returns:
            K_factor_dB, P_los, P_nlos, n_rays_los, n_rays_nlos
        """
        los_mask = np.array(los_nlos_flag == 1, dtype=bool)
        nlos_mask = np.array(~los_mask, dtype=bool)
        
        P_los = np.sum(power[los_mask]) if np.any(los_mask) else 0.0
        P_nlos = np.sum(power[nlos_mask]) if np.any(nlos_mask) else 0.0
        n_rays_los = np.sum(los_mask)
        n_rays_nlos = np.sum(nlos_mask)
        
        if P_nlos > 0:
            K_factor_dB = 10 * np.log10(P_los / P_nlos) if P_los > 0 else -np.inf
        else:
            K_factor_dB = np.inf if P_los > 0 else np.nan
        
        return K_factor_dB, P_los, P_nlos, n_rays_los, n_rays_nlos
    
    def classify_ue_type(self, k_factor_db):
        """Classify UE: LoS (K>10), Mixed (0-10), NLoS (K<0)"""
        if np.isnan(k_factor_db) or np.isinf(k_factor_db):
            return 'Unknown'
        elif k_factor_db > 10:
            return 'LoS'
        elif k_factor_db >= 0:
            return 'Mixed'
        else:
            return 'NLoS'

# ===== SECTION 8: Visualization =====
class Visualizer:
    """Generate visualizations"""
    
    def __init__(self, config):
        self.config = config
    
    def plot_ue_polar(self, ue_idx, ray_data, bs_beams, ue_beams):
        """Generate polar plot for single UE with GCS->LCS coordinate transformation"""
        fig, (ax_bs, ax_ue) = plt.subplots(1, 2, subplot_kw=dict(projection='polar'), figsize=(12, 5))
        
        power = ray_data['power']
        power_norm = power / np.max(power)
        
        # BS side: Transform rays from GCS to LCS
        phi_t_lcs = []
        theta_t_lcs = []
        for theta_gcs_deg, phi_gcs_deg in zip(ray_data['theta_t_deg'], ray_data['phi_t_deg']):
            theta_lcs_rad, phi_lcs_rad = gcs_to_lcs(
                np.radians(theta_gcs_deg), 
                np.radians(phi_gcs_deg), 
                self.config.TX_Orientation
            )
            phi_t_lcs.append(phi_lcs_rad)
            theta_t_lcs.append(theta_lcs_rad)
        
        phi_t_lcs = np.array(phi_t_lcs)
        theta_t_lcs = np.array(theta_t_lcs)
        elevation_t_lcs = 90.0 - np.degrees(theta_t_lcs)  # zenith to elevation
        
        ax_bs.scatter(phi_t_lcs, elevation_t_lcs, c=power_norm, 
                     s=20, alpha=0.6, cmap='hot', zorder=1)
        
        # Overlay BS beams (LCS) - use hollow circles
        bs_angles = [beam_index_to_angle(b, self.config.N_BS, self.config.K_BS) for b in bs_beams]
        for theta_az, theta_el in bs_angles:
            phi_rad = np.radians(theta_az)
            ax_bs.plot(phi_rad, theta_el, 'o', markersize=12, 
                      markerfacecolor='none', markeredgecolor='blue', 
                      markeredgewidth=2.5, zorder=2)
        
        ax_bs.set_title(f'UE {ue_idx} - BS Side (LCS)', fontsize=12)
        ax_bs.set_theta_zero_location('N')
        ax_bs.set_theta_direction(-1)
        ax_bs.set_ylim([-90, 90])
        
        # UE side: NO transformation needed (UE orientation is identity)
        # Ray angles already in UE's local frame
        phi_r_rad = np.radians(ray_data['phi_r_deg'])
        theta_r_rad = np.radians(ray_data['theta_r_deg'])
        elevation_r = 90.0 - np.degrees(theta_r_rad)
        
        ax_ue.scatter(phi_r_rad, elevation_r, c=power_norm, 
                     s=20, alpha=0.6, cmap='hot', zorder=1)
        
        # Overlay UE beams (LCS) - use hollow squares
        ue_angles = [beam_index_to_angle(b, self.config.N_UE, self.config.K_UE) for b in ue_beams]
        for theta_az, theta_el in ue_angles:
            phi_rad = np.radians(theta_az)
            ax_ue.plot(phi_rad, theta_el, 's', markersize=10, 
                      markerfacecolor='none', markeredgecolor='green', 
                      markeredgewidth=2.5, zorder=2)
        
        ax_ue.set_title(f'UE {ue_idx} - UE Side (LCS)', fontsize=12)
        ax_ue.set_theta_zero_location('N')
        ax_ue.set_theta_direction(-1)
        ax_ue.set_ylim([-90, 90])
        
        plt.tight_layout()
        save_path = os.path.join(self.config.polar_dir, f'ue_{ue_idx}_polar_plot.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_beam_usage_heatmap(self, results_df, side='bs'):
        """Plot beam usage heatmap"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        n_beams = 64 if side == 'bs' else 16
        beam_usage = np.zeros(n_beams, dtype=int)
        
        for beams_str in results_df[f'{side}_beams_str']:
            if pd.notna(beams_str):
                beams = [int(b) for b in str(beams_str).split(',')]
                for b in beams:
                    if 0 <= b < n_beams:
                        beam_usage[b] += 1
        
        # Plot as bar chart
        ax.bar(range(n_beams), beam_usage, color='steelblue')
        ax.set_xlabel('Beam Index', fontsize=12)
        ax.set_ylabel('Usage Count', fontsize=12)
        ax.set_title(f'{"BS" if side == "bs" else "UE"} Beam Usage Distribution', fontsize=14)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.config.viz_dir, f'{side}_beam_usage.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_spread_vs_loss(self, results_df):
        """Scatter plot: angular spread vs capacity loss"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Color by UE type
        colors = {'LoS': 'blue', 'Mixed': 'green', 'NLoS': 'red', 'Unknown': 'gray'}
        
        for ue_type, color in colors.items():
            mask = results_df['ue_type'] == ue_type
            if np.any(mask):
                ax.scatter(results_df.loc[mask, 'sigma_phi_r'], 
                          results_df.loc[mask, 'loss_pct'],
                          c=color, label=ue_type, alpha=0.6, s=30)
        
        ax.set_xlabel('Angular Spread $\sigma_{\phi}$ (degrees)', fontsize=12)
        ax.set_ylabel('Capacity Loss (%)', fontsize=12)
        ax.set_title('Angular Spread vs Capacity Loss', fontsize=14)
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.config.viz_dir, 'spread_vs_loss.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_alignment_cdf(self, results_df):
        """CDF of alignment metrics"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # BS alignment
        bs_align = results_df['rho_align_bs'].dropna().sort_values()
        cdf_bs = np.arange(1, len(bs_align) + 1) / len(bs_align)
        ax.plot(bs_align, cdf_bs, label='BS Alignment', linewidth=2)
        
        # UE alignment
        ue_align = results_df['rho_align_ue'].dropna().sort_values()
        cdf_ue = np.arange(1, len(ue_align) + 1) / len(ue_align)
        ax.plot(ue_align, cdf_ue, label='UE Alignment', linewidth=2)
        
        ax.set_xlabel('Alignment Score $\\rho_{align}$', fontsize=12)
        ax.set_ylabel('CDF', fontsize=12)
        ax.set_title('Cumulative Distribution of Alignment Scores', fontsize=14)
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        save_path = os.path.join(self.config.viz_dir, 'alignment_cdf.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

# ===== SECTION 9: Statistical Analysis =====
class StatisticalAnalyzer:
    """Compute statistical summaries and correlations"""
    
    def __init__(self, config):
        self.config = config
    
    def compute_correlations(self, df):
        """Compute correlations between spread and loss"""
        valid_mask = df['sigma_phi_r'].notna() & df['loss_pct'].notna()
        if np.sum(valid_mask) < 3:
            return {'pearson_r': np.nan, 'pearson_p': np.nan, 
                   'spearman_r': np.nan, 'spearman_p': np.nan}
        
        sigma = df.loc[valid_mask, 'sigma_phi_r']
        loss = df.loc[valid_mask, 'loss_pct']
        
        pearson_r, pearson_p = pearsonr(sigma, loss)
        spearman_r, spearman_p = spearmanr(sigma, loss)
        
        return {
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p
        }
    
    def compare_los_vs_nlos(self, df):
        """Compare metrics by UE type"""
        summary = {}
        for ue_type in ['LoS', 'Mixed', 'NLoS']:
            mask = df['ue_type'] == ue_type
            if np.sum(mask) == 0:
                continue
            
            summary[ue_type] = {
                'count': np.sum(mask),
                'rho_align_bs_mean': df.loc[mask, 'rho_align_bs'].mean(),
                'rho_align_ue_mean': df.loc[mask, 'rho_align_ue'].mean(),
                'sigma_phi_r_mean': df.loc[mask, 'sigma_phi_r'].mean(),
                'loss_pct_mean': df.loc[mask, 'loss_pct'].mean(),
                'K_factor_mean': df.loc[mask, 'K_factor_dB'].mean()
            }
        
        return summary
    
    def generate_summary_stats(self, df):
        """Generate overall summary statistics"""
        numeric_cols = ['rho_align_bs', 'rho_align_ue', 'sigma_phi_r', 'sigma_theta_r',
                        'K_factor_dB', 'loss_pct']
        
        summary = df[numeric_cols].describe()
        return summary

# ===== SECTION 10: Main Pipeline =====
class P1N_AnalysisPipeline:
    """Main analysis pipeline"""
    
    def __init__(self, config):
        self.config = config
        self.loader = RayDataLoader(config)
        self.alignment_analyzer = AlignmentAnalyzer(config)
        self.spread_analyzer = AngularSpreadAnalyzer(config)
        self.k_factor_calc = RicianKFactorCalculator()
        self.visualizer = Visualizer(config)
        self.stats_analyzer = StatisticalAnalyzer(config)
    
    def process_single_ue(self, ue_idx, p1l_row):
        """Process single UE: compute all metrics"""
        # Load ray data
        ray_data = self.loader.load_p1a_ray_data(ue_idx)
        
        # Parse beam indices from v3 CSV format
        ue_beams = self.loader.parse_beam_string(p1l_row['ue_beams'])
        bs_beams = self.loader.parse_beam_string(p1l_row['bs_beams'])
        
        # K-factor
        K_dB, P_los, P_nlos, n_los, n_nlos = self.k_factor_calc.compute_k_factor(
            ray_data['power'], ray_data['los_nlos_flag'])
        ue_type = self.k_factor_calc.classify_ue_type(K_dB)
        
        # Prepare ray angles: (elevation, azimuth) format
        # Convert zenith to elevation: elevation = 90° - zenith
        # Beam angles return (theta_az, theta_el) in same convention
        ray_angles_bs = [(90.0 - theta, phi) for theta, phi in zip(ray_data['theta_t_deg'], ray_data['phi_t_deg'])]
        ray_angles_ue = [(90.0 - theta, phi) for theta, phi in zip(ray_data['theta_r_deg'], ray_data['phi_r_deg'])]
        
        # Beam angles (returns (azimuth_deg, elevation_deg))
        bs_beam_angles = self.alignment_analyzer.compute_beam_angles(
            bs_beams, self.config.N_BS, self.config.K_BS) if bs_beams else []
        ue_beam_angles = self.alignment_analyzer.compute_beam_angles(
            ue_beams, self.config.N_UE, self.config.K_UE) if ue_beams else []
        
        # Alignment metrics
        rho_bs = self.alignment_analyzer.compute_alignment_gaussian(
            ray_angles_bs, bs_beam_angles, ray_data['power']) if bs_beam_angles else 0.0
        rho_ue = self.alignment_analyzer.compute_alignment_gaussian(
            ray_angles_ue, ue_beam_angles, ray_data['power']) if ue_beam_angles else 0.0
        
        top_k_bs = self.alignment_analyzer.compute_top_k_alignment(
            ray_angles_bs, bs_beam_angles, ray_data['power']) if bs_beam_angles else 0.0
        top_k_ue = self.alignment_analyzer.compute_top_k_alignment(
            ray_angles_ue, ue_beam_angles, ray_data['power']) if ue_beam_angles else 0.0
        
        # LoS alignment
        los_flag = ray_data['los_nlos_flag']
        los_mask = np.array(los_flag == 1, dtype=bool)
        if np.any(los_mask):
            los_idx = np.where(los_mask)[0][0]
            los_angle_bs = (ray_data['theta_t_deg'][los_idx], ray_data['phi_t_deg'][los_idx])
            los_angle_ue = (ray_data['theta_r_deg'][los_idx], ray_data['phi_r_deg'][los_idx])
            los_align_bs = self.alignment_analyzer.compute_los_alignment(los_angle_bs, bs_beam_angles)
            los_align_ue = self.alignment_analyzer.compute_los_alignment(los_angle_ue, ue_beam_angles)
        else:
            los_align_bs = los_align_ue = 0
        
        # Angular spreads
        spreads = self.spread_analyzer.compute_spreads_per_ue(ray_data)
        
        # Capacity metrics from v3 CSV
        C_AE = p1l_row['C_AE'] if 'C_AE' in p1l_row else np.nan
        C_S2 = p1l_row['C_S2'] if 'C_S2' in p1l_row else np.nan
        loss_pct = 100.0 * (1 - C_S2 / C_AE) if (C_AE > 0 and not np.isnan(C_AE)) else np.nan
        
        result = {
            'ue': ue_idx,
            'n_rays_total': len(ray_data['power']),
            'n_rays_los': n_los,
            'n_rays_nlos': n_nlos,
            'P_total': np.sum(ray_data['power']),
            'P_los': P_los,
            'P_nlos': P_nlos,
            'K_factor_dB': K_dB,
            'ue_type': ue_type,
            'n_ue_beams': len(ue_beams),
            'n_bs_beams': len(bs_beams),
            'rho_align_bs': rho_bs,
            'rho_align_ue': rho_ue,
            'top10_align_bs': top_k_bs,
            'top10_align_ue': top_k_ue,
            'los_align_bs': los_align_bs,
            'los_align_ue': los_align_ue,
            'C_AE': C_AE,
            'C_S2': C_S2,
            'loss_pct': loss_pct,
            'bs_beams_str': ','.join(map(str, bs_beams)),
            'ue_beams_str': ','.join(map(str, ue_beams)),
            **spreads
        }
        
        return result, ray_data, bs_beams, ue_beams
    
    def run_full_analysis(self):
        """Execute full analysis pipeline"""
        print("="*70)
        print("P1N: PADP-DFT Codebook Relationship Analysis")
        print("="*70)
        self.config.print_config()
        print()
        
        start_time = time.time()
        
        # Step 1: Load P1L CSV
        print("[1/5] Loading P1L beam management results...")
        p1l_df = self.loader.load_p1l_csv()
        n_ues = len(p1l_df)
        print(f"  Found {n_ues} unique UEs")
        print()
        
        # Step 2: Process all UEs
        print("[2/5] Processing UEs (alignment + spread analysis)...")
        results = []
        
        for idx, row in p1l_df.iterrows():
            ue = int(row['ue'])
            
            if (idx + 1) % 100 == 0:
                elapsed = time.time() - start_time
                rate = (idx + 1) / elapsed
                eta = (n_ues - idx - 1) / rate if rate > 0 else 0
                print(f"  Progress: {idx+1}/{n_ues} UEs ({100*(idx+1)/n_ues:.1f}%) | "
                      f"Rate: {rate:.1f} UE/s | ETA: {eta/60:.1f} min")
            
            try:
                result, ray_data, bs_beams, ue_beams = self.process_single_ue(ue, row)
                results.append(result)
                
            except Exception as e:
                import traceback
                print(f"  Warning: UE {ue} failed: {e}")
                if idx < 3:  # Print detailed trace for first few failures
                    traceback.print_exc()
        
        print(f"  Successfully processed {len(results)}/{n_ues} UEs")
        print()
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Save results
        print("[3/5] Saving analysis results...")
        alignment_path = os.path.join(self.config.analysis_dir, 'alignment_summary.csv')
        spread_path = os.path.join(self.config.analysis_dir, 'angular_spread_summary.csv')
        
        # Alignment summary
        alignment_cols = ['ue', 'n_rays_total', 'n_rays_los', 'n_rays_nlos', 
                         'P_total', 'P_los', 'P_nlos', 'K_factor_dB', 'ue_type',
                         'n_ue_beams', 'n_bs_beams', 'rho_align_bs', 'rho_align_ue',
                         'top10_align_bs', 'top10_align_ue', 'los_align_bs', 'los_align_ue',
                         'C_AE', 'C_S2', 'loss_pct']
        results_df[alignment_cols].to_csv(alignment_path, index=False)
        print(f"  Saved: {alignment_path}")
        
        # Angular spread summary
        spread_cols = ['ue', 'phi_t_mean', 'theta_t_mean', 'sigma_phi_t', 'sigma_theta_t',
                      'phi_r_mean', 'theta_r_mean', 'sigma_phi_r', 'sigma_theta_r',
                      'sigma_phi_t_nlos', 'sigma_theta_t_nlos', 
                      'sigma_phi_r_nlos', 'sigma_theta_r_nlos',
                      'C_AE', 'C_S2', 'loss_pct']
        results_df[spread_cols].to_csv(spread_path, index=False)
        print(f"  Saved: {spread_path}")
        print()
        
        # Step 4: Generate visualizations
        print("[4/5] Generating visualizations...")
        
        # Per-UE polar plots: Select top UEs by alignment
        print(f"  Selecting top {self.config.n_polar_plots} UEs by alignment...")
        results_df['avg_alignment'] = (results_df['rho_align_bs'] + results_df['rho_align_ue']) / 2
        top_ues = results_df.nlargest(self.config.n_polar_plots, 'avg_alignment')['ue'].tolist()
        print(f"  Top UEs: {sorted(top_ues)}")
        
        # Reload top UEs for polar plotting
        print(f"  Generating {len(top_ues)} polar plots...")
        for ue in top_ues:
            row = p1l_df[p1l_df['ue'] == ue].iloc[0]
            try:
                _, ray_data, bs_beams, ue_beams = self.process_single_ue(ue, row)
                self.visualizer.plot_ue_polar(ue, ray_data, bs_beams, ue_beams)
            except Exception as e:
                print(f"    Warning: Polar plot for UE {ue} failed: {e}")
        
        # Beam usage
        print("  Generating beam usage plots...")
        self.visualizer.plot_beam_usage_heatmap(results_df, side='bs')
        self.visualizer.plot_beam_usage_heatmap(results_df, side='ue')
        
        # Spread vs loss
        print("  Generating spread vs loss plot...")
        self.visualizer.plot_spread_vs_loss(results_df)
        
        # Alignment CDF
        print("  Generating alignment CDF...")
        self.visualizer.plot_alignment_cdf(results_df)
        print()
        
        # Step 5: Statistical analysis
        print("[5/5] Computing statistical analysis...")
        
        # Correlations
        corr = self.stats_analyzer.compute_correlations(results_df)
        print(f"  Pearson correlation (spread vs loss): r={corr['pearson_r']:.3f}, p={corr['pearson_p']:.3e}")
        print(f"  Spearman correlation (spread vs loss): ρ={corr['spearman_r']:.3f}, p={corr['spearman_p']:.3e}")
        
        # LoS vs NLoS comparison
        los_nlos_comp = self.stats_analyzer.compare_los_vs_nlos(results_df)
        
        # Save statistical summary
        stats_summary_path = os.path.join(self.config.stats_dir, 'summary_statistics.csv')
        summary_stats = self.stats_analyzer.generate_summary_stats(results_df)
        summary_stats.to_csv(stats_summary_path)
        print(f"  Saved: {stats_summary_path}")
        
        # Save correlation results
        corr_path = os.path.join(self.config.stats_dir, 'correlation_analysis.txt')
        with open(corr_path, 'w') as f:
            f.write("Correlation Analysis: Angular Spread vs Capacity Loss\n")
            f.write("="*60 + "\n\n")
            f.write(f"Pearson correlation: r = {corr['pearson_r']:.4f}, p = {corr['pearson_p']:.3e}\n")
            f.write(f"Spearman correlation: ρ = {corr['spearman_r']:.4f}, p = {corr['spearman_p']:.3e}\n")
        print(f"  Saved: {corr_path}")
        
        # Save LoS vs NLoS comparison
        los_nlos_path = os.path.join(self.config.stats_dir, 'los_vs_nlos_comparison.txt')
        with open(los_nlos_path, 'w') as f:
            f.write("LoS vs NLoS Comparison\n")
            f.write("="*60 + "\n\n")
            for ue_type, metrics in los_nlos_comp.items():
                f.write(f"{ue_type} UEs (n={metrics['count']}):\n")
                f.write(f"  BS Alignment: {metrics['rho_align_bs_mean']:.3f}\n")
                f.write(f"  UE Alignment: {metrics['rho_align_ue_mean']:.3f}\n")
                f.write(f"  Angular Spread: {metrics['sigma_phi_r_mean']:.2f}°\n")
                f.write(f"  Capacity Loss: {metrics['loss_pct_mean']:.1f}%\n")
                f.write(f"  K-factor: {metrics['K_factor_mean']:.1f} dB\n\n")
        print(f"  Saved: {los_nlos_path}")
        print()
        
        # Final summary
        elapsed_total = time.time() - start_time
        print("="*70)
        print("Analysis Complete!")
        print(f"  Total time: {elapsed_total/60:.1f} minutes")
        print(f"  UEs processed: {len(results)}")
        print(f"  Results directory: {self.config.results_dir}")
        print("="*70)

# ===== MAIN =====
def main():
    """Main entry point"""
    config = P1N_Config()
    pipeline = P1N_AnalysisPipeline(config)
    
    try:
        pipeline.run_full_analysis()
    except Exception as e:
        print(f"\nERROR: Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())

