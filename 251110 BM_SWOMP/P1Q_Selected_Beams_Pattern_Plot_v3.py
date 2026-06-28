"""
P1Q 선택된 Beam들의 방향성 패턴 및 모든 Ray 시각화

**목적:**
- P1P CSV 파일들에서 선택된 beam 인덱스 추출
- 각 beam의 방향성 패턴과 P1A의 모든 rays (LoS/NLoS) 오버레이
- Azimuth/Elevation cut 시각화
- Eigenbeam 계산 및 시각화

**실행 모드:**
1. Manual Mode (manual_mode = True)
   - 수동으로 UE와 빔 지정
   - 저장 경로: P1Q_Project/Manual_Beam_Patterns/
   
2. Auto Mode - P1P (manual_mode = False)
   - P1P CSV에서 자동 로드 (전체)
   - 저장 경로: P1Q_Project/P1P_Beam_Patterns/

**주요 클래스:**
1. P1QResultsLoader: CSV 파일에서 UE/Beam 추출 (P1P만 지원)
2. RayDataLoader: P1A에서 LoS/NLoS rays 로딩
3. BeamPatternPlotter: Beam/Eigenbeam 시각화
4. P1P 클래스들 임포트: BeamDomainCapacity, BeamDomainTransform, BeamformingMatrix, DFTCodebook
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path
from typing import List, Tuple
import pandas as pd
import os

# TensorFlow (for Wen2011 optimization)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# P1P 클래스들 임포트
from P1P_BM_SWOMP_2511v3 import (
    BeamDomainCapacity,
    BeamDomainTransform,
    BeamformingMatrix,
    DFTCodebook,
    P1P_Config as P1P_Config_Ref
)


class P1Q_Config:
    """P1Q 플롯 설정 (중요도 순 배치)"""
    
    def __init__(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # ===== 1. 실행 모드 설정 (MASTER CONTROL) =====
        self.manual_mode = False  # True: Manual mode, False: Auto mode
        
        # Manual mode 설정 (manual_mode=True일 때만 사용)
        self.manual_ue = None
        self.manual_ue_beams = []
        self.manual_bs_beams = []
        
        # ===== 2. 경로 설정 =====
        self.P1A_INPUT_DIR = os.path.join(script_dir, "P1A_RT_Results")
        self.P1I_INPUT_DIR = os.path.join(script_dir, "P1I_Weichsel_Chunk_Results")
        self.P1P_INPUT_DIR = os.path.join(script_dir, "P1P_Project")
        self.P1Q_OUTPUT_DIR = os.path.join(script_dir, "P1Q_Project")
        os.makedirs(self.P1Q_OUTPUT_DIR, exist_ok=True)
        
        # ===== 3. 데이터 소스 설정 =====
        self.default_area = 1
        self.default_freq = 7.5  # GHz
        
        # P1P CSV files (for Auto mode)
        self.csv_files_p1p = [
            "A1_7.5GHz_P1P_1107_1447.csv"
        ]
        
        # Runtime variables (자동 설정됨, 직접 수정 금지)
        self.csv_input_dir = None
        self.csv_files = None
        self.output_dir = None
        
        # ===== 4. 하드웨어 구조 (안테나 배열) =====
        # UE hardware (N_UE,*) - 1 TRX = 1 Layer
        self.n_ue_layer = 4
        self.n_ue_layer_ae = 4
        self.n_ue_ae = self.n_ue_layer * self.n_ue_layer_ae  # 16
        self.n_ue_layer_row = 2
        self.n_ue_layer_col = 2
        
        # BS hardware (N_BS,*)
        self.n_bs_layer = 64
        self.n_bs_layer_ae = 16
        self.n_bs_ae = self.n_bs_layer * self.n_bs_layer_ae  # 1024
        self.n_bs_layer_row = 4
        self.n_bs_layer_col = 4
        self.n_bs_full_row = 32  # BS full array
        
        # Physical parameters
        self.antenna_spacing = 0.5  # λ
        self.bs_orientation = (
            np.radians(246.0),  # alpha
            np.radians(93.0),   # theta (3° downtilt)
            np.radians(0.0)     # gamma
        )
        
        # ===== 5. DFT 코드북 파라미터 =====
        # Codebook (N_*,Beams)
        self.n_oversample = 2
        self.n_ue_layer_beams = (self.n_oversample * self.n_ue_layer_row) ** 2  # 16
        self.n_bs_layer_beams = (self.n_oversample * self.n_bs_layer_row) ** 2  # 64
        
        # Selection (r_*)
        self.r_mode = self.n_ue_layer  # Number of modes for Two-Stage selection (MIMO rank)
        
        # ===== 6. Wen2011 최적화 (채널 정규화 + 용량 계산) =====
        self.SNR_dB = 10.0
        self.SNR_linear = 10.0 ** (self.SNR_dB / 10.0)
        self.outer_max_iter = 50
        self.outer_eps = 0.1
        self.inner_max_iter = 50
        self.regularization = 1e-20
        
        # ===== 7. 빔 패턴 계산 해상도 =====
        self.azimuth_step = 1.0  # deg
        self.elevation_step = 1.0  # deg
        self.beam_max_gain_azimuth_step = 5.0  # deg
        self.beam_max_gain_elevation_step = 5.0  # deg
        
        # ===== 8. 시각화 범위 =====
        self.power_vmin = -180  # dB
        self.power_vmax = -80   # dB
        self.ylim_range = 30.0  # dB
        
        # ===== 9. 시각화 크기 =====
        self.figure_width = 12  # inches
        self.figure_height_per_beam = 5  # inches
        self.figure_dpi = 150
        
        # ===== 10. Ray 시각화 스타일 =====
        self.phi_spread_threshold_azimuth = 30.0  # deg
        self.theta_spread_threshold_zenith = 30.0  # deg
        self.individual_ray_width = 3.0  # deg
        self.ray_height_min = 5.0  # deg
        self.ray_height_max = 30.0  # deg
        self.los_alpha_individual = 0.6
        self.los_edgecolor = 'black'
        self.los_linewidth_individual = 0.5
        self.nlos_alpha_individual = 0.5
        self.nlos_edgecolor = 'none'
        self.nlos_linewidth_individual = 0
        self.los_alpha_wedge = 0.8
        self.los_linewidth_wedge = 2
        self.nlos_alpha_wedge = 0.6
        self.nlos_linewidth_wedge = 1
    
    def initialize(self):
        """
        실행 모드에 따라 runtime variables 초기화
        
        실행 전 반드시 호출해야 함
        
        공통 사항:
        - 모든 모드에서 P1A (Ray data), P1I (Weichselberger) 사용
        
        차이점:
        - Manual mode: CSV 사용 안함, 수동 UE/빔 지정, Manual_Beam_Patterns 저장
        - Auto mode: CSV에서 자동 로드, P1P 폴더 저장
        """
        if self.manual_mode:
            # Manual mode: CSV 사용 안함, P1A/P1I 사용
            self.csv_input_dir = None
            self.csv_files = None
            self.output_dir = os.path.join(self.P1Q_OUTPUT_DIR, "Manual_Beam_Patterns")
            os.makedirs(self.output_dir, exist_ok=True)
        else:
            # Auto mode: CSV + P1A + P1I 모두 사용 (P1P만 지원)
            self.csv_input_dir = self.P1P_INPUT_DIR
            self.csv_files = self.csv_files_p1p
            self.output_dir = os.path.join(self.P1Q_OUTPUT_DIR, "P1P_Beam_Patterns")
            os.makedirs(self.output_dir, exist_ok=True)


class P1QResultsLoader:
    """
    CSV 파일에서 UE/Beam 인덱스 추출 (Auto mode 전용)
    
    P1P 형식만 지원:
    - P1P: bs_beams, ue_beams (전체 사용)
    """
    
    def __init__(self, config: P1Q_Config):
        self.config = config
        self.csv_dir = Path(config.csv_input_dir)
    
    # ===== Helper 메서드 =====
    
    @staticmethod
    def _parse_beam_string(beams_str: str) -> List[int]:
        """
        빔 문자열 파싱 헬퍼 (공통 로직)
        
        Args:
            beams_str: "1,2,3,..." or "[1,2,3,...]" 형식
        
        Returns:
            beams: [int] 빔 인덱스 리스트
        """
        beams_str = beams_str.strip('[]"')
        return [int(b.strip()) for b in beams_str.split(',') if b.strip()]
    
    # ===== Public 메서드 =====
    
    def load_all_ues(self, csv_files: List[str]) -> np.ndarray:
        """CSV 파일들에서 모든 UE 인덱스 추출"""
        return self.load_all_ues_p1p(csv_files)
    
    def load_all_beams_for_ue(self, csv_files: List[str], target_ue: int) -> np.ndarray:
        """특정 UE의 BS 빔 리스트 추출 (중복 포함)"""
        return self.load_all_beams_for_ue_p1p(csv_files, target_ue)
    
    def load_ue_beams_for_ue_p1p(self, csv_files: List[str], target_ue: int) -> np.ndarray:
        """
        P1P CSV에서 특정 UE의 UE 빔 인덱스 추출
        
        Args:
            csv_files: CSV 파일명 리스트
            target_ue: 대상 UE index
        
        Returns:
            ue_beams: [r_ue] UE beam indices (중복 없음)
        """
        for csv_file in csv_files:
            csv_path = self.csv_dir / csv_file
            if not csv_path.exists():
                continue
            
            df = pd.read_csv(csv_path)
            if 'ue' not in df.columns or 'ue_beams' not in df.columns:
                continue
            
            # Filter target UE
            target_df = df[df['ue'] == target_ue]
            
            if len(target_df) == 0:
                continue
            
            # Parse ue_beams (첫 번째 row만 사용, 중복 없음)
            ue_beams_str = target_df['ue_beams'].iloc[0]
            if pd.isna(ue_beams_str):
                continue
            
            ue_beams = self._parse_beam_string(ue_beams_str)
            return np.array(ue_beams)
        
        raise ValueError(f"UE {target_ue} ue_beams not found in P1P CSV files")
    
    # ===== P1P 전용 메서드 =====
    
    def load_all_ues_p1p(self, csv_files: List[str]) -> np.ndarray:
        """
        P1P CSV에서 모든 UE 인덱스 추출 (전체 row 사용)
        
        Args:
            csv_files: CSV 파일명 리스트
        
        Returns:
            all_ues: [M] Sorted unique UE indices
        """
        all_ues = []
        
        for csv_file in csv_files:
            csv_path = self.csv_dir / csv_file
            if not csv_path.exists():
                print(f"  Warning: {csv_file} not found")
                continue
            
            df = pd.read_csv(csv_path)
            if 'ue' not in df.columns:
                print(f"  Warning: {csv_file} missing 'ue' column")
                continue
            
            # P1P: Use all rows
            ues_in_file = df['ue'].unique().tolist()
            n_ues = len(ues_in_file)
            print(f"  {csv_file}: {n_ues} UEs")
            
            all_ues.extend(ues_in_file)
        
        unique_ues = np.unique(all_ues)
        print(f"  Total unique UEs: {len(unique_ues)}")
        
        return unique_ues
    
    def load_all_beams_for_ue_p1p(self, csv_files: List[str], target_ue: int) -> np.ndarray:
        """
        P1P CSV에서 특정 UE의 빔 리스트 추출 (중복 포함)
        
        Args:
            csv_files: CSV 파일명 리스트
            target_ue: 대상 UE index
        
        Returns:
            all_beams: [K] Full beam list with repetitions
        """
        ue_beams = []
        
        for csv_file in csv_files:
            csv_path = self.csv_dir / csv_file
            if not csv_path.exists():
                continue
            
            df = pd.read_csv(csv_path)
            if 'ue' not in df.columns or 'bs_beams' not in df.columns:
                continue
            
            # Filter target UE
            target_df = df[df['ue'] == target_ue]
            
            if len(target_df) == 0:
                continue
            
            # Parse beam list
            for beams_str in target_df['bs_beams'].dropna():
                ue_beams.extend(self._parse_beam_string(beams_str))
        
        if not ue_beams:
            raise ValueError(f"UE {target_ue} not found in P1P CSV files")
        
        return np.array(ue_beams)


class RayDataLoader:
    """P1A Ray 데이터 로딩 (LoS/NLoS 모두)"""
    
    def __init__(self, config: P1Q_Config):
        self.config = config
        # P1A 경로를 config에서 가져옴 (area, freq는 나중에 동적으로 구성)
        self.p1a_base_path = Path(config.P1A_INPUT_DIR)
    
    def load_rays_for_ue(self, ue: int, area: int = None, freq: float = None, 
                         include_los: bool = True, include_nlos: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        특정 UE의 rays 로딩 (GCS)
        
        Args:
            ue: UE index
            area: Area index (None: config.default_area)
            freq: Frequency in GHz (None: config.default_freq)
            include_los: LoS rays 포함 여부
            include_nlos: NLoS rays 포함 여부
        
        Returns:
            rays_gcs: [N, 4] (theta_deg, phi_deg, power, cluster_idx)
            los_flags: [N] (1=LoS, 0=NLoS)
            cluster_ids: [N] Cluster/path indices
        """
        if area is None:
            area = self.config.default_area
        if freq is None:
            freq = self.config.default_freq
        
        p1a_path = self.p1a_base_path / f"Area{area}_{freq}GHz_Rays_ALL_RXs.npz"
        rays_data = np.load(p1a_path)
        
        # UE 필터링
        mask = rays_data['rx_indices'] == ue
        theta_gcs = rays_data['theta_t_deg'][mask].ravel()
        phi_gcs = rays_data['phi_t_deg'][mask].ravel()
        power = rays_data['power'][mask].ravel()
        
        # Cluster index (source_path_idx)
        if 'source_path_idx' in rays_data:
            cluster_ids = rays_data['source_path_idx'][mask].ravel()
        else:
            cluster_ids = np.zeros_like(power, dtype=int)
        
        # LoS flag
        los_flag = rays_data.get('los_flag', None)
        if los_flag is not None:
            los_flags = los_flag[mask].ravel()
        else:
            los_flags = np.zeros_like(power)  # All NLoS if no flag
        
        # LoS/NLoS 필터링
        if include_los and include_nlos:
            # 모두 포함
            final_mask = np.ones_like(los_flags, dtype=bool)
        elif include_los:
            # LoS만
            final_mask = los_flags == 1
        elif include_nlos:
            # NLoS만
            final_mask = los_flags == 0
        else:
            # 아무것도 포함 안함
            final_mask = np.zeros_like(los_flags, dtype=bool)
        
        rays_gcs = np.stack([
            theta_gcs[final_mask],
            phi_gcs[final_mask],
            power[final_mask],
            cluster_ids[final_mask]
        ], axis=1)
        
        return rays_gcs, los_flags[final_mask], cluster_ids[final_mask]
    
    def transform_rays_to_lcs(self, rays_gcs: np.ndarray) -> np.ndarray:
        """
        GCS rays를 LCS로 변환
        
        Args:
            rays_gcs: [N, 4] (theta_gcs, phi_gcs, power, cluster_idx)
        
        Returns:
            rays_lcs: [N, 4] (theta_lcs, phi_lcs, power, cluster_idx)
        """
        alpha, theta_bs, gamma = self.config.bs_orientation
        
        rays_lcs = []
        for i in range(rays_gcs.shape[0]):
            theta_gcs_rad = np.radians(rays_gcs[i, 0])
            phi_gcs_rad = np.radians(rays_gcs[i, 1])
            power = rays_gcs[i, 2]
            cluster_idx = rays_gcs[i, 3]
            
            # GCS → LCS transformation
            theta_lcs_rad, phi_lcs_rad = self._gcs_to_lcs(
                theta_gcs_rad, phi_gcs_rad, alpha, theta_bs, gamma
            )
            
            rays_lcs.append([
                np.degrees(theta_lcs_rad),
                np.degrees(phi_lcs_rad),
                power,
                cluster_idx
            ])
        
        return np.array(rays_lcs)
    
    def _gcs_to_lcs(self, theta_gcs, phi_gcs, alpha, theta_bs, gamma):
        """GCS → LCS 변환 (3GPP TS 38.901)"""
        # GCS direction vector
        sin_theta = np.sin(theta_gcs)
        cos_theta = np.cos(theta_gcs)
        sin_phi = np.sin(phi_gcs)
        cos_phi = np.cos(phi_gcs)
        
        r_gcs = np.array([
            sin_theta * cos_phi,
            sin_theta * sin_phi,
            cos_theta
        ])
        
        # Rotation matrix (inverse)
        cos_a, sin_a = np.cos(alpha), np.sin(alpha)
        cos_b, sin_b = np.cos(theta_bs), np.sin(theta_bs)
        cos_g, sin_g = np.cos(gamma), np.sin(gamma)
        
        # R_inv = R(-alpha, pi-theta_bs, -gamma)
        R_inv = np.array([
            [cos_a*cos_b, -sin_a*cos_g + cos_a*sin_b*sin_g, sin_a*sin_g + cos_a*sin_b*cos_g],
            [sin_a*cos_b, cos_a*cos_g + sin_a*sin_b*sin_g, -cos_a*sin_g + sin_a*sin_b*cos_g],
            [-sin_b, cos_b*sin_g, cos_b*cos_g]
        ])
        
        r_lcs = R_inv @ r_gcs
        
        # LCS angles
        theta_lcs = np.arccos(np.clip(r_lcs[2], -1, 1))
        phi_lcs = np.arctan2(r_lcs[1], r_lcs[0])
        
        return theta_lcs, phi_lcs
    
    def load_weichselberger_for_ue(self, ue: int, area: int = None, freq: float = None):
        """
        특정 UE의 Weichselberger 파라미터 로딩 (P1I)
        
        Args:
            ue: UE index
            area: Area index (None: config.default_area)
            freq: Frequency in GHz (None: config.default_freq)
        
        Returns:
            U_bs: [1024, 1024] complex64 BS eigenvector
            U_ue: [16, 16] complex64 UE eigenvector
            Omega: [1024, 16] float32 coupling matrix (UPLINK: [BS, UE])
            H_mean: [1024, 16] complex64 mean channel (UPLINK: [BS, UE])
        """
        if area is None:
            area = self.config.default_area
        if freq is None:
            freq = self.config.default_freq
        
        # P1I chunk files
        p1i_dir = Path(self.config.P1I_INPUT_DIR)
        chunk_files = sorted(p1i_dir.glob(f"Area{area}_{freq}GHz_Weichsel_Chunk_*.npz"))
        
        # Find UE in chunks
        for chunk_file in chunk_files:
            data = np.load(chunk_file)
            ue_indices = data['ue_indices']
            
            if ue in ue_indices:
                # Find index in chunk
                idx_in_chunk = np.where(ue_indices == ue)[0][0]
                
                # Load Weichselberger parameters
                U_bs = data['P1G_U_BS'][idx_in_chunk]  # [1024, 1024]
                U_ue = data['P1G_U_UE'][idx_in_chunk]  # [16, 16]
                Omega = data['P1G_Omega'][idx_in_chunk]  # [16, 1024] (UL: [UE, BS])
                H_mean = data['P1H_H_mean'][idx_in_chunk]  # [16, 1024] (UL: [UE, BS])
                
                # Transpose to match UPLINK convention [BS, UE]
                Omega = Omega.T  # [1024, 16]
                H_mean = H_mean.T  # [1024, 16]
                
                # Normalize channel (Wen2011 requirement)
                Omega_norm, H_mean_norm, kappa = self._normalize_channel(
                    Omega, H_mean, U_bs.shape[0], U_ue.shape[0]
                )
                
                return (
                    tf.constant(U_bs, dtype=tf.complex64),
                    tf.constant(U_ue, dtype=tf.complex64),
                    tf.constant(Omega_norm, dtype=tf.float32),
                    tf.constant(H_mean_norm, dtype=tf.complex64)
                )
        
        raise ValueError(f"UE {ue} not found in P1I chunks for Area {area}, {freq} GHz")
    
    def _normalize_channel(self, Omega: np.ndarray, H_mean: np.ndarray,
                          n_bs: int, n_ue: int) -> tuple:
        """Wen2011 channel normalization (P1O method)
        
        Args:
            Omega: [n_bs, n_ue] coupling matrix (UPLINK)
            H_mean: [n_bs, n_ue] mean channel (UPLINK)
            n_bs: BS dimension
            n_ue: UE dimension
        
        Returns:
            Omega_norm: [n_bs, n_ue] normalized coupling matrix
            H_mean_norm: [n_bs, n_ue] normalized mean channel
            kappa: K-factor (P_los / P_nlos)
        """
        rho_target = self.config.SNR_linear
        N_M = n_bs * n_ue
        epsilon = 1e-30
        threshold = 1e-10
        
        P_nlos = np.sum(Omega)
        P_los = np.sum(np.abs(H_mean)**2)
        
        if P_los < threshold * P_nlos:
            # Rayleigh channel (NLoS only)
            kappa = 0.0
            Omega_norm = (N_M * rho_target / (P_nlos + epsilon)) * Omega
            H_mean_norm = np.zeros_like(H_mean)
        elif P_nlos < threshold * P_los:
            # Deterministic channel (LoS only)
            kappa = float('inf')
            Omega_norm = np.zeros_like(Omega)
            scale = np.sqrt(N_M * rho_target / (P_los + epsilon))
            H_mean_norm = scale * H_mean
        else:
            # Rician channel (LoS + NLoS)
            kappa = P_los / P_nlos
            P_nlos_target = N_M * (1.0 / (kappa + 1.0)) * rho_target
            P_los_target = N_M * (kappa / (kappa + 1.0)) * rho_target
            
            Omega_norm = (P_nlos_target / P_nlos) * Omega
            scale = np.sqrt(P_los_target / P_los)
            H_mean_norm = scale * H_mean
        
        return Omega_norm, H_mean_norm, kappa


class DFTCodebookGenerator:
    """2D DFT 코드북 생성 (P1O 호환)"""
    
    @staticmethod
    def generate_2d_dft_codebook(N: int, K: int) -> np.ndarray:
        """
        2D DFT codebook 생성
        
        Args:
            N: Antennas per dimension (4)
            K: Oversample factor (2)
        
        Returns:
            F: [N*N, (N*K)*(N*K)] = [16, 64] complex codebook
        """
        # 1D DFT [N, NK]
        F_1d = DFTCodebookGenerator._generate_1d_dft(N, K)
        
        # 2D Kronecker product
        F_2d = np.einsum('ij,kl->ikjl', F_1d, F_1d)
        F_2d = F_2d.reshape(N * N, (N * K) * (N * K))
        
        return F_2d
    
    @staticmethod
    def _generate_1d_dft(N: int, K: int) -> np.ndarray:
        """1D DFT [N, NK]"""
        NK = N * K
        indices_i = np.arange(N)
        indices_j = np.arange(NK)
        
        i_grid, j_grid = np.meshgrid(indices_i, indices_j, indexing='ij')
        
        phase = -2 * np.pi * i_grid * j_grid / NK
        F_1d = np.exp(1j * phase) / np.sqrt(N)
        
        return F_1d


class SteeringVectorGenerator:
    """Steering vector 생성 (column-major, +1j phase, 3GPP 38.901 element pattern)"""
    
    @staticmethod
    def _element_pattern_38901(theta_rad: float, phi_rad: float) -> float:
        """
        3GPP TS 38.901 Table 7.3-1 안테나 엘리먼트 패턴
        
        Args:
            theta_rad: Zenith angle [rad] (0~π)
            phi_rad: Azimuth angle [rad] (-π~π)
        
        Returns:
            element_gain: Linear scale element gain
        """
        # Parameters from 38.901 Table 7.3-1
        theta_3dB = np.radians(65)  # Vertical 3dB beamwidth
        phi_3dB = np.radians(65)    # Horizontal 3dB beamwidth
        a_max = 30                   # Maximum attenuation [dB]
        g_e_max = 8                  # Element maximum gain [dB]
        
        # Vertical pattern (referenced to θ = π/2)
        a_v = -min(12 * ((theta_rad - np.pi/2) / theta_3dB)**2, a_max)
        
        # Horizontal pattern (referenced to φ = 0)
        a_h = -min(12 * (phi_rad / phi_3dB)**2, a_max)
        
        # Combined pattern
        a_db = -min(-(a_v + a_h), a_max) + g_e_max
        
        return 10**(a_db / 10)
    
    @staticmethod
    def _ae_index_to_physical_coords(ae_idx: int) -> tuple:
        """AE index → (panel_row, panel_col, elem_row, elem_col)
        
        Column-major order at both levels
        
        Args:
            ae_idx: 0~1023
        
        Returns:
            (panel_row, panel_col, elem_row, elem_col)
        """
        panel_major = ae_idx // 16
        elem_in_panel = ae_idx % 16
        
        panel_row = panel_major % 8
        panel_col = panel_major // 8
        
        elem_row = elem_in_panel % 4
        elem_col = elem_in_panel // 4
        
        return panel_row, panel_col, elem_row, elem_col
    
    @staticmethod
    def _get_ae_physical_position(panel_row: int, panel_col: int,
                                  elem_row: int, elem_col: int) -> tuple:
        """물리적 위치 계산 (wavelengths)
        
        Panel spacing = 2.5λ
        Element spacing = 0.5λ
        
        Args:
            panel_row, panel_col: 0~7
            elem_row, elem_col: 0~3
        
        Returns:
            (y_pos, z_pos) in wavelengths
        """
        y_panel = panel_col * 2.5
        z_panel = panel_row * 2.5
        
        y_elem = elem_col * 0.5
        z_elem = elem_row * 0.5
        
        y_total = y_panel + y_elem
        z_total = z_panel + z_elem
        
        return y_total, z_total
    
    @staticmethod
    def generate(theta_zenith_rad: float, phi_rad: float, 
                 config: P1Q_Config) -> np.ndarray:
        """
        1024 AE steering vector (panel structure + 38.901 pattern)
        
        Args:
            theta_zenith_rad: Zenith angle [rad]
            phi_rad: Azimuth angle [rad]
            config: P1Q_Config
        
        Returns:
            h: [1024] complex steering vector with element pattern
        """
        k = 2 * np.pi
        
        # Zenith → Elevation
        theta_elev = np.pi / 2 - theta_zenith_rad
        
        sin_theta = np.sin(theta_elev)
        cos_theta = np.cos(theta_elev)
        sin_phi = np.sin(phi_rad)
        cos_phi = np.cos(phi_rad)
        
        # Element pattern gain (inverse conjugate for correct plotting)
        theta_pattern = np.pi - theta_zenith_rad
        phi_pattern = phi_rad + np.pi
        phi_pattern = np.arctan2(np.sin(phi_pattern), np.cos(phi_pattern))
        
        element_gain = SteeringVectorGenerator._element_pattern_38901(
            theta_pattern, phi_pattern
        )
        element_field = np.sqrt(element_gain)
        
        h = np.zeros(1024, dtype=complex)
        
        # Panel structure: 8×8 panels, each 4×4 AE
        for ae_idx in range(1024):
            p_row, p_col, e_row, e_col = SteeringVectorGenerator._ae_index_to_physical_coords(ae_idx)
            y, z = SteeringVectorGenerator._get_ae_physical_position(p_row, p_col, e_row, e_col)
            
            phase = k * (y * cos_theta * sin_phi + z * sin_theta)
            h[ae_idx] = element_field * np.exp(1j * phase)
        
        return h
    
    @staticmethod
    def compute_element_gain(theta_zenith_rad: float, phi_rad: float, config: P1Q_Config) -> float:
        """
        단일 안테나 요소의 이득 패턴 계산
        
        Args:
            theta_zenith_rad: Zenith angle [rad]
            phi_rad: Azimuth angle [rad]
            config: Configuration
        
        Returns:
            element_gain: Antenna element gain (linear scale)
        """
        cos_theta = np.cos(theta_zenith_rad)
        sin_theta = np.sin(theta_zenith_rad)
        sin_phi = np.sin(phi_rad)
        
        # BS orientation
        alpha, theta_rot, gamma = config.bs_orientation
        cos_alpha = np.cos(alpha)
        sin_alpha = np.sin(alpha)
        cos_theta_rot = np.cos(theta_rot)
        sin_theta_rot = np.sin(theta_rot)
        
        # Rotated coordinates
        cos_theta_prime = (cos_alpha * cos_theta_rot * cos_theta + 
                          cos_alpha * sin_theta_rot * sin_theta * sin_phi -
                          sin_alpha * sin_theta * np.cos(phi_rad))
        
        # Patch antenna element pattern: cos^3(theta')
        cos_theta_prime_clipped = np.clip(cos_theta_prime, 0.0, 1.0)
        element_gain = cos_theta_prime_clipped ** 3
        
        return element_gain


# ----- BeamformingMatrix -----
class BeamformingMatrix:
    """Block-diagonal beamforming matrix construction"""
    
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


class BeamPatternAnalyzer:
    """Beam 방향성 패턴 계산"""
    
    def __init__(self, config: P1Q_Config):
        self.config = config
        self.dft_gen = DFTCodebookGenerator()
        self.sv_gen = SteeringVectorGenerator()
        self.F = None  # Layer-level [16, 64]
    
    def _ensure_codebook(self):
        """DFT codebook 생성 (lazy initialization)"""
        if self.F is None:
            # BS Layer-level codebook [16, 64] only
            self.F = self.dft_gen.generate_2d_dft_codebook(
                self.config.n_bs_layer_row,
                self.config.n_oversample
            )
    
    def compute_azimuth_cut(self, beam_idx: int, elevation_deg: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Azimuth cut at fixed elevation
        
        Args:
            beam_idx: Beam index (layer-level)
            elevation_deg: Fixed elevation [deg]
        
        Returns:
            azimuth_vals: [M] Azimuth angles [deg]
            gain_vals: [M] Beam gains
        """
        self._ensure_codebook()
        
        # Expand layer-level beam to AE-level [1024]
        F_tf = tf.constant(self.F, dtype=tf.complex64)
        beam_indices_tf = tf.constant([beam_idx], dtype=tf.int32)
        W_BS = BeamformingMatrix.construct_beamforming_blockdiag_tf(
            F_tf, beam_indices_tf, self.config.n_bs_layer
        )
        w = W_BS[:, 0].numpy()  # [1024]
        
        azimuth_vals = np.arange(-180, 180, self.config.azimuth_step)
        gain_vals = np.zeros_like(azimuth_vals, dtype=float)
        
        theta_zenith_rad = np.radians(90.0 - elevation_deg)
        
        for i, phi_deg in enumerate(azimuth_vals):
            phi_rad = np.radians(phi_deg)
            h = self.sv_gen.generate(theta_zenith_rad, phi_rad, self.config)
            gain_vals[i] = np.abs(np.vdot(w, h))**2
        
        return azimuth_vals, gain_vals
    
    def compute_elevation_cut(self, beam_idx: int, azimuth_deg: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Elevation cut at fixed azimuth
        
        Args:
            beam_idx: Beam index (layer-level)
            azimuth_deg: Fixed azimuth [deg]
        
        Returns:
            elevation_vals: [M] Elevation angles [deg]
            gain_vals: [M] Beam gains
        """
        self._ensure_codebook()
        
        # Expand layer-level beam to AE-level [1024]
        F_tf = tf.constant(self.F, dtype=tf.complex64)
        beam_indices_tf = tf.constant([beam_idx], dtype=tf.int32)
        W_BS = BeamformingMatrix.construct_beamforming_blockdiag_tf(
            F_tf, beam_indices_tf, self.config.n_bs_layer
        )
        w = W_BS[:, 0].numpy()  # [1024]
        
        elevation_vals = np.arange(-90, 90, self.config.elevation_step)
        gain_vals = np.zeros_like(elevation_vals, dtype=float)
        
        phi_rad = np.radians(azimuth_deg)
        
        for i, elev_deg in enumerate(elevation_vals):
            theta_zenith_rad = np.radians(90.0 - elev_deg)
            h = self.sv_gen.generate(theta_zenith_rad, phi_rad, self.config)
            gain_vals[i] = np.abs(np.vdot(w, h))**2
        
        return elevation_vals, gain_vals
    
    def compute_ae_gain(self, theta_zenith_rad: float, phi_rad: float) -> float:
        """
        AE 패턴 이득 계산 - 3GPP 38.901 Table 7.3-1, LCS 기준
        Peak at (theta_zenith=π/2, phi=0), normalized to peak=1.0 (0dB)
        
        Args:
            theta_zenith_rad: Zenith angle [rad]
            phi_rad: Azimuth angle [rad]
        
        Returns:
            gain: AE pattern gain (linear scale, peak normalized to 1.0)
        """
        gain_with_8db_peak = self.sv_gen._element_pattern_38901(theta_zenith_rad, phi_rad)
        return gain_with_8db_peak / 10**(8.0 / 10)  # Normalize: 8dB peak → 0dB peak
    
    def compute_2d_pattern(self, beam_idx: int, 
                          azimuth_range: Tuple[float, float, float],
                          elevation_range: Tuple[float, float, float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        2D beam pattern 계산
        
        Args:
            beam_idx: Beam index (layer-level)
            azimuth_range: (min, max, step) [deg]
            elevation_range: (min, max, step) [deg]
        
        Returns:
            phi_grid: [M, N] Azimuth mesh
            elev_grid: [M, N] Elevation mesh
            gain_grid: [M, N] Beam gain
        """
        self._ensure_codebook()
        
        # Expand layer-level beam to AE-level [1024]
        F_tf = tf.constant(self.F, dtype=tf.complex64)
        beam_indices_tf = tf.constant([beam_idx], dtype=tf.int32)
        W_BS = BeamformingMatrix.construct_beamforming_blockdiag_tf(
            F_tf, beam_indices_tf, self.config.n_bs_layer
        )
        w = W_BS[:, 0].numpy()  # [1024]
        
        azimuth_vals = np.arange(*azimuth_range)
        elevation_vals = np.arange(*elevation_range)
        
        phi_grid, elev_grid = np.meshgrid(azimuth_vals, elevation_vals)
        gain_grid = np.zeros_like(phi_grid, dtype=float)
        
        for i in range(phi_grid.shape[0]):
            for j in range(phi_grid.shape[1]):
                phi_rad = np.radians(phi_grid[i, j])
                theta_zenith_rad = np.radians(90.0 - elev_grid[i, j])
                
                h = self.sv_gen.generate(theta_zenith_rad, phi_rad, self.config)
                gain_grid[i, j] = np.abs(np.vdot(w, h))**2
        
        return phi_grid, elev_grid, gain_grid
    
    def extract_beam_domain_params_dl(self, beam_indices: np.ndarray, ue_beams: np.ndarray,
                                      U_bs_full: tf.Tensor, U_ue_full: tf.Tensor,
                                      Omega_ul: tf.Tensor, H_mean_ul: tf.Tensor):
        """
        선택된 BS 빔에 대한 DL 빔 도메인 파라미터 추출 (P1P Stage 2와 동일)
        
        Args:
            beam_indices: [K] BS beam indices (layer-level)
            ue_beams: [r_ue] UE beam indices (TRX-level)
            U_bs_full: [1024, 1024] BS eigenvector
            U_ue_full: [16, 16] UE eigenvector
            Omega_ul: [1024, 16] coupling matrix UPLINK [RX=BS, TX=UE]
            H_mean_ul: [1024, 16] mean channel UPLINK [RX=BS, TX=UE]
        
        Returns:
            U_ue_beam_dl: [4, 4] UE eigenvector (DL beam domain)
            U_bs_beam_dl: [K, K] BS eigenvector (DL beam domain)
            Omega_beam_dl: [4, K] coupling matrix DL [RX=UE, TX=BS]
            H_mean_beam_dl: [4, K] mean channel DL [RX=UE, TX=BS]
            W_bs: [1024, K] BS beamforming matrix
            W_ue: [16, 4] UE beamforming matrix
        """
        self._ensure_codebook()
        K = len(beam_indices)
        
        # DL parameters (P1P Stage 2와 동일)
        Omega_dl = tf.transpose(Omega_ul)  # [16, 1024]
        H_mean_dl = tf.linalg.adjoint(H_mean_ul)  # [16, 1024]
        
        # W_BS [1024, K] - BS Layer codebook 사용
        F_bs_tf = tf.constant(self.F, dtype=tf.complex64)
        beam_indices_tf = tf.constant(beam_indices, dtype=tf.int32)
        W_bs = BeamformingMatrix.construct_beamforming_blockdiag_tf(
            F_bs_tf, beam_indices_tf, self.config.n_bs_layer
        )
        
        # W_UE [16, 4] - UE TRX codebook 사용 (P1P Stage 1과 동일)
        F_ue_trx = DFTCodebook.generate_2d_dft_codebook_tf(
            self.config.n_ue_layer_row,
            self.config.n_oversample
        )
        ue_beams_tf = tf.constant(ue_beams, dtype=tf.int32)
        W_ue = BeamformingMatrix.construct_beamforming_blockdiag_tf(
            F_ue_trx, ue_beams_tf, self.config.n_ue_layer
        )
        
        # Mean channel transform: [4, K] (RX, TX) order
        H_mean_beam_dl = tf.linalg.adjoint(W_ue) @ H_mean_dl @ W_bs
        
        # BS TX covariance (DL, selected K beams)
        _, U_bs_beam_dl, _ = BeamDomainTransform.transform_covariance_dual(
            U_bs_full, U_ue_full, Omega_dl, W_bs, W_ue,
            link_direction="DL", target_side="BS"
        )
        
        # UE RX covariance (DL, 4 beams fixed)
        _, U_ue_beam, _ = BeamDomainTransform.transform_covariance_dual(
            U_bs_full, U_ue_full, Omega_dl, W_bs, W_ue,
            link_direction="DL", target_side="UE"
        )
        
        # Coupling matrix (DL) - P1P Stage 2와 동일
        V_bs = tf.linalg.adjoint(U_bs_full) @ W_bs @ U_bs_beam_dl
        V_ue = tf.linalg.adjoint(U_ue_full) @ W_ue @ U_ue_beam
        V_bs_abs2 = tf.abs(V_bs)**2
        V_ue_abs2 = tf.abs(V_ue)**2
        
        # Omega_beam_dl: [4, K] (RX, TX) order
        Omega_beam_dl = tf.transpose(V_ue_abs2) @ Omega_dl @ V_bs_abs2
        
        return U_ue_beam, U_bs_beam_dl, Omega_beam_dl, H_mean_beam_dl, W_bs, W_ue
    
    def optimize_bs_transmit_dl(self, U_ue_beam: tf.Tensor, U_bs_beam: tf.Tensor,
                                Omega_beam_dl: tf.Tensor, H_mean_beam_dl: tf.Tensor):
        """
        DL BS transmit covariance 최적화 (Wen2011) - P1P 클래스 사용
        
        Args:
            U_ue_beam: [4, 4] UE eigenvector (DL RX)
            U_bs_beam: [K, K] BS eigenvector (DL TX)
            Omega_beam_dl: [4, K] coupling matrix [RX=UE, TX=BS]
            H_mean_beam_dl: [4, K] mean channel [RX=UE, TX=BS]
        
        Returns:
            C_bs: float, DL capacity
            P_bs: [K, K] BS transmit covariance
            eigvals: [K] eigenvalues (descending)
            eigvecs: [K, K] eigenvectors
        """
        # P1P Config 참조로 BeamDomainCapacity 생성
        p1p_config_ref = P1P_Config_Ref()
        capacity_calc = BeamDomainCapacity(p1p_config_ref)
        
        # Wen2011 DL optimization: UE receive, BS transmit
        C_bs, P_bs, k_final, eps_final = capacity_calc.wen2011_optimize_tf(
            U_ue_beam,          # [4, 4] RX: DL UE receive
            U_bs_beam,          # [K, K] TX: DL BS transmit (optimize)
            Omega_beam_dl,      # [4, K] [RX, TX] order
            H_mean_beam_dl      # [4, K] [RX, TX] order
        )
        
        # Eigenvalue decomposition
        eigvals, eigvecs = tf.linalg.eigh(P_bs)
        eigvals = tf.math.real(eigvals)
        
        # Sort descending
        sorted_indices = tf.argsort(eigvals, direction='DESCENDING')
        eigvals_sorted = tf.gather(eigvals, sorted_indices)
        eigvecs_sorted = tf.gather(eigvecs, sorted_indices, axis=1)
        
        return C_bs.numpy(), P_bs, eigvals_sorted.numpy(), eigvecs_sorted.numpy()
    
    def compute_final_beamforming_vectors(self, W_bs: tf.Tensor,
                                         eigvecs: np.ndarray, eigvals: np.ndarray) -> np.ndarray:
        """
        방향빔 blkdiag × 디지털 빔포밍 = 최종 빔포밍 벡터 [1024, K]
        
        Args:
            W_bs: [1024, K] BS beamforming matrix (directional beams)
            eigvecs: [K, K] eigenvectors from P_bs_opt
            eigvals: [K] eigenvalues from P_bs_opt
        
        Returns:
            W_final: [1024, K] final beamforming matrix
                     각 컬럼 w_k [1024]가 하나의 eigenbeam
        """
        # Digital beamforming [K, K]: eigenvectors × sqrt(eigenvalues)
        sqrt_eigvals = np.sqrt(np.maximum(eigvals, 0))
        D = eigvecs * sqrt_eigvals[np.newaxis, :]
        D_tf = tf.constant(D, dtype=tf.complex64)
        
        # W_final = W_BS @ D [1024, K]
        W_final = tf.matmul(W_bs, D_tf)
        
        return W_final.numpy()
    
    def compute_combined_beam_gain(self, theta_rad: float, phi_rad: float,
                                   W_final: np.ndarray) -> float:
        """
        최종 빔포밍 벡터들과 AE-level steering vector의 빔 이득
        
        Args:
            theta_rad: Zenith angle [rad]
            phi_rad: Azimuth angle [rad]
            W_final: [1024, K] final beamforming vectors
        
        Returns:
            gain: Total beam gain (linear scale)
        """
        # AE-level steering vector [1024]
        h = self.sv_gen.generate(theta_rad, phi_rad, self.config)
        
        # Total gain: sum_k |w_k^H @ h|^2
        total_gain = 0.0
        for k in range(W_final.shape[1]):
            w_k = W_final[:, k]  # [1024]
            gain_k = np.abs(np.vdot(w_k, h))**2
            total_gain += gain_k
        
        return total_gain


class BeamPatternPlotter:
    """Beam 패턴 시각화"""
    
    def __init__(self, beam_analyzer: BeamPatternAnalyzer, ray_loader: RayDataLoader):
        self.beam_analyzer = beam_analyzer
        self.ray_loader = ray_loader
        self.config = beam_analyzer.config
    
    def _get_cluster_color(self, cluster_id: int) -> tuple:
        """
        클러스터 인덱스를 색깔로 매핑 (최대 20개 구별)
        
        Args:
            cluster_id: Cluster index
        
        Returns:
            color: RGBA tuple
        """
        cmap = plt.get_cmap('tab20')
        return cmap(cluster_id % 20)
    
    def plot_codebook_beams(self, ue: int, beam_indices: np.ndarray, beam_counts: dict, output_path: str):
        """
        UE의 모든 beam에 대한 플롯 생성 (각 beam당 1 row, 2 columns)
        
        Args:
            ue: UE index
            beam_indices: [K] Beam indices for this UE
            beam_counts: dict {beam_idx: count}
            output_path: Output PNG file path
        """
        # Limit to first 4 beams
        MAX_BEAMS = 4
        if len(beam_indices) > MAX_BEAMS:
            beam_indices = beam_indices[:MAX_BEAMS]
            print(f"  Note: Limiting codebook plot to first {MAX_BEAMS} beams")
        
        # Load rays once
        rays_gcs, los_flags, cluster_ids = self.ray_loader.load_rays_for_ue(ue, include_los=True, include_nlos=True)
        rays_lcs = self.ray_loader.transform_rays_to_lcs(rays_gcs)
        
        # Compute power-weighted mean across ALL rays using circular averaging
        padp_theta, padp_phi = self._compute_circular_power_weighted_mean(rays_lcs)
        
        # Create figure (2 columns, K rows)
        n_beams = len(beam_indices)
        fig = plt.figure(figsize=(self.config.figure_width, 
                                  self.config.figure_height_per_beam * n_beams))
        
        for row_idx, beam_idx in enumerate(beam_indices):
            beam_count = beam_counts.get(beam_idx, 0)
            
            # Compute beam's global maximum gain for consistent normalization
            beam_max_gain = self._compute_codebook_beam_max_gain(beam_idx)
            
            # Subplot 1: Azimuth cut (at power-weighted mean zenith)
            ax1 = fig.add_subplot(n_beams, 2, row_idx * 2 + 1, projection='polar')
            self._plot_codebook_azimuth_cut(ax1, ue, beam_idx, beam_count, beam_max_gain, padp_theta, 
                                  rays_lcs, los_flags, cluster_ids, add_colorbar=(row_idx == 0))
            
            # Subplot 2: Zenith cut (at power-weighted mean azimuth)
            ax2 = fig.add_subplot(n_beams, 2, row_idx * 2 + 2, projection='polar')
            # Add colorbar only to the first row (top-right subplot)
            self._plot_codebook_zenith_cut(ax2, ue, beam_idx, beam_count, beam_max_gain, padp_phi, 
                                 rays_lcs, los_flags, cluster_ids, add_colorbar=False)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.config.figure_dpi, bbox_inches='tight')
        plt.close()
    
    def _compute_codebook_beam_max_gain(self, beam_idx: int) -> float:
        """
        코드북 빔의 전체 3D 패턴에서 최대 gain 계산
        
        Args:
            beam_idx: Beam index (layer-level)
        
        Returns:
            max_gain: Maximum gain across all directions (linear scale)
        """
        self.beam_analyzer._ensure_codebook()
        
        # Expand layer-level beam to AE-level [1024]
        F_tf = tf.constant(self.beam_analyzer.F, dtype=tf.complex64)
        beam_indices_tf = tf.constant([beam_idx], dtype=tf.int32)
        W_BS = BeamformingMatrix.construct_beamforming_blockdiag_tf(
            F_tf, beam_indices_tf, self.config.n_bs_layer
        )
        w = W_BS[:, 0].numpy()  # [1024]
        
        # Sample azimuth and elevation
        azimuth_vals = np.arange(-180, 180, self.config.beam_max_gain_azimuth_step)
        elevation_vals = np.arange(-90, 90, self.config.beam_max_gain_elevation_step)
        
        max_gain = 0.0
        for phi_deg in azimuth_vals:
            for elev_deg in elevation_vals:
                theta_zenith_rad = np.radians(90.0 - elev_deg)
                phi_rad = np.radians(phi_deg)
                h = self.beam_analyzer.sv_gen.generate(
                    theta_zenith_rad, phi_rad, self.beam_analyzer.config
                )
                gain = np.abs(np.vdot(w, h))**2
                max_gain = max(max_gain, gain)
        
        return max_gain
    
    def _compute_circular_power_weighted_mean(self, rays_lcs: np.ndarray) -> Tuple[float, float]:
        """
        전체 ray들을 대상으로 circular power weighted mean 계산
        
        Args:
            rays_lcs: [N_rays, 4] array [zenith_deg, azimuth_deg, power, cluster_idx]
        
        Returns:
            (mean_zenith_deg, mean_azimuth_deg): Circular power weighted mean [deg]
        """
        zenith_deg = rays_lcs[:, 0]  # [N_rays]
        azimuth_deg = rays_lcs[:, 1]  # [N_rays]
        powers = rays_lcs[:, 2]      # [N_rays]
        
        # Convert to radians for circular operations
        zenith_rad = np.radians(zenith_deg)
        azimuth_rad = np.radians(azimuth_deg)
        
        # Zenith: 직선 평균 (0~π 범위)
        mean_zenith_rad = np.average(zenith_rad, weights=powers)
        mean_zenith_deg = np.degrees(mean_zenith_rad)
        
        # Azimuth: Circular 평균 (-π~π 범위)
        x = powers * np.cos(azimuth_rad)
        y = powers * np.sin(azimuth_rad)
        
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_weights = np.sum(powers)
        
        if sum_weights > 0:
            mean_azimuth_rad = np.arctan2(sum_y, sum_x)
        else:
            mean_azimuth_rad = 0.0
        
        mean_azimuth_deg = np.degrees(mean_azimuth_rad)
        
        return mean_zenith_deg, mean_azimuth_deg
    
    def _plot_codebook_azimuth_cut(self, ax, ue: int, beam_idx: int, beam_count: int, beam_max_gain: float, zenith_deg: float,
                         rays_lcs: np.ndarray, los_flags: np.ndarray, cluster_ids: np.ndarray, 
                         add_colorbar: bool):
        """Codebook beam azimuth cut at fixed zenith (polar plot) with cluster wedges"""
        # Compute beam pattern
        elevation_deg = 90 - zenith_deg  # Convert zenith to elevation for beam analyzer
        azimuth_vals, gain_vals = self.beam_analyzer.compute_azimuth_cut(beam_idx, elevation_deg)
        
        # Normalize gain (dB) using beam's global maximum
        gain_db = 10 * np.log10(gain_vals + 1e-10)
        beam_max_gain_db = 10 * np.log10(beam_max_gain + 1e-10)
        gain_db_normalized = gain_db - beam_max_gain_db
        
        # Compute AE pattern (38.901, normalized to peak = 0dB)
        theta_zenith_rad = np.radians(zenith_deg)
        ae_gain_vals = np.array([self.beam_analyzer.compute_ae_gain(theta_zenith_rad, np.radians(phi)) 
                                 for phi in azimuth_vals])
        ae_gain_db = 10 * np.log10(ae_gain_vals + 1e-10)
        
        # Plot beam pattern (phase conjugate: azimuth → azimuth - 180°)
        azimuth_rad = np.radians(azimuth_vals - 180)
        ax.plot(azimuth_rad, gain_db_normalized, 'b-', linewidth=2, label=f'Beam {beam_idx}')
        
        # Plot AE pattern (LCS 기준, phase conjugate 없이)
        azimuth_rad_lcs = np.radians(azimuth_vals)
        ax.plot(azimuth_rad_lcs, ae_gain_db, 'k--', linewidth=1.5, label='AE Pattern', alpha=0.7)
        
        # Fixed ylim for codebook plot
        ylim_min = -60.0
        ylim_max = 0.0
        
        # Plot cluster wedges
        unique_clusters = np.unique(cluster_ids)
        
        for cluster_id in unique_clusters:
            # Get rays in this cluster
            cluster_mask = cluster_ids == cluster_id
            cluster_rays = rays_lcs[cluster_mask]
            cluster_los = los_flags[cluster_mask]
            
            # Extract angles and power
            theta_lcs = cluster_rays[:, 0]  # zenith
            phi_lcs = cluster_rays[:, 1]    # azimuth
            power = cluster_rays[:, 2]
            
            is_los = np.any(cluster_los)
            
            # Check if cluster has large spread
            phi_mean = np.arctan2(np.mean(np.sin(np.radians(phi_lcs))),
                                   np.mean(np.cos(np.radians(phi_lcs))))
            phi_mean_deg = np.degrees(phi_mean)
            phi_diff = np.angle(np.exp(1j * np.radians(phi_lcs - phi_mean_deg)), deg=True)
            phi_spread = np.max(phi_diff) - np.min(phi_diff)
            
            # If spread > threshold, draw individual rays with small width
            if phi_spread > self.config.phi_spread_threshold_azimuth:
                individual_ray_width = self.config.individual_ray_width
                
                for i in range(len(phi_lcs)):
                    phi_ray = phi_lcs[i]
                    power_ray = power[i]
                    
                    # Power (in dB)
                    power_ray_db = 10 * np.log10(power_ray + 1e-20)
                    
                    # Color by cluster index
                    color = self._get_cluster_color(cluster_id)
                    
                    # Plot individual ray (phase conjugate)
                    phi_center_rad = np.radians(phi_ray - 180)
                    width_rad = np.radians(individual_ray_width)
                    
                    # Height proportional to power
                    power_normalized = (power_ray_db - self.config.power_vmin) / (self.config.power_vmax - self.config.power_vmin)
                    power_normalized = np.clip(power_normalized, 0, 1)
                    height = -(self.config.ray_height_min + power_normalized * (self.config.ray_height_max - self.config.ray_height_min))
                    
                    # Style based on LoS
                    if is_los:
                        alpha = self.config.los_alpha_individual
                        edgecolor = self.config.los_edgecolor
                        linewidth = self.config.los_linewidth_individual
                    else:
                        alpha = self.config.nlos_alpha_individual
                        edgecolor = self.config.nlos_edgecolor
                        linewidth = self.config.nlos_linewidth_individual
                    
                    # Plot individual ray bar
                    ax.bar(x=phi_center_rad, height=height, width=width_rad, 
                           bottom=ylim_max, color=color, alpha=alpha,
                           edgecolor=edgecolor, linewidth=linewidth)
            else:
                # Normal case: draw as a single wedge
                # Power (average in dB)
                power_avg_db = 10 * np.log10(np.mean(power) + 1e-20)
                
                # Color by cluster index
                color = self._get_cluster_color(cluster_id)
                
                # Plot wedge (phase conjugate)
                phi_center_rad = np.radians(phi_mean_deg - 180)
                width_rad = np.radians(phi_spread)
                
                # Height proportional to power
                power_normalized = (power_avg_db - self.config.power_vmin) / (self.config.power_vmax - self.config.power_vmin)
                power_normalized = np.clip(power_normalized, 0, 1)
                height = -(self.config.ray_height_min + power_normalized * (self.config.ray_height_max - self.config.ray_height_min))
                
                # Style based on LoS
                if is_los:
                    alpha = self.config.los_alpha_wedge
                    edgecolor = self.config.los_edgecolor
                    linewidth = self.config.los_linewidth_wedge
                else:
                    alpha = self.config.nlos_alpha_wedge
                    edgecolor = self.config.los_edgecolor
                    linewidth = self.config.nlos_linewidth_wedge
                
                # Plot bar from outer edge inward
                ax.bar(x=phi_center_rad, height=height, width=width_rad, 
                       bottom=ylim_max, color=color, alpha=alpha,
                       edgecolor=edgecolor, linewidth=linewidth)
        
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_ylim(ylim_min, ylim_max)
        zenith_deg_conj = 180 - zenith_deg  # Phase conjugate
        ax.set_xlabel('Azimuth (deg, LCS)', fontsize=9)
        ax.set_ylabel('Normalized Gain (dB)', fontsize=9, rotation=0)
        ax.yaxis.set_label_coords(0.9, 1)
        ax.set_title(f'UE {ue} Beam {beam_idx} (×{beam_count}) Azimuth Cut\n(Zenith={zenith_deg_conj:.1f}°)', fontsize=10)
        ax.grid(True)
        ax.legend(fontsize=8, loc='center', bbox_to_anchor=(0.5, 0.25))  # lower center와 center 중간 세로 위치
        
        # Add colorbar (only if requested)
        if add_colorbar:
            unique_clusters = np.unique(cluster_ids)
            cmap = plt.get_cmap('tab20')
            n_clusters = len(unique_clusters)
            bounds = np.arange(n_clusters + 1) - 0.5
            norm = plt.matplotlib.colors.BoundaryNorm(bounds, cmap.N)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, pad=0.1, fraction=0.046)
            cbar.set_label('Cluster Index', fontsize=8)
            # Set ticks to cluster indices (not boundaries)
            cbar.set_ticks(unique_clusters)
    
    def _plot_codebook_zenith_cut(self, ax, ue: int, beam_idx: int, beam_count: int, beam_max_gain: float, azimuth_deg: float,
                        rays_lcs: np.ndarray, los_flags: np.ndarray, cluster_ids: np.ndarray,
                        add_colorbar: bool):
        """Codebook beam zenith cut at fixed azimuth (polar plot) with cluster wedges"""
        # Compute beam pattern (elevation-based internally)
        elevation_vals, gain_vals = self.beam_analyzer.compute_elevation_cut(beam_idx, azimuth_deg)
        
        # Normalize gain (dB) using beam's global maximum
        gain_db = 10 * np.log10(gain_vals + 1e-10)
        beam_max_gain_db = 10 * np.log10(beam_max_gain + 1e-10)
        gain_db_normalized = gain_db - beam_max_gain_db
        
        # Compute AE pattern (38.901, normalized to peak = 0dB)
        phi_rad = np.radians(azimuth_deg)
        ae_gain_vals = np.array([self.beam_analyzer.compute_ae_gain(np.radians(90 - elev), phi_rad) 
                                 for elev in elevation_vals])
        ae_gain_db = 10 * np.log10(ae_gain_vals + 1e-10)
        
        # Plot beam pattern (convert elevation to zenith, phase conjugate)
        zenith_vals = 90 - elevation_vals  # Convert elevation to zenith
        zenith_rad = np.radians(180 - zenith_vals)  # Phase conjugate: zenith → 180° - zenith
        ax.plot(zenith_rad, gain_db_normalized, 'b-', linewidth=2, label=f'Beam {beam_idx}')
        
        # Plot AE pattern (LCS 기준, phase conjugate 없이)
        zenith_rad_lcs = np.radians(zenith_vals)
        ax.plot(zenith_rad_lcs, ae_gain_db, 'k--', linewidth=1.5, label='AE Pattern', alpha=0.7)
        
        # Fixed ylim for codebook plot
        ylim_min = -60.0
        ylim_max = 0.0
        
        # Plot cluster wedges
        unique_clusters = np.unique(cluster_ids)
        
        for cluster_id in unique_clusters:
            # Get rays in this cluster
            cluster_mask = cluster_ids == cluster_id
            cluster_rays = rays_lcs[cluster_mask]
            cluster_los = los_flags[cluster_mask]
            
            # Extract angles and power
            theta_lcs = cluster_rays[:, 0]  # zenith
            phi_lcs = cluster_rays[:, 1]    # azimuth
            power = cluster_rays[:, 2]
            
            is_los = np.any(cluster_los)
            
            # Check if cluster has large spread (based on zenith)
            theta_cluster_clipped = np.clip(theta_lcs, 0, 180)
            theta_min = np.min(theta_cluster_clipped)
            theta_max = np.max(theta_cluster_clipped)
            theta_spread = theta_max - theta_min
            
            # If spread > threshold, draw individual rays with small width
            if theta_spread > self.config.theta_spread_threshold_zenith:
                individual_ray_width = self.config.individual_ray_width
                
                for i in range(len(theta_lcs)):
                    theta_ray = theta_lcs[i]
                    power_ray = power[i]
                    
                    # Power (in dB)
                    power_ray_db = 10 * np.log10(power_ray + 1e-20)
                    
                    # Color by cluster index
                    color = self._get_cluster_color(cluster_id)
                    
                    # Plot individual ray (phase conjugate)
                    theta_center_rad = np.radians(180 - theta_ray)
                    width_rad = np.radians(individual_ray_width)
                    
                    # Height proportional to power
                    power_normalized = (power_ray_db - self.config.power_vmin) / (self.config.power_vmax - self.config.power_vmin)
                    power_normalized = np.clip(power_normalized, 0, 1)
                    height = -(self.config.ray_height_min + power_normalized * (self.config.ray_height_max - self.config.ray_height_min))
                    
                    # Style based on LoS
                    if is_los:
                        alpha = self.config.los_alpha_individual
                        edgecolor = self.config.los_edgecolor
                        linewidth = self.config.los_linewidth_individual
                    else:
                        alpha = self.config.nlos_alpha_individual
                        edgecolor = self.config.nlos_edgecolor
                        linewidth = self.config.nlos_linewidth_individual
                    
                    # Plot individual ray bar
                    ax.bar(x=theta_center_rad, height=height, width=width_rad,
                           bottom=ylim_max, color=color, alpha=alpha,
                           edgecolor=edgecolor, linewidth=linewidth)
            else:
                # Normal case: draw as a single wedge
                # Average zenith
                theta_mean = np.mean(theta_lcs)
                
                # Zenith spread (for wedge width)
                theta_cluster_clipped = np.clip(theta_lcs, 0, 180)
                theta_min = np.min(theta_cluster_clipped)
                theta_max = np.max(theta_cluster_clipped)
                theta_spread = theta_max - theta_min
                
                # Power (average in dB)
                power_avg_db = 10 * np.log10(np.mean(power) + 1e-20)
                
                # Color by cluster index
                color = self._get_cluster_color(cluster_id)
                
                # Plot wedge (phase conjugate)
                theta_center_rad = np.radians(180 - theta_mean)
                width_rad = np.radians(theta_spread)
                
                # Height proportional to power
                power_normalized = (power_avg_db - self.config.power_vmin) / (self.config.power_vmax - self.config.power_vmin)
                power_normalized = np.clip(power_normalized, 0, 1)
                height = -(self.config.ray_height_min + power_normalized * (self.config.ray_height_max - self.config.ray_height_min))
                
                # Style based on LoS
                if is_los:
                    alpha = self.config.los_alpha_wedge
                    edgecolor = self.config.los_edgecolor
                    linewidth = self.config.los_linewidth_wedge
                else:
                    alpha = self.config.nlos_alpha_wedge
                    edgecolor = self.config.los_edgecolor
                    linewidth = self.config.nlos_linewidth_wedge
                
                # Plot bar from outer edge inward
                ax.bar(x=theta_center_rad, height=height, width=width_rad,
                       bottom=ylim_max, color=color, alpha=alpha,
                       edgecolor=edgecolor, linewidth=linewidth)
        
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_ylim(ylim_min, ylim_max)
        azimuth_deg_conj = (azimuth_deg - 180) % 360  # Phase conjugate and ensure in [0, 360)
        ax.set_ylabel('Normalized Gain (dB)', fontsize=9, rotation=0)
        ax.yaxis.set_label_coords(0.9, 1)
        ax.set_title(f'UE {ue} Beam {beam_idx} (×{beam_count}) Zenith Cut\n(Azimuth={azimuth_deg_conj:.1f}°)', fontsize=10)
        ax.grid(True)
        ax.legend(fontsize=8, loc='center left')  # 270° 방향 (패턴 없는 곳)
        
        # Add colorbar (only if requested)
        if add_colorbar:
            unique_clusters = np.unique(cluster_ids)
            cmap = plt.get_cmap('tab20')
            n_clusters = len(unique_clusters)
            bounds = np.arange(n_clusters + 1) - 0.5
            norm = plt.matplotlib.colors.BoundaryNorm(bounds, cmap.N)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, pad=0.1, fraction=0.046)
            cbar.set_label('Cluster Index', fontsize=8)
            # Set ticks to cluster indices (not boundaries)
            cbar.set_ticks(unique_clusters)
    
    def plot_eigenbeams(self, ue: int, W_bs: np.ndarray, eigvecs: np.ndarray, eigvals: np.ndarray, output_path: str):
        """
        Eigenbeam pattern plot with PADP (individual eigenbeams)
        
        Args:
            ue: UE index
            W_bs: [1024, K] BS beamforming matrix (directional beams)
            eigvecs: [K, K] eigenvectors from P_bs_opt
            eigvals: [K] eigenvalues (for display only, not used in gain calculation)
            output_path: Output PNG file path
        """
        # Compute eigenbeam vectors without eigenvalues (eigenvectors only)
        # W_eigenbeam = W_bs @ eigvecs [1024, K]
        W_eigenbeam = W_bs @ eigvecs
        # Load rays for PADP computation
        rays_gcs, los_flags, cluster_ids = self.ray_loader.load_rays_for_ue(ue, include_los=True, include_nlos=True)
        rays_lcs = self.ray_loader.transform_rays_to_lcs(rays_gcs)
        
        # Compute power-weighted mean across ALL rays using circular averaging
        padp_theta, padp_phi = self._compute_circular_power_weighted_mean(rays_lcs)
        
        # Identify significant eigenbeams (eigenvalue > threshold)
        eigval_threshold = 0.01
        significant_indices = np.where(eigvals > eigval_threshold)[0]
        n_significant = len(significant_indices)
        
        if n_significant == 0:
            significant_indices = [0]
            n_significant = 1
        
        # Create figure (n_significant rows × 2 columns)
        fig = plt.figure(figsize=(self.config.figure_width, 
                                  self.config.figure_height_per_beam * n_significant))
        
        azimuth_vals = np.arange(-180, 180, 1)
        elevation_vals = np.arange(-90, 90, 1)
        elevation_deg = 90 - padp_theta
        zenith_deg_conj = 180 - padp_theta  # Phase conjugate
        azimuth_deg = padp_phi
        
        # Compute maximum gain for Azimuth cut: fixed Zenith (elevation_deg) across all eigenbeams
        max_gain_azimuth_cut = 0.0
        theta_rad_fixed = np.radians(90 - elevation_deg)  # Fixed Zenith
        
        for beam_idx in significant_indices:
            w_single = W_eigenbeam[:, beam_idx:beam_idx+1]
            for phi_deg in azimuth_vals:
                phi_rad = np.radians(phi_deg)
                gain = self.beam_analyzer.compute_combined_beam_gain(theta_rad_fixed, phi_rad, w_single)
                max_gain_azimuth_cut = max(max_gain_azimuth_cut, gain)
        
        max_gain_azimuth_cut_db = 10 * np.log10(max_gain_azimuth_cut + 1e-10)
        
        # Compute maximum gain for Zenith cut: fixed Azimuth (azimuth_deg) across all eigenbeams
        max_gain_zenith_cut = 0.0
        phi_rad_fixed = np.radians(azimuth_deg)  # Fixed Azimuth
        
        for beam_idx in significant_indices:
            w_single = W_eigenbeam[:, beam_idx:beam_idx+1]
            for elev_deg in elevation_vals:
                theta_rad = np.radians(90 - elev_deg)
                gain = self.beam_analyzer.compute_combined_beam_gain(theta_rad, phi_rad_fixed, w_single)
                max_gain_zenith_cut = max(max_gain_zenith_cut, gain)
        
        max_gain_zenith_cut_db = 10 * np.log10(max_gain_zenith_cut + 1e-10)
        
        # Plot each significant eigenbeam
        for row_idx, beam_idx in enumerate(significant_indices):
            # Extract single eigenbeam [1024, 1]
            w_single = W_eigenbeam[:, beam_idx:beam_idx+1]
            eigval_single = eigvals[beam_idx]
            
            # Azimuth cut
            ax1 = fig.add_subplot(n_significant, 2, 2*row_idx+1, projection='polar')
            azimuth_gains = []
            for phi_deg in azimuth_vals:
                theta_rad = np.radians(90 - elevation_deg)
                phi_rad = np.radians(phi_deg)
                gain = self.beam_analyzer.compute_combined_beam_gain(theta_rad, phi_rad, w_single)
                azimuth_gains.append(gain)
            
            azimuth_gains = np.array(azimuth_gains)
            gain_db = 10 * np.log10(azimuth_gains + 1e-10)
            gain_db_normalized = gain_db - max_gain_azimuth_cut_db
            
            # Plot eigenbeam pattern
            azimuth_rad = np.radians(azimuth_vals - 180)
            ax1.plot(azimuth_rad, gain_db_normalized, 'r-', linewidth=2, label=f'EB{beam_idx+1}')
            
            # Fixed ylim for eigenbeam plot
            ylim_min_azimuth = -60.0
            ylim_max_azimuth = 0.0
            
            # Plot PADP cluster wedges
            self._plot_ray_wedges_azimuth(ax1, rays_lcs, los_flags, cluster_ids, ylim_min_azimuth, ylim_max_azimuth)
            
            ax1.set_theta_zero_location('N')
            ax1.set_theta_direction(-1)
            ax1.set_ylim(ylim_min_azimuth, ylim_max_azimuth)
            ax1.set_ylabel('Normalized Gain (dB)', fontsize=9, rotation=0)
            ax1.yaxis.set_label_coords(0.9, 1)
            ax1.set_title(f'UE {ue} EB{beam_idx+1} (λ_BS={eigval_single:.2f}) Azimuth\n(Zenith={zenith_deg_conj:.1f}°)', fontsize=10)
            ax1.grid(True)
            ax1.legend(fontsize=8, loc='center', bbox_to_anchor=(0.5, 0.25))  # 코드북 플롯과 동일
            
            # Add colorbar (only on first row, left subplot - Azimuth)
            if row_idx == 0:
                unique_clusters = np.unique(cluster_ids)
                cmap = plt.get_cmap('tab20')
                n_clusters = len(unique_clusters)
                bounds = np.arange(n_clusters + 1) - 0.5
                norm = plt.matplotlib.colors.BoundaryNorm(bounds, cmap.N)
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=ax1, pad=0.1, fraction=0.046)
                cbar.set_label('Cluster Index', fontsize=8)
                # Set ticks to cluster indices (not boundaries)
                cbar.set_ticks(unique_clusters)
            
            # Zenith cut
            ax2 = fig.add_subplot(n_significant, 2, 2*row_idx+2, projection='polar')
            zenith_gains = []
            for elev_deg in elevation_vals:
                theta_rad = np.radians(90 - elev_deg)
                phi_rad = np.radians(azimuth_deg)
                gain = self.beam_analyzer.compute_combined_beam_gain(theta_rad, phi_rad, w_single)
                zenith_gains.append(gain)
            
            zenith_gains = np.array(zenith_gains)
            gain_db = 10 * np.log10(zenith_gains + 1e-10)
            gain_db_normalized = gain_db - max_gain_zenith_cut_db
            
            # Plot eigenbeam pattern (phase conjugate)
            theta_vals = 90 - elevation_vals
            theta_rad = np.radians(180 - theta_vals)  # Phase conjugate: zenith → 180° - zenith
            ax2.plot(theta_rad, gain_db_normalized, 'r-', linewidth=2, label=f'EB{beam_idx+1}')
            
            # Fixed ylim for eigenbeam plot
            ylim_min_zenith = -60.0
            ylim_max_zenith = 0.0
            
            # Plot PADP cluster wedges
            self._plot_ray_wedges_zenith(ax2, rays_lcs, los_flags, cluster_ids, ylim_min_zenith, ylim_max_zenith)
            
            ax2.set_theta_zero_location('N')
            ax2.set_theta_direction(-1)
            ax2.set_ylim(ylim_min_zenith, ylim_max_zenith)
            ax2.set_ylabel('Normalized Gain (dB)', fontsize=9, rotation=0)
            ax2.yaxis.set_label_coords(0.9, 1)
            azimuth_deg_conj = (azimuth_deg - 180) % 360
            ax2.set_title(f'UE {ue} EB{beam_idx+1} (λ_BS={eigval_single:.2f}) Zenith\n(Azimuth={azimuth_deg_conj:.1f}°)', fontsize=10)
            ax2.grid(True)
            ax2.legend(fontsize=8, loc='center left')  # 코드북 플롯과 동일
        
        # Suptitle with all eigenvalues (BS transmit power, up to 4)
        eigval_str = ', '.join([f'{v:.2f}' for v in eigvals[:min(4, len(eigvals))]])
        fig.suptitle(f'UE {ue} Eigenbeams (λ_BS: [{eigval_str}])', fontsize=11, y=0.995)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.config.figure_dpi, bbox_inches='tight')
        plt.close()
    
    def _plot_ray_wedges_azimuth(self, ax, rays_lcs: np.ndarray, los_flags: np.ndarray, 
                                 cluster_ids: np.ndarray, ylim_min: float, ylim_max: float):
        """Plot ray cluster wedges for azimuth cut"""
        unique_clusters = np.unique(cluster_ids)
        
        for cluster_id in unique_clusters:
            cluster_mask = cluster_ids == cluster_id
            cluster_rays = rays_lcs[cluster_mask]
            cluster_los = los_flags[cluster_mask]
            
            phi_lcs = cluster_rays[:, 1]
            power = cluster_rays[:, 2]
            is_los = np.any(cluster_los)
            
            # Check spread
            phi_mean = np.arctan2(np.mean(np.sin(np.radians(phi_lcs))),
                                   np.mean(np.cos(np.radians(phi_lcs))))
            phi_mean_deg = np.degrees(phi_mean)
            phi_diff = np.angle(np.exp(1j * np.radians(phi_lcs - phi_mean_deg)), deg=True)
            phi_spread = np.max(phi_diff) - np.min(phi_diff)
            
            if phi_spread > self.config.phi_spread_threshold_azimuth:
                # Individual rays
                for i in range(len(phi_lcs)):
                    phi_ray = phi_lcs[i]
                    power_ray = power[i]
                    power_ray_db = 10 * np.log10(power_ray + 1e-20)
                    
                    # Color by cluster index
                    color = self._get_cluster_color(cluster_id)
                    
                    phi_center_rad = np.radians(phi_ray - 180)
                    width_rad = np.radians(self.config.individual_ray_width)
                    
                    power_normalized = (power_ray_db - self.config.power_vmin) / (self.config.power_vmax - self.config.power_vmin)
                    power_normalized = np.clip(power_normalized, 0, 1)
                    height = -(self.config.ray_height_min + power_normalized * (self.config.ray_height_max - self.config.ray_height_min))
                    
                    alpha = self.config.los_alpha_individual if is_los else self.config.nlos_alpha_individual
                    edgecolor = self.config.los_edgecolor if is_los else self.config.nlos_edgecolor
                    linewidth = self.config.los_linewidth_individual if is_los else self.config.nlos_linewidth_individual
                    
                    ax.bar(x=phi_center_rad, height=height, width=width_rad,
                           bottom=ylim_max, color=color, alpha=alpha,
                           edgecolor=edgecolor, linewidth=linewidth)
            else:
                # Wedge
                power_avg_db = 10 * np.log10(np.mean(power) + 1e-20)
                
                # Color by cluster index
                color = self._get_cluster_color(cluster_id)
                
                phi_center_rad = np.radians(phi_mean_deg - 180)
                width_rad = np.radians(phi_spread)
                
                power_normalized = (power_avg_db - self.config.power_vmin) / (self.config.power_vmax - self.config.power_vmin)
                power_normalized = np.clip(power_normalized, 0, 1)
                height = -(self.config.ray_height_min + power_normalized * (self.config.ray_height_max - self.config.ray_height_min))
                
                alpha = self.config.los_alpha_wedge if is_los else self.config.nlos_alpha_wedge
                edgecolor = self.config.los_edgecolor  # Always use black edge for better cluster separation
                linewidth = self.config.los_linewidth_wedge if is_los else self.config.nlos_linewidth_wedge
                
                ax.bar(x=phi_center_rad, height=height, width=width_rad,
                       bottom=ylim_max, color=color, alpha=alpha,
                       edgecolor=edgecolor, linewidth=linewidth)
    
    def _plot_ray_wedges_zenith(self, ax, rays_lcs: np.ndarray, los_flags: np.ndarray,
                                cluster_ids: np.ndarray, ylim_min: float, ylim_max: float):
        """Plot ray cluster wedges for zenith cut"""
        unique_clusters = np.unique(cluster_ids)
        
        for cluster_id in unique_clusters:
            cluster_mask = cluster_ids == cluster_id
            cluster_rays = rays_lcs[cluster_mask]
            cluster_los = los_flags[cluster_mask]
            
            theta_lcs = cluster_rays[:, 0]
            power = cluster_rays[:, 2]
            is_los = np.any(cluster_los)
            
            # Check spread
            theta_mean = np.mean(theta_lcs)
            theta_spread = np.max(theta_lcs) - np.min(theta_lcs)
            
            if theta_spread > self.config.theta_spread_threshold_zenith:
                # Individual rays
                for i in range(len(theta_lcs)):
                    theta_ray = theta_lcs[i]
                    power_ray = power[i]
                    power_ray_db = 10 * np.log10(power_ray + 1e-20)
                    
                    # Color by cluster index
                    color = self._get_cluster_color(cluster_id)
                    
                    theta_center_rad = np.radians(180 - theta_ray)  # Phase conjugate
                    width_rad = np.radians(self.config.individual_ray_width)
                    
                    power_normalized = (power_ray_db - self.config.power_vmin) / (self.config.power_vmax - self.config.power_vmin)
                    power_normalized = np.clip(power_normalized, 0, 1)
                    height = -(self.config.ray_height_min + power_normalized * (self.config.ray_height_max - self.config.ray_height_min))
                    
                    alpha = self.config.los_alpha_individual if is_los else self.config.nlos_alpha_individual
                    edgecolor = self.config.los_edgecolor if is_los else self.config.nlos_edgecolor
                    linewidth = self.config.los_linewidth_individual if is_los else self.config.nlos_linewidth_individual
                    
                    ax.bar(x=theta_center_rad, height=height, width=width_rad,
                           bottom=ylim_max, color=color, alpha=alpha,
                           edgecolor=edgecolor, linewidth=linewidth)
            else:
                # Wedge
                power_avg_db = 10 * np.log10(np.mean(power) + 1e-20)
                
                # Color by cluster index
                color = self._get_cluster_color(cluster_id)
                
                theta_center_rad = np.radians(180 - theta_mean)  # Phase conjugate
                width_rad = np.radians(theta_spread)
                
                power_normalized = (power_avg_db - self.config.power_vmin) / (self.config.power_vmax - self.config.power_vmin)
                power_normalized = np.clip(power_normalized, 0, 1)
                height = -(self.config.ray_height_min + power_normalized * (self.config.ray_height_max - self.config.ray_height_min))
                
                alpha = self.config.los_alpha_wedge if is_los else self.config.nlos_alpha_wedge
                edgecolor = self.config.los_edgecolor  # Always use black edge for better cluster separation
                linewidth = self.config.los_linewidth_wedge if is_los else self.config.nlos_linewidth_wedge
                
                ax.bar(x=theta_center_rad, height=height, width=width_rad,
                       bottom=ylim_max, color=color, alpha=alpha,
                       edgecolor=edgecolor, linewidth=linewidth)


def main():
    """메인 실행"""
    print("="*70)
    print("P1Q 선택된 Beam들의 방향성 패턴 시각화")
    print("="*70)
    
    # Configuration
    config = P1Q_Config()
    config.initialize()  # 실행 모드에 따라 runtime variables 초기화
    
    # Initialize components
    print("\n[1/4] Initializing components...")
    ray_loader = RayDataLoader(config)
    beam_analyzer = BeamPatternAnalyzer(config)
    plotter = BeamPatternPlotter(beam_analyzer, ray_loader)
    
    if config.manual_mode:
        # Manual mode: use manually specified UE and beams
        print("\n[2/4] Manual Mode: Processing specified UE and beams...")
        
        if config.manual_ue is None or len(config.manual_bs_beams) == 0:
            print("  Error: manual_mode=True but manual_ue or manual_bs_beams not set")
            return
        
        print(f"  Mode: Manual")
        print(f"  Output directory: {config.output_dir}")
        
        all_ues = [config.manual_ue]
        
        # Use full beam list with repetitions
        all_beams = np.array(config.manual_bs_beams)
        
        # Compute beam_counts for plotting (display unique beams)
        from collections import Counter
        beam_counts = Counter(config.manual_bs_beams)
        unique_beams_display = sorted(beam_counts.keys())
        
        print(f"  UE {config.manual_ue}: K={len(all_beams)} (unique: {len(unique_beams_display)} beams {unique_beams_display})")
        
        # Load Weichselberger parameters
        print(f"\n[3/4] Loading Weichselberger parameters and optimizing...")
        try:
            U_bs_full, U_ue_full, Omega_ul, H_mean_ul = ray_loader.load_weichselberger_for_ue(config.manual_ue)
            
            # Manual mode: ue_beams는 manual_ue_beams 사용 (없으면 기본값)
            if len(config.manual_ue_beams) > 0:
                ue_beams = np.array(config.manual_ue_beams)
            else:
                # 기본값: identity matrix에 해당하는 빔 인덱스 (0, 1, 2, 3)
                ue_beams = np.arange(config.n_ue_layer)
            
            # DL beam domain extraction (use full beam list with repetitions)
            U_ue_beam, U_bs_beam, Omega_beam_dl, H_mean_beam_dl, W_bs, W_ue = beam_analyzer.extract_beam_domain_params_dl(
                all_beams, ue_beams, U_bs_full, U_ue_full, Omega_ul, H_mean_ul
            )
            
            # DL Wen2011 optimization
            print(f"  Running DL Wen2011 optimization...")
            C_bs, P_bs, eigvals, eigvecs = beam_analyzer.optimize_bs_transmit_dl(
                U_ue_beam, U_bs_beam, Omega_beam_dl, H_mean_beam_dl
            )
            print(f"  Capacity: {C_bs:.2f} bps/Hz")
            print(f"  Eigenvalues: {eigvals}")
            
            has_eigenbeam = True
        except Exception as e:
            print(f"  Warning: Wen2011 optimization failed: {e}")
            print(f"  Proceeding with codebook-only plot")
            has_eigenbeam = False
        
        # Generate plots
        print(f"\n[4/4] Generating plots...")
        output_path_codebook = Path(config.output_dir) / f"ue_{config.manual_ue:04d}_codebook_manual.png"
        plotter.plot_codebook_beams(config.manual_ue, np.array(unique_beams_display), beam_counts, str(output_path_codebook))
        print(f"  Codebook plot saved: {output_path_codebook.name}")
        
        if has_eigenbeam:
            output_path_eigenbeam = Path(config.output_dir) / f"ue_{config.manual_ue:04d}_eigenbeam_manual.png"
            plotter.plot_eigenbeams(config.manual_ue, W_bs.numpy(), eigvecs, eigvals, str(output_path_eigenbeam))
            print(f"  Eigenbeam plot saved: {output_path_eigenbeam.name}")
        
        print(f"\nComplete! Generated plots for UE {config.manual_ue}")
    else:
        # Auto mode: load from CSV files (P1P만 지원)
        print(f"\n[2/4] Auto Mode: Loading UEs from P1P CSV files...")
        print(f"  CSV directory: {config.csv_input_dir}")
        print(f"  Output directory: {config.output_dir}")
        
        loader = P1QResultsLoader(config)
        all_ues = loader.load_all_ues(config.csv_files)
        print(f"  Found {len(all_ues)} UEs: {all_ues[:20]}...")
        
        # Process each UE
        print(f"\n[3/4] Processing {len(all_ues)} UEs...")
        
        n_ues_processed = 0
        for ue_idx, target_ue in enumerate(all_ues, 1):
            # Load full beam list (with repetitions)
            all_beams = loader.load_all_beams_for_ue(config.csv_files, target_ue)
            
            if len(all_beams) == 0:
                continue
            
            # Load ue_beams from P1P CSV
            try:
                ue_beams = loader.load_ue_beams_for_ue_p1p(config.csv_files, target_ue)
            except ValueError as e:
                print(f"    Warning: {e}, skipping UE {target_ue}")
                continue
            
            # Compute beam_counts for plotting
            from collections import Counter
            beam_counts = Counter(all_beams)
            unique_beams_display = sorted(beam_counts.keys())
            
            print(f"  [{ue_idx}/{len(all_ues)}] UE {target_ue}: K={len(all_beams)} (unique: {len(unique_beams_display)} beams {unique_beams_display}), ue_beams={ue_beams}")
            
            # Try DL Wen2011 optimization
            has_eigenbeam = False
            W_bs_plot = None
            eigvecs_plot = None
            eigvals_plot = None
            try:
                U_bs_full, U_ue_full, Omega_ul, H_mean_ul = ray_loader.load_weichselberger_for_ue(target_ue)
                U_ue_beam, U_bs_beam, Omega_beam_dl, H_mean_beam_dl, W_bs, W_ue = beam_analyzer.extract_beam_domain_params_dl(
                    all_beams, ue_beams, U_bs_full, U_ue_full, Omega_ul, H_mean_ul
                )
                C_bs, P_bs, eigvals, eigvecs = beam_analyzer.optimize_bs_transmit_dl(
                    U_ue_beam, U_bs_beam, Omega_beam_dl, H_mean_beam_dl
                )
                W_bs_plot = W_bs.numpy()
                eigvecs_plot = eigvecs
                eigvals_plot = eigvals
                has_eigenbeam = True
                print(f"    DL Wen2011: C={C_bs:.2f} bps/Hz")
            except Exception as e:
                print(f"    Warning: Wen2011 failed: {e}")
            
            # Plot codebook
            output_path_codebook = Path(config.output_dir) / f"ue_{target_ue:04d}_codebook.png"
            plotter.plot_codebook_beams(target_ue, np.array(unique_beams_display), beam_counts, str(output_path_codebook))
            print(f"    Codebook plot saved")
            n_ues_processed += 1
            
            # Plot eigenbeam
            if has_eigenbeam:
                output_path_eigenbeam = Path(config.output_dir) / f"ue_{target_ue:04d}_eigenbeam.png"
                plotter.plot_eigenbeams(target_ue, W_bs_plot, eigvecs_plot, eigvals_plot, str(output_path_eigenbeam))
                print(f"    Eigenbeam plot saved")
        
        print(f"\n[4/4] Complete! Generated {n_ues_processed} plots for {len(all_ues)} UEs")
    
    print("\n" + "="*70)
    print(f"All plots saved to: {config.output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()

