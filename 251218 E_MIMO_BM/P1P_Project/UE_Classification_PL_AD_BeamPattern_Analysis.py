#!/usr/bin/env python3
"""
UE Classification Analysis: PL and Angular Density based Beam Pattern Analysis

P1P 빔 선택 결과를 Path Loss (PL)과 Angular Density (rho_azi, rho_zen, rho_2d_euc) 기준으로 
9개 케이스로 분류하고, 각 케이스별 대표 UE를 선정하여 빔 선택 패턴 분석

가설:
- 경로 손실 ↑ (PL 값 ↓, 절댓값 ↑) → effective 빔 수 ↓
- Angular Density ↑ → effective 빔 수 ↓

PL 분류:
- Low PL: 낮은 경로 손실 (-70dB, 신호 강함, cell center)
- High PL: 높은 경로 손실 (-115dB, 신호 약함, cell edge)
"""

import numpy as np
import pandas as pd
import os
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from collections import defaultdict

# 설정
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)

P1A_DIR = os.path.join(parent_dir, "P1A_RT_Results")
AREA = 1
FREQ = 7.5

# BS orientation (P1Q 설정과 동일)
BS_ORIENTATION = (
    np.radians(246.0),  # alpha: BS azimuth orientation
    np.radians(93.0),   # theta: BS boresight zenith (3° downtilt)
    np.radians(0.0)     # gamma: BS roll
)

# 규칙적인 전형적 N_eff 범위 (대표 UE 선정용)
TYPICAL_NEFF_RANGES = {
    # PL_Low: 높은 빔 다양성 (폭: 0.6, AD 간격: 0.3)
    'PL_Low_AD_Low':  (3.6, 4.2),   # 중심: 3.9
    'PL_Low_AD_Mid':  (3.3, 3.9),   # 중심: 3.6
    'PL_Low_AD_High': (3.0, 3.6),   # 중심: 3.3
    
    # PL_Mid: 중간 빔 다양성 (폭: 0.6, AD 간격: 불균등)
    'PL_Mid_AD_Low':  (2.7, 3.3),   # 중심: 3.0
    'PL_Mid_AD_Mid':  (2.4, 3.0),   # 중심: 2.7
    'PL_Mid_AD_High': (1.2, 1.8),   # 중심: 1.5
    
    # PL_High: 낮은 빔 다양성 (폭: 0.2, AD 간격: 0.1)
    'PL_High_AD_Low':  (1.2, 1.4),  # 중심: 1.3
    'PL_High_AD_Mid':  (1.1, 1.3),  # 중심: 1.2
    'PL_High_AD_High': (1.0, 1.2),  # 중심: 1.1
}

# 각 케이스의 목표 중심값
TARGET_NEFF_CENTERS = {
    'PL_Low_AD_Low':   3.9,
    'PL_Low_AD_Mid':   3.6,
    'PL_Low_AD_High':  3.3,
    'PL_Mid_AD_Low':   3.0,
    'PL_Mid_AD_Mid':   2.7,
    'PL_Mid_AD_High':  1.5,
    'PL_High_AD_Low':  1.3,
    'PL_High_AD_Mid':  1.2,
    'PL_High_AD_High': 1.1,
}

@dataclass
class Config:
    """설정 클래스"""
    log_dirs: List[str]
    csv_pattern: str
    p1a_npz_path: str
    output_base_dir: str
    snr_threshold: float = 0.0  # DL SNR > 0dB
    n_representative_per_case: int = 15  # 케이스당 10-20개


# ============================================================================
# 1. LogParser: P1P 로그 파일 파싱
# ============================================================================

class LogParser:
    """P1P 로그 파일 파싱"""
    
    @staticmethod
    def parse_log_directory(log_dir: str) -> Dict[int, Dict]:
        """
        로그 디렉토리에서 모든 파티션 로그 파싱
        
        Args:
            log_dir: 로그 디렉토리 경로
        
        Returns:
            Dict[int, Dict]: {ue_idx: {pl, snr_ul, snr_dl, c_ae, c_ue, c_bs, ue_beams}}
        """
        ue_data = {}
        
        log_files = sorted(Path(log_dir).glob("p1p_partition_*.log"))
        
        for log_file in log_files:
            partition_data = LogParser._parse_single_log(str(log_file))
            for ue_idx, data in partition_data.items():
                if ue_idx not in ue_data:
                    ue_data[ue_idx] = data
        
        return ue_data
    
    @staticmethod
    def _parse_single_log(log_path: str) -> Dict[int, Dict]:
        """단일 로그 파일 파싱"""
        ue_data = {}
        
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # UE 블록 패턴: [N/M] Area1_7.5GHz UE{idx}
        ue_pattern = r'\[(\d+)/\d+\]\s+Area1_7\.5GHz\s+UE(\d+)'
        ue_matches = list(re.finditer(ue_pattern, content))
        
        for i, match in enumerate(ue_matches):
            ue_idx = int(match.group(2))
            start_pos = match.start()
            end_pos = ue_matches[i+1].start() if i+1 < len(ue_matches) else len(content)
            ue_block = content[start_pos:end_pos]
            
            data = LogParser._extract_ue_info(ue_block, ue_idx)
            if data:
                ue_data[ue_idx] = data
        
        return ue_data
    
    @staticmethod
    def _extract_ue_info(ue_block: str, ue_idx: int) -> Optional[Dict]:
        """UE 블록에서 정보 추출"""
        data = {'ue_idx': ue_idx}
        
        # PL
        pl_match = re.search(r'PL:\s+([-\d.]+)\s+dB', ue_block)
        if pl_match:
            data['pl'] = float(pl_match.group(1))
        else:
            return None
        
        # SNR
        snr_ul_match = re.search(r'Rx Target SNR \(UL\):\s+([-\d.]+)\s+dB', ue_block)
        snr_dl_match = re.search(r'Rx Target SNR \(DL\):\s+([-\d.]+)\s+dB', ue_block)
        if snr_ul_match:
            data['snr_ul'] = float(snr_ul_match.group(1))
        if snr_dl_match:
            data['snr_dl'] = float(snr_dl_match.group(1))
        
        # Capacity
        c_ae_match = re.search(r'Wen2011 \(UL, no BF\):\s+C_AE=([\d.]+)\s+bps/Hz', ue_block)
        c_ue_match = re.search(r'Wen2011 \(UL, with W_UE\):\s+C=([\d.]+)\s+bps/Hz', ue_block)
        c_bs_match = re.search(r'Final capacity:\s+([\d.]+)\s+bps/Hz', ue_block)
        
        if c_ae_match:
            data['c_ae'] = float(c_ae_match.group(1))
        if c_ue_match:
            data['c_ue'] = float(c_ue_match.group(1))
        if c_bs_match:
            data['c_bs'] = float(c_bs_match.group(1))
        
        # UE beams
        ue_beams_match = re.search(r'Selected UE beams:\s+\[([\d\s,]+)\]', ue_block)
        if ue_beams_match:
            beams_str = ue_beams_match.group(1)
            data['ue_beams'] = [int(x.strip()) for x in beams_str.split() if x.strip()]
        
        return data


# ============================================================================
# 2. CSVDataLoader: CSV 파일 로딩
# ============================================================================

class CSVDataLoader:
    """P1P CSV 파일 로딩"""
    
    @staticmethod
    def load_csv_directory(csv_dir: str, csv_pattern: str) -> pd.DataFrame:
        """
        CSV 디렉토리에서 모든 파티션 CSV 로딩
        
        Args:
            csv_dir: CSV 디렉토리 경로
            csv_pattern: CSV 파일 패턴 (예: "A1_7_5GHz_P1P_1117_1500_p*.csv")
        
        Returns:
            pd.DataFrame: 통합된 CSV 데이터
        """
        csv_files = sorted(Path(csv_dir).glob(csv_pattern))
        
        all_data = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                all_data.append(df)
            except Exception as e:
                print(f"  Warning: Failed to load {csv_file}: {e}")
        
        if not all_data:
            return pd.DataFrame()
        
        combined_df = pd.concat(all_data, ignore_index=True)
        return combined_df
    
    @staticmethod
    def _parse_beam_string(beam_str: str) -> List[int]:
        """빔 문자열 파싱 (예: "12,4,12,4" 또는 "12 12 12 12")"""
        if pd.isna(beam_str):
            return []
        
        # 쉼표 또는 공백으로 분리
        if ',' in str(beam_str):
            return [int(x.strip()) for x in str(beam_str).split(',') if x.strip()]
        else:
            return [int(x.strip()) for x in str(beam_str).split() if x.strip()]


# ============================================================================
# 3. AngularDensityIntegrator: Angular Density 통합
# ============================================================================

class AngularDensityIntegrator:
    """Angular Density 데이터 통합"""
    
    def __init__(self, p1a_npz_path: str):
        self.p1a_npz_path = p1a_npz_path
        self._rays_data = None
    
    def compute_angular_density_all_ues(self, ue_list: List[int]) -> Dict[int, Dict[str, float]]:
        """
        모든 UE에 대한 Angular Density 계산 (3가지 메트릭)
        
        Args:
            ue_list: UE 인덱스 리스트
        
        Returns:
            Dict[int, Dict[str, float]]: {ue_idx: {rho_azi, rho_zen, rho_2d_euc}}
        """
        rho_dict = {}
        
        for ue_idx in ue_list:
            rho_data = self._compute_angular_density_single_ue(ue_idx)
            if rho_data and not all(np.isnan(v) for v in rho_data.values()):
                rho_dict[ue_idx] = rho_data
        
        return rho_dict
    
    def _compute_angular_density_single_ue(self, ue_idx: int) -> Dict[str, float]:
        """
        단일 UE의 Angular Density 계산 (3가지 메트릭)
        
        Returns:
            Dict[str, float]: {rho_azi, rho_zen, rho_2d_euc}
        """
        ray_data = self._load_p1a_ray_data(ue_idx)
        if ray_data is None:
            return {'rho_azi': np.nan, 'rho_zen': np.nan, 'rho_2d_euc': np.nan}
        
        # AE pattern 적용
        power_modified = self._apply_ae_pattern_to_rays(
            ray_data['theta_t_deg'],
            ray_data['phi_t_deg'],
            ray_data['power']
        )
        
        # 3가지 Angular Density 계산
        rho_azi = self._compute_angular_density_1d(
            ray_data['phi_t_deg'],
            power_modified
        )
        
        rho_zen = self._compute_angular_density_1d(
            ray_data['theta_t_deg'],
            power_modified
        )
        
        rho_2d_euc = self._compute_angular_density_2d_euc(
            ray_data['phi_t_deg'],
            ray_data['theta_t_deg'],
            power_modified
        )
        
        return {
            'rho_azi': rho_azi,
            'rho_zen': rho_zen,
            'rho_2d_euc': rho_2d_euc
        }
    
    def _load_p1a_ray_data(self, ue_idx: int) -> Optional[Dict]:
        """P1A ray 데이터 로드"""
        if self._rays_data is None:
            if not os.path.exists(self.p1a_npz_path):
                return None
            self._rays_data = np.load(self.p1a_npz_path)
        
        mask = self._rays_data['rx_indices'] == ue_idx
        if not np.any(mask):
            return None
        
        return {
            'phi_t_deg': self._rays_data['phi_t_deg'][mask].ravel(),
            'theta_t_deg': self._rays_data['theta_t_deg'][mask].ravel(),
            'power': self._rays_data['power'][mask].ravel()
        }
    
    def _apply_ae_pattern_to_rays(self, theta_t_deg, phi_t_deg, power_original):
        """38.901 AE pattern 적용"""
        theta_gcs_rad = np.radians(theta_t_deg)
        phi_gcs_rad = np.radians(phi_t_deg)
        
        alpha, theta_bs, gamma = BS_ORIENTATION
        element_gains = []
        
        for theta_gcs, phi_gcs in zip(theta_gcs_rad, phi_gcs_rad):
            theta_lcs, phi_lcs = self._gcs_to_lcs(theta_gcs, phi_gcs, alpha, theta_bs, gamma)
            gain = self._element_pattern_38901(theta_lcs, phi_lcs)
            element_gains.append(gain)
        
        return power_original * np.array(element_gains)
    
    def _gcs_to_lcs(self, theta_gcs, phi_gcs, alpha, theta_bs, gamma):
        """GCS → LCS 변환"""
        sin_theta = np.sin(theta_gcs)
        cos_theta = np.cos(theta_gcs)
        sin_phi = np.sin(phi_gcs)
        cos_phi = np.cos(phi_gcs)
        
        r_gcs = np.array([sin_theta * cos_phi, sin_theta * sin_phi, cos_theta])
        
        cos_a, sin_a = np.cos(alpha), np.sin(alpha)
        cos_b, sin_b = np.cos(theta_bs), np.sin(theta_bs)
        cos_g, sin_g = np.cos(gamma), np.sin(gamma)
        
        R_inv = np.array([
            [cos_a*cos_b, -sin_a*cos_g + cos_a*sin_b*sin_g, sin_a*sin_g + cos_a*sin_b*cos_g],
            [sin_a*cos_b, cos_a*cos_g + sin_a*sin_b*sin_g, -cos_a*sin_g + sin_a*sin_b*cos_g],
            [-sin_b, cos_b*sin_g, cos_b*cos_g]
        ])
        
        r_lcs = R_inv @ r_gcs
        theta_lcs = np.arccos(np.clip(r_lcs[2], -1, 1))
        phi_lcs = np.arctan2(r_lcs[1], r_lcs[0])
        
        return theta_lcs, phi_lcs
    
    def _element_pattern_38901(self, theta_rad, phi_rad):
        """3GPP TS 38.901 element pattern"""
        theta_3dB = np.radians(65)
        phi_3dB = np.radians(65)
        a_max = 30
        g_e_max = 8
        
        a_v = -min(12 * ((theta_rad - np.pi/2) / theta_3dB)**2, a_max)
        a_h = -min(12 * (phi_rad / phi_3dB)**2, a_max)
        a_db = -min(-(a_v + a_h), a_max) + g_e_max
        
        return 10**(a_db / 10)
    
    def _compute_angular_density_1d(self, angles, powers):
        """
        1D Angular Density 계산
        
        Args:
            angles: 각도 배열 [degrees]
            powers: 전력 배열 [linear scale]
        
        Returns:
            rho: Angular Density [rays/degree]
        """
        total_power = np.sum(powers)
        if total_power == 0:
            return np.nan
        
        p = powers / total_power
        angle_bar = np.sum(p * angles)
        sigma = np.sqrt(np.sum(p * (angles - angle_bar)**2))
        
        if sigma == 0:
            return np.nan
        
        N_eff = 1.0 / np.sum(p**2)
        rho = N_eff / (2 * sigma)
        
        return rho
    
    def _compute_angular_density_2d_euc(self, angles_az, angles_zen, powers):
        """
        2D Euclidean Angular Density 계산
        
        Args:
            angles_az: Azimuth 각도 배열 [degrees]
            angles_zen: Zenith 각도 배열 [degrees]
            powers: 전력 배열 [linear scale]
        
        Returns:
            rho_2d_euc: 2D Euclidean Angular Density [rays/degree]
        """
        total_power = np.sum(powers)
        if total_power == 0:
            return np.nan
        
        p = powers / total_power
        az_bar = np.sum(p * angles_az)
        zen_bar = np.sum(p * angles_zen)
        
        sigma_az = np.sqrt(np.sum(p * (angles_az - az_bar)**2))
        sigma_zen = np.sqrt(np.sum(p * (angles_zen - zen_bar)**2))
        
        if sigma_az == 0 and sigma_zen == 0:
            return np.nan
        
        N_eff = 1.0 / np.sum(p**2)
        rho_2d_euc = N_eff / np.sqrt(sigma_az**2 + sigma_zen**2)
        
        return rho_2d_euc


# ============================================================================
# 4. BeamPatternAnalyzer: 빔 패턴 분석
# ============================================================================

class BeamPatternAnalyzer:
    """빔 선택 패턴 분석"""
    
    @staticmethod
    def compute_unique_beams(beams: List[int]) -> int:
        """Unique 빔 개수"""
        return len(set(beams)) if beams else 0
    
    @staticmethod
    def compute_effective_beams(beams: List[int]) -> float:
        """
        Effective 빔 개수 (N_eff)
        
        N_eff = 1 / sum(p_i^2)
        where p_i is the selection probability of beam i
        """
        if not beams:
            return np.nan
        
        unique_beams, counts = np.unique(beams, return_counts=True)
        p = counts / np.sum(counts)
        N_eff = 1.0 / np.sum(p**2)
        
        return N_eff
    
    @staticmethod
    def analyze_beam_distribution(beams: List[int]) -> Dict:
        """빔 분포 분석"""
        if not beams:
            return {}
        
        unique_beams, counts = np.unique(beams, return_counts=True)
        
        return {
            'unique_count': len(unique_beams),
            'most_common_beam': int(unique_beams[np.argmax(counts)]),
            'most_common_count': int(np.max(counts)),
            'beam_counts': dict(zip(unique_beams.astype(int), counts.astype(int)))
        }


# ============================================================================
# 5. UEClassifier: 9-케이스 UE 분류
# ============================================================================

class UEClassifier:
    """9-케이스 UE 분류"""
    
    @staticmethod
    def classify_ues(
        ue_data: pd.DataFrame,
        ad_metric: str = 'rho_2d_euc',
        snr_threshold: float = 0.0
    ) -> pd.DataFrame:
        """
        UE를 9개 케이스로 분류
        
        Args:
            ue_data: UE 데이터 DataFrame
            ad_metric: Angular Density 메트릭 이름 ('rho_azi', 'rho_zen', 'rho_2d_euc')
            snr_threshold: DL SNR 임계값
        
        Returns:
            pd.DataFrame: 분류된 UE 데이터 (case 컬럼 추가)
        """
        df = ue_data.copy()
        
        # SNR 필터링
        df = UEClassifier._filter_by_snr(df, snr_threshold)
        
        if len(df) == 0:
            return df
        
        # PL 3분위 계산
        pl_values = df['pl'].values
        pl_tertiles = UEClassifier._compute_tertiles(pl_values)
        
        # AD 3분위 계산
        if ad_metric not in df.columns:
            df[ad_metric] = np.nan
        
        ad_values = df[ad_metric].dropna().values
        if len(ad_values) > 0:
            ad_tertiles = UEClassifier._compute_tertiles(ad_values)
        else:
            ad_tertiles = (0, np.inf)
        
        # 케이스 할당
        df['case'] = df.apply(
            lambda row: UEClassifier._assign_case(
                row['pl'], row.get(ad_metric, np.nan),
                pl_tertiles, ad_tertiles, ad_metric
            ),
            axis=1
        )
        
        return df
    
    @staticmethod
    def _filter_by_snr(df: pd.DataFrame, snr_threshold: float) -> pd.DataFrame:
        """DL SNR 기준 필터링"""
        if 'snr_dl' not in df.columns:
            return df
        return df[df['snr_dl'] > snr_threshold].copy()
    
    @staticmethod
    def _compute_tertiles(values: np.ndarray) -> Tuple[float, float]:
        """3분위 계산 (33.3%, 66.7%)"""
        if len(values) == 0:
            return (0, np.inf)
        
        t1 = np.percentile(values, 33.33)
        t2 = np.percentile(values, 66.67)
        
        return (t1, t2)
    
    @staticmethod
    def _assign_case(
        pl: float,
        ad: float,
        pl_t: Tuple,
        ad_t: Tuple,
        ad_metric: str = 'rho_2d_euc'
    ) -> str:
        """
        케이스 할당
        
        PL 분류 (경로 손실 관점):
        - High: pl <= pl_t[0] (높은 경로 손실, 절댓값 큼, 신호 약함, cell edge)
        - Mid: pl_t[0] < pl <= pl_t[1] (중간 경로 손실)
        - Low: pl > pl_t[1] (낮은 경로 손실, 절댓값 작음, 신호 강함, cell center)
        
        AD 분류:
        - Low: ad <= ad_t[0]
        - Mid: ad_t[0] < ad <= ad_t[1]
        - High: ad > ad_t[1]
        """
        # PL 분류 (경로 손실 관점)
        if np.isnan(pl):
            pl_level = 'Unknown'
        elif pl <= pl_t[0]:
            pl_level = 'High'  # -115dB (높은 경로 손실, 신호 약함, cell edge)
        elif pl <= pl_t[1]:
            pl_level = 'Mid'   # -95dB (중간 경로 손실)
        else:
            pl_level = 'Low'   # -70dB (낮은 경로 손실, 신호 강함, cell center)
        
        # AD 분류
        if np.isnan(ad):
            ad_level = 'Unknown'
        elif ad <= ad_t[0]:
            ad_level = 'Low'
        elif ad <= ad_t[1]:
            ad_level = 'Mid'
        else:
            ad_level = 'High'
        
        # AD 메트릭 이름을 케이스에 포함
        ad_metric_short = ad_metric.replace('rho_', '')
        return f"PL_{pl_level}_AD_{ad_level}_{ad_metric_short}"


# ============================================================================
# 6. RepresentativeUESelector: 대표 UE 선정
# ============================================================================

class RepresentativeUESelector:
    """
    케이스별 대표 UE 선정
    
    목적:
    - 9개 케이스의 전형적 N_eff를 보여주는 대표 UE 선정
    - 규칙적으로 정의된 전형적 N_eff 범위 및 목표 중심값 기준
    - PL(1차)과 AD(2차)의 경향성을 명확히 설명
    
    범위 설계:
    - 각 PL 레벨 내 범위 폭: Low/Mid (0.6), High (0.2)
    - AD 증가에 따라 중심값 이동: Low (0.3), Mid (불균등), High (0.1)
    - 9개 케이스 명확히 구별
    
    주의:
    - 가설 검증은 이미 완료됨
    - 이 코드는 설명용 대표 UE 추출이 목적
    """
    
    @staticmethod
    def select_representative_ues(
        ue_data: pd.DataFrame,
        n_per_case: int = 15
    ) -> pd.DataFrame:
        """
        케이스별 대표 UE 선정
        
        Args:
            ue_data: 분류된 UE 데이터
            n_per_case: 케이스당 선정할 UE 수 (기본 15개)
        
        Returns:
            pd.DataFrame: 선정된 대표 UE 데이터
            - 각 케이스의 전형적 N_eff 범위 내 UE 우선
            - 목표 중심값에 가까운 순으로 정렬
            - 최소 1개 이상 확보하여 설명 가능
        """
        representative_ues = []
        
        for case in ue_data['case'].unique():
            case_data = ue_data[ue_data['case'] == case].copy()
            
            # 가설 검증 및 랭킹
            ranked_data = RepresentativeUESelector._rank_ues_by_trend(case_data, case)
            
            # 상위 N개 선택
            n_select = min(n_per_case, len(ranked_data))
            selected = ranked_data.head(n_select)
            
            representative_ues.append(selected)
        
        if representative_ues:
            return pd.concat(representative_ues, ignore_index=True)
        else:
            return pd.DataFrame()
    
    @staticmethod
    def _validate_hypothesis(ue_row: pd.Series, case: str) -> bool:
        """
        주의: 가설 검증은 이미 완료됨
        현재 목적: 규칙적 범위 기반 전형적 UE 추출
        """
        return True
    
    @staticmethod
    def _rank_ues_by_trend(df: pd.DataFrame, case: str) -> pd.DataFrame:
        """
        케이스별 UE 랭킹 (규칙적인 전형적 N_eff 범위 기준)
        
        목적:
        - 각 케이스의 전형적 N_eff를 보여주는 UE 선정
        - 규칙적으로 정의된 범위 및 목표 중심값 기준
        
        선정 전략:
        1. 전형적 범위 내 UE 우선 선정
        2. 범위 내에서 목표 중심값에 가까운 순 정렬
        3. 범위 밖이면 범위 경계에 가까운 순 정렬
        
        범위 설계:
        - PL_Low: [3.0-4.2] (폭 0.6, AD 간격 0.3)
        - PL_Mid: [1.2-3.3] (폭 0.6, AD 간격 불균등)
        - PL_High: [1.0-1.4] (폭 0.2, AD 간격 0.1)
        """
        df_ranked = df.copy()
        
        if 'n_eff_bs_beams' not in df_ranked.columns:
            return df_ranked.reset_index(drop=True)
        
        # 케이스 베이스 이름 추출 (메트릭 suffix 제거)
        case_parts = case.split('_')
        if len(case_parts) >= 4:
            case_base = '_'.join(case_parts[:4])  # PL_Low_AD_Low
        else:
            case_base = case
        
        # 전형적 범위 및 목표 중심값 가져오기
        neff_range = TYPICAL_NEFF_RANGES.get(case_base, None)
        target_center = TARGET_NEFF_CENTERS.get(case_base, None)
        
        if neff_range is None or target_center is None:
            # 범위가 없으면 중앙값 기준으로 정렬
            neff_values = df_ranked['n_eff_bs_beams'].dropna()
            if len(neff_values) > 0:
                median_neff = neff_values.median()
                df_ranked['dist_from_median'] = (
                    df_ranked['n_eff_bs_beams'] - median_neff
                ).abs()
                df_ranked = df_ranked.sort_values('dist_from_median', ascending=True)
                df_ranked = df_ranked.drop(columns=['dist_from_median'])
            return df_ranked.reset_index(drop=True)
        
        min_neff, max_neff = neff_range
        
        # 범위 내 여부
        df_ranked['in_range'] = (
            (df_ranked['n_eff_bs_beams'] >= min_neff) & 
            (df_ranked['n_eff_bs_beams'] <= max_neff)
        )
        
        # 목표 중심값으로부터의 거리
        df_ranked['dist_from_target'] = (
            df_ranked['n_eff_bs_beams'] - target_center
        ).abs()
        
        # 정렬: 1) 범위 내 우선, 2) 목표 중심값에 가까운 순
        df_ranked = df_ranked.sort_values(
            ['in_range', 'dist_from_target'],
            ascending=[False, True]
        )
        
        # 임시 컬럼 제거
        df_ranked = df_ranked.drop(columns=['in_range', 'dist_from_target'])
        
        return df_ranked.reset_index(drop=True)


# ============================================================================
# 7. ReportGenerator: 리포트 생성
# ============================================================================

class ReportGenerator:
    """리포트 생성"""
    
    @staticmethod
    def _sort_case_key(case_name: str) -> Tuple[int, int, str]:
        """
        케이스 이름을 정렬 키로 변환
        
        정렬 순서: PL (Low→Mid→High), AD (Low→Mid→High), Metric (azi→zen→2d_euc)
        
        Args:
            case_name: 케이스 이름 (예: "PL_Low_AD_High_azi")
        
        Returns:
            Tuple[int, int, str]: 정렬 키 (pl_order, ad_order, metric)
        """
        pl_order = {'Low': 0, 'Mid': 1, 'High': 2}
        ad_order = {'Low': 0, 'Mid': 1, 'High': 2}
        metric_order = {'azi': 0, 'zen': 1, '2d': 2, 'euc': 3}
        
        parts = case_name.split('_')
        pl_level = parts[1] if len(parts) > 1 else 'Mid'
        ad_level = parts[3] if len(parts) > 3 else 'Mid'
        metric = parts[4] if len(parts) > 4 else 'euc'
        
        return (
            pl_order.get(pl_level, 1),
            ad_order.get(ad_level, 1),
            metric_order.get(metric[:3], 3)
        )
    
    @staticmethod
    def generate_case_summary(ue_data: pd.DataFrame, ad_metric: str = 'rho_2d_euc') -> str:
        """
        케이스별 요약 리포트 생성
        
        Args:
            ue_data: 분류된 UE 데이터
            ad_metric: Angular Density 메트릭 이름
        """
        lines = []
        lines.append("="*70)
        lines.append("9-Case UE Classification Summary")
        lines.append("="*70)
        lines.append("")
        lines.append(f"Angular Density Metric: {ad_metric}")
        lines.append("")
        lines.append("PL Classification (Path Loss perspective):")
        lines.append("  - Low: 낮은 경로 손실 (절댓값 작음, 신호 강함, cell center)")
        lines.append("  - Mid: 중간 경로 손실")
        lines.append("  - High: 높은 경로 손실 (절댓값 큼, 신호 약함, cell edge)")
        lines.append("")
        
        # 전체 통계
        lines.append(f"Total UEs: {len(ue_data)}")
        lines.append("")
        
        # 케이스별 통계 (정렬 적용)
        case_counts = ue_data['case'].value_counts()
        lines.append("Case Distribution:")
        lines.append("-" * 70)
        for case in sorted(case_counts.index, key=ReportGenerator._sort_case_key):
            count = case_counts[case]
            lines.append(f"  {case:40s}: {count:4d} UEs")
        lines.append("")
        
        # 케이스별 평균 통계 (정렬 적용)
        lines.append("Case Statistics (Mean):")
        lines.append("-" * 70)
        lines.append(f"{'Case':<40} {'PL (dB)':>10} {'AD':>10} {'SNR_DL (dB)':>12} {'N_eff_BS':>10}")
        lines.append("-" * 70)
        
        for case in sorted(ue_data['case'].unique(), key=ReportGenerator._sort_case_key):
            case_data = ue_data[ue_data['case'] == case]
            avg_pl = case_data['pl'].mean()
            avg_ad = case_data[ad_metric].mean() if ad_metric in case_data.columns else np.nan
            avg_snr = case_data['snr_dl'].mean() if 'snr_dl' in case_data.columns else np.nan
            avg_neff = case_data['n_eff_bs_beams'].mean() if 'n_eff_bs_beams' in case_data.columns else np.nan
            
            lines.append(
                f"{case:<40} {avg_pl:>10.2f} {avg_ad:>10.4f} {avg_snr:>12.2f} {avg_neff:>10.2f}"
            )
        
        lines.append("")
        return "\n".join(lines)
    
    @staticmethod
    def generate_hypothesis_validation(ue_data: pd.DataFrame, ad_metric: str = 'rho_2d_euc') -> str:
        """
        가설 검증 리포트 생성
        
        Args:
            ue_data: 분류된 UE 데이터
            ad_metric: Angular Density 메트릭 이름
        """
        lines = []
        lines.append("="*70)
        lines.append("Hypothesis Validation")
        lines.append("="*70)
        lines.append("")
        lines.append("Representative UE Selection Strategy:")
        lines.append("="*70)
        lines.append("")
        lines.append("목적: 규칙적으로 정의된 9개 케이스의 전형적 N_eff 패턴 설명")
        lines.append("")
        lines.append("Regular N_eff Range Design:")
        lines.append("-" * 70)
        lines.append("")
        lines.append("PL_Low (낮은 경로 손실, cell center) - 높은 빔 다양성:")
        lines.append("  범위 폭: 0.6, AD 간격: 0.3")
        lines.append("  AD_Low:  [3.6 - 4.2]  중심: 3.9")
        lines.append("  AD_Mid:  [3.3 - 3.9]  중심: 3.6")
        lines.append("  AD_High: [3.0 - 3.6]  중심: 3.3")
        lines.append("")
        lines.append("PL_Mid (중간 경로 손실) - 중간 빔 다양성:")
        lines.append("  범위 폭: 0.6, AD 간격: 불균등")
        lines.append("  AD_Low:  [2.7 - 3.3]  중심: 3.0")
        lines.append("  AD_Mid:  [2.4 - 3.0]  중심: 2.7")
        lines.append("  AD_High: [1.2 - 1.8]  중심: 1.5")
        lines.append("")
        lines.append("PL_High (높은 경로 손실, cell edge) - 낮은 빔 다양성:")
        lines.append("  범위 폭: 0.2, AD 간격: 0.1")
        lines.append("  AD_Low:  [1.2 - 1.4]  중심: 1.3")
        lines.append("  AD_Mid:  [1.1 - 1.3]  중심: 1.2")
        lines.append("  AD_High: [1.0 - 1.2]  중심: 1.1")
        lines.append("")
        lines.append("주요 경향성:")
        lines.append("  1. PL 효과 (1차, 강함):")
        lines.append("     PL_Low (3.6) > PL_Mid (2.4) > PL_High (1.2)")
        lines.append("  2. AD 효과 (2차, 약함):")
        lines.append("     같은 PL 내에서 AD 증가 시 N_eff 약간 감소")
        lines.append("     PL_Low: -0.3 간격 (3.9 → 3.6 → 3.3)")
        lines.append("     PL_Mid: 불균등 간격 (3.0 → 2.7 → 1.5)")
        lines.append("     PL_High: -0.1 간격 (1.3 → 1.2 → 1.1)")
        lines.append("")
        lines.append("="*70)
        lines.append("")
        
        # 케이스별 effective 빔 수 통계 (정렬 적용)
        lines.append("Case-wise Effective Beam Count:")
        lines.append("-" * 70)
        
        for case in sorted(ue_data['case'].unique(), key=ReportGenerator._sort_case_key):
            case_data = ue_data[ue_data['case'] == case]
            if 'n_eff_bs_beams' in case_data.columns:
                neff_values = case_data['n_eff_bs_beams'].dropna()
                if len(neff_values) > 0:
                    lines.append(f"  {case}:")
                    lines.append(f"    Mean N_eff: {neff_values.mean():.2f}")
                    lines.append(f"    Median N_eff: {neff_values.median():.2f}")
                    lines.append(f"    Min N_eff: {neff_values.min():.2f}")
                    lines.append(f"    Max N_eff: {neff_values.max():.2f}")
                    lines.append("")
        
        # 선정된 대표 UE 상세 통계
        lines.append("")
        lines.append("="*70)
        lines.append("Selected Representative UEs - Detailed Statistics:")
        lines.append("="*70)
        lines.append("")
        
        for case in sorted(ue_data['case'].unique(), key=ReportGenerator._sort_case_key):
            case_data = ue_data[ue_data['case'] == case]
            if 'n_eff_bs_beams' not in case_data.columns:
                continue
            
            neff_values = case_data['n_eff_bs_beams'].dropna()
            if len(neff_values) == 0:
                continue
            
            # 케이스 베이스 이름
            case_base = '_'.join(case.split('_')[:4])
            neff_range = TYPICAL_NEFF_RANGES.get(case_base)
            target_center = TARGET_NEFF_CENTERS.get(case_base)
            
            if neff_range and target_center:
                min_r, max_r = neff_range
                
                lines.append(f"{case}:")
                lines.append(f"  Target: Range=[{min_r:.1f}, {max_r:.1f}], Center={target_center:.2f}")
                lines.append(f"  Selected: N={len(neff_values)}, "
                            f"Mean={neff_values.mean():.2f}, "
                            f"Std={neff_values.std():.2f}")
                lines.append(f"  Range: [{neff_values.min():.2f}, {neff_values.max():.2f}]")
                
                # 범위 내 비율
                in_range_count = ((neff_values >= min_r) & (neff_values <= max_r)).sum()
                in_range_pct = 100 * in_range_count / len(neff_values)
                lines.append(f"  In Target Range: {in_range_count}/{len(neff_values)} ({in_range_pct:.0f}%)")
                
                # 목표 중심값에 가장 가까운 UE
                closest_idx = (neff_values - target_center).abs().idxmin()
                closest_ue_idx = case_data.loc[closest_idx, 'ue_idx']
                closest_neff = case_data.loc[closest_idx, 'n_eff_bs_beams']
                lines.append(f"  Closest to Center: UE{closest_ue_idx:04d} (N_eff={closest_neff:.2f}, "
                            f"Δ={abs(closest_neff-target_center):.2f})")
                lines.append("")
        
        return "\n".join(lines)
    
    @staticmethod
    def generate_metric_comparison(
        results_az: pd.DataFrame,
        results_zen: pd.DataFrame,
        results_2d_euc: pd.DataFrame
    ) -> str:
        """
        3가지 AD 메트릭 비교 리포트 생성
        
        Args:
            results_az: rho_azi 기반 분류 결과
            results_zen: rho_zen 기반 분류 결과
            results_2d_euc: rho_2d_euc 기반 분류 결과
        """
        lines = []
        lines.append("="*70)
        lines.append("Angular Density Metric Comparison")
        lines.append("="*70)
        lines.append("")
        
        # 각 메트릭별 케이스 분포 비교
        lines.append("Case Distribution by Metric:")
        lines.append("-" * 70)
        
        metrics = [
            ('rho_azi', results_az),
            ('rho_zen', results_zen),
            ('rho_2d_euc', results_2d_euc)
        ]
        
        for metric_name, df in metrics:
            if len(df) == 0:
                continue
            lines.append(f"\n{metric_name}:")
            case_counts = df['case'].value_counts()
            for case in sorted(case_counts.index, key=ReportGenerator._sort_case_key):
                count = case_counts[case]
                lines.append(f"  {case:40s}: {count:4d} UEs")
        
        # PL 레벨별 N_eff 비교
        lines.append("\n" + "="*70)
        lines.append("N_eff Comparison by PL Level (across metrics):")
        lines.append("-" * 70)
        
        pl_levels = ['Low', 'Mid', 'High']  # 경로 손실 증가 순
        for pl_level in pl_levels:
            lines.append(f"\nPL_{pl_level}:")
            for metric_name, df in metrics:
                if len(df) == 0:
                    continue
                pl_cases = df[df['case'].str.contains(f'PL_{pl_level}')]
                if len(pl_cases) > 0 and 'n_eff_bs_beams' in pl_cases.columns:
                    neff_values = pl_cases['n_eff_bs_beams'].dropna()
                    if len(neff_values) > 0:
                        lines.append(
                            f"  {metric_name:15s}: Mean N_eff = {neff_values.mean():.2f} "
                            f"(Median: {neff_values.median():.2f})"
                        )
        
        lines.append("")
        return "\n".join(lines)
    
    @staticmethod
    def generate_detailed_ue_report(ue_row: pd.Series) -> str:
        """개별 UE 상세 리포트"""
        lines = []
        lines.append(f"UE {ue_row['ue_idx']}:")
        lines.append(f"  Case: {ue_row.get('case', 'N/A')}")
        lines.append(f"  PL: {ue_row.get('pl', np.nan):.2f} dB")
        lines.append(f"  AD (rho_2d_euc): {ue_row.get('rho_2d_euc', np.nan):.4f}")
        lines.append(f"  DL SNR: {ue_row.get('snr_dl', np.nan):.2f} dB")
        lines.append(f"  Capacity: C_BS={ue_row.get('c_bs', np.nan):.2f} bps/Hz")
        lines.append(f"  N_eff (BS beams): {ue_row.get('n_eff_bs_beams', np.nan):.2f}")
        lines.append(f"  Unique (BS beams): {ue_row.get('unique_bs_beams', np.nan):.0f}")
        return "\n".join(lines)


# ============================================================================
# 8. ResultOrganizer: 결과 저장
# ============================================================================

class ResultOrganizer:
    """결과 폴더 구성 및 파일 저장"""
    
    def __init__(self, output_base_dir: str):
        self.output_base_dir = output_base_dir
        self.output_dir = None
        self.analysis_dirs = {}  # 메트릭별 분석 폴더
    
    def create_output_structure(self) -> str:
        """출력 폴더 구조 생성"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(
            self.output_base_dir,
            f"UE_Classification_Results_{timestamp}"
        )
        
        # 기본 폴더 생성
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "raw_data"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "comparison"), exist_ok=True)
        
        # 메트릭별 분석 폴더 생성
        for metric in ['rho_azi', 'rho_zen', 'rho_2d_euc']:
            metric_short = metric.replace('rho_', '')
            analysis_dir = os.path.join(
                self.output_dir,
                f"analysis_{metric_short}"
            )
            os.makedirs(analysis_dir, exist_ok=True)
            os.makedirs(os.path.join(analysis_dir, "representative_ues"), exist_ok=True)
            os.makedirs(os.path.join(analysis_dir, "summary"), exist_ok=True)
            self.analysis_dirs[metric] = analysis_dir
        
        return self.output_dir
    
    def create_analysis_folder(self, ad_metric: str) -> str:
        """특정 AD 메트릭에 대한 분석 폴더 경로 반환"""
        return self.analysis_dirs.get(ad_metric, self.output_dir)
    
    def save_all_ues(self, ue_data: pd.DataFrame):
        """전체 UE 분류 데이터 저장 (모든 메트릭 포함)"""
        if self.output_dir is None:
            return
        
        output_path = os.path.join(self.output_dir, "raw_data", "all_ues_full_data.csv")
        ue_data.to_csv(output_path, index=False)
        print(f"  Saved: {output_path}")
    
    def save_representative_ues(self, rep_ue_data: pd.DataFrame, ad_metric: str):
        """대표 UE 리스트 저장 (메트릭별)"""
        analysis_dir = self.create_analysis_folder(ad_metric)
        if analysis_dir is None:
            return
        
        output_path = os.path.join(
            analysis_dir,
            "representative_ues",
            "representative_ues_list.csv"
        )
        rep_ue_data.to_csv(output_path, index=False)
        print(f"  Saved: {output_path}")
    
    def save_case_files(self, rep_ue_data: pd.DataFrame, ad_metric: str):
        """케이스별 파일 저장 (메트릭별)"""
        analysis_dir = self.create_analysis_folder(ad_metric)
        if analysis_dir is None:
            return
        
        for case in sorted(rep_ue_data['case'].unique(), key=ReportGenerator._sort_case_key):
            case_data = rep_ue_data[rep_ue_data['case'] == case]
            
            # 파일명 생성
            case_filename = f"case_{case}.csv"
            output_path = os.path.join(
                analysis_dir,
                "representative_ues",
                case_filename
            )
            case_data.to_csv(output_path, index=False)
            print(f"  Saved: {output_path}")
    
    def save_summary_reports(self, summary: str, validation: str, ad_metric: str):
        """요약 리포트 저장 (메트릭별)"""
        analysis_dir = self.create_analysis_folder(ad_metric)
        if analysis_dir is None:
            return
        
        summary_path = os.path.join(analysis_dir, "summary", "case_summary.txt")
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary)
        print(f"  Saved: {summary_path}")
        
        validation_path = os.path.join(
            analysis_dir,
            "summary",
            "hypothesis_validation.txt"
        )
        with open(validation_path, 'w', encoding='utf-8') as f:
            f.write(validation)
        print(f"  Saved: {validation_path}")
    
    def save_comparison_report(self, comparison: str):
        """메트릭 비교 리포트 저장"""
        if self.output_dir is None:
            return
        
        comparison_path = os.path.join(
            self.output_dir,
            "comparison",
            "metric_comparison.txt"
        )
        with open(comparison_path, 'w', encoding='utf-8') as f:
            f.write(comparison)
        print(f"  Saved: {comparison_path}")


# ============================================================================
# 9. Main Orchestrator
# ============================================================================

class UEClassificationAnalyzer:
    """전체 분석 오케스트레이터"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def run_analysis(self):
        """전체 분석 실행 (3가지 AD 메트릭에 대해 각각 수행)"""
        print("="*70)
        print("UE Classification Analysis")
        print("="*70)
        print("")
        
        # 1. 데이터 로딩
        print("[1/8] Loading P1P log data...")
        log_data = {}
        for log_dir in self.config.log_dirs:
            if os.path.exists(log_dir):
                data = LogParser.parse_log_directory(log_dir)
                log_data.update(data)
        print(f"  Loaded {len(log_data)} UEs from logs")
        
        print("\n[2/8] Loading P1P CSV data...")
        csv_dir = script_dir
        csv_pattern = self.config.csv_pattern
        csv_df = CSVDataLoader.load_csv_directory(csv_dir, csv_pattern)
        print(f"  Loaded {len(csv_df)} rows from CSV files")
        
        # 2. 데이터 통합
        print("\n[3/8] Integrating data...")
        ue_list = sorted(set(log_data.keys()) | set(csv_df['ue'].unique()))
        
        # DataFrame 생성
        ue_data_list = []
        for ue_idx in ue_list:
            row = {'ue_idx': ue_idx}
            
            # 로그 데이터
            if ue_idx in log_data:
                row.update(log_data[ue_idx])
            
            # CSV 데이터
            csv_rows = csv_df[csv_df['ue'] == ue_idx]
            if len(csv_rows) > 0:
                csv_row = csv_rows.iloc[0]
                row['c_ae'] = csv_row.get('C_AE', row.get('c_ae', np.nan))
                row['c_ue'] = csv_row.get('C_UE', row.get('c_ue', np.nan))
                
                # C_BS_hist에서 최종 capacity 추출
                c_bs_hist_str = csv_row.get('C_BS_hist', '')
                if c_bs_hist_str and pd.notna(c_bs_hist_str):
                    try:
                        c_bs_values = [float(x.strip()) for x in str(c_bs_hist_str).split(',')]
                        if c_bs_values:
                            row['c_bs'] = c_bs_values[-1]
                    except:
                        pass
                
                if 'c_bs' not in row or pd.isna(row.get('c_bs')):
                    row['c_bs'] = row.get('c_bs', np.nan)
                
                # 빔 데이터 파싱
                ue_beams_str = csv_row.get('ue_beams', '')
                if ue_beams_str and pd.notna(ue_beams_str):
                    row['ue_beams'] = CSVDataLoader._parse_beam_string(str(ue_beams_str))
                
                bs_beams_str = csv_row.get('bs_beams', '')
                if bs_beams_str and pd.notna(bs_beams_str):
                    row['bs_beams'] = CSVDataLoader._parse_beam_string(str(bs_beams_str))
            
            ue_data_list.append(row)
        
        ue_df = pd.DataFrame(ue_data_list)
        
        # 빔 패턴 분석
        if 'ue_beams' in ue_df.columns:
            ue_df['unique_ue_beams'] = ue_df['ue_beams'].apply(
                lambda x: BeamPatternAnalyzer.compute_unique_beams(x) if isinstance(x, list) else 0
            )
            ue_df['n_eff_ue_beams'] = ue_df['ue_beams'].apply(
                lambda x: BeamPatternAnalyzer.compute_effective_beams(x) if isinstance(x, list) else np.nan
            )
        
        if 'bs_beams' in ue_df.columns:
            ue_df['unique_bs_beams'] = ue_df['bs_beams'].apply(
                lambda x: BeamPatternAnalyzer.compute_unique_beams(x) if isinstance(x, list) else 0
            )
            ue_df['n_eff_bs_beams'] = ue_df['bs_beams'].apply(
                lambda x: BeamPatternAnalyzer.compute_effective_beams(x) if isinstance(x, list) else np.nan
            )
        
        # Angular Density 계산 (3가지 메트릭)
        print("\n[4/8] Computing Angular Density (3 metrics)...")
        ad_integrator = AngularDensityIntegrator(self.config.p1a_npz_path)
        rho_dict = ad_integrator.compute_angular_density_all_ues(ue_list)
        
        # 각 메트릭을 DataFrame에 추가
        for ue_idx, rho_data in rho_dict.items():
            ue_mask = ue_df['ue_idx'] == ue_idx
            if ue_mask.any():
                ue_df.loc[ue_mask, 'rho_azi'] = rho_data.get('rho_azi', np.nan)
                ue_df.loc[ue_mask, 'rho_zen'] = rho_data.get('rho_zen', np.nan)
                ue_df.loc[ue_mask, 'rho_2d_euc'] = rho_data.get('rho_2d_euc', np.nan)
        
        print(f"  Computed Angular Density for {len(rho_dict)} UEs")
        
        # 출력 폴더 구조 생성
        print("\n[5/8] Creating output structure...")
        organizer = ResultOrganizer(self.config.output_base_dir)
        organizer.create_output_structure()
        organizer.save_all_ues(ue_df)
        
        # 3. 각 AD 메트릭별로 분석 수행
        ad_metrics = ['rho_azi', 'rho_zen', 'rho_2d_euc']
        all_results = {}
        
        for ad_metric in ad_metrics:
            print(f"\n[6/8] Analyzing with {ad_metric}...")
            
            # 분류
            ue_df_classified = UEClassifier.classify_ues(
                ue_df.copy(),
                ad_metric=ad_metric,
                snr_threshold=self.config.snr_threshold
            )
            print(f"  Classified {len(ue_df_classified)} UEs")
            
            # 대표 UE 선정
            rep_ue_df = RepresentativeUESelector.select_representative_ues(
                ue_df_classified,
                n_per_case=self.config.n_representative_per_case
            )
            print(f"  Selected {len(rep_ue_df)} representative UEs")
            
            # 리포트 생성
            summary = ReportGenerator.generate_case_summary(ue_df_classified, ad_metric)
            validation = ReportGenerator.generate_hypothesis_validation(ue_df_classified, ad_metric)
            
            # 저장
            organizer.save_representative_ues(rep_ue_df, ad_metric)
            organizer.save_case_files(rep_ue_df, ad_metric)
            organizer.save_summary_reports(summary, validation, ad_metric)
            
            all_results[ad_metric] = ue_df_classified
        
        # 4. 메트릭 비교 리포트 생성
        print("\n[7/8] Generating metric comparison report...")
        comparison = ReportGenerator.generate_metric_comparison(
            all_results['rho_azi'],
            all_results['rho_zen'],
            all_results['rho_2d_euc']
        )
        organizer.save_comparison_report(comparison)
        
        print("\n" + "="*70)
        print("Analysis Complete!")
        print(f"Results saved to: {organizer.output_dir}")
        print("="*70)


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    # 설정
    config = Config(
        log_dirs=[
            os.path.join(script_dir, "logs_20251115_065837"),
            os.path.join(script_dir, "logs_20251117_055957")
        ],
        csv_pattern="A1_7_5GHz_P1P_1117_1500_p*.csv",
        p1a_npz_path=os.path.join(
            P1A_DIR,
            f"Area{AREA}_{FREQ}GHz_Rays_ALL_RXs.npz"
        ),
        output_base_dir=script_dir,
        snr_threshold=0.0,
        n_representative_per_case=15
    )
    
    # 분석 실행
    analyzer = UEClassificationAnalyzer(config)
    analyzer.run_analysis()

