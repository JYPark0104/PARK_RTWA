"""
P1Q 선택된 Beam들의 방향성 패턴 및 모든 Ray 시각화 (v5)

**버전 변경사항:**
- v5: P1P v5 호환 (P1P_Config 참조, TxSNR 기반 송신 전력)
- v3: P1P v3 호환 (독립 SNR_dB 설정)

**설계 원칙:**
- P1P_Config에서 송신 전력, SNR, 하드웨어 파라미터 참조
- 설정 중복 정의 방지
- P1P 업데이트 시 P1Q 자동 반영

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
   - P1P CSV에서 자동 로드 (일반/PERT 모드)
   - 저장 경로: P1Q_Project/P1P_Beam_Patterns_{timestamp}/
   - PERT 모드: P1Q_Project/P1P_Beam_Patterns_PERT_{timestamp}/

**주요 클래스:**
1. P1PResultLoader: P1P 결과 CSV 로딩 (일반/PERT 모드 지원)
2. RayDataLoader: P1A에서 LoS/NLoS rays 로딩
3. BeamPatternPlotter: Beam/Eigenbeam 시각화
4. P1QPathManager: 경로 및 파일명 관리
5. P1P 클래스들 임포트: BeamDomainCapacity, BeamDomainTransform, BeamformingMatrix, DFTCodebook (from P1P v5)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path
from typing import List, Tuple
import pandas as pd
import os
import re
import csv

# TensorFlow (for Wen2011 optimization)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# P1P 클래스들 임포트
from P1P_BM_SWOMP_2511v5 import (
    BeamDomainCapacity,
    BeamDomainTransform,
    BeamformingMatrix,
    DFTCodebook,
    P1P_Config,
    DataLoader as P1P_DataLoader  # 추가
)


class P1Q_Config:
    """P1Q 플롯 설정 (중요도 순 배치)"""
    
    def __init__(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # ===== P1P Config 참조 (중복 정의 방지) =====
        self._p1p_config = P1P_Config()
        
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
            "A1_7_5GHz_P1P_1124_0000_p0.csv",
            "A1_7_5GHz_P1P_1124_0000_p1.csv",
            "A1_7_5GHz_P1P_1124_0000_p2.csv",
            "A1_7_5GHz_P1P_1124_0000_p3.csv",
            "A1_7_5GHz_P1P_1124_0000_p4.csv",
            "A1_7_5GHz_P1P_1124_0000_p5.csv",
            "A1_7_5GHz_P1P_1124_0000_p6.csv",
            "A1_7_5GHz_P1P_1124_0000_p7.csv",
            "A1_7_5GHz_P1P_1124_0000_p8.csv",
            "A1_7_5GHz_P1P_1124_0000_p9.csv",
            "A1_7_5GHz_P1P_1124_0000_p10.csv",
            "A1_7_5GHz_P1P_1124_0000_p11.csv",
            "A1_7_5GHz_P1P_1124_0000_p12.csv",
            "A1_7_5GHz_P1P_1124_0000_p13.csv",
            "A1_7_5GHz_P1P_1124_0000_p14.csv",
            "A1_7_5GHz_P1P_1124_0000_p15.csv"
            # "A1_7_5GHz_P1P_PERT_1113_0127.csv"  # PERT 모드: trial_id 컬럼 포함, trial 0~10
            # "A1_7_5GHz_P1P_PERT_1114_0139.csv"  # PERT 모드 최신
        ]
        
        # Runtime variables (자동 설정됨, 직접 수정 금지)
        self.csv_input_dir = None
        self.csv_files = None
        self.output_dir = None
        self.timestamp = None      # 추가: Auto 모드 전용
        self.is_pert_mode = None   # 추가: Auto 모드 전용
        
        # ===== 4. 하드웨어 구조 (P1P Config 참조) =====
        # UE hardware (N_UE,*)
        self.n_ue_layer = self._p1p_config.n_ue_layer
        self.n_ue_layer_ae = self._p1p_config.n_ue_layer_ae
        self.n_ue_ae = self._p1p_config.n_ue_ae
        self.n_ue_layer_row = self._p1p_config.n_ue_layer_row
        self.n_ue_layer_col = self._p1p_config.n_ue_layer_col
        
        # BS hardware (N_BS,*)
        self.n_bs_layer = self._p1p_config.n_bs_layer
        self.n_bs_layer_ae = self._p1p_config.n_bs_layer_ae
        self.n_bs_ae = self._p1p_config.n_bs_ae
        self.n_bs_layer_row = self._p1p_config.n_bs_layer_row
        self.n_bs_layer_col = self._p1p_config.n_bs_layer_col
        self.n_bs_full_row = 32  # BS full array (P1Q 전용, 물리적 배열)
        
        # Physical parameters
        self.antenna_spacing = 0.5  # λ
        # BS orientation: (alpha_rad, theta_bs_rad, gamma_rad) [rad]
        # alpha: Azimuth bearing, theta_bs: Downtilt, gamma: Roll
        self.bs_orientation = (
            np.radians(246.0),  # alpha_rad [rad]
            np.radians(93.0),   # theta_bs_rad [rad] (3° downtilt = 93° zenith)
            np.radians(0.0)     # gamma_rad [rad]
        )
        
        # ===== 5. DFT 코드북 파라미터 (P1P Config 참조) =====
        self.n_oversample = self._p1p_config.n_oversample
        self.n_ue_layer_beams = self._p1p_config.n_ue_layer_beams
        self.n_bs_layer_beams = self._p1p_config.n_bs_layer_beams
        
        # Selection (r_*)
        self.min_layer = self._p1p_config.min_layer  # MIMO rank & max eigenvalue size (=4, fixed)
        self.r_mode = self.n_ue_layer  # MIMO rank
        
        # ===== 6. 송신 전력 및 SNR (P1P Config 참조) =====
        # 열잡음
        self.N0_dBm_per_Hz = self._p1p_config.N0_dBm_per_Hz
        self.BW_Hz = self._p1p_config.BW_Hz
        self.N_dBm = self._p1p_config.N_dBm
        self.N_mW = self._p1p_config.N_mW
        
        # 구현 손실
        self.L_tx_bs_dB = self._p1p_config.L_tx_bs_dB
        self.L_rx_ue_dB = self._p1p_config.L_rx_ue_dB
        self.L_tx_ue_dB = self._p1p_config.L_tx_ue_dB
        self.L_rx_bs_dB = self._p1p_config.L_rx_bs_dB
        
        # 송신 전력
        self.P_ue_total_dBm = self._p1p_config.P_ue_total_dBm
        self.P_ue_total_mW = self._p1p_config.P_ue_total_mW
        self.P_bs_total_dBm = self._p1p_config.P_bs_total_dBm
        self.P_bs_total_mW = self._p1p_config.P_bs_total_mW
        
        # TxSNR
        self.TxSNR_ue = self._p1p_config.TxSNR_ue
        self.TxSNR_ue_layer = self._p1p_config.TxSNR_ue_layer
        self.TxSNR_bs = self._p1p_config.TxSNR_bs
        self.TxSNR_bs_layer = self._p1p_config.TxSNR_bs_layer
        
        # ===== 7. Wen2011 최적화 파라미터 =====
        self.outer_max_iter = 50
        self.outer_eps = 0.1
        self.inner_max_iter = 50
        self.regularization = 1e-20
        
        # ===== 8. 빔 패턴 계산 해상도 =====
        self.azimuth_step = 1.0  # deg
        self.elevation_step = 1.0  # deg
        self.beam_max_gain_azimuth_step = 5.0  # deg
        self.beam_max_gain_elevation_step = 5.0  # deg
        
        # ===== 9. 시각화 범위 =====
        self.power_vmin = -180  # dB
        self.power_vmax = -80   # dB
        self.ylim_range = 30.0  # dB
        
        # ===== 10. 시각화 크기 =====
        self.figure_width = 12  # inches
        self.figure_height_per_beam = 5  # inches
        self.figure_dpi = 150
        
        # ===== 11. Ray 시각화 스타일 =====
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
        - Auto mode: CSV에서 자동 로드, P1P 폴더 저장 (timestamp 및 모드 기반)
        """
        if self.manual_mode:
            # Manual mode: CSV 사용 안함, P1A/P1I 사용
            self.csv_input_dir = None
            self.csv_files = None
            self.timestamp = None
            self.is_pert_mode = False
            self.output_dir = os.path.join(self.P1Q_OUTPUT_DIR, "Manual_Beam_Patterns")
            os.makedirs(self.output_dir, exist_ok=True)
        else:
            # Auto mode: CSV + P1A + P1I 모두 사용 (P1P만 지원)
            self.csv_input_dir = self.P1P_INPUT_DIR
            self.csv_files = self.csv_files_p1p
            
            # CSV 파일명에서 timestamp와 PERT 모드 추출
            first_csv = self.csv_files[0]
            self.timestamp = P1QPathManager.extract_timestamp_from_csv(first_csv)
            self.is_pert_mode = P1QPathManager.detect_pert_from_csv_filename(first_csv)
            
            # 출력 폴더 생성
            self.output_dir = P1QPathManager.generate_output_folder(
                self.P1Q_OUTPUT_DIR, self.timestamp, self.is_pert_mode)
            os.makedirs(self.output_dir, exist_ok=True)


class P1QPathManager:
    """
    P1Q 경로 관리 (SRP: 파일시스템 경로 생성 전담)
    
    폴더명, 파일명을 일관되게 생성하여 출력 구조 체계화
    CSV 파일명 파싱, 출력 경로 생성, 플롯 파일명 생성 담당
    """
    
    @staticmethod
    def extract_timestamp_from_csv(csv_filename: str) -> str:
        """
        CSV 파일명에서 타임태그 추출
        
        Pattern: A{area}_{freq}GHz_P1P{_PERT}_{MMDD_HHMM}{_pN}.csv
        
        Args:
            csv_filename: "A1_7.5GHz_P1P_PERT_1113_0127.csv" or "A1_7_5GHz_P1P_1124_0000_p0.csv"
        
        Returns:
            "1113_0127" 또는 "1124_0000" 또는 "unknown"
        """
        # Support both formats: with/without partition suffix (_pN)
        match = re.search(r'P1P(?:_PERT)?_(\d{4}_\d{4})(?:_p\d+)?\.csv', csv_filename)
        return match.group(1) if match else "unknown"
    
    @staticmethod
    def detect_pert_from_csv_filename(csv_filename: str) -> bool:
        """
        CSV 파일명에서 PERT 모드 감지
        
        Args:
            csv_filename: "A1_7.5GHz_P1P_PERT_1113_0127.csv"
        
        Returns:
            True if "_PERT_" in filename, else False
        """
        return "_PERT_" in csv_filename
    
    @staticmethod
    def generate_output_folder(base_dir: str, timestamp: str, is_pert: bool) -> str:
        """
        Auto 모드 출력 폴더 경로 생성
        
        Args:
            base_dir: P1Q_OUTPUT_DIR
            timestamp: "1113_0127"
            is_pert: PERT 모드 여부
        
        Returns:
            일반: "{base_dir}/P1P_Beam_Patterns_{timestamp}"
            PERT: "{base_dir}/P1P_Beam_Patterns_PERT_{timestamp}"
        """
        mode_suffix = "_PERT" if is_pert else ""
        folder_name = f"P1P_Beam_Patterns{mode_suffix}_{timestamp}"
        return os.path.join(base_dir, folder_name)
    
    @staticmethod
    def generate_codebook_filename(ue: int, trial_id: int = None, is_manual: bool = False) -> str:
        """
        Codebook PNG 파일명 생성
        
        Args:
            ue: UE 번호
            trial_id: Trial ID (None=일반, 0~10=PERT)
            is_manual: Manual 모드 여부
        
        Returns:
            Manual: "ue_{ue:04d}_codebook_manual.png"
            일반: "ue_{ue:04d}_codebook.png"
            PERT: "ue_{ue:04d}_trial_{trial_id:02d}_codebook.png"
        """
        if is_manual:
            return f"ue_{ue:04d}_codebook_manual.png"
        elif trial_id is not None:
            return f"ue_{ue:04d}_trial_{trial_id:02d}_codebook.png"
        else:
            return f"ue_{ue:04d}_codebook.png"
    
    @staticmethod
    def generate_eigenbeam_filename(ue: int, trial_id: int = None, is_manual: bool = False) -> str:
        """
        Eigenbeam PNG 파일명 생성
        
        Returns:
            Manual: "ue_{ue:04d}_eigenbeam_manual.png"
            일반: "ue_{ue:04d}_eigenbeam.png"
            PERT: "ue_{ue:04d}_trial_{trial_id:02d}_eigenbeam.png"
        """
        if is_manual:
            return f"ue_{ue:04d}_eigenbeam_manual.png"
        elif trial_id is not None:
            return f"ue_{ue:04d}_trial_{trial_id:02d}_eigenbeam.png"
        else:
            return f"ue_{ue:04d}_eigenbeam.png"


class P1PResultLoader:
    """
    P1P 결과 CSV 로딩 (Auto 모드 전용)
    
    P1P에서 생성한 빔 선택 결과 CSV를 읽어 UE/빔 정보 추출
    일반 모드와 PERT 모드 모두 지원
    """
    
    def __init__(self, config: P1Q_Config):
        self.config = config
        self.csv_dir = Path(config.csv_input_dir)
    
    # ===== Helper 메서드 =====
    
    @staticmethod
    def detect_csv_format(csv_path: str) -> dict:
        """
        CSV 헤더 분석하여 형식 정보 반환
        
        첫 번째 행(헤더)을 읽어 trial_id 컬럼 존재 확인
        
        Args:
            csv_path: CSV 파일 전체 경로
        
        Returns:
            {
                'has_trial_id': bool,
                'columns': list
            }
        """
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)
            return {
                'has_trial_id': 'trial_id' in headers,
                'columns': headers
            }
    
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
    
    # ===== PERT 모드 전용 메서드 =====
    
    def load_all_trials_for_ue(self, csv_files: List[str], target_ue: int) -> np.ndarray:
        """
        특정 UE의 모든 trial_id 추출 (PERT 모드 전용)
        
        CSV에서 ue==target_ue인 행들의 trial_id 추출
        
        Args:
            csv_files: CSV 파일 경로 리스트
            target_ue: 대상 UE 번호
        
        Returns:
            sorted unique trial_ids: [0, 1, 2, ..., 10]
        """
        all_trials = []
        
        for csv_file in csv_files:
            csv_path = self.csv_dir / csv_file
            if not csv_path.exists():
                continue
            
            df = pd.read_csv(csv_path)
            if 'ue' not in df.columns or 'trial_id' not in df.columns:
                continue
            
            # Filter target UE
            target_df = df[df['ue'] == target_ue]
            
            if len(target_df) == 0:
                continue
            
            # Extract trial_ids
            trials_in_file = target_df['trial_id'].unique().tolist()
            all_trials.extend(trials_in_file)
        
        if not all_trials:
            raise ValueError(f"UE {target_ue} trials not found in P1P CSV files")
        
        unique_trials = np.unique(all_trials)
        return np.sort(unique_trials)
    
    def load_ue_beams_for_trial(self, csv_files: List[str], target_ue: int, trial_id: int) -> np.ndarray:
        """
        특정 (UE, trial)의 UE 빔 추출 (PERT 모드 전용)
        
        조건: ue==target_ue AND trial_id==trial_id
        
        Args:
            csv_files: CSV 파일 경로 리스트
            target_ue: 대상 UE 번호
            trial_id: Trial ID
        
        Returns:
            UE beams array (unique sorted)
        """
        for csv_file in csv_files:
            csv_path = self.csv_dir / csv_file
            if not csv_path.exists():
                continue
            
            df = pd.read_csv(csv_path)
            if 'ue' not in df.columns or 'trial_id' not in df.columns or 'ue_beams' not in df.columns:
                continue
            
            # Filter target UE and trial_id
            target_df = df[(df['ue'] == target_ue) & (df['trial_id'] == trial_id)]
            
            if len(target_df) == 0:
                continue
            
            # Parse ue_beams (첫 번째 row만 사용, 중복 없음)
            ue_beams_str = target_df['ue_beams'].iloc[0]
            if pd.isna(ue_beams_str):
                continue
            
            ue_beams = self._parse_beam_string(ue_beams_str)
            return np.array(sorted(set(ue_beams)))
        
        raise ValueError(f"UE {target_ue} trial {trial_id} ue_beams not found in P1P CSV files")
    
    def load_bs_beams_for_trial(self, csv_files: List[str], target_ue: int, trial_id: int) -> np.ndarray:
        """
        특정 (UE, trial)의 BS 빔 추출 (PERT 모드 전용)
        
        조건: ue==target_ue AND trial_id==trial_id
        
        Args:
            csv_files: CSV 파일 경로 리스트
            target_ue: 대상 UE 번호
            trial_id: Trial ID
        
        Returns:
            BS beams array (중복 포함, 순서 유지)
        """
        all_beams = []
        
        for csv_file in csv_files:
            csv_path = self.csv_dir / csv_file
            if not csv_path.exists():
                continue
            
            df = pd.read_csv(csv_path)
            if 'ue' not in df.columns or 'trial_id' not in df.columns or 'bs_beams' not in df.columns:
                continue
            
            # Filter target UE and trial_id
            target_df = df[(df['ue'] == target_ue) & (df['trial_id'] == trial_id)]
            
            if len(target_df) == 0:
                continue
            
            # Parse beam list (모든 row 사용, 중복 포함)
            for beams_str in target_df['bs_beams'].dropna():
                all_beams.extend(self._parse_beam_string(beams_str))
        
        if not all_beams:
            raise ValueError(f"UE {target_ue} trial {trial_id} bs_beams not found in P1P CSV files")
        
        return np.array(all_beams)


class RayDataLoader:
    """P1A Ray 데이터 로딩 (LoS/NLoS 모두)"""
    
    def __init__(self, config: P1Q_Config):
        self.config = config
        # P1A 경로를 config에서 가져옴 (area, freq는 나중에 동적으로 구성)
        self.p1a_base_path = Path(config.P1A_INPUT_DIR)
        # P1P DataLoader 인스턴스 생성 (정규화 메서드 사용)
        self._p1p_data_loader = P1P_DataLoader(config._p1p_config)
    
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
        theta_gcs_deg = rays_data['theta_t_deg'][mask].ravel()
        phi_gcs_deg = rays_data['phi_t_deg'][mask].ravel()
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
            theta_gcs_deg[final_mask],
            phi_gcs_deg[final_mask],
            power[final_mask],
            cluster_ids[final_mask]
        ], axis=1)
        
        return rays_gcs, los_flags[final_mask], cluster_ids[final_mask]
    
    def transform_rays_to_lcs(self, rays_gcs: np.ndarray) -> np.ndarray:
        """
        GCS rays를 LCS로 변환
        
        Args:
            rays_gcs: [N, 4] array
                - Column 0: theta_gcs [deg] - GCS Zenith angle
                - Column 1: phi_gcs [deg] - GCS Azimuth angle
                - Column 2: power - Ray power
                - Column 3: cluster_idx - Cluster index
        
        Returns:
            rays_lcs: [N, 4] array
                - Column 0: theta_lcs [deg] - LCS Zenith angle
                - Column 1: phi_lcs [deg] - LCS Azimuth angle
                - Column 2: power - Ray power
                - Column 3: cluster_idx - Cluster index
        """
        # BS orientation from config (already in radians)
        # Note: Config stores theta_bs as Zenith angle, but rotation matrix needs Downtilt
        alpha_rad, theta_bs_zenith_rad, gamma_rad = self.config.bs_orientation
        
        # Convert Zenith to Downtilt for rotation matrix
        # Downtilt = Zenith - 90° (for standard definition where 90° = horizontal)
        downtilt_rad = theta_bs_zenith_rad - np.radians(90.0)
        
        rays_lcs = []
        for i in range(rays_gcs.shape[0]):
            # Input angles from rays_gcs are in degrees
            theta_gcs_deg = rays_gcs[i, 0]
            phi_gcs_deg = rays_gcs[i, 1]
            power = rays_gcs[i, 2]
            cluster_idx = rays_gcs[i, 3]
            
            # Convert to radians for transformation
            theta_gcs_rad = np.radians(theta_gcs_deg)
            phi_gcs_rad = np.radians(phi_gcs_deg)
            
            # GCS → LCS transformation (all inputs in radians)
            # Note: _gcs_to_lcs expects downtilt_rad, not zenith_rad
            theta_lcs_rad, phi_lcs_rad = self._gcs_to_lcs(
                theta_gcs_rad, phi_gcs_rad, alpha_rad, downtilt_rad, gamma_rad
            )
            
            # Convert back to degrees for output
            rays_lcs.append([
                np.degrees(theta_lcs_rad),
                np.degrees(phi_lcs_rad),
                power,
                cluster_idx
            ])
        
        return np.array(rays_lcs)
    
    def _gcs_to_lcs(self, theta_gcs_rad, phi_gcs_rad, alpha_rad, theta_bs_rad, gamma_rad):
        """GCS → LCS 변환 (3GPP TS 38.901, P1H/P1F 기반 수정)
        
        Args:
            theta_gcs_rad: GCS Zenith angle [rad]
            phi_gcs_rad: GCS Azimuth angle [rad]
            alpha_rad: BS Azimuth orientation [rad]
            theta_bs_rad: BS Downtilt angle [rad] (NOT Zenith, but downtilt itself)
            gamma_rad: BS Roll orientation [rad]
        
        Returns:
            theta_lcs_rad: LCS Zenith angle [rad]
            phi_lcs_rad: LCS Azimuth angle [rad]
        """
        # GCS direction vector
        sin_theta = np.sin(theta_gcs_rad)
        cos_theta = np.cos(theta_gcs_rad)
        sin_phi = np.sin(phi_gcs_rad)
        cos_phi = np.cos(phi_gcs_rad)
        
        rho_hat = np.array([
            sin_theta * cos_phi,
            sin_theta * sin_phi,
            cos_theta
        ])
        
        # Forward rotation matrix: R = R_z(alpha) * R_y(beta) * R_x(gamma)
        # where beta = theta_bs (downtilt)
        a, b, c = alpha_rad, theta_bs_rad, gamma_rad
        
        row_1 = np.array([
            np.cos(a) * np.cos(b),
            np.cos(a) * np.sin(b) * np.sin(c) - np.sin(a) * np.cos(c),
            np.cos(a) * np.sin(b) * np.cos(c) + np.sin(a) * np.sin(c)
        ])
        
        row_2 = np.array([
            np.sin(a) * np.cos(b),
            np.sin(a) * np.sin(b) * np.sin(c) + np.cos(a) * np.cos(c),
            np.sin(a) * np.sin(b) * np.cos(c) - np.cos(a) * np.sin(c)
        ])
        
        row_3 = np.array([
            -np.sin(b),
            np.cos(b) * np.sin(c),
            np.cos(b) * np.cos(c)
        ])
        
        R = np.array([row_1, row_2, row_3])
        
        # Reverse rotation matrix: R_inv = R.T (transpose = inverse for rotation matrices)
        R_inv = R.T
        
        # Rotate
        rot_rho = R_inv @ rho_hat
        
        # LCS angles
        z = rot_rho[2]
        z = np.clip(z, -1.0, 1.0)
        theta_lcs_rad = np.arccos(z)
        
        # phi_prime = angle(v2 @ rot_rho), where v2 = [1+0j, 1j, 0]
        v2 = np.array([1+0j, 1j, 0])
        phi_complex = np.dot(v2, rot_rho.astype(complex))
        phi_lcs_rad = np.angle(phi_complex)
        
        return theta_lcs_rad, phi_lcs_rad
    
    def load_weichselberger_for_ue(self, ue: int, area: int = None, freq: float = None):
        """
        특정 UE의 Weichselberger 파라미터 로딩 (P1I, P1P v5 호환)
        
        Args:
            ue: UE index
            area: Area index (None: config.default_area)
            freq: Frequency in GHz (None: config.default_freq)
        
        Returns:
            U_bs_full: [1024, 1024] tf.Tensor complex64 BS eigenvector
            U_ue_full: [16, 16] tf.Tensor complex64 UE eigenvector
            Omega_ul_norm: [1024, 16] tf.Tensor float32 normalized coupling matrix (UPLINK: [BS, UE])
            H_mean_ul_norm: [1024, 16] tf.Tensor complex64 normalized mean channel (UPLINK: [BS, UE])
            pl_scalar: float (Python scalar) pathloss scalar
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
                
                # P1P DataLoader 메서드 직접 호출 (중복 제거)
                Omega_norm, H_mean_norm, pl_scalar = self._p1p_data_loader._extract_pl_and_normalize(
                    Omega, H_mean, U_bs.shape[0], U_ue.shape[0]
                )
                
                return (
                    tf.constant(U_bs, dtype=tf.complex64),
                    tf.constant(U_ue, dtype=tf.complex64),
                    tf.constant(Omega_norm, dtype=tf.float32),
                    tf.constant(H_mean_norm, dtype=tf.complex64),
                    pl_scalar  # Python float
                )
        
        raise ValueError(f"UE {ue} not found in P1I chunks for Area {area}, {freq} GHz")


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


class BeamPatternAnalyzer:
    """Beam 방향성 패턴 계산"""
    
    def __init__(self, config: P1Q_Config):
        self.config = config
        # DFTCodebook은 static method만 사용하므로 인스턴스 불필요
        self.sv_gen = SteeringVectorGenerator()
        self.F = None  # Layer-level [16, 64]
    
    def _ensure_codebook(self):
        """DFT codebook 생성 (lazy initialization, P1P DFTCodebook 사용)"""
        if self.F is None:
            # BS Layer-level codebook [16, 64] only
            # P1P DFTCodebook 사용 (TensorFlow → NumPy 변환)
            F_tf = DFTCodebook.generate_2d_dft_codebook_tf(
                self.config.n_bs_layer_row,
                self.config.n_oversample
            )
            self.F = F_tf.numpy()  # NumPy 배열로 변환 (기존 코드 호환성)
    
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
                                      Omega_ul_norm: tf.Tensor, H_mean_ul_norm: tf.Tensor):
        """
        선택된 BS 빔에 대한 DL 빔 도메인 파라미터 추출 (P1P Stage 2와 동일, P1P v5 호환)
        
        Args:
            beam_indices: [K] BS beam indices (layer-level)
            ue_beams: [r_ue] UE beam indices (TRX-level)
            U_bs_full: [1024, 1024] BS eigenvector
            U_ue_full: [16, 16] UE eigenvector
            Omega_ul_norm: [1024, 16] normalized coupling matrix UPLINK [RX=BS, TX=UE]
            H_mean_ul_norm: [1024, 16] normalized mean channel UPLINK [RX=BS, TX=UE]
        
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
        
        # DL parameters (P1P Stage 2와 동일) - 정규화된 채널 사용
        Omega_dl = tf.transpose(Omega_ul_norm)  # [16, 1024]
        H_mean_dl = tf.linalg.adjoint(H_mean_ul_norm)  # [16, 1024]
        
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
                                Omega_beam_dl: tf.Tensor, H_mean_beam_dl: tf.Tensor,
                                pl_scalar: float):
        """
        DL BS transmit covariance 최적화 (Wen2011, P1P v5 완전 호환)
        
        Args:
            U_ue_beam: [4, 4] UE eigenvector (DL RX)
            U_bs_beam: [K, K] BS eigenvector (DL TX)
            Omega_beam_dl: [4, K] normalized coupling matrix [RX=UE, TX=BS]
            H_mean_beam_dl: [4, K] normalized mean channel [RX=UE, TX=BS]
            pl_scalar: float, pathloss scalar (from P1P DataLoader)
        
        Returns:
            C_bs: float, DL capacity
            P_bs: [K, K] BS transmit covariance
            Lambda_bs: [min_layer] top eigenvalues (descending, P1P 방식)
            V_bs: [K, min_layer] top eigenvectors (P1P 방식)
        """
        # P1P BeamDomainCapacity 사용 (P1Q Config가 P1P Config 참조)
        capacity_calc = BeamDomainCapacity(self.config)
        
        # Power budget 계산 (P1P v5 방식 - Line 2073-2075 참조)
        TxSNR_bs_tf = tf.constant(self.config.TxSNR_bs, dtype=tf.float32)
        pl_scalar_tf = tf.constant(pl_scalar, dtype=tf.float32)
        power_budget = TxSNR_bs_tf * pl_scalar_tf
        
        # Wen2011 DL optimization (P1P v5 인터페이스 완전 준수)
        C_bs, P_bs, Lambda_P_r, k_final, eps_final = capacity_calc.wen2011_optimize_tf(
            U_ue_beam,          # [4, 4] RX: DL UE receive
            U_bs_beam,          # [K, K] TX: DL BS transmit (optimize)
            Omega_beam_dl,      # [4, K] [RX, TX] order
            H_mean_beam_dl,     # [4, K] [RX, TX] order
            power_budget        # TX power budget (scalar)
        )
        
        # Eigenvalue decomposition (P1P 방식)
        eigvals, eigvecs = tf.linalg.eigh(P_bs)
        eigvals = tf.math.real(eigvals)
        
        # Sort descending
        sorted_indices = tf.argsort(eigvals, direction='DESCENDING')
        eigvals_sorted = tf.gather(eigvals, sorted_indices)
        eigvecs_sorted = tf.gather(eigvecs, sorted_indices, axis=1)
        
        # 상위 min_layer개만 추출 (P1P Line 1798)
        Lambda_bs = eigvals_sorted[:self.config.min_layer]
        V_bs = eigvecs_sorted[:, :self.config.min_layer]
        
        return C_bs.numpy(), P_bs, Lambda_bs.numpy(), V_bs.numpy()
    
    def compute_final_beamforming_vectors(self, W_bs: tf.Tensor,
                                         V_bs: np.ndarray, Lambda_bs: np.ndarray) -> np.ndarray:
        """
        Final beamforming vectors (eigenbeam 계산)
        
        Args:
            W_bs: [1024, K] BS beamforming matrix (directional beams)
            V_bs: [K, min_layer] top eigenvectors (P1P 방식)
            Lambda_bs: [min_layer] top eigenvalues (사용 안 함, 호환성)
        
        Returns:
            w_final: [min_layer, 1024] eigenbeams (AE-level)
        """
        # Digital beamforming [K, min_layer]: eigenvectors × sqrt(eigenvalues)
        sqrt_eigvals = np.sqrt(np.maximum(Lambda_bs, 0))
        D = V_bs * sqrt_eigvals[np.newaxis, :]
        D_tf = tf.constant(D, dtype=tf.complex64)
        
        # W_final = W_BS @ D [1024, min_layer]
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
        
        # Sort beam_indices by count (descending) - most selected beams first
        beam_indices_sorted = sorted(beam_indices, key=lambda idx: beam_counts.get(idx, 0), reverse=True)
        # Limit to first MAX_BEAMS after sorting
        if len(beam_indices_sorted) > MAX_BEAMS:
            beam_indices_sorted = beam_indices_sorted[:MAX_BEAMS]
        
        # Create figure (2 columns, K rows)
        n_beams = len(beam_indices_sorted)
        fig = plt.figure(figsize=(self.config.figure_width, 
                                  self.config.figure_height_per_beam * n_beams))
        
        for row_idx, beam_idx in enumerate(beam_indices_sorted):
            beam_count = beam_counts.get(beam_idx, 0)
            
            # Compute beam's global maximum gain and peak direction
            beam_max_gain, peak_zenith_deg, peak_azimuth_deg = self._compute_codebook_beam_max_gain(beam_idx)
            
            # Subplot 1: Azimuth cut (at beam's peak zenith)
            ax1 = fig.add_subplot(n_beams, 2, row_idx * 2 + 1, projection='polar')
            self._plot_codebook_azimuth_cut(ax1, ue, beam_idx, beam_count, beam_max_gain, peak_zenith_deg, 
                                  rays_lcs, los_flags, cluster_ids, add_colorbar=(row_idx == 0))
            
            # Subplot 2: Zenith cut (at beam's peak azimuth)
            ax2 = fig.add_subplot(n_beams, 2, row_idx * 2 + 2, projection='polar')
            # Add colorbar only to the first row (top-right subplot)
            self._plot_codebook_zenith_cut(ax2, ue, beam_idx, beam_count, beam_max_gain, peak_azimuth_deg, 
                                 rays_lcs, los_flags, cluster_ids, add_colorbar=False)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.config.figure_dpi, bbox_inches='tight')
        plt.close()
    
    def _compute_codebook_beam_max_gain(self, beam_idx: int) -> Tuple[float, float, float]:
        """
        코드북 빔의 전체 3D 패턴에서 최대 gain 및 peak direction 계산
        
        Args:
            beam_idx: Beam index (layer-level)
        
        Returns:
            max_gain: Maximum gain across all directions (linear scale)
            peak_zenith_deg: Peak direction zenith angle [deg]
            peak_azimuth_deg: Peak direction azimuth angle [deg]
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
        peak_zenith_deg = 0.0
        peak_azimuth_deg = 0.0
        
        for phi_deg in azimuth_vals:
            for elev_deg in elevation_vals:
                theta_zenith_rad = np.radians(90.0 - elev_deg)
                phi_rad = np.radians(phi_deg)
                h = self.beam_analyzer.sv_gen.generate(
                    theta_zenith_rad, phi_rad, self.beam_analyzer.config
                )
                gain = np.abs(np.vdot(w, h))**2
                if gain > max_gain:
                    max_gain = gain
                    peak_zenith_deg = 90.0 - elev_deg  # Convert elevation to zenith
                    peak_azimuth_deg = phi_deg
        
        return max_gain, peak_zenith_deg, peak_azimuth_deg
    
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
            theta_lcs_deg = cluster_rays[:, 0]  # LCS Zenith [deg]
            phi_lcs_deg = cluster_rays[:, 1]    # LCS Azimuth [deg]
            power = cluster_rays[:, 2]
            
            is_los = np.any(cluster_los)
            
            # Check if cluster has large spread
            phi_mean_rad = np.arctan2(np.mean(np.sin(np.radians(phi_lcs_deg))),
                                   np.mean(np.cos(np.radians(phi_lcs_deg))))
            phi_mean_deg = np.degrees(phi_mean_rad)
            phi_diff = np.angle(np.exp(1j * np.radians(phi_lcs_deg - phi_mean_deg)), deg=True)
            phi_spread = np.max(phi_diff) - np.min(phi_diff)
            
            # If spread > threshold, draw individual rays with small width
            if phi_spread > self.config.phi_spread_threshold_azimuth:
                individual_ray_width = self.config.individual_ray_width
                
                for i in range(len(phi_lcs_deg)):
                    phi_ray_deg = phi_lcs_deg[i]
                    power_ray = power[i]
                    
                    # Power (in dB)
                    power_ray_db = 10 * np.log10(power_ray + 1e-20)
                    
                    # Color by cluster index
                    color = self._get_cluster_color(cluster_id)
                    
                    # Plot individual ray (LCS 기준)
                    phi_center_rad = np.radians( - phi_ray_deg)
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
                
                # Plot wedge (LCS 기준)
                phi_center_rad = np.radians( - phi_mean_deg)
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
    
    def plot_eigenbeams(self, ue: int, W_bs: np.ndarray, V_bs: np.ndarray, Lambda_bs: np.ndarray, output_path: str):
        """
        Eigenbeam pattern plot with PADP (individual eigenbeams)
        
        Args:
            ue: UE index
            W_bs: [1024, K] BS beamforming matrix (directional beams)
            V_bs: [K, min_layer] top eigenvectors (P1P 방식)
            Lambda_bs: [min_layer] top eigenvalues (for display only, not used in gain calculation)
            output_path: Output PNG file path
        """
        # Compute eigenbeam vectors without eigenvalues (eigenvectors only)
        # W_eigenbeam = W_bs @ V_bs [1024, min_layer]
        W_eigenbeam = W_bs @ V_bs
        # Load rays for PADP computation
        rays_gcs, los_flags, cluster_ids = self.ray_loader.load_rays_for_ue(ue, include_los=True, include_nlos=True)
        rays_lcs = self.ray_loader.transform_rays_to_lcs(rays_gcs)
        
        # Compute power-weighted mean across ALL rays using circular averaging
        padp_theta, padp_phi = self._compute_circular_power_weighted_mean(rays_lcs)
        
        # Identify significant eigenbeams (eigenvalue > threshold)
        eigval_threshold = 0.01
        significant_indices = np.where(Lambda_bs > eigval_threshold)[0]
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
            eigval_single = Lambda_bs[beam_idx]
            
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
            ax1.set_title(f'UE {ue} EB{beam_idx+1} (λ_BS={eigval_single:.2e}) Azimuth\n(Zenith={zenith_deg_conj:.1f}°)', fontsize=10)
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
            ax2.set_title(f'UE {ue} EB{beam_idx+1} (λ_BS={eigval_single:.2e}) Zenith\n(Azimuth={azimuth_deg_conj:.1f}°)', fontsize=10)
            ax2.grid(True)
            ax2.legend(fontsize=8, loc='center left')  # 코드북 플롯과 동일
        
        # Suptitle with all eigenvalues (BS transmit power, up to 4)
        eigval_str = ', '.join([f'{v:.2e}' for v in Lambda_bs[:min(4, len(Lambda_bs))]])
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
            
            phi_lcs_deg = cluster_rays[:, 1]  # LCS Azimuth [deg]
            power = cluster_rays[:, 2]
            is_los = np.any(cluster_los)
            
            # Check spread
            phi_mean_rad = np.arctan2(np.mean(np.sin(np.radians(phi_lcs_deg))),
                                   np.mean(np.cos(np.radians(phi_lcs_deg))))
            phi_mean_deg = np.degrees(phi_mean_rad)
            phi_diff = np.angle(np.exp(1j * np.radians(phi_lcs_deg - phi_mean_deg)), deg=True)
            phi_spread = np.max(phi_diff) - np.min(phi_diff)
            
            if phi_spread > self.config.phi_spread_threshold_azimuth:
                # Individual rays
                for i in range(len(phi_lcs_deg)):
                    phi_ray_deg = phi_lcs_deg[i]
                    power_ray = power[i]
                    power_ray_db = 10 * np.log10(power_ray + 1e-20)
                    
                    # Color by cluster index
                    color = self._get_cluster_color(cluster_id)
                    
                    # PADP ray: LCS 기준 (phase conjugate 제거)
                    phi_center_rad = np.radians( - phi_ray_deg)
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
                
                # PADP wedge: LCS 기준 (phase conjugate 제거)
                phi_center_rad = np.radians( - phi_mean_deg)
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


def run_manual_mode(config, ray_loader, beam_analyzer, plotter):
    """
    Manual 모드 실행
    
    데이터 소스:
    - UE/빔: config에서 직접 지정
    - P1A: Ray 데이터
    - P1I: Weichselberger 파라미터
    """
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
        U_bs_full, U_ue_full, Omega_ul_norm, H_mean_ul_norm, pl_scalar = \
            ray_loader.load_weichselberger_for_ue(config.manual_ue)
        
        # Manual mode: ue_beams는 manual_ue_beams 사용 (없으면 기본값)
        if len(config.manual_ue_beams) > 0:
            ue_beams = np.array(config.manual_ue_beams)
        else:
            # 기본값: identity matrix에 해당하는 빔 인덱스 (0, 1, 2, 3)
            ue_beams = np.arange(config.n_ue_layer)
        
        # DL beam domain extraction (use full beam list with repetitions)
        U_ue_beam, U_bs_beam, Omega_beam_dl, H_mean_beam_dl, W_bs, W_ue = beam_analyzer.extract_beam_domain_params_dl(
            all_beams, ue_beams, U_bs_full, U_ue_full, Omega_ul_norm, H_mean_ul_norm
        )
        
        # DL Wen2011 optimization
        print(f"  Running DL Wen2011 optimization...")
        C_bs, P_bs, Lambda_bs, V_bs = beam_analyzer.optimize_bs_transmit_dl(
            U_ue_beam, U_bs_beam, Omega_beam_dl, H_mean_beam_dl, pl_scalar
        )
        print(f"  Capacity: {C_bs:.2f} bps/Hz")
        # P1P 방식: scientific notation (Line 1803-1805)
        lambda_str = ', '.join(f"{v:.2e}" for v in Lambda_bs)
        lambda_sum = Lambda_bs.sum()
        print(f"  P_BS eigenvalues (sum={lambda_sum:.2e}): {lambda_str}")
        
        has_eigenbeam = True
    except Exception as e:
        print(f"  Warning: Wen2011 optimization failed: {e}")
        print(f"  Proceeding with codebook-only plot")
        has_eigenbeam = False
    
    # Generate plots
    print(f"\n[4/4] Generating plots...")
    codebook_file = P1QPathManager.generate_codebook_filename(config.manual_ue, is_manual=True)
    eigenbeam_file = P1QPathManager.generate_eigenbeam_filename(config.manual_ue, is_manual=True)
    
    output_path_codebook = Path(config.output_dir) / codebook_file
    plotter.plot_codebook_beams(config.manual_ue, np.array(unique_beams_display), beam_counts, str(output_path_codebook))
    print(f"  Codebook plot saved: {output_path_codebook.name}")
    
    if has_eigenbeam:
        output_path_eigenbeam = Path(config.output_dir) / eigenbeam_file
        plotter.plot_eigenbeams(config.manual_ue, W_bs.numpy(), V_bs, Lambda_bs, str(output_path_eigenbeam))
        print(f"  Eigenbeam plot saved: {output_path_eigenbeam.name}")
    
    print(f"\nComplete! Generated plots for UE {config.manual_ue}")


def run_auto_mode(config, ray_loader, beam_analyzer, plotter):
    """
    Auto 모드 실행
    
    데이터 소스:
    - UE/빔: P1P CSV
    - P1A + P1I
    """
    print("\n[2/4] Auto Mode: Loading P1P results from CSV...")
    loader = P1PResultLoader(config)
    csv_files = [os.path.join(config.csv_input_dir, f) for f in config.csv_files]
    
    # CSV 형식 재확인
    csv_format = loader.detect_csv_format(csv_files[0])
    
    print(f"  CSV timestamp: {config.timestamp}")
    print(f"  PERT mode (filename): {config.is_pert_mode}")
    print(f"  PERT mode (header): {csv_format['has_trial_id']}")
    print(f"  Output folder: {config.output_dir}")
    
    # UE 목록 로드
    all_ues = loader.load_all_ues(csv_files)
    print(f"  Total UEs: {len(all_ues)}")
    
    # 모드별 처리
    if config.is_pert_mode and csv_format['has_trial_id']:
        process_pert_mode(config, loader, ray_loader, beam_analyzer, plotter, csv_files, all_ues)
    else:
        process_normal_mode(config, loader, ray_loader, beam_analyzer, plotter, csv_files, all_ues)


def process_normal_mode(config, loader, ray_loader, beam_analyzer, plotter, csv_files, all_ues):
    """일반 모드: trial 없음"""
    print("\n[3/4] Processing normal mode...")
    
    n_ues_processed = 0
    for ue_idx, target_ue in enumerate(all_ues, 1):
        print(f"\n  [{ue_idx}/{len(all_ues)}] UE {target_ue}")
        
        # P1P CSV에서 빔 로드
        all_beams = loader.load_all_beams_for_ue(csv_files, target_ue)
        
        if len(all_beams) == 0:
            continue
        
        # Load ue_beams from P1P CSV
        try:
            ue_beams = loader.load_ue_beams_for_ue_p1p(csv_files, target_ue)
        except ValueError as e:
            print(f"    Warning: {e}, skipping UE {target_ue}")
            continue
        
        # Compute beam_counts for plotting
        from collections import Counter
        beam_counts = Counter(all_beams)
        unique_beams_display = sorted(beam_counts.keys())
        
        print(f"    K={len(all_beams)} (unique: {len(unique_beams_display)} beams {unique_beams_display}), ue_beams={ue_beams}")
        
        # Try DL Wen2011 optimization
        has_eigenbeam = False
        W_bs_plot = None
        V_bs_plot = None
        Lambda_bs_plot = None
        try:
            U_bs_full, U_ue_full, Omega_ul_norm, H_mean_ul_norm, pl_scalar = \
                ray_loader.load_weichselberger_for_ue(target_ue)
            U_ue_beam, U_bs_beam, Omega_beam_dl, H_mean_beam_dl, W_bs, W_ue = beam_analyzer.extract_beam_domain_params_dl(
                all_beams, ue_beams, U_bs_full, U_ue_full, Omega_ul_norm, H_mean_ul_norm
            )
            C_bs, P_bs, Lambda_bs, V_bs = beam_analyzer.optimize_bs_transmit_dl(
                U_ue_beam, U_bs_beam, Omega_beam_dl, H_mean_beam_dl, pl_scalar
            )
            W_bs_plot = W_bs.numpy()
            V_bs_plot = V_bs
            Lambda_bs_plot = Lambda_bs
            has_eigenbeam = True
            print(f"    DL Wen2011: C={C_bs:.2f} bps/Hz")
        except Exception as e:
            print(f"    Warning: Wen2011 failed: {e}")
        
        # 파일명 생성 (trial_id=None)
        codebook_file = P1QPathManager.generate_codebook_filename(target_ue)
        eigenbeam_file = P1QPathManager.generate_eigenbeam_filename(target_ue)
        
        # Plot codebook
        output_path_codebook = Path(config.output_dir) / codebook_file
        plotter.plot_codebook_beams(target_ue, np.array(unique_beams_display), beam_counts, str(output_path_codebook))
        print(f"    Codebook plot saved")
        n_ues_processed += 1
        
        # Plot eigenbeam
        if has_eigenbeam:
            output_path_eigenbeam = Path(config.output_dir) / eigenbeam_file
            plotter.plot_eigenbeams(target_ue, W_bs_plot, V_bs_plot, Lambda_bs_plot, str(output_path_eigenbeam))
            print(f"    Eigenbeam plot saved")
    
    print(f"\n[4/4] Complete! Generated {n_ues_processed} plots for {len(all_ues)} UEs")


def process_pert_mode(config, loader, ray_loader, beam_analyzer, plotter, csv_files, all_ues):
    """PERT 모드: trial별 반복"""
    print("\n[3/4] Processing PERT mode...")
    
    n_ues_processed = 0
    for ue_idx, target_ue in enumerate(all_ues, 1):
        print(f"\n  [{ue_idx}/{len(all_ues)}] UE {target_ue}")
        
        # 해당 UE의 모든 trial 로드
        try:
            all_trials = loader.load_all_trials_for_ue(csv_files, target_ue)
            print(f"    Found {len(all_trials)} trials: {all_trials}")
        except ValueError as e:
            print(f"    Warning: {e}, skipping UE {target_ue}")
            continue
        
        for trial_id in all_trials:
            print(f"    [Trial {trial_id}]")
            
            # P1P CSV에서 trial별 빔 로드
            try:
                ue_beams = loader.load_ue_beams_for_trial(csv_files, target_ue, trial_id)
                bs_beams = loader.load_bs_beams_for_trial(csv_files, target_ue, trial_id)
            except ValueError as e:
                print(f"      Warning: {e}, skipping trial {trial_id}")
                continue
            
            # Compute beam_counts for plotting
            from collections import Counter
            beam_counts = Counter(bs_beams)
            unique_beams_display = sorted(beam_counts.keys())
            
            print(f"      K={len(bs_beams)} (unique: {len(unique_beams_display)} beams {unique_beams_display}), ue_beams={ue_beams}")
            
            # Try DL Wen2011 optimization
            has_eigenbeam = False
            W_bs_plot = None
            V_bs_plot = None
            Lambda_bs_plot = None
            try:
                U_bs_full, U_ue_full, Omega_ul_norm, H_mean_ul_norm, pl_scalar = \
                    ray_loader.load_weichselberger_for_ue(target_ue)
                U_ue_beam, U_bs_beam, Omega_beam_dl, H_mean_beam_dl, W_bs, W_ue = beam_analyzer.extract_beam_domain_params_dl(
                    bs_beams, ue_beams, U_bs_full, U_ue_full, Omega_ul_norm, H_mean_ul_norm
                )
                C_bs, P_bs, Lambda_bs, V_bs = beam_analyzer.optimize_bs_transmit_dl(
                    U_ue_beam, U_bs_beam, Omega_beam_dl, H_mean_beam_dl, pl_scalar
                )
                W_bs_plot = W_bs.numpy()
                V_bs_plot = V_bs
                Lambda_bs_plot = Lambda_bs
                has_eigenbeam = True
                print(f"      DL Wen2011: C={C_bs:.2f} bps/Hz")
            except Exception as e:
                print(f"      Warning: Wen2011 failed: {e}")
            
            # 파일명 생성 (trial_id 포함)
            codebook_file = P1QPathManager.generate_codebook_filename(target_ue, trial_id)
            eigenbeam_file = P1QPathManager.generate_eigenbeam_filename(target_ue, trial_id)
            
            # Plot codebook
            output_path_codebook = Path(config.output_dir) / codebook_file
            plotter.plot_codebook_beams(target_ue, np.array(unique_beams_display), beam_counts, str(output_path_codebook))
            print(f"      Codebook plot saved")
            
            # Plot eigenbeam
            if has_eigenbeam:
                output_path_eigenbeam = Path(config.output_dir) / eigenbeam_file
                plotter.plot_eigenbeams(target_ue, W_bs_plot, V_bs_plot, Lambda_bs_plot, str(output_path_eigenbeam))
                print(f"      Eigenbeam plot saved")
        
        n_ues_processed += 1
    
    print(f"\n[4/4] Complete! Generated plots for {n_ues_processed} UEs")


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
        run_manual_mode(config, ray_loader, beam_analyzer, plotter)
    else:
        run_auto_mode(config, ray_loader, beam_analyzer, plotter)
    
    print("\n" + "="*70)
    print(f"All plots saved to: {config.output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()

