# ======================================================================
# P3_BF_Pair_2509v2.py
# P3A: Ray Data → 레이어 채널 샘플 생성
# 기능: P1 Ray 데이터를 레이어 기반 채널 샘플로 변환하여 저장
# 
# 개선사항:
# - 레이어 기반 빔포밍 채널 이득 계산
# - 2차원 크로네커 DFT 코드북 생성
# - TX-RX 빔 페어별 채널 이득 누적 계산
# - npz 형태로 빔 이득 + 방향 정보 저장
# ======================================================================

# ===== SECTION 1: 환경 설정 및 라이브러리 =====
import os
import subprocess
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc'
gpu_num = 0  # Use "" to use the CPU
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
os.environ["TENSORBOARD_BINARY"] = "tensorboard"
os.environ["TENSORBOARD_PLUGINS"] = "scalars,images,histograms,graphs,projector,profile"

# Import Sionna components
import sionna
from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, Camera,\
                      PathSolver, RadioMapSolver, subcarrier_frequencies
from pathlib import Path
import geopandas as gpd
import numpy as np
import numpy.typing

import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
for g in gpus:
    try:
        tf.config.experimental.set_memory_growth(g, True)   # 모두 True 로 통일
    except RuntimeError as e:
        print(e)
        
# Avoid warnings from TensorFlow
tf.get_logger().setLevel("ERROR")

# XLA JIT 컴파일 활성화 (연산 최적화) 
# 주의: 전역 설정보다는 @tf.function(jit_compile=True) 데코레이터 권장
tf.config.optimizer.set_jit(True)

# GPU 스레드 최적화 (L40S의 고성능 활용)
tf.config.threading.set_inter_op_parallelism_threads(0)  # 연산 간 병렬성: 모든 CPU 코어 사용
tf.config.threading.set_intra_op_parallelism_threads(0)  # 연산 내 병렬성: 모든 CPU 코어 사용

tf.random.set_seed(1)  # Set global random seed for reproducibility

# Additional imports for P2
import glob
import re

# 주피터/IPython 화면 클리어 지원
try:
    from IPython.display import clear_output
    JUPYTER_AVAILABLE = True
except ImportError:
    JUPYTER_AVAILABLE = False
from sionna.phy.channel.tr38901 import PanelArray, Rays, Topology
from sionna.phy import SPEED_OF_LIGHT
from sionna.phy.utils import expand_to_rank
from sionna.phy.block import Object
# Mathematical functions from tensorflow
PI = tf.constant(np.pi, tf.float32)
sin = tf.sin
cos = tf.cos
acos = tf.acos


# ===== SECTION 2: P3A_Config (레이어 기반 빔포밍 설정 확장) =====
class P3A_Config:
    """P1 Ray 데이터를 기반으로 레이어 채널 샘플 생성 설정 (Ray → 레이어 채널 샘플 저장)"""
    
    def __init__(self):
        # 파일 경로 및 명명 패턴 설정 (중앙 집중화)

        # 스크립트의 디렉토리를 기준으로 절대 경로 설정
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # P1B 필터링된 Ray 데이터 입력 경로 설정 (P1A → P1B → P3A 파이프라인)
        self.P1_INPUT_DIR = os.path.join(script_dir, "P1B_Valid_Results")
        self.P1_FILE_PATTERN = "Area{area}_{freq}GHz_Rays_Valid_RXs.npz"  # P1B 필터링된 데이터 입력
        self.P3A_OUTPUT_DIR = os.path.join(script_dir, "P3A_Layer_Ch_Results")
        self.P3A_FILE_PATTERN = "Area{area}_{freq}GHz_RX{rx}_layer_ch.npy"  # 레이어 채널 샘플 개별 저장
        
        # 후처리 청크 저장 설정 (레이어 채널 npy 파일들을 npz 청크로 묶기)
        self.ENABLE_POST_CHUNK = True        # 후처리 청크 기능 활성화
        self.CHUNK_SIZE = 100               # 청크당 RX 개수
        self.CHUNK_PATTERN = "Area{area}_{freq}GHz_Layer_Ch_RX{start}-{end}.npz"
        self.CLEANUP_INDIVIDUAL_FILES = True  # 청크 생성 후 개별 파일 삭제 여부
        
        # 콘솔 출력 제어 설정
        self.ENABLE_SCREEN_CLEAR = True      # 화면 클리어 기능 활성화/비활성화
        self.PROGRESS_CLEAR_INTERVAL = 10    # N개마다 화면 클리어 (테스트: 10, 실제: 100)
        
        # Area 필터링 설정 (None이면 모든 area 처리, 리스트로 특정 area 지정 가능)
        self.target_areas = [1]
        
        # P1 데이터 디렉토리 스캔하여 실험 설정 자동 감지
        self.detect_p1_data()
        
        # OFDM 시스템 파라미터
        self.OFDM_FFT = 8                              # 부반송파 수 (주파수 차원)
        self.OFDM_SCS = 60e3                           # 부반송파 간격 (Hz)
        self.OFDM_BW = self.OFDM_FFT * self.OFDM_SCS     # OFDM 총 대역폭 (Hz)
        self.num_tx = 1
        self.num_rx = 1

        # fc 리스트를 self.p1_data_combinations에서 추출하여 설정
        if self.p1_data_combinations:
            self.fcs = sorted(list(set([item[1] for item in self.p1_data_combinations])))
        else:
            self.fcs = []

        # 토폴로지 파라미터 (UE 이동성 관련 통계치)
        self.Topology_Statistics = {
            "moving_end": "rx",
            "velocities_mean": 1,
            "velocities_stddev": 0.1,
            # orientation은 고정값 사용으로 제거됨 (TX: config.TX_Orientation, RX: 0,0,0)
            "los_minval": 0,
            "los_maxval": 2
        }

        # 도플러 파라미터 (코히어런스 시간 기반)
        if self.fcs:
            v_max = self.Topology_Statistics["velocities_mean"] + 3*self.Topology_Statistics["velocities_stddev"]
            coherence_time = 3e8 / (v_max * self.fcs[0] * 1e9)
        else:
            coherence_time = 0.0 # 기본값 설정

        # 채널 생성 파라미터
        self.static_ch_realizations = 625              # 독립 정적 채널 개수
        self.doppler_time_realizations = 1              # 각 정적 채널의 도플러 샘플 수
        self.doppler_sym_ofdm = 1                       # 각 도플러 샘플당 OFDM 심볼 수
        self.doppler_time_max_sec = 0                   # 도플러 시간 샘플링 최대값 (초)

        # 기본 시스템 파라미터
        self.N_BS = 1                                    # 기지국 수
        self.batch_size = 1                              # 배치 크기

        # TX 방향 설정 (고정 방향)
        self.TX_Orientation = {
            "azimuth_deg": 246,
            "downtilt_deg": 3,  # 아래로 3도 기울임 (기지국 전형적 설정)
            "roll_deg": 0
        }

        self.TX_Array = {
            "location": [0,0,0],
            "rotation": [0,0],
            "num_rows_per_panel": 4, # 패널당 수직 안테나 수
            "num_cols_per_panel": 4, # 패널당 수평 안테나 수
            "num_rows": 8,
            "num_cols": 8,
            "polarization": "single",
            "polarization_type": "V",
            "antenna_pattern": "38.901",
            "panel_vertical_spacing": 2.5, # 패널 간 수직 간격 (λ 단위)
            "panel_horizontal_spacing": 2.5 # 패널 간 수평 간격 (λ 단위)
        } # 256 안테나 구성
        
        self.RX_Array = {
            "num_rows_per_panel": 2,    # 패널당 수직 안테나 수
            "num_cols_per_panel": 2,    # 패널당 수평 안테나 수
            "num_rows": 1,              # 수직 패널 수
            "num_cols": 1,              # 수평 패널 수
            "polarization": "single",   # 편파 타입
            "polarization_type": "V",   # 수직 편파
            "antenna_pattern": "omni"   # 전방향 패턴
        }
        
        # XPR (Cross-Polarization Ratio) 설정
        self.mean_xpr_list = {"UMi-LOS":9,"UMi-NLOS":8, "UMa-LOS":8,"UMa-NLOS":7}
        self.stddev_xpr_list = {"UMi-LOS":3,"UMi-NLOS":3, "UMa-LOS":4,"UMa-NLOS":4}
        self.mean_xpr = self.mean_xpr_list["UMa-NLOS"]
        self.stddev_xpr = self.stddev_xpr_list["UMa-NLOS"]

        # ===== 새로 추가: 레이어 기반 빔포밍 설정 =====
        # 안테나 엘리먼트 구조 정의
        self.TX_LAYERS = self.TX_Array["num_rows"] * self.TX_Array["num_cols"]
        self.RX_LAYERS = self.RX_Array["num_rows"] * self.RX_Array["num_cols"] 
        self.TX_AE_PER_LAYER = self.TX_Array["num_rows_per_panel"] * self.TX_Array["num_cols_per_panel"]
        self.RX_AE_PER_LAYER = self.RX_Array["num_rows_per_panel"] * self.RX_Array["num_cols_per_panel"]
        self.TX_NUM_ANTENNAS = self.TX_LAYERS * self.TX_AE_PER_LAYER  # N_t = 64 * 16 = 1024
        self.RX_NUM_ANTENNAS = self.RX_LAYERS * self.RX_AE_PER_LAYER  # N_r = 1 * 4 = 4
        
        # 2차원 오버샘플링 DFT 코드북 구성
        self.TX_1D_OVERSAMPLE = 2  # 1차원당 2배 → 2차원에서 4배 오버샘플링
        self.RX_1D_OVERSAMPLE = 2  # 1차원당 2배 → 2차원에서 4배 오버샘플링
        self.TX_BEAMS_PER_LAYER = (self.TX_Array["num_rows_per_panel"] * self.TX_1D_OVERSAMPLE) * \
                                  (self.TX_Array["num_cols_per_panel"] * self.TX_1D_OVERSAMPLE)
        self.RX_BEAMS_PER_LAYER = (self.RX_Array["num_rows_per_panel"] * self.RX_1D_OVERSAMPLE) * \
                                  (self.RX_Array["num_cols_per_panel"] * self.RX_1D_OVERSAMPLE)
        
        # ===== 레이어 차원 설정 (L_t, L_r) =====
        # TX 레이어 차원: 64개 코드북과 1:1 매핑
        self.L_t = self.TX_LAYERS  # 64개 레이어
        self.L_r = self.RX_LAYERS  # 1개 레이어
        
        # CSV 빔 랭킹 데이터 설정
        self.CSV_BEAM_DATA_DIR = os.path.join(script_dir, "BF_Analysis_Results")
        self.CSV_BEAM_PATTERN = "01_data_rx_rankings_Area{area}_{freq}GHz.csv"
        self.CSV_OPTIMAL_BEAM_COLUMN = "Rank1_RX_Beam"
        self.CSV_RX_INDEX_COLUMN = "RX_Index"
        
        # 설정 정보 출력
        self.print_config()
        
    def detect_p1_data(self):
        """P1 저장 데이터를 스캔하여 실험 설정 자동 감지 (unified npz format)"""
        
        # P1 출력 디렉토리에서 통합 npz 파일들 스캔
        scan_pattern = f"{self.P1_INPUT_DIR}/{self.P1_FILE_PATTERN.format(area='*', freq='*')}"
        files = glob.glob(scan_pattern)
        
        # (area, freq, rx) 조합을 저장할 집합
        combinations = set()
        
        # 파일명에서 area_index, frequency 추출하고 npz 파일에서 RX 정보 읽기
        for file_path in files:
            filename = os.path.basename(file_path)
            # Area{X}_{freq}GHz_Rays_Valid_RXs.npz 패턴 매칭 (P1B 필터링된 데이터)
            match = re.match(r'Area(\d+)_(.+)GHz_Rays_Valid_RXs\.npz', filename)
            if match:
                area_index = int(match.group(1))
                frequency = float(match.group(2))
                
                # target_areas 필터링 적용
                if self.target_areas is not None and area_index not in self.target_areas:
                    continue
                
                # npz 파일을 열어서 RX 인덱스들 확인
                try:
                    with np.load(file_path) as data:
                        if 'rx_indices' in data:
                            rx_indices = data['rx_indices']
                            for rx_index in rx_indices:
                                combinations.add((area_index, frequency, int(rx_index)))
                        else:
                            print(f"Warning: {filename} does not contain 'rx_indices' key")
                except Exception as e:
                    print(f"Warning: Failed to read {filename}: {e}")
        
        # 정렬하여 리스트로 저장
        self.p1_data_combinations = sorted(list(combinations))
        
        print(f"P1 데이터 자동 감지 (unified npz format):")
        print(f"  - 감지된 조합 수: {len(self.p1_data_combinations)}")
        
        # Area별로 그룹핑하여 출력
        area_groups = {}
        for combo in self.p1_data_combinations:
            area, freq, rx = combo
            key = f"Area{area}_{freq}GHz"
            if key not in area_groups:
                area_groups[key] = []
            area_groups[key].append(rx)
        
        for area_freq, rx_list in sorted(area_groups.items()):
            print(f"    - {area_freq}: RX{min(rx_list)}-RX{max(rx_list)} ({len(rx_list)} RXs)")
    
    def load_p1_ray_data(self, area_index, frequency, rx_index):
        """Load ray data for specific area, frequency, and RX from unified npz file
        
        Parameters
        ----------
        area_index : int
            Area index
        frequency : float  
            Frequency in GHz
        rx_index : int
            RX index
            
        Returns
        -------
        dict
            Dictionary containing ray parameters for the specified RX
        """
        # Load unified npz file
        npz_filename = self.P1_FILE_PATTERN.format(area=area_index, freq=frequency)
        npz_filepath = os.path.join(self.P1_INPUT_DIR, npz_filename)
        
        try:
            with np.load(npz_filepath) as data:
                # Find RX index position
                rx_indices = data['rx_indices']
                rx_position = np.where(rx_indices == rx_index)[0]
                
                if len(rx_position) == 0:
                    raise ValueError(f"RX{rx_index} not found in {npz_filename}")
                
                rx_pos = rx_position[0]
                
                # Extract ray data for specified RX
                ray_data = {
                    'phi_r': data['phi_r'][rx_pos],      # AoA
                    'phi_t': data['phi_t'][rx_pos],      # AoD  
                    'theta_r': data['theta_r'][rx_pos],  # ZoA
                    'theta_t': data['theta_t'][rx_pos],  # ZoD
                    'power': data['power'][rx_pos],      # Power
                    'tau': data['tau'][rx_pos],          # Delay
                }
                
                return ray_data
                
        except Exception as e:
            print(f"Error loading ray data from {npz_filepath}: {e}")
            return None
    
    def load_csv_optimal_rx_beam(self, area_index, frequency, rx_index):
        """CSV에서 특정 RX의 최적 빔 인덱스 로드
        
        Parameters
        ----------
        area_index : int
            Area index
        frequency : float  
            Frequency in GHz
        rx_index : int
            RX index
            
        Returns
        -------
        int or None
            Optimal RX beam index (Rank1_RX_Beam), None if not found
        """
        # CSV 파일 경로 생성
        csv_filename = self.CSV_BEAM_PATTERN.format(area=area_index, freq=frequency)
        csv_filepath = os.path.join(self.CSV_BEAM_DATA_DIR, csv_filename)
        
        try:
            import pandas as pd
            
            # CSV 파일 로드
            df = pd.read_csv(csv_filepath)
            
            # 해당 RX_Index 행 찾기
            rx_row = df[df[self.CSV_RX_INDEX_COLUMN] == rx_index]
            
            if len(rx_row) == 0:
                print(f"Warning: RX{rx_index} not found in {csv_filename}")
                return None
            
            # 최적 RX 빔 인덱스 추출
            optimal_rx_beam = rx_row[self.CSV_OPTIMAL_BEAM_COLUMN].iloc[0]
            
            return int(optimal_rx_beam)
            
        except Exception as e:
            print(f"Error loading CSV optimal beam from {csv_filepath}: {e}")
            return None
    
    def create_chunk_from_files(self, area_index, frequency, rx_indices):
        """개별 레이어 채널 npy 파일들을 청크로 묶기"""
        if not rx_indices:
            return None
        
        chunk_data = {}
        chunk_metadata = {
            'area_index': area_index,
            'frequency_ghz': frequency, 
            'rx_indices': np.array(rx_indices),
            'rx_count': len(rx_indices)
        }
        
        # 개별 파일들 로드 (진행률 표시)
        total_rx = len(rx_indices)
        for idx, rx_idx in enumerate(rx_indices, 1):
            individual_file = os.path.join(self.P3A_OUTPUT_DIR, 
                                         self.P3A_FILE_PATTERN.format(area=area_index, freq=frequency, rx=rx_idx))
            
            # 진행률 표시 (캐리지 리턴 방식)
            progress_msg = f"  파일 로딩 중... RX{rx_idx} ({idx}/{total_rx})"
            print(f"\r{progress_msg:<60}", end='', flush=True)
            
            if os.path.exists(individual_file):
                # 레이어 채널 데이터 로드: [tot_samples, L_r, L_t]
                layer_ch_data = np.load(individual_file)
                chunk_data[f'layer_ch_rx_{rx_idx}'] = {
                    'layer_channel': layer_ch_data,  # [tot_samples, L_r, L_t]
                    'data_shape': layer_ch_data.shape,
                    'rx_index': rx_idx,
                    'source_file': os.path.basename(individual_file)
                }
            else:
                print(f"\nWarning: {individual_file} not found")
        
        print()  # 진행률 완료 후 줄바꿈
        
        # 청크 파일 저장
        chunk_filename = self.CHUNK_PATTERN.format(
            area=area_index, freq=frequency, start=min(rx_indices), end=max(rx_indices)
        )
        chunk_filepath = os.path.join(self.P3A_OUTPUT_DIR, chunk_filename)
        
        # 메타데이터와 함께 저장
        save_data = {**chunk_metadata, **chunk_data}
        np.savez_compressed(chunk_filepath, **save_data)
        
        chunk_size_mb = os.path.getsize(chunk_filepath) / (1024**2)
        print(f"레이어 채널 청크 저장 완료: {chunk_filename} ({chunk_size_mb:.1f} MB, {len(rx_indices)}개 RX)")
        
        # 개별 파일 정리
        if self.CLEANUP_INDIVIDUAL_FILES:
            for rx_idx in rx_indices:
                individual_file = os.path.join(self.P3A_OUTPUT_DIR,
                                             self.P3A_FILE_PATTERN.format(area=area_index, freq=frequency, rx=rx_idx))
                if os.path.exists(individual_file):
                    os.remove(individual_file)
            print(f"개별 파일 {len(rx_indices)}개 정리 완료")
        
        return chunk_filepath

    def print_config(self):
        """P3A 설정 정보 출력"""
        print(f"P3A 레이어 채널 샘플 생성 설정:")
        print(f"  - Static Channel Realizations: {self.static_ch_realizations}")
        print(f"  - Doppler Time Realizations: {self.doppler_time_realizations}")
        print(f"  - Doppler Symbols per OFDM Symbol: {self.doppler_sym_ofdm}")
        print(f"  - OFDM FFT Size: {self.OFDM_FFT}")
        print(f"  - OFDM SCS: {self.OFDM_SCS}")
        print(f"  - OFDM BW: {self.OFDM_BW}")
        print(f"  - TX 1D Oversample: {self.TX_1D_OVERSAMPLE}")
        print(f"  - RX 1D Oversample: {self.RX_1D_OVERSAMPLE}")
        print(f"  - ===== 레이어 차원 설정 =====")
        print(f"  - TX 레이어 차원 (L_t): {self.L_t} (물리적 패널 구조)")
        print(f"  - RX 레이어 차원 (L_r): {self.L_r} (물리적 패널 구조, 확장 가능)")
        print(f"  - TX 코드북 개수: {self.TX_BEAMS_PER_LAYER} (설계상 L_t와 1:1 할당)")
        print(f"  - RX 코드북 개수: {self.RX_BEAMS_PER_LAYER} (CSV에서 최적 빔 선택)")
        print(f"  - TX Antennas (N_t): {self.TX_NUM_ANTENNAS}, RX Antennas (N_r): {self.RX_NUM_ANTENNAS}")
        print(f"  - 최종 출력 형태: [tot_samples, L_r={self.L_r}, L_t={self.L_t}]")
        print(f"    (tot_samples = static_ch_realizations * doppler_time_realizations * doppler_sym_ofdm * OFDM_FFT)")
        print(f"  - CSV 최적 빔 컬럼: {self.CSV_OPTIMAL_BEAM_COLUMN}")
        
        # Area 필터링 정보 출력
        if self.target_areas is None:
            print(f"  - Area 필터링: 모든 Area 처리")
        else:
            areas_str = ', '.join([f"Area{area}" for area in sorted(self.target_areas)])
            print(f"  - Area 필터링: {areas_str} 처리")


# ===== SECTION 3: 레이어 기반 DFT 코드북 생성 함수들 =====

def create_layer_based_dft_codebooks(config):
    """레이어 빔포밍에 사용할 2차원 크로네커 DFT 코드북 생성"""
    # TX 코드북: F_{4,2,4,2} = F_{4,2} ⊗ F_{4,2} → 64개 빔
    tx_layer_codebook = create_2d_kronecker_dft_codebook(
        config.TX_Array["num_rows_per_panel"], config.TX_Array["num_cols_per_panel"], 
        config.TX_1D_OVERSAMPLE)
    
    # RX 코드북: F_{2,2,2,2} = F_{2,2} ⊗ F_{2,2} → 16개 빔
    rx_layer_codebook = create_2d_kronecker_dft_codebook(
        config.RX_Array["num_rows_per_panel"], config.RX_Array["num_cols_per_panel"], 
        config.RX_1D_OVERSAMPLE)
    
    return tx_layer_codebook, rx_layer_codebook

def create_2d_kronecker_dft_codebook(N_rows, N_cols, K):
    """2차원 크로네커 곱 DFT 코드북 생성 (tmp.tex eq.39)
    
    입력:
        N_rows: 수직방향 안테나 개수
        N_cols: 수평방향 안테나 개수  
        K: 1차원 오버샘플링 인수
        
    출력:
        codebook_2d: (N_rows*N_cols, N_rows*K*N_cols*K) 크로네커 곱 DFT 코드북
        
    수식: F_{N_rows,K,N_cols,K} = F_{N_rows,K} ⊗ F_{N_cols,K}
    """
    # 1차원 DFT 코드북 생성
    F_rows = create_1d_dft_codebook(N_rows, K)  # (N_rows, N_rows*K) 수직방향
    F_cols = create_1d_dft_codebook(N_cols, K)  # (N_cols, N_cols*K) 수평방향
    
    # 크로네커 곱 (Sionna Column-Major 순서: 수평 먼저, 수직 나중)
    codebook_2d = tf.linalg.LinearOperatorKronecker([
        tf.linalg.LinearOperatorFullMatrix(F_cols),  # 수평 (Column-Major)
        tf.linalg.LinearOperatorFullMatrix(F_rows)   # 수직
    ]).to_dense()
    
    return codebook_2d

def create_1d_dft_codebook(N, K):
    """1차원 오버샘플링 DFT 코드북 생성 (tmp.tex eq.22)"""
    M = N * K
    i = tf.cast(tf.range(N), tf.float32)
    j = tf.cast(tf.range(M), tf.float32)
    i_grid, j_grid = tf.meshgrid(i, j, indexing='ij')
    phase = -2.0 * tf.constant(np.pi) * i_grid * j_grid / tf.cast(M, tf.float32)
    codebook = tf.complex(tf.cos(phase), tf.sin(phase)) / tf.cast(tf.sqrt(tf.cast(N, tf.float32)), tf.complex64)
    return tf.cast(codebook, tf.complex64)

def generate_h_ae_batch(topology, ray_pdap, config, OFDM_ChGen):
    """H_AE 배치 생성
    
    출력: H_AE_batch (RX_NUM_ANTENNAS, TX_NUM_ANTENNAS, doppler_time_realizations, doppler_sym_ofdm, OFDM_FFT)
    """
    # Static 채널 생성
    h_delay_bin_static, aoa_delay_bin, zoa_delay_bin = OFDM_ChGen._compute_ch_mimo_ofdm_38901_static(
        topology, ray_pdap, config.OFDM_FFT, config.OFDM_SCS)
    
    # 차원 정렬: transpose 적용 (기존 코드와 일치)
    h_delay_bin_static = tf.transpose(h_delay_bin_static, [0,3,5,6,1,2,7,4])  
    aoa_delay_bin = tf.transpose(aoa_delay_bin, [0,3,1,2,4])  
    zoa_delay_bin = tf.transpose(zoa_delay_bin, [0,3,1,2,4]) 
    
    doppler_collection = []
    
    for doppler_idx in range(config.doppler_time_realizations):
        symbol_collection = []
        
        for sym_idx in range(config.doppler_sym_ofdm):
            # 도플러 시간 샘플링
            doppler_times = tf.random.uniform(
                shape=[config.doppler_sym_ofdm],
                minval=0.0,
                maxval=config.doppler_time_max_sec,
                dtype=OFDM_ChGen.rdtype
            )
            
            # 도플러 효과 적용
            h_freq = OFDM_ChGen._apply_doppler_ch_mimo_ofdm_freq(
                topology, doppler_times, h_delay_bin_static, aoa_delay_bin, zoa_delay_bin)
            
            # 차원 변환: [B, N_BS, N_UE, doppler_sym_ofdm, OFDM_FFT, N_r, N_t] → [N_r, N_t, OFDM_FFT]
            h_freq_transposed = tf.transpose(h_freq, [0,3,4,5,6,1,2])  # [B, N_BS, N_UE, doppler_sym_ofdm, OFDM_FFT, N_r, N_t]
            h_freq_squeezed = tf.squeeze(h_freq_transposed, axis=[0, 1, 2, 3])  # [OFDM_FFT, N_r, N_t]
            h_freq_final = tf.transpose(h_freq_squeezed, [1, 2, 0])  # [N_r, N_t, OFDM_FFT]
            symbol_collection.append(h_freq_final)
        
        doppler_symbols = tf.stack(symbol_collection, axis=3)  # [N_r, N_t, OFDM_FFT, doppler_sym_ofdm]
        doppler_collection.append(doppler_symbols)
    
    # 최종 배치: [N_r, N_t, doppler_time_realizations, doppler_sym_ofdm, OFDM_FFT]
    H_AE_batch = tf.stack(doppler_collection, axis=2)
    
    return H_AE_batch


# ===== SECTION 4: 유틸리티 함수들 =====

@tf.function
def radian_to_degree(radian):
    return radian * (180.0 / PI)

@tf.function
def degree_to_radian(degree):
    return degree * (PI / 180.0)

def print_vram_usage():
    """현재 GPU VRAM 사용량을 출력"""
    mem_info = tf.config.experimental.get_memory_info('/GPU:0')
    used_memory = mem_info['current'] / (1024**3)
    peak_memory = mem_info['peak'] / (1024**3)
    print(f"현재 GPU VRAM 사용량: {used_memory:.2f} GB / {peak_memory:.2f} GB")

@tf.function(jit_compile=True)
def random_binary_mask_tf_complex64(n, k=3):
    """
    n개 중에서 랜덤하게 k개를 1로 설정하는 이진 마스크를 생성합니다.
    
    Args:
        n: 전체 요소 수
        k: 1로 설정할 요소 수 (기본값: 3)
    
    Returns:
        shape이 [n]인 이진 마스크 텐서
    """
    # 모두 0으로 초기화된 텐서 생성
    mask = tf.zeros([n], dtype=tf.complex64)
    
    # 랜덤하게 인덱스 선택
    indices = tf.random.shuffle(tf.range(n))[:k]
    
    # 선택된 인덱스 위치에 1 설정
    updates = tf.ones([k], dtype=tf.complex64)
    mask = tf.tensor_scatter_nd_update(mask, tf.expand_dims(indices, axis=-1), updates)
    
    return mask


# ===== SECTION 5: ChCoeGen 클래스 확장 =====
class ChCoeGen(Object):
    """P3A: Ray → 레이어 채널 샘플 생성 (독립 정적채널 + 도플러 시변채널)
    
    기존 기능:
    - _compute_ch_mimo_ofdm_38901_static(): 정적 MIMO OFDM 채널 생성 (N_r × N_t)
    - _apply_doppler_ch_mimo_ofdm_freq(): 도플러 효과 적용
    
    새로운 기능:
    - initialize_layer_channel_config(): 레이어 채널 설정 초기화
    - compute_layer_channel_samples(): 레이어 채널 샘플 생성 (L_r × L_t)
    
    구조:
    - static_ch_realizations: 독립 정적 채널 개수 (외부 루프)
    - 레이어 채널 생성: H_AE (N_r × N_t) → H_Layer (L_r × L_t)
    - 최종 출력: 레이어 채널 샘플 시계열 (npy 개별 저장)
    
    P1-P2-P3A 연계:
    - P1: 3D RT 시뮬레이션 → Ray 파라미터 (각도, 지연, 전력)
    - P2: Ray → OFDM 채널 (통계적 다양성 강화)
    - P3A: OFDM 채널 → 레이어 채널 샘플 → CSV 기반 최적 빔 적용
    """

    def __init__(self, carrier_frequency, subcarrier_spacing, tx_array, rx_array, subclustering, precision=None):
        super().__init__(precision=precision)
        
        # Wavelength (m)
        self.carrier_frequency = carrier_frequency
        self.subcarrier_spacing = subcarrier_spacing
        self._lambda_0 = tf.constant(SPEED_OF_LIGHT/carrier_frequency, self.rdtype)
        self._tx_array = tx_array
        self._rx_array = rx_array
        self._subclustering = subclustering

        # Sub-cluster information for intra cluster delay spread clusters
        # This is hardcoded from Table 7.5-5
        self._sub_cl_1_ind = tf.constant([0,1,2,3,4,5,6,7,18,19], tf.int32)
        self._sub_cl_2_ind = tf.constant([8,9,10,11,16,17], tf.int32)
        self._sub_cl_3_ind = tf.constant([12,13,14,15], tf.int32)
        self._sub_cl_delay_offsets = tf.constant([0, 1.28, 2.56], self.rdtype)

    # ===== 새로 추가: 레이어 채널 관련 메서드들 =====
    def initialize_layer_channel_config(self, config):
        """레이어 채널 설정 초기화"""
        self.config = config
        self.tx_layer_codebook, self.rx_layer_codebook = create_layer_based_dft_codebooks(config)
        
    def compute_layer_channel_samples(self, topology, ray_pdap, optimal_rx_beam_idx):
        """레이어 채널 샘플 생성
        
        Parameters
        ----------
        topology : Topology
            네트워크 토폴로지 정보
        ray_pdap : Rays
            Ray 데이터 (각도, 지연, 전력 등)
        optimal_rx_beam_idx : int
            CSV에서 가져온 최적 RX 빔 인덱스
            
        Returns
        -------
        tf.Tensor
            레이어 채널 샘플, shape: [tot_samples, L_r=1, L_t=64]
            (tot_samples = static_ch_realizations * doppler_time_realizations * doppler_sym_ofdm * OFDM_FFT)
            - L_r=1: RX 레이어 차원 
            - L_t=64: TX 레이어 차원 (TX_LAYERS와 설계상 일치)
        """
        # 최적 RX 빔 벡터 선택 (CSV에서 지정된 1개 빔)
        W_rx_optimal = self.rx_layer_codebook[:, optimal_rx_beam_idx:optimal_rx_beam_idx+1]  # [N_r, 1]
        
        # TX 코드북 전체 사용 (L_t개 레이어에 L_t개 코드북 1:1 할당)
        W_tx_all = self.tx_layer_codebook  # [N_t, TX_BEAMS_PER_LAYER] = [N_t, 64]
        # 참고: 현재 설계에서 L_t = TX_LAYERS = 64 = TX_BEAMS_PER_LAYER
        
        # 레이어 채널 샘플 생성
        all_static_ch_realizations = []
        
        for static_idx in range(self.config.static_ch_realizations):
            # 정적 채널 생성 (매번 새로운 초기 위상)
            h_delay_bin_static, aoa_delay_bin, zoa_delay_bin = self._compute_ch_mimo_ofdm_38901_static(
                topology, ray_pdap, self.config.OFDM_FFT, self.config.OFDM_SCS)
            
            # 차원 정렬: transpose 적용 (P2와 일치)
            h_delay_bin_static = tf.transpose(h_delay_bin_static, [0,3,5,6,1,2,7,4])
            aoa_delay_bin = tf.transpose(aoa_delay_bin, [0,3,1,2,4])
            zoa_delay_bin = tf.transpose(zoa_delay_bin, [0,3,1,2,4])
            
            # 도플러 시변 채널 생성
            current_doppler_collection = []
            
            for doppler_idx in range(self.config.doppler_time_realizations):
                # 도플러 시간 샘플링
                doppler_times = tf.random.uniform(
                    shape=[self.config.doppler_sym_ofdm],
                    minval=0.0,
                    maxval=self.config.doppler_time_max_sec,
                    dtype=self.rdtype
                )
                
                # 도플러 효과 적용 → OFDM 채널
                h_freq_doppler = self._apply_doppler_ch_mimo_ofdm_freq(
                    topology, doppler_times,
                    h_delay_bin_static, aoa_delay_bin, zoa_delay_bin
                )
                
                # 차원 변환: [B, N_r, N_t, N_BS, N_UE, doppler_sym_ofdm, OFDM_FFT] 
                # → [N_r, N_t, tot_samples] (시간-주파수 결합)
                h_freq_4d = tf.squeeze(h_freq_doppler, axis=[0, 3, 4])  # [N_r, N_t, doppler_sym_ofdm, OFDM_FFT]
                tot_samples = tf.shape(h_freq_4d)[2] * tf.shape(h_freq_4d)[3]  # doppler_sym_ofdm * OFDM_FFT
                h_freq_squeezed = tf.reshape(h_freq_4d, [self.config.RX_NUM_ANTENNAS, self.config.TX_NUM_ANTENNAS, tot_samples])
                # h_freq_squeezed: [N_r=4, N_t=1024, tot_samples]
                
                # 효율적인 블록 대각 빔포밍: W_rx^H @ H_AE @ W_tx_blkdiag
                # H_AE: [N_r=4, N_t=1024, tot_samples] (안테나 엘리먼트 채널)
                # W_rx_optimal: [N_r=4, 1] (CSV 최적 RX 빔 1개)
                # tx_layer_codebook: [TX_AE_PER_LAYER=16, TX_BEAMS_PER_LAYER=64] (패널당 코드북)
                
                # Step 1: H_AE를 패널별로 reshape
                # [N_r, N_t, tot_samples] → [N_r, TX_LAYERS, TX_AE_PER_LAYER, tot_samples]  
                h_ae_panels = tf.reshape(h_freq_squeezed, 
                    [self.config.RX_NUM_ANTENNAS, self.config.TX_LAYERS, self.config.TX_AE_PER_LAYER, tot_samples])
                
                # Step 2: RX 빔포밍 (모든 패널에 동일 적용)
                W_rx_H = tf.transpose(tf.math.conj(W_rx_optimal))  # [1, N_r=4]
                h_rx_beamformed = tf.einsum('Rr,rLas->RLas', W_rx_H, h_ae_panels)
                # 결과: [L_r=1, TX_LAYERS=64, TX_AE_PER_LAYER=16, tot_samples]
                
                # Step 3: 완전 벡터화된 블록 대각 TX 빔포밍 (사전할당, 메모리 효율적)
                # 블록 대각 1:1 매핑: 패널 k → 코드북 k
                # h_rx_beamformed: [L_r=1, TX_LAYERS=64, TX_AE_PER_LAYER=16, tot_samples]
                # tx_layer_codebook: [TX_AE_PER_LAYER=16, TX_BEAMS_PER_LAYER=64]
                
                # 대각선 코드북 추출: 패널 k는 코드북 k 사용 (벡터화)
                diagonal_indices = tf.range(self.config.TX_LAYERS)  # [0, 1, 2, ..., 63]
                w_diagonal = tf.gather(self.tx_layer_codebook, diagonal_indices, axis=1)
                # w_diagonal: [TX_AE_PER_LAYER=16, TX_LAYERS=64] (각 열이 해당 패널의 코드북)
                
                # 완전 벡터화된 블록 대각 빔포밍 (루프 없음, 사전할당)
                # [L_r=1, TX_LAYERS=64, TX_AE_PER_LAYER=16, tot_samples] · [TX_AE_PER_LAYER=16, TX_LAYERS=64]
                h_layer = tf.einsum('RLas,aL->RLs', h_rx_beamformed, w_diagonal)
                # 최종: [L_r=1, L_t=64, tot_samples] (메모리 효율적, 불필요한 복원 생략)
                
                current_doppler_collection.append(h_layer)
            
            # 도플러 샘플 결합 (시간 축으로 결합)
            single_static_realization = tf.concat(current_doppler_collection, axis=2)  # [L_r=1, L_t=64, total_tot_samples]
            all_static_ch_realizations.append(single_static_realization)
        
        # 모든 정적 채널 결합 (시간 축으로 결합)
        final_layer_ch_data = tf.concat(all_static_ch_realizations, axis=2)  # [L_r=1, L_t=64, all_time_samples]
        
        # 최종 차원 정렬: [L_r=1, L_t=64, all_time_samples] → [all_time_samples, L_r=1, L_t=64]
        final_layer_ch_data = tf.transpose(final_layer_ch_data, [2, 0, 1])  # [all_time_samples, L_r=1, L_t=64]
        
        return final_layer_ch_data
    
    @tf.function(jit_compile=True)
    def _unit_sphere_vector(self, theta, phi):
        """Generate vector on unit sphere (7.1-6)"""
        rho_hat = tf.stack([sin(theta)*cos(phi),
                            sin(theta)*sin(phi),
                            cos(theta)], axis=-1)
        return tf.expand_dims(rho_hat, axis=-1)
    
    @tf.function(jit_compile=True)
    def _unit_sphere_vector_Modified(self, theta, phi):
        """Generate vector on unit sphere (7.1-6) - Modified version"""
        rho_hat = tf.stack([sin(theta)*cos(phi),
                            sin(theta)*sin(phi),
                            cos(theta)], axis=-1)
        return tf.expand_dims(rho_hat, axis=1)
    
    @tf.function(jit_compile=True)
    def _unit_sphere_vector_Modified2(self, theta, phi):
        """Modified unit sphere vector for Doppler calculations"""
        rho_hat = tf.stack([sin(theta)*cos(phi),
                            sin(theta)*sin(phi),
                            cos(theta)], axis=-1)
        return tf.expand_dims(rho_hat, axis=-3)

    @tf.function(jit_compile=True)
    def _forward_rotation_matrix(self, orientations):
        """Forward composite rotation matrix (7.1-4)"""
        a, b, c = orientations[...,0], orientations[...,1], orientations[...,2]
        row_1 = tf.stack([cos(a)*cos(b),
            cos(a)*sin(b)*sin(c)-sin(a)*cos(c),
            cos(a)*sin(b)*cos(c)+sin(a)*sin(c)], axis=-1)
        row_2 = tf.stack([sin(a)*cos(b),
            sin(a)*sin(b)*sin(c)+cos(a)*cos(c),
            sin(a)*sin(b)*cos(c)-cos(a)*sin(c)], axis=-1)
        row_3 = tf.stack([-sin(b),
            cos(b)*sin(c),
            cos(b)*cos(c)], axis=-1)
        rot_mat = tf.stack([row_1, row_2, row_3], axis=-2)
        return rot_mat

    @tf.function(jit_compile=True)
    def _reverse_rotation_matrix(self, orientations):
        """Reverse composite rotation matrix (7.1-4)"""
        rot_mat = self._forward_rotation_matrix(orientations)
        rot_mat_inv = tf.linalg.matrix_transpose(rot_mat)
        return rot_mat_inv

    @tf.function(jit_compile=True)
    def _rot_pos(self, orientations, positions):
        """Rotate the positions according to the orientations"""
        rot_mat = self._forward_rotation_matrix(orientations)
        return tf.matmul(rot_mat, positions)

    @tf.function(jit_compile=True)
    def _gcs_to_lcs(self, orientations, theta, phi):
        """Compute the angles theta, phi in LCS rotated according to orientations"""
        rho_hat = self._unit_sphere_vector(theta, phi)
        rot_inv = self._reverse_rotation_matrix(orientations)
        rot_rho = tf.matmul(rot_inv, rho_hat)
        v1 = tf.constant([0,0,1], self.rdtype)
        v1 = tf.reshape(v1, [1]*(rot_rho.shape.rank-1)+[3])
        v2 = tf.constant([1+0j,1j,0], self.cdtype)
        v2 = tf.reshape(v2, [1]*(rot_rho.shape.rank-1)+[3])
        z = tf.matmul(v1, rot_rho)
        z = tf.clip_by_value(z, tf.constant(-1., self.rdtype),
                             tf.constant(1., self.rdtype))
        theta_prime = acos(z)
        phi_prime = tf.math.angle((tf.matmul(v2, tf.cast(rot_rho,
            self.cdtype))))
        theta_prime = tf.squeeze(theta_prime, axis=[phi.shape.rank,
            phi.shape.rank+1])
        phi_prime = tf.squeeze(phi_prime, axis=[phi.shape.rank,
            phi.shape.rank+1])
        return (theta_prime, phi_prime)

    @tf.function(jit_compile=True)
    def _compute_psi(self, orientations, theta, phi):
        """Compute displacement angle for the transformation of LCS-GCS field components"""
        a = orientations[...,0]
        b = orientations[...,1]
        c = orientations[...,2]
        real = sin(c)*cos(theta)*sin(phi-a)
        real += cos(c)*(cos(b)*sin(theta)-sin(b)*cos(theta)*cos(phi-a))
        imag = sin(c)*cos(phi-a) + sin(b)*cos(c)*sin(phi-a)
        psi = tf.math.angle(tf.complex(real, imag))
        return psi

    @tf.function(jit_compile=True)
    def _l2g_response(self, f_prime, orientations, theta, phi):
        """Transform field components from LCS to GCS (7.1-11)"""
        psi = self._compute_psi(orientations, theta, phi)
        row1 = tf.stack([cos(psi), -sin(psi)], axis=-1)
        row2 = tf.stack([sin(psi), cos(psi)], axis=-1)
        mat = tf.stack([row1, row2], axis=-2)
        f = tf.matmul(mat, tf.expand_dims(f_prime, -1))
        return f

    @tf.function(jit_compile=True)
    def _step_11_get_tx_antenna_positions(self, topology):
        """Compute positions of TX antenna elements in GCS"""
        tx_orientations = topology.tx_orientations
        tx_orientations = tf.expand_dims(tx_orientations, 2)
        tx_ant_pos_lcs = self._tx_array.ant_pos
        tx_ant_pos_lcs = tf.reshape(tx_ant_pos_lcs,
            [1,1]+tx_ant_pos_lcs.shape+[1])
        tx_ant_pos_gcs = self._rot_pos(tx_orientations, tx_ant_pos_lcs)
        tx_ant_pos_gcs = tf.reshape(tx_ant_pos_gcs,
            tf.shape(tx_ant_pos_gcs)[:-1])
        d_bar_tx = tx_ant_pos_gcs
        return d_bar_tx

    @tf.function(jit_compile=True)
    def _step_11_get_rx_antenna_positions(self, topology):
        """Compute positions of RX antenna elements in GCS"""
        rx_orientations = topology.rx_orientations
        rx_orientations = tf.expand_dims(rx_orientations, 2)
        rx_ant_pos_lcs = self._rx_array.ant_pos
        rx_ant_pos_lcs = tf.reshape(rx_ant_pos_lcs,
            [1,1]+rx_ant_pos_lcs.shape+[1])
        rx_ant_pos_gcs = self._rot_pos(rx_orientations, rx_ant_pos_lcs)
        rx_ant_pos_gcs = tf.reshape(rx_ant_pos_gcs,
            tf.shape(rx_ant_pos_gcs)[:-1])
        d_bar_rx = rx_ant_pos_gcs
        return d_bar_rx

    @tf.function(jit_compile=True)
    def _step_10(self, shape):
        """Generate random and uniformly distributed phases for all rays"""
        phi = tf.random.uniform(
                             tf.concat([shape, [4]], axis=0),
                             minval=-PI,
                             maxval=PI,
                             dtype=self.rdtype)
        return phi

    @tf.function(jit_compile=True)
    def _compute_ch_mimo_ofdm_38901_static(self, topology, rays, fft_size, scs):
        """Ray-to-OFDM 채널 계수 변환 (정적 채널)"""
        
        # 1. Delay bin index 계산
        delay_bin_index = tf.cast(tf.floor(rays.delays*scs*fft_size), dtype=tf.int32)
        delay_bin_index = delay_bin_index-tf.reduce_min(delay_bin_index, axis=-1, keepdims=True)

        # 2. 각도와 powers를 delay bin domain으로 확장
        def expand_to_delay_bin(values, delay_bin_index):
            one_hot = tf.one_hot(delay_bin_index, depth=fft_size, dtype=values.dtype)
            values_expanded = tf.expand_dims(values, axis=-1)
            delay_bin_values = one_hot * values_expanded
            shape = tf.shape(delay_bin_values)
            return tf.reshape(delay_bin_values, [shape[0], shape[1], shape[2], shape[3]*shape[4], shape[5]])

        aoa_delay_bin = expand_to_delay_bin(rays.aoa, delay_bin_index)
        aod_delay_bin = expand_to_delay_bin(rays.aod, delay_bin_index)
        zoa_delay_bin = expand_to_delay_bin(rays.zoa, delay_bin_index)
        zod_delay_bin = expand_to_delay_bin(rays.zod, delay_bin_index)
        xpr_delay_bin = expand_to_delay_bin(rays.xpr, delay_bin_index)
        powers_delay_bin = expand_to_delay_bin(rays.powers, delay_bin_index)

        # 3. Phase matrix 계산
        phi = self._step_10(tf.shape(aoa_delay_bin))
        raw_scaling = tf.sqrt(1/xpr_delay_bin)
        safe_scaling = tf.where(
            tf.math.is_inf(raw_scaling) | tf.math.is_nan(raw_scaling),
            tf.zeros_like(raw_scaling),
            raw_scaling
        )
        xpr_scaling = tf.complex(safe_scaling, tf.constant(0., self.rdtype))
        e0 = tf.exp(tf.complex(tf.constant(0., self.rdtype), phi[...,0]))
        e3 = tf.exp(tf.complex(tf.constant(0., self.rdtype), phi[...,3]))
        e1 = xpr_scaling*tf.exp(tf.complex(tf.constant(0., self.rdtype), phi[...,1]))
        e2 = xpr_scaling*tf.exp(tf.complex(tf.constant(0., self.rdtype), phi[...,2]))
        shape_phase = tf.concat([tf.shape(e0), [2,2]], axis=-1)
        h_phase = tf.reshape(tf.stack([e0, e1, e2, e3], axis=-1), shape_phase)

        # 4. Field matrix 계산
        tx_orientations = topology.tx_orientations
        rx_orientations = topology.rx_orientations
        
        s = tf.shape(tx_orientations)
        shape = tf.concat([s[:2], [1,1,1,s[-1]]], 0)
        tx_orientations = tf.reshape(tx_orientations, shape)
        
        zod_prime, aod_prime = self._gcs_to_lcs(tx_orientations, zod_delay_bin, aod_delay_bin)
        
        s = tf.shape(rx_orientations)
        shape = tf.concat([[s[0],1],[s[1],1,1,s[-1]]], 0)
        rx_orientations = tf.reshape(rx_orientations, shape)
        
        zoa_prime, aoa_prime = self._gcs_to_lcs(rx_orientations, zoa_delay_bin, aoa_delay_bin)
        
        f_tx_pol1_prime = tf.stack(self._tx_array.ant_pol1.field(zod_prime,aod_prime), axis=-1)
        f_rx_pol1_prime = tf.stack(self._rx_array.ant_pol1.field(zoa_prime,aoa_prime), axis=-1)
        f_tx_pol1 = self._l2g_response(f_tx_pol1_prime, tx_orientations, zod_delay_bin, aod_delay_bin)
        f_rx_pol1 = self._l2g_response(f_rx_pol1_prime, rx_orientations, zoa_delay_bin, aoa_delay_bin)
        
        if self._tx_array.polarization == 'dual':
            f_tx_pol2_prime = tf.stack(self._tx_array.ant_pol2.field(zod_prime, aod_prime), axis=-1)
            f_tx_pol2 = self._l2g_response(f_tx_pol2_prime, tx_orientations, zod_delay_bin, aod_delay_bin)
        
        if self._rx_array.polarization == 'dual':
            f_rx_pol2_prime = tf.stack(self._rx_array.ant_pol2.field(zoa_prime, aoa_prime), axis=-1)
            f_rx_pol2 = self._l2g_response(f_rx_pol2_prime, rx_orientations, zoa_delay_bin, aoa_delay_bin)
        
        f_tx_pol1_complex = tf.complex(f_tx_pol1, tf.constant(0., self.rdtype))
        pol1_tx = tf.matmul(h_phase, f_tx_pol1_complex)

        num_ant_tx = self._tx_array.num_ant
        if self._tx_array.polarization == 'single':
            f_tx_array = tf.tile(tf.expand_dims(pol1_tx, 0),
                tf.concat([[num_ant_tx], tf.ones([tf.rank(pol1_tx)], tf.int32)], axis=0))
        else:
            pol2_tx = tf.matmul(h_phase, tf.complex(f_tx_pol2, tf.constant(0., self.rdtype)))
            pol_tx = tf.stack([pol1_tx, pol2_tx], 0)
            ant_ind_pol2 = self._tx_array.ant_ind_pol2
            num_ant_pol2 = ant_ind_pol2.shape[0]
            gather_ind = tf.scatter_nd(tf.reshape(ant_ind_pol2, [-1,1]),
                tf.ones([num_ant_pol2], tf.int32), [num_ant_tx])
            f_tx_array = tf.gather(pol_tx, gather_ind, axis=0)
        
        num_ant_rx = self._rx_array.num_ant
        if self._rx_array.polarization == 'single':
            f_rx_array = tf.tile(tf.expand_dims(f_rx_pol1, 0),
                tf.concat([[num_ant_rx], tf.ones([tf.rank(f_rx_pol1)], tf.int32)], axis=0))
            f_rx_array = tf.complex(f_rx_array, tf.constant(0., self.rdtype))
        else:
            pol_rx = tf.stack([f_rx_pol1, f_rx_pol2], 0)
            ant_ind_pol2 = self._rx_array.ant_ind_pol2
            num_ant_pol2 = ant_ind_pol2.shape[0]
            gather_ind = tf.scatter_nd(tf.reshape(ant_ind_pol2, [-1,1]),
                tf.ones([num_ant_pol2], tf.int32), [num_ant_rx])
            f_rx_array = tf.complex(tf.gather(pol_rx, gather_ind, axis=0),
                tf.constant(0., self.rdtype))
        
        h_field = tf.reduce_sum(tf.expand_dims(f_rx_array, 1)*tf.expand_dims(f_tx_array, 0), [-2,-1])
        h_field = tf.transpose(h_field, perm=[2,3,4,5,6,0,1])

        # 5. Array offsets 계산
        lambda_0 = self._lambda_0
        r_hat_rx = self._unit_sphere_vector(zoa_delay_bin, aoa_delay_bin)
        r_hat_rx = tf.squeeze(r_hat_rx, axis=r_hat_rx.shape.rank-1)
        r_hat_tx = self._unit_sphere_vector(zod_delay_bin, aod_delay_bin)
        r_hat_tx = tf.squeeze(r_hat_tx, axis=r_hat_tx.shape.rank-1)
        
        d_bar_rx = self._step_11_get_rx_antenna_positions(topology)
        d_bar_tx = self._step_11_get_tx_antenna_positions(topology)
        r_hat_tx = tf.expand_dims(r_hat_tx, -2)
        r_hat_rx = tf.expand_dims(r_hat_rx, -2)
        
        s = tf.shape(d_bar_tx)
        shape = tf.concat([s[:2], [1,1,1], s[2:]], 0)
        d_bar_tx = tf.reshape(d_bar_tx, shape)
        
        s = tf.shape(d_bar_rx)
        shape = tf.concat([[s[0]], [1,s[1],1,1], s[2:]], 0)
        d_bar_rx = tf.reshape(d_bar_rx, shape)
        
        s = tf.shape(d_bar_rx)
        shape = tf.concat([[s[0]], [tf.shape(r_hat_rx)[1]], s[2:]], 0)
        d_bar_rx = tf.broadcast_to(d_bar_rx, shape)
        
        exp_rx = 2*PI/lambda_0*tf.reduce_sum(r_hat_rx*d_bar_rx, axis=-1, keepdims=True)
        exp_rx = tf.exp(tf.complex(tf.constant(0., self.rdtype), exp_rx))
        
        exp_tx = 2*PI/lambda_0*tf.reduce_sum(r_hat_tx*d_bar_tx, axis=-1)
        exp_tx = tf.exp(tf.complex(tf.constant(0., self.rdtype), exp_tx))
        exp_tx = tf.expand_dims(exp_tx, -2)
        
        h_array = exp_rx*exp_tx

        # 최종 채널 계수 계산
        h_field_array = tf.expand_dims(h_field*h_array, -1)
        power_scaling = tf.complex(tf.sqrt(powers_delay_bin), tf.constant(0., self.rdtype))
        power_scaling_reshaped = tf.reshape(power_scaling, tf.concat([tf.shape(power_scaling), [1,1,1]], 0))
        h_delay_bin_static = h_field_array*power_scaling_reshaped

        return h_delay_bin_static, aoa_delay_bin, zoa_delay_bin

    @tf.function(jit_compile=True)
    def _apply_doppler_ch_mimo_ofdm_freq(self, topology, doppler_times, h_delay_bin_static, aoa_delay_bin, zoa_delay_bin):
        """도플러 효과 적용"""
        
        # Doppler matrix 계산
        velocities = topology.velocities  # [B, N_UE, 3]
        B = tf.shape(velocities)[0]
        N_UE = tf.shape(velocities)[1]
        
        if topology.moving_end == 'rx':
            v_bar = tf.reshape(velocities, [B, 1, 1, N_UE, 1, 1, 3])  # DL: [B, 1, 1, N_UE, 1, 1, 3]
        elif topology.moving_end == 'tx':
            v_bar = tf.reshape(velocities, [B, 1, N_UE, 1, 1, 1, 3])  # UL: [B, 1, N_UE, 1, 1, 1, 3]

        r_hat_rx = self._unit_sphere_vector_Modified2(zoa_delay_bin, aoa_delay_bin) # [B, N_Rays, N_BS, N_UE, 1, N_FFT, 3]
        exponent = 2*PI/self._lambda_0*tf.reduce_sum(r_hat_rx*v_bar, -1) * tf.reshape(doppler_times, [1,1,1,1,len(doppler_times),1])
        h_doppler = tf.exp(tf.complex(tf.constant(0., self.rdtype), exponent))
        h_doppler = tf.expand_dims(tf.expand_dims(h_doppler, 2), 2) # [B, N_Rays, 1, 1, N_BS, N_UE, N_sym, N_FFT]

        # [B, N_Rays, N_r, N_t, N_BS, N_UE, N_sym, N_FFT]
        h_delay_bin_doppler = h_delay_bin_static*h_doppler # Element-wise multiplication

        # [B, N_Rays, N_r, N_t, N_BS, N_UE, N_sym, N_FFT] -> # [B, N_r, N_t, N_BS, N_UE, N_sym, N_FFT]
        h_delay_bin_doppler = tf.reduce_sum(h_delay_bin_doppler, axis=1) # rays 합산 in delay bin domain

        # FFT 내장함수 사용 (delay bin domain -> frequency domain)
        # [B, N_r, N_t, N_BS, N_UE, N_sym, N_FFT]
        h_freq_doppler = tf.signal.fft(h_delay_bin_doppler) 

        return  h_freq_doppler


# ===== SECTION 6: 메인 실행 (빔포밍 채널 이득 계산) =====
def main():
    """P3A: 레이어 채널 샘플 생성을 통한 CSV 기반 최적 빔 적용"""
    
    config = P3A_Config()
    
    # 전체 시작 시간 기록 (진행률 계산용)
    overall_start_time = time.time()
    
    # P1 데이터 기반 동적 루프
    total_rx_files = len(config.p1_data_combinations)
    rx_file_count = 0
    
    # 동적 자리수 맞춤을 위한 너비 계산
    total_rx_width = len(str(total_rx_files))
    
    # P3A: RX별 개별 처리 (CSV 기반)
    for area_index, fc, RX_index in config.p1_data_combinations:
        rx_file_count += 1
        carrier_frequency = fc * 10**9
        
        # 전체 시작 시간 기록
        tic_total = time.time()
        batch_size = config.batch_size
        N_UE = config.num_rx
        N_BS = config.N_BS
        
        # P1 데이터 로딩
        ray_data = config.load_p1_ray_data(area_index, fc, RX_index)
        if ray_data is None:
            print(f"Error: Failed to load ray data for Area{area_index}_{fc}GHz_RX{RX_index}")
            continue
        
        # CSV에서 최적 RX 빔 로딩
        optimal_rx_beam = config.load_csv_optimal_rx_beam(area_index, fc, RX_index)
        if optimal_rx_beam is None:
            print(f"Error: Failed to load optimal RX beam for Area{area_index}_{fc}GHz_RX{RX_index}")
            continue
        
        print(f"Processing Area{area_index}_{fc}GHz_RX{RX_index}: Optimal RX Beam = {optimal_rx_beam}")
        
        # Ray 데이터를 TensorFlow 텐서로 변환
        ray_aoa = tf.convert_to_tensor(ray_data['phi_r'])
        ray_aod = tf.convert_to_tensor(ray_data['phi_t'])
        ray_zoa = tf.convert_to_tensor(ray_data['theta_r'])
        ray_zod = tf.convert_to_tensor(ray_data['theta_t'])
        ray_power = tf.convert_to_tensor(ray_data['power'])
        ray_delay = tf.convert_to_tensor(ray_data['tau'])
        
        # 안테나 어레이 설정
        from sionna.phy.channel.tr38901 import PanelArray
        
        ArrayTX = PanelArray(num_rows_per_panel=config.TX_Array["num_rows_per_panel"],
                          num_cols_per_panel=config.TX_Array["num_cols_per_panel"],
                          num_rows=config.TX_Array["num_rows"],
                          num_cols=config.TX_Array["num_cols"],
                          polarization=config.TX_Array["polarization"],
                          polarization_type=config.TX_Array["polarization_type"],
                          antenna_pattern=config.TX_Array["antenna_pattern"],
                          carrier_frequency=carrier_frequency,
                          panel_vertical_spacing=config.TX_Array["panel_vertical_spacing"],
                          panel_horizontal_spacing=config.TX_Array["panel_horizontal_spacing"])

        ArrayRX = PanelArray(num_rows_per_panel=config.RX_Array["num_rows_per_panel"],
                                    num_cols_per_panel=config.RX_Array["num_cols_per_panel"],
                                    num_rows=config.RX_Array["num_rows"],
                                    num_cols=config.RX_Array["num_cols"],
                                    polarization=config.RX_Array["polarization"],
                                    polarization_type=config.RX_Array["polarization_type"],
                                    antenna_pattern=config.RX_Array["antenna_pattern"],
                                    carrier_frequency=carrier_frequency)

        # 채널 생성기 초기화
        LayerChGen = ChCoeGen(carrier_frequency, config.OFDM_SCS, ArrayTX, ArrayRX, False)
        LayerChGen.initialize_layer_channel_config(config)
        
        # Ray 및 Topology 설정
        ray_xpr = 10**(tf.random.normal(shape=[batch_size,1,N_UE,1,ray_aoa.shape[-1]], 
                                                mean=config.mean_xpr, stddev=config.stddev_xpr)/10)

        from sionna.phy.channel.tr38901 import Rays, Topology
        ray_pdap = Rays(delays=ray_delay, powers=ray_power, aoa=ray_aoa, aod=ray_aod,
                            zoa=ray_zoa, zod=ray_zod, xpr=ray_xpr)

        # 토폴로지 생성
        speed = tf.abs(tf.random.normal(shape=[batch_size, N_UE, 1],
                                                mean=config.Topology_Statistics["velocities_mean"],
                                                stddev=config.Topology_Statistics["velocities_stddev"],
                                                dtype=tf.float32))

        angle = tf.random.uniform(shape=[batch_size, N_UE, 1],
                                            minval=0.0,
                                            maxval=2 * PI,
                                            dtype=tf.float32)

        velocities = tf.concat([speed * tf.cos(angle), speed * tf.sin(angle), tf.zeros_like(speed)], axis=-1)
        moving_end = config.Topology_Statistics["moving_end"]
        
        los_aoa = tf.zeros([batch_size,N_BS,N_UE])
        los_aod = tf.zeros([batch_size,N_BS,N_UE])
        los_zoa = tf.zeros([batch_size,N_BS,N_UE])
        los_zod = tf.zeros([batch_size,N_BS,N_UE])
        los = tf.random.uniform(shape=[batch_size,N_BS,N_UE], 
                                        minval=config.Topology_Statistics["los_minval"], 
                                        maxval=config.Topology_Statistics["los_maxval"], 
                                        dtype=tf.int32) > 0
        distance_3d = tf.ones([1,N_BS,N_UE])
        # TX 방향: config.TX_Orientation 설정값 사용 (고정 방향, 단위: 라디안)
        tx_orientations = tf.constant([[[  
            np.deg2rad(config.TX_Orientation["azimuth_deg"]),
            np.deg2rad(config.TX_Orientation["downtilt_deg"]), 
            np.deg2rad(config.TX_Orientation["roll_deg"])
        ]]], dtype=tf.float32)
        # RX 방향: 고정 (0,0,0) - 단말 방향 고정 (단위: 라디안)
        rx_orientations = tf.zeros(shape=[batch_size, N_UE, 3], dtype=tf.float32)

        topology = Topology(velocities, moving_end, los_aoa, los_aod, los_zoa, los_zod, los, 
                           distance_3d, tx_orientations, rx_orientations)

        # 레이어 채널 샘플 생성
        print(f"  레이어 채널 샘플 생성 중... (최적 RX 빔: {optimal_rx_beam})")
        layer_ch_samples = LayerChGen.compute_layer_channel_samples(topology, ray_pdap, optimal_rx_beam)
        
        # 레이어 채널 데이터 저장
        os.makedirs(config.P3A_OUTPUT_DIR, exist_ok=True)
        output_filename = config.P3A_FILE_PATTERN.format(area=area_index, freq=fc, rx=RX_index)
        output_filepath = os.path.join(config.P3A_OUTPUT_DIR, output_filename)
        
        np.save(output_filepath, layer_ch_samples.numpy())
        
        toc_total = time.time()
        
        # 진행률 출력
        if config.ENABLE_SCREEN_CLEAR and (rx_file_count % config.PROGRESS_CLEAR_INTERVAL == 0 or rx_file_count == total_rx_files):
            # 시간 및 성능 통계 계산
            current_time = time.time()
            overall_elapsed = current_time - overall_start_time
            avg_time_per_rx = overall_elapsed / rx_file_count
            remaining_rx = total_rx_files - rx_file_count
            estimated_remaining = avg_time_per_rx * remaining_rx
            
            # 시간 포맷팅 함수
            def format_time(seconds):
                if seconds < 60:
                    return f"{seconds:.1f}초"
                elif seconds < 3600:
                    return f"{int(seconds//60)}분 {int(seconds%60)}초"
                else:
                    return f"{int(seconds//3600)}시간 {int((seconds%3600)//60)}분"
            
            # 진행 상황 요약 출력
            print(f"P3A 레이어 채널 생성 진행률:")
            print(f"✓ [{rx_file_count:>{total_rx_width}}/{total_rx_files}] Area{area_index}_{fc}GHz_RX{RX_index}: {toc_total - tic_total:.1f}s → {output_filename}")
            print(f"  - 레이어 채널 형태: {layer_ch_samples.shape}")
            print(f"  - 최적 RX 빔: {optimal_rx_beam}")
            print(f"현재 완료율: {rx_file_count/total_rx_files*100:.1f}% ({rx_file_count}/{total_rx_files})")
            print(f"전체 경과 시간: {format_time(overall_elapsed)}")
            print(f"평균 처리 속도: {avg_time_per_rx:.1f}초/RX")
            print(f"예상 남은 시간: {format_time(estimated_remaining)}")
            print("-" * 80)
    
    # 모든 RX 처리 완료 후 청크 생성
    if config.ENABLE_POST_CHUNK:
        print("\n청크 생성 시작...")
        area_freq_groups = {}
        for area_index, frequency, rx_index in config.p1_data_combinations:
            key = (area_index, frequency)
            if key not in area_freq_groups:
                area_freq_groups[key] = []
            area_freq_groups[key].append(rx_index)
        
        for (area_index, frequency), rx_list in area_freq_groups.items():
            sorted_rx_list = sorted(rx_list)
            total_chunks = (len(sorted_rx_list) + config.CHUNK_SIZE - 1) // config.CHUNK_SIZE
            
            print(f"Area{area_index}_{frequency}GHz: {total_chunks}개 청크 생성 중... (총 {len(sorted_rx_list)}개 RX)")
            
            # CHUNK_SIZE씩 묶어서 청크 생성
            for chunk_idx, i in enumerate(range(0, len(sorted_rx_list), config.CHUNK_SIZE), 1):
                chunk_rx_list = sorted_rx_list[i:i + config.CHUNK_SIZE]
                print(f"청크 {chunk_idx}/{total_chunks}: Area{area_index}_{frequency}GHz_RX{min(chunk_rx_list)}-{max(chunk_rx_list)} 생성 중...")
                config.create_chunk_from_files(area_index, frequency, chunk_rx_list)
        
        print(f"\n=== P3A 후처리 청크 생성 완료 ===")
        total_chunks_created = sum((len(rx_list) + config.CHUNK_SIZE - 1) // config.CHUNK_SIZE 
                                 for rx_list in area_freq_groups.values())
        total_rx_processed = sum(len(rx_list) for rx_list in area_freq_groups.values())
        print(f"총 {total_chunks_created}개 청크 생성 완료 (총 {total_rx_processed}개 RX 처리)")
    
    # P3A 완료 메시지
    print(f"\n=== P3A 레이어 채널 생성 완료 ===")
    print(f"총 {total_rx_files}개 RX 처리 완료")
    print(f"출력 디렉토리: {config.P3A_OUTPUT_DIR}")
    # 동적 크기 계산
    tot_samples = config.static_ch_realizations * config.doppler_time_realizations * config.doppler_sym_ofdm * config.OFDM_FFT
    complex_float64_bytes = 16  # 8바이트(real) + 8바이트(imag)
    file_size_bytes = tot_samples * config.L_r * config.L_t * complex_float64_bytes
    file_size_mb = file_size_bytes // (1024 * 1024)
    
    print(f"출력 형태: [tot_samples, L_r={config.L_r}, L_t={config.L_t}] (tot_samples = {tot_samples})")
    print(f"파일당 크기: {file_size_mb}MB (복소수 float64 기준)")


if __name__ == "__main__":
    main()
