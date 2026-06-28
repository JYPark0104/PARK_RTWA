# ======================================================================
# P1H_Rays_to_MeanCh_2510v1.py
# P1H: LoS Ray 기반 평균 채널 행렬 추정 도구
# 
# === 최상위 목적 ===
# P1A 결과 (LoS Ray)로부터 평균 채널 행렬 추정
# - P1B 유효 RX 기준: P1B Valid RXs에 있는 RX만 처리
# - LoS Ray만 사용 (zero-mean이 아닌 deterministic channel)
# - Initial phase [0,0,0] 고정 (랜덤 위상 없음)
# - 서브캐리어 평균을 통한 평균 채널 행렬 계산
# 
# === 핵심 설계 원리 ===
# 1. 데이터 흐름: P1B (유효 RX 목록) → P1A LoS Ray → OFDM 채널 → H_mean
# 2. Deterministic: Initial phase [0,0,0] 고정
# 3. RX별 독립 처리: P1B의 각 유효 RX에 대해 H_mean 저장
# 
# === 주요 구성 요소 ===
# - P1H_Config: P1B 유효 RX 스캔 + P1A LoS Ray 로딩 + P1H 저장 경로 설정
# - ChCoeGen: Ray → Antenna Element OFDM 채널 생성 (initial phase [0,0,0])
# - MeanChannel_Engine: OFDM 채널 생성 + 서브캐리어 평균
# - MeanChannel_Manager: 평균 채널 파일 저장/로딩 관리
# 
# === 입력/출력 ===
# 입력 1: P1B 결과 (Area{area}_{freq}GHz_Rays_Valid_RXs.npz)
#         - rx_indices: 유효한 RX 목록 (음수 지연 제거 등 검증 완료)
# 입력 2: P1A 결과 (Area{area}_{freq}GHz_Rays_ALL_RXs.npz)
#         - LoS Ray only (los_nlos_flag==1 필터링)
#         - tau, power, theta_r_deg, theta_t_deg, phi_r_deg, phi_t_deg
# 출력: Area{area}_{freq}GHz_RX{rx}_MeanCh.npz (P1B의 각 유효 RX에 대해)
#       - H_mean: [n_r, n_t] 평균 채널 행렬 (complex64)
#       - metadata: area, freq, rx, OFDM_FFT, OFDM_SCS
# 
# === 핵심 기술 ===
# - P1B 유효 RX 스캔: detect_p1b_valid_rxs()
# - LoS Ray compact 필터링: load_p1a_los_ray() (los_nlos_flag==1, non-LoS 제거)
#   * 기존: (1,1,1,1,400) with 380개 zero → NaN 발생
#   * 수정: (1,1,1,1,20) LoS only → 유효 Ray만 사용
# - Initial phase [0,0,0] 고정: ChCoeGen._step_10_LoS()
# - 서브캐리어 평균: H_mean = mean(H_ofdm, axis=FFT)
#
# === 주요 수정 이력 ===
# [251014] P1H 신규 작성: LoS Ray 기반 평균 채널 행렬 추정
# 1. P1B 유효 RX 스캔 (detect_p1b_valid_rxs)
# 2. P1A LoS Ray 로딩 (los_nlos_flag==1 필터링)
# 3. Initial phase [0,0,0] 고정
# 4. 서브캐리어 평균 계산
# 5. 의존성 기반 순서: Config → ChCoeGen → Engine → Manager → main
#
# [251015] 치명적 버그 수정: LoS Ray compact 필터링 (Line 375-383)
# - 문제: non-LoS를 0으로 채워서 380개 무효 Ray 포함 → NaN 발생
# - 해결: boolean indexing으로 LoS만 추출하여 compact array 생성
# - 결과: (1,1,1,1,400) → (1,1,1,1,20) 유효 Ray만 사용
#

# ======================================================================

# ===== SECTION 1: 환경 설정 =====
import os
import subprocess
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import glob
import re
from pathlib import Path
import geopandas as gpd
import numpy.typing

# TensorFlow 환경 설정 (GPU 메모리 할당 방식)
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc'
gpu_num = 0  # Use "" to use the CPU
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
os.environ["TENSORBOARD_BINARY"] = "tensorboard"
os.environ["TENSORBOARD_PLUGINS"] = "scalars,images,histograms,graphs,projector,profile"

# XLA 메모리 최적화 (TensorFlow import 전)
os.environ['XLA_FLAGS'] = '--xla_gpu_force_compilation_parallelism=1'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

# TensorFlow 코어 임포트
import tensorflow as tf

# GPU 메모리 설정 (L40S: 동적 메모리 증가)
gpus = tf.config.list_physical_devices('GPU')
for g in gpus:
    try:
        # 메모리 동적 증가 설정 (필요한 만큼 할당)
        tf.config.experimental.set_memory_growth(g, True)
        print(f"GPU 메모리 설정: 동적 증가 모드 (L40S ~46GB)")
    except RuntimeError as e:
        print(f"GPU 설정 경고: {e}")

# TensorFlow 최적화 설정
tf.get_logger().setLevel("ERROR")
tf.config.optimizer.set_jit(True)
tf.config.threading.set_inter_op_parallelism_threads(0) # 연산 간 병렬성: 모든 GPU 코어 사용
tf.config.threading.set_intra_op_parallelism_threads(0) # 연산 내 병렬성: 모든 GPU 코어 사용
tf.random.set_seed(1)

# Sionna 컴포넌트 임포트
import sionna
from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, Camera,\
                      PathSolver, RadioMapSolver, subcarrier_frequencies
from sionna.phy.channel.tr38901 import PanelArray, Rays, Topology
from sionna.phy import SPEED_OF_LIGHT
from sionna.phy.utils import expand_to_rank
from sionna.phy.block import Object

# 주피터/IPython 화면 클리어 지원
try:
    from IPython.display import clear_output
    JUPYTER_AVAILABLE = True
except ImportError:
    JUPYTER_AVAILABLE = False

# P1E 분리성 분석을 위한 추가 임포트
import pandas as pd
import json
from datetime import datetime

# 수학 상수 및 함수 정의
PI = tf.constant(np.pi, tf.float32)
sin = tf.sin
cos = tf.cos
acos = tf.acos

# ===== SECTION 2: P1H_Config (P1B 유효 RX + P1A LoS Ray 기반 설정) =====
class P1H_Config:
    """P1B 유효 RX 기준, P1A LoS Ray 기반 평균 채널 행렬 추정 설정
    
    데이터 흐름: P1B (유효 RX 목록) → P1A LoS Ray → 평균 채널
    """
    
    def __init__(self):
        # 파일 경로 및 명명 패턴 설정 (중앙 집중화)

        # 스크립트의 디렉토리를 기준으로 절대 경로 설정
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # P1A LoS Ray 입력 설정
        self.P1A_INPUT_DIR = os.path.join(script_dir, "P1A_RT_Results")
        self.P1A_FILE_PATTERN = "Area{area}_{freq}GHz_Rays_ALL_RXs.npz"
        
        # P1B Valid RXs 입력 설정 (유효 RX 목록 참조용)
        self.P1B_INPUT_DIR = os.path.join(script_dir, "P1B_Valid_Results")
        self.P1B_FILE_PATTERN = "Area{area}_{freq}GHz_Rays_Valid_RXs.npz"
        
        # P1H 평균 채널 저장 설정
        self.P1H_OUTPUT_DIR = os.path.join(script_dir, "P1H_MeanCh_Results")
        self.P1H_FILE_PATTERN = "Area{area}_{freq}GHz_RX{rx}_MeanCh.npz"
        
        # 콘솔 출력 제어 설정
        self.ENABLE_SCREEN_CLEAR = True      # 화면 클리어 기능 활성화/비활성화
        self.PROGRESS_CLEAR_INTERVAL = 10    # 화면 클리어 개수
        
        # 필터링 설정
        self.target_areas = [1]  # 처리할 area 목록 (None이면 전체)
        self.target_rxs = None   # 처리할 RX 목록 (None이면 전체)
        
        # P1B 유효 RX 디렉토리 스캔하여 실험 설정 자동 감지
        self.detect_p1b_valid_rxs()
        
        # fc 리스트를 self.p1b_valid_combinations에서 추출하여 설정
        if self.p1b_valid_combinations:
            self.fcs = sorted(list(set([item[1] for item in self.p1b_valid_combinations])))
        else:
            self.fcs = []

        # 원본 노트북의 Topology 설정 (UE 이동성 관련 통계치 설정)
        self.Topology_Statistics = {
            "moving_end": "rx",
            "velocities_mean": 1,
            "velocities_stddev": 0.1,
            "orientations_mean": 0,
            "orientations_stddev": 3.141592653589793/5,  # PI/5
            "los_minval": 0,
            "los_maxval": 2
        }

        # 도플러 파라미터 (코히어런스 시간 기반)
        if self.fcs:
            v_max = self.Topology_Statistics["velocities_mean"] + 3*self.Topology_Statistics["velocities_stddev"]
            coherence_time = 3e8 / (v_max * self.fcs[0] * 1e9)
        else:
            coherence_time = 0.0 # 기본값 설정

        # OFDM 채널 시스템 파라미터
        self.OFDM_FFT = 128                                 # OFDM 채널의 부반송파 수
        self.OFDM_SCS = 120e3                               # OFDM 채널의 부반송파 간격 (Hz)
        self.OFDM_BW = self.OFDM_FFT * self.OFDM_SCS        # OFDM 총 대역폭 (Hz)
        self.num_tx = 1
        self.num_rx = 1

        # 평균 채널 추정용 채널 생성 파라미터 (LoS deterministic)
        self.static_ch_realizations = 1     # LoS 채널 (initial phase [0,0,0] 고정)
        self.doppler_time_realizations = 1  # 도플러 없음
        self.doppler_sym_ofdm = 1           # OFDM 심볼 1개
        self.doppler_time_max_sec = 0       # 도플러 시간 샘플링 없음
        
        # L40S GPU 병렬 처리 최적화 설정
        self.batch_size_rx = 1                          # RX 배치 크기 (단일 RX 처리)
        self.use_mixed_precision = False                # 복소수 연산에서 혼합 정밀도 비활성화
        self.gpu_memory_limit_gb = 46                   # L40S 메모리 (동적 할당)
        
        # 메모리 계산 상수
        self.complex64_bytes = 8                        # complex64 = float32(4) * 2
        self.bytes_per_gb = 1024**3                     # 바이트를 GB로 변환
        self.bytes_per_mb = 1024**2                     # 바이트를 MB로 변환

        # 원본 노트북의 기본 시스템 파라미터들
        self.N_BS = 1                                    # 원본: N_BS = 1
        self.batch_size = 1                              # 원본: batch_size = 1

        # TX 방향 설정 (고정 방향, 실제 기지국 환경 반영)
        self.TX_Orientation = {
            "azimuth_deg": 246,
            "downtilt_deg": 3,  # 아래로 3도 기울임 (기지국 전형적 설정)
            "roll_deg": 0
        }

        # RX 방향 설정 (고정 방향)
        self.RX_Orientation = {
            "azimuth_deg": 0,
            "elevation_deg": 0,
            "roll_deg": 0
        }
        
        self.TX_Array = {
            "location": [0,0,0],
            "rotation": [0,0],
            "num_rows_per_panel": 4, 
            "num_cols_per_panel": 4,
            "num_rows": 8,
            "num_cols": 8,
            "polarization": "single",
            "polarization_type": "V",
            "antenna_pattern": "38.901",
            "panel_vertical_spacing": 2.5,
            "panel_horizontal_spacing": 2.5
        }
        
        self.RX_Array = {
            "num_rows_per_panel": 4,        
            "num_cols_per_panel": 4,       
            "num_rows": 1,             
            "num_cols": 1,             
            "polarization": "single",  
            "polarization_type": "V",  
            "antenna_pattern": "38.901"  
        }

        self.n_t = self.TX_Array["num_rows_per_panel"] * self.TX_Array["num_cols_per_panel"] * self.TX_Array["num_rows"] * self.TX_Array["num_cols"]  # TX 안테나 수
        self.n_r = self.RX_Array["num_rows_per_panel"] * self.RX_Array["num_cols_per_panel"] * self.RX_Array["num_rows"] * self.RX_Array["num_cols"]  # RX 안테나 수
        # P1D는 OFDM_FFT를 직접 사용 (P1E의 fft_size=1과 다름)
        
        # XPR 설정
        self.mean_xpr_list = {"UMi-LOS":9,"UMi-NLOS":8, "UMa-LOS":8,"UMa-NLOS":7}
        self.stddev_xpr_list = {"UMi-LOS":3,"UMi-NLOS":3, "UMa-LOS":4,"UMa-NLOS":4}
        self.mean_xpr = self.mean_xpr_list["UMa-NLOS"]
        self.stddev_xpr = self.stddev_xpr_list["UMa-NLOS"]

        # P1H 설정 정보 출력
        self.print_p1h_config()
        
    def detect_p1b_valid_rxs(self):
        """P1B Valid RXs 데이터를 스캔하여 유효한 RX 목록 자동 감지
        
        P1H는 P1B에서 필터링된 유효한 RX들에 대해서만 P1A의 LoS Ray로 평균 채널 계산
        """
        
        # P1B 출력 디렉토리에서 Valid RXs npz 파일들 스캔
        scan_pattern = f"{self.P1B_INPUT_DIR}/{self.P1B_FILE_PATTERN.format(area='*', freq='*')}"
        files = glob.glob(scan_pattern)
        
        # (area, freq, rx) 조합을 저장할 집합
        combinations = set()
        
        # 파일명에서 area_index, frequency 추출하고 내부 유효 RX 스캔
        for file_path in files:
            filename = os.path.basename(file_path)
            # Area{X}_{freq}GHz_Rays_Valid_RXs.npz 패턴 매칭
            match = re.match(r'Area(\d+)_(.+)GHz_Rays_Valid_RXs\.npz', filename)
            if match:
                area_index = int(match.group(1))
                frequency = float(match.group(2))
                
                # target_areas 필터링 적용
                if self.target_areas is not None and area_index not in self.target_areas:
                    continue
                
                # npz 파일을 열어서 유효 RX 인덱스 목록 확인
                try:
                    with np.load(file_path) as data:
                        if 'rx_indices' in data:
                            rx_indices = data['rx_indices']
                            for rx_index in rx_indices:
                                # target_rxs 필터링 적용
                                if self.target_rxs is not None and rx_index not in self.target_rxs:
                                    continue
                                combinations.add((area_index, frequency, int(rx_index)))
                except Exception as e:
                    print(f"Warning: P1B 파일 스캔 실패 ({filename}): {e}")
        
        # 정렬하여 리스트로 저장
        self.p1b_valid_combinations = sorted(list(combinations))
        
        print(f"P1B 유효 RX 데이터 자동 감지:")
        print(f"  - 감지된 유효 RX 조합 수: {len(self.p1b_valid_combinations)}")
        
        # Area별로 그룹핑하여 출력
        area_groups = {}
        for combo in self.p1b_valid_combinations:
            area, freq, rx = combo
            key = f"Area{area}_{freq}GHz"
            if key not in area_groups:
                area_groups[key] = []
            area_groups[key].append(rx)
        
        for area_freq, rx_list in sorted(area_groups.items()):
            print(f"    - {area_freq}: RX{min(rx_list)}-RX{max(rx_list)} ({len(rx_list)} Valid RXs)")
    
    def load_p1a_los_ray(self, area_index, frequency, rx_index):
        """Load LoS ray data for specific area, frequency, and RX from P1A ALL RXs npz file
        
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
            Dictionary containing LoS ray parameters for the specified RX
            Returns None if no LoS ray found
        """
        # Load P1A ALL RXs npz file
        npz_filename = self.P1A_FILE_PATTERN.format(area=area_index, freq=frequency)
        npz_filepath = os.path.join(self.P1A_INPUT_DIR, npz_filename)
        
        try:
            with np.load(npz_filepath) as data:
                # Find RX index position
                rx_indices = data['rx_indices']
                rx_position = np.where(rx_indices == rx_index)[0]
                
                if len(rx_position) == 0:
                    raise ValueError(f"RX{rx_index} not found in {npz_filename}")
                
                rx_pos = rx_position[0]
                
                # Extract all ray data for specified RX
                all_ray_data = {
                    'phi_r_deg': data['phi_r_deg'][rx_pos],      # AoA in degrees
                    'phi_t_deg': data['phi_t_deg'][rx_pos],      # AoD in degrees
                    'theta_r_deg': data['theta_r_deg'][rx_pos],  # ZoA in degrees
                    'theta_t_deg': data['theta_t_deg'][rx_pos],  # ZoD in degrees
                    'power': data['power'][rx_pos],              # Power
                    'tau': data['tau'][rx_pos],                  # Delay
                }
                
                # Filter LoS rays only (los_nlos_flag == 1)
                if 'los_nlos_flag' in data:
                    los_nlos_flag = data['los_nlos_flag'][rx_pos]
                    
                    # LoS ray mask (los_nlos_flag == 1 AND valid power)
                    valid_mask = ~np.isnan(all_ray_data['power']) & (all_ray_data['power'] > 0)
                    los_mask = valid_mask & (los_nlos_flag == 1)
                    
                    total_rays = int(np.sum(valid_mask))
                    los_rays = int(np.sum(los_mask))
                    
                    if los_rays == 0:
                        print(f"  Warning: RX{rx_index}에 LoS Ray 없음 (total: {total_rays})")
                        return None
                    
                    # Extract LoS rays only - compact array (non-LoS 제거)
                    # P1A 데이터: (1, 1, 1, 1, N_rays) 형태
                    # LoS Ray만 추출하여 compact array 생성 (0 채우기 제거)
                    los_ray_data = {}
                    for key in ['phi_r_deg', 'phi_t_deg', 'theta_r_deg', 'theta_t_deg', 'power', 'tau']:
                        # Extract only LoS rays (boolean indexing)
                        los_values = all_ray_data[key][los_mask]
                        # Reshape to maintain 5D structure: (1, 1, 1, 1, N_los_rays)
                        los_ray_data[key] = los_values.reshape(1, 1, 1, 1, -1)
                    
                    los_ray_data['los_rays'] = los_rays
                    los_ray_data['total_rays'] = total_rays
                    
                    return los_ray_data
                else:
                    raise ValueError(f"los_nlos_flag not found in {npz_filename}")
                
        except Exception as e:
            print(f"  Error loading LoS ray from {npz_filepath}: {e}")
            return None
    
    def print_p1h_config(self):
        """P1H 설정 정보 출력"""
        print(f"P1H 평균 채널 행렬 추정 설정:")
        print(f"  - LoS Ray 기반 (Initial phase [0,0,0] 고정)")
        print(f"  - RX 목록: P1B 유효 RX 기준")
        print(f"  - Ray 소스: P1A LoS Ray (los_nlos_flag==1)")
        print(f"  - OFDM FFT Size: {self.OFDM_FFT}")
        print(f"  - OFDM SCS: {self.OFDM_SCS/1e3:.0f} kHz")
        print(f"  - 안테나 구성: TX {self.n_t}개, RX {self.n_r}개")
        print(f"  - 출력: H_mean [{self.n_r}×{self.n_t}] (complex64)")
        print(f"  - P1B 입력 (RX 목록): {self.P1B_INPUT_DIR}")
        print(f"  - P1A 입력 (LoS Ray): {self.P1A_INPUT_DIR}")
        print(f"  - P1H 결과 저장: {self.P1H_OUTPUT_DIR}")
        
        # Area 필터링 정보 출력
        if self.target_areas is None:
            print(f"  - Area 필터링: 모든 Area 처리")
        else:
            areas_str = ', '.join([f"Area{area}" for area in sorted(self.target_areas)])
            print(f"  - Area 필터링: {areas_str} 처리")
        
        # RX 필터링 정보 출력
        if self.target_rxs is None:
            print(f"  - RX 필터링: 모든 RX 처리")
        else:
            rxs_str = ', '.join([f"RX{rx}" for rx in sorted(self.target_rxs)])
            print(f"  - RX 필터링: {rxs_str} 처리")


# ===== ChCoeGen 클래스 (LoS 채널 생성기) =====
class ChCoeGen(Object):
    """P1H: LoS Ray → Antenna Element OFDM Channel 변환 (P1A LoS Ray 기반)
    
    핵심 메서드:
    - _compute_ch_mimo_ofdm_38901_static(): 정적 MIMO OFDM 채널 생성
      * LoS 채널: 고정 초기 위상(_step_10_LoS) [0,0,0,0]
      * Ray → delay bin domain 변환
    - _apply_doppler_ch_mimo_ofdm_freq(): 도플러 효과 적용
      * P1H에서는 도플러 없음 (time=0)
    
    P1H 구조:
    - static_ch_realizations = 1: LoS deterministic (초기 위상 고정)
    - doppler_time_realizations = 1: 도플러 없음
    - 최종 심볼 수: 1 × 1 × 1 = 1
    
    P1 series 연계:
    - P1A: 3D RT 시뮬레이션 → LoS/NLoS Ray 파라미터
    - P1H: P1A LoS Ray only → 평균 채널 행렬 (서브캐리어 평균)
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
    
    def _unit_sphere_vector(self, theta, phi):
        r"""
        Generate vector on unit sphere (7.1-6)

        Input
        -------
        theta : Arbitrary shape, `tf.float`
            Zenith [radian]

        phi : Same shape as ``theta``, `tf.float`
            Azimuth [radian]

        Output
        --------
        rho_hat : ``phi.shape`` + [3, 1]
            Vector on unit sphere

        """
        rho_hat = tf.stack([sin(theta)*cos(phi),
                            sin(theta)*sin(phi),
                            cos(theta)], axis=-1)
        return tf.expand_dims(rho_hat, axis=-1)
    
    def _forward_rotation_matrix(self, orientations):
        r"""
        Forward composite rotation matrix (7.1-4)

        Input
        ------
            orientations : [...,3], `tf.float`
                Orientation to which to rotate [radian]

        Output
        -------
        R : [...,3,3], `tf.float`
            Rotation matrix
        """
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

    def _rot_pos(self, orientations, positions):
        r"""
        Rotate the ``positions`` according to the ``orientations``

        Input
        ------
        orientations : [...,3], `tf.float`
            Orientation to which to rotate [radian]

        positions : [...,3,1], `tf.float`
            Positions to rotate

        Output
        -------
        : [...,3,1], `tf.float`
            Rotated positions
        """
        rot_mat = self._forward_rotation_matrix(orientations)
        return tf.matmul(rot_mat, positions)

    def _reverse_rotation_matrix(self, orientations):
        r"""
        Reverse composite rotation matrix (7.1-4)

        Input
        ------
        orientations : [...,3], `tf.float`
            Orientations to rotate to  [radian]

        Output
        -------
        R_inv : [...,3,3], `tf.float`
            Inverse of the rotation matrix corresponding to ``orientations``
        """
        rot_mat = self._forward_rotation_matrix(orientations)
        rot_mat_inv = tf.linalg.matrix_transpose(rot_mat)
        return rot_mat_inv

    @tf.function(jit_compile=True)
    def _gcs_to_lcs(self, orientations, theta, phi):
        # pylint: disable=line-too-long
        r"""
        Compute the angles ``theta``, ``phi`` in LCS rotated according to
        ``orientations`` (7.1-7/8)

        Input
        ------
        orientations : [...,3] of rank K, `tf.float`
            Orientations to which to rotate to [radian]

        theta : Broadcastable to the first K-1 dimensions of ``orientations``, `tf.float`
            Zenith to rotate [radian]

        phi : Same dimension as ``theta``, `tf.float`
            Azimuth to rotate [radian]

        Output
        -------
        theta_prime : Same dimension as ``theta``, `tf.float`
            Rotated zenith

        phi_prime : Same dimensions as ``theta`` and ``phi``, `tf.float`
            Rotated azimuth
        """
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

    def _compute_psi(self, orientations, theta, phi):
        # pylint: disable=line-too-long
        r"""
        Compute displacement angle :math:`Psi` for the transformation of LCS-GCS
        field components in (7.1-15) of TR38.901 specification

        Input
        ------
        orientations : [...,3], tf.float
            Orientations to which to rotate to [radian]

        theta :  Broadcastable to the first K-1 dimensions of ``orientations``, tf.float
            Spherical position zenith [radian]

        phi : Same dimensions as ``theta``, tf.float
            Spherical position azimuth [radian]

        Output
        -------
            Psi : Same shape as ``theta`` and ``phi``, tf.float
                Displacement angle :math:`Psi`
        """
        a = orientations[...,0]
        b = orientations[...,1]
        c = orientations[...,2]
        real = sin(c)*cos(theta)*sin(phi-a)
        real += cos(c)*(cos(b)*sin(theta)-sin(b)*cos(theta)*cos(phi-a))
        imag = sin(c)*cos(phi-a) + sin(b)*cos(c)*sin(phi-a)
        psi = tf.math.angle(tf.complex(real, imag))
        return psi

    def _l2g_response(self, f_prime, orientations, theta, phi):
        # pylint: disable=line-too-long
        r"""
        Transform field components from LCS to GCS (7.1-11)

        Input
        ------
        f_prime : K-Dim Tensor of shape [...,2], tf.float
            Field components

        orientations : K-Dim Tensor of shape [...,3], tf.float
            Orientations of LCS-GCS [radian]

        theta : K-1-Dim Tensor with matching dimensions to ``f_prime`` and ``phi``, tf.float
            Spherical position zenith [radian]

        phi : Same dimensions as ``theta``, tf.float
            Spherical position azimuth [radian]

        Output
        ------
            F : K+1-Dim Tensor with shape [...,2,1], tf.float
                The first K dimensions are identical to those of ``f_prime``
        """
        psi = self._compute_psi(orientations, theta, phi)
        row1 = tf.stack([cos(psi), -sin(psi)], axis=-1)
        row2 = tf.stack([sin(psi), cos(psi)], axis=-1)
        mat = tf.stack([row1, row2], axis=-2)
        f = tf.matmul(mat, tf.expand_dims(f_prime, -1))
        return f

    @tf.function(jit_compile=True)
    def _step_11_get_tx_antenna_positions(self, topology):
        r"""Compute d_bar_tx in (7.5-22), i.e., the positions in GCS of elements
        forming the transmit panel

        Input
        -----
        topology : Topology
            Topology of the network

        Output
        -------
        d_bar_tx : [batch_size, num TXs, num TX antenna, 3]
            Positions of the antenna elements in the GCS
        """
        # Get BS orientations got broadcasting
        tx_orientations = topology.tx_orientations
        tx_orientations = tf.expand_dims(tx_orientations, 2)

        # Get antenna element positions in LCS and reshape for broadcasting
        tx_ant_pos_lcs = self._tx_array.ant_pos
        tx_ant_pos_lcs = tf.reshape(tx_ant_pos_lcs,
            [1,1]+tx_ant_pos_lcs.shape+[1])

        # Compute antenna element positions in GCS
        tx_ant_pos_gcs = self._rot_pos(tx_orientations, tx_ant_pos_lcs)
        tx_ant_pos_gcs = tf.reshape(tx_ant_pos_gcs,
            tf.shape(tx_ant_pos_gcs)[:-1])

        d_bar_tx = tx_ant_pos_gcs

        return d_bar_tx

    @tf.function(jit_compile=True)
    def _step_11_get_rx_antenna_positions(self, topology):
        r"""Compute d_bar_rx in (7.5-22), i.e., the positions in GCS of elements
        forming the receive antenna panel

        Input
        -----
        topology : Topology
            Topology of the network

        Output
        -------
        d_bar_rx : [batch_size, num RXs, num RX antenna, 3]
            Positions of the antenna elements in the GCS
        """
        # Get UT orientations got broadcasting
        rx_orientations = topology.rx_orientations
        rx_orientations = tf.expand_dims(rx_orientations, 2)

        # Get antenna element positions in LCS and reshape for broadcasting
        rx_ant_pos_lcs = self._rx_array.ant_pos
        rx_ant_pos_lcs = tf.reshape(rx_ant_pos_lcs,
            [1,1]+rx_ant_pos_lcs.shape+[1])

        # Compute antenna element positions in GCS
        rx_ant_pos_gcs = self._rot_pos(rx_orientations, rx_ant_pos_lcs)
        rx_ant_pos_gcs = tf.reshape(rx_ant_pos_gcs,
            tf.shape(rx_ant_pos_gcs)[:-1])

        d_bar_rx = rx_ant_pos_gcs

        return d_bar_rx

    def _step_10_LoS(self, shape):
        r"""
        Generate fixed phases for LoS channel (P1H deterministic)

        Input
        -----
        shape : Shape tensor
            Shape of the leading dimensions for the tensor of phases to generate

        Output
        ------
        phi : [shape] + [4], tf.float
            Fixed phases [0,0,0,0] for LoS deterministic channel
        """
        # P1H: LoS Ray with fixed initial phase [0,0,0,0]
        phi = tf.zeros(
                             tf.concat([shape, [4]], axis=0),
                             dtype=self.rdtype)
        return phi

    @tf.function(jit_compile=True)
    def _compute_ch_mimo_ofdm_38901_static(self, topology, rays, fft_size, scs):
        r"""
        Ray-to-OFDM 채널 계수 변환: 물리적 Ray 데이터를 OFDM delay bin domain으로 변환하여 
        도플러 시뮬레이션을 위한 정적 채널 기반 행렬을 생성한다.
        
        이 메서드는 3GPP TR38.901 표준에 따라 Ray 기반 물리 채널 모델을 OFDM 시스템의 
        주파수 도메인 채널로 변환하는 핵심 과정을 수행한다. 도플러 효과 적용 이전의 
        정적(시간 불변) 채널 계수를 계산하며, 후속 도플러 시뮬레이션의 기반이 된다.

        주요 처리 과정:
        1. **Delay Bin Mapping**: Ray 지연시간을 OFDM FFT bin 인덱스로 매핑
           - delay_bin_index = floor(τ * scs * fft_size) - min_delay_bin
           
        2. **Domain Transformation**: Ray 파라미터들을 delay bin domain으로 확장
           - Ray domain [B, N_BS, N_UE, N_Rays] → Delay bin domain [B, N_BS, N_UE, N_Rays×FFT, FFT]
           - One-hot encoding을 통해 각 Ray를 해당 delay bin에 할당
           
        3. **Polarization Matrix**: XPR 기반 편파 위상 행렬 계산
           - 2×2 편파 행렬: [[e^{jφ₀}, √(1/XPR)·e^{jφ₁}], [√(1/XPR)·e^{jφ₂}, e^{jφ₃}]]
           - 랜덤 위상 φᵢ와 Cross-Polarization Ratio (XPR) 적용
           
        4. **Antenna Field Response**: 송수신 안테나 필드 패턴 계산
           - GCS(Global) ↔ LCS(Local) 좌표계 변환
           - 안테나 패턴 방향성 및 편파 특성 적용
           - 단일/이중 편파 안테나 지원
           
        5. **Array Geometry**: 안테나 어레이 기하학적 위상 보정
           - 안테나 소자 간 위치 차이로 인한 위상 보정
           - exp(j·2π/λ₀·r̂·d̄) 형태의 어레이 응답
           
        6. **Power Scaling**: Ray 전력에 따른 채널 계수 스케일링
           - h_delay_bin_static = h_field_array × √(power_delay_bin)

        Parameters
        ----------
        topology : Topology
            네트워크 토폴로지 정보 (위치, 방향, 속도 등)
            - tx_orientations: [batch_size, N_BS, 3] 송신기 방향 [rad]  
            - rx_orientations: [batch_size, N_UE, 3] 수신기 방향 [rad]
            
        rays : Rays  
            물리적 전파 Ray 데이터 (3GPP TR38.901 기반)
            - delays: [batch_size, N_BS, N_UE, N_Rays] 지연시간 [s]
            - powers: [batch_size, N_BS, N_UE, N_Rays] 전력 [linear scale]
            - aoa: [batch_size, N_BS, N_UE, N_Rays] 도착 방위각 [rad]
            - aod: [batch_size, N_BS, N_UE, N_Rays] 출발 방위각 [rad] 
            - zoa: [batch_size, N_BS, N_UE, N_Rays] 도착 천정각 [rad]
            - zod: [batch_size, N_BS, N_UE, N_Rays] 출발 천정각 [rad]
            - xpr: [batch_size, N_BS, N_UE, N_Rays] Cross-Polarization Ratio [linear]
            
        fft_size : int
            OFDM FFT 크기 (부반송파 수)
            
        scs : float
            부반송파 간격 [Hz]

        Returns  
        -------
        h_delay_bin_static : tf.Tensor, dtype=tf.complex64
            도플러 시뮬레이션용 정적 채널 계수 행렬 (delay bin domain)
            Shape: [B, N_Rays, N_r, N_t, N_BS, N_UE, 1, N_FFT]
            - 마지막 차원 1은 시간 차원 (도플러 확장용)
            - delay bin domain에서 정의된 복소 채널 계수
            
        aoa_delay_bin : tf.Tensor, dtype=tf.float32
            Delay bin domain으로 확장된 도착 방위각 (AoA in radians)
            Shape: [B, N_Rays, N_BS, N_UE, N_FFT]
            
        zoa_delay_bin : tf.Tensor, dtype=tf.float32  
            Delay bin domain으로 확장된 도착 천정각 (ZoA in radians)
            Shape: [B, N_Rays, N_BS, N_UE, N_FFT]

        Notes
        -----
        - 이 메서드는 도플러 효과 적용 이전의 정적 채널 상태를 계산한다
        - 출력된 h_delay_bin_static은 _apply_doppler_ch_mimo_ofdm_freq()의 입력으로 사용된다
        - Ray domain에서 OFDM delay bin domain으로의 변환이 핵심이며, 이는 물리적 
          전파 모델과 OFDM 시스템을 연결하는 중요한 과정이다
        - 3GPP TR38.901 표준의 step 10-11에 해당하는 채널 계수 생성 과정을 구현한다
        
        Mathematical Background
        -----------------------
        최종 채널 계수는 다음과 같이 계산된다:
        
        H(f,t) = Σᵣ√Pᵣ · Fᵣˣ(Ωᵣ) · Fᵣᵗˣ(Ωᵣ) · exp(j·φᵣ) · exp(j·2π/λ₀·r̂ᵣ·d̄) · δ(f-fᵣ)
        
        여기서:
        - Pᵣ: Ray r의 전력  
        - Fᵣˣ/Fᵗˣ: 수신/송신 안테나 필드 응답
        - φᵣ: 편파 및 랜덤 위상
        - r̂ᵣ: Ray r의 방향 벡터
        - d̄: 안테나 소자 위치 벡터
        - δ(f-fᵣ): 주파수 도메인 델타 함수 (delay bin 매핑)
        """

        # 1. Delay bin index 계산
        def compute_delay_bin_index():
            delay_bin_index = tf.cast(tf.floor(rays.delays*scs*fft_size), dtype=tf.int32)
            return delay_bin_index-tf.reduce_min(delay_bin_index, axis=-1, keepdims=True)
        
        delay_bin_index = compute_delay_bin_index()

        # 2. 각도와 powers를 delay bin domain으로 확장
        def expand_to_delay_bin(values, delay_bin_index):
            one_hot = tf.one_hot(delay_bin_index, depth=fft_size, dtype=values.dtype)
            values_expanded = tf.expand_dims(values, axis=-1)
            delay_bin_values = one_hot * values_expanded
            shape = tf.shape(delay_bin_values)
            return tf.reshape(delay_bin_values, [shape[0], shape[1], shape[2], shape[3]*shape[4], shape[5]])

        def expand_angles_to_delay_bin():
            aoa_delay_bin = expand_to_delay_bin(rays.aoa, delay_bin_index)
            aod_delay_bin = expand_to_delay_bin(rays.aod, delay_bin_index)
            zoa_delay_bin = expand_to_delay_bin(rays.zoa, delay_bin_index)
            zod_delay_bin = expand_to_delay_bin(rays.zod, delay_bin_index)
            xpr_delay_bin = expand_to_delay_bin(rays.xpr, delay_bin_index)
            powers_delay_bin = expand_to_delay_bin(rays.powers, delay_bin_index)
            return aoa_delay_bin, aod_delay_bin, zoa_delay_bin, zod_delay_bin, xpr_delay_bin, powers_delay_bin

        aoa_delay_bin, aod_delay_bin, zoa_delay_bin, zod_delay_bin, xpr_delay_bin, powers_delay_bin = expand_angles_to_delay_bin()
        # 3. Phase matrix 계산
        def compute_phase_matrix():
            phi = self._step_10_LoS(tf.shape(aoa_delay_bin))

            # 일단 sqrt(1/xpr_delay_bin) 계산
            raw_scaling = tf.sqrt(1/xpr_delay_bin)
            
            # inf 값을 0으로 대체
            safe_scaling = tf.where(
                tf.math.is_inf(raw_scaling) | tf.math.is_nan(raw_scaling),  # inf나 nan인 경우
                tf.zeros_like(raw_scaling),  # 0으로 대체
                raw_scaling  # 그 외에는 원래 값 유지
            )

            xpr_scaling = tf.complex(safe_scaling, tf.constant(0., self.rdtype))
            e0 = tf.exp(tf.complex(tf.constant(0., self.rdtype), phi[...,0]))
            e3 = tf.exp(tf.complex(tf.constant(0., self.rdtype), phi[...,3]))
            e1 = xpr_scaling*tf.exp(tf.complex(tf.constant(0., self.rdtype), phi[...,1]))
            e2 = xpr_scaling*tf.exp(tf.complex(tf.constant(0., self.rdtype), phi[...,2]))
            shape_phase = tf.concat([tf.shape(e0), [2,2]], axis=-1)
            return tf.reshape(tf.stack([e0, e1, e2, e3], axis=-1), shape_phase)

        h_phase = compute_phase_matrix()
        # 4. Field matrix 계산
        def compute_field_matrix():
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
            return tf.transpose(h_field, perm=[2,3,4,5,6,0,1])

        h_field = compute_field_matrix()
        # 5. Array offsets 계산
        def compute_array_offsets():
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
            
            return exp_rx*exp_tx

        h_array = compute_array_offsets()
        # Doppler matrix 전까지 계산
        def compute_before_doppler():
            h_field_array =  tf.expand_dims(h_field*h_array, -1)
            power_scaling = tf.complex(tf.sqrt(powers_delay_bin), tf.constant(0., self.rdtype))
            power_scaling_reshaped = tf.reshape(power_scaling, tf.concat([tf.shape(power_scaling), [1,1,1]], 0))
            return h_field_array*power_scaling_reshaped
        
        h_delay_bin_static = compute_before_doppler()

        return h_delay_bin_static, aoa_delay_bin, zoa_delay_bin

    @tf.function(jit_compile=True)
    def _apply_doppler_ch_mimo_ofdm_freq(self, topology, doppler_times, h_delay_bin_static, aoa_delay_bin, zoa_delay_bin):
        
        # h_delay_bin_static shape
            # Original: [B, N_BS, N_UE, N_Rays, N_FFT, N_r, N_t, N_sym]
            # Reshaped: [B, N_Rays, N_r, N_t, N_BS, N_UE, N_sym, N_FFT]
        # zoa_delay_bin, aoa_delay_bin shape
            # Original: [B, N_BS, N_UE, N_Rays, N_FFT]
            # Reshaped: [B, N_Rays, N_BS, N_UE, N_FFT]

        # Doppler matrix 계산
        velocities = topology.velocities  # [B, N_UE, 3]
        B = tf.shape(velocities)[0]
        N_UE = tf.shape(velocities)[1]
        
        if topology.moving_end == 'rx':
            v_bar = tf.reshape(velocities, [B, 1, 1, N_UE, 1, 1, 3])  # DL: [B, 1, 1, N_UE, 1, 1, 3]
        elif topology.moving_end == 'tx':
            v_bar = tf.reshape(velocities, [B, 1, N_UE, 1, 1, 1, 3])  # UL: [B, 1, N_UE, 1, 1, 1, 3]

        r_hat_rx = tf.expand_dims(tf.squeeze(self._unit_sphere_vector(zoa_delay_bin, aoa_delay_bin), -1), 4) # [B, N_Rays, N_BS, N_UE, 1, N_FFT, 3], axis=-3 -> N_sym
        # [B, N_Rays, N_BS, N_UE, 1, N_FFT] * [1,1,1,1,N_sym,1] = [B, N_Rays, N_BS, N_UE, N_sym, N_FFT]
        exponent = 2*PI/self._lambda_0*tf.reduce_sum(r_hat_rx*v_bar, -1) * tf.reshape(doppler_times, [1,1,1,1,len(doppler_times),1])
        h_doppler = tf.exp(tf.complex(tf.constant(0., self.rdtype), exponent))
        h_doppler = tf.expand_dims(tf.expand_dims(h_doppler, 2), 2) # [B, N_Rays, 1, 1, N_BS, N_UE, N_sym, N_FFT] axis = 2,3 for N_r, N_t

        # [B, N_Rays, N_r, N_t, N_BS, N_UE, N_sym, N_FFT]
        h_delay_bin_doppler = h_delay_bin_static*h_doppler # Element-wise multiplication

        # [B, N_Rays, N_r, N_t, N_BS, N_UE, N_sym, N_FFT] -> # [B, N_r, N_t, N_BS, N_UE, N_sym, N_FFT]
        h_delay_bin_doppler = tf.reduce_sum(h_delay_bin_doppler, axis=1) # rays 합산 in delay bin domain

        # FFT 내장함수 사용 (delay bin domain -> frequency domain)
        # [B, N_r, N_t, N_BS, N_UE, N_sym, N_FFT]
        h_freq_doppler = tf.signal.fft(h_delay_bin_doppler) 

        return  h_freq_doppler

# ===== SECTION 3: 평균 채널 추정 엔진 =====
class MeanChannel_Engine:
    """평균 채널 행렬 추정: LoS Ray → OFDM 채널 → 서브캐리어 평균"""
    
    def __init__(self, config, OFDM_ChGen, topology, ray_los):
        """
        Parameters:
            config: P1H_Config 인스턴스
            OFDM_ChGen: ChCoeGen 인스턴스 (initial phase [0,0,0])
            topology: Topology 인스턴스
            ray_los: Rays 인스턴스 (LoS Ray only)
        """
        self.config = config
        self.OFDM_ChGen = OFDM_ChGen
        self.topology = topology
        self.ray_los = ray_los
        
    def generate_single_ofdm_channel(self):
        """단일 LoS OFDM 채널 생성 (deterministic)
        
        Returns:
            H_ofdm: [1, n_r, n_t, 1, 1, 1, n_fft] OFDM 채널
        """
        # 1. Static Channel 생성 (initial phase [0,0,0])
        h_delay_bin_static, aoa_delay_bin, zoa_delay_bin = \
            self.OFDM_ChGen._compute_ch_mimo_ofdm_38901_static(
                self.topology, self.ray_los, 
                self.config.OFDM_FFT, self.config.OFDM_SCS)
        
        # 2. 필수 transpose 처리
        h_delay_bin_static = tf.transpose(h_delay_bin_static, [0,3,5,6,1,2,7,4])
        aoa_delay_bin = tf.transpose(aoa_delay_bin, [0,3,1,2,4])
        zoa_delay_bin = tf.transpose(zoa_delay_bin, [0,3,1,2,4])
        
        # 3. Doppler 처리 (도플러 없음: time=0)
        doppler_times = tf.zeros(
            shape=[self.config.doppler_sym_ofdm],
            dtype=self.OFDM_ChGen.rdtype)
        
        ofdm_ch_doppler = self.OFDM_ChGen._apply_doppler_ch_mimo_ofdm_freq(
            self.topology, doppler_times, 
            h_delay_bin_static, aoa_delay_bin, zoa_delay_bin)
        
        # 4. 차원 변환: [B, N_r, N_t, N_BS, N_UE, N_sym, N_FFT] 
        #           → [B, N_BS, N_UE, N_sym, N_FFT, N_r, N_t]
        channel_sample = tf.transpose(ofdm_ch_doppler, [0,3,4,5,6,1,2])
        
        # 5. Squeeze: [1, 1, 1, 1, N_FFT, N_r, N_t] → [1, N_r, N_t, 1, 1, 1, N_FFT]
        channel_sample = tf.transpose(channel_sample, [0,5,6,1,2,3,4])
        
        return channel_sample
    
    def compute_mean_channel(self, area_index, fc, RX_index):
        """평균 채널 행렬 계산 (서브캐리어 평균)
        
        Args:
            area_index, fc, RX_index: RX 식별 정보
        
        Returns:
            dict: 평균 채널 데이터
                - area_idx, freq_ghz, rx_idx
                - H_mean: [n_r, n_t] 평균 채널 행렬 (complex64)
                - metadata: OFDM_FFT, OFDM_SCS, n_r, n_t
        """
        # 1. 단일 OFDM 채널 생성
        H_ofdm = self.generate_single_ofdm_channel()  # [1, n_r, n_t, 1, 1, 1, n_fft]
        
        # 2. 서브캐리어 평균
        H_mean = tf.reduce_mean(H_ofdm, axis=-1)  # [1, n_r, n_t, 1, 1, 1]
        
        # 3. Squeeze to [n_r, n_t]
        H_mean = tf.squeeze(H_mean)  # [n_r, n_t]
        
        # 4. NumPy 변환
        H_mean_np = H_mean.numpy()  # complex64
        
        # 5. Frobenius norm 제곱 계산 (LoS 전력)
        # 주의: 안테나 정규화는 적용하지 않음 (다운링크/업링크 방향에 따라 달라짐)
        H_mean_fro_sq = float(np.sum(np.abs(H_mean_np)**2))
        
        print(f"  완료: H_mean [{self.config.n_r}×{self.config.n_t}], ||H_mean||_F^2 = {H_mean_fro_sq:.4e}")
        
        result = {
            'area_idx': area_index,
            'freq_ghz': fc,
            'rx_idx': RX_index,
            'H_mean': H_mean_np,
            'H_mean_fro_sq': H_mean_fro_sq,
            'n_r': self.config.n_r,
            'n_t': self.config.n_t,
            'OFDM_FFT': self.config.OFDM_FFT,
            'OFDM_SCS': self.config.OFDM_SCS
        }
        
        return result

# ===== SECTION 4: 평균 채널 파일 관리 =====
class MeanChannel_Manager:
    """평균 채널 행렬 파일 저장 및 로딩 관리"""
    
    def __init__(self, config):
        self.config = config
        os.makedirs(self.config.P1H_OUTPUT_DIR, exist_ok=True)
    
    def mean_channel_file_exists(self, area_idx: int, freq_ghz: float, rx_idx: int) -> bool:
        """RX별 평균 채널 파일 존재 여부 확인 (재개용)
        
        Args:
            area_idx: Area 인덱스
            freq_ghz: 주파수 (GHz)
            rx_idx: RX 인덱스
        
        Returns:
            bool: 파일 존재 여부
        """
        filename = self.config.P1H_FILE_PATTERN.format(
            area=area_idx, freq=freq_ghz, rx=rx_idx
        )
        filepath = os.path.join(self.config.P1H_OUTPUT_DIR, filename)
        return os.path.exists(filepath)
    
    def save_rx_mean_channel(self, mean_ch_data: dict) -> str:
        """RX별 평균 채널 npz 파일 저장
        
        Args:
            mean_ch_data: compute_mean_channel() 반환값
                - area_idx, freq_ghz, rx_idx
                - H_mean: [n_r, n_t] 평균 채널 행렬
                - n_r, n_t, OFDM_FFT, OFDM_SCS
        
        Returns:
            str: 저장된 파일 경로
        """
        area_idx = mean_ch_data['area_idx']
        freq_ghz = mean_ch_data['freq_ghz']
        rx_idx = mean_ch_data['rx_idx']
        
        filename = self.config.P1H_FILE_PATTERN.format(
            area=area_idx, freq=freq_ghz, rx=rx_idx
        )
        filepath = os.path.join(self.config.P1H_OUTPUT_DIR, filename)
        
        # 메타데이터 생성
        metadata = {
            'area_idx': area_idx,
            'freq_ghz': freq_ghz,
            'rx_idx': rx_idx,
            'n_r': mean_ch_data['n_r'],
            'n_t': mean_ch_data['n_t'],
            'OFDM_FFT': mean_ch_data['OFDM_FFT'],
            'OFDM_SCS': mean_ch_data['OFDM_SCS'],
            'H_mean_fro_sq': mean_ch_data['H_mean_fro_sq']
        }
        
        # npz 파일 저장
        np.savez_compressed(
            filepath,
            H_mean=mean_ch_data['H_mean'],
            H_mean_fro_sq=mean_ch_data['H_mean_fro_sq'],
            metadata=metadata
        )
        
        return filepath
    
    def load_rx_mean_channel(self, area_idx: int, freq_ghz: float, rx_idx: int) -> dict:
        """저장된 평균 채널 파일 로딩
        
        Returns:
            dict: H_mean, H_mean_fro_sq, metadata
        """
        filename = self.config.P1H_FILE_PATTERN.format(
            area=area_idx, freq=freq_ghz, rx=rx_idx
        )
        filepath = os.path.join(self.config.P1H_OUTPUT_DIR, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Mean channel file not found: {filepath}")
        
        data = np.load(filepath, allow_pickle=True)
        
        # 예전 파일 호환성 위해 get 사용
        H_mean_fro_sq = data.get('H_mean_fro_sq', None)
        if H_mean_fro_sq is None:
            # 예전 파일: 직접 계산
            H_mean_fro_sq = float(np.sum(np.abs(data['H_mean'])**2))
        
        return {
            'H_mean': data['H_mean'],
            'H_mean_fro_sq': H_mean_fro_sq,
            'metadata': data['metadata'].item()
        }


# ===== SECTION 5: 메인 실행 P1H =====
def main():
    """P1H: P1A LoS Ray 기반 평균 채널 행렬 추정"""
    
    # ===== 메모리 정리 및 GPU 설정 =====
    print("GPU 메모리 정리 및 설정 중...", end=' ', flush=True)
    
    # 1. Python 가비지 컬렉션 강제 실행
    import gc
    gc.collect()
    
    # 2. TensorFlow GPU 메모리 정리
    try:
        # 기존 세션/그래프 정리
        tf.keras.backend.clear_session()
        
        # GPU 메모리 확인
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            # GPU 정보 출력
            gpu_details = tf.config.experimental.get_device_details(gpus[0])
            print(f"GPU: {gpu_details.get('device_name', 'Unknown')}", end=' ')
            print(f"[동적 메모리 할당 모드]", end=' ')
            
        # 3. TensorFlow 옵티마이저 추가 설정
        tf.config.optimizer.set_experimental_options({
            'memory_optimization': True,
            'layout_optimizer': True
        })
        
        print("완료")
    except Exception as e:
        print(f"경고: GPU 설정 실패 ({e})")
    
    config = P1H_Config()
    
    # 전체 시작 시간 기록 (진행률 계산용)
    overall_start_time = time.time()
    
    # 평균 채널 파일 관리자 초기화
    mean_ch_manager = MeanChannel_Manager(config)
    print(f"\n=== P1H 평균 채널 행렬 추정 시작 ===")
    print(f"입력 디렉토리: {config.P1A_INPUT_DIR}")
    print(f"출력 디렉토리: {config.P1H_OUTPUT_DIR}")
    print(f"파일 패턴: {config.P1H_FILE_PATTERN}")
    print()
    
    # P1B 유효 RX 기반 동적 루프 (P1A에서 LoS Ray 로딩)
    total_rx_files = len(config.p1b_valid_combinations)
    rx_file_count = 0
    
    # 동적 자리수 맞춤을 위한 너비 계산
    total_rx_width = len(str(total_rx_files))
    
    for area_index, fc, RX_index in config.p1b_valid_combinations:
        rx_file_count += 1
        
        # 재개 로직: 이미 저장된 RX는 건너뛰기
        if mean_ch_manager.mean_channel_file_exists(area_index, fc, RX_index):
            print(f"\n[{rx_file_count}/{total_rx_files}] Area{area_index}_{fc}GHz_RX{RX_index}: 이미 완료됨 (스킵)")
            continue
        
        carrier_frequency = fc*10**9
        
        # 시작 시간 기록
        tic_total = time.time()
        batch_size = config.batch_size
        N_UE = config.num_rx
        N_BS = config.N_BS
        
        print(f"\n[{rx_file_count}/{total_rx_files}] Area{area_index}_{fc}GHz_RX{RX_index}")
        
        # Step 1: P1A LoS Ray 로딩
        los_ray_data = config.load_p1a_los_ray(area_index, fc, RX_index)
        if los_ray_data is None:
            print(f"  Error: LoS Ray 없음 (스킵)")
            continue
        
        print(f"  LoS Ray: {los_ray_data['los_rays']}개 (전체: {los_ray_data['total_rays']}개)")
        
        # Convert LoS ray data to TensorFlow tensors
        # LoS Ray만 compact array로 추출: (1, 1, 1, 1, N_los_rays)
        # N_los_rays는 RX마다 다를 수 있음 (일반적으로 20개 내외)
        ray_aoa_rad = tf.convert_to_tensor(np.deg2rad(los_ray_data['phi_r_deg']))
        ray_aod_rad = tf.convert_to_tensor(np.deg2rad(los_ray_data['phi_t_deg']))
        ray_zoa_rad = tf.convert_to_tensor(np.deg2rad(los_ray_data['theta_r_deg']))
        ray_zod_rad = tf.convert_to_tensor(np.deg2rad(los_ray_data['theta_t_deg']))
        ray_power = tf.convert_to_tensor(los_ray_data['power'])
        ray_delay = tf.convert_to_tensor(los_ray_data['tau'])
        
        # Compact array 검증 출력
        n_los_rays = ray_power.shape[-1]
        print(f"  Compact array shape: (1,1,1,1,{n_los_rays}) - 유효 LoS Ray만 포함")
        
        # 안테나 어레이 설정 (정적, 한 번만 수행)
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

        # Step 2: 채널 생성기 초기화 (initial phase [0,0,0])
        OFDM_ChGen = ChCoeGen(carrier_frequency, config.OFDM_SCS, ArrayTX, ArrayRX, False)
        
        # XPR 설정 (Cross-Polarization Ratio)
        # Shape: [batch_size, N_BS, N_UE, 1, N_los_rays]
        # ray_aoa_rad.shape[-1] 사용으로 LoS Ray 개수에 자동 맞춤
        ray_xpr = 10**(tf.random.normal(shape=[batch_size, N_BS, N_UE, 1, ray_aoa_rad.shape[-1]], 
                                                mean=config.mean_xpr, stddev=config.stddev_xpr)/10)

        ray_los = Rays(delays=ray_delay, powers=ray_power, aoa=ray_aoa_rad, aod=ray_aod_rad,
                       zoa=ray_zoa_rad, zod=ray_zod_rad, xpr=ray_xpr)

        # 토폴로지 생성 (도플러 없음)
        velocities = tf.zeros([batch_size, N_UE, 3], dtype=tf.float32)  # 속도 0 (정적)
        moving_end = config.Topology_Statistics["moving_end"]
        los_aoa = tf.zeros([batch_size,N_BS,N_UE])
        los_aod = tf.zeros([batch_size,N_BS,N_UE])
        los_zoa = tf.zeros([batch_size,N_BS,N_UE])
        los_zod = tf.zeros([batch_size,N_BS,N_UE])
        los = tf.ones([batch_size,N_BS,N_UE], dtype=tf.bool)  # LoS 채널
        distance_3d = tf.ones([1,N_BS,N_UE])

        # TX 방향: config.TX_Orientation 설정값 사용 (고정 방향)
        tx_orientations = tf.constant([[[  
            np.deg2rad(config.TX_Orientation["azimuth_deg"]),
            np.deg2rad(config.TX_Orientation["downtilt_deg"]), 
            np.deg2rad(config.TX_Orientation["roll_deg"])
        ]]], dtype=tf.float32)

        # RX 방향: 고정 (0,0,0) - 단말 방향 고정
        rx_orientations = tf.constant([[[  
            np.deg2rad(config.RX_Orientation["azimuth_deg"]),
            np.deg2rad(config.RX_Orientation["elevation_deg"]), 
            np.deg2rad(config.RX_Orientation["roll_deg"])
        ]]], dtype=tf.float32)

        topology = Topology(velocities,
                            moving_end,
                            los_aoa,
                            los_aod,
                            los_zoa,
                            los_zod,
                            los,
                            distance_3d,
                            tx_orientations,
                            rx_orientations)

        # Step 3: 평균 채널 행렬 계산
        mean_ch_engine = MeanChannel_Engine(config, OFDM_ChGen, topology, ray_los)
        
        start_computation = time.time()
        mean_ch_data = mean_ch_engine.compute_mean_channel(area_index, fc, RX_index)
        computation_time = time.time() - start_computation
        
        print(f"  계산 시간: {computation_time:.2f}초")
        
        # Step 4: 평균 채널 저장
        if mean_ch_data:
            mean_ch_manager.save_rx_mean_channel(mean_ch_data)
        else:
            print(f"  Error: 평균 채널 계산 실패")
        
        # RX 처리 후 메모리 정리
        del mean_ch_engine, los_ray_data
        
        # 매 RX마다 가비지 컬렉션 수행 (메모리 최적화)
        import gc
        gc.collect()
        tf.keras.backend.clear_session()
        
        toc_total = time.time()

        # 설정된 간격마다 또는 마지막 RX일 때 화면 클리어 + 완료 메시지 출력
        if config.ENABLE_SCREEN_CLEAR and (rx_file_count % config.PROGRESS_CLEAR_INTERVAL == 0 or rx_file_count == total_rx_files):
            # 화면 클리어 (TF 로그 축적 방지)
            if JUPYTER_AVAILABLE:
                clear_output(wait=True)
            else:
                import platform
                os.system('cls' if platform.system() == 'Windows' else 'clear')
            
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
            
            # 진행 상황 요약 재출력
            print("P1H 평균 채널 행렬 추정 진행률:")
            print(f"✓ [{rx_file_count:>{total_rx_width}}/{total_rx_files}] Area{area_index}_{fc}GHz_RX{RX_index} (평균 채널 저장 완료)")
            print(f"현재 완료율: {rx_file_count/total_rx_files*100:.1f}% ({rx_file_count}/{total_rx_files})")
            print(f"전체 경과 시간: {format_time(overall_elapsed)}")
            print(f"평균 처리 속도: {avg_time_per_rx:.1f}초/RX")
            print(f"예상 남은 시간: {format_time(estimated_remaining)}")
            print("-" * 80)
        
    # 분석 완료 메시지
    total_rx_processed = len(config.p1b_valid_combinations)
    overall_time = time.time() - overall_start_time
    
    print(f"\n=== P1H 평균 채널 행렬 추정 완료 ===")
    print(f"총 {total_rx_processed}개 유효 RX 처리 완료 (P1B 기준)")
    print(f"LoS Ray 기반 (initial phase [0,0,0])")
    print(f"총 소요 시간: {overall_time/60:.1f}분")
    print(f"저장 위치: {config.P1H_OUTPUT_DIR}")
    
    # 최종 메모리 정리
    import gc
    gc.collect()
    tf.keras.backend.clear_session()

if __name__ == "__main__":
    main()
