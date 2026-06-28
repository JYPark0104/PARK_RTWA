# ======================================================================
# P1G_Rays_to_CouplingMat_2510v1.py
# P1G: Weichselberger 모델 커플링 행렬 추정 도구
# 
# === 최상위 목적 ===
# P1F 결과 (Marginal CCM)로부터 Weichselberger 모델 파라미터 추정
# - Step 2: 고유벡터 행렬 추정 (U_tx, U_rx) via 고유값 분해
# - Step 3: 커플링 행렬 추정 (Ω) via 고유빔 공간 평균 전력
# 
# === 핵심 설계 원리 ===
# 1. 데이터 흐름: P1F Marginal CCM → EVD → 채널 샘플 생성 → 커플링 행렬
# 2. 고유값 정렬: 내림차순 정렬 (고유빔 중요도 순)
# 3. RX별 독립 처리: 각 RX의 U_tx, U_rx, Ω를 개별 파일로 저장
# 
# === 주요 구성 요소 ===
# - P1G_Config: P1F 결과 로딩 + P1G 저장 경로 설정
# - ChCoeGen: Ray → Antenna Element OFDM 채널 생성 (P1F 재사용)
# - CouplingMatrix_Engine: EVD + 채널 샘플링 + 커플링 행렬 계산
# - CouplingMatrix_Manager: 커플링 행렬 파일 저장/로딩 관리
# 
# === 입력/출력 ===
# 입력: P1F 결과 (Area{area}_{freq}GHz_RX{rx}_Marginal_CCM.npz)
#       - R_BS: [n_t, n_t] TX marginal covariance
#       - R_UE: [n_r, n_r] RX marginal covariance
# 출력: Area{area}_{freq}GHz_RX{rx}_CouplingMat.npz
#       - U_tx: [n_t, n_t] TX 고유벡터 행렬
#       - U_rx: [n_r, n_r] RX 고유벡터 행렬
#       - Lambda_tx: [n_t] TX 고유값 (내림차순)
#       - Lambda_rx: [n_r] RX 고유값 (내림차순)
#       - Omega: [n_r, n_t] 커플링 행렬
#       - metadata: 샘플 수, 안테나 구성 등
# 
# === 핵심 기술 ===
# - 고유값 분해 (tf.linalg.eigh) + 내림차순 정렬
# - 온라인 누적: 커플링 행렬 = E[|u_rx^H @ H @ u_tx|^2]
# - P1F 채널 생성 방식 재사용 (독립 정적 채널 발생)
#
# === 주요 수정 이력 ===
# [251014] P1G 신규 작성: Weichselberger 커플링 행렬 추정
# 1. P1F Marginal CCM 로딩
# 2. Step 2: EVD with 고유값 내림차순 정렬
# 3. Step 3: 커플링 행렬 온라인 누적 계산
# 4. 의존성 기반 순서: Config → ChCoeGen → Engine → Manager → main
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

# ===== SECTION 2: P1G_Config (P1F Marginal CCM 기반 설정) =====
class P1G_Config:
    """P1F Marginal CCM 결과를 기반으로 Weichselberger 커플링 행렬 추정 설정"""
    
    def __init__(self):
        # 파일 경로 및 명명 패턴 설정 (중앙 집중화)

        # 스크립트의 디렉토리를 기준으로 절대 경로 설정
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # P1F Marginal CCM 입력 설정
        self.P1F_CCM_INPUT_DIR = os.path.join(script_dir, "P1F_Marginal_CCM_Results")
        self.P1F_CCM_FILE_PATTERN = "Area{area}_{freq}GHz_RX{rx}_Marginal_CCM.npz"
        
        # P1B Valid RXs 입력 설정 (Ray 데이터 로딩용)
        self.P1B_INPUT_DIR = os.path.join(script_dir, "P1B_Valid_Results")
        self.P1B_FILE_PATTERN = "Area{area}_{freq}GHz_Rays_Valid_RXs.npz"
        
        # P1G 커플링 행렬 저장 설정
        self.P1G_OUTPUT_DIR = os.path.join(script_dir, "P1G_CouplingMat_Results")
        self.P1G_FILE_PATTERN = "Area{area}_{freq}GHz_RX{rx}_CouplingMat.npz"
        
        # 콘솔 출력 제어 설정
        self.ENABLE_SCREEN_CLEAR = True      # 화면 클리어 기능 활성화/비활성화
        self.PROGRESS_CLEAR_INTERVAL = 10    # 화면 클리어 개수
        
        # 필터링 설정
        self.target_areas = [1]  # 처리할 area 목록 (None이면 전체)
        self.target_rxs = None   # 처리할 RX 목록 (None이면 전체)
        
        # P1F 데이터 디렉토리 스캔하여 실험 설정 자동 감지
        self.detect_p1f_data()
        
        # fc 리스트를 self.p1f_data_combinations에서 추출하여 설정
        if self.p1f_data_combinations:
            self.fcs = sorted(list(set([item[1] for item in self.p1f_data_combinations])))
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

        # 커플링 행렬 추정용 채널 샘플 생성 파라미터  
        self.static_ch_realizations = 256 # 독립 정적 채널 개수
        self.doppler_time_realizations = 1 # 각 정적 채널의 도플러 샘플 수
        self.doppler_sym_ofdm = 1 # 각 도플러 샘플당 OFDM 심볼 수
        self.doppler_time_max_sec = coherence_time*0 # 도플러 시간 샘플링 최대값 (초)
                
        # 고유값 및 커플링 행렬 통계 분석 설정
        self.eigenvalue_threshold_ratio = 1e-3  # 평균 고유값 대비 상대 임계값 비율
        self.coupling_threshold_ratio = 1e-3    # 평균 커플링 대비 상대 임계값 비율
        
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

        # P1G 설정 정보 출력
        self.print_p1g_config()
        
    def detect_p1f_data(self):
        """P1F Marginal CCM 데이터를 스캔하여 실험 설정 자동 감지"""
        
        # P1F 출력 디렉토리에서 Marginal CCM npz 파일들 스캔
        scan_pattern = f"{self.P1F_CCM_INPUT_DIR}/{self.P1F_CCM_FILE_PATTERN.format(area='*', freq='*', rx='*')}"
        files = glob.glob(scan_pattern)
        
        # (area, freq, rx) 조합을 저장할 집합
        combinations = set()
        
        # 파일명에서 area_index, frequency, rx_index 추출
        for file_path in files:
            filename = os.path.basename(file_path)
            # Area{X}_{freq}GHz_RX{rx}_Marginal_CCM.npz 패턴 매칭
            match = re.match(r'Area(\d+)_(.+)GHz_RX(\d+)_Marginal_CCM\.npz', filename)
            if match:
                area_index = int(match.group(1))
                frequency = float(match.group(2))
                rx_index = int(match.group(3))
                
                # target_areas 필터링 적용
                if self.target_areas is not None and area_index not in self.target_areas:
                    continue
                
                # target_rxs 필터링 적용
                if self.target_rxs is not None and rx_index not in self.target_rxs:
                            continue
                        
                combinations.add((area_index, frequency, rx_index))
        
        # 정렬하여 리스트로 저장
        self.p1f_data_combinations = sorted(list(combinations))
        
        print(f"P1F Marginal CCM 데이터 자동 감지:")
        print(f"  - 감지된 조합 수: {len(self.p1f_data_combinations)}")
        
        # Area별로 그룹핑하여 출력
        area_groups = {}
        for combo in self.p1f_data_combinations:
            area, freq, rx = combo
            key = f"Area{area}_{freq}GHz"
            if key not in area_groups:
                area_groups[key] = []
            area_groups[key].append(rx)
        
        for area_freq, rx_list in sorted(area_groups.items()):
            print(f"    - {area_freq}: RX{min(rx_list)}-RX{max(rx_list)} ({len(rx_list)} RXs)")
    
    def load_p1b_ray_data(self, area_index, frequency, rx_index):
        """Load ray data for specific area, frequency, and RX from P1B Valid RXs npz file
        
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
            Dictionary containing ray parameters and LoS/NLoS verification for the specified RX
        """
        # Load P1B Valid RXs npz file
        npz_filename = self.P1B_FILE_PATTERN.format(area=area_index, freq=frequency)
        npz_filepath = os.path.join(self.P1B_INPUT_DIR, npz_filename)
        
        try:
            with np.load(npz_filepath) as data:
                # Find RX index position
                rx_indices = data['rx_indices']
                rx_position = np.where(rx_indices == rx_index)[0]
                
                if len(rx_position) == 0:
                    raise ValueError(f"RX{rx_index} not found in {npz_filename}")
                
                rx_pos = rx_position[0]
                
                # Extract ray data for specified RX (P1B format)
                ray_data = {
                    'phi_r_deg': data['phi_r_deg'][rx_pos],    # AoA in degrees
                    'phi_t_deg': data['phi_t_deg'][rx_pos],    # AoD in degrees
                    'theta_r_deg': data['theta_r_deg'][rx_pos], # ZoA in degrees
                    'theta_t_deg': data['theta_t_deg'][rx_pos], # ZoD in degrees
                    'power': data['power'][rx_pos],            # Power
                    'tau': data['tau'][rx_pos],                # Delay
                }
                
                # Enhanced: LoS/NLoS verification (P1D 품질 보증)
                los_nlos_info = self._verify_nlos_ratio(data, rx_pos, area_index, frequency, rx_index)
                ray_data.update(los_nlos_info)
                
                return ray_data
                
        except Exception as e:
            print(f"Error loading ray data from {npz_filepath}: {e}")
            return None
    
    def _verify_nlos_ratio(self, data, rx_pos, area_index, frequency, rx_index):
        """P1D 기본 검증: NLoS Ray 비율 확인 (P1B 정상 수행 전제)"""
        
        # 기본 Ray 개수 확인
        if 'power' in data:
            ray_powers = data['power'][rx_pos]
            valid_ray_mask = ~np.isnan(ray_powers) & (ray_powers > 0)
            total_rays = int(np.sum(valid_ray_mask))
        else:
            total_rays = 0
        
        # Enhanced P1A 데이터의 LoS/NLoS 정보 확인
        if 'los_nlos_flag' in data and total_rays > 0:
            los_nlos_flags = data['los_nlos_flag'][rx_pos][valid_ray_mask]
            nlos_rays = int(np.sum(los_nlos_flags == 0))
            nlos_ratio = nlos_rays / total_rays
            
            return {
                'has_los_nlos_info': True,
                'total_rays': total_rays,
                'nlos_rays': nlos_rays,
                'nlos_ratio': nlos_ratio
            }
        else:
            return {
                'has_los_nlos_info': False,
                'total_rays': total_rays,
                'nlos_rays': 0,
                'nlos_ratio': 0.0
            }
    
    def create_chunk_from_files(self, area_index, frequency, rx_indices):
        """개별 npy 파일들을 npz 청크로 묶기"""
        if not rx_indices:
            return None
        
        chunk_data = {}
        chunk_metadata = {
            'area_idx': area_index,
            'freq_ghz': frequency, 
            'rx_indices': np.array(rx_indices),
            'rx_count': len(rx_indices)
        }
        
        # 개별 파일들 로드 (진행률 표시)
        total_rx = len(rx_indices)
        
        # Static 파일 모드: 각 RX의 모든 Static 파일들을 로드  
        chunk_metadata['static_ch_realizations'] = self.static_ch_realizations
        chunk_metadata['storage_mode'] = 'static'
        
        for rx_enumeration_idx, current_rx_identifier in enumerate(rx_indices, 1):
            progress_message = f"  Static 파일 로딩 중... RX{current_rx_identifier} ({rx_enumeration_idx}/{total_rx})"
            print(f"\r{progress_message:<60}", end='', flush=True)
            
            rx_static_data_collection = {}
            for static_realization_number in range(1, self.static_ch_realizations + 1):
                static_file_path_current = os.path.join(self.P1D_OUTPUT_DIR, 
                                            self.P1D_STATIC_FILE_PATTERN.format(
                                                  area=area_index, freq=frequency, 
                                                rx=current_rx_identifier, static_idx=static_realization_number))
                
                if os.path.exists(static_file_path_current):
                    static_data_loaded_current = np.load(static_file_path_current)
                    rx_static_data_collection[f'static_{static_realization_number}'] = dict(static_data_loaded_current)
                else:
                    print(f"\nWarning: {static_file_path_current} not found")
        
        # 청크 파일 저장
        chunk_filename = self.CHUNK_PATTERN.format(
            area=area_index, freq=frequency, start=min(rx_indices), end=max(rx_indices)
        )
        chunk_filepath = os.path.join(self.P1D_OUTPUT_DIR, chunk_filename)
        
        # 메타데이터와 함께 저장
        save_data = {**chunk_metadata, **chunk_data}
        np.savez_compressed(chunk_filepath, **save_data)
        
        chunk_size_mb = os.path.getsize(chunk_filepath) / (1024**2)
        print(f"청크 저장 완료: {chunk_filename} ({chunk_size_mb:.1f} MB)")
        
        # 개별 Static 파일 정리
        if self.CLEANUP_INDIVIDUAL_FILES:
            cleanup_deleted_count = 0
            for cleanup_rx_idx in rx_indices:
                for cleanup_static_idx in range(1, self.static_ch_realizations + 1):
                    cleanup_static_file_path = os.path.join(self.P1D_OUTPUT_DIR,
                                             self.P1D_STATIC_FILE_PATTERN.format(
                                                      area=area_index, freq=frequency, 
                                                 rx=cleanup_rx_idx, static_idx=cleanup_static_idx))
                    if os.path.exists(cleanup_static_file_path):
                        os.remove(cleanup_static_file_path)
                        cleanup_deleted_count += 1
            print(f"개별 Static 파일 {cleanup_deleted_count}개 정리 완료")
        
        return chunk_filepath
    

    def load_p1f_marginal_ccm(self, area_index, frequency, rx_index):
        """Load P1F Marginal CCM data for specific RX
        
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
            Dictionary containing R_BS, R_UE, metadata
        """
        filename = self.P1F_CCM_FILE_PATTERN.format(area=area_index, freq=frequency, rx=rx_index)
        filepath = os.path.join(self.P1F_CCM_INPUT_DIR, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"P1F Marginal CCM file not found: {filepath}")
        
        data = np.load(filepath, allow_pickle=True)
        
        return {
            'R_BS': data['R_BS'],
            'R_UE': data['R_UE'],
            'metadata': data['metadata'].item(),
            'validation': data.get('validation', {})
        }
    
    def print_p1g_config(self):
        """P1G 설정 정보 출력"""
        print(f"P1G 커플링 행렬 추정 설정:")
        print(f"  - 커플링 추정용 Static 실현: {self.static_ch_realizations}")
        print(f"  - OFDM FFT Size: {self.OFDM_FFT}")
        print(f"  - 예상 총 OFDM 샘플 수 (per RX): {self.static_ch_realizations * self.OFDM_FFT}")
        print(f"  - 안테나 구성: TX {self.n_t}개, RX {self.n_r}개")
        print(f"  - 출력: U_tx [{self.n_t}×{self.n_t}], U_rx [{self.n_r}×{self.n_r}], Ω [{self.n_r}×{self.n_t}]")
        print(f"  - 고유값 Threshold: λ_mean × {self.eigenvalue_threshold_ratio:.0e}")
        print(f"  - 커플링 Threshold: Ω_mean × {self.coupling_threshold_ratio:.0e}")
        print(f"  - P1F 입력: {self.P1F_CCM_INPUT_DIR}")
        print(f"  - P1G 결과 저장: {self.P1G_OUTPUT_DIR}")
        
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


# ===== MATLAB Column-Major 유틸리티 함수 =====
@tf.function(jit_compile=True)
def vec_mat_py_tf(H_tf):
    """MATLAB vec(H) 구현 (Column-Major, TensorFlow Only, JIT 최적화)
    
    Args:
        H_tf: 입력 행렬/텐서
            - 2D: [n_r, n_t] → [n_r*n_t]
            - 3D: [batch, n_r, n_t] → [batch, n_r*n_t]
    
    Returns:
        vectorized 결과 (Column-Major 순서)
    """
    if len(H_tf.shape) == 2:
        return tf.reshape(tf.transpose(H_tf), [-1])
    elif len(H_tf.shape) == 3:
        return tf.reshape(tf.transpose(H_tf, perm=[0, 2, 1]), [tf.shape(H_tf)[0], -1])
    else:
        raise ValueError(f"Unsupported shape: {H_tf.shape}")

@tf.function(jit_compile=True)
def kron_mat_py_tf(A_tf, B_tf):
    """MATLAB kron(A,B) 구현 (einsum & reshape, TensorFlow Only, JIT 최적화)
    
    Args:
        A_tf: 첫 번째 행렬 [m, n]
        B_tf: 두 번째 행렬 [p, q]
    
    Returns:
        크로네커 곱 A ⊗ B [m*p, n*q]
    """
    # einsum으로 4D 텐서 생성: A[i,j] * B[k,l] -> [i,k,j,l]
    kron_4d = tf.einsum('ij,kl->ikjl', A_tf, B_tf)
    # reshape으로 2D 행렬로 변환 (Column-Major vec 일관성)
    return tf.reshape(kron_4d, [tf.shape(A_tf)[0] * tf.shape(B_tf)[0], 
                               tf.shape(A_tf)[1] * tf.shape(B_tf)[1]])

# ===== ChCoeGen 클래스 (핵심 부분만 복사) =====
class ChCoeGen(Object):
    """P1D: Ray → Antenna Element OFDM Channel 변환 (P1B Valid RXs 기반)
    
    핵심 메서드:
    - _compute_ch_mimo_ofdm_38901_static(): 정적 MIMO OFDM 채널 생성
      * 매번 새로운 초기 위상(_step_10) 생성
      * Ray → delay bin domain 변환
    - _apply_doppler_ch_mimo_ofdm_freq(): 도플러 효과 적용
      * 직접 float 시간 샘플링
      * 시변 채널 생성
    
    P1D 구조:
    - static_ch_realizations: 독립 정적 채널 개수 (외부 루프)
    - doppler_time_realizations: 각 정적 채널의 시변 샘플 수 (내부 루프)
    - 최종 심볼 수: static_ch_realizations × doppler_time_realizations × doppler_sym_ofdm
    
    P1 series 연계:
    - P1A: 3D RT 시뮬레이션 → Ray 파라미터 (각도, 지연, 전력)
    - P1B: Ray 필터링 → Valid RXs (음수 지연 제거)
    - P1D: Valid RXs → Antenna Element OFDM 채널
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

    def _step_10(self, shape):
        r"""
        Generate random and uniformly distributed phases for all rays and
        polarization combinations

        Input
        -----
        shape : Shape tensor
            Shape of the leading dimensions for the tensor of phases to generate

        Output
        ------
        phi : [shape] + [4], tf.float
            Phases for all polarization combinations
        """
        phi = tf.random.uniform(
                             tf.concat([shape, [4]], axis=0),
                             minval=-PI,
                             maxval=PI,
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
            phi = self._step_10(tf.shape(aoa_delay_bin))

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

# ===== SECTION 2.5: 고유값 분해 유틸리티 =====
@tf.function
def perform_eigen_decomposition_sorted(R_matrix: tf.Tensor) -> tuple:
    """고유값 분해 및 내림차순 정렬 (Step 2)
    
    Args:
        R_matrix: Hermitian 행렬 [dim, dim]
    
    Returns:
        U: 고유벡터 행렬 [dim, dim] (고유값 내림차순 정렬)
        Lambda: 고유값 벡터 [dim] (내림차순 정렬, 실수)
    """
    # 고유값 분해 (오름차순)
    eigenvals, eigenvecs = tf.linalg.eigh(R_matrix)
    
    # Hermitian 행렬의 고유값은 실수 (허수부는 수치 오차)
    eigenvals_real = tf.math.real(eigenvals)
    
    # 내림차순 정렬
    sorted_indices = tf.argsort(eigenvals_real, direction='DESCENDING')
    Lambda_sorted = tf.gather(eigenvals_real, sorted_indices)
    U_sorted = tf.gather(eigenvecs, sorted_indices, axis=1)
    
    return U_sorted, Lambda_sorted

# ===== SECTION 3: 커플링 행렬 검증 유틸리티 =====
def validate_eigendecomposition(U: tf.Tensor, Lambda: tf.Tensor, R: tf.Tensor,
                                threshold: float, matrix_name: str = "U") -> dict:
    """고유값 분해 결과 검증 (재구성 오차, CMD, threshold 분석)
        
        Args:
        U: 고유벡터 행렬 [dim, dim]
        Lambda: 고유값 벡터 [dim] (실수, 내림차순 정렬됨)
        R: 원본 공분산 행렬 [dim, dim]
        threshold: 유효 고유값 판단 임계값
        matrix_name: 행렬 이름 (로깅용)
    
    Returns:
        dict: 검증 결과
            - reconstruction_error: ||R - UΛU^H||_F / ||R||_F
            - cmd: trace(R^H UΛU^H) / (||R||_F ||UΛU^H||_F)
            - max_eigenvalue: 최대 고유값
            - min_eigenvalue: 최소 고유값
            - num_eigenvalues: 총 고유값 개수
            - num_above_threshold: threshold 이상 고유값 개수
            - ratio_above_threshold: threshold 이상 고유값 비율
    """
    # 1. EVD 재구성: R_reconstructed = U Λ U^H
    Lambda_diag = tf.linalg.diag(tf.cast(Lambda, U.dtype))
    R_reconstructed = tf.matmul(tf.matmul(U, Lambda_diag), tf.linalg.adjoint(U))
    
    # 2. 재구성 오차: ||R - UΛU^H||_F / ||R||_F
    # Frobenius norm: sqrt(sum(|X_ij|^2))
    diff = R - R_reconstructed
    reconstruction_error_abs = tf.sqrt(tf.reduce_sum(tf.square(tf.abs(diff))))
    R_norm = tf.sqrt(tf.reduce_sum(tf.square(tf.abs(R))))
    reconstruction_error = reconstruction_error_abs / R_norm
    
    # 3. Correlation Matrix Distance (CMD): trace(R^H UΛU^H) / (||R||_F ||UΛU^H||_F)
    R_recon_norm = tf.sqrt(tf.reduce_sum(tf.square(tf.abs(R_reconstructed))))
    trace_product = tf.math.real(tf.linalg.trace(tf.matmul(tf.linalg.adjoint(R), R_reconstructed)))
    cmd = trace_product / (R_norm * R_recon_norm)
    
    # 4. 고유값 범위 및 통계
    max_eig = tf.reduce_max(Lambda)
    min_eig = tf.reduce_min(Lambda)
    
    # 5. Threshold 이상 고유값 개수
    num_total = int(tf.shape(Lambda)[0].numpy())
    num_above = int(tf.reduce_sum(tf.cast(Lambda >= threshold, tf.int32)).numpy())
    ratio_above = num_above / num_total if num_total > 0 else 0.0
    
    return {
        'reconstruction_error': float(reconstruction_error.numpy()),
        'cmd': float(cmd.numpy()),
        'max_eigenvalue': float(max_eig.numpy()),
        'min_eigenvalue': float(min_eig.numpy()),
        'num_eigenvalues': num_total,
        'num_above_threshold': num_above,
        'ratio_above_threshold': ratio_above,
        'threshold': threshold
    }


# ===== SECTION 4: 커플링 행렬 추정 엔진 =====
class CouplingMatrix_Engine:
    """커플링 행렬 추정: EVD + 채널 샘플링 + 커플링 행렬 계산"""
    
    def __init__(self, config, OFDM_ChGen, topology, ray_pdap, U_tx, U_rx, trace_tx, trace_rx):
        self.config = config
        self.OFDM_ChGen = OFDM_ChGen
        self.topology = topology
        self.ray_pdap = ray_pdap
        
        # 고유벡터 행렬 (TensorFlow 텐서로 변환)
        self.U_tx = tf.convert_to_tensor(U_tx, dtype=tf.complex64)
        self.U_rx = tf.convert_to_tensor(U_rx, dtype=tf.complex64)
        
        # Trace normalization 스케일 (복구용, Python float)
        self.trace_tx = trace_tx
        self.trace_rx = trace_rx
        
        # 커플링 행렬 온라인 평균 누적 변수
        self.count = tf.Variable(0, dtype=tf.int64, trainable=False, name="coupling_count")
        self.Omega_mean = tf.Variable(
            tf.zeros([config.n_r, config.n_t], dtype=tf.float32),
            trainable=False, name="Omega_mean")
        
        # 수렴도 추적 변수
        self.last_convergence_metric = None
        
    @tf.function(jit_compile=True)
    def _update_coupling_matrix(self, H_batch):
        """커플링 행렬 온라인 평균 업데이트 (Step 3, 수치적 안정성)
        
        Args:
            H_batch: [batch_size, n_r, n_t] 채널 샘플 배치
        
        Returns:
            normalized_change: ||update||_F / ||Omega_mean_old||_F (수렴도 지표)
        
        커플링 행렬 공식:
            ω_{q,p} = E[|u_rx^H @ H @ u_tx|^2]
        
        온라인 평균 업데이트:
            mean_new = mean_old + (mean_batch - mean_old) * (batch_size / count_new)
        """
        batch_size = tf.shape(H_batch)[0]
        
        if batch_size > 0:
            # 고유빔 공간 변환: G = U_rx^H @ H @ U_tx
            # H_batch: [batch, n_r, n_t]
            # U_rx: [n_r, n_r], U_tx: [n_t, n_t]
            
            # Step 1: H @ U_tx → [batch, n_r, n_t]
            H_U_tx = tf.matmul(H_batch, self.U_tx)
            
            # Step 2: U_rx^H @ (H @ U_tx) → [batch, n_r, n_t]
            G_batch = tf.matmul(tf.linalg.adjoint(self.U_rx), H_U_tx)
            
            # 커플링 행렬 요소: |G[q,p]|^2
            power_batch = tf.abs(G_batch) ** 2  # [batch, n_r, n_t]
            
            # 배치 평균
            power_mean_batch = tf.reduce_mean(power_batch, axis=0)  # [n_r, n_t]
            
            # 온라인 평균 업데이트
            count_old = tf.cast(self.count, tf.float32)
            count_new = count_old + tf.cast(batch_size, tf.float32)
            
            # Welford 온라인 알고리즘: mean_new = mean_old + delta * (batch_size / count_new)
            delta = power_mean_batch - self.Omega_mean
            update_amount = delta * (tf.cast(batch_size, tf.float32) / count_new)
            
            # 수렴도 계산: ||update||_F / ||Omega_mean_old||_F
            Omega_norm = tf.sqrt(tf.reduce_sum(tf.square(self.Omega_mean)))
            update_norm = tf.sqrt(tf.reduce_sum(tf.square(update_amount)))
            
            # 첫 iteration (Omega_norm ≈ 0)에서는 normalized_change = 1.0
            # 이후에는 Omega_norm > 0이므로 regularization 불필요
            normalized_change = tf.cond(
                Omega_norm > 1e-20,
                lambda: update_norm / Omega_norm,
                lambda: tf.constant(1.0, dtype=tf.float32)
            )
            
            self.Omega_mean.assign_add(update_amount)
            self.count.assign_add(tf.cast(batch_size, tf.int64))
            
            return normalized_change
        
        return tf.constant(0.0, dtype=tf.float32)
    
    def process_static_realization(self, static_idx, is_last_static=False):
        """단일 Static 실현 생성 및 커플링 행렬 누적
        
        Parameters:
            static_idx: 0-based 인덱스 (내부 처리용)
            is_last_static: 마지막 static 여부 (수렴도 추적용)
        """
        
        # 1. Static Channel 생성
        h_delay_bin_static, aoa_delay_bin, zoa_delay_bin = \
            self.OFDM_ChGen._compute_ch_mimo_ofdm_38901_static(
                self.topology, self.ray_pdap, 
                self.config.OFDM_FFT, self.config.OFDM_SCS)
        
        # 2. 필수 transpose 처리
        h_delay_bin_static = tf.transpose(h_delay_bin_static, [0,3,5,6,1,2,7,4])
        aoa_delay_bin = tf.transpose(aoa_delay_bin, [0,3,1,2,4])
        zoa_delay_bin = tf.transpose(zoa_delay_bin, [0,3,1,2,4])
        
        # 3. Doppler 처리
        for doppler_idx in range(self.config.doppler_time_realizations):
            doppler_times = tf.random.uniform(
                shape=[self.config.doppler_sym_ofdm],
                minval=0.0,
                maxval=self.config.doppler_time_max_sec,
                dtype=self.OFDM_ChGen.rdtype)
            
            ofdm_ch_doppler = self.OFDM_ChGen._apply_doppler_ch_mimo_ofdm_freq(
                self.topology, doppler_times, 
                h_delay_bin_static, aoa_delay_bin, zoa_delay_bin)
            
            # 4. 차원 변환: [B, N_r, N_t, N_BS, N_UE, N_sym, N_FFT] 
            #           → [B, N_sym, N_FFT, N_r, N_t]
            channel_sample = tf.transpose(ofdm_ch_doppler, [0,3,4,5,6,1,2])
            channel_sample = tf.squeeze(channel_sample, axis=[1,2])
            
            # 5. 커플링 행렬 업데이트
            # [1, n_sym, n_fft, n_r, n_t] → [n_sym*n_fft, n_r, n_t]
            batch_size = tf.shape(channel_sample)[0]
            n_sym = tf.shape(channel_sample)[1]
            n_fft = tf.shape(channel_sample)[2]
            n_r = tf.shape(channel_sample)[3]
            n_t = tf.shape(channel_sample)[4]
            
            H_batch = tf.reshape(channel_sample, 
                                [batch_size * n_sym * n_fft, n_r, n_t])
            
            # 커플링 행렬 온라인 업데이트
            convergence_metric = self._update_coupling_matrix(H_batch)
            
            # 마지막 static의 마지막 doppler iteration에서 수렴도 저장
            if is_last_static and doppler_idx == self.config.doppler_time_realizations - 1:
                self.last_convergence_metric = float(convergence_metric.numpy())
            
            # 메모리 해제
            del ofdm_ch_doppler, channel_sample, H_batch
        
        # 메모리 해제 (Static별)
        del h_delay_bin_static, aoa_delay_bin, zoa_delay_bin
    
    def finalize_coupling_matrix(self, area_index, fc, RX_index, threshold_ratio: float):
        """누적 완료 후 커플링 행렬 추출 및 통계
        
        Args:
            area_index, fc, RX_index: RX 식별 정보
            threshold_ratio: 평균 커플링 대비 상대 임계값 비율
        
        Returns:
            dict: 커플링 행렬 데이터
                - area_idx, freq_ghz, rx_idx
                - Omega: [n_r, n_t] 커플링 행렬
                - n_samples: 누적된 OFDM 샘플 수
                - coupling_stats: 커플링 행렬 통계
        """
        N_total = int(self.count.numpy())
        
        if N_total < 1:
            print(f"  Error: 샘플 부족 (N={N_total})")
            return None
        
        # 커플링 행렬 (Weichselberger coupling matrix)
        Omega_final = self.Omega_mean.numpy()  # [n_r, n_t] float32
        
        # 통계 및 threshold 계산
        Omega_mean = float(np.mean(Omega_final))
        threshold = Omega_mean * threshold_ratio
        
        # Threshold 이상 원소 개수
        num_total_elements = self.config.n_r * self.config.n_t
        num_above = int(np.sum(Omega_final >= threshold))
        ratio_above = num_above / num_total_elements if num_total_elements > 0 else 0.0
        
        # Omega_final 통계 (저장용)
        max_coupling = float(np.max(Omega_final))
        min_coupling = float(np.min(Omega_final))
        mean_coupling = Omega_mean
        sum_coupling = float(np.sum(Omega_final))
        
        coupling_stats = {
            'max_coupling': max_coupling,
            'min_coupling': min_coupling,
            'mean_coupling': mean_coupling,
            'sum_coupling': sum_coupling,
            'num_elements': num_total_elements,
            'num_above_threshold': num_above,
            'ratio_above_threshold': ratio_above,
            'threshold': threshold
        }
        
        print(f"  완료: Ω [{self.config.n_r}×{self.config.n_t}], >{threshold:.2e}: {num_above}/{num_total_elements}, sum={sum_coupling:.6f}")
        
        result = {
            'area_idx': area_index,
            'freq_ghz': fc,
            'rx_idx': RX_index,
            'Omega': Omega_final,
            'n_samples': N_total,
            'n_r': self.config.n_r,
            'n_t': self.config.n_t,
            'static_ch_realizations': self.config.static_ch_realizations,
            'coupling_stats': coupling_stats
        }
        
        return result

# ===== SECTION 5: 커플링 행렬 파일 관리 =====
class CouplingMatrix_Manager:
    """커플링 행렬 파일 저장 및 로딩 관리"""
    
    def __init__(self, config):
        self.config = config
        os.makedirs(self.config.P1G_OUTPUT_DIR, exist_ok=True)
    
    def coupling_file_exists(self, area_idx: int, freq_ghz: float, rx_idx: int) -> bool:
        """RX별 커플링 행렬 파일 존재 여부 확인 (재개용)
        
        Args:
            area_idx: Area 인덱스
            freq_ghz: 주파수 (GHz)
            rx_idx: RX 인덱스
        
        Returns:
            bool: 파일 존재 여부
        """
        filename = self.config.P1G_FILE_PATTERN.format(
            area=area_idx, freq=freq_ghz, rx=rx_idx
        )
        filepath = os.path.join(self.config.P1G_OUTPUT_DIR, filename)
        return os.path.exists(filepath)
    
    def save_rx_coupling_matrix(self, coupling_data: dict, U_tx: np.ndarray, U_rx: np.ndarray, 
                                Lambda_tx: np.ndarray, Lambda_rx: np.ndarray,
                                val_tx: dict, val_rx: dict) -> str:
        """RX별 커플링 행렬 npz 파일 저장
        
        Args:
            coupling_data: finalize_coupling_matrix() 반환값
                - area_idx, freq_ghz, rx_idx
                - Omega: 커플링 행렬
                - n_samples, coupling_stats
            U_tx, U_rx: 고유벡터 행렬
            Lambda_tx, Lambda_rx: 고유값 벡터
            val_tx, val_rx: 고유값 분해 검증 결과
        
        Returns:
            str: 저장된 파일 경로
        """
        area_idx = coupling_data['area_idx']
        freq_ghz = coupling_data['freq_ghz']
        rx_idx = coupling_data['rx_idx']
        
        filename = self.config.P1G_FILE_PATTERN.format(
            area=area_idx, freq=freq_ghz, rx=rx_idx
        )
        filepath = os.path.join(self.config.P1G_OUTPUT_DIR, filename)
        
        # 메타데이터 생성
        metadata = {
            'area_idx': area_idx,
            'freq_ghz': freq_ghz,
            'rx_idx': rx_idx,
            'n_samples': coupling_data['n_samples'],
            'n_r': coupling_data['n_r'],
            'n_t': coupling_data['n_t'],
            'static_ch_realizations': coupling_data['static_ch_realizations']
        }
        
        # npz 파일 저장
        np.savez_compressed(
            filepath,
            U_tx=U_tx,
            U_rx=U_rx,
            Lambda_tx=Lambda_tx,
            Lambda_rx=Lambda_rx,
            Omega=coupling_data['Omega'],
            metadata=metadata,
            coupling_stats=coupling_data['coupling_stats'],
            evd_validation_tx=val_tx,
            evd_validation_rx=val_rx
        )
        
        return filepath
    
    def load_rx_coupling_matrix(self, area_idx: int, freq_ghz: float, rx_idx: int) -> dict:
        """저장된 커플링 행렬 파일 로딩
        
        Returns:
            dict: U_tx, U_rx, Lambda_tx, Lambda_rx, Omega, metadata, coupling_stats, evd_validation
        """
        filename = self.config.P1G_FILE_PATTERN.format(
            area=area_idx, freq=freq_ghz, rx=rx_idx
        )
        filepath = os.path.join(self.config.P1G_OUTPUT_DIR, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Coupling matrix file not found: {filepath}")
        
        data = np.load(filepath, allow_pickle=True)
        
        return {
            'U_tx': data['U_tx'],
            'U_rx': data['U_rx'],
            'Lambda_tx': data['Lambda_tx'],
            'Lambda_rx': data['Lambda_rx'],
            'Omega': data['Omega'],
            'metadata': data['metadata'].item(),
            'coupling_stats': data['coupling_stats'].item(),
            'evd_validation_tx': data['evd_validation_tx'].item(),
            'evd_validation_rx': data['evd_validation_rx'].item()
        }


# ===== SECTION 6: 메인 실행 P1G =====
def main():
    """P1G: P1F Marginal CCM 기반 Weichselberger 커플링 행렬 추정"""
    
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
    
    config = P1G_Config()
    
    # 전체 시작 시간 기록 (진행률 계산용)
    overall_start_time = time.time()
    
    # 커플링 행렬 파일 관리자 초기화
    coupling_manager = CouplingMatrix_Manager(config)
    print(f"\n=== P1G 커플링 행렬 추정 시작 ===")
    print(f"입력 디렉토리: {config.P1F_CCM_INPUT_DIR}")
    print(f"출력 디렉토리: {config.P1G_OUTPUT_DIR}")
    print(f"파일 패턴: {config.P1G_FILE_PATTERN}")
    print()
    
    # P1F Marginal CCM 데이터 기반 동적 루프
    total_rx_files = len(config.p1f_data_combinations)
    rx_file_count = 0
    
    # 동적 자리수 맞춤을 위한 너비 계산
    total_rx_width = len(str(total_rx_files))
    
    for area_index, fc, RX_index in config.p1f_data_combinations:
        rx_file_count += 1
        
        # 재개 로직: 이미 저장된 RX는 건너뛰기
        if coupling_manager.coupling_file_exists(area_index, fc, RX_index):
            print(f"\n[{rx_file_count}/{total_rx_files}] Area{area_index}_{fc}GHz_RX{RX_index}: 이미 완료됨 (스킵)")
            continue
        
        carrier_frequency = fc*10**9
        
        # 전체 시작 시간 기록
        tic_total = time.time()
        batch_size = config.batch_size
        N_UE = config.num_rx
        N_BS = config.N_BS
        
        print(f"\n[{rx_file_count}/{total_rx_files}] Area{area_index}_{fc}GHz_RX{RX_index}")
        
        # Step 1: P1F Marginal CCM 로딩
        try:
            p1f_data = config.load_p1f_marginal_ccm(area_index, fc, RX_index)
            R_BS = p1f_data['R_BS']
            R_UE = p1f_data['R_UE']
        except Exception as e:
            print(f"  Error: P1F 데이터 로딩 실패 - {e}")
            continue
        
        # Step 2: Trace normalization for numerical stability
        R_BS_tf = tf.convert_to_tensor(R_BS, dtype=tf.complex64)
        R_UE_tf = tf.convert_to_tensor(R_UE, dtype=tf.complex64)
        
        # Trace로 정규화 (수치 안정성)
        trace_tx = tf.math.real(tf.linalg.trace(R_BS_tf))
        trace_rx = tf.math.real(tf.linalg.trace(R_UE_tf))
        
        # Trace가 너무 작으면 경고 (공분산 행렬이므로 trace > 0 보장)
        if trace_tx < 1e-30 or trace_rx < 1e-30:
            print(f"  Warning: 매우 작은 trace 값 (trace_tx={trace_tx:.2e}, trace_rx={trace_rx:.2e})")
        
        R_BS_normalized = R_BS_tf / tf.cast(trace_tx, tf.complex64)
        R_UE_normalized = R_UE_tf / tf.cast(trace_rx, tf.complex64)
        
        # 정규화된 행렬로 EVD 수행
        U_tx, Lambda_tx = perform_eigen_decomposition_sorted(R_BS_normalized)
        U_rx, Lambda_rx = perform_eigen_decomposition_sorted(R_UE_normalized)
        
        # 고유값 검증 (평균 고유값 대비 상대 threshold 계산)
        lambda_tx_mean = float(tf.reduce_mean(Lambda_tx).numpy())
        lambda_rx_mean = float(tf.reduce_mean(Lambda_rx).numpy())
        threshold_tx = lambda_tx_mean * config.eigenvalue_threshold_ratio
        threshold_rx = lambda_rx_mean * config.eigenvalue_threshold_ratio
        
        val_tx = validate_eigendecomposition(U_tx, Lambda_tx, R_BS_normalized, threshold_tx, "U_tx")
        val_rx = validate_eigendecomposition(U_rx, Lambda_rx, R_UE_normalized, threshold_rx, "U_rx")
        
        # EVD 재구성 오차 및 CMD 출력 (Trace 정보 포함)
        print(f"  EVD 검증: R_tx Trace={trace_tx.numpy():.2e}, 재구성 오차={val_tx['reconstruction_error']:.2e}, CMD={val_tx['cmd']:.6f} (>{val_tx['threshold']:.0e}: {val_tx['num_above_threshold']}/{val_tx['num_eigenvalues']})")
        print(f"            R_rx Trace={trace_rx.numpy():.2e}, 재구성 오차={val_rx['reconstruction_error']:.2e}, CMD={val_rx['cmd']:.6f} (>{val_rx['threshold']:.0e}: {val_rx['num_above_threshold']}/{val_rx['num_eigenvalues']})")
        
        # Step 3: 커플링 행렬 계산
        # P1B Valid RXs 데이터 로딩 (채널 생성용)
        ray_data = config.load_p1b_ray_data(area_index, fc, RX_index)
        if ray_data is None:
            print(f"  Error: Ray 데이터 로딩 실패")
            continue
        
        # Convert ray data to TensorFlow tensors
        ray_aoa_rad = tf.convert_to_tensor(np.deg2rad(ray_data['phi_r_deg']))
        ray_aod_rad = tf.convert_to_tensor(np.deg2rad(ray_data['phi_t_deg']))
        ray_zoa_rad = tf.convert_to_tensor(np.deg2rad(ray_data['theta_r_deg']))
        ray_zod_rad = tf.convert_to_tensor(np.deg2rad(ray_data['theta_t_deg']))
        ray_power = tf.convert_to_tensor(ray_data['power'])
        ray_delay = tf.convert_to_tensor(ray_data['tau'])
        
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

        # 채널 생성기 초기화 (한 번만 수행)
        OFDM_ChGen = ChCoeGen(carrier_frequency, config.OFDM_SCS, ArrayTX, ArrayRX, False)
        
        # 최종 결과 저장 리스트 (사전 할당)
        all_static_ch_realizations_data = [None] * config.static_ch_realizations
        
        # 매번 새로운 랜덤성 주입: XPR, topology 등
        ray_xpr = 10**(tf.random.normal(shape=[batch_size,1,N_UE,1,ray_aoa_rad.shape[-1]], 
                                                mean=config.mean_xpr, stddev=config.stddev_xpr)/10)

        ray_pdap = Rays(delays=ray_delay, powers=ray_power, aoa=ray_aoa_rad, aod=ray_aod_rad,
                            zoa=ray_zoa_rad, zod=ray_zod_rad, xpr=ray_xpr)

        # 새로운 토폴로지 생성 (매번 다른 속도, 방향)
        speed = tf.abs(tf.random.normal(shape=[batch_size, N_UE, 1],
                                                mean=config.Topology_Statistics["velocities_mean"],
                                                stddev=config.Topology_Statistics["velocities_stddev"],
                                                dtype=tf.float32))

        angle = tf.random.uniform(shape=[batch_size, N_UE, 1],
                                            minval=0.0,
                                            maxval=2 * PI,
                                            dtype=tf.float32)

        velocities = tf.concat([speed * tf.cos(angle),    # vx: speed * cos(angle)
                                        speed * tf.sin(angle),    # vy: speed * sin(angle)
                                        tf.zeros_like(speed)],    # vz: 0
                                        axis=-1)

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

        # 커플링 행렬 추정 엔진 초기화 (trace 값 Python float로 변환하여 전달)
        coupling_engine = CouplingMatrix_Engine(config, OFDM_ChGen, topology, ray_pdap, U_tx, U_rx, 
                                                float(trace_tx.numpy()), float(trace_rx.numpy()))
        
        start_accumulation = time.time()
        
        # Static 루프 - 커플링 행렬 온라인 누적
        num_static_realizations = config.static_ch_realizations
        
        for static_idx in range(1, num_static_realizations + 1):
            # 채널 생성 및 커플링 행렬 누적 (0-based 인덱스로 전달)
            is_last = (static_idx == num_static_realizations)
            coupling_engine.process_static_realization(static_idx - 1, is_last_static=is_last)
            
            # 진행 표시 (매 64개마다)
            if static_idx % 64 == 0:
                progress_pct = static_idx / num_static_realizations * 100
                print(f"  Static {static_idx}/{num_static_realizations} ({progress_pct:.0f}%)", end=' ', flush=True)
        
        accumulation_time = time.time() - start_accumulation
        accumulated_samples = int(coupling_engine.count.numpy())
        
        # 수렴도 출력
        conv_str = f", 수렴: {coupling_engine.last_convergence_metric:.2e}" if coupling_engine.last_convergence_metric else ""
        print(f"\r  커플링 행렬 누적 완료 ({accumulated_samples}개 샘플, {accumulation_time:.1f}초{conv_str})")
        
        # 커플링 행렬 추출 및 저장
        coupling_data = coupling_engine.finalize_coupling_matrix(area_index, fc, RX_index, config.coupling_threshold_ratio)
        
        if coupling_data:
            # npz 파일 저장 (고유벡터 + 커플링 행렬 + 검증 결과)
            coupling_manager.save_rx_coupling_matrix(
                coupling_data, 
                U_tx.numpy(), U_rx.numpy(), 
                Lambda_tx.numpy(), Lambda_rx.numpy(),
                val_tx, val_rx
            )
        else:
            print(f"  Error: 커플링 행렬 추출 실패")
        
        # RX 처리 후 메모리 정리
        del coupling_engine, ray_data, U_tx, U_rx, Lambda_tx, Lambda_rx
        
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
            print("P1G 커플링 행렬 추정 진행률:")
            print(f"✓ [{rx_file_count:>{total_rx_width}}/{total_rx_files}] Area{area_index}_{fc}GHz_RX{RX_index} (커플링 행렬 저장 완료)")
            print(f"현재 완료율: {rx_file_count/total_rx_files*100:.1f}% ({rx_file_count}/{total_rx_files})")
            print(f"전체 경과 시간: {format_time(overall_elapsed)}")
            print(f"평균 처리 속도: {avg_time_per_rx:.1f}초/RX")
            print(f"예상 남은 시간: {format_time(estimated_remaining)}")
            print("-" * 80)
        
    # 분석 완료 메시지
    total_rx_processed = len(config.p1f_data_combinations)
    overall_time = time.time() - overall_start_time
    
    print(f"\n=== P1G 커플링 행렬 추정 완료 ===")
    print(f"총 {total_rx_processed}개 RX 처리 완료")
    print(f"Static 실현: {config.static_ch_realizations}개")
    print(f"총 소요 시간: {overall_time/60:.1f}분")
    print(f"저장 위치: {config.P1G_OUTPUT_DIR}")
    
    # 최종 메모리 정리
    import gc
    gc.collect()
    tf.keras.backend.clear_session()

if __name__ == "__main__":
    main()
