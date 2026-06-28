# ======================================================================
# P1D_Rays_to_AE_Separability_2509.py
# P1D: 통합형 채널 분리성 분석 도구
# 
# === 최상위 목적 ===
# MIMO 채널의 Kronecker/Weichselberger 모델 분리성 분석
# - 고유값/고유벡터 오차 기반 채널 모델 적합성 판단
# - 온라인 누적 공분산으로 메모리 효율적 대용량 데이터 처리
# 
# === 핵심 설계 원리 ===
# 1. 데이터 흐름: Ray Data → AE OFDM Channel → 온라인 누적 → 분리성 분석
# 2. 메모리 최적화: Static 채널별 실시간 누적, 원시 데이터 즉시 폐기
# 3. 분석 정확성: P1E 원본 분석 로직 완전 보존 및 통합
# 
# === 주요 구성 요소 ===
# - SeparabilityAnalyzer: P1E 기반 분리성 분석 (JIT 최적화)
# - ChannelAnalyzer: 채널 생성 + 온라인 누적 + 분석 orchestration
# - P1D_ResultManager: P1E 호환 CSV 결과 저장
# 
# === 입력/출력 ===
# 입력: P1B Valid RXs 결과 (Ray 추적 데이터)
# 출력: 채널 분리성 분석 CSV (epsilon_lambda, epsilon_U, 모델 선택 등)
# 
# === 핵심 기술 ===
# - 독립적 정적 채널 발생 (매번 새로운 초기 위상)
# - 온라인 공분산 누적 (_online_rx_cov_mat_update)
# - TensorFlow JIT 컴파일 기반 수치 계산 최적화
#
# === 주요 수정 이력 ===
# [250930] 크로네커 채널 모델 구현 오류 수정 (P1E 연동)
# 1. Column-Major vec(): LinearOperatorKronecker와 일치 (transpose + reshape)
# 2. Hermitian ECM: 복소수 α 사용 시 하삼각 conjugate 적용
# 3. 채널 생성: sqrtm(R_BS) transpose 제거 (Hermitian 속성)
# P1E 검증 결과: R_BS/R_UE < 1e-5, R_AE 4.4%, eps_d/eps_U 추가 개선 필요
#
# [251003] Block Welford 누적 방식 적용 (2510v3)
# 1. BlockWelfordAccumulator 클래스 도입: Sub-matrix 분할 관리
# 2. 수치 안정성 개선: (n_r×n_r) 작은 블록 단위 독립 업데이트
# 3. 코드 단순화: vec 기반 병합 로직 제거, 직접 업데이트 방식
# 4. R_BS ⊗ R_UE 구조 최적화: Column-wise 평균 + 블록 M2 관리
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

# TensorFlow 코어 임포트
import tensorflow as tf

# GPU 메모리 증가 허용 설정
gpus = tf.config.list_physical_devices('GPU')
for g in gpus:
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except RuntimeError as e:
        print(e)

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

# ===== SECTION 2: P1D_Config (P1B Valid RXs 기반 설정) =====
class P1D_Config:
    """P1B Valid RXs 데이터를 기반으로 실험 설정을 동적으로 구성 (Antenna Element OFDM 채널 생성)"""
    
    def __init__(self):
        # 파일 경로 및 명명 패턴 설정 (중앙 집중화)

        # 스크립트의 디렉토리를 기준으로 절대 경로 설정
        script_dir = os.path.dirname(os.path.abspath(__file__))

        self.P1B_INPUT_DIR = os.path.join(script_dir, "P1B_Valid_Results")
        self.P1B_FILE_PATTERN = "Area{area}_{freq}GHz_Rays_Valid_RXs.npz"  # P1B valid RXs format
        
        # P1D 분리성 분석 설정 (개별 채널 데이터 저장 없음, CSV 메트릭만 저장)
        self.P1D_ANALYSIS_OUTPUT_DIR = os.path.join(script_dir, "P1D_Separability_Results")
        
        # 콘솔 출력 제어 설정
        self.ENABLE_SCREEN_CLEAR = True      # 화면 클리어 기능 활성화/비활성화
        self.PROGRESS_CLEAR_INTERVAL = 10    # 화면 클리어 개수
        
        # 필터링 설정
        self.target_areas = [1]  # 처리할 area 목록 (None이면 전체)
        self.target_rxs = None   # 처리할 RX 목록 (None이면 전체)
        
        # P1B 데이터 디렉토리 스캔하여 실험 설정 자동 감지
        self.detect_p1b_data()
        
        # fc 리스트를 self.p1b_data_combinations에서 추출하여 설정
        if self.p1b_data_combinations:
            self.fcs = sorted(list(set([item[1] for item in self.p1b_data_combinations])))
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
        self.OFDM_FFT = 256                                 # OFDM 채널의 부반송파 수
        self.OFDM_SCS = 120e3                               # OFDM 채널의 부반송파 간격 (Hz)
        self.OFDM_BW = self.OFDM_FFT * self.OFDM_SCS        # OFDM 총 대역폭 (Hz)
        self.num_tx = 1
        self.num_rx = 1

        # 채널 생성 파라미터  
        self.static_ch_realizations = 4096 # 독립 정적 채널 개수
        self.doppler_time_realizations = 1 # 각 정적 채널의 도플러 샘플 수 (>1일 때 개별 파일 저장)
        self.doppler_sym_ofdm = 1 # 각 도플러 샘플당 OFDM 심볼 수
        self.doppler_time_max_sec = coherence_time*0 # 도플러 시간 샘플링 최대값 (초)
        
        # 분리성 임계값 설정 (tex 문서 기준)
        self.epsilon_lambda_threshold = 0.1             # 고유값 분리성 오차 임계값
        self.epsilon_U_threshold = 0.1                  # 고유벡터 분리성 오차 임계값
        self.channel_power_threshold = 1e-20            # 채널 파워 임계값
        
        # L40S GPU 병렬 처리 최적화 설정
        self.batch_size_rx = 1                          # RX 배치 크기 (단일 RX 처리)
        self.use_mixed_precision = False                # 복소수 연산에서 혼합 정밀도 비활성화
        self.gpu_memory_limit_gb = 45                   # L40S 메모리 한계
        
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

        # P1D 설정 정보 출력
        self.print_p1d_Config()
        
    def detect_p1b_data(self):
        """P1B Valid RXs 데이터를 스캔하여 실험 설정 자동 감지"""
        
        # P1B 출력 디렉토리에서 Valid RXs npz 파일들 스캔
        scan_pattern = f"{self.P1B_INPUT_DIR}/{self.P1B_FILE_PATTERN.format(area='*', freq='*')}"
        files = glob.glob(scan_pattern)
        
        # (area, freq, rx) 조합을 저장할 집합
        combinations = set()
        
        # 파일명에서 area_index, frequency 추출하고 npz 파일에서 Valid RX 정보 읽기
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
                
                # npz 파일을 열어서 Valid RX 인덱스들 확인
                try:
                    with np.load(file_path) as data:
                        if 'rx_indices' not in data:
                            print(f"Warning: {filename} does not contain 'rx_indices' key")
                            continue
                        
                        rx_indices = data['rx_indices']
                        # target_rxs 필터링 적용 (set intersection으로 효율화)
                        if self.target_rxs is not None:
                            rx_indices = [rx for rx in rx_indices if int(rx) in self.target_rxs]
                        
                        combinations.update((area_index, frequency, int(rx)) for rx in rx_indices)
                except Exception as e:
                    print(f"Warning: Failed to read {filename}: {e}")
        
        # 정렬하여 리스트로 저장
        self.p1b_data_combinations = sorted(list(combinations))
        
        print(f"P1B Valid RXs 데이터 자동 감지:")
        print(f"  - 감지된 조합 수: {len(self.p1b_data_combinations)}")
        
        # Area별로 그룹핑하여 출력
        area_groups = {}
        for combo in self.p1b_data_combinations:
            area, freq, rx = combo
            key = f"Area{area}_{freq}GHz"
            if key not in area_groups:
                area_groups[key] = []
            area_groups[key].append(rx)
        
        for area_freq, rx_list in sorted(area_groups.items()):
            print(f"    - {area_freq}: RX{min(rx_list)}-RX{max(rx_list)} ({len(rx_list)} Valid RXs)")
    
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
    

    def print_p1d_Config(self):
        """P1D 설정 정보 출력"""
        print(f"P1D 분리성 분석 설정:")
        print(f"  - Static Channel Realizations: {self.static_ch_realizations}")
        print(f"  - Doppler Time Realizations: {self.doppler_time_realizations}")
        print(f"  - Doppler Symbols of OFDM: {self.doppler_sym_ofdm}")
        print(f"  - OFDM FFT Size: {self.OFDM_FFT}")
        print(f"  - 분석 모드: 온라인 누적 분리성 분석")
        print(f"  - 예상 총 샘플 수 (per RX): {self.static_ch_realizations * self.doppler_time_realizations * self.doppler_sym_ofdm}")
        print(f"  - 안테나 구성: TX {self.n_t}개, RX {self.n_r}개")
        print(f"  - 분리성 임계값: λ={self.epsilon_lambda_threshold}, U={self.epsilon_U_threshold}")
        print(f"  - 결과 저장: {self.P1D_ANALYSIS_OUTPUT_DIR}")
        
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

# ===== SECTION 2.5: Block Welford 누적기 =====
class BlockWelfordAccumulator:
    """
    Sub-matrix 분할 Welford 알고리즘 누적기
    
    R_AE를 (n_t×n_t)개의 (n_r×n_r) 블록으로 분할하여 관리.
    수치적으로 안정적이며 R_BS ⊗ R_UE 구조에 최적화됨.
    
    상태 변수:
    - count: 누적된 샘플 수
    - mean_cols: [n_t, n_r, 1] H의 각 열(column)별 평균 벡터
    - M2_blocks: [n_t, n_t, n_r, n_r] M2_AE를 구성하는 sub-matrix 블록들
    
    수학적 배경:
    - R_AE = E[vec(H) @ vec(H)^H] ≈ R_BS ⊗ R_UE (Kronecker 모델)
    - M2_blocks[j,k] = Σ (h_j - μ_j) @ (h_k - μ_k)^H
    - 각 블록을 독립적으로 업데이트하여 수치 안정성 확보
    """
    
    def __init__(self, n_r, n_t, dtype=tf.complex64):
        """
        누적기 초기화
        
        Args:
            n_r: 수신 안테나 개수
            n_t: 송신 안테나 개수
            dtype: 데이터 타입 (tf.complex64 또는 tf.complex128)
        """
        self.n_r = n_r
        self.n_t = n_t
        self.dtype = dtype
        self._rdtype = tf.float32 if dtype==tf.complex64 else tf.float64
        
        # tf.Variable로 상태 유지 (JIT 컴파일 호환)
        self.count = tf.Variable(0, dtype=tf.int64, trainable=False, name="count")
        self.mean_H = tf.Variable(
            tf.zeros([n_r, n_t], dtype=dtype), 
            trainable=False, name="mean_H")
        self.M2_blocks = tf.Variable(
            tf.zeros([n_t, n_t, n_r, n_r], dtype=dtype), 
            trainable=False, name="M2_blocks")
    
    @tf.function(jit_compile=True)
    def update(self, H_batch):
        """
        채널 행렬 배치를 사용하여 통계량 업데이트 (Column-wise Welford)
        
        Args:
            H_batch: [batch_size, n_r, n_t] 형상의 채널 행렬 배치
        
        동작 원리:
        1. 새 배치(B)의 열별 평균과 M2 블록 계산
        2. Welford 병합 공식으로 누적 상태(A)와 병합:
           - count_new = count_A + count_B
           - mean_new = mean_A + (mean_B - mean_A) * (count_B / count_new)
           - M2_new = M2_A + M2_B + correction_term
        3. correction_term = delta @ delta^H * (count_A * count_B / count_new)
        """
        batch_size = tf.shape(H_batch)[0]
        
        # batch_size > 0인 경우만 업데이트 수행
        if batch_size > 0:
            # === 1. 새 배치 통계 계산 ===
            count_B = tf.cast(batch_size, self.dtype)
            
            # 배치 평균: [n_r, n_t]
            mean_H_B = tf.reduce_mean(H_batch, axis=0)
            
            # 중심화
            C_batch = H_batch - mean_H_B
            
            # M2 블록 계산: einsum으로 모든 sub-matrix 한번에
            # M2_blocks[j,k,i,r] = sum_b C[b,i,j] × conj(C[b,r,k])
            # Column-Major vec 기준 R_AE와 일관성 유지
            M2_blocks_B = tf.einsum('bij,brk->jkir', 
                                    C_batch, tf.math.conj(C_batch))
            
            # === 2. Welford 병합 ===
            count_A = tf.cast(self.count, self.dtype)
            count_new = count_A + count_B
            
            # 델타 계산: [n_r, n_t]
            delta_matrix = mean_H_B - self.mean_H
            
            # 평균 업데이트
            mean_H_new = self.mean_H + delta_matrix * (count_B / count_new)
            
            # M2 블록 업데이트 (보정항 포함)
            # correction_term[j,k,i,r] = ΔH[i,j] × conj(ΔH[r,k])
            # delta_matrix: [n_r, n_t]
            correction_term_blocks = tf.einsum('ij,rk->jkir',
                                               delta_matrix,
                                               tf.math.conj(delta_matrix))
            factor = (count_A * count_B) / count_new
            M2_blocks_new = self.M2_blocks + M2_blocks_B + correction_term_blocks * factor
            
            # === 3. 상태 변수 업데이트 ===
            self.count.assign(tf.cast(count_new, tf.int64))
            self.mean_H.assign(mean_H_new)
            self.M2_blocks.assign(M2_blocks_new)
    
    def finalize(self):
        """
        누적된 통계량으로부터 최종 공분산 행렬 계산
        
        Returns:
            dict: 공분산 행렬들
                - R_AE_sam: [1, n_r*n_t, n_r*n_t] 전체 공분산
                - R_BS_sam: [1, n_t, n_t] BS 공분산
                - R_UE_sam: [1, n_r, n_r] UE 공분산
        
        계산 공식:
        - R_AE = M2_AE / (N-1)
        - R_BS = trace(M2_blocks[j,k]) / ((N-1) * n_r) for each j,k
        - R_UE = sum(M2_blocks[j,j]) / ((N-1) * n_t)
        """
        N = self.count
        if N < 2:
            print("Warning: 샘플 수 부족 (N < 2)")
            nan_val = tf.constant(np.nan, self.dtype)
            return {
                'R_AE_sam': tf.fill([1, self.n_r*self.n_t, self.n_r*self.n_t], nan_val),
                'R_BS_sam': tf.fill([1, self.n_t, self.n_t], nan_val),
                'R_UE_sam': tf.fill([1, self.n_r, self.n_r], nan_val)
            }
        
        N_minus_1 = tf.cast(N - 1, self.dtype)
        
        # 1. R_AE_sam: 블록 행렬 → 전체 행렬
        # Step 1.1: M2_blocks[j,k,i,r] → [j,i,k,r] (Column-Major vec 순서 준비)
        M2_blocks_transposed = tf.transpose(self.M2_blocks, [0,2,1,3])
        
        # Step 1.2: [j,i,k,r] → [(j*n_r+i), (k*n_r+r)] (2D 행렬로 펼치기)
        M2_AE = tf.reshape(M2_blocks_transposed, 
                           [self.n_t * self.n_r, self.n_t * self.n_r])
        
        # Step 1.3: M2 → R (샘플 공분산)
        R_AE_sam = M2_AE / N_minus_1
        
        # 2. R_BS_sam: 각 블록의 trace
        # M2_blocks[k,j,i,i] → M2_BS[j,k]
        # 수학적 유도: M2_BS[j,k] = sum_i M2_blocks[k,j,i,i]
        M2_BS = tf.einsum('kjii->jk', self.M2_blocks)
        R_BS_sam = M2_BS / (N_minus_1 * tf.cast(self.n_r, self.dtype))
        
        # 3. R_UE_sam: 대각 블록들의 합
        # sum_j M2_blocks[j,j,:,:] → M2_UE[:,:]
        M2_UE = tf.einsum('jjir->ir', self.M2_blocks)
        R_UE_sam = M2_UE / (N_minus_1 * tf.cast(self.n_t, self.dtype))
        
        return {
            'R_AE_sam': tf.expand_dims(R_AE_sam, 0),
            'R_BS_sam': tf.expand_dims(R_BS_sam, 0),
            'R_UE_sam': tf.expand_dims(R_UE_sam, 0)
        }

# ===== SECTION 3: P1D 분리성 분석 엔진 =====
class SeparabilityAnalyzer:
    """P1D 분리성 분석 엔진"""
    
    def __init__(self, config):
        self.config = config
        self._jit_warmup_done = False
    
    @tf.function # EVD에 대해서는 JIT 컴파일 비활성화
    def _eigendecomposition_jit(self, R_AE_tf, R_BS_tf, R_UE_tf):
        """
        고유값 분해 및 내림차순 정렬
        
        반환: 정렬된 고유값/고유벡터만 반환 (idx 불필요)
        """
        # 고유값 분해
        lambda_AE_tf, U_AE_tf = tf.linalg.eigh(R_AE_tf)
        lambda_BS_tf, U_BS_tf = tf.linalg.eigh(R_BS_tf)
        lambda_UE_tf, U_UE_tf = tf.linalg.eigh(R_UE_tf)
        
        # 실수 고유값만 추출
        lambda_AE_tf = tf.math.real(lambda_AE_tf)
        lambda_BS_tf = tf.math.real(lambda_BS_tf)
        lambda_UE_tf = tf.math.real(lambda_UE_tf)
        
        # 내림차순 정렬 인덱스 계산
        idx_AE_tf = tf.argsort(lambda_AE_tf, direction='DESCENDING', axis=-1)
        idx_BS_tf = tf.argsort(lambda_BS_tf, direction='DESCENDING', axis=-1)
        idx_UE_tf = tf.argsort(lambda_UE_tf, direction='DESCENDING', axis=-1)
        
        # 내림차순 정렬된 고유값/고유벡터
        lambda_AE_desc_tf = tf.gather(lambda_AE_tf, idx_AE_tf, batch_dims=1)
        lambda_BS_desc_tf = tf.gather(lambda_BS_tf, idx_BS_tf, batch_dims=1)
        lambda_UE_desc_tf = tf.gather(lambda_UE_tf, idx_UE_tf, batch_dims=1)
        
        U_AE_desc_tf = tf.gather(U_AE_tf, idx_AE_tf, batch_dims=1, axis=2)
        U_BS_desc_tf = tf.gather(U_BS_tf, idx_BS_tf, batch_dims=1, axis=2)
        U_UE_desc_tf = tf.gather(U_UE_tf, idx_UE_tf, batch_dims=1, axis=2)
        
        return lambda_AE_desc_tf, lambda_BS_desc_tf, lambda_UE_desc_tf, U_AE_desc_tf, U_BS_desc_tf, U_UE_desc_tf
    
    @tf.function(jit_compile=True)
    def _compute_otimes_desc_eigval_eigvec(self, lambda_A, lambda_B, U_A, U_B):
        """크로네커 곱셈 및 내림차순 정렬 고유값 및 고유벡터 계산
        
        반환:
        - desc_sorted: 고유값 크기 기준 내림차순 정렬
        """
        # 고유값 크로네커 곱 (정렬 인덱스 추적용)
        lambda_outer = tf.tensordot(lambda_A, lambda_B, axes=0)
        lambda_kron_ordered = tf.reshape(lambda_outer, [-1])
        
        # 고유벡터 크로네커 곱 (static method 사용)
        U_kron_ordered = kron_mat_py_tf(U_A, U_B)
        
        # 고유값 크기 기준 내림차순 정렬
        lambda_desc_sorted = tf.sort(lambda_kron_ordered, direction='DESCENDING')
        sort_indices = tf.argsort(lambda_kron_ordered, direction='DESCENDING')
        U_desc_sorted = tf.gather(U_kron_ordered, sort_indices, axis=1)
        
        return lambda_desc_sorted, U_desc_sorted
    
    # P1E 원본에는 _separability_metrics_jit 같은 별도 JIT 메서드가 없음
    # compute_separability_metrics 안에서 직접 TensorFlow 연산 수행
    
    def warmup_jit_functions(self):
        """JIT 함수들을 미리 컴파일"""
        if self._jit_warmup_done:
            return
            
        print("JIT XLA 컴파일 워밍업 중...", end=' ', flush=True)
        
        n_r = self.config.n_r
        n_t = self.config.n_t
        n_sym = self.config.doppler_sym_ofdm
        n_fft = self.config.OFDM_FFT
        batch_size_rx = self.config.batch_size_rx
        
        try:
            # === 1. BlockWelfordAccumulator 워밍업 ===
            dummy_accumulator = BlockWelfordAccumulator(n_r, n_t)
            dummy_H_batch = tf.zeros([n_sym * n_fft, n_r, n_t], dtype=tf.complex64)
            dummy_accumulator.update(dummy_H_batch)
            del dummy_accumulator
            
            # === 2. 고유값 분해 워밍업 ===
            dummy_R_AE = tf.eye(n_r * n_t, batch_shape=[batch_size_rx], dtype=tf.complex64)
            dummy_R_b = tf.eye(n_t, batch_shape=[batch_size_rx], dtype=tf.complex64)  
            dummy_R_u = tf.eye(n_r, batch_shape=[batch_size_rx], dtype=tf.complex64)
            self._eigendecomposition_jit(dummy_R_AE, dummy_R_b, dummy_R_u)
            
            # 3. 크로네커 근사 워밍업 (크로네커 곱 + desc_sorted 정렬)
            dummy_lambda_u = tf.ones([n_r], dtype=tf.float32)
            dummy_lambda_b = tf.ones([n_t], dtype=tf.float32) 
            dummy_U_u = tf.eye(n_r, dtype=tf.complex64)
            dummy_U_b = tf.eye(n_t, dtype=tf.complex64)
            self._compute_otimes_desc_eigval_eigvec(dummy_lambda_b, dummy_lambda_u, dummy_U_b, dummy_U_u)
            
            # 4. MATLAB 표준 static method 워밍업 (실제 사용 shape 기준)
            
            # vec_mat_py_tf: [n_sym*n_fft, n_r, n_t] 형태만 실제 사용
            actual_samples = n_sym * n_fft  # 실제 사용되는 샘플 수
            dummy_H_actual = tf.zeros([actual_samples, n_r, n_t], dtype=tf.complex64)
            vec_mat_py_tf(dummy_H_actual)  # vec 워밍업 (실제 shape)
            
            # kron_mat_py_tf: 실제 사용되는 3가지 shape 조합 모두 워밍업
            dummy_A_large = tf.eye(n_t, dtype=tf.complex64)      # [1024, 1024]
            dummy_B_small = tf.eye(n_r, dtype=tf.complex64)      # [16, 16] 
            dummy_A_vec = tf.ones([n_t, 1], dtype=tf.complex64)  # [1024, 1]
            dummy_B_vec = tf.ones([1, n_r], dtype=tf.complex64)  # [1, 16]
            
            kron_mat_py_tf(dummy_A_large, dummy_B_small)    # [1024,1024] ⊗ [16,16] (가장 빈번)
            kron_mat_py_tf(dummy_B_small, dummy_A_large)    # [16,16] ⊗ [1024,1024] (순서 바뀐 버전)
            kron_mat_py_tf(dummy_A_vec, dummy_B_vec)        # [1024,1] ⊗ [1,16] (벡터 크로네커 곱)

            print("완료")
            self._jit_warmup_done = True
            
        except Exception as e:
            print(f"JIT 워밍업 실패: {e}")
            print("JIT 컴파일 없이 계속 진행...")
    
    # P1D는 온라인 누적 방식을 사용하므로 compute_rx_covariance_matrices는 불필요
    # _online_rx_cov_mat_update를 직접 사용하여 실시간 누적
    
    def compute_eigen_data(self, covariance_data):
        """
        공분산 데이터로부터 eigen_data 계산
        
        입력: covariance_data (R_AE, R_BS, R_UE 포함)
        출력: eigen_data (정렬된 고유값/고유벡터 포함)
        """
        R_AE_tf = covariance_data['R_AE_tf']           # [1, n_r*n_t, n_r*n_t] TensorFlow 텐서
        R_BS_tf = covariance_data['R_BS_tf']       # [1, n_t, n_t] TensorFlow 텐서
        R_UE_tf = covariance_data['R_UE_tf']       # [1, n_r, n_r] TensorFlow 텐서
        valid_rx_mask = covariance_data['valid_rx_mask']  # [1] 유효성 마스크
        
        n_r, n_t = covariance_data['n_r'], covariance_data['n_t']
        
        # 추가 유효성 검증 및 실패 사유 분석 (단일 RX)
        is_basic_valid = valid_rx_mask[0]  # 단일 RX
        
        if not is_basic_valid:
            return {'status': 'failed', 'reason': "실패(R_AE_trace)"}
                
        # NaN/Inf 검사 (TensorFlow에서 직접, 복소수 지원)
        # 복소수 공분산 행렬들의 실수부와 허수부를 각각 체크
        has_nan = (tf.reduce_any(tf.math.is_nan(tf.math.real(R_AE_tf))) or
                   tf.reduce_any(tf.math.is_nan(tf.math.imag(R_AE_tf))) or
                   tf.reduce_any(tf.math.is_nan(tf.math.real(R_BS_tf))) or
                   tf.reduce_any(tf.math.is_nan(tf.math.imag(R_BS_tf))) or
                   tf.reduce_any(tf.math.is_nan(tf.math.real(R_UE_tf))) or
                   tf.reduce_any(tf.math.is_nan(tf.math.imag(R_UE_tf)))).numpy()
        
        has_inf = (tf.reduce_any(tf.math.is_inf(tf.math.real(R_AE_tf))) or
                   tf.reduce_any(tf.math.is_inf(tf.math.imag(R_AE_tf))) or
                   tf.reduce_any(tf.math.is_inf(tf.math.real(R_BS_tf))) or
                   tf.reduce_any(tf.math.is_inf(tf.math.imag(R_BS_tf))) or
                   tf.reduce_any(tf.math.is_inf(tf.math.real(R_UE_tf))) or
                   tf.reduce_any(tf.math.is_inf(tf.math.imag(R_UE_tf)))).numpy()
        
        if has_nan:
            return {'status': 'failed', 'reason': "실패(NaN)"}
        elif has_inf:
            return {'status': 'failed', 'reason': "실패(Inf)"}
        
        # 호환성을 위해 리스트 형태 유지
        enhanced_valid_mask = [True]
        
        R_AE_sam_trace_tf = tf.math.real(tf.linalg.trace(R_AE_tf[0]))
        R_BS_sam_trace_tf = tf.math.real(tf.linalg.trace(R_BS_tf[0]))
        R_UE_sam_trace_tf = tf.math.real(tf.linalg.trace(R_UE_tf[0]))
        
        # 트레이스 기준 스케일링 정규화 (complex64로 캐스팅)
        scale_factor = tf.cast((R_BS_sam_trace_tf * R_UE_sam_trace_tf) / R_AE_sam_trace_tf, tf.complex64)
        R_AE_tf = R_AE_tf * scale_factor
                
        # JIT 컴파일된 고유값 분해 실행
        try:
            lambda_AE_desc_tf, lambda_BS_desc_tf, lambda_UE_desc_tf, \
            U_AE_desc_tf, U_BS_desc_tf, U_UE_desc_tf = \
                self._eigendecomposition_jit(R_AE_tf, R_BS_tf, R_UE_tf)
        except Exception as e:
            print(f"고유값 분해 실패: {e}")
            return None
        
        return {
            'lambda_AE_desc_tf': lambda_AE_desc_tf,    # [1, n_r*n_t] AE 내림차순 고유값
            'lambda_BS_desc_tf': lambda_BS_desc_tf,    # [1, n_t] BS 내림차순 고유값
            'lambda_UE_desc_tf': lambda_UE_desc_tf,    # [1, n_r] UE 내림차순 고유값
            'U_AE_desc_tf': U_AE_desc_tf,              # [1, n_r*n_t, n_r*n_t] AE 내림차순 고유벡터
            'U_BS_desc_tf': U_BS_desc_tf,              # [1, n_t, n_t] BS 내림차순 고유벡터
            'U_UE_desc_tf': U_UE_desc_tf,              # [1, n_r, n_r] UE 내림차순 고유벡터
            'R_AE_tf': R_AE_tf,                        # [1, n_r*n_t, n_r*n_t] 원본 R_AE 행렬
            'R_BS_tf': R_BS_tf,                        # [1, n_t, n_t] 원본 R_BS 행렬
            'R_UE_tf': R_UE_tf,                        # [1, n_r, n_r] 원본 R_UE 행렬
            'valid_rx_mask': enhanced_valid_mask,      # 향상된 유효성 마스크
            'n_r': n_r,
            'n_t': n_t
        }
    
    def compute_kronecker_data(self, eigen_data):
        """
        eigen_data로부터 kronecker_data 계산
        
        입력: eigen_data (정렬된 고유값/고유벡터 포함)
        출력: kronecker_data (크로네커 근사 고유값/고유벡터 포함)
        """
        if eigen_data is None:
            return None
            
        n_r_n_t = eigen_data['lambda_AE_desc_tf'].shape[1]
        valid_rx_mask = eigen_data['valid_rx_mask']
        
        if valid_rx_mask[0]:  # 단일 RX
            # === [250930 수정] 크로네커 근사: BS ⊗ UE (Column-Major vec) ===
            # R_AE ≈ (U_BS ⊗ U_UE)(Λ_BS ⊗ Λ_UE)(U_BS ⊗ U_UE)^H
            # Column-Major vec 기준: BS ⊗ UE 순서
            # 내림차순 정렬된 고유값/고유벡터를 직접 사용
            
            # 통합 크로네커 근사 실행 (크로네커 곱 + 내림차순 정렬 한번에 처리)
            # BS ⊗ UE 순서 (Column-Major vec 기준)
            lambda_KM_desc, U_KM_desc = self._compute_otimes_desc_eigval_eigvec(
                eigen_data['lambda_BS_desc_tf'][0], 
                eigen_data['lambda_UE_desc_tf'][0], 
                tf.math.conj(eigen_data['U_BS_desc_tf'][0]), # U_AE = U_BS^* @ U_UE
                eigen_data['U_UE_desc_tf'][0])
            
            # 배치 차원 추가 (이미 desc_sorted 결과)
            lambda_KM_desc_tf = tf.expand_dims(lambda_KM_desc, axis=0)      # [1, n_r*n_t] 내림차순 정렬
            U_KM_desc_tf = tf.expand_dims(U_KM_desc, axis=0)                # [1, n_r*n_t, n_r*n_t] 내림차순 정렬
        else:
            # 실패한 경우 NaN으로 채운 TensorFlow 텐서 생성
            nan_float = tf.constant(np.nan, dtype=tf.float32)
            shape_lambda = tf.shape(eigen_data['lambda_AE_desc_tf'])
            shape_U = tf.shape(eigen_data['U_AE_desc_tf'])
            
            lambda_KM_desc_tf = tf.fill(shape_lambda, nan_float)
            U_KM_desc_tf = tf.fill(shape_U, nan_float)
        
        return {
            'lambda_KM_desc_tf': lambda_KM_desc_tf, # [1, n_r*n_t] 내림차순 정렬된 고유값
            'U_KM_desc_tf': U_KM_desc_tf,  # [1, n_r*n_t, n_r*n_t] 내림차순 정렬된 고유벡터
            'valid_rx_mask': valid_rx_mask
        }
    
    def compute_separability_metrics(self, eigen_data, kron_data):
        """분리성 메트릭 계산 (P1E 원본 기반)"""
        if eigen_data is None or kron_data is None:
            return None
        
        # P1D는 단일 RX 처리이므로 valid_rx_mask는 항상 True
        valid_rx_mask = [True]
        
        # 모든 메트릭을 NaN으로 초기화 (단일 RX)
        epsilon_d_desc_np = [np.nan]     # eps_d: lambda_AE_desc vs lambda_KM_desc
        epsilon_U_desc_np = [np.nan]     # eps_U: U_AE_desc vs U_KM_desc
        epsilon_u_1_np = [np.nan]        # eps_u_1: 최상위 고유벡터 분리성
        eps_R_fro_np = [np.nan]          # R_AE - R_KM 프로베니우스 노름 오차
        Inner_MaxAll_np = [np.nan]       # 고유벡터 내적 전체 최대값
        Inner_MinRowMax_np = [np.nan]    # 고유벡터 내적 행별 최대값의 최소값
        Inner_MinSumSq_np = [np.nan]     # 고유벡터 내적 제곱합 최소값 (직교성)
        AE_rank_np = [np.nan]
        BS_rank_np = [np.nan]
        UE_rank_np = [np.nan]
        KM_rank_np = [np.nan]
        
        # 유효한 RX에 대해서만 실제 메트릭 계산 (단일 RX)
        if valid_rx_mask[0]:
            # 내림차순 정렬된 고유값/고유벡터 사용
            lambda_AE_desc_tf = eigen_data['lambda_AE_desc_tf'][0] # [n_r*n_t] AE 내림차순 고유값
            lambda_KM_desc_tf = kron_data['lambda_KM_desc_tf'][0] # [n_r*n_t] KM 내림차순 고유값
            U_AE_desc_tf = eigen_data['U_AE_desc_tf'][0] # [n_r*n_t, n_r*n_t] AE 내림차순 고유벡터
            U_KM_desc_tf = kron_data['U_KM_desc_tf'][0] # [n_r*n_t, n_r*n_t] KM 내림차순 고유벡터

            # === eps_d 계산 (고유값 오차) ===
            eps_d_desc_tf = (tf.linalg.norm(lambda_AE_desc_tf - lambda_KM_desc_tf) / 
                            tf.linalg.norm(lambda_AE_desc_tf))
            epsilon_d_desc_np[0] = float(eps_d_desc_tf.numpy())

            # === eps_U 계산 (고유벡터 오차) - 간소화 ===
            U_inner_prod_all = tf.abs(tf.linalg.matmul(U_AE_desc_tf, U_KM_desc_tf, adjoint_a=True))
            
            # 핵심 메트릭 계산
            max_align = tf.reduce_max(U_inner_prod_all)
            min_align = tf.reduce_min(tf.reduce_max(U_inner_prod_all, axis=1))
            unitary_check = tf.reduce_min(tf.reduce_sum(tf.square(U_inner_prod_all), axis=1))
            
            # CSV 출력용 저장
            Inner_MaxAll_np[0] = float(max_align.numpy())
            Inner_MinRowMax_np[0] = float(min_align.numpy())
            Inner_MinSumSq_np[0] = float(unitary_check.numpy())
            
            # P1D 스타일 eps_U (고유값 가중평균 - 이론값 기준)
            U_inner_prod_max_per_row = tf.reduce_max(U_inner_prod_all, axis=1)
            weights_by_eigvals = lambda_KM_desc_tf / tf.reduce_sum(lambda_KM_desc_tf)
            weighted_inner_prod = tf.reduce_sum(weights_by_eigvals * U_inner_prod_max_per_row)
            eps_U_desc_tf = 1.0 - weighted_inner_prod
            epsilon_U_desc_np[0] = float(eps_U_desc_tf.numpy())
            
            # 통합 결과 출력 (한 줄)
            print(f"     MaxAll={max_align.numpy():.3f}, MinRowMax={min_align.numpy():.3f}, MinSumSq={unitary_check.numpy():.3f}, eps_U={eps_U_desc_tf.numpy():.4f}")

            # === eps_u_1 계산 (최상위 고유벡터) - 간소화 ===
            u_AE_1 = U_AE_desc_tf[:, 0]
            u_BS_1 = eigen_data['U_BS_desc_tf'][0][:, 0]
            u_UE_1 = eigen_data['U_UE_desc_tf'][0][:, 0]
            
            # BS⊗UE 크로네커 곱과 내적 계산 (벡터 → 행렬 → 벡터)
            u_BS_1_mat = tf.expand_dims(u_BS_1, axis=1)  # [n_t, 1]
            u_UE_1_mat = tf.expand_dims(u_UE_1, axis=0)  # [1, n_r]  
            u_kron_1_bu = tf.reshape(kron_mat_py_tf(u_BS_1_mat, u_UE_1_mat), [-1])
            inner_prod_bu = tf.abs(tf.reduce_sum(tf.math.conj(u_AE_1) * u_kron_1_bu))
            epsilon_u_1_np[0] = (1.0 - inner_prod_bu).numpy()

            # === R_AE vs R_KM 프로베니우스 노름 ===
            R_AE_tf = eigen_data['R_AE_tf'][0]
            R_b_tf = eigen_data['R_BS_tf'][0]  
            R_u_tf = eigen_data['R_UE_tf'][0]
            R_KM_tf = kron_mat_py_tf(R_b_tf, R_u_tf)
            
            eps_R_fro_tf = tf.linalg.norm(R_AE_tf - R_KM_tf) / (tf.linalg.norm(R_AE_tf) + 1e-30)
            eps_R_fro_np[0] = float(tf.math.real(eps_R_fro_tf).numpy())
            
            # === 랭크 계산 ===
            tol = 1e-6
            AE_rank_np[0] = int(tf.reduce_sum(tf.cast(tf.abs(lambda_AE_desc_tf) > tol, tf.int32)).numpy())
            KM_rank_np[0] = int(tf.reduce_sum(tf.cast(tf.abs(lambda_KM_desc_tf) > tol, tf.int32)).numpy())
            BS_rank_np[0] = int(tf.reduce_sum(tf.cast(tf.abs(eigen_data['lambda_BS_desc_tf'][0]) > tol, tf.int32)).numpy())
            UE_rank_np[0] = int(tf.reduce_sum(tf.cast(tf.abs(eigen_data['lambda_UE_desc_tf'][0]) > tol, tf.int32)).numpy())
        
        return {
            'epsilon_d_desc': epsilon_d_desc_np[0],    # eps_d: lambda_AE_desc vs lambda_KM_desc
            'epsilon_U_desc': epsilon_U_desc_np[0],    # eps_U: U_AE_desc vs U_KM_desc
            'epsilon_u_1': epsilon_u_1_np[0],          # eps_u_1: 최상위 고유벡터
            'eps_R_fro': eps_R_fro_np[0],              # R_AE - R_KM 프로베니우스 노름 오차
            'Inner_MaxAll': Inner_MaxAll_np[0],        # 고유벡터 내적 전체 최대값
            'Inner_MinRowMax': Inner_MinRowMax_np[0],  # 고유벡터 내적 행별 최대값의 최소값
            'Inner_MinSumSq': Inner_MinSumSq_np[0],    # 고유벡터 내적 제곱합 최소값
            'AE_rank': AE_rank_np[0],
            'BS_rank': BS_rank_np[0],
            'UE_rank': UE_rank_np[0],
            'KM_rank': KM_rank_np[0],
            'valid_rx_mask': valid_rx_mask
        }
    
    def determine_channel_model(self, epsilon_d, epsilon_U, n_r, n_t):
        """
        채널 모델 선택
        
        Args:
            epsilon_d: 고유값 오차 (eps_d_desc 사용)
            epsilon_U: 고유벡터 오차 (eps_U_desc 사용)
        """
        if epsilon_d < self.config.epsilon_lambda_threshold and epsilon_U < self.config.epsilon_U_threshold:
            return {
                'recommended_model': 'Kronecker',
                'separability_status': 'Kronecker Separable',
                'num_req_params': n_t**2 + n_r**2
            }
        elif epsilon_d >= self.config.epsilon_lambda_threshold and epsilon_U < self.config.epsilon_U_threshold:
            return {
                'recommended_model': 'Weichselberger',
                'separability_status': 'Eigenvector Only Separable',
                'num_req_params': n_t**2 + n_r**2 + (n_t * n_r) // 2
            }
        else:
            return {
                'recommended_model': 'Non-Separable',
                'separability_status': 'Non-Separable',
                'num_req_params': (n_t * n_r)**2
            }

# ===== P1D 채널 분석기 (Block Welford 방식) =====
class ChannelAnalyzer:
    """P1D 채널 생성 + Block Welford 누적 + 분리성 분석"""
    
    def __init__(self, config, OFDM_ChGen, topology, ray_pdap):
        self.config = config
        self.OFDM_ChGen = OFDM_ChGen
        self.topology = topology
        self.ray_pdap = ray_pdap
        
        # 분리성 분석 엔진
        self.analyzer = SeparabilityAnalyzer(config)
        
        # === Block Welford 누적기 (새로운 방식) ===
        self.accumulator = BlockWelfordAccumulator(config.n_r, config.n_t)
        
        # 수렴성 정보 (진행 상황 출력용)
        self.convergence_info = None
        
    def process_static_realization(self, static_idx):
        """단일 Static 실현 생성 및 즉시 누적"""
        
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
            
            # === 5. Block Welford 업데이트 (새로운 방식) ===
            # [1, n_sym, n_fft, n_r, n_t] → [n_sym*n_fft, n_r, n_t]
            batch_size = tf.shape(channel_sample)[0]
            n_sym = tf.shape(channel_sample)[1]
            n_fft = tf.shape(channel_sample)[2]
            n_r = tf.shape(channel_sample)[3]
            n_t = tf.shape(channel_sample)[4]
            
            H_batch = tf.reshape(channel_sample, 
                                [batch_size * n_sym * n_fft, n_r, n_t])
            
            # BlockWelfordAccumulator에 직접 업데이트
            self.accumulator.update(H_batch)
            
            # 메모리 해제
            del ofdm_ch_doppler, channel_sample, H_batch
        
        # 메모리 해제 (Static별)
        del h_delay_bin_static, aoa_delay_bin, zoa_delay_bin
        
        # 2^k 기준 진행 상황 출력 (k>=1) 및 시간 정보 포함
        current_static = static_idx + 1  # 1-based index
        
        # 2^k 체크: 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096...
        should_print = False
        if current_static == 1:
            should_print = True  # 첫 번째는 항상 출력
        elif current_static == self.config.static_ch_realizations:
            should_print = True  # 마지막은 항상 출력
        else:
            # 2^k 체크 (k>=1)
            import math
            if current_static > 1 and (current_static & (current_static - 1)) == 0:
                # current_static이 2의 거듭제곱인지 확인
                k = int(math.log2(current_static))
                if k >= 1:
                    should_print = True
        
        if should_print:
            # 시작 시간이 없으면 현재 시간으로 초기화
            if not hasattr(self, '_static_start_time'):
                import time
                self._static_start_time = time.time()
            
            # 경과 시간 계산
            import time
            elapsed_time = time.time() - self._static_start_time
            avg_time_per_static = elapsed_time / current_static
            estimated_total = avg_time_per_static * self.config.static_ch_realizations
            estimated_remaining = estimated_total - elapsed_time
            
            # 시간 포맷팅
            def format_time(seconds):
                if seconds < 60:
                    return f"{seconds:.1f}초"
                elif seconds < 3600:
                    return f"{int(seconds//60)}분{int(seconds%60):02d}초"
                else:
                    hours = int(seconds // 3600)
                    minutes = int((seconds % 3600) // 60)
                    return f"{hours}시간{minutes:02d}분"
            
            progress_info = f"Static {current_static}/{self.config.static_ch_realizations}"
            progress_info += f" ({current_static/self.config.static_ch_realizations*100:.1f}%)"
            progress_info += f" | 경과: {format_time(elapsed_time)}"
            progress_info += f", 예상잔여: {format_time(estimated_remaining)}"
            
            # === 수렴성 정보 (Block Welford 방식으로 수정) ===
            if current_static > 1:
                try:
                    # 중간 finalize (현재 상태의 R 행렬들 계산)
                    temp_covs = self.accumulator.finalize()
                    R_AE_sam = temp_covs['R_AE_sam'][0]
                    R_BS_sam = temp_covs['R_BS_sam'][0]
                    R_UE_sam = temp_covs['R_UE_sam'][0]
                    
                    # R_AE - R_KM 프로베니우스 노름
                    R_KM_sam = kron_mat_py_tf(R_BS_sam, R_UE_sam)
                    diff_matrix = R_AE_sam - R_KM_sam
                    fro_diff = tf.sqrt(tf.reduce_sum(tf.square(tf.abs(diff_matrix))))
                    fro_AE = tf.sqrt(tf.reduce_sum(tf.square(tf.abs(R_AE_sam))))
                    eps_R_fro = float((fro_diff / (fro_AE + 1e-12)).numpy())
                    
                    # R_UE ⊗ R_BS (순서 바뀐 버전)
                    R_KM_sam2 = kron_mat_py_tf(R_UE_sam, R_BS_sam)
                    diff_matrix2 = R_AE_sam - R_KM_sam2
                    fro_diff2 = tf.sqrt(tf.reduce_sum(tf.square(tf.abs(diff_matrix2))))
                    eps_R_fro2 = float((fro_diff2 / (fro_AE + 1e-12)).numpy())
                    
                    self.convergence_info = f"eps_R:{eps_R_fro:.1e}, eps_R_2:{eps_R_fro2:.1e}"
                    progress_info += f"\n    수렴상태: {self.convergence_info}"
                        
                except Exception as e:
                    pass
            
            # 개행 포함한 명확한 출력
            print(f"\n{progress_info}")
        else:
            # 2^k가 아닌 경우는 간단한 점 출력으로 진행 표시
            if current_static % 32 == 0:  # 32의 배수마다 점 출력
                print(".", end='', flush=True)
    
    def finalize_analysis(self, area_index, fc, RX_index):
        """누적 완료 후 최종 분리성 분석"""
        
        # === 1. BlockWelfordAccumulator에서 최종 공분산 행렬 계산 ===
        final_covs = self.accumulator.finalize()
        
        N_total = int(self.accumulator.count.numpy())
        
        if N_total < 2:
            print("Error: 누적된 샘플이 부족하여 분석할 수 없습니다.")
            return None
        
        print(f"\n=== RX{RX_index} 분리성 분석 시작 (OFDM 샘플 {N_total}개 누적) ===")
        
        # === 2. P1E 원본과 동일한 covariance_data 구조 생성 ===
        covariance_data = {
            'R_AE_tf': final_covs['R_AE_sam'],
            'R_BS_tf': final_covs['R_BS_sam'], 
            'R_UE_tf': final_covs['R_UE_sam'],
            'R_AE_trace': [tf.math.real(
                tf.linalg.trace(final_covs['R_AE_sam'][0])).numpy()],
            'valid_rx_mask': [True],
            'n_ofdm_sam_final': N_total,
            'n_r': self.config.n_r,
            'n_t': self.config.n_t
        }
        
        print("    처리:", end=' ', flush=True)
        
        # === 3-5. 고유값 분해 → 크로네커 근사 → 메트릭 계산 (기존 동일) ===
        
        # 3. 고유값 분해
        print(" 고유값분해", end='', flush=True)
        start_eigen = time.time()
        eigen_data_tf = self.analyzer.compute_eigen_data(covariance_data)
        eigen_time = time.time() - start_eigen
        
        if eigen_data_tf is None or (isinstance(eigen_data_tf, dict) and eigen_data_tf.get('status') == 'failed'):
            print(f"({eigen_time:.1f}초) -> 고유값 분해 실패")
            return None
        print(f"({eigen_time:.1f}초)", end='', flush=True)
        
        # 4. 크로네커 근사 (P1E와 동일)
        print(" 크로네커근사", end='', flush=True)
        start_kronecker = time.time()
        kron_data_tf = self.analyzer.compute_kronecker_data(eigen_data_tf)
        kronecker_time = time.time() - start_kronecker
        print(f"({kronecker_time:.1f}초)", end='', flush=True)
        
        if kron_data_tf is None:
            print(" -> 크로네커 근사 실패")
            return None
        
        # 5. 분리성 메트릭 계산 (P1E와 동일)
        print(" 메트릭계산", end='', flush=True)
        start_metrics = time.time()  
        metrics_data_np = self.analyzer.compute_separability_metrics(eigen_data_tf, kron_data_tf)
        metrics_time = time.time() - start_metrics
        print(f"({metrics_time:.1f}초)", end='', flush=True)
        
        if metrics_data_np is None:
            print(" -> 메트릭 계산 실패")
            return None
        
        # === 6. 결과 생성 ===
        valid_rx_mask = covariance_data['valid_rx_mask']
        
        if not valid_rx_mask[0]:
            print(" -> RX 유효성 검사 실패")
            return None
        
        result = {
            'area_idx': area_index,
            'freq_ghz': fc,
            'rx_idx': RX_index,
            'data_source': 'P1D_BlockWelford',  # 새로운 방식 표시
            'R_AE_trace': covariance_data['R_AE_trace'][0],
            'epsilon_d_desc': metrics_data_np['epsilon_d_desc'],       # eps_d (기본)
            'epsilon_d_kron': metrics_data_np['epsilon_d_desc'],       # eps_d (desc_sorted)
            'epsilon_U_desc': metrics_data_np['epsilon_U_desc'],       # eps_U (기본)
            'epsilon_u_1': metrics_data_np['epsilon_u_1'],
            'eps_R_fro': metrics_data_np['eps_R_fro'],
            'AE_rank': metrics_data_np['AE_rank'],
            'BS_rank': metrics_data_np['BS_rank'],
            'UE_rank': metrics_data_np['UE_rank'],
            'KM_rank': metrics_data_np['KM_rank'],
            'n_r': self.config.n_r,
            'n_t': self.config.n_t,
            'n_ofdm_sam_final': N_total,
            'static_realizations': self.config.static_ch_realizations,
            **self.analyzer.determine_channel_model(
                metrics_data_np['epsilon_d_desc'], metrics_data_np['epsilon_U_desc'], 
                self.config.n_r, self.config.n_t)
        }
        
        return result

# ===== SECTION 4: P1D 분석 결과 저장 관리 =====
class P1D_ResultManager:
    """P1D 분석 결과 저장 및 관리"""
    
    def __init__(self, config):
        self.config = config
        os.makedirs(self.config.P1D_ANALYSIS_OUTPUT_DIR, exist_ok=True)
        
        # RX별 실시간 CSV 업데이트를 위한 설정
        self.progress_csv_path = os.path.join(config.P1D_ANALYSIS_OUTPUT_DIR, "P1D_Progress_RealTime.csv")
    
    def _compute_summary_statistics(self, valid_results):
        """요약 통계 계산"""
        metrics = ['epsilon_d_desc', 'epsilon_d_kron', 'epsilon_U_desc', 'R_AE_trace']
        
        summary = {}
        
        # 기본 분리성 메트릭 및 채널 이득 통계
        for metric in metrics:
            values = [r[metric] for r in valid_results if not np.isnan(r[metric])]
            if values:
                summary[f'{metric}_mean'] = np.mean(values)
                summary[f'{metric}_std'] = np.std(values)
                summary[f'{metric}_min'] = np.min(values)
                summary[f'{metric}_max'] = np.max(values)
                summary[f'{metric}_median'] = np.median(values)
            else:
                summary.update({f'{metric}_{stat}': np.nan 
                              for stat in ['mean', 'std', 'min', 'max', 'median']})
        
        summary['n_valid_rx'] = len(valid_results)
        
        # 모델 선택 통계
        if any('recommended_model' in r for r in valid_results):
            model_counts = {}
            model_results = [r for r in valid_results if 'recommended_model' in r]
            
            for result in model_results:
                model = result['recommended_model']
                model_counts[model] = model_counts.get(model, 0) + 1
            
            total_with_model = len(model_results)
            
            # 모델별 비율 계산
            for model in ['Kronecker', 'Weichselberger', 'Non-Separable']:
                count = model_counts.get(model, 0)
                summary[f'{model.lower()}_count'] = count
                summary[f'{model.lower()}_ratio'] = count / total_with_model if total_with_model > 0 else 0
            
            # 분리성 상태별 통계
            status_counts = {}
            for result in model_results:
                status = result['separability_status']
                status_counts[status] = status_counts.get(status, 0) + 1
            
            summary['kronecker_separable_count'] = status_counts.get('Kronecker Separable', 0)
            summary['weichselberger_separable_count'] = status_counts.get('Weichselberger Separable', 0) 
            summary['non_separable_count'] = status_counts.get('Non-Separable', 0)
            
            # 파라미터 효율성 통계
            param_counts = [r['num_req_params'] for r in model_results if 'num_req_params' in r]
            if param_counts:
                summary['avg_num_req_params'] = np.mean(param_counts)
                summary['min_num_req_params'] = np.min(param_counts) 
                summary['max_num_req_params'] = np.max(param_counts)
        
        return summary
    
    def save_rx_progress(self, analysis_result):
        """RX별 분석 결과를 실시간 CSV에 즉시 추가"""
        if not analysis_result:
            return  # None 결과는 저장하지 않음
        
        # 진행 상황 CSV 데이터 생성
        row_data = {
            'area_idx': analysis_result['area_idx'],
            'freq_ghz': analysis_result['freq_ghz'],
            'rx_idx': analysis_result['rx_idx'],
            'R_AE_trace': analysis_result['R_AE_trace'],
            'epsilon_d_desc': analysis_result['epsilon_d_desc'],         # eps_d (기본)
            'epsilon_d_kron': analysis_result['epsilon_d_desc'],         # eps_d (desc_sorted)
            'epsilon_U_desc': analysis_result['epsilon_U_desc'],         # eps_U (기본)
            'epsilon_u_1': analysis_result['epsilon_u_1'],
            'eps_R_fro': analysis_result.get('eps_R_fro', np.nan),
            'AE_rank': analysis_result.get('AE_rank', np.nan),
            'BS_rank': analysis_result.get('BS_rank', np.nan),
            'UE_rank': analysis_result.get('UE_rank', np.nan),
            'KM_rank': analysis_result.get('KM_rank', np.nan),
            'recommended_model': analysis_result.get('recommended_model', 'Unknown')
        }
        
        # DataFrame 생성
        df_row = pd.DataFrame([row_data])
        
        # 기존 파일이 있으면 추가, 없으면 새로 생성
        if os.path.exists(self.progress_csv_path):
            df_row.to_csv(self.progress_csv_path, mode='a', header=False, index=False)
        else:
            df_row.to_csv(self.progress_csv_path, index=False)
    
    def _save_csv_summary(self, csv_file, valid_results, summary_stats):
        """CSV 요약 파일 저장 (P1E 원본과 동일한 형식)"""
        
        # RX별 상세 데이터 (P1E progress.csv와 동일한 컬럼 순서)
        df_detail = pd.DataFrame(valid_results)
        
        # P1E progress.csv와 정확히 동일한 컬럼 순서 + epsilon_d_kron (디버깅용) 추가
        progress_cols = ['area_idx', 'freq_ghz', 'rx_idx', 'R_AE_trace',
                        'epsilon_d_desc', 'epsilon_d_kron', 'epsilon_U_desc', 'epsilon_u_1', 'eps_R_fro',
                        'AE_rank', 'BS_rank', 'UE_rank', 'KM_rank', 'recommended_model']
        
        # 사용 가능한 컬럼만 선택
        available_cols = [col for col in progress_cols if col in df_detail.columns]
        df_detail = df_detail[available_cols]
        
        # P1E와 동일한 단순 CSV 형식 (헤더 1줄 + 데이터)
        df_detail.to_csv(csv_file, index=False)
    
    def save_all_results(self, all_results):
        """전체 결과를 area-freq별로 그룹화하여 저장"""
        
        # Area-Freq별 그룹화
        area_freq_groups = {}
        for result in all_results:
            if result is not None:
                key = (result['area_idx'], result['freq_ghz'])
                if key not in area_freq_groups:
                    area_freq_groups[key] = []
                area_freq_groups[key].append(result)
        
        # 각 Area-Freq별 저장
        for (area_idx, freq), results in area_freq_groups.items():
            self._save_area_freq_results(area_idx, freq, results)
        
        # 전체 요약 저장
        if all_results:
            valid_results = [r for r in all_results if r is not None]
            if valid_results:
                self._save_overall_summary(valid_results)
    
    def _save_area_freq_results(self, area_idx, freq, rx_results):
        """Area-Freq별 결과 저장"""
        
        valid_results = [r for r in rx_results if r is not None]
        if not valid_results:
            print(f"Warning: Area{area_idx}_{freq}GHz에 대한 유효한 결과가 없습니다.")
            return
        
        summary_stats = self._compute_summary_statistics(valid_results)
        
        # CSV 요약 저장
        csv_filename = f"Area{area_idx}_{freq}GHz_Separability_Summary.csv"
        csv_filepath = os.path.join(self.config.P1D_ANALYSIS_OUTPUT_DIR, csv_filename)
        
        self._save_csv_summary(csv_filepath, valid_results, summary_stats)
        print(f"Area-Freq CSV 저장: {csv_filename}")
    
    def _save_overall_summary(self, valid_results):
        """전체 요약 저장"""
        summary_stats = self._compute_summary_statistics(valid_results)
        
        # 전체 요약 CSV
        csv_filename = "P1D_Overall_Separability_Summary.csv"
        csv_filepath = os.path.join(self.config.P1D_ANALYSIS_OUTPUT_DIR, csv_filename)
        
        self._save_csv_summary(csv_filepath, valid_results, summary_stats)
        print(f"전체 요약 CSV 저장: {csv_filename}")

# ===== SECTION 5: 메인 실행 P1D =====
def main():
    """P1D: P1B Valid RXs 기반 온라인 누적 분리성 분석"""
    
    # ===== 메모리 정리 및 GPU 설정 =====
    print("GPU 메모리 정리 및 설정 중...", end=' ', flush=True)
    
    # 1. Python 가비지 컬렉션 강제 실행
    import gc
    gc.collect()
    
    # 2. TensorFlow GPU 메모리 정리
    try:
        # 기존 세션/그래프 정리
        tf.keras.backend.clear_session()
        
        # GPU 메모리 정리
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                # 메모리 증가 허용 설정 (필요한 만큼만 할당)
                tf.config.experimental.set_memory_growth(gpu, True)
                
            # 가용한 GPU 메모리 확인
            gpu_details = tf.config.experimental.get_device_details(gpus[0])
            print(f"GPU: {gpu_details.get('device_name', 'Unknown')}", end=' ')
            
        # 3. TensorFlow XLA 설정 (메모리 효율성 향상)
        tf.config.optimizer.set_jit(True)  # XLA JIT 컴파일 활성화
        
        print("완료")
    except Exception as e:
        print(f"경고: GPU 설정 실패 ({e})")
    
    config = P1D_Config()
    
    # JIT 워밍업 (전체 프로그램에서 한 번만 수행)
    print("=== P1D JIT 컴파일 워밍업 수행 ===")
    warmup_analyzer = SeparabilityAnalyzer(config)
    warmup_analyzer.warmup_jit_functions()
    del warmup_analyzer  # 메모리 정리
    print("JIT 워밍업 완료. 이제 모든 RX에서 빠른 처리가 가능합니다.\n")
    
    # 전체 시작 시간 기록 (진행률 계산용)
    overall_start_time = time.time()
    
    # 결과 관리자 초기화
    result_manager = P1D_ResultManager(config)
    all_results = []  # 모든 RX 결과를 수집
    
    # P1B Valid RXs 데이터 기반 동적 루프
    total_rx_files = len(config.p1b_data_combinations)
    rx_file_count = 0
    
    # 동적 자리수 맞춤을 위한 너비 계산
    total_rx_width = len(str(total_rx_files))
    
    for area_index, fc, RX_index in config.p1b_data_combinations:
        rx_file_count += 1
        carrier_frequency = fc*10**9
        
        # 전체 시작 시간 기록
        tic_total = time.time()
        batch_size = config.batch_size
        N_UE = config.num_rx
        N_BS = config.N_BS
        
        # P1B Valid RXs 데이터 로딩
        ray_data = config.load_p1b_ray_data(area_index, fc, RX_index)
        if ray_data is None:
            print(f"Error: Failed to load ray data for Area{area_index}_{fc}GHz_RX{RX_index}")
            continue
        
        # P1B 정상 수행 전제 - 기본 Ray 구성 확인만
        nlos_info = {k: v for k, v in ray_data.items() if k.startswith(('has_los', 'total_rays', 'nlos_rays', 'nlos_ratio'))}
        if nlos_info.get('has_los_nlos_info'):
            print(f"  Ray 구성: {nlos_info['total_rays']}개 Ray (NLoS 비율: {nlos_info['nlos_ratio']*100:.1f}%)")
        else:
            print(f"  Ray 구성: {nlos_info.get('total_rays', 0)}개 Ray")
        
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

        # P1D 온라인 누적 분리성 분석
        print(f"\n=== P1D 분석 시작: Area{area_index}_{fc}GHz_RX{RX_index} ===")
        
        # 채널 분석기 초기화
        channel_analyzer = ChannelAnalyzer(config, OFDM_ChGen, topology, ray_pdap)
        
        print(f"온라인 누적 처리 시작 ({config.static_ch_realizations}개 Static 실현)...")
        start_accumulation = time.time()
        
        # Static 루프 - 온라인 누적 (진행 상황은 process_static_realization에서 출력)
        for static_idx in range(config.static_ch_realizations):
            channel_analyzer.process_static_realization(static_idx)
        
        accumulation_time = time.time() - start_accumulation
        accumulated_samples = int(channel_analyzer.accumulator.count.numpy())
        print(f"\n온라인 누적 완료 (OFDM 샘플 {accumulated_samples}개, {accumulation_time:.1f}초)")
        
        # 최종 분리성 분석 수행
        print("분리성 분석 수행 중...")
        start_analysis = time.time()
        analysis_result = channel_analyzer.finalize_analysis(area_index, fc, RX_index)
        analysis_time = time.time() - start_analysis
        
        if analysis_result:
            # 전체 결과 수집에 추가 (개별 파일 저장 없음 - CSV만 저장)
            all_results.append(analysis_result)
            
            # RX별 실시간 CSV 업데이트 (즉시 저장)
            result_manager.save_rx_progress(analysis_result)
            
            # 결과 요약 출력
            print(f"\n=== P1D 분석 결과 (분석: {analysis_time:.1f}초) ===")
            print(f"모델: {analysis_result['recommended_model']}")
            print(f"분리성 상태: {analysis_result['separability_status']}")
            print(f"eps_d_desc: {analysis_result['epsilon_d_desc']:.4f} (임계값: {config.epsilon_lambda_threshold})")
            print(f"eps_d_kron: {analysis_result['epsilon_d_desc']:.4f} [desc_sorted]")
            print(f"eps_U_desc: {analysis_result['epsilon_U_desc']:.4f} (임계값: {config.epsilon_U_threshold})")
            print(f"eps_R_fro: {analysis_result['eps_R_fro']:.4f} (R_AE - R_KM 프로베니우스 노름)")
            print(f"R_AE_trace: {analysis_result['R_AE_trace']:.2e}")
            print(f"필요 파라미터 수: {analysis_result['num_req_params']}")
            print(f"안테나 구성: TX={analysis_result['n_t']}, RX={analysis_result['n_r']}")
            print(f"OFDM 샘플 수: {analysis_result['n_ofdm_sam_final']}")
            print(f"✓ 실시간 CSV 업데이트: {os.path.basename(result_manager.progress_csv_path)}")
        else:
            print("Error: 분리성 분석 실패")
            all_results.append(None)  # 실패한 경우도 기록
        
        # RX 처리 후 메모리 정리 (대용량 처리 최적화)
        del channel_analyzer, ray_data
        if 'analysis_result' in locals() and analysis_result:
            # 큰 텐서들이 있을 수 있는 분석 결과는 유지 (CSV 저장용)
            pass
        
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
            print("P1D+P1E 통합 분석 진행률:")
            print(f"✓ [{rx_file_count:>{total_rx_width}}/{total_rx_files}] Area{area_index}_{fc}GHz_RX{RX_index} (통합 분석 완료)")
            print(f"현재 완료율: {rx_file_count/total_rx_files*100:.1f}% ({rx_file_count}/{total_rx_files})")
            print(f"전체 경과 시간: {format_time(overall_elapsed)}")
            print(f"평균 처리 속도: {avg_time_per_rx:.1f}초/RX")
            print(f"예상 남은 시간: {format_time(estimated_remaining)}")
            print(f"화면 클리어 주기: {config.PROGRESS_CLEAR_INTERVAL}개")
            print("-" * 80)
        
    # 분석 완료 메시지 및 결과 저장
    total_rx_processed = len(config.p1b_data_combinations)
    
    print(f"\n=== P1D 분석 완료 ===")
    print(f"총 {total_rx_processed}개 RX 분리성 분석 완료")
    print(f"각 RX당 샘플 수: {config.static_ch_realizations * config.doppler_time_realizations * config.doppler_sym_ofdm}개")
    print(f"결과 저장 위치: {config.P1D_ANALYSIS_OUTPUT_DIR}")
    print(f"분석 내용: 온라인 누적 공분산 → 분리성 분석 → 모델 선택")
    
    # 전체 결과를 CSV로 저장
    print(f"\n=== CSV 요약 생성 중 ===")
    if all_results:
        result_manager.save_all_results(all_results)
        valid_count = len([r for r in all_results if r is not None])
        print(f"최종 요약 CSV 저장 완료: {valid_count}/{len(all_results)}개 RX 결과 포함")
        
        # 실시간 CSV 파일 정보 출력
        if os.path.exists(result_manager.progress_csv_path):
            print(f"실시간 진행 CSV: {os.path.basename(result_manager.progress_csv_path)} ({valid_count}개 RX)")
        
        # 최종 메모리 정리
        import gc
        gc.collect()
        tf.keras.backend.clear_session()
        print("최종 메모리 정리 완료")
    else:
        print("Warning: 저장할 결과가 없습니다.")

if __name__ == "__main__":
    main()
