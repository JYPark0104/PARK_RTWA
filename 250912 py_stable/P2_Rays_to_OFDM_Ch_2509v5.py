# ======================================================================
# P2_Rays_to_OFDM_Ch_250809v3.py
# P2 v3: Ray Data → OFDM Channel 변환
# 원본: BlockChannel_with_Rays_to_h_freq.ipynb + RT_functions_1_0_2.py + channel_coefficients_JIN_L40S_102_Profile.py
# 
# v3 개선사항:
# - 독립적 정적 채널 발생: 매번 새로운 초기 위상으로 다양한 채널 환경 생성
# - 도플러 시변 채널 발생: 각 정적 채널에 대해 시간 변화 샘플링
# - 직접 시간 샘플링: float 시간값 직접 사용
# ======================================================================

# ===== SECTION 1: 환경 설정 (원본 Cell 0 그대로) =====
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
PI = tf.constant(3.141592653589793, tf.float32)
sin = tf.sin
cos = tf.cos
acos = tf.acos

# Constants (PI는 sionna.constants에서 import)

# ===== SECTION 2: P2_Config v3 (통계적 다양성을 위한 설정 확장) =====
class P2_Config:
    """P1 저장 데이터를 기반으로 실험 설정을 동적으로 구성 (v3: 계층적 채널 생성)"""
    
    def __init__(self):
        # 파일 경로 및 명명 패턴 설정 (중앙 집중화)

        # 스크립트의 디렉토리를 기준으로 절대 경로 설정
        script_dir = os.path.dirname(os.path.abspath(__file__))

        self.P1_INPUT_DIR = os.path.join(script_dir, "Sionna_RT_Results")
        self.P1_FILE_PATTERN = "Area{area}_{freq}GHz_Rays_ALL_RXs.npz"  # Updated to unified npz format
        self.P2_OUTPUT_DIR = os.path.join(script_dir, "OFDM_Ch_Results")
        self.P2_FILE_PATTERN = "Area{area}_{freq}GHz_RX{rx}_ofdm_ch.npy"
        
        # 후처리 청크 저장 설정 (기존 로직 보존하면서 파일 관리 개선)
        self.ENABLE_POST_CHUNK = True        # 후처리 청크 기능 활성화
        self.CHUNK_SIZE = 100               # 청크당 RX 개수
        self.CHUNK_PATTERN = "Area{area}_{freq}GHz_OFDM_Ch_RX{start}-{end}.npz"
        self.CLEANUP_INDIVIDUAL_FILES = True  # 청크 생성 후 개별 파일 삭제 여부
        
        # 콘솔 출력 제어 설정
        self.ENABLE_SCREEN_CLEAR = True      # 화면 클리어 기능 활성화/비활성화
        self.PROGRESS_CLEAR_INTERVAL = 10    # N개마다 화면 클리어 (테스트: 10, 실제: 100)
        
        # Area 필터링 설정 (None이면 모든 area 처리, 리스트로 특정 area 지정 가능)
        self.target_areas = [1]              # Area1만 처리 (예: [1, 3, 5] 또는 None)
        
        # P1 데이터 디렉토리 스캔하여 실험 설정 자동 감지
        self.detect_p1_data()
        
        # 원본 노트북의 고정 OFDM 시스템 파라미터들 (명확성을 위한 변수명 변경)
        self.OFDM_FFT = 128                              # 원본: N_FFT → 결과 OFDM 채널의 부반송파 수
        self.OFDM_SCS = 240e3                           # 원본: scs → 결과 OFDM 채널의 부반송파 간격 (Hz)
        self.OFDM_BW = self.OFDM_FFT * self.OFDM_SCS     # 원본: BW → OFDM 총 대역폭 (Hz)
        self.num_tx = 1
        self.num_rx = 1

        # v4: fc 리스트를 self.p1_data_combinations에서 추출하여 설정
        if self.p1_data_combinations:
            self.fcs = sorted(list(set([item[1] for item in self.p1_data_combinations])))
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

        # v3: 도플러 파라미터 (코히어런스 시간 기반)
        if self.fcs:
            v_max = self.Topology_Statistics["velocities_mean"] + 3*self.Topology_Statistics["velocities_stddev"]
            coherence_time = 3e8 / (v_max * self.fcs[0] * 1e9)
        else:
            coherence_time = 0.0 # 기본값 설정

        # v3: 채널 생성 파라미터
        self.static_ch_realizations = 400               # 독립 정적 채널 개수
        self.doppler_time_realizations = 1            # 각 정적 채널의 도플러 샘플 수
        self.doppler_sym_ofdm = 1 #10                      # 각 도플러 샘플당 OFDM 심볼 수
        self.doppler_time_max_sec = 0 # coherence_time   # 도플러 시간 샘플링 최대값 (초)
        
        # v3: 설정 정보 출력
        self.print_v3_config()

        # 원본 노트북의 기본 시스템 파라미터들
        self.N_BS = 1                                    # 원본: N_BS = 1
        self.batch_size = 1                              # 원본: batch_size = 1
        
        self.TX_Array = {
            "location": [0,0,0],
            "rotation": [0,0],
            "num_rows_per_panel": 8, # 원본: 4
            "num_cols_per_panel": 8, # 원본: 4
            "num_rows": 2,
            "num_cols": 2,
            "polarization": "single",
            "polarization_type": "V",
            "antenna_pattern": "38.901",
            "panel_vertical_spacing": 4.5, # 원본: 2.5
            "panel_horizontal_spacing": 4.5 # 원본: 2.5
        } # 256 안테나 구성
        
        self.RX_Array = {
            "num_rows_per_panel": 1,
            "num_cols_per_panel": 1,
            "num_rows": 1,
            "num_cols": 1,
            "polarization": "single",
            "polarization_type": "V",
            "antenna_pattern": "omni"
        }
        
        # 원본 노트북의 XPR 설정
        self.mean_xpr_list = {"UMi-LOS":9,"UMi-NLOS":8, "UMa-LOS":8,"UMa-NLOS":7}
        self.stddev_xpr_list = {"UMi-LOS":3,"UMi-NLOS":3, "UMa-LOS":4,"UMa-NLOS":4}
        self.mean_xpr = self.mean_xpr_list["UMa-NLOS"]
        self.stddev_xpr = self.stddev_xpr_list["UMa-NLOS"]
        
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
            # Area{X}_{freq}GHz_Rays_ALL_RXs.npz 패턴 매칭
            match = re.match(r'Area(\d+)_(.+)GHz_Rays_ALL_RXs\.npz', filename)
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
    
    def create_chunk_from_files(self, area_index, frequency, rx_indices):
        """개별 npy 파일들을 npz 청크로 묶기"""
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
            individual_file = os.path.join(self.P2_OUTPUT_DIR, 
                                         self.P2_FILE_PATTERN.format(area=area_index, freq=frequency, rx=rx_idx))
            
            # 진행률 표시 (캐리지 리턴 방식)
            progress_msg = f"  파일 로딩 중... RX{rx_idx} ({idx}/{total_rx})"
            print(f"\r{progress_msg:<60}", end='', flush=True)
            
            if os.path.exists(individual_file):
                ofdm_data = np.load(individual_file)
                chunk_data[f'ofdm_ch_rx_{rx_idx}'] = ofdm_data
            else:
                print(f"\nWarning: {individual_file} not found")
        
        print()  # 진행률 완료 후 줄바꿈
        
        # 청크 파일 저장
        chunk_filename = self.CHUNK_PATTERN.format(
            area=area_index, freq=frequency, start=min(rx_indices), end=max(rx_indices)
        )
        chunk_filepath = os.path.join(self.P2_OUTPUT_DIR, chunk_filename)
        
        # 메타데이터와 함께 저장
        save_data = {**chunk_metadata, **chunk_data}
        np.savez_compressed(chunk_filepath, **save_data)
        
        chunk_size_mb = os.path.getsize(chunk_filepath) / (1024**2)
        print(f"청크 저장 완료: {chunk_filename} ({chunk_size_mb:.1f} MB)")
        
        # 개별 파일 정리
        if self.CLEANUP_INDIVIDUAL_FILES:
            for rx_idx in rx_indices:
                individual_file = os.path.join(self.P2_OUTPUT_DIR,
                                             self.P2_FILE_PATTERN.format(area=area_index, freq=frequency, rx=rx_idx))
                if os.path.exists(individual_file):
                    os.remove(individual_file)
            print(f"개별 파일 {len(rx_indices)}개 정리 완료")
        
        return chunk_filepath
    

    def print_v3_config(self):
        """v3 설정 정보 출력"""
        print(f"P2 v3 채널 생성 설정:")
        print(f"  - Static Channel Realizations: {self.static_ch_realizations}")
        print(f"  - Doppler Time Realizations: {self.doppler_time_realizations}")
        print(f"  - 예상 총 심볼 수 (per file): {self.static_ch_realizations * self.doppler_time_realizations * self.doppler_sym_ofdm}")
        
        # Area 필터링 정보 출력
        if self.target_areas is None:
            print(f"  - Area 필터링: 모든 Area 처리")
        else:
            areas_str = ', '.join([f"Area{area}" for area in sorted(self.target_areas)])
            print(f"  - Area 필터링: {areas_str} 처리")


# ===== SECTION 3: 유틸리티 함수들 (원본에서 복사) =====

# RT_functions_1_0_2.py에서 필요한 함수들 복사
def radian_to_degree(radian):
    return radian * (180.0 / PI)

def degree_to_radian(degree):
    return degree * (PI / 180.0)

# channel_coefficients_JIN_L40S_102_Profile.py에서 필요한 함수들 복사
def print_vram_usage():
    """현재 GPU VRAM 사용량을 출력"""
    mem_info = tf.config.experimental.get_memory_info('/GPU:0')
    used_memory = mem_info['current'] / (1024**3)
    peak_memory = mem_info['peak'] / (1024**3)
    print(f"현재 GPU VRAM 사용량: {used_memory:.2f} GB / {peak_memory:.2f} GB")

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

# ===== ChannelCoefficientsGeneratorJIN 클래스 (핵심 부분만 복사) =====
class ChannelCoefficientsGeneratorJIN(Object):
    """P2 v3: Ray → OFDM Channel 변환 (독립 정적채널 + 도플러 시변채널)
    
    핵심 메서드:
    - _compute_ch_mimo_ofdm_38901_static(): 정적 MIMO OFDM 채널 생성
      * 매번 새로운 초기 위상(_step_10) 생성
      * Ray → delay bin domain 변환
    - _apply_doppler_ch_mimo_ofdm_freq(): 도플러 효과 적용
      * 직접 float 시간 샘플링
      * 시변 채널 생성
    
    v3 구조:
    - static_ch_realizations: 독립 정적 채널 개수 (외부 루프)
    - doppler_time_realizations: 각 정적 채널의 시변 샘플 수 (내부 루프)
    - 최종 심볼 수: static_ch_realizations × doppler_time_realizations × doppler_sym_ofdm
    
    P1-P2-P3 연계:
    - P1: 3D RT 시뮬레이션 → Ray 파라미터 (각도, 지연, 전력)
    - P2 v3: Ray → OFDM 채널 (통계적 다양성 강화)
    - P3: OFDM 채널 → 채널 공분산 행렬 (신뢰도 높은 통계량)
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

    # ===== 14개 필수 메서드들 =====
    
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
    
    def _unit_sphere_vector_Modified(self, theta, phi):
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
        return tf.expand_dims(rho_hat, axis=1)
    
    def _unit_sphere_vector_Modified2(self, theta, phi):

        rho_hat = tf.stack([sin(theta)*cos(phi),
                            sin(theta)*sin(phi),
                            cos(theta)], axis=-1)
        return tf.expand_dims(rho_hat, axis=-3)

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
            Delay bin domain으로 확장된 도착 방위각
            Shape: [B, N_Rays, N_BS, N_UE, N_FFT]
            
        zoa_delay_bin : tf.Tensor, dtype=tf.float32  
            Delay bin domain으로 확장된 도착 천정각
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

        r_hat_rx = self._unit_sphere_vector_Modified2(zoa_delay_bin, aoa_delay_bin) # [B, N_Rays, N_BS, N_UE, 1, N_FFT, 3], axis=-3 -> N_sym
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

# ===== SECTION 4: 메인 실행 v3 (계층적 채널 생성) =====
def main():
    """P2 v3: 계층적 채널 생성을 통한 통계적 다양성 확보"""
    
    config = P2_Config()
    
    # 전체 시작 시간 기록 (진행률 계산용)
    overall_start_time = time.time()
    
    # P1 데이터 기반 동적 루프
    total_rx_files = len(config.p1_data_combinations)
    rx_file_count = 0
    
    # 동적 자리수 맞춤을 위한 너비 계산
    total_rx_width = len(str(total_rx_files))
    
    for area_index, fc, RX_index in config.p1_data_combinations:
        rx_file_count += 1
        carrier_frequency = fc*10**9
        
        # 전체 시작 시간 기록
        tic_total = time.time()
        batch_size = config.batch_size
        N_UE = config.num_rx
        N_BS = config.N_BS
        
        # P1 데이터 로딩 (unified npz format)
        ray_data = config.load_p1_ray_data(area_index, fc, RX_index)
        if ray_data is None:
            print(f"Error: Failed to load ray data for Area{area_index}_{fc}GHz_RX{RX_index}")
            continue
        
        # Convert ray data to TensorFlow tensors
        ray_aoa = tf.convert_to_tensor(ray_data['phi_r'])
        ray_aod = tf.convert_to_tensor(ray_data['phi_t'])
        ray_zoa = tf.convert_to_tensor(ray_data['theta_r'])
        ray_zod = tf.convert_to_tensor(ray_data['theta_t'])
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
        OFDM_ChGen = ChannelCoefficientsGeneratorJIN(carrier_frequency, config.OFDM_SCS, ArrayTX, ArrayRX, False)
        
        # v3: 최종 결과 저장 리스트 (사전 할당)
        all_static_ch_realizations_data = [None] * config.static_ch_realizations
        
        # 매번 새로운 랜덤성 주입: XPR, topology 등
        ray_xpr = 10**(tf.random.normal(shape=[batch_size,1,N_UE,1,ray_aoa.shape[-1]], 
                                                mean=config.mean_xpr, stddev=config.stddev_xpr)/10)

        ray_pdap = Rays(delays=ray_delay, powers=ray_power, aoa=ray_aoa, aod=ray_aod,
                            zoa=ray_zoa, zod=ray_zod, xpr=ray_xpr)

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
        tx_orientations = tf.random.normal(shape=[batch_size,N_BS,3], 
                                                    mean=config.Topology_Statistics["orientations_mean"], 
                                                    stddev=config.Topology_Statistics["orientations_stddev"], 
                                                    dtype=tf.float32)
        rx_orientations = tf.random.normal(shape=[batch_size,N_UE,3], 
                                                    mean=config.Topology_Statistics["orientations_mean"], 
                                                    stddev=config.Topology_Statistics["orientations_stddev"], 
                                                    dtype=tf.float32)

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

        # v3: 외부 루프 - 독립 정적 채널 생성
        for static_idx in range(config.static_ch_realizations):

            # Static Channel 생성 (매번 새로운 초기 위상)
            h_delay_bin_static, aoa_delay_bin, zoa_delay_bin = OFDM_ChGen._compute_ch_mimo_ofdm_38901_static(
                topology, ray_pdap, config.OFDM_FFT, config.OFDM_SCS)
            
            # 필수 transpose 처리 (차원 정렬)
            h_delay_bin_static = tf.transpose(h_delay_bin_static, [0,3,5,6,1,2,7,4])  
            aoa_delay_bin = tf.transpose(aoa_delay_bin, [0,3,1,2,4])  
            zoa_delay_bin = tf.transpose(zoa_delay_bin, [0,3,1,2,4]) 
            
            # 현재 Static Realization의 Doppler 샘플들을 담을 리스트
            current_doppler_real_collection = [None] * config.doppler_time_realizations
            
            # v3: 내부 루프 - 도플러 시변 채널 생성
            for doppler_idx in range(config.doppler_time_realizations):
                
                # v3: 직접 float 시간 샘플링
                doppler_times = tf.random.uniform(
                    shape=[config.doppler_sym_ofdm],
                    minval=0.0,
                    maxval=config.doppler_time_max_sec,
                    dtype=OFDM_ChGen.rdtype
                )
                
                # Doppler 효과 적용
                ofdm_ch_doppler_instance = OFDM_ChGen._apply_doppler_ch_mimo_ofdm_freq(
                    topology, doppler_times,
                    h_delay_bin_static, aoa_delay_bin, zoa_delay_bin
                )
                
                # 저장용 차원 순서로 변환: [B, N_BS, N_UE, doppler_sym_ofdm, OFDM_FFT, N_r, N_t]
                transposed_instance = tf.transpose(ofdm_ch_doppler_instance, [0,3,4,5,6,1,2])
                current_doppler_real_collection[doppler_idx] = transposed_instance
            
            # 내부 루프 결과 종합: 하나의 정적 채널의 시간 변화 데이터
            single_static_realization_data = tf.concat(current_doppler_real_collection, axis=3)
            
            # 외부 루프 결과 리스트에 저장 (사전 할당된 위치)
            all_static_ch_realizations_data[static_idx] = single_static_realization_data
            
        
        # 최종 데이터 생성: 모든 독립 정적 채널을 시간 축으로 결합
        final_ofdm_ch_data = tf.concat(all_static_ch_realizations_data, axis=3)
        
        # 저장: [total_symbols, OFDM_FFT, N_r, N_t] 형태로 squeeze
        os.makedirs(config.P2_OUTPUT_DIR, exist_ok=True)
        output_filename = config.P2_FILE_PATTERN.format(area=area_index, freq=fc, rx=RX_index)
        np.save(f'{config.P2_OUTPUT_DIR}/{output_filename}', 
                tf.squeeze(final_ofdm_ch_data, axis=[0,1,2]).numpy())  # B, N_BS, N_UE 제거
        
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
            print("P2 OFDM 채널 생성 진행률:")
            print(f"✓ [{rx_file_count:>{total_rx_width}}/{total_rx_files}] Area{area_index}_{fc}GHz_RX{RX_index}: {toc_total - tic_total:.1f}s → {output_filename}")
            print(f"현재 완료율: {rx_file_count/total_rx_files*100:.1f}% ({rx_file_count}/{total_rx_files})")
            print(f"전체 경과 시간: {format_time(overall_elapsed)}")
            print(f"평균 처리 속도: {avg_time_per_rx:.1f}초/RX")
            print(f"예상 남은 시간: {format_time(estimated_remaining)}")
            print(f"화면 클리어 주기: {config.PROGRESS_CLEAR_INTERVAL}개")
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
        
        print(f"\n=== P2 후처리 청크 생성 완료 ===")
        total_chunks_created = sum((len(rx_list) + config.CHUNK_SIZE - 1) // config.CHUNK_SIZE 
                                 for rx_list in area_freq_groups.values())
        total_rx_processed = sum(len(rx_list) for rx_list in area_freq_groups.values())
        print(f"총 {total_chunks_created}개 청크 생성 완료 (총 {total_rx_processed}개 RX 처리)")

if __name__ == "__main__":
    main()
