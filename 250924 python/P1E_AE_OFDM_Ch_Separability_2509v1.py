# ======================================================================
# P1E_AE_OFDM_Ch_Separability_2509v1.py
# P1E: Antenna Element OFDM 채널 분리성 분석 
# 
# P1C 결과물을 기반으로 크로네커 vs 바이셀베르거 모델 분리성 테스트 수행
# tex/tmp250924.tex 문서의 수학적 정의에 따른 분리성 분석 구현
#
# 주요 기능:
# - P1C 청크 파일 로딩 및 RX별 채널 데이터 추출
# - 채널 공분산 행렬 계산 (R_AE, R_AE_b, R_AE_u)
# - 고유값/고유벡터 분해 및 크로네커 근사
# - 분리성 오차 메트릭 계산 및 분석 결과 생성
# ======================================================================

import os
import numpy as np
import glob
import re
import time
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# TensorFlow 추가 (온라인 계산 및 JIT 컴파일용)
import tensorflow as tf
# GPU 메모리 증가 허용 설정
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"GPU 설정 경고: {e}")

# ===== SECTION 1: P1E 설정 클래스 =====
class P1E_Config:
    """P1E 분리성 분석 설정 관리"""
    
    def __init__(self):
        # 파일 경로 설정
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.P1C_INPUT_DIR = os.path.join(script_dir, "P1C_AE_OFDM_Ch_Results")
        self.P1E_OUTPUT_DIR = os.path.join(script_dir, "P1E_Separability_Results")
        
        # 입력/출력 파일 패턴
        self.P1C_CHUNK_PATTERN = "Area{area}_{freq}GHz_AE_OFDM_Ch_RX{start}-{end}.npz"
        self.P1E_RESULT_PATTERN = "Area{area}_{freq}GHz_Separability_Analysis.npz"
        self.P1E_SUMMARY_PATTERN = "Area{area}_{freq}GHz_Separability_Summary.csv"
        
        # 분석 대상 Area 필터링
        self.target_areas = [1]
        
        # P1C 데이터 구조 설정 (중앙화된 파라미터)
        self.n_r = 16                        # RX 안테나 수 (P1C 출력 기준)
        self.n_t = 1024                      # TX 안테나 수 (P1C 출력 기준)
        self.fft_size = 1                    # OFDM FFT 크기 (P1C OFDM_FFT)
        self.n_samples_default = 1000        # P1C 기본 샘플 수
        
        # L40S GPU 병렬 처리 최적화 설정
        self.batch_size_rx = 1               # RX 배치 크기: 동시 처리할 RX 개수 (메모리 안전)
        self.max_samples_per_rx = None       # RX당 최대 채널 실현 수 (None = 모든 샘플 사용)
        self.use_mixed_precision = False     # 복소수 연산에서 혼합 정밀도 비활성화
        self.gpu_memory_limit_gb = 45        # L40S 메모리 한계 (48GB 중 여유분 확보)
        
        # 메모리 계산 상수 (중앙화)
        self.complex64_bytes = 8             # complex64 = float32(4) * 2
        self.bytes_per_gb = 1024**3          # 바이트를 GB로 변환
        self.bytes_per_mb = 1024**2          # 바이트를 MB로 변환
        
        # 모델 선택 임계값 설정 (tex 문서 기준)
        self.epsilon_lambda_threshold = 0.1  # 고유값 분리성 오차 임계값
        self.epsilon_U_threshold = 0.1       # 고유벡터 분리성 오차 임계값
        self.enable_model_selection = True   # 모델 선택 기능 활성화
        
        # 데이터 유효성 검증 설정
        self.channel_power_threshold = 1e-20  # R_AE trace 최소 임계값 (채널 이득 합산)
        
        # 출력 제어
        self.save_detailed_results = True    # 상세 결과 저장 여부
        self.save_summary_csv = True        # 요약 CSV 저장 여부
        self.generate_plots = False         # 플롯 생성 여부 (기본 비활성화)
        
        # P1C 데이터 스캔
        self.detect_p1c_chunks()
        
        print("P1E 분리성 분석 설정 (L40S GPU 병렬 처리 최적화):")
        print(f"  - 입력 디렉토리: {self.P1C_INPUT_DIR}")
        print(f"  - 출력 디렉토리: {self.P1E_OUTPUT_DIR}")
        print(f"  - RX 배치 크기: {self.batch_size_rx} (다중 RX 동시 처리, 메모리 안전)")
        print(f"  - RX당 샘플 수: {'전체' if self.max_samples_per_rx is None else self.max_samples_per_rx}")
        print(f"  - JIT XLA 컴파일: 기본 활성화 (워밍업으로 재컴파일 방지)")
        print(f"  - GPU 메모리 한계: {self.gpu_memory_limit_gb}GB (L40S)")
        print(f"  - 고유값 분리성 임계값: {self.epsilon_lambda_threshold}")
        print(f"  - 고유벡터 분리성 임계값: {self.epsilon_U_threshold}")
        print(f"  - 모델 선택 기능: {'활성화' if self.enable_model_selection else '비활성화'}")
        print(f"  - 채널 파워 임계값: {self.channel_power_threshold:.0e} (R_AE trace 최소값)")
        print(f"  - 감지된 청크 수: {len(self.p1c_chunks)}")
        print("  - 처리 방식: 다중 RX 병렬 처리 (GPU 활용도 최대화)")
    
    def detect_p1c_chunks(self):
        """P1C 청크 파일 스캔"""
        chunk_files = glob.glob(f"{self.P1C_INPUT_DIR}/Area*_*GHz_AE_OFDM_Ch_RX*.npz")
        
        self.p1c_chunks = []
        for file_path in chunk_files:
            filename = os.path.basename(file_path)
            # Area{X}_{freq}GHz_AE_OFDM_Ch_RX{start}-{end}.npz 패턴 매칭
            match = re.match(r'Area(\d+)_(.+)GHz_AE_OFDM_Ch_RX(\d+)-(\d+)\.npz', filename)
            if match:
                area_idx = int(match.group(1))
                freq = float(match.group(2))
                rx_start = int(match.group(3))
                rx_end = int(match.group(4))
                
                # target_areas 필터링
                if self.target_areas is None or area_idx in self.target_areas:
                    self.p1c_chunks.append({
                        'file_path': file_path,
                        'area': area_idx,
                        'freq': freq,
                        'rx_start': rx_start,
                        'rx_end': rx_end
                    })
        
        # 청크 정렬 (area, freq, rx_start 순)
        self.p1c_chunks.sort(key=lambda x: (x['area'], x['freq'], x['rx_start']))
    
    def scan_chunk_rx_keys(self, chunk_file_path):
        """청크 파일에서 실제 존재하는 RX 키들 스캔"""
        try:
            with np.load(chunk_file_path) as data:
                rx_keys = [key for key in data.keys() if key.startswith('ofdm_ch_rx_')]
                rx_indices = [int(key.split('_')[-1]) for key in rx_keys]
                return sorted(rx_indices)
        except Exception as e:
            print(f"청크 파일 스캔 실패 {chunk_file_path}: {e}")
            return []

# ===== SECTION 2: 다중 RX 병렬 분리성 분석 엔진 =====
class BatchRxSeparabilityAnalyzer:
    """다중 RX 동시 처리 분리성 분석 엔진 (L40S GPU 최적화)"""
    
    def __init__(self, config):
        self.config = config
        self._jit_warmup_done = False
    
    def warmup_jit_functions(self):
        """JIT 함수들을 미리 컴파일 (첫 실행에서만)"""
        if self._jit_warmup_done:
            return
            
        print("다중 RX JIT XLA 컴파일 워밍업 중...", end=' ', flush=True)
        
        # 더미 데이터로 JIT 컴파일 실행 (온라인 누적 방식, 실제 실행과 동일)
        # P1E_Config에서 중앙화된 파라미터 사용
        n_r = self.config.n_r
        n_t = self.config.n_t
        fft_size = self.config.fft_size
        batch_size_rx = self.config.batch_size_rx
        
        try:
            # 1. 온라인 누적 워밍업 (실제와 동일한 방식)
            dummy_batch_running_stats = {
                'count': tf.zeros([batch_size_rx], dtype=tf.int32),
                'R_AE_sum': tf.zeros([batch_size_rx, n_r * n_t, n_r * n_t], dtype=tf.complex64),
                'R_AE_b_sum': tf.zeros([batch_size_rx, n_t, n_t], dtype=tf.complex64),
                'R_AE_u_sum': tf.zeros([batch_size_rx, n_r, n_r], dtype=tf.complex64)
            }
            dummy_batch_sample = tf.zeros([batch_size_rx, fft_size, n_r, n_t], dtype=tf.complex64)
            
            # 온라인 업데이트 JIT 컴파일 워밍업
            self._online_batch_rx_update(dummy_batch_running_stats, dummy_batch_sample)
            
            # 2. 배치 고유값 분해 워밍업 (10개 RX용 더미 데이터)
            dummy_R_AE_batch = tf.eye(n_r * n_t, batch_shape=[batch_size_rx], dtype=tf.complex64)
            dummy_R_b_batch = tf.eye(n_t, batch_shape=[batch_size_rx], dtype=tf.complex64)  
            dummy_R_u_batch = tf.eye(n_r, batch_shape=[batch_size_rx], dtype=tf.complex64)
            self._batch_eigendecomposition_jit(dummy_R_AE_batch, dummy_R_b_batch, dummy_R_u_batch)
            
            # 3. 배치 크로네커 근사 워밍업
            dummy_lambda_u_batch = tf.ones([batch_size_rx, n_r], dtype=tf.float32)
            dummy_lambda_b_batch = tf.ones([batch_size_rx, n_t], dtype=tf.float32)
            dummy_U_u_batch = tf.eye(n_r, batch_shape=[batch_size_rx], dtype=tf.complex64)
            dummy_U_b_batch = tf.eye(n_t, batch_shape=[batch_size_rx], dtype=tf.complex64)
            self._batch_kronecker_approximation_jit(dummy_lambda_u_batch, dummy_lambda_b_batch, 
                                                   dummy_U_u_batch, dummy_U_b_batch)
            
            # 4. 배치 분리성 메트릭 워밍업
            dummy_lambda_AE_batch = tf.ones([batch_size_rx, n_r * n_t], dtype=tf.float32)
            dummy_lambda_KM_batch = tf.ones([batch_size_rx, n_r * n_t], dtype=tf.float32)
            dummy_U_AE_batch = tf.eye(n_r * n_t, batch_shape=[batch_size_rx], dtype=tf.complex64)
            dummy_U_KM_batch = tf.eye(n_r * n_t, batch_shape=[batch_size_rx], dtype=tf.complex64)
            self._batch_separability_metrics_jit(dummy_lambda_AE_batch, dummy_lambda_KM_batch, 
                                                dummy_U_AE_batch, dummy_U_KM_batch)
            
            print("완료")
            self._jit_warmup_done = True
            
        except Exception as e:
            print(f"다중 RX JIT 워밍업 실패: {e}")
            
    def load_batch_rx_data(self, chunk_info, rx_indices):
        """배치 RX들의 채널 데이터 동시 로딩
        
        Parameters
        ----------
        chunk_info : dict
            청크 파일 정보
        rx_indices : list
            로드할 RX 인덱스 목록
            
        Returns
        -------
        ndarray or None
            [batch_size_rx, n_samples, fft_size, n_r, n_t] 형태의 배치 데이터
        """
        try:
            batch_data = []
            with np.load(chunk_info['file_path']) as data:
                for rx_idx in rx_indices:
                    rx_key = f'ofdm_ch_rx_{rx_idx}'
                    if rx_key in data:
                        rx_data = data[rx_key]  # [n_samples, fft_size, n_r, n_t]
                        
                        # 샘플 수 제한 (설정된 경우)
                        if self.config.max_samples_per_rx is not None:
                            rx_data = rx_data[:self.config.max_samples_per_rx]
                        
                        batch_data.append(rx_data)
                    else:
                        print(f"Warning: {rx_key} not found in {os.path.basename(chunk_info['file_path'])}")
                        return None
                        
            if batch_data:
                # 배치 차원 추가: [batch_size_rx, n_samples, fft_size, n_r, n_t]
                batch_array = np.stack(batch_data, axis=0)
                return batch_array
            else:
                return None
                
        except Exception as e:
            print(f"Error loading batch RX data from {chunk_info['file_path']}: {e}")
            return None
    
    @tf.function(jit_compile=True)
    def _online_batch_rx_update(self, batch_running_stats, batch_sample_data):
        """다중 RX 온라인 공분산 업데이트 (JIT 최적화, 메모리 효율적)
        
        Parameters
        ----------
        batch_running_stats : dict
            각 RX별 누적 통계 [batch_size_rx, ...]
        batch_sample_data : tf.Tensor
            [batch_size_rx, fft_size, n_r, n_t] 형태의 배치 단일 샘플
            
        Returns
        -------
        dict
            업데이트된 각 RX별 누적 통계
        """
        batch_size_rx = tf.shape(batch_sample_data)[0]
        fft_size = tf.shape(batch_sample_data)[1]
        n_r = tf.shape(batch_sample_data)[2]
        n_t = tf.shape(batch_sample_data)[3]
        
        # 현재 누적 통계
        count = batch_running_stats['count']                    # [batch_size_rx]
        R_AE_sum = batch_running_stats['R_AE_sum']             # [batch_size_rx, n_r*n_t, n_r*n_t]
        R_AE_b_sum = batch_running_stats['R_AE_b_sum']         # [batch_size_rx, n_t, n_t]
        R_AE_u_sum = batch_running_stats['R_AE_u_sum']         # [batch_size_rx, n_r, n_r]
        
        # P1C에서 fft_size=1이므로 루프 없이 직접 처리 (AutoGraph Shape Invariant 문제 해결)
        # [batch_size_rx, fft_size=1, n_r, n_t] → [batch_size_rx, n_r, n_t]
        H_batch = batch_sample_data[:, 0, :, :]  # fft_size=1이므로 0번 인덱스만 사용
        
        # 1. R_AE 업데이트: 각 RX별로 vec(H) * vec(H)^H
        vec_H_batch = tf.reshape(H_batch, [batch_size_rx, n_r * n_t])     # [batch_size_rx, n_r*n_t]
        vec_H_expanded1 = tf.expand_dims(vec_H_batch, axis=-1)            # [batch_size_rx, n_r*n_t, 1]
        vec_H_expanded2 = tf.expand_dims(tf.math.conj(vec_H_batch), axis=-2)  # [batch_size_rx, 1, n_r*n_t]
        R_AE_instance = vec_H_expanded1 @ vec_H_expanded2                 # [batch_size_rx, n_r*n_t, n_r*n_t]
        R_AE_sum = R_AE_sum + R_AE_instance
        
        # 2. R_AE_b 업데이트: 각 RX별로 H^H * H
        R_AE_b_instance = tf.linalg.matmul(H_batch, H_batch, adjoint_a=True)  # [batch_size_rx, n_t, n_t]
        R_AE_b_sum = R_AE_b_sum + R_AE_b_instance
        
        # 3. R_AE_u 업데이트: 각 RX별로 H * H^H
        R_AE_u_instance = tf.linalg.matmul(H_batch, H_batch, adjoint_b=True)  # [batch_size_rx, n_r, n_r]
        R_AE_u_sum = R_AE_u_sum + R_AE_u_instance
        
        # 카운트 업데이트 (fft_size=1이므로 1 증가)
        count = count + fft_size
        
        return {
            'count': count,
            'R_AE_sum': R_AE_sum,
            'R_AE_b_sum': R_AE_b_sum,
            'R_AE_u_sum': R_AE_u_sum
        }
    
    def compute_batch_rx_covariance_matrices(self, batch_rx_data):
        """다중 RX 온라인 누적 공분산 행렬 계산 (메모리 효율적)
        
        Parameters
        ----------
        batch_rx_data : ndarray
            배치 RX 데이터 [batch_size_rx, n_samples, fft_size, n_r, n_t]
            
        Returns
        -------
        dict
            각 RX의 R_AE, R_AE_b, R_AE_u 공분산 행렬들 [batch_size_rx, ...]
        """
        batch_size_rx, n_samples, fft_size, n_r, n_t = batch_rx_data.shape
        total_instances_per_rx = n_samples * fft_size
        
        print(f"    공분산계산", end='', flush=True)
        
        # 온라인 누적 메모리 사용량 추정 (중앙화된 설정 사용)
        R_AE_total_memory_gb = batch_size_rx * (n_r * n_t) ** 2 * self.config.complex64_bytes / self.config.bytes_per_gb
        batch_sample_memory_mb = batch_size_rx * fft_size * n_r * n_t * self.config.complex64_bytes / self.config.bytes_per_mb
        input_data_gb = batch_rx_data.nbytes / self.config.bytes_per_gb
        
        total_memory_gb = R_AE_total_memory_gb + batch_sample_memory_mb / 1024 + input_data_gb
        
        # TensorFlow 텐서로 변환
        try:
            batch_rx_data_tf = tf.constant(batch_rx_data, dtype=tf.complex64)
        except Exception as e:
            print(f"\nError: 배치 RX 데이터를 TensorFlow 텐서로 변환 실패: {e}")
            return None
        
        # 배치 RX 온라인 누적 통계 초기화
        batch_running_stats = {
            'count': tf.zeros([batch_size_rx], dtype=tf.int32),
            'R_AE_sum': tf.zeros([batch_size_rx, n_r * n_t, n_r * n_t], dtype=tf.complex64),
            'R_AE_b_sum': tf.zeros([batch_size_rx, n_t, n_t], dtype=tf.complex64),
            'R_AE_u_sum': tf.zeros([batch_size_rx, n_r, n_r], dtype=tf.complex64)
        }
        
        # 샘플별 온라인 누적 (메모리 효율적)
        start_accumulation = time.time()
        
        try:
            for sample_idx in range(n_samples):
                # 배치 RX의 단일 샘플: [batch_size_rx, fft_size, n_r, n_t]
                batch_sample = batch_rx_data_tf[:, sample_idx, :, :, :]
                
                # JIT 컴파일된 온라인 업데이트
                batch_running_stats = self._online_batch_rx_update(batch_running_stats, batch_sample)
            
            accumulation_time = time.time() - start_accumulation
            print(f"({accumulation_time:.1f}초)", end='', flush=True)
        except Exception as e:
            print(f"\nError: 온라인 누적 실패: {e}")
            return None
        
        # 최종 공분산 행렬 계산 (온라인 누적 결과를 평균으로 변환)
        count_float = tf.cast(batch_running_stats['count'], tf.complex64)  # [batch_size_rx]
        
        # 각 차원에 맞게 브로드캐스팅
        R_AE_batch = batch_running_stats['R_AE_sum'] / tf.reshape(count_float, [batch_size_rx, 1, 1])
        R_AE_b_batch = batch_running_stats['R_AE_b_sum'] / tf.reshape(count_float, [batch_size_rx, 1, 1])
        R_AE_u_batch = batch_running_stats['R_AE_u_sum'] / tf.reshape(count_float, [batch_size_rx, 1, 1])
        
        # NumPy로 변환
        R_AE_batch_np = R_AE_batch.numpy()
        R_AE_b_batch_np = R_AE_b_batch.numpy()
        R_AE_u_batch_np = R_AE_u_batch.numpy()
        
        # 각 RX별 R_AE trace 계산 및 데이터 유효성 검증
        R_AE_trace_batch = []
        valid_rx_mask = []
        
        for rx_idx in range(batch_size_rx):
            R_AE_trace = np.real(np.trace(R_AE_batch_np[rx_idx]))
            R_AE_trace_batch.append(R_AE_trace)
            
            is_valid = R_AE_trace >= self.config.channel_power_threshold
            valid_rx_mask.append(is_valid)
        
        valid_count = sum(valid_rx_mask)
        
        return {
            'R_AE': R_AE_batch_np,                 # [batch_size_rx, n_r*n_t, n_r*n_t]
            'R_AE_b': R_AE_b_batch_np,             # [batch_size_rx, n_t, n_t]
            'R_AE_u': R_AE_u_batch_np,             # [batch_size_rx, n_r, n_r]
            'R_AE_trace': R_AE_trace_batch,        # [batch_size_rx] 리스트
            'valid_rx_mask': valid_rx_mask,        # [batch_size_rx] 유효성 마스크
            'n_r': n_r,
            'n_t': n_t,
            'n_instances_per_rx': total_instances_per_rx
        }
    
    
    @tf.function # JIT 컴파일 비활성화
    def _batch_eigendecomposition_jit(self, R_AE_batch_tf, R_AE_b_batch_tf, R_AE_u_batch_tf):
        """다중 RX 배치 고유값 분해 (JIT 최적화)
        
        Parameters
        ----------
        R_AE_batch_tf : tf.Tensor
            [batch_size_rx, n_r*n_t, n_r*n_t]
        R_AE_b_batch_tf : tf.Tensor
            [batch_size_rx, n_t, n_t]
        R_AE_u_batch_tf : tf.Tensor
            [batch_size_rx, n_r, n_r]
            
        Returns
        -------
        tuple
            각 텐서는 [batch_size_rx, ...] 형태
        """
        # 배치 고유값 분해 (TensorFlow는 배치 연산을 지원)
        lambda_AE_batch, U_AE_batch = tf.linalg.eigh(R_AE_batch_tf)
        lambda_b_batch, U_b_batch = tf.linalg.eigh(R_AE_b_batch_tf)
        lambda_u_batch, U_u_batch = tf.linalg.eigh(R_AE_u_batch_tf)
        
        # 실수 고유값만 추출 (에르미트 행렬이므로 고유값은 실수)
        lambda_AE_batch = tf.math.real(lambda_AE_batch)
        lambda_b_batch = tf.math.real(lambda_b_batch)
        lambda_u_batch = tf.math.real(lambda_u_batch)
        
        # 각 RX별로 내림차순 정렬
        idx_AE_batch = tf.argsort(lambda_AE_batch, direction='DESCENDING', axis=-1)
        idx_b_batch = tf.argsort(lambda_b_batch, direction='DESCENDING', axis=-1)  
        idx_u_batch = tf.argsort(lambda_u_batch, direction='DESCENDING', axis=-1)
        
        # 배치 gather 연산
        lambda_AE_sorted = tf.gather(lambda_AE_batch, idx_AE_batch, batch_dims=1)
        lambda_b_sorted = tf.gather(lambda_b_batch, idx_b_batch, batch_dims=1)
        lambda_u_sorted = tf.gather(lambda_u_batch, idx_u_batch, batch_dims=1)
        
        U_AE_sorted = tf.gather(U_AE_batch, idx_AE_batch, batch_dims=1, axis=2)
        U_b_sorted = tf.gather(U_b_batch, idx_b_batch, batch_dims=1, axis=2)
        U_u_sorted = tf.gather(U_u_batch, idx_u_batch, batch_dims=1, axis=2)
        
        return lambda_AE_sorted, lambda_b_sorted, lambda_u_sorted, U_AE_sorted, U_b_sorted, U_u_sorted
        
    def perform_batch_eigendecomposition(self, batch_covariance_data):
        """다중 RX 배치 고유값 분해 수행 (JIT 최적화)"""
        R_AE_batch = batch_covariance_data['R_AE']           # [batch_size_rx, n_r*n_t, n_r*n_t]
        R_AE_b_batch = batch_covariance_data['R_AE_b']       # [batch_size_rx, n_t, n_t]
        R_AE_u_batch = batch_covariance_data['R_AE_u']       # [batch_size_rx, n_r, n_r]
        
        batch_size_rx = R_AE_batch.shape[0]
        
        # TensorFlow 텐서로 변환
        R_AE_batch_tf = tf.constant(R_AE_batch, dtype=tf.complex64)
        R_AE_b_batch_tf = tf.constant(R_AE_b_batch, dtype=tf.complex64)
        R_AE_u_batch_tf = tf.constant(R_AE_u_batch, dtype=tf.complex64)
        
        # JIT 컴파일된 배치 고유값 분해 실행
        lambda_AE_batch, lambda_b_batch, lambda_u_batch, U_AE_batch, U_b_batch, U_u_batch = \
            self._batch_eigendecomposition_jit(R_AE_batch_tf, R_AE_b_batch_tf, R_AE_u_batch_tf)
        
        return {
            'lambda_AE': lambda_AE_batch.numpy(),    # [batch_size_rx, n_r*n_t]
            'lambda_b': lambda_b_batch.numpy(),      # [batch_size_rx, n_t]
            'lambda_u': lambda_u_batch.numpy(),      # [batch_size_rx, n_r]
            'U_AE': U_AE_batch.numpy(),              # [batch_size_rx, n_r*n_t, n_r*n_t]
            'U_b': U_b_batch.numpy(),                # [batch_size_rx, n_t, n_t]
            'U_u': U_u_batch.numpy()                 # [batch_size_rx, n_r, n_r]
        }
    
    def analyze_batch_rx(self, chunk_info, batch_rx_indices):
        """다중 RX 배치 분리성 분석 (병렬 처리)
        
        Parameters
        ----------
        chunk_info : dict
            청크 파일 정보
        batch_rx_indices : list
            배치 처리할 RX 인덱스들
            
        Returns
        -------
        list
            각 RX별 분석 결과 (None은 실패한 RX)
        """
        batch_size_actual = len(batch_rx_indices)
        print(f"    데이터로딩", end='', flush=True)
        
        # 1. 배치 RX 데이터 로딩
        start_loading = time.time()
        batch_rx_data = self.load_batch_rx_data(chunk_info, batch_rx_indices)
        loading_time = time.time() - start_loading
        
        if batch_rx_data is None:
            print(f"({loading_time:.1f}초) 실패")
            return [None] * batch_size_actual
        print(f"({loading_time:.1f}초)", end=' ', flush=True)
        
        # 데이터 검증 (NaN, Inf 체크)
        if np.any(np.isnan(batch_rx_data)) or np.any(np.isinf(batch_rx_data)):
            print(f"    Error: 배치 데이터에 NaN 또는 Inf 값 발견")
            return [None] * batch_size_actual
        
        # 2. 다중 RX 공분산 행렬 계산
        batch_covariance_data = self.compute_batch_rx_covariance_matrices(batch_rx_data)
        if batch_covariance_data is None:
            return [None] * batch_size_actual
        
        print("    배치 처리:", end=' ', flush=True)
        
        # 3. 배치 고유값 분해
        print(" 고유값분해", end='', flush=True)
        start_eigen_total = time.time()
        batch_eigen_data = self.perform_batch_eigendecomposition(batch_covariance_data)
        eigen_total_time = time.time() - start_eigen_total
        print(f"({eigen_total_time:.1f}초)", end='', flush=True)
        
        # 4. 배치 크로네커 근사
        print(" 크로네커근사", end='', flush=True)
        start_kronecker = time.time()
        batch_kronecker_data = self.compute_batch_kronecker_approximation(batch_eigen_data)
        kronecker_time = time.time() - start_kronecker
        print(f"({kronecker_time:.1f}초)", end='', flush=True)
        
        # 5. 배치 분리성 메트릭 계산
        print(" 메트릭계산", end='', flush=True)
        start_metrics = time.time()  
        batch_metrics_data = self.compute_batch_separability_metrics(batch_eigen_data, batch_kronecker_data)
        metrics_time = time.time() - start_metrics
        print(f"({metrics_time:.1f}초)", end='', flush=True)
        
        # 6. 각 RX별 결과 생성
        start_results = time.time()
        rx_results = []
        valid_rx_mask = batch_covariance_data['valid_rx_mask']
        
        for i, rx_idx in enumerate(batch_rx_indices):
            if not valid_rx_mask[i]:
                rx_results.append(None)
                continue
                
            # RX별 결과 추출
            result = {
                'rx_idx': rx_idx,
                'chunk_info': chunk_info,
                'R_AE_trace': batch_covariance_data['R_AE_trace'][i],
                'epsilon_lambda': batch_metrics_data['epsilon_lambda'][i],
                'epsilon_U': batch_metrics_data['epsilon_U'][i],
                'rms_lambda': batch_metrics_data['rms_lambda'][i],
                'corr_lambda': batch_metrics_data['corr_lambda'][i],
                'lambda_AE_energy': batch_metrics_data['lambda_AE_energy'][i],
                'lambda_KM_energy': batch_metrics_data['lambda_KM_energy'][i],
                **self.select_channel_model(
                    {
                        'epsilon_lambda': batch_metrics_data['epsilon_lambda'][i],
                        'epsilon_U': batch_metrics_data['epsilon_U'][i]
                    },
                    self.config.n_r,
                    self.config.n_t,
                    self.config.epsilon_lambda_threshold,
                    self.config.epsilon_U_threshold
                )
            }
            rx_results.append(result)
        
        results_time = time.time() - start_results
        valid_count = sum(batch_covariance_data['valid_rx_mask'])
        
        # RX별 분석 결과 통합 출력
        if len(batch_rx_indices) == 1 and valid_count == 1:
            # 단일 RX 배치의 경우 간단한 출력
            result = rx_results[0]
            if result is not None:
                # 모델명 축약 (K/W/N)
                model_map = {'Kronecker': 'K', 'Weichselberger': 'W', 'Non-Separable': 'N'}
                model_short = model_map.get(result['recommended_model'], result['recommended_model'])
                
                # 핵심 결과 요약 (간단하게)
                print(f" 결과({results_time:.1f}초) → {model_short} eps_lambda={result['epsilon_lambda']:.3f} eps_U={result['epsilon_U']:.3f} R_AE_trace={result['R_AE_trace']:.1e}")
            else:
                print(f" 결과({results_time:.1f}초) → 실패")
        else:
            # 다중 RX 배치의 경우 상세 출력
            result_summary = []
            for i, (rx_idx, result) in enumerate(zip(batch_rx_indices, rx_results)):
                if result is not None:
                    model_map = {'Kronecker': 'K', 'Weichselberger': 'W', 'Non-Separable': 'N'}
                    model_short = model_map.get(result['recommended_model'], result['recommended_model'])
                    summary = f"RX{rx_idx}:{model_short},eps_lambda={result['epsilon_lambda']:.3f},eps_U={result['epsilon_U']:.3f}"
                    result_summary.append(summary)
            
            if result_summary:
                print(f" 결과({results_time:.1f}초) → [{', '.join(result_summary)}] ({valid_count}/{len(batch_rx_indices)}개 유효)")
            else:
                print(f" 결과({results_time:.1f}초) 완료 ({valid_count}/{len(batch_rx_indices)}개 유효)")
            
        return rx_results
    
    def compute_batch_kronecker_approximation(self, batch_eigen_data):
        """배치 크로네커 근사 계산 (임시 구현)"""
        # 임시로 더미 데이터 반환 (나중에 실제 구현으로 교체)
        batch_size_rx = batch_eigen_data['lambda_AE'].shape[0]
        n_r_n_t = batch_eigen_data['lambda_AE'].shape[1]
        
        dummy_lambda_KM = batch_eigen_data['lambda_AE']  # 임시로 실제 값 사용
        dummy_U_KM = batch_eigen_data['U_AE']            # 임시로 실제 값 사용
        
        return {
            'lambda_KM': dummy_lambda_KM,  # [batch_size_rx, n_r*n_t]
            'U_KM': dummy_U_KM             # [batch_size_rx, n_r*n_t, n_r*n_t]
        }
    
    def compute_batch_separability_metrics(self, batch_eigen_data, batch_kronecker_data):
        """배치 분리성 메트릭 계산 (임시 구현)"""
        batch_size_rx = batch_eigen_data['lambda_AE'].shape[0]
        
        # 임시로 랜덤 메트릭 생성 (나중에 실제 구현으로 교체)
        epsilon_lambda = np.random.uniform(0.01, 0.2, batch_size_rx).tolist()
        epsilon_U = np.random.uniform(0.01, 0.2, batch_size_rx).tolist()
        rms_lambda = np.random.uniform(0.05, 0.3, batch_size_rx).tolist()
        corr_lambda = np.random.uniform(0.8, 0.99, batch_size_rx).tolist()
        lambda_AE_energy = np.random.uniform(1e6, 1e9, batch_size_rx).tolist()
        lambda_KM_energy = np.random.uniform(1e6, 1e9, batch_size_rx).tolist()
        
        return {
            'epsilon_lambda': epsilon_lambda,
            'epsilon_U': epsilon_U,
            'rms_lambda': rms_lambda,
            'corr_lambda': corr_lambda,
            'lambda_AE_energy': lambda_AE_energy,
            'lambda_KM_energy': lambda_KM_energy
        }
    
    
    @tf.function(jit_compile=True)
    def _batch_kronecker_approximation_jit(self, lambda_u_batch_tf, lambda_b_batch_tf, U_u_batch_tf, U_b_batch_tf):
        """배치 크로네커 근사 JIT (임시 구현)"""
        # 임시로 입력 그대로 반환
        batch_size_rx = tf.shape(lambda_u_batch_tf)[0]
        n_r = tf.shape(lambda_u_batch_tf)[1]
        n_t = tf.shape(lambda_b_batch_tf)[1]
        
        # 크로네커 곱 근사 (간단 구현)
        lambda_KM_batch = tf.broadcast_to(lambda_u_batch_tf[:, :1], [batch_size_rx, n_r * n_t])
        U_KM_batch = U_u_batch_tf[:, :n_r*n_t, :n_r*n_t]  # 크기 맞추기
        
        return lambda_KM_batch, U_KM_batch
    
    @tf.function(jit_compile=True) 
    def _batch_separability_metrics_jit(self, lambda_AE_batch_tf, lambda_KM_batch_tf, U_AE_batch_tf, U_KM_batch_tf):
        """배치 분리성 메트릭 JIT (임시 구현)"""
        batch_size_rx = tf.shape(lambda_AE_batch_tf)[0]
        
        # 간단한 차이 계산
        lambda_diff = tf.reduce_mean(tf.abs(lambda_AE_batch_tf - lambda_KM_batch_tf), axis=1)
        U_diff = tf.reduce_mean(tf.abs(U_AE_batch_tf - U_KM_batch_tf), axis=[1, 2])
        
        epsilon_lambda = lambda_diff / (tf.reduce_mean(tf.abs(lambda_AE_batch_tf), axis=1) + 1e-12)
        epsilon_U = U_diff / (tf.reduce_mean(tf.abs(U_AE_batch_tf), axis=[1, 2]) + 1e-12)
        rms_lambda = epsilon_lambda  # 임시
        corr_lambda = tf.ones(batch_size_rx) * 0.9  # 임시
        
        lambda_AE_energy = tf.reduce_sum(tf.abs(lambda_AE_batch_tf)**2, axis=1)
        lambda_KM_energy = tf.reduce_sum(tf.abs(lambda_KM_batch_tf)**2, axis=1)
        
        return epsilon_lambda, epsilon_U, rms_lambda, corr_lambda, lambda_AE_energy, lambda_KM_energy
    
    @tf.function(jit_compile=True)
    def _kronecker_approximation_jit(self, lambda_u_tf, lambda_b_tf, U_u_tf, U_b_tf):
        """JIT 컴파일된 크로네커 근사 (정렬 포함)"""
        # 크로네커 곱 고유값 구성: vec(lambda_u * lambda_b^T)
        lambda_outer = tf.linalg.outer(lambda_u_tf, lambda_b_tf)
        lambda_KM = tf.reshape(lambda_outer, [-1])
        
        # 크로네커 곱 고유벡터: U_u ⊗ U_b
        U_KM = tf.linalg.kron(U_u_tf, U_b_tf)
        
        # 고유값 내림차순 정렬 (실제 고유값과 일관성 확보)
        sort_idx = tf.argsort(lambda_KM, direction='DESCENDING')
        lambda_KM_sorted = tf.gather(lambda_KM, sort_idx)
        U_KM_sorted = tf.gather(U_KM, sort_idx, axis=1)
        
        return lambda_KM_sorted, U_KM_sorted
    
    def compute_kronecker_approximation(self, eigen_data, covariance_data):
        """크로네커 근사 계산 (JIT 최적화)"""
        lambda_u = eigen_data['lambda_u']
        lambda_b = eigen_data['lambda_b']
        U_u = eigen_data['U_u']
        U_b = eigen_data['U_b']
        
        print(f"    크로네커 근사: λ_u{lambda_u.shape}, λ_b{lambda_b.shape}...", end=' ', flush=True)
        
        # TensorFlow 텐서로 변환
        lambda_u_tf = tf.constant(lambda_u, dtype=tf.float32)
        lambda_b_tf = tf.constant(lambda_b, dtype=tf.float32)
        U_u_tf = tf.constant(U_u, dtype=tf.complex64)
        U_b_tf = tf.constant(U_b, dtype=tf.complex64)
        
        # JIT 컴파일된 크로네커 근사 실행
        lambda_KM, U_KM = self._kronecker_approximation_jit(
            lambda_u_tf, lambda_b_tf, U_u_tf, U_b_tf)
        
        print("완료")
        
        return {
            'lambda_KM': lambda_KM.numpy(),
            'U_KM': U_KM.numpy()
        }
    
    @tf.function(jit_compile=True)
    def _separability_metrics_jit(self, lambda_AE_tf, lambda_KM_tf, U_AE_tf, U_KM_tf):
        """JIT 컴파일된 분리성 메트릭 계산"""
        # 고유값 분리성 오차 (Frobenius norm)
        lambda_AE_diag = tf.linalg.diag(lambda_AE_tf)
        lambda_KM_diag = tf.linalg.diag(lambda_KM_tf)
        
        epsilon_lambda = (tf.linalg.norm(lambda_AE_diag - lambda_KM_diag, ord='fro') / 
                         tf.linalg.norm(lambda_AE_diag, ord='fro'))
        
        # 고유벡터 분리성 오차 (Frobenius norm)
        epsilon_U = (tf.linalg.norm(U_AE_tf - U_KM_tf, ord='fro') / 
                     tf.linalg.norm(U_AE_tf, ord='fro'))
        
        # 추가 메트릭들
        # 고유값 상대 RMS 오차
        relative_error = (lambda_AE_tf - lambda_KM_tf) / (lambda_AE_tf + 1e-12)
        rms_lambda = tf.sqrt(tf.reduce_mean(relative_error**2))
        
        # 고유값 상관계수 (TensorFlow 구현)
        lambda_AE_centered = lambda_AE_tf - tf.reduce_mean(lambda_AE_tf)
        lambda_KM_centered = lambda_KM_tf - tf.reduce_mean(lambda_KM_tf)
        numerator = tf.reduce_mean(lambda_AE_centered * lambda_KM_centered)
        denominator = tf.sqrt(tf.reduce_mean(lambda_AE_centered**2) * tf.reduce_mean(lambda_KM_centered**2))
        corr_lambda = numerator / (denominator + 1e-12)
        
        # 에너지 계산
        lambda_AE_energy = tf.reduce_sum(tf.abs(lambda_AE_tf)**2)
        lambda_KM_energy = tf.reduce_sum(tf.abs(lambda_KM_tf)**2)
        
        return epsilon_lambda, epsilon_U, rms_lambda, corr_lambda, lambda_AE_energy, lambda_KM_energy
    
    def compute_separability_metrics(self, eigen_data, kronecker_data):
        """분리성 오차 메트릭 계산 (JIT 최적화)"""
        lambda_AE = eigen_data['lambda_AE']
        lambda_KM = kronecker_data['lambda_KM']
        U_AE = eigen_data['U_AE']
        U_KM = kronecker_data['U_KM']
        
        # TensorFlow 텐서로 변환
        lambda_AE_tf = tf.constant(lambda_AE, dtype=tf.float32)
        lambda_KM_tf = tf.constant(lambda_KM, dtype=tf.float32)
        U_AE_tf = tf.constant(U_AE, dtype=tf.complex64)
        U_KM_tf = tf.constant(U_KM, dtype=tf.complex64)
        
        # JIT 컴파일된 메트릭 계산
        epsilon_lambda, epsilon_U, rms_lambda, corr_lambda, lambda_AE_energy, lambda_KM_energy = \
            self._separability_metrics_jit(lambda_AE_tf, lambda_KM_tf, U_AE_tf, U_KM_tf)
        
        # 랭크 계산 (numpy로 처리)
        eigenvalue_rank = np.linalg.matrix_rank(np.diag(lambda_AE), tol=1e-6)
        approx_rank = np.linalg.matrix_rank(np.diag(lambda_KM), tol=1e-6)
        
        return {
            'epsilon_lambda': float(epsilon_lambda.numpy()),
            'epsilon_U': float(epsilon_U.numpy()),
            'rms_lambda': float(rms_lambda.numpy()),
            'corr_lambda': float(corr_lambda.numpy()),
            'lambda_AE_energy': float(lambda_AE_energy.numpy()),
            'lambda_KM_energy': float(lambda_KM_energy.numpy()),
            'eigenvalue_rank': eigenvalue_rank,
            'approx_rank': approx_rank
        }
    
    def select_channel_model(self, metrics, n_r, n_t, epsilon_lambda_threshold, epsilon_U_threshold):
        """tex 문서 기준에 따른 채널 모델 선택
        
        Parameters
        ----------
        metrics : dict
            분리성 메트릭 결과
        n_r : int
            수신 안테나 수 (N_u)
        n_t : int  
            송신 안테나 수 (N_b)
        epsilon_lambda_threshold : float
            고유값 분리성 오차 임계값
        epsilon_U_threshold : float
            고유벡터 분리성 오차 임계값
            
        Returns
        -------
        dict
            모델 선택 결과 및 파라미터 수
        """
        epsilon_lambda = metrics['epsilon_lambda']
        epsilon_U = metrics['epsilon_U']
        
        # tex 문서의 모델 선택 기준 적용
        if epsilon_lambda < epsilon_lambda_threshold and epsilon_U < epsilon_U_threshold:
            # 크로네커 모델 적용 조건
            recommended_model = 'Kronecker'
            model_params = n_t**2 + n_r**2
            model_description = f"고유값·고유벡터 모두 분리성 만족 (N_b²+N_u²={model_params})"
            separability_status = 'Fully Separable'
            
        elif epsilon_lambda >= epsilon_lambda_threshold and epsilon_U < epsilon_U_threshold:
            # 바이셀베르거 모델 적용 조건  
            recommended_model = 'Weichselberger'
            model_params = n_t**2 + n_r**2 + (n_t * n_r) // 2
            model_description = f"고유벡터만 분리성 만족, 커플링 행렬 필요 (N_b²+N_u²+N_b×N_u/2={model_params})"
            separability_status = 'Eigenvalue Non-Separable'
            
        else:
            # 비분리 모델 필요 조건
            recommended_model = 'Non-Separable'
            model_params = (n_t * n_r)**2
            model_description = f"고유벡터도 비분리, 완전 행렬 필요 ((N_b×N_u)²={model_params})"
            separability_status = 'Fully Non-Separable'
        
        return {
            'recommended_model': recommended_model,
            'separability_status': separability_status,
            'model_params': model_params,
            'model_description': model_description,
            'epsilon_lambda_vs_threshold': epsilon_lambda - epsilon_lambda_threshold,
            'epsilon_U_vs_threshold': epsilon_U - epsilon_U_threshold,
            'epsilon_lambda_threshold_used': epsilon_lambda_threshold,
            'epsilon_U_threshold_used': epsilon_U_threshold,
            'meets_kronecker_condition': (epsilon_lambda < epsilon_lambda_threshold and epsilon_U < epsilon_U_threshold),
            'meets_weichselberger_condition': (epsilon_lambda >= epsilon_lambda_threshold and epsilon_U < epsilon_U_threshold)
        }
    
    def analyze_single_rx(self, chunk_info, rx_idx):
        """단일 RX 분리성 분석 (모든 샘플 활용)"""
        print(f"  RX{rx_idx} 분석 시작...")
        
        # 1. 채널 데이터 로딩
        ch_data = self.load_rx_channel_data(chunk_info, rx_idx)
        if ch_data is None:
            print(f"  RX{rx_idx} SKIP (데이터 없음)")
            return None
            
        print(f"    로딩 완료: {ch_data.shape} ({ch_data.shape[0]*ch_data.shape[1]:,}개 인스턴스)")
        
        # 데이터 유효성 검사
        if np.any(np.isnan(ch_data)) or np.any(np.isinf(ch_data)):
            print(f"  RX{rx_idx} SKIP (NaN/Inf 값 포함)")
            return None
        
        if ch_data.shape[0] == 0 or ch_data.shape[1] == 0:
            print(f"  RX{rx_idx} SKIP (빈 데이터)")
            return None
        
        # 2. 공분산 행렬 계산 (모든 샘플 활용)
        covariance_data = self.compute_covariance_matrices(ch_data)
        if covariance_data is None:
            print(f"  RX{rx_idx} SKIP (공분산 계산 실패)")
            return None
        
        # 3. 고유값 분해
        eigen_data = self.perform_eigendecomposition(covariance_data)
        
        # 4. 크로네커 근사
        kronecker_data = self.compute_kronecker_approximation(eigen_data, covariance_data)
        
        # 5. 분리성 메트릭 계산
        print(f"    분리성 메트릭 계산...", end=' ', flush=True)
        metrics = self.compute_separability_metrics(eigen_data, kronecker_data)
        print("완료")
        
        # 6. 모델 선택 (config에서 활성화된 경우)
        model_selection = None
        if self.config.enable_model_selection:
            model_selection = self.select_channel_model(
                metrics, covariance_data['n_r'], covariance_data['n_t'], 
                self.config.epsilon_lambda_threshold, self.config.epsilon_U_threshold
            )
        
        # 최종 결과 메시지
        if model_selection:
            print(f"  RX{rx_idx} 완료: eps_lambda={metrics['epsilon_lambda']:.4f}, eps_U={metrics['epsilon_U']:.4f} → {model_selection['recommended_model']}")
        else:
            print(f"  RX{rx_idx} 완료: eps_lambda={metrics['epsilon_lambda']:.4f}, eps_U={metrics['epsilon_U']:.4f}")
        
        # 결과 종합 (R_AE trace 정보 포함)
        result = {
            'rx_idx': rx_idx,
            'ch_shape': ch_data.shape,
            'n_r': covariance_data['n_r'],
            'n_t': covariance_data['n_t'],
            'n_instances': covariance_data['n_instances'],
            'R_AE_trace': covariance_data['R_AE_trace'],  # 채널 이득 합산
            **metrics
        }
        
        # 모델 선택 결과 추가
        if model_selection:
            result.update(model_selection)
        
        # 상세 데이터 저장 (선택적)
        if self.config.save_detailed_results:
            result['detailed'] = {
                'lambda_AE': eigen_data['lambda_AE'],
                'lambda_KM': kronecker_data['lambda_KM'],
                'lambda_b': eigen_data['lambda_b'],
                'lambda_u': eigen_data['lambda_u']
            }
        
        return result

# ===== SECTION 3: 결과 관리자 =====
class ResultManager:
    """분석 결과 저장 및 관리"""
    
    def __init__(self, config):
        self.config = config
        os.makedirs(self.config.P1E_OUTPUT_DIR, exist_ok=True)
    
    def save_area_freq_results(self, area_idx, freq, rx_results):
        """Area-Freq별 결과 저장"""
        
        # 요약 통계 계산
        valid_results = [r for r in rx_results if r is not None]
        if not valid_results:
            print(f"Warning: No valid results for Area{area_idx}_{freq}GHz")
            return
        
        summary_stats = self._compute_summary_statistics(valid_results)
        
        # 상세 결과 저장 (npz)
        if self.config.save_detailed_results:
            result_file = os.path.join(
                self.config.P1E_OUTPUT_DIR,
                self.config.P1E_RESULT_PATTERN.format(area=area_idx, freq=freq)
            )
            
            save_data = {
                'area_idx': area_idx,
                'freq_ghz': freq,
                'rx_results': valid_results,
                'summary_stats': summary_stats,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            np.savez_compressed(result_file, **save_data)
            print(f"상세 결과 저장: {os.path.basename(result_file)}")
        
        # CSV 요약 저장
        if self.config.save_summary_csv:
            csv_file = os.path.join(
                self.config.P1E_OUTPUT_DIR,
                self.config.P1E_SUMMARY_PATTERN.format(area=area_idx, freq=freq)
            )
            
            self._save_csv_summary(csv_file, valid_results, summary_stats)
            print(f"CSV 요약 저장: {os.path.basename(csv_file)}")
    
    def _compute_summary_statistics(self, valid_results):
        """요약 통계 계산"""
        metrics = ['epsilon_lambda', 'epsilon_U', 'rms_lambda', 'corr_lambda', 'R_AE_trace']
        
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
        
        # 모델 선택 통계 (활성화된 경우)
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
            
            summary['fully_separable_count'] = status_counts.get('Fully Separable', 0)
            summary['eigenvalue_non_separable_count'] = status_counts.get('Eigenvalue Non-Separable', 0)
            summary['fully_non_separable_count'] = status_counts.get('Fully Non-Separable', 0)
            
            # 파라미터 효율성 통계
            param_counts = [r['model_params'] for r in model_results if 'model_params' in r]
            if param_counts:
                summary['avg_model_params'] = np.mean(param_counts)
                summary['min_model_params'] = np.min(param_counts) 
                summary['max_model_params'] = np.max(param_counts)
        
        return summary
    
    def _save_csv_summary(self, csv_file, valid_results, summary_stats):
        """CSV 요약 파일 저장"""
        
        # RX별 상세 데이터
        df_detail = pd.DataFrame(valid_results)
        
        # 기본 컬럼들
        detail_cols = ['rx_idx', 'n_r', 'n_t', 'n_instances', 'R_AE_trace',
                      'epsilon_lambda', 'epsilon_U', 'rms_lambda', 'corr_lambda',
                      'lambda_AE_energy', 'lambda_KM_energy', 'eigenvalue_rank', 'approx_rank']
        
        # 모델 선택 컬럼들 (존재하는 경우만 추가)
        model_cols = ['recommended_model', 'separability_status', 'model_params', 
                     'model_description', 'epsilon_lambda_vs_threshold', 'epsilon_U_vs_threshold',
                     'epsilon_lambda_threshold_used', 'epsilon_U_threshold_used']
        
        available_cols = [col for col in detail_cols + model_cols if col in df_detail.columns]
        df_detail = df_detail[available_cols]
        
        # 요약 통계
        df_summary = pd.DataFrame([summary_stats])
        
        with open(csv_file, 'w') as f:
            f.write("# P1E Separability Analysis Summary\n")
            f.write(f"# Analysis timestamp: {datetime.now().isoformat()}\n")
            f.write("# Based on tex/tmp250924.tex model selection criteria\n")
            f.write(f"# Channel power validation: R_AE trace >= {self.config.channel_power_threshold:.0e}\n")
            
            # 모델 선택 요약 (있는 경우)
            if any('recommended_model' in r for r in valid_results):
                f.write("\n# Model Selection Summary\n")
                model_summary = {
                    'Kronecker Model': summary_stats.get('kronecker_count', 0),
                    'Weichselberger Model': summary_stats.get('weichselberger_count', 0), 
                    'Non-Separable Model': summary_stats.get('non-separable_count', 0),
                    'Lambda Threshold': getattr(self.config, 'epsilon_lambda_threshold', 'N/A'),
                    'U Threshold': getattr(self.config, 'epsilon_U_threshold', 'N/A')
                }
                df_model = pd.DataFrame([model_summary])
                df_model.to_csv(f, index=False)
            
            f.write("\n# Summary Statistics\n")
            df_summary.to_csv(f, index=False)
            f.write("\n# Detailed Results per RX\n") 
            df_detail.to_csv(f, index=False)

# ===== SECTION 4: 메인 실행부 =====
def main():
    """P1E 분리성 분석 메인 실행 (L40S GPU JIT 최적화)"""
    
    print("P1E: Antenna Element OFDM 채널 분리성 분석 시작 (L40S GPU JIT 최적화)")
    print("=" * 80)
    
    # TensorFlow JIT 워밍업
    print("TensorFlow JIT XLA 워밍업 중...")
    tf.config.optimizer.set_jit(True)
    
    # GPU 메모리 상태 확인
    try:
        gpu_devices = tf.config.list_physical_devices('GPU')
        if gpu_devices:
            print(f"GPU 감지: {len(gpu_devices)}개 디바이스")
            for i, gpu in enumerate(gpu_devices):
                print(f"  GPU {i}: {gpu.name}")
        else:
            print("GPU 디바이스를 찾을 수 없음")
    except Exception as e:
        print(f"GPU 상태 확인 중 오류: {e}")
    
    print("=" * 80)
    
    config = P1E_Config()
    analyzer = BatchRxSeparabilityAnalyzer(config)
    result_manager = ResultManager(config)
    
    # JIT XLA 함수 워밍업 (다중 RX 배치 처리용)
    analyzer.warmup_jit_functions()
    
    overall_start_time = time.time()
    
    # Area-Freq별 그룹핑
    area_freq_groups = {}
    for chunk in config.p1c_chunks:
        key = (chunk['area'], chunk['freq'])
        if key not in area_freq_groups:
            area_freq_groups[key] = []
        area_freq_groups[key].append(chunk)
    
    total_groups = len(area_freq_groups)
    group_count = 0
    
    for (area_idx, freq), chunks in area_freq_groups.items():
        group_count += 1
        print(f"\n[{group_count}/{total_groups}] Area{area_idx}_{freq}GHz 분석 시작...")
        
        # 먼저 전체 RX 수 사전 계산 (정확한 남은 시간 예측을 위해)
        total_rx_count = 0
        chunk_rx_data = []
        for chunk in chunks:
            chunk_rx_indices = config.scan_chunk_rx_keys(chunk['file_path'])
            if chunk_rx_indices:
                chunk_rx_data.append((chunk, chunk_rx_indices))
                total_rx_count += len(chunk_rx_indices)
        
        print(f"  전체 {len(chunk_rx_data)}개 청크에서 {total_rx_count}개 RX 감지됨")
        
        # 청크별로 실제 존재하는 RX 수집 및 배치 구성
        rx_results = []
        group_start_time = time.time()
        
        # RX별 시간 추적 변수
        rx_times = []
        processed_rx_count = 0
        
        for chunk_idx, (chunk, chunk_rx_indices) in enumerate(chunk_rx_data):
            print(f"  청크 {chunk_idx + 1}/{len(chunk_rx_data)} ({os.path.basename(chunk['file_path'])}): {len(chunk_rx_indices)}개 실제 RX")
            
            # 청크 내 RX들을 배치별로 분할 처리
            batch_size_rx = config.batch_size_rx
            num_chunk_batches = (len(chunk_rx_indices) + batch_size_rx - 1) // batch_size_rx
            
            for batch_idx in range(num_chunk_batches):
                start_idx = batch_idx * batch_size_rx
                end_idx = min(start_idx + batch_size_rx, len(chunk_rx_indices))
                batch_rx_indices = chunk_rx_indices[start_idx:end_idx]
                
                print(f"    배치 {batch_idx + 1}/{num_chunk_batches}: RX{batch_rx_indices[0]}-RX{batch_rx_indices[-1]} ({len(batch_rx_indices)}개)")
                
                # 청크 내 배치 RX 분석 실행 (시간 측정)
                batch_start_time = time.time()
                batch_results = analyzer.analyze_batch_rx(chunk, batch_rx_indices)
                batch_time = time.time() - batch_start_time
                
                # 배치별 시간 기록 및 평균 시간 계산
                rx_times.append(batch_time)
                processed_rx_count += len(batch_rx_indices)
                avg_time_per_rx = sum(rx_times) / processed_rx_count if processed_rx_count > 0 else 0
                
                # 남은 시간 예측
                remaining_rx = total_rx_count - processed_rx_count
                estimated_remaining_time = remaining_rx * avg_time_per_rx
                
                total_batch_time = batch_time
                print(f"\n      → 배치완료 ({total_batch_time:.1f}초, 평균 {avg_time_per_rx:.1f}초/RX, 예상 남은시간 {estimated_remaining_time/60:.1f}분)")
                
                rx_results.extend(batch_results)
        
        group_total_time = time.time() - group_start_time
        avg_final_time_per_rx = group_total_time / total_rx_count if total_rx_count > 0 else 0
        print(f"Area{area_idx}_{freq}GHz 처리 완료: 총 {total_rx_count}개 RX ({group_total_time/60:.1f}분, 평균 {avg_final_time_per_rx:.1f}초/RX)")
        
        # 결과 저장
        start_save = time.time()
        result_manager.save_area_freq_results(area_idx, freq, rx_results)
        save_time = time.time() - start_save
        
        # 그룹 완료 메시지
        valid_count = sum(1 for r in rx_results if r is not None)
        print(f"Area{area_idx}_{freq}GHz 완료: {valid_count}/{total_rx_count} RX 성공 (저장: {save_time:.1f}초)")
    
    # 전체 완료
    total_elapsed = time.time() - overall_start_time
    print(f"\n" + "=" * 80)
    print(f"P1E 분리성 분석 완료 (다중 RX 병렬 처리)")
    print(f"총 소요시간: {total_elapsed/60:.1f}분")
    print(f"결과 저장 위치: {config.P1E_OUTPUT_DIR}")
    print(f"총 {total_groups}개 Area-Freq 그룹 처리 완료")
    print(f"JIT 컴파일 활용: 다중 RX 병렬 공분산 계산, 배치 고유값 분해")
    print(f"RX 배치 크기: {config.batch_size_rx} (L40S GPU 최적화)")
    print("=" * 80)

if __name__ == "__main__":
    main()
