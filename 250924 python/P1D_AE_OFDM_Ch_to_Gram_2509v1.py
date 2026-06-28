# ======================================================================
# P1D_AE_OFDM_Ch_to_Gram_2509.py
# P1D: Antenna Element OFDM Channel → Gram Matrix 변환
# 입력: P1C 결과 (AE OFDM 채널 데이터)
# 출력: Gram matrix 샘플들 (평균 채널 용량 계산용)
# 
# P1D 특징:
# - P1C AE OFDM 채널 결과를 입력으로 사용
# - 각 심볼, 각 부반송파에 대해 Gram matrix G = H × H^H 계산
# - 메모리 효율적 배치 처리
# - 평균 채널 용량 계산을 위한 Gram matrix 저장
# ======================================================================

# ===== SECTION 1: 환경 설정 =====
import os
import time
from datetime import datetime
import numpy as np
import glob
import re
from pathlib import Path

# TensorFlow 설정
import tensorflow as tf
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc'
gpu_num = 0
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

gpus = tf.config.list_physical_devices('GPU')
for g in gpus:
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except RuntimeError as e:
        print(e)

tf.get_logger().setLevel("ERROR")
tf.config.optimizer.set_jit(True)
tf.config.threading.set_inter_op_parallelism_threads(0)
tf.config.threading.set_intra_op_parallelism_threads(0)
tf.random.set_seed(1)

# 주피터/IPython 화면 클리어 지원
try:
    from IPython.display import clear_output
    JUPYTER_AVAILABLE = True
except ImportError:
    JUPYTER_AVAILABLE = False

# ===== SECTION 2: P1D_Config =====
class P1D_Config:
    """P1C AE OFDM 채널 데이터를 기반으로 Gram matrix 계산 설정"""
    
    def __init__(self):
        # 파일 경로 및 명명 패턴 설정
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        self.P1C_INPUT_DIR = os.path.join(script_dir, "P1C_AE_OFDM_Ch_Results")
        self.P1C_FILE_PATTERN = "Area{area}_{freq}GHz_RX{rx}_ae_ofdm_ch.npy"
        self.P1C_CHUNK_PATTERN = "Area{area}_{freq}GHz_AE_OFDM_Ch_RX{start}-{end}.npz"
        
        self.P1D_OUTPUT_DIR = os.path.join(script_dir, "P1D_Gram_Matrix_Results")
        self.P1D_FILE_PATTERN = "Area{area}_{freq}GHz_RX{rx}_gram_matrix.npy"
        self.P1D_CHUNK_PATTERN = "Area{area}_{freq}GHz_Gram_Matrix_RX{start}-{end}.npz"
        
        # 처리 설정
        self.BATCH_SIZE = 100  # 메모리 효율적 배치 처리 (심볼 단위)
        self.USE_CHUNKS = True  # P1C 청크 파일 우선 사용
        self.ENABLE_POST_CHUNK = True  # 후처리 청크 생성
        self.CHUNK_SIZE = 100  # 청크당 RX 개수
        self.CLEANUP_INDIVIDUAL_FILES = True
        
        # 콘솔 출력 제어
        self.ENABLE_SCREEN_CLEAR = True
        self.PROGRESS_CLEAR_INTERVAL = 10
        
        # Area 필터링 설정
        self.target_areas = [1]
        
        # P1C 데이터 스캔
        self.detect_p1c_data()
        
        # P1C에서 가져온 안테나 배열 정보 (Gram matrix 크기 계산용)
        self.N_r = 16   # RX 안테나 수: 4×4×1×1
        self.N_t = 1024 # TX 안테나 수: 4×4×8×8
        self.OFDM_FFT = 1
        
        print(f"P1D Gram Matrix 계산 설정:")
        print(f"  - RX 안테나 수 (N_r): {self.N_r}")
        print(f"  - TX 안테나 수 (N_t): {self.N_t}")
        print(f"  - Gram Matrix 크기: [{self.N_r}, {self.N_r}]")
        print(f"  - 배치 크기: {self.BATCH_SIZE} 심볼")
        print(f"  - 청크 파일 우선 사용: {self.USE_CHUNKS}")
    
    def detect_p1c_data(self):
        """P1C AE OFDM 채널 데이터를 스캔하여 처리 대상 자동 감지"""
        
        # 청크 파일 우선 스캔
        chunk_files = []
        if self.USE_CHUNKS:
            chunk_pattern = f"{self.P1C_INPUT_DIR}/{self.P1C_CHUNK_PATTERN.format(area='*', freq='*', start='*', end='*')}"
            chunk_files = glob.glob(chunk_pattern)
        
        # 개별 파일 스캔
        individual_pattern = f"{self.P1C_INPUT_DIR}/{self.P1C_FILE_PATTERN.format(area='*', freq='*', rx='*')}"
        individual_files = glob.glob(individual_pattern)
        
        # 조합 저장을 위한 집합
        combinations = set()
        
        # 청크 파일에서 조합 추출
        for file_path in chunk_files:
            filename = os.path.basename(file_path)
            # Area{X}_{freq}GHz_AE_OFDM_Ch_RX{start}-{end}.npz 패턴 매칭
            match = re.match(r'Area(\d+)_(.+)GHz_AE_OFDM_Ch_RX(\d+)-(\d+)\.npz', filename)
            if match:
                area_index = int(match.group(1))
                frequency = float(match.group(2))
                start_rx = int(match.group(3))
                end_rx = int(match.group(4))
                
                # target_areas 필터링 적용
                if self.target_areas is not None and area_index not in self.target_areas:
                    continue
                
                # 청크 파일을 열어서 실제 RX 인덱스들 확인
                try:
                    with np.load(file_path) as data:
                        if 'rx_indices' in data:
                            rx_indices = data['rx_indices']
                            for rx_index in rx_indices:
                                combinations.add((area_index, frequency, int(rx_index), 'chunk'))
                        else:
                            print(f"Warning: {filename} does not contain 'rx_indices' key")
                except Exception as e:
                    print(f"Warning: Failed to read chunk {filename}: {e}")
        
        # 개별 파일에서 조합 추출 (청크에 없는 것만)
        chunk_combinations = {(area, freq, rx) for area, freq, rx, _ in combinations}
        
        for file_path in individual_files:
            filename = os.path.basename(file_path)
            # Area{X}_{freq}GHz_RX{rx}_ae_ofdm_ch.npy 패턴 매칭
            match = re.match(r'Area(\d+)_(.+)GHz_RX(\d+)_ae_ofdm_ch\.npy', filename)
            if match:
                area_index = int(match.group(1))
                frequency = float(match.group(2))
                rx_index = int(match.group(3))
                
                # target_areas 필터링 적용
                if self.target_areas is not None and area_index not in self.target_areas:
                    continue
                
                # 청크에 없는 경우만 추가
                if (area_index, frequency, rx_index) not in chunk_combinations:
                    combinations.add((area_index, frequency, rx_index, 'individual'))
        
        # 정렬하여 리스트로 저장
        self.p1c_data_combinations = sorted(list(combinations))
        
        print(f"P1C AE OFDM 채널 데이터 자동 감지:")
        print(f"  - 감지된 조합 수: {len(self.p1c_data_combinations)}")
        
        # 파일 타입별로 그룹핑하여 출력
        chunk_count = sum(1 for combo in self.p1c_data_combinations if combo[3] == 'chunk')
        individual_count = sum(1 for combo in self.p1c_data_combinations if combo[3] == 'individual')
        print(f"    - 청크 파일: {chunk_count}개")
        print(f"    - 개별 파일: {individual_count}개")
        
        # Area별로 그룹핑하여 출력
        area_groups = {}
        for combo in self.p1c_data_combinations:
            area, freq, rx, file_type = combo
            key = f"Area{area}_{freq}GHz"
            if key not in area_groups:
                area_groups[key] = []
            area_groups[key].append(rx)
        
        for area_freq, rx_list in sorted(area_groups.items()):
            print(f"    - {area_freq}: RX{min(rx_list)}-RX{max(rx_list)} ({len(rx_list)} RXs)")
    
    def load_p1c_channel_data(self, area_index, frequency, rx_index, file_type):
        """P1C AE OFDM 채널 데이터 로드
        
        Parameters
        ----------
        area_index : int
            Area index
        frequency : float
            Frequency in GHz
        rx_index : int
            RX index
        file_type : str
            'chunk' or 'individual'
            
        Returns
        -------
        np.ndarray
            Channel data with shape [total_symbols, OFDM_FFT, N_r, N_t]
        """
        
        if file_type == 'chunk':
            # 청크 파일에서 데이터 로드
            # 해당 RX가 포함된 청크 파일 찾기
            chunk_pattern = f"{self.P1C_INPUT_DIR}/Area{area_index}_{frequency}GHz_AE_OFDM_Ch_RX*-*.npz"
            chunk_files = glob.glob(chunk_pattern)
            
            for chunk_file in chunk_files:
                try:
                    with np.load(chunk_file) as data:
                        if 'rx_indices' in data and rx_index in data['rx_indices']:
                            key = f'ofdm_ch_rx_{rx_index}'
                            if key in data:
                                return data[key]
                except Exception as e:
                    print(f"Warning: Failed to read from chunk {chunk_file}: {e}")
            
            print(f"Warning: RX{rx_index} not found in any chunk files")
            return None
            
        else:  # individual
            # 개별 파일에서 데이터 로드
            filename = self.P1C_FILE_PATTERN.format(area=area_index, freq=frequency, rx=rx_index)
            filepath = os.path.join(self.P1C_INPUT_DIR, filename)
            
            try:
                return np.load(filepath)
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
                return None
    
    def create_gram_chunk_from_files(self, area_index, frequency, rx_indices):
        """개별 Gram matrix npy 파일들을 npz 청크로 묶기"""
        if not rx_indices:
            return None
        
        chunk_data = {}
        chunk_metadata = {
            'area_index': area_index,
            'frequency_ghz': frequency,
            'rx_indices': np.array(rx_indices),
            'rx_count': len(rx_indices),
            'gram_matrix_shape': [self.N_r, self.N_r]
        }
        
        # 개별 파일들 로드
        total_rx = len(rx_indices)
        for idx, rx_idx in enumerate(rx_indices, 1):
            individual_file = os.path.join(self.P1D_OUTPUT_DIR,
                                         self.P1D_FILE_PATTERN.format(area=area_index, freq=frequency, rx=rx_idx))
            
            # 진행률 표시
            progress_msg = f"  Gram matrix 파일 로딩 중... RX{rx_idx} ({idx}/{total_rx})"
            print(f"\r{progress_msg:<60}", end='', flush=True)
            
            if os.path.exists(individual_file):
                gram_data = np.load(individual_file)
                chunk_data[f'gram_matrix_rx_{rx_idx}'] = gram_data
            else:
                print(f"\nWarning: {individual_file} not found")
        
        print()  # 진행률 완료 후 줄바꿈
        
        # 청크 파일 저장
        chunk_filename = self.P1D_CHUNK_PATTERN.format(
            area=area_index, freq=frequency, start=min(rx_indices), end=max(rx_indices)
        )
        chunk_filepath = os.path.join(self.P1D_OUTPUT_DIR, chunk_filename)
        
        # 메타데이터와 함께 저장
        save_data = {**chunk_metadata, **chunk_data}
        np.savez_compressed(chunk_filepath, **save_data)
        
        chunk_size_mb = os.path.getsize(chunk_filepath) / (1024**2)
        print(f"Gram matrix 청크 저장 완료: {chunk_filename} ({chunk_size_mb:.1f} MB)")
        
        # 개별 파일 정리
        if self.CLEANUP_INDIVIDUAL_FILES:
            for rx_idx in rx_indices:
                individual_file = os.path.join(self.P1D_OUTPUT_DIR,
                                             self.P1D_FILE_PATTERN.format(area=area_index, freq=frequency, rx=rx_idx))
                if os.path.exists(individual_file):
                    os.remove(individual_file)
            print(f"개별 Gram matrix 파일 {len(rx_indices)}개 정리 완료")
        
        return chunk_filepath

# ===== SECTION 3: Gram Matrix 계산 클래스 =====
class GramMatrixCalculator:
    """AE OFDM 채널로부터 Gram matrix 계산"""
    
    def __init__(self, config):
        self.config = config
    
    @tf.function(jit_compile=True)
    def compute_gram_matrix_batch(self, channel_batch):
        """배치 단위로 Gram matrix 계산
        
        Parameters
        ----------
        channel_batch : tf.Tensor
            Shape [batch_size, OFDM_FFT, N_r, N_t]
            
        Returns
        -------
        tf.Tensor
            Gram matrices with shape [batch_size, OFDM_FFT, N_r, N_r]
        """
        # channel_batch: [batch_size, OFDM_FFT, N_r, N_t]
        
        # 각 심볼, 각 부반송파에 대해 Gram matrix 계산
        # H shape: [N_r, N_t]
        # G = H × H^H shape: [N_r, N_r]
        
        # Hermitian conjugate (complex conjugate transpose)
        channel_h = tf.linalg.adjoint(channel_batch)  # [batch_size, OFDM_FFT, N_t, N_r]
        
        # Gram matrix: G = H × H^H
        gram_matrices = tf.linalg.matmul(channel_batch, channel_h)  # [batch_size, OFDM_FFT, N_r, N_r]
        
        return gram_matrices
    
    def process_channel_data(self, channel_data):
        """전체 채널 데이터를 배치 단위로 처리하여 Gram matrix 계산
        
        Parameters
        ----------
        channel_data : np.ndarray
            Shape [total_symbols, OFDM_FFT, N_r, N_t]
            
        Returns
        -------
        np.ndarray
            Gram matrices with shape [total_symbols, OFDM_FFT, N_r, N_r]
        """
        total_symbols = channel_data.shape[0]
        
        # 결과 저장을 위한 배열 사전 할당
        gram_results = np.zeros((total_symbols, self.config.OFDM_FFT, self.config.N_r, self.config.N_r), 
                               dtype=np.complex64)
        
        # 배치 단위로 처리
        for start_idx in range(0, total_symbols, self.config.BATCH_SIZE):
            end_idx = min(start_idx + self.config.BATCH_SIZE, total_symbols)
            
            # 현재 배치 추출
            batch_data = channel_data[start_idx:end_idx]
            
            # TensorFlow 텐서로 변환
            batch_tensor = tf.convert_to_tensor(batch_data, dtype=tf.complex64)
            
            # Gram matrix 계산
            gram_batch = self.compute_gram_matrix_batch(batch_tensor)
            
            # 결과 저장
            gram_results[start_idx:end_idx] = gram_batch.numpy()
            
            # 진행률 표시 (매 10배치마다)
            if (start_idx // self.config.BATCH_SIZE) % 10 == 0:
                progress = (end_idx / total_symbols) * 100
                print(f"    Gram matrix 계산 진행률: {progress:.1f}% ({end_idx}/{total_symbols})", end='\r')
        
        print()  # 진행률 완료 후 줄바꿈
        return gram_results

# ===== SECTION 4: 메인 실행 P1D =====
def main():
    """P1D: P1C AE OFDM 채널로부터 Gram matrix 계산"""
    
    config = P1D_Config()
    calculator = GramMatrixCalculator(config)
    
    # 전체 시작 시간 기록
    overall_start_time = time.time()
    
    # 처리 대상 데이터 기반 동적 루프
    total_rx_files = len(config.p1c_data_combinations)
    rx_file_count = 0
    
    # 동적 자리수 맞춤을 위한 너비 계산
    total_rx_width = len(str(total_rx_files))
    
    # 출력 디렉토리 생성
    os.makedirs(config.P1D_OUTPUT_DIR, exist_ok=True)
    
    for area_index, fc, RX_index, file_type in config.p1c_data_combinations:
        rx_file_count += 1
        
        # 시작 시간 기록
        tic_total = time.time()
        
        # P1C 채널 데이터 로딩
        channel_data = config.load_p1c_channel_data(area_index, fc, RX_index, file_type)
        if channel_data is None:
            print(f"Error: Failed to load channel data for Area{area_index}_{fc}GHz_RX{RX_index}")
            continue
        
        print(f"Area{area_index}_{fc}GHz_RX{RX_index}: Channel shape {channel_data.shape}")
        
        # Gram matrix 계산
        print("  Gram matrix 계산 중...")
        gram_matrices = calculator.process_channel_data(channel_data)
        
        # 저장
        output_filename = config.P1D_FILE_PATTERN.format(area=area_index, freq=fc, rx=RX_index)
        output_filepath = os.path.join(config.P1D_OUTPUT_DIR, output_filename)
        np.save(output_filepath, gram_matrices)
        
        toc_total = time.time()
        
        # 파일 크기 정보
        file_size_mb = os.path.getsize(output_filepath) / (1024**2)
        
        # 설정된 간격마다 또는 마지막 RX일 때 화면 클리어 + 완료 메시지 출력
        if config.ENABLE_SCREEN_CLEAR and (rx_file_count % config.PROGRESS_CLEAR_INTERVAL == 0 or rx_file_count == total_rx_files):
            # 화면 클리어
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
            print("P1D Gram Matrix 계산 진행률:")
            print(f"✓ [{rx_file_count:>{total_rx_width}}/{total_rx_files}] {output_filename} ({file_size_mb:.1f} MB)")
            print(f"현재 완료율: {rx_file_count/total_rx_files*100:.1f}% ({rx_file_count}/{total_rx_files})")
            print(f"전체 경과 시간: {format_time(overall_elapsed)}")
            print(f"평균 처리 속도: {avg_time_per_rx:.1f}초/RX")
            print(f"예상 남은 시간: {format_time(estimated_remaining)}")
            print(f"화면 클리어 주기: {config.PROGRESS_CLEAR_INTERVAL}개")
            print("-" * 80)
    
    # 모든 RX 처리 완료 후 청크 생성
    if config.ENABLE_POST_CHUNK:
        print("\nGram matrix 청크 생성 시작...")
        area_freq_groups = {}
        for area_index, frequency, rx_index, _ in config.p1c_data_combinations:
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
                config.create_gram_chunk_from_files(area_index, frequency, chunk_rx_list)
        
        print(f"\n=== P1D 후처리 청크 생성 완료 ===")
        total_chunks_created = sum((len(rx_list) + config.CHUNK_SIZE - 1) // config.CHUNK_SIZE 
                                 for rx_list in area_freq_groups.values())
        total_rx_processed = sum(len(rx_list) for rx_list in area_freq_groups.values())
        print(f"총 {total_chunks_created}개 Gram matrix 청크 생성 완료 (총 {total_rx_processed}개 RX 처리)")

if __name__ == "__main__":
    main()
