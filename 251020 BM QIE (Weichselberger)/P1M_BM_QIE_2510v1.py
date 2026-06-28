# ======================================================================
# P1M_BM_QIE_2510v1.py
# P1M: QIE Clustering for Beam Management
# 
# === 최상위 목적 ===
# P1L 빔 관리 결과 기반 UE 클러스터링
# - Inner product-based similarity metric (non-orthogonal DFT codebook)
# - Hierarchical clustering with threshold
# - QIE (Quasi-Identical Environment) 구성
# 
# === 알고리즘 구조 ===
# 1. P1L CSV 파싱: 각 UE의 최종 BS 빔 인덱스 추출
# 2. DFT 코드북 생성: BS 64개 빔 벡터
# 3. Beam similarity 계산: ρ_ij = mean(|W_i^H W_j|)
# 4. Hierarchical clustering: distance = 1 - ρ_ij
# 5. Cluster statistics & CSV 저장
# 
# === 입력/출력 ===
# 입력: P1L CSV (Area{area}_{freq}GHz_SU_BM_{timestamp}_done.csv)
# 출력: P1M CSV (Area{area}_{freq}GHz_QIE_{timestamp}.csv)
#
# === 주요 수정 이력 ===
# [251019] P1M v1 신규 작성: QIE 클러스터링
#
# ======================================================================

# ===== SECTION 1: 환경 설정 =====
import os
import time
from datetime import datetime
import numpy as np
import glob
import csv
from pathlib import Path
from typing import List, Tuple, Dict

# TensorFlow 환경 설정
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc'
gpu_num = 0
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

# GPU 메모리 설정
gpus = tf.config.list_physical_devices('GPU')
for g in gpus:
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except RuntimeError as e:
        pass

tf.get_logger().setLevel("ERROR")
tf.config.optimizer.set_jit(True)
tf.random.set_seed(42)
np.random.seed(42)

# scipy clustering
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

# ======================================================================
# 코드 구조: 의존성 기반 순서
# ======================================================================
# LEVEL 0: 독립적 기본 클래스
#   - P1M_Config: 설정 관리
#   - DFTCodebook: DFT 코드북 생성
#   - P1L_CSVParser: P1L 결과 파싱
#
# LEVEL 1: Config 의존 클래스
#   - BeamInnerProductCalculator: ρ_ij 계산
#   - P1M_ResultManager: CSV 결과 저장
#
# LEVEL 2: 복합 의존 클래스
#   - QIEClusterer: Hierarchical clustering
#
# LEVEL 3: 최상위 실행
#   - main: 전체 파이프라인 실행
# ======================================================================

# ===== LEVEL 0: 독립적 기본 클래스 =====

# ----- P1M_Config -----
class P1M_Config:
    """P1M QIE 클러스터링 설정"""
    
    def __init__(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 입력 데이터 경로
        self.P1L_INPUT_DIR = os.path.join(script_dir, "P1L_BeamMgmt_Results")
        
        # 출력 결과 경로
        self.P1M_OUTPUT_DIR = os.path.join(script_dir, "P1M_QIE_Results")
        os.makedirs(self.P1M_OUTPUT_DIR, exist_ok=True)
        
        # 필터링 설정
        self.target_areas = [1]
        self.target_freqs = [7.5]
        
        # DFT 코드북 설정 (P1L과 동일)
        self.bs_ant_per_dim = 4
        self.bs_oversample = 2
        self.n_cb_bs = (self.bs_ant_per_dim * self.bs_oversample) ** 2  # 64
        self.n_ant_bs = self.bs_ant_per_dim ** 2  # 16
        
        # Clustering 설정
        self.rho_threshold = 0.3  # distance threshold = 1 - rho_threshold = 0.7
        self.linkage_method = 'average'  # 'single', 'complete', 'average', 'ward'
        
        # Strict clustering constraints
        self.min_intra_rho = 0.7
        self.max_inter_rho = 0.1
        
        # Area grid configs (from P1A)
        self.AREA_GRID_CONFIGS = {
            1: {
                'tx_position': [-51.561, -21.794, 19],
                'x_start': -136.138, 'x_stop': 58.862, 'num_x': 40,
                'y_start': -117.667, 'y_stop': 77.333, 'num_y': 40
            }
        }
        
        # P1L CSV 파일 스캔
        self.detect_p1l_csv()
    
    def detect_p1l_csv(self):
        """P1L 결과 CSV 파일 스캔"""
        self.csv_files = []
        
        for area in self.target_areas:
            for freq in self.target_freqs:
                # _done.csv 파일 찾기
                pattern = f"{self.P1L_INPUT_DIR}/Area{area}_{freq}GHz_SU_BM_*_done.csv"
                files = glob.glob(pattern)
                
                if files:
                    # 가장 최근 파일 선택
                    latest_file = max(files, key=os.path.getmtime)
                    self.csv_files.append((area, freq, latest_file))
                    print(f"P1L CSV 감지: Area{area}_{freq}GHz - {os.path.basename(latest_file)}")
        
        if not self.csv_files:
            print("경고: P1L CSV 파일이 없습니다.")

# ----- DFTCodebook -----
class DFTCodebook:
    """DFT 코드북 생성 (TensorFlow)"""
    
    @staticmethod
    def generate_1d_dft_tf(N: int, K: int) -> tf.Tensor:
        """1D DFT 코드북 (TensorFlow)
        
        [F_{N,K}]_{i,j} = (1/√N) exp(-j 2π ij / (NK))
        
        Args:
            N: 안테나 수 per 차원
            K: 오버샘플링 계수
        
        Returns:
            F: [N, NK] tf.complex64
        """
        i = tf.cast(tf.range(N)[:, None], tf.float32)
        j = tf.cast(tf.range(N * K)[None, :], tf.float32)
        pi = tf.constant(3.141592653589793, dtype=tf.float32)
        phase = -2.0 * pi * i * j / float(N * K)
        sqrt_N = tf.sqrt(tf.constant(float(N), dtype=tf.float32))
        sqrt_N_complex = tf.cast(sqrt_N, tf.complex64)
        return tf.exp(tf.complex(0.0, phase)) / sqrt_N_complex
    
    @staticmethod
    def generate_2d_dft_codebook_tf(ant_per_dim: int, oversample: int) -> tf.Tensor:
        """2D DFT 코드북 생성 (TensorFlow)
        
        F = F_{N,K} ⊗ F_{N,K} ∈ C^{N²×(NK)²}
        
        Args:
            ant_per_dim: 1차원당 안테나 수 (N)
            oversample: 오버샘플링 계수 (K)
        
        Returns:
            F: [ant_per_dim², (ant_per_dim*oversample)²] tf.complex64
        
        Example:
            BS: ant_per_dim=4, oversample=2 → F: [16, 64]
        """
        F_1d = DFTCodebook.generate_1d_dft_tf(ant_per_dim, oversample)
        
        # Kronecker product: F ⊗ F
        n_ant = ant_per_dim
        n_beams = ant_per_dim * oversample
        
        # tf.einsum: F_ij * F_kl = F_ikjl → reshape → [N²,(NK)²]
        F_kron = tf.einsum('ij,kl->ikjl', F_1d, F_1d)
        F_kron = tf.reshape(F_kron, [n_ant**2, n_beams**2])
        
        return F_kron

# ----- P1L_CSVParser -----
class P1L_CSVParser:
    """P1L 빔 관리 결과 CSV 파싱"""
    
    @staticmethod
    def parse_csv(csv_path: str) -> Dict[int, List[int]]:
        """P1L CSV에서 각 UE의 최종 BS 빔 인덱스 추출
        
        Args:
            csv_path: P1L CSV 파일 경로
        
        Returns:
            ue_beams: {ue_id: [beam_idx_1, beam_idx_2, ...]}
        """
        ue_beams = {}
        
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                ue = int(row['ue'].strip())
                stage = row['stage'].strip()
                
                # S2 greed 단계만 처리
                if stage == 'S2':
                    beams_str = row['beams'].strip().strip('"')
                    
                    # 빔 인덱스 파싱: "9,1,8" → [9, 1, 8]
                    if beams_str:
                        beam_indices = [int(b.strip()) for b in beams_str.split(',')]
                        # 각 UE의 마지막 S2 행이 최종 빔 세트
                        ue_beams[ue] = beam_indices
        
        return ue_beams

# ----- AreaGridMapper -----
class AreaGridMapper:
    """UE 인덱스 → 공간 좌표 변환"""
    
    @staticmethod
    def ue_to_coordinates(ue_id: int, area: int, config: P1M_Config) -> Tuple[float, float]:
        """UE 인덱스를 (x, y) 좌표로 변환
        
        Args:
            ue_id: UE 인덱스 (grid position)
            area: Area ID
            config: P1M_Config
        
        Returns:
            (x, y): 좌표 [m]
        """
        if area not in config.AREA_GRID_CONFIGS:
            raise ValueError(f"Area {area} not in AREA_GRID_CONFIGS")
        
        grid = config.AREA_GRID_CONFIGS[area]
        
        # Grid 좌표 생성
        x_coords = np.linspace(grid['x_start'], grid['x_stop'], grid['num_x'])
        y_coords = np.linspace(grid['y_start'], grid['y_stop'], grid['num_y'])
        
        # UE ID: 1-based → 0-based
        ue_idx = ue_id - 1
        
        # UE index → (x_idx, y_idx) row-major mapping
        x_idx = ue_idx % grid['num_x']
        y_idx = ue_idx // grid['num_x']
        
        return x_coords[x_idx], y_coords[y_idx]
    
    @staticmethod
    def get_all_ue_coordinates(ue_ids: List[int], area: int, config: P1M_Config) -> np.ndarray:
        """모든 UE의 좌표 배열 생성
        
        Args:
            ue_ids: UE ID 리스트
            area: Area ID
            config: P1M_Config
        
        Returns:
            coords: [n_ues, 2] array of (x, y) coordinates
        """
        coords = np.zeros((len(ue_ids), 2))
        for i, ue_id in enumerate(ue_ids):
            coords[i] = AreaGridMapper.ue_to_coordinates(ue_id, area, config)
        return coords

# ===== LEVEL 1: Config 의존 클래스 =====

# ----- BeamInnerProductCalculator -----
class BeamInnerProductCalculator:
    """Beam inner product 기반 similarity 계산 (TensorFlow 최적화)"""
    
    def __init__(self, F_bs: tf.Tensor):
        """
        Args:
            F_bs: BS DFT 코드북 [M, N_cb] (M=16, N_cb=64)
        """
        self.F_bs = F_bs
    
    def compute_pairwise_similarity(self, ue_beams: Dict[int, List[int]]) -> Tuple[np.ndarray, List[int]]:
        """모든 UE 쌍에 대한 similarity matrix 계산 (TensorFlow 배치 최적화)
        
        ρ_ij = mean(|W_i^H W_j|)
        
        Args:
            ue_beams: {ue_id: [beam_idx_1, ...]}
        
        Returns:
            rho_matrix: [n_ues, n_ues] similarity matrix
            ue_ids: UE ID 리스트 (matrix 인덱스 대응)
        """
        ue_ids = sorted(ue_beams.keys())
        n_ues = len(ue_ids)
        
        print(f"Pairwise similarity 계산: {n_ues}개 UE")
        
        # 1. 모든 UE의 beam matrix 준비
        print("  [1/3] Beam matrix 준비 중...")
        W_list = []
        for ue_id in ue_ids:
            beams = ue_beams[ue_id]
            W = tf.gather(self.F_bs, beams, axis=1)  # [M, L_i]
            W_list.append(W)
        
        # 2. Upper triangle 인덱스 생성 (i < j)
        n_pairs = n_ues * (n_ues - 1) // 2
        print(f"  [2/3] Similarity 계산 중... ({n_pairs}개 UE 쌍)")
        
        # Similarity matrix 초기화 (TensorFlow)
        rho_matrix_tf = tf.Variable(tf.zeros((n_ues, n_ues), dtype=tf.float32))
        
        # 대각 원소 = 1
        tf.linalg.set_diag(rho_matrix_tf, tf.ones(n_ues, dtype=tf.float32))
        
        # 3. 배치 단위로 계산 (GPU 성능 고려)
        batch_size = 50000  # L40S (48GB) 고성능 GPU - 대용량 배치
        pair_indices = []
        for i in range(n_ues):
            for j in range(i + 1, n_ues):
                pair_indices.append((i, j))
        
        n_batches = (n_pairs + batch_size - 1) // batch_size
        print(f"  배치 크기: {batch_size}, 총 배치 수: {n_batches}")
        
        import time
        start_time = time.time()
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_pairs)
            batch_pairs = pair_indices[start_idx:end_idx]
            
            # 배치 내 모든 쌍에 대해 계산
            for i, j in batch_pairs:
                rho_ij = self._compute_single_similarity_tf(W_list[i], W_list[j])
                rho_matrix_tf[i, j].assign(rho_ij)
                rho_matrix_tf[j, i].assign(rho_ij)  # 대칭
            
            # 진행률 출력 (1% 단위 또는 매 배치)
            progress = (end_idx / n_pairs) * 100
            elapsed = time.time() - start_time
            
            # 조건: 1% 단위 또는 마지막 배치
            if batch_idx % max(1, n_batches // 100) == 0 or batch_idx == n_batches - 1:
                # ETA 계산
                if end_idx > 0:
                    total_time_estimate = (elapsed / end_idx) * n_pairs
                    remaining_time = total_time_estimate - elapsed
                    eta_str = f"{remaining_time/60:.1f}m" if remaining_time >= 60 else f"{remaining_time:.0f}s"
                else:
                    eta_str = "계산 중"
                
                print(f"    진행률: {progress:.1f}% ({end_idx}/{n_pairs} 쌍) "
                      f"경과: {elapsed:.1f}s, ETA: {eta_str}")
        
        print("  [3/3] 완료")
        
        # NumPy로 변환
        rho_matrix = rho_matrix_tf.numpy()
        
        return rho_matrix, ue_ids
    
    @tf.function
    def _compute_single_similarity_tf(self, W_i: tf.Tensor, W_j: tf.Tensor) -> tf.Tensor:
        """단일 UE 쌍에 대한 similarity 계산 (TensorFlow JIT)
        
        Args:
            W_i: UE i의 beam matrix [M, L_i]
            W_j: UE j의 beam matrix [M, L_j]
        
        Returns:
            rho_ij: scalar similarity (tf.Tensor)
        """
        # R_ij = W_i^H @ W_j  # [L_i, L_j]
        R_ij = tf.linalg.matmul(W_i, W_j, adjoint_a=True)
        
        # ρ_ij = mean(|R_ij|)
        rho_ij = tf.reduce_mean(tf.abs(R_ij))
        
        return rho_ij

# ----- P1M_ResultManager -----
class P1M_ResultManager:
    """P1M QIE 클러스터링 결과 CSV 저장"""
    
    def __init__(self, config: P1M_Config):
        self.config = config
        self._csv_path = None
    
    def init_csv(self, area: int, freq: float, timestamp: str, method: str = ""):
        """CSV 파일 초기화
        
        Args:
            area: Area ID
            freq: Frequency (GHz)
            timestamp: 타임스탬프 문자열
            method: 방법 구분자 ("_ex", "_c_st", "_c_bs", "_c_rx")
        """
        # 간소화된 파일명: A1_7.5G_QIE_1019_1303_c_st.csv
        freq_str = str(freq).replace('.', 'p')  # 7.5 → 7p5
        if method:
            filename = f"A{area}_{freq_str}G_QIE_{timestamp}{method}.csv"
        else:
            filename = f"A{area}_{freq_str}G_QIE_{timestamp}.csv"
        self._csv_path = os.path.join(self.config.P1M_OUTPUT_DIR, filename)
        
        print(f"\nP1M CSV 생성: {filename}")
    
    def save_clustering_results(self, ue_ids: List[int], cluster_labels: np.ndarray,
                                ue_beams: Dict[int, List[int]], rho_matrix: np.ndarray):
        """클러스터링 결과 저장
        
        Args:
            ue_ids: UE ID 리스트
            cluster_labels: [n_ues] 클러스터 레이블
            ue_beams: {ue_id: [beam_idx_1, ...]}
            rho_matrix: [n_ues, n_ues] similarity matrix
        """
        if self._csv_path is None:
            raise ValueError("CSV path not initialized. Call init_csv() first.")
        
        # 클러스터별 통계 계산
        cluster_stats = self._compute_cluster_stats(cluster_labels, rho_matrix)
        
        # Global validation 통계 계산
        global_stats = self._compute_global_stats(cluster_labels, rho_matrix)
        
        # CSV 작성
        with open(self._csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Metadata 주석
            writer.writerow([f"# Clustering Method: {self._extract_method_name()}"])
            writer.writerow([f"# Total UEs: {len(ue_ids)}"])
            writer.writerow([f"# Total Clusters: {global_stats['n_clusters']}"])
            writer.writerow([f"# Intra-cluster ρ (mean): {global_stats['intra_mean']:.4f}"])
            writer.writerow([f"# Intra-cluster ρ (std): {global_stats['intra_std']:.4f}"])
            writer.writerow([f"# Intra-cluster ρ (min): {global_stats['intra_min']:.4f}"])
            writer.writerow([f"# Intra-cluster ρ (max): {global_stats['intra_max']:.4f}"])
            if global_stats['has_inter']:
                writer.writerow([f"# Inter-cluster ρ (mean): {global_stats['inter_mean']:.4f}"])
                writer.writerow([f"# Inter-cluster ρ (std): {global_stats['inter_std']:.4f}"])
                writer.writerow([f"# Inter-cluster ρ (min): {global_stats['inter_min']:.4f}"])
                writer.writerow([f"# Inter-cluster ρ (max): {global_stats['inter_max']:.4f}"])
                writer.writerow([f"# Separation (intra_mean - inter_mean): {global_stats['separation']:.4f}"])
            else:
                writer.writerow([f"# Inter-cluster pairs: None (single cluster)"])
            writer.writerow(["#"])
            
            # Header
            writer.writerow([
                'ue', 'cluster_id', 'bs_beam_indices', 'n_beams',
                'intra_cluster_rho_mean', 'intra_cluster_rho_max'
            ])
            
            # 각 UE 행
            for idx, ue_id in enumerate(ue_ids):
                cluster_id = int(cluster_labels[idx])
                beams = ue_beams[ue_id]
                n_beams = len(beams)
                beams_str = ','.join(map(str, beams))
                
                rho_mean = cluster_stats[cluster_id]['rho_mean']
                rho_max = cluster_stats[cluster_id]['rho_max']
                
                writer.writerow([
                    f"{ue_id:4d}",
                    f"{cluster_id:3d}",
                    beams_str,
                    f"{n_beams:2d}",
                    f"{rho_mean:.6f}",
                    f"{rho_max:.6f}"
                ])
        
        print(f"결과 저장 완료: {self._csv_path}")
    
    def _extract_method_name(self) -> str:
        """CSV 파일명에서 method 추출"""
        if self._csv_path is None:
            return "Unknown"
        filename = os.path.basename(self._csv_path)
        if '_ex' in filename:
            return "Exact Beam Set"
        elif '_c_st' in filename:
            return "Complete Linkage (strict, threshold=0.95)"
        elif '_c_bs' in filename:
            return "Complete Linkage (baseline, threshold=0.90)"
        elif '_c_rx' in filename:
            return "Complete Linkage (relaxed, threshold=0.85)"
        else:
            return "Unknown"
    
    def _compute_global_stats(self, cluster_labels: np.ndarray,
                              rho_matrix: np.ndarray) -> Dict[str, float]:
        """Global validation 통계 계산
        
        Args:
            cluster_labels: [n_ues]
            rho_matrix: [n_ues, n_ues]
        
        Returns:
            stats: global clustering statistics
        """
        unique_clusters = np.unique(cluster_labels)
        n_clusters = len(unique_clusters)
        
        # Intra-cluster similarity
        intra_rhos = []
        for cluster_id in unique_clusters:
            cluster_mask = (cluster_labels == cluster_id)
            cluster_indices = np.where(cluster_mask)[0]
            
            if len(cluster_indices) > 1:
                for i in cluster_indices:
                    for j in cluster_indices:
                        if i < j:
                            intra_rhos.append(rho_matrix[i, j])
        
        # Inter-cluster similarity
        inter_rhos = []
        for i in range(len(cluster_labels)):
            for j in range(i + 1, len(cluster_labels)):
                if cluster_labels[i] != cluster_labels[j]:
                    inter_rhos.append(rho_matrix[i, j])
        
        stats = {
            'n_clusters': n_clusters,
            'intra_mean': float(np.mean(intra_rhos)) if intra_rhos else 1.0,
            'intra_std': float(np.std(intra_rhos)) if intra_rhos else 0.0,
            'intra_min': float(np.min(intra_rhos)) if intra_rhos else 1.0,
            'intra_max': float(np.max(intra_rhos)) if intra_rhos else 1.0,
            'has_inter': len(inter_rhos) > 0
        }
        
        if inter_rhos:
            stats['inter_mean'] = float(np.mean(inter_rhos))
            stats['inter_std'] = float(np.std(inter_rhos))
            stats['inter_min'] = float(np.min(inter_rhos))
            stats['inter_max'] = float(np.max(inter_rhos))
            stats['separation'] = stats['intra_mean'] - stats['inter_mean']
        
        return stats
    
    def _compute_cluster_stats(self, cluster_labels: np.ndarray,
                               rho_matrix: np.ndarray) -> Dict[int, Dict[str, float]]:
        """클러스터별 intra-cluster 통계 계산
        
        Args:
            cluster_labels: [n_ues]
            rho_matrix: [n_ues, n_ues]
        
        Returns:
            stats: {cluster_id: {'rho_mean': float, 'rho_max': float}}
        """
        unique_clusters = np.unique(cluster_labels)
        stats = {}
        
        for cluster_id in unique_clusters:
            cluster_mask = (cluster_labels == cluster_id)
            cluster_indices = np.where(cluster_mask)[0]
            
            if len(cluster_indices) <= 1:
                # 단일 UE 클러스터
                stats[cluster_id] = {'rho_mean': 1.0, 'rho_max': 1.0}
            else:
                # Intra-cluster similarity 추출 (대각 제외)
                intra_rho = []
                for i in cluster_indices:
                    for j in cluster_indices:
                        if i < j:
                            intra_rho.append(rho_matrix[i, j])
                
                stats[cluster_id] = {
                    'rho_mean': float(np.mean(intra_rho)),
                    'rho_max': float(np.max(intra_rho))
                }
        
        return stats

# ----- SpatialMapVisualizer -----
class SpatialMapVisualizer:
    """QIE 클러스터 공간 분포 시각화"""
    
    def __init__(self, config: P1M_Config):
        self.config = config
    
    def plot_cluster_map(self, ue_ids: List[int], cluster_labels: np.ndarray,
                        area: int, freq: float, timestamp: str):
        """클러스터 공간 분포 시각화 및 저장
        
        Args:
            ue_ids: UE ID 리스트
            cluster_labels: [n_ues] 클러스터 레이블
            area: Area ID
            freq: Frequency (GHz)
            timestamp: 타임스탬프 문자열
        """
        import matplotlib.pyplot as plt
        from matplotlib import cm
        
        # UE 좌표 변환
        coords = AreaGridMapper.get_all_ue_coordinates(ue_ids, area, self.config)
        x_coords = coords[:, 0]
        y_coords = coords[:, 1]
        
        # Grid config 가져오기
        grid = self.config.AREA_GRID_CONFIGS[area]
        tx_pos = grid['tx_position']
        
        # Figure 생성
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # 클러스터 수
        n_clusters = len(np.unique(cluster_labels))
        
        # Colormap 선택 (클러스터 수에 따라)
        cmap = self._select_colormap(n_clusters)
        
        # 클러스터 label randomize (많은 클러스터의 경우 색상 혼합 효과)
        if n_clusters > 20:
            # 클러스터 ID를 랜덤하게 재매핑
            unique_labels = np.unique(cluster_labels)
            np.random.seed(42)  # 재현성을 위한 seed
            shuffled_labels = np.random.permutation(unique_labels)
            label_map = {old: new for old, new in zip(unique_labels, shuffled_labels)}
            cluster_labels_plot = np.array([label_map[label] for label in cluster_labels])
        else:
            cluster_labels_plot = cluster_labels
        
        # UE 클러스터 scatter plot
        scatter = ax.scatter(x_coords, y_coords, c=cluster_labels_plot, 
                            cmap=cmap, s=100, alpha=0.7, edgecolors='black', linewidths=0.5)
        
        # BS 위치 표시
        ax.scatter(tx_pos[0], tx_pos[1], c='red', s=300, marker='^',
                  edgecolors='black', linewidths=2, zorder=10)
        
        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax, label='Cluster ID')
        cbar.set_label('Cluster ID', fontsize=12)
        
        # 축 레이블 및 제목
        ax.set_xlabel('X [m]', fontsize=12)
        ax.set_ylabel('Y [m]', fontsize=12)
        ax.set_title(f'QIE Clustering Spatial Distribution\n'
                    f'Area{area}_{freq}GHz ({n_clusters} clusters, {len(ue_ids)} UEs)',
                    fontsize=14, fontweight='bold')
        
        # Grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Aspect ratio
        ax.set_aspect('equal', adjustable='box')
        
        # Tight layout
        plt.tight_layout()
        
        # 저장
        freq_str = str(freq).replace('.', 'p')
        filename = f"A{area}_{freq_str}G_QIE_{timestamp}_map.png"
        filepath = os.path.join(self.config.P1M_OUTPUT_DIR, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Spatial map 저장: {filename}")
    
    def plot_comparison_map(self, ue_ids: List[int], 
                           cluster_labels_subset: np.ndarray,
                           cluster_labels_strict: np.ndarray,
                           area: int, freq: float, timestamp: str):
        """두 클러스터링 방법 비교 시각화
        
        Args:
            ue_ids: UE ID 리스트
            cluster_labels_subset: [n_ues] Subset-based 클러스터 레이블
            cluster_labels_strict: [n_ues] Strict agglomerative 클러스터 레이블
            area: Area ID
            freq: Frequency (GHz)
            timestamp: 타임스탬프 문자열
        """
        import matplotlib.pyplot as plt
        
        # UE 좌표 변환
        coords = AreaGridMapper.get_all_ue_coordinates(ue_ids, area, self.config)
        x_coords = coords[:, 0]
        y_coords = coords[:, 1]
        
        # Grid config
        grid = self.config.AREA_GRID_CONFIGS[area]
        tx_pos = grid['tx_position']
        
        # Figure 생성 (1×2 subplot)
        fig, axes = plt.subplots(1, 2, figsize=(24, 10))
        
        # 클러스터 수
        n_clusters_subset = len(np.unique(cluster_labels_subset))
        n_clusters_strict = len(np.unique(cluster_labels_strict))
        
        # Colormap 선택
        cmap_subset = self._select_colormap(n_clusters_subset)
        cmap_strict = self._select_colormap(n_clusters_strict)
        
        # Label randomize for Method 1 (많은 클러스터)
        if n_clusters_subset > 20:
            unique_labels = np.unique(cluster_labels_subset)
            np.random.seed(42)
            shuffled_labels = np.random.permutation(unique_labels)
            label_map = {old: new for old, new in zip(unique_labels, shuffled_labels)}
            cluster_labels_subset_plot = np.array([label_map[label] for label in cluster_labels_subset])
        else:
            cluster_labels_subset_plot = cluster_labels_subset
        
        # ===== Left: Subset-Based Clustering =====
        ax = axes[0]
        scatter1 = ax.scatter(x_coords, y_coords, c=cluster_labels_subset_plot, 
                             cmap=cmap_subset, s=100, alpha=0.7, 
                             edgecolors='black', linewidths=0.5)
        ax.scatter(tx_pos[0], tx_pos[1], c='red', s=300, marker='^',
                  edgecolors='black', linewidths=2, zorder=10)
        
        cbar1 = plt.colorbar(scatter1, ax=ax, label='Cluster ID')
        cbar1.set_label('Cluster ID', fontsize=11)
        
        ax.set_xlabel('X [m]', fontsize=11)
        ax.set_ylabel('Y [m]', fontsize=11)
        ax.set_title(f'Method 1: Subset-Based Clustering\n'
                    f'{n_clusters_subset} clusters, {len(ue_ids)} UEs',
                    fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_aspect('equal', adjustable='box')
        
        # ===== Right: Strict Agglomerative Clustering =====
        ax = axes[1]
        scatter2 = ax.scatter(x_coords, y_coords, c=cluster_labels_strict, 
                             cmap=cmap_strict, s=100, alpha=0.7, 
                             edgecolors='black', linewidths=0.5)
        ax.scatter(tx_pos[0], tx_pos[1], c='red', s=300, marker='^',
                  edgecolors='black', linewidths=2, zorder=10)
        
        cbar2 = plt.colorbar(scatter2, ax=ax, label='Cluster ID')
        cbar2.set_label('Cluster ID', fontsize=11)
        
        ax.set_xlabel('X [m]', fontsize=11)
        ax.set_ylabel('Y [m]', fontsize=11)
        ax.set_title(f'Method 2: Strict Agglomerative Clustering\n'
                    f'{n_clusters_strict} clusters, {len(ue_ids)} UEs',
                    fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_aspect('equal', adjustable='box')
        
        # Overall title
        fig.suptitle(f'QIE Clustering Comparison: Area{area}_{freq}GHz',
                    fontsize=15, fontweight='bold', y=0.98)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # 저장
        freq_str = str(freq).replace('.', 'p')
        filename = f"A{area}_{freq_str}G_QIE_{timestamp}_cmp.png"
        filepath = os.path.join(self.config.P1M_OUTPUT_DIR, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Comparison map 저장: {filename}")
    
    def _select_colormap(self, n_clusters: int):
        """클러스터 수에 따른 colormap 선택"""
        import matplotlib.pyplot as plt
        
        if n_clusters <= 9:
            return plt.cm.Set1
        elif n_clusters <= 20:
            return plt.cm.Paired
        else:
            return plt.cm.get_cmap('hsv', n_clusters)

# ===== LEVEL 2: 복합 의존 클래스 =====

# ----- ScipyCompleteHierarchical -----
class ScipyCompleteHierarchical:
    """Fast hierarchical clustering with complete linkage for inter-cluster separation"""
    
    @staticmethod
    def cluster(rho_matrix: np.ndarray, distance_threshold: float = 0.9) -> np.ndarray:
        """Complete linkage clustering
        
        Args:
            rho_matrix: [n_ues, n_ues] similarity matrix
            distance_threshold: merge if distance ≤ t (e.g., 0.9 → max_inter_rho ≈ 0.1)
        
        Returns:
            cluster_labels: [n_ues]
        """
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform
        
        distance_matrix = 1.0 - rho_matrix
        distance_condensed = squareform(distance_matrix, checks=False)
        
        # 'complete': merge based on max distance (minimizes inter-cluster ρ)
        Z = linkage(distance_condensed, method='complete')
        
        cluster_labels = fcluster(Z, t=distance_threshold, criterion='distance')
        return cluster_labels

# ----- QIEClusterer (REMOVED - too slow, O(n³) complexity) -----
# Replaced with ScipyCompleteHierarchical for speed (O(n² log n))
# Old implementation enforced dual constraints (min_intra_rho, max_inter_rho)
# New implementation uses complete linkage which naturally minimizes inter-cluster similarity

# ----- ExactBeamSetClusterer -----
class ExactBeamSetClusterer:
    """Subset-based beam set clustering"""
    
    @staticmethod
    def cluster(ue_ids: List[int], ue_beams: Dict[int, List[int]]) -> np.ndarray:
        """Cluster UEs with subset/superset beam set relationships
        
        Args:
            ue_ids: UE ID 리스트 (sorted)
            ue_beams: {ue_id: [beam_idx_1, ...]}
        
        Returns:
            cluster_labels: [n_ues] cluster ID
        """
        n_ues = len(ue_ids)
        cluster_labels = np.zeros(n_ues, dtype=int)
        
        # Union-Find for merging clusters
        parent = list(range(n_ues))
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            root_x = find(x)
            root_y = find(y)
            if root_x != root_y:
                parent[root_y] = root_x
        
        # Convert beam lists to sets
        beam_sets = [set(ue_beams[ue_id]) for ue_id in ue_ids]
        
        # Check all pairs for exact beam set matches
        for i in range(n_ues):
            for j in range(i + 1, n_ues):
                set_i = beam_sets[i]
                set_j = beam_sets[j]
                
                # Check if exact match
                if set_i == set_j:
                    union(i, j)
        
        # Assign cluster IDs based on root
        root_to_cluster = {}
        next_cluster_id = 1
        
        for idx in range(n_ues):
            root = find(idx)
            if root not in root_to_cluster:
                root_to_cluster[root] = next_cluster_id
                next_cluster_id += 1
            cluster_labels[idx] = root_to_cluster[root]
        
        return cluster_labels

# ===== LEVEL 3: 최상위 실행 =====

def main():
    """P1M: QIE 클러스터링 메인 실행"""
    print("=" * 70)
    print("P1M: QIE Clustering for Beam Management")
    print("=" * 70)
    
    # Config 초기화
    config = P1M_Config()
    
    if not config.csv_files:
        print("처리할 P1L CSV 파일이 없습니다.")
        return
    
    # DFT 코드북 생성
    print("\nDFT 코드북 생성...")
    F_bs = DFTCodebook.generate_2d_dft_codebook_tf(
        config.bs_ant_per_dim, config.bs_oversample
    )  # [16, 64]
    print(f"F_bs shape: {F_bs.shape}")
    
    # 각 Area-Freq에 대해 처리
    for area, freq, csv_path in config.csv_files:
        print("\n" + "=" * 70)
        print(f"Processing: Area{area}_{freq}GHz")
        print("=" * 70)
        
        # 1. P1L CSV 파싱
        print("\n[1/6] P1L CSV 파싱...")
        ue_beams = P1L_CSVParser.parse_csv(csv_path)
        print(f"총 {len(ue_beams)}개 UE 감지")
        
        if len(ue_beams) < 2:
            print("클러스터링을 위한 UE 수 부족 (최소 2개 필요)")
            continue
        
        # 2. Beam similarity 계산
        print("\n[2/6] Beam similarity 계산...")
        calculator = BeamInnerProductCalculator(F_bs)
        rho_matrix, ue_ids = calculator.compute_pairwise_similarity(ue_beams)
        print(f"ρ matrix shape: {rho_matrix.shape}")
        print(f"ρ range: [{rho_matrix.min():.4f}, {rho_matrix.max():.4f}]")
        print(f"ρ mean (off-diagonal): {np.mean(rho_matrix[np.triu_indices_from(rho_matrix, k=1)]):.4f}")
        
        # 3. 클러스터링 방법 1: Exact Beam Set Clustering
        print("\n[3/7] 방법 1: Exact Beam Set Clustering...")
        cluster_labels_subset = ExactBeamSetClusterer.cluster(ue_ids, ue_beams)
        
        n_clusters_subset = len(np.unique(cluster_labels_subset))
        print(f"  생성된 클러스터 수: {n_clusters_subset}")
        
        unique, counts = np.unique(cluster_labels_subset, return_counts=True)
        print("  클러스터 크기 분포:")
        for cid, count in zip(unique, counts):
            print(f"    Cluster {cid}: {count}개 UE")
        
        print("\n  Validation:")
        print_validation_stats(cluster_labels_subset, rho_matrix)
        
        # 4. 클러스터링 방법 2: Complete Linkage (multiple thresholds)
        print("\n[4/7] 방법 2: Complete Linkage Clustering...")
        
        threshold_configs = [
            (0.95, 'strict'),   # target max_inter_rho ≈ 0.05
            (0.90, 'baseline'), # target max_inter_rho ≈ 0.10
            (0.85, 'relaxed')   # target max_inter_rho ≈ 0.15
        ]
        
        results_method2 = []
        
        for distance_threshold, label in threshold_configs:
            print(f"\n  Threshold={distance_threshold} ({label})...")
            cluster_labels = ScipyCompleteHierarchical.cluster(rho_matrix, distance_threshold)
            
            n_clusters = len(np.unique(cluster_labels))
            print(f"    생성된 클러스터 수: {n_clusters}")
            
            unique, counts = np.unique(cluster_labels, return_counts=True)
            print("    클러스터 크기 분포 (상위 10개):")
            for cid, count in zip(unique[:10], counts[:10]):
                print(f"      Cluster {cid}: {count}개 UE")
            if len(unique) > 10:
                print(f"      ... (나머지 {len(unique) - 10}개 클러스터)")
            
            # Validation stats
            print("\n    Validation:")
            print_validation_stats(cluster_labels, rho_matrix)
            
            results_method2.append((cluster_labels, label))
        
        # Use strict (0.95) as default for comparison
        cluster_labels_complete = results_method2[0][0]
        
        # 5. 두 방법 비교 통계
        print("\n[5/7] 두 방법 비교 (Exact vs Complete-strict)...")
        print_comparison_stats(cluster_labels_subset, cluster_labels_complete, ue_ids)
        
        # 6. 결과 저장
        print("\n[6/7] 결과 저장...")
        timestamp = datetime.now().strftime("%m%d_%H%M")
        result_mgr = P1M_ResultManager(config)
        
        # Method 1: Exact match
        result_mgr.init_csv(area, freq, timestamp, "_ex")
        result_mgr.save_clustering_results(ue_ids, cluster_labels_subset, ue_beams, rho_matrix)
        
        # Method 2: All complete linkage thresholds
        method_suffix_map = {'strict': '_c_st', 'baseline': '_c_bs', 'relaxed': '_c_rx'}
        for (labels, label), (threshold, _) in zip(results_method2, threshold_configs):
            result_mgr.init_csv(area, freq, timestamp, method_suffix_map[label])
            result_mgr.save_clustering_results(ue_ids, labels, ue_beams, rho_matrix)
        
        # 7. Spatial map 시각화
        print("\n[7/7] Spatial map 생성...")
        visualizer = SpatialMapVisualizer(config)
        
        # Individual maps
        visualizer.plot_cluster_map(ue_ids, cluster_labels_subset, area, freq, timestamp + "_ex")
        visualizer.plot_cluster_map(ue_ids, cluster_labels_complete, area, freq, timestamp + "_c_st")
        
        # Comparison: exact vs complete-strict
        visualizer.plot_comparison_map(ue_ids, cluster_labels_subset, cluster_labels_complete, 
                                       area, freq, timestamp)
    
    print("\n" + "=" * 70)
    print("P1M 완료")
    print("=" * 70)

def print_comparison_stats(cluster_labels_1: np.ndarray, cluster_labels_2: np.ndarray, 
                          ue_ids: List[int]):
    """두 클러스터링 방법 비교 통계
    
    Args:
        cluster_labels_1: Exact-match 클러스터 레이블
        cluster_labels_2: Complete linkage 클러스터 레이블
        ue_ids: UE ID 리스트
    """
    n_ues = len(ue_ids)
    n_clusters_1 = len(np.unique(cluster_labels_1))
    n_clusters_2 = len(np.unique(cluster_labels_2))
    
    print(f"방법 1 (Exact): {n_clusters_1}개 클러스터")
    print(f"방법 2 (Complete): {n_clusters_2}개 클러스터")
    print(f"차이: {abs(n_clusters_1 - n_clusters_2)}개")
    
    # Agreement: 같은 클러스터에 속한 UE 쌍의 비율
    same_in_both = 0
    different_in_both = 0
    
    for i in range(n_ues):
        for j in range(i + 1, n_ues):
            same_1 = (cluster_labels_1[i] == cluster_labels_1[j])
            same_2 = (cluster_labels_2[i] == cluster_labels_2[j])
            
            if same_1 == same_2:
                if same_1:
                    same_in_both += 1
                else:
                    different_in_both += 1
    
    total_pairs = n_ues * (n_ues - 1) // 2
    agreement = (same_in_both + different_in_both) / total_pairs * 100
    
    print(f"\n클러스터링 일치도:")
    print(f"  전체 UE 쌍: {total_pairs}개")
    print(f"  양쪽 모두 같은 클러스터: {same_in_both}개")
    print(f"  양쪽 모두 다른 클러스터: {different_in_both}개")
    print(f"  일치도: {agreement:.2f}%")
    
    # 방법 2가 방법 1을 얼마나 세분화했는지
    if n_clusters_2 > n_clusters_1:
        print(f"\n방법 2가 방법 1을 더 세분화 (평균 {n_clusters_2/n_clusters_1:.2f}배)")
    elif n_clusters_2 < n_clusters_1:
        print(f"\n방법 2가 방법 1을 더 통합 (평균 {n_clusters_1/n_clusters_2:.2f}배)")
    else:
        print(f"\n두 방법의 클러스터 수 동일")

def print_validation_stats(cluster_labels: np.ndarray, rho_matrix: np.ndarray):
    """클러스터링 검증 통계 출력
    
    Intra-cluster vs Inter-cluster similarity 비교
    """
    unique_clusters = np.unique(cluster_labels)
    n_clusters = len(unique_clusters)
    
    # Intra-cluster similarity
    intra_rhos = []
    for cluster_id in unique_clusters:
        cluster_mask = (cluster_labels == cluster_id)
        cluster_indices = np.where(cluster_mask)[0]
        
        if len(cluster_indices) > 1:
            for i in cluster_indices:
                for j in cluster_indices:
                    if i < j:
                        intra_rhos.append(rho_matrix[i, j])
    
    # Inter-cluster similarity
    inter_rhos = []
    for i in range(len(cluster_labels)):
        for j in range(i + 1, len(cluster_labels)):
            if cluster_labels[i] != cluster_labels[j]:
                inter_rhos.append(rho_matrix[i, j])
    
    print(f"Intra-cluster ρ: mean={np.mean(intra_rhos):.4f}, std={np.std(intra_rhos):.4f}, "
          f"min={np.min(intra_rhos):.4f}, max={np.max(intra_rhos):.4f}")
    
    if inter_rhos:
        print(f"Inter-cluster ρ: mean={np.mean(inter_rhos):.4f}, std={np.std(inter_rhos):.4f}, "
              f"min={np.min(inter_rhos):.4f}, max={np.max(inter_rhos):.4f}")
        
        # Separation quality: intra와 inter의 차이가 클수록 좋음
        separation = np.mean(intra_rhos) - np.mean(inter_rhos)
        print(f"Separation (intra_mean - inter_mean): {separation:.4f}")
    else:
        print("Inter-cluster pairs: None (모든 UE가 단일 클러스터)")

def regenerate_maps():
    """기존 CSV에서 그림만 재생성"""
    print("=" * 70)
    print("P1M: Spatial Map Regeneration")
    print("=" * 70)
    
    config = P1M_Config()
    
    # 최신 CSV 파일 찾기
    result_dir = config.P1M_OUTPUT_DIR
    csv_files = [f for f in os.listdir(result_dir) if f.endswith('.csv')]
    
    if not csv_files:
        print("CSV 파일이 없습니다.")
        return
    
    # 최신 타임스탬프 찾기
    csv_files.sort(reverse=True)
    latest_timestamp = csv_files[0].split('_')[3] + '_' + csv_files[0].split('_')[4]
    
    print(f"\n최신 타임스탬프: {latest_timestamp}")
    
    # Area, Freq 추출 (첫 번째 파일에서)
    parts = csv_files[0].split('_')
    area = int(parts[0][1:])  # A1 -> 1
    freq = float(parts[1].replace('p', '.').replace('G', ''))  # 7p5G -> 7.5
    
    print(f"Area: {area}, Freq: {freq}GHz")
    
    # CSV 읽기 함수
    def read_clustering_csv(csv_path):
        ue_ids, cluster_labels = [], []
        with open(csv_path, 'r') as f:
            lines = [line for line in f if not line.strip().startswith('#') and 
                     not line.strip().startswith('"#') and line.strip()]
        
        reader = csv.DictReader(lines)
        for row in reader:
            try:
                ue_ids.append(int(row['ue'].strip()))
                cluster_labels.append(int(row['cluster_id'].strip()))
            except (KeyError, ValueError):
                continue
        return ue_ids, np.array(cluster_labels)
    
    # Method별로 처리
    methods = [('ex', 'Exact'), ('c_st', 'Complete-Strict'), 
               ('c_bs', 'Complete-Baseline'), ('c_rx', 'Complete-Relaxed')]
    
    visualizer = SpatialMapVisualizer(config)
    clustering_results = {}
    
    for method_suffix, method_name in methods:
        csv_filename = f"A{area}_{str(freq).replace('.', 'p')}G_QIE_{latest_timestamp}_{method_suffix}.csv"
        csv_path = os.path.join(result_dir, csv_filename)
        
        if not os.path.exists(csv_path):
            print(f"\n건너뜀: {csv_filename} (파일 없음)")
            continue
        
        print(f"\n처리 중: {method_name}")
        print(f"  CSV: {csv_filename}")
        
        # CSV 읽기
        ue_ids, cluster_labels = read_clustering_csv(csv_path)
        clustering_results[method_suffix] = (ue_ids, cluster_labels)
        
        print(f"  UE 수: {len(ue_ids)}, 클러스터 수: {len(np.unique(cluster_labels))}")
        
        # Spatial map 생성
        map_timestamp = latest_timestamp + '_' + method_suffix
        visualizer.plot_cluster_map(ue_ids, cluster_labels, area, freq, map_timestamp)
    
    # Comparison map 생성 (ex vs c_st)
    if 'ex' in clustering_results and 'c_st' in clustering_results:
        print(f"\nComparison map 생성 (Exact vs Complete-Strict)...")
        ue_ids_ex, labels_ex = clustering_results['ex']
        ue_ids_cst, labels_cst = clustering_results['c_st']
        
        if len(ue_ids_ex) == len(ue_ids_cst):
            visualizer.plot_comparison_map(ue_ids_ex, labels_ex, labels_cst, 
                                          area, freq, latest_timestamp)
        else:
            print("  경고: UE 수가 일치하지 않아 비교 그림을 생성할 수 없습니다.")
    
    print("\n" + "=" * 70)
    print("그림 재생성 완료")
    print("=" * 70)

if __name__ == "__main__":
    import sys
    
    # 명령줄 인자 확인
    if len(sys.argv) > 1 and sys.argv[1] == '--regen-maps':
        regenerate_maps()
    else:
        main()

