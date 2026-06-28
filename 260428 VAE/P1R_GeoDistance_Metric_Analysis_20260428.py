#!/usr/bin/env python3
"""
P1R_GeoDistance_Metric_Analysis_20260428.py

과제 B: Geometrical Distance 기반 Metric 분석
- RX 포인트 간 Euclidean Distance에 따른 채널 Metric 분포 시각화
- PDP 유사도, Covariance Matrix 거리, Eigenvalue 거리 등 다양한 Metric
- Distance vs. Metric 상관관계 분석

입력 데이터:
  - P1B Valid RXs: tau, power (PDP 원본)
  - P1F Marginal CCM: R_BS [1024×1024], R_UE [16×16]
  - P1I Weichselberger Chunks: U_BS, U_UE, Omega, H_mean

출력:
  - Distance vs. Metric scatter/binned plots
  - Correlation coefficient 요약 테이블
"""

import os
import sys
import glob
import re
import json
import time
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

warnings.filterwarnings('ignore', category=RuntimeWarning)

# ===== SECTION 1: Configuration =====
class P1R_Config:
    """P1R Geometrical Distance Metric Analysis 설정"""
    
    def __init__(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # === 입력 데이터 경로 (251009_CCM_Collection 기준) ===
        # 사용자 환경에 맞게 수정 필요
        self.DATA_BASE_DIR = os.path.join(os.path.dirname(script_dir), 
                                           "251009_CCM_Collection (8 GB)")
        
        # P1B Valid RXs
        self.P1B_INPUT_DIR = os.path.join(self.DATA_BASE_DIR, "P1B_Valid_Results")
        self.P1B_FILE_PATTERN = "Area{area}_{freq}GHz_Rays_Valid_RXs.npz"
        
        # P1F Marginal CCM
        self.P1F_CCM_DIR = os.path.join(self.DATA_BASE_DIR, "P1F_Marginal_CCM_Results")
        self.P1F_CCM_PATTERN = "Area{area}_{freq}GHz_RX{rx}_Marginal_CCM.npz"
        
        # P1I Weichselberger Chunks (251020 폴더)
        self.P1I_DIR = os.path.join(os.path.dirname(script_dir),
                                     "251020 BM QIE (Weichselberger)",
                                     "P1I_Weichsel_Chunk_Results")
        
        # === 출력 경로 ===
        self.OUTPUT_DIR = os.path.join(script_dir, "P1R_GeoDistance_Results")
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        
        # === 실험 설정 ===
        self.target_area = 1
        self.target_freq = 7.5
        
        # Area Grid (P1M과 동일)
        self.AREA_GRID_CONFIGS = {
            1: {
                'tx_position': [-51.561, -21.794, 19],
                'x_start': -136.138, 'x_stop': 58.862, 'num_x': 40,
                'y_start': -117.667, 'y_stop': 77.333, 'num_y': 40
            }
        }
        
        # === 분석 파라미터 ===
        self.n_distance_bins = 20       # Distance binning 개수
        self.max_pairs_sample = 50000   # 메모리 제한용 최대 쌍 수 (None=전체)
        self.random_seed = 42
        
        # PDP 파라미터
        self.pdp_n_bins = 128           # PDP 히스토그램 bin 수
        
        # 안테나 설정
        self.n_bs = 1024   # BS 안테나 수 (64 Layer × 16 AE/Layer)
        self.n_ue = 16     # UE 안테나 수 (4 Layer × 4 AE/Layer)
        
    def print_config(self):
        print(f"=== P1R Configuration ===")
        print(f"  Data base: {self.DATA_BASE_DIR}")
        print(f"  Area: {self.target_area}, Freq: {self.target_freq} GHz")
        print(f"  Distance bins: {self.n_distance_bins}")
        print(f"  Max pairs sample: {self.max_pairs_sample}")
        print(f"  Output: {self.OUTPUT_DIR}")


# ===== SECTION 2: RX Position & Distance =====
class GeoDistanceCalculator:
    """RX 좌표 계산 및 pairwise distance 생성"""
    
    def __init__(self, config: P1R_Config):
        self.config = config
    
    def ue_to_coordinates(self, ue_id: int) -> Tuple[float, float]:
        """UE ID (1-based) → (x, y) 좌표 [m]"""
        grid = self.config.AREA_GRID_CONFIGS[self.config.target_area]
        x_coords = np.linspace(grid['x_start'], grid['x_stop'], grid['num_x'])
        y_coords = np.linspace(grid['y_start'], grid['y_stop'], grid['num_y'])
        ue_idx = ue_id - 1
        x_idx = ue_idx % grid['num_x']
        y_idx = ue_idx // grid['num_x']
        return x_coords[x_idx], y_coords[y_idx]
    
    def get_all_coordinates(self, ue_ids: List[int]) -> np.ndarray:
        """모든 UE 좌표 [n_ues, 2]"""
        coords = np.zeros((len(ue_ids), 2))
        for i, uid in enumerate(ue_ids):
            coords[i] = self.ue_to_coordinates(uid)
        return coords
    
    def compute_pairwise_distances(self, coords: np.ndarray) -> np.ndarray:
        """Pairwise Euclidean distance (condensed form → square form)"""
        return squareform(pdist(coords, metric='euclidean'))


# ===== SECTION 3: Data Loaders =====
class P1R_DataLoader:
    """P1B, P1F, P1I 데이터 로딩"""
    
    def __init__(self, config: P1R_Config):
        self.config = config
        self._p1b_cache = None
        self._p1i_cache = {}
    
    def load_p1b_valid_rxs(self) -> dict:
        """P1B Valid RXs NPZ 로딩 (전체 RX)"""
        if self._p1b_cache is not None:
            return self._p1b_cache
        
        filename = self.config.P1B_FILE_PATTERN.format(
            area=self.config.target_area, freq=self.config.target_freq)
        filepath = os.path.join(self.config.P1B_INPUT_DIR, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"P1B file not found: {filepath}")
        
        print(f"  Loading P1B: {filename}")
        data = np.load(filepath, allow_pickle=True)
        
        self._p1b_cache = {
            'rx_indices': data['rx_indices'],
            'tau': data['tau'],
            'power': data['power'],
        }
        print(f"    Valid RXs: {len(data['rx_indices'])}")
        return self._p1b_cache
    
    def load_p1f_ccm(self, rx_id: int) -> Optional[dict]:
        """P1F Marginal CCM 로딩 (단일 RX)"""
        filename = self.config.P1F_CCM_PATTERN.format(
            area=self.config.target_area, freq=self.config.target_freq, rx=rx_id)
        filepath = os.path.join(self.config.P1F_CCM_DIR, filename)
        
        if not os.path.exists(filepath):
            return None
        
        data = np.load(filepath, allow_pickle=True)
        return {
            'R_BS': data['R_BS'],
            'R_UE': data['R_UE'],
        }
    
    def _build_p1i_index(self):
        """P1I 청크 인덱스 구축 (한 번만 실행) — UE ID → (filepath, idx_in_chunk) 매핑"""
        if hasattr(self, '_p1i_index'):
            return
        
        self._p1i_index = {}  # {ue_id: (filepath, idx_in_chunk)}
        scan_pattern = os.path.join(self.config.P1I_DIR,
            f"Area{self.config.target_area}_{self.config.target_freq}GHz_Weichsel_Chunk_*.npz")
        chunk_files = sorted(glob.glob(scan_pattern))
        
        print(f"    P1I 청크 인덱스 구축: {len(chunk_files)}개 파일")
        for filepath in chunk_files:
            data = np.load(filepath, allow_pickle=True)
            self._p1i_cache[filepath] = data
            ue_indices = data['ue_indices'].tolist()
            for idx, uid in enumerate(ue_indices):
                self._p1i_index[uid] = (filepath, idx)
        print(f"    P1I 인덱스 완료: {len(self._p1i_index)}개 UE 매핑")
    
    def load_p1i_ue_data(self, ue_id: int) -> Optional[dict]:
        """P1I Weichselberger Chunk에서 UE 데이터 로딩 (인덱스 기반 O(1) 조회)"""
        self._build_p1i_index()
        
        if ue_id not in self._p1i_index:
            return None
        
        filepath, idx = self._p1i_index[ue_id]
        data = self._p1i_cache[filepath]
        return {
            'U_BS': data['P1G_U_BS'][idx],
            'U_UE': data['P1G_U_UE'][idx],
            'Omega': data['P1G_Omega'][idx],
            'H_mean': data['P1H_H_mean'][idx],
        }
    
    def get_available_rx_ids(self) -> List[int]:
        """P1F CCM 파일이 존재하는 RX ID 목록"""
        pattern = os.path.join(self.config.P1F_CCM_DIR,
            f"Area{self.config.target_area}_{self.config.target_freq}GHz_RX*_Marginal_CCM.npz")
        files = glob.glob(pattern)
        rx_ids = []
        for f in files:
            m = re.search(r'RX(\d+)_Marginal_CCM', os.path.basename(f))
            if m:
                rx_ids.append(int(m.group(1)))
        return sorted(rx_ids)


# ===== SECTION 4: Metric Computation =====
class MetricCalculator:
    """채널 Metric 계산 (pairwise)"""
    
    @staticmethod
    def pdp_from_rays(tau: np.ndarray, power: np.ndarray, n_bins: int = 128
                      ) -> Tuple[np.ndarray, np.ndarray]:
        """Ray 데이터로부터 PDP 히스토그램 생성
        
        Args:
            tau: [n_rays] delay (s)
            power: [n_rays] linear power
            n_bins: 히스토그램 bin 수
        
        Returns:
            pdp: [n_bins] normalized PDP
            bin_edges: [n_bins+1] bin 경계
        """
        # 유효 ray만 사용 (power > 0)
        valid = power > 0
        if np.sum(valid) == 0:
            return np.zeros(n_bins), np.linspace(0, 1, n_bins + 1)
        
        tau_v = tau[valid]
        power_v = power[valid]
        
        bin_edges = np.linspace(tau_v.min(), tau_v.max() + 1e-12, n_bins + 1)
        pdp, _ = np.histogram(tau_v, bins=bin_edges, weights=power_v)
        
        # Normalize
        total = pdp.sum()
        if total > 0:
            pdp = pdp / total
        
        return pdp, bin_edges
    
    @staticmethod
    def pdp_correlation(pdp_i: np.ndarray, pdp_j: np.ndarray) -> float:
        """PDP 간 Pearson correlation (1=동일, 0=무관)"""
        if np.std(pdp_i) < 1e-12 or np.std(pdp_j) < 1e-12:
            return 0.0
        corr, _ = pearsonr(pdp_i, pdp_j)
        return float(corr)
    
    @staticmethod
    def covariance_frobenius_distance(R_i: np.ndarray, R_j: np.ndarray) -> float:
        """Covariance Matrix 간 Frobenius distance (정규화)
        
        d_F(R_i, R_j) = ||R_i/||R_i||_F - R_j/||R_j||_F||_F
        """
        norm_i = np.linalg.norm(R_i, 'fro')
        norm_j = np.linalg.norm(R_j, 'fro')
        if norm_i < 1e-12 or norm_j < 1e-12:
            return 2.0  # max distance
        R_i_n = R_i / norm_i
        R_j_n = R_j / norm_j
        return float(np.linalg.norm(R_i_n - R_j_n, 'fro'))
    
    @staticmethod
    def chordal_distance(R_i: np.ndarray, R_j: np.ndarray, k: int = 10) -> float:
        """Chordal distance between dominant subspaces
        
        Top-k eigenvector subspace 간 chordal distance
        d_c = sqrt(k - ||U_i^H U_j||_F^2)
        """
        eigvals_i, eigvecs_i = np.linalg.eigh(R_i)
        eigvals_j, eigvecs_j = np.linalg.eigh(R_j)
        
        # Top-k (내림차순)
        U_i = eigvecs_i[:, -k:]
        U_j = eigvecs_j[:, -k:]
        
        inner = np.linalg.norm(U_i.conj().T @ U_j, 'fro') ** 2
        dist_sq = max(0.0, k - inner)
        return float(np.sqrt(dist_sq))
    
    @staticmethod
    def eigenvalue_distance(R_i: np.ndarray, R_j: np.ndarray) -> float:
        """Eigenvalue 분포 간 normalized L2 distance
        
        정규화된 고유값 벡터 간 L2 거리
        """
        eig_i = np.sort(np.real(np.linalg.eigvalsh(R_i)))[::-1]
        eig_j = np.sort(np.real(np.linalg.eigvalsh(R_j)))[::-1]
        
        sum_i = np.sum(eig_i)
        sum_j = np.sum(eig_j)
        if sum_i < 1e-12 or sum_j < 1e-12:
            return 2.0
        
        eig_i_n = eig_i / sum_i
        eig_j_n = eig_j / sum_j
        return float(np.linalg.norm(eig_i_n - eig_j_n))
    
    @staticmethod
    def coupling_matrix_distance(Omega_i: np.ndarray, Omega_j: np.ndarray) -> float:
        """Coupling matrix (Omega) 간 normalized Frobenius distance"""
        norm_i = np.linalg.norm(Omega_i, 'fro')
        norm_j = np.linalg.norm(Omega_j, 'fro')
        if norm_i < 1e-12 or norm_j < 1e-12:
            return 2.0
        return float(np.linalg.norm(Omega_i / norm_i - Omega_j / norm_j, 'fro'))


# ===== SECTION 5: Analysis Engine =====
class GeoDistanceAnalyzer:
    """Geometrical Distance vs. Metric 분석 엔진"""
    
    def __init__(self, config: P1R_Config):
        self.config = config
        self.loader = P1R_DataLoader(config)
        self.geo = GeoDistanceCalculator(config)
        self.metric = MetricCalculator()
        
        # 결과 저장
        self.ue_ids = []
        self.coords = None
        self.dist_matrix = None
        self.metrics = {}  # {metric_name: condensed_array}
    
    @staticmethod
    def _print_progress(current, total, t_start, prefix=""):
        """진행률 + ETA 출력"""
        elapsed = time.time() - t_start
        pct = current / total * 100
        if current > 0:
            eta = elapsed / current * (total - current)
            print(f"\r  {prefix}{current}/{total} ({pct:.0f}%) | "
                  f"경과 {elapsed:.0f}s | 남은 ~{eta:.0f}s", end="", flush=True)
        else:
            print(f"\r  {prefix}{current}/{total} ({pct:.0f}%)", end="", flush=True)
    
    def prepare_data(self):
        """데이터 준비: RX 목록, 좌표, 거리 행렬"""
        print("\n[1/4] 데이터 준비")
        
        # P1F CCM이 있는 RX만 사용
        print("  RX 목록 스캔 중...")
        self.ue_ids = self.loader.get_available_rx_ids()
        n_ues = len(self.ue_ids)
        print(f"  Available RXs (with P1F CCM): {n_ues}")
        
        if n_ues < 2:
            raise ValueError("최소 2개 RX 필요")
        
        # 좌표 계산
        self.coords = self.geo.get_all_coordinates(self.ue_ids)
        print(f"  Coordinate range: x=[{self.coords[:,0].min():.1f}, {self.coords[:,0].max():.1f}], "
              f"y=[{self.coords[:,1].min():.1f}, {self.coords[:,1].max():.1f}]")
        
        # Pairwise distance
        self.dist_matrix = self.geo.compute_pairwise_distances(self.coords)
        print(f"  Distance range: [{self.dist_matrix[self.dist_matrix>0].min():.1f}, "
              f"{self.dist_matrix.max():.1f}] m")
    
    def _select_pairs(self) -> List[Tuple[int, int]]:
        """분석할 (i, j) 쌍 선택 (upper triangle, 샘플링)"""
        n = len(self.ue_ids)
        all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        
        if self.config.max_pairs_sample and len(all_pairs) > self.config.max_pairs_sample:
            rng = np.random.RandomState(self.config.random_seed)
            indices = rng.choice(len(all_pairs), self.config.max_pairs_sample, replace=False)
            pairs = [all_pairs[k] for k in sorted(indices)]
            print(f"  Sampled {len(pairs)} pairs from {len(all_pairs)} total")
        else:
            pairs = all_pairs
            print(f"  Using all {len(pairs)} pairs")
        
        return pairs
    
    def compute_pdp_metrics(self, pairs: List[Tuple[int, int]]):
        """PDP Correlation 계산"""
        print("\n[2/4] PDP Correlation 계산")
        
        # P1B 데이터 로딩
        try:
            p1b = self.loader.load_p1b_valid_rxs()
        except FileNotFoundError as e:
            print(f"  ⚠ P1B 데이터 없음, PDP metric 건너뜀: {e}")
            return
        
        rx_indices_p1b = p1b['rx_indices'].tolist()
        
        # 각 RX의 PDP 사전 계산
        print("  PDP 사전 계산 중...")
        pdp_cache = {}
        t0 = time.time()
        for idx, uid in enumerate(self.ue_ids):
            if uid in rx_indices_p1b:
                pos = rx_indices_p1b.index(uid)
                tau_rx = p1b['tau'][pos]
                power_rx = p1b['power'][pos]
                pdp, _ = self.metric.pdp_from_rays(tau_rx, power_rx, self.config.pdp_n_bins)
                pdp_cache[uid] = pdp
            if (idx + 1) % 50 == 0 or idx == len(self.ue_ids) - 1:
                self._print_progress(idx + 1, len(self.ue_ids), t0, "PDP 계산: ")
        print()  # 줄바꿈
        
        print(f"  PDP 계산 완료: {len(pdp_cache)}/{len(self.ue_ids)} RXs")
        
        # Pairwise PDP correlation
        pdp_corr = np.full(len(pairs), np.nan)
        computed = 0
        for k, (i, j) in enumerate(pairs):
            uid_i, uid_j = self.ue_ids[i], self.ue_ids[j]
            if uid_i in pdp_cache and uid_j in pdp_cache:
                pdp_corr[k] = self.metric.pdp_correlation(pdp_cache[uid_i], pdp_cache[uid_j])
                computed += 1
        
        self.metrics['PDP Correlation'] = pdp_corr
        print(f"  PDP Correlation 완료: {computed}/{len(pairs)} pairs")
    
    def compute_covariance_metrics(self, pairs: List[Tuple[int, int]]):
        """Covariance Matrix 기반 Metric 계산 (R_UE 사용 — 메모리 효율)"""
        print("\n[3/4] Covariance Metric 계산 (R_UE)")
        
        # R_UE를 P1I 청크에서 일괄 로딩 (개별 P1F 파일 I/O 회피)
        print("  R_UE 로딩 중 (P1I 청크 일괄)...")
        rue_cache = {}
        self.loader._build_p1i_index()
        ue_set = set(self.ue_ids)
        for filepath, data in self.loader._p1i_cache.items():
            ue_indices = data['ue_indices'].tolist()
            r_ue_all = data['P1F_R_UE']  # [n_ues_in_chunk, n_ue, n_ue]
            for idx, uid in enumerate(ue_indices):
                if uid in ue_set:
                    rue_cache[uid] = r_ue_all[idx]
        
        print(f"  R_UE 로딩 완료: {len(rue_cache)}/{len(self.ue_ids)} RXs")
        
        # Metric 배열 초기화
        frob_dist = np.full(len(pairs), np.nan)
        chord_dist = np.full(len(pairs), np.nan)
        eig_dist = np.full(len(pairs), np.nan)
        
        t0 = time.time()
        computed = 0
        for k, (i, j) in enumerate(pairs):
            uid_i, uid_j = self.ue_ids[i], self.ue_ids[j]
            if uid_i in rue_cache and uid_j in rue_cache:
                R_i = rue_cache[uid_i]
                R_j = rue_cache[uid_j]
                
                frob_dist[k] = self.metric.covariance_frobenius_distance(R_i, R_j)
                chord_dist[k] = self.metric.chordal_distance(R_i, R_j, k=min(4, R_i.shape[0]))
                eig_dist[k] = self.metric.eigenvalue_distance(R_i, R_j)
                computed += 1
            
            if (k + 1) % 5000 == 0:
                elapsed = time.time() - t0
                eta = elapsed / (k + 1) * (len(pairs) - k - 1)
                print(f"    {k+1}/{len(pairs)} pairs ({elapsed:.1f}s, 남은 ~{eta:.0f}s)")
        
        self.metrics['Cov Frobenius (R_UE)'] = frob_dist
        self.metrics['Chordal Dist (R_UE)'] = chord_dist
        self.metrics['Eigenvalue Dist (R_UE)'] = eig_dist
        
        elapsed = time.time() - t0
        print(f"  Covariance metrics 완료: {computed} pairs ({elapsed:.1f}s)")
    
    def compute_coupling_metrics(self, pairs: List[Tuple[int, int]]):
        """Coupling Matrix (Omega) 기반 Metric 계산"""
        print("\n[3.5/4] Coupling Matrix Metric 계산")
        
        # Omega를 P1I 청크에서 일괄 로딩 (이미 캐시됨)
        print("  Omega 로딩 중 (P1I 청크 일괄)...")
        omega_cache = {}
        self.loader._build_p1i_index()
        ue_set = set(self.ue_ids)
        for filepath, data in self.loader._p1i_cache.items():
            ue_indices = data['ue_indices'].tolist()
            omega_all = data['P1G_Omega']  # [n_ues_in_chunk, n_bs, n_ue]
            for idx, uid in enumerate(ue_indices):
                if uid in ue_set:
                    omega_cache[uid] = omega_all[idx]
        
        print(f"  Omega 로딩 완료: {len(omega_cache)}/{len(self.ue_ids)} RXs")
        
        if len(omega_cache) < 2:
            print("  ⚠ Omega 데이터 부족, 건너뜀")
            return
        
        omega_dist = np.full(len(pairs), np.nan)
        computed = 0
        for k, (i, j) in enumerate(pairs):
            uid_i, uid_j = self.ue_ids[i], self.ue_ids[j]
            if uid_i in omega_cache and uid_j in omega_cache:
                omega_dist[k] = self.metric.coupling_matrix_distance(
                    omega_cache[uid_i], omega_cache[uid_j])
                computed += 1
        
        self.metrics['Coupling Mat Dist (Ω)'] = omega_dist
        print(f"  Coupling metric 완료: {computed} pairs")
    
    def run_analysis(self) -> dict:
        """전체 분석 실행"""
        self.prepare_data()
        pairs = self._select_pairs()
        
        # 거리 배열 (pairs에 대응)
        geo_distances = np.array([self.dist_matrix[i, j] for i, j in pairs])
        
        # Metric 계산
        self.compute_pdp_metrics(pairs)
        self.compute_covariance_metrics(pairs)
        self.compute_coupling_metrics(pairs)
        
        # Correlation 분석
        print("\n[4/4] Correlation 분석")
        correlation_results = {}
        for name, values in self.metrics.items():
            valid = ~np.isnan(values)
            if np.sum(valid) < 10:
                print(f"  {name}: 유효 데이터 부족 ({np.sum(valid)})")
                continue
            
            d = geo_distances[valid]
            v = values[valid]
            
            r_pearson, p_pearson = pearsonr(d, v)
            r_spearman, p_spearman = spearmanr(d, v)
            
            correlation_results[name] = {
                'pearson_r': r_pearson,
                'pearson_p': p_pearson,
                'spearman_r': r_spearman,
                'spearman_p': p_spearman,
                'n_valid': int(np.sum(valid)),
            }
            
            print(f"  {name}:")
            print(f"    Pearson  r={r_pearson:+.4f} (p={p_pearson:.2e})")
            print(f"    Spearman ρ={r_spearman:+.4f} (p={p_spearman:.2e})")
        
        return {
            'geo_distances': geo_distances,
            'pairs': pairs,
            'metrics': self.metrics,
            'correlations': correlation_results,
            'ue_ids': self.ue_ids,
            'coords': self.coords,
        }


# ===== SECTION 6: Visualization =====
class P1R_Visualizer:
    """Distance vs. Metric 시각화"""
    
    def __init__(self, config: P1R_Config):
        self.config = config
        plt.rcParams.update({
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 11,
            'figure.dpi': 150,
        })
    
    def plot_all_metrics(self, results: dict):
        """모든 metric에 대한 Distance vs. Metric 그래프"""
        geo_dist = results['geo_distances']
        metrics = results['metrics']
        correlations = results['correlations']
        
        n_metrics = len(metrics)
        if n_metrics == 0:
            print("  시각화할 metric 없음")
            return
        
        fig, axes = plt.subplots(2, max(3, (n_metrics + 1) // 2), 
                                  figsize=(6 * max(3, (n_metrics + 1) // 2), 10))
        axes = axes.flatten()
        
        for idx, (name, values) in enumerate(metrics.items()):
            ax = axes[idx]
            valid = ~np.isnan(values)
            if np.sum(valid) < 10:
                ax.set_title(f"{name}\n(데이터 부족)")
                continue
            
            d = geo_dist[valid]
            v = values[valid]
            
            # Scatter (alpha 조절)
            n_pts = len(d)
            alpha = max(0.02, min(0.3, 1000 / n_pts))
            ax.scatter(d, v, s=1, alpha=alpha, c='steelblue', rasterized=True)
            
            # Binned mean ± std
            bin_edges = np.linspace(d.min(), d.max(), self.config.n_distance_bins + 1)
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            bin_means = np.zeros(self.config.n_distance_bins)
            bin_stds = np.zeros(self.config.n_distance_bins)
            
            for b in range(self.config.n_distance_bins):
                mask = (d >= bin_edges[b]) & (d < bin_edges[b + 1])
                if b == self.config.n_distance_bins - 1:
                    mask = (d >= bin_edges[b]) & (d <= bin_edges[b + 1])
                if np.sum(mask) > 0:
                    bin_means[b] = np.mean(v[mask])
                    bin_stds[b] = np.std(v[mask])
            
            ax.plot(bin_centers, bin_means, 'r-', linewidth=2, label='Bin mean')
            ax.fill_between(bin_centers, bin_means - bin_stds, bin_means + bin_stds,
                           alpha=0.2, color='red', label='±1σ')
            
            # Correlation 표시
            if name in correlations:
                corr = correlations[name]
                ax.text(0.02, 0.98, 
                        f"Pearson r={corr['pearson_r']:+.3f}\n"
                        f"Spearman ρ={corr['spearman_r']:+.3f}",
                        transform=ax.transAxes, va='top', fontsize=9,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            ax.set_xlabel('Geometrical Distance [m]')
            ax.set_ylabel(name)
            ax.set_title(name)
            ax.legend(loc='lower right', fontsize=8)
            ax.grid(True, alpha=0.3)
        
        # 빈 axes 숨기기
        for idx in range(n_metrics, len(axes)):
            axes[idx].set_visible(False)
        
        fig.suptitle(f'Geometrical Distance vs. Channel Metrics\n'
                     f'Area{self.config.target_area}, {self.config.target_freq}GHz, '
                     f'{len(results["ue_ids"])} RXs',
                     fontsize=14, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        
        filepath = os.path.join(self.config.OUTPUT_DIR, 
                                'P1R_fig1_distance_vs_metrics.png')
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {os.path.basename(filepath)}")
    
    def plot_binned_boxplot(self, results: dict):
        """Distance bin별 box plot"""
        geo_dist = results['geo_distances']
        metrics = results['metrics']
        
        n_metrics = len(metrics)
        if n_metrics == 0:
            return
        
        fig, axes = plt.subplots(2, max(3, (n_metrics + 1) // 2),
                                  figsize=(6 * max(3, (n_metrics + 1) // 2), 10))
        axes = axes.flatten()
        
        n_bins = min(10, self.config.n_distance_bins)
        
        for idx, (name, values) in enumerate(metrics.items()):
            ax = axes[idx]
            valid = ~np.isnan(values)
            if np.sum(valid) < 10:
                ax.set_title(f"{name}\n(데이터 부족)")
                continue
            
            d = geo_dist[valid]
            v = values[valid]
            
            bin_edges = np.linspace(d.min(), d.max(), n_bins + 1)
            box_data = []
            labels = []
            for b in range(n_bins):
                if b < n_bins - 1:
                    mask = (d >= bin_edges[b]) & (d < bin_edges[b + 1])
                else:
                    mask = (d >= bin_edges[b]) & (d <= bin_edges[b + 1])
                if np.sum(mask) > 5:
                    box_data.append(v[mask])
                    center = 0.5 * (bin_edges[b] + bin_edges[b + 1])
                    labels.append(f'{center:.0f}')
                else:
                    box_data.append([])
                    labels.append('')
            
            bp = ax.boxplot(box_data, labels=labels, patch_artist=True,
                           showfliers=False, widths=0.7)
            for patch in bp['boxes']:
                patch.set_facecolor('lightsteelblue')
            
            ax.set_xlabel('Distance bin center [m]')
            ax.set_ylabel(name)
            ax.set_title(name)
            ax.grid(True, alpha=0.3, axis='y')
            ax.tick_params(axis='x', rotation=45)
        
        for idx in range(n_metrics, len(axes)):
            axes[idx].set_visible(False)
        
        fig.suptitle(f'Metric Distribution by Distance Bin\n'
                     f'Area{self.config.target_area}, {self.config.target_freq}GHz',
                     fontsize=14, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        
        filepath = os.path.join(self.config.OUTPUT_DIR,
                                'P1R_fig2_distance_binned_boxplot.png')
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {os.path.basename(filepath)}")
    
    def plot_correlation_summary(self, results: dict):
        """Correlation coefficient 요약 bar chart"""
        correlations = results['correlations']
        if not correlations:
            return
        
        names = list(correlations.keys())
        pearson_r = [correlations[n]['pearson_r'] for n in names]
        spearman_r = [correlations[n]['spearman_r'] for n in names]
        
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(names))
        w = 0.35
        
        bars1 = ax.bar(x - w/2, pearson_r, w, label='Pearson r', color='steelblue')
        bars2 = ax.bar(x + w/2, spearman_r, w, label='Spearman ρ', color='coral')
        
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
        ax.set_ylabel('Correlation Coefficient')
        ax.set_title('Geo Distance vs. Metric: Correlation Summary')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.set_ylim(-1.1, 1.1)
        
        # 값 표시
        for bar in bars1:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.03 * np.sign(h),
                    f'{h:.3f}', ha='center', va='bottom' if h >= 0 else 'top', fontsize=8)
        for bar in bars2:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.03 * np.sign(h),
                    f'{h:.3f}', ha='center', va='bottom' if h >= 0 else 'top', fontsize=8)
        
        fig.tight_layout()
        filepath = os.path.join(self.config.OUTPUT_DIR,
                                'P1R_fig3_correlation_summary.png')
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {os.path.basename(filepath)}")
    
    def plot_spatial_map(self, results: dict):
        """RX 공간 분포 + TX 위치 시각화"""
        coords = results['coords']
        ue_ids = results['ue_ids']
        grid = self.config.AREA_GRID_CONFIGS[self.config.target_area]
        tx_pos = grid['tx_position']
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(coords[:, 0], coords[:, 1], s=3, c='steelblue', alpha=0.6, label=f'RX ({len(ue_ids)})')
        ax.scatter(tx_pos[0], tx_pos[1], s=200, c='red', marker='^', 
                   edgecolors='black', linewidths=1.5, zorder=5, label='TX (BS)')
        
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_title(f'RX Spatial Distribution (Area{self.config.target_area})')
        ax.legend()
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        fig.tight_layout()
        filepath = os.path.join(self.config.OUTPUT_DIR,
                                'P1R_fig0_spatial_map.png')
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {os.path.basename(filepath)}")
    
    def plot_spatial_metric_maps(self, results: dict):
        """각 UE 위치에 '주변 UE들과의 평균 metric'을 색상으로 표시하는 공간 맵
        
        각 UE i에 대해: mean_metric[i] = mean(metric(i,j) for all sampled j)
        → 지도 위에 해당 값을 heatmap으로 표시
        → 채널 특성이 공간적으로 어떻게 분포하는지 직관적으로 보여줌
        """
        coords = results['coords']
        ue_ids = results['ue_ids']
        pairs = results['pairs']
        metrics = results['metrics']
        grid = self.config.AREA_GRID_CONFIGS[self.config.target_area]
        tx_pos = grid['tx_position']
        
        n_ues = len(ue_ids)
        n_metrics = len(metrics)
        if n_metrics == 0:
            print("  공간 맵 시각화할 metric 없음")
            return
        
        # 각 UE별 평균 metric 계산
        ue_mean_metrics = {}  # {metric_name: [n_ues] array}
        for name, values in metrics.items():
            ue_sums = np.zeros(n_ues)
            ue_counts = np.zeros(n_ues)
            for k, (i, j) in enumerate(pairs):
                if not np.isnan(values[k]):
                    ue_sums[i] += values[k]
                    ue_counts[i] += 1
                    ue_sums[j] += values[k]
                    ue_counts[j] += 1
            
            ue_mean = np.full(n_ues, np.nan)
            valid_mask = ue_counts > 0
            ue_mean[valid_mask] = ue_sums[valid_mask] / ue_counts[valid_mask]
            ue_mean_metrics[name] = ue_mean
        
        # 그래프: metric별 공간 맵
        fig, axes = plt.subplots(2, max(3, (n_metrics + 1) // 2),
                                  figsize=(7 * max(3, (n_metrics + 1) // 2), 12))
        axes = axes.flatten()
        
        for idx, (name, ue_mean) in enumerate(ue_mean_metrics.items()):
            ax = axes[idx]
            valid = ~np.isnan(ue_mean)
            if np.sum(valid) < 5:
                ax.set_title(f"{name}\n(데이터 부족)")
                continue
            
            sc = ax.scatter(coords[valid, 0], coords[valid, 1],
                           c=ue_mean[valid], s=15, cmap='viridis',
                           alpha=0.8, edgecolors='none')
            ax.scatter(tx_pos[0], tx_pos[1], s=200, c='red', marker='^',
                      edgecolors='black', linewidths=1.5, zorder=5, label='TX')
            
            cbar = fig.colorbar(sc, ax=ax, shrink=0.8)
            cbar.set_label(f'Mean {name}', fontsize=9)
            
            ax.set_xlabel('X [m]')
            ax.set_ylabel('Y [m]')
            ax.set_title(f'Spatial Map: {name}')
            ax.set_aspect('equal')
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.2)
        
        # 빈 axes 숨기기
        for idx in range(n_metrics, len(axes)):
            axes[idx].set_visible(False)
        
        fig.suptitle(f'Spatial Distribution of Channel Metrics (per-UE mean)\n'
                     f'Area{self.config.target_area}, {self.config.target_freq}GHz, '
                     f'{n_ues} RXs',
                     fontsize=14, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fig.subplots_adjust(wspace=0.4, hspace=0.35)
        
        filepath = os.path.join(self.config.OUTPUT_DIR,
                                'P1R_fig4_spatial_metric_maps.png')
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {os.path.basename(filepath)}")
    
    def plot_local_correlation_map(self, results: dict):
        """각 UE 위치에서의 '로컬 상관계수'를 지도에 표시
        
        각 UE i에 대해: 해당 UE가 포함된 쌍들만 모아서
        local_corr[i] = corr(distance, metric) for pairs involving UE i
        → 공간적으로 상관관계가 강한/약한 지역을 시각화
        """
        coords = results['coords']
        ue_ids = results['ue_ids']
        pairs = results['pairs']
        geo_distances = results['geo_distances']
        metrics = results['metrics']
        grid = self.config.AREA_GRID_CONFIGS[self.config.target_area]
        tx_pos = grid['tx_position']
        
        n_ues = len(ue_ids)
        n_metrics = len(metrics)
        if n_metrics == 0:
            return
        
        # 각 UE별 관련 pair 인덱스 구축
        ue_pair_indices = [[] for _ in range(n_ues)]
        for k, (i, j) in enumerate(pairs):
            ue_pair_indices[i].append(k)
            ue_pair_indices[j].append(k)
        
        # 대표 metric 선택 (Cov Frobenius가 가장 상관 높았으므로)
        target_metrics = ['Cov Frobenius (R_UE)', 'Eigenvalue Dist (R_UE)', 'Coupling Mat Dist (Ω)']
        available = [m for m in target_metrics if m in metrics]
        if not available:
            available = list(metrics.keys())[:3]
        
        n_plots = len(available)
        fig, axes = plt.subplots(1, n_plots, figsize=(8 * n_plots, 7))
        if n_plots == 1:
            axes = [axes]
        else:
            axes = np.atleast_1d(axes)
        
        min_pairs_for_corr = 20  # 로컬 상관 계산에 필요한 최소 쌍 수
        
        for plot_idx, name in enumerate(available):
            ax = axes[plot_idx]
            values = metrics[name]
            
            local_corr = np.full(n_ues, np.nan)
            for i in range(n_ues):
                pair_idx = ue_pair_indices[i]
                if len(pair_idx) < min_pairs_for_corr:
                    continue
                
                d_local = geo_distances[pair_idx]
                v_local = values[pair_idx]
                valid = ~np.isnan(v_local)
                if np.sum(valid) < min_pairs_for_corr:
                    continue
                
                r, _ = spearmanr(d_local[valid], v_local[valid])
                local_corr[i] = r
            
            valid_ue = ~np.isnan(local_corr)
            if np.sum(valid_ue) < 5:
                ax.set_title(f"{name}\n(데이터 부족)")
                continue
            
            sc = ax.scatter(coords[valid_ue, 0], coords[valid_ue, 1],
                           c=local_corr[valid_ue], s=15, cmap='RdBu_r',
                           vmin=-0.5, vmax=0.5,
                           alpha=0.8, edgecolors='none')
            ax.scatter(tx_pos[0], tx_pos[1], s=200, c='lime', marker='^',
                      edgecolors='black', linewidths=1.5, zorder=5, label='TX')
            
            cbar = fig.colorbar(sc, ax=ax, shrink=0.8)
            cbar.set_label('Local Spearman ρ', fontsize=9)
            
            mean_corr = np.nanmean(local_corr[valid_ue])
            ax.set_xlabel('X [m]')
            ax.set_ylabel('Y [m]')
            ax.set_title(f'Local Correlation: {name}\n(mean ρ={mean_corr:+.3f})')
            ax.set_aspect('equal')
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.2)
        
        # 빈 axes 숨기기
        for idx in range(n_plots, len(np.atleast_1d(axes))):
            np.atleast_1d(axes)[idx].set_visible(False)
        
        fig.suptitle(f'Local Spatial Correlation Map (per-UE Spearman ρ)\n'
                     f'Area{self.config.target_area}, {self.config.target_freq}GHz',
                     fontsize=13, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        fig.subplots_adjust(wspace=0.35)
        
        filepath = os.path.join(self.config.OUTPUT_DIR,
                                'P1R_fig5_local_correlation_map.png')
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {os.path.basename(filepath)}")


# ===== SECTION 7: Result Export =====
def save_correlation_csv(results: dict, config: P1R_Config):
    """Correlation 결과 CSV 저장"""
    correlations = results['correlations']
    if not correlations:
        return
    
    filepath = os.path.join(config.OUTPUT_DIR, 'P1R_correlation_summary.csv')
    
    with open(filepath, 'w') as f:
        f.write("Metric,Pearson_r,Pearson_p,Spearman_rho,Spearman_p,N_valid\n")
        for name, corr in correlations.items():
            f.write(f"{name},{corr['pearson_r']:.6f},{corr['pearson_p']:.2e},"
                    f"{corr['spearman_r']:.6f},{corr['spearman_p']:.2e},"
                    f"{corr['n_valid']}\n")
    
    print(f"  Saved: {os.path.basename(filepath)}")


def save_analysis_metadata(results: dict, config: P1R_Config, elapsed_sec: float):
    """분석 메타데이터 JSON 저장"""
    meta = {
        'script': 'P1R_GeoDistance_Metric_Analysis_20260428.py',
        'timestamp': datetime.now().isoformat(),
        'config': {
            'area': config.target_area,
            'freq_ghz': config.target_freq,
            'n_distance_bins': config.n_distance_bins,
            'max_pairs_sample': config.max_pairs_sample,
            'pdp_n_bins': config.pdp_n_bins,
        },
        'data_summary': {
            'n_ues': len(results['ue_ids']),
            'n_pairs_analyzed': len(results['pairs']),
            'distance_range_m': [
                float(results['geo_distances'].min()),
                float(results['geo_distances'].max()),
            ],
        },
        'correlations': {
            name: {k: float(v) if isinstance(v, (float, np.floating)) else v 
                   for k, v in corr.items()}
            for name, corr in results['correlations'].items()
        },
        'elapsed_sec': elapsed_sec,
    }
    
    filepath = os.path.join(config.OUTPUT_DIR, 'P1R_analysis_metadata.json')
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {os.path.basename(filepath)}")


# ===== SECTION 8: Main =====
def main():
    print("=" * 60)
    print("P1R: Geometrical Distance vs. Channel Metric Analysis")
    print("=" * 60)
    
    config = P1R_Config()
    config.print_config()
    
    t_start = time.time()
    
    # 분석 실행
    analyzer = GeoDistanceAnalyzer(config)
    results = analyzer.run_analysis()
    
    # 시각화
    print("\n[Visualization]")
    viz = P1R_Visualizer(config)
    viz.plot_spatial_map(results)
    viz.plot_all_metrics(results)
    viz.plot_binned_boxplot(results)
    viz.plot_correlation_summary(results)
    viz.plot_spatial_metric_maps(results)
    viz.plot_local_correlation_map(results)
    
    # 결과 저장
    print("\n[Export]")
    elapsed = time.time() - t_start
    save_correlation_csv(results, config)
    save_analysis_metadata(results, config, elapsed)
    
    print(f"\n총 소요 시간: {elapsed:.1f}s")
    print(f"결과 저장 위치: {config.OUTPUT_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
