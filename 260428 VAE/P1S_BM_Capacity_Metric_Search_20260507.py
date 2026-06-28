#!/usr/bin/env python3
"""
P1S_BM_Capacity_Metric_Search_20260507.py

과제 3: Beam Management / Capacity 기준 최적 Metric 탐색
- "두 RX의 채널이 비슷하다"를 BM/Capacity 관점에서 가장 잘 포착하는 metric 탐색
- Ground Truth: Capacity 차이, Beam Set Jaccard, Cross-capacity loss
- 후보 Metric: Covariance, Eigenvalue, Chordal, Coupling, PDP, Geodesic, KL 등
- 평가: Spearman/Kendall 상관, ROC-AUC, Precision@K

입력 데이터:
  - P1P SWOMP 결과 CSV: Capacity (C_AE, C_UE, C_BS_hist), Beam Set (ue_beams, bs_beams)
  - P1I Weichselberger Chunks: R_UE [16×16], Omega [n_bs×n_ue], U_BS, U_UE, H_mean
  - P1B Valid RXs: tau, power (PDP)

출력:
  - Metric 순위표 (각 GT 기준)
  - ROC-AUC / Precision@K 결과
  - Correlation heatmap
  - 최적 metric 조합 탐색 결과

GPU 불필요: 모든 연산은 CPU (NumPy/SciPy) 기반
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
from scipy.stats import pearsonr, spearmanr, kendalltau
from scipy.linalg import logm, eigvalsh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

warnings.filterwarnings('ignore', category=RuntimeWarning)


# ===== SECTION 1: Configuration =====
class P1S_Config:
    """P1S BM/Capacity 기준 Metric 탐색 설정"""
    
    def __init__(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # === 입력 데이터 경로 ===
        self.DATA_BASE_DIR = os.path.join(os.path.dirname(script_dir),
                                           "251009_CCM_Collection (8 GB)")
        
        # P1B Valid RXs
        self.P1B_INPUT_DIR = os.path.join(self.DATA_BASE_DIR, "P1B_Valid_Results")
        self.P1B_FILE_PATTERN = "Area{area}_{freq}GHz_Rays_Valid_RXs.npz"
        
        # P1I Weichselberger Chunks
        self.P1I_DIR = os.path.join(os.path.dirname(script_dir),
                                     "251020 BM QIE (Weichselberger)",
                                     "P1I_Weichsel_Chunk_Results")
        
        # P1P SWOMP 결과 (1117 실행 사용 — 가장 안정적)
        self.P1P_DIR = os.path.join(os.path.dirname(script_dir),
                                     "251218 E_MIMO_BM", "PIP_Project")
        self.P1P_RUN_PREFIX = "A1_7_5GHz_P1P_1117_1500"
        
        # === 출력 경로 ===
        self.OUTPUT_DIR = os.path.join(script_dir, "P1S_BM_Metric_Results")
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        
        # === 실험 설정 ===
        self.target_area = 1
        self.target_freq = 7.5
        
        # Area Grid
        self.AREA_GRID_CONFIGS = {
            1: {
                'tx_position': [-51.561, -21.794, 19],
                'x_start': -136.138, 'x_stop': 58.862, 'num_x': 40,
                'y_start': -117.667, 'y_stop': 77.333, 'num_y': 40
            }
        }
        
        # === 분석 파라미터 ===
        self.max_pairs_sample = 50000   # 최대 쌍 수
        self.random_seed = 42
        self.pdp_n_bins = 128
        
        # BM 관련
        self.n_bs_layers = 64           # BS layer 수
        self.n_ue_trx = 4              # UE TRX 수
        self.bs_codebook_size = 64     # BS 코드북 크기 (per layer)
        self.ue_codebook_size = 16     # UE 코드북 크기 (per TRX)
        
        # Capacity loss threshold (GT3 이진 분류용)
        self.bm_loss_threshold = 0.10  # 10% loss 이하면 "빔 공유 가능"
        
        # 안테나 설정
        self.n_bs = 1024
        self.n_ue = 16
        
    def print_config(self):
        print(f"=== P1S Configuration ===")
        print(f"  Data base: {self.DATA_BASE_DIR}")
        print(f"  P1P run: {self.P1P_RUN_PREFIX}")
        print(f"  Area: {self.target_area}, Freq: {self.target_freq} GHz")
        print(f"  Max pairs: {self.max_pairs_sample}")
        print(f"  BM loss threshold: {self.bm_loss_threshold*100:.0f}%")
        print(f"  Output: {self.OUTPUT_DIR}")



# ===== SECTION 2: Data Loaders =====
class P1P_ResultLoader:
    """P1P SWOMP 결과 CSV 로딩 — Capacity + Beam Set 추출"""
    
    def __init__(self, config: P1S_Config):
        self.config = config
    
    def load_all_partitions(self) -> dict:
        """모든 파티션 CSV를 로딩하여 UE별 데이터 통합
        
        Returns:
            {ue_id: {
                'C_AE': float,      # Full digital capacity (UL)
                'C_UE': float,      # Stage 1 UE beam capacity (UL)
                'C_BS_final': float, # Stage 2 최종 capacity (DL, 64 layers)
                'ue_beams': list,   # UE 빔 인덱스 [4개]
                'bs_beams': list,   # BS 빔 인덱스 [64개]
            }}
        """
        pattern = os.path.join(self.config.P1P_DIR,
                               f"{self.config.P1P_RUN_PREFIX}_p*.csv")
        csv_files = sorted(glob.glob(pattern))
        
        if not csv_files:
            raise FileNotFoundError(f"P1P CSV not found: {pattern}")
        
        print(f"  P1P CSV 파일: {len(csv_files)}개")
        
        all_ues = {}
        for filepath in csv_files:
            with open(filepath, 'r') as f:
                header = f.readline().strip()
                for line in f:
                    parsed = self._parse_csv_line(line.strip())
                    if parsed is not None:
                        all_ues[parsed['ue']] = parsed
        
        print(f"  로딩 완료: {len(all_ues)} UEs")
        return all_ues
    
    def _parse_csv_line(self, line: str) -> Optional[dict]:
        """P1P CSV 한 줄 파싱 (복잡한 nested comma 구조 처리)"""
        try:
            # 첫 3개 필드: ue, C_AE, C_UE
            parts = line.split(',')
            ue_id = int(parts[0])
            c_ae = float(parts[1])
            c_ue = float(parts[2])
            
            # C_BS_hist: 큰따옴표로 감싸진 comma-separated 값
            # 나머지 필드도 큰따옴표로 감싸진 것들이 있음
            # 정규식으로 큰따옴표 블록 추출
            quoted_blocks = re.findall(r'"([^"]*)"', line)
            
            if len(quoted_blocks) < 4:
                return None
            
            # C_BS_hist (첫 번째 quoted block)
            c_bs_hist_str = quoted_blocks[0]
            c_bs_values = [float(x) for x in c_bs_hist_str.split(',')]
            c_bs_final = c_bs_values[-1] if c_bs_values else 0.0
            
            # lambda_UE (두 번째 quoted block) — 사용하지 않음
            
            # ue_beams (세 번째 quoted block)
            ue_beams_str = quoted_blocks[2]
            ue_beams = [int(x) for x in ue_beams_str.split(',')]
            
            # bs_beams (마지막 quoted block)
            bs_beams_str = quoted_blocks[-1]
            bs_beams = [int(x) for x in bs_beams_str.split(',')]
            
            # 유효성 검사: capacity가 너무 낮으면 무효 RX
            if c_ae < 1.0:
                return None
            
            return {
                'ue': ue_id,
                'C_AE': c_ae,
                'C_UE': c_ue,
                'C_BS_final': c_bs_final,
                'C_BS_hist': c_bs_values,
                'ue_beams': ue_beams,
                'bs_beams': bs_beams,
            }
        except (ValueError, IndexError):
            return None


class P1I_ChunkLoader:
    """P1I Weichselberger Chunk 데이터 로딩"""
    
    def __init__(self, config: P1S_Config):
        self.config = config
        self._index = None  # {ue_id: (filepath, idx)}
        self._cache = {}    # {filepath: np.load data}
    
    def build_index(self):
        """UE ID → (filepath, idx_in_chunk) 인덱스 구축"""
        if self._index is not None:
            return
        
        self._index = {}
        pattern = os.path.join(self.config.P1I_DIR,
            f"Area{self.config.target_area}_{self.config.target_freq}GHz_Weichsel_Chunk_*.npz")
        chunk_files = sorted(glob.glob(pattern))
        
        print(f"  P1I 청크 인덱스 구축: {len(chunk_files)}개 파일")
        for filepath in chunk_files:
            data = np.load(filepath, allow_pickle=True)
            self._cache[filepath] = data
            ue_indices = data['ue_indices'].tolist()
            for idx, uid in enumerate(ue_indices):
                self._index[uid] = (filepath, idx)
        print(f"  P1I 인덱스 완료: {len(self._index)}개 UE")
    
    def get_ue_data(self, ue_id: int) -> Optional[dict]:
        """UE 데이터 로딩 (R_UE, Omega, U_BS, U_UE, H_mean)"""
        self.build_index()
        if ue_id not in self._index:
            return None
        
        filepath, idx = self._index[ue_id]
        data = self._cache[filepath]
        
        return {
            'R_UE': data['P1F_R_UE'][idx],       # [16, 16] complex
            'U_BS': data['P1G_U_BS'][idx],       # [1024, 1024] complex
            'U_UE': data['P1G_U_UE'][idx],       # [16, 16] complex
            'Omega': data['P1G_Omega'][idx],     # [n_ue, n_bs] (DL) or [n_bs, n_ue]
            'H_mean': data['P1H_H_mean'][idx],   # [n_ue, n_bs] (DL)
        }
    
    def get_batch_r_ue(self, ue_ids: List[int]) -> Dict[int, np.ndarray]:
        """R_UE 일괄 로딩 (메모리 효율)"""
        self.build_index()
        result = {}
        ue_set = set(ue_ids)
        for filepath, data in self._cache.items():
            ue_indices = data['ue_indices'].tolist()
            r_ue_all = data['P1F_R_UE']
            for idx, uid in enumerate(ue_indices):
                if uid in ue_set:
                    result[uid] = r_ue_all[idx]
        return result
    
    def get_batch_omega(self, ue_ids: List[int]) -> Dict[int, np.ndarray]:
        """Omega 일괄 로딩"""
        self.build_index()
        result = {}
        ue_set = set(ue_ids)
        for filepath, data in self._cache.items():
            ue_indices = data['ue_indices'].tolist()
            omega_all = data['P1G_Omega']
            for idx, uid in enumerate(ue_indices):
                if uid in ue_set:
                    result[uid] = omega_all[idx]
        return result
    
    def get_batch_h_mean(self, ue_ids: List[int]) -> Dict[int, np.ndarray]:
        """H_mean 일괄 로딩"""
        self.build_index()
        result = {}
        ue_set = set(ue_ids)
        for filepath, data in self._cache.items():
            ue_indices = data['ue_indices'].tolist()
            h_mean_all = data['P1H_H_mean']
            for idx, uid in enumerate(ue_indices):
                if uid in ue_set:
                    result[uid] = h_mean_all[idx]
        return result


class P1B_PDPLoader:
    """P1B Valid RXs에서 PDP 데이터 로딩"""
    
    def __init__(self, config: P1S_Config):
        self.config = config
        self._data = None
    
    def load(self):
        """P1B NPZ 로딩"""
        if self._data is not None:
            return
        
        filename = self.config.P1B_FILE_PATTERN.format(
            area=self.config.target_area, freq=self.config.target_freq)
        filepath = os.path.join(self.config.P1B_INPUT_DIR, filename)
        
        if not os.path.exists(filepath):
            print(f"  ⚠ P1B 파일 없음: {filepath}")
            return
        
        data = np.load(filepath, allow_pickle=True)
        self._data = {
            'rx_indices': data['rx_indices'].tolist(),
            'tau': data['tau'],
            'power': data['power'],
        }
        print(f"  P1B 로딩 완료: {len(self._data['rx_indices'])} RXs")
    
    def get_pdp(self, ue_id: int, n_bins: int = 128) -> Optional[np.ndarray]:
        """UE의 PDP 히스토그램 반환"""
        self.load()
        if self._data is None or ue_id not in self._data['rx_indices']:
            return None
        
        pos = self._data['rx_indices'].index(ue_id)
        tau = self._data['tau'][pos]
        power = self._data['power'][pos]
        
        valid = power > 0
        if np.sum(valid) == 0:
            return np.zeros(n_bins)
        
        tau_v = tau[valid]
        power_v = power[valid]
        
        bin_edges = np.linspace(tau_v.min(), tau_v.max() + 1e-12, n_bins + 1)
        pdp, _ = np.histogram(tau_v, bins=bin_edges, weights=power_v)
        
        total = pdp.sum()
        if total > 0:
            pdp = pdp / total
        return pdp



# ===== SECTION 3: Ground Truth Computation =====
class GroundTruthCalculator:
    """BM/Capacity 기준 Ground Truth 계산"""
    
    def __init__(self, p1p_data: dict, config: P1S_Config):
        self.p1p = p1p_data
        self.config = config
    
    def capacity_difference(self, ue_i: int, ue_j: int) -> float:
        """GT1: Normalized Capacity Difference
        
        |C(i) - C(j)| / max(C(i), C(j))
        사용 capacity: C_BS_final (최종 DL capacity, 64 layers)
        """
        c_i = self.p1p[ue_i]['C_BS_final']
        c_j = self.p1p[ue_j]['C_BS_final']
        max_c = max(c_i, c_j)
        if max_c < 1e-6:
            return 0.0
        return abs(c_i - c_j) / max_c
    
    def beam_set_jaccard_ue(self, ue_i: int, ue_j: int) -> float:
        """GT2a: UE Beam Set Jaccard Similarity
        
        |B_i ∩ B_j| / |B_i ∪ B_j|
        """
        b_i = set(self.p1p[ue_i]['ue_beams'])
        b_j = set(self.p1p[ue_j]['ue_beams'])
        
        intersection = len(b_i & b_j)
        union = len(b_i | b_j)
        if union == 0:
            return 0.0
        return intersection / union
    
    def beam_set_jaccard_bs(self, ue_i: int, ue_j: int) -> float:
        """GT2b: BS Beam Set Jaccard Similarity
        
        |B_i ∩ B_j| / |B_i ∪ B_j| (64개 BS 빔)
        """
        b_i = set(self.p1p[ue_i]['bs_beams'])
        b_j = set(self.p1p[ue_j]['bs_beams'])
        
        intersection = len(b_i & b_j)
        union = len(b_i | b_j)
        if union == 0:
            return 0.0
        return intersection / union
    
    def beam_set_overlap_ratio_bs(self, ue_i: int, ue_j: int) -> float:
        """GT2c: BS Beam Overlap Ratio (순서 고려)
        
        Layer별 빔 일치 비율 (64 layers 중 같은 빔 사용하는 layer 비율)
        """
        b_i = self.p1p[ue_i]['bs_beams']
        b_j = self.p1p[ue_j]['bs_beams']
        
        n = min(len(b_i), len(b_j))
        if n == 0:
            return 0.0
        
        matches = sum(1 for k in range(n) if b_i[k] == b_j[k])
        return matches / n
    
    def capacity_profile_correlation(self, ue_i: int, ue_j: int) -> float:
        """GT3: Capacity History Profile Correlation
        
        C_BS_hist (64-dim vector) 간 Pearson correlation
        → 빔 추가에 따른 용량 증가 패턴이 유사한지
        """
        hist_i = np.array(self.p1p[ue_i]['C_BS_hist'])
        hist_j = np.array(self.p1p[ue_j]['C_BS_hist'])
        
        n = min(len(hist_i), len(hist_j))
        if n < 5:
            return 0.0
        
        hist_i = hist_i[:n]
        hist_j = hist_j[:n]
        
        if np.std(hist_i) < 1e-6 or np.std(hist_j) < 1e-6:
            return 0.0
        
        corr, _ = pearsonr(hist_i, hist_j)
        return float(corr)



# ===== SECTION 4: Candidate Metric Computation =====
class CandidateMetrics:
    """후보 채널 Metric 계산 (pairwise)"""
    
    # --- Covariance 기반 ---
    
    @staticmethod
    def cov_frobenius_distance(R_i: np.ndarray, R_j: np.ndarray) -> float:
        """M1: Normalized Frobenius distance between covariance matrices"""
        norm_i = np.linalg.norm(R_i, 'fro')
        norm_j = np.linalg.norm(R_j, 'fro')
        if norm_i < 1e-12 or norm_j < 1e-12:
            return 2.0
        return float(np.linalg.norm(R_i / norm_i - R_j / norm_j, 'fro'))
    
    @staticmethod
    def chordal_distance(R_i: np.ndarray, R_j: np.ndarray, k: int = 4) -> float:
        """M2: Chordal distance between top-k eigensubspaces"""
        eigvals_i, eigvecs_i = np.linalg.eigh(R_i)
        eigvals_j, eigvecs_j = np.linalg.eigh(R_j)
        
        U_i = eigvecs_i[:, -k:]
        U_j = eigvecs_j[:, -k:]
        
        inner = np.linalg.norm(U_i.conj().T @ U_j, 'fro') ** 2
        dist_sq = max(0.0, k - inner)
        return float(np.sqrt(dist_sq))
    
    @staticmethod
    def eigenvalue_distance(R_i: np.ndarray, R_j: np.ndarray) -> float:
        """M3: Normalized eigenvalue distribution L2 distance"""
        eig_i = np.sort(np.real(np.linalg.eigvalsh(R_i)))[::-1]
        eig_j = np.sort(np.real(np.linalg.eigvalsh(R_j)))[::-1]
        
        sum_i = np.sum(np.maximum(eig_i, 0))
        sum_j = np.sum(np.maximum(eig_j, 0))
        if sum_i < 1e-12 or sum_j < 1e-12:
            return 2.0
        
        eig_i_n = np.maximum(eig_i, 0) / sum_i
        eig_j_n = np.maximum(eig_j, 0) / sum_j
        return float(np.linalg.norm(eig_i_n - eig_j_n))
    
    @staticmethod
    def dominant_eigvec_alignment(R_i: np.ndarray, R_j: np.ndarray) -> float:
        """M4: Dominant eigenvector alignment (1 = aligned, 0 = orthogonal)
        
        |u1_i^H u1_j|^2
        """
        _, eigvecs_i = np.linalg.eigh(R_i)
        _, eigvecs_j = np.linalg.eigh(R_j)
        
        u1_i = eigvecs_i[:, -1]  # largest eigenvalue
        u1_j = eigvecs_j[:, -1]
        
        return float(np.abs(u1_i.conj() @ u1_j) ** 2)
    
    @staticmethod
    def geodesic_distance(R_i: np.ndarray, R_j: np.ndarray) -> float:
        """M5: Geodesic distance on SPD manifold
        
        d_g(R_i, R_j) = ||log(R_i^{-1/2} R_j R_i^{-1/2})||_F
        
        Note: R_UE [16×16]이므로 계산 가능. 특이 행렬 처리를 위해 regularization 적용.
        """
        eps = 1e-8
        n = R_i.shape[0]
        R_i_reg = R_i + eps * np.eye(n)
        R_j_reg = R_j + eps * np.eye(n)
        
        # R_i^{-1/2}
        eigvals, eigvecs = np.linalg.eigh(R_i_reg)
        eigvals = np.maximum(eigvals, eps)
        R_i_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.conj().T
        
        # R_i^{-1/2} R_j R_i^{-1/2}
        M = R_i_inv_sqrt @ R_j_reg @ R_i_inv_sqrt
        
        # log(M) — Hermitian이므로 고유값 분해 사용
        eigvals_m, eigvecs_m = np.linalg.eigh(M)
        eigvals_m = np.maximum(eigvals_m, eps)
        log_M = eigvecs_m @ np.diag(np.log(eigvals_m)) @ eigvecs_m.conj().T
        
        return float(np.linalg.norm(log_M, 'fro'))
    
    @staticmethod
    def kl_divergence_gaussian(R_i: np.ndarray, R_j: np.ndarray) -> float:
        """M6: Symmetrized KL divergence between zero-mean Gaussians
        
        KL_sym(R_i, R_j) = 0.5 * [KL(N(0,R_i)||N(0,R_j)) + KL(N(0,R_j)||N(0,R_i))]
        KL(N(0,A)||N(0,B)) = 0.5 * [tr(B^{-1}A) - n + log(det(B)/det(A))]
        """
        eps = 1e-8
        n = R_i.shape[0]
        R_i_reg = R_i + eps * np.eye(n)
        R_j_reg = R_j + eps * np.eye(n)
        
        # 고유값으로 안정적 계산
        eig_i = np.maximum(np.real(np.linalg.eigvalsh(R_i_reg)), eps)
        eig_j = np.maximum(np.real(np.linalg.eigvalsh(R_j_reg)), eps)
        
        # KL(i||j) = 0.5 * [tr(R_j^{-1} R_i) - n + log(det(R_j)/det(R_i))]
        # 간소화: 고유값 기반 근사 (대각화 가정 X, 정확한 trace 계산)
        try:
            R_j_inv = np.linalg.inv(R_j_reg)
            R_i_inv = np.linalg.inv(R_i_reg)
            
            tr_ji = np.real(np.trace(R_j_inv @ R_i_reg))
            tr_ij = np.real(np.trace(R_i_inv @ R_j_reg))
            
            logdet_i = np.sum(np.log(eig_i))
            logdet_j = np.sum(np.log(eig_j))
            
            kl_ij = 0.5 * (tr_ji - n + logdet_j - logdet_i)
            kl_ji = 0.5 * (tr_ij - n + logdet_i - logdet_j)
            
            return float(0.5 * (kl_ij + kl_ji))
        except np.linalg.LinAlgError:
            return 100.0  # fallback
    
    # --- Weichselberger 기반 ---
    
    @staticmethod
    def coupling_matrix_distance(Omega_i: np.ndarray, Omega_j: np.ndarray) -> float:
        """M7: Coupling matrix normalized Frobenius distance"""
        norm_i = np.linalg.norm(Omega_i, 'fro')
        norm_j = np.linalg.norm(Omega_j, 'fro')
        if norm_i < 1e-12 or norm_j < 1e-12:
            return 2.0
        return float(np.linalg.norm(Omega_i / norm_i - Omega_j / norm_j, 'fro'))
    
    @staticmethod
    def mean_channel_distance(H_i: np.ndarray, H_j: np.ndarray) -> float:
        """M8: Mean channel normalized Frobenius distance"""
        norm_i = np.linalg.norm(H_i, 'fro')
        norm_j = np.linalg.norm(H_j, 'fro')
        if norm_i < 1e-12 and norm_j < 1e-12:
            return 0.0
        if norm_i < 1e-12 or norm_j < 1e-12:
            return 2.0
        return float(np.linalg.norm(H_i / norm_i - H_j / norm_j, 'fro'))
    
    # --- PDP 기반 ---
    
    @staticmethod
    def pdp_correlation(pdp_i: np.ndarray, pdp_j: np.ndarray) -> float:
        """M9: PDP Pearson correlation (1=동일, 0=무관)"""
        if np.std(pdp_i) < 1e-12 or np.std(pdp_j) < 1e-12:
            return 0.0
        corr, _ = pearsonr(pdp_i, pdp_j)
        return float(corr)
    
    # --- 복합 ---
    
    @staticmethod
    def effective_rank_difference(R_i: np.ndarray, R_j: np.ndarray) -> float:
        """M10: Effective rank difference
        
        eff_rank(R) = exp(H(λ_normalized)) where H is Shannon entropy
        """
        def _eff_rank(R):
            eig = np.maximum(np.real(np.linalg.eigvalsh(R)), 0)
            eig_sum = np.sum(eig)
            if eig_sum < 1e-12:
                return 1.0
            p = eig / eig_sum
            p = p[p > 1e-12]
            entropy = -np.sum(p * np.log(p))
            return np.exp(entropy)
        
        return abs(_eff_rank(R_i) - _eff_rank(R_j))
    
    @staticmethod
    def capacity_distance_approx(R_i: np.ndarray, R_j: np.ndarray, 
                                  snr_db: float = 10.0) -> float:
        """M11: Approximate capacity distance (R_UE 기반)
        
        |C_approx(R_i) - C_approx(R_j)| where C ≈ log2(det(I + snr * R/tr(R)))
        """
        snr = 10 ** (snr_db / 10)
        n = R_i.shape[0]
        
        def _cap(R):
            tr_R = np.real(np.trace(R))
            if tr_R < 1e-12:
                return 0.0
            R_norm = R / tr_R * n  # normalize to tr=n
            eig = np.maximum(np.real(np.linalg.eigvalsh(R_norm)), 0)
            return float(np.sum(np.log2(1 + snr / n * eig)))
        
        return abs(_cap(R_i) - _cap(R_j))
    
    @staticmethod
    def geo_distance(coord_i: np.ndarray, coord_j: np.ndarray) -> float:
        """M12: Euclidean distance (baseline 비교용)"""
        return float(np.linalg.norm(coord_i - coord_j))



# ===== SECTION 5: Analysis Engine =====
class BMMetricAnalyzer:
    """BM/Capacity 기준 Metric 탐색 엔진"""
    
    def __init__(self, config: P1S_Config):
        self.config = config
        self.p1p_loader = P1P_ResultLoader(config)
        self.p1i_loader = P1I_ChunkLoader(config)
        self.pdp_loader = P1B_PDPLoader(config)
        
        # 결과 저장
        self.ue_ids = []
        self.p1p_data = {}
        self.gt_values = {}   # {gt_name: [n_pairs]}
        self.metric_values = {}  # {metric_name: [n_pairs]}
        self.pairs = []
    
    def prepare_data(self):
        """데이터 준비: P1P 결과 + P1I 인덱스 구축"""
        print("\n[1/5] 데이터 준비")
        
        # P1P 결과 로딩
        self.p1p_data = self.p1p_loader.load_all_partitions()
        
        # P1I 인덱스 구축
        self.p1i_loader.build_index()
        
        # P1P와 P1I 모두 존재하는 UE만 사용
        p1p_ues = set(self.p1p_data.keys())
        p1i_ues = set(self.p1i_loader._index.keys()) if self.p1i_loader._index else set()
        
        common_ues = sorted(p1p_ues & p1i_ues)
        self.ue_ids = common_ues
        print(f"  공통 UE (P1P ∩ P1I): {len(self.ue_ids)}")
        
        if len(self.ue_ids) < 10:
            raise ValueError(f"공통 UE 부족: {len(self.ue_ids)}")
    
    def select_pairs(self) -> List[Tuple[int, int]]:
        """분석할 (i, j) 쌍 선택"""
        print("\n[2/5] 쌍 선택")
        n = len(self.ue_ids)
        all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        
        if self.config.max_pairs_sample and len(all_pairs) > self.config.max_pairs_sample:
            rng = np.random.RandomState(self.config.random_seed)
            indices = rng.choice(len(all_pairs), self.config.max_pairs_sample, replace=False)
            self.pairs = [all_pairs[k] for k in sorted(indices)]
            print(f"  Sampled {len(self.pairs)} pairs from {len(all_pairs)} total")
        else:
            self.pairs = all_pairs
            print(f"  Using all {len(self.pairs)} pairs")
        
        return self.pairs
    
    def compute_ground_truths(self):
        """Ground Truth 계산"""
        print("\n[3/5] Ground Truth 계산")
        
        gt_calc = GroundTruthCalculator(self.p1p_data, self.config)
        n_pairs = len(self.pairs)
        
        # GT 배열 초기화
        gt_cap_diff = np.zeros(n_pairs)
        gt_jaccard_ue = np.zeros(n_pairs)
        gt_jaccard_bs = np.zeros(n_pairs)
        gt_overlap_bs = np.zeros(n_pairs)
        gt_cap_profile = np.zeros(n_pairs)
        
        t0 = time.time()
        for k, (i, j) in enumerate(self.pairs):
            uid_i = self.ue_ids[i]
            uid_j = self.ue_ids[j]
            
            gt_cap_diff[k] = gt_calc.capacity_difference(uid_i, uid_j)
            gt_jaccard_ue[k] = gt_calc.beam_set_jaccard_ue(uid_i, uid_j)
            gt_jaccard_bs[k] = gt_calc.beam_set_jaccard_bs(uid_i, uid_j)
            gt_overlap_bs[k] = gt_calc.beam_set_overlap_ratio_bs(uid_i, uid_j)
            gt_cap_profile[k] = gt_calc.capacity_profile_correlation(uid_i, uid_j)
            
            if (k + 1) % 10000 == 0:
                print(f"    GT: {k+1}/{n_pairs} ({time.time()-t0:.1f}s)")
        
        self.gt_values = {
            'Capacity Diff (norm)': gt_cap_diff,
            'Beam Jaccard (UE)': gt_jaccard_ue,
            'Beam Jaccard (BS)': gt_jaccard_bs,
            'Beam Overlap (BS)': gt_overlap_bs,
            'Cap Profile Corr': gt_cap_profile,
        }
        
        elapsed = time.time() - t0
        print(f"  GT 계산 완료 ({elapsed:.1f}s)")
        
        # GT 통계 출력
        for name, values in self.gt_values.items():
            print(f"    {name}: mean={np.mean(values):.4f}, "
                  f"std={np.std(values):.4f}, "
                  f"range=[{np.min(values):.4f}, {np.max(values):.4f}]")
    
    def compute_candidate_metrics(self):
        """후보 Metric 계산"""
        print("\n[4/5] 후보 Metric 계산")
        
        n_pairs = len(self.pairs)
        metrics = CandidateMetrics()
        
        # --- 데이터 일괄 로딩 ---
        print("  데이터 일괄 로딩...")
        r_ue_cache = self.p1i_loader.get_batch_r_ue(self.ue_ids)
        omega_cache = self.p1i_loader.get_batch_omega(self.ue_ids)
        h_mean_cache = self.p1i_loader.get_batch_h_mean(self.ue_ids)
        
        # PDP 캐시
        self.pdp_loader.load()
        pdp_cache = {}
        for uid in self.ue_ids:
            pdp = self.pdp_loader.get_pdp(uid, self.config.pdp_n_bins)
            if pdp is not None:
                pdp_cache[uid] = pdp
        print(f"  캐시 완료: R_UE={len(r_ue_cache)}, Omega={len(omega_cache)}, "
              f"H_mean={len(h_mean_cache)}, PDP={len(pdp_cache)}")
        
        # 좌표 계산 (Geo distance용)
        # ※ 검증 완료: P1A의 RX 생성 순서는 x-major (for y: for x:)
        #   → x_idx = idx % num_x, y_idx = idx // num_x 이 정확함
        grid = self.config.AREA_GRID_CONFIGS[self.config.target_area]
        x_coords = np.linspace(grid['x_start'], grid['x_stop'], grid['num_x'])
        y_coords = np.linspace(grid['y_start'], grid['y_stop'], grid['num_y'])
        
        def uid_to_coord(uid):
            idx = uid - 1
            x_idx = idx % grid['num_x']
            y_idx = idx // grid['num_x']
            return np.array([x_coords[x_idx], y_coords[y_idx]])
        
        # --- Metric 배열 초기화 ---
        m_cov_frob = np.full(n_pairs, np.nan)
        m_chordal = np.full(n_pairs, np.nan)
        m_eigval = np.full(n_pairs, np.nan)
        m_dom_eigvec = np.full(n_pairs, np.nan)
        m_geodesic = np.full(n_pairs, np.nan)
        m_kl_div = np.full(n_pairs, np.nan)
        m_coupling = np.full(n_pairs, np.nan)
        m_mean_ch = np.full(n_pairs, np.nan)
        m_pdp_corr = np.full(n_pairs, np.nan)
        m_eff_rank = np.full(n_pairs, np.nan)
        m_cap_approx = np.full(n_pairs, np.nan)
        m_geo_dist = np.full(n_pairs, np.nan)
        
        # --- Pairwise 계산 ---
        print("  Pairwise metric 계산 중...")
        t0 = time.time()
        
        for k, (i, j) in enumerate(self.pairs):
            uid_i = self.ue_ids[i]
            uid_j = self.ue_ids[j]
            
            # Geo distance (항상 계산 가능)
            coord_i = uid_to_coord(uid_i)
            coord_j = uid_to_coord(uid_j)
            m_geo_dist[k] = metrics.geo_distance(coord_i, coord_j)
            
            # R_UE 기반 metrics
            if uid_i in r_ue_cache and uid_j in r_ue_cache:
                R_i = r_ue_cache[uid_i]
                R_j = r_ue_cache[uid_j]
                
                m_cov_frob[k] = metrics.cov_frobenius_distance(R_i, R_j)
                m_chordal[k] = metrics.chordal_distance(R_i, R_j, k=4)
                m_eigval[k] = metrics.eigenvalue_distance(R_i, R_j)
                m_dom_eigvec[k] = metrics.dominant_eigvec_alignment(R_i, R_j)
                m_eff_rank[k] = metrics.effective_rank_difference(R_i, R_j)
                m_cap_approx[k] = metrics.capacity_distance_approx(R_i, R_j)
                
                # Geodesic/KL: R_UE 16×16이므로 전체 계산 가능 (~0.1ms/pair)
                m_geodesic[k] = metrics.geodesic_distance(R_i, R_j)
                m_kl_div[k] = metrics.kl_divergence_gaussian(R_i, R_j)
            
            # Omega 기반
            if uid_i in omega_cache and uid_j in omega_cache:
                m_coupling[k] = metrics.coupling_matrix_distance(
                    omega_cache[uid_i], omega_cache[uid_j])
            
            # H_mean 기반
            if uid_i in h_mean_cache and uid_j in h_mean_cache:
                m_mean_ch[k] = metrics.mean_channel_distance(
                    h_mean_cache[uid_i], h_mean_cache[uid_j])
            
            # PDP 기반
            if uid_i in pdp_cache and uid_j in pdp_cache:
                m_pdp_corr[k] = metrics.pdp_correlation(
                    pdp_cache[uid_i], pdp_cache[uid_j])
            
            # 진행률
            if (k + 1) % 5000 == 0:
                elapsed = time.time() - t0
                eta = elapsed / (k + 1) * (n_pairs - k - 1)
                print(f"    {k+1}/{n_pairs} ({elapsed:.1f}s, 남은 ~{eta:.0f}s)")
        
        self.metric_values = {
            'Cov Frobenius (R_UE)': m_cov_frob,
            'Chordal Dist (R_UE)': m_chordal,
            'Eigenvalue Dist (R_UE)': m_eigval,
            'Dom Eigvec Align (R_UE)': m_dom_eigvec,
            'Geodesic Dist (R_UE)': m_geodesic,
            'KL Divergence (R_UE)': m_kl_div,
            'Coupling Mat Dist (Ω)': m_coupling,
            'Mean Channel Dist (H̄)': m_mean_ch,
            'PDP Correlation': m_pdp_corr,
            'Effective Rank Diff': m_eff_rank,
            'Capacity Approx Diff': m_cap_approx,
            'Geo Distance [m]': m_geo_dist,
        }
        
        elapsed = time.time() - t0
        print(f"  Metric 계산 완료 ({elapsed:.1f}s)")
    
    def evaluate_metrics(self) -> dict:
        """Metric vs. Ground Truth 상관 분석 + ROC-AUC"""
        print("\n[5/5] 평가")
        
        results = {}
        
        for gt_name, gt_vals in self.gt_values.items():
            results[gt_name] = {}
            
            # GT 방향 결정: distance metric은 GT와 양의 상관 기대
            # similarity metric (Jaccard, Overlap, Profile Corr)은 음의 상관 기대
            is_similarity_gt = gt_name in ['Beam Jaccard (UE)', 'Beam Jaccard (BS)',
                                            'Beam Overlap (BS)', 'Cap Profile Corr']
            
            for m_name, m_vals in self.metric_values.items():
                # 유효 데이터만
                valid = ~np.isnan(m_vals) & ~np.isnan(gt_vals)
                n_valid = np.sum(valid)
                
                if n_valid < 100:
                    continue
                
                g = gt_vals[valid]
                m = m_vals[valid]
                
                # Spearman correlation
                rho, p_rho = spearmanr(m, g)
                
                # Kendall tau (큰 데이터셋에서는 느리므로 서브샘플)
                if n_valid > 10000:
                    rng = np.random.RandomState(42)
                    sub_idx = rng.choice(n_valid, 10000, replace=False)
                    tau, p_tau = kendalltau(m[sub_idx], g[sub_idx])
                else:
                    tau, p_tau = kendalltau(m, g)
                
                # ROC-AUC: "빔 공유 가능" 이진 분류
                # Similarity GT: threshold 이상이면 positive
                # Distance GT: threshold 이하이면 positive
                auc = self._compute_roc_auc(m, g, is_similarity_gt, m_name, 
                                            gt_name=gt_name)
                
                # Precision@K
                p_at_10 = self._precision_at_k(m, g, is_similarity_gt, m_name, 
                                               k=10, gt_name=gt_name)
                
                results[gt_name][m_name] = {
                    'spearman_rho': float(rho),
                    'spearman_p': float(p_rho),
                    'kendall_tau': float(tau),
                    'kendall_p': float(p_tau),
                    'roc_auc': auc,
                    'precision_at_10': p_at_10,
                    'n_valid': int(n_valid),
                }
        
        # 결과 출력
        self._print_results(results)
        return results
    
    def _compute_roc_auc(self, metric: np.ndarray, gt: np.ndarray,
                          is_similarity_gt: bool, metric_name: str,
                          gt_name: str = '') -> float:
        """ROC-AUC 계산
        
        Binary label 정의:
        - BS GT (Jaccard/Overlap): 고정 threshold (sparse 분포 대응)
        - 기타 similarity GT: 상위 20%
        - Distance GT: 하위 20%
        """
        try:
            from sklearn.metrics import roc_auc_score
        except ImportError:
            return self._manual_auc(metric, gt, is_similarity_gt)
        
        # BS GT는 sparse 분포 → 고정 threshold 사용
        if gt_name in ['Beam Jaccard (BS)', 'Beam Overlap (BS)']:
            # "하나라도 빔 공유" = positive (Jaccard > 0)
            labels = (gt > 0).astype(int)
        elif gt_name == 'Beam Jaccard (UE)':
            # UE 빔 4개 중 1개 이상 공유 (Jaccard > 0)
            labels = (gt > 0).astype(int)
        elif is_similarity_gt:
            threshold = np.percentile(gt, 80)
            labels = (gt >= threshold).astype(int)
        else:
            threshold = np.percentile(gt, 20)
            labels = (gt <= threshold).astype(int)
        
        if np.sum(labels) < 10 or np.sum(1 - labels) < 10:
            return 0.5
        
        # Metric 방향: distance metric은 작을수록 "비슷" → score = -metric
        is_distance_metric = metric_name not in ['Dom Eigvec Align (R_UE)', 
                                                   'PDP Correlation']
        scores = -metric if is_distance_metric else metric
        
        try:
            return float(roc_auc_score(labels, scores))
        except ValueError:
            return 0.5
    
    def _manual_auc(self, metric: np.ndarray, gt: np.ndarray,
                     is_similarity_gt: bool) -> float:
        """sklearn 없을 때 간단한 AUC 근사"""
        if is_similarity_gt:
            threshold = np.percentile(gt, 80)
            labels = gt >= threshold
        else:
            threshold = np.percentile(gt, 20)
            labels = gt <= threshold
        
        pos = metric[labels]
        neg = metric[~labels]
        
        if len(pos) == 0 or len(neg) == 0:
            return 0.5
        
        # Mann-Whitney U statistic
        n_pos = len(pos)
        n_neg = len(neg)
        # 샘플링으로 근사
        rng = np.random.RandomState(42)
        n_sample = min(5000, n_pos, n_neg)
        pos_s = rng.choice(pos, n_sample, replace=True)
        neg_s = rng.choice(neg, n_sample, replace=True)
        
        u = np.mean(pos_s < neg_s) + 0.5 * np.mean(pos_s == neg_s)
        return float(u)
    
    def _precision_at_k(self, metric: np.ndarray, gt: np.ndarray,
                         is_similarity_gt: bool, metric_name: str, k: int = 10,
                         gt_name: str = '') -> float:
        """Precision@K: metric 기준 가장 가까운 K개 중 GT positive 비율
        
        각 UE에 대해 metric 기준 가장 가까운 K개 이웃 중
        GT 기준으로도 가까운(positive) 비율의 평균
        """
        # 간소화: 전체 쌍에서 metric 기준 상위/하위 K% 중 GT positive 비율
        n = len(metric)
        k_count = max(10, int(n * 0.01))  # 상위 1%
        
        # Metric 정렬 (distance면 오름차순, similarity면 내림차순)
        is_distance_metric = metric_name not in ['Dom Eigvec Align (R_UE)',
                                                   'PDP Correlation']
        
        if is_distance_metric:
            top_k_idx = np.argsort(metric)[:k_count]
        else:
            top_k_idx = np.argsort(metric)[-k_count:]
        
        # GT positive 정의 (BS GT는 고정 threshold)
        if gt_name in ['Beam Jaccard (BS)', 'Beam Overlap (BS)']:
            gt_positive = gt > 0
        elif gt_name == 'Beam Jaccard (UE)':
            gt_positive = gt > 0
        elif is_similarity_gt:
            threshold = np.percentile(gt, 80)
            gt_positive = gt >= threshold
        else:
            threshold = np.percentile(gt, 20)
            gt_positive = gt <= threshold
        
        precision = np.mean(gt_positive[top_k_idx])
        return float(precision)
    
    def _print_results(self, results: dict):
        """결과 요약 출력"""
        print("\n" + "=" * 80)
        print("  METRIC RANKING (by |Spearman ρ|)")
        print("=" * 80)
        
        for gt_name, metrics in results.items():
            print(f"\n  GT: {gt_name}")
            print(f"  {'Metric':<30} {'Spearman ρ':>12} {'Kendall τ':>12} "
                  f"{'ROC-AUC':>10} {'P@10':>8}")
            print(f"  {'-'*30} {'-'*12} {'-'*12} {'-'*10} {'-'*8}")
            
            # |Spearman|로 정렬
            sorted_metrics = sorted(metrics.items(),
                                     key=lambda x: abs(x[1]['spearman_rho']),
                                     reverse=True)
            
            for m_name, vals in sorted_metrics:
                print(f"  {m_name:<30} {vals['spearman_rho']:>+12.4f} "
                      f"{vals['kendall_tau']:>+12.4f} "
                      f"{vals['roc_auc']:>10.4f} "
                      f"{vals['precision_at_10']:>8.4f}")
    
    def compute_composite_metrics(self, results: dict):
        """Composite Metric: 상위 metric 조합으로 Logistic Regression 학습
        
        상위 metric을 z-score 정규화 후 결합하여 단일 metric 대비 AUC 개선 확인.
        동일 CV 조건에서 단일 best metric과 직접 비교 + ablation study.
        """
        print("\n[5.5/5] Composite Metric 탐색 (Logistic Regression)")
        
        try:
            from sklearn.linear_model import LogisticRegressionCV
            from sklearn.preprocessing import StandardScaler
            from sklearn.model_selection import cross_val_score, StratifiedKFold
        except ImportError:
            print("  ⚠ sklearn 미설치 — Composite metric 건너뜀")
            return {}
        
        # 상위 metric 선택 (Cap Profile Corr 기준 상위 + Geo Distance)
        top_metric_names = [
            'Eigenvalue Dist (R_UE)',
            'Capacity Approx Diff',
            'Mean Channel Dist (H̄)',
            'Coupling Mat Dist (Ω)',
            'Cov Frobenius (R_UE)',
            'Geo Distance [m]',
        ]
        
        # Feature matrix 구성
        available = [m for m in top_metric_names if m in self.metric_values]
        X_all = np.column_stack([self.metric_values[m] for m in available])
        valid_mask = ~np.any(np.isnan(X_all), axis=1)
        X_valid = X_all[valid_mask]
        
        print(f"  Features: {len(available)}개 metric, {np.sum(valid_mask)} valid pairs")
        
        # z-score 정규화
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_valid)
        
        # --- Feature 간 상관 진단 (multicollinearity) ---
        print("\n  [Feature Correlation Matrix]")
        corr_matrix = np.corrcoef(X_scaled.T)
        print(f"  {'':>25}", end='')
        for m in available:
            print(f" {m[:8]:>8}", end='')
        print()
        for i, m_i in enumerate(available):
            print(f"  {m_i:>25}", end='')
            for j in range(len(available)):
                print(f" {corr_matrix[i,j]:>8.3f}", end='')
            print()
        
        # 높은 상관 쌍 경고
        for i in range(len(available)):
            for j in range(i+1, len(available)):
                if abs(corr_matrix[i,j]) > 0.7:
                    print(f"  ⚠ 높은 상관: {available[i]} ↔ {available[j]} "
                          f"(r={corr_matrix[i,j]:.3f})")
        
        composite_results = {}
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for gt_name, gt_vals in self.gt_values.items():
            gt_valid = gt_vals[valid_mask]
            
            # Binary label 생성 (GT별 적절한 threshold)
            if gt_name in ['Beam Jaccard (BS)', 'Beam Overlap (BS)', 'Beam Jaccard (UE)']:
                y = (gt_valid > 0).astype(int)
            elif gt_name in ['Cap Profile Corr']:
                threshold = np.percentile(gt_valid, 80)
                y = (gt_valid >= threshold).astype(int)
            else:
                threshold = np.percentile(gt_valid, 20)
                y = (gt_valid <= threshold).astype(int)
            
            # 클래스 불균형 체크
            n_pos = np.sum(y)
            n_neg = np.sum(1 - y)
            if n_pos < 50 or n_neg < 50:
                print(f"\n  {gt_name}: 클래스 불균형 (pos={n_pos}, neg={n_neg}) — 건너뜀")
                continue
            
            # --- (A) Composite: 전체 feature ---
            lr = LogisticRegressionCV(Cs=5, cv=3, max_iter=1000, solver='lbfgs',
                                      random_state=42)
            cv_aucs = cross_val_score(lr, X_scaled, y, cv=cv, scoring='roc_auc')
            
            # 전체 데이터로 학습하여 계수 확인
            lr.fit(X_scaled, y)
            coefs = lr.coef_[0]
            
            # --- (B) 단일 metric baseline (동일 CV 조건) ---
            single_aucs = {}
            for feat_idx, feat_name in enumerate(available):
                X_single = X_scaled[:, feat_idx:feat_idx+1]
                from sklearn.linear_model import LogisticRegression
                lr_single = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs')
                single_cv = cross_val_score(lr_single, X_single, y, cv=cv, scoring='roc_auc')
                single_aucs[feat_name] = float(np.mean(single_cv))
            
            best_single_name = max(single_aucs, key=single_aucs.get)
            best_single_auc = single_aucs[best_single_name]
            
            # --- (C) Ablation: feature 하나씩 제거 ---
            ablation = {}
            for drop_idx, drop_name in enumerate(available):
                keep_idx = [i for i in range(len(available)) if i != drop_idx]
                X_ablated = X_scaled[:, keep_idx]
                lr_abl = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs')
                abl_cv = cross_val_score(lr_abl, X_ablated, y, cv=cv, scoring='roc_auc')
                ablation[drop_name] = float(np.mean(cv_aucs)) - float(np.mean(abl_cv))
            
            composite_results[gt_name] = {
                'cv_auc_mean': float(np.mean(cv_aucs)),
                'cv_auc_std': float(np.std(cv_aucs)),
                'cv_aucs': cv_aucs.tolist(),
                'coefficients': {m: float(c) for m, c in zip(available, coefs)},
                'best_single_metric': best_single_name,
                'best_single_auc': best_single_auc,
                'single_metric_aucs': single_aucs,
                'improvement_over_single': float(np.mean(cv_aucs)) - best_single_auc,
                'ablation_importance': ablation,
                'n_pos': int(n_pos),
                'n_neg': int(n_neg),
            }
            
            # 출력
            improvement = float(np.mean(cv_aucs)) - best_single_auc
            print(f"\n  {gt_name}:")
            print(f"    Composite AUC = {np.mean(cv_aucs):.4f} ± {np.std(cv_aucs):.4f}")
            print(f"    Best single   = {best_single_auc:.4f} ({best_single_name})")
            print(f"    Improvement   = +{improvement:.4f}")
            print(f"    (pos={n_pos}, neg={n_neg})")
            
            # 계수 (|coef| 순)
            sorted_coefs = sorted(zip(available, coefs), 
                                   key=lambda x: abs(x[1]), reverse=True)
            print(f"    Coefficients:")
            for m_name, coef in sorted_coefs:
                print(f"      {m_name:<28} {coef:+.4f}")
            
            # Ablation (중요도 순)
            sorted_abl = sorted(ablation.items(), key=lambda x: x[1], reverse=True)
            print(f"    Ablation (drop → AUC loss):")
            for m_name, loss in sorted_abl[:4]:
                print(f"      -{m_name:<27} Δ={loss:+.4f}")
        
        # Feature correlation matrix 저장
        composite_results['_feature_correlation'] = {
            'features': available,
            'correlation_matrix': corr_matrix.tolist(),
        }
        
        return composite_results
    
    def run(self) -> dict:
        """전체 분석 실행"""
        self.prepare_data()
        self.select_pairs()
        self.compute_ground_truths()
        self.compute_candidate_metrics()
        results = self.evaluate_metrics()
        composite = self.compute_composite_metrics(results)
        return results, composite



# ===== SECTION 6: Visualization =====
class P1S_Visualizer:
    """결과 시각화"""
    
    def __init__(self, config: P1S_Config):
        self.config = config
        plt.rcParams.update({
            'font.size': 9,
            'axes.titlesize': 11,
            'axes.labelsize': 10,
            'figure.dpi': 150,
        })
    
    def plot_correlation_heatmap(self, results: dict):
        """GT vs. Metric 상관계수 히트맵"""
        gt_names = list(results.keys())
        
        # 모든 metric 이름 수집
        all_metrics = set()
        for gt_data in results.values():
            all_metrics.update(gt_data.keys())
        metric_names = sorted(all_metrics)
        
        # Spearman ρ 행렬 구성
        rho_matrix = np.full((len(gt_names), len(metric_names)), np.nan)
        for i, gt_name in enumerate(gt_names):
            for j, m_name in enumerate(metric_names):
                if m_name in results[gt_name]:
                    rho_matrix[i, j] = results[gt_name][m_name]['spearman_rho']
        
        fig, ax = plt.subplots(figsize=(14, 6))
        im = ax.imshow(rho_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        
        ax.set_xticks(range(len(metric_names)))
        ax.set_xticklabels(metric_names, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(len(gt_names)))
        ax.set_yticklabels(gt_names, fontsize=9)
        
        # 값 표시
        for i in range(len(gt_names)):
            for j in range(len(metric_names)):
                if not np.isnan(rho_matrix[i, j]):
                    color = 'white' if abs(rho_matrix[i, j]) > 0.5 else 'black'
                    ax.text(j, i, f'{rho_matrix[i, j]:.3f}',
                           ha='center', va='center', fontsize=7, color=color)
        
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Spearman ρ')
        
        ax.set_title('Ground Truth vs. Candidate Metric: Spearman Correlation',
                     fontsize=12, fontweight='bold')
        fig.tight_layout()
        
        filepath = os.path.join(self.config.OUTPUT_DIR,
                                'P1S_fig1_correlation_heatmap.png')
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {os.path.basename(filepath)}")
    
    def plot_roc_auc_comparison(self, results: dict):
        """ROC-AUC 비교 bar chart"""
        gt_names = list(results.keys())
        
        all_metrics = set()
        for gt_data in results.values():
            all_metrics.update(gt_data.keys())
        metric_names = sorted(all_metrics)
        
        # AUC 행렬
        auc_matrix = np.full((len(gt_names), len(metric_names)), 0.5)
        for i, gt_name in enumerate(gt_names):
            for j, m_name in enumerate(metric_names):
                if m_name in results[gt_name]:
                    auc_matrix[i, j] = results[gt_name][m_name]['roc_auc']
        
        fig, axes = plt.subplots(len(gt_names), 1, figsize=(12, 4 * len(gt_names)))
        if len(gt_names) == 1:
            axes = [axes]
        
        for i, (gt_name, ax) in enumerate(zip(gt_names, axes)):
            aucs = auc_matrix[i]
            sorted_idx = np.argsort(aucs)[::-1]
            
            colors = ['steelblue' if aucs[j] > 0.6 else 
                      'coral' if aucs[j] < 0.4 else 'gray' 
                      for j in sorted_idx]
            
            bars = ax.barh(range(len(metric_names)),
                          [aucs[j] for j in sorted_idx],
                          color=colors, edgecolor='black', linewidth=0.5)
            
            ax.set_yticks(range(len(metric_names)))
            ax.set_yticklabels([metric_names[j] for j in sorted_idx], fontsize=8)
            ax.set_xlabel('ROC-AUC')
            ax.set_title(f'GT: {gt_name}')
            ax.axvline(x=0.5, color='red', linestyle='--', linewidth=1, label='Random')
            ax.set_xlim(0.3, 1.0)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3, axis='x')
            
            # 값 표시
            for bar_idx, j in enumerate(sorted_idx):
                ax.text(aucs[j] + 0.01, bar_idx, f'{aucs[j]:.3f}', 
                       va='center', fontsize=7)
        
        fig.suptitle('ROC-AUC: Metric Discriminative Power for BM Similarity',
                     fontsize=13, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        
        filepath = os.path.join(self.config.OUTPUT_DIR,
                                'P1S_fig2_roc_auc_comparison.png')
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {os.path.basename(filepath)}")
    
    def plot_top_metric_scatter(self, analyzer: 'BMMetricAnalyzer', results: dict):
        """상위 metric vs. GT scatter plot (대표 GT 2개)"""
        # 대표 GT 선택: Beam Jaccard (BS), Capacity Diff
        target_gts = ['Beam Jaccard (BS)', 'Capacity Diff (norm)']
        available_gts = [g for g in target_gts if g in results]
        if not available_gts:
            available_gts = list(results.keys())[:2]
        
        fig, axes = plt.subplots(len(available_gts), 3, figsize=(15, 5 * len(available_gts)))
        if len(available_gts) == 1:
            axes = axes.reshape(1, -1)
        
        for row, gt_name in enumerate(available_gts):
            # 상위 3개 metric 선택
            sorted_metrics = sorted(results[gt_name].items(),
                                     key=lambda x: abs(x[1]['spearman_rho']),
                                     reverse=True)[:3]
            
            gt_vals = analyzer.gt_values[gt_name]
            
            for col, (m_name, m_info) in enumerate(sorted_metrics):
                ax = axes[row, col]
                m_vals = analyzer.metric_values[m_name]
                
                valid = ~np.isnan(m_vals) & ~np.isnan(gt_vals)
                m = m_vals[valid]
                g = gt_vals[valid]
                
                # 서브샘플링 (시각화용)
                n_plot = min(5000, len(m))
                rng = np.random.RandomState(42)
                idx = rng.choice(len(m), n_plot, replace=False)
                
                ax.scatter(m[idx], g[idx], s=1, alpha=0.1, c='steelblue', rasterized=True)
                
                # Binned mean
                n_bins = 20
                bin_edges = np.linspace(np.percentile(m, 1), np.percentile(m, 99), n_bins + 1)
                bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
                bin_means = np.zeros(n_bins)
                for b in range(n_bins):
                    mask = (m >= bin_edges[b]) & (m < bin_edges[b + 1])
                    if np.sum(mask) > 10:
                        bin_means[b] = np.mean(g[mask])
                
                ax.plot(bin_centers, bin_means, 'r-', linewidth=2, label='Bin mean')
                
                ax.set_xlabel(m_name, fontsize=8)
                ax.set_ylabel(gt_name, fontsize=8)
                ax.set_title(f'ρ={m_info["spearman_rho"]:+.3f}, AUC={m_info["roc_auc"]:.3f}',
                            fontsize=9)
                ax.legend(fontsize=7)
                ax.grid(True, alpha=0.3)
        
        fig.suptitle('Top Metrics vs. Ground Truth (scatter + binned mean)',
                     fontsize=12, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        
        filepath = os.path.join(self.config.OUTPUT_DIR,
                                'P1S_fig3_top_metric_scatter.png')
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {os.path.basename(filepath)}")
    
    def plot_metric_ranking_summary(self, results: dict):
        """전체 GT에 대한 metric 종합 순위"""
        all_metrics = set()
        for gt_data in results.values():
            all_metrics.update(gt_data.keys())
        metric_names = sorted(all_metrics)
        gt_names = list(results.keys())
        
        # 각 metric의 평균 |Spearman ρ| 계산
        avg_abs_rho = {}
        for m_name in metric_names:
            rhos = []
            for gt_name in gt_names:
                if m_name in results[gt_name]:
                    rhos.append(abs(results[gt_name][m_name]['spearman_rho']))
            avg_abs_rho[m_name] = np.mean(rhos) if rhos else 0.0
        
        # 정렬
        sorted_metrics = sorted(avg_abs_rho.items(), key=lambda x: x[1], reverse=True)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        names = [x[0] for x in sorted_metrics]
        values = [x[1] for x in sorted_metrics]
        
        colors = plt.cm.viridis(np.linspace(0.8, 0.2, len(names)))
        bars = ax.barh(range(len(names)), values, color=colors, edgecolor='black', linewidth=0.5)
        
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel('Average |Spearman ρ| across all GTs')
        ax.set_title('Overall Metric Ranking\n(Higher = Better predictor of BM/Capacity similarity)',
                     fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        for i, v in enumerate(values):
            ax.text(v + 0.005, i, f'{v:.4f}', va='center', fontsize=8)
        
        fig.tight_layout()
        filepath = os.path.join(self.config.OUTPUT_DIR,
                                'P1S_fig4_metric_ranking.png')
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {os.path.basename(filepath)}")



    def plot_composite_comparison(self, composite: dict):
        """Composite vs. Single Best Metric AUC 비교 (fig5)"""
        if not composite:
            return
        
        # _feature_correlation 키 제외
        gt_names = [k for k in composite.keys() if not k.startswith('_')]
        if not gt_names:
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(gt_names))
        w = 0.35
        
        composite_aucs = [composite[g]['cv_auc_mean'] for g in gt_names]
        single_aucs = [composite[g]['best_single_auc'] for g in gt_names]
        single_names = [composite[g]['best_single_metric'] for g in gt_names]
        improvements = [composite[g]['improvement_over_single'] for g in gt_names]
        
        bars1 = ax.bar(x - w/2, single_aucs, w, label='Best Single Metric', 
                       color='lightcoral', edgecolor='black', linewidth=0.5)
        bars2 = ax.bar(x + w/2, composite_aucs, w, label='Composite (6 features)', 
                       color='steelblue', edgecolor='black', linewidth=0.5)
        
        # 값 표시
        for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
            ax.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 0.005,
                    f'{single_aucs[i]:.3f}', ha='center', va='bottom', fontsize=8)
            ax.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 0.005,
                    f'{composite_aucs[i]:.3f}', ha='center', va='bottom', fontsize=8,
                    fontweight='bold')
            # 개선량 표시
            if improvements[i] > 0.005:
                ax.annotate(f'+{improvements[i]:.3f}', 
                           xy=(x[i] + w/2, composite_aucs[i]),
                           xytext=(x[i] + w/2 + 0.15, composite_aucs[i] + 0.02),
                           fontsize=8, color='green', fontweight='bold',
                           arrowprops=dict(arrowstyle='->', color='green', lw=1))
        
        ax.set_xticks(x)
        ax.set_xticklabels([f'{g}\n(best: {n[:15]})' for g, n in zip(gt_names, single_names)],
                          fontsize=8, ha='center')
        ax.set_ylabel('ROC-AUC (5-fold CV)')
        ax.set_title('Composite Metric vs. Best Single Metric\n'
                     '(Same CV conditions, LogisticRegressionCV)',
                     fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Random')
        ax.set_ylim(0.45, max(max(composite_aucs), max(single_aucs)) + 0.08)
        ax.grid(True, alpha=0.3, axis='y')
        
        fig.tight_layout()
        filepath = os.path.join(self.config.OUTPUT_DIR,
                                'P1S_fig5_composite_vs_single.png')
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {os.path.basename(filepath)}")
    
    def plot_ablation(self, composite: dict):
        """Ablation study 시각화 (fig6) — feature 제거 시 AUC 손실"""
        if not composite:
            return
        
        gt_names = [k for k in composite.keys() 
                    if not k.startswith('_') and 'ablation_importance' in composite[k]]
        if not gt_names:
            return
        
        # 대표 GT 2개 선택 (가장 composite AUC 높은 것)
        gt_names_sorted = sorted(gt_names, 
                                  key=lambda g: composite[g]['cv_auc_mean'], reverse=True)
        target_gts = gt_names_sorted[:min(3, len(gt_names_sorted))]
        
        fig, axes = plt.subplots(1, len(target_gts), figsize=(7 * len(target_gts), 5))
        if len(target_gts) == 1:
            axes = [axes]
        
        for ax, gt_name in zip(axes, target_gts):
            ablation = composite[gt_name]['ablation_importance']
            sorted_abl = sorted(ablation.items(), key=lambda x: x[1], reverse=True)
            
            names = [x[0] for x in sorted_abl]
            values = [x[1] for x in sorted_abl]
            
            colors = ['darkred' if v > 0.01 else 'coral' if v > 0.005 else 'gray' 
                      for v in values]
            
            bars = ax.barh(range(len(names)), values, color=colors, 
                          edgecolor='black', linewidth=0.5)
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names, fontsize=8)
            ax.set_xlabel('AUC loss when removed')
            ax.set_title(f'{gt_name}\n(Composite AUC={composite[gt_name]["cv_auc_mean"]:.3f})',
                        fontsize=10)
            ax.axvline(x=0, color='black', linewidth=0.5)
            ax.grid(True, alpha=0.3, axis='x')
            
            for i, v in enumerate(values):
                ax.text(v + 0.001, i, f'{v:+.4f}', va='center', fontsize=7)
        
        fig.suptitle('Feature Ablation Study: AUC Loss When Each Feature is Removed',
                     fontsize=12, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        
        filepath = os.path.join(self.config.OUTPUT_DIR,
                                'P1S_fig6_ablation_study.png')
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {os.path.basename(filepath)}")


# ===== SECTION 7: Result Export =====
def save_results(results: dict, analyzer: 'BMMetricAnalyzer', config: P1S_Config, 
                 elapsed_sec: float, composite: dict = None):
    """결과 저장 (CSV + JSON)"""
    
    # --- CSV: 전체 결과 테이블 ---
    csv_path = os.path.join(config.OUTPUT_DIR, 'P1S_metric_evaluation.csv')
    
    def fmt_pval(p):
        """p-value 포맷: underflow 시 < 1e-300 표기"""
        if p == 0.0:
            return "< 1e-300"
        return f"{p:.2e}"
    
    with open(csv_path, 'w') as f:
        f.write("GT,Metric,Spearman_rho,Spearman_p,Kendall_tau,Kendall_p,"
                "ROC_AUC,Precision_at_10,N_valid\n")
        for gt_name, metrics in results.items():
            for m_name, vals in metrics.items():
                f.write(f"{gt_name},{m_name},"
                        f"{vals['spearman_rho']:.6f},{fmt_pval(vals['spearman_p'])},"
                        f"{vals['kendall_tau']:.6f},{fmt_pval(vals['kendall_p'])},"
                        f"{vals['roc_auc']:.6f},{vals['precision_at_10']:.6f},"
                        f"{vals['n_valid']}\n")
    print(f"  Saved: {os.path.basename(csv_path)}")
    
    # --- CSV: Metric 종합 순위 ---
    ranking_path = os.path.join(config.OUTPUT_DIR, 'P1S_metric_ranking.csv')
    
    all_metrics = set()
    for gt_data in results.values():
        all_metrics.update(gt_data.keys())
    
    avg_scores = {}
    for m_name in all_metrics:
        rhos = []
        aucs = []
        for gt_name in results:
            if m_name in results[gt_name]:
                rhos.append(abs(results[gt_name][m_name]['spearman_rho']))
                aucs.append(results[gt_name][m_name]['roc_auc'])
        avg_scores[m_name] = {
            'avg_abs_rho': np.mean(rhos) if rhos else 0,
            'avg_auc': np.mean(aucs) if aucs else 0.5,
        }
    
    sorted_metrics = sorted(avg_scores.items(), 
                            key=lambda x: x[1]['avg_abs_rho'], reverse=True)
    
    with open(ranking_path, 'w') as f:
        f.write("Rank,Metric,Avg_Abs_Spearman,Avg_ROC_AUC\n")
        for rank, (m_name, scores) in enumerate(sorted_metrics, 1):
            f.write(f"{rank},{m_name},{scores['avg_abs_rho']:.6f},"
                    f"{scores['avg_auc']:.6f}\n")
    print(f"  Saved: {os.path.basename(ranking_path)}")
    
    # --- JSON: 메타데이터 ---
    meta = {
        'script': 'P1S_BM_Capacity_Metric_Search_20260507.py',
        'timestamp': datetime.now().isoformat(),
        'config': {
            'area': config.target_area,
            'freq_ghz': config.target_freq,
            'max_pairs_sample': config.max_pairs_sample,
            'p1p_run': config.P1P_RUN_PREFIX,
            'bm_loss_threshold': config.bm_loss_threshold,
        },
        'data_summary': {
            'n_ues': len(analyzer.ue_ids),
            'n_pairs': len(analyzer.pairs),
            'n_ground_truths': len(results),
            'n_candidate_metrics': len(analyzer.metric_values),
        },
        'top_metrics': {
            gt_name: sorted(
                [(m, v['spearman_rho'], v['roc_auc']) 
                 for m, v in metrics.items()],
                key=lambda x: abs(x[1]), reverse=True
            )[:3]
            for gt_name, metrics in results.items()
        },
        'overall_ranking': [
            {'rank': r+1, 'metric': m, 'avg_abs_rho': s['avg_abs_rho'], 
             'avg_auc': s['avg_auc']}
            for r, (m, s) in enumerate(sorted_metrics[:5])
        ],
        'composite_metric': composite if composite else {},
        'elapsed_sec': elapsed_sec,
    }
    
    json_path = os.path.join(config.OUTPUT_DIR, 'P1S_analysis_metadata.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {os.path.basename(json_path)}")
    
    # --- CSV: Composite Metric 결과 ---
    if composite:
        comp_path = os.path.join(config.OUTPUT_DIR, 'P1S_composite_metric.csv')
        with open(comp_path, 'w') as f:
            f.write("GT,Composite_AUC,Composite_std,Best_Single_Metric,"
                    "Best_Single_AUC,Improvement,N_pos,N_neg\n")
            for gt_name, comp_data in composite.items():
                if gt_name.startswith('_'):
                    continue
                f.write(f"{gt_name},{comp_data['cv_auc_mean']:.6f},"
                        f"{comp_data['cv_auc_std']:.6f},"
                        f"{comp_data['best_single_metric']},"
                        f"{comp_data['best_single_auc']:.6f},"
                        f"{comp_data['improvement_over_single']:+.6f},"
                        f"{comp_data['n_pos']},{comp_data['n_neg']}\n")
        print(f"  Saved: {os.path.basename(comp_path)}")
        
        # Ablation CSV
        abl_path = os.path.join(config.OUTPUT_DIR, 'P1S_ablation_study.csv')
        with open(abl_path, 'w') as f:
            f.write("GT,Dropped_Feature,AUC_Loss\n")
            for gt_name, comp_data in composite.items():
                if gt_name.startswith('_') or 'ablation_importance' not in comp_data:
                    continue
                for feat, loss in sorted(comp_data['ablation_importance'].items(),
                                          key=lambda x: x[1], reverse=True):
                    f.write(f"{gt_name},{feat},{loss:+.6f}\n")
        print(f"  Saved: {os.path.basename(abl_path)}")


# ===== SECTION 8: Main =====
def main():
    print("=" * 70)
    print("P1S: BM/Capacity 기준 최적 Metric 탐색")
    print("  - GPU 불필요: 모든 연산 CPU (NumPy/SciPy) 기반")
    print("=" * 70)
    
    config = P1S_Config()
    config.print_config()
    
    t_start = time.time()
    
    # 분석 실행
    analyzer = BMMetricAnalyzer(config)
    results, composite = analyzer.run()
    
    # 시각화
    print("\n[Visualization]")
    viz = P1S_Visualizer(config)
    viz.plot_correlation_heatmap(results)
    viz.plot_roc_auc_comparison(results)
    viz.plot_top_metric_scatter(analyzer, results)
    viz.plot_metric_ranking_summary(results)
    viz.plot_composite_comparison(composite)
    viz.plot_ablation(composite)
    
    # 결과 저장
    print("\n[Export]")
    elapsed = time.time() - t_start
    save_results(results, analyzer, config, elapsed, composite)
    
    # 최종 요약
    print(f"\n{'=' * 70}")
    print(f"  총 소요 시간: {elapsed:.1f}s")
    print(f"  결과 저장: {config.OUTPUT_DIR}")
    print(f"{'=' * 70}")
    
    # Top-3 요약 출력
    print("\n  ★ 종합 Top-3 Metric (BM/Capacity 유사도 예측력):")
    all_metrics_set = set()
    for gt_data in results.values():
        all_metrics_set.update(gt_data.keys())
    
    avg_rho = {}
    for m in all_metrics_set:
        rhos = [abs(results[g][m]['spearman_rho']) 
                for g in results if m in results[g]]
        avg_rho[m] = np.mean(rhos) if rhos else 0
    
    for rank, (m, rho) in enumerate(sorted(avg_rho.items(), 
                                            key=lambda x: x[1], reverse=True)[:3], 1):
        print(f"    #{rank}: {m} (avg |ρ| = {rho:.4f})")


if __name__ == '__main__':
    main()
