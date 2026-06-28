#!/usr/bin/env python3
"""
P4_BF_Analysis_2509v2.py
P4: 빔포밍 결과 분석 및 TX/RX 빔 클러스터링

P3_BF_Pair_2509v2.py에서 생성된 빔포밍 채널 이득 데이터를 분석하여
TX/RX 빔 집합 유사도 기반 클러스터링을 수행합니다.

주요 기능:
- 유사도 기준 계층적 클러스터링
- 고유 빔 추출 및 자카드 유사도 계산
- TX/RX 빔 분리 분석
- CSV 우선 결과 출력
- 정사각형 단일 클러스터 공간 맵
- 간소화된 코드 구조

Author: AI Assistant
Date: 2024-09-21
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datetime import datetime
import json
from pathlib import Path
import glob
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import warnings
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from collections import Counter

# 경고 메시지 필터링
warnings.filterwarnings('ignore', category=FutureWarning)

# Matplotlib 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# ========================================================================
# CONFIGURATION CLASS - 간소화
# ========================================================================

@dataclass
class P4_Config:
    """P4 빔포밍 성능 분석 설정 - 핵심 파라미터만 유지"""
    
    # 파일 경로 설정
    INPUT_DIR: str = "P2A_BF_Results"
    OUTPUT_DIR: str = "P2B_Analysis_Results"
    
    # 분석 대상 설정
    TARGET_AREA: int = 1
    TARGET_FREQUENCY: float = 7.5
    
    # TX 빔 클러스터링 파라미터
    TX_TOP_K_BEAMS: int = 8                 # 상위 TX 빔 개수
    TX_MIN_CLUSTER_SIZE: int = 3            # TX 클러스터 최소 크기
    TX_MIN_INTRA_SIMILARITY: float = 0.4    # TX 클러스터 내 최소 유사도
    
    # RX 빔 클러스터링 파라미터
    RX_TOP_K_BEAMS: int = 4                 # 상위 RX 빔 개수
    RX_MIN_CLUSTER_SIZE: int = 3            # RX 클러스터 최소 크기
    RX_MIN_INTRA_SIMILARITY: float = 0.4    # RX 클러스터 내 최소 유사도
    
    # 클러스터링 활성화 설정
    ENABLE_TX_CLUSTERING: bool = True       # TX 빔 클러스터링 수행
    ENABLE_RX_CLUSTERING: bool = True       # RX 빔 클러스터링 수행
    
    # 빔 분석 파라미터
    TOP_BEAM_RANKS: int = 32                # 상위 빔 순위 개수
    LOG_SAFETY_EPSILON: float = 1e-18       # log 계산 안전성
    
    # 출력 설정
    CSV_PRIORITY: bool = True               # CSV 우선 저장
    SAVE_DPI: int = 300                     # 저장 DPI
    
    # RX 위치 설정 (P1과 일치)
    AREA1_RX_CONFIG = {
        'x_params': {'start': -136.138, 'stop': 58.862, 'num': 40},
        'y_params': {'start': -117.667, 'stop': 77.333, 'num': 40}, 
        'z_params': {'values': [1.5]},
    }
    AREA1_TX_POSITION = [-51.561, -21.794, 19]
    
    def __post_init__(self):
        """설정 후처리"""
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        
        # 총 RX 개수 계산
        x_num = self.AREA1_RX_CONFIG['x_params']['num']
        y_num = self.AREA1_RX_CONFIG['y_params']['num'] 
        z_num = len(self.AREA1_RX_CONFIG['z_params']['values'])
        self.TOTAL_RX_COUNT = x_num * y_num * z_num
        
        print(f"P4 빔포밍 분석 설정:")
        print(f"  - 분석 대상: Area{self.TARGET_AREA}_{self.TARGET_FREQUENCY}GHz")
        print(f"  - 총 RX 개수: {self.TOTAL_RX_COUNT}")
        print(f"  - TX 클러스터링: {self.TX_TOP_K_BEAMS}개 빔, 유사도 ≥ {self.TX_MIN_INTRA_SIMILARITY}")
        print(f"  - RX 클러스터링: {self.RX_TOP_K_BEAMS}개 빔, 유사도 ≥ {self.RX_MIN_INTRA_SIMILARITY}")

# 전역 설정 인스턴스
config = P4_Config()

# ========================================================================
# DATA LOADING - P3 결과 로딩
# ========================================================================

class P3DataLoader:
    """P3 결과 파일 로딩"""
    
    def __init__(self, config: P4_Config):
        self.config = config
    
    def load_p3_results(self) -> Dict[str, Any]:
        """P3 결과 파일들을 로딩하여 통합 데이터 구조로 반환"""
        area = self.config.TARGET_AREA
        freq = self.config.TARGET_FREQUENCY
        
        print(f"Loading P3 results for Area{area}_{freq}GHz...")
        
        results = {
            'beam_gains_by_rx': {},
            'loading_summary': {
                'area_index': area,
                'frequency_ghz': freq,
                'loaded_rx_count': 0,
                'failed_rx_count': 0,
                'total_expected': self.config.TOTAL_RX_COUNT
            }
        }
        
        # 개별 RX 빔 데이터 로딩
        success_count = 0
        fail_count = 0
        
        for rx_idx in range(1, self.config.TOTAL_RX_COUNT + 1):
            beam_data = self.load_rx_beam_data(area, freq, rx_idx)
            if beam_data is not None:
                results['beam_gains_by_rx'][rx_idx] = beam_data
                success_count += 1
            else:
                fail_count += 1
        
        # 개별 파일 로딩 실패 시 청크 파일에서 로딩
        if success_count == 0:
            print("Individual files not found, trying chunk files...")
            chunk_results = self.load_from_chunks(area, freq)
            if chunk_results:
                results['beam_gains_by_rx'].update(chunk_results)
                success_count = len(chunk_results)
                fail_count = self.config.TOTAL_RX_COUNT - success_count
        
        # 로딩 요약 업데이트
        results['loading_summary']['loaded_rx_count'] = success_count
        results['loading_summary']['failed_rx_count'] = fail_count
        
        print(f"P3 data loading completed: {success_count}/{self.config.TOTAL_RX_COUNT} RX loaded")
        
        if success_count == 0:
            raise FileNotFoundError(f"No P3 data found for Area{area}_{freq}GHz")
        
        return results
    
    def load_rx_beam_data(self, area: int, freq: float, rx_idx: int) -> Optional[Dict[str, Any]]:
        """개별 RX 빔 데이터 로딩"""
        filename = f"Area{area}_{freq}GHz_RX{rx_idx}_layer_beam_data.npz"
        filepath = os.path.join(self.config.INPUT_DIR, filename)
        
        try:
            with np.load(filepath, allow_pickle=True) as data:
                return {
                    'beam_gains': data['beam_gains'],
                    'tx_orientations': data['tx_orientations'],
                    'rx_orientations': data['rx_orientations'],
                    'layer_metadata': data['layer_metadata'].item(),
                    'rx_index': rx_idx,
                    'source_file': filename
                }
        except FileNotFoundError:
            return None
        except Exception as e:
            print(f"Warning: Failed to load {filename}: {e}")
            return None
    
    def load_from_chunks(self, area: int, freq: float) -> Dict[int, Dict[str, Any]]:
        """청크 파일에서 빔 데이터 로딩"""
        chunk_pattern = f"Area{area}_{freq}GHz_Layer_Beam_Data_RX*.npz"
        chunk_files = glob.glob(os.path.join(self.config.INPUT_DIR, chunk_pattern))
        
        beam_data_by_rx = {}
        
        for chunk_file in chunk_files:
            try:
                with np.load(chunk_file, allow_pickle=True) as data:
                    if 'rx_indices' in data:
                        rx_indices = data['rx_indices']
                        print(f"Loading chunk {os.path.basename(chunk_file)} with {len(rx_indices)} RX...")
                        
                        for rx_idx in rx_indices:
                            rx_idx_int = int(rx_idx)
                            beam_key = f'beam_data_rx_{rx_idx_int}'
                            
                            if beam_key in data:
                                individual_file = f"Area{area}_{freq}GHz_RX{rx_idx_int}_layer_beam_data.npz"
                                individual_path = os.path.join(self.config.INPUT_DIR, individual_file)
                                
                                if os.path.exists(individual_path):
                                    beam_file_data = self.load_rx_beam_data_direct(individual_path, rx_idx_int)
                                    if beam_file_data:
                                        beam_data_by_rx[rx_idx_int] = beam_file_data
                                else:
                                    try:
                                        beam_chunk_data = data[beam_key]
                                        if hasattr(beam_chunk_data, 'item'):
                                            beam_obj = beam_chunk_data.item()
                                            if isinstance(beam_obj, dict) and 'beam_gains' in beam_obj:
                                                beam_data_by_rx[rx_idx_int] = {
                                                    'beam_gains': beam_obj['beam_gains'],
                                                    'tx_orientations': beam_obj['tx_orientations'],
                                                    'rx_orientations': beam_obj['rx_orientations'],
                                                    'layer_metadata': beam_obj['layer_metadata'],
                                                    'rx_index': rx_idx_int,
                                                    'source_file': os.path.basename(chunk_file)
                                                }
                                    except Exception as inner_e:
                                        print(f"Warning: Failed to process RX{rx_idx_int} from chunk: {inner_e}")
            except Exception as e:
                print(f"Warning: Failed to load chunk file {chunk_file}: {e}")
        
        return beam_data_by_rx
    
    def load_rx_beam_data_direct(self, filepath: str, rx_idx: int) -> Optional[Dict[str, Any]]:
        """개별 빔 데이터 파일 직접 로딩"""
        try:
            with np.load(filepath, allow_pickle=True) as data:
                return {
                    'beam_gains': data['beam_gains'],
                    'tx_orientations': data['tx_orientations'], 
                    'rx_orientations': data['rx_orientations'],
                    'layer_metadata': data['layer_metadata'].item(),
                    'rx_index': rx_idx,
                    'source_file': os.path.basename(filepath)
                }
        except Exception as e:
            print(f"Warning: Failed to load individual file {filepath}: {e}")
            return None

# ========================================================================
# RX POSITION GENERATION - 수신기 위치 생성
# ========================================================================

class RXPositionGenerator:
    """P1 area 1 수신기 위치 정보 생성"""
    
    def __init__(self, config: P4_Config):
        self.config = config
        
    def generate_area1_rx_positions(self) -> Dict[int, Dict[str, Any]]:
        """Area 1 수신기 위치 정보 생성 (P1과 일치)"""
        rx_config = self.config.AREA1_RX_CONFIG
        
        # 그리드 좌표 생성
        x_coords = np.linspace(
            rx_config['x_params']['start'],
            rx_config['x_params']['stop'], 
            rx_config['x_params']['num']
        )
        y_coords = np.linspace(
            rx_config['y_params']['start'],
            rx_config['y_params']['stop'],
            rx_config['y_params']['num']
        )
        z_coords = rx_config['z_params']['values']
        
        positions = {}
        rx_idx = 1
        
        # P1과 동일한 순서로 위치 생성: z -> y -> x 순서
        for z in z_coords:
            for y in y_coords:
                for x in x_coords:
                    # 송신기와의 거리 계산
                    tx_pos = self.config.AREA1_TX_POSITION
                    distance_2d = np.sqrt((x - tx_pos[0])**2 + (y - tx_pos[1])**2)
                    distance_3d = np.sqrt((x - tx_pos[0])**2 + (y - tx_pos[1])**2 + (z - tx_pos[2])**2)
                    
                    positions[rx_idx] = {
                        'x': float(x),
                        'y': float(y), 
                        'z': float(z),
                        'distance_2d_m': float(distance_2d),
                        'distance_3d_m': float(distance_3d),
                        'grid_indices': {
                            'x_idx': int(np.where(np.isclose(x_coords, x))[0][0]),
                            'y_idx': int(np.where(np.isclose(y_coords, y))[0][0]),
                            'z_idx': int(z_coords.index(z))
                        }
                    }
                    rx_idx += 1
        
        print(f"Generated {len(positions)} RX positions for Area 1")
        return positions

# ========================================================================
# BEAM RANKING ANALYSIS - 빔 순위 분석
# ========================================================================

class BeamRankingAnalyzer:
    """빔 순위 분석 및 성능 평가"""
    
    def __init__(self, config: P4_Config):
        self.config = config
        
    def rank_beam_pairs(self, p3_results: Dict[str, Any], 
                       rx_positions: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        """단말별 빔 순위 분석"""
        
        print("Analyzing beam rankings...")
        
        analysis_results = {
            'rx_beam_rankings': {},
            'tx_beam_dominance': {},
            'system_statistics': {},
            'analysis_metadata': {
                'total_rx_analyzed': 0,
                'analysis_timestamp': datetime.now().isoformat(),
                'top_beam_ranks': self.config.TOP_BEAM_RANKS
            }
        }
        
        beam_gains_by_rx = p3_results['beam_gains_by_rx']
        
        # 단말별 빔 순위 분석
        for rx_idx, beam_data in beam_gains_by_rx.items():
            rx_analysis = self.analyze_rx_beam_ranking(rx_idx, beam_data, rx_positions)
            if rx_analysis:
                analysis_results['rx_beam_rankings'][rx_idx] = rx_analysis
        
        # 송신 빔별 최상위 단말 분석
        analysis_results['tx_beam_dominance'] = self.analyze_tx_beam_dominance(
            analysis_results['rx_beam_rankings']
        )
        
        # 전체 시스템 통계 계산
        analysis_results['system_statistics'] = self.compute_system_statistics(
            analysis_results['rx_beam_rankings'], analysis_results['tx_beam_dominance']
        )
        
        analysis_results['analysis_metadata']['total_rx_analyzed'] = len(analysis_results['rx_beam_rankings'])
        
        print(f"Beam ranking analysis completed for {len(analysis_results['rx_beam_rankings'])} RX")
        
        return analysis_results
    
    def analyze_rx_beam_ranking(self, rx_idx: int, beam_data: Dict[str, Any], 
                              rx_positions: Dict[int, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """개별 단말의 빔 순위 분석"""
        
        try:
            beam_gains = beam_data['beam_gains']  # Shape: (RX_BEAMS, TX_BEAMS)
            layer_metadata = beam_data['layer_metadata']
            
            # 빔 이득을 dB로 변환
            beam_gains_db = 10 * np.log10(np.maximum(beam_gains, self.config.LOG_SAFETY_EPSILON))
            
            # 모든 빔 페어의 이득과 인덱스 추출
            rx_beams, tx_beams = np.meshgrid(range(beam_gains.shape[0]), range(beam_gains.shape[1]), indexing='ij')
            rx_beams = rx_beams.flatten()
            tx_beams = tx_beams.flatten()
            gains_db = beam_gains_db.flatten()
            
            # 이득 기준 정렬 (내림차순)
            sort_indices = np.argsort(-gains_db)
            
            # 상위 빔 페어 추출
            top_count = min(self.config.TOP_BEAM_RANKS, len(sort_indices))
            top_beam_pairs = []
            
            for i in range(top_count):
                idx = sort_indices[i]
                rx_beam_idx = int(rx_beams[idx])
                tx_beam_idx = int(tx_beams[idx])
                gain_db = float(gains_db[idx])
                
                top_beam_pairs.append({
                    'rank': i + 1,
                    'rx_beam_idx': rx_beam_idx,
                    'tx_beam_idx': tx_beam_idx,
                    'gain_db': gain_db,
                    'gain_linear': float(beam_gains[rx_beam_idx, tx_beam_idx])
                })
            
            # 위치 정보 추가
            position_info = rx_positions.get(rx_idx, {})
            
            return {
                'rx_index': rx_idx,
                'position': position_info,
                'top_beam_pairs': top_beam_pairs,
                'best_gain_db': float(top_beam_pairs[0]['gain_db']),
                'best_tx_beam': int(top_beam_pairs[0]['tx_beam_idx']),
                'best_rx_beam': int(top_beam_pairs[0]['rx_beam_idx']),
                'total_valid_pairs': len(gains_db),
                'layer_info': {
                    'TX_BEAMS_PER_LAYER': int(layer_metadata['TX_BEAMS_PER_LAYER']),
                    'RX_BEAMS_PER_LAYER': int(layer_metadata['RX_BEAMS_PER_LAYER'])
                }
            }
            
        except Exception as e:
            print(f"Warning: Failed to analyze RX {rx_idx}: {e}")
            return None
    
    def analyze_tx_beam_dominance(self, rx_beam_rankings: Dict[int, Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        """송신 빔별 최상위 단말 분석"""
        
        tx_beam_dominance = {}
        
        # 각 단말의 최상위 송신 빔 수집
        for rx_idx, rx_analysis in rx_beam_rankings.items():
            if rx_analysis['top_beam_pairs']:
                best_tx_beam = rx_analysis['best_tx_beam']
                best_gain_db = rx_analysis['best_gain_db']
                position = rx_analysis['position']
                
                if best_tx_beam not in tx_beam_dominance:
                    tx_beam_dominance[best_tx_beam] = {
                        'tx_beam_idx': best_tx_beam,
                        'dominant_rx_list': [],
                        'rx_count': 0,
                        'avg_gain_db': 0.0,
                        'max_gain_db': -np.inf,
                        'min_gain_db': np.inf
                    }
                
                # 단말 정보 추가
                tx_beam_dominance[best_tx_beam]['dominant_rx_list'].append({
                    'rx_idx': rx_idx,
                    'gain_db': best_gain_db,
                    'position': position,
                    'distance_2d_m': position.get('distance_2d_m', 0)
                })
        
        # 송신 빔별 통계 계산
        for tx_beam_idx, beam_info in tx_beam_dominance.items():
            rx_list = beam_info['dominant_rx_list']
            gains_db = [rx['gain_db'] for rx in rx_list]
            
            beam_info.update({
                'rx_count': len(rx_list),
                'avg_gain_db': float(np.mean(gains_db)),
                'max_gain_db': float(np.max(gains_db)),
                'min_gain_db': float(np.min(gains_db)),
                'std_gain_db': float(np.std(gains_db)),
                'avg_distance_2d_m': float(np.mean([rx['distance_2d_m'] for rx in rx_list]))
            })
        
        # 단말 수 기준 정렬
        sorted_beams = dict(sorted(tx_beam_dominance.items(), 
                                 key=lambda x: x[1]['rx_count'], reverse=True))
        
        return sorted_beams
    
    def compute_system_statistics(self, rx_beam_rankings: Dict[int, Dict[str, Any]], 
                                tx_beam_dominance: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        """전체 시스템 통계 계산"""
        
        if not rx_beam_rankings:
            return {}
        
        # 전체 단말 이득 통계
        all_gains_db = [rx['best_gain_db'] for rx in rx_beam_rankings.values()]
        all_distances = [rx['position'].get('distance_2d_m', 0) for rx in rx_beam_rankings.values()]
        
        # 송신 빔 사용률 통계
        used_tx_beams = len(tx_beam_dominance)
        total_tx_beams = max(rx_beam_rankings.values(), 
                           key=lambda x: x['layer_info']['TX_BEAMS_PER_LAYER'])['layer_info']['TX_BEAMS_PER_LAYER']
        
        # 단말당 최상위 송신 빔 분포
        beam_usage_counts = [info['rx_count'] for info in tx_beam_dominance.values()]
        
        return {
            'gain_statistics': {
                'mean_gain_db': float(np.mean(all_gains_db)),
                'std_gain_db': float(np.std(all_gains_db)),
                'min_gain_db': float(np.min(all_gains_db)),
                'max_gain_db': float(np.max(all_gains_db)),
                'median_gain_db': float(np.median(all_gains_db))
            },
            'distance_statistics': {
                'mean_distance_m': float(np.mean(all_distances)),
                'std_distance_m': float(np.std(all_distances)),
                'min_distance_m': float(np.min(all_distances)),
                'max_distance_m': float(np.max(all_distances))
            },
            'beam_usage_statistics': {
                'used_tx_beams': used_tx_beams,
                'total_tx_beams': total_tx_beams,
                'usage_ratio': float(used_tx_beams / total_tx_beams),
                'mean_rx_per_beam': float(np.mean(beam_usage_counts)),
                'std_rx_per_beam': float(np.std(beam_usage_counts)),
                'max_rx_per_beam': int(np.max(beam_usage_counts)),
                'min_rx_per_beam': int(np.min(beam_usage_counts))
            },
            'coverage_statistics': {
                'total_analyzed_rx': len(rx_beam_rankings),
                'total_expected_rx': self.config.TOTAL_RX_COUNT,
                'analysis_coverage_ratio': float(len(rx_beam_rankings) / self.config.TOTAL_RX_COUNT)
            }
        }

# ========================================================================
# BEAM CLUSTERING BASE ANALYZER - 공통 클러스터링 베이스 클래스
# ========================================================================

class BaseBeamClusteringAnalyzer:
    """빔 집합 유사도 기반 클러스터링 분석 베이스 클래스"""
    
    def __init__(self, config: P4_Config, beam_type: str):
        self.config = config
        self.beam_type = beam_type
        self.output_dir = config.OUTPUT_DIR
        
        # 빔 타입별 설정 로딩
        if beam_type == "TX":
            self.TOP_K_BEAMS = config.TX_TOP_K_BEAMS
            self.MIN_CLUSTER_SIZE = config.TX_MIN_CLUSTER_SIZE
            self.MIN_INTRA_SIMILARITY = config.TX_MIN_INTRA_SIMILARITY
        elif beam_type == "RX":
            self.TOP_K_BEAMS = config.RX_TOP_K_BEAMS
            self.MIN_CLUSTER_SIZE = config.RX_MIN_CLUSTER_SIZE
            self.MIN_INTRA_SIMILARITY = config.RX_MIN_INTRA_SIMILARITY
        else:
            raise ValueError(f"Unsupported beam_type: {beam_type}")
    
    def compute_jaccard_similarity(self, beam_set_data: List[Dict]) -> np.ndarray:
        """자카드 유사도 행렬 계산"""
        print(f"Computing {self.beam_type} beam Jaccard similarity matrix...")
        
        n = len(beam_set_data)
        similarity_matrix = np.zeros((n, n))
        
        beam_set_key = f'top_{self.beam_type.lower()}_beam_set'
        
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    set_i = beam_set_data[i][beam_set_key]
                    set_j = beam_set_data[j][beam_set_key]
                    
                    intersection = len(set_i.intersection(set_j))
                    union = len(set_i.union(set_j))
                    jaccard_sim = intersection / union if union > 0 else 0
                    
                    similarity_matrix[i, j] = jaccard_sim
                    similarity_matrix[j, i] = jaccard_sim
        
        print(f"{self.beam_type} similarity matrix computed: {n}x{n}")
        print(f"Average {self.beam_type} similarity: {np.mean(similarity_matrix[np.triu_indices(n, k=1)]):.3f}")
        
        return similarity_matrix
    
    def cluster_by_similarity(self, similarity_matrix: np.ndarray, 
                            method: str = 'average') -> Tuple[np.ndarray, List[Dict], np.ndarray]:
        """유사도 기준 계층적 클러스터링"""
        print(f"Performing {self.beam_type} beam clustering with {method} linkage...")
        
        distance_matrix = 1.0 - similarity_matrix
        linkage_matrix = linkage(squareform(distance_matrix), method=method)
        distance_threshold = 1.0 - self.MIN_INTRA_SIMILARITY
        cluster_labels = fcluster(linkage_matrix, distance_threshold, criterion='distance')
        
        initial_n_clusters = len(np.unique(cluster_labels))
        print(f"Initial {self.beam_type} clusters with similarity ≥{self.MIN_INTRA_SIMILARITY}: {initial_n_clusters}")
        
        # 클러스터 품질 검증
        clusters_info = []
        valid_cluster_id = 1
        
        for original_cluster_id in range(1, initial_n_clusters + 1):
            cluster_indices = np.where(cluster_labels == original_cluster_id)[0]
            
            if len(cluster_indices) >= self.MIN_CLUSTER_SIZE:
                if len(cluster_indices) > 1:
                    intra_similarities = [
                        similarity_matrix[i, j] for i in cluster_indices 
                        for j in cluster_indices if i < j
                    ]
                    avg_intra_sim = np.mean(intra_similarities)
                else:
                    avg_intra_sim = 1.0
                
                if avg_intra_sim >= self.MIN_INTRA_SIMILARITY:
                    clusters_info.append({
                        'cluster_id': valid_cluster_id,
                        'size': len(cluster_indices),
                        'member_indices': cluster_indices.tolist(),
                        'avg_intra_similarity': avg_intra_sim
                    })
                    valid_cluster_id += 1
        
        # 클러스터 라벨 재할당
        final_cluster_labels = np.zeros_like(cluster_labels)
        for i, cluster in enumerate(clusters_info):
            for member_idx in cluster['member_indices']:
                final_cluster_labels[member_idx] = cluster['cluster_id']
        
        print(f"Final valid {self.beam_type} clusters: {len(clusters_info)}")
        return linkage_matrix, clusters_info, final_cluster_labels
    
    def plot_spatial_clusters(self, beam_set_data: List[Dict], 
                            clusters_info: List[Dict], 
                            cluster_labels: np.ndarray,
                            spatial_analysis: Dict[str, Any]) -> str:
        """클러스터링 결과 공간적 분포 시각화"""
        print(f"Creating {self.beam_type} beam spatial cluster map...")
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        
        # 위치 데이터 준비
        positions = np.array([[data['position']['x'], data['position']['y']] 
                            for data in beam_set_data if data['position']])
        
        # 클러스터별 색상 설정
        n_clusters = len(clusters_info)
        if n_clusters <= 10:
            colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
        elif n_clusters <= 20:
            colors = plt.cm.tab20(np.linspace(0, 1, n_clusters))
        else:
            colors = plt.cm.hsv(np.linspace(0, 1, n_clusters))
        
        # 클러스터별 공간 분포 플롯
        for i, cluster in enumerate(clusters_info):
            cluster_id = cluster['cluster_id']
            member_indices = cluster['member_indices']
            cluster_size = cluster['size']
            
            cluster_positions = positions[member_indices]
            marker_size = max(30, min(100, cluster_size * 2))
            
            ax.scatter(cluster_positions[:, 0], cluster_positions[:, 1], 
                      c=[colors[i]], label=f'C{cluster_id} ({cluster_size})',
                      s=marker_size, alpha=0.7, edgecolors='white', linewidth=1)
        
        # 축 설정
        ax.set_xlabel('X Position (m)', fontsize=14)
        ax.set_ylabel('Y Position (m)', fontsize=14)
        ax.set_title(f'Spatial Distribution of {self.beam_type} Beam Clusters (Min Similarity ≥ {self.MIN_INTRA_SIMILARITY}, {n_clusters} Clusters)', 
                    fontsize=16, pad=20)
        
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # 범례 설정 (적응형)
        if n_clusters <= 15:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                     fontsize=10, frameon=True, fancybox=True, shadow=True)
        elif n_clusters <= 30:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                     ncol=2, fontsize=9, frameon=True)
        
        plt.tight_layout()
        
        # 저장
        prefix = "03" if self.beam_type == "TX" else "04"
        filename = f"{prefix}_{self.beam_type.lower()}_beam_clustering_spatial_map_sim{self.MIN_INTRA_SIMILARITY}_Area{self.config.TARGET_AREA}_{self.config.TARGET_FREQUENCY}GHz.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=self.config.SAVE_DPI, bbox_inches='tight')
        plt.close()
        
        print(f"{self.beam_type} spatial cluster map saved: {filepath}")
        return filepath

    def plot_similarity_heatmap(self, similarity_matrix: np.ndarray, 
                               cluster_labels: np.ndarray) -> str:
        """유사도 행렬 히트맵 시각화"""
        print(f"Creating {self.beam_type} beam similarity matrix heatmap...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 1. 원본 유사도 행렬
        im1 = ax1.imshow(similarity_matrix, cmap='viridis', aspect='auto')
        ax1.set_title(f'{self.beam_type} Beam Set Jaccard Similarity Matrix')
        ax1.set_xlabel('RX Index')
        ax1.set_ylabel('RX Index')
        plt.colorbar(im1, ax=ax1, label='Jaccard Similarity')
        
        # 2. 클러스터별로 정렬된 유사도 행렬
        sorted_indices = np.argsort(cluster_labels)
        sorted_similarity = similarity_matrix[np.ix_(sorted_indices, sorted_indices)]
        sorted_labels = cluster_labels[sorted_indices]
        
        im2 = ax2.imshow(sorted_similarity, cmap='viridis', aspect='auto')
        ax2.set_title('Similarity Matrix (Sorted by Clusters)')
        ax2.set_xlabel('RX Index (Sorted)')
        ax2.set_ylabel('RX Index (Sorted)')
        plt.colorbar(im2, ax=ax2, label='Jaccard Similarity')
        
        # 클러스터 경계선 추가
        unique_labels = np.unique(sorted_labels)
        boundaries = []
        for label in unique_labels[:-1]:
            boundary = np.where(sorted_labels == label)[0][-1] + 0.5
            boundaries.append(boundary)
        
        for boundary in boundaries:
            ax2.axhline(y=boundary, color='red', linestyle='--', alpha=0.7)
            ax2.axvline(x=boundary, color='red', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        # 저장
        prefix = "03" if self.beam_type == "TX" else "04"
        filename = f"{prefix}_{self.beam_type.lower()}_beam_similarity_heatmap_sim{self.MIN_INTRA_SIMILARITY}_Area{self.config.TARGET_AREA}_{self.config.TARGET_FREQUENCY}GHz.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=self.config.SAVE_DPI, bbox_inches='tight')
        plt.close()
        
        print(f"{self.beam_type} similarity heatmap saved: {filepath}")
        return filepath

    def plot_dendrogram(self, linkage_matrix: np.ndarray) -> str:
        """계층적 클러스터링 덴드로그램 시각화"""
        print(f"Creating {self.beam_type} beam dendrogram...")
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        dendrogram(linkage_matrix, ax=ax, truncate_mode='lastp', p=30, 
                  leaf_rotation=90, leaf_font_size=8)
        
        ax.set_title(f'Hierarchical Clustering Dendrogram ({self.beam_type} Beam Set Jaccard Similarity)')
        ax.set_xlabel('Sample Index or (Cluster Size)')
        ax.set_ylabel('Distance (1 - Jaccard Similarity)')
        
        plt.tight_layout()
        
        # 저장
        prefix = "03" if self.beam_type == "TX" else "04"
        filename = f"{prefix}_{self.beam_type.lower()}_beam_clustering_dendrogram_sim{self.MIN_INTRA_SIMILARITY}_Area{self.config.TARGET_AREA}_{self.config.TARGET_FREQUENCY}GHz.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=self.config.SAVE_DPI, bbox_inches='tight')
        plt.close()
        
        print(f"{self.beam_type} dendrogram saved: {filepath}")
        return filepath
    
    def compute_cluster_locations(self, beam_set_data: List[Dict], 
                                clusters_info: List[Dict], 
                                cluster_labels: np.ndarray) -> Dict[str, Any]:
        """클러스터 공간 분포 분석"""
        print("Analyzing spatial cluster distribution...")
        
        spatial_analysis = {}
        
        for cluster in clusters_info:
            cluster_id = cluster['cluster_id']
            member_indices = cluster['member_indices']
            
            positions = []
            distances = []
            
            for idx in member_indices:
                pos = beam_set_data[idx]['position']
                if pos:
                    positions.append([pos['x'], pos['y']])
                distances.append(beam_set_data[idx]['distance_2d'])
            
            if positions:
                positions = np.array(positions)
                distances = np.array(distances)
                
                centroid = np.mean(positions, axis=0)
                spatial_spread = np.std(positions, axis=0)
                max_distance_from_centroid = np.max([
                    np.linalg.norm(pos - centroid) for pos in positions
                ]) if len(positions) > 1 else 0
                
                distance_stats = {
                    'mean': float(np.mean(distances)),
                    'std': float(np.std(distances)),
                    'min': float(np.min(distances)),
                    'max': float(np.max(distances))
                }
                
                spatial_analysis[cluster_id] = {
                    'centroid': centroid.tolist(),
                    'spatial_spread': spatial_spread.tolist(),
                    'max_spread': float(max_distance_from_centroid),
                    'distance_stats': distance_stats,
                    'member_count': len(member_indices)
                }
        
        return spatial_analysis

    def compute_beam_statistics(self, beam_set_data: List[Dict], 
                              clusters_info: List[Dict]) -> Dict[str, Any]:
        """클러스터별 빔 통계"""
        print("Computing cluster beam statistics...")
        
        beam_stats = {}
        beam_key = f'top_{self.beam_type.lower()}_beams'
        
        for cluster in clusters_info:
            cluster_id = cluster['cluster_id']
            member_indices = cluster['member_indices']
            
            all_beams = []
            beam_frequencies = Counter()
            
            for idx in member_indices:
                beams = beam_set_data[idx][beam_key]
                all_beams.extend(beams)
                beam_frequencies.update(beams)
            
            most_common_beams = beam_frequencies.most_common(5)
            total_beams = len(all_beams)
            entropy = -sum((count/total_beams) * np.log2(count/total_beams) 
                         for count in beam_frequencies.values())
            
            beam_stats[cluster_id] = {
                'unique_beams': len(beam_frequencies),
                'most_common_beams': most_common_beams,
                'beam_diversity_entropy': float(entropy),
                'total_beam_uses': total_beams,
                'average_beams_per_rx': total_beams / len(member_indices)
            }
        
        return beam_stats

    def evaluate_cluster_quality(self, similarity_matrix: np.ndarray, 
                                clusters_info: List[Dict], 
                                cluster_labels: np.ndarray) -> Dict[str, Any]:
        """클러스터 품질 평가"""
        print("Evaluating cluster quality...")
        
        intra_similarities = []
        inter_similarities = []
        
        for i, cluster_i in enumerate(clusters_info):
            indices_i = cluster_i['member_indices']
            
            if len(indices_i) > 1:
                intra_sim = np.mean([similarity_matrix[i, j] 
                                   for i in indices_i for j in indices_i if i < j])
                intra_similarities.append(intra_sim)
            
            for j, cluster_j in enumerate(clusters_info):
                if i < j:
                    indices_j = cluster_j['member_indices']
                    inter_sim = np.mean([similarity_matrix[i, j] 
                                       for i in indices_i for j in indices_j])
                    inter_similarities.append(inter_sim)
        
        avg_intra = np.mean(intra_similarities) if intra_similarities else 0
        avg_inter = np.mean(inter_similarities) if inter_similarities else 0
        separation_ratio = avg_intra / avg_inter if avg_inter > 0 else float('inf')
        
        return {
            'avg_intra_cluster_similarity': float(avg_intra),
            'avg_inter_cluster_similarity': float(avg_inter),
            'separation_ratio': float(separation_ratio),
            'num_clusters': len(clusters_info),
            'cluster_sizes': [c['size'] for c in clusters_info]
        }

# ========================================================================
# TX BEAM CLUSTERING ANALYZER - 새 방식으로 교체
# ========================================================================

class TXBeamClusteringAnalyzer(BaseBeamClusteringAnalyzer):
    """TX 빔 집합 유사도 기반 클러스터링 분석"""
    
    def __init__(self, config: P4_Config):
        super().__init__(config, "TX")
    
    def run_clustering_analysis(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """TX 빔 클러스터링 전체 분석 실행"""
        
        print("="*60)
        print("TX BEAM CLUSTERING ANALYSIS")
        print("="*60)
        
        results = {}
        
        # 1. 데이터 전처리
        rx_beam_rankings = analysis_results['rx_beam_rankings']
        beam_set_data = self.extract_unique_tx_beams(rx_beam_rankings)
        
        if len(beam_set_data) < self.MIN_CLUSTER_SIZE:
            print("Error: Insufficient data for clustering analysis")
            return {'error': 'Insufficient data'}
        
        results['num_rx_analyzed'] = len(beam_set_data)
        results['beam_set_data'] = beam_set_data
        results['original_dataframe'] = self.create_dataframe_from_rankings(rx_beam_rankings)
        
        # 2. 유사도 행렬 계산
        similarity_matrix = self.compute_jaccard_similarity(beam_set_data)
        results['similarity_matrix'] = similarity_matrix
        
        # 3. 유사도 기준 클러스터링
        linkage_matrix, clusters_info, cluster_labels = self.cluster_by_similarity(similarity_matrix)
        
        results['linkage_matrix'] = linkage_matrix
        results['clusters_info'] = clusters_info
        results['cluster_labels'] = cluster_labels
        
        # 4. 공간적 분석
        spatial_analysis = self.compute_cluster_locations(beam_set_data, clusters_info, cluster_labels)
        results['spatial_analysis'] = spatial_analysis
        
        # 5. 빔 통계 분석
        beam_stats = self.compute_beam_statistics(beam_set_data, clusters_info)
        results['beam_statistics'] = beam_stats
        
        # 6. 클러스터 품질 평가
        quality_metrics = self.evaluate_cluster_quality(similarity_matrix, clusters_info, cluster_labels)
        results['quality_metrics'] = quality_metrics
        
        # 7. 시각화 생성
        print("\nGenerating visualizations...")
        visualizations = {}
        
        try:
            visualizations['spatial_map'] = self.plot_spatial_clusters(
                beam_set_data, clusters_info, cluster_labels, spatial_analysis)
            
            visualizations['similarity_heatmap'] = self.plot_similarity_heatmap(
                similarity_matrix, cluster_labels)
            
            visualizations['dendrogram'] = self.plot_dendrogram(linkage_matrix)
            
            results['visualizations'] = visualizations
            
        except Exception as e:
            print(f"Warning: Error generating visualizations: {e}")
            results['visualizations'] = {}
        
        # 8. 결과 저장 (CSV 우선)
        print("\nSaving results...")
        self.save_results_csv(results)
        self.save_cluster_summary_csv(results)
        
        # 9. 분석 요약 출력
        self.print_analysis_summary(results)
        
        print("\nTX Beam Clustering Analysis completed successfully!")
        return results
    
    def extract_unique_tx_beams(self, rx_beam_rankings: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """각 RX의 상위 8개 고유 TX 빔 집합 추출 (개선된 로직)"""
        
        print(f"Extracting top {self.TOP_K_BEAMS} unique TX beam sets...")
        
        beam_set_data = []
        
        for rx_idx, rx_analysis in rx_beam_rankings.items():
            position = rx_analysis['position']
            
            # 32순위에서 고유한 상위 K개 TX 빔 추출 (순서 유지)
            unique_tx_beams = []
            corresponding_gains = []
            seen_tx_beams = set()
            
            # 1~32순위까지 확인하면서 고유한 TX 빔 K개를 찾을 때까지 진행
            for beam_pair in rx_analysis['top_beam_pairs']:
                tx_beam = beam_pair['tx_beam_idx']
                gain_db = beam_pair['gain_db']
                
                # 아직 보지 못한 TX 빔이면 추가
                if tx_beam not in seen_tx_beams:
                    unique_tx_beams.append(tx_beam)
                    corresponding_gains.append(gain_db)
                    seen_tx_beams.add(tx_beam)
                    
                    # K개 고유 빔을 찾으면 중단
                    if len(unique_tx_beams) >= self.TOP_K_BEAMS:
                        break
            
            if len(unique_tx_beams) >= self.MIN_CLUSTER_SIZE:
                beam_set_data.append({
                    'rx_idx': rx_idx,
                    'position': position,
                    'top_tx_beams': unique_tx_beams,
                    'top_tx_beam_set': set(unique_tx_beams),
                    'top_gains': corresponding_gains,
                    'distance_2d': position.get('distance_2d_m', 0) if position else 0
                })
        
        print(f"Extracted {len(beam_set_data)} valid RX beam sets")
        return beam_set_data
    
    def create_dataframe_from_rankings(self, rx_beam_rankings: Dict[int, Dict[str, Any]]) -> pd.DataFrame:
        """빔 순위 데이터를 DataFrame으로 변환 (CSV 저장용)"""
        
        data_rows = []
        
        for rx_idx, rx_analysis in rx_beam_rankings.items():
            row = {
                'RX_Index': rx_idx,
                'X_Position': rx_analysis['position'].get('x', 0),
                'Y_Position': rx_analysis['position'].get('y', 0),
                'Z_Position': rx_analysis['position'].get('z', 0),
                'Distance_2D_m': rx_analysis['position'].get('distance_2d_m', 0),
                'Best_TX_Beam': rx_analysis['best_tx_beam'],
                'Best_RX_Beam': rx_analysis['best_rx_beam'],
                'Best_Gain_dB': rx_analysis['best_gain_db']
            }
            
            # 상위 빔 페어 정보 추가
            for i, beam_pair in enumerate(rx_analysis['top_beam_pairs'][:self.config.TOP_BEAM_RANKS]):
                row[f'Rank{i+1}_TX_Beam'] = beam_pair['tx_beam_idx']
                row[f'Rank{i+1}_RX_Beam'] = beam_pair['rx_beam_idx']
                row[f'Rank{i+1}_Gain_dB'] = beam_pair['gain_db']
            
            data_rows.append(row)
        
        return pd.DataFrame(data_rows)
    
    

    def compute_cluster_locations(self, beam_set_data: List[Dict], 
                                clusters_info: List[Dict], 
                                cluster_labels: np.ndarray) -> Dict[str, Any]:
        """클러스터의 공간적 분포 분석"""
        print("Analyzing spatial cluster distribution...")
        
        spatial_analysis = {}
        
        for cluster in clusters_info:
            cluster_id = cluster['cluster_id']
            member_indices = cluster['member_indices']
            
            # 클러스터 멤버들의 위치 정보
            positions = []
            distances = []
            
            for idx in member_indices:
                pos = beam_set_data[idx]['position']
                if pos:
                    positions.append([pos['x'], pos['y']])
                distances.append(beam_set_data[idx]['distance_2d'])
            
            if positions:
                positions = np.array(positions)
                distances = np.array(distances)
                
                # 공간적 통계 계산
                centroid = np.mean(positions, axis=0)
                spatial_spread = np.std(positions, axis=0)
                max_distance_from_centroid = np.max([
                    np.linalg.norm(pos - centroid) for pos in positions
                ]) if len(positions) > 1 else 0
                
                # 거리 분포 통계
                distance_stats = {
                    'mean': float(np.mean(distances)),
                    'std': float(np.std(distances)),
                    'min': float(np.min(distances)),
                    'max': float(np.max(distances))
                }
                
                spatial_analysis[cluster_id] = {
                    'centroid': centroid.tolist(),
                    'spatial_spread': spatial_spread.tolist(),
                    'max_spread': float(max_distance_from_centroid),
                    'distance_stats': distance_stats,
                    'member_count': len(member_indices)
                }
        
        return spatial_analysis

    def compute_beam_statistics(self, beam_set_data: List[Dict], 
                              clusters_info: List[Dict]) -> Dict[str, Any]:
        """클러스터별 빔 사용 통계 분석"""
        print("Computing cluster beam statistics...")
        
        beam_stats = {}
        
        for cluster in clusters_info:
            cluster_id = cluster['cluster_id']
            member_indices = cluster['member_indices']
            
            # 모든 TX 빔 수집
            all_tx_beams = []
            beam_frequencies = Counter()
            
            for idx in member_indices:
                tx_beams = beam_set_data[idx]['top_tx_beams']
                all_tx_beams.extend(tx_beams)
                beam_frequencies.update(tx_beams)
            
            # 가장 흔한 빔들 (클러스터 대표 빔)
            most_common_beams = beam_frequencies.most_common(5)
            
            # 빔 다양성 (Shannon entropy)
            total_beams = len(all_tx_beams)
            entropy = -sum((count/total_beams) * np.log2(count/total_beams) 
                         for count in beam_frequencies.values())
            
            beam_stats[cluster_id] = {
                'unique_beams': len(beam_frequencies),
                'most_common_beams': most_common_beams,
                'beam_diversity_entropy': float(entropy),
                'total_beam_uses': total_beams,
                'average_beams_per_rx': total_beams / len(member_indices)
            }
        
        return beam_stats

    def evaluate_cluster_quality(self, similarity_matrix: np.ndarray, 
                                clusters_info: List[Dict], 
                                cluster_labels: np.ndarray) -> Dict[str, Any]:
        """클러스터 품질 평가"""
        print("Evaluating cluster quality...")
        
        # 클러스터 간/내 유사도 분석
        intra_similarities = []
        inter_similarities = []
        
        for i, cluster_i in enumerate(clusters_info):
            indices_i = cluster_i['member_indices']
            
            # 클러스터 내 유사도
            if len(indices_i) > 1:
                intra_sim = np.mean([similarity_matrix[i, j] 
                                   for i in indices_i for j in indices_i if i < j])
                intra_similarities.append(intra_sim)
            
            # 다른 클러스터들과의 유사도
            for j, cluster_j in enumerate(clusters_info):
                if i < j:
                    indices_j = cluster_j['member_indices']
                    inter_sim = np.mean([similarity_matrix[i, j] 
                                       for i in indices_i for j in indices_j])
                    inter_similarities.append(inter_sim)
        
        # 분리도 (Separation ratio)
        avg_intra = np.mean(intra_similarities) if intra_similarities else 0
        avg_inter = np.mean(inter_similarities) if inter_similarities else 0
        separation_ratio = avg_intra / avg_inter if avg_inter > 0 else float('inf')
        
        quality_metrics = {
            'avg_intra_cluster_similarity': float(avg_intra),
            'avg_inter_cluster_similarity': float(avg_inter),
            'separation_ratio': float(separation_ratio),
            'num_clusters': len(clusters_info),
            'cluster_sizes': [c['size'] for c in clusters_info]
        }
        
        return quality_metrics



    def save_results_csv(self, results: Dict[str, Any]) -> str:
        """클러스터링 결과를 CSV로 저장 (우선순위)"""
        print("Exporting clustering results to CSV...")
        
        beam_set_data = results['beam_set_data']
        cluster_labels = results['cluster_labels']
        original_df = results['original_dataframe']
        
        # RX 인덱스와 클러스터 매핑
        rx_to_cluster = {}
        for i, data in enumerate(beam_set_data):
            rx_idx = data['rx_idx']
            cluster_id = cluster_labels[i]
            rx_to_cluster[rx_idx] = cluster_id
        
        # CSV 데이터 준비
        csv_data = []
        
        for _, row in original_df.iterrows():
            rx_idx = int(row['RX_Index'])
            
            if rx_idx not in rx_to_cluster:
                continue  # 클러스터링되지 않은 RX는 제외
            
            cluster_id = rx_to_cluster[rx_idx]
            
            # 실제 클러스터링에 사용된 고유한 TX 빔 집합 찾기
            beam_data = next(data for data in beam_set_data if data['rx_idx'] == rx_idx)
            unique_tx_beams = sorted(list(beam_data['top_tx_beam_set']))
            unique_tx_beams_str = ','.join(map(str, unique_tx_beams))
            
            # 기본 정보 (소수점 2자리로 제한)
            csv_row = {
                'Cluster_ID': int(cluster_id),
                'RX_Index': rx_idx,
                'X_Position': round(float(row['X_Position']), 2),
                'Y_Position': round(float(row['Y_Position']), 2),
                'Distance_2D_m': round(float(row['Distance_2D_m']), 2),
                'Unique_TX_Beams_Used_for_Clustering': unique_tx_beams_str,
                'Num_Unique_TX_Beams': len(unique_tx_beams)
            }
            
            # 상위 8개 빔 정보 추가
            for rank in range(1, self.TOP_K_BEAMS + 1):
                tx_beam_col = f'Rank{rank}_TX_Beam'
                rx_beam_col = f'Rank{rank}_RX_Beam'
                gain_col = f'Rank{rank}_Gain_dB'
                
                if tx_beam_col in row and pd.notna(row[tx_beam_col]):
                    csv_row[f'TX_Beam_{rank}'] = int(row[tx_beam_col])
                    csv_row[f'RX_Beam_{rank}'] = int(row[rx_beam_col])
                    csv_row[f'Gain_dB_{rank}'] = round(float(row[gain_col]), 2)
                else:
                    csv_row[f'TX_Beam_{rank}'] = None
                    csv_row[f'RX_Beam_{rank}'] = None
                    csv_row[f'Gain_dB_{rank}'] = None
            
            csv_data.append(csv_row)
        
        # DataFrame 생성 및 클러스터별 정렬
        df_result = pd.DataFrame(csv_data)
        df_result = df_result.sort_values(['Cluster_ID', 'RX_Index'])
        
        # CSV 저장
        filename = f"03_tx_beam_clustering_detailed_results_sim{self.MIN_INTRA_SIMILARITY}_Area{self.config.TARGET_AREA}_{self.config.TARGET_FREQUENCY}GHz.csv"
        filepath = os.path.join(self.output_dir, filename)
        df_result.to_csv(filepath, index=False)
        
        print(f"Detailed clustering results saved to CSV: {filepath}")
        print(f"CSV contains {len(df_result)} RXs across {len(df_result['Cluster_ID'].unique())} clusters")
        
        return filepath

    def save_cluster_summary_csv(self, results: Dict[str, Any]) -> str:
        """클러스터 요약 정보를 CSV로 저장"""
        print("Exporting cluster summary to CSV...")
        
        clusters_info = results['clusters_info']
        spatial_analysis = results['spatial_analysis']
        beam_statistics = results['beam_statistics']
        
        summary_data = []
        
        for cluster in clusters_info:
            cluster_id = cluster['cluster_id']
            spatial = spatial_analysis[cluster_id]
            beam_stats = beam_statistics[cluster_id]
            
            # 가장 흔한 상위 5개 TX 빔
            top_beams = beam_stats['most_common_beams'][:5]
            top_beams_str = "; ".join([f"TX{beam}({count})" for beam, count in top_beams])
            
            summary_row = {
                'Cluster_ID': cluster_id,
                'Size': cluster['size'],
                'Avg_Intra_Similarity': round(cluster['avg_intra_similarity'], 2),
                'Centroid_X': round(spatial['centroid'][0], 2),
                'Centroid_Y': round(spatial['centroid'][1], 2),
                'Spatial_Spread_m': round(spatial['max_spread'], 2),
                'Avg_Distance_2D_m': round(spatial['distance_stats']['mean'], 2),
                'Distance_Std_m': round(spatial['distance_stats']['std'], 2),
                'Unique_TX_Beams': beam_stats['unique_beams'],
                'Beam_Diversity_Entropy': round(beam_stats['beam_diversity_entropy'], 2),
                'Top_TX_Beams': top_beams_str
            }
            
            summary_data.append(summary_row)
        
        # DataFrame 생성 및 정렬
        df_summary = pd.DataFrame(summary_data)
        df_summary = df_summary.sort_values('Cluster_ID')
        
        # CSV 저장
        filename = f"03_tx_beam_clustering_summary_sim{self.MIN_INTRA_SIMILARITY}_Area{self.config.TARGET_AREA}_{self.config.TARGET_FREQUENCY}GHz.csv"
        filepath = os.path.join(self.output_dir, filename)
        df_summary.to_csv(filepath, index=False)
        
        print(f"Cluster summary saved to CSV: {filepath}")
        return filepath

    def print_analysis_summary(self, results: Dict[str, Any]) -> None:
        """분석 결과 요약 출력"""
        print("\n" + "="*60)
        print("ANALYSIS SUMMARY")
        print("="*60)
        
        print(f"Number of RXs analyzed: {results['num_rx_analyzed']}")
        print(f"Number of clusters found: {results['quality_metrics']['num_clusters']}")
        print(f"Average intra-cluster similarity: {results['quality_metrics']['avg_intra_cluster_similarity']:.3f}")
        print(f"Average inter-cluster similarity: {results['quality_metrics']['avg_inter_cluster_similarity']:.3f}")
        print(f"Separation ratio: {results['quality_metrics']['separation_ratio']:.2f}")
        
        print("\nCluster Details:")
        print("-" * 40)
        for cluster in results['clusters_info']:
            cluster_id = cluster['cluster_id']
            size = cluster['size']
            avg_sim = cluster['avg_intra_similarity']
            spatial = results['spatial_analysis'][cluster_id]
            beam_stats = results['beam_statistics'][cluster_id]
            
            print(f"Cluster {cluster_id}: {size} RXs")
            print(f"  - Avg intra-similarity: {avg_sim:.3f}")
            print(f"  - Spatial spread: {spatial['max_spread']:.1f}m")
            print(f"  - Unique TX beams: {beam_stats['unique_beams']}")
            print(f"  - Beam diversity: {beam_stats['beam_diversity_entropy']:.2f}")
            
            # 가장 흔한 빔들 표시
            top_beams = beam_stats['most_common_beams'][:3]
            beam_str = ", ".join([f"TX{beam}({count})" for beam, count in top_beams])
            print(f"  - Top beams: {beam_str}")
            print()

# ========================================================================
# RX BEAM CLUSTERING ANALYZER - RX 빔 클러스터링 분석
# ========================================================================

class RXBeamClusteringAnalyzer(BaseBeamClusteringAnalyzer):
    """RX 빔 집합 유사도 기반 클러스터링 분석"""
    
    def __init__(self, config: P4_Config):
        super().__init__(config, "RX")
    
    def run_clustering_analysis(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """RX 빔 클러스터링 전체 분석 실행"""
        
        print("="*60)
        print("RX BEAM CLUSTERING ANALYSIS")
        print("="*60)
        
        results = {}
        
        # 1. 데이터 전처리
        rx_beam_rankings = analysis_results['rx_beam_rankings']
        beam_set_data = self.extract_unique_rx_beams(rx_beam_rankings)
        
        if len(beam_set_data) < self.MIN_CLUSTER_SIZE:
            print("Error: Insufficient data for RX clustering analysis")
            return {'error': 'Insufficient data'}
        
        results['num_rx_analyzed'] = len(beam_set_data)
        results['beam_set_data'] = beam_set_data
        
        # 2. 유사도 행렬 계산
        similarity_matrix = self.compute_jaccard_similarity(beam_set_data)
        results['similarity_matrix'] = similarity_matrix
        
        # 3. 유사도 기준 클러스터링
        linkage_matrix, clusters_info, cluster_labels = self.cluster_by_similarity(similarity_matrix)
        
        results['linkage_matrix'] = linkage_matrix
        results['clusters_info'] = clusters_info
        results['cluster_labels'] = cluster_labels
        
        # 4. 공간적 분석
        spatial_analysis = self.compute_cluster_locations(beam_set_data, clusters_info, cluster_labels)
        results['spatial_analysis'] = spatial_analysis
        
        # 5. 빔 통계 분석
        beam_stats = self.compute_beam_statistics(beam_set_data, clusters_info)
        results['beam_statistics'] = beam_stats
        
        # 6. 클러스터 품질 평가
        quality_metrics = self.evaluate_cluster_quality(similarity_matrix, clusters_info, cluster_labels)
        results['quality_metrics'] = quality_metrics
        
        # 7. 시각화 생성
        print("\nGenerating RX clustering visualizations...")
        visualizations = {}
        
        try:
            visualizations['spatial_map'] = self.plot_spatial_clusters(
                beam_set_data, clusters_info, cluster_labels, spatial_analysis)
            
            visualizations['similarity_heatmap'] = self.plot_similarity_heatmap(
                similarity_matrix, cluster_labels)
            
            visualizations['dendrogram'] = self.plot_dendrogram(linkage_matrix)
            
            results['visualizations'] = visualizations
            
        except Exception as e:
            print(f"Warning: Error generating RX visualizations: {e}")
            results['visualizations'] = {}
        
        # 8. 결과 저장 (CSV 우선)
        print("\nSaving RX clustering results...")
        self.save_results_csv(results)
        self.save_cluster_summary_csv(results)
        
        # 9. 분석 요약 출력
        self.print_analysis_summary(results)
        
        print("\nRX Beam Clustering Analysis completed successfully!")
        return results
    
    def extract_unique_rx_beams(self, rx_beam_rankings: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """각 RX의 상위 K개 고유 RX 빔 집합 추출"""
        
        print(f"Extracting top {self.TOP_K_BEAMS} unique RX beam sets...")
        
        beam_set_data = []
        
        for rx_idx, rx_analysis in rx_beam_rankings.items():
            position = rx_analysis['position']
            
            # 순위에서 고유한 상위 K개 RX 빔 추출
            unique_rx_beams = []
            corresponding_gains = []
            seen_rx_beams = set()
            
            for beam_pair in rx_analysis['top_beam_pairs']:
                rx_beam = beam_pair['rx_beam_idx']
                gain_db = beam_pair['gain_db']
                
                if rx_beam not in seen_rx_beams:
                    unique_rx_beams.append(rx_beam)
                    corresponding_gains.append(gain_db)
                    seen_rx_beams.add(rx_beam)
                    
                    if len(unique_rx_beams) >= self.TOP_K_BEAMS:
                        break
            
            if len(unique_rx_beams) >= self.MIN_CLUSTER_SIZE:
                beam_set_data.append({
                    'rx_idx': rx_idx,
                    'position': position,
                    'top_rx_beams': unique_rx_beams,
                    'top_rx_beam_set': set(unique_rx_beams),
                    'top_gains': corresponding_gains,
                    'distance_2d': position.get('distance_2d_m', 0) if position else 0
                })
        
        print(f"Extracted {len(beam_set_data)} valid RX beam sets")
        return beam_set_data
    
    



    def save_results_csv(self, results: Dict[str, Any]) -> str:
        """RX 클러스터링 결과를 CSV로 저장"""
        
        beam_set_data = results['beam_set_data']
        cluster_labels = results['cluster_labels']
        
        csv_data = []
        for i, data in enumerate(beam_set_data):
            cluster_id = cluster_labels[i]
            unique_rx_beams = sorted(list(data['top_rx_beam_set']))
            
            csv_row = {
                'Cluster_ID': int(cluster_id),
                'RX_Index': data['rx_idx'],
                'X_Position': round(data['position']['x'], 2) if data['position'] else 0,
                'Y_Position': round(data['position']['y'], 2) if data['position'] else 0,
                'Distance_2D_m': round(data['distance_2d'], 2),
                'Unique_RX_Beams': ','.join(map(str, unique_rx_beams)),
                'Num_Unique_RX_Beams': len(unique_rx_beams)
            }
            csv_data.append(csv_row)
        
        df_result = pd.DataFrame(csv_data)
        df_result = df_result.sort_values(['Cluster_ID', 'RX_Index'])
        
        filename = f"04_rx_beam_clustering_results_sim{self.MIN_INTRA_SIMILARITY}_Area{self.config.TARGET_AREA}_{self.config.TARGET_FREQUENCY}GHz.csv"
        filepath = os.path.join(self.output_dir, filename)
        df_result.to_csv(filepath, index=False)
        
        print(f"RX clustering results saved: {filepath}")
        return filepath

    def save_cluster_summary_csv(self, results: Dict[str, Any]) -> str:
        """RX 클러스터 요약 CSV 저장"""
        
        clusters_info = results['clusters_info']
        spatial_analysis = results['spatial_analysis']
        beam_statistics = results['beam_statistics']
        
        summary_data = []
        for cluster in clusters_info:
            cluster_id = cluster['cluster_id']
            spatial = spatial_analysis[cluster_id]
            beam_stats = beam_statistics[cluster_id]
            
            top_beams = beam_stats['most_common_beams'][:5]
            top_beams_str = "; ".join([f"RX{beam}({count})" for beam, count in top_beams])
            
            summary_data.append({
                'Cluster_ID': cluster_id,
                'Size': cluster['size'],
                'Avg_Intra_Similarity': round(cluster['avg_intra_similarity'], 2),
                'Spatial_Spread_m': round(spatial['max_spread'], 2),
                'Unique_RX_Beams': beam_stats['unique_beams'],
                'Top_RX_Beams': top_beams_str
            })
        
        df_summary = pd.DataFrame(summary_data)
        
        filename = f"04_rx_beam_clustering_summary_sim{self.MIN_INTRA_SIMILARITY}_Area{self.config.TARGET_AREA}_{self.config.TARGET_FREQUENCY}GHz.csv"
        filepath = os.path.join(self.output_dir, filename)
        df_summary.to_csv(filepath, index=False)
        
        print(f"RX cluster summary saved: {filepath}")
        return filepath

    def print_analysis_summary(self, results: Dict[str, Any]) -> None:
        """RX 클러스터링 분석 요약 출력"""
        print("\n" + "="*60)
        print("RX BEAM CLUSTERING ANALYSIS SUMMARY")
        print("="*60)
        
        print(f"Number of RXs analyzed: {results['num_rx_analyzed']}")
        print(f"Number of clusters found: {results['quality_metrics']['num_clusters']}")
        print(f"Average intra-cluster similarity: {results['quality_metrics']['avg_intra_cluster_similarity']:.3f}")
        print(f"Separation ratio: {results['quality_metrics']['separation_ratio']:.2f}")

# ========================================================================
# BASIC VISUALIZATION - 간소화된 시각화
# ========================================================================

class BasicVisualizer:
    """기본 시각화 기능 (간소화)"""
    
    def __init__(self, config: P4_Config):
        self.config = config
        
    def create_basic_visualizations(self, analysis_results: Dict[str, Any]) -> Dict[str, str]:
        """기본 분석 결과 시각화 생성"""
        
        print("Creating basic visualizations...")
        
        saved_plots = {}
        
        # 1. 송신 빔별 단말 수 분포
        saved_plots['tx_beam_distribution'] = self.plot_tx_beam_distribution(analysis_results)
        
        # 2. 단말 위치별 최상위 송신 빔 맵
        saved_plots['spatial_tx_beam_map'] = self.plot_spatial_tx_beam_map(analysis_results)
        
        # 3. 단말 위치별 최상위 수신 빔 맵
        saved_plots['spatial_rx_beam_map'] = self.plot_spatial_rx_beam_map(analysis_results)
        
        # 4. 빔 이득 CDF 분포
        saved_plots['gain_cdf'] = self.plot_gain_cdf(analysis_results)
        
        print(f"Created {len(saved_plots)} basic visualization plots")
        
        return saved_plots
    
    def plot_tx_beam_distribution(self, analysis_results: Dict[str, Any]) -> str:
        """송신 빔별 단말 수 분포 히스토그램"""
        
        tx_beam_dominance = analysis_results['tx_beam_dominance']
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 송신 빔 인덱스와 단말 수
        tx_beam_indices = list(tx_beam_dominance.keys())
        rx_counts = [info['rx_count'] for info in tx_beam_dominance.values()]
        
        # 송신 빔별 단말 수 막대 그래프
        ax.bar(tx_beam_indices, rx_counts, alpha=0.7, color='skyblue', edgecolor='navy')
        ax.set_xlabel('TX Beam Index')
        ax.set_ylabel('Number of Dominant RX')
        ax.set_title('TX Beam Dominance Distribution')
        ax.grid(True, alpha=0.3)
        
        # 상위 5개 빔 강조
        top_5_indices = sorted(tx_beam_dominance.keys(), 
                              key=lambda x: tx_beam_dominance[x]['rx_count'], reverse=True)[:5]
        for i, beam_idx in enumerate(top_5_indices):
            if beam_idx in tx_beam_indices:
                bar_idx = tx_beam_indices.index(beam_idx)
                ax.bar(beam_idx, rx_counts[bar_idx], color='red', alpha=0.8, 
                       label=f'Top {i+1}' if i == 0 else "")
        
        if top_5_indices:
            ax.legend()
        
        plt.tight_layout()
        
        # 저장
        filename = f"02_basic_tx_distribution_Area{self.config.TARGET_AREA}_{self.config.TARGET_FREQUENCY}GHz.png"
        filepath = os.path.join(self.config.OUTPUT_DIR, filename)
        plt.savefig(filepath, dpi=self.config.SAVE_DPI, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def plot_spatial_tx_beam_map(self, analysis_results: Dict[str, Any]) -> str:
        """단말 위치별 최상위 송신 빔 공간 분포 맵"""
        
        rx_beam_rankings = analysis_results['rx_beam_rankings']
        
        # 위치와 빔 데이터 추출
        positions = []
        tx_beams = []
        gains_db = []
        
        for rx_analysis in rx_beam_rankings.values():
            pos = rx_analysis['position']
            if pos:
                positions.append([pos['x'], pos['y']])
                tx_beams.append(rx_analysis['best_tx_beam'])
                gains_db.append(rx_analysis['best_gain_db'])
        
        if not positions:
            print("Warning: No position data available for spatial map")
            return ""
        
        positions = np.array(positions)
        tx_beams = np.array(tx_beams)
        gains_db = np.array(gains_db)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # 1. 송신 빔 인덱스 분포 맵
        scatter1 = ax1.scatter(positions[:, 0], positions[:, 1], c=tx_beams, 
                            cmap='hsv', s=30, alpha=0.8)
        ax1.set_xlabel('X Position (m)')
        ax1.set_ylabel('Y Position (m)')
        ax1.set_title('Spatial Distribution of Best TX Beam')
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        # 송신기 위치 표시
        tx_pos = self.config.AREA1_TX_POSITION
        ax1.scatter(tx_pos[0], tx_pos[1], c='red', s=200, marker='^', 
                   label='TX Position', edgecolors='black', linewidth=2)
        ax1.legend()
        
        # 컬러바 추가
        cbar1 = plt.colorbar(scatter1, ax=ax1)
        cbar1.set_label('TX Beam Index')
        
        # 2. 빔 이득 분포 맵
        scatter2 = ax2.scatter(positions[:, 0], positions[:, 1], c=gains_db, 
                              cmap='RdYlBu_r', s=30, alpha=0.8)
        ax2.set_xlabel('X Position (m)')
        ax2.set_ylabel('Y Position (m)')
        ax2.set_title('Spatial Distribution of Best Beam Gain')
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal')
        
        # 송신기 위치 표시
        ax2.scatter(tx_pos[0], tx_pos[1], c='black', s=200, marker='^', 
                   label='TX Position', edgecolors='white', linewidth=2)
        ax2.legend()
        
        # 컬러바 추가
        cbar2 = plt.colorbar(scatter2, ax=ax2)
        cbar2.set_label('Beam Gain (dB)')
        
        plt.tight_layout()
        
        # 저장
        filename = f"02_basic_spatial_tx_map_Area{self.config.TARGET_AREA}_{self.config.TARGET_FREQUENCY}GHz.png"
        filepath = os.path.join(self.config.OUTPUT_DIR, filename)
        plt.savefig(filepath, dpi=self.config.SAVE_DPI, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def plot_spatial_rx_beam_map(self, analysis_results: Dict[str, Any]) -> str:
        """단말 위치별 최상위 수신 빔 공간 분포 맵"""
        
        rx_beam_rankings = analysis_results['rx_beam_rankings']
        
        # 위치와 빔 데이터 추출
        positions = []
        rx_beams = []
        gains_db = []
        
        for rx_analysis in rx_beam_rankings.values():
            pos = rx_analysis['position']
            if pos:
                positions.append([pos['x'], pos['y']])
                rx_beams.append(rx_analysis['best_rx_beam'])
                gains_db.append(rx_analysis['best_gain_db'])
        
        if not positions:
            print("Warning: No position data available for RX spatial map")
            return ""
        
        positions = np.array(positions)
        rx_beams = np.array(rx_beams)
        gains_db = np.array(gains_db)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # 1. 수신 빔 인덱스 분포 맵
        scatter1 = ax1.scatter(positions[:, 0], positions[:, 1], c=rx_beams, 
                            cmap='hsv', s=30, alpha=0.8)
        ax1.set_xlabel('X Position (m)')
        ax1.set_ylabel('Y Position (m)')
        ax1.set_title('Spatial Distribution of Best RX Beam')
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        # 송신기 위치 표시
        tx_pos = self.config.AREA1_TX_POSITION
        ax1.scatter(tx_pos[0], tx_pos[1], c='red', s=200, marker='^', 
                   label='TX Position', edgecolors='black', linewidth=2)
        ax1.legend()
        
        # 컬러바 추가
        cbar1 = plt.colorbar(scatter1, ax=ax1)
        cbar1.set_label('RX Beam Index')
        
        # 2. 빔 이득 분포 맵 (TX 맵과 동일한 이득 정보)
        scatter2 = ax2.scatter(positions[:, 0], positions[:, 1], c=gains_db, 
                              cmap='RdYlBu_r', s=30, alpha=0.8)
        ax2.set_xlabel('X Position (m)')
        ax2.set_ylabel('Y Position (m)')
        ax2.set_title('Spatial Distribution of Best Beam Gain')
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal')
        
        # 송신기 위치 표시
        ax2.scatter(tx_pos[0], tx_pos[1], c='black', s=200, marker='^', 
                   label='TX Position', edgecolors='white', linewidth=2)
        ax2.legend()
        
        # 컬러바 추가
        cbar2 = plt.colorbar(scatter2, ax=ax2)
        cbar2.set_label('Beam Gain (dB)')
        
        plt.tight_layout()
        
        # 저장
        filename = f"02_basic_spatial_rx_map_Area{self.config.TARGET_AREA}_{self.config.TARGET_FREQUENCY}GHz.png"
        filepath = os.path.join(self.config.OUTPUT_DIR, filename)
        plt.savefig(filepath, dpi=self.config.SAVE_DPI, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def plot_gain_cdf(self, analysis_results: Dict[str, Any]) -> str:
        """빔 이득 CDF 분포"""
        
        rx_beam_rankings = analysis_results['rx_beam_rankings']
        gains_db = [rx['best_gain_db'] for rx in rx_beam_rankings.values()]
        
        if not gains_db:
            print("Warning: No gain data available for CDF plot")
            return ""
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # CDF 계산 및 플롯
        sorted_gains = np.sort(gains_db)
        cdf_values = np.arange(1, len(sorted_gains) + 1) / len(sorted_gains)
        
        ax.plot(sorted_gains, cdf_values, linewidth=2, color='blue')
        ax.set_xlabel('Best Beam Gain (dB)')
        ax.set_ylabel('Cumulative Probability')
        ax.set_title('CDF of Best Beam Gain Distribution')
        ax.grid(True, alpha=0.3)
        
        # 통계 정보 추가
        mean_gain = np.mean(gains_db)
        median_gain = np.median(gains_db)
        percentile_10 = np.percentile(gains_db, 10)
        percentile_90 = np.percentile(gains_db, 90)
        
        ax.axvline(mean_gain, color='red', linestyle='--', alpha=0.7, 
                  label=f'Mean: {mean_gain:.1f} dB')
        ax.axvline(median_gain, color='green', linestyle='--', alpha=0.7, 
                  label=f'Median: {median_gain:.1f} dB')
        ax.axvline(percentile_10, color='orange', linestyle=':', alpha=0.7, 
                  label=f'10th percentile: {percentile_10:.1f} dB')
        ax.axvline(percentile_90, color='purple', linestyle=':', alpha=0.7, 
                  label=f'90th percentile: {percentile_90:.1f} dB')
        
        ax.legend()
        
        # 저장
        filename = f"02_basic_gain_cdf_Area{self.config.TARGET_AREA}_{self.config.TARGET_FREQUENCY}GHz.png"
        filepath = os.path.join(self.config.OUTPUT_DIR, filename)
        plt.savefig(filepath, dpi=self.config.SAVE_DPI, bbox_inches='tight')
        plt.close()
        
        return filepath

# ========================================================================
# RESULTS EXPORT - 간소화된 결과 내보내기
# ========================================================================

class ResultsExporter:
    """분석 결과 내보내기 및 저장 (간소화)"""
    
    def __init__(self, config: P4_Config):
        self.config = config
        
    def export_analysis_results(self, analysis_results: Dict[str, Any], 
                              saved_plots: Dict[str, str]) -> Dict[str, str]:
        """분석 결과를 CSV 형식으로 내보내기 (우선순위)"""
        
        print("Exporting analysis results...")
        
        exported_files = {}
        
        # 1. CSV 형식으로 주요 결과 저장 (우선순위)
        if self.config.CSV_PRIORITY:
            exported_files.update(self.export_csv_results(analysis_results))
        
        # 2. 요약 리포트 생성
        exported_files['summary_report'] = self.generate_summary_report(analysis_results, saved_plots)
        
        print(f"Exported {len(exported_files)} result files")
        
        return exported_files
    
    def export_csv_results(self, analysis_results: Dict[str, Any]) -> Dict[str, str]:
        """주요 결과를 CSV 형식으로 저장"""
        
        exported_files = {}
        
        # 1. 단말별 빔 순위 CSV
        rx_rankings_data = []
        for rx_idx, rx_analysis in analysis_results['rx_beam_rankings'].items():
            row = {
                'RX_Index': rx_idx,
                'X_Position': rx_analysis['position'].get('x', 0),
                'Y_Position': rx_analysis['position'].get('y', 0),
                'Z_Position': rx_analysis['position'].get('z', 0),
                'Distance_2D_m': rx_analysis['position'].get('distance_2d_m', 0),
                'Best_TX_Beam': rx_analysis['best_tx_beam'],
                'Best_RX_Beam': rx_analysis['best_rx_beam'],
                'Best_Gain_dB': rx_analysis['best_gain_db'],
                'Total_Valid_Pairs': rx_analysis['total_valid_pairs']
            }
            
            # 상위 빔 페어 정보 추가
            for i, beam_pair in enumerate(rx_analysis['top_beam_pairs'][:self.config.TOP_BEAM_RANKS]):
                row[f'Rank{i+1}_TX_Beam'] = beam_pair['tx_beam_idx']
                row[f'Rank{i+1}_RX_Beam'] = beam_pair['rx_beam_idx']
                row[f'Rank{i+1}_Gain_dB'] = beam_pair['gain_db']
            
            rx_rankings_data.append(row)
        
        # DataFrame 생성 및 저장
        if rx_rankings_data:
            df_rx = pd.DataFrame(rx_rankings_data)
            filename = f"01_data_rx_rankings_Area{self.config.TARGET_AREA}_{self.config.TARGET_FREQUENCY}GHz.csv"
            filepath = os.path.join(self.config.OUTPUT_DIR, filename)
            df_rx.to_csv(filepath, index=False)
            exported_files['rx_rankings_csv'] = filepath
        
        # 2. 송신 빔별 통계 CSV
        tx_dominance_data = []
        total_analyzed_rx = len(analysis_results['rx_beam_rankings'])
        
        for tx_beam_idx, beam_info in analysis_results['tx_beam_dominance'].items():
            dominant_count = beam_info['rx_count']
            dominance_percentage = (dominant_count / total_analyzed_rx * 100) if total_analyzed_rx > 0 else 0.0
            
            tx_dominance_data.append({
                'TX_Beam_Index': tx_beam_idx,
                'Dominant_RX_Count': dominant_count,
                'Dominance_Percentage': round(dominance_percentage, 2),
                'Avg_Gain_dB': beam_info['avg_gain_db'],
                'Max_Gain_dB': beam_info['max_gain_db'],
                'Min_Gain_dB': beam_info['min_gain_db'],
                'Std_Gain_dB': beam_info['std_gain_db'],
                'Avg_Distance_2D_m': beam_info['avg_distance_2d_m']
            })
        
        if tx_dominance_data:
            df_tx = pd.DataFrame(tx_dominance_data)
            filename = f"01_data_tx_dominance_Area{self.config.TARGET_AREA}_{self.config.TARGET_FREQUENCY}GHz.csv"
            filepath = os.path.join(self.config.OUTPUT_DIR, filename)
            df_tx.to_csv(filepath, index=False)
            exported_files['tx_dominance_csv'] = filepath
        
        return exported_files
    
    def generate_summary_report(self, analysis_results: Dict[str, Any], 
                              saved_plots: Dict[str, str]) -> str:
        """요약 리포트 생성"""
        
        system_stats = analysis_results['system_statistics']
        tx_beam_dominance = analysis_results['tx_beam_dominance']
        
        # 리포트 작성
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append(f"P4v2 빔포밍 성능 분석 리포트")
        report_lines.append(f"Area {self.config.TARGET_AREA}, {self.config.TARGET_FREQUENCY} GHz")
        report_lines.append(f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # 1. 전체 시스템 통계
        report_lines.append("1. 전체 시스템 통계")
        report_lines.append("-" * 40)
        
        coverage_stats = system_stats['coverage_statistics']
        total_analyzed = coverage_stats['total_analyzed_rx']
        total_expected = coverage_stats['total_expected_rx']
        
        report_lines.append(f"분석된 단말 수: {total_analyzed}/{total_expected}")
        report_lines.append(f"분석 커버리지: {coverage_stats['analysis_coverage_ratio']*100:.1f}%")
        report_lines.append("")
        
        gain_stats = system_stats['gain_statistics']
        report_lines.append("빔 이득 통계:")
        report_lines.append(f"  - 평균: {gain_stats['mean_gain_db']:.2f} dB")
        report_lines.append(f"  - 표준편차: {gain_stats['std_gain_db']:.2f} dB")
        report_lines.append(f"  - 범위: [{gain_stats['min_gain_db']:.2f}, {gain_stats['max_gain_db']:.2f}] dB")
        report_lines.append(f"  - 중앙값: {gain_stats['median_gain_db']:.2f} dB")
        report_lines.append("")
        
        beam_usage = system_stats['beam_usage_statistics']
        report_lines.append("빔 사용 통계:")
        report_lines.append(f"  - 사용된 송신 빔: {beam_usage['used_tx_beams']}/{beam_usage['total_tx_beams']}")
        report_lines.append(f"  - 빔 사용률: {beam_usage['usage_ratio']*100:.1f}%")
        report_lines.append(f"  - 송신 빔당 평균 단말 수: {beam_usage['mean_rx_per_beam']:.1f}")
        report_lines.append(f"  - 송신 빔당 최대 단말 수: {beam_usage['max_rx_per_beam']}")
        report_lines.append("")
        
        # 2. 상위 송신 빔 분석
        report_lines.append("2. 상위 송신 빔 분석")
        report_lines.append("-" * 40)
        
        sorted_beams = sorted(tx_beam_dominance.items(), 
                            key=lambda x: x[1]['rx_count'], reverse=True)
        
        report_lines.append("상위 10개 송신 빔 (단말 수 기준):")
        report_lines.append("순위  빔인덱스  단말수  평균이득(dB)  최대이득(dB)  평균거리(m)")
        report_lines.append("-" * 65)
        
        for i, (beam_idx, beam_info) in enumerate(sorted_beams[:10], 1):
            report_lines.append(f"{i:2d}    {beam_idx:4d}     {beam_info['rx_count']:3d}    "
                              f"{beam_info['avg_gain_db']:6.1f}       {beam_info['max_gain_db']:6.1f}     "
                              f"{beam_info['avg_distance_2d_m']:6.1f}")
        
        report_lines.append("")
        
        # 3. 생성된 파일 목록
        report_lines.append("3. 생성된 파일 목록")
        report_lines.append("-" * 40)
        
        report_lines.append("시각화 파일:")
        for plot_name, plot_path in saved_plots.items():
            if plot_path:
                report_lines.append(f"  - {plot_name}: {os.path.basename(plot_path)}")
        
        report_lines.append("")
        report_lines.append("데이터 파일:")
        if self.config.CSV_PRIORITY:
            report_lines.append(f"  - RX 순위 CSV: 01_data_rx_rankings_Area{self.config.TARGET_AREA}_{self.config.TARGET_FREQUENCY}GHz.csv")
            report_lines.append(f"  - TX 통계 CSV: 01_data_tx_dominance_Area{self.config.TARGET_AREA}_{self.config.TARGET_FREQUENCY}GHz.csv")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        
        # 리포트 저장
        filename = f"01_data_summary_Area{self.config.TARGET_AREA}_{self.config.TARGET_FREQUENCY}GHz.txt"
        filepath = os.path.join(self.config.OUTPUT_DIR, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        return filepath

# ========================================================================
# MAIN EXECUTION PIPELINE - 간소화된 파이프라인
# ========================================================================

class P4Pipeline:
    """P4 메인 실행 파이프라인 (TX/RX 빔 클러스터링)"""
    
    def __init__(self):
        self.config = config
        self.data_loader = P3DataLoader(self.config)
        self.position_generator = RXPositionGenerator(self.config)
        self.beam_analyzer = BeamRankingAnalyzer(self.config)
        self.tx_beam_clustering_analyzer = TXBeamClusteringAnalyzer(self.config)
        self.rx_beam_clustering_analyzer = RXBeamClusteringAnalyzer(self.config)
        self.basic_visualizer = BasicVisualizer(self.config)
        self.results_exporter = ResultsExporter(self.config)
    
    def execute_analysis(self):
        """전체 P4 분석 파이프라인 실행 (TX/RX 빔 클러스터링 포함)"""
        
        print("=" * 80)
        print("P4: TX/RX 빔 클러스터링 분석 파이프라인 시작")
        print(f"Target: Area{self.config.TARGET_AREA}_{self.config.TARGET_FREQUENCY}GHz")
        print("=" * 80)
        
        try:
            # 1. P3 결과 데이터 로딩
            print("\n[Step 1/6] P3 결과 데이터 로딩")
            p3_results = self.data_loader.load_p3_results()
            
            # 2. P1 수신기 위치 정보 생성
            print("\n[Step 2/6] P1 수신기 위치 정보 생성")
            rx_positions = self.position_generator.generate_area1_rx_positions()
            
            # 3. 빔 순위 분석
            print("\n[Step 3/6] 빔 순위 분석 실행")
            analysis_results = self.beam_analyzer.rank_beam_pairs(p3_results, rx_positions)
            
            # 4. TX 빔 클러스터링 분석
            tx_beam_clustering_results = None
            if self.config.ENABLE_TX_CLUSTERING:
                print("\n[Step 4a/6] TX 빔 클러스터링 분석")
                tx_beam_clustering_results = self.tx_beam_clustering_analyzer.run_clustering_analysis(analysis_results)
            
            # 5. RX 빔 클러스터링 분석
            rx_beam_clustering_results = None
            if self.config.ENABLE_RX_CLUSTERING:
                print("\n[Step 4b/6] RX 빔 클러스터링 분석")
                rx_beam_clustering_results = self.rx_beam_clustering_analyzer.run_clustering_analysis(analysis_results)
            
            # 6. 기본 시각화 및 결과 내보내기
            print("\n[Step 5/6] 기본 시각화 및 결과 저장")
            saved_plots = self.basic_visualizer.create_basic_visualizations(analysis_results)
            exported_files = self.results_exporter.export_analysis_results(analysis_results, saved_plots)
            
            # 완료 요약
            print("\n[완료] 분석 완료 요약")
            self.print_completion_summary(analysis_results, saved_plots, exported_files, 
                                        tx_beam_clustering_results, rx_beam_clustering_results)
            
            return (analysis_results, saved_plots, exported_files, 
                   tx_beam_clustering_results, rx_beam_clustering_results)
            
        except Exception as e:
            print(f"Error during P4v2 analysis: {e}")
            raise
    
    def print_completion_summary(self, analysis_results: Dict[str, Any], 
                               saved_plots: Dict[str, str], 
                               exported_files: Dict[str, str],
                               tx_beam_clustering_results: Optional[Dict[str, Any]] = None,
                               rx_beam_clustering_results: Optional[Dict[str, Any]] = None):
        """분석 완료 요약 출력"""
        
        print("=" * 80)
        print("P4 TX/RX 빔포밍 성능 분석 완료")
        print("=" * 80)
        
        # 분석 통계 요약
        system_stats = analysis_results['system_statistics']
        coverage_stats = system_stats['coverage_statistics']
        beam_usage = system_stats['beam_usage_statistics']
        
        print(f"분석 결과 요약:")
        print(f"  - 분석된 단말 수: {coverage_stats['total_analyzed_rx']}")
        print(f"  - 사용된 송신 빔 수: {beam_usage['used_tx_beams']}/{beam_usage['total_tx_beams']}")
        print(f"  - 빔 사용률: {beam_usage['usage_ratio']*100:.1f}%")
        
        # 생성된 파일 요약
        print(f"\n생성된 파일:")
        print(f"  - 시각화 파일: {len([p for p in saved_plots.values() if p])}개")
        print(f"  - 데이터 파일: {len(exported_files)}개")
        print(f"  - 저장 위치: {self.config.OUTPUT_DIR}")
        
        # 주요 결과 하이라이트
        gain_stats = system_stats['gain_statistics']
        tx_beam_dominance = analysis_results['tx_beam_dominance']
        top_beam = max(tx_beam_dominance.items(), key=lambda x: x[1]['rx_count'])
        
        print(f"\n주요 결과:")
        print(f"  - 평균 빔 이득: {gain_stats['mean_gain_db']:.1f} dB")
        print(f"  - 최상위 송신 빔: Beam {top_beam[0]} ({top_beam[1]['rx_count']}개 단말)")
        print(f"  - 빔 이득 범위: [{gain_stats['min_gain_db']:.1f}, {gain_stats['max_gain_db']:.1f}] dB")
        
        # TX 빔 클러스터링 결과
        if tx_beam_clustering_results and 'clusters_info' in tx_beam_clustering_results:
            print(f"\nTX 빔 클러스터링 분석 결과:")
            
            tx_clusters = tx_beam_clustering_results['clusters_info']
            tx_quality_metrics = tx_beam_clustering_results.get('quality_metrics', {})
            
            print(f"  - 발견된 TX 클러스터 수: {len(tx_clusters)}")
            print(f"  - 분석된 단말 수: {tx_beam_clustering_results.get('num_rx_analyzed', 0)}")
            print(f"  - 상위 K개 TX 빔: {self.config.TX_TOP_K_BEAMS}")
            print(f"  - TX 최소 유사도 기준: {self.config.TX_MIN_INTRA_SIMILARITY}")
            
            if tx_quality_metrics:
                print(f"  - TX 클러스터 내 평균 유사도: {tx_quality_metrics.get('avg_intra_cluster_similarity', 0):.3f}")
                print(f"  - TX 클러스터 간 평균 유사도: {tx_quality_metrics.get('avg_inter_cluster_similarity', 0):.3f}")
            
            if tx_clusters:
                # 가장 큰 클러스터 정보
                largest_cluster = max(tx_clusters, key=lambda c: c['size'])
                print(f"  - TX 최대 클러스터: C{largest_cluster['cluster_id']} ({largest_cluster['size']}개 단말)")
                
        # RX 빔 클러스터링 결과
        if rx_beam_clustering_results and 'clusters_info' in rx_beam_clustering_results:
            print(f"\nRX 빔 클러스터링 분석 결과:")
            
            rx_clusters = rx_beam_clustering_results['clusters_info']
            rx_quality_metrics = rx_beam_clustering_results.get('quality_metrics', {})
            
            print(f"  - 발견된 RX 클러스터 수: {len(rx_clusters)}")
            print(f"  - 분석된 단말 수: {rx_beam_clustering_results.get('num_rx_analyzed', 0)}")
            print(f"  - 상위 K개 RX 빔: {self.config.RX_TOP_K_BEAMS}")
            print(f"  - RX 최소 유사도 기준: {self.config.RX_MIN_INTRA_SIMILARITY}")
            
            if rx_quality_metrics:
                print(f"  - RX 클러스터 내 평균 유사도: {rx_quality_metrics.get('avg_intra_cluster_similarity', 0):.3f}")
                print(f"  - RX 클러스터 간 평균 유사도: {rx_quality_metrics.get('avg_inter_cluster_similarity', 0):.3f}")
            
            if rx_clusters:
                # 가장 큰 클러스터 정보
                largest_cluster = max(rx_clusters, key=lambda c: c['size'])
                print(f"  - RX 최대 클러스터: C{largest_cluster['cluster_id']} ({largest_cluster['size']}개 단말)")
        
        print("=" * 80)

# ========================================================================
# MAIN EXECUTION
# ========================================================================

def main():
    """메인 실행 함수"""
    try:
        pipeline = P4Pipeline()
        result = pipeline.execute_analysis()
        return result
    except Exception as e:
        print(f"P4 TX/RX 빔 클러스터링 분석 실패: {e}")
        return None, None, None, None, None

if __name__ == "__main__":
    main()
