#!/usr/bin/env python3
"""
P3_OFDM_Ch_to_ChMeanCov_2509v5.py

Channel Covariance Matrix (CCM) 계산: 데이터 효율성 최적화 + 중간 결과 영구 보존
- 메모리 제약 해제: 선택적 로딩으로 8-16배 효율화
- 독립형 아키텍처: 외부 의존성 없는 완전 자립 설계
- 블록별 순차 계산: R_ii, R_ij 블록 단위 처리
- 무제한 확장성: 대용량 사용자 + 다중 Area 지원
- 중간 저장: R_ii 완료 후 즉시 저장, 긴 처리시간 대응
- 영구 보존: working directory로 중간 결과 보존, 재시작 시 이어서 계속

핵심 구성요소:
1. P3_Config: Area-RX 매핑 및 설정 중앙 관리
2. CCM_BlockEngine: 블록별 선택적 처리 엔진
3. CCM_BlockManager: 블록 관리, 저장, 재조합
4. RXCoordinateExtractor: P1 area 설정 기반 RX 좌표 추출
5. DataEfficiencyFilter: 채널 품질 및 거리 기반 필터링
6. validate_rand_sampled_ccm: pMSE 기반 랜덤 샘플링 일관성 검증

데이터 효율성 최적화:
- Zero 채널 제거: All-zero 채널 RX 자동 감지 및 제외
- 거리 기반 필터링: 물리적으로 의미있는 RX 쌍만 R_ij 계산
- 메타데이터 추적: 필터링된 RX 정보 완전 보존
- 대규모 지원: area_5 (1600 RX) 등 대규모 시나리오 효율적 처리

기술 표준:
- 파일 패턴: P2_Area{id}_RX{id}_{freq}GHz*.npy
- 3단계 스텁 체계: 기본/상세/구현 차등 적용
- 동적 정보 추출: 파일명에서 실험 설정 자동 감지
- pMSE 평가: dB 스케일 출력, 오버랩 비율 분석
- 스마트 필터링: 계산량 최대 90% 감소
"""

# 기본 시스템 라이브러리
import os
import re
import numpy as np
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt
import tempfile
import shutil
import time
import math

from typing import Dict, List, Tuple, Optional, Any

# 주피터/IPython 화면 클리어 지원
try:
    from IPython.display import clear_output
    JUPYTER_AVAILABLE = True
except ImportError:
    JUPYTER_AVAILABLE = False

# GPU 최적화 설정
try:
    for device in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(device, True)
    print(f"GPU 메모리 증가 허용 설정 완료: {len(tf.config.list_physical_devices('GPU'))}개 GPU")
except:
    print("GPU 설정 실패, CPU 모드로 실행")

# Matplotlib 설정 (해상도 및 크기 최적화)
plt.rcParams['figure.figsize'] = (10, 5)  # 적당한 크기
plt.rcParams['figure.dpi'] = 100          # 기본 해상도
plt.rcParams['savefig.dpi'] = 150         # 저장 시 고해상도
plt.rcParams['font.size'] = 10            # 기본 폰트 크기

# ===============================
# P3 핵심 클래스
# ===============================

class P3_Config:
    """Area-RX 매핑 및 설정 관리 (데이터 효율성 최적화)
    
    역할: 
    - P2 파일 패턴 자동 감지 및 Area-RX 매핑
    - 모든 설정값 중앙 집중화 (은닉 의존성 제거)
    - 동적 안테나 수 추출 및 캐시 관리
    - 스마트 데이터 필터링 제어
    """
    
    def __init__(self):
        """P3_Config 초기화 (데이터 효율성 강화)"""
        # 스크립트 디렉토리 기준 경로 설정
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.p2_data_dir = os.path.join(script_dir, "OFDM_Ch_Results")
        self.ccm_save_dir = os.path.join(script_dir, "ChMeanCov_Results")
        
        # Area 필터링 설정 (None이면 모든 area 처리, 리스트로 특정 area 지정 가능)
        self.target_areas = [1]              # 예: [1, 3, 5] 또는 None
        
        # 영구 중간 계산 디렉토리 (커널 종료 시 결과 보존)
        self.ccm_working_dir = os.path.join(script_dir, "ChMeanCov_WorkingDir")
        
        # 파일 패턴 설정 (개별 파일 + 청크 파일 지원)
        self.P2_INDIVIDUAL_PATTERN = r"Area(\d+)_(\d+(?:\.\d+)?)GHz_RX(\d+)_.*\.npy"
        self.P2_CHUNK_PATTERN = r"Area(\d+)_(\d+(?:\.\d+)?)GHz_OFDM_Ch_RX(\d+)-(\d+)\.npz"
        self.CCM_FILE_PATTERN = "CCM_Area{area}_{freq}GHz.npy"
        
        # 파일 로딩 우선순위 설정
        self.PREFER_CHUNK_FILES = True  # True: 청크 파일 우선, False: 개별 파일 우선
        
        # Area-RX 매핑 구조
        self.area_rx_mapping: Dict[int, List[int]] = {}  # {area_id: [rx_id_list]}
        self.rx_file_mapping: Dict[Tuple[int, int], Dict[str, Any]] = {}  # {(area_id, rx_id): file_info}
        self.antenna_counts: Dict[int, int] = {}  # {area_id: N_t} 동적 추출 캐시
        
        # 데이터 효율성 필터링 설정
        self.filtering_enabled = True  # 마스터 필터링 스위치
        
        # Zero 채널 필터링 설정
        self.zero_channel_filtering = {
            'enable_filtering': True,           # All-zero 채널 RX 제거 활성화
            'threshold': 1e-20,                # 채널이 zero로 간주되는 임계값 (power 기준)
            'check_method': 'power_sum',       # 'power_sum', 'frobenius_norm', 'max_abs'
            'preserve_info': True,             # 제외된 RX 정보 메타데이터에 보존
        }
        
        # 거리 기반 R_ij 필터링 설정
        self.distance_filtering = {
            'enable_filtering': True,          # 거리 기반 R_ij 필터링 활성화
            'max_distance_meters': 25.0,     # R_ij 계산 최대 거리 (미터)
            'filtering_mode': 'skip',          # 'skip': 계산 안함, 'zero': zero 행렬 저장
            'distance_metric': 'euclidean',   # 'euclidean', 'manhattan'
            'preserve_distance_info': True,   # 거리 통계 메타데이터에 보존
        }
        
        # P1 Area 설정 연동
        self.p1_integration = {
            'use_p1_coordinates': True,       # P1 area 설정에서 RX 좌표 추출
            'p1_config_source': 'internal',   # 'internal': 내장 설정, 'file': 외부 파일
            'fallback_to_uniform_grid': True, # P1 설정 없을 시 균등 그리드로 fallback
        }
        
        # 실행 모드 설정 (명시적)
        # 'validation': 랜덤 샘플링 일관성 검증 (저장 안함)
        # 'save': CCM 계산 및 파일로 저장
        # 'correct': 저장된 CCM의 유효성 검사 및 보정 후 덮어쓰기
        self.execution_mode = 'save'

        # 랜덤 샘플링 CCM 검증 설정 (부록 J) - execution_mode='validation' 일 때 사용
        self.validation_config = {
            'num_trials': 5,                 # 반복 실험 횟수
            'seed_base': 42,                 # 시드 기준값 (trial별 42+trial)
        }

        # 샘플링 설정 (부록 J)
        self.sampling_config = {
            'sampling_ratio': 0.8,           # 기존 설정 유지
            'random_sampling': True,         # 검증용: True로 변경
            'seed': 42,                      # 재현가능성용 시드
        }
                # CCM 보정 설정 (동적 스테이지 기반)
        self.ccm_correction = {
            'enable_correction': True,
            'max_iterations': 1500,
            'eps_min_stages': {     # 반복 횟수에 따른 eps_min 단계별 조정
                0: 1e-4, # 1e-5
                499: 1e-3 # 1e-4
            },
            'tolerance_stages': {   # 반복 횟수에 따른 tolerance 단계별 조정
                0: 1e-15
            }
        }
        
        # 콘솔 출력 제어 설정 (P2 방식 참고 + RX 많은 환경 최적화)
        self.ENABLE_SCREEN_CLEAR = True      # 화면 클리어 기능 활성화/비활성화
        self.PROGRESS_CLEAR_INTERVAL = 1     # Area마다 상세 정보 출력
        self.ENABLE_MEMORY_MONITORING = False # GPU 메모리 사용량 모니터링 (선택적)
        self.ENABLE_DETAILED_PROGRESS = True  # 상세 진행률 표시
        
        # 중간 저장 설정 (영구 저장으로 개선)
        self.ENABLE_INTERMEDIATE_SAVE = True  # R_ii 완료 후 중간 저장 활성화
        self.CLEANUP_INTERMEDIATE_FILE = True # R_ii 백업 파일(*_R_ii_blocks.npz) 자동 삭제
        
        # 중간 결과 재사용 설정
        self.RESUME_FROM_EXISTING = True      # 기존 중간 결과 재사용 (ChMeanCov_WorkingDir/ 보존)
        
        # 출력 250줄 제한 대응 설정 (RX 많은 환경)
        self.MAX_OUTPUT_LINES = 100           # 안전한 최대 출력 라인 수 (250줄 제한 고려)
        self.ENABLE_BLOCK_LEVEL_CLEAR = True  # 블록 처리 중 화면 클리어
        self.RX_CLEAR_THRESHOLD = 10          # RX 개수가 이 값을 초과하면 중간 클리어 활성화
        self.BLOCK_CLEAR_INTERVAL = 10        # N개 블록마다 중간 클리어
        
        # 자동 감지 실행
        self.detect_area_rx_mapping()
        
        # 출력 디렉토리 생성 (working dir 포함)
        os.makedirs(self.ccm_save_dir, exist_ok=True)
        os.makedirs(self.ccm_working_dir, exist_ok=True)
    
    def detect_area_rx_mapping(self) -> Dict[int, List[int]]:
        """P2 파일 패턴 자동 감지: 개별 파일 + 청크 파일 지원"""
        if not os.path.exists(self.p2_data_dir):
            raise FileNotFoundError(f"P2 데이터 디렉토리가 존재하지 않습니다: {self.p2_data_dir}")
        
        # 패턴 컴파일
        individual_pattern = re.compile(self.P2_INDIVIDUAL_PATTERN)
        chunk_pattern = re.compile(self.P2_CHUNK_PATTERN)
        
        # 임시 저장소 (우선순위 처리를 위해)
        individual_files = {}  # {(area_id, rx_id): file_info}
        chunk_files = {}       # {(area_id, rx_id): file_info}
        
        for filename in os.listdir(self.p2_data_dir):
            file_path = os.path.join(self.p2_data_dir, filename)
            
            # 1. 개별 파일 매칭
            individual_match = individual_pattern.match(filename)
            if individual_match:
                area_id = int(individual_match.group(1))
                freq_ghz = float(individual_match.group(2))
                rx_id = int(individual_match.group(3))
                
                # target_areas 필터링 적용
                if self.target_areas is not None and area_id not in self.target_areas:
                    continue
                
                individual_files[(area_id, rx_id)] = {
                    'area_id': area_id,
                    'freq_ghz': freq_ghz,
                    'rx_id': rx_id,
                    'file_path': file_path,
                    'filename': filename,
                    'file_type': 'individual'
                }
                continue
            
            # 2. 청크 파일 매칭
            chunk_match = chunk_pattern.match(filename)
            if chunk_match:
                area_id = int(chunk_match.group(1))
                freq_ghz = float(chunk_match.group(2))
                rx_start = int(chunk_match.group(3))
                rx_end = int(chunk_match.group(4))
                
                # target_areas 필터링 적용
                if self.target_areas is not None and area_id not in self.target_areas:
                    continue
                
                # 청크 파일에서 실제 RX 목록 확인
                try:
                    with np.load(file_path) as chunk_data:
                        if 'rx_indices' in chunk_data:
                            rx_indices = chunk_data['rx_indices']
                        else:
                            # rx_indices가 없으면 파일명 기반으로 추정
                            rx_indices = list(range(rx_start, rx_end + 1))
                    
                    # 각 RX에 대해 청크 파일 정보 저장
                    for rx_id in rx_indices:
                        chunk_files[(area_id, rx_id)] = {
                            'area_id': area_id,
                            'freq_ghz': freq_ghz,
                            'rx_id': rx_id,
                            'file_path': file_path,
                            'filename': filename,
                            'file_type': 'chunk',
                            'chunk_rx_indices': rx_indices,
                            'chunk_key': f'ofdm_ch_rx_{rx_id}'
                        }
                
                except Exception as e:
                    print(f"Warning: 청크 파일 {filename} 읽기 실패: {e}")
                    continue
        
        # 3. 우선순위에 따른 파일 선택 및 매핑 구축
        for (area_id, rx_id) in set(individual_files.keys()) | set(chunk_files.keys()):
            # Area-RX 매핑 업데이트
            if area_id not in self.area_rx_mapping:
                self.area_rx_mapping[area_id] = []
            if rx_id not in self.area_rx_mapping[area_id]:
                self.area_rx_mapping[area_id].append(rx_id)
            
            # 우선순위에 따른 파일 선택
            if self.PREFER_CHUNK_FILES and (area_id, rx_id) in chunk_files:
                self.rx_file_mapping[(area_id, rx_id)] = chunk_files[(area_id, rx_id)]
            elif (area_id, rx_id) in individual_files:
                self.rx_file_mapping[(area_id, rx_id)] = individual_files[(area_id, rx_id)]
            elif (area_id, rx_id) in chunk_files:
                self.rx_file_mapping[(area_id, rx_id)] = chunk_files[(area_id, rx_id)]
        
        # RX ID 정렬
        for area_id in self.area_rx_mapping:
            self.area_rx_mapping[area_id].sort()
        
        print(f"파일 감지 완료:")
        print(f"  - 개별 파일: {len(individual_files)}개")
        print(f"  - 청크 파일: {len(set(f['file_path'] for f in chunk_files.values()))}개 (RX {len(chunk_files)}개)")
        print(f"  - 최종 매핑: {sum(len(rxs) for rxs in self.area_rx_mapping.values())}개 RX")
        
        # Area 필터링 정보 출력
        if self.target_areas is None:
            print(f"  - Area 필터링: 모든 Area 처리")
        else:
            areas_str = ', '.join([f"Area{area}" for area in sorted(self.target_areas)])
            print(f"  - Area 필터링: {areas_str} 처리")
        
        return self.area_rx_mapping
    
    def get_area_info(self, area_id: int) -> Dict[str, Any]:
        """Area 정보 + 동적 안테나 수 추출 (개별 파일 + 청크 파일 지원)"""
        if area_id not in self.area_rx_mapping:
            raise ValueError(f"Area {area_id}가 존재하지 않습니다")
        
        rx_ids = self.area_rx_mapping[area_id]
        
        # 동적 안테나 수 추출 (첫 번째 파일에서)
        if area_id not in self.antenna_counts:
            first_rx = rx_ids[0]
            file_info = self.rx_file_mapping[(area_id, first_rx)]
            file_path = file_info['file_path']
            file_type = file_info['file_type']
            
            # 파일 타입에 따른 데이터 로딩
            if file_type == 'individual':
                # 개별 파일에서 로딩 (기존 방식)
                h_data = np.load(file_path)
                self.antenna_counts[area_id] = h_data.shape[-1]  # N_t = h_data.shape[-1]
            elif file_type == 'chunk':
                # 청크 파일에서 특정 RX 데이터 로딩
                chunk_key = file_info['chunk_key']  # 'ofdm_ch_rx_{rx_id}'
                with np.load(file_path) as chunk_data:
                    if chunk_key in chunk_data:
                        h_data = chunk_data[chunk_key]
                        self.antenna_counts[area_id] = h_data.shape[-1]  # N_t = h_data.shape[-1]
                    else:
                        raise KeyError(f"청크 파일 {file_path}에서 키 {chunk_key}를 찾을 수 없습니다")
            else:
                raise ValueError(f"지원하지 않는 파일 타입: {file_type}")
        
        return {
            'area_id': area_id,
            'rx_ids': rx_ids,
            'rx_count': len(rx_ids),
            'antenna_count': self.antenna_counts[area_id]
        }
    

    def get_save_path(self, area_id: int, is_all_rxs: bool = True) -> str:
        """저장 경로: 파일 정보 기반 (P3 v1 방식)"""
        if area_id not in self.area_rx_mapping or not self.area_rx_mapping[area_id]:
            raise ValueError(f"Area {area_id} 데이터가 없습니다")
        
        first_rx = self.area_rx_mapping[area_id][0]
        freq_ghz = self.rx_file_mapping[(area_id, first_rx)]['freq_ghz']
        
        # 파일명 생성: 모든 RX 처리시 _ALL_RXs 추가
        if is_all_rxs:
            filename = f"CCM_Area{area_id}_{freq_ghz}GHz_ALL_RXs.npz"
        else:
            filename = self.CCM_FILE_PATTERN.format(area=area_id, freq=freq_ghz)
        
        return os.path.join(self.ccm_save_dir, filename)
    
    def validate_file_pattern(self, filename: str) -> bool:
        """P2 파일명 패턴 검증 (개별 파일 + 청크 파일)"""
        individual_pattern = re.compile(self.P2_INDIVIDUAL_PATTERN)
        chunk_pattern = re.compile(self.P2_CHUNK_PATTERN)
        
        return (bool(individual_pattern.match(filename)) or 
                bool(chunk_pattern.match(filename)))
    
    def get_working_dir(self, area_id: int) -> str:
        """Area별 working directory 경로 반환 및 생성"""
        if area_id not in self.area_rx_mapping:
            raise ValueError(f"Area {area_id}가 존재하지 않습니다")
            
        first_rx = self.area_rx_mapping[area_id][0]
        freq_ghz = self.rx_file_mapping[(area_id, first_rx)]['freq_ghz']
        area_dir = f"Area{area_id}_{freq_ghz}GHz"
        working_path = os.path.join(self.ccm_working_dir, area_dir)
        
        # 디렉토리 구조 생성
        os.makedirs(working_path, exist_ok=True)
        os.makedirs(os.path.join(working_path, "R_ii_blocks"), exist_ok=True)
        os.makedirs(os.path.join(working_path, "stats_blocks"), exist_ok=True)  
        os.makedirs(os.path.join(working_path, "R_ij_blocks"), exist_ok=True)
        
        return working_path
    
    def check_existing_calculations(self, area_id: int, rx_ids: List[int]) -> Dict[str, Any]:
        """기존 중간 계산 결과 확인"""
        if not self.RESUME_FROM_EXISTING:
            return {
                'completed_r_ii': [],
                'completed_r_ij': [],
                'total_r_ii': len(rx_ids),
                'total_r_ij': len(rx_ids) * (len(rx_ids) - 1) // 2
            }
        
        working_dir = self.get_working_dir(area_id)
        
        # R_ii 완료 상태 확인
        completed_r_ii = []
        for rx_id in rx_ids:
            r_ii_path = os.path.join(working_dir, "R_ii_blocks", f"R_ii_{rx_id}.npy")
            stats_path = os.path.join(working_dir, "stats_blocks", f"stats_{rx_id}.npz")
            if os.path.exists(r_ii_path) and os.path.exists(stats_path):
                completed_r_ii.append(rx_id)
        
        # R_ij 완료 상태 확인
        completed_r_ij = []
        for i in range(len(rx_ids)):
            for j in range(i + 1, len(rx_ids)):
                rx_i, rx_j = rx_ids[i], rx_ids[j]
                r_ij_path = os.path.join(working_dir, "R_ij_blocks", f"R_ij_{rx_i}_{rx_j}.npy")
                if os.path.exists(r_ij_path):
                    completed_r_ij.append((rx_i, rx_j))
        
        return {
            'completed_r_ii': completed_r_ii,
            'completed_r_ij': completed_r_ij,
            'total_r_ii': len(rx_ids),
            'total_r_ij': len(rx_ids) * (len(rx_ids) - 1) // 2
        }


class RXCoordinateExtractor:
    """P1 Area 설정에서 RX 좌표 추출 클래스
    
    역할:
    - P1 프로젝트의 AREA_CONFIGS에서 RX 좌표 정보 추출
    - 다양한 배치 방법 (grid, explicit, radial, street) 지원
    - P3에서 거리 기반 필터링에 필요한 좌표 정보 제공
    """
    
    # P1 Area 설정 (P1_RT_to_Rays_250831v5.py에서 추출)
    P1_AREA_CONFIGS = {
        'area_1': {
            'tx_positions': [[-51.560643657685944, -21.793815800460543, 19]],
            'rx_placement': {
                'method': 'grid',
                'x_params': {'start': -136.138, 'stop': 58.862, 'num': 40},
                'y_params': {'start': -117.667, 'stop': 77.333, 'num': 40},
                'z_params': {'values': [1.5]}
            },
            'description': 'Area 1: Standard coverage zone (Grid-based)'
        },
        'area_2': {
            'tx_positions': [[-51.560643657685944, -21.793815800460543, 19]],
            'rx_placement': {
                'method': 'grid',
                'x_params': {'start': 26, 'stop': 41, 'num': 4},
                'y_params': {'start': -20, 'stop': -10, 'num': 3},
                'z_params': {'values': [1.5]}
            },
            'description': 'Area 2: Dense urban coverage (Grid-based)'
        },
        'area_3': {
            'tx_positions': [[-51.560643657685944, -21.793815800460543, 19]],
            'rx_placement': {
                'method': 'explicit',
                'x_coords': [100],
                'y_coords': [100],
                'z_coords': [400]
            },
            'description': 'Area 3: Extended coverage zone (Single point)'
        },
        'area_4': {
            'tx_positions': [[-51.560643657685944, -21.793815800460543, 19]],
            'rx_placement': {
                'method': 'grid',
                'x_params': {'start': -125, 'stop': 75, 'num': 20},
                'y_params': {'start': -125, 'stop': 75, 'num': 20},
                'z_params': {'values': [1.5]}
            },
            'description': 'Area 4: Southern coverage zone (Grid-based)'
        },
        'area_5': {
            'tx_positions': [[-51.560643657685944, -21.793815800460543, 19]],
            'rx_placement': {
                'method': 'grid',
                'x_params': {'start': -125, 'stop': 75, 'num': 40},
                'y_params': {'start': -125, 'stop': 75, 'num': 40},
                'z_params': {'values': [1.5]}
            },
            'description': 'Area 5: Large-scale coverage zone (1600 RX)'
        }
    }
    
    @staticmethod
    def get_area_config(area_id: int) -> Dict[str, Any]:
        """지정된 Area의 P1 설정을 반환"""
        area_key = f'area_{area_id}'
        if area_key not in RXCoordinateExtractor.P1_AREA_CONFIGS:
            raise ValueError(f"Area {area_id}의 P1 설정을 찾을 수 없습니다. 사용 가능한 Area: {list(RXCoordinateExtractor.P1_AREA_CONFIGS.keys())}")
        return RXCoordinateExtractor.P1_AREA_CONFIGS[area_key]
    
    @staticmethod
    def extract_rx_coordinates(area_id: int) -> Dict[int, Tuple[float, float, float]]:
        """지정된 Area의 RX 좌표 목록을 생성 (P1과 동일한 순서 보장)"""
        area_config = RXCoordinateExtractor.get_area_config(area_id)
        placement_config = area_config['rx_placement']
        method = placement_config['method']
        
        rx_positions = {}
        
        if method == 'grid':
            x_p = placement_config['x_params']
            y_p = placement_config['y_params']
            z_p = placement_config['z_params']
            
            rx_x_coords = np.linspace(x_p['start'], x_p['stop'], x_p['num'])
            rx_y_coords = np.linspace(y_p['start'], y_p['stop'], y_p['num'])
            rx_z_coords = z_p['values']
            
            # P1과 동일한 순서: z -> y -> x (중요!)
            rx_id = 1  # RX ID는 1부터 시작 (P2 데이터와 일치)
            for z in rx_z_coords:
                for y in rx_y_coords:
                    for x in rx_x_coords:
                        rx_positions[rx_id] = (float(x), float(y), float(z))
                        rx_id += 1
                        
        elif method == 'explicit':
            x_coords = placement_config['x_coords']
            y_coords = placement_config['y_coords'] 
            z_coords = placement_config['z_coords']
            
            rx_id = 1
            for z in z_coords:
                for y in y_coords:
                    for x in x_coords:
                        rx_positions[rx_id] = (float(x), float(y), float(z))
                        rx_id += 1
                        
        elif method == 'radial':
            center = placement_config['center']
            radii = placement_config['radii_m']
            angles_p = placement_config['angles_deg']
            z_coords = placement_config['z_params']['values']
            
            angles_rad = np.linspace(np.deg2rad(angles_p['start']), 
                                   np.deg2rad(angles_p['stop']), 
                                   angles_p['num'])
            
            rx_id = 1
            for z in z_coords:
                for r in radii:
                    for angle in angles_rad:
                        x = center[0] + r * np.cos(angle)
                        y = center[1] + r * np.sin(angle)
                        rx_positions[rx_id] = (float(x), float(y), float(z))
                        rx_id += 1
                        
        elif method == 'street':
            points = placement_config['path_points']
            num_points = placement_config['num_points']
            z_coords = placement_config['z_params']['values']
            
            start_point = np.array(points[0])
            end_point = np.array(points[1])
            
            rx_id = 1
            for z in z_coords:
                for i in range(num_points):
                    t = i / (num_points - 1) if num_points > 1 else 0
                    point = start_point + t * (end_point - start_point)
                    rx_positions[rx_id] = (float(point[0]), float(point[1]), float(z))
                    rx_id += 1
        else:
            raise ValueError(f"지원하지 않는 RX 배치 방법: {method}")
        
        print(f"Area {area_id}: {len(rx_positions)}개 RX 좌표 추출 완료 (방법: {method})")
        return rx_positions
    
    @staticmethod
    def get_rx_count(area_id: int) -> int:
        """Area의 총 RX 수 반환 (좌표 생성 없이)"""
        area_config = RXCoordinateExtractor.get_area_config(area_id)
        placement_config = area_config['rx_placement']
        method = placement_config['method']
        
        if method == 'grid':
            x_num = placement_config['x_params']['num']
            y_num = placement_config['y_params']['num']
            z_num = len(placement_config['z_params']['values'])
            return x_num * y_num * z_num
            
        elif method == 'explicit':
            x_num = len(placement_config['x_coords'])
            y_num = len(placement_config['y_coords'])
            z_num = len(placement_config['z_coords'])
            return x_num * y_num * z_num
            
        elif method == 'radial':
            r_num = len(placement_config['radii_m'])
            a_num = placement_config['angles_deg']['num']
            z_num = len(placement_config['z_params']['values'])
            return r_num * a_num * z_num
            
        elif method == 'street':
            point_num = placement_config['num_points']
            z_num = len(placement_config['z_params']['values'])
            return point_num * z_num
        else:
            return 0


class DataEfficiencyFilter:
    """데이터 효율성 필터링 엔진
    
    역할:
    - All-zero 채널 RX 감지 및 제거
    - 거리 기반 R_ij 계산 필터링
    - 필터링 결과 메타데이터 생성
    """
    
    def __init__(self, config: P3_Config):
        self.config = config
        self.filtering_stats = {
            'zero_channel_filtering': {
                'total_rx': 0,
                'excluded_rx_ids': [],
                'valid_rx_ids': [],
                'exclusion_ratio': 0.0
            },
            'distance_filtering': {
                'total_possible_pairs': 0,
                'calculated_pairs': 0,
                'skipped_pairs': 0,
                'filtering_ratio': 0.0,
                'distance_statistics': {}
            }
        }
    
    def check_zero_channel(self, h_raw: tf.Tensor, rx_id: int) -> bool:
        """채널 데이터가 all-zero인지 확인"""
        if not self.config.zero_channel_filtering['enable_filtering']:
            return False  # 필터링 비활성화 시 모든 채널을 유효로 간주
        
        threshold = self.config.zero_channel_filtering['threshold']
        check_method = self.config.zero_channel_filtering['check_method']
        
        if check_method == 'power_sum':
            # 전력 합계 기준
            power_sum = tf.reduce_sum(tf.square(tf.abs(h_raw)))
            is_zero = power_sum < threshold
        elif check_method == 'frobenius_norm':
            # Frobenius norm 기준
            frobenius_norm = tf.norm(h_raw, ord='fro')
            is_zero = frobenius_norm < threshold
        elif check_method == 'max_abs':
            # 최대 절대값 기준
            max_abs = tf.reduce_max(tf.abs(h_raw))
            is_zero = max_abs < threshold
        else:
            raise ValueError(f"지원하지 않는 check_method: {check_method}")
        
        return bool(is_zero.numpy())
    
    def check_zero_channel_fast(self, h_raw: tf.Tensor, rx_id: int) -> bool:
        """빠른 Zero 채널 검사 (샘플링 기반)"""
        if not self.config.zero_channel_filtering['enable_filtering']:
            return False
        
        threshold = self.config.zero_channel_filtering['threshold']
        
        # 샘플링 기반 빠른 검사 (처음 10% 데이터만 검사)
        sample_size = max(1, h_raw.shape[0] // 10)
        h_sample = h_raw[:sample_size]
        
        # 빠른 power sum 계산
        power_sum = tf.reduce_sum(tf.square(tf.abs(h_sample)))
        
        # 조기 종료: threshold보다 크면 바로 non-zero로 판단
        if power_sum > threshold * 10:  # 샘플이므로 threshold 10배로 조정
            return False
        
        # 의심스러우면 전체 데이터 검사
        return self.check_zero_channel(h_raw, rx_id)
    
    def filter_zero_channels(self, area_id: int, rx_list: List[int], 
                           load_channel_func) -> Tuple[List[int], List[int]]:
        """모든 RX의 채널을 검사하여 valid/invalid RX 분류 (최적화된 배치 처리)"""
        if not self.config.zero_channel_filtering['enable_filtering']:
            return rx_list, []  # 필터링 비활성화 시 모든 RX를 유효로 간주
        
        total_rx = len(rx_list)
        print(f"  [필터링] Zero 채널 검사 중... ({total_rx}개 RX)")
        
        valid_rx_ids = []
        excluded_rx_ids = []
        
        # 배치 크기 설정 (진행률 표시용)
        batch_size = max(1, total_rx // 20)  # 5% 단위로 진행률 표시
        processed = 0
        
        for i, rx_id in enumerate(rx_list):
            try:
                h_raw = load_channel_func(area_id, rx_id)
                if self.check_zero_channel_fast(h_raw, rx_id):
                    excluded_rx_ids.append(rx_id)
                else:
                    valid_rx_ids.append(rx_id)
                
                processed += 1
                
                # 진행률 표시 (5% 단위)
                if processed % batch_size == 0 or processed == total_rx:
                    progress = (processed / total_rx) * 100
                    print(f"    진행률: {processed}/{total_rx} ({progress:.0f}%) - 현재까지 {len(excluded_rx_ids)}개 제외")
                    
            except Exception as e:
                print(f"    Warning: RX{rx_id} 채널 로딩 실패, 제외됨: {e}")
                excluded_rx_ids.append(rx_id)
                processed += 1
        
        # 통계 업데이트
        self.filtering_stats['zero_channel_filtering'] = {
            'total_rx': len(rx_list),
            'excluded_rx_ids': excluded_rx_ids,
            'valid_rx_ids': valid_rx_ids,
            'exclusion_ratio': len(excluded_rx_ids) / len(rx_list) if rx_list else 0.0
        }
        
        print(f"  Zero 채널 필터링 완료: {len(valid_rx_ids)}개 유효, {len(excluded_rx_ids)}개 제외")
        if excluded_rx_ids and len(excluded_rx_ids) <= 10:
            print(f"    제외된 RX: {excluded_rx_ids}")
        elif excluded_rx_ids:
            print(f"    제외된 RX: {excluded_rx_ids[:5]}... (총 {len(excluded_rx_ids)}개)")
        
        return valid_rx_ids, excluded_rx_ids
    
    def should_calculate_r_ij(self, area_id: int, rx_i: int, rx_j: int, 
                             rx_coordinates: Dict[int, Tuple[float, float, float]]) -> bool:
        """거리 기반으로 R_ij 계산 필요성 판단"""
        if not self.config.distance_filtering['enable_filtering']:
            return True  # 필터링 비활성화 시 모든 쌍 계산
        
        if rx_i not in rx_coordinates or rx_j not in rx_coordinates:
            print(f"    Warning: RX{rx_i} 또는 RX{rx_j} 좌표 없음, 계산 건너뜀")
            return False
        
        # 거리 계산
        pos_i = rx_coordinates[rx_i]
        pos_j = rx_coordinates[rx_j]
        
        if self.config.distance_filtering['distance_metric'] == 'euclidean':
            distance = math.sqrt((pos_i[0] - pos_j[0])**2 + 
                               (pos_i[1] - pos_j[1])**2 + 
                               (pos_i[2] - pos_j[2])**2)
        elif self.config.distance_filtering['distance_metric'] == 'manhattan':
            distance = abs(pos_i[0] - pos_j[0]) + abs(pos_i[1] - pos_j[1]) + abs(pos_i[2] - pos_j[2])
        else:
            raise ValueError(f"지원하지 않는 distance_metric: {self.config.distance_filtering['distance_metric']}")
        
        max_distance = self.config.distance_filtering['max_distance_meters']
        return distance <= max_distance
    
    def _calculate_distance(self, pos_i: Tuple[float, float, float], pos_j: Tuple[float, float, float]) -> float:
        """두 RX 간 거리 계산"""
        if self.config.distance_filtering['distance_metric'] == 'euclidean':
            return math.sqrt((pos_i[0] - pos_j[0])**2 + 
                           (pos_i[1] - pos_j[1])**2 + 
                           (pos_i[2] - pos_j[2])**2)
        elif self.config.distance_filtering['distance_metric'] == 'manhattan':
            return abs(pos_i[0] - pos_j[0]) + abs(pos_i[1] - pos_j[1]) + abs(pos_i[2] - pos_j[2])
        else:
            raise ValueError(f"지원하지 않는 distance_metric: {self.config.distance_filtering['distance_metric']}")

    def calculate_valid_r_ij_pairs(self, rx_coordinates: Dict[int, Tuple[float, float, float]], 
                                  valid_rx_list: List[int]) -> Tuple[List[Tuple[int, int]], Dict]:
        """계산해야 할 R_ij 쌍 목록을 미리 계산하고 통계 반환"""
        valid_pairs = []
        total_pairs = len(valid_rx_list) * (len(valid_rx_list) - 1) // 2
        
        if not self.config.distance_filtering['enable_filtering']:
            # 거리 필터링 비활성화 시 모든 쌍 계산
            for i, rx_i in enumerate(valid_rx_list):
                for j in range(i + 1, len(valid_rx_list)):
                    rx_j = valid_rx_list[j]
                    valid_pairs.append((rx_i, rx_j))
            
            stats = {
                'total_pairs': total_pairs,
                'calculated_pairs': len(valid_pairs),
                'skipped_pairs': 0,
                'distance_distribution': {}
            }
        else:
            # 거리 기반 필터링
            max_distance = self.config.distance_filtering['max_distance_meters']
            distance_list = []
            
            for i, rx_i in enumerate(valid_rx_list):
                for j in range(i + 1, len(valid_rx_list)):
                    rx_j = valid_rx_list[j]
                    
                    if rx_i in rx_coordinates and rx_j in rx_coordinates:
                        distance = self._calculate_distance(rx_coordinates[rx_i], rx_coordinates[rx_j])
                        distance_list.append(distance)
                        
                        if distance <= max_distance:
                            valid_pairs.append((rx_i, rx_j))
                    else:
                        # 좌표 없는 RX는 건너뜀
                        pass
            
            # 거리 분포 계산
            if distance_list:
                distance_array = np.array(distance_list)
                distance_distribution = {
                    'mean': float(np.mean(distance_array)),
                    'std': float(np.std(distance_array)),
                    'min': float(np.min(distance_array)),
                    'max': float(np.max(distance_array)),
                    'median': float(np.median(distance_array)),
                    'within_threshold': len(valid_pairs)
                }
            else:
                distance_distribution = {}
            
            stats = {
                'total_pairs': total_pairs,
                'calculated_pairs': len(valid_pairs),
                'skipped_pairs': total_pairs - len(valid_pairs),
                'filtering_ratio': (total_pairs - len(valid_pairs)) / total_pairs if total_pairs > 0 else 0.0,
                'distance_distribution': distance_distribution
            }
        
        # 필터링 통계 업데이트
        self.filtering_stats['distance_filtering'] = stats
        

        # 사용자에게 결과 미리 알려주기
        print(f"  R_ij 쌍 계산 분석 완료:")
        print(f"    총 가능한 쌍: {total_pairs:,}개")
        print(f"    실제 계산할 쌍: {len(valid_pairs):,}개")
        if total_pairs > 0:
            efficiency = (len(valid_pairs) / total_pairs) * 100
            print(f"    계산 비율: {efficiency:.1f}% (거리 제한으로 {total_pairs - len(valid_pairs):,}개 생략)")
        
        if distance_list and self.config.distance_filtering['enable_filtering']:
            stats_dist = stats['distance_distribution']
            print(f"    거리 범위: {stats_dist['min']:.1f}m ~ {stats_dist['max']:.1f}m")
            print(f"    평균 거리: {stats_dist['mean']:.1f}m (임계값: {max_distance}m)")
        
        return valid_pairs, stats
    
    def calculate_distance_statistics(self, area_id: int, rx_list: List[int], 
                                    rx_coordinates: Dict[int, Tuple[float, float, float]]) -> Dict[str, float]:
        """RX 간 거리 통계 계산"""
        if not self.config.distance_filtering['enable_filtering']:
            return {}
        
        distances = []
        total_pairs = 0
        calculated_pairs = 0
        
        for i, rx_i in enumerate(rx_list):
            for j in range(i + 1, len(rx_list)):
                rx_j = rx_list[j]
                total_pairs += 1
                
                if rx_i in rx_coordinates and rx_j in rx_coordinates:
                    pos_i = rx_coordinates[rx_i]
                    pos_j = rx_coordinates[rx_j]
                    
                    if self.config.distance_filtering['distance_metric'] == 'euclidean':
                        distance = math.sqrt((pos_i[0] - pos_j[0])**2 + 
                                           (pos_i[1] - pos_j[1])**2 + 
                                           (pos_i[2] - pos_j[2])**2)
                    else:  # manhattan
                        distance = abs(pos_i[0] - pos_j[0]) + abs(pos_i[1] - pos_j[1]) + abs(pos_i[2] - pos_j[2])
                    
                    distances.append(distance)
                    
                    if distance <= self.config.distance_filtering['max_distance_meters']:
                        calculated_pairs += 1
        
        if distances:
            stats = {
                'min_distance': float(np.min(distances)),
                'max_distance': float(np.max(distances)),
                'mean_distance': float(np.mean(distances)),
                'median_distance': float(np.median(distances)),
                'std_distance': float(np.std(distances))
            }
        else:
            stats = {}
        
        # 필터링 통계 업데이트
        self.filtering_stats['distance_filtering'] = {
            'total_possible_pairs': total_pairs,
            'calculated_pairs': calculated_pairs,
            'skipped_pairs': total_pairs - calculated_pairs,
            'filtering_ratio': (total_pairs - calculated_pairs) / total_pairs if total_pairs > 0 else 0.0,
            'distance_statistics': stats
        }
        
        return stats
    
    def get_filtering_metadata(self) -> Dict[str, Any]:
        """필터링 결과 메타데이터 반환"""
        return {
            'v5_data_efficiency': {
                'zero_channel_filtering': self.filtering_stats['zero_channel_filtering'],
                'distance_filtering': self.filtering_stats['distance_filtering'],
                'total_efficiency_gain': {
                    'rx_reduction': self.filtering_stats['zero_channel_filtering']['exclusion_ratio'],
                    'r_ij_reduction': self.filtering_stats['distance_filtering']['filtering_ratio'],
                    'combined_reduction': (
                        self.filtering_stats['zero_channel_filtering']['exclusion_ratio'] + 
                        self.filtering_stats['distance_filtering']['filtering_ratio'] * 
                        (1 - self.filtering_stats['zero_channel_filtering']['exclusion_ratio'])
                    )
                }
            }
        }


class CCM_BlockEngine:
    """CCM 블록별 선택적 처리 엔진 (스마트 필터링 통합)
    
    역할:
    - CCM 블록별 선택적 로딩 및 계산
    - CCM 수학적 연산 (R_ii, R_ij 블록)  
    - CCM 보정 처리 (PSD, Hermitian, Unit Diagonal)
    - 데이터 효율성 필터링 (Zero 채널, 거리 기반)
    """
    
    # Termination reasons as class attributes
    REASON_MAX_ITER = "최대 반복 횟수 도달"
    REASON_EIG_STABLE = "최소 고유값 안정"
    REASON_CONVERGED = "변화량 기반 수렴"
    
    def __init__(self, config: P3_Config):
        self.config = config
        self.sampling_history = {}  # {(trial, rx_id): indices_array} - 랜덤 샘플링 추적용
        
        # 데이터 효율성 필터링 엔진 초기화
        self.efficiency_filter = DataEfficiencyFilter(config)
        
    def process_area_ccm(self, area_id: int, progress_callback=None, start_time=None) -> Dict[str, Any]:
        """Area별로 모든 CCM 블록과 채널 통계를 계산하여 영구 working directory에 저장합니다. (중간 결과 보존)"""
        area_info = self.config.get_area_info(area_id)
        original_rx_list = area_info['rx_ids']
        N_t = area_info['antenna_count']
        
        # Working directory 설정 (영구 저장)
        working_dir = self.config.get_working_dir(area_id)

        validation_logs = []
        stage_start_time = time.time()
        
        print(f"=== Area {area_id} 처리 중 ===")
        total_r_ij = len(original_rx_list) * (len(original_rx_list) - 1) // 2
        print(f"  {len(original_rx_list)}개 RX, {N_t}개 안테나, 총 {len(original_rx_list) + total_r_ij}개 블록 (R_ii {len(original_rx_list)}개 + R_ij {total_r_ij}개)")
        
        # P1 좌표 추출 (거리 필터링용)
        rx_coordinates = {}
        if self.config.distance_filtering['enable_filtering'] and self.config.p1_integration['use_p1_coordinates']:
            try:
                rx_coordinates = RXCoordinateExtractor.extract_rx_coordinates(area_id)
                print(f"  P1 좌표 추출: {len(rx_coordinates)}개 RX 좌표")
            except Exception as e:
                print(f"  Warning: P1 좌표 추출 실패, 거리 필터링 비활성화: {e}")
                self.config.distance_filtering['enable_filtering'] = False
        
        # Zero 채널 필터링
        valid_rx_list, excluded_rx_list = self.efficiency_filter.filter_zero_channels(
            area_id, original_rx_list, self._load_single_user
        )
        
        num_rx = len(valid_rx_list)
        if num_rx != len(original_rx_list):
            print(f"  필터링 후 유효 RX: {num_rx}개 (제외: {len(original_rx_list) - num_rx}개)")
        else:
            print(f"  처리 대상 RX: {num_rx}개")
        
        # v5.1: 기존 중간 결과 확인 및 재사용 계획
        existing_results = self.config.check_existing_calculations(area_id, valid_rx_list)
        if existing_results['completed_r_ii']:
            print(f"  기존 R_ii 결과 재사용: {len(existing_results['completed_r_ii'])}/{existing_results['total_r_ii']}개")
        if existing_results['completed_r_ij']:
            print(f"  기존 R_ij 결과 재사용: {len(existing_results['completed_r_ij'])}/{existing_results['total_r_ij']}개")
        
        # 출력 라인 추적기 및 진행률 추적기 초기화 (RX 많은 환경 대응)
        line_tracker = OutputLineTracker(self.config)
        progress_tracker = BlockProgressTracker(area_id, num_rx, valid_rx_list)
        need_block_clear = (self.config.ENABLE_BLOCK_LEVEL_CLEAR and 
                           num_rx > self.config.RX_CLEAR_THRESHOLD)
        
        if self.config.ENABLE_MEMORY_MONITORING:
            print_memory_usage()

        # 1. R_ii (대각 블록) 및 채널 통계 계산 및 저장
        print(f"  [1/2] R_ii 대각 블록 처리 중... ({num_rx}개)")
        line_tracker.add_lines(1)
        progress_tracker.start_r_ii_phase()
        
        for i, rx_i in enumerate(valid_rx_list):
            if progress_callback:
                progress_callback(f"R_ii 블록 처리 (RX {rx_i})", i + 1, num_rx)
            
            # v5.1: 기존 결과가 있으면 건너뛰기
            if rx_i in existing_results['completed_r_ii']:
                if self.config.ENABLE_DETAILED_PROGRESS:
                    print(f"    R_ii({rx_i}): 기존 결과 사용 (건너뜀)")
                # 진행률 추적 업데이트 (시간은 0으로)
                progress_tracker.update_r_ii_progress(0.0)
                continue
            
            block_start_time = time.time()
            h_raw_i = self._load_single_user(area_id, rx_i)
            
            # R_ii 계산, 보정 및 채널 통계 추출
            result_ii = self._calc_R_ii_block(h_raw_i)
            
            # 결과 저장 -> working directory에 저장 (v5.1)
            np.save(os.path.join(working_dir, "R_ii_blocks", f'R_ii_{rx_i}.npy'), result_ii['corrected_matrix'].numpy())
            stats_to_save = {k: v.numpy() for k, v in result_ii['channel_stats'].items()}
            np.savez(os.path.join(working_dir, "stats_blocks", f'stats_{rx_i}.npz'), **stats_to_save)
            
            # 경고 로깅
            if result_ii['final_min_eigenvalue'] < 0:
                validation_logs.append({'rx_id': rx_i, 'min_eig': result_ii['final_min_eigenvalue']})
            
            # 진행률 추적 업데이트
            block_time = time.time() - block_start_time
            progress_tracker.update_r_ii_progress(block_time)
            
            # 상세 진행률 표시
            if self.config.ENABLE_DETAILED_PROGRESS:
                print(f"    R_ii({rx_i}): {block_time:.2f}s, 최소고유값={result_ii['final_min_eigenvalue']:.2e}")
                line_tracker.add_lines(1)
            
            # 중간 화면 클리어 및 실시간 진행률 업데이트 (RX 많은 환경 대응)
            if need_block_clear and (i + 1) % self.config.BLOCK_CLEAR_INTERVAL == 0:
                # 실시간 진행률 정보 생성
                progress_summary = progress_tracker.format_progress_summary()
                if self.config.ENABLE_MEMORY_MONITORING:
                    progress_summary += f"\n{print_memory_usage.__doc__}"
                    
                if line_tracker.check_and_clear(context_info=progress_summary):
                    if self.config.ENABLE_MEMORY_MONITORING:
                        print_memory_usage()
                        line_tracker.add_lines(1)
        
        r_ii_time = time.time() - stage_start_time

        # R_ii 완료 후 중간 저장 (v2.2 -> v5.1: working_dir 기반)
        intermediate_save_path = None
        if self.config.ENABLE_INTERMEDIATE_SAVE:
            temp_manager = CCM_BlockManager(self.config)
            intermediate_save_path = temp_manager.save_r_ii_intermediate(
                area_id, valid_rx_list, N_t, working_dir
            )

        # R_ij 쌍 미리 계산 및 통계 출력 (R_ij 처리 전)
        valid_r_ij_pairs, distance_stats = self.efficiency_filter.calculate_valid_r_ij_pairs(
            rx_coordinates, valid_rx_list
        )

        # 2. R_ij (비대각 블록) 계산 및 저장 (사전 계산된 쌍 목록 사용)
        print(f"  [2/2] R_ij 상호 블록 처리 중... ({len(valid_r_ij_pairs):,}개 쌍)")
        line_tracker.add_lines(1)
        stage_start_time = time.time()
        progress_tracker.start_r_ij_phase()
        progress_tracker.set_r_ij_total_pairs(len(valid_r_ij_pairs))  # 실제 계산할 쌍 수 설정
        calculated_ij_ops = 0
        total_pairs = len(valid_rx_list) * (len(valid_rx_list) - 1) // 2
        skipped_ij_ops = total_pairs - len(valid_r_ij_pairs)
        
        # 사전 계산된 유효한 쌍들만 처리 (v5.1: 기존 결과 건너뛰기 지원)
        for pair_idx, (rx_i, rx_j) in enumerate(valid_r_ij_pairs):
            calculated_ij_ops += 1
            if progress_callback:
                progress_callback(f"R_ij 블록 처리 ({rx_i},{rx_j})", calculated_ij_ops, len(valid_r_ij_pairs))
            
            # v5.1: 기존 결과가 있으면 건너뛰기
            if (rx_i, rx_j) in existing_results['completed_r_ij']:
                if (self.config.ENABLE_DETAILED_PROGRESS and 
                    calculated_ij_ops % max(1, len(valid_r_ij_pairs)//10) == 0):
                    print(f"    R_ij({rx_i},{rx_j}): 기존 결과 사용 (건너뜀)")
                # 진행률 추적 업데이트 (시간은 0으로)
                progress_tracker.update_r_ij_progress(0.0)
                continue
            
            block_start_time = time.time()
            h_raw_i = self._load_single_user(area_id, rx_i)
            h_raw_j = self._load_single_user(area_id, rx_j)
            
            # R_ii 계산 시 저장했던 통계 정보 로드 (v5.1: working_dir 기반)
            with np.load(os.path.join(working_dir, "stats_blocks", f'stats_{rx_i}.npz')) as data_i:
                sigma_i = tf.constant(data_i['sigma'])
            with np.load(os.path.join(working_dir, "stats_blocks", f'stats_{rx_j}.npz')) as data_j:
                sigma_j = tf.constant(data_j['sigma'])
            
            R_ij = self._calc_R_ij_block(h_raw_i, h_raw_j, sigma_i, sigma_j)
            np.save(os.path.join(working_dir, "R_ij_blocks", f'R_ij_{rx_i}_{rx_j}.npy'), R_ij.numpy())
            
            # 진행률 추적 업데이트
            block_time = time.time() - block_start_time
            progress_tracker.update_r_ij_progress(block_time)
            
            # 상세 진행률 표시 (일부만)
            if (self.config.ENABLE_DETAILED_PROGRESS and 
                calculated_ij_ops % max(1, len(valid_r_ij_pairs)//10) == 0):
                print(f"    R_ij({rx_i},{rx_j}): {block_time:.2f}s [{calculated_ij_ops}/{len(valid_r_ij_pairs)}]")
                line_tracker.add_lines(1)
            
            # 중간 화면 클리어 및 실시간 진행률 업데이트 (RX 많은 환경 대응)
            if need_block_clear and calculated_ij_ops % self.config.BLOCK_CLEAR_INTERVAL == 0:
                # 실시간 진행률 정보 생성
                progress_summary = progress_tracker.format_progress_summary()
                if self.config.ENABLE_MEMORY_MONITORING:
                    progress_summary += f"\nGPU 메모리 상태 포함"
                    
                if line_tracker.check_and_clear(context_info=progress_summary):
                    if self.config.ENABLE_MEMORY_MONITORING:
                        print_memory_usage()
                        line_tracker.add_lines(1)
        
        # 거리 필터링으로 생략된 쌍들을 위한 zero 행렬 저장 (옵션)
        if (self.config.distance_filtering['enable_filtering'] and 
            self.config.distance_filtering['filtering_mode'] == 'zero' and
            skipped_ij_ops > 0):
            print(f"  거리 필터링으로 생략된 {skipped_ij_ops:,}개 쌍에 대해 zero 행렬 저장 중...")
            zero_count = 0
            for i in range(len(valid_rx_list)):
                for j in range(i + 1, len(valid_rx_list)):
                    rx_i, rx_j = valid_rx_list[i], valid_rx_list[j]
                    if (rx_i, rx_j) not in valid_r_ij_pairs:
                        R_ij_zero = tf.zeros([N_t, N_t], dtype=tf.complex64)
                        np.save(os.path.join(working_dir, "R_ij_blocks", f'R_ij_{rx_i}_{rx_j}.npy'), R_ij_zero.numpy())
                        zero_count += 1
            print(f"    Zero 행렬 저장 완료: {zero_count:,}개")

        r_ij_time = time.time() - stage_start_time

        # 3. 최종 처리 결과 종합 및 통계 출력 (필터링 통계 포함)
        total_processing_time = r_ii_time + r_ij_time
        
        if self.config.ENABLE_DETAILED_PROGRESS:
            print(f"  처리 완료: Area {area_id}")
            print(f"    R_ii 블록: {num_rx}개, {format_time(r_ii_time)}")
            print(f"    R_ij 블록: {calculated_ij_ops}개 계산, {skipped_ij_ops}개 생략, {format_time(r_ij_time)}")
            print(f"    총 처리 시간: {format_time(total_processing_time)}")
            print(f"    평균 블록 처리 속도: {total_processing_time/(num_rx + calculated_ij_ops):.2f}초/블록")
            if len(original_rx_list) != num_rx or calculated_ij_ops < total_pairs:
                print(f"    필터링 결과: RX {len(original_rx_list)}→{num_rx}, R_ij {total_pairs}→{calculated_ij_ops}")
            if self.config.ENABLE_MEMORY_MONITORING:
                print_memory_usage()
        
        # R_ii 보정 결과 요약
        if validation_logs:
            print(f"  [경고 요약] {len(validation_logs)}개의 R_ii 블록에서 음수 고유값 발견:")
            for log in validation_logs:
                print(f"    - RX{log['rx_id']}: Min Eigenvalue = {log['min_eig']:.4e}")
        else:
            print("  [성공] 모든 R_ii 블록 보정이 유효성 검사를 통과했습니다.")

        # 패키징에 필요한 메타데이터 반환 (필터링 정보 포함)
        return {
            'area_id': area_id,
            'rx_ids': valid_rx_list,  # 필터링된 RX 목록
            'original_rx_ids': original_rx_list,  # 원본 RX 목록
            'excluded_rx_ids': excluded_rx_list,  # 제외된 RX 목록
            'antenna_count': N_t,
            'intermediate_save_path': intermediate_save_path,
            'processing_stats': {
                'r_ii_time': r_ii_time,
                'r_ij_time': r_ij_time,
                'total_time': total_processing_time,
                'num_blocks': num_rx + calculated_ij_ops,
                'validation_warnings': len(validation_logs),
                'original_rx_count': len(original_rx_list),
                'filtered_rx_count': num_rx,
                'calculated_r_ij_count': calculated_ij_ops,
                'skipped_r_ij_count': skipped_ij_ops
            },
            'v5_filtering_metadata': self.efficiency_filter.get_filtering_metadata()
        }
    
    def _load_single_user(self, area_id: int, rx_id: int) -> tf.Tensor:
        """단일 사용자 채널 데이터 로딩 (개별 파일 + 청크 파일 지원, 샘플링 적용)"""
        file_info = self.config.rx_file_mapping[(area_id, rx_id)]
        file_path = file_info['file_path']
        file_type = file_info['file_type']
        
        # 파일 타입에 따른 데이터 로딩
        if file_type == 'individual':
            # 개별 파일에서 로딩 (기존 방식)
            h_data = np.load(file_path)
        elif file_type == 'chunk':
            # 청크 파일에서 특정 RX 데이터 로딩
            chunk_key = file_info['chunk_key']  # 'ofdm_ch_rx_{rx_id}'
            with np.load(file_path) as chunk_data:
                if chunk_key in chunk_data:
                    h_data = chunk_data[chunk_key]
                else:
                    raise KeyError(f"청크 파일 {file_path}에서 키 {chunk_key}를 찾을 수 없습니다")
        else:
            raise ValueError(f"지원하지 않는 파일 타입: {file_type}")
        
        # 샘플링: 설정된 비율만큼 추출 (순차 vs 랜덤)
        total_symbols = h_data.shape[0]
        symbols_sampled = int(total_symbols * self.config.sampling_config['sampling_ratio'])
        
        if self.config.sampling_config['random_sampling']:
            # 랜덤 샘플링
            np.random.seed(self.config.sampling_config['seed'])
            indices = np.random.choice(total_symbols, symbols_sampled, replace=False)
            indices = np.sort(indices)  # 시간 순서 유지
            h_data = h_data[indices, :, :, :]
            
            # 검증 모드일 때만 히스토리 저장
            if self.config.execution_mode == 'validation':
                current_seed = self.config.sampling_config['seed']
                trial = current_seed - self.config.validation_config['seed_base']  # trial 번호 역산
                self.sampling_history[(trial, rx_id)] = indices.copy()
        else:
            # 순차 샘플링 (기존 방식)
            h_data = h_data[:symbols_sampled, :, :, :]
        
        # 형태 변환: [symbols_sampled, OFDM_FFT, 1, N_t] → [T, N_t] (T = symbols_sampled * OFDM_FFT)
        h_squeezed = h_data.squeeze(axis=2)  # [symbols_sampled, OFDM_FFT, N_t]
        
        # Flatten 시간-주파수 차원: [symbols_sampled, OFDM_FFT, N_t] → [T, N_t]
        T = h_squeezed.shape[0] * h_squeezed.shape[1]  # symbols_sampled * OFDM_FFT
        N_t = h_squeezed.shape[2]
        h_reshaped = h_squeezed.reshape(T, N_t)  # [T, N_t]
        
        return tf.constant(h_reshaped, dtype=tf.complex64)
        
    def _load_user_pair(self, area_id: int, rx_id_i: int, rx_id_j: int) -> Tuple[tf.Tensor, tf.Tensor]:
        """사용자 쌍 채널 데이터 로딩"""
        h_i = self._load_single_user(area_id, rx_id_i)
        h_j = self._load_single_user(area_id, rx_id_j)
        return h_i, h_j
    
    def _calc_R_ii_block(self, h_raw: tf.Tensor) -> Dict[str, Any]:
        """h_raw로부터 R_ii 블록(사용자 내부 공분산)을 계산하고 보정합니다."""
        
        # 1. 평균 채널 및 중심화된 채널 계산
        ch_mean = tf.reduce_mean(h_raw, axis=0)
        centered_h = h_raw - ch_mean
        
        # 2. 정규화되지 않은 공분산 계산
        R_unnormalized = tf.linalg.einsum('ij,ik->jk', tf.math.conj(centered_h), centered_h) / h_raw.shape[0]

        # 3. 정규화 스케일(sigma) 계산 (v2.2: 타입 일관성 확보)
        sigma_sq = tf.reduce_mean(tf.math.real(tf.linalg.diag_part(R_unnormalized)))
        sigma = tf.sqrt(sigma_sq)
        
        # 4. R_ii 정규화 (v2.2: 타입 일관성 확보)
        R_ii_normalized = R_unnormalized / tf.cast(sigma_sq + 1e-9, tf.complex64)

        # 5. CCM 보정 적용
        if self.config.ccm_correction['enable_correction']:
            correction_result = self._apply_ccm_correction(R_ii_normalized)
            R_ii_final = correction_result['corrected_matrix']
            final_min_eigenvalue = correction_result['final_min_eigenvalue']
        else:
            R_ii_final = R_ii_normalized
            # 보정을 안 할 경우, 유효성 검증을 위해 직접 계산
            eigenvals = tf.linalg.eigvalsh(R_ii_final)
            final_min_eigenvalue = tf.reduce_min(eigenvals)

        # 6. 부록 M: 채널 통계 계산 (v2.2: 타입 일관성 확보)
        mean_pwr_ue = tf.reduce_sum(tf.square(tf.abs(ch_mean)))
        k_factor = tf.math.real(mean_pwr_ue) / (sigma_sq + 1e-9)
        pathloss = tf.reduce_mean(tf.reduce_sum(tf.square(tf.abs(h_raw)), axis=1))
        
        channel_stats = {
            'ch_mean': ch_mean,
            'pathloss': pathloss,
            'k_factor': k_factor,
            'sigma': sigma
        }

        # 7. 최종 결과 반환
        return {
            'corrected_matrix': R_ii_final,
            'final_min_eigenvalue': final_min_eigenvalue,
            'channel_stats': channel_stats
        }
    
    def _calc_R_ij_block(self, h_raw_i: tf.Tensor, h_raw_j: tf.Tensor, sigma_i: tf.Tensor, sigma_j: tf.Tensor) -> tf.Tensor:
        """두 사용자(h_raw_i, h_raw_j) 간의 R_ij 블록(상호 공분산)을 계산합니다."""
        
        # 1. 각 사용자별 시간 평균 제거
        h_i_mean = tf.reduce_mean(h_raw_i, axis=0, keepdims=True)
        h_j_mean = tf.reduce_mean(h_raw_j, axis=0, keepdims=True)
        h_i_centered = h_raw_i - h_i_mean
        h_j_centered = h_raw_j - h_j_mean
        
        # 2. 교차공분산 계산
        R_ij_unscaled = tf.linalg.einsum('ij,ik->jk', tf.math.conj(h_i_centered), h_j_centered) / h_raw_i.shape[0]
        
        # 3. 스케일링 적용
        sigma_i_complex = tf.cast(sigma_i, tf.complex64)
        sigma_j_complex = tf.cast(sigma_j, tf.complex64)
        R_ij_scaled = R_ij_unscaled / (sigma_i_complex * tf.math.conj(sigma_j_complex))
        
        return R_ij_scaled
    
    def _apply_ccm_correction(self, R_in: tf.Tensor) -> dict:  # [구현 스텁 v2.1]
        """CCM 보정 (대각 스케일링, 동적 파라미터, 인덱스 추적)
        
        처리 흐름: 사전 정규화 → 반복 보정 (고유값 보정 → 재구성 → Unit Diagonal → 순서 복원) → 사후 복원
        - 설정값: config.ccm_correction (max_iterations, eps_min_stages, tolerance_stages)
        - 데이터 흐름: R_in → R_current → ... → R_corrected_normalized → R_final_restored
        """
        # --- 1. 대각 스케일링 기반 정규화 (Pre-Normalization) ---
        safe_diag = tf.maximum(tf.math.real(tf.linalg.diag_part(R_in)), 1e-9)
        d_sqrt = tf.sqrt(safe_diag)
        D_scale_inv = tf.cast(tf.linalg.diag(1.0 / d_sqrt), dtype=tf.complex64)
        D_scale = tf.cast(tf.linalg.diag(d_sqrt), dtype=tf.complex64)
        
        R_current = D_scale_inv @ R_in @ D_scale_inv

        # --- 2. 반복 보정 루프 ---
        ccm_config = self.config.ccm_correction
        max_iter = ccm_config['max_iterations']
        
        map_sorted_to_original = tf.range(tf.shape(R_current)[0])
        termination_reason = self.REASON_MAX_ITER

        for iteration in range(max_iter):
            # 동적 파라미터 업데이트
            if iteration in ccm_config['eps_min_stages']:
                eps_min = ccm_config['eps_min_stages'][iteration]
            if iteration in ccm_config['tolerance_stages']:
                tolerance = ccm_config['tolerance_stages'][iteration]

            R_prev_restored = R_current
            
            # 고유값 분해 및 오름차순 정렬
            eigenvals, eigenvecs = tf.linalg.eigh(R_current)
            eigenvals_real = tf.math.real(eigenvals)
            sort_indices = tf.argsort(eigenvals_real, direction='ASCENDING')
            eigenvals_sorted = tf.gather(eigenvals_real, sort_indices)
            eigenvecs_sorted = tf.gather(eigenvecs, sort_indices, axis=1)
            
            map_sorted_to_original = tf.gather(map_sorted_to_original, sort_indices)

            # 조기 종료 조건 1: 최소 고유값 안정
            if tf.reduce_min(eigenvals_sorted) >= eps_min and iteration > 10:
                termination_reason = self.REASON_EIG_STABLE
                break
            
            # 고유값 보정 (선형 보간)
            positive_eigenvals = eigenvals_sorted[eigenvals_sorted >= eps_min]
            val_pos = tf.reduce_min(positive_eigenvals) if tf.shape(positive_eigenvals)[0] > 0 else eps_min
            num_neg_corrected = tf.reduce_sum(tf.cast(eigenvals_sorted < eps_min, tf.float32))
            lin_ramp = tf.range(1.0, num_neg_corrected + 1.0) / (num_neg_corrected + 1.0)
            eigvals_neg_corrected = (1.0 - lin_ramp) * eps_min + lin_ramp * val_pos
            eigenvals_corrected = tf.concat([eigvals_neg_corrected, positive_eigenvals], axis=0)
            
            # 행렬 재구성 및 Unit Diagonal 강제
            R_positive = eigenvecs_sorted @ tf.linalg.diag(tf.cast(eigenvals_corrected, tf.complex64)) @ tf.linalg.adjoint(eigenvecs_sorted)
            R_unit_diag = tf.linalg.set_diag(R_positive, tf.ones(tf.shape(R_positive)[0], dtype=tf.complex64))
            R_corrected_sorted = (R_unit_diag + tf.linalg.adjoint(R_unit_diag)) / 2.0

            # 원본 순서로 행렬 복원 (안정적 수렴 체크를 위함)
            map_original_to_sorted = tf.argsort(map_sorted_to_original)
            R_restored_rows = tf.gather(R_corrected_sorted, map_original_to_sorted, axis=0)
            R_corrected_restored = tf.gather(R_restored_rows, map_original_to_sorted, axis=1)
            
            R_current = R_corrected_restored

            # 조기 종료 조건 2: 변화량 기반 수렴
            diff_norm = tf.math.real(tf.norm(R_current - R_prev_restored))
            if diff_norm < tolerance and iteration > 10:
                termination_reason = self.REASON_CONVERGED
                break
        
        R_corrected_normalized = R_current
        
        # --- 3. 사후 복원 (Post-Restoration) ---
        R_final_restored = D_scale @ R_corrected_normalized @ D_scale
        
        final_min_eig = tf.reduce_min(tf.math.real(tf.linalg.eigvalsh(R_final_restored)))
        
        return {
            'corrected_matrix': R_final_restored,
            'final_min_eigenvalue': float(final_min_eig.numpy()),
            'termination_reason': termination_reason
        }
    
    def process_multiple_areas(self, area_ids: List[int], progress_callback=None) -> Dict[str, Any]:
        """CCM 다중 Area 통합 처리 (v5.1: working_dir 기반, 영구 저장)"""
        area_results = {}
        
        for idx, area_id in enumerate(area_ids):
            if progress_callback:
                progress_callback(f"Area {area_id} 처리 시작", idx, len(area_ids))
            
            area_results[area_id] = self.process_area_ccm(area_id, progress_callback)
            
            if progress_callback:
                progress_callback(f"Area {area_id} 처리 완료", idx + 1, len(area_ids))
        
        return {
            'processed_areas': area_ids,
            'area_results': area_results
        }
    
    def _plot_eigenvalues(self, R_ii_matrix: tf.Tensor, rx_id: int, trial: int, area_id: int):
        """R_ii 행렬의 고유값 분포를 로그 스케일로 시각화"""
        # 고유값 계산 및 로그 변환
        eigenvals = tf.linalg.eigvals(R_ii_matrix)
        eigenvals_real = tf.math.real(eigenvals).numpy()
        
        # 정렬된 고유값으로 플롯 생성
        sorted_log_eigenvals = sorted(np.log10(eigenvals_real))
        
        plt.figure(figsize=(10, 2.5))
        
        # 연속적인 선 그리기 (NaN 안전 처리)
        valid_indices = ~np.isnan(sorted_log_eigenvals)
        if np.any(valid_indices):
            plt.plot(np.where(valid_indices)[0], np.array(sorted_log_eigenvals)[valid_indices], 
                    'b-', linewidth=1.5)  # 유효한 값만 플롯
        else:
            plt.plot(sorted_log_eigenvals, 'b-', linewidth=1.5)  # 모든 값이 유효한 경우
        
        # eps_min 기준선 추가
        eps_min_stages = self.config.ccm_correction['eps_min_stages']
        eps_min = list(eps_min_stages.values())[0] if eps_min_stages else 1e-4
        plt.axhline(y=np.log10(eps_min), color='r', linestyle='--', alpha=0.7, 
                   label=f'eps_min={eps_min:.0e}')
        
        # 라벨 및 제목 설정
        plt.title(f'Area{area_id} RX{rx_id} Trial{trial} - R_ii Eigenvalue Distribution (log10)')
        plt.xlabel('Eigenvalue Index (sorted)')
        plt.ylabel('log10(Eigenvalue)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 표시 후 메모리 해제
        plt.show()
        plt.close()
    
    def validate_rand_sampled_ccm(self) -> Dict[Tuple[int, int], List[float]]:
        """모든 Area에 대해 랜덤 샘플링 CCM 계산 검증 (NMSE 기반, 검증 전용, 저장 안함)"""
        # 1. P3_Config에서 동적으로 추출
        config = self.config.validation_config
        num_trials = config['num_trials']
        seed_base = config['seed_base']
        area_ids = list(self.config.area_rx_mapping.keys())  # 동적 Area 목록
        
        validation_results = {}  # {(rx_i, rx_j): diff_norms_list}
        
        for area_id in area_ids:
            area_info = self.config.get_area_info(area_id)
            rx_ids = area_info['rx_ids']
            
            print(f"=== 랜덤 샘플링 CCM 계산 검증: Area {area_id}, {num_trials} trials ===")
            
            # 2. sigma 저장용 딕셔너리 (스칼라, 전체 보관)
            all_sigmas = {}  # {trial: {rx_id: sigma_scalar}}
            
            # 3. R_ii 블록별 처리 (대각 블록, sigma도 같이 수집)
            for rx_id in rx_ids:
                R_ii_trials = []
                
                for trial in range(num_trials):
                    if trial not in all_sigmas:
                        all_sigmas[trial] = {}
                    
                    # trial별로 시드 변경
                    self.config.sampling_config['seed'] = seed_base + trial
                    
                    # 채널 로딩 및 R_ii 계산
                    h_raw = self._load_single_user(area_id, rx_id)
                    result_ii = self._calc_R_ii_block(h_raw)
                    R_ii_trials.append(result_ii['corrected_matrix'])
                    
                    # sigma 저장 (스칼라로 변환)
                    sigma_tensor = result_ii['channel_stats']['sigma']
                    all_sigmas[trial][rx_id] = float(sigma_tensor.numpy())
                
                # NMSE 계산 및 출력
                diff_norms = []
                for i in range(1, num_trials):
                    diff = R_ii_trials[i] - R_ii_trials[0]
                    diff_norm = tf.norm(diff, ord='fro')
                    reference_norm = tf.norm(R_ii_trials[0], ord='fro')
                    nmse = (diff_norm / reference_norm) ** 2
                    nmse_db = 10 * tf.math.log(nmse) / tf.math.log(10.0)
                    diff_norms.append(float(nmse_db.numpy()))
                
                avg_nmse_db = np.mean(diff_norms)
                std_nmse_db = np.std(diff_norms)
                print(f"  R_ii({rx_id}): NMSE = {avg_nmse_db:.2f} ± {std_nmse_db:.2f} dB")
                validation_results[(rx_id, rx_id)] = diff_norms
            
            # 4. R_ij 블록별 처리 (비대각 블록)
            for i, rx_i in enumerate(rx_ids):
                for j in range(i + 1, len(rx_ids)):
                    rx_j = rx_ids[j]
                    R_ij_trials = []
                    
                    for trial in range(num_trials):
                        # trial별로 시드 변경
                        self.config.sampling_config['seed'] = seed_base + trial
                        
                        # 채널 로딩
                        h_raw_i = self._load_single_user(area_id, rx_i)
                        h_raw_j = self._load_single_user(area_id, rx_j)
                        
                        # sigma 로딩 (이미 계산됨)
                        sigma_i = tf.constant(all_sigmas[trial][rx_i])
                        sigma_j = tf.constant(all_sigmas[trial][rx_j])
                        
                        # R_ij 계산
                        R_ij = self._calc_R_ij_block(h_raw_i, h_raw_j, sigma_i, sigma_j)
                        R_ij_trials.append(R_ij)
                    
                    # NMSE 계산 및 출력
                    diff_norms = []
                    for k in range(1, num_trials):
                        diff = R_ij_trials[k] - R_ij_trials[0]
                        diff_norm = tf.norm(diff, ord='fro')
                        reference_norm = tf.norm(R_ij_trials[0], ord='fro')
                        nmse = (diff_norm / reference_norm) ** 2
                        nmse_db = 10 * tf.math.log(nmse) / tf.math.log(10.0)
                        diff_norms.append(float(nmse_db.numpy()))
                    
                    avg_nmse_db = np.mean(diff_norms)
                    std_nmse_db = np.std(diff_norms)
                    print(f"  R_ij({rx_i},{rx_j}): NMSE = {avg_nmse_db:.2f} ± {std_nmse_db:.2f} dB")
                    validation_results[(rx_i, rx_j)] = diff_norms
            
            print(f"Area {area_id} 검증 완료\n")
        
        return validation_results


class CCM_BlockManager:
    """CCM 블록 관리, 저장, 재조합 클래스 (효율적 블록 관리)
    
    역할:
    - 임시 블록 파일들을 하나의 .npz 파일로 패키징
    - R_ii 중간 저장 기능
    - 필터링 메타데이터 통합
    """
    
    def __init__(self, config: P3_Config):
        self.config = config
    
    def save_r_ii_intermediate(self, area_id: int, rx_ids: List[int], antenna_count: int, working_dir: str) -> str:
        """R_ii 블록만 중간 저장합니다. (v2.2)"""
        print(f"\n  [중간저장] Area {area_id}: R_ii 블록 저장 중...")
        
        data_to_save = {
            'metadata': np.array([
                f"area_id={area_id}",
                f"rx_ids={','.join(map(str, rx_ids))}",
                f"antenna_count={antenna_count}",
                f"status=r_ii_completed",
                f"storage_format=blocks_upper_triangle"
            ])
        }
        
        # R_ii 블록들 로딩 및 저장 (v5.1: working_dir 기반)
        for rx_id in rx_ids:
            r_ii_file = os.path.join(working_dir, "R_ii_blocks", f'R_ii_{rx_id}.npy')
            if os.path.exists(r_ii_file):
                data_to_save[f'R_ii_{rx_id}'] = np.load(r_ii_file)
        
        # 저장 경로 생성
        save_path = self.config.get_save_path(area_id, is_all_rxs=True)
        intermediate_path = save_path.replace('.npz', '_R_ii_blocks.npz')
        
        np.savez_compressed(intermediate_path, **data_to_save)
        print(f"    R_ii 중간 저장 완료: {intermediate_path}")
        
        return intermediate_path
    
    def save_area_results(self, area_info: Dict[str, Any], working_dir: str):
        """임시 디렉터리의 블록 파일들을 하나의 .npz 파일로 패키징합니다. (v5.0: 필터링 메타데이터 포함)"""
        area_id = area_info['area_id']
        rx_ids = area_info['rx_ids']
        num_rx = len(rx_ids)
        packaging_start_time = time.time()
        
        print(f"\n  [패키징] Area {area_id}: 임시 블록 파일들을 .npz로 통합 중...")
        
        # v5.0: 필터링 메타데이터 준비
        metadata_list = [
            f"area_id={area_id}",
            f"rx_ids={','.join(map(str, rx_ids))}",
            f"antenna_count={area_info['antenna_count']}",
            f"storage_format=blocks_upper_triangle"
        ]
        
        # v5.0: 필터링 정보 추가
        if 'original_rx_ids' in area_info:
            metadata_list.extend([
                f"original_rx_count={len(area_info['original_rx_ids'])}",
                f"filtered_rx_count={len(rx_ids)}",
                f"excluded_rx_ids={','.join(map(str, area_info.get('excluded_rx_ids', [])))}",
                f"v5_data_efficiency=enabled"
            ])
        
        data_to_save = {
            'metadata': np.array(metadata_list)
        }
        
        # R_ii 블록들 로딩 (v5.1: working_dir 기반)
        for rx_id in rx_ids:
            r_ii_file = os.path.join(working_dir, "R_ii_blocks", f'R_ii_{rx_id}.npy')
            if os.path.exists(r_ii_file):
                data_to_save[f'R_ii_{rx_id}'] = np.load(r_ii_file)
        
        # R_ij 블록들 로딩 (상삼각행렬 방식, v5.1: working_dir 기반)
        for i, rx_i in enumerate(rx_ids):
            for j in range(i + 1, len(rx_ids)):
                rx_j = rx_ids[j]
                r_ij_file = os.path.join(working_dir, "R_ij_blocks", f'R_ij_{rx_i}_{rx_j}.npy')
                if os.path.exists(r_ij_file):
                    data_to_save[f'R_ij_{rx_i}_{rx_j}'] = np.load(r_ij_file)
        
        # v5.0: 필터링 메타데이터 추가
        if 'v5_filtering_metadata' in area_info:
            data_to_save['v5_filtering_stats'] = str(area_info['v5_filtering_metadata'])
        
        # 최종 .npz 파일로 저장
        save_path = self.config.get_save_path(area_id, is_all_rxs=True)
        np.savez_compressed(save_path, **data_to_save)
        
        packaging_time = time.time() - packaging_start_time
        
        # 파일 크기 확인 (안전장치)
        try:
            file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
            print(f"    패키징 완료: {save_path}")
            print(f"    파일 크기: {file_size_mb:.1f}MB, 패키징 시간: {packaging_time:.2f}초")
        except OSError:
            print(f"    패키징 완료: {save_path}")
            print(f"    패키징 시간: {packaging_time:.2f}초")
        print(f"    블록 수: {len([k for k in data_to_save.keys() if k.startswith('R_')])}개")
        
        # 처리 통계가 있으면 함께 출력
        if 'processing_stats' in area_info and self.config.ENABLE_DETAILED_PROGRESS:
            stats = area_info['processing_stats']
            print(f"    전체 처리 시간: {format_time(stats['total_time'])}")
            if stats['validation_warnings'] > 0:
                print(f"    경고: {stats['validation_warnings']}개 R_ii 블록에서 음수 고유값")
        
        # 중간 파일 정리 (v2.2)
        if (self.config.CLEANUP_INTERMEDIATE_FILE and 
            'intermediate_save_path' in area_info and 
            area_info['intermediate_save_path'] and 
            os.path.exists(area_info['intermediate_save_path'])):
            try:
                os.remove(area_info['intermediate_save_path'])
                print(f"    중간 파일 정리 완료: {os.path.basename(area_info['intermediate_save_path'])}")
            except Exception as e:
                print(f"    Warning: 중간 파일 정리 실패: {e}")
    
    def load_area_results(self, area_id: int):
        """Area 결과 파일 로드 (v5.0: 블록 저장 방식 지원)"""
        # P3_Config 경로에서 블록별 결과 복원 (우선순위: ALL_RXs -> 일반)
        save_path_all = self.config.get_save_path(area_id, is_all_rxs=True)
        save_path_partial = self.config.get_save_path(area_id, is_all_rxs=False)
        
        # ALL_RXs 파일 우선 확인
        if os.path.exists(save_path_all):
            save_path = save_path_all
        elif os.path.exists(save_path_partial):
            save_path = save_path_partial
        else:
            raise FileNotFoundError(f"Area {area_id} CCM 파일이 존재하지 않습니다: {save_path_all} 또는 {save_path_partial}")
        
        # .npz 파일 로딩 및 블록 재조합
        with np.load(save_path) as data:
            metadata = data['metadata']
            
            # 메타데이터에서 RX 정보 추출
            rx_ids_line = next((line for line in metadata if line.startswith('rx_ids=')), None)
            if rx_ids_line is None:
                raise ValueError(f"rx_ids 정보를 찾을 수 없습니다: {save_path}")
            
            rx_ids = [int(x) for x in rx_ids_line.split('=')[1].split(',')]
            
            # R_ii 블록들 로딩
            R_blocks = {}
            for rx_id in rx_ids:
                r_ii_key = f'R_ii_{rx_id}'
                if r_ii_key in data:
                    R_blocks[(rx_id, rx_id)] = data[r_ii_key]
            
            # R_ij 블록들 로딩
            for i, rx_i in enumerate(rx_ids):
                for j in range(i + 1, len(rx_ids)):
                    rx_j = rx_ids[j]
                    r_ij_key = f'R_ij_{rx_i}_{rx_j}'
                    if r_ij_key in data:
                        R_blocks[(rx_i, rx_j)] = data[r_ij_key]
                        R_blocks[(rx_j, rx_i)] = data[r_ij_key].T.conj()  # Hermitian 대칭
        
        return R_blocks
    
    def get_available_areas(self) -> List[int]:
        """처리 완료된 Area 목록 반환"""
        available_areas = []
        for area_id in self.config.area_rx_mapping.keys():
            save_path = self.config.get_save_path(area_id, is_all_rxs=True)
            if os.path.exists(save_path):
                available_areas.append(area_id)
        return sorted(available_areas)
    
    def validate_ccm_shape(self, ccm_matrix: tf.Tensor, area_id: int, user_subset: Optional[List[int]] = None) -> bool:
        """CCM 행렬 크기 검증"""
        area_info = self.config.get_area_info(area_id)
        
        # 사용자 서브셋이 지정되지 않으면 전체 RX 사용
        if user_subset is None:
            K = len(area_info['rx_ids'])
        else:
            K = len(user_subset)
        
        N_t = area_info['antenna_count']
        expected_size = K * N_t
        
        actual_shape = tf.shape(ccm_matrix)
        return (actual_shape[0] == expected_size and 
                actual_shape[1] == expected_size)


# ===============================
# P3 v5 유틸리티 함수들
# ===============================

def format_time(seconds: float) -> str:
    """시간 포맷팅 유틸리티"""
    if seconds < 60:
        return f"{seconds:.1f}초"
    elif seconds < 3600:
        return f"{int(seconds//60)}분 {int(seconds%60)}초"
    else:
        return f"{int(seconds//3600)}시간 {int((seconds%3600)//60)}분"

def format_file_size(size_bytes):
    """파일 크기를 읽기 쉬운 형식으로 포맷팅"""
    if size_bytes < 1024**2:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes / (1024**2):.1f} MB"
    else:
        return f"{size_bytes / (1024**3):.1f} GB"

def print_memory_usage():
    """GPU 메모리 사용량 출력"""
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            details = tf.config.experimental.get_memory_info(gpus[0])
            used_gb = details['current'] / (1024**3)
            peak_gb = details['peak'] / (1024**3)
            print(f"GPU 메모리: 현재 {used_gb:.1f}GB, 최대 {peak_gb:.1f}GB")
        else:
            print("GPU 메모리 정보 없음 (CPU 모드)")
    except Exception as e:
        print(f"GPU 메모리 확인 실패: {e}")

def clear_output_safe():
    """안전한 화면 클리어 (Jupyter/콘솔 환경 자동 감지)"""
    if JUPYTER_AVAILABLE:
        clear_output(wait=True)
    else:
        os.system('clear' if os.name == 'posix' else 'cls')

def check_ccm_validity(matrix: np.ndarray, tol: float = 1e-8) -> Tuple[bool, List[str]]:
    """
    CCM의 유효 조건을 검사합니다.
    1. Hermitian property
    2. Positive semi-definiteness
    """
    reasons = []
    # 1. Hermitian 검사
    if not np.allclose(matrix, matrix.T.conj(), atol=tol):
        reasons.append("Not Hermitian")
    
    # 2. Positive Semi-Definite 검사
    try:
        eigenvalues = np.linalg.eigvalsh(matrix)
        if np.min(eigenvalues) < -tol:
            reasons.append(f"Not positive semi-definite (min eigenvalue: {np.min(eigenvalues):.4e})")
    except np.linalg.LinAlgError:
        reasons.append("Eigenvalue computation failed")

    is_valid = len(reasons) == 0
    return is_valid, reasons

class OutputLineTracker:
    """출력 라인 추적 및 화면 클리어 관리"""
    
    def __init__(self, config: P3_Config):
        self.config = config
        self.current_lines = 0
        
    def add_lines(self, count: int = 1):
        """출력 라인 수 추가"""
        self.current_lines += count
        
    def check_and_clear(self, context_info: str = "") -> bool:
        """필요 시 화면 클리어 및 컨텍스트 정보 재출력"""
        if (self.config.ENABLE_SCREEN_CLEAR and 
            self.current_lines >= self.config.MAX_OUTPUT_LINES):
            self._clear_and_reprint(context_info)
            return True
        return False
        
    def _clear_and_reprint(self, context_info: str):
        """화면 클리어 및 컨텍스트 재출력"""
        if JUPYTER_AVAILABLE:
            clear_output(wait=True)
        else:
            os.system('clear' if os.name == 'posix' else 'cls')
        
        if context_info:
            print(context_info)
            print("=" * 50)
        
        self.current_lines = context_info.count('\n') + 2 if context_info else 0

class BlockProgressTracker:
    """블록 처리 진행률 추적"""
    
    def __init__(self, area_id: int, num_rx: int, valid_rx_list: List[int] = None):
        self.area_id = area_id
        self.num_rx = num_rx
        self.valid_rx_list = valid_rx_list or list(range(num_rx))  # 실제 RX ID 목록
        self.r_ii_progress = {'completed': 0, 'total_time': 0.0}
        self.r_ij_progress = {'completed': 0, 'total_time': 0.0}
        self.r_ij_total_pairs = None  # 거리 필터링 적용된 실제 계산할 쌍 수
        self.phase = 'init'
        
    def start_r_ii_phase(self):
        """R_ii 처리 단계 시작"""
        self.phase = 'r_ii'
        
    def start_r_ij_phase(self):
        """R_ij 처리 단계 시작"""
        self.phase = 'r_ij'
        
    def set_r_ij_total_pairs(self, total_pairs: int):
        """거리 필터링 적용된 실제 R_ij 쌍 수 설정"""
        self.r_ij_total_pairs = total_pairs
        
    def update_r_ii_progress(self, block_time: float):
        """R_ii 진행률 업데이트"""
        self.r_ii_progress['completed'] += 1
        self.r_ii_progress['total_time'] += block_time
        
    def update_r_ij_progress(self, block_time: float):
        """R_ij 진행률 업데이트"""
        self.r_ij_progress['completed'] += 1
        self.r_ij_progress['total_time'] += block_time
        
    def get_progress_info(self) -> Dict[str, Any]:
        """상세 진행률 정보 반환"""
        # 거리 필터링 적용된 실제 R_ij 쌍 수 사용
        total_r_ij = self.r_ij_total_pairs if self.r_ij_total_pairs is not None else (self.num_rx * (self.num_rx - 1) // 2)
        
        r_ii_progress_ratio = self.r_ii_progress['completed'] / self.num_rx if self.num_rx > 0 else 0.0
        r_ij_progress_ratio = self.r_ij_progress['completed'] / total_r_ij if total_r_ij > 0 else 0.0
        
        # 현재 처리 중인 RX ID 정보 추가
        current_rx_id = None
        if self.r_ii_progress['completed'] < len(self.valid_rx_list):
            current_rx_id = self.valid_rx_list[self.r_ii_progress['completed']]
        
        return {
            'area_id': self.area_id,
            'phase': self.phase,
            'r_ii': {
                'completed': self.r_ii_progress['completed'],
                'total': self.num_rx,
                'progress_ratio': r_ii_progress_ratio,
                'total_time': self.r_ii_progress['total_time'],
                'avg_time_per_block': self.r_ii_progress['total_time'] / max(1, self.r_ii_progress['completed']),
                'estimated_remaining': self._estimate_r_ii_remaining(),
                'current_rx_id': current_rx_id
            },
            'r_ij': {
                'completed': self.r_ij_progress['completed'],
                'total': total_r_ij,
                'progress_ratio': r_ij_progress_ratio,
                'total_time': self.r_ij_progress['total_time'],
                'avg_time_per_block': self.r_ij_progress['total_time'] / max(1, self.r_ij_progress['completed']),
                'estimated_remaining': self._estimate_r_ij_remaining()
            }
        }
    
    def _estimate_r_ii_remaining(self) -> float:
        """R_ii 남은 시간 추정"""
        if self.r_ii_progress['completed'] == 0:
            return 0.0
        
        avg_time = self.r_ii_progress['total_time'] / self.r_ii_progress['completed']
        remaining_blocks = self.num_rx - self.r_ii_progress['completed']
        return avg_time * remaining_blocks
    
    def _estimate_r_ij_remaining(self) -> float:
        """R_ij 남은 시간 추정"""
        if self.r_ij_progress['completed'] == 0:
            return 0.0
        
        total_r_ij = self.num_rx * (self.num_rx - 1) // 2
        avg_time = self.r_ij_progress['total_time'] / self.r_ij_progress['completed']
        remaining_blocks = total_r_ij - self.r_ij_progress['completed']
        return avg_time * remaining_blocks
    
    def format_progress_summary(self) -> str:
        """진행률 요약 문자열 생성"""
        progress_info = self.get_progress_info()
        
        summary = f"=== Area {self.area_id} 진행률 ===\n"
        
        # R_ii 진행률 (실제 RX ID 정보 포함)
        r_ii_info = progress_info['r_ii']
        summary += f"R_ii: {r_ii_info['completed']}/{r_ii_info['total']} "
        summary += f"({r_ii_info['progress_ratio']*100:.1f}%) "
        summary += f"({format_time(r_ii_info['total_time'])})"
        if r_ii_info['current_rx_id'] is not None:
            summary += f" | 현재: RX{r_ii_info['current_rx_id']}"
        if r_ii_info['estimated_remaining'] > 0:
            summary += f" | 남은 시간: {format_time(r_ii_info['estimated_remaining'])}"
        summary += "\n"
        
        # R_ij 진행률
        r_ij_info = progress_info['r_ij']
        summary += f"R_ij: {r_ij_info['completed']}/{r_ij_info['total']} "
        summary += f"({r_ij_info['progress_ratio']*100:.1f}%) "
        summary += f"({format_time(r_ij_info['total_time'])})"
        if r_ij_info['estimated_remaining'] > 0:
            summary += f" | 남은 시간: {format_time(r_ij_info['estimated_remaining'])}"
        
        return summary


# ===============================
# P3 v5 보정 모드 함수
# ===============================

def run_correction_mode(ccm_dir: str, config: P3_Config):
    """
    ccm_dir의 R_ii 행렬을 검사하고, 유효하지 않으면 원본 데이터로
    재계산 및 보정을 거쳐 원본 ccm 파일을 덮어씁니다. (v5.0: 필터링 지원)
    """
    correction_start_time = time.time()
    print(f"--- P3 CCM 블록 보정 모드 시작 ---")
    print(f"대상 디렉터리: {ccm_dir}")

    # 재계산을 위해 CCM_BlockEngine 인스턴스 생성
    engine = CCM_BlockEngine(config)
    
    # CCM 블록 파일 목록 찾기
    ccm_files = [f for f in os.listdir(ccm_dir) if f.endswith('_ALL_RXs.npz')]
    if not ccm_files:
        print("보정할 '_ALL_RXs.npz' 파일이 없습니다.")
        return

    corrected_file_count = 0
    total_corrected_blocks = 0
    total_files = len(ccm_files)
    file_width = len(str(total_files))
    
    print(f"검사 대상 파일: {total_files}개")
    if config.ENABLE_MEMORY_MONITORING:
        print("초기 메모리 상태:")
        print_memory_usage()

    for i, ccm_filename in enumerate(ccm_files):
        # 파일명에서 area_id 파싱
        match = re.search(r"Area(\d+)", ccm_filename)
        if not match:
            print(f"Could not parse area_id from '{ccm_filename}'. Skipping.")
            continue
        area_id = int(match.group(1))

        ccm_filepath = os.path.join(ccm_dir, ccm_filename)
        file_start_time = time.time()
        print(f"\n[{i+1:>{file_width}}/{total_files}] Area {area_id} 파일 검사: {ccm_filename}")
        
        try:
            data = np.load(ccm_filepath)
            data_dict = {key: data[key] for key in data.files}
            file_size = os.path.getsize(ccm_filepath)
            print(f"  파일 크기: {format_file_size(file_size)}")
        except Exception as e:
            print(f"  오류: 파일 로딩 실패 - {e}")
            continue

        # 메타데이터에서 rx_ids 파싱
        metadata = data_dict.get('metadata', [])
        rx_ids_str = next((s for s in metadata if s.startswith('rx_ids=')), None)
        if not rx_ids_str:
            print(f"Could not find 'rx_ids' in metadata for '{ccm_filename}'. Skipping.")
            continue
        rx_ids = [int(rid) for rid in rx_ids_str.split('=')[1].split(',')]

        file_was_modified = False
        corrected_blocks_in_file = 0
        num_blocks_in_file = len(rx_ids)

        for j, rx_id in enumerate(rx_ids):
            r_ii_key = f'R_ii_{rx_id}'
            print(f"  [{j+1}/{num_blocks_in_file}] Checking R_ii for RX {rx_id}... ", end='', flush=True)

            if r_ii_key not in data_dict:
                print("SKIPPED (not found in file)")
                continue

            R_ii_original = data_dict[r_ii_key]
            is_valid, reasons = check_ccm_validity(R_ii_original)

            if is_valid:
                print("OK (Valid)")
            else:
                print(f"FAILED ({', '.join(reasons)})")
                file_was_modified = True
                corrected_blocks_in_file += 1
                
                # 재계산
                print(f"    재계산 중... ", end='', flush=True)
                try:
                    h_raw = engine._load_single_user(area_id, rx_id)
                    result_ii = engine._calc_R_ii_block(h_raw)
                    R_ii_corrected = result_ii['corrected_matrix'].numpy()
                    
                    # 검증
                    is_corrected_valid, _ = check_ccm_validity(R_ii_corrected)
                    if is_corrected_valid:
                        data_dict[r_ii_key] = R_ii_corrected
                        print("OK (Corrected)")
                    else:
                        print("FAILED (Correction failed)")
                except Exception as e:
                    print(f"ERROR ({e})")

        if file_was_modified:
            print(f"  파일 업데이트 중 ({corrected_blocks_in_file}개 블록 수정)... ", end='', flush=True)
            try:
                np.savez_compressed(ccm_filepath, **data_dict)
                print("OK")
                corrected_file_count += 1
                total_corrected_blocks += corrected_blocks_in_file
            except Exception as e:
                print(f"ERROR: {e}")
        
        file_time = time.time() - file_start_time
        print(f"  처리 시간: {format_time(file_time)}")

    print(f"\n--- P3 CCM 블록 보정 모드 완료 ---")
    print(f"수정된 파일: {corrected_file_count}/{total_files}개")
    print(f"수정된 블록: {total_corrected_blocks}개")
    print(f"전체 소요 시간: {format_time(time.time() - correction_start_time)}")


def main_p3_v5_pipeline(config: P3_Config):
    """P3 메인 파이프라인 (v3 호환 스타일)"""
    area_list = list(config.area_rx_mapping.keys())
    total_areas = len(area_list)
    
    # 동적 자리수 맞춤을 위한 너비 계산
    area_width = len(str(max(area_list))) if area_list else 1
    
    print("=== P3 CCM 블록 처리 파이프라인 ===")
    print(f"감지된 Area: {area_list} (총 {total_areas}개)")
    
    # 전체 RX 통계 계산
    total_rx_count = sum(len(config.area_rx_mapping[area_id]) for area_id in area_list)
    print(f"총 처리 예정 RX: {total_rx_count}개")
    
    if config.ENABLE_MEMORY_MONITORING:
        print("초기 메모리 상태:")
        print_memory_usage()
    
    # v5: 실행 모드에 따른 명시적 분기
    if config.execution_mode == 'correct':
        print("\n=== P3 CCM 블록 [보정 모드] ===")
        run_correction_mode(config.ccm_save_dir, config)
        return None

    elif config.execution_mode == 'validation':
        print("\n=== P3 랜덤 샘플링 CCM [검증 모드] ===")
        engine = CCM_BlockEngine(config)
        validation_start_time = time.time()
        validation_results = engine.validate_rand_sampled_ccm()
        validation_time = time.time() - validation_start_time
        
        print(f"\n검증 완료: {len(validation_results)}개 블록 분석 완료")
        print(f"검증 소요 시간: {format_time(validation_time)}")
        print("검증 모드에서는 CCM 파일이 저장되지 않았습니다.")
        return validation_results
    
    elif config.execution_mode == 'save':
        print("\n=== P3 CCM 블록 [저장 모드 - 메모리 최적화] ===")
        results = process_ccm_pipeline_v5(area_list, config)
        return results
    
    else:
        raise ValueError(f"지원하지 않는 실행 모드: {config.execution_mode}")


# ===============================
# P3 v5 메인 처리 함수
# ===============================

def process_ccm_pipeline_v5(area_ids: List[int] = None, config: P3_Config = None) -> Dict[str, Any]:
    """P3 v5.0 메인 파이프라인: 데이터 효율성 최적화"""
    
    if config is None:
        config = P3_Config()
    
    if area_ids is None:
        area_ids = list(config.area_rx_mapping.keys())
    
    # 필터링 설정 표시 (내부 함수이므로 간결하게)
    if config.zero_channel_filtering['enable_filtering'] or config.distance_filtering['enable_filtering']:
        filter_info = []
        if config.zero_channel_filtering['enable_filtering']:
            filter_info.append(f"Zero 채널 (임계값: {config.zero_channel_filtering['threshold']:.0e})")
        if config.distance_filtering['enable_filtering']:
            filter_info.append(f"거리 기반 ({config.distance_filtering['max_distance_meters']}m)")
        print(f"활성화된 필터링: {', '.join(filter_info)}")
    
    # CCM 엔진 및 관리자 초기화
    ccm_engine = CCM_BlockEngine(config)
    ccm_manager = CCM_BlockManager(config)
    
    overall_start_time = time.time()
    results = {}
    
    for area_id in area_ids:
        area_start_time = time.time()
        print(f"\n처리 시작: Area {area_id}")
        
        # v5.1: 영구 working directory 사용 (임시 디렉토리 제거)
        # CCM 블록 처리
        area_result = ccm_engine.process_area_ccm(area_id)
        
        # 결과 패키징 및 저장 (저장 모드일 때만)
        if config.execution_mode == 'save':
            working_dir = config.get_working_dir(area_id)
            ccm_manager.save_area_results(area_result, working_dir)
        
        area_time = time.time() - area_start_time
        area_result['area_processing_time'] = area_time
        results[area_id] = area_result
        
        print(f"Area {area_id} 완료: {format_time(area_time)}")
    
    overall_time = time.time() - overall_start_time
    
    print(f"\n=== P3 CCM 블록 처리 완료 ===")
    print(f"전체 처리 시간: {format_time(overall_time)}")
    print(f"처리 완료된 Area: {list(results.keys())} (총 {len(results)}개)")
    
    # 상세 통계 출력
    total_original_rx = sum(len(r.get('original_rx_ids', r['rx_ids'])) for r in results.values())
    total_filtered_rx = sum(len(r['rx_ids']) for r in results.values())
    total_blocks = sum(r['processing_stats']['num_blocks'] for r in results.values())
    
    print(f"총 처리된 RX: {total_filtered_rx}개")
    print(f"총 계산된 블록: {total_blocks}개")
    
    if total_original_rx > 0 and total_original_rx != total_filtered_rx:
        rx_efficiency = (total_original_rx - total_filtered_rx) / total_original_rx * 100
        print(f"필터링으로 제외된 RX: {total_original_rx - total_filtered_rx}개 ({rx_efficiency:.1f}%)")
    
    if config.ENABLE_MEMORY_MONITORING:
        print("최종 메모리 상태:")
        print_memory_usage()
    
    return {
        'config': config,
        'area_results': results,
        'overall_processing_time': overall_time,
        'v5_summary': {
            'total_original_rx': total_original_rx,
            'total_filtered_rx': total_filtered_rx,
            'efficiency_gain': rx_efficiency if total_original_rx > 0 else 0
        }
    }


# ===============================
# 메인 실행부 (스크립트 직접 실행 시)
# ===============================

if __name__ == "__main__":
    print("P3 CCM 블록 처리 파이프라인")
    
    try:
        # 설정 초기화 및 데이터 감지
        config = P3_Config()
        available_areas = list(config.area_rx_mapping.keys())
        
        if available_areas:
            # 메인 파이프라인 실행
            result = main_p3_v5_pipeline(config)
        else:
            print("처리 가능한 P2 데이터가 없습니다.")
            print("OFDM_Ch_Results 디렉토리에 P2 파일을 확인하세요.")
    
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()
