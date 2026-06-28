#!/usr/bin/env python3
"""
P1B Valid RX Filter - Dynamic Version
Author: Jonghyun Kim
Date: 2025-09-15
Version: v1

Dynamic script to filter P1A ray NPZ data by removing negative delay RX points.
Supports multiple area/frequency combinations with automatic file detection.
"""

import json
import numpy as np
import os
import glob
import re
from datetime import datetime
from typing import Dict, List, Tuple, Optional

class P1B_Config:
    """P1B Valid RX Filter 설정 및 동적 파일 감지"""
    
    def __init__(self):
        # 파일 경로 설정
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.P1A_INPUT_DIR = os.path.join(script_dir, "P1A_RT_Results")
        self.P1B_OUTPUT_DIR = os.path.join(script_dir, "P1B_Valid_Results") 
        
        # 파일 패턴 설정
        self.P1A_RAY_PATTERN = "Area{area}_{freq}GHz_Rays_ALL_RXs.npz"
        self.P1A_METADATA_PATTERN = "P1_Ray_Metadata_*_Filtered.json"  # 필터링된 메타데이터 우선
        self.P1A_ORIG_METADATA_PATTERN = "P1_Ray_Metadata_*.json"     # 원본 메타데이터 백업
        
        self.P1B_OUTPUT_PATTERN = "Area{area}_{freq}GHz_Rays_Valid_RXs.npz"
        self.P1B_INFO_PATTERN = "Area{area}_{freq}GHz_Rays_Valid_RXs_FilterInfo.json"
        
        # 필터링 설정
        self.TAU_MIN_THRESHOLD = -0.1  # 음수 지연 임계값 (초)
        
        # Ray Efficiency 필터링 설정
        self.ENABLE_RAY_EFFICIENCY_FILTER = True  # Ray efficiency 필터링 활성화/비활성화
        self.RAY_EFFICIENCY_THRESHOLD = 1.0        # Ray efficiency 최소 임계값 (0.0-1.0)
        
        # LoS/NLoS Ray 필터링 설정 (Enhanced)
        # Options: 'disabled', 'nlos_only', 'los_only'
        self.LOS_NLOS_FILTER_MODE = 'nlos_only'    # 'disabled': 비활성화, 'nlos_only': NLoS Ray만 유지, 'los_only': LoS Ray만 유지
        
        # 처리 대상 필터링 (None이면 모든 조합 처리)
        self.target_areas = None  # [1, 2, 3] 등으로 제한 가능
        self.target_frequencies = None  # [7.5, 28.0] 등으로 제한 가능
        
        # P1A 데이터 자동 감지
        self.detect_p1a_data()
        
        # 출력 디렉토리 생성
        os.makedirs(self.P1B_OUTPUT_DIR, exist_ok=True)
        
        # 설정 정보 출력
        self.print_config()
    
    def detect_p1a_data(self):
        """P1A 저장 데이터를 스캔하여 사용 가능한 조합 감지"""
        
        # P1A 결과 디렉토리에서 ray 파일들 스캔
        scan_pattern = os.path.join(self.P1A_INPUT_DIR, self.P1A_RAY_PATTERN.format(area='*', freq='*'))
        files = glob.glob(scan_pattern)
        
        # (area, freq) 조합 저장할 집합
        combinations = set()
        
        # 파일명에서 area_index, frequency 추출
        for file_path in files:
            filename = os.path.basename(file_path)
            # Area{X}_{freq}GHz_Rays_ALL_RXs.npz 패턴 매칭
            match = re.match(r'Area(\d+)_(.+)GHz_Rays_ALL_RXs\.npz', filename)
            if match:
                area_index = int(match.group(1))
                frequency = float(match.group(2))
                
                # 타겟 필터링 적용
                if self.target_areas is not None and area_index not in self.target_areas:
                    continue
                if self.target_frequencies is not None and frequency not in self.target_frequencies:
                    continue
                
                combinations.add((area_index, frequency))
        
        # 정렬하여 리스트로 저장
        self.p1a_data_combinations = sorted(list(combinations))
        
        # 메타데이터 파일 감지
        self.detect_metadata_files()
        
        print(f"P1A 데이터 자동 감지:")
        print(f"  - 감지된 조합 수: {len(self.p1a_data_combinations)}")
        for area, freq in self.p1a_data_combinations:
            print(f"    - Area{area}_{freq}GHz")
    
    def detect_metadata_files(self):
        """메타데이터 파일들 감지"""
        
        # 필터링된 메타데이터 파일 우선 검색
        filtered_pattern = os.path.join(self.P1A_INPUT_DIR, self.P1A_METADATA_PATTERN)
        filtered_files = glob.glob(filtered_pattern)
        
        # 원본 메타데이터 파일 검색
        original_pattern = os.path.join(self.P1A_INPUT_DIR, self.P1A_ORIG_METADATA_PATTERN)
        original_files = glob.glob(original_pattern)
        
        # 사용할 메타데이터 파일 결정 (필터링된 것 우선)
        if filtered_files:
            self.metadata_file = filtered_files[0]  # 최신 것 사용
            self.use_filtered_metadata = True
            print(f"  - 메타데이터: {os.path.basename(self.metadata_file)} (필터링됨)")
        elif original_files:
            self.metadata_file = original_files[0]  # 최신 것 사용  
            self.use_filtered_metadata = False
            print(f"  - 메타데이터: {os.path.basename(self.metadata_file)} (원본)")
        else:
            self.metadata_file = None
            self.use_filtered_metadata = False
            print(f"  - 메타데이터: 없음 (RX 검증 건너뜀)")
    
    def print_config(self):
        """설정 정보 출력"""
        print(f"P1B Valid RX 필터 설정:")
        print(f"  - 입력 디렉토리: {self.P1A_INPUT_DIR}")
        print(f"  - 출력 디렉토리: {self.P1B_OUTPUT_DIR}")
        print(f"  - 타우 최소 임계값: {self.TAU_MIN_THRESHOLD}초")
        print(f"  - Ray Efficiency 필터링: {'활성화' if self.ENABLE_RAY_EFFICIENCY_FILTER else '비활성화'}")
        if self.ENABLE_RAY_EFFICIENCY_FILTER:
            print(f"    └─ 임계값: {self.RAY_EFFICIENCY_THRESHOLD}")
        print(f"  - LoS/NLoS Ray 필터링: {self.LOS_NLOS_FILTER_MODE}")
        if self.LOS_NLOS_FILTER_MODE == 'nlos_only':
            print(f"    └─ NLoS Ray만 유지 (LoS Ray 제거)")
        elif self.LOS_NLOS_FILTER_MODE == 'los_only':
            print(f"    └─ LoS Ray만 유지 (NLoS Ray 제거)")
        print(f"  - 처리 대상 Area: {self.target_areas or 'All'}")
        print(f"  - 처리 대상 Freq: {self.target_frequencies or 'All'}")

class P1B_RayNPZFilter:
    """Dynamic filter for ray NPZ data to remove negative delay RX"""
    
    def __init__(self, config: P1B_Config):
        self.config = config
        self.tau_min_threshold = config.TAU_MIN_THRESHOLD
        
    def get_rx_validity_map_from_metadata(self, area: int, freq: float) -> Optional[Dict[int, bool]]:
        """Get RX validity mapping from metadata file for specific area/frequency
        
        Args:
            area: Area index
            freq: Frequency in GHz
            
        Returns:
            Dict: {rx_index: is_valid, ...} or None if no metadata available
        """
        
        if self.config.metadata_file is None:
            print(f"Warning: No metadata file available for Area{area}_{freq}GHz")
            return None
        
        try:
            with open(self.config.metadata_file, 'r') as f:
                metadata = json.load(f)
            
            validity_map = {}
            
            if 'ray_statistics_summary' in metadata:
                ray_stats = metadata['ray_statistics_summary']
                
                # Area/frequency 특정 키 패턴 생성
                target_prefix = f"Area{area}_{freq}GHz_RX"
                
                for rx_key, rx_data in ray_stats.items():
                    # 해당 area/frequency의 RX 데이터만 처리
                    if rx_key.startswith(target_prefix):
                        rx_index = int(rx_key.split('_RX')[1])
                        
                        # 필터링된 메타데이터인 경우 validity 필드 확인
                        if self.config.use_filtered_metadata:
                            is_valid = 'validity' not in rx_data or rx_data['validity']['is_valid']
                        else:
                            # 원본 메타데이터인 경우 tau_min으로 판단
                            is_valid = True
                            if isinstance(rx_data, dict) and 'parameters' in rx_data:
                                parameters = rx_data['parameters']
                                if 'tau' in parameters and isinstance(parameters['tau'], dict):
                                    tau_data = parameters['tau']
                                    tau_min = tau_data.get('min', 0)
                                    if tau_min < self.tau_min_threshold:
                                        is_valid = False
                        
                        validity_map[rx_index] = is_valid
            
            if validity_map:
                valid_count = sum(validity_map.values())
                total_count = len(validity_map)
                print(f"Area{area}_{freq}GHz: {valid_count}/{total_count} valid RX loaded from metadata")
                return validity_map
            else:
                print(f"Warning: No RX data found for Area{area}_{freq}GHz in metadata")
                return None
                
        except Exception as e:
            print(f"Error loading metadata for Area{area}_{freq}GHz: {e}")
            return None
    
    def load_ray_data(self, area: int, freq: float) -> Optional[Dict[str, np.ndarray]]:
        """Load ray NPZ data for specific area/frequency
        
        Args:
            area: Area index
            freq: Frequency in GHz
            
        Returns:
            Dict containing ray data arrays or None if file not found
        """
        
        # 입력 파일 경로 구성
        npz_filename = self.config.P1A_RAY_PATTERN.format(area=area, freq=freq)
        npz_path = os.path.join(self.config.P1A_INPUT_DIR, npz_filename)
        
        if not os.path.exists(npz_path):
            print(f"Error: Ray data file not found: {npz_path}")
            return None
        
        try:
            data = np.load(npz_path, allow_pickle=True)
            
            # Convert to dictionary for easier manipulation
            ray_data = {}
            for key in data.keys():
                ray_data[key] = data[key].copy()
            
            data.close()  # Close the npz file
            
            print(f"Loaded ray data: Area{area}_{freq}GHz")
            if 'tau' in ray_data:
                print(f"  - Original shape: {ray_data['tau'].shape}")
            if 'rx_indices' in ray_data:
                print(f"  - RX count: {len(ray_data['rx_indices'])}")
            
            # Check for enhanced P1A data format with Path-Ray relationship
            has_path_ray_info = 'source_path_idx' in ray_data and 'los_nlos_flag' in ray_data
            if has_path_ray_info:
                print(f"  - Enhanced P1A format detected with Path-Ray relationship and LoS/NLoS information")
                if 'los_nlos_flag' in ray_data:
                    total_rays = np.sum(ray_data['counts'])
                    los_rays = np.sum(ray_data['los_nlos_flag'] == 1)  
                    nlos_rays = np.sum(ray_data['los_nlos_flag'] == 0)
                    print(f"  - Ray composition: {los_rays} LoS, {nlos_rays} NLoS (Total: {total_rays})")
            else:
                print(f"  - Legacy P1A format (no Path-Ray relationship information)")
            
            return ray_data
            
        except Exception as e:
            print(f"Error loading ray data for Area{area}_{freq}GHz: {e}")
            return None
    
    def create_rx_mask(self, rx_indices: np.ndarray, validity_map: dict) -> tuple:
        """Create boolean mask for valid RX
        
        Returns:
            (valid_mask, valid_rx_indices)
        """
        
        # Create boolean mask for valid RX
        valid_mask = np.zeros(len(rx_indices), dtype=bool)
        
        for i, rx_idx in enumerate(rx_indices):
            if rx_idx in validity_map:
                valid_mask[i] = validity_map[rx_idx]
        
        # Get valid RX indices
        valid_rx_indices = rx_indices[valid_mask]
        
        return valid_mask, valid_rx_indices
    
    def calculate_ray_efficiency(self, ray_data: dict) -> Tuple[np.ndarray, Dict[int, float]]:
        """Calculate ray efficiency for each RX
        
        Ray efficiency = valid_rays / total_rays
        where valid_rays are rays with power > 0
        
        Args:
            ray_data: Dictionary containing ray data arrays
            
        Returns:
            Tuple of (efficiency_array, efficiency_dict)
            - efficiency_array: numpy array of efficiency values for each RX
            - efficiency_dict: dict mapping RX_index to efficiency value
        """
        
        if 'power' not in ray_data or 'counts' not in ray_data or 'rx_indices' not in ray_data:
            raise ValueError("Ray data must contain 'power', 'counts', and 'rx_indices' arrays")
            
        rx_indices = ray_data['rx_indices']
        power_data = ray_data['power']  # shape: (num_rx, 1, 1, 1, 1, num_rays)
        counts_data = ray_data['counts']  # shape: (num_rx, 1, 1, 1)
        
        efficiency_array = np.zeros(len(rx_indices))
        efficiency_dict = {}
        
        for i, rx_id in enumerate(rx_indices):
            # Extract power data for this RX
            rx_power = power_data[i, 0, 0, 0, 0, :]  # shape: (num_rays,)
            total_rays = int(counts_data[i, 0, 0, 0])  # Total ray count
            
            # Count valid rays (power > 0)
            valid_rays = int(np.sum(rx_power > 0))
            
            # Calculate efficiency
            if total_rays > 0:
                efficiency = valid_rays / total_rays
            else:
                efficiency = 0.0
                
            efficiency_array[i] = efficiency
            efficiency_dict[int(rx_id)] = efficiency
            
        return efficiency_array, efficiency_dict
    
    def filter_by_ray_efficiency(self, ray_data: dict) -> Tuple[np.ndarray, np.ndarray, Dict[str, any]]:
        """Filter RXs by ray efficiency threshold
        
        Args:
            ray_data: Dictionary containing ray data arrays
            
        Returns:
            Tuple of (valid_mask, valid_rx_indices, efficiency_info)
            - valid_mask: Boolean array indicating which RXs pass the filter
            - valid_rx_indices: Array of RX indices that pass the filter
            - efficiency_info: Dictionary with efficiency statistics
        """
        
        if not self.config.ENABLE_RAY_EFFICIENCY_FILTER:
            # If filtering is disabled, return all as valid
            rx_indices = ray_data['rx_indices']
            valid_mask = np.ones(len(rx_indices), dtype=bool)
            return valid_mask, rx_indices, {'status': 'disabled'}
            
        # Calculate ray efficiency
        efficiency_array, efficiency_dict = self.calculate_ray_efficiency(ray_data)
        
        # Apply efficiency threshold filter
        efficiency_mask = efficiency_array >= self.config.RAY_EFFICIENCY_THRESHOLD
        valid_mask = efficiency_mask
            
        rx_indices = ray_data['rx_indices']
        valid_rx_indices = rx_indices[valid_mask]
        
        # Gather statistics
        efficiency_info = {
            'status': 'enabled',
            'threshold': self.config.RAY_EFFICIENCY_THRESHOLD,
            'total_rxs': len(rx_indices),
            'passed_rxs': len(valid_rx_indices),
            'filtered_rxs': len(rx_indices) - len(valid_rx_indices),
            'avg_efficiency': float(efficiency_array.mean()),
            'min_efficiency': float(efficiency_array.min()),
            'max_efficiency': float(efficiency_array.max()),
            'failed_rx_list': [int(rx_id) for i, rx_id in enumerate(rx_indices) if not valid_mask[i]]
        }
        
        return valid_mask, valid_rx_indices, efficiency_info
    
    def filter_by_los_nlos(self, ray_data: dict) -> Tuple[np.ndarray, Dict[str, any]]:
        """Filter rays based on LoS/NLoS flag according to configuration
        
        Args:
            ray_data: Dictionary containing ray data arrays
            
        Returns:
            Tuple of (ray_mask, los_nlos_info)
            - ray_mask: Boolean array indicating which rays pass the filter [num_rx, 1, 1, 1, 1, num_rays]
            - los_nlos_info: Dictionary with LoS/NLoS filtering statistics
        """
        
        filter_mode = self.config.LOS_NLOS_FILTER_MODE
        
        if filter_mode == 'disabled':
            # No filtering, return all rays as valid
            if 'tau' in ray_data:
                ray_shape = ray_data['tau'].shape
                ray_mask = np.ones(ray_shape, dtype=bool)
            else:
                ray_mask = None
            return ray_mask, {'status': 'disabled'}
        
        # Check if enhanced P1A format with LoS/NLoS information is available
        if 'los_nlos_flag' not in ray_data:
            print(f"Warning: LoS/NLoS filtering requested but 'los_nlos_flag' not found in data")
            print(f"         Falling back to no filtering (legacy P1A format)")
            if 'tau' in ray_data:
                ray_shape = ray_data['tau'].shape
                ray_mask = np.ones(ray_shape, dtype=bool)
            else:
                ray_mask = None
            return ray_mask, {'status': 'unavailable', 'reason': 'legacy_format'}
        
        # Apply LoS/NLoS filtering
        los_nlos_flags = ray_data['los_nlos_flag']  # shape: [num_rx, 1, 1, 1, 1, num_rays]
        
        if filter_mode == 'nlos_only':
            # Keep only NLoS rays (los_nlos_flag == 0)
            ray_mask = (los_nlos_flags == 0)
            filter_description = "NLoS rays only (LoS rays removed)"
        elif filter_mode == 'los_only':
            # Keep only LoS rays (los_nlos_flag == 1) 
            ray_mask = (los_nlos_flags == 1)
            filter_description = "LoS rays only (NLoS rays removed)"
        else:
            raise ValueError(f"Unknown LOS_NLOS_FILTER_MODE: {filter_mode}")
        
        # Calculate statistics
        total_rays = los_nlos_flags.size
        los_rays = np.sum(los_nlos_flags == 1)
        nlos_rays = np.sum(los_nlos_flags == 0)
        kept_rays = np.sum(ray_mask)
        filtered_rays = total_rays - kept_rays
        
        los_nlos_info = {
            'status': 'applied',
            'filter_mode': filter_mode,
            'filter_description': filter_description,
            'total_rays': int(total_rays),
            'original_los_rays': int(los_rays),
            'original_nlos_rays': int(nlos_rays),
            'kept_rays': int(kept_rays),
            'filtered_rays': int(filtered_rays),
            'filter_ratio': float(filtered_rays / total_rays) if total_rays > 0 else 0.0
        }
        
        return ray_mask, los_nlos_info
    
    def filter_ray_data(self, ray_data: dict, valid_mask: np.ndarray) -> dict:
        """Filter ray data to keep only valid RX"""
        
        filtered_data = {}
        
        # Copy scalar values as-is
        for key in ['area_index', 'frequency_ghz']:
            if key in ray_data:
                filtered_data[key] = ray_data[key].copy()
        
        # Filter RX-dependent arrays
        if 'rx_indices' in ray_data:
            filtered_data['rx_indices'] = ray_data['rx_indices'][valid_mask]
            filtered_data['num_rx'] = len(filtered_data['rx_indices'])
        
        # Filter high-dimensional arrays (shape: (N_RX, 1, 1, 1, 1, N_rays))
        # Include enhanced P1A fields with Path-Ray relationship and LoS/NLoS information
        for key in ['tau', 'power', 'theta_r_deg', 'theta_t_deg', 'phi_r_deg', 'phi_t_deg', 
                   'source_path_idx', 'los_nlos_flag']:
            if key in ray_data:
                filtered_data[key] = ray_data[key][valid_mask]
        
        # Filter counts array (shape: (N_RX, 1, 1, 1))
        if 'counts' in ray_data:
            filtered_data['counts'] = ray_data['counts'][valid_mask]
        
        return filtered_data
    
    def _remove_all_los_rx(self, ray_data: dict, area: int, freq: float) -> dict:
        """모든 Ray가 LoS인 RX를 제거 (P1D 분리성 분석 품질 보증)
        
        Args:
            ray_data: Ray 데이터 딕셔너리
            area: Area index (로그용)  
            freq: Frequency (로그용)
            
        Returns:
            All-LoS RX가 제거된 ray 데이터
        """
        
        if 'los_nlos_flag' not in ray_data:
            print(f"    3-2단계: LoS/NLoS 정보 없음 - All-LoS RX 필터링 건너뜀")
            return ray_data
        
        # RX별 LoS/NLoS Ray 개수 계산
        rx_indices = ray_data.get('rx_indices', [])
        los_nlos_flags = ray_data['los_nlos_flag']  # [N_RX, max_rays]
        
        valid_rx_indices = []
        all_los_rx_count = 0
        
        for i, rx_idx in enumerate(rx_indices):
            # 해당 RX의 유효한 Ray 마스크 (NaN이나 0이 아닌 power)
            if 'power' in ray_data:
                valid_ray_mask = ~np.isnan(ray_data['power'][i]) & (ray_data['power'][i] > 0)
            else:
                # power 정보가 없으면 los_nlos_flag 기준으로 판단
                valid_ray_mask = los_nlos_flags[i] >= 0  # -1이 아닌 유효한 flag
            
            valid_los_nlos = los_nlos_flags[i][valid_ray_mask]
            
            if len(valid_los_nlos) == 0:
                # 유효한 Ray가 없는 경우 제거
                all_los_rx_count += 1
                continue
            
            # NLoS Ray(0) 개수 확인
            nlos_count = np.sum(valid_los_nlos == 0)
            
            if nlos_count > 0:
                # NLoS Ray가 하나라도 있으면 유지
                valid_rx_indices.append(rx_idx)
            else:
                # 모든 Ray가 LoS인 경우 제거
                all_los_rx_count += 1
        
        # 결과 출력
        original_rx_count = len(rx_indices)
        final_rx_count = len(valid_rx_indices)
        removed_rx_count = original_rx_count - final_rx_count
        
        print(f"    3-2단계: All-LoS RX 제거 - {original_rx_count:,} → {final_rx_count:,}개 ({removed_rx_count:,}개 제거)")
        
        if removed_rx_count == 0:
            return ray_data
        
        # 유효한 RX들만 필터링
        if final_rx_count == 0:
            print(f"    ⚠️  경고: 모든 RX가 All-LoS로 제거됨 (Area{area}_{freq}GHz)")
            # 빈 데이터 반환
            filtered_data = {key: np.array([]) if key != 'rx_indices' else [] for key in ray_data.keys()}
            return filtered_data
        
        # RX 인덱스 매핑 생성
        valid_rx_positions = [i for i, rx_idx in enumerate(rx_indices) if rx_idx in valid_rx_indices]
        
        # 모든 배열 필드 필터링
        filtered_data = {'rx_indices': valid_rx_indices}
        
        for key, value in ray_data.items():
            if key == 'rx_indices':
                continue
            elif isinstance(value, np.ndarray) and len(value.shape) > 0 and value.shape[0] == original_rx_count:
                # RX 차원을 가진 배열들 필터링
                filtered_data[key] = value[valid_rx_positions]
            else:
                # 스칼라나 RX와 무관한 데이터는 그대로 유지
                filtered_data[key] = value
        
        return filtered_data
    
    def apply_ray_mask(self, ray_data: dict, ray_mask: np.ndarray) -> dict:
        """Apply ray-level mask to filter out specific rays (e.g., LoS/NLoS filtering)
        
        Args:
            ray_data: Dictionary containing ray data arrays
            ray_mask: Boolean array indicating which rays to keep [num_rx, 1, 1, 1, 1, num_rays]
            
        Returns:
            Dictionary with ray-level filtered data
        """
        
        if ray_mask is None:
            return ray_data
        
        filtered_data = {}
        
        # Copy scalar values as-is (handle different types properly)
        for key in ['area_index', 'frequency_ghz', 'rx_indices', 'num_rx']:
            if key in ray_data:
                value = ray_data[key]
                # Handle different data types appropriately
                if isinstance(value, np.ndarray):
                    filtered_data[key] = value.copy()
                elif isinstance(value, list):
                    filtered_data[key] = value.copy()
                else:
                    # Scalar values (int, float, etc.) - assign directly
                    filtered_data[key] = value
        
        # Apply ray mask to ray data arrays
        ray_keys = ['tau', 'power', 'theta_r_deg', 'theta_t_deg', 'phi_r_deg', 'phi_t_deg', 
                   'source_path_idx', 'los_nlos_flag']
        
        for key in ray_keys:
            if key in ray_data:
                # Apply mask to filter rays
                masked_data = ray_data[key].copy()
                masked_data[~ray_mask] = 0  # Set filtered rays to zero (or could remove them entirely)
                filtered_data[key] = masked_data
        
        # Update counts to reflect actual number of kept rays
        if 'counts' in ray_data and ray_mask is not None:
            # Calculate new counts based on non-zero rays after masking
            filtered_data['counts'] = np.sum(ray_mask, axis=-1)  # Sum over ray dimension
        
        return filtered_data
    
    def save_filtered_data(self, filtered_data: Dict[str, np.ndarray], area: int, freq: float, 
                          original_count: int, filtered_count: int, los_nlos_info: dict = None) -> Tuple[str, str]:
        """Save filtered ray data and metadata for specific area/frequency
        
        Args:
            filtered_data: Filtered ray data dictionary
            area: Area index
            freq: Frequency in GHz
            original_count: Original RX count
            filtered_count: Filtered RX count
            
        Returns:
            Tuple of (npz_path, info_path)
        """
        
        # 출력 파일 경로 구성
        npz_filename = self.config.P1B_OUTPUT_PATTERN.format(area=area, freq=freq)
        npz_path = os.path.join(self.config.P1B_OUTPUT_DIR, npz_filename)
        
        info_filename = self.config.P1B_INFO_PATTERN.format(area=area, freq=freq)
        info_path = os.path.join(self.config.P1B_OUTPUT_DIR, info_filename)
        
        # Save NPZ data
        save_data = {k: v for k, v in filtered_data.items()}
        np.savez_compressed(npz_path, **save_data)
        print(f"Filtered ray data saved: {npz_filename}")
        
        # Create comprehensive and systematic metadata
        filtering_info = self._create_comprehensive_metadata(
            area, freq, original_count, filtered_count, filtered_data, los_nlos_info
        )
        
        with open(info_path, 'w') as f:
            json.dump(filtering_info, f, indent=2)
        print(f"Filtering info saved: {info_filename}")
        
        return npz_path, info_path
    
    def _create_comprehensive_metadata(self, area: int, freq: float, original_count: int, 
                                     filtered_count: int, filtered_data: Dict[str, np.ndarray], 
                                     los_nlos_info: dict) -> dict:
        """Create comprehensive and systematic metadata for P1B filtering results
        
        Args:
            area: Area index
            freq: Frequency in GHz
            original_count: Original RX count
            filtered_count: Filtered RX count
            filtered_data: Final filtered data dictionary
            los_nlos_info: LoS/NLoS filtering information
            
        Returns:
            Comprehensive metadata dictionary
        """
        
        current_time = datetime.now()
        
        # ===== Session Information =====
        session_info = {
            'filter_timestamp': current_time.strftime('%Y%m%d_%H%M%S'),
            'filter_datetime_iso': current_time.isoformat(),
            'pipeline_version': 'P1B_Valid_RX_Filter_v1',
            'filter_description': 'Enhanced RX and Ray level filtering with Path-Ray relationship support',
            'area_index': area,
            'frequency_ghz': freq,
            'combination_key': f"Area{area}_{freq}GHz"
        }
        
        # ===== Complete Configuration Snapshot =====
        configuration = {
            'tau_min_threshold': self.tau_min_threshold,
            'ray_efficiency_filter': {
                'enabled': self.config.ENABLE_RAY_EFFICIENCY_FILTER,
                'threshold': self.config.RAY_EFFICIENCY_THRESHOLD
            },
            'los_nlos_filter': {
                'mode': self.config.LOS_NLOS_FILTER_MODE,
                'description': self._get_filter_mode_description(self.config.LOS_NLOS_FILTER_MODE)
            },
            'target_filters': {
                'target_areas': self.config.target_areas,
                'target_frequencies': self.config.target_frequencies
            },
            'file_patterns': {
                'input_pattern': self.config.P1A_RAY_PATTERN,
                'output_pattern': self.config.P1B_OUTPUT_PATTERN,
                'info_pattern': self.config.P1B_INFO_PATTERN
            },
            'directories': {
                'input_dir': self.config.P1A_INPUT_DIR,
                'output_dir': self.config.P1B_OUTPUT_DIR
            }
        }
        
        # ===== Input Data Information =====
        input_filename = self.config.P1A_RAY_PATTERN.format(area=area, freq=freq)
        input_filepath = os.path.join(self.config.P1A_INPUT_DIR, input_filename)
        
        input_data_info = {
            'filename': input_filename,
            'filepath': input_filepath,
            'file_exists': os.path.exists(input_filepath),
            'file_size_mb': round(os.path.getsize(input_filepath) / (1024**2), 2) if os.path.exists(input_filepath) else None,
            'p1a_format': 'enhanced' if los_nlos_info and 'status' in los_nlos_info and los_nlos_info['status'] != 'unavailable' else 'legacy',
            'metadata_source': {
                'file': os.path.basename(self.config.metadata_file) if self.config.metadata_file else None,
                'type': 'filtered' if self.config.use_filtered_metadata else 'original'
            }
        }
        
        # ===== Filtering Pipeline Details =====
        filtering_pipeline = {
            'stages_applied': [],
            'stage_1_rx_metadata_filter': {
                'description': 'RX-level filtering based on metadata validity',
                'method': 'metadata_based_rx_exclusion',
                'original_rx_count': original_count,
                'filtered_rx_count': filtered_count,
                'excluded_rx_count': original_count - filtered_count,
                'exclusion_ratio': (original_count - filtered_count) / original_count if original_count > 0 else 0.0
            },
            'stage_2_ray_efficiency_filter': {
                'description': 'Ray-level efficiency filtering',
                'enabled': self.config.ENABLE_RAY_EFFICIENCY_FILTER,
                'threshold': self.config.RAY_EFFICIENCY_THRESHOLD if self.config.ENABLE_RAY_EFFICIENCY_FILTER else None,
                'method': 'valid_rays_ratio_threshold'
            },
            'stage_3_los_nlos_ray_filter': {
                'description': 'Ray-level LoS/NLoS filtering',
                'enabled': self.config.LOS_NLOS_FILTER_MODE != 'disabled',
                'mode': self.config.LOS_NLOS_FILTER_MODE,
                'filter_details': los_nlos_info if los_nlos_info else {'status': 'not_applied'}
            }
        }
        
        # Add applied stages
        filtering_pipeline['stages_applied'].append('stage_1_rx_metadata_filter')
        if self.config.ENABLE_RAY_EFFICIENCY_FILTER:
            filtering_pipeline['stages_applied'].append('stage_2_ray_efficiency_filter')
        if self.config.LOS_NLOS_FILTER_MODE != 'disabled':
            filtering_pipeline['stages_applied'].append('stage_3_los_nlos_ray_filter')
        
        # ===== Output Data Information =====
        output_filename = self.config.P1B_OUTPUT_PATTERN.format(area=area, freq=freq)
        output_filepath = os.path.join(self.config.P1B_OUTPUT_DIR, output_filename)
        
        output_data_info = {
            'filename': output_filename,
            'filepath': output_filepath,
            'final_rx_count': filtered_count,
            'data_fields': list(filtered_data.keys()) if filtered_data else [],
            'enhanced_fields': ['source_path_idx', 'los_nlos_flag'],
            'has_path_ray_relationship': all(field in filtered_data for field in ['source_path_idx', 'los_nlos_flag']) if filtered_data else False
        }
        
        # ===== Data Statistics =====
        statistics = {
            'rx_level': {
                'original_count': original_count,
                'filtered_count': filtered_count,
                'exclusion_count': original_count - filtered_count,
                'retention_ratio': filtered_count / original_count if original_count > 0 else 0.0,
                'exclusion_ratio': (original_count - filtered_count) / original_count if original_count > 0 else 0.0
            },
            'ray_level': {},
            'data_quality': {
                'has_enhanced_p1a_format': input_data_info['p1a_format'] == 'enhanced',
                'filtering_completeness': len(filtering_pipeline['stages_applied']) / 3  # 3 total possible stages
            }
        }
        
        # Add ray-level statistics if available
        if los_nlos_info and los_nlos_info.get('status') == 'applied':
            statistics['ray_level'] = {
                'original_total_rays': los_nlos_info.get('total_rays', 0),
                'original_los_rays': los_nlos_info.get('original_los_rays', 0),
                'original_nlos_rays': los_nlos_info.get('original_nlos_rays', 0),
                'filtered_rays': los_nlos_info.get('filtered_rays', 0),
                'kept_rays': los_nlos_info.get('kept_rays', 0),
                'ray_filter_ratio': los_nlos_info.get('filter_ratio', 0.0),
                'ray_retention_ratio': 1.0 - los_nlos_info.get('filter_ratio', 0.0)
            }
        
        # Calculate file size reduction if both files exist
        try:
            if os.path.exists(input_filepath) and os.path.exists(output_filepath):
                input_size = os.path.getsize(input_filepath)
                output_size = os.path.getsize(output_filepath) 
                size_reduction_ratio = (input_size - output_size) / input_size if input_size > 0 else 0.0
                output_data_info['file_size_mb'] = round(output_size / (1024**2), 2)
                output_data_info['size_reduction_ratio'] = size_reduction_ratio
                output_data_info['size_reduction_mb'] = round((input_size - output_size) / (1024**2), 2)
        except Exception as e:
            output_data_info['file_size_calculation_error'] = str(e)
        
        # ===== Performance Information =====
        performance = {
            'processing_timestamp': session_info['filter_timestamp'],
            'total_stages_applied': len(filtering_pipeline['stages_applied']),
            'filtering_efficiency': {
                'rx_processing_efficiency': filtered_count / original_count if original_count > 0 else 0.0,
                'overall_data_reduction': 1.0 - (filtered_count / original_count if original_count > 0 else 1.0)
            }
        }
        
        # ===== Comprehensive Metadata Structure =====
        comprehensive_metadata = {
            'session_info': session_info,
            'configuration': configuration,
            'input_data_info': input_data_info,
            'filtering_pipeline': filtering_pipeline,
            'output_data_info': output_data_info,
            'statistics': statistics,
            'performance': performance,
            
            # Legacy compatibility fields
            'filtering_applied': True,
            'filter_timestamp': session_info['filter_timestamp'],
            'area_index': area,
            'frequency_ghz': freq
        }
        
        return comprehensive_metadata
    
    def _get_filter_mode_description(self, mode: str) -> str:
        """Get human-readable description for filter mode"""
        descriptions = {
            'disabled': 'LoS/NLoS filtering disabled',
            'nlos_only': 'Keep only NLoS rays (remove LoS rays)',
            'los_only': 'Keep only LoS rays (remove NLoS rays)'
        }
        return descriptions.get(mode, f'Unknown mode: {mode}')
    
    def print_summary(self, area: int, freq: float, original_data: Dict[str, np.ndarray], 
                     filtered_data: Dict[str, np.ndarray], valid_rx_indices: np.ndarray):
        """간소화된 필터링 요약 (P1B 정상 수행 전제)"""
        
        original_count = original_data['tau'].shape[0]
        filtered_count = filtered_data['tau'].shape[0]
        filter_ratio = (original_count - filtered_count) / original_count
        
        print(f"\n--- Area{area}_{freq}GHz 필터링 완료 ---")
        print(f"RX: {original_count:,} → {filtered_count:,}개 ({filter_ratio:.1%} 제거)")
        print(f"유효 RX 범위: RX{min(valid_rx_indices)}-RX{max(valid_rx_indices)}")
        print("-" * 40)
    
    def process_area_frequency(self, area: int, freq: float) -> Optional[Dict[str, np.ndarray]]:
        """Process ray data filtering for specific area/frequency combination
        
        Args:
            area: Area index
            freq: Frequency in GHz
            
        Returns:
            Filtered ray data dictionary or None if processing failed
        """
        
        print(f"\nArea{area}_{freq}GHz 처리 중...")
        
        # Load ray data
        ray_data = self.load_ray_data(area, freq)
        if ray_data is None:
            return None
        
        # Get validity map from metadata
        validity_map = self.get_rx_validity_map_from_metadata(area, freq)
        
        # If no metadata available, process all RX as valid
        if validity_map is None:
            print(f"Warning: No validity map available for Area{area}_{freq}GHz, keeping all RX")
            # Create validity map with all RX as valid
            if 'rx_indices' in ray_data:
                validity_map = {int(rx_idx): True for rx_idx in ray_data['rx_indices']}
            else:
                print(f"Error: No rx_indices found in ray data for Area{area}_{freq}GHz")
                return None
        
        # ===== Ray Tracing 실패 필터링 시스템 =====
        # 1단계: 음수 지연 필터링 (실제 데이터 기반)
        # 2단계: Ray Efficiency 필터링 (실제 데이터 기반)
        
        original_count = len(ray_data['rx_indices'])
        print(f"  원본 RX 개수: {original_count:,}개")
        
        # 실제 데이터 기반으로 1단계와 2단계 필터링 해당 수 계산
        step1_filtering_count = 0  # 1단계 필터링 해당 수
        step2_filtering_count = 0  # 2단계 필터링 해당 수
        
        # 전체 RX에 대해 각각 체크
        for i, rx_idx in enumerate(ray_data['rx_indices']):
            # 1단계: 음수 지연 체크
            rx_tau = ray_data['tau'][i, 0, 0, 0, 0, :]
            has_negative_delay = np.any(rx_tau < self.config.TAU_MIN_THRESHOLD)
            if has_negative_delay:
                step1_filtering_count += 1
            
            # 2단계: Ray efficiency 체크
            if self.config.ENABLE_RAY_EFFICIENCY_FILTER:
                rx_power = ray_data['power'][i, 0, 0, 0, 0, :]
                total_rays = ray_data['counts'][i, 0, 0, 0]
                valid_rays = np.sum(rx_power > 0)
                efficiency = valid_rays / total_rays if total_rays > 0 else 0
                if efficiency < self.config.RAY_EFFICIENCY_THRESHOLD:
                    step2_filtering_count += 1
        
        print(f"  1단계 필터링 해당 수: {step1_filtering_count:,}개 (음수 지연)")
        print(f"  2단계 필터링 해당 수: {step2_filtering_count:,}개 (Ray Efficiency < {self.config.RAY_EFFICIENCY_THRESHOLD})")
        
        # 기존 메타데이터 기반 필터링 (호환성 유지)
        step1_mask, step1_valid_indices = self.create_rx_mask(ray_data['rx_indices'], validity_map)
        step1_valid_set = set(step1_valid_indices)
        step1_passed_count = len(step1_valid_indices)
        filt_count_1 = original_count - step1_passed_count  # 메타데이터 기반 제외 개수
        
        print(f"  메타데이터 기반 제외: {filt_count_1:,}개")
        
        # 2단계: Ray Efficiency 필터링 (메타데이터 통과 RX 대상)
        filt_count_2 = 0
        final_valid_indices = step1_valid_indices  # 기본값
        
        if self.config.ENABLE_RAY_EFFICIENCY_FILTER:
            # 1단계 통과 데이터에 대해 Ray efficiency 계산
            efficiency_array, efficiency_dict = self.calculate_ray_efficiency(ray_data)
            
            # 1단계 통과 RX들 중에서 Ray efficiency 기준 미달 찾기
            step2_failed_rxs = []
            for i, rx_idx in enumerate(ray_data['rx_indices']):
                if rx_idx in step1_valid_set:  # 1단계는 통과했지만
                    if efficiency_array[i] < self.config.RAY_EFFICIENCY_THRESHOLD:  # 2단계 기준 미달
                        step2_failed_rxs.append(int(rx_idx))
            
            filt_count_2 = len(step2_failed_rxs)  # 2단계에서 제외된 개수
            step2_passed_count = step1_passed_count - filt_count_2
            
            # 최종 통과 RX 리스트 생성 (1단계 통과 + 2단계 통과)
            final_valid_indices = [rx for rx in step1_valid_indices if int(rx) not in step2_failed_rxs]
            final_valid_indices = np.array(final_valid_indices)
            
            # 최종 마스크 생성
            final_mask = np.zeros(len(ray_data['rx_indices']), dtype=bool)
            for i, rx_idx in enumerate(ray_data['rx_indices']):
                if rx_idx in final_valid_indices:
                    final_mask[i] = True
            
            valid_mask = final_mask
            valid_rx_indices = final_valid_indices
        else:
            print(f"  2단계 - Ray Efficiency 필터링: 비활성화됨")
            valid_mask = step1_mask
            valid_rx_indices = step1_valid_indices
        
        # ===== RX 레벨 필터링 결과 요약 =====
        rx_final_count = len(valid_rx_indices)
        rx_total_filtered = original_count - rx_final_count  # 실제 제외된 총 개수
        
        print(f"  ===== Ray Tracing 실패 필터링 결과 (RX 레벨) =====")        
        print(f"    원본 RX: {original_count:,}개 -> RX 필터링 후: {rx_final_count:,}개")
        print(f"    총 제외: {rx_total_filtered:,}개 ({rx_total_filtered/original_count*100:.1f}%)")
        print(f"       ├─ 1단계 (음수 지연): {step1_filtering_count:,}개")
        print(f"       └─ 2단계 (Ray Efficiency): {step2_filtering_count:,}개")
        print(f"    Ray Tracing 성공률: {rx_final_count/original_count*100:.1f}%")
        
        if rx_final_count == 0:
            print(f"경고: Area{area}_{freq}GHz의 모든 RX가 Ray Tracing 실패로 필터링되어 처리를 건너뜁니다.")
            return None
        
        # Apply RX-level filtering first
        rx_filtered_data = self.filter_ray_data(ray_data, valid_mask)
        
        # ===== 3단계: Ray 레벨 LoS/NLoS 필터링 =====
        print(f"  3단계 - LoS/NLoS Ray 필터링: {self.config.LOS_NLOS_FILTER_MODE}")
        ray_mask, los_nlos_info = self.filter_by_los_nlos(rx_filtered_data)
        
        if los_nlos_info['status'] == 'applied':
            print(f"    필터링 모드: {los_nlos_info['filter_description']}")
            print(f"    원본 Ray: {los_nlos_info['total_rays']:,}개 (LoS: {los_nlos_info['original_los_rays']:,}, NLoS: {los_nlos_info['original_nlos_rays']:,})")
            print(f"    유지 Ray: {los_nlos_info['kept_rays']:,}개, 제거 Ray: {los_nlos_info['filtered_rays']:,}개")
            print(f"    Ray 제거율: {los_nlos_info['filter_ratio']*100:.1f}%")
        elif los_nlos_info['status'] == 'unavailable':
            print(f"    LoS/NLoS 필터링 불가: {los_nlos_info['reason']}")
        else:
            print(f"    LoS/NLoS 필터링: 비활성화")
        
        # Apply ray-level LoS/NLoS filtering
        filtered_ray_data = self.apply_ray_mask(rx_filtered_data, ray_mask)
        
        # === 3-2단계: 모든 LoS RX 제거 (P1D 분리성 분석 품질 보증) ===
        if self.config.LOS_NLOS_FILTER_MODE == 'nlos_only':
            filtered_ray_data = self._remove_all_los_rx(filtered_ray_data, area, freq)
        
        # ===== 최종 결과 요약 =====
        print(f"  ===== 전체 필터링 결과 요약 =====")
        print(f"    RX 필터링: {original_count:,} → {rx_final_count:,}개 ({rx_total_filtered:,}개 제거)")
        if los_nlos_info['status'] == 'applied':
            print(f"    Ray 필터링: {los_nlos_info['total_rays']:,} → {los_nlos_info['kept_rays']:,}개 ({los_nlos_info['filtered_rays']:,}개 제거)")
        print(f"    최종 데이터: {rx_final_count:,} RX, Ray 필터링 적용됨")
        
        # Save filtered data with LoS/NLoS filtering info
        npz_path, info_path = self.save_filtered_data(filtered_ray_data, area, freq, 
                                                     original_count, rx_final_count, los_nlos_info)
        
        # Print summary
        self.print_summary(area, freq, ray_data, filtered_ray_data, valid_rx_indices)
        
        return filtered_ray_data
    
    def process_all_combinations(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Process all detected area/frequency combinations
        
        Returns:
            Dictionary of {area_freq_key: filtered_data, ...}
        """
        
        if not self.config.p1a_data_combinations:
            print("No P1A data combinations detected. Please check input directory.")
            return {}
        
        print(f"\n=== P1B Valid RX Filtering ===")
        print(f"Processing {len(self.config.p1a_data_combinations)} area/frequency combinations...")
        
        results = {}
        success_count = 0
        
        for area, freq in self.config.p1a_data_combinations:
            filtered_data = self.process_area_frequency(area, freq)
            
            if filtered_data is not None:
                area_freq_key = f"Area{area}_{freq}GHz"
                results[area_freq_key] = filtered_data
                success_count += 1
            else:
                print(f"Failed to process Area{area}_{freq}GHz")
        
        print(f"\n=== P1B Processing Completed ===")
        print(f"Successfully processed: {success_count}/{len(self.config.p1a_data_combinations)} combinations")
        print(f"Output directory: {self.config.P1B_OUTPUT_DIR}")
        
        return results

def main():
    """Main execution - dynamic P1B valid RX filtering"""
    
    print("="*80)
    print("P1B VALID RX FILTERING - DYNAMIC VERSION")
    print("Dynamic ray data filtering with automatic file detection")
    print("="*80)
    
    try:
        # Initialize configuration and auto-detect P1A data
        config = P1B_Config()
        
        # Initialize filter
        ray_filter = P1B_RayNPZFilter(config)
        
        # Process all detected combinations
        results = ray_filter.process_all_combinations()
        
        # Print final summary
        if results:
            print(f"\n=== Final Results ===")
            for area_freq_key, filtered_data in results.items():
                rx_count = len(filtered_data['rx_indices']) if 'rx_indices' in filtered_data else 0
                print(f"  - {area_freq_key}: {rx_count:,} valid RX")
            print(f"Total combinations processed: {len(results)}")
        else:
            print("No data was successfully processed.")
        
        return results
        
    except Exception as e:
        print(f"Error during P1B processing: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()
