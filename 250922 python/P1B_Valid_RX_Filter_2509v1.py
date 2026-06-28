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
        for key in ['tau', 'power', 'theta_r', 'theta_t', 'phi_r', 'phi_t']:
            if key in ray_data:
                filtered_data[key] = ray_data[key][valid_mask]
        
        # Filter counts array (shape: (N_RX, 1, 1, 1))
        if 'counts' in ray_data:
            filtered_data['counts'] = ray_data['counts'][valid_mask]
        
        return filtered_data
    
    def save_filtered_data(self, filtered_data: Dict[str, np.ndarray], area: int, freq: float, 
                          original_count: int, filtered_count: int) -> Tuple[str, str]:
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
        
        # Save filtering info as JSON
        filtering_info = {
            'filtering_applied': True,
            'filter_timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'area_index': area,
            'frequency_ghz': freq,
            'original_rx_count': original_count,
            'filtered_rx_count': filtered_count,
            'valid_rx_count': filtered_count,
            'filter_ratio': (original_count - filtered_count) / original_count if original_count > 0 else 0.0,
            'tau_min_threshold': self.tau_min_threshold,
            'input_file': self.config.P1A_RAY_PATTERN.format(area=area, freq=freq),
            'metadata_file': os.path.basename(self.config.metadata_file) if self.config.metadata_file else None,
            'use_filtered_metadata': self.config.use_filtered_metadata
        }
        
        with open(info_path, 'w') as f:
            json.dump(filtering_info, f, indent=2)
        print(f"Filtering info saved: {info_filename}")
        
        return npz_path, info_path
    
    def print_summary(self, area: int, freq: float, original_data: Dict[str, np.ndarray], 
                     filtered_data: Dict[str, np.ndarray], valid_rx_indices: np.ndarray):
        """Print filtering summary for specific area/frequency"""
        
        original_shape = original_data['tau'].shape
        filtered_shape = filtered_data['tau'].shape
        
        print(f"\n--- Area{area}_{freq}GHz Filtering Summary ---")
        print(f"Original RX count:    {original_shape[0]:,}")
        print(f"Filtered RX count:    {filtered_shape[0]:,}")
        print(f"Reduction ratio:      {(original_shape[0] - filtered_shape[0]) / original_shape[0]:.1%}")
        print(f"Original data size:   {original_shape}")
        print(f"Filtered data size:   {filtered_shape}")
        
        # Calculate size reduction
        original_size = np.prod(original_shape) * 4  # float32 = 4 bytes
        filtered_size = np.prod(filtered_shape) * 4
        size_reduction = (original_size - filtered_size) / original_size
        
        print(f"Memory reduction:     {size_reduction:.1%}")
        print(f"Valid RX range:       RX{min(valid_rx_indices)}-RX{max(valid_rx_indices)}")
        print("-" * 50)
    
    def process_area_frequency(self, area: int, freq: float) -> Optional[Dict[str, np.ndarray]]:
        """Process ray data filtering for specific area/frequency combination
        
        Args:
            area: Area index
            freq: Frequency in GHz
            
        Returns:
            Filtered ray data dictionary or None if processing failed
        """
        
        print(f"\nProcessing Area{area}_{freq}GHz...")
        
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
        
        # Create RX mask
        valid_mask, valid_rx_indices = self.create_rx_mask(ray_data['rx_indices'], validity_map)
        
        # Check if any RX would be filtered out
        original_count = len(ray_data['rx_indices'])
        filtered_count = len(valid_rx_indices)
        
        if filtered_count == 0:
            print(f"Warning: All RX would be filtered out for Area{area}_{freq}GHz, skipping...")
            return None
        
        # Filter ray data
        filtered_ray_data = self.filter_ray_data(ray_data, valid_mask)
        
        # Save filtered data
        npz_path, info_path = self.save_filtered_data(filtered_ray_data, area, freq, 
                                                     original_count, filtered_count)
        
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
