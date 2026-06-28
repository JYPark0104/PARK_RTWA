"""Directional Beam Gain Test with Complete Patterns

빔 이득 계산 - 완전 구현:
1. AE Pattern: 3GPP TR 38.901 2차 가우시안 (60°/60° beamwidth)
2. Array Pattern: 2D 크로네커 DFT + Sionna column-major transpose
3. Orientation: P1A PADP 각도 역변환 (0,0,0 → 246,3,0)

Total Gain = Element Factor × Array Factor

=== Core Principles ===

1. Pattern Multiplication (패턴 곱셈의 원리)
   - Total Gain = Element Factor × Array Factor
   - Element Factor (EF): 개별 AE의 방사 패턴 (3GPP TR 38.901)
   - Array Factor (AF): 배열 빔포밍 (DFT codebook × steering vector)
   - Scan Loss: 큰 각도로 빔을 조향하면 EF가 감소

2. Element Pattern (3GPP TR 38.901)
   - Vertical/Horizontal 3dB beamwidth: 60°/60°
   - 2차 가우시안: A(theta) = -12 * (theta / theta_3dB)^2
   - Boresight (theta=0°): EF ≈ 1.0 (최대)
   - Large angle (theta=60°+): EF << 1.0 (scan loss)

3. Array Pattern (2D Kronecker DFT)
   - 4×4 subarray, K=2 oversample → 64 beams
   - Sionna column-major flattening → transpose permutation 필요
   - DFT row-major [0,1,2,3...] → Physical column-major [0,4,8,12,1,5,9,13,...]

4. BS Orientation (P1A → P1O)
   - P1A: BS Orientation (0, 0, 0) → PADP in GCS
   - P1O: BS Orientation (246°, 3°↓, 0°) assumed
   - Correction: Apply INVERSE rotation to P1A PADP
   - GCS angles → LCS angles via R^(-1) = R^T

5. Verification Result
   - Without EF: Beam 27 (큰 각도) 최고 → P1O와 불일치
   - With EF: Beam 0 (broadside) 최고 → P1O와 일치
   - Reason: Scan loss makes broadside beam optimal

=== Critical Notes ===

- H_mean에는 이미 EF 반영됨 (P1H 생성 시)
- PADP 단일 각도로는 P1O 재현 불가 (H_mean 필요)
- 이 코드는 교육/검증 목적 (EF 효과 확인)

=== P1O Capacity Optimization (환각 주의!) ===

P1O는 통계적 채널 정보(P1F 공분산 + P1H 평균 채널)로 빔을 선택하지만,
결과적으로 PADP 방향과 잘 정렬됨:

- UE 12 Capacity loss: 0.34% (매우 낮음)
- P1O 선택 빔 → PADP 방향 정렬 (검증된 사실)
- Peak direction ≈ PADP direction

**중요**: P1O는 P1A PADP를 직접 사용하지 않음. Weichselberger 통계로
빔을 선택하지만, 물리적 원리상 PADP 방향과 자연스럽게 정렬됨.
"""

import numpy as np
import tensorflow as tf
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ===== Configuration =====

class VerificationConfig:
    """빔 이득 측정 설정"""
    
    def __init__(self):
        # BS Orientation (P1F/P1O 설정값)
        # P1A는 (0,0,0)에서 측정 → P1O는 (246°, 3°↓, 0°) 가정
        # → P1A 각도를 P1O LCS로 역변환하려면 INVERSE rotation 필요
        self.bs_azimuth_deg = 246.0
        self.bs_downtilt_deg = 3.0
        self.bs_roll_deg = 0.0
        
        # 측정 각도 (GCS, radians)
        # UE 12 PADP 각도 (from P1A ray data, GCS)
        self.theta_gcs = np.radians(100.28)
        self.phi_gcs = np.radians(-106.70)
        
        # P1O 빔 할당 (레이어 순서대로 빔 인덱스)
        # 예: 순환 패턴 [63]를 64개 레이어에 반복
        self.p1o_beams = [0] * 64
        
        # 개별 측정 테스트 케이스 (layer_idx, beam_idx)
        # Layer 0의 모든 빔 (0-63) 측정
        self.test_cases = [(0, beam_idx) for beam_idx in range(64)]
        
        # 물리적 정보 출력할 레이어들
        self.layers_to_inspect = [0, 1, 2]
        
        # Steering Vector 좌표계 모드
        # 'absolute': 절대 좌표 (array 원점 기준)
        # 'layer_centered': 상대 좌표 (layer 중심 기준)
        self.coordinate_mode = 'layer_centered'  # P1O와 일치하도록 기본값 설정
        
        # Element pattern 설정
        self.apply_element_pattern = True  # 엘리먼트 패턴 적용
        self.theta_3dB_deg = 60.0  # Vertical 3dB beamwidth
        self.phi_3dB_deg = 60.0    # Horizontal 3dB beamwidth


# ===== Level 0: 기본 유틸리티 함수 =====

def rotation_matrix(alpha_rad, beta_rad, gamma_rad):
    """3GPP TS 38.901 Sec 7.1-4 rotation matrix
    
    Args:
        alpha_rad: Azimuth (bearing) [rad]
        beta_rad: Downtilt (elevation tilt) [rad]
        gamma_rad: Roll [rad]
    
    Returns:
        R: [3, 3] rotation matrix
    """
    ca, sa = np.cos(alpha_rad), np.sin(alpha_rad)
    cb, sb = np.cos(beta_rad), np.sin(beta_rad)
    cg, sg = np.cos(gamma_rad), np.sin(gamma_rad)
    
    R = np.array([
        [ca*cb, ca*sb*sg - sa*cg, ca*sb*cg + sa*sg],
        [sa*cb, sa*sb*sg + ca*cg, sa*sb*cg - ca*sg],
        [-sb,   cb*sg,            cb*cg           ]
    ])
    
    return R


def gcs_to_lcs(theta_gcs_rad, phi_gcs_rad, azimuth_deg, downtilt_deg, roll_deg):
    """GCS angles → LCS angles via rotation
    
    Args:
        theta_gcs_rad, phi_gcs_rad: GCS angles [rad]
        azimuth_deg, downtilt_deg, roll_deg: BS orientation [deg]
    
    Returns:
        (theta_lcs_rad, phi_lcs_rad)
    """
    # Convert orientation to radians
    alpha_rad = np.radians(azimuth_deg)
    beta_rad = np.radians(downtilt_deg)
    gamma_rad = np.radians(roll_deg)
    
    # GCS unit direction vector
    sin_theta = np.sin(theta_gcs_rad)
    cos_theta = np.cos(theta_gcs_rad)
    sin_phi = np.sin(phi_gcs_rad)
    cos_phi = np.cos(phi_gcs_rad)
    
    r_gcs = np.array([
        sin_theta * cos_phi,
        sin_theta * sin_phi,
        cos_theta
    ])
    
    # Rotate to LCS (GCS → LCS는 inverse rotation)
    R = rotation_matrix(alpha_rad, beta_rad, gamma_rad)
    R_inv = R.T  # Orthogonal matrix: R^-1 = R^T
    r_lcs = R_inv @ r_gcs
    
    # LCS angles
    x_lcs, y_lcs, z_lcs = r_lcs
    
    # Clamp to avoid numerical errors
    z_lcs_clamped = np.clip(z_lcs, -1.0, 1.0)
    theta_lcs_rad = np.arccos(z_lcs_clamped)
    
    phi_lcs_rad = np.arctan2(y_lcs, x_lcs)
    
    return (theta_lcs_rad, phi_lcs_rad)


def get_transpose_permutation_4x4():
    """4×4 transpose permutation (row-major ↔ column-major)
    
    DFT codebook은 row-major 가정 (row=z, col=y)
    Physical layout은 column-major (col=y, row=z)
    
    Returns:
        [16] permutation array for transpose
    """
    perm = []
    for i in range(4):
        for j in range(4):
            src_idx = i * 4 + j  # Row-major: (row i, col j)
            dst_idx = j * 4 + i  # Column-major: (col j, row i)
            perm.append(dst_idx)
    return np.array(perm)
    # Result: [0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15]


def layer_to_ae_indices(layer_idx):
    """Layer index → 1024 AE 중 해당 16 AE의 indices
    
    Args:
        layer_idx: 0~63
    
    Returns:
        ae_start, ae_end
    """
    # blkdiag 구조상 연속된 16 AE 할당
    ae_start = layer_idx * 16
    ae_end = ae_start + 16
    return ae_start, ae_end


def ae_index_to_physical_coords(ae_idx):
    """AE index (0~1023) → (panel_row, panel_col, elem_row, elem_col)
    
    Sionna flattening order: COLUMN-MAJOR
    - PanelArray: for j (col) for i (row) → panel_row slowly varying
    - AntennaPanel: idx = i + j*num_rows → elem_row slowly varying
    
    Args:
        ae_idx: 0~1023
    
    Returns:
        (panel_row, panel_col, elem_row, elem_col)
    """
    panel_major = ae_idx // 16
    elem_in_panel = ae_idx % 16
    
    # Column-major order
    panel_row = panel_major % 8    # slowly varying
    panel_col = panel_major // 8   # fast varying
    
    # Column-major order
    elem_row = elem_in_panel % 4   # slowly varying
    elem_col = elem_in_panel // 4  # fast varying
    
    return panel_row, panel_col, elem_row, elem_col


# ===== Level 1: DFT 코드북 =====

class DFTCodebook:
    """Layer-level DFT codebook (4×4 subarray, 64 beams)"""
    
    @staticmethod
    def generate_1d_dft_tf(N: int, K: int) -> tf.Tensor:
        """1D DFT codebook
        
        [F_{N,K}]_{i,j} = (1/√N) exp(-j 2π ij / (NK))
        
        Args:
            N: Antennas per dimension
            K: Oversample factor
        
        Returns:
            F: [N, NK] complex64
        """
        i = tf.cast(tf.range(N)[:, None], tf.float32)
        j = tf.cast(tf.range(N * K)[None, :], tf.float32)
        pi = tf.constant(np.pi, dtype=tf.float32)
        phase = -2.0 * pi * i * j / float(N * K)
        sqrt_N = tf.sqrt(tf.constant(float(N), dtype=tf.float32))
        sqrt_N_complex = tf.cast(sqrt_N, tf.complex64)
        return tf.exp(tf.complex(0.0, phase)) / sqrt_N_complex
    
    @staticmethod
    def generate_2d_dft_codebook_tf(ant_per_dim: int, oversample: int) -> tf.Tensor:
        """2D DFT codebook
        
        F = F_{N,K} ⊗ F_{N,K}
        
        Args:
            ant_per_dim: 1D antennas (N)
            oversample: Oversample factor (K)
        
        Returns:
            F: [ant_per_dim², (ant_per_dim*oversample)²] complex64
        
        Example:
            Layer: (4, 2) → [16, 64]
        """
        F_1d = DFTCodebook.generate_1d_dft_tf(ant_per_dim, oversample)
        
        # Kronecker product
        n_ant = ant_per_dim
        n_beams = ant_per_dim * oversample
        
        F_kron = tf.einsum('ij,kl->ikjl', F_1d, F_1d)
        F_kron = tf.reshape(F_kron, [n_ant**2, n_beams**2])
        
        return F_kron


# ===== Level 2: 물리적 안테나 위치 및 스티어링 벡터 =====

def get_ae_physical_position(panel_row, panel_col, elem_row, elem_col):
    """물리적 위치 (non-centered wavelengths)
    
    Panel: 0~7
    Element: 0~3
    
    Returns:
        (y_pos, z_pos) in wavelengths
    """
    # Panel spacing = 2.5λ
    # Element spacing = 0.5λ
    # Non-centered: 첫 번째 AE가 원점 (0, 0)
    
    y_panel = panel_col * 2.5
    z_panel = panel_row * 2.5
    
    y_elem = elem_col * 0.5
    z_elem = elem_row * 0.5
    
    y_total = y_panel + y_elem
    z_total = z_panel + z_elem
    
    return y_total, z_total


def get_layer_center(layer_idx):
    """Layer의 16개 AE 중심 좌표 계산
    
    Args:
        layer_idx: 0~63
    
    Returns:
        (y_center, z_center) in wavelengths
    """
    ae_start, ae_end = layer_to_ae_indices(layer_idx)
    
    # 16개 AE의 평균 위치 계산
    y_sum = 0.0
    z_sum = 0.0
    
    for ae_idx in range(ae_start, ae_end):
        p_row, p_col, e_row, e_col = ae_index_to_physical_coords(ae_idx)
        y, z = get_ae_physical_position(p_row, p_col, e_row, e_col)
        y_sum += y
        z_sum += z
    
    y_center = y_sum / 16
    z_center = z_sum / 16
    
    return y_center, z_center


def element_pattern_3gpp(theta_rad, phi_rad, 
                         theta_3dB_deg=60.0, 
                         phi_3dB_deg=60.0,
                         A_max_dB=25.0):
    """3GPP TR 38.901 안테나 엘리먼트 패턴
    
    G_E(theta, phi) = -min[A_EV(theta) + A_EH(phi), A_max]
    
    A_EV(theta) = -min[12 * (theta / theta_3dB)^2, SLA_V]
    A_EH(phi) = -min[12 * (phi / phi_3dB)^2, SLA_H]
    
    Args:
        theta_rad: Elevation angle from boresight [rad]
        phi_rad: Azimuth angle from boresight [rad]
        theta_3dB_deg: Vertical 3dB beamwidth (default 60°)
        phi_3dB_deg: Horizontal 3dB beamwidth (default 60°)
        A_max_dB: Maximum attenuation (default 25 dB)
    
    Returns:
        gain_linear: Element gain (linear scale)
    """
    theta_deg = np.degrees(np.abs(theta_rad))
    phi_deg = np.degrees(np.abs(phi_rad))
    
    # Vertical pattern: -min[12 * (theta / theta_3dB)^2, 30]
    A_EV = 12 * (theta_deg / theta_3dB_deg)**2
    A_EV = np.minimum(A_EV, 30.0)  # SLA_V = 30 dB
    
    # Horizontal pattern: -min[12 * (phi / phi_3dB)^2, 30]
    A_EH = 12 * (phi_deg / phi_3dB_deg)**2
    A_EH = np.minimum(A_EH, 30.0)  # SLA_H = 30 dB
    
    # Combined attenuation (dB)
    A_E_dB = -(A_EV + A_EH)
    A_E_dB = np.maximum(A_E_dB, -A_max_dB)
    
    # Convert to linear scale
    gain_linear = 10**(A_E_dB / 10)
    
    return gain_linear


def generate_1024_steering_vector(theta_lcs, phi_lcs, layer_idx=None, coordinate_mode='absolute'):
    """1024 AE 전체 steering vector (LCS 기준)
    
    Args:
        theta_lcs: Zenith angle in LCS (radians)
        phi_lcs: Azimuth angle in LCS (radians)
        layer_idx: Layer index (0-63), required for 'layer_centered' mode
        coordinate_mode: 'absolute' or 'layer_centered'
            - 'absolute': 절대 좌표 (array 원점 기준)
            - 'layer_centered': 상대 좌표 (layer 중심 기준)
    
    Returns:
        [1024] complex array
    
    Note:
        DFT 코드북과 steering vector 모두 LCS에서 정의됨
        GCS → LCS 변환은 호출 전에 수행되어야 함
    """
    k = 2 * np.pi  # wavenumber (λ=1)
    sv = np.zeros(1024, dtype=complex)
    
    # Layer-centered mode일 경우 중심 좌표 계산
    if coordinate_mode == 'layer_centered':
        if layer_idx is None:
            raise ValueError("layer_idx is required for 'layer_centered' mode")
        y_center, z_center = get_layer_center(layer_idx)
    else:
        y_center, z_center = 0.0, 0.0  # absolute mode는 원점 기준
    
    for ae_idx in range(1024):
        # AE index → 물리적 좌표
        p_row, p_col, e_row, e_col = ae_index_to_physical_coords(ae_idx)
        y_abs, z_abs = get_ae_physical_position(p_row, p_col, e_row, e_col)
        
        # 상대 좌표로 변환
        y_rel = y_abs - y_center
        z_rel = z_abs - z_center
        
        # Phase (exp(-jkr)) in LCS
        # P1O convention: BS points DOWNWARD (Beam 0 = nadir)
        # Standard steering: theta=0°=zenith, P1O DFT: theta=180°=zenith
        # → Z direction flipped
        phase = k * (y_rel * np.sin(theta_lcs) * np.sin(phi_lcs) -
                     z_rel * np.sin(theta_lcs) * np.cos(phi_lcs))
        
        sv[ae_idx] = np.exp(-1j * phase)
    
    return sv


# ===== Level 3: 빔 이득 계산 =====

class BeamGainCalculator:
    """물리적 안테나 배열 기반 빔 이득 측정
    
    DFT 코드북과 steering vector 모두 LCS에서 정의
    BS orientation을 통해 GCS → LCS 변환 수행
    """
    
    def __init__(self, bs_orientation, coordinate_mode='absolute', 
                 apply_transpose=True, apply_element_pattern=True,
                 theta_3dB_deg=60.0, phi_3dB_deg=60.0):
        """초기화
        
        Args:
            bs_orientation: BS orientation (GCS 파라미터)
                {'azimuth_deg', 'downtilt_deg', 'roll_deg'}
            coordinate_mode: 'absolute' or 'layer_centered'
            apply_transpose: DFT codeword transpose 적용 여부
                True: DFT (row-major) → Physical (column-major) 변환
                False: DFT 그대로 사용 (이전 동작)
            apply_element_pattern: 3GPP 엘리먼트 패턴 적용 여부
            theta_3dB_deg: Vertical 3dB beamwidth
            phi_3dB_deg: Horizontal 3dB beamwidth
        """
        # DFT 코드북 (LCS 기준, row-major 가정)
        self.F_codebook = DFTCodebook.generate_2d_dft_codebook_tf(4, 2).numpy()  # [16, 64]
        
        # BS orientation (GCS → LCS 변환용)
        self.bs_orientation = bs_orientation
        
        # Coordinate mode
        self.coordinate_mode = coordinate_mode
        
        # Transpose permutation (DFT → Physical)
        self.apply_transpose = apply_transpose
        if apply_transpose:
            self.transpose_perm = get_transpose_permutation_4x4()
        else:
            self.transpose_perm = None
        
        # Element pattern parameters
        self.apply_element_pattern = apply_element_pattern
        self.theta_3dB_deg = theta_3dB_deg
        self.phi_3dB_deg = phi_3dB_deg
    
    def compute_beam_gain(self, layer_idx: int, beam_idx: int, 
                         theta_gcs: float, phi_gcs: float) -> float:
        """빔 이득 측정
        
        Args:
            layer_idx: Layer index (0-63)
            beam_idx: Beam index (0-63)
            theta_gcs, phi_gcs: Direction in GCS [rad]
        
        Returns:
            gain: |w^H h|^2
        
        Process:
            1. GCS → LCS 변환
            2. LCS steering vector 생성
            3. 빔 이득 계산
        """
        # 1. GCS → LCS 역변환 (P1A는 (0,0,0), P1O는 (246,3,0) 가정)
        # P1A 각도를 P1O LCS로 맞추려면 INVERSE rotation
        theta_lcs, phi_lcs = gcs_to_lcs(
            theta_gcs, phi_gcs,
            -self.bs_orientation['azimuth_deg'],   # Reverse
            -self.bs_orientation['downtilt_deg'],  # Reverse
            -self.bs_orientation['roll_deg']       # Reverse
        )
        
        # 2. Steering vector (LCS 기준, Physical layout)
        ae_start, ae_end = layer_to_ae_indices(layer_idx)
        sv_1024 = generate_1024_steering_vector(
            theta_lcs, phi_lcs,
            layer_idx=layer_idx,
            coordinate_mode=self.coordinate_mode
        )
        sv_layer = sv_1024[ae_start:ae_end]  # [16] Column-major order
        
        # 3. DFT beam vector (LCS 기준, Row-major 가정)
        w_beam = self.F_codebook[:, beam_idx]  # [16]
        
        # 4. Transpose 적용 (DFT row-major → Physical column-major)
        if self.apply_transpose:
            w_beam = w_beam[self.transpose_perm]
        
        # 5. Array Factor (DFT beamforming)
        array_factor = np.abs(np.vdot(w_beam, sv_layer))**2
        
        # 6. Element Factor (3GPP TR 38.901)
        if self.apply_element_pattern:
            element_gain = element_pattern_3gpp(
                theta_lcs, phi_lcs,
                self.theta_3dB_deg, self.phi_3dB_deg
            )
            # Total Gain = EF × AF
            gain = element_gain * array_factor
        else:
            gain = array_factor
        
        return float(gain)
    
    def measure_beam_assignment(self, beam_assignment: list, 
                                theta_gcs: float, phi_gcs: float) -> list:
        """빔 할당의 각 레이어별 이득 측정
        
        Args:
            beam_assignment: [64] 레이어별 빔 인덱스
            theta_gcs, phi_gcs: Direction in GCS [rad]
        
        Returns:
            List of measurement results per layer
        """
        results = []
        
        for layer_idx in range(64):
            beam_idx = beam_assignment[layer_idx]
            gain = self.compute_beam_gain(layer_idx, beam_idx, theta_gcs, phi_gcs)
            
            panel_row = layer_idx // 8
            panel_col = layer_idx % 8
            
            results.append({
                'layer_idx': layer_idx,
                'beam_idx': beam_idx,
                'gain': gain,
                'panel_row': panel_row,
                'panel_col': panel_col
            })
        
        return results


# ===== Level 4: 출력 함수 =====

def print_measurement_results(results: list, theta: float, phi: float):
    """측정 결과 출력
    
    Args:
        results: measure_beam_assignment() 결과
        theta, phi: 측정 방향 [rad]
    """
    elev_deg = 90.0 - np.degrees(theta)
    az_deg = np.degrees(phi)
    
    print("\n" + "="*80)
    print(f"Beam Assignment Measurement")
    print(f"Direction: elev={elev_deg:.1f}°, az={az_deg:.1f}°")
    print("="*80)
    
    # Summary
    total_gain = sum(r['gain'] for r in results)
    print(f"\nSummary:")
    print(f"  Total gain across all 64 layers: {total_gain:.2f}")
    print(f"  Average gain per layer: {total_gain/64:.2f}")
    
    # Top 10 layers by gain
    sorted_by_gain = sorted(results, key=lambda x: x['gain'], reverse=True)
    
    print(f"\nTop 10 layers by gain:")
    print("-"*80)
    for i, r in enumerate(sorted_by_gain[:10], 1):
        print(f"  {i:2d}. Layer {r['layer_idx']:2d} Panel({r['panel_row']},{r['panel_col']}): "
              f"Beam {r['beam_idx']:2d}, gain = {r['gain']:6.2f}")
    
    # Bottom 10 layers by gain
    print(f"\nBottom 10 layers by gain:")
    print("-"*80)
    for i, r in enumerate(sorted_by_gain[-10:], 1):
        print(f"  {i:2d}. Layer {r['layer_idx']:2d} Panel({r['panel_row']},{r['panel_col']}): "
              f"Beam {r['beam_idx']:2d}, gain = {r['gain']:6.2f}")
    
    print("="*80)


def print_layer_physical_info(layer_indices: list):
    """레이어들의 물리적 위치 정보 출력
    
    Args:
        layer_indices: 출력할 레이어 인덱스 리스트
    """
    print("\n" + "="*70)
    print("Layer Physical Information")
    print("="*70)
    
    for layer_idx in layer_indices:
        panel_row, panel_col = layer_idx // 8, layer_idx % 8
        ae_start, ae_end = layer_to_ae_indices(layer_idx)
        
        # Panel center (approximate)
        panel_center_y = panel_col * 2.5 + 1.5 * 0.5
        panel_center_z = panel_row * 2.5 + 1.5 * 0.5
        
        print(f"\nLayer {layer_idx}:")
        print(f"  Panel: ({panel_row}, {panel_col})")
        print(f"  AE range: [{ae_start}:{ae_end}]")
        print(f"  Panel center (approx): y={panel_center_y:.2f}λ, z={panel_center_z:.2f}λ")
        
        # First and last AE positions
        p_row_f, p_col_f, e_row_f, e_col_f = ae_index_to_physical_coords(ae_start)
        y_f, z_f = get_ae_physical_position(p_row_f, p_col_f, e_row_f, e_col_f)
        
        ae_last = ae_end - 1
        p_row_l, p_col_l, e_row_l, e_col_l = ae_index_to_physical_coords(ae_last)
        y_l, z_l = get_ae_physical_position(p_row_l, p_col_l, e_row_l, e_col_l)
        
        print(f"  First AE [{ae_start}]: y={y_f:.2f}λ, z={z_f:.2f}λ")
        print(f"  Last AE [{ae_last}]: y={y_l:.2f}λ, z={z_l:.2f}λ")


# ===== Level 5: 메인 실행 =====

def main():
    """메인 실행"""
    print("\n" + "="*80)
    print(" Physical Array Verification")
    print(" 물리적 안테나 배열 기반 빔 이득 측정 (BS Orientation 반영)")
    print("="*80)
    
    # Configuration
    config = VerificationConfig()
    
    # BS Orientation
    print(f"\nBS Orientation (GCS):")
    print(f"  Azimuth:  {config.bs_azimuth_deg:.1f}°")
    print(f"  Downtilt: {config.bs_downtilt_deg:.1f}°")
    print(f"  Roll:     {config.bs_roll_deg:.1f}°")
    
    # Measurement direction (GCS)
    elev_gcs_deg = 90.0 - np.degrees(config.theta_gcs)
    az_gcs_deg = np.degrees(config.phi_gcs)
    print(f"\nMeasurement direction (GCS):")
    print(f"  Elevation: {elev_gcs_deg:.2f}° (zenith={np.degrees(config.theta_gcs):.2f}°)")
    print(f"  Azimuth:   {az_gcs_deg:.2f}°")
    
    # Convert to LCS for display
    bs_orientation = {
        'azimuth_deg': config.bs_azimuth_deg,
        'downtilt_deg': config.bs_downtilt_deg,
        'roll_deg': config.bs_roll_deg
    }
    # P1A 각도를 P1O LCS로 역변환
    theta_lcs, phi_lcs = gcs_to_lcs(
        config.theta_gcs, config.phi_gcs,
        -config.bs_azimuth_deg,   # Reverse
        -config.bs_downtilt_deg,  # Reverse
        -config.bs_roll_deg       # Reverse
    )
    elev_lcs_deg = 90.0 - np.degrees(theta_lcs)
    az_lcs_deg = np.degrees(phi_lcs)
    print(f"\nMeasurement direction (LCS):")
    print(f"  Elevation: {elev_lcs_deg:.2f}° (zenith={np.degrees(theta_lcs):.2f}°)")
    print(f"  Azimuth:   {az_lcs_deg:.2f}°")
    
    print(f"\nCoordinate mode: {config.coordinate_mode}")
    print(f"Element pattern: {'Enabled' if config.apply_element_pattern else 'Disabled'}")
    if config.apply_element_pattern:
        print(f"  Vertical 3dB BW: {config.theta_3dB_deg}°")
        print(f"  Horizontal 3dB BW: {config.phi_3dB_deg}°")
    
    # ===== Transpose 비교 실험 =====
    print("\n" + "="*80)
    print("TRANSPOSE PERMUTATION TEST")
    print("="*80)
    
    # Before transpose (이전 동작)
    print("\n[1] WITHOUT Transpose (이전 동작)")
    calculator_before = BeamGainCalculator(
        bs_orientation, 
        coordinate_mode=config.coordinate_mode,
        apply_transpose=False,
        apply_element_pattern=config.apply_element_pattern,
        theta_3dB_deg=config.theta_3dB_deg,
        phi_3dB_deg=config.phi_3dB_deg
    )
    
    gains_before = []
    for beam_idx in range(64):
        gain = calculator_before.compute_beam_gain(0, beam_idx, config.theta_gcs, config.phi_gcs)
        gains_before.append((beam_idx, gain))
    
    gains_before_sorted = sorted(gains_before, key=lambda x: x[1], reverse=True)
    best_before = gains_before_sorted[0]
    
    print(f"  Best beam: Beam {best_before[0]}, gain={best_before[1]:.4f}")
    print(f"  Top 5: ", end="")
    for i in range(5):
        b, g = gains_before_sorted[i]
        print(f"Beam {b}({g:.4f})", end="  ")
    print()
    
    # After transpose (수정된 동작)
    print("\n[2] WITH Transpose (수정 후)")
    calculator_after = BeamGainCalculator(
        bs_orientation,
        coordinate_mode=config.coordinate_mode,
        apply_transpose=True,
        apply_element_pattern=config.apply_element_pattern,
        theta_3dB_deg=config.theta_3dB_deg,
        phi_3dB_deg=config.phi_3dB_deg
    )
    
    gains_after = []
    for beam_idx in range(64):
        gain = calculator_after.compute_beam_gain(0, beam_idx, config.theta_gcs, config.phi_gcs)
        gains_after.append((beam_idx, gain))
    
    gains_after_sorted = sorted(gains_after, key=lambda x: x[1], reverse=True)
    best_after = gains_after_sorted[0]
    
    print(f"  Best beam: Beam {best_after[0]}, gain={best_after[1]:.4f}")
    print(f"  Top 5: ", end="")
    for i in range(5):
        b, g = gains_after_sorted[i]
        print(f"Beam {b}({g:.4f})", end="  ")
    print()
    
    # P1O 비교
    print("\n[3] P1O Selection")
    print(f"  P1O Beam: Beam 0 (UE 12)")
    
    # 결과 분석
    print("\n" + "-"*80)
    print("ANALYSIS")
    print("-"*80)
    print(f"Before transpose: Beam {best_before[0]} (gain={best_before[1]:.4f})")
    print(f"After transpose:  Beam {best_after[0]} (gain={best_after[1]:.4f})")
    print(f"P1O selection:    Beam 0")
    
    if best_after[0] == 0:
        print("\n✓ SUCCESS: Transpose 적용 후 P1O와 일치!")
        print("  → Indexing order가 근본 원인")
    elif best_after[0] < best_before[0]:
        print(f"\n△ PARTIAL: Beam index 감소 ({best_before[0]} → {best_after[0]})")
        print("  → Transpose가 영향 있지만 완전히 해결은 안됨")
    else:
        print("\n✗ NO EFFECT: Transpose 영향 없음")
        print("  → 다른 원인 존재")
    
    print("="*80)
    
    # 기존 개별 빔 측정 (transpose 적용된 calculator 사용)
    calculator = calculator_after  # Use corrected calculator
    
    print(f"\n\nDetailed measurements with transpose...")
    print("-"*80)
    
    results = []
    for layer_idx, beam_idx in config.test_cases:
        gain = calculator.compute_beam_gain(
            layer_idx, beam_idx, config.theta_gcs, config.phi_gcs
        )
        panel_row, panel_col = layer_idx // 8, layer_idx % 8
        results.append({
            'layer_idx': layer_idx,
            'beam_idx': beam_idx,
            'gain': gain,
            'panel_row': panel_row,
            'panel_col': panel_col
        })
    
    # Gain 기준 내림차순 정렬
    results_sorted = sorted(results, key=lambda x: x['gain'], reverse=True)
    
    # 결과 출력
    print("\nAll beams sorted by gain (descending):")
    print("-"*80)
    for i, r in enumerate(results_sorted, 1):
        print(f"  {i:2d}. Layer {r['layer_idx']:2d} Panel({r['panel_row']},{r['panel_col']}), "
              f"Beam {r['beam_idx']:2d}: gain = {r['gain']:.4f}")
    
    print("\n" + "="*80)
    print("Verification Complete")
    print("="*80)


if __name__ == "__main__":
    main()
