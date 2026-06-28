# ======================================================================
# P1I_Weichsel_Chunk_2510v1.py
# P1I: Weichselberger SU-MIMO 채널 통계 파라미터 청크 저장
# 
# === 최상위 목적 ===
# P1F (Marginal CCM), P1G (Coupling Matrix), P1H (Mean Channel) 데이터를
# RX별 개별 파일에서 100개 RX 단위 청크 파일로 재구성
# 
# === [중요] 다운링크/업링크 혼동 방지 명명 규칙 ===
# **코드 전체에서 tx/rx 표기 금지, BS/UE 표기로 통일**
# 
# 이유:
#   - 입력 데이터는 다운링크 기준으로 생성됨 (TX=BS, RX=UE)
#   - 향후 업링크 실험에서는 반대가 됨 (TX=UE, RX=BS)
#   - tx/rx 표기는 링크 방향에 따라 의미가 바뀌므로 혼란 유발
# 
# 해결책:
#   - 안테나 차원: n_ant_bs, n_ant_ue (링크 방향 무관)
#   - 고유벡터: U_BS, U_UE (링크 방향 무관)
#   - 고유값: Lambda_BS, Lambda_UE (링크 방향 무관)
#   - 샘플 인덱스: ue_indices, n_UEs (UE 위치 인덱스)
# 
# === CCM (Covariance Matrix) 정의 방식 ===
# P1F/P1G에서 사용한 정의 (중요!):
#   R_tx = E[H^H H]  (transmit correlation)
#   R_rx = E[H H^H]  (receive correlation)
# 
# 이 정의 방식에 따라:
#   - 다운링크 (H_DL): R_tx = R_BS, R_rx = R_UE
#   - 업링크 (H_UL = H_DL^H):
#     * R_tx,UL = E[H_UL^H H_UL] = E[H_DL H_DL^H] = R_rx,DL = R_UE
#     * R_rx,UL = E[H_UL H_UL^H] = E[H_DL^H H_DL] = R_tx,DL = R_BS
#   - 따라서 R_BS, R_UE는 링크 방향 무관하게 사용 가능
# 
# [주의] 다른 정의 방식:
#   일부 논문에서는 R_tx = E[H^T H^*]를 사용 (transpose-conjugate 순서 다름)
#   이 경우 추가 변환 필요: R_tx_alt = R_rx.T, R_rx_alt = R_tx.T
# 
# === 업링크 실험 시 데이터 변환 규칙 ===
# 저장된 데이터를 업링크 실험에 사용할 때 (H_UL = H_DL^H):
#   1. R_BS, R_UE: 그대로 사용 (위의 정의 방식 덕분)
#   2. U_BS, U_UE: 그대로 사용 (고유벡터 행렬)
#   3. Lambda_BS, Lambda_UE: 그대로 사용 (고유값)
#   4. H_mean: H_mean.conj().T 필요 (Hermitian conjugate)
#   5. Omega: Omega.T 필요 (실수 행렬이므로 transpose만)
# 
# 채널 생성 공식:
#   - 다운링크: H_DL = U_UE @ (Omega^(1/2) o H_iid) @ U_BS^H
#   - 업링크: H_UL = H_DL^H = U_BS @ (Omega.T^(1/2) o H_iid^H) @ U_UE^H
# 
# 이 규칙은 P1I 이후의 모든 코드에서도 동일하게 적용되어야 함!
# 
# === 채널 모델 분류 (데이터 조합별) ===
# 
# 1. **P1H only (Deterministic)**
#    - 결정론적 채널 (Deterministic MIMO)
#    - H = H_mean (고정)
#    - 사용 예: 정적 환경, LoS dominant
# 
# 2. **P1F only (Stochastic Rayleigh)**
#    - 확률적 Rayleigh 채널 (NLoS only)
#    - 공간 상관 모델:
#      • P1G 있음: Jointly-correlated (Weichselberger) Rayleigh
#        H = U_rx @ Ω^(1/2) ⊙ H_iid @ U_tx^H
#      • P1G 없음: Separately-correlated (Kronecker) Rayleigh
#        H = R_UE^(1/2) @ H_iid @ R_BS^(1/2)
# 
# 3. **P1H & P1F (Stochastic Rician)**
#    - 확률적 Rician 채널 (LoS + NLoS)
#    - H = H_mean + H_stochastic
#    - 공간 상관 모델:
#      • P1G 있음: Jointly-correlated (Weichselberger) Rician
#      • P1G 없음: Separately-correlated (Kronecker) Rician
# 
# === 데이터 의존성 ===
# - P1G ⊂ P1F: P1G는 P1F의 고유벡터 기반으로 생성됨
# - 처리 기준: P1F ∪ P1H (합집합)
# - P1G는 선택적 (없으면 Kronecker 모델 가정)
# 
# === 주요 구성 요소 ===
# - P1I_Config: P1F/P1G/P1H 데이터 스캔 + 청크 저장 경로 설정
# - ChunkDataAggregator: 100개 UEs 단위 데이터 집계 + 메타데이터 보강
# - ChunkDataManager: 청크 파일 저장/로딩/검증 관리
# 
# === 입력 데이터 ===
# P1F (Marginal CCM - 확률적 성분):
#   - R_BS: [n_ant_bs, n_ant_bs] BS marginal covariance
#   - R_UE: [n_ant_ue, n_ant_ue] UE marginal covariance
#   - metadata, validation
# 
# P1G (Coupling Matrix - jointly-correlated 성분, 선택적):
#   - U_BS, U_UE: [n_ant_bs, n_ant_bs], [n_ant_ue, n_ant_ue] 고유벡터 행렬
#   - Lambda_BS, Lambda_UE: [n_ant_bs], [n_ant_ue] 고유값
#   - Omega: [n_ant_ue, n_ant_bs] 커플링 행렬
#   - metadata, coupling_stats, evd_validation
# 
# P1H (Mean Channel - 결정론적 LoS 성분):
#   - H_mean: [n_ant_ue, n_ant_bs] 평균 채널
#   - metadata
# 
# === 출력 청크 구조 ===
# npz 파일: Area{area}_{freq}GHz_Weichsel_Chunk_{idx}_UE{start}-{end}.npz
#   - idx: 청크 순번 (자릿수는 총 청크 개수로 자동 결정, 예: 01, 02, ..., 11)
# {
#   'ue_indices': [100] UE 위치 인덱스 배열
#   
#   'P1F': {  # 확률적 성분 (선택적)
#       'R_BS': [100, n_ant_bs, n_ant_bs],
#       'R_UE': [100, n_ant_ue, n_ant_ue],
#       'has_stochastic': [100] bool,  # P1F 존재 여부
#       'metadata': [100] (object),
#       'validation': [100] (object)
#   },
#   
#   'P1G': {  # Jointly-correlated 성분 (선택적, P1F ⊂ P1G)
#       'U_BS': [100, n_ant_bs, n_ant_bs],
#       'U_UE': [100, n_ant_ue, n_ant_ue],
#       'Lambda_BS': [100, n_ant_bs],
#       'Lambda_UE': [100, n_ant_ue],
#       'Omega': [100, n_ant_ue, n_ant_bs],
#       'has_coupling': [100] bool,  # P1G 존재 여부
#       'metadata': [100] (object),
#       'coupling_stats': [100] (object),
#       'evd_validation_bs': [100] (object),
#       'evd_validation_ue': [100] (object)
#   },
#   
#   'P1H': {  # 결정론적 LoS 성분 (선택적)
#       'H_mean': [100, n_ant_ue, n_ant_bs],
#       'has_los': [100] bool,  # P1H 존재 여부
#       'metadata': [100] (object)
#   },
#   
#   'enhanced_metadata': [100] (object)  # 채널 모델 분류 + 통계
#       - 'channel_model': 'Deterministic' / 'Rayleigh-Kronecker' / 
#                          'Rayleigh-Weichselberger' / 'Rician-Kronecker' / 
#                          'Rician-Weichselberger'
#       - 'pathloss_dB': Pathloss [dB] = -10*log10(||H_mean||_F^2 + sum(Omega))
#       - 'H_mean_power': ||H_mean||_F^2 (LoS 전력)
#       - 'Omega_sum': sum(Omega) (NLoS 평균 전력)
#       - 'total_power_gain': 총 채널 전력 이득
#       - 채널 통계치 (trace, coupling, rank, K-factor 등)
#   
#   'chunk_metadata': {
#       'area_idx': int,
#       'freq_ghz': float,
#       'ue_start': int,
#       'ue_end': int,
#       'n_UEs': int
#   }
# }
#
# === 주요 수정 이력 ===
# [251015] P1I 신규 작성: Weichselberger SU-MIMO 채널 통계 청크 저장
# 1. P1F ∪ P1H 합집합 기준 처리
# 2. 채널 모델 자동 분류 (Deterministic/Rayleigh/Rician × Kronecker/Weichselberger)
# 3. 보강된 메타데이터 생성 (trace, coupling, rank, K-factor 등)
# 4. 100개 UEs 위치 단위 청크 생성 + 검증
# 5. 의존성 기반 순서: Config → Aggregator → Manager → main
# 6. **BS/UE 명명 규칙 전면 적용** (다운링크/업링크 혼동 방지)
#    - tx/rx 표기 완전 제거
#    - 모든 변수/키/메타데이터에서 BS/UE 표기로 통일
#    - 업링크 실험 시 H_mean/Omega 변환 규칙 명시
#    - 이 규칙은 P1I 이후의 모든 후속 코드에서도 준수되어야 함
#
# [251015] Pathloss 추가
# - enhanced_metadata에 pathloss_dB 포함
# - 공식: Pathloss_dB = -10*log10(||H_mean||_F^2 + sum(Omega))
# - P1H에서 H_mean_fro_sq 로딩 (예전 파일 호환성)
# - H_mean_power, Omega_sum, total_power_gain도 함께 저장
#
# ======================================================================

# ===== SECTION 1: 환경 설정 =====
import os
import time
from datetime import datetime
import numpy as np
import glob
import re
from pathlib import Path

# 주피터/IPython 화면 클리어 지원
try:
    from IPython.display import clear_output
    JUPYTER_AVAILABLE = True
except ImportError:
    JUPYTER_AVAILABLE = False

# ===== SECTION 2: P1I_Config =====
class P1I_Config:
    """P1F/P1G/P1H 데이터 스캔 및 청크 저장 설정"""
    
    def __init__(self):
        # 스크립트의 디렉토리를 기준으로 절대 경로 설정
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 입력 데이터 디렉토리 설정
        self.P1F_INPUT_DIR = os.path.join(script_dir, "P1F_Marginal_CCM_Results")
        self.P1G_INPUT_DIR = os.path.join(script_dir, "P1G_CouplingMat_Results")
        self.P1H_INPUT_DIR = os.path.join(script_dir, "P1H_MeanCh_Results")
        
        # 입력 파일 패턴
        self.P1F_FILE_PATTERN = "Area{area}_{freq}GHz_RX{rx}_Marginal_CCM.npz"
        self.P1G_FILE_PATTERN = "Area{area}_{freq}GHz_RX{rx}_CouplingMat.npz"
        self.P1H_FILE_PATTERN = "Area{area}_{freq}GHz_RX{rx}_MeanCh.npz"
        
        # 청크 저장 설정
        self.P1I_OUTPUT_DIR = os.path.join(script_dir, "P1I_Weichsel_Chunk_Results")
        # 패턴 주의: {idx}는 런타임에 자릿수가 결정됨 (총 청크 개수 기반)
        self.P1I_CHUNK_PATTERN = "Area{area}_{freq}GHz_Weichsel_Chunk_{idx}_UE{start}-{end}.npz"
        self.CHUNK_SIZE = 100  # 청크당 UE 위치 개수
        
        # 개별 파일 정리 설정
        self.CLEANUP_INDIVIDUAL_FILES = False  # 청크 생성 후 개별 파일 삭제 여부 (기본값: 보존)
        
        # 콘솔 출력 제어 설정
        self.ENABLE_SCREEN_CLEAR = True
        self.PROGRESS_CLEAR_INTERVAL = 5  # 5개 청크마다 화면 클리어
        
        # 필터링 설정
        self.target_areas = [1]  # 처리할 area 목록 (None이면 전체)
        self.target_freqs = None  # 처리할 주파수 목록 (None이면 전체)
        
        # 입력 데이터 스캔
        self.detect_input_data()
    
    def detect_input_data(self):
        """P1F/P1G/P1H 데이터 스캔 및 채널 모델 분류
        
        처리 기준: P1F ∪ P1H (합집합)
        - P1G는 P1F의 부분집합 (선택적)
        - P1G 없으면 Kronecker 모델 가정
        """
        
        print("=" * 80)
        print("P1I: SU-MIMO 채널 통계 청크 저장")
        print("=" * 80)
        print()
        
        # 각 데이터셋 스캔
        p1f_combos = self._scan_dataset(self.P1F_INPUT_DIR, self.P1F_FILE_PATTERN, "P1F Marginal CCM")
        p1g_combos = self._scan_dataset(self.P1G_INPUT_DIR, self.P1G_FILE_PATTERN, "P1G Coupling Matrix")
        p1h_combos = self._scan_dataset(self.P1H_INPUT_DIR, self.P1H_FILE_PATTERN, "P1H Mean Channel")
        
        # P1F ∪ P1H 합집합 계산 (처리 대상)
        processing_combos = p1f_combos | p1h_combos
        
        # 정렬하여 리스트로 저장
        self.data_combinations = sorted(list(processing_combos))
        
        # 각 RX의 데이터 존재 여부 저장
        self.p1f_available = {}
        self.p1g_available = {}
        self.p1h_available = {}
        
        for combo in self.data_combinations:
            self.p1f_available[combo] = combo in p1f_combos
            self.p1g_available[combo] = combo in p1g_combos
            self.p1h_available[combo] = combo in p1h_combos
        
        # 채널 모델 분류
        n_deterministic = 0  # P1H only
        n_rayleigh_kronecker = 0  # P1F only, P1G 없음
        n_rayleigh_weichsel = 0  # P1F only, P1G 있음
        n_rician_kronecker = 0  # P1F & P1H, P1G 없음
        n_rician_weichsel = 0  # P1F & P1H, P1G 있음
        
        for combo in self.data_combinations:
            has_p1f = self.p1f_available[combo]
            has_p1g = self.p1g_available[combo]
            has_p1h = self.p1h_available[combo]
            
            if has_p1h and not has_p1f:
                n_deterministic += 1
            elif has_p1f and not has_p1h:
                if has_p1g:
                    n_rayleigh_weichsel += 1
                else:
                    n_rayleigh_kronecker += 1
            elif has_p1f and has_p1h:
                if has_p1g:
                    n_rician_weichsel += 1
                else:
                    n_rician_kronecker += 1
        
        print(f"데이터 스캔 결과:")
        print(f"  - P1F (Marginal CCM, 확률적): {len(p1f_combos)}개 UEs")
        print(f"  - P1G (Coupling Matrix, Weichselberger): {len(p1g_combos)}개 UEs")
        print(f"  - P1H (Mean Channel, LoS): {len(p1h_combos)}개 UEs")
        print(f"  - P1F ∪ P1H (청킹 대상): {len(processing_combos)}개 UEs")
        print()
        print(f"채널 모델 분류:")
        print(f"  1. Deterministic (P1H only): {n_deterministic}개 UEs")
        print(f"  2. Rayleigh-Kronecker (P1F, P1G 없음): {n_rayleigh_kronecker}개 UEs")
        print(f"  3. Rayleigh-Weichselberger (P1F, P1G 있음): {n_rayleigh_weichsel}개 UEs")
        print(f"  4. Rician-Kronecker (P1F & P1H, P1G 없음): {n_rician_kronecker}개 UEs")
        print(f"  5. Rician-Weichselberger (P1F & P1H, P1G 있음): {n_rician_weichsel}개 UEs")
        
        # P1G ⊂ P1F 검증
        p1g_not_in_p1f = p1g_combos - p1f_combos
        if p1g_not_in_p1f:
            print(f"\n[경고] P1G는 P1F 없이 {len(p1g_not_in_p1f)}개 존재 (이론적으로 불가능)")
        
        # Area별로 그룹핑하여 출력
        area_groups = {}
        for combo in self.data_combinations:
            area, freq, rx = combo
            key = f"Area{area}_{freq}GHz"
            if key not in area_groups:
                area_groups[key] = []
            area_groups[key].append(rx)
        
        print(f"\n청킹 대상 상세:")
        for area_freq, ue_list in sorted(area_groups.items()):
            print(f"  - {area_freq}: UE{min(ue_list)}-UE{max(ue_list)} ({len(ue_list)} UEs)")
    
    def _scan_dataset(self, input_dir: str, file_pattern: str, dataset_name: str) -> set:
        """개별 데이터셋 스캔
        
        Args:
            input_dir: 입력 디렉토리
            file_pattern: 파일명 패턴
            dataset_name: 데이터셋 이름 (출력용)
        
        Returns:
            set: (area, freq, ue_idx) 조합 집합
        """
        scan_pattern = f"{input_dir}/{file_pattern.replace('{area}', '*').replace('{freq}', '*').replace('{rx}', '*')}"
        files = glob.glob(scan_pattern)
        
        combinations = set()
        
        # 파일명 패턴 정규식 생성 (확장자 제거)
        pattern_base = file_pattern.split('.')[0]  # .npz 제거
        pattern_regex = pattern_base.replace('{area}', r'(\d+)').replace('{freq}', r'(.+)').replace('{rx}', r'(\d+)')
        
        for file_path in files:
            filename = os.path.basename(file_path)
            # 확장자 제거
            filename_base = filename.rsplit('.', 1)[0]
            
            match = re.match(pattern_regex, filename_base)
            if match:
                area_index = int(match.group(1))
                frequency = float(match.group(2))
                ue_index = int(match.group(3))
                
                # 필터링 적용
                if self.target_areas is not None and area_index not in self.target_areas:
                    continue
                if self.target_freqs is not None and frequency not in self.target_freqs:
                    continue
                
                combinations.add((area_index, frequency, ue_index))
        
        return combinations
    
    def group_by_area_freq(self) -> dict:
        """(Area, Freq)별로 UE 리스트 그룹핑
        
        Returns:
            dict: {(area, freq): [ue1, ue2, ...]}
        """
        groups = {}
        for area, freq, ue_idx in self.data_combinations:
            key = (area, freq)
            if key not in groups:
                groups[key] = []
            groups[key].append(ue_idx)
        
        # UE 인덱스 정렬
        for key in groups:
            groups[key].sort()
        
        return groups

# ===== SECTION 3: ChunkDataAggregator =====
class ChunkDataAggregator:
    """100개 UEs 단위 데이터 집계"""
    
    def __init__(self, config: P1I_Config):
        self.config = config
    
    def create_chunks(self, area: int, freq: float, ue_list: list) -> list:
        """UE 리스트를 청크 단위로 분할
        
        Args:
            area: Area 인덱스
            freq: 주파수 (GHz)
            ue_list: UE 인덱스 리스트 (정렬된 상태)
        
        Returns:
            list: [(ue_start, ue_end, [ue_indices]), ...] 청크 정보 리스트
        """
        chunks = []
        chunk_size = self.config.CHUNK_SIZE
        
        for i in range(0, len(ue_list), chunk_size):
            chunk_ues = ue_list[i:i+chunk_size]
            ue_start = chunk_ues[0]
            ue_end = chunk_ues[-1]
            chunks.append((ue_start, ue_end, chunk_ues))
        
        return chunks
    
    def aggregate_chunk_data(self, area: int, freq: float, ue_indices: list) -> dict:
        """청크에 포함될 UE 데이터 집계 및 채널 모델 분류
        
        [중요] BS/UE 명명 규칙
        이 메서드는 P1F/P1G/P1H 데이터를 읽어서 BS/UE 표기로 통일된 
        청크 데이터를 생성합니다.
        
        변환 규칙:
        - 입력 파일(P1G)은 U_tx/U_rx 키를 사용하지만
        - 출력 데이터는 U_BS/U_UE 키로 변환됨
        - 이는 다운링크/업링크 방향과 무관하게 일관성을 보장함
        
        업링크 실험 시 (H_UL = H_DL^H):
        - R_BS, R_UE: 그대로 사용 (P1F 정의: R_tx=E[H^H H], R_rx=E[H H^H])
        - U_BS, U_UE, Lambda_BS, Lambda_UE: 그대로 사용
        - H_mean: H_mean.conj().T 필요 (Hermitian conjugate)
        - Omega: Omega.T 필요 (실수 행렬이므로 transpose만)
        
        Args:
            area: Area 인덱스
            freq: 주파수 (GHz)
            ue_indices: 청크에 포함될 UE 인덱스 리스트
        
        Returns:
            dict: 청크 데이터 (P1F/P1G/P1H 통합 + 채널 모델 분류)
                  모든 키는 BS/UE 표기 사용
        """
        n_UEs = len(ue_indices)
        
        # 첫 번째 UE로 배열 크기 결정 (P1F 또는 P1H 시도)
        first_ue = ue_indices[0]
        combo_key = (area, freq, first_ue)
        
        n_ant_bs, n_ant_ue = None, None
        if self.config.p1f_available.get(combo_key, False):
            sample_data = self._load_p1f_data(area, freq, first_ue)
            n_ant_bs = sample_data['R_BS'].shape[0]
            n_ant_ue = sample_data['R_UE'].shape[0]
        elif self.config.p1h_available.get(combo_key, False):
            sample_data = self._load_p1h_data(area, freq, first_ue)
            n_ant_ue, n_ant_bs = sample_data['H_mean'].shape
        else:
            raise RuntimeError(f"첫 UE{first_ue}: P1F/P1H 둘 다 없음 (데이터 불일치)")
        
        if n_ant_bs is None or n_ant_ue is None:
            raise RuntimeError(f"첫 UE{first_ue}: 안테나 크기 결정 실패")
        
        # 사전 할당
        chunk_data = {
            'ue_indices': np.array(ue_indices, dtype=np.int32),
            'P1F': {
                'R_BS': np.zeros((n_UEs, n_ant_bs, n_ant_bs), dtype=np.complex64),
                'R_UE': np.zeros((n_UEs, n_ant_ue, n_ant_ue), dtype=np.complex64),
                'has_stochastic': np.zeros(n_UEs, dtype=bool),
                'metadata': np.empty(n_UEs, dtype=object),
                'validation': np.empty(n_UEs, dtype=object)
            },
            'P1G': {
                'U_BS': np.zeros((n_UEs, n_ant_bs, n_ant_bs), dtype=np.complex64),
                'U_UE': np.zeros((n_UEs, n_ant_ue, n_ant_ue), dtype=np.complex64),
                'Lambda_BS': np.zeros((n_UEs, n_ant_bs), dtype=np.float32),
                'Lambda_UE': np.zeros((n_UEs, n_ant_ue), dtype=np.float32),
                'Omega': np.zeros((n_UEs, n_ant_ue, n_ant_bs), dtype=np.float32),
                'has_coupling': np.zeros(n_UEs, dtype=bool),
                'metadata': np.empty(n_UEs, dtype=object),
                'coupling_stats': np.empty(n_UEs, dtype=object),
                'evd_validation_bs': np.empty(n_UEs, dtype=object),
                'evd_validation_ue': np.empty(n_UEs, dtype=object)
            },
            'P1H': {
                'H_mean': np.zeros((n_UEs, n_ant_ue, n_ant_bs), dtype=np.complex64),
                'has_los': np.zeros(n_UEs, dtype=bool),
                'metadata': np.empty(n_UEs, dtype=object)
            },
            'enhanced_metadata': np.empty(n_UEs, dtype=object),
            'chunk_metadata': {
                'area_idx': area,
                'freq_ghz': freq,
                'ue_start': ue_indices[0],
                'ue_end': ue_indices[-1],
                'n_UEs': n_UEs,
                'chunk_size': self.config.CHUNK_SIZE,
                'n_ant_bs': n_ant_bs,
                'n_ant_ue': n_ant_ue
            }
        }
        
        # 각 UE 데이터 로딩 및 채널 모델 분류
        for idx, ue_idx in enumerate(ue_indices):
            combo_key = (area, freq, ue_idx)
            has_p1f = self.config.p1f_available.get(combo_key, False)
            has_p1g = self.config.p1g_available.get(combo_key, False)
            has_p1h = self.config.p1h_available.get(combo_key, False)
            
            # P1F 데이터 (확률적 성분, 선택적)
            p1f_data = None
            chunk_data['P1F']['has_stochastic'][idx] = has_p1f
            if has_p1f:
                p1f_data = self._load_p1f_data(area, freq, ue_idx)
                chunk_data['P1F']['R_BS'][idx] = p1f_data['R_BS']
                chunk_data['P1F']['R_UE'][idx] = p1f_data['R_UE']
                chunk_data['P1F']['metadata'][idx] = p1f_data['metadata']
                chunk_data['P1F']['validation'][idx] = p1f_data['validation']
            else:
                # P1F 없음 → Deterministic
                chunk_data['P1F']['R_BS'][idx] = np.zeros((n_ant_bs, n_ant_bs), dtype=np.complex64)
                chunk_data['P1F']['R_UE'][idx] = np.zeros((n_ant_ue, n_ant_ue), dtype=np.complex64)
                chunk_data['P1F']['metadata'][idx] = {'note': 'Deterministic, no stochastic component'}
                chunk_data['P1F']['validation'][idx] = {}
            
            # P1G 데이터 (Weichselberger 성분, 선택적)
            p1g_data = None
            chunk_data['P1G']['has_coupling'][idx] = has_p1g
            if has_p1g:
                p1g_data = self._load_p1g_data(area, freq, ue_idx)
                chunk_data['P1G']['U_BS'][idx] = p1g_data['U_BS']
                chunk_data['P1G']['U_UE'][idx] = p1g_data['U_UE']
                chunk_data['P1G']['Lambda_BS'][idx] = p1g_data['Lambda_BS']
                chunk_data['P1G']['Lambda_UE'][idx] = p1g_data['Lambda_UE']
                chunk_data['P1G']['Omega'][idx] = p1g_data['Omega']
                chunk_data['P1G']['metadata'][idx] = p1g_data['metadata']
                chunk_data['P1G']['coupling_stats'][idx] = p1g_data['coupling_stats']
                chunk_data['P1G']['evd_validation_bs'][idx] = p1g_data['evd_validation_bs']
                chunk_data['P1G']['evd_validation_ue'][idx] = p1g_data['evd_validation_ue']
            else:
                # P1G 없음 → Kronecker 모델
                chunk_data['P1G']['U_BS'][idx] = np.eye(n_ant_bs, dtype=np.complex64)
                chunk_data['P1G']['U_UE'][idx] = np.eye(n_ant_ue, dtype=np.complex64)
                chunk_data['P1G']['Lambda_BS'][idx] = np.zeros(n_ant_bs, dtype=np.float32)
                chunk_data['P1G']['Lambda_UE'][idx] = np.zeros(n_ant_ue, dtype=np.float32)
                chunk_data['P1G']['Omega'][idx] = np.zeros((n_ant_ue, n_ant_bs), dtype=np.float32)
                chunk_data['P1G']['metadata'][idx] = {'note': 'Kronecker model, no coupling matrix'}
                chunk_data['P1G']['coupling_stats'][idx] = {}
                chunk_data['P1G']['evd_validation_bs'][idx] = {}
                chunk_data['P1G']['evd_validation_ue'][idx] = {}
            
            # P1H 데이터 (결정론적 LoS 성분, 선택적)
            p1h_data = None
            chunk_data['P1H']['has_los'][idx] = has_p1h
            if has_p1h:
                p1h_data = self._load_p1h_data(area, freq, ue_idx)
                chunk_data['P1H']['H_mean'][idx] = p1h_data['H_mean']
                chunk_data['P1H']['metadata'][idx] = p1h_data['metadata']
            else:
                # P1H 없음 → NLoS only
                chunk_data['P1H']['H_mean'][idx] = np.zeros((n_ant_ue, n_ant_bs), dtype=np.complex64)
                chunk_data['P1H']['metadata'][idx] = {'note': 'NLoS only, no LoS component'}
            
            # 보강된 메타데이터 생성 (채널 모델 분류 포함)
            enhanced_meta = self._compute_enhanced_metadata(
                p1f_data, p1g_data, p1h_data,
                has_p1f, has_p1g, has_p1h
            )
            chunk_data['enhanced_metadata'][idx] = enhanced_meta
        
        return chunk_data
    
    def _load_p1f_data(self, area: int, freq: float, ue_idx: int) -> dict:
        """P1F Marginal CCM 데이터 로딩"""
        filename = self.config.P1F_FILE_PATTERN.format(area=area, freq=freq, rx=ue_idx)
        filepath = os.path.join(self.config.P1F_INPUT_DIR, filename)
        
        data = np.load(filepath, allow_pickle=True)
        return {
            'R_BS': data['R_BS'],
            'R_UE': data['R_UE'],
            'metadata': data['metadata'].item() if data['metadata'].ndim == 0 else data['metadata'],
            'validation': data['validation'].item() if data['validation'].ndim == 0 else data['validation']
        }
    
    def _load_p1g_data(self, area: int, freq: float, ue_idx: int) -> dict:
        """P1G Coupling Matrix 데이터 로딩
        
        [중요] tx/rx → BS/UE 키 변환
        입력 파일(P1G)은 다운링크 기준으로 생성되어 U_tx/U_rx 키를 사용하지만,
        이 메서드는 이를 U_BS/U_UE 키로 변환하여 반환합니다.
        
        이는 P1G 생성 시 R_tx=E[H^H H], R_rx=E[H H^H] 정의를 사용했기 때문에,
        U_BS/U_UE로 명명하면 업링크에서도 그대로 사용 가능합니다.
        
        업링크 실험 시 (H_UL = H_DL^H):
        - U_BS, U_UE, Lambda_BS, Lambda_UE: 그대로 사용
        - Omega: Omega.T 필요 (실수 행렬이므로 transpose만)
        """
        filename = self.config.P1G_FILE_PATTERN.format(area=area, freq=freq, rx=ue_idx)
        filepath = os.path.join(self.config.P1G_INPUT_DIR, filename)
        
        data = np.load(filepath, allow_pickle=True)
        
        # [주의] tx/rx → BS/UE 키 변환
        return {
            'U_BS': data['U_tx'],          # tx → BS 변환
            'U_UE': data['U_rx'],          # rx → UE 변환
            'Lambda_BS': data['Lambda_tx'], # tx → BS 변환
            'Lambda_UE': data['Lambda_rx'], # rx → UE 변환
            'Omega': data['Omega'],
            'metadata': data['metadata'].item() if data['metadata'].ndim == 0 else data['metadata'],
            'coupling_stats': data['coupling_stats'].item() if data['coupling_stats'].ndim == 0 else data['coupling_stats'],
            'evd_validation_bs': data['evd_validation_tx'].item() if data['evd_validation_tx'].ndim == 0 else data['evd_validation_tx'],  # tx → bs
            'evd_validation_ue': data['evd_validation_rx'].item() if data['evd_validation_rx'].ndim == 0 else data['evd_validation_rx']   # rx → ue
        }
    
    def _load_p1h_data(self, area: int, freq: float, ue_idx: int) -> dict:
        """P1H Mean Channel 데이터 로딩"""
        filename = self.config.P1H_FILE_PATTERN.format(area=area, freq=freq, rx=ue_idx)
        filepath = os.path.join(self.config.P1H_INPUT_DIR, filename)
        
        data = np.load(filepath, allow_pickle=True)
        
        # H_mean_fro_sq 로딩 (예전 파일 호환성)
        H_mean_fro_sq = data.get('H_mean_fro_sq', None)
        if H_mean_fro_sq is None:
            # 예전 파일: 직접 계산
            H_mean_fro_sq = float(np.sum(np.abs(data['H_mean'])**2))
        else:
            H_mean_fro_sq = float(H_mean_fro_sq)
        
        return {
            'H_mean': data['H_mean'],
            'H_mean_fro_sq': H_mean_fro_sq,
            'metadata': data['metadata'].item() if data['metadata'].ndim == 0 else data['metadata']
        }
    
    def _compute_enhanced_metadata(self, p1f_data: dict, p1g_data: dict, 
                                   p1h_data: dict, 
                                   has_stochastic: bool, has_coupling: bool, has_los: bool) -> dict:
        """채널 모델 분류 및 메타데이터 보강
        
        Args:
            p1f_data: P1F Marginal CCM 데이터 (None if Deterministic)
            p1g_data: P1G Coupling Matrix 데이터 (None if Kronecker)
            p1h_data: P1H Mean Channel 데이터 (None if NLoS only)
            has_stochastic: P1F 존재 여부
            has_coupling: P1G 존재 여부
            has_los: P1H 존재 여부
        
        Returns:
            dict: 채널 모델 분류 + 통계적 메타데이터
        """
        # 1. 채널 모델 분류
        if has_los and not has_stochastic:
            channel_model = 'Deterministic'
        elif has_stochastic and not has_los:
            if has_coupling:
                channel_model = 'Rayleigh-Weichselberger'
            else:
                channel_model = 'Rayleigh-Kronecker'
        elif has_stochastic and has_los:
            if has_coupling:
                channel_model = 'Rician-Weichselberger'
            else:
                channel_model = 'Rician-Kronecker'
        else:
            channel_model = 'Unknown'
        
        # 2. P1F 기반 통계 (확률적 성분)
        if p1f_data is not None:
            R_BS = p1f_data['R_BS']
            R_UE = p1f_data['R_UE']
            n_ant_bs, n_ant_ue = R_BS.shape[0], R_UE.shape[0]
            
            trace_R_BS = float(np.trace(R_BS).real)
            trace_R_UE = float(np.trace(R_UE).real)
            normalized_trace_BS = trace_R_BS / n_ant_bs
            normalized_trace_UE = trace_R_UE / n_ant_ue
            is_deterministic_like_bs = normalized_trace_BS < 0.01
            is_deterministic_like_ue = normalized_trace_UE < 0.01
        else:
            trace_R_BS = None
            trace_R_UE = None
            normalized_trace_BS = None
            normalized_trace_UE = None
            is_deterministic_like_bs = None
            is_deterministic_like_ue = None
        
        # 3. P1G 기반 통계 (Weichselberger 성분)
        if p1g_data is not None:
            Lambda_BS = p1g_data['Lambda_BS']
            Lambda_UE = p1g_data['Lambda_UE']
            Omega = p1g_data['Omega']
            
            mean_coupling = float(np.mean(Omega))
            max_coupling = float(np.max(Omega))
            std_coupling = float(np.std(Omega))
            is_low_coupling = mean_coupling < 0.1
            
            # 유효 랭크 = (sum λ)^2 / sum(λ^2)
            sum_lambda_bs = np.sum(Lambda_BS)
            sum_lambda_ue = np.sum(Lambda_UE)
            rank_effective_bs = float(sum_lambda_bs**2 / np.sum(Lambda_BS**2)) if sum_lambda_bs > 0 else 0
            rank_effective_ue = float(sum_lambda_ue**2 / np.sum(Lambda_UE**2)) if sum_lambda_ue > 0 else 0
            
            # 고유값 집중도 (상위 10%)
            n_ant_bs, n_ant_ue = Lambda_BS.shape[0], Lambda_UE.shape[0]
            n_top_bs = max(1, n_ant_bs // 10)
            n_top_ue = max(1, n_ant_ue // 10)
            power_concentration_bs = float(np.sum(Lambda_BS[:n_top_bs]) / sum_lambda_bs) if sum_lambda_bs > 0 else 0
            power_concentration_ue = float(np.sum(Lambda_UE[:n_top_ue]) / sum_lambda_ue) if sum_lambda_ue > 0 else 0
        else:
            mean_coupling = None
            max_coupling = None
            std_coupling = None
            is_low_coupling = None
            rank_effective_bs = None
            rank_effective_ue = None
            power_concentration_bs = None
            power_concentration_ue = None
        
        # 4. Rician K-factor 추정 (P1H & P1F 둘 다 있으면)
        K_factor_estimate = None
        los_power_ratio = None
        los_power = None
        if p1h_data is not None and p1f_data is not None:
            los_power = p1h_data['H_mean_fro_sq']
            stochastic_power = trace_R_BS * trace_R_UE / (n_ant_bs * n_ant_ue)
            if stochastic_power > 0:
                K_factor_estimate = los_power / stochastic_power
                los_power_ratio = los_power / (los_power + stochastic_power)
        elif p1h_data is not None:
            # P1H만 있는 경우 (Deterministic)
            los_power = p1h_data['H_mean_fro_sq']
        
        # 5. Pathloss 계산
        # Pathloss_dB = -10 * log10(||H_mean||_F^2 + sum(Omega))
        pathloss_dB = None
        H_mean_power = 0.0
        Omega_sum = 0.0
        
        if p1h_data is not None:
            H_mean_power = p1h_data['H_mean_fro_sq']
        
        if p1g_data is not None:
            Omega = p1g_data['Omega']
            Omega_sum = float(np.sum(Omega))
        
        total_power_gain = H_mean_power + Omega_sum
        
        if total_power_gain > 0:
            pathloss_dB = -10.0 * np.log10(total_power_gain)
        # else: pathloss_dB는 None으로 유지 (유효 채널 없음)
        
        enhanced_meta = {
            # 채널 모델 분류
            'channel_model': channel_model,
            'has_stochastic': has_stochastic,
            'has_coupling': has_coupling,
            'has_los': has_los,
            
            # Pathloss (핵심 지표)
            'pathloss_dB': pathloss_dB,
            'H_mean_power': H_mean_power,
            'Omega_sum': Omega_sum,
            'total_power_gain': total_power_gain,
            
            # P1F 통계 (확률적 성분)
            'trace_R_BS': trace_R_BS,
            'trace_R_UE': trace_R_UE,
            'normalized_trace_BS': normalized_trace_BS,
            'normalized_trace_UE': normalized_trace_UE,
            'is_deterministic_like_bs': is_deterministic_like_bs,
            'is_deterministic_like_ue': is_deterministic_like_ue,
            
            # P1G 통계 (Weichselberger 성분)
            'mean_coupling': mean_coupling,
            'max_coupling': max_coupling,
            'std_coupling': std_coupling,
            'is_low_coupling': is_low_coupling,
            'rank_effective_bs': rank_effective_bs,
            'rank_effective_ue': rank_effective_ue,
            'power_concentration_bs': power_concentration_bs,
            'power_concentration_ue': power_concentration_ue,
            
            # Rician K-factor (P1H & P1F 있으면)
            'K_factor_estimate': K_factor_estimate,
            'los_power_ratio': los_power_ratio,
        }
        
        return enhanced_meta

# ===== SECTION 4: ChunkDataManager =====
class ChunkDataManager:
    """청크 파일 저장 및 로딩 관리"""
    
    def __init__(self, config: P1I_Config, total_chunks: int = None):
        self.config = config
        os.makedirs(self.config.P1I_OUTPUT_DIR, exist_ok=True)
        
        # 청크 순번 자릿수 결정 (총 청크 개수 기반)
        if total_chunks is not None:
            self.idx_width = len(str(total_chunks))
        else:
            self.idx_width = 2  # 기본값
    
    def chunk_file_exists(self, area: int, freq: float, chunk_idx: int, ue_start: int, ue_end: int) -> bool:
        """청크 파일 존재 여부 확인
        
        Args:
            area: Area 인덱스
            freq: 주파수 (GHz)
            chunk_idx: 청크 순번 (1-based)
            ue_start: 청크 시작 UE
            ue_end: 청크 종료 UE
        
        Returns:
            bool: 파일 존재 여부
        """
        idx_str = str(chunk_idx).zfill(self.idx_width)
        filename = self.config.P1I_CHUNK_PATTERN.format(
            area=area, freq=freq, idx=idx_str, start=ue_start, end=ue_end
        )
        filepath = os.path.join(self.config.P1I_OUTPUT_DIR, filename)
        return os.path.exists(filepath)
    
    def save_chunk(self, chunk_data: dict, chunk_idx: int) -> str:
        """청크 데이터 저장
        
        [중요] BS/UE 명명 규칙
        저장되는 모든 키는 BS/UE 표기를 사용합니다 (tx/rx 금지).
        이는 다운링크/업링크 방향과 무관하게 일관된 데이터 구조를 보장합니다.
        
        저장 키 규칙:
          - P1F_R_BS, P1F_R_UE: CCM (정의: R_tx=E[H^H H], R_rx=E[H H^H])
          - P1G_U_BS, P1G_U_UE: BS/UE 고유벡터 (tx/rx 아님!)
          - P1G_Lambda_BS, P1G_Lambda_UE: BS/UE 고유값
          - P1G_Omega: 커플링 행렬 (업링크 시 transpose 필요)
          - P1G_evd_validation_bs/ue: BS/UE EVD 검증
          - P1H_H_mean: 평균 채널 (업링크 시 hermitian conjugate 필요)
          - chunk_metadata['n_ant_bs'], ['n_ant_ue']: 안테나 수
        
        업링크 실험 시 데이터 로딩 후 (H_UL = H_DL^H):
          - R_BS, R_UE: 그대로 사용 (위의 CCM 정의 덕분)
          - H_mean_uplink = H_mean.conj().T (Hermitian conjugate)
          - Omega_uplink = Omega.T (실수 행렬이므로 transpose만)
        
        Args:
            chunk_data: aggregate_chunk_data() 반환값
            chunk_idx: 청크 순번 (1-based)
        
        Returns:
            str: 저장된 파일 경로
        """
        meta = chunk_data['chunk_metadata']
        idx_str = str(chunk_idx).zfill(self.idx_width)
        
        filename = self.config.P1I_CHUNK_PATTERN.format(
            area=meta['area_idx'],
            freq=meta['freq_ghz'],
            idx=idx_str,
            start=meta['ue_start'],
            end=meta['ue_end']
        )
        filepath = os.path.join(self.config.P1I_OUTPUT_DIR, filename)
        
        # 청크 데이터를 평평한 딕셔너리로 변환 (npz 저장용)
        # [주의] 모든 키는 BS/UE 표기 사용 (다운링크/업링크 혼동 방지)
        save_dict = {
            'ue_indices': chunk_data['ue_indices'],
            # P1F 데이터 (확률적 성분)
            'P1F_R_BS': chunk_data['P1F']['R_BS'],
            'P1F_R_UE': chunk_data['P1F']['R_UE'],
            'P1F_has_stochastic': chunk_data['P1F']['has_stochastic'],
            'P1F_metadata': chunk_data['P1F']['metadata'],
            'P1F_validation': chunk_data['P1F']['validation'],
            # P1G 데이터 (Weichselberger 성분) - BS/UE 표기!
            'P1G_U_BS': chunk_data['P1G']['U_BS'],
            'P1G_U_UE': chunk_data['P1G']['U_UE'],
            'P1G_Lambda_BS': chunk_data['P1G']['Lambda_BS'],
            'P1G_Lambda_UE': chunk_data['P1G']['Lambda_UE'],
            'P1G_Omega': chunk_data['P1G']['Omega'],
            'P1G_has_coupling': chunk_data['P1G']['has_coupling'],
            'P1G_metadata': chunk_data['P1G']['metadata'],
            'P1G_coupling_stats': chunk_data['P1G']['coupling_stats'],
            'P1G_evd_validation_bs': chunk_data['P1G']['evd_validation_bs'],
            'P1G_evd_validation_ue': chunk_data['P1G']['evd_validation_ue'],
            # P1H 데이터 (결정론적 LoS 성분)
            'P1H_H_mean': chunk_data['P1H']['H_mean'],
            'P1H_has_los': chunk_data['P1H']['has_los'],
            'P1H_metadata': chunk_data['P1H']['metadata'],
            # 보강된 메타데이터 (채널 모델 분류)
            'enhanced_metadata': chunk_data['enhanced_metadata'],
            # 청크 메타데이터 (n_ant_bs, n_ant_ue 포함)
            'chunk_metadata': chunk_data['chunk_metadata']
        }
        
        np.savez_compressed(filepath, **save_dict)
        
        return filepath
    
    def load_chunk(self, area: int, freq: float, chunk_idx: int, ue_start: int, ue_end: int) -> dict:
        """저장된 청크 데이터 로딩
        
        [중요] BS/UE 명명 규칙
        로딩되는 모든 키는 BS/UE 표기를 사용합니다.
        후속 코드에서 이 데이터를 사용할 때 반드시 BS/UE 표기를 유지하세요.
        
        로딩 키 규칙:
          - P1F_R_BS, P1F_R_UE: CCM (정의: R_tx=E[H^H H], R_rx=E[H H^H])
          - P1G_U_BS, P1G_U_UE: BS/UE 고유벡터
          - P1G_Lambda_BS, P1G_Lambda_UE: BS/UE 고유값
          - 다운링크든 업링크든 BS는 BS, UE는 UE로 일관되게 사용
        
        업링크 실험 시 (H_UL = H_DL^H):
          - R_BS, R_UE: 그대로 사용 (위의 CCM 정의 덕분)
          - H_mean: H_mean.conj().T 필요 (Hermitian conjugate)
          - Omega: Omega.T 필요 (실수 행렬이므로 transpose만)
        
        Args:
            area: Area 인덱스
            freq: 주파수 (GHz)
            chunk_idx: 청크 순번 (1-based)
            ue_start: 청크 시작 UE
            ue_end: 청크 종료 UE
        
        Returns:
            dict: 청크 데이터 (aggregate_chunk_data()와 동일한 구조)
        """
        idx_str = str(chunk_idx).zfill(self.idx_width)
        filename = self.config.P1I_CHUNK_PATTERN.format(
            area=area, freq=freq, idx=idx_str, start=ue_start, end=ue_end
        )
        filepath = os.path.join(self.config.P1I_OUTPUT_DIR, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Chunk file not found: {filepath}")
        
        data = np.load(filepath, allow_pickle=True)
        
        # 계층적 딕셔너리로 재구성
        # [주의] 모든 키는 BS/UE 표기 사용
        chunk_data = {
            'ue_indices': data['ue_indices'],
            'P1F': {
                'R_BS': data['P1F_R_BS'],
                'R_UE': data['P1F_R_UE'],
                'has_stochastic': data['P1F_has_stochastic'],
                'metadata': data['P1F_metadata'],
                'validation': data['P1F_validation']
            },
            'P1G': {  # BS/UE 표기!
                'U_BS': data['P1G_U_BS'],
                'U_UE': data['P1G_U_UE'],
                'Lambda_BS': data['P1G_Lambda_BS'],
                'Lambda_UE': data['P1G_Lambda_UE'],
                'Omega': data['P1G_Omega'],
                'has_coupling': data['P1G_has_coupling'],
                'metadata': data['P1G_metadata'],
                'coupling_stats': data['P1G_coupling_stats'],
                'evd_validation_bs': data['P1G_evd_validation_bs'],
                'evd_validation_ue': data['P1G_evd_validation_ue']
            },
            'P1H': {
                'H_mean': data['P1H_H_mean'],
                'has_los': data['P1H_has_los'],
                'metadata': data['P1H_metadata']
            },
            'enhanced_metadata': data['enhanced_metadata'],
            'chunk_metadata': data['chunk_metadata'].item()
        }
        
        return chunk_data
    
    def verify_chunk(self, area: int, freq: float, chunk_idx: int, ue_start: int, ue_end: int, 
                     expected_ue_indices: list) -> tuple:
        """청크 파일 검증 (삭제 전 안전성 확인)
        
        Args:
            area: Area 인덱스
            freq: 주파수 (GHz)
            chunk_idx: 청크 순번 (1-based)
            ue_start: 청크 시작 UE
            ue_end: 청크 종료 UE
            expected_ue_indices: 예상 UE 인덱스 리스트
        
        Returns:
            tuple: (is_valid, error_message)
        """
        try:
            # 1. 파일 존재 확인
            if not self.chunk_file_exists(area, freq, chunk_idx, ue_start, ue_end):
                return False, "청크 파일이 존재하지 않음"
            
            # 2. 청크 로딩 시도
            chunk_data = self.load_chunk(area, freq, chunk_idx, ue_start, ue_end)
            
            # 3. UE 인덱스 검증
            loaded_ue_indices = chunk_data['ue_indices'].tolist()
            if loaded_ue_indices != expected_ue_indices:
                return False, f"UE 인덱스 불일치: 예상 {expected_ue_indices}, 실제 {loaded_ue_indices}"
            
            # 4. 데이터 shape 검증
            n_UEs = len(expected_ue_indices)
            meta = chunk_data['chunk_metadata']
            n_ant_bs = meta['n_ant_bs']
            n_ant_ue = meta['n_ant_ue']
            
            # P1F shape 검증
            if chunk_data['P1F']['R_BS'].shape != (n_UEs, n_ant_bs, n_ant_bs):
                return False, f"P1F R_BS shape 불일치"
            if chunk_data['P1F']['R_UE'].shape != (n_UEs, n_ant_ue, n_ant_ue):
                return False, f"P1F R_UE shape 불일치"
            if chunk_data['P1F']['has_stochastic'].shape != (n_UEs,):
                return False, f"P1F has_stochastic shape 불일치"
            
            # P1G shape 검증
            if chunk_data['P1G']['U_BS'].shape != (n_UEs, n_ant_bs, n_ant_bs):
                return False, f"P1G U_BS shape 불일치"
            if chunk_data['P1G']['Omega'].shape != (n_UEs, n_ant_ue, n_ant_bs):
                return False, f"P1G Omega shape 불일치"
            if chunk_data['P1G']['has_coupling'].shape != (n_UEs,):
                return False, f"P1G has_coupling shape 불일치"
            
            # P1H shape 검증
            if chunk_data['P1H']['H_mean'].shape != (n_UEs, n_ant_ue, n_ant_bs):
                return False, f"P1H H_mean shape 불일치"
            if chunk_data['P1H']['has_los'].shape != (n_UEs,):
                return False, f"P1H has_los shape 불일치"
            
            # Enhanced metadata 검증
            if chunk_data['enhanced_metadata'].shape != (n_UEs,):
                return False, f"enhanced_metadata shape 불일치"
            
            # 5. 메타데이터 검증
            if meta['area_idx'] != area or meta['freq_ghz'] != freq:
                return False, f"메타데이터 불일치"
            if meta['ue_start'] != ue_start or meta['ue_end'] != ue_end:
                return False, f"UE 범위 불일치"
            
            return True, "검증 성공"
            
        except Exception as e:
            return False, f"검증 중 오류: {str(e)}"
    
    def cleanup_individual_files(self, area: int, freq: float, ue_indices: list) -> tuple:
        """청크에 포함된 개별 파일 삭제
        
        Args:
            area: Area 인덱스
            freq: 주파수 (GHz)
            ue_indices: 삭제할 UE 인덱스 리스트
        
        Returns:
            tuple: (n_deleted, n_failed, failed_files)
        """
        n_deleted = 0
        n_failed = 0
        failed_files = []
        
        for ue_idx in ue_indices:
            # P1F 파일 삭제
            p1f_file = self.config.P1F_FILE_PATTERN.format(area=area, freq=freq, rx=ue_idx)
            p1f_path = os.path.join(self.config.P1F_INPUT_DIR, p1f_file)
            if os.path.exists(p1f_path):
                try:
                    os.remove(p1f_path)
                    n_deleted += 1
                except Exception as e:
                    n_failed += 1
                    failed_files.append((p1f_path, str(e)))
            
            # P1G 파일 삭제
            p1g_file = self.config.P1G_FILE_PATTERN.format(area=area, freq=freq, rx=ue_idx)
            p1g_path = os.path.join(self.config.P1G_INPUT_DIR, p1g_file)
            if os.path.exists(p1g_path):
                try:
                    os.remove(p1g_path)
                    n_deleted += 1
                except Exception as e:
                    n_failed += 1
                    failed_files.append((p1g_path, str(e)))
            
            # P1H 파일 삭제
            p1h_file = self.config.P1H_FILE_PATTERN.format(area=area, freq=freq, rx=ue_idx)
            p1h_path = os.path.join(self.config.P1H_INPUT_DIR, p1h_file)
            if os.path.exists(p1h_path):
                try:
                    os.remove(p1h_path)
                    n_deleted += 1
                except Exception as e:
                    n_failed += 1
                    failed_files.append((p1h_path, str(e)))
        
        return n_deleted, n_failed, failed_files

# ===== SECTION 5: main =====
def main():
    """P1I Weichselberger 청크 저장 메인 실행"""
    
    print("=" * 80)
    print("P1I: Weichselberger 모델 파라미터 청크 저장")
    print("=" * 80)
    print()
    
    # 설정 초기화
    config = P1I_Config()
    
    # (Area, Freq)별로 RX 그룹핑
    area_freq_groups = config.group_by_area_freq()
    
    if not area_freq_groups:
        print("경고: 처리할 데이터가 없습니다.")
        return
    
    # 전체 통계
    total_chunks = sum(
        (len(ue_list) + config.CHUNK_SIZE - 1) // config.CHUNK_SIZE
        for ue_list in area_freq_groups.values()
    )
    print(f"\n청크 생성 계획:")
    print(f"  - 청크 크기: {config.CHUNK_SIZE}개 UEs/청크")
    print(f"  - 예상 청크 수: {total_chunks}개")
    print(f"  - 개별 파일 정리: {'예 (청크 검증 후 삭제)' if config.CLEANUP_INDIVIDUAL_FILES else '아니오 (보존)'}")
    print()
    
    # Aggregator 초기화
    aggregator = ChunkDataAggregator(config)
    
    # Manager 초기화 (총 청크 개수 전달)
    manager = ChunkDataManager(config, total_chunks=total_chunks)
    
    # 청크 처리 시작
    overall_start = time.time()
    chunk_count = 0
    
    for (area, freq), ue_list in sorted(area_freq_groups.items()):
        print(f"\n{'='*80}")
        print(f"Area{area}_{freq}GHz 처리 중 ({len(ue_list)}개 UEs)")
        print(f"{'='*80}")
        
        # 청크 분할
        chunks = aggregator.create_chunks(area, freq, ue_list)
        
        for local_chunk_idx, (ue_start, ue_end, ue_indices) in enumerate(chunks, 1):
            chunk_count += 1
            
            # 청크 파일 존재 여부 확인
            if manager.chunk_file_exists(area, freq, chunk_count, ue_start, ue_end):
                print(f"[{chunk_count:>{len(str(total_chunks))}}/{total_chunks}] "
                      f"Chunk_{str(chunk_count).zfill(manager.idx_width)} UE{ue_start}-{ue_end} (건너뜀: 이미 존재)")
                continue
            
            # 청크 데이터 집계
            start_time = time.time()
            chunk_data = aggregator.aggregate_chunk_data(area, freq, ue_indices)
            aggregation_time = time.time() - start_time
            
            # 청크 파일 저장
            start_save = time.time()
            saved_path = manager.save_chunk(chunk_data, chunk_count)
            save_time = time.time() - start_save
            
            # 파일 크기 계산
            file_size_mb = os.path.getsize(saved_path) / (1024 * 1024)
            
            # 청크 검증
            is_valid, validation_msg = manager.verify_chunk(area, freq, chunk_count, ue_start, ue_end, ue_indices)
            
            chunk_idx_str = str(chunk_count).zfill(manager.idx_width)
            
            if not is_valid:
                print(f"[{chunk_count:>{len(str(total_chunks))}}/{total_chunks}] "
                      f"Chunk_{chunk_idx_str} UE{ue_start}-{ue_end} ({len(ue_indices)}개 UEs) → "
                      f"청크 검증 실패: {validation_msg}")
                continue
            
            # 개별 파일 정리 (옵션 활성화 시)
            cleanup_msg = ""
            if config.CLEANUP_INDIVIDUAL_FILES:
                n_deleted, n_failed, failed_files = manager.cleanup_individual_files(area, freq, ue_indices)
                if n_failed > 0:
                    cleanup_msg = f", 정리: {n_deleted}개 성공 {n_failed}개 실패"
                else:
                    cleanup_msg = f", 정리: {n_deleted}개 파일 삭제"
            
            print(f"[{chunk_count:>{len(str(total_chunks))}}/{total_chunks}] "
                  f"Chunk_{chunk_idx_str} UE{ue_start}-{ue_end} ({len(ue_indices)}개 UEs) → "
                  f"{file_size_mb:.1f}MB ({aggregation_time:.1f}s + {save_time:.1f}s){cleanup_msg}")
            
            # 진행률 출력 및 화면 클리어
            if config.ENABLE_SCREEN_CLEAR and chunk_count % config.PROGRESS_CLEAR_INTERVAL == 0:
                current_time = time.time()
                elapsed_time = current_time - overall_start
                avg_time_per_chunk = elapsed_time / chunk_count
                remaining_chunks = total_chunks - chunk_count
                estimated_remaining = avg_time_per_chunk * remaining_chunks
                
                # 화면 클리어
                if JUPYTER_AVAILABLE:
                    clear_output(wait=True)
                else:
                    import platform
                    os.system('cls' if platform.system() == 'Windows' else 'clear')
                
                # 시간 포맷팅 함수
                def format_time(seconds):
                    if seconds < 60:
                        return f"{seconds:.1f}초"
                    elif seconds < 3600:
                        return f"{int(seconds//60)}분 {int(seconds%60)}초"
                    else:
                        return f"{int(seconds//3600)}시간 {int((seconds%3600)//60)}분"
                
                # 진행 상황 요약
                print("P1I Weichselberger 청크 생성 진행률:")
                print(f"현재 완료율: {chunk_count/total_chunks*100:.1f}% ({chunk_count}/{total_chunks} 청크)")
                print(f"전체 경과 시간: {format_time(elapsed_time)}")
                print(f"평균 처리 속도: {avg_time_per_chunk:.1f}초/청크")
                print(f"예상 남은 시간: {format_time(estimated_remaining)}")
                print("-" * 80)
    
    # 완료 메시지
    total_time = time.time() - overall_start
    print(f"\n{'='*80}")
    print(f"=== P1I Weichselberger 청크 저장 완료 ===")
    print(f"{'='*80}")
    print(f"총 {chunk_count}개 청크 생성 완료")
    print(f"총 소요 시간: {total_time/60:.1f}분")
    print(f"저장 위치: {config.P1I_OUTPUT_DIR}")
    print()

if __name__ == "__main__":
    main()

