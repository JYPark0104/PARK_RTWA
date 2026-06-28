# ======================================================================
# P1J_Weichsel_SU_MIMO_Capacity_2510v1.py
# P1J: Weichselberger SU-MIMO 업링크 채널 용량 계산 (Wen2011 알고리듬)
# 
# === 최상위 목적 ===
# P1I 청크 데이터(다운링크 기준)를 업링크로 전환하여 UE 위치별 최적 입력 공분산 및 채널 용량 계산
# - Wen2011 점근적 합-전송률 최대화 알고리듬 구현 (SU-MIMO 업링크, K=1)
# - Weichselberger Rician 채널 모델 기반 최적화
# 
# === P1I 데이터 구조 (다운링크 기준) ===
# P1I 청크는 다운링크(DL: BS→UE)를 기준으로 저장됨:
#   - P1G_U_BS: [100, n_bs, n_bs] BS 고유벡터
#   - P1G_U_UE: [100, n_ue, n_ue] UE 고유벡터
#   - P1G_Omega: [100, n_ue, n_bs] 커플링 행렬 (DL)
#   - P1H_H_mean: [100, n_ue, n_bs] 평균 채널 (DL)
# 
# === 다운링크 → 업링크 전환 (H_UL = H_DL^H) ===
# Wen2011은 업링크(UL: UE→BS) 기준이므로 전환 필요:
#   - U_bs (UL BS 수신 = DL BS 송신): P1G_U_BS 그대로 사용
#   - U_ue (UL UE 송신 = DL UE 수신): P1G_U_UE 그대로 사용
#   - Omega_ul: [n_bs, n_ue] = P1G_Omega.T (전치 필요!)
#   - H_mean_ul: [n_bs, n_ue] = P1H_H_mean.conj().T (Hermitian 전치 필요!)
# 
# === Wen2011 알고리듬 구조 (업링크 UE→BS, K=1) ===
# Input: U_bs, U_ue, Ω, H̄ (업링크 Weichselberger 파라미터)
#   - U_bs: [n_bs, n_bs] BS 수신측 고유벡터 (unitary)
#   - U_ue: [n_ue, n_ue] UE 송신측 고유벡터 (unitary)
#   - Ω: [n_bs, n_ue] 커플링 행렬 (업링크)
#   - H̄: [n_bs, n_ue] LOS 성분 (업링크)
# Output: P* [n_ue, n_ue] 최적 UE 입력 공분산, I* 최대 합-전송률
#
# 이중 루프:
#   외부: P 최적화 (water-filling)
#     내부: 고정점 (γ, ψ) 계산
#       (10) T = U_ue diag(Ω^T γ) U_ue^H, R = U_bs diag(Ω ψ) U_bs^H
#       (8) Ξ = T + H̄^H (I+R)^{-1} H̄
#       (11a) γ_n = u_bs,n^H (I+R)^{-1} u_bs,n
#       (11b) ψ_m = u_ue,m^H P (I+Ξ P)^{-1} u_ue,m
#     (7) I(P) = log|I+Ξ P| + log|I+R| - γ^T Ω ψ
#     (26,29,30) Water-filling: P = U_Ξ (1/μ - 1/λ_Ξ)^+ U_Ξ^H
#
# === 주요 구성 요소 ===
# - P1J_Config: P1I 청크 로딩 설정 + 용량 결과 저장 경로
# - WeichselbergerCapacityOptimizer: Wen2011 알고리듬 구현 (업링크)
# - P1J_DataProcessor: 청크 로딩 + DL→UL 전환 + UE 위치별 처리
# - CapacityResults_Manager: 용량 결과 파일 저장/로딩 관리
# 
# === 입력/출력 ===
# 입력: P1I 청크 (다운링크, Area{area}_{freq}GHz_Weichsel_Chunk_{idx}_UE{start}-{end}.npz)
# 출력: Area{area}_{freq}GHz_UE{ue}_Capacity.npz (업링크 UE 위치별)
#   - P_opt: [n_ue, n_ue] 최적 UE 입력 공분산
#   - sum_rate_opt: 최대 점근적 합-전송률
#   - kappa: Rician K-팩터
#   - power_total: tr(P) = n_ue × SNR_linear
#
# === 주요 수정 이력 ===
# [251015] P1J 신규 작성: Weichselberger 업링크 채널 용량 계산
# 1. P1I BS/UE 명명 규칙 준수
# 2. DL→UL 전환 명확히 구현 (Omega.T, H_mean.conj().T)
# 3. Wen2011 알고리듬 업링크 기준 구현 (K=1)
# 4. Wen2011 식 (4) 채널 정규화: K-팩터 보존하며 목표 ρ 달성
# 5. UE 위치별 독립 파일 저장
#
# ======================================================================

# ===== SECTION 1: 환경 설정 =====
import os
import time
from datetime import datetime
import numpy as np
import glob
import re
import csv
from pathlib import Path

# TensorFlow 환경 설정
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc'
gpu_num = 0
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

# TensorFlow 코어 임포트
import tensorflow as tf

# GPU 메모리 설정
gpus = tf.config.list_physical_devices('GPU')
for g in gpus:
    try:
        tf.config.experimental.set_memory_growth(g, True)
        print(f"GPU 메모리 설정: 동적 증가 모드")
    except RuntimeError as e:
        print(f"GPU 설정 경고: {e}")

# TensorFlow 최적화 설정
tf.get_logger().setLevel("ERROR")
tf.config.optimizer.set_jit(True)
tf.random.set_seed(1)
np.random.seed(1)

# 주피터/IPython 화면 클리어 지원
try:
    from IPython.display import clear_output
    JUPYTER_AVAILABLE = True
except ImportError:
    JUPYTER_AVAILABLE = False

# ===== SECTION 2: P1J_Config =====
class P1J_Config:
    """P1I 청크 데이터 로딩 및 용량 계산 설정"""
    
    def __init__(self):
        # 스크립트의 디렉토리를 기준으로 절대 경로 설정
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # P1I 청크 입력 설정
        self.P1I_INPUT_DIR = os.path.join(script_dir, "P1I_Weichsel_Chunk_Results")
        self.P1I_CHUNK_PATTERN = "Area{area}_{freq}GHz_Weichsel_Chunk_{idx}_UE{start}-{end}.npz"
        
        # P1J 용량 결과 저장 설정
        self.P1J_OUTPUT_DIR = os.path.join(script_dir, "P1J_Capacity_Results")
        self.P1J_FILE_PATTERN = "Area{area}_{freq}GHz_UE{ue}_Capacity.npz"
        
        # 콘솔 출력 제어 설정
        self.ENABLE_SCREEN_CLEAR = True
        self.PROGRESS_CLEAR_INTERVAL = 10
        
        # 필터링 설정
        self.target_areas = [1]  # 처리할 area 목록 (None이면 전체)
        self.target_freqs = None  # 처리할 주파수 목록 (None이면 전체)
        
        # 최적화 알고리듬 설정 (Wen2011) - 빠른 실행 설정
        self.eps_inner = 1e-4      # 내부 루프 (고정점) 수렴 tolerance (TIGHT 유지)
        self.eps_outer = 0.05      # 외부 루프 (water-filling) 수렴 tolerance
        self.max_iter_inner = 50   # 내부 루프 최대 반복 횟수
        self.max_iter_outer = 50   # 외부 루프 최대 반복 횟수 (논문: 1-2회로 충분)
        self.early_stop_rate_thresh = 0.5  # 조기 종료: rate 증가 < 0.5 bits/Hz/sec
        self.regularization = 1e-10 # 수치 안정성 regularization
        self.debug_mode = False    # 디버그 출력 (False: 비활성화)
        
        # === SNR 및 전력 제약 설정 ===
        # Wen2011 식 (4): K-팩터 보존하며 목표 ρ 달성, tr(P) = N_ue * SNR_linear
        self.SNR_dB = 10.0
        self.SNR_linear = 10.0**(self.SNR_dB / 10.0)
        
        # P1I 청크 데이터 스캔
        self.detect_chunk_data()
    
    def detect_chunk_data(self):
        """P1I 청크 파일 스캔 및 실제 UE 목록 구성"""
        
        # 청크 파일 스캔
        scan_pattern = f"{self.P1I_INPUT_DIR}/*.npz"
        chunk_files = glob.glob(scan_pattern)
        
        # 파일명 파싱용 정규식
        pattern = r'Area(\d+)_(.+)GHz_Weichsel_Chunk_(\d+)_UE(\d+)-(\d+)\.npz'
        
        chunk_candidates = []
        
        for filepath in chunk_files:
            filename = os.path.basename(filepath)
            match = re.match(pattern, filename)
            if match:
                area = int(match.group(1))
                freq = float(match.group(2))
                chunk_idx = int(match.group(3))
                ue_start = int(match.group(4))
                ue_end = int(match.group(5))
                
                # 필터링 적용
                if self.target_areas is not None and area not in self.target_areas:
                    continue
                if self.target_freqs is not None and freq not in self.target_freqs:
                    continue
                
                chunk_candidates.append((area, freq, chunk_idx, ue_start, ue_end, filepath))
        
        # 정렬 (area, freq, chunk_idx 순)
        chunk_candidates.sort(key=lambda x: (x[0], x[1], x[2]))
        
        if not chunk_candidates:
            print("경고: P1I 청크 파일이 없습니다.")
            self.chunk_info = []
            return
        
        # 각 청크 파일을 로딩하여 실제 UE 개수 확인
        print(f"청크 파일 스캔 중...")
        self.chunk_info = []
        area_groups = {}
        total_ues = 0
        
        for area, freq, chunk_idx, ue_start, ue_end, filepath in chunk_candidates:
            # 청크 파일 로딩하여 실제 UE 개수 확인
            try:
                data = np.load(filepath, allow_pickle=True)
                ue_indices = data['ue_indices'].tolist()
                actual_ue_count = len(ue_indices)
                
                self.chunk_info.append((area, freq, chunk_idx, ue_start, ue_end, filepath, ue_indices))
                
                key = f"Area{area}_{freq}GHz"
                if key not in area_groups:
                    area_groups[key] = {'chunks': [], 'ue_count': 0}
                area_groups[key]['chunks'].append((chunk_idx, ue_start, ue_end))
                area_groups[key]['ue_count'] += actual_ue_count
                total_ues += actual_ue_count
            except Exception as e:
                print(f"경고: {filepath} 로딩 실패: {e}")
                continue
        
        print(f"P1I 청크 데이터 스캔 결과:")
        print(f"  - 총 청크 파일: {len(self.chunk_info)}개")
        print(f"  - 총 UE 위치: {total_ues}개")
        print()
        print(f"Area별 상세:")
        for area_freq, info in sorted(area_groups.items()):
            print(f"  - {area_freq}: {len(info['chunks'])}개 청크, {info['ue_count']}개 UE")

# ===== SECTION 3: WeichselbergerCapacityOptimizer =====
class WeichselbergerCapacityOptimizer:
    """Wen2011 점근적 합-전송률 최대화 알고리듬 (SU-MIMO 업링크 K=1)
    
    Wen2011 Algorithm 1 (업링크 UE→BS):
      Input: U_bs, U_ue, Ω, H̄
      Output: P* (최적 UE 입력 공분산), I*
      
      외부 루프:
        내부 루프: 고정점 (γ, ψ)
          (10) T = U_ue diag(Ω^T γ) U_ue^H, R = U_bs diag(Ω ψ) U_bs^H
          (8) Ξ = T + H̄^H (I+R)^{-1} H̄
          (11a) γ_n = u_bs,n^H (I+R)^{-1} u_bs,n
          (11b) ψ_m = u_ue,m^H P (I+Ξ P)^{-1} u_ue,m
        (7) I(P) = log|I+Ξ P| + log|I+R| - γ^T Ω ψ
        (26,29,30) Water-filling: P = U_Ξ (1/μ - 1/λ_Ξ)^+ U_Ξ^H
    """
    
    def __init__(self, 
                 U_bs: tf.Tensor,
                 U_ue: tf.Tensor,
                 Omega_ul: tf.Tensor,
                 H_mean_ul: tf.Tensor,
                 config: P1J_Config):
        """업링크 UE→BS 채널 용량 최적화 초기화
        
        Args:
            U_bs: [n_bs, n_bs] BS 수신측 고유벡터 (unitary)
            U_ue: [n_ue, n_ue] UE 송신측 고유벡터 (unitary)
            Omega_ul: [n_bs, n_ue] 커플링 행렬 (업링크, DL transpose 후)
            H_mean_ul: [n_bs, n_ue] LoS 성분 (업링크, DL hermitian transpose 후)
            config: P1J_Config 객체
        """
        self.U_bs = tf.cast(U_bs, tf.complex64)
        self.U_ue = tf.cast(U_ue, tf.complex64)
        self.config = config
        
        self.n_bs = U_bs.shape[0]
        self.n_ue = U_ue.shape[0]
        
        # === Wen2011 채널 정규화 (식 4) ===
        # ||Ω||_1/(N·M) = ρ/(κ+1), tr(H̄H̄^H)/(N·M) = κρ/(κ+1)
        
        Omega_tf = tf.cast(Omega_ul, tf.float32)
        H_mean_tf = tf.cast(H_mean_ul, tf.complex64)
        epsilon = 1e-30
        
        # 1. 현재 채널 전력 계산
        P_nlos_current = tf.reduce_sum(Omega_tf)  # ||Ω||_1
        P_los_current = tf.reduce_sum(tf.square(tf.abs(H_mean_tf)))  # tr(H̄H̄^H)
        P_total_current = P_los_current + P_nlos_current
        
        rho_target = config.SNR_linear
        N_M = float(self.n_bs * self.n_ue)
        
        # 2. 케이스 구분: NLoS only / LoS only / Mixed
        threshold = 1e-10  # 상대 임계값: 한쪽이 다른쪽의 1e-10 미만이면 무시
        
        if P_los_current < threshold * P_nlos_current:
            # === NLoS only (Rayleigh) ===
            self.kappa = 0.0
            self.Omega = (N_M * rho_target / (P_nlos_current + epsilon)) * Omega_tf
            self.H_mean_ul = tf.zeros_like(H_mean_tf)
            
        elif P_nlos_current < threshold * P_los_current:
            # === LoS only (Deterministic) ===
            self.kappa = float('inf')
            self.Omega = tf.zeros_like(Omega_tf)
            self.H_mean_ul = tf.cast(
                tf.sqrt(N_M * rho_target / (P_los_current + epsilon)), 
                dtype=tf.complex64
            ) * H_mean_tf
            
        else:
            # === Mixed (Rician): K-팩터 보존하며 재스케일링 ===
            self.kappa = float((P_los_current / P_nlos_current).numpy())
            P_nlos_target = N_M * (1.0 / (self.kappa + 1.0)) * rho_target
            P_los_target = N_M * (self.kappa / (self.kappa + 1.0)) * rho_target
            
            self.Omega = (P_nlos_target / P_nlos_current) * Omega_tf
            self.H_mean_ul = tf.cast(
                tf.sqrt(P_los_target / P_los_current), 
                dtype=tf.complex64
            ) * H_mean_tf
        
        # 3. 전력 제약: tr(P) = N_ue * SNR_linear
        self.power_total = float(self.n_ue) * config.SNR_linear
        
        self.reg = config.regularization
        self.eps_inner = config.eps_inner
        self.eps_outer = config.eps_outer
        self.max_iter_inner = config.max_iter_inner
        self.max_iter_outer = config.max_iter_outer
    
    def optimize(self) -> dict:
        """전체 최적화 실행
        
        Returns:
            dict: {
                'P_opt': [n_ue, n_ue] 최적 UE 입력 공분산,
                'sum_rate_opt': 최대 점근적 합-전송률,
                'sum_rate_history': [t] 최적화 이력,
                'converged': bool,
                'n_iter_outer': 외부 루프 반복,
                'n_iter_inner_avg': 평균 내부 루프 반복,
                'kappa': Rician K-팩터,
                'power_total': tr(P)
            }
        """
        # 초기화: P = I (균등 전력 할당)
        P = tf.eye(self.n_ue, dtype=tf.complex64) * (self.power_total / self.n_ue)
        
        sum_rate_history = []
        sum_rate_prev = -np.inf
        inner_iter_counts = []
        act_rank = None
        act_pow_ratio = None
        water_level = None
        
        for t in range(self.max_iter_outer):
            # Step A: 고정점 계산
            gamma, psi, sum_rate, n_iter_inner = self._compute_fixed_point(P, t)
            sum_rate_history.append(float(sum_rate.numpy()))
            inner_iter_counts.append(n_iter_inner)
            
            # Rate 증가량 계산
            rate_diff = abs(sum_rate.numpy() - sum_rate_prev) if t > 0 else float('inf')
            
            # 수렴 상태 출력 (10 iteration마다)
            if (t + 1) % 10 == 0:
                print(f"  [외부 t={t+1}] rate={sum_rate.numpy():.4f} bits/Hz/s, Δrate={rate_diff:.2e}, 내부 iter={n_iter_inner}")
            
            # Step B: Water-filling (조기 종료/수렴 전에 실행하여 지표 저장)
            Xi = self._compute_Xi(gamma, psi)
            P, act_rank, act_pow_ratio, water_level = self._water_filling(Xi, debug_outer_iter=t)
            
            # 조기 종료: rate 증가 미미 (5회 이후 체크)
            if t >= 5 and rate_diff < self.config.early_stop_rate_thresh:
                print(f"  [조기 종료] t={t+1}, Δrate={rate_diff:.2e} < {self.config.early_stop_rate_thresh}")
                return {
                    'P_opt': P.numpy(),
                    'sum_rate_opt': float(sum_rate.numpy()),
                    'sum_rate_history': sum_rate_history,
                    'converged': True,
                    'n_iter_outer': t + 1,
                    'n_iter_inner_avg': np.mean(inner_iter_counts),
                    'kappa': self.kappa,
                    'power_total': self.power_total,
                    'act_rank': act_rank,
                    'act_pow_ratio': act_pow_ratio,
                    'water_level': water_level
                }
            
            # 수렴 확인 (tolerance 기준)
            if rate_diff < self.eps_outer:
                print(f"  [외부 수렴] iter={t+1}, rate={sum_rate.numpy():.4f} bits/Hz/s")
                return {
                    'P_opt': P.numpy(),
                    'sum_rate_opt': float(sum_rate.numpy()),
                    'sum_rate_history': sum_rate_history,
                    'converged': True,
                    'n_iter_outer': t + 1,
                    'n_iter_inner_avg': np.mean(inner_iter_counts),
                    'kappa': self.kappa,
                    'power_total': self.power_total,
                    'act_rank': act_rank,
                    'act_pow_ratio': act_pow_ratio,
                    'water_level': water_level
                }
            
            sum_rate_prev = sum_rate.numpy()
        
        # 최대 반복 횟수 도달
        return {
            'P_opt': P.numpy(),
            'sum_rate_opt': sum_rate_history[-1],
            'sum_rate_history': sum_rate_history,
            'converged': False,
            'n_iter_outer': self.max_iter_outer,
            'n_iter_inner_avg': np.mean(inner_iter_counts),
            'kappa': self.kappa,
            'power_total': self.power_total,
            'act_rank': act_rank,
            'act_pow_ratio': act_pow_ratio,
            'water_level': water_level
        }
    
    def _compute_fixed_point(self, P: tf.Tensor, outer_iter: int) -> tuple:
        """고정점 방정식 (11) 해결 (최적 수렴점 선택)
        
        모든 iteration을 수행하고 gamma_diff + psi_diff가 최소인 지점의 (γ, ψ) 반환
        
        Args:
            P: [n_ue, n_ue] UE 입력 공분산
            outer_iter: 외부 루프 iteration 번호
        
        Returns:
            (gamma, psi, sum_rate, n_iter)
                gamma: [n_bs] BS 고정점 변수
                psi: [n_ue] UE 고정점 변수
                sum_rate: I(P)
                n_iter: 최적 수렴 iteration 번호
        """
        # @tf.function으로 최적 수렴점 계산
        best_gamma, best_psi, best_idx = self._fixed_point_loop_tf(P)
        
        # 최적 (γ, ψ)로 Xi, R 재계산
        T, R = self._compute_T_R(best_gamma, best_psi)
        Xi = self._compute_Xi_from_T_R(T, R)
        
        # (7) I(P) 계산
        sum_rate = self._compute_sum_rate(P, Xi, R, best_gamma, best_psi)
        
        # 10 iteration마다 수렴 상태 출력
        if (outer_iter + 1) % 10 == 0:
            print(f"  [내부 최적 수렴 t={outer_iter+1}] best_iter={int(best_idx.numpy())+1}/{self.max_iter_inner}")
        
        return best_gamma, best_psi, sum_rate, int(best_idx.numpy()) + 1
    
    @tf.function
    def _fixed_point_loop_tf(self, P: tf.Tensor) -> tuple:
        """고정점 반복 (TensorFlow 정적 그래프)
        
        Args:
            P: [n_ue, n_ue] UE 입력 공분산
        
        Returns:
            (best_gamma, best_psi, best_idx)
        """
        # 초기화
        gamma = tf.ones(self.n_bs, dtype=tf.float32)
        psi = tf.ones(self.n_ue, dtype=tf.float32)
        
        # regularization을 complex64로 캐스팅
        reg_complex = tf.cast(self.reg, tf.complex64)
        
        # History 저장
        gamma_history = tf.TensorArray(tf.float32, size=self.max_iter_inner, dynamic_size=False)
        psi_history = tf.TensorArray(tf.float32, size=self.max_iter_inner, dynamic_size=False)
        diff_history = tf.TensorArray(tf.float32, size=self.max_iter_inner, dynamic_size=False)
        
        # 고정 반복
        for i in tf.range(self.max_iter_inner):
            gamma_prev = tf.identity(gamma)
            psi_prev = tf.identity(psi)
            
            # abs() 클리핑: 음수 방지 (이론적으로 γ, ψ ≥ 0)
            gamma = tf.abs(gamma)
            psi = tf.abs(psi)
            
            # (10) T, R 계산
            T, R = self._compute_T_R(gamma, psi)
            
            # (8) Ξ 계산
            Xi = self._compute_Xi_from_T_R(T, R)
            
            # (11a) γ 업데이트: solve가 inv보다 수치적으로 안정
            I_R = tf.eye(self.n_bs, dtype=tf.complex64) + R + reg_complex * tf.eye(self.n_bs, dtype=tf.complex64)
            I_R_inv_U_bs = tf.linalg.solve(I_R, self.U_bs)
            gamma = tf.math.real(
                tf.reduce_sum(tf.math.conj(self.U_bs) * I_R_inv_U_bs, axis=0)
            )
            
            # (11b) ψ 업데이트: solve가 inv보다 수치적으로 안정
            I_Xi_P = tf.eye(self.n_ue, dtype=tf.complex64) + Xi @ P + reg_complex * tf.eye(self.n_ue, dtype=tf.complex64)
            I_Xi_P_inv_U_ue = tf.linalg.solve(I_Xi_P, self.U_ue)
            psi = tf.math.real(
                tf.reduce_sum(tf.math.conj(self.U_ue) * (P @ I_Xi_P_inv_U_ue), axis=0)
            )
            
            # 수렴도 계산 및 저장
            gamma_diff = tf.norm(gamma - gamma_prev)
            psi_diff = tf.norm(psi - psi_prev)
            total_diff = gamma_diff + psi_diff
            
            gamma_history = gamma_history.write(i, gamma)
            psi_history = psi_history.write(i, psi)
            diff_history = diff_history.write(i, total_diff)
        
        # argmin(diff/iter): 수렴 효율성 기반 선택
        all_diffs = diff_history.stack()
        iter_indices = tf.cast(tf.range(1, self.max_iter_inner + 1), tf.float32)
        diff_per_iter = all_diffs / iter_indices
        best_idx = tf.argmin(diff_per_iter, output_type=tf.int32)
        
        # 최적 (gamma, psi) 추출
        all_gammas = gamma_history.stack()
        all_psis = psi_history.stack()
        
        best_gamma = all_gammas[best_idx]
        best_psi = all_psis[best_idx]
        
        return best_gamma, best_psi, best_idx
    
    @tf.function
    def _compute_T_R(self, gamma: tf.Tensor, psi: tf.Tensor) -> tuple:
        """(10) T, R 행렬 계산
        
        Args:
            gamma: [n_bs] BS 고정점 변수
            psi: [n_ue] UE 고정점 변수
        
        Returns:
            (T, R)
                T: [n_ue, n_ue] T = U_ue diag(Ω^T γ) U_ue^H
                R: [n_bs, n_bs] R = U_bs diag(Ω ψ) U_bs^H
        """
        # T = U_ue diag(Ω^T γ) U_ue^H
        Omega_T_gamma = tf.matmul(tf.transpose(self.Omega), tf.expand_dims(gamma, -1))
        T = self.U_ue @ tf.linalg.diag(tf.cast(tf.squeeze(Omega_T_gamma), tf.complex64)) @ tf.linalg.adjoint(self.U_ue)
        
        # R = U_bs diag(Ω ψ) U_bs^H
        Omega_psi = tf.matmul(self.Omega, tf.expand_dims(psi, -1))
        R = self.U_bs @ tf.linalg.diag(tf.cast(tf.squeeze(Omega_psi), tf.complex64)) @ tf.linalg.adjoint(self.U_bs)
        
        return T, R
    
    @tf.function
    def _compute_Xi_from_T_R(self, T: tf.Tensor, R: tf.Tensor) -> tf.Tensor:
        """(8) Ξ 행렬 계산
        
        Args:
            T: [n_ue, n_ue]
            R: [n_bs, n_bs]
        
        Returns:
            Ξ: [n_ue, n_ue] Ξ = T + H̄^H (I+R)^{-1} H̄
        """
        reg_complex = tf.cast(self.reg, tf.complex64)
        I_R = tf.eye(self.n_bs, dtype=tf.complex64) + R + reg_complex * tf.eye(self.n_bs, dtype=tf.complex64)
        H_mean_H = tf.linalg.adjoint(self.H_mean_ul)
        Xi = T + H_mean_H @ tf.linalg.solve(I_R, self.H_mean_ul)
        return Xi
    
    def _compute_Xi(self, gamma: tf.Tensor, psi: tf.Tensor) -> tf.Tensor:
        """γ, ψ로부터 Ξ 계산
        
        Args:
            gamma: [n_bs]
            psi: [n_ue]
        
        Returns:
            Ξ: [n_ue, n_ue]
        """
        T, R = self._compute_T_R(gamma, psi)
        return self._compute_Xi_from_T_R(T, R)
    
    @tf.function
    def _compute_sum_rate(self, P: tf.Tensor, Xi: tf.Tensor, R: tf.Tensor,
                          gamma: tf.Tensor, psi: tf.Tensor) -> tf.Tensor:
        """(7) I(P) 계산
        
        Args:
            P: [n_ue, n_ue]
            Xi: [n_ue, n_ue]
            R: [n_bs, n_bs]
            gamma: [n_bs]
            psi: [n_ue]
        
        Returns:
            I(P) in bits/Hz/sec (log2)
        """
        I_Xi_P = tf.eye(self.n_ue, dtype=tf.complex64) + Xi @ P
        I_R = tf.eye(self.n_bs, dtype=tf.complex64) + R
        
        logdet1 = tf.math.real(tf.linalg.slogdet(I_Xi_P)[1])
        logdet2 = tf.math.real(tf.linalg.slogdet(I_R)[1])
        
        coupling_term = tf.reduce_sum(
            gamma * tf.squeeze(tf.matmul(self.Omega, tf.expand_dims(psi, -1)))
        )
        
        sum_rate_nats = logdet1 + logdet2 - coupling_term
        
        # Natural log → bits/Hz/sec
        return sum_rate_nats / tf.math.log(2.0)
    
    def _water_filling(self, Xi: tf.Tensor, debug_outer_iter: int = -1) -> tuple:
        """(26,29,30) Water-filling: p_i = (ν - 1/λ_i)^+ where ν = 1/μ
        
        Args:
            Xi: [n_ue, n_ue] 등가 채널
            debug_outer_iter: 외부 iteration 번호 (디버그용)
        
        Returns:
            P: [n_ue, n_ue] 최적 UE 입력 공분산
            act_rank: 활성 rank (최소 전력 임계값 이상)
            act_pow_ratio: 전력 제약 대비 실제 할당 비율 (≈1.0)
            water_level_nu: ν = 1/μ
        """
        # Eigenvalue decomposition
        eigvals, eigvecs = tf.linalg.eigh(Xi)
        eigvals_positive = tf.maximum(tf.math.real(eigvals), 0.0)
        
        # 극소 고유값 필터링: 1/λ overflow 방지
        valid_mask = eigvals_positive > 1e-30
        
        # 채널이 거의 0인 경우
        if tf.reduce_sum(tf.cast(valid_mask, tf.float32)) == 0:
            P = tf.zeros_like(Xi, dtype=tf.complex64)
            return P, tf.constant(0.0), tf.constant(0.0), tf.constant(0.0)
        
        # Water-filling level ν = 1/μ 계산 (유효한 고유값만 사용)
        valid_eigvals = eigvals_positive * tf.cast(valid_mask, tf.float32)
        nu = self._find_inverse_water_level_tf(valid_eigvals, valid_mask)
        
        # p_i = (ν - 1/λ_i)^+ (유효한 고유값만 계산)
        inv_eigvals = tf.where(valid_mask, 1.0 / eigvals_positive, 0.0)
        Lambda_P_raw = tf.maximum(nu - inv_eigvals, 0.0)
        
        # 유효하지 않은 고유값에는 전력 할당 금지
        Lambda_P = Lambda_P_raw * tf.cast(valid_mask, tf.float32)
        
        # 최소 전력 임계값: 평균 전력의 1% 미만은 비활성으로 간주
        avg_power = self.power_total / tf.cast(self.n_ue, tf.float32)
        min_power_threshold = 0.01 * avg_power
        active_power_mask = Lambda_P >= min_power_threshold
        
        # 활성 rank (최소 임계값 이상)
        act_rank = tf.reduce_sum(tf.cast(active_power_mask, tf.float32))
        
        # 전력 제약 대비 실제 할당된 전력 비율
        allocated_power = tf.reduce_sum(Lambda_P)
        act_pow_ratio = allocated_power / tf.constant(self.power_total, dtype=tf.float32)
        
        # P = U_Ξ Λ_P U_Ξ^H
        P = eigvecs @ tf.linalg.diag(tf.cast(Lambda_P, tf.complex64)) @ tf.linalg.adjoint(eigvecs)
        
        return P, act_rank, act_pow_ratio, nu
    
    @tf.function
    def _find_inverse_water_level_tf(self, eigvals: tf.Tensor, valid_mask: tf.Tensor) -> tf.Tensor:
        """Water-filling level ν = 1/μ 계산 (Bisection search)
        
        Args:
            eigvals: [n_ue] 고유값 (유효하지 않은 값은 0)
            valid_mask: [n_ue] 유효한 고유값 mask
        
        Returns:
            nu: ν = 1/μ (수치적으로 안정)
        """
        # 유효한 고유값만 추출
        valid_eigvals = tf.boolean_mask(eigvals, valid_mask)
        inv_valid_eigvals = 1.0 / valid_eigvals
        
        # ν 탐색 범위
        nu_min = 0.0
        nu_max = self.power_total + tf.reduce_max(inv_valid_eigvals)
        power_total_tf = tf.constant(self.power_total, dtype=tf.float32)
        
        # Bisection loop
        def cond(nu_min, nu_max, i):
            return tf.logical_and(i < 100, tf.abs(nu_max - nu_min) >= 1e-10)
        
        def body(nu_min, nu_max, i):
            nu_mid = (nu_min + nu_max) / 2.0
            allocated_power = tf.reduce_sum(tf.maximum(nu_mid - inv_valid_eigvals, 0.0))
            
            nu_min = tf.cond(allocated_power < power_total_tf,
                           lambda: nu_mid,
                           lambda: nu_min)
            nu_max = tf.cond(allocated_power < power_total_tf,
                           lambda: nu_max,
                           lambda: nu_mid)
            
            return nu_min, nu_max, i + 1
        
        nu_min_final, nu_max_final, _ = tf.while_loop(cond, body, [nu_min, nu_max, 0])
        
        return (nu_min_final + nu_max_final) / 2.0

# ===== SECTION 4: P1J_DataProcessor =====
class P1J_DataProcessor:
    """P1I 청크 데이터 로딩 및 DL→UL 전환 + UE 위치별 처리"""
    
    def __init__(self, config: P1J_Config):
        self.config = config
    
    def load_chunk(self, filepath: str) -> dict:
        """P1I 청크 파일 로딩
        
        Args:
            filepath: 청크 파일 경로
        
        Returns:
            dict: 청크 데이터 (BS/UE 키)
        """
        data = np.load(filepath, allow_pickle=True)
        
        # 계층적 딕셔너리로 재구성
        chunk_data = {
            'ue_indices': data['ue_indices'],
            'P1G': {
                'U_BS': data['P1G_U_BS'],
                'U_UE': data['P1G_U_UE'],
                'Omega': data['P1G_Omega'],
                'has_coupling': data['P1G_has_coupling']
            },
            'P1H': {
                'H_mean': data['P1H_H_mean'],
                'has_los': data['P1H_has_los']
            },
            'enhanced_metadata': data['enhanced_metadata'],
            'chunk_metadata': data['chunk_metadata'].item()
        }
        
        return chunk_data
    
    def process_ue(self, ue_idx: int, chunk_data: dict, idx_in_chunk: int) -> dict:
        """단일 UE 위치 처리 (DL→UL 전환 + 최적화)
        
        Args:
            ue_idx: UE 위치 인덱스
            chunk_data: 청크 데이터
            idx_in_chunk: 청크 내 UE 위치 (0-based)
        
        Returns:
            dict: {
                'ue_idx': UE 인덱스,
                'result': 최적화 결과 또는 None,
                'skip_reason': 건너뜀 사유 (해당 시)
            }
        """
        # 메타데이터 확인
        meta = chunk_data['enhanced_metadata'][idx_in_chunk]
        channel_model = meta['channel_model']
        has_coupling = chunk_data['P1G']['has_coupling'][idx_in_chunk]
        
        # Weichselberger 모델 체크
        if not has_coupling:
            return {
                'ue_idx': ue_idx,
                'result': None,
                'skip_reason': f'Kronecker model (no coupling), channel={channel_model}'
            }
        
        # DL 데이터 추출 및 즉시 TensorFlow로 변환
        U_BS = tf.constant(chunk_data['P1G']['U_BS'][idx_in_chunk])
        U_UE = tf.constant(chunk_data['P1G']['U_UE'][idx_in_chunk])
        Omega_dl = tf.constant(chunk_data['P1G']['Omega'][idx_in_chunk])  # [n_ue, n_bs]
        H_mean_dl = tf.constant(chunk_data['P1H']['H_mean'][idx_in_chunk])  # [n_ue, n_bs]
        
        # DL → UL 전환 (TensorFlow GPU)
        Omega_ul = tf.transpose(Omega_dl)  # [n_bs, n_ue]
        H_mean_ul = tf.linalg.adjoint(H_mean_dl)  # [n_bs, n_ue] Hermitian transpose
        
        # NaN 체크 및 처리 (TensorFlow GPU, 복소수는 실수부/허수부 각각 체크)
        is_nan_real = tf.math.is_nan(tf.math.real(H_mean_ul))
        is_nan_imag = tf.math.is_nan(tf.math.imag(H_mean_ul))
        is_nan = tf.logical_or(is_nan_real, is_nan_imag)
        if tf.reduce_any(is_nan):
            H_mean_ul = tf.where(is_nan, tf.cast(0.0, tf.complex64), H_mean_ul)
        
        # Omega 최소값 제한: 평균의 regularization 배수로 thresholding
        omega_mean = tf.reduce_mean(Omega_ul)
        omega_threshold = omega_mean * self.config.regularization
        Omega_ul_safe = tf.maximum(Omega_ul, omega_threshold)
        
        # Optimization 실행
        try:
            optimizer = WeichselbergerCapacityOptimizer(
                U_bs=U_BS,
                U_ue=U_UE,
                Omega_ul=Omega_ul_safe,  # Thresholded Omega 사용
                H_mean_ul=H_mean_ul,
                config=self.config
            )
            
            result = optimizer.optimize()
            
            # 결과 유효성 체크
            sum_rate = result.get('sum_rate_opt', 0.0)
            if sum_rate < 0.01:
                result['warning'] = f'Near-zero capacity: {sum_rate:.2e} (poor channel or negligible Omega)'
            
            return {
                'ue_idx': ue_idx,
                'result': result,
                'skip_reason': None
            }
            
        except Exception as e:
            # Optimization 실패
            return {
                'ue_idx': ue_idx,
                'result': None,
                'skip_reason': f'Optimization error: {str(e)[:100]}'
            }

# ===== SECTION 5: CapacityResults_Manager =====
class CapacityResults_Manager:
    """용량 결과 파일 저장 및 로딩 관리"""
    
    def __init__(self, config: P1J_Config):
        self.config = config
        os.makedirs(self.config.P1J_OUTPUT_DIR, exist_ok=True)
    
    def result_exists(self, area: int, freq: float, ue: int) -> bool:
        """결과 파일 존재 여부 확인"""
        filename = self.config.P1J_FILE_PATTERN.format(area=area, freq=freq, ue=ue)
        filepath = os.path.join(self.config.P1J_OUTPUT_DIR, filename)
        return os.path.exists(filepath)
    
    def save_result(self, area: int, freq: float, ue: int, 
                    result: dict, channel_model: str) -> str:
        """용량 결과 저장
        
        Args:
            area: Area 인덱스
            freq: 주파수 (GHz)
            ue: UE 위치 인덱스
            result: 최적화 결과
            channel_model: 채널 모델 타입
        
        Returns:
            str: 저장된 파일 경로
        """
        filename = self.config.P1J_FILE_PATTERN.format(area=area, freq=freq, ue=ue)
        filepath = os.path.join(self.config.P1J_OUTPUT_DIR, filename)
        
        # 메타데이터
        metadata = {
            'area_idx': area,
            'freq_ghz': freq,
            'ue_idx': ue,
            'channel_model': channel_model,
            'kappa': result.get('kappa', None),
            'power_total': result.get('power_total', None),
            'SNR_dB': self.config.SNR_dB,
            'SNR_linear': self.config.SNR_linear,
            'timestamp': datetime.now().isoformat()
        }
        
        # Water-filling 지표 (TensorFlow → numpy)
        act_rank = float(result['act_rank'].numpy()) if hasattr(result['act_rank'], 'numpy') else float(result['act_rank'])
        act_pow_ratio = float(result['act_pow_ratio'].numpy()) if hasattr(result['act_pow_ratio'], 'numpy') else float(result['act_pow_ratio'])
        water_level = float(result['water_level'].numpy()) if hasattr(result['water_level'], 'numpy') else float(result['water_level'])
        
        save_dict = {
            'P_opt': result['P_opt'],
            'sum_rate_opt': result['sum_rate_opt'],  # bits/Hz/sec
            'sum_rate_history': np.array(result['sum_rate_history']),
            'converged': result['converged'],
            'n_iter_outer': result['n_iter_outer'],
            'n_iter_inner_avg': result['n_iter_inner_avg'],
            'kappa': result.get('kappa', None),
            'power_total': result.get('power_total', None),
            'act_rank': act_rank,
            'act_pow_ratio': act_pow_ratio,
            'water_level': water_level,
            'metadata': metadata
        }
        
        np.savez_compressed(filepath, **save_dict)
        
        return filepath
    
    def append_to_csv(self, area: int, freq: float, ue: int, 
                      result: dict, channel_model: str, elapsed_time: float):
        """실시간 CSV에 결과 추가
        
        Args:
            area: Area 인덱스
            freq: 주파수 (GHz)
            ue: UE 위치 인덱스
            result: 최적화 결과
            channel_model: 채널 모델 타입
            elapsed_time: 처리 시간 (초)
        """
        # CSV 파일명에 타임스탬프 (UTC+9, 최초 생성 시에만)
        if not hasattr(self, '_csv_timestamp'):
            from datetime import timedelta
            utc_plus_9 = datetime.utcnow() + timedelta(hours=9)
            self._csv_timestamp = utc_plus_9.strftime('%Y%m%d_%H%M%S')
        
        csv_filename = f"Area{area}_{freq}GHz_Capacity_{self._csv_timestamp}.csv"
        csv_filepath = os.path.join(self.config.P1J_OUTPUT_DIR, csv_filename)
        
        # CSV 헤더 (간소화)
        headers = [
            'ue', 'rate', 'conv', 'iter', 'rank', 'pow_r', 'mu', 'time_s', 'kappa', 'ch_model'
        ]
        
        # 채널 모델 약어: Ray/Ric + W/K, DLoS (deterministic LoS)
        if 'Rician' in channel_model and 'Weichsel' in channel_model:
            ch_abbr = 'RicW'
        elif 'Rayleigh' in channel_model and 'Weichsel' in channel_model:
            ch_abbr = 'RayW'
        elif 'Rician' in channel_model and 'Kron' in channel_model:
            ch_abbr = 'RicK'
        elif 'Rayleigh' in channel_model and 'Kron' in channel_model:
            ch_abbr = 'RayK'
        else:
            ch_abbr = 'DLoS'  # Deterministic LoS (no stochastic component)
        
        # Water-filling 지표 (TensorFlow → numpy)
        act_rank = float(result['act_rank'].numpy()) if hasattr(result['act_rank'], 'numpy') else float(result['act_rank'])
        act_pow_ratio = float(result['act_pow_ratio'].numpy()) if hasattr(result['act_pow_ratio'], 'numpy') else float(result['act_pow_ratio'])
        water_level = float(result['water_level'].numpy()) if hasattr(result['water_level'], 'numpy') else float(result['water_level'])
        
        # 데이터 행 (자릿수 정리)
        row = [
            f"{ue:4d}",
            f"{result['sum_rate_opt']:7.3f}",  # bits/Hz/sec
            'Y' if result['converged'] else 'N',
            f"{result['n_iter_outer']:2d}",
            f"{int(act_rank):2d}",
            f"{act_pow_ratio:5.3f}",
            f"{water_level:8.2e}",
            f"{elapsed_time:6.2f}",
            f"{result.get('kappa', 0.0):.2e}",
            ch_abbr
        ]
        
        # CSV 파일 존재 여부 확인
        file_exists = os.path.exists(csv_filepath)
        
        # CSV에 append
        with open(csv_filepath, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(headers)
            writer.writerow(row)
    
    def load_result(self, area: int, freq: float, ue: int) -> dict:
        """저장된 용량 결과 로딩"""
        filename = self.config.P1J_FILE_PATTERN.format(area=area, freq=freq, ue=ue)
        filepath = os.path.join(self.config.P1J_OUTPUT_DIR, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Result file not found: {filepath}")
        
        data = np.load(filepath, allow_pickle=True)
        
        result = {
            'P_opt': data['P_opt'],
            'sum_rate_opt': float(data['sum_rate_opt']),
            'sum_rate_history': data['sum_rate_history'],
            'converged': bool(data['converged']),
            'n_iter_outer': int(data['n_iter_outer']),
            'n_iter_inner_avg': float(data['n_iter_inner_avg']),
            'kappa': data.get('kappa', None),
            'power_total': data.get('power_total', None),
            'metadata': data['metadata'].item()
        }
        
        return result

# ===== SECTION 6: main =====
def main():
    """P1J Weichselberger 업링크 용량 계산 메인 실행"""
    
    print("=" * 80)
    print("P1J: Weichselberger SU-MIMO 업링크 채널 용량 계산 (Wen2011)")
    print("=" * 80)
    print()
    
    # 설정 초기화
    config = P1J_Config()
    
    if not config.chunk_info:
        print("경고: 처리할 P1I 청크 데이터가 없습니다.")
        return
    
    # 프로세서 및 매니저 초기화
    processor = P1J_DataProcessor(config)
    manager = CapacityResults_Manager(config)
    
    # 전체 통계
    total_ues = sum(len(ue_indices) for _, _, _, _, _, _, ue_indices in config.chunk_info)
    
    print(f"\n처리 계획:")
    print(f"  - 총 청크: {len(config.chunk_info)}개")
    print(f"  - 총 UE 위치: {total_ues}개")
    print(f"\n설정:")
    print(f"  - 채널 정규화: Wen2011 식 (4) - K-팩터 보존, 목표 ρ 달성")
    print(f"  - SNR: {config.SNR_dB} dB (linear: {config.SNR_linear:.0f})")
    print(f"  - 전력 제약: tr(P) = N_ue × SNR = N_ue × {config.SNR_linear:.0f}")
    print()
    
    # 청크 처리 시작
    overall_start = time.time()
    ue_count = 0
    processed_count = 0
    skipped_count = 0
    total_ue_time = 0.0  # 누적 처리 시간
    
    for chunk_idx, (area, freq, chunk_num, ue_start, ue_end, filepath, _) in enumerate(config.chunk_info, 1):
        print(f"\n{'='*80}")
        print(f"[{chunk_idx}/{len(config.chunk_info)}] Area{area}_{freq}GHz Chunk {chunk_num} 처리 중")
        print(f"{'='*80}")
        
        # 청크 로딩
        chunk_load_start = time.time()
        chunk_data = processor.load_chunk(filepath)
        chunk_load_time = time.time() - chunk_load_start
        
        ue_indices = chunk_data['ue_indices']
        n_ue_in_chunk = len(ue_indices)
        
        print(f"청크 로딩 완료: {n_ue_in_chunk}개 UE ({chunk_load_time:.2f}s)")
        print()
        
        # 각 UE 처리
        for idx_in_chunk, ue in enumerate(ue_indices):
            ue_count += 1
            
            # 기존 결과 확인
            if manager.result_exists(area, freq, ue):
                print(f"[{ue_count:>{len(str(total_ues))}}/{total_ues}] "
                      f"UE{ue} → 건너뜀 (이미 존재)")
                skipped_count += 1
                continue
            
            # UE 처리
            ue_start_time = time.time()
            proc_result = processor.process_ue(ue, chunk_data, idx_in_chunk)
            ue_time = time.time() - ue_start_time
            
            if proc_result['result'] is None:
                # 건너뜀
                print(f"[{ue_count:>{len(str(total_ues))}}/{total_ues}] "
                      f"UE{ue} → 건너뜀: {proc_result['skip_reason']}")
                skipped_count += 1
            else:
                # 성공
                result = proc_result['result']
                meta = chunk_data['enhanced_metadata'][idx_in_chunk]
                channel_model = meta['channel_model']
                
                # 결과 저장
                saved_path = manager.save_result(area, freq, ue, result, channel_model)
                file_size_kb = os.path.getsize(saved_path) / 1024
                
                # 실시간 CSV 저장
                manager.append_to_csv(area, freq, ue, result, channel_model, ue_time)
                
                conv_str = "수렴" if result['converged'] else "미수렴"
                kappa_val = result.get('kappa', 0)
                
                # 평균 시간 및 남은 시간 계산
                total_ue_time += ue_time
                avg_time = total_ue_time / processed_count if processed_count > 0 else 0
                remaining_ues = total_ues - ue_count
                eta_sec = avg_time * remaining_ues
                eta_min = eta_sec / 60
                eta_hr = eta_min / 60
                
                # ETA 포맷
                if eta_hr >= 1:
                    eta_str = f"{eta_hr:.1f}h"
                elif eta_min >= 1:
                    eta_str = f"{eta_min:.1f}m"
                else:
                    eta_str = f"{eta_sec:.0f}s"
                
                # Warning 표시 (near-zero capacity)
                warning_str = ""
                if 'warning' in result:
                    warning_str = f" [WARNING: {result['warning']}]"
                
                print(f"[{ue_count:>{len(str(total_ues))}}/{total_ues}] "
                      f"UE{ue:4d} → "
                      f"I={result['sum_rate_opt']:6.2f}, "
                      f"it={result['n_iter_outer']:2d}, "
                      f"{conv_str[:2]:2s}, "
                      f"{ue_time:5.1f}s "
                      f"| avg={avg_time:5.1f}s, ETA={eta_str}{warning_str}")
                processed_count += 1
            
            # 진행률 출력 및 화면 클리어
            if config.ENABLE_SCREEN_CLEAR and ue_count % config.PROGRESS_CLEAR_INTERVAL == 0:
                current_time = time.time()
                elapsed_time = current_time - overall_start
                avg_time_per_ue = elapsed_time / ue_count
                remaining_ues = total_ues - ue_count
                estimated_remaining = avg_time_per_ue * remaining_ues
                
                # 화면 클리어
                if JUPYTER_AVAILABLE:
                    clear_output(wait=True)
                else:
                    import platform
                    os.system('cls' if platform.system() == 'Windows' else 'clear')
                
                # 시간 포맷팅
                def format_time(seconds):
                    if seconds < 60:
                        return f"{seconds:.1f}초"
                    elif seconds < 3600:
                        return f"{int(seconds//60)}분 {int(seconds%60)}초"
                    else:
                        return f"{int(seconds//3600)}시간 {int((seconds%3600)//60)}분"
                
                # 진행 상황 요약
                print("P1J Weichselberger 업링크 용량 계산 진행률:")
                print(f"현재 완료율: {ue_count/total_ues*100:.1f}% ({ue_count}/{total_ues} UE)")
                print(f"처리 완료: {processed_count}개, 건너뜀: {skipped_count}개")
                print(f"전체 경과 시간: {format_time(elapsed_time)}")
                print(f"평균 처리 속도: {avg_time_per_ue:.2f}초/UE")
                print(f"예상 남은 시간: {format_time(estimated_remaining)}")
                print("-" * 80)
    
    # 완료 메시지
    total_time = time.time() - overall_start
    print(f"\n{'='*80}")
    print(f"=== P1J Weichselberger 업링크 용량 계산 완료 ===")
    print(f"{'='*80}")
    print(f"총 {ue_count}개 UE 처리")
    print(f"  - 최적화 완료: {processed_count}개")
    print(f"  - 건너뜀: {skipped_count}개")
    print(f"총 소요 시간: {total_time/60:.1f}분")
    print(f"저장 위치: {config.P1J_OUTPUT_DIR}")
    print()

if __name__ == "__main__":
    main()
