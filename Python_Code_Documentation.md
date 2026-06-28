# Python 코드 문서

이 문서는 워크스페이스 내 각 폴더에 포함된 Python 코드들의 목적, 입력/출력 데이터, 주요 기능을 정리한 것입니다.

---

## 📁 250912 py_stable

안정화된 버전의 Ray Tracing → OFDM 채널 → 통계 분석 파이프라인

### P1_RT_to_Rays_2509v6.py

| 항목 | 내용 |
|------|------|
| **목적** | Sionna RT 기반 Ray Tracing 시뮬레이션 및 Ray 데이터 추출 |
| **입력** | Blender 3D 씬 파일 (`.xml`), Area 설정 |
| **출력** | `Area{X}_{freq}GHz_Rays_ALL_RXs.npz` |

**주요 기능:**
- Area 기반 TX/RX 그리드 배치 시스템
- Multi-frequency 지원 (3.5, 7.5, 28 GHz)
- ITU 재질 속성 튜닝 (산란/XPD 계수)
- float32 데이터 저장 (메모리 50% 절감)

**추출 데이터:**
- `tau`: 경로 지연 [s]
- `power`: 경로 전력
- `theta_r_deg`, `phi_r_deg`: 도착각 (ZoA, AoA) [deg]
- `theta_t_deg`, `phi_t_deg`: 출발각 (ZoD, AoD) [deg]
- `rx_indices`: 수신기 인덱스
- `counts`: RX별 유효 Ray 수

---

### P2_Rays_to_OFDM_Ch_2509v5.py

| 항목 | 내용 |
|------|------|
| **목적** | Ray 데이터를 OFDM 채널 행렬로 변환 |
| **입력** | `Rays_*.mat` 파일 |
| **출력** | `OFDM_Ch_*.mat` 파일 |

**주요 파라미터:**
- 주파수: 28 GHz (FR2)
- 대역폭: 400 MHz
- 서브캐리어 수: 3168개
- 안테나 구성: UPA (Uniform Planar Array)
  - TX: 8×8 (64 안테나)
  - RX: 4×4 (16 안테나)

**추출 데이터:**
- `H_ofdm`: OFDM 채널 행렬 [N_sc × N_rx × N_tx]
- `H_ofdm_normalized`: 정규화된 채널 행렬

---

### P3_OFDM_Ch_to_ChMeanCov_2509v5.py

| 항목 | 내용 |
|------|------|
| **목적** | OFDM 채널의 평균 및 공분산 행렬 계산 |
| **입력** | `OFDM_Ch_*.mat` 파일 |
| **출력** | `ChMeanCov_*.mat` 파일 |

**추출 데이터:**
- `H_mean`: 채널 평균 [N_rx × N_tx]
- `H_cov`: 채널 공분산 행렬 [N_rx*N_tx × N_rx*N_tx]
- `H_corr`: 채널 상관 행렬

---

## 📁 250922 python

빔포밍 분석 및 레이어 채널 처리가 추가된 확장 파이프라인

### P1A_RT_to_Rays_2509v6.py
Ray Tracing 결과 파싱 (P1과 동일)

### P1B_Valid_RX_Filter_2509v1.py

| 항목 | 내용 |
|------|------|
| **목적** | 유효한 수신기 위치 필터링 |
| **입력** | `Rays_*.mat` 파일 |
| **출력** | `ValidRX_*.mat` 파일 |

**필터링 기준:**
- 최소 수신 전력 임계값
- 최소 경로 수
- 유효 커버리지 영역

---

### P2A_BF_Pair_2509v2.py

| 항목 | 내용 |
|------|------|
| **목적** | 빔포밍 쌍(TX-RX 빔 조합) 생성 |
| **입력** | `Rays_*.mat` 파일 |
| **출력** | `BF_Pair_*.mat` 파일 |

**추출 데이터:**
- `bf_gain`: 빔포밍 이득
- `best_tx_beam_idx`: 최적 TX 빔 인덱스
- `best_rx_beam_idx`: 최적 RX 빔 인덱스
- `beam_pair_gain`: 빔 쌍별 이득 행렬

---

### P2B_BF_Analysis_2509v2.py
빔포밍 성능 분석 및 시각화

### P3A_Rays_to_Layer_Ch_2509v1.py
Ray 데이터를 레이어별 채널로 변환 (SVD 기반)

### P4A_Layer_Ch_to_ChMeanCov_2509v1.py
레이어 채널의 통계적 특성 계산

---

## 📁 250924 python

안테나 요소(AE) 분석 및 분리성 분석이 추가된 최신 파이프라인

### P1C_Rays_to_AE_OFDM_Ch_2509v1.py

| 항목 | 내용 |
|------|------|
| **목적** | Ray 데이터를 안테나 요소(AE) 단위 OFDM 채널로 변환 |
| **입력** | `Rays_*.mat` 파일 |
| **출력** | `AE_OFDM_Ch_*.mat` 파일 |

**추출 데이터:**
- `H_ae`: 안테나 요소별 채널 행렬
- `ae_response`: 안테나 요소 응답
- `polarization_matrix`: 편파 행렬

---

### P1D_AE_OFDM_Ch_to_Gram_2509v1.py

| 항목 | 내용 |
|------|------|
| **목적** | AE OFDM 채널의 Gram 행렬 계산 |
| **입력** | `AE_OFDM_Ch_*.mat` 파일 |
| **출력** | `Gram_*.mat` 파일 |

**추출 데이터:**
- `G`: Gram 행렬 (H^H × H)
- `eigenvalues`: 고유값
- `eigenvectors`: 고유벡터
- `channel_capacity`: 채널 용량

---

### P1E_AE_OFDM_Ch_Separability_2509v1.py

| 항목 | 내용 |
|------|------|
| **목적** | 채널 분리성(Separability) 분석 |
| **입력** | `AE_OFDM_Ch_*.mat` 파일 |
| **출력** | `Separability_*.mat` 파일 |

**추출 데이터:**
- `separability_index`: 분리성 지표
- `kronecker_error`: Kronecker 근사 오차
- `spatial_correlation`: 공간 상관
- `frequency_correlation`: 주파수 상관

---

### P2X_SU_BM1_Path_Specific.py

| 항목 | 내용 |
|------|------|
| **목적** | 단일 사용자(SU) 빔 관리 - 경로별 분석 |
| **입력** | `Rays_*.mat`, `BF_Pair_*.mat` 파일 |
| **출력** | `SU_BM_PathSpecific_*.mat` 파일 |

**추출 데이터:**
- `path_beam_gain`: 경로별 빔 이득
- `los_beam_idx`: LOS 경로 최적 빔
- `nlos_beam_gain`: NLOS 경로 빔 이득

---

## 📁 251004_Ch_Separ_(CorrRician)

채널 분리성 분석 및 Correlated Rician 채널 검증 파이프라인

### P1D_Rays_to_AE_Separability_2510v3.py

| 항목 | 내용 |
|------|------|
| **목적** | MIMO 채널의 Kronecker/Weichselberger 모델 분리성 분석 |
| **입력** | P1B Valid RXs 결과 (`Area{X}_{freq}GHz_Rays_Valid_RXs.npz`) |
| **출력** | 분리성 분석 CSV (`P1D_Separability_Results/`) |

**핵심 기술:**
- Block Welford 알고리즘: 온라인 누적 공분산 계산
- TensorFlow JIT 컴파일 최적화
- Column-Major vec() 구현 (MATLAB 호환)

**주요 클래스:**
- `P1D_Config`: P1B 데이터 기반 동적 설정
- `SeparabilityAnalyzer`: 분리성 메트릭 계산 (ε_λ, ε_U)
- `ChannelAnalyzer`: 채널 생성 + 온라인 누적 orchestration
- `BlockWelfordAccumulator`: 수치 안정적 공분산 누적
- `ChCoeGen`: Ray → AE OFDM 채널 변환 (TR 38.901 기반)

**추출 데이터:**
- `epsilon_lambda`: 고유값 분리성 오차
- `epsilon_U`: 고유벡터 분리성 오차
- `R_BS`, `R_UE`: Marginal 공분산 행렬
- 모델 선택 결과 (Kronecker/Weichselberger)

---

### P1E_Validation_Correlated_Rician_2510v3.py

| 항목 | 내용 |
|------|------|
| **목적** | P1D 코드 무결성 검증 (Correlated Rician Channel 테스트) |
| **입력** | 테스트 파라미터 (K_Ric_dB, alpha_b, alpha_u) |
| **출력** | 검증 결과 CSV (`P1E_Validation_Results/`) |

**검증 항목:**
- Ground Truth: R_BS_tru ⊗ R_UE_tru → R_AE_tru
- 온라인 누적 공분산 수렴성 모니터링
- Kronecker 모델 분리성 성립 확인

**주요 클래스:**
- `P1E_Config`: P1D_Config 기반 1:1 정렬 테스트 설정
- `CorrelatedRicianGenerator`: Exponential Correlation Model 기반 테스트 데이터 생성
- `P1D_ModuleValidator`: P1D 모듈 검증기
- `ConvergenceMonitor`: 수렴성 모니터링 및 CSV 저장

**추출 데이터:**
- `R_AE_err_fro_rel`: R_AE 상대 오차
- `R_BS_err_fro_rel`, `R_UE_err_fro_rel`: Marginal 공분산 오차
- `eps_d`, `eps_U`: 분리성 메트릭

---

## 📁 251009_CCM_Collection (8 GB)

대용량 Marginal CCM (Channel Covariance Matrix) 수집 파이프라인

### P1F_Rays_to_Marginal_CCM_2510v1.py

| 항목 | 내용 |
|------|------|
| **목적** | SU-MIMO 채널의 Marginal CCM (R_BS, R_UE) 추정 및 저장 |
| **입력** | P1B Valid RXs 결과 |
| **출력** | `Area{area}_{freq}GHz_RX{rx}_Marginal_CCM.npz` (RX별 개별 파일) |

**핵심 설계:**
- R_AE = R_BS ⊗ R_UE에서 R_BS (TX marginal), R_UE (RX marginal) 추정
- 온라인 누적 공분산으로 메모리 효율적 대용량 데이터 처리
- R_AE 미포함 (메모리 절약: [16K×16K] complex64 ≈ 2GB/RX)

**주요 클래스:**
- `P1F_Config`: Marginal CCM 수집용 설정
- `ChCoeGen`: Ray → AE OFDM 채널 생성
- `BlockWelfordAccumulator`: 온라인 공분산 누적
- `MarginalCCM_Engine`: 채널 생성 + 누적 + Marginal CCM 추출
- `MarginalCCM_Manager`: 파일 저장/로딩 관리

**추출 데이터:**
- `R_BS`: [n_t × n_t] TX marginal covariance
- `R_UE`: [n_r × n_r] RX marginal covariance
- `metadata`: 샘플 수, 안테나 구성, 검증 정보

**시스템 파라미터:**
- OFDM FFT: 128 (메모리 절감)
- Static Channel Realizations: 256
- 총 OFDM 샘플: 32,768/RX
- 안테나: TX 1024개 (8×8 패널 × 4×4 요소), RX 16개

---

## 📊 전체 파이프라인 요약

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           데이터 처리 파이프라인                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  [Blender 3D Scene + Sionna RT]                                             │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────────┐                                                        │
│  │ P1A: RT → Rays  │ ─────────────────────────────────────────────────┐     │
│  │ (Ray Tracing)   │                                                  │     │
│  └─────────────────┘                                                  │     │
│         │                                                             │     │
│         ▼                                                             │     │
│  ┌─────────────────┐                                                  │     │
│  │ P1B: Valid RX   │ (음수 지연 제거, LoS/NLoS 필터링)                  │     │
│  │     Filter      │                                                  │     │
│  └─────────────────┘                                                  │     │
│         │                                                             │     │
│         ├──────────────────┬──────────────────┬───────────────────────┤     │
│         ▼                  ▼                  ▼                       │     │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                │     │
│  │ P1C/P1D:    │    │ P2A: BF     │    │ P3A: Layer  │                │     │
│  │ AE OFDM Ch  │    │ Pair        │    │ Channel     │                │     │
│  │ + 분리성    │    └─────────────┘    └─────────────┘                │     │
│  └─────────────┘           │                  │                       │     │
│         │                  ▼                  ▼                       │     │
│         ▼           ┌─────────────┐    ┌─────────────┐                │     │
│  ┌─────────────┐    │ P2B: BF     │    │ P4A: Layer  │                │     │
│  │ P1E: 검증   │    │ Analysis    │    │ MeanCov     │                │     │
│  │ (Corr.Ric.) │    └─────────────┘    └─────────────┘                │     │
│  └─────────────┘           │                                          │     │
│         │                  ▼                                          │     │
│         ▼           ┌─────────────┐                                   │     │
│  ┌─────────────┐    │ P2X: Path   │                                   │     │
│  │ P1F: CCM    │    │ Specific    │                                   │     │
│  │ Collection  │    └─────────────┘                                   │     │
│  └─────────────┘                                                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔧 공통 시스템 파라미터

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| 중심 주파수 | 7.5 GHz (기본) | FR1/FR2 대역 지원 |
| OFDM FFT | 128~256 | 서브캐리어 수 |
| OFDM SCS | 120 kHz | 서브캐리어 간격 |
| TX 안테나 | 8×8 패널 × 4×4 요소 | 1024개 안테나 요소 |
| RX 안테나 | 4×4 UPA | 16개 안테나 요소 |
| 안테나 간격 | 0.5λ | 반파장 간격 |
| GPU | NVIDIA L40S | 45GB VRAM |

---

## 📝 버전 히스토리

- **250912 py_stable**: 기본 파이프라인 (RT → OFDM → 통계)
- **250922 python**: 빔포밍 분석 + 레이어 채널 추가
- **250924 python**: AE 레벨 분석 + 분리성 분석 + 경로별 빔 분석 추가
- **251004_Ch_Separ_(CorrRician)**: Kronecker 모델 분리성 분석 + Correlated Rician 검증
- **251009_CCM_Collection**: 대용량 Marginal CCM 수집 (RX별 독립 파일 저장)

---

## 🔑 핵심 알고리즘

### Block Welford 알고리즘
온라인 공분산 누적을 위한 수치 안정적 알고리즘
- Sub-matrix 분할 관리로 메모리 효율화
- (n_r × n_r) 작은 블록 단위 독립 업데이트

### Column-Major vec() 구현
MATLAB LinearOperatorKronecker와 호환되는 벡터화
```python
def vec_mat_py_tf(H_tf):
    # [n_r, n_t] → [n_r*n_t] (Column-Major)
    return tf.reshape(tf.transpose(H_tf), [-1])
```

### Kronecker 분리성 메트릭
- ε_λ (epsilon_lambda): 고유값 분리성 오차
- ε_U (epsilon_U): 고유벡터 분리성 오차
- 임계값: 0.1 이하면 Kronecker 모델 적합
