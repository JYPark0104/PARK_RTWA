# Sionna RT 기반 채널 모델링 파이프라인 (P1-P3) 개요

이 문서는 3D 환경 시뮬레이션부터 채널 공분산 행렬(CCM) 생성까지 이어지는 Sionna RT 기반 파이프라인의 전체 구조와 데이터 흐름을 설명합니다. 각 단계는 독립적인 Python 스크립트(P1, P2, P3)로 구성되어 있으며, 설정 파일을 통해 유기적으로 연동됩니다.

## 전체 파이프라인 구조

파이프라인은 P1, P2, P3의 세 단계로 구성되며, 각 단계의 출력은 다음 단계의 입력으로 사용되는 직선적인 데이터 흐름을 가집니다.

```
[P1: 3D Scene → Ray Data]
  ├─ 입력: 3D Scene (.xml), 시뮬레이션 설정 (P1_Config)
  ├─ 처리: P1_RT_to_Rays_250806v4.py
  │   └─ Ray Tracing 및 TR 38.901 통계 모델 적용
  └─ 출력: 통계적 레이 데이터 (.npy) → (Sionna_RT_Results/)
      │
      ▼
[P2: Ray Data → OFDM Channel]
  ├─ 입력: 통계적 레이 데이터 (P1 출력)
  ├─ 처리: P2_Rays_to_OFDM_Ch_250809v2.py
  │   └─ 안테나 어레이 및 도플러 효과 적용
  └─ 출력: 시변 OFDM 채널 (.npy) → (OFDM_Ch_Results/)
      │
      ▼
[P3: OFDM Channel → Channel Statistics]
  ├─ 입력: 시변 OFDM 채널 (P2 출력)
  ├─ 처리: P3_OFDM_Ch_to_ChMeanCov_250817v2.py
  │   └─ 채널 평균 및 공분산 계산/보정
  └─ 출력: 채널 공분산 행렬(CCM) 및 채널 평균 (.npz) → (ChMeanCov_Results/)
```

---

## 1. P1: 3D 환경 기반 전파 경로 및 통계적 레이 생성 (`P1_RT_to_Rays_250806v4.py`)

### 1.1. 목적

3D 도시 환경 모델(.xml)을 기반으로 Sionna RT 엔진을 사용하여 물리적인 전파 경로(Path)를 계산하고, 이를 3GPP TR 38.901 표준에 따라 통계적인 레이(Ray) 데이터로 변환하여 저장합니다.

### 1.2. 주요 입력

-   **3D Scene 파일**: `Jonggak.xml` (Mitsuba3 렌더러 형식)
-   **설정 파일**: `P1_Config` 클래스 내 파라미터
    -   `AREA_INDICES`: 시뮬레이션을 수행할 지역 ID 목록
    -   `FREQUENCY_CONFIGS`: 시뮬레이션 주파수 (GHz) 목록
    -   `AREA_CONFIGS`: 지역별 TX/RX 위치 및 그리드 설정
    -   `ENABLE_RAY_SAVING`: 레이 데이터 저장 여부

### 1.3. 통합 설정 (`P1_Config`)

`P1_Config` 클래스는 전체 파이프라인의 시작점으로서, 모든 시뮬레이션 시나리오를 정의하고 제어하는 중앙 관제 센터 역할을 합니다. 사용자는 이 클래스의 속성을 변경하여 원하는 실험 환경을 손쉽게 구성할 수 있습니다.

-   **시나리오 제어**: `AREA_INDICES`와 `FREQUENCY_CONFIGS`의 조합으로 다중 지역, 다중 주파수 시뮬레이션을 자동으로 수행합니다. 예를 들어, 2개의 Area와 3개의 주파수를 설정하면 총 6개의 시나리오가 순차적으로 실행됩니다.
-   **T/RX 배치**: `AREA_CONFIGS` 딕셔너리를 통해 각 Area별로 서로 다른 송신기(TX) 위치와 수신기(RX) 그리드 배치를 정의할 수 있어, 다양한 커버리지 분석이 가능합니다.
-   **파이프라인 제어**: `ENABLE_PATH_SAVING`, `ENABLE_RAY_GENERATION` 등의 플래그를 통해 Ray Tracing만 수행할지, 통계적 레이 생성까지 진행할지를 선택할 수 있습니다.

### 1.4. 핵심 메서드

-   **`Pipeline.execute_main()`**: 파이프라인의 메인 진입점으로, `P1_Config` 설정을 읽어 다중 Area, 다중 주파수 루프를 실행하고, 각 시나리오에 대해 `process_frequency_memory_only()`를 호출합니다.
-   **`PathRT.trace_rays()`**: Sionna RT의 `PathSolver`를 호출하여 3D 씬과 현재 설정된 주파수를 기반으로 물리적 전파 경로를 계산하는 가장 핵심적인 RT 연산 부분입니다.
-   **`RayGen.subrayProposed()`**: 계산된 단일 물리 경로(Cluster)를 입력받아, TR 38.901 표준의 통계 모델(지연, 각도 확산)을 적용하여 다수의 통계적 레이(Sub-ray)를 생성합니다.
-   **`DataIO.save_rays_v4()`**: 생성된 최종 레이 데이터를 `P1_Config`에 정의된 명명 규칙에 따라 `.npy` 파일로 저장합니다.

### 1.5. 주요 출력

-   **레이 데이터 파일 (`.npy`)**:
    -   `power`: 레이별 수신 전력
    -   `tau`: 레이별 지연 시간
    -   `theta_r`, `phi_r`: 도착 각도 (Zenith, Azimuth)
    -   `theta_t`, `phi_t`: 출발 각도 (Zenith, Azimuth)
    -   저장 위치: `Sionna_RT_Results/`

---

## 2. P2: 레이 데이터를 OFDM 채널로 변환 (`P2_Rays_to_OFDM_Ch_250809v2.py`)

### 2.1. 목적

P1에서 생성된 통계적 레이 데이터를 입력받아, 안테나 어레이 특성과 사용자 이동성(도플러 효과)을 적용하여 시변(Time-Varying) MIMO-OFDM 채널 계수를 생성합니다.

### 2.2. 주요 입력

-   **P1 레이 데이터 (`.npy`)**: `Sionna_RT_Results/` 폴더의 모든 레이 데이터 파일
-   **설정 파일**: `P2_Config` 클래스 내 파라미터
    -   `OFDM_FFT`, `OFDM_SCS`: OFDM 시스템 파라미터
    -   `TX_Array`, `RX_Array`: 송수신 안테나 어레이 구성 (패널, 소자 개수, 간격 등)
    -   `Topology_Statistics`: 사용자 이동성(속도, 방향) 통계 모델

### 2.3. 통합 설정 (`P2_Config`)

`P2_Config` 클래스는 P1의 출력에 의존하여 동적으로 설정을 구성하는 특징을 가집니다. P1과 P2를 분리하여 실행할 수 있도록, P1의 결과물이 저장된 폴더를 스캔하여 어떤 Area, 주파수, RX 조합을 처리해야 할지 스스로 판단합니다.

-   **동적 시나리오 감지**: `detect_p1_data()` 메서드가 `Sionna_RT_Results` 폴더를 탐색하여 파일명에 포함된 정보를 기반으로 처리해야 할 시나리오 목록(`area_indices`, `fcs`, `RX_indices`)을 자동으로 생성합니다. 이를 통해 P1의 설정 변경에 유연하게 대응합니다.
-   **시스템 파라미터 정의**: P1 데이터와 독립적인 OFDM 시스템, 안테나 어레이, 도플러 모델과 관련된 파라미터들을 정의합니다. 예를 들어, `OFDM_FFT`, `TX_Array` 설정은 P2에서 채널의 형태를 결정하는 중요한 요소입니다.
-   **독립성**: `P1_Config`를 직접 참조하지 않고, 오직 파일 기반으로만 P1과 데이터를 주고받기 때문에 P1과 P2 스크립트 간의 의존성이 낮아져 독립적인 실행과 테스트가 용이합니다.

### 2.4. 핵심 메서드

-   **`main()`**: `P2_Config`가 자동으로 감지한 Area, 주파수, RX 조합에 대한 3중 루프를 실행하며, 각 조합에 대해 채널 생성을 수행하고 결과를 저장합니다.
-   **`ChannelCoefficientsGeneratorJIN._compute_ch_mimo_ofdm_38901_static()`**: P1의 레이 데이터(각도, 지연, 전력)에 `TX_Array`, `RX_Array`에 정의된 안테나 특성(패턴, 편파, 소자 위치)을 결합하여, 도플러 효과가 적용되기 전의 **정적 다중 경로 채널**을 `delay-bin` 영역에서 계산합니다.
-   **`ChannelCoefficientsGeneratorJIN._apply_doppler_ch_mimo_ofdm_freq()`**: 정적 채널에 `Topology`로 모델링된 사용자 이동 속도 벡터를 적용하여 각 레이에 대한 **도플러 위상 천이**를 계산합니다. 이후 `FFT`를 통해 최종적으로 시간과 주파수에 따라 변하는 MIMO-OFDM 채널을 생성합니다.

### 2.5. 주요 출력

-   **OFDM 채널 데이터 (`.npy`)**:
    -   형상(shape): `[num_ofdm_symbols, num_subcarriers, num_rx_antennas, num_tx_antennas]`
    -   시변 MIMO-OFDM 채널 계수 `H` 행렬
    -   저장 위치: `OFDM_Ch_Results/`

---

## 3. P3: OFDM 채널 통계 분석 (`P3_OFDM_Ch_to_ChMeanCov_250817v2.py`)

### 3.1. 목적

P2에서 생성된 시변 OFDM 채널 데이터를 입력받아, 시간-주파수 영역에 대한 통계적 평균을 계산하여 채널의 장기적인 특성인 **채널 평균(Channel Mean)**과 **채널 공분산 행렬(Channel Covariance Matrix, CCM)**을 추출합니다.

### 3.2. 주요 입력

-   **P2 OFDM 채널 데이터 (`.npy`)**: `OFDM_Ch_Results/` 폴더의 모든 채널 데이터 파일
-   **설정 파일**: `P3_Config` 클래스 내 파라미터
    -   `sampling_ratio`: 통계 계산에 사용할 채널 데이터 샘플링 비율
    -   `ccm_correction`: CCM이 Positive Semi-Definite(PSD) 특성을 갖도록 보정하는 알고리즘 설정

### 3.3. 통합 설정 (`P3_Config`)

`P3_Config` 클래스는 `P2_Config`와 유사하게, 이전 단계의 출력 폴더(`OFDM_Ch_Results`)를 스캔하여 처리 대상을 동적으로 결정합니다. 이를 통해 다중 Area 환경에서 생성된 복잡한 결과물들을 체계적으로 분석하고 관리합니다.

-   **동적 Area-RX 매핑**: `detect_area_rx_mapping()` 메서드가 P2의 출력 파일명을 정규식으로 분석하여, 어떤 Area에 어떤 RX들이 포함되어 있는지를 파악하고 `area_rx_mapping` 딕셔너리를 생성합니다. 이는 다중 사용자 CCM을 구성하는 데 필수적인 정보입니다.
-   **통계 및 검증 설정**: CCM 계산에 필요한 `sampling_ratio` 뿐만 아니라, `ccm_correction`(CCM 보정 알고리즘 파라미터), `validation_config`(pMSE 기반 결과 검증 설정) 등 후처리 및 분석에 관련된 모든 설정을 중앙에서 관리합니다.
-   **실행 모드 제어**: `execution_mode` 플래그 ('save' 또는 'validation')를 통해 CCM을 실제로 저장할지, 아니면 랜덤 샘플링의 일관성을 검증하는 분석만 수행할지를 결정할 수 있어, 코드 변경 없이 다양한 목적의 실행이 가능합니다.

### 3.4. 핵심 메서드

-   **`CCM_BlockEngine.process_area_ccm()`**: 특정 Area에 속한 모든 RX에 대해 CCM 계산을 총괄하는 메서드입니다. 내부적으로 `_calc_R_ii_block`과 `_calc_R_ij_block`을 호출하여 모든 블록을 계산합니다.
-   **`CCM_BlockEngine._calc_R_ii_block()`**: 단일 사용자의 시변 채널 데이터를 입력받아, 시간/주파수 평균을 통해 **자기 공분산(R_ii)** 행렬과 채널 평균, 경로 손실, K-factor 등의 통계치를 계산합니다.
-   **`CCM_BlockEngine._apply_ccm_correction()`**: 물리적으로 유효한 공분산 행렬이 만족해야 하는 3가지 조건(Positive Semi-Definite, Unit Diagonal, Hermitian)을 보장하는 핵심 보정 알고리즘입니다. 반복 과정에서 보정 강도를 3단계로 점진적으로 제어하여 원본 행렬의 왜곡을 최소화(NMSE -30dB 이하)하며 안정적인 행렬로 수렴시킵니다.
-   **`CCM_BlockManager.build_area_ccm()`**: `CCM_BlockEngine`이 계산한 개별 `R_ii`, `R_ij` 블록들을 마치 레고처럼 조립하여 해당 Area의 모든 사용자를 포함하는 거대한 **전체 CCM**을 최종적으로 구성합니다.

### 3.5. 주요 출력

-   **채널 통계 데이터 (`.npz`)**:
    -   `ccm`: 최종 채널 공분산 행렬
    -   `stats_rx{id}_ch_mean`: 사용자별 채널 평균
    -   `stats_rx{id}_pathloss`: 사용자별 경로 손실
    -   `stats_rx{id}_k_factor`: 사용자별 K-Factor
    -   저장 위치: `ChMeanCov_Results/`
