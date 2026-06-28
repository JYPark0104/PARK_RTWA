# Requirements Document

## Introduction

본 기능은 단일 기지국(RAN Twin) 환경에서 제안된 하이브리드 메트릭(Covariance 기반 공간 메트릭 + 1D PDP 기반 시간 메트릭)이 실제 물리적 지형(Topology)을 얼마나 잘 반영하는지 평가하고, 두 메트릭 사이의 최적 융합 가중치(Golden Ratio: alpha, beta)를 GPU 병렬 연산으로 자동 탐색하여 시각화하는 단일 파이프라인 파이썬 스크립트(P5A_Hybrid_Metric_Auto_Weight_Search_2606v1.py)를 작성하는 작업이다.

이 파이프라인은 두 가지 입력 모드를 지원한다.
- 합성 데이터 모드(synthetic): 거리 기반 층화 샘플링으로 1024개의 가상 UE를 생성하여 Covariance(R)와 1D PDP(p_tau)를 합성한다.
- 실측 데이터 모드(real): 워크스페이스의 `25*` 폴더(예: `251004_Ch_Separ_(CorrRician)/P1B_Valid_Results/`, `251009_CCM_Collection (8 GB)/P1F_Marginal_CCM_Results/`)에 저장된 npz 파일에서 covariance matrix(R_BS 또는 R_UE)와 ray 데이터(tau, power)로부터 1D PDP를 추출하여 사용한다.

본 분석의 목표는 두 가지를 하나의 그림에서 동시에 증명하는 데 있다. (1) 물리적 거리와 하이브리드 메트릭 간의 선형 상관관계(Correlation), (2) t-SNE 를 통한 지형 위상(Topology) 복원. 모든 핵심 연산(Pairwise distance, Bures-Wasserstein, 1D Wasserstein, Spearman 상관계수)은 Python for-loop 없이 PyTorch GPU 텐서 브로드캐스팅으로 벡터화 처리한다. 결과물로 2행 4열(2x4) Two-Tier Visualization 서브플롯 (Row 1: Correlation Scatter — Physical vs Metric 산점도, Row 2: t-SNE Topology Restoration — 위상 복원 산점도; 각 행마다 Ground Truth / alpha=1.0 / alpha=0.0 / Golden Ratio 4개 컬럼)을 논문 삽입 품질의 PNG로 저장하고, suptitle 에 자동 탐색된 최적 alpha:beta 비율과 최대 Spearman 상관계수를 출력한다. 합성 모드(synthetic)와 실측 모드(real) 모두 동일한 2x4 시각화 그리드를 사용한다.

## Glossary

- **Pipeline_Script**: 본 기능을 구현한 단일 파이썬 실행 파일. 워크스페이스 네이밍 규칙(`P{단계}{순서}_{설명}_{날짜}v{버전}.py`)에 따라 `P5A_Hybrid_Metric_Auto_Weight_Search_2606v1.py`로 명명한다.
- **Output_Folder**: `/home/dclcom61/twin_minji/260604 metric 도출/` 디렉터리. 본 파이프라인 스크립트, 결과 폴더(`P5A_Hybrid_Metric_Results/`), `.md` 문서, `.log` 파일이 모두 이 폴더 또는 하위 폴더에 저장된다.
- **NPZ_Source_Folders**: 워크스페이스 내부의 실측 데이터를 담은 폴더 집합으로, 정규식 `^25\d{4}` 로 시작하는 폴더(예: `251004_Ch_Separ_(CorrRician)`, `251009_CCM_Collection (8 GB)`, `251009_Ch_Separ_Test`)를 의미한다.
- **UE**: User Equipment. 본 기능에서는 기지국(BS) 좌표 (0, 0)을 기준으로 한 단일 단말 샘플을 의미한다.
- **N_UE**: 처리 대상 UE 개수. 기본값은 1024이며 `--num_ue` 인자로 변경 가능하다.
- **Coverage_Sector**: 기지국 (0, 0)을 중심으로 반경 10m 이상 500m 이하, 중심 방위각 0° 기준 ±60° (즉 120°) 섹터로 정의된 2D 영역.
- **Distance_Stratified_Sampling**: 거리 축을 N_bins (5 이상 10 이하의 정수, 기본 10)개의 균등 구간으로 나누고, 각 구간에 동일 개수의 UE를 배치하는 샘플링 방식. N_UE 가 N_bins 의 배수가 아닐 경우의 처리는 R3 의 옵션 A(잔여 분배) 또는 옵션 B(N_UE 자동 보정) 중 하나를 따른다.
- **R**: 공간 도메인 covariance matrix. 합성 모드에서는 4x4 Hermitian PSD 행렬, 실측 모드에서는 npz의 `R_BS`(8x8 또는 16x16 등) 또는 `R_UE`를 사용한다. 차원은 모드에 따라 일관되게 N_UE개의 동일 크기 행렬로 구성된다.
- **p_tau**: 1D Power Delay Profile. 합성 모드에서는 32 bin, 실측 모드에서는 ray의 `tau`/`power`를 32 bin 히스토그램으로 변환한 1D 벡터. 각 벡터의 합은 1로 정규화한다.
- **N_tau**: PDP 빈 개수. 기본값은 32.
- **Physical_Distance_Matrix**: UE의 (x, y) 좌표 간 유클리드 거리 행렬. 크기 N_UE x N_UE.
- **W_BW_sq_Matrix**: 모든 UE 쌍에 대한 Bures-Wasserstein 거리 제곱 행렬. 크기 N_UE x N_UE.
- **W1_Matrix**: 모든 UE 쌍에 대한 1D Wasserstein 거리 행렬. CDF 기반 L1 노름으로 계산하며 크기 N_UE x N_UE.
- **Scale_Factor_Matrix**: 모든 UE 쌍 (i, j)에 대해 `sqrt(Tr(R_i) * Tr(R_j))` 값을 가진 N_UE x N_UE 행렬.
- **Time_Metric_Matrix**: `Scale_Factor_Matrix * W1_Matrix` (원소별 곱). 크기 N_UE x N_UE.
- **W_BW_norm**: `W_BW_sq_Matrix`를 Min-Max 정규화한 행렬. 모든 원소는 [0, 1] 범위.
- **W1_norm**: `Time_Metric_Matrix`를 Min-Max 정규화한 행렬. 모든 원소는 [0, 1] 범위.
- **Alpha**: 공간 메트릭(W_BW_norm) 가중치. 0.0 이상 1.0 이하의 실수.
- **Beta**: 시간 메트릭(W1_norm) 가중치. `Beta = 1.0 - Alpha`.
- **N_alpha**: alpha 스캔 분할 개수. 기본값은 100 (즉 `torch.linspace(0.0, 1.0, 100)`).
- **Total_Metric_Tensor**: alpha 100개 조합에 대한 융합 메트릭 행렬 텐서. 크기 N_alpha x N_UE x N_UE.
- **Off_Diagonal_Mask**: N_UE x N_UE 불리언 마스크. `i != j` 인 원소를 True 로 가지며, 거리 행렬을 1D 벡터로 평탄화(flatten)할 때 자기 자신과의 거리(대각 성분)를 제외하기 위해 사용된다. 본 기능에서는 동등한 표현으로 상삼각(k=1) off-diagonal 인덱스를 사용해도 된다.
- **Physical_Distance_Vector**: `Physical_Distance_Matrix` 의 off-diagonal 원소들을 평탄화한 1D 텐서. 길이는 `N_UE * (N_UE - 1) / 2` (상삼각만 사용 시) 또는 `N_UE * (N_UE - 1)` (Off_Diagonal_Mask 전체 사용 시).
- **Metric_Distance_Vector**: 임의 메트릭 행렬(W_BW_norm, W1_norm, Total_Metric_Tensor[k] 등) 의 off-diagonal 원소들을 `Physical_Distance_Vector` 와 동일한 인덱싱 규칙으로 평탄화한 1D 텐서.
- **Spearman_Correlation**: `Physical_Distance_Vector` 와 `Metric_Distance_Vector` 사이의 Spearman 순위 상관계수. 본 기능에서는 PyTorch 기반 rank 변환과 Pearson 상관으로 GPU에서 계산하며, 검증 단계에서 `scipy.stats.spearmanr` 와 비교한다.
- **Golden_Ratio**: 최대 Spearman 상관계수를 달성하는 (alpha*, beta*) 쌍.
- **Max_Correlation**: 모든 alpha 후보 중 Spearman 상관계수의 최댓값.
- **Subplot_Grid**: 2행 4열 (2x4) Matplotlib Subplot 그리드. 행 차원은 두 가지 시각화 관점(Correlation, Topology Restoration), 열 차원은 네 가지 메트릭 시나리오(Ground Truth, alpha=1.0, alpha=0.0, Golden Ratio)를 의미한다.
- **Correlation_Scatter_Row**: Subplot_Grid 의 1행 (Row 1). 각 서브플롯은 X축 = `Physical_Distance_Vector` (단위 m), Y축 = 해당 시나리오의 `Metric_Distance_Vector` 의 산점도이며, 점들은 `alpha=0.1` 의 투명도로 그려져 밀도(scatter density)가 시각화된다. 각 서브플롯 우측 상단에는 해당 시나리오의 Spearman 상관계수가 텍스트 박스로 표시된다.
- **Topology_Restoration_Row**: Subplot_Grid 의 2행 (Row 2). 각 서브플롯은 해당 시나리오의 거리 행렬을 t-SNE 로 2D 임베딩한 위상 복원 산점도이며, 점 색상은 기지국 (0, 0)으로부터의 거리 Radius 에 `cmap='viridis'` 를 매핑한다. (2,1) 컬럼은 t-SNE 대신 실제 (x, y) 좌표를 사용하고, 기지국 (0, 0) 위치에 별(`*`) 마커를 추가로 표시한다.
- **Subplot_Figure**: 2행 4열 (2x4) Matplotlib Figure. Row 1 은 Correlation_Scatter_Row, Row 2 는 Topology_Restoration_Row 로 구성되며, 논문 삽입을 전제로 가독성과 출판 품질을 만족한다. (이전 1x4 단일 행 구성에서 2x4 Two-Tier 구성으로 전면 갱신됨.)
- **Run_Timestamp**: 스크립트 실행 시점의 `YYYYMMDD_HHMMSS` 형식 문자열. 모든 산출물 파일명 끝에 부착되어 덮어쓰기를 방지한다.
- **Result_Folder**: `<Output_Folder>/P5A_Hybrid_Metric_Results/` 디렉터리. 모든 PNG, CSV, NPZ, 로그 산출물의 저장 경로.

## Requirements

### Requirement 1: 파이프라인 스크립트 파일 및 문서 산출

**User Story:** 연구원으로서, 워크스페이스 네이밍 규칙에 맞는 하나의 실행 가능한 파이썬 스크립트와 그에 대응하는 설명 문서/로그 파일을 받고 싶다. 그래야 향후 재현과 추적이 가능하다.

#### Acceptance Criteria

1. THE Pipeline_Script SHALL `/home/dclcom61/twin_minji/260604 metric 도출/P5A_Hybrid_Metric_Auto_Weight_Search_2606v1.py` 경로에 단일 파일로 생성된다.
2. THE Pipeline_Script SHALL 파일 상단 주석에 Python 버전, 실행 서버 명(예: dclcom55, dclcom45), GPU 정보, 주요 라이브러리(torch, numpy, scipy, matplotlib, scikit-learn) 버전을 명시한다.
3. THE Pipeline_Script SHALL 동일 폴더에 `P5A_Hybrid_Metric_Auto_Weight_Search_2606v1.md` 설명 문서를 함께 생성하며, 해당 문서에는 목적, 입력/출력, 실행 예시, 의존성을 포함한다.
4. THE Pipeline_Script SHALL 동일 폴더에 `P5A_Hybrid_Metric_Auto_Weight_Search_2606v1.log` 로그 파일을 생성하며, 각 줄은 `YYYY-MM-DD - 파일명 - 진행 내용` 형식으로 기록된다.
5. WHEN 시뮬레이션 파라미터(N_UE, N_bins, N_tau, N_alpha, 반경, 섹터 각도 등) 중 어느 하나가 기본값과 다르게 설정되면, THE Pipeline_Script SHALL 해당 변수 정의 라인 인라인 주석에 변경 사유와 변경 일자를 함께 기록한다.

### Requirement 2: 실행 환경 및 데이터 모드 선택 인자

**User Story:** 연구원으로서, 합성 데이터와 실측 npz 데이터 중 어느 것을 사용할지 명령줄에서 선택하고 싶다. 그래야 동일한 분석 파이프라인을 두 데이터 소스에 모두 적용할 수 있다.

#### Acceptance Criteria

1. THE Pipeline_Script SHALL `--mode` 인자를 노출하며 허용 값은 `synthetic` 와 `real` 두 가지뿐이다.
2. WHEN `--mode synthetic` 으로 실행되면, THE Pipeline_Script SHALL 거리 기반 층화 샘플링으로 N_UE개의 합성 데이터(R, p_tau, x, y)를 생성한다.
3. WHEN `--mode real` 으로 실행되면, THE Pipeline_Script SHALL `--npz_root` 로 지정된 디렉터리(기본값 `/home/dclcom61/twin_minji`) 하위에서 NPZ_Source_Folders를 자동 탐색하여 실측 covariance와 ray 데이터를 읽는다.
4. THE Pipeline_Script SHALL `--num_ue` (기본 1024), `--num_bins` (기본 10), `--num_tau_bins` (기본 32), `--num_alpha` (기본 100), `--device`, `--seed`, `--output_dir`, `--bin_remainder_policy` (기본 `redistribute`) 인자를 노출하며 각각의 기본값을 코드에 명시한다.
5. IF `--device cuda` 가 지정되었으나 `torch.cuda.is_available()` 이 False 이면, THEN THE Pipeline_Script SHALL CPU 폴백 메시지를 로그에 기록하고 CPU에서 동일한 연산을 수행한다.
6. THE Pipeline_Script SHALL 실행 시작 시 `torch.manual_seed(seed)` 와 `numpy.random.seed(seed)` 를 호출하여 합성 모드 결과의 재현성을 보장한다.

### Requirement 3: 합성 데이터 생성 (거리 기반 층화 샘플링)

**User Story:** 연구원으로서, 근거리/원거리 단말의 채널 특성 차이가 명확하게 드러나도록 거리 분포가 균일한 합성 데이터를 받고 싶다. 그래야 메트릭의 distance-discriminative 성능을 공정하게 평가할 수 있다.

#### Acceptance Criteria

1. WHEN `--mode synthetic` 이 활성화되면, THE Pipeline_Script SHALL 기지국을 (0, 0)으로 두고 반경 10m 이상 500m 이하, 방위각 -60° 이상 +60° 이하 (총 120°) 섹터 내에 N_UE개의 UE를 배치한다.
2. THE Pipeline_Script SHALL 거리 축을 N_bins (5 이상 10 이하의 정수, 기본 10) 개의 균등 구간으로 나누고, 각 구간에 가능한 한 동일한 개수의 UE를 배치한다. (기본값 변경 사유: 사용자 지정으로 8 → 10 으로 상향, AGENTS.md 의 파라미터 변경 사유 인라인 주석 규칙을 코드에도 동일하게 적용한다.)
3. WHERE `N_UE` 가 `N_bins` 의 배수일 때, THE Pipeline_Script SHALL 각 구간에 정확히 `N_UE / N_bins` 개의 UE를 배치한다.
4. WHERE `N_UE` 가 `N_bins` 의 배수가 아닐 때, THE Pipeline_Script SHALL 다음 두 옵션 중 하나를 적용하고 그 처리 방식을 로그에 명시적으로 기록한다.
   - 옵션 A (권장, 기본 동작): 각 구간에 `floor(N_UE / N_bins)` 개를 균등 배치한 뒤 잔여 `N_UE mod N_bins` 개의 UE 를 가장 먼 거리 구간부터 한 개씩 추가 배치(가장 먼 구간 우선) 또는 모든 구간에 라운드 로빈으로 균등 분배한다.
   - 옵션 B: 실행 시작 시 `N_UE` 를 `N_bins` 의 가장 가까운 배수 (예: 1024 → 1020) 로 자동 보정하고, 보정 사실과 보정된 N_UE 값을 로그에 기록한다.
5. THE Pipeline_Script SHALL `--num_ue` 가 `--num_bins` 의 배수가 아닐 때 어느 옵션(A 또는 B)을 사용할지 결정하는 인자(`--bin_remainder_policy`, 허용값 `redistribute` (옵션 A) | `truncate` (옵션 B), 기본 `redistribute`)를 노출한다.
6. THE Pipeline_Script SHALL 각 UE에 대해 4x4 Hermitian PSD covariance R을 생성하며, R의 trace는 거리 d가 클수록 단조 감소하고, R의 effective rank는 거리 d가 클수록 단조 증가하는 경향을 갖도록 한다.
7. THE Pipeline_Script SHALL 각 UE의 R 생성 과정에서 AoA 위상 회전 항을 거리에 따라 반영하여, 동일 거리대 UE들끼리는 공간 도메인에서 유사하고 거리가 다른 UE들끼리는 차이가 발생하도록 한다.
8. THE Pipeline_Script SHALL 각 UE에 대해 N_tau (기본 32) 빈의 1D PDP `p_tau` 를 생성하며, 각 벡터의 원소 합은 1.0 ± 1e-6 범위에서 정규화된다.
9. WHEN UE의 거리가 가까울수록, THE Pipeline_Script SHALL p_tau 가 좁은(sharp) 첫 빈 집중 형태가 되도록 생성한다.
10. WHEN UE의 거리가 멀수록, THE Pipeline_Script SHALL p_tau 의 delay spread 꼬리가 길어지고 형상이 평탄화되도록 생성하며, 이때 비선형 노이즈를 주입한다.

### Requirement 4: 실측 NPZ 데이터 적재 및 PDP 추출

**User Story:** 연구원으로서, 워크스페이스의 기존 NPZ 파일에서 covariance matrix와 PDP를 자동으로 추출하여 분석에 사용하고 싶다. 그래야 합성과 실측을 같은 파이프라인으로 비교 검증할 수 있다.

#### Acceptance Criteria

1. WHEN `--mode real` 이 활성화되면, THE Pipeline_Script SHALL `--npz_root` 하위 폴더 중 정규식 `^25\d{4}` 와 일치하는 폴더만 후보로 선택한다.
2. THE Pipeline_Script SHALL `--cov_pattern` (기본값 `*Marginal_CCM.npz`) 글롭 패턴으로 covariance용 npz 파일을 수집하고, 수집된 각 파일에서 `R_BS` 또는 `R_UE` 키 중 사용자 지정(`--cov_key`, 기본 `R_BS`) 키의 행렬을 적재한다.
3. THE Pipeline_Script SHALL `--ray_pattern` (기본값 `*Rays_Valid_RXs.npz`) 글롭 패턴으로 ray 데이터 npz를 수집하고, 각 RX 인덱스의 `tau` 및 `power` 배열로부터 N_tau 빈의 정규화된 PDP를 계산한다.
4. THE Pipeline_Script SHALL covariance 파일과 ray 파일을 RX 인덱스(`rx_indices` 또는 파일명 내 `RX{n}` 부분) 기준으로 매칭하여 동일 UE에 대한 (R, p_tau) 쌍을 구성한다.
5. IF 매칭되는 UE 수가 N_UE 보다 적으면, THEN THE Pipeline_Script SHALL 가용한 UE 수에 맞추어 N_UE 를 자동으로 축소하고 그 사실을 로그에 기록한다.
6. IF 매칭되는 UE 수가 N_UE 보다 많으면, THEN THE Pipeline_Script SHALL 거리 기반 층화 샘플링으로 N_UE 개를 선택하며, 거리 정보가 npz 메타에 없을 경우 RX 좌표 파일(JSON 또는 npz의 `rx_xyz`/`rx_position` 키)에서 (x, y)를 읽는다.
7. WHEN 적재된 covariance 행렬이 Hermitian이 아닌 경우, THE Pipeline_Script SHALL `(R + R.conj().T) / 2` 형태의 Hermitian 대칭화를 수행한 뒤 사용한다.
8. IF `--mode real` 에서 covariance 또는 ray npz 중 하나라도 0개 파일이 매칭되면, THEN THE Pipeline_Script SHALL 사용 가능한 폴더와 패턴 후보를 로그에 출력하고 비정상 종료한다.
9. THE Pipeline_Script SHALL 적재된 covariance 행렬들을 동일 차원으로 정렬(차원이 다르면 가장 빈도 높은 차원만 선택)하고, 모든 UE의 p_tau 길이를 N_tau 로 통일한다.

### Requirement 5: GPU 벡터화된 Pairwise Distance 연산

**User Story:** 연구원으로서, 1024 x 1024 쌍방향 거리 연산이 Python for-loop 없이 GPU에서 한 번에 처리되기를 원한다. 그래야 alpha 스캔이 실시간 수준으로 동작한다.

#### Acceptance Criteria

1. THE Pipeline_Script SHALL Python 레벨의 `for` 루프나 list comprehension 없이 `Physical_Distance_Matrix`, `W_BW_sq_Matrix`, `W1_Matrix`, `Scale_Factor_Matrix` 를 모두 PyTorch 텐서 브로드캐스팅으로 계산한다 (단, alpha 스캔과 무관한 1회성 데이터 적재 단계는 제외한다).
2. THE Pipeline_Script SHALL `Physical_Distance_Matrix[i, j] = ||(x_i, y_i) - (x_j, y_j)||_2` 를 N_UE x N_UE 크기의 부동소수 텐서로 산출한다.
3. THE Pipeline_Script SHALL `W_BW_sq_Matrix[i, j]` 를 Bures-Wasserstein 거리 제곱 정의 `Tr(R_i) + Tr(R_j) - 2 * Tr( (R_i^{1/2} R_j R_i^{1/2})^{1/2} )` 로 계산한다.
4. THE Pipeline_Script SHALL Bures-Wasserstein 계산 중 `torch.linalg.eigh` 를 사용하며, 고유값에 대해 `torch.clamp(min=eps)` (eps 기본값 1e-10) 를 적용하여 NaN 또는 음수 sqrt를 방지한다.
5. THE Pipeline_Script SHALL `W1_Matrix[i, j]` 를 1D Wasserstein 거리 정의 `sum_k |CDF_i[k] - CDF_j[k]| * Δτ` 로 계산하며, CDF는 `torch.cumsum(p_tau, dim=-1)` 으로 구한다.
6. THE Pipeline_Script SHALL `Scale_Factor_Matrix[i, j] = sqrt(Tr(R_i) * Tr(R_j))` 를 N_UE x N_UE 텐서로 산출한다.
7. THE Pipeline_Script SHALL `Time_Metric_Matrix = Scale_Factor_Matrix * W1_Matrix` 를 원소별 곱으로 계산한다.
8. THE Pipeline_Script SHALL `W_BW_sq_Matrix` 와 `Time_Metric_Matrix` 각각에 대해, off-diagonal 원소를 기준으로 Min-Max 정규화하여 `W_BW_norm` 과 `W1_norm` (둘 다 [0, 1] 범위)을 산출한다.
9. IF `W_BW_sq_Matrix` 또는 `Time_Metric_Matrix` 의 max 와 min 이 같으면, THEN THE Pipeline_Script SHALL 0으로 채워진 행렬을 정규화 결과로 반환하고 경고를 로그에 기록한다.
10. WHEN 모든 거리 행렬이 산출되면, THE Pipeline_Script SHALL 행렬 대각 성분이 1e-6 이하의 절댓값을 갖는지 검증하고, 그렇지 않으면 경고 로그를 남긴다.

### Requirement 6: GPU 기반 Auto Weight Search

**User Story:** 연구원으로서, alpha 가중치 후보 100개를 동시에 평가하여 Spearman 상관이 가장 높은 (alpha*, beta*)를 자동으로 받고 싶다. 그래야 수동 튜닝 없이 Golden Ratio를 식별할 수 있다.

#### Acceptance Criteria

1. THE Pipeline_Script SHALL `torch.linspace(0.0, 1.0, N_alpha)` 로 alpha 후보를 생성하며 N_alpha 의 기본값은 100이다.
2. THE Pipeline_Script SHALL `Total_Metric_Tensor = alpha * W_BW_norm + (1 - alpha) * W1_norm` 을 N_alpha x N_UE x N_UE 크기의 3D 텐서로 한 번에 계산하며, alpha 차원에 대한 Python for-loop를 사용하지 않는다.
3. THE Pipeline_Script SHALL N_alpha 개 각각의 후보 행렬과 `Physical_Distance_Matrix` 사이의 Spearman 상관계수를 계산하며, 두 행렬을 1D 벡터(`Physical_Distance_Vector`, `Metric_Distance_Vector`) 로 평탄화(flatten)한 뒤 상관을 산출한다. 평탄화 방식은 다음 두 가지 중 하나를 채택할 수 있으며, 채택한 방식이 두 벡터에 동일하게 적용되는 한 결과는 동등하다.
   - 방식 A (권장): 상삼각(`torch.triu_indices(N_UE, N_UE, offset=1)`) off-diagonal 인덱싱.
   - 방식 B: 행렬 전체를 `flatten()` 한 뒤 `Off_Diagonal_Mask` 로 자기 자신과의 거리(대각 성분)를 제외.
   파이프라인 내부에서 두 방식 중 어떤 것을 사용했는지 로그에 명시한다.
4. THE Pipeline_Script SHALL Spearman 상관계수 계산을 GPU에서 수행 가능한 방식(rank 변환 후 Pearson 상관, 또는 `scipy.stats.spearmanr` 를 CPU로 옮겨 계산)으로 구현하며, 어느 방식을 사용하더라도 동일 입력에 대해 결과 차이가 1e-4 이하가 되도록 검증 단계를 둔다.
5. WHEN N_alpha 개의 상관계수가 산출되면, THE Pipeline_Script SHALL 최댓값과 그 인덱스로부터 `Golden_Ratio = (alpha*, 1 - alpha*)` 와 `Max_Correlation` 을 추출한다.
6. THE Pipeline_Script SHALL alpha=0.0 과 alpha=1.0 에서의 상관계수 값을 별도로 보존하여 후속 시각화에서 baseline 비교에 사용한다.
7. WHEN Spearman 상관계수 계산 중 분산이 0인 입력이 발생하면, THE Pipeline_Script SHALL 해당 alpha 후보의 상관계수를 NaN 으로 표시하고 Golden Ratio 후보에서 제외한다.

### Requirement 7: 시각화 (2x4 Two-Tier Visualization)

**User Story:** 연구원으로서, 거리 상관관계(Row 1: Correlation Scatter)와 t-SNE 위상 복원(Row 2: Topology Restoration)을 같은 4개 시나리오(Ground Truth, alpha=1.0, alpha=0.0, Golden Ratio)에 대해 동시에 비교하는 2x4 Two-Tier 그리드를 한 화면에서 보고 싶다. 그래야 가중치 융합이 거리 보존과 위상 복원 모두에 미치는 효용을 논문 삽입 수준의 품질로 평가할 수 있다.

#### Acceptance Criteria

1. THE Pipeline_Script SHALL 2행 4열(2x4) Matplotlib Subplot_Figure (Subplot_Grid) 를 생성하며, 각 컬럼은 순서대로 `Ground Truth`, `Spatial Only (alpha=1.0)`, `Temporal Only (alpha=0.0)`, `Golden Ratio (alpha={alpha*:.3f}, beta={beta*:.3f})` 시나리오에 대응한다.

##### Row 1 — Correlation_Scatter_Row (거리 상관관계 플롯)

2. THE Pipeline_Script SHALL Row 1 의 모든 서브플롯에서 X축으로 `Physical_Distance_Vector` (단위 m), Y축으로 해당 시나리오의 `Metric_Distance_Vector` 를 사용한 산점도를 그리며, 점은 `alpha=0.1` 투명도와 작은 마커 크기로 그려 scatter density 가 가시화되도록 한다.
3. THE Pipeline_Script SHALL Row 1 의 모든 서브플롯에 X축 라벨로 `"Physical Distance (m)"`, Y축 라벨로 `"Metric Distance (normalized)"` 를 표기하며, (1,1) Ground Truth 서브플롯은 Y축 라벨로도 `"Physical Distance (m)"` 를 사용하여 단위가 일치함을 명시한다.
4. THE Pipeline_Script SHALL Row 1 의 (1,1) 서브플롯에서 Ground Truth 시나리오로서 X축, Y축 모두 `Physical_Distance_Vector` 를 사용하여 완벽한 선형 대각선(y = x)이 나타나도록 그린다.
5. THE Pipeline_Script SHALL Row 1 의 (1,2) 서브플롯에서 Y축으로 `W_BW_norm` (alpha=1.0, 공간 100%) 의 `Metric_Distance_Vector` 를 사용한다.
6. THE Pipeline_Script SHALL Row 1 의 (1,3) 서브플롯에서 Y축으로 `W1_norm` (alpha=0.0, 시간 100%) 의 `Metric_Distance_Vector` 를 사용한다.
7. THE Pipeline_Script SHALL Row 1 의 (1,4) 서브플롯에서 Y축으로 Golden Ratio 시점의 `Total_Metric_Tensor[k*]` (k* = argmax Spearman) 의 `Metric_Distance_Vector` 를 사용하며, 가장 얇고 선명한 우상향 트렌드가 나타나야 한다는 시각적 기대를 캡션 또는 주석으로 명시한다.
8. THE Pipeline_Script SHALL Row 1 의 모든 서브플롯 우측 상단에 해당 시나리오의 Spearman 상관계수 ρ 를 `"ρ = {value:.4f}"` 형식의 텍스트 박스로 표시하며, 텍스트 박스에는 반투명 흰색 배경과 얇은 테두리를 적용해 산점도 위에서도 가독성을 보장한다.
9. THE Pipeline_Script SHALL Row 1 의 (1,4) 서브플롯에 Golden Ratio 데이터에 대한 회귀선 또는 LOWESS 트렌드 라인을 얇은 단색 선(예: 0.8pt, 단일 색상)으로 오버레이하여, 우상향 추세가 다른 시나리오 대비 더 선명함을 강조한다. (선택적으로 (1,2), (1,3) 서브플롯에도 동일 종류의 트렌드 라인을 비교용으로 추가할 수 있다.)

##### Row 2 — Topology_Restoration_Row (t-SNE 위상 복원 플롯)

10. THE Pipeline_Script SHALL Row 2 의 모든 서브플롯에서 점 색상을 기지국 (0, 0) 으로부터의 거리 Radius 값에 `cmap='viridis'` 매핑으로 통일하여 시나리오 간 직접 비교가 가능하도록 한다.
11. THE Pipeline_Script SHALL Row 2 의 (2,1) 서브플롯에서 UE 의 실제 (x, y) 좌표 산점도(부채꼴 형상)를 그리고, 기지국 위치 (0, 0) 에 검은색 별(`marker='*'`) 마커(크기 200 이상)를 추가로 표시하며, 축 라벨은 각각 `"x (m)"`, `"y (m)"` 으로 한다.
12. THE Pipeline_Script SHALL Row 2 의 (2,2), (2,3), (2,4) 서브플롯에서 각각 `W_BW_norm`, `W1_norm`, Golden Ratio `Total_Metric_Tensor[k*]` 거리 행렬을 사용해 t-SNE 2D 임베딩(`sklearn.manifold.TSNE`, `metric='precomputed'`, `init='random'`, `perplexity=30`)을 수행하여 위상 복원 산점도를 그린다. (`perplexity` 등 기본값을 변경할 경우 변경 사유와 변경 일자를 인라인 주석으로 함께 기록한다.)
13. THE Pipeline_Script SHALL Row 2 의 (2,2)–(2,4) 서브플롯의 축 라벨을 `"t-SNE Dim 1"`, `"t-SNE Dim 2"` 로 통일하고, t-SNE 임베딩 좌표에는 의미 있는 절대 단위가 없으므로 축 눈금은 그대로 두되 격자(`grid`) 는 옅게 표시한다.
14. THE Pipeline_Script SHALL Row 2 의 (2,4) Golden Ratio 위상 복원이 (2,1) Ground Truth 의 부채꼴 형상을 가장 잘 복원해야 한다는 시각적 기대를 캡션 또는 주석으로 명시한다.

##### 공통 — Suptitle / 레이아웃 / 컬러바 / 저장

15. THE Pipeline_Script SHALL Subplot_Figure 의 `suptitle` 에 `"Golden Ratio = alpha:beta = {alpha*:.3f} : {beta*:.3f}, Max Spearman = {Max_Correlation:.4f}"` 를 폰트 크기 18 pt 이상으로 출력한다.
16. THE Pipeline_Script SHALL `tight_layout(rect=[0, 0, 1, 0.94])` 또는 동등한 처리로 suptitle, 행/열 라벨, 축 라벨, 텍스트 박스, 트렌드 라인, 컬러바가 서로 겹치지 않도록 보장하며, 저장 직전 레이아웃을 한번 더 시각적으로 검증한다 (AGENTS.md 의 그래프 체킹 규칙 준수).
17. THE Pipeline_Script SHALL Row 2 의 (2,1)–(2,4) 서브플롯에 대해 행 단위로 공유되는 단일 colorbar (Figure 우측 또는 Row 2 우측에 배치) 를 추가하여 Radius (m) 의 의미를 명확히 한다. Row 1 의 산점도에는 기본적으로 colorbar 를 두지 않으며, 필요 시 점의 밀도(2D histogram density) 를 표시하는 별도 colorbar 를 행 단위로 추가할 수 있다.
18. THE Pipeline_Script SHALL 모든 라벨/제목/틱 폰트 크기를 가독성을 위해 12 pt 이상으로 설정한다.
19. THE Pipeline_Script SHALL 본 Figure 가 논문 삽입용임을 전제로, 저장 시 `dpi=300` 이상, `bbox_inches='tight'` 옵션을 사용하고, 출력 PNG의 픽셀 폭이 2400 px 이상이 되도록 figure 크기(`figsize`)를 설정한다.
20. THE Pipeline_Script SHALL 결과 Figure 를 `<Result_Folder>/P5A_HybridMetric_2x4_{mode}_{Run_Timestamp}.png` 파일로 저장한다.

### Requirement 8: 결과 저장 및 로그 출력

**User Story:** 연구원으로서, 실행할 때마다 산출물이 덮어쓰지 않고 타임스탬프와 함께 저장되며, alpha-Spearman 곡선과 핵심 행렬을 후속 분석에 재사용할 수 있어야 한다.

#### Acceptance Criteria

1. THE Pipeline_Script SHALL 실행 시작 시 `Result_Folder` 가 존재하지 않으면 자동으로 생성한다.
2. THE Pipeline_Script SHALL Spearman 상관계수 vs alpha 곡선을 별도의 Matplotlib Figure로 저장하고, 파일명은 `P5A_AlphaCurve_2x4_{mode}_{Run_Timestamp}.png` 로 한다.
3. THE Pipeline_Script SHALL alpha 후보 100개와 그에 대응하는 상관계수, 그리고 Golden Ratio, Max_Correlation 을 `P5A_AlphaScan_2x4_{mode}_{Run_Timestamp}.csv` 로 저장한다.
4. THE Pipeline_Script SHALL `Physical_Distance_Matrix`, `W_BW_norm`, `W1_norm`, Golden Ratio 정보가 포함된 `P5A_Matrices_2x4_{mode}_{Run_Timestamp}.npz` 파일을 저장한다.
5. WHEN 동일한 Run_Timestamp 의 결과 파일이 이미 존재하면, THE Pipeline_Script SHALL 새로운 타임스탬프(`HHMMSS_<sequence>`)를 부여하여 기존 파일 덮어쓰기를 방지한다.
6. THE Pipeline_Script SHALL 표준출력과 `.log` 파일에 다음 항목을 기록한다: 실행 명령어, 모드, 디바이스, 시드, N_UE, N_bins, N_tau, N_alpha, `bin_remainder_policy` 적용 결과(잔여 분배 위치 또는 보정된 N_UE), Spearman 평탄화 방식(상삼각 또는 Off_Diagonal_Mask), 데이터 적재 결과 요약(파일 수, 매칭 UE 수), 거리 행렬 통계(min/max/mean), Golden Ratio, Max_Correlation, 총 실행 시간(초).
7. WHEN 예외가 발생하면, THE Pipeline_Script SHALL 스택 트레이스를 `.log` 파일에 기록한 뒤 비정상 종료 코드(1)로 종료한다.

### Requirement 9: 정합성 및 검증

**User Story:** 연구원으로서, 파이프라인이 산출하는 행렬과 통계가 수치적으로 타당함을 자동으로 점검받고 싶다. 그래야 가중치 탐색 결과를 신뢰할 수 있다.

#### Acceptance Criteria

1. THE Pipeline_Script SHALL `W_BW_sq_Matrix` 와 `W1_Matrix` 의 대칭성 잔차 (`||M - M.T||_F / ||M||_F`) 를 계산하여 1e-5 미만임을 검증하고, 이를 로그에 기록한다.
2. THE Pipeline_Script SHALL `W_BW_norm` 과 `W1_norm` 의 모든 원소가 [0.0, 1.0] 범위 안에 있음을 검증하고, 위반 시 경고를 출력한다.
3. THE Pipeline_Script SHALL alpha=1.0 일 때 `Total_Metric` 이 `W_BW_norm` 과 동일함을, alpha=0.0 일 때 `W1_norm` 과 동일함을 단위 검증한다 (Frobenius 오차 1e-6 미만).
4. WHEN 합성 모드에서 거리 d 가 단조 증가하는 UE 시퀀스를 가정하면, THE Pipeline_Script SHALL `Tr(R)` 가 평균적으로 단조 감소함을 (Spearman 상관 -0.5 이하) 검증하여 합성 데이터 품질을 점검한다.
5. THE Pipeline_Script SHALL Spearman 상관계수 계산의 두 구현(GPU rank 기반과 CPU `scipy.stats.spearmanr`)을 임의 표본에 대해 비교하여 차이가 1e-4 이하임을 일회 점검 후 결과를 로그에 남긴다.
