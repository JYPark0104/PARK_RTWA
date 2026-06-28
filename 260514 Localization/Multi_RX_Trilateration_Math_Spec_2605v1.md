# Multi-RX Anchor 기반 Target Localization — 수학적 명세

> **작성일**: 2026-05-14
> **버전**: v1
> **목적**: 1 TX × N_RX 채널 자료에서 RX들을 anchor로 활용하여 Target(TX 또는 새 UE)의 위치를 추정하는 알고리즘의 수식 정의
> **선행 자료**: `P1U_MultiBS_Localization_Spec.md` (multi-BS 가정, 본 문서로 대체됨)

---

## 0. 시나리오 정의 및 표기법

### 0.1 두 시나리오

| 항목 | 시나리오 A: TX Self-Localization | 시나리오 B: UE Localization (leave-one-out) |
|------|--------------------------------|---------------------------------------------|
| **Target** | TX (BS) | Query UE |
| **Anchor** | K개의 RX (위치 알려짐, 측정값 알려짐) | K-1개의 RX (= 1037-1) |
| **Ground Truth** | BS 좌표 `[-51.561, -21.794, 19]` | Query UE의 grid 좌표 |
| **반복 횟수** | 1회 | 1037회 (각 RX를 한 번씩 query로) |
| **목적** | 알고리즘 디버깅 + 메트릭 비교 | 통계적 정확도 평가 |

### 0.2 표기법

| 기호 | 의미 |
|------|------|
| $\mathcal{A} = \{1, 2, ..., K\}$ | Anchor index 집합 (선택된 RX들) |
| $\mathbf{r}_i \in \mathbb{R}^2$ | i번째 anchor의 좌표 (P1R grid에서 알려짐) |
| $\mathbf{p} \in \mathbb{R}^2$ | Target 후보 위치 (40×40 grid 위 점) |
| $\mathbf{p}^* \in \mathbb{R}^2$ | Target의 실제 위치 (검증용) |
| $\boldsymbol{\tau}_i = (\tau_{i,1}, ..., \tau_{i,L_i})$ | i번째 anchor에서 측정된 ray delay |
| $\mathbf{P}_i = (P_{i,1}, ..., P_{i,L_i})$ | i번째 anchor에서 측정된 ray power |
| $\mathbf{R}_i^{TX} \in \mathbb{C}^{1024 \times 1024}$ | i번째 anchor에서 본 TX-side covariance |
| $\mathbf{R}_i^{RX} \in \mathbb{C}^{16 \times 16}$ | i번째 anchor의 RX-side covariance |
| $\hat{d}_i$ | i번째 anchor에서 추정한 Target까지 거리 |
| $\hat{\boldsymbol{\theta}}_i = (\theta_i^{az}, \theta_i^{el})$ | i번째 anchor에서 추정한 Target 방향 (옵션) |
| $w_i \in [0, 1]$ | i번째 anchor의 신뢰 가중치 |
| $c = 3 \times 10^8$ m/s | 광속 |

---

## 1. 거리 추정 (Anchor i → Target)

### 1.1 First-peak Delay

$$\hat{d}_i^{\text{first}} = c \cdot \min_{l : P_{i,l} > P_{th}} \tau_{i,l}$$

- $P_{th}$: noise floor 임계값 (예: peak power의 −10 dB)
- 장점: LOS 환경에서 가장 정확
- 단점: NLOS에서 first peak가 indirect path일 수 있음 → bias 발생

### 1.2 Power-Weighted Mean Delay

$$\hat{d}_i^{\text{wmean}} = c \cdot \bar{\tau}_i = c \cdot \frac{\sum_l P_{i,l} \cdot \tau_{i,l}}{\sum_l P_{i,l}}$$

- 장점: NLOS에서 robust
- 단점: 항상 LOS 거리보다 큼 (multipath 평균)
- 보정: $\hat{d}_i^{\text{LOS-corr}} = \hat{d}_i^{\text{wmean}} - \alpha \cdot \tau_{i}^{\text{rms}}$ (실험적 $\alpha$)

### 1.3 RMS Delay Spread

$$\tau_i^{\text{rms}} = \sqrt{\frac{\sum_l P_{i,l} (\tau_{i,l} - \bar{\tau}_i)^2}{\sum_l P_{i,l}}}$$

- 거리 추정 보조 (LOS dominance 지표)

### 1.4 거리 추정 신뢰도 (anchor 가중치 후보)

**Rician K-factor**:
$$K_i = \frac{P_i^{\text{LOS}}}{\sum_{l \in \text{NLOS}} P_{i,l}}$$

- $K_i \to \infty$: 강한 LOS → 신뢰도 높음
- $K_i \to 0$: NLOS 우세 → 신뢰도 낮음

**가중치 변환** (예시 3가지, 비교 평가 대상):
$$w_i^{(1)} = \frac{K_i}{1 + K_i}, \quad w_i^{(2)} = \frac{1}{1 + \tau_i^{\text{rms}} / \tau_0}, \quad w_i^{(3)} = \frac{P_i^{\text{total}}}{\max_j P_j^{\text{total}}}$$

---

## 2. 방향 추정 (Anchor i에서 본 Target 방향) — 옵션

### 2.1 RX-side Covariance에서 AoA 추출

$\mathbf{R}_i^{RX} \in \mathbb{C}^{16 \times 16}$의 고유분해:
$$\mathbf{R}_i^{RX} = \mathbf{V}_i \boldsymbol{\Lambda}_i \mathbf{V}_i^H, \quad \boldsymbol{\Lambda}_i = \text{diag}(\lambda_{i,1} \geq \lambda_{i,2} \geq ... \geq \lambda_{i,16})$$

**Conventional Beamforming** (AoA 후보 grid scan):
$$P_i^{CB}(\theta) = \mathbf{a}^H(\theta) \mathbf{R}_i^{RX} \mathbf{a}(\theta)$$

- $\mathbf{a}(\theta) \in \mathbb{C}^{16}$: steering vector at angle $\theta$
- $\hat{\theta}_i^{az} = \arg\max_\theta P_i^{CB}(\theta)$

**MUSIC** (subspace 기반, 더 정밀):
$$P_i^{MUSIC}(\theta) = \frac{1}{\mathbf{a}^H(\theta) \mathbf{V}_i^{\text{noise}} (\mathbf{V}_i^{\text{noise}})^H \mathbf{a}(\theta)}$$

- $\mathbf{V}_i^{\text{noise}}$: 작은 고유값에 대응하는 고유벡터 (noise subspace)

### 2.2 좌표계 변환 (LCS → GCS)

P1A에서 정의된 RX antenna orientation을 사용해 Local → Global 변환:
$$(\theta^{GCS}, \phi^{GCS}) = \text{LCS\_to\_GCS}(\theta^{LCS}, \phi^{LCS}, \boldsymbol{\Omega}_i^{\text{rx}})$$

→ P1N에 이미 구현된 `gcs_to_lcs()` 역함수 재사용

### 2.3 방향 일치도 (anchor i에서 본 후보 위치 p의 방향과 추정 AoA 비교)

$$\theta_i^{\text{geom}}(\mathbf{p}) = \text{atan2}(p_y - r_{i,y}, p_x - r_{i,x})$$

$$\Delta\theta_i(\mathbf{p}) = |\theta_i^{\text{geom}}(\mathbf{p}) - \hat{\theta}_i^{az}|_{\text{wrap}}$$

→ **Hybrid loss에 추가 가능** (§3.2)

---

## 3. Trilateration Loss

### 3.1 거리 기반 Loss

$$\boxed{\mathcal{L}_{\text{dist}}(\mathbf{p}) = \sum_{i \in \mathcal{A}} w_i \cdot \big(\|\mathbf{p} - \mathbf{r}_i\|_2 - \hat{d}_i\big)^2}$$

- 각 anchor마다 "후보 위치까지의 거리"와 "측정된 거리"의 잔차 제곱합
- 최소제곱 (Weighted Least Squares) 형태

**해석적 최적해 없음 (비선형)** → 다음 두 방법:
1. **Grid search**: 모든 후보 $\mathbf{p}$ 평가 (40×40 = 1600점) → 비용 적음
2. **반복 최적화**: Gauss-Newton, Levenberg-Marquardt (초기값 grid argmin)

### 3.2 거리 + 방향 Hybrid Loss

$$\boxed{\mathcal{L}_{\text{hybrid}}(\mathbf{p}) = \sum_{i \in \mathcal{A}} w_i \big[\underbrace{(\|\mathbf{p} - \mathbf{r}_i\| - \hat{d}_i)^2}_{\text{distance}} + \gamma \cdot \underbrace{(\Delta\theta_i(\mathbf{p}))^2}_{\text{angle}}\big]}$$

- $\gamma > 0$: 방향 항 가중치 (cross-validation으로 튜닝)
- 단일 anchor에서도 위치 좁힘 가능 (LoS + AoA → 1점)

### 3.3 Coordinate Frame 통일 주의사항

| 항목 | 통일 기준 |
|------|----------|
| **Origin** | P1R `AREA_GRID_CONFIGS`의 grid origin |
| **단위** | meters |
| **Z축** | 2D 위치 추정만 수행 → BS 높이 19m 무시. 단, $\hat{d}_i$는 3D 거리이므로 보정 필요 |
| **Z 보정** | $\hat{d}_i^{\text{2D}} = \sqrt{(\hat{d}_i)^2 - (z_{BS} - z_{RX})^2}$ |

---

## 4. 확률 히트맵 (Boltzmann Posterior)

### 4.1 Softmax 변환

$$P(\mathbf{p}_k) = \frac{\exp(-\beta \cdot \mathcal{L}(\mathbf{p}_k))}{\sum_{j} \exp(-\beta \cdot \mathcal{L}(\mathbf{p}_j))}$$

### 4.2 자동 $\beta$ 스케일링

$$\beta = \frac{\ln(\rho)}{\text{median}(\mathcal{L}) - \min(\mathcal{L})}$$

- $\rho$: posterior peak/median ratio target (예: $\rho = 10$)
- → 히트맵의 sharpness가 anchor 수에 무관하게 일정

### 4.3 Numerical Stability (log-sum-exp)

$$\log P(\mathbf{p}_k) = -\beta \mathcal{L}(\mathbf{p}_k) - \log \sum_j \exp(-\beta \mathcal{L}(\mathbf{p}_j))$$

`scipy.special.logsumexp` 사용

### 4.4 Point Estimate 두 가지 (둘 다 산출)

1. **Argmax**: $\hat{\mathbf{p}}_{\text{MAP}} = \arg\max_k P(\mathbf{p}_k)$
2. **Weighted Mean**: $\hat{\mathbf{p}}_{\text{MMSE}} = \sum_k \mathbf{p}_k \cdot P(\mathbf{p}_k)$

→ Multi-modal 분포에서는 두 결과가 크게 다를 수 있음 (둘 다 보고)

---

## 5. Anchor 선택 전략 (시나리오 A 핵심)

| 전략 | 설명 | 예상 효과 |
|------|------|----------|
| **Random K** | 1037 RX 중 K개 무작위 선택 | Baseline |
| **Top-K K-factor** | $K_i$ 상위 K개 (LOS 강한 anchor) | 거리 추정 정확도 ↑ |
| **GDOP-min K** | Geometric Dilution of Precision 최소화 | 위치 추정 분산 ↓ |
| **PDP-diverse K** | anchor 간 PDP 패턴이 다양하도록 (Wasserstein 거리 최대화) | Multipath 정보량 ↑ |

### 5.1 GDOP (Geometric Dilution of Precision)

각 anchor에서 Target까지의 단위벡터:
$$\mathbf{u}_i = \frac{\mathbf{p}^* - \mathbf{r}_i}{\|\mathbf{p}^* - \mathbf{r}_i\|}$$

설계 행렬: $\mathbf{H} = [\mathbf{u}_1; \mathbf{u}_2; ...; \mathbf{u}_K]^T \in \mathbb{R}^{K \times 2}$

$$\text{GDOP} = \sqrt{\text{tr}((\mathbf{H}^T \mathbf{H})^{-1})}$$

- **GDOP 작을수록 좋음** (anchor가 Target 주변에 잘 분포)
- **시나리오 B에서는 $\mathbf{p}^*$를 모르므로 후보 grid의 중앙으로 근사**

### 5.2 K 값별 평가

K = 3, 5, 10, 50, 100, 1037에 대해 정확도 곡선 그리기 → 교수님이 미팅에서 언급한 "3개"의 의미 검증

---

## 6. 평가 지표

### 6.1 Localization Error

$$e = \|\hat{\mathbf{p}} - \mathbf{p}^*\|_2 \quad \text{[meters]}$$

산출 통계:
- Mean, Median, 90th percentile
- CDF: $P(e < 1\text{m})$, $P(e < 5\text{m})$, $P(e < 10\text{m})$

### 6.2 Heatmap Entropy (불확실성 지표)

$$H(P) = -\sum_k P(\mathbf{p}_k) \log P(\mathbf{p}_k) \quad \text{[nats]}$$

- 낮을수록 확신 높음
- Uniform 분포: $H = \log(N_{\text{grid}}) = \log(1600) \approx 7.38$

### 6.3 Anchor 수에 따른 정확도 곡선

$\text{Mean error}(K)$ vs $K \in \{3, 5, 10, 50, 100, 1037\}$ → 한계 곡선

### 6.4 Posterior Calibration

"히트맵의 90% 영역이 실제로 90% 확률로 Target을 포함하는가" — Reliability diagram

---

## 7. Cov 메트릭의 보조 결합 (Bayesian Fusion)

### 7.1 결합 방식

$$\mathcal{L}_{\text{combined}}(\mathbf{p}) = \alpha \cdot \mathcal{L}_{\text{trilat}}(\mathbf{p}) + (1-\alpha) \cdot \mathcal{D}_{\text{cov}}(\mathbf{R}^{\text{query}}, \mathbf{R}^{\text{cell}}(\mathbf{p}))$$

- $\mathcal{D}_{\text{cov}}$: cell별 평균 R과 query R 간의 메트릭 (chordal, geodesic, Bures, KL_complex 중 선택)
- $\alpha \in [0, 1]$: PDP vs Cov 가중치 (cross-validation 튜닝)

### 7.2 시나리오별 활용

- **시나리오 A**: $\mathcal{D}_{\text{cov}}$는 anchor RX들의 R로부터 추정한 AoA가 후보 BS 위치와 일치하는지 평가 (§2 참조)
- **시나리오 B**: $\mathbf{R}^{\text{query}}$는 query UE의 R, $\mathbf{R}^{\text{cell}}(\mathbf{p})$는 cell DB의 평균 R → 표준 fingerprinting 형태

---

## 8. 두 시나리오 알고리즘 의사코드

### 8.1 시나리오 A: TX Self-Localization

```
INPUT:
  - {(r_i, tau_i, P_i, R_i^TX, R_i^RX)} for i = 1..N_RX (1037)
  - K (anchor 수)
  - selection_strategy ∈ {"random", "topK_kfactor", "gdop_min"}
  - distance_method ∈ {"first_peak", "wmean"}
  - use_angle ∈ {True, False}

ALGORITHM:
  1. anchor_idx = SELECT_ANCHORS(strategy, K)
  2. for i in anchor_idx:
        d_i = ESTIMATE_DISTANCE(tau_i, P_i, distance_method)
        d_i_2d = sqrt(d_i^2 - (z_BS - z_RX)^2)         # Z 보정
        if use_angle:
            theta_i = ESTIMATE_AOA(R_i^RX, method="MUSIC")
        K_i = COMPUTE_K_FACTOR(P_i, los_nlos_flag_i)
        w_i = K_i / (1 + K_i)
  3. for each candidate p in 40x40 grid:
        L(p) = sum_i w_i * (||p - r_i|| - d_i_2d)^2
        if use_angle:
            L(p) += gamma * sum_i w_i * (delta_theta_i(p))^2
  4. P(p) = softmax(-beta * L(p))                      # auto beta
  5. p_MAP = argmax_p P(p)
     p_MMSE = sum_p p * P(p)
  6. error = ||p_MAP - p_BS_true||                     # ground truth = [-51.561, -21.794]

OUTPUT:
  - heatmap [40x40]
  - p_MAP, p_MMSE, error_MAP, error_MMSE
  - heatmap_entropy
```

### 8.2 시나리오 B: UE Localization (leave-one-out)

```
INPUT:
  - 시나리오 A와 동일 자료 풀
  - K (anchor 수)
  - cov_metric ∈ {"chordal", "geodesic", "bures", "kl_complex"}

ALGORITHM:
  results = []
  for query_id in range(N_RX):                          # 1037번 반복
    1. anchor_idx = [i for i in range(N_RX) if i != query_id]
    2. anchor_idx = SELECT_K(anchor_idx, K, strategy)
    3. R_query = R_query_id^RX
    4. cell_DB = BUILD_CELL_DB(anchor_idx)              # 5m grid 평균 R
    5. for each candidate p in 40x40 grid:
        cell = GET_CELL(p)
        D_cov(p) = cov_metric(R_query, cell_DB[cell])
       (옵션) PDP trilateration도 결합
    6. P(p) = softmax(-beta * D_cov(p))
    7. p_MAP = argmax_p P(p)
    8. error = ||p_MAP - r_query_id||
    results.append(error)

OUTPUT:
  - mean(results), median(results), CDF
  - per-metric ranking
```

---

## 9. Numerical Stability 주의사항

1. **거리 추정의 단위**: `tau_i`가 초(s)인지 ns인지 확인. `c · tau`로 거리 환산 시 단위 일치 필수
2. **Z 보정 음수 방지**: $(\hat{d}_i)^2 < (z_{BS} - z_{RX})^2$이면 `max(0, ...)` 클리핑
3. **K-factor 분모 0 방지**: 모든 ray가 LOS인 경우 → $K_i \to \infty$ → 가중치 cap (예: $w_i \leq 0.99$)
4. **NLoS heavy anchor 제거**: $\tau_i^{\text{rms}} > \tau_{th}$인 anchor는 outlier로 제외 (옵션)
5. **`scipy.special.logsumexp` 사용**: softmax 직접 계산 시 overflow
6. **Grid 경계 효과**: argmax가 grid 경계에 위치하면 실제 위치는 그 너머일 수 있음 → flag 추가
7. **Anchor 좌표 일치**: P1R `uid_to_coord()`와 P1A `tx_position`이 같은 좌표계인지 검증 (이미 검증된 사실)

---

## 10. 본 명세 → 코드 파일 매핑

| 수식 영역 | 구현 파일 |
|----------|----------|
| §1 거리 추정 (first_peak, wmean, RMS spread) | `P1V_PDP_Similarity_2605v1.py` |
| §1.4 K-factor, 가중치 | `P1V_PDP_Similarity_2605v1.py` |
| §2 AoA 추정 (CB, MUSIC) | `P1U_TX_Localization_2605v1.py` 내부 헬퍼 |
| §2.2 좌표계 변환 | P1N `gcs_to_lcs()` import |
| §3 Trilateration Loss | `P1T_Metrics_Lib_2605v1.py` (`trilateration_loss`, `hybrid_loss`) |
| §4 Softmax heatmap | `P1T_Metrics_Lib_2605v1.py` (`softmax_heatmap`) |
| §5 Anchor 선택 | `P1U_TX_Localization_2605v1.py` 내부 |
| §6 평가 지표 | `P1U_TX_Localization_2605v1.py`, `P1W_UE_Localization_LOO_2605v1.py` |
| §7 Cov 결합 | `P1W_UE_Localization_LOO_2605v1.py` |
| §8.1 시나리오 A | `P1U_TX_Localization_2605v1.py` |
| §8.2 시나리오 B | `P1W_UE_Localization_LOO_2605v1.py` |

---

## 11. 미해결 사항 / 추후 결정

1. **AoA 활용 여부 (시나리오 A)**: §3.2 hybrid loss를 처음부터 쓸 것인가, 거리만으로 baseline 잡고 §3.2를 ablation으로 추가할 것인가?
2. **시나리오 B의 Cov 메트릭 선정**: P1S 결과 ranking 기반으로 chordal/geodesic 우선 시도, Bures-Wasserstein은 비교군
3. **K-factor 계산을 위한 los_nlos_flag**: P1B NPZ에 이미 있는지, 없으면 P1A의 LOS path 메타데이터 별도 로딩 필요 (확인 필요)
4. **3D vs 2D**: 현재 §3.3에서 BS 높이 19m를 분리했지만, full 3D trilateration도 옵션. 결정 필요
5. **시나리오 A의 단일 실행 vs Anchor 샘플링 반복**: K=10 random anchor 시 실행마다 결과 다름 → Monte Carlo로 N회 반복 후 통계 산출 권장 (예: 100회)

---

**END OF SPEC**
