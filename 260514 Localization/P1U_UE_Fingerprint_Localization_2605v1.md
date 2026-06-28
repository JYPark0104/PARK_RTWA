# P1U_UE_Fingerprint_Localization_2605v1.py — 설명서

## 개요

`channel_data_GHMTwin2_cutting.npz` (1 TX × 2,492 RX, SISO) 의 사전 계산된 채널 핑거프린트를 사용해, **하나의 query RX 위치를 다른 RX 들의 PDP / 스칼라 지표 유사도로 추정** (fingerprinting localization) 하고, 결과 likelihood 를 **`Map_Mesh.obj` 의 top-view 위에 heatmap 으로 overlay** 하는 스크립트입니다.

---

## 1. 입력 데이터

### 1-1. NPZ (`channel_data_GHMTwin2_cutting.npz`, 7.6 MB)

| 키 | shape | dtype | 설명 |
|----|-------|-------|------|
| `rx_positions` | (42, 2) | float64 | **위치가 알려진 42 개 RX 의 (x, y)**. y=81.756 m 의 1차원 line. |
| `rsrp_all` | (2492,) | float64 | 전체 2,492 RX 의 RSRP (dBm). 168 개는 dead (-inf). |
| `target_rx_index` | (1,) | int64 | NPZ 가 가리키는 query RX 인덱스 (=0). |
| `tau_rxN` | (P_n,) | float32 | RX n 의 path 별 지연 시간 (sec 또는 ns; 자동 감지). |
| `power_rxN` | (P_n,) | float32 | RX n 의 path 별 power (linear). |
| `aoa_rxN` | (P_n,) | float32 | RX n 의 path 별 AoA (deg). 본 스크립트에서는 미사용. |
| `R_TX_rxN`, `R_RX_rxN` | (1, 1) | complex128 | SISO 이므로 스칼라. SPD-manifold metric 미사용 사유. |

> ⚠️ `rx_positions` (42 개) 와 채널 데이터 (2,492 개) 의 **일관성 미스매치**. 본 스크립트는 위치가 있는 42 개 RX 만 사용합니다. 원본 (cutting 안 됨) 위치가 확보되면 동일 코드로 재실행하면 됩니다.

### 1-2. OBJ (`Map_Mesh.obj`, 109 MB)

| 항목 | 값 |
|------|-----|
| 출력 도구 | Open3D |
| Vertex / Triangle | 1,078,902 / 361,561 |
| Material | 없음 (mtllib / usemtl 없음) |
| Per-vertex color | 있음 (RGB 값) |
| bbox X / Y / Z | [-127.2, 118.2] / [-125.9, 118.4] / [-0.4, 61.6] m |
| 단위 | meter |

본 스크립트는 OBJ 의 **vertex 만** 읽어 (face 무시) top-view scatter 로 활용. 30,000 개 sub-sampling 으로 plot 시간 1 초 미만.

---

## 2. 알고리즘 — 5 가지 채널 유사도 메트릭

### 2-1. 1D PDP 빌드 (공통)

* tau, power 의 단위 자동 감지 (q99 < 1e-3 → sec, 그 외 → ns).
* 전역 [tau_min, tau_max] 결정 후 모든 RX 가 동일 bin edges 사용.
* `power` 가중 히스토그램 → L1 정규화 → \(p \in \mathbb{R}^{n\_bins}\).
* path 0 개 (dead) RX 는 균등 분포로 처리.

### 2-2. 메트릭 5 종 (모두 "거리 = 작을수록 유사")

| # | 메트릭 | 정의 |
|---|--------|------|
| 1 | **Pearson** | \(d = 1 - \mathrm{corr}(p, q)\) |
| 2 | **Cosine** | \(d = 1 - \dfrac{p \cdot q}{\lVert p \rVert \, \lVert q \rVert}\) |
| 3 | **Wasserstein-1** | \(W_1(p, q) = \int \lvert F_p(\tau) - F_q(\tau) \rvert\, d\tau\)  (scipy.stats.wasserstein_distance) |
| 4 | **Jensen-Shannon** | \(\mathrm{JSD}(p, q) = \tfrac{1}{2}(D_{KL}(p \Vert m) + D_{KL}(q \Vert m)),\ m = (p+q)/2\) (bits) |
| 5 | **RMS delay diff** | \(\lvert \tau_{rms}(p) - \tau_{rms}(q) \rvert\) (ns) |

> Chordal / AIRM / Bures-Wasserstein 같은 SPD-manifold metric 은 \(R_{TX}, R_{RX}\) 가 1×1 (SISO) 이라 자명한 0 이므로 제외.

### 2-3. likelihood (softmax)

\[
L_i = \frac{\exp(-d_i / \tau)}{\sum_j \exp(-d_j / \tau)},\quad
\tau = \mathrm{median}(d) \ \text{또는 사용자 지정}
\]

* 메트릭 별 스케일 차이 (예: pearson∈[0,2] vs Wasserstein∈[0,1000ns]) 를 자동 흡수하기 위해 **temperature = median 거리** 를 기본값으로 사용. `--temperature` 로 수동 지정 가능.

### 2-4. 추정 / 평가

* 추정 위치: \(\hat{i} = \arg\max_i L_i\) → 해당 RX 의 (x, y).
* localization error: \(\lVert \mathbf{x}_{\hat{i}} - \mathbf{x}_{\text{GT}} \rVert_2\) (m).

---

## 3. 시각화

### 3-1. 메트릭 별 heatmap (5 장)

* 배경: Map_Mesh.obj top-view scatter (z 를 grayscale 로) — 빌딩 형상.
* Heatmap: 41 개 DB RX 위치의 likelihood 를 **2D 보간** (`scipy.interpolate.griddata`) 후 imshow.
   * y=81.756 fixed line 데이터라 보간 결과는 가로 stripe 처럼 보임 → `--no-2d-interp` 로 끄면 점만 표시.
* 마커: ★(red) = Query/GT, X(lime) = argmax 추정, ▲(orange) = TX(BS).

### 3-2. 메트릭 비교 plot (1 장)

* 5 metric × 2 row:
   * Top: 메트릭 거리 vs query 와의 geodesic 거리 (산점도) — 교수님 지적사항 직접 검증.
   * Bottom: 메트릭 별 likelihood 막대.

---

## 4. 사용법

### 기본 실행 (모든 인자 default)

```bash
cd "260514 Localization"
python P1U_UE_Fingerprint_Localization_2605v1.py
```

### 주요 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--npz <path>` | `../260512 RT Result Data/channel_data_GHMTwin2_cutting.npz` | 입력 NPZ |
| `--obj <path>` | `../260512 RT Result Data/Map_Mesh.obj` | 입력 OBJ |
| `--query <int>` | 0 | Query RX 인덱스 (0..N_pos-1) |
| `--tx-x / --tx-y / --tx-z` | (-51.561, -21.794, 19.0) | TX 좌표 (m) |
| `--fc-ghz` | 7.5 | 캐리어 주파수 |
| `--n-bins` | 128 | PDP 히스토그램 bin 수 |
| `--obj-sample` | 30000 | OBJ vertex sub-sampling 수 |
| `--no-2d-interp` | False | 2D 보간 끄고 점만 표시 |
| `--temperature` | None (median 자동) | softmax temperature |
| `--out-dir <path>` | `P1U_UE_Fingerprint_Results/` | 결과 저장 폴더 |

### 다른 query 로 재실행 예시

```bash
python P1U_UE_Fingerprint_Localization_2605v1.py --query 21
```

---

## 5. 결과물 (`P1U_UE_Fingerprint_Results/`)

| 파일 | 내용 |
|------|------|
| `P1U_heatmap_pearson_<TS>.png`     | Pearson 메트릭 heatmap |
| `P1U_heatmap_cosine_<TS>.png`      | Cosine 메트릭 heatmap |
| `P1U_heatmap_wasserstein_<TS>.png` | Wasserstein 메트릭 heatmap |
| `P1U_heatmap_jsd_<TS>.png`         | Jensen-Shannon 메트릭 heatmap |
| `P1U_heatmap_tau_rms_<TS>.png`     | RMS delay diff 메트릭 heatmap |
| `P1U_metric_comparison_<TS>.png`   | 5 메트릭 거리/likelihood 비교 |
| `P1U_summary_<TS>.csv`             | DB 별 모든 메트릭 거리, likelihood |
| `P1U_localization_error_<TS>.csv`  | 메트릭 별 argmax 추정 위치 / GT 거리 |

`<TS>` = `YYYYMMDD_HHMMSS` (덮어쓰기 방지).

---

## 6. 한계 및 다음 단계

### 6-1. 현재 한계

1. **위치가 1D line 위 42 점** → 2D heatmap 의 dynamic range 가 제한적. 일반 fingerprinting 의 dense 격자 효과 안 보임.
2. **SISO** → 안테나 array 기반 metric (covariance / SPD manifold / AoA spectrum) 미적용.
3. **단일 BS** → 진정한 trilateration / GDOP 분석 불가.
4. **RT 메타 누락** (실제 TX 위치, fc, 안테나 형상) → P1A 기본값 가정.

### 6-2. 다음 단계 제안

| 단계 | 작업 | 산출물 |
|------|------|--------|
| **P1V** | PDP / PADP 거리-유사도 산점도 분석 (교수님 지적사항 정량 검증) | `P1V_PDP_Similarity_2605v1.py` |
| **P1V2** | UE 이동 trajectory 가정 시 Bayesian 시간 재귀로 추정 갱신 | `P1V2_Trajectory_Tracking_2605v1.py` |
| **P1A_MultiBS** | Map_Mesh.obj scene + multi-TX 위치로 RT 재실행 | `P1A_RT_to_Rays_MultiBS_2605v1.py` + `Map_Mesh_scene.xml` |
| **P1W** | multi-BS 결합 likelihood 로 진정한 trilateration heatmap | `P1W_MultiBS_Trilat_2605v1.py` |

---

## 7. 의존성

| 패키지 | 최소 버전 | 용도 |
|--------|-----------|------|
| numpy | 1.24 | 배열 연산 / NPZ 로드 |
| scipy | 1.10 | wasserstein_distance, jensenshannon, griddata |
| matplotlib | 3.7 | 시각화 |

scipy 가 없으면 Wasserstein 은 CDF 차이로 fallback (정확도 비슷), JSD 는 직접 구현 (수치 OK), 다만 2D 보간은 동작 안 함 (`--no-2d-interp` 필수).

---

## 8. 트러블슈팅

| 증상 | 원인 / 해결 |
|------|-------------|
| `IndexError: rx_positions[42]` | NPZ 의 rx_positions 가 42 개 뿐. `--query` 를 0..41 범위 안으로. |
| heatmap 이 가로 stripe 처럼 보임 | 데이터가 1D line 이라 정상. `--no-2d-interp` 로 점만 표시. |
| OBJ 읽는데 너무 오래 걸림 | `--obj-sample 5000` 으로 줄임 (시각화 품질만 떨어짐). |
| `tau_rms diff` 만 모든 RX 에서 0 | path 가 1 개뿐인 RX 에서는 RMS 가 0 → 수치 OK, 단 정보량 부족. |

---

## 9. 변경 기록

`P1U_UE_Fingerprint_Localization_2605v1.log` 참조.
