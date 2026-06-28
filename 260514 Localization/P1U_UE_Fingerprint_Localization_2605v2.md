# P1U_UE_Fingerprint_Localization_2605v2.py — 설명서

## 개요

`Area1_7.5GHz_Rays_Valid_RXs.npz` (P1B 필터링 완료, NLoS only, 1,037 RX × max 400 paths) 와 P1A 의 Area 1 grid 정의를 결합하여, **dense 2D heatmap fingerprinting localization** 을 수행합니다.

v1 (sparse 1D, 42 위치, PDP only) 의 한계를 모두 해소한 본격 사용 버전입니다.

---

## 1. v1 → v2 변경점

| 항목 | v1 | **v2** |
|------|:---:|:---:|
| 입력 NPZ | channel_data_GHMTwin2_cutting.npz | **Area1_7.5GHz_Rays_Valid_RXs.npz** |
| RX 개수 (위치 환원 가능) | 42 (1D line) | **1,037 (2D grid)** |
| 사용 메트릭 | PDP 5 종 | **PDP 5 + AoA/AoD 3 = 8 종** |
| Heatmap 형태 | sparse 1D stripe | **dense 2D plane** |
| TX / fc 정보 | 가정 | **NPZ + P1A 정합** |
| LoS/NLoS 처리 | 미구분 | **NLoS only (P1B 결과)** |
| Path 추출 | 가변 길이 직접 | **counts 또는 tau != 0 mask** |

---

## 2. 입력 데이터

### 2-1. NPZ 의 layout (확인 완료)

| 키 | shape | 의미 |
|----|-------|------|
| `tau`, `power`, `theta_r_deg`, `phi_r_deg`, `theta_t_deg`, `phi_t_deg`, `los_nlos_flag`, `source_path_idx` | (1037, 1, 1, 1, 1, 400) | 6차원 (RX, B, N_BS, N_UE, ?, max_paths). zero-padded. |
| `counts` | (1037, 1, 1, 1, 1) | 각 RX 의 valid path 개수 (380~400) |
| `rx_indices` | (1037,) | 원본 1,600 grid 의 flat 인덱스 |
| `frequency_ghz` | scalar | 7.5 |
| `area_index` | scalar | 1 |
| `num_rx` | scalar | 1037 |

### 2-2. zero-padding 의 위치 (검증 완료)

* Sionna 의 path-solver 가 max_paths=400 으로 가변 길이 path 를 zero-pad.
* P1B 의 LoS-removal 도 LoS source 의 ray slot 들을 0 으로 만듦.
* **rx0 의 경우 앞 20 slot 이 0** (LoS 1 source × 20 ray), 뒤 380 slot 이 valid.
* `los_nlos_flag` 는 **0 = NLoS**, padding 도 0 → 구분 시 `tau != 0` 가 가장 안전.

### 2-3. rx_indices → (x, y, z) 환원 (P1A 의 generate_area_rx_grid 와 동일 순서)

```python
n_x = n_y = 40
rx_x = np.linspace(-136.138, 58.862, n_x)   # 5.0 m step
rx_y = np.linspace(-117.667, 77.333, n_y)   # 5.0 m step
ix = flat_idx % n_x
iy = (flat_idx // n_x) % n_y
pos = (rx_x[ix], rx_y[iy], 1.5)
```

* 원본 grid 1,600 (40×40×1) 중 P1B 가 1,037 만 valid 로 인정 → flat 인덱스가 sparse 한 형태.
* TX = (-51.561, -21.794, 19.0) m  (= AREA1_TX_XYZ)
* 주파수 = 7.5 GHz

---

## 3. Fingerprint 빌더

### 3-1. PDP (1D)

* power-weighted tau 히스토그램 → L1 정규화 → 128-dim 벡터.
* tau 단위 자동 감지 (sec / ns).

### 3-2. AoA spectrum (2D)

* power-weighted **(theta_r_deg, phi_r_deg)** 2D 히스토그램.
* bins: theta 18 (10° step) × phi 36 (10° step) = **648 bins** → flat L1 정규화.
* RX 의 각도적 multipath 환경을 fingerprint 화.

### 3-3. AoD spectrum (2D)

* AoA 와 동일한 빈 설정으로 (theta_t_deg, phi_t_deg).
* TX 측에서 본 multipath 분포 → 빌딩 구조에 따른 정합성 강함.

---

## 4. 8 가지 메트릭 (모두 "거리 = 작을수록 유사")

| # | 메트릭 | 설명 |
|---|--------|------|
| 1 | `pdp_pearson` | \(1 - \mathrm{corr}(p, q)\) |
| 2 | `pdp_cosine` | \(1 - \dfrac{p \cdot q}{\lVert p \rVert \lVert q \rVert}\) |
| 3 | `pdp_wasserstein` | scipy.stats.wasserstein_distance over tau axis (ns) |
| 4 | `pdp_jsd` | scipy.spatial.distance.jensenshannon (bits) |
| 5 | `tau_rms` | \(\lvert \tau_{rms}(p) - \tau_{rms}(q) \rvert\) (ns) |
| 6 | `aoa_cosine` | AoA spectrum cosine distance |
| 7 | `aod_cosine` | AoD spectrum cosine distance |
| 8 | `joint_aoa_aod_cosine` | concat(AoA, AoD) → cosine distance |

likelihood = softmax(\(-d / \tau_T\)),  \(\tau_T = \mathrm{median}(d)\) (자동, metric 별 스케일 보정).

---

## 5. 시각화

### 5-1. 메트릭 별 heatmap (총 8 장)

* 배경: Map_Mesh.obj top-view (vertex 30K sub-sample, z 를 grayscale).
* heatmap: 1,036 DB RX 위치의 likelihood 를 **`scipy.interpolate.griddata` cubic** 로 256×256 격자에 보간 → imshow.
* 마커: ★ (red) = Query/GT, X (lime) = argmax, ▲ (orange) = TX(BS).

### 5-2. comparison plot (1 장)

* 8 metric × 2 row (distance vs geo, likelihood vs geo).
* 가까울수록 distance ↓, likelihood ↑ 인지 시각 검증.

---

## 6. 사용법

### 기본 실행

```bash
cd "260514 Localization"
python P1U_UE_Fingerprint_Localization_2605v2.py
```

### 주요 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--npz <path>` | `../251020 BM QIE (Weichselberger)/P1B_Valid_Results/Area1_7.5GHz_Rays_Valid_RXs.npz` | 입력 NPZ |
| `--obj <path>` | `../260512 RT Result Data/Map_Mesh.obj` | 입력 OBJ |
| `--query <int>` | 500 | Query RX 인덱스 (0..1036) |
| `--tx-x / --tx-y / --tx-z` | (-51.561, -21.794, 19.0) | TX 좌표 |
| `--n-bins` | 128 | PDP 히스토그램 bin 수 |
| `--obj-sample` | 30000 | OBJ vertex sub-sample 수 |
| `--temperature` | None (median 자동) | softmax temperature |
| `--out-dir <path>` | `P1U_v2_Results/` | 결과 폴더 |

### 다른 query 비교

```bash
python P1U_UE_Fingerprint_Localization_2605v2.py --query 100
python P1U_UE_Fingerprint_Localization_2605v2.py --query 800
```

---

## 7. 산출물 (`P1U_v2_Results/`)

| 파일 | 내용 |
|------|------|
| `P1U_v2_heatmap_pdp_pearson_<TS>.png` | PDP Pearson heatmap |
| `P1U_v2_heatmap_pdp_cosine_<TS>.png` | PDP Cosine heatmap |
| `P1U_v2_heatmap_pdp_wasserstein_<TS>.png` | Wasserstein heatmap |
| `P1U_v2_heatmap_pdp_jsd_<TS>.png` | Jensen-Shannon heatmap |
| `P1U_v2_heatmap_tau_rms_<TS>.png` | RMS delay heatmap |
| `P1U_v2_heatmap_aoa_cosine_<TS>.png` | AoA cosine heatmap |
| `P1U_v2_heatmap_aod_cosine_<TS>.png` | AoD cosine heatmap |
| `P1U_v2_heatmap_joint_aoa_aod_cosine_<TS>.png` | 결합 heatmap |
| `P1U_v2_metric_comparison_<TS>.png` | 8 metric 비교 |
| `P1U_v2_summary_<TS>.csv` | DB 별 모든 metric 거리 / likelihood |
| `P1U_v2_localization_error_<TS>.csv` | metric 별 argmax 오차 |

`<TS>` = `YYYYMMDD_HHMMSS`.

---

## 8. 한계 및 다음 단계

### 8-1. 현재 한계

1. **단일 BS** → 진정한 trilateration 안 됨.
2. **NLoS only 데이터** — LoS 가 함께 있을 때 (full data) 와 정확도 다를 수 있음. 비교 실험 권장.
3. **AoA/AoD bin 해상도 10°** — 더 dense 한 빌딩에서는 5° 권장.
4. **z = 1.5 m fixed** — UE 높이 다양화 시 grid 재정의 필요.

### 8-2. 다음 단계

| 단계 | 작업 | 산출물 |
|------|------|--------|
| **P1V** | AoA/AoD-only fingerprint 의 거리-유사도 산점도 | `P1V_PDP_Similarity_2605v1.py` |
| **P1V2** | UE 이동 trajectory 시 Bayesian 갱신 (motion model + 측정 갱신) | `P1V2_Trajectory_Tracking_2605v1.py` |
| **P1A_MultiBS** | Map_Mesh.obj scene + multi-TX 위치 RT | `P1A_RT_to_Rays_MultiBS_2605v1.py` |
| **P1W** | multi-BS likelihood 곱 → 진정한 trilateration heatmap | `P1W_MultiBS_Trilat_2605v1.py` |

---

## 9. 트러블슈팅

| 증상 | 원인 / 해결 |
|------|-------------|
| `IndexError: query 1037 out of range` | RX 1,037 개. `--query` 0..1036. |
| 모든 metric 의 heatmap 이 같은 위치를 argmax | NLoS only 데이터의 한계. 다른 query 로도 검증. |
| AoA/AoD heatmap 이 PDP 보다 더 "넓게" 퍼짐 | Angular 분포는 거리에 약하고 환경 구조에 민감 → 정상. |
| 메트릭 계산이 너무 오래 걸림 | 1037 × 1036 = ~1.07M pair. 30초 미만 정상. |

---

## 10. 변경 기록

`P1U_UE_Fingerprint_Localization_2605v2.log` 참조.
