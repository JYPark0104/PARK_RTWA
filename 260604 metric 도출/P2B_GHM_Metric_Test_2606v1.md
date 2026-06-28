# P2B_GHM_Metric_Test_2606v1.py

## 목적
신규 RT 데이터(`260604 GHM_test` / `channel_data_260531_GHM_Twin_v0_1.npz`)로
기존 P2A 실험보다 **더 나은 거리-메트릭 선형 scatter** 를 얻을 수 있는지 검증.

## 입력 데이터 특성 (탐색으로 확인)
- `rx_positions (5071,2)`, `tx_positions (3,3)` 포함 → **TX 3개(다중 기지국)**.
- per-(TX,RX): `tau[ns]`, `power[dB(m)]`, `aoa[deg, -180~180]`. 경로 수 가변(최대 1070, 중앙값 33).
- `R_TX/R_RX = (1,1)` → **SISO**. 공간 공분산(BW) 불가 → 공간 정보는 **AoA(APS)** 로 구성.
- `power` 는 **dB 스케일**(-120~-107 dBm). 반드시 선형(`10**(p/10)`) 변환 후 누적해야 함
  (clip(0) 으로 처리하면 전부 0 이 되어 메트릭 붕괴).

## 비교 메트릭 (v1.1: 다변측위 L2 결합으로 개편)
| 기호 | 정의 | 의미 |
|---|---|---|
| PDP-Wass (참고) | 지연 PDP(32 bin) 1D Wasserstein(CDF-L1) | 거리(range) |
| AoA-Wass (참고) | 방위 APS(48 bin) 1D Wasserstein(CDF-L1) | 방위(bearing) |
| **Multi-BS AoA (L2)** | 3개 TX AoA-Wass 거리의 **L2 norm** `√(D0²+D1²+D2²)` | 직교 위상정보 결합 |
| **Multi-BS RSRP (L2)** | 기지국별 RSRP `(n,3)` 벡터 간 **유클리디안 거리** | 다변측위 거리 |
| **Multi-BS HYBRID** | `α·AoA_L2 + (1-α)·RSRP_L2`, α 자동탐색 | 궁극 융합 |

> 수정 사유: 기존 `w0·D0 + w1·D1 + w2·D2`(스칼라 L1 합)는 직교하는 다기지국
> 위상정보를 1차원으로 찌그러뜨림 → **L2(유클리디안) 벡터 결합**으로 교체.
> RSRP 도 단일 BS 절대차(상관 음수) 대신 **(n,3) 벡터의 L2 쌍거리**로 산출.

## 결과 (거리 층화샘플 n=900, dist 5~1113 m)
| 메트릭 | Spearman ρ |
|---|---|
| PDP-Wass (단일, 참고) | +0.217 |
| AoA-Wass TX0 (단일, 참고) | +0.467 |
| **Multi-BS AoA (L2)** | **+0.495** |
| Multi-BS RSRP (L2) | +0.198 |
| **Multi-BS HYBRID** | **+0.505** (α_AoA=0.85) |

## 해석 / 결론
- **L2 결합이 기하학적으로 타당**: 직교 위상정보를 보존 → AoA-L2 0.495.
- HYBRID(AoA-L2 + RSRP-L2) 0.505 로 단일 BS(0.467) 대비 향상.
- 한계 요인은 다중경로 환경의 **range(거리) 관측성 부족**:
  RSRP-L2 의 거리 상관이 0.20 수준에 그쳐 거리 성분이 약함. 방위(AoA)가 지배적
  신호이며, AoA scatter 는 ~700 m 까지 상승 후 포화.
- t-SNE: HYBRID 맵이 거리(색) 군집을 가장 정합적으로 복원하나,
  Ground-Truth 부채꼴만큼 선명하지는 않음(다중경로 확산 한계).

## 산출물
- `P2B_GHM_Metric_Results/P2B_GHM_2x4_<timestamp>.png` : 2×4 (상관 scatter + t-SNE)
- `P2B_GHM_Metric_Results/P2B_GHM_corr_<timestamp>.csv` : 메트릭별 ρ 요약

## 실행 환경
- Python 3.10.12 / dclcom61 / numpy·scipy·scikit-learn(TSNE)·matplotlib (CPU)
