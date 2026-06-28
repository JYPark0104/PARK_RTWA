# P2B_GHM_Metric_Compare_2606v1.py

## 목적
다중 기지국(Multi-BS) AoA 결합의 두 방식을 **동일 샘플·동일 전처리**로 한 번에
계산하여 공정 비교한다. 이전 실행본(`114308`, 가중합)과 최신본(`115151`, L2)이
서로 다른 코드에서 나왔고, 문서 결론과 달리 가중합(ρ=0.539)이 L2(ρ=0.505)보다
높았기 때문에 원인을 한 그림에서 검증한다.

## 비교 대상
| 기호 | 정의 | 비고 |
|---|---|---|
| PDP-Wass (참고) | 지연 PDP(32 bin) 1D Wasserstein | 단일 BS, 거리 정보 |
| AoA-Wass TX0 (참고) | 방위 APS(48 bin) 1D Wasserstein | 단일 BS, 방위 정보 |
| **AoA (L1 가중합)** | `w0·D0 + w1·D1 + w2·D2` (단체격자 자동탐색) | TX별 가중치 자유도 |
| **AoA (L2 유클리드)** | `√(D0²+D1²+D2²)` | 3 TX 동일가중 |
| RSRP (n,3) L2 | 기지국별 RSRP 벡터 유클리드 거리 | 다변측위 거리 |
| HYBRID(L1)+RSRP | `α·AoA_L1 + (1-α)·RSRP`, α 자동탐색 | |
| HYBRID(L2)+RSRP | `α·AoA_L2 + (1-α)·RSRP`, α 자동탐색 | |

## 핵심 비교 포인트
- **자유도 차이**: L1 가중합은 TX별 기여도를 단체(simplex) 격자에서 자유 탐색하므로
  거리 관측성이 약한 TX(예: TX1)의 비중을 낮춰 노이즈를 억제할 수 있다.
- L2 유클리드는 3 TX를 동일가중으로 제곱합 → 약한 BS의 분산이 그대로 섞임.
- 따라서 Spearman ρ 기준으로는 **L1 가중합이 우세**할 것으로 예상(검증 목적).

## 산출물
- `P2B_GHM_Metric_Results/P2B_GHM_compare_<timestamp>.png` : 2×4
  (상단: GT / AoA-L1 / AoA-L2 scatter + ρ 막대비교, 하단: HYBRID L1·L2 scatter + GT·우세 t-SNE)
- `P2B_GHM_Metric_Results/P2B_GHM_compare_<timestamp>.csv` : 메트릭별 ρ + WINNER

## 실행 환경
- Python 3.10.12 / dclcom61 / numpy·scipy·scikit-learn(TSNE)·matplotlib (CPU)
- N(샘플) 900, 가중치 단체격자 step=0.05, 가속용 rank-Pearson Spearman 사용
