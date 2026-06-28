# P2C_GHM_UserGrouping_2606v2.py

## 목적
v1 결론(HYBRID-L1 메트릭으로 **거친 그룹핑은 가능**하나 K가 커질수록 ARI 하락)을
보완하기 위해 **방위(섹터) 기반 특징**을 추가하고, 세밀 그룹핑 성능을 비교한다.

## 핵심 아이디어
- 다중경로 환경의 지배적 신호는 **AoA(방위)**. 사용자별로 각 TX 에서 본
  **전력가중 평균 방위각(원형평균)** `θ_T = atan2(Σp·sinφ, Σp·cosφ)` 를 구한다.
- 3개 BS 의 방위 `(θ0, θ1, θ2)` 삼각형이 사용자 위치를 강하게 규정(방위 교차).
- 따라서 방위 벡터로 군집화하면 v1 분포형 메트릭보다 공간 정합(ARI) 개선 기대.

## 비교 방식 (정답 = 물리좌표 KMeans, ARI/NMI 비교)
| 기호 | 정의 | 군집기 |
|---|---|---|
| M0 HYBRID-L1 | v1 메트릭 거리행렬 (베이스라인) | Spectral(precomputed) |
| M1 Sector-AoA | 3 TX 원형평균 방위각의 **원형거리 L2** | Spectral(precomputed) |
| M2 Sector-KM | `[cosθ, sinθ]×3 = (n,6)` 특징 | KMeans |
| M3 Sector+Range | 방위특징 + 표준화 RSRP(n,3) 결합 | KMeans |

> 원형성 보존: 각도는 `cos/sin` 으로 임베딩해 ±180° 경계 불연속을 제거.
> M3 는 방위(교차)에 거리(range) 성분을 더해 반경 방향 분리도를 보강.

## 산출물
- `P2C_GHM_UserGrouping_Results/P2C_v2_Sector_<timestamp>.png`
  - 상단: ARI/NMI vs K 곡선 + 방위 특징공간(θ0-θ1) 산점도 + 방식별 max ARI 막대
  - 하단: 정답 + M0/M1/M3 군집을 물리좌표 위에 색칠(공간 정합 육안 확인)
- `P2C_GHM_UserGrouping_Results/P2C_v2_Sector_<timestamp>.csv`

## 실행 환경
- Python 3.10.12 / dclcom61 (deepgadget)
- numpy · scipy · scikit-learn(KMeans·Spectral) · matplotlib (CPU)
- 입력: `channel_data_260531_GHM_Twin_v0_1.npz` (TX 3, RX 5071, SISO)
- 변경 사유(2026-06-04): v1 대비 방위(섹터) 특징(M1~M3) 추가 — 세밀 그룹핑 ARI 개선 목적.
