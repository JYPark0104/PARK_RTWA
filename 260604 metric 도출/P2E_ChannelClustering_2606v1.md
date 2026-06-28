# P2E_ChannelClustering_2606v1.py

## 목적
P2D 채널 거리(Bures-Cov + 1D-Wasserstein-PDP, trace=1, ρ≈0.60) 기반 6G Digital Twin
**user grouping**(채널 유사 UE 묶기). 커스텀 메트릭 → 사전계산 거리행렬 군집화.

## 방법
- 거리행렬(N×N) 벡터화: `D = 0.77·minmax(Bures-shape) + 0.23·minmax(PDP-Wass)`.
- 군집: **K-medoids**(자체 PAM 구현) / **Spectral**(precomputed affinity) /
  **Agglomerative**(precomputed, average linkage).
- 최적 k: **Silhouette(precomputed)** k=2~10 스캔.

## 결과 (n=1000, 3-BS)
| 방법 | 최적 k | Silhouette |
|---|---|---|
| K-medoids | 8 | +0.098 |
| **Spectral** | **5** | **+0.117** |
| Agglomerative | 2 | +0.106 |

## 해석
- **Spectral(k=5)·K-medoids(k=8)는 물리지도에서 공간적으로 응집된 구역(섹터)** 형성
  → 채널 유사도 군집이 지리적 그룹으로 연결됨(메트릭-거리 상관 0.60의 직접 증거).
  user grouping(빔/스케줄링 그룹화)에 활용 가능.
- **Silhouette가 0.1 수준으로 낮은 건 실패가 아니라**, 데이터가 **연속적 채널 장**
  (뚜렷한 군집 경계 없는 continuum)이기 때문. 경계가 부드러워 silhouette는 낮아도
  공간 분할은 일관적.
- **Agglomerative(average linkage)는 k=2로 퇴화**(한 거대 군집 + 잔여) — chaining 현상.
  → `linkage='complete'` 또는 'ward(유클리드 임베딩)'로 개선 여지.

## 산출물 (P2E_ChannelClustering_Results/)
- `P2E_clustering_<stamp>.png` : (a)Silhouette vs k + (b~d) 방법별 최적 k 물리지도
- `P2E_clustering_<stamp>.csv` : 방법×k silhouette 표

## 실행 환경
- Python 3.10.12 / dclcom61 / numpy·scipy·scikit-learn·matplotlib
- 데이터: 260512 RT Result Data/channel_data_260531_GHM_Twin_v0_1.npz (TX3, SISO)
