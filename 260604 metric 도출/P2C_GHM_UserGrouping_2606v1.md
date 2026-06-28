# P2C_GHM_UserGrouping_2606v1.py

## 목적
P2B 에서 도출한 채널 메트릭(**HYBRID(L1)** = `α·AoA_L1 + (1-α)·RSRP`)이
**K-clustering(User Grouping)** 에 실제로 쓸 수 있는지를 정량 지표로 검증한다.
Spearman ρ(상관)만으로는 "군집이 잘 되는지"를 알 수 없으므로, 직접 군집화한 뒤
물리 공간 그룹과의 정합도를 측정한다.

## 방법
1. P2B 와 동일 파이프라인으로 샘플(거리 층화 n=900)·특징(APS, RSRP)·
   HYBRID(L1) 거리행렬 `D (n×n, [0,1])` 생성.
2. **정답(GT) 그룹**: 물리좌표 `(x,y)` 에 `KMeans(K)` → `labels_gt`.
3. **메트릭 그룹**: 거리행렬 `D` 에 세 방식 적용
   - (a) `AgglomerativeClustering(average, precomputed)`
   - (b) `SpectralClustering(affinity=exp(-D²/2σ²), precomputed)`
   - (c) `KMeans` on `MDS(2D)` 임베딩
4. **평가지표**
   | 지표 | 의미 | 판정 |
   |---|---|---|
   | ARI | 메트릭 군집 vs 정답(외부 정합도) | ≳0.3 의미있음 / ~0 무작위 |
   | NMI | 정보이론 기반 정합도 | 높을수록 일치 |
   | Silhouette(precomputed) | 메트릭 공간 내 군집 응집도 | >0 군집 구조 존재 |
   | random ARI | 무작위 라벨 베이스라인 | ≈0 기준선 |
5. `K = 2..10` 스윕 곡선 + 대표 `K=6` 의 공간 산점도(정답/메트릭 군집 색칠).

## 산출물
- `P2C_GHM_UserGrouping_Results/P2C_UserGrouping_<timestamp>.png`
  - 상단: ARI / NMI / Silhouette vs K 곡선 + 메트릭 MDS 임베딩
  - 하단: 정답 군집 + 메트릭 3방식 군집을 **물리좌표 위에** 색칠(공간 정합 육안 확인)
- `P2C_GHM_UserGrouping_Results/P2C_UserGrouping_<timestamp>.csv`
  - K·method 별 ARI/NMI/silhouette + 무작위 베이스라인

## 해석 가이드
- ARI 가 random 곡선 위로 뚜렷이 올라오면 → 거리-비례 그룹핑이 동작.
- ARI 가 random 과 붙어 있으면 → 메트릭은 연속 분포라 '자연 그룹' 식별 곤란
  (섹터/각도 기반 또는 거리성분 보강 필요).

## 실행 환경
- Python 3.10.12 / dclcom61 (deepgadget)
- numpy · scipy · scikit-learn(KMeans·Spectral·Agglomerative·MDS) · matplotlib (CPU)
- 입력: `channel_data_260531_GHM_Twin_v0_1.npz` (TX 3, RX 5071, SISO)
- 연산: 거리행렬 900×900, K 스윕 2..10
