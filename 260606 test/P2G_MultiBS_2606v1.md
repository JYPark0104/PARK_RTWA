# P2G_MultiBS_2606v1.py — multi-BS 거리 복원 검증

## 목적
"단일 BS 천장(R²≈0.14)을 multi-BS(삼각측량)가 푸는가?"를 3-BS 데이터로 직접 검증.
나아가 병목이 (a) 정보(RSRP noise)인지 (b) 메트릭 형태(가중합 pairwise)인지 분리.

## 데이터
`260512 RT Result Data/channel_data_260531_GHM_Twin_v0_1.npz`
- rsrp_all (3,5071), rx_positions(5071,2), tx_positions(3,3)
- (R_TX/R_RX 는 (1,1) SISO -> 공분산 없음, RSRP만 사용)
- 큰 영역: d_phys 중앙 525m, 최대 1821m, NLoS 다수

## 실험 (서브샘플 n=3000, 쌍 4.5M)
1. RSRP pairwise: f_b=|Δrsrp_b|, OLS로 1/2/3-BS R²
2. 이상적 거리 pairwise: f_b=|Δrange_b| (참 BS거리=ToA 상한)
3. 정식 삼각측량: 참거리 3개 -> 위치 추정 -> 쌍거리

## 결과 (20260606) — 핵심
| 방법 | R² |
|---|---|
| RSRP pairwise 1-BS(최고) | 0.019 |
| RSRP pairwise 3-BS | 0.026 |
| 이상적 거리 pairwise 3-BS | 0.153 |
| **정식 삼각측량(위치추정후 거리)** | **1.000** (RMSE 1e-13m) |

## 해석 — 병목은 둘
1. **정보 품질**: RSRP는 NLoS shadowing으로 거리추정이 noisy → 3-BS RSRP도 R²=0.026.
   정확한 거리량(ToA)이 필요.
2. **메트릭 형태**: 거리 정보가 완벽(참거리)해도 '가중합 pairwise'(Σw_b|Δr_b|)는
   R²=0.153에 그침. 유클리드 거리는 pairwise 채널차의 선형결합으로 표현 불가.
   같은 정보를 **위치 추정(삼각측량)** 에 쓰면 R²=1.000.

## 결론 (전체 프로젝트 관통)
> "채널 거리 = 항들의 가중합" 프레임은 특징을 아무리 늘려도 본질적 천장이 있다.
> UE-UE 물리거리 복원의 정답은 (1) ToA 같은 정확한 거리량 + (2) 가중합이 아니라
> **multi-BS 위치 추정(localize-then-distance)** 모델이다.

## 실행
```bash
python3 P2G_MultiBS_2606v1.py
```
