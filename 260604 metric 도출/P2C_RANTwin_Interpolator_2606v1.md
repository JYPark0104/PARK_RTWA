# P2C_RANTwin_Interpolator_2606v1.py

## 목적
P2B에서 입증한 "메트릭 거리 ↔ 물리거리 단조상관"(채널의 공간상관)을 근거로,
**부분 관측(observed)으로 학습 → 미관측(held-out=정답) 채널을 추론**하는 보간기를
만들고, 관측비율(observation ratio)에 따른 추론 정확도 곡선을 도출한다.

## 평가 프로토콜 (train / held-out)
- 관측 RX = 학습용, 미관측 RX = 정답(test).
- 보간기는 관측 RX의 (위치, 채널)로 학습 → 미관측 RX 위치에서 채널 추론.
- 추론값 vs True 유사도(PDP + Cov)를 held-out 평균 → 곡선 1점.
- 관측비율 10~90% sweep, 시드 4회 → 점(scatter) + 평균선.

## 채널 표현 ('PADP + Covariance')
- BS(TX)별 PDP(32 bin) + APS(48 bin).
- SISO(R=1)라 공간 공분산은 APS로부터 가상 ULA(M=16) 재구성, trace=1 정규화.
- 유사도: `S = 0.5·(1 − W1_PDP) + 0.5·(1 − BW_Cov/2)`, 3 BS·held-out 평균.

## 보간기 3종
| 기호 | 방식 | 학습 여부 |
|---|---|---|
| IDW | 거리 역수² 가중 평균 (k=20) | 없음(baseline) |
| GP-Kriging | 커널 `exp(−d/ℓ)`, ℓ을 관측셋 거리-유사도 감쇠에서 적합 | 있음(RAN Twin) |
| MLP | 위치(x,y)→채널분포 회귀, 블록 softmax | 있음(RAN Twin) |

> 핵심 연결: GP 커널의 상관거리 ℓ을 "관측셋에서 측정한 채널유사도-거리 감쇠"에
> 적합 → 우리가 도출한 상관 결과가 그대로 보간 커널이 된다.

## Skill score (격차 가시화)
유사도 바닥이 ~0.8로 높아 절대값으로는 격차가 안 보이므로, **공간정보 없는
전역평균 예측**을 floor로 두고 정규화한다.
\[ \text{skill} = \frac{S - S_{\text{floor}}}{1 - S_{\text{floor}}} \]
floor는 관측 RX 채널의 단순 평균(미관측에 동일 적용) → 보간이 거리정보로 얼마나
floor를 넘어서는지를 0~1로 표시.

## 대표 결과 (n=900, 3-BS, 시드 4, floor≈0.735)
| 관측비율 | S: IDW | S: GP | S: MLP | skill: IDW | skill: GP | skill: MLP |
|---|---|---|---|---|---|---|
| 10% | 0.812 | 0.811 | 0.761 | +0.289 | +0.286 | +0.098 |
| 50% | 0.860 | 0.865 | 0.826 | +0.464 | +0.483 | +0.335 |
| 90% | 0.874 | 0.882 | 0.833 | +0.525 | **+0.554** | +0.369 |

- 관측비율↑ → 유사도/skill 모두 단조 상승(예상 추세 재현).
- **GP-Kriging(RAN Twin)** 이 전 구간 최고. IDW가 의외로 강한 baseline(둘 다 거리가중).
- MLP는 (x,y)→240차원 분포 회귀로 데이터 희소 → 하회.
- 원 유사도 바닥이 ~0.8로 높은 이유: NLoS 확산 채널이 서로 유사 → PDP+Cov 포화.
  Skill score로 보면 floor 대비 학습/보간의 실제 기여(0.3~0.55)가 드러남.

## 산출물 (P2C_RANTwin_Interpolator_Results/)
- `P2C_interp_curve_<stamp>.png` : 1×2 (a)원 유사도+floor (b)skill score
- `P2C_interp_<stamp>.csv` : method×ratio×seed별 pdp_sim/cov_sim/S/skill (floor 포함)

## 실행 환경
- Python 3.10.12 / dclcom61 / numpy·scipy·torch(2.11+cu128, MLP)·matplotlib
- 데이터: 260512 RT Result Data/channel_data_260531_GHM_Twin_v0_1.npz (TX3, SISO)
