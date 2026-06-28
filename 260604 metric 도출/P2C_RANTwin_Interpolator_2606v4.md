# P2C_RANTwin_Interpolator_2606v4.py  (GP-잔차 학습 — 음성 결과)

## 가설과 결과
"GP 예측을 base 로 두고 학습 보정(δ, 초기 s≈0)을 더하면 최소 GP 보장 + 초과 여지"
라는 가설로 구현했으나, **held-out 에서 GP 보다 크게 하회**(실패)했다.

| skill score | 10% | 50% | 90% |
|---|---|---|---|
| IDW | +0.370 | +0.537 | +0.576 |
| GP-Kriging | +0.375 | +0.570 | **+0.616** |
| GP+Residual(v4) | +0.094 | +0.321 | +0.411 |

## 실패 원인 (중요)
- 보정항 δ(x) 가 **귀납적 MLP** 라, 학습 타깃인 "관측 채널 − GP_LOO" 의 잔차를
  적합하는데, 이 잔차의 대부분은 **전이 불가능한 국소 aleatoric 노이즈**(NLoS의
  RX별 고유 변동)다. MLP 가 이를 암기 → 질의 시점에 잘못된 보정을 가해 held-out 악화.
- 즉 'GP 이상 보장' 은 query 보정 δ 가 0 일 때만 성립하는데, 학습이 δ 를 키워
  노이즈를 외우면서 보장이 깨진다. weight decay 3e-4 로도 부족.

## 결론 (P2C v1~v4 종합)
| 모델 | skill@90% | 성격 |
|---|---|---|
| v1 MLP(MSE) | +0.369 | 귀납 |
| v2 MLP-FF(KL) | +0.253 | 귀납(고주파 암기 악화) |
| v3 DeepKernel | +0.529 | 전이(학습 커널) — 학습형 중 최고 |
| v4 GP+Residual | +0.411 | GP base+귀납 보정(노이즈 과적합) |
| IDW | +0.576 | 전이(고정) |
| **GP-Kriging** | **+0.616** | **전이(단일 ℓ 적합) — 전체 최고** |

- **이 데이터(평활한 채널장 + 제한 표본 + 큰 NLoS aleatoric)에서는 GP-Kriging 이
  사실상 최적**이다. 어떤 학습형 보강(귀납 MLP/Fourier/딥커널/잔차)도 GP 를 못 넘는다.
  추가 유연성이 '전이 불가능한 노이즈' 를 과적합하기 때문.
- 학습형이 이점을 가지려면: (a) 관측 내 **검증분할 early-stopping** 으로 GP 하회를
  원천 차단, (b) **이질적/대규모 데이터**(여러 맵)로 전역 커널의 일반화 우위 확보.

## 산출물
- `P2C_interp_curve_v4_<stamp>.png`, `P2C_interp_v4_<stamp>.csv`

## 실행 환경
- Python 3.10.12 / dclcom61 / numpy·scipy·torch(2.11+cu128)·matplotlib
- 데이터: 260512 RT Result Data/channel_data_260531_GHM_Twin_v0_1.npz (TX3, SISO)
