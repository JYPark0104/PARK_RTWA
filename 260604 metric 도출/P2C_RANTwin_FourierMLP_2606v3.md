# P2C_RANTwin_FourierMLP_2606v3.py

## 목적 (v2→v3)
v1~v2의 일반 MLP 패배 원인을 **저주파 편향(Spectral Bias)** 으로 보고, NeRF식
**고주파 위치 인코딩(Fourier Features)** 을 도입해 위치-only 입력으로 보간을
이길 수 있는지 검증.

## 구현 아키텍처 (사용자 스펙)
- `PositionalEncoding(L=10)` : γ(p)=[p, sin(2^kπp), cos(2^kπp)], k=0..L-1. out_dim=2(1+2L)=42.
- `ChannelInterpolatorMLP` : PE → Linear(256)+GELU ×4(얕음) → Linear(240) → Softplus →
  블록별 L1 정규화(PDP 3×32, APS 3×48 확률분포).
- `channel_loss` : `W_MSE·MSE + W_WASS·Wass1D(CDF-L1) + W_L1·L1` (1.0/0.5/0.2).
- 좌표 [-1,1] 정규화, Adam lr 2e-3, weight_decay 1e-4, 800 epochs.

## 결과 (n=900, 3-BS, 시드4, floor≈0.738) — **여전히 보간 우세(negative)**
| 관측비율 | IDW(위치) | Plain MLP | Fourier MLP |
|---|---|---|---|
| 10% | **0.812** | 0.738 | 0.720 |
| 50% | **0.860** | 0.767 | 0.767 |
| 90% | **0.874** | 0.777 | 0.787 |

- **IDW(위치보간)가 압도적 1위.** MLP 둘 다 floor 근처에 머묾.
- Fourier MLP ≥ Plain MLP(고관측 구간) → 저주파편향 일부 완화 확인(가설 부분 성립).
  그러나 IDW와의 격차를 메우진 못함.

## 핵심 진단 (왜 안 되는가)
1. **국소 비모수(IDW/kNN) vs 전역 모수(MLP)**: 채널이 위치의 매끄러운 장(field)이고
   좌표가 정확하면, 이웃을 복사하는 국소보간이 거의 최적. 전역 MLP는 240차원
   다봉 함수를 매끄럽게 근사 → 점별 재구성에서 평활화 손실.
2. NeRF가 통하는 조건(단일 장면의 조밀한 지도학습)과 달리, 여기는 **희소 표본 +
   held-out 일반화** → 국소보간이 유리한 정석 영역.
3. 따라서 **정확한 위치가 주어진 보간 과제에서는 MLP(Fourier 포함)가 보간을
   이기기 어렵다**는 결론이 v1/v2/v3 에 걸쳐 일관됨.

## 학습이 실제로 이기는 조건 (다음 후보)
- (A) **위치 잡음/미상**: 보간이 이웃을 잘못 고름 → 학습이 RF측정 융합으로 우위.
- (B) **Residual/Hybrid**: IDW 베이스 + MLP가 잔차 학습(국소+전역 결합).
- (C) **외삽/시간예측**: 관측 영역 밖 또는 미래 시점 → 보간 불가, 학습 필요.

## 산출물 (P2C_RANTwin_Interpolator_Results/)
- `P2C_v3_fourier_curve_<stamp>.png` : 1×2 (원 유사도 / skill)
- `P2C_v3_fourier_<stamp>.csv`

## 실행 환경
- Python 3.10.12 / dclcom61 / numpy·torch(2.11+cu128)·matplotlib
- 데이터: 260512 RT Result Data/channel_data_260531_GHM_Twin_v0_1.npz (TX3, SISO)
