# P2C_RANTwin_Interpolator_2606v3.py

## 목적
P2C v1/v2 의 학습형 보간기(MLP)가 GP-Kriging·IDW 보다 낮았던 문제를 구조적으로
바로잡는다. (held-out=미관측 RX 평가, 관측비율 sweep)

## v1 → v2 → v3 경과
| 버전 | 학습 모델 | 핵심 아이디어 | skill@90% |
|---|---|---|---|
| v1 | MLP(MSE) | 위치(x,y)→240차원 분포 회귀 | +0.369 |
| v2 | MLP-FF(KL) | Fourier feature + KL loss + 표본↑ | **+0.253 (악화)** |
| **v3** | **DeepKernel** | **학습형 전이적(attention) 보간** | **+0.529** |
| (참고) IDW | 거리역수² | baseline | +0.576 |
| (참고) GP | exp(−d/ℓ), ℓ 적합 | RAN Twin 커널 | **+0.616** |

## 왜 v2(FF+KL)가 더 나빠졌나 — 귀납 vs 전이
- GP/IDW(**전이적**): 질의 주변 *실제 관측 채널*을 직접 혼합 → NLoS 국소 디테일 활용.
- 위치→분포 회귀 MLP(**귀납적**): 정보를 가중치에 압축 → 큰 국소분산 앞에서 매끄러운
  평균(floor 근처)으로 붕괴. Fourier scale↑ 시 고주파 암기로 held-out 더 악화.
- 결론: 위치→분포 회귀로는 **구조적으로** GP를 못 이긴다.

## v3 해결책 — DeepKernel Attention (학습형 전이적 보간)
- 임베딩 `g: FourierFeature(x,y)→MLP→ℝ^32`, 학습형 온도 τ·길이스케일 ℓ.
- 어텐션 `w(q,j)=softmax_j(−‖g(x_q)−g(x_j)‖²/τ − ‖x_q−x_j‖²/ℓ²)`,
  예측 `= Σ_j w(q,j)·Y_obs[j]` (관측 분포들의 볼록결합 → 분포 정합 자동).
- 학습: 관측셋 **LOO**(자기 제외) 어텐션으로 관측 채널 복원 KL 최소화
  → 데이터 누수 없이 "관측으로 일반화되는 커널"을 학습, held-out 에 그대로 적용.

## 결과 해석
- DeepKernel 은 v2(+0.253)→ **+0.529** 로 대폭 회복(전이적 재구성이 옳은 방향).
- 다만 **GP(+0.616)·IDW(+0.576)에는 여전히 소폭 미달**.
  → 채널장이 비교적 매끄럽고 표본이 제한적이라, 단일 적합 길이스케일 GP 가 거의
    최적(bias-variance) → 학습 커널의 추가 유연성은 한계효용이 작고 약간 과적합.
- 즉 "학습 모델이 무조건 우월"하지 않으며, **이 데이터 규모/평활도에서는 GP 가 최적
  baseline**. 학습형은 비정상(이질적 환경)·대규모 데이터에서 이점이 커진다.

## 추가로 GP 초과를 노린다면 (후속 옵션)
1. **국소 k-NN 어텐션**(top-k) + τ 작게 초기화 → over-smoothing 억제.
2. **GP-잔차 학습**: GP 예측을 base 로 두고 DeepKernel 이 보정만 학습.
3. 이질 환경(여러 맵) 합쳐 학습 → 전역 커널의 일반화 이점 부각.

## 산출물 (P2C_RANTwin_Interpolator_Results/)
- `P2C_interp_curve_v3_<stamp>.png` : (a)원 유사도+floor (b)skill score
- `P2C_interp_v3_<stamp>.csv` : method×ratio×seed별 pdp_sim/cov_sim/S/skill

## 실행 환경
- Python 3.10.12 / dclcom61 / numpy·scipy·torch(2.11+cu128)·matplotlib
- 데이터: 260512 RT Result Data/channel_data_260531_GHM_Twin_v0_1.npz (TX3, SISO)
