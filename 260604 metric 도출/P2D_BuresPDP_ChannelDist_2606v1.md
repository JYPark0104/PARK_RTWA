# P2D_BuresPDP_ChannelDist_2606v1.py

## 목적
6G NLoS Digital Twin 연구용 : 두 노드(RX) 간 전파 채널 거리를 **Bures metric(Cov) +
1D Wasserstein(PDP) 결합 수식**으로 계산하고 물리거리와의 상관(Scatter + Spearman ρ)
도출.

## 채널 거리 수식 (기지국별 D²)
```
D² = γ·( Tr(R1)+Tr(R2) − 2 Tr( √( √R1 · R2 · √R1 ) ) )            # Bures(Cov)
   + δ·( √( Tr(R1)Tr(R2) ) · Σ_i | CDF1(i) − CDF2(i) | )          # PDP Wasserstein
```
- R : 48-bin APS 공분산. **대각 취급 시 Bures = Hellinger = Σ(√r1−√r2)²** (효율 동치 구현).
- Tr(R) = 총 수신전력 = APS 배열 합.  p̂ = 합1 정규화 32-bin PDP, CDF = cumsum.
- 3 BS D² 합산 → √ → `D_total`.

## 구현 함수
- `compute_channel_distance(profile1, profile2, gamma, delta, cov_mode, reduce)`
  - `cov_mode='diag'`(Hellinger, 기본) / `'full'`(가상 ULA M=16 재구성 + `scipy.linalg.sqrtm`).
  - PDP 항: `cumsum`/`abs` 로 1D Wasserstein 그대로.
- `compute_terms_all()` : 분석용으로 raw 항과 trace=1 정규화 항을 동시 반환.
- profile(240) = 3 BS × 80, BS당 [0:32]=raw PDP, [32:80]=raw APS.

## 결과 (n_pair≈1498, 노드 1200, dist 13~1543 m) — **핵심**
| 구분 | 메트릭 | Spearman ρ |
|---|---|---|
| (A) 수식 raw | Bures(raw) | −0.157 |
| (A) 수식 raw | PDP(raw, √Tr 스케일) | −0.146 |
| (A) 수식 raw | 결합(γ=δ=1) | **−0.161** |
| (B) trace=1 | Bures-shape | +0.568 |
| (B) trace=1 | Wasserstein | +0.383 |
| (B) trace=1 | **결합(γ:δ≈0.77:0.23)** | **+0.600** |

## 핵심 해석 (반드시 주의)
- **수식을 raw 전력 그대로 쓰면 음상관(ρ≈−0.16)** : `Tr(R)=총전력`·`√(Tr1Tr2)`
  스케일이 **전력 크기에 지배**되어, 가까운(고전력) 쌍이 오히려 큰 채널거리를 가짐.
  (P2A 문서의 "BW가 trace에 지배 → ρ≈−0.16" 함정과 정확히 일치.)
- **해법 = trace=1 정규화**(R/Tr) → Bures가 '각도 형상' 차이만 측정. PDP는 순수
  Wasserstein. 이때 **결합 ρ=+0.600** 으로 프로젝트 최고(>P2B 0.539).
- 즉 γ,δ 가중 이전에 **각 항의 trace 정규화가 필수**. Bures-shape(0.568)가 지배적
  공간정보, Wasserstein(0.383)이 거리 보강.

## 산출물 (P2D_BuresPDP_ChannelDist_Results/)
- `P2D_BuresPDP_scatter_<stamp>.png` : 2×3 (상단 raw 함정 / 하단 trace=1 해법)
- `P2D_BuresPDP_corr_<stamp>.csv`

## 실행 환경
- Python 3.10.12 / dclcom61 / numpy·scipy·matplotlib
- 데이터: 260512 RT Result Data/channel_data_260531_GHM_Twin_v0_1.npz (TX3, SISO)
