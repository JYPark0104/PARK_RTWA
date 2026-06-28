# 채널 거리 메트릭 ↔ 물리거리 가중치 학습 — 종합 리포트 (ver0.1 + 진단)

- 작성일: 2026-06-06
- 서버: dclserver78 (NVIDIA H100 NVL 95GB x 2)
- 환경: Python 3.10.12 / torch 2.12.0+cu130 / numpy 2.2.6 / scipy 1.15.3
- 데이터: Area1 7.5GHz (단일 BS, RX 1037 공통), multi-BS 검증은 channel_data_260531(BS 3개)

---

## 0. 목표와 한 줄 결론

**목표**: 단일 BS 채널에서 UE 쌍의 채널 거리 메트릭
`d_metric = w_R·d_RSRP + w_P·d_PDP + w_C·d_Cov` 를 NNLS(origin-through, w≥0)로
UE-UE 물리거리 `d_phys` 에 맞춰 학습하고 유효성을 검증.

**한 줄 결론**:
> 메트릭/가중치는 **수학적으로 최적**이지만 `d_phys` 복원 R²≈0.14 가 천장.
> 이는 적합 버그가 아니라 **(1) 단일-BS 정보 부족 + (2) "가중합 pairwise" 메트릭 형태의 한계**.
> 정답 방향은 **ToA 같은 정확한 거리량 + multi-BS 위치추정(localize-then-distance)**.

---

## 1. 파이프라인 단계별 산출 (단계1~4)

### 단계 1 — 점 테이블 (`P2G_PointTable_Builder_2606v1.py`)
- P1A rays(RSRP/PDP/위치) ∩ P1F CCM(R_BS 1024² BS-side) → 공통 N=1037.
- R_BS 저랭크 인수분해: rank=96 (P2G_dCov_RankCheck로 r=32→96 확정, 최대오차 0.91%).
  스트리밍+멀티GPU로 메모리 8.7GB→0.8GB.
- 위치 환원(idx-1, 40×40 grid) 및 BS거리/축 sanity 통과.

### 단계 2 — 거리 항 3함수 + 거리행렬 (`P2G_DistTerms_2606v1.py`)
- d_RSRP = |ΔRSRP| (dB, L1)
- d_PDP = 1-Wasserstein (ns, 미세격자 CDF-L1; scipy 점질량 W1과 단위테스트 일치)
- d_Cov = Bures-Wasserstein (`tr+tr-2‖B_iᴴB_j‖_*`, trace정규화+εI; 풀랭크 동치 검증 완료)
- 산출: D_rsrp, D_pdp, D_cov (1037², NxN)

### 단계 3 — 쌍 테이블 + 포화 분석 (`P2G_PairTable_2606v1.py`)
- 상삼각 M=537,166 쌍. X=[d_RSRP,d_PDP,d_Cov], y=d_phys.
- **Spearman**: d_RSRP 0.318 > d_PDP 0.241 > d_Cov 0.195
- **d_Cov 포화**: median 곡선이 ~38m에서 plateau(≈1.39, √2=1.414)에 도달 → 그 이상 분해능 없음.
  d_Cov<1.40(비포화) 65.5%. 거리 구간별 비포화율: 0-10m 97% → 80m+ 61%.

![단계3](P2G_PairTable_Results/P2G_terms_vs_dphys_20260606_084852.png)

### 단계 4 — NNLS 가중치 학습 (`P2G_WeightFit_2606v1.py`)
- scale-only(RMS) 정규화 → nnls(origin-through,w≥0) → 원스케일 역변환.
- **full**: `w_R=0.510 m/dB, w_P=0.025 m/ns, w_C=57.41 m` | R²=0.143, RMSE=42.0m
- ablation R²: RSRP-only −1.13, PDP-only −2.12, Cov-only 0.058, RSRP+PDP −1.04, full 0.143
- 주의: RMS후 d_Cov가 ~1.03 상수에 가까워(std/mean=0.10) **origin-through에서 의사-절편** 역할.

![단계4](P2G_WeightFit_Results/P2G_weightfit_20260606_085451.png)

---

## 2. 진단: R²<0 의 원인 분해 (`P2G_FitDiag_2606v1.py`)

| 점검 | 결과 | 판정 |
|---|---|---|
| (1) intercept 비교 (full) | 0.143 → 0.144 (불변) | full엔 d_Cov가 이미 의사절편 |
| (1b) RSRP+PDP intercept | **−1.04 → +0.099** | ablation의 R²<0은 origin-through 산물 |
| (2b) 근거리(<40m) 재적합 | R²≤0.106 | 근거리도 약함 |
| (3) RMS후 std/mean | RSRP 0.86, PDP 1.25, Cov 0.10 | 스케일 문제 아님(RSRP/PDP 분산 충분) |

근거리(<10m) 쌍의 d_Cov 실측: 같은 5m인데 **0.09~1.41로 요동** →
"채널 유사도 ≠ 물리적 근접". 거리 내 분산이 거리 간 분산보다 큼 → 낮은 R²의 본질.

![진단](P2G_FitDiag_Results/P2G_fitdiag_20260606_085714.png)

### weight 최적성 / slope 1 불가
- NNLS≈OLS → **전역 최적 weight 확인**. 더 나은 선형 조합 없음.
- 회귀 항등식 `std(pred)/std(y)=corr=0.38` → 예측이 평균으로 수축(90m 구름)은 수학적 필연.
- 강제 slope=1 시 MSE 1.45배 악화(R²<0). 즉 weight 문제 아님, **corr=0.38(정보)** 문제.

---

## 3. 보강 시도와 한계 측정

### ver0.2 — UE-side 공분산 추가 (`P2G_AddUECov_2606v1.py`)
- R_UE(16²) → d_Cov_UE. d_phys corr=**0.053**(거의 무상관).
- ver0.1 R²=0.1443 → FULL4 R²=**0.1444 (+0.0000)**. **효과 없음.**
- 이유: R_UE는 UE 국소 산란(도래각)이라 BS-UE 거리와 무관.

![UEcov](P2G_AddUECov_Results/P2G_adduecov_20260606_090737.png)

### 타깃 재설정 (`P2G_Retarget_2606v1.py`)
- T_phys 0.144 > T_range 0.110 > T_angle 0.086 → 타깃 바꿔도 천장 못 넘음.
- 단 역할 구조: **d_Cov_BS↔방위각(0.28), d_RSRP↔거리(0.31)**.

![retarget](P2G_Retarget_Results/P2G_retarget_20260606_091240.png)

### multi-BS 검증 (`P2G_MultiBS_2606v1.py`)
| 방법 | R² |
|---|---|
| RSRP pairwise 3-BS | 0.026 |
| 이상적 거리(ToA) pairwise 3-BS | 0.153 |
| **정식 삼각측량(위치추정→거리)** | **1.000** (RMSE 1e-13m) |

**병목 2개 확정**: (1) RSRP는 NLoS shadowing으로 거리추정 noisy → ToA 필요.
(2) 정보가 완벽해도 "가중합 pairwise"는 R²=0.15 한계 → **위치추정 모델** 필요.

![multibs](P2G_MultiBS_Results/P2G_multibs_20260606_091605.png)

---

## 4. 종합 결론 & 다음 단계

### 결론
1. ver0.1 메트릭과 NNLS 가중치는 **정확·최적**. R²≈0.14 는 측정된 **경계값**.
2. 단일-BS에 어떤 단일-링크 특징(BS-side/UE-side)을 더해도 천장 불변.
3. "채널거리=가중합" 프레임은 **정보가 완벽해도** 유클리드 거리를 못 만든다(R²=0.15 상한).
4. 정답: **ToA(정확 거리량) + multi-BS 위치추정(localize-then-distance)** → R²=1.0 입증.

### 권고 다음 단계
- (a) Area1 rays의 **LoS ToA/AoD** 추출 → 단일-BS 거리·각도 직접 추정 정확도 측정.
- (b) multi-BS ToA 트라일래터레이션 파이프라인(채널→ToA→위치→거리) 구축.
- (c) 메트릭이 정말 필요한 용도(채널 군집/검색)로 재포지셔닝 검토.

### 산출물 색인 (260606 test/)
| 단계 | 스크립트 | 결과 폴더 |
|---|---|---|
| BW 근사검증 | P2G_BW_Approx_Check_2606v1.py | (로그) |
| 랭크검증 | P2G_dCov_RankCheck_2606v1.py | (로그) |
| 1 점테이블 | P2G_PointTable_Builder_2606v1.py | P2G_PointTable_Results/ |
| GPU 유틸 | P2G_gpu_utils_2606v1.py | - |
| 2 거리항 | P2G_DistTerms_2606v1.py | P2G_DistTerms_Results/ |
| 3 쌍테이블 | P2G_PairTable_2606v1.py | P2G_PairTable_Results/ |
| 4 가중치 | P2G_WeightFit_2606v1.py | P2G_WeightFit_Results/ |
| 진단 | P2G_FitDiag_2606v1.py | P2G_FitDiag_Results/ |
| ver0.2 UE | P2G_AddUECov_2606v1.py | P2G_AddUECov_Results/ |
| 타깃재설정 | P2G_Retarget_2606v1.py | P2G_Retarget_Results/ |
| multi-BS | P2G_MultiBS_2606v1.py | P2G_MultiBS_Results/ |
