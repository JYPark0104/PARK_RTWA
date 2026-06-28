# P2G_WeightFit_2606v1.py — 단계 4: NNLS 가중치 학습

## 목적
`d_metric = w_R·d_RSRP + w_P·d_PDP + w_C·d_Cov ≈ d_phys[m]` 를
origin-through(절편 없음), `w>=0` 제약으로 적합하여 가중치를 산출.

## 절차
1. **scale-only(RMS) 정규화**: `s_k=sqrt(mean(X_k²))`, `X_norm=X/s` (중심화 금지 → origin-through 보존)
2. `scipy.optimize.nnls(X_norm, y)` → `w_norm>=0`
3. **원스케일 역변환**: `w_k = w_norm_k / s_k` → 단위 `w_R[m/dB]`, `w_P[m/ns]`, `w_C[m]`

## R² 정의
`pred = X@w (=d_metric)` 를 `y=d_phys` 와 **직접**(기울기-1, 1:1선) 비교:
`R² = 1 - Σ(y-pred)² / Σ(y-ȳ)²`. R²=0 은 "평균 예측" 수준.

## 입력 / 출력
- 입력: `P2G_PairTable_Results/P2G_PairTable_*.npz` (X, y)
- 출력(`P2G_WeightFit_Results/`):
  - `P2G_weights_<stamp>.npz` (w_full, scales, ablation 배열)
  - `P2G_weights_ablation_<stamp>.csv`
  - `P2G_weightfit_<stamp>.png` (ablation R²/RMSE + 기여도)
  - `P2G_weightfit_report_<stamp>.log`

## 핵심 결과 (20260606)
- RMS 스케일: d_RSRP=31.6 dB, d_PDP=240.8 ns, d_Cov=1.356
- **full**: `w_R=0.510 m/dB, w_P=0.025 m/ns, w_C=57.41 m` | R²=**0.143**, RMSE=**42.0m**
- ablation R²: Cov-only 0.058 > full 0.143; RSRP-only −1.13, PDP-only −2.12 (음수=평균보다 나쁨)
- 음수 weight 없음(NNLS 정상). w_C>0 → 근거리 재적합 생략.

## ⚠️ 해석상 중요한 주의 (d_Cov 의 의사-절편 효과)
- **(d) RMS 정규화 후 d_Cov 분포**: median 1.026, std/mean=0.10 → 대부분 ~1.03 에 **상수처럼 포화**.
- origin-through(절편 없음)에서 거의 상수인 d_Cov 는 **사실상 절편(intercept) 역할**을 한다.
  그래서 기여도 (b) 에서 d_Cov 가 77.5m(83%)로 보이지만, 이는 "평균 거리(≈93m)를
  맞추는 상수항"에 가깝고 **거리 분해 능력(slope) 이 아님**.
- 근거: d_Cov 없는 RSRP+PDP 는 R²=−1.04(평균조차 못 맞춤) → d_Cov 가 절편을 대신해
  R²를 −1→+0.14 로 끌어올린 것. 실제 변별은 d_RSRP/d_PDP 가 slope 로 약간 기여.
- **물리적 타당성**: 단일 BS 채널 특징으로 UE-UE 거리를 직접 복원하는 것은 본질적으로
  어려움(거리·각도 모호성) → R²≈0.14 는 비현실적으로 낮은 값이 아님.

## 후속 권고 (단계5 검토용)
- 절편 항 허용(`d_metric = b + Σw_k d_k`) vs origin-through 비교 → d_Cov 의 진짜 slope 기여 분리.
- d_phys 가 아니라 BS-거리/각도 같은 단일-BS로 식별 가능한 타깃 재검토 가능성.

## 실행
```bash
python3 P2G_WeightFit_2606v1.py
```
