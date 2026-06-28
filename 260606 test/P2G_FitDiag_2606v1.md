# P2G_FitDiag_2606v1.py — 단계4 진단: R²<0 원인 분해

## 목적
ablation 에서 단일/2항 R²<0 이 **origin-through 제약(적합 설정)** 탓인지
**물리적 정보 부족(메트릭 한계)** 탓인지 분리.

## 점검 3종
1. intercept 비교: origin-through(NNLS) / intercept+비음수(lsq_linear) / intercept+OLS
2. 거리 구간별 R² + 근거리(<40m) 재적합 상한 R²
3. RMS 스케일 후 세 열 분포 (RSRP/PDP가 d_Cov에 눌렸는지)

## 입력 / 출력
- 입력: `P2G_PairTable_Results/P2G_PairTable_*.npz`
- 출력(`P2G_FitDiag_Results/`): `P2G_fitdiag_<stamp>.png`, `..._report_<stamp>.log`

## 결과 (20260606) — 핵심 판정

### (1) intercept 비교
| 모델 | b[m] | R² | RMSE |
|---|---|---|---|
| full origin-through | 0 | 0.143 | 42.0 |
| full intercept+비음수 | −18.0 | **0.144** | 42.0 |
| **RSRP+PDP** origin-through | 0 | **−1.04** | 64.8 |
| **RSRP+PDP** intercept | 76.0 | **+0.099** | 43.1 |

→ **단일/2항의 R²<0 은 origin-through 제약의 산물이 맞다**(intercept 주면 −1.04→+0.10).
   단, full 은 d_Cov가 이미 의사-절편이라 intercept 추가해도 R² 거의 불변(0.143→0.144).

### (2) 거리 구간별 / 근거리 재적합
- 전역 weight로 본 구간별 R²: <40m −37, 40-80m −8 (좁은 구간 제한범위 artifact, pred가 ~93m 상수라 음수).
- **근거리(<40m)만으로 재적합한 상한**: origin-through R²=0.038, intercept R²=**0.106** (RMSE~9m).
  → 근거리로 한정·재적합해도 R²는 0.04~0.11. 메트릭은 근거리에서도 거리를 약하게만 추적.

### (3) RMS 스케일 후 열 분포 (std/mean)
| 열 | std/mean | 판정 |
|---|---|---|
| d_RSRP | 0.855 | 건강한 분산 |
| d_PDP | 1.252 | 건강한 분산 |
| d_Cov | **0.102** | 거의 상수(포화) |

→ **스케일 문제 아님.** RSRP/PDP는 분산이 충분(눌리지 않음). 단지 d_phys와 상관이 약할 뿐.

## 최종 판정
- **버그/설정 문제 일부 사실**: origin-through가 ablation의 R²<0 공포를 만들었다(intercept로 해소).
- **그러나 천장은 물리 한계**: intercept·근거리 재적합 모두 R²≈0.10~0.14 에서 멈춤.
  산점도상 d_metric 이 d_phys 무관하게 ~90m 수평 구름 → 거리 변별력 자체가 약함.
- 결론: **1-BS·BS-side baseline 으로는 UE-UE 물리거리를 충분히 복원 못 함.**
  (BS-side 공분산 38m 포화, RSRP/PDP 단독 방향 모호성) → ver0.2 UE-side / multi-BS 근거.

## 실행
```bash
python3 P2G_FitDiag_2606v1.py
```
