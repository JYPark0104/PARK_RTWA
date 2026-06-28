# P2G_AddUECov_2606v1.py — ver0.2: UE-side 공분산(R_UE) 추가 효과

## 목적
ver0.1(d_RSRP/d_PDP/d_Cov_BS) R²≈0.144 천장을, UE-side 공분산(R_UE, 16x16)
4번째 항으로 더하면 올릴 수 있는지 직접 측정.

## 방법
- R_UE(16x16) → 기존 d_Cov 절차(Ã=A/tr(A)+εI, Bures-Wasserstein, 풀랭크 r=16)로
  d_Cov_UE NxN 산출 (GPU 파이프라인 재사용).
- 4항 결합 후 절편 허용 OLS로 조합별 R²/RMSE/corr 비교.

## 입력 / 출력
- 입력: `P1F_*/*.npz`(R_UE), `P2G_DistMat_*.npz`(rx_idx/pos), `P2G_PairTable_*.npz`(X,y)
- 출력(`P2G_AddUECov_Results/`): `P2G_DcovUE_*.npz`, `P2G_adduecov_*.png`, `*.log`

## 결과 (20260606) — UE-side 는 도움 안 됨
- d_Cov_UE 와 d_phys 상관 = **0.053** (거의 무상관)
- 조합별 R²:
  | 조합 | R² | corr |
  |---|---|---|
  | ver0.1 (RSRP+PDP+Cov_BS) | 0.1443 | 0.380 |
  | Cov_UE only | 0.0028 | 0.053 |
  | Cov_BS+Cov_UE | 0.0590 | 0.243 |
  | RSRP+PDP+Cov_UE | 0.1004 | 0.317 |
  | **FULL4 (+Cov_UE)** | **0.1444** | 0.380 |
- **ver0.1 → FULL4: R² +0.0000** (변화 없음). d_Cov_UE 제거해도 −0.0000.

## 해석
- R_UE 는 UE 안테나의 **수신 도래각 분포(국소 산란)** 정보 → BS-UE *거리*와 무관.
  가까운 UE도 국소 산란이 다르고, 먼 UE도 비슷할 수 있어 corr≈0.05.
- 따라서 단일 BS 에 UE-side 를 더해도 **거리 변별력은 안 오름**.
- 진짜 지렛대는 multi-BS(삼각측량) — 단일 링크 특징의 추가가 아님.

## 결론
> ver0.2(UE-side 추가)는 천장(R²≈0.144)을 못 넘는다. 단일 BS 정보로는
> UE-UE 물리거리 복원이 본질적으로 제한되며, multi-BS 가 필요함을 재확인.

## 실행
```bash
python3 P2G_AddUECov_2606v1.py   # GPU(H100x2), full-permission
```
주의: pairwise_nuclear 배치는 rblk=64/sub=128 (큰 배치는 cusolver 배치 한계로 실패).
