# P2G_BW_Approx_Check_2606v1.py

## 목적
계획서 **단계 2-0** — 채널 메트릭의 공분산 항 `d_Cov`(Bures-Wasserstein)를
대규모(1024x1024 R_BS)로 확장하기 전에, "저랭크 인수분해 + nuclear-norm" 근사가
**full Bures와 실제로 일치하는지**를 작은 실데이터(R_UE, 16x16)로 먼저 검증하는 스니펫.

## 배경 (왜 필요한가)
공분산 항은 Bures-Wasserstein 거리로 정의된다.

```
d_BW^2(A,B) = tr(A) + tr(B) - 2 F(A,B),   F(A,B) = tr[(A^{1/2} B A^{1/2})^{1/2}]
```

여기서 fidelity 항 `F`를 어떻게 계산하느냐가 비용/정확도를 좌우한다.
53.7만 쌍 x 1024x1024 에 sqrtm 직접 적용은 비현실적이므로 저랭크 인수분해
`B_i = U_i Λ_i^{1/2}` 후 `F ≈ ||B_i^H B_j||_*` (P2A 방식)로 근사하려 한다.
이 근사가 맞는지부터 확인하는 것이 본 스니펫의 목적이다.

## 비교한 세 가지 fidelity
| 표기 | 정의 | 의미 |
|---|---|---|
| `F_full` | `tr[(A^{1/2} B A^{1/2})^{1/2}]` (eigh 기반) | 기준값(정의) |
| `F_sym` | `‖A^{1/2} B^{1/2}‖_*` (각 행렬 대칭 제곱근의 곱) | 이론상 정답 형태 |
| `F_fac` | `‖B_i^H B_j‖_*`, `B_i = U_i Λ_i^{1/2}` | P2A 방식(저랭크 확장 대상) |

안정화: `Ã = A/tr(A) + εI` (ε=1e-6).

## 검증 결과 (요약)
- **full-rank 3-way 일치**: `F_full = F_sym = F_fac`, 최대 상대오차 ≈ `4.5e-11` → PASS
- **이유**: full-rank에서 `U`는 정방 unitary → `S_i S_j = U_i (B_i^H B_j) U_j^H`.
  unitary 곱은 특이값을 바꾸지 않으므로 `‖B_i^H B_j‖_* = ‖S_i S_j‖_*`.
  즉 P2A의 `2 - 2‖B_i^H B_j‖_*` 는 **버그가 아니라 정확한 식**(trace=1 정규화 시).
- **저랭크 절사 수렴** (대표 쌍 0-1, F_full=0.913725):

  | rank r | F_fac(r) | rel_err |
  |---|---|---|
  | 2 | 0.878120 | 3.90e-02 |
  | 4 | 0.911748 | 2.16e-03 |
  | 8 | 0.913697 | 3.11e-05 |
  | 16 | 0.913725 | 7.87e-12 |

  → r 증가에 따라 단조 수렴. 스트리밍 저랭크 인수분해(상위 r만 보관)가 안전.

## 결론 / 다음 단계 영향
1. `d_Cov`는 `F_sym`(=`F_fac` 동치) 형태로 계산하면 된다. 저랭크 절사로 안전하게 확장 가능.
2. 1024x1024 R_BS는 점별 eigh 상위 r 인수만 **스트리밍 보관**(8.7GB→약 0.14GB) 후
   `‖B_i^H B_j‖_*`로 pairwise 계산(P2A `pairwise_all` 패턴).
3. r 선택: R_UE(M=16)에서 r=8이 rel~3e-5. R_BS(M=1024)는 r=16~32 권장, 본 계산서로 재검증 가능.

## 입력 데이터 (읽기 전용)
- `251009_CCM_Collection (8 GB)/P1F_Marginal_CCM_Results/*.npz` (키 `R_UE` 16x16)

## 실행 환경
- Python 3.10.12 / numpy 2.2.6 / scipy 1.15.3
- 서버: dclserver78 (twin_minji workspace)

## 실행
```bash
cd "260606 test"
python3 P2G_BW_Approx_Check_2606v1.py
```
