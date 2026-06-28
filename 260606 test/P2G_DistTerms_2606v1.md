# P2G_DistTerms_2606v1.py

## 목적
계획서 **단계 2**. 거리 항 3함수(d_RSRP / d_PDP / d_Cov) 구현 + 단위테스트, 그리고
점 테이블 -> 3개 거리행렬(NxN) 산출. 모든 계산은 모든 GPU(H100x2) 사용.

## 거리 항 정의 (단위/가정)
| 항 | 정의 | 단위 |
|---|---|---|
| d_RSRP | `|rsrp_i - rsrp_j|`, rsrp=10log10(Σpower) | dB (L1) |
| d_PDP | 1D Wasserstein W1, 공통 [0,tau_max] 균일격자 CDF-L1 | ns |
| d_Cov | `sqrt(max(tr_i+tr_j-2‖B_iᴴB_j‖_*, 0))` (Bures-Wasserstein) | 무차원 |

### d_PDP 설계 (동적범위 대응)
- 점질량 직접 W1(scipy)은 537k 쌍에 비현실적 -> **공통 미세 균일격자 CDF-L1**(GPU 배치).
- 공통 τ축 `[0, global tau_max=4226ns]` (점별 tau_max 금지), 정규화는 유효경로(power>0)만.
- **clip 금지**: tau_max clip 시 잘린 질량이 W1 을 크게 왜곡(scipy 대비 rel>100%).
- 단위 **ns**(s 면 ~1e-7 로 정밀도 손실) -> weight 는 m/ns.
- 격자 L=1024, Δτ=4.127ns. 근거: 점별 지연확산 중앙 423ns -> 약 103 bin 으로 분해.

### d_Cov 설계 (성능/안정)
- `‖G‖_* = Σσ(G) = Σ sqrt(λ(GᴴG))`, G=B_iᴴB_j (r x r). 정확성은 BW_Approx_Check,
  r=96 충분성은 dCov_RankCheck 에서 검증.
- 성능: 배치 `svdvals` 는 334/s 로 느림 -> `eigvalsh(GᴴG)` 106700/s (~300x).
- 안정: 축퇴(중복 고유값) 시 eigvalsh 비수렴 -> **float64 + 1e-12 jitter**.
- 메모리: 행블록(rblk=64)×열청크(sub=128)로 배치 제한(eigvalsh 워크스페이스 OOM 회피).
- 저랭크 floor: 자기거리 d_Cov(i,i)=√(2·tail_i)≈0.04 (≠0, 메트릭은 i<j 만 사용).

## 단위테스트 (가까운 쌍 기준 + 공통 점검)
- d_RSRP: 단위 dB, 대칭, 자기거리=0, NaN/inf=0.
- d_PDP: 격자 CDF-L1 vs scipy 점질량 W1, **W1 크기 분위수별** 상대오차.
- d_Cov: 대칭, 클램프>=0, 자기거리=√(2tail), NaN/inf=0.

## 실행 결과 (2026-06-06, stamp 20260606_084040)
- **d_PDP 단위테스트**(400쌍, vs scipy 점질량):
  - 전체 rel 중앙 0.93%, abs 최대 3.33ns(≈Δτ, binning 바닥)
  - 가까운 쌍(W1<4.9ns) rel 중앙 8.4%/최대 96% (분모 작아 큼, 절대오차는 ~Δτ로 무해)
  - 먼 쌍(W1 50-100%) rel 중앙 0.47%
- **3개 거리행렬**(1037x1037) on 2GPU **7.3s** (d_Cov 100만쌍 BW 7.1s)
  - d_RSRP[dB] off-diag 3.8e-5 ~ 77.07 (med 16.6)
  - d_PDP[ns]  off-diag 1.8e-4 ~ 1290.5 (med 92.4)
  - d_Cov[-]   off-diag 0.094 ~ 1.414 (med 1.39, sqrt2≈1.414 부근에 집중)
- 출력: `P2G_DistTerms_Results/P2G_DistMat_{stamp}.npz` (D_rsrp/D_pdp/D_cov, 단위 메타 포함)

## 주의 / 관찰
- d_Cov 가 대부분 √2(=최대) 부근에 집중 -> BS-side 공분산은 위치가 달라지면 빠르게
  거의 직교. 단계 4 fit 에서 d_Cov 의 유효 동적범위(가까운 쌍 위주)를 확인할 것.
- 발견·수정 버그: PDP 정규화에서 분모에 EPS(1e-12) 더하면 선형 power(~1e-10~1e-20)를
  오염 -> CDF 붕괴. EPS 제거(0행만 보호)로 해결. (scipy 대조로 검출)

## 입력 (읽기 전용) : P2G_PointTable_Results/P2G_PointTable_*.npz
## 의존 : P2G_gpu_utils_2606v1 (get_devices/warmup_linalg/pairwise_nuclear)
## 실행 환경 : Python 3.10.12 / torch 2.12.0+cu130 / numpy 2.2.6 / scipy 1.15.3, H100x2
## 실행
```bash
cd "260606 test"
python3 P2G_DistTerms_2606v1.py   # full-permission(GPU) 필요
```
