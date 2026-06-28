# P2A_Hybrid_Metric_Linearity_2606v2.py 설명 문서

## 목적
단일 기지국(RAN Twin, single-BS) 환경에서 **하이브리드 채널 메트릭**(공간 + 1D PDP)의
우수성을 증명하기 위해 다음을 동시에 수행/시각화한다.

1. **물리 거리 ↔ 메트릭 선형(Spearman) 상관관계** (Correlation Scatter)
2. **t-SNE 기반 지형 위상(Topology) 복원** (radius 색 매핑)
3. **GPU(2-GPU) 자동 가중치 탐색** (Golden Ratio α, β)

원 프롬프트(§1)는 거리 층화 '합성' 데이터 생성을 가정했으나, 사용자 요청에 따라
`251009_CCM_Collection` 의 **실측 RT npz** 를 사용한다.

## 입력 데이터
- `251009_CCM_Collection (8 GB)/P1A_RT_Results/Area1_7.5GHz_Rays_ALL_RXs.npz`
  - **LoS 포함** 레이트레이싱 결과 (1600 RX grid, 최대 400 path/RX)
  - 사용 필드: `power`, `tau`(지연), `phi_t_deg`(BS측 AoD 방위각), `rx_indices`, `los_nlos_flag`
- Area1 grid: x=linspace(-136.138, 58.862, 40), y=linspace(-117.667, 77.333, 40), z=1.5,
  기지국(TX) = (-51.561, -21.794, 19)

## 핵심 설계 결정 (탐색으로 확정)
1. **rx_indices 는 1-based(1..1600)** → 위치 복원은 `(idx-1)` 로 grid 환원해야 정확
   (AoD-방위각 상관 0.80→0.82 로 개선 확인).
2. **제공된 1024×1024 R_BS(P1F CCM) 의 BW 는 사용하지 않음**: 물리거리와 음의 상관
   (ρ≈−0.16). 원인 = (a) trace(=전력)가 BW 를 지배, (b) 데이터가 NLoS 위주로 공분산이
   확산(diffuse)되어 위치 변별력 약함.
3. **공간 공분산은 레이 AoD+power 로 재구성**하고, **trace=1 로 정규화**하여 BW 가
   '각도 형상' 차이만 측정하도록 함(정규화 안 하면 음의 상관 재발).
4. **두 가지 공간 메트릭을 모두 계산·비교**한다.
   - (A) **Bures-Wasserstein**: 가상 ULA(M=32) 공간 공분산의 저랭크 BW.
   - (B) **AoD-Wasserstein**: 방위각 APS(48 bin)의 1D Wasserstein(CDF-L1).
5. **시간 메트릭 = PDP(32 bin) 1D Wasserstein(CDF-L1)**. 프롬프트의
   `Scale=√(Tr_i·Tr_j)` 가중은 실측 전력 동적범위가 ~9 자릿수라 min-max 후 메트릭을
   붕괴시켜 **제외**(근거리 쌍이 전부 지배). 코드 주석에 명시.
6. **거리 층화 샘플링**: 반경을 10구간으로 나눠 bin 당 최대 80개(부족 시 전부) → N≈697.

## 저랭크 Bures-Wasserstein (검증된 고속 공식)
`R_i ≈ U_i Λ_i U_iᴴ`, `B_i = U_i Λ_i^{1/2}` (M×r, r=16) 일 때
```
W_BW²(i,j) = Tr(R_i) + Tr(R_j) − 2·‖B_iᴴ B_j‖_*      ( ‖·‖_* = nuclear norm )
```
- `‖B_iᴴ B_j‖_*` = r×r 행렬 특이값 합 → 1037² 쌍을 배치 SVD 로 고속 계산(for-loop 없음).
- trace=1 정규화이므로 `W_BW² = 2 − 2·‖B_iᴴ B_j‖_*`.
- 정확 BW 대비 상대오차: r=16 에서 ~5e-6.

## 수치 안정화
- 고유/특이값 `clamp(min=1e-12)`: 완전한 0 → cuSOLVER SVD 비수렴/NaN 방지(노이즈 바닥).
- 축퇴(중복 고유값) 행렬은 `torch.linalg.eigh` 가 비수렴 → **`torch.svd_lowrank`** 사용.
- 결과 행렬 `torch.nan_to_num` 가드.
- **Spearman 은 GPU 내 이중 argsort 랭크 변환 + Pearson** (scipy CPU 병목 회피).

## 멀티-GPU (RTX 5090 × 2) 활용
- `torch.cuda.device_count()` 로 GPU 감지, 단일 GPU/CPU 자동 폴백.
- 공분산 재구성·인수분해: N 개 UE 를 GPU 들에 분배.
- pairwise(W_BW²/W_pdp/W_aps): **행(row) 청크를 GPU 들에 분배** → 비동기 커널로 동시 실행
  후 `synchronize` + concat. 컴팩트 텐서(B, CDF)는 각 GPU 에 복제.

## 파이프라인 단계
1. `load_rays()` : npz 로딩 + (idx-1) 위치 복원 + path≥20 필터
2. `stratified_indices()` : 거리 10구간 층화 샘플링
3. `build_pdp_aps()` : PDP(32)·AoD-APS(48) 분포
4. `build_cov_factors()` : 공간 공분산 재구성 + trace 정규화 + 저랭크 인수분해 (2-GPU)
5. `pairwise_all()` : W_BW²·W_pdp·W_aps (2-GPU 샤딩)
6. `minmax_offdiag()` : 각 행렬 0~1 독립 정규화
7. `spearman_search()` : α 0~1 (100) 스캔, GPU Spearman → Golden Ratio (방식별)
8. `figure_2x4()` ×2 + `figure_compare()` : 시각화
9. 결과 저장 (npz/csv, 타임스탬프)

## 출력물 (P2A_Hybrid_Metric_Results/)
- `P2A_2x4_BW_<stamp>.png` : Bures-Wasserstein 방식 2×4 (상관 scatter + t-SNE)
- `P2A_2x4_AoDW_<stamp>.png` : AoD-Wasserstein 방식 2×4
- `P2A_compare_<stamp>.png` : 두 방식 α-Spearman 곡선 + 핵심 ρ 막대
- `P2A_result_<stamp>.npz`, `P2A_summary_<stamp>.csv`

## 대표 결과 (N=697, 거리 층화)
| 방식 | 공간 단독 | 시간 단독 | **하이브리드** | α_opt |
|---|---|---|---|---|
| Bures-Wasserstein | 0.340 | 0.274 | **0.387** | 0.86 |
| AoD-Wasserstein | 0.159 | 0.274 | 0.277 | 0.22 |

- **두 방식 모두 하이브리드 > 단일 메트릭** → 하이브리드 메트릭의 우수성 입증.
- 절대 상관값(≈0.39)은 데이터가 NLoS 위주(LoS path 1.7%)인 실측 환경의 한계.
  (이상적 합성 데이터의 '완벽한 직선' 과는 차이가 있음 — 문서/주석에 명시.)

## 실행 환경
- Python 3.10.12 / 서버 dclcom61(deepgadget), RTX 5090 32GB ×2 (sm_120, CUDA 12.8)
- torch 2.11.0+cu128, numpy 2.2.6, scipy 1.15.3, scikit-learn 1.7.2, matplotlib 3.10.9

## 실행 방법
```bash
cd "260604 metric 도출"
python3 P2A_Hybrid_Metric_Linearity_2606v2.py
```
