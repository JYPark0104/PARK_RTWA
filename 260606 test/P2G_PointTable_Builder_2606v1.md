# P2G_PointTable_Builder_2606v1.py

## 목적
계획서 **단계 1**. P1A 레이(RSRP/PDP/위치) + P1F 실제 CCM(공분산)을 RX 인덱스로
조인해 **점 단위 표준 테이블**을 만들고, 위치 환원 sanity check 리포트를 남긴다.
단계 2~5 는 이 npz 파일 하나만 입력으로 받는다.

## 산출물
### (A) 점 테이블 npz : `P2G_PointTable_Results/P2G_PointTable_{stamp}.npz`
| 키 | shape | 설명 |
|---|---|---|
| `pos` | (N,2) f64 | UE 위치 (x,y) [m] — d_phys 타깃용 |
| `rsrp` | (N,) f64 | `10*log10(Σ power)` [dB] |
| `tau_raw` | (N,P) f32 | 경로별 도착시간 [s] (point-mass W1용) |
| `pwr_raw` | (N,P) f32 | 경로별 선형 전력 (0=무효 경로) |
| `npaths` | (N,) i32 | 유효 경로 수 |
| `cov_factor` | (N,M,r) c64 | R_BS 저랭크 인수 `B_i=U_iΛ_i^{1/2}` (Ã=A/tr(A)+εI) |
| `cov_trace` | (N,) f64 | `tr(Ã_i)`=1+εM (d_BW² 결합용) |
| `cov_tail` | (N,) f64 | 상위 r 제외 에너지 비율(진단) |
| `rx_idx` | (N,) i64 | 원본 RX 인덱스 |
| 메타 | - | area/freq/grid_n/grid_step/bs_xy/M/rank/eps/min_paths |

### (B) 점검 리포트 : `P2G_PointTable_Results/P2G_pointtable_report_{stamp}.log`
조인/shape/NaN/정렬 + 위치 환원 sanity check ①②③.

## 핵심 설계
- **조인**: P1F 1037 RX ⊂ P1A 1600 RX (누락 0). 교집합 N=1037 (필터 후).
- **위치 환원**: rx_idx 1-based → `f=idx-1`, `ix=f%40`, `iy=(f//40)%40`,
  `pos=(RX_X[ix], RX_Y[iy])`. grid step = 195/39 ≈ 5.0 m.
- **메모리 스트리밍**: R_BS(1024², 동시 ~8.7GB) 미적재. 파일 하나씩 열어 상위 r
  인수 `B_i`(1024×r)만 보관 → 약 0.27GB (r=32).
- **모든 GPU 사용**: trace 정규화 + 배치 eigh + 상위 r 추출을
  `P2G_gpu_utils.batched_topr_eigh_factor`로 위임, H100×2 샤딩. (작업 지침 준수)

## R_BS = BS(TX)-side 확인
metadata `n_t=1024 == R_BS.shape[0]`, `n_r=16 == R_UE.shape[0]` 일치 →
R_BS 는 TX(=BS, 1024안테나) marginal. (P1F 생성 스크립트 주석과 일치)

## 저랭크 r 확정 = 96
`P2G_dCov_RankCheck_2606v1.py` 수렴 측정으로 확정. r=32 는 가까운 쌍 d_Cov 최대오차
26% 로 부적합 → **r=96** (최대 0.91%<1%, p95 0.29%, 중앙 0.24%). RANK=96 으로 재빌드.

## 실행 결과 (2026-06-06, 최종 stamp 20260606_073907, r=96)
- N=1037, npaths 전부 400 (max paths)
- 저랭크 r=96 꼬리 에너지: mean 9.29e-4, max 1.02e-3 (≈99.9% 보존)
- **GPU eigh 30.9s** (H100×2), 점 테이블 773.5 MB
- sanity check (r=32/r=96 동일)
  - ① BS 거리 0.97~148.4 m (중앙 69 m) / 대각선 275.8 m → 타당
  - ② 최근접 UE-UE 중앙값 5.000 m = grid step (1036점 ~5m, 1점 20m 고립)
  - ③ ix+1→X +5m, iy+1→Y +5m → **축 뒤바뀜 없음**

## 알려진 이슈/주의
- d_Cov rel 최대오차는 r 무관 약 0.6% floor 존재(εI 바닥+최근접쌍 민감도). r=96 은
  max<1% 의 knee. 더 높은 r 은 비용 대비 이득 미미.
- GPU 접근이 막힌 샌드박스에서는 실행 불가 → full-permission 으로 실행해야 함.

## 입력 (읽기 전용)
- `251009_CCM_Collection (8 GB)/P1A_RT_Results/Area1_7.5GHz_Rays_ALL_RXs.npz`
- `251009_CCM_Collection (8 GB)/P1F_Marginal_CCM_Results/*.npz` (R_BS)

## 실행 환경
- Python 3.10.12 / torch 2.12.0+cu130 / numpy 2.2.6 / scipy 1.15.3
- 서버 dclserver78, NVIDIA H100 NVL 95GB × 2

## 실행
```bash
cd "260606 test"
python3 P2G_PointTable_Builder_2606v1.py   # full-permission(GPU) 필요
```
