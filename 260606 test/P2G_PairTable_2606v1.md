# P2G_PairTable_2606v1.py — 단계 3: 쌍 테이블(X·y) 생성 + 진단

## 목적
거리행렬(NxN, 단계2 산출)에서 상삼각(i<j) 쌍을 추출해
단계4(NNLS 가중치 학습)의 입력 `X`, `y`를 만들고, 각 거리 항이 물리거리와
어떤 관계인지 진단(산점도 + d_Cov 포화 분석)한다.

## 입력
- `P2G_DistTerms_Results/P2G_DistMat_*.npz` (최신본 자동 선택)
  - `D_rsrp`(dB), `D_pdp`(ns), `D_cov`(무차원), `pos`(N,2)

## 처리
1. `d_phys(i,j) = ||pos_i - pos_j||_2` (UE-UE Euclidean) → 학습 타깃 `y[m]`
2. 상삼각 추출 → `X=[d_RSRP, d_PDP, d_Cov]` (M,3), `M=N(N-1)/2`
3. 각 항 vs d_phys Spearman 상관
4. d_Cov 포화 분석
   - 비포화(d_Cov<1.40) 쌍 비율 및 그 d_phys 분포
   - 거리 구간별 중앙 d_Cov로 plateau 추정 → 유효 선형 범위 산정
   - 거리 구간별 비포화 비율
5. 산점도 4-패널 PNG, 쌍 테이블 NPZ 저장

## 출력 (`P2G_PairTable_Results/`)
- `P2G_PairTable_<stamp>.npz` : `X, y, pair_i, pair_j, col_names`
- `P2G_terms_vs_dphys_<stamp>.png` : 3항 산점도 + d_Cov 포화 분해
- `P2G_pairtable_report_<stamp>.log`

## 유효 선형 범위 정의 (주의)
d_Cov 의 실제 plateau 는 √2(1.414)가 아니라 ≈1.39 수준이라, 고정 √2 기준은
부적합. 따라서 **plateau(원거리 구간 중앙값)의 99% 에 중앙 d_Cov 가 최초 도달하는
거리**를 포화 시작점으로 정의한다.

## 핵심 결과 (20260606)
- M = 537,166 쌍 (N=1037)
- Spearman: d_RSRP 0.318 > d_PDP 0.241 > d_Cov 0.195
- d_Cov 비포화(<1.40) 65.5%, plateau≈1.39, 유효 선형 ~38m
  - 0-10m 97% / 10-20m 91% / 20-40m 82% / 40-80m 67% / 80m+ 61% 비포화
  - **해석**: d_Cov 는 근거리(~38m 이내) 분해능만 가지며 그 이상은 포화 →
    단계4 가중치 학습 시 d_Cov 는 근거리 항으로 작용.

## 실행
```bash
python3 P2G_PairTable_2606v1.py
```
경량(numpy/CPU). 한글 라벨은 Noto Sans CJK KR 사용.
