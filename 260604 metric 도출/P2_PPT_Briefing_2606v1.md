# 6G Digital Twin 채널거리 Metric — 발표 브리핑 (PPT 생성용)

> **이 파일 하나 + 아래 PNG/CSV들을 Claude에 첨부하면 PPT를 만들 수 있습니다.**
> 핵심 메시지: **"APS는 Bures, PDP는 Wasserstein으로 재고, trace 정규화 후
> 0.77:0.23로 결합했을 때 Spearman ρ = 0.600 으로 가장 좋았다."**

---

## 0. 데이터
- 소스: `channel_data_260531_GHM_Twin_v0_1.npz` (GHM Twin v0.1, RT 시뮬레이션)
- 구성: **3개 BS(TX) × ~5071 RX**, SISO, 링크별 ray 리스트(지연 τ·전력·방위 AoA)
- 분석 표현: 각 RX → BS별 **PDP(지연분포)** + **APS(방위분포)**
- 목표: **물리거리 ↔ 채널거리** 의 단조 관계(Spearman ρ) 극대화 → 보간·군집·섹터화에 활용

---

## 1. 핵심 Metric (P2D) — 최종 채택
**채널거리 = γ·(APS Bures 거리) + δ·(PDP Wasserstein 거리)**, 3개 BS 합산.

- **APS 항 (각도)**: APS를 **trace=1 정규화**(전력 제거, 모양만) 후
  대각 Bures(=Hellinger) `Σ(√a₁−√a₂)²`
- **PDP 항 (지연)**: PDP 정규화 후 1D Wasserstein `Σ|CDF₁−CDF₂|`
- 두 항 각각 min-max 정규화 → `D = (1−w)·Bures + w·Wass`, **w 격자탐색(0~1, 101점)**
- 최적 **γ:δ = 0.77:0.23**

### ★ 결과 (Spearman ρ vs 물리거리, n_pair=1498)
| 구분 | 항 | ρ |
|---|---|---|
| (A) raw (전력 미정규화) | Bures(raw) | −0.157 |
| | PDP(raw, √Tr 스케일) | −0.146 |
| | **결합(raw)** | **−0.161** ← 전력 지배 함정 |
| (B) trace=1 정규화 | Bures-shape | +0.568 |
| | Wasserstein | +0.383 |
| | **결합 (0.77:0.23)** | **+0.600** ← 최종 최고 |

**스토리 포인트**: raw로 쓰면 총전력이 모양을 덮어 ρ가 음수(−0.16)로 무너짐
→ **trace=1 정규화가 핵심 해법**, 결합으로 +0.600 달성.

---

## 2. 조합 비교 근거 (P2B) — "왜 이 구성인가"
| Metric | ρ | 비고 |
|---|---|---|
| PDP-Wass (단일) | 0.217 | 지연만 |
| AoA-Wass TX0 (단일) | 0.467 | 단일 BS 방위 |
| **Multi-BS AoA L1 가중합** | **0.539** | w=(0.40,0.20,0.40) — L2보다 우수 |
| Multi-BS AoA L2 유클리드 | 0.495 | |
| Multi-BS RSRP L2 | 0.198 | 전력은 약함 |
| HYBRID L1+RSRP | 0.547 | α_AoA=0.88 |
- 결론: **방위(AoA)가 주력**, 전력(RSRP)은 약함. 조합은 **L1 가중합 > L2**.
- → P2D에서 Bures(각도) 위주 + PDP 보조 구성으로 이어짐(0.600).

---

## 3. 응용 ① 방위 섹터화 (P2F) — "채널이 위치를 복원"
- 채널 지배 방위(APS 원형평균)로 K-means → **Silhouette 0.308 (k=3)** (P2E 대비 ~3배 선명)
- 채널 방위 vs 기하 방위 **정렬도 R_align = 0.72** (오프셋 174°≈180°, 도래각 반전: 물리적 타당)
- 공정 비교(둘 다 3-BS K-means): **ARI = 0.383, 색 일치율 71.9%**
- → 채널만으로 지리 방위/섹터 구조를 잘 복원.

## 4. 응용 ② 채널 군집 (P2E) — 사용자 그룹핑
- P2D 채널거리로 K-medoids/Spectral/Agglomerative 군집
- Spectral 최적 k=5 (Silhouette 0.117), 물리지도상 공간적으로 일관된 군집
- (연속 채널장 특성상 silhouette은 낮지만 공간 일관성 확보)

---

## 5. 추천 슬라이드 흐름
1. **배경/문제** — 물리거리 ↔ 채널 유사도 관계가 디지털 트윈/보간/군집에 필요
2. **데이터** — GHM Twin, 3-BS, PDP+APS 표현
3. **Metric 정의** — APS=Bures, PDP=Wasserstein (수식)
4. **함정** — raw 전력 지배 → ρ=−0.16 (P2D scatter 상단)
5. **해법** — trace=1 정규화 + 결합(0.77:0.23) → **ρ=+0.600** (P2D PRESENT)
6. **조합 비교 표** — L1/L2/Bures/PDP (P2B compare)
7. **응용 1: 섹터화** — R=0.72, 채널→위치 복원 (P2F)
8. **응용 2: 군집** — 사용자 그룹핑 (P2E)
9. **결론** — 최적 metric/조합 + 활용 방향

---

## 6. Claude에 첨부할 파일 목록
### 텍스트(내용 이해용)
- `P2_PPT_Briefing_2606v1.md` (이 파일)
- `P2D_BuresPDP_ChannelDist_2606v1.md`
- `P2B_GHM_Metric_Compare_2606v1.md`
- `P2F_AngularSectorization_2606v1.md`
- `P2E_ChannelClustering_2606v1.md`

### 그림(슬라이드 배치용 PNG)
- `P2D_BuresPDP_ChannelDist_Results/P2D_BuresPDP_PRESENT_20260604_131946.png` ← 메인(ρ=0.600)
- `P2D_BuresPDP_ChannelDist_Results/P2D_BuresPDP_scatter_20260604_131946.png` ← raw vs trace 2×3
- `P2B_GHM_Metric_Results/P2B_GHM_compare_20260604_120731.png` ← 조합 비교
- `P2F_AngularSectorization_Results/P2F_sectorization_20260604_134338.png` ← 섹터화
- `P2E_ChannelClustering_Results/P2E_clustering_20260604_132419.png` ← 군집

### 수치(표 인용용 CSV)
- `P2D_BuresPDP_ChannelDist_Results/P2D_BuresPDP_corr_20260604_131946.csv`
- `P2B_GHM_Metric_Results/P2B_GHM_compare_20260604_120731.csv`
- `P2F_AngularSectorization_Results/P2F_sectorization_20260604_134338.csv`

---

## 7. Claude용 프롬프트 (복붙)
> 첨부한 `P2_PPT_Briefing_2606v1.md`와 각 .md/PNG/CSV를 바탕으로 6G Digital Twin
> 채널거리 metric 발표 PPT(9슬라이드)를 만들어줘. 핵심 메시지는 "APS는 Bures,
> PDP는 Wasserstein으로 재고 trace 정규화 후 0.77:0.23로 결합했을 때 Spearman
> ρ=0.600으로 가장 좋았다". 브리핑의 슬라이드 흐름(5번 항목)을 따르고, 각 슬라이드에
> 해당 PNG를 배치하고 CSV 수치를 표로 넣어줘. 톤은 학술 발표용, 한국어.
