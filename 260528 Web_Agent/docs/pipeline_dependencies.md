# Pipeline Dependencies — 19 metric × 13 stage × derived 매핑

## Stage → 출력 디렉토리

| stage | 소스 폴더 (REPO_ROOT 기준)          | 출력 디렉토리                   |
| ----- | ----------------------------------- | ------------------------------- |
| P1A   | `251218 E_MIMO_BM` (fork 사용)      | `P1A_RT_Results/`               |
| P1B   | `251218 E_MIMO_BM`                  | `P1B_Valid_Results/`            |
| P1C   | `250924 python`                     | `P1C_AE_OFDM_Ch_Results/`       |
| P1D   | `251004_Ch_Separ_(CorrRician)`      | `P1D_Separability_Results/`     |
| P1F   | `251218 E_MIMO_BM`                  | `P1F_Marginal_CCM_Results/`     |
| P1G   | `251218 E_MIMO_BM`                  | `P1G_CouplingMat_Results/`      |
| P1H   | `251218 E_MIMO_BM`                  | `P1H_MeanCh_Results/`           |
| P1I   | `251218 E_MIMO_BM`                  | `P1I_Weichsel_Chunk_Results/`   |
| P1J   | `251020 BM QIE (Weichselberger)`    | `P1J_Capacity_Results/`         |
| P1L   | `251020 BM QIE (Weichselberger)`    | `P1L_BM_Greedy_Results/`        |
| P1M   | `251020 BM QIE (Weichselberger)`    | `P1M_QIE_Results/`              |
| P1N   | `251020 BM QIE (Weichselberger)`    | `P1N_PADP_DFT_Results/`         |
| P1O   | `251028 BM_SU_MIMO_Uplink`          | `P1O_Uplink_BM_Results/`        |
| P1P   | `251218 E_MIMO_BM`                  | `P1P_SWOMP_Results/`            |
| P1Q   | `251218 E_MIMO_BM`                  | `P1Q_BeamPattern_Results/`      |

Coverage Map 은 25* 가 아닌 Sionna `RadioMapSolver` 를 직접 호출하는 별도 어댑터  
→ 출력: `Coverage_Map_Results/`

## 19 Metric → Stage / Derived 매핑

| metric ID          | 라벨                           | 요구 stages (위상정렬 후)                 | derived | coverage_map | needs_rx |
| ------------------ | ------------------------------ | ----------------------------------------- | ------- | ------------ | -------- |
| `coverage_map`     | Coverage Map (TX 광역 heatmap) | (없음, RadioMapSolver 직접)               | -       | ✓            | ✗        |
| `ray_dump`         | Raw Ray dump                   | P1A → P1B                                 | -       | -            | ✓        |
| `rsrp`             | RSRP (RX별 dB)                 | P1A → P1B → derived(rsrp)                 | ✓       | -            | ✓        |
| `pdp`              | PDP (지연 프로파일)            | P1A → P1B → derived(pdp)                  | ✓       | -            | ✓        |
| `padp`             | PADP (지연·각도 프로파일)      | P1A → P1B → derived(padp)                 | ✓       | -            | ✓        |
| `ray_stats`        | Ray Stats (DS / K / AS)        | P1A → P1B → derived(ray_stats)            | ✓       | -            | ✓        |
| `ae_ofdm_channel`  | AE OFDM Channel                | P1A → P1B → P1C                           | -       | -            | ✓        |
| `marginal_ccm`     | Marginal CCM (R_BS, R_UE)      | P1A → P1B → P1F                           | -       | -            | ✓        |
| `coupling_matrix`  | Coupling Matrix (Ω, U_tx, U_rx)| P1A → P1B → P1F → P1G                     | -       | -            | ✓        |
| `mean_channel`     | Mean Channel H̄                 | P1A → P1B → P1H                           | -       | -            | ✓        |
| `separability`     | Channel Separability           | P1A → P1B → P1D                           | -       | -            | ✓        |
| `su_capacity`      | SU-MIMO Capacity               | P1A → P1B → P1F → P1G → P1H → P1I → P1J  | -       | -            | ✓        |
| `greedy_beams`     | Greedy beam selection          | P1A → P1B → P1F → P1G → P1H → P1I → P1L  | -       | -            | ✓        |
| `swomp_beams`      | SWOMP beam selection           | P1A → P1B → P1F → P1G → P1H → P1I → P1P  | -       | -            | ✓        |
| `uplink_beams`     | Uplink BM (P1O)                | P1A → P1B → derived(padp) → P1O           | (padp)  | -            | ✓        |
| `qie_clustering`   | QIE Clustering                 | P1A → P1B → P1F → P1G → P1H → P1I → P1M  | -       | -            | ✓        |
| `padp_dft_alignment` | PADP-DFT alignment           | P1A → P1B → derived(padp) → P1N           | (padp)  | -            | ✓        |
| `beam_pattern`     | Beam Pattern visualization     | P1A → P1B → P1F → P1G → P1H → P1I → P1P → P1Q | -   | -            | ✓        |
| `eigenbeam_pattern`| Eigenbeam pattern              | P1A → P1B → P1F → P1G → P1H → P1I → P1Q  | -       | -            | ✓        |

> Coverage Map 만 단독 선택 가능 (RX 불필요). 그 외 metric 은 RX placement 필수.

## 위상정렬 알고리즘

`metric_catalog.resolve_metrics()` 는:

1. 선택된 metric 들의 stage 의존성 합집합 산출
2. 25* 의 자연 순서 `P1A → P1B → P1C → P1D → P1F → P1G → P1H → P1I → P1J → P1L → P1M → P1N → P1O → P1P → P1Q` 에 따라 stages_ordered 산출
3. derived metric 들은 P1B 완료 후 병렬로 실행 가능
4. Coverage Map 은 stages 와 독립 — RT pipeline 과 별개로 진행

## 7 종 프리셋 (frontend / backend 공유)

| preset id                | metrics 묶음 |
| ------------------------ | ------------ |
| `coverage_only`          | `coverage_map` |
| `rt_only`                | `ray_dump`, `rsrp`, `pdp`, `padp`, `ray_stats` |
| `rt_with_ccm`            | RT + `marginal_ccm`, `coupling_matrix`, `mean_channel` |
| `rt_ccm_swomp`           | rt_with_ccm + `su_capacity`, `swomp_beams`, `beam_pattern` |
| `rt_ccm_greedy`          | rt_with_ccm + `su_capacity`, `greedy_beams`, `beam_pattern` |
| `ray_visualization`      | `ray_dump`, `rsrp`, `padp`, `coverage_map` |
| `bm_full`                | RT + CCM + `su_capacity` + 모든 BM (greedy/swomp/uplink/qie/padp_dft) + `beam_pattern`, `eigenbeam_pattern` |

## NPZ 표준 (251218 E_MIMO_BM 단일)

각 stage 의 NPZ key/shape/dtype 은 `backend/npz_validator.py` 의 `SCHEMAS` 에 명시.
주요 stage 의 핵심 키:

| stage | 핵심 key                                                                                |
| ----- | --------------------------------------------------------------------------------------- |
| P1A   | `tau / power / theta_r_deg / theta_t_deg / phi_r_deg / phi_t_deg / counts / source_path_idx / los_nlos_flag / area_index / frequency_ghz / num_rx / rx_indices` (13 keys) |
| P1B   | P1A 동일 + (옵션) `valid_rx_mask`                                                       |
| P1F   | `R_BS (n_t, n_t) complex64`, `R_UE (n_r, n_r) complex64`, `metadata`, `validation`      |
| P1G   | `Omega (n_r, n_t) float32`, `U_tx (n_t, n_t) complex64`, `U_rx (n_r, n_r) complex64`    |
| P1H   | `H_bar (n_r, n_t) complex64`                                                             |
| P1I   | `ue_indices`, `P1F_R_BS/R_UE`, `P1G_U_BS/U_UE/Lambda_BS/Lambda_UE/Omega`, `P1H_H_mean`, `*_metadata` (chunked) |

`backend/tests/e2e_pipeline_test.py --stage validate` 가 모든 NPZ 의 key/dtype/shape 을 자동 검증.
