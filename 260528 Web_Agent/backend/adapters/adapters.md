# backend/adapters/

25\* 폴더의 13개 P1\* 모듈 + Coverage Map 어댑터.

## 구조

```
adapters/
├── __init__.py            # public re-export
├── base_adapter.py        # STAGE_REGISTRY, BaseAdapter, session_workdir, run_stages
├── p1_adapters.py         # run_p1a / run_p1b / ... / run_p1q + STAGE_RUNNERS
└── coverage_map_adapter.py  # Sionna RadioMapSolver 어댑터 (25* 외)
```

## REPO_ROOT

`base_adapter.py`에서 Web Agent 위치(`…/260528 Web_Agent`) 기준으로 `twin_minji` 루트를 자동 해석한다.  
호스트별 `/home/dclcom61/twin_minji` 하드코딩은 사용하지 않는다 (2026-06-04, dclserver78).

## STAGE_REGISTRY

각 stage의 위치:

| Stage | 폴더 | 파일 | Config | 결과 폴더 |
|-------|------|------|--------|----------|
| P1A | 251218 E_MIMO_BM | P1A_RT_to_Rays_2509v6.py (+ fork) | P1A_Config | P1A_RT_Results |
| P1B | 251218 E_MIMO_BM | P1B_Valid_RX_Filter_2509v1.py | P1B_Config | P1A_RT_Results (in-place) |
| P1C | 250924 python | P1C_Rays_to_AE_OFDM_Ch_2509v1.py | P1C_Config | P1C_AE_OFDM_Ch_Results |
| P1D | 251004_Ch_Separ_(CorrRician) | P1D_Rays_to_AE_Separability_2510v3.py | P1D_Config | P1D_Separability_Results |
| P1F | 251218 E_MIMO_BM | P1F_Rays_to_Marginal_CCM_2510v1.py | P1F_Config | P1F_Marginal_CCM_Results |
| P1G | 251218 E_MIMO_BM | P1G_Rays_to_CouplingMat_2510v1.py | P1G_Config | P1G_CouplingMat_Results |
| P1H | 251218 E_MIMO_BM | P1H_Rays_to_MeanCh_2510v1.py | P1H_Config | P1H_MeanCh_Results |
| P1I | 251218 E_MIMO_BM | P1I_Weichsel_Chunk_2510v1.py | P1I_Config | P1I_Weichsel_Results |
| P1J | 251020 BM QIE (Weichselberger) | P1J_Weichsel_SU_MIMO_Capacity_2510v1.py | P1J_Config | P1J_Capacity_Results |
| P1L | 251020 BM QIE (Weichselberger) | P1L_BeamMgmt_SU_MIMO_2510v3.py | P1L_Config | P1L_BM_Greedy_Results |
| P1M | 251020 BM QIE (Weichselberger) | P1M_BM_QIE_2510v1.py | P1M_Config | P1M_QIE_Results |
| P1N | 251020 BM QIE (Weichselberger) | P1N_Analyze_PADP_DFT_2510v1.py | P1N_Config | P1N_PADP_DFT_Results |
| P1O | 251028 BM_SU_MIMO_Uplink | P1O_PADP_to_BM_2510v6.py | P1O_Config | P1O_Uplink_BM_Results |
| P1P | 251218 E_MIMO_BM | P1P_BM_SWOMP_2511v5.py | P1P_Config | P1P_SWOMP_Results |
| P1Q | 251218 E_MIMO_BM | P1Q_Selected_Beams_Pattern_Plot_v6.py | P1Q_Config | P1Q_BeamPattern_Results |

## 호출 패턴

```python
from backend.adapters.base_adapter import run_stages
from backend.adapters.p1_adapters import run_p1a

# 단일 stage:
run_p1a(
    session_dir="sessions/abc",
    scene_xml_path="sessions/abc/scene/scene.xml",
    area_configs={"area_1": {"tx_positions": [[-51, -21, 19]],
                              "rx_placement": {...}, "description": "..."}},
    area_indices=[1],
    frequency_ghz_list=[7.5],
    rt_overrides={"MAX_DEPTH": 5, "RT_SEED": 41},
)

# 전체 위상정렬 후 일괄:
plan = resolve_metrics(["swomp_beams"])  # metric_catalog
run_stages(
    session_dir="sessions/abc",
    stage_ids=plan.stages_ordered,
    overrides_per_stage={
        "P1A": {...},
        "P1F": {...},
    },
)
```

## fork 우선 import

`STAGE_REGISTRY` 의 `forked_module`이 설정되어 있으면 `backend/forked_25x/`에서
`<name>.py`를 먼저 찾고, 없으면 원본 25\*에서 import. 현재 P1A만 fork.

P1A fork는 추가로 `web_run_pipeline()` 함수를 제공하므로 BaseAdapter는 그것을 직접 호출.

## cwd 컨텍스트

각 P1\*는 결과 NPZ를 `cwd/{result_dirname}/`에 저장한다. BaseAdapter는
`session_workdir(session_dir)` context로 cwd를 변경한 뒤 실행 → 자동으로
세션 폴더 안에 결과 폴더 생성.

또한 `sys.path`에 25\* source folder를 임시 추가해서 형제 .py가 import 가능하도록 함.

## 알려진 한계

- 25\* 원본 모듈은 Config 인스턴스를 module-level 싱글톤으로 들고 있으므로,
  동일 process에서 같은 stage를 재호출하면 이전 override의 영향을 받을 수 있다.
  Job runner는 각 잡마다 새 subprocess를 띄우는 방식을 권장 (multiprocessing.Process).
- P1C ~ P1Q 의 일부 모듈은 25\* 원본의 INPUT_DIR/OUTPUT_DIR 키 이름이 다를 수 있어,
  실제 풀 파이프라인 E2E 검증 시 override 키 보정이 필요할 수 있다 (E2E TODO).
