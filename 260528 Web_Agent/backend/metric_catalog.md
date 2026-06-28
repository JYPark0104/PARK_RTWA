# metric_catalog.py

18개 결과 metric (+ coverage_map 특수 경로)을 25\* stage chain으로 자동 매핑.

## 매핑 표 (플랜 4.6)

| Metric | 필요 stage | derived? | RX 필요 |
|--------|-----------|----------|--------|
| `coverage_map` | (Sionna RadioMapSolver) | - | ❌ TX-only |
| `ray_dump` | P1A → P1B | - | ✅ |
| `pdp` | P1A → P1B + derive_pdp | ✅ | ✅ |
| `padp` | P1A → P1B + derive_padp | ✅ | ✅ |
| `rsrp` | P1A → P1B + derive_rsrp | ✅ | ✅ |
| `ray_stats` | P1A → P1B + derive_ray_stats | ✅ | ✅ |
| `ae_ofdm_channel` | P1A→P1B→P1C | - | ✅ |
| `marginal_ccm` | P1A→P1B→P1F | - | ✅ |
| `coupling_matrix` | P1A→P1B→P1F→P1G | - | ✅ |
| `mean_channel` | P1A→P1B→P1H | - | ✅ |
| `separability` | P1A→P1B→P1D | - | ✅ |
| `su_capacity` | ... → P1J | - | ✅ |
| `greedy_beams` | ... → P1L | - | ✅ |
| `swomp_beams` | ... → P1P | - | ✅ |
| `uplink_beams` | ... → P1O | - | ✅ |
| `qie_clustering` | ... → P1L → P1M | - | ✅ |
| `padp_dft_alignment` | P1A→P1B→P1L→P1N | - | ✅ |
| `beam_pattern` | ... → P1P → P1Q | - | ✅ |
| `eigenbeam_pattern` | ... → P1P → P1Q (eigenbeam) | - | ✅ |

## API

```python
from backend.metric_catalog import resolve_metrics, requires_rx, PRESETS

plan = resolve_metrics(["rsrp", "swomp_beams"])
plan.stages_ordered     # ['P1A', 'P1B', 'P1F', 'P1G', 'P1H', 'P1I', 'P1P']
plan.derived            # ['rsrp']
plan.coverage_map       # False
plan.needs_rx           # True

requires_rx(["coverage_map"])     # False
requires_rx(["coverage_map", "rsrp"])  # True
```

## 프리셋 (7종)

| 프리셋 | metrics |
|--------|---------|
| `coverage_only` | `coverage_map` |
| `rt_only` | `ray_dump` |
| `rt_ccm` | `ray_dump`, `marginal_ccm`, `coupling_matrix`, `mean_channel` |
| `rt_ccm_swomp` | 위 + `su_capacity`, `swomp_beams` |
| `rt_ccm_greedy` | 위 + `su_capacity`, `greedy_beams` |
| `ray_viz` | `pdp`, `padp`, `rsrp`, `ray_stats` |
| `bm_full` | Greedy + SWOMP + Uplink + QIE + Beam Pattern + Eigenbeam |

## 의존성 해소 알고리즘

1. 선택된 metric의 `stages`를 모두 수집
2. 각 stage의 prerequisites도 `DEPENDENCY` 그래프로 재귀 확장
3. Kahn's algorithm으로 위상정렬 (in_degree=0인 노드부터)
4. 알파벳 순 안정 정렬 (재현성)

## Coverage Map 특수 경로

- `coverage_map` metric만 단독으로 체크 시 → P1A/P1B 미실행, RadioMapSolver만 실행
- 다른 RX 필요 metric과 함께 체크 시 → 둘 다 실행 (Coverage Map은 항상 별도 잡)
