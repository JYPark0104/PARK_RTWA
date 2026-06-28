# coverage_map_adapter.py

Sionna RT의 `RadioMapSolver`를 호출해 광역 path-gain heatmap을 생성하는 **25\* 외 신규 어댑터**.

## 특수성

- **RX 배치 불필요**: TX만 1개 이상 있으면 실행. metric_catalog에서 `needs_rx=False`로 분기.
- **별도 결과 폴더**: `{session_dir}/Coverage_Map_Results/`
- **diffraction 미지원**: Sionna 1.0~1.2는 `RadioMapSolver`에서 diffraction off (specular/diffuse/refraction만).

## 입력 (Plan 4.7 표)

| 필드 | 기본값 | 설명 |
|------|--------|------|
| `frequency_ghz` | 7.5 | 캐리어 주파수 |
| `cell_size` | (1, 1) | 측정 cell [m] |
| `height_m` | 1.5 | 평면 z 좌표 |
| `center` | None → AABB 중심 | 평면 중심 |
| `size` | None → AABB XY | 평면 가로/세로 |
| `samples_per_tx` | 1e8 | Monte Carlo 샘플 |
| `max_depth` | 5 | 반사 차수 |
| `specular/diffuse/refraction` | True / True / True | reflection 모드 |
| `seed` | 41 | 재현성 |

## 출력

```
Coverage_Map_Results/
├── CoverageMap_{yymmdd_HHMMSS}.npz   # path_gain, path_gain_db, cell_size, center, size, tx_positions, ...
└── CoverageMap_{yymmdd_HHMMSS}.png   # heatmap + TX overlay
```

NPZ 키:
- `path_gain` (float32, (num_tx, ny, nx))
- `path_gain_db` (float32, 동일 shape)
- `cell_size`, `center`, `size`, `height_m`, `frequency_ghz`
- `tx_positions` (num_tx, 3)
- `max_depth`

## API

```python
from backend.adapters.coverage_map_adapter import (
    run_coverage_map, TXPlacement, AntennaSimpleCfg, CoverageMapConfig
)

result = run_coverage_map(
    session_dir=Path("sessions/abc"),
    scene_xml_path=Path("sessions/abc/scene/scene.xml"),
    tx_list=[TXPlacement(position=(-51, -21, 19))],
    antenna_cfg=AntennaSimpleCfg(bs_rows=1, bs_cols=1, pattern="iso"),
    cov_cfg=CoverageMapConfig(frequency_ghz=7.5, cell_size=(2.0, 2.0)),
)
# result["npz_path"], result["png_path"], result["summary"]
```

## 자동 AABB

`center`/`size` 가 None이면 `scene/scene_info.json`의 AABB를 사용해 자동 계산.
즉 `scene_builder.build_scene()`이 먼저 실행되어 있어야 한다.

## CLI 테스트

```bash
python -m backend.adapters.coverage_map_adapter \
    --scene sessions/abc/scene/scene.xml \
    --out   sessions/abc \
    --tx    "-51,-21,19" \
    --bs-rows 4 --bs-cols 4 \
    --cell 2.0 --samples 1e6
```

`samples 1e6`은 빠른 smoke test용. 운영에서는 `1e8` 권장.
