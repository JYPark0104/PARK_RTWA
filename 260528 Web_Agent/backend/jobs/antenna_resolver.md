# antenna_resolver.py

사용자가 입력하는 **rows × cols** 안테나 grid를 25*의 P1F `PanelArray` dict로 자동 분해.

## 규칙

- `patch` (panel 내부 element)는 가능한 한 **4×4** 고정 (25* 검증값, 3GPP TR 38.901 권장).
- 4의 배수가 아니면 **2 → 1** fallback.
- `grid` (panel 행렬)는 나머지: `num_rows = rows / num_rows_per_panel`.
- spacing = `0.5λ` (Sionna PlanarArray 규약).
- pattern = `"tr38901"`, polarization = `"V"` (P1F TX_Array 기본).

## 예시

| 입력 | patch | grid | 총 AE |
|------|-------|------|-------|
| 4×4 | 4×4 | 1×1 | 16 |
| 32×32 | 4×4 | 8×8 | 1024 (25* BS 기본) |
| 64×64 | 4×4 | 16×16 | 4096 |
| 16×64 | 4×4 | 4×16 | 1024 |
| 3×3 | 1×1 | 3×3 | 9 (fallback) |

## API

```python
from backend.jobs.antenna_resolver import resolve_panel, simple_to_advanced, total_ae

bs = resolve_panel(64, 64)
print(total_ae(bs))  # 4096

full = simple_to_advanced(bs_rows=64, bs_cols=64, ue_rows=4, ue_cols=4)
# {"bs_panel": {...}, "ue_panel": {...}, "n_t": 4096, "n_r": 16}
```

## 25* 매핑

- P1F 입력 키와 1:1 동일: `num_rows_per_panel`, `num_cols_per_panel`, `num_rows`, `num_cols`, `pattern`, `polarization`, spacings.
- P1A는 항상 1×1 isotropic을 사용 (PathSolver 인자 별도) — antenna_resolver는 P1F+ 단계에서만 사용.

## 디폴트

- `DEFAULT_BS_PANEL`: 32×32 patch(4×4) grid(8×8) → 1024 AE (25* BS 기본)
- `DEFAULT_UE_PANEL`: 4×4 patch(4×4) grid(1×1) → 16 AE (25* UE 기본)
