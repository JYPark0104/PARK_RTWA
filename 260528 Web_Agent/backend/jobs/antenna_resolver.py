"""antenna_resolver.py — (rows, cols) → PanelArray dict 자동 분해.

규칙 (plan 4.5 + 결정사항):
- panel 내 patch는 가능한 한 4×4 (25* 검증값, 3GPP TR 38.901 권장).
- 4의 배수가 아니면 2 → 1 fallback.
- grid는 나머지: num_rows = rows / num_rows_per_panel.
- spacing은 0.5λ 디폴트, pattern "tr38901", polarization "V" (P1F TX_Array 기본).

산출되는 dict은 25*의 P1F `TX_Array` / `RX_Array` 와 동일 키.
"""

from __future__ import annotations

from typing import Literal


PATTERN_DEFAULT = "tr38901"
POLARIZATION_DEFAULT: Literal["V", "H", "VH"] = "V"
SPACING_DEFAULT = 0.5  # λ 단위 (Sionna PlanarArray 규약)


def _pick_patch(n: int) -> int:
    """n을 4, 2, 1 순서로 나누어 떨어지는 가장 큰 patch 크기 선택."""

    for k in (4, 2, 1):
        if n % k == 0:
            return k
    return 1


def resolve_panel(
    rows: int,
    cols: int,
    pattern: str = PATTERN_DEFAULT,
    polarization: Literal["V", "H", "VH"] = POLARIZATION_DEFAULT,
    spacing_lambda: float = SPACING_DEFAULT,
) -> dict:
    """안테나 grid (rows × cols)를 P1F PanelArray dict로 분해.

    Examples
    --------
    >>> resolve_panel(32, 32)["num_rows_per_panel"]
    4
    >>> r = resolve_panel(64, 64)
    >>> (r["num_rows_per_panel"], r["num_cols_per_panel"], r["num_rows"], r["num_cols"])
    (4, 4, 16, 16)
    >>> resolve_panel(3, 3)["num_rows_per_panel"]
    1
    """

    if rows < 1 or cols < 1:
        raise ValueError(f"rows/cols must be positive (got rows={rows}, cols={cols})")

    nrp = _pick_patch(rows)
    ncp = _pick_patch(cols)
    nr = rows // nrp
    nc = cols // ncp

    return {
        "num_rows_per_panel": nrp,
        "num_cols_per_panel": ncp,
        "num_rows": nr,
        "num_cols": nc,
        # Panel 간격은 보통 element 간격의 patch 배수
        "vertical_spacing_panel": spacing_lambda * nrp,
        "horizontal_spacing_panel": spacing_lambda * ncp,
        # 패널 내 element 간격
        "vertical_spacing_element": spacing_lambda,
        "horizontal_spacing_element": spacing_lambda,
        "pattern": pattern,
        "polarization": polarization,
    }


def total_ae(panel_dict: dict) -> int:
    """PanelArray dict의 총 안테나 element 수."""

    return (
        panel_dict["num_rows_per_panel"]
        * panel_dict["num_cols_per_panel"]
        * panel_dict["num_rows"]
        * panel_dict["num_cols"]
    )


def estimated_ccm_bytes(n_t: int, dtype_bytes: int = 8) -> int:
    """P1F R_BS / R_UE NPZ 크기 추정. complex64 = 8 bytes."""

    return n_t * n_t * dtype_bytes


# 일반 모드 디폴트 (BS 1024 = 32×32, UE 16 = 4×4)
DEFAULT_BS_ROWS = 32
DEFAULT_BS_COLS = 32
DEFAULT_UE_ROWS = 4
DEFAULT_UE_COLS = 4

DEFAULT_BS_PANEL = resolve_panel(DEFAULT_BS_ROWS, DEFAULT_BS_COLS)
DEFAULT_UE_PANEL = resolve_panel(DEFAULT_UE_ROWS, DEFAULT_UE_COLS)


def simple_to_advanced(
    bs_rows: int,
    bs_cols: int,
    ue_rows: int,
    ue_cols: int,
) -> dict:
    """일반 모드 입력 (rows, cols) → 고급 모드용 전체 PanelArray dict 변환.

    Returns
    -------
    dict
        {"bs_panel": {...}, "ue_panel": {...}, "n_t": int, "n_r": int}
    """

    bs = resolve_panel(bs_rows, bs_cols)
    ue = resolve_panel(ue_rows, ue_cols)
    return {
        "bs_panel": bs,
        "ue_panel": ue,
        "n_t": total_ae(bs),
        "n_r": total_ae(ue),
    }


if __name__ == "__main__":
    import json
    cases = [(4, 4), (8, 8), (32, 32), (64, 64), (16, 64), (3, 3), (1, 1)]
    for r, c in cases:
        d = resolve_panel(r, c)
        n = total_ae(d)
        print(f"{r:3d} x {c:3d} -> patch {d['num_rows_per_panel']}x{d['num_cols_per_panel']}, "
              f"grid {d['num_rows']}x{d['num_cols']}, total AE={n}")
    print("--- simple_to_advanced (BS 64x64, UE 4x4) ---")
    print(json.dumps(simple_to_advanced(64, 64, 4, 4), indent=2))
