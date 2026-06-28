"""npz_validator.py — 25* 표준 NPZ 포맷 검증기.

251218 E_MIMO_BM 기준 단일 포맷.

각 stage별 기대 키 / dtype / shape rank를 검사하고,
누락/형식 오류 시 ValidationError를 발생시킨다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


class NpzValidationError(ValueError):
    """NPZ 포맷 검증 실패."""


@dataclass
class FieldSpec:
    """단일 NPZ 키 명세."""

    name: str
    dtype_kinds: tuple[str, ...] = ("f", "i", "u")     # numpy.dtype.kind
    ndim: tuple[int, ...] | None = None                 # 허용 차원 수 (None이면 무관)
    optional: bool = False
    note: str = ""


@dataclass
class StageSchema:
    """단일 stage의 NPZ 스키마."""

    stage_id: str
    filename_pattern: str                # 예: "Area*_*GHz_Rays_ALL_RXs.npz"
    fields: list[FieldSpec] = field(default_factory=list)
    description: str = ""


# ---------------------------------------------------------------------------
# 251218 표준 스키마
# ---------------------------------------------------------------------------
SCHEMAS: dict[str, StageSchema] = {
    "P1A": StageSchema(
        stage_id="P1A",
        filename_pattern="Area*_*GHz_Rays_ALL_RXs.npz",
        description="P1A 표준 ray NPZ (251218 형식, 13 keys). "
                    "ray 텐서는 (num_rx, BS_panels, UE_panels, *, *, num_rays) 6-D Sionna 호환 포맷.",
        fields=[
            # 6-D (num_rx, n_bs_panel, n_ue_panel, dummy, dummy, n_rays) — Sionna 호환
            FieldSpec("tau",            dtype_kinds=("f",),       ndim=(5, 6)),
            FieldSpec("power",          dtype_kinds=("f",),       ndim=(5, 6)),
            FieldSpec("theta_r_deg",    dtype_kinds=("f",),       ndim=(5, 6)),
            FieldSpec("theta_t_deg",    dtype_kinds=("f",),       ndim=(5, 6)),
            FieldSpec("phi_r_deg",      dtype_kinds=("f",),       ndim=(5, 6)),
            FieldSpec("phi_t_deg",      dtype_kinds=("f",),       ndim=(5, 6)),
            FieldSpec("counts",         dtype_kinds=("i", "u"),   ndim=(3, 4)),
            FieldSpec("source_path_idx",dtype_kinds=("i", "u"),   ndim=(5, 6)),
            FieldSpec("los_nlos_flag",  dtype_kinds=("i", "u"),   ndim=(5, 6)),
            FieldSpec("area_index",     dtype_kinds=("i", "u"),   ndim=(0,)),
            FieldSpec("frequency_ghz",  dtype_kinds=("f", "i"),   ndim=(0,)),
            FieldSpec("num_rx",         dtype_kinds=("i", "u"),   ndim=(0,)),
            FieldSpec("rx_indices",     dtype_kinds=("i", "u"),   ndim=(1,)),
        ],
    ),
    "P1B": StageSchema(
        stage_id="P1B",
        filename_pattern="Area*_*GHz_Rays_Valid_RXs.npz",
        description="P1B Valid RX filter (P1A 동일 + valid_rx_mask 추가)",
        fields=[
            FieldSpec("tau",            dtype_kinds=("f",),       ndim=(5, 6)),
            FieldSpec("power",          dtype_kinds=("f",),       ndim=(5, 6)),
            FieldSpec("theta_r_deg",    dtype_kinds=("f",),       ndim=(5, 6)),
            FieldSpec("theta_t_deg",    dtype_kinds=("f",),       ndim=(5, 6)),
            FieldSpec("phi_r_deg",      dtype_kinds=("f",),       ndim=(5, 6)),
            FieldSpec("phi_t_deg",      dtype_kinds=("f",),       ndim=(5, 6)),
            FieldSpec("counts",         dtype_kinds=("i", "u"),   ndim=(3, 4, 5)),
            FieldSpec("rx_indices",     dtype_kinds=("i", "u"),   ndim=(1,), optional=True),
            FieldSpec("valid_rx_mask",  dtype_kinds=("b", "i"),   ndim=(1,), optional=True),
        ],
    ),
    "P1F": StageSchema(
        stage_id="P1F",
        filename_pattern="*_MarginalCCM_*.npz",
        description="P1F marginal CCM (R_BS, R_UE)",
        fields=[
            FieldSpec("R_BS",           dtype_kinds=("c", "f"),   ndim=(2, 3, 4),
                      note="(n_t, n_t) per RX, complex64 추천"),
            FieldSpec("R_UE",           dtype_kinds=("c", "f"),   ndim=(2, 3, 4)),
            FieldSpec("frequency_ghz",  dtype_kinds=("f", "i"),   ndim=(0,), optional=True),
        ],
    ),
    "P1G": StageSchema(
        stage_id="P1G",
        filename_pattern="*_CouplingMat_*.npz",
        description="P1G coupling matrix Ω + U_tx, U_rx",
        fields=[
            FieldSpec("Omega",          dtype_kinds=("f",),       ndim=(2, 3, 4)),
            FieldSpec("U_tx",           dtype_kinds=("c", "f"),   ndim=(2, 3, 4)),
            FieldSpec("U_rx",           dtype_kinds=("c", "f"),   ndim=(2, 3, 4)),
        ],
    ),
    "P1H": StageSchema(
        stage_id="P1H",
        filename_pattern="*_MeanCh_*.npz",
        description="P1H mean channel H̄",
        fields=[
            FieldSpec("H_bar",          dtype_kinds=("c", "f"),   ndim=(2, 3, 4)),
        ],
    ),
    "P1I": StageSchema(
        stage_id="P1I",
        filename_pattern="*_Weichsel_Chunk_*.npz",
        description="P1I Weichselberger chunked stats (P1F/P1G/P1H 통합 UE 청크)",
        fields=[
            FieldSpec("ue_indices",     dtype_kinds=("i", "u"),   ndim=(1,)),
            FieldSpec("P1F_R_BS",       dtype_kinds=("c", "f"),   ndim=(3,)),
            FieldSpec("P1F_R_UE",       dtype_kinds=("c", "f"),   ndim=(3,)),
            FieldSpec("P1F_has_stochastic", dtype_kinds=("b",),  ndim=(1,), optional=True),
            FieldSpec("P1G_Omega",      dtype_kinds=("f", "c"),   ndim=(3,), optional=True),
            FieldSpec("P1G_U_BS",       dtype_kinds=("c", "f"),   ndim=(3,), optional=True),
            FieldSpec("P1G_U_UE",       dtype_kinds=("c", "f"),   ndim=(3,), optional=True),
            FieldSpec("P1H_H_mean",     dtype_kinds=("c", "f"),   ndim=(3,), optional=True),
        ],
    ),
}


# ---------------------------------------------------------------------------
# 검증 함수
# ---------------------------------------------------------------------------
@dataclass
class ValidationReport:
    """단일 NPZ 검증 결과."""

    npz_path: str
    stage_id: str
    ok: bool
    missing_keys: list[str] = field(default_factory=list)
    type_errors: list[str] = field(default_factory=list)
    extra_keys: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "npz_path": self.npz_path,
            "stage_id": self.stage_id,
            "ok": self.ok,
            "missing_keys": self.missing_keys,
            "type_errors": self.type_errors,
            "extra_keys": self.extra_keys,
            "notes": self.notes,
        }


def validate_npz(npz_path: Path, stage_id: str, strict: bool = False) -> ValidationReport:
    """단일 NPZ 파일 검증.

    Parameters
    ----------
    npz_path : Path
    stage_id : str
        SCHEMAS의 key (P1A, P1B, P1F, P1G, P1H, P1I).
    strict : bool
        True면 extra_keys도 에러로 취급.
    """

    schema = SCHEMAS.get(stage_id)
    if schema is None:
        raise NpzValidationError(f"No schema for stage_id={stage_id}")

    npz_path = Path(npz_path)
    if not npz_path.exists():
        return ValidationReport(
            npz_path=str(npz_path),
            stage_id=stage_id,
            ok=False,
            type_errors=[f"file does not exist"],
        )

    try:
        data = np.load(npz_path, allow_pickle=False)
    except Exception as e:
        return ValidationReport(
            npz_path=str(npz_path),
            stage_id=stage_id,
            ok=False,
            type_errors=[f"failed to load npz: {e}"],
        )

    actual_keys = set(data.keys())
    expected_keys = {f.name for f in schema.fields}
    required_keys = {f.name for f in schema.fields if not f.optional}

    missing = sorted(required_keys - actual_keys)
    extra = sorted(actual_keys - expected_keys)
    type_errors: list[str] = []
    notes: list[str] = []

    for f in schema.fields:
        if f.name not in actual_keys:
            continue
        arr = data[f.name]
        if arr.dtype.kind not in f.dtype_kinds:
            type_errors.append(
                f"'{f.name}': dtype kind '{arr.dtype.kind}' not in {f.dtype_kinds}"
            )
        if f.ndim is not None and arr.ndim not in f.ndim:
            type_errors.append(
                f"'{f.name}': ndim={arr.ndim} not in {f.ndim}"
            )
        if f.note:
            notes.append(f"{f.name}: {f.note}")

    ok = (len(missing) == 0) and (len(type_errors) == 0)
    if strict and extra:
        ok = False

    return ValidationReport(
        npz_path=str(npz_path),
        stage_id=stage_id,
        ok=ok,
        missing_keys=missing,
        extra_keys=extra,
        type_errors=type_errors,
        notes=notes,
    )


def validate_session(session_dir: Path, stages: list[str] | None = None) -> list[ValidationReport]:
    """세션 디렉토리의 모든 NPZ를 자동 매칭 + 검증.

    각 stage의 filename_pattern에 매칭되는 .npz를 찾아 validate_npz 실행.
    """

    session_dir = Path(session_dir)
    targets = stages or list(SCHEMAS.keys())
    reports: list[ValidationReport] = []
    for sid in targets:
        sch = SCHEMAS.get(sid)
        if sch is None:
            continue
        for npz in sorted(session_dir.rglob(sch.filename_pattern)):
            reports.append(validate_npz(npz, sid))
    return reports


# ---------------------------------------------------------------------------
# Self-test (mock NPZ)
# ---------------------------------------------------------------------------
def _self_test() -> None:
    """단위 테스트: mock NPZ로 각 schema 검증."""

    import tempfile

    print("--- npz_validator self-test ---")
    cases = [
        ("P1A", lambda d: _make_p1a_mock(d)),
        ("P1F", lambda d: _make_p1f_mock(d)),
    ]
    for stage, mkfn in cases:
        with tempfile.TemporaryDirectory() as td:
            npz_path = mkfn(Path(td))
            rep = validate_npz(npz_path, stage)
            assert rep.ok, f"{stage} expected OK: {rep.to_dict()}"
            print(f"  [{stage}] OK")

    # 음성 케이스: P1A에서 power 누락
    with tempfile.TemporaryDirectory() as td:
        npz = _make_p1a_mock(Path(td), drop="power")
        rep = validate_npz(npz, "P1A")
        assert not rep.ok and "power" in rep.missing_keys
        print(f"  [P1A] missing key detected OK")

    print("  all assertions passed.")


def _make_p1a_mock(td: Path, drop: str | None = None) -> Path:
    shape = (4, 1, 1, 1, 50)
    arrays: dict[str, Any] = {
        "tau":            np.zeros(shape, dtype=np.float32),
        "power":          np.zeros(shape, dtype=np.float32),
        "theta_r_deg":    np.zeros(shape, dtype=np.float32),
        "theta_t_deg":    np.zeros(shape, dtype=np.float32),
        "phi_r_deg":      np.zeros(shape, dtype=np.float32),
        "phi_t_deg":      np.zeros(shape, dtype=np.float32),
        "counts":         np.zeros((4, 1, 1), dtype=np.int32),
        "source_path_idx":np.zeros(shape, dtype=np.int32),
        "los_nlos_flag":  np.zeros(shape, dtype=np.int32),
        "area_index":     np.int32(1),
        "frequency_ghz":  np.float32(7.5),
        "num_rx":         np.int32(4),
        "rx_indices":     np.arange(4, dtype=np.int32),
    }
    if drop:
        arrays.pop(drop, None)
    p = td / "Area1_7.5GHz_Rays_ALL_RXs.npz"
    np.savez_compressed(p, **arrays)
    return p


def _make_p1f_mock(td: Path) -> Path:
    arr = np.zeros((4, 64, 64), dtype=np.complex64)
    p = td / "Area1_7.5GHz_MarginalCCM_2605281234.npz"
    np.savez_compressed(p, R_BS=arr, R_UE=arr, frequency_ghz=np.float32(7.5))
    return p


if __name__ == "__main__":
    _self_test()
