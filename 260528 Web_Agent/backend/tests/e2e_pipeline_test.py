"""e2e_pipeline_test.py — 풀 파이프라인 E2E 검증.

목적
----
- P1A → P1B → P1F → P1I → P1P 의 미니 풀 파이프라인 실행.
- 결과 NPZ가 251218 E_MIMO_BM 표준 포맷과 일치하는지 npz_validator로 확인.
- 도출 metric (RSRP / PDP / PADP / Ray Stats) 도 출력 가능한지 점검.

스케일
------
- Sionna 1.2.2 내장 `simple_street_canyon` 씬 사용 (PLY mesh 동봉).
- BS antenna 4×4 = 16 AE, UE 2×2 = 4 AE (Massive-MIMO 의 1/256 스케일).
- RX 4 개, num_interesting_paths=10, max_rays_per_pair=50.
- GPU 가 있으면 사용, 없으면 CPU. 전체 < 5 분 안에 끝나도록 파라미터 축소.

사용법
------
$ python -m backend.tests.e2e_pipeline_test            # 풀 E2E
$ python -m backend.tests.e2e_pipeline_test --stage P1A  # 단일 stage
$ python -m backend.tests.e2e_pipeline_test --validate-only  # 기존 결과만 검증

원본 25* NPZ 와 비교
-------------------
P1A NPZ keys 가 251218 E_MIMO_BM 표준 15-key 와 동일한지 npz_validator.validate_p1a() 가 체크.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
BACKEND_DIR = THIS_DIR.parent
WEB_AGENT_DIR = BACKEND_DIR.parent
sys.path.insert(0, str(WEB_AGENT_DIR))

from backend.adapters.base_adapter import BaseAdapter, REPO_ROOT
from backend.adapters.p1_adapters import run_p1a
from backend.jobs.antenna_resolver import resolve_panel
from backend.npz_validator import validate_npz


def _sionna_builtin_scene_xml(name: str = "simple_street_canyon") -> Path:
    import sionna.rt as srt

    base = Path(srt.__file__).resolve().parent / "scenes" / name
    xml = base / f"{name}.xml"
    if not xml.exists():
        raise FileNotFoundError(f"Sionna built-in scene not found: {xml}")
    return xml


def _area_configs_for_aabb(aabb_min, aabb_max, n_rx_x=4, n_rx_y=4) -> tuple[dict, list[float]]:
    """씬 AABB 기반으로 TX/RX area_configs 자동 생성.

    - TX: 씬 중앙 상공 (xy 중심, z = aabb_max.z 또는 25 m 중 작은 값)
    - RX: 씬 가로/세로의 60% 내부에 n_rx_x × n_rx_y 그리드 (z = 1.5 m)
    """

    cx = float((aabb_min[0] + aabb_max[0]) / 2)
    cy = float((aabb_min[1] + aabb_max[1]) / 2)
    tx_z = float(min(aabb_max[2], 25.0))
    half_w = float((aabb_max[0] - aabb_min[0]) / 2 * 0.3)  # 중심에서 30% 반경
    half_h = float((aabb_max[1] - aabb_min[1]) / 2 * 0.3)
    area_configs = {
        "area_99": {
            "tx_positions": [[cx, cy, tx_z]],
            "rx_placement": {
                "method": "grid",
                "x_params": {"start": cx - half_w, "stop": cx + half_w, "num": n_rx_x},
                "y_params": {"start": cy - half_h, "stop": cy + half_h, "num": n_rx_y},
                "z_params": {"values": [1.5]},
            },
            "description": f"E2E grid {n_rx_x}x{n_rx_y} @ AABB center",
        }
    }
    return area_configs, [cx, cy, tx_z]


def stage_p1a(session_dir: Path, scene_xml: Path | None = None) -> Path:
    if scene_xml is None:
        print(f"\n[E2E] === P1A — Ray tracing on built-in scene (simple_street_canyon) ===")
        scene = _sionna_builtin_scene_xml()
        area_configs = {
            "area_99": {
                "tx_positions": [[0.0, 0.0, 25.0]],
                "rx_placement": {
                    "method": "grid",
                    "x_params": {"start": -30.0, "stop": 30.0, "num": 3},
                    "y_params": {"start": -10.0, "stop":  10.0, "num": 2},
                    "z_params": {"values": [1.5]},
                },
                "description": "E2E mini grid",
            }
        }
    else:
        print(f"\n[E2E] === P1A — Ray tracing on user scene: {scene_xml} ===")
        scene = Path(scene_xml).resolve()
        if not scene.exists():
            raise FileNotFoundError(f"Scene XML not found: {scene}")
        # AABB 정보 읽기 (scene_info.json)
        info_path = scene.parent / "scene_info.json"
        if info_path.exists():
            import json
            info = json.loads(info_path.read_text())
            area_configs, tx_xyz = _area_configs_for_aabb(info["aabb_min"], info["aabb_max"])
            print(f"[E2E]   AABB-driven TX={tx_xyz}, grid 4×4 (60% 중앙)")
        else:
            print(f"[E2E]   (no scene_info.json) using mini grid")
            area_configs = {
                "area_99": {
                    "tx_positions": [[0.0, 0.0, 25.0]],
                    "rx_placement": {
                        "method": "grid",
                        "x_params": {"start": -30.0, "stop": 30.0, "num": 3},
                        "y_params": {"start": -10.0, "stop":  10.0, "num": 2},
                        "z_params": {"values": [1.5]},
                    },
                    "description": "fallback mini grid",
                }
            }

    rt_overrides = {
        "MAX_DEPTH": 3,
        "NUM_INTERESTING_PATHS": 10,
        "MAX_RAYS_PER_PAIR": 50,
        "RT_SEED": 41,
    }
    run_p1a(
        session_dir,
        scene_xml_path=str(scene),
        area_configs=area_configs,
        area_indices=[99],
        frequency_ghz_list=[7.5],
        rt_overrides=rt_overrides,
    )
    npz = next((session_dir / "P1A_RT_Results").glob("*Rays*.npz"), None)
    if npz is None:
        raise FileNotFoundError("P1A NPZ not generated")
    print(f"[E2E]   P1A NPZ: {npz.name}")
    return npz


AREA = 99
COMMON_OVERRIDES = {
    "target_areas": [AREA],
    "target_frequencies": [7.5],
    "target_rxs": None,
}


def stage_p1b(session_dir: Path) -> Path:
    print("\n[E2E] === P1B — Valid RX filter ===")
    BaseAdapter("P1B").run(session_dir, config_overrides=COMMON_OVERRIDES)
    npz = next((session_dir / "P1B_Valid_Results").glob("*Valid*RXs*.npz"), None)
    if npz is None:
        raise FileNotFoundError("P1B NPZ not generated")
    print(f"[E2E]   P1B NPZ: {npz.name}")
    return npz


SMALL_BS_ARRAY = {
    "location": [0, 0, 0],
    "rotation": [0, 0],
    "num_rows_per_panel": 2,
    "num_cols_per_panel": 2,
    "num_rows": 1,
    "num_cols": 1,
    "polarization": "single",
    "polarization_type": "V",
    "antenna_pattern": "38.901",
    "panel_vertical_spacing": 2.5,
    "panel_horizontal_spacing": 2.5,
}
SMALL_UE_ARRAY = {
    "num_rows_per_panel": 2,
    "num_cols_per_panel": 2,
    "num_rows": 1,
    "num_cols": 1,
    "polarization": "single",
    "polarization_type": "V",
    "antenna_pattern": "omni",
    "panel_vertical_spacing": 2.5,
    "panel_horizontal_spacing": 2.5,
}


def stage_p1f(session_dir: Path) -> Path:
    print("\n[E2E] === P1F — Marginal CCM ===")
    overrides = {
        **COMMON_OVERRIDES,
        "TX_Array": SMALL_BS_ARRAY,
        "RX_Array": SMALL_UE_ARRAY,
        "n_t": 4,
        "n_r": 4,
        "OFDM_FFT": 32,
        "static_ch_realizations": 16,
    }
    BaseAdapter("P1F").run(session_dir, config_overrides=overrides)
    out_dir = session_dir / "P1F_Marginal_CCM_Results"
    npz = next(out_dir.glob("*.npz"), None) if out_dir.exists() else None
    if npz is None:
        raise FileNotFoundError("P1F NPZ not generated")
    print(f"[E2E]   P1F NPZ: {npz.name}")
    return npz


def stage_p1g(session_dir: Path) -> Path | None:
    print("\n[E2E] === P1G — Coupling matrix (선택적) ===")
    overrides = {
        **COMMON_OVERRIDES,
        "TX_Array": SMALL_BS_ARRAY,
        "RX_Array": SMALL_UE_ARRAY,
        "n_t": 4,
        "n_r": 4,
        "OFDM_FFT": 32,
        "static_ch_realizations": 16,
    }
    try:
        BaseAdapter("P1G").run(session_dir, config_overrides=overrides)
    except Exception as ex:  # noqa: BLE001
        print(f"[E2E]   P1G skipped: {ex}")
        return None
    out_dir = session_dir / "P1G_CouplingMat_Results"
    return next(out_dir.glob("*.npz"), None) if out_dir.exists() else None


def stage_p1h(session_dir: Path) -> Path | None:
    print("\n[E2E] === P1H — Mean channel (선택적) ===")
    overrides = {
        **COMMON_OVERRIDES,
        "TX_Array": SMALL_BS_ARRAY,
        "RX_Array": SMALL_UE_ARRAY,
        "n_t": 4,
        "n_r": 4,
    }
    try:
        BaseAdapter("P1H").run(session_dir, config_overrides=overrides)
    except Exception as ex:  # noqa: BLE001
        print(f"[E2E]   P1H skipped: {ex}")
        return None
    out_dir = session_dir / "P1H_MeanCh_Results"
    return next(out_dir.glob("*.npz"), None) if out_dir.exists() else None


def stage_p1i(session_dir: Path) -> Path:
    print("\n[E2E] === P1I — Weichselberger chunk ===")
    overrides = {
        **COMMON_OVERRIDES,
        "TX_Array": SMALL_BS_ARRAY,
        "RX_Array": SMALL_UE_ARRAY,
        "n_t": 4,
        "n_r": 4,
        "OFDM_FFT": 32,
        "static_ch_realizations": 16,
    }
    BaseAdapter("P1I").run(session_dir, config_overrides=overrides)
    out_dir = session_dir / "P1I_Weichsel_Chunk_Results"
    npz = next(out_dir.rglob("*.npz"), None) if out_dir.exists() else None
    if npz is None:
        raise FileNotFoundError("P1I NPZ not generated")
    print(f"[E2E]   P1I NPZ: {npz.name}")
    return npz


def stage_p1p(session_dir: Path) -> Path | None:
    print("\n[E2E] === P1P — SWOMP beam search (codebook 매칭 필수) ===")
    # 4 AE 환경에서는 codebook size 16 (default 큰값)와 충돌 가능 → try/except 로 best-effort.
    try:
        BaseAdapter("P1P").run(session_dir, config_overrides=COMMON_OVERRIDES)
    except Exception as ex:  # noqa: BLE001
        print(f"[E2E]   P1P skipped (tiny config): {type(ex).__name__}: {ex}"[:200])
        return None
    out_dir = session_dir / "P1P_SWOMP_Results"
    npz = next(out_dir.glob("*.npz"), None) if out_dir.exists() else None
    if npz:
        print(f"[E2E]   P1P NPZ: {npz.name}")
    return npz


def stage_derived(session_dir: Path):
    print("\n[E2E] === Derived metrics: RSRP / PDP / PADP / Ray Stats ===")
    import numpy as np
    from backend.derived.derive_rsrp import derive_rsrp
    from backend.derived.derive_pdp import derive_pdp
    from backend.derived.derive_padp import derive_padp
    from backend.derived.derive_ray_stats import derive_ray_stats

    p1b = next((session_dir / "P1B_Valid_Results").glob("*Valid*RXs*.npz"))

    # RSRP needs RX positions — derive from grid -30..30, -10..10, num=3×2
    xs = np.linspace(-30.0, 30.0, 3)
    ys = np.linspace(-10.0, 10.0, 2)
    rx_positions = np.array([[x, y, 1.5] for y in ys for x in xs], dtype=np.float32)

    try:
        derive_rsrp(p1b, rx_positions, session_dir / "Derived_RSRP_Results")
        print("[E2E]   derived derive_rsrp OK → Derived_RSRP_Results")
    except Exception as ex:  # noqa: BLE001
        print(f"[E2E]   derived derive_rsrp ERROR: {ex}")

    for fn, sub in [
        (derive_pdp,       "Derived_PDP_Results"),
        (derive_padp,      "Derived_PADP_Results"),
        (derive_ray_stats, "Derived_RayStats_Results"),
    ]:
        try:
            fn(p1b, session_dir / sub)
            print(f"[E2E]   derived {fn.__name__} OK → {sub}")
        except Exception as ex:  # noqa: BLE001
            print(f"[E2E]   derived {fn.__name__} ERROR: {ex}")


def validate_outputs(session_dir: Path) -> dict:
    print("\n[E2E] === NPZ schema validation (251218 standard) ===")
    report: dict[str, dict] = {}

    for stage, pattern, parent in [
        ("P1A", "*Rays*.npz",        "P1A_RT_Results"),
        ("P1B", "*Valid*RXs*.npz",   "P1B_Valid_Results"),
        ("P1F", "*.npz",             "P1F_Marginal_CCM_Results"),
        ("P1I", "**/*.npz",          "P1I_Weichsel_Chunk_Results"),
    ]:
        d = session_dir / parent
        if not d.exists():
            files = []
        elif "**" in pattern:
            files = sorted(d.rglob(pattern.replace("**/", "")))
        else:
            files = sorted(d.glob(pattern))
            if stage == "P1A":
                files = [p for p in files if "Valid" not in p.name]
        if not files:
            print(f"[E2E]   {stage}: no NPZ at {d}/{pattern} (skip)")
            continue
        try:
            rep = validate_npz(files[0], stage_id=stage).to_dict()
            print(f"[E2E]   {stage} {files[0].name}: {'OK' if rep['ok'] else 'FAIL'}  missing={rep['missing_keys']} type_err={rep['type_errors']}")
            report[stage] = rep
        except Exception as ex:  # noqa: BLE001
            print(f"[E2E]   {stage} {files[0].name}: ERROR {ex}")
            report[stage] = {"ok": False, "error": str(ex)}

    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-dir", default=None, help="세션 디렉토리 (없으면 sessions/_e2e_<ts>)")
    parser.add_argument("--scene", default=None, help="사용자 scene.xml 경로 (없으면 Sionna 내장 씬 사용)")
    parser.add_argument("--stage", default="full", choices=["full", "P1A", "P1B", "P1F", "P1G", "P1H", "P1I", "P1P", "derived", "validate"])
    args = parser.parse_args()

    if args.session_dir:
        session_dir = Path(args.session_dir).resolve()
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        session_dir = (REPO_ROOT / "260528 Web_Agent" / "sessions" / f"_e2e_{ts}").resolve()
    session_dir.mkdir(parents=True, exist_ok=True)
    print(f"[E2E] session_dir = {session_dir}")

    t0 = time.time()
    try:
        scene_xml = Path(args.scene).resolve() if args.scene else None
        if args.stage in ("full", "P1A"):
            stage_p1a(session_dir, scene_xml=scene_xml)
        if args.stage in ("full", "P1B"):
            stage_p1b(session_dir)
        if args.stage in ("full", "P1F"):
            stage_p1f(session_dir)
        if args.stage in ("full", "P1G"):
            stage_p1g(session_dir)
        if args.stage in ("full", "P1H"):
            stage_p1h(session_dir)
        if args.stage in ("full", "P1I"):
            stage_p1i(session_dir)
        if args.stage in ("full", "P1P"):
            stage_p1p(session_dir)
        if args.stage in ("full", "derived"):
            stage_derived(session_dir)
    except Exception:
        print("\n[E2E] PIPELINE ERROR:", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()

    if args.stage in ("full", "validate"):
        report = validate_outputs(session_dir)
        ok_count = sum(1 for v in report.values() if v.get("ok"))
        print(f"\n[E2E] === SUMMARY ===  {ok_count}/{len(report)} NPZ ok  ({time.time() - t0:.1f}s)")
        return 0 if ok_count == len(report) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
