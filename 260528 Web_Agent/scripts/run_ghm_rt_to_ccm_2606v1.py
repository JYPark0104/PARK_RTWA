#!/usr/bin/env python3
# ============================================================================
# run_ghm_rt_to_ccm_2606v1.py
#   GHM Twin 씬 → 진짜 MIMO 공분산(R_BS, R_UE) 산출 오프라인 러너
#
#   체인:  scene_builder(.ply/.obj → scene.xml)
#          → P1A (RT, 4각도 ray: aoa/aod/zoa/zod)
#          → P1B (Valid RX filter)
#          → derived (RSRP, PADP)  [--with-derived]
#          → P1F (PanelArray로 Marginal CCM: R_BS, R_UE)
#
#   목적: GHM npz의 공분산은 (1,1) SISO라 무의미. P1F에 MIMO PanelArray를
#         적용해 ray 4각도로부터 R_BS, R_UE(MIMO 공분산)를 합성한다.
#
# ----------------------------------------------------------------------------
# 실행 환경
#   Python : 3.10  (.venv-webagent / .venv-h100)
#   서버   : H100(GPU 가속, 권장)  또는  dclcom61(CPU 모드, 느림 30~120분)
#   GPU    : RT(P1A)는 H100 sm_90 권장. dclcom61(5090 sm_120)은 CPU만.
#   주요 라이브러리: tensorflow 2.21, sionna 1.2.2, mitsuba 3.8.0, drjit 1.3.1,
#                    trimesh>=4.0, numpy, scipy
#
# 사용 (H100 예시):
#   source .venv-h100/bin/activate
#   WEBAGENT_USE_GPU=1 WEBAGENT_USE_GPU_RT=1 \
#   python scripts/run_ghm_rt_to_ccm_2606v1.py \
#       --mesh "/home/user/scenes/GHM Twin_v0.1.ply" \
#       --npz  "/home/user/scenes/channel_data_260531_GHM_Twin_v0_1.npz" \
#       --rx-subset 64 --freq 7.5 --max-depth 5
#
# 사용 (dclcom61 CPU, 작은 검증):
#   source .venv-webagent/bin/activate
#   python scripts/run_ghm_rt_to_ccm_2606v1.py --rx-subset 8 --max-depth 3
#
#   --dry-run : 씬 빌드 + 설정 출력까지만, RT/CCM은 건너뜀 (GPU 불필요)
# ============================================================================

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# --- Web_Agent backend 를 import 가능하게 경로 추가 -------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
WEB_AGENT_ROOT = SCRIPT_DIR.parent              # .../260528 Web_Agent
if str(WEB_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(WEB_AGENT_ROOT))

# 기본 경로 (이 서버 기준; H100에서는 --mesh/--npz 로 직접 지정)
_WEB_AGENT_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _WEB_AGENT_ROOT.parent
_GHM_DIR = _REPO_ROOT / "260604 GHM_test"
DEFAULT_MESH = str(_GHM_DIR / "GHM Twin_v0.1.ply")
DEFAULT_NPZ = str(_GHM_DIR / "channel_data_260531_GHM_Twin_v0_1.npz")
DEFAULT_RX_Z = 1.5  # GHM rx_positions 는 (N,2). 수신 높이[m]


# ---------------------------------------------------------------------------
# GHM npz → TX/RX 좌표 추출
# ---------------------------------------------------------------------------
def load_ghm_geometry(npz_path: Path, rx_subset: int | None, rx_z: float):
    """GHM npz에서 tx_positions(3,3), rx_positions(N,2)를 읽어 P1A 입력으로 정리.

    Returns
    -------
    tx_positions : list[[x,y,z]]      (3개)
    rx_points    : list[[x,y,z]]      (rx_subset개; None이면 전체)
    """
    with np.load(npz_path, allow_pickle=True) as d:
        if "tx_positions" not in d or "rx_positions" not in d:
            raise KeyError(
                f"{npz_path.name}에 tx_positions/rx_positions 키가 없음. "
                f"keys={list(d.keys())[:10]}..."
            )
        tx = np.asarray(d["tx_positions"], dtype=float)   # (3,3)
        rx = np.asarray(d["rx_positions"], dtype=float)    # (N,2)

    tx_positions = [[float(p[0]), float(p[1]), float(p[2])] for p in tx]

    if rx.shape[1] == 2:
        rx = np.column_stack([rx, np.full(len(rx), rx_z)])  # z 채움

    if rx_subset is not None and rx_subset < len(rx):
        # 공간적으로 골고루: 등간격 샘플링
        idx = np.linspace(0, len(rx) - 1, rx_subset).astype(int)
        rx = rx[idx]

    rx_points = [[float(p[0]), float(p[1]), float(p[2])] for p in rx]
    return tx_positions, rx_points


# ---------------------------------------------------------------------------
# P1A 직접 실행 (임의 RX 좌표 'points' 방식 런타임 패치)
# ---------------------------------------------------------------------------
def run_p1a_points(
    session_dir: Path,
    scene_xml: Path,
    tx_positions: list,
    rx_points: list,
    area_index: int,
    freq_ghz: float,
    max_depth: int,
    num_paths: int,
    max_rays: int,
):
    """P1A fork 를 직접 import → Scene.generate_area_rx_grid 에 'points' 지원
    추가(런타임 몽키패치) → web_apply_config_override → execute_main().

    BaseAdapter 를 쓰지 않는 이유: 임의 scatter RX 좌표(GHM 5071점)는
    grid/explicit(데카르트 곱)/radial/street 로 표현 불가하기 때문. 기존 fork
    파일은 수정하지 않고, 이 함수 안에서만 메서드를 확장한다.
    """
    from backend.adapters.base_adapter import (
        STAGE_REGISTRY,
        load_stage_module,
        session_workdir,
    )

    meta = STAGE_REGISTRY["P1A"]
    mod = load_stage_module(meta)

    # --- 'points' RX 배치 방식 추가 (원본 메서드 보존) -----------------------
    _orig_gen = mod.Scene.generate_area_rx_grid

    def _gen_with_points(area_idx):
        ac = mod.Utils.get_area_config(area_idx)
        pc = ac.get("rx_placement", {})
        if pc.get("method") == "points":
            pts = pc["points"]
            print(f"Area {area_idx}: {len(pts)} RX positions using 'points' method")
            return [[float(p[0]), float(p[1]), float(p[2])] for p in pts]
        return _orig_gen(area_idx)

    mod.Scene.generate_area_rx_grid = staticmethod(_gen_with_points)

    area_key = f"area_{area_index}"
    area_configs = {
        area_key: {
            "tx_positions": tx_positions,
            "rx_placement": {"method": "points", "points": rx_points},
            "description": f"GHM Twin area {area_index} (RT→CCM runner)",
        }
    }

    overrides = {
        "WEB_OVERRIDE_SCENE_PATH": str(scene_xml),
        "ENABLE_SCENE_PREVIEW": False,
        "AREA_CONFIGS": area_configs,
        "AREA_INDICES": [area_index],
        "FREQUENCY_CONFIGS": [freq_ghz],
        "MAX_DEPTH": max_depth,
        "NUM_INTERESTING_PATHS": num_paths,
        "MAX_RAYS_PER_PAIR": max_rays,
        "SAVE_ALL_RX": True,
        "ENABLE_RAY_GENERATION": True,
        "ENABLE_RAY_SAVING": True,
    }
    # P1A fork 는 module-level p1a_config(global)만 사용 → setattr override 로 충분
    mod.web_apply_config_override(overrides)

    try:
        with session_workdir(session_dir, meta.source_folder):
            mod.Pipeline.execute_main()
    finally:
        # 몽키패치 원복 (다른 stage/재실행 영향 방지)
        mod.Scene.generate_area_rx_grid = staticmethod(_orig_gen)


def _find_ray_npz(session_dir: Path, area_index: int, prefer_valid: bool = True) -> Path:
    """P1B Valid NPZ 우선, 없으면 P1A ALL_RXs."""

    freq_tag = f"Area{area_index}_*GHz"
    if prefer_valid:
        cands = sorted(session_dir.glob(f"P1B_Valid_Results/{freq_tag}_Rays_Valid_RXs.npz"))
        if cands:
            return cands[-1]
    cands = sorted(session_dir.glob(f"P1A_RT_Results/{freq_tag}_Rays_ALL_RXs.npz"))
    if not cands:
        raise FileNotFoundError(f"Ray NPZ not found under {session_dir} (area={area_index})")
    return cands[-1]


def _run_derived_metrics(
    session_dir: Path,
    rx_points: list,
    area_index: int,
    label: str | None = None,
    skip_rsrp: bool = False,
    tx_positions: list | None = None,
    cell_size_m: float = 5.0,
    ghm_channel_npz: Path | str | None = None,
    mesh_ply_path: Path | str | None = None,
    scene_title: str | None = None,
) -> dict[str, str]:
    """RSRP + PADP (P1B/ALL ray NPZ + UE 좌표)."""

    import numpy as np

    from backend.derived.derive_padp import derive_padp
    from backend.derived.derive_rsrp import derive_rsrp

    npz_path = _find_ray_npz(session_dir, area_index, prefer_valid=True)
    rx_positions = np.asarray(rx_points, dtype=np.float32)
    out: dict[str, str] = {}

    rsrp_dir = session_dir / "Derived_RSRP_Results"
    if skip_rsrp and rsrp_dir.exists() and any(rsrp_dir.glob("RSRP_*.csv")):
        print("\n[derived] RSRP — 기존 결과 있음, 건너뜀")
        existing = sorted(rsrp_dir.glob("RSRP_*.csv"))[-1]
        out["rsrp_csv"] = str(existing)
        out["rsrp_png"] = str(existing.with_suffix(".png"))
    else:
        print(f"\n[derived] RSRP ← {npz_path.name}")
        t0 = time.time()
        r_rsrp = derive_rsrp(
            npz_path,
            rx_positions,
            rsrp_dir,
            label=label,
            tx_positions=tx_positions,
            cell_size_m=cell_size_m,
            ghm_channel_npz=ghm_channel_npz,
            mesh_ply_path=mesh_ply_path,
            scene_title=scene_title,
        )
        out["rsrp_csv"] = r_rsrp["csv_path"]
        out["rsrp_png"] = r_rsrp["png_path"]
        print(f"         RSRP 완료 ({time.time()-t0:.1f}s) → {Path(r_rsrp['csv_path']).name}")

    print(f"[derived] PADP ← {npz_path.name}")
    t0 = time.time()
    r_padp = derive_padp(
        npz_path,
        session_dir / "Derived_PADP_Results",
        label=label,
        max_rx_per_plot=min(4, rx_positions.shape[0]),
    )
    out["padp_npz"] = r_padp["npz_path"]
    out["padp_png"] = r_padp["png_path"]
    print(f"         PADP 완료 ({time.time()-t0:.1f}s) → {Path(r_padp['npz_path']).name}")

    return out


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="GHM 씬 → P1A(RT) → P1B → derived(RSRP,PADP) → P1F(MIMO CCM) 오프라인 러너"
    )
    ap.add_argument("--mesh", default=DEFAULT_MESH, help="GHM 메시 .ply/.obj 경로")
    ap.add_argument("--npz", default=DEFAULT_NPZ, help="GHM channel_data npz (TX/RX 좌표)")
    ap.add_argument("--session-dir", default=None, help="출력 세션 디렉토리 (기본 자동)")
    ap.add_argument("--area-index", type=int, default=1, help="P1A/P1B/P1F area 인덱스")
    ap.add_argument("--freq", type=float, default=7.5, help="반송 주파수 [GHz]")
    ap.add_argument("--rx-subset", type=int, default=64,
                    help="RX 샘플 수 (0 또는 음수면 전체 5071). CPU 검증은 8~64 권장")
    ap.add_argument("--rx-z", type=float, default=DEFAULT_RX_Z, help="RX 높이 [m]")
    ap.add_argument("--max-depth", type=int, default=5, help="RT 최대 반사/회절 깊이")
    ap.add_argument("--num-paths", type=int, default=20, help="RX당 분석 경로 수")
    ap.add_argument("--max-rays", type=int, default=400, help="TX-RX쌍당 최대 ray 수")
    ap.add_argument("--material", default="itu_concrete",
                    choices=["itu_concrete", "itu_ceiling_board", "itu_glass"],
                    help="단일 메시 변환 시 적용할 ITU 재질")
    # P1F PanelArray (MIMO) — 공분산 크기 결정
    ap.add_argument("--bs-panel", type=int, nargs=2, default=[4, 4],
                    metavar=("ROWS", "COLS"), help="BS 패널당 행/열 (R_BS 크기↑)")
    ap.add_argument("--bs-grid", type=int, nargs=2, default=[2, 2],
                    metavar=("NROWS", "NCOLS"), help="BS 패널 격자 (총 TX안테나=panel×grid)")
    ap.add_argument("--ue-panel", type=int, nargs=2, default=[2, 2],
                    metavar=("ROWS", "COLS"), help="UE 패널당 행/열 (R_UE 크기)")
    ap.add_argument("--dry-run", action="store_true",
                    help="씬 빌드 + 설정 출력까지만 (RT/CCM 건너뜀, GPU 불필요)")
    ap.add_argument("--with-derived", action=argparse.BooleanOptionalAction, default=True,
                    help="P1B 후 RSRP·PADP 산출 (기본 ON)")
    ap.add_argument("--skip-p1f", action="store_true",
                    help="P1F Marginal CCM(R_BS,R_UE) 건너뜀")
    ap.add_argument(
        "--from-stage",
        choices=("full", "post-p1b"),
        default="full",
        help="post-p1b: 기존 세션에서 PADP+P1F만 재개 (--session-dir 필수)",
    )
    args = ap.parse_args()

    mesh_path = Path(args.mesh)
    npz_path = Path(args.npz)
    rx_subset = None if args.rx_subset is None or args.rx_subset <= 0 else args.rx_subset

    if args.from_stage == "post-p1b" and not args.session_dir:
        ap.error("--from-stage post-p1b 는 --session-dir 이 필요합니다")

    if not mesh_path.exists():
        ap.error(f"--mesh 없음: {mesh_path}")
    if not npz_path.exists():
        ap.error(f"--npz 없음: {npz_path}")

    ts = time.strftime("%y%m%d_%H%M%S")
    if args.session_dir:
        session_dir = Path(args.session_dir)
        if not session_dir.is_absolute():
            session_dir = WEB_AGENT_ROOT / session_dir
    else:
        tag = "full5071" if rx_subset is None else f"rx{rx_subset}"
        session_dir = WEB_AGENT_ROOT / "sessions" / f"ghm_{tag}_{ts}"
    session_dir.mkdir(parents=True, exist_ok=True)
    scene_dir = session_dir / "scene"

    print("=" * 78)
    print("GHM RT → CCM Runner")
    print(f"  from-stage  : {args.from_stage}")
    print(f"  mesh        : {mesh_path}")
    print(f"  npz         : {npz_path}")
    print(f"  session     : {session_dir}")
    print(f"  area/freq   : {args.area_index} / {args.freq} GHz")
    print(f"  rx-subset   : {'ALL' if rx_subset is None else rx_subset}")
    print(f"  max-depth   : {args.max_depth}")
    print(f"  BS panel/grid: {args.bs_panel} / {args.bs_grid}  → "
          f"n_t={args.bs_panel[0]*args.bs_panel[1]*args.bs_grid[0]*args.bs_grid[1]}")
    print(f"  UE panel    : {args.ue_panel}  → "
          f"n_r={args.ue_panel[0]*args.ue_panel[1]}")
    print(f"  derived     : {args.with_derived} (RSRP, PADP)")
    print(f"  P1F CCM     : {not args.skip_p1f}")
    print("=" * 78)

    # --- 0) 좌표 로드 ------------------------------------------------------
    tx_positions, rx_points = load_ghm_geometry(npz_path, rx_subset, args.rx_z)
    print(f"[geom] TX {len(tx_positions)}개, RX {len(rx_points)}개")
    for i, p in enumerate(tx_positions, 1):
        print(f"       TX{i}: ({p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f})")

    from backend.adapters.p1_adapters import run_p1b, run_p1f

    if args.from_stage == "full":
        # --- 1) 씬 빌드 (.ply/.obj → scene.xml) -------------------------------
        from backend.jobs.scene_builder import build_scene

        print("[scene] 메시 → Mitsuba scene.xml 변환 중...")
        info = build_scene(mesh_path, scene_dir, material=args.material)
        scene_xml = scene_dir / "scene.xml"
        print(f"[scene] 완료: {scene_xml}")
        print(f"        verts={info.n_vertices:,} faces={info.n_faces:,} "
              f"size={[round(s,1) for s in info.size]} m")

        if args.dry_run:
            print("\n[dry-run] RT/CCM 단계는 건너뜀. 씬/설정 검증 완료.")
            return

        # --- 2) P1A: RT (4각도 ray) -------------------------------------------
        print("\n[P1A] Ray Tracing 시작...")
        t0 = time.time()
        run_p1a_points(
            session_dir=session_dir,
            scene_xml=scene_xml,
            tx_positions=tx_positions,
            rx_points=rx_points,
            area_index=args.area_index,
            freq_ghz=args.freq,
            max_depth=args.max_depth,
            num_paths=args.num_paths,
            max_rays=args.max_rays,
        )
        print(f"[P1A] 완료 ({time.time()-t0:.1f}s) → {session_dir/'P1A_RT_Results'}")

        # --- 3) P1B: Valid RX filter ------------------------------------------
        print("\n[P1B] Valid RX 필터링...")
        t0 = time.time()
        run_p1b(session_dir, overrides={"target_areas": [args.area_index]})
        print(f"[P1B] 완료 ({time.time()-t0:.1f}s) → {session_dir/'P1B_Valid_Results'}")
    else:
        print("\n[resume] P1A/P1B 건너뜀 — 기존 세션 사용")
        if not (session_dir / "P1B_Valid_Results").exists():
            ap.error("P1B_Valid_Results 없음. --from-stage full 로 먼저 실행하세요.")

    derived_paths: dict[str, str] = {}
    if args.with_derived:
        derived_paths = _run_derived_metrics(
            session_dir,
            rx_points,
            args.area_index,
            label=f"Area{args.area_index}",
            skip_rsrp=(args.from_stage == "post-p1b"),
            tx_positions=tx_positions,
            cell_size_m=5.0,
            ghm_channel_npz=npz_path,
            mesh_ply_path=mesh_path,
            scene_title="260531_GHM Twin_v0.1",
        )

    # --- 4) P1F: Marginal CCM (MIMO 공분산) -------------------------------
    if args.skip_p1f:
        print("\n[skip-p1f] P1F Marginal CCM 건너뜀.")
        p1f_dir = None
    else:
        print("\n[P1F] MIMO 공분산(R_BS, R_UE) 합성...")
        t0 = time.time()
        p1f_overrides = {
        "target_areas": [args.area_index],
        "TX_Array": {
            "location": [0, 0, 0],
            "rotation": [0, 0],
            "num_rows_per_panel": args.bs_panel[0],
            "num_cols_per_panel": args.bs_panel[1],
            "num_rows": args.bs_grid[0],
            "num_cols": args.bs_grid[1],
            "polarization": "single",
            "polarization_type": "V",
            "antenna_pattern": "38.901",
            "panel_vertical_spacing": 2.5,
            "panel_horizontal_spacing": 2.5,
        },
        "RX_Array": {
            "num_rows_per_panel": args.ue_panel[0],
            "num_cols_per_panel": args.ue_panel[1],
            "num_rows": 1,
            "num_cols": 1,
            "polarization": "single",
            "polarization_type": "V",
            "antenna_pattern": "38.901",
        },
        }
        run_p1f(session_dir, overrides=p1f_overrides)
        p1f_dir = session_dir / "P1F_Marginal_CCM_Results"
        print(f"[P1F] 완료 ({time.time()-t0:.1f}s) → {p1f_dir}")

    # --- 요약 -------------------------------------------------------------
    summary = {
        "timestamp": ts,
        "mesh": str(mesh_path),
        "npz": str(npz_path),
        "session_dir": str(session_dir),
        "area_index": args.area_index,
        "freq_ghz": args.freq,
        "n_tx": len(tx_positions),
        "n_rx": len(rx_points),
        "bs_panel": args.bs_panel,
        "bs_grid": args.bs_grid,
        "ue_panel": args.ue_panel,
        "with_derived": args.with_derived,
        "skip_p1f": args.skip_p1f,
        "outputs": {
            "p1a": str(session_dir / "P1A_RT_Results"),
            "p1b": str(session_dir / "P1B_Valid_Results"),
            "derived_rsrp": str(session_dir / "Derived_RSRP_Results"),
            "derived_padp": str(session_dir / "Derived_PADP_Results"),
            "p1f": str(session_dir / "P1F_Marginal_CCM_Results"),
            **derived_paths,
        },
    }
    (session_dir / f"run_summary_{ts}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\n[done] 산출물:")
    print("  - RT rays     : P1A_RT_Results/*_Rays_ALL_RXs.npz")
    print("  - RSRP        : Derived_RSRP_Results/RSRP_*.csv, *.png")
    print("  - PADP        : Derived_PADP_Results/PADP_*.npz, *.png")
    if not args.skip_p1f:
        print("  - Covariance  : P1F_Marginal_CCM_Results/*_Marginal_CCM.npz (R_BS, R_UE)")
    print(f"  요약 JSON: {session_dir/('run_summary_'+ts+'.json')}")


if __name__ == "__main__":
    main()
