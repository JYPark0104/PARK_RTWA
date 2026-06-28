"""pipeline_executor.py — 전체 잡 worker.

JobRunner 가 호출하는 단일 함수. submit payload 를 받아:

1. metric_catalog.resolve_metrics(selected) 로 stage_ordered 산출
2. coverage_map metric 이 포함되어 있으면 coverage_map_adapter 호출
3. P1A / ... / P1Q 어댑터 순차 호출
4. derived metric (pdp/padp/rsrp/ray_stats) 산출
5. NPZ 검증

각 step마다 emit() 으로 진행률/로그를 push.
"""

from __future__ import annotations

import json
import logging
import traceback
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..adapters.coverage_map_adapter import (
    AntennaSimpleCfg,
    CoverageMapConfig,
    TXPlacement,
    run_coverage_map,
)
from ..adapters.p1_adapters import run_stage
from ..derived import (
    derive_los_map,
    derive_padp,
    derive_pdp,
    derive_ray_stats,
    derive_rsrp,
)
from ..metric_catalog import resolve_metrics
from ..npz_validator import validate_session
from .antenna_resolver import simple_to_advanced
from .job_runner import JobState

log = logging.getLogger("pipeline_executor")


SESSIONS_ROOT = Path(__file__).resolve().parent.parent / "sessions"


# ---------------------------------------------------------------------------
# RX placement 변환
# ---------------------------------------------------------------------------
def _build_rx_placement_dict(
    rx_grid: dict | None, rx_clicks: dict | None, scene_ply: "Path | None" = None
) -> dict:
    """Pydantic RXGridConfig 또는 RXClickConfig → 25* AREA_CONFIGS 형식 rx_placement."""

    if rx_clicks and rx_clicks.get("positions"):
        positions = rx_clicks["positions"]
        return {
            "method": "explicit",
            "x_coords": [p[0] for p in positions],
            "y_coords": [p[1] for p in positions],
            "z_coords": [p[2] for p in positions],
        }
    rx_grid = rx_grid or {}
    method = rx_grid.get("method", "grid")
    if method == "ground_grid":
        # [PARK_2 RT 이식] 지면 레이캐스팅으로 격자 RX 좌표를 백엔드에서 미리 계산하여
        #   P1A 에는 "points"(쌍으로 묶인 x,y,z)로 전달 → 표준 NPZ 스키마 유지.
        from ..rt_agent_park2.ground import compute_ground_rx_grid
        if scene_ply is None:
            raise ValueError("ground_grid 방식에는 scene PLY 경로가 필요합니다.")
        _mh = rx_grid.get("max_height", None)
        _mh = float(_mh) if _mh is not None and _mh != "" else None
        _region = None
        if all(rx_grid.get(k) is not None for k in ("x_start", "x_stop", "y_start", "y_stop")):
            _region = (float(rx_grid["x_start"]), float(rx_grid["x_stop"]),
                       float(rx_grid["y_start"]), float(rx_grid["y_stop"]))
        pts = compute_ground_rx_grid(
            scene_ply,
            grid_n=int(rx_grid.get("grid_n") or 20),
            margin=float(rx_grid.get("margin") or 0.0),
            rx_height=float(rx_grid.get("rx_height") or 1.5),
            raycasting_z=(float(rx_grid["raycasting_z"]) if rx_grid.get("raycasting_z") is not None else None),
            max_height=_mh,
            region=_region,
            spacing=(float(rx_grid["spacing"]) if rx_grid.get("spacing") not in (None, "") else None),
        )
        return {"method": "points", "points": pts}
    if method == "grid":
        return {
            "method": "grid",
            "x_params": {"start": rx_grid["x_start"], "stop": rx_grid["x_stop"], "num": rx_grid["x_num"]},
            "y_params": {"start": rx_grid["y_start"], "stop": rx_grid["y_stop"], "num": rx_grid["y_num"]},
            "z_params": {"values": rx_grid.get("z_values", [1.5])},
        }
    if method == "explicit":
        return {
            "method": "explicit",
            "x_coords": rx_grid["x_coords"],
            "y_coords": rx_grid["y_coords"],
            "z_coords": rx_grid.get("z_values", [1.5]),
        }
    if method == "radial":
        return {
            "method": "radial",
            "center": list(rx_grid["center_xy"]),
            "radii_m": rx_grid["radii_m"],
            "angles_deg": {
                "start": rx_grid["angles_start"],
                "stop": rx_grid["angles_stop"],
                "num": rx_grid["angles_num"],
            },
            "z_params": {"values": rx_grid.get("z_values", [1.5])},
        }
    if method == "street":
        return {
            "method": "street",
            "path_points": rx_grid["path_points"],
            "num_points": rx_grid["num_points"],
            "z_params": {"values": rx_grid.get("z_values", [1.5])},
        }
    raise ValueError(f"Unknown RX method: {method}")


def _resolve_rx_positions(rx_placement: dict) -> np.ndarray:
    """rx_placement dict를 (N, 3) 좌표 배열로 변환 (RSRP 그래프용)."""

    m = rx_placement["method"]
    if m == "points":
        return np.asarray(rx_placement["points"], dtype=np.float32)
    if m == "explicit":
        xs = rx_placement["x_coords"]
        ys = rx_placement["y_coords"]
        zs = rx_placement["z_coords"]
        pts = []
        for z in zs:
            for y in ys:
                for x in xs:
                    pts.append((x, y, z))
        return np.asarray(pts, dtype=np.float32)
    if m == "grid":
        x_p = rx_placement["x_params"]
        y_p = rx_placement["y_params"]
        zs = rx_placement["z_params"]["values"]
        xs = np.linspace(x_p["start"], x_p["stop"], x_p["num"])
        ys = np.linspace(y_p["start"], y_p["stop"], y_p["num"])
        pts = [(x, y, z) for z in zs for y in ys for x in xs]
        return np.asarray(pts, dtype=np.float32)
    if m == "radial":
        c = rx_placement["center"]
        radii = rx_placement["radii_m"]
        a = rx_placement["angles_deg"]
        zs = rx_placement["z_params"]["values"]
        angles = np.linspace(np.deg2rad(a["start"]), np.deg2rad(a["stop"]), a["num"])
        pts = [
            (c[0] + r * np.cos(t), c[1] + r * np.sin(t), z)
            for z in zs
            for r in radii
            for t in angles
        ]
        return np.asarray(pts, dtype=np.float32)
    if m == "street":
        sp, ep = rx_placement["path_points"]
        n = rx_placement["num_points"]
        zs = rx_placement["z_params"]["values"]
        ts = np.linspace(0, 1, n)
        pts = [(sp[0] + t * (ep[0] - sp[0]), sp[1] + t * (ep[1] - sp[1]), z) for z in zs for t in ts]
        return np.asarray(pts, dtype=np.float32)
    raise ValueError(f"Unknown RX method: {m}")


# ---------------------------------------------------------------------------
# Pipeline executor (worker)
# ---------------------------------------------------------------------------
def execute_pipeline(job: JobState, emit: Callable[[dict], None]) -> dict:
    """JobRunner에 등록되는 worker.

    payload (dict):
        session_uuid: str
        tx_list: [{position, orientation, name}]
        rx_grid: RXGridConfig | None
        rx_clicks: RXClickConfig | None
        antenna: AntennaConfig (Pydantic .dict())
        rt: RTConfig (.dict())
        metrics: MetricSelection (.dict())
    """

    # JobRunner가 payload를 별도로 보관하지만, 여기서는 closure로 받지 못하므로
    # job.output_paths에 임시 저장한 payload를 사용한다.
    from .job_runner import runner as _runner
    runner_payload = _runner._payloads.get(job.job_id, ("", {}))[1]
    session_uuid = (runner_payload or {}).get("session_uuid")
    if not session_uuid:
        raise RuntimeError(f"No payload for job {job.job_id}")

    session_dir = SESSIONS_ROOT / session_uuid

    # 실행 시점에 디스크 config 스냅샷을 우선 로드 → queued 동안 수정한 설정이 반영됨.
    from .queue_store import load_session_config
    disk_cfg = load_session_config(session_dir)
    payload = disk_cfg if disk_cfg else runner_payload

    scene_xml = session_dir / "scene" / "scene.xml"
    if not scene_xml.exists():
        raise FileNotFoundError(
            f"scene.xml not found at {scene_xml}. Did you run scene_builder?"
        )

    emit({"kind": "log", "message": f"session={session_uuid}, scene={scene_xml}"})

    metrics_sel = payload["metrics"]["metrics"]
    plan = resolve_metrics(metrics_sel)
    emit({"kind": "log", "message": f"resolved plan: stages={plan.stages_ordered} derived={plan.derived} coverage={plan.coverage_map}"})

    output_paths: dict[str, str] = {}

    antenna = payload["antenna"]
    if antenna["mode"] == "simple":
        s = antenna["simple"]
        ant_full = simple_to_advanced(s["bs_rows"], s["bs_cols"], s["ue_rows"], s["ue_cols"])
    else:
        ant_full = {"bs_panel": antenna["bs_panel"], "ue_panel": antenna["ue_panel"]}

    rt = payload["rt"]

    # ----- Batch RT 엔진 (PARK_2 방식) — P1A/metric 경로와 분기 ---------------
    if rt.get("engine") == "batch":
        emit({"kind": "log", "message": "RT 엔진: batch (PARK_2 batch_rx) — metric 단계는 건너뜁니다."})
        scene_ply = session_dir / "scene" / "meshes" / "scene_mesh.ply"
        if not scene_ply.exists():
            cands = sorted((session_dir / "scene" / "meshes").glob("*.ply"))
            scene_ply = cands[0] if cands else scene_ply
        from ..rt_agent_park2 import run_batch_rt
        try:
            bp = run_batch_rt(payload, session_dir, scene_xml, scene_ply, emit)
            output_paths.update(bp)
        except Exception as exc:
            emit({"kind": "error", "stage": "BatchRT", "message": f"{type(exc).__name__}: {exc}"})
            emit({"kind": "log", "message": traceback.format_exc()})
            raise
        emit({"kind": "progress", "progress": 1.0})
        return {"output_paths": output_paths}

    # ----- Coverage Map (특수 경로) ----------------------------------------
    if plan.coverage_map:
        emit({"kind": "stage_start", "stage": "CoverageMap", "message": "Running coverage map..."})
        cov = rt.get("coverage_map", {})
        cov_cfg = CoverageMapConfig(
            frequency_ghz=rt.get("frequency_ghz", 7.5),
            cell_size=(cov.get("cell_size_x", 1.0), cov.get("cell_size_y", 1.0)),
            height_m=cov.get("height_m", 1.5),
            center=tuple(cov.get("center_xy") + [cov.get("height_m", 1.5)]) if cov.get("center_xy") else None,
            size=tuple(cov.get("size_xy")) if cov.get("size_xy") else None,
            samples_per_tx=int(cov.get("samples_per_tx", 1e8)),
            max_depth=cov.get("max_depth", rt.get("max_depth", 5)),
            specular_reflection=cov.get("specular_reflection", True),
            diffuse_reflection=cov.get("diffuse_reflection", True),
            refraction=cov.get("refraction", True),
            seed=rt.get("seed", 41),
        )
        # Coverage Map용 antenna: 사용자 입력 그대로 (단일 BS, 단일 grid)
        ant_simple = AntennaSimpleCfg(
            bs_rows=antenna["simple"]["bs_rows"] if antenna["mode"] == "simple" else 1,
            bs_cols=antenna["simple"]["bs_cols"] if antenna["mode"] == "simple" else 1,
        )
        tx_list = [
            TXPlacement(position=tuple(tx["position"]), orientation=tuple(tx.get("orientation", (0, 0, 0))))
            for tx in payload["tx_list"]
        ]
        cov_res = run_coverage_map(session_dir, scene_xml, tx_list, ant_simple, cov_cfg)
        output_paths["coverage_npz"] = cov_res["npz_path"]
        output_paths["coverage_png"] = cov_res["png_path"]
        emit({"kind": "stage_end", "stage": "CoverageMap", "message": json.dumps(cov_res["summary"])})

    # ----- 25* stages (P1A → ...) -----------------------------------------
    rx_placement = None
    if plan.stages_ordered:
        # Geo-Radio Env. Twin 은 재질별로 여러 .ply 로 쪼개져 scene_mesh.ply 가 없다.
        # 지면 레이캐스팅(ground_grid)은 모든 PLY 를 함께 써야 누락이 없다. (batch 경로와 동일)
        _meshes_dir = session_dir / "scene" / "meshes"
        _all_plys = sorted(_meshes_dir.glob("*.ply"))
        _scene_ply_arg = [str(p) for p in _all_plys] if _all_plys else (_meshes_dir / "scene_mesh.ply")
        rx_placement = _build_rx_placement_dict(
            payload.get("rx_grid"), payload.get("rx_clicks"),
            scene_ply=_scene_ply_arg,
        )
        area_configs = {
            "area_99": {
                "tx_positions": [list(tx["position"]) for tx in payload["tx_list"]],
                "rx_placement": rx_placement,
                "description": "Web Agent dynamic area",
            }
        }

        n_stages = len(plan.stages_ordered)
        for i, stage_id in enumerate(plan.stages_ordered):
            if job.state == "cancelled":
                emit({"kind": "log", "message": "cancelled by user"})
                return {"output_paths": output_paths}
            progress = i / max(n_stages, 1)
            emit({"kind": "stage_start", "stage": stage_id, "progress": progress})
            try:
                if stage_id == "P1A":
                    run_stage(
                        "P1A",
                        session_dir=session_dir,
                        scene_xml_path=str(scene_xml),
                        area_configs=area_configs,
                        area_indices=[99],
                        frequency_ghz_list=[rt.get("frequency_ghz", 7.5)],
                        rt_overrides={
                            "MAX_DEPTH": rt.get("max_depth", 5),
                            "RT_SEED": rt.get("seed", 41),
                            "RANDOM_SEED": rt.get("seed", 41),
                            "NUM_INTERESTING_PATHS": rt.get("num_interesting_paths", 20),
                            "MAX_RAYS_PER_PAIR": rt.get("max_rays_per_pair", 400),
                            "PATHSOLVER_LOS": rt.get("pathsolver_los", True),
                            "PATHSOLVER_SPECULAR_REFLECTION": rt.get("pathsolver_specular_reflection", True),
                            "PATHSOLVER_DIFFUSE_REFLECTION": rt.get("pathsolver_diffuse_reflection", True),
                            "PATHSOLVER_REFRACTION": rt.get("pathsolver_refraction", True),
                            "PATHSOLVER_SYNTHETIC_ARRAY": rt.get("pathsolver_synthetic_array", False),
                            "ITU_SCATTERING_COEFF": rt.get("itu_scattering_coeff", 0.2),
                            "ITU_XPD_COEFF": rt.get("itu_xpd_coeff", 0.5),
                            # PARK_2 config.yaml 이식 (고급 옵션)
                            "PATHSOLVER_DIFFRACTION": rt.get("pathsolver_diffraction", False),
                            "PATHSOLVER_EDGE_DIFFRACTION": rt.get("pathsolver_edge_diffraction", False),
                            "PATHSOLVER_DIFFRACTION_LIT_REGION": rt.get("pathsolver_diffraction_lit_region", True),
                            "RT_NUM_SAMPLES": rt.get("num_samples", 0) or 0,
                            "RT_MAX_NUM_PATHS": rt.get("max_num_paths", 0) or 0,
                            # RadioMaterial 산란 패턴 (전기적 특성은 P1A ITU 모델 유지)
                            "SCATTERING_PATTERN": rt.get("scattering_pattern", "lambertian"),
                            "DIRECTIVE_ALPHA_R": rt.get("directive_alpha_r", 10),
                            "BACKSCATTERING_ALPHA_R": rt.get("backscattering_alpha_r", 20),
                            "BACKSCATTERING_ALPHA_I": rt.get("backscattering_alpha_i", 30),
                            "BACKSCATTERING_LAMBDA": rt.get("backscattering_lambda", 0.7),
                        },
                    )
                else:
                    # P1B 이후 — 추가 PanelArray override 필요 시 ant_full을 사용
                    overrides: dict[str, Any] = {}
                    if stage_id in {"P1F", "P1G", "P1H", "P1I", "P1J", "P1L", "P1M", "P1N", "P1O", "P1P", "P1Q"}:
                        bs, ue = ant_full["bs_panel"], ant_full["ue_panel"]
                        overrides.update({
                            "BS_NUM_ROWS_PER_PANEL": bs["num_rows_per_panel"],
                            "BS_NUM_COLS_PER_PANEL": bs["num_cols_per_panel"],
                            "BS_NUM_ROWS": bs["num_rows"],
                            "BS_NUM_COLS": bs["num_cols"],
                            "UE_NUM_ROWS_PER_PANEL": ue["num_rows_per_panel"],
                            "UE_NUM_COLS_PER_PANEL": ue["num_cols_per_panel"],
                            "UE_NUM_ROWS": ue["num_rows"],
                            "UE_NUM_COLS": ue["num_cols"],
                        })
                    run_stage(stage_id, session_dir=session_dir, overrides=overrides)
                emit({"kind": "stage_end", "stage": stage_id, "progress": (i + 1) / n_stages})
            except Exception as exc:
                emit({"kind": "error", "stage": stage_id, "message": f"{type(exc).__name__}: {exc}"})
                emit({"kind": "log", "message": traceback.format_exc()})
                raise

    # ----- Derived metrics -------------------------------------------------
    if plan.derived:
        # P1B의 ALL_RXs NPZ 자동 탐색
        candidates = sorted(session_dir.rglob("Area*_*GHz_Rays_ALL_RXs.npz"))
        if not candidates:
            emit({"kind": "log", "message": "no Ray NPZ found for derived metrics — skipping"})
        else:
            npz_path = candidates[-1]   # 가장 최근
            emit({"kind": "log", "message": f"derived input NPZ: {npz_path.name}"})
            rx_positions = _resolve_rx_positions(rx_placement) if rx_placement else np.zeros((0, 3))
            for m in plan.derived:
                emit({"kind": "stage_start", "stage": f"derive_{m}"})
                try:
                    if m == "rsrp":
                        tx_positions = [
                            list(tx["position"])
                            for tx in payload.get("tx_list", [])
                            if tx.get("position")
                        ]
                        r = derive_rsrp(
                            npz_path,
                            rx_positions,
                            session_dir / "Derived_RSRP_Results",
                            tx_positions=tx_positions or None,
                        )
                        output_paths[f"derived_rsrp_csv"] = r["csv_path"]
                        output_paths[f"derived_rsrp_png"] = r["png_path"]
                    elif m == "pdp":
                        r = derive_pdp(npz_path, session_dir / "Derived_PDP_Results")
                        output_paths[f"derived_pdp_npz"] = r["npz_path"]
                        output_paths[f"derived_pdp_png"] = r["png_path"]
                    elif m == "padp":
                        r = derive_padp(npz_path, session_dir / "Derived_PADP_Results")
                        output_paths[f"derived_padp_npz"] = r["npz_path"]
                        output_paths[f"derived_padp_png"] = r["png_path"]
                    elif m == "ray_stats":
                        r = derive_ray_stats(npz_path, session_dir / "Derived_RayStats_Results")
                        output_paths[f"derived_ray_stats_csv"] = r["csv_path"]
                    emit({"kind": "stage_end", "stage": f"derive_{m}", "message": json.dumps(r.get("summary", {}))})
                except Exception as exc:
                    emit({"kind": "error", "stage": f"derive_{m}", "message": str(exc)})
                    raise

    # ----- Review 번들 (요구사항 9): all-TX RSRP map + LoS/NLoS map ---------
    # RT(P1A)가 돌아 Ray NPZ가 생성된 경우, metric 선택과 무관하게 Review용
    # 대표 이미지(RSRP 히트맵 + LoS/NLoS 맵)를 생성한다.
    if "P1A" in plan.stages_ordered:
        review_candidates = sorted(session_dir.rglob("Area*_*GHz_Rays_ALL_RXs.npz"))
        if review_candidates:
            npz_path = review_candidates[-1]
            review_dir = session_dir / "Review_Results"
            rx_positions = _resolve_rx_positions(rx_placement) if rx_placement else np.zeros((0, 3))
            tx_positions = [list(tx["position"]) for tx in payload.get("tx_list", []) if tx.get("position")]
            mesh_ply = session_dir / "scene" / "meshes" / "scene_mesh.ply"
            mesh_ply = mesh_ply if mesh_ply.exists() else None

            emit({"kind": "stage_start", "stage": "Review", "message": "RSRP + LoS/NLoS map 생성"})
            try:
                r_rsrp = derive_rsrp(
                    npz_path, rx_positions, review_dir,
                    label="review", tx_positions=tx_positions or None,
                    mesh_ply_path=mesh_ply, dashboard=True,
                )
                output_paths["review_rsrp_png"] = r_rsrp["png_path"]
                output_paths["review_rsrp_csv"] = r_rsrp["csv_path"]

                r_los = derive_los_map(
                    npz_path, rx_positions, review_dir,
                    label="review", tx_positions=tx_positions or None,
                    mesh_ply_path=mesh_ply,
                )
                output_paths["review_los_png"] = r_los["png_path"]
                output_paths["review_los_csv"] = r_los["csv_path"]
                emit({"kind": "stage_end", "stage": "Review",
                      "message": json.dumps({"rsrp": r_rsrp["summary"], "los": r_los["summary"]})})
            except Exception as exc:
                emit({"kind": "log", "message": f"Review 생성 실패(계속): {type(exc).__name__}: {exc}"})

    # ----- NPZ 검증 --------------------------------------------------------
    reports = validate_session(session_dir, stages=[s for s in plan.stages_ordered if s in {"P1A","P1B","P1F","P1G","P1H","P1I"}])
    if reports:
        ok_count = sum(1 for r in reports if r.ok)
        emit({"kind": "log", "message": f"npz_validator: {ok_count}/{len(reports)} OK"})
        for r in reports:
            if not r.ok:
                emit({"kind": "log", "message": f"VALIDATION FAIL {r.npz_path}: missing={r.missing_keys} type_errors={r.type_errors}"})

    emit({"kind": "progress", "progress": 1.0})
    return {"output_paths": output_paths}


class JobCancelled(Exception):
    """협력적 중단 신호 — emit 시점에 job.state == 'cancelled' 면 발생."""


def _pipeline_worker(job: JobState, emit: Callable[[dict], None]) -> dict:
    """execute_pipeline 래퍼: 세션 상태 영속화 + 협력적 중단.

    - 시작 시 status=processing
    - emit 마다 진행률을 디스크에 반영하고, 취소 요청이면 JobCancelled 발생
      (다음 stage/batch 경계에서 정지)
    - 완료/실패/취소를 세션 메타에 기록
    """
    from .job_runner import runner as _runner
    from .queue_store import make_store

    store = make_store()
    runner_payload = _runner._payloads.get(job.job_id, ("", {}))[1]
    session_uuid = (runner_payload or {}).get("session_uuid", "")

    if session_uuid:
        store.mark_processing(session_uuid)

    def emit2(event: dict) -> None:
        if job.state == "cancelled":
            raise JobCancelled()
        if session_uuid and ("progress" in event or event.get("kind") in ("stage_start", "stage_end")):
            store.update_progress(session_uuid, event.get("progress"), event.get("stage"))
        emit(event)

    try:
        out = execute_pipeline(job, emit2)
        if session_uuid:
            store.mark_done(session_uuid)
        return out
    except JobCancelled:
        if session_uuid:
            store.mark_cancelled(session_uuid)
        emit({"kind": "log", "message": "⏹ 사용자 요청으로 중단되었습니다."})
        return {"output_paths": {}}
    except Exception as exc:  # noqa: BLE001
        if session_uuid:
            store.mark_failed(session_uuid, f"{type(exc).__name__}: {exc}")
        raise


def register_default_worker() -> None:
    """JobRunner에 'pipeline' kind worker로 등록."""

    from .job_runner import runner

    runner.register_worker("pipeline", _pipeline_worker)
