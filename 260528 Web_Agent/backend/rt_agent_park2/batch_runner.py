"""batch_runner.py — PARK_2 batch_rx Ray Tracing 엔진을 Web Agent 잡으로 구동.

요구사항:
  - 4.RT 단계에서 'batch RT' 엔진 선택 시 호출.
  - PARK_2 (2-1.RayTracingAgent) 의 batch_rx 방식 RT 를 그대로 구동:
      scene 로드 → TX별 batch RT(PathSolver) → PostProcess(RSRP/R_TX/R_RX)
      → channel_data npz + RSRP/LoS/PDP/PADP/공분산/hitmap + USD/OBJ export
  - 배치 루프마다 emit() 으로 진행률(progress) + PARK_2 식 로그를 전송.

설계 메모:
  - 엔진 모듈(agents/, configs/, RT_utils)은 ./engine/ 에 벤더링(사본). 원본 무수정.
  - 표준 251218 NPZ 파이프라인(P1A/metric)과는 별개 경로(A안 분기). channel_data 는
    PARK_2 포맷.
  - 진행바 텍스트(`[TX0] Batch RT |███| 35/102 ...`)는 웹 로그 패널에 배치당 한 줄.

실행 환경: Python 3.10 / sionna 1.2.2 / open3d / numpy 2.2 (컨테이너 venv-webagent)
"""

from __future__ import annotations

import os
import sys
import json
import time
import random
import traceback
import datetime
from pathlib import Path
from typing import Callable

import numpy as np

_ENGINE_DIR = Path(__file__).resolve().parent / "engine"


def _ensure_engine_on_path() -> None:
    p = str(_ENGINE_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)


def _bar(cur: int, total: int, width: int = 28) -> str:
    frac = cur / max(total, 1)
    filled = int(round(width * frac))
    return "█" * filled + "░" * (width - filled)


def _xy_bounds(rx_xy: list) -> list:
    """전체 RX (x,y) 의 [minx, miny, maxx, maxy] (스캐터 축 고정용)."""
    if not rx_xy:
        return [0.0, 0.0, 1.0, 1.0]
    xs = [float(p[0]) for p in rx_xy]
    ys = [float(p[1]) for p in rx_xy]
    return [min(xs), min(ys), max(xs), max(ys)]


def _write_live_scatter(session_dir: Path, payload: dict) -> None:
    """Processing 대시보드용 라이브 커버리지 스냅샷 (배치마다 덮어쓰기)."""
    try:
        (Path(session_dir) / "live_scatter.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )
    except Exception:
        pass


def _build_config(payload: dict, session_dir: Path, scene_xml: Path, scene_ply: Path):
    """웹 payload → PARK_2 RT_Config (batch_rx 모드)."""
    _ensure_engine_on_path()
    from configs.default_config import RT_Config  # type: ignore

    rt = payload["rt"]
    ant = payload["antenna"]
    simple = ant.get("simple", {}) if ant.get("mode", "simple") == "simple" else {}
    bs_rows = int(simple.get("bs_rows", 16)); bs_cols = int(simple.get("bs_cols", 16))
    ue_rows = int(simple.get("ue_rows", 4));  ue_cols = int(simple.get("ue_cols", 4))

    tx_positions = [list(map(float, tx["position"])) for tx in payload.get("tx_list", []) if tx.get("position")]
    if not tx_positions:
        raise ValueError("batch RT: TX 가 하나 이상 필요합니다.")

    # Geo-Radio Env. Twin: 재질별로 쪼개진 전체 PLY (지면 레이캐스팅은 전부를 한 메시처럼 사용).
    # 하나만 쓰면 MA(재질부여) 영역의 지면을 못 찾아 RX 가 dead zone 으로 빠진다.
    _meshes_dir = Path(session_dir) / "scene" / "meshes"
    all_plys = [str(p) for p in sorted(_meshes_dir.glob("*.ply"))] or [str(scene_ply)]
    ground_src = all_plys   # ground.py 함수들은 단일/리스트 모두 허용

    # RX: ground_grid 등에서 만들어진 (x,y[,z]) → 2D (x,y) 로. z 는 엔진이 지면+height 재계산.
    rx_xy: list[list[float]] = []
    rg = payload.get("rx_grid") or {}
    rc = payload.get("rx_clicks") or {}
    # 설치 최대 높이 (지면고도+rx_height 가 이 값 초과면 제거 — 건물 옥상 RX 방지). 모든 방법 공용.
    _mh = rg.get("max_height", None)
    max_height = float(_mh) if _mh is not None and _mh != "" else None
    from .ground import compute_ground_rx_grid, filter_xy_by_ground_height
    if rg.get("method") == "ground_grid":
        _region = None
        if all(rg.get(k) is not None for k in ("x_start", "x_stop", "y_start", "y_stop")):
            _region = (float(rg["x_start"]), float(rg["x_stop"]),
                       float(rg["y_start"]), float(rg["y_stop"]))
        pts = compute_ground_rx_grid(
            ground_src,
            grid_n=int(rg.get("grid_n") or 20),
            margin=float(rg.get("margin") or 0.0),
            rx_height=float(rg.get("rx_height") or 1.5),
            raycasting_z=(float(rg["raycasting_z"]) if rg.get("raycasting_z") is not None else None),
            max_height=max_height,
            region=_region,
            spacing=(float(rg["spacing"]) if rg.get("spacing") not in (None, "") else None),
        )
        rx_xy = [[float(p[0]), float(p[1])] for p in pts]
        rx_height = float(rg.get("rx_height") or 1.5)
    elif rc.get("positions"):
        rx_xy = [[float(p[0]), float(p[1])] for p in rc["positions"]]
        rx_height = 1.5
    else:
        # grid 등 다른 방식: (x,y) 평면 격자
        from ..jobs.pipeline_executor import _resolve_rx_positions, _build_rx_placement_dict
        pl = _build_rx_placement_dict(rg, rc, scene_ply=scene_ply)
        arr = _resolve_rx_positions(pl)
        rx_xy = [[float(x), float(y)] for x, y, *_ in arr]
        rx_height = 1.5
    # ground_grid 가 아닌 방식도 설치 최대 높이 적용 (지면 스냅 후 높이 기준)
    if max_height is not None and rg.get("method") != "ground_grid" and rx_xy:
        try:
            rx_xy = filter_xy_by_ground_height(ground_src, rx_xy, max_height, rx_height=rx_height)
        except Exception:
            pass
    if not rx_xy:
        raise ValueError("batch RT: RX 위치가 비어 있습니다. (3.TX/RX 에서 배치 필요)")

    out_dir = session_dir / "Batch_RT_Results"

    cfg = RT_Config(
        map_xml=str(scene_xml), map_ply=str(scene_ply),
        map_title=payload.get("session_uuid", "web")[:8],
        tx_position=tuple(tx_positions[0]),
    )
    cfg.map_ply_all = all_plys   # 지면 레이캐스팅용 전체 PLY (Geo-Radio 다중)
    cfg.rt_mode = "batch_rx"
    cfg.batch_size = int(rt.get("batch_size", 50) or 50)
    cfg.tx_positions = tx_positions
    cfg.rx_positions = rx_xy
    cfg.rx_height = rx_height

    # PathSolver
    cfg.num_samples = int(rt.get("num_samples", 100000) or 100000)
    cfg.max_depth = int(rt.get("max_depth", 7) or 7)
    cfg.max_num_paths = int(rt.get("max_num_paths", 10000) or 10000)
    cfg.seed = int(rt.get("seed", 41) or 41)
    cfg.los = bool(rt.get("pathsolver_los", True))
    cfg.specular_reflection = bool(rt.get("pathsolver_specular_reflection", True))
    cfg.diffuse_reflection = bool(rt.get("pathsolver_diffuse_reflection", True))
    cfg.refraction = bool(rt.get("pathsolver_refraction", True))
    cfg.diffraction = bool(rt.get("pathsolver_diffraction", False))
    cfg.edge_diffraction = bool(rt.get("pathsolver_edge_diffraction", False))
    cfg.diffraction_lit_region = bool(rt.get("pathsolver_diffraction_lit_region", True))
    # PARK_2 postprocess(_extract_mimo_cir)는 synthetic_array=True shape 를 전제로 한다.
    # (웹 P1A 용 pathsolver_synthetic_array 기본값 False 와 무관하게 batch 엔진은 True 고정)
    cfg.synthetic_array = True

    # 재질
    # sionna RadioMaterial 은 scattering_coefficient ∈ (0,1), xpd_coefficient ∈ [0,1] 을 강제한다.
    # 범위 밖 입력(예: dB 값 오입력 -3.8)은 하드 크래시(ValueError)를 내므로 안전하게 보정한다.
    _sc = float(rt.get("itu_scattering_coeff", 0.7) or 0.7)
    cfg.scattering_coefficient = _sc if (0.0 < _sc < 1.0) else 0.7  # 범위 밖 → 기본값 복구
    cfg._scattering_coefficient_raw = _sc  # 경고 로깅용 원본 보존
    _xpd = float(rt.get("itu_xpd_coeff", 0.0) or 0.0)
    cfg.xpd_coefficient = min(max(_xpd, 0.0), 1.0)  # [0,1] 클램프
    cfg._xpd_coefficient_raw = _xpd
    cfg.scattering_pattern = str(rt.get("scattering_pattern", "lambertian") or "lambertian")
    cfg.directive_alpha_r = int(rt.get("directive_alpha_r", 10) or 10)
    cfg.backscattering_alpha_r = int(rt.get("backscattering_alpha_r", 20) or 20)
    cfg.backscattering_alpha_i = int(rt.get("backscattering_alpha_i", 30) or 30)
    cfg.backscattering_lambda = float(rt.get("backscattering_lambda", 0.7) or 0.7)

    # 통신/안테나 (BS→TX, UE→RX)
    cfg.frequency = float(rt.get("frequency_ghz", 7.5) or 7.5) * 1e9
    cfg.num_tx_rows, cfg.num_tx_cols = bs_rows, bs_cols
    cfg.num_rx_rows, cfg.num_rx_cols = ue_rows, ue_cols
    cfg.tx_pattern = str(rt.get("tx_pattern", "iso") or "iso")
    cfg.tx_polarization = str(rt.get("tx_polarization", "V") or "V")
    cfg.rx_pattern = str(rt.get("rx_pattern", "dipole") or "dipole")
    cfg.rx_polarization = str(rt.get("rx_polarization", "V") or "V")
    pol = lambda s: 2 if str(s).lower() in ("vh", "cross", "dual") else 1
    cfg.num_tx_ant = bs_rows * bs_cols * pol(cfg.tx_polarization)
    cfg.num_rx_ant = ue_rows * ue_cols * pol(cfg.rx_polarization)

    cfg.threshold_watt = 1e-20
    cfg.target_tx_index = 0
    cfg.target_rx_index = 0
    cfg.output_dir = str(out_dir)
    cfg.visualize_ray_tracing = False
    return cfg


def run_batch_rt(payload: dict, session_dir: Path, scene_xml: Path, scene_ply: Path,
                 emit: Callable[[dict], None], intg_mode: bool = False) -> dict:
    """batch RT 전체 구동. output_paths(dict) 반환.

    intg_mode=True (Intg 통합 엔진):
      - samples_per_src 고정(배치 수 비의존, 결정론적). num_samples = '배치(소스)당 샘플 수'.
      - 랜덤 배치 항상 ON (seed+batch 파생).
      - RT 루프 후 build_superset() 으로 단일 multi-TX superset NPZ 저장 +
        reshaper(to_p1a) 로 P1B/C/D 호환 뷰 생성.
    """
    _ensure_engine_on_path()
    from sionna.rt import PlanarArray  # type: ignore
    from RT_utils import get_adaptive_rx_positions, setup_multi_rx_scene  # type: ignore
    from agents import m3_scene_agent as scene_agent  # type: ignore
    from agents import m4_rt_agent as rt_agent  # type: ignore
    from agents import m5_postprocess_agent as pp  # type: ignore
    from agents import m6_viz_agent as viz_agent  # type: ignore
    from agents import m7_export_agent as export_agent  # type: ignore

    cfg = _build_config(payload, session_dir, scene_xml, scene_ply)
    output_paths: dict[str, str] = {}

    emit({"kind": "log", "message": "=" * 56})
    emit({"kind": "log", "message": ("🧩 Intg 통합 Ray Tracing 시작" if intg_mode
                                     else "🤖 Batch Ray Tracing (PARK_2 방식) 시작")})
    emit({"kind": "log", "message": f"  RT 모드: {'INTG' if intg_mode else 'BATCH_RX'} | TX {len(cfg.tx_positions)}대 | RX {len(cfg.rx_positions)}개 | batch {cfg.batch_size}"})
    emit({"kind": "log", "message": f"  주파수 {cfg.frequency/1e9:.2f} GHz | max_depth {cfg.max_depth} | samples {cfg.num_samples:,} | scattering {cfg.scattering_coefficient}"})
    # 범위 밖 재질 파라미터 보정 경고 (sionna 가 하드 에러를 내므로 _build_config 에서 미리 보정함)
    _sc_raw = getattr(cfg, "_scattering_coefficient_raw", cfg.scattering_coefficient)
    if not (0.0 < _sc_raw < 1.0):
        emit({"kind": "log", "message": f"  ⚠️ itu_scattering_coeff={_sc_raw} 는 (0,1) 범위 밖 → {cfg.scattering_coefficient} 로 보정"})
    _xpd_raw = getattr(cfg, "_xpd_coefficient_raw", cfg.xpd_coefficient)
    if not (0.0 <= _xpd_raw <= 1.0):
        emit({"kind": "log", "message": f"  ⚠️ itu_xpd_coeff={_xpd_raw} 는 [0,1] 범위 밖 → {cfg.xpd_coefficient} 로 보정"})
    emit({"kind": "log", "message": f"  안테나 TX {cfg.num_tx_rows}×{cfg.num_tx_cols}={cfg.num_tx_ant}p / RX {cfg.num_rx_rows}×{cfg.num_rx_cols}={cfg.num_rx_ant}p"})
    emit({"kind": "log", "message": "=" * 56})

    # ── Scene ──────────────────────────────────────────────────────────────
    emit({"kind": "stage_start", "stage": "Scene", "progress": 0.02})
    scene, rx_pos_3d = scene_agent.run(cfg)
    emit({"kind": "stage_end", "stage": "Scene", "message": f"RX 3D {len(rx_pos_3d)}개 배치"})

    num_total_rx = len(rx_pos_3d)
    batch_size = cfg.batch_size
    num_batches = (num_total_rx + batch_size - 1) // batch_size
    if intg_mode:
        # [Intg] samples_per_src 고정: 배치 수에 비의존(결정론적). num_samples = '배치당 샘플 수'.
        # (batch 모드의 num_samples//num_batches 분할을 폐기 → 총 RT 시간↑, 재현성↑)
        samples_per_batch = max(10000, cfg.num_samples)
    else:
        samples_per_batch = max(10000, cfg.num_samples // max(num_batches, 1))

    num_tx = len(cfg.tx_positions)
    per_tx_all_results: list = []
    per_tx_rays: list = []
    per_tx_ray_counts: list = []
    target_result = None
    target_tx = min(cfg.target_tx_index, num_tx - 1)

    VIZ_FRAC = 0.15  # 마지막 15% 는 viz/export 용으로 예약
    orig_rx_positions = list(cfg.rx_positions)
    orig_num_samples = cfg.num_samples
    scatter_bounds = _xy_bounds(orig_rx_positions)  # 라이브 커버리지 축 고정

    # Randomized RX batch sequence: RX 를 무작위로 섞어 배치를 구성한다.
    # (순차 배치 시 [RX0~49] 초반에 RSRP/Ray 가 몰려 가로줄 클러스터가 생기는 현상 완화)
    # 셔플은 내부 처리 순서에만 적용하고, 결과는 항상 원래 RX 인덱스로 되돌려 저장한다.
    _rt_cfg = payload.get("rt", {})
    random_batch = bool(_rt_cfg.get("random_batch", False)) or intg_mode  # intg 는 항상 랜덤배치 ON
    _seed = _rt_cfg.get("random_batch_seed", None)
    if intg_mode and _seed in (None, ""):
        _seed = cfg.seed  # intg: 재현성 위해 마스터 seed 로 배치 셔플 고정
    perm = list(range(num_total_rx))
    if random_batch:
        random.Random(int(_seed) if _seed not in (None, "") else None).shuffle(perm)
        emit({"kind": "log", "message": f"🔀 Randomized RX batch sequence ON (seed={_seed if _seed not in (None,'') else 'random'})"})
    ordered_rx_positions = [orig_rx_positions[i] for i in perm]

    emit({"kind": "stage_start", "stage": "RT", "message": f"TX {num_tx}대 batch RT"})
    for t, txpos in enumerate(cfg.tx_positions):
        cfg.tx_position = tuple(txpos)
        emit({"kind": "log", "message": "=" * 56})
        emit({"kind": "log", "message": f"📡 TX {t+1}/{num_tx} → ({txpos[0]:.2f}, {txpos[1]:.2f}, {txpos[2]:.2f})"})
        emit({"kind": "log", "message": "=" * 56})
        emit({"kind": "log", "message": f"📦 Batch RT: {num_total_rx} RX → {num_batches} 배치 "
                                        f"(배치당 {batch_size}, samples/batch {samples_per_batch:,})"})

        pp.reset_batch()
        all_rx_positions = list(ordered_rx_positions)
        batch_paths_list = []
        scatter_points: list = []   # 이 TX 의 누적 (x, y, rsrp|null)

        for b in range(num_batches):
            s_idx = b * batch_size
            e_idx = min(s_idx + batch_size, num_total_rx)
            batch_rx_xy = all_rx_positions[s_idx:e_idx]
            batch_global = perm[s_idx:e_idx]   # 이 배치 RX 들의 '원래' 인덱스

            for rx_name in list(scene.receivers.keys()):
                scene.remove(rx_name)
            batch_rx_3d = get_adaptive_rx_positions(cfg.map_ply_all or cfg.map_ply, batch_rx_xy, rx_height=cfg.rx_height, verbose=False)
            setup_multi_rx_scene(scene, cfg.tx_position, batch_rx_3d,
                                 num_tx_ant=cfg.num_tx_ant, num_rx_ant=cfg.num_rx_ant, verbose=False)
            scene.tx_array = PlanarArray(num_rows=cfg.num_tx_rows, num_cols=cfg.num_tx_cols,
                                         pattern=cfg.tx_pattern, polarization=cfg.tx_polarization)
            scene.rx_array = PlanarArray(num_rows=cfg.num_rx_rows, num_cols=cfg.num_rx_cols,
                                         pattern=cfg.rx_pattern, polarization=cfg.rx_polarization)
            current_tx = list(scene.transmitters.values())[0]
            for rx in scene.receivers.values():
                rx.look_at(current_tx)

            cfg.rx_positions = batch_rx_xy
            cfg.num_samples = samples_per_batch
            batch_raw, batch_paths = rt_agent.run(scene, cfg, quiet=True)
            batch_paths_list.append((batch_paths, batch_rx_3d, list(batch_global)))
            stats = pp.run_batch(raw_data=batch_raw, batch_rx_positions=batch_rx_xy,
                                 config=cfg, batch_idx=b + 1, quiet=True)

            # 라이브 커버리지 스캐터: 이번 배치 RX 의 (x,y,RSRP) 누적 후 스냅샷 기록 (추가 RT 비용 0)
            try:
                acc = getattr(pp, "_accumulated_results", [])
                base = len(acc) - len(batch_rx_xy)
                for k, xy in enumerate(batch_rx_xy):
                    r = acc[base + k] if 0 <= base + k < len(acc) else None
                    rsrp = r.get("total_rsrp_dbm") if isinstance(r, dict) else None
                    rv = (None if rsrp is None or not np.isfinite(rsrp) else round(float(rsrp), 1))
                    scatter_points.append([round(float(xy[0]), 2), round(float(xy[1]), 2), rv])
                _write_live_scatter(session_dir, {
                    "points": scatter_points,
                    "bounds": scatter_bounds,
                    "rsrp_min": stats["rsrp_min"], "rsrp_max": stats["rsrp_max"],
                    "tx_index": t, "num_tx": num_tx,
                    "total_rx": num_total_rx, "done_rx": len(scatter_points),
                    "updated": time.time(),
                })
            except Exception:
                pass

            rsrp_txt = (f"RSRP {stats['rsrp_min']:>6.1f}~{stats['rsrp_max']:>6.1f} dBm"
                        if stats["rsrp_min"] is not None else "RSRP    N/A")
            line = (f"[TX{t}] Batch RT |{_bar(b+1, num_batches)}| {b+1:>4}/{num_batches} "
                    f"({(b+1)/num_batches*100:5.1f}%)  누적 {stats['accumulated']:>5} RX | "
                    f"직전배치 유효 {stats['valid']:>2}/{stats['num_rx']:<2} | {rsrp_txt}")
            overall = ((t * num_batches + (b + 1)) / max(num_tx * num_batches, 1)) * (1.0 - VIZ_FRAC)
            # 배치 진행은 로그로 쏟지 않고 구조화 이벤트로 → 프런트가 제자리 갱신 바로 표시
            emit({
                "kind": "batch_progress",
                "tx_index": t, "num_tx": num_tx,
                "tx_pos": [float(txpos[0]), float(txpos[1]), float(txpos[2])],
                "batch": b + 1, "num_batches": num_batches,
                "batch_frac": (b + 1) / max(num_batches, 1),
                "accumulated": int(stats["accumulated"]),
                "valid": int(stats["valid"]), "batch_rx": int(stats["num_rx"]),
                "rsrp_min": stats["rsrp_min"], "rsrp_max": stats["rsrp_max"],
                "total_rx": num_total_rx,
                "progress": overall,
                "message": line,
            })

        cfg.rx_positions = orig_rx_positions
        cfg.num_samples = orig_num_samples
        result_t = pp.finalize_batch(cfg)
        processed_results = list(getattr(pp.run, "_all_results", []))
        # 무작위 배치였으면 처리 순서 → 원래 RX 인덱스 순서로 복원 (출력 정합성 보장)
        if random_batch and len(processed_results) == num_total_rx:
            restored = [None] * num_total_rx
            for k, oi in enumerate(perm):
                restored[oi] = processed_results[k]
            all_results = restored
            pp.run._all_results = restored
            # target RX 결과도 '원래' target_rx_index 기준으로 재선정 (PDP/PADP 시각화용)
            ti = min(cfg.target_rx_index, num_total_rx - 1)
            tr0 = restored[ti]
            if isinstance(tr0, dict):
                from configs.default_config import SimulationResult  # type: ignore
                result_t = SimulationResult(
                    tau=tr0["tau"], power_dbm=tr0["power_dbm"],
                    aoa_azimuth=tr0["aoa_azimuth"], aod_azimuth=tr0["aod_azimuth"],
                    total_rsrp_dbm=tr0["total_rsrp_dbm"], R_TX=tr0["R_TX"], R_RX=tr0["R_RX"],
                )
        else:
            all_results = processed_results
        per_tx_all_results.append(all_results)

        rsrp_vals = [r["total_rsrp_dbm"] for r in all_results if r["total_rsrp_dbm"] != -np.inf]
        n_valid = len(rsrp_vals)
        n_dead = len(all_results) - n_valid
        emit({"kind": "log", "message": f"📊 [PostProcess] 전체 배치 누적 완료: {len(all_results)}개 RX"})
        emit({"kind": "log", "message": f"   유효 RX: {n_valid}개 | Dead Zone: {n_dead}개"})
        if rsrp_vals:
            emit({"kind": "log", "message": f"   RSRP 범위: {min(rsrp_vals):.1f} ~ {max(rsrp_vals):.1f} dBm"})
        tr = all_results[cfg.target_rx_index] if all_results else None
        if tr and tr["total_rsrp_dbm"] != -np.inf:
            emit({"kind": "log", "message": f"📍 Target RX {cfg.target_rx_index}: "
                                            f"유효 경로 {len(tr['tau'])}개 | RSRP: {tr['total_rsrp_dbm']:.2f} dBm"})

        rays_by_rx, ray_counts = export_agent.extract_rays_batch(
            batch_paths_list, num_total_rx, cfg.viz.max_rays_per_rx)
        per_tx_rays.append(rays_by_rx)
        per_tx_ray_counts.append(ray_counts)
        if t == target_tx:
            target_result = result_t
    emit({"kind": "stage_end", "stage": "RT", "message": f"TX {num_tx}대 RT 완료"})

    if target_result is None:
        from configs.default_config import SimulationResult  # type: ignore
        target_result = SimulationResult()
    cfg.tx_position = tuple(cfg.tx_positions[target_tx])

    # ── Output 1 (channel_data) ─────────────────────────────────────────────
    # Intg 모드면 먼저 superset 을 만들어 rx_valid_mask(dead/rt_fail)를 channel_data 에도 싣는다.
    intg_superset = None
    if intg_mode:
        emit({"kind": "stage_start", "stage": "Intg", "progress": 1.0 - VIZ_FRAC})
        try:
            from .intg.intg_writer import build_superset
            from .intg import superset_schema as S
            intg_superset = build_superset(
                per_tx_all_results=per_tx_all_results,
                tx_positions=cfg.tx_positions,
                rx_positions_3d=rx_pos_3d,
                frequency_hz=cfg.frequency,
                num_tx_ant=cfg.num_tx_ant,
                num_rx_ant=cfg.num_rx_ant,
                target_tx_index=cfg.target_tx_index,
                target_rx_index=cfg.target_rx_index,
                rng_seed=cfg.seed,
                max_rays_cap=cfg.viz.max_rays_per_rx,
            )
            problems = S.validate(intg_superset)
            if problems:
                emit({"kind": "log", "message": "⚠️ superset 검증 경고: " + "; ".join(problems)})
        except Exception as exc:
            emit({"kind": "log", "message": f"⚠️ Intg build_superset 실패(계속): {type(exc).__name__}: {exc}"})
            emit({"kind": "log", "message": traceback.format_exc()})

    emit({"kind": "stage_start", "stage": "Output1", "progress": 1.0 - VIZ_FRAC})
    _mask = intg_superset.get("rx_valid_mask") if intg_superset is not None else None
    out1 = pp.save_output1_multi(per_tx_all_results, cfg.output_dir_channel, cfg.map_title,
                                 config=cfg, rx_positions_3d=rx_pos_3d, rx_valid_mask=_mask)
    output_paths["batch_channel_npz"] = out1
    emit({"kind": "stage_end", "stage": "Output1", "message": Path(out1).name})

    # ── Intg: superset NPZ + reshaper 뷰 (P1A 호환) ─────────────────────────
    if intg_mode and intg_superset is not None:
        try:
            from .intg.reshapers.to_p1a import superset_to_p1a
            from .intg import superset_schema as S

            sup = intg_superset
            ts = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
            intg_dir = Path(cfg.output_dir) / "Intg_Results"
            intg_dir.mkdir(parents=True, exist_ok=True)

            # 단일 multi-TX superset NPZ (canonical 출력, 덮어쓰기 방지 타임스탬프)
            sup_path = intg_dir / f"superset_{cfg.map_title}_{ts}.npz"
            np.savez_compressed(sup_path, **sup)
            output_paths["intg_superset_npz"] = str(sup_path)

            # P1A 호환 뷰 (TX별 단일-TX 파일) → P1B/C/D 가 무수정으로 소비
            n_tx = int(np.asarray(sup["tx_positions"]).shape[0])
            freq_ghz = float(np.asarray(sup["frequency_ghz"]).item())
            for t in range(n_tx):
                p1a = superset_to_p1a(sup, tx_index=t, area_index=t + 1)
                p1a_path = intg_dir / f"Area{t+1}_{freq_ghz:.1f}GHz_Rays_ALL_RXs.npz"
                np.savez(p1a_path, **p1a)
                output_paths[f"intg_p1a_tx{t}"] = str(p1a_path)

            mask = np.asarray(sup["rx_valid_mask"])
            n_valid = int((mask == S.RX_VALID).sum())
            n_dead = int((mask == S.RX_DEAD).sum())
            n_fail = int((mask == S.RX_RT_FAIL).sum())
            _co = sup.get("cov_omitted")
            if _co is not None and int(np.asarray(_co).reshape(-1)[0]) == 1:
                emit({"kind": "log", "message": f"   ⚠️ 공분산 R_TX 생략 (TX안테나 {cfg.num_tx_ant}p × RX {mask.size} 가 커서 superset OOM 방지) "
                                                f"— channel_data per-pair 에는 유지됨"})
            emit({"kind": "log", "message": f"   superset shape: T={n_tx} R={mask.size} "
                                            f"K={int(np.asarray(sup['max_paths']).item())} "
                                            f"P={int(np.asarray(sup['max_rays']).item())}"})
            emit({"kind": "stage_end", "stage": "Intg",
                  "message": f"superset {sup_path.name} | P1A뷰 {n_tx}개 | "
                             f"RX valid {n_valid}/dead {n_dead}/fail {n_fail}"})
        except Exception as exc:
            emit({"kind": "log", "message": f"⚠️ Intg superset 저장 실패(계속): {type(exc).__name__}: {exc}"})
            emit({"kind": "log", "message": traceback.format_exc()})

    # ── Viz (RSRP/LoS/PDP/PADP/공분산/rx_positions) ─────────────────────────
    emit({"kind": "stage_start", "stage": "Viz", "progress": 1.0 - VIZ_FRAC * 0.6})
    try:
        viz_files = viz_agent.run(target_result, scene, cfg, rx_pos_3d,
                                  per_tx_all_results=per_tx_all_results,
                                  tx_positions=cfg.tx_positions)
        for f in viz_files or []:
            output_paths[f"viz_{Path(f).stem}"] = f
        emit({"kind": "stage_end", "stage": "Viz", "message": f"{len(viz_files or [])}개 이미지"})
    except Exception as exc:
        emit({"kind": "log", "message": f"⚠️ Viz 실패(계속): {type(exc).__name__}: {exc}"})

    # ── Hitmap (TX별) ───────────────────────────────────────────────────────
    try:
        os.makedirs(cfg.output_dir_viz, exist_ok=True)
        for t, ray_counts in enumerate(per_tx_ray_counts):
            cfg.tx_position = tuple(cfg.tx_positions[t])
            viz_agent._plot_ray_hitmap_from_counts(
                ray_counts=ray_counts, config=cfg, rx_pos_3d=rx_pos_3d,
                output_dir=cfg.output_dir_viz, title_suffix=f"TX{t} Data",
                filename=f"hitmap_data_TX{t}.png", cmap="YlOrRd", max_clip=None)
        cfg.tx_position = tuple(cfg.tx_positions[target_tx])
    except Exception as exc:
        emit({"kind": "log", "message": f"⚠️ Hitmap 실패(계속): {type(exc).__name__}: {exc}"})

    # ── Export (USD/OBJ) ────────────────────────────────────────────────────
    emit({"kind": "stage_start", "stage": "Export", "progress": 1.0 - VIZ_FRAC * 0.3})
    try:
        exp_files = export_agent.run_multi(per_tx_rays, cfg, rx_pos_3d, tx_positions=cfg.tx_positions)
        for f in exp_files or []:
            output_paths[f"export_{Path(f).stem}"] = f
        emit({"kind": "stage_end", "stage": "Export", "message": f"{len(exp_files or [])}개 파일"})
    except Exception as exc:
        emit({"kind": "log", "message": f"⚠️ Export 실패(계속): {type(exc).__name__}: {exc}"})

    emit({"kind": "progress", "progress": 1.0})
    emit({"kind": "log", "message": f"🎉 Batch RT 완료 — 출력: {cfg.output_dir}"})
    return output_paths
