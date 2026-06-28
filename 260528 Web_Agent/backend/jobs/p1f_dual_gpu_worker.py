"""P1F Marginal CCM 2×GPU worker (subprocess-safe).

실행:
  cd "260528 Web_Agent" && source .venv-webagent/bin/activate
  python -m backend.jobs.p1f_dual_gpu_worker --gpu 0 --chunk-file session/.p1f_dualgpu/chunk_0.json

실행 환경 (2026-06-04): Python 3.10, dclserver78, tensorflow 2.21, sionna 1.2.2
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_WEB_AGENT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = _WEB_AGENT_ROOT.parent
ORIG_P1F = REPO_ROOT / "251218 E_MIMO_BM" / "P1F_Rays_to_Marginal_CCM_2510v1.py"
SOURCE_FOLDER = "251218 E_MIMO_BM"


def _num_gpus() -> int:
    want = int(os.environ.get("WEBAGENT_P1F_NUM_GPUS", "2"))
    try:
        out = subprocess.check_output(["nvidia-smi", "-L"], text=True, timeout=10)
        avail = sum(1 for line in out.splitlines() if line.strip().startswith("GPU"))
        return max(1, min(want, avail))
    except Exception:
        return 1


def list_pending_combos(session_dir: Path) -> list[tuple[int, float, int]]:
    p1b_dir = session_dir / "P1B_Valid_Results"
    out_dir = session_dir / "P1F_Marginal_CCM_Results"
    if not p1b_dir.exists():
        raise FileNotFoundError(f"P1B_Valid_Results 없음: {p1b_dir}")

    pending: list[tuple[int, float, int]] = []
    for npz_path in sorted(p1b_dir.glob("*_Valid_RXs.npz")):
        m = re.search(r"Area(\d+)_([\d.]+)GHz", npz_path.name)
        if not m:
            continue
        area_index = int(m.group(1))
        fc = float(m.group(2))
        with np.load(npz_path, allow_pickle=True) as d:
            if "rx_indices" in d.files:
                rx_indices = np.asarray(d["rx_indices"], dtype=np.int64).reshape(-1)
            else:
                n = int(d["tau"].shape[0])
                rx_indices = np.arange(1, n + 1, dtype=np.int64)
        for rx in rx_indices:
            rx_i = int(rx)
            fname = f"Area{area_index}_{fc}GHz_RX{rx_i}_Marginal_CCM.npz"
            if not (out_dir / fname).exists():
                pending.append((area_index, fc, rx_i))
    return pending


def _split_round_robin(items: list, n: int) -> list[list]:
    buckets: list[list] = [[] for _ in range(n)]
    for i, item in enumerate(items):
        buckets[i % n].append(item)
    return buckets


def _apply_config_overrides(cfg_cls: type, overrides: dict[str, Any]) -> None:
    if not overrides or getattr(cfg_cls, "_webagent_patched", False):
        return
    orig_init = cfg_cls.__init__
    orig_setattr = cfg_cls.__setattr__
    cfg_cls._webagent_overrides = dict(overrides)

    def _locked_setattr(obj, name, value, _orig=orig_setattr):
        if name in getattr(obj, "_webagent_locked", ()):
            return
        _orig(obj, name, value)

    def _patched_init(self, *args, _orig_init=orig_init, _cls=cfg_cls, **kwargs):
        ovs = getattr(_cls, "_webagent_overrides", {})
        object.__setattr__(self, "_webagent_locked", set(ovs.keys()))
        for k, v in ovs.items():
            object.__setattr__(self, k, v)
        _orig_init(self, *args, **kwargs)
        object.__setattr__(self, "_webagent_locked", set())
        for k, v in ovs.items():
            setattr(self, k, v)
        for hook in ("detect_p1a_data", "detect_p1b_data", "detect_p1f_data", "rescan", "refresh", "_rescan"):
            fn = getattr(self, hook, None)
            if callable(fn):
                try:
                    fn()
                except Exception:
                    pass

    cfg_cls.__init__ = _patched_init
    cfg_cls.__setattr__ = _locked_setattr
    cfg_cls._webagent_patched = True


def _import_p1f_original() -> Any:
    src = REPO_ROOT / SOURCE_FOLDER
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    name = f"webagent_p1f_worker_{os.environ.get('CUDA_VISIBLE_DEVICES', 'x')}"
    spec = importlib.util.spec_from_file_location(name, ORIG_P1F)
    if spec is None or spec.loader is None:
        raise ImportError(f"P1F 원본 로드 실패: {ORIG_P1F}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _process_single_rx(mod: Any, config: Any, ccm_manager: Any, area_index: int, fc: float, RX_index: int) -> bool:
    tf = mod.tf
    np_mod = mod.np
    PI = mod.PI
    PanelArray = mod.PanelArray
    Rays = mod.Rays
    Topology = mod.Topology
    ChCoeGen = mod.ChCoeGen
    MarginalCCM_Engine = mod.MarginalCCM_Engine

    if ccm_manager.ccm_file_exists(area_index, fc, RX_index):
        return True

    carrier_frequency = fc * 10**9
    ray_data = config.load_p1b_ray_data(area_index, fc, RX_index)
    if ray_data is None:
        print(f"[GPU{os.environ.get('CUDA_VISIBLE_DEVICES')}] ray load fail RX{RX_index}")
        return False

    ray_aoa_rad = tf.convert_to_tensor(np_mod.deg2rad(ray_data["phi_r_deg"]))
    ray_aod_rad = tf.convert_to_tensor(np_mod.deg2rad(ray_data["phi_t_deg"]))
    ray_zoa_rad = tf.convert_to_tensor(np_mod.deg2rad(ray_data["theta_r_deg"]))
    ray_zod_rad = tf.convert_to_tensor(np_mod.deg2rad(ray_data["theta_t_deg"]))
    ray_power = tf.convert_to_tensor(ray_data["power"])
    ray_delay = tf.convert_to_tensor(ray_data["tau"])

    ArrayTX = PanelArray(
        num_rows_per_panel=config.TX_Array["num_rows_per_panel"],
        num_cols_per_panel=config.TX_Array["num_cols_per_panel"],
        num_rows=config.TX_Array["num_rows"],
        num_cols=config.TX_Array["num_cols"],
        polarization=config.TX_Array["polarization"],
        polarization_type=config.TX_Array["polarization_type"],
        antenna_pattern=config.TX_Array["antenna_pattern"],
        carrier_frequency=carrier_frequency,
        panel_vertical_spacing=config.TX_Array["panel_vertical_spacing"],
        panel_horizontal_spacing=config.TX_Array["panel_horizontal_spacing"],
    )
    ArrayRX = PanelArray(
        num_rows_per_panel=config.RX_Array["num_rows_per_panel"],
        num_cols_per_panel=config.RX_Array["num_cols_per_panel"],
        num_rows=config.RX_Array["num_rows"],
        num_cols=config.RX_Array["num_cols"],
        polarization=config.RX_Array["polarization"],
        polarization_type=config.RX_Array["polarization_type"],
        antenna_pattern=config.RX_Array["antenna_pattern"],
        carrier_frequency=carrier_frequency,
    )

    OFDM_ChGen = ChCoeGen(carrier_frequency, config.OFDM_SCS, ArrayTX, ArrayRX, False)
    batch_size = config.batch_size
    N_UE = config.num_rx
    N_BS = config.N_BS

    ray_xpr = 10 ** (
        tf.random.normal(
            shape=[batch_size, 1, N_UE, 1, ray_aoa_rad.shape[-1]],
            mean=config.mean_xpr,
            stddev=config.stddev_xpr,
        )
        / 10
    )
    ray_pdap = Rays(
        delays=ray_delay, powers=ray_power,
        aoa=ray_aoa_rad, aod=ray_aod_rad, zoa=ray_zoa_rad, zod=ray_zod_rad, xpr=ray_xpr,
    )

    speed = tf.abs(
        tf.random.normal(
            shape=[batch_size, N_UE, 1],
            mean=config.Topology_Statistics["velocities_mean"],
            stddev=config.Topology_Statistics["velocities_stddev"],
            dtype=tf.float32,
        )
    )
    angle = tf.random.uniform(shape=[batch_size, N_UE, 1], minval=0.0, maxval=2 * PI, dtype=tf.float32)
    velocities = tf.concat([speed * tf.cos(angle), speed * tf.sin(angle), tf.zeros_like(speed)], axis=-1)

    topology = Topology(
        velocities,
        config.Topology_Statistics["moving_end"],
        tf.zeros([batch_size, N_BS, N_UE]),
        tf.zeros([batch_size, N_BS, N_UE]),
        tf.zeros([batch_size, N_BS, N_UE]),
        tf.zeros([batch_size, N_BS, N_UE]),
        tf.random.uniform(
            shape=[batch_size, N_BS, N_UE],
            minval=config.Topology_Statistics["los_minval"],
            maxval=config.Topology_Statistics["los_maxval"],
            dtype=tf.int32,
        ) > 0,
        tf.ones([1, N_BS, N_UE]),
        tf.constant([[[
            np_mod.deg2rad(config.TX_Orientation["azimuth_deg"]),
            np_mod.deg2rad(config.TX_Orientation["downtilt_deg"]),
            np_mod.deg2rad(config.TX_Orientation["roll_deg"]),
        ]]], dtype=tf.float32),
        tf.constant([[[
            np_mod.deg2rad(config.RX_Orientation["azimuth_deg"]),
            np_mod.deg2rad(config.RX_Orientation["elevation_deg"]),
            np_mod.deg2rad(config.RX_Orientation["roll_deg"]),
        ]]], dtype=tf.float32),
    )

    print(f"\n=== P1F RX{RX_index} [GPU {os.environ.get('CUDA_VISIBLE_DEVICES')}] ===")
    ccm_engine = MarginalCCM_Engine(config, OFDM_ChGen, topology, ray_pdap)
    t0 = time.time()
    for static_idx in range(1, config.static_ch_realizations + 1):
        ccm_engine.process_static_realization(static_idx - 1)
        if static_idx % 64 == 0:
            print(f"  {static_idx}/{config.static_ch_realizations}", end=" ", flush=True)
    print(f"\n누적 {time.time()-t0:.1f}s")

    ccm_data = ccm_engine.finalize_marginal_ccm(area_index, fc, RX_index)
    ok = bool(ccm_data)
    if ok:
        ccm_manager.save_rx_marginal_ccm(ccm_data)
    del ccm_engine, ray_data
    import gc
    gc.collect()
    tf.keras.backend.clear_session()
    return ok


def run_worker(gpu_id: int, session_dir: Path, combos: list, overrides: dict) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["WEBAGENT_MANAGED_RUNTIME"] = "1"
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

    os.chdir(session_dir)
    mod = _import_p1f_original()
    _apply_config_overrides(mod.P1F_Config, overrides or {})

    import gc
    gc.collect()
    mod.tf.keras.backend.clear_session()

    config = mod.P1F_Config()
    ccm_manager = mod.MarginalCCM_Manager(config)
    total = len(combos)
    t0 = time.time()
    print(f"[P1F worker] GPU={gpu_id}  RX={total}  session={session_dir}")

    for i, (area_index, fc, rx_i) in enumerate(combos, 1):
        _process_single_rx(mod, config, ccm_manager, int(area_index), float(fc), int(rx_i))
        if i % 10 == 0 or i == total:
            el = time.time() - t0
            print(f"[GPU{gpu_id}] {i}/{total}  ETA {(total-i)*el/max(i,1)/60:.1f}min")


def _session_path_overrides(session_dir: Path) -> dict[str, str]:
    """원본 P1F는 ``__file__`` 기준 경로 — worker는 세션 절대경로로 override."""

    session_dir = Path(session_dir).resolve()
    return {
        "P1B_INPUT_DIR": str(session_dir / "P1B_Valid_Results"),
        "P1F_CCM_OUTPUT_DIR": str(session_dir / "P1F_Marginal_CCM_Results"),
    }


def run_dual_gpu_orchestrator(session_dir: Path, config_overrides: dict | None = None) -> None:
    """2×GPU subprocess 오케스트레이터 (spawn pickle 이슈 회피)."""

    session_dir = Path(session_dir).resolve()
    pending = list_pending_combos(session_dir)
    if not pending:
        print("[P1F dual-GPU] 미처리 RX 없음")
        return

    n_gpu = _num_gpus()
    chunks = _split_round_robin(pending, n_gpu)
    work = session_dir / ".p1f_dualgpu"
    work.mkdir(parents=True, exist_ok=True)

    overrides = {**_session_path_overrides(session_dir), **dict(config_overrides or {})}
    print("=" * 72)
    print(f"[P1F dual-GPU] session={session_dir}")
    print(f"  pending={len(pending)}  workers={n_gpu}")
    for i, ch in enumerate(chunks):
        print(f"    GPU {i}: {len(ch)} RX")
    print("=" * 72)

    py = sys.executable
    root = _WEB_AGENT_ROOT
    procs: list[tuple[int, subprocess.Popen]] = []

    for gpu_id in range(n_gpu):
        if not chunks[gpu_id]:
            continue
        chunk_file = work / f"chunk_gpu{gpu_id}.json"
        chunk_file.write_text(
            json.dumps({"combos": chunks[gpu_id], "overrides": overrides}, ensure_ascii=False),
            encoding="utf-8",
        )
        env = os.environ.copy()
        env["WEBAGENT_USE_GPU"] = "1"
        env["WEBAGENT_P1F_NUM_GPUS"] = "1"
        cmd = [
            py, "-m", "backend.jobs.p1f_dual_gpu_worker",
            "--gpu", str(gpu_id),
            "--session-dir", str(session_dir),
            "--chunk-file", str(chunk_file),
        ]
        log_f = open(work / f"worker_gpu{gpu_id}.log", "w", encoding="utf-8")
        p = subprocess.Popen(cmd, cwd=str(root), env=env, stdout=log_f, stderr=subprocess.STDOUT)
        procs.append((gpu_id, p))
        print(f"[P1F dual-GPU] started worker GPU{gpu_id} PID={p.pid}")

    failures = []
    for gpu_id, p in procs:
        rc = p.wait()
        if rc != 0:
            failures.append((gpu_id, rc))
            print(f"[P1F dual-GPU] GPU{gpu_id} exit={rc}  log: {work}/worker_gpu{gpu_id}.log")
        else:
            print(f"[P1F dual-GPU] GPU{gpu_id} done")

    if failures:
        raise RuntimeError(f"P1F dual-GPU workers failed: {failures}")

    remain = len(list_pending_combos(session_dir))
    print(f"[P1F dual-GPU] orchestrator done. remaining RX: {remain}")


def _cli() -> None:
    ap = argparse.ArgumentParser(description="P1F dual-GPU worker process")
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--session-dir", type=Path, required=True)
    ap.add_argument("--chunk-file", type=Path, required=True)
    args = ap.parse_args()

    payload = json.loads(args.chunk_file.read_text(encoding="utf-8"))
    combos = [tuple(x) for x in payload["combos"]]
    overrides = payload.get("overrides") or {}
    run_worker(args.gpu, args.session_dir.resolve(), combos, overrides)


if __name__ == "__main__":
    _cli()
