"""base_adapter.py — 25* 모듈을 Web Agent에서 호출하기 위한 공통 베이스.

설계 원칙
---------
1. 25* 원본 폴더는 절대 무수정. forked_25x/에 fork가 있으면 그것을 우선 import.
2. 각 P1*은 자체적인 `P1X_Config` 클래스 + `Pipeline.execute_main()` 패턴을 가짐.
3. 어댑터는 cwd를 세션 디렉토리로 변경한 뒤 Config의 *_SAVE_DIR/UNIFIED_SAVE_DIR을
   override 하여 결과가 세션 폴더에 떨어지게 한다.
4. 25* 모듈은 한 번 import되면 module-level Config 인스턴스를 가지므로,
   동일 process에서 여러 어댑터를 순차 호출할 때는 override → 실행 → 복원 패턴 사용.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable


# ---------------------------------------------------------------------------
# Stage Registry — 각 P1*의 모듈 위치 / Config 클래스 이름 / 결과 폴더
# ---------------------------------------------------------------------------
# twin_minji 루트 (호스트별 절대경로 하드코딩 대신 Web_Agent 기준 자동 해석)
_WEB_AGENT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = _WEB_AGENT_ROOT.parent
FORKED_DIR = Path(__file__).resolve().parent.parent / "forked_25x"


@dataclass(frozen=True)
class StageMeta:
    """단일 P1* stage 메타데이터."""

    stage_id: str                        # 'P1A' ~ 'P1Q'
    source_folder: str                   # 25* 폴더 명 (REPO_ROOT/<source_folder>)
    module_filename: str                 # 'P1A_RT_to_Rays_2509v6.py'
    config_class: str                    # 'P1A_Config'
    result_dirname: str                  # 결과 디렉토리 이름 (cwd 기준)
    forked_module: str | None = None     # forked_25x 내 모듈 이름 (없으면 None)


STAGE_REGISTRY: dict[str, StageMeta] = {
    "P1A": StageMeta(
        stage_id="P1A",
        source_folder="251218 E_MIMO_BM",
        module_filename="P1A_RT_to_Rays_2509v6.py",
        config_class="P1A_Config",
        result_dirname="P1A_RT_Results",
        forked_module="P1A_RT_to_Rays_2509v6_web",
    ),
    "P1B": StageMeta(
        stage_id="P1B",
        source_folder="251218 E_MIMO_BM",
        module_filename="P1B_Valid_RX_Filter_2509v1.py",
        config_class="P1B_Config",
        result_dirname="P1B_Valid_Results",  # P1B는 자체 P1B_Valid_Results 폴더 생성
    ),
    "P1C": StageMeta(
        stage_id="P1C",
        source_folder="250924 python",
        module_filename="P1C_Rays_to_AE_OFDM_Ch_2509v1.py",
        config_class="P1C_Config",
        result_dirname="P1C_AE_OFDM_Ch_Results",
    ),
    "P1D": StageMeta(
        stage_id="P1D",
        source_folder="251004_Ch_Separ_(CorrRician)",
        module_filename="P1D_Rays_to_AE_Separability_2510v3.py",
        config_class="P1D_Config",
        result_dirname="P1D_Separability_Results",
    ),
    "P1F": StageMeta(
        stage_id="P1F",
        source_folder="251218 E_MIMO_BM",
        module_filename="P1F_Rays_to_Marginal_CCM_2510v1.py",
        config_class="P1F_Config",
        result_dirname="P1F_Marginal_CCM_Results",
        forked_module="P1F_Rays_to_Marginal_CCM_2510v2_dualgpu",
    ),
    "P1G": StageMeta(
        stage_id="P1G",
        source_folder="251218 E_MIMO_BM",
        module_filename="P1G_Rays_to_CouplingMat_2510v1.py",
        config_class="P1G_Config",
        result_dirname="P1G_CouplingMat_Results",
    ),
    "P1H": StageMeta(
        stage_id="P1H",
        source_folder="251218 E_MIMO_BM",
        module_filename="P1H_Rays_to_MeanCh_2510v1.py",
        config_class="P1H_Config",
        result_dirname="P1H_MeanCh_Results",
    ),
    "P1I": StageMeta(
        stage_id="P1I",
        source_folder="251218 E_MIMO_BM",
        module_filename="P1I_Weichsel_Chunk_2510v1.py",
        config_class="P1I_Config",
        result_dirname="P1I_Weichsel_Chunk_Results",
    ),
    "P1J": StageMeta(
        stage_id="P1J",
        source_folder="251020 BM QIE (Weichselberger)",
        module_filename="P1J_Weichsel_SU_MIMO_Capacity_2510v1.py",
        config_class="P1J_Config",
        result_dirname="P1J_Capacity_Results",
    ),
    "P1L": StageMeta(
        stage_id="P1L",
        source_folder="251020 BM QIE (Weichselberger)",
        module_filename="P1L_BeamMgmt_SU_MIMO_2510v3.py",
        config_class="P1L_Config",
        result_dirname="P1L_BM_Greedy_Results",
    ),
    "P1M": StageMeta(
        stage_id="P1M",
        source_folder="251020 BM QIE (Weichselberger)",
        module_filename="P1M_BM_QIE_2510v1.py",
        config_class="P1M_Config",
        result_dirname="P1M_QIE_Results",
    ),
    "P1N": StageMeta(
        stage_id="P1N",
        source_folder="251020 BM QIE (Weichselberger)",
        module_filename="P1N_Analyze_PADP_DFT_2510v1.py",
        config_class="P1N_Config",
        result_dirname="P1N_PADP_DFT_Results",
    ),
    "P1O": StageMeta(
        stage_id="P1O",
        source_folder="251028 BM_SU_MIMO_Uplink",
        module_filename="P1O_PADP_to_BM_2510v6.py",
        config_class="P1O_Config",
        result_dirname="P1O_Uplink_BM_Results",
    ),
    "P1P": StageMeta(
        stage_id="P1P",
        source_folder="251218 E_MIMO_BM",
        module_filename="P1P_BM_SWOMP_2511v5.py",
        config_class="P1P_Config",
        result_dirname="P1P_SWOMP_Results",
    ),
    "P1Q": StageMeta(
        stage_id="P1Q",
        source_folder="251218 E_MIMO_BM",
        module_filename="P1Q_Selected_Beams_Pattern_Plot_v6.py",
        config_class="P1Q_Config",
        result_dirname="P1Q_BeamPattern_Results",
    ),
}


# ---------------------------------------------------------------------------
# 동적 모듈 import (fork 우선)
# ---------------------------------------------------------------------------
def _import_from_path(module_name: str, file_path: Path):
    """파일 경로에서 직접 모듈 import."""

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot build spec for {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_TF_THREADS_PATCHED = False


def _patch_tf_threading_once() -> None:
    """TF threading API를 no-op으로 패치.

    25* 모듈은 module-level에서 `tf.config.threading.set_inter_op_parallelism_threads(0)` 등을
    호출하지만 TF가 이미 init된 후에 호출하면 RuntimeError가 난다. 같은 process에서 여러
    25* 모듈을 import해야 하는 Web Agent에서는 첫 import 이후엔 모두 실패하므로
    no-op으로 패치한다.
    """

    global _TF_THREADS_PATCHED
    if _TF_THREADS_PATCHED:
        return
    try:
        import tensorflow as tf
        tf.config.threading.set_inter_op_parallelism_threads = lambda *a, **kw: None
        tf.config.threading.set_intra_op_parallelism_threads = lambda *a, **kw: None
        _TF_THREADS_PATCHED = True
    except Exception:
        pass


def load_stage_module(meta: StageMeta):
    """P1* 모듈을 fork 우선 → 원본 25* 순서로 import."""

    try:
        from backend.jobs.runtime_env import init_runtime
        init_runtime()  # 기본 GPU ON (runtime_env.apply_webagent_gpu_env)
    except Exception:
        pass
    _patch_tf_threading_once()

    if meta.forked_module:
        fork_path = FORKED_DIR / f"{meta.forked_module}.py"
        if fork_path.exists():
            return _import_from_path(f"webagent_fork_{meta.stage_id}", fork_path)

    orig_path = REPO_ROOT / meta.source_folder / meta.module_filename
    if not orig_path.exists():
        raise FileNotFoundError(
            f"[{meta.stage_id}] source not found: {orig_path}. "
            f"Check that the 25* folder still exists."
        )
    return _import_from_path(f"webagent_orig_{meta.stage_id}", orig_path)


# ---------------------------------------------------------------------------
# cwd / sys.path 컨텍스트
# ---------------------------------------------------------------------------
@contextmanager
def session_workdir(session_dir: Path, source_folder: str):
    """25* 모듈의 NPZ 입출력이 세션 폴더 기준으로 일어나도록 cwd 변경.

    Some 25* 모듈은 ".." 상대경로로 sibling 폴더의 NPZ를 찾기도 하므로
    sys.path에 25* source folder도 추가.
    """

    prev_cwd = Path.cwd()
    prev_path = list(sys.path)
    src = REPO_ROOT / source_folder
    try:
        session_dir = Path(session_dir)
        session_dir.mkdir(parents=True, exist_ok=True)
        os.chdir(session_dir)
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
        yield session_dir
    finally:
        os.chdir(prev_cwd)
        sys.path[:] = prev_path


# ---------------------------------------------------------------------------
# Base adapter
# ---------------------------------------------------------------------------
class BaseAdapter:
    """공통 어댑터.

    사용 예
    -------
    adapter = BaseAdapter("P1A")
    adapter.run(
        session_dir=Path("sessions/abc"),
        config_overrides={"WEB_OVERRIDE_SCENE_PATH": "...", "MAX_DEPTH": 5},
    )
    """

    def __init__(self, stage_id: str):
        if stage_id not in STAGE_REGISTRY:
            raise KeyError(f"Unknown stage: {stage_id}")
        self.meta = STAGE_REGISTRY[stage_id]

    def run(
        self,
        session_dir: Path,
        config_overrides: dict[str, Any] | None = None,
        post_hook: Callable[[Any, Path], None] | None = None,
    ) -> dict:
        """단일 P1* stage 실행.

        Parameters
        ----------
        session_dir : Path
            세션 디렉토리. 25* 모듈은 이 폴더를 cwd로 실행되어 결과 폴더가 여기에 생긴다.
        config_overrides : dict | None
            모듈의 Config 클래스 인스턴스에 setattr할 값들.
        post_hook : Callable | None
            stage 실행 후 호출. signature (module, session_dir).
        """

        module = load_stage_module(self.meta)

        # 25* 모듈은 대부분 Config.__init__ 안에서 `os.path.dirname(os.path.abspath(__file__))`로
        # 입출력 디렉토리를 잡는다. 이를 세션 디렉토리로 redirect 하기 위해 모듈의
        # __file__ 을 세션 폴더 안의 proxy 경로로 바꿔준다.
        session_dir = Path(session_dir)
        session_dir.mkdir(parents=True, exist_ok=True)
        proxy_file = session_dir / f"_{self.meta.stage_id.lower()}_proxy.py"
        if not proxy_file.exists():
            proxy_file.write_text("# Web Agent proxy module file. Do not delete.\n")
        original_file = getattr(module, "__file__", None)
        module.__file__ = str(proxy_file)

        # config 인스턴스 찾기 (예: P1A_Config의 인스턴스 = p1a_config)
        cfg_instance = self._find_config_instance(module)
        applied: dict[str, Any] = {}
        if config_overrides:
            for k, v in config_overrides.items():
                if hasattr(cfg_instance, k):
                    applied[k] = getattr(cfg_instance, k)
                setattr(cfg_instance, k, v)

        # 25*는 main() 내부에서 새로운 Config() 인스턴스를 만드는 경우가 많아
        # 위의 setattr만으로는 부족하다. 해당 Config 클래스의 __init__을 monkey-patch 해서
        # 새로 만들어지는 인스턴스에도 override 가 적용되도록 한다.
        # 25* 의 __init__ 안에서는 `self.target_areas = [1]` 같은 하드코딩 할당 + 그 직후 scan 호출
        # 패턴이 흔하므로, 단순히 __init__ 종료 후 setattr 하면 이미 scan 결과가 잘못 산출됨.
        # → 클래스의 __setattr__ 을 일시적으로 wrapping 하여, override key 에 대한 할당을 무시.
        cfg_cls = getattr(module, self.meta.config_class, None)
        if cfg_cls is not None:
            cfg_cls._webagent_overrides = dict(config_overrides or {})
            if not getattr(cfg_cls, "_webagent_patched", False):
                orig_init = cfg_cls.__init__
                orig_setattr = cfg_cls.__setattr__

                def _locked_setattr(obj, name, value, _orig=orig_setattr):
                    if name in getattr(obj, "_webagent_locked", ()):
                        return
                    _orig(obj, name, value)

                def _patched_init(self, *args, _orig_init=orig_init, _cls=cfg_cls, **kwargs):
                    overrides = getattr(_cls, "_webagent_overrides", {})
                    # pre-set overrides (lock them so __init__ can't overwrite)
                    object.__setattr__(self, "_webagent_locked", set(overrides.keys()))
                    for k, v in overrides.items():
                        object.__setattr__(self, k, v)
                    _orig_init(self, *args, **kwargs)
                    # unlock and re-apply overrides (covers attrs that __init__ creates later)
                    object.__setattr__(self, "_webagent_locked", set())
                    for k, v in overrides.items():
                        setattr(self, k, v)
                    # detect_* 메서드 호출하는 stage 도 있으므로 재스캔 시도
                    for hook in ("detect_p1a_data", "detect_p1b_data", "detect_p1f_data",
                                 "detect_p1g_data", "detect_p1h_data", "detect_p1i_data",
                                 "rescan", "refresh", "_rescan"):
                        fn = getattr(self, hook, None)
                        if callable(fn):
                            try:
                                fn()
                            except Exception:  # noqa: BLE001
                                pass

                cfg_cls.__init__ = _patched_init
                cfg_cls.__setattr__ = _locked_setattr
                cfg_cls._webagent_patched = True

        with session_workdir(session_dir, self.meta.source_folder):
            # 25* 모듈 일부 (P1P 등) 가 argparse 를 쓰기 때문에 호출 측의 sys.argv 가
            # 노출되면 충돌난다. 25* 실행 동안 sys.argv 를 빈 인자로 임시 교체.
            prev_argv = sys.argv
            sys.argv = [str(self.meta.module_filename)]
            try:
                # 우선순위: web_run_pipeline (fork) → Pipeline.execute_main() (P1A) → main() (P1B~P1Q)
                if hasattr(module, "web_run_pipeline"):
                    module.web_run_pipeline(config_overrides or {})
                else:
                    Pipeline = getattr(module, "Pipeline", None)
                    main_fn = getattr(module, "main", None)
                    if Pipeline is not None and hasattr(Pipeline, "execute_main"):
                        Pipeline.execute_main()
                    elif callable(main_fn):
                        main_fn()
                    else:
                        raise RuntimeError(
                            f"[{self.meta.stage_id}] module has neither Pipeline.execute_main() nor main()."
                        )
            finally:
                sys.argv = prev_argv

        if post_hook is not None:
            post_hook(module, Path(session_dir))

        if original_file is not None:
            module.__file__ = original_file

        return {
            "stage": self.meta.stage_id,
            "result_dir": str(Path(session_dir) / self.meta.result_dirname),
            "applied_overrides": applied,
        }

    def _find_config_instance(self, module) -> Any:
        """모듈 안에서 Config 클래스의 module-level 인스턴스를 찾는다.

        25*는 보통 클래스 정의 직후 `p1a_config = P1A_Config()` 패턴이지만
        파일마다 이름이 다를 수 있으므로 fallback: Config 클래스 자체를 반환.
        """

        # 패턴 1: 소문자 stage_id + '_config'
        guess = f"{self.meta.stage_id.lower()}_config"
        if hasattr(module, guess):
            return getattr(module, guess)

        # 패턴 2: Config 클래스 자체에 setattr (class attribute로 동작)
        cls = getattr(module, self.meta.config_class, None)
        if cls is not None:
            return cls

        raise RuntimeError(
            f"[{self.meta.stage_id}] no config instance/class found "
            f"(looked for {guess} or {self.meta.config_class})"
        )


def run_stages(
    session_dir: Path,
    stage_ids: Iterable[str],
    overrides_per_stage: dict[str, dict[str, Any]] | None = None,
) -> list[dict]:
    """여러 stage를 순차 실행. metric_catalog의 위상정렬 결과를 그대로 전달.

    Parameters
    ----------
    overrides_per_stage : {"P1A": {...}, "P1F": {...}}
        stage_id별 config_overrides.
    """

    overrides_per_stage = overrides_per_stage or {}
    results: list[dict] = []
    for sid in stage_ids:
        adapter = BaseAdapter(sid)
        cfg = overrides_per_stage.get(sid, {})
        r = adapter.run(session_dir=Path(session_dir), config_overrides=cfg)
        results.append(r)
    return results
