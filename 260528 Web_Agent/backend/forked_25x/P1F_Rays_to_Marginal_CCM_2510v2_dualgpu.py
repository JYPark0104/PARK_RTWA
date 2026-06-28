# Web Agent fork — P1F 2×GPU (subprocess workers)
# 실행 환경: Python 3.10, dclserver78

from __future__ import annotations

from pathlib import Path

from backend.jobs.p1f_dual_gpu_worker import run_dual_gpu_orchestrator


class P1F_Config:
    """BaseAdapter 호환 스텁."""


def web_run_pipeline(config_overrides: dict | None = None) -> None:
    run_dual_gpu_orchestrator(Path.cwd(), config_overrides)


def main() -> None:
    web_run_pipeline()


if __name__ == "__main__":
    main()
