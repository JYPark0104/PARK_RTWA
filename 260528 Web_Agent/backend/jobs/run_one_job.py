"""run_one_job.py — 잡 1개를 '독립 자식 프로세스'로 실행하는 진입점.

사용:
    python -m backend.jobs.run_one_job <session_uuid>

설계 목적:
- RT 파이프라인을 웹 서버(uvicorn)와 '다른 프로세스'에서 실행 → 웹 API 이벤트 루프가
  RT 의 GIL/CPU 부하에 절대 묶이지 않음 (웹 안정화의 핵심).
- 중단은 부모(JobRunner)가 이 프로세스를 즉시 kill 하여 처리 (즉시 정지).
- 진행률/로그는 execute_pipeline 이 세션 폴더의 _job_events.jsonl 로 남기고,
  세션 메타(session_meta.json)에도 반영 → 부모/프론트가 폴링·tail 로 표시.

실행 환경: Python 3.10 / sionna 1.2.x / GPU (WEBAGENT_USE_GPU 환경변수 상속)
서버: dclserver78 (H100), 컨테이너 WebAgent_park_server78.
"""
from __future__ import annotations

import sys


def main() -> int:
    if len(sys.argv) < 2 or not sys.argv[1].strip():
        print("usage: python -m backend.jobs.run_one_job <session_uuid>", file=sys.stderr)
        return 2
    session_uuid = sys.argv[1].strip()
    # 무거운 import(sionna/tf)는 run_job_child 내부의 지연 import 경로에서 이뤄짐.
    from backend.jobs.pipeline_executor import run_job_child
    return int(run_job_child(session_uuid))


if __name__ == "__main__":
    raise SystemExit(main())
