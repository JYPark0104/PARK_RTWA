"""job_runner.py — 단일 비동기 잡 매니저.

- 동시 실행 1개 (단일 사용자 모드).
- 추가 요청은 asyncio.Queue에 쌓임.
- 각 잡은 ThreadPoolExecutor (max_workers=1) 에서 실행 → 메인 이벤트 루프 비차단.
- 진행률은 ws_manager.broker로 push.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import logging
import traceback
import uuid as _uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from ..ws_manager import broker

log = logging.getLogger("job_runner")


def _now() -> str:
    kst = _dt.timezone(_dt.timedelta(hours=9))
    return _dt.datetime.now(kst).isoformat()


@dataclass
class JobState:
    """단일 잡의 in-memory 상태."""

    job_id: str
    session_uuid: str
    state: str = "pending"   # pending|running|succeeded|failed|cancelled
    progress: float = 0.0
    current_stage: str = ""
    stages_done: list[str] = field(default_factory=list)
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    output_paths: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "session_uuid": self.session_uuid,
            "state": self.state,
            "progress": self.progress,
            "current_stage": self.current_stage,
            "stages_done": list(self.stages_done),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "output_paths": dict(self.output_paths),
        }


class JobRunner:
    """단일 사용자 모드 잡 큐 + 실행자."""

    def __init__(self) -> None:
        self._jobs: dict[str, JobState] = {}
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._worker_task: asyncio.Task | None = None
        self._workers: dict[str, Callable[[JobState, Callable[..., None]], dict]] = {}

    def register_worker(self, kind: str, fn: Callable[[JobState, Callable[..., None]], dict]) -> None:
        """worker 함수 등록.

        worker signature:
            fn(job: JobState, emit: Callable[..., None]) -> dict
        """

        self._workers[kind] = fn

    def submit(self, kind: str, session_uuid: str, payload: dict[str, Any]) -> JobState:
        """잡 등록 → 큐에 push."""

        job_id = f"job-{_uuid.uuid4().hex[:12]}"
        job = JobState(job_id=job_id, session_uuid=session_uuid)
        job.output_paths["__kind__"] = kind
        job.output_paths["__payload__"] = "<see payload>"
        self._jobs[job_id] = job
        # payload를 별도로 보관
        self._payloads[job_id] = (kind, payload)
        try:
            self._queue.put_nowait(job_id)
        except RuntimeError:
            # 이벤트 루프 시작 전: 즉시 실행 보류
            asyncio.get_event_loop().call_soon(lambda: self._queue.put_nowait(job_id))
        return job

    _payloads: dict[str, tuple[str, dict]] = {}

    def get(self, job_id: str) -> JobState | None:
        return self._jobs.get(job_id)

    def list_jobs(self) -> list[dict]:
        return [j.to_dict() for j in self._jobs.values()]

    def cancel(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if not job:
            return False
        if job.state in {"succeeded", "failed", "cancelled"}:
            return False
        if job.state == "pending":
            job.state = "cancelled"
            job.finished_at = _now()
            return True
        # running 중인 잡은 cooperative cancel (worker가 emit를 통해 polling)
        job.state = "cancelled"  # worker가 다음 step에서 확인
        return True

    async def start(self) -> None:
        """이벤트 루프에서 한 번 호출 — 워커 태스크 가동."""

        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._run_loop())

    async def shutdown(self) -> None:
        if self._worker_task is not None:
            self._worker_task.cancel()
        self._executor.shutdown(wait=False)

    async def _run_loop(self) -> None:
        log.info("JobRunner worker loop started")
        while True:
            try:
                job_id = await self._queue.get()
            except asyncio.CancelledError:
                break
            job = self._jobs.get(job_id)
            if job is None:
                continue
            if job.state == "cancelled":
                continue
            kind, payload = self._payloads.get(job_id, ("unknown", {}))
            worker = self._workers.get(kind)
            if worker is None:
                job.state = "failed"
                job.error = f"No worker for kind={kind}"
                await broker.push(job_id, {"kind": "error", "message": job.error})
                continue
            job.state = "running"
            job.started_at = _now()
            await broker.push(job_id, {
                "kind": "log",
                "message": f"Job {job_id} started ({kind})",
            })

            loop = asyncio.get_event_loop()

            def emit(event: dict) -> None:
                """worker → broker push (스레드 안전)."""

                if "progress" in event:
                    job.progress = float(event["progress"])
                if "stage" in event and event.get("kind") == "stage_start":
                    job.current_stage = event["stage"]
                if event.get("kind") == "stage_end":
                    s = event.get("stage")
                    if s and s not in job.stages_done:
                        job.stages_done.append(s)
                asyncio.run_coroutine_threadsafe(broker.push(job_id, event), loop)

            def run_sync() -> dict:
                return worker(job, emit)

            try:
                out = await loop.run_in_executor(self._executor, run_sync)
                if job.state == "cancelled":
                    await broker.push(job_id, {"kind": "log", "message": "cancelled"})
                else:
                    job.state = "succeeded"
                    job.output_paths.update(out.get("output_paths", {}) if isinstance(out, dict) else {})
                    await broker.push(job_id, {"kind": "done", "message": "completed"})
            except Exception as exc:  # noqa: BLE001
                job.state = "failed"
                job.error = f"{type(exc).__name__}: {exc}"
                tb = traceback.format_exc()
                log.exception("Job %s failed", job_id)
                await broker.push(job_id, {
                    "kind": "error",
                    "message": job.error,
                    "stage": job.current_stage,
                })
                await broker.push(job_id, {
                    "kind": "log",
                    "message": tb,
                })
            finally:
                job.finished_at = _now()


runner = JobRunner()
