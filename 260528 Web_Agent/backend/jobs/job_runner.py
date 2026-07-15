"""job_runner.py — 단일 비동기 잡 매니저 (프로세스 분리 실행).

핵심 설계 (웹 안정화):
- 각 잡을 **별도 자식 프로세스**(`python -m backend.jobs.run_one_job <uuid>`)로 실행.
  → 웹 API(uvicorn) 이벤트 루프가 RT 의 GIL/CPU 부하에 **절대 묶이지 않음**.
- **중단(cancel) = 자식 프로세스 즉시 kill** → 즉시 정지하고 다음 큐 잡으로 넘어감.
- 진행률/로그: 자식이 세션 폴더의 `_job_events.jsonl` 로 append → 부모가 tail 하여
  ws_manager.broker 로 전달. (대시보드는 세션 메타 폴링으로도 동작하므로 WS 는 보조.)
- 동시 실행 1개(FIFO). 큐는 asyncio.Queue.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import json
import logging
import os
import subprocess
import sys
import threading
import time
import uuid as _uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from ..ws_manager import broker

log = logging.getLogger("job_runner")


def _tail_error(logpath, max_chars: int = 6000, n_lines: int = 8) -> str:
    """자식 프로세스 로그의 마지막 의미있는 에러를 짧게 추출 (UI 표시용).
    exit code 만으로는 원인 파악이 어려워, 로그 tail(특히 Error/Exception/cuInit 등)을 올린다."""
    try:
        if not logpath:
            return ""
        from pathlib import Path as _P
        p = _P(logpath)
        if not p.exists():
            return ""
        data = p.read_text(encoding="utf-8", errors="ignore")[-max_chars:]
        lines = [ln.rstrip() for ln in data.splitlines() if ln.strip()]
        if not lines:
            return ""
        # 대표 에러 키워드가 있으면 그 줄부터 끝까지(최대 n_lines) 우선
        markers = ("Error", "Exception", "Traceback", "cuInit", "CUDA", "failed", "assert")
        idx = None
        for i, ln in enumerate(lines):
            if any(m in ln for m in markers):
                idx = i
                break
        chosen = lines[idx:] if idx is not None else lines
        tail = chosen[-n_lines:]
        return " / ".join(tail)[:800]
    except Exception:
        return ""

# 경로 (pipeline_executor 를 top-import 하지 않아 순환참조 회피)
_BACKEND = Path(__file__).resolve().parent.parent          # .../backend
APP_ROOT = _BACKEND.parent                                 # .../260528 Web_Agent (backend 패키지의 부모)
SESSIONS_ROOT = _BACKEND / "sessions"
LOG_DIR = APP_ROOT / "logs"


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
    """단일 사용자 모드 잡 큐 + 프로세스 분리 실행자."""

    _payloads: dict[str, tuple[str, dict]] = {}

    def __init__(self) -> None:
        self._jobs: dict[str, JobState] = {}
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._executor = ThreadPoolExecutor(max_workers=4)  # proc.wait 대기용
        self._worker_task: asyncio.Task | None = None
        self._kinds: set[str] = set()
        self._procs: dict[str, subprocess.Popen] = {}       # job_id → 실행 중 자식 프로세스

    # ------------------------------------------------------------------
    def register_worker(self, kind: str, fn: Callable | None = None) -> None:
        """kind 등록 (실행은 자식 프로세스 subprocess 로 이뤄지므로 fn 은 무시)."""
        self._kinds.add(kind)

    def submit(self, kind: str, session_uuid: str, payload: dict[str, Any]) -> JobState:
        job_id = f"job-{_uuid.uuid4().hex[:12]}"
        job = JobState(job_id=job_id, session_uuid=session_uuid)
        self._jobs[job_id] = job
        self._payloads[job_id] = (kind, {**(payload or {}), "session_uuid": session_uuid})
        try:
            self._queue.put_nowait(job_id)
        except RuntimeError:
            asyncio.get_event_loop().call_soon(lambda: self._queue.put_nowait(job_id))
        return job

    def get(self, job_id: str) -> JobState | None:
        return self._jobs.get(job_id)

    def list_jobs(self) -> list[dict]:
        return [j.to_dict() for j in self._jobs.values()]

    def cancel(self, job_id: str) -> bool:
        """중단: 대기 중이면 스킵 표시, 실행 중이면 자식 프로세스를 즉시 kill."""
        job = self._jobs.get(job_id)
        if not job:
            return False
        if job.state in {"succeeded", "failed", "cancelled"}:
            return False
        if job.state == "pending":
            job.state = "cancelled"
            job.finished_at = _now()
            return True
        # running → 자식 프로세스 즉시 종료
        job.state = "cancelled"
        proc = self._procs.get(job_id)
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()          # SIGTERM
            except Exception:
                pass
            try:
                proc.wait(timeout=1.0)
            except Exception:
                try:
                    proc.kill()           # SIGKILL (즉시)
                except Exception:
                    pass
        # 자식이 kill 되어 세션 메타를 못 쓰므로 부모가 중단 표시
        _kind, payload = self._payloads.get(job_id, ("", {}))
        su = (payload or {}).get("session_uuid", "") or job.session_uuid
        if su:
            try:
                from .queue_store import make_store
                make_store().mark_cancelled(su)
            except Exception:
                pass
        return True

    async def start(self) -> None:
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._run_loop())

    async def shutdown(self) -> None:
        if self._worker_task is not None:
            self._worker_task.cancel()
        for proc in list(self._procs.values()):
            try:
                if proc.poll() is None:
                    proc.kill()
            except Exception:
                pass
        self._executor.shutdown(wait=False)

    # ------------------------------------------------------------------
    async def _run_loop(self) -> None:
        log.info("JobRunner worker loop started (process-isolated)")
        loop = asyncio.get_event_loop()
        while True:
            try:
                job_id = await self._queue.get()
            except asyncio.CancelledError:
                break
            job = self._jobs.get(job_id)
            if job is None or job.state == "cancelled":
                continue
            kind, payload = self._payloads.get(job_id, ("unknown", {}))
            session_uuid = (payload or {}).get("session_uuid", "") or job.session_uuid
            if kind != "pipeline" or not session_uuid:
                job.state = "failed"
                job.error = f"invalid job (kind={kind}, session={session_uuid})"
                await broker.push(job_id, {"kind": "error", "message": job.error})
                continue

            job.state = "running"
            job.started_at = _now()
            await broker.push(job_id, {"kind": "log", "message": f"Job {job_id} started (pipeline, isolated process)"})

            sdir = SESSIONS_ROOT / session_uuid
            evpath = sdir / "_job_events.jsonl"
            try:
                sdir.mkdir(parents=True, exist_ok=True)
                if evpath.exists():
                    evpath.unlink()
            except Exception:
                pass

            # 자식 stdout/stderr → 로그 파일 (실패 시 이 파일 tail 을 UI 로 surfacing)
            logpath = None
            try:
                LOG_DIR.mkdir(parents=True, exist_ok=True)
                logpath = LOG_DIR / f"job_{session_uuid}_{time.strftime('%y%m%d_%H%M%S')}.log"
                logf = open(logpath, "w")
            except Exception:
                logf = subprocess.DEVNULL
                logpath = None

            env = dict(os.environ)
            env.setdefault("PYTHONUNBUFFERED", "1")
            try:
                proc = subprocess.Popen(
                    [sys.executable, "-m", "backend.jobs.run_one_job", session_uuid],
                    cwd=str(APP_ROOT), env=env,
                    stdout=logf, stderr=subprocess.STDOUT,
                )
            except Exception as exc:  # noqa: BLE001
                job.state = "failed"
                job.error = f"failed to spawn: {exc}"
                await broker.push(job_id, {"kind": "error", "message": job.error})
                job.finished_at = _now()
                continue
            self._procs[job_id] = proc

            # 이벤트 파일 tail → broker 전달 + job 상태 갱신 (별도 스레드)
            stop = {"stop": False}

            def _monitor() -> None:
                # 파일 생성 대기
                for _ in range(600):
                    if evpath.exists() or stop["stop"]:
                        break
                    time.sleep(0.1)
                if not evpath.exists():
                    return
                try:
                    f = open(evpath, "r", encoding="utf-8")
                except Exception:
                    return
                try:
                    while True:
                        line = f.readline()
                        if line:
                            s = line.strip()
                            if not s:
                                continue
                            try:
                                ev = json.loads(s)
                            except Exception:
                                continue
                            if "progress" in ev:
                                try:
                                    job.progress = float(ev["progress"])
                                except Exception:
                                    pass
                            if ev.get("kind") == "stage_start" and ev.get("stage"):
                                job.current_stage = ev["stage"]
                            if ev.get("kind") == "stage_end" and ev.get("stage") and ev["stage"] not in job.stages_done:
                                job.stages_done.append(ev["stage"])
                            if ev.get("kind") == "done" and isinstance(ev.get("output_paths"), dict):
                                job.output_paths.update(ev["output_paths"])
                            try:
                                asyncio.run_coroutine_threadsafe(broker.push(job_id, ev), loop)
                            except Exception:
                                pass
                        else:
                            if proc.poll() is not None:
                                # 프로세스 종료 → 남은 줄 마저 읽고 종료
                                for rest in f:
                                    rs = rest.strip()
                                    if not rs:
                                        continue
                                    try:
                                        ev = json.loads(rs)
                                        asyncio.run_coroutine_threadsafe(broker.push(job_id, ev), loop)
                                    except Exception:
                                        pass
                                break
                            time.sleep(0.15)
                finally:
                    try:
                        f.close()
                    except Exception:
                        pass

            mon = threading.Thread(target=_monitor, daemon=True)
            mon.start()

            # 프로세스 종료 대기 (스레드풀에서 → 이벤트 루프 비차단)
            rc = await loop.run_in_executor(self._executor, proc.wait)
            stop["stop"] = True
            try:
                mon.join(timeout=3)
            except Exception:
                pass
            self._procs.pop(job_id, None)
            try:
                if logf not in (None, subprocess.DEVNULL):
                    logf.close()
            except Exception:
                pass

            if job.state == "cancelled":
                await broker.push(job_id, {"kind": "log", "message": "⏹ 중단됨 — 다음 작업으로 넘어갑니다."})
            elif rc == 0:
                job.state = "succeeded"
                await broker.push(job_id, {"kind": "done", "message": "completed"})
            else:
                job.state = "failed"
                detail = _tail_error(logpath)
                job.error = f"exit code {rc}" + (f" — {detail}" if detail else "")
                msg = f"작업 실패 (exit {rc})"
                if detail:
                    msg += f"\n{detail}"
                await broker.push(job_id, {"kind": "error", "message": msg})
            job.finished_at = _now()


runner = JobRunner()
