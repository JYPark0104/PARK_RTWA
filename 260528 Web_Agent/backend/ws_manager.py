"""ws_manager.py — WebSocket 진행률 브로커 (단일 사용자).

job_id를 key로 모든 연결된 WebSocket에 ProgressEvent를 push.

매우 가벼운 broker — 단일 사용자 가정.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import json
import logging
from collections import defaultdict
from typing import Any

from fastapi import WebSocket


log = logging.getLogger("ws_manager")


def _now() -> str:
    kst = _dt.timezone(_dt.timedelta(hours=9))
    return _dt.datetime.now(kst).isoformat()


class WebSocketBroker:
    """단일 사용자용 진행률 브로커."""

    def __init__(self) -> None:
        self._connections: dict[str, list[WebSocket]] = defaultdict(list)
        self._lock = asyncio.Lock()
        self._history: dict[str, list[dict]] = defaultdict(list)
        self.max_history = 500

    async def connect(self, job_id: str, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self._connections[job_id].append(ws)
        # send history 누락 방지
        for ev in self._history.get(job_id, []):
            try:
                await ws.send_text(json.dumps(ev))
            except Exception:
                break

    async def disconnect(self, job_id: str, ws: WebSocket) -> None:
        async with self._lock:
            if ws in self._connections.get(job_id, []):
                self._connections[job_id].remove(ws)

    async def push(self, job_id: str, event: dict[str, Any]) -> None:
        event = {**event, "job_id": job_id, "timestamp": event.get("timestamp") or _now()}
        history = self._history.setdefault(job_id, [])
        history.append(event)
        if len(history) > self.max_history:
            del history[: len(history) - self.max_history]

        async with self._lock:
            targets = list(self._connections.get(job_id, []))
        text = json.dumps(event)
        dead: list[WebSocket] = []
        for ws in targets:
            try:
                await ws.send_text(text)
            except Exception:
                dead.append(ws)
        if dead:
            async with self._lock:
                for d in dead:
                    if d in self._connections.get(job_id, []):
                        self._connections[job_id].remove(d)

    def push_sync(self, job_id: str, event: dict[str, Any]) -> None:
        """동기 컨텍스트(잡 워커 스레드)에서 호출.

        이벤트 루프가 떠 있는 다른 스레드에 코루틴을 schedule.
        """

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.run_coroutine_threadsafe(self.push(job_id, event), loop)
                return
        except RuntimeError:
            pass
        # fallback
        asyncio.run(self.push(job_id, event))

    def get_history(self, job_id: str) -> list[dict]:
        return list(self._history.get(job_id, []))


broker = WebSocketBroker()
