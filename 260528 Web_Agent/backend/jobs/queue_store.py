"""queue_store.py — 멀티유저 잡 큐 영속화 + 재시작 복구 (단일 FIFO / 단일 워커).

설계 결정 (사용자 확정):
  - 동시 실행 1개 (기존 JobRunner 단일 워커 그대로).
  - user = 이름표(인증 아님), 권한 제한 없음.
  - 세션 중심 상태: 세션이 queued → processing → done 으로 이동(재실행 시 갱신).
  - 재시작 복구: processing 이던 세션 → interrupted, queued → 순서 유지하여 재등록.
  - 큐 제어: queued 취소 / processing 협력적 중단 / reorder 없음 /
    queued 세션의 config 수정·삭제 허용(순서 불변).

영속화 파일 (sessions/ 루트):
  - _users.json          : {"users": [{"id","name"}]}
  - _queue.json          : {"order": [session_uuid, ...]}   (FIFO 순서)
  - 각 세션 dir/session_meta.json : status/progress/타임스탬프/user (session_labeler)
  - 각 세션 dir/session_config.json : 실행 config 스냅샷 (Run 시 저장, 워커가 실행 시점에 로드)

실행 환경: Python 3.10 / FastAPI (컨테이너 venv-webagent)
"""

from __future__ import annotations

import datetime as _dt
import json
import threading
import uuid as _uuid
from pathlib import Path

from .session_labeler import load_meta, update_meta

_LOCK = threading.RLock()

# 완료 계열 상태
_DONE_STATES = {"done", "failed", "interrupted", "cancelled"}


def _now() -> str:
    kst = _dt.timezone(_dt.timedelta(hours=9))
    return _dt.datetime.now(kst).isoformat()


def _atomic_write_text(path: Path, text: str) -> None:
    """temp 에 쓰고 os.replace 로 교체 → 읽는 쪽이 반쪽/빈 파일을 보지 않음(500 방지)."""
    import os
    import tempfile
    try:
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix="." + path.name + ".", suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    except Exception:
        try:
            path.write_text(text, encoding="utf-8")
        except Exception:
            pass


class QueueStore:
    """sessions 루트 기준 큐/유저 영속화 관리자."""

    def __init__(self, sessions_root: Path) -> None:
        self.root = Path(sessions_root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.users_path = self.root / "_users.json"
        self.queue_path = self.root / "_queue.json"

    # ------------------------------------------------------------------
    # Users CRUD
    # ------------------------------------------------------------------
    def _read_users(self) -> list[dict]:
        if not self.users_path.exists():
            return []
        try:
            return json.loads(self.users_path.read_text(encoding="utf-8")).get("users", [])
        except Exception:
            return []

    def _write_users(self, users: list[dict]) -> None:
        _atomic_write_text(self.users_path, json.dumps({"users": users}, ensure_ascii=False, indent=2))

    def list_users(self) -> list[dict]:
        with _LOCK:
            return self._read_users()

    def add_user(self, name: str) -> dict:
        name = (name or "").strip()
        if not name:
            raise ValueError("user 이름이 비어 있습니다.")
        with _LOCK:
            users = self._read_users()
            if any(u["name"] == name for u in users):
                raise ValueError(f"이미 존재하는 user 이름: {name}")
            user = {"id": _uuid.uuid4().hex[:8], "name": name}
            users.append(user)
            self._write_users(users)
            return user

    def rename_user(self, user_id: str, name: str) -> dict:
        name = (name or "").strip()
        if not name:
            raise ValueError("user 이름이 비어 있습니다.")
        with _LOCK:
            users = self._read_users()
            for u in users:
                if u["id"] == user_id:
                    u["name"] = name
                    self._write_users(users)
                    return u
            raise KeyError(f"user not found: {user_id}")

    def delete_user(self, user_id: str) -> bool:
        with _LOCK:
            users = self._read_users()
            new = [u for u in users if u["id"] != user_id]
            self._write_users(new)
            return len(new) != len(users)

    # ------------------------------------------------------------------
    # Queue order (FIFO)
    # ------------------------------------------------------------------
    def _read_order(self) -> list[str]:
        if not self.queue_path.exists():
            return []
        try:
            return json.loads(self.queue_path.read_text(encoding="utf-8")).get("order", [])
        except Exception:
            return []

    def _write_order(self, order: list[str]) -> None:
        _atomic_write_text(self.queue_path, json.dumps({"order": order}, ensure_ascii=False, indent=2))

    def enqueue(self, uuid: str, job_id: str, user_id: str = "", user_name: str = "",
                engine: str = "") -> None:
        """세션을 큐 끝에 등록(이미 있으면 순서 유지) + 메타 status=queued."""
        with _LOCK:
            order = self._read_order()
            if uuid not in order:
                order.append(uuid)
                self._write_order(order)
            changes = dict(status="queued", progress=0.0, current_stage="",
                           job_id=job_id, queued_at=_now(), started_at="",
                           finished_at="", last_error="")
            if engine:
                changes["engine"] = engine
            if user_id:
                changes["user_id"] = user_id
            if user_name:
                changes["user_name"] = user_name
            update_meta(self.root / uuid, **changes)

    def remove_from_order(self, uuid: str) -> None:
        with _LOCK:
            order = self._read_order()
            if uuid in order:
                order.remove(uuid)
                self._write_order(order)

    # ------------------------------------------------------------------
    # 상태 전이
    # ------------------------------------------------------------------
    def mark_processing(self, uuid: str) -> None:
        with _LOCK:
            update_meta(self.root / uuid, status="processing",
                        started_at=_now(), progress=0.0)

    def update_progress(self, uuid: str, progress: float | None, stage: str | None) -> None:
        changes: dict = {}
        if progress is not None:
            changes["progress"] = float(progress)
        if stage:
            changes["current_stage"] = stage
        if changes:
            update_meta(self.root / uuid, **changes)

    def mark_done(self, uuid: str) -> None:
        with _LOCK:
            self.remove_from_order(uuid)
            update_meta(self.root / uuid, status="done", progress=1.0,
                        finished_at=_now(), current_stage="")

    def mark_failed(self, uuid: str, error: str) -> None:
        with _LOCK:
            self.remove_from_order(uuid)
            update_meta(self.root / uuid, status="failed",
                        finished_at=_now(), last_error=(error or "")[:500])

    def mark_cancelled(self, uuid: str) -> None:
        with _LOCK:
            self.remove_from_order(uuid)
            update_meta(self.root / uuid, status="cancelled", finished_at=_now())

    def mark_interrupted(self, uuid: str) -> None:
        with _LOCK:
            self.remove_from_order(uuid)
            update_meta(self.root / uuid, status="interrupted", finished_at=_now())

    # ------------------------------------------------------------------
    # 대시보드 (1초 폴링용)
    # ------------------------------------------------------------------
    def _meta_brief(self, uuid: str, live_progress: float | None = None) -> dict | None:
        meta = load_meta(self.root / uuid)
        if meta is None:
            return None
        prog = live_progress if live_progress is not None else meta.progress
        elapsed = None
        if meta.started_at and meta.status == "processing":
            try:
                t0 = _dt.datetime.fromisoformat(meta.started_at)
                elapsed = (_dt.datetime.now(t0.tzinfo) - t0).total_seconds()
            except Exception:
                elapsed = None
        # (B) 실행 중 ETA — 경과시간+진행률 기반 남은 시간(초). 초기 구간이면 None.
        eta = None
        if meta.status == "processing":
            try:
                from ..time_estimator.eta import compute_eta
                eta = compute_eta(elapsed, prog)
            except Exception:
                eta = None
        return {
            "uuid": meta.uuid, "label": meta.label,
            "user_id": meta.user_id, "user_name": meta.user_name,
            "scene_name": meta.scene_name,
            "display_name": getattr(meta, "display_name", "") or meta.scene_name,
            "status": meta.status, "progress": prog,
            "current_stage": meta.current_stage, "engine": meta.engine,
            "queued_at": meta.queued_at, "started_at": meta.started_at,
            "finished_at": meta.finished_at, "last_error": meta.last_error,
            "elapsed_sec": elapsed,
            "eta_sec": eta,
            "bs_rows": meta.bs_rows, "bs_cols": meta.bs_cols,
            "ue_rows": meta.ue_rows, "ue_cols": meta.ue_cols,
        }

    def dashboard(self, live_progress_by_uuid: dict[str, float] | None = None) -> dict:
        live = live_progress_by_uuid or {}
        with _LOCK:
            order = self._read_order()
            all_metas = []
            for d in self.root.iterdir():
                if d.is_dir():
                    m = load_meta(d)
                    if m:
                        all_metas.append(m)

        queueing = []
        for uuid in order:
            b = self._meta_brief(uuid)
            if b and b["status"] == "queued":
                queueing.append(b)

        processing = None
        done = []
        for m in all_metas:
            if m.status == "processing":
                processing = self._meta_brief(m.uuid, live.get(m.uuid))
                # 라이브 커버리지 스캐터 스냅샷 첨부 (batch 엔진에서 배치마다 기록)
                try:
                    sp = self.root / m.uuid / "live_scatter.json"
                    if sp.exists():
                        processing["scatter"] = json.loads(sp.read_text(encoding="utf-8"))
                except Exception:
                    pass
            elif m.status in _DONE_STATES:
                done.append(self._meta_brief(m.uuid))
        done.sort(key=lambda x: x.get("finished_at") or "", reverse=True)
        return {"queueing": queueing, "processing": processing, "done": done}

    # ------------------------------------------------------------------
    # 재시작 복구
    # ------------------------------------------------------------------
    def recover(self) -> list[str]:
        """processing → interrupted, queued 순서 반환(재등록은 호출측에서)."""
        with _LOCK:
            for d in self.root.iterdir():
                if not d.is_dir():
                    continue
                m = load_meta(d)
                if m and m.status == "processing":
                    update_meta(d, status="interrupted", finished_at=_now(),
                                last_error="백엔드 재시작으로 중단됨")
            # queued 순서대로 반환 (실제 status 가 queued 인 것만)
            order = self._read_order()
            valid = []
            for uuid in order:
                m = load_meta(self.root / uuid)
                if m and m.status == "queued":
                    valid.append(uuid)
            self._write_order(valid)
            return valid


# ---------------------------------------------------------------------------
# Config 스냅샷 (Run 시 저장 / 워커가 실행 시점에 로드 / 큐 편집 반영)
# ---------------------------------------------------------------------------
def save_session_config(session_dir: Path, payload: dict) -> None:
    (Path(session_dir) / "session_config.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_session_config(session_dir: Path) -> dict | None:
    p = Path(session_dir) / "session_config.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 싱글톤 (SESSIONS_ROOT 기준)
# ---------------------------------------------------------------------------
_STORE: "QueueStore | None" = None


def make_store() -> "QueueStore":
    """pipeline_executor.SESSIONS_ROOT 기준 QueueStore 싱글톤."""
    global _STORE
    if _STORE is None:
        from .pipeline_executor import SESSIONS_ROOT
        _STORE = QueueStore(SESSIONS_ROOT)
    return _STORE
