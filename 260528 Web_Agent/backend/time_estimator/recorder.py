"""recorder.py — 잡 실행 관찰 + 완료 시 이력 기록.

StageTimer
    파이프라인이 emit 하는 이벤트를 관찰(observe)해서:
      - stage_start / stage_end 사이의 wall-clock 소요시간을 단계별로 수집
      - batch_progress 이벤트에서 실제 num_tx / total_rx(=num_rx) 를 포착
    실행에 절대 개입하지 않는다(수동 관찰만).

record_run
    완료 시 StageTimer 가 모은 단계별 소요시간 + config feature + 머신 지문을
    run_history JSONL 에 1줄 append. best-effort(예외 삼킴).

실행 환경: Python 3.10 / 표준 라이브러리(time)
"""

from __future__ import annotations

import time
from typing import Any

from .cost_basis import extract_features
from .machine import machine_fingerprint, machine_key
from .run_history import append_record


class StageTimer:
    """emit 이벤트를 관찰해 단계별 소요시간과 실제 규모(num_tx/num_rx)를 수집."""

    def __init__(self) -> None:
        self._starts: dict[str, float] = {}
        self.durations: dict[str, float] = {}
        self.order: list[str] = []
        self.num_tx: int | None = None
        self.num_rx: int | None = None
        self._t0 = time.monotonic()

    def observe(self, event: dict) -> None:
        kind = event.get("kind")
        stage = event.get("stage")
        if kind == "stage_start" and stage:
            self._starts[stage] = time.monotonic()
        elif kind == "stage_end" and stage:
            t0 = self._starts.get(stage)
            if t0 is not None:
                self.durations[stage] = time.monotonic() - t0
                if stage not in self.order:
                    self.order.append(stage)
        if kind == "batch_progress":
            if event.get("num_tx"):
                self.num_tx = int(event["num_tx"])
            if event.get("total_rx"):
                self.num_rx = int(event["total_rx"])

    def total_sec(self) -> float:
        return time.monotonic() - self._t0

    def has_data(self) -> bool:
        return bool(self.durations)


def record_run(payload: dict, timer: StageTimer) -> bool:
    """완료된 잡 1건을 이력에 기록. 성공 여부(True/False) 반환, 예외 없음."""
    try:
        if not timer.has_data():
            return False
        fp = machine_fingerprint()
        feat = extract_features(
            payload,
            num_rx_actual=timer.num_rx,
            num_tx_actual=timer.num_tx,
        )
        record: dict[str, Any] = {
            "ts": _now_kst(),
            "machine_id": fp.get("machine_id"),
            "gpu_mode": fp.get("gpu_mode"),
            "gpu_name": fp.get("gpu_name"),
            "n_gpu": fp.get("n_gpu"),
            "machine_key": machine_key(fp),
            "engine": feat.get("engine"),
            "num_tx": feat.get("num_tx"),
            "num_rx": feat.get("num_rx"),
            "bs_ant": feat.get("bs_ant"),
            "ue_ant": feat.get("ue_ant"),
            "num_ant_pairs": feat.get("num_ant_pairs"),
            "max_depth": feat.get("max_depth"),
            "num_samples": feat.get("num_samples"),
            "batch_size": feat.get("batch_size"),
            "frequency_ghz": feat.get("frequency_ghz"),
            "total_sec": round(timer.total_sec(), 3),
            "stages": {k: round(v, 3) for k, v in timer.durations.items()},
        }
        return append_record(record)
    except Exception:
        return False


def _now_kst() -> str:
    import datetime as _dt

    kst = _dt.timezone(_dt.timedelta(hours=9))
    return _dt.datetime.now(kst).isoformat()
