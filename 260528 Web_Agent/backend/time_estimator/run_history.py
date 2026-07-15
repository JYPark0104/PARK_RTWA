"""run_history.py — 완료 실행 이력 저장/조회 (JSONL, best-effort, reset-safe).

저장 위치
--------
    backend/time_estimator/store/run_history.jsonl

이 위치는 세션 삭제/Reset 이 건드리지 않는 앱 레벨 고정 경로다(사용자 확정 Q4).
'시간 예측 부서'가 자기 데이터를 자기 폴더에서 소유한다.

레코드 스키마 (1줄 = 1 JSON)
    {
      "ts": "2026-07-02T15:00:00+09:00",
      "machine_id","gpu_mode","gpu_name","n_gpu","machine_key",
      "engine","num_tx","num_rx","bs_ant","ue_ant","num_ant_pairs",
      "max_depth","num_samples","batch_size","frequency_ghz",
      "total_sec": 1234.5,
      "stages": {"Scene": 12.3, "RT": 1200.0, "Output1": 22.2}
    }

원칙: 기록은 실행을 절대 방해하지 않는다 → 모든 예외를 삼킨다.

실행 환경: Python 3.10 / 표준 라이브러리
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

_LOCK = threading.Lock()

_STORE_DIR = Path(__file__).resolve().parent / "store"
_HISTORY_PATH = _STORE_DIR / "run_history.jsonl"


def history_path() -> Path:
    return _HISTORY_PATH


def append_record(record: dict[str, Any]) -> bool:
    """레코드 1건을 JSONL 에 append. 성공하면 True, 실패해도 예외 없이 False."""
    try:
        with _LOCK:
            _STORE_DIR.mkdir(parents=True, exist_ok=True)
            with _HISTORY_PATH.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return True
    except Exception:
        return False


def load_records() -> list[dict[str, Any]]:
    """전체 이력 로드. 손상된 줄은 건너뜀. 실패 시 빈 리스트."""
    try:
        if not _HISTORY_PATH.exists():
            return []
        out: list[dict[str, Any]] = []
        with _HISTORY_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
        return out
    except Exception:
        return []


def matching_records(target: dict[str, Any]) -> list[dict[str, Any]]:
    """target config 와 같은 영역의 기록만 필터 (cost_basis.match_config)."""
    from .cost_basis import match_config

    recs = load_records()
    return [r for r in recs if match_config(target, r)]
