"""time_estimator — RTWA '시간 예측 담당 부서' (self-contained department).

이 패키지는 Web Agent(백엔드) 산하에서 **실행 시간 예측**만 전담한다.
다른 모듈은 이 패키지의 공개 API 4개만 알면 된다.

공개 API
--------
- record_run(...)        : 잡 완료 시 per-stage 소요시간을 이력(JSONL)에 append (best-effort)
- estimate(...)          : (A) 실행 전 예측 — config별 회귀로 총 소요시간 mean±std
- compute_eta(...)       : (B) 실행 중 ETA — elapsed/progress 기반 남은 시간(초)
- StageTimer            : 잡 실행 중 stage 경계를 관찰해 per-stage 소요시간을 모으는 수집기

설계 원리(요약)
--------------
- 단계(stage)마다 '시간을 지배하는 변수(cost basis) 1개'를 정해 1D 회귀 → 합산.
- 하드웨어(머신+GPU/CPU 모드)와 형태-변경/배율 파라미터는 'config 매칭'으로 분리.
- 데이터가 적어도 안 깨지게 폴백 사다리(affine→비례→평균→null).
- 기록은 절대 실행을 방해하지 않음(예외 삼킴). reset/세션삭제에 안 지워지는 위치에 저장.

실행 환경
--------
- Python  : 3.10 (컨테이너 venv-webagent)
- 서버    : dclcom45 / dclcom55 / H100 등 (machine 지문으로 자동 구분)
- 의존성  : 표준 라이브러리만 사용 (numpy 등 불요)
"""

from __future__ import annotations

from .eta import compute_eta
from .estimator import estimate
from .recorder import StageTimer, record_run

__all__ = ["record_run", "estimate", "compute_eta", "StageTimer"]
