"""eta.py — (B) 실행 중 남은 시간(ETA) 계산.

원리 (사용자 확정 Q8)
    남은시간 = 경과시간 ÷ 진행률 × (1 − 진행률)   (wall-clock 기반)
    단, 진행률이 임계값 미만(예: 2%, Scene 로딩/precompute 구간)이면 None → "준비 중".

실행 환경: Python 3.10 / 표준 라이브러리
"""

from __future__ import annotations

# 이 진행률 미만이면 ETA 를 신뢰하지 않음(초기 precompute 구간에서 폭주 방지)
MIN_PROGRESS_FOR_ETA = 0.02


def compute_eta(elapsed_sec: float | None, progress: float | None,
                min_progress: float = MIN_PROGRESS_FOR_ETA) -> float | None:
    """경과시간 + 진행률 → 남은 시간(초). 신뢰 불가면 None.

    None 을 반환하는 경우:
      - elapsed/progress 가 없음
      - progress 가 임계값 미만 (아직 '준비 중')
      - progress 가 1.0 이상 (사실상 완료)
    """
    if elapsed_sec is None or progress is None:
        return None
    try:
        elapsed = float(elapsed_sec)
        p = float(progress)
    except (TypeError, ValueError):
        return None
    if p < min_progress or p >= 1.0 or elapsed <= 0:
        return None
    return elapsed / p * (1.0 - p)


def fmt_hms(sec: float | None) -> str:
    """초 → 'Xh Ym Zs' (프런트 표시 보조용, 백엔드 로그에도 사용)."""
    if sec is None:
        return "—"
    sec = max(0, int(round(sec)))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"
