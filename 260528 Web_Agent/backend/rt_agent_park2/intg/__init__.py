"""intg — 통합(Intg) RT: 단일 superset NPZ 생성 + 소비자별 reshaper.

구조 (Adapter 패턴):
    통합 RT ─▶ superset NPZ ─┬─(reshapers.to_p1a)──────▶ P1B/C/D
                             └─(reshapers.to_batch_channel)─▶ RX Inspector/Scenario/viz

하류 모듈은 수정하지 않고, reshaper 가 superset 을 각 소비자 포맷으로 변환한다.
"""

from . import superset_schema  # noqa: F401
