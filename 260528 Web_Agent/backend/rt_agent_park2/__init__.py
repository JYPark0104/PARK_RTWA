"""
rt_agent_park2
==============
PARK_2_RayTracing_Agent_v2 의 Ray Tracing Agent 방식을 Web Agent 백엔드로 이식한 패키지.

원본: /home/dclserver78/PARK_2_RayTracing_Agent_v2/2-1.RayTracingAgent
  (원본은 워크스페이스 밖이라 읽기 전용 참고. 본 패키지는 사본/재구현.)

구성:
  - ground.py : 지면 레이캐스팅 기반 RX 격자 생성 + TX 지면 스냅
                (원본 m2_config_agent.py + RT_utils.get_adaptive_rx_positions 이식)

설계 메모 (A안: 표준 NPZ 유지):
  RX 지면 격자/TX 스냅은 백엔드에서 좌표를 미리 계산하여, 기존 P1A 파이프라인에는
  "explicit (x,y,z) 좌표"로 전달한다. → P1A 의 251218 표준 NPZ 스키마와 metric
  파이프라인을 전혀 건드리지 않고 통합된다.
"""

from .ground import (
    compute_ground_rx_grid,
    snap_tx_to_ground,
    ground_z_at,
    GroundRaycaster,
)

__all__ = [
    "compute_ground_rx_grid",
    "snap_tx_to_ground",
    "ground_z_at",
    "GroundRaycaster",
    "run_batch_rt",
]


def run_batch_rt(*args, **kwargs):
    """지연 import 래퍼 (engine/sionna 의존이 무거우므로 호출 시점에 로드)."""
    from .batch_runner import run_batch_rt as _impl
    return _impl(*args, **kwargs)
