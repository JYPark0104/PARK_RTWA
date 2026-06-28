"""superset_schema.py — 통합(Intg) RT 의 단일 superset NPZ 스키마 정의.

실행 환경:
- Python 3.10 / numpy / (RT 생성 시) sionna 1.2.x
- 서버: dclserver78 (컨테이너 venv /opt/venvs/webagent)

설계 (2026 통합 결정 반영):
- 통합 RT 는 P1A(ray-level 기하) + batch(MIMO 공분산/집계) 정보를 모두 담는
  **단일 multi-TX superset NPZ** 하나만 생성한다.
- 하류 모듈(P1B/C/D, batch 시각화)은 수정하지 않는다. 대신 reshaper 가
  superset → 각 소비자 포맷으로 변환해 떠먹인다 (Adapter 패턴).
    superset ──(to_p1a)──▶ AreaX_{freq}GHz_Rays_ALL_RXs.npz  → P1B/C/D
    superset ──(to_batch_channel)──▶ channel_data_*.npz      → RX Inspector/Scenario/viz

핵심 구분:
- path-level : Sionna 가 찾은 '유효 경로' 단위 (batch/공분산 입력). 작은 차원.
- ray-level  : 각 path 를 TR38.901 sub-ray 로 확장한 단위 (P1A 호환). 패딩 큰 차원.
  ray 는 path 에서 파생되므로 source_path_idx 로 ray→path 매핑을 보존한다.

배열 형태 (결정: (B) 스택 배열 — P1A 호환성 최고):
- 모든 (TX, RX) 를 같은 축으로 스택. ray 차원 P 는 '파일 내 최대 유효 ray 수'(동적, 400 하드코딩 폐기).
- path 차원 K 는 '파일 내 최대 유효 path 수'(동적).
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# 패딩 값 규약 (P1B 가 counts/path_counts 로 실제 길이를 보므로, 패딩 값은 '무해'해야 함)
#   - tau/power/angle : 0.0   (음수지연 오탐 방지: tau 패딩 >= -0.1 이어야 함 → 0 안전)
#   - source_path_idx : -1    (유효 ray 가 아님을 표시)
#   - los_nlos_flag   : 0     (NLoS 로 취급)
# ---------------------------------------------------------------------------
PAD_FLOAT = 0.0
PAD_SOURCE_PATH_IDX = -1
PAD_LOS_FLAG = 0

# rx_valid_mask 코드값 (비파괴 필터 — 삭제하지 않고 사유만 기록)
RX_VALID = 0      # 정상
RX_DEAD = 1       # 물리적 dead (유효 경로 0)
RX_RT_FAIL = 2    # RT 실패 (음수지연 tau<-0.1 또는 ray_efficiency<thr)

# 기본 임계값 (P1B 와 동일)
TAU_MIN_THRESHOLD = -0.1
RAY_EFFICIENCY_THRESHOLD = 1.0


# ---------------------------------------------------------------------------
# 스키마 키 정의 (문서화 + 검증용)
#   shape 기호: T=num_tx, R=num_rx, K=max_paths(동적), P=max_rays(동적),
#               At=n_tx_ant, Ar=n_rx_ant
# ---------------------------------------------------------------------------
SUPERSET_KEYS = {
    # --- path-level (batch / 공분산 입력) ---
    "path_tau":        ("(T,R,K)",  "float32", "유효 경로 지연 [s]"),
    "path_power":      ("(T,R,K)",  "float32", "경로 전력 (선형 또는 dBm, 생성 시 규약 고정)"),
    "path_phi_r":      ("(T,R,K)",  "float32", "수신 방위 AoA [deg]"),
    "path_phi_t":      ("(T,R,K)",  "float32", "송신 방위 AoD [deg]"),
    "path_theta_r":    ("(T,R,K)",  "float32", "수신 고도 ZoA [deg]"),
    "path_theta_t":    ("(T,R,K)",  "float32", "송신 고도 ZoD [deg]"),
    "path_los_flag":   ("(T,R,K)",  "int32",   "경로별 LoS(1)/NLoS(0)"),
    "path_counts":     ("(T,R)",    "int32",   "(TX,RX)별 유효 경로 수"),

    # --- ray-level (P1A 호환, TR38.901 확장) ---
    "tau":             ("(T,R,P)",  "float32", "ray 지연 [s]"),
    "power":           ("(T,R,P)",  "float32", "ray 전력"),
    "phi_r_deg":       ("(T,R,P)",  "float32", "수신 방위 AoA [deg]"),
    "phi_t_deg":       ("(T,R,P)",  "float32", "송신 방위 AoD [deg]"),
    "theta_r_deg":     ("(T,R,P)",  "float32", "수신 고도 ZoA [deg]"),
    "theta_t_deg":     ("(T,R,P)",  "float32", "송신 고도 ZoD [deg]"),
    "source_path_idx": ("(T,R,P)",  "int32",   "ray→path 매핑 (패딩=-1)"),
    "los_nlos_flag":   ("(T,R,P)",  "int32",   "ray별 LoS(1)/NLoS(0)"),
    "counts":          ("(T,R)",    "int32",   "(TX,RX)별 유효 ray 수"),

    # --- channel-level (batch 고유, MIMO 공분산 — 유일 출처) ---
    "R_TX":            ("(T,R,At,At)", "complex128", "TX 공분산"),
    "R_RX":            ("(T,R,Ar,Ar)", "complex128", "RX 공분산"),

    # --- 집계 / 메타 ---
    "rsrp_all":        ("(T,R)",    "float64", "총 RSRP [dBm] (dead=-inf)"),
    "los_all":         ("(T,R)",    "bool",    "(TX,RX)별 LoS 존재"),
    "tx_positions":    ("(T,3)",    "float64", "TX (x,y,z)"),
    "rx_positions":    ("(R,3)",    "float64", "RX (x,y,z)"),
    "rx_valid_mask":   ("(R,)",     "int8",    "0=valid 1=dead 2=rt_fail (비파괴)"),
    "num_tx_ant":      ("(1,)",     "int64",   "TX 안테나 포트 수"),
    "num_rx_ant":      ("(1,)",     "int64",   "RX 안테나 포트 수"),
    "target_tx_index": ("(1,)",     "int64",   ""),
    "target_rx_index": ("(1,)",     "int64",   ""),
    "frequency_ghz":   ("()",       "float64", ""),
    "max_paths":       ("(1,)",     "int64",   "K (path 패딩 길이)"),
    "max_rays":        ("(1,)",     "int64",   "P (ray 패딩 길이)"),
}


def describe() -> str:
    """스키마를 사람이 읽기 좋은 표 문자열로."""
    lines = ["superset NPZ 스키마 (T=num_tx, R=num_rx, K=max_paths, P=max_rays):"]
    for k, (shape, dt, desc) in SUPERSET_KEYS.items():
        lines.append(f"  {k:18s} {shape:14s} {dt:11s} {desc}")
    return "\n".join(lines)


def validate(npz: dict) -> list[str]:
    """superset dict 의 필수 키/차원 정합성 가벼운 검증. 문제 메시지 리스트 반환(빈=정상)."""
    problems: list[str] = []
    for key in ("tau", "path_tau", "rsrp_all", "rx_positions", "counts", "path_counts"):
        if key not in npz:
            problems.append(f"필수 키 누락: {key}")
    if "rsrp_all" in npz:
        arr = np.asarray(npz["rsrp_all"])
        if arr.ndim != 2:
            problems.append(f"rsrp_all 은 (T,R) 2D 여야 함 (현재 {arr.shape})")
    if "tau" in npz and "rsrp_all" in npz:
        t_r = np.asarray(npz["rsrp_all"]).shape
        tau_tr = np.asarray(npz["tau"]).shape[:2]
        if t_r != tau_tr:
            problems.append(f"tau (T,R)={tau_tr} 가 rsrp_all (T,R)={t_r} 와 불일치")
    return problems


if __name__ == "__main__":
    print(describe())
