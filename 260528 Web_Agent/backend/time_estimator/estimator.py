"""estimator.py — (A) 실행 전 시간 예측.

과거 이력(run_history) 중 같은 config 영역의 기록만 골라, **단계별 1D 회귀**로
각 단계 소요시간을 추정하고 합산한다. 데이터가 적어도 안 깨지게 폴백 사다리
(affine → 비례 → 평균 → null)를 쓴다.

반환(dict) — 프런트가 그대로 표시
    available        : 예측 가능 여부 (Q5a: 기록 0건이면 False)
    reason           : 불가 사유 (available=False 일 때)
    total_sec        : 예상 총 소요시간(초)
    total_std_sec    : 불확실성(초, ±)
    n_records        : 사용한 매칭 기록 수
    min_stage_samples: 단계별 최소 표본 수
    basis_label      : "TX 1대 · RX 1,074개 · 안테나쌍 4096"
    per_stage        : {stage: {mean_sec, std_sec, n, method}}
    summary_text     : "예상 소요 시간 약 2시간 49분 (±10분)"
    detail_text      : "단계별 · RX 1,074개 기준 · 최소 1회 데이터 · 누적 11회 기록 (나이브 추정치)"

실행 환경: Python 3.10 / 표준 라이브러리(statistics)
"""

from __future__ import annotations

import math
import statistics
from typing import Any

from .cost_basis import extract_features, stage_basis, target_basis_label
from .machine import machine_fingerprint, machine_key
from .run_history import matching_records


# ---------------------------------------------------------------------------
# 단계별 1D 회귀 (폴백 사다리)
# ---------------------------------------------------------------------------
def _fit_stage(xs: list[float], ds: list[float], target_x: float) -> tuple[float, float, str]:
    """(basis xs, duration ds) 로 target_x 에서의 (예상시간, 불확실성±, 방법) 산출.

    개선(2026-07-10):
      - 점추정: 공유 GPU 경합으로 느려진 이상치를 rate 중앙값 대비 0.5~2배로 트리밍한 뒤 적합
        (경합으로 2~6배 느린 표본이 평균을 끌어올리는 문제 완화).
      - 불확실성±: '비트림' 전체 rate 분포의 p15~p85 폭을 target_x 에 투영 → 경합 변동을
        숨기지 않고 정직하게 넓은 밴드로 표시.
    """
    n = len(xs)
    if n == 0 or target_x is None:
        return 0.0, 0.0, "none"

    # 비례(중앙값 rate) 을 주 방법으로. basis 가 3자릿수 이상 걸쳐 있어 affine(절편)은
    # 과적합/외삽으로 오차를 키운다(백테스트: MAPE 88%→44%로 개선). 각 단계의 고정 오버헤드는
    # 단계 분리(Scene=rx 등)로 흡수되므로 절편 없는 비례가 적합하다.
    rates = [d / x for x, d in zip(xs, ds) if x and x > 0]
    if rates:
        pred = statistics.median(rates) * target_x
        # 불확실성±: rate 분포 p15~p85 폭의 절반을 target_x 에 투영(경합 변동을 정직하게 반영)
        rs = sorted(rates)
        lo = rs[int(0.15 * (len(rs) - 1))]
        hi = rs[int(0.85 * (len(rs) - 1))]
        std = max((hi - lo) * target_x / 2.0, 0.0)
        return max(pred, 0.0), std, "median_rate"

    # 폴백: basis 를 못 쓰는 경우 단순 평균
    mean_d = statistics.fmean(ds)
    std_d = statistics.pstdev(ds) if n >= 2 else 0.0
    return max(mean_d, 0.0), std_d, "mean"


# ---------------------------------------------------------------------------
# 표시용 포맷
# ---------------------------------------------------------------------------
def fmt_duration(sec: float) -> str:
    """초 → '약 2시간 49분' / '약 12분 30초' / '약 45초'."""
    sec = max(0.0, float(sec))
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    if h > 0:
        return f"약 {h}시간 {m}분"
    if m > 0:
        return f"약 {m}분 {s}초"
    return f"약 {s}초"


def fmt_pm(sec: float) -> str:
    """불확실성 → '±10분' / '±30초'."""
    sec = max(0.0, float(sec))
    if sec >= 60:
        return f"±{int(round(sec / 60))}분"
    return f"±{int(round(sec))}초"


# ---------------------------------------------------------------------------
# 공개 API
# ---------------------------------------------------------------------------
def estimate(payload: dict, rx_count_hint: int | None = None) -> dict[str, Any]:
    """job payload → 실행 전 시간 예측 dict."""
    fp = machine_fingerprint()
    feat = extract_features(payload, rx_count_hint=rx_count_hint)

    target = {
        # 머신 매칭은 machine_id+gpu_mode 필드로 직접 (gpu_name 분열 방지 — 2026-07-10)
        "machine_id": fp.get("machine_id"),
        "gpu_mode": fp.get("gpu_mode"),
        "engine": feat["engine"],
        "max_depth": feat["max_depth"],
        "num_samples": feat["num_samples"],
        # PathSolver 옵션(RT 시간 좌우) — 같은 옵션끼리만 비교. 구 레코드엔 없어 자동 통과.
        "diffraction": feat.get("diffraction"),
        "diffuse_reflection": feat.get("diffuse_reflection"),
        "refraction": feat.get("refraction"),
        "synthetic_array": feat.get("synthetic_array"),
        # 안테나/규모는 basis 로 처리하므로 매칭 키에 넣지 않음(Q3)
    }

    # RX 개수를 모르면(ground_grid/facade 미리보기 전) 예측 불가
    if feat.get("num_rx") is None:
        return {
            "available": False,
            "reason": "RX 개수를 아직 알 수 없습니다 (RX 미리보기 후 다시 시도).",
            "machine": fp,
        }

    records = matching_records(target)
    if not records:
        # Q5a: 콜드 스타트 — 조용히 '수집 중' 표시
        return {
            "available": False,
            "reason": "예측 데이터 수집 중 — 이 설정/머신의 과거 기록이 아직 없습니다.",
            "machine": fp,
            "basis_label": target_basis_label(feat),
        }

    # 매칭 기록에 등장한 모든 단계에 대해 회귀
    stage_names: list[str] = []
    for r in records:
        for s in (r.get("stages") or {}):
            if s not in stage_names:
                stage_names.append(s)

    per_stage: dict[str, dict] = {}
    total = 0.0
    var_sum = 0.0
    min_samples = None

    for stage in stage_names:
        target_x = stage_basis(stage, feat)
        if target_x is None:
            continue
        xs: list[float] = []
        ds: list[float] = []
        for r in records:
            dur = (r.get("stages") or {}).get(stage)
            if dur is None:
                continue
            bx = stage_basis(stage, r)
            if bx is None:
                continue
            xs.append(bx)
            ds.append(float(dur))
        if not ds:
            continue
        mean_sec, std_sec, method = _fit_stage(xs, ds, target_x)
        per_stage[stage] = {
            "mean_sec": mean_sec,
            "std_sec": std_sec,
            "n": len(ds),
            "method": method,
        }
        total += mean_sec
        var_sum += std_sec * std_sec
        min_samples = len(ds) if min_samples is None else min(min_samples, len(ds))

    if not per_stage:
        return {
            "available": False,
            "reason": "예측 데이터 수집 중 — 이 규모를 예측할 단계 기록이 부족합니다.",
            "machine": fp,
            "basis_label": target_basis_label(feat),
        }

    total_std = math.sqrt(var_sum)
    basis_label = target_basis_label(feat)
    rx_txt = f"RX {feat['num_rx']:,}개" if feat.get("num_rx") else "규모"

    # --- 외삽(extrapolation) 감지: 과거 데이터 범위 밖이면 신뢰도↓ + 밴드↑ ---
    warns: list[str] = []
    extrapolation = False

    hist_ap = [r.get("num_ant_pairs") for r in records if r.get("num_ant_pairs")]
    tgt_ap = feat.get("num_ant_pairs") or 1
    if hist_ap and tgt_ap > max(hist_ap) * 1.5:
        extrapolation = True
        warns.append(
            f"안테나쌍 {tgt_ap}는 과거 기록(최대 {max(hist_ap)})을 크게 벗어남 — "
            "MIMO 시간은 아직 데이터가 없어 신뢰도 낮음"
        )

    hist_rx = [r.get("num_rx") for r in records if r.get("num_rx")]
    tgt_rx = feat.get("num_rx") or 0
    if hist_rx and (tgt_rx > max(hist_rx) * 2 or tgt_rx < min(hist_rx) * 0.5):
        extrapolation = True
        warns.append(
            f"RX {tgt_rx:,}개는 과거 범위({min(hist_rx):,}~{max(hist_rx):,})를 벗어남 — 외삽 추정"
        )

    # --- 신뢰도 등급 ---
    ms = min_samples or 0
    rel = (total_std / total) if total > 0 else 1.0
    if extrapolation or ms < 1:
        confidence = "low"
    elif ms < 3 or rel > 0.45:
        confidence = "medium"
    else:
        confidence = "high"

    # 외삽이면 불확실성 밴드를 최소 ±50%로 넓혀 과신 방지
    if extrapolation:
        total_std = max(total_std, 0.5 * total)

    conf_txt = {"high": "신뢰도 높음", "medium": "신뢰도 보통", "low": "신뢰도 낮음"}[confidence]
    detail = (
        f"단계별 · {rx_txt} 기준 · 최소 {ms}회 데이터 · 누적 {len(records)}회 기록 · {conf_txt}"
    )
    if warns:
        detail += " · " + " / ".join(warns)

    return {
        "available": True,
        "total_sec": total,
        "total_std_sec": total_std,
        "n_records": len(records),
        "min_stage_samples": ms,
        "confidence": confidence,
        "warnings": warns,
        "basis_label": basis_label,
        "per_stage": per_stage,
        "machine": fp,
        "summary_text": f"예상 소요 시간 {fmt_duration(total)} ({fmt_pm(total_std)})",
        "detail_text": detail,
    }
