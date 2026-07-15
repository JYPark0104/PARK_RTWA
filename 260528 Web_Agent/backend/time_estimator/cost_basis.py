"""cost_basis.py — 단계별 cost basis + config 매칭 키 (예측의 심장).

사용자 확정 사항
---------------
- Q2/Q7: batch/intg 엔진 우선.
- Q3: 안테나 수도 실행시간을 '지배적으로' 좌우하므로 RT basis 에 포함.

단계별 cost basis (지배항 1개 원칙, 두 변수가 곱해지면 곱을 1D로)
    Scene   : num_rx                                  (RX 배치 레이캐스팅)
    RT      : num_tx × num_rx × num_ant_pairs          ← Q3: 안테나 수 포함(지배적)
    Intg    : num_tx × num_rx
    Output* : num_tx × num_rx
    (미지 단계 폴백: num_rx)

여기서 num_ant_pairs = (BS 안테나 수) × (UE 안테나 수).

config 매칭 키 (형태/배율 파라미터는 basis 에서 빼고 매칭으로 처리)
    machine_key (필수), engine (필수),
    max_depth, num_samples  (관대한 와일드카드: 한쪽에 없으면 통과)

실행 환경: Python 3.10 / 표준 라이브러리
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# num_rx 해석 (실행 전 예측용 — 레이캐스팅 없이 config 로 최대한 계산)
# ---------------------------------------------------------------------------
def resolve_num_rx(payload: dict, rx_count_hint: int | None = None) -> int | None:
    """job payload 로부터 RX 개수를 추정.

    - rx_count_hint 가 주어지면 그대로 신뢰(프런트가 미리보기로 계산한 실제 개수).
    - grid/explicit/radial/street/clicks 는 config 만으로 계산 가능.
    - ground_grid/facade 는 레이캐스팅 필요 → hint 없으면 None (상한 근사 가능).
    """
    if rx_count_hint and rx_count_hint > 0:
        return int(rx_count_hint)

    rc = payload.get("rx_clicks") or {}
    if rc.get("positions"):
        return len(rc["positions"])

    rg = payload.get("rx_grid") or {}
    method = rg.get("method", "grid")
    nz = max(1, len(rg.get("z_values") or [1.5]))

    if method == "grid":
        xn, yn = rg.get("x_num"), rg.get("y_num")
        if xn and yn:
            return int(xn) * int(yn) * nz
    elif method == "explicit":
        xc, yc = rg.get("x_coords"), rg.get("y_coords")
        if xc and yc:
            return len(xc) * len(yc) * nz
    elif method == "radial":
        radii, an = rg.get("radii_m"), rg.get("angles_num")
        if radii and an:
            return len(radii) * int(an) * nz
    elif method == "street":
        n_pts = rg.get("num_points")
        if n_pts:
            return int(n_pts) * nz
    elif method == "ground_grid":
        # 레이캐스팅으로 지면 hit 만 남으므로 grid_n^2 는 '상한'일 뿐 — hint 없으면 미상.
        return None
    elif method == "facade":
        return None
    return None


# ---------------------------------------------------------------------------
# feature 추출
# ---------------------------------------------------------------------------
def extract_features(payload: dict, rx_count_hint: int | None = None,
                     num_rx_actual: int | None = None,
                     num_tx_actual: int | None = None) -> dict[str, Any]:
    """job payload(또는 disk config) → 예측/기록용 feature dict.

    num_rx_actual / num_tx_actual 가 주어지면(기록 시점, 실제 실행값) 그것을 우선한다.
    """
    rt = payload.get("rt", {}) or {}
    antenna = payload.get("antenna", {}) or {}
    simple = antenna.get("simple", {}) or {}

    num_tx = num_tx_actual if num_tx_actual else len(payload.get("tx_list", []) or [])
    num_rx = num_rx_actual if num_rx_actual else resolve_num_rx(payload, rx_count_hint)

    bs_ant = int(simple.get("bs_rows", 1) or 1) * int(simple.get("bs_cols", 1) or 1)
    ue_ant = int(simple.get("ue_rows", 1) or 1) * int(simple.get("ue_cols", 1) or 1)

    return {
        "engine": rt.get("engine", "p1a"),
        "num_tx": int(num_tx or 0),
        "num_rx": (int(num_rx) if num_rx else None),
        "bs_ant": bs_ant,
        "ue_ant": ue_ant,
        "num_ant_pairs": bs_ant * ue_ant,
        "max_depth": rt.get("max_depth"),
        "num_samples": rt.get("num_samples"),
        "batch_size": rt.get("batch_size"),
        "frequency_ghz": rt.get("frequency_ghz"),
        # PathSolver 플래그 — RT 시간에 큰 영향(회절/확산반사/투과 켜면 몇 배 느려짐).
        #   기록/매칭에 포함해 '같은 연산 옵션'끼리만 비교 → 분산 축소. (2026-07-10 추가)
        "diffraction": rt.get("pathsolver_diffraction"),
        "diffuse_reflection": rt.get("pathsolver_diffuse_reflection"),
        "refraction": rt.get("pathsolver_refraction"),
        "specular_reflection": rt.get("pathsolver_specular_reflection"),
        "edge_diffraction": rt.get("pathsolver_edge_diffraction"),
        "synthetic_array": rt.get("pathsolver_synthetic_array"),
    }


# ---------------------------------------------------------------------------
# 단계별 basis
# ---------------------------------------------------------------------------
def stage_basis(stage: str, feat: dict) -> float | None:
    """stage 이름 + feature → 그 단계의 1D cost basis 값. 계산 불가 시 None."""
    ntx = feat.get("num_tx") or 0
    nrx = feat.get("num_rx")
    nap = feat.get("num_ant_pairs") or 1
    if nrx is None:
        return None
    s = (stage or "").lower()

    if s.startswith("rt"):
        return float(ntx * nrx * nap) if ntx and nrx else None
    if s.startswith("intg"):
        return float(ntx * nrx) if ntx and nrx else None
    if s.startswith("output"):
        return float(ntx * nrx) if ntx and nrx else None
    if s.startswith("viz"):
        return float(ntx * nrx) if ntx and nrx else None
    if s.startswith("scene"):
        return float(max(nrx, 1))
    # 미지 단계 폴백
    return float(max(nrx, 1))


def target_basis_label(feat: dict) -> str:
    """A 예측 표시용 '기준' 라벨 (MGA의 '1,074장 기준' 대응)."""
    nrx = feat.get("num_rx")
    ntx = feat.get("num_tx")
    if nrx:
        return f"TX {ntx}대 · RX {nrx:,}개 · 안테나쌍 {feat.get('num_ant_pairs')}"
    return f"TX {ntx}대 · 안테나쌍 {feat.get('num_ant_pairs')}"


# ---------------------------------------------------------------------------
# config 매칭
# ---------------------------------------------------------------------------
# 반드시 일치 (machine_key 대신 machine_id+gpu_mode 필드로 직접 — gpu_name 분열 방지)
_STRICT_KEYS = ("machine_id", "gpu_mode", "engine")
# 한쪽에 없으면 통과, 둘 다 있으면 일치해야 (구 레코드엔 플래그가 없어 자동 통과 → 하위호환)
_WILDCARD_KEYS = (
    "max_depth", "num_samples",
    "diffraction", "diffuse_reflection", "refraction", "synthetic_array",
)


def match_config(target: dict, record: dict) -> bool:
    """record(과거 기록)가 target(예측 대상)과 같은 config 영역인지 판정.

    - strict 키: 양쪽 모두 있고 값이 같아야 함(다르거나 record에 없으면 불일치).
    - wildcard 키: 한쪽이라도 없으면 통과, 둘 다 있으면 같아야 함.
    """
    for k in _STRICT_KEYS:
        tv, rv = target.get(k), record.get(k)
        if tv is None:
            continue  # target 이 모르면 이 키로는 거르지 않음
        if rv is None or rv != tv:
            return False
    for k in _WILDCARD_KEYS:
        tv, rv = target.get(k), record.get(k)
        if tv is None or rv is None:
            continue
        if tv != rv:
            return False
    return True
