"""session_labeler.py — 세션 UUID + 자동 라벨 생성.

라벨 포맷 (plan 결정):
    {Scene}_{BS}x{BS}_{UE}x{UE}_{MetricsHash}_{Timestamp}_{uuid8}

예시:
    Jonggak_64x64_4x4_RSRP-PADP-SWOMP_2605281402_a1b2c3d4

uuid는 격리 root key. label은 사용자 친화 표시 (편집 가능).
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import re
import uuid as _uuid
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path


# ---------------------------------------------------------------------------
# Label 생성
# ---------------------------------------------------------------------------
_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]")

# 표시용 라벨/이름: 파일시스템 경로가 아니라 화면 표시·메타(JSON, ensure_ascii=False)에만 쓰이므로
# 한글 등 유니코드는 보존하고, 경로/제어 위험 문자만 '_' 로 치환한다.
# (세션 디렉터리는 UUID, RT 산출물 파일명은 uuid 기반 map_title 을 별도 sanitize 하므로 안전)
_UNSAFE_DISPLAY_RE = re.compile(r'[\x00-\x1f\x7f/\\:*?"<>|]')


def sanitize(name: str, max_len: int = 24) -> str:
    """파일명용 엄격 정규화 (ASCII 영숫자 + ._- 만). 실제 파일/폴더명에 쓸 때 사용."""

    return _SAFE_RE.sub("_", name)[:max_len] or "scene"


def sanitize_display(name: str, max_len: int = 120) -> str:
    """표시용 라벨/이름 정규화 — 한글 등 유니코드 보존, 경로/제어 위험 문자만 치환."""

    cleaned = _UNSAFE_DISPLAY_RE.sub("_", (name or "")).strip()
    cleaned = cleaned.lstrip(".")           # 숨김파일/상위경로(.. ) 흉내 방지
    cleaned = cleaned.strip()
    return cleaned[:max_len] or "scene"


def metrics_hash(metrics: list[str], max_show: int = 3) -> str:
    """선택한 metric을 라벨에 짧게 표시.

    - 처음 max_show개만 dash로 연결 (예: RSRP-PADP-SWOMP)
    - 나머지는 short hash로 표기
    """

    if not metrics:
        return "NONE"
    short_map = {
        "rsrp": "RSRP",
        "padp": "PADP",
        "pdp": "PDP",
        "swomp_beams": "SWOMP",
        "greedy_beams": "GRDY",
        "su_capacity": "CAP",
        "marginal_ccm": "CCM",
        "coupling_matrix": "OMG",
        "mean_channel": "HBAR",
        "uplink_beams": "UL",
        "qie_clustering": "QIE",
        "padp_dft_alignment": "PDDFT",
        "beam_pattern": "BPAT",
        "eigenbeam_pattern": "EBPAT",
        "coverage_map": "COV",
        "ray_dump": "RAW",
        "ae_ofdm_channel": "AEOF",
        "separability": "SEP",
        "ray_stats": "STAT",
    }
    pretty = [short_map.get(m, m.upper()) for m in metrics]
    if len(pretty) <= max_show:
        return "-".join(pretty)
    head = "-".join(pretty[:max_show])
    rest_hash = hashlib.sha1(",".join(metrics[max_show:]).encode()).hexdigest()[:4]
    return f"{head}-h{rest_hash}"


def kst_short_ts() -> str:
    """YYMMDDhhmm in KST."""

    kst = _dt.timezone(_dt.timedelta(hours=9))
    return _dt.datetime.now(kst).strftime("%y%m%d%H%M")


# ---------------------------------------------------------------------------
# 세션 모델
# ---------------------------------------------------------------------------
@dataclass
class SessionMeta:
    """세션 디렉토리 안의 session_meta.json에 저장될 메타."""

    uuid: str
    label: str
    created_at_kst: str
    scene_name: str = ""
    # 생성 시 사용자가 입력한 이름(대표 이름). scene 로드로 덮어쓰지 않음. (2026-06-20 추가)
    display_name: str = ""
    bs_rows: int = 1
    bs_cols: int = 1
    ue_rows: int = 1
    ue_cols: int = 1
    metrics: list[str] = field(default_factory=list)
    notes: str = ""
    # --- user 태깅 (멀티유저: 이름표, 인증 아님) ---
    user_id: str = ""
    user_name: str = ""
    # --- 시각화 스킨 (scene_library 의 텍스처 GLB 이름) ---
    skin_name: str = ""
    # --- 잡 큐/실행 상태 (세션 중심: 세션이 Q→P→Done 으로 이동) ---
    status: str = "idle"          # idle|queued|processing|done|failed|interrupted|cancelled
    progress: float = 0.0         # 0.0~1.0 (마지막 알려진 값)
    current_stage: str = ""
    engine: str = ""              # p1a|batch (표시용)
    job_id: str = ""              # 마지막/현재 job id
    queued_at: str = ""
    started_at: str = ""
    finished_at: str = ""
    last_error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def build_label(
    scene_name: str,
    bs_rows: int,
    bs_cols: int,
    ue_rows: int,
    ue_cols: int,
    metrics: list[str],
    short_uuid: str,
) -> str:
    """라벨 문자열 조립."""

    parts = [
        sanitize_display(scene_name, max_len=40) or "scene",
        f"{bs_rows}x{bs_cols}",
        f"{ue_rows}x{ue_cols}",
        metrics_hash(metrics),
        kst_short_ts(),
        short_uuid,
    ]
    return "_".join(parts)


def create_session(
    base_dir: Path,
    scene_name: str,
    bs_rows: int,
    bs_cols: int,
    ue_rows: int,
    ue_cols: int,
    metrics: list[str],
    notes: str = "",
    user_id: str = "",
    user_name: str = "",
) -> SessionMeta:
    """새 세션 디렉토리 생성 + meta json 저장."""

    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    sid = _uuid.uuid4()
    short = sid.hex[:8]
    label = build_label(scene_name, bs_rows, bs_cols, ue_rows, ue_cols, metrics, short)
    meta = SessionMeta(
        uuid=str(sid),
        label=label,
        created_at_kst=_dt.datetime.now(_dt.timezone(_dt.timedelta(hours=9))).isoformat(),
        scene_name=scene_name,
        display_name=scene_name,  # 생성 시 입력 이름을 대표 이름으로 고정 보존
        bs_rows=bs_rows,
        bs_cols=bs_cols,
        ue_rows=ue_rows,
        ue_cols=ue_cols,
        metrics=list(metrics),
        notes=notes,
        user_id=user_id,
        user_name=user_name,
    )
    sdir = base_dir / str(sid)
    sdir.mkdir(parents=True, exist_ok=True)
    save_meta(sdir, meta)
    return meta


def save_meta(session_dir: Path, meta: SessionMeta) -> None:
    """meta json 원자적 저장 — temp 에 쓰고 os.replace 로 교체.
    (다른 프로세스가 읽는 순간에도 항상 '완전한 예전 파일 또는 완전한 새 파일'만 보임 → 반쪽 읽기/500 방지)
    """
    import json
    import os
    import tempfile

    d = Path(session_dir)
    p = d / "session_meta.json"
    data = json.dumps(meta.to_dict(), indent=2, ensure_ascii=False)
    try:
        fd, tmp = tempfile.mkstemp(dir=str(d), prefix=".session_meta.", suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(data)
        os.replace(tmp, p)   # 원자적 교체
    except Exception:
        try:
            p.write_text(data, encoding="utf-8")  # 폴백
        except Exception:
            pass


def load_meta(session_dir: Path) -> SessionMeta | None:
    """session_meta.json 로드. (알 수 없는 키는 무시 — 스키마 진화 대비)"""

    import json

    p = Path(session_dir) / "session_meta.json"
    if not p.exists():
        return None
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        # 쓰기 도중이거나 손상된 경우: 대시보드가 500 나지 않도록 조용히 스킵
        return None
    if not isinstance(raw, dict):
        return None
    known = {f.name for f in fields(SessionMeta)}
    filtered = {k: v for k, v in raw.items() if k in known}
    return SessionMeta(**filtered)


def update_meta(session_dir: Path, **changes) -> SessionMeta | None:
    """세션 메타의 일부 필드만 갱신하고 저장."""

    meta = load_meta(session_dir)
    if meta is None:
        return None
    for k, v in changes.items():
        if hasattr(meta, k):
            setattr(meta, k, v)
    save_meta(session_dir, meta)
    return meta


def update_label(session_dir: Path, new_label: str) -> SessionMeta:
    """사용자가 라벨 편집할 때 호출."""

    meta = load_meta(session_dir)
    if meta is None:
        raise FileNotFoundError(f"session_meta.json not found in {session_dir}")
    meta.label = sanitize_display(new_label, max_len=120) or meta.label
    save_meta(session_dir, meta)
    return meta


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        m = create_session(
            base_dir=Path(td),
            scene_name="Jonggak",
            bs_rows=64, bs_cols=64,
            ue_rows=4, ue_cols=4,
            metrics=["rsrp", "padp", "swomp_beams"],
        )
        print("uuid :", m.uuid)
        print("label:", m.label)
        m2 = update_label(Path(td) / m.uuid, "Jonggak experiment 1")
        print("after edit:", m2.label)
