"""app.py — FastAPI 메인 진입점.

엔드포인트:
- POST   /api/sessions                              세션 생성 (UUID + 라벨)
- GET    /api/sessions                              세션 목록
- GET    /api/sessions/{uuid}                       세션 정보
- PATCH  /api/sessions/{uuid}/label                 라벨 편집
- POST   /api/sessions/{uuid}/scene                 OBJ/PLY/ZIP 업로드 → scene.xml 변환
- POST   /api/sessions/{uuid}/job                   잡 제출 (TX/RX + RT + metrics)
- GET    /api/jobs/{job_id}                         잡 상태
- POST   /api/jobs/{job_id}/cancel                  잡 취소
- WS     /ws/jobs/{job_id}                          진행률/로그 스트림
- GET    /api/metrics/catalog                       18 metric 카탈로그
- GET    /api/metrics/presets                       7종 프리셋
- GET    /api/sessions/{uuid}/files                 결과 NPZ/PNG 트리
- GET    /api/sessions/{uuid}/files/{path}          개별 파일 다운로드
- POST   /api/sessions/{uuid}/zip                   세션 전체 zip
- GET    /api/health                                헬스체크
"""

from __future__ import annotations

import io
import json
import logging
import shutil
import zipfile
from pathlib import Path
from typing import Optional

from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from .jobs.job_runner import runner
from .jobs.pipeline_executor import SESSIONS_ROOT, register_default_worker
from .jobs.scene_builder import build_scene, build_scene_from_files
from .jobs.session_labeler import (
    SessionMeta,
    create_session,
    load_meta,
    save_meta,
    update_label,
    update_meta,
)
from .jobs.queue_store import make_store, save_session_config, load_session_config
from .metric_catalog import (
    METRIC_CATALOG,
    PRESETS,
    list_user_visible_metrics,
    requires_rx,
    resolve_metrics,
)
from .schemas import (
    JobSubmitRequest,
    SessionCreateRequest,
    SessionLabelUpdate,
    UserCreateRequest,
    UserRenameRequest,
)
from .ws_manager import broker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
log = logging.getLogger("app")

app = FastAPI(title="RT Web Agent", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def _startup() -> None:
    SESSIONS_ROOT.mkdir(parents=True, exist_ok=True)
    register_default_worker()
    await runner.start()

    # 재시작 복구: processing → interrupted, queued → 순서 유지하여 재등록.
    try:
        store = make_store()
        queued = store.recover()
        for uuid in queued:
            job = runner.submit(kind="pipeline", session_uuid=uuid,
                                payload={"session_uuid": uuid})
            update_meta(SESSIONS_ROOT / uuid, job_id=job.job_id)
        if queued:
            log.info("재시작 복구: queued 세션 %d개 재등록", len(queued))
    except Exception as exc:  # noqa: BLE001
        log.exception("큐 복구 실패: %s", exc)
    try:
        from backend.jobs.runtime_env import init_runtime
        summary = init_runtime()  # WEBAGENT_USE_GPU env로 토글
        policy = summary.get("gpu_policy") or {}
        if summary.get("gpus"):
            log.info(
                "GPU 모드: %d장 — %s | mitsuba=%s | %s",
                len(summary["gpus"]),
                ", ".join(g.get("device_name") or g["name"] for g in summary["gpus"]),
                summary.get("mitsuba_variant"),
                policy.get("reason", ""),
            )
        else:
            log.warning(
                "CPU 모드 — %s (강제 GPU: WEBAGENT_FORCE_CPU=0, Blackwell 실험: WEBAGENT_ALLOW_BLACKWELL_GPU=1)",
                policy.get("reason", "GPU 미감지"),
            )
    except Exception as exc:
        log.exception("runtime_env 초기화 실패: %s", exc)
    log.info("startup complete (sessions=%s)", SESSIONS_ROOT)


@app.on_event("shutdown")
async def _shutdown() -> None:
    await runner.shutdown()


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
@app.get("/api/health")
async def health() -> dict:
    return {"ok": True, "sessions_root": str(SESSIONS_ROOT)}


# ---------------------------------------------------------------------------
# Metric catalog / presets
# ---------------------------------------------------------------------------
@app.get("/api/metrics/catalog")
async def get_metric_catalog() -> dict:
    return {"metrics": list_user_visible_metrics()}


@app.get("/api/metrics/presets")
async def get_presets() -> dict:
    return {"presets": PRESETS}


@app.post("/api/metrics/resolve")
async def resolve_metric(payload: dict) -> dict:
    selected = payload.get("metrics") or []
    plan = resolve_metrics(selected)
    return plan.to_dict()


# ---------------------------------------------------------------------------
# Users (이름표 — 인증 아님, 신뢰된 연구실 내부망 전제)
# ---------------------------------------------------------------------------
@app.get("/api/users")
async def list_users() -> dict:
    return {"users": make_store().list_users()}


@app.post("/api/users")
async def add_user(req: UserCreateRequest) -> dict:
    try:
        return make_store().add_user(req.name)
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@app.patch("/api/users/{user_id}")
async def rename_user(user_id: str, req: UserRenameRequest) -> dict:
    try:
        return make_store().rename_user(user_id, req.name)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except KeyError:
        raise HTTPException(404, "user not found")


@app.delete("/api/users/{user_id}")
async def delete_user(user_id: str) -> dict:
    return {"deleted": make_store().delete_user(user_id)}


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------
@app.post("/api/sessions")
async def post_session(req: SessionCreateRequest) -> dict:
    a = req.antenna.simple
    meta = create_session(
        base_dir=SESSIONS_ROOT,
        scene_name=req.scene_name,
        bs_rows=a.bs_rows,
        bs_cols=a.bs_cols,
        ue_rows=a.ue_rows,
        ue_cols=a.ue_cols,
        metrics=req.metrics.metrics,
        notes=req.notes,
        user_id=req.user_id,
        user_name=req.user_name,
    )
    return meta.to_dict()


@app.get("/api/sessions")
async def list_sessions() -> dict:
    out: list[dict] = []
    if SESSIONS_ROOT.exists():
        for d in sorted(SESSIONS_ROOT.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if not d.is_dir():
                continue
            meta = load_meta(d)
            if meta:
                out.append(meta.to_dict())
    return {"sessions": out}


@app.get("/api/sessions/{uuid}")
async def get_session(uuid: str) -> dict:
    meta = load_meta(SESSIONS_ROOT / uuid)
    if not meta:
        raise HTTPException(404, "session not found")
    return meta.to_dict()


@app.patch("/api/sessions/{uuid}/label")
async def patch_session_label(uuid: str, payload: SessionLabelUpdate) -> dict:
    sdir = SESSIONS_ROOT / uuid
    if not sdir.exists():
        raise HTTPException(404, "session not found")
    meta = update_label(sdir, payload.label)
    return meta.to_dict()


@app.delete("/api/sessions/{uuid}")
async def delete_session(uuid: str) -> dict:
    sdir = SESSIONS_ROOT / uuid
    # 큐/처리 중이면 취소 후 큐에서 제거
    meta = load_meta(sdir)
    if meta:
        store = make_store()
        if meta.job_id:
            runner.cancel(meta.job_id)
        store.remove_from_order(uuid)
    if sdir.exists():
        shutil.rmtree(sdir)
    return {"deleted": True}


# ---------------------------------------------------------------------------
# Scene upload
# ---------------------------------------------------------------------------
@app.post("/api/sessions/{uuid}/scene")
async def post_scene(
    uuid: str,
    file: UploadFile = File(...),
    material: str = Form("itu_concrete"),
) -> dict:
    sdir = SESSIONS_ROOT / uuid
    if not sdir.exists():
        raise HTTPException(404, "session not found")
    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".obj", ".ply", ".zip"}:
        raise HTTPException(400, f"Unsupported file type: {suffix}")

    upload_dir = sdir / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    target = upload_dir / file.filename
    with target.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    scene_dir = sdir / "scene"
    if scene_dir.exists():
        shutil.rmtree(scene_dir)
    info = build_scene(target, scene_dir, material=material)
    return {"scene_info": info.to_dict(), "scene_xml": str(scene_dir / "scene.xml")}


@app.post("/api/sessions/{uuid}/scene_bundle")
async def post_scene_bundle(
    uuid: str,
    files: list[UploadFile] = File(...),
) -> dict:
    """Geo-Radio Env. Twin 업로드: 재질이 부여된 scene .xml + 여러 .ply(meshes) 다중 업로드.

    단일 메시 업로드(post_scene)와 달리 기본재질을 부여하지 않고, 사용자가 XML 에
    지정한 ITU 재질을 그대로 보존한 채 RT 가 로드할 scene.xml 을 구성한다.
    """
    sdir = SESSIONS_ROOT / uuid
    if not sdir.exists():
        raise HTTPException(404, "session not found")
    if not files:
        raise HTTPException(400, "업로드된 파일이 없습니다.")

    # 업로드 파일을 세션 uploads/bundle/ 에 평탄하게 저장.
    bundle_dir = sdir / "uploads" / "bundle"
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    xml_paths: list[Path] = []
    ply_paths: list[Path] = []
    for uf in files:
        name = Path(uf.filename or "").name
        suffix = Path(name).suffix.lower()
        if suffix not in {".xml", ".ply"}:
            continue  # 무관 파일(.glb/.mat 등)은 무시
        dest = bundle_dir / name
        with dest.open("wb") as f:
            shutil.copyfileobj(uf.file, f)
        if suffix == ".xml":
            xml_paths.append(dest)
        else:
            ply_paths.append(dest)

    if not xml_paths:
        raise HTTPException(400, "scene .xml 파일이 포함되어야 합니다.")
    if len(xml_paths) > 1:
        # 가장 짧은 이름의 XML 을 scene XML 로 간주.
        xml_paths.sort(key=lambda p: len(p.name))
    if not ply_paths:
        raise HTTPException(400, "메시 .ply 파일이 1개 이상 포함되어야 합니다.")

    scene_dir = sdir / "scene"
    if scene_dir.exists():
        shutil.rmtree(scene_dir)
    try:
        info = build_scene_from_files(xml_paths[0], ply_paths, scene_dir)
    except ValueError as ex:
        raise HTTPException(400, str(ex))

    update_meta(sdir, scene_name=xml_paths[0].stem)
    return {"scene_info": info.to_dict(), "scene_xml": str(scene_dir / "scene.xml")}


# ---------------------------------------------------------------------------
# Geo-Radio 청크 스테이징 업로드 (.xml + meshes 폴더를 분할 전송 → 이어붙임)
# 대용량 단일 multipart 가 프록시에서 끊기는 문제(Network Error) 회피용.
# ---------------------------------------------------------------------------
def _bundle_stage_dir(uuid: str) -> Path:
    sdir = SESSIONS_ROOT / uuid
    if not sdir.exists():
        raise HTTPException(404, "session not found")
    return sdir / "uploads" / "bundle_stage"


def _safe_basename(name: str, allowed: set[str]) -> str:
    base = Path(str(name)).name
    if not base or "/" in base or "\\" in base or base.startswith(".."):
        raise HTTPException(400, f"허용되지 않는 파일명: {name}")
    if Path(base).suffix.lower() not in allowed:
        raise HTTPException(400, f"허용되지 않는 확장자: {base}")
    return base


@app.post("/api/sessions/{uuid}/scene_stage/reset")
async def scene_stage_reset(uuid: str) -> dict:
    """스테이징 폴더 초기화 (새 업로드 시작)."""
    stage = _bundle_stage_dir(uuid)
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True, exist_ok=True)
    return {"ok": True}


@app.post("/api/sessions/{uuid}/scene_stage/chunk")
async def scene_stage_chunk(
    uuid: str,
    filename: str = Form(...),
    chunk_index: int = Form(...),
    data: UploadFile = File(...),
) -> dict:
    """파일 1개의 청크를 이어붙인다. chunk_index==0 이면 새로 쓰기, 이후 append."""
    stage = _bundle_stage_dir(uuid)
    stage.mkdir(parents=True, exist_ok=True)
    base = _safe_basename(filename, {".xml", ".ply"})
    dest = stage / base
    mode = "wb" if int(chunk_index) == 0 else "ab"
    with dest.open(mode) as f:
        shutil.copyfileobj(data.file, f)
    return {"ok": True, "file": base, "size": dest.stat().st_size}


def _stage_xml_plys(stage: Path) -> tuple[Path, list[Path]]:
    if not stage.exists():
        raise HTTPException(400, "업로드된 파일이 없습니다. 먼저 청크를 전송하세요.")
    xmls = sorted(stage.glob("*.xml"), key=lambda p: len(p.name))
    plys = sorted(stage.glob("*.ply"))
    if not xmls:
        raise HTTPException(400, "scene .xml 파일이 포함되어야 합니다.")
    if not plys:
        raise HTTPException(400, "메시 .ply 파일이 1개 이상 포함되어야 합니다.")
    return xmls[0], plys


@app.post("/api/sessions/{uuid}/scene_stage/build")
async def scene_stage_build(uuid: str) -> dict:
    """스테이징된 .xml + .ply 들로 이 세션의 씬을 빌드 (직접 적용)."""
    sdir = SESSIONS_ROOT / uuid
    if not sdir.exists():
        raise HTTPException(404, "session not found")
    stage = _bundle_stage_dir(uuid)
    xml_path, ply_paths = _stage_xml_plys(stage)

    scene_dir = sdir / "scene"
    if scene_dir.exists():
        shutil.rmtree(scene_dir)
    try:
        info = build_scene_from_files(xml_path, ply_paths, scene_dir)
    except ValueError as ex:
        raise HTTPException(400, str(ex))
    update_meta(sdir, scene_name=xml_path.stem)
    shutil.rmtree(stage, ignore_errors=True)
    return {"scene_info": info.to_dict(), "scene_xml": str(scene_dir / "scene.xml")}


@app.post("/api/sessions/{uuid}/scene_stage/save_library")
async def scene_stage_save_library(uuid: str, req: dict) -> dict:
    """스테이징된 .xml + .ply 들을 Geo-Radio 번들 라이브러리에 저장."""
    stage = _bundle_stage_dir(uuid)
    xml_path, ply_paths = _stage_xml_plys(stage)
    name = req.get("name") or xml_path.stem
    stem = _safe_lib_name(name)
    SCENE_LIBRARY_RADIO.mkdir(parents=True, exist_ok=True)
    entry = SCENE_LIBRARY_RADIO / stem
    if entry.resolve().parent != SCENE_LIBRARY_RADIO.resolve():
        raise HTTPException(400, "허용되지 않는 경로")
    if entry.exists():
        raise HTTPException(409, f"이미 같은 이름의 라이브러리가 있습니다: {stem}")
    try:
        build_scene_from_files(xml_path, ply_paths, entry)
    except ValueError as ex:
        shutil.rmtree(entry, ignore_errors=True)
        raise HTTPException(400, str(ex))
    shutil.rmtree(stage, ignore_errors=True)
    return {"ok": True, "name": stem, **(_radio_entry_summary(entry) or {})}
async def get_scene_info(uuid: str) -> dict:
    p = SESSIONS_ROOT / uuid / "scene" / "scene_info.json"
    if not p.exists():
        raise HTTPException(404, "no scene uploaded yet")
    return json.loads(p.read_text(encoding="utf-8"))


@app.get("/api/sessions/{uuid}/scene/mesh/{name}")
async def get_scene_mesh(uuid: str, name: str) -> FileResponse:
    """3D 뷰어가 메시 PLY를 다운로드 받을 때 사용."""

    p = SESSIONS_ROOT / uuid / "scene" / "meshes" / name
    if not p.exists():
        raise HTTPException(404, "mesh not found")
    return FileResponse(p, media_type="application/octet-stream", filename=name)


# ---------------------------------------------------------------------------
# Scene Library (서버 내 .ply+.xml 사전 보관 맵) — PARK_2 RT 이식
# ---------------------------------------------------------------------------
SCENE_LIBRARY = Path(__file__).resolve().parent.parent / "scene_library"
# Geo-Radio Env. Twin 번들 라이브러리: 각 엔트리는 {name}/ 폴더에
# scene.xml + meshes/*.ply + scene_info.json (finalize 완료 상태)로 저장된다.
SCENE_LIBRARY_RADIO = Path(__file__).resolve().parent.parent / "scene_library_radio"


@app.get("/api/scene_library")
async def list_scene_library() -> dict:
    """scene_library 폴더의 .ply 맵 목록 (있으면 동명 .xml 쌍 표시)."""
    items: list[dict] = []
    if SCENE_LIBRARY.exists():
        for ply in sorted(SCENE_LIBRARY.glob("*.ply")):
            xml = ply.with_suffix(".xml")
            skin = ply.with_suffix(".glb")
            items.append({
                "name": ply.stem,
                "ply": ply.name,
                "has_xml": xml.exists(),
                "has_skin": skin.exists(),
                "skin_size": (skin.stat().st_size if skin.exists() else 0),
                "size": ply.stat().st_size,
            })
    return {"scenes": items, "library_path": str(SCENE_LIBRARY)}


def _safe_lib_name(name: str) -> str:
    """라이브러리 이름 정규화 (경로 탈출 방지). 확장자 제거 후 안전 문자만."""
    stem = Path(str(name)).stem.strip()
    if not stem:
        raise HTTPException(400, "이름이 비어 있습니다.")
    # 경로 구분자/상위참조 제거
    if "/" in stem or "\\" in stem or stem.startswith("."):
        raise HTTPException(400, f"허용되지 않는 이름: {name}")
    return stem


@app.post("/api/scene_library")
async def upload_scene_library(
    file: UploadFile = File(...),
    name: str = Form(...),
) -> dict:
    """scene_library 에 새 .ply 맵 업로드 (이름 지정)."""
    SCENE_LIBRARY.mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename or "").suffix.lower()
    if suffix != ".ply":
        raise HTTPException(400, "scene_library 업로드는 .ply 파일만 지원합니다.")
    stem = _safe_lib_name(name)
    target = SCENE_LIBRARY / f"{stem}.ply"
    if target.resolve().parent != SCENE_LIBRARY.resolve():
        raise HTTPException(400, "허용되지 않는 경로")
    if target.exists():
        raise HTTPException(409, f"이미 같은 이름의 라이브러리가 있습니다: {stem}")
    with target.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"ok": True, "name": stem, "ply": target.name, "size": target.stat().st_size}


@app.delete("/api/scene_library/{name}")
async def delete_scene_library(name: str) -> dict:
    """scene_library 의 맵 삭제 (.ply + 동명 .xml + .glb 스킨)."""
    stem = _safe_lib_name(name)
    removed = []
    for ext in (".ply", ".xml", ".glb"):
        p = SCENE_LIBRARY / f"{stem}{ext}"
        if p.exists() and p.resolve().parent == SCENE_LIBRARY.resolve():
            p.unlink()
            removed.append(p.name)
    if not removed:
        raise HTTPException(404, f"라이브러리를 찾을 수 없습니다: {stem}")
    return {"deleted": removed}


# ---------------------------------------------------------------------------
# Scene Skin (시각화용 텍스처 GLB — scene_library 항목별 1:1)
# ---------------------------------------------------------------------------
@app.post("/api/scene_library/{name}/skin")
async def upload_scene_skin(name: str, file: UploadFile = File(...)) -> dict:
    """라이브러리 맵에 텍스처 GLB 스킨 업로드 ({name}.glb). RT 와 무관, 9.Scenario 시각화용."""
    stem = _safe_lib_name(name)
    if not (SCENE_LIBRARY / f"{stem}.ply").exists():
        raise HTTPException(404, f"먼저 동명의 .ply 맵이 라이브러리에 있어야 합니다: {stem}")
    suffix = Path(file.filename or "").suffix.lower()
    if suffix != ".glb":
        raise HTTPException(400, "스킨은 .glb (glTF binary) 만 지원합니다.")
    target = SCENE_LIBRARY / f"{stem}.glb"
    if target.resolve().parent != SCENE_LIBRARY.resolve():
        raise HTTPException(400, "허용되지 않는 경로")
    with target.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"ok": True, "name": stem, "skin": target.name, "size": target.stat().st_size}


@app.delete("/api/scene_library/{name}/skin")
async def delete_scene_skin(name: str) -> dict:
    stem = _safe_lib_name(name)
    p = SCENE_LIBRARY / f"{stem}.glb"
    if p.exists() and p.resolve().parent == SCENE_LIBRARY.resolve():
        p.unlink()
        return {"deleted": [p.name]}
    raise HTTPException(404, f"스킨을 찾을 수 없습니다: {stem}")


@app.get("/api/scene_library/{name}/skin")
async def get_scene_skin(name: str) -> FileResponse:
    """텍스처 GLB 스킨 다운로드 (3D 뷰어용)."""
    stem = _safe_lib_name(name)
    p = SCENE_LIBRARY / f"{stem}.glb"
    if not p.exists() or p.resolve().parent != SCENE_LIBRARY.resolve():
        raise HTTPException(404, f"스킨 없음: {stem}")
    return FileResponse(p, media_type="model/gltf-binary", filename=p.name)


@app.get("/api/sessions/{uuid}/skin")
async def get_session_skin(uuid: str) -> dict:
    """세션의 시각화 스킨 정보. meta.skin_name 우선, 없으면 scene_name 으로 추정."""
    meta = load_meta(SESSIONS_ROOT / uuid)
    if not meta:
        raise HTTPException(404, "session not found")
    candidates = [meta.skin_name, meta.scene_name]
    for cand in candidates:
        if not cand:
            continue
        stem = _safe_lib_name(cand)
        if (SCENE_LIBRARY / f"{stem}.glb").exists():
            return {"has_skin": True, "name": stem,
                    "url": f"/api/scene_library/{stem}/skin"}
    return {"has_skin": False}


@app.post("/api/sessions/{uuid}/scene_from_library")
async def scene_from_library(uuid: str, req: dict) -> dict:
    """scene_library 의 .ply 를 세션 씬으로 적용 (업로드와 동일 경로로 빌드)."""
    sdir = SESSIONS_ROOT / uuid
    if not sdir.exists():
        raise HTTPException(404, "session not found")
    name = req.get("name") or req.get("ply")
    if not name:
        raise HTTPException(400, "name (ply stem or filename) required")
    ply = SCENE_LIBRARY / name
    if ply.suffix.lower() != ".ply":
        ply = SCENE_LIBRARY / f"{name}.ply"
    if not ply.exists() or ply.resolve().parent != SCENE_LIBRARY.resolve():
        raise HTTPException(404, f"library scene not found: {name}")
    material = req.get("material", "itu_concrete")

    upload_dir = sdir / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    target = upload_dir / ply.name
    shutil.copyfile(ply, target)

    scene_dir = sdir / "scene"
    if scene_dir.exists():
        shutil.rmtree(scene_dir)
    info = build_scene(target, scene_dir, material=material)
    # 텍스처 스킨(.glb)이 라이브러리에 있으면 세션 메타에 연결 (9.Scenario 시각화용)
    skin_name = ply.stem if (SCENE_LIBRARY / f"{ply.stem}.glb").exists() else ""
    update_meta(sdir, skin_name=skin_name)
    return {"scene_info": info.to_dict(), "scene_xml": str(scene_dir / "scene.xml"),
            "source": "library", "name": ply.stem, "skin_name": skin_name}


# ---------------------------------------------------------------------------
# Scene Library (Geo-Radio) — 재질 부여된 번들(.xml + meshes/*.ply) 사전 보관
# ---------------------------------------------------------------------------
def _radio_entry_summary(entry: Path) -> dict | None:
    """라이브러리 엔트리 폴더의 scene_info.json 을 읽어 목록용 요약 반환."""
    info_p = entry / "scene_info.json"
    if not info_p.exists():
        return None
    try:
        info = json.loads(info_p.read_text(encoding="utf-8"))
    except Exception:
        return None
    mesh_files = info.get("mesh_files", [])
    size = 0
    for rel in mesh_files:
        mp = entry / rel
        if mp.exists():
            size += mp.stat().st_size
    return {
        "name": entry.name,
        "n_meshes": len(mesh_files),
        "materials": info.get("materials", []),
        "n_faces": info.get("n_faces", 0),
        "size": size,
        "has_xml": (entry / "scene.xml").exists(),
    }


@app.get("/api/scene_library_radio")
async def list_scene_library_radio() -> dict:
    """Geo-Radio 번들 라이브러리 목록."""
    items: list[dict] = []
    if SCENE_LIBRARY_RADIO.exists():
        for entry in sorted(SCENE_LIBRARY_RADIO.iterdir()):
            if not entry.is_dir() or entry.name.startswith("_"):
                continue
            summ = _radio_entry_summary(entry)
            if summ:
                items.append(summ)
    return {"scenes": items, "library_path": str(SCENE_LIBRARY_RADIO)}


@app.post("/api/scene_library_radio")
async def upload_scene_library_radio(
    files: list[UploadFile] = File(...),
    name: str = Form(...),
) -> dict:
    """Geo-Radio 번들(.xml + 여러 .ply) 을 라이브러리에 저장 (이름 지정).

    업로드 파일을 임시로 받아 build_scene_from_files 로 finalize 한 뒤,
    {name}/ 엔트리(scene.xml + meshes/ + scene_info.json)로 보관한다.
    """
    if not files:
        raise HTTPException(400, "업로드된 파일이 없습니다.")
    SCENE_LIBRARY_RADIO.mkdir(parents=True, exist_ok=True)
    stem = _safe_lib_name(name)
    entry = SCENE_LIBRARY_RADIO / stem
    if entry.resolve().parent != SCENE_LIBRARY_RADIO.resolve():
        raise HTTPException(400, "허용되지 않는 경로")
    if entry.exists():
        raise HTTPException(409, f"이미 같은 이름의 라이브러리가 있습니다: {stem}")

    stage = SCENE_LIBRARY_RADIO / f"_upload_{stem}"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True, exist_ok=True)
    try:
        xml_paths: list[Path] = []
        ply_paths: list[Path] = []
        for uf in files:
            fname = Path(uf.filename or "").name
            suffix = Path(fname).suffix.lower()
            if suffix not in {".xml", ".ply"}:
                continue
            dest = stage / fname
            with dest.open("wb") as f:
                shutil.copyfileobj(uf.file, f)
            (xml_paths if suffix == ".xml" else ply_paths).append(dest)
        if not xml_paths:
            raise HTTPException(400, "scene .xml 파일이 포함되어야 합니다.")
        if not ply_paths:
            raise HTTPException(400, "메시 .ply 파일이 1개 이상 포함되어야 합니다.")
        xml_paths.sort(key=lambda p: len(p.name))
        try:
            info = build_scene_from_files(xml_paths[0], ply_paths, entry)
        except ValueError as ex:
            shutil.rmtree(entry, ignore_errors=True)
            raise HTTPException(400, str(ex))
    finally:
        shutil.rmtree(stage, ignore_errors=True)

    return {"ok": True, "name": stem, **(_radio_entry_summary(entry) or {})}


@app.delete("/api/scene_library_radio/{name}")
async def delete_scene_library_radio(name: str) -> dict:
    """Geo-Radio 번들 라이브러리 엔트리 삭제 (폴더 전체)."""
    stem = _safe_lib_name(name)
    entry = SCENE_LIBRARY_RADIO / stem
    if entry.exists() and entry.is_dir() and entry.resolve().parent == SCENE_LIBRARY_RADIO.resolve():
        shutil.rmtree(entry)
        return {"deleted": [stem]}
    raise HTTPException(404, f"라이브러리를 찾을 수 없습니다: {stem}")


@app.post("/api/sessions/{uuid}/scene_from_library_radio")
async def scene_from_library_radio(uuid: str, req: dict) -> dict:
    """Geo-Radio 번들 라이브러리 엔트리를 세션 씬으로 적용 (폴더 복사)."""
    sdir = SESSIONS_ROOT / uuid
    if not sdir.exists():
        raise HTTPException(404, "session not found")
    name = req.get("name")
    if not name:
        raise HTTPException(400, "name required")
    stem = _safe_lib_name(name)
    entry = SCENE_LIBRARY_RADIO / stem
    if not entry.exists() or entry.resolve().parent != SCENE_LIBRARY_RADIO.resolve():
        raise HTTPException(404, f"library scene not found: {stem}")
    info_p = entry / "scene_info.json"
    if not info_p.exists():
        raise HTTPException(400, f"손상된 라이브러리 엔트리(scene_info.json 없음): {stem}")

    scene_dir = sdir / "scene"
    if scene_dir.exists():
        shutil.rmtree(scene_dir)
    shutil.copytree(entry, scene_dir)
    update_meta(sdir, scene_name=stem, skin_name="")
    info = json.loads((scene_dir / "scene_info.json").read_text(encoding="utf-8"))
    return {"scene_info": info, "scene_xml": str(scene_dir / "scene.xml"),
            "source": "library_radio", "name": stem}


# ---------------------------------------------------------------------------
# 지면 레이캐스팅: TX 스냅 / RX 지면 격자 (PARK_2 RT 이식)
# ---------------------------------------------------------------------------
def _session_scene_ply(uuid: str) -> Path:
    """세션 씬의 지형 PLY 경로 (build_scene 가 만든 meshes/scene_mesh.ply)."""
    meshes = SESSIONS_ROOT / uuid / "scene" / "meshes"
    cand = meshes / "scene_mesh.ply"
    if cand.exists():
        return cand
    plys = sorted(meshes.glob("*.ply")) if meshes.exists() else []
    if not plys:
        raise HTTPException(404, "scene PLY not found — 먼저 씬을 업로드/선택하세요.")
    return plys[0]


def _session_scene_plys(uuid: str) -> list[Path]:
    """세션 씬의 모든 지형 PLY 경로.

    Geo-Radio Env. Twin 은 재질별로 여러 .ply 로 쪼개져 있으므로, 지면 레이캐스팅
    (RX 격자 / TX 스냅)은 단 하나가 아니라 **전부**를 하나의 메시처럼 사용해야
    MA(재질 부여) 영역이 구멍처럼 누락되지 않는다.
    """
    meshes = SESSIONS_ROOT / uuid / "scene" / "meshes"
    plys = sorted(meshes.glob("*.ply")) if meshes.exists() else []
    if not plys:
        raise HTTPException(404, "scene PLY not found — 먼저 씬을 업로드/선택하세요.")
    return plys


@app.post("/api/sessions/{uuid}/tx_snap")
def tx_snap(uuid: str, req: dict) -> dict:
    """클릭한 (x, y) 를 지면 + offset_m 위치로 스냅한 '진짜 TX 위치' 반환.

    body: {x, y, offset_m?=2.0}
    """
    from .rt_agent_park2.ground import snap_tx_to_ground

    ply = _session_scene_plys(uuid)
    try:
        x = float(req["x"]); y = float(req["y"])
    except (KeyError, TypeError, ValueError):
        raise HTTPException(400, "x, y (float) required")
    offset = float(req.get("offset_m", 2.0))
    try:
        return snap_tx_to_ground(ply, x, y, offset_m=offset)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"tx_snap failed: {type(exc).__name__}: {exc}")


@app.post("/api/sessions/{uuid}/rx_ground_grid")
def rx_ground_grid(uuid: str, req: dict) -> dict:
    """지면 격자 RX 미리보기. body: {grid_n=20, margin=0.0, rx_height=1.5, raycasting_z?}

    returns: {positions: [[x,y,z],...], count, candidates: grid_n^2}
    """
    from .rt_agent_park2.ground import compute_ground_rx_grid

    ply = _session_scene_plys(uuid)
    grid_n = max(2, min(200, int(req.get("grid_n", 20))))   # 안전 캡 (폭주 방지)
    margin = float(req.get("margin", 0.0))
    rx_height = float(req.get("rx_height", 1.5))
    rcz = req.get("raycasting_z", None)
    rcz = float(rcz) if rcz is not None else None
    mh = req.get("max_height", None)
    mh = float(mh) if mh is not None and mh != "" else None
    # 간격(미터) 방식: spacing>0 이면 정사각 간격 격자, 아니면 grid_n 밀도 방식
    sp = req.get("spacing", None)
    sp = float(sp) if sp is not None and sp != "" else None
    # 직사각형 영역(x_start/x_stop/y_start/y_stop) 지정 시 그 안에서만 후보 생성/raycasting
    region = None
    if all(req.get(k) is not None for k in ("x_start", "x_stop", "y_start", "y_stop")):
        region = (float(req["x_start"]), float(req["x_stop"]),
                  float(req["y_start"]), float(req["y_stop"]))
    try:
        pos = compute_ground_rx_grid(ply, grid_n=grid_n, margin=margin,
                                     rx_height=rx_height, raycasting_z=rcz, max_height=mh,
                                     region=region, spacing=sp)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"rx_ground_grid failed: {type(exc).__name__}: {exc}")
    return {"positions": pos, "count": len(pos), "candidates": grid_n * grid_n}


# ---------------------------------------------------------------------------
# Scenario Generator (Mobility) — batch RT 산출물(USDA/OBJ) 기반
# ---------------------------------------------------------------------------
def _session_batch_usda_obj(uuid: str) -> tuple[Path, Path]:
    """세션의 batch RT 산출물에서 RT_scene_*.usda + Map_Mesh.obj 경로를 찾는다."""
    base = SESSIONS_ROOT / uuid / "Batch_RT_Results" / "output_USDA_and_check"
    if not base.exists():
        raise HTTPException(404, "batch RT 산출물이 없습니다. 먼저 batch RT 잡을 실행하세요.")
    usdas = sorted(base.glob("RT_scene_*.usda"))
    objs = sorted(base.glob("Map_Mesh.obj"))
    if not usdas:
        raise HTTPException(404, "RT_scene_*.usda 를 찾을 수 없습니다.")
    if not objs:
        raise HTTPException(404, "Map_Mesh.obj 를 찾을 수 없습니다.")
    return usdas[-1], objs[0]


@app.get("/api/sessions/{uuid}/scenario/data")
def scenario_data(uuid: str) -> dict:
    """2D 탑뷰 캔버스용 데이터: RX/TX 좌표 + 맵 엣지."""
    from .rt_agent_park2.scenario_builder import get_scenario_data

    usda, obj = _session_batch_usda_obj(uuid)
    try:
        return get_scenario_data(usda, obj)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"scenario_data failed: {type(exc).__name__}: {exc}")


@app.post("/api/sessions/{uuid}/scenario/generate")
def scenario_generate(uuid: str, req: dict) -> dict:
    """선택 경로 → blender_mobility.py 생성 + 웹 미리보기 데이터.

    body: {path: [rx_idx,...], tx_index=0, fps=30, frames_per_step=3}
    """
    from .rt_agent_park2.scenario_builder import generate_scenario

    usda, obj = _session_batch_usda_obj(uuid)
    path = req.get("path") or []
    if not path:
        raise HTTPException(400, "path (RX index 리스트) 가 필요합니다.")
    out_dir = SESSIONS_ROOT / uuid / "Scenario_Results"
    try:
        res = generate_scenario(
            usda, obj, out_dir,
            path_indices=[int(i) for i in path],
            tx_index=int(req.get("tx_index", 0)),
            fps=int(req.get("fps", 30)),
            frames_per_step=int(req.get("frames_per_step", 3)),
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"scenario_generate failed: {type(exc).__name__}: {exc}")
    # 다운로드용 상대경로
    rel = Path(res["script_path"]).relative_to(SESSIONS_ROOT / uuid)
    res["script_rel"] = str(rel)

    # 세션에 영속화 — 세션을 다시 열거나 새로고침해도 8.Scenario/9.Scenario Results 복원.
    state = {
        "session_uuid": uuid,
        "path": [int(i) for i in path],
        "tx_index": int(req.get("tx_index", 0)),
        "fps": int(req.get("fps", 30)),
        "frames_per_step": int(req.get("frames_per_step", 3)),
        **res,
    }
    try:
        state_path = SESSIONS_ROOT / uuid / "Scenario_Results" / "scenario_state.json"
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass  # 영속화 실패가 생성 자체를 막지 않도록 무시
    return res


@app.get("/api/sessions/{uuid}/scenario/result")
async def scenario_result(uuid: str) -> dict:
    """저장된 시나리오(8/9) 복원용. 없으면 204 대신 빈 dict({exists:false}) 반환."""
    state_path = SESSIONS_ROOT / uuid / "Scenario_Results" / "scenario_state.json"
    if not state_path.exists():
        return {"exists": False}
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state["exists"] = True
        return state
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"scenario_result load failed: {type(exc).__name__}: {exc}")


@app.get("/api/sessions/{uuid}/scenario/channel_state")
def scenario_channel_state(uuid: str, start: int = 0, count: int = 20) -> dict:
    """시나리오 경로 RX들의 채널 관측량(RSRP/PADP/Covariance) 청크 반환.

    9.Scenario Results 우측 'Channel State' 패널이 청크 단위로 병렬 로딩하며 진행바 표시.
    items[k] 는 경로의 step (start+k) 에 대응. (PNG 없이 클라 렌더용 경량 JSON)
    """
    from .rt_agent_park2.rx_inspector import find_channel_npz
    from .rt_agent_park2.scenario_channel import channel_state

    sdir = SESSIONS_ROOT / uuid
    state_path = sdir / "Scenario_Results" / "scenario_state.json"
    if not state_path.exists():
        raise HTTPException(404, "scenario state 없음 — 먼저 8.Scenario 에서 생성하세요.")
    st = json.loads(state_path.read_text(encoding="utf-8"))
    path = [int(i) for i in st.get("path", [])]
    tx_index = int(st.get("tx_index", 0))
    total = len(path)
    npz = find_channel_npz(sdir)
    if npz is None:
        raise HTTPException(404, "channel_data npz 없음 — batch RT 결과가 필요합니다.")
    start = max(0, int(start)); count = max(1, min(100, int(count)))
    sl = path[start:start + count]
    try:
        items = channel_state(npz, sl, tx_index)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"channel_state failed: {type(exc).__name__}: {exc}")
    return {"start": start, "count": len(items), "total": total, "tx_index": tx_index, "items": items}


@app.get("/api/sessions/{uuid}/rx_inspect")
def rx_inspect(uuid: str, rx: int, tx: int = 0) -> dict:
    """7.RT Results 인터랙티브: (tx, rx) 의 PADP/PDP/Covariance PNG + RSRP.

    batch RT 산출물 channel_data_*.npz 가 필요하다.
    """
    from .rt_agent_park2.rx_inspector import find_channel_npz, inspect_rx

    sdir = SESSIONS_ROOT / uuid
    npz = find_channel_npz(sdir)
    if npz is None:
        raise HTTPException(404, "channel_data 가 없습니다. batch RT 잡을 먼저 실행하세요.")
    out_dir = sdir / "RX_Inspect_Results"
    try:
        res = inspect_rx(npz, rx_idx=int(rx), tx_index=int(tx), out_dir=out_dir)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"rx_inspect failed: {type(exc).__name__}: {exc}")
    base = SESSIONS_ROOT / uuid
    for k in ("padp_png", "pdp_png", "cov_png"):
        res[k + "_rel"] = str(Path(res[k]).relative_to(base))
    return res


# ---------------------------------------------------------------------------
# Job submit / status / cancel
# ---------------------------------------------------------------------------
@app.post("/api/sessions/{uuid}/job")
async def post_job(uuid: str, req: JobSubmitRequest) -> dict:
    sdir = SESSIONS_ROOT / uuid
    if not sdir.exists():
        raise HTTPException(404, "session not found")
    if uuid != req.session_uuid:
        raise HTTPException(400, "session_uuid mismatch")

    # RX 필요 metric인데 RX 없으면 거부
    if requires_rx(req.metrics.metrics):
        has_rx = bool((req.rx_grid and any([req.rx_grid.x_num, req.rx_grid.x_coords]))
                      or (req.rx_clicks and req.rx_clicks.positions))
        if not has_rx:
            raise HTTPException(400, "Selected metrics require RX, but no RX provided.")

    payload = req.model_dump(mode="python")
    # 실행 config 스냅샷 저장 (워커가 실행 시점에 로드 → queued 동안 수정 반영)
    save_session_config(sdir, payload)

    # 세션 메타 갱신: 실제 실행 안테나/metric/engine 반영 (1.Sessions 표시 버그 수정)
    a = req.antenna.simple
    update_meta(
        sdir,
        bs_rows=a.bs_rows, bs_cols=a.bs_cols, ue_rows=a.ue_rows, ue_cols=a.ue_cols,
        metrics=list(req.metrics.metrics), engine=req.rt.engine,
    )

    store = make_store()
    meta = load_meta(sdir)
    # 큐에 등록 후 runner 제출 (payload 는 최소 — 워커가 디스크 config 사용)
    job = runner.submit(kind="pipeline", session_uuid=uuid,
                        payload={"session_uuid": uuid})
    store.enqueue(uuid, job.job_id,
                  user_id=(meta.user_id if meta else ""),
                  user_name=(meta.user_name if meta else ""),
                  engine=req.rt.engine)
    return {**job.to_dict(), "queued": True, "session_uuid": uuid}


@app.get("/api/queue")
async def get_queue() -> dict:
    """1초 폴링용 큐 대시보드: queueing / processing(라이브 진행률) / done."""
    store = make_store()
    # processing 세션의 라이브 진행률을 runner 메모리에서 읽어 덮어쓴다.
    live: dict[str, float] = {}
    for j in runner.list_jobs():
        if j.get("state") == "running" and j.get("session_uuid"):
            live[j["session_uuid"]] = float(j.get("progress") or 0.0)
    return store.dashboard(live)


@app.post("/api/sessions/{uuid}/cancel")
async def cancel_session(uuid: str) -> dict:
    """queued 취소 또는 processing 협력적 중단."""
    sdir = SESSIONS_ROOT / uuid
    meta = load_meta(sdir)
    if not meta:
        raise HTTPException(404, "session not found")
    store = make_store()
    if meta.job_id:
        runner.cancel(meta.job_id)   # processing 이면 다음 emit 경계에서 정지
    if meta.status == "queued":
        store.mark_cancelled(uuid)   # 워커가 pop 시 skip (runner.cancel 로 state=cancelled)
    elif meta.status == "processing":
        update_meta(sdir, last_error="중단 요청됨 — 현재 단계 종료 후 정지")
    return {"ok": True, "status": (load_meta(sdir) or meta).status}


@app.get("/api/sessions/{uuid}/config")
async def get_session_config(uuid: str) -> dict:
    """저장된 실행 config 스냅샷 (2~5단계 복원용). 없으면 {exists:false}."""
    cfg = load_session_config(SESSIONS_ROOT / uuid)
    if cfg is None:
        return {"exists": False}
    return {"exists": True, "config": cfg}


@app.put("/api/sessions/{uuid}/config")
async def put_session_config(uuid: str, req: JobSubmitRequest) -> dict:
    """queued 세션의 config 수정 (순서 불변). 메타 안테나/metric 도 갱신."""
    sdir = SESSIONS_ROOT / uuid
    if not sdir.exists():
        raise HTTPException(404, "session not found")
    payload = req.model_dump(mode="python")
    save_session_config(sdir, payload)
    a = req.antenna.simple
    update_meta(sdir, bs_rows=a.bs_rows, bs_cols=a.bs_cols,
                ue_rows=a.ue_rows, ue_cols=a.ue_cols,
                metrics=list(req.metrics.metrics), engine=req.rt.engine)
    return {"ok": True}


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str) -> dict:
    job = runner.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    return job.to_dict()


@app.get("/api/jobs")
async def list_jobs() -> dict:
    return {"jobs": runner.list_jobs()}


@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job(job_id: str) -> dict:
    ok = runner.cancel(job_id)
    return {"cancelled": ok}


# ---------------------------------------------------------------------------
# WebSocket progress
# ---------------------------------------------------------------------------
@app.websocket("/ws/jobs/{job_id}")
async def ws_job(websocket: WebSocket, job_id: str) -> None:
    await broker.connect(job_id, websocket)
    try:
        while True:
            # 클라이언트로부터 ping 같은 메시지를 받아 keep-alive 유지
            msg = await websocket.receive_text()
            if msg == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        await broker.disconnect(job_id, websocket)
    except Exception:
        await broker.disconnect(job_id, websocket)


# ---------------------------------------------------------------------------
# Results files
# ---------------------------------------------------------------------------
@app.get("/api/sessions/{uuid}/files")
async def list_session_files(uuid: str) -> dict:
    sdir = SESSIONS_ROOT / uuid
    if not sdir.exists():
        raise HTTPException(404, "session not found")
    nodes: list[dict] = []
    for path in sorted(sdir.rglob("*")):
        if path.is_dir():
            continue
        rel = path.relative_to(sdir)
        nodes.append({
            "path": str(rel).replace("\\", "/"),
            "size": path.stat().st_size,
            "kind": path.suffix.lstrip(".").lower(),
        })
    return {"files": nodes}


@app.get("/api/sessions/{uuid}/files/{path:path}")
async def get_session_file(uuid: str, path: str, download: bool = False) -> FileResponse:
    sdir = SESSIONS_ROOT / uuid
    target = (sdir / path).resolve()
    if not str(target).startswith(str(sdir.resolve())):
        raise HTTPException(400, "Path traversal blocked")
    if not target.exists() or target.is_dir():
        raise HTTPException(404, "file not found")
    # 결과 파일은 같은 파일명으로 재생성될 수 있으므로(예: RT 재실행 → rsrp_heatmap_TX0.png 갱신)
    # no-cache 로 매번 ETag/Last-Modified 재검증한다. 변경 없으면 304(본문 없음, 빠름),
    # 변경되면 새 내용을 받는다. (max-age 캐시는 옛 이미지가 그대로 보이는 stale 문제 유발)
    headers = {"Cache-Control": "no-cache"}
    if download:
        # 첨부파일로 명시 → 브라우저가 탭에서 열려다 실패하지 않고 첫 클릭에 바로 다운로드.
        import urllib.parse as _ul
        fname = Path(path).name
        headers["Content-Disposition"] = f"attachment; filename*=UTF-8''{_ul.quote(fname)}"
    return FileResponse(target, headers=headers)


@app.post("/api/sessions/{uuid}/zip")
async def zip_session(uuid: str) -> StreamingResponse:
    sdir = SESSIONS_ROOT / uuid
    if not sdir.exists():
        raise HTTPException(404, "session not found")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sdir.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(sdir))
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=session_{uuid}.zip"},
    )
