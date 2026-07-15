"""scenario_builder.py — Mobility Scenario 생성 (m_mobility_scenario_builder.py 이식)

원본: /home/dclserver78/twin_minji_verPARK/260607_temp_folder/m_mobility_scenario_builder.py
  (원본은 matplotlib 인터랙티브 창에서 드래그로 RX 경로 선택 → blender_mobility.py 생성)

웹 통합:
  - 마우스 드래그 RX 선택은 프런트(2D 탑뷰 캔버스)로 이전.
  - 백엔드는 비대화형 부분만 담당:
      parse_usda()            : RT_scene_*.usda 에서 TX/RX 위치 + Ray 폴리라인 추출 (pxr 미사용, 텍스트 파서)
      load_obj_edges()        : Map_Mesh.obj 탑뷰 엣지 (서브샘플)
      get_scenario_data()     : 프런트 캔버스용 데이터(RX/TX/edges) 반환
      generate_scenario()     : 선택 경로 → blender_mobility.py 생성 + 웹 미리보기용 scenario_data 반환

USDA 구조 (m7_export_agent._build_usda_multi_tx):
  /RayTracingScene/TX{t}_Main      def Xform → xformOp:translate
  /RayTracingScene/RX{i}_Marker    def Xform → xformOp:translate
  /RayTracingScene/TX{t}_Rays/R{rx}_{path}  def BasisCurves → point3f[] points
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# USDA 텍스트 파서 (pxr 미사용)
# ---------------------------------------------------------------------------
_RE_MARKER = re.compile(
    r'def\s+Xform\s+"(TX\d+_Main|RX\d+_Marker)"\s*\{\s*double3\s+xformOp:translate\s*=\s*\(([^)]+)\)'
)
_RE_TX_RAYS = re.compile(r'def\s+Xform\s+"TX(\d+)_Rays"')
_RE_CURVE = re.compile(
    r'def\s+BasisCurves\s+"R(\d+)_(\d+)"\s*\{(.*?)\}',
    re.DOTALL,
)
_RE_POINTS_ATTR = re.compile(r'point3f\[\]\s+points\s*=\s*\[([^\]]*)\]')
_RE_WIDTHS_ATTR = re.compile(r'float\[\]\s+widths\s*=\s*\[([^\]]*)\]')
_RE_MATBIND = re.compile(r'material:binding\s*=\s*</[^>]*?/(Mat_Ray_\w+)>')
_RE_POINT = re.compile(r'\(([^)]+)\)')


def _coords(s: str) -> tuple[float, float, float]:
    parts = [float(p) for p in s.split(",")]
    return (parts[0], parts[1], parts[2])


def _extract_markers(text: str):
    """텍스트에서 TX_Main/RX_Marker 위치만 추출 (idx 오름차순)."""
    rx_data: list[tuple] = []
    tx_data: list[tuple] = []
    for m in _RE_MARKER.finditer(text):
        name = m.group(1)
        x, y, z = _coords(m.group(2))
        if name.startswith("TX") and name.endswith("_Main"):
            idx = int(name[2:name.index("_Main")])
            tx_data.append((idx, name, x, y, z))
        elif name.startswith("RX") and name.endswith("_Marker"):
            idx = int(name[2:name.index("_Marker")])
            rx_data.append((idx, name, x, y, z))
    rx_data.sort(key=lambda d: d[0])
    tx_data.sort(key=lambda d: d[0])
    return rx_data, tx_data


def _read_markers_head(usda_path: str | Path) -> str:
    """마커(TX_Main/RX_Marker)는 광선(TX_Rays) 블록 앞에 위치.
    광선 블록 시작 전까지만 읽어 대용량 커브(수 GB) 파싱을 건너뛴다(경량 로드)."""
    lines: list[str] = []
    with open(usda_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if '_Rays"' in line and "def " in line:
                break
            lines.append(line)
    return "".join(lines)


def parse_usda(usda_path: str | Path, markers_only: bool = False) -> dict:
    """USDA → {rx_data, tx_data, ray_map}.

    rx_data : [(idx, name, x, y, z), ...]  (idx 오름차순)
    tx_data : [(idx, name, x, y, z), ...]
    ray_map : {tx_idx: {rx_idx: [ [(x,y,z),...], ... ]}}

    markers_only=True → 광선 커브 파싱 생략, 마커 위치만 반환(캔버스 경량 로드).
    """
    if markers_only:
        rx_data, tx_data = _extract_markers(_read_markers_head(usda_path))
        if not rx_data and not tx_data:  # 순서가 예상과 다르면 전체 재시도(폴백)
            rx_data, tx_data = _extract_markers(
                Path(usda_path).read_text(encoding="utf-8", errors="ignore"))
        return {"rx_data": rx_data, "tx_data": tx_data, "ray_map": {}}

    text = Path(usda_path).read_text(encoding="utf-8", errors="ignore")
    rx_data, tx_data = _extract_markers(text)

    # TX_Rays 블록별로 텍스트를 분할하여 BasisCurves 추출
    ray_map: dict[int, dict[int, list]] = {}
    tx_ray_matches = list(_RE_TX_RAYS.finditer(text))
    for i, mm in enumerate(tx_ray_matches):
        tx_idx = int(mm.group(1))
        start = mm.end()
        end = tx_ray_matches[i + 1].start() if i + 1 < len(tx_ray_matches) else len(text)
        block = text[start:end]
        per_rx: dict[int, list] = {}
        for cm in _RE_CURVE.finditer(block):
            rx_idx = int(cm.group(1))
            body = cm.group(3)
            pm = _RE_POINTS_ATTR.search(body)
            if not pm:
                continue
            pts = [_coords(p) for p in _RE_POINT.findall(pm.group(1))]
            if not pts:
                continue
            wm = _RE_WIDTHS_ATTR.search(body)
            width = 0.3
            if wm:
                wvals = [float(v) for v in wm.group(1).split(",") if v.strip()]
                if wvals:
                    width = wvals[0]
            mb = _RE_MATBIND.search(body)
            los = (mb.group(1) == "Mat_Ray_LOS") if mb else (len(pts) == 2)
            per_rx.setdefault(rx_idx, []).append({"pts": pts, "width": width, "los": los})
        ray_map[tx_idx] = per_rx

    return {"rx_data": rx_data, "tx_data": tx_data, "ray_map": ray_map}


# ---------------------------------------------------------------------------
# OBJ 탑뷰 엣지 (원본 load_obj_edges + 서브샘플)
# ---------------------------------------------------------------------------
def load_obj_edges(obj_file: str | Path, max_edges: int = 60000) -> list:
    """Map_Mesh.obj → 탑뷰 (x,y) 엣지 리스트 [[[x1,y1],[x2,y2]], ...] (서브샘플)."""
    verts: list[tuple] = []
    edges: list = []
    seen: set = set()
    with open(obj_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                p = line.split()
                verts.append((float(p[1]), float(p[2])))
            elif line.startswith("f "):
                idx = [int(p.split("/")[0]) - 1 for p in line.split()[1:]]
                for i in range(len(idx)):
                    a, b = idx[i], idx[(i + 1) % len(idx)]
                    if 0 <= a < len(verts) and 0 <= b < len(verts):
                        key = (min(a, b), max(a, b))
                        if key in seen:
                            continue
                        seen.add(key)
                        edges.append([[verts[a][0], verts[a][1]], [verts[b][0], verts[b][1]]])
    if len(edges) > max_edges:
        step = len(edges) // max_edges + 1
        edges = edges[::step]
    return edges


# ---------------------------------------------------------------------------
# 프런트 캔버스용 데이터
# ---------------------------------------------------------------------------
# 대용량 USDA(수 GB) 재파싱 방지용 캐시. key = (usda sig, obj sig, max_edges)
# 파일 경로+수정시각(mtime)+크기가 같으면 파싱 결과를 재사용 → 8.Scenario 재방문 시 즉시 로드.
_SCENARIO_DATA_CACHE: dict = {}
_SCENARIO_DATA_CACHE_MAX = 4


def _file_sig(p: str | Path):
    try:
        st = Path(p).stat()
        return (str(p), st.st_mtime_ns, st.st_size)
    except OSError:
        return (str(p), 0, 0)


def get_scenario_data(usda_path: str | Path, obj_path: str | Path,
                      max_edges: int = 60000) -> dict:
    # 캐시 조회 (동일 파일이면 재파싱 생략)
    key = (_file_sig(usda_path), _file_sig(obj_path), int(max_edges))
    cached = _SCENARIO_DATA_CACHE.get(key)
    if cached is not None:
        return cached

    parsed = parse_usda(usda_path, markers_only=True)  # 캔버스는 위치만 → 광선 파싱 생략(경량)
    edges = load_obj_edges(obj_path, max_edges=max_edges) if Path(obj_path).exists() else []
    result = {
        "rx": [{"idx": i, "x": x, "y": y, "z": z} for (i, _n, x, y, z) in parsed["rx_data"]],
        "tx": [{"idx": i, "x": x, "y": y, "z": z} for (i, _n, x, y, z) in parsed["tx_data"]],
        "edges": edges,
        "num_rx": len(parsed["rx_data"]),
        "num_tx": len(parsed["tx_data"]),
    }

    # 캐시 저장 (크기 제한: 오래된 것부터 제거)
    if len(_SCENARIO_DATA_CACHE) >= _SCENARIO_DATA_CACHE_MAX:
        _SCENARIO_DATA_CACHE.pop(next(iter(_SCENARIO_DATA_CACHE)))
    _SCENARIO_DATA_CACHE[key] = result
    return result


# ---------------------------------------------------------------------------
# scenario_data 추출 (선택 TX + 선택 RX 경로)
# ---------------------------------------------------------------------------
def _extract_scenario(parsed: dict, sel_tx_idx: int, path_indices: list[int]):
    """반환: (tx_pos, scenario_data)
    scenario_data : [(name, (x,y,z), [ [(x,y,z),...], ... ]), ...]  (경로 순서)
    """
    rx_by_idx = {i: (i, n, x, y, z) for (i, n, x, y, z) in parsed["rx_data"]}
    tx_by_idx = {i: (i, n, x, y, z) for (i, n, x, y, z) in parsed["tx_data"]}
    ray_map = parsed["ray_map"].get(sel_tx_idx, {})

    tx_entry = tx_by_idx.get(sel_tx_idx)
    tx_pos = (round(tx_entry[2], 4), round(tx_entry[3], 4), round(tx_entry[4], 4)) if tx_entry else (0.0, 0.0, 0.0)

    scenario_data = []   # 블렌더용: (name, pos, [point-list, ...])
    preview_steps = []   # 웹 미리보기용: {name, pos, rays:[{pts,los,width}]}
    for ri in path_indices:
        entry = rx_by_idx.get(ri)
        if entry is None:
            continue
        name = entry[1]
        pos = (round(entry[2], 4), round(entry[3], 4), round(entry[4], 4))
        ray_objs = ray_map.get(ri, [])
        # 블렌더 SCENARIO_DATA: 점 리스트만
        blender_rays = []
        web_rays = []
        for ro in ray_objs:
            pts = ro["pts"] if isinstance(ro, dict) else ro
            rpts = [(round(p[0], 4), round(p[1], 4), round(p[2], 4)) for p in pts]
            blender_rays.append(rpts)
            if isinstance(ro, dict):
                web_rays.append({"pts": [list(p) for p in rpts],
                                 "los": bool(ro.get("los", len(rpts) == 2)),
                                 "width": float(ro.get("width", 0.3))})
            else:
                web_rays.append({"pts": [list(p) for p in rpts],
                                 "los": len(rpts) == 2, "width": 0.3})
        scenario_data.append((name, pos, blender_rays))
        preview_steps.append({"name": name, "pos": list(pos), "rays": web_rays})
    return tx_pos, scenario_data, preview_steps


# ---------------------------------------------------------------------------
# 생성: blender_mobility.py + 웹 미리보기 데이터
# ---------------------------------------------------------------------------
def generate_scenario(
    usda_path: str | Path,
    obj_path: str | Path,
    out_dir: str | Path,
    path_indices: list[int],
    tx_index: int = 0,
    fps: int = 30,
    frames_per_step: int = 3,
) -> dict:
    """선택 경로로 blender_mobility.py 생성 + 웹 미리보기용 데이터 반환."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    parsed = parse_usda(usda_path)

    if not path_indices:
        raise ValueError("선택된 RX 경로가 비어 있습니다.")

    sel_tx_idx = tx_index
    tx_name = next((n for (i, n, *_r) in parsed["tx_data"] if i == sel_tx_idx), f"TX{sel_tx_idx}_Main")
    scenario_path = [f"RX{ri}_Marker" for ri in path_indices]
    tx_pos, scenario_data, preview_steps = _extract_scenario(parsed, sel_tx_idx, path_indices)

    obj_basename = Path(obj_path).name
    script_path = out_dir / "blender_mobility.py"
    _write_blender_script(
        script_path, scenario_path, scenario_data, tx_pos, tx_name,
        obj_basename, int(fps), int(frames_per_step),
    )

    # Unreal용 자체 완결형 스크립트 (파일 하나 딸깍 → TX/RX/광선 + Sequencer 재생)
    unreal_script_path = out_dir / "unreal_scenario.py"
    _write_unreal_script(
        unreal_script_path, scenario_data, tx_pos, tx_name,
        int(fps), int(frames_per_step),
    )

    # 웹 미리보기용(JSON 직렬화) — 키프레임 재생 + LoS색/굵기
    preview = {
        "tx_name": tx_name,
        "tx_pos": list(tx_pos),
        "fps": int(fps),
        "frames_per_step": int(frames_per_step),
        "total_frames": len(scenario_path) * int(frames_per_step),
        "steps": preview_steps,
    }
    n_rays = sum(len(r) for _n, _p, r in scenario_data)
    return {
        "script_path": str(script_path),
        "unreal_script_path": str(unreal_script_path),
        "num_rx": len(scenario_data),
        "num_rays": n_rays,
        "tx_name": tx_name,
        "preview": preview,
    }


def _write_blender_script(out_path: Path, scenario_path, scenario_data, tx_pos,
                          tx_name, obj_basename, fps, frames_per_step) -> None:
    """원본 create_blender_script 의 자체 완결형 bpy 스크립트를 그대로 생성."""
    header = [
        '"""',
        'blender_mobility.py  (경량 / 자체 완결형)',
        '=========================================',
        '거대한 USDA 임포트가 필요 없습니다. 이 스크립트가 선택한 RX/TX/Ray 만 생성합니다.',
        '',
        '사용법:',
        f'  1. (선택) 배경 지도를 보려면 File -> Import -> Wavefront OBJ -> {obj_basename}',
        '  2. Scripting 탭 -> 이 스크립트 Run',
        '  3. Layout 탭 -> Space bar 재생',
        '',
        f'선택 TX  : {tx_name}',
        f'시나리오 : {" -> ".join(scenario_path)}',
        '"""',
        '',
        'import bpy',
        'import bmesh',
        '',
        '# ── 설정 ──────────────────────────────────────────────────',
        f'FRAMES_PER_STEP = {frames_per_step}',
        'START_FRAME     = 1',
        f'FPS             = {fps}',
        'RX_RADIUS       = 1.2',
        'TX_RADIUS       = 2.0',
        'RAY_WIDTH       = 0.3',
        f'TX_NAME         = {tx_name!r}',
        f'TX_POS          = {tx_pos!r}',
        '',
        '# ── 데이터: (이름, 위치, [ray 폴리라인, ...]) ─────────────',
        f'SCENARIO_DATA = {scenario_data!r}',
        '',
    ]
    out_path.write_text("\n".join(header) + _BLENDER_BODY, encoding="utf-8")


# 원본 create_blender_script 의 body 부분 (verbatim)
_BLENDER_BODY = r'''
COLL_NAME = "RT_Mobility"

old = bpy.data.collections.get(COLL_NAME)
if old:
    for o in list(old.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    bpy.data.collections.remove(old)
coll = bpy.data.collections.new(COLL_NAME)
bpy.context.scene.collection.children.link(coll)


def make_mat(name, r, g, b, emission=2.0):
    if name in bpy.data.materials:
        bpy.data.materials.remove(bpy.data.materials[name])
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value     = (r, g, b, 1.0)
    bsdf.inputs["Emission Color"].default_value = (r, g, b, 1.0)
    bsdf.inputs["Emission Strength"].default_value = emission
    bsdf.inputs["Roughness"].default_value = 0.4
    out = nodes.new("ShaderNodeOutputMaterial")
    out.location = (300, 0)
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


mat_tx  = make_mat("RT_TX",  1.0, 0.1, 0.1, emission=3.0)
mat_rx  = make_mat("RT_RX",  0.1, 1.0, 0.1, emission=3.0)
mat_ray = make_mat("RT_Ray", 1.0, 0.75, 0.0, emission=5.0)


def make_sphere_mesh(name, radius):
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=16, v_segments=10, radius=radius)
    bm.to_mesh(me)
    bm.free()
    return me


rx_mesh = make_sphere_mesh("RX_sphere", RX_RADIUS)
rx_mesh.materials.append(mat_rx)
tx_mesh = make_sphere_mesh("TX_sphere", TX_RADIUS)
tx_mesh.materials.append(mat_tx)


def add_marker(name, mesh, pos):
    o = bpy.data.objects.new(name, mesh)
    o.location = pos
    coll.objects.link(o)
    return o


def add_ray(name, points):
    cu = bpy.data.curves.new(name, 'CURVE')
    cu.dimensions = '3D'
    cu.bevel_depth = RAY_WIDTH
    cu.materials.append(mat_ray)
    sp = cu.splines.new('POLY')
    sp.points.add(len(points) - 1)
    for i, (x, y, z) in enumerate(points):
        sp.points[i].co = (x, y, z, 1.0)
    o = bpy.data.objects.new(name, cu)
    coll.objects.link(o)
    return o


print("=" * 60)
print("오브젝트 생성 중...")
print("=" * 60)

tx_obj = add_marker(TX_NAME, tx_mesh, TX_POS)

groups = []
for idx, (name, pos, rays) in enumerate(SCENARIO_DATA):
    rxo = add_marker(f"{name}_{idx}", rx_mesh, pos)
    ray_objs = [add_ray(f"{name}_ray{idx}_{j}", pts) for j, pts in enumerate(rays)]
    groups.append((rxo, ray_objs))

print(f"  TX 1개, RX {len(groups)}개, Ray {sum(len(r) for _, r in groups)}개 생성")


total_frames = len(groups) * FRAMES_PER_STEP
bpy.context.scene.frame_start = START_FRAME
bpy.context.scene.frame_end   = START_FRAME + total_frames - 1
bpy.context.scene.render.fps  = FPS
print(f"  프레임 범위: {bpy.context.scene.frame_start} ~ {bpy.context.scene.frame_end}")


def set_vis(obj, frame, visible):
    obj.hide_viewport = not visible
    obj.hide_render   = not visible
    obj.keyframe_insert(data_path="hide_viewport", frame=frame)
    obj.keyframe_insert(data_path="hide_render",   frame=frame)


def set_group_vis(g, frame, visible):
    rxo, ray_objs = g
    set_vis(rxo, frame, visible)
    for r in ray_objs:
        set_vis(r, frame, visible)


for step, g in enumerate(groups):
    start = START_FRAME + step * FRAMES_PER_STEP
    nxt   = start + FRAMES_PER_STEP
    for gg in groups:
        set_group_vis(gg, start, gg is g)
    if nxt <= bpy.context.scene.frame_end:
        set_group_vis(g, nxt, False)

last = bpy.context.scene.frame_end
for g in groups:
    set_group_vis(g, last, False)

bpy.context.scene.frame_set(START_FRAME)


def force_constant_all():
    from bpy_extras import anim_utils
    for obj in bpy.data.objects:
        if not obj.animation_data or not obj.animation_data.action:
            continue
        action = obj.animation_data.action
        fcurves = None
        try:
            slot = obj.animation_data.action_slot
            channelbag = anim_utils.action_get_channelbag_for_slot(action, slot)
            fcurves = channelbag.fcurves
        except Exception:
            if hasattr(action, "fcurves"):
                fcurves = action.fcurves
        if fcurves is None:
            continue
        for fc in fcurves:
            if "hide" in fc.data_path:
                for kp in fc.keyframe_points:
                    kp.interpolation = "CONSTANT"


try:
    force_constant_all()
    print("  CONSTANT 보간 적용 완료")
except Exception as e:
    print(f"  CONSTANT 보간 적용 실패: {e}")

print("=" * 60)
print("Mobility Scenario 생성 완료! Space bar 로 재생하세요.")
print("=" * 60)
'''


# ---------------------------------------------------------------------------
# Unreal용 자체 완결형 스크립트 생성 (unreal_scenario.py)
#   blender_mobility.py 와 대칭: SCENARIO_DATA 를 파일에 구워 넣고,
#   Unreal 에디터에서 파일 하나만 Execute Python Script 하면
#   TX/RX/광선 스폰 + Sequencer 키프레임 재생(SEQ_RayPlayback)까지 완료.
# ---------------------------------------------------------------------------
def _write_unreal_script(out_path: Path, scenario_data, tx_pos, tx_name,
                         fps: int, frames_per_step: int) -> None:
    header = [
        '"""',
        'unreal_scenario.py  (자체 완결형 / Unreal Engine)',
        '=================================================',
        'Unreal 에디터에서  Tools → Execute Python Script...  로 이 파일을 실행하면',
        'TX/RX/광선을 생성하고 Sequencer 키프레임 재생(SEQ_RayPlayback)까지 한 번에 만든다.',
        '거대한 임포트 불필요 — 이 파일 하나로 딸깍.',
        '',
        '요구 플러그인 : Python Editor Script Plugin + Sequencer Scripting',
        '사전 준비    : 도시 메시(glb 등)는 별도로 임포트해 배치해 두세요.',
        '',
        '정합이 안 맞으면 아래 FLIP_Y / FLIP_X / SCALE 를 조정 후 재실행.',
        '재생이 반대로(전부 보임/전부 숨김) 보이면 VIS_KEY_VISIBLE 을 False 로.',
        '먼저 MAX_STEPS=5 로 소수만 띄워 정합을 맞춘 뒤 0(전체)으로.',
        '',
        f'선택 TX : {tx_name}',
        '"""',
        '',
        'import unreal',
        '',
        '# ── 설정 ──────────────────────────────────────────────',
        f'FPS             = {int(fps)}',
        f'FRAMES_PER_STEP = {int(frames_per_step)}',
        f'TX_NAME         = {tx_name!r}',
        f'TX_POS          = {tuple(tx_pos)!r}',
        'SCALE           = 100.0      # m → cm (Unreal 단위)',
        'FLIP_X          = False',
        'FLIP_Y          = True       # glb를 glTF로 임포트한 경우 보통 True',
        'FLIP_Z          = False',
        'RX_RADIUS_M     = 1.2',
        'TX_RADIUS_M     = 2.0',
        'RAY_RADIUS_M    = 0.3',
        'EMISSIVE        = 4.0',
        'MAX_STEPS       = 0          # 0=전체, N=앞 N스텝만(테스트)',
        'MODE            = "sequence" # "static"(전부표시) 또는 "sequence"(재생)',
        'VIS_KEY_VISIBLE = True',
        'ASSET_DIR       = "/Game/RT_Rays"',
        'FOLDER          = "RT_Rays"',
        'SEQ_ASSET       = "/Game/RT_Rays/SEQ_RayPlayback"',
        '',
        '# ── 데이터: (RX이름, RX위치, [광선 폴리라인, ...]) ─────',
        f'SCENARIO_DATA = {scenario_data!r}',
        '',
    ]
    out_path.write_text("\n".join(header) + _UNREAL_BODY, encoding="utf-8")


# 자체 완결형 body (verbatim). 위 헤더가 정의한 설정/데이터를 사용한다.
_UNREAL_BODY = r'''
PARENT_MAT = None


def _log(m):
    unreal.log("[unreal_scenario] " + str(m))


def to_vec(x_m, y_m, z_m):
    x = -x_m if FLIP_X else x_m
    y = -y_m if FLIP_Y else y_m
    z = -z_m if FLIP_Z else z_m
    return unreal.Vector(x * SCALE, y * SCALE, z * SCALE)


def eas():
    return unreal.get_editor_subsystem(unreal.EditorActorSubsystem)


def get_parent_material():
    path = ASSET_DIR + "/M_RTRay"
    if unreal.EditorAssetLibrary.does_asset_exist(path):
        return unreal.load_asset(path)
    at = unreal.AssetToolsHelpers.get_asset_tools()
    mat = at.create_asset("M_RTRay", ASSET_DIR, unreal.Material, unreal.MaterialFactoryNew())
    mel = unreal.MaterialEditingLibrary
    p = mel.create_material_expression(mat, unreal.MaterialExpressionVectorParameter, -400, 0)
    p.set_editor_property("parameter_name", "Color")
    p.set_editor_property("default_value", unreal.LinearColor(1, 1, 1, 1))
    mel.connect_material_property(p, "", unreal.MaterialProperty.MP_EMISSIVE_COLOR)
    mel.recompile_material(mat)
    return mat


def make_mic(name, rgb):
    path = ASSET_DIR + "/" + name
    if unreal.EditorAssetLibrary.does_asset_exist(path):
        return unreal.load_asset(path)
    at = unreal.AssetToolsHelpers.get_asset_tools()
    mic = at.create_asset(name, ASSET_DIR, unreal.MaterialInstanceConstant,
                          unreal.MaterialInstanceConstantFactoryNew())
    mel = unreal.MaterialEditingLibrary
    mel.set_material_instance_parent(mic, PARENT_MAT)
    mel.set_material_instance_vector_parameter_value(
        mic, "Color", unreal.LinearColor(rgb[0], rgb[1], rgb[2], 1.0))
    return mic


def clear_previous():
    s = eas()
    n = 0
    for a in s.get_all_level_actors():
        try:
            if str(a.get_folder_path()).startswith(FOLDER):
                s.destroy_actor(a)
                n += 1
        except Exception:
            pass
    _log("이전 %s 액터 %d개 제거" % (FOLDER, n))


def _set_mat(actor, mic):
    try:
        c = actor.get_component_by_class(unreal.StaticMeshComponent)
        if c:
            c.set_material(0, mic)
            c.set_editor_property("cast_shadow", False)
    except Exception:
        pass


def spawn_marker(mesh, loc, radius_m, mic, label, folder):
    a = eas().spawn_actor_from_object(mesh, loc, unreal.Rotator(0, 0, 0))
    s = (radius_m * SCALE) / 50.0
    a.set_actor_scale3d(unreal.Vector(s, s, s))
    _set_mat(a, mic)
    a.set_actor_label(label)
    a.set_folder_path(folder)
    return a


def spawn_segment(cyl, p0, p1, radius_m, mic, label, folder):
    v0 = to_vec(p0[0], p0[1], p0[2])
    v1 = to_vec(p1[0], p1[1], p1[2])
    dx, dy, dz = v1.x - v0.x, v1.y - v0.y, v1.z - v0.z
    length = (dx * dx + dy * dy + dz * dz) ** 0.5
    if length < 1.0:
        return None
    mid = unreal.Vector((v0.x + v1.x) / 2, (v0.y + v1.y) / 2, (v0.z + v1.z) / 2)
    rot = unreal.MathLibrary.make_rot_from_z(unreal.Vector(dx, dy, dz))
    a = eas().spawn_actor_from_object(cyl, mid, rot)
    sr = (radius_m * SCALE) / 50.0
    a.set_actor_scale3d(unreal.Vector(sr, sr, length / 100.0))
    _set_mat(a, mic)
    a.set_actor_label(label)
    a.set_folder_path(folder)
    return a


def build_sequence(groups):
    if unreal.EditorAssetLibrary.does_asset_exist(SEQ_ASSET):
        unreal.EditorAssetLibrary.delete_asset(SEQ_ASSET)
    at = unreal.AssetToolsHelpers.get_asset_tools()
    seq = at.create_asset("SEQ_RayPlayback", ASSET_DIR, unreal.LevelSequence,
                          unreal.LevelSequenceFactoryNew())
    seq.set_display_rate(unreal.FrameRate(FPS, 1))
    total = FRAMES_PER_STEP * len(groups) + 1
    seq.set_playback_start(0)
    seq.set_playback_end(total)
    for si, actors in enumerate(groups):
        on0 = si * FRAMES_PER_STEP
        on1 = on0 + FRAMES_PER_STEP
        for a in actors:
            binding = seq.add_possessable(a)
            track = binding.add_track(unreal.MovieSceneVisibilityTrack)
            section = track.add_section()
            section.set_range(0, total)
            ch = section.get_all_channels()[0]
            vis = VIS_KEY_VISIBLE
            hid = not VIS_KEY_VISIBLE
            ch.add_key(unreal.FrameNumber(0), hid)
            ch.add_key(unreal.FrameNumber(int(on0)), vis)
            ch.add_key(unreal.FrameNumber(int(on1)), hid)
    unreal.EditorAssetLibrary.save_asset(SEQ_ASSET)
    _log("Level Sequence 생성: %s (%d프레임, %dfps)" % (SEQ_ASSET, total, FPS))


def main():
    global PARENT_MAT
    try:
        steps = SCENARIO_DATA[:MAX_STEPS] if MAX_STEPS > 0 else SCENARIO_DATA
        _log("스텝 %d개, MODE=%s" % (len(steps), MODE))
        if not unreal.EditorAssetLibrary.does_directory_exist(ASSET_DIR):
            unreal.EditorAssetLibrary.make_directory(ASSET_DIR)
        PARENT_MAT = get_parent_material()
        mic_tx = make_mic("MI_TX", (EMISSIVE * 2.0, EMISSIVE * 0.05, EMISSIVE * 0.05))
        mic_rx = make_mic("MI_RX", (EMISSIVE * 0.05, EMISSIVE * 2.0, EMISSIVE * 0.05))
        mic_ray = make_mic("MI_Ray", (EMISSIVE * 2.0, EMISSIVE * 1.5, 0.0))
        sphere = unreal.load_asset("/Engine/BasicShapes/Sphere.Sphere")
        cyl = unreal.load_asset("/Engine/BasicShapes/Cylinder.Cylinder")
        clear_previous()

        spawn_marker(sphere, to_vec(TX_POS[0], TX_POS[1], TX_POS[2]),
                     TX_RADIUS_M, mic_tx, "RT_TX", FOLDER + "/TX")

        groups = []
        seg_total = 0
        for si, item in enumerate(steps):
            name = item[0]
            rx_pos = item[1]
            rays = item[2]
            grp = []
            sfolder = "%s/Step_%03d" % (FOLDER, si)
            grp.append(spawn_marker(sphere, to_vec(rx_pos[0], rx_pos[1], rx_pos[2]),
                                    RX_RADIUS_M, mic_rx, "RX_%03d_%s" % (si, name), sfolder))
            for ri, poly in enumerate(rays):
                for pi in range(len(poly) - 1):
                    a = spawn_segment(cyl, poly[pi], poly[pi + 1], RAY_RADIUS_M, mic_ray,
                                      "Ray_%03d_%d_%d" % (si, ri, pi), sfolder)
                    if a:
                        grp.append(a)
                        seg_total += 1
            groups.append([a for a in grp if a])
            if (si + 1) % 20 == 0:
                _log("  ... 스텝 %d/%d (누적 선분 %d)" % (si + 1, len(steps), seg_total))

        _log("스폰 완료: 스텝 %d, 선분 %d" % (len(groups), seg_total))
        if MODE == "sequence":
            build_sequence(groups)
            _log("재생: 콘텐츠 브라우저에서 %s 더블클릭 → ▶" % SEQ_ASSET)
        else:
            _log("MODE=static: 전부 표시. 정합 확인 후 MODE='sequence'로 재실행.")
        _log("=== 완료 ===")
    except Exception as e:
        import traceback
        unreal.log_error("[unreal_scenario] 예외: %s\n%s" % (e, traceback.format_exc()))


main()
'''
