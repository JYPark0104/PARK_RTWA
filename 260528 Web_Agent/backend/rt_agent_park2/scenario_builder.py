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


def parse_usda(usda_path: str | Path) -> dict:
    """USDA → {rx_data, tx_data, ray_map}.

    rx_data : [(idx, name, x, y, z), ...]  (idx 오름차순)
    tx_data : [(idx, name, x, y, z), ...]
    ray_map : {tx_idx: {rx_idx: [ [(x,y,z),...], ... ]}}
    """
    text = Path(usda_path).read_text(encoding="utf-8", errors="ignore")

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
def get_scenario_data(usda_path: str | Path, obj_path: str | Path,
                      max_edges: int = 60000) -> dict:
    parsed = parse_usda(usda_path)
    edges = load_obj_edges(obj_path, max_edges=max_edges) if Path(obj_path).exists() else []
    return {
        "rx": [{"idx": i, "x": x, "y": y, "z": z} for (i, _n, x, y, z) in parsed["rx_data"]],
        "tx": [{"idx": i, "x": x, "y": y, "z": z} for (i, _n, x, y, z) in parsed["tx_data"]],
        "edges": edges,
        "num_rx": len(parsed["rx_data"]),
        "num_tx": len(parsed["tx_data"]),
    }


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
