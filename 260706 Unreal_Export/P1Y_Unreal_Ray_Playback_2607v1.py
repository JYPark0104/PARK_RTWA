"""
P1Y_Unreal_Ray_Playback_2607v1.py
=================================================================
[Unreal Engine 에디터 내부 실행 스크립트]

P1X가 만든 ray_playback_*.json 을 읽어 Unreal에 다음을 생성한다.
  - TX 마커 (빨강 발광 구체)
  - 스텝별 RX 마커 (초록 발광 구체)
  - 광선 폴리라인 (노랑 발광, 선분마다 얇은 실린더)
blender_mobility.py 를 Blender에서 재생하던 것의 Unreal 재현판.

[MODE]
  "static"   : 전부 한 번에 표시 (정합/색/스케일 검증용) ← 먼저 이걸로!
  "sequence" : Level Sequence 생성 + 스텝별 visibility 키프레임 (재생/영상용)

먼저 MODE="static" 으로 도시와 정합(위치/스케일/축)을 맞춘 뒤,
MODE="sequence" 로 바꿔 재생을 만든다.

실행: Tools → Execute Python Script...
요구: Python Editor Script Plugin (+ 재생은 Sequencer Scripting)
-----------------------------------------------------------------
"""

import glob
import json
import os

import unreal

# ===================== CONFIG (필요시 수정) =====================
JSON_DIR = r"C:\Users\01jun\Desktop\26.07.07. Unreal Engine 튜토리얼"  # ray_playback_*.json 위치
MODE = "static"          # "static" 먼저 → 정합 확인 후 "sequence"

SCALE = 100.0            # 미터(RT) → cm(Unreal)
FLIP_X = False           # 도시와 안 맞으면 토글
FLIP_Y = False           # Blender(우수)↔Unreal(좌수) 보정: 보통 여기부터 시도
FLIP_Z = False

RX_RADIUS_M = 1.2        # RX 구체 반경 [m]
TX_RADIUS_M = 2.0        # TX 구체 반경 [m]
RAY_RADIUS_M = 0.3       # 광선 튜브 반경 [m]
EMISSIVE = 4.0

MAX_STEPS = 0            # 0 = 전체, >0 이면 앞에서 그만큼만 (테스트용)
ASSET_DIR = "/Game/RT_Rays"
FOLDER = "RT_Rays"

# 재생(sequence) 설정
SEQ_ASSET = "/Game/RT_Rays/SEQ_RayPlayback"
VIS_KEY_VISIBLE = True   # visibility 키 값 극성. 재생 시 반대로 보이면 False 로 바꿀 것
# ===============================================================

PARENT_MAT = None


def log(m):
    unreal.log("[P1Y] " + str(m))


def latest_json():
    fs = glob.glob(os.path.join(JSON_DIR, "ray_playback_*.json"))
    return max(fs, key=os.path.getmtime) if fs else None


def to_vec(x_m, y_m, z_m):
    x = -x_m if FLIP_X else x_m
    y = -y_m if FLIP_Y else y_m
    z = -z_m if FLIP_Z else z_m
    return unreal.Vector(x * SCALE, y * SCALE, z * SCALE)


def eas():
    return unreal.get_editor_subsystem(unreal.EditorActorSubsystem)


# --------------------------------------------------------------
# 머티리얼 (발광 색 파라미터 부모 + 3색 인스턴스)
# --------------------------------------------------------------
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


# --------------------------------------------------------------
# 스폰
# --------------------------------------------------------------
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
    log("이전 %s 액터 %d개 제거" % (FOLDER, n))


def set_mat(actor, mic):
    try:
        c = actor.get_component_by_class(unreal.StaticMeshComponent)
        if c:
            c.set_material(0, mic)
            c.set_editor_property("cast_shadow", False)
    except Exception:
        pass


def spawn_marker(mesh, loc, radius_m, mic, label, folder):
    a = eas().spawn_actor_from_object(mesh, loc, unreal.Rotator(0, 0, 0))
    # 기본 구체 반경 50cm(scale1) → 원하는 반경[m]*100cm / 50
    s = (radius_m * SCALE) / 50.0
    a.set_actor_scale3d(unreal.Vector(s, s, s))
    set_mat(a, mic)
    a.set_actor_label(label)
    a.set_folder_path(folder)
    return a


def spawn_segment(cyl, p0, p1, radius_m, mic, label, folder):
    v0 = to_vec(*p0)
    v1 = to_vec(*p1)
    dx, dy, dz = v1.x - v0.x, v1.y - v0.y, v1.z - v0.z
    length = (dx * dx + dy * dy + dz * dz) ** 0.5
    if length < 1.0:
        return None
    mid = unreal.Vector((v0.x + v1.x) / 2, (v0.y + v1.y) / 2, (v0.z + v1.z) / 2)
    rot = unreal.MathLibrary.make_rot_from_z(unreal.Vector(dx, dy, dz))
    a = eas().spawn_actor_from_object(cyl, mid, rot)
    # 기본 실린더: 높이 100cm, 반경 50cm (scale1)
    sr = (radius_m * SCALE) / 50.0
    a.set_actor_scale3d(unreal.Vector(sr, sr, length / 100.0))
    set_mat(a, mic)
    a.set_actor_label(label)
    a.set_folder_path(folder)
    return a


# --------------------------------------------------------------
# 재생(Level Sequence + visibility 키프레임)
# --------------------------------------------------------------
def build_sequence(step_actor_groups, fps, frames_per_step, start_frame):
    """step_actor_groups: [[actor, actor, ...], ...] 스텝별 액터 묶음."""
    if unreal.EditorAssetLibrary.does_asset_exist(SEQ_ASSET):
        unreal.EditorAssetLibrary.delete_asset(SEQ_ASSET)
    at = unreal.AssetToolsHelpers.get_asset_tools()
    seq = at.create_asset("SEQ_RayPlayback", ASSET_DIR, unreal.LevelSequence,
                          unreal.LevelSequenceFactoryNew())
    seq.set_display_rate(unreal.FrameRate(fps, 1))
    total = start_frame + len(step_actor_groups) * frames_per_step
    seq.set_playback_start(0)
    seq.set_playback_end(total)

    for si, actors in enumerate(step_actor_groups):
        on0 = start_frame + si * frames_per_step
        on1 = on0 + frames_per_step
        for a in actors:
            binding = seq.add_possessable(a)
            track = binding.add_track(unreal.MovieSceneVisibilityTrack)
            section = track.add_section()
            section.set_range(0, total)
            ch = section.get_all_channels()[0]
            vis, hid = (VIS_KEY_VISIBLE, not VIS_KEY_VISIBLE)
            # 기본 숨김 → 자기 스텝 구간만 보이게
            ch.add_key(unreal.FrameNumber(0), hid)
            ch.add_key(unreal.FrameNumber(int(on0)), vis)
            ch.add_key(unreal.FrameNumber(int(on1)), hid)
    unreal.EditorAssetLibrary.save_asset(SEQ_ASSET)
    log("Level Sequence 생성: %s (총 %d프레임, %dfps)" % (SEQ_ASSET, total, fps))


# --------------------------------------------------------------
# 메인
# --------------------------------------------------------------
def main():
    global PARENT_MAT
    try:
        jf = latest_json()
        if not jf:
            unreal.log_error("[P1Y] ray_playback_*.json 없음. JSON_DIR 확인: %s" % JSON_DIR)
            return
        log("JSON: %s" % jf)
        with open(jf, encoding="utf-8") as f:
            data = json.load(f)

        tx_pos = data["tx_pos"]
        fps = int(data.get("fps", 30))
        fps_step = int(data.get("frames_per_step", 3))
        start_frame = int(data.get("start_frame", 1))
        steps = data["steps"]
        if MAX_STEPS > 0:
            steps = steps[:MAX_STEPS]
        log("스텝 %d개, MODE=%s" % (len(steps), MODE))

        if not unreal.EditorAssetLibrary.does_directory_exist(ASSET_DIR):
            unreal.EditorAssetLibrary.make_directory(ASSET_DIR)
        PARENT_MAT = get_parent_material()
        mic_tx = make_mic("MI_TX", (EMISSIVE * 2.0, EMISSIVE * 0.05, EMISSIVE * 0.05))
        mic_rx = make_mic("MI_RX", (EMISSIVE * 0.05, EMISSIVE * 2.0, EMISSIVE * 0.05))
        mic_ray = make_mic("MI_Ray", (EMISSIVE * 2.0, EMISSIVE * 1.5, EMISSIVE * 0.0))

        sphere = unreal.load_asset("/Engine/BasicShapes/Sphere.Sphere")
        cyl = unreal.load_asset("/Engine/BasicShapes/Cylinder.Cylinder")

        clear_previous()

        # TX (한 번만)
        spawn_marker(sphere, to_vec(*tx_pos), TX_RADIUS_M, mic_tx,
                     "RT_TX", FOLDER + "/TX")

        step_groups = []
        seg_total = 0
        for si, st in enumerate(steps):
            grp = []
            sfolder = "%s/Step_%03d" % (FOLDER, si)
            # RX
            grp.append(spawn_marker(sphere, to_vec(*st["rx_pos"]), RX_RADIUS_M, mic_rx,
                                    "RX_%03d_%s" % (si, st["rx_name"]), sfolder))
            # rays
            for ri, poly in enumerate(st["rays"]):
                for pi in range(len(poly) - 1):
                    a = spawn_segment(cyl, poly[pi], poly[pi + 1], RAY_RADIUS_M, mic_ray,
                                      "Ray_%03d_%d_%d" % (si, ri, pi), sfolder)
                    if a:
                        grp.append(a)
                        seg_total += 1
            step_groups.append([a for a in grp if a])
            if (si + 1) % 20 == 0:
                log("  ... 스텝 %d/%d (누적 광선선분 %d)" % (si + 1, len(steps), seg_total))

        log("스폰 완료: 스텝 %d, 광선선분 %d" % (len(step_groups), seg_total))

        if MODE == "sequence":
            build_sequence(step_groups, fps, fps_step, start_frame)
            log("재생: 콘텐츠 브라우저에서 %s 더블클릭 → Sequencer 재생" % SEQ_ASSET)
        else:
            log("MODE=static: 전부 표시. 정합 확인 후 MODE='sequence'로 재실행하세요.")
        log("=== 완료 ===")
    except Exception as e:
        import traceback
        unreal.log_error("[P1Y] 예외: %s\n%s" % (e, traceback.format_exc()))


main()
