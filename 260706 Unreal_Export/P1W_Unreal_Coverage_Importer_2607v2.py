"""
P1W_Unreal_Coverage_Importer_2607v2.py
=================================================================
[Unreal Engine 에디터 내부 실행 스크립트] - 안정화 버전(v2)

v1 대비 변경 (멈춤 방지):
  - ScopedSlowTask 모달 진행바 제거 → unreal.log 진행 출력
  - MAX_RX 기본 300 (가볍게 시작)
  - 각 단계/에러를 로그로 출력 (어디서 멈추는지 즉시 확인)
  - 머티리얼/그림자 처리에 예외 처리 추가
  - CSV_DIR 기본값을 사용자 데스크톱 폴더로 지정

기능(동일): P1V CSV(unreal_tx/rx_*.csv) → TX 발광 구체 + RX 커버리지
            포인트 클라우드(RSRP 색칠) 생성.

실행: Tools → Execute Python Script... → 이 파일
요구: Python Editor Script Plugin
좌표: 미터(RT) → cm(×SCALE). ※ 씬 정합은 별도(트윈 버전 확인 필요)
실행환경: Unreal Engine 5.x 내장 Python (표준 csv 사용)
=================================================================
"""

import csv
import glob
import os

import unreal

# ===================== CONFIG (필요시 수정) =====================
# 기본값을 사용자 다운로드 폴더로 지정 (수정 불필요하도록)
CSV_DIR = r"C:\Users\01jun\Desktop\26.07.07. Unreal Engine 튜토리얼"

SCALE = 100.0        # 미터 → cm
FLIP_X = False
FLIP_Y = True
RX_SCALE = 3.0       # RX 구체 크기 [m]
TX_SCALE = 12.0      # TX 마커 크기 [m]
MAX_RX = 300         # 서브샘플 상한 (안정 확인 후 늘리기)
EMISSIVE = 3.0
SKIP_DEAD = True
NUM_BINS = 12
ASSET_DIR = "/Game/RT_Coverage"
FOLDER = "RT_Coverage"
LOG_EVERY = 50       # N개마다 진행 로그
# ===============================================================

PARENT_MAT = None


def log(msg):
    unreal.log("[P1W] " + str(msg))


def colormap(t):
    t = max(0.0, min(1.0, t))
    stops = [(0.0, (0, 0, 1)), (0.25, (0, 1, 1)), (0.5, (0, 1, 0)),
             (0.75, (1, 1, 0)), (1.0, (1, 0, 0))]
    for i in range(len(stops) - 1):
        t0, c0 = stops[i]
        t1, c1 = stops[i + 1]
        if t <= t1:
            f = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
            return (c0[0] + (c1[0] - c0[0]) * f,
                    c0[1] + (c1[1] - c0[1]) * f,
                    c0[2] + (c1[2] - c0[2]) * f)
    return (1, 0, 0)


def latest(pattern):
    files = glob.glob(os.path.join(CSV_DIR, pattern))
    return max(files, key=os.path.getmtime) if files else None


def to_vec(x_m, y_m, z_m):
    x = -x_m if FLIP_X else x_m
    y = -y_m if FLIP_Y else y_m
    return unreal.Vector(x * SCALE, y * SCALE, z_m * SCALE)


def get_or_create_parent_material():
    path = ASSET_DIR + "/M_RTCoverage"
    if unreal.EditorAssetLibrary.does_asset_exist(path):
        return unreal.load_asset(path)
    at = unreal.AssetToolsHelpers.get_asset_tools()
    mat = at.create_asset("M_RTCoverage", ASSET_DIR, unreal.Material,
                          unreal.MaterialFactoryNew())
    mel = unreal.MaterialEditingLibrary
    p = mel.create_material_expression(mat, unreal.MaterialExpressionVectorParameter, -400, 0)
    p.set_editor_property("parameter_name", "EmissiveColor")
    p.set_editor_property("default_value", unreal.LinearColor(1, 1, 1, 1))
    mel.connect_material_property(p, "", unreal.MaterialProperty.MP_EMISSIVE_COLOR)
    mel.recompile_material(mat)
    unreal.EditorAssetLibrary.save_asset(path)
    return mat


def make_mic(name, color_rgb):
    path = ASSET_DIR + "/" + name
    if unreal.EditorAssetLibrary.does_asset_exist(path):
        return unreal.load_asset(path)
    at = unreal.AssetToolsHelpers.get_asset_tools()
    mic = at.create_asset(name, ASSET_DIR, unreal.MaterialInstanceConstant,
                          unreal.MaterialInstanceConstantFactoryNew())
    mel = unreal.MaterialEditingLibrary
    mel.set_material_instance_parent(mic, PARENT_MAT)
    mel.set_material_instance_vector_parameter_value(
        mic, "EmissiveColor",
        unreal.LinearColor(color_rgb[0], color_rgb[1], color_rgb[2], 1.0))
    return mic


def eas():
    return unreal.get_editor_subsystem(unreal.EditorActorSubsystem)


def clear_previous():
    s = eas()
    removed = 0
    for a in s.get_all_level_actors():
        try:
            if str(a.get_folder_path()).startswith(FOLDER):
                s.destroy_actor(a)
                removed += 1
        except Exception:
            pass
    log("이전 RT_Coverage 액터 %d개 제거" % removed)


def spawn(mesh, loc, scale, mic, label, folder):
    a = eas().spawn_actor_from_object(mesh, loc, unreal.Rotator(0, 0, 0))
    a.set_actor_scale3d(unreal.Vector(scale, scale, scale))
    try:
        smc = a.get_component_by_class(unreal.StaticMeshComponent)
        if smc:
            if mic:
                smc.set_material(0, mic)
            smc.set_editor_property("cast_shadow", False)
    except Exception as e:
        log("컴포넌트 설정 경고: %s" % e)
    try:
        a.set_actor_label(label)
        a.set_folder_path(folder)
    except Exception:
        pass
    return a


def main():
    global PARENT_MAT
    try:
        log("=== 시작 ===")
        rx_csv = latest("unreal_rx_*.csv")
        tx_csv = latest("unreal_tx_*.csv")
        if not rx_csv or not tx_csv:
            unreal.log_error("[P1W] CSV 없음. CSV_DIR 확인: %s" % CSV_DIR)
            return
        log("TX CSV: %s" % tx_csv)
        log("RX CSV: %s" % rx_csv)

        log("[1/5] 에셋 폴더/머티리얼 준비")
        if not unreal.EditorAssetLibrary.does_directory_exist(ASSET_DIR):
            unreal.EditorAssetLibrary.make_directory(ASSET_DIR)
        PARENT_MAT = get_or_create_parent_material()

        log("[2/5] 색 구간 머티리얼 인스턴스 생성 (%d개)" % NUM_BINS)
        bin_mics = []
        for i in range(NUM_BINS):
            c = colormap(i / (NUM_BINS - 1))
            bin_mics.append(make_mic("MI_RT_bin%02d" % i,
                                     (c[0] * EMISSIVE, c[1] * EMISSIVE, c[2] * EMISSIVE)))
        tx_mic = make_mic("MI_RT_TX", (EMISSIVE * 2.0, EMISSIVE * 0.1, EMISSIVE * 0.1))

        sphere = unreal.load_asset("/Engine/BasicShapes/Sphere.Sphere")

        log("[3/5] 이전 액터 정리")
        clear_previous()

        log("[4/5] TX 스폰")
        n_tx = 0
        with open(tx_csv, newline="") as f:
            for row in csv.DictReader(f):
                spawn(sphere, to_vec(float(row["x"]), float(row["y"]), float(row["z"])),
                      TX_SCALE, tx_mic, "RT_TX_%s" % row["tx_index"], FOLDER + "/TX")
                n_tx += 1
        log("  TX %d개 완료" % n_tx)

        log("[5/5] RX 커버리지 스폰 (MAX_RX=%d)" % MAX_RX)
        with open(rx_csv, newline="") as f:
            rows = list(csv.DictReader(f))
        step = max(1, len(rows) // MAX_RX)
        sel = rows[::step]
        n_rx = 0
        for idx, r in enumerate(sel):
            if SKIP_DEAD and r["rsrp_dbm"] == "-inf":
                continue
            b = min(NUM_BINS - 1, max(0, int(float(r["rsrp_norm"]) * NUM_BINS)))
            spawn(sphere, to_vec(float(r["x"]), float(r["y"]), float(r["z"])),
                  RX_SCALE, bin_mics[b], "RT_RX_%s" % r["rx_index"], FOLDER + "/RX")
            n_rx += 1
            if (idx + 1) % LOG_EVERY == 0:
                log("  ... %d / %d" % (idx + 1, len(sel)))

        log("=== 완료: TX %d, RX %d (step=%d) ===" % (n_tx, n_rx, step))
        log("World Outliner의 '%s' 폴더 확인. 액터 선택 후 뷰포트에서 F 키로 이동." % FOLDER)
    except Exception as e:
        import traceback
        unreal.log_error("[P1W] 예외: %s\n%s" % (e, traceback.format_exc()))


main()
