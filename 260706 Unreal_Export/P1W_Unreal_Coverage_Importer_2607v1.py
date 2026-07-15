"""
P1W_Unreal_Coverage_Importer_2607v1.py
=================================================================
[Unreal Engine 에디터 내부에서 실행하는 스크립트]

P1V가 뽑은 커버리지 CSV(unreal_tx_*.csv / unreal_rx_*.csv)를 읽어서
Unreal 레벨에 아래를 생성한다.
  - TX 타워 마커 (밝은 빨강 발광 구체)
  - RX 커버리지 포인트 클라우드 (RSRP 세기별 색칠: 파랑=약함 → 빨강=강함)

blender.py(bpy)가 씬에 마커/커브를 스폰하던 것의 Unreal 대응판.

※ 본 버전(v1)은 "정적 커버리지 클라우드"까지만 만든다.
   시간축 키프레임 재생(모빌리티)은 다음 단계 P1X에서 Sequencer로 붙인다.
   먼저 CSV→Unreal 파이프라인이 화면에 뜨는지 검증하는 것이 목적.

-----------------------------------------------------------------
[실행 방법 - Unreal 에디터]
  1) 상단 메뉴 Tools(툴) → Execute Python Script... → 이 파일 선택
     (또는 Output Log 하단 Cmd 입력창을 Python 모드로 바꿔 파일 실행)
  2) 실행 전, 아래 CONFIG의 CSV_DIR 을 본인 PC 경로로 반드시 수정할 것.

[요구 플러그인] Python Editor Script Plugin (활성화 완료 상태여야 함)

[좌표계] CSV는 미터(RT 좌표). Unreal은 cm 단위 → SCALE(기본 ×100)로 변환.
         축이 뒤집혀 보이면 CONFIG의 FLIP_* 옵션으로 보정.

[실행 환경] Unreal Engine 5.x 내장 Python 3.x (numpy 불필요, 표준 csv 사용)
=================================================================
"""

import csv
import glob
import os

import unreal

# ===================== CONFIG (여기만 수정) =====================
# unreal_tx_*.csv / unreal_rx_*.csv 들이 들어있는 폴더 (본인 PC 경로!)
CSV_DIR = r"C:\Users\01jun\Documents\RT_Export"

SCALE = 100.0        # 미터 → cm 변환 (Unreal 기본 단위 cm)
FLIP_X = False       # 축이 좌우로 뒤집히면 True
FLIP_Y = True        # RT(Y up-north) ↔ Unreal(왼손좌표) 보정용, 필요시 토글

RX_SCALE = 3.0       # RX 구체 크기 [m] (맵이 ~1.6km라 3m 정도가 적당)
TX_SCALE = 12.0      # TX 마커 크기 [m]
MAX_RX = 2000        # 성능 보호용 RX 서브샘플 상한 (전체 5071 쓰려면 늘리기)
EMISSIVE = 3.0       # 발광 강도 (클수록 밝음)
SKIP_DEAD = True     # dead zone(RSRP=-inf) RX 제외 여부

NUM_BINS = 12        # RSRP 색 구간 수
ASSET_DIR = "/Game/RT_Coverage"     # 생성 머티리얼 저장 경로
FOLDER = "RT_Coverage"              # World Outliner 정리 폴더명
# ===============================================================

PARENT_MAT = None  # 런타임에 채워짐


# --------------------------------------------------------------
# 유틸
# --------------------------------------------------------------
def colormap(t):
    """t in [0,1] → (r,g,b) 5-stop 그라디언트 (파랑→시안→초록→노랑→빨강)."""
    t = max(0.0, min(1.0, t))
    stops = [
        (0.00, (0.0, 0.0, 1.0)),
        (0.25, (0.0, 1.0, 1.0)),
        (0.50, (0.0, 1.0, 0.0)),
        (0.75, (1.0, 1.0, 0.0)),
        (1.00, (1.0, 0.0, 0.0)),
    ]
    for i in range(len(stops) - 1):
        t0, c0 = stops[i]
        t1, c1 = stops[i + 1]
        if t <= t1:
            f = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
            return (c0[0] + (c1[0] - c0[0]) * f,
                    c0[1] + (c1[1] - c0[1]) * f,
                    c0[2] + (c1[2] - c0[2]) * f)
    return (1.0, 0.0, 0.0)


def latest(pattern):
    files = glob.glob(os.path.join(CSV_DIR, pattern))
    return max(files, key=os.path.getmtime) if files else None


def to_unreal_vec(x_m, y_m, z_m):
    """미터(RT) → Unreal cm 벡터 (축 보정 포함)."""
    x = -x_m if FLIP_X else x_m
    y = -y_m if FLIP_Y else y_m
    return unreal.Vector(x * SCALE, y * SCALE, z_m * SCALE)


# --------------------------------------------------------------
# 머티리얼 생성 (발광 색 파라미터 1개짜리 부모 + 구간별 인스턴스)
# --------------------------------------------------------------
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
    p.set_editor_property("default_value", unreal.LinearColor(1.0, 1.0, 1.0, 1.0))
    mel.connect_material_property(p, "", unreal.MaterialProperty.MP_EMISSIVE_COLOR)
    mel.recompile_material(mat)
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


# --------------------------------------------------------------
# 액터 스폰 / 정리
# --------------------------------------------------------------
def _actor_subsystem():
    return unreal.get_editor_subsystem(unreal.EditorActorSubsystem)


def clear_previous():
    """이전 실행에서 만든 RT_Coverage 폴더 액터 제거 (중복 방지)."""
    eas = _actor_subsystem()
    removed = 0
    for a in eas.get_all_level_actors():
        try:
            if str(a.get_folder_path()).startswith(FOLDER):
                eas.destroy_actor(a)
                removed += 1
        except Exception:
            pass
    if removed:
        unreal.log("이전 RT_Coverage 액터 %d개 제거" % removed)


def spawn_sphere(mesh, loc, scale, mic, label, folder):
    eas = _actor_subsystem()
    a = eas.spawn_actor_from_object(mesh, loc, unreal.Rotator(0, 0, 0))
    a.set_actor_scale3d(unreal.Vector(scale, scale, scale))
    smc = a.get_component_by_class(unreal.StaticMeshComponent)
    if smc:
        smc.set_material(0, mic)
        smc.set_cast_shadow(False)  # 포인트 수천개 → 그림자 끄면 훨씬 가벼움
    a.set_actor_label(label)
    a.set_folder_path(folder)
    return a


# --------------------------------------------------------------
# 메인
# --------------------------------------------------------------
def main():
    global PARENT_MAT

    rx_csv = latest("unreal_rx_*.csv")
    tx_csv = latest("unreal_tx_*.csv")
    if not rx_csv or not tx_csv:
        unreal.log_error("CSV를 찾을 수 없음. CSV_DIR 확인: %s" % CSV_DIR)
        return
    unreal.log("TX CSV: %s" % tx_csv)
    unreal.log("RX CSV: %s" % rx_csv)

    if not unreal.EditorAssetLibrary.does_directory_exist(ASSET_DIR):
        unreal.EditorAssetLibrary.make_directory(ASSET_DIR)

    PARENT_MAT = get_or_create_parent_material()

    # 색 구간별 머티리얼 인스턴스 + TX용
    bin_mics = []
    for i in range(NUM_BINS):
        t = i / (NUM_BINS - 1)
        c = colormap(t)
        col = (c[0] * EMISSIVE, c[1] * EMISSIVE, c[2] * EMISSIVE)
        bin_mics.append(make_mic("MI_RT_bin%02d" % i, col))
    tx_mic = make_mic("MI_RT_TX", (EMISSIVE * 2.0, EMISSIVE * 0.1, EMISSIVE * 0.1))

    sphere = unreal.load_asset("/Engine/BasicShapes/Sphere.Sphere")

    clear_previous()

    # ---- TX ----
    n_tx = 0
    with open(tx_csv, newline="") as f:
        for row in csv.DictReader(f):
            loc = to_unreal_vec(float(row["x"]), float(row["y"]), float(row["z"]))
            spawn_sphere(sphere, loc, TX_SCALE, tx_mic,
                         "RT_TX_%s" % row["tx_index"], FOLDER + "/TX")
            n_tx += 1

    # ---- RX ----
    with open(rx_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    step = max(1, len(rows) // MAX_RX)
    sel = rows[::step]

    task = unreal.ScopedSlowTask(len(sel), "RX 커버리지 포인트 생성 중...")
    task.make_dialog(True)
    n_rx = 0
    for r in sel:
        if task.should_cancel():
            break
        task.enter_progress_frame(1)
        if SKIP_DEAD and r["rsrp_dbm"] == "-inf":
            continue
        norm = float(r["rsrp_norm"])
        b = min(NUM_BINS - 1, max(0, int(norm * NUM_BINS)))
        loc = to_unreal_vec(float(r["x"]), float(r["y"]), float(r["z"]))
        spawn_sphere(sphere, loc, RX_SCALE, bin_mics[b],
                     "RT_RX_%s" % r["rx_index"], FOLDER + "/RX")
        n_rx += 1

    unreal.log("=" * 50)
    unreal.log("P1W 완료: TX %d개, RX %d개 스폰 (서브샘플 step=%d)" % (n_tx, n_rx, step))
    unreal.log("World Outliner의 '%s' 폴더 확인" % FOLDER)
    unreal.log("=" * 50)


main()
