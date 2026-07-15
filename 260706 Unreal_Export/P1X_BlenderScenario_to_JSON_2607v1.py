"""
P1X_BlenderScenario_to_JSON_2607v1.py
=================================================================
RTWA가 생성한 blender_mobility.py 에서 SCENARIO_DATA / 설정값을 추출해
Unreal 임포터용 JSON으로 변환한다. (Approach 2: 검증된 blender.py 재사용)

[입력]  blender_mobility.py  (RTWA Scenario_Results 산출물)
  - SCENARIO_DATA = [(rx_name, (x,y,z), [ [ (x,y,z),... ], ... ]), ...]
  - TX_NAME, TX_POS, FPS, FRAMES_PER_STEP, START_FRAME, RX/TX/RAY 반경

[출력]  P1X_RayData_Results/ray_playback_<sessionid>_<ts>.json
  {
    "tx_name", "tx_pos":[x,y,z],
    "fps", "frames_per_step", "start_frame",
    "rx_radius","tx_radius","ray_width",
    "num_steps","num_rays_total",
    "steps": [ {"rx_name","rx_pos":[x,y,z],"rays":[[[x,y,z],...],...]}, ... ]
  }

[방법]  bpy 실행 없이 ast로 안전 파싱 (import bpy 하지 않음)

-----------------------------------------------------------------
실행 환경: Python 3.10+ (표준 라이브러리 ast/json 만 사용)
실행 서버: dclserver78
사용법:
  python3 P1X_BlenderScenario_to_JSON_2607v1.py [blender_mobility.py 경로]
  (경로 생략 시 아래 DEFAULT_SRC 사용)
=================================================================
"""

import ast
import json
import os
import re
import sys
from datetime import datetime

DEFAULT_SRC = ("../260528 Web_Agent/backend/sessions/"
               "053019db-d3b3-446e-a093-904736965da1/Scenario_Results/blender_mobility.py")
RESULTS_DIR = "P1X_RayData_Results"

# 추출 대상 (스칼라/튜플 리터럴)
SCALARS = ["TX_NAME", "TX_POS", "FPS", "FRAMES_PER_STEP", "START_FRAME",
           "RX_RADIUS", "TX_RADIUS", "RAY_WIDTH"]


def extract_assignments(src_text):
    """ast로 모듈 최상위 할당문에서 대상 값들을 literal_eval."""
    tree = ast.parse(src_text)
    found = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for tgt in node.targets:
            if isinstance(tgt, ast.Name) and (tgt.id in SCALARS or tgt.id == "SCENARIO_DATA"):
                try:
                    found[tgt.id] = ast.literal_eval(node.value)
                except Exception as e:
                    print(f"[WARN] {tgt.id} literal_eval 실패: {e}")
    return found


def session_id_from_path(path):
    m = re.search(r"sessions/([0-9a-fA-F]{8})", path)
    return m.group(1) if m else "unknown"


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SRC
    if not os.path.exists(src):
        print(f"[ERROR] 입력 없음: {src}")
        sys.exit(1)

    with open(src, encoding="utf-8") as f:
        text = f.read()

    vals = extract_assignments(text)
    if "SCENARIO_DATA" not in vals:
        print("[ERROR] SCENARIO_DATA 를 찾지 못함")
        sys.exit(1)

    scenario = vals["SCENARIO_DATA"]
    steps = []
    n_rays = 0
    for item in scenario:
        rx_name, rx_pos, rays = item
        ray_list = [[list(p) for p in poly] for poly in rays]
        n_rays += len(ray_list)
        steps.append({
            "rx_name": rx_name,
            "rx_pos": list(rx_pos),
            "rays": ray_list,
        })

    out = {
        "source": os.path.abspath(src),
        "session_id": session_id_from_path(src),
        "tx_name": vals.get("TX_NAME", "TX"),
        "tx_pos": list(vals.get("TX_POS", (0.0, 0.0, 0.0))),
        "fps": int(vals.get("FPS", 30)),
        "frames_per_step": int(vals.get("FRAMES_PER_STEP", 3)),
        "start_frame": int(vals.get("START_FRAME", 1)),
        "rx_radius": float(vals.get("RX_RADIUS", 1.2)),
        "tx_radius": float(vals.get("TX_RADIUS", 2.0)),
        "ray_width": float(vals.get("RAY_WIDTH", 0.3)),
        "num_steps": len(steps),
        "num_rays_total": n_rays,
        "steps": steps,
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(RESULTS_DIR, f"ray_playback_{out['session_id']}_{ts}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)

    # 좌표 범위 요약(정합 확인용)
    xs = [p[0] for s in steps for poly in s["rays"] for p in poly]
    ys = [p[1] for s in steps for poly in s["rays"] for p in poly]
    zs = [p[2] for s in steps for poly in s["rays"] for p in poly]

    print("=" * 60)
    print("  P1X 변환 완료")
    print("=" * 60)
    print(f"  입력      : {src}")
    print(f"  세션      : {out['session_id']}")
    print(f"  TX        : {out['tx_name']} @ {out['tx_pos']}")
    print(f"  FPS/스텝  : {out['fps']} / {out['frames_per_step']} 프레임")
    print(f"  스텝(RX)  : {out['num_steps']}개")
    print(f"  광선 총수 : {out['num_rays_total']}개")
    if xs:
        print(f"  좌표범위  : x[{min(xs):.1f},{max(xs):.1f}] "
              f"y[{min(ys):.1f},{max(ys):.1f}] z[{min(zs):.1f},{max(zs):.1f}]")
    print(f"  → {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
