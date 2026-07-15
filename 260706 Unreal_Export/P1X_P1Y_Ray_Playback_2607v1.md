# P1X / P1Y — RT 광선 재생 (Blender.py → Unreal)

blender_mobility.py(RTWA 산출물)의 **검증된 SCENARIO_DATA**(TX/RX/광선 폴리라인)를
Unreal로 옮겨, Blender에서 하던 **키프레임 재생**을 재현한다. (Approach 2)

```
blender_mobility.py(053019db)
   │  P1X (서버, ast 파싱)
   ▼
ray_playback_<id>_<ts>.json
   │  P1Y (Unreal Python)
   ▼
Unreal: TX/RX/광선 스폰 + Level Sequence 키프레임 재생 → 영상
```

---

## P1X_BlenderScenario_to_JSON_2607v1.py  (서버 실행)
- **목적**: blender_mobility.py → Unreal 임포터용 JSON 변환
- **방법**: `import bpy` 없이 `ast.literal_eval`로 `SCENARIO_DATA`/설정 안전 추출
- **입력**: `.../sessions/053019db-.../Scenario_Results/blender_mobility.py`
- **출력**: `P1X_RayData_Results/ray_playback_<sessionid>_<ts>.json`
  - `tx_pos, fps, frames_per_step, start_frame, rx/tx/ray_radius`
  - `steps: [{rx_name, rx_pos, rays:[[[x,y,z],...],...]}, ...]`
- **실행**: `python3 P1X_BlenderScenario_to_JSON_2607v1.py [blender_mobility.py 경로]`
- **검증 결과(053019db)**: 173 스텝, 광선 887개, TX1_Main@(86.65,−60.31,39.54),
  FPS30/3프레임, 좌표 x[−25,135] y[−162,−28] z[−0.7,45] (glb v1.0 범위 내 → 정합)

## P1Y_Unreal_Ray_Playback_2607v1.py  (Unreal 에디터 실행)
- **목적**: JSON → TX(빨강)/RX(초록)/광선(노랑) 스폰 + 재생
- **MODE**
  - `static`: 전부 표시 (정합/색/스케일 검증) ← **먼저 이걸로**
  - `sequence`: Level Sequence + 스텝별 visibility 키프레임 (재생/영상)
- **광선 렌더**: 폴리라인 선분마다 얇은 실린더(발광) + `make_rot_from_z`로 정렬
- **CONFIG**
  | 항목 | 기본 | 설명 |
  |------|------|------|
  | `JSON_DIR` | 데스크톱 폴더 | ray_playback_*.json 위치 |
  | `MODE` | static | static→sequence |
  | `SCALE` | 100 | m→cm |
  | `FLIP_X/Y/Z` | False | 도시와 안 맞으면 토글 (Blender↔Unreal 좌표계) |
  | `RX/TX/RAY_RADIUS_M` | 1.2/2.0/0.3 | 크기[m] |
  | `MAX_STEPS` | 0 | 0=전체, N=앞 N스텝만(테스트) |
  | `VIS_KEY_VISIBLE` | True | 재생 시 보임/숨김 반대면 False |

## 정합(중요)
- 광선 좌표는 blender(Z-up) 기준. Unreal은 cm·좌수계.
- 도시 glb와 안 맞으면 `FLIP_Y`부터 토글(Blender 우수↔Unreal 좌수 보정), 그다음 `FLIP_X`.
- 먼저 `MAX_STEPS=5`, `MODE=static`로 소수만 띄워 도시와 맞춰본 뒤 전체/재생으로.

## 실행 환경
- P1X: Python 3.10+ (표준 ast/json)
- P1Y: Unreal Engine 5.x 내장 Python + Python Editor Script / Sequencer Scripting 플러그인

## 한계/주의
- Sequencer visibility 키 극성은 UE 버전에 따라 반대일 수 있음 → `VIS_KEY_VISIBLE` 토글로 대응.
- 광선 선분이 수천 개면 스폰에 수십 초. `MAX_STEPS`로 조절.
