# Unreal Scenario 통합 (RTWA "9. Scenario Results")

RTWA가 blender_mobility.py를 만들 때, **Unreal용 자체 완결형 `unreal_scenario.py`도 함께 생성**하고
웹 UI에서 다운로드 + 도움말을 제공하도록 통합.

## 변경 파일
| 파일 | 변경 |
|------|------|
| `backend/rt_agent_park2/scenario_builder.py` | `_write_unreal_script()` 추가 + `_UNREAL_BODY` 템플릿. `generate_scenario()`가 `unreal_scenario.py` 생성 + 반환값에 `unreal_script_path` 추가 |
| `backend/app.py` | `/scenario/generate` 응답/영속화에 `unreal_script_rel` 추가 |
| `frontend/src/pages/ScenarioResultsPage.tsx` | `unreal_scenario.py 다운로드` 버튼 + `도움말 : Unreal Engine 사용법` 추가 |

## unreal_scenario.py (산출물) 특징
- **자체 완결형 한 파일**: SCENARIO_DATA(TX/RX/광선 폴리라인)를 파일에 baked-in.
  blender_mobility.py 와 대칭 — Unreal에서 파일 하나만 실행하면 됨(별도 JSON 불필요).
- **동작**: Tools → Execute Python Script → TX/RX/광선 스폰 + Sequencer 키프레임 재생(`SEQ_RayPlayback`) 생성.
- **CONFIG(파일 상단)**: `SCALE=100`, `FLIP_Y=True`(glTF 임포트 기본), `MODE="sequence"`,
  `MAX_STEPS`, `VIS_KEY_VISIBLE`, 색/발광/반경 등.
- **요구**: Python Editor Script + Sequencer Scripting 플러그인. 도시 메시는 사용자가 별도 임포트.

## 정합/재생 주의
- 도시(glb→glTF 임포트)와 광선 정합: 보통 `FLIP_Y=True`. 안 맞으면 `FLIP_X`/`SCALE` 조정.
- 재생이 반대(전부 보임/숨김)면 `VIS_KEY_VISIBLE=False`.

## 반영(적용) 절차
- **백엔드**: 코드만 수정됨 → **RT 잡이 없는 때 uvicorn 재시작** 필요 (RT 잡은 uvicorn 자식 프로세스라
  실행 중 재시작 시 죽을 수 있음 — job_runner.shutdown()이 자식 kill).
- **프론트**: vite 재빌드/재시작(안전, RT 무관).
- **기존 세션**: unreal_scenario.py가 없으므로, 재시작 후 8.Scenario에서 **재생성**하면 버튼 노출.
  (신규 시나리오는 자동 생성)

## 검증
- `scenario_builder.py`, `app.py` py_compile OK.
- `ScenarioResultsPage.tsx` 진단 0건.
- 053019db 실데이터(173스텝)로 `unreal_scenario.py` 생성 + py_compile OK (94KB).

## 근거 산출물(참고)
- 개발/검증용 원본: `260706 Unreal_Export/P1X_*`(변환기), `P1Y_*`(임포터).
  productization: 위 로직을 scenario_builder._write_unreal_script 에 self-contained 로 이식.
- 사용법 문서: `references/Unreal Engine 사용법.md`
