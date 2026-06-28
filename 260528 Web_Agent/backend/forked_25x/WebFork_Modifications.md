# WebFork_Modifications.md

`260528 Web_Agent/backend/forked_25x/` 폴더에 들어 있는 25\* fork 파일들의 수정 사유와 diff 요약.

## 정책

- **원본 25\* 폴더는 절대 무수정.** 수정이 필요한 경우 해당 파일을 이 폴더로 복사한 뒤 `_web` suffix를 붙여 수정.
- 원본 함수/클래스의 시그니처는 가급적 유지. Web Agent 호환성을 위한 수정만 적용.
- 새 helper는 파일 **끝에** 추가하고 주석으로 `# [Web Agent Fork]` 표기.
- 어댑터 (`backend/adapters/*.py`)는 fork된 모듈을 우선 import 한다.

## 변경 이력

### 2026-05-28: P1A_RT_to_Rays_2509v6_web.py

원본: `251218 E_MIMO_BM/P1A_RT_to_Rays_2509v6.py` (1990 LOC)

#### 변경 1: P1A_Config 클래스 확장 — Web Agent 전용 필드

| 신규 필드 | 기본값 | 사유 |
|----------|--------|------|
| `WEB_OVERRIDE_SCENE_PATH` | `None` | 세션마다 다른 scene.xml 절대 경로를 외부에서 주입 |
| `PATHSOLVER_LOS` | `True` | 원본에서 hardcoded → 사용자 옵션화 |
| `PATHSOLVER_SPECULAR_REFLECTION` | `True` | 동상 |
| `PATHSOLVER_DIFFUSE_REFLECTION` | `True` | 동상 |
| `PATHSOLVER_REFRACTION` | `True` | 동상 |
| `PATHSOLVER_SYNTHETIC_ARRAY` | `False` | 동상 |

기존 25\* 검증값(`True, True, True, True, False`)을 디폴트로 유지하므로 CLI 모드에서는 원본과 100% 동일하게 동작.

#### 변경 2: `Scene.find_file()` 동적 경로 지원

```python
# [Web Agent Fork] WEB_OVERRIDE_SCENE_PATH가 설정되어 있으면 우선 사용
if getattr(p1a_config, "WEB_OVERRIDE_SCENE_PATH", None):
    override = p1a_config.WEB_OVERRIDE_SCENE_PATH
    if os.path.exists(override):
        return override
    raise FileNotFoundError(f"[Web Agent] WEB_OVERRIDE_SCENE_PATH not found: {override}")
```

설정되지 않은 경우 기존 `SCENE_FILE_PATHS` 탐색 로직 그대로 유지.

#### 변경 3: `PathRT.trace_rays()` PathSolver flag 옵션화

```python
paths = p_solver(
    scene=scene,
    max_depth=p1a_config.MAX_DEPTH,
    los=getattr(p1a_config, "PATHSOLVER_LOS", True),
    specular_reflection=getattr(p1a_config, "PATHSOLVER_SPECULAR_REFLECTION", True),
    diffuse_reflection=getattr(p1a_config, "PATHSOLVER_DIFFUSE_REFLECTION", True),
    refraction=getattr(p1a_config, "PATHSOLVER_REFRACTION", True),
    synthetic_array=getattr(p1a_config, "PATHSOLVER_SYNTHETIC_ARRAY", False),
    seed=p1a_config.RT_SEED,
)
```

기본값은 원본과 동일.

#### 변경 4: 파일 끝에 Web Agent helper 추가

`web_apply_config_override(overrides: dict)` — 런타임에 `p1a_config` 속성 일괄 덮어쓰기.
`web_run_pipeline(overrides: dict | None = None)` — overrides 적용 + `Pipeline.execute_main()` 호출.

CLI 모드(`if __name__ == "__main__"`)는 그대로 두어 25\*과 동일하게 단독 실행 가능.

#### 영향 분석

- ✅ 원본 단일 실행 (`python P1A_*.py`) — 동작 동일.
- ✅ Web Agent 어댑터 호출 — `web_run_pipeline({...})`로 모든 설정 주입.
- ✅ 다른 25\* 파일에서 import (`from P1A_RT_to_Rays_2509v6 import ...`) — 클래스/함수 시그니처 동일하므로 호환.

## 향후 fork 예정

- `P1B_Valid_RX_Filter_2509v1.py`: NPZ 입력 경로 외부 주입 (필요 시)
- 그 외 P1C~P1Q: 일반적으로 NPZ I/O만 하므로 fork 없이 어댑터에서 cwd 변경으로 해결 시도. 한계 도달 시에만 fork.
