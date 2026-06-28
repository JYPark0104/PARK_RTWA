# Web Agent 작업 중 사용한 터미널 명령어

AGENTS.md 규칙: 새로운 명령어가 사용될 때마다 계속 업데이트.

## 디렉토리 스캐폴드 (2026-05-28)

```bash
cd "260528 Web_Agent"
mkdir -p backend/{adapters,jobs,forked_25x,derived,sessions,tests}
mkdir -p frontend/{src/{pages,components,lib,store},public}
mkdir -p docs logs
```

## Python venv + pip 부트스트랩

```bash
cd "260528 Web_Agent"
python3.10 -m venv --without-pip .venv-webagent
source .venv-webagent/bin/activate
curl -sSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
python /tmp/get-pip.py
pip install --upgrade pip
pip install -r backend/requirements.txt
```

## 의존성 설치 / 점검

| 명령 | 용도 |
|------|------|
| `pip install sionna tensorflow mitsuba trimesh geopandas` | 핵심 RT 의존성 |
| `pip install "tensorflow[and-cuda]==2.21.0"` | nvidia-* pip 휠 (cudnn 9, cublas, nccl 등) 같이 설치 |
| `pip install fastapi 'uvicorn[standard]' python-multipart websockets pydantic` | API 서버 |
| `pip install matplotlib numpy scipy` | 시각화 / 수치 |
| `pip install trimesh[easy]` | embreex 포함 trimesh 풀 셋 |
| `pip list | grep -E "sionna|tensorflow|mitsuba|trimesh|fastapi"` | 설치 확인 |
| `python -c "import sionna; print(sionna.__version__)"` | smoke test |
| `python -c "import sionna.rt; print(sionna.rt.__file__)"` | sionna rt 위치 확인 |

## GPU 모드 토글 (RTX 5090 sm_120 — 실험적)

```bash
# GPU 모드 (H100 등 기본 — import 시 자동 ON)
python -m backend.tests.e2e_pipeline_test --stage full

# CPU 강제 (5090 Blackwell 자동 폴백과 동일)
WEBAGENT_FORCE_CPU=1 python -m backend.tests.e2e_pipeline_test --stage full

# Blackwell(5090)에서 GPU 실험 (SEGV 위험)
WEBAGENT_ALLOW_BLACKWELL_GPU=1 WEBAGENT_USE_GPU=1 WEBAGENT_USE_GPU_RT=1 python -m backend.tests.e2e_pipeline_test --stage P1A

# 자세한 진단
nvidia-smi
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
python -c "import drjit; arr = drjit.cuda.Float(1.0); print(arr)"
```

## 개발 서버 실행

| 명령 | 용도 |
|------|------|
| `bash scripts/run_gpu_backend.sh` | **GPU 기본** 백엔드 (port 8800) |
| `uvicorn backend.app:app --reload --host 0.0.0.0 --port 8800` | GPU import 정책 자동 적용 |
| `WEBAGENT_FORCE_CPU=1 uvicorn ...` | CPU 강제 |
| `VITE_API_TARGET=http://localhost:8800 npm run dev` | 프록시 대상을 8800 백엔드로 맞춤 |
| `cd frontend && npm install` | Node 의존성 설치 |
| `cd frontend && npm run dev` | Vite 개발 서버 (포트 5173) |
| `cd frontend && npm run build` | 프로덕션 빌드 |
| `cd frontend && node ./node_modules/typescript/bin/tsc --noEmit` | TS 타입 체크만 |

## E2E 테스트 (Sionna 내장 씬)

```bash
cd "/home/dclserver78/twin_minji/260528 Web_Agent"   # 또는 해당 호스트의 twin_minji 경로
source .venv-webagent/bin/activate

# 풀 파이프라인
python -m backend.tests.e2e_pipeline_test --stage full

# 개별 stage 만 (P1A / P1B / P1F / P1G / P1H / P1I / P1P / derived / validate)
python -m backend.tests.e2e_pipeline_test --stage P1A
python -m backend.tests.e2e_pipeline_test --stage validate --session-dir sessions/_e2e_<TS>
```

## NPZ 검사 / 디버깅

```bash
# NPZ key/shape 인스펙트
python -c "
import numpy as np
d = np.load('sessions/<uuid>/P1A_RT_Results/Area1_7.5GHz_Rays_ALL_RXs.npz', allow_pickle=False)
for k in d.keys(): print(f'{k:30s} {d[k].shape} {d[k].dtype}')
"

# 단일 NPZ 스키마 검증
python -c "
from backend.npz_validator import validate_npz
from pathlib import Path
print(validate_npz(Path('sessions/<uuid>/P1F_Marginal_CCM_Results/...npz'), 'P1F').to_dict())
"

# 세션 결과 전체 검증
python -c "
from backend.npz_validator import validate_session
from pathlib import Path
for r in validate_session(Path('sessions/<uuid>')): print(r.to_dict())
"
```

## Git / 파일 관리

```bash
# 25* 원본 fork (수정 전 복사 필수)
cp "251218 E_MIMO_BM/P1A_RT_to_Rays_2509v6.py" \
   "260528 Web_Agent/backend/forked_25x/P1A_RT_to_Rays_2509v6_web.py"

# 큰 결과 파일은 git 추적 제외 (AGENTS.md 규칙 8)
echo "260528 Web_Agent/sessions/" >> .gitignore
echo "260528 Web_Agent/.venv-webagent/" >> .gitignore
echo "260528 Web_Agent/frontend/node_modules/" >> .gitignore
```

## 디스크 / 프로세스

| 명령 | 용도 |
|------|------|
| `du -sh sessions/*` | 세션별 디스크 사용량 |
| `ps -ef | grep uvicorn` | 백엔드 프로세스 확인 |
| `lsof -i :8000` | 8000 포트 사용 프로세스 |
| `kill <pid>` | 프로세스 종료 |

## 자주 만나는 문제

| 증상 | 해결 |
|------|------|
| `cuInit: UNKNOWN ERROR` | GPU 사용 불가 (드라이버 mismatch 등) — CPU 로도 동작하므로 무시 가능 |
| `Inter op parallelism cannot be modified after initialization` | 두 번째 stage import 시 발생. base_adapter 가 자동 no-op patch |
| `free(): invalid pointer` (종료 시) | Sionna/TF/Mitsuba 의 알려진 cleanup 이슈. 결과에 영향 없음 |
| 25* 모듈이 잘못된 폴더에 결과 저장 | `module.__file__` override 동작 확인 — base_adapter 가 세션 디렉토리로 redirect |
