# backend/requirements.txt — 의존성 설명

## 호스트 환경 (2026-05-28 dclcom61 확인 결과)

| 항목 | 상태 |
|------|------|
| Python | 3.10.12 (시스템) |
| pip | **미설치** (`apt install python3-pip` 또는 user-mode 설치 필요) |
| GPU | NVIDIA RTX 5090 × 2 (32GB VRAM) |
| CUDA | 13.0 (Driver 580.95.05) |
| Node | v20.18.2 (Cursor 임베디드 — npm 없음) |
| Docker | 사용 가능 (oai_sionna 이미지 존재) |

## 패키지 그룹

### Sionna RT 스택
- `tensorflow>=2.18.0`: RTX 5090 Blackwell SM 12.0 지원
- `sionna>=1.0.2`: Ray Tracing + RadioMapSolver
- `mitsuba>=3.5`: Sionna 의존, 씬 로드용
- `drjit>=0.4`: Mitsuba JIT 백엔드

### 과학 계산
- `numpy 1.24.3+`: 25\* 호환 버전
- `scipy 1.14.1+`: EVD, 클러스터링
- `matplotlib 3.7+`: 시각화
- `pandas 2.0+`: 결과 CSV 처리

### 메시 / 씬
- `trimesh 4.0+`: OBJ/PLY 로드 변환
- `pyembree`: 빠른 ray-mesh BVH (없으면 trimesh native 사용)

### FastAPI 서버
- `fastapi 0.110+`, `uvicorn[standard] 0.27+`
- `python-multipart`: 파일 업로드
- `websockets 12+`: 진행률 푸시
- `pydantic 2.5+`: 스키마
- `aiofiles 23+`: 비동기 파일 I/O

## 설치 절차 (사용자 액션 필요)

### 1. pip 설치 (둘 중 하나)

#### A. 시스템 패키지 (권장, sudo 필요)
```bash
sudo apt update && sudo apt install -y python3-pip python3-venv python3.10-venv
```

#### B. user-mode get-pip.py (sudo 없이)
```bash
curl https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
python3 /tmp/get-pip.py --user
export PATH="$HOME/.local/bin:$PATH"  # ~/.bashrc에 추가 권장
```

### 2. 가상환경 + 의존성

```bash
cd "260528 Web_Agent/backend"
python3.10 -m venv .venv-webagent
source .venv-webagent/bin/activate
pip install -U pip wheel
pip install -r requirements.txt
```

### 3. Sionna smoke test

```bash
python -c "
import sionna
print('Sionna:', sionna.__version__)
from sionna.rt import load_scene, RadioMapSolver, PathSolver
print('RT modules loaded OK')

import tensorflow as tf
print('TF:', tf.__version__)
print('GPUs:', [g.name for g in tf.config.list_physical_devices('GPU')])
"
```

## 호환성 노트

- **CUDA 13 + RTX 5090**: TF 2.17은 미지원. TF 2.18+ 또는 nightly 필요.
- **Sionna 1.0.2 + RTX 5090**: 1.0.2 자체는 호환되지만 mitsuba/drjit 버전이 SM 12.0 지원 필수.
- **충돌 시 옵션**:
  - 기존 docker 컨테이너 (`oai_sionna_luuuuuu-oai_sionna_proxy`) 활용
  - `tensorflow[and-cuda]==2.18.0`로 설치 (cuda toolkit 포함)
  - nightly: `pip install tf-nightly`
