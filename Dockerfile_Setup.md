# Dockerfile / Dev Container 설정 문서 (초안)

`twin_minji_verPARK` 프로젝트(특히 `260528 Web_Agent` + `25*` 파이프라인)를
현재 호스트(dclserver78)의 `.venv-webagent` 의존성 그대로 컨테이너로 재현하기 위한
초안 설정 문서입니다.

## 배경

- 현재 Web_Agent 및 P1* 파이프라인은 **컨테이너 없이** 호스트 venv(`.venv-webagent`)에서 직접 실행 중.
- 같은 서버의 다른 프로젝트(MGA / CTPE / RTA)는 Docker 컨테이너로 상시 가동.
- 안정성/이식성/상시가동을 위해 본 프로젝트도 컨테이너화 필요 → 그 **첫 초안**.

## 생성/수정 파일

| 파일 | 역할 |
|------|------|
| `Dockerfile` | 컨테이너 이미지 정의 (CUDA12.5+cuDNN9 베이스, py3.10, Node20, pip 의존성) |
| `.devcontainer/devcontainer.json` | VS Code Dev Container 설정 (GPU 노출, 포트 포워딩, 워크스페이스 마운트) |
| `.dockerignore` | 빌드 컨텍스트 최소화 (대용량 데이터/venv/node_modules 제외) |
| `requirements-docker.txt` | `.venv-webagent` 의 `pip freeze` 고정본 (tensorflow → `[and-cuda]`) |

## 핵심 설계 결정

1. **베이스 이미지**: `nvidia/cuda:12.5.1-cudnn-runtime-ubuntu22.04`
   - TF 2.21 의 CUDA 12.x / cuDNN9 요구와 정렬, Ubuntu 22.04 = Python 3.10.
2. **CUDA 사용자 라이브러리**: 호스트 venv 는 시스템 CUDA 에 의존했으나,
   컨테이너에는 시스템 CUDA 가 없으므로 `tensorflow[and-cuda]==2.21.0` 로
   nvidia-*-cu12 휠(cuDNN/cuBLAS/cuSOLVER 등)을 함께 설치.
   `libcuda.so`(드라이버)는 런타임에 `--gpus all` 로 주입.
3. **코드/데이터 비-COPY**: `.npz/.blend/.tar.gz` 등 수~수십 GB 데이터를
   이미지에 굽지 않기 위해, 코드는 빌드 시 COPY 하지 않고 런타임에
   `/workspace` 로 **마운트**. 빌드 컨텍스트도 `.dockerignore` 로 최소화.
4. **포트**: 8800(백엔드) / 5173(Vite dev) / 8000(빌드 통합 운영 시).

## 사용법 (초안)

### A) VS Code Dev Container 로 열기
명령 팔레트 → "Dev Containers: Reopen in Container".
생성 후 `postCreateCommand` 가 TF/sionna/GPU 인식 여부를 출력.

### B) 수동 docker 빌드/실행
```bash
cd /home/dclserver78/twin_minji_verPARK
docker build -t twin-rt-webagent:latest .

docker run --gpus all --shm-size=32g -it \
  --name WebAgent_park_server78 \
  -p 8800:8800 -p 5173:5173 \
  -v "$PWD":/workspace \
  twin-rt-webagent:latest

# 컨테이너 안에서
cd "/workspace/260528 Web_Agent"
bash scripts/run_gpu_backend.sh        # 백엔드 :8800
# (별도 셸) cd frontend && VITE_API_TARGET=http://localhost:8800 npm run dev
```

## 검증 체크리스트 (빌드 후 확인 권장)

- [ ] `docker build` 성공
- [ ] 컨테이너 내부 `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"` → GPU 2장
- [ ] `python -c "import sionna.rt"` 정상 import
- [ ] 백엔드 `curl http://localhost:8800/api/metrics/catalog` → metric JSON
- [ ] `http://165.132.192.78:5173` 접속

## TODO / 다음 단계

- [ ] 실제 `docker build` 1회 수행하여 의존성 충돌/누락 점검 (특히 geopandas 계열, drjit cuda variant)
- [ ] 빌드 성공 후 프런트엔드 `npm run build` → 백엔드 StaticFiles 통합(MGA 방식)으로 단일 포트(8000) 운영 옵션 추가
- [ ] `--restart unless-stopped` 정책으로 상시 가동 (MGA 와 동일 패턴)
- [ ] H100 sm_90 에서 drjit `cuda_ad_mono_polarized` variant 정상 동작 확인

---

## PARK_2_RayTracing_Agent_v2 설정 승계 (2026-06-06)

향후 `PARK_2_RayTracing_Agent_v2` 의 RT 작업을 본 폴더로 병합할 예정이라,
PARK_2 의 RT/OptiX·H100 안정화 설정을 승계했다.

### 승계한 항목

| 구분 | 항목 | 이유 |
|------|------|------|
| 베이스 이미지 | `nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04` | OptiX/Mitsuba RT 백엔드 동작(devel = nvcc/헤더 포함) |
| ENV | `NVIDIA_DRIVER_CAPABILITIES=all` | **OptiX `libnvoptix.so.1` 주입 필수** |
| ENV | `NVIDIA_VISIBLE_DEVICES=all` | 전체 GPU 노출 |
| 시스템 패키지 | `libglib2.0-0`, `htop`, `net-tools` | open3d/geo 런타임 + 진단 |
| EXPOSE/forward | `7651` | PARK_2 RT 서비스 포트 |
| containerEnv | `OMP_NUM_THREADS=16` | OpenMP 스레드 제한 (import 전 필수) |
| containerEnv | `DRJIT_THREAD_COUNT=16` | Dr.Jit/Mitsuba 스레드 제한 (RT 핵심) |
| containerEnv | `CUDA_VISIBLE_DEVICES=0` | GPU 1장 격리 → 다중 GPU 충돌 방지 |
| containerEnv | `NCCL_P2P_DISABLE=1` | NVLink P2P 비활성 → 초기화 데드락 방지 |
| containerEnv | `HOME=/root` | 홈 디렉토리 고정 |

### 승계하지 않은 항목 (의도적)

- **Python 패키지 버전**: PARK_2 는 TF 2.20.0 / sionna 1.2.1 / mitsuba 3.7.1 이지만,
  Web_Agent 의 검증 핀(TF 2.21.0 / sionna 1.2.2 / mitsuba 3.8.0)을 유지.
  RT 코드는 상위 호환으로 가정. 실제 병합 시 충돌 발생하면 본 핀 우선으로 조정.
- **open3d / contextily / jupyterlab**: PARK_2 가 설치하지만 본 핀과 충돌 가능성이
  있어 Dockerfile 에 주석으로만 남김. RT 코드 병합 후 빌드 검증하며 활성화 권장.

### 주의: CUDA_VISIBLE_DEVICES=0

PARK_2 안정화 설정대로 **GPU 0번 1장만** 노출한다. Web_Agent 가 2장을 모두
쓰려면 `"0,1"` 로 변경(단, PARK_2 가 경고한 다중 GPU 데드락 위험을 감수).
