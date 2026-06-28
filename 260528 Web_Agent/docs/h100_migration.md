# H100 이주 가이드

> RTX 5090 (sm_120) 서버는 stable 패키지(TF 2.21 / drjit 1.3.1)가 sm_120 PTX를 아직 임베드하지 않아
> 안정적으로는 **CPU 모드**로만 동작합니다 (docs/gpu_diagnostic_report.md 참조).
> H100(sm_90, Hopper)에서는 모든 stable 패키지가 native 지원되어 **풀 GPU 가속**이 가능합니다.

## 예상 가속

| 단계 | 5090 CPU (현재) | H100 GPU (이주 후) | 비고 |
|---|---|---|---|
| **P1A (PathSolver)** | 분~수십 분 | **수십 초** | Mitsuba CUDA + sionna paths.cir GPU |
| **P1A Coverage Map** | 분~수십 분 | **수 초~수십 초** | RadioMapSolver는 RT 횟수가 많아 GPU 효과 가장 큼 |
| **P1F/P1G/P1I** | 분 단위 | **초 단위** | TF matmul 다수 → GPU 가속 즉시 적용 |
| **derived (PADP/PDP)** | 거의 동일 | 거의 동일 | numpy bound |

→ 실 도시 씬(1.06M verts, TX 3, 4×4 RX grid, BS 32×32 + coverage on) 기준
   **CPU 30~120분 → GPU 1~3분** 수준 (대략 30~50배).

## 사전 준비 (H100 서버)

| 항목 | 요구 | 확인 명령 |
|---|---|---|
| 운영체제 | Ubuntu 22.04+ 권장 | `lsb_release -a` |
| Python | 3.10 (sionna 1.2.2 호환) | `python3.10 --version` |
| GPU 드라이버 | 525+ (CUDA 12.x 지원) | `nvidia-smi` |
| 디스크 | venv ~6 GB, 씬 ~0.5 GB | `df -h ~` |
| 네트워크 | pypi 접근 가능 | `pip install --dry-run pip` |

Python 3.10이 없다면:
```bash
sudo apt update
sudo apt install -y python3.10 python3.10-venv python3.10-dev
```

## 1단계: 5090(dclcom61)에서 H100으로 코드 이주

```bash
cd ~/twin_minji/260528\ Web_Agent

H100_HOST=user@h100.example.com \
H100_PATH=/home/user/Web_Agent \
H100_SCENES=/home/user/scenes \
./scripts/migrate_to_h100.sh
```

옵션:
- `AUTO_SETUP=1` → H100에서 setup_h100.sh까지 ssh로 연달아 실행
- `SYNC_SESSIONS=1` → `sessions/` 도 함께 동기화 (기본 제외, 결과물 재생성 권장)

전송 대상:
- `backend/`, `frontend/`, `docs/`, `scripts/`, `README.md`, `requirements.txt`
- 제외: `.venv*`, `node_modules`, `__pycache__`, `dist`, `sessions/*` (기본)
- 추가: `${HOME}/twin_minji/260512 RT Result Data/Map_Mesh.obj` (있을 시)

## 2단계: H100측 환경 구성 (한 번만)

```bash
ssh user@h100.example.com
cd /home/user/Web_Agent
bash scripts/setup_h100.sh
```

`setup_h100.sh`가 자동 수행:
1. `.venv-h100/` 생성 (Python 3.10)
2. `tensorflow[and-cuda]==2.21.0`, `sionna==1.2.2`, `mitsuba==3.8.0`, `drjit==1.3.1`
3. 백엔드 의존성 (`backend/requirements.txt` 또는 fallback set)
4. GPU smoke test:
   - `tf.list_physical_devices('GPU')` → H100 감지
   - 4096² matmul (`<10ms` 기대)
   - sionna paths.cir 1샘플 → 정상 shape 출력

설치 시간: 10~15분 (대부분 tensorflow CUDA wheel ~2.5 GB 다운로드).

## 3단계: H100 백엔드 기동

```bash
cd /home/user/Web_Agent
bash scripts/run_h100_backend.sh             # 포트 8000
# 또는
PORT=8001 bash scripts/run_h100_backend.sh
```

스크립트 내부에서 강제 설정:
- `WEBAGENT_USE_GPU=1` → TF/sionna GPU
- `WEBAGENT_USE_GPU_RT=1` → mitsuba **cuda_ad_mono_polarized** variant (H100 sm_90 정상 지원)
- `TF_FORCE_GPU_ALLOW_GROWTH=true`

기동 후 로그에서 다음 라인 확인:
```
[webagent.runtime] GPU 활성화: 1장 — NVIDIA H100 ...
[app] GPU 활성화: 1장 — ...
```

## 4단계: 프론트엔드 접근

### (A) 5090에서 H100 백엔드로 SSH 터널 (가장 간단)
5090(dclcom61) 측에서:
```bash
cd ~/twin_minji/260528\ Web_Agent
H100_HOST=user@h100.example.com bash scripts/tunnel_from_5090.sh
```
그러면 `localhost:8000` → H100:8000. 5090에서 이미 띄운 프론트엔드(`:5173`)가 그대로 H100 백엔드를 호출함 (`vite.config.ts` 프록시가 `localhost:8000`을 가리킴).

### (B) H100에서 프론트엔드까지 띄움
```bash
ssh user@h100.example.com
cd /home/user/Web_Agent/frontend
npm install
npm run dev -- --host 0.0.0.0 --port 5173
```
브라우저에서 `http://h100-host:5173/`. (단, npm/Node가 H100에 있어야 함.)

## 검증 절차

이주 직후 다음을 차례로 점검:

1. **API 헬스체크**: `curl http://localhost:8000/api/metrics/catalog` → 19개 metric JSON
2. **세션 생성 + 작은 씬 + RSRP**:
   - `sessions/_e2e_*` 폴더가 38초 안에 P1A/P1B/derived까지 완료되는지
3. **Map_Mesh.obj 전체 E2E**:
   - 1.06M verts + TX 3 + RX 16 + BS 32×32 + coverage on
   - **목표: 90초 이내 완료 (CPU 모드는 30분+)**
   - `backend/logs/` 또는 WebSocket 로그에서 `[P1A] paths.cir` 시간 확인

## 트러블슈팅

| 증상 | 원인 | 해결 |
|---|---|---|
| `tf.list_physical_devices('GPU')` 비어 있음 | LD_LIBRARY_PATH 누락 | `.venv-h100/bin/activate` 안 했거나 `tensorflow[and-cuda]` 미설치 |
| `cuInit UNKNOWN ERROR` | 드라이버 vs CUDA toolkit 불일치 | `nvidia-smi` 드라이버 ≥ 525, CUDA 12.x |
| Mitsuba `cuda_ad_mono_polarized not available` | drjit-cuda 미설치 | `pip install drjit` (CPU/CUDA가 같은 패키지) |
| 포트 8000 충돌 | 다른 서비스 사용 중 | `PORT=8001 bash scripts/run_h100_backend.sh` |
| WebSocket 연결 실패 | tunneling/방화벽 | SSH 터널은 8000만 노출하므로 vite proxy가 `/ws/` 도 같이 처리 |

## 본 서버(5090, dclcom61)는 유지?

권장: **유지하고 개발 환경으로 사용** + H100은 실행 백엔드로.
- 5090: 프론트엔드 개발/HMR, 코드 수정, 작은 E2E 테스트 (.venv-webagent / .venv-nightly)
- H100: 실 도시 씬 풀 파이프라인 GPU 실행
- 동기화: `scripts/migrate_to_h100.sh` 재실행 (rsync delta만 전송됨, 코드 변경 시 보통 <1초)

## 추후 개선

- 5090 풀 GPU는 `drjit 1.5+` (PTX 8.7) 또는 `master` 빌드 출시 후 가능. 그 때 `.venv-nightly`로도 RT가 GPU로 떨어짐.
- 현재로서는 H100 이주가 가장 안정적이고 빠른 선택.
