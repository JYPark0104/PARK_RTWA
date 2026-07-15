# RTWA 운영 가이드 (안정화·점검)

RTWA(Ray Tracing Web Agent) 운영/복구 표준. venv·컨테이너·포트 혼선으로 인한 불안정을 없애기 위한 문서.
(2026-07-09 정리)

## 1. 아키텍처 한눈에
- **컨테이너**: `WebAgent_park_server78` (이미지 `twin-rt-webagent`), 내부 root 실행.
- **포트 매핑(호스트→컨테이너)**: `8900→8800`(backend API), `5273→5173`(vite frontend).
  - 브라우저는 **호스트:5273** 로 접속. 프론트는 `VITE_API_TARGET=http://localhost:8800` 로 백엔드에 프록시.
- **venv (중요)**:
  - ✅ **정상**: `/opt/venvs/webagent` — open3d·sionna·tensorflow·trimesh·mitsuba 등 전부 포함. **반드시 이걸 사용.**
  - ⛔ **함정**: `/workspace/260528 Web_Agent/.venv-webagent` — 부분/빈 venv. open3d 없음 → 지면격자/TX스냅 실패. **사용 금지.**
  - `/opt/venvs/torch` — 별개(torch용).
- **잡 실행**: RT 잡은 uvicorn 과 별도 **자식 프로세스**(`python -m backend.jobs.run_one_job <uuid>`)로 실행 → 웹 API 가 RT 부하에 묶이지 않음. 중단=자식 kill.

## 2. 시작/정지/상태 — 단일 진입점
컨테이너 안에서 `scripts/webagent_ctl.sh` 하나로 관리한다. (항상 올바른 venv 강제 + 프리플라이트 + 좀비정리 + 헬스체크)

```bash
docker exec -it WebAgent_park_server78 bash -lc \
  'cd "/workspace/260528 Web_Agent" && bash scripts/webagent_ctl.sh status'

# 백엔드만 재시작 (가장 자주 씀)
docker exec WebAgent_park_server78 bash -lc \
  'cd "/workspace/260528 Web_Agent" && bash scripts/webagent_ctl.sh restart backend'

# 전체(백엔드+프론트) 시작 / 정지
... webagent_ctl.sh start all
... webagent_ctl.sh stop all

# 필수 모듈만 점검 (venv 진단)
... webagent_ctl.sh preflight
```

수동 기동이 필요하면(스크립트 없이) 반드시 이 형태로:
```bash
cd "/workspace/260528 Web_Agent" && \
WEBAGENT_USE_GPU=1 WEBAGENT_USE_GPU_RT=1 TF_CPP_MIN_LOG_LEVEL=1 TF_FORCE_GPU_ALLOW_GROWTH=true \
nohup /opt/venvs/webagent/bin/uvicorn backend.app:app --host 0.0.0.0 --port 8800 --workers 1 \
> logs/uvicorn_$(date +%y%m%d_%H%M%S).log 2>&1 &
```
> `/opt/venvs/webagent/bin/uvicorn` 로 시작하는지 항상 확인. `.venv-webagent` 로 시작하면 안 됨.

## 3. 상태 진단 — /api/health
```bash
docker exec WebAgent_park_server78 bash -lc 'curl -s http://localhost:8800/api/health'
```
반환 필드:
- `ok`: 필수 모듈(open3d/sionna/trimesh) 전부 있으면 true. **false 면 잘못된 venv 등 문제.**
- `venv_ok`: 실행 python 이 `/opt/venvs/webagent` 인지.
- `python`, `python_version`, `modules`(모듈별 설치여부), `missing_critical`.
- `gpu`: 정책 사유, TF 인식 GPU 수, mitsuba variant, `nvidia_smi`(index, mem_used, mem_total, util).

또한 백엔드 기동 로그에 필수 모듈 누락 시 큰 경고가 찍힌다(startup dep 가드).

## 4. 자주 겪는 문제와 복구

### (a) "RX 밀도/간격을 바꿔도 개수가 안 변함", TX 스냅 실패
- 원인: 백엔드가 **잘못된 venv(.venv-webagent)** 로 떠서 open3d import 실패.
- 확인: `curl .../api/health` → `ok:false` / `missing_critical:["open3d",...]`.
- 복구: `webagent_ctl.sh restart backend` (올바른 venv 로 재기동).

### (b) GPU VRAM 이 꽉 참 / 특정 GPU 터짐
- 원인: **멈춘(hung) RT 잡**이 메모리를 붙잡고 있음(대시보드 GPU Load 0% 인데 VRAM 만 큼).
- 확인: `docker exec WebAgent_park_server78 bash -lc 'pgrep -af run_one_job'` + `nvidia-smi`.
- 복구: 웹 UI 에서 해당 잡 **취소**(권장), 또는 그 `run_one_job` PID kill.
  참고: TF 는 이미 memory-growth 모드(`TF_FORCE_GPU_ALLOW_GROWTH`)라 사전 독점은 아님.

### (c) nvidia-smi 도 GPU 를 못 잡음 (NVML Unknown Error)
- 컨테이너의 일시적 GPU 접근 상실. 복구: **컨테이너 재시작** 후 백엔드/프론트 재기동.
  ```bash
  docker restart WebAgent_park_server78
  # 이후 webagent_ctl.sh start all
  ```

### (d) 좀비/중복 uvicorn
- `webagent_ctl.sh` 의 start/restart 가 기존 uvicorn 을 정리 후 기동하므로 대부분 자동 해결.
- 수동: `pgrep -af 'uvicorn backend.app:app'` 로 확인 후 kill.

## 5. 성능 메모 (이미 반영됨)
- **메시 로딩**: 씬 메시(수천 PLY)를 재질별로 서버 병합(`/scene/geometry`+`/scene/merged`) → 3.TX/RX·9.시나리오에서 요청 폭주 없이 로딩.
- **O2I RX 개수**: 슬라이더 실시간 계산 제거, '벽면 RX 수 계산' 버튼으로만 계산(과부하 방지).
- **TF VRAM**: `backend/__init__.py` 가 import 시 `TF_FORCE_GPU_ALLOW_GROWTH=true` 설정 → 필요분만 할당.

## 6. 관련 스크립트
- `scripts/webagent_ctl.sh` — 통합 관리(권장 진입점).
- `scripts/run_gpu_backend.sh` — 단일 백엔드 포그라운드 기동(기본 venv=/opt/venvs/webagent 로 수정됨).
- `scripts/run_h100_backend.sh` — (구) H100 기동 스크립트. venv=.venv-h100/포트 8000 이라 현재 표준과 다름 → webagent_ctl 사용 권장.
