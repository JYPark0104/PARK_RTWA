# 260528 RT Web Agent — Sionna RT × Weichselberger BM Web GUI

Sionna RT 기반 Ray Tracing → Marginal CCM → Weichselberger 채널 통계 → Beam Management
파이프라인 (25* 폴더) 을 하나의 웹 UI로 조작/실행/시각화하는 단일 사용자용 도구.

## 기능 요약

- **씬 업로드**: OBJ / PLY / `(Mitsuba XML + PLY)` zip → 자동으로 Sionna 호환 XML 변환
- **3D 뷰어**: react-three-fiber, OrbitControls, AABB 박스, surface-snap 클릭 (TX/RX 배치)
- **TX/RX 배치**: 클릭 / Grid / Radial / Street / Explicit 자동 생성 (4종 + 클릭) — Coverage Map 측정 영역 오버레이 지원
- **RT Config**: 주파수, 안테나 rows×cols (BS / UE), max_depth, seed, PathSolver flag (LoS / Specular / Diffuse / Refraction / Synthetic) + Coverage Map 옵션
- **결과 metric 선택**: 18 종 (+ Coverage Map = 19 종) 체크박스, 의존성 자동 위상정렬, 7 종 프리셋
- **잡 실행**: FastAPI + asyncio + ThreadPoolExecutor 비동기 실행, WebSocket 실시간 진행률/로그
- **결과 뷰**: NPZ 트리 + 미리보기 (PNG/CSV/JSON) + Coverage Map / RSRP 자동 hero 영상, 일괄 zip 다운로드

## 디렉토리

```
260528 Web_Agent/
├── backend/                 # FastAPI 서버
│   ├── app.py               # API 엔드포인트
│   ├── schemas.py           # Pydantic 스키마
│   ├── ws_manager.py        # WebSocket 브로커
│   ├── metric_catalog.py    # 19 metric → 25* stage 의존성 그래프
│   ├── npz_validator.py     # 251218 표준 NPZ 스키마 검증기
│   ├── adapters/            # 25* P1A ~ P1Q 호출 어댑터
│   ├── derived/             # PDP / PADP / RSRP / Ray Stats 도출 (25* 에 없는 metric)
│   ├── forked_25x/          # 25* 모듈의 Web Agent 용 fork (원본 무수정 원칙)
│   ├── jobs/                # scene_builder, antenna_resolver, session_labeler, job_runner
│   └── tests/e2e_pipeline_test.py
│
├── frontend/                # React + Vite + TS + Tailwind + r3f
│   └── src/pages/           # 7 페이지 (Sessions / Scene / Devices / RT / Metrics / Run / Results)
│
├── docs/                    # API reference, pipeline dependency, terminal commands
├── logs/                    # 작업 로그 (.log)
└── sessions/                # (런타임) 세션별 결과 디렉토리
```

## 환경 (호스트 `dclcom61`)

- Python 3.10, `260528 Web_Agent/.venv-webagent/`
- Sionna 1.2.2 + TensorFlow 2.21.0 (CPU/GPU)
- Mitsuba 3, Trimesh + Embreex, GeoPandas, FastAPI, Uvicorn, Websockets
- Node 20 + Vite + React 18 + Three.js + react-three-fiber + Zustand + TailwindCSS

상세 패키지 목록 → `backend/requirements.txt` / `backend/requirements.md`
커맨드 모음        → `docs/terminal_commands.md`

## 빠른 실행

```bash
# 백엔드
cd "/home/dclcom61/twin_minji/260528 Web_Agent"
source .venv-webagent/bin/activate
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload

# 프런트엔드 (별도 터미널)
cd "/home/dclcom61/twin_minji/260528 Web_Agent/frontend"
npm install        # 최초 1회
npm run dev        # http://localhost:5173
```

브라우저 `http://<dclcom61>:5173` 접속 → Sessions → Scene → Devices → RT → Metrics → Run → Results 순서로 진행.

## E2E 검증

```bash
cd "/home/dclcom61/twin_minji/260528 Web_Agent"
source .venv-webagent/bin/activate
python -m backend.tests.e2e_pipeline_test --stage full
```

Sionna 내장 `simple_street_canyon` 씬에서 P1A → P1B → P1F → P1G → P1H → P1I → derived 풀 파이프라인
실행 후 NPZ 가 `251218 E_MIMO_BM` 표준 스키마와 일치하는지 자동 검증한다 (4/4 통과 기준).

## 설계 원칙

1. **25* 원본 무수정** — 필요시 `backend/forked_25x/` 에 복사 후 수정 (P1A fork 만 존재)
2. **NPZ 표준 통일** — 모든 P1* 결과는 `251218 E_MIMO_BM` 포맷 단일
3. **세션 격리** — 모든 결과/씬은 `sessions/<uuid>/` 안에서만 생성
4. **GPU 활용** — Sionna/TF 가 GPU 를 사용하도록 호스트 환경 그대로 활용

## 작업 규칙

`AGENTS.md` 의 규칙 (네이밍 / 로그 / 파라미터 변경 사유 주석 등) 을 모두 준수.
