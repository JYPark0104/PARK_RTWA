# API Reference — RT Web Agent FastAPI

Base URL: `http://<host>:8000`  
Frontend Vite proxy: `/api/*` → `http://localhost:8000/api/*`, `/ws/*` → `ws://localhost:8000/ws/*`

## 헬스 / 메타

| Method | Path | 설명 |
| ------ | ---- | ---- |
| GET    | `/api/health`            | 서버 status / 버전 |
| GET    | `/api/metrics/catalog`   | 19개 metric 카탈로그 (`MetricSpec[]`) |
| GET    | `/api/metrics/presets`   | 7종 프리셋 (id → label / description / metrics 목록) |
| POST   | `/api/metrics/resolve`   | metric ID 리스트 → 의존성 위상정렬 결과 |

### POST /api/metrics/resolve

요청
```json
{ "metrics": ["padp", "marginal_ccm", "su_capacity"] }
```
응답
```json
{
  "stages_ordered": ["P1A", "P1B", "P1F", "P1G", "P1H", "P1I", "P1J"],
  "derived": ["padp"],
  "coverage_map": false,
  "needs_rx": true
}
```

## 세션

| Method | Path | 설명 |
| ------ | ---- | ---- |
| POST   | `/api/sessions`                   | 새 세션 생성 (라벨 자동 + 편집 가능) |
| GET    | `/api/sessions`                   | 세션 리스트 |
| GET    | `/api/sessions/{uuid}`            | 단일 세션 메타 |
| PATCH  | `/api/sessions/{uuid}/label`      | 라벨 수동 변경 |

### POST /api/sessions

요청
```json
{
  "scene_name": "Jonggak",
  "bs_rows": 32, "bs_cols": 32,
  "ue_rows": 4,  "ue_cols": 4,
  "metrics": ["coverage_map", "rsrp"]
}
```
응답: `SessionMeta` (uuid, label, created_at_kst, ...)

## 씬

| Method | Path | 설명 |
| ------ | ---- | ---- |
| POST   | `/api/sessions/{uuid}/scene`              | OBJ / PLY / zip 업로드 → Mitsuba XML 자동 변환 |
| GET    | `/api/sessions/{uuid}/scene_info`         | AABB / center / 정점·면 수 |
| GET    | `/api/sessions/{uuid}/scene/mesh/{name}`  | PLY 파일 다운로드 (3D 뷰어 로드용) |

`POST /scene` 은 multipart/form-data 로 `file` + 옵션 `material` (기본 `itu_concrete`) 전송.

## 잡 (RT 파이프라인 실행)

| Method | Path | 설명 |
| ------ | ---- | ---- |
| POST   | `/api/sessions/{uuid}/job`     | 잡 제출 (TX/RX/Antenna/RT/Metrics) |
| GET    | `/api/jobs/{job_id}`           | 잡 상태 조회 |
| POST   | `/api/jobs/{job_id}/cancel`    | 잡 취소 |
| WS     | `/ws/jobs/{job_id}`            | 실시간 progress / log 이벤트 |

### POST /api/sessions/{uuid}/job — payload

```json
{
  "session_uuid": "<uuid>",
  "tx_list": [{"position":[0,0,25],"orientation":[0,0,0],"name":"tx1"}],
  "rx_grid": {
    "method": "grid",
    "x_start": -100, "x_stop": 100, "x_num": 20,
    "y_start": -100, "y_stop": 100, "y_num": 20,
    "z_values": [1.5]
  },
  "rx_clicks": null,
  "antenna": { "mode": "simple", "simple": {"bs_rows":32,"bs_cols":32,"ue_rows":4,"ue_cols":4} },
  "rt": {
    "mode": "simple",
    "frequency_ghz": 7.5,
    "max_depth": 5, "seed": 41,
    "num_interesting_paths": 20, "max_rays_per_pair": 400,
    "pathsolver_los": true, "pathsolver_specular_reflection": true,
    "pathsolver_diffuse_reflection": true, "pathsolver_refraction": true,
    "pathsolver_synthetic_array": false,
    "itu_scattering_coeff": 0.2, "itu_xpd_coeff": 0.5,
    "coverage_map": {"enabled": false, "cell_size_x":1.0, "cell_size_y":1.0, "height_m":1.5, "samples_per_tx":1e8, "max_depth":5, "specular_reflection":true, "diffuse_reflection":true, "refraction":true }
  },
  "metrics": { "metrics": ["rsrp", "marginal_ccm", "coverage_map"] }
}
```

응답: `JobStatus`  
```json
{
  "job_id": "...", "session_uuid": "...",
  "state": "pending", "progress": 0.0,
  "current_stage": "", "stages_done": [],
  "started_at": null, "finished_at": null, "error": null, "output_paths": {}
}
```

### WebSocket `/ws/jobs/{job_id}` 이벤트

```json
{ "kind": "stage_start", "stage": "P1A", "timestamp": "2026-05-28T15:30:00.123Z", "message": "..." }
{ "kind": "stage_progress", "stage": "P1A", "progress": 0.42, "message": "RX 100/240" }
{ "kind": "stage_end", "stage": "P1A", "timestamp": "...", "duration_s": 12.3 }
{ "kind": "log", "level": "info", "message": "..." }
{ "kind": "done", "timestamp": "...", "output_paths": {...} }
{ "kind": "error", "timestamp": "...", "message": "..." }
```

## 결과 파일

| Method | Path | 설명 |
| ------ | ---- | ---- |
| GET    | `/api/sessions/{uuid}/files`             | 세션 안의 모든 결과 파일 목록 `[{path,size,kind}]` |
| GET    | `/api/sessions/{uuid}/files/{path}`      | 임의 파일 다운로드 (npz, png, csv, json, ply ...) |
| GET    | `/api/sessions/{uuid}/zip`               | 세션 전체 zip 다운로드 |

각 파일의 `kind` 는 확장자 기반 (`npz`, `png`, `csv`, `json`, `ply`, `xml`, ...). 프런트엔드는 `kind` 로 미리보기 컴포넌트를 선택.

## 에러

- `400` : Pydantic validation 실패, metric 조합 모순, 씬/TX 누락
- `404` : 세션/잡 미존재
- `409` : 이미 진행 중인 잡 존재 (단일 사용자 모드)
- `500` : 25* 모듈 import 실패, NPZ 검증 실패, 기타 런타임 예외

## 단일 사용자 모드 제약

본 도구는 호스트 머신 한 명의 연구자가 사용하도록 설계됨:
- 동시 잡 1 개 (asyncio.Queue + ThreadPoolExecutor max_workers=1)
- WebSocket 브로커는 단일 채널 (멀티 클라이언트는 동일 잡 구독 가능)
- 인증 없음, CORS 는 dev 모드에서 wide-open
