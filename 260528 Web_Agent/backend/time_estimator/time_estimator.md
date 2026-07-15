# time_estimator — 시간 예측 담당 부서

Web Agent(백엔드) 산하의 **실행 시간 예측 전담 모듈**. 다른 모듈은 이 패키지의
공개 API 4개(`record_run`, `estimate`, `compute_eta`, `StageTimer`)만 알면 된다.

## 목적 (사용자 확정 사항 반영)

- **(A) 실행 전 예측** — Run(잡 제출) 창에서 "예상 소요 시간 약 N분 (±M분)"을 미리 표시.
- **(B) 실행 중 ETA** — Sessions / RT Results 의 Processing 카드에 "⏳ 남은 시간"을 표시.

확정된 설계 결정:
- **Q1** 머신 지문(`machine_id + gpu_mode + gpu_name`)을 예측의 필수 매칭 키로 사용.
- **Q2/Q7** batch / intg RT 엔진 우선 지원.
- **Q3** 안테나 수를 RT cost basis 에 **곱으로 포함**(지배적 영향).
- **Q4** 이력은 앱 레벨 고정 경로에 JSONL 로 누적(세션 삭제/Reset 에 안 지워짐).
- **Q5** 콜드 스타트(기록 0건)면 조용히 "예측 데이터 수집 중" 표시.
- **Q6** 표시 형식: `예상 소요 시간 약 2시간 49분 (±10분)` + `단계별 · RX N개 기준 · 최소 k회 데이터 · 누적 m회 기록 (나이브 추정치)`.
- **Q8** ETA = 경과 ÷ 진행률 × (1−진행률), 진행률 2% 미만이면 "준비 중".
- **Q9** 기존 1초 폴링(`/api/queue`) 응답에 `eta_sec` 필드만 추가.

## 파일 구성

| 파일 | 역할 |
|------|------|
| `__init__.py`   | 공개 API 4개 재노출 |
| `machine.py`    | 실행 머신/하드웨어 지문 (호스트명 + GPU/CPU 모드 + GPU명) |
| `cost_basis.py` | 단계별 cost basis 정의 + config 매칭 키 + num_rx 해석 |
| `run_history.py`| 이력 JSONL append/load (best-effort, reset-safe) |
| `estimator.py`  | (A) 실행 전 예측 — 단계별 1D 회귀(폴백 사다리) → mean±std |
| `eta.py`        | (B) 실행 중 ETA 계산 |
| `recorder.py`   | `StageTimer`(관찰) + `record_run`(완료 시 기록) |
| `store/run_history.jsonl` | 누적 이력 데이터 (자동 생성) |

## cost basis (단계별 지배항 1개)

| 단계 | basis |
|------|-------|
| Scene   | `num_rx` (RX 배치 레이캐스팅) |
| RT      | `num_tx × num_rx × num_ant_pairs`  ← 안테나 수 포함(Q3) |
| Intg    | `num_tx × num_rx` |
| Output* | `num_tx × num_rx` |
| (미지)  | `num_rx` 폴백 |

`num_ant_pairs = (BS_rows·BS_cols) × (UE_rows·UE_cols)`.
`max_depth`, `num_samples` 는 basis 에 넣지 않고 **config 매칭(와일드카드)** 으로 분리.

## 회귀 폴백 사다리 (단계별)

1. **affine** `d ≈ a·x + b` (표본 2+ & x 분산 있음, a>0 & 예측>0 일 때만)
2. **비례** `d ≈ rate·x` (rate = mean(d/x))
3. **평균** `d ≈ mean(d)`
4. 표본 0 → 그 단계는 건너뜀

총 예측 = 단계별 mean 합, 불확실성 = 단계별 std 제곱합의 제곱근.

## 통합 지점 (다른 모듈 수정)

- `backend/jobs/pipeline_executor.py::_pipeline_worker`
  → `StageTimer` 로 stage 경계/규모를 관찰하고, 성공 시 `record_run` 호출(best-effort).
- `backend/jobs/queue_store.py::_meta_brief`
  → processing 세션에 `eta_sec = compute_eta(elapsed, progress)` 추가.
- `backend/app.py`
  → `POST /api/sessions/{uuid}/estimate` 엔드포인트 추가 (A 예측 조회).
- `frontend/src/lib/api.ts` → `estimateJob()` + `TimeEstimate` 타입 + `QueueItem.eta_sec`.
- `frontend/src/pages/JobRunPage.tsx` → 예상 소요 시간 카드(A).
- `frontend/src/components/QueueDashboard.tsx` → Processing 카드에 "남은 시간"(B).

## 원칙

- **기록은 실행을 절대 방해하지 않는다** — 모든 예외를 삼킨다(best-effort).
- **하드웨어가 다르면 비교하지 않는다** — machine_key 불일치 기록은 제외.
- **데이터가 적어도 안 깨진다** — 폴백 사다리로 우아하게 degrade, 콜드 스타트는 조용히 미표시.

## 실행 환경

- Python 3.10 (컨테이너 `.venv-webagent`)
- 표준 라이브러리만 사용 (numpy/pandas 불요)
- 서버: dclcom45 / dclcom55 / H100 등 — machine 지문으로 자동 구분
