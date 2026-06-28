# queue_store.py — 멀티유저 잡 큐 영속화

## 목적
단일 사용자 가정이던 Web RT Agent를 **여러 연구원이 동시 접속**해도 안전하게 쓰도록,
세션 중심 FIFO 큐를 디스크에 영속화하고 재시작에도 복구한다.

## 설계 결정 (사용자 확정)
- 동시 실행 1개 (기존 `JobRunner` 단일 워커 유지 — GPU 데드락 회피).
- user = 이름표(인증 아님), 권한 제한 없음 (신뢰된 연구실 내부망 전제).
- 세션 중심 상태: 세션이 `queued → processing → done` 으로 이동, 재실행 시 갱신.
- 재시작 복구: `processing` → `interrupted`, `queued` → 순서 유지하여 재등록.
- 큐 제어: queued 취소 / processing 협력적 중단 / **reorder 없음**(공정성) /
  queued 세션의 config 수정·삭제 허용(순서 불변).
- 대시보드 폴링 1초.

## 영속화 파일 (sessions/ 루트)
- `_users.json`  : `{"users":[{"id","name"}]}`
- `_queue.json`  : `{"order":[session_uuid,...]}`  (FIFO)
- 세션별 `session_meta.json` : status/progress/타임스탬프/user (session_labeler 확장)
- 세션별 `session_config.json` : 실행 config 스냅샷 (Run 시 저장, 워커가 실행 시점에 로드
  → queued 동안 수정한 설정이 그대로 반영)

## 주요 API
- Users: `list_users / add_user / rename_user / delete_user`
- Queue: `enqueue(uuid, job_id, ...) / remove_from_order / dashboard(live_progress) / recover`
- 상태: `mark_processing / update_progress / mark_done / mark_failed / mark_cancelled / mark_interrupted`
- Config: `save_session_config / load_session_config`
- 싱글톤: `make_store()` (pipeline_executor.SESSIONS_ROOT 기준)

## 연동
- `app.py` : `/api/users*`, `/api/queue`, `/api/sessions/{uuid}/cancel|config`,
  `post_job`(config 저장+메타 갱신+enqueue), startup 복구.
- `pipeline_executor._pipeline_worker` : 상태 훅 + 협력적 중단(emit 시점 취소 체크),
  실행 시점에 `session_config.json` 우선 로드.

## 실행 환경
Python 3.10 / FastAPI / 컨테이너 venv-webagent (WebAgent_park_server78).
