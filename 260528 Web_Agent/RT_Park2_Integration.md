# RT (PARK_2) 통합 — Web Agent RX 격자 / Ray Tracing 고도화

PARK_2_RayTracing_Agent_v2 의 RX 지면 배치 / TX 지면 스냅 / 고급 RT 옵션 / Review
시각화를 260528 Web Agent 에 이식한 작업 기록.

## 사용자 결정 사항
- Q1=A안: **표준 251218 NPZ 유지** (metric 파이프라인 무수정). 새 RX/TX 좌표는
  백엔드에서 미리 계산해 P1A 에 explicit 좌표로 전달.
- Q2=(a): 컨테이너에 **open3d** 추가 (PARK_2 레이캐스팅 코드 거의 그대로 이식).
- Q3: **scene_library** 에 .ply + .xml 쌍 저장.
- Q4: TX = 클릭 (x,y) → 지면 레이캐스팅 → 지면 + offset(기본 2m, **조절 가능**) 위 빨간 TX,
  클릭 원점은 연한 마커. explicit/radial/street 는 무관(그대로).
- Q5: 백엔드에 **사본 이식** (`backend/rt_agent_park2/`). 원본은 워크스페이스 밖 → 읽기전용 참고.
- Q6: 기존 Coverage Map 은 그대로 두고 **별도로** RSRP/LoS Review 추가.

## 구현 매핑 (단계별)

| 영역 | 파일 | 내용 |
|------|------|------|
| 환경 | `requirements-docker.txt`, 컨테이너 | open3d==0.19.0 추가/설치 |
| Scene | `scene_library/` + `app.py` `/api/scene_library`(목록·적용) | .ply+.xml 쌍 라이브러리 |
| RX/TX | `backend/rt_agent_park2/ground.py` | 지면 격자(`compute_ground_rx_grid`), TX 스냅(`snap_tx_to_ground`) |
| RX/TX API | `app.py` `/tx_snap`, `/rx_ground_grid` | 프런트 클릭/미리보기용 |
| 스키마 | `schemas.py` | `RXGridConfig.ground_grid`, `RTConfig.tx_ground_offset_m` + 고급 RT 옵션 |
| RT 실행 | `jobs/pipeline_executor.py` | ground_grid→explicit 변환, 고급 RT override→P1A |
| P1A | `forked_25x/P1A_..._web.py` | diffraction/num_samples/max_num_paths 등 override 수용 |
| Review | `derived/derive_los_map.py` (신규) + `pipeline_executor` Review 블록 | all-TX RSRP + LoS/NLoS 맵 생성 |
| 프런트 | `pages/ScenePage/DevicesPage/RTConfigPage/JobRunPage`, `components/SceneViewer`, `pages/ResultsPage` | scene_library 선택, ground_grid, TX 듀얼마커, 고급옵션, 진행바, Review 섹션 |

## LoS/NLoS 분류 규칙 (derive_los_map)
- 표준 NPZ `los_nlos_flag` (ray 단위, LoS=1/NLoS=0).
- RX 분류: 유효 ray(power>0) 중 flag==1 존재 → LoS / 모두 0 → NLoS / 유효 ray 없음 → Dead.
- 출력: `Review_Results/LOS_NLOS_review_*.png|csv` (LoS 초록·NLoS 주황·Dead 회색, TX 빨간 ▲).

## 검증 (2026-06-06 재개 시점)
- 백엔드 import OK (derive_los_map, pipeline_executor, rt_agent_park2).
- frontend `tsc --noEmit` 통과.
- 컨테이너 백엔드 재기동 후 `/api/metrics/catalog` HTTP 200.

## TODO / 미검증
- 실제 GHM 씬 **풀 RT 잡 1회** E2E 실행으로 ground_grid RX + TX 스냅 + 고급옵션 +
  Review PNG 까지 끝까지 도는지 확인 (브라우저 인터랙티브 권장, 수분 소요).

## 완료 보강 (2026-06-06 마무리)
- (4) 고급옵션 반영 완성: `scattering_pattern`(lambertian/directive/backscattering)을
  P1A fork `tune_materials` 에서 실제 RadioMaterial 에 반영(`_apply_scattering_pattern`).
  전기적 특성(permittivity/conductivity/thickness)은 P1A 의 ITU 재질 모델(주파수 의존)이
  더 정확하므로 **의도적으로 덮어쓰지 않음**(UI 노출은 참고값).
- 컴포넌트 단위 검증 통과(ground grid/TX snap/RSRP map/LoS map/import/services). 상세는 .log 참조.
