# facade.py — 건물 외벽(수직면) RX 배치 모듈 (O2I)

## 목적
O2I(Outdoor-to-Indoor) penetration loss 관찰을 위해, 지면이 아니라 **건물 외벽**에
RX를 배치한다. `ground.py`(지면 격자 RX)와 대칭되는 RX 배치 방식 모듈이며,
Web '3. TX/RX' 페이지의 RX 배치 방식 중 `facade`(=O2I, 건물벽면)로 노출된다.

## 알고리즘
1. `scene.xml` 을 파싱해 각 `<shape>` 의 (PLY 경로, 재질(mat-itu_*), 건물 오브젝트명)을 얻는다.
2. 높이 레이어 `z = z_min, z_min+z_distance, ... (<= z_max)` 를 만든다. (전역 평면)
3. 각 건물 메시를 trimesh 로 로드하고, 각 z 평면과의 교선(`trimesh.intersections.mesh_plane`,
   `return_faces=True`)을 구한다.
4. 각 교선 세그먼트에서 원본 삼각형의 **face normal** 을 읽는다.
   - `|normal_z| > max_normal_z` 이면 수평면(지붕/지면)으로 보고 **제외**한다.
   - 수평 성분(nx, ny)을 정규화해 **벽 바깥 법선**으로 삼는다.
5. 세그먼트를 `spacing`(m) 간격으로 중점 샘플하고, 각 점을 바깥 법선으로
   `epsilon`(m) 만큼 **밖으로 이격**한다. (표면 self-occlusion 방지 → O2I 외벽 입사 필드)
6. 각 RX 에 메타(`host_object`, `material`, `normal`, `z_layer`, `rx_index`)를 기록한다.

* 재질 무관: 모든 건물 오브젝트의 수직면을 대상으로 한다(재질 필터 없음).
* 벽 바깥 법선(normal)을 저장하므로 이후 **입사각·재질 의존 penetration** 계산이 가능하다.

## 공개 API
- `compute_facade_rx(scene_xml, z_min, z_max, z_distance, spacing, epsilon, max_normal_z, max_points)`
  → `{"points": [[x,y,z],...], "meta": [...], "z_layers": [...], "count": int, "params": {...}}`
- `save_facade_metadata(out_dir, result, timestamp)`
  → `{"json": path, "npz": path}` (사이드카 저장, 덮어쓰기 금지: timestamp 사용)

## 파라미터 (Web 조정 가능)
| 이름 | 의미 | 기본값 |
|---|---|---|
| z_min | 최저 높이 [m] | 1.0 |
| z_max | 최대 높이 [m] | 30.0 |
| z_distance | 높이 간격 [m] (z_min, z_min+z_distance, ...) | 3.0 |
| facade_spacing | 컨투어를 따라 RX 간격 [m] | 5.0 |
| facade_epsilon | 벽 바깥 이격 거리 [m] | 0.3 |
| facade_max_normal_z | 수평면 배제 임계(|normal_z|) | 0.5 (UI 미노출) |

## 파이프라인 연동 (A안: 표준 NPZ 유지)
- `pipeline_executor._build_rx_placement_dict()` 의 `method=='facade'` 분기가 호출한다.
- 산출 좌표는 P1A 에 `{"method":"points","points":[...]}` 로 전달 → 표준 Ray NPZ 스키마 유지.
- 메타데이터는 `{session}/P1A_RT_Results/facade_rx_meta_<ts>.json|.npz` 로 저장.
  - npz 키: `positions(N,3)`, `normals(N,3)`, `z_layer(N)`, `host_object(N)`, `material(N)`, `rx_index(N)`.
  - RX 순서 = P1A `add_receivers` 순서 = points 순서 = `rx1..rxN`.
- 미리보기: `POST /api/sessions/{uuid}/rx_facade` (좌표만 반환, 메타 미저장).

## 한계 / 주의
- 벽 바깥 방향은 메시 face normal 이 바깥을 향한다는 가정에 의존한다(Blender export 기본).
- Sionna RT 회절 미포함/포함 여부와 무관하게, 이 모듈은 '외벽까지의 실외 필드' 절반만 만든다.
  실내 깊이(d_2D-in) 및 실내 손실은 별도 후처리에서 38.901 O2I 모델로 얹어야 한다.
- 인접/가려진 벽에도 RX 가 생성될 수 있다(RT 가 자연히 무경로 처리). 필요 시 후처리에서 필터.

## 실행 환경
- Python 3.10 / trimesh 4.12.2 / numpy 2.x (컨테이너 `.venv-webagent`)
- 서버: dclserver78 (H100)

## 자체 테스트
```bash
.venv-webagent/bin/python backend/rt_agent_park2/facade.py \
  "scene_library_radio/Jonggak_ground/scene.xml" \
  --z_min 1 --z_max 30 --z_distance 5 --spacing 8 --epsilon 0.3
```
