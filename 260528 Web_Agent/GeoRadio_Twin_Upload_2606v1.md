# Geo-Radio Env. Twin 업로드 (재질 부여된 씬) — 2606v1

## 실행 환경
- Python 3.10 / 서버 dclcom61 (.venv-webagent)
- 백엔드: FastAPI + trimesh>=4.0, numpy
- 프론트: React 18 + Vite 5 + @react-three/fiber + three 0.166

## 목적
씬 선택 단계(2. Scene)를 두 가지 Twin 모드로 분기한다.

| 모드 | 의미 | 입력 | 재질 |
|---|---|---|---|
| **Geo Env. Twin** (기본) | 기하만 있는 맵 (MGA 산출물) | 단일 `.obj`/`.ply`/`.zip` 또는 라이브러리 | 업로드 시 선택한 **기본재질 1종** 부여 |
| **Geo-Radio Env. Twin** | 재질이 이미 부여된 맵 (MAA 산출물) | `scene .xml` + `meshes/*.ply` 다중 업로드 (또는 `.zip`) | **XML 에 지정된 ITU 재질을 그대로 보존** |

RT(P1A 등)는 모드와 무관하게 `{session}/scene/scene.xml` 을 `load_scene(merge_shapes=False)`
로 읽으므로, 두 모드 모두 동일한 산출물(scene.xml + meshes/ + scene_info.json)을 만든다.

## 동작 원리
### 백엔드 (`backend/jobs/scene_builder.py`)
- `SceneInfo.material_assigned: bool` 필드 추가 (Geo-Radio 여부 표시).
- `_finalize_material_assigned_scene(out_dir, xml_path, search_dirs, source_type)` 공용 헬퍼:
  1. XML 이 참조하는 모든 `.ply` 를 basename 으로 `search_dirs` 에서 찾아 `out_dir/meshes/` 로 모음.
  2. XML 의 `filename` 참조를 전부 `meshes/<basename>` 으로 **재작성** → RT 가 PLY 를 확실히 찾음.
     (재질 `<ref>`/`<bsdf>` 등 다른 속성은 절대 건드리지 않음 = 사용자 재질 보존)
  3. BSDF id `mat-itu_*` 에서 재질 목록, 모든 PLY 에서 통합 AABB/정점·면 통계 계산.
- `build_scene_from_xml_zip` 을 위 헬퍼 기반으로 리팩터 (staging 디렉토리로 전개 후 정착).
- `build_scene_from_files(xml_path, ply_paths, out_dir)` 멀티파일 업로드용 빌더 신규 추가.

### 한글 파일명 NFC/NFD 대응
- Blender/리눅스 FS 가 한글 파일명을 **NFD(분해형)** 로 저장 → XML(**NFC 완성형**)과 불일치.
- `_finalize_material_assigned_scene` 에서 `unicodedata.normalize("NFC", name)` 로 색인/매칭하여 해결.
- (영문 파일명 씬은 영향 없음. 종각처럼 한글 건물명 씬에서 필수)

### 백엔드 API (`backend/app.py`)
- `POST /api/sessions/{uuid}/scene_bundle` (`files: list[UploadFile]`):
  업로드된 `.xml`/`.ply` 를 `uploads/bundle/` 에 저장 → `build_scene_from_files` 호출 →
  `scene_name` 메타를 XML stem 으로 갱신.

### 프론트 (`frontend/src/pages/ScenePage.tsx`, `lib/api.ts`)
- `twinMode: 'geo' | 'geo_radio'` 토글 버튼 추가.
- geo 모드: 기존 UI(라이브러리 + 단일 업로드 + 기본재질) 유지.
- geo_radio 모드: **라이브러리 불러오기 + 편집** + **직접 다중 파일 업로드**(`accept=".xml,.ply" multiple`) 병행.
- `apiClient.uploadSceneBundle(uuid, files[])` (multipart, timeout 0).
- `SceneInfo` 에 `source_type: 'xml_bundle'`, `material_assigned?` 추가, 요약 패널에 모드 배지 표시.

### Geo-Radio 번들 라이브러리 (`backend/app.py`, `scene_library_radio/`)
- Geo 라이브러리는 단일 `.ply`(+`.xml`/`.glb`) 단위지만, Geo-Radio 는 **번들(.xml + meshes/*.ply)**
  이라 별도 저장소 `scene_library_radio/` 사용. 각 엔트리 = `{name}/` 폴더(finalize 완료:
  `scene.xml` + `meshes/` + `scene_info.json`).
- 엔드포인트:
  - `GET    /api/scene_library_radio`                          목록(name, n_meshes, materials, n_faces, size)
  - `POST   /api/scene_library_radio` (files[], name)          번들 업로드 → finalize 후 엔트리 저장
  - `DELETE /api/scene_library_radio/{name}`                   엔트리 폴더 삭제
  - `POST   /api/sessions/{uuid}/scene_from_library_radio`     엔트리를 세션 scene/ 으로 copytree 적용
- 프론트 api.ts: `listSceneLibraryRadio / uploadSceneLibraryRadio / deleteSceneLibraryRadio / sceneFromLibraryRadio`.

## 사용법 (사용자)
1. 2. Scene 탭에서 **Geo-Radio Env. Twin** 선택.
2. MAA 산출 폴더에서 `scene .xml` 과 `meshes` 폴더 안의 `.ply` 들을 **모두 함께 선택**.
   (파일이 매우 많으면 폴더를 `.zip` 으로 압축 후 Geo 탭의 `.zip` 직접 업로드 사용)
3. 업로드 후 요약 패널의 `materials` 목록과 `mode: Geo-Radio` 배지 확인 → TX/RX → RT 실행.

## 검증
- 종각 멀티재질 번들(519 PLY, itu_concrete/ceiling_board/glass) → 정상, 한글 참조 정규화 OK.
- soup_small zip(루트레벨 ply, 단일재질) → 정상, ref `meshes/` 재작성 OK, staging 정리 OK.
- frontend `tsc --noEmit` 통과, `py_compile` 통과.

## 청크 스테이징 업로드 (대용량 Network Error 회피) — 추가
- 증상: `.xml` + 대용량 `.ply`(예 concrete 21MB)를 한 번의 multipart 로 보내면 중간 프록시/터널에서
  연결이 끊겨 axios `Network Error`.
- 해결: 입력을 **① scene .xml / ② meshes 폴더(webkitdirectory)** 로 분리하고, 각 파일을 **4MB 청크**로
  쪼개 스테이징 폴더에 순차 append → 재조립 후 빌드. 각 요청이 작아 프록시 한도와 무관하게 안정적.
- 엔드포인트 (`backend/app.py`):
  - `POST /api/sessions/{uuid}/scene_stage/reset`         스테이징 초기화
  - `POST /api/sessions/{uuid}/scene_stage/chunk`         (filename, chunk_index, data) 청크 append
  - `POST /api/sessions/{uuid}/scene_stage/build`         스테이징 → 이 세션 씬 빌드
  - `POST /api/sessions/{uuid}/scene_stage/save_library`  스테이징 → Geo-Radio 라이브러리에 저장
- 프론트(`ScenePage.tsx`): ① .xml 입력 + ② meshes 폴더 입력(webkitdirectory) 분리, 4MB 청크 업로드 +
  진행률 바, [이 세션에 적용] / [라이브러리에 저장] 버튼.
- 검증: 37MB PLY 10청크 재조립 md5 일치, 재조립 후 build 통과.
