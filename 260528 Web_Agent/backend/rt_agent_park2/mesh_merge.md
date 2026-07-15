# mesh_merge.py

씬의 수천 개 PLY 를 '재질별 비인덱스 삼각형 배열(float32)'로 병합하여 3D 뷰어 로딩을 최적화.

## 왜?
기존 뷰어는 PLY 를 **파일당 1 HTTP 요청**으로 로드 → 1625개면 요청 폭주 →
dev 프록시/백엔드가 못 버텨 타임아웃·"메시 로딩 실패" 누적.
→ 서버에서 **재질별로 병합**(1625→보통 2~3파일)하여 요청 수를 급감시킴.

## 동작
- `build_merged_geometry(scene_dir)`:
  1. scene.xml 로 ply→material 매핑.
  2. 각 PLY 를 fast parser 로 파싱(binary_little_endian, vertex xyz + face list).
     미지원 포맷은 trimesh fallback.
  3. 재질별로 삼각형(비인덱스, positions only)을 concat → `meshes_merged/<mat>.f32` 저장.
  4. `meshes_merged/manifest.json` 캐시. scene.xml/meshes 가 manifest 보다 새로우면 재빌드.
- 성능: eda2dd32(1625 PLY, 45,196 삼각형) 병합 **0.12초**, 결과 ~1.6MB(3파일).

## API (app.py)
- `GET /api/sessions/{uuid}/scene/geometry` → manifest {materials:[{name,rel,n_tri,bytes}]} (필요 시 빌드)
- `GET /api/sessions/{uuid}/scene/merged/{name}.f32` → 병합 바이너리(little-endian f32, 삼각형 soup)

## 프론트 (SceneViewer.tsx Meshes)
- geometry manifest → 재질당 1 fetch(.f32) → BufferGeometry(position, computeVertexNormals).
- 병합 API 실패 시 **기존 파일당 로드 방식으로 자동 폴백**(비치명적).

## 포맷
- `<mat>.f32`: little-endian float32, 길이 = n_tri×9 (삼각형당 3정점×xyz).
