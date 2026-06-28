# scene_builder.py

업로드된 `.obj`/`.ply` 또는 Mitsuba XML+PLY zip을 Sionna RT가 직접 로드 가능한 Mitsuba 2.1.0 XML 씬으로 변환.

## 입력

| 확장자 | 처리 |
|--------|------|
| `.obj` | trimesh로 로드 → PLY로 export → Jonggak.xml 스타일 XML 자동 래핑, 단일 ITU 재질 매핑 |
| `.ply` | 그대로 사용 → XML 래퍼만 생성 |
| `.zip` | 압축 해제 → 내부 XML/PLY 무결성 검증 → 그대로 사용 |

## 출력

```
{session_dir}/scene/
├── scene.xml             # Sionna load_scene()에 전달
├── meshes/*.ply
└── scene_info.json       # AABB, 단위(meters), 메시 통계
```

## 사용 예

```python
from backend.jobs.scene_builder import build_scene
info = build_scene(
    src_path="upload.obj",
    out_dir="sessions/abcd1234/scene",
    material="itu_concrete",
)
print(info.aabb_min, info.aabb_max)  # 자동 그리드 RX 배치 영역 추천에 사용
```

## CLI

```
python -m backend.jobs.scene_builder upload.obj sessions/abcd1234/scene --material itu_concrete
```

## 함수

| 이름 | 설명 |
|------|------|
| `build_scene(src, out_dir, material)` | 확장자 자동 분기 |
| `build_scene_from_single_mesh(...)` | 단일 obj/ply 처리 |
| `build_scene_from_xml_zip(...)` | xml+ply zip 처리 + 무결성 검증 |

## SceneInfo 데이터클래스

| 필드 | 의미 |
|------|------|
| `source_type` | `obj` / `ply` / `xml_zip` |
| `mesh_files` | scene.xml 기준 상대 경로 리스트 |
| `materials` | 사용된 ITU 재질 (`itu_concrete` 등) |
| `aabb_min/max` | GCS 단위 (미터) |
| `center`, `size` | Coverage Map 기본 영역 추천 등에 사용 |
| `n_vertices`, `n_faces` | 통계 |

## ITU 재질

- `itu_concrete` (기본)
- `itu_ceiling_board`
- `itu_glass`

Jonggak.xml과 동일한 reflectance 값 사용. Sionna `Scene.tune_materials()`로 scattering/xpd coefficient는 어댑터에서 변경 가능.

## 의존성

- `trimesh>=4.0` (embreex 자동 포함)
- numpy

## 안전

- zip 입력은 zipslip 방지 (resolve 후 out_dir 외부 거부)
- XML의 `filename=...` 참조 PLY 존재 검증
