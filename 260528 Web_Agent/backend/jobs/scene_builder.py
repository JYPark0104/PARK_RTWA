"""scene_builder.py — 업로드된 .obj/.ply/.zip을 Sionna가 로드 가능한 Mitsuba XML 씬으로 변환.

실행 환경:
- Python 3.10
- 서버: dclcom61 (.venv-webagent)
- 의존성: trimesh>=4.0, lxml(있으면 사용, 없으면 stdlib xml)

지원 입력:
1) 단일 .obj/.ply → trimesh로 로드 → PLY로 export → Jonggak.xml 스타일 Mitsuba XML 자동 래핑
2) .zip (Mitsuba XML + meshes/*.ply 포함) → 압축 해제 + XML 무결성 검증

출력:
- {session_dir}/scene/scene.xml          : Sionna load_scene()에 전달할 XML
- {session_dir}/scene/meshes/*.ply       : 메시 PLY 파일
- {session_dir}/scene/scene_info.json    : AABB, 단위, 메시 통계 메타
"""

from __future__ import annotations

import json
import shutil
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Literal, Optional
from xml.sax.saxutils import escape as xml_escape

import numpy as np
import trimesh


ITUMaterial = Literal["itu_concrete", "itu_ceiling_board", "itu_glass"]
DEFAULT_MATERIAL: ITUMaterial = "itu_concrete"

# Sionna RT가 인식하는 BSDF id prefix
MATERIAL_BSDF_ID = {
    "itu_concrete": "mat-itu_concrete",
    "itu_ceiling_board": "mat-itu_ceiling_board",
    "itu_glass": "mat-itu_glass",
}

# 각 재질별 디폴트 reflectance (Jonggak.xml과 동일)
MATERIAL_REFLECTANCE = {
    "itu_concrete": "0.8 0.8 0.8",
    "itu_ceiling_board": "0.8 0.8 0.8",
    "itu_glass": "0.8 0.8 0.8",
}


@dataclass
class SceneInfo:
    """변환된 씬의 메타데이터."""

    source_type: Literal["obj", "ply", "xml_zip", "xml_bundle"]
    mesh_files: list[str]
    materials: list[str]
    aabb_min: list[float]
    aabb_max: list[float]
    center: list[float]
    size: list[float]
    n_vertices: int
    n_faces: int
    units: str = "meters"
    # Geo-Radio Env. Twin 여부: 업로드 XML에 사용자가 이미 ITU 재질을 부여한 경우 True.
    # (단일 메시 업로드 = Geo Env. Twin = 기본재질 1종 → False)  (2026-06-19 추가)
    material_assigned: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def _aabb_from_mesh(mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray]:
    """메시의 axis-aligned bounding box (min, max)."""

    bounds = mesh.bounds  # shape (2, 3)
    return bounds[0].astype(float), bounds[1].astype(float)


def _write_mitsuba_xml(
    xml_path: Path,
    mesh_entries: Iterable[tuple[str, ITUMaterial]],
    used_materials: set[ITUMaterial],
    integrator_max_depth: int = 12,
) -> None:
    """Mitsuba 2.1.0 scene XML 작성. Jonggak.xml과 동일한 골격.

    Parameters
    ----------
    mesh_entries: Iterable[(ply_relative_path, ITUMaterial)]
    """

    lines: list[str] = []
    lines.append('<?xml version="1.0" encoding="utf-8"?>')
    lines.append('<scene version="2.1.0">')
    lines.append('    <integrator type="path">')
    lines.append(
        f'        <integer name="max_depth" value="{integrator_max_depth}"/>'
    )
    lines.append('    </integrator>')

    for mat in sorted(used_materials):
        bsdf_id = MATERIAL_BSDF_ID[mat]
        refl = MATERIAL_REFLECTANCE[mat]
        lines.append(f'    <bsdf type="diffuse" id="{xml_escape(bsdf_id)}" name="{xml_escape(bsdf_id)}">')
        lines.append(f'        <rgb value="{refl}" name="reflectance"/>')
        lines.append('    </bsdf>')

    for idx, (ply_rel, mat) in enumerate(mesh_entries, start=1):
        bsdf_ref = MATERIAL_BSDF_ID[mat]
        shape_id = f"elm__{idx}"
        lines.append(f'    <shape type="ply" id="{shape_id}" name="{shape_id}">')
        lines.append(f'        <string name="filename" value="{xml_escape(ply_rel)}"/>')
        lines.append('        <boolean name="face_normals" value="true"/>')
        lines.append(f'        <ref id="{xml_escape(bsdf_ref)}" name="bsdf"/>')
        lines.append('    </shape>')

    lines.append('</scene>')
    xml_path.write_text("\n".join(lines), encoding="utf-8")


def build_scene_from_single_mesh(
    src_path: Path,
    out_dir: Path,
    material: ITUMaterial = DEFAULT_MATERIAL,
    integrator_max_depth: int = 12,
) -> SceneInfo:
    """단일 .obj/.ply 파일을 Mitsuba XML 씬으로 변환.

    1. trimesh로 로드
    2. PLY로 export (`meshes/scene_mesh.ply`)
    3. scene.xml에 단일 shape로 등록
    """

    out_dir = Path(out_dir)
    meshes_dir = out_dir / "meshes"
    meshes_dir.mkdir(parents=True, exist_ok=True)

    mesh = trimesh.load(src_path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(
            f"Loaded object is not a single Trimesh (got {type(mesh).__name__}). "
            "Use a zipped multi-mesh upload for compound scenes."
        )

    # PLY로 저장 (binary, little-endian, face_normals 호환)
    ply_filename = "scene_mesh.ply"
    ply_out = meshes_dir / ply_filename
    mesh.export(ply_out)

    # XML 작성
    xml_path = out_dir / "scene.xml"
    _write_mitsuba_xml(
        xml_path,
        [(f"meshes/{ply_filename}", material)],
        used_materials={material},
        integrator_max_depth=integrator_max_depth,
    )

    aabb_min, aabb_max = _aabb_from_mesh(mesh)
    center = (aabb_min + aabb_max) / 2.0
    size = aabb_max - aabb_min

    src_type = "obj" if src_path.suffix.lower() == ".obj" else "ply"
    info = SceneInfo(
        source_type=src_type,
        mesh_files=[f"meshes/{ply_filename}"],
        materials=[material],
        aabb_min=aabb_min.tolist(),
        aabb_max=aabb_max.tolist(),
        center=center.tolist(),
        size=size.tolist(),
        n_vertices=int(mesh.vertices.shape[0]),
        n_faces=int(mesh.faces.shape[0]),
    )
    (out_dir / "scene_info.json").write_text(
        json.dumps(info.to_dict(), indent=2), encoding="utf-8"
    )
    return info


def _finalize_material_assigned_scene(
    out_dir: Path,
    xml_path: Path,
    search_dirs: list[Path],
    source_type: Literal["xml_zip", "xml_bundle"],
) -> SceneInfo:
    """이미 재질이 부여된 Mitsuba XML(+PLY들)을 세션 scene/ 으로 정착시킨다.

    Geo-Radio Env. Twin 공용 로직 (zip 업로드 / 멀티파일 업로드 양쪽에서 사용).

    동작:
    1. XML이 참조하는 모든 .ply 를 basename 으로 search_dirs 에서 찾아 out_dir/meshes/ 로 모음.
    2. XML 의 filename 참조를 모두 `meshes/<basename>` 으로 재작성 → out_dir/scene.xml.
       (RT 의 load_scene() 이 PLY 를 확실히 찾도록 경로를 정규화. 재질 ref/bsdf 는 원본 보존.)
    3. BSDF id(mat-itu_*) 에서 재질 목록을, 모든 PLY 에서 통합 AABB 를 계산.

    사용자가 지정한 재질(<ref id="mat-itu_glass"/> 등)은 절대 덮어쓰지 않는다.
    """

    out_dir = Path(out_dir)
    meshes_dir = out_dir / "meshes"
    meshes_dir.mkdir(parents=True, exist_ok=True)

    xml_text = xml_path.read_text(encoding="utf-8", errors="ignore")
    referenced = _extract_ply_refs(xml_text)
    if not referenced:
        raise ValueError("scene.xml 에 <string name=\"filename\" value=\"*.ply\"/> 참조가 없습니다.")

    # search_dirs 하위의 모든 .ply 를 NFC 정규화 basename → 실제 경로로 색인.
    # (Blender/리눅스 파일시스템이 한글 파일명을 NFD 로 저장해 XML(NFC)과 어긋나는 문제 방지)
    import unicodedata as _ud

    def _nfc(s: str) -> str:
        return _ud.normalize("NFC", s)

    ply_index: dict[str, Path] = {}
    for d in search_dirs:
        if not d.exists():
            continue
        for hit in d.rglob("*.ply"):
            if hit.is_file():
                ply_index.setdefault(_nfc(hit.name), hit)

    def _locate(ref: str) -> Optional[Path]:
        basename = Path(ref.replace("\\", "/")).name
        for d in search_dirs:
            direct = d / ref
            if direct.exists():
                return direct
        return ply_index.get(_nfc(basename))

    mesh_files: list[str] = []
    rewritten = xml_text
    for ref in referenced:
        basename = Path(ref.replace("\\", "/")).name
        found = _locate(ref)
        if found is None:
            raise ValueError(f"참조된 PLY 를 업로드 파일에서 찾을 수 없습니다: {ref}")
        dest = meshes_dir / basename
        if found.resolve() != dest.resolve():
            shutil.copyfile(found, dest)
        new_ref = f"meshes/{basename}"
        # XML 내 정확한 value="..." 토큰만 치환 (재질 ref 등 다른 속성은 건드리지 않음).
        rewritten = rewritten.replace(f'value="{ref}"', f'value="{new_ref}"')
        mesh_files.append(new_ref)

    (out_dir / "scene.xml").write_text(rewritten, encoding="utf-8")

    # 재질 목록: BSDF/ref id 의 mat-* 전체에서 추출 (사용자 정의 그대로).
    #   2026-07-06: 기존엔 mat-itu_* 만 잡아 커스텀 재질(irr_glass 등)이 목록/범례에서 누락됐다.
    #   → mat- 접두어의 모든 재질을 포함하도록 수정.
    import re as _re

    materials = sorted(set(_re.findall(r'id="mat-([A-Za-z0-9_]+)"', rewritten)))
    if not materials:
        materials = [DEFAULT_MATERIAL]

    # AABB / 정점·면 통계: 모든 PLY 통합.
    aabb_min = np.array([np.inf, np.inf, np.inf])
    aabb_max = np.array([-np.inf, -np.inf, -np.inf])
    n_v = 0
    n_f = 0
    for mfile in mesh_files:
        mesh = trimesh.load(out_dir / mfile, force="mesh")
        if isinstance(mesh, trimesh.Trimesh):
            mn, mx = _aabb_from_mesh(mesh)
            aabb_min = np.minimum(aabb_min, mn)
            aabb_max = np.maximum(aabb_max, mx)
            n_v += int(mesh.vertices.shape[0])
            n_f += int(mesh.faces.shape[0])

    center = (aabb_min + aabb_max) / 2.0
    size = aabb_max - aabb_min
    info = SceneInfo(
        source_type=source_type,
        mesh_files=mesh_files,
        materials=materials,
        aabb_min=aabb_min.tolist(),
        aabb_max=aabb_max.tolist(),
        center=center.tolist(),
        size=size.tolist(),
        n_vertices=n_v,
        n_faces=n_f,
        material_assigned=True,
    )
    (out_dir / "scene_info.json").write_text(
        json.dumps(info.to_dict(), indent=2), encoding="utf-8"
    )
    return info


def build_scene_from_xml_zip(zip_path: Path, out_dir: Path) -> SceneInfo:
    """Mitsuba XML + PLY 메시들이 포함된 .zip → Geo-Radio Env. Twin 씬.

    zip 구조 가정:
    - .xml 1개 (루트 또는 폴더 안)
    - meshes/ 또는 같은 폴더의 .ply 파일들이 XML 의 filename 참조와 (basename 기준) 일치

    실패 시 ValueError.
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stage = out_dir / "_bundle_src"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        if not names:
            raise ValueError("Uploaded zip is empty.")
        xml_candidates = [n for n in names if n.lower().endswith(".xml") and not n.endswith("/")]
        if not xml_candidates:
            raise ValueError("Uploaded zip does not contain a .xml file.")
        # Mitsuba scene XML 은 보통 1개. 가장 짧은 경로(루트에 가까운)를 사용.
        xml_candidates.sort(key=lambda p: (p.count("/"), len(p)))
        xml_inside = xml_candidates[0]

        for name in names:
            if not name or name.endswith("/"):
                continue
            target = (stage / name).resolve()
            if not str(target).startswith(str(stage.resolve())):
                raise ValueError(f"Refusing zip entry outside session: {name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(name) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)

    info = _finalize_material_assigned_scene(
        out_dir, stage / xml_inside, [stage], source_type="xml_zip"
    )
    shutil.rmtree(stage, ignore_errors=True)
    return info


def build_scene_from_files(
    xml_path: Path,
    ply_paths: list[Path],
    out_dir: Path,
) -> SceneInfo:
    """멀티파일 업로드(.xml + 여러 .ply) → Geo-Radio Env. Twin 씬.

    브라우저 다중 파일 선택은 폴더 구조 없이 파일들이 평탄하게 올라온다.
    XML 참조(예: ``meshes/foo.ply``)는 basename 으로 매칭해 정착시킨다.
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    search_dirs = list({Path(xml_path).parent, *[Path(p).parent for p in ply_paths]})
    return _finalize_material_assigned_scene(
        out_dir, Path(xml_path), search_dirs, source_type="xml_bundle"
    )


def _extract_ply_refs(xml_text: str) -> list[str]:
    """매우 가벼운 XML 파싱: `<string name="filename" value="..."/>` 의 PLY 경로 추출."""

    import re as _re

    return _re.findall(
        r'<string\s+name="filename"\s+value="([^"]+\.ply)"', xml_text
    )


def build_scene(
    src_path: Path,
    out_dir: Path,
    material: ITUMaterial = DEFAULT_MATERIAL,
    integrator_max_depth: int = 12,
) -> SceneInfo:
    """입력 확장자에 따라 적절한 빌더로 분기."""

    src_path = Path(src_path)
    suffix = src_path.suffix.lower()
    if suffix == ".zip":
        return build_scene_from_xml_zip(src_path, Path(out_dir))
    if suffix in {".obj", ".ply"}:
        return build_scene_from_single_mesh(
            src_path,
            Path(out_dir),
            material=material,
            integrator_max_depth=integrator_max_depth,
        )
    raise ValueError(
        f"Unsupported scene upload extension: {suffix} (expected .obj / .ply / .zip)"
    )


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Build Mitsuba scene from obj/ply/zip.")
    p.add_argument("input", help="입력 .obj / .ply / .zip 경로")
    p.add_argument("output_dir", help="세션 scene/ 출력 디렉토리")
    p.add_argument(
        "--material",
        default=DEFAULT_MATERIAL,
        choices=list(MATERIAL_BSDF_ID.keys()),
        help="단일 메시 업로드 시 매핑할 ITU 재질",
    )
    args = p.parse_args()
    info = build_scene(Path(args.input), Path(args.output_dir), material=args.material)
    print(json.dumps(info.to_dict(), indent=2))
