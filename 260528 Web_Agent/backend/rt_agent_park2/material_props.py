"""
material_props.py
=================
씬 재질(material) 속성 조회/계산 유틸.

역할:
  - ITU-R P.2040 재질의 (εr, σ)를 주파수로부터 계산 (Sionna 로드 없이 경량 계산).
  - 재질별 '문헌 기반 산란계수 S' 기본값 제공.
  - scene.xml 을 파싱하여 재질 목록 + 타입(ITU/custom) + shape 수 + (custom의 경우) 명시된 εr/σ 를 반환.

주의:
  - ITU 모델 상수는 Sionna 1.2.2 `sionna/rt/radio_materials/itu.py`(ITU-R P.2040-3, Table 3)와 동일.
  - 산란계수 S는 ITU-R P.2040에 정의되지 않음(표면 거칠기 파라미터). 아래 값은 문헌(Degli-Esposti 등)
    측정치에 근거한 '대략적 기본값'이며 표면 상태에 따라 편차가 큼 → 사용자가 GUI에서 조정 가능.

작성: 2026-07-06
"""
from __future__ import annotations

import re
from pathlib import Path
from xml.etree import ElementTree as ET

# ── ITU-R P.2040-3 Table 3 (Sionna itu.py 와 동일) ────────────────────────────
# material_name: { (min_ghz, max_ghz): (a, b, c, d) }  →  εr=a·f^b, σ=c·f^d  (f=GHz)
ITU_MATERIALS_PROPERTIES = {
    "concrete":          {(1., 100.):   (5.24, 0.0, 0.0462, 0.7822)},
    "brick":             {(1., 40.):    (3.91, 0.0, 0.0238, 0.16)},
    "plasterboard":      {(1., 100.):   (2.73, 0.0, 0.0085, 0.9395)},
    "wood":              {(0.001, 100.):(1.99, 0.0, 0.0047, 1.0718)},
    "glass":             {(0.1, 100.):  (6.31, 0.0, 0.0036, 1.3394),
                          (220., 450.): (5.79, 0.0, 0.0004, 1.658)},
    "ceiling_board":     {(1.0, 100.):  (1.48, 0.0, 0.0011, 1.0750),
                          (220., 450.): (1.52, 0.0, 0.0029, 1.029)},
    "chipboard":         {(1.0, 100.):  (2.58, 0.0, 0.0217, 0.7800)},
    "plywood":           {(1.0, 40.):   (2.71, 0.0, 0.33, 0.0)},
    "marble":            {(1.0, 60.):   (7.074, 0.0, 0.0055, 0.9262)},
    "floorboard":        {(50., 100.):  (3.66, 0.0, 0.0044, 1.3515)},
    "metal":             {(1.0, 100.):  (1.0, 0.0, 1e7, 0.0)},
    "very_dry_ground":   {(1.0, 10.):   (3.0, 0.0, 0.00015, 2.52)},
    "medium_dry_ground": {(1.0, 10.):   (15., -0.1, 0.035, 1.63)},
    "wet_ground":        {(1.0, 10.):   (30., -0.4, 0.15, 1.30)},
}

# ── 문헌 기반 산란계수 S 기본값 (표면 거칠기; ITU 표준 아님) ──────────────────
#   콘크리트/벽돌 = 거칠어서 큼, 유리/금속 = 매끈해서 작음.
LITERATURE_SCATTERING = {
    "concrete": 0.4, "brick": 0.4, "plasterboard": 0.3, "wood": 0.4,
    "glass": 0.2, "ceiling_board": 0.3, "chipboard": 0.4, "plywood": 0.4,
    "marble": 0.3, "floorboard": 0.4, "metal": 0.2,
    "very_dry_ground": 0.5, "medium_dry_ground": 0.5, "wet_ground": 0.5,
}
DEFAULT_SCATTERING = 0.3   # 표에 없는(커스텀 포함) 재질 기본값


def itu_eps_sigma(itu_type: str, freq_hz: float) -> tuple[float, float]:
    """ITU 재질의 (εr, σ[S/m]) 를 주파수(Hz)로부터 계산. 범위 밖이면 가장 가까운 구간 사용."""
    props = ITU_MATERIALS_PROPERTIES.get(itu_type)
    if props is None:
        raise KeyError(itu_type)
    f_ghz = freq_hz / 1e9
    chosen = None
    for (fmin, fmax), params in props.items():
        if fmin <= f_ghz <= fmax:
            chosen = params
            break
    if chosen is None:
        # 범위 밖 → 경계에 가장 가까운 구간으로 클램프 (표시용, 하드에러 방지)
        best, bestd = None, None
        for (fmin, fmax), params in props.items():
            d = 0.0 if fmin <= f_ghz <= fmax else min(abs(f_ghz - fmin), abs(f_ghz - fmax))
            if bestd is None or d < bestd:
                best, bestd = params, d
        chosen = best
    a, b, c, d = chosen
    eps_r = a * (f_ghz ** b)
    sigma = c * (f_ghz ** d)
    return float(eps_r), float(sigma)


def scattering_default_for(name: str) -> float:
    """재질명(예: 'itu_concrete' / 'concrete' / 'irr_glass')에 대한 문헌 기본 S."""
    key = name[4:] if name.startswith("itu_") else name
    return LITERATURE_SCATTERING.get(key, DEFAULT_SCATTERING)


def _bsdf_float(bsdf: ET.Element, attr: str) -> float | None:
    for f in bsdf.findall("float"):
        if f.get("name") == attr:
            try:
                return float(f.get("value"))
            except (TypeError, ValueError):
                return None
    return None


def parse_scene_materials(scene_xml: str | Path, freq_hz: float) -> list[dict]:
    """scene.xml → 재질 목록.

    각 항목: {name, kind('itu'|'custom'), itu_type, eps_r, sigma, xpd_coefficient,
              scattering_xml, scattering_default, shape_count}
      - ITU 재질: eps_r/sigma 는 freq_hz 로 계산.
      - custom 재질: bsdf(radio-material)에 명시된 εr/σ/xpd/scattering 을 그대로 읽음.
    """
    scene_xml = Path(scene_xml)
    txt = scene_xml.read_text(encoding="utf-8", errors="ignore")

    # shape 별 material ref 개수
    refs = re.findall(r'<ref id="mat-([A-Za-z0-9_]+)"', txt)
    from collections import Counter
    shape_counts = Counter(refs)

    # bsdf 정의 파싱
    out: list[dict] = []
    try:
        root = ET.fromstring(txt)
    except ET.ParseError:
        root = None

    bsdfs = root.findall("bsdf") if root is not None else []
    seen = set()
    for bsdf in bsdfs:
        bid = bsdf.get("id", "")
        if not bid.startswith("mat-"):
            continue
        name = bid[4:]
        seen.add(name)
        bsdf_type = bsdf.get("type", "")
        itu_type = name[4:] if name.startswith("itu_") else None
        is_itu = (itu_type is not None) and (itu_type in ITU_MATERIALS_PROPERTIES)

        entry = {
            "name": name,
            "kind": "itu" if is_itu else "custom",
            "itu_type": itu_type if is_itu else None,
            "bsdf_type": bsdf_type,
            "shape_count": int(shape_counts.get(name, 0)),
            "scattering_default": round(scattering_default_for(name), 3),
        }
        if is_itu:
            eps, sig = itu_eps_sigma(itu_type, freq_hz)
            entry["eps_r"] = round(eps, 4)
            entry["sigma"] = round(sig, 6)
            entry["xpd_coefficient"] = None       # ITU 기본 0, config(itu_xpd_coeff)로 설정됨
            entry["scattering_xml"] = None
        else:
            eps = _bsdf_float(bsdf, "relative_permittivity")
            sig = _bsdf_float(bsdf, "conductivity")
            entry["eps_r"] = round(eps, 4) if eps is not None else None
            entry["sigma"] = round(sig, 6) if sig is not None else None
            entry["xpd_coefficient"] = _bsdf_float(bsdf, "xpd_coefficient")
            entry["scattering_xml"] = _bsdf_float(bsdf, "scattering_coefficient")
        out.append(entry)

    # bsdf 정의가 없고 ref 만 있는 경우(방어적) — ref 이름으로 최소 항목 생성
    for name in shape_counts:
        if name in seen:
            continue
        itu_type = name[4:] if name.startswith("itu_") else None
        is_itu = (itu_type is not None) and (itu_type in ITU_MATERIALS_PROPERTIES)
        entry = {
            "name": name, "kind": "itu" if is_itu else "custom",
            "itu_type": itu_type if is_itu else None, "bsdf_type": "",
            "shape_count": int(shape_counts[name]),
            "scattering_default": round(scattering_default_for(name), 3),
            "xpd_coefficient": None, "scattering_xml": None,
        }
        if is_itu:
            eps, sig = itu_eps_sigma(itu_type, freq_hz)
            entry["eps_r"] = round(eps, 4); entry["sigma"] = round(sig, 6)
        else:
            entry["eps_r"] = None; entry["sigma"] = None
        out.append(entry)

    out.sort(key=lambda e: (-e["shape_count"], e["name"]))
    return out
