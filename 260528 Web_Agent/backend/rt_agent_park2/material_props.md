# material_props.py

씬 재질(material) 속성 조회/계산 유틸. `GET /api/sessions/{uuid}/scene/materials` 및
3.TX/RX 'Material Properties' 섹션이 사용.

## 제공 기능
- `itu_eps_sigma(itu_type, freq_hz)` → (εr, σ). ITU-R P.2040-3 모델(εr=a·f^b, σ=c·f^d).
  상수는 Sionna 1.2.2 `itu.py` 와 동일.
- `scattering_default_for(name)` → 문헌 기반 산란계수 S 기본값(콘크리트 0.4·유리 0.2 등).
  ※ S는 표면 거칠기 파라미터로 ITU-R P.2040에 정의되지 않음(Sionna 기본 0.0). 사용자 튜닝 대상.
- `parse_scene_materials(scene_xml, freq_hz)` → 재질 목록.
  각 항목: name, kind('itu'|'custom'), itu_type, bsdf_type, shape_count,
  scattering_default, eps_r, sigma, xpd_coefficient, scattering_xml.
  · ITU 재질: εr/σ 를 freq_hz 로 계산.
  · custom 재질(radio-material): scene.xml 에 명시된 εr/σ/xpd/scattering 을 그대로 읽음.

## 재질별 산란계수(S) 배선 흐름
1. 3.TX/RX 'Material Properties' → `rt.material_scattering = {재질명: S}` (프론트 store)
2. buildPayload → payload.rt.material_scattering
3. schemas.RTConfig.material_scattering (보존)
4. batch_runner._build_config → cfg.material_scattering ((0,1) 범위 밖 값은 드롭)
5. m3_scene_agent: 재질별 오버라이드 적용. 오버라이드 없는 ITU 재질=전역 S, custom=XML 보존.
   (P1A 경로는 forked_25x/P1A_..._web.py tune_materials + ITU_MATERIAL_SCATTERING)

## 주의
- ITU σ는 주파수 의존(예: concrete 3.5→7GHz 0.123→0.212), εr은 대역 내 상수(ITU 모델 특성).
- 커스텀 irr_glass 는 εr=6.31, σ=5.0 고정(주파수 특성 보완은 별도 과제).
