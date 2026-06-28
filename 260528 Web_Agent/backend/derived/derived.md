# backend/derived/

25\* 원본에 직접 산출 함수가 없는 4개 도출 metric을 P1B Ray NPZ로부터 계산.

## 모듈

| 파일 | metric | 입력 | 출력 |
|------|--------|------|------|
| `derive_rsrp.py` | RSRP | P1B NPZ + RX 좌표 | CSV + 2D scatter PNG |
| `derive_pdp.py` | PDP (Power Delay Profile) | P1B NPZ | NPZ (100 bin) + PNG subplot |
| `derive_padp.py` | PADP (Power Angular Delay Profile) | P1B NPZ | NPZ (100×72×36) + τ-φ PNG |
| `derive_ray_stats.py` | DS / K / AS | P1B NPZ | CSV |
| `common.py` | 공용 유틸 | - | load_p1b_npz, iter_rays_per_rx, kst_timestamp |

## 입력 NPZ 키 (251218 표준)

| 키 | shape | 설명 |
|----|-------|------|
| `tau` | (num_rx, 1, 1, 1, max_rays) | 지연 [s] |
| `power` | 동일 | linear power |
| `theta_r_deg / theta_t_deg` | 동일 | 수신/송신 zenith [deg] |
| `phi_r_deg / phi_t_deg` | 동일 | 수신/송신 azimuth [deg] |
| `counts` | (num_rx, 1, 1) | RX별 유효 ray 수 |
| `source_path_idx`, `los_nlos_flag` | 동일 | optional |
| `rx_indices`, `area_index`, `frequency_ghz`, `num_rx` | scalar/list | 메타 |

## 결정사항 매핑

- **RSRP**: `10*log10(sum(power))` per RX. 무효 RX는 -300 dB.
- **PDP**: tau range을 RX별로 자동 binning. 100 bin power-weighted histogram.
- **PADP**: tau × azimuth × zenith = 100 × 72 × 36. azimuth는 [-180, 180], zenith는 [0, 180].
- **Ray Stats**:
  - Delay spread = power-weighted RMS τ
  - K-factor = LoS power / NLoS power (`los_nlos_flag==1` 기준)
  - ASA/ASD = circular std (azimuth)
  - ZSA/ZSD = linear std (zenith)

## 그래프 규칙 (AGENTS.md)

- `tight_layout()` 사용
- 범례 위치 `upper right`
- 폰트 사이즈 9~12 균일
- 색맵 `viridis`, vmin/vmax 정규화 dB
- subplot grid는 RX 16개 이내 (PDP), 4개 이내 (PADP)
