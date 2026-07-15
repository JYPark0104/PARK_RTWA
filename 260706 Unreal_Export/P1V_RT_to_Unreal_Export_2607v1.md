# P1V_RT_to_Unreal_Export_2607v1.py

## 목적
RT 채널 데이터(`channel_data_*.npz`)를 **Unreal Engine 시각화용 CSV/JSON**으로 변환하는
"데이터 다리(bridge)" 스크립트. 서버(dclserver78)에서 계산된 RT 결과를 로컬 PC의 Unreal이
읽을 수 있는 범용 포맷으로 뽑아낸다.

파이프라인 상 Unreal은 **데이터를 소비만 하는 최종 시각화(터미널) 노드**이며,
이 스크립트가 그 입력을 만든다.

## 입력
- `channel_data_*.npz` (기본: `260512 RT Result Data/channel_data_260531_GHM_Twin_v0_1.npz`)
- 주요 키
  | 키 | shape | 의미 |
  |----|-------|------|
  | `tx_positions` | (n_tx, 3) | TX 위치 xyz [m] (높이 포함) |
  | `rx_positions` | (n_rx, 2) | RX 위치 xy [m] (**높이 없음**) |
  | `rsrp_all` | (n_tx, n_rx) | TX별 RX RSRP [dBm], dead zone = -inf |
  | `los_all` | (n_tx, n_rx) | TX별 LoS/NLoS bool |
  | `tau/power/aoa_tx{t}_rx{r}` | (n_paths,) | 경로별 지연/전력/도착각(방위 1개) |

## 출력 (`P1V_Unreal_Export_Results/`, 타임스탬프 부여 → 덮어쓰기 금지)
| 파일 | 내용 |
|------|------|
| `unreal_tx_<ts>.csv` | `tx_index, x, y, z` |
| `unreal_rx_<ts>.csv` | `rx_index, x, y, z, rsrp_dbm, rsrp_norm, los, n_paths` |
| `unreal_los_lines_<ts>.csv` (옵션) | LoS 쌍 TX→RX 직선 (start/end 좌표 + RSRP) |
| `unreal_meta_<ts>.json` | 좌표계/단위/범위/개수 등 메타데이터 |

- `rsrp_norm`: 유효 RSRP min~max 기준 [0,1] 정규화값 → Unreal에서 색/높이 매핑에 바로 사용
- dead zone(-inf) RX는 `rsrp_dbm = -inf`, `rsrp_norm = 0`으로 표기

## 사용법
```bash
python3 P1V_RT_to_Unreal_Export_2607v1.py \
    --npz "../260512 RT Result Data/channel_data_260531_GHM_Twin_v0_1.npz" \
    --tx 0 --rx-height 1.5 --los-lines
```
| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--npz` | 입력 npz 경로 | 위 기본 파일 |
| `--tx` | 커버리지 색칠 기준 TX 인덱스 | `target_tx_index` |
| `--rx-height` | RX 높이 가정값 [m] | 1.5 |
| `--los-lines` | LoS 직선 CSV 추가 생성 | off |
| `--max-rx` | RX 서브샘플 상한 | 전체 |

## ⚠️ 한계 (중요)
이 `.npz`에는 **광선의 3D 반사 경로(interaction points)가 없다.** `tau`(지연)·`power`(전력)·
`aoa`(방위 도착각 1개)만 있으므로 "건물에 튕기는 ray" geometry는 그릴 수 없다.

- ✅ 가능: TX 타워, RX 커버리지 포인트 클라우드(RSRP 색칠), LoS/NLoS 구분, LoS 직선
- ❌ 불가: 다중 반사 ray 폴리라인 → `P1A_RT_to_Rays` 계열 또는 Web_Agent `SCENARIO_DATA` 필요

## 좌표계 주의
- 원본 npz 좌표(**미터, RT 좌표계**)를 그대로 export.
- Unreal 임포트 단계에서 **m → cm (×100)**, 축 방향(Z-up 관례) 확인 필요.
- `meta.json`의 `coordinate_system`, `extent_m`에 원본 단위/범위를 명시.

## Unreal 활용 가이드 (다음 단계)
1. `unreal_rx_*.csv`를 DataTable로 임포트
2. Blueprint가 각 행 → RX 위치에 마커 스폰, `rsrp_norm`을 색(파랑↔빨강) 또는 높이에 매핑
3. `unreal_tx_*.csv` → TX 타워 배치
4. (옵션) `unreal_los_lines_*.csv` → LoS 직선/빔 스폰
5. 대규모(5071개)이므로 Niagara 또는 HISM(Instanced Static Mesh) 권장

## 실행 환경
- Python 3.10+, numpy만 필요 (GPU 불필요)
- 실행 서버: dclserver78 (Linux)

## 데이터 검증 결과 (기본 파일 기준)
- TX 3개 (높이 112/71/47 m), RX 5071개
- dead zone 482개, RSRP 범위 -120.00 ~ -56.47 dBm
- TX0 기준 LoS 직선 1259개
