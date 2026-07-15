# RTWA superset NPZ 사용설명서 (일반)

RTWA(Ray Tracing Web Agent)의 **Intg 엔진**이 만드는 **superset NPZ**의 스키마와 사용법.
한 번의 RT 실행 결과(여러 TX × 여러 RX × 여러 경로)를 **조밀한 배열 하나**에 담은 표준(canonical) 파일이다.
파일명 형태: `superset_<sessionid>_<날짜>_<시각>.npz`

## 차원 기호
| 기호 | 의미 | 저장 key |
|---|---|---|
| `T` | TX 개수 | `tx_positions` 로 확인 |
| `R` | RX 개수 | `rx_positions` 로 확인 |
| `P` | 경로 최대 개수 | `max_paths` |
| `K` | 선별 경로 최대 개수 | `max_rays` |
| `A_t` / `A_r` | TX/RX 안테나 포트 수 | `num_tx_ant` / `num_rx_ant` |

대부분 배열의 축 순서는 **`[tx, rx, path]`** = `[T, R, P]` 또는 `[T, R, K]`.
SISO면 `A_t=A_r=1`, MIMO면 그만큼 커진다.

---

## 1) 메타 / 지오메트리
| key | shape | dtype | 설명 |
|---|---|---|---|
| `tx_positions` | (T, 3) | float64 | TX 좌표 [x,y,z] (m) |
| `rx_positions` | (R, 3) | float64 | RX 좌표 [x,y,z] (m) |
| `frequency_ghz` | () | float64 | 반송 주파수 [GHz] |
| `num_tx_ant` / `num_rx_ant` | (1,) | int64 | TX/RX 안테나 포트 수 |
| `max_paths` / `max_rays` | (1,) | int64 | 경로(P) / 선별경로(K) 상한 |
| `target_tx_index` / `target_rx_index` | (1,) | int64 | 진단·기본 표시용 기준 인덱스 |
| `rx_valid_mask` | (R,) | int8 | RX 유효 마스크 (1=유효, 0=제외) |

## 2) 전체 경로 블록 (dense superset) — 축 `[T, R, P]`
RT가 찾은 **모든 경로**. RX마다 실제 유효 경로 수만큼(`path_counts`) 앞에서부터 채워지고 나머지는 패딩.

| key | shape | 단위/의미 |
|---|---|---|
| `path_tau` | (T, R, P) | 지연 [**ns**] |
| `path_power` | (T, R, P) | 경로 전력 [**dB**] |
| `path_phi_r` / `path_phi_t` | (T, R, P) | 수신/송신 방위각 [**deg**] |
| `path_theta_r` / `path_theta_t` | (T, R, P) | 수신/송신 고도(zenith)각 [**deg**] |
| `path_los_flag` | (T, R, P) | 경로 LoS 여부 (1=LoS, 0=NLoS) |
| `path_counts` | (T, R) | (tx,rx)별 유효 경로 개수 (0~P) |

## 3) 선별 경로 블록 (채널 생성용 canonical) — 축 `[T, R, K]`
2)의 경로 중 대표(상위 전력 등) **최대 K개**만 추린 세트. **단위가 2)와 다르니 주의.**

| key | shape | 단위/의미 |
|---|---|---|
| `tau` | (T, R, K) | 지연 [**초, s**] |
| `power` | (T, R, K) | 선형 채널이득 **|a|²** (TX 전력 미포함) |
| `phi_r_deg` / `phi_t_deg` | (T, R, K) | 수신/송신 방위각 [deg] |
| `theta_r_deg` / `theta_t_deg` | (T, R, K) | 수신/송신 고도각 [deg] |
| `source_path_idx` | (T, R, K) | 이 경로가 2)의 몇 번째(P 인덱스)인지 매핑 |
| `los_nlos_flag` | (T, R, K) | 경로 LoS 여부 (1/0) |
| `counts` | (T, R) | (tx,rx)별 선별 경로 개수 (0~K) |

## 4) 유도 관측량 (per TX×RX)
| key | shape | 설명 |
|---|---|---|
| `rsrp_all` | (T, R) | RSRP [dBm] = 10·log10(Σ|a|²) + power_offset. 무효는 -inf/NaN 가능 |
| `los_all` | (T, R) | 해당 (tx,rx)에 LoS 경로 존재 여부(bool) |
| `R_TX` | (T, R, A_t, A_t) | TX단 공간 공분산 (SISO면 1×1) |
| `R_RX` | (T, R, A_r, A_r) | RX단 공간 공분산 (SISO면 1×1) |
| `cov_omitted` | (2,) | 공분산 계산 생략 카운트 [R_TX, R_RX] |

> ⚠️ **단위 요약**
> - 2) 전체 경로: tau [ns] · power [dB] · 각도 [deg]
> - 3) 선별 경로: tau [s] · power [선형 |a|²] · 각도 [deg]

---

## 로딩 & 인덱싱 규칙
- `np.load(path, allow_pickle=True)` 로 열고 key 로 접근.
- 특정 (tx,rx)의 경로만 쓰려면 **반드시 count 로 잘라라**(뒤쪽은 패딩):
  - 전체: `n = path_counts[t, r]` → `path_tau[t, r, :n]`
  - 선별: `m = counts[t, r]` → `tau[t, r, :m]`
- 무효/미도달 (tx,rx)는 `counts==0` 또는 `path_counts==0`. RSRP가 비유한값일 수 있으니 걸러라.
- 선별↔전체 연결: `source_path_idx[t,r,k]` = 2) 블록의 path 인덱스.

## 용도별 사용법
- **커버리지 맵 / RSRP 히트맵**: `rsrp_all[t]` (R,) 를 `rx_positions` 와 함께 산포/격자로.
- **LoS/NLoS 분석**: `los_all[t]` 또는 경로별 `path_los_flag`.
- **PDP / PADP**: 한 (tx,rx)의 `path_tau`(ns) vs `path_power`(dB), 각도축은 `path_phi_r`/`path_theta_r`.
- **채널 계수 H 생성 (38.901 등)**: 선별 블록의 `tau`(s), `power`(|a|²), 각도(`*_deg`) 사용.
- **공간 공분산**: `R_RX`/`R_TX` (MIMO 실행 시 A×A 행렬).

## 최소 예시 (Python)
```python
import numpy as np
d = np.load("superset_XXXX.npz", allow_pickle=True)
T, R = d["path_counts"].shape
freq = float(d["frequency_ghz"])

# (a) 임의 (tx,rx) 전체 경로
t, r = 0, 0
n = int(d["path_counts"][t, r])
tau_ns  = d["path_tau"][t, r, :n]        # [ns]
pow_dB  = d["path_power"][t, r, :n]      # [dB]
aoa_deg = d["path_phi_r"][t, r, :n]      # 수신 방위각 [deg]

# (b) 채널생성용 선별 경로 (tau[s], power 선형)
m = int(d["counts"][t, r])
tau_s   = d["tau"][t, r, :m]             # [s]
gain    = d["power"][t, r, :m]           # |a|^2 (선형)

# (c) TX t 의 전 RX 커버리지
rsrp = d["rsrp_all"][t]                  # (R,) [dBm]
los  = d["los_all"][t]                   # (R,) bool
xyz  = d["rx_positions"]                 # (R,3)

# (d) TX-RX 거리
dist = np.linalg.norm(d["rx_positions"][r] - d["tx_positions"][t])
```

## 주의사항
- `power`(|a|²)는 Sionna `cir()` 기준이라 **TX 전력 미포함**. 절대 수신전력은 RSRP(`power_offset` 포함)를 쓰거나 직접 오프셋을 더할 것. `power_offset` 은 이 파일이 아니라 **RT 실행 config** 에 있다.
- 안테나가 다중(MIMO)이면 `num_tx_ant`/`num_rx_ant` 가 1보다 크고 `R_TX`/`R_RX` 축이 그만큼 커진다. 파일마다 `num_*_ant` 로 확인해서 하드코딩하지 말 것.
- 배열이 크다(수백 MB~GB). 필요한 슬라이스만 접근하고, 전체를 float64 로 복사하지 말 것.
