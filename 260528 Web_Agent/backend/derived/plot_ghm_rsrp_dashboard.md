# plot_ghm_rsrp_dashboard.py

GHM Twin RT 결과(`channel_data_260531_GHM_Twin_v0_1.npz`)와 동일한 **RSRP 대시보드** PNG를 생성합니다.

## 출력 형식

- 좌: **RX별 scatter** (jet, -110~-60 dBm) + PLY 건물 외곽선 + Dead Zone(흰 점) + TX(빨간 ★)
- 우: **RSRP 히스토그램** + Total/Valid/Dead/Mean/Max/Min 통계 박스
- 제목: `RSRP Heatmap — {scene}` / `All-TX combined RSRP (3 TXs, linear sum -> dB)`

## RSRP 정의

- 입력 `rsrp_all`: shape `(3, N)` [dBm], TX별 사전 계산값
- 합성: `10*log10(Σ_tx 10^(RSRP_tx/10))`
- Dead zone: **3 TX 모두 `-inf`** (5071 중 94개)

## 호출

```python
from backend.derived.plot_ghm_rsrp_dashboard import load_ghm_channel_rsrp, save_ghm_rsrp_dashboard

ch = load_ghm_channel_rsrp(Path(".../channel_data_260531_GHM_Twin_v0_1.npz"))
save_ghm_rsrp_dashboard(
    Path("RSRP_GHM_dashboard.png"),
    ch["rx_positions"],
    ch["rsrp_dbm"],
    ch["tx_positions"],
    ch["dead_mask"],
    scene_title="260531_GHM Twin_v0.1",
    mesh_ply_path=Path(".../GHM Twin_v0.1.ply"),
)
```

## Web Agent 연동

`derive_rsrp(..., ghm_channel_npz=..., mesh_ply_path=...)` 시 이 레이아웃을 기본 PNG로 사용합니다.
