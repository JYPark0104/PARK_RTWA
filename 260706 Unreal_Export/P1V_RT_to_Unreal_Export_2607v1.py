"""
P1V_RT_to_Unreal_Export_2607v1.py
=================================================================
RT 채널 데이터(channel_data_*.npz) → Unreal Engine 시각화용 CSV/JSON export.

[역할]
  파이프라인의 "데이터 다리(bridge)". 서버에서 계산된 RT 결과(.npz)를
  Unreal이 읽을 수 있는 범용 포맷(CSV/JSON)으로 변환한다.
  Unreal은 이 파일을 소비만 하는 최종 시각화(터미널) 노드이다.

[입력]
  channel_data_*.npz  (예: 260512 RT Result Data/channel_data_260531_GHM_Twin_v0_1.npz)
  주요 키:
    - tx_positions   (n_tx, 3)   : TX 위치 xyz [m] (높이 포함)
    - rx_positions   (n_rx, 2)   : RX 위치 xy  [m] (높이 없음 → 가정값 부여)
    - rsrp_all       (n_tx, n_rx): TX별 RX RSRP [dBm], dead zone = -inf
    - los_all        (n_tx, n_rx): TX별 LoS/NLoS bool
    - tau/power/aoa_tx{t}_rx{r}  : 경로별 지연/전력/도착각(방위 1개)

[출력]  (P1V_Unreal_Export_Results/ 하위, 타임스탬프 부여 → 덮어쓰기 금지)
    - unreal_tx_<ts>.csv    : tx_index, x, y, z
    - unreal_rx_<ts>.csv    : rx_index, x, y, z, rsrp_dbm, rsrp_norm, los, n_paths
    - unreal_los_lines_<ts>.csv (옵션) : LoS 쌍의 TX→RX 직선 (start/end 좌표)
    - unreal_meta_<ts>.json : 좌표/단위/범위/개수 등 메타데이터

[한계]  (중요)
    이 .npz에는 광선의 3D 반사 경로(interaction points)가 없다.
    따라서 "튕기는 ray" geometry는 그릴 수 없고, 커버리지/포인트 클라우드 +
    (LoS 한정) 직선까지만 export 한다. 완전한 ray 경로가 필요하면
    P1A_RT_to_Rays 계열 또는 Web_Agent SCENARIO_DATA를 소스로 사용할 것.

[좌표계 주의]
    본 파일은 원본 .npz 좌표(미터, RT 좌표계)를 그대로 export 한다.
    Blender↔Unreal 단위/축 변환(m→cm, 축 방향)은 Unreal 임포트 단계에서
    처리하는 것을 권장한다. meta.json에 원본 단위/범위를 명시한다.

-----------------------------------------------------------------
실행 환경
    Python      : 3.10+
    실행 서버   : dclserver78 (Linux)
    주요 라이브러리 : numpy (표준 라이브러리 외 numpy만 필요)
    GPU         : 불필요 (CPU 후처리)
-----------------------------------------------------------------
사용법
    python3 P1V_RT_to_Unreal_Export_2607v1.py \
        --npz "../260512 RT Result Data/channel_data_260531_GHM_Twin_v0_1.npz" \
        --tx 0 --rx-height 1.5 --los-lines

    옵션:
      --npz PATH        입력 npz 경로 (기본: 아래 DEFAULT_NPZ)
      --tx  INT         커버리지 색칠 기준 TX 인덱스 (기본: target_tx_index)
      --rx-height FLOAT RX 높이 가정값 [m] (기본 1.5, npz에 z 없음)
      --los-lines       LoS 쌍 TX→RX 직선 CSV 추가 생성
      --max-rx INT      RX 서브샘플 상한 (기본: 전체)
"""

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np


# ============================================================
# 기본 설정
# ============================================================
DEFAULT_NPZ = "../260512 RT Result Data/channel_data_260531_GHM_Twin_v0_1.npz"
RESULTS_DIR = "P1V_Unreal_Export_Results"
RX_HEIGHT_DEFAULT = 1.5  # RX 높이 가정: 지상 UE 1.5 m (npz에 z 미포함)


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def load_npz(path: str):
    if not os.path.exists(path):
        print(f"[ERROR] 입력 파일 없음: {path}")
        sys.exit(1)
    return np.load(path, allow_pickle=True)


def normalize_rsrp(rsrp_row: np.ndarray):
    """RSRP[dBm] → [0,1] 정규화. dead zone(-inf)은 NaN 처리 후 0으로.

    Unreal에서 색/높이 매핑에 바로 쓸 수 있도록 유효값 min~max 기준 정규화.
    """
    finite = rsrp_row[np.isfinite(rsrp_row)]
    if finite.size == 0:
        return np.zeros_like(rsrp_row), (np.nan, np.nan)
    vmin, vmax = float(finite.min()), float(finite.max())
    span = (vmax - vmin) if (vmax - vmin) > 1e-9 else 1.0
    norm = (rsrp_row - vmin) / span
    norm[~np.isfinite(rsrp_row)] = 0.0
    return np.clip(norm, 0.0, 1.0), (vmin, vmax)


def n_paths_for(data, tx: int, rx: int) -> int:
    key = f"tau_tx{tx}_rx{rx}"
    return int(len(data[key])) if key in data else 0


def export(npz_path, tx_sel, rx_height, want_los_lines, max_rx):
    data = load_npz(npz_path)

    tx_pos = np.asarray(data["tx_positions"], dtype=float)      # (n_tx, 3)
    rx_pos = np.asarray(data["rx_positions"], dtype=float)      # (n_rx, 2)
    rsrp_all = np.asarray(data["rsrp_all"], dtype=float)        # (n_tx, n_rx)
    los_all = np.asarray(data["los_all"], dtype=bool)           # (n_tx, n_rx)

    n_tx = tx_pos.shape[0]
    n_rx = rx_pos.shape[0]

    if tx_sel is None:
        tx_sel = int(data["target_tx_index"][0]) if "target_tx_index" in data else 0
    if not (0 <= tx_sel < n_tx):
        print(f"[ERROR] --tx 범위 초과: 0 ~ {n_tx - 1}")
        sys.exit(1)

    # RX 서브샘플(선택)
    rx_idx = np.arange(n_rx)
    if max_rx is not None and max_rx < n_rx:
        step = int(np.ceil(n_rx / max_rx))
        rx_idx = rx_idx[::step]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = _ts()

    # ---- TX CSV ----------------------------------------------------------
    tx_csv = os.path.join(RESULTS_DIR, f"unreal_tx_{ts}.csv")
    with open(tx_csv, "w", encoding="utf-8") as f:
        f.write("tx_index,x,y,z\n")
        for t in range(n_tx):
            x, y, z = tx_pos[t]
            f.write(f"{t},{x:.4f},{y:.4f},{z:.4f}\n")

    # ---- RX CSV (선택 TX 기준 커버리지) ---------------------------------
    rsrp_row = rsrp_all[tx_sel]
    rsrp_norm, (rsrp_min, rsrp_max) = normalize_rsrp(rsrp_row)
    los_row = los_all[tx_sel]

    rx_csv = os.path.join(RESULTS_DIR, f"unreal_rx_{ts}.csv")
    n_dead = 0
    with open(rx_csv, "w", encoding="utf-8") as f:
        f.write("rx_index,x,y,z,rsrp_dbm,rsrp_norm,los,n_paths\n")
        for r in rx_idx:
            x, y = rx_pos[r]
            z = rx_height
            val = rsrp_row[r]
            if not np.isfinite(val):
                n_dead += 1
                val_str = "-inf"
            else:
                val_str = f"{val:.4f}"
            los = 1 if los_row[r] else 0
            npaths = n_paths_for(data, tx_sel, int(r))
            f.write(f"{r},{x:.4f},{y:.4f},{z:.4f},{val_str},{rsrp_norm[r]:.4f},{los},{npaths}\n")

    # ---- LoS 직선 CSV (옵션) --------------------------------------------
    los_csv = None
    n_los_lines = 0
    if want_los_lines:
        los_csv = os.path.join(RESULTS_DIR, f"unreal_los_lines_{ts}.csv")
        tx_x, tx_y, tx_z = tx_pos[tx_sel]
        with open(los_csv, "w", encoding="utf-8") as f:
            f.write("rx_index,tx_x,tx_y,tx_z,rx_x,rx_y,rx_z,rsrp_dbm\n")
            for r in rx_idx:
                if not los_row[r]:
                    continue
                x, y = rx_pos[r]
                val = rsrp_row[r]
                val_str = "-inf" if not np.isfinite(val) else f"{val:.4f}"
                f.write(f"{r},{tx_x:.4f},{tx_y:.4f},{tx_z:.4f},"
                        f"{x:.4f},{y:.4f},{rx_height:.4f},{val_str}\n")
                n_los_lines += 1

    # ---- 메타 JSON -------------------------------------------------------
    meta = {
        "source_npz": os.path.abspath(npz_path),
        "generated_at": ts,
        "coordinate_system": {
            "units": "meters",
            "axis": "RT native (원본 npz 좌표 그대로)",
            "note": "Unreal 임포트 시 m->cm(x100), 축 방향 확인 필요",
        },
        "counts": {
            "n_tx": n_tx,
            "n_rx_total": n_rx,
            "n_rx_exported": int(len(rx_idx)),
            "n_dead_zone": n_dead,
            "n_los_lines": n_los_lines,
        },
        "selected_tx_index": int(tx_sel),
        "rx_height_assumed_m": rx_height,
        "rsrp_dbm_range": {"min": rsrp_min, "max": rsrp_max},
        "extent_m": {
            "x": [float(rx_pos[:, 0].min()), float(rx_pos[:, 0].max())],
            "y": [float(rx_pos[:, 1].min()), float(rx_pos[:, 1].max())],
        },
        "files": {
            "tx_csv": os.path.basename(tx_csv),
            "rx_csv": os.path.basename(rx_csv),
            "los_lines_csv": os.path.basename(los_csv) if los_csv else None,
        },
        "limitation": "npz에 ray 3D 반사경로 없음 → 커버리지/포인트클라우드 + LoS 직선까지만 export",
    }
    meta_json = os.path.join(RESULTS_DIR, f"unreal_meta_{ts}.json")
    with open(meta_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # ---- 콘솔 요약 -------------------------------------------------------
    print("=" * 64)
    print("  P1V Unreal Export 완료")
    print("=" * 64)
    print(f"  입력      : {npz_path}")
    print(f"  기준 TX   : {tx_sel}  (위치 {tx_pos[tx_sel].tolist()})")
    print(f"  TX 수     : {n_tx}")
    print(f"  RX export : {len(rx_idx)} / {n_rx}  (dead zone {n_dead})")
    print(f"  RSRP 범위 : {rsrp_min:.2f} ~ {rsrp_max:.2f} dBm")
    if want_los_lines:
        print(f"  LoS 직선  : {n_los_lines}개")
    print("-" * 64)
    print(f"  → {tx_csv}")
    print(f"  → {rx_csv}")
    if los_csv:
        print(f"  → {los_csv}")
    print(f"  → {meta_json}")
    print("=" * 64)


def main():
    p = argparse.ArgumentParser(description="RT npz → Unreal 시각화 CSV/JSON export")
    p.add_argument("--npz", default=DEFAULT_NPZ, help="입력 npz 경로")
    p.add_argument("--tx", type=int, default=None, help="커버리지 기준 TX 인덱스")
    p.add_argument("--rx-height", type=float, default=RX_HEIGHT_DEFAULT,
                   help="RX 높이 가정값 [m] (npz에 z 없음)")
    p.add_argument("--los-lines", action="store_true", help="LoS 쌍 TX→RX 직선 CSV 생성")
    p.add_argument("--max-rx", type=int, default=None, help="RX 서브샘플 상한")
    args = p.parse_args()

    export(args.npz, args.tx, args.rx_height, args.los_lines, args.max_rx)


if __name__ == "__main__":
    main()
