"""
check_npz.py
============
channel_data_*.npz 파일 내용을 빠르게 확인하는 유틸리티.

사용법:
  python3 check_npz.py                          # ← DEFAULT_NPZ_PATH 사용
  python3 check_npz.py <npz_파일_경로> [옵션]

옵션:
  --rx <인덱스>     특정 RX의 상세 데이터 출력 (tau, power, aoa)
  --keys            키 목록과 shape만 출력
  --rsrp-plot       RSRP 분포를 터미널 바 차트로 출력

예시:
  python3 check_npz.py
  python3 check_npz.py 2-1.RayTracingAgent/output_GHMTwin2/channel_data_GHMTwin2_cutting.npz
  python3 check_npz.py output/channel_data_map.npz --rx 5
  python3 check_npz.py output/channel_data_map.npz --keys
"""

import argparse
import sys
import os
import numpy as np

# ============================================================
# ★ 기본 확인 대상 파일 — 여기만 바꾸면 인수 없이 바로 실행 가능
# ============================================================
DEFAULT_NPZ_PATH = "2-1.RayTracingAgent/output_GHMTwin2/channel_data_GHMTwin2_cutting.npz"


# ============================================================
# 헬퍼
# ============================================================

def load(path: str):
    if not os.path.exists(path):
        print(f"❌ 파일 없음: {path}")
        sys.exit(1)
    return np.load(path, allow_pickle=True)


def _rsrp_bar(value: float, min_v: float, max_v: float, width: int = 30) -> str:
    """RSRP 값을 터미널 바 차트로 변환."""
    if np.isinf(value):
        return "[" + " " * width + "] -inf"
    ratio = (value - min_v) / (max_v - min_v + 1e-9)
    filled = int(ratio * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {value:7.2f} dBm"


# ============================================================
# 출력 모드
# ============================================================

def show_keys(data):
    """키 목록 + shape + dtype만 출력."""
    print(f"\n{'키':<25} {'shape':<20} {'dtype'}")
    print("-" * 60)
    for k in sorted(data.keys()):
        arr = data[k]
        print(f"  {k:<23} {str(arr.shape):<20} {arr.dtype}")
    print(f"\n총 {len(data.keys())}개 키")


def show_summary(data):
    """전체 RX 요약 테이블 출력 (기본 모드)."""
    rsrp = data["rsrp_all"]
    num_rx = len(rsrp)
    target_idx = int(data["target_rx_index"][0]) if "target_rx_index" in data else 0
    rx_positions = data["rx_positions"] if "rx_positions" in data else None

    valid_rsrp = rsrp[~np.isinf(rsrp)]
    dead_count = int(np.sum(np.isinf(rsrp)))

    print("\n" + "=" * 70)
    print(f"  📦 파일 요약")
    print("=" * 70)
    print(f"  총 RX 수       : {num_rx}개")
    print(f"  Dead Zone      : {dead_count}개")
    print(f"  유효 RX        : {num_rx - dead_count}개")
    print(f"  target_rx_index: {target_idx}")
    if len(valid_rsrp) > 0:
        print(f"  RSRP 범위      : {valid_rsrp.min():.2f} ~ {valid_rsrp.max():.2f} dBm")
        print(f"  RSRP 평균      : {valid_rsrp.mean():.2f} dBm")
    print("=" * 70)

    print(f"\n{'RX':>4}  {'RSRP (dBm)':>12}  {'유효경로':>8}  {'위치 (x, y)':>24}  {'비고'}")
    print("-" * 70)

    for i in range(num_rx):
        rsrp_val = rsrp[i]
        rsrp_str = f"{rsrp_val:8.2f}" if not np.isinf(rsrp_val) else "   -inf "

        # 유효경로 수
        tau_key = f"tau_rx{i}"
        n_paths = len(data[tau_key]) if tau_key in data else 0

        # 위치
        if rx_positions is not None and i < len(rx_positions):
            pos = rx_positions[i]
            pos_str = f"({pos[0]:8.2f}, {pos[1]:8.2f})"
        else:
            pos_str = "N/A"

        tag = " ← target" if i == target_idx else ""
        dead_tag = " [DEAD]" if np.isinf(rsrp_val) else ""

        print(f"  RX{i:>2}  {rsrp_str} dBm  {n_paths:>8}개  {pos_str:>24}{tag}{dead_tag}")

    print()


def show_rx_detail(data, rx_idx: int):
    """특정 RX의 상세 데이터 출력."""
    rsrp = data["rsrp_all"]
    num_rx = len(rsrp)

    if rx_idx < 0 or rx_idx >= num_rx:
        print(f"❌ RX 인덱스 범위 초과: 0 ~ {num_rx - 1}")
        sys.exit(1)

    tau_key   = f"tau_rx{rx_idx}"
    power_key = f"power_rx{rx_idx}"
    aoa_key   = f"aoa_rx{rx_idx}"
    R_TX_key  = f"R_TX_rx{rx_idx}"
    R_RX_key  = f"R_RX_rx{rx_idx}"

    print("\n" + "=" * 60)
    print(f"  📡 RX {rx_idx} 상세 정보")
    print("=" * 60)

    # 위치
    if "rx_positions" in data and rx_idx < len(data["rx_positions"]):
        pos = data["rx_positions"][rx_idx]
        print(f"  위치       : ({pos[0]:.4f}, {pos[1]:.4f})")

    # RSRP
    rsrp_val = rsrp[rx_idx]
    print(f"  RSRP       : {rsrp_val:.4f} dBm" if not np.isinf(rsrp_val) else "  RSRP       : -inf (Dead Zone)")

    if tau_key not in data:
        print(f"  ⚠️  tau_rx{rx_idx} 키 없음")
        return

    tau   = data[tau_key]
    power = data[power_key] if power_key in data else None
    aoa   = data[aoa_key]   if aoa_key   in data else None

    print(f"  유효경로 수: {len(tau)}개")
    print()

    # 경로별 상세 (최대 20개)
    max_show = 20
    print(f"  {'경로':>5}  {'지연 (ns)':>12}  {'전력 (dBm)':>12}  {'AoA (deg)':>12}")
    print("  " + "-" * 50)
    for j in range(min(len(tau), max_show)):
        tau_v   = f"{tau[j]:12.4f}"
        pow_v   = f"{power[j]:12.4f}" if power is not None else "         N/A"
        aoa_v   = f"{aoa[j]:12.4f}"   if aoa   is not None else "         N/A"
        print(f"  {j:>5}  {tau_v}  {pow_v}  {aoa_v}")

    if len(tau) > max_show:
        print(f"  ... (총 {len(tau)}개 중 {max_show}개만 표시)")

    # 공분산 행렬
    print()
    if R_TX_key in data:
        print(f"  R_TX shape : {data[R_TX_key].shape}")
        print(f"  R_TX       :\n{data[R_TX_key]}")
    if R_RX_key in data:
        print(f"  R_RX shape : {data[R_RX_key].shape}")
        print(f"  R_RX       :\n{data[R_RX_key]}")
    print()


def show_rsrp_plot(data):
    """RSRP 분포를 터미널 바 차트로 출력."""
    rsrp = data["rsrp_all"]
    target_idx = int(data["target_rx_index"][0]) if "target_rx_index" in data else 0

    valid_rsrp = rsrp[~np.isinf(rsrp)]
    if len(valid_rsrp) == 0:
        print("유효한 RSRP 값이 없습니다.")
        return

    min_v, max_v = valid_rsrp.min(), valid_rsrp.max()

    print("\n" + "=" * 65)
    print("  📊 RSRP 분포 (터미널 바 차트)")
    print("=" * 65)
    print(f"  범위: {min_v:.2f} ~ {max_v:.2f} dBm\n")

    for i, v in enumerate(rsrp):
        tag = " ← target" if i == target_idx else ""
        dead = " [DEAD]" if np.isinf(v) else ""
        bar = _rsrp_bar(v, min_v, max_v) if not np.isinf(v) else "[" + " " * 30 + "] -inf"
        print(f"  RX{i:>2}  {bar}{tag}{dead}")
    print()


# ============================================================
# 진입점
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="channel_data_*.npz 파일 내용 확인 유틸리티",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("path", nargs="?", default=None,
                        help=f"npz 파일 경로 (생략 시 DEFAULT_NPZ_PATH 사용)")
    parser.add_argument("--rx",        type=int, default=None, help="특정 RX 상세 출력")
    parser.add_argument("--keys",      action="store_true",    help="키 목록 + shape만 출력")
    parser.add_argument("--rsrp-plot", action="store_true",    help="RSRP 바 차트 출력")

    args = parser.parse_args()

    # 경로 결정: CLI 인수 > DEFAULT_NPZ_PATH
    target_path = args.path if args.path else DEFAULT_NPZ_PATH
    if not args.path:
        print(f"ℹ️  경로 미지정 → DEFAULT_NPZ_PATH 사용: {DEFAULT_NPZ_PATH}")

    data = load(target_path)
    print(f"\n📂 {os.path.abspath(target_path)}")
    print(f"   파일 크기: {os.path.getsize(target_path) / 1024:.1f} KB")

    if args.keys:
        show_keys(data)
    elif args.rx is not None:
        show_summary(data)
        show_rx_detail(data, args.rx)
    elif args.rsrp_plot:
        show_summary(data)
        show_rsrp_plot(data)
    else:
        # 기본: 요약 + 바 차트
        show_summary(data)
        show_rsrp_plot(data)


if __name__ == "__main__":
    main()
