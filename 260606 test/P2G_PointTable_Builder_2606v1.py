# -*- coding: utf-8 -*-
"""
P2G_PointTable_Builder_2606v1.py
================================
[단계 1] 점 단위 표준 테이블(point table) 빌더 + 점검 리포트.

산출물 2가지
  (A) 점 테이블 npz : 단계 2~5 가 이 파일 하나만 입력으로 받도록 표준화
        pos        (N,2)  float64   UE 위치 (x,y) [m]   -> d_phys 계산용
        rsrp       (N,)   float64   10*log10(sum power) [dB]
        tau_raw    (N,P)  float32   경로별 도착시간 [s]  (point-mass W1 용)
        pwr_raw    (N,P)  float32   경로별 선형 전력      (power==0 = 무효 경로)
        npaths     (N,)   int32     유효 경로 수(pwr>0)
        cov_factor (N,M,r) complex64  R_BS 저랭크 인수 B_i=U_iΛ_i^{1/2}
                                       (Ã=A/tr(A)+εI 기준, 상위 r 고유쌍)
        cov_trace  (N,)   float64   tr(Ã_i) (=1+εM, d_BW^2 계산용)
        rx_idx     (N,)   int64     원본 RX 인덱스
        + 메타(area, freq, grid, M, r, eps)
  (B) 점검 리포트(.log, 타임스탬프) : 아래 ①②③ + shape/NaN/정렬 확인

위치 환원 sanity check
  ① BS 거리 범위가 물리적으로 타당한지(수 m~수백 m)
  ② 최근접 UE-UE d_phys 히스토그램이 격자 간격(약 5 m)과 일치하는지
  ③ 알려진 점쌍 좌표 직접 확인 — ix 한 칸 차이=X축 5 m, iy 한 칸 차이=Y축 5 m
     (히스토그램만으론 못 잡는 ix<->iy 축 뒤바뀜 탐지)

메모리 전략 (필수)
  1037개 R_BS(1024x1024, 동시적재 ~8.7GB)를 모두 메모리에 올리지 않는다.
  파일을 하나씩 열어(스트리밍) 상위 r 고유쌍 인수 B_i(1024xr, complex64 ~0.13MB)만
  보관하고 원본은 폐기. -> 약 1037 x 1024 x r x 8B. r=32 기준 약 0.27GB.

GPU 전략 (작업 지침: 모든 GPU 사용)
  trace 정규화 + 배치 eigh -> 상위 r 추출을 P2G_gpu_utils.batched_topr_eigh_factor
  로 위임. 서버의 모든 CUDA GPU(H100 x2)에 샤딩(디바이스당 스레드, 디스크 I/O는
  CPU, 수치연산은 GPU). GPU 없으면 자동 CPU 폴백.
  주의) 샌드박스가 GPU 접근을 막으므로 full-permission(샌드박스 해제)로 실행.

----------------------------------------------------------------------
실행 환경
  - Python   : 3.10.12
  - 실행 서버 : dclserver78 (twin_minji workspace)
  - 주요 라이브러리 : numpy 2.2.6 / scipy 1.15.3
입력 데이터(읽기 전용)
  - 251009_CCM_Collection (8 GB)/P1A_RT_Results/Area1_7.5GHz_Rays_ALL_RXs.npz
  - 251009_CCM_Collection (8 GB)/P1F_Marginal_CCM_Results/*.npz (R_BS 1024x1024)
----------------------------------------------------------------------
"""

import os
import re
import glob
import time
import datetime

import numpy as np

from P2G_gpu_utils_2606v1 import get_devices, batched_topr_eigh_factor


# ============================================================
# 0. 설정
# ============================================================
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
WS_ROOT   = os.path.dirname(BASE_DIR)
DATA_ROOT = os.path.join(WS_ROOT, "251009_CCM_Collection (8 GB)")
RAYS_NPZ  = os.path.join(DATA_ROOT, "P1A_RT_Results",
                         "Area1_7.5GHz_Rays_ALL_RXs.npz")
P1F_DIR   = os.path.join(DATA_ROOT, "P1F_Marginal_CCM_Results")
RESULT_DIR = os.path.join(BASE_DIR, "P2G_PointTable_Results")

# Area1 grid (P1A_RT_to_Rays_2509v6.py AREA_CONFIGS['area_1'])
GRID_N = 40
RX_X   = np.linspace(-136.138, 58.862, GRID_N)   # step = 195/39 = 5.0 m
RX_Y   = np.linspace(-117.667, 77.333, GRID_N)
BS_XY  = np.array([-51.561, -21.794])
GRID_STEP = float(RX_X[1] - RX_X[0])             # 약 5.0 m

# 하이퍼파라미터
RANK      = 96        # R_BS 저랭크 인수 차원
                      # 2026-06-06 32->96: P2G_dCov_RankCheck 결과 r=32는 가까운
                      # 쌍 d_Cov 최대오차 26% -> r=96에서 최대 0.91%(<1%)로 확정
MIN_PATHS = 20        # 최소 유효 경로 수 필터
EPS       = 1e-6      # Ã = A/tr(A) + εI
SEED      = 42


# ============================================================
# 보조: 로그 동시 출력
# ============================================================
class Tee:
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")
    def __call__(self, *args):
        msg = " ".join(str(a) for a in args)
        print(msg)
        self.f.write(msg + "\n")
        self.f.flush()
    def close(self):
        self.f.close()


# ============================================================
# 1. RX 교집합 + 인덱스 매핑
# ============================================================
def build_rx_join(log):
    d = np.load(RAYS_NPZ, allow_pickle=True)
    rx_ind = np.asarray(d["rx_indices"], dtype=np.int64)   # (1600,)
    row_of = {int(rx): i for i, rx in enumerate(rx_ind)}

    files = sorted(glob.glob(os.path.join(P1F_DIR, "*.npz")))
    rx_file = {}
    for f in files:
        m = re.search(r"RX(\d+)_", os.path.basename(f))
        if m:
            rx_file[int(m.group(1))] = f

    common = sorted(set(row_of) & set(rx_file))
    log(f"[join] P1A rx={len(row_of)}, P1F files={len(rx_file)}, "
        f"교집합 N={len(common)}, P1F-not-in-P1A={len(set(rx_file)-set(row_of))}")
    return d, row_of, rx_file, np.array(common, dtype=np.int64)


# ============================================================
# 2. 점 단위 RSRP / PDP-raw / 위치 (P1A)
# ============================================================
def build_point_scalars(d, row_of, rx_common, log):
    rows = np.array([row_of[int(r)] for r in rx_common])
    P = d["power"].shape[-1]
    pwr = np.asarray(d["power"]).reshape(d["power"].shape[0], -1)[rows].astype(np.float32)
    tau = np.asarray(d["tau"]).reshape(d["tau"].shape[0], -1)[rows].astype(np.float32)

    valid = pwr > 0
    npaths = valid.sum(1).astype(np.int32)

    # RSRP = 총 수신전력(dB). 무효 경로(pwr==0) 제외
    psum = np.where(valid, pwr, 0.0).sum(1)
    rsrp = 10.0 * np.log10(np.clip(psum, 1e-30, None))

    # 위치 환원: rx_indices 1-based -> f=idx-1, ix=f%40, iy=(f//40)%40
    f = rx_common - 1
    ix = f % GRID_N
    iy = (f // GRID_N) % GRID_N
    pos = np.stack([RX_X[ix], RX_Y[iy]], axis=1)

    log(f"[point] N={len(rx_common)}, P(max paths)={P}, "
        f"npaths[min/med/max]={npaths.min()}/{int(np.median(npaths))}/{npaths.max()}")
    return dict(pos=pos, rsrp=rsrp, tau_raw=tau, pwr_raw=pwr, npaths=npaths,
                ix=ix, iy=iy), rows


# ============================================================
# 3. R_BS 저랭크 인수분해 (모든 GPU 샤딩 + 스트리밍 I/O)
# ============================================================
def stream_cov_factors(rx_file, rx_common, M_expect, devices, log):
    N = len(rx_common)

    def load_fn(i):
        e = np.load(rx_file[int(rx_common[i])], allow_pickle=True)
        return np.asarray(e["R_BS"]).astype(np.complex64)

    B, tr, tail = batched_topr_eigh_factor(
        load_fn, N, M_expect, RANK, EPS, devices, chunk=48, log=log)
    return B, tr, tail, M_expect


# ============================================================
# 4. 위치 환원 sanity check (①②③)
# ============================================================
def sanity_checks(pt, rx_common, log):
    pos, ix, iy = pt["pos"], pt["ix"], pt["iy"]

    # ① BS 거리 범위
    rad = np.sqrt(((pos - BS_XY) ** 2).sum(1))
    log("\n[① BS 거리 범위]")
    log(f"   radius[min/med/max] = {rad.min():.2f}/{np.median(rad):.2f}/"
        f"{rad.max():.2f} m  (grid 대각선 ~{np.hypot(np.ptp(RX_X), np.ptp(RX_Y)):.1f} m)")

    # ② 최근접 UE-UE d_phys 히스토그램 (격자 간격 ~5 m 기대)
    #    메모리 위해 청크로 최근접만 계산
    N = pos.shape[0]
    nn = np.full(N, np.inf)
    CH = 256
    for c0 in range(0, N, CH):
        c1 = min(N, c0 + CH)
        dd = np.sqrt(((pos[c0:c1, None, :] - pos[None, :, :]) ** 2).sum(2))
        for k in range(c1 - c0):
            dd[k, c0 + k] = np.inf
        nn[c0:c1] = dd.min(1)
    log("\n[② 최근접 UE-UE 거리]")
    log(f"   nn d_phys[min/med/max] = {nn.min():.3f}/{np.median(nn):.3f}/"
        f"{nn.max():.3f} m  (격자 step={GRID_STEP:.3f} m)")
    edges = [0, GRID_STEP*0.5, GRID_STEP*1.5, GRID_STEP*2.5, GRID_STEP*5, np.inf]
    hist, _ = np.histogram(nn, bins=edges)
    log(f"   히스토그램 bins(step 배수) {['~0.5','~1.5','~2.5','~5','>5']}: "
        f"{hist.tolist()}")
    if abs(np.median(nn) - GRID_STEP) > 0.5:
        log("   [경고] 최근접 중앙값이 격자 step과 어긋남 -> 환원/단위 점검 필요")

    # ③ 알려진 점쌍 좌표 직접 확인 (축 뒤바뀜)
    log("\n[③ 축 매핑 직접 확인]")
    # base 점 하나 고르고, ix+1(같은 iy) / iy+1(같은 ix) 이웃을 실제 점에서 탐색
    key_ixiy = {(int(a), int(b)): k for k, (a, b) in enumerate(zip(ix, iy))}
    base = None
    for k in range(N):
        a, b = int(ix[k]), int(iy[k])
        if (a + 1, b) in key_ixiy and (a, b + 1) in key_ixiy:
            base = k; break
    if base is None:
        log("   [경고] ix+1, iy+1 이웃이 모두 존재하는 base 점을 못 찾음(부분집합 희소)")
    else:
        a, b = int(ix[base]), int(iy[base])
        kx = key_ixiy[(a + 1, b)]
        ky = key_ixiy[(a, b + 1)]
        dx = pos[kx] - pos[base]
        dy = pos[ky] - pos[base]
        log(f"   base(rx={rx_common[base]}) ix,iy=({a},{b}) pos={pos[base].round(3)}")
        log(f"   ix+1 이웃(rx={rx_common[kx]}): Δ={dx.round(3)} "
            f"-> X축 {dx[0]:+.2f}m, Y축 {dx[1]:+.2f}m  "
            f"({'OK: ix→X축' if abs(dx[0]-GRID_STEP)<0.5 and abs(dx[1])<0.5 else '확인!'})")
        log(f"   iy+1 이웃(rx={rx_common[ky]}): Δ={dy.round(3)} "
            f"-> X축 {dy[0]:+.2f}m, Y축 {dy[1]:+.2f}m  "
            f"({'OK: iy→Y축' if abs(dy[1]-GRID_STEP)<0.5 and abs(dy[0])<0.5 else '확인!'})")
    return rad, nn


# ============================================================
# main
# ============================================================
def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log = Tee(os.path.join(RESULT_DIR, f"P2G_pointtable_report_{stamp}.log"))
    np.random.seed(SEED)
    try:
        devices = get_devices()
        log(f"[env] numpy {np.__version__}, RANK={RANK}, EPS={EPS}, "
            f"MIN_PATHS={MIN_PATHS}, stamp={stamp}")
        log(f"[env] devices = {[str(d) for d in devices]}")

        # 메타: R_BS 가 BS-side 인지 확인용 n_t 확보
        d, row_of, rx_file, rx_common = build_rx_join(log)
        meta0 = np.load(rx_file[int(rx_common[0])], allow_pickle=True)
        md = meta0["metadata"].item()
        n_t, n_r = int(md["n_t"]), int(md["n_r"])
        rbs_shape = np.asarray(meta0["R_BS"]).shape
        rue_shape = np.asarray(meta0["R_UE"]).shape
        log(f"[verify] metadata n_t={n_t}(=R_BS dim {rbs_shape[0]}? "
            f"{n_t==rbs_shape[0]}), n_r={n_r}(=R_UE dim {rue_shape[0]}? "
            f"{n_r==rue_shape[0]}) -> R_BS=TX(BS)-side {'확인' if n_t==rbs_shape[0]==1024 else '재확인필요'}")

        pt, rows = build_point_scalars(d, row_of, rx_common, log)

        # MIN_PATHS 필터
        keep = np.where(pt["npaths"] >= MIN_PATHS)[0]
        if len(keep) < len(rx_common):
            log(f"[filter] npaths>={MIN_PATHS}: {len(rx_common)} -> {len(keep)} 점 유지")
        rx_common = rx_common[keep]
        for k in ("pos", "rsrp", "tau_raw", "pwr_raw", "npaths", "ix", "iy"):
            pt[k] = pt[k][keep]

        # 공분산 저랭크 인수분해 (모든 GPU)
        B, tr, tail, M = stream_cov_factors(rx_file, rx_common, n_t, devices, log)

        # NaN/inf 점검
        log("\n[점검: NaN/inf/정렬]")
        log(f"   pos finite={np.isfinite(pt['pos']).all()}, "
            f"rsrp finite={np.isfinite(pt['rsrp']).all()}, "
            f"cov_factor finite={np.isfinite(B).all()}")
        log(f"   rx_idx 정렬(오름차순)={bool(np.all(np.diff(rx_common)>0))}")
        log(f"   shapes: pos{pt['pos'].shape} rsrp{pt['rsrp'].shape} "
            f"tau_raw{pt['tau_raw'].shape} pwr_raw{pt['pwr_raw'].shape} "
            f"cov_factor{B.shape} cov_trace{tr.shape}")

        # 위치 sanity check
        rad, nn = sanity_checks(pt, rx_common, log)

        # 저장 (덮어쓰기 금지: 타임스탬프)
        out = os.path.join(RESULT_DIR, f"P2G_PointTable_{stamp}.npz")
        np.savez_compressed(
            out,
            pos=pt["pos"], rsrp=pt["rsrp"], tau_raw=pt["tau_raw"],
            pwr_raw=pt["pwr_raw"], npaths=pt["npaths"],
            cov_factor=B, cov_trace=tr, cov_tail=tail,
            rx_idx=rx_common,
            area_idx=int(md["area_idx"]), freq_ghz=float(md["freq_ghz"]),
            grid_n=GRID_N, grid_step=GRID_STEP, bs_xy=BS_XY,
            M=M, rank=RANK, eps=EPS, min_paths=MIN_PATHS)
        log(f"\n[save] point table -> {out}  ({os.path.getsize(out)/1e6:.1f} MB)")
        log(f"[done] N={len(rx_common)} 점, cov_factor {B.shape} 보관")
    finally:
        log.close()


if __name__ == "__main__":
    main()
