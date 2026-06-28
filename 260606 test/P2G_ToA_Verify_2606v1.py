# -*- coding: utf-8 -*-
"""
P2G_ToA_Verify_2606v1.py
========================
[검증] normalize_delays=False 면 첫 도착 지연(ToA)이 실제 거리/c 와 맞는가?

목적
  P1A 수정본(P1A_RT_to_Rays_ToA: normalize_delays=False)을 전체 재생성하기 전에,
  소수 RX 만 RT 돌려서 "첫 도착 tau × c ≈ TX-RX 3D 거리" 를 직접 확인.
  (LoS RX: c·tau_first ≈ 3D거리, NLoS RX: c·tau_first ≥ 3D거리)

설정 (원본 area_1 와 동일)
  scene  : 250912 py_stable/Jonggak/Jonggak.xml
  TX     : [-51.561, -21.794, 19], 7.5 GHz, 1x1 iso/V
  RX grid: x∈[-136.138,58.862]×40, y∈[-117.667,77.333]×40, z=1.5 (일부만 샘플)
  PathSolver: max_depth=5, seed=41 (원본과 동일)

주의
  - GPU 초기화에서 import 멈춤 사례가 있어 CPU(CUDA_VISIBLE_DEVICES="")로 실행 권장.
  - 소수 RX(기본 16개)만 사용 -> 빠른 검증.
----------------------------------------------------------------------
실행 환경 : sionna 1.2.1 (venv: /home/dclserver78/sionna_0310/venv)
  CUDA_VISIBLE_DEVICES="" ./venv/bin/python P2G_ToA_Verify_2606v1.py
----------------------------------------------------------------------
"""

import os
import datetime

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

for _fp in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",):
    if os.path.exists(_fp):
        fm.fontManager.addfont(_fp)
        plt.rcParams["font.family"] = fm.FontProperties(fname=_fp).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, PathSolver
from sionna.phy import SPEED_OF_LIGHT

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WS_ROOT  = os.path.dirname(BASE_DIR)
SCENE    = os.path.join(WS_ROOT, "250912 py_stable", "Jonggak", "Jonggak.xml")
OUT_DIR  = os.path.join(BASE_DIR, "P2G_ToA_Verify_Results")

TX_POS = [-51.561, -21.794, 19.0]
FREQ_HZ = 7.5e9
MAX_DEPTH = 5
RT_SEED = 41
GRID_N = 40
RX_X = np.linspace(-136.138, 58.862, GRID_N)
RX_Y = np.linspace(-117.667, 77.333, GRID_N)
RX_Z = 1.5
N_SAMPLE_RX = 16


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_f = open(os.path.join(OUT_DIR, f"P2G_toa_verify_report_{stamp}.log"),
                 "w", encoding="utf-8")

    def log(*a):
        m = " ".join(str(x) for x in a); print(m, flush=True); log_f.write(m + "\n"); log_f.flush()

    try:
        c = float(SPEED_OF_LIGHT)
        log(f"[cfg] scene={SCENE}\n      TX={TX_POS}, f={FREQ_HZ/1e9}GHz, "
            f"max_depth={MAX_DEPTH}, c={c:.3f}")

        # RX 후보: 전체 그리드에서 거리 폭이 고르게 나오도록 N개 추출
        gx, gy = np.meshgrid(RX_X, RX_Y, indexing="xy")
        rx_xy = np.stack([gx.ravel(), gy.ravel()], 1)
        rx3d = np.column_stack([rx_xy, np.full(len(rx_xy), RX_Z)])
        dist3d_all = np.linalg.norm(rx3d - np.array(TX_POS), axis=1)
        order = np.argsort(dist3d_all)
        pick = order[np.linspace(0, len(order) - 1, N_SAMPLE_RX).astype(int)]
        rx3d = rx3d[pick]
        dist3d = dist3d_all[pick]
        log(f"[rx] {N_SAMPLE_RX}개 추출, 3D거리[m] {dist3d.min():.1f}~{dist3d.max():.1f}")

        # scene/안테나/TX/RX
        scene = load_scene(SCENE, merge_shapes=False)
        scene.frequency = FREQ_HZ
        arr = PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0.5,
                          horizontal_spacing=0.5, pattern="iso", polarization="V")
        scene.tx_array = arr
        scene.rx_array = arr
        scene.add(Transmitter(name="tx", position=TX_POS))
        for i, p in enumerate(rx3d):
            scene.add(Receiver(name=f"rx{i}", position=[float(p[0]), float(p[1]), float(p[2])]))
        log(f"[scene] objects={len(scene.objects)}, RX={len(rx3d)} 추가")

        # RT (원본과 동일 옵션) + 절대지연
        solver = PathSolver()
        paths = solver(scene=scene, max_depth=MAX_DEPTH, los=True,
                       specular_reflection=True, diffuse_reflection=True,
                       refraction=True, synthetic_array=False, seed=RT_SEED)
        a_, tau = paths.cir(normalize_delays=False, out_type="numpy")
        a_ = np.asarray(a_); tau = np.asarray(tau)
        log(f"[rt] a_ shape={a_.shape}, tau shape={tau.shape}")

        # tau -> [num_rx, num_paths] 로 정리 (단일 안테나/단일 TX)
        tau_r = np.squeeze(tau)            # (num_rx, num_paths) 기대
        pwr = (np.abs(np.squeeze(a_)) ** 2)
        if tau_r.ndim == 1:
            tau_r = tau_r[None, :]; pwr = pwr[None, :]
        # 유효 경로: tau 유한 & >0 & power>0 (무효는 보통 -1 또는 inf)
        valid = np.isfinite(tau_r) & (tau_r > 0) & (pwr > 0)
        toa = np.full(len(rx3d), np.nan)
        for i in range(len(rx3d)):
            if valid[i].any():
                toa[i] = tau_r[i][valid[i]].min()
        d_toa = c * toa                    # 첫 도착 -> 거리 추정[m]

        ok = np.isfinite(d_toa)
        diff = d_toa - dist3d
        # LoS 판정: c·tau_first ≈ 3D거리 (1m 이내)
        is_los = np.abs(diff) < 1.0
        log("\n[검증 표] rx | 3D거리[m] | c·tau_first[m] | 차이[m] | 판정")
        for i in range(len(rx3d)):
            tag = "LoS✓" if (ok[i] and is_los[i]) else ("NLoS(반사)" if ok[i] else "경로없음")
            log(f"  rx{i:<2} {dist3d[i]:8.2f}   "
                f"{d_toa[i]:10.2f}   {diff[i]:8.2f}   {tag}")
        if ok.sum() >= 2:
            cc = np.corrcoef(d_toa[ok], dist3d[ok])[0, 1]
            log(f"\n[요약] corr(c·tau_first, 3D거리)={cc:.4f}  "
                f"(LoS RX {is_los[ok].sum()}/{ok.sum()})")
            los_ok = ok & is_los
            if los_ok.any():
                log(f"  LoS RX: |c·tau - 거리| 평균={np.abs(diff[los_ok]).mean():.3f}m, "
                    f"최대={np.abs(diff[los_ok]).max():.3f}m  -> ToA=거리 확인")
            log(f"  tau_first 중앙값={np.nanmedian(toa)*1e9:.1f}ns (≠0 이면 절대지연 복원)")

        # 그래프
        fig, ax = plt.subplots(figsize=(7, 6.5))
        m = ok
        cols = np.where(is_los[m], "tab:blue", "tab:red")
        ax.scatter(dist3d[m], d_toa[m], c=cols, s=60, zorder=3)
        lim = max(dist3d.max(), np.nanmax(d_toa)) * 1.05
        ax.plot([0, lim], [0, lim], "k--", lw=1.3, label="1:1 (ToA=거리)")
        ax.set_xlabel("실제 TX-RX 3D 거리 [m]")
        ax.set_ylabel("c · tau_first [m]")
        ax.set_title("normalize_delays=False 검증\n파랑=LoS, 빨강=NLoS(첫 도착이 반사)")
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.grid(True, ls=":", alpha=0.4); ax.legend(loc="upper left")
        fig.tight_layout()
        png = os.path.join(OUT_DIR, f"P2G_toa_verify_{stamp}.png")
        fig.savefig(png, dpi=150); plt.close(fig)
        log(f"\n[viz] saved -> {png}")
        log("[done] ToA 검증 완료")
    finally:
        log_f.close()


if __name__ == "__main__":
    main()
