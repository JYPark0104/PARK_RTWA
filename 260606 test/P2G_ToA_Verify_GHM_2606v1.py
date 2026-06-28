# -*- coding: utf-8 -*-
"""
P2G_ToA_Verify_GHM_2606v1.py
============================
[검증] normalize_delays=False 면 첫 도착 지연(ToA)=거리/c 인가? (GHM Twin scene)

배경
  - Jonggak scene 은 mesh 104개 누락으로 RT 불가.
  - 기존 두 데이터셋(P1A Area1, GHM channel_data) 모두 normalize_delays=True 라
    첫 탭=0(excess) → 기존 데이터로는 ToA 검증 불가.
  - GHM Twin 은 단일 통합 ply(환경 전체)가 있고 좌표계가 tx/rx 위치와 정합.
    → 최소 Mitsuba xml 로 감싸 RT 직접 실행, normalize_delays=False 로 ToA 확인.
    (ToA/LoS 는 지오메트리만 정확하면 됨; 재질은 직선경로 지연에 무관)

데이터/좌표
  ply : 260604 GHM_test/GHM Twin_v0.1.ply  (bbox x[-928,1373] y[-885,906] z[0,160])
  TX  : channel_data tx_positions (3개), RX : rx_positions 에서 샘플 (z=1.5 가정)

검증
  각 (TX,RX): c·tau_first vs ||TX - (rx_xy, 1.5)||_3D
  LoS RX: 일치(≈0 차이), NLoS RX: c·tau_first ≥ 거리.
----------------------------------------------------------------------
실행 : CUDA_VISIBLE_DEVICES="" /home/dclserver78/sionna_0310/venv/bin/python ...
  (GPU 초기화 hang 회피 위해 CPU 실행)
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
GHM_DIR  = os.path.join(WS_ROOT, "260604 GHM_test")
PLY      = os.path.join(GHM_DIR, "GHM Twin_v0.1.ply")
CH_NPZ   = os.path.join(GHM_DIR, "channel_data_260531_GHM_Twin_v0_1.npz")
OUT_DIR  = os.path.join(BASE_DIR, "P2G_ToA_Verify_Results")

FREQ_HZ = 7.5e9
MAX_DEPTH = 5
RT_SEED = 41
RX_Z = 1.5
N_SAMPLE_RX = 12


def build_min_xml(ply_path, out_xml):
    xml = f"""<scene version="2.1.0">
  <bsdf type="diffuse" id="mat-itu_concrete" name="mat-itu_concrete">
    <rgb value="0.8 0.8 0.8" name="reflectance"/>
  </bsdf>
  <shape type="ply" id="env" name="env">
    <string name="filename" value="{ply_path}"/>
    <boolean name="face_normals" value="true"/>
    <ref id="mat-itu_concrete" name="bsdf"/>
  </shape>
</scene>
"""
    with open(out_xml, "w", encoding="utf-8") as f:
        f.write(xml)
    return out_xml


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_f = open(os.path.join(OUT_DIR, f"P2G_toa_verify_GHM_report_{stamp}.log"),
                 "w", encoding="utf-8")

    def log(*a):
        m = " ".join(str(x) for x in a); print(m, flush=True); log_f.write(m + "\n"); log_f.flush()

    try:
        c = float(SPEED_OF_LIGHT)
        d = np.load(CH_NPZ, allow_pickle=True)
        txp = np.asarray(d["tx_positions"], dtype=float)         # (3,3)
        rxp = np.asarray(d["rx_positions"], dtype=float)         # (Nall,2)
        log(f"[cfg] ply={PLY}\n      TX={txp.tolist()}, f={FREQ_HZ/1e9}GHz, depth={MAX_DEPTH}")

        # RX 샘플: TX0 기준 3D 거리 폭이 고르게
        rx3d_all = np.column_stack([rxp, np.full(len(rxp), RX_Z)])
        dist0 = np.linalg.norm(rx3d_all - txp[0], axis=1)
        order = np.argsort(dist0)
        pick = order[np.linspace(0, len(order) - 1, N_SAMPLE_RX).astype(int)]
        rx3d = rx3d_all[pick]
        log(f"[rx] {N_SAMPLE_RX}개, TX0 3D거리[m] {dist0[pick].min():.1f}~{dist0[pick].max():.1f}")

        xml = build_min_xml(PLY, os.path.join(OUT_DIR, "GHM_Twin_min.xml"))
        scene = load_scene(xml, merge_shapes=False)
        scene.frequency = FREQ_HZ
        arr = PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0.5,
                          horizontal_spacing=0.5, pattern="iso", polarization="V")
        scene.tx_array = arr; scene.rx_array = arr
        for t, p in enumerate(txp):
            scene.add(Transmitter(name=f"tx{t}", position=[float(p[0]), float(p[1]), float(p[2])]))
        for i, p in enumerate(rx3d):
            scene.add(Receiver(name=f"rx{i}", position=[float(p[0]), float(p[1]), float(p[2])]))
        log(f"[scene] objects={len(scene.objects)}, TX={len(txp)}, RX={len(rx3d)}")

        solver = PathSolver()
        paths = solver(scene=scene, max_depth=MAX_DEPTH, los=True,
                       specular_reflection=True, diffuse_reflection=True,
                       refraction=True, synthetic_array=False, seed=RT_SEED)
        a_, tau = paths.cir(normalize_delays=False, out_type="numpy")
        a_ = np.asarray(a_); tau = np.asarray(tau)
        log(f"[rt] a_ {a_.shape}, tau {tau.shape}")

        # tau -> [num_rx, num_tx, num_paths]
        tau_s = np.squeeze(tau)
        pwr = np.abs(np.squeeze(a_)) ** 2
        # 예상 축: (num_rx, num_tx, num_paths). 안전 정리
        nrx, ntx = len(rx3d), len(txp)
        if tau_s.shape[0] != nrx:  # 혹시 축 순서 다르면 보정 시도
            tau_s = np.moveaxis(tau_s, list(range(tau_s.ndim)),
                                sorted(range(tau_s.ndim), key=lambda k: -tau_s.shape[k]))
        log(f"[fmt] tau_s {tau_s.shape} (nrx={nrx}, ntx={ntx})")

        dd, dtoa, los = [], [], []
        for i in range(nrx):
            for t in range(ntx):
                tt = tau_s[i, t]
                pp = pwr[i, t]
                v = np.isfinite(tt) & (tt > 0) & (pp > 0)
                if not v.any():
                    continue
                toa = tt[v].min()
                dist = float(np.linalg.norm(rx3d[i] - txp[t]))
                dd.append(dist); dtoa.append(c * toa)
                los.append(abs(c * toa - dist) < 1.0)
        dd = np.array(dd); dtoa = np.array(dtoa); los = np.array(los)
        log(f"\n[검증] (TX,RX) 경로쌍 {len(dd)}개")
        log("  거리[m] | c·tau_first[m] | 차이[m] | 판정")
        for k in range(len(dd)):
            tag = "LoS✓" if los[k] else "NLoS(반사)"
            log(f"   {dd[k]:8.2f}  {dtoa[k]:10.2f}  {dtoa[k]-dd[k]:8.2f}  {tag}")
        if len(dd) >= 2:
            cc = np.corrcoef(dtoa, dd)[0, 1]
            log(f"\n[요약] corr(c·tau_first, 거리)={cc:.4f}, LoS {los.sum()}/{len(dd)}")
            if los.any():
                e = np.abs(dtoa[los] - dd[los])
                log(f"  LoS 쌍: |c·tau - 거리| 평균={e.mean():.4f}m 최대={e.max():.4f}m "
                    f"-> ToA=거리 확인")
            log(f"  (대조) 기존 데이터는 normalize_delays=True 라 첫 탭=0 → ToA 소실")

        fig, ax = plt.subplots(figsize=(7, 6.5))
        ax.scatter(dd[~los], dtoa[~los], c="tab:red", s=55, label="NLoS(첫 도착=반사)", zorder=3)
        ax.scatter(dd[los], dtoa[los], c="tab:blue", s=55, label="LoS(첫 도착=직선)", zorder=3)
        lim = max(dd.max(), dtoa.max()) * 1.05
        ax.plot([0, lim], [0, lim], "k--", lw=1.3, label="1:1 (ToA=거리)")
        ax.set_xlabel("실제 TX-RX 3D 거리 [m]"); ax.set_ylabel("c · tau_first [m]")
        ax.set_title("GHM Twin: normalize_delays=False → 첫 도착 ToA 검증")
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.grid(True, ls=":", alpha=0.4); ax.legend(loc="upper left")
        fig.tight_layout()
        png = os.path.join(OUT_DIR, f"P2G_toa_verify_GHM_{stamp}.png")
        fig.savefig(png, dpi=150); plt.close(fig)
        log(f"\n[viz] saved -> {png}")
        log("[done] GHM ToA 검증 완료")
    finally:
        log_f.close()


if __name__ == "__main__":
    main()
