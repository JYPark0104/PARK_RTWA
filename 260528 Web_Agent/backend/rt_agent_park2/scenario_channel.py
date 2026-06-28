"""scenario_channel.py — 시나리오 경로 RX들의 채널 관측량(RSRP/PADP/Covariance) 추출.

9.Scenario Results 우측 'Channel State' 패널용. PNG 를 굽지 않고 **클라이언트 렌더용
경량 JSON** 만 반환한다(PADP=3D 콩나물 three.js, Cov=2D heatmap 캔버스).

핵심 최적화:
  - channel_data npz 를 1회만 로드해 모듈 캐시 → 경로를 청크로 나눠 요청해도 빠름.
  - Covariance 가 큰 MIMO(예: 64×64+)면 블록평균으로 표시용 ≤64×64 로 다운샘플.

입력 npz 키 (m5.save_output1_multi):
  rsrp_all (num_tx,num_rx), los_all (num_tx,num_rx)?,
  tau_tx{t}_rx{i}, power_tx{t}_rx{i}(dBm), aoa_tx{t}_rx{i}(deg),
  R_RX_tx{t}_rx{i}, R_TX_tx{t}_rx{i}

실행 환경: Python 3.10 / numpy (컨테이너 venv-webagent)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_NPZ_CACHE: dict = {}
_MAX_COV_DISP = 64   # 공분산 표시 최대 차원 (초과 시 블록평균 다운샘플)


def _load(npz_path: str | Path):
    """npz 를 **지연(lazy) NpzFile** 로 열어 캐시. (배열 11만개+ 환경에서 dict() 전체 materialize 금지)

    Returns (NpzFile, set(files)) — 키 멤버십 O(1), 배열은 접근 시점에만 1개씩 압축해제.
    """
    p = Path(npz_path)
    key = (str(p), p.stat().st_mtime if p.exists() else 0.0)
    if key not in _NPZ_CACHE:
        _NPZ_CACHE.clear()
        z = np.load(p, allow_pickle=False)   # lazy NpzFile (전체 materialize 안 함)
        _NPZ_CACHE[key] = (z, set(z.files))
    return _NPZ_CACHE[key]


def _downsample_abs(R) -> tuple[np.ndarray, int]:
    """|R| 을 표시용 ≤_MAX_COV_DISP 정사각으로 (블록평균) 축소 + [0,1] 정규화. (disp, 원차원n)."""
    M = np.abs(np.asarray(R))
    if M.ndim != 2 or M.size == 0:
        return np.zeros((1, 1)), int(M.shape[0] if M.ndim >= 1 else 0)
    n = M.shape[0]
    if n <= _MAX_COV_DISP:
        disp = M.astype(float)
    else:
        idx = np.linspace(0, n, _MAX_COV_DISP + 1).astype(int)
        disp = np.empty((_MAX_COV_DISP, _MAX_COV_DISP), dtype=float)
        for a in range(_MAX_COV_DISP):
            for b in range(_MAX_COV_DISP):
                blk = M[idx[a]:idx[a + 1], idx[b]:idx[b + 1]]
                disp[a, b] = float(blk.mean()) if blk.size else 0.0
    mx = float(disp.max()) if disp.size else 0.0
    norm = (disp / mx) if mx > 0 else disp
    return norm, n


def _cov_payload(R):
    if R is None or np.asarray(R).size == 0:
        return None
    norm, n = _downsample_abs(R)
    return {
        "m": [round(float(x), 4) for x in norm.reshape(-1)],
        "disp": int(norm.shape[0]),
        "n": int(n),
    }


def channel_state(npz_path: str | Path, rx_indices: list[int], tx_index: int = 0) -> list[dict]:
    """주어진 RX 인덱스들의 채널 관측량 리스트 반환 (입력 순서 유지)."""
    z, keyset = _load(npz_path)
    g = lambda k: (z[k] if k in keyset else None)   # 지연 접근 (필요한 배열만 압축해제)
    t = int(tx_index)
    rsrp_all = g("rsrp_all")
    los_all = g("los_all")
    out: list[dict] = []

    for ri in rx_indices:
        i = int(ri)
        pre = f"tx{t}_rx{i}"
        tau = g(f"tau_{pre}")
        powd = g(f"power_{pre}")
        aoa = g(f"aoa_{pre}")
        R_RX = g(f"R_RX_{pre}")
        R_TX = g(f"R_TX_{pre}")

        rsrp = None
        if rsrp_all is not None and t < rsrp_all.shape[0] and i < rsrp_all.shape[1]:
            v = float(rsrp_all[t, i])
            rsrp = v if np.isfinite(v) else None
        los = None
        if los_all is not None and getattr(los_all, "ndim", 0) == 2 and t < los_all.shape[0] and i < los_all.shape[1]:
            los = bool(los_all[t, i])

        padp = None
        npaths = 0
        if tau is not None and np.asarray(tau).size > 0:
            tau_a = np.asarray(tau, dtype=float).reshape(-1)
            pw_dbm = np.asarray(powd, dtype=float).reshape(-1)
            ao = np.asarray(aoa, dtype=float).reshape(-1) if aoa is not None else np.zeros_like(tau_a)
            p_lin = 10.0 ** ((pw_dbm - 30.0) / 10.0)
            s = p_lin.sum()
            p_norm = (p_lin / s) if s > 0 else p_lin
            npaths = int(tau_a.size)
            padp = {
                "tau": [round(float(x), 2) for x in tau_a],
                "aoa": [round(float(x), 2) for x in ao],
                "power": [round(float(x), 6) for x in p_norm],
            }
            if los is None:
                los = (npaths == 1)

        out.append({
            "rx_idx": i,
            "rsrp": (round(rsrp, 2) if rsrp is not None else None),
            "los": los,
            "num_paths": npaths,
            "padp": padp,
            "r_rx": _cov_payload(R_RX),
            "r_tx": _cov_payload(R_TX),
        })
    return out
