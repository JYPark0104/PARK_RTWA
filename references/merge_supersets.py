"""
merge_supersets.py — RTWA superset NPZ 병합 (RX 축 이어붙이기)

용도: O2I(또는 일반) RT 를 '나눠서' 돌린 뒤(예: 높이 1~60m 먼저, 61m~ 나중) 결과를
      하나의 superset 으로 합친다. RX 단위로 인덱싱돼 있으므로 RX 축으로 concat 한다.

사용:
  python merge_supersets.py OUT.npz IN_1.npz IN_2.npz [IN_3.npz ...]

전제(반드시 동일해야 물리적으로 일관):
  - 같은 씬 / 같은 TX 집합(개수·좌표·순서) / 같은 주파수 / 같은 안테나(num_tx_ant,num_rx_ant)
  - 같은 RT 세팅(max_depth, samples, 반사/회절 등) — 이건 파일에 다 안 담기므로 사용자가 보장
  - 합칠 RX 들은 '서로 다른' RX (겹치면 중복됨). 높이대가 겹치지 않게 나눠 돌릴 것.

동작:
  - [T,R,...] 배열: RX 축(axis=1) concat. 경로차원(P=max_paths, K=max_rays)이 다르면 0-패딩으로 맞춤.
  - [R,...] 배열(rx_positions, rx_valid_mask): axis=0 concat.
  - 스칼라/메타(frequency, num_*_ant, tx_positions ...): 동일성 검사 후 유지. max_paths/max_rays 는 재계산.

작성: 2026-07-11
"""
from __future__ import annotations
import sys, time
import numpy as np
from pathlib import Path


def _classify(files0, T, R):
    """첫 파일 기준으로 각 key 의 병합 방식 분류."""
    ax1, ax0, keep = [], [], []
    for k in files0.files:
        a = files0[k]
        if a.ndim >= 2 and a.shape[0] == T and a.shape[1] == R:
            ax1.append(k)
        elif a.ndim >= 1 and a.shape[0] == R:
            ax0.append(k)
        else:
            keep.append(k)
    return ax1, ax0, keep


def _pad_trailing(a, target_trail):
    """axis>=2 의 trailing shape 을 target_trail 로 0-패딩."""
    if a.ndim <= 2:
        return a
    cur = a.shape[2:]
    if tuple(cur) == tuple(target_trail):
        return a
    pad = [(0, 0), (0, 0)] + [(0, t - c) for c, t in zip(cur, target_trail)]
    return np.pad(a, pad, mode="constant")


def merge(out_path, in_paths):
    ds = [np.load(p, allow_pickle=True) for p in in_paths]
    d0 = ds[0]
    T = int(d0["tx_positions"].shape[0])
    R0 = int(d0["rx_positions"].shape[0])
    ax1, ax0, keep = _classify(d0, T, R0)

    # --- 스칼라/메타 동일성 검사 ---
    def _eq(k):
        a = d0[k]
        for d in ds[1:]:
            if k not in d.files:
                raise KeyError(f"'{k}' 가 {in_paths}중 일부에 없음")
            if not np.array_equal(np.asarray(a), np.asarray(d[k])):
                return False
        return True

    for k in ("frequency_ghz", "num_tx_ant", "num_rx_ant", "tx_positions"):
        if k in d0.files and not _eq(k):
            raise ValueError(f"⚠️ '{k}' 가 파일 간 다릅니다 — 같은 조건에서 돌린 결과만 병합 가능")

    out = {}

    # --- RX-축(axis=1, [T,R,...]) ---
    for k in ax1:
        arrs = [d[k] for d in ds]
        # trailing(경로차원) 최대에 맞춰 0-패딩
        max_trail = None
        for a in arrs:
            if a.ndim > 2:
                tr = a.shape[2:]
                max_trail = tr if max_trail is None else tuple(max(x, y) for x, y in zip(max_trail, tr))
        if max_trail is not None:
            arrs = [_pad_trailing(a, max_trail) for a in arrs]
        out[k] = np.concatenate(arrs, axis=1)

    # --- RX-축(axis=0, [R,...]) ---
    for k in ax0:
        out[k] = np.concatenate([d[k] for d in ds], axis=0)

    # --- 메타 유지 + 재계산 ---
    for k in keep:
        out[k] = d0[k]
    R_new = int(out["rx_positions"].shape[0])
    if "max_paths" in out and "path_tau" in out:
        out["max_paths"] = np.array([out["path_tau"].shape[2]], dtype=np.int64)
    if "max_rays" in out and "tau" in out:
        out["max_rays"] = np.array([out["tau"].shape[2]], dtype=np.int64)

    np.savez_compressed(out_path, **out)
    total_R = sum(int(d["rx_positions"].shape[0]) for d in ds)
    print(f"✅ 병합 완료: {out_path}")
    print(f"   입력 {len(ds)}개 · RX {'+'.join(str(int(d['rx_positions'].shape[0])) for d in ds)} = {R_new}")
    print(f"   max_paths={int(out['max_paths'][0]) if 'max_paths' in out else '?'}, "
          f"max_rays={int(out['max_rays'][0]) if 'max_rays' in out else '?'}")
    assert R_new == total_R, "RX 개수 합 불일치"
    return out_path


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("usage: python merge_supersets.py OUT.npz IN_1.npz IN_2.npz [IN_3 ...]")
        raise SystemExit(2)
    merge(sys.argv[1], sys.argv[2:])
