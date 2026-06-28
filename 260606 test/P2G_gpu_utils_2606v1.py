# -*- coding: utf-8 -*-
"""
P2G_gpu_utils_2606v1.py
=======================
P2G 메트릭 파이프라인 공용 멀티-GPU 유틸리티.

작업 지침: "계산 시 서버의 모든 GPU 를 사용한다."
  - get_devices(): 가용 CUDA GPU 전부를 반환(없으면 CPU 폴백).
  - batched_topr_eigh_factor(): 다수의 Hermitian 행렬을 모든 GPU 에 샤딩하여
      배치 eigh -> 상위 r 고유쌍 인수 B_i=U_iΛ_i^{1/2} 추출 (스트리밍 I/O).
  - pairwise_nuclear(): 저랭크 인수들로 BW fidelity ||B_i^H B_j||_* 를
      모든 GPU 에 행-샤딩하여 일괄 계산.

설계 메모
  - GPU 접근이 막힌 샌드박스에서는 호출이 실패할 수 있으므로, 실행은
    full-permission(샌드박스 해제)로 해야 한다.
  - 디스크 I/O(npz 로드)는 CPU, 수치연산(eigh/SVD)은 GPU 로 분리.
  - 디바이스별 작업은 ThreadPoolExecutor(디바이스 1개당 워커 1개)로 병렬화.
    CUDA 호출이 GIL 을 해제하므로 2-GPU 동시 가동.

----------------------------------------------------------------------
실행 환경
  - Python 3.10.12 / torch 2.12.0+cu130 / numpy 2.2.6
  - 서버: dclserver78, NVIDIA H100 NVL 95GB x 2
----------------------------------------------------------------------
"""

import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch


def get_devices(prefer_gpu=True):
    """가용 GPU 전부 반환. 없으면 [cpu]."""
    if prefer_gpu and torch.cuda.is_available() and torch.cuda.device_count() > 0:
        return [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
    return [torch.device("cpu")]


def _hermitize_t(A):
    return 0.5 * (A + A.mH)


def warmup_linalg(devices, dtype=torch.complex64):
    """torch.linalg 의 lazy 초기화를 '메인 스레드'에서 미리 실행.
    (멀티스레드에서 첫 호출 시 'lazy wrapper should be called at most once'
     충돌이 나므로, 스레드 생성 전 디바이스별로 한 번씩 워밍업.)"""
    for d in devices:
        a = torch.eye(4, dtype=dtype, device=d)
        a = 0.5 * (a + a.mH)
        torch.linalg.eigh(a)
        torch.linalg.svdvals(a)
        if d.type == "cuda":
            torch.cuda.synchronize(d)


def batched_topr_eigh_factor(load_fn, n_items, M, r, eps, devices,
                             chunk=48, dtype=torch.complex64, log=print):
    """다수의 Hermitian PSD 행렬 -> 상위 r 저랭크 인수 (모든 GPU 샤딩).

    Parameters
    ----------
    load_fn : callable(i:int) -> np.ndarray (M,M) complex
        i 번째 원본 행렬을 디스크/메모리에서 로드(스레드에서 호출됨).
    n_items : int            전체 행렬 수 N
    M, r    : int            차원, 저랭크 랭크
    eps     : float          Ã = A/tr(A) + εI 안정화
    devices : list[torch.device]
    chunk   : int            GPU 배치 크기(디바이스당)

    Returns
    -------
    B    : (N, M, r) complex64 (CPU)   인수 B_i = U_iΛ_i^{1/2} (내림차순)
    tr   : (N,) float64                tr(Ã_i) (= 1 + εM)
    tail : (N,) float64                상위 r 에서 제외된 에너지 비율
    """
    B = np.zeros((n_items, M, r), dtype=np.complex64)
    tr = np.zeros(n_items, dtype=np.float64)
    tail = np.zeros(n_items, dtype=np.float64)
    warmup_linalg(devices, dtype=dtype)
    eyeM = {d: (eps * torch.eye(M, dtype=dtype, device=d)) for d in devices}
    splits = np.array_split(np.arange(n_items), len(devices))
    t0 = time.time()
    done = [0]

    def worker(dev, idxs):
        for c0 in range(0, len(idxs), chunk):
            sub = idxs[c0:c0 + chunk]
            mats = np.stack([load_fn(int(i)) for i in sub]).astype(np.complex64)
            A = torch.from_numpy(mats).to(dev)             # (c,M,M)
            A = _hermitize_t(A)
            trA = torch.diagonal(A, dim1=-2, dim2=-1).sum(-1).real  # (c,)
            At = A / trA.to(A.dtype)[:, None, None] + eyeM[dev]
            trAt = torch.diagonal(At, dim1=-2, dim2=-1).sum(-1).real
            w, U = torch.linalg.eigh(At)                   # 오름차순
            w_r = w[:, -r:].clamp(min=0.0)                 # 상위 r
            U_r = U[:, :, -r:]
            w_r = torch.flip(w_r, dims=[1])                # 내림차순
            U_r = torch.flip(U_r, dims=[2])
            Bsub = (U_r * torch.sqrt(w_r)[:, None, :]).to(torch.complex64)
            tail_sub = 1.0 - w_r.sum(1) / trAt
            B[sub] = Bsub.cpu().numpy()
            tr[sub] = trAt.double().cpu().numpy()
            tail[sub] = tail_sub.double().cpu().numpy()
            done[0] += len(sub)
            log(f"[eigh] {done[0]}/{n_items} on {len(devices)}GPU, "
                f"{time.time()-t0:.1f}s")
            del A, At, w, U, U_r, Bsub
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)

    with ThreadPoolExecutor(max_workers=len(devices)) as ex:
        list(ex.map(lambda da: worker(*da), zip(devices, splits)))
    log(f"[eigh] done. factor mem ~{B.nbytes/1e6:.1f} MB, "
        f"tail mean={tail.mean():.3e} max={tail.max():.3e}, "
        f"{time.time()-t0:.1f}s")
    return B, tr, tail


def pairwise_nuclear(B, devices, sub=128, rblk=64, log=print):
    """저랭크 인수 B (N,M,r) 로 BW fidelity F(i,j)=||B_i^H B_j||_* 일괄 계산.
    모든 GPU 에 행-샤딩. 반환 F (N,N) float64 (CPU).
    d_BW^2(i,j) = tr_i + tr_j - 2 F(i,j) 는 호출측에서 결합.

    [성능] ||G||_* = Σσ_k(G) = Σ sqrt(λ_k(G^H G)). G^H G(r x r) 의 eigvalsh 는
    배치 svdvals 보다 ~300배 빠름(작은 행렬 다량). eigvalsh 사용.
    """
    N = B.shape[0]
    warmup_linalg(devices)
    Bt = torch.from_numpy(B)
    row_splits = np.array_split(np.arange(N), len(devices))
    Bd = {d: Bt.to(d) for d in devices}
    F = torch.zeros((N, N), dtype=torch.float64)
    t0 = time.time()

    def worker(dev, rows):
        if len(rows) == 0:
            return
        r = B.shape[2]
        jit = (1e-12 * torch.eye(r, dtype=torch.complex128, device=dev))
        r0, r1 = int(rows[0]), int(rows[-1]) + 1
        Bi_all = Bd[dev][r0:r1]
        nrow = r1 - r0
        out = torch.empty((nrow, N), device=dev, dtype=torch.float32)
        # 배치(=rblk*sub)를 제한해야 eigvalsh 워크스페이스 OOM 회피.
        nblk = (nrow + rblk - 1) // rblk
        for bi, rb in enumerate(range(0, nrow, rblk)):
            rb1 = min(nrow, rb + rblk)
            Bi = Bi_all[rb:rb1]
            for j0 in range(0, N, sub):
                j1 = min(N, j0 + sub)
                G = torch.einsum("imr,jms->ijrs", Bi.conj(), Bd[dev][j0:j1])
                # ||G||_* = Σ sqrt(λ(G^H G)). float64 + 미세 jitter 로 축퇴 eigh 수렴.
                GhG = (G.mH @ G).to(torch.complex128) + jit
                ev = torch.linalg.eigvalsh(GhG).clamp(min=0.0)
                out[rb:rb1, j0:j1] = torch.sqrt(ev).sum(-1).to(torch.float32)
                del G, GhG, ev
            if (bi + 1) % 2 == 0 or bi + 1 == nblk:
                log(f"[pairwise] {dev} rowblk {bi+1}/{nblk}, {time.time()-t0:.1f}s")
        F[r0:r1] = out.double().cpu()
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)

    with ThreadPoolExecutor(max_workers=len(devices)) as ex:
        list(ex.map(lambda da: worker(*da), zip(devices, row_splits)))
    log(f"[pairwise] F ({N}x{N}) on {len(devices)}GPU, {time.time()-t0:.1f}s")
    return torch.nan_to_num(F).numpy()
