# runtime_env.py

Web Agent GPU/CPU 런타임 정책.

## 기본 (2026-06-04)

- **GPU ON** (`WEBAGENT_USE_GPU=1` 기본값)
- **Mitsuba CUDA RT ON** (`WEBAGENT_USE_GPU_RT=1` 기본값) — P1A PathSolver 가속
- `backend` 패키지 import 시 `apply_webagent_gpu_env()` 1회 실행

## 예외

| 환경변수 | 동작 |
|----------|------|
| `WEBAGENT_FORCE_CPU=1` | 강제 CPU |
| `WEBAGENT_USE_GPU=0` | TF GPU 끔 |
| `WEBAGENT_USE_GPU_RT=0` | Mitsuba LLVM, TF만 GPU |
| `WEBAGENT_ALLOW_BLACKWELL_GPU=1` | sm_120(5090)에서도 GPU 시도 (불안정) |

Blackwell(sm≥12)은 stable wheel 미지원으로 **자동 CPU 폴백**.

## 기동

```bash
bash scripts/run_gpu_backend.sh
# 또는
WEBAGENT_USE_GPU=1 WEBAGENT_USE_GPU_RT=1 uvicorn backend.app:app --host 0.0.0.0 --port 8800
```
