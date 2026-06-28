# RTX 5090 GPU 활성화 진단 보고서

**작성일**: 2026-05-28
**환경**: dclcom61 / Ubuntu / Python 3.10 / venv `.venv-webagent`

## 1. 하드웨어 / 드라이버

| 항목 | 값 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5090 × 2 (각 32 GB) |
| Driver | 580.95.05 |
| Driver-reported CUDA | 13.0 |
| Compute Capability | **12.0 (sm_120, Blackwell)** ← 매우 신규 아키텍처 |
| `cuInit()` 직접 호출 | OK (return 0, device count = 2) |
| `nvidia-smi` | 정상 |

## 2. 소프트웨어 스택 (현재 stable PyPI)

| 패키지 | 버전 | CUDA 빌드 | sm_120 native binary |
| --- | --- | --- | --- |
| `tensorflow` | 2.21.0 | CUDA 12.5.1 + cuDNN 9 | ❌ (sm_90까지) |
| `drjit` | 1.3.1 | (numba+cuda) | ❌ |
| `mitsuba` | 3.8.0 | (drjit 의존) | ❌ |
| `sionna` | 1.2.2 | TF/drjit 의존 | ❌ |
| `nvidia-cudnn-cu12` | 9.22.0.52 | CUDA 12.x | sm_90까지 |
| `nvidia-cublas-cu12` | 12.9.2.10 | CUDA 12.9 | sm_90까지 |
| `nvidia-nvjitlink-cu12` | 12.9.86 | CUDA 12.9 | (PTX JIT 가능) |

## 3. 실험 결과 요약

| 시나리오 | 결과 |
| --- | --- |
| `import tensorflow; tf.linalg.matmul(4096, 4096)` GPU0 | ✓ 5.5 ms/iter (sm_120 PTX JIT 캐시 후 정상) |
| `nvidia-smi` 출력 | ✓ 2장 모두 정상 인식 |
| `drjit.cuda.Float(1.0)` | ✗ `cuInit() failed`, 메시지 `(null)` (실제 원인은 sm_120 module load 실패) |
| `mitsuba.set_variant("cuda_ad_mono_polarized")` 후 시뮬레이션 | ✗ (drjit-cuda 의존) |
| `mitsuba LLVM + sionna PathSolver` (TF GPU OFF) | ✓ 0.34 s |
| `paths.cir()` (TF GPU OFF) | ✓ 0.02 s |
| `paths.cir()` (TF GPU ON) | ✗ **SEGV** (process 즉시 종료) |
| `import tf, drjit, mitsuba, sionna` 같은 process 공존 | ✗ 종료 시 `free(): invalid pointer` |

## 4. 본질적 원인

1. **RTX 5090은 sm_120 (Blackwell) 컴퓨트 캐퍼빌리티** — 2026년 초 출시된 매우 새로운 아키텍처
2. 현재 PyPI에 있는 stable wheels는 모두 **sm_50~sm_90까지의 native binary만 포함**
3. sm_120용으로는 **PTX 코드를 JIT 컴파일**해야 하는데:
   - 단순 ops (`matmul`, `random.normal`)는 컴파일 성공 ✓
   - 복잡한 ops (특히 sionna PathSolver 결과를 TF tensor로 변환 시 사용되는 fused/custom op)는 **JIT 단계 또는 첫 실행에서 SEGV** ✗
4. drjit-cuda 1.3.1은 sm_120 PTX를 *전혀* 임베드하지 않아 모듈 로드 실패 → `cuInit failed` 메시지 (잘못된 메시지, 실제는 module load 실패)

## 5. 향후 해소 시점 (upstream 의존)

| 패키지 | sm_120 native 지원 예정 |
| --- | --- |
| TensorFlow | **TF 2.22+ (CUDA 12.8 빌드)** — 2026 Q2~Q3 stable 예상 |
| drjit | 1.5+ (TBD) — github master에서 CUDA 12.8 빌드 옵션 추가 중 |
| mitsuba | drjit 업그레이드와 함께 |
| sionna | drjit/mitsuba 호환 매트릭스 유지 시 자연스럽게 |

## 6. 즉시 사용 가능한 옵션

### Option A — 안정 우선 CPU 모드 (현재 default)
- 작은 안테나(4×4): 16 RX × 1 freq × 1 area E2E = **38 s**
- 큰 안테나(BS 1024 AE / UE 16 AE): 수 분~수십 분 추정
- 동작 보장 ✓, 결과 정확성 보장 ✓

### Option B — Stage별 subprocess 분리 (구현 +1~2h)
- P1A (`paths.cir`): subprocess CPU
- P1F/P1G/P1H/P1I/P1P (heavy SVD/matmul): main GPU
- 큰 안테나에서 GPU 효과 기대
- 위험: TF process 분리로 인한 overhead, 첫 stage 종료 시 SEGV 가능

### Option C — Nightly 업그레이드 실험 (구현 +30 min, 위험 큼)
- `pip install --pre tf-nightly[and-cuda]` (CUDA 12.8 빌드, sm_120 native 가능성)
- `pip install git+https://github.com/mitsuba-renderer/drjit`
- 위험: sionna 1.2.2가 tf-nightly와 호환 안 될 가능성
- venv 백업 필수, 실패 시 즉시 롤백

## 7. 권장 사항

1. **단기 (지금)**: Option A 유지. `prefer_gpu=False`를 default로 + 환경변수 `WEBAGENT_USE_GPU=1`로 실험적 활성화 가능
2. **중기 (1~2주)**: Option B 구현 — 큰 안테나 워크로드 대비
3. **장기 (TF 2.22 출시 후)**: 환경 업그레이드, 전 stage GPU
