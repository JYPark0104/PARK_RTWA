# P2G_gpu_utils_2606v1.py

## 목적
P2G 메트릭 파이프라인 **공용 멀티-GPU 유틸리티**. 작업 지침 "계산 시 서버의 모든
GPU 사용"을 모든 단계에서 일관되게 적용하기 위한 모듈.

## 제공 함수
| 함수 | 설명 |
|---|---|
| `get_devices(prefer_gpu=True)` | 가용 CUDA GPU 전부를 `torch.device` 리스트로 반환. 없으면 `[cpu]` 폴백 |
| `warmup_linalg(devices)` | `torch.linalg` lazy 초기화를 메인 스레드에서 선실행 (멀티스레드 충돌 방지) |
| `batched_topr_eigh_factor(load_fn, N, M, r, eps, devices, ...)` | N개 Hermitian 행렬을 모든 GPU에 샤딩, 배치 eigh→상위 r 인수 `B_i=U_iΛ_i^{1/2}` 추출. 반환 `(B[N,M,r], tr[N], tail[N])` |
| `pairwise_nuclear(B, devices, ...)` | 저랭크 인수로 BW fidelity `F(i,j)=‖B_iᴴB_j‖_*` 를 행-샤딩 일괄 계산. 반환 `F[N,N]` |

## 설계 포인트
- **I/O와 연산 분리**: npz 로드는 CPU(`load_fn`), eigh/SVD는 GPU.
- **디바이스당 1스레드** ThreadPoolExecutor 로 2-GPU 동시 가동 (CUDA가 GIL 해제).
- **lazy-init 충돌 해결**: `torch.linalg.eigh/svdvals` 첫 호출이 스레드에서 동시에
  일어나면 `RuntimeError: lazy wrapper should be called at most once` 발생 →
  스레드 생성 전 `warmup_linalg`로 디바이스별 1회 선호출하여 회피.
- **메모리**: 원본 행렬은 청크 단위로만 GPU에 올리고 인수만 CPU로 회수.

## d_BW² 결합 (호출측)
`batched_topr_eigh_factor` 의 `B`,`tr` + `pairwise_nuclear` 의 `F` 로
`d_BW²(i,j) = tr_i + tr_j − 2·F(i,j)`, `d_Cov = sqrt(max(d_BW²,0))`.
(F 형태의 정확성은 P2G_BW_Approx_Check_2606v1 에서 검증 완료)

## 주의
- 샌드박스가 GPU 접근을 막으므로 **full-permission(샌드박스 해제)로 실행**해야 함.

## 실행 환경
- Python 3.10.12 / torch 2.12.0+cu130 / numpy 2.2.6
- 서버 dclserver78, NVIDIA H100 NVL 95GB × 2

## 검증(간접)
- `P2G_PointTable_Builder_2606v1.py` 에서 `batched_topr_eigh_factor` 사용 시
  CPU scipy 결과와 동일한 tail 에너지(mean 2.21e-3) 재현, eigh 275s→28.8s.
