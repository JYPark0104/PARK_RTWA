# P1F_Rays_to_Marginal_CCM_2510v2_dualgpu.py

Web Agent 전용 P1F fork. **H100 2장을 동시에** 쓰도록 RX 작업을 분할합니다.

## 방식

- `multiprocessing` **spawn** (TF/CUDA fork-safe)
- Worker 0 → `CUDA_VISIBLE_DEVICES=0`, Worker 1 → `=1`
- 각 worker가 원본 `P1F_Rays_to_Marginal_CCM_2510v1.py`를 **독립 import** 후 할당 RX만 처리
- 이미 있는 `*_Marginal_CCM.npz`는 skip (재개 안전)

## 환경 변수

| 변수 | 기본 | 설명 |
|------|------|------|
| `WEBAGENT_P1F_NUM_GPUS` | `2` | 사용할 GPU 수 상한 |

## 적용

`base_adapter.py`의 P1F `forked_module`로 자동 선택.  
`run_p1f()` / `run_ghm_rt_to_ccm_2606v1.py --with-derived` 가 이 모듈을 호출합니다.

## 기존 단일 프로세스 작업 중단 후 재실행

이미 돌고 있는 단일-GPU P1F(PID)는 **중단 후** 재실행해야 2-GPU fork가 적용됩니다.

```bash
# 예: 남은 RX만 dual-GPU로
cd "260528 Web_Agent" && source .venv-webagent/bin/activate
export WEBAGENT_USE_GPU=1 WEBAGENT_P1F_NUM_GPUS=2
python scripts/run_ghm_rt_to_ccm_2606v1.py \
  --from-stage post-p1b \
  --session-dir sessions/ghm_full5071_260604_161504 \
  --skip-derived   # RSRP/PADP 이미 있으면
```

`--skip-derived` 플래그가 없으면 derived만 건너뛰려면 스크립트 옵션 확인 필요.
