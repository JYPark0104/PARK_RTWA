# p1f_dual_gpu_worker.py

P1F Marginal CCM을 **2개 GPU subprocess**로 병렬 실행합니다 (`multiprocessing.spawn` pickle 이슈 회피).

## 오케스트레이터

`run_dual_gpu_orchestrator(session_dir, config_overrides)` — fork / BaseAdapter에서 호출.

## Worker

```bash
python -m backend.jobs.p1f_dual_gpu_worker --gpu 0 --session-dir ... --chunk-file ...
```

세션 `.p1f_dualgpu/worker_gpu{0,1}.log`에 worker 로그 저장.
