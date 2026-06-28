# Intg (통합 RT) 모듈

## 목적
P1A(ray-level 기하) + batch(MIMO 공분산/집계) 를 **단일 multi-TX superset NPZ** 하나로 생성하고,
**reshaper(Adapter)** 로 각 하류 소비자 포맷으로 변환해 떠먹인다. → 하류(P1B/C/D, batch 뷰) 무수정.

```
통합 RT ─▶ superset NPZ ─┬─ to_p1a ─────────▶ AreaX_{freq}GHz_Rays_ALL_RXs.npz → P1B/C/D
                          └─ to_batch_channel ▶ channel_data_*.npz             → RX Inspector/Scenario/viz
```

## 구성
- `superset_schema.py` : superset NPZ 키/shape/dtype/패딩값 정의 + validate().
  - path-level(`path_*`, batch·공분산용) + ray-level(P1A 호환, TR38.901 확장) 둘 다 포함.
  - 배열형태 (B) 스택 `(T,R,K)`/`(T,R,P)`. 400 하드코딩 폐기 → 동적 K/P.
  - `rx_valid_mask` (0=valid,1=dead,2=rt_fail) 비파괴 필터.
- `reshapers/to_p1a.py` : superset → P1A 포맷 (tx_index 슬라이스, `(R,1,1,1,1,P)`).
- `reshapers/to_batch_channel.py` : superset → channel_data (path-level 가변길이 + R_TX/R_RX).

## 확정 설계
- 공분산 = batch `R_TX`/`R_RX` **단일 출처**.
- samples_per_src 고정(batch_size 비의존), 랜덤배치 기본 ON, seed+batch_idx 파생.
- 멀티TX 단일 NPZ. P1A 단일-TX 소비자는 reshaper 가 tx별로 뽑아줌(P1B 무수정).

## 진행 (TODO)
- [x] superset 스키마 + 두 reshaper (순수함수) — 합성데이터 shape 검증 통과
- [x] intg_writer: batch 코어 + P1A RayGen(subrayProposed) 재사용으로 ray 확장 → superset 생성
- [x] 4.RT 'Intg Mode' 추가 (P1A/batch 유지) — schema engine="intg", batch_runner intg_mode 분기,
      pipeline_executor engine 분기, 프론트 토글(RTConfigPage/useStore), ScenarioPage 게이트
- [x] rx_valid_mask 산출(dead/rt_fail) 통합 — build_superset 내 비파괴 마스크
- [ ] 소규모 씬에서 P1A 원본 NPZ 와 배열 1:1 대조 검증 (실제 GPU RT 미실행)
- [ ] (옵션) P1B 멀티-TX 직접 소비 — 현재는 to_p1a 가 TX별 P1A 파일로 분리 제공(P1B 무수정)
- [ ] ETA 예측 시스템

## Intg 엔진 동작 (구현됨)
4.RT 에서 'Intg 통합 RT' 선택 → `engine="intg"`:
1. batch 코어로 RT (samples_per_src 고정, 랜덤배치 항상 ON, seed=RT seed 파생)
2. `pp.save_output1_multi` → channel_data_*.npz (batch viz/Scenario 호환)
3. `build_superset` → `<session>/Batch_RT_Results/Intg_Results/superset_<title>_<ts>.npz` (canonical)
4. `superset_to_p1a` → 같은 폴더에 `Area{t+1}_{freq}GHz_Rays_ALL_RXs.npz` (TX별, P1B/C/D 무수정 소비)
5. viz/hitmap/export 는 batch 와 동일 경로 재사용
