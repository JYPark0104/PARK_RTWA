# P2G_ToA_Verify_GHM_2606v1.py — ToA 검증 (성공)

## 결론 (먼저)
**`normalize_delays=False` 로 두면 첫 도착 지연(ToA)이 실제 거리와 정확히 일치한다.**
- LoS 경로쌍 9/23: `c·tau_first = 거리`, |오차| **최대 0.0001m**(서브밀리미터)
- NLoS 경로쌍: `c·tau_first ≥ 거리` (첫 도착이 반사라 더 긴 경로) — 물리적으로 정상
- corr(c·tau_first, 거리)=0.90 (NLoS 초과분 포함)

## 왜 GHM Twin 으로 했나
- Jonggak scene: mesh 104/519 누락 → RT 불가.
- 기존 데이터(P1A Area1, GHM channel_data): 둘 다 `normalize_delays=True` 라
  첫 탭=0(excess) → 기존 데이터로는 ToA 검증 불가.
- GHM Twin 은 단일 통합 ply(환경 전체) 보유 + 좌표계가 tx/rx 와 정합
  → 최소 Mitsuba xml 로 감싸 직접 RT (재질은 직선경로 지연에 무관).

## 방법
1. `GHM Twin_v0.1.ply` 를 감싸는 최소 xml 생성(diffuse itu_concrete).
2. tx_positions 3개 + rx_positions 12개(z=1.5) 배치, 7.5GHz, max_depth=5.
3. `paths.cir(normalize_delays=False)` → 각 (TX,RX) 첫 도착 tau.
4. `c·tau_first` vs `‖TX-RX‖_3D` 비교.

## 결과 (20260606_105306)
| 구분 | 수 | c·tau_first vs 거리 |
|---|---|---|
| LoS | 9 | 일치(오차≤0.0001m) |
| NLoS | 14 | 초과(반사 경로) |

실행: CPU(sionna 1.2.1), ~5초.

## 의의
- 데이터 재생성 시 `normalize_delays=False` 만 바꾸면 **LoS RX 에서 거리=c·ToA** 확보.
- multi-BS ToA → 삼각측량 → 위치복원(앞서 R²=1.0 입증) 경로가 실데이터로 가능.
- 주의: NLoS RX 는 첫 도착이 반사라 거리>기하거리. LoS 선별(los flag/일치성) 필요.

## 입력/출력
- 입력: `260604 GHM_test/GHM Twin_v0.1.ply`, `channel_data_260531_GHM_Twin_v0_1.npz`
- 출력(`P2G_ToA_Verify_Results/`): `GHM_Twin_min.xml`, `P2G_toa_verify_GHM_*.png`, `*.log`

## 실행
```bash
CUDA_VISIBLE_DEVICES="" /home/dclserver78/sionna_0310/venv/bin/python P2G_ToA_Verify_GHM_2606v1.py
```
(GPU 초기화 hang 회피 위해 CPU. 소수 RX라 CPU로 충분.)
