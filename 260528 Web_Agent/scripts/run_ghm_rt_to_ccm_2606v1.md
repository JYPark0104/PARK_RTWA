# run_ghm_rt_to_ccm_2606v1.py

GHM Twin 씬으로부터 **진짜 MIMO 공분산(R_BS, R_UE)** 을 산출하는 오프라인 러너.

## 왜 필요한가
- GHM npz(`channel_data_260531_GHM_Twin_v0_1.npz`)의 `R_TX`, `R_RX` 는 `(1,1)` **SISO**
  공분산이라 MIMO 분석에 쓸 수 없다.
- P1F(`Rays_to_Marginal_CCM`)는 ray의 **4각도**(aoa/aod/zoa/zod)에 PanelArray
  배열응답을 적용해 MIMO 공분산을 *합성*한다. 그런데 GHM npz의 ray는 방위각(aoa)만 있다.
- 따라서 P1A를 다시 돌려 4각도 ray를 만든 뒤 P1B→P1F 로 MIMO 공분산을 합성한다.

## 파이프라인
```
scene_builder(.ply/.obj → scene.xml)
  → P1A  (RT, 4각도 ray)        → P1A_RT_Results/Area{X}_{f}GHz_Rays_ALL_RXs.npz
  → P1B  (Valid RX filter)      → P1B_Valid_Results/..._Valid_RXs.npz
  → P1F  (PanelArray CCM 합성)  → P1F_Marginal_CCM_Results/..._Marginal_CCM.npz  (R_BS, R_UE)
```

## 입력
- `--mesh` : GHM 메시(`GHM Twin_v0.1.ply` 또는 `Map_Mesh.obj`). 단일 메시는
  `scene_builder` 가 `itu_concrete` 단일 재질 scene.xml 로 래핑.
- `--npz`  : GHM channel_data npz. `tx_positions(3,3)`, `rx_positions(N,2)` 만 사용.
  (TX 3개 좌표 + RX 좌표; RX z 는 `--rx-z`, 기본 1.5 m)

## 핵심 옵션
| 옵션 | 의미 | 비고 |
|---|---|---|
| `--rx-subset` | RX 샘플 수 | 0/음수면 전체 5071. CPU 검증은 8~64 권장. 등간격 샘플 |
| `--max-depth` | RT 반사/회절 깊이 | CPU면 3, GPU면 5~8 |
| `--bs-panel ROWS COLS` / `--bs-grid NR NC` | BS PanelArray | `n_t = panel×grid` → R_BS 크기 |
| `--ue-panel ROWS COLS` | UE PanelArray | `n_r = rows×cols` → R_UE 크기 |
| `--dry-run` | 씬 빌드+설정 출력만 | RT/CCM 건너뜀, GPU 불필요 |

예) `--bs-panel 4 4 --bs-grid 2 2` → n_t=64 → R_BS 64×64.

## 실행
### H100 (권장, GPU 가속)
```bash
source .venv-h100/bin/activate
WEBAGENT_USE_GPU=1 WEBAGENT_USE_GPU_RT=1 \
python scripts/run_ghm_rt_to_ccm_2606v1.py \
    --mesh "/home/user/scenes/GHM Twin_v0.1.ply" \
    --npz  "/home/user/scenes/channel_data_260531_GHM_Twin_v0_1.npz" \
    --rx-subset 64 --freq 7.5 --max-depth 5 \
    --bs-panel 4 4 --bs-grid 2 2 --ue-panel 2 2
```
H100 이주 절차는 `docs/h100_migration.md` (migrate_to_h100.sh → setup_h100.sh) 참조.
씬 자산(GHM ply/npz)은 `H100_SCENES` 로 함께 rsync 하거나 수동 복사.

### dclcom61 (CPU 모드, 느림)
```bash
source .venv-webagent/bin/activate
python scripts/run_ghm_rt_to_ccm_2606v1.py --rx-subset 8 --max-depth 3   # 검증용
```
- 이 서버(5090 sm_120)는 GPU RT 불가 → mitsuba `llvm`(CPU) variant 로만 동작.
- 전체 5071 RX × 깊은 RT 는 CPU에서 수십 분~시간. 검증은 소수 RX 권장.

## 출력
- `P1F_Marginal_CCM_Results/Area{X}_{f}GHz_RX{r}_Marginal_CCM.npz` 안에
  `R_BS`(n_t×n_t), `R_UE`(n_r×n_r) MIMO 공분산.
- `run_summary_<ts>.json` : 실행 파라미터/출력 경로 요약.

## 설계 메모
- 기존 파일 **무수정**: P1A fork(`forked_25x/P1A_..._web.py`)와 P1F 원본은 건드리지 않음.
  GHM의 흩뿌려진 RX 좌표(데카르트 곱으로 표현 불가)를 넣기 위해, 이 러너 안에서
  `Scene.generate_area_rx_grid` 에 `points` 방식을 **런타임 몽키패치**로만 추가하고
  실행 후 원복한다.
- P1B/P1F 는 `BaseAdapter`(config 잠금 패치 포함)를 그대로 사용.

## 의존성/환경
- Python 3.10, tensorflow 2.21, sionna 1.2.2, mitsuba 3.8.0, drjit 1.3.1, trimesh≥4.0
- venv: `.venv-h100`(GPU) 또는 `.venv-webagent`(CPU)
