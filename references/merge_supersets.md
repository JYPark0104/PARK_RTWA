# merge_supersets.py — superset 병합 (나눠 돌린 RT 이어붙이기)

O2I(또는 일반) RT 를 **높이대(또는 RX 집합)를 나눠** 돌린 뒤 결과를 하나로 합치는 도구.
예: 1~60m 를 이미 돌렸고, 61m~ 만 추가로 돌려서 합치고 싶을 때 (전체 재실행 불필요).

## 왜 가능한가
superset NPZ 는 **RX 단위**로 인덱싱된다. 배열 축이:
- `[T, R, ...]` (path_*, tau, power, rsrp_all, R_TX/R_RX, ...) → RX 는 **axis=1**
- `[R, ...]` (rx_positions, rx_valid_mask) → RX 는 **axis=0**
그래서 서로 다른 RX 집합(겹치지 않는 높이대)을 **RX 축으로 concat** 하면 하나로 합쳐진다.
경로 차원(max_paths P)이 두 파일에서 다르면 **0-패딩**으로 큰 쪽에 맞춘다.

## 반드시 동일해야 하는 조건 (안 지키면 물리적으로 비일관)
- 같은 씬 / 같은 TX 집합(개수·좌표·순서) / 같은 주파수 / 같은 안테나(num_tx_ant, num_rx_ant)
- 같은 RT 세팅(max_depth, num_samples, 반사/회절/투과 등) — 파일에 다 안 담기므로 **사용자가 보장**
- 합칠 RX(높이대)가 **겹치지 않게** 나눠 돌릴 것 (겹치면 중복 RX)
스크립트가 frequency/num_ant/tx_positions 동일성은 자동 검사하고, 다르면 에러로 막는다.

## 워크플로 (RTWA)
1. RTWA 에서 **새 facade(O2I) 잡**을 z_min=61(기존 높이 그리드 이어서), z_max=원하는값,
   **z_distance·씬·TX·안테나·주파수·RT세팅을 기존과 동일하게** 두고 실행.
   → 61m~ RX 만 담긴 새 superset 이 나온다.
2. 두 superset 을 병합:
   ```bash
   python merge_supersets.py  merged_1to<max>m.npz  superset_1to60m.npz  superset_61plus.npz
   ```
   (입력은 2개 이상 가능: `... OUT.npz IN1 IN2 IN3 ...`)
3. 병합 결과의 RX 순서 = [1~60m RX..., 61m~ RX...].

## O2I 메타데이터 주의 (중요)
O2I penetration 후처리는 RX 별 **host 건물/재질/법선(facade_rx_meta)** 이 필요하다.
이것도 **같은 RX 순서로** 이어붙여야 한다. superset 과 별도 파일(JSON/NPZ)로 관리된다면,
동일하게 [1~60m, 61m~] 순서로 concat 해야 인덱스가 맞는다. (필요하면 이 병합도 도와줄 수 있음)

## 검증 상태
- 합성 데이터(서로 다른 max_paths 포함) 왕복 병합으로 로직 검증 완료:
  RX concat, max_paths 패딩(값 보존/0-패딩), rsrp/R_RX/valid_mask concat 모두 정상.
- 메모리: 이 씬의 dense superset 은 비압축 시 수십 GB. 서버 RAM(≈377GB) 로 병합 가능하나,
  매우 큰 파일이면 여유 RAM 확인 권장.

## 한계/대안
- superset 은 '경로 축(P)' 패딩이 커서 용량이 크다. RX 를 많이 나눠 여러 번 합쳐도 되지만
  최종 P = 입력들 중 최대 경로수로 커진다(패딩 증가). 문제되면 압축 저장(기본)으로 완화됨.
