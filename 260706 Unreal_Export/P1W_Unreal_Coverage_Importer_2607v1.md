# P1W_Unreal_Coverage_Importer_2607v1.py

## 목적
**Unreal Engine 에디터 내부에서 실행**하는 파이썬 스크립트.
P1V가 뽑은 커버리지 CSV(`unreal_tx_*.csv`, `unreal_rx_*.csv`)를 읽어
Unreal 레벨에 다음을 생성한다.

- **TX 타워 마커** — 밝은 빨강 발광 구체 (높이 포함)
- **RX 커버리지 포인트 클라우드** — RSRP 세기별 색칠 (파랑=약함 → 빨강=강함)

`blender.py`(bpy)가 씬에 마커/커브를 스크립트로 스폰하던 것의 Unreal 대응판.

> v1 범위: **정적 커버리지 클라우드까지.** 시간축 키프레임 재생(모빌리티)은
> 다음 단계 **P1X**에서 Sequencer로 붙인다. 먼저 CSV→Unreal 파이프라인이
> 화면에 뜨는지 검증하는 것이 목표.

## 실행 방법 (Unreal 에디터)
1. `260706 Unreal_Export/P1V_Unreal_Export_Results/`의 CSV들을 **로컬 PC로 전송**
   (예: `scp -r dclserver78:"~/.../P1V_Unreal_Export_Results" C:\Users\01jun\Documents\RT_Export`)
2. 이 스크립트 파일도 로컬 PC로 복사
3. 스크립트 상단 **CONFIG의 `CSV_DIR`** 을 CSV가 있는 본인 PC 폴더 경로로 수정
4. Unreal 상단 메뉴 **Tools(툴) → Execute Python Script...** → 이 파일 선택
5. World Outliner에 `RT_Coverage` 폴더가 생기고, 뷰포트에 TX/RX가 나타남

## CONFIG 옵션
| 항목 | 기본값 | 설명 |
|------|--------|------|
| `CSV_DIR` | (수정 필수) | CSV들이 있는 PC 폴더 |
| `SCALE` | 100.0 | 미터→cm 변환 |
| `FLIP_X` / `FLIP_Y` | False / True | 축 뒤집힘 보정 |
| `RX_SCALE` / `TX_SCALE` | 3.0 / 12.0 | 구체 크기 [m] |
| `MAX_RX` | 2000 | RX 서브샘플 상한(성능 보호). 전체(5071) 쓰려면 늘리기 |
| `EMISSIVE` | 3.0 | 발광 강도 |
| `SKIP_DEAD` | True | dead zone(RSRP=-inf) 제외 |
| `NUM_BINS` | 12 | RSRP 색 구간 수 |

## 동작 원리
1. `M_RTCoverage` 부모 머티리얼 생성 (EmissiveColor 벡터 파라미터 → 발광)
2. 색 구간(NUM_BINS)별 Material Instance + TX용 인스턴스 생성 (`/Game/RT_Coverage/`)
3. 엔진 기본 구체(`/Engine/BasicShapes/Sphere`) 를 각 위치에 스폰,
   RSRP 정규화값(`rsrp_norm`)으로 색 구간 선택
4. 재실행 시 이전 `RT_Coverage` 폴더 액터를 먼저 제거(중복 방지)
5. 포인트가 수천 개라 그림자 캐스팅은 끔(`set_cast_shadow(False)`)

## 좌표계 주의
- CSV는 미터(RT 좌표) → `SCALE`(×100)로 cm 변환.
- 도시 메시와 겹쳐 배치하려면 **메시와 포인트의 원점/축/스케일이 동일**해야 한다.
- 좌우/앞뒤가 뒤집혀 보이면 `FLIP_X`, `FLIP_Y` 토글로 맞춘다.

## 성능 팁
- 처음엔 `MAX_RX=2000` 권장. 부드러우면 5071로 올리기.
- 개별 액터 스폰 방식이라 수천 개면 수십 초 걸릴 수 있음(정상).
- 더 대규모/고성능이 필요하면 향후 Instanced Static Mesh(HISM) 방식으로 전환.

## 요구 사항
- Python Editor Script Plugin 활성화
- Unreal Engine 5.x

## 한계 / 다음 단계
- v1은 정적 배치까지. **키프레임 재생 → P1X (Sequencer 스크립팅)** 에서 진행.
- 튕기는 Ray 경로 시각화는 이 CSV로 불가(신호세기 데이터만) → 별도 ray export 필요.
