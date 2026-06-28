# scenario_channel.py — 시나리오 경로 채널 관측량 추출

## 목적
9.Scenario Results 우측 'Channel State' 패널용. 경로 RX들의 RSRP/PADP/Covariance 를
**PNG 없이 클라이언트 렌더용 경량 JSON** 으로 반환한다.
(PADP=three.js 3D 콩나물, Cov=2D heatmap 캔버스 로 프런트가 직접 렌더)

## 최적화
- channel_data npz 를 **1회만 로드해 모듈 캐시**(_NPZ_CACHE, 한 세션만 유지) →
  경로를 청크로 나눠 요청해도 첫 청크 외엔 빠름. 재생 중 계산 0(사전/병렬 로딩).
- Covariance 가 큰 MIMO(예: 64×64+)면 블록평균으로 표시용 ≤64×64(_MAX_COV_DISP) 다운샘플 + [0,1] 정규화.

## API
- `channel_state(npz_path, rx_indices, tx_index=0) -> list[dict]`
  - 각 item: `{rx_idx, rsrp(dBm|None), los(bool|None), num_paths,
    padp:{tau[],aoa[],power[](합=1 정규화)}|None, r_rx:{m[],disp,n}|None, r_tx:{...}|None}`

## 연동
- 엔드포인트 `GET /api/sessions/{uuid}/scenario/channel_state?start=&count=`
  (scenario_state.json 의 path/tx_index 사용, 청크 반환 {start,count,total,tx_index,items})
- 프런트 `components/ChannelStatePanel.tsx` (PadpMini 3D, CovHeatmap 2D),
  `pages/ScenarioResultsPage.tsx` 가 20개씩 청크 병렬 로딩 + 진행바 + step 동기 표시.

## 실행 환경
Python 3.10 / numpy / 컨테이너 venv-webagent (WebAgent_park_server78).
