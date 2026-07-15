# Unreal Engine 사용법 — RT 결과 시각화 (처음 쓰는 사람용)

> 목적: RTWA(Ray Tracing Web Agent)의 RT 결과(TX/RX/광선)를 **Unreal Engine**에서
> 도시 트윈 위에 띄우고, Blender처럼 **키프레임으로 재생**해 "있어보이는 영상"을 만든다.
> 이 문서는 **Unreal을 한 번도 안 써본 사람도** 순서대로 따라올 수 있게 작성했다.

작성: 2026-07-10 / 대상 버전: Unreal Engine 5.4~5.5

---

## 0. 큰 그림

```
[RTWA 웹]  → blender_mobility.py / unreal_scenario.py 다운로드
                       │
[로컬 PC]  Unreal 설치 → 프로젝트 생성 → 도시 트윈(glb) 임포트
                       │
           unreal_scenario.py 실행(딸깍) → TX/RX/광선 생성 + 재생
                       │
           카메라 + 영상 렌더
```

- **Unreal 에디터는 로컬 PC(GUI 있는 Windows/Mac)** 에서 돌린다. (서버 X)
- 도시 메시와 RT 데이터는 **같은 좌표계**여야 겹쳐서 보인다.

---

## 1. 설치 & 프로젝트 만들기

1. **Epic Games Launcher** 설치 → 로그인
2. 좌측 **Unreal Engine → 라이브러리 → `+`** 로 엔진 설치 (5.4/5.5 권장, 30~60GB)
3. 실행 → **프로젝트 브라우저**
   - **게임 → 삼인칭(Third Person)** 템플릿 (조작 익히기 좋음) 또는 빈 프로젝트
   - **블루프린트**, **Desktop**, 품질 **Maximum**(노트북이면 나중에 Medium), **시작용 콘텐츠 ✓**
   - **프로젝트 이름**은 영문/숫자 (예: `RT_Twin`) → **생성**
   - 첫 로딩 때 셰이더 컴파일로 시간 걸림(정상)

### 1-1. Python 플러그인 켜기 (필수)
- 상단 **Edit(편집) → Plugins(플러그인)** → 검색 `Python`
- **Python Editor Script Plugin** ✓, **Sequencer Scripting** ✓ → **재시작**
- 재시작 후 **Tools(툴) → Execute Python Script...** 메뉴가 보이면 OK

---

## 2. GPU 확인 (버벅일 때)

- 외장 GPU 사용 확인: `Ctrl+Shift+Esc`(작업 관리자) → **성능** 탭 → GPU 목록
  - Unreal 조작 시 **NVIDIA GPU 사용률이 오르면** 정상
- 내장으로 돌면: **설정 → 시스템 → 디스플레이 → 그래픽** → `UnrealEditor.exe` → **고성능(High performance)**
- 버벅이면: 상단 **Settings → Engine Scalability Settings → Medium**, 그리고 우하단 **"Compiling Shaders"** 카운터가 0이 될 때까지 대기
- 참고: 렌더링은 **GPU 전용**. "CPU로 렌더" 옵션은 없다.

---

## 3. 도시 트윈(glb) 임포트

> `.ply`는 Unreal이 직접 못 읽는다. **`.glb`(권장) / `.obj` / `.usd`** 로 준비.
> `.glb`는 **텍스처까지 한 파일**에 포함 → 가장 편함. (Blender: File → Export → glTF Binary(.glb))

1. 하단 **Content Drawer**(`Ctrl+Space`) 열기
2. `.glb` 파일을 **탐색기에서 드래그 앤 드롭** (또는 Import)
3. 임포트 옵션 창 → 기본값으로 **Import** (메시+머티리얼+텍스처 자동 생성)
   - 재질(Material Instance)이 수십~수백 개 생겨도 **정상** (glb에 재질이 많아서)
4. 생성된 **Static Mesh 애셋들을 뷰포트로 드래그**
   - 메시가 여러 개면 **전부 한 번에 선택해서 드래그**하면 상대 위치가 유지됨
5. 배치 후 정합을 위해, 도시 메시 전부 선택 →
   **Details → Transform → Location (0,0,0) / Rotation (0,0,0) / Scale (1,1,1)**

### 3-1. 드래그하면 좌표가 무시되고 커서에 놓일 때
- Static Mesh 애셋은 "위치 정보"를 안 들고 있어서 그렇다.
- **여러 메시를 한꺼번에 선택해 드래그**하거나, 놓은 뒤 **Location을 (0,0,0)** 으로 맞추면 원래 상대 위치로 조립된다.
- 그래도 안 되면 **File → Import Into Level** 또는 Datasmith glTF로 임포트(계층/위치 보존).

---

## 4. 기본 지면(Landscape) 제거

템플릿 기본 지면이 z=0에 있어 **z=0 아래 메시를 가린다.** 우리 도시 트윈을 쓰니 제거한다.

1. 아웃라이너 **검색창에 `Landscape`** 입력
   → `Landscape` + `LandscapeStreamingProxy_...` 들이 모두 나옴 (**실제 지형은 프록시가 그림**)
2. **전부 선택** (첫 항목 클릭 → 맨 아래 Shift+클릭) → **Delete**
   - 삭제 안 되면: 다중 선택 상태로 Details에서 **"액터 게임에서 숨김" 체크** + 눈 아이콘(👁)으로 에디터 숨김
3. ⚠️ **도시 메시는 건드리지 말 것.** Lighting(하늘/조명)도 그대로 둔다.

> 주의: "액터 게임에서 숨김"은 **Play 모드에서만** 숨김. 에디터 뷰포트 숨김은 아웃라이너 **눈 아이콘**.

---

## 5. 이동/시점 조작

### 에디터 뷰포트 (Play 아님, 검토에 최적)
- **우클릭 + 드래그** = 시점 회전
- 우클릭 누른 채 **W/A/S/D** = 날아다니기 (충돌 없음, 어디든 통과)
- **우클릭 + 마우스 휠** = 비행 속도 조절 (또는 우상단 카메라 아이콘의 Camera Speed)
- **마우스 휠** = 전후, **F** = 선택 오브젝트로 카메라 이동

### Play(재생) 모드
- 기본 폰이 날아다니지만 **충돌 구**가 있어 건물에 막힐 수 있음("투명 벽")
- 자유 통과하려면: 시각화용이니 **도시 메시 충돌 끄기**
  - 도시 메시 선택 → Details → **Collision → Collision Presets → `NoCollision`**
- Play 콘솔(백틱 `` ` ``)에서 `ghost`(통과 비행)/`fly`/`walk` — 단, 기본 DefaultPawn엔 효과 없을 수 있음 → 위의 충돌 끄기가 확실

### FPS/성능 보기
- 뷰포트 좌상단 화살표 메뉴 → **실시간(Ctrl+R) 켜기** 후 **FPS 표시**
- 또는 콘솔(`` ` ``)에 `stat fps`, `stat unit`(GPU 시간), `stat gpu`
- "실시간 오버라이드"로 회색이면 → 메뉴에서 **실시간 오버라이드 끄기**

---

## 6. RT 데이터 불러와 재생 (핵심) ★

RTWA "9. Scenario Results"에서 받은 **`unreal_scenario.py`** 한 파일이면 된다.
(데이터가 파일 안에 들어 있어 **별도 JSON 불필요** — blender_mobility.py와 동일한 "딸깍 실행" 방식)

1. **`unreal_scenario.py`** 를 로컬 PC에 저장
2. Unreal 상단 **Tools(툴) → Execute Python Script...** → 이 파일 선택
3. 실행되면:
   - 콘텐츠 브라우저 **`/Game/RT_Rays/`**(= 콘텐츠 → RT_Rays) 에 머티리얼/시퀀스 생성
   - 아웃라이너 **`RT_Rays`** 폴더에 TX(빨강)/RX(초록)/광선(노랑) 액터 생성
   - 재생 모드면 **`SEQ_RayPlayback`** Level Sequence 생성
4. 재생: **콘텐츠 → RT_Rays → `SEQ_RayPlayback` 더블클릭** → Sequencer 창 → **▶ 재생**
   - RX가 스텝마다 순차로 켜지며 광선이 재생됨 (Blender와 동일)

### 6-1. 좌표 정합 (도시와 광선이 안 맞을 때)
- `unreal_scenario.py` 상단 **CONFIG**에서 조정 (파일을 편집기로 열어 값만 수정 → 저장 → 재실행):
  - `SCALE = 100.0` : 미터→cm (도시가 glTF ×100로 들어왔으면 100 유지)
  - `FLIP_Y = True` : **glb를 glTF로 임포트한 경우 보통 이 값이 맞음(기본값)**
  - 그래도 좌우/앞뒤가 뒤집히면 `FLIP_X` 도 토글, 둘 다 켜면 180° 회전
- 정합 확인 팁: `MAX_STEPS = 5` 로 소수만 띄워 빠르게 맞춘 뒤 `0`(전체)로.
- `RT_TX` 선택 → **F** → TX 빨강 구체가 어느 건물 옥상에 얹히는지 확인

### 6-2. 재생이 이상할 때
- 다 보이거나 다 숨으면(visibility 반대) → CONFIG `VIS_KEY_VISIBLE = False` 로 바꿔 재실행
- 실행 직후 화면이 비어도 정상(프레임 0에서 대부분 숨김) → Sequencer 타임라인을 좌우로 긁어(scrub) 확인
- 멈춘 것 같으면 → **Window → Output Log** 에서 `[unreal_scenario]`/`[P1Y]` 진행 로그 확인

---

## 7. (다음 단계) 카메라 & 영상

1. **Cine Camera Actor** 배치 → Sequencer에 **Camera Cuts** 트랙 추가 → 키프레임으로 카메라 이동
   - (A) 도시 위 고정 궤도 샷 / (B) 재생되는 RX를 따라가는 추적 샷
2. **비주얼 폴리시**: Post Process Volume(Bloom/노출/톤), 시간대·안개, (선택) 광선을 Niagara로
3. **영상 출력**: **Window → Cinematics → Movie Render Queue** → `SEQ_RayPlayback` 등록 → mp4/PNG 시퀀스 렌더

---

## 8. 자주 겪는 문제 (트러블슈팅)

| 증상 | 원인 | 해결 |
|------|------|------|
| glb 드래그하면 재질이 수백 개 뜸 | glb에 재질 많음 | 정상. 임포트 끝날 때까지 대기 |
| 메시가 커서 위치에 놓임 | Static Mesh는 위치정보 없음 | 여러 개 한번에 드래그 / Location 0 |
| z=0 아래가 안 보임 | 기본 Landscape가 가림 | Landscape+프록시 전부 삭제/숨김 |
| Play에서 투명 벽 | 도시 충돌 + 폰 충돌 | 도시 Collision → NoCollision |
| 광선이 도시와 안 겹침 | 좌표계 다름(glTF 변환) | `FLIP_Y=True`(기본), 안되면 FLIP_X |
| 스크립트 실행이 0%에서 멈춘 듯 | 대량 스폰/모달 | Output Log로 진행 확인, 개수 줄이기 |
| 재생해도 안 보임/다 보임 | visibility 극성 | `VIS_KEY_VISIBLE` 토글 |
| 버벅임 | Maximum 품질/셰이더 컴파일 | Scalability Medium, 컴파일 대기, 전원 연결 |

---

## 9. 요약 체크리스트

- [ ] Unreal 설치 + 프로젝트 + Python/Sequencer 플러그인
- [ ] 도시 glb 임포트 → 뷰포트 배치 → Location 0
- [ ] 기본 Landscape(+프록시) 제거
- [ ] `unreal_scenario.py` 실행 → TX/RX/광선 생성
- [ ] 좌표 정합(FLIP_Y 등) → TX가 건물 위에
- [ ] `SEQ_RayPlayback` 재생 확인
- [ ] (다음) 카메라 + Movie Render Queue로 영상

---

## 10. 다음 할 일 — "그럴싸한 영상" 만들기 (2026-07-10 시점)

여기까지 완료:
- ✅ Unreal 설치/프로젝트/플러그인
- ✅ 도시 트윈(glb) 임포트 + 정합(FLIP_Y)
- ✅ RTWA → `unreal_scenario.py` 다운로드 → 실행 → **광선 키프레임 재생(SEQ_RayPlayback)** end-to-end
- ✅ 시퀀스 루프 재생
- ✅ 성능: 작업 시 Scalability 낮춤 / Lumen(GI) 낮춤 (RTX5050)

### 남은 단계
```
① 카메라 워크 (영상 핵심)
   - Cine Camera Actor 배치 → Sequencer에 Camera Cuts 트랙 추가 → 키프레임으로 이동
   - (A) 고정 궤도 샷: 도시 위를 천천히 도는 시네마틱 (빠르고 무난)
   - (B) RX 추적 샷: 재생되며 켜지는 RX를 카메라가 따라감 (역동적, 손 더 감)

② 비주얼 폴리시 (확 예뻐짐)
   - Post Process Volume: Bloom(광선 번짐), 노출, 톤 매핑
   - 시간대/하늘/안개 (야경이 특히 멋짐), 필요 시 Lumen 켜기(최종만)
   - (선택) 광선을 Niagara "흐르는 빛"으로 업그레이드

③ 영상 출력
   - Window → Cinematics → Movie Render Queue
   - SEQ_RayPlayback 등록 → mp4 / PNG 시퀀스로 렌더
   - ★ 오프라인 렌더라 에디터 fps와 무관하게 부드럽고 고화질로 나옴
     → 작업은 가볍게(저품질), 최종 렌더만 고품질(에픽/시네마틱)로
```

### 재개 시 체크
- [ ] 레벨 저장돼 있나 (File → Save Current Level, 이름 예: RT_Twin_Map)
- [ ] RTWA 백엔드/vite 살아있나 (컨테이너 재시작했으면 §"컨테이너 재시작 시" 명령으로 재기동)
- [ ] 광선 안 떠 있으면 `unreal_scenario.py` 다시 실행

### 성능 메모 (RTX 5050)
- 작업: Settings → Engine Scalability → Medium, Global Illumination/Reflections → Low, 콘솔 `r.ScreenPercentage 50`, 날 땐 Unlit 뷰
- 광선 많아 무거우면 `unreal_scenario.py`의 MAX_STEPS 축소
- 최종 영상 품질은 Movie Render Queue가 결정 (에디터 버벅임과 별개)

### 컨테이너 재시작 시 RTWA 복구 (호스트에서)
```bash
docker exec -d WebAgent_park_server78 bash -lc 'cd "/workspace/260528 Web_Agent" && WEBAGENT_USE_GPU=1 WEBAGENT_USE_GPU_RT=1 TF_CPP_MIN_LOG_LEVEL=1 nohup .venv-webagent/bin/python -m uvicorn backend.app:app --host 0.0.0.0 --port 8800 --workers 1 > "logs/uvicorn_restart_$(date +%y%m%d_%H%M%S).log" 2>&1 &'
docker exec -d WebAgent_park_server78 bash -lc 'cd "/workspace/260528 Web_Agent/frontend" && VITE_API_TARGET=http://localhost:8800 nohup npm run dev > "../logs/vite_restart_$(date +%y%m%d_%H%M%S).log" 2>&1 &'
```
