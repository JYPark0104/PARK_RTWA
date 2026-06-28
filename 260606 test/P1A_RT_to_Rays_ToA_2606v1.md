# P1A_RT_to_Rays_ToA_2606v1.py — 절대지연(ToA) 보존 재생성용 복사본

## 무엇인가
원본 `251009_CCM_Collection (8 GB)/P1A_RT_to_Rays_2509v6.py` 의 **복사본**.
딱 한 줄만 변경: `paths.cir(normalize_delays=True → False)`.

> AGENTS 규칙 준수: 참고용 원본은 수정하지 않고 복사본을 만들어 작업.
> 사용자 승인(옵션1) 후 진행.

## 왜 바꿨나 (배경)
- 기존 데이터의 `tau` 는 `normalize_delays=True` 때문에 **첫 도착 경로=0**(excess delay).
  → LoS 절대지연(=거리/c)이 제거되어 ToA 거리정보가 없음.
  (P2G 검증: c·tau_first vs 기하거리 corr=0.07, tau_first 중앙값=0ns)
- `normalize_delays=False` 로 두면 `tau` = **절대 전파지연**.
  → tau[LoS] = 거리/c 가 복원되어 단일/multi-BS ToA 측위 가능.

## 변경의 안전성 (중요)
`normalize_delays` 는 한 RX의 모든 경로를 **동일 상수(첫 도착 지연)** 만큼 평행이동할 뿐:
- 지연 확산(max-min), PDP **모양**, 각도, 공분산(R_BS/R_UE) = **불변**
  (공분산은 공통 위상이 a·aᴴ 에서 상쇄 → True/False 무관)
- 바뀌는 것 = **절대 오프셋(LoS 거리/c)** 뿐. 즉 거리 정보만 추가됨.
- 부수효과: d_PDP 가 절대 오프셋을 포함하게 되어 거리 변별력이 올라갈 수 있음(향후 확인).

## 변경 위치
`extract_cir()` (786행 부근):
```python
# 변경 전
a_, tau = paths.cir(normalize_delays=True, out_type="numpy")
# 변경 후
a_, tau = paths.cir(normalize_delays=False, out_type="numpy")
```
`extract_cir` 는 RT 경로에서 단 한 번 정의되어 모든 호출(예: 1860, 1970행)에 적용됨.

## 실행 주의
- Sionna RT 환경 + scene(.xml/.ply) + GPU 필요. RT 재실행은 무겁다.
- **소규모(일부 RX/소영역)로 먼저 검증** 권장:
  생성 직후 LoS RX 에 대해 `c·tau_first ≈ ||pos-BS||` 인지 확인.
- LoS RX 만 ToA=거리 대응이 정확하므로 `los_nlos_flag` 로 LoS 선별 검증.

## 검증 체크리스트 (재생성 후)
1. tau_first 중앙값 ≠ 0 (절대지연 복원 확인)
2. LoS RX: corr(c·tau_first, 기하거리) ≈ 1.0, 오차 작음
3. NLoS RX: c·tau_first ≥ 기하거리 (반사로 더 긴 경로)
4. R_BS/R_UE, PDP 모양은 기존과 동일(상대구조 불변) 재확인

## 실행 (환경 준비 시)
```bash
python3 P1A_RT_to_Rays_ToA_2606v1.py   # Sionna RT + scene + GPU 필요
```
