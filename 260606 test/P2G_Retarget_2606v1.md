# P2G_Retarget_2606v1.py — 타깃 재설정: 단일-BS 메트릭은 무엇을 맞추나

## 목적
단일-BS 채널 메트릭(d_RSRP/d_PDP/d_Cov_BS)이 UE-UE 거리(d_phys) 말고
BS-기준 양(거리차/방위차)을 더 잘 설명하는지 재평가.

## 타깃
- T_phys = ||pos_i-pos_j|| (UE-UE)
- T_range = |r_i-r_j|, r=||pos-BS||
- T_angle = BS에서 본 방위각 차 (0~180deg)

## 입력 / 출력
- 입력: `P2G_DistMat_*.npz`, BS_XY=[-51.561,-21.794]
- 출력(`P2G_Retarget_Results/`): `P2G_retarget_*.png`, `*.log`

## 결과 (20260606)
| 타깃 | R²(3특징 OLS) |
|---|---|
| T_phys (UE-UE) | 0.144 |
| T_range (BS거리차) | 0.110 |
| T_angle (BS방위차) | 0.086 |

특징×타깃 corr 구조:
- d_RSRP ↔ T_range(0.31), T_phys(0.30) : 경로손실=거리 신호(단 NLoS로 약함)
- d_Cov_BS ↔ T_angle(0.28) : BS-side 공분산은 '방위각' 정보를 가장 많이 담음
- d_PDP : 전반적으로 약함

## 해석
- 타깃을 BS-거리/각도로 바꿔도 R²가 더 오르지 않음(오히려 낮음).
- 즉 메트릭이 '엉뚱한 타깃에 맞춰진' 게 아니라, 단일-BS 정보 자체가 NLoS
  shadowing으로 거리·각도 모두 약하게만 식별. (d_Cov=각도, d_RSRP=거리 역할 확인)

## 실행
```bash
python3 P2G_Retarget_2606v1.py
```
