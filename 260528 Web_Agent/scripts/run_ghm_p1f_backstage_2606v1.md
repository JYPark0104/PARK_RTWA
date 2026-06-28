# run_ghm_p1f_backstage_2606v1.sh

P1F Marginal CCM을 **백그라운드(nohup)** + **2×GPU fork**로 재개합니다.

## 명령

```bash
cd "260528 Web_Agent"
chmod +x scripts/run_ghm_p1f_backstage_2606v1.sh

./scripts/run_ghm_p1f_backstage_2606v1.sh start    # 백그라운드 시작
./scripts/run_ghm_p1f_backstage_2606v1.sh status  # 진행률 + GPU
./scripts/run_ghm_p1f_backstage_2606v1.sh logs    # tail 로그
./scripts/run_ghm_p1f_backstage_2606v1.sh stop    # 중단
```

## 파일

| 경로 | 설명 |
|------|------|
| `logs/ghm_p1f_backstage.pid` | 메인 프로세스 PID |
| `logs/ghm_p1f_backstage.log` | 표준 출력 로그 |

## 환경 변수

- `SESSION_DIR` — 기본 `sessions/ghm_full5071_260604_161504`
- `WEBAGENT_P1F_NUM_GPUS` — 기본 `2`

## 동작

- `P1F_Rays_to_Marginal_CCM_2510v2_dualgpu` fork 사용
- `--no-with-derived` — RSRP/PADP 생략, P1F만
- 기존 npz 있는 RX는 skip
