# ClaudeHoldem

> HUNL(Heads-Up No-Limit Texas Hold'em)에서 **중급자 인간 플레이어를 이기는** Deep CFR 기반 포커 AI.
> **MacBook M1 Pro 단일 머신 / 개인 프로젝트 완주**가 목표이며, Pluribus 급 재현은 범위 밖.

## 프로젝트 상태

**현재 Phase**: Phase 2 (CFR+ / Linear CFR + Leduc Poker) — 착수 대기
**직전 완료**: Phase 1 (Regret Matching + Kuhn CFR) ✅ 2026-04-22

### Phase 진행 트래커

| Phase | 주제 | 목표 | 상태 |
|---|---|---|---|
| 0 | 환경 세팅 | uv + .venv + Claude Code 하네스 + M1 MPS 동작 | ✅ 2026-04-21 |
| **1** | **Regret Matching + Kuhn CFR** | **RPS/Kuhn Nash 수렴, Vanilla CFR, exploitability** | ✅ **2026-04-22** |
| 2 | CFR+ / Linear CFR + Leduc | Leduc Poker에서 2-bet round CFR 변형 비교 | 🔜 다음 |
| 3 | Leduc Deep CFR | 첫 딥러닝 CFR, advantage/strategy net | ⬜ |
| 4 | RLCard NL Hold'em Deep CFR | HUNL에서 실제 학습, 주요 마일스톤 | ⬜ |
| 5 | 친구 대결, 튜닝, 마무리 | 중급자 상대 평가, 회고 | ⬜ |

## Phase 1 성과 요약

### Exit Criteria 5/5 ✅ (margin 포함)

| Criterion | Target | Actual | Margin |
|---|---|---|---|
| RPS L1 to uniform | ≤ 0.05 | 0.023 | 54% |
| Kuhn game value | −1/18 ± 0.001 | −0.055571 | 편차 1.6% |
| Kuhn P1 Jack bet prob | ∈ [0, 1/3] | 0.234993 | 범위 70% |
| Exploitability @ 10k iter | < 5 mbb/g | **2.136 mbb/g** | threshold의 43% |
| 모든 test GREEN | pass | 193/193 | 100% |

### 수학적 검증
- **Zinkevich 2007 Vanilla CFR** "A pattern" → Kuhn Nash 수렴
- **Lanctot 2013 §3.4 3-pass BR** (infoset aggregation) → α-family 4 Nash에서 1e-12 정밀도로 BR(P1)=−1/18, BR(P2)=+1/18
- **O(1/√T) 이론 수렴률 실증**: log-log slope (100→10000) = **−0.52** (이론 −0.5, 오차 4%)

### 시각화
- W&B seed run: https://wandb.ai/zox004/poker-ai-hunl/runs/auf1uzeo
- W&B summary: https://wandb.ai/zox004/poker-ai-hunl/runs/szgdzgqt
- 2-panel convergence plot (log-log + linear w/ Exit Criterion reference lines)

### 속도
- 예상 2주 → **실제 2일** (Week 1 RPS + Week 2 Kuhn 연속 진행)
- **9 commits**, 193 tests GREEN (unit 153 + integration 14 + regression 26)

## 프로젝트 구조

```
src/poker_ai/
├── games/          kuhn.py (Phase 1) — Phase 2 leduc.py / Phase 4 nlh_rlcard.py
├── algorithms/     regret_matching.py, vanilla_cfr.py (Phase 1) — Phase 2 cfr_plus.py, mccfr.py
├── networks/       Phase 3부터
├── eval/           exploitability.py (3-pass BR, Leduc-ready)
└── utils/          Phase 2 reservoir_buffer.py

experiments/        Hydra + W&B harness
├── conf/           phase1_rps.yaml, phase1_kuhn.yaml
└── phase1_*.py     단일 엔트리 포인트 per Phase

tests/              193 tests: unit 153 + integration 14 + regression 26
```

## 운영 규칙

- **문서**: [CLAUDE.md](./CLAUDE.md) 프로젝트 헌법, [PHASE.md](./PHASE.md) 현재 Phase 추적, [ROADMAP.md](./ROADMAP.md) 5 Phase 상세
- **하네스**: `.claude/agents/` (cfr-reviewer, rl-debugger, test-writer), `.claude/skills/poker-ai-dev/`
- **실행**: `uv run pytest` (전체 193), `uv run python -m experiments.phase1_kuhn_vanilla` (10k full run ≈ 4초)

## 하드웨어 / 예산

| 항목 | 비용 |
|---|---|
| MacBook M1 Pro (기존) | $0 |
| Claude Code 구독 (기존) | $0 |
| W&B 개인 플랜 | $0 |
| RunPod 클라우드 (Phase 4 선택) | $30~100 (예정) |

---

**다음 세션**: ROADMAP.md Phase 2 섹션 참조하며 CFR+ vs Linear CFR 비교, Leduc 엔진(직접 구현 vs OpenSpiel), MCCFR 샘플링 방식, exploitability 확장 논의.
