# 현재 Phase 추적

> 매 세션 시작 시 Claude Code가 이 파일을 먼저 읽고 컨텍스트를 파악한다.
> 각 Phase 완료 시 이 파일을 업데이트한다.

## 현재 상태

**Phase**: 1 (Regret Matching + Kuhn CFR) — 시작 대기
**시작일**: 2026-04-21 (Phase 1 착수 시점 기준)
**목표 완료일**: 2026-05-05 (시작 + 2주)

## 다음 할 일 (Next Action) — Phase 1 Week 1

- [ ] `test-writer` 에이전트로 `tests/unit/test_regret_matching.py` FAILING 테스트 먼저 작성
  - 상대가 "항상 바위"일 때 내 전략이 "항상 보"로 수렴
  - seed 고정 (42, 123, 456) + tolerance
- [ ] `src/poker_ai/algorithms/regret_matching.py` 구현 (~50줄, red → green)
- [ ] 두 플레이어 동시 regret matching → 균등 분포 (1/3, 1/3, 1/3) 수렴 시각화
- [ ] W&B에 convergence curve 로깅 (project="poker-ai-hunl")

## 지금까지 한 일 (Done)

### Phase 0 (환경 세팅) — **완료 2026-04-21**
- uv 0.11.7 설치 (Homebrew)
- `uv init --bare --python 3.11 --name poker-ai` + `.python-version` = 3.11
- Python 3.11.15 자동 다운로드, `.venv` 구축
- 런타임 deps: numpy 2.4.4, torch 2.11.0, rlcard[torch] 1.2.0, wandb 0.26.0, hydra-core 1.3.2
- Dev deps: pytest 9.0.3, ruff 0.15.11, mypy 1.20.1 (strict mode)
- src layout 구성 (hatchling, `pyproject.toml`의 `[tool.hatch.build.targets.wheel]`)
- 프로젝트 디렉터리: `src/poker_ai/{games,algorithms,networks,eval,utils}`, `tests/{unit,integration,regression}`, `experiments/conf`, `notebooks`, `scripts`
- `.gitignore` (checkpoints/, wandb/, .venv/, .claude/settings.local.json 등)
- `.claude/` 하네스 배치: 서브에이전트 3종(cfr-reviewer, rl-debugger, test-writer) + skill(poker-ai-dev) + hooks.json + settings.json
- M1 Pro MPS 동작 검증: `scripts/check_mps.py` (matmul + autograd backward OK, ~100ms for 1024²)
- W&B 로그인 완료 (entity: zox004)
- `git init` + 첫 커밋: "Phase 0: project scaffolding and Claude Code harness"

## 현재 고민 / 블로커

_없음._

## 이번 Phase(1)의 Exit Criteria

- [ ] RPS regret matching이 균등 분포로 수렴 (W&B 스크린샷)
- [ ] Kuhn CFR의 게임 가치가 **-1/18 ± 0.001** 로 수렴
- [ ] Kuhn CFR의 Player 1 Jack bet 확률이 **[0, 1/3]** 범위
- [ ] Exploitability가 10,000 iter 후 **< 0.01 mbb/g**
- [ ] 모든 unit test 통과

## 참고 문서

- [ROADMAP.md](./ROADMAP.md) — 전체 5 Phase 상세
- [CLAUDE.md](./CLAUDE.md) — 프로젝트 헌법
- `.claude/agents/` — 서브에이전트 3종
- `.claude/skills/poker-ai-dev/SKILL.md` — 포커 AI 스킬

---

## Phase 로그

### Phase 0 (환경 세팅)
**완료일**: 2026-04-21
**소요 시간**: 1일 (세션 1회)
**달성한 것**:
- uv 기반 Python 3.11 프로젝트 골격 완성
- Claude Code 하네스(agents/skills/hooks) 배치
- MPS + torch 2.11 동작 검증
- W&B 연동 준비
- git repo 초기 커밋
**배운 것**:
- `uv init --bare`는 build-system을 생성하지 않으므로 src layout 쓰려면 수동으로 `[tool.hatch.build.targets.wheel]` 추가 필요
- `uv add` 기본 동작은 `requires-python` 하한의 **최신** Python을 선택 → 3.11 픽스는 `uv python pin 3.11` 필요
- `.claude/settings.local.json`은 per-machine이므로 .gitignore 필수
**다음 Phase로 이월된 이슈**:
- 없음

<!-- 템플릿
### Phase N (제목)
**완료일**: YYYY-MM-DD
**소요 시간**: N일 / N시간
**달성한 것**: ...
**배운 것**: ...
**다음 Phase로 이월된 이슈**: ...
-->
