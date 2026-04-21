# 현재 Phase 추적

> 매 세션 시작 시 Claude Code가 이 파일을 먼저 읽고 컨텍스트를 파악한다.
> 각 Phase 완료 시 이 파일을 업데이트한다.

## 현재 상태

**Phase**: 1 (Regret Matching + Kuhn CFR) — Week 1 착수
**시작일**: 2026-04-21
**목표 완료일**: 2026-05-05 (+ 2주)

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
**소요 시간**: 1일 (단일 세션, 약 1시간)
**커밋**: `4247b16`, `a4652e0` — `origin/main`에 push 완료 (https://github.com/zox004/ClaudeHoldem, private)

**달성한 것** (Exit Criteria):
- [x] `uv run python -c "import torch; print(torch.backends.mps.is_available())"` → **True** (검증 재실행 확인)
- [x] `uv run pytest` → **"no tests ran in 0.00s"**
- [x] Claude Code가 `CLAUDE.md` 기반 프로젝트 요약 가능 (한국어 대화 + 최종 목표·Phase·원칙 3가지 산출)
- [x] `.claude/agents/` 에 서브에이전트 3개 배치 — `cfr-reviewer.md`, `rl-debugger.md`, `test-writer.md`
- [x] W&B 로그인 완료 (entity: `zox004`, 프로젝트 `poker-ai-hunl`은 첫 `wandb.init()`에서 자동 생성)

**추가로 완료한 것** (ROADMAP Phase 0 할 일 리스트):
- [x] uv 0.11.7 설치 (Homebrew)
- [x] `uv init --bare --python 3.11 --name poker-ai`, Python 3.11.15 픽스
- [x] 런타임 deps 설치 (numpy, torch 2.11, rlcard[torch], wandb, hydra-core) + dev deps (pytest, ruff, mypy strict)
- [x] src layout (`src/poker_ai`) + hatchling build backend
- [x] 프로젝트 디렉터리 생성 + 각 패키지 `__init__.py`
- [x] `.gitignore` (checkpoints/, wandb/, .venv/, `.claude/settings.local.json` 등)
- [x] `.claude/` 하네스(agents 3종 + poker-ai-dev skill + hooks.json + settings.json) 배치
- [x] `scripts/check_mps.py` 작성·실행 (matmul + autograd backward OK)
- [x] `git init` + 2개 커밋 + private 원격 저장소 push

**배운 것**:
- `uv init --bare`는 build-system 섹션을 생성하지 않는다 → src layout 쓰려면 `[build-system]` + `[tool.hatch.build.targets.wheel]`을 수동으로 추가해야 `from poker_ai...` import가 된다
- `uv add`만으로는 Python 버전이 고정되지 않는다 (`requires-python` 하한의 **최신** Python을 선택, 이번에 3.14.4가 잡힘) → `uv python pin 3.11` + `uv sync`로 명시 고정 필요
- `.claude/settings.local.json`은 per-machine permission 설정이므로 절대 커밋 금지 (`.gitignore`에 추가)
- `uv run wandb login`은 Claude Code의 `!` 셸에서 TTY가 없어 대화형 프롬프트가 안 뜬다 → 키를 인자로 직접 전달 (`uv run wandb login <KEY>`) 또는 별도 터미널에서 수행
- `gh` CLI가 2022년 버전(2.20.2)이라 `--json visibility` 같은 최근 필드가 없다. 기능 자체는 호환되지만 `brew upgrade gh` 권장

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
