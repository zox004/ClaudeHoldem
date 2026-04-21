# 현재 Phase 추적

> 매 세션 시작 시 Claude Code가 이 파일을 먼저 읽고 컨텍스트를 파악한다.
> 각 Phase 완료 시 이 파일을 업데이트한다.

## 현재 상태

**Phase**: 0 (환경 세팅)
**시작일**: YYYY-MM-DD
**목표 완료일**: YYYY-MM-DD (시작 + 4일)

## 다음 할 일 (Next Action)

- [ ] `uv init poker-ai` 실행
- [ ] 의존성 설치: `uv add numpy torch rlcard[torch] wandb pytest ruff mypy hydra-core`
- [ ] M1 Pro MPS 사용 가능 확인
- [ ] W&B 프로젝트 `poker-ai-hunl` 생성
- [ ] `.claude/` 하네스 디렉터리 복사
- [ ] `CLAUDE.md` 프로젝트 루트에 배치
- [ ] 첫 커밋 (git init + .gitignore)

## 지금까지 한 일 (Done)

_아직 없음. 로드맵만 확정._

## 현재 고민 / 블로커

_없음._

## 이번 Phase의 Exit Criteria

- [ ] `uv run python -c "import torch; print(torch.backends.mps.is_available())"` → True
- [ ] `uv run pytest` 가 "no tests ran" 출력
- [ ] Claude Code에게 "프로젝트 목표 알려줘" 물었을 때 CLAUDE.md 기반 응답
- [ ] `/agents` 로 3개 서브에이전트 확인 가능
- [ ] W&B 로그인 완료

## 참고 문서

- [ROADMAP.md](./ROADMAP.md) — 전체 5 Phase 상세
- [CLAUDE.md](./CLAUDE.md) — 프로젝트 헌법
- `.claude/agents/` — 서브에이전트 3종
- `.claude/skills/poker-ai-dev/SKILL.md` — 포커 AI 스킬

---

## Phase 로그 (각 Phase 완료 시 기록)

### Phase 0 (환경 세팅)
_진행 중_

<!-- 템플릿
### Phase N (제목)
**완료일**: YYYY-MM-DD
**소요 시간**: N일 / N시간
**달성한 것**: 
- ...
**배운 것**: 
- ...
**다음 Phase로 이월된 이슈**: 
- ...
-->
