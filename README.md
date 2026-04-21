# Poker AI Roadmap 하네스 — 시작 가이드

이 디렉터리는 **HUNL Deep CFR 포커 AI 개발 프로젝트**를 위한 완성된 Claude Code 하네스입니다. 파일들을 당신의 새 프로젝트 디렉터리에 복사해서 바로 시작하면 됩니다.

## 포함된 파일

```
.
├── README.md              ← 이 파일
├── ROADMAP.md             ← 5 Phase 상세 로드맵 (Week 0 ~ Week 16)
├── CLAUDE.md              ← 프로젝트 헌법. 루트에 둘 것
├── PHASE.md               ← 현재 Phase 추적. 매 Phase 완료 시 업데이트
└── .claude/
    ├── settings.json      ← Claude Code 기본 설정
    ├── hooks.json         ← 파일 수정 시 자동 검증 훅
    ├── agents/
    │   ├── cfr-reviewer.md    ← CFR 수식 정확성 검증 전문가
    │   ├── rl-debugger.md     ← RL 학습 디버깅 전문가
    │   └── test-writer.md     ← TDD 강제. 구현 전 실패 테스트 작성
    └── skills/
        └── poker-ai-dev/
            └── SKILL.md    ← HUNL 구현 도메인 컨벤션
```

## 시작하는 법

### 1. 프로젝트 디렉터리 세팅
```bash
# 새 프로젝트 만들기
mkdir ~/poker-ai && cd ~/poker-ai

# 이 하네스 파일들을 루트에 복사
cp -r /path/to/poker-ai-roadmap/* .
cp -r /path/to/poker-ai-roadmap/.claude .

# uv 프로젝트 초기화
uv init .
uv add numpy torch rlcard[torch] wandb pytest ruff mypy hydra-core

# Git 시작
git init
echo -e "checkpoints/\nwandb/\n.venv/\n__pycache__/\n*.pyc\n.DS_Store" > .gitignore
git add . && git commit -m "chore: initial harness + roadmap"
```

### 2. Claude Code 열기
```bash
cd ~/poker-ai
claude   # Claude Code 실행
```

Claude Code가 열리면:
- `CLAUDE.md`를 자동으로 읽음
- `/agents` 로 3개 서브에이전트 확인
- `/skill` 로 `poker-ai-dev` skill 확인

### 3. 첫 질문 던져보기
```
지금 Phase 0 세팅 중이야. 아직 필요한 파일이나 할 일 있어?
```

Claude가 `PHASE.md`와 `CLAUDE.md`를 참조해서 체크리스트를 확인시켜 줄 것입니다.

### 4. Phase 1 시작
Phase 0 exit criteria 만족하면:
```
Phase 1 시작하자. 로드맵대로 Rock-Paper-Scissors regret matching부터. 
test-writer 에이전트로 테스트 먼저 짜줘.
```

## 서브에이전트 호출 패턴

| 상황 | 호출 문장 |
|---|---|
| 새 알고리즘 시작 | "test-writer로 [모듈명] 테스트 먼저 짜줘" |
| CFR 구현 리뷰 | "cfr-reviewer로 [파일명] 검토해줘" |
| 학습 안 됨 | "rl-debugger로 W&B run [URL] 분석해줘" |
| 포커 도메인 코드 | "poker-ai-dev skill 참조해서 [작업]" |

## 중요 원칙

- **CLAUDE.md 수정은 신중하게**: 200줄 이내로 유지
- **Phase exit criteria 건너뛰지 말 것**: 특히 Phase 1의 Kuhn Nash 검증
- **매 Phase 완료 시 PHASE.md 업데이트**: 다음 세션 컨텍스트 유지용
- **W&B 로깅은 처음부터**: 나중에 추가하려면 귀찮아짐

## 문제 생겼을 때

1. **Claude가 TDD를 안 따름** → "잠깐, test-writer 에이전트 먼저 호출해서 실패 테스트부터"
2. **Claude가 hallucination** (존재하지 않는 CFR 변형 주장) → "cfr-reviewer 에이전트로 정확성 체크"
3. **학습이 안 됨** → "rl-debugger로 분석, 그 다음 `PHASE.md`에 블로커 기록"
4. **컨텍스트가 흐트러짐 (긴 세션)** → `/clear` 후 `CLAUDE.md + PHASE.md` 로 재시작

## 예상 진행 속도 (주 10~15시간 투자)

| 주차 | Phase | 마일스톤 |
|---|---|---|
| Week 0 | 0 | 환경 세팅 |
| Week 1-2 | 1 | Kuhn Poker CFR 완주, Nash 수렴 |
| Week 3-4 | 2 | Leduc CFR+/MCCFR 비교 |
| Week 5-7 | 3 | Leduc Deep CFR 수렴 |
| Week 8-12 | 4 | RLCard NL Hold'em Deep CFR |
| Week 13-16 | 5 | 친구 대결, 튜닝, 마무리 |

**현실적 이탈 시나리오**:
- Phase 3에서 Deep CFR 수렴 안 돼서 +2주
- Phase 4에서 HUNL 학습 시간 길어서 +2주
- 친구 대결 결과 미달로 Phase 4 튜닝 재진입

최악의 경우에도 **5개월 이내 프로젝트 완주** 가능성이 높습니다.

## 예산 재확인

| 항목 | 비용 |
|---|---|
| 하드웨어 (M1 Pro 기존) | $0 |
| Claude Code 구독 | 기존 |
| W&B | $0 (개인) |
| RunPod 클라우드 (선택, Phase 4) | $30~100 |

---

**다음 할 일**: 이 디렉터리를 당신 프로젝트 루트에 복사하고, `PHASE.md`에 오늘 날짜를 기록한 뒤 Phase 0 체크리스트를 시작하세요.
