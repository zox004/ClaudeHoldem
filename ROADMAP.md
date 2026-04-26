# HUNL Poker AI 개발 로드맵

> **목표**: 중급자 수준의 일반 플레이어를 상대로 이기는 Heads-Up No-Limit Texas Hold'em AI 개발
> **환경**: MacBook M1 Pro, Python 3.11 + PyTorch (MPS), Claude Code
> **최종 알고리즘**: Deep CFR (+ subgame solving 선택적)
> **예상 기간**: 3~4개월 (주 10~15시간 투자 기준)

---

## 프로젝트 성공 기준 (Definition of Done)

이 프로젝트는 다음 중 **하나 이상**이 달성되면 성공으로 본다:

1. **정량 기준**: RLCard NL Hold'em 환경에서 학습한 봇이 random agent 대비 +200 mbb/hand 이상, rule-based agent 대비 +50 mbb/hand 이상 (10,000 hands, 95% CI)
2. **정성 기준**: 포커 룰에 익숙한 친구(pot odds·3-bet·range 중 2개 이상 이해) 상대 1,000 hands에서 net win
3. **학습 기준**: Leduc Hold'em에서 Deep CFR exploitability가 10 mbb/g 이하로 수렴

**비-목표** (이번 프로젝트에서 추구하지 않는 것):
- Slumbot 격파
- 6-max/9-max 확장
- 프로·상급자 격파
- Pluribus 수준 재현

---

## 전체 로드맵 개요

```
Phase 0 (Week 0)      환경 세팅 + 하네스 구축
Phase 1 (Week 1~2)    Regret Matching + Kuhn CFR
Phase 2 (Week 3~4)    Leduc CFR+/MCCFR
Phase 3 (Week 5~7)    Leduc Deep CFR
Phase 4 (Week 8~12)   RLCard NL Hold'em Deep CFR
Phase 5 (Week 13~16)  평가·튜닝·친구 대결
```

각 Phase는 **exit criteria (다음 단계로 갈 자격)** 를 가진다. 이걸 충족 못하면 다음으로 넘어가지 않는다. 디버깅 단계에서 원인 분리 가능성을 확보하기 위해서다.

---

## Phase 0: 환경 세팅 + Claude Code 하네스 구축 (Week 0, 2~4일)

### 목표
개발 환경이 완비되고 Claude Code가 프로젝트 컨텍스트를 이해한 상태로 작동한다.

### 할 일
1. **Python 환경**: `uv` 설치 → `uv init poker-ai` → Python 3.11 픽스
2. **의존성 설치** (런타임/dev 분리):
   ```bash
   # 런타임 의존성
   uv add numpy torch rlcard[torch] wandb hydra-core
   # 개발 전용 의존성 (배포 시 제외)
   uv add --dev pytest ruff mypy
   ```
3. **M1 Pro MPS 확인**:
   ```python
   import torch
   assert torch.backends.mps.is_available()
   ```
4. **프로젝트 구조 생성**:
   ```
   poker-ai/
   ├── CLAUDE.md                    ← 프로젝트 헌법
   ├── .claude/
   │   ├── agents/                  ← 서브에이전트
   │   │   ├── cfr-reviewer.md
   │   │   ├── rl-debugger.md
   │   │   └── test-writer.md
   │   ├── skills/                  ← 스킬 (Claude Code)
   │   │   └── poker-ai-dev/
   │   │       └── SKILL.md
   │   ├── hooks.json               ← 자동화 훅
   │   └── settings.json
   ├── src/poker_ai/
   │   ├── games/                   ← Kuhn, Leduc, NLH wrapper
   │   ├── algorithms/              ← CFR, CFR+, MCCFR, Deep CFR
   │   ├── networks/                ← PyTorch 모델
   │   ├── eval/                    ← exploitability, head-to-head
   │   └── utils/
   ├── tests/
   │   ├── unit/
   │   ├── integration/
   │   └── regression/              ← Nash 수렴 검증
   ├── experiments/                 ← Hydra configs
   ├── notebooks/                   ← 시각화·탐색
   └── pyproject.toml
   ```
5. **CLAUDE.md 작성** (별도 파일 참조)
6. **서브에이전트 3종 배치** (별도 파일 참조)
7. **Hooks 설정**: 파일 수정 시 `pytest --lf` 자동 실행
8. **W&B 프로젝트 생성**: `poker-ai-hunl`

### Exit Criteria
- [ ] `uv run python -c "import torch; print(torch.backends.mps.is_available())"` → True
- [ ] `uv run pytest` 가 "no tests ran" 출력 (아직 테스트 없음)
- [ ] Claude Code가 `CLAUDE.md`를 읽고 프로젝트 요약을 말할 수 있음
- [ ] `/agents` 명령으로 3개 서브에이전트 확인 가능
- [ ] W&B 로그인 완료

---

## Phase 1: Regret Matching + Kuhn CFR (Week 1~2)

### 목표
CFR의 가장 작은 단위를 수학적으로 검증된 형태로 구현한다. 이 Phase 이후 "내 CFR 구현은 옳다"는 확신을 가진 상태가 된다.

### Week 1: Rock-Paper-Scissors Regret Matching

**하루차**:
- `src/poker_ai/algorithms/regret_matching.py` 작성 (~50줄)
- `tests/unit/test_regret_matching.py`: 상대가 "항상 바위"일 때 내 전략이 "항상 보"로 수렴하는지 검증

**이틀차**:
- 두 플레이어가 동시에 regret matching 하면 균등 분포 (1/3, 1/3, 1/3)로 수렴하는지 시각화
- W&B에 convergence curve 로깅

### Week 2: Kuhn Poker Vanilla CFR

**이틀차까지**:
- `src/poker_ai/games/kuhn.py`: Kuhn Poker 게임 엔진 (3장 덱, 12 information set)
- `src/poker_ai/algorithms/vanilla_cfr.py`: 재귀적 CFR
- `src/poker_ai/eval/exploitability.py`: best response 계산 기반 exploitability

**나머지**:
- **검증 테스트** (regression):
  ```python
  def test_kuhn_cfr_converges_to_known_nash():
      trainer = VanillaCFR(KuhnPoker())
      trainer.train(iterations=10_000)
      # 알려진 Nash 게임 가치 = -1/18
      assert abs(trainer.game_value() - (-1/18)) < 0.001
      # Player 1 Jack bet 확률은 [0, 1/3] 범위 내
      assert 0 <= trainer.strategy("J")[BET] <= 1/3 + 0.01
  ```
- W&B에 exploitability 곡선 저장

### Exit Criteria
- [ ] RPS regret matching이 균등 분포로 수렴 (W&B 스크린샷)
- [ ] Kuhn CFR의 게임 가치가 **-1/18 ± 0.001** 로 수렴
- [ ] Kuhn CFR의 Player 1 Jack bet 확률이 **[0, 1/3]** 범위
- [ ] Exploitability가 10,000 iter 후 **< 5 mbb/g** (Zinkevich 2007 O(1/√T) 이론 수렴률과 일관된 실무 기준. Phase 2에서 CFR+로 더 엄격한 bar 설정 예정)
- [ ] 모든 unit test 통과

### 산출물
- `vanilla_cfr.py` (~150줄)
- `kuhn.py` (~100줄)
- `exploitability.py` (~80줄)
- 테스트 파일들
- W&B 수렴 곡선

---

## Phase 2: Leduc Hold'em — CFR+, MCCFR (Week 3~4)

### 목표
더 큰 게임(288 infoset)에서 CFR 개선 기법들의 효과를 체감한다. 샘플링이 왜 필요한지 체득한다.

### Week 3: Leduc 환경 + Vanilla CFR + CFR+

- `src/poker_ai/games/leduc.py`: 직접 구현 OR RLCard wrapper
  - 결정: **직접 구현 권장** (게임 트리 투명성)
- Vanilla CFR을 Leduc에 적용 → 10만 iter 돌리는 데 몇 분 걸리는지 측정
- `src/poker_ai/algorithms/cfr_plus.py`: 
  - 음수 regret 0 clipping
  - Alternating update
  - Linear averaging
- CFR vs CFR+ exploitability 수렴 속도 비교 (W&B에 두 곡선 중첩)

### Week 4: External Sampling MCCFR

- `src/poker_ai/algorithms/mccfr.py`:
  - External sampling 버전
  - Outcome sampling 버전 (선택)
- 반복당 시간 측정 → Vanilla 대비 몇 배 빠른지 확인
- **샘플링이 있으면 variance가 크다는 것을 직접 관찰** (seed 5개 돌려서 평균·표준편차)

### Exit Criteria
- [ ] Leduc Vanilla CFR이 exploitability **< 1 mbb/g** 로 수렴 (10만 iter)
- [ ] Leduc CFR+가 같은 exploitability에 **5~10배 빠르게** 도달
- [ ] External Sampling MCCFR이 반복당 CPU 시간에서 **10배 이상 빠름**
- [ ] 5 seed 평균 exploitability 곡선 (variance band 포함) 시각화
- [ ] 모든 regression test 통과

### 산출물
- Leduc 게임 엔진
- 3가지 CFR 변형 구현
- 알고리즘 비교 리포트 (`notebooks/phase2_comparison.ipynb`)

---

## Phase 3: Leduc Deep CFR (Week 5~7) — **종결 2026-04-26, NEGATIVE RESULT + path pivot**

> **Phase 3 Outcome**: Brown 2019 Deep CFR이 medium-scale games (Leduc 288 infosets)에서 architectural floor (σ̄_expl 140-150) 보유 — 9일간 4 axes (Cap / Huber / advantage_epochs / T) 검증으로 확증. Exit Criteria primary metric (σ̄_expl < 50 mbb/g) MISS. **Phase 4 algorithm을 Deep CFR scale-up이 아닌 MCCFR + abstraction (Pluribus path)으로 pivot**. 자세한 분석은 `PHASE.md` Phase 3 Conclusion 섹션 + 19 educational assets 참조.

### 목표
신경망이 regret table을 대체할 수 있음을 Leduc에서 확인한다. Deep CFR의 모든 구성요소를 손에 익힌다.

### Week 5: 네트워크 + Infoset 인코딩

- `src/poker_ai/networks/advantage_net.py`: MLP 또는 작은 ResNet (hidden 64~256)
- `src/poker_ai/networks/avg_strategy_net.py`: 비슷한 구조
- `src/poker_ai/utils/infoset_encoding.py`: Leduc infoset → fixed-size vector
- Network 단독 학습 테스트: 인위적 (infoset, regret) 페어를 MLP가 외우는지

### Week 6: Deep CFR 본체

- `src/poker_ai/algorithms/deep_cfr.py`:
  - External sampling traversal
  - Advantage memory (reservoir buffer)
  - Strategy memory (reservoir buffer)
  - Network 학습 루프
- M1 Pro MPS 백엔드로 학습
- 배치 사이즈, 학습률, CFR iter 당 네트워크 학습 epoch 튜닝

### Week 7: 검증 + 디버깅

- **가장 중요한 검증**: Deep CFR exploitability가 tabular CFR의 exploitability와 **같은 스케일로 수렴**하는지
  - 목표: Deep CFR이 **< 50 mbb/g** (tabular보다 약간 나쁜 건 정상)
- W&B에 tabular CFR vs Deep CFR 곡선 중첩
- 만약 수렴 안 하면 **`rl-debugger` 서브에이전트 활용**

### Exit Criteria
- [ ] Leduc Deep CFR이 exploitability **< 50 mbb/g** 수렴
- [ ] Tabular CFR 대비 같은 game 수에서 **3배 이내** exploitability
- [ ] M1 Pro MPS에서 학습 1 iteration이 수 초 이내
- [ ] Advantage network MSE loss가 감소하는 곡선 확인
- [ ] 체크포인트 저장/로딩 정상 작동

### 산출물
- Deep CFR 구현 (~500~800줄)
- Leduc에서 학습된 모델 체크포인트
- Phase 3 회고 노트 (`notebooks/phase3_reflection.ipynb`)

---

## Phase 4: RLCard NL Hold'em — **MCCFR + Abstraction (Pluribus path)** (Week 8~12+)

> **Algorithm choice 변경 (2026-04-26)**: Phase 3 Deep CFR architectural floor 발견 후, Phase 4는 Pluribus 표준 (Linear MCCFR + card/action abstraction + subgame solving)으로 진행. Brown 2019 Deep CFR scale-up 옵션은 reference로만 보존. Step 2 (Leduc abstraction validate, < 5 mbb/g target)에서 path 검증 후 commit.

### 목표
실제 HUNL에 근접한 환경에서 작동하는 봇을 만든다. 이 Phase가 프로젝트의 하이라이트다. **Algorithm**: Linear MCCFR (Phase 2 재활용) + card abstraction (E[HS²] 버킷팅) + action abstraction (3-6 사이즈) + subgame solving (Phase 5에서 확장).

### Week 8: NLH 환경 + 인코딩 설계

- RLCard의 `no-limit-holdem` 환경 래퍼 (`src/poker_ai/games/nlh_rlcard.py`)
- Action abstraction: `{fold, call, 0.5 pot, 1 pot, 2 pot, all-in}` 6개
- Card encoding: 
  - 옵션 A: 간단한 one-hot (suit × rank = 52차원)
  - 옵션 B: E[HS²] 버킷팅 (라운드당 50~200 버킷) ← 권장
- Betting history encoding: 4 라운드 × 최대 N 액션 × 채널
- **`/skill`로 `poker-ai-dev` skill 활성화**

### Week 9~10: MCCFR + Abstraction 스케일업 (Pluribus path)

- **Algorithm**: Phase 2 MCCFR (`src/poker_ai/algorithms/mccfr.py`) 재활용 — GameProtocol 사용으로 abstracted HUNL game에 즉시 transfer
- Card abstraction implementation:
  - E[HS²] 버킷팅 — round 별로 50~200 buckets
  - 또는 round-by-round refinement (preflop coarse → river fine)
- Action abstraction: `{fold, call, 0.5 pot, 1 pot, 2 pot, all-in}` 6개 (또는 5개)
- Self-play data 생성 병렬화 (M1 Pro 10 코어, multiprocessing.Pool)
- **이 단계에서 학습 시간은 며칠~1주일 단위**
- W&B dashboard에 실시간 모니터링
- **Note**: Brown 2019 Deep CFR scale-up은 Phase 3에서 architectural floor 발견 (PHASE.md Phase 3 Conclusion 참조). Pluribus 표준 path가 우리 인프라 (M1 Pro)와 목표 (intermediate-level human bot)에 fit.

### Week 11: Baseline 대결 평가

- `src/poker_ai/eval/head_to_head.py`: 두 봇 대결 + mbb/hand 계산
- Baseline 준비:
  1. **Random agent** (합법 액션 중 균등)
  2. **Always-call agent**
  3. **Rule-based agent** (직접 구현: "강한 패만 raise, 약한 패 fold, 중간은 call")
  4. **RLCard 내장 DQN** (있다면)
- 각 baseline 대상 10,000 hands 대결 → mbb/hand + CI 계산
- 이전 체크포인트와의 대결 → 학습 진행 확인

### Week 12: 디버깅 + 안정화

- exploitability를 직접 측정할 수는 없지만 (게임이 너무 커서), **이전 체크포인트 대비 꾸준히 이기는지**로 학습 진행 확인
- 만약 오래된 체크포인트가 최신을 이기는 현상 발생 → self-play 불안정 신호, K-best pool 도입 검토
- CPU 코어 수, 배치 사이즈, 학습률 등 최종 튜닝

### Exit Criteria
- [ ] Random agent 상대 **+500 mbb/hand 이상** (압도)
- [ ] Rule-based agent 상대 **+50 mbb/hand 이상** (95% CI)
- [ ] 초기 체크포인트 (iter 1000) 상대 **+200 mbb/hand 이상**
- [ ] 10,000 hands 대결 시간이 1시간 이내
- [ ] 학습 로그, 체크포인트, 평가 결과 전부 W&B에 보관

### 산출물
- HUNL 대응 Deep CFR 봇
- 학습된 모델 체크포인트 (마일스톤 5~10개)
- Baseline 비교 리포트

---

## Phase 5: 평가, 튜닝, 친구 대결 (Week 13~16)

### 목표
인간 플레이어 상대 실제 대결로 프로젝트 성공을 확정한다.

### Week 13: 대결 인터페이스

- `src/poker_ai/interface/cli_poker.py`: 터미널에서 봇과 대결
- 또는 간단한 Flask/FastAPI 웹 UI
- 핸드 히스토리 자동 저장 (추후 분석용)

### Week 14~15: 친구 대결 및 분석

- **중수 친구** 1~2명 섭외 (pot odds·3-bet 중 하나는 아는 수준)
- 세션당 200~500 hands × 3~5 세션
- 결과 분석:
  - 봇이 지는 상황은 어떤 패턴인가?
  - 특정 보드 텍스처에서 underperform?
  - Action abstraction이 부족해서 exploit 당함?
- W&B에 핸드 히스토리 통계

### Week 16: 개선 또는 마무리

**옵션 A** (결과 만족): 
- 프로젝트 회고 블로그 포스트 작성
- GitHub README 정비, 라이선스 결정, 공개

**옵션 B** (결과 부족):
- 가장 큰 exploit 패턴 1~2개 타겟팅
- action abstraction 확장 (예: 0.25 pot, 3 pot 추가)
- 추가 학습 1~2주

### Exit Criteria
- [ ] 중수 친구 상대 1,000+ hands 대결 데이터 확보
- [ ] 프로젝트 회고 문서 작성
- [ ] GitHub 리포지토리 정비 (README, 라이선스)

---

## 각 Phase별 Claude Code 활용 전략

| Phase | 주로 쓰는 에이전트 | 핵심 작업 스타일 |
|---|---|---|
| 0 | main agent | 셋업 스크립트, CLAUDE.md 초안 |
| 1 | `test-writer`, `cfr-reviewer` | TDD 엄격 적용, 수식 1:1 대조 |
| 2 | `cfr-reviewer` | 알고리즘 정확성 리뷰 |
| 3 | `rl-debugger`, `cfr-reviewer` | 신경망+CFR 동시 디버깅 |
| 4 | `rl-debugger` | 장시간 학습 모니터링, 실패 triage |
| 5 | main agent | UI 개발, 분석 스크립트 |

---

## 리스크 관리

| 리스크 | 완화 전략 |
|---|---|
| Kuhn CFR이 Nash에 수렴 안 함 | regression test로 빠른 감지, `cfr-reviewer` 에이전트 호출 |
| Leduc에서 CFR+가 Vanilla보다 안 빠름 | 구현 오류 신호, alternating update·음수 clip·linear averaging 세 가지 체크리스트 순회 |
| Deep CFR이 Leduc에서 발산 | 배치 사이즈 키우기, 학습률 낮추기, reservoir buffer 크기 확장, 네트워크 정규화 |
| HUNL 학습이 M1 Pro에서 너무 느림 | 네트워크 작게, abstraction 거칠게, self-play worker 8개로 제한, 필요시 RunPod $50 사용 |
| 친구 섭외 실패 | rule-based agent를 더 정교하게 만들어 대체 검증 |
| 학습 시간이 예상보다 길어짐 | 각 Phase exit criteria를 "완벽"이 아닌 "충분"으로 해석, 다음 Phase로 이동 |

---

## 예산

| 항목 | 비용 |
|---|---|
| 하드웨어 | $0 (기존 M1 Pro) |
| Claude Code | 기존 구독 |
| W&B | 무료 (개인 프로젝트) |
| RunPod 클라우드 버스트 (선택) | $30~100 (Phase 4 가속용) |
| **합계** | **$30~100** |

---

## 참고 자료 핵심 모음

**필독 논문**:
- Zinkevich et al. 2007, *Regret Minimization in Games with Incomplete Information* (Vanilla CFR 원조)
- Tammelin 2014, *Solving Large Imperfect Information Games Using CFR+*
- Lanctot et al. 2009, *Monte Carlo Sampling for Regret Minimization*
- Brown et al. 2019, *Deep Counterfactual Regret Minimization* (arXiv:1811.00164)
- Neller & Lanctot 2013, *An Introduction to Counterfactual Regret Minimization* (튜토리얼)

**필독 구현**:
- EricSteinberger/Deep-CFR (저자 본인 구현)
- google-deepmind/open_spiel (참조 표준)
- datamllab/rlcard (환경)

**블로그**:
- Justin Sermeno, *Vanilla CFR for Engineers*
- labml.ai, *CFR on Kuhn Poker* (주석 구현)
- Thomas Trenner, *Steps to Building a Poker AI* 시리즈

---

**다음 단계**: `CLAUDE.md`를 읽고 `/agents` 로 서브에이전트 확인 후 Phase 0 시작.
