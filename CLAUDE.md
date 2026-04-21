# CLAUDE.md — Poker AI Project

> 이 파일은 Claude Code가 이 프로젝트에서 어떻게 행동해야 하는지 정의한다.
> 길이를 200줄 이내로 유지한다 (LLM이 일관되게 따를 수 있는 지시는 ~200개가 상한).

## 프로젝트 한 줄 요약

HUNL(Heads-Up No-Limit Texas Hold'em)에서 **중급자 인간 플레이어를 이기는** Deep CFR 기반 포커 AI를 **MacBook M1 Pro**로 개발한다. 최종 목표는 Pluribus 급이 아니라 **개인 프로젝트 완주**다.

## 핵심 지시사항 (READ EVERY SESSION)

### 1. TDD를 강제한다
- 구현보다 **실패하는 테스트가 먼저** 작성되어야 한다
- 사용자가 "X를 구현해줘"라고 하면, **먼저 "FAILING 테스트를 작성하고 red output을 보여드리겠습니다"** 로 응답하라
- 실패 출력 확인 후에만 구현 작성
- Kuhn Poker처럼 수학적 정답이 있는 경우 **반드시 regression test**로 수렴값을 검증

### 2. CFR 수식 구현은 논문과 1:1 대조한다
- `vanilla_cfr.py`, `cfr_plus.py`, `mccfr.py`, `deep_cfr.py`를 작성할 때는 **해당 논문의 수식 번호를 주석에 명시**
- 예: `# Zinkevich 2007 Eq. (7): r^T_i(I, a) = v_i(σ^T, I·a) - v_i(σ^T, I)`
- 특히 counterfactual value, positive regret clipping, average strategy 계산은 **수식 오류가 조용히 성능 저하를 일으키는** 영역이므로 과하게 꼼꼼히 검토

### 3. 수렴 테스트는 seed 고정 + tolerance 포함
```python
@pytest.mark.parametrize("seed", [42, 123, 456])
def test_kuhn_converges(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    trainer = VanillaCFR(KuhnPoker())
    trainer.train(iterations=10_000)
    # Nash 게임 가치 = -1/18, tolerance 0.005
    assert abs(trainer.game_value() - (-1/18)) < 0.005
```

### 4. 파일당 300~500줄을 넘지 않는다
- 한 세션에서 한 모듈을 끝낼 수 있는 크기로 유지
- 300줄 넘으면 책임 분리 권유 (예: `cfr.py` → `cfr_core.py` + `cfr_utils.py`)

### 5. 포커 도메인 불변식 (INVARIANTS)
- **Information set key는 결정론적 문자열**. dict 순회 순서에 의존하면 안 됨
- **Perfect recall 가정**: 동일 플레이어의 과거 관측은 infoset에 모두 포함
- **합법 액션 마스킹**: 네트워크 출력에 항상 legal mask 적용 후 softmax
- **Chance node는 플레이어 아님**: regret 업데이트 대상 아님
- **2인 zero-sum 가정하에서만 Nash 수렴**: 멀티플레이어는 Phase 5 밖

### 6. 단위 (Units)
- 포커 성능 단위는 **mbb/hand (milli-big-blinds per hand)**
- exploitability 단위도 같음
- 항상 이 단위로 로그에 출력

## 프로젝트 구조

```
src/poker_ai/
├── games/        # 게임 엔진: kuhn.py, leduc.py, nlh_rlcard.py
├── algorithms/   # vanilla_cfr.py, cfr_plus.py, mccfr.py, deep_cfr.py
├── networks/     # PyTorch 모델 정의
├── eval/         # exploitability.py, head_to_head.py, best_response.py
└── utils/        # infoset_encoding.py, reservoir_buffer.py, seeding.py
```

## 빌드 · 테스트 · 실행

```bash
# 테스트
uv run pytest                    # 전체
uv run pytest tests/unit -x      # unit만, 첫 실패에서 멈춤
uv run pytest --lf               # 마지막 실패만 재실행

# 린트
uv run ruff check .
uv run mypy src/

# 실험 실행 (Hydra)
uv run python -m poker_ai.train +experiment=kuhn_vanilla_cfr
uv run python -m poker_ai.train +experiment=leduc_deep_cfr
```

## 하드웨어 주의사항 (M1 Pro)

- **GPU 디바이스**: `torch.device("mps")` 사용
- **알려진 이슈**: 
  - `float64` 일부 연산은 CPU로 fallback됨
  - `torch.cumsum`의 MPS 버그 이력 있음 → PyTorch 2.4+ 사용
  - 분산 학습 (DistributedDataParallel) 불가
- **메모리**: 통합 메모리 16GB 중 학습에 쓸 수 있는 건 현실적으로 8~10GB
- **CPU**: 성능 코어 8개 + 효율 코어 2개. self-play worker는 **최대 8개**로 제한

## 실험 관리

- **매 실험은 반드시 W&B 로깅**: `wandb.init(project="poker-ai-hunl", config=cfg)`
- **로깅 필수 항목**: exploitability, advantage_loss, strategy_loss, game_value, iter_per_sec, memory_usage
- **체크포인트**: 1000 iter마다 `{iter, model_state, optimizer_state, rng_state, buffer_size}` 전부 저장
- **Hydra**: 하이퍼파라미터는 `experiments/conf/` 아래 yaml로, 실험 재현 가능하게

## 절대 하지 말 것

- ❌ `checkpoints/`, `wandb/`, `.venv/` 커밋
- ❌ 하드코딩된 path (`/Users/xxx/...`) — 항상 `Path.home()` 또는 hydra config
- ❌ `print` 로 디버깅 (`logging` 또는 W&B 사용)
- ❌ CFR 구현에서 random seed 누락
- ❌ 합법 액션 마스크 없이 argmax
- ❌ Phase exit criteria 미충족 상태로 다음 Phase 이동 (사용자가 명시적으로 요청하지 않은 한)

## 서브에이전트 활용

다음 상황에서는 서브에이전트를 명시적으로 호출한다:

| 상황 | 에이전트 | 호출 방법 |
|---|---|---|
| CFR 구현이 수렴 안 함 | `cfr-reviewer` | "cfr-reviewer 에이전트로 `vanilla_cfr.py` 검토해줘" |
| 학습 곡선이 이상함 (loss 폭발, exploitability 증가) | `rl-debugger` | "rl-debugger 에이전트로 W&B run ID [xxx] 분석해줘" |
| 새 알고리즘 구현 시작 | `test-writer` | "test-writer 에이전트로 Deep CFR 테스트부터 작성해줘" |

## 스킬 (Claude Code Skills)

이 프로젝트는 `poker-ai-dev` skill을 제공한다. Phase 4(HUNL)부터 자동 활성화 권장:
```
/skill activate poker-ai-dev
```

## 대화 스타일

- 한국어로 대화. 코드 주석은 영어 (국제적 가독성)
- 모호한 요청은 되묻기: "지금 Phase 어느 단계죠?"
- 구현 전 항상 접근 방식을 **한 단락으로 요약**한 뒤 "이 방향 맞으신가요?" 체크

## 현재 Phase 추적

Phase 진행 상황은 `PHASE.md`에 추적한다. 매 세션 시작 시 `PHASE.md`를 읽고 현재 작업 컨텍스트를 확인할 것.

## 참고: 포커 용어 빠른 참조

- **mbb/hand**: milli-big-blinds per hand. 100 mbb/hand = 10 bb/100 hands
- **Infoset**: information set. 동일 플레이어의 같은 관측 시퀀스
- **CFV**: counterfactual value. 해당 infoset에 도달했을 때의 기대값
- **Exploitability**: 두 플레이어가 best-respond 할 때의 평균 값. Nash에서 0
- **Best response**: 상대 전략 고정 시 최적 대응 전략
- **Perfect recall**: 자신의 과거 관측·액션을 모두 기억
- **Abstraction**: card/action bucketing을 통한 게임 크기 축소

---

**이 파일을 매 세션 시작 시 자동으로 읽는다. 수정 시 간결함을 유지할 것.**
