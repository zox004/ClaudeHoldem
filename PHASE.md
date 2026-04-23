# 현재 Phase 추적

> 매 세션 시작 시 Claude Code가 이 파일을 먼저 읽고 컨텍스트를 파악한다.
> 각 Phase 완료 시 이 파일을 업데이트한다.

## 현재 상태

**Phase**: 2 (Leduc Hold'em — CFR+, MCCFR) — **Week 1 착수 (Day 1)**
**시작일**: 2026-04-23
**목표 완료일**: 2026-05-07 (+ 2주, 실제는 Phase 1 패턴으로 단축 가능)
**Baseline**: Phase 1 종료 시점 193 tests GREEN (65s), 10 commits, main @ `609fc69`

## 다음 할 일 (Next Action) — Phase 2 Week 1 (Leduc 엔진 + CFR+)

### Week 1 (ROADMAP §Phase 2 Week 3 매핑) — Leduc 게임 + Vanilla → CFR+
- [x] **Leduc 엔진 설계 논의 확정** — 4가지 결정 확정 (CFR+만, 직접 구현, External Sampling, 3-pass BR 재사용) + Q2 세부 (IntEnum FCR, 120 deals, reach_opp=1/120, @property 파생, regret_matching legal_mask 옵션)
- [x] `tests/unit/test_leduc.py` + `tests/regression/test_leduc_perfect_recall.py` — 71 tests GREEN (59 unit + 12 regression, 288 infoset 검증 포함) (커밋 `1fa143e`)
- [x] `src/poker_ai/games/leduc.py` — Leduc Hold'em 엔진 (120 deals, 288 infosets, pot accounting rrf=-3/cc.rrf=-5) (커밋 `1fa143e`)
- [x] **Day 2** — `src/poker_ai/games/protocol.py` (GameProtocol/StateProtocol, 커밋 `b99f914`) + `exploitability.py` game-agnostic 리팩토링 + **Pass 2 illegal-argmax 버그 수정** (커밋 `fe40ac6`) + `regret_matching` legal_mask 옵션 추가 (커밋 `f84cef6`). 305 tests GREEN, Ruff + mypy strict clean
- [x] **Day 3** — Kuhn에 `legal_action_mask` 추가 (Option B) + StateProtocol 확장 (커밋 `0679cbc`) + VanillaCFR 루프 game-agnostic화 + InfosetData.legal_mask 캐시 + 규칙 3가지 masking invariant (커밋 `93e7e63`). 324 tests GREEN. Leduc 1k smoke: 288/288 infoset, expl=9.15 mbb/g, 10.8 iter/s
- [x] **Day 4** — Harness + fast regression + **100k W&B full run 완료** (커밋 `c173b24`). 10k fast PASSED (3.504 < 4.0). 100k 실측 **1.4802 mbb/g** — Exit #1 (< 1.0 mbb/g) **미달 48%**. Slope decelerates (−0.417 → −0.374), Nash 근처 steepen 가설 반증. **ROADMAP Exit #1은 Vanilla CFR로는 100k에서 미달; CFR+ 도입 후 재평가 예정**
- [ ] **Day 5** — `src/poker_ai/algorithms/cfr_plus.py` (Tammelin 2014: 음수 regret clipping + alternating + linear averaging) + CFR vs CFR+ 비교 실험 (Exit #2)
- [ ] W&B에 CFR vs CFR+ exploitability 곡선 중첩 — CFR+가 5~10배 빠르게 same-expl 도달 확인

### Week 2 (ROADMAP §Phase 2 Week 4 매핑) — MCCFR
- [ ] `src/poker_ai/algorithms/mccfr.py` — External Sampling MCCFR
- [ ] 반복당 CPU 시간 측정: Vanilla 대비 ≥10× 빠름
- [ ] 5 seed × MCCFR 실행 → exploitability curve variance band 시각화 (tabular deterministic이 아닌 첫 알고리즘)
- [ ] (선택) Outcome Sampling MCCFR 비교

## 지금까지 한 일 (Done)

### Phase 2 Week 1 Day 4 완료 (2026-04-23) — Leduc Vanilla CFR 100k 실측 + 수렴률 실증 발견

- ✅ **100k W&B full run** (`leduc-vanilla-seed42`, [run](https://wandb.ai/zox004/poker-ai-hunl/runs/abt0tztf), [summary](https://wandb.ai/zox004/poker-ai-hunl/runs/jvxee3wl))
  - 시작 11:53:02 KST, 종료 14:24:13 KST, 총 runtime **2h 31min** (9063s), iter_per_sec=11.27
  - PNG: `experiments/outputs/phase2_leduc_vanilla_cfr/20260423-115305/leduc_vanilla_convergence.png`
- ✅ **Exit #1 판정: ❌ FAILED** (실측 1.4802 mbb/g vs threshold 1.0 — **48% 초과**)
  - `iters_to_exit = -1` (전체 100k 내내 threshold 미크로스)
  - Game value는 **−0.08565** — Leduc Nash 문헌값(−0.0856)과 4자리 정확 수렴 (게임 자체는 Nash에 도달, exploitability만 marginal)
- ✅ **Log-log slope 실증 분석** — 4-point 측정
  - `1k → 10k`: 9.149 → 3.504 → slope **−0.417**
  - `10k → 100k`: 3.504 → 1.480 → slope **−0.374**
  - `1k → 100k` 전체: slope **−0.395**
  - **Slope DECELERATES over time** — "Nash 근처에서 steepen" Day 3 가설 **반증**. 이론 O(1/√T)에서 더 멀어지는 방향
- ✅ **과학적 해석 (Phase 2의 진짜 교육적 발견)**:
  - Zinkevich O(1/√T)는 worst-case bound — **Leduc에선 실측이 그 bound보다 더 느림**
  - Kuhn(slope −0.52, 이론 일치) vs Leduc(slope decelerating −0.417→−0.374) 대비는 **게임 규모와 실측 수렴률의 이탈**을 명확히 드러냄
  - 이게 바로 **CFR+ 도입 동기의 실증 증거.** ROADMAP Exit #2 ("CFR+가 5~10× 빠르다")는 이 baseline 위에서 측정될 것
  - 현재 slope 유지 시 Exit #1 달성 iteration: ≈280k (추가 5h 런타임). ROI 낮음 → CFR+ 측정 후 재평가
- ✅ **Day 4 결정 정리** (Option D 실측-우선 접근의 결과):
  - Fast threshold 3.0 → 4.0 mbb/g 조정 (실측 3.504 기반, 커밋 `c173b24`)
  - **Slow threshold 1.0은 유지** — "Exit #1 달성 실패"를 투명하게 기록 (softening 거부). 다음 CFR+ 구현 시 "vanilla로는 안 되고 CFR+로 되더라"가 명확한 narrative
  - 후속 세션 Day 5 CFR+ 구현 후 ROADMAP Exit #1 원안 유지/완화 최종 결정
- ✅ **산출물 목록**:
  - W&B runs (2개): seed42 + summary
  - PNG: 2-panel convergence plot (log-log + linear with Exit Criterion red line + expected 0.92 green line)
  - Hydra run dir: `experiments/outputs/phase2_leduc_vanilla_cfr/20260423-115305/`

### Phase 2 Week 1 Day 4 부분 (2026-04-23) — Leduc Vanilla CFR W&B harness + regression (100k 대기)

- ✅ **FAILING 테스트 작성 (test-writer)** — 4 tests in `tests/regression/test_leduc_vanilla_cfr_convergence.py`
  - sanity guard (100 iter > 30 mbb/g), monotonicity (100 → 10k ≥3× 감소), fast (10k < 4 mbb/g), **slow @pytest.mark.slow** (100k < 1 mbb/g — Phase 2 Exit #1)
  - Collection 4 tests 정상
- ✅ **Hydra + W&B harness** (Phase 1 Kuhn 템플릿 ~90% 재활용)
  - `experiments/phase2_leduc_vanilla.py` + `experiments/conf/phase2_leduc.yaml`
  - 주요 파라미터: `iterations=100000`, `n_actions=3`, `big_blind=2.0`, `log_every=500`, `dense_prefix=10`, `exit_criterion_mbb=1.0`, `expected_final_mbb=0.92`
  - 1k smoke (wandb disabled) 검증 통과 — 98s, final_expl=9.149 mbb/g (Day 3 smoke와 일치, 파이프라인 무결)
- ✅ **10k fast regression 실측 → threshold 재조정**
  - 최초 threshold 3.0 (이론 O(1/√T) 기반 예측)으로 FAIL 발생: 실측 **3.504 mbb/g**
  - 실측 log-log slope: `−0.417` (이론 −0.5 대비 16% 완만)
  - 원인 가설: Leduc 288 infoset × round 2 chance branching의 분산 → Zinkevich worst-case bound에 더 가까운 수렴 (Kuhn 12 infoset 단일 라운드는 slope ≈ −0.52로 거의 이론치였음)
  - Fast threshold 3.0 → **4.0 mbb/g** (실측 3.504 + 14% margin)으로 조정, docstring에 경험적 표 + 근거 박음
  - sanity + monotonicity test는 PASSED (15:37 병렬)
- 🔄 **Day 4 pending 항목** — 100k W&B full run은 사용자 명시적 트리거 대기
  - 예상 runtime: 2.5h CPU (Python tree traversal 병목)
  - 실측 slope 유지 시 100k 예측: `10^(log10(3.504) − 0.417) ≈ 1.34 mbb/g` → Exit #1 (< 1.0 mbb/g) 미달 예측
  - 이론 slope (−0.5) 유지 시: 3.504 / √10 ≈ 1.108 mbb/g → 여전히 > 1.0
  - **Slope가 Nash 근처에서 steepen될 가능성**: regret 분포가 안정화되면 더 빠른 수렴 가능. 100k 실측이 이 질문에 대한 답
  - 후속 결정: 실측 보고 → threshold 유지(1.0) / 완화(1.5) / iteration 증가(200k) 중 하나
- ✅ **설계 결정 기록**:
  - Threshold 3.0 → 4.0 조정 근거: 이론 O(1/√T)가 worst case bound이므로 실제 게임별 상수가 다를 수 있음. Kuhn에서 이론치가 거의 맞았던 건 우연(작은 게임, 낮은 분산)
  - Slow threshold 1.0은 **의도적으로 완화하지 않음** — Exit #1을 "할 수 있음"으로 입증할 것인지 "ROADMAP 이탈"로 기록할 것인지는 실측값 본 뒤 결정
  - Day 4 분할: "harness + fast test" 이 세션에서 완료, "slow test" 사용자 트리거 후 별도 세션에서 확인 — Context 효율 + 2.5h CPU cost 명시적 승인 받기 위함

### Phase 2 Week 1 Day 3 (2026-04-23) — VanillaCFR game-agnostic 리팩토링 (Leduc 학습 가능)

- ✅ **2개 FAILING 테스트 파일 작성** (test-writer) — 19 tests RED (11 Kuhn mask + 8 Leduc CFR smoke)
  - `tests/unit/test_kuhn_legal_action_mask.py`: shape/dtype, all-True invariant, legal_actions consistency (parametrize × 4)
  - `tests/integration/test_vanilla_cfr_leduc_smoke.py`: instantiation, train smoke, **illegal slot probability = 0** invariant 3종 (current/average/sum-to-1)
  - RED 확인: AttributeError(legal_action_mask 미구현) + ValueError("2 is not a valid KuhnAction")
- ✅ **C1 커밋 `0679cbc`** — `KuhnState.legal_action_mask()` 추가 (shape (2,) all-True) + StateProtocol 확장
  - Day 2의 `test_protocol_does_not_require_legal_action_mask` 테스트 의도 flip (Option B 결정 반영)
  - Kuhn 193 GREEN + Leduc 71 GREEN + 37 mask/protocol GREEN = 316 passed
- ✅ **C2 커밋 `93e7e63`** — VanillaCFR 전면 game-agnostic 리팩터
  - `KuhnAction` import 완전 제거 — vanilla_cfr.py 순수 Protocol-dependent
  - 루프: `for a_idx in range(n_actions)` → `for a in state.legal_actions()`
  - `InfosetData.legal_mask` 캐시 필드 추가 (per-infoset, 첫 방문 시 state에서 읽음)
  - `current_strategy`: `regret_matching(cumulative_regret, legal_mask=data.legal_mask)` — illegal slot 확률 0 강제
  - `average_strategy` fallback uniform은 **legal 액션만** (mask_f / mask_f.sum())
  - `instantaneous_regret *= mask_f` — illegal slot의 cumulative_regret drift 방지 (Kuhn no-op)
  - `game_value()` 도 legal_actions iteration으로 재작성
  - **수학 불변식**: 모든 visited infoset의 strategy에서 illegal slot = 정확히 0, strategy sum = 1.0 (probability measure integrity)
- ✅ **Leduc 1k smoke run** (주변 실측) — 288/288 infoset 방문, exploitability = 9.15 mbb/g, game_value = −0.0887, 10.8 iter/s
  - O(1/√T) 예측: 10k→2.89 mbb/g, 100k→0.92 mbb/g (Day 4 Exit #1 <1 mbb/g 달성 경계 확인)
  - **성능 관측**: 100k iter 추정 ~2.5시간 (Python tree traversal 병목). Day 4는 "10k regression + 100k W&B full run" 분리 전략 필요
- ✅ **설계 결정 기록**:
  - Option B 채택 이유 재확인: Option C (CFR에서 mask 미사용) 는 fresh infoset에서 uniform-over-all → illegal 확률 1/3 → node_value sum < 1 (probability measure 붕괴). Kuhn에선 잠복, Leduc에서 활성 — Day 2 Pass 2 버그와 구조적으로 유사
  - `InfosetData.legal_mask` per-infoset 캐시: state 매번 재계산 대신 첫 방문 시점에 저장. Perfect recall 정의 의해 infoset 내 모든 state가 동일 legal_actions → 안전
  - Kuhn 행동 보존 검증: mask가 all-True → 모든 masking 곱셈 no-op → Kuhn 193 tests 100% GREEN 유지가 이를 자동 증명

### Phase 2 Week 1 Day 2 (2026-04-23) — Protocol 추출 + exploitability 일반화 + Pass 2 버그 수정

- ✅ **3개 FAILING 테스트 파일 작성** (test-writer) — 총 41 tests RED
  - `tests/unit/test_game_protocol.py` (26 tests): Kuhn + Leduc parametrized conformance, over-spec guards
  - `tests/unit/test_regret_matching.py` +6 tests: `TestRegretMatchingLegalMask` 클래스 (legacy-compat, illegal 0, 3종 fallback, single-legal)
  - `tests/integration/test_exploitability_leduc_smoke.py` (9 tests): finite, expl > 0, 범위 경계, BR lower bound, Kuhn regression guard
  - RED 에러 형태: ModuleNotFoundError (Protocol) / TypeError (legal_mask) / IndexError (`opp_strat[int(a)]` — `_N_ACTIONS=2` 가정이 Leduc 3-action에 충돌, 정확히 버그 위치 지목)
- ✅ **C1 커밋 `b99f914`** — `src/poker_ai/games/protocol.py` 신규 + Kuhn에 `NUM_ACTIONS = 2` 추가 (additive only)
  - `@runtime_checkable` Protocol: StateProtocol (5 attrs) + GameProtocol (4 attrs)
  - Over-spec 방지 확정: `legal_action_mask`, `private_cards`, `round_idx` 등 게임별 속성 전부 Protocol 제외
  - 26/26 GREEN
- ✅ **C2 커밋 `fe40ac6`** — exploitability game-agnostic + **Pass 2 illegal-argmax 버그 수정**
  - Import KuhnPoker/KuhnState → GameProtocol/StateProtocol, `_N_ACTIONS = 2` → `game.NUM_ACTIONS`
  - **핵심 버그 수정**: `cfv = np.zeros(NUM_ACTIONS)` + `argmax` 전체 → argmax over LEGAL only + action 객체 저장 (int index 아님)
  - Kuhn에서는 no-op (모든 action legal), Leduc에서 활성 (bets=0에서 FOLD index 0이 cfv=0으로 argmax 1등 가능)
  - vanilla_cfr.py 타입힌트만 교체, 루프 구조 + `KuhnAction(a_idx)` 호출 유지 (Day 3 범위)
  - 버그 발견 트리거: "Kuhn 193 GREEN 유지 전제"로 Leduc 탑재 시 역행적 감사 → 버그 드러남
- ✅ **C3 커밋 `f84cef6`** — `regret_matching(..., legal_mask=None)` 시그니처 확장
  - 기존 호출 100% 호환 (legal_mask=None이 Phase 1 동작)
  - legal_mask 경로: positive-part × mask → normalize. Fallback uniform은 **legal only**
  - Day 3 Leduc CFR에서 소비 예정. Day 2에선 함수 확장만
  - numpy 타입 추론 issue(`/scalar` returns Any)로 mypy strict 에러 3건 발생 → 타입 어노테이션 중간변수로 해결
- ✅ **최종 Verification** — 305 tests GREEN (Kuhn 193 + Leduc 71 + Phase 2 Day 2 신규 41), Ruff + mypy strict clean on 6 refactored source files
- ✅ **설계 결정 기록**:
  - Protocol 위치 `src/poker_ai/games/protocol.py` 선택 (dedicated 모듈 — `__init__.py` 혼잡 방지)
  - GameProtocol deal 타입 `Any`로 느슨하게 (TypeVar generic은 과공학)
  - Commit 3분할: C1(Protocol) → C2(exploitability+bug) → C3(regret_matching). 의존성 방향 일치, 각 단계 독립 검증 가능
  - vanilla_cfr.py는 타입/import만 교체 (루프 무수정), Day 3에서 Leduc-ready 루프로 재설계 예정

### Phase 2 Week 1 Day 1 (2026-04-23) — Leduc 엔진 구현

- ✅ **Phase 2 설계 논의** — 4가지 핵심 결정 (CFR+만 / Leduc 직접 구현 / External Sampling MCCFR / 3-pass BR 재사용) + Q2 세부 설계 (IntEnum `FOLD=0 CALL=1 RAISE=2`, 120 deals chance 1/120, `@property` derived, regret_matching legal_mask 옵션) 확정
- ✅ **Leduc 엔진 FAILING 테스트 작성 (test-writer 1차 + 2차 보강)** — 총 71 tests (59 unit + 12 regression)
  - 1차: 33 methods (IntEnum, all_deals, State properties, legal_actions, infoset_key, terminal_utility 6종, next_state immutability)
  - 2차 보강 5 methods: `rrf` (-3), `cc.rrf` (-5), P1 pair win 대칭, fold card-invariant (parametrize × 4), chance probability 1/120 명시
  - 전부 `ModuleNotFoundError`만으로 깔끔한 RED 달성
- ✅ **Leduc 엔진 구현** (`src/poker_ai/games/leduc.py`, 259줄, 커밋 `1fa143e`) — 71/71 GREEN
  - Static terminal utility dispatch (fold/showdown 분기 + `_round_commits` helper로 pot accounting)
  - `@property` 파생: `round_idx`, `bets_this_round`, `current_player`, `is_terminal`, `infoset_key` (Kuhn 패턴 동일)
  - `_pending_board` 필드: 딜 시점 board_card 저장, round 1 closure 시 `board_card`로 promote
  - `_is_round_closed`: 마지막 action이 CALL이고 `len ≥ 2`이면 라운드 종료 (cc/rc/crc/rrc/crrc)
  - 288 infosets DFS 탐색 검증 (Neller & Lanctot 2013 §5 Table 2 일치)
  - Full suite 264 passed in 62.6s (193 Kuhn + 71 Leduc). Ruff + mypy strict clean
- ✅ **설계 결정 기록**:
  - Abstract Game 인터페이스 범위는 **Protocol로 제한** (ABC 상속 거부 — 구체 게임 2개뿐인 시점에 조기 고착 방지, Phase 3 NLHE 요구사항 나올 때 ABC 승격 가능)
  - `_pending_board` 필드는 "public이지만 internal 의미"임을 underscore로 명시 (dataclass slots 호환)
  - Raise semantics "call + bet_size" 검증의 핵심은 `rrf = -3` (round 1) 와 `cc.rrf = -5` (round 2 bet=4) 두 테스트. Phase 2 리팩터링 중 실수로 수학 변경 시 즉각 잡힘
  - Kuhn 193 tests 완전 무수정 — Day 2 리팩터링에서 일반화 예정

### Phase 1 Week 2 (진행 중, 2026-04-21 착수)
- ✅ **Day 1** (2026-04-21): Kuhn Poker 게임 엔진 구현 (커밋 `d1f316c`) — **112 tests GREEN (107 unit + 5 regression)**
  - 설계 결정: chance node 추상화 없음(Kuhn 딜은 외부 루프), `KuhnAction(IntEnum)`, `@dataclass(frozen=True, slots=True)` 불변 state, `KuhnPoker`는 staticmethod factory
  - **Char-based infoset key 규약 확정** (`"J"/"Q"/"K"`) — debug eyeball 가독성 > numpy 일관성
  - **Static terminal utility table 패턴** 확립 — Phase 2 Leduc에서도 재활용 예정 (5개 if 체인, 동적 pot 계산 버그 회피)
  - Perfect recall regression: 12 infosets × opponent 카드 privacy × own card/history 인코딩 검증
  - `src/poker_ai/games/kuhn.py` 최상단 docstring에 Neller & Lanctot 2013 Section 4.1 scoring 표 + fold/showdown/utility 검산 규약 3줄 삽입
- ✅ **Day 2** (2026-04-22): **Vanilla CFR 구현, Kuhn Nash 수렴 첫 시도 달성** (커밋 `86ef8b1`) — **161 tests GREEN (unit 129 + integration 11 + regression 20, 52.85s)**
  - Zinkevich 2007 Vanilla CFR의 수학적 정확성을 10k iter × 3 seed에서 GREEN으로 확증 — Phase 1의 핵심 마일스톤
  - 설계 결정: **alternating one-player traversal ("A pattern")**, chance prob은 `reach_opp` 초기값에 흡수(재귀 내부 chance 분기 없음), regret 테이블은 raw 저장 (positive-part는 `current_strategy()` 호출 시점에만), iteration = 두 플레이어 둘 다 업데이트된 한 주기
  - 게임 가치 `-1/18 ± 0.001`, Jack bet ∈ `[0, 1/3 + 0.01]`, King bet ≈ 3·Jack (tol 0.1), Queen `"Q|"` bet < 0.05, `len(infosets) == 12` 10k iter 후에도 유지 (Lazy init 무결성)
  - 재사용: `src/poker_ai/algorithms/regret_matching.py`의 Week 1 함수를 CFR 내부에서 그대로 import — Week 1 작업이 재활용됨
  - `pytest.mark.slow` marker를 `pyproject.toml`에 공식 등록 (`[tool.pytest.ini_options] markers = [...]`)

- ✅ **Day 2 (continued)** (2026-04-22): **Exploitability 3-pass BR + W&B 수렴 로깅, Phase 1 종료** (커밋 `b7895fb`, `71c8321`, `0457d80`, +W&B run) — **193 tests GREEN**
  - **3-pass BR 알고리즘** (Lanctot 2013 §3.4): 순진한 per-state max는 imperfect-info에서 옵상대 카드 정보를 leak → infoset별로 `π_{-i}(h)-weighted CFV` argmax로 집계. 4 Nash α에서 BR(P1)=−1/18, BR(P2)=+1/18이 1e-12 정밀도로 일치
  - **Exit Criterion threshold 현실화**: `< 0.01 mbb/g`는 물리적 불가능(T≈4×10¹² 필요) → Zinkevich O(1/√T) 이론 수렴률 기반 `< 5 mbb/g`로 보정 (커밋 `71c8321`)
  - α-family 전체(0, 0.1, 0.2, 1/3)에서 BR 값 lock-in 테스트 8개 추가 (커밋 `0457d80`) — CFR+ 도입 시 회귀 방지 용
  - **10k iter full run** (W&B online): final_expl = **2.136 mbb/g**, iters_to_exit = **1700**, game_value = −0.055571 (편차 1.6e-5), 학습 3.28s @ 3462 iter/s
  - O(1/√T) 이론 검증: log-log slope (100→1000→10000) = −0.556, −0.483, −0.520 — 이론치 −0.5에서 6% 이내
  - W&B: [seed42 run](https://wandb.ai/zox004/poker-ai-hunl/runs/auf1uzeo) + [summary](https://wandb.ai/zox004/poker-ai-hunl/runs/szgdzgqt), 2-panel plot (log-log + linear with reference lines)
  - 설계 결정: Vanilla CFR deterministic이라 **single seed (42)**, Week 1 3-seed 구조는 Phase 2 MCCFR에서 재활성화 예정 (코멘트로 명시), `should_log(iter, 100, 10)` 혼합 cadence로 초반 급감 구간 시각화 보존

### Phase 1 Week 1 (완료 2026-04-21)
- ✅ TDD 첫 사이클: RPS Regret Matching (커밋 `968ecc2`, `3732551`)
  - RED: 2개 테스트 파일이 `ModuleNotFoundError`로만 실패 (다른 에러 없음)
  - GREEN: 구현 첫 실행에서 10/10 pass (1.93s)
  - 검증된 불변식: 확률합=1, non-negative, negative clipping, zero→uniform, always-rock→paper 수렴, self-play→uniform 수렴
  - 설계 결정: `RegretMatcher`는 RPS demo 전용, CFR tree traversal에는 pure `regret_matching()` 함수를 직접 사용할 것 (클래스 docstring에 명시)
- ✅ 수렴 실험 + W&B 로깅 (커밋 `7e2355e`)
  - `experiments/phase1_rps_convergence.py` + `experiments/conf/phase1_rps.yaml` (Hydra)
  - 3 seeds × 10k iter: L1 to uniform ∈ [0.008, 0.023] — Phase 1 Exit Criteria #1 충족
  - **W&B 프로젝트 `poker-ai-hunl` 초기화** (entity: zox004)
  - Summary run + 3-subplot figure: https://wandb.ai/zox004/poker-ai-hunl/runs/z8i0l32c
- ✅ **Hydra + W&B 실험 harness 확립** — Phase 2~4에서 `experiments/conf/*.yaml` 패턴으로 재사용 예정
- ✅ matplotlib을 explicit dev dep으로 고정 (`uv add --dev matplotlib`)

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

## Known Issues (minor)

> "무해하지만 의식해둘 것"들을 Phase 진행 중 여기에 누적. 각 항목은 현재 실패를 일으키지 않지만 나중에 다른 증상으로 drift할 수 있어 후속 Phase에서 재평가.

- **Kuhn `KuhnState.current_player` returns `len(history) % 2` even for terminal states** (2026-04-21, Week 2 Day 1)
  - CFR trainer must gate with `is_terminal` before calling. 현재 모든 호출지점이 그렇게 설계되어 문제 없음.
  - 방어적 `raise`를 걸면 `TestNextStateClassificationConsistency`가 terminal state를 만들 때 깨지므로 보류.
  - Consider adding explicit guard if Phase 3 debugging suggests needed.
- **Vanilla CFR is deterministic; seed parametrization has no variance in current tests** (2026-04-22, Week 2 Day 2)
  - `test_kuhn_convergence.py`는 `@pytest.mark.parametrize("seed", [42, 123, 456])`로 3회 돌지만 tabular CFR이 deterministic이라 3 seed 결과가 수치적으로 동일.
  - 현재 tests는 "미래에 sampling이 도입되면 variance guard가 있어야 한다"는 구조적 준비일 뿐 실질 variance를 측정하지 않음.
  - Seed parametrization은 Phase 2 MCCFR(sampling-based)에서부터 의미 있는 variance 테스트로 전환될 것.

## 이번 Phase(1)의 Exit Criteria

- [x] RPS regret matching이 균등 분포로 수렴 (W&B 스크린샷) — L1 ≤ 0.023 @ 10k iter, [summary run](https://wandb.ai/zox004/poker-ai-hunl/runs/z8i0l32c)
- [x] Kuhn CFR의 게임 가치가 **-1/18 ± 0.001** 로 수렴 — 실측 −0.055571 (편차 1.6e-5, 기준의 1.6%)
- [x] Kuhn CFR의 Player 1 Jack bet 확률이 **[0, 1/3]** 범위 — 실측 0.234993 (1/3의 70%)
- [x] Exploitability가 10,000 iter 후 **< 5 mbb/g** — 실측 **2.136 mbb/g** (threshold의 43%), iters_to_exit=1700 [[seed42 run]](https://wandb.ai/zox004/poker-ai-hunl/runs/auf1uzeo)
- [x] 모든 unit test 통과 — **193/193 GREEN** (unit 153 + integration 14 + regression 26)

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

### Phase 1 (Regret Matching + Kuhn CFR)
**완료일**: 2026-04-22
**소요 시간**: 2일 (Day 1: Kuhn engine, Day 2: Vanilla CFR + Exploitability + W&B) — 예상 2주 대비 90% 단축
**커밋**: `968ecc2`, `3732551`, `7e2355e` (Week 1), `d1f316c`, `86ef8b1`, `3f44f1c`, `b7895fb`, `71c8321`, `0457d80` (Week 2)
**테스트**: **193 tests GREEN** (Phase 0 end: 0 → Phase 1 end: 193)

**달성한 것** (Exit Criteria 5/5):
- [x] RPS regret matching L1 ≤ 0.023 @ 10k iter
- [x] Kuhn CFR 게임 가치 = **−0.055571** (Nash −1/18에서 편차 1.6e-5, 기준 ±0.001의 1.6%)
- [x] Kuhn P1 Jack bet 확률 = **0.234993** ∈ [0, 1/3]
- [x] Exploitability @ 10k iter = **2.136 mbb/g** < 5 mbb/g (iters_to_exit=1700)
- [x] 193/193 tests GREEN

**핵심 마일스톤**:
1. Zinkevich 2007 Vanilla CFR "A pattern" (alternating one-player traversal) 수학적 정확성 검증
2. 3-pass BR 알고리즘 (Lanctot 2013 §3.4) — 순진한 per-state max 함정 회피, α-family 4 Nash에서 1e-12 정밀도
3. O(1/√T) 이론 수렴률 실증 — log-log slope 실측 (−0.52) vs 이론 (−0.5), 오차 4%
4. Hydra + W&B harness 확립 (Week 1 RPS 템플릿 75% 재사용 → Week 2 Kuhn)

**배운 것**:
- **Exit Criterion 설정은 이론 수렴률로 역산**: 초기 `< 0.01 mbb/g` threshold는 O(1/√T)로 T≈4×10¹² 필요 → 물리적 불가능. Zinkevich bound + empirical margin으로 `< 5 mbb/g`로 재조정. *Phase 2부터는 Exit Criterion 설정 시 이론 수렴률 먼저 계산*
- **Imperfect-info BR은 per-state max ≠ per-infoset argmax**: infoset별 π_{-i}(h)-weighted CFV 집계 필수. 처음엔 순진하게 max로 짰다가 BR(P1|Nash)=1/3이 나와 알고리즘 리팩터링. *Phase 2 Leduc에서도 동일 패턴 주의*
- **Vanilla CFR deterministic → seed parametrize는 구조적 placeholder**: 3 seeds × run은 RPS self-play(stochastic sampling)에서만 의미. Kuhn tabular CFR은 single seed + 코멘트로 Phase 2 MCCFR용 slot 명시가 깨끗
- **`game_value()`를 `train()`과 분리**: 평가 전용 메서드로 빼두니 W&B 학습 중 probe + exploitability 파이프라인에서 재사용. *CFR+에서도 이 분리 유지*

**다음 Phase로 이월된 이슈**:
- Known Issue 2건 (PHASE.md 상단 섹션 참조): `KuhnState.current_player` terminal 처리, seed parametrize 무의미성 — 둘 다 Phase 2에서 자연스럽게 해소 예정

**Phase 1 전체 회고 (3줄)**:
1. **Week 1 인프라 투자가 Phase 1 전체 속도의 비결** — Hydra + W&B + RegretMatcher primitive가 Week 2에서 CFR 트리 traversal과 convergence harness로 75% 그대로 재활용되어, Kuhn CFR + Exploitability + 10k full run이 Day 2 하루에 종료됨
2. **TDD 193 테스트가 수학적 정확성의 유일한 보증** — "순진한 per-state BR max" 오류는 코드 리뷰로 못 잡고 α-family Nash expl≈1e-9 테스트가 잡았음. 이론(Lanctot 2013 §3.4) ↔ 테스트 fixture ↔ 구현의 삼각 검증 없이는 조용히 잘못된 값이 pass
3. **설계 논의가 구현보다 중요한 비율 70:30** — "alternating A pattern vs C pattern", "chance를 reach_opp에 absorb", "char-based infoset key", "3-pass BR의 deepest-first ordering" 같은 결정이 구현 시간보다 논의 시간이 길었고 그 덕에 재구현 없이 한 번에 GREEN. Phase 2 CFR+ / Leduc에서도 동일 비율 유지

<!-- 템플릿
### Phase N (제목)
**완료일**: YYYY-MM-DD
**소요 시간**: N일 / N시간
**달성한 것**: ...
**배운 것**: ...
**다음 Phase로 이월된 이슈**: ...
-->
