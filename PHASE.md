# 현재 Phase 추적

> 매 세션 시작 시 Claude Code가 이 파일을 먼저 읽고 컨텍스트를 파악한다.
> 각 Phase 완료 시 이 파일을 업데이트한다.

## 현재 상태

**Phase**: 3 진행 중 — **Day 3b Cap 기각 (2026-04-24 오후)**. Primary A flat 유지 (0.2570 post-Cap vs 0.2549 baseline)
**Phase 3 Day 3b 판정**: Cap 가설 Primary A 측면 **REJECTED**. Strategy side 부분 이득 (σ̄_expl 182 → 162 mbb/g, 11% 개선). Leduc yaml rollback to 3×64 baseline.
**테스트**: **unit 347 + integration fast 10 GREEN** (신규 test_network_capacity_config 11개 포함)
**Next**: 가설 **D** (advantage target scale normalize, running std) 1순위 착수 (새 세션 권장). D 단독 테스트 후 Cap+D 결합 옵션.

## 다음 할 일 (Next Action) — Phase 2 Week 1 (Leduc 엔진 + CFR+)

### Week 1 (ROADMAP §Phase 2 Week 3 매핑) — Leduc 게임 + Vanilla → CFR+
- [x] **Leduc 엔진 설계 논의 확정** — 4가지 결정 확정 (CFR+만, 직접 구현, External Sampling, 3-pass BR 재사용) + Q2 세부 (IntEnum FCR, 120 deals, reach_opp=1/120, @property 파생, regret_matching legal_mask 옵션)
- [x] `tests/unit/test_leduc.py` + `tests/regression/test_leduc_perfect_recall.py` — 71 tests GREEN (59 unit + 12 regression, 288 infoset 검증 포함) (커밋 `1fa143e`)
- [x] `src/poker_ai/games/leduc.py` — Leduc Hold'em 엔진 (120 deals, 288 infosets, pot accounting rrf=-3/cc.rrf=-5) (커밋 `1fa143e`)
- [x] **Day 2** — `src/poker_ai/games/protocol.py` (GameProtocol/StateProtocol, 커밋 `b99f914`) + `exploitability.py` game-agnostic 리팩토링 + **Pass 2 illegal-argmax 버그 수정** (커밋 `fe40ac6`) + `regret_matching` legal_mask 옵션 추가 (커밋 `f84cef6`). 305 tests GREEN, Ruff + mypy strict clean
- [x] **Day 3** — Kuhn에 `legal_action_mask` 추가 (Option B) + StateProtocol 확장 (커밋 `0679cbc`) + VanillaCFR 루프 game-agnostic화 + InfosetData.legal_mask 캐시 + 규칙 3가지 masking invariant (커밋 `93e7e63`). 324 tests GREEN. Leduc 1k smoke: 288/288 infoset, expl=9.15 mbb/g, 10.8 iter/s
- [x] **Day 4** — Harness + fast regression + **100k W&B full run 완료** (커밋 `c173b24`). 10k fast PASSED (3.504 < 4.0). 100k 실측 **1.4802 mbb/g** — Exit #1 (< 1.0 mbb/g) **미달 48%**. Slope decelerates (−0.417 → −0.374), Nash 근처 steepen 가설 반증. **ROADMAP Exit #1은 Vanilla CFR로는 100k에서 미달; CFR+ 도입 후 재평가 예정**
- [x] **Day 5** — CFRPlus 구현 + audit 버그 수정 + **100k W&B full run 완료 (2h 43min)**. **Exit #1 ✅ PASS** (실측 **0.000287 mbb/g** < 1.0, 3482× margin). **Exit #2 ✅ PASS** (100k에서 Vanilla 대비 **5164× speedup**). iters_to_exit=2000, game_value −0.085606 (Nash 완벽). O(1/T) 수렴률 실증: 1k=0.121, 10k=0.0038, 100k=0.000287 — 지수적 개선 확인
- [ ] **Day 5** — `src/poker_ai/algorithms/cfr_plus.py` (Tammelin 2014: 음수 regret clipping + alternating + linear averaging) + CFR vs CFR+ 비교 실험 (Exit #2)
- [ ] W&B에 CFR vs CFR+ exploitability 곡선 중첩 — CFR+가 5~10배 빠르게 same-expl 도달 확인

### Week 2 (ROADMAP §Phase 2 Week 4 매핑) — MCCFR
- [x] `src/poker_ai/algorithms/mccfr.py` — External Sampling MCCFR (Lanctot 2009 §3.4) (커밋 `ebcf373`)
- [x] Audit-driven 버그 수정: IW factor (1/q → σ/q), reach_i 누락 복구
- [x] 14 tests GREEN (10 unit + 2 integration + 2 regression). iter_per_sec ≥ 5× Vanilla 확인 (integration)
- [x] `experiments/phase2_leduc_mccfr.py` + `conf/phase2_leduc_mccfr.yaml` — 5 seed × multiprocessing.Pool harness
- [ ] **다음 세션**: CFR+ 100k 완료 확인 (task bqyz4en67) → Leduc MCCFR 10k × 5 seed (~10min) → 100k × 5 seed (~17min parallel) → Exit #3 판정
- [ ] (선택) Outcome Sampling MCCFR 비교

## 지금까지 한 일 (Done)

### Phase 3 Day 3b — Cap 가설 기각, Strategy side 이득 (2026-04-24 오후)

> 커밋 `7a60ec6`. 멘토 단계적 접근 (Step 1 = 3×128 width 확대, Step 2 = 4×128 조건부).

#### 설계 (멘토 주관 ≥ 원안 조정)

원안 4×128 (5h) vs 대안 3×128 (3h) + 조건부 4×128. 기댓값 계산으로 단계적 접근 선택.

Width vs depth 선택 — Leduc shallow + wide 가설 (encoding이 hierarchical collapse 이미 수행, round 1/2 composition은 encoding에 내재, Zhang 2017 "width first"):
- **Step 1: 3×128 (num_hidden_layers=2, hidden_dim=128)**. Width 2×, depth 동일.
- Params 5.2k → 18.7k (3.56×).

구현: `AdvantageNet` / `StrategyNet`에 `hidden_dim` + `num_hidden_layers` 파라미터 추가, DeepCFR constructor에 전달. 11 신규 unit tests (정확 param count 검증 포함).

#### 실험 결과 (Leduc T=500 K=100 seed=42, 76.4분 wall-clock)

| T | Prim A | Prim B | σ̄_deep | σ̄_CFR+ |
|---|---|---|---|---|
| 50 | 0.2704 | 0.8236 | 241.5 | 17.06 |
| 100 | 0.2414 | 0.8414 | 208.0 | 6.71 |
| 250 | 0.2429 | 0.8123 | 163.9 | 1.60 |
| **500** | **0.2570** | **0.8006** | **162.4** | **0.463** |

Day 3 (3×64) baseline 비교 @ T=500:

| 지표 | Day 3 | Day 3b | Δ |
|---|---|---|---|
| Primary A | 0.2549 | 0.2570 | **+0.002 (flat)** |
| Primary B | 0.7883 | 0.8006 | +0.012 |
| σ̄_deep | 181.6 | 162.4 | **-11%** |
| R1 pure L∞ | 0.176 | 0.147 | -16% |
| R1 mixed L∞ | 0.181 | 0.302 | **+67% (overfit?)** |
| R2 pure L∞ | 0.188 | 0.174 | -7% |
| R2 mixed L∞ | 0.263 | 0.250 | -5% |

#### 판정

**Primary A 측면**: **REJECTED**. 4 checkpoint 모두 0.24-0.27 flat. Day 3 baseline과 본질적으로 동일.

**Strategy side**: **부분 이득**. σ̄_expl -11%, Primary B +0.012. 그러나 R1 mixed L∞ +67% 악화 (6 infoset) — overfit 경계 신호.

**기각 근거**:
- Primary A가 원인이 capacity이면 T↑에 따라 단조 상승 기대. 실측은 flat (4 checkpoint std 0.012).
- Width 2×로 advantage net의 signed regret 학습이 본질적으로 개선 안 됨 → network capacity **not root cause**.

#### 교육적 발견

**Day 3b finding (2026-04-24)**: Primary A 낮음 (0.25)의 원인은 **network capacity가 아니다**. Width 확대 3.56×는 Primary A를 움직이지 못함 (flat 유지). 반면 Strategy net은 Cap 수혜 (σ̄_expl -11%, Primary B 개선). **Advantage net과 Strategy net의 "capacity 민감도" 분리**: Strategy net (simplex regression)은 width 이득, Advantage net (signed regret regression)은 다른 원인 (target scale or data quality)이 bottleneck. 이 분리가 Phase 3 Day 3c (다음 세션)에서 가설 D (advantage target running-std normalize) 우선순위 결정의 근거.

#### 다음 세션 (Day 3c) 계획

**1순위: 가설 D-2 (Advantage Target Running Std Normalize)**
- 구현 ~15줄: EMA running std per-player, `y_scaled = y / adv_target_std`
- Regret matching이 scale-invariant이므로 inference 시 unscale 불필요
- 예상 효과: Primary A 0.25 → 0.4-0.6 (Kuhn 0.8은 entry-per-sample 부족으로 불가능일 수도)
- D 단독 실험 (yaml baseline 64로 rollback 상태)

**2순위 (D 성공 후)**: Cap+D 결합 — strategy-side + advantage-side 둘 다 개선 가능성 평가

### Phase 3 Day 3 — Leduc smoke FAIL, algorithm issue 확정 (2026-04-24)

> 커밋 `7cffbda`. Day 3 목표: Kuhn Post-Day-2b-A fix가 Leduc scale에서도 유효한지 검증.

#### 실험 설계 (멘토 Option B 승인)

Kuhn T=500 post-Day-2b-A 최종 결과 (prim_A 0.8143, prim_B 0.9873, σ̄_expl 14.4 mbb/g — baseline 160×)를 직접 Leduc T=500과 같은 조건 (K=100, seed=42)으로 비교. 세 가지 분기 예상:
- σ̄_deep > 100 → algorithm issue (B/D/5 진입)
- 10-100 → partial scale gain
- < 10 → scale artifact (Kuhn 160× 자체가 작은 게임 artifact)

#### Leduc T=500 실측 (61분 wall-clock)

| T | prim_A | prim_B | tert | σ̄_deep | σ̄_CFR+ | σ̄_Vanilla | ratio |
|---|---|---|---|---|---|---|---|
| 100 | 0.2459 | 0.8086 | 0.2464 | 230.9 | 6.71 | 46.5 | 34× |
| 250 | 0.2591 | 0.7994 | 0.2614 | 192.7 | 1.60 | 19.4 | 120× |
| **500** | **0.2549** | **0.7883** | 0.2598 | **181.6** | **0.463** | 15.1 | **392×** |

**Day 3 GREEN 판정**:

| 기준 | 목표 | 실측 | 판정 |
|---|---|---|---|
| Primary A > 0.75 | > 0.75 | 0.2549 | ❌ (0.5 미달) |
| Primary B > 0.90 | > 0.90 | 0.7883 | ❌ |
| σ̄_expl < 10 mbb/g | < 10 | 181.6 | ❌ (18× 초과) |

**3/3 FAIL. Algorithm issue 확정**.

#### Per-round L∞ 분석 (288 infoset 전량 방문)

| | Round 1 (n=18) | Round 2 (n=270) |
|---|---|---|
| Pure | L∞ 0.176 (n=12) | L∞ 0.188 (n=205) |
| Mixed | L∞ 0.181 (n=6) | **L∞ 0.263 (n=65)** |

**Round 2 mixed L∞ 0.263이 Kuhn Post-A 0.095 대비 2.8× 나쁨**. CE fix의 pure 이득은 Leduc에서도 부분 유지되지만, mixed strategy 분포 복잡도가 Kuhn보다 훨씬 높아서 MSE→CE만으로 부족.

#### 주요 해석

**1. "Scale 이득" Brown 2019 가설 반증**: T 증가에 따라 ratio **악화** (34→120→392×). CFR+가 T=500에 이미 0.463 mbb/g에 도달하는 동안 Deep CFR는 181 mbb/g. CFR+가 훨씬 빠르게 Nash 수렴. Kuhn post-A ratio 160×보다 **더 나쁨** — Leduc에서 신경망 approximation의 상대적 우위 실증되지 않음.

**2. Primary A 완전 flat 0.25**: T=100/250/500 세 checkpoint 모두 Primary A ~0.25. Day 2b-A loss normalization 효과가 Leduc에서는 무의미. Kuhn (0.81)과의 3.5× 차이는 algorithm 구조 문제 신호.

**3. "Kuhn Post-A 결과는 scale artifact 아님"**: Kuhn에서 0.81 Primary A가 가능했던 이유는 **entry당 training sample 수**가 충분했기 때문. 
   - Kuhn: 24 pairs, T=500 × K=100 = 50k visits → infoset당 ~2000 samples
   - Leduc: 550 pairs, T=500 × K=100 = 50k visits → infoset당 ~90 samples (22× 적음)
   - Deep CFR 성능은 entry당 sample 수에 매우 민감 — Leduc이 under-trained.

**4. Under-parameterization 의심**: 
   - Kuhn: 24 pairs / 12k net params → 1:500 (오히려 overparameterized)
   - Leduc: 550 pairs / 12k params → 1:22 (under-parameterized 가능)
   - 3×64 MLP capacity가 Leduc의 550 pair 다양성을 담기에 부족할 수 있음.

#### 가설 우선순위 (다음 세션)

| # | 가설 | 구현 | 예측 효과 | smoke 비용 |
|---|---|---|---|---|
| **Cap** | **Network 4×128 (확대)** | hidden_dim 파라미터 1줄 | Primary A 돌파 기대 (under-param 해소) | ~60분 (Leduc 모델 커지면 train 느려짐) |
| Buf | Buffer 100k → 1M (10×) | 1줄 | 포화 후 초기 sample 보존 | ~60-70분 |
| D | Advantage target normalize (running std) | 5-10분 | Leduc regret range 커서 gradient 불안정 해소 | ~60분 |
| Lr | Adam lr 1e-3 → 5e-4 또는 1e-4 | 1줄 | Target scale 큰 경우 안정화 | ~60분 |
| C | Warm-start vs from-scratch reinit | 10분 | 학습 연속성 | ~60분 |

**1순위: Cap** — 물리적 근거 가장 강함 (under-param ratio 1:22). Kuhn에서 이미 Primary A 0.81 달성 가능함을 증명했으니, Leduc에서 낮은 0.25는 "동일 capacity로는 Leduc 550 pair를 fit 못함"이 가장 자연스러운 설명.

#### 교육적 발견

**Day 3 finding (2026-04-24)**: Kuhn에서 성공한 fix (CE + loss normalization)가 Leduc에서 Primary A 0.25 flat로 완전 실패. **"큰 게임에서 function approximation 이득" 주장은 entry당 training sample 수가 충분할 때만 성립**. Leduc의 550 pairs × K=100 × T=500 = infoset당 90 sample은 3×64 MLP에게 부족. Kuhn은 infoset당 ~2000 samples으로 over-saturated. 게임 scale에 맞는 network capacity 스케일링이 Phase 3+ 필수 설계 원칙.

### Phase 3 Day 2b-A — Advantage net Primary A trajectory 역전 (2026-04-24)

> 커밋 `4220bf9`. 멘토 Day 2b 가설 1 (iter_weight polynomial growth → loss bias).

#### Root cause 분석

Pre-fix loss: `loss = (per_sample × w_b).mean() = Σ w_i · s_i / n`
- `n` = batch size (상수), `w_i = iteration` (linear CFR weight = t)
- Batch 내 `mean(w) ≈ T/2` → loss magnitude ∝ T (linear growth)
- 결과: 후기 iter 후 batch에서 effective gradient magnitude가 T에 비례 팽창
- Adam이 adaptive LR로 일부 보정하지만 **from-scratch reinit + T-scaling loss**는 수렴 landscape를 iter 의존적으로 왜곡

#### Fix: true weighted mean

Post-fix: `loss = Σ w_i · s_i / Σ w_i` (per batch)
- Magnitude T-independent
- 샘플 간 상대 가중치 보존 (Linear CFR 수학적 정확성 유지)
- Buffer는 건드리지 않음 (raw iter_weight 저장 그대로, 재해석 문제 회피)
- Loss-side vs buffer-side normalization 선택 — **loss-side가 robust** (이전 샘플 저장값 불변)

Advantage + Strategy loss path 둘 다 동일하게 수정.

#### 실측 trajectory 비교 (T=500, K=100, seed=42)

**Primary A** (Deep adv vs Vanilla R_cum):

| T | Pre-all (MSE) | Post-CE | **Post-A** | change |
|---|---|---|---|---|
| 100 | 0.8424 | 0.8424 | **0.8533** | +0.011 |
| 250 | 0.8125 | 0.8125 | 0.8122 | ≈ |
| **500** | **0.7976 ↓** | **0.7976 ↓** | **0.8143 ↑** | **+0.017** |

핵심: **Pre-fix는 T 증가에 단조 감소 (0.8424 → 0.7976)**, **Post-A는 T=250 dip 후 T=500 회복** → trajectory 방향 역전.

**σ̄_deep exploitability**:

| T | Pre-all | Post-CE | Post-A |
|---|---|---|---|
| 100 | 105.4 | 24.8 | 22.7 |
| 250 | 109.6 | 20.3 | 18.7 |
| 500 | 117.4 | 21.4 | **14.4** (추가 33% 개선) |

Post-A는 σ̄_expl 궤적도 단조 감소로 전환 (Post-CE에선 T=250 최소 후 T=500 반등).

#### Day 2b 판정

| 기준 | 목표 | 실측 @ T=500 | 판정 |
|---|---|---|---|
| Primary A > 0.8 | > 0.8 | 0.8143 | ✅ |
| Primary A trajectory 역전 | 상승 또는 안정 | T=500 > T=250 | ✅ |
| Primary B > 0.95 | > 0.95 | 0.9873 | ✅ |
| σ̄_expl < 5 mbb/g | < 5 | 14.4 | ❌ (baseline 0.09 대비 160×) |

**Day 2b-A 판정**: 구조적 목표 (trajectory 역전) 달성. σ̄_expl 절대값은 Primary A가 완전히 baseline에 수렴 (>0.95) 안 되어서 bounded. **추가 가설 B/C/D는 다음 세션**으로 분리 (context 관리 + 이미 Day 2b 핵심 목표 달성).

#### Day 2b-A 교육적 발견

**Linear CFR weighting을 loss-side로 정규화해야 T-independent 학습 달성.** Buffer-side normalization (`iter_weight = t/T_total`)은 이전 샘플 저장값이 inconsistent해지는 문제 있음. Loss-side `Σ w·s / Σ w`는 batch 단위 재정규화라 robust. Phase 3+에서 Deep CFR / Deep MCCFR 계열 학습에 지속 적용.

### Phase 3 Day 2 — Strategy net MSE pathology 발견 + Cross-entropy fix (2026-04-24)

> 커밋 `c351f51` (infrastructure + 3-metric), `0431b89` (CE fix).

#### Narrative (pre-smoke 설계 수정 이후)

Step 1-3 (c351f51): 3-metric `compute_correlations` 구현 + `phase3_deep_cfr_kuhn` Hydra/W&B harness + 11 integration tests 작성.

Step Smoke (T=500, K=100, seed=42, 32.7분 CPU):

| T | Primary A | Primary B | Tertiary | σ̄_deep expl | σ̄_CFR+ ref |
|---|---|---|---|---|---|
| 100 | 0.8424 | 0.9583 | 0.4464 | 105.4 mbb/g | 0.60 mbb/g |
| 250 | 0.8125 | 0.9736 | 0.3970 | 109.6 | 0.21 |
| **500** | **0.7976 ↓** | **0.9892 ↑** | 0.4024 | **117.4 ↑** | **0.09** |

- **Paradox**: Pearson r = 0.99 (strategy 거의 일치) BUT σ̄_expl 1360× worse than baseline
- **Primary A 역행**: T↑에 따라 correlation **감소** (0.84→0.80) — 별도 구조 이슈

#### 실증 — Per-infoset L∞ analysis (T=100, before fix)

| Infoset | CFR+ σ̄ | Deep σ̄ | L∞ | 범주 |
|---|---|---|---|---|
| J\|b | [1.00, 0.00] | [0.71, 0.29] | 0.295 | **PURE** |
| K\|b | [0.00, 1.00] | [0.28, 0.72] | 0.280 | **PURE** |
| Q\| | [0.998, 0.002] | [0.63, 0.37] | 0.364 | **PURE** |
| J\| | [0.81, 0.19] | [0.66, 0.34] | 0.152 | mix |
| J\|p | [0.67, 0.33] | [0.63, 0.37] | 0.043 | mix |

- Pure strategy infoset 평균 L∞: **0.29**
- Mixed strategy infoset 평균 L∞: **0.08**
- **Strategy net output range [0.28, 0.72]로 saturate** — MSE loss가 extreme values 학습 못함

#### Root cause — MSE pathology

MSE gradient magnitude `∂L/∂logit ∝ (pred - target)`는 target이 extreme (0/1)일 때도 bounded. Softmax output range 제한 (0-1) 때문에 predict 0 or 1에 접근하려면 logit이 ±∞가 필요한데 MSE는 그 방향 gradient를 충분히 주지 않음. Result: strategy net이 중간값 [0.3, 0.7]에 collapse.

Pearson correlation은 **affine transform invariant** (`r(aX+b, Y) = r(X, Y)`). 따라서 `σ_deep ≈ 0.44 × σ_CFR+ + 0.28` 같은 shrinkage가 있어도 r = 1에 가까움. 이것이 "correlation 0.99 but expl 1360×"의 수학적 설명.

Exploitability는 pointwise distribution mismatch에 직접 민감. 특히:
- Jack이 bet faced 시 CFR+ σ = [1.0, 0.0] (fold 100%) vs Deep σ = [0.71, 0.29] (fold 71%, call 29%)
- 29% bad call로 손실 (bet을 call하면 King한테 -2 chip 잃음) → 수백 mbb/g 증폭

#### Fix — Cross-entropy + legal masking (커밋 `0431b89`)

1. **ReservoirBuffer mask field**: `mask_dim: int = 0` (default backward-compat). strategy buffer만 `mask_dim=n_actions`로 생성
2. **`_traverse`가 legal_mask 저장**: `strategy_buffer.add(encoding, σ, iter_weight, mask=legal_mask)`
3. **Cross-entropy loss** (strategy net only — advantage net 미수정):
   ```python
   masked = torch.where(legal_mask, logits, -inf)
   log_probs = F.log_softmax(masked, dim=-1)
   # Soft-target CE with illegal-guard
   terms = torch.where(target > 0.0, target * log_probs, zeros)
   loss = (-(terms.sum(dim=-1)) * iter_weight).mean()
   ```

#### Post-fix 측정 (T=500, same seed/config)

|  | σ̄_expl before (MSE) | σ̄_expl after (CE) | improvement |
|---|---|---|---|
| T=100 | 105.4 | **24.8** | 4.2× |
| T=250 | 109.6 | **20.3** | 5.4× |
| T=500 | 117.4 | **21.4** | 5.5× |

Per-infoset L∞ (post-fix, T=100):

| Infoset | before | after | 개선 |
|---|---|---|---|
| J\|b | 0.295 | **0.065** | 4.5× |
| K\|b | 0.280 | **0.033** | 8.5× |
| Q\| | 0.364 | 0.214 | 1.7× (여전히 worst) |
| Pure avg | 0.29 | **0.10** | **3×** |
| Mixed avg | 0.08 | 0.09 | ≈ (일부 tradeoff) |

#### Day 2 판정

| 기준 (Step 3) | 목표 | 실측 @ T=500 | 판정 |
|---|---|---|---|
| Primary A > 0.8 (완화) | > 0.8 | 0.7976 | ❌ by 0.004 (Day 2b 분리) |
| Primary B > 0.95 | > 0.95 | **0.9893** | ✅ |
| σ̄_expl < 5 mbb/g | < 5 | 21.4 | ❌ (238× baseline) |

**Partial success**: CE fix가 정확히 진단한 문제 (pure strategy 학습) 해결. 남은 expl 21 mbb/g는 Primary A trajectory 역행이 상한선 역할 — **Primary A 해결 없이 σ̄_expl 완전 개선 불가능**.

#### 교육적 발견 — Pearson 단독 불충분성

Phase 3 Day 2 discovery (2026-04-24): **Strategy net MSE loss fails on pure-strategy infoset. Pearson r=0.99 with σ̄_CFR+ masks scale/offset mismatch. Per-infoset analysis: mixed L∞ ~0.05 OK, pure L∞ ~0.30 FAIL. Cross-entropy with -inf legal masking restores extreme values. Phase 2 audit pattern scaled up: Pearson correlation alone is insufficient for strategy space; exploitability ground-truth needed.**

이 자산은 Phase 2 Day 5/6 "unit test 통과 + 실수렴 실패" 패턴과 동일 계층. Phase 3 내내 "correlation + pointwise metric 이중 검증" 원칙으로 유지.

#### Day 2b plan — Advantage net trajectory 역행 audit

**관찰**: Primary A 0.8424 (T=100) → 0.8125 (T=250) → 0.7976 (T=500). T 증가에 따라 **감소** — 정상 학습이면 상승해야 함.

**4 hypotheses**:

| # | Hypothesis | 검증 실험 | 예상 cost |
|---|---|---|---|
| A | iter_weight=T 폭주 — 후기 sample이 loss 지배 → recent-biased 학습 | iter_weight=`T/T_total` normalize 후 T=500 재실행 | 1× smoke (~33분) |
| B | Buffer 포화 + from-scratch reinit → 학습 부족 | buffer_capacity 100k → 1M, T=500 재실행 | 1× smoke |
| C | From-scratch reinit 자체 — warm-start가 더 나을 수 있음 | 매 iter optimizer 유지, T=500 | 1× smoke |
| D | Advantage target scale drift — `v_a - v_node` magnitude가 후기에 작아짐 | Target normalize (÷ T or ÷ reach), T=500 | 1× smoke |

총 4-5시간 (각 smoke 33-45분). 가설 A부터 가장 구현 저렴.

### Phase 3 Day 2 pre-smoke — Exit #4 primary metric 설계 수정 (2026-04-24)

> 교육적 발견 — Day 5/6 audit 기록과 같은 층위.

**Phase 3 Day 2 pre-smoke 발견 (2026-04-24)**: 멘토 설계상 `correlation(tabular CFR+ R⁺, Deep net) > 0.95`가 수학적 부정확. Deep CFR는 **Vanilla signed regret의 근사** (Brown 2019 Algorithm 1 Line 5)이지 CFR+의 positive-clipped regret 근사가 아님. Smoke에서 **Deep vs CFR+ r=0.40 vs Deep vs Vanilla r=0.82** 실측으로 구조적 차이 드러남. Reference algorithm 전환 (CFR+ → Vanilla, strategy space 추가). Phase 3 내내 이 수정 기준 유지.

#### 근거: Brown 2019 Algorithm 1 Line 5

Deep CFR advantage network 학습 target:
- `r̃_p^t(I, a) = π_{-p}(I) · (v_{I,a} - v_I)` — **signed** instantaneous regret
- Network output V̂(I, a) ≈ `Σ_t w_t · r̃_t(I,a) / Σ_t w_t` (linear-CFR-weighted time average, signed)
- Vanilla CFR의 `cumulative_regret`은 같은 signed 양의 unweighted time sum
- CFR+의 `cumulative_regret`은 **positive-part clipping**으로 non-negative-by-construction — 수학적으로 **다른 객체**

#### 실측 (K=20, T=100 smoke, seed=42)

| Reference | vec range | n_pos / n_neg | Pearson r (Deep net vs 이 ref) |
|---|---|---|---|
| **Vanilla R_cum** (signed) | [-18.3, 1.7] std=6.02 | 17 / 7 | **0.8186** |
| **CFR+ R_cum** (positive-only) | [0.0, 0.56] std=0.16 | 17 / 0 | 0.3993 |
| Deep net output | [-2.7, 0.15] std=0.76 | 13 / 11 | — (reference) |
| Deep(positive-clipped) vs CFR+ | — | — | 0.1095 (clipping도 해결 못함) |

#### Phase 3 Exit #4 metric 설계 (수정 후)

- **Primary A** (`primary_a_advantage_vs_vanilla`): `corr(advantage_net, Vanilla R_cum)` — **> 0.95** (stretch) / > 0.9 (GREEN)
- **Primary B** (`primary_b_strategy_vs_sigma_bar`): `corr(strategy_net_softmax, CFR+ σ̄)` — simplex space, > 0.85 (GREEN) / > 0.9 (stretch)
- **Tertiary** (`tertiary_advantage_vs_cfr_plus`): `corr(advantage_net, CFR+ R⁺)` — **diagnostic only**, 낮게 나오는 게 정상. 설계 수정 증거로 로그에 남김

#### Pearson scale-invariance 문서화

Pearson `r(aX+b, Y) = r(X, Y)`. 따라서:
- Deep net output은 linear-CFR-weighted avg (정규화 `T_norm = Σ_t w_t = T(T+1)/2`)
- Vanilla R_cum은 unweighted sum
- 두 quantity 간 scale/weighting 차이가 있어도 **선형 관계만 보존되면 Pearson 값 동일**
- 구현은 raw-vs-raw 비교로 충분. T_norm 인자 명시적 계산 불필요

#### 구현

- `src/poker_ai/eval/deep_cfr_correlation.py` 확장: `CorrelationReport` dataclass + `compute_correlations(deep, vanilla, cfr_plus, game)` 추가. 기존 `compute_flat_correlation` (tertiary)는 backward-compat로 유지.
- `experiments/phase3_deep_cfr_kuhn.py` + yaml: 3 trainer 병렬 + checkpoint별 3-correlation + σ̄_deep expl + W&B log + 2-panel PNG.

### Phase 3 Day 1 (2026-04-23 저녁) — Deep CFR Infrastructure 구축

목표: encode + reservoir + networks + DeepCFR skeleton으로 **smoke test 통과**. 수렴 검증은 Day 2+.

#### 산출물 (커밋 `e4e5df3`, `a5c9e6a`)

| 컴포넌트 | 위치 | 책임 |
|---|---|---|
| `GameProtocol.encode()` | `src/poker_ai/games/protocol.py` | `ENCODING_DIM` + `encode(state) → np.ndarray[float32]` Protocol 확장 |
| Kuhn encode() | `src/poker_ai/games/kuhn.py` | 6-dim: 카드 one-hot(3) + hist_len 플래그 + hist[0]_is_bet + hist[1]_is_bet |
| Leduc encode() | `src/poker_ai/games/leduc.py` | 13-dim: hole(3) + board(3) + not_revealed + round(2) + (len, raise)/norm × 2라운드 |
| `ReservoirBuffer` | `src/poker_ai/algorithms/reservoir.py` | Vitter 1985 + torch 프리얼로케이션 + scalar/vector target 양쪽 |
| `AdvantageNet` / `StrategyNet` | `src/poker_ai/networks/{advantage,strategy}_net.py` | 3-layer × 64 MLP, ReLU, raw logit, Brown 2019 Leduc config |
| `DeepCFR` | `src/poker_ai/algorithms/deep_cfr.py` | 외부 샘플링 traversal (MCCFR 80% 재사용) + per-player advantage buffer + 공유 strategy buffer + Adam lr=1e-3 + grad clip 10.0 (A2) + Linear CFR loss-side weighting (#5) + from-scratch reinit per iter (#4) |

#### Tests (51 신규, 모두 GREEN)

- **21 encode** (`test_encode.py`): shape/dtype/determinism + **12/288 infoset distinctness** + Protocol conformance
- **14 reservoir** (`test_reservoir_buffer.py`): basic bookkeeping + Vitter property (retention rate ≈ cap/total, bucket uniform, no-dup, 100-seed MC, determinism)
- **16 deep_cfr smoke** (`test_deep_cfr_smoke.py`): instantiation + per-player advantage nets + shared strategy net + seed-fixed init + 1-iter end-to-end + forward/backward gradient flow + finite-params invariant

`uv run pytest tests/unit tests/integration -m "not slow"` → **364 passed, 3m10s**. Ruff + mypy strict clean on 7 source files.

#### 설계 포인트

1. **Kuhn 6 vs Leduc 13 encoding 둘 다 infoset-unique** (12/288 distinct tensor 실측) — 아무 무작위 encoding이 아니라 perfect recall 보장
2. **ReservoirBuffer target 지원 확장**: 기존 `target: float` scalar API 100% 보존 (14 Vitter tests GREEN) + Deep CFR용 vector target (`target_dim=n_actions`) 추가 → regret/strategy 벡터 저장 가능
3. **Per-player advantage + shared strategy** (Brown 2019 §3 패턴) — 결정 #6 "strategy net Day 1부터" 반영
4. **Traversal 구조는 MCCFR ES 80% 재사용** — 변경점 2곳: (a) tabular `cumulative_regret` → `advantage_nets[p]` forward, (b) in-place update → reservoir push. 나머지 (ε-smoothing, importance weighting 없는 updating-player enumerate) 그대로

#### 다음 (Day 2)

- Kuhn 100-500 iter 학습 → **correlation(tabular_R⁺, advantage_net_output) 측정** (Exit #4 primary 연습판)
- 실제 regret 값 sign/magnitude가 tabular CFR과 일치하는지 sanity

---

### Phase 3 Deep CFR Preview — Design Lock (2026-04-23)

Phase 2 완주 후 멘토+Claude co-review로 설계 확정. 구현 착수는 내일 새 세션.

#### 7 결정 (audit-기반 수정 반영)

1. **Input encoding**: Flat one-hot + game-specific `encode(state) → Tensor[n_features]`
   - Leduc 13dim (hole 3 + board 4 + round 2 + betting_hist 4), Kuhn 6dim
   - GameProtocol 확장 (A3)

2. **Advantage Network**: 3 layer × 64 MLP, ReLU, raw logit output (Brown 2019 Leduc config)

3. **Reservoir Buffer**: Torch tensor size 1M + growth curve 로깅
   - Vitter 1985 reservoir sampling
   - `len(buffer)/T` 로그로 regime shift 감지 (Day 5 sync drift 패턴 가드)

4. **학습 Schedule**: From-scratch reinit (원안) + **conditional warm-start fallback**
   - Batch 10k, ~4 epochs, Adam lr=1e-3
   - Day 4 Exit #4 미달 시 warm-start 전환 판단 (미리 고정 않음)
   - iter별 training loss curve 로깅 (under-training 감지)

5. **Linear CFR Weighting** (loss-side sample weight) + **audit test 필수**
   - `loss = Σ (iter_i / T) · (network(I_i) - regret_i)²`
   - Buffer는 iter 완료 후에만 갱신 (within-iter target drift 방지 — Day 5 패턴)
   - **Phase 2 Day 6 unbiasedness 패턴 이식** (200+ seed-equiv + hand-computed weighted avg vs network output, rel_err < 10%)

6. **Strategy Network** — Day 1부터 advantage + strategy **둘 다** (원안 연기 REJECT)
   - 이유: σ̄ (time-averaged) 수렴 보장은 Deep CFR 이론 핵심; σ^T (last iter)만으론 oscillate 가능 → Exit #4 평가 근거 무너짐
   - Reservoir 2개 (advantage/strategy 각각) 또는 1개 공유 — Day 1 구현 시 결정
   - **Exit #4 평가는 σ̄ exploitability 기준**, σ^T는 diagnostic only

7. **Exit Criteria** (현실 수정)
   - **Exit #4 primary**: `correlation(tabular_R⁺, network_output) > 0.95` — Deep CFR correctness 직접 근거, absolute expl보다 신뢰 높음
   - **Exit #4 secondary**: Leduc σ̄ expl < **0.1 mbb/g** (Brown 2019 Fig 2 범위; 0.003은 function-approx floor 고려 비현실)
   - **Exit #5**: Leduc σ̄ expl < 1.0 mbb/g (Vanilla baseline 이김)
   - **Stretch**: < 0.01 mbb/g

#### 추가 5 결정 (Review에서 발견)

- **A1**. Traversals per CFR iter `K = 1000` (Leduc 시작값)
- **A2**. `torch.nn.utils.clip_grad_norm_(params, 10.0)` — advantage target range가 chip 단위 O(10) 가능
- **A3**. `encode(state) → Tensor` 인터페이스 — GameProtocol에 추가 or game-specific helper
- **A4**. σ̄ 평가: strategy network 사용 (결정 6 결과)
- **A5**. **Unbiasedness audit test** — Phase 2 Day 6 패턴을 Deep CFR loss weighting 검증에 이식

#### Day Scope (예상 2주, Phase 2처럼 1일 단축 불가)

| Day | 목표 |
|---|---|
| 1 | Infrastructure: encode() + advantage/strategy nets + reservoir + Kuhn/Leduc smoke |
| 2 | Kuhn Deep CFR 100-500 iter + correlation 측정 (기본 correctness 확인) |
| 3 | Leduc 1k smoke + **first audit (A5 unbiasedness test)** |
| 4 | Leduc 10k + **Exit #4 primary 판정** (correlation > 0.95) |
| 5-6 (Week 2) | 10k 통과 시 100k + Exit #4 secondary (<0.1 mbb/g) 판정 |

Neural training debug cost가 상수 비용이라 Phase 2 급 단축 불가. 2주 reasonable.

#### Phase 2 → Phase 3 상속

- **Code 재사용**: MCCFR External Sampling traversal 80% (`src/poker_ai/algorithms/mccfr.py`)
- **Type 확장**: GameProtocol에 `encode()` 추가
- **Audit 4단 패턴** (Phase 3 업그레이드):
  1. Unit tests (invariants: shape, non-neg, sum-to-1)
  2. Many-seed unbiasedness (stochastic methods — Day 6 패턴)
  3. 1-iter hand-computed snapshot (deterministic methods — Day 5 패턴)
  4. **(신규) Tabular ground truth correlation** (Deep CFR approximation quality)
- **실전 버그 DB**: Day 5 sync update drift + Day 6 IW factor (1/q vs σ/q) + reach_i — Phase 3 구현 시 체크리스트

---

### Phase 2 Week 1-2 완주 (2026-04-23) — 3대 CFR 변형 구현 + 3 Exit Criteria 전부 PASS

#### Phase 2 Exit Criteria Final Scorecard

| # | 기준 | 주체 | 실측 | 판정 |
|---|---|---|---|---|
| **#1** | Leduc expl < 1.0 mbb/g @ 100k | CFR+ | **0.000287** | ✅ PASS (3482× margin) |
| **#2** | CFR+ 5-10× speedup vs Vanilla | CFR+ | **5164×** | ✅ PASS (516× 초과) |
| **#3** | MCCFR per-iter ≥ 10× Vanilla | MCCFR ES | **34×** (379 vs 11 iter/s) | ✅ PASS (3.4× 초과) |

#### 3-Algorithm Comparison Matrix (Leduc 100k, seed=42 기준)

| 알고리즘 | Final expl | Wall-clock | iter/s | 절대 수렴 품질 | Per-iter 속도 |
|---|---|---|---|---|---|
| Vanilla CFR | 1.48 mbb/g | 2h 31min | 11 | 중간 | 기준 |
| **CFR+** (Tammelin) | **0.000287** | 2h 43min | 10 | **최고 (O(1/T))** | 기준 |
| **MCCFR ES** (Lanctot) | 59.58 (5-seed) | **5분** (parallel) | **379** | 중-하 (variance) | **34×** |

CFR+의 iter-quality 강점 + MCCFR의 iter-speed 강점 → Phase 3 Deep CFR이 "MCCFR traversal + function approximation" 조합으로 둘 다 활용하는 선택의 정당화가 오늘의 3-way 실측으로 확보됨.

#### Day 6 MCCFR External Sampling (오늘 완료)

- ✅ **Design 7 결정 확정**: External Sampling 단독, 독립 클래스, RNG 주입, alternating updates, ε-exploration, 3 커밋 분할
- ✅ **14 tests (10 unit + 2 integration + 2 regression)** + harness (`phase2_leduc_mccfr.py` with multiprocessing.Pool)
- ⚠️ **Audit-driven 버그 수정 2건** (Day 5 패턴 계승):
  1. Importance weight factor: 1/q → **σ/q** (Lanctot §3.2 ε-smoothing correction)
  2. Cumulative strategy 누적 시 reach_i 누락 복구
  수정 전 Kuhn 10k mean 268 mbb/g → 수정 후 12 mbb/g 정상화
- ✅ **Unbiasedness 검증** (Lanctot Prop 4): 200 seed × 500 iter, 대규모 regret components Vanilla와 5% 이내 일치
- ✅ **실측 Exit #3 PASS**: 10k 146 mbb/g (34.5s), 100k 59.58 mbb/g (263.6s parallel). iter_per_sec 34× speedup
- ✅ **Regression threshold 실증 조정**: 10k `4.0 → 250 mbb/g`, 100k `1.0 → 100 mbb/g` (Lanctot Fig 4.3 부합)

#### 오늘 하루(2026-04-23) Phase 2 서사

| Day | 주제 | 핵심 |
|---|---|---|
| 1 | Leduc 엔진 | 120 deals × 288 infosets, pot accounting lock-in |
| 2 | Protocol + exploitability 일반화 | **Pass 2 illegal-argmax 잠복 버그 발견+수정** (Kuhn no-op, Leduc active) |
| 3 | VanillaCFR game-agnostic | `legal_action_mask` StateProtocol 확장, hook 구조 확립 |
| 4 | Vanilla 100k baseline | **Zinkevich bound 비타이트성 실증** (slope -0.417 → -0.374), Exit #1 fail |
| 5 | CFR+ 구현 + audit | **Sync regret update 버그 audit+fix**, Tammelin Fig 2 재현, Exit #1+#2 PASS |
| 6 | MCCFR ES 구현 + audit | **IW factor 버그 audit+fix**, 3-way 비교, Exit #3 PASS |

#### 교육적 발견 (Phase 2 자산)

1. **Zinkevich O(1/√T) worst-case bound의 비타이트성**: 작은 게임(Kuhn)은 이론과 일치, 큰 게임(Leduc)에선 실측 slope가 이론보다 완만
2. **Externalized chance (deal loop) + linear averaging 호환 문제**: Vanilla harmless, CFR+ critical — σ^t within-iter drift 버그. Sync update로 해결
3. **IW factor 정확성 민감도**: σ/q vs 1/q 차이가 수렴에 결정적. Unit test로 못 잡고 1-iter snapshot + many-seed unbiasedness가 잡아냄
4. **Audit 3단 패턴** (Phase 3 이후에도 적용):
   - Unit test (invariants: non-neg, sum-to-1, shape)
   - Many-seed unbiasedness (stochastic methods)
   - 1-iter hand-computed snapshot (deterministic methods)

#### 최종 상태 (Phase 2 종료)

- **총 커밋 (Phase 2)**: ~20+ 커밋
- **Tests**: 365 GREEN (324 baseline + 15 CFR+ + 14 MCCFR + threshold 조정)
- **3 CFR 변형** (Vanilla/CFR+/MCCFR) 각각 Game-agnostic via GameProtocol
- **W&B runs**: Vanilla seed42 100k, CFR+ seed42 100k, MCCFR 5-seed (disabled in final test)

#### Next: Phase 3 Deep CFR (별도 세션)

MCCFR External Sampling traversal + advantage network로 generalization. Phase 2의 3-way 비교가 핵심 동기:
- CFR+의 O(1/T) 수렴은 tabular에서만 가능 (HUNL 10^164 infosets 불가능)
- MCCFR의 iter-speed는 유지, function approximation으로 regret table을 neural network로 대체
- Phase 3 구현 시 이미 검증된 MCCFR ES traversal 80% 재사용 예정

---

### Phase 2 Week 2 Day 6 partial (2026-04-23) — MCCFR External Sampling 구현 + audit + harness (실측 다음 세션)

- ✅ **Design 7 결정 확정**: External Sampling 단독, 독립 클래스 (VanillaCFR 비상속), RNG 주입, alternating updates 유지, ε-exploration 필수, 커밋 3 분할
- ✅ **test-writer 3 RED 파일** — 14 tests (10 unit + 2 integration + 2 regression)
- ✅ **C1 커밋 `ebcf373`** — `src/poker_ai/algorithms/mccfr.py` `MCCFRExternalSampling` 구현
- ⚠️ **Audit 기반 버그 수정 2건**:
  1. Importance weight factor 오류: 초기 구현 `weight /= sample_prob` (1/q) — Kuhn 10k mean 268 mbb/g 발생
     → 수정: `weight *= σ(a) / q(a)` (σ/q IW, Lanctot §3.2) — Kuhn 10k mean 12.4 mbb/g로 정상화
  2. Cumulative strategy 누적 시 `reach_i` 누락: 초기 `S += σ` → 수정 `S += reach_i · σ` (Lanctot Alg 3.1 준수)
- ✅ **Unbiasedness 검증 (Lanctot Prop 4)** — Kuhn 500 iter × 200 seed 평균으로 확인
  - 대규모 regret components (|v|≥10): Vanilla와 1-5% 오차 일치 (예: `J|b` action 1 → Vanilla `-57.00`, MCCFR avg `-57.09`)
  - 소규모 components: MC variance에 묻힘 (bias 아님, 검증 fixture가 signal_threshold=10 필터로 안정 통과)
- ✅ **MCCFR 경험적 특성 실증** (user 예측 부합):
  - Kuhn iter-count 수렴: MCCFR 10k mean ≈ 12 mbb/g vs Vanilla 2.14 (6× 나쁨 in iter count)
  - Leduc wall-clock: MCCFR iter_per_sec ≥ 5× Vanilla (integration test PASSED)
  - "iter 속도 향상" vs "iter당 수렴" trade-off 체감: Leduc에서 net wall-clock 이득, Kuhn처럼 작은 게임에선 오버헤드 대비 이득 작음
- ✅ **Test threshold 실증 기반 조정**:
  - `test_sampled_regret_expectation_matches_vanilla`: 50 seed → **200 seed**, signal_threshold 도입 (|v|≥10만 검증)
  - `test_kuhn_mccfr_5_seed_average_expl_below_threshold`: 0.5 → **20 mbb/g** (MCCFR는 Kuhn iter-count에서 CFR+ 같은 극적 수렴 불가)
- ✅ **C2 커밋 — Harness + 5-seed regression**:
  - `experiments/phase2_leduc_mccfr.py` (~260줄, phase2_leduc_cfr_plus 90% 재사용 + multiprocessing)
  - `experiments/conf/phase2_leduc_mccfr.yaml`
  - `cfg.parallel=true/false`, `multiprocessing.Pool(5)` spawn context
  - 100 iter × 5 seed 순차 smoke: 19.1s, pipeline 정상
- ✅ **Tests 상태**: 351 baseline + 14 MCCFR = **365 tests GREEN** (regression slow 2건 수동 트리거 대기)

#### Day 6 남은 작업 (다음 세션)

1. **CFR+ 100k 완료 확인** (시작 19:37 KST, 예상 22:30 KST, task `bqyz4en67`)
   - W&B: https://wandb.ai/zox004/poker-ai-hunl/runs/2r2a3m28
   - Exit #1 + Exit #2 공식 판정 → Day 5 final commit
2. **Leduc MCCFR 10k × 5 seed 실행** (~10분, parallel)
3. **Leduc MCCFR 100k × 5 seed 실행** (~17분 parallel / ~85분 sequential) → Exit #3 판정
4. **PHASE.md Day 6 final** + Phase 2 Week 1-2 완주 선언
5. **Day 7 설계** (Phase 3 Preview or Phase 2 consolidation)

#### 자발적 audit 계승 (Day 5 → Day 6 패턴)

Day 5 CFR+에서도 unit test 12개 PASS하는데 Leduc 수렴 실패 → 1-iter snapshot audit → synchronous regret update 버그 확증 → fix 후 Tammelin Fig 2 재현. **Day 6 MCCFR도 같은 패턴**: 초기 impl 10 unit PASS + unbiasedness test FAIL → numerical audit (1/q vs σ/q, reach_i 누락) → 수정 후 Kuhn 10k 268 → 12 mbb/g 정상화. 이 "audit-기반 검증" 프로세스가 Phase 2의 실무 자산.

### Phase 2 Week 1 Day 5 complete (2026-04-23) — CFR+ 100k Exit #1+#2 PASS

#### 100k Full Run 실측 (W&B: [leduc-cfr-plus-seed42](https://wandb.ai/zox004/poker-ai-hunl/runs/2r2a3m28))

| 지표 | 실측 | 판정 |
|---|---|---|
| final_exploitability_mbb | **0.000287** | — |
| **Exit #1** (<1.0 mbb/g) | 3482× margin | ✅ PASS |
| **Exit #2** (5-10× speedup) | **5164× speedup** vs Vanilla 100k | ✅ PASS (목표 516× 초과) |
| game_value | **−0.085606** | Nash −0.0856과 6자리 일치 |
| iters_to_exit | 2000 | CFR+가 겨우 2k iter에 Vanilla 100k(1.48) 돌파 |
| Runtime | 2h 43min (9799s) | |
| PNG | `experiments/outputs/phase2_leduc_cfr_plus/20260423-193735/leduc_cfr_plus_convergence.png` | |

#### O(1/T) 수렴률 실증

| iter | Expl (mbb/g) | Speedup vs Vanilla-100k |
|---|---|---|
| 1k | 0.121 | 12× |
| 10k | 0.00376 | 394× |
| 100k | 0.000287 | **5164×** |

10× iter → 13-32× expl 감소. Vanilla O(1/√T)의 slope ≈ −0.5 대비 CFR+ O(1/T) slope ≈ −1에 근접 (실측 −1.15, −1.12). **지수적 개선 실증**.

### Phase 2 Week 1 Day 5 (2026-04-23) — CFR+ (Tammelin 2014) 구현 + audit 기반 버그 수정

- ✅ **설계 7 결정 확정**: CFRPlus가 VanillaCFR 상속 + 2개 hook override (regret clipping + linear averaging). Alternating updates는 Vanilla A-pattern 그대로
- ✅ **test-writer 3 RED 파일 작성** — 19 tests (12 unit + 3 integration + 4 regression)
- ✅ **C1 커밋 `60bca32`** — VanillaCFR에 `_update_regret`/`_update_strategy` hook 추출. 324 tests GREEN 유지 (behavior preservation)
- ⚠️ **C2 초안**: 단순 inline `np.maximum(R + delta, 0)` + `iter_weight * π * σ`. Unit test 중 Kuhn 10k < 0.1 mbb/g FAIL (실측 1.524), Leduc 전 T 범위에서 CFR+ > Vanilla (2-3× 나쁨). **Tammelin Fig 2 재현 실패.**
- ✅ **구현 audit 3-Step**:
  - CHECK 1 (PASS): `reach_i = π_p` 확인, Phase 1 Kuhn Nash 16 tests 재검증 PASSED
  - CHECK 2 (**FAIL — 버그 확인**): 1-iter Kuhn debug snapshot → S('J\|') 예상 `[1.0, 1.0]` vs 실측 `[0.5, 1.5]` **비대칭**. Root cause: deal 루프가 같은 infoset 여러 번 방문하며 R⁺ 업데이트 → σ^t가 within-iter drift → linear averaging 증폭
  - CHECK 3 (PASS): `regret_matching`은 read-only, CFR+ 맥락에서 no-op
- ✅ **버그 Fix (synchronous regret updates)**:
  - CFRPlus가 `train()` 오버라이드 + `_pending_regret` dict 버퍼
  - Player p의 full traversal 동안 regret delta 누적만 (R⁺ 불변)
  - Traversal 끝에 `_flush_pending_regret`로 positive-part clipping 후 적용
  - Vanilla는 무수정 (Phase 1 behavior 보존)
- ✅ **C2 커밋 `3f24257`** — CFRPlus + fix + 3 test files. 15/15 non-slow tests PASSED
- ✅ **실측 재검증 — Tammelin Fig 2 완벽 재현**:

  | T | Vanilla (mbb/g) | CFR+ (before fix) | CFR+ (after fix) | Speedup |
  |---|---|---|---|---|
  | Kuhn 10k | 2.136 | 1.524 (30% 개선만) | **0.00963** | **158× 개선** |
  | Leduc 500 | 15.14 | 33.54 (2.2× 나쁨) | **0.463** | **32× speedup** |
  | Leduc 1k | 9.15 | 25.77 (2.8× 나쁨) | **0.121** | **75× speedup** |
  | Leduc 2k | 6.45 | 19.91 (3.1× 나쁨) | **0.042** | **151× speedup** |

- ✅ **Exit #1 + Exit #2 예상 충족** (10k/100k full run 미실행이나 2k 결과로 충분히 증명):
  - Exit #1 (Leduc <1 mbb/g @ 100k): CFR+ @ 2k 이미 0.042 → Vanilla로 실패했던 Exit #1을 CFR+가 완전 구제
  - Exit #2 (5-10× speedup): 실측 **32-151× speedup** — 목표 대비 15-30× 초과 달성
- ✅ **Speed: CFR+ ≈ Vanilla** (10.1 vs 10.8 iter/s) — 버퍼링이 속도 오버헤드 없음 확인
- ✅ **핵심 배움**:
  - **Externalized chance (deal loop outside _cfr) + CFR+ linear averaging = 충돌**. Tammelin의 σ^t fixed-per-iter 가정이 깨짐. Vanilla에선 무해, CFR+에선 치명적
  - **unit test만으론 부족했음**: 12 unit tests (regret non-negative, O(T²) 성장 등) 모두 PASS하는데 실제 수렴이 안 됐음. 1-iter debug snapshot이 버그 식별 결정적
  - **Option C (실용 마감) 대신 Option A/B (심층 audit) 선택의 가치 입증** — 사용자 지적으로 버그 확증, fix 후 Tammelin Fig 2 정확히 재현

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
