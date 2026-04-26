# Phase 3 Lessons — Negative-result-driven path validation

> **Status**: Outline draft (2026-04-26). Final write-up after Phase 4
> Step 3 milestones. Internal documentation for the project's research
> notes; not intended for external publication unless explicitly
> requested by the user.

## Central thesis

Brown 2019 Deep CFR has an architectural floor on medium-scale games
(Leduc, 288 infosets) that single- and multi-axis interventions cannot
push past — proven empirically across nine days of experiments. The
validated alternative is the Pluribus path: Linear MCCFR with lossy
abstraction, where lossy abstraction is **net positive** under finite
compute because the sample-efficiency gain dominates the
information-loss cost. This document records 21 educational assets and
18 self-audits accumulated across Phase 3, organized for transfer into
Phase 4 HUNL implementation.

## Outline

### 1. Phase 3 timeline (skim)
- Day 0–2: scaffolding + Kuhn smoke + reference algorithm correction
- Day 3–3c: Leduc Deep CFR baseline + Fair NAE framework
- Day 4: Cap 4×128 single-seed scan
- Day 5: σ_seed multi-seed measurement (n=5)
- Day 5 Step 5: L-B (Schmid baseline) implementation failure
- Day 6: Huber loss rejected (transient + over-regularization)
- Day 7: advantage_epochs ↑ marginal
- Day 8: Cap × epoch saturation
- Day 9: T axis ADVERSE
- Phase 3 conclusion + pivot decision
- Phase 4 Step 2: Pluribus path validation PASS

### 2. Three core findings

#### 2.1 Brown 2019 Deep CFR architectural floor (Phase 3 days 4–9)

3-axis evidence:
- Cap (3×64 → 4×128): Δ Primary A = 0.8 σ (noise within)
- Huber (MSE → robust): transient peak +7.6 σ at T=100, baseline at T=500
- (e) advantage_epochs (4 → 10): +1.45 σ borderline
- T axis (T=500 → T=2000): σ̄_expl ADVERSE +18.5 σ within-band

Mechanism candidates considered and outcomes:
- (a) capacity → falsified at 4×128 saturation
- (b) target variance → L-B impl-broken, Huber rejected
- (b') decouple advantage/strategy → false framing (unidirectional)
- (c) self-correlation noise → empirically rejected (3.65 σ above
  random init floor)
- (d)/(f) metric mismatch → rejected (Pearson 0.9565 ≥ 0.95 cutoff)
- (e) Brown 2019 defaults → epoch ↑ saturates with Cap

Conclusion: σ̄_expl ≈ 140–150 mbb/g floor on Leduc with this algorithm,
regardless of axis. (Education asset #19)

#### 2.2 Lossy abstraction is net positive under finite compute (Step 2)

Quantified: 33 % rank-entropy loss → -58 % σ̄_expl, -50 % wall-clock.
Same MCCFR algorithm, same compute budget. Mechanism: 192 buckets
share 100k samples (~520/bucket) vs 288 raw infosets (~347/bucket).
The information-loss cost from collapsing {J, Q} → "L" is more than
offset by the lower per-bucket sampling variance. (Education asset #20)

This is the textbook Pluribus result, now empirically grounded for our
Phase 4 transfer: when HUNL goes 10^14 raw infosets → 10^7 buckets,
the same mechanism stands to gain even more.

#### 2.3 Algorithm-floor-aware metric design (assets #8, #21)

Metrics imported uncritically across algorithms produce misleading
verdicts. MCCFR T=100k External Sampling has an algorithm-floor near
60 mbb/g due to sampling variance — comparing it to Vanilla CFR's
< 1 mbb/g convergence target gives spurious "fail" judgments.
The correct framing is **same-algorithm baseline relative comparison**.

Generalizes the earlier Day 3c lesson (Pearson reference quality ≠
network quality unless the reference has a matched compute budget).

### 3. The 21 educational assets — compact reference

Rendered as a table in the final write-up. Each entry: id, one-line
asset, day registered, Phase 4 transfer flag.

[TODO: insert table from PHASE.md Phase 3 Conclusion]

Phase-4-critical (must check during HUNL implementation):
- #4 capacity decouples advantage vs strategy — multi-seed proven
- #11 algorithmic changes need T≥50 convergence smoke (unit tests
  insufficient)
- #13 variance-reduction must preserve E[r̂]=E[r_legacy] (Lemma 1)
- #17 Deep CFR σ̄_expl is not monotone (multi-seed/multi-checkpoint
  averaging required)
- #18 σ̄_expl(T) can be ADVERSE past saturation
- #19 Brown 2019 Deep CFR architectural floor on medium games
- #20 Lossy abstraction is net positive under finite compute
- #21 Algorithm-floor-aware metric cutoffs

### 4. Methodology — Mentor-Coder self-audit pattern

[Separate appendix per mentor's option D framing]

Across Phase 3, 18 self-corrections were registered:

- Mentor: 7 (CFR+ → Vanilla ref / buffer-side → loss-side weighting /
  single-seed → fair-data ceiling / Spearman → Pearson / bidirectional
  → unidirectional / "Deep CFR is our path" → Pluribus path / cross-
  algorithm cutoff)
- Claude: 15 (D-2 EMA root cause / linear weighting source / wandb
  online recovery / yaml [50] missing / num_hidden_layers=3 typo /
  L-B double-architectural-error / Huber δ data-grounded / (b')
  framing / #15 unbiasedness asset / σ̄_expl monotone assumption / T
  extension prediction wrong / #19 reframe / 3-bucket = identity /
  Step 2 cutoff cross-algorithm / sequential vs parallel run choice)

Pattern: each iteration, the implicit assumption from the prior
session's design decision was challenged by either fresh data or a
review pass. The pattern itself was treated as an asset (#8: "first
metric design is provisional, must iterate against data").

For HUNL Phase 4, the same pattern is expected to surface around
abstraction granularity choices, action-space cardinality, and
subgame-solver convergence guarantees.

### 5. Phase 4 transfer plan

The Phase 3 → Phase 4 transition is on the Pluribus path:
- HUNL game engine (own implementation, GameProtocol-compatible)
- E[HS²] card abstraction per round (Ganzfried & Sandholm 2014)
- 6-action abstraction: {fold, call, 0.5p, 1p, 2p, all-in}
- Linear MCCFR adaptation (Phase 2 + Step 2 wrapper pattern)
- Subgame solving deferred to Phase 5

Risk register (carried from Phase 3):
- GameProtocol scaling (all_deals → sample_deal): Step 3 design issue
- HUNL state-space abstraction granularity calibration
- Multi-seed budget for HUNL (~5×) vs single experiment time

### 6. References

- Brown 2019 Deep CFR (ICML)
- Schmid 2019 VR-MCCFR
- Ganzfried & Sandholm 2014 abstraction
- Brown & Sandholm 2017 Libratus
- Brown & Sandholm 2019 Pluribus
- Lanctot 2009 MCCFR
- Tammelin 2014 CFR+
- Zinkevich 2007 CFR

---

[End of outline. Final write-up: ~5–10 pages, content from PHASE.md
Phase 3 Conclusion + Step 2 sections, plus narrative connecting the
three core findings.]
