# Phase 4 HUNL Design Spec — Draft for Mentor Review

> **Status**: Design discussion draft (2026-04-26), claude authored,
> awaiting mentor review. Outcome of M0 (`38e0bb2` GameProtocol +
> sample_deal scaling) plus the 11-item brainstorm
> ([claude 8 + mentor 3]) plus mentor's two architectural opinions
> (action layering = option B; betting history = padding-first).

## Summary

| Item | Decision |
|---|---|
| 1 State representation | hole(2) + board(0–5) + pot + stacks + 4-round betting history (frozen dataclass) |
| 2 Round structure | preflop / flop / turn / river — Pluribus blueprint match |
| 3 Action space (raw) | continuous bet sizes ∈ [min_raise, stack] + fold + call |
| 4 Action layering | **Mentor B**: Raw HUNLGame (M1) + AbstractedHUNLGame wrapper (M2) |
| 5 GameProtocol impl | `sample_deal(rng)` direct, `all_deals()` raises, `terminal_utility`, `encode` |
| 6 IntEnum (raw) | FOLD / CALL / BET (with float size) — see §3 for the bet-size design |
| 7 State immutability | `@dataclass(frozen=True, slots=True)` (Phase 2 LeducState pattern) |
| 8 Source pattern | LeducPoker structural extension; not scratch |
| 9 Stack depth | **100 BB** (Slumbot / DecisionHoldem standard) |
| 10 Hand evaluator | **treys** library (external, validated). Cross-check tests vs naive 1 k random hands |
| 11 Test strategy | 150–200 unit tests; 7 category split; M1 GREEN gate |
| Betting history enc. | **Padding to max length** (mentor): 40 slots = 10 actions × 4 rounds |

Open questions for mentor sign-off appear as `[Q]` markers in the
relevant sections.

---

## 1. State representation

```python
@dataclass(frozen=True, slots=True)
class HUNLState:
    private_cards:   tuple[int, int]              # P1 hole; P2 hole stored on the deal
    deal:            tuple[int, ...]              # full deal: 2 + 5 = 7 card ids
    board_cards:     tuple[int, ...]              # 0 / 3 / 4 / 5 cards revealed
    round_idx:       int                          # 0=preflop, 1=flop, 2=turn, 3=river
    pot:             float                        # chips in the middle, in BB
    stacks:          tuple[float, float]          # remaining behind-pot chips
    round_history:   tuple[tuple[HUNLAction, ...], ...]   # 4 tuples
    last_bet_size:   float                        # last raise size for min-raise rule
```

Cards are encoded `0..51` (rank × 4 + suit). Both players see the
board, so `board_cards` is public; `private_cards` is acting-player
perspective at infoset_key time.

Stack depth tracking is necessary for legal-action computation:
`stacks[acting]` constrains max bet, all-in determined by `stacks=0`.

[Q]: ``last_bet_size`` is needed for min-raise legality. Carry it on
HUNLState explicitly vs reconstruct from history at every call. Claude
prefers explicit (cheap, avoids subtle history-parsing bugs) — confirm.

---

## 2. Round structure

Four rounds, each with up to 10 actions before ε-edge cap (which we'll
enforce as the betting-history padding length):

- **Preflop** — board cards = 0; small/big blinds posted before action;
  P1 (small blind) acts first; round closes when both players have
  acted and bets equalised, or one folds.
- **Flop** — board reveals 3 cards; P1 acts first; raise sizes start
  at 1 BB minimum.
- **Turn** — board reveals 4th card; same betting rules as flop.
- **River** — board reveals 5th card; same betting rules; if not
  folded, showdown.

Round closure mirrors Phase 2 Leduc but with 4 rounds instead of 2,
and with continuous bet sizes instead of fixed 1-step raises.

---

## 3. Action space (raw, pre-abstraction)

Raw HUNL has continuous bet sizes. Three discrete actions plus a
continuous parameter:

```python
class HUNLAction(IntEnum):
    FOLD = 0
    CALL = 1
    BET  = 2     # carries a size; see HUNLBet below

@dataclass(frozen=True, slots=True)
class HUNLBet:
    """Raise/bet with a chip-amount payload."""
    size: float        # in BB; min_raise ≤ size ≤ stack
```

Legal actions are always a subset of `{FOLD, CALL, BET}`, but BET's
legality requires a size selection. The traversal API is therefore:

```python
state.legal_actions() -> tuple[HUNLAction, ...]  # always returned
state.legal_bet_sizes() -> tuple[float, ...]     # discrete grid (set by abstraction)
state.next_state(action: HUNLAction, bet_size: float = 0.0) -> HUNLState
```

For raw HUNLGame, `legal_bet_sizes()` returns an arbitrarily fine
grid (e.g. `np.arange(min_raise, stack + ε, 0.1 BB)`); for
AbstractedHUNLGame (M2) it returns a 4-element subset
(`{0.5p, 1p, 2p, all-in}` filtered by legality).

[Q]: Phase 2's `legal_actions() -> tuple[IntEnum, ...]` doesn't carry
bet sizes. Two options for the contract:
  - (a) extend StateProtocol with `legal_bet_sizes`; `next_state`
        takes both `action` and `bet_size`.
  - (b) flatten: `legal_actions()` returns
        `tuple[HUNLAction | (HUNLAction, float), ...]`.
  - (c) precompose discrete sizes into the action enum at the wrapper
        level only — raw HUNLGame's `legal_actions()` returns just
        `{FOLD, CALL, BET_GRID...}`.
Claude prefers (a) for the raw game (clearer types) and (c) at the
abstracted wrapper (matches Step 2's `LeducAction` pattern). Confirm.

---

## 4. Action layering — mentor option B

Two-layer design:

- **HUNLGame** (raw): continuous bet sizes, `legal_bet_sizes()` returns
  a fine grid. Used for ground-truth exploitability evaluation and as
  the swap-in foundation that abstraction wrappers build on.
- **AbstractedHUNLGame** (wrapper, M2): inherits the raw game tree,
  collapses `legal_bet_sizes()` to the 6-action set
  `{fold, call, 0.5p, 1p, 2p, all-in}`. Mirrors Step 2's
  `AbstractedLeducPoker` wrapping pattern exactly.

Step 2 demonstrated the wrapping is GameProtocol-clean and Phase 2
MCCFR plugs in unchanged. Same pattern transfers here.

---

## 5. GameProtocol implementation

```python
class HUNLGame:
    NUM_ACTIONS:  int = 3       # FOLD, CALL, BET — bet size is a separate axis
    ENCODING_DIM: int = TBD     # see §6

    def all_deals(self) -> tuple[Any, ...]:
        raise NotImplementedError(
            "HUNL has ~10^14 deals; use sample_deal(rng) (Phase 4 M0 contract)"
        )

    def sample_deal(self, rng: np.random.Generator) -> tuple[int, ...]:
        # 7-card unique permutation: 2 hole P1 + 2 hole P2 + 3 flop + 1 turn + 1 river.
        # Internal ordering choice: (p1_h1, p1_h2, p2_h1, p2_h2, flop1, flop2, flop3, turn, river)
        # — 9 elements; or compact (p1_pair, p2_pair, board5) — clarify in M1.
        ...

    def state_from_deal(self, deal): ...
    def terminal_utility(self, state) -> float: ...
    def encode(self, state) -> np.ndarray: ...
```

`sample_deal`: 52-card deck, draw 9 without replacement (or 7 if we
flatten hole pairs — M1 design choice). `np.random.Generator.choice`
with `replace=False`.

[Q]: Pre-allocate the 7-card or 9-card deal layout? Claude prefers
9-element flat tuple `(p1_h1, p1_h2, p2_h1, p2_h2, b1, b2, b3, b4, b5)`
for direct indexing; but tradeoff is that downstream encoders must
remember the layout. Confirm.

---

## 6. Encoding

Three encoding components — concatenated:

| Component | Dim | Notes |
|---|---|---|
| Acting-player hole cards | 2 × 52 = 104 (one-hot) | or 2 × 13 rank + 2 × 4 suit + pair flag = 36 |
| Board cards | 5 × 52 = 260 (one-hot, masked unrevealed) | or compact 13 + 4 per slot |
| Round one-hot | 4 | preflop / flop / turn / river |
| Pot (BB) | 1 | scalar, normalised by 100 BB |
| Stacks | 2 | scalar each, normalised by 100 BB |
| Last bet size | 1 | normalised by pot |
| Betting history | 40 (mentor's padding decision) | 10 slots × 4 rounds, encoded per slot |

Naive total (one-hot): 104 + 260 + 4 + 1 + 2 + 1 + 40 = **412**
Compact alternative (rank/suit): 36 + 60 + 4 + 1 + 2 + 1 + 40 = **144**

[Q]: One-hot vs compact rank/suit. Compact is friendlier to MCCFR
infoset_key string (smaller alphabet → shorter keys); one-hot is
network-friendly if Phase 5 brings a value network back. **Claude
recommends compact rank/suit for M1** since M1's MCCFR is tabular; the
one-hot encoder can be added later if we revive Deep CFR (we won't,
per Phase 3 conclusion). Confirm.

**M2 reconsideration (mentor sign-off 2026-04-26)**: compact integer
rank encoding is acceptable for M1 because MCCFR-tabular treats
infoset_keys as opaque strings — there is no ordinal-vs-categorical
signal to learn. **At M2 onwards, if a value network or any neural
component is reintroduced**, the compact encoding must be replaced
with one-hot (or a learned embedding) — otherwise rank ordinality
("rank 7 = rank 6 + 1") leaks a false linear structure into the
input that the network may exploit incorrectly. Recorded here so
the M2 transition does not silently re-use the M1 encoding.

Per-slot betting history encoding (mentor's padding-first call):
```
slot = (action_id ∈ {0=fold, 1=call, 2=bet}, normalised_size: float)
```
The 40 slots store `(action_id, size_in_pot)` pairs; unused slots have
`(255, 0.0)` sentinel.

---

## 7. IntEnum action (raw)

```python
class HUNLAction(IntEnum):
    FOLD = 0
    CALL = 1
    BET  = 2
```

Bet size carried separately in `next_state(action, bet_size)`. The
abstracted wrapper layers a discrete bet-size action enum on top of
this — see §4.

---

## 8. State immutability — Phase 2 pattern

`@dataclass(frozen=True, slots=True)` per LeducState. All updates via
`state.next_state(...)` returning a new instance. No mutation in-place.

This guards against shared-traversal-state bugs that would otherwise
appear under MCCFR's recursion (Phase 2 Day 1 lesson).

---

## 9. Stack depth — **100 BB** (mentor decision)

Standard heads-up cash. Slumbot 2017 / DecisionHoldem 2022 use 100 BB.
Compatibility for Phase 4 M4 benchmark. 200 BB / 50 BB are alternatives
but break benchmark direct comparison; defer to Phase 5 if needed.

---

## 10. Terminal utility — **treys hand evaluator** (claude rec)

External library (`treys` on PyPI, ~10 kLoC, C-extension equivalent
speed). Validated against millions of hand evaluations; faster than
hand-rolled Python.

Validation strategy: random-hand cross-check tests that compute the
same 7-card best-5 ranking via two independent methods (treys vs a
naive enumerate-21 approach) on 1 000 random hands; equality is the
test invariant. ~50 LoC test, runs in seconds.

Risk register: external dependency adds one line to `pyproject.toml`.
Reverse plan: if treys is too slow on the MCCFR hot path, write a
NumPy-vectorised replacement in Phase 5 (~1 week scope).

[Q]: Confirm dependency addition is acceptable — `numpy`, `torch`,
`wandb`, `omegaconf`, `hydra-core` are already listed.

Showdown logic (using treys):
```
def terminal_utility(state):
    if state ended on a fold:
        return ±pot  (sign depends on who folded)
    # showdown
    p1_rank = treys_eval(state.private_cards[0:2] + state.board_cards)
    p2_rank = treys_eval(state.private_cards_p2 + state.board_cards)
    if p1_rank < p2_rank:   # treys: lower rank = stronger hand
        return +pot/2 (P1 wins half pot above ante)
    elif p1_rank > p2_rank:
        return -pot/2 (P2 wins)
    else:
        return 0     (chop)
```

---

## 11. Test strategy — 150–200 tests, 7 categories

| Category | Count target | Examples |
|---|---|---|
| State transitions | 40 | fold ends round, call advances if matched, bet opens betting, etc. |
| Betting validity | 30 | min-raise rule, all-in, illegal-size rejection, fold-after-no-bet |
| Round transitions | 20 | preflop → flop board reveal, turn → river, terminal at river showdown |
| Showdown evaluation | 20 | high card, pair, two pair, …, straight flush; ties; treys cross-check |
| All-in handling | 15 | side-pot N/A heads-up; main pot = matched amount; remainder returned |
| Edge cases | 30 | exactly all-in min raise, blind defense, ε-tie pot rounding |
| GameProtocol compliance | 10 | NUM_ACTIONS / ENCODING_DIM constants, sample_deal uniform-ish, raises on all_deals() |

GREEN gate before M1 closure: all 150–200 unit tests pass; integration
smoke (1 random-walk traversal to terminal + utility computation)
passes.

---

## M1 deliverables (1 month)

- `src/poker_ai/games/hunl.py` — HUNLGame + HUNLState + HUNLAction
  + HUNLBet (~600–800 LoC)
- `src/poker_ai/games/hunl_hand_eval.py` — treys wrapper +
  cross-check helpers (~150 LoC)
- `tests/unit/test_hunl.py` — 150–200 unit tests (~1 200–1 500 LoC)
- `tests/unit/test_hunl_hand_eval.py` — 1 k random hand cross-check
- One integration smoke run: random-walk traversal to terminal,
  encode + sample_deal + terminal_utility round-trip

M1 review checkpoint: GREEN tests + GameProtocol compliance verified
+ Phase 2 MCCFR runs 1 iter on HUNLGame without error (no learning
expected — sanity only) + **traversals/sec baseline measurement
during the smoke run** (mentor's optional GREEN gate addition,
2026-04-26 sign-off; serves as M2/M3 abstraction comparison
reference; cost is ~1 line in the smoke script).

## Risk register

| Risk | Mitigation |
|---|---|
| treys API or build issue on M1 | naive eval fallback (M1 hot-path test confirms speed) |
| Betting history padding length insufficient (>10 actions/round real) | log overflows, increase post-M2 if observed |
| State immutability cost on hot path (frozen dataclass copy overhead) | benchmark M1 close; pre-allocate path if needed |
| Hand evaluator speed | profile post-M1; fall back to NumPy-vectorised in Phase 5 |
| Stack-depth contract change later | wrap as a parameter, defaults 100 BB, switch is one config line. **HUNLGame.__init__ takes ``starting_stack: float = 100.0`` (BB units); state.stacks initialised from it; terminal_utility caps at min(stacks) for all-in resolution. Changing 100 → 200 / 50 BB is a single yaml line.** |

---

## Open questions for mentor (recap)

- Q1 §1: ``last_bet_size`` carried on HUNLState? (claude: yes)
- Q2 §3: legal action contract — (a)/(b)/(c)? (claude: a for raw, c for abstracted)
- Q3 §5: 9-element vs 7-element flat deal layout? (claude: 9-flat)
- Q4 §6: one-hot vs compact rank/suit encoding? (claude: compact for M1)
- Q5 §10: treys external dependency OK? (claude: yes)

Sign-off on these five → M1 implementation begins. Self-audits running:
claude 15 / mentor 7. Spec authored 2026-04-26.
