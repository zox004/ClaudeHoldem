"""Local Best Response (LBR) — Phase 4 M3.3 (proper infoset-aggregated).

Reference: Lisý & Bowling 2016 (IJCAI), "Equilibrium Approximation
Quality of Current No-Limit Poker Bots", §3.

LBR plays myopically (single-step lookahead) at every infoset, with a
fixed rollout policy for any future LBR decisions. By Lemma 1 of the
paper, LBR's expected value is a strict lower bound on the true
best-response value: ``LBR(σ) ≤ BR(σ)`` for every strategy σ.

Implementation
--------------
The construction is the standard "build the LBR policy over visited
infosets, then evaluate" 2-pass:

  Pass 1 (collect): traverse the game tree once per deal. At each LBR
    infoset I encountered, accumulate
    ``sum_a += π_{-i}(h) · v(state.next(a) | rollout LBR + σ opp)``
    for every legal action a, plus the per-action visit count.
    Continue traversal by playing ``rollout_policy`` at LBR's nodes.

  Pass 2 (argmax): for every visited infoset I, set
    ``policy[I] = argmax_a (sum_a / count_a)``.

  Pass 3 (evaluate): traverse the game tree under ``policy`` for LBR
    and ``σ`` for the opponent; the per-deal mean is the LBR
    exploitability sample.

Aggregation by infoset_key matches :mod:`poker_ai.eval.exploitability`'s
Pass 1 pattern (``π_{-i}``-weighted CFV per infoset). The argmax is
taken over the *infoset-aggregated* CFV, NOT over per-state values —
this is what makes LBR a *lower* bound on BR rather than an
information-leaking upper bound.

Two execution modes
-------------------
- **Exact (Kuhn / Leduc)**: ``game.all_deals()`` is enumerable, so the
  chance dimension is fully traversed. The opponent dimension is
  σ-weighted enumeration. Result is deterministic.
- **Sampled (HUNL / AbstractedHUNL)**: ``game.all_deals()`` raises
  ``NotImplementedError``; chance and opponent are both Monte Carlo
  sampled. ``n_samples`` controls deal samples; an inner ``n_mc``
  budget controls per-(I, a) subtree value estimates so traversal
  depth stays linear instead of exponential in the opponent's
  σ-enumeration tree.

Implementation history (M3.3 self-audit #22)
--------------------------------------------
The first M3.3 pass implemented a state-level argmax which let LBR
peek at the opponent's hidden card (because ``state.next_state`` was
evaluated in chip units against the realised state, not the
infoset). That implementation was an *upper* bound on BR, not a
lower bound — directly violating Lemma 1. Caught by
``test_kuhn_near_nash_small`` (LBR=0.125 ≫ exact=0.002 for Kuhn
T=10k). The fix is the infoset-aggregation pass above.

Phase 5 hooks
-------------
- AIVAT (Burch et al. 2018) — variance-reduced unbiased
  exploitability estimator. Distinct algorithm; ~500 LoC.
- Distribution-aware rollout (e.g. learned σ_rollout instead of
  always-call). Tightens LBR. Lisý & Bowling §4 alternative.
- Truncated exact BR over abstracted state space.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import numpy as np

from poker_ai.games.protocol import GameProtocol, StateProtocol


# =============================================================================
# Public API
# =============================================================================
@dataclass(frozen=True, slots=True)
class LBRConfig:
    """Configuration bundle for an LBR exploitability evaluation.

    Attributes:
        n_samples: number of deal samples (sampled mode) or ignored
            (exact mode — enumerates ``game.all_deals()``).
        seed: RNG seed for reproducibility.
        paired: when True, the same deal sample is used for both P0
            and P1 LBR computations (control variate).
    """

    n_samples: int = 200
    seed: int = 42
    paired: bool = True


_RolloutPolicy = Callable[[StateProtocol], "IntEnum | int"]
_DEFAULT_MC_SUBTREE: int = 30


def default_rollout_policy(game: GameProtocol) -> _RolloutPolicy:
    """Returns the standard "always-call-or-check" rollout policy.

    Picks the action with integer value 1 if it is legal at the
    current state (CALL in Kuhn / Leduc / raw HUNL / AbstractedHUNL),
    else falls back to the first legal action. This is the Lisý &
    Bowling 2016 default and matches HUNL benchmark conventions.
    """
    del game

    def _rollout(state: StateProtocol) -> IntEnum | int:
        legal = state.legal_actions()
        for a in legal:
            if int(a) == 1:
                return a
        return legal[0]

    return _rollout


def lbr_value(
    game: GameProtocol,
    strategy: dict[str, np.ndarray],
    responder: int,
    n_samples: int,
    rng: np.random.Generator,
    rollout_policy: _RolloutPolicy | None = None,
) -> float:
    """LBR's expected value for ``responder`` given the opponent's
    ``strategy``. Lower bound on the true BR value.

    Args:
        n_samples: number of deal samples used in sampled mode (for
            games whose ``all_deals`` is not enumerable). Ignored
            in exact mode.

    Raises:
        ValueError: if ``n_samples < 1`` or if ``rollout_policy``
            returns an illegal action at any state.
    """
    if n_samples < 1:
        raise ValueError(f"n_samples must be ≥ 1, got {n_samples}")
    if rollout_policy is None:
        rollout_policy = default_rollout_policy(game)
    num_actions = game.NUM_ACTIONS
    deals = _generate_deals(game, n_samples, rng)
    is_exact = _is_exact_mode(game)
    policy = _build_lbr_policy(
        game, strategy, responder, deals, rollout_policy, num_actions,
        rng, is_exact,
    )
    total = 0.0
    for deal in deals:
        state = game.state_from_deal(deal)
        total += _evaluate_with_policy(
            game, state, strategy, responder, policy, rollout_policy,
            num_actions, rng, is_exact,
        )
    return float(total / len(deals))


def lbr_exploitability(
    game: GameProtocol,
    strategy: dict[str, np.ndarray],
    n_samples: int,
    rng: np.random.Generator,
    rollout_policy: _RolloutPolicy | None = None,
    paired: bool = True,
) -> tuple[float, float]:
    """LBR exploitability ``(LBR_P0 + LBR_P1) / 2`` plus the sample
    standard error.

    Returns ``(mean, se)``. ``se`` is the sample standard error of
    the mean over deals in *paired* mode, or the combined SE of two
    independent sample means in *unpaired* mode. ``se = 0.0`` when
    ``n_samples < 2``.

    Args:
        paired: when True (default), use the same deal sample for
            both P0 and P1 evaluations (control variate variance
            reduction). When False, draw independent deal samples
            from RNGs split off ``rng``.
    """
    if n_samples < 1:
        raise ValueError(f"n_samples must be ≥ 1, got {n_samples}")
    if rollout_policy is None:
        rollout_policy = default_rollout_policy(game)
    num_actions = game.NUM_ACTIONS
    is_exact = _is_exact_mode(game)

    if paired:
        deals = _generate_deals(game, n_samples, rng)
        policy_p0 = _build_lbr_policy(
            game, strategy, 0, deals, rollout_policy, num_actions,
            rng, is_exact,
        )
        policy_p1 = _build_lbr_policy(
            game, strategy, 1, deals, rollout_policy, num_actions,
            rng, is_exact,
        )
        per_sample: list[float] = []
        for deal in deals:
            v0 = _evaluate_with_policy(
                game, game.state_from_deal(deal), strategy, 0,
                policy_p0, rollout_policy, num_actions, rng, is_exact,
            )
            v1 = _evaluate_with_policy(
                game, game.state_from_deal(deal), strategy, 1,
                policy_p1, rollout_policy, num_actions, rng, is_exact,
            )
            per_sample.append((v0 + v1) / 2.0)
        arr = np.asarray(per_sample, dtype=np.float64)
        mean = float(arr.mean())
        if len(per_sample) >= 2:
            se = float(arr.std(ddof=1) / np.sqrt(len(per_sample)))
        else:
            se = 0.0
        return mean, se

    # paired=False: split rng for P0 / P1 sub-runs.
    seed_p0 = int(rng.integers(0, 2**31 - 1))
    seed_p1 = int(rng.integers(0, 2**31 - 1))
    rng_p0 = np.random.default_rng(seed_p0)
    rng_p1 = np.random.default_rng(seed_p1)
    deals_p0 = _generate_deals(game, n_samples, rng_p0)
    deals_p1 = _generate_deals(game, n_samples, rng_p1)
    policy_p0 = _build_lbr_policy(
        game, strategy, 0, deals_p0, rollout_policy, num_actions,
        rng_p0, is_exact,
    )
    policy_p1 = _build_lbr_policy(
        game, strategy, 1, deals_p1, rollout_policy, num_actions,
        rng_p1, is_exact,
    )
    samples_p0 = [
        _evaluate_with_policy(
            game, game.state_from_deal(d), strategy, 0,
            policy_p0, rollout_policy, num_actions, rng_p0, is_exact,
        )
        for d in deals_p0
    ]
    samples_p1 = [
        _evaluate_with_policy(
            game, game.state_from_deal(d), strategy, 1,
            policy_p1, rollout_policy, num_actions, rng_p1, is_exact,
        )
        for d in deals_p1
    ]
    a0 = np.asarray(samples_p0, dtype=np.float64)
    a1 = np.asarray(samples_p1, dtype=np.float64)
    mean = float((a0.mean() + a1.mean()) / 2.0)
    if len(samples_p0) >= 2 and len(samples_p1) >= 2:
        var0 = float(a0.var(ddof=1))
        var1 = float(a1.var(ddof=1))
        # SE of (mean_p0 + mean_p1) / 2 with independent estimators.
        se = float(np.sqrt((var0 + var1) / len(samples_p0)) / 2.0)
    else:
        se = 0.0
    return mean, se


# =============================================================================
# Mode detection + deal generation
# =============================================================================
def _is_exact_mode(game: GameProtocol) -> bool:
    """True if ``game.all_deals()`` is enumerable (Kuhn, Leduc).
    False for HUNL (which raises NotImplementedError)."""
    try:
        game.all_deals()
        return True
    except NotImplementedError:
        return False


def _generate_deals(
    game: GameProtocol,
    n_samples: int,
    rng: np.random.Generator,
) -> list[Any]:
    """Returns the deal list — full enumeration when available, else
    ``n_samples`` independent samples."""
    if _is_exact_mode(game):
        return list(game.all_deals())
    return [game.sample_deal(rng) for _ in range(n_samples)]


# =============================================================================
# Policy construction (Pass 1 + Pass 2)
# =============================================================================
def _build_lbr_policy(
    game: GameProtocol,
    strategy: dict[str, np.ndarray],
    responder: int,
    deals: list[Any],
    rollout_policy: _RolloutPolicy,
    num_actions: int,
    rng: np.random.Generator,
    is_exact: bool,
) -> dict[str, Any]:
    sums: dict[str, dict[int, float]] = {}
    counts: dict[str, dict[int, int]] = {}
    legal_cache: dict[str, tuple[Any, ...]] = {}

    if is_exact:
        chance_weight = 1.0 / len(deals)
        for deal in deals:
            state = game.state_from_deal(deal)
            _exact_phase1(
                game, state, strategy, responder, rollout_policy,
                num_actions, sums, counts, legal_cache, chance_weight,
            )
    else:
        for deal in deals:
            state = game.state_from_deal(deal)
            _sampled_phase1(
                game, state, strategy, responder, rollout_policy,
                num_actions, sums, counts, legal_cache, rng,
            )
    return _policy_from_sums(sums, counts, legal_cache)


def _policy_from_sums(
    sums: dict[str, dict[int, float]],
    counts: dict[str, dict[int, int]],
    legal_cache: dict[str, tuple[Any, ...]],
) -> dict[str, Any]:
    policy: dict[str, Any] = {}
    for infoset, action_sums in sums.items():
        legal = legal_cache[infoset]
        cfv = []
        for a in legal:
            i = int(a)
            cnt = counts[infoset].get(i, 0)
            avg = (
                action_sums.get(i, 0.0) / cnt
                if cnt > 0
                else -float("inf")
            )
            cfv.append(avg)
        best_idx = int(np.argmax(cfv))
        policy[infoset] = legal[best_idx]
    return policy


# =============================================================================
# Exact-mode traversal (Kuhn / Leduc)
# =============================================================================
def _exact_phase1(
    game: GameProtocol,
    state: StateProtocol,
    strategy: dict[str, np.ndarray],
    responder: int,
    rollout_policy: _RolloutPolicy,
    num_actions: int,
    sums: dict[str, dict[int, float]],
    counts: dict[str, dict[int, int]],
    legal_cache: dict[str, tuple[Any, ...]],
    reach_opp: float,
) -> None:
    if state.is_terminal:
        return
    if state.current_player == responder:
        infoset = state.infoset_key
        legal = state.legal_actions()
        legal_cache.setdefault(infoset, legal)
        sums.setdefault(infoset, {})
        counts.setdefault(infoset, {})
        for a in legal:
            v = _exact_rollout(
                game, state.next_state(a), strategy, responder,
                rollout_policy, num_actions,
            )
            i_a = int(a)
            sums[infoset][i_a] = sums[infoset].get(i_a, 0.0) + reach_opp * v
            counts[infoset][i_a] = counts[infoset].get(i_a, 0) + 1
        chosen = _legal_action_for_int(legal, int(rollout_policy(state)))
        _exact_phase1(
            game, state.next_state(chosen), strategy, responder,
            rollout_policy, num_actions, sums, counts, legal_cache,
            reach_opp,
        )
        return
    # Opponent: σ-weighted enumeration.
    opp = _opp_strategy_at(strategy, state.infoset_key, num_actions)
    for a in state.legal_actions():
        p = float(opp[int(a)])
        if p == 0.0:
            continue
        _exact_phase1(
            game, state.next_state(a), strategy, responder,
            rollout_policy, num_actions, sums, counts, legal_cache,
            reach_opp * p,
        )


def _exact_rollout(
    game: GameProtocol,
    state: StateProtocol,
    strategy: dict[str, np.ndarray],
    responder: int,
    rollout_policy: _RolloutPolicy,
    num_actions: int,
) -> float:
    """Subtree value with rollout for LBR + σ for opponent (exact)."""
    if state.is_terminal:
        u_p1 = game.terminal_utility(state)
        return u_p1 if responder == 0 else -u_p1
    if state.current_player == responder:
        chosen = _legal_action_for_int(
            state.legal_actions(), int(rollout_policy(state))
        )
        return _exact_rollout(
            game, state.next_state(chosen), strategy, responder,
            rollout_policy, num_actions,
        )
    opp = _opp_strategy_at(strategy, state.infoset_key, num_actions)
    total = 0.0
    for a in state.legal_actions():
        p = float(opp[int(a)])
        if p == 0.0:
            continue
        total += p * _exact_rollout(
            game, state.next_state(a), strategy, responder,
            rollout_policy, num_actions,
        )
    return total


# =============================================================================
# Sampled-mode traversal (HUNL / AbstractedHUNL)
# =============================================================================
def _sampled_phase1(
    game: GameProtocol,
    state: StateProtocol,
    strategy: dict[str, np.ndarray],
    responder: int,
    rollout_policy: _RolloutPolicy,
    num_actions: int,
    sums: dict[str, dict[int, float]],
    counts: dict[str, dict[int, int]],
    legal_cache: dict[str, tuple[Any, ...]],
    rng: np.random.Generator,
) -> None:
    while not state.is_terminal:
        if state.current_player == responder:
            infoset = state.infoset_key
            legal = state.legal_actions()
            legal_cache.setdefault(infoset, legal)
            sums.setdefault(infoset, {})
            counts.setdefault(infoset, {})
            for a in legal:
                v = _sampled_subtree_mc(
                    game, state.next_state(a), strategy, responder,
                    rollout_policy, num_actions, rng,
                    _DEFAULT_MC_SUBTREE,
                )
                i_a = int(a)
                sums[infoset][i_a] = sums[infoset].get(i_a, 0.0) + v
                counts[infoset][i_a] = counts[infoset].get(i_a, 0) + 1
            chosen = _legal_action_for_int(legal, int(rollout_policy(state)))
            state = state.next_state(chosen)
        else:
            state = _sample_opp_child(state, strategy, num_actions, rng)


def _sampled_subtree_mc(
    game: GameProtocol,
    state: StateProtocol,
    strategy: dict[str, np.ndarray],
    responder: int,
    rollout_policy: _RolloutPolicy,
    num_actions: int,
    rng: np.random.Generator,
    n_mc: int,
) -> float:
    """MC estimate of subtree value: rollout for LBR, σ-sampled for opp.

    Single full trajectory per MC sample (not exhaustive enumeration);
    the inner loop keeps cost O(depth × n_mc) instead of exponential.
    """
    total = 0.0
    for _ in range(n_mc):
        s = state
        while not s.is_terminal:
            if s.current_player == responder:
                a = _legal_action_for_int(
                    s.legal_actions(), int(rollout_policy(s))
                )
            else:
                s = _sample_opp_child(s, strategy, num_actions, rng)
                continue
            s = s.next_state(a)
        u_p1 = game.terminal_utility(s)
        total += u_p1 if responder == 0 else -u_p1
    return total / n_mc


def _sample_opp_child(
    state: StateProtocol,
    strategy: dict[str, np.ndarray],
    num_actions: int,
    rng: np.random.Generator,
) -> StateProtocol:
    """Samples one opponent action by σ (renormalised over legal),
    returns the resulting child state."""
    opp = _opp_strategy_at(strategy, state.infoset_key, num_actions)
    legal = state.legal_actions()
    weights = np.array([float(opp[int(a)]) for a in legal], dtype=np.float64)
    s = weights.sum()
    if s <= 0.0:
        weights = np.ones(len(legal), dtype=np.float64)
        s = float(len(legal))
    weights = weights / s
    idx = int(rng.choice(len(legal), p=weights))
    return state.next_state(legal[idx])


# =============================================================================
# Pass 3 — evaluation under fixed policy
# =============================================================================
def _evaluate_with_policy(
    game: GameProtocol,
    state: StateProtocol,
    strategy: dict[str, np.ndarray],
    responder: int,
    policy: dict[str, Any],
    rollout_policy: _RolloutPolicy,
    num_actions: int,
    rng: np.random.Generator,
    is_exact: bool,
) -> float:
    if is_exact:
        return _exact_evaluate(
            game, state, strategy, responder, policy, rollout_policy,
            num_actions,
        )
    return _sampled_evaluate(
        game, state, strategy, responder, policy, rollout_policy,
        num_actions, rng,
    )


def _exact_evaluate(
    game: GameProtocol,
    state: StateProtocol,
    strategy: dict[str, np.ndarray],
    responder: int,
    policy: dict[str, Any],
    rollout_policy: _RolloutPolicy,
    num_actions: int,
) -> float:
    if state.is_terminal:
        u_p1 = game.terminal_utility(state)
        return u_p1 if responder == 0 else -u_p1
    if state.current_player == responder:
        infoset = state.infoset_key
        if infoset in policy:
            chosen = policy[infoset]
        else:
            chosen = _legal_action_for_int(
                state.legal_actions(), int(rollout_policy(state))
            )
        return _exact_evaluate(
            game, state.next_state(chosen), strategy, responder,
            policy, rollout_policy, num_actions,
        )
    opp = _opp_strategy_at(strategy, state.infoset_key, num_actions)
    total = 0.0
    for a in state.legal_actions():
        p = float(opp[int(a)])
        if p == 0.0:
            continue
        total += p * _exact_evaluate(
            game, state.next_state(a), strategy, responder,
            policy, rollout_policy, num_actions,
        )
    return total


def _sampled_evaluate(
    game: GameProtocol,
    state: StateProtocol,
    strategy: dict[str, np.ndarray],
    responder: int,
    policy: dict[str, Any],
    rollout_policy: _RolloutPolicy,
    num_actions: int,
    rng: np.random.Generator,
) -> float:
    while not state.is_terminal:
        if state.current_player == responder:
            infoset = state.infoset_key
            if infoset in policy:
                chosen = policy[infoset]
            else:
                chosen = _legal_action_for_int(
                    state.legal_actions(), int(rollout_policy(state))
                )
            state = state.next_state(chosen)
        else:
            state = _sample_opp_child(state, strategy, num_actions, rng)
    u_p1 = game.terminal_utility(state)
    return u_p1 if responder == 0 else -u_p1


# =============================================================================
# Helpers
# =============================================================================
def _legal_action_for_int(
    legal: tuple[Any, ...], a_int: int
) -> Any:
    """Returns the legal IntEnum member matching ``int(action) == a_int``;
    raises ValueError when no such action is legal."""
    for x in legal:
        if int(x) == a_int:
            return x
    raise ValueError(
        f"action {a_int} is not legal at this state; "
        f"legal_actions={[int(y) for y in legal]}"
    )


def _opp_strategy_at(
    strategy: dict[str, np.ndarray],
    infoset_key: str,
    num_actions: int,
) -> np.ndarray:
    """Lookup with uniform-fallback. Mirrors
    :func:`poker_ai.eval.exploitability._opponent_strategy_at`."""
    return strategy.get(infoset_key, np.full(num_actions, 1.0 / num_actions))
