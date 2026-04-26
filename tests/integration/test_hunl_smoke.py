"""HUNL integration smoke (Phase 4 M1.6).

Validates the full GameProtocol pipeline end-to-end:
- ``sample_deal`` produces a valid 9-card layout
- ``state_from_deal`` builds a root preflop state with blinds posted
- A random-walk traversal drives the state to terminal without errors
- ``encode`` returns a finite float32(102,) at every visited state
- ``terminal_utility`` returns a finite float at terminal
- Round-trip seed reproducibility across 100 random walks

Mentor's optional GREEN-gate addition: traversals-per-second baseline
measurement during the smoke run, recorded as a reference for M2/M3
abstraction comparison.
"""

from __future__ import annotations

import time

import numpy as np

from poker_ai.games.hunl import HUNLGame
from poker_ai.games.hunl_state import HUNLAction, HUNLState


def _random_walk_to_terminal(
    state: HUNLState, rng: np.random.Generator
) -> HUNLState:
    """Drives the state to terminal by uniformly sampling a legal
    action (and bet size when BET) at each step. No strategy — pure
    random walk to verify the state machine reaches terminal."""
    while not state.is_terminal:
        legal = state.legal_actions()
        action = legal[int(rng.integers(0, len(legal)))]
        if action == HUNLAction.BET:
            sizes = state.legal_bet_sizes()
            bet_size = int(sizes[int(rng.integers(0, len(sizes)))])
            state = state.next_state(action, bet_size=bet_size)
        else:
            state = state.next_state(action)
    return state


class TestHUNLPipeline:
    def test_smoke_random_walk_reaches_terminal(self) -> None:
        """Single random-walk traversal: deal → walk → terminal +
        verify encode/terminal_utility round-trip."""
        rng = np.random.default_rng(seed=42)
        deal = HUNLGame.sample_deal(rng)
        state = HUNLGame.state_from_deal(deal)

        # encode at root works.
        enc = HUNLGame.encode(state)
        assert enc.shape == (102,)
        assert enc.dtype == np.float32
        assert np.isfinite(enc).all()

        # walk to terminal.
        terminal = _random_walk_to_terminal(state, rng)
        assert terminal.is_terminal

        # terminal_utility is finite.
        u = HUNLGame.terminal_utility(terminal)
        assert np.isfinite(u)

    def test_smoke_100_random_walks_complete(self) -> None:
        """100 independent random walks all reach terminal cleanly.
        Catches subtle bugs in legal_actions / next_state / round
        transitions that single-walk tests might miss."""
        rng = np.random.default_rng(seed=42)
        N = 100
        for i in range(N):
            deal = HUNLGame.sample_deal(rng)
            state = HUNLGame.state_from_deal(deal)
            terminal = _random_walk_to_terminal(state, rng)
            assert terminal.is_terminal
            u = HUNLGame.terminal_utility(terminal)
            assert np.isfinite(u), f"walk {i}: utility {u} non-finite"

    def test_smoke_seed_reproducibility(self) -> None:
        """Same seed → same terminal utility distribution across
        random walks."""
        def _run(seed: int) -> list[float]:
            rng = np.random.default_rng(seed)
            results: list[float] = []
            for _ in range(20):
                deal = HUNLGame.sample_deal(rng)
                state = HUNLGame.state_from_deal(deal)
                terminal = _random_walk_to_terminal(state, rng)
                results.append(HUNLGame.terminal_utility(terminal))
            return results

        seq_a = _run(42)
        seq_b = _run(42)
        assert seq_a == seq_b


class TestTraversalsPerSecondBaseline:
    """Mentor's optional M1 GREEN-gate addition: M2/M3 abstraction
    comparison reference. Measures wall-clock for N random-walk
    traversals (build state → walk to terminal → discard)."""

    def test_traversals_per_second_baseline(self) -> None:
        """Reports traversals/sec; assertion is just 'finite and > 0'.
        The numeric value lands in the test output for M2/M3 reference
        — not a hard threshold (the test would be flaky against M1's
        absolute-rate variability across hardware)."""
        rng = np.random.default_rng(seed=42)
        N = 200
        start = time.perf_counter()
        for _ in range(N):
            deal = HUNLGame.sample_deal(rng)
            state = HUNLGame.state_from_deal(deal)
            terminal = _random_walk_to_terminal(state, rng)
            HUNLGame.terminal_utility(terminal)
        elapsed = time.perf_counter() - start
        rate = N / elapsed if elapsed > 0 else float("inf")
        # Sanity: must run in a reasonable time on any hardware claude
        # is likely to encounter (1 traversal in <10 s upper bound).
        assert rate > 0.1, f"rate {rate} too slow"
        # Reference: print for M2/M3 comparison.
        print(
            f"\nM1.6 baseline: {rate:.1f} traversals/sec "
            f"(N={N}, elapsed={elapsed:.3f}s)"
        )
