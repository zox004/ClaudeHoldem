"""Integration tests: best-response + exploitability against CFR trainer output.

Verifies that the exploitability module correctly interacts with the Vanilla CFR
trainer by exercising the full path strategy → BR → exploitability on real
training output.

Target modules:
    src/poker_ai/eval/exploitability.py (pending)
    src/poker_ai/algorithms/vanilla_cfr.py (implemented)
"""

from __future__ import annotations

import numpy as np
import pytest

from poker_ai.algorithms.vanilla_cfr import VanillaCFR
from poker_ai.eval.exploitability import best_response_value, exploitability
from poker_ai.games.kuhn import KuhnPoker

from tests.unit.test_exploitability import (  # reuse fixture helpers
    make_kuhn_nash_strategy,
    make_uniform_strategy,
)


def test_trained_cfr_is_less_exploitable_than_uniform() -> None:
    """A CFR-trained strategy (even after only 100 iterations) should already
    be less exploitable than the uniform strategy.

    This is a structural sanity check: if the BR module is broken such that
    it returns a constant, or if it mis-applies sign, this test fails.
    """
    uniform_expl = exploitability(KuhnPoker(), make_uniform_strategy())

    cfr = VanillaCFR(game=KuhnPoker(), n_actions=2)
    cfr.train(100)
    cfr_expl = exploitability(KuhnPoker(), cfr.average_strategy())

    assert cfr_expl < uniform_expl, (
        f"CFR(100 iter) expl={cfr_expl:.6f} not less than uniform expl={uniform_expl:.6f}"
    )


def test_best_response_values_sum_to_zero_at_nash() -> None:
    """Zero-sum: at Nash, v_P1 + v_P2 = 0 (they are mirror images). So
    BR_P1(Nash_P2) + BR_P2(Nash_P1) = v_P1_Nash + v_P2_Nash = 0.
    """
    nash = make_kuhn_nash_strategy(0.0)
    game = KuhnPoker()
    br0 = best_response_value(game, nash, responding_player=0)
    br1 = best_response_value(game, nash, responding_player=1)
    assert abs(br0 + br1) < 1e-9, f"BR sum = {br0 + br1:.12f}, expected 0"


def test_exploitability_strictly_decreases_from_100_to_1k_iters() -> None:
    """Longer training produces strictly-smaller-expl strategies. Uses two
    fresh trainers (not a single trainer re-trained) to make the comparison
    clean and independent.
    """
    game = KuhnPoker()

    cfr_short = VanillaCFR(game=game, n_actions=2)
    cfr_short.train(100)
    expl_short = exploitability(game, cfr_short.average_strategy())

    cfr_long = VanillaCFR(game=game, n_actions=2)
    cfr_long.train(1000)
    expl_long = exploitability(game, cfr_long.average_strategy())

    assert expl_long < expl_short, (
        f"expl(1000 iter)={expl_long:.6f} not smaller than expl(100 iter)={expl_short:.6f}"
    )
