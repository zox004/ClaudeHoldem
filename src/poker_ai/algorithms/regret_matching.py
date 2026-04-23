"""Regret Matching (Hart & Mas-Colell 2000; Neller & Lanctot 2013 Algorithm 1).

The building block of CFR: given cumulative regrets over actions, produce a
probability distribution proportional to the positive part of regret.

This module also provides :class:`RegretMatcher`, a *single-infoset* convenience
wrapper used for RPS-style demonstrations and Phase 1 self-play experiments.
"""

from __future__ import annotations

import numpy as np


def regret_matching(
    cumulative_regret: np.ndarray,
    legal_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Convert cumulative regrets to a strategy (probability distribution).

    Hart & Mas-Colell 2000; Neller & Lanctot 2013 Algorithm 1.

    Base rule (``legal_mask=None``, Phase 1 Kuhn-compatible path):
        positive = max(cumulative_regret, 0)
        if sum(positive) > 0:  strategy = positive / sum(positive)
        else:                  strategy = uniform over all actions

    With ``legal_mask`` (Phase 2 Leduc path for state-dependent legality):
        positive = max(cumulative_regret, 0) * legal_mask
        if sum(positive) > 0:  strategy = positive / sum(positive)
        else:                  strategy = uniform over LEGAL actions only
    Illegal slots always receive probability 0, regardless of any positive
    regret they may have accumulated by accident.

    Args:
        cumulative_regret: 1-D array of per-action cumulative regrets.
        legal_mask: Optional boolean mask of same length. If supplied, must
            have at least one ``True`` entry.

    Returns:
        Probability distribution with the same shape as the input. Entries
        are non-negative and sum to 1. When a ``legal_mask`` is supplied,
        illegal slots are exactly zero.
    """
    regret = np.asarray(cumulative_regret, dtype=np.float64)
    positive = np.maximum(regret, 0.0)
    if legal_mask is not None:
        mask_f = np.asarray(legal_mask, dtype=np.float64)
        positive = positive * mask_f
    total = positive.sum()
    if total > 0.0:
        strategy: np.ndarray = positive / total
        return strategy
    if legal_mask is not None:
        mask_f = np.asarray(legal_mask, dtype=np.float64)
        n_legal = mask_f.sum()
        assert n_legal > 0, "legal_mask must have at least one True entry"
        fallback_legal: np.ndarray = mask_f / n_legal
        return fallback_legal
    fallback_uniform: np.ndarray = np.ones_like(regret) / regret.size
    return fallback_uniform


class RegretMatcher:
    """Single-infoset regret matcher for RPS-style demonstrations.

    For extensive-form games like Kuhn/Leduc, use the functional
    :func:`regret_matching` directly within CFR tree traversal.
    Do NOT reuse this class for CFR implementations.
    """

    def __init__(self, n_actions: int, rng: np.random.Generator) -> None:
        self.n_actions = n_actions
        self._rng = rng
        self._cumulative_regret = np.zeros(n_actions, dtype=np.float64)
        self._cumulative_strategy = np.zeros(n_actions, dtype=np.float64)

    def current_strategy(self) -> np.ndarray:
        """Strategy for this iteration, derived from cumulative regret."""
        return regret_matching(self._cumulative_regret)

    def sample_action(self) -> int:
        """Sample an action index from ``current_strategy`` using the injected RNG."""
        strategy = self.current_strategy()
        return int(self._rng.choice(self.n_actions, p=strategy))

    def update(self, action_utilities: np.ndarray) -> None:
        """Accumulate one step of regret and strategy.

        For each action ``a``, instantaneous regret is ``u_a - <strategy, u>``.
        Both cumulative regret and cumulative strategy are updated; the latter
        feeds :meth:`average_strategy`.
        """
        utilities = np.asarray(action_utilities, dtype=np.float64)
        strategy = self.current_strategy()
        expected_utility = float(strategy @ utilities)
        self._cumulative_regret += utilities - expected_utility
        self._cumulative_strategy += strategy

    def average_strategy(self) -> np.ndarray:
        """Time-averaged strategy over all :meth:`update` calls.

        NOTE: This simple normalization works only for single-infoset games.
        In extensive-form CFR, average strategy must be weighted by
        reach probabilities per infoset.
        """
        total = self._cumulative_strategy.sum()
        if total > 0.0:
            normalized: np.ndarray = self._cumulative_strategy / total
            return normalized
        uniform: np.ndarray = (
            np.ones(self.n_actions, dtype=np.float64) / self.n_actions
        )
        return uniform
