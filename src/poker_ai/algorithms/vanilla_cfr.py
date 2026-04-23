"""Vanilla Counterfactual Regret Minimization (CFR) for 2-player zero-sum games.

Tabular implementation following Zinkevich et al. 2007, "Regret Minimization in
Games with Incomplete Information"; pseudocode structure mirrors Neller &
Lanctot 2013, Algorithm 2.

Traversal pattern
-----------------
**Alternating one-player traversal** (the "A pattern"): one logical iteration
performs two tree traversals — first with ``updating_player = 0`` (P1), then
with ``updating_player = 1`` (P2). Regret and strategy accumulators are only
written at nodes where the acting player IS the updating player. This matches
Zinkevich's formulation 1:1 and extends cleanly to CFR+'s alternating update.

Chance handling
---------------
Kuhn's chance event (the 3-card deal) is a single root-level uniform draw
over 6 permutations. Rather than introducing a chance-node abstraction, the
1/6 deal probability is absorbed into the initial ``reach_opp`` passed to the
recursion. The counterfactual reach :math:`\\pi^\\sigma_{-i}(h)` formally
includes both the opponent's action probabilities and chance's probability,
so absorbing 1/6 into ``reach_opp`` makes the recursion match the paper's
definition directly — no extra weighting factor in the traversal itself.

Interface
---------
- :class:`InfosetData` — per-infoset mutable record of cumulative counterfactual
  regret (Neller & Lanctot 2013 Alg. 2 line 11; stored RAW, no positive-part
  clipping) and reach-weighted cumulative strategy (line 12).
- :class:`VanillaCFR` — the trainer. Public methods:
    - ``train(iterations)``
    - ``current_strategy(infoset_key)`` — regret matching on current cumulative
      regret (Eq.: Neller & Lanctot 2013 Alg. 1 / this repo's
      :func:`poker_ai.algorithms.regret_matching.regret_matching`)
    - ``average_strategy()`` — time-averaged strategy per infoset, normalised
    - ``game_value()`` — P1-perspective value of the average-strategy profile
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from poker_ai.algorithms.regret_matching import regret_matching
from poker_ai.games.kuhn import KuhnAction  # Kuhn-specific action constructor; Day 3 will generalize
from poker_ai.games.protocol import GameProtocol, StateProtocol


@dataclass
class InfosetData:
    """Per-infoset mutable record for Vanilla CFR.

    ``cumulative_regret`` is stored as raw (un-clipped) sums; the positive-part
    transformation happens only at strategy-computation time via
    :func:`regret_matching`. This keeps the data compatible with CFR+ variants
    that impose different clipping semantics on the stored table itself.
    """

    cumulative_regret: np.ndarray
    cumulative_strategy: np.ndarray


class VanillaCFR:
    """Tabular Vanilla CFR for Kuhn Poker (Zinkevich et al. 2007).

    One ``train`` iteration = one alternating cycle (both players updated once).
    """

    def __init__(self, game: GameProtocol, n_actions: int = 2) -> None:
        self.game = game
        self.n_actions = n_actions
        self.infosets: dict[str, InfosetData] = {}
        self.iteration: int = 0

    # ------------------------------------------------------------------ public

    def train(self, iterations: int) -> None:
        for _ in range(iterations):
            for updating_player in (0, 1):
                for deal in self.game.all_deals():
                    root = self.game.state_from_deal(deal)
                    # Initial counterfactual reach for opponent absorbs chance:
                    #   reach_opp = π_chance × π_{opp} = (1/6) × 1.0
                    self._cfr(
                        root,
                        updating_player=updating_player,
                        reach_i=1.0,
                        reach_opp=1.0 / len(self.game.all_deals()),
                    )
            self.iteration += 1

    def current_strategy(self, infoset_key: str) -> np.ndarray:
        """Regret matching on the current cumulative regret table.

        For a not-yet-visited infoset, the implicit cumulative regret is zero
        → :func:`regret_matching` returns the uniform distribution. This is
        the correct base case for CFR recursion's first touch of any node.
        """
        if infoset_key not in self.infosets:
            return np.full(self.n_actions, 1.0 / self.n_actions)
        return regret_matching(self.infosets[infoset_key].cumulative_regret)

    def average_strategy(self) -> dict[str, np.ndarray]:
        """Time-averaged strategy per infoset (Neller & Lanctot 2013 Alg. 2 line 12).

        Each entry is the reach-weighted cumulative strategy normalised to a
        probability distribution. Infosets whose cumulative strategy is all
        zero (i.e. never reached with positive probability by any iteration)
        fall back to uniform.
        """
        out: dict[str, np.ndarray] = {}
        for key, data in self.infosets.items():
            total = data.cumulative_strategy.sum()
            if total > 0.0:
                out[key] = data.cumulative_strategy / total
            else:
                out[key] = np.full(self.n_actions, 1.0 / self.n_actions)
        return out

    def game_value(self) -> float:
        """Expected value from Player 1's perspective under the average-strategy
        profile, averaged over the 6 Kuhn deals with uniform probability.

        Independent traversal (no regret/strategy accumulation) — kept separate
        from :meth:`train` to avoid intermingling evaluation with training.
        """
        avg = self.average_strategy()

        def sigma(key: str) -> np.ndarray:
            return avg.get(key, np.full(self.n_actions, 1.0 / self.n_actions))

        def expected_utility(state: StateProtocol) -> float:
            if state.is_terminal:
                # terminal_utility is already from P1's perspective.
                return self.game.terminal_utility(state)
            s = sigma(state.infoset_key)
            total = 0.0
            for a_idx in range(self.n_actions):
                child = state.next_state(KuhnAction(a_idx))
                total += s[a_idx] * expected_utility(child)
            return total

        deals = self.game.all_deals()
        weight = 1.0 / len(deals)
        total = 0.0
        for deal in deals:
            total += weight * expected_utility(self.game.state_from_deal(deal))
        return float(total)

    # ---------------------------------------------------------------- internal

    def _cfr(
        self,
        state: StateProtocol,
        updating_player: int,
        reach_i: float,
        reach_opp: float,
    ) -> float:
        """Counterfactual regret minimization recursion (Neller & Lanctot 2013 Alg. 2).

        Returns ``updating_player``'s expected utility from ``state`` onwards
        under the current strategy profile. The returned value is NOT weighted
        by reach probabilities; reach is used only for regret/strategy
        accumulation at the current infoset.

        Args:
            state: current game state.
            updating_player: 0 (P1) or 1 (P2); the player whose regret table
                gets updated during this traversal.
            reach_i: π^σ_{updating_player}(state), updating player's own
                cumulative action-probability product along the path.
            reach_opp: π^σ_{-updating_player}(state), product of the opponent's
                action probabilities AND chance probabilities along the path.
        """
        # --- Terminal base case: return u_i(leaf) for the updating player.
        # Zero-sum: P2's utility is -P1's utility.
        if state.is_terminal:
            u_p1 = self.game.terminal_utility(state)
            return u_p1 if updating_player == 0 else -u_p1

        # --- Lazy InfosetData initialisation on first visit.
        key = state.infoset_key
        if key not in self.infosets:
            self.infosets[key] = InfosetData(
                cumulative_regret=np.zeros(self.n_actions),
                cumulative_strategy=np.zeros(self.n_actions),
            )

        # --- Current strategy σ(I) via regret matching.
        strategy = self.current_strategy(key)
        acting_player = state.current_player

        # --- Recurse on each action to gather v_σ(I·a) for updating_player.
        action_values = np.zeros(self.n_actions)
        for a_idx in range(self.n_actions):
            next_state = state.next_state(KuhnAction(a_idx))
            if acting_player == updating_player:
                # updating player's own action scales π_i
                action_values[a_idx] = self._cfr(
                    next_state,
                    updating_player=updating_player,
                    reach_i=reach_i * strategy[a_idx],
                    reach_opp=reach_opp,
                )
            else:
                # opponent's action scales π_{-i}
                action_values[a_idx] = self._cfr(
                    next_state,
                    updating_player=updating_player,
                    reach_i=reach_i,
                    reach_opp=reach_opp * strategy[a_idx],
                )

        # --- Node value under σ from updating_player's perspective.
        node_value = float(strategy @ action_values)

        # --- Accumulate regret and strategy ONLY when the acting player IS
        # the updating player (alternating one-player traversal).
        if acting_player == updating_player:
            # Counterfactual regret: r(I, a) = π_{-i}(h) · (v_σ(I·a) - v_σ(I))
            # Neller & Lanctot 2013 Alg. 2, line 11. Stored RAW — clipping is
            # deferred to current_strategy() per Eq. (5).
            instantaneous_regret = reach_opp * (action_values - node_value)
            self.infosets[key].cumulative_regret += instantaneous_regret
            # Reach-weighted cumulative strategy: s(I, a) += π_i(h) · σ(I, a)
            # Neller & Lanctot 2013 Alg. 2, line 12.
            self.infosets[key].cumulative_strategy += reach_i * strategy

        return node_value
