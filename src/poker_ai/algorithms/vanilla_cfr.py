"""Vanilla Counterfactual Regret Minimization (CFR) for 2-player zero-sum games.

Tabular implementation following Zinkevich et al. 2007, "Regret Minimization in
Games with Incomplete Information"; pseudocode structure mirrors Neller &
Lanctot 2013, Algorithm 2.

Game-agnostic via :class:`poker_ai.games.protocol.GameProtocol` (Phase 2 Day 3
refactor). Works on any game that satisfies the Protocol — Kuhn (Phase 1) and
Leduc (Phase 2) share the same trainer class.

Traversal pattern
-----------------
**Alternating one-player traversal** (the "A pattern"): one logical iteration
performs two tree traversals — first with ``updating_player = 0`` (P1), then
with ``updating_player = 1`` (P2). Regret and strategy accumulators are only
written at nodes where the acting player IS the updating player. This matches
Zinkevich's formulation 1:1 and extends cleanly to CFR+'s alternating update.

Chance handling
---------------
The chance event (card deal) is a single root-level uniform draw over all
deals (Kuhn: 6 permutations, Leduc: 120 permutations). Rather than introducing
a chance-node abstraction, the ``1/n_deals`` probability is absorbed into the
initial ``reach_opp`` passed to the recursion. The counterfactual reach
:math:`\\pi^\\sigma_{-i}(h)` formally includes both the opponent's action
probabilities and chance's probability, so absorbing ``1/n_deals`` into
``reach_opp`` makes the recursion match the paper's definition directly.

Legal-action handling
---------------------
For games with state-dependent legality (Leduc's FOLD illegal at
``bets_this_round=0``, RAISE illegal at cap), the CFR loop iterates
``state.legal_actions()`` only, and strategies are produced by
``regret_matching(cumulative_regret, legal_mask=state.legal_action_mask())``.
The ``legal_mask`` is cached per-infoset inside :class:`InfosetData` on first
visit (all states in one infoset share legal actions by the perfect-recall
definition). Illegal slots in ``action_values``, ``strategy``, and
``cumulative_regret`` remain exactly zero throughout training.

Interface
---------
- :class:`InfosetData` — per-infoset mutable record of cumulative counterfactual
  regret (Neller & Lanctot 2013 Alg. 2 line 11; stored RAW, no positive-part
  clipping) and reach-weighted cumulative strategy (line 12), plus the
  cached ``legal_mask`` for this infoset.
- :class:`VanillaCFR` — the trainer. Public methods:
    - ``train(iterations)``
    - ``current_strategy(infoset_key)`` — regret matching on current cumulative
      regret with cached legal mask applied (illegal slots → 0)
    - ``average_strategy()`` — time-averaged strategy per infoset, normalised
    - ``game_value()`` — P1-perspective value of the average-strategy profile
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from poker_ai.algorithms.regret_matching import regret_matching
from poker_ai.games.protocol import GameProtocol, StateProtocol


@dataclass
class InfosetData:
    """Per-infoset mutable record for Vanilla CFR.

    ``cumulative_regret`` is stored as raw (un-clipped) sums; the positive-part
    transformation happens only at strategy-computation time via
    :func:`regret_matching`. This keeps the data compatible with CFR+ variants
    that impose different clipping semantics on the stored table itself.

    ``legal_mask`` is captured from the first visited state of this infoset
    (all states in the same infoset share legal actions by perfect-recall
    definition). Used to keep illegal slots at exactly 0 in strategy outputs
    and in regret accumulation.
    """

    cumulative_regret: np.ndarray
    cumulative_strategy: np.ndarray
    legal_mask: np.ndarray


class VanillaCFR:
    """Tabular Vanilla CFR (Zinkevich et al. 2007), game-agnostic via Protocol.

    One ``train`` iteration = one alternating cycle (both players updated once).
    """

    def __init__(self, game: GameProtocol, n_actions: int = 2) -> None:
        self.game = game
        self.n_actions = n_actions
        self.infosets: dict[str, InfosetData] = {}
        self.iteration: int = 0

    # ------------------------------------------------------------------ public

    def train(self, iterations: int) -> None:
        n_deals = len(self.game.all_deals())
        for _ in range(iterations):
            for updating_player in (0, 1):
                for deal in self.game.all_deals():
                    root = self.game.state_from_deal(deal)
                    # Initial counterfactual reach for opponent absorbs chance:
                    #   reach_opp = π_chance × π_{opp} = (1/n_deals) × 1.0
                    self._cfr(
                        root,
                        updating_player=updating_player,
                        reach_i=1.0,
                        reach_opp=1.0 / n_deals,
                    )
            self.iteration += 1

    def current_strategy(self, infoset_key: str) -> np.ndarray:
        """Regret matching on the current cumulative regret table.

        Uses the cached ``legal_mask`` so illegal slots get exactly 0
        probability. For an unvisited infoset, returns uniform over all
        actions; CFR's recursion always visits an infoset (initialising
        its ``legal_mask``) before calling this, so the uniform-over-all
        fallback is only reachable by external callers probing unknown keys.
        """
        if infoset_key not in self.infosets:
            uniform: np.ndarray = np.full(self.n_actions, 1.0 / self.n_actions)
            return uniform
        data = self.infosets[infoset_key]
        return regret_matching(data.cumulative_regret, legal_mask=data.legal_mask)

    def average_strategy(self) -> dict[str, np.ndarray]:
        """Time-averaged strategy per infoset (Neller & Lanctot 2013 Alg. 2 line 12).

        Each entry is the reach-weighted cumulative strategy normalised to a
        probability distribution. Infosets whose cumulative strategy is all
        zero (never reached with positive probability) fall back to uniform
        over LEGAL actions only — illegal slots remain 0.
        """
        out: dict[str, np.ndarray] = {}
        for key, data in self.infosets.items():
            total = data.cumulative_strategy.sum()
            if total > 0.0:
                normalized: np.ndarray = data.cumulative_strategy / total
                out[key] = normalized
            else:
                mask_f = data.legal_mask.astype(np.float64)
                fallback: np.ndarray = mask_f / mask_f.sum()
                out[key] = fallback
        return out

    def game_value(self) -> float:
        """Expected value from Player 1's perspective under the average-strategy
        profile, averaged over all deals with uniform probability.

        Independent traversal (no regret/strategy accumulation) — kept separate
        from :meth:`train` to avoid intermingling evaluation with training.
        """
        avg = self.average_strategy()

        def sigma(state: StateProtocol) -> np.ndarray:
            """Strategy at the given state — avg-strategy if visited, else
            uniform over legal actions."""
            key = state.infoset_key
            if key in avg:
                return avg[key]
            mask_f = state.legal_action_mask().astype(np.float64)
            fallback: np.ndarray = mask_f / mask_f.sum()
            return fallback

        def expected_utility(state: StateProtocol) -> float:
            if state.is_terminal:
                # terminal_utility is already from P1's perspective.
                return self.game.terminal_utility(state)
            s = sigma(state)
            total = 0.0
            for a in state.legal_actions():
                child = state.next_state(a)
                total += float(s[int(a)]) * expected_utility(child)
            return total

        deals = self.game.all_deals()
        weight = 1.0 / len(deals)
        grand_total = 0.0
        for deal in deals:
            grand_total += weight * expected_utility(self.game.state_from_deal(deal))
        return float(grand_total)

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
            state: current game state (any :class:`StateProtocol`).
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

        # --- Lazy InfosetData initialisation on first visit, with legal_mask.
        key = state.infoset_key
        if key not in self.infosets:
            self.infosets[key] = InfosetData(
                cumulative_regret=np.zeros(self.n_actions),
                cumulative_strategy=np.zeros(self.n_actions),
                legal_mask=state.legal_action_mask(),
            )

        # --- Current strategy σ(I) via regret matching (legal-masked).
        strategy = self.current_strategy(key)
        acting_player = state.current_player

        # --- Recurse on each LEGAL action to gather v_σ(I·a). Illegal slots
        # in action_values stay 0 (never written) and receive 0 strategy
        # mass, so they contribute nothing to node_value.
        action_values = np.zeros(self.n_actions)
        for a in state.legal_actions():
            a_idx = int(a)
            next_state = state.next_state(a)
            if acting_player == updating_player:
                # updating player's own action scales π_i
                action_values[a_idx] = self._cfr(
                    next_state,
                    updating_player=updating_player,
                    reach_i=reach_i * float(strategy[a_idx]),
                    reach_opp=reach_opp,
                )
            else:
                # opponent's action scales π_{-i}
                action_values[a_idx] = self._cfr(
                    next_state,
                    updating_player=updating_player,
                    reach_i=reach_i,
                    reach_opp=reach_opp * float(strategy[a_idx]),
                )

        # --- Node value under σ from updating_player's perspective.
        node_value = float(strategy @ action_values)

        # --- Accumulate regret and strategy ONLY when the acting player IS
        # the updating player (alternating one-player traversal).
        if acting_player == updating_player:
            # Counterfactual regret: r(I, a) = π_{-i}(h) · (v_σ(I·a) - v_σ(I))
            # Neller & Lanctot 2013 Alg. 2, line 11. Mask illegal slots to
            # prevent negative-regret drift on them (Kuhn no-op; Leduc active).
            mask_f = self.infosets[key].legal_mask.astype(np.float64)
            instantaneous_regret = reach_opp * (action_values - node_value) * mask_f
            self._update_regret(key, instantaneous_regret)
            # Reach-weighted cumulative strategy: Neller & Lanctot 2013 Alg. 2
            # line 12. strategy[illegal] = 0 already (from masked regret_matching).
            self._update_strategy(key, reach_i, strategy)

        return node_value

    # ------------------------------------------------ update hooks (Day 5)

    def _update_regret(self, key: str, instantaneous_regret: np.ndarray) -> None:
        """Vanilla CFR regret accumulation: raw storage (no clipping).

        Positive-part clipping is deferred to strategy-computation time via
        :func:`regret_matching`. CFR+ (Tammelin 2014) overrides this hook
        to clip at storage time instead.
        """
        self.infosets[key].cumulative_regret += instantaneous_regret

    def _update_strategy(
        self, key: str, reach_i: float, strategy: np.ndarray
    ) -> None:
        """Vanilla CFR cumulative strategy: unweighted reach-probability sum.

        CFR+ overrides this hook to apply linear averaging (weight by iter t).
        """
        self.infosets[key].cumulative_strategy += reach_i * strategy
