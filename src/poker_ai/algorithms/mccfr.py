"""External Sampling Monte Carlo CFR (Lanctot 2009 PhD thesis §3.4).

Monte Carlo variant of CFR that samples non-updating player actions and
chance outcomes instead of enumerating them, achieving dramatic wall-clock
speedup on larger games (Leduc, NLHE) while maintaining an unbiased
regret estimator (Lanctot 2009 Proposition 4).

External Sampling specifics
---------------------------
Per iteration, for each updating player p ∈ {0, 1}:
- Chance: sample a single deal (uniform over all_deals)
- Non-updating player -p: at each of their decision nodes, sample ONE
  action from ε-smoothed current strategy. Importance weight
  ``1/sample_prob`` is accumulated into a running `weight` parameter.
- Updating player p: enumerate ALL legal actions (same as Vanilla).

Regret update at p's infosets:
  instant_regret(I, a) = action_values[a] - node_value
where action_values already carry the accumulated importance weight via
the recursive `weight` parameter, making the estimator unbiased.

ε-exploration (Lanctot §3.2) is REQUIRED for numerical stability. Without
ε-smoothing, low-probability actions in the opponent's strategy cause
importance weights to explode (1/prob → ∞), inflating variance.
Default ε=0.05 gives importance weight bound `|A|/ε` per sampled node.

CFR+ extensions (MCCFR+) are intentionally omitted in this Phase 2
implementation — pure MCCFR first, CFR+ combination deferred.

Reference: Lanctot, Marc (2009). "Monte Carlo Sampling for Regret
Minimization in Extensive Games". PhD thesis, University of Alberta.
"""

from __future__ import annotations

import numpy as np

from poker_ai.algorithms.regret_matching import regret_matching
from poker_ai.algorithms.vanilla_cfr import InfosetData
from poker_ai.games.protocol import GameProtocol, StateProtocol


class MCCFRExternalSampling:
    """External Sampling MCCFR (Lanctot 2009 §3.4).

    Independent class — does NOT inherit from VanillaCFR. MCCFR's
    sampling-based traversal is fundamentally different from Vanilla's
    full-tree enumeration, so the hook-override pattern used for CFR+
    would be forced. Reuses :class:`InfosetData` dataclass via import.

    Parameters:
        game: GameProtocol instance (Kuhn, Leduc, ...).
        n_actions: action space size (Kuhn=2, Leduc=3).
        rng: ``np.random.Generator`` for reproducibility. Callers must
            supply one (e.g., ``np.random.default_rng(seed)``) — MCCFR
            does not create its own to force explicit seed management.
        epsilon: ε-exploration parameter (Lanctot §3.2). Default 0.05
            per Lanctot's recommendation. Larger ε → lower variance
            but slower convergence (more uniform sampling).
    """

    def __init__(
        self,
        game: GameProtocol,
        n_actions: int,
        rng: np.random.Generator,
        epsilon: float = 0.05,
    ) -> None:
        self.game = game
        self.n_actions = n_actions
        self.rng = rng
        self.epsilon = epsilon
        self.infosets: dict[str, InfosetData] = {}
        self.iteration: int = 0

    # ------------------------------------------------------------------ public

    def train(self, iterations: int) -> None:
        """Each iteration: alternating-player traversal from a single sampled deal.

        For each updating player p:
          1. Sample chance (one deal)
          2. Traverse with external sampling of non-updating player actions
          3. Regret/strategy updated at p's infosets via the recursion
        """
        deals = self.game.all_deals()
        n_deals = len(deals)
        for _ in range(iterations):
            for updating_player in (0, 1):
                deal_idx = int(self.rng.integers(0, n_deals))
                deal = deals[deal_idx]
                root = self.game.state_from_deal(deal)
                self._traverse(
                    root,
                    updating_player=updating_player,
                    reach_i=1.0,
                    weight=1.0,
                )
            self.iteration += 1

    def current_strategy(self, infoset_key: str) -> np.ndarray:
        """Regret matching with legal mask (same API as VanillaCFR)."""
        if infoset_key not in self.infosets:
            uniform: np.ndarray = np.full(self.n_actions, 1.0 / self.n_actions)
            return uniform
        data = self.infosets[infoset_key]
        return regret_matching(data.cumulative_regret, legal_mask=data.legal_mask)

    def average_strategy(self) -> dict[str, np.ndarray]:
        """Normalized cumulative_strategy per infoset (same as VanillaCFR)."""
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
        """Full-tree evaluation of avg-strategy profile (deterministic)."""
        avg = self.average_strategy()

        def sigma(state: StateProtocol) -> np.ndarray:
            key = state.infoset_key
            if key in avg:
                return avg[key]
            mask_f = state.legal_action_mask().astype(np.float64)
            fallback: np.ndarray = mask_f / mask_f.sum()
            return fallback

        def expected_utility(state: StateProtocol) -> float:
            if state.is_terminal:
                return self.game.terminal_utility(state)
            s = sigma(state)
            total = 0.0
            for a in state.legal_actions():
                child = state.next_state(a)
                total += float(s[int(a)]) * expected_utility(child)
            return total

        deals = self.game.all_deals()
        weight = 1.0 / len(deals)
        grand = 0.0
        for deal in deals:
            grand += weight * expected_utility(self.game.state_from_deal(deal))
        return float(grand)

    # ---------------------------------------------------------------- internal

    def _traverse(
        self,
        state: StateProtocol,
        updating_player: int,
        reach_i: float,
        weight: float,
    ) -> float:
        """External-sampling CFR recursion.

        Returns the importance-weighted utility-from-below at ``state``
        from ``updating_player``'s perspective.

        Parameters:
            reach_i: π_i^σ(h) — updating player's reach probability.
                Accumulated as we traverse updating player's actions
                (scaled by σ(a) at each own-action branching). Used in
                the ``cumulative_strategy`` update.
            weight: importance-sampling correction ``Π σ(a) / q(a)`` over
                non-updating player's sampled actions along the path.
                Each opponent sample node contributes a ``σ(a)/q(a)``
                factor, where ``σ`` is the true strategy and ``q`` is
                the ε-smoothed sampling distribution. Keeps the Lanctot
                Prop 4 regret estimator unbiased.
        """
        if state.is_terminal:
            u_p1 = self.game.terminal_utility(state)
            u = u_p1 if updating_player == 0 else -u_p1
            return weight * u

        key = state.infoset_key
        if key not in self.infosets:
            self.infosets[key] = InfosetData(
                cumulative_regret=np.zeros(self.n_actions),
                cumulative_strategy=np.zeros(self.n_actions),
                legal_mask=state.legal_action_mask(),
            )

        strategy = self.current_strategy(key)
        acting_player = state.current_player

        if acting_player == updating_player:
            # Enumerate all legal actions (no sampling on updating side).
            action_values = np.zeros(self.n_actions)
            for a in state.legal_actions():
                a_idx = int(a)
                next_state = state.next_state(a)
                action_values[a_idx] = self._traverse(
                    next_state,
                    updating_player=updating_player,
                    reach_i=reach_i * float(strategy[a_idx]),
                    weight=weight,
                )
            node_value = float(strategy @ action_values)

            # Regret update (mask enforces illegal-slot zeroing). The
            # ``weight`` (σ/q product from opponent sampling) is already
            # baked into action_values via the recursion; the regret
            # difference (action_values[a] - node_value) inherits this
            # correction.
            mask_f = self.infosets[key].legal_mask.astype(np.float64)
            instant_regret = (action_values - node_value) * mask_f
            self.infosets[key].cumulative_regret += instant_regret

            # Strategy update: S += π_i^σ(h) · σ^t(I, a).
            self.infosets[key].cumulative_strategy += reach_i * strategy

            return node_value

        # --- Non-updating player: ε-smoothed single-action sample.
        smoothed = self._epsilon_smoothed(
            strategy, self.infosets[key].legal_mask
        )
        sampled_idx = int(self.rng.choice(self.n_actions, p=smoothed))
        sampled_action = next(
            a for a in state.legal_actions() if int(a) == sampled_idx
        )
        # Importance weight correction: ``σ(a) / q(a)``.
        # σ(a) can be 0 for actions only reached via ε-exploration —
        # in that case the sample contributes 0 to regret (correct
        # under true-σ measure; ε-exploration sampled it purely for
        # discovery, not contribution).
        true_prob = float(strategy[sampled_idx])
        sample_prob = float(smoothed[sampled_idx])
        weight_factor = true_prob / sample_prob if sample_prob > 0.0 else 0.0
        return self._traverse(
            state.next_state(sampled_action),
            updating_player=updating_player,
            reach_i=reach_i,
            weight=weight * weight_factor,
        )

    def _epsilon_smoothed(
        self, strategy: np.ndarray, legal_mask: np.ndarray
    ) -> np.ndarray:
        """``(1-ε)·σ + ε·uniform_over_legal`` — used ONLY for sampling probs.

        Regret and strategy accumulators keep the original (un-smoothed)
        ``strategy``. Smoothing applies solely to the sampling distribution
        so that ε-bounded sample_prob keeps importance weights finite
        (Lanctot §3.2). Illegal slots stay exactly 0.
        """
        mask_f = legal_mask.astype(np.float64)
        n_legal = mask_f.sum()
        uniform_legal = mask_f / n_legal
        smoothed = (1.0 - self.epsilon) * strategy + self.epsilon * uniform_legal
        # Mask any residual probability on illegal slots (defensive; strategy
        # should already be 0 on illegal via regret_matching's mask path).
        smoothed = smoothed * mask_f
        total = smoothed.sum()
        result: np.ndarray = smoothed / total
        return result
