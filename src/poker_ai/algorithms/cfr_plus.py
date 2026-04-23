"""CFR+ (Tammelin 2014) = regret clipping + linear averaging + alternating
updates, all three components.

Subclasses :class:`VanillaCFR` by overriding the two accumulation hooks
(``_update_regret`` and ``_update_strategy``) introduced in Phase 2 Day 5
C1. The tree traversal, infoset caching, legal-action handling, and
``train()`` loop are inherited unchanged. VanillaCFR already implements
alternating one-player traversal (the "A pattern"), which is the third
CFR+ component — so no further change is needed.

Mathematical differences from Vanilla CFR
-----------------------------------------
1. **Regret clipping** (storage-level positive-part):
   ``R⁺(I, a) ← max(0, R⁺(I, a) + r(I, a))``
   Vanilla stores raw regrets that may go negative; CFR+ clamps at each
   update so accumulated regret is always non-negative. This eliminates
   the "climb back from negative" delay when an action that was previously
   bad becomes favourable — the source of CFR+'s empirical speedup.

2. **Linear averaging** (iteration-weighted strategy sum):
   ``S(I, a) += t · π_i(h) · σ(I, a)``
   Vanilla weights every iteration equally (``π_i``-only); CFR+ weights
   iteration t by t itself. Since later iterations' strategies are closer
   to Nash, weighting them more heavily accelerates the time-averaged
   profile's convergence.

3. **Alternating updates**: already the Vanilla A-pattern (each ``train``
   iter does two traversals, one per updating player). No change.

CFR+ vs "Regret Matching+"
--------------------------
"Regret Matching+" (RM+) sometimes refers only to regret clipping. CFR+
is the full algorithm with all three components. Classes using only
clipping without linear averaging should be named ``RegretMatchingPlus``
or similar to avoid confusion.

Reference: Tammelin, Oskari (2014). "Solving Large Imperfect Information
Games Using CFR+." arXiv:1407.5042.
"""

from __future__ import annotations

import numpy as np

from poker_ai.algorithms.vanilla_cfr import VanillaCFR
from poker_ai.games.protocol import GameProtocol


class CFRPlus(VanillaCFR):
    """CFR+ (Tammelin 2014) = regret clipping + linear averaging +
    alternating updates, all three components.

    Drop-in replacement for :class:`VanillaCFR` — same constructor
    signature, same public API. Use on any game satisfying GameProtocol.
    Empirically converges 10-100× faster than Vanilla CFR on Leduc
    Hold'em (Tammelin 2014 Figure 2).

    Synchronous regret updates (Phase 2 Day 5 audit fix)
    ----------------------------------------------------
    Tammelin's CFR+ requires ``σ^t`` to be fixed throughout iteration t —
    i.e., R⁺ must not change mid-iteration. VanillaCFR's externalized-chance
    traversal (deal loop outside ``_cfr``) revisits the same infoset once
    per reaching deal, and mid-traversal R⁺ updates cause σ^t to drift
    across visits within the same iteration. For Vanilla this is harmless
    (uniform weighting averages drift out); for CFR+ linear averaging
    amplifies drift and dramatically degrades convergence.

    Fix: override :meth:`train` to buffer regret deltas during a player's
    traversal and apply them (with clipping) only after the full traversal.
    ``cumulative_strategy`` updates still happen immediately — they use the
    pre-traversal (consistent) ``strategy`` value cached at each ``_cfr``
    entry point. Since ``current_strategy`` reads ``cumulative_regret``
    which is now unchanged during the traversal, strategies are consistent
    per infoset within the iter, matching Tammelin's σ^t fixed-per-iter
    assumption.
    """

    def __init__(self, game: GameProtocol, n_actions: int = 2) -> None:
        super().__init__(game, n_actions=n_actions)
        self._pending_regret: dict[str, np.ndarray] = {}

    def train(self, iterations: int) -> None:
        """Train with synchronous (end-of-traversal) regret updates.

        Inside a player's traversal, :meth:`_update_regret` buffers deltas
        in ``self._pending_regret`` instead of mutating ``cumulative_regret``.
        After the full deal loop for this player, we flush the buffer with
        the CFR+ positive-part clipping (``R⁺ ← max(0, R⁺ + Σ r)``).
        """
        n_deals = len(self.game.all_deals())
        for _ in range(iterations):
            for updating_player in (0, 1):
                for deal in self.game.all_deals():
                    root = self.game.state_from_deal(deal)
                    self._cfr(
                        root,
                        updating_player=updating_player,
                        reach_i=1.0,
                        reach_opp=1.0 / n_deals,
                    )
                self._flush_pending_regret()
            self.iteration += 1

    def _flush_pending_regret(self) -> None:
        """Apply buffered regret deltas with positive-part clipping, then
        clear the buffer. Called at the end of each updating-player's
        full traversal.
        """
        for key, delta in self._pending_regret.items():
            updated = self.infosets[key].cumulative_regret + delta
            self.infosets[key].cumulative_regret = np.maximum(0.0, updated)
        self._pending_regret.clear()

    def _update_regret(self, key: str, instantaneous_regret: np.ndarray) -> None:
        """CFR+ regret accumulation: BUFFER instead of apply-in-place.

        Deltas are summed into ``self._pending_regret[key]``; the actual
        ``R⁺`` update (with clipping) happens in :meth:`_flush_pending_regret`
        at the end of the player's traversal. This keeps ``cumulative_regret``
        unchanged throughout the traversal so ``current_strategy(key)``
        returns the same σ^t at every visit to ``key`` within the iter.

        Illegal slots are already zeroed in ``instantaneous_regret`` by the
        ``_cfr`` caller's mask multiplication, so the pending buffer also
        keeps them at 0.
        """
        if key not in self._pending_regret:
            self._pending_regret[key] = np.zeros_like(instantaneous_regret)
        self._pending_regret[key] += instantaneous_regret

    def _update_strategy(
        self, key: str, reach_i: float, strategy: np.ndarray
    ) -> None:
        """CFR+ linear-averaged strategy accumulation.

        ``S(I, a) += t · π_i(h) · σ(I, a)`` where ``t`` is the current
        iteration number (1-indexed).

        The ``train()`` loop increments ``self.iteration`` AFTER each
        full-cycle traversal, so during iteration t's traversal
        ``self.iteration == t - 1``. We add 1 to get the 1-indexed weight
        that matches Tammelin's formulation. ``strategy`` is the value
        cached at ``_cfr`` entry — consistent per infoset within the iter
        thanks to buffered regret updates (see :meth:`train` docstring).
        """
        iter_weight = self.iteration + 1
        self.infosets[key].cumulative_strategy += (
            iter_weight * reach_i * strategy
        )
