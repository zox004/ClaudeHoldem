"""Phase 4 M2.4 — MCCFR sanity smoke on AbstractedHUNLGame.

End-to-end M2 validation: Phase 2's MCCFR algorithm runs unchanged on
the new AbstractedHUNLGame wrapper, leveraging GameProtocol structural
typing. The smoke is preflop-leaning by virtue of the small T budget;
preflop-only mode with terminal-after-preflop is M3 work, not M2.

Goal: prove that MCCFR + AbstractedHUNLGame integrates without errors
and produces a well-formed strategy. Numeric exploitability is **not**
checked here — measuring exploitability on a 10^14-leaf game tree
requires either (a) postflop abstraction (M3) or (b) cloud burst, both
out of scope for M2.
"""

from __future__ import annotations

import numpy as np

from poker_ai.algorithms.mccfr import MCCFRExternalSampling
from poker_ai.games.hunl_abstraction import AbstractedHUNLGame


class TestMCCFRRunsOnAbstractedHUNL:
    def test_mccfr_100_iterations_completes(self) -> None:
        """100 MCCFR iterations on AbstractedHUNLGame must complete
        without raising. Uses small abstractor (n_trials=300) for
        speed; the bucket map is deterministic at the test seed."""
        game = AbstractedHUNLGame(n_buckets=50, n_trials=300, seed=42)
        rng = np.random.default_rng(seed=42)
        trainer = MCCFRExternalSampling(
            game=game, n_actions=game.NUM_ACTIONS, rng=rng
        )
        # 100 iterations × 2 players' traversals × variable tree depth.
        trainer.train(100)
        assert trainer.iteration == 100

    def test_average_strategy_has_entries(self) -> None:
        """After training, the trainer's average_strategy() returns a
        non-empty dict keyed by abstracted infoset_keys — proves the
        wrapping pattern actually populated the strategy table."""
        game = AbstractedHUNLGame(n_buckets=50, n_trials=300, seed=42)
        rng = np.random.default_rng(seed=42)
        trainer = MCCFRExternalSampling(
            game=game, n_actions=game.NUM_ACTIONS, rng=rng
        )
        trainer.train(50)
        avg = trainer.average_strategy()
        assert len(avg) > 0
        # Each key starts with "<bucket>|<round>:..."
        sample_key = next(iter(avg))
        prefix = sample_key.split("|")[0]
        assert prefix.isdigit()
        # Each value is a probability simplex over NUM_ACTIONS.
        sample_strat = avg[sample_key]
        assert sample_strat.shape == (game.NUM_ACTIONS,)
        assert np.allclose(sample_strat.sum(), 1.0, atol=1e-6)

    def test_seed_reproducibility(self) -> None:
        """Same seed → same final strategy keys."""
        def _run() -> set[str]:
            game = AbstractedHUNLGame(n_buckets=50, n_trials=300, seed=42)
            rng = np.random.default_rng(42)
            trainer = MCCFRExternalSampling(
                game=game, n_actions=game.NUM_ACTIONS, rng=rng
            )
            trainer.train(30)
            return set(trainer.average_strategy().keys())
        keys_a = _run()
        keys_b = _run()
        assert keys_a == keys_b
