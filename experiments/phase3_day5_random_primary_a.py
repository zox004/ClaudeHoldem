"""Phase 3 Day 5 Step 3 (option I) — Random AdvantageNet Primary A floor.

Measures the Pearson Primary A metric for an *untrained* AdvantageNet over
multiple weight-init seeds. The result quantifies the metric's natural
noise band when the network has zero information about the regret target,
so subsequent comparisons (Day 4 trained 0.247, Day 5 D' σ_seed ablation)
can be read against a calibrated lower bound.

Procedure:
    1. Train Vanilla CFR + CFR+ once at T=500 (deterministic).
    2. For each of N init seeds, instantiate DeepCFR (no train()) and
       compute_correlations against the same Vanilla/CFR+ references.
    3. Report mean ± std for Kuhn and Leduc.

Run:
    uv run python -m experiments.phase3_day5_random_primary_a
"""

from __future__ import annotations

import logging

import numpy as np
import torch

from poker_ai.algorithms.cfr_plus import CFRPlus
from poker_ai.algorithms.deep_cfr import DeepCFR
from poker_ai.algorithms.vanilla_cfr import VanillaCFR
from poker_ai.eval.deep_cfr_correlation import compute_correlations
from poker_ai.games.kuhn import KuhnPoker
from poker_ai.games.leduc import LeducPoker
from poker_ai.games.protocol import GameProtocol

logging.basicConfig(format="%(message)s", level=logging.INFO)
log = logging.getLogger(__name__)


def measure_random_primary_a(
    game: GameProtocol,
    n_actions: int,
    encoding_dim: int,
    init_seeds: list[int],
    reference_T: int = 500,
    reference_seed: int = 42,
) -> np.ndarray:
    """Returns array of Primary A values, one per init_seed."""
    torch.manual_seed(reference_seed)
    np.random.seed(reference_seed)
    vanilla = VanillaCFR(game=game, n_actions=n_actions)
    vanilla.train(reference_T)
    cfp = CFRPlus(game=game, n_actions=n_actions)
    cfp.train(reference_T)

    primaries: list[float] = []
    for s in init_seeds:
        torch.manual_seed(s)
        np.random.seed(s)
        deep = DeepCFR(
            game=game,
            n_actions=n_actions,
            encoding_dim=encoding_dim,
            traversals_per_iter=10,
            batch_size=8,
            advantage_epochs=1,
            strategy_epochs=1,
            seed=s,
        )
        # NO train() — measure the random-init floor.
        rep = compute_correlations(deep, vanilla, cfp, game)
        primaries.append(float(rep.primary_a_advantage_vs_vanilla))
    return np.asarray(primaries, dtype=np.float64)


def main() -> None:
    seeds = [42, 43, 44, 45, 46]
    log.info("Random AdvantageNet Primary A floor — %d init seeds %s", len(seeds), seeds)
    log.info("Reference: Vanilla + CFR+ @ T=500, seed=42 (deterministic)")
    log.info("")

    kuhn_primaries = measure_random_primary_a(KuhnPoker(), 2, 6, seeds)
    log.info("Kuhn  Primary A samples: %s", [round(x, 4) for x in kuhn_primaries])
    log.info(
        "Kuhn  mean=%.4f  std=%.4f  min=%.4f  max=%.4f",
        kuhn_primaries.mean(), kuhn_primaries.std(ddof=1),
        kuhn_primaries.min(), kuhn_primaries.max(),
    )
    log.info("")

    leduc_primaries = measure_random_primary_a(LeducPoker(), 3, 13, seeds)
    log.info("Leduc Primary A samples: %s", [round(x, 4) for x in leduc_primaries])
    log.info(
        "Leduc mean=%.4f  std=%.4f  min=%.4f  max=%.4f",
        leduc_primaries.mean(), leduc_primaries.std(ddof=1),
        leduc_primaries.min(), leduc_primaries.max(),
    )
    log.info("")

    # Calibration vs Day 4 trained baseline.
    day4_leduc_trained = 0.2470
    delta = day4_leduc_trained - leduc_primaries.mean()
    sigma_init = leduc_primaries.std(ddof=1)
    effect_size = delta / sigma_init if sigma_init > 0 else float("inf")
    log.info("Day 4 Leduc trained Primary A = %.4f", day4_leduc_trained)
    log.info("Day 4 - random_floor       Δ  = %.4f", delta)
    log.info("Effect size (Δ / σ_init)      = %.2f", effect_size)


if __name__ == "__main__":
    main()
