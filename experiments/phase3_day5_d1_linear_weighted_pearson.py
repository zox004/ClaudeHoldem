"""Phase 3 Day 5 Step 4 (d1) — Vanilla linear-weighted vs uniform Pearson.

Tests hypothesis (d) / (f): does the buffer's linear-CFR weighting
(iter_weight=t in :class:`DeepCFR`) by itself bias the Pearson reference
away from Vanilla's uniform-weighted cumulative regret?

If Pearson(R_uniform, R_linear) is far below 1.0, then a perfectly trained
network can never reach Pearson 1.0 against the (uniform-weighted) Vanilla
reference — part of the Day 4 0.25 floor would be a metric-mismatch
artifact rather than a genuine network limit.

Cutoffs (mentor sign-off, Day 5 brainstorm):
    > 0.95  → metric mismatch negligible; (d)/(f) rejected
    0.7-0.95 → partial mismatch; floor contains 0.02–0.05 artifact
    0.5-0.7  → strong mismatch; floor contains 0.05–0.10 artifact
    < 0.5    → very strong mismatch; metric redefinition warranted

Game-value sanity is reported as a free byproduct: VanillaCFR with
linear-weight tracking still uses ``cumulative_regret`` (uniform) for
strategy decisions, so its game value should be the standard Vanilla
value at T=500 — any deviation from Nash signals a tracking-side bug.

Run:
    uv run python -m experiments.phase3_day5_d1_linear_weighted_pearson
"""

from __future__ import annotations

import logging

import numpy as np
import torch

from poker_ai.algorithms.vanilla_cfr import VanillaCFR
from poker_ai.eval.exploitability import exploitability
from poker_ai.games.kuhn import KuhnPoker
from poker_ai.games.leduc import LeducPoker
from poker_ai.games.protocol import GameProtocol

logging.basicConfig(format="%(message)s", level=logging.INFO)
log = logging.getLogger(__name__)

def _flat_legal_pairs(
    cfr: VanillaCFR, n_actions: int
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (uniform_R, linear_R) flat arrays over (infoset, legal_action)."""
    uniform: list[float] = []
    linear: list[float] = []
    for key in sorted(cfr.infosets):
        data = cfr.infosets[key]
        for a in range(n_actions):
            if not data.legal_mask[a]:
                continue
            uniform.append(float(data.cumulative_regret[a]))
            linear.append(float(data.cumulative_regret_linear[a]))
    return np.asarray(uniform, dtype=np.float64), np.asarray(linear, dtype=np.float64)


def _exploitability_at(game: GameProtocol, cfr: VanillaCFR) -> float:
    """Vanilla σ̄ exploitability — Nash-convergence proxy for game-value
    sanity (the linear-weighted shadow shouldn't perturb σ̄)."""
    sigma_bar = cfr.average_strategy()
    return float(exploitability(game, sigma_bar))


def measure_d1(
    game: GameProtocol,
    n_actions: int,
    iterations: int = 500,
    seed: int = 42,
) -> dict[str, float]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    cfr = VanillaCFR(game=game, n_actions=n_actions, track_linear_weighted=True)
    cfr.train(iterations)

    uniform, linear = _flat_legal_pairs(cfr, n_actions)
    pearson = float(np.corrcoef(uniform, linear)[0, 1])

    expl = _exploitability_at(game, cfr)

    return {
        "pearson_uniform_vs_linear": pearson,
        "n_pairs": float(len(uniform)),
        "uniform_abs_mean": float(np.abs(uniform).mean()),
        "linear_abs_mean": float(np.abs(linear).mean()),
        "uniform_std": float(uniform.std(ddof=0)),
        "linear_std": float(linear.std(ddof=0)),
        "exploitability_T_iter": expl,
    }


def _interpret_pearson(p: float) -> str:
    if p > 0.95:
        return "metric mismatch NEGLIGIBLE — (d)/(f) rejected"
    if p > 0.70:
        return "PARTIAL mismatch — floor contains 0.02-0.05 artifact"
    if p > 0.50:
        return "STRONG mismatch — floor contains 0.05-0.10 artifact"
    return "VERY STRONG mismatch — metric redefinition warranted"


def main() -> None:
    log.info("=== Phase 3 Day 5 Step 4 (d1) — Vanilla linear vs uniform ===")
    log.info("seed=42, T=500. Linear-weighted shadow accumulation; strategy")
    log.info("decisions still use uniform cumulative_regret (no convergence")
    log.info("perturbation).")
    log.info("")

    for name, game, n_act in (
        ("Kuhn",  KuhnPoker(),  2),
        ("Leduc", LeducPoker(), 3),
    ):
        log.info("--- %s ---", name)
        r = measure_d1(game, n_act)
        log.info("n_pairs                  = %d", int(r["n_pairs"]))
        log.info("uniform |R|_mean         = %.4f", r["uniform_abs_mean"])
        log.info("linear  |R|_mean         = %.4f  (T(T+1)/2 ≈ %.0fx)",
                 r["linear_abs_mean"], 500 * 501 / 2)
        log.info("Pearson(uniform, linear) = %.4f", r["pearson_uniform_vs_linear"])
        log.info("→ %s", _interpret_pearson(r["pearson_uniform_vs_linear"]))
        log.info(
            "exploitability @ T=500   = %.4f chips  (sanity: σ̄ unaffected)",
            r["exploitability_T_iter"],
        )
        log.info("")


if __name__ == "__main__":
    main()
