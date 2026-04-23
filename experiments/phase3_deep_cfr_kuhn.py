"""Phase 3 Day 2 — Kuhn Deep CFR approximation quality experiment.

Trains DeepCFR + VanillaCFR + CFRPlus on Kuhn in parallel (same seed, same T)
and measures the 3-correlation report + σ̄_deep exploitability at each
checkpoint, per mentor design revision of 2026-04-24.

Correlations measured (see :mod:`poker_ai.eval.deep_cfr_correlation`):
- **Primary A** (``primary_a_advantage_vs_vanilla``): Deep CFR advantage net
  output vs Vanilla CFR signed cumulative_regret — the mathematically
  correct reference per Brown 2019 Algorithm 1 Line 5.
- **Primary B** (``primary_b_strategy_vs_sigma_bar``): Deep CFR strategy net
  output (masked softmax) vs tabular CFR+ σ̄ — simplex-space comparison.
- **Tertiary** (``tertiary_advantage_vs_cfr_plus``): Deep CFR advantage net
  output vs CFR+ positive-only cumulative_regret — retained diagnostic;
  expected to stay low due to sign mismatch (documented in PHASE.md).

Exit criteria (Day 2, mentor 2026-04-24):
- GREEN: Primary A > 0.9 AND Primary B > 0.85 @ T=2000
- STRETCH: Primary A > 0.95 AND Primary B > 0.9 AND σ̄ expl < 1.0 mbb/g
- FAIL: Primary A < 0.7 → audit mandatory

Run:
    uv run python -m experiments.phase3_deep_cfr_kuhn   # full T=2000
    uv run python -m experiments.phase3_deep_cfr_kuhn iterations=500  # smoke
    uv run python -m experiments.phase3_deep_cfr_kuhn wandb.mode=disabled
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from omegaconf import DictConfig, OmegaConf

from poker_ai.algorithms.cfr_plus import CFRPlus
from poker_ai.algorithms.deep_cfr import DeepCFR
from poker_ai.algorithms.vanilla_cfr import VanillaCFR
from poker_ai.eval.deep_cfr_correlation import (
    CorrelationReport,
    compute_correlations,
    deep_cfr_average_strategy,
)
from poker_ai.eval.exploitability import exploitability, exploitability_mbb
from poker_ai.games.kuhn import KuhnPoker

log = logging.getLogger(__name__)


def _train_until(
    deep: DeepCFR,
    vanilla: VanillaCFR,
    cfp: CFRPlus,
    target_T: int,
) -> tuple[float, float, float]:
    """Advance each trainer to ``target_T`` iterations in-place.

    Returns (deep_sec, vanilla_sec, cfp_sec) — per-trainer delta elapsed.
    """
    t0 = time.perf_counter()
    delta_deep = target_T - deep.iteration
    if delta_deep > 0:
        deep.train(delta_deep)
    deep_sec = time.perf_counter() - t0

    t0 = time.perf_counter()
    delta_van = target_T - vanilla.iteration
    if delta_van > 0:
        vanilla.train(delta_van)
    van_sec = time.perf_counter() - t0

    t0 = time.perf_counter()
    delta_cfp = target_T - cfp.iteration
    if delta_cfp > 0:
        cfp.train(delta_cfp)
    cfp_sec = time.perf_counter() - t0

    return deep_sec, van_sec, cfp_sec


def _measure_checkpoint(
    deep: DeepCFR,
    vanilla: VanillaCFR,
    cfp: CFRPlus,
    game: KuhnPoker,
    big_blind: float,
) -> dict[str, float | int]:
    """Compute 3-correlation + σ̄_deep expl at the current iteration."""
    rep: CorrelationReport = compute_correlations(deep, vanilla, cfp, game)
    sigma_bar_deep = deep_cfr_average_strategy(deep, game)
    expl_chips = exploitability(game, sigma_bar_deep)
    expl_mbb = exploitability_mbb(game, sigma_bar_deep, big_blind=big_blind)

    return {
        "T": int(deep.iteration),
        "primary_a": float(rep.primary_a_advantage_vs_vanilla),
        "primary_b": float(rep.primary_b_strategy_vs_sigma_bar),
        "tertiary": float(rep.tertiary_advantage_vs_cfr_plus),
        "sigma_bar_expl_chips": float(expl_chips),
        "sigma_bar_expl_mbb": float(expl_mbb),
        "n_pairs": int(rep.n_pairs),
    }


def _run_training(cfg: DictConfig) -> dict[str, object]:
    run_name = f"kuhn-deep-cfr-seed{cfg.seed}"
    run = wandb.init(
        project=cfg.wandb.project,
        name=run_name,
        tags=list(cfg.wandb.tags) + [f"seed{cfg.seed}"],
        config=OmegaConf.to_container(cfg, resolve=True),
        reinit=True,
        mode=cfg.wandb.mode,
    )
    assert run is not None

    # Seed all three trainers identically.
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))

    game = KuhnPoker()
    deep = DeepCFR(
        game=game,
        n_actions=2,
        encoding_dim=6,
        device=str(cfg.deep_cfr.device),
        seed=int(cfg.seed),
        traversals_per_iter=int(cfg.deep_cfr.traversals_per_iter),
        buffer_capacity=int(cfg.deep_cfr.buffer_capacity),
        batch_size=int(cfg.deep_cfr.batch_size),
        advantage_epochs=int(cfg.deep_cfr.advantage_epochs),
        strategy_epochs=int(cfg.deep_cfr.strategy_epochs),
    )
    vanilla = VanillaCFR(game=game, n_actions=2)
    cfp = CFRPlus(game=game, n_actions=2)

    big_blind = float(cfg.big_blind)
    checkpoints: list[int] = [int(c) for c in cfg.checkpoints]
    # Clamp checkpoints to <= cfg.iterations.
    checkpoints = [c for c in checkpoints if c <= int(cfg.iterations)]
    if not checkpoints or checkpoints[-1] < int(cfg.iterations):
        checkpoints.append(int(cfg.iterations))

    history: list[dict[str, float | int]] = []

    loop_start = time.perf_counter()
    for target_T in checkpoints:
        deep_sec, van_sec, cfp_sec = _train_until(
            deep, vanilla, cfp, target_T
        )
        metrics = _measure_checkpoint(deep, vanilla, cfp, game, big_blind)
        metrics["deep_iter_per_sec"] = (
            (target_T - (history[-1]["T"] if history else 0)) / deep_sec
            if deep_sec > 0
            else float("inf")
        )
        history.append(metrics)

        log.info(
            "T=%4d  prim_A=%.4f  prim_B=%.4f  tert=%.4f  σ̄_expl=%.4f mbb/g  "
            "dt_deep=%.1fs dt_van=%.1fs dt_cfp=%.1fs",
            target_T,
            metrics["primary_a"],
            metrics["primary_b"],
            metrics["tertiary"],
            metrics["sigma_bar_expl_mbb"],
            deep_sec, van_sec, cfp_sec,
        )

        wandb.log(
            {
                "primary_a_advantage_vs_vanilla": metrics["primary_a"],
                "primary_b_strategy_vs_sigma_bar": metrics["primary_b"],
                "tertiary_advantage_vs_cfr_plus": metrics["tertiary"],
                "sigma_bar_expl_chips": metrics["sigma_bar_expl_chips"],
                "sigma_bar_expl_mbb": metrics["sigma_bar_expl_mbb"],
                "deep_iter_per_sec": metrics["deep_iter_per_sec"],
            },
            step=target_T,
        )

    total_sec = time.perf_counter() - loop_start
    final = history[-1]

    prim_a_green = float(cfg.primary_a_green_threshold)
    prim_b_green = float(cfg.primary_b_green_threshold)
    stretch_expl = float(cfg.sigma_bar_expl_mbb_stretch)
    green = (
        final["primary_a"] > prim_a_green
        and final["primary_b"] > prim_b_green
    )
    stretch = (
        green
        and final["primary_a"] > 0.95
        and final["primary_b"] > 0.90
        and final["sigma_bar_expl_mbb"] < stretch_expl
    )

    log.info(
        "FINAL T=%d  prim_A=%.4f  prim_B=%.4f  tert=%.4f  σ̄_expl=%.4f mbb/g  "
        "GREEN=%s  STRETCH=%s  elapsed=%.1fs",
        int(final["T"]),
        final["primary_a"],
        final["primary_b"],
        final["tertiary"],
        final["sigma_bar_expl_mbb"],
        green,
        stretch,
        total_sec,
    )

    wandb.summary.update(
        {
            "final_T": int(final["T"]),
            "final_primary_a": float(final["primary_a"]),
            "final_primary_b": float(final["primary_b"]),
            "final_tertiary": float(final["tertiary"]),
            "final_sigma_bar_expl_mbb": float(final["sigma_bar_expl_mbb"]),
            "total_seconds": total_sec,
            "green": green,
            "stretch": stretch,
        }
    )
    wandb.finish()

    return {
        "history": history,
        "final": final,
        "green": green,
        "stretch": stretch,
        "total_sec": total_sec,
    }


def _plot_convergence(
    history: list[dict[str, float | int]],
    cfg: DictConfig,
    save_path: Path,
) -> plt.Figure:
    steps = np.array([h["T"] for h in history], dtype=np.int64)
    prim_a = np.array([h["primary_a"] for h in history])
    prim_b = np.array([h["primary_b"] for h in history])
    tert = np.array([h["tertiary"] for h in history])
    expl = np.array([h["sigma_bar_expl_mbb"] for h in history])

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    ax_a = axes[0]
    ax_a.plot(steps, prim_a, "o-", label="Primary A (Deep adv vs Vanilla)", color="tab:blue")
    ax_a.plot(steps, prim_b, "s-", label="Primary B (Deep strat vs CFR+ σ̄)", color="tab:green")
    ax_a.plot(steps, tert, "^--", label="Tertiary (Deep adv vs CFR+ R⁺, diag)", color="tab:gray", alpha=0.7)
    ax_a.axhline(y=float(cfg.primary_a_green_threshold), color="tab:blue", linestyle=":", alpha=0.5)
    ax_a.axhline(y=float(cfg.primary_b_green_threshold), color="tab:green", linestyle=":", alpha=0.5)
    ax_a.set_xlabel("iteration")
    ax_a.set_ylabel("Pearson r")
    ax_a.set_title("Correlation trajectory")
    ax_a.grid(True, alpha=0.3)
    ax_a.legend(loc="lower right", fontsize=8)
    ax_a.set_ylim(-0.1, 1.05)

    ax_b = axes[1]
    ax_b.semilogy(steps, expl, "o-", label="σ̄_deep exploitability")
    ax_b.axhline(y=float(cfg.sigma_bar_expl_mbb_stretch), color="green", linestyle=":",
                 label=f"STRETCH (< {cfg.sigma_bar_expl_mbb_stretch} mbb/g)")
    ax_b.set_xlabel("iteration")
    ax_b.set_ylabel("exploitability (mbb/g, log)")
    ax_b.set_title("σ̄_deep exploitability (Exit #4 secondary dry-run)")
    ax_b.grid(True, which="both", alpha=0.3)
    ax_b.legend(loc="upper right", fontsize=8)

    fig.suptitle(
        f"Kuhn Deep CFR — seed={cfg.seed}  T={cfg.iterations}  "
        f"K={cfg.deep_cfr.traversals_per_iter}",
        y=1.02,
    )
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


@hydra.main(version_base="1.3", config_path="conf", config_name="phase3_deep_cfr_kuhn")
def main(cfg: DictConfig) -> None:
    log.info("config:\n%s", OmegaConf.to_yaml(cfg))

    result = _run_training(cfg)
    history = result["history"]
    assert isinstance(history, list)

    run_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    png_path = run_dir / "kuhn_deep_cfr_correlation.png"
    fig = _plot_convergence(history, cfg, png_path)
    log.info("saved figure: %s", png_path)

    summary_run = wandb.init(
        project=cfg.wandb.project,
        name="kuhn-deep-cfr-summary",
        tags=list(cfg.wandb.tags) + ["summary"],
        config=OmegaConf.to_container(cfg, resolve=True),
        reinit=True,
        mode=cfg.wandb.mode,
    )
    assert summary_run is not None
    wandb.log({"convergence_figure": wandb.Image(str(png_path))})
    final = result["final"]
    assert isinstance(final, dict)
    wandb.summary.update(
        {
            "final_T": int(final["T"]),
            "final_primary_a": float(final["primary_a"]),
            "final_primary_b": float(final["primary_b"]),
            "final_tertiary": float(final["tertiary"]),
            "final_sigma_bar_expl_mbb": float(final["sigma_bar_expl_mbb"]),
            "green": bool(result["green"]),
            "stretch": bool(result["stretch"]),
            "total_seconds": float(result["total_sec"]),
        }
    )
    wandb.finish()
    plt.close(fig)


if __name__ == "__main__":
    main()
