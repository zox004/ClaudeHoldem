"""Phase 3 Day 3 — Leduc Deep CFR approximation quality experiment.

Direct scale comparison to Kuhn Day 2b-A. Trains DeepCFR + VanillaCFR +
CFRPlus on Leduc (same seed=42, K=100, T=500) and measures the 3-metric
correlation report + σ̄_deep exploitability at checkpoints 100/250/500.

Extends :mod:`phase3_deep_cfr_kuhn` by (a) switching the game to
:class:`LeducPoker`, (b) bumping ``n_actions=3`` / ``encoding_dim=13``,
and (c) adding a round-1 vs round-2 per-infoset breakdown to track the
mentor's pure/mixed hypothesis (Day 3 observation 2).

Run:
    uv run python -m experiments.phase3_deep_cfr_leduc
    uv run python -m experiments.phase3_deep_cfr_leduc iterations=200
    uv run python -m experiments.phase3_deep_cfr_leduc wandb.mode=disabled
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
    _collect_infoset_states,
    _strategy_net_output_masked,
    compute_correlations,
    deep_cfr_average_strategy,
)
from poker_ai.eval.exploitability import exploitability, exploitability_mbb
from poker_ai.games.leduc import LeducPoker

log = logging.getLogger(__name__)


def _train_until(
    deep: DeepCFR,
    vanilla: VanillaCFR,
    cfp: CFRPlus,
    target_T: int,
) -> tuple[float, float, float]:
    t0 = time.perf_counter()
    delta_deep = target_T - deep.iteration
    history_start = len(deep.train_history)
    if delta_deep > 0:
        deep.train(delta_deep)
    deep_sec = time.perf_counter() - t0

    # Phase 3 Day 5 H Tier 1 logging: forward per-iter train stats to W&B.
    # Aggregates the 3 events per iteration (advantage_p0, advantage_p1,
    # strategy) into a single wandb.log call at step=iter so that step axis
    # stays monotone and merges with checkpoint metrics logged at
    # step=target_T.
    new_events = deep.train_history[history_start:]
    by_iter: dict[int, dict[str, float]] = {}
    for ev in new_events:
        it = int(ev["iter"])  # type: ignore[arg-type]
        bucket = by_iter.setdefault(it, {})
        net = str(ev["net"])
        if net == "advantage":
            p = int(ev["player"])  # type: ignore[arg-type]
            tag = f"adv_p{p}"
        else:
            tag = "strat"
        bucket[f"train/{tag}_loss_initial"] = float(ev["loss_initial"])  # type: ignore[arg-type]
        bucket[f"train/{tag}_loss_final"] = float(ev["loss_final"])  # type: ignore[arg-type]
        bucket[f"train/{tag}_target_abs_mean"] = float(ev["target_abs_mean"])  # type: ignore[arg-type]
        bucket[f"train/{tag}_target_abs_std"] = float(ev["target_abs_std"])  # type: ignore[arg-type]
        bucket[f"train/{tag}_grad_norm_max"] = float(ev["grad_norm_max"])  # type: ignore[arg-type]
        bucket[f"train/{tag}_n_samples"] = float(ev["n_samples"])  # type: ignore[arg-type]
        # Tier 2 (L-B active): forward baseline_* keys if present.
        for key in ("baseline_n_keys", "baseline_abs_mean", "baseline_var"):
            if key in ev:
                bucket[f"train/{tag}_{key}"] = float(ev[key])  # type: ignore[arg-type]
    for it in sorted(by_iter):
        wandb.log(by_iter[it], step=it)

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


def _round_from_key(infoset_key: str) -> int:
    """Leduc infoset key format ``<rank>|<r1>[.<board><r2>]`` — round 2 has
    a ``'.'`` separator. Returns 0 (round 1) or 1 (round 2)."""
    return 1 if "." in infoset_key else 0


def _per_round_linf(
    deep: DeepCFR,
    cfp: CFRPlus,
    game: LeducPoker,
) -> dict[str, float]:
    """Per-round mean L∞ between σ_deep and σ_CFR+ (pure/mixed split)."""
    infoset_data = _collect_infoset_states(game)
    sigma_cfp_map = cfp.average_strategy()

    round_linf: dict[int, dict[str, list[float]]] = {
        0: {"pure": [], "mixed": []},
        1: {"pure": [], "mixed": []},
    }

    for key, (_player, legal_mask, state) in infoset_data.items():
        if key not in sigma_cfp_map:
            continue
        c = sigma_cfp_map[key]
        encoding = game.encode(state)
        d = _strategy_net_output_masked(deep, encoding, legal_mask)
        legal_slots = legal_mask.nonzero()[0]
        if legal_slots.size == 0:
            continue
        linf = float(np.max(np.abs(d[legal_slots] - c[legal_slots])))
        # Pure: any legal slot has prob ≥ 0.95 or ≤ 0.05.
        is_pure = bool(
            (c[legal_slots] >= 0.95).any() or (c[legal_slots] <= 0.05).any()
        )
        rnd = _round_from_key(key)
        round_linf[rnd]["pure" if is_pure else "mixed"].append(linf)

    def _mean(xs: list[float]) -> float:
        return float(np.mean(xs)) if xs else float("nan")

    return {
        "r1_pure_linf_mean": _mean(round_linf[0]["pure"]),
        "r1_mixed_linf_mean": _mean(round_linf[0]["mixed"]),
        "r2_pure_linf_mean": _mean(round_linf[1]["pure"]),
        "r2_mixed_linf_mean": _mean(round_linf[1]["mixed"]),
        "r1_pure_n": float(len(round_linf[0]["pure"])),
        "r1_mixed_n": float(len(round_linf[0]["mixed"])),
        "r2_pure_n": float(len(round_linf[1]["pure"])),
        "r2_mixed_n": float(len(round_linf[1]["mixed"])),
    }


def _measure_checkpoint(
    deep: DeepCFR,
    vanilla: VanillaCFR,
    cfp: CFRPlus,
    game: LeducPoker,
    big_blind: float,
) -> dict[str, float | int]:
    rep: CorrelationReport = compute_correlations(deep, vanilla, cfp, game)
    sigma_bar_deep = deep_cfr_average_strategy(deep, game)
    expl_chips = exploitability(game, sigma_bar_deep)
    expl_mbb = exploitability_mbb(game, sigma_bar_deep, big_blind=big_blind)

    # Tabular σ̄ expl baselines for the same checkpoint (reference).
    sigma_cfp = cfp.average_strategy()
    cfp_expl_mbb = exploitability_mbb(game, sigma_cfp, big_blind=big_blind)
    sigma_van = vanilla.average_strategy()
    van_expl_mbb = exploitability_mbb(game, sigma_van, big_blind=big_blind)

    metrics: dict[str, float | int] = {
        "T": int(deep.iteration),
        "primary_a": float(rep.primary_a_advantage_vs_vanilla),
        "primary_b": float(rep.primary_b_strategy_vs_sigma_bar),
        "tertiary": float(rep.tertiary_advantage_vs_cfr_plus),
        "sigma_bar_expl_chips": float(expl_chips),
        "sigma_bar_expl_mbb": float(expl_mbb),
        "cfr_plus_expl_mbb": float(cfp_expl_mbb),
        "vanilla_expl_mbb": float(van_expl_mbb),
        "n_pairs": int(rep.n_pairs),
    }
    metrics.update(_per_round_linf(deep, cfp, game))
    return metrics


def _run_training(cfg: DictConfig) -> dict[str, object]:
    run_name = f"leduc-deep-cfr-seed{cfg.seed}"
    run = wandb.init(
        project=cfg.wandb.project,
        name=run_name,
        tags=list(cfg.wandb.tags) + [f"seed{cfg.seed}"],
        config=OmegaConf.to_container(cfg, resolve=True),
        reinit=True,
        mode=cfg.wandb.mode,
    )
    assert run is not None

    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))

    game = LeducPoker()
    deep = DeepCFR(
        game=game,
        n_actions=3,
        encoding_dim=13,
        device=str(cfg.deep_cfr.device),
        seed=int(cfg.seed),
        traversals_per_iter=int(cfg.deep_cfr.traversals_per_iter),
        buffer_capacity=int(cfg.deep_cfr.buffer_capacity),
        batch_size=int(cfg.deep_cfr.batch_size),
        advantage_epochs=int(cfg.deep_cfr.advantage_epochs),
        strategy_epochs=int(cfg.deep_cfr.strategy_epochs),
        hidden_dim=int(cfg.deep_cfr.get("hidden_dim", 64)),
        num_hidden_layers=int(cfg.deep_cfr.get("num_hidden_layers", 2)),
        advantage_baseline=str(cfg.deep_cfr.get("advantage_baseline", "none")),
        baseline_alpha=float(cfg.deep_cfr.get("baseline_alpha", 0.1)),
    )
    vanilla = VanillaCFR(game=game, n_actions=3)
    cfp = CFRPlus(game=game, n_actions=3)

    big_blind = float(cfg.big_blind)
    checkpoints: list[int] = [int(c) for c in cfg.checkpoints]
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
            "T=%4d  prim_A=%.4f  prim_B=%.4f  tert=%.4f  σ̄_expl=%.4f  "
            "cfp=%.4f van=%.3f  r1[pure %.3f n=%d, mix %.3f n=%d] "
            "r2[pure %.3f n=%d, mix %.3f n=%d]  "
            "dt_deep=%.1fs dt_van=%.1fs dt_cfp=%.1fs",
            target_T,
            metrics["primary_a"],
            metrics["primary_b"],
            metrics["tertiary"],
            metrics["sigma_bar_expl_mbb"],
            metrics["cfr_plus_expl_mbb"],
            metrics["vanilla_expl_mbb"],
            metrics["r1_pure_linf_mean"], int(metrics["r1_pure_n"]),
            metrics["r1_mixed_linf_mean"], int(metrics["r1_mixed_n"]),
            metrics["r2_pure_linf_mean"], int(metrics["r2_pure_n"]),
            metrics["r2_mixed_linf_mean"], int(metrics["r2_mixed_n"]),
            deep_sec, van_sec, cfp_sec,
        )

        wandb.log(
            {k: v for k, v in metrics.items() if k != "T"},
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
        and final["sigma_bar_expl_mbb"] < stretch_expl
    )

    log.info(
        "FINAL T=%d  prim_A=%.4f  prim_B=%.4f  σ̄_expl=%.4f mbb/g  "
        "cfp_ref=%.4f  GREEN=%s  elapsed=%.1fs",
        int(final["T"]),
        final["primary_a"],
        final["primary_b"],
        final["sigma_bar_expl_mbb"],
        final["cfr_plus_expl_mbb"],
        green,
        total_sec,
    )

    wandb.summary.update(
        {
            "final_T": int(final["T"]),
            "final_primary_a": float(final["primary_a"]),
            "final_primary_b": float(final["primary_b"]),
            "final_tertiary": float(final["tertiary"]),
            "final_sigma_bar_expl_mbb": float(final["sigma_bar_expl_mbb"]),
            "final_cfr_plus_expl_mbb": float(final["cfr_plus_expl_mbb"]),
            "total_seconds": total_sec,
            "green": green,
        }
    )
    wandb.finish()

    return {
        "history": history,
        "final": final,
        "green": green,
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
    expl_deep = np.array([h["sigma_bar_expl_mbb"] for h in history])
    expl_cfp = np.array([h["cfr_plus_expl_mbb"] for h in history])
    expl_van = np.array([h["vanilla_expl_mbb"] for h in history])

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    ax_a = axes[0]
    ax_a.plot(steps, prim_a, "o-", label="Primary A (adv vs Vanilla)", color="tab:blue")
    ax_a.plot(steps, prim_b, "s-", label="Primary B (strat vs CFR+ σ̄)", color="tab:green")
    ax_a.plot(steps, tert, "^--", label="Tertiary (adv vs CFR+ R⁺, diag)", color="tab:gray", alpha=0.7)
    ax_a.axhline(y=float(cfg.primary_a_green_threshold), color="tab:blue", linestyle=":", alpha=0.5)
    ax_a.axhline(y=float(cfg.primary_b_green_threshold), color="tab:green", linestyle=":", alpha=0.5)
    ax_a.set_xlabel("iteration")
    ax_a.set_ylabel("Pearson r")
    ax_a.set_title("Leduc correlation trajectory")
    ax_a.grid(True, alpha=0.3)
    ax_a.legend(loc="lower right", fontsize=8)
    ax_a.set_ylim(-0.1, 1.05)

    ax_b = axes[1]
    ax_b.semilogy(steps, expl_deep, "o-", label="σ̄_deep", color="tab:blue")
    ax_b.semilogy(steps, expl_cfp, "s-", label="σ̄_CFR+ (baseline)", color="tab:green")
    ax_b.semilogy(steps, expl_van, "^-", label="σ̄_Vanilla", color="tab:orange")
    ax_b.axhline(y=float(cfg.sigma_bar_expl_mbb_stretch), color="red", linestyle=":",
                 label=f"GREEN (< {cfg.sigma_bar_expl_mbb_stretch} mbb/g)")
    ax_b.set_xlabel("iteration")
    ax_b.set_ylabel("exploitability (mbb/g, log)")
    ax_b.set_title("σ̄ exploitability trajectory")
    ax_b.grid(True, which="both", alpha=0.3)
    ax_b.legend(loc="upper right", fontsize=8)

    fig.suptitle(
        f"Leduc Deep CFR — seed={cfg.seed}  T={cfg.iterations}  "
        f"K={cfg.deep_cfr.traversals_per_iter}",
        y=1.02,
    )
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


@hydra.main(version_base="1.3", config_path="conf", config_name="phase3_deep_cfr_leduc")
def main(cfg: DictConfig) -> None:
    log.info("config:\n%s", OmegaConf.to_yaml(cfg))

    result = _run_training(cfg)
    history = result["history"]
    assert isinstance(history, list)

    run_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    png_path = run_dir / "leduc_deep_cfr_correlation.png"
    fig = _plot_convergence(history, cfg, png_path)
    log.info("saved figure: %s", png_path)

    summary_run = wandb.init(
        project=cfg.wandb.project,
        name="leduc-deep-cfr-summary",
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
            "final_cfr_plus_expl_mbb": float(final["cfr_plus_expl_mbb"]),
            "green": bool(result["green"]),
            "total_seconds": float(result["total_sec"]),
        }
    )
    wandb.finish()
    plt.close(fig)


if __name__ == "__main__":
    main()
