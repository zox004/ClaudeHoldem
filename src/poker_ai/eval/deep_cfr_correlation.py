"""Deep CFR approximation-quality metrics (Phase 3 Day 2).

Three correlations between Deep CFR outputs and tabular baselines, with
design-lock-updated roles after 2026-04-24 discovery of a CFR+ vs Vanilla
reference mismatch (see PHASE.md Phase 3 Day 2 log):

- **Primary A** (``advantage_vs_vanilla``)
    Pearson r between ``deep.advantage_nets[p]`` output and
    ``vanilla.cumulative_regret`` — BOTH signed quantities. Brown 2019
    Algorithm 1 Line 5 target is the signed instantaneous regret
    ``π_{-p}(I) · (v_{I,a} - v_I)``; its time-sum is Vanilla CFR's
    cumulative regret. Exit #4 primary gate.

- **Primary B** (``strategy_vs_sigma_bar``)
    Pearson r between ``deep.strategy_net`` output (after masked softmax)
    and the tabular CFR+ time-averaged strategy σ̄. Simplex space, scale-
    independent, directly answers "does Deep CFR produce the right policy".

- **Tertiary** (``advantage_vs_cfr_plus``, diagnostic)
    Original Phase 3 Preview metric, retained as documentation. Deep net
    (signed) vs CFR+ cumulative_regret (positive-by-construction) is a
    mathematical mismatch — this metric stays low even at convergence
    and serves as proof of the design correction.

Scale invariance note: Pearson r(aX+b, Y) = r(X, Y), so the Vanilla /
Deep scale difference (Vanilla raw R_cum grows with T while Deep net
output is a linear-CFR-weighted *average*) does not affect correlation
values. The linear relationship is preserved regardless of T_norm choice.

Reference: Brown, Lerer, Gross, Sandholm 2019, "Deep Counterfactual
Regret Minimization", ICML.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from poker_ai.algorithms.cfr_plus import CFRPlus
from poker_ai.algorithms.deep_cfr import DeepCFR
from poker_ai.algorithms.regret_matching import regret_matching
from poker_ai.algorithms.vanilla_cfr import VanillaCFR
from poker_ai.games.protocol import GameProtocol, StateProtocol


@dataclass
class CorrelationReport:
    """Structured output of :func:`compute_correlations`."""

    primary_a_advantage_vs_vanilla: float
    primary_b_strategy_vs_sigma_bar: float
    tertiary_advantage_vs_cfr_plus: float
    n_pairs: int
    # Vectors retained for plotting / per-infoset drilldown.
    tab_vanilla_vec: np.ndarray = field(default_factory=lambda: np.empty(0))
    tab_cfr_plus_vec: np.ndarray = field(default_factory=lambda: np.empty(0))
    tab_sigma_bar_vec: np.ndarray = field(default_factory=lambda: np.empty(0))
    net_advantage_vec: np.ndarray = field(default_factory=lambda: np.empty(0))
    net_strategy_vec: np.ndarray = field(default_factory=lambda: np.empty(0))


def _collect_infoset_states(
    game: GameProtocol,
) -> dict[str, tuple[int, np.ndarray, StateProtocol]]:
    """DFS all game states, keeping one representative per infoset_key."""
    out: dict[str, tuple[int, np.ndarray, StateProtocol]] = {}

    def dfs(state: StateProtocol) -> None:
        if state.is_terminal:
            return
        key = state.infoset_key
        if key not in out:
            out[key] = (state.current_player, state.legal_action_mask(), state)
        for a in state.legal_actions():
            dfs(state.next_state(a))

    for deal in game.all_deals():
        dfs(game.state_from_deal(deal))
    return out


def _advantage_net_output(
    deep_trainer: DeepCFR,
    player: int,
    encoding_np: np.ndarray,
) -> np.ndarray:
    with torch.no_grad():
        x = torch.from_numpy(encoding_np).to(deep_trainer.device)
        logits = deep_trainer.advantage_nets[player](x)
    out: np.ndarray = logits.detach().cpu().numpy().astype(np.float64)
    return out


def _strategy_net_output_masked(
    deep_trainer: DeepCFR,
    encoding_np: np.ndarray,
    legal_mask: np.ndarray,
) -> np.ndarray:
    """Strategy net raw logits → softmax over legal actions → probabilities.

    Illegal slots are set to -inf before softmax so their posterior is
    exactly 0. Matches inference convention for extracting σ̄ from the
    strategy network.
    """
    with torch.no_grad():
        x = torch.from_numpy(encoding_np).to(deep_trainer.device)
        logits = deep_trainer.strategy_net(x).detach().cpu().numpy().astype(
            np.float64
        )
    mask_f = legal_mask.astype(np.float64)
    # Mask illegal slots with -inf so they become 0 after softmax.
    masked = np.where(mask_f > 0, logits, -np.inf)
    # Numerical-stable softmax.
    m = np.max(masked)
    exp = np.exp(masked - m)
    z = exp.sum()
    if z <= 0.0 or not np.isfinite(z):
        # Fallback: uniform over legal.
        fallback: np.ndarray = mask_f / mask_f.sum()
        return fallback
    probs: np.ndarray = exp / z
    return probs


def _tabular_average_strategy(
    tab_trainer: CFRPlus, infoset_key: str, legal_mask: np.ndarray
) -> np.ndarray:
    """Normalised cumulative_strategy per infoset (CFR+ σ̄)."""
    mask_f = legal_mask.astype(np.float64)
    fallback: np.ndarray = mask_f / mask_f.sum()
    if infoset_key not in tab_trainer.infosets:
        return fallback
    data = tab_trainer.infosets[infoset_key]
    total = data.cumulative_strategy.sum()
    if total > 0.0:
        out: np.ndarray = data.cumulative_strategy / total
        return out
    return fallback


def compute_correlations(
    deep_trainer: DeepCFR,
    vanilla_trainer: VanillaCFR,
    cfr_plus_trainer: CFRPlus,
    game: GameProtocol,
) -> CorrelationReport:
    """Full 3-metric report over the shared flat (infoset, legal_action) axis.

    Each vector has length ``n_pairs = Σ_{infoset} legal_action_count``.
    For Kuhn (legal mask all-True): 12 × 2 = 24. For Leduc: variable.

    An :class:`AssertionError` is raised if no pairs are collected.
    """
    infoset_data = _collect_infoset_states(game)

    tab_vanilla: list[float] = []
    tab_cfr_plus: list[float] = []
    tab_sigma_bar: list[float] = []
    net_advantage: list[float] = []
    net_strategy: list[float] = []

    for key, (player, legal_mask, state) in infoset_data.items():
        encoding_np = game.encode(state)
        net_adv = _advantage_net_output(deep_trainer, player, encoding_np)
        net_strat = _strategy_net_output_masked(
            deep_trainer, encoding_np, legal_mask
        )

        van_R = (
            vanilla_trainer.infosets[key].cumulative_regret
            if key in vanilla_trainer.infosets
            else np.zeros(game.NUM_ACTIONS, dtype=np.float64)
        )
        cfr_R = (
            cfr_plus_trainer.infosets[key].cumulative_regret
            if key in cfr_plus_trainer.infosets
            else np.zeros(game.NUM_ACTIONS, dtype=np.float64)
        )
        tab_sigma = _tabular_average_strategy(
            cfr_plus_trainer, key, legal_mask
        )

        for a_idx in range(game.NUM_ACTIONS):
            if not legal_mask[a_idx]:
                continue
            tab_vanilla.append(float(van_R[a_idx]))
            tab_cfr_plus.append(float(cfr_R[a_idx]))
            tab_sigma_bar.append(float(tab_sigma[a_idx]))
            net_advantage.append(float(net_adv[a_idx]))
            net_strategy.append(float(net_strat[a_idx]))

    assert len(tab_vanilla) > 0, "no infoset pairs collected"

    tv = np.asarray(tab_vanilla, dtype=np.float64)
    tc = np.asarray(tab_cfr_plus, dtype=np.float64)
    ts = np.asarray(tab_sigma_bar, dtype=np.float64)
    na = np.asarray(net_advantage, dtype=np.float64)
    ns = np.asarray(net_strategy, dtype=np.float64)

    def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
        if x.std() == 0.0 or y.std() == 0.0:
            return float("nan")
        return float(np.corrcoef(x, y)[0, 1])

    return CorrelationReport(
        primary_a_advantage_vs_vanilla=_safe_corr(tv, na),
        primary_b_strategy_vs_sigma_bar=_safe_corr(ts, ns),
        tertiary_advantage_vs_cfr_plus=_safe_corr(tc, na),
        n_pairs=len(tv),
        tab_vanilla_vec=tv,
        tab_cfr_plus_vec=tc,
        tab_sigma_bar_vec=ts,
        net_advantage_vec=na,
        net_strategy_vec=ns,
    )


# ---------------------------------------------------------------------------
# Backward-compat: original tertiary-only API kept for existing integration
# tests. New code should use :func:`compute_correlations` above.
# ---------------------------------------------------------------------------


def compute_flat_correlation(
    deep_trainer: DeepCFR,
    tab_trainer: CFRPlus,
    game: GameProtocol,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Tertiary diagnostic — Deep advantage vs CFR+ cumulative_regret.

    Retained for backward compatibility and as documentation of the design
    correction made on 2026-04-24 (CFR+ R⁺ is NOT Deep CFR's target — see
    :func:`compute_correlations` for the full 3-metric report).
    """
    infoset_data = _collect_infoset_states(game)
    tab_values: list[float] = []
    net_values: list[float] = []

    for key, (player, legal_mask, state) in infoset_data.items():
        if key not in tab_trainer.infosets:
            continue
        tab_R = tab_trainer.infosets[key].cumulative_regret
        encoding_np = game.encode(state)
        net_out = _advantage_net_output(deep_trainer, player, encoding_np)
        for a_idx in range(game.NUM_ACTIONS):
            if legal_mask[a_idx]:
                tab_values.append(float(tab_R[a_idx]))
                net_values.append(float(net_out[a_idx]))

    assert len(tab_values) > 0
    tab_vec = np.asarray(tab_values, dtype=np.float64)
    net_vec = np.asarray(net_values, dtype=np.float64)
    r = float(np.corrcoef(tab_vec, net_vec)[0, 1])
    return r, tab_vec, net_vec


# ---------------------------------------------------------------------------
# σ̄ extraction from Deep CFR strategy net — used by harness / exploitability
# ---------------------------------------------------------------------------


def deep_cfr_average_strategy(
    deep_trainer: DeepCFR,
    game: GameProtocol,
) -> dict[str, np.ndarray]:
    """Extract σ̄ per infoset from ``strategy_net`` (legal-masked softmax).

    Used by :mod:`poker_ai.eval.exploitability` to evaluate σ̄ expl as the
    Exit #4 secondary metric.
    """
    infoset_data = _collect_infoset_states(game)
    out: dict[str, np.ndarray] = {}
    for key, (_player, legal_mask, state) in infoset_data.items():
        encoding_np = game.encode(state)
        out[key] = _strategy_net_output_masked(
            deep_trainer, encoding_np, legal_mask
        )
    return out


__all__ = [
    "CorrelationReport",
    "compute_correlations",
    "compute_flat_correlation",
    "deep_cfr_average_strategy",
    "regret_matching",
]
