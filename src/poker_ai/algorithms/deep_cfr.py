"""Deep CFR (Brown, Lerer, Gross, Sandholm 2019) — Phase 3 Day 1 skeleton.

External-sampling CFR traversal + per-player advantage network + shared
strategy network. The traversal structure is lifted from
:class:`MCCFRExternalSampling` (Phase 2 Day 6) with two edits:

1. Regret lookup: tabular ``cumulative_regret`` array → forward pass on
   ``advantage_nets[player]`` with the encoded infoset.
2. Regret update: in-place table write → push
   ``(encoding, regret_vector, iter_weight)`` into ``advantage_buffers[p]``
   (a Vitter reservoir) for end-of-iteration network retraining.

Strategy sampling (Brown 2019 Algorithm 1 line 16-17): at the NON-updating
player's decision nodes, push ``(encoding, current_strategy, iter_weight)``
to ``strategy_buffer`` — the strategy network is retrained on this buffer
after both players' advantage updates complete.

Design lock references (see PHASE.md Phase 3 Preview):
- Decision 1 (encode): ``game.encode(state) -> np.ndarray[ENCODING_DIM]``
- Decision 3 (reservoir): torch-backed Vitter buffer
- Decision 4 (schedule): **from-scratch reinit** of networks per iter
  (Brown 2019 §3 default). Warm-start fallback is Day 4 contingency.
- Decision 5 (Linear CFR): loss-side sample weighting by ``iter_weight``
- Decision 6 (strategy net from Day 1): σ̄ eval baseline — this module
  trains BOTH networks every iteration.
- A1 (K=1000 traversals) — configurable via ``traversals_per_iter``
- A2 (grad clip norm 10.0)

Day 1 scope: smoke-level. Convergence audits (Day 3+) verify correctness.

Reference: Brown et al. 2019, "Deep Counterfactual Regret Minimization", ICML.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from poker_ai.algorithms.regret_matching import regret_matching
from poker_ai.algorithms.reservoir import ReservoirBuffer
from poker_ai.games.protocol import GameProtocol, StateProtocol
from poker_ai.networks.advantage_net import AdvantageNet
from poker_ai.networks.strategy_net import StrategyNet

_GRAD_CLIP_NORM = 10.0   # Design lock A2
_LEARNING_RATE = 1e-3    # Brown 2019 §3.1 Adam default


class DeepCFR:
    """Deep CFR trainer.

    Parameters:
        game: GameProtocol (Kuhn or Leduc; Phase 4+ adds HUNL).
        n_actions: action-space size (must match ``game.NUM_ACTIONS``).
        encoding_dim: feature vector length (must match ``game.ENCODING_DIM``).
        device: torch device string (``"cpu"``, ``"mps"``, ``"cuda"``).
        seed: RNG seed for reproducibility across traversal, sampling, and
            network init.
        traversals_per_iter: K in Brown 2019 Alg 1. Design lock A1 = 1000.
        buffer_capacity: reservoir size for each advantage + strategy buffer.
            Design lock §3: 1_000_000 production; scale down for tests.
        batch_size: minibatch size for network training.
        advantage_epochs: epochs-per-iter for advantage net retraining.
        strategy_epochs: epochs-per-iter for strategy net retraining.
        epsilon: ε-exploration for external sampling (Lanctot §3.2).
    """

    def __init__(
        self,
        game: GameProtocol,
        n_actions: int,
        encoding_dim: int,
        device: str = "cpu",
        seed: int = 0,
        traversals_per_iter: int = 1000,
        buffer_capacity: int = 1_000_000,
        batch_size: int = 256,
        advantage_epochs: int = 4,
        strategy_epochs: int = 4,
        epsilon: float = 0.05,
    ) -> None:
        self.game = game
        self.n_actions = n_actions
        self.encoding_dim = encoding_dim
        self.device = torch.device(device)
        self.seed = seed
        self.traversals_per_iter = traversals_per_iter
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.advantage_epochs = advantage_epochs
        self.strategy_epochs = strategy_epochs
        self.epsilon = epsilon

        # Global torch seed for reproducible network init; the sampling RNG
        # is kept separate (numpy Generator) so it does not perturb torch.
        torch.manual_seed(seed)
        self._rng = np.random.default_rng(seed)

        self.advantage_nets: dict[int, nn.Module] = {
            p: AdvantageNet(encoding_dim, n_actions).to(self.device)
            for p in (0, 1)
        }
        self.strategy_net: nn.Module = StrategyNet(encoding_dim, n_actions).to(
            self.device
        )

        self.advantage_buffers: dict[int, ReservoirBuffer] = {
            p: ReservoirBuffer(
                capacity=buffer_capacity,
                feature_dim=encoding_dim,
                device=device,
                seed=seed + 1 + p,
                target_dim=n_actions,
            )
            for p in (0, 1)
        }
        # Strategy buffer stores legal masks too — cross-entropy training
        # needs them (target==0 is ambiguous between "illegal" and "pure
        # strategy zero"). Phase 3 Day 2 fix.
        self.strategy_buffer: ReservoirBuffer = ReservoirBuffer(
            capacity=buffer_capacity,
            feature_dim=encoding_dim,
            device=device,
            seed=seed + 3,
            target_dim=n_actions,
            mask_dim=n_actions,
        )

        self.iteration: int = 0

    # -------------------------------------------------------------- public API

    def train(self, iterations: int) -> None:
        """Brown 2019 Algorithm 1: alternating-player external sampling.

        Per iter T:
            1. For each updating player p ∈ {0, 1}:
               K traversals → advantage_buffers[p] populated with
               (encoding, regret_vector, T).
               Retrain advantage_nets[p] from scratch on advantage_buffers[p].
            2. Retrain strategy_net from scratch on strategy_buffer.
        """
        deals = self.game.all_deals()
        n_deals = len(deals)

        for _ in range(iterations):
            self.iteration += 1
            for updating_player in (0, 1):
                for _ in range(self.traversals_per_iter):
                    deal_idx = int(self._rng.integers(0, n_deals))
                    root = self.game.state_from_deal(deals[deal_idx])
                    self._traverse(root, updating_player=updating_player)
                self._train_advantage_net(updating_player)
            self._train_strategy_net()

    # --------------------------------------------------------------- internal

    def _traverse(
        self,
        state: StateProtocol,
        updating_player: int,
    ) -> float:
        """External-sampling traversal (Lanctot 2009 §3.4 structure).

        Returns updating_player's utility-from-below at ``state``.
        """
        if state.is_terminal:
            u_p1 = self.game.terminal_utility(state)
            return u_p1 if updating_player == 0 else -u_p1

        encoding_np = self.game.encode(state)
        legal_mask = state.legal_action_mask()
        mask_f = legal_mask.astype(np.float64)
        acting = state.current_player

        # Current strategy: regret-matching on network output (masked).
        strategy = self._strategy_from_advantage_net(
            acting, encoding_np, legal_mask
        )

        if acting == updating_player:
            action_values = np.zeros(self.n_actions, dtype=np.float64)
            for a in state.legal_actions():
                a_idx = int(a)
                action_values[a_idx] = self._traverse(
                    state.next_state(a),
                    updating_player=updating_player,
                )
            node_value = float(strategy @ action_values)

            # Regret target (masked). Brown 2019 Eq.(4): sampled regret =
            # action_values - node_value at I.
            regret_target = (action_values - node_value) * mask_f

            # Push (encoding, regret_vector, T) into advantage buffer.
            encoding_t = torch.from_numpy(encoding_np).to(self.device)
            regret_t = torch.from_numpy(
                regret_target.astype(np.float32)
            ).to(self.device)
            self.advantage_buffers[updating_player].add(
                features=encoding_t,
                target=regret_t,
                iter_weight=float(self.iteration),
            )
            return node_value

        # Non-updating player: (a) push strategy sample (with legal mask for
        # cross-entropy training — Phase 3 Day 2 fix), (b) single-action
        # external sample via ε-smoothed distribution.
        encoding_t = torch.from_numpy(encoding_np).to(self.device)
        strategy_t = torch.from_numpy(
            strategy.astype(np.float32)
        ).to(self.device)
        mask_t = torch.from_numpy(legal_mask.astype(bool)).to(self.device)
        self.strategy_buffer.add(
            features=encoding_t,
            target=strategy_t,
            iter_weight=float(self.iteration),
            mask=mask_t,
        )

        smoothed = self._epsilon_smoothed(strategy, mask_f)
        sampled_idx = int(self._rng.choice(self.n_actions, p=smoothed))
        sampled_action = next(
            a for a in state.legal_actions() if int(a) == sampled_idx
        )
        return self._traverse(
            state.next_state(sampled_action),
            updating_player=updating_player,
        )

    def _strategy_from_advantage_net(
        self,
        player: int,
        encoding_np: np.ndarray,
        legal_mask: np.ndarray,
    ) -> np.ndarray:
        """Forward ``advantage_nets[player]`` → positive-part regret → σ."""
        with torch.no_grad():
            x = torch.from_numpy(encoding_np).to(self.device)
            logits = self.advantage_nets[player](x)
        regrets = logits.detach().cpu().numpy().astype(np.float64)
        return regret_matching(regrets, legal_mask=legal_mask)

    def _epsilon_smoothed(
        self,
        strategy: np.ndarray,
        mask_f: np.ndarray,
    ) -> np.ndarray:
        """``(1-ε)σ + ε·uniform_over_legal`` with strict illegal-slot zeroing."""
        n_legal = mask_f.sum()
        uniform_legal = mask_f / n_legal
        smoothed = (1.0 - self.epsilon) * strategy + self.epsilon * uniform_legal
        smoothed = smoothed * mask_f
        total = smoothed.sum()
        out: np.ndarray = smoothed / total
        return out

    # ------------------------------------------------------ network training

    def _train_advantage_net(self, player: int) -> None:
        """Re-init + fit ``advantage_nets[player]`` on ``advantage_buffers[player]``.

        Brown 2019 §3 trains the network from scratch each iteration to avoid
        stale regret accumulation. Linear CFR weighting is applied as a
        per-sample loss weight (design lock decision #5).
        """
        buf = self.advantage_buffers[player]
        n = len(buf)
        if n == 0:
            return

        # From-scratch reinit (design lock #4 default path).
        self.advantage_nets[player] = AdvantageNet(
            self.encoding_dim, self.n_actions
        ).to(self.device)
        net = self.advantage_nets[player]
        optimizer = torch.optim.Adam(net.parameters(), lr=_LEARNING_RATE)

        features, targets, weights = buf.sample_all()
        batch = min(self.batch_size, n)
        for _ in range(self.advantage_epochs):
            perm = torch.randperm(n, device=self.device)
            for start in range(0, n, batch):
                idx = perm[start:start + batch]
                x_b = features[idx]
                y_b = targets[idx]
                w_b = weights[idx]

                pred = net(x_b)
                # Per-sample weighted MSE (Linear CFR loss weighting).
                # Normalized weighted mean: Σ w_i · s_i / Σ w_i (not / n),
                # so T-dependent weight scale does not inflate effective
                # gradient magnitude as iteration grows (Phase 3 Day 2b
                # hypothesis A fix — iter_weight=t polynomial growth would
                # otherwise bias optimisation toward recent samples).
                per_sample = ((pred - y_b) ** 2).mean(dim=1)
                w_sum = w_b.sum().clamp(min=1e-8)
                loss = (per_sample * w_b).sum() / w_sum

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), _GRAD_CLIP_NORM)
                optimizer.step()

    def _train_strategy_net(self) -> None:
        """Re-init + fit ``strategy_net`` on ``strategy_buffer`` via
        soft-target **cross-entropy** with legal-action masking.

        Phase 3 Day 2 fix (2026-04-24): replaced MSE with cross-entropy
        after smoke revealed MSE collapses to smoothed middle values
        (output range ~[0.3, 0.7]) and fails on pure-strategy infosets
        where CFR+ σ̄ has Nash-style extremes (0.0 / 1.0). Cross-entropy
        has stronger gradient pressure on extreme targets.

        Loss (per sample, averaged over batch):
            ``- Σ_a target[a] · log softmax(logits_masked)[a]``
        where ``logits_masked[a] = logits[a] if legal else -inf`` so that
        softmax assigns exactly 0 probability to illegal actions.
        """
        buf = self.strategy_buffer
        n = len(buf)
        if n == 0:
            return

        self.strategy_net = StrategyNet(
            self.encoding_dim, self.n_actions
        ).to(self.device)
        net = self.strategy_net
        optimizer = torch.optim.Adam(net.parameters(), lr=_LEARNING_RATE)

        features, targets, weights, masks = buf.sample_all_with_masks()
        batch = min(self.batch_size, n)
        neg_inf = torch.tensor(float("-inf"), device=self.device)
        for _ in range(self.strategy_epochs):
            perm = torch.randperm(n, device=self.device)
            for start in range(0, n, batch):
                idx = perm[start:start + batch]
                x_b = features[idx]
                y_b = targets[idx]      # (B, n_actions) — prob simplex
                w_b = weights[idx]
                m_b = masks[idx]        # (B, n_actions) — bool legal mask

                logits = net(x_b)                         # (B, n_actions) raw
                # Legal-mask the logits by subst. -inf on illegal slots.
                masked = torch.where(m_b, logits, neg_inf)
                log_probs = torch.nn.functional.log_softmax(masked, dim=-1)
                # Soft-target cross-entropy: -Σ target[a] · log_probs[a].
                # target[a] = 0 on illegal by construction; 0 · -inf is
                # guarded by ``nan_to_num`` to handle any residual noise.
                terms = torch.where(
                    y_b > 0.0,
                    y_b * log_probs,
                    torch.zeros_like(y_b),
                )
                per_sample_ce = -terms.sum(dim=-1)
                # Normalized weighted mean (Day 2b hypothesis A): same
                # rationale as advantage path — decouple effective gradient
                # from T-dependent weight scale.
                w_sum = w_b.sum().clamp(min=1e-8)
                loss = (per_sample_ce * w_b).sum() / w_sum

                optimizer.zero_grad()
                loss.backward()  # type: ignore[no-untyped-call]
                nn.utils.clip_grad_norm_(net.parameters(), _GRAD_CLIP_NORM)
                optimizer.step()
