"""
Learned reward simulator R_hat(s_t, B_t) -> predicted_reward (Dyna).

Architecture
------------
Input:  [s_t ; mean_pool(phi(x) for x in B_t) ; max_pool(phi(x) for x in B_t)]
        dim = state_dim + 2 * phi_dim

3-layer MLP (hidden_dim=256, ReLU, LayerNorm)
3 output heads: [r_ood, r_cov, r_unc]

Training
--------
Collect (s_t, phi_batch, r_t) triples from real rollouts.
Train with MSE on each output head.
Update after every real environment interaction.

Drift detection
---------------
Periodically compare simulator predictions against real reward on held-out
transitions.  Log mean absolute error; warn if above threshold.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class RewardSimulatorNet(nn.Module):
    """3-layer MLP simulator with LayerNorm."""

    def __init__(self, input_dim: int, hidden_dim: int, n_outputs: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BatchSimulator:
    """
    Learned simulator for predicting round-level rewards given state + batch.

    Implements the Dyna model: trained on real transitions, used to generate
    imagined rollouts for PPO training.
    """

    def __init__(self, config: Dict[str, Any], state_dim: int, phi_dim: int, device: str = "cpu"):
        """
        Parameters
        ----------
        config    : full config dict.
        state_dim : dimension of the state vector.
        phi_dim   : dimension of phi(x) candidate features.
        device    : torch device string.
        """
        cfg = config["simulator"]
        self.hidden_dim: int = cfg["hidden_dim"]
        self.n_outputs: int = cfg["n_outputs"]
        self.lr: float = cfg["lr"]
        self.buffer_size: int = cfg["buffer_size"]
        self.min_samples: int = cfg["min_samples_to_train"]
        self.device = device

        # Input: state + mean_phi + max_phi
        self.input_dim = state_dim + 2 * phi_dim

        self.net = RewardSimulatorNet(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            n_outputs=self.n_outputs,
        ).to(device)

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

        # Replay buffer: list of (input_vec, target_vec) pairs
        self._buffer: Deque[Tuple[np.ndarray, np.ndarray]] = deque(maxlen=self.buffer_size)

        # Training step counter
        self._train_steps: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_transition(
        self,
        state: np.ndarray,          # [state_dim]
        batch_phi: np.ndarray,      # [B, phi_dim]
        reward_components: Dict[str, float],  # dict with r_ood, r_cov, r_unc
    ) -> None:
        """
        Store a (state, batch, reward) triple in the replay buffer.

        Parameters
        ----------
        state           : full state vector at the start of the round.
        batch_phi       : phi(x) features for each item in selected batch [B, phi_dim].
        reward_components : {'r_ood': ..., 'r_cov': ..., 'r_unc': ...}
        """
        input_vec = self._build_input(state, batch_phi)
        target = np.array([
            reward_components.get("r_ood", 0.0),
            reward_components.get("r_cov", 0.0),
            reward_components.get("r_unc", 0.0),
        ], dtype=np.float32)
        self._buffer.append((input_vec, target))

    def update(self, n_epochs: int = 1, batch_size: int = 32) -> Optional[float]:
        """
        Train the simulator on buffered transitions.

        Returns average MSE loss, or None if buffer is too small.
        """
        if len(self._buffer) < self.min_samples:
            return None

        inputs = np.stack([b[0] for b in self._buffer])
        targets = np.stack([b[1] for b in self._buffer])

        dataset = TensorDataset(
            torch.tensor(inputs, dtype=torch.float32),
            torch.tensor(targets, dtype=torch.float32),
        )
        loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=True)

        self.net.train()
        total_loss = 0.0
        count = 0
        for _ in range(n_epochs):
            for x_batch, y_batch in loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                self.optimizer.zero_grad()
                pred = self.net(x_batch)
                loss = self.criterion(pred, y_batch)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                count += 1

        self._train_steps += 1
        return total_loss / max(count, 1)

    @torch.no_grad()
    def predict(
        self,
        state: np.ndarray,       # [state_dim]
        batch_phi: np.ndarray,   # [B, phi_dim]
    ) -> Dict[str, float]:
        """
        Predict reward components for a (state, batch) pair.

        Returns
        -------
        dict with keys 'r_ood', 'r_cov', 'r_unc', 'total'
        (total uses reward weights from training data distribution).
        """
        input_vec = self._build_input(state, batch_phi)
        x = torch.tensor(input_vec[None], dtype=torch.float32).to(self.device)
        self.net.eval()
        out = self.net(x).squeeze(0).cpu().numpy()  # [3]
        return {
            "r_ood": float(out[0]),
            "r_cov": float(out[1]),
            "r_unc": float(out[2]),
        }

    def imagined_rollout(
        self,
        state: np.ndarray,
        candidate_phi: np.ndarray,  # [P, phi_dim]
        batch_size: int,
        n_rollouts: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Generate imagined rollouts by greedily assembling batches using
        the simulator's predicted reward as the objective.

        Each rollout greedily picks *batch_size* candidates (without replacement)
        maximising predicted r_ood + r_cov + r_unc.

        Returns
        -------
        List of dicts, each with keys 'batch_phi', 'reward_components'.
        """
        rollouts = []
        rng = np.random.default_rng()

        for _ in range(n_rollouts):
            pool_phi = candidate_phi.copy()
            selected_phi: List[np.ndarray] = []

            for _ in range(batch_size):
                if len(pool_phi) == 0:
                    break
                # Score each remaining candidate individually
                scores = np.array([
                    self._score_candidate(state, selected_phi, pool_phi[j])
                    for j in range(len(pool_phi))
                ])
                best = int(np.argmax(scores))
                selected_phi.append(pool_phi[best])
                pool_phi = np.delete(pool_phi, best, axis=0)

            if selected_phi:
                batch_phi_arr = np.stack(selected_phi)
                reward_comp = self.predict(state, batch_phi_arr)
                rollouts.append({
                    "batch_phi": batch_phi_arr,
                    "reward_components": reward_comp,
                })

        return rollouts

    def validate_drift(
        self,
        real_transitions: List[Tuple[np.ndarray, np.ndarray, Dict[str, float]]],
    ) -> Dict[str, float]:
        """
        Compute mean absolute error between simulator predictions and real rewards.

        Parameters
        ----------
        real_transitions : list of (state, batch_phi, reward_components).

        Returns
        -------
        dict with 'mae_ood', 'mae_cov', 'mae_unc'.
        """
        if not real_transitions:
            return {}

        mae = {"r_ood": 0.0, "r_cov": 0.0, "r_unc": 0.0}
        keys = list(mae.keys())

        for state, batch_phi, real_r in real_transitions:
            pred = self.predict(state, batch_phi)
            for k in keys:
                mae[k] += abs(pred[k] - real_r.get(k, 0.0))

        n = len(real_transitions)
        return {f"mae_{k[2:]}": v / n for k, v in mae.items()}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_input(self, state: np.ndarray, batch_phi: np.ndarray) -> np.ndarray:
        """Concatenate [state ; mean_phi ; max_phi]."""
        if len(batch_phi) == 0:
            mean_phi = np.zeros(batch_phi.shape[-1] if batch_phi.ndim > 1 else 0, dtype=np.float32)
            max_phi = mean_phi.copy()
        else:
            mean_phi = batch_phi.mean(axis=0)
            max_phi = batch_phi.max(axis=0)
        return np.concatenate([state, mean_phi, max_phi]).astype(np.float32)

    def _score_candidate(
        self,
        state: np.ndarray,
        selected_phi: List[np.ndarray],
        candidate_phi: np.ndarray,
    ) -> float:
        """Simulate reward if candidate is added to the current partial batch."""
        trial_phi = np.stack(selected_phi + [candidate_phi])
        r = self.predict(state, trial_phi)
        return r["r_ood"] + r["r_cov"] + r["r_unc"]
