"""
Stage 3: PPO fine-tuning with Dyna model-based augmentation.

Training loop (per PPO iteration)
----------------------------------
1. Collect 1 real rollout from ALEnvironment → store (s, a, r, log_p, V) in buffer.
2. Update simulator with new real (s_t, B_t, r_t) triple.
3. Generate Dyna ratio × simulated rollouts from the learned simulator.
4. Compute GAE advantages over the combined buffer.
5. Run PPO update for ppo_epochs_per_update mini-batch epochs.
6. Every full_retrain_every real rounds: predictor full retrain (handled in env).
7. Log: OOD MSE, ID MSE, coverage, batch diversity, simulator prediction error.

PPO config
----------
clip_epsilon = 0.2, lr = 3e-4, entropy_coef = 0.01, value_coef = 0.5,
gamma = 1.0, gae_lambda = 0.95, ppo_epochs = 4, batch_size = 64.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from environment.al_env import ALEnvironment
from policy.network import PolicyNetwork, ValueNetwork
from policy.features import FeatureExtractor
from simulator.batch_simulator import BatchSimulator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Experience buffer
# ---------------------------------------------------------------------------

@dataclass
class Transition:
    """One step of collected experience."""
    h_obs: np.ndarray
    h_pool: np.ndarray
    h_pb: np.ndarray
    query_features: np.ndarray   # [P, query_dim] — policy cross-attention input
    phi_features: np.ndarray     # [P, phi_dim]   — Dyna simulator input
    action: int
    log_prob: float
    reward: float
    value: float
    mask: np.ndarray             # [P] bool, True = masked out


@dataclass
class RolloutBuffer:
    """Flat buffer of transitions from real + imagined rollouts."""
    transitions: List[Transition] = field(default_factory=list)

    def clear(self) -> None:
        self.transitions.clear()

    def __len__(self) -> int:
        return len(self.transitions)


# ---------------------------------------------------------------------------
# GAE advantage computation
# ---------------------------------------------------------------------------

def compute_gae(
    rewards: List[float],
    values: List[float],
    last_value: float,
    gamma: float,
    gae_lambda: float,
) -> Tuple[List[float], List[float]]:
    """
    Generalised Advantage Estimation.

    Returns (advantages, returns) as lists aligned with the input rewards.
    """
    advantages = []
    gae = 0.0
    next_value = last_value

    for r, v in zip(reversed(rewards), reversed(values)):
        delta = r + gamma * next_value - v
        gae = delta + gamma * gae_lambda * gae
        advantages.insert(0, gae)
        next_value = v

    returns = [a + v for a, v in zip(advantages, values)]
    return advantages, returns


# ---------------------------------------------------------------------------
# PPO Trainer
# ---------------------------------------------------------------------------

class PPOTrainer:
    """
    PPO fine-tuning with Dyna model-based rollout augmentation.

    Parameters
    ----------
    env       : active learning environment.
    policy    : cross-attention policy network.
    value_net : value baseline network.
    simulator : learned reward simulator (Dyna).
    config    : full config dict.
    device    : torch device string.
    """

    def __init__(
        self,
        env: ALEnvironment,
        policy: PolicyNetwork,
        value_net: ValueNetwork,
        simulator: BatchSimulator,
        config: Dict[str, Any],
        device: str = "cpu",
    ):
        self.env = env
        self.policy = policy.to(device)
        self.value_net = value_net.to(device)
        self.simulator = simulator
        self.config = config
        self.device = device

        cfg = config["ppo"]
        self.clip_eps: float = cfg["clip_epsilon"]
        self.lr: float = cfg["lr"]
        self.entropy_coef: float = cfg["entropy_coef"]
        self.value_coef: float = cfg["value_coef"]
        self.gamma: float = cfg["gamma"]
        self.gae_lambda: float = cfg["gae_lambda"]
        self.ppo_epochs: int = cfg["ppo_epochs_per_update"]
        self.ppo_batch: int = cfg["ppo_batch_size"]
        self.dyna_early: int = cfg["dyna_ratio_early"]
        self.dyna_late: int = cfg["dyna_ratio_late"]
        self.dyna_transition: int = cfg["dyna_transition_step"]
        self.max_grad_norm: float = cfg["max_grad_norm"]

        cfg_s = config["state"]
        self.h_obs_dim = cfg_s["h_obs_dim"]
        self.h_pool_dim = cfg_s["h_pool_dim"]
        self.h_pb_dim = cfg_s["h_pb_dim"]
        self.state_dim = cfg_s["state_dim"]

        self.optimizer = torch.optim.Adam(
            list(self.policy.parameters()) + list(self.value_net.parameters()),
            lr=self.lr,
        )
        self.buffer = RolloutBuffer()
        self.feature_extractor = FeatureExtractor(config)
        self._ppo_iter: int = 0

    # ------------------------------------------------------------------
    # Main training entry point
    # ------------------------------------------------------------------

    def save_checkpoint(
        self,
        path: str,
        logs: List[Dict[str, float]],
        round_log: List[Dict[str, float]],
    ) -> None:
        """Save full training state to *path*."""
        ensemble = self.env.ensemble
        torch.save(
            {
                "ppo_iter": self._ppo_iter,
                "policy": self.policy.state_dict(),
                "value_net": self.value_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "simulator_net": self.simulator.net.state_dict(),
                "simulator_buffer": list(self.simulator._buffer),
                "simulator_train_steps": self.simulator._train_steps,
                "ensemble": [m.state_dict() for m in ensemble.members],
                "logs": logs,
                "round_log": round_log,
            },
            path,
        )
        logger.info(f"Checkpoint saved → {path}")

    def load_checkpoint(self, path: str) -> tuple[List[Dict[str, float]], List[Dict[str, float]]]:
        """
        Restore training state from *path*.

        Returns the accumulated (logs, round_log) so training can resume appending.
        """
        ckpt = torch.load(path, map_location=self.device)
        self._ppo_iter = ckpt["ppo_iter"]
        self.policy.load_state_dict(ckpt["policy"])
        self.value_net.load_state_dict(ckpt["value_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.simulator.net.load_state_dict(ckpt["simulator_net"])
        from collections import deque
        self.simulator._buffer = deque(ckpt["simulator_buffer"], maxlen=self.simulator.buffer_size)
        self.simulator._train_steps = ckpt["simulator_train_steps"]
        ensemble = self.env.ensemble
        for member, sd in zip(ensemble.members, ckpt["ensemble"]):
            member.load_state_dict(sd)
        logger.info(f"Resumed from checkpoint {path} at iter {self._ppo_iter}")
        return ckpt["logs"], ckpt["round_log"]

    def train(
        self,
        n_iterations: int,
        checkpoint_every: int = 5,
        output_dir: Optional[str] = None,
    ) -> tuple[List[Dict[str, float]], List[Dict[str, float]]]:
        """
        Run *n_iterations* PPO iterations.

        Each iteration:
          1. Collect 1 real episode.
          2. Update simulator.
          3. Generate Dyna simulated transitions.
          4. Compute GAE.
          5. PPO update.

        Parameters
        ----------
        n_iterations     : total number of PPO iterations to run.
        checkpoint_every : save a checkpoint every this many iterations (0 = disabled).
        output_dir       : directory to write checkpoints; required if checkpoint_every > 0.

        Returns
        -------
        logs         : per-PPO-iteration log dicts.
        round_log    : per-AL-round metrics (num_labeled_cells, test_mse, ood_val_mse);
                       comparable column-for-column with random_sample.py's CSV output.
        """
        logs: List[Dict[str, float]] = []
        round_log: List[Dict[str, float]] = []

        for _ in range(n_iterations):
            self.buffer.clear()

            # 1. Real rollout
            real_transitions, real_sim_data, ep_info, round_data = self._collect_real_rollout()
            round_log.extend(round_data)
            self.buffer.transitions.extend(real_transitions)

            # 2. Update simulator with new real data
            for state, phi_batch, r_comp in real_sim_data:
                self.simulator.add_transition(state, phi_batch, r_comp)
            sim_loss = self.simulator.update()

            # 3. Dyna simulated rollouts
            dyna_ratio = self.dyna_early if self._ppo_iter < self.dyna_transition else self.dyna_late
            if len(self.simulator._buffer) >= self.simulator.min_samples:
                simulated = self._generate_dyna_transitions(dyna_ratio, real_transitions)
                self.buffer.transitions.extend(simulated)

            # 4. Compute GAE over combined buffer
            self._compute_advantages()

            # 5. PPO update
            ppo_stats = self._ppo_update()

            # 6. Drift validation every 10 iters
            drift_stats = {}
            if self._ppo_iter % 10 == 0 and real_sim_data:
                drift_stats = self.simulator.validate_drift(
                    [(s, phi, r) for s, phi, r in real_sim_data]
                )

            log = {
                "ppo_iter": self._ppo_iter,
                "ood_val_mse": ep_info.get("ood_val_mse", 0.0),
                "id_val_mse": ep_info.get("id_val_mse", 0.0),
                "test_mse": ep_info.get("test_mse", 0.0),
                "num_labeled_cells": ep_info.get("num_labeled_cells", 0),
                "sim_loss": sim_loss or 0.0,
                "buffer_size": len(self.buffer),
                **ppo_stats,
                **drift_stats,
            }
            logs.append(log)

            logger.info(
                f"PPO iter {self._ppo_iter:04d} | "
                f"OOD MSE: {log['ood_val_mse']:.4f} | "
                f"ID MSE: {log['id_val_mse']:.4f} | "
                f"PolicyLoss: {ppo_stats.get('policy_loss', 0):.4f} | "
                f"ValueLoss: {ppo_stats.get('value_loss', 0):.4f}"
            )

            if checkpoint_every > 0 and output_dir and (self._ppo_iter + 1) % checkpoint_every == 0:
                ckpt_path = os.path.join(output_dir, f"checkpoint_iter_{self._ppo_iter:04d}.pt")
                self.save_checkpoint(ckpt_path, logs, round_log)

            self._ppo_iter += 1

        return logs, round_log

    # ------------------------------------------------------------------
    # Real rollout collection
    # ------------------------------------------------------------------

    def _collect_real_rollout(
        self,
    ) -> Tuple[
        List[Transition],
        List[Tuple[np.ndarray, np.ndarray, Dict[str, float]]],
        Dict[str, Any],
        List[Dict[str, float]],
    ]:
        """
        Run one full episode in the real environment.

        Returns
        -------
        transitions : list of Transition (one per selection step).
        sim_data    : list of (state, phi_batch, reward_components) per round.
        info        : last episode info dict.
        round_data  : per-AL-round metrics (num_labeled_cells, test_mse, ood_val_mse).
        """
        state = self.env.reset()
        transitions: List[Transition] = []
        sim_data: List[Tuple] = []
        info: Dict[str, Any] = {}
        round_data: List[Dict[str, float]] = []

        round_step_states: List[np.ndarray] = []
        round_phi: List[np.ndarray] = []
        done = False

        while not done:
            query_feats, phi_feats = self.env.get_candidate_features(state)
            P = query_feats.shape[0]

            if P == 0:
                break

            # Convert state to tensor components
            h_obs_t, h_pool_t, h_pb_t, q_t, mask_t = self._state_to_tensors(
                state, query_feats
            )

            # Policy action
            with torch.no_grad():
                action_t, log_prob_t = self.policy.act(
                    h_obs_t, h_pool_t, h_pb_t, q_t, mask_t, greedy=False
                )
                value_t = self.value_net(
                    torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
                )

            action = int(action_t.item())
            log_prob = float(log_prob_t.item())
            value = float(value_t.item())

            transitions.append(Transition(
                h_obs=state[:self.h_obs_dim].copy(),
                h_pool=state[self.h_obs_dim:self.h_obs_dim + self.h_pool_dim].copy(),
                h_pb=state[self.h_obs_dim + self.h_pool_dim:].copy(),
                query_features=query_feats.copy(),
                phi_features=phi_feats.copy(),
                action=action,
                log_prob=log_prob,
                reward=0.0,  # filled in at end of round
                value=value,
                mask=np.zeros(P, dtype=bool),
            ))

            round_step_states.append(state.copy())
            round_phi.append(phi_feats[action].copy() if len(phi_feats) > 0 else np.zeros(phi_feats.shape[-1]))

            next_state, reward, done, info = self.env.step(action)

            # End of round: assign reward to all steps in this round
            is_end_of_round = (reward != 0.0) or done
            if is_end_of_round:
                n_steps = self.env.batch_size
                # Assign terminal reward to last step, 0 to others
                for i in range(-n_steps, 0):
                    if i == -1:
                        transitions[i].reward = reward
                    else:
                        transitions[i].reward = 0.0

                # Simulator data: (initial state, phi of selected batch, reward components)
                if round_step_states:
                    phi_batch = np.stack(round_phi[-n_steps:]) if len(round_phi) >= n_steps else np.stack(round_phi)
                    r_comp = info.get("reward_components", {})
                    sim_data.append((round_step_states[-n_steps], phi_batch, r_comp))

                round_step_states.clear()
                round_phi.clear()

                logger.info(
                    f"  [iter {self._ppo_iter}] round {info.get('round', '?')} done | "
                    f"labeled={info.get('num_labeled_conds', '?')} conds | "
                    f"reward={reward:.4f}"
                )

                # Collect per-round metrics for comparison with random baseline
                if info:
                    round_data.append({
                        "ppo_iter": self._ppo_iter,
                        "round": info.get("round", -1),
                        "num_labeled_cells": info.get("num_labeled_cells", 0),
                        "num_labeled_conds": info.get("num_labeled_conds", 0),
                        "test_mse": info.get("test_mse", 0.0),
                        "ood_val_mse": info.get("ood_val_mse", 0.0),
                        "id_val_mse": info.get("id_val_mse", 0.0),
                    })

            state = next_state

        return transitions, sim_data, info, round_data

    # ------------------------------------------------------------------
    # Dyna simulated transitions
    # ------------------------------------------------------------------

    def _generate_dyna_transitions(
        self,
        n_simulated: int,
        real_transitions: List[Transition],
    ) -> List[Transition]:
        """
        Generate imagined transitions by re-sampling real states and using
        the simulator to predict rewards for random batch selections.
        """
        if not real_transitions:
            return []

        imagined: List[Transition] = []
        rng = np.random.default_rng()

        for _ in range(n_simulated):
            # Sample a real state as starting point
            t = real_transitions[int(rng.integers(len(real_transitions)))]
            state = np.concatenate([t.h_obs, t.h_pool, t.h_pb])

            # Use simulator to predict reward for the same action
            phi_single = t.phi_features[t.action:t.action + 1]  # [1, phi_dim]
            sim_r = self.simulator.predict(state, phi_single)

            # Synthesise reward (weighted sum matching RewardComputer weights)
            cfg_r = self.config["reward"]
            reward = (
                cfg_r["w_ood"] * sim_r["r_ood"]
                + cfg_r["w_cov"] * sim_r["r_cov"]
                + cfg_r["w_unc"] * sim_r["r_unc"]
            )

            imagined.append(Transition(
                h_obs=t.h_obs.copy(),
                h_pool=t.h_pool.copy(),
                h_pb=t.h_pb.copy(),
                query_features=t.query_features.copy(),
                phi_features=t.phi_features.copy(),
                action=t.action,
                log_prob=t.log_prob,
                reward=reward,
                value=t.value,
                mask=t.mask.copy(),
            ))

        return imagined

    # ------------------------------------------------------------------
    # GAE over buffer
    # ------------------------------------------------------------------

    def _compute_advantages(self) -> None:
        """Compute and store GAE advantages in-place on buffer transitions."""
        rewards = [t.reward for t in self.buffer.transitions]
        values = [t.value for t in self.buffer.transitions]
        advantages, returns = compute_gae(
            rewards, values,
            last_value=0.0,  # terminal
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )
        for t, adv, ret in zip(self.buffer.transitions, advantages, returns):
            t._advantage = adv
            t._return = ret

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def _ppo_update(self) -> Dict[str, float]:
        """Run ppo_epochs mini-batch updates over the buffer."""
        transitions = self.buffer.transitions
        if not transitions:
            return {}

        # Determine max pool size for padding
        max_P = max(t.query_features.shape[0] for t in transitions)
        q_dim = transitions[0].query_features.shape[1]

        # Build padded tensors
        N = len(transitions)
        h_obs_arr = np.stack([t.h_obs for t in transitions])
        h_pool_arr = np.stack([t.h_pool for t in transitions])
        h_pb_arr = np.stack([t.h_pb for t in transitions])
        query_arr = np.zeros((N, max_P, q_dim), dtype=np.float32)
        mask_arr = np.ones((N, max_P), dtype=bool)
        actions_arr = np.array([t.action for t in transitions], dtype=np.int64)
        old_log_probs_arr = np.array([t.log_prob for t in transitions], dtype=np.float32)
        advantages_arr = np.array([t._advantage for t in transitions], dtype=np.float32)
        returns_arr = np.array([t._return for t in transitions], dtype=np.float32)

        for i, t in enumerate(transitions):
            P = t.query_features.shape[0]
            query_arr[i, :P] = t.query_features
            mask_arr[i, :P] = False

        # Normalise advantages
        adv_mean = advantages_arr.mean()
        adv_std = advantages_arr.std() + 1e-8
        advantages_arr = (advantages_arr - adv_mean) / adv_std

        dataset = TensorDataset(
            torch.tensor(h_obs_arr, dtype=torch.float32),
            torch.tensor(h_pool_arr, dtype=torch.float32),
            torch.tensor(h_pb_arr, dtype=torch.float32),
            torch.tensor(query_arr, dtype=torch.float32),
            torch.tensor(mask_arr, dtype=torch.bool),
            torch.tensor(actions_arr, dtype=torch.long),
            torch.tensor(old_log_probs_arr, dtype=torch.float32),
            torch.tensor(advantages_arr, dtype=torch.float32),
            torch.tensor(returns_arr, dtype=torch.float32),
        )
        loader = DataLoader(dataset, batch_size=self.ppo_batch, shuffle=True)

        total_p_loss = 0.0
        total_v_loss = 0.0
        total_entropy = 0.0
        count = 0

        self.policy.train()
        self.value_net.train()

        for _ in range(self.ppo_epochs):
            for batch in loader:
                (h_obs_b, h_pool_b, h_pb_b, query_b, mask_b,
                 acts_b, old_lp_b, adv_b, ret_b) = [x.to(self.device) for x in batch]

                state_b = torch.cat([h_obs_b, h_pool_b, h_pb_b], dim=-1)

                log_probs, entropy = self.policy.log_prob_and_entropy(
                    h_obs_b, h_pool_b, h_pb_b, query_b, acts_b, mask_b
                )
                values = self.value_net(state_b)

                # PPO clipped objective
                ratio = torch.exp(log_probs - old_lp_b)
                obj1 = ratio * adv_b
                obj2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_b
                policy_loss = -torch.min(obj1, obj2).mean()

                # Value loss
                value_loss = F.mse_loss(values, ret_b)

                # Total loss
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy.mean()
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.policy.parameters()) + list(self.value_net.parameters()),
                    self.max_grad_norm,
                )
                self.optimizer.step()

                total_p_loss += policy_loss.item()
                total_v_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                count += 1

        n = max(count, 1)
        return {
            "policy_loss": total_p_loss / n,
            "value_loss": total_v_loss / n,
            "entropy": total_entropy / n,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _state_to_tensors(
        self,
        state: np.ndarray,
        query_feats: np.ndarray,
    ) -> Tuple[torch.Tensor, ...]:
        """Split state vector into h_obs, h_pool, h_pb tensors + query + mask."""
        ho = torch.tensor(state[:self.h_obs_dim], dtype=torch.float32).unsqueeze(0).to(self.device)
        hp = torch.tensor(
            state[self.h_obs_dim:self.h_obs_dim + self.h_pool_dim],
            dtype=torch.float32,
        ).unsqueeze(0).to(self.device)
        hb = torch.tensor(state[self.h_obs_dim + self.h_pool_dim:], dtype=torch.float32).unsqueeze(0).to(self.device)
        q = torch.tensor(query_feats, dtype=torch.float32).unsqueeze(0).to(self.device)
        mask = torch.zeros(1, query_feats.shape[0], dtype=torch.bool).to(self.device)
        return ho, hp, hb, q, mask
