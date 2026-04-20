"""
Stage 1 & 2: Teacher rollout generation + Behaviour Cloning.

Teacher policy
--------------
Greedy sequential selection using heuristic score:
    score(x) = u(x) + 0.2 * d(x, D_t) + 0.2 * d(x, B_partial)

Generates N_teacher=20 episodes of full active-learning runs.
Stores (state, action) pairs — no reward needed.

Behaviour Cloning
-----------------
L_BC = -Σ log π_θ(a_teacher | s)
Train for 20 epochs with Adam lr=1e-3 until cross-entropy loss stabilises.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from environment.al_env import ALEnvironment
from policy.network import PolicyNetwork, ValueNetwork
from policy.features import FeatureExtractor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Teacher policy
# ---------------------------------------------------------------------------

def teacher_score(
    pool_uncertainties: np.ndarray,   # [P]
    d_dt: np.ndarray,                 # [P] min cosine dist to D_t
    d_batch: np.ndarray,              # [P] min cosine dist to current partial batch
    w_unc: float = 1.0,
    w_dist: float = 0.2,
    w_batch: float = 0.2,
) -> np.ndarray:
    """
    Heuristic teacher scoring function.

    score(x) = w_unc * u(x) + w_dist * d(x, D_t) + w_batch * d(x, B_partial)
    """
    return w_unc * pool_uncertainties + w_dist * d_dt + w_batch * d_batch


def run_teacher_episode(
    env: ALEnvironment,
    config: Dict[str, Any],
) -> List[Tuple[np.ndarray, np.ndarray, int]]:
    """
    Run one episode with the teacher heuristic policy.

    Returns
    -------
    List of (state, query_features [P, query_dim], action) tuples.
    One tuple per within-round selection step.
    """
    cfg_bc = config["bc"]
    w_unc = cfg_bc["teacher_uncertainty_weight"]
    w_dist = cfg_bc["teacher_distance_weight"]
    w_batch = cfg_bc["teacher_batch_distance_weight"]

    feature_extractor = FeatureExtractor(config)
    transitions: List[Tuple[np.ndarray, np.ndarray, int]] = []

    env.freeze_predictor = True
    state = env.reset()
    done = False

    while not done:
        pool_embs = env.pool_embeddings       # [P, D]
        pool_uncs = env.pool_uncertainties    # [P]
        labeled_embs = env.labeled_embeddings # [L, D]
        partial_embs = env._embed_conditions(env._current_batch_conds)

        query_feats = feature_extractor.build_query_features(
            pool_embeddings=pool_embs,
            pool_uncertainties=pool_uncs,
            labeled_embeddings=labeled_embs,
            partial_batch_embeddings=partial_embs,
        )  # [P, query_dim]

        from policy.features import FeatureExtractor as _FE
        d_dt = _FE._min_cosine_distances(pool_embs, labeled_embs)
        d_batch = _FE._min_cosine_distances(pool_embs, partial_embs)

        scores = teacher_score(pool_uncs, d_dt, d_batch, w_unc, w_dist, w_batch)

        # Greedy: select candidate with highest score
        action = int(np.argmax(scores))

        # Store (state, query_features, action)
        transitions.append((state.copy(), query_feats.copy(), action))

        state, _, done, _ = env.step(action)

    return transitions


def generate_teacher_rollouts(
    env: ALEnvironment,
    config: Dict[str, Any],
) -> List[Tuple[np.ndarray, np.ndarray, int]]:
    """
    Generate N_teacher episodes using the teacher policy.

    Returns flat list of (state, query_features, action) transitions.
    """
    n_episodes = config["bc"]["n_teacher_episodes"]
    all_transitions: List[Tuple[np.ndarray, np.ndarray, int]] = []

    for ep in range(n_episodes):
        logger.info(f"Teacher rollout {ep + 1}/{n_episodes}")
        transitions = run_teacher_episode(env, config)
        all_transitions.extend(transitions)
        logger.info(f"  Collected {len(transitions)} steps")

    env.freeze_predictor = False  # restore for PPO training
    logger.info(f"Total teacher transitions: {len(all_transitions)}")
    return all_transitions


# ---------------------------------------------------------------------------
# Behaviour Cloning training
# ---------------------------------------------------------------------------

class BCDataset(torch.utils.data.Dataset):
    """Dataset of (state_parts, query_features, action) for BC training."""

    def __init__(
        self,
        transitions: List[Tuple[np.ndarray, np.ndarray, int]],
        config: Dict[str, Any],
    ):
        cfg_s = config["state"]
        h_obs_dim = cfg_s["h_obs_dim"]
        h_pool_dim = cfg_s["h_pool_dim"]
        h_pb_dim = cfg_s["h_pb_dim"]

        # All transitions may have different pool sizes (P varies).
        # We pad query_features to max P, store mask.
        max_P = max(q.shape[0] for _, q, _ in transitions)

        states = np.stack([t[0] for t in transitions])         # [N, state_dim]
        actions = np.array([t[2] for t in transitions], dtype=np.int64)  # [N]

        # Pad query features along pool dim
        q_dim = transitions[0][1].shape[1]
        query_padded = np.zeros((len(transitions), max_P, q_dim), dtype=np.float32)
        masks = np.ones((len(transitions), max_P), dtype=bool)   # True = padded = mask out
        for i, (_, qf, _) in enumerate(transitions):
            P = qf.shape[0]
            query_padded[i, :P] = qf
            masks[i, :P] = False

        self.h_obs = torch.tensor(states[:, :h_obs_dim], dtype=torch.float32)
        self.h_pool = torch.tensor(states[:, h_obs_dim:h_obs_dim + h_pool_dim], dtype=torch.float32)
        self.h_pb = torch.tensor(states[:, h_obs_dim + h_pool_dim:], dtype=torch.float32)
        self.query = torch.tensor(query_padded, dtype=torch.float32)
        self.mask = torch.tensor(masks, dtype=torch.bool)
        self.actions = torch.tensor(actions, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.actions)

    def __getitem__(self, idx):
        return (
            self.h_obs[idx],
            self.h_pool[idx],
            self.h_pb[idx],
            self.query[idx],
            self.mask[idx],
            self.actions[idx],
        )


def train_bc(
    policy: PolicyNetwork,
    transitions: List[Tuple[np.ndarray, np.ndarray, int]],
    config: Dict[str, Any],
    device: str = "cpu",
) -> PolicyNetwork:
    """
    Train policy with behaviour cloning (cross-entropy imitation loss).

    Parameters
    ----------
    policy      : PolicyNetwork to train (modified in-place).
    transitions : list of (state, query_features, action) from teacher rollouts.
    config      : full config dict.
    device      : torch device string.

    Returns
    -------
    Trained PolicyNetwork.
    """
    cfg_bc = config["bc"]
    epochs = cfg_bc["epochs"]
    lr = cfg_bc["lr"]

    dataset = BCDataset(transitions, config)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    policy = policy.to(device)
    policy.train()

    for epoch in range(epochs):
        total_loss = 0.0
        count = 0
        for h_obs, h_pool, h_pb, query, mask, actions in loader:
            h_obs = h_obs.to(device)
            h_pool = h_pool.to(device)
            h_pb = h_pb.to(device)
            query = query.to(device)
            mask = mask.to(device)
            actions = actions.to(device)

            optimizer.zero_grad()
            logits = policy(h_obs, h_pool, h_pb, query, mask)  # [B, P]
            loss = criterion(logits, actions)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(actions)
            count += len(actions)

        avg_loss = total_loss / max(count, 1)
        logger.info(f"BC Epoch {epoch + 1:03d} | Cross-Entropy Loss: {avg_loss:.4f}")

    return policy
