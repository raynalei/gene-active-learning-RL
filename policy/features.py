"""
Candidate feature construction for the policy network.

For each candidate x in the pool, we build two things:

1. phi(x) — full feature vector used by the teacher / simulator
   phi(x) = [h_obs ; h_pool ; h_pb ; z_x ; u(x) ; d(x, D_t) ; d(x, B_partial)]

2. query(x) — lightweight query for cross-attention
   query(x) = [z_x ; u(x) ; d(x, D_t) ; d(x, B_partial)]
   Dimension: embedding_dim + 3  (defaults to 256 + 3 = 259)
"""

from __future__ import annotations

from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F


class FeatureExtractor:
    """
    Constructs candidate-level features given the current state and pool.

    All inputs are numpy arrays; outputs are also numpy (policy network
    handles conversion to tensors).
    """

    def __init__(self, config: dict):
        self.embedding_dim: int = config["state"]["embedding_dim"]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_candidate_features(
        self,
        pool_embeddings: np.ndarray,         # [P, D]
        pool_uncertainties: np.ndarray,      # [P]
        labeled_embeddings: np.ndarray,      # [L, D]
        partial_batch_embeddings: np.ndarray, # [b, D]  may be empty
        state_vector: np.ndarray,            # [state_dim]
    ) -> np.ndarray:
        """
        Build phi(x) for each candidate in the pool.

        Returns
        -------
        np.ndarray, shape [P, phi_dim]
        where phi_dim = state_dim + embedding_dim + 3
        """
        P = len(pool_embeddings)
        d_dt = self._min_cosine_distances(pool_embeddings, labeled_embeddings)      # [P]
        d_b = self._min_cosine_distances(pool_embeddings, partial_batch_embeddings) # [P]

        # state_vector repeated for each candidate
        state_rep = np.tile(state_vector, (P, 1))  # [P, state_dim]

        phi = np.concatenate([
            state_rep,                          # [P, state_dim]
            pool_embeddings,                    # [P, D]
            pool_uncertainties[:, None],        # [P, 1]
            d_dt[:, None],                      # [P, 1]
            d_b[:, None],                       # [P, 1]
        ], axis=-1)
        return phi.astype(np.float32)

    def build_query_features(
        self,
        pool_embeddings: np.ndarray,          # [P, D]
        pool_uncertainties: np.ndarray,       # [P]
        labeled_embeddings: np.ndarray,       # [L, D]
        partial_batch_embeddings: np.ndarray, # [b, D]
    ) -> np.ndarray:
        """
        Build cross-attention query features for each candidate.

        query(x) = [z_x ; u(x) ; d(x, D_t) ; d(x, B_partial)]
        Dimension = embedding_dim + 3  (259 by default).

        Returns
        -------
        np.ndarray, shape [P, embedding_dim + 3]
        """
        d_dt = self._min_cosine_distances(pool_embeddings, labeled_embeddings)
        d_b = self._min_cosine_distances(pool_embeddings, partial_batch_embeddings)

        query = np.concatenate([
            pool_embeddings,               # [P, D]
            pool_uncertainties[:, None],   # [P, 1]
            d_dt[:, None],                 # [P, 1]
            d_b[:, None],                  # [P, 1]
        ], axis=-1)
        return query.astype(np.float32)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _min_cosine_distances(
        query: np.ndarray,   # [Q, D]
        ref: np.ndarray,     # [R, D]
    ) -> np.ndarray:
        """
        Minimum cosine distance from each query row to any row in ref.
        Returns shape [Q].  If ref is empty, returns ones (max distance).
        """
        if len(ref) == 0:
            return np.ones(len(query), dtype=np.float32)

        q = torch.tensor(query, dtype=torch.float32)
        r = torch.tensor(ref, dtype=torch.float32)
        q_norm = F.normalize(q, dim=-1)
        r_norm = F.normalize(r, dim=-1)
        sim = q_norm @ r_norm.T           # [Q, R]
        max_sim = sim.max(dim=-1).values  # [Q]
        return (1.0 - max_sim.numpy()).astype(np.float32)
