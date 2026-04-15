"""
State computation for the active learning MDP.

The observation s_{t,b} is the concatenation of three blocks:

    s = [h_obs ; h_pool ; h_pb]

Block A  h_obs  (518-d with model_dim=256)
  - mean_pool and max_pool over labeled set embeddings    2*256
  - single_gene_coverage                                  1
  - gene_pair_coverage                                    1
  - pathway_coverage                                      1
  - id_val_mse, ood_val_mse, mean_pool_uncertainty        3

Block B  h_pool  (788-d)
  - mean_pool and max_pool over pool candidate embeddings 2*256
  - uncertainty_histogram  (10 bins)                      10
  - distance_histogram     (10 bins, min dist to D_t)     10
  - top_uncertainty_embedding  (mean embed of top-10%)    256

Block C  h_pb  (516-d)
  - mean_pool and max_pool over partial batch embeddings  2*256
  - mean_pairwise_distance within partial batch           1
  - new_gene_coverage_delta                               1
  - new_gene_pair_delta                                   1
  - selection_progress  (b-1)/B                           1
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple, Any

import numpy as np
import torch
import torch.nn.functional as F


class StateComputer:
    """
    Computes the concatenated state vector given the current AL state.

    All embedding tensors are expected on CPU; they are moved to the target
    device only during network forward passes (policy/value).
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Parameters
        ----------
        config : full config dict.
        """
        cfg_s = config["state"]
        cfg_al = config["active_learning"]
        self.embedding_dim: int = cfg_s["embedding_dim"]
        self.unc_bins: int = cfg_s["uncertainty_bins"]
        self.dist_bins: int = cfg_s["distance_bins"]
        self.top_frac: float = cfg_s["top_uncertainty_fraction"]
        self.batch_size: int = cfg_al["batch_size"]

        # Expected output dims (used for zero-padding when sets are empty)
        self.h_obs_dim: int = cfg_s["h_obs_dim"]
        self.h_pool_dim: int = cfg_s["h_pool_dim"]
        self.h_pb_dim: int = cfg_s["h_pb_dim"]
        self.state_dim: int = cfg_s["state_dim"]

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def compute(
        self,
        labeled_embeddings: np.ndarray,         # [|D_t|, D]
        pool_embeddings: np.ndarray,             # [|U_t|, D]
        pool_uncertainties: np.ndarray,          # [|U_t|]
        partial_batch_embeddings: np.ndarray,    # [b, D]  may be empty
        labeled_gene_sets: List[Set[str]],       # perturbed genes per labeled sample
        partial_batch_gene_sets: List[Set[str]], # perturbed genes per selected candidate
        all_single_genes: Set[str],              # all unique single genes seen so far
        all_gene_pairs: Set[frozenset],          # all unique gene pairs seen so far
        all_pathway_pairs: Set[frozenset],       # all unique pathway pairs seen so far
        pool_gene_sets: List[Set[str]],          # genes for each pool candidate
        pathway_map: Dict[str, int],             # gene -> pathway_id
        id_val_mse: float,
        ood_val_mse: float,
        selection_step: int,                     # b: 1-indexed step within round
    ) -> np.ndarray:
        """
        Compute full state vector.

        Returns
        -------
        np.ndarray, shape [state_dim]
        """
        h_obs = self._compute_h_obs(
            labeled_embeddings,
            labeled_gene_sets,
            all_single_genes,
            all_gene_pairs,
            all_pathway_pairs,
            pathway_map,
            id_val_mse,
            ood_val_mse,
            pool_uncertainties,
        )
        h_pool = self._compute_h_pool(
            pool_embeddings,
            pool_uncertainties,
            labeled_embeddings,
        )
        h_pb = self._compute_h_pb(
            partial_batch_embeddings,
            partial_batch_gene_sets,
            labeled_gene_sets,
            all_gene_pairs,
            selection_step,
        )
        state = np.concatenate([h_obs, h_pool, h_pb], axis=0)
        assert state.shape[0] == self.state_dim, (
            f"State dim mismatch: got {state.shape[0]}, expected {self.state_dim}"
        )
        return state.astype(np.float32)

    # ------------------------------------------------------------------
    # Block A: h_obs
    # ------------------------------------------------------------------

    def _compute_h_obs(
        self,
        labeled_embeddings: np.ndarray,
        labeled_gene_sets: List[Set[str]],
        all_single_genes: Set[str],
        all_gene_pairs: Set[frozenset],
        all_pathway_pairs: Set[frozenset],
        pathway_map: Dict[str, int],
        id_val_mse: float,
        ood_val_mse: float,
        pool_uncertainties: np.ndarray,
    ) -> np.ndarray:
        D = self.embedding_dim

        if len(labeled_embeddings) == 0:
            mean_pool = np.zeros(D, dtype=np.float32)
            max_pool = np.zeros(D, dtype=np.float32)
        else:
            mean_pool = labeled_embeddings.mean(axis=0)
            max_pool = labeled_embeddings.max(axis=0)

        # Coverage statistics
        single_cov, pair_cov, path_cov = self._coverage_stats(
            labeled_gene_sets, all_single_genes, all_gene_pairs,
            all_pathway_pairs, pathway_map,
        )

        mean_unc = float(pool_uncertainties.mean()) if len(pool_uncertainties) > 0 else 0.0

        h_obs = np.concatenate([
            mean_pool,           # 256
            max_pool,            # 256
            [single_cov,         # 1
             pair_cov,           # 1
             path_cov,           # 1
             id_val_mse,         # 1
             ood_val_mse,        # 1
             mean_unc],          # 1
        ])
        return h_obs.astype(np.float32)

    # ------------------------------------------------------------------
    # Block B: h_pool
    # ------------------------------------------------------------------

    def _compute_h_pool(
        self,
        pool_embeddings: np.ndarray,     # [P, D]
        pool_uncertainties: np.ndarray,  # [P]
        labeled_embeddings: np.ndarray,  # [L, D]
    ) -> np.ndarray:
        D = self.embedding_dim

        if len(pool_embeddings) == 0:
            return np.zeros(self.h_pool_dim, dtype=np.float32)

        mean_pool = pool_embeddings.mean(axis=0)   # [D]
        max_pool = pool_embeddings.max(axis=0)     # [D]

        # Uncertainty histogram
        unc_hist = self._histogram(pool_uncertainties, self.unc_bins)

        # Min-distance histogram from each pool candidate to D_t
        min_dists = self._min_cosine_distances_to_set(pool_embeddings, labeled_embeddings)
        dist_hist = self._histogram(min_dists, self.dist_bins)

        # Mean embedding of top-10% most uncertain candidates
        k = max(1, int(self.top_frac * len(pool_embeddings)))
        top_idx = np.argsort(pool_uncertainties)[-k:]
        top_unc_emb = pool_embeddings[top_idx].mean(axis=0)  # [D]

        h_pool = np.concatenate([
            mean_pool,     # 256
            max_pool,      # 256
            unc_hist,      # 10
            dist_hist,     # 10
            top_unc_emb,   # 256
        ])
        return h_pool.astype(np.float32)

    # ------------------------------------------------------------------
    # Block C: h_pb
    # ------------------------------------------------------------------

    def _compute_h_pb(
        self,
        partial_batch_embeddings: np.ndarray,    # [b, D]  may be empty
        partial_batch_gene_sets: List[Set[str]],
        labeled_gene_sets: List[Set[str]],
        existing_gene_pairs: Set[frozenset],
        selection_step: int,
    ) -> np.ndarray:
        D = self.embedding_dim
        b = len(partial_batch_embeddings)

        if b == 0:
            mean_pool = np.zeros(D, dtype=np.float32)
            max_pool = np.zeros(D, dtype=np.float32)
            mean_pw_dist = 0.0
            new_gene_cov_delta = 0.0
            new_pair_delta = 0.0
        else:
            mean_pool = partial_batch_embeddings.mean(axis=0)
            max_pool = partial_batch_embeddings.max(axis=0)
            mean_pw_dist = self._mean_pairwise_cosine_distance(partial_batch_embeddings)
            new_gene_cov_delta, new_pair_delta = self._coverage_delta(
                partial_batch_gene_sets, labeled_gene_sets, existing_gene_pairs
            )

        progress = (selection_step - 1) / max(self.batch_size, 1)

        h_pb = np.concatenate([
            mean_pool,          # 256
            max_pool,           # 256
            [mean_pw_dist,      # 1
             new_gene_cov_delta, # 1
             new_pair_delta,    # 1
             progress],         # 1
        ])
        return h_pb.astype(np.float32)

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _coverage_stats(
        gene_sets: List[Set[str]],
        all_single_genes: Set[str],
        all_gene_pairs: Set[frozenset],
        all_pathway_pairs: Set[frozenset],
        pathway_map: Dict[str, int],
    ) -> Tuple[float, float, float]:
        """Return (single_gene_coverage, gene_pair_coverage, pathway_coverage)."""
        if not all_single_genes:
            single_cov = 0.0
        else:
            seen_single = {g for gs in gene_sets for g in gs}
            single_cov = len(seen_single & all_single_genes) / len(all_single_genes)

        if not all_gene_pairs:
            pair_cov = 0.0
        else:
            seen_pairs: Set[frozenset] = set()
            for gs in gene_sets:
                genes = list(gs)
                for i in range(len(genes)):
                    for j in range(i + 1, len(genes)):
                        seen_pairs.add(frozenset({genes[i], genes[j]}))
            pair_cov = len(seen_pairs & all_gene_pairs) / len(all_gene_pairs)

        if not all_pathway_pairs:
            path_cov = 0.0
        else:
            seen_path_pairs: Set[frozenset] = set()
            for gs in gene_sets:
                pathways = {pathway_map.get(g, -1) for g in gs if g in pathway_map}
                pathways_list = [p for p in pathways if p >= 0]
                for i in range(len(pathways_list)):
                    for j in range(i + 1, len(pathways_list)):
                        seen_path_pairs.add(frozenset({pathways_list[i], pathways_list[j]}))
            path_cov = len(seen_path_pairs & all_pathway_pairs) / len(all_pathway_pairs)

        return single_cov, pair_cov, path_cov

    @staticmethod
    def _coverage_delta(
        partial_gene_sets: List[Set[str]],
        labeled_gene_sets: List[Set[str]],
        existing_pairs: Set[frozenset],
    ) -> Tuple[float, float]:
        """New gene and gene-pair coverage added by the partial batch."""
        labeled_genes = {g for gs in labeled_gene_sets for g in gs}
        new_genes = {g for gs in partial_gene_sets for g in gs} - labeled_genes

        new_pairs: Set[frozenset] = set()
        for gs in partial_gene_sets:
            genes = list(gs)
            for i in range(len(genes)):
                for j in range(i + 1, len(genes)):
                    fp = frozenset({genes[i], genes[j]})
                    if fp not in existing_pairs:
                        new_pairs.add(fp)

        return float(len(new_genes)), float(len(new_pairs))

    @staticmethod
    def _histogram(values: np.ndarray, n_bins: int) -> np.ndarray:
        """Normalised histogram of *values* into *n_bins* equal-width bins."""
        if len(values) == 0:
            return np.zeros(n_bins, dtype=np.float32)
        counts, _ = np.histogram(values, bins=n_bins)
        total = counts.sum()
        return (counts / max(total, 1)).astype(np.float32)

    @staticmethod
    def _min_cosine_distances_to_set(
        query: np.ndarray,   # [Q, D]
        ref: np.ndarray,     # [R, D]
    ) -> np.ndarray:
        """Min cosine distance from each query row to any row in ref. Shape [Q]."""
        if len(ref) == 0:
            return np.ones(len(query), dtype=np.float32)

        q = torch.tensor(query, dtype=torch.float32)
        r = torch.tensor(ref, dtype=torch.float32)
        q_norm = F.normalize(q, dim=-1)
        r_norm = F.normalize(r, dim=-1)
        # cosine similarity [Q, R]
        sim = q_norm @ r_norm.T
        min_sim = sim.max(dim=-1).values  # highest similarity = smallest distance
        dist = 1.0 - min_sim.numpy()
        return dist.astype(np.float32)

    @staticmethod
    def _mean_pairwise_cosine_distance(embeddings: np.ndarray) -> float:
        """Mean pairwise cosine distance within *embeddings* [N, D]."""
        N = len(embeddings)
        if N < 2:
            return 0.0
        e = torch.tensor(embeddings, dtype=torch.float32)
        e_norm = F.normalize(e, dim=-1)
        sim = e_norm @ e_norm.T  # [N, N]
        # upper triangle (excluding diagonal)
        idx = torch.triu_indices(N, N, offset=1)
        sims = sim[idx[0], idx[1]]
        return float((1.0 - sims).mean().item())
