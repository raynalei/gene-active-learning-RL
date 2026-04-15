"""
Multi-component reward for the active learning MDP.

r_t = w_ood * r_OOD + w_cov * r_cov + w_red * r_red + w_unc * r_unc + w_des * r_DES

Components
----------
r_OOD  OOD_val_MSE_before - OOD_val_MSE_after
         Positive when the selected batch improves OOD generalisation.

r_cov  delta_gene_pair_coverage + 0.5 * delta_pathway_coverage
         Encourages exploring diverse gene combinations.

r_red  -mean pairwise cosine similarity within the selected batch
         Penalises redundant / repetitive selections (range [-1, 0]).

r_unc  mean_pool_uncertainty_before - mean_pool_uncertainty_after
         Encourages resolving model uncertainty.

r_DES  DES_after - DES_before
         Differential Expression Score improvement:
         DES = (1/|T|) * sum_{p in T} |G_true(p) ∩ G_pred(p)| / |G_true(p)|
         where G_true(p) and G_pred(p) are the sets of differentially expressed
         genes under perturbation p in observed and predicted data respectively.
"""

from __future__ import annotations

from typing import Dict, Any, List, Set

import numpy as np
import torch
import torch.nn.functional as F


class RewardComputer:
    """
    Stateless reward computation.  Call compute() once per round, after the
    predictor has been updated and new MSE / coverage values are available.
    """

    def __init__(self, config: Dict[str, Any]):
        cfg = config["reward"]
        self.w_ood: float = cfg["w_ood"]
        self.w_cov: float = cfg["w_cov"]
        self.w_red: float = cfg["w_red"]
        self.w_unc: float = cfg["w_unc"]
        self.w_des: float = cfg["w_des"]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        ood_mse_before: float,
        ood_mse_after: float,
        gene_pair_coverage_before: float,
        gene_pair_coverage_after: float,
        pathway_coverage_before: float,
        pathway_coverage_after: float,
        batch_embeddings: np.ndarray,       # [B, D]
        pool_uncertainty_before: float,
        pool_uncertainty_after: float,
        des_before: float,
        des_after: float,
    ) -> Dict[str, float]:
        """
        Compute the full reward and its decomposition.

        Parameters
        ----------
        ood_mse_before / after      : OOD validation MSE before/after predictor update.
        gene_pair_coverage_before/after : fraction of unique gene pairs seen.
        pathway_coverage_before/after   : fraction of unique pathway pairs seen.
        batch_embeddings            : embeddings of the selected batch [B, D].
        pool_uncertainty_before/after   : mean uncertainty over pool before/after.
        des_before / after          : Differential Expression Score before/after update.

        Returns
        -------
        dict with keys: 'total', 'r_ood', 'r_cov', 'r_red', 'r_unc', 'r_des'
        """
        r_ood = self._r_ood(ood_mse_before, ood_mse_after)
        r_cov = self._r_cov(
            gene_pair_coverage_before, gene_pair_coverage_after,
            pathway_coverage_before, pathway_coverage_after,
        )
        r_red = self._r_red(batch_embeddings)
        r_unc = self._r_unc(pool_uncertainty_before, pool_uncertainty_after)
        r_des = self._r_des(des_before, des_after)

        total = (
            self.w_ood * r_ood
            + self.w_cov * r_cov
            + self.w_red * r_red
            + self.w_unc * r_unc
            + self.w_des * r_des
        )

        return {
            "total": float(total),
            "r_ood": float(r_ood),
            "r_cov": float(r_cov),
            "r_red": float(r_red),
            "r_unc": float(r_unc),
            "r_des": float(r_des),
        }

    # ------------------------------------------------------------------
    # Individual components
    # ------------------------------------------------------------------

    @staticmethod
    def _r_ood(mse_before: float, mse_after: float) -> float:
        """Improvement in OOD validation MSE (positive = better)."""
        return mse_before - mse_after

    @staticmethod
    def _r_cov(
        pair_before: float,
        pair_after: float,
        path_before: float,
        path_after: float,
    ) -> float:
        """Weighted coverage gain over gene pairs and pathway pairs."""
        delta_pair = pair_after - pair_before
        delta_path = path_after - path_before
        return delta_pair + 0.5 * delta_path

    @staticmethod
    def _r_red(batch_embeddings: np.ndarray) -> float:
        """
        Redundancy penalty: negative mean pairwise cosine similarity.

        Returns a value in (-1, 0].  0 means perfectly diverse batch.
        """
        if len(batch_embeddings) < 2:
            return 0.0
        e = torch.tensor(batch_embeddings, dtype=torch.float32)
        e_norm = F.normalize(e, dim=-1)
        sim = e_norm @ e_norm.T         # [B, B]
        N = sim.shape[0]
        idx = torch.triu_indices(N, N, offset=1)
        mean_sim = sim[idx[0], idx[1]].mean().item()
        return -mean_sim

    @staticmethod
    def _r_unc(unc_before: float, unc_after: float) -> float:
        """Reduction in mean pool uncertainty (positive = uncertainty resolved)."""
        return unc_before - unc_after

    @staticmethod
    def _r_des(des_before: float, des_after: float) -> float:
        """Improvement in Differential Expression Score (positive = better recall)."""
        return des_after - des_before
