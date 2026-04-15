"""
Ensemble predictor: 5-member GeneCellTransformerPredictor ensemble.

Each member is a full Transformer predictor (imported from predict.py).
predict()       -> mean prediction across members [B, G]
uncertainty()   -> mean per-gene variance across members, averaged to scalar [B]
get_embedding() -> penultimate layer (post-transformer, mean-pooled over genes) [B, D]
"""

import sys
import os

# Allow importing predict.py from the project root.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
import torch.nn as nn
import numpy as np

from predict import GeneCellTransformerPredictor, sanitize_gene_embeddings


class EnsemblePredictor(nn.Module):
    """
    5-member ensemble of GeneCellTransformerPredictor models.

    Different random seeds initialise different members, giving diverse
    predictions whose spread is used as the epistemic uncertainty signal u(x).
    """

    def __init__(
        self,
        gene_embeddings: np.ndarray,
        cell_dim: int,
        num_guides: int,
        config: dict,
    ):
        """
        Parameters
        ----------
        gene_embeddings : np.ndarray
            Shape [G, Dg].  Passed through sanitize_gene_embeddings before use.
        cell_dim : int
            Dimensionality of cell embedding vectors (Dc).
        num_guides : int
            Vocabulary size for perturbation guide tokens.
        config : dict
            Full config dict (loaded from default.yaml).
        """
        super().__init__()

        cfg = config["predictor"]
        self.ensemble_size: int = cfg["ensemble_size"]
        self.model_dim: int = cfg["model_dim"]

        gene_embeddings = sanitize_gene_embeddings(gene_embeddings)

        self.members = nn.ModuleList()
        for seed in range(self.ensemble_size):
            torch.manual_seed(seed)
            member = GeneCellTransformerPredictor(
                gene_embeddings=gene_embeddings,
                cell_dim=cell_dim,
                num_guides=num_guides,
                model_dim=cfg["model_dim"],
                num_heads=cfg["num_heads"],
                num_layers=cfg["num_layers"],
                ff_dim=cfg["ff_dim"],
                dropout=cfg["dropout"],
            )
            self.members.append(member)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def forward(
        self,
        cell_emb: torch.Tensor,
        pert_gene_ids: torch.Tensor,
        pert_gene_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Alias for predict(); returns mean prediction [B, G]."""
        return self.predict(cell_emb, pert_gene_ids, pert_gene_mask)

    @torch.no_grad()
    def predict(
        self,
        cell_emb: torch.Tensor,
        pert_gene_ids: torch.Tensor,
        pert_gene_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Mean prediction across all ensemble members.

        Returns
        -------
        torch.Tensor
            Shape [B, G].
        """
        preds = torch.stack(
            [m(cell_emb, pert_gene_ids, pert_gene_mask) for m in self.members]
        )  # [E, B, G]
        return preds.mean(dim=0)

    @torch.no_grad()
    def uncertainty(
        self,
        cell_emb: torch.Tensor,
        pert_gene_ids: torch.Tensor,
        pert_gene_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Epistemic uncertainty u(x): variance across members, averaged over genes.

        Returns
        -------
        torch.Tensor
            Shape [B].  Higher value = more uncertain prediction.
        """
        preds = torch.stack(
            [m(cell_emb, pert_gene_ids, pert_gene_mask) for m in self.members]
        )  # [E, B, G]
        var = preds.var(dim=0)   # [B, G]
        return var.mean(dim=-1)  # [B]

    @torch.no_grad()
    def get_embedding(
        self,
        cell_emb: torch.Tensor,
        pert_gene_ids: torch.Tensor,
        pert_gene_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Penultimate representation z_x: transformer output mean-pooled over genes,
        averaged across all ensemble members.

        The 'penultimate layer' for GeneCellTransformerPredictor is the output of
        the TransformerEncoder before the out_head linear layer.  We register a
        forward hook to capture it, then mean-pool the [B, G, D] tensor to [B, D].

        Returns
        -------
        torch.Tensor
            Shape [B, model_dim].
        """
        member_embs = []
        for member in self.members:
            captured: dict = {}

            def _hook(module, inp, output, store=captured):
                store["out"] = output  # [B, G, D]

            handle = member.transformer.register_forward_hook(_hook)
            try:
                member(cell_emb, pert_gene_ids, pert_gene_mask)
            finally:
                handle.remove()

            # mean pool over gene tokens → [B, D]
            member_embs.append(captured["out"].mean(dim=1))

        return torch.stack(member_embs).mean(dim=0)  # [B, model_dim]

    # ------------------------------------------------------------------
    # Convenience: reset one member to re-initialise weights
    # ------------------------------------------------------------------

    def reset_member(
        self,
        idx: int,
        gene_embeddings: np.ndarray,
        cell_dim: int,
        num_guides: int,
        config: dict,
    ) -> None:
        """Re-initialise member *idx* with a fresh random seed."""
        cfg = config["predictor"]
        gene_embeddings = sanitize_gene_embeddings(gene_embeddings)
        torch.manual_seed(idx)
        new_member = GeneCellTransformerPredictor(
            gene_embeddings=gene_embeddings,
            cell_dim=cell_dim,
            num_guides=num_guides,
            model_dim=cfg["model_dim"],
            num_heads=cfg["num_heads"],
            num_layers=cfg["num_layers"],
            ff_dim=cfg["ff_dim"],
            dropout=cfg["dropout"],
        )
        self.members[idx] = new_member
