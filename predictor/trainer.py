"""
Predictor training strategy:
- finetune()      warm-start from current weights, 15 epochs
- full_retrain()  reset all member weights and retrain from scratch, 50 epochs
- update()        calls full_retrain every 4 rounds, finetune otherwise
"""

import sys
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import copy
from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from predict import (
    ExpressionDataset,
    sanitize_gene_embeddings,
    train_one_epoch,
    evaluate,
)
from predictor.ensemble import EnsemblePredictor


@dataclass
class _TrainState:
    """Mutable training bookkeeping shared across finetune / full_retrain calls."""
    round_idx: int = 0


class PredictorTrainer:
    """
    Manages training of an EnsemblePredictor.

    Supports two modes:
      - warm-start fine-tuning (fast, preserves learned weights)
      - full retrain from scratch (reset weights, more epochs)

    update() selects the mode automatically based on round index.
    """

    def __init__(
        self,
        ensemble: EnsemblePredictor,
        gene_embeddings: np.ndarray,
        cell_dim: int,
        num_guides: int,
        config: Dict[str, Any],
        device: str,
    ):
        """
        Parameters
        ----------
        ensemble     : EnsemblePredictor to train.
        gene_embeddings : [G, Dg] array used to re-init members on full retrain.
        cell_dim     : cell embedding dimension.
        num_guides   : perturbation guide vocabulary size.
        config       : full config dict.
        device       : torch device string.
        """
        self.ensemble = ensemble
        self.gene_embeddings = sanitize_gene_embeddings(gene_embeddings)
        self.cell_dim = cell_dim
        self.num_guides = num_guides
        self.config = config
        self.device = device
        self._state = _TrainState()

        cfg = config["predictor"]
        self.full_retrain_every: int = cfg["full_retrain_every"]
        self.finetune_epochs: int = cfg["finetune_epochs"]
        self.full_retrain_epochs: int = cfg["full_retrain_epochs"]
        self.batch_size: int = cfg["batch_size"]
        self.lr: float = cfg["lr"]
        self.weight_decay: float = cfg["weight_decay"]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, data: Dict[str, np.ndarray], round_idx: int) -> Dict[str, float]:
        """
        Select training mode based on round index and run training.

        Parameters
        ----------
        data      : dict with keys:
                    'cell_embeddings'   [N, Dc]
                    'expression_matrix' [N, G]
                    'pert_gene_ids'     [N, P]
                    'pert_gene_mask'    [N, P]
        round_idx : current AL round (0-based).

        Returns
        -------
        dict with 'train_loss' and optionally 'val_loss'.
        """
        self._state.round_idx = round_idx
        if round_idx % self.full_retrain_every == 0:
            return self.full_retrain(data)
        else:
            return self.finetune(data)

    def finetune(
        self,
        data: Dict[str, np.ndarray],
        epochs: int | None = None,
    ) -> Dict[str, float]:
        """
        Warm-start fine-tune all members on *data* without resetting weights.

        Parameters
        ----------
        data   : same format as update().
        epochs : override finetune_epochs if provided.
        """
        n_epochs = epochs if epochs is not None else self.finetune_epochs
        return self._train_all_members(data, n_epochs, reset_weights=False)

    def full_retrain(
        self,
        data: Dict[str, np.ndarray],
        epochs: int | None = None,
    ) -> Dict[str, float]:
        """
        Reset all ensemble members and retrain from scratch on *data*.

        Parameters
        ----------
        data   : same format as update().
        epochs : override full_retrain_epochs if provided.
        """
        n_epochs = epochs if epochs is not None else self.full_retrain_epochs
        # Re-initialise each member with its original seed.
        for idx in range(self.ensemble.ensemble_size):
            self.ensemble.reset_member(
                idx,
                self.gene_embeddings,
                self.cell_dim,
                self.num_guides,
                self.config,
            )
        return self._train_all_members(data, n_epochs, reset_weights=False)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_loader(
        self,
        data: Dict[str, np.ndarray],
        shuffle: bool,
    ) -> DataLoader:
        dataset = ExpressionDataset(
            cell_embeddings=data["cell_embeddings"].astype(np.float32),
            expression_matrix=data["expression_matrix"].astype(np.float32),
            pert_gene_ids=data["pert_gene_ids"].astype(np.int64),
            pert_gene_mask=data["pert_gene_mask"].astype(np.float32),
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

    def _train_all_members(
        self,
        data: Dict[str, np.ndarray],
        epochs: int,
        reset_weights: bool,
    ) -> Dict[str, float]:
        """
        Train each ensemble member with an internal 10% val split for
        best-epoch selection — matching random_sample.py's val_ratio=0.1
        and predict.py's fit_model behaviour.
        """
        N = len(data["cell_embeddings"])
        criterion = nn.MSELoss()
        total_loss = 0.0

        # Internal val split (same ratio as random_sample.py default)
        if N >= 2:
            val_size = max(1, int(0.1 * N))
            all_idx = np.arange(N)
            train_local, val_local = train_test_split(
                all_idx, test_size=val_size, random_state=self._state.round_idx
            )
        else:
            train_local = np.arange(N)
            val_local = np.arange(N)

        train_data = {k: v[train_local] for k, v in data.items()}
        val_data = {k: v[val_local] for k, v in data.items()}

        train_loader = self._build_loader(train_data, shuffle=True)
        val_loader = self._build_loader(val_data, shuffle=False)

        for member in self.ensemble.members:
            member = member.to(self.device)
            member.train()
            optimizer = torch.optim.AdamW(
                member.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
            best_val = float("inf")
            best_state = None
            for _ in range(epochs):
                train_one_epoch(member, train_loader, optimizer, criterion, self.device)
                val_loss = evaluate(member, val_loader, criterion, self.device)
                if val_loss < best_val:
                    best_val = val_loss
                    best_state = {k: v.cpu() for k, v in member.state_dict().items()}
            if best_state is not None:
                member.load_state_dict(best_state)
            total_loss += best_val

        avg_loss = total_loss / max(self.ensemble.ensemble_size, 1)
        return {"train_loss": avg_loss}

    def evaluate_on(
        self,
        data: Dict[str, np.ndarray],
    ) -> float:
        """Return average MSE across ensemble members on *data*."""
        loader = self._build_loader(data, shuffle=False)
        criterion = nn.MSELoss()
        total = 0.0
        for member in self.ensemble.members:
            member = member.to(self.device).eval()
            total += evaluate(member, loader, criterion, self.device)
        return total / max(self.ensemble.ensemble_size, 1)
