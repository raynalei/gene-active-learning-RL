"""
Active Learning Environment — gym-style MDP.

Data split strategy (consistent with random_sample.py and predict.py)
----------------------------------------------------------------------
All splits happen at the CELL level first, mirroring random_sample.py:

  1. test_cells   (20%)  — held out permanently; never used for training or
                           pool selection. Reported as 'test_mse' so results
                           are directly comparable to the random baseline.

  2. trainable_cells (80%) — everything else:
       a. ood_val_cells  — cells belonging to OOD-1 conditions (genes seen
                           individually in training, combinations never seen).
                           Used to compute ood_val_mse each round.
       b. id_val_cells   — ~10% of remaining trainable cells, used as an
                           ID validation set during predictor training.
       c. train_pool     — the remainder; partitioned into labeled D_t and
                           unlabeled pool U_t.

Active learning loop
---------------------
The AGENT selects at the CONDITION level (one unique guide_merged = one
perturbation experiment). When a condition is queried:
  - All of its cells that fall in train_pool (i.e. not test, not val) are
    added to the labeled set D_t.
  - The predictor is then trained on all cells in D_t using cell-level data,
    exactly as predict.py / random_sample.py do.

Embeddings used for STATE COMPUTATION remain condition-level (mean cell
embedding per condition), because the agent's observation describes the
distribution of experiments, not individual cells.
"""

from __future__ import annotations

import sys
import os
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from predict import _parse_guide_merged, _encode_guide_merged
from predictor.ensemble import EnsemblePredictor
from predictor.trainer import PredictorTrainer
from environment.state import StateComputer
from environment.reward import RewardComputer
from policy.features import FeatureExtractor


# ---------------------------------------------------------------------------
# Oracle
# ---------------------------------------------------------------------------

class Oracle:
    """
    Returns ground-truth cell-level data for queried perturbation conditions.

    query() returns ALL cells belonging to the condition that are in the
    trainable (non-test) split — consistent with how random_sample.py builds
    its labeled set by adding individual cells.
    """

    def __init__(
        self,
        condition_names: List[str],
        cell_embeddings: np.ndarray,    # [N, Dc]  all cells
        expression_matrix: np.ndarray,  # [N, G]   all cells
        pert_gene_ids: np.ndarray,      # [N, P]
        pert_gene_mask: np.ndarray,     # [N, P]
        cond_cell_indices: List[List[int]],  # per-condition cell lists
        train_cell_set: Set[int],            # cells eligible for labeling
    ):
        self.condition_names = condition_names
        self.cell_embeddings = cell_embeddings
        self.expression_matrix = expression_matrix
        self.pert_gene_ids = pert_gene_ids
        self.pert_gene_mask = pert_gene_mask
        self.cond_cell_indices = cond_cell_indices
        self.train_cell_set = train_cell_set
        self._name2idx = {n: i for i, n in enumerate(condition_names)}

    def query(self, condition_names: List[str]) -> List[int]:
        """
        Return trainable cell indices for the queried conditions.

        Only cells inside train_cell_set are returned; test/val cells are
        never exposed as labels.
        """
        indices = []
        for name in condition_names:
            cond_i = self._name2idx[name]
            for cell_idx in self.cond_cell_indices[cond_i]:
                if cell_idx in self.train_cell_set:
                    indices.append(cell_idx)
        return indices


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_norman2019(
    h5ad_path: str,
    cell_embeddings_path: str,
) -> Tuple[
    List[str],          # condition names (unique guide_merged)
    np.ndarray,         # cell_embeddings [N, Dc]
    np.ndarray,         # expression_matrix [N, G]
    np.ndarray,         # pert_gene_ids [N, P]
    np.ndarray,         # pert_gene_mask [N, P]
    List[List[int]],    # cond_cell_indices[c] = list of cell indices
    List[Set[str]],     # gene_sets per condition
    np.ndarray,         # mean_cell_emb per condition [C, Dc]  (for state)
]:
    import scanpy as sc
    import scipy.sparse as sp
    from predict import PERT_GENE_IDS_CACHE, PERT_GENE_MASK_CACHE

    adata = sc.read_h5ad(h5ad_path)
    _encode_guide_merged(adata)

    X = adata.X
    if sp.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)

    cell_embs = np.load(cell_embeddings_path).astype(np.float32)
    N = len(cell_embs)

    pg_ids = PERT_GENE_IDS_CACHE if PERT_GENE_IDS_CACHE is not None else np.zeros((N, 1), dtype=np.int64)
    pg_mask = PERT_GENE_MASK_CACHE if PERT_GENE_MASK_CACHE is not None else np.zeros((N, 1), dtype=np.float32)

    guide_values = adata.obs["guide_merged"].tolist()
    parsed = [_parse_guide_merged(v) for v in guide_values]
    guide_strings = ["+".join(gs) if gs else "ctrl" for gs in parsed]
    unique_guides = list(dict.fromkeys(guide_strings))

    name2idx = {n: i for i, n in enumerate(unique_guides)}
    cond_cell_indices: List[List[int]] = [[] for _ in unique_guides]
    for cell_i, gs in enumerate(guide_strings):
        cond_cell_indices[name2idx[gs]].append(cell_i)

    gene_sets = [set(parsed[cond_cell_indices[c][0]]) for c in range(len(unique_guides))]

    # Condition-level mean cell embeddings (for state computation only)
    mean_cell_emb = np.stack([
        cell_embs[cond_cell_indices[c]].mean(axis=0) for c in range(len(unique_guides))
    ])

    return (
        unique_guides,
        cell_embs,
        X,
        pg_ids,
        pg_mask,
        cond_cell_indices,
        gene_sets,
        mean_cell_emb,
    )


def _build_ood_split(
    gene_sets: List[Set[str]],
    ood_val_fraction: float,
    seed: int,
) -> Tuple[List[int], List[int]]:
    """
    OOD-1: double knockouts where both genes appear as single knockouts.
    Returns (ood_val_condition_indices, ood_pool_condition_indices).
    """
    single_genes: Set[str] = {g for gs in gene_sets if len(gs) == 1 for g in gs}
    ood1 = [i for i, gs in enumerate(gene_sets) if len(gs) == 2 and gs.issubset(single_genes)]
    rng = np.random.default_rng(seed)
    rng.shuffle(ood1)
    n_val = max(1, int(ood_val_fraction * len(ood1)))
    return ood1[:n_val], ood1[n_val:]


def _build_pathway_map(gene_names: List[str], seed: int = 0, n_pathways: int = 50) -> Dict[str, int]:
    rng = np.random.default_rng(seed)
    return {g: int(rng.integers(0, n_pathways)) for g in gene_names}


# ---------------------------------------------------------------------------
# Main environment
# ---------------------------------------------------------------------------

class ALEnvironment:
    """
    Gym-style active learning environment for gene perturbation experiment selection.

    Split hierarchy (cell-level, matching random_sample.py)
    --------------------------------------------------------
    All N cells
    ├── test_cells          (20%)  fixed holdout, never trained on
    └── trainable_cells     (80%)
        ├── ood_val_cells          cells of OOD-1 val conditions
        ├── id_val_cells           ~10% of remaining, for ID monitoring
        └── train_pool_cells       available for labeling

    Agent operates at condition level; labeling a condition reveals all of
    its train_pool cells to the predictor.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        ensemble: EnsemblePredictor,
        trainer: PredictorTrainer,
        gene_embeddings: np.ndarray,
        h5ad_path: str,
        cell_embeddings_path: str,
        pathway_map: Optional[Dict[str, int]] = None,
        device: str = "cpu",
    ):
        self.config = config
        self.ensemble = ensemble
        self.trainer = trainer
        self.device = device

        cfg_al = config["active_learning"]
        self.batch_size: int = cfg_al["batch_size"]
        self.num_rounds: int = cfg_al["num_rounds"]
        self.initial_labeled_size: int = cfg_al["initial_labeled_size"]
        self._test_ratio: float = cfg_al["test_ratio"]
        self._id_val_ratio: float = cfg_al["id_val_ratio"]
        self._ood_split_seed: int = cfg_al["ood_split_seed"]

        # ------------------------------------------------------------------
        # Load data
        # ------------------------------------------------------------------
        (
            self._cond_names,
            self._cell_embs,       # [N, Dc]  all cells
            self._expr,            # [N, G]   all cells
            self._pg_ids,          # [N, P]
            self._pg_mask,         # [N, P]
            self._cond_cell_idx,   # List[List[int]]
            self._gene_sets,       # List[Set[str]]
            self._mean_cell_emb,   # [C, Dc]  condition-level means (for state)
        ) = _load_norman2019(h5ad_path, cell_embeddings_path)

        C = len(self._cond_names)
        N = self._cell_embs.shape[0]
        all_cell_idx = np.arange(N)

        # ------------------------------------------------------------------
        # Cell-level test split — identical logic to random_sample.py
        # ------------------------------------------------------------------
        trainable_cells, test_cells = train_test_split(
            all_cell_idx,
            test_size=self._test_ratio,
            random_state=self._ood_split_seed,
            shuffle=True,
        )
        self._test_cell_indices: np.ndarray = test_cells
        self._train_cell_set: Set[int] = set(trainable_cells.tolist())

        # ------------------------------------------------------------------
        # OOD-1 split (condition-level)
        # ------------------------------------------------------------------
        ood_val_conds, ood_pool_conds = _build_ood_split(
            self._gene_sets, cfg_al["ood_val_fraction"], self._ood_split_seed
        )
        self._ood_val_cond_indices: List[int] = ood_val_conds
        self._ood_pool_cond_set: Set[int] = set(ood_pool_conds)

        # OOD-1 val cells: cells of OOD-1 val conditions that are NOT in test split
        ood_val_cells = [
            ci for c in ood_val_conds
            for ci in self._cond_cell_idx[c]
            if ci in self._train_cell_set
        ]
        self._ood_val_cell_indices: np.ndarray = np.array(ood_val_cells, dtype=np.int64)

        # Remaining trainable cells (not OOD-val)
        ood_val_cell_set = set(ood_val_cells)
        remaining_trainable = np.array(
            [ci for ci in trainable_cells if ci not in ood_val_cell_set],
            dtype=np.int64,
        )

        # ------------------------------------------------------------------
        # ID val split — ~10% of remaining trainable cells
        # ------------------------------------------------------------------
        if len(remaining_trainable) > 1:
            pool_cells, id_val_cells = train_test_split(
                remaining_trainable,
                test_size=self._id_val_ratio,
                random_state=self._ood_split_seed + 1,
                shuffle=True,
            )
        else:
            pool_cells = remaining_trainable
            id_val_cells = np.array([], dtype=np.int64)

        self._id_val_cell_indices: np.ndarray = id_val_cells
        # Final set of cells eligible to be labeled
        self._pool_cell_set: Set[int] = set(pool_cells.tolist())

        # ------------------------------------------------------------------
        # Condition-level pool: conditions that have cells in pool_cell_set
        # (exclude OOD-1 val conditions; include OOD-pool conditions)
        # ------------------------------------------------------------------
        ood_val_cond_set = set(ood_val_conds)
        self._all_trainable_conds: List[int] = [
            c for c in range(C)
            if c not in ood_val_cond_set
            and any(ci in self._pool_cell_set for ci in self._cond_cell_idx[c])
        ]

        # ------------------------------------------------------------------
        # Coverage vocabularies
        # ------------------------------------------------------------------
        self._all_single_genes: Set[str] = {
            g for gs in self._gene_sets if len(gs) == 1 for g in gs
        }
        self._all_gene_pairs: Set[FrozenSet] = {
            frozenset({a, b})
            for gs in self._gene_sets
            for i, a in enumerate(gs)
            for b in list(gs)[i + 1:]
        }
        all_gene_names = list({g for gs in self._gene_sets for g in gs})
        self._pathway_map = pathway_map or _build_pathway_map(
            all_gene_names, seed=self._ood_split_seed
        )
        self._all_pathway_pairs: Set[FrozenSet] = {
            frozenset({self._pathway_map.get(a, -1), self._pathway_map.get(b, -1)})
            for gs in self._gene_sets
            for i, a in enumerate(list(gs))
            for b in list(gs)[i + 1:]
            if self._pathway_map.get(a, -1) >= 0 and self._pathway_map.get(b, -1) >= 0
        }

        # ------------------------------------------------------------------
        # Control expression baseline (mean of unperturbed cells)
        # ------------------------------------------------------------------
        ctrl_cond_indices = [c for c, gs in enumerate(self._gene_sets) if len(gs) == 0]
        if ctrl_cond_indices:
            ctrl_cells = np.array([
                ci for c in ctrl_cond_indices for ci in self._cond_cell_idx[c]
            ], dtype=np.int64)
            self._ctrl_expr: np.ndarray = self._expr[ctrl_cells].mean(axis=0)
        else:
            self._ctrl_expr: np.ndarray = self._expr.mean(axis=0)
        self._de_threshold: float = 0.5

        # Precompute ground-truth DE gene sets for OOD val conditions (fixed)
        self._ood_val_true_de: Dict[int, Set[int]] = self._compute_true_de_sets(
            self._ood_val_cond_indices
        )

        # ------------------------------------------------------------------
        # Oracle
        # ------------------------------------------------------------------
        self.oracle = Oracle(
            condition_names=self._cond_names,
            cell_embeddings=self._cell_embs,
            expression_matrix=self._expr,
            pert_gene_ids=self._pg_ids,
            pert_gene_mask=self._pg_mask,
            cond_cell_indices=self._cond_cell_idx,
            train_cell_set=self._pool_cell_set,
        )

        # ------------------------------------------------------------------
        # Helper objects
        # ------------------------------------------------------------------
        self.state_computer = StateComputer(config)
        self.reward_computer = RewardComputer(config)
        self.feature_extractor = FeatureExtractor(config)

        # When True, _end_of_round skips trainer.update() (used during BC
        # teacher rollouts to avoid 400× expensive predictor retraining).
        self.freeze_predictor: bool = False

        # ------------------------------------------------------------------
        # Episode state (set by reset())
        # ------------------------------------------------------------------
        self._labeled_cond: List[int] = []   # labeled condition indices
        self._labeled_cells: List[int] = []  # corresponding cell indices
        self._pool_conds: List[int] = []     # remaining condition pool

        self._current_batch_conds: List[int] = []
        self._selection_step: int = 1
        self._round_idx: int = 0

        # Cached per-round embeddings (condition-level, for state)
        self._pool_embeddings: Optional[np.ndarray] = None
        self._labeled_embeddings: Optional[np.ndarray] = None
        self._pool_uncertainties: Optional[np.ndarray] = None

        # Metrics
        self._id_val_mse: float = 0.0
        self._ood_val_mse: float = 0.0
        self._test_mse: float = 0.0
        self._coverage_state: Dict[str, float] = {
            "gene_pair_coverage": 0.0,
        }

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        """Start a new episode. Seed labeled set D_0, recompute embeddings."""
        rng = np.random.default_rng(self.config.get("seed", 42) + self._round_idx)
        shuffled = list(self._all_trainable_conds)
        rng.shuffle(shuffled)

        init_n = min(self.initial_labeled_size, len(shuffled))
        self._labeled_cond = list(shuffled[:init_n])
        self._pool_conds = list(shuffled[init_n:])

        # Expand conditions → cell indices (only pool cells)
        self._labeled_cells = self.oracle.query(
            [self._cond_names[c] for c in self._labeled_cond]
        )

        self._current_batch_conds = []
        self._selection_step = 1
        self._round_idx = 0

        self._recompute_embeddings()
        # Skip _update_metrics() at reset — MSE defaults to 0.0 until round 0
        # completes training. Avoids an expensive full-dataset eval before any
        # predictor update has happened.
        return self._get_state()

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Select candidate at position *action* in the current pool ordering.

        Within round (b < B): update h_pb, return reward=0.
        End of round (b == B): query oracle, update predictor, compute reward.
        """
        assert 0 <= action < len(self._pool_conds), \
            f"action {action} out of range [0, {len(self._pool_conds)})"

        selected_cond = self._pool_conds[action]
        self._current_batch_conds.append(selected_cond)
        self._pool_conds.pop(action)

        # Keep cached embeddings/uncertainties in sync with _pool_conds so that
        # within-round calls to env.pool_embeddings / env.pool_uncertainties
        # always have the same length as _pool_conds.
        if self._pool_embeddings is not None:
            self._pool_embeddings = np.delete(self._pool_embeddings, action, axis=0)
        if self._pool_uncertainties is not None:
            self._pool_uncertainties = np.delete(self._pool_uncertainties, action, axis=0)

        info: Dict[str, Any] = {}

        if self._selection_step < self.batch_size:
            self._selection_step += 1
            reward = 0.0
            done = False
        else:
            reward, info = self._end_of_round()
            self._round_idx += 1
            done = (
                self._round_idx >= self.num_rounds
                or len(self._pool_conds) < self.batch_size
            )
            self._current_batch_conds = []
            self._selection_step = 1

        return self._get_state(), reward, done, info

    def compute_reward(
        self,
        batch_embeddings: np.ndarray,
        ood_mse_before: float,
        ood_mse_after: float,
        cov_before: Dict[str, float],
        cov_after: Dict[str, float],
        unc_before: float,
        unc_after: float,
        des_before: float,
        des_after: float,
    ) -> Dict[str, float]:
        return self.reward_computer.compute(
            ood_mse_before=ood_mse_before,
            ood_mse_after=ood_mse_after,
            gene_pair_coverage_before=cov_before["gene_pair_coverage"],
            gene_pair_coverage_after=cov_after["gene_pair_coverage"],
            batch_embeddings=batch_embeddings,
            pool_uncertainty_before=unc_before,
            pool_uncertainty_after=unc_after,
            des_before=des_before,
            des_after=des_after,
        )

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def _get_state(self) -> np.ndarray:
        partial_embs = self._embed_conditions(self._current_batch_conds)
        partial_gs = [self._gene_sets[c] for c in self._current_batch_conds]
        labeled_gs = [self._gene_sets[c] for c in self._labeled_cond]

        return self.state_computer.compute(
            labeled_embeddings=self._labeled_embeddings,
            pool_embeddings=self._pool_embeddings,
            pool_uncertainties=self._pool_uncertainties,
            partial_batch_embeddings=partial_embs,
            labeled_gene_sets=labeled_gs,
            partial_batch_gene_sets=partial_gs,
            all_single_genes=self._all_single_genes,
            all_gene_pairs=self._all_gene_pairs,
            all_pathway_pairs=self._all_pathway_pairs,
            pool_gene_sets=[self._gene_sets[c] for c in self._pool_conds],
            pathway_map=self._pathway_map,
            id_val_mse=self._id_val_mse,
            ood_val_mse=self._ood_val_mse,
            selection_step=self._selection_step,
        )

    def _recompute_embeddings(self) -> None:
        """Recompute condition-level embeddings after predictor update."""
        self._labeled_embeddings = self._embed_conditions(self._labeled_cond)
        self._pool_embeddings = self._embed_conditions(self._pool_conds)
        self._pool_uncertainties = self._compute_uncertainties(self._pool_conds)

    def _embed_conditions(self, cond_indices: List[int]) -> np.ndarray:
        """Mean-pool condition embedding via ensemble (for state computation)."""
        if len(cond_indices) == 0:
            return np.zeros((0, self.config["state"]["embedding_dim"]), dtype=np.float32)
        chunk = self.config.get("embed_batch_size", 32)
        results = []
        rep_cells = [self._cond_cell_idx[c][0] for c in cond_indices]
        for i in range(0, len(cond_indices), chunk):
            ci = cond_indices[i:i + chunk]
            rc = rep_cells[i:i + chunk]
            cell_t = torch.tensor(self._mean_cell_emb[ci], dtype=torch.float32).to(self.device)
            pg_ids_t = torch.tensor(self._pg_ids[rc], dtype=torch.long).to(self.device)
            pg_mask_t = torch.tensor(self._pg_mask[rc], dtype=torch.float32).to(self.device)
            results.append(self.ensemble.get_embedding(cell_t, pg_ids_t, pg_mask_t).cpu().numpy())
        return np.concatenate(results, axis=0)

    def _compute_uncertainties(self, cond_indices: List[int]) -> np.ndarray:
        if len(cond_indices) == 0:
            return np.zeros(0, dtype=np.float32)
        chunk = self.config.get("embed_batch_size", 32)
        results = []
        rep_cells = [self._cond_cell_idx[c][0] for c in cond_indices]
        for i in range(0, len(cond_indices), chunk):
            ci = cond_indices[i:i + chunk]
            rc = rep_cells[i:i + chunk]
            cell_t = torch.tensor(self._mean_cell_emb[ci], dtype=torch.float32).to(self.device)
            pg_ids_t = torch.tensor(self._pg_ids[rc], dtype=torch.long).to(self.device)
            pg_mask_t = torch.tensor(self._pg_mask[rc], dtype=torch.float32).to(self.device)
            results.append(self.ensemble.uncertainty(cell_t, pg_ids_t, pg_mask_t).cpu().numpy())
        return np.concatenate(results, axis=0)

    # ------------------------------------------------------------------
    # Round-level operations
    # ------------------------------------------------------------------

    def _end_of_round(self) -> Tuple[float, Dict[str, Any]]:
        # Snapshot before
        ood_before = self._ood_val_mse
        cov_before = dict(self._coverage_state)
        unc_before = float(self._pool_uncertainties.mean()) if len(self._pool_uncertainties) > 0 else 0.0
        des_before = self._compute_des()

        # Move batch conditions → labeled; reveal cell-level labels from oracle
        batch_names = [self._cond_names[c] for c in self._current_batch_conds]
        new_cells = self.oracle.query(batch_names)
        self._labeled_cond.extend(self._current_batch_conds)
        self._labeled_cells.extend(new_cells)

        # Train predictor on ALL labeled cells (cell-level, like random_sample.py)
        if not self.freeze_predictor:
            self.trainer.update(self._build_train_data(), self._round_idx)

        # Recompute state & metrics
        self._recompute_embeddings()
        self._update_metrics()

        ood_after = self._ood_val_mse
        cov_after = dict(self._coverage_state)
        unc_after = float(self._pool_uncertainties.mean()) if len(self._pool_uncertainties) > 0 else 0.0
        des_after = self._compute_des()

        batch_embs = self._embed_conditions(self._current_batch_conds)
        reward_dict = self.compute_reward(
            batch_embeddings=batch_embs,
            ood_mse_before=ood_before,
            ood_mse_after=ood_after,
            cov_before=cov_before,
            cov_after=cov_after,
            unc_before=unc_before,
            unc_after=unc_after,
            des_before=des_before,
            des_after=des_after,
        )

        info = {
            "round": self._round_idx,
            "num_labeled_cells": len(self._labeled_cells),
            "num_labeled_conds": len(self._labeled_cond),
            "id_val_mse": self._id_val_mse,
            "ood_val_mse": ood_after,
            "test_mse": self._test_mse,   # comparable to random_sample.py
            "reward_components": reward_dict,
        }
        return reward_dict["total"], info

    def _build_train_data(self) -> Dict[str, np.ndarray]:
        """
        Cell-level training data from all labeled cells.
        Mirrors random_sample.py: train on individual cells, not condition means.
        """
        idx = np.array(self._labeled_cells, dtype=np.int64)
        return {
            "cell_embeddings": self._cell_embs[idx],
            "expression_matrix": self._expr[idx],
            "pert_gene_ids": self._pg_ids[idx],
            "pert_gene_mask": self._pg_mask[idx],
        }

    def _update_metrics(self) -> None:
        """Recompute ID MSE, OOD-1 MSE, and test MSE on cell-level data."""
        def _cell_level_dict(cell_indices: np.ndarray) -> Dict[str, np.ndarray]:
            return {
                "cell_embeddings": self._cell_embs[cell_indices],
                "expression_matrix": self._expr[cell_indices],
                "pert_gene_ids": self._pg_ids[cell_indices],
                "pert_gene_mask": self._pg_mask[cell_indices],
            }

        if len(self._labeled_cells) > 0:
            if len(self._id_val_cell_indices) > 0:
                self._id_val_mse = self.trainer.evaluate_on(
                    _cell_level_dict(self._id_val_cell_indices)
                )
            if len(self._ood_val_cell_indices) > 0:
                self._ood_val_mse = self.trainer.evaluate_on(
                    _cell_level_dict(self._ood_val_cell_indices)
                )
            # Test MSE — same held-out 20% as random_sample.py
            self._test_mse = self.trainer.evaluate_on(
                _cell_level_dict(self._test_cell_indices)
            )

        # Coverage stats
        labeled_gs = [self._gene_sets[c] for c in self._labeled_cond]
        pair_cov = self._compute_coverage(labeled_gs)
        self._coverage_state = {
            "gene_pair_coverage": pair_cov,
        }

    def _compute_coverage(self, labeled_gs: List[Set[str]]) -> float:
        seen_pairs: Set[FrozenSet] = {
            frozenset({a, b})
            for gs in labeled_gs
            for i, a in enumerate(list(gs))
            for b in list(gs)[i + 1:]
        }
        return (
            len(seen_pairs & self._all_gene_pairs) / len(self._all_gene_pairs)
            if self._all_gene_pairs else 0.0
        )

    def _compute_true_de_sets(self, cond_indices: List[int]) -> Dict[int, Set[int]]:
        """Compute ground-truth differentially expressed gene index sets."""
        de_sets: Dict[int, Set[int]] = {}
        for c in cond_indices:
            cells = np.array(self._cond_cell_idx[c], dtype=np.int64)
            if len(cells) == 0:
                de_sets[c] = set()
                continue
            expr_mean = self._expr[cells].mean(axis=0)
            diff = np.abs(expr_mean - self._ctrl_expr)
            de_sets[c] = set(np.where(diff > self._de_threshold)[0].tolist())
        return de_sets

    def _compute_des(self) -> float:
        """
        Differential Expression Score on the OOD validation set.

        DES = (1/|T|) * sum_{p in T} |G_true(p) ∩ G_pred(p)| / |G_true(p)|
        """
        if not self._ood_val_cond_indices:
            return 0.0
        total = 0.0
        count = 0
        for c in self._ood_val_cond_indices:
            g_true = self._ood_val_true_de.get(c, set())
            if not g_true:
                continue
            cell_t = torch.tensor(
                self._mean_cell_emb[[c]], dtype=torch.float32
            ).to(self.device)
            rep_cell = self._cond_cell_idx[c][0]
            pg_ids_t = torch.tensor(
                self._pg_ids[[rep_cell]], dtype=torch.long
            ).to(self.device)
            pg_mask_t = torch.tensor(
                self._pg_mask[[rep_cell]], dtype=torch.float32
            ).to(self.device)
            with torch.no_grad():
                pred = self.ensemble.predict(cell_t, pg_ids_t, pg_mask_t)
            pred_expr = pred[0].cpu().numpy()
            diff_pred = np.abs(pred_expr - self._ctrl_expr)
            g_pred = set(np.where(diff_pred > self._de_threshold)[0].tolist())
            total += len(g_true & g_pred) / len(g_true)
            count += 1
        return total / max(count, 1)

    # ------------------------------------------------------------------
    # Accessors for policy / training code
    # ------------------------------------------------------------------

    @property
    def pool_embeddings(self) -> np.ndarray:
        return self._pool_embeddings

    @property
    def pool_uncertainties(self) -> np.ndarray:
        return self._pool_uncertainties

    @property
    def labeled_embeddings(self) -> np.ndarray:
        return self._labeled_embeddings

    @property
    def pool_size(self) -> int:
        return len(self._pool_conds)

    def get_candidate_features(
        self, state: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        partial_embs = self._embed_conditions(self._current_batch_conds)
        query = self.feature_extractor.build_query_features(
            pool_embeddings=self._pool_embeddings,
            pool_uncertainties=self._pool_uncertainties,
            labeled_embeddings=self._labeled_embeddings,
            partial_batch_embeddings=partial_embs,
        )
        phi = self.feature_extractor.build_candidate_features(
            pool_embeddings=self._pool_embeddings,
            pool_uncertainties=self._pool_uncertainties,
            labeled_embeddings=self._labeled_embeddings,
            partial_batch_embeddings=partial_embs,
            state_vector=state,
        )
        return query, phi
