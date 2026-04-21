"""
Condition-level random active learning baseline.

Mirrors ALEnvironment exactly:
  - Same cell-level splits (test 20%, id_val 10%, pool rest)
  - Same OOD-1 condition exclusion from pool
  - Selection at condition level (not cell level)
  - Labeling reveals all pool-eligible cells of the queried condition

Usage
-----
python random_sample.py \
    --gene_embeddings path/to/gene_embs.npy \
    --cell_embeddings path/to/cell_embs.npy \
    --h5ad            path/to/norman2019.h5ad \
    [--initial_labeled_size 4]   \
    [--query_size 1]             \
    [--rounds 20]
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from predict import (
    sanitize_gene_embeddings,
    get_cached_pert_gene_ids,
    get_cached_pert_gene_mask,
    get_cached_num_guides,
    _parse_guide_merged,
    _encode_guide_merged,
)
from predictor.ensemble import EnsemblePredictor
from predictor.trainer import PredictorTrainer

DEFAULT_OUTPUT_DIR = Path("random")


# ---------------------------------------------------------------------------
# Condition map (mirrors _load_norman2019 in al_env.py)
# ---------------------------------------------------------------------------

def _build_condition_map(adata) -> Tuple[List[str], List[List[int]], List[Set[str]]]:
    """
    Parse guide_merged to build condition → cell index mapping.

    Returns
    -------
    cond_names        : list of unique condition name strings
    cond_cell_indices : cond_cell_indices[c] = list of cell indices for condition c
    gene_sets         : gene_sets[c] = set of perturbed genes for condition c
    """
    guide_values = adata.obs["guide_merged"].tolist()
    parsed = [_parse_guide_merged(v) for v in guide_values]
    guide_strings = ["+".join(gs) if gs else "ctrl" for gs in parsed]
    unique_guides = list(dict.fromkeys(guide_strings))

    name2idx = {n: i for i, n in enumerate(unique_guides)}
    cond_cell_indices: List[List[int]] = [[] for _ in unique_guides]
    for cell_i, gs in enumerate(guide_strings):
        cond_cell_indices[name2idx[gs]].append(cell_i)

    gene_sets = [set(parsed[cond_cell_indices[c][0]]) for c in range(len(unique_guides))]
    return unique_guides, cond_cell_indices, gene_sets


# ---------------------------------------------------------------------------
# OOD-1 split (mirrors _build_ood_split in al_env.py)
# ---------------------------------------------------------------------------

def _build_ood_split(
    gene_sets: List[Set[str]],
    ood_val_fraction: float,
    seed: int,
) -> Tuple[List[int], List[int]]:
    """OOD-1: double knockouts where both genes appear as single knockouts."""
    single_genes: Set[str] = {g for gs in gene_sets if len(gs) == 1 for g in gs}
    ood1 = [i for i, gs in enumerate(gene_sets) if len(gs) == 2 and gs.issubset(single_genes)]
    rng = np.random.default_rng(seed)
    rng.shuffle(ood1)
    n_val = max(1, int(ood_val_fraction * len(ood1)))
    return ood1[:n_val], ood1[n_val:]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_data_dict(
    cell_embeddings: np.ndarray,
    expression_matrix: np.ndarray,
    pert_gene_ids: np.ndarray,
    pert_gene_mask: np.ndarray,
    indices: np.ndarray,
) -> Dict:
    idx = np.asarray(indices, dtype=np.int64)
    return {
        "cell_embeddings": cell_embeddings[idx],
        "expression_matrix": expression_matrix[idx],
        "pert_gene_ids": pert_gene_ids[idx],
        "pert_gene_mask": pert_gene_mask[idx],
    }


def _query_cells(
    cond_indices: List[int],
    cond_cell_indices: List[List[int]],
    pool_cell_set: Set[int],
) -> List[int]:
    """Return pool-eligible cell indices for queried conditions."""
    cells = []
    for c in cond_indices:
        for ci in cond_cell_indices[c]:
            if ci in pool_cell_set:
                cells.append(ci)
    return cells


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--gene_embeddings", type=str, required=True,
                        help="Path to gene embedding .npy, shape [G, Dg]")
    parser.add_argument("--cell_embeddings", type=str, required=True,
                        help="Path to cell embedding .npy, shape [N, Dc]")
    parser.add_argument("--h5ad", type=str, required=True,
                        help="Path to h5ad file with guide_merged obs column")

    # Condition-level AL parameters (mirror ALEnvironment defaults)
    parser.add_argument("--initial_labeled_size", type=int, default=4,
                        help="Number of conditions in initial labeled set D_0")
    parser.add_argument("--query_size", type=int, default=1,
                        help="Number of conditions to query per round (mirrors active_learning.batch_size)")
    parser.add_argument("--rounds", type=int, default=20,
                        help="Total AL rounds")

    # Data splits — must match ALEnvironment to ensure identical test set
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--id_val_ratio", type=float, default=0.1)
    parser.add_argument("--ood_val_fraction", type=float, default=0.2)
    parser.add_argument("--ood_split_seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)

    # Predictor hyperparameters
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--ensemble_size", type=int, default=5)
    parser.add_argument("--model_dim", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--ff_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", type=str, default=None)

    parser.add_argument("--method_name", type=str, default="Random")
    parser.add_argument("--save_curve", type=str,
                        default=str(DEFAULT_OUTPUT_DIR / "random_al_curve.png"))
    parser.add_argument("--save_curve_csv", type=str,
                        default=str(DEFAULT_OUTPUT_DIR / "random_al_curve.csv"))

    args = parser.parse_args()

    import scanpy as sc
    import scipy.sparse as sp

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)

    # ------------------------------------------------------------------
    # Load data — call _encode_guide_merged once to populate caches
    # ------------------------------------------------------------------
    gene_embeddings = sanitize_gene_embeddings(np.load(args.gene_embeddings))
    cell_embeddings = np.load(args.cell_embeddings).astype(np.float32)

    adata = sc.read_h5ad(args.h5ad)
    _encode_guide_merged(adata)   # populates PERT_GENE_IDS/MASK caches

    X = adata.X
    if sp.issparse(X):
        X = X.toarray()
    expression_matrix = np.asarray(X, dtype=np.float32)

    n_cells = cell_embeddings.shape[0]
    num_guides = get_cached_num_guides()
    pert_gene_ids = get_cached_pert_gene_ids(n_cells)
    pert_gene_mask = get_cached_pert_gene_mask(n_cells)

    print("gene_embeddings shape:", gene_embeddings.shape)
    print("cell_embeddings shape:", cell_embeddings.shape)
    print("expression_matrix shape:", expression_matrix.shape)

    # ------------------------------------------------------------------
    # Build condition map
    # ------------------------------------------------------------------
    cond_names, cond_cell_indices, gene_sets = _build_condition_map(adata)
    C = len(cond_names)
    print(f"Total conditions: {C}")

    # ------------------------------------------------------------------
    # Build predictor
    # ------------------------------------------------------------------
    predictor_config = {
        "predictor": {
            "ensemble_size": args.ensemble_size,
            "model_dim": args.model_dim,
            "num_heads": args.num_heads,
            "num_layers": args.num_layers,
            "ff_dim": args.ff_dim,
            "dropout": args.dropout,
            "full_retrain_every": 1,
            "finetune_epochs": args.epochs,
            "full_retrain_epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
        }
    }

    cell_dim = cell_embeddings.shape[1]
    ensemble = EnsemblePredictor(gene_embeddings, cell_dim, num_guides, predictor_config)
    trainer = PredictorTrainer(
        ensemble=ensemble,
        gene_embeddings=gene_embeddings,
        cell_dim=cell_dim,
        num_guides=num_guides,
        config=predictor_config,
        device=device,
    )

    # ------------------------------------------------------------------
    # Cell-level splits (identical to ALEnvironment)
    # ------------------------------------------------------------------
    all_cell_idx = np.arange(n_cells)
    trainable_cells, test_cells = train_test_split(
        all_cell_idx,
        test_size=args.test_ratio,
        random_state=args.ood_split_seed,
        shuffle=True,
    )
    train_cell_set: Set[int] = set(trainable_cells.tolist())

    # OOD-1 condition split
    ood_val_conds, _ = _build_ood_split(gene_sets, args.ood_val_fraction, args.ood_split_seed)
    ood_val_cond_set = set(ood_val_conds)

    ood_val_cells = [
        ci for c in ood_val_conds
        for ci in cond_cell_indices[c]
        if ci in train_cell_set
    ]
    ood_val_cell_set = set(ood_val_cells)

    remaining_trainable = np.array(
        [ci for ci in trainable_cells if ci not in ood_val_cell_set],
        dtype=np.int64,
    )

    if len(remaining_trainable) > 1:
        pool_cells_arr, id_val_cells = train_test_split(
            remaining_trainable,
            test_size=args.id_val_ratio,
            random_state=args.ood_split_seed + 1,
            shuffle=True,
        )
    else:
        pool_cells_arr = remaining_trainable
        id_val_cells = np.array([], dtype=np.int64)

    pool_cell_set: Set[int] = set(pool_cells_arr.tolist())

    # Condition pool: exclude OOD-val conditions; keep only those with pool cells
    cond_pool = [
        c for c in range(C)
        if c not in ood_val_cond_set
        and any(ci in pool_cell_set for ci in cond_cell_indices[c])
    ]
    print(f"Condition pool size: {len(cond_pool)}")

    # Eval data
    val_data = _build_data_dict(
        cell_embeddings, expression_matrix, pert_gene_ids, pert_gene_mask, id_val_cells
    ) if len(id_val_cells) > 0 else None

    test_data = _build_data_dict(
        cell_embeddings, expression_matrix, pert_gene_ids, pert_gene_mask, test_cells
    )

    ood_val_data = _build_data_dict(
        cell_embeddings, expression_matrix, pert_gene_ids, pert_gene_mask,
        np.array(ood_val_cells, dtype=np.int64)
    ) if ood_val_cells else None

    # ------------------------------------------------------------------
    # Initial labeled set (condition-level)
    # ------------------------------------------------------------------
    shuffled_pool = list(cond_pool)
    rng.shuffle(shuffled_pool)

    init_n = min(args.initial_labeled_size, len(shuffled_pool))
    labeled_conds: List[int] = list(shuffled_pool[:init_n])
    pool_conds: List[int] = list(shuffled_pool[init_n:])

    labeled_cells: Set[int] = set(_query_cells(labeled_conds, cond_cell_indices, pool_cell_set))

    results = []

    # ------------------------------------------------------------------
    # Active learning loop
    # ------------------------------------------------------------------
    for round_id in range(args.rounds):
        labeled_idx = np.array(sorted(labeled_cells), dtype=np.int64)

        if len(labeled_idx) < 2:
            print("Not enough labeled samples to continue.")
            break

        train_data = _build_data_dict(
            cell_embeddings, expression_matrix, pert_gene_ids, pert_gene_mask, labeled_idx
        )
        trainer.update(train_data, round_id)

        val_mse = trainer.evaluate_on(val_data) if val_data is not None else 0.0
        test_mse = trainer.evaluate_on(test_data)
        ood_val_mse = trainer.evaluate_on(ood_val_data) if ood_val_data is not None else 0.0

        results.append({
            "round": round_id,
            "num_labeled_cells": int(len(labeled_idx)),
            "num_labeled_conds": int(len(labeled_conds)),
            "best_val_mse": float(val_mse),
            "ood_val_mse": float(ood_val_mse),
            "test_mse": float(test_mse),
        })

        print(
            f"[Round {round_id}] "
            f"Labeled: {len(labeled_conds)} conds / {len(labeled_idx)} cells | "
            f"Val MSE: {val_mse:.6f} | OOD MSE: {ood_val_mse:.6f} | Test MSE: {test_mse:.6f}"
        )

        if not pool_conds:
            print("Condition pool exhausted. Stop.")
            break

        # Random condition query
        n_query = min(args.query_size, len(pool_conds))
        chosen_idx_set = set(rng.choice(len(pool_conds), size=n_query, replace=False).tolist())
        chosen_conds = [pool_conds[i] for i in chosen_idx_set]
        pool_conds = [c for i, c in enumerate(pool_conds) if i not in chosen_idx_set]

        labeled_conds.extend(chosen_conds)
        for ci in _query_cells(chosen_conds, cond_cell_indices, pool_cell_set):
            labeled_cells.add(ci)

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    if not results:
        print("No results to plot.")
        return

    results_df = pd.DataFrame(results)
    results_df.insert(0, "method", args.method_name)

    csv_path = Path(args.save_curve_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(csv_path, index=False)
    print(f"Active learning curve csv saved to: {csv_path}")

    save_path = Path(args.save_curve)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    num_labeled = results_df["num_labeled_cells"].tolist()
    val_mse_list = results_df["best_val_mse"].tolist()
    test_mse_list = results_df["test_mse"].tolist()

    plt.figure(figsize=(7, 5))
    plt.plot(num_labeled, test_mse_list, marker="o", linewidth=2.2, markersize=6,
             color="#1f77b4", label=f"{args.method_name} Test MSE")
    plt.plot(num_labeled, val_mse_list, marker="s", linewidth=1.5, markersize=5,
             linestyle="--", color="#9ecae1", label=f"{args.method_name} Val MSE")
    plt.xlabel("Number of labeled samples")
    plt.ylabel("MSE")
    plt.title("Active Learning Performance")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Active learning curve plot saved to: {save_path}")


if __name__ == "__main__":
    main()
