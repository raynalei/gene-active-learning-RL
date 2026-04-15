import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from predict import (
    load_expression_from_h5ad,
    sanitize_gene_embeddings,
    get_cached_pert_gene_ids,
    get_cached_pert_gene_mask,
    get_cached_num_guides,
)
from predictor.ensemble import EnsemblePredictor
from predictor.trainer import PredictorTrainer

DEFAULT_OUTPUT_DIR = Path("random")


def random_query(unlabeled_indices: np.ndarray, query_size: int, rng: np.random.Generator):
    """Randomly sample new instances from the unlabeled pool."""
    query_size = min(query_size, len(unlabeled_indices))
    return rng.choice(unlabeled_indices, size=query_size, replace=False)


def _build_data_dict(
    cell_embeddings: np.ndarray,
    expression_matrix: np.ndarray,
    pert_gene_ids: np.ndarray,
    pert_gene_mask: np.ndarray,
    indices: np.ndarray,
) -> dict:
    idx = np.asarray(indices, dtype=np.int64)
    return {
        "cell_embeddings": cell_embeddings[idx],
        "expression_matrix": expression_matrix[idx],
        "pert_gene_ids": pert_gene_ids[idx],
        "pert_gene_mask": pert_gene_mask[idx],
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--gene_embeddings", type=str, required=True,
                        help="Path to gene embedding .npy, shape [G, Dg]")
    parser.add_argument("--cell_embeddings", type=str, required=True,
                        help="Path to cell embedding .npy, shape [N, Dc]")
    parser.add_argument("--h5ad", type=str, required=True,
                        help="Path to h5ad file. adata.X must be [N, G]")

    parser.add_argument("--initial_labeled_size", type=int, default=100)
    parser.add_argument("--query_size", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=10)

    # Fixed val set: carved from trainable cells before AL loop (matches RL env)
    parser.add_argument("--id_val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

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
    parser.add_argument(
        "--save_curve",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR / "random_al_curve.png"),
    )
    parser.add_argument(
        "--save_curve_csv",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR / "random_al_curve.csv"),
    )

    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    gene_embeddings = sanitize_gene_embeddings(np.load(args.gene_embeddings))
    cell_embeddings = np.load(args.cell_embeddings).astype(np.float32)
    expression_matrix = load_expression_from_h5ad(args.h5ad)

    print("gene_embeddings shape:", gene_embeddings.shape)
    print("cell_embeddings shape:", cell_embeddings.shape)
    print("expression_matrix shape:", expression_matrix.shape)

    assert expression_matrix.shape[0] == cell_embeddings.shape[0], (
        f"Number of cells mismatch: expression has {expression_matrix.shape[0]}, "
        f"cell_embeddings has {cell_embeddings.shape[0]}"
    )
    assert expression_matrix.shape[1] == gene_embeddings.shape[0], (
        f"Number of genes mismatch: expression has {expression_matrix.shape[1]}, "
        f"gene_embeddings has {gene_embeddings.shape[0]}"
    )

    n_cells = cell_embeddings.shape[0]
    num_guides = get_cached_num_guides()
    pert_gene_ids = get_cached_pert_gene_ids(n_cells)
    pert_gene_mask = get_cached_pert_gene_mask(n_cells)

    # ------------------------------------------------------------------
    # Build ensemble predictor — always full-retrain each round to match
    # the original random_sample behaviour (new model per round).
    # ------------------------------------------------------------------
    predictor_config = {
        "predictor": {
            "ensemble_size": args.ensemble_size,
            "model_dim": args.model_dim,
            "num_heads": args.num_heads,
            "num_layers": args.num_layers,
            "ff_dim": args.ff_dim,
            "dropout": args.dropout,
            "full_retrain_every": 1,   # reset weights every round
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
    # Fixed dataset splits (mirrors ALEnvironment in al_env.py)
    #
    # all cells
    # ├── test_cells       (test_ratio = 20%)  — permanent holdout
    # └── trainable_cells  (80%)
    #     ├── val_cells    (id_val_ratio = 10% of trainable) — fixed val
    #     └── pool_cells   — available for AL labeling
    # ------------------------------------------------------------------
    all_indices = np.arange(n_cells)
    trainable_idx, test_idx = train_test_split(
        all_indices,
        test_size=args.test_ratio,
        random_state=args.seed,
        shuffle=True,
    )
    pool_idx, val_idx = train_test_split(
        trainable_idx,
        test_size=args.id_val_ratio,
        random_state=args.seed + 1,
        shuffle=True,
    )

    val_data = _build_data_dict(
        cell_embeddings, expression_matrix, pert_gene_ids, pert_gene_mask, val_idx
    )
    test_data = _build_data_dict(
        cell_embeddings, expression_matrix, pert_gene_ids, pert_gene_mask, test_idx
    )

    # ------------------------------------------------------------------
    # Initial labeled set sampled from pool only (not val, not test)
    # ------------------------------------------------------------------
    initial_labeled_size = min(args.initial_labeled_size, len(pool_idx))
    initial_labeled = rng.choice(pool_idx, size=initial_labeled_size, replace=False)

    labeled_set = set(initial_labeled.tolist())
    unlabeled_set = set(pool_idx.tolist()) - labeled_set

    results = []

    for round_id in range(args.rounds):
        labeled_indices = np.array(sorted(labeled_set), dtype=np.int64)

        if len(labeled_indices) < 2:
            print("Not enough labeled samples to continue.")
            break

        train_data = _build_data_dict(
            cell_embeddings, expression_matrix, pert_gene_ids, pert_gene_mask,
            labeled_indices,
        )
        trainer.update(train_data, round_id)

        val_mse = trainer.evaluate_on(val_data)
        test_mse = trainer.evaluate_on(test_data)

        results.append({
            "round": round_id,
            "num_labeled": int(len(labeled_indices)),
            "best_val_mse": float(val_mse),
            "test_mse": float(test_mse),
        })

        print(
            f"[Round {round_id}] "
            f"Labeled: {len(labeled_indices)} | "
            f"Val MSE: {val_mse:.6f} | "
            f"Test MSE: {test_mse:.6f}"
        )

        if len(unlabeled_set) == 0:
            print("Unlabeled pool is empty. Stop.")
            break

        unlabeled_indices = np.array(sorted(unlabeled_set))
        queried = random_query(unlabeled_indices, args.query_size, rng)
        for idx in queried:
            labeled_set.add(int(idx))
            unlabeled_set.remove(int(idx))

    if not results:
        print("No results to plot.")
        return

    results_df = pd.DataFrame(results)
    results_df.insert(0, "method", args.method_name)

    csv_path = Path(args.save_curve_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(csv_path, index=False)

    save_path = Path(args.save_curve)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    num_labeled = results_df["num_labeled"].tolist()
    val_mse_list = results_df["best_val_mse"].tolist()
    test_mse_list = results_df["test_mse"].tolist()

    plt.figure(figsize=(7, 5))
    plt.plot(
        num_labeled,
        test_mse_list,
        marker="o",
        linewidth=2.2,
        markersize=6,
        color="#1f77b4",
        label=f"{args.method_name} Test MSE",
    )
    plt.plot(
        num_labeled,
        val_mse_list,
        marker="s",
        linewidth=1.5,
        markersize=5,
        linestyle="--",
        color="#9ecae1",
        label=f"{args.method_name} Val MSE",
    )
    plt.xlabel("Number of labeled samples")
    plt.ylabel("MSE")
    plt.title("Active Learning Performance")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"Active learning curve csv saved to: {csv_path}")
    print(f"Active learning curve plot saved to: {save_path}")


if __name__ == "__main__":
    main()
