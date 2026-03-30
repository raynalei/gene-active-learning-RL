import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

from predict import (
    load_expression_from_h5ad,
    fit_model,
    TrainConfig,
    build_dataloader,
)

DEFAULT_OUTPUT_DIR = Path("random")


def random_query(unlabeled_indices: np.ndarray, query_size: int, rng: np.random.Generator):
    """
    Randomly sample new instances from the unlabeled pool.
    """
    query_size = min(query_size, len(unlabeled_indices))
    chosen = rng.choice(unlabeled_indices, size=query_size, replace=False)
    return chosen

@torch.no_grad()
def evaluate_on_test(
    model,
    cell_embeddings: np.ndarray,
    expression_matrix: np.ndarray,
    test_indices: np.ndarray,
    batch_size: int = 64,
    device: str = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    loader = build_dataloader(
        cell_embeddings=cell_embeddings,
        expression_matrix=expression_matrix,
        indices=test_indices,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    criterion = nn.MSELoss()
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_samples = 0

    for cell_emb, expr, pert_gene_ids, pert_gene_mask in loader:
        cell_emb = cell_emb.to(device)
        expr = expr.to(device)
        pert_gene_ids = pert_gene_ids.to(device)
        pert_gene_mask = pert_gene_mask.to(device)

        pred = model(cell_emb, pert_gene_ids, pert_gene_mask)
        loss = criterion(pred, expr)

        bs = cell_emb.size(0)
        total_loss += loss.item() * bs
        total_samples += bs

    return total_loss / max(total_samples, 1)


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

    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=10)

    parser.add_argument("--model_dim", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--ff_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)

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

    rng = np.random.default_rng(args.seed)

    gene_embeddings = np.load(args.gene_embeddings).astype(np.float32)
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
    all_indices = np.arange(n_cells)

    # Split test set from the full dataset first
    train_pool_idx, test_idx = train_test_split(
        all_indices,
        test_size=args.test_ratio,
        random_state=args.seed,
        shuffle=True,
    )

    initial_labeled_size = min(args.initial_labeled_size, len(train_pool_idx))
    initial_labeled = rng.choice(train_pool_idx, size=initial_labeled_size, replace=False)

    labeled_set = set(initial_labeled.tolist())
    unlabeled_set = set(train_pool_idx.tolist()) - labeled_set

    results = []

    for round_id in range(args.rounds):
        labeled_indices = np.array(sorted(list(labeled_set)))
        unlabeled_indices = np.array(sorted(list(unlabeled_set)))

        if len(labeled_indices) < 2:
            print("Not enough labeled samples to continue.")
            break

        # Need at least one validation sample
        val_size = max(1, int(len(labeled_indices) * args.val_ratio))
        if val_size >= len(labeled_indices):
            val_size = len(labeled_indices) - 1

        if val_size <= 0:
            print("Not enough labeled samples after validation split.")
            break

        train_idx, val_idx = train_test_split(
            labeled_indices,
            test_size=val_size,
            random_state=args.seed + round_id,
            shuffle=True,
        )

        config = TrainConfig(
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            epochs=args.epochs,
        )

        model, best_val_loss = fit_model(
            gene_embeddings=gene_embeddings,
            cell_embeddings=cell_embeddings,
            expression_matrix=expression_matrix,
            train_indices=train_idx,
            val_indices=val_idx,
            config=config,
            model_dim=args.model_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            ff_dim=args.ff_dim,
            dropout=args.dropout,
        )

        test_mse = evaluate_on_test(
            model=model,
            cell_embeddings=cell_embeddings,
            expression_matrix=expression_matrix,
            test_indices=test_idx,
            batch_size=max(64, args.batch_size),
            device=config.device,
        )

        results.append({
            "round": round_id,
            "num_labeled": int(len(labeled_indices)),
            "best_val_mse": float(best_val_loss),
            "test_mse": float(test_mse),
        })

        print(
            f"[Round {round_id}] "
            f"Labeled: {len(labeled_indices)} | "
            f"Val MSE: {best_val_loss:.6f} | "
            f"Test MSE: {test_mse:.6f}"
        )

        if len(unlabeled_indices) == 0:
            print("Unlabeled pool is empty. Stop.")
            break

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
    val_mse = results_df["best_val_mse"].tolist()
    test_mse = results_df["test_mse"].tolist()

    plt.figure(figsize=(7, 5))
    plt.plot(
        num_labeled,
        test_mse,
        marker="o",
        linewidth=2.2,
        markersize=6,
        color="#1f77b4",
        label=f"{args.method_name} Test MSE",
    )
    plt.plot(
        num_labeled,
        val_mse,
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
