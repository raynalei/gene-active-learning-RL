import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from tqdm import tqdm

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from predict import (
    load_expression_from_h5ad,
    sanitize_gene_embeddings,
    get_cached_pert_gene_ids,
    get_cached_pert_gene_mask,
    get_cached_num_guides,
    build_dataloader,
)
from predictor.ensemble import EnsemblePredictor
from predictor.trainer import PredictorTrainer

DEFAULT_OUTPUT_DIR = Path("random")


def random_query(unlabeled_indices: np.ndarray, query_size: int, rng: np.random.Generator):
    """Randomly sample new instances from the unlabeled pool."""
    query_size = min(query_size, len(unlabeled_indices))
    return rng.choice(unlabeled_indices, size=query_size, replace=False)


def enable_dropout_in_eval(model: nn.Module) -> None:
    """Keep dropout active for MC dropout while leaving the rest of the model in eval mode."""
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


@torch.no_grad()
def uncertainty_query(
    model,
    cell_embeddings: np.ndarray,
    expression_matrix: np.ndarray,
    unlabeled_indices: np.ndarray,
    query_size: int,
    batch_size: int,
    num_workers: int,
    mc_dropout_passes: int,
    device: str,
) -> np.ndarray:
    """Select instances with the largest predictive variance estimated by MC dropout."""
    query_size = min(query_size, len(unlabeled_indices))
    mc_dropout_passes = max(1, mc_dropout_passes)
    if query_size == 0:
        return np.array([], dtype=np.int64)

    loader = build_dataloader(
        cell_embeddings=cell_embeddings,
        expression_matrix=expression_matrix,
        indices=unlabeled_indices,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    model = model.to(device)
    model.eval()
    enable_dropout_in_eval(model)

    uncertainty_scores = []
    n_pool = len(unlabeled_indices)
    n_batches = len(loader)
    pbar = tqdm(
        loader,
        desc="Uncertainty (MC dropout)",
        leave=False,
        unit="batch",
        total=n_batches,
        dynamic_ncols=True,
    )
    for cell_emb, _, pert_gene_ids, pert_gene_mask in pbar:
        cell_emb = cell_emb.to(device)
        pert_gene_ids = pert_gene_ids.to(device)
        pert_gene_mask = pert_gene_mask.to(device)

        mc_predictions = []
        for _ in range(mc_dropout_passes):
            pred = model(cell_emb, pert_gene_ids, pert_gene_mask)
            mc_predictions.append(pred.unsqueeze(0))

        stacked = torch.cat(mc_predictions, dim=0)
        predictive_var = stacked.var(dim=0, unbiased=False).mean(dim=1)
        uncertainty_scores.append(predictive_var.cpu())
        pbar.set_postfix(
            pool=n_pool,
            pick=query_size,
            mc=mc_dropout_passes,
            refresh=False,
        )

    scores = torch.cat(uncertainty_scores, dim=0).numpy()
    kth = scores.shape[0] - query_size
    topk = np.argpartition(scores, kth)[kth:]
    topk = topk[np.argsort(scores[topk])[::-1]]
    return unlabeled_indices[topk]


@torch.no_grad()
def uncertainty_ensemble_query(
    ensemble: EnsemblePredictor,
    cell_embeddings: np.ndarray,
    expression_matrix: np.ndarray,
    unlabeled_indices: np.ndarray,
    query_size: int,
    batch_size: int,
    num_workers: int,
    device: str,
) -> np.ndarray:
    """Select instances with the largest predictive variance across ensemble members (epistemic)."""
    query_size = min(query_size, len(unlabeled_indices))
    if query_size == 0:
        return np.array([], dtype=np.int64)

    loader = build_dataloader(
        cell_embeddings=cell_embeddings,
        expression_matrix=expression_matrix,
        indices=unlabeled_indices,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    ensemble = ensemble.to(device)
    ensemble.eval()

    uncertainty_scores = []
    n_pool = len(unlabeled_indices)
    n_batches = len(loader)
    pbar = tqdm(
        loader,
        desc="Uncertainty (ensemble)",
        leave=False,
        unit="batch",
        total=n_batches,
        dynamic_ncols=True,
    )
    for cell_emb, _, pert_gene_ids, pert_gene_mask in pbar:
        cell_emb = cell_emb.to(device)
        pert_gene_ids = pert_gene_ids.to(device)
        pert_gene_mask = pert_gene_mask.to(device)

        u = ensemble.uncertainty(cell_emb, pert_gene_ids, pert_gene_mask)
        uncertainty_scores.append(u.cpu())
        pbar.set_postfix(
            pool=n_pool,
            pick=query_size,
            members=len(ensemble.members),
            refresh=False,
        )

    scores = torch.cat(uncertainty_scores, dim=0).numpy()
    kth = scores.shape[0] - query_size
    topk = np.argpartition(scores, kth)[kth:]
    topk = topk[np.argsort(scores[topk])[::-1]]
    return unlabeled_indices[topk]


@torch.no_grad()
def compute_perturbation_representations(
    model,
    indices: np.ndarray,
    num_cells: int,
    device: str,
    batch_size: int = 1024,
    show_progress: bool = False,
    progress_desc: str = "Pert embeddings",
) -> torch.Tensor:
    """Build one perturbation embedding per cell by averaging perturbed-gene embeddings."""
    pert_gene_ids = get_cached_pert_gene_ids(num_cells)
    pert_gene_mask = get_cached_pert_gene_mask(num_cells)
    representations = []

    model = model.to(device)
    model.eval()

    n = len(indices)
    n_batches = (n + batch_size - 1) // batch_size if n else 0
    batch_starts = range(0, n, batch_size)
    if show_progress and n_batches > 0:
        batch_starts = tqdm(
            batch_starts,
            total=n_batches,
            desc=progress_desc,
            leave=False,
            unit="batch",
            dynamic_ncols=True,
        )

    for start in batch_starts:
        batch_indices = indices[start:start + batch_size]
        batch_gene_ids = torch.tensor(
            pert_gene_ids[batch_indices],
            dtype=torch.long,
            device=device,
        )
        batch_gene_mask = torch.tensor(
            pert_gene_mask[batch_indices],
            dtype=torch.float32,
            device=device,
        )

        gene_embs = model.pert_gene_emb(batch_gene_ids)
        mask = batch_gene_mask.unsqueeze(-1)
        counts = mask.sum(dim=1)
        pert_sum = (gene_embs * mask).sum(dim=1)
        avg_pert = pert_sum / counts.clamp_min(1.0)
        ctrl_context = model.ctrl_embedding.unsqueeze(0).expand(len(batch_indices), -1)
        batch_repr = torch.where(counts > 0, avg_pert, ctrl_context)
        representations.append(batch_repr)

    return torch.cat(representations, dim=0)


def min_distance_to_reference(
    candidates: torch.Tensor,
    reference: torch.Tensor,
    chunk_size: int = 1024,
) -> torch.Tensor:
    """Compute each candidate's minimum Euclidean distance to the reference set."""
    if reference.numel() == 0:
        return torch.full(
            (candidates.shape[0],),
            float("inf"),
            device=candidates.device,
        )

    min_distances = []
    for start in range(0, candidates.shape[0], chunk_size):
        chunk = candidates[start:start + chunk_size]
        distances = torch.cdist(chunk, reference)
        min_distances.append(distances.min(dim=1).values)

    return torch.cat(min_distances, dim=0)


@torch.no_grad()
def diversity_query(
    model,
    labeled_indices: np.ndarray,
    unlabeled_indices: np.ndarray,
    query_size: int,
    num_cells: int,
    device: str,
    representation_batch_size: int = 1024,
    distance_chunk_size: int = 1024,
) -> np.ndarray:
    """Greedily choose perturbations farthest from the labeled set in perturbation space."""
    query_size = min(query_size, len(unlabeled_indices))
    if query_size == 0:
        return np.array([], dtype=np.int64)

    labeled_repr = compute_perturbation_representations(
        model=model,
        indices=labeled_indices,
        num_cells=num_cells,
        device=device,
        batch_size=representation_batch_size,
        show_progress=True,
        progress_desc="Diversity: labeled → repr",
    )
    unlabeled_repr = compute_perturbation_representations(
        model=model,
        indices=unlabeled_indices,
        num_cells=num_cells,
        device=device,
        batch_size=representation_batch_size,
        show_progress=True,
        progress_desc="Diversity: unlabeled → repr",
    )

    min_distances = min_distance_to_reference(
        candidates=unlabeled_repr,
        reference=labeled_repr,
        chunk_size=distance_chunk_size,
    )
    available = torch.ones(len(unlabeled_indices), dtype=torch.bool, device=device)
    selected_positions = []

    greedy_iter = range(query_size)
    greedy_iter = tqdm(
        greedy_iter,
        desc="Diversity: greedy pick",
        leave=False,
        unit="pick",
        total=query_size,
        dynamic_ncols=True,
    )
    for _ in greedy_iter:
        masked_distances = min_distances.masked_fill(~available, float("-inf"))
        next_position = int(torch.argmax(masked_distances).item())
        if not torch.isfinite(masked_distances[next_position]):
            break

        selected_positions.append(next_position)
        available[next_position] = False

        new_reference = unlabeled_repr[next_position:next_position + 1]
        for start in range(0, unlabeled_repr.shape[0], distance_chunk_size):
            chunk = unlabeled_repr[start:start + distance_chunk_size]
            distances = torch.cdist(chunk, new_reference).squeeze(1)
            min_distances[start:start + distance_chunk_size] = torch.minimum(
                min_distances[start:start + distance_chunk_size],
                distances,
            )

    return unlabeled_indices[np.array(selected_positions, dtype=np.int64)]


def query_unlabeled_pool(
    strategy: str,
    model,
    labeled_indices: np.ndarray,
    unlabeled_indices: np.ndarray,
    query_size: int,
    rng: np.random.Generator,
    cell_embeddings: np.ndarray,
    expression_matrix: np.ndarray,
    batch_size: int,
    num_workers: int,
    mc_dropout_passes: int,
    device: str,
    ensemble: Optional[EnsemblePredictor] = None,
) -> np.ndarray:
    if strategy == "random":
        return random_query(unlabeled_indices, query_size, rng)
    if strategy == "uncertainty":
        return uncertainty_query(
            model=model,
            cell_embeddings=cell_embeddings,
            expression_matrix=expression_matrix,
            unlabeled_indices=unlabeled_indices,
            query_size=query_size,
            batch_size=batch_size,
            num_workers=num_workers,
            mc_dropout_passes=mc_dropout_passes,
            device=device,
        )
    if strategy == "uncertainty_ensemble":
        if ensemble is None:
            raise ValueError("uncertainty_ensemble requires ensemble=...")
        return uncertainty_ensemble_query(
            ensemble=ensemble,
            cell_embeddings=cell_embeddings,
            expression_matrix=expression_matrix,
            unlabeled_indices=unlabeled_indices,
            query_size=query_size,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
        )
    if strategy == "diversity":
        return diversity_query(
            model=model,
            labeled_indices=labeled_indices,
            unlabeled_indices=unlabeled_indices,
            query_size=query_size,
            num_cells=cell_embeddings.shape[0],
            device=device,
        )
    raise ValueError(f"Unknown query strategy: {strategy}")


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

    parser.add_argument(
        "--query_strategy",
        type=str,
        default="random",
        choices=["random", "uncertainty", "uncertainty_ensemble", "diversity"],
        help=(
            "random | uncertainty (MC dropout on first member) | "
            "uncertainty_ensemble (variance across ensemble members) | diversity."
        ),
    )
    parser.add_argument(
        "--mc_dropout_passes",
        type=int,
        default=8,
        help="Stochastic forward passes for --query_strategy uncertainty (MC dropout) only.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="DataLoader workers for uncertainty query (0 = main process only).",
    )

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

    default_method_names = {
        "random": "Random",
        "uncertainty": "Uncertainty",
        "uncertainty_ensemble": "Uncertainty Ensemble",
        "diversity": "Diversity",
    }
    if args.method_name == "Random":
        args.method_name = default_method_names[args.query_strategy]

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

    round_pbar = tqdm(
        range(args.rounds),
        desc="Active learning",
        unit="round",
        dynamic_ncols=True,
    )
    for round_id in round_pbar:
        labeled_indices = np.array(sorted(labeled_set), dtype=np.int64)

        if len(labeled_indices) < 2:
            tqdm.write("Not enough labeled samples to continue.")
            break

        round_pbar.set_postfix(
            strategy=args.query_strategy,
            stage="train",
            labeled=len(labeled_indices),
            pool=len(unlabeled_set),
            refresh=False,
        )

        train_data = _build_data_dict(
            cell_embeddings, expression_matrix, pert_gene_ids, pert_gene_mask,
            labeled_indices,
        )
        trainer.update(train_data, round_id)

        round_pbar.set_postfix(
            strategy=args.query_strategy,
            stage="eval",
            labeled=len(labeled_indices),
            pool=len(unlabeled_set),
            refresh=False,
        )

        val_mse = trainer.evaluate_on(val_data)
        test_mse = trainer.evaluate_on(test_data)

        results.append({
            "round": round_id,
            "num_labeled": int(len(labeled_indices)),
            "best_val_mse": float(val_mse),
            "test_mse": float(test_mse),
        })

        round_pbar.set_postfix(
            strategy=args.query_strategy,
            stage="query",
            labeled=len(labeled_indices),
            pool=len(unlabeled_set),
            val_mse=f"{val_mse:.4f}",
            test_mse=f"{test_mse:.4f}",
            refresh=True,
        )

        tqdm.write(
            f"[Round {round_id}] "
            f"Labeled: {len(labeled_indices)} | "
            f"Val MSE: {val_mse:.6f} | "
            f"Test MSE: {test_mse:.6f}"
        )

        if len(unlabeled_set) == 0:
            tqdm.write("Unlabeled pool is empty. Stop.")
            break

        unlabeled_indices = np.array(sorted(unlabeled_set))
        # MC dropout / diversity use the first ensemble member; uncertainty_ensemble uses full ensemble.
        query_model = ensemble.members[0]
        queried = query_unlabeled_pool(
            strategy=args.query_strategy,
            model=query_model,
            ensemble=ensemble if args.query_strategy == "uncertainty_ensemble" else None,
            labeled_indices=labeled_indices,
            unlabeled_indices=unlabeled_indices,
            query_size=args.query_size,
            rng=rng,
            cell_embeddings=cell_embeddings,
            expression_matrix=expression_matrix,
            batch_size=max(64, args.batch_size),
            num_workers=args.num_workers,
            mc_dropout_passes=args.mc_dropout_passes,
            device=device,
        )
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
