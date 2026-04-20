import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from predict import (
    GeneCellTransformerPredictor,
    TrainConfig,
    build_dataloader,
    evaluate,
    get_cached_num_guides,
    load_expression_from_h5ad,
    sanitize_gene_embeddings,
    train_one_epoch,
)


ENSEMBLE_SIZE = 5


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fit_ensemble_member(
    gene_embeddings: np.ndarray,
    cell_embeddings: np.ndarray,
    expression_matrix: np.ndarray,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    config: TrainConfig,
    seed: int,
    model_dim: int = 256,
    num_heads: int = 8,
    num_layers: int = 4,
    ff_dim: int = 512,
    dropout: float = 0.1,
):
    set_global_seed(seed)

    model = GeneCellTransformerPredictor(
        gene_embeddings=gene_embeddings,
        cell_dim=cell_embeddings.shape[1],
        num_guides=get_cached_num_guides(),
        model_dim=model_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        ff_dim=ff_dim,
        dropout=dropout,
    ).to(config.device)

    train_loader = build_dataloader(
        cell_embeddings=cell_embeddings,
        expression_matrix=expression_matrix,
        indices=train_indices,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = build_dataloader(
        cell_embeddings=cell_embeddings,
        expression_matrix=expression_matrix,
        indices=val_indices,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(config.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, config.device)
        val_loss = evaluate(model, val_loader, criterion, config.device)

        print(
            f"[Member {seed}] Epoch {epoch + 1:03d} | "
            f"Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_val_loss


def fit_ensemble(
    gene_embeddings: np.ndarray,
    cell_embeddings: np.ndarray,
    expression_matrix: np.ndarray,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    config: TrainConfig,
    base_seed: int,
    model_dim: int = 256,
    num_heads: int = 8,
    num_layers: int = 4,
    ff_dim: int = 512,
    dropout: float = 0.1,
):
    gene_embeddings = sanitize_gene_embeddings(gene_embeddings)

    members = []
    member_metrics = []

    for member_idx in range(ENSEMBLE_SIZE):
        member_seed = base_seed + member_idx
        print(f"Training ensemble member {member_idx + 1}/{ENSEMBLE_SIZE} with seed {member_seed}")
        model, best_val_loss = fit_ensemble_member(
            gene_embeddings=gene_embeddings,
            cell_embeddings=cell_embeddings,
            expression_matrix=expression_matrix,
            train_indices=train_indices,
            val_indices=val_indices,
            config=config,
            seed=member_seed,
            model_dim=model_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_dim=ff_dim,
            dropout=dropout,
        )
        members.append(model)
        member_metrics.append(
            {
                "member_index": member_idx,
                "seed": member_seed,
                "best_val_loss": float(best_val_loss),
            }
        )

    return members, member_metrics


@torch.no_grad()
def predict_with_ensemble(
    models,
    cell_embeddings: np.ndarray,
    expression_matrix: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    device: str,
):
    loader = build_dataloader(
        cell_embeddings=cell_embeddings,
        expression_matrix=expression_matrix,
        indices=indices,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    for model in models:
        model.to(device)
        model.eval()

    batch_means = []
    batch_stds = []
    batch_targets = []

    for cell_emb, expr, pert_gene_ids, pert_gene_mask in loader:
        cell_emb = cell_emb.to(device)
        pert_gene_ids = pert_gene_ids.to(device)
        pert_gene_mask = pert_gene_mask.to(device)
        member_preds = []

        for model in models:
            member_preds.append(model(cell_emb, pert_gene_ids, pert_gene_mask))

        stacked_preds = torch.stack(member_preds, dim=0)
        batch_means.append(stacked_preds.mean(dim=0).cpu())
        batch_stds.append(stacked_preds.std(dim=0, unbiased=False).cpu())
        batch_targets.append(expr.cpu())

    mean_predictions = torch.cat(batch_means, dim=0).numpy()
    uncertainty = torch.cat(batch_stds, dim=0).numpy()
    targets = torch.cat(batch_targets, dim=0).numpy()
    return mean_predictions, uncertainty, targets


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--gene_embeddings", type=str, required=True,
                        help="Path to gene embedding .npy, shape [G, Dg]")
    parser.add_argument("--cell_embeddings", type=str, required=True,
                        help="Path to cell embedding .npy, shape [N, Dc]")
    parser.add_argument("--h5ad", type=str, required=True,
                        help="Path to h5ad file. adata.X must be [N, G]")

    parser.add_argument("--train_idx", type=str, default=None,
                        help="Optional .npy file for train indices")
    parser.add_argument("--val_idx", type=str, default=None,
                        help="Optional .npy file for val indices")
    parser.add_argument("--predict_idx", type=str, default=None,
                        help="Optional .npy file for indices used for ensemble inference")

    parser.add_argument("--save_path", type=str, default="ensemble_transformer_predictor.pt")
    parser.add_argument("--save_mean_path", type=str, default="ensemble_mean_predictions.npy")
    parser.add_argument("--save_uncertainty_path", type=str, default="ensemble_uncertainty.npy")

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--model_dim", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--ff_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)

    return parser.parse_args()


def main():
    args = parse_args()

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

    num_cells = cell_embeddings.shape[0]
    all_indices = np.arange(num_cells)

    if args.train_idx is not None and args.val_idx is not None:
        train_indices = np.load(args.train_idx)
        val_indices = np.load(args.val_idx)
    else:
        rng = np.random.default_rng(args.seed)
        rng.shuffle(all_indices)
        split = int(0.8 * num_cells)
        train_indices = all_indices[:split]
        val_indices = all_indices[split:]

    predict_indices = np.load(args.predict_idx) if args.predict_idx is not None else val_indices

    config = TrainConfig(
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
    )

    models, member_metrics = fit_ensemble(
        gene_embeddings=gene_embeddings,
        cell_embeddings=cell_embeddings,
        expression_matrix=expression_matrix,
        train_indices=train_indices,
        val_indices=val_indices,
        config=config,
        base_seed=args.seed,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
    )

    mean_predictions, uncertainty, targets = predict_with_ensemble(
        models=models,
        cell_embeddings=cell_embeddings,
        expression_matrix=expression_matrix,
        indices=predict_indices,
        batch_size=args.batch_size,
        device=config.device,
    )

    mean_mse = float(np.mean((mean_predictions - targets) ** 2))
    mean_uncertainty = float(uncertainty.mean())

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_type": "ensemble_transformer_predictor",
            "ensemble_size": ENSEMBLE_SIZE,
            "member_state_dicts": [model.state_dict() for model in models],
            "member_metrics": member_metrics,
            "gene_embeddings_shape": gene_embeddings.shape,
            "cell_embeddings_shape": cell_embeddings.shape,
            "expression_shape": expression_matrix.shape,
            "train_indices": train_indices,
            "val_indices": val_indices,
            "predict_indices": predict_indices,
            "args": vars(args),
        },
        save_path,
    )

    mean_path = Path(args.save_mean_path)
    mean_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(mean_path, mean_predictions)

    uncertainty_path = Path(args.save_uncertainty_path)
    uncertainty_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(uncertainty_path, uncertainty)

    print(f"Saved ensemble checkpoint to: {save_path}")
    print(f"Saved predictive mean to: {mean_path}")
    print(f"Saved predictive uncertainty to: {uncertainty_path}")
    print(f"Predictive mean MSE: {mean_mse:.6f}")
    print(f"Mean predictive std: {mean_uncertainty:.6f}")


if __name__ == "__main__":
    main()
