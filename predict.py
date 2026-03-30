import os
import math
import argparse
from dataclasses import dataclass

import numpy as np
import scanpy as sc
import scipy.sparse as sp

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


PERT_GENE_IDS_CACHE = None
PERT_GENE_MASK_CACHE = None
PERT_GENE_VOCAB_CACHE = None


def _normalize_guide_value(value) -> str:
    if value is None:
        return "__missing__"
    if isinstance(value, float) and math.isnan(value):
        return "__missing__"
    value_str = str(value)
    if value_str.lower() == "nan":
        return "__missing__"
    return value_str


def _parse_guide_merged(value) -> list[str]:
    value_str = _normalize_guide_value(value)
    if value_str in {"__missing__", "", "ctrl"}:
        return []

    genes = []
    for token in value_str.split("+"):
        gene = token.strip()
        if not gene or gene.lower() == "ctrl":
            continue
        genes.append(gene)

    # Preserve order while dropping duplicates inside one perturbation.
    return list(dict.fromkeys(genes))


def _encode_guide_merged(adata) -> tuple[np.ndarray, np.ndarray]:
    global PERT_GENE_IDS_CACHE, PERT_GENE_MASK_CACHE, PERT_GENE_VOCAB_CACHE

    if "guide_merged" not in adata.obs.columns:
        PERT_GENE_IDS_CACHE = np.zeros((adata.n_obs, 1), dtype=np.int64)
        PERT_GENE_MASK_CACHE = np.zeros((adata.n_obs, 1), dtype=np.float32)
        PERT_GENE_VOCAB_CACHE = np.array([], dtype=object)
        return PERT_GENE_IDS_CACHE, PERT_GENE_MASK_CACHE

    parsed_guides = [_parse_guide_merged(value) for value in adata.obs["guide_merged"].tolist()]
    perturbed_genes = sorted({gene for gene_list in parsed_guides for gene in gene_list})
    gene2idx = {gene: idx for idx, gene in enumerate(perturbed_genes)}

    max_genes_per_cell = max((len(gene_list) for gene_list in parsed_guides), default=0)
    padded_width = max(1, max_genes_per_cell)
    pert_gene_ids = np.zeros((adata.n_obs, padded_width), dtype=np.int64)
    pert_gene_mask = np.zeros((adata.n_obs, padded_width), dtype=np.float32)

    for row_idx, gene_list in enumerate(parsed_guides):
        for col_idx, gene in enumerate(gene_list):
            pert_gene_ids[row_idx, col_idx] = gene2idx[gene]
            pert_gene_mask[row_idx, col_idx] = 1.0

    PERT_GENE_IDS_CACHE = pert_gene_ids
    PERT_GENE_MASK_CACHE = pert_gene_mask
    PERT_GENE_VOCAB_CACHE = np.array(perturbed_genes, dtype=object)
    return PERT_GENE_IDS_CACHE, PERT_GENE_MASK_CACHE


def get_cached_pert_gene_ids(num_cells: int) -> np.ndarray:
    if PERT_GENE_IDS_CACHE is None or len(PERT_GENE_IDS_CACHE) != num_cells:
        return np.zeros((num_cells, 1), dtype=np.int64)
    return PERT_GENE_IDS_CACHE


def get_cached_pert_gene_mask(num_cells: int) -> np.ndarray:
    if PERT_GENE_MASK_CACHE is None or len(PERT_GENE_MASK_CACHE) != num_cells:
        return np.zeros((num_cells, 1), dtype=np.float32)
    return PERT_GENE_MASK_CACHE


def get_cached_num_guides() -> int:
    if PERT_GENE_VOCAB_CACHE is None:
        return 0
    return int(len(PERT_GENE_VOCAB_CACHE))


def load_expression_from_h5ad(h5ad_path: str) -> np.ndarray:
    """
    Load adata.X from h5ad as the target expression matrix.

    Returns
    -------
    X : np.ndarray
        Shape [num_cells, num_genes]
    """
    adata = sc.read_h5ad(h5ad_path)
    _encode_guide_merged(adata)
    X = adata.X

    if sp.issparse(X):
        X = X.toarray()

    X = np.asarray(X, dtype=np.float32)
    return X


def sanitize_gene_embeddings(gene_embeddings: np.ndarray) -> np.ndarray:
    """
    Replace non-finite values in gene embeddings so downstream training does not
    produce NaN/Inf losses when padded genes are present.
    """
    gene_embeddings = np.asarray(gene_embeddings, dtype=np.float32)
    invalid_mask = ~np.isfinite(gene_embeddings)
    invalid_count = int(invalid_mask.sum())
    if invalid_count > 0:
        gene_embeddings = np.nan_to_num(
            gene_embeddings,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
    return gene_embeddings


class ExpressionDataset(Dataset):
    """
    Dataset for expression prediction.

    cell_embeddings: [N, Dc]
    expression_matrix: [N, G]
    """
    def __init__(
        self,
        cell_embeddings: np.ndarray,
        expression_matrix: np.ndarray,
        pert_gene_ids: np.ndarray,
        pert_gene_mask: np.ndarray,
    ):
        assert cell_embeddings.shape[0] == expression_matrix.shape[0], \
            "cell_embeddings and expression_matrix must have the same number of cells."
        assert cell_embeddings.shape[0] == pert_gene_ids.shape[0], \
            "cell_embeddings and pert_gene_ids must have the same number of cells."
        assert cell_embeddings.shape[0] == pert_gene_mask.shape[0], \
            "cell_embeddings and pert_gene_mask must have the same number of cells."

        self.cell_embeddings = torch.tensor(cell_embeddings, dtype=torch.float32)
        self.expression_matrix = torch.tensor(expression_matrix, dtype=torch.float32)
        self.pert_gene_ids = torch.tensor(pert_gene_ids, dtype=torch.long)
        self.pert_gene_mask = torch.tensor(pert_gene_mask, dtype=torch.float32)

    def __len__(self):
        return self.cell_embeddings.shape[0]

    def __getitem__(self, idx):
        return (
            self.cell_embeddings[idx],
            self.expression_matrix[idx],
            self.pert_gene_ids[idx],
            self.pert_gene_mask[idx],
        )


class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding.
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) *
            (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])

        pe = pe.unsqueeze(0)  # [1, L, D]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, G, D]
        """
        return x + self.pe[:, :x.size(1), :]


class GeneCellTransformerPredictor(nn.Module):
    """
    Use gene embeddings + cell embeddings to predict expression for all genes.

    gene_embeddings: [G, Dg]  (fixed)
    cell_embedding: [B, Dc]

    Output:
        predicted expression: [B, G]
    """
    def __init__(
        self,
        gene_embeddings: np.ndarray,
        cell_dim: int,
        num_guides: int,
        model_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        ff_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        gene_embeddings = torch.tensor(gene_embeddings, dtype=torch.float32)
        self.num_genes = gene_embeddings.shape[0]
        gene_dim = gene_embeddings.shape[1]

        self.register_buffer("gene_embeddings", gene_embeddings)

        self.gene_proj = nn.Linear(gene_dim, model_dim)
        self.cell_proj = nn.Linear(cell_dim, model_dim)
        self.pert_gene_emb = nn.Embedding(max(num_guides, 1), model_dim)
        self.ctrl_embedding = nn.Parameter(torch.zeros(model_dim))

        self.pos_encoder = PositionalEncoding(model_dim, max_len=self.num_genes)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.out_head = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, 1)
        )

    def forward(
        self,
        cell_embedding: torch.Tensor,
        pert_gene_ids: torch.Tensor,
        pert_gene_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        cell_embedding: [B, Dc]
        pert_gene_ids: [B, P]
        pert_gene_mask: [B, P]
        returns: [B, G]
        """
        batch_size = cell_embedding.shape[0]

        # [G, Dg] -> [G, D]
        gene_tokens = self.gene_proj(self.gene_embeddings)  # [G, D]
        gene_tokens = gene_tokens.unsqueeze(0).expand(batch_size, -1, -1)  # [B, G, D]

        # [B, Dc] -> [B, D] -> [B, 1, D]
        cell_context = self.cell_proj(cell_embedding).unsqueeze(1)  # [B, 1, D]
        pert_gene_embs = self.pert_gene_emb(pert_gene_ids)  # [B, P, D]
        pert_gene_mask = pert_gene_mask.unsqueeze(-1)  # [B, P, 1]
        pert_counts = pert_gene_mask.sum(dim=1)  # [B, 1]
        pert_sum = (pert_gene_embs * pert_gene_mask).sum(dim=1)  # [B, D]
        avg_pert = pert_sum / pert_counts.clamp_min(1.0)
        ctrl_context = self.ctrl_embedding.unsqueeze(0).expand(batch_size, -1)
        pert_context = torch.where(pert_counts > 0, avg_pert, ctrl_context).unsqueeze(1)

        # Condition each gene token on the cell embedding and perturbation guide.
        x = gene_tokens + cell_context + pert_context  # [B, G, D]
        x = self.pos_encoder(x)
        x = self.transformer(x)

        # Predict one scalar per gene
        out = self.out_head(x).squeeze(-1)  # [B, G]
        return out


@dataclass
class TrainConfig:
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 1e-5
    epochs: int = 20
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def build_dataloader(
    cell_embeddings: np.ndarray,
    expression_matrix: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int
) -> DataLoader:
    pert_gene_ids = get_cached_pert_gene_ids(cell_embeddings.shape[0])
    pert_gene_mask = get_cached_pert_gene_mask(cell_embeddings.shape[0])
    dataset = ExpressionDataset(
        cell_embeddings=cell_embeddings[indices],
        expression_matrix=expression_matrix[indices],
        pert_gene_ids=pert_gene_ids[indices],
        pert_gene_mask=pert_gene_mask[indices],
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for cell_emb, expr, pert_gene_ids, pert_gene_mask in loader:
        cell_emb = cell_emb.to(device)
        expr = expr.to(device)
        pert_gene_ids = pert_gene_ids.to(device)
        pert_gene_mask = pert_gene_mask.to(device)

        optimizer.zero_grad()
        pred = model(cell_emb, pert_gene_ids, pert_gene_mask)
        loss = criterion(pred, expr)
        loss.backward()
        optimizer.step()

        bs = cell_emb.size(0)
        total_loss += loss.item() * bs
        total_samples += bs

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
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


def fit_model(
    gene_embeddings: np.ndarray,
    cell_embeddings: np.ndarray,
    expression_matrix: np.ndarray,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    config: TrainConfig,
    model_dim: int = 256,
    num_heads: int = 8,
    num_layers: int = 4,
    ff_dim: int = 512,
    dropout: float = 0.1,
):
    device = config.device
    gene_embeddings = sanitize_gene_embeddings(gene_embeddings)

    model = GeneCellTransformerPredictor(
        gene_embeddings=gene_embeddings,
        cell_dim=cell_embeddings.shape[1],
        num_guides=get_cached_num_guides(),
        model_dim=model_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        ff_dim=ff_dim,
        dropout=dropout,
    ).to(device)

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
        weight_decay=config.weight_decay
    )
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(config.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch + 1:03d} | "
            f"Train MSE: {train_loss:.6f} | "
            f"Val MSE: {val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_val_loss


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

    parser.add_argument("--save_path", type=str, default="transformer_predictor.pt")

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=20)

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
    print("num_guides:", get_cached_num_guides())

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
        rng = np.random.default_rng(42)
        rng.shuffle(all_indices)
        split = int(0.8 * num_cells)
        train_indices = all_indices[:split]
        val_indices = all_indices[split:]

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
        train_indices=train_indices,
        val_indices=val_indices,
        config=config,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "best_val_loss": best_val_loss,
            "gene_embeddings_shape": gene_embeddings.shape,
            "cell_embeddings_shape": cell_embeddings.shape,
            "expression_shape": expression_matrix.shape,
            "num_guides": get_cached_num_guides(),
            "args": vars(args),
        },
        args.save_path
    )

    print(f"Best validation MSE: {best_val_loss:.6f}")
    print(f"Model saved to: {args.save_path}")


if __name__ == "__main__":
    main()
