"""
Condition-level active learning baselines.

Three query strategies — random, uncertainty (MC dropout or ensemble variance),
and diversity — all operating at **condition level** to match the RL framework
(ALEnvironment).

Mirrors ALEnvironment exactly:
  - Same cell-level splits (test 20%, id_val 10%, pool rest)
  - Same OOD-1 condition exclusion from pool
  - Selection at condition level (not cell level)
  - Labeling reveals all pool-eligible cells of the queried condition
  - initial_labeled_size = 4 conditions (matches configs/default.yaml)

Usage
-----
python baseline.py \
    --gene_embeddings path/to/gene_embs.npy \
    --cell_embeddings path/to/cell_embs.npy \
    --h5ad            path/to/norman2019.h5ad \
    --query_strategy  uncertainty_ensemble \
    [--initial_labeled_size 4]  \
    [--query_size 16]           \
    [--rounds 20]
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml

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
    sanitize_gene_embeddings,
    get_cached_pert_gene_ids,
    get_cached_pert_gene_mask,
    get_cached_num_guides,
    _parse_guide_merged,
    _encode_guide_merged,
    ExpressionDataset,
    train_one_epoch,
    evaluate,
)
from predictor.ensemble import EnsemblePredictor

DEFAULT_OUTPUT_DIR = Path("baselines")


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


def _condition_tensors(
    cond_indices: List[int],
    cond_cell_indices: List[List[int]],
    cell_embeddings: np.ndarray,
    pert_gene_ids: np.ndarray,
    pert_gene_mask: np.ndarray,
    device: str,
    chunk_size: int = 256,
):
    """
    Yield (mean_cell_emb_t, pg_ids_t, pg_mask_t, slice) for each chunk.

    mean_cell_emb : mean over all cells of the condition  — matches _mean_cell_emb[c]
                    in ALEnvironment.
    pg_ids / mask : from the first representative cell    — same for every cell of
                    a condition, matches _pg_ids[_cond_cell_idx[c][0]].
    """
    n = len(cond_indices)
    for start in range(0, n, chunk_size):
        chunk = cond_indices[start:start + chunk_size]
        mean_embs = np.stack([
            cell_embeddings[cond_cell_indices[c]].mean(axis=0) for c in chunk
        ])
        rep_cells = [cond_cell_indices[c][0] for c in chunk]
        cell_t = torch.tensor(mean_embs, dtype=torch.float32).to(device)
        pg_ids_t = torch.tensor(pert_gene_ids[rep_cells], dtype=torch.long).to(device)
        pg_mask_t = torch.tensor(pert_gene_mask[rep_cells], dtype=torch.float32).to(device)
        yield cell_t, pg_ids_t, pg_mask_t, slice(start, start + len(chunk))


# ---------------------------------------------------------------------------
# Query strategies (condition-level)
# ---------------------------------------------------------------------------

def enable_dropout_in_eval(model: nn.Module) -> None:
    """Keep dropout active for MC dropout while leaving the rest of the model in eval mode."""
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


@torch.no_grad()
def uncertainty_ensemble_query(
    ensemble: EnsemblePredictor,
    pool_conds: List[int],
    cond_cell_indices: List[List[int]],
    cell_embeddings: np.ndarray,
    pert_gene_ids: np.ndarray,
    pert_gene_mask: np.ndarray,
    query_size: int,
    device: str,
) -> List[int]:
    """
    Score each pool condition by ensemble variance (epistemic uncertainty),
    using mean cell embedding per condition — identical to ALEnvironment._compute_uncertainties.
    """
    query_size = min(query_size, len(pool_conds))
    if query_size == 0:
        return []

    ensemble.to(device)
    ensemble.eval()

    scores = np.empty(len(pool_conds), dtype=np.float32)
    for cell_t, pg_ids_t, pg_mask_t, sl in _condition_tensors(
        pool_conds, cond_cell_indices, cell_embeddings, pert_gene_ids, pert_gene_mask, device
    ):
        scores[sl] = ensemble.uncertainty(cell_t, pg_ids_t, pg_mask_t).cpu().numpy()

    topk_pos = np.argpartition(scores, -query_size)[-query_size:]
    topk_pos = topk_pos[np.argsort(scores[topk_pos])[::-1]]
    return [pool_conds[i] for i in topk_pos]


def uncertainty_mc_query(
    ensemble: EnsemblePredictor,
    pool_conds: List[int],
    cond_cell_indices: List[List[int]],
    cell_embeddings: np.ndarray,
    pert_gene_ids: np.ndarray,
    pert_gene_mask: np.ndarray,
    query_size: int,
    mc_dropout_passes: int,
    device: str,
) -> List[int]:
    """
    Score each pool condition by predictive variance via MC dropout on the first
    ensemble member, using mean cell embedding per condition.
    """
    query_size = min(query_size, len(pool_conds))
    if query_size == 0:
        return []

    model = ensemble.members[0].to(device)
    model.eval()
    enable_dropout_in_eval(model)

    scores = np.empty(len(pool_conds), dtype=np.float32)
    for cell_t, pg_ids_t, pg_mask_t, sl in _condition_tensors(
        pool_conds, cond_cell_indices, cell_embeddings, pert_gene_ids, pert_gene_mask, device
    ):
        mc_preds = []
        for _ in range(max(1, mc_dropout_passes)):
            with torch.no_grad():
                pred = model(cell_t, pg_ids_t, pg_mask_t)   # [B, G]
            mc_preds.append(pred.unsqueeze(0))
        stacked = torch.cat(mc_preds, dim=0)                # [T, B, G]
        var = stacked.var(dim=0, unbiased=False).mean(dim=1)  # [B]
        scores[sl] = var.cpu().numpy()

    topk_pos = np.argpartition(scores, -query_size)[-query_size:]
    topk_pos = topk_pos[np.argsort(scores[topk_pos])[::-1]]
    return [pool_conds[i] for i in topk_pos]


@torch.no_grad()
def diversity_query(
    ensemble: EnsemblePredictor,
    labeled_conds: List[int],
    pool_conds: List[int],
    cond_cell_indices: List[List[int]],
    cell_embeddings: np.ndarray,
    pert_gene_ids: np.ndarray,
    pert_gene_mask: np.ndarray,
    query_size: int,
    device: str,
) -> List[int]:
    """
    Greedy max-distance selection in ensemble embedding space — mirrors the
    pool_embeddings used by ALEnvironment (ensemble.get_embedding with mean cell emb).
    """
    query_size = min(query_size, len(pool_conds))
    if query_size == 0:
        return []

    ensemble.to(device)
    ensemble.eval()

    def _embed(cond_list: List[int]) -> torch.Tensor:
        parts = []
        for cell_t, pg_ids_t, pg_mask_t, _ in _condition_tensors(
            cond_list, cond_cell_indices, cell_embeddings, pert_gene_ids, pert_gene_mask, device
        ):
            parts.append(ensemble.get_embedding(cell_t, pg_ids_t, pg_mask_t).cpu())
        return torch.cat(parts, dim=0)  # [N, D]

    labeled_repr = _embed(labeled_conds) if labeled_conds else torch.empty(0)
    pool_repr = _embed(pool_conds)       # [P, D]

    # min distance of each pool condition to the labeled set
    if labeled_repr.numel() == 0:
        min_dists = torch.full((len(pool_conds),), float("inf"))
    else:
        min_dists = torch.cdist(pool_repr, labeled_repr.to(pool_repr.device)).min(dim=1).values

    available = torch.ones(len(pool_conds), dtype=torch.bool)
    selected: List[int] = []

    for _ in tqdm(range(query_size), desc="Diversity: greedy pick", leave=False,
                  unit="pick", dynamic_ncols=True):
        masked = min_dists.masked_fill(~available, float("-inf"))
        best = int(torch.argmax(masked).item())
        if not torch.isfinite(masked[best]):
            break
        selected.append(pool_conds[best])
        available[best] = False
        # update min distances with the newly selected condition
        new_dists = torch.cdist(
            pool_repr, pool_repr[best:best + 1]
        ).squeeze(1)
        min_dists = torch.minimum(min_dists, new_dists)

    return selected


def query_condition_pool(
    strategy: str,
    ensemble: EnsemblePredictor,
    labeled_conds: List[int],
    pool_conds: List[int],
    cond_cell_indices: List[List[int]],
    cell_embeddings: np.ndarray,
    pert_gene_ids: np.ndarray,
    pert_gene_mask: np.ndarray,
    query_size: int,
    rng: np.random.Generator,
    mc_dropout_passes: int,
    device: str,
) -> List[int]:
    if strategy == "random":
        n = min(query_size, len(pool_conds))
        chosen = rng.choice(len(pool_conds), size=n, replace=False).tolist()
        return [pool_conds[i] for i in chosen]

    if strategy == "uncertainty":
        return uncertainty_mc_query(
            ensemble=ensemble,
            pool_conds=pool_conds,
            cond_cell_indices=cond_cell_indices,
            cell_embeddings=cell_embeddings,
            pert_gene_ids=pert_gene_ids,
            pert_gene_mask=pert_gene_mask,
            query_size=query_size,
            mc_dropout_passes=mc_dropout_passes,
            device=device,
        )

    if strategy == "uncertainty_ensemble":
        return uncertainty_ensemble_query(
            ensemble=ensemble,
            pool_conds=pool_conds,
            cond_cell_indices=cond_cell_indices,
            cell_embeddings=cell_embeddings,
            pert_gene_ids=pert_gene_ids,
            pert_gene_mask=pert_gene_mask,
            query_size=query_size,
            device=device,
        )

    if strategy == "diversity":
        return diversity_query(
            ensemble=ensemble,
            labeled_conds=labeled_conds,
            pool_conds=pool_conds,
            cond_cell_indices=cond_cell_indices,
            cell_embeddings=cell_embeddings,
            pert_gene_ids=pert_gene_ids,
            pert_gene_mask=pert_gene_mask,
            query_size=query_size,
            device=device,
        )

    raise ValueError(f"Unknown query strategy: {strategy!r}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Condition-level AL baselines (random / uncertainty / diversity)"
    )

    parser.add_argument("--gene_embeddings", type=str, required=True,
                        help="Path to gene embedding .npy, shape [G, Dg]")
    parser.add_argument("--cell_embeddings", type=str, required=True,
                        help="Path to cell embedding .npy, shape [N, Dc]")
    parser.add_argument("--h5ad", type=str, required=True,
                        help="Path to h5ad file with guide_merged obs column")

    # Condition-level AL parameters — mirror ALEnvironment / configs/default.yaml
    parser.add_argument("--initial_labeled_size", type=int, default=4,
                        help="Number of conditions in initial labeled set D_0")
    parser.add_argument("--query_size", type=int, default=1,
                        help="Conditions to query per round "
                             "(mirrors active_learning.batch_size=16)")
    parser.add_argument("--rounds", type=int, default=20,
                        help="Total AL rounds")

    # Data splits — must match ALEnvironment to ensure identical test set
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--id_val_ratio", type=float, default=0.1)
    parser.add_argument("--ood_val_fraction", type=float, default=0.2)
    parser.add_argument("--ood_split_seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)

    # Predictor hyperparameters — defaults match configs/default.yaml
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=3,
                        help="Training epochs per round per member "
                             "(mirrors PredictorTrainer finetune_epochs=3)")
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
            "uncertainty_ensemble (variance across ensemble members) | diversity"
        ),
    )
    parser.add_argument(
        "--mc_dropout_passes",
        type=int,
        default=8,
        help="Stochastic forward passes for --query_strategy uncertainty (MC dropout) only.",
    )

    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config (e.g. configs/fast.yaml). "
                             "Values are applied as defaults; explicit CLI flags take priority.")
    parser.add_argument("--method_name", type=str, default=None,
                        help="Display name in output (default: derived from query_strategy)")
    parser.add_argument("--save_curve", type=str,
                        default=str(DEFAULT_OUTPUT_DIR / "al_curve.png"))
    parser.add_argument("--save_curve_csv", type=str,
                        default=str(DEFAULT_OUTPUT_DIR / "al_curve.csv"))

    args = parser.parse_args()

    # Apply config file values as defaults (CLI flags override)
    if args.config is not None:
        with open(args.config) as f:
            cfg: Dict[str, Any] = yaml.safe_load(f)
        cli_set = {a.dest for a in parser._actions if a.option_strings}
        explicitly_set = {
            k for k, v in vars(args).items()
            if k in cli_set and v != parser.get_default(k)
        }
        mapping = {
            # predictor
            "ensemble_size":        ("predictor", "ensemble_size"),
            "model_dim":            ("predictor", "model_dim"),
            "num_heads":            ("predictor", "num_heads"),
            "num_layers":           ("predictor", "num_layers"),
            "ff_dim":               ("predictor", "ff_dim"),
            "dropout":              ("predictor", "dropout"),
            "batch_size":           ("predictor", "batch_size"),
            "lr":                   ("predictor", "lr"),
            "weight_decay":         ("predictor", "weight_decay"),
            "epochs":               ("predictor", "finetune_epochs"),
            # active learning
            "initial_labeled_size": ("active_learning", "initial_labeled_size"),
            "query_size":           ("active_learning", "batch_size"),
            "rounds":               ("active_learning", "num_rounds"),
            "ood_val_fraction":     ("active_learning", "ood_val_fraction"),
            "ood_split_seed":       ("active_learning", "ood_split_seed"),
            "test_ratio":           ("active_learning", "test_ratio"),
            "id_val_ratio":         ("active_learning", "id_val_ratio"),
            # top-level
            "seed":                 ("seed",),
            "device":               ("device",),
        }
        for arg_key, cfg_path in mapping.items():
            if arg_key in explicitly_set:
                continue
            node = cfg
            for part in cfg_path[:-1]:
                node = node.get(part, {})
            val = node.get(cfg_path[-1]) if isinstance(node, dict) else cfg.get(cfg_path[0])
            if val is not None:
                setattr(args, arg_key, val)

    default_method_names = {
        "random": "Random",
        "uncertainty": "Uncertainty (MC)",
        "uncertainty_ensemble": "Uncertainty (Ensemble)",
        "diversity": "Diversity",
    }
    if args.method_name is None:
        args.method_name = default_method_names[args.query_strategy]

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
    _encode_guide_merged(adata)   # populates PERT_GENE_IDS / MASK caches

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
        }
    }

    cell_dim = cell_embeddings.shape[1]
    ensemble = EnsemblePredictor(gene_embeddings, cell_dim, num_guides, predictor_config)
    criterion = torch.nn.MSELoss()

    def _train_ensemble(data: Dict) -> None:
        """One epoch of training over all ensemble members."""
        dataset = ExpressionDataset(
            cell_embeddings=data["cell_embeddings"].astype(np.float32),
            expression_matrix=data["expression_matrix"].astype(np.float32),
            pert_gene_ids=data["pert_gene_ids"].astype(np.int64),
            pert_gene_mask=data["pert_gene_mask"].astype(np.float32),
        )
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=0, pin_memory=torch.cuda.is_available(),
        )
        for member in ensemble.members:
            member = member.to(device)
            optimizer = torch.optim.AdamW(
                member.parameters(), lr=args.lr, weight_decay=args.weight_decay
            )
            for _ in range(args.epochs):
                train_one_epoch(member, loader, optimizer, criterion, device)

    def _eval_ensemble(data: Dict) -> float:
        """Average MSE across ensemble members."""
        dataset = ExpressionDataset(
            cell_embeddings=data["cell_embeddings"].astype(np.float32),
            expression_matrix=data["expression_matrix"].astype(np.float32),
            pert_gene_ids=data["pert_gene_ids"].astype(np.int64),
            pert_gene_mask=data["pert_gene_mask"].astype(np.float32),
        )
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=0, pin_memory=torch.cuda.is_available(),
        )
        total = 0.0
        for member in ensemble.members:
            member = member.to(device)
            total += evaluate(member, loader, criterion, device)
        return total / max(ensemble.ensemble_size, 1)

    # ------------------------------------------------------------------
    # Cell-level splits (identical to ALEnvironment)
    # ------------------------------------------------------------------
    all_cell_idx = np.arange(n_cells)
    trainable_cells, test_cells = train_test_split(
        all_cell_idx,
        test_size=args.test_ratio,
        random_state=args.ood_split_seed,   # matches ALEnvironment
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
    round_pbar = tqdm(
        range(args.rounds),
        desc=f"AL [{args.query_strategy}]",
        unit="round",
        dynamic_ncols=True,
    )
    for round_id in round_pbar:
        labeled_idx = np.array(sorted(labeled_cells), dtype=np.int64)

        if len(labeled_idx) < 2:
            tqdm.write("Not enough labeled samples to continue.")
            break

        round_pbar.set_postfix(
            labeled_conds=len(labeled_conds),
            pool_conds=len(pool_conds),
            stage="train",
            refresh=False,
        )

        train_data = _build_data_dict(
            cell_embeddings, expression_matrix, pert_gene_ids, pert_gene_mask, labeled_idx
        )
        _train_ensemble(train_data)

        round_pbar.set_postfix(
            labeled_conds=len(labeled_conds),
            pool_conds=len(pool_conds),
            stage="eval",
            refresh=False,
        )

        val_mse = _eval_ensemble(val_data) if val_data is not None else 0.0
        test_mse = _eval_ensemble(test_data)
        ood_val_mse = _eval_ensemble(ood_val_data) if ood_val_data is not None else 0.0

        results.append({
            "round": round_id,
            "num_labeled_cells": int(len(labeled_idx)),
            "num_labeled_conds": int(len(labeled_conds)),
            "id_val_mse":  float(val_mse),       # matches rl round_log column name
            "ood_val_mse": float(ood_val_mse),
            "test_mse":    float(test_mse),
        })

        round_pbar.set_postfix(
            labeled_conds=len(labeled_conds),
            pool_conds=len(pool_conds),
            val_mse=f"{val_mse:.4f}",
            ood_mse=f"{ood_val_mse:.4f}",
            test_mse=f"{test_mse:.4f}",
            refresh=True,
        )
        tqdm.write(
            f"[Round {round_id}] "
            f"Labeled: {len(labeled_conds)} conds / {len(labeled_idx)} cells | "
            f"Val MSE: {val_mse:.6f} | OOD MSE: {ood_val_mse:.6f} | Test MSE: {test_mse:.6f}"
        )

        if not pool_conds:
            tqdm.write("Condition pool exhausted. Stop.")
            break

        round_pbar.set_postfix(
            labeled_conds=len(labeled_conds),
            pool_conds=len(pool_conds),
            stage="query",
            refresh=False,
        )

        chosen_conds = query_condition_pool(
            strategy=args.query_strategy,
            ensemble=ensemble,
            labeled_conds=labeled_conds,
            pool_conds=pool_conds,
            cond_cell_indices=cond_cell_indices,
            cell_embeddings=cell_embeddings,
            pert_gene_ids=pert_gene_ids,
            pert_gene_mask=pert_gene_mask,
            query_size=args.query_size,
            rng=rng,
            mc_dropout_passes=args.mc_dropout_passes,
            device=device,
        )

        chosen_set = set(chosen_conds)
        pool_conds = [c for c in pool_conds if c not in chosen_set]
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
    val_mse_list = results_df["id_val_mse"].tolist()
    test_mse_list = results_df["test_mse"].tolist()

    plt.figure(figsize=(7, 5))
    plt.plot(num_labeled, test_mse_list, marker="o", linewidth=2.2, markersize=6,
             color="#1f77b4", label=f"{args.method_name} Test MSE")
    plt.plot(num_labeled, val_mse_list, marker="s", linewidth=1.5, markersize=5,
             linestyle="--", color="#9ecae1", label=f"{args.method_name} Val MSE")
    plt.xlabel("Number of labeled cells")
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
