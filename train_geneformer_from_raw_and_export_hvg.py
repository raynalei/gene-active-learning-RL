import argparse
import math
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from scipy import sparse
from torch.utils.data import DataLoader, Dataset
from transformers import BertConfig, BertForMaskedLM


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a Geneformer-style MLM from adata.raw and export embeddings for adata.X HVGs."
    )
    parser.add_argument(
        "--input-h5ad",
        type=Path,
        default=Path("/ocean/projects/cis250217p/ylei5/reinforce/norman_2019_adata.h5ad"),
    )
    parser.add_argument(
        "--geneformer-dir",
        type=Path,
        default=Path("/ocean/projects/cis250217p/ylei5/Geneformer"),
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("/ocean/projects/cis250217p/ylei5/reinforce/norman_2019_scratch_geneformer"),
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--mask-prob", type=float, default=0.15)
    parser.add_argument("--target-sum", type=float, default=1e4)
    parser.add_argument("--max-position-embeddings", type=int, default=4096)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-hidden-layers", type=int, default=4)
    parser.add_argument("--num-attention-heads", type=int, default=4)
    parser.add_argument("--intermediate-size", type=int, default=512)
    parser.add_argument("--hidden-dropout-prob", type=float, default=0.02)
    parser.add_argument("--attention-dropout-prob", type=float, default=0.02)
    parser.add_argument(
        "--emb-layer",
        type=int,
        default=-1,
        choices=[-1, 0],
        help="-1 is second-to-last hidden layer, 0 is last hidden layer.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_count_column(obs: pd.DataFrame) -> str:
    for col in ["read_count", "UMI_count", "n_counts"]:
        if col in obs.columns:
            return col
    raise ValueError("obs must contain one of read_count, UMI_count, or n_counts.")


def get_ensembl_ids(var: pd.DataFrame) -> np.ndarray:
    for col in ["gene_id", "ensembl_id"]:
        if col in var.columns:
            return var[col].astype(str).to_numpy()
    raise ValueError("var must contain gene_id or ensembl_id.")


def load_geneformer_resources(geneformer_dir: Path):
    sys.path.insert(0, str(geneformer_dir))
    from geneformer import GENE_MEDIAN_FILE, TOKEN_DICTIONARY_FILE

    with open(GENE_MEDIAN_FILE, "rb") as f:
        gene_median_dict = pickle.load(f)
    with open(TOKEN_DICTIONARY_FILE, "rb") as f:
        gene_token_dict = pickle.load(f)
    return gene_median_dict, gene_token_dict


def build_training_sequences(adata, gene_median_dict, gene_token_dict, target_sum, max_position_embeddings):
    if adata.raw is None:
        raise ValueError("Input h5ad does not contain adata.raw.")

    raw_adata = adata.raw.to_adata()
    raw_adata.obs = adata.obs.copy()

    raw_gene_ids = get_ensembl_ids(raw_adata.var)
    keep_mask = np.array(
        [(gid in gene_token_dict) and (gid in gene_median_dict) for gid in raw_gene_ids]
    )
    if keep_mask.sum() == 0:
        raise ValueError("No raw genes overlap with Geneformer vocabulary.")

    raw_gene_ids = raw_gene_ids[keep_mask]
    raw_gene_tokens = np.array([gene_token_dict[g] for g in raw_gene_ids], dtype=np.int64)
    raw_gene_medians = np.array([gene_median_dict[g] for g in raw_gene_ids], dtype=np.float32)
    raw_x = raw_adata.X[:, keep_mask]
    raw_x = raw_x.tocsr() if sparse.issparse(raw_x) else np.asarray(raw_x)
    n_counts = raw_adata.obs[get_count_column(raw_adata.obs)].astype(float).to_numpy()

    hvg_gene_ids = get_ensembl_ids(adata.var)
    hvg_gene_ids = [gid for gid in hvg_gene_ids if gid in set(raw_gene_ids)]

    cls_token = gene_token_dict["<cls>"]
    eos_token = gene_token_dict["<eos>"]
    max_gene_tokens = max_position_embeddings - 2

    sequences = []
    ordered_gene_ids = []
    for i in range(raw_x.shape[0]):
        row = raw_x[i]
        row = row.toarray().ravel() if sparse.issparse(row) else np.asarray(row).ravel()
        scaled = row / max(n_counts[i], 1.0) * target_sum
        scaled = scaled / raw_gene_medians
        nz = np.flatnonzero(scaled > 0)
        if nz.size == 0:
            sequences.append(np.array([cls_token, eos_token], dtype=np.int64))
            ordered_gene_ids.append(np.array([], dtype=object))
            continue

        ranked_idx = nz[np.argsort(-scaled[nz])][:max_gene_tokens]
        ranked_tokens = raw_gene_tokens[ranked_idx]
        ranked_gene_ids = raw_gene_ids[ranked_idx]
        sequences.append(
            np.concatenate(
                [
                    np.array([cls_token], dtype=np.int64),
                    ranked_tokens,
                    np.array([eos_token], dtype=np.int64),
                ]
            )
        )
        ordered_gene_ids.append(ranked_gene_ids)

    return sequences, ordered_gene_ids, hvg_gene_ids


class TokenSequenceDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


class MLMCollator:
    def __init__(self, pad_token_id, mask_token_id, cls_token_id, eos_token_id, mask_prob):
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
        self.cls_token_id = cls_token_id
        self.eos_token_id = eos_token_id
        self.mask_prob = mask_prob

    def __call__(self, batch):
        max_len = max(len(x) for x in batch)
        input_ids = torch.full((len(batch), max_len), self.pad_token_id, dtype=torch.long)
        labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)

        for i, seq in enumerate(batch):
            seq = torch.tensor(seq, dtype=torch.long)
            seq_len = seq.numel()
            input_ids[i, :seq_len] = seq
            attention_mask[i, :seq_len] = 1

            maskable = (
                (seq != self.pad_token_id)
                & (seq != self.cls_token_id)
                & (seq != self.eos_token_id)
            )
            random_mask = torch.rand(seq_len) < self.mask_prob
            masked_positions = maskable & random_mask
            labels[i, :seq_len][masked_positions] = seq[masked_positions]
            input_ids[i, :seq_len][masked_positions] = self.mask_token_id

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def train_model(model, dataloader, device, epochs, lr, weight_decay, warmup_ratio):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = epochs * len(dataloader)
    warmup_steps = int(total_steps * warmup_ratio)

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / max(1, warmup_steps)
        return max(0.0, float(total_steps - current_step) / max(1, total_steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    step = 0
    for epoch in range(epochs):
        running_loss = 0.0
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            running_loss += loss.item()
            step += 1
        print(f"epoch={epoch+1} mean_loss={running_loss / max(1, len(dataloader)):.4f}")


def export_hvg_embeddings(model, sequences, ordered_gene_ids, hvg_gene_ids, batch_size, emb_layer, device):
    model.eval()
    layer_index = model.config.num_hidden_layers + emb_layer
    pad_token_id = model.config.pad_token_id

    gene_sums = {gid: np.zeros(model.config.hidden_size, dtype=np.float64) for gid in hvg_gene_ids}
    gene_counts = {gid: 0 for gid in hvg_gene_ids}
    hvg_gene_set = set(hvg_gene_ids)

    for start in range(0, len(sequences), batch_size):
        batch_sequences = sequences[start : start + batch_size]
        batch_genes = ordered_gene_ids[start : start + batch_size]
        max_len = max(len(x) for x in batch_sequences)
        padded = np.full((len(batch_sequences), max_len), pad_token_id, dtype=np.int64)
        mask = np.zeros((len(batch_sequences), max_len), dtype=np.int64)
        for i, seq in enumerate(batch_sequences):
            padded[i, : len(seq)] = seq
            mask[i, : len(seq)] = 1

        with torch.no_grad():
            outputs = model(
                input_ids=torch.tensor(padded, device=device),
                attention_mask=torch.tensor(mask, device=device),
            )
            hidden = outputs.hidden_states[layer_index].detach().cpu().numpy()

        for i, genes in enumerate(batch_genes):
            for pos, gene_id in enumerate(genes, start=1):
                if gene_id not in hvg_gene_set:
                    continue
                gene_sums[gene_id] += hidden[i, pos, :]
                gene_counts[gene_id] += 1

    valid_genes = [gid for gid in hvg_gene_ids if gene_counts[gid] > 0]
    emb = np.vstack([gene_sums[gid] / gene_counts[gid] for gid in valid_genes])
    return valid_genes, emb


def main():
    args = parse_args()
    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gene_median_dict, gene_token_dict = load_geneformer_resources(args.geneformer_dir)
    sequences, ordered_gene_ids, hvg_gene_ids = build_training_sequences(
        sc.read_h5ad(args.input_h5ad),
        gene_median_dict,
        gene_token_dict,
        args.target_sum,
        args.max_position_embeddings,
    )

    vocab_size = max(gene_token_dict.values()) + 1
    config = BertConfig(
        vocab_size=vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        intermediate_size=args.intermediate_size,
        hidden_dropout_prob=args.hidden_dropout_prob,
        attention_probs_dropout_prob=args.attention_dropout_prob,
        max_position_embeddings=args.max_position_embeddings,
        pad_token_id=gene_token_dict["<pad>"],
    )
    model = BertForMaskedLM(config).to(device)
    model.config.output_hidden_states = True

    dataset = TokenSequenceDataset(sequences)
    collator = MLMCollator(
        pad_token_id=gene_token_dict["<pad>"],
        mask_token_id=gene_token_dict["<mask>"],
        cls_token_id=gene_token_dict["<cls>"],
        eos_token_id=gene_token_dict["<eos>"],
        mask_prob=args.mask_prob,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator)

    train_model(
        model=model,
        dataloader=dataloader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
    )

    valid_genes, emb = export_hvg_embeddings(
        model=model,
        sequences=sequences,
        ordered_gene_ids=ordered_gene_ids,
        hvg_gene_ids=hvg_gene_ids,
        batch_size=args.batch_size,
        emb_layer=args.emb_layer,
        device=device,
    )

    csv_path = args.output_prefix.with_suffix(".csv")
    npy_path = args.output_prefix.with_suffix(".npy")
    genes_path = args.output_prefix.with_name(args.output_prefix.name + "_genes.txt")
    model_dir = args.output_prefix.with_name(args.output_prefix.name + "_model")

    pd.DataFrame(emb, index=valid_genes).to_csv(csv_path)
    np.save(npy_path, emb)
    genes_path.write_text("\n".join(valid_genes) + "\n")
    model.save_pretrained(model_dir)

    print(f"saved csv: {csv_path}")
    print(f"saved npy: {npy_path}")
    print(f"saved genes: {genes_path}")
    print(f"saved model: {model_dir}")
    print(f"num_hvg_genes={len(valid_genes)} emb_dim={emb.shape[1]}")


if __name__ == "__main__":
    main()
