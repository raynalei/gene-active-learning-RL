import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pad Geneformer HVG embeddings back to the full HVG list using NaN for missing genes."
    )
    parser.add_argument(
        "--input-h5ad",
        type=Path,
        default=Path("/ocean/projects/cis250217p/ylei5/reinforce/norman_2019_adata.h5ad"),
        help="Input h5ad containing the full HVG list in adata.var.",
    )
    parser.add_argument(
        "--embedding-npy",
        type=Path,
        default=Path("/ocean/projects/cis250217p/ylei5/reinforce/norman_2019_scratch_geneformer.npy"),
        help="Numpy embedding array for the exported Geneformer genes.",
    )
    parser.add_argument(
        "--exported-genes",
        type=Path,
        default=Path("/ocean/projects/cis250217p/ylei5/reinforce/norman_2019_scratch_geneformer_genes.txt"),
        help="Gene list corresponding to rows in --embedding-npy.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("/ocean/projects/cis250217p/ylei5/reinforce/norman_2019_scratch_geneformer_padded"),
        help="Prefix for padded outputs.",
    )
    return parser.parse_args()


def get_hvg_gene_ids(adata):
    for col in ["gene_id", "ensembl_id"]:
        if col in adata.var.columns:
            return adata.var[col].astype(str).tolist()
    raise ValueError("adata.var must contain gene_id or ensembl_id.")


def main():
    args = parse_args()

    adata = sc.read_h5ad(args.input_h5ad, backed="r")
    hvg_gene_ids = get_hvg_gene_ids(adata)

    emb = np.load(args.embedding_npy)
    exported_genes = [
        line.strip()
        for line in args.exported_genes.read_text().splitlines()
        if line.strip()
    ]

    if emb.shape[0] != len(exported_genes):
        raise ValueError(
            f"Embedding rows ({emb.shape[0]}) do not match exported genes ({len(exported_genes)})."
        )

    gene_to_idx = {gene: i for i, gene in enumerate(exported_genes)}
    padded = np.full((len(hvg_gene_ids), emb.shape[1]), np.nan, dtype=np.float32)

    found = 0
    for i, gene in enumerate(hvg_gene_ids):
        idx = gene_to_idx.get(gene)
        if idx is None:
            continue
        padded[i] = emb[idx]
        found += 1

    npy_path = args.output_prefix.with_suffix(".npy")
    csv_path = args.output_prefix.with_suffix(".csv")
    genes_path = args.output_prefix.with_name(args.output_prefix.name + "_genes.txt")
    missing_path = args.output_prefix.with_name(args.output_prefix.name + "_missing_genes.txt")

    np.save(npy_path, padded)
    pd.DataFrame(padded, index=hvg_gene_ids).to_csv(csv_path, na_rep="NA")
    genes_path.write_text("\n".join(hvg_gene_ids) + "\n")
    missing_genes = [gene for gene in hvg_gene_ids if gene not in gene_to_idx]
    missing_path.write_text("\n".join(missing_genes) + ("\n" if missing_genes else ""))

    print(f"saved padded npy: {npy_path}")
    print(f"saved padded csv: {csv_path}")
    print(f"saved full hvg genes: {genes_path}")
    print(f"saved missing genes: {missing_path}")
    print(f"original_rows={emb.shape[0]} padded_rows={padded.shape[0]} emb_dim={padded.shape[1]}")
    print(f"found={found} missing={len(missing_genes)}")


if __name__ == "__main__":
    main()
