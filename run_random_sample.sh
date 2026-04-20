#!/bin/bash
#SBATCH -A cis250217p
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH -t 8:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --job-name=random_sample
#SBATCH --output=/ocean/projects/cis250217p/ylei5/reinforce/random/logs/random_sample_%j.log

source /ocean/projects/cis250217p/shared/envs/miniconda3/bin/activate
conda activate yu_env

cd /ocean/projects/cis250217p/ylei5/reinforce


python random_sample.py \
  --gene_embeddings norman_2019_scratch_geneformer_padded.npy \
  --cell_embeddings norman_2019_01B-resolution_singlecell_cell_embedding_t4_resolution.npy \
  --h5ad norman_2019_adata.h5ad \
  --initial_labeled_size 100 \
  --query_size 100 \
  --rounds 200 \
  --epochs 10 \
  --batch_size 32 \
  --method_name Random \
  --save_curve random_al_curve.png \
  --save_curve_csv random_al_curve.csv
