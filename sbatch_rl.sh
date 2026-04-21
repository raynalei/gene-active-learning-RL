#!/bin/sh
#SBATCH -A cis250217p
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH -t 20:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH -o /ocean/projects/cis250217p/ylei5/reinforce/logs/%j.out
#SBATCH -e /ocean/projects/cis250217p/ylei5/reinforce/logs/%j.err

############################################
# Runtime environment
############################################
source /ocean/projects/cis250217p/shared/envs/miniconda3/bin/activate
conda activate yu_env

echo "===== NODE INFO ====="
hostname
echo "===== GPU INFO ====="
nvidia-smi || echo "No NVIDIA GPU visible."

export PYTHONUNBUFFERED=1
export PYTHONPATH=/ocean/projects/cis250217p/ylei5:$PYTHONPATH



############################################
# Paths
############################################
ROOT=/ocean/projects/cis250217p/ylei5/reinforce
GENE=$ROOT/norman_2019_scratch_geneformer_padded.npy
CELL=$ROOT/norman_2019_01B-resolution_singlecell_cell_embedding_t4_resolution.npy
H5AD=$ROOT/norman_2019_adata.h5ad
OUT=$ROOT/results/ppo

mkdir -p $ROOT/logs
mkdir -p $OUT

############################################
# Launch
############################################
cd $ROOT

python main.py \
    --gene_embeddings "$GENE" \
    --cell_embeddings "$CELL" \
    --h5ad            "$H5AD" \
    --config          configs/fast.yaml \
    --output_dir      "$OUT" \
    --device          cuda \
    --seed            42 \
    --checkpoint_every 1 \
    --bc_checkpoint results/rl/seed42/policy_bc.pt \
    --override active_learning.batch_size=2
