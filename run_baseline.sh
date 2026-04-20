#!/bin/bash
# random_sample.py — 与 readme.md 中参数一致；在本仓库目录下提交运行。
# 用法: sbatch run_random_sample.sh   或   bash run_random_sample.sh

#SBATCH --job-name=diversity-ppo
#SBATCH --output=/home/hairuow/RL_project/gene-active-learning-RL-feature-ppo/logs/random_%j.out
#SBATCH --error=/home/hairuow/RL_project/gene-active-learning-RL-feature-ppo/logs/random_%j.err
#SBATCH --time=24:00:00
# #SBATCH --account=local
#SBATCH --partition=model3
#SBATCH --gres=gpu:1                   # 请求1个GPU
#SBATCH --cpus-per-task=8              # CPU核心数
#SBATCH --mem=64G                      # 内存，您可以根据需要调整
# #SBATCH --nodes=1                      # 单节点
# #SBATCH --ntasks-per-node=1

set -euo pipefail

PROJECT_ROOT="/home/hairuow/RL_project/gene-active-learning-RL-feature-ppo"
mkdir -p "${PROJECT_ROOT}/logs"

echo "作业开始时间: $(date)"
echo "作业ID: ${SLURM_JOB_ID:-local}"
echo "节点: ${SLURM_NODELIST:-localhost}"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "PROJECT_ROOT: ${PROJECT_ROOT}"

# module load python/3.9
# module load cuda/11.8
# module load cudnn/8.6

source ~/miniconda3/etc/profile.d/conda.sh
conda activate genai

cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# 可通过环境变量切换策略，例如: QUERY_STRATEGY=uncertainty sbatch run_random_sample.sh
QUERY_STRATEGY="${QUERY_STRATEGY:-random}"

case "${QUERY_STRATEGY}" in
  random)
    SAVE_PNG="random/random_al_curve.png"
    SAVE_CSV="random/random_al_curve.csv"
    EXTRA_ARGS=()
    ;;
  uncertainty)
    SAVE_PNG="random/uncertainty_al_curve.png"
    SAVE_CSV="random/uncertainty_al_curve.csv"
    EXTRA_ARGS=(--mc_dropout_passes 8 --num_workers 0)
    ;;
  uncertainty_ensemble)
    SAVE_PNG="random/uncertainty_ensemble_al_curve.png"
    SAVE_CSV="random/uncertainty_ensemble_al_curve.csv"
    EXTRA_ARGS=(--num_workers 0)
    ;;
  diversity)
    SAVE_PNG="random/diversity_al_curve.png"
    SAVE_CSV="random/diversity_al_curve.csv"
    EXTRA_ARGS=()
    ;;
  *)
    echo "Unknown QUERY_STRATEGY=${QUERY_STRATEGY}, use random|uncertainty|uncertainty_ensemble|diversity"
    exit 1
    ;;
esac

echo "开始 baseline.py (query_strategy=${QUERY_STRATEGY})..."
python baseline.py \
  --gene_embeddings norman_2019_scratch_geneformer_padded.npy \
  --cell_embeddings norman_2019_01B-resolution_singlecell_cell_embedding_t4_resolution.npy \
  --h5ad norman_2019_adata.h5ad \
  --initial_labeled_size 100 \
  --query_size 100 \
  --rounds 20 \
  --epochs 10 \
  --batch_size 32 \
  --query_strategy "${QUERY_STRATEGY}" \
  "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" \
  --save_curve "${SAVE_PNG}" \
  --save_curve_csv "${SAVE_CSV}"

echo "完成时间: $(date)"
