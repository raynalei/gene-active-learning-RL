#!/usr/bin/env bash
# Run all ablation experiments (reward components + pipeline stages).
#
# Required env vars:
#   GENE   path to gene_embeddings.npy
#   CELL   path to cell_embeddings.npy
#   H5AD   path to data.h5ad
#
# Optional env vars:
#   OUTPUT_BASE   output root dir  (default: results/ablation)
#   DEVICE        cuda | cpu       (default: cuda)
#   SEED          integer          (default: 42)
#
# Usage:
#   GENE=data/gene.npy CELL=data/cell.npy H5AD=data/norman.h5ad bash scripts/run_ablation.sh

set -euo pipefail

GENE=${GENE:?Error: GENE is not set}
CELL=${CELL:?Error: CELL is not set}
H5AD=${H5AD:?Error: H5AD is not set}
OUTPUT_BASE=${OUTPUT_BASE:-results/ablation}
DEVICE=${DEVICE:-cuda}
SEED=${SEED:-42}

cd "$(dirname "$0")/.."

DATA="--gene_embeddings $GENE --cell_embeddings $CELL --h5ad $H5AD"
BASE="$DATA --device $DEVICE --seed $SEED"

echo "========================================"
echo "Ablation experiments  seed=$SEED"
echo "Output base: $OUTPUT_BASE"
echo "========================================"

# ------------------------------------------------------------------
# Reward component ablations (set one weight to 0 at a time)
# ------------------------------------------------------------------
for component in w_cov w_unc w_des w_red; do
    OUT="$OUTPUT_BASE/no_${component}/seed${SEED}"
    echo ""
    echo "--- reward.${component}=0  ->  $OUT"
    python main.py $BASE \
        --output_dir "$OUT" \
        --override "reward.${component}=0"
done

# ------------------------------------------------------------------
# Pipeline ablations
# ------------------------------------------------------------------
echo ""
echo "--- no BC warm-start  ->  $OUTPUT_BASE/no_bc/seed${SEED}"
python main.py $BASE \
    --output_dir "$OUTPUT_BASE/no_bc/seed${SEED}" \
    --no_bc

echo ""
echo "--- no Dyna  ->  $OUTPUT_BASE/no_dyna/seed${SEED}"
python main.py $BASE \
    --output_dir "$OUTPUT_BASE/no_dyna/seed${SEED}" \
    --no_dyna

echo ""
echo "--- no BC + no Dyna  ->  $OUTPUT_BASE/no_bc_no_dyna/seed${SEED}"
python main.py $BASE \
    --output_dir "$OUTPUT_BASE/no_bc_no_dyna/seed${SEED}" \
    --no_bc --no_dyna

echo ""
echo "========================================"
echo "All ablations done. Results in: $OUTPUT_BASE"
echo "========================================"
