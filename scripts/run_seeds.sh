#!/usr/bin/env bash
# Run RL + Random baselines across multiple seeds, then aggregate results.
#
# Required env vars:
#   GENE   path to gene_embeddings.npy
#   CELL   path to cell_embeddings.npy
#   H5AD   path to data.h5ad
#
# Optional env vars:
#   OUTPUT_BASE   output root dir              (default: results)
#   DEVICE        cuda | cpu                   (default: cuda)
#   SEEDS         space-separated seed list    (default: "0 1 2 3 4")
#
# Usage:
#   GENE=data/gene.npy CELL=data/cell.npy H5AD=data/norman.h5ad bash scripts/run_seeds.sh

set -euo pipefail

GENE=${GENE:?Error: GENE is not set}
CELL=${CELL:?Error: CELL is not set}
H5AD=${H5AD:?Error: H5AD is not set}
OUTPUT_BASE=${OUTPUT_BASE:-results}
DEVICE=${DEVICE:-cuda}
SEEDS=${SEEDS:-"0 1 2 3 4"}

cd "$(dirname "$0")/.."

DATA="--gene_embeddings $GENE --cell_embeddings $CELL --h5ad $H5AD"

echo "========================================"
echo "Multi-seed runs: seeds=[$SEEDS]"
echo "Output base: $OUTPUT_BASE"
echo "========================================"

for SEED in $SEEDS; do
    echo ""
    echo "--- RL  seed=$SEED  ->  $OUTPUT_BASE/rl/seed${SEED}"
    python main.py $DATA \
        --device "$DEVICE" \
        --seed "$SEED" \
        --output_dir "$OUTPUT_BASE/rl/seed${SEED}"

    echo ""
    echo "--- Random  seed=$SEED  ->  $OUTPUT_BASE/random/seed${SEED}"
    mkdir -p "$OUTPUT_BASE/random/seed${SEED}"
    python random_sample.py $DATA \
        --device "$DEVICE" \
        --seed "$SEED" \
        --save_curve_csv "$OUTPUT_BASE/random/seed${SEED}/random_al_curve.csv" \
        --save_curve     "$OUTPUT_BASE/random/seed${SEED}/random_al_curve.png"
done

echo ""
echo "========================================"
echo "All seeds done. Aggregating..."
echo "========================================"

python scripts/aggregate_results.py --output_base "$OUTPUT_BASE"

echo "Aggregation complete. Results in: $OUTPUT_BASE/aggregated/"
