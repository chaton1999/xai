#!/bin/bash
set -euo pipefail

# Directory where the .npz and .json files will be stored
OUTDIR="data_features"
mkdir -p "${OUTDIR}"

# Common PhysNet hyperparameters
F=128
K=64
NUM_BLOCKS=5
SR_CUT=6.0
LR_CUT=10.0

echo "Output directory: ${OUTDIR}"
echo "Starting feature extraction..."

# -------- Part 1: checkpoints 1..1000 with step 1 --------
for ck in $(seq 1 1000); do
    echo ">>> Extracting checkpoint ${ck} (step 1 range)..."
    python extraction_w_b.py \
        --ckpt "all_runs/model.ckpt-${ck}/models/model.ckpt-${ck}" \
        --out "${OUTDIR}/${ck}_physnet_vars.npz" \
        --F ${F} --K ${K} --num-blocks ${NUM_BLOCKS} \
        --sr-cut ${SR_CUT} --lr-cut ${LR_CUT}
done

# -------- Part 2: checkpoints 2000..1000000 with step 1000 --------
# (1000 was already done above)
for ck in $(seq 2000 1000 1000000); do
    echo ">>> Extracting checkpoint ${ck} (step 1000 range)..."
    python extraction_w_b.py \
        --ckpt "all_runs/model.ckpt-${ck}/models/model.ckpt-${ck}" \
        --out "${OUTDIR}/${ck}_physnet_vars.npz" \
        --F ${F} --K ${K} --num-blocks ${NUM_BLOCKS} \
        --sr-cut ${SR_CUT} --lr-cut ${LR_CUT}
done

# Summary
N_FILES=$(ls -1 "${OUTDIR}"/*_physnet_vars.npz | wc -l || echo 0)
echo "--------------------------------------------------"
echo "Done. Number of .npz feature files in ${OUTDIR}: ${N_FILES}"
echo "(.json metadata files are next to each .npz)"

