#!/usr/bin/env bash
set -euo pipefail

INPUT="opt_opt_rs.xyz"
CHARGE="0.0"

CKPT_A="model.ckpt-1"
TAG_A="ckpt_1"

CKPT_B="model.ckpt-1000000"
TAG_B="ckpt_1000000"

PY=python  # your env-physnet python (3.5.5)

# 1) Optimize independently
echo "=== OPT A ==="
$PY optimize_nn.py -i "$INPUT" --checkpoint "$CKPT_A" --tag "$TAG_A" --charge "$CHARGE"

echo "=== OPT B ==="
$PY optimize_nn.py -i "$INPUT" --checkpoint "$CKPT_B" --tag "$TAG_B" --charge "$CHARGE"

# 2) Vibrations + mode trajectories on each optimized structure
OPT_A="opt_$(basename "$INPUT" .xyz)_${TAG_A}.xyz"
OPT_B="opt_$(basename "$INPUT" .xyz)_${TAG_B}.xyz"

echo "=== VIB A ==="
$PY normal_modes_one.py -i "$OPT_A" --checkpoint "$CKPT_A" --tag "$TAG_A" --charge "$CHARGE" \
  --freq_min 20 --amp 0.5 --frames 1000 --path linear --export_modes all

echo "=== VIB B ==="
$PY normal_modes_one.py -i "$OPT_B" --checkpoint "$CKPT_B" --tag "$TAG_B" --charge "$CHARGE" \
  --freq_min 20 --amp 0.5 --frames 1000 --path linear --export_modes all

echo "DONE."
echo "Outputs:"
echo "  opt_*_${TAG_A}.xyz , opt_*_${TAG_B}.xyz"
echo "  nm_${TAG_A}/ , nm_${TAG_B}/"

