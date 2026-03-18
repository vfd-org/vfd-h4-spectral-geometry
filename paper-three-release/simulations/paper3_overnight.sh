#!/bin/bash
# Paper III overnight production run
# Expected runtime: ~7-8 hours on RTX 3080
# Parameters: 25 beta x 80 ICs, T=2000

set -e
cd "$(dirname "$0")"

echo "=== Paper III Overnight Production Run ==="
echo "Start: $(date)"
echo ""

# Run the sweep
python3 -u paper3_gpu_sweep.py \
    --n_beta 25 \
    --n_ic 80 \
    --T 2000 \
    --batch 10 \
    --output results/paper3_production

echo ""
echo "=== Generating figures ==="

# Generate figures from production data
python3 -u paper3_plot_figures.py \
    --input results/paper3_production \
    --output ../papers/paper-003-h4-attractors/figures

echo ""
echo "=== Done ==="
echo "End: $(date)"
