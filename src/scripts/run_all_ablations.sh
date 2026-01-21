#!/bin/bash

# Script to run all ablation studies

set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REPO_DIR="$(cd -- "${SRC_DIR}/.." && pwd)"

echo "================================================"
echo "  Ablation Studies for Continuous-Time Attention"
echo "================================================"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

# Ensure relative paths in configs resolve correctly
cd "${REPO_DIR}"

mkdir -p "${REPO_DIR}/results/ablation"

echo ""
echo "=========================================="
echo "  Ablation 1: Data Size"
echo "=========================================="

python "${SRC_DIR}/experiments/run_ablation_datasize.py" \
    --config "${SRC_DIR}/configs/wikitext103.yaml" \
    --data_sizes 0.001 0.01 0.05 0.1 \
    --output_dir "${REPO_DIR}/results/ablation"

echo ""
echo "=========================================="
echo "  Ablation 2: PDE Steps (Table 4)"
echo "=========================================="

python "${SRC_DIR}/experiments/run_ablation_pdesteps.py" \
    --config "${SRC_DIR}/configs/wikitext103.yaml" \
    --pde_steps_list 0 1 2 4 8 \
    --output_dir "${REPO_DIR}/results/ablation"

echo ""
echo "=========================================="
echo "  Ablation 3: PDE Types (Table 5)"
echo "=========================================="

python "${SRC_DIR}/experiments/run_ablation_pdetype.py" \
    --config "${SRC_DIR}/configs/wikitext103.yaml" \
    --pde_types diffusion wave reaction-diffusion advection-diffusion \
    --output_dir "${REPO_DIR}/results/ablation"

echo ""
echo "================================================"
echo "  All ablation studies completed!"
echo "  Results saved in ${REPO_DIR}/results/ablation"
echo "================================================"

