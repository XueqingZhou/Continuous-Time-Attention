#!/bin/bash

# Script to run all ablation studies

set -e

echo "================================================"
echo "  Ablation Studies for Continuous-Time Attention"
echo "================================================"

# Check tokenizer
if [ ! -d "./tokenizer/bert-base-uncased" ]; then
    echo "Tokenizer not found. Running preparation..."
    bash scripts/prepare_tokenizer.sh
fi

mkdir -p results/ablation

echo ""
echo "=========================================="
echo "  Ablation 1: Data Size"
echo "=========================================="

python experiments/run_ablation_datasize.py \
    --tokenizer_path ./tokenizer/bert-base-uncased \
    --data_sizes 0.001 0.01 0.05 0.1 \
    --max_length 512 \
    --batch_size 128 \
    --num_epochs 10 \
    --output_dir results/ablation

echo ""
echo "=========================================="
echo "  Ablation 2: PDE Steps (Table 4)"
echo "=========================================="

python experiments/run_ablation_pdesteps.py \
    --tokenizer_path ./tokenizer/bert-base-uncased \
    --pde_steps_list 0 1 2 4 8 \
    --max_length 512 \
    --batch_size 128 \
    --num_epochs 20 \
    --output_dir results/ablation

echo ""
echo "=========================================="
echo "  Ablation 3: PDE Types (Table 5)"
echo "=========================================="

python experiments/run_ablation_pdetype.py \
    --tokenizer_path ./tokenizer/bert-base-uncased \
    --pde_types diffusion wave reaction-diffusion advection-diffusion \
    --pde_steps 4 \
    --max_length 512 \
    --batch_size 128 \
    --num_epochs 20 \
    --output_dir results/ablation

echo ""
echo "================================================"
echo "  All ablation studies completed!"
echo "  Results saved in ./results/ablation/"
echo "================================================"

