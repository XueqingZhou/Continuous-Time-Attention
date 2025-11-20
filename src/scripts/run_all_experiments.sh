#!/bin/bash

# Script to run all experiments from the paper

set -e  # Exit on error

echo "================================================"
echo "  Continuous-Time Attention - Full Experiments"
echo "================================================"

# Check if tokenizer exists
if [ ! -d "./tokenizer/bert-base-uncased" ]; then
    echo "Tokenizer not found. Running preparation script..."
    bash scripts/prepare_tokenizer.sh
fi

# Create results directory
mkdir -p results

echo ""
echo "=========================================="
echo "  Table 1: Classification Tasks"
echo "=========================================="

echo ""
echo "Running IMDb experiment..."
python experiments/run_classification.py \
    --dataset imdb \
    --tokenizer_path ./tokenizer/bert-base-uncased \
    --max_length 256 \
    --embed_dim 128 \
    --num_heads 4 \
    --hidden_dim 256 \
    --num_layers 2 \
    --pde_type diffusion \
    --pde_steps 1 \
    --batch_size 64 \
    --num_epochs 5 \
    --output_dir results

echo ""
echo "Running AG News experiment..."
python experiments/run_classification.py \
    --dataset ag_news \
    --tokenizer_path ./tokenizer/bert-base-uncased \
    --max_length 256 \
    --batch_size 64 \
    --num_epochs 5 \
    --output_dir results

echo ""
echo "Running SST-2 experiment..."
python experiments/run_classification.py \
    --dataset sst2 \
    --tokenizer_path ./tokenizer/bert-base-uncased \
    --max_length 128 \
    --batch_size 512 \
    --num_epochs 2 \
    --output_dir results

echo ""
echo "=========================================="
echo "  Table 2: Character-Level IMDb (LRA)"
echo "=========================================="

python experiments/run_char_level.py \
    --max_length 2048 \
    --embed_dim 256 \
    --num_heads 4 \
    --hidden_dim 1024 \
    --num_layers 4 \
    --pde_type diffusion \
    --pde_steps 1 \
    --batch_size 16 \
    --num_epochs 10 \
    --output_dir results

echo ""
echo "=========================================="
echo "  Tables 3-5: WikiText-103 LM"
echo "=========================================="

python experiments/run_language_modeling.py \
    --tokenizer_path ./tokenizer/bert-base-uncased \
    --max_length 512 \
    --embed_dim 128 \
    --num_heads 4 \
    --hidden_dim 256 \
    --num_layers 2 \
    --pde_type diffusion \
    --pde_steps 4 \
    --batch_size 16 \
    --num_epochs 20 \
    --output_dir results

echo ""
echo "================================================"
echo "  All experiments completed!"
echo "  Results saved in ./results/"
echo "================================================"

