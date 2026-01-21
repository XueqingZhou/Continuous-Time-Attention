#!/bin/bash

# Script to run all experiments from the paper

set -e  # Exit on error

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REPO_DIR="$(cd -- "${SRC_DIR}/.." && pwd)"

echo "================================================"
echo "  Continuous-Time Attention - Full Experiments"
echo "================================================"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

# Ensure relative paths in configs resolve correctly
cd "${REPO_DIR}"

# Check if tokenizer exists (for classification tasks)
if [ ! -d "${REPO_DIR}/src/tokenizer/bert-base-uncased" ]; then
    echo "Tokenizer not found at ${REPO_DIR}/src/tokenizer/bert-base-uncased"
    echo "Please prepare it offline before running classification experiments."
    exit 1
fi

# Create results directory
mkdir -p "${REPO_DIR}/results"

echo ""
echo "=========================================="
echo "  Table 1: Classification Tasks"
echo "=========================================="

echo ""
echo "Running IMDb experiment..."
python "${SRC_DIR}/experiments/run_classification.py" \
    --config "${SRC_DIR}/configs/imdb.yaml" \
    --output_dir "${REPO_DIR}/results"

echo ""
echo "Running AG News experiment..."
python "${SRC_DIR}/experiments/run_classification.py" \
    --config "${SRC_DIR}/configs/ag_news.yaml" \
    --output_dir "${REPO_DIR}/results"

echo ""
echo "Running SST-2 experiment..."
python "${SRC_DIR}/experiments/run_classification.py" \
    --config "${SRC_DIR}/configs/sst2.yaml" \
    --output_dir "${REPO_DIR}/results"

echo ""
echo "=========================================="
echo "  Table 2: Character-Level IMDb (LRA)"
echo "=========================================="

python "${SRC_DIR}/experiments/run_char_level.py" \
    --config "${SRC_DIR}/configs/char_level_imdb.yaml" \
    --output_dir "${REPO_DIR}/results"

echo ""
echo "=========================================="
echo "  Tables 3-5: WikiText-103 LM"
echo "=========================================="

python "${SRC_DIR}/experiments/run_language_modeling.py" \
    --config "${SRC_DIR}/configs/wikitext103.yaml" \
    --output_dir "${REPO_DIR}/results"

echo ""
echo "================================================"
echo "  All experiments completed!"
echo "  Results saved in ${REPO_DIR}/results"
echo "================================================"

