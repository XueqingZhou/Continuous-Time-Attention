#!/bin/bash

set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

OUT_DIR="${1:-${REPO_DIR}/offline_bundle}"

echo "Building offline bundle in: ${OUT_DIR}"
mkdir -p "${OUT_DIR}"

echo "Packing HuggingFace cache..."
tar czf "${OUT_DIR}/hf_cache.tar.gz" -C "${HOME}" .cache/huggingface

echo "Packing BERT tokenizer..."
tar czf "${OUT_DIR}/tokenizer_bert-base-uncased.tar.gz" -C "${REPO_DIR}/tokenizer" bert-base-uncased

echo "Packing tinyllama tokenizer..."
tar czf "${OUT_DIR}/tokenizer_tinyllama.tar.gz" -C "${REPO_DIR}/local_models" tinyllama

echo "Writing manifest..."
(
  cd "${OUT_DIR}"
  sha256sum hf_cache.tar.gz tokenizer_bert-base-uncased.tar.gz tokenizer_tinyllama.tar.gz \
    > manifest.sha256
)

echo "Done. Bundle files:"
ls -lh "${OUT_DIR}"
