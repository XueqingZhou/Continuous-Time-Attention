#!/bin/bash

set -e

if [ -z "${1}" ]; then
  echo "Usage: $0 <offline_bundle_dir>"
  exit 1
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

BUNDLE_DIR="${1}"

echo "Restoring offline bundle from: ${BUNDLE_DIR}"

echo "Verifying checksums..."
(cd "${BUNDLE_DIR}" && sha256sum -c manifest.sha256)

echo "Restoring HuggingFace cache to ${HOME}/.cache/huggingface ..."
mkdir -p "${HOME}/.cache"
tar xzf "${BUNDLE_DIR}/hf_cache.tar.gz" -C "${HOME}"

echo "Restoring BERT tokenizer to ${REPO_DIR}/tokenizer/bert-base-uncased ..."
mkdir -p "${REPO_DIR}/tokenizer"
tar xzf "${BUNDLE_DIR}/tokenizer_bert-base-uncased.tar.gz" -C "${REPO_DIR}/tokenizer"

echo "Restoring tinyllama tokenizer to ${REPO_DIR}/local_models/tinyllama ..."
mkdir -p "${REPO_DIR}/local_models"
tar xzf "${BUNDLE_DIR}/tokenizer_tinyllama.tar.gz" -C "${REPO_DIR}/local_models"

echo "Done."
