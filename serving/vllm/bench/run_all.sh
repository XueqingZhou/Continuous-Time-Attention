#!/usr/bin/env bash
set -euo pipefail

# One-command end-to-end vLLM baseline vs CTA, plus tables/plots/traces.
#
# Usage:
#   bash serving/vllm/bench/run_all.sh [MODEL_PATH]
#
# Defaults to the repo-local tinyllama under local_models/.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

MODEL_PATH="${1:-$ROOT_DIR/local_models/tinyllama}"
OUT_DIR="$ROOT_DIR/results/serving/vllm"
IMG_DIR="$ROOT_DIR/assets/images"

export PYTHONPATH="$ROOT_DIR/src:$ROOT_DIR:${PYTHONPATH:-}"

# PROMPT_LENS are now token counts (not character counts)
# Note: prompt_len + max_new_tokens must not exceed model's max_model_len
# For tinyllama with max_model_len=2048 and max_new_tokens=64, max prompt_len should be ~1984
# Using ranges that work with the model constraints
PROMPT_LENS=(512 1024 1536)
DTYPE="float16"
MAX_NEW_TOKENS=64
PREFILL_TOKENS=1
ITERS=5
WARMUP=2

echo "[run_all] model=$MODEL_PATH"
echo "[run_all] out_dir=$OUT_DIR img_dir=$IMG_DIR"

echo "[run_all] baseline bench..."
unset VLLM_CTA_ENABLE VLLM_CTA_STEPS VLLM_CTA_ALPHA VLLM_CTA_LAYOUT \
  VLLM_CTA_USE_TRITON VLLM_CTA_FUSED VLLM_CTA_ALLOW_TRANSPOSE \
  VLLM_CTA_LEN_THRESHOLD VLLM_CTA_STEPS_POLICY VLLM_CTA_MAX_STEPS \
  VLLM_CTA_USE_BUFFER_POOL VLLM_CTA_TRACE || true

python "$ROOT_DIR/serving/vllm/bench/bench_vllm.py" \
  --model "$MODEL_PATH" \
  --dtype "$DTYPE" \
  --prompt-lens "${PROMPT_LENS[@]}" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --prefill-tokens "$PREFILL_TOKENS" \
  --iters "$ITERS" --warmup "$WARMUP" \
  --out-dir "$OUT_DIR" \
  --out-name bench_vllm_baseline

echo "[run_all] cta bench..."
export VLLM_CTA_ENABLE=1
export VLLM_CTA_STEPS=4
export VLLM_CTA_ALPHA=0.10
export VLLM_CTA_LAYOUT=bld
export VLLM_CTA_USE_TRITON=1
export VLLM_CTA_FUSED=1
export VLLM_CTA_ALLOW_TRANSPOSE=1
export VLLM_CTA_LEN_THRESHOLD=0
export VLLM_CTA_STEPS_POLICY=fixed
export VLLM_CTA_MAX_STEPS=8
export VLLM_CTA_USE_BUFFER_POOL=1
export VLLM_CTA_TRACE=1

python "$ROOT_DIR/serving/vllm/bench/bench_vllm.py" \
  --model "$MODEL_PATH" \
  --dtype "$DTYPE" \
  --prompt-lens "${PROMPT_LENS[@]}" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --prefill-tokens "$PREFILL_TOKENS" \
  --iters "$ITERS" --warmup "$WARMUP" \
  --out-dir "$OUT_DIR" \
  --out-name bench_vllm_cta

echo "[run_all] compare + plots..."
python "$ROOT_DIR/serving/vllm/bench/compare_vllm_bench.py" \
  --baseline "$OUT_DIR/bench_vllm_baseline.csv" \
  --cta "$OUT_DIR/bench_vllm_cta.csv" \
  --out-dir "$OUT_DIR" \
  --out-name compare_vllm_bench

python "$ROOT_DIR/serving/vllm/bench/plot_vllm_bench.py" \
  --baseline "$OUT_DIR/bench_vllm_baseline.csv" \
  --cta "$OUT_DIR/bench_vllm_cta.csv" \
  --out-dir "$IMG_DIR" \
  --out-prefix vllm_cta

echo "[run_all] traces (prefill-like)..."
# Use larger prompt_len (token count) for profiling to see CTA impact more clearly
# But keep it within model's max_model_len (2048 for tinyllama) - max_new_tokens
TRACE_PROMPT_LEN=1536
unset VLLM_CTA_ENABLE || true
python "$ROOT_DIR/serving/vllm/bench/profile_vllm.py" \
  --model "$MODEL_PATH" \
  --dtype "$DTYPE" \
  --prompt-len "$TRACE_PROMPT_LEN" \
  --max-new-tokens 1 \
  --iters 3 --warmup 1 \
  --out-dir "$OUT_DIR" \
  --tag vllm_baseline_prefill_tinyllama

export VLLM_CTA_ENABLE=1
python "$ROOT_DIR/serving/vllm/bench/profile_vllm.py" \
  --model "$MODEL_PATH" \
  --dtype "$DTYPE" \
  --prompt-len "$TRACE_PROMPT_LEN" \
  --max-new-tokens 1 \
  --iters 3 --warmup 1 \
  --out-dir "$OUT_DIR" \
  --tag vllm_cta_prefill_tinyllama

echo "[run_all] done. key artifacts:"
ls -1 "$OUT_DIR"/bench_vllm_*.{csv,json} "$OUT_DIR"/compare_vllm_bench.{csv,md} \
  "$OUT_DIR"/trace_vllm_*_tinyllama.json "$OUT_DIR"/summary_vllm_*_tinyllama.txt \
  "$IMG_DIR"/vllm_cta_*.png

