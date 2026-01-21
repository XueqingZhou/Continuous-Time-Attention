## vLLM CTA integration (prefill-only)

Goal: insert a **CTA post-attn mixer** after the attention output in vLLM, start
with prefill-only, then switch to Triton fused kernels.

### Requirements
- vLLM 0.13.0
- CUDA + Triton (optional, for fused kernel)
- Ensure this repo is importable from vLLM workers:
  - `export PYTHONPATH=/path/to/Continuous-Time-Attention/src:$PYTHONPATH`

### Patch-based integration
We provide a minimal patch in `patches/vllm_0_13_0_cta_post_attn.patch`:
1. inject a `CtaPostAttnMixer` into decoder layers
2. call CTA after self-attn (prefill-only)
3. expose basic knobs (steps/alpha/layout/use_triton/fused)

Apply (example):
```bash
cd /path/to/vllm
git apply /path/to/Continuous-Time-Attention/serving/vllm/patches/vllm_0_13_0_cta_post_attn.patch
```

### Mixer module
CTA mixer lives in:
- `src/cta/serving_vllm.py`

Features:
- **Torch fallback** with `layout=bld` (no transpose).
- **Triton fused** for `steps in {2,4,8}` (optional BLD<->BDL transpose).

### Runtime knobs (env)
- `VLLM_CTA_ENABLE=1`: enable CTA mixing (prefill-only)
- `VLLM_CTA_STEPS=4`, `VLLM_CTA_ALPHA=0.10`
- `VLLM_CTA_LAYOUT=bld`, `VLLM_CTA_USE_TRITON=1`, `VLLM_CTA_FUSED=1`
- `VLLM_CTA_LEN_THRESHOLD=0`: skip CTA for shorter prompts
- `VLLM_CTA_STEPS_POLICY=fixed|linear|log2`, `VLLM_CTA_MAX_STEPS=8`
- `VLLM_CTA_STEPS_SCALE=2048`: scale for `linear` policy
- `VLLM_CTA_USE_BUFFER_POOL=1`: reuse scratch buffers
- `VLLM_CTA_TRACE=1`: emit profiler range for CTA

### Validation
1. Run with prefill-only on a small model.
2. Use `serving/vllm/bench/bench_vllm.py` to capture baseline.
3. Enable CTA and compare prefill latency / tokens/s / peak mem.

Note: vLLM internal APIs can change across versions; if the patch does not
apply cleanly, follow the same logic and adjust the patch locally.
