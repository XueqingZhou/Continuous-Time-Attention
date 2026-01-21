# vLLM Serving Notes: CTA Integration

This note captures the integration logic for CTA post-attn mixing in vLLM.

## 1) Integration point
Insert CTA **after self-attn** and before MLP:
- prefill: enabled
- decode: default disabled (to avoid latency regression)

## 2) Prefill-only behavior
CTA is controlled by a prefill gate:
- `prefill_only=True` keeps decode unaffected
- `min_seq_len` skips short prompts
- `steps_policy` maps prompt length to an operator budget

## 3) Layout and memory
vLLM uses `bld` layout. CTA supports:
- `bld` (no transpose, torch fallback)
- optional transpose to `bdl` for Triton fused kernels

If using transpose:
- enable `use_buffer_pool` to reuse scratch tensors
- avoid repeated allocations for each request

## 4) Telemetry hooks
Use `record_function("cta_post_attn_mixer")` so traces reveal:
- attn vs CTA vs layout
- kernel launch overhead vs fused path

## 5) Evidence checklist
- prefill latency vs prompt length (baseline vs CTA)
- decode tokens/s unchanged (or controlled)
- peak memory change and stability (p95)
