# Kernel Fusion Notes: Time-Tiling for Diffusion

This note summarizes the fused multi-step diffusion kernel and why it matters.

## 1) Why fuse steps?
The non-fused path launches one kernel per step:
- `steps = 8` means 8 kernel launches
- each step reads/writes the full tensor (HBM roundtrips)

Fusion targets:
- fewer launches
- fewer global memory roundtrips
- better utilization for long prefill

## 2) Exact fusion with halo
For `steps = S`, the output at position `i` depends on `[i-S, ..., i+S]` from
the original input. The fused kernel loads a halo and performs `S` updates
inside registers:

```
tile = [i-S ... i+S]
for t in 1..S:
  update inner positions
```

This yields **exact** results (no approximation) as long as the halo is present.

## 3) Practical constraints
- `S in {2,4,8}` is a good sweet spot for register pressure.
- `BLOCK` should be smaller for fused kernels (more live values).
- bf16/fp16 should accumulate in fp32 to maintain stability.

## 4) Evidence to collect
- kernel launch count (fused should be 1 per call)
- total CUDA time vs non-fused
- effective bandwidth trend (GB/s)
- profiler trace highlighting the reduced kernel overhead

## 5) Common failure modes
- Missing halo or incorrect boundary mask causes drift.
- Updating halo positions incorrectly (must preserve boundary semantics).
- Too-large BLOCK leads to register spills and performance regressions.
