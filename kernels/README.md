## kernels/

This directory contains **experimental** high-performance kernels for Continuous-Time Attention (CTA).

### Goal

CTA's PDE refinement can be viewed as a **local stencil token-mixing operator**. This is attractive for inference systems because:

- **\(O(L)\)** per PDE step in sequence length (local neighborhood access)
- Regular memory access pattern (stencil) → **kernel-friendly**
- Multi-step PDE refinement (`pde_steps > 1`) is a candidate for **fusion** to reduce HBM roundtrips in prefill

### Status

| Kernel | Status | Notes |
|---|---|---|
| Diffusion stencil (forward) | ✅ | Triton kernel with autotune in `diffusion_triton.py` (optional dependency) |
| Diffusion stencil (backward) | ✅ | Triton K3 backward kernel |
| Multi-step fused diffusion | ✅ | Fused steps {2,4,8} in `diffusion_triton.py` (time-tiling + halo) |

### Usage

This directory does **not** add Triton as a hard dependency for the repo. If you have Triton installed:

```bash
python kernels/diffusion_triton.py --device cuda --steps 4 --fused --bench
```

### Benchmark (fused vs non-fused)

```bash
python benchmarks/bench_triton_fused_diffusion.py --dtype bf16 --out_dir results/bench
python benchmarks/plot_triton_fused_diffusion.py --in_dir results/bench --out_dir assets/images
```

