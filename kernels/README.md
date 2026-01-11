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
| Diffusion stencil (forward) | 🟡 | Minimal Triton stub in `diffusion_triton.py` (optional dependency) |
| Diffusion stencil (backward) | 🔜 | Planned |
| Multi-step fused diffusion | 🔜 | Planned: fuse multiple steps to reduce global memory traffic |

### Usage

This directory does **not** add Triton as a hard dependency for the repo. If you have Triton installed:

```bash
python kernels/diffusion_triton.py --device cuda
```

