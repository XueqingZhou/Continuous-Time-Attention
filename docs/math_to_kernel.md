# From PDE to Kernel: Math-to-Systems Mapping

This note explains how a PDE-guided token-mixing operator becomes a GPU-friendly
stencil kernel, and what design choices matter for correctness and performance.

## 1) Continuous-time view
- Treat token interactions as a dynamical system: `du/dt = F(u)`.
- For diffusion: `du/dt = alpha * d^2u/dx^2` yields local smoothing.
- Locality is the key systems affordance: each update touches a small window.

## 2) Discretization into a stencil
For a 1D sequence:

```
y[i] = x[i] + alpha * (x[i+1] - 2*x[i] + x[i-1])
```

This is a 3-point stencil. After `steps` updates, the dependency radius grows
linearly: radius `= steps`.

## 3) Layout matters
- Choose `[B, D, L]` for Triton: `stride(L) = 1` makes coalesced reads.
- Choose `[B, L, D]` for model-level integration: avoids transposes in attention
  blocks (batch-first convention).
- The goal is to minimize layout conversions while keeping kernels fast.

## 4) Boundary policy
The reference uses **copy boundary**:
- `y[..., 0] = x[..., 0]`, `y[..., -1] = x[..., -1]`.
- This aligns with stable PDE behavior and simplifies correctness checks.

## 5) Golden reference + tests
Always build:
- a fp32 reference (stable across bf16/fp16)
- tests over `L in {1..5}+random`, `steps in {1,2,4,8}`, multiple `alpha`
- layout tests (`bdl` vs `bld`) to avoid hidden transpose bugs

## 6) Systems takeaway
The PDE view is not just math: it gives a **local, regular, fused** operator that
can be mapped to Triton and integrated into a serving stack with predictable
memory access and tunable step budgets.
