# GPU bundle adjustment for the COLMAP-port mapper

`colmap_incremental_mapper --gpu-ba 1` (a `--features gpu` build) solves the
reduced camera system of large bundle adjustments on the GPU through wgpu
(`visloc_ba_gpu::GpuReducedSolver`). Without the flag nothing changes: the
CPU path stays bit-identical.

## What runs where

The GPU takes the bundle adjustments with at least 1,000 free frames (the
global ones on large scenes; `VISLOC_PORT_ITERATIVE_MIN_FRAMES`). Smaller
systems keep the CPU's block Cholesky. In one Levenberg-Marquardt trial:

| Step | Where | Precision |
|---|---|---|
| Linearization (residuals, Jacobians, `J_cᵀJ_p` blocks) | CPU | f64 |
| Point inverses, diagonal blocks and right-hand side of the reduced system | CPU, the CPU path's sums in the same order | f64 |
| Off-diagonal blocks (`-Σ W_tp V_p⁻¹ W_spᵀ` over shared points) | GPU | f32 terms, exact sums |
| Preconditioner: dense inverses over runs of 32 frames, coarse level | CPU, from the blocks the GPU returns | f64 |
| Conjugate gradients | GPU | f32 vectors, fixed-order reductions |
| Point back-substitution, trial cost, step acceptance | CPU | f64 |

The camera-point blocks are uploaded once per linearization and the
elimination's index once per bundle adjustment; per trial only the point
inverses (9 floats a point) and the bounds below go up.

## Exact, deterministic sums

Each off-diagonal entry is a sum of terms from every point two frames share.
A workgroup takes a run of up to 160 edges of one frame's row and adds its
points' terms into 64-bit fixed-point counters in workgroup memory, carried
over two 32-bit atomics. Every term is first divided by the Cauchy-Schwarz
bound `sqrt(D_t[r][r] D_s[c][c])`, with `D` the per-frame elimination
diagonal the CPU already has, so the whole sum fits in `[-1, 1]`. A unit is
`2^-61`, so each f32 term is added exactly. Integer addition is associative,
so the blocks do not depend on the order the terms arrive in. The reductions
of the conjugate gradients sum in a fixed order too. A solve is therefore
deterministic on a given device, but it is not bit-identical to the CPU path.

## Validation

- **Random block systems.** On Laplacian-like systems (35–66 iterations),
  the GPU solve takes the iteration count of an f64 reference with the same
  preconditioner, and its step is within 4e-4 of the reference's.
- **Elimination.** On a random 150-node / 4,000-point problem, the GPU
  blocks match f64 sums within 2.2e-8 of each entry's bound. A repeated
  elimination gives identical blocks.
- **SmallCity pass 1** (1,499 images; local llvmpipe replay, which checks
  correctness, not speed). The large bundle adjustments took the CPU's
  iteration counts (61, 116, 136 vs 61, 116, 135, ...). Camera centres
  differ from the CPU model by a median of 5e-6 and at most 0.015% of the
  scene extent. Forming the blocks on the GPU instead of the CPU changed the
  final camera centres by at most 5e-15.

Speed on a real GPU: pending a pod benchmark.
