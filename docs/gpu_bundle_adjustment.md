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

## Speed on a GPU: not faster yet

On an NVIDIA L4 (24 GB, 300 GB/s) with 21 CPU threads, the GPU path is
slower than the CPU's. SmallCity, all 5,822 photos, mapper seed 0, commit
`3dd9550`, the same matches for both runs:

| | CPU | `--gpu-ba 1` |
|---|---|---|
| Mapper | **1,168 s** | 1,193 s |
| Peak memory | 18.8 GiB | **16.9 GiB** |
| Elimination, per LM trial (5,000+ frames) | **1.58 s** | 1.87 s |
| PCG, per iteration (5,000+ frames) | **7.9 ms** | 14.9 ms |

- **Products.** The GPU's matrix-vector product reads every off-diagonal
  block twice, once per triangle: 980 MB per product at SmallCity's size,
  at about 65 GB/s, a fifth of the L4's bandwidth. The CPU reads each block
  once and applies it both ways.
- **Elimination.** It loses at every size from 1,000 frames up (0.52 s
  against 0.30 s a trial at 1,000 to 2,000 frames). The likely cause is the
  fixed-point atomics: consecutive frames share most of their points, so a
  workgroup's threads keep adding into the same few counters.
- **The two runs part** after the first bundle adjustment the GPU solves
  (1,005 frames: one 3D point fewer), as any last-bit difference makes the
  incremental mapper do. They end at a median camera-centre error of
  0.281 m (CPU) and 0.577 m (GPU) against the release's own model. The GPU
  run's final bundle adjustment rejected 7 of its 16 trials, the CPU run's
  none of 14. One run each, so this does not show that the GPU path is
  less accurate; it does not show the opposite either.

What it would take to win:

- a block product that reads each block once, in contiguous runs per
  workgroup, writing the transposed half per edge and gathering it in a
  second pass;
- the elimination back on the CPU, which already forms the blocks faster;
  the off-diagonal blocks then go up once per trial (490 MB in f32);
- a GPU with more bandwidth: the CPU here is 21 threads of a large server
  part.
