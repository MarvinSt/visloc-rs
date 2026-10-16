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
| Point elimination: the reduced system's blocks and right-hand side | CPU, as the CPU path | f64, uploaded in f32 |
| Preconditioner: dense inverses over runs of 32 frames, coarse level | CPU | f64 |
| Conjugate gradients | GPU | f32 vectors, fixed-order reductions |
| Point back-substitution, trial cost, step acceptance | CPU | f64 |

Each trial uploads the off-diagonal blocks (490 MB in f32 at SmallCity's
size), the diagonal blocks, the right-hand side and the preconditioner,
through one host-visible buffer kept between solves; the edge index goes up
once per bundle adjustment. The CPU path makes the same
f32 copy of the blocks for its own product, so the GPU path adds no
conversion.

## The product

The off-diagonal block of each edge `(a, b)` (frames `a < b`, edges sorted
by `a`) is stored once and acts on both of its block rows: `y_b += B x_a`
and `y_a += Bᵀ x_b`. The conjugate gradients' product reads each block
once, as the CPU's does, in two passes:

1. One workgroup per frame `a` reads the blocks of its edges, a contiguous
   run, with consecutive invocations on consecutive floats. It sums
   `Bᵀ x_b` into row `a` and writes `B x_a`, six floats, per edge.
2. One workgroup per frame `b` gathers those six floats from the edges
   ending at `b` (an index built once per bundle adjustment) and adds the
   diagonal block's product.

That is 144 bytes a block read once, plus 24 bytes an edge written and read
back. The previous product read each block twice, once per block row, and
gathered every row's blocks from scattered places.

## Deterministic sums

Every sum of the conjugate gradients runs in a fixed order: per row in the
product, then per workgroup and in one workgroup for the dot products. A
solve is therefore deterministic on a given device, but it is not
bit-identical to the CPU path.

## Point elimination on the GPU (opt-in)

`VISLOC_GPU_ELIMINATION=1` forms the off-diagonal blocks on the device
instead, from the camera-point blocks (uploaded once per linearization) and
the point inverses (9 floats a point per trial), and keeps them there for
the solve. The CPU then factors the preconditioner from the run blocks and
aggregate sums the device returns.

Each off-diagonal entry is a sum of terms from every point two frames share.
A workgroup takes a run of up to 160 edges of one frame's row and adds its
points' terms into 64-bit fixed-point counters in workgroup memory, carried
over two 32-bit atomics. Every term is first divided by the Cauchy-Schwarz
bound `sqrt(D_t[r][r] D_s[c][c])`, with `D` the per-frame elimination
diagonal the CPU already has, so the whole sum fits in `[-1, 1]`. A unit is
`2^-61`, so each f32 term is added exactly. Integer addition is associative,
so the blocks do not depend on the order the terms arrive in.

It is off by default because the CPU forms the blocks faster on the machine
measured below.

## Validation

- **Random block systems.** On Laplacian-like systems (35–66 iterations),
  the GPU solve takes the iteration count of an f64 reference with the same
  preconditioner, and its step is within 4e-4 of the reference's.
- **Elimination.** On a random 150-node / 4,000-point problem, the GPU
  blocks match f64 sums within 2.2e-8 of each entry's bound. A repeated
  elimination gives identical blocks.
- **SmallCity pass 1** (1,499 images; local llvmpipe replay, which checks
  correctness, not speed). The large bundle adjustments took about the
  CPU's iteration counts: 59, 116, 136, 66, ... against 61, 116, 135, 62,
  ..., and 1,359 against 1,309 over the final one's six trials. Both models
  register all 1,499 images, and the camera centres differ from the CPU
  model by a median of 4e-5 and at most 0.005% of the scene extent. The
  previous version (each block read twice, the blocks formed on the GPU)
  differed by a median of 5e-6 and at most 0.015%. With
  `VISLOC_GPU_ELIMINATION=1` the replay takes the same iteration counts,
  and its camera centres are within 5e-15 of the default's.
- **The CPU path** is unchanged: the replay without `--gpu-ba` writes the
  same model byte for byte as before this version.

## Speed on a GPU: the product is faster, the mapper is not

On an NVIDIA L4 (24 GB, 300 GB/s) with 15 CPU threads. SmallCity, all 5,822
photos, mapper seed 0, commit `e0ab6ee`, the same matches for both runs
(`gsplat_photos`' GPU SIFT on that machine):

| | CPU | `--gpu-ba 1` |
|---|---|---|
| Mapper | 846.7 s | 845.0 s |
| Peak memory | 18.9 GiB | 19.1 GiB |
| Large bundle adjustments: elimination | 80.2 s | 82.9 s |
| Large bundle adjustments: conjugate gradients, with the preconditioner's setup | 80.8 s (20,706 iterations) | 85.3 s (23,947 iterations) |
| PCG iteration at 3.4M edges | 6.8 ms | **3.4 ms** |
| Per-solve cost at 3.4M edges | 0.04 s | 0.59 s |
| Median camera-centre error against the release's model | 0.297 m | 0.522 m |

The last two pairs of figures come from a least-squares split of each large
bundle adjustment's solve time into a part per conjugate-gradient iteration
and a part per solve, both proportional to its edges.

- **The product is twice as fast as the CPU's.** The first version took
  14.9 ms an iteration against the CPU's 7.9 ms (table below). The new
  product moves about 720 MB an iteration at SmallCity's size, so 3.4 ms is
  about 210 GB/s, 70% of the L4's bandwidth.
- **Each solve's upload cost as much as the iterations saved.** It went
  through a staging buffer that `Queue::write_buffer` allocates per call,
  ~0.5 GB of host memory each time: 0.59 s at 3.4M edges, 34 s over the
  run. Uploads now go through one buffer kept between solves; that version
  has not been timed on a GPU yet.
- **The rest of the mapper stays on the CPU.** The large bundle
  adjustments' conjugate gradients are 81 s of the CPU run's 847 s, and
  their elimination another 80 s. Even an instant solve would save under a
  tenth.
- **Accuracy.** The runs part after the first bundle adjustment the GPU
  solves, as any last-bit difference makes the incremental mapper do. Both
  models show this capture's scale drift between its four passes: the
  passes' scales differ by up to 2.0% (CPU) and 2.9% (GPU), against 0.9% in
  the best CPU run (0.147 m). Mapper seeds alone spread the median error
  from 0.15 m to 0.65 m. Both GPU runs so far ended in the worse part of
  that range, 0.522 m here and 0.577 m below. Two runs do not show that the
  GPU path is less accurate.
- The logs' summary is in `benchmarks/smallcity_gpu/gpu_ba_l4.txt`.

### The first version

On an NVIDIA L4 with 21 CPU threads, the first version (the product that
read each block twice, the elimination on the GPU) was slower than the CPU
path. SmallCity, all 5,822 photos, mapper seed 0, commit `3dd9550`, the same
matches for both runs:

| | CPU | `--gpu-ba 1` |
|---|---|---|
| Mapper | **1,168 s** | 1,193 s |
| Peak memory | 18.8 GiB | **16.9 GiB** |
| Elimination, per LM trial (5,000+ frames) | **1.58 s** | 1.87 s |
| PCG, per iteration (5,000+ frames) | **7.9 ms** | 14.9 ms |

- **Products.** The GPU's matrix-vector product read every off-diagonal
  block twice, once per triangle: 980 MB per product at SmallCity's size,
  at about 65 GB/s, a fifth of the L4's bandwidth. The CPU reads each block
  once and applies it both ways.
- **Elimination.** It loses at every size from 1,000 frames up (0.52 s
  against 0.30 s a trial at 1,000 to 2,000 frames). The likely cause is the
  fixed-point atomics: consecutive frames share most of their points, so a
  workgroup's threads keep adding into the same few counters.
- **The two runs part** after the first bundle adjustment the GPU solves
  (1,005 frames: one 3D point fewer). They end at a median camera-centre
  error of 0.281 m (CPU) and 0.577 m (GPU) against the release's own model.
  The GPU run's final bundle adjustment rejected 7 of its 16 trials, the CPU
  run's none of 14.
