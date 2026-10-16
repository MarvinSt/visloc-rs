# Multi-view stereo (`visloc-mvs`)

`visloc-mvs` turns a COLMAP sparse model with undistorted (pinhole) images
into a dense point cloud: a depth and normal map per image by PatchMatch,
fused into `fused.ply`. It runs on the CPU, or on the GPU through wgpu with
`--features gpu`.

```text
cargo run --release -p visloc-mvs --features gpu --example mvs -- \
    --sparse <dir with cameras.txt, images.txt, points3D.txt> \
    --images <image dir> --out <dir> --gpu \
    [--max-image-size 3200] [--sources 10] [--iterations 3] \
    [--geom-passes 2] [--scales 1] [--view-selection confidence|sampled]
```

The image names in `images.txt` are relative to `--images`. The cameras must
be `PINHOLE` or `SIMPLE_PINHOLE`: undistort the images first.

## Method

Depth estimation follows ACMH and ACMM (Q. Xu and W. Tao, "Multi-Scale
Geometric Consistency Guided Multi-View Stereo", CVPR 2019), implemented
from the paper:

- **Hypotheses.** Each pixel holds a plane, a unit normal facing the camera
  and its offset, so a hypothesis spreads to a neighbour as a plane, not as
  a depth.
- **Propagation.** Red-black checkerboard updates. Each pixel takes the
  lowest-cost plane of eight regions around it: four V shapes of 7 pixels
  and four strips of 11 pixels along the axes, all of the other colour.
- **View selection.** The eight planes' costs in every source view vote. A
  view counts when more than `n1 = 2` of them are below
  `0.8 exp(-t² / 90)` (`t` the iteration) and fewer than `n2 = 3` above
  1.2. Its weight is the mean of `exp(-m² / (2 · 0.3²))` over its good
  costs. The previous iteration's most important view keeps a double
  weight. `--view-selection sampled` follows ACMM's released code
  instead: those weights, times a prior from the four adjacent pixels'
  selected views, are a distribution that 15 draws sample, and a view's
  weight is its count. Views without weight are not matched.
- **Refinement.** A random and a perturbed depth and normal, combined with
  the current ones into six more planes.
- **Matching cost.** One minus a bilateral-weighted NCC over every other row
  and column of an 11×11 window, warped by the plane's homography. The
  window skips its centre, so where every sample's grey level is far from
  the centre's, all the weights vanish and the pixel does not match.
- **Geometric consistency** (`--geom-passes`). Further passes start from the
  previous depth maps and add 0.2 times each source view's
  forward-backward reprojection error, truncated at 3 pixels.
- **Multi-scale** (`--scales 3`). ACMM's pyramid: estimates at a coarser
  scale are upsampled (joint bilateral, the best-weighted neighbour's plane)
  and kept where a photometric pass at the finer scale does not beat them
  by more than 0.1 (the detail restorer), then refined by geometric passes.
- **Source views and depth ranges.** Up to 10 source views per image by
  MVSNet's score over the shared sparse points (a Gaussian in the
  triangulation angle around 5°). The depth range comes from the image's
  sparse points: COLMAP's 1% and 99% quantiles, widened by 25%.
- **Fusion.** A pixel's 3D point is kept when a source view agrees within
  2 pixels of reprojection error, 1% of relative depth and 30° of normal
  angle; the agreeing pixels are averaged into one point and not used
  again. ACMM's code asks for two agreeing views (and 10°): with these
  depth maps one view scores much higher (below), `--min-consistent 2`
  restores two.

The GPU path runs the same per-pixel update as compute shaders, with the
same sampling regions, view selection and random numbers. The source images
are half-float texture layers that the texture unit samples bilinearly, as
ACMM samples CUDA textures. The GPU differs from the CPU path only by
rounding: on ETH3D pipes at 400 px its F1 score is 34.44% against the
CPU's 34.47%.

## ETH3D high-resolution benchmark

Scores come from the official ETH3D multi-view evaluator
(`ETH3DMultiViewEvaluation`) against the laser scans of the training
scenes, at the benchmark's 2 cm tolerance, with the scenes' ground-truth
camera poses (`dslr_calibration_undistorted`).

### Full resolution on a GPU

One run on an NVIDIA L4 pod at commit `cf329e2`, before the NaN fix, the
skipped unweighted views and the texture sampling. Settings: images at
3200 px, 10 source views, three iterations per pass, two geometric passes,
and ACMM's fusion (2 px, 1% depth, 30°, two agreeing source views). The
times cover the whole `mvs` run: loading, depth maps, fusion and writing
the cloud.

| Scene | Images | F1 | Accuracy | Completeness | Time | F1, multi-scale | Time, multi-scale |
|---|---|---|---|---|---|---|---|
| pipes | 14 | 70.6% | 95.8% | 55.9% | 2:45 | 71.9% | 4:28 |
| meadow | 15 | 57.4% | 92.2% | 41.7% | 5:08 | 62.7% | 7:12 |
| terrace | 23 | 87.8% | 94.8% | 81.7% | 6:02 | 88.3% | 9:03 |
| office | 26 | 69.7% | 86.6% | 58.4% | 6:35 | | |
| kicker | 31 | 72.3% | 87.8% | 61.5% | 9:20 | | |
| relief | 31 | 86.9% | 93.8% | 80.9% | 9:39 | | |
| relief 2 | 31 | 85.0% | 91.6% | 79.3% | 8:59 | | |
| courtyard | 38 | 86.0% | 90.3% | 82.1% | 8:12 | | |
| playground | 38 | 65.7% | 79.3% | 56.0% | 11:21 | | |

- **Mean over these nine scenes** (single-scale): F1 75.7%, accuracy
  90.3%, completeness 66.4%. Over all 13 training scenes at 2 cm, ACMM
  reports F1 78.86% (accuracy 90.67%, completeness 70.42%) and COLMAP
  67.66% (91.85%, 55.13%) (arXiv 2303.09758, 2210.07582). The scene sets
  differ, so those means only bracket ours.
- **Multi-scale** (`--scales 3 --geom-passes 2`) raised F1 on all three
  scenes it ran on, by 1.3, 5.3 and 0.5 points, mostly through
  completeness.
- **Not yet run:** terrains, delivery area, electro and facade.
- **What falls short is completeness.** Accuracy is at ACMM's level;
  completeness is lower, most on meadow and playground.

### Fusion settings

The same depth maps fused with different rules (CPU runs with one
geometric pass; F1 at 2 cm). One agreeing source view instead of ACMM's
two adds far more completeness than it costs accuracy:

| Run | 30°, two views | any angle, two views | 30°, one view | any angle, one view |
|---|---|---|---|---|
| pipes, 800 px | 53.8% | 56.4% | 67.6% | 68.3% |
| meadow, 800 px | 51.3% | 52.4% | 56.4% | 57.1% |
| pipes, 1600 px | 66.9% | 68.3% | **76.0%** | 75.6% |

On pipes at 1600 px, one view takes completeness from 51.1% to 65.4% and
accuracy from 96.9% to 90.7%. On pipes at 800 px, a stricter depth
(0.5%) or reprojection (1 px) check, or dropping pixels by matching cost,
only lowered F1. The full-resolution runs above predate this default and
use two views.

### Local runs on pipes (CPU)

| Images | Passes | F1 | Accuracy | Completeness |
|---|---|---|---|---|
| 800 px | photometric | 45.8% | 96.7% | 30.0% |
| 800 px | + 1 geometric | 53.7% | 95.3% | 37.4% |
| 1600 px | + 1 geometric | 67.0% | 97.0% | 51% |

Resolution matters most: ACMM and COLMAP are evaluated at up to 3200 px.
