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
    [--geom-passes 2] [--scales 1]
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
  weight.
- **Refinement.** A random and a perturbed depth and normal, combined with
  the current ones into six more planes.
- **Matching cost.** One minus a bilateral-weighted NCC over every other row
  and column of an 11×11 window, warped by the plane's homography.
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
- **Fusion.** ACMM's: a pixel's 3D point is kept when at least two source
  views agree within 2 pixels of reprojection error, 1% of relative depth
  and 30° of normal angle; the agreeing pixels are averaged into one point
  and not used again.

The GPU path runs the same per-pixel update as compute shaders, with the
same sampling regions, view selection and random numbers. It differs from
the CPU path only by floating-point rounding: on ETH3D pipes at 800 px its
F1 score is 53.73% against the CPU's 53.67%.

## ETH3D high-resolution benchmark

Scores come from the official ETH3D multi-view evaluator
(`ETH3DMultiViewEvaluation`) against the laser scans of the training
scenes, at the benchmark's 2 cm tolerance, with the scenes' ground-truth
camera poses (`dslr_calibration_undistorted`).

### Local runs on pipes (CPU)

| Images | Passes | F1 | Accuracy | Completeness |
|---|---|---|---|---|
| 800 px | photometric | 45.8% | 96.7% | 30.0% |
| 800 px | + 1 geometric | 53.7% | 95.3% | 37.4% |
| 1600 px | + 1 geometric | 67.0% | 97.0% | 51% |

Resolution matters most: ACMM and COLMAP are evaluated at up to 3200 px.
