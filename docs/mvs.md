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
    [--geom-passes 2] [--scales 1] [--no-planar-prior] \
    [--view-selection confidence|sampled]
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
- **Planar prior** (ACMP: Q. Xu and W. Tao, "Planar Prior Assisted
  PatchMatch Multi-View Stereo", AAAI 2020; with one scale by default,
  `--no-planar-prior` leaves it out). After the photometric pass, each
  view's credible pixels (the lowest-cost pixel of each 5×5 block, if its
  cost is below 0.1) are triangulated in the image (Delaunay). The three
  back-projected corners of each triangle span the plane that is the
  prior of the pixels inside it. A planar prior pass then starts from the
  photometric map, or from the prior where the cost is not credible, and
  ranks a pixel's hypotheses by
  `exp(-c²/0.18) · (0.5 + exp(-Δd²/(2σd²)) · exp(-Δθ²/(2σθ²)))`: `c` the
  cost, `Δd` and `Δθ` the depth and normal angle to the prior, `σd` 1/64 of
  the depth range, `σθ` 5°. Where weak texture leaves the costs unable to
  tell planes apart, the hypothesis closest to the prior wins.
- **Geometric consistency** (`--geom-passes`, 2 by default). Further
  passes start from the previous depth maps and add 0.2 times each source
  view's forward-backward reprojection error, truncated at 3 pixels.
- **Multi-scale** (`--scales 3`). ACMM's pyramid: estimates at a coarser
  scale are upsampled (joint bilateral, the best-weighted neighbour's plane)
  and kept where a photometric pass at the finer scale does not beat them
  by more than 0.1 (the detail restorer), then refined by geometric passes.
  On ETH3D it helps weakly textured scenes and costs accuracy on detailed
  ones (below); the example runs one scale by default, and multi-scale
  runs without the planar prior.
- **Source views and depth ranges.** Up to 10 source views per image by
  MVSNet's score over the shared sparse points (a Gaussian in the
  triangulation angle around 5°). The depth range comes from the image's
  sparse points: COLMAP's 1% and 99% quantiles, widened by 25%.
- **Fusion.** A pixel's 3D point is kept when a source view agrees within
  2 pixels of reprojection error, 1% of relative depth and 10° of normal
  angle; the agreeing pixels are averaged into one point and not used
  again. ACMM's and ACMP's code ask for two agreeing views: with these
  depth maps one view scores higher (below), and `--min-consistent 2`
  restores two.

The GPU path runs the same per-pixel update as compute shaders, with the
same sampling regions, view selection and random numbers. The source images
are half-float texture layers that the texture unit samples bilinearly, as
ACMM samples CUDA textures. The GPU differs from the CPU path only by
rounding: on ETH3D pipes at 400 px its F1 score is 34.44% against the
CPU's 34.47%.

**Memory.** A view keeps its colours, 3 bytes a pixel; its grey levels
are the colours' luma, computed when the view is matched. A depth map
keeps a depth and a normal, 16 bytes a pixel, and its costs only with
`--save-maps` (no pass or fusion reads them). A geometric pass releases
the previous pass's maps as it goes: a view's normals once it is
re-estimated, its depths after the last view that uses it as a source.
Fusion drops each map after the last view that reads it. So one set of
maps is held at a time, not two, and the output does not change: on
pipes, `fused.ply` is byte-identical on the CPU and the GPU, with one
scale and with three. With the images pre-resized to 800 px (one
iteration per pass, CPU), peak memory falls from 305 to 218 MiB with one
scale and from 364 to 229 MiB with three. At that size glibc keeps some
freed memory: 175 MiB with one scale under
`MALLOC_MMAP_THRESHOLD_=131072`.

On ETH3D facade (76 images at 3200 px, an NVIDIA L4) peak memory falls
from 24.2 to 12.0 GiB with one scale (in 10:16 instead of 11:40) and from
28.7 to 12.4 GiB with three. The F1 scores at 2 cm, 64.96% and 62.03%,
are within 0.03 points of the earlier runs' (64.96% and 62.06%). The
shaders did not change, but that pod had another NVIDIA driver (580.178
instead of 595.91), and the clouds differ slightly: 77.75 million fused
points instead of 77.66 million with one scale.

## ETH3D high-resolution benchmark

Scores come from the official ETH3D multi-view evaluator
(`ETH3DMultiViewEvaluation`) against the laser scans of the training
scenes, at the benchmark's 2 cm tolerance, with the scenes' ground-truth
camera poses (`dslr_calibration_undistorted`).

### All 13 training scenes on a GPU

One run on an NVIDIA L4 pod at commit `21b6596` with what are now the
example's defaults: images at 3200 px, 10 source views, three iterations
per pass, one scale, the photometric pass, the planar prior pass and two
geometric passes. The depth maps were fused four ways. F1, accuracy and
completeness are for the default fusion (one agreeing source view within
10°); the other columns are the F1 of the run before the planar prior
(commit `3f05992`, the same fusion) and of the same maps fused with 15°
and with two agreeing views at 30°. Times cover the whole `mvs` run:
loading, depth maps, a fusion and writing the cloud and the maps.

| Scene | Images | F1 | Accuracy | Completeness | F1 without the prior | F1, 15° | F1, two views | Time |
|---|---|---|---|---|---|---|---|---|
| pipes | 14 | 81.3% | 93.8% | 71.8% | 78.0% | 81.2% | 73.7% | 1:54 |
| meadow | 15 | 64.2% | 90.7% | 49.7% | 60.4% | 64.9% | 60.3% | 2:49 |
| terrace | 23 | 90.0% | 95.4% | 85.3% | 88.8% | 89.9% | 89.3% | 3:48 |
| office | 26 | 78.5% | 79.1% | 77.9% | 71.3% | 77.6% | 76.9% | 3:59 |
| kicker | 31 | 77.6% | 77.9% | 77.3% | 75.4% | 76.8% | 75.6% | 5:33 |
| relief | 31 | 87.9% | 92.4% | 83.9% | 87.7% | 87.5% | 87.1% | 5:30 |
| relief 2 | 31 | 87.0% | 90.4% | 83.9% | 86.9% | 86.7% | 85.0% | 5:12 |
| courtyard | 38 | 89.2% | 89.9% | 88.4% | 88.5% | 89.3% | 86.6% | 6:18 |
| playground | 38 | 67.6% | 78.8% | 59.3% | 67.4% | 67.7% | 65.9% | 6:36 |
| terrains | 42 | 91.4% | 91.4% | 91.4% | 89.7% | 91.2% | 90.8% | 7:23 |
| delivery area | 44 | 89.7% | 88.8% | 90.6% | 87.8% | 89.3% | 87.5% | 6:48 |
| electro | 45 | 88.8% | 95.9% | 82.6% | 86.7% | 88.8% | 87.5% | 7:14 |
| facade | 76 | 65.5% | 76.3% | 57.3% | 64.9% | 65.5% | 64.0% | 14:17 |

- **Mean over the 13 scenes:** F1 81.4%, accuracy 87.7%, completeness
  76.9%. Over the same scenes at 2 cm, ACMMP reports F1 83.42%, ACMP
  79.79%, ACMM 78.86% (accuracy 90.67%, completeness 70.42%) and COLMAP
  67.66% (91.85%, 55.13%) (arXiv 2303.09758, 2210.07582); later methods
  reach 84 to 89% (arXiv 2503.13721).
- **The planar prior raises F1 on all 13 scenes,** from 79.5% to 81.4%
  on average. It adds the most in weak texture, on office (+7.2 points),
  meadow (+3.8) and pipes (+3.4), and 0.1 to 0.3 points on the detailed
  relief scenes and playground. Completeness rises from 73.5% to 76.9% at
  about the same accuracy (88.0%, 87.7%).
- **The angle:** 10° and 15° are close (81.3% at 15°); 10° is higher on
  eight scenes and lower on three, and it is ACMM's and ACMP's angle, so
  it is now the default. 30° scores 80.6%. Two agreeing views at 30°
  score 79.3% (accuracy 89.0%, completeness 72.5%).
- **Time:** 77 minutes for all 13 scenes (65 without the planar prior),
  from 1:54 (14 images) to 14:17 (76 images). On facade the photometric
  pass takes 2:56, the planar prior pass 3:19, the geometric passes 2:35
  each and the fusion on the CPU 2:17. Peak memory 13.3 GiB (facade;
  `--save-maps` keeps the costs).

### Without the planar prior

The run before the planar prior (commit `3f05992`, otherwise the same,
fused with 15°) scored F1 79.4% (accuracy 85.9%, completeness 74.8%) in
65 minutes; 77.3% with two views at 30°. Its peak memory was 24.2 GiB
on facade, 12.0 GiB since the maps are released as they go (Memory,
above). Against the first full-resolution run (commit `cf329e2`, before
the NaN fix, the skipped unweighted views and the texture sampling), the
two-view F1 agrees within 0.1 point on each of the 12 scenes that run
covered, which took 53 minutes instead of 99.

### Multi-scale

The same scenes with ACMM's three scales (`--scales 3`), two geometric
passes at each and no planar prior, at commit `33b7418`, whose PatchMatch
code is that of `3f05992`. F1, accuracy and completeness are for one
agreeing view within 15°; the last F1 column is the single-scale run
without the prior, with the same fusion.

| Scene | Images | F1 | Accuracy | Completeness | F1, one scale | Time |
|---|---|---|---|---|---|---|
| pipes | 14 | 79.7% | 91.2% | 70.7% | 78.0% | 1:59 |
| meadow | 15 | 68.4% | 81.9% | 58.8% | 61.5% | 2:39 |
| terrace | 23 | 88.7% | 92.7% | 84.9% | 88.7% | 3:50 |
| office | 26 | 65.7% | 64.5% | 67.0% | 70.4% | 4:01 |
| kicker | 31 | 80.0% | 82.3% | 77.8% | 75.1% | 5:29 |
| relief | 31 | 83.9% | 87.8% | 80.3% | 87.5% | 5:21 |
| relief 2 | 31 | 81.9% | 84.7% | 79.2% | 86.7% | 5:24 |
| courtyard | 38 | 86.4% | 87.2% | 85.6% | 88.7% | 6:01 |
| playground | 38 | 66.0% | 73.4% | 59.9% | 67.5% | 6:32 |
| terrains | 42 | 84.2% | 83.8% | 84.5% | 89.4% | 7:19 |
| delivery area | 44 | 81.3% | 79.3% | 83.3% | 87.5% | 7:00 |
| electro | 45 | 86.4% | 93.4% | 80.5% | 86.8% | 7:10 |
| facade | 76 | 62.1% | 71.0% | 55.1% | 65.0% | 13:19 |

- **Mean over the 13 scenes:** F1 78.0%, accuracy 82.6%, completeness
  74.4%, against 79.4%, 85.9% and 74.8% with one scale.
- **It helps weakly textured scenes and costs accuracy on detailed
  ones:** +7.0 points on meadow, +4.9 on kicker and +1.6 on pipes, but
  −6.3 on delivery area, −5.2 on terrains, −4.8 on relief 2 and −4.7 on
  office. The better of the two runs per scene would average 80.5%.
- **Fusion:** the variants rank as with one scale: 77.3% with 30°, 76.4%
  without the normal check, 76.0% with two views at 30°.
- **Time:** 76 minutes, 1.2 times the single-scale run. Peak memory
  28.7 GiB (facade), 12.4 GiB since the maps are released as they go.

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
only lowered F1. At 3200 px (above) a tighter normal check pays: 10° and
15° beat 30°, and on the multi-scale maps 30° beats no check.

### Local runs on pipes (CPU)

Before the NaN fix, with two-view fusion:

| Images | Passes | F1 | Accuracy | Completeness |
|---|---|---|---|---|
| 800 px | photometric | 45.8% | 96.7% | 30.0% |
| 800 px | + 1 geometric | 53.7% | 95.3% | 37.4% |
| 1600 px | + 1 geometric | 67.0% | 97.0% | 51% |

Resolution matters most: ACMM and COLMAP are evaluated at up to 3200 px.
