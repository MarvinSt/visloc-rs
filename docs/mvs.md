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
    [--geom-passes 2] [--scales 3] [--no-planar-prior] \
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
  PatchMatch Multi-View Stereo", AAAI 2020; at every scale by default,
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
  tell planes apart, the hypothesis closest to the prior wins. Where a
  pixel has a prior, the refinement draws its random depth within 3σd of
  the prior's and its random normal around the prior's, as ACMP's code
  does.
- **Geometric consistency** (`--geom-passes`, 2 by default). Further
  passes start from the previous depth maps and add 0.2 times each source
  view's forward-backward reprojection error, truncated at 3 pixels.
- **Multi-scale** (three scales by default, `--scales 1` runs one), in the
  order of ACMMP's released code (Q. Xu, W. Kong, W. Tao and M. Pollefeys,
  "Multi-Scale Geometric Consistency Guided and Planar Prior Assisted
  Multi-View Stereo", TPAMI 2022): at every scale of the image pyramid a
  photometric pass, the planar prior pass and the geometric passes. A
  finer scale's photometric pass starts from the coarser maps, upsampled
  (joint bilateral: each pixel takes the plane of the best-weighted coarse
  neighbour). Its detail restorer lets a pixel leave its upsampled plane
  only for one whose cost is below that of the plane at infinity (the
  rotation's homography) by 0.1. Where weak texture leaves every plane
  about as costly, the coarser estimate stays; elsewhere the pass
  re-estimates at the finer scale. That is what ACMM's and ACMMP's code
  compute: their baseline plane is the upsampled depth with no normal.
  ACMM's paper compares with the upsampled plane's own cost instead
  (`--restorer start`), which holds planes that are slightly off: on a
  synthetic plane a start 2% too deep stays 2% too deep, where the plane
  at infinity lets it converge to within 0.05%. Our first multi-scale
  scheme, an independent photometric pass at each finer scale kept where
  it beat the upsampled plane by 0.1, lost accuracy on detailed scenes
  for that reason (below).
- **Source views and depth ranges.** Up to 10 source views per image by
  MVSNet's score over the shared sparse points (a Gaussian in the
  triangulation angle around 5°). The depth range comes from the image's
  sparse points: COLMAP's 1% and 99% quantiles, widened by 25%.
- **Fusion.** A pixel's 3D point is kept when a source view agrees within
  2 pixels of reprojection error, 1% of relative depth and 10° of normal
  angle; the agreeing pixels are averaged into one point and not used
  again. ACMM's and ACMP's code ask for two agreeing views: with these
  depth maps one view scores higher (below), and `--min-consistent 2`
  restores two. ACMMP's code also asks the agreeing views to agree well on
  average (`--min-mean-consistency 0.3`) and keeps the reference pixel's
  point instead of the average (`--no-average`). On pipes and meadow
  pre-resized to 800 px the first costs about 10 points of F1 and the
  second changes it by 0.1 point, so both are off by default.

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

One run on an NVIDIA L4 pod at commit `2c9cebe` with what are now the
example's defaults: images at 3200 px, 10 source views, three iterations
per pass, three scales (800, 1600 and 3200 px) in ACMMP's order, each
with a photometric pass, the planar prior pass and two geometric passes.
The same run estimated every scene with one scale too (`--scales 1`, the
previous default). F1, accuracy and completeness are for three scales and
the default fusion (one agreeing source view within 10°); the other F1
columns are for one scale (the same fusion) and for three scales fused
within 15°. Times cover the whole `mvs` run: loading, depth maps, a
fusion and writing the cloud and the maps. (Facade ran on a second pod,
the same GPU type and driver, after its upload failed.)

| Scene | Images | F1 | Accuracy | Completeness | F1, one scale | F1, 15° | Time | Time, one scale |
|---|---|---|---|---|---|---|---|---|
| pipes | 14 | 84.4% | 93.9% | 76.7% | 81.2% | 84.1% | 1:53 | 1:40 |
| meadow | 15 | 76.0% | 84.9% | 68.8% | 63.9% | 76.2% | 2:34 | 2:27 |
| terrace | 23 | 91.0% | 94.8% | 87.5% | 90.0% | 90.7% | 3:43 | 3:12 |
| office | 26 | 79.0% | 76.6% | 81.7% | 78.6% | 77.8% | 3:47 | 3:19 |
| kicker | 31 | 82.3% | 78.9% | 85.9% | 77.6% | 81.3% | 4:54 | 4:24 |
| relief | 31 | 87.8% | 91.6% | 84.3% | 87.9% | 87.4% | 5:20 | 4:31 |
| relief 2 | 31 | 86.5% | 89.3% | 83.9% | 86.9% | 86.1% | 5:24 | 4:26 |
| courtyard | 38 | 90.4% | 90.0% | 90.7% | 89.1% | 90.4% | 6:05 | 5:07 |
| playground | 38 | 69.8% | 77.2% | 63.7% | 67.4% | 69.5% | 6:28 | 5:45 |
| terrains | 42 | 91.2% | 90.9% | 91.5% | 91.3% | 91.0% | 6:54 | 6:09 |
| delivery area | 44 | 90.1% | 89.0% | 91.1% | 89.6% | 89.8% | 6:44 | 5:40 |
| electro | 45 | 89.4% | 94.5% | 84.9% | 88.7% | 89.2% | 6:55 | 6:03 |
| facade | 76 | 67.5% | 74.4% | 61.7% | 65.4% | 67.3% | 13:46 | 11:53 |
MEAN s avg      : 13 scenes, F1 81.36, accuracy 87.64, completeness 76.84  (r36 on these: 81.44)
MEAN s one15    : 13 scenes, F1 81.16, accuracy 85.94, completeness 77.81  (r36 on these: 81.44)
MEAN m avg      : 13 scenes, F1 83.49, accuracy 86.62, completeness 80.95  (r36 on these: 81.44)
MEAN m one15    : 13 scenes, F1 83.13, accuracy 85.04, completeness 81.70  (r36 on these: 81.44)

- **Mean over the 13 scenes:** F1 83.5%, accuracy 86.6%, completeness
  81.0%, against 81.4% (87.6%, 76.8%) with one scale. Over the same
  scenes at 2 cm, ACMMP reports F1 83.42%, HPM-MVS 84.58%, ACMP 79.79%,
  ACMM 78.86% (accuracy 90.67%, completeness 70.42%) and COLMAP 67.66%
  (91.85%, 55.13%) (arXiv 2303.09758, 2210.07582); later methods reach
  84 to 89% (arXiv 2503.13721).
- **Three scales against one:** +2.1 points on average, most in weak
  texture: meadow +12.1, kicker +4.7, pipes +3.2, playground +2.4 and
  facade +2.1. The detailed scenes stay within half a point (relief
  −0.1, relief 2 −0.4, terrains −0.1), where the earlier multi-scale
  scheme lost 3.6 to 6.3 points (below). Completeness rises by 4.1 points
  for 1.0 point of accuracy.
- **The planar prior's draws** around the prior change little: with one
  scale, 81.36% against 81.44% in the previous run (commit `21b6596`,
  below), within 0.3 point on every scene.
- **Fusion:** 10° beats 15° with three scales too (83.1% at 15°).
- **Time:** 74 minutes for all 13 scenes with three scales, 65 with one,
  from 1:53 (14 images) to 13:46 (76 images). On facade the finest scale
  takes 2:27 for the photometric pass, 2:27 for the planar prior pass and
  2:08 and 2:05 for the geometric passes; the two coarser scales take
  2:26 together. Peak memory 14.6 GiB on facade (13.7 GiB with one scale;
  `--save-maps` keeps the costs).

### One scale and the planar prior

One run at commit `21b6596`: one scale, the photometric pass, the planar
prior pass (its random draws not yet around the prior) and two geometric
passes. F1, accuracy and completeness are for the default fusion; the
other columns are the F1 of the run before the planar prior (commit
`3f05992`, the same fusion) and of the same maps fused with 15° and with
two agreeing views at 30°.

- **Mean over the 13 scenes:** F1 81.4%, accuracy 87.7%, completeness
  76.9%.
- **The planar prior raises F1 on all 13 scenes,** from 79.5% to 81.4%
  on average. It adds the most in weak texture, on office (+7.2 points),
  meadow (+3.8) and pipes (+3.4), and 0.1 to 0.3 points on the detailed
  relief scenes and playground. Completeness rises from 73.5% to 76.9% at
  about the same accuracy (88.0%, 87.7%).
- **The angle:** 10° and 15° are close (81.3% at 15°); 10° is higher on
  eight scenes and lower on three, and it is ACMM's and ACMP's angle, so
  it is the default. 30° scores 80.6%. Two agreeing views at 30°
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

### Multi-scale with the paper's detail restorer

The same scenes with our first multi-scale scheme: three scales, two
geometric passes at each and no planar prior, at commit `33b7418`, whose
PatchMatch code is that of `3f05992`. A finer scale ran an independent
photometric pass, kept where its photometric cost beat the upsampled
plane's by more than 0.1. F1, accuracy and completeness are for one
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
  Upsampled planes that are slightly off cost less than 0.1 more than the
  right ones, so they stayed, and geometric consistency between views
  that all inherited them held them there; ACMMP's order with the
  restorer against the plane at infinity (above) re-estimates them.
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
