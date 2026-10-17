# LaMAria: batch visual-inertial bundle adjustment

The refined Basalt VIO still drifts slowly in heading and scale
([`lamaria_drift_analysis.md`](lamaria_drift_analysis.md)). Its sliding
window only sees short KLT tracks: on sequence_1_19, half of the tracks
last one frame and 90 % at most six, and an established track dies with a
probability of 10 % per frame. Points that stay in view for tens of seconds
are therefore re-detected as new landmarks many times, and nothing ties
keyframes more than a few seconds apart together.

[`scripts/lamaria_batch_ba.py`](../scripts/lamaria_batch_ba.py) refines the
VIO trajectory afterwards with one batch optimization over the whole
sequence. It uses no ground truth.

## Method

1. **Keyframes.** Every 10th frame of the VIO trajectory (2 Hz).
2. **Features.** SIFT on both cameras (OpenCV, 1,500 per image, contrast
   threshold 0.01).
3. **Matching.** Each keyframe is matched against the keyframes 1, 2, 3, 4,
   6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192 and 256 steps later (0.5 to
   128 s), within and across the two cameras, if their optical axes differ
   by less than 75° and the VIO puts them less than 200 m apart (see
   [Matches further apart](#matches-further-apart)). A
   candidate must lie within an epipolar band around the VIO's relative pose:
   2 px plus 0.1 px per keyframe step, at most 6 px. The descriptor ratio
   test (0.8) and a mutual check run among those candidates.
4. **Tracks.** Connected components of the match graph. Components with two
   features in one image are dropped. Each track is triangulated with the
   VIO poses (midpoint). Observations more than 4 px off are dropped, and the
   triangulation angle must reach 1°.
5. **Solve.** Levenberg-Marquardt over:
   - every keyframe's pose, velocity and IMU biases, with GTSAM's
     preintegrated IMU factor between consecutive keyframes (the
     calibration's noise densities);
   - every landmark, with Huber reprojection terms (σ 1 px, threshold 2 px);
   - each camera's IMU-camera rotation, one per eighth of the sequence (see
     [During a recording](#during-a-recording)).

   The landmarks are eliminated by the Schur complement. A landmark keeps
   its observations within 64 keyframes of its first one (`--band`), so the
   reduced system is banded and is factored by LAPACK's banded Cholesky.
   The camera rotations form a dense border. A prior on the first pose fixes
   the gauge.

   Each iteration first drops the observations more than 10 px off, then
   the landmarks left with a single observation. A landmark's damping is its
   Hessian diagonal floored at 1 % of its largest entry. Without both, a
   landmark seen with little parallax could take a step of hundreds of
   metres, and every damped step of an iteration raised the cost.

   From the 8th iteration on the threshold tightens to 3 px. By then the
   solution has settled and the median error is about 0.3 px, so an
   observation still 3 px off is almost always a wrong match. Huber alone
   lets each such observation pull like a 2 px residual; the 1–2 % of them
   bend the trajectory against the IMU (see [Results](#results)).
6. **Output.** The VIO's per-frame trajectory, each frame moved by the
   correction of the surrounding keyframes (interpolated).
7. **Revisits (optional).** `loops` matches places visited again, and
   `loop-correct` turns those matches into relative poses for a pose graph
   over the solved keyframes. A second `solve` then starts from the
   corrected trajectory and adds the revisit matches (see
   [Revisits](#revisits)).

## The IMU-camera rotation of each camera

The batch problem observes each camera's rotation relative to the IMU.

### Device 1WM09370081264

On sequence_1_19 only cam0 is off:

| Estimate | cam0 (in the IMU frame) | cam1 |
|---|---|---:|
| Batch BA, first 75 s | 0.60° | 0.03° |
| Batch BA, whole sequence | [0.49, −0.26, 0.17]° (0.58°) | 0.04° |
| Gyroscope vs pseudo-GT, cam0 ([device table](lamaria_drift_analysis.md#per-device-refinement)) | [0.57, −0.29, 0.37]° (0.74°) | — |

The device table measured the cam0 rotation against the pseudo-GT, which is
given in the cam0 frame. It then rotated the whole rig, which also turned
cam1 by 0.74°. That explains why the table's rotation "overshoots" on this
device.

With cam0's rotation set to the BA estimate, the VIO alone gives these
results on sequence_1_19:

| Metric | Factory rotation | Corrected cam0 |
|---|---:|---:|
| Pose recall at 1 m | 76.3 % | 85.4 % |
| Sim(3) RMSE against the pseudo-GT | 0.88 m | 0.78 m |
| CP Score | 71.96 | 71.25 (14 control points) |

### Device 1WM093701G1276

On sequence_2_11, three 75 s windows each estimate about
[0.14, 0.13, −0.03]° (0.19°) in the IMU frame, for both cameras. The two
cameras agree to within 0.05–0.08°. The device table applies 0.36°.

### During a recording

The rotations also change while the glasses are worn. `--ext-segments 8`
(the default) estimates each camera's rotation separately in eight equal
time segments of the keyframes (2–4 min each), with a prior of 0.05°
(`--ext-seg-sigma`) on the change between consecutive segments. On
1WM093701G1276 both cameras then turn together relative to the IMU, by
about 0.1° over a recording. On 1WM09370081264 the segments stay within
0.02–0.05° of each other:

| Sequence | Device | Range of the common rotation | Correlation of the two cameras |
|---|---|---:|---:|
| sequence_1_19 | 1WM09370081264 | 0.05° (y) | 0.65 |
| sequence_1_20 | 1WM09370081264 | 0.02° (z) | 0.34 |
| sequence_2_11 | 1WM093701G1276 | 0.11° (y) | 0.94 |
| sequence_3_17 | 1WM093701G1276 | 0.09° (y) | 0.97 |
| sequence_4_10 | 1WM093701G1276 | 0.09° (x) | 0.97 |
| sequence_5_11 | 1WM093701G1276 | 0.10° (y) | 0.90 |

The common rotation is the mean of the two cameras' segment estimates, in
the IMU frame. The range is that of its component that varies most, and the
correlation is that of the two cameras' estimates in that component.

The number of segments matters little between four and eight. With the
prior scaled to the same smoothness per unit of time (0.07° for four
segments, 0.035° for sixteen), the first solve scores:

| Sequence | One rotation | 4 segments | 8 segments | 16 segments |
|---|---:|---:|---:|---:|
| sequence_2_11 | 68.88 | **71.84** | 71.18 | 69.97 |
| sequence_3_17 | 50.32 | 59.57 | **60.25** | 59.23 |
| sequence_5_11 | **59.81** | 56.83 | 56.22 | |

## Results

Official evaluator, one training sequence per track, plus sequence_1_20, a
second Short sequence of the device of sequence_1_19. The VIO and mapper
columns are the per-device-refined runs of
[`lamaria_drift_analysis.md`](lamaria_drift_analysis.md#with-the-online-mapper);
on sequence_1_20 the VIO has only the per-device time alignment, and there is
no mapper run. The batch BA starts from that VIO trajectory and uses the
factory camera rig. On sequence_4_10 the VIO is the low-light configuration
(0.5 px KLT threshold) and the BA's SIFT runs on CLAHE-equalized images. The
first version of the solver had neither the outlier rejection nor the
landmark damping floor; the 10 px column has both but keeps 10 px to the end;
the 3 px column also tightens it to 3 px from the 8th iteration on (see
[Method](#method)). The last column, the default, also estimates the camera
rotations in eight time segments (see [During a
recording](#during-a-recording)).

| Sequence | Track | Device | VIO | VIO + mapper | Batch BA, first version | Batch BA, 10 px | Batch BA, 3 px | Batch BA |
|---|---|---|---:|---:|---:|---:|---:|---:|
| sequence_1_19 | Short | 1WM09370081264 | 71.96 | 68.54 | 83.96 | 83.02 | 80.63 | **86.53** |
| sequence_2_11 | Medium | 1WM093701G1276 | 49.68 | 55.51 | 52.29 | 61.03 | 68.88 | **71.18** |
| sequence_3_17 | Long | 1WM093701G1276 | 50.06 | 46.68 | 49.59 | 50.14 | 50.32 | **60.25** |
| sequence_4_10 | Low light | 1WM093701G1276 | 39.00 | 36.65 | 44.28 | 45.79 | 47.43 | **47.65** |
| sequence_5_11 | Moving platform | 1WM093701G1276 | 43.14 | 49.84 | 55.70 | 59.06 | **59.81** | 56.22 |
| **Mean of the five tracks** | | | 50.77 | 51.44 | 57.16 | 59.81 | 61.41 | **64.36** |
| sequence_1_20 | Short | 1WM09370081264 | 38.10 | | 43.48 | 43.39 | 44.43 (**76.13** with `--modes yaw`, see [below](#slow-heading-drift)) | 44.41 |

The 3 px stage drops 1–2 % of the observations (11,400 of 874,000 on
sequence_2_11). The IMU then fits better: on sequence_2_11 the whitened
residual per factor falls from 1.61 to 0.91 for the position and from 6.60
to 4.94 for the velocity, and the fitted scale drift from +2.8 % to +1.8 %.
The trajectory improves on every sequence with pseudo-GT (table below); the
CP Score of sequence_1_19 falls by 2.4 even so, its CP errors growing from
0.33 to 0.37 m at the median, where the score falls steeply.

The eight rotation segments act on the slow drift, not on the local
accuracy. The RMSE of 60 s segments stays the same (table below), while the
Sim(3) RMSE falls on four of the five sequences with pseudo-GT: on
sequence_3_17 from 1.97 to 1.31 m, and its pose recall at 1 m rises from
17.0 % to 56.0 %. The fitted scale drift falls from +2.3 % to +1.4 % on
sequence_3_17 and from +1.9 % to +1.4 % on sequence_2_11 (drift table
below). sequence_5_11 loses 3.6 points: most of its CP errors grow by
0.1–0.6 m (1.09 → 1.23 m at the median, in a 2D similarity fit).

The same runs against the pseudo-GT:

| Sequence | Metric | VIO | VIO + mapper | Batch BA, 10 px | Batch BA, 3 px | Batch BA |
|---|---|---:|---:|---:|---:|---:|
| sequence_1_19 | Pose recall at 1 m | 76.3 % | 60.6 % | 91.5 % | 91.7 % | **94.0 %** |
| sequence_1_19 | Sim(3) RMSE | 0.88 m | 0.96 m | 0.55 m | 0.55 m | **0.44 m** |
| sequence_2_11 | Pose recall at 1 m | 30.9 % | 39.4 % | 51.3 % | 70.2 % | **76.8 %** |
| sequence_2_11 | Sim(3) RMSE | 2.31 m | 1.92 m | 1.51 m | 1.01 m | **0.87 m** |
| sequence_3_17 | Pose recall at 1 m | 24.7 % | 21.7 % | 16.3 % | 17.0 % | **56.0 %** |
| sequence_3_17 | Sim(3) RMSE | 2.07 m | 2.48 m | 2.02 m | 1.97 m | **1.31 m** |
| sequence_4_10 | Pose recall at 1 m | 14.1 % | | 19.3 % | **21.5 %** | 20.9 % |
| sequence_4_10 | Sim(3) RMSE | 3.11 m | | 2.36 m | 2.11 m | **2.05 m** |
| the four above | RMSE of 60 s segments, each Sim(3)-aligned | 0.151 / 0.264 / 0.182 / 0.381 m | | 0.127 / 0.206 / 0.147 / 0.336 m | 0.125 / 0.169 / 0.141 / 0.315 m | 0.125 / 0.169 / 0.141 / 0.317 m |
| sequence_1_20 | Pose recall at 1 m | 3.8 % | | 11.8 % | 14.4 % (**100 %** with `--modes yaw`) | 16.4 % |
| sequence_1_20 | Sim(3) RMSE | 3.03 m | | 2.41 m | **2.31 m** | 2.35 m |

On sequence_5_11 (no pseudo-GT) the BA brings AA2976 from 0.94 m (VIO) to
0.07 m and the fitted scale drift (without PB2) from −5.4 % to −2.5 %. PB2,
the only CP before the funicular ride, stays 19 m off.

The batch BA is the most accurate locally on every sequence with pseudo-GT.
The slow drift that remains, fitted to the CP residuals as a heading and a
scale drift linear in time:

| Sequence | Heading drift, 10 px | Scale drift, 10 px | Heading drift, 3 px | Scale drift, 3 px | Heading drift | Scale drift |
|---|---:|---:|---:|---:|---:|---:|
| sequence_1_19 | −0.14° | −0.6 % | −0.30° | −0.6 % | −0.19° | −0.5 % |
| sequence_2_11 | −0.34° | +2.8 % | −0.47° | +1.9 % | −0.50° | +1.4 % |
| sequence_3_17 | +0.82° | +2.3 % | +0.81° | +2.3 % | +0.48° | +1.4 % |
| sequence_4_10 | −0.58° | +4.0 % | −0.92° | +3.6 % | −0.86° | +3.5 % |
| sequence_5_11 (without PB2) | −0.35° | −2.5 % | −0.22° | −2.3 % | +0.25° | −2.4 % |

The sequences of 1WM093701G1276 keep 1.4–3.5 % of scale drift (2–4 % with
one rotation per camera), and their local scale (Sim(3) fits in 60 s
windows) varies by ±2–4 %, against ±1 % on sequence_1_19. Their IMU factors
also fit worse: the whitened position residual per factor is 1.6–1.8,
against 0.4–0.5 on 1WM09370081264 (10 px solve). These do not remove it
(sequence_2_11, 10 px, one rotation per camera, CP Score 61.03):

| Change | CP Score | Note |
|---|---:|---|
| Accelerometer y axis ×1.03 / ×0.97 | 60.40 / 56.48 | The IMU cost nearly doubles (20,600 → 39,100 / 36,700): the BA pins each axis' scale to well under 1 % |
| Accelerometer delayed by −2 / +2 ms | 61.14 / 61.14 | Final cost +1,100 / −90 |
| Accelerometer bias random walk ×10 | 60.49 | |
| IMU noise densities ×2 | 58.02 | |
| Keyframes at 4 Hz (every 5th frame), matches and band up to 128 keyframes | 60.05 | Scale drift +2.8 → +1.8 %, heading drift −0.35 → −0.02°, IMU position residual 1.61 → 0.75; the CPs of the first half gain what those of the second half lose |
| Camera translations estimated too (prior 1 cm) | 60.21 | They move by 2–4 mm; the scale stays (1.026 against the pseudo-GT) |
| Vertical Earth rotation in the IMU factors (`--omega-coriolis`) | 60.99 | The gyro bias absorbed it already |
| Cross-camera matches left out | 64.89 | With the first rotation prior loose as well, −2.2 on sequence_1_19, −2.0 on sequence_4_10 and −7.9 on sequence_5_11 against the 10 px solve; the stereo constraints carry sequence_5_11's funicular ride |

A fit of the accelerometer against the pseudo-GT's second differences had
suggested a y scale of 0.97 on this device; the BA rules that out. The
camera exposure does not explain the local scale either: it correlates
weakly with it on sequence_2_11 (−0.33), and on sequence_4_10 the exposure
stays at 14 ms while the local scale varies by ±7 %.

The IMU is not what shrinks the trajectory. With the BA's rotations fixed and
its positions replaced by the pseudo-GT's (aligned rigidly per 60 s window),
the scale the preintegrated IMU prefers is 1.006 ± 0.004 on sequence_2_11
(window to window ±1.7 %). The BA instead keeps the global scale of its
starting point, like its heading drift: the VIO is 2.4 % short against the
pseudo-GT, the BA 2.7 %, and a start stretched by 2.5 % ends 0.9 % short at a
slightly lower cost (229,625 against 229,821, with the first rotation prior
loose, see [below](#the-first-keyframes-rotation-prior)). The scale is a weak
direction of the objective; within a 60 s window the IMU constrains it to
only about ±1.7 %.

### Where the scale drift comes from

The scale drift of sequence_2_11 is the BA's own. In Sim(3) fits of 60 s
windows against the pseudo-GT, the VIO's local scale has no trend, the BA's
grows by 2.5 % per 1,000 s, and a solve started at the pseudo-GT drifts as
much: the objective prefers the drift. Changing one ingredient at a time
(eight segments unless noted):

| Run on sequence_2_11 | CP Score | Local scale, trend per 1,000 s |
|---|---:|---:|
| VIO | 49.68 | −0.3 % |
| Batch BA | 71.18 | +2.5 % |
| Started at the pseudo-GT (one rotation per camera, 10 px; diagnosis only) | 54.46 | +2.5 % |
| cam0 only | 64.75 | +2.9 % |
| cam1 only | 60.55 | +3.1 % |
| Cross-camera matches left out | **72.70** | +1.7 % |
| Matches at most 16 keyframes (8 s) apart | 68.19 | +1.7 % |
| IMU noise densities ×0.5 | 66.66 | +3.3 % |
| Intrinsics (fx, fy, cx, cy, k1, k2) estimated per segment too | 66.60 | **+1.2 %** |
| Intrinsics estimated once per camera (one rotation per camera) | 62.19 | +2.3 % |

Each camera alone drifts, so the rig does not cause it. Leaving out the
cross-camera matches or the long matches takes a third of it away, but
neither removes it. The solve leaves a small radial residual on
1WM093701G1276: observations lie 0.02–0.04 px further out than their
projections at 150–300 px from the principal point, on both cameras, against
at most 0.02 px inwards on 1WM09370081264. Estimating the intrinsics finds a
focal length 0.1–0.3 % longer and k1 ≈ +0.002 on both cameras of
1WM093701G1276 (−0.1 % and 0 on sequence_1_19), but that reduces the drift
only when the intrinsics may change per segment, and it costs points either
way.

The radial residual grows with the SIFT keypoint size, on both devices. At
200–250 px from the principal point (about 45° off axis, where the pinhole
image stretches fastest radially), by keypoint octave:

| Sequence | Octave −1 (upsampled) | Octave 0 | Octave 1 | Octave 2 |
|---|---:|---:|---:|---:|
| sequence_2_11 | +0.024 px | +0.038 px | +0.058 px | +0.075 px |
| sequence_1_19 | −0.007 px | −0.001 px | +0.012 px | +0.051 px |

Only the offset common to all octaves differs between the devices. Moving
every keypoint by its octave's and radius's mean residual (`--xy-correction`,
a table from the first solve) leaves the local scale trend of
sequence_2_11 at +2.55 % per 1,000 s (+2.53 % with three times the
correction; CP Score 71.75 and 68.01), and keeping only matches between
keypoints of similar size raises it (+3.22 % within a size ratio of 1.25,
+2.82 % within 1.5). Neither the detector nor the residual distortion drives
the drift.

The ratio of the image's angular scale to the gyroscope's does. Changing the
focal lengths linearly over the recording (`--focal-trend`, the mean
unchanged; first solve, eight segments) moves the trend in proportion:

| Focal length change from the first to the last segment | CP Score | Pose recall at 1 m | Local scale trend per 1,000 s |
|---|---:|---:|---:|
| −0.4 % | 63.42 | 55.8 % | +3.80 % |
| 0 (the calibration) | 71.18 | 76.8 % | +2.48 % |
| +0.4 % | **74.93** | **84.5 %** | +1.32 % |
| +1.2 % | 70.59 | 81.5 % | (fitted scale drift −1.2 %) |

Without outlier rejection, so that every run keeps the same observations,
the objective itself is lowest near +0.4 %: final costs 256,222 (−0.4 %),
254,336 (0), 253,950 (+0.4 %) and 255,206 (+0.8 %). Scaling the gyroscope
instead changes the scale drift alike but turns the heading: a gyroscope
scale from 1.003 to 0.997 over the recording raises the trend to +3.78 %
(CP Score 65.82), the opposite to +1.73 % (60.37, heading drift +0.21°). A
constant gyroscope scale of 1.003 gives 72.79 (pose recall 83.2 %) at a much
higher cost (190,737 against 181,127; the rotation residual per factor goes
from 7.6 to 13.4), so the gyroscope's own scale is right.

The same +0.4 % does not suit every sequence (first solve):

| Sequence | Device | Calibration | Focal length +0.4 % over the recording |
|---|---|---:|---:|
| sequence_1_19 | 1WM09370081264 | **86.53** | 81.24 |
| sequence_2_11 | 1WM093701G1276 | 71.18 | **74.93** |
| sequence_4_10 | 1WM093701G1276 | 47.65 | **50.97** |
| sequence_5_11 | 1WM093701G1276 | **56.22** | 51.27 |
| sequence_1_20 | 1WM09370081264 | 44.41 | **45.29** |

`--est-focal` estimates the focal lengths instead: one scale per camera and
time segment, next to its rotation, with priors on its change between
segments (0.05 %, `--focal-seg-sigma`) and on its value (1 %). The fit
improves on both sides: on sequence_2_11 the reprojection cost falls from
164,733 to 160,559 and the IMU cost from 16,386 to 15,489, and the IMU
position, velocity and rotation residuals per factor all shrink. The
estimates are consistent by device and conditions, which suggests the
temperature:

| Sequence | Device | Conditions | Focal length against the calibration, cam0 / cam1 (mean of the segments) | First and last segment, cam0 |
|---|---|---|---:|---:|
| sequence_1_19 | 1WM09370081264 | day | −0.11 % / −0.08 % | −0.15 % .. −0.15 % |
| sequence_1_20 | 1WM09370081264 | day | −0.11 % / −0.06 % | −0.13 % .. −0.10 % |
| sequence_2_11 | 1WM093701G1276 | day | +0.34 % / +0.26 % | +0.18 % .. +0.36 % |
| sequence_3_17 | 1WM093701G1276 | day | +0.41 % / +0.37 % | +0.32 % .. +0.45 % |
| sequence_4_10 | 1WM093701G1276 | night | −0.04 % / −0.04 % | −0.08 % .. +0.01 % |
| sequence_5_11 | 1WM093701G1276 | day, funicular | +0.57 % / +0.53 % | +0.41 % .. +0.59 % |

The pipeline with the revisits (first solve, `loop-correct`, second solve,
each with `--est-focal`):

| Sequence | First solve | `--est-focal` | Pipeline | `--est-focal` | Pose recall at 1 m, pipeline / `--est-focal` |
|---|---:|---:|---:|---:|---:|
| sequence_1_19 | **86.53** | 83.28 | 87.22 | | 96.5 % / |
| sequence_2_11 | **71.18** | 68.74 | 77.31 | **77.72** | 75.8 % / **94.1 %** |
| sequence_3_17 | 60.25 | **65.96** | 60.00 | **66.16** | 55.5 % / **63.4 %** |
| sequence_4_10 | 47.65 | **48.86** | 49.52 | **50.75** | 24.1 % / **25.8 %** |
| sequence_5_11 | **56.22** | 46.96 | **57.15** | 48.02 | |
| sequence_1_20 | 44.41 | **46.18** | 43.41 | **44.79** | 12.7 % / **14.1 %** |

On the moving platform the estimate goes wrong: there the IMU feels the
funicular, which the scene inside the car does not show, and the focal
lengths absorb the conflict (the fitted scale drift grows from −6.5 % to
−8.6 %). Pinning the mean of each camera's scales to the calibration
(`--focal-mean-sigma 0.00001`) does not save it (48.14).

Elsewhere the mean is what helps: with it pinned, sequence_3_17's first solve
scores 56.49, against 65.96 with the focal lengths estimated and 60.25 with
the calibration's.

### Keypoint size

A SIFT keypoint detected at a coarser scale is located less precisely. The
residuals of the first solve (robust spread, 1.4826 × MAD of the x residual)
by keypoint octave:

| Sequence | Octave −1 (2.7 px) | Octave 0 (4.8 px) | Octave 1 (9.3 px) | Octave 2 (18 px) | Octave 3 (37 px) |
|---|---:|---:|---:|---:|---:|
| sequence_1_19 | 0.19 px | 0.23 px | 0.32 px | 0.49 px | 0.59 px |
| sequence_2_11 | 0.24 px | 0.30 px | 0.40 px | 0.51 px | 0.59 px |
| sequence_4_10 | 0.42 px | 0.43 px | 0.51 px | 0.60 px | 0.66 px |

About ×1.27 per octave, that is σ ∝ size^0.35, except at the finest scale in
low light, where noise dominates. `--sigma-size 2.7 --sigma-size-pow 0.35`
gives each observation σ = 1 px × max(1, size / 2.7 px)^0.35 (first solve):

| Sequence | σ = 1 px | σ by keypoint size |
|---|---:|---:|
| sequence_1_19 | 86.53 | **87.46** |
| sequence_2_11 | 71.18 | **72.57** |
| sequence_3_17 | **60.25** | 60.09 |
| sequence_4_10 | **47.65** | 46.65 |
| sequence_5_11 | 56.22 | **56.54** |
| **Mean of the five tracks** | 64.36 | **64.66** |
| sequence_1_20 | 44.41 | **44.94** |

### Low light: features after denoising

On sequence_4_10 the BA keeps 191 observations per keyframe, against 360–500
on the other sequences, and its 60 s segments are twice as far off (0.32 m
RMSE). The images are dark (median grey level 10–30) and noisy at gain 15,
and the CLAHE equalization amplifies the noise: half the matches between
consecutive keyframes of the other sequences survive (137 instead of 259–362
per image pair), and 72 % of the keypoints come from the upsampled octave,
where the residuals are no smaller than one octave up (0.42 px MAD on
sequence_4_10, against 0.24 px on sequence_2_11).

Non-local means denoising (OpenCV, h = 7) before the equalization
(`features --denoise 7`) leaves fewer keypoints, but the ones that match agree
with the IMU. Each variant below runs features, matching and a solve on one
200 s window of keyframes; the RMSE is that of a Sim(3) fit to the pseudo-GT
over the window, and the IMU column the whitened position residual per
factor:

| Variant | Keypoints per image | Observations | RMSE, keyframes 1000–1400 | IMU | RMSE, 200–600 | RMSE, 2200–2600 |
|---|---:|---:|---:|---:|---:|---:|
| CLAHE (the default for low light) | 1,500 | 87,593 | 0.890 m | 2.09 | 0.706 m | **0.440 m** |
| No equalization | 1,403 | 81,460 | 0.809 m | 1.80 | | |
| CLAHE, clip limit 1.5 instead of 3 | 1,500 | 85,757 | 0.904 m | 2.27 | | |
| Gaussian blur (σ 1 px) before CLAHE | 1,500 | 95,751 | 0.889 m | 2.20 | | |
| 3,000 keypoints | 3,000 | 127,258 | 0.864 m | 4.58 | | |
| Non-local means, h = 4 | 1,083 | 82,983 | 0.476 m | 1.16 | | |
| Non-local means, h = 7 | 679 | 62,415 | **0.399 m** | 0.69 | **0.462 m** | 0.471 m |
| Non-local means, h = 10 | 508 | 50,101 | 0.428 m | 0.43 | | |
| Non-local means, h = 7, contrast threshold 0.005 | 1,326 | 74,238 | 0.900 m | 1.96 | | |

The weak keypoints are the problem: with the denoising but a lower contrast
threshold, the window is as far off as without it.

On the whole sequence the denoised features (556 per image, 2.6 million
matches against 4.1 million) take the fitted scale drift from +3.5 % to
+0.8 % and most CP errors under a metre:

| sequence_4_10 | First solve | + `loop-correct` | + second solve | Pose recall at 1 m |
|---|---:|---:|---:|---:|
| CLAHE | 47.65 | 47.54 | 49.52 | 24.1 % |
| Non-local means (h = 7), then CLAHE | 63.31 | 62.52 | 63.73 | **68.0 %** |
| The same, with `--est-focal` | **63.64** | **63.26** | **64.30** | 66.9 % |

What is left comes from the observations, not from the optimizer: the same
first solve started at the pseudo-GT moves away from it and ends at 60.10
(pose recall 45.9 %), below the solve started at the VIO. The local error
of 60 s segments is still about twice that of the day sequences (0.15–0.43 m),
the heading wanders by ±1° over the recording, and 2D errors of 1.5–2 m
remain at 60–180 s, 840 s and 1,020–1,080 s.

Other preprocessing and detector settings on top of the denoising, on four
200 s windows (Sim(3) RMSE against the pseudo-GT; one IMU-camera rotation
per camera):

| Variant | Keyframes 200–600 | 1000–1400 | 1600–2000 | 2200–2600 |
|---|---:|---:|---:|---:|
| Non-local means (h = 7), then CLAHE | 0.462 m | 0.399 m | 0.619 m | **0.471 m** |
| The same, SIFT contrast threshold 0.02 instead of 0.01 | **0.429 m** | **0.340 m** | **0.457 m** | 0.523 m |
| The same, contrast threshold 0.03 | 0.449 m | | | |
| Non-local means h = 10, contrast threshold 0.02 | 0.518 m | | | |
| CLAHE clip limit 2 instead of 3 | 0.484 m | 0.382 m | 0.578 m | 0.562 m |
| Gamma 0.5 instead of CLAHE | 0.516 m | 0.344 m | | 0.478 m |
| Keyframes at 4 Hz (offsets and band doubled) | | 0.344 m | | |
| Keyframes at 4 Hz, contrast threshold 0.02 | | 0.342 m | | |

No equalization, CLAHE tiles of 4 × 4, dropping the keypoints on dark
ground (mean grey level under 10) or larger than 20 px, and RootSIFT
descriptors (0.809 m) do not beat the denoised baseline on the 1000–1400
window either. The contrast threshold of 0.02 wins three windows of four, but
on the whole sequence (402 keypoints per image, 2.1 million matches) it ends
where the baseline does: 62.97 after the first solve and **64.35** after the
second (pose recall 62.1 %). The 200 s windows measure the local shape, while
the CP Score of this one-way route follows the drift over minutes.

The VIO's KLT tracks cannot fill in either. `basalt_euroc_vio_demo
--dump-tracks` writes the tracked points of every frame. A track lives a
median of one frame, 1.8 frames on average on sequence_4_11 at night and
2.5 by day (the first 2,000 frames of sequence_3_17); sampled at the 2 Hz
keyframes, 20–28 tracks per keyframe are seen twice or more, against 45–162
SIFT landmarks.

Learned features do not help on this window either. XFeat (1,500 keypoints
per image) and ALIKED-n16 (score threshold 0.2, 494 keypoints per image), on
the raw images and matched by the same VIO-guided ratio test, end at 0.833 m
and 0.836 m on the 1000–1400 window, against 0.399 m for denoised SIFT. They
locate their keypoints less precisely (median reprojection error 0.60 and
0.52 px, against 0.47 px), and the solve bends against the IMU (position
residual 4.4 and 1.6 per factor, against 0.69).

### The first minutes of a recording

The gyroscope disagrees with the solved rotations at the start of every
recording. The whitened rotation residual per IMU factor (median over 15 s,
eight-segment solves) falls to its usual 1–2 within about two minutes:

| Sequence | 0–15 s | 15–30 s | 30–45 s | 45–60 s | 60–75 s | 75–90 s | 90–105 s |
|---|---:|---:|---:|---:|---:|---:|---:|
| sequence_1_19 | 25.9 | 13.6 | 2.4 | 2.3 | 1.9 | 6.2 | 1.9 |
| sequence_2_11 | 96.8 | 10.9 | 9.3 | 7.7 | 4.6 | 3.9 | 3.2 |
| sequence_3_17 | 11.7 | 8.1 | 5.7 | 2.9 | 2.6 | 3.7 | 1.1 |
| sequence_4_10 | 305.6 | 16.0 | 16.3 | 9.9 | 10.6 | 4.6 | 3.6 |

On sequence_4_10 the residual of the first two minutes does not depend on
how far the head turns between keyframes (about 10 for turns of 2–5° and of
more than 20°), which points at a bias rather than a scale: the gyro bias
still settles while the device warms up, and the tight bias random walk
(1e-7) keeps it constant. The recordings carry no IMU temperature (the field
is NaN in the VRS records). Letting the bias move does not help in general
(first solve):

| Run | sequence_2_11 | sequence_4_10 |
|---|---:|---:|
| Default | **71.18** | 47.65 |
| Gyro bias random walk ×100 | | 49.38 |
| Gyro bias random walk ×1000 | 66.56 | 50.53 |
| ×1000 at the start, decaying with a time constant of 120 s (`--gyro-rw-warmup 120`) | 67.51 | **50.72** |
| The same with 300 s | | 49.29 |
| Accelerometer bias random walk ×30 at the start, 120 s | 68.36 | |

sequence_4_11, the second low-light sequence, shows how much is at stake.
Its VIO drifts in heading (−1.5° fitted to the CPs), and the BA keeps most
of it (−1.3°, CP Score 55.30 after the first solve). With the gyro bias
random walk ×100, the gyro bias of its first keyframe triples (2.8 mrad/s on
the y axis, then 0.45 mrad/s after nine minutes), the heading drift falls to
−0.5° and the first solve scores **70.47** (pose recall 28 % → 60 %). The
other sequences of 1WM093701G1276 let their gyro bias move as much with the
looser random walk (by 1–5 mrad/s on the y axis), and lose; the stable
1WM09370081264 gains a little. First solve with `--est-focal --sigma-size 2.7
--sigma-size-pow 0.35` (sequence_5_11 without `--est-focal`):

| Gyro bias random walk | sequence_1_19 | sequence_1_20 | sequence_2_11 | sequence_3_17 | sequence_4_10 | sequence_4_11 | sequence_5_11 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ×1 (1e-7, the default) | 84.12 | 46.64 | **70.93** | 65.24 | 62.18 | 55.30 | **56.54** |
| ×10 | **86.36** | **47.22** | 68.27 | **66.12** | **62.49** | 54.67 | 53.91 |
| ×100 | 85.64 | 46.66 | 64.10 | 61.47 | 55.21 | **70.47** | 55.30 |

Neither the costs nor the IMU residuals tell the cases apart: the cost falls
by 900–1,050 with ×100 on both sequence_4_11 and sequence_2_11, and
sequence_4_11 has the smallest whitened gyroscope residual of all with the
default (1.7 per factor, against 3.5–8.1 elsewhere). The default stays.

### What the gain on sequence_1_19 comes from

First version of the solver:

| Run | CP Score | Pose recall at 1 m |
|---|---:|---:|
| VIO | 71.96 | 76.3 % |
| VIO with the BA's cam0 rotation | 71.25 | 85.4 % |
| Batch BA, matches at most 4 keyframes (2 s) apart | 70.87 | 85.8 % |
| Batch BA, matches up to 64 keyframes (32 s) apart | **83.96** | 91.4 % |
| Batch BA, matches up to 128 keyframes (64 s) apart | 80.52 | **92.4 %** |

The camera rotation accounts for the first step in pose recall. The CP Score
gain comes from the long-range matches.

### Revisits

`loops` matches keyframes that are more than 64 keyframes apart but within
15 m (by the VIO) and look in similar directions. It matches them without a
pose prior, keeps a pair only if an essential matrix finds at least 25
inliers, and checks the rotation against the VIO (10°). On sequence_2_11 it
accepts 1,566 keyframe pairs:

- They lie 5.1 m apart in the pseudo-GT at the median and 19.3 m at most,
  so there is no false revisit.
- 750 of them are the revisit 120–240 s later between 570 s and 820 s.

Landmarks seen on both visits couple keyframes outside the band. The solver
then runs conjugate gradients on the full reduced system, preconditioned by
the banded factor, to a relative residual of 1e-4 (`--pcg-tol`, at most 200
iterations). Solving each step to 1e-9 instead took 5,816 iterations over the
30 steps of sequence_2_11's second solve, against 3,156, for the same result
(CP Score 77.31 and 77.30).

Added straight to the solve, the revisits do not close the loops. The
solved sequence_2_11 is about 1.3 m off between its two visits, and a revisit
observation that far off fails the 4 px triangulation check or the 10 px
rejection of the first iterations. What survives already agrees with the
trajectory: the solve keeps fewer observations than without the revisits
(845,000 against 874,000, as merged tracks conflict more often), stops at its
iteration limit and scores 53.15, against 68.88 without them. On
sequence_1_20 the 10 px solve fell from 43.39 to 38.97 the same way.

`loop-correct` closes the loops before a solve sees them:

1. It reads a finished solve (`solve --save-states`). For each revisit image
   pair (a, b), the features of a that are observations of a solved landmark
   give 3D points of a's visit. PnP with RANSAC (OpenCV, 3 px, at least 15
   inliers) places b's camera against them, and a's against b's visit. Each
   placement is a relative pose between keyframes a and b.
2. An edge that disagrees with the solve by more than 0.3 m plus 1 cm per
   second between the visits, or by more than 2°, is dropped: so much drift
   is not plausible.
3. The rest join a pose graph (GTSAM) whose odometry edges are the solve's
   consecutive relative poses (2e-4 rad and 2 cm per keyframe step); the loop
   edges are Cauchy-robust (0.005 rad, 0.1 m). The graph's keyframe
   corrections are spread over every frame, as `solve` spreads its own.
4. A second solve starts from that trajectory (as `--vio`) with `--loops`.
   The triangulation check and the outlier rejection now measure the revisit
   observations against the corrected trajectory.

On sequence_2_11 PnP places 1,160 edges, 1,050 within the gate. They disagree
with the solve by 1.29 m at the median, and the 941 edges of the revisit
120–240 s later agree with each other (1.67 m at the 90th percentile). The
graph moves the keyframes by 0.48 m at the median and 3.0 m at most, and the
revisited CP AA1812 goes from 1.8 to 1.1 m. With one rotation per camera
(the 3 px solve):

| Sequence | Revisit image pairs | Loop edges in the gate | Batch BA, 3 px | + `loop-correct` | + second solve |
|---|---:|---:|---:|---:|---:|
| sequence_1_19 | 966 | 220 | 80.63 | **81.79** | 81.76 |
| sequence_2_11 | 1,922 | 1,050 | 68.88 | 71.53 | **73.34** |
| sequence_3_17 | 415 | 45 | **50.32** | 50.15 | 50.06 |
| sequence_4_10 | 540 | 31 | 47.43 | 47.29 | **49.24** |
| sequence_5_11 | 529 | 84 | 59.81 | **61.31** | 61.20 |
| **Mean of the five tracks** | | | 61.41 | 62.41 | **63.12** |
| sequence_1_20 | 1,052 | 266 | **44.43** | 43.33 | 43.35 |

With the default eight rotation segments, in both solves:

| Sequence | Loop edges in the gate | Batch BA | + `loop-correct` | + second solve |
|---|---:|---:|---:|---:|
| sequence_1_19 | 219 | 86.53 | **87.35** | 87.22 |
| sequence_2_11 | 1,065 | 71.18 | 75.59 | **77.31** |
| sequence_3_17 | 46 | **60.25** | 60.03 | 60.00 |
| sequence_4_10 | 31 | 47.65 | 47.54 | **49.52** |
| sequence_5_11 | 84 | 56.22 | **57.45** | 57.15 |
| **Mean of the five tracks** | | 64.36 | 65.59 | **66.24** |
| sequence_1_20 | 270 | **44.41** | 43.44 | 43.41 |

The segmented solve of sequence_2_11 disagrees less with its revisits
(0.94 m at the median instead of 1.29 m), and the loops add 6.1 points
instead of 4.5.

On sequence_1_20 the revisits lie mostly under a minute apart and the solve
agrees with them already (0.10–0.18 m at the median). They cannot correct its
slow heading drift, and the 10 % of edges that disagree by more than a metre
cost a point.

### With the focal lengths, the keypoint sizes and denoising

The same pipeline (first solve, `loop-correct`, second solve), each solve
with the focal lengths estimated per segment (`--est-focal`, see [Where the
scale drift comes from](#where-the-scale-drift-comes-from)) and the pixel
noise by keypoint size (`--sigma-size 2.7 --sigma-size-pow 0.35`, see
[Keypoint size](#keypoint-size)); on sequence_4_10 the features are denoised
first ([Low light](#low-light-features-after-denoising)), and on the moving
platform the focal lengths stay at the calibration. Final CP Score (pose
recall at 1 m):

| Sequence | Track | Before (eight segments) | Focal lengths | Focal lengths and keypoint sizes |
|---|---|---:|---:|---:|
| sequence_1_19 | Short | **87.22** (96.5 %) | 84.47 (95.9 %) | 84.14 (**99.3 %**) |
| sequence_2_11 | Medium | 77.31 (75.8 %) | 77.72 (94.1 %) | **81.26** (**98.3 %**) |
| sequence_3_17 | Long | 60.00 (55.5 %) | **66.16** (**63.4 %**) | 65.38 (61.1 %) |
| sequence_4_10 | Low light | 49.52 (24.1 %) | **64.30** (66.9 %), denoised | 63.99 (59.6 %), denoised |
| sequence_5_11 | Moving platform | 57.15 | 57.15 | **57.70** (keypoint sizes only) |
| **Mean of the five tracks** | | 66.24 | 69.96 | **70.49** |
| sequence_1_20 | Short | 43.41 (12.7 %) | 44.79 (14.1 %) | **45.36** (**17.0 %**) |

The denoised features alone, without the focal lengths, give sequence_4_10
63.73 (68.0 %).

### Matches further apart

`match` used to stop at 64 keyframes (32 s) and 80 m. Matching also the
keyframes 96, 128, 192 and 256 steps later (up to 128 s), up to 200 m apart,
adds 23–24 % image pairs and 4–6 % matches (sequence_2_11: 81,565 pairs and
6.30 million matches, against 66,322 and 6.04 million) and takes up to
twice as long. The solve must take them in: its `--max-offset` was 64 as
well, which dropped them again and gave the same result as before.

The first solve still keeps a landmark's observations within 64 keyframes
of its first one; there the longer matches only merge tracks. The second
solve, which keeps every observation of a landmark, gains the most
(sequence_4_10: 62.18 → 63.44 after the first solve, 63.99 → 67.81 after the
second). The merged tracks are fewer and longer, 8–11 % fewer observations
pass the triangulation check against the starting trajectory, and the
solves take no longer. Final CP Score (pose recall at 1 m), the pipeline and
options of the table above:

| Sequence | Track | Up to 64 keyframes, 80 m | Up to 256 keyframes, 200 m |
|---|---|---:|---:|
| sequence_1_19 | Short | **84.14** (**99.3 %**) | 83.15 (93.4 %) |
| sequence_2_11 | Medium | 81.26 (98.3 %) | **83.83** (**99.3 %**) |
| sequence_3_17 | Long | 65.38 (61.1 %) | **65.41** (**61.7 %**) |
| sequence_4_10 | Low light | 63.99 (59.6 %) | **67.81** (**68.4 %**) |
| sequence_5_11 | Moving platform | **57.70** | 57.17 |
| **Mean of the five tracks** | | 70.49 | **71.47** |
| sequence_1_20 | Short | 45.36 (17.0 %) | **46.76** (**23.0 %**) |
| sequence_4_11 | Low light | 54.09 (27.2 %) | **56.49** (**37.6 %**) |

sequence_4_11 starts from the VIO of the test-submission pipeline
(`scripts/run_lamaria_test_submission.py --split training`), with denoised
features. These are now the defaults (`match --offsets`, `match
--max-dist`, `solve --max-offset`).

### What is left on sequence_3_17

The CP errors left on the Long sequence are not a slow drift: fitting a
heading or scale drift to them leaves the median at 0.80 m (0.77 m without).
Against the pseudo-GT, the heading stays within ±0.7° for 1,500 s, then turns
by +0.7° between 1,500 and 1,590 s, and the position is 1.4 m further off
between 1,410 and 1,500 s (the CPs ETH2 and ETH1 there are 2.8 m and 2.0 m
off). That stretch crosses an open area against the sun, with dark ground and
a car passing close: the keyframes keep 160–170 observations (64 at the
least), against about 500 elsewhere. Over 1,380–1,580 s the VIO itself is
0.80 m off after a Sim(3) fit, its two halves 2.2 % apart in scale, and the
pipeline is 0.92 m off. Contrast equalization does not help these day
images on that window (one rotation per camera): 0.63 m without, 0.66 m with
CLAHE, 0.60 m with a clip limit of 1.5, 0.75 m with a gamma of 0.7.

In every image region the BA rejects about 1 % of the observations, so the
wearer's feet and the passers-by in the lower part of the images are already
left out by the matching.

### IMU-camera time offset

The BA reports the whitened IMU residual per factor. A sweep of the gyroscope
time offset on top of the per-device alignment (±2 ms, 150 s windows)
minimizes the total cost within 0.5 ms of the device table on both devices
(−0.4 ms on sequence_1_19, −0.5 ms on sequence_2_11). The time alignment
is therefore not what limits the sequences of 1WM093701G1276.

Nor does the exposure shift it. sequence_4_10's SLAM cameras expose for
13.98 ms throughout (gain 15), sequence_2_11's for 0.08 ms at the median (the
VRS image records), so a timestamp at the start or the end of the exposure
would move sequence_4_10 by 7 ms against the device table. Shifting both IMU
channels instead raises the cost of 150 s windowed solves (keyframes 100–400
and 1500–1800, one rotation per camera) on both sides:

| IMU shifted by | 0 ms | −6 ms | +6 ms | −12 ms | +12 ms |
|---|---:|---:|---:|---:|---:|
| Final cost, window 1 | **23,201** | 30,381 | 30,663 | 40,583 | 40,596 |
| Final cost, window 2 | **19,785** | 24,655 | 24,846 | 31,228 | 30,672 |

The timestamps are the middle of the exposure.

## Slow heading drift

### The solution keeps its starting point's drift

On sequence_1_20 the VIO trajectory drifts −5.3° in heading over 17
minutes, and the batch BA still ends at −4.6° (fitted to the CPs, CP Score
43.4). The drift is linear in time and does not follow the turns.

The usual suspects do not cause it:

| Run | Heading drift after the BA (fitted to the CPs) | CP Score |
|---|---:|---:|
| Both cameras | −4.64° | 43.4 |
| cam0 only | −5.99° | 43.4 |
| cam1 only | −4.55° | 44.1 |
| Gyro bias random walk ×1000 | −4.92° | 41.6 |
| Focal lengths, principal points and radial distortion estimated too | −4.42° | 45.2 |

The estimated focal lengths change by −0.13 % and the radial terms stay
near zero.

The starting point decides instead. Given the same observations, the BA
ends near the drift it starts from:

| Starting trajectory | Its heading drift (against the pseudo-GT) | After the BA (fitted to the CPs) | CP Score | Final cost |
|---|---:|---:|---:|---:|
| The VIO | −5.3° | −4.6° | 43.4 | 198,233 |
| A VIO with the device table's rig rotation | +10.5° | +7.2° | 29.3 | 200,686 |
| The pseudo-GT (diagnosis only) | 0° | −1.0° | 74.4 | (fewer observations) |
| The VIO, turned by −0.31°/min (diagnosis only) | −0.4° | −0.7° | 74.6 | 198,185 |

The objective barely tells these solutions apart: the drift-free one
started from the turned VIO ends 48 below the drifted one, out of 198,000.
Nothing separates them either. Along the straight path from the solution
started at the VIO to the one started at the pseudo-GT the cost falls
monotonically (by 12,400, most of it local differences), and it is nearly
flat at the first end. A heading drift that grows linearly in time moves
every keyframe and landmark a little. The gyroscope hardly sees it, because
one constant bias offset absorbs it; vision sees it only through the long
chain of overlapping windows. The damped Levenberg-Marquardt steps do not
follow such a long, flat direction: each one moves it by a fraction of a
degree, and the solve stops at its relative tolerance first.

### Deformation modes

`--modes yaw` adds that direction to the reduced system as one more border
unknown, next to the camera rotations: keyframe k turns about the vertical
by s_k times the mode's amplitude (s from 0 at the first keyframe to 1 at
the last). Positions are re-integrated from the turned displacements,
velocities turn, every landmark moves with its first keyframe, and the
gyro bias takes one offset (the mean body-frame vertical times the rate),
so that the preintegrated rotations still agree.

- The landmark coupling of the mode is eliminated with the landmarks, so
  they follow it optimally.
- The mode is undamped until a step fails; a rejected step damps it
  relative to its own curvature, and LM falls back to plain steps.
- Each step of the mode is capped at 0.01 rad (a trust region solved on
  the border alone).
- The mode's share of a step is applied as the exact finite deformation.
  GTSAM's Pose3 retraction couples rotation and translation, and a 0.09 rad
  turn through it would move far keyframes by decimetres.
- The mode joins at the fourth iteration, after the large first steps.

`--modes scale` stretches the displacements linearly in time instead.

On sequence_1_20 the yaw mode settles at −0.090 rad after 17 iterations:
CP Score 74.8 instead of 43.4, pose recall at 1 m 100 % instead of 12 %, a
fitted heading drift of −0.5°, from the VIO and without any ground truth.

Where the VIO starts with little drift, the mode moves the heading by a few
tenths of a degree, mostly the wrong way:

| Sequence | Without | `--modes yaw` | Fitted heading drift without / with |
|---|---:|---:|---:|
| sequence_1_19 | **83.02** | 74.81 | −0.14° / +0.46° |
| sequence_2_11 | **61.03** | 59.35 | −0.34° / −0.60° |
| sequence_3_17 | 50.14 | **53.64** | +0.82° / −0.05° |
| sequence_4_10 | **45.79** | 40.30 | −0.58° / −1.31° |
| sequence_5_11 | **59.06** | 56.19 | −1.88° / −2.25° (with PB2) |
| sequence_1_20 | 43.39 | **74.84** | −4.64° / −0.49° |

The objective is nearly flat along this mode: on sequence_1_20 the drifted
and the drift-free solutions end within 0–1,000 of each other out of
198,000. Its minimum is therefore only good to a few tenths of a degree,
which costs more than it brings when the start is already that good. The
mode stays off by default; it is the remedy for a start that drifts by
degrees, such as a VIO without the per-device refinement.

The tighter outlier threshold does not change that (3 px solve with one
rotation per camera, with and without the mode): sequence_1_20 goes from
44.43 to **76.13**, but sequence_1_19 from 80.63 to 74.87, sequence_2_11 from
68.88 to 59.17 and sequence_4_10 from 47.43 to 40.42. The cost cannot pick the cases either: the
mode lowers it by 126 on sequence_1_20 and by 107 on sequence_4_10, where it
turns the heading the wrong way, out of about 150,000–200,000. Neither does
Earth's rotation: with its full vector in the IMU factors (`--omega-coriolis`,
the north taken from the control points' LV95 frame for this check only), the
mode still ends at +0.50° on sequence_1_19 (74.50).

The current solver (eight rotation segments, the 3 px threshold) does the
same (first solve):

| Sequence | Without | `--modes yaw` | Cost on the observations active in both, without / with |
|---|---:|---:|---:|
| sequence_1_19 | **86.53** | 76.25 | |
| sequence_2_11 | **71.18** | 61.01 | 180,923 / 180,903 |
| sequence_4_10 (denoised features) | **63.31** | 51.50 | |
| sequence_1_20 | 44.41 | **77.74** (pose recall 99.7 %) | 147,923 / 147,685 |

On sequence_2_11 the mode also lowers the cost, by 20, and loses 10 points.

The scale mode, where the accelerometer does observe the mode, helps four
of the five tracks a little, but the low-light one, whose vision is weak,
loses 7 points:

| Sequence | Without | `--modes scale` | Mode at the end |
|---|---:|---:|---:|
| sequence_1_19 | 83.02 | **83.52** | −1.3 % |
| sequence_2_11 | 61.03 | **63.02** | +0.4 % |
| sequence_3_17 | 50.14 | **50.32** | −0.3 % |
| sequence_4_10 | **45.79** | 38.75 | −5.5 % |
| sequence_5_11 | 59.06 | **60.22** | +2.0 % |
| **Mean** | **59.81** | 59.17 | |

It stays off by default as well.

### A heading drift that is not linear: sequence_2_12

sequence_2_12 (Medium, 28 minutes) starts from a VIO whose heading error,
against the pseudo-GT per 120 s window, runs −4.1° … +1.6° … −0.4°: it is
neither linear in time nor a single step. The batch BA keeps it (−5.4° …
+1.2°): CP Score 21.0, pose recall 0 % (the VIO: 18.0). Locally it is
fine, with its 60 s segments 0.25 m off.

Started from the same VIO with only the heading error removed (from the
pseudo-GT, per 60 s window; diagnosis only), the first solve keeps the
heading within ±0.4° and scores 55.23 (pose recall 34.7 %, Sim(3) RMSE
1.67 m against 5.56 m). It also ends lower, at 172,310 against 173,335 with
0.3 % fewer observations, and its IMU cost is lower too. The better
solution is there in the objective; the solve from the VIO does not reach
it. Nothing tried so far gets there without the pseudo-GT (first solve, CP
Score):

| Run | CP Score | Heading error per 300 s (°) |
|---|---:|---|
| Default | 20.98 | |
| Gyro bias random walk ×100 | 21.77 | −3.32 −1.58 −0.15 +0.58 +0.73 +0.18 |
| Warm-up random walk (`--gyro-rw-warmup 300`) | 20.90 | −2.80 −1.58 −0.14 +0.66 +0.70 +0.10 |
| `--modes yaw` | 24.14 | −3.41 −1.49 −0.31 +0.42 +0.72 +0.49 |
| `--modes yaw8` (eight time pieces) | 17.65 | −3.03 −1.92 −0.57 +0.79 +2.18 +2.47 |
| IMU noise halved (`--imu-noise-scale 0.5`) | 19.22 | −4.24 −2.09 −0.40 +0.43 +0.73 +0.72 |
| No focal lengths, no keypoint sizes | 18.09 | −4.21 −2.12 −0.44 +0.40 +0.65 +0.47 |
| The same, one rotation per camera | 18.43 | −4.32 −2.21 −0.52 +0.40 +0.78 +0.64 |
| Started with the heading removed (pseudo-GT, diagnosis) | **55.23** | −0.09 −0.09 −0.24 −0.12 +0.40 +0.41 |
| Started at the pseudo-GT (diagnosis; 30 % of the observations) | **71.82** | +0.28 −0.01 −0.07 −0.18 +0.10 −0.12 |

`--modes yawN` splits the heading mode into N time pieces. Each piece turns
the heading over its own piece only, and the gyro bias takes the piece's turn
spread over the whole recording: a constant bias absorbs the pieces' mean
rate, and the gyroscope sees how they differ. Its steps on sequence_2_12
change sign from one iteration to the next, and it stops higher than the
default. Shapes chosen without ground truth do not help either. Starts with
an exponential heading change A (1 − e^(−t/300 s)) cost more than the VIO
after eight iterations for A = −7.5° (where the pseudo-GT puts it) and only
slightly less for −2.5° (230,149 against 230,437). Turning the solve to
the heading the gyroscope integrates with the solve's bias doubles the
error: that heading drifts +5° the other way.

The VIO does not cause it at its start either. With the solve's gyro bias
subtracted from the start (`calib_gyro_bias`), or with a 100 times looser
gyro bias random walk, its first five minutes keep the same heading error
(−3.6° and −4.6° in the first 30 s, against −4.0°). Within 30–90 s the
heading changes the solve, the gyroscope and the pseudo-GT measure between
pseudo-GT timestamps agree to 0.2–0.4° per interval. The solve still turns
1.5° more than the other two over the minute.

### The first keyframe's rotation prior

The gauge prior pins the first keyframe's pose, rotation included, at the
VIO's. On sequence_2_11 the first 30 IMU factors (15 s) then carry half of
the whole sequence's gyroscope residual and 44 % of its velocity residual:
the first keyframe has no visual observation, the VIO initialized while the
wearer walked, and its tilt is off. `--prior-rot-sigma 0.05` (rad) lets the
solve fix the tilt; gravity observes roll and pitch, and the 0.05 rad prior
still fixes the heading's gauge. The IMU cost halves on every sequence
(sequence_2_11: 20,600 → 11,700), but the CP Scores split (with the 3 px
threshold, prior loose minus pinned): +0.6, +0.1, −1.0, −4.6 and +4.2 on
sequence_1_19, sequence_1_20, sequence_2_11, sequence_4_10 and
sequence_5_11. A different start shifts the gyro bias and with it the
heading drift, along the same flat direction as above. It stays at 1e-4 by
default.

### Starting from the online mapper

On Medium and Moving platform the online mapper's trajectory beats the VIO
([`lamaria_plan.md`](lamaria_plan.md), lever 11). Starting the BA from it
(the observations still selected with the VIO) gives:

| Sequence | Start: VIO | Start: mapper | Final cost VIO / mapper start | Fitted heading drift VIO / mapper start |
|---|---:|---:|---:|---:|
| sequence_2_11 | **61.03** | 54.41 | 240,741 / 228,230 | −0.34° / +0.67° |
| sequence_5_11 | 59.06 | **62.84** | 294,609 / 293,437 | |

On sequence_2_11 the mapper start ends better by every measure but the
Score: a lower cost, half the IMU cost (11,300 against 20,600), the same
Sim(3) RMSE against the pseudo-GT (1.50 m) and a lower 60 s segment RMSE
(0.199 m against 0.206 m). It keeps the mapper's heading drift, though, and
that is what the CPs see. The lowest cost does not pick the better start.

## Usage

```bash
pip install numpy scipy opencv-python gtsam numba
S=sequence_1_19
python scripts/lamaria_batch_ba.py features --mav0 data/$S/mav0 --vio vio/$S.tum --work work/$S
python scripts/lamaria_batch_ba.py match --work work/$S --vio vio/$S.tum --calib calib/$S.json
python scripts/lamaria_batch_ba.py solve --work work/$S --vio vio/$S.tum --calib calib/$S.json \
  --imu data/$S/mav0/imu0/data.csv --out out/$S.tum
```

With the revisits:

```bash
python scripts/lamaria_batch_ba.py loops --work work/$S --vio vio/$S.tum --calib calib/$S.json
python scripts/lamaria_batch_ba.py solve --work work/$S --vio vio/$S.tum --calib calib/$S.json \
  --imu data/$S/mav0/imu0/data.csv --out out/$S.ba1.tum --save-states
python scripts/lamaria_batch_ba.py loop-correct --work work/$S --calib calib/$S.json \
  --states out/$S.ba1_states.npz --tum out/$S.ba1.tum --out out/$S.lc.tum
python scripts/lamaria_batch_ba.py solve --work work/$S --vio out/$S.lc.tum --calib calib/$S.json \
  --imu data/$S/mav0/imu0/data.csv --out out/$S.tum --loops
```

- **IMU.** `--imu` is the IMU as the VIO read it, with the factory
  rectification and the per-device time alignment.
- **Calibration.** `--calib` is the Basalt calibration of the ASL pinhole
  images, with the factory camera rig: the BA estimates the camera rotations
  itself.
- **Output.** `solve` writes the corrected trajectory and a JSON summary
  next to it: the camera rotations, the biases, the costs and the whitened
  IMU residual per factor.
- **Robustness.** `--outlier-px` (10) and `--lm-damp-floor` (0.01) are the
  outlier rejection and the landmark damping floor; 0 turns them off.
  `--outlier-px2` (3) is the tighter threshold from iteration
  `--outlier-start2` (8) on.
- **Camera rotations.** `--ext-segments` (8) is the number of time segments
  with their own IMU-camera rotations, `--ext-seg-sigma` (0.05°) the prior
  between consecutive ones; `--ext-segments 1` keeps one rotation per camera.
  `--est-extrinsic` (01) picks the cameras whose rotations are estimated.
- **Revisits.** `solve --save-states` keeps what `loop-correct` needs.
  `loop-correct --gate BASE,RATE,MAX_ROT` (0.3 m, 0.01 m/s, 2°) sets the
  drift a loop edge may disagree by.
- **Other options.** `--prior-rot-sigma` loosens the first keyframe's
  rotation prior (1e-4 rad). `--omega-coriolis x,y,z` puts Earth's rotation
  (rad/s, in the VIO's world frame) into the IMU factors.
- **Heading drift.** `--modes yaw` lets the solve correct a slow heading
  drift of its starting trajectory (see [Slow heading drift](#slow-heading-drift));
  `--modes yawN` splits it into N time pieces.
- **Low light.** `features --clahe` equalizes the contrast before SIFT. On
  sequence_4_10 frames it raises the detections from 130–300 to 1,100–1,500
  per image. `--denoise 7` runs non-local means first (see [Low
  light](#low-light-features-after-denoising)).
- **Focal lengths.** `solve --est-focal` estimates each camera's focal
  length per time segment (`--focal-seg-sigma`, `--focal-sigma`,
  `--focal-mean-sigma`); leave it off on a moving platform. `loop-correct`
  takes the estimates from the states.
- **Keypoint sizes.** `solve --sigma-size 2.7 --sigma-size-pow 0.35` gives
  larger keypoints a larger pixel sigma. `features` stores the sizes; for a
  `features.npz` written before, `sizes --mav0 ... --work ...` (with the
  same `--clahe`/`--denoise`) recovers them into `kp_size.npz`.
- **Recommended.** The [table above](#with-the-focal-lengths-the-keypoint-sizes-and-denoising)
  uses `--est-focal --sigma-size 2.7 --sigma-size-pow 0.35` in both solves,
  `features --clahe --denoise 7` in low light, and no `--est-focal` on the
  moving platform.
- **Cost.** On this 4-core container, a 15-minute sequence takes about
  15 min for the features (2 processes; non-local means adds about as much)
  and 2–7 min for the matching (with numba). The solve uses
  `NUMBA_NUM_THREADS` threads for the reprojection terms and the BLAS
  threads for the banded factor: on the 20-minute sequence_2_11 the first
  solve takes 269 s on one thread and 181 s on four (the same result), the
  second solve with the revisits 483 s on four. `loops` takes 3–6 min with
  four processes and `loop-correct` a minute. sequence_3_17's solves need
  about 6.5 GB of memory.
