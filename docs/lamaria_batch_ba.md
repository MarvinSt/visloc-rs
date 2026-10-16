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
   6, 8, 12, 16, 24, 32, 48 and 64 steps later (0.5 to 32 s), within and
   across the two cameras, if their optical axes differ by less than 75°. A
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
   - each camera's IMU-camera rotation.

   The landmarks are eliminated by the Schur complement. Every landmark is
   observed within 64 keyframes, so the reduced system is banded and is
   factored by LAPACK's banded Cholesky. The two camera rotations form a
   dense border. A prior on the first pose fixes the gauge.

   Each iteration first drops the observations more than 10 px off, then
   the landmarks left with a single observation. A landmark's damping is its
   Hessian diagonal floored at 1 % of its largest entry. Without both, a
   landmark seen with little parallax could take a step of hundreds of
   metres, and every damped step of an iteration raised the cost.
6. **Output.** The VIO's per-frame trajectory, each frame moved by the
   correction of the surrounding keyframes (interpolated).

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
landmark damping floor (see [Method](#method)).

| Sequence | Track | Device | VIO | VIO + mapper | Batch BA, first version | Batch BA |
|---|---|---|---:|---:|---:|---:|
| sequence_1_19 | Short | 1WM09370081264 | 71.96 | 68.54 | **83.96** | 83.02 |
| sequence_2_11 | Medium | 1WM093701G1276 | 49.68 | 55.51 | 52.29 | **61.03** |
| sequence_3_17 | Long | 1WM093701G1276 | 50.06 | 46.68 | 49.59 | **50.14** |
| sequence_4_10 | Low light | 1WM093701G1276 | 39.00 | 36.65 | 44.28 | **45.79** |
| sequence_5_11 | Moving platform | 1WM093701G1276 | 43.14 | 49.84 | 55.70 | **59.06** |
| **Mean of the five tracks** | | | 50.77 | 51.44 | 57.16 | **59.81** |
| sequence_1_20 | Short | 1WM09370081264 | 38.10 | | 43.48 | 43.39 (**74.84** with `--modes yaw`, see [below](#slow-heading-drift)) |

The same runs against the pseudo-GT:

| Sequence | Metric | VIO | VIO + mapper | Batch BA |
|---|---|---:|---:|---:|
| sequence_1_19 | Pose recall at 1 m | 76.3 % | 60.6 % | **91.5 %** |
| sequence_1_19 | Sim(3) RMSE | 0.88 m | 0.96 m | **0.55 m** |
| sequence_2_11 | Pose recall at 1 m | 30.9 % | 39.4 % | **51.3 %** |
| sequence_2_11 | Sim(3) RMSE | 2.31 m | 1.92 m | **1.51 m** |
| sequence_3_17 | Pose recall at 1 m | **24.7 %** | 21.7 % | 16.3 % |
| sequence_3_17 | Sim(3) RMSE | 2.07 m | 2.48 m | **2.02 m** |
| sequence_4_10 | Pose recall at 1 m | 14.1 % | | **19.3 %** |
| sequence_4_10 | Sim(3) RMSE | 3.11 m | | **2.36 m** |
| the four above | RMSE of 60 s segments, each Sim(3)-aligned | 0.151 / 0.264 / 0.182 / 0.381 m | | **0.127 / 0.206 / 0.147 / 0.336 m** |
| sequence_1_20 | Pose recall at 1 m | 3.8 % | | 11.8 % (**100 %** with `--modes yaw`) |
| sequence_1_20 | Sim(3) RMSE | 3.03 m | | 2.41 m (**0.59 m** with `--modes yaw`) |

On sequence_5_11 (no pseudo-GT) the BA brings AA2976 from 0.94 m (VIO) to
0.07 m and the fitted scale drift (without PB2) from −5.4 % to −2.5 %. PB2,
the only CP before the funicular ride, stays 19 m off.

The batch BA is the most accurate locally on every sequence with pseudo-GT.
The slow drift that remains, fitted to the CP residuals as a heading and a
scale drift linear in time:

| Sequence | Heading drift | Scale drift |
|---|---:|---:|
| sequence_1_19 | −0.14° | −0.6 % |
| sequence_2_11 | −0.34° | +2.8 % |
| sequence_3_17 | +0.82° | +2.3 % |
| sequence_4_10 | −0.58° | +4.0 % |
| sequence_5_11 (without PB2) | −0.35° | −2.5 % |

The sequences of 1WM093701G1276 keep 2–4 % of scale drift, and their local
scale (Sim(3) fits in 60 s windows) varies by ±2–4 %, against ±1 % on
sequence_1_19. Their IMU factors also fit worse: the whitened position
residual per factor is 1.6–1.8, against 0.4–0.5 on 1WM09370081264.
These do not remove it (sequence_2_11, CP Score 61.03):

| Change | CP Score | Note |
|---|---:|---|
| Accelerometer y axis ×1.03 / ×0.97 | 60.40 / 56.48 | The IMU cost nearly doubles (20,600 → 39,100 / 36,700): the BA pins each axis' scale to well under 1 % |
| Accelerometer delayed by −2 / +2 ms | 61.14 / 61.14 | Final cost +1,100 / −90 |
| Accelerometer bias random walk ×10 | 60.49 | |
| IMU noise densities ×2 | 58.02 | |

A fit of the accelerometer against the pseudo-GT's second differences had
suggested a y scale of 0.97 on this device; the BA rules that out. The
camera exposure does not explain the local scale either: it correlates
weakly with it on sequence_2_11 (−0.33), and on sequence_4_10 the exposure
stays at 14 ms while the local scale varies by ±7 %.

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
the banded factor; this converges in about 20 iterations.

With the first version of the solver the revisits raised sequence_2_11 from
52.29 to 54.41. On sequence_1_20 the stage accepts 836 keyframe pairs (all
within 15 m in the pseudo-GT, 40 s apart at the median and 127 s at most),
but the BA with them falls from 43.39 to 38.97: the IMU cost grows from
17,000 to 30,000 and the heading drift from −4.6° to −5.8°. Some revisit
matches are probably wrong (repeated structures pass the essential matrix
and the 10° rotation check), so the stage stays optional.

### IMU-camera time offset

The BA reports the whitened IMU residual per factor. A sweep of the gyroscope
time offset on top of the per-device alignment (±2 ms, 150 s windows)
minimizes the total cost within 0.5 ms of the device table on both devices
(−0.4 ms on sequence_1_19, −0.5 ms on sequence_2_11). The time alignment
is therefore not what limits the sequences of 1WM093701G1276.

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

## Usage

```bash
pip install numpy scipy opencv-python gtsam numba
S=sequence_1_19
python scripts/lamaria_batch_ba.py features --mav0 data/$S/mav0 --vio vio/$S.tum --work work/$S
python scripts/lamaria_batch_ba.py match --work work/$S --vio vio/$S.tum --calib calib/$S.json
python scripts/lamaria_batch_ba.py loops --work work/$S --vio vio/$S.tum --calib calib/$S.json  # optional
python scripts/lamaria_batch_ba.py solve --work work/$S --vio vio/$S.tum --calib calib/$S.json \
  --imu data/$S/mav0/imu0/data.csv --out out/$S.tum [--loops]
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
- **Heading drift.** `--modes yaw` lets the solve correct a slow heading
  drift of its starting trajectory (see [Slow heading drift](#slow-heading-drift)).
- **Low light.** `features --clahe` equalizes the contrast before SIFT. On
  sequence_4_10 frames it raises the detections from 130–300 to 1,100–1,500
  per image.
- **Cost.** On this 4-core container, a 15-minute sequence takes about
  15 min for the features (2 processes) and 3 min for the matching (with
  numba). The solve takes 9–20 min on one thread; the longest, on the
  30-minute sequence_3_17.
