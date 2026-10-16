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

Official evaluator, one training sequence per track so far. The VIO and
mapper columns are the per-device-refined runs of
[`lamaria_drift_analysis.md`](lamaria_drift_analysis.md#with-the-online-mapper).
The batch BA starts from that VIO trajectory and uses the factory camera rig.
On sequence_4_10 the VIO is the low-light configuration (0.5 px KLT
threshold) and the BA's SIFT runs on CLAHE-equalized images. The mean uses
the BA without revisits.

| Sequence | Track | Device | VIO | VIO + mapper | Batch BA | + revisits |
|---|---|---|---:|---:|---:|---:|
| sequence_1_19 | Short | 1WM09370081264 | 71.96 | 68.54 | **83.96** | — |
| sequence_2_11 | Medium | 1WM093701G1276 | 49.68 | **55.51** | 52.29 | 54.41 |
| sequence_3_17 | Long | 1WM093701G1276 | **50.06** | 46.68 | 49.59 | — |
| sequence_4_10 | Low light | 1WM093701G1276 | 39.00 | 36.65 | **44.28** | — |
| sequence_5_11 | Moving platform | 1WM093701G1276 | 43.14 | 49.84 | **55.70** | — |
| **Mean** | | | 50.77 | 51.44 | **57.16** | |

The same runs against the pseudo-GT:

| Sequence | Metric | VIO | VIO + mapper | Batch BA |
|---|---|---:|---:|---:|
| sequence_1_19 | Pose recall at 1 m | 76.3 % | 60.6 % | **91.4 %** |
| sequence_1_19 | Sim(3) RMSE | 0.88 m | 0.96 m | **0.55 m** |
| sequence_2_11 | Pose recall at 1 m | 30.9 % | **39.4 %** | 31.1 % (38.3 % with revisits) |
| sequence_2_11 | Sim(3) RMSE | 2.31 m | **1.92 m** | 1.96 m |
| sequence_3_17 | Pose recall at 1 m | **24.7 %** | 21.7 % | 15.2 % |
| sequence_3_17 | Sim(3) RMSE | **2.07 m** | 2.48 m | 2.08 m |
| sequence_4_10 | Pose recall at 1 m | 14.1 % | | **18.7 %** |
| sequence_4_10 | Sim(3) RMSE | 3.11 m | | **2.53 m** |
| the four above | RMSE of 60 s segments, each Sim(3)-aligned | 0.151 / 0.264 / 0.182 / 0.381 m | | **0.127 / 0.221 / 0.148 / 0.340 m** |

On sequence_5_11 (no pseudo-GT) the BA improves 11 of its 14 CPs, for example AA2976
from 0.94 m to 0.17 m, and halves the fitted scale drift (−5.4 % → −2.9 %). PB2, the
only CP before the funicular ride, stays 19 m off.

The batch BA is the most accurate locally on every sequence with pseudo-GT. On
sequence_1_19 it also removes the slow drift: fitting a heading and a scale
drift to its CP residuals leaves −0.06° and −0.6 % over the CP span, and the
remaining CP errors do not follow a smooth drift. On the two sequences of
1WM093701G1276 it does not: sequence_2_11 keeps +3.0 % of scale drift and
sequence_3_17 +2.4 % with +0.8° of heading drift, as much as the VIO. Their
local scale (Sim(3) fits in 60 s windows) varies by ±2–4 % with and without
the BA, against ±1 % on sequence_1_19.

### What the gain on sequence_1_19 comes from

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

### IMU-camera time offset

The BA reports the whitened IMU residual per factor. A sweep of the gyroscope
time offset on top of the per-device alignment (±2 ms, 150 s windows)
minimizes the total cost within 0.5 ms of the device table on both devices
(−0.4 ms on sequence_1_19, −0.5 ms on sequence_2_11). The time alignment
is therefore not what limits the sequences of 1WM093701G1276.

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
- **Low light.** `features --clahe` equalizes the contrast before SIFT. On
  sequence_4_10 frames it raises the detections from 130–300 to 1,100–1,500
  per image.
- **Cost.** On this 4-core container, a 15-minute sequence takes about
  15 min for the features (2 processes) and 3 min for the matching (with
  numba). The solve takes 15–50 min on one thread.
