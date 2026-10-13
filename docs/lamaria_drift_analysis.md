# LaMAria: where the remaining error comes from

This page analyses the per-control-point (CP) errors that the current
submission setup leaves on the training sequences. The setup is the rectified
IMU, `gyro_bias_std` 1e-7, the multicam big-window config, the online mapper
and the 5° loop gate ([`lamaria_online_mapper.md`](lamaria_online_mapper.md)).
It started from the Moving-platform track, which had the largest gap to the
leaderboard (36.19 on sequence_5_11), and ended at a cause shared by all
tracks.

The Score uses the horizontal (2D) error of each CP after the evaluator's
robust Sim(3) alignment. Points fall from 20 below 5 cm to 0 at 10 m
(`lamaria/utils/metrics.py`). A CP at 1 m earns 12 points, at 2 m 8, and at
4 m about 5.

## Moving platform: sequence_5_11

### The CP errors

| Time | CP | Horizontal error | Points |
|---:|---|---:|---:|
| 12 s | PB2 | 20.81 m | 0.0 |
| 264 s | AA2918 | 3.81 m | 5.6 |
| 323 s | AA2846 | 2.59 m | 7.2 |
| 389 s | AA2976 | 2.86 m | 6.9 |
| 469 s | AA2917 | 0.59 m | 14.4 |
| 515 s | AA2916 | 1.32 m | 10.7 |
| 573 s | AA3739 | 2.98 m | 6.7 |
| 626 s | AA1708 | 3.40 m | 6.1 |
| 753 s | AA1726 | 4.20 m | 5.1 |
| 836 s | AA1143 | 2.67 m | 7.1 |
| 919 s | AA1144 | 2.16 m | 7.8 |
| 966 s | AA1133 | 1.04 m | 11.9 |
| 1052 s | AA4030 | 3.47 m | 6.0 |
| 1100 s | AA4031 | 3.61 m | 5.9 |

Five of the 14 CPs carry no height (their vertical variance is 1e18). Their
3D errors of 21–31 m are vertical only and do not count.

### The platform: a funicular ride

PB2 is the only CP before a ride on the Polybahn funicular, from 115 s to
190 s. The cameras see the cabin interior, the passengers and, through the
windows, the slope outside. The VIO estimates the ride at 133 m horizontally
and 30 m up. Shifting the part before the ride until PB2 fits gives 157 m and
38.6 m. The VIO therefore underestimates the ride by 15–22 %. Its speed
settles at 2.0 m/s after the acceleration phase, while integrating the IMU
alone over the acceleration phase reaches 2.5 m/s.

With the pre-ride part shifted onto PB2 (an upper bound for any fix of the
ride), the Score rises from 36.19 to 41.37.

The IMU cannot observe a constant cruise speed, so the cruise speed comes
from vision, which mixes the static cabin with the moving exterior. A
post-solve outlier filter (see [VIO outlier filter](#vio-outlier-filter))
does not change the ride: 133.7 m / 30.7 m at 3 px and 134.8 m / 31.0 m at
2 px.

### The other 13 CPs

The remaining CPs are 0.6–4.2 m off, and their residual vectors vary
smoothly along the route. A 2D similarity fit with a yaw drift and a scale
drift that are linear in time explains them (next section).

## Drift models fitted to the CP residuals

The fits take the triangulated CPs from the evaluator's output and fit them
to the surveyed positions with a 2D similarity, then add a yaw drift, then
a yaw and a scale drift, both linear in time (robust loss). Each drift adds
one parameter. PB2 is excluded on sequence_5_11.

| Sequence | Median, similarity | + yaw drift | + yaw + scale drift | Fitted yaw / scale drift over the CP span |
|---|---:|---:|---:|---|
| sequence_1_19 | 1.57 m | 0.76 m | **0.46 m** | −2.5° / −2.5 % |
| sequence_2_11 | 0.76 m | 0.90 m | 0.89 m | −1.2° / +0.4 % |
| sequence_3_17 | 1.85 m | 1.26 m | **0.92 m** | −2.2° / −2.1 % |
| sequence_4_10 | 2.47 m | 1.97 m | 2.28 m | +1.8° / +1.2 % |
| sequence_5_11 | 2.88 m | 2.52 m | **0.70 m** | −2.3° / −7.6 % |

On Short, Long and Moving, two drift parameters bring the median CP error
below 1 m, which would score about 12–15 points per CP instead of 5–8. A
slow drift in heading and scale is therefore the main error left on these
sequences, not a local failure. On sequence_5_11 the scale term dominates.
Medium is already accurate except for a few CPs, and Low light has errors
that a smooth drift does not explain.

### Scale

Sim(3) fits against the pseudo-GT in 60 s windows show the local scale
varying by ±2–5 % around the global value on every sequence with pseudo-GT.

The global scale grows with exposure:

| Sequence | Global Sim(3) scale (estimate → GT) |
|---|---:|
| sequence_1_19 | 0.960 |
| sequence_2_11 | 0.972 |
| sequence_3_17 | 0.987 |
| sequence_4_10 (low light, 14 ms exposure) | 0.919 |

Shifting the IMU by −3 ms and −6 ms on sequence_4_10 moves the scale to
0.957 and 1.015, so an IMU–camera time offset of about half the exposure
sets the global scale. Sim(3) absorbs a constant scale. The local
variations, however, do not change with the shift.

On sequence_5_11 the exposure (read from the `.vrs` image records) is
0.02 ms for most of the route after 240 s. Exposure changes therefore do not
explain its −7.6 % scale drift.

### Yaw

Against the pseudo-GT, the VIO's heading drifts in the same direction on all
four sequences:

| Sequence | Yaw at end | Rate |
|---|---:|---:|
| sequence_1_19 | −4.3° over 916 s | −0.28°/min |
| sequence_2_11 | −3.1° over 1185 s | −0.15°/min |
| sequence_3_17 | −6.9° over 1790 s | −0.23°/min |
| sequence_4_10 | −1.4° over 1418 s | −0.06°/min |

The mean, −0.18°/min, equals the vertical component of Earth's rotation at
Zurich (0.184°/min) in size and sign. Compensating it (see below) changes
sequence_5_11's VIO only slightly: Score 32.27 → 32.98, and the fitted yaw
drift goes from −3.18° to −2.87° instead of the 2.6° reduction the
hypothesis predicts. The body-frame projection of a vertical rate depends
only on roll and pitch. With the wearer's head roughly upright, the gyro bias
estimate absorbs most of it. The systematic heading drift therefore has
another cause, which is still open.

## Changes in the VIO (both opt-in, default off)

### VIO outlier filter

Upstream's `sqrt_keypoint_vio.cpp` leaves `filterOutliers` as a TODO, and
the Rust port follows it: a visual observation stays in every later solve,
bounded only by its Huber weight. `config.vio_filter_outliers: true` adds
the filter of upstream's older `keypoint_vio.cpp`.

- After each window solve, it removes the observations whose reprojection
  error exceeds `vio_outlier_threshold` (3 px).
- It removes landmarks left with fewer than four observations.
- It then re-solves for the three iterations after `vio_filter_iteration`.

The default trajectory is bit-identical. The filter has been measured only on
the ride above, where it changes nothing. A full-sequence comparison is
still open.

### Earth-rotation compensation

`config.vio_earth_rotation_latitude_deg` subtracts the vertical component
of Earth's rotation, `R_wb^T [0, 0, w_e sin(lat)]`, from every gyro sample.
`R_wb` is the newest window state. Measured on sequence_5_11 (VIO only):

| Run | Score | Fitted yaw drift |
|---|---:|---:|
| Baseline | 32.27 | −3.18° |
| Gyro corrected offline from the baseline's attitude | 32.98 | −2.87° |
| `config.vio_earth_rotation_latitude_deg: 47.376` (online) | 33.00 | −2.88° |

The online correction matches the offline one. A single sequence cannot
separate +0.7 from the Score's sensitivity to small input changes
([`lamaria_imu_rectification.md`](lamaria_imu_rectification.md#imu-time-offset)),
so it stays off.

## Open items

1. **Heading drift.** Its systematic sign suggests a deterministic cause,
   for example a temperature-driven gyro bias change that the tight bias
   random walk cannot follow, or a small gyro–camera misalignment.
2. **Local scale drift.** On sequence_5_11 it is the largest error term.
   The online mapper's joint VI-BA (`--joint-vi-ba-weight 10`) leaves the
   local scale of sequence_3_17 unchanged, so the IMU factors carry too
   little weight there.
3. **The funicular ride.** It is worth at most +5.2 on sequence_5_11. The
   test set's Moving sequences are 11–24 GB, against 10 GB for
   sequence_5_11, and may contain longer rides.
