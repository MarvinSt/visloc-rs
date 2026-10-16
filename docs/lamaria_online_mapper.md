# LaMAria: online mapper on the rectified IMU

The online mapper (`examples/basalt_euroc_online_slam_demo.rs`) runs the same
VIO as `basalt_euroc_vio_demo` and, in parallel, a keyframe mapper with loop
detection, map merging and periodic global optimisation. It writes the
mapper's corrections back onto every VIO frame (`trajectory_online.tum`).
Earlier, on the raw IMU with `gyro_bias_std` 1e-6, it barely helped:
sequence_2_11 went 37.72 → 38.10. With the factory-rectified IMU
([`lamaria_imu_rectification.md`](lamaria_imu_rectification.md)) it is the
largest single gain measured on LaMAria so far.

## Protocol

- **Sequences:** one training sequence per track, as in the rectification
  study.
- **Setup:** rectified IMU, multicam big-window config, variant-A noise.
  `gyro_bias_std` is 1e-7 (the driver default) and, for comparison, 1e-8.
- **Run:** `basalt_euroc_online_slam_demo --threads 2 --no-urgent-keyframes`,
  as fast as possible. The demo's `trajectory_vio.tum` is bit-identical to
  the VIO-only runs. For example, sequence_3_17 scores 35.7575 in both.
- **Scoring:** the official `cvg/lamaria` evaluator. ATE is the Sim(3)
  RMSE against the pseudo-dense GT, which is not published for
  sequence_5_11.
- **Mapper:** the default (pose-graph) mapper. The joint VI-BA option was
  tried separately (see below).

## Results (`gyro_bias_std` 1e-7, the driver default)

| Sequence | Track | Score VIO → +mapper | CP@1m | Pose R@5m | ATE |
|---|---|---|---|---|---|
| sequence_1_19 | Short | 52.26 → **53.92** (+1.66) | 28.6 → 28.6 % | 100 → 100 % | 2.17 → **1.81 m** |
| sequence_2_11 | Medium | 38.36 → **51.41** (+13.05) | 5.6 → **50.0 %** | 89.8 → **100 %** | 3.21 → **2.41 m** |
| sequence_3_17 | Long | 35.76 → **40.58** (+4.82) | 7.4 → **11.1 %** | 86.2 → **89.7 %** | 4.63 → **3.47 m** |
| sequence_4_10 | Low light | 32.42 → **36.18** (+3.77) | 6.2 → 6.2 % | 86.2 → **88.9 %** | 3.58 → **3.46 m** |
| sequence_5_11 | Moving platform | 32.27 → **36.19** (+3.93) | 0.0 → **7.1 %** | n/a | n/a |
| **Mean** | | 38.21 → **43.66 (+5.44)** | | | |

The mapper improves the Score on all five tracks, and ATE on all four
sequences with pseudo-GT. On sequence_2_11 it removes the remaining yaw
drift: −2.1° at the end of the sequence becomes +0.4° (1e-8 run, see below).

## Interaction with `gyro_bias_std` 1e-8

| Sequence | Score VIO → +mapper at 1e-8 | ATE |
|---|---|---|
| sequence_1_19 | 51.30 → 41.25 (**−10.06**) | 2.15 → 3.00 m |
| sequence_2_11 | 38.23 → **54.04** (+15.81) | 3.22 → **2.20 m** |
| sequence_3_17 | 38.20 → **43.64** (+5.44) | 4.18 → **2.99 m** |
| sequence_4_10 | 35.98 → 31.54 (**−4.43**) | 3.74 → 4.29 m |
| sequence_5_11 | 25.62 → **35.89** (+10.28) | n/a |
| **Mean** | 37.87 → 41.27 (+3.41) | |

At 1e-8 the mapper's mean (41.27) stays below the 1e-7 mapper's mean
(43.66), even though 1e-8 + mapper is the best result on Medium and Long.
Its effect is also erratic. On sequence_1_19 the 60-second
segment fits stay accurate while the whole trajectory bends, which is the
signature of an accepted wrong loop or merge. At 1e-7 the effect is
positive on every track, so **1e-7 stays the default**. 1e-8 also loses
0.35 Score on average without the mapper
([`lamaria_imu_rectification.md`](lamaria_imu_rectification.md#re-tuning-gyro_bias_std)).

Joint VI-BA (`--joint-vi-ba-weight 10`) was measured once, on
sequence_3_17 at 1e-7. It scored 39.21 (ATE 3.79 m) against 40.58
(3.47 m) for the default mapper, so it stays off. This matches the EuRoC
screen in
[`vi_slam_global_consistency_plan.md`](vi_slam_global_consistency_plan.md).

## Cost

With two runs sharing 4 cores, the online demo's total wall time is
1.06–1.19× its own VIO wall time (`total_wall` / `vio_wall` in the
`pacing=` line). The longest single global optimisation took 6–25 min per
sequence. Compared with separate VIO-only runs, the overhead is roughly
1.15–1.4×; those runs were under different CPU contention, so this range
is approximate. Peak RSS was 1.4–2.4 GB. The offline mapper is not an
alternative at this window size. Its marg-data dump is ~74 MB per packet
with the 10/30 window, which would be hundreds of GB on a Long sequence,
and it optimises poses only (no IMU factors).

## Loop rotation gate

A loop match enters the global BA as shared landmarks. Before that, the mapper
checks that the two-view RANSAC rotation agrees with the map
(`loop_match_max_rotation_error_deg`, default 30°). Per-loop logging
(`BASALT_ONLINE_MAPPER_LOOP_LOG=1`) on sequence_1_19 showed two things.

- **No false places.** Every loop candidate pairs keyframes within 21 m of
  each other in the pseudo-GT, and the map's distance matched it. A position
  gate would have rejected nothing.
- **Mixed frames.** The rotation check compared the map's **IMU-frame**
  relative rotation with the RANSAC **camera-frame** rotation. On Aria the
  cameras are tilted ~70° from the IMU, so the "error" scaled with the
  viewpoint change itself (0.5–2× the relative rotation between the
  keyframes). The gate rejected genuine loops with a large turn and let
  inconsistent ones through. On EuRoC the camera is rotated mostly about the
  vertical axis, so the mix-up stayed small there.

Commit `70e8c32` compares both in camera frames. It applies the same
correction to the optional loop-closure factors, which converted a
camera-frame measurement straight into an IMU-pose factor.

With the fix, the median rotation disagreement of loop candidates is
1.2–1.4°. At 30°, almost every candidate now passes: about 258 loops on
sequence_1_19 against about 55 before, and the score fell 53.92 → 52.88. A
5° gate drops only the tail of broken two-view estimates (1–55 per
sequence):

| Sequence | Track | Score before → fix + 5° gate | ATE |
|---|---|---|---|
| sequence_1_19 | Short | 53.92 → **54.39** | 1.81 → **1.75 m** |
| sequence_2_11 | Medium | 51.41 → **56.35** | 2.41 → **1.96 m** |
| sequence_3_17 | Long | 40.58 → **41.61** | 3.47 → **3.25 m** |
| sequence_4_10 | Low light | 36.18 → 36.20 | 3.46 → 3.45 m |
| sequence_5_11 | Moving platform | 36.19 → 36.19 | n/a |
| **Mean** | | 43.66 → **44.95 (+1.29)** | |

No track gets worse. Reruns of the old binary under a different CPU load
moved the mapper score by at most 0.05, so the gains on Short, Medium and
Long are not timing noise. The fix does not repair the 1e-8 failure on
sequence_1_19, which scored 41.11 with the fix against 41.25 before, so that
failure has another cause.

The submission driver passes `--loop-match-max-rot-error 5` with
`--slam-exe`. The library default stays 30° until EuRoC is re-checked with
the fix.

## With the per-device IMU-camera refinement

The per-device time and rotation refinement
([`lamaria_drift_analysis.md`](lamaria_drift_analysis.md#with-the-online-mapper))
raises the VIO mean from 38.21 to 49.87, and the mapper mean only from 44.95
to 50.44. The mapper still wins on Medium (+5.83 over the VIO) and Moving
platform (+6.70), but now loses on Short (−3.42), Long (−3.38) and Low light
(−2.87). Locally it remains more accurate than the VIO; it loses on the
long-range shape.

## Use in the submission driver

`scripts/run_lamaria_test_submission.py --slam-exe
<basalt_euroc_online_slam_demo>` runs this setup and submits
`trajectory_online.tum`. That file has one pose per frame, as the
submission format requires: for example, 23,749 lines including the header
on sequence_2_11. If a run writes only `trajectory_vio.tum`, the driver
submits that with a `WARN` line. `--vio-exe` alone keeps the VIO-only
behaviour.

## Open items

1. **The 1e-8 failure.** It is not caused by a false place or by the
   rotation gate (see [Loop rotation gate](#loop-rotation-gate)). A likely
   suspect is over-confident VIO relative-pose priors at 1e-8, which are
   plain quadratic factors in the mapper's BA.
2. **EuRoC with the fixed gate.** The camera-frame fix changes which loops
   pass on every dataset, so the EuRoC online results should be re-measured
   before the library default changes.
3. **Remaining yaw drift on Long.** On sequence_3_17 the mapper reduces the
   yaw drift only from −6.9° to −5.7°. Few places on that route are
   revisited.
4. **One sequence per track.** The same caveat as for the earlier studies
   applies.
