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
| sequence_4_10 | 35.98 → 31.54 (**−4.43**) | 3.74 → 4.29 m |
| sequence_5_11 | 25.62 → **35.89** (+10.28) | n/a |

At 1e-8 the mapper's effect is erratic. On sequence_1_19 the 60-second
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

## Use in the submission driver

`scripts/run_lamaria_test_submission.py --slam-exe
<basalt_euroc_online_slam_demo>` runs this setup and submits
`trajectory_online.tum`. That file has one pose per frame, as the
submission format requires: for example, 23,749 lines including the header
on sequence_2_11. If a run writes only `trajectory_vio.tum`, the driver
submits that with a `WARN` line. `--vio-exe` alone keeps the VIO-only
behaviour.

## Open items

1. **Wrong-loop safeguard.** The 1e-8 failures show that a single accepted
   wrong loop can cost 5–10 points. A consistency check before merging
   would reduce the risk at 1e-7 as well, for example by rejecting merges
   whose correction contradicts the VIO's relative motion by more than a
   bound.
2. **Remaining yaw drift on Long.** On sequence_3_17 the mapper reduces the
   yaw drift only from −6.9° to −5.7°. Few places on that route are
   revisited.
3. **One sequence per track.** The same caveat as for the earlier studies
   applies.
