# visloc-rs vs kornia-slam

[kornia-slam](https://github.com/kornia/kornia-slam) (v0.1, commit `81b1b24`)
is a Rust visual-inertial SLAM system built on kornia-rs: ORB features,
ORB-SLAM3-style inertial initialisation, local visual-inertial BA, and opt-in
DBoW2 loop closing with pose-graph optimisation. Its IROS 2026 R4R paper
reports stereo-inertial EuRoC results. Its roadmap lists LaMAria /
egocentric evaluation as future work ([kornia-slam#78](https://github.com/kornia/kornia-slam/issues/78)).
This page compares the two systems where numbers exist.

## EuRoC (stereo-inertial, ATE RMSE, SE(3) alignment)

| Sequence | visloc-rs | kornia-slam | ORB-SLAM3 (kornia-slam paper) |
|---|---:|---:|---:|
| MH_01_easy | **0.0155 m** | 0.081 m | 0.037 m |
| MH_03_medium | **0.0256 m** | 0.066 m | 0.029 m† |
| MH_05_difficult | **0.057–0.063 m** | 0.188 m | 0.062 m† |

- **visloc-rs:** online VI-SLAM numbers from
  [`vi_slam_benchmarks.md`](vi_slam_benchmarks.md). MH_05 is 0.0633 m with the
  default frontend and 0.0570 m with `optical_flow_levels` 4.
- **kornia-slam:** the paper's Table I, the median of five runs. † The paper
  marks these ORB-SLAM3 values as partial trajectories below its 95 %
  coverage gate.
- **Caveat:** the two sets of numbers were measured on different hosts and
  were not re-run side by side here, because the EuRoC host rate-limits
  automated downloads (HTTP 429). Both use rigid SE(3) alignment without
  scale correction.

## LaMAria (measured side by side)

Both systems were run on LaMAria training `sequence_1_19` (Short, 18,352
frames, 15 min). The input was the ASL export with Aria's factory-rectified IMU
([`lamaria_imu_rectification.md`](lamaria_imu_rectification.md)). Scoring used
the official `cvg/lamaria` evaluator, unmodified.

| System | Mode | Frames tracked | Score | CP@1m | Pose R@5m |
|---|---|---:|---:|---:|---:|
| **visloc-rs** | multicam VIO (`gyro_bias_std` 1e-7) | all | **52.26** | 28.6 % | 100 % |
| **visloc-rs** | + online mapper | all | **53.92** | 28.6 % | 100 % |
| kornia-slam | mono + IMU + loop closing (defaults) | 7 / 18,352 | — (Sim(3) alignment fails) | — | — |

### How kornia-slam was run

- **Mode:** monocular-inertial with loop closing
  (`sensors: (cameras: Mono, imu: true)`,
  `pipeline: OrbSlam((loop_closing: Enabled(...)))`), with the ORB-SLAM3
  vocabulary.
  - Its stereo mode expects a rectifiable pair. Aria's two SLAM cameras
    diverge by ~75°, so stereo was not used.
  - Otherwise the configuration was left at its defaults.
- **Input adaptation, none of which touches the estimator:**
  - `cam0/sensor.yaml` was generated from LaMAria's pinhole calibration:
    pinhole model, zero distortion, `T_BS` from `T_b_s`.
  - The RGB PNGs were converted to single-channel PNGs. All three channels
    are identical, so the conversion is lossless. kornia-slam's EuRoC reader
    rejects 3-channel PNGs.
- **Local patch:** without ground truth, the app writes no trajectory. A
  14-line, env-gated patch (`KORNIA_TRAJ_OUT`) dumps each frame's
  `pose_world_to_cam` after `SlamSystem::process`. The patch is local to this
  comparison and was not upstreamed.
- **Build:** `cargo build --release -p kornia-slam-app --no-default-features`
  (without Rerun); run with `--no-tui`. It took 621 s for the sequence.

### Why kornia-slam does not track

A `--debug` run shows the monocular bootstrap rejecting every candidate pair as
`LowMatches`, even between consecutive frames 50 ms apart. The two-view
initialiser requires ≥ 100 ORB matches under a 0.6 nearest-neighbour ratio
and a 50-bit Hamming threshold (`initialization/two_view.rs`), and Aria's wide
(~116° HFOV), low-resolution (780×584) SLAM-camera images do not meet it. The
system bootstraps once, at frame 3,564, keeps 134 map points with two
observations each, and loses track within 7 frames. With only those frames,
the evaluator's robust Sim(3) to the control points fails, so the sequence has
no score. Submitted as is, it would count as 0.

## Ideas worth taking from kornia-slam

- **Loop verification.** Before a loop becomes a constraint, kornia-slam
  requires PnP RANSAC against the candidate's landmarks, reprojection and
  image-coverage tests, and consistency across adjacent query keyframes
  ("episode consistency"). The visloc-rs online mapper accepts a loop match
  after two-view RANSAC plus a rotation gate. That gate compared IMU-frame
  and camera-frame rotations until commit `70e8c32`, which fixed it. A
  multi-keyframe consistency check is the natural next safeguard; a single
  wrong loop costs 5–10 Score on LaMAria
  ([`lamaria_online_mapper.md`](lamaria_online_mapper.md)).
- **Gravity-preserving 4-DoF pose-graph updates after inertial
  initialisation.** The visloc-rs mapper's pose factors are plain quadratic
  6-DoF.
