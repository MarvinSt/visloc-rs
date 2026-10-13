# Changelog

All notable changes to `visloc-rs` will be documented here.

## Unreleased

### Fixed

- **`colmap_incremental_mapper` replays a `gsplat_photos`/`gsplat_euroc`
  export with the options it ran with.** Those pipelines use COLMAP's
  single-camera `abs_pose_min_num_inliers=30`, but a replay of their
  `port/` directory used the port's rig-control default of 8 unless
  `VISLOC_ABS_POSE_MIN_INLIERS=30` was set. That registered images the
  pipeline rejects; ETH3D meadow, for one, replayed at 15/15 images with a
  26.6 cm centre RMSE instead of 14/15 at 5.9 cm. The export now writes
  `port/mapper_options.txt` (the options applied, including `--mapper-opt`
  overrides), and the replay applies it, so a replay reproduces the
  pipeline's model byte for byte. Earlier replay-based comparisons in this
  log ran with 8 on both sides. `MapperOptions::apply_override` now holds
  the `key=value` parser that `--mapper-opt` uses.

- **PnP RANSAC no longer throws away its best pose.** After the search,
  `PnPRansac` refit the pose estimator on all the inliers and refined
  from that refit alone. With P3P the refit solves from three of the
  inliers and can land far from the search's winner, and the refinement
  then started from the bad pose: on the Courthouse replay 50 image
  registrations failed this way (around 700 correspondences, 3-6 inliers
  left), leaving 544 of 553 images registered. It now starts from the
  better of the winner and the refit. Courthouse registers all 553 with
  no failed attempt; the ETH3D courtyard, pipes, kicker and terrains
  replays are unchanged (same images, centre RMSE within 0.07 cm).

- **`gsplat_photos` GPU matching saturated VLFeat-compatible descriptors.**
  Those descriptors come back already in COLMAP's byte scale (norm about
  512), and the default u8 GPU bank quantises with `round(512 x)` again,
  so every non-zero value clipped to 255 and the descriptors were nearly
  binary. On SmallCity (GPU matching) this cut the verified pairs by 12%
  and the 3D points by 70% (292k instead of 958k), and pass 4 drifted
  (own-fit median 0.62 m instead of 0.13 m). They are now divided by 512
  before matching, which is exact: the bank recovers the same bytes and
  CPU matching is bit-identical. CPU-matched runs (`--cpu`, the ETH3D
  numbers above) were not affected.
- **`gsplat_photos` SIFT finds matches on high-resolution photos.** It now
  uses COLMAP's VLFeat detector (subpixel localization) and descriptor, and
  feeds SIFT grey values in [0, 1] as `GrayImage` documents (it passed
  0–255, which made the contrast threshold 255× too permissive). The default
  detector reports integer pixel positions. On one ETH3D courtyard pair with
  1,749 points in common it found 37 matches; the new path finds 721. ETH3D
  courtyard (1,600 px) registers 38/38 photos instead of 15, and pipes 11/14
  instead of 2. The VLFeat path runs on the CPU (matching stays on the GPU);
  `--sift-opt vlfeat_compatible_detector=0 --sift-opt
  vlfeat_compatible_descriptor=0 --sift-opt vlfeat_bilinear_orientations=0`
  restores GPU extraction.
- **`gsplat_photos` no longer drops photos a pixel off the common size.**
  Photos whose scaled size is within 1% of the most common one are resized
  to it instead of being dropped. ETH3D courtyard's undistorted images come
  in four sizes a few pixels apart, and 14 of its 38 photos were lost,
  splitting the scene into fragments.
- **`gsplat_photos` panicked on large photo sets** (wgpu: descriptor bank
  over `max_storage_buffer_binding_size`, e.g. 1,940 photos x 4,000 SIFT);
  it now matches in blocks. The trainer's logged L1 no longer rounds to 0
  on large images (2.8 MP), gradients were unaffected.
- **README hero splat filter read the wrong PLY columns.** Its
  large-and-transparent rule used SH coefficients as opacity and scale; the
  columns are now looked up by name.
- **`config.optical_flow_imu_seed_rotation` can be set from a Basalt config
  JSON.** The adapter read the key, but `BasaltConfig::from_json` rejected it
  as unknown, so only code could enable it.
- **Bundle adjustment uses the camera's full lens model.** Pose/structure BA
  (serial, parallel, dense-QR assembly and the GNC residual vector) now
  linearises the same lens that `Camera::project` measures, with analytic
  Jacobians for OPENCV (k1, k2, p1, p2), FULL_OPENCV, OPENCV_FISHEYE,
  SIMPLE_RADIAL_FISHEYE, RADIAL_FISHEYE, FOV and Double Sphere. These cameras
  were previously rejected with `UnsupportedCameraModel`; SIMPLE_RADIAL and
  RADIAL are now accepted too. Pipelines that used to skip BA silently on
  such a camera (they swallow the error) now run it. Unknown models stay
  rejected, and stereo observations with any camera other than PINHOLE /
  SIMPLE_PINHOLE now return `UnsupportedCameraModel` instead of being
  modelled as a pinhole. Distortion-free and radial-only cameras are
  bit-for-bit unchanged (pinned by tests). Not yet covered: rig observations
  still project with a pinhole, and the matrix-free / GPU backends still
  decline distorted cameras and fall back to the regular solver.

### Added

- **`visloc_web`: run the photo pipeline from a web browser.**
  `cargo run --release --example visloc_web` serves a page on
  127.0.0.1:8080. From it you can:
  - pick a photo folder in a folder browser;
  - choose SfM only, splats, or splats and a mesh, plus the main
    `gsplat_photos` options;
  - start a run, follow a progress bar parsed from the run's output, and
    see the log without the mapper's per-step diagnostics;
  - cancel a run;
  - turn the reconstruction in an in-page 3D viewer (sparse points in
    their colours, cameras as frustums);
  - download `scene.ply`, `mesh.ply`, the COLMAP text model and the full
    log;
  - open an existing output folder just to view it.

  Each run is a `gsplat_photos` child process, one at a time. The server
  uses only the standard library, so no new dependencies. Downloads are
  limited to the run's output folder.

- **The COLMAP-port BA refines rig calibrations
  (`ba_refine_sensor_from_rig`).** The non-reference sensors'
  `sensor_from_rig` poses of the observed rigs become 6-DoF nodes of the
  reduced system (stepped as `sensor_from_rig ∘ exp(ξ)`), coupled directly
  to the frames observing through them and, through the points, to
  everything else, on both solve paths. Local BA refines a rig's sensors
  only while every registered frame of it is in the window
  (`incremental_mapper.cc:1036-1065`). Off by default
  (`--mapper-opt ba_refine_sensor_from_rig=1`). A synthetic 3-sensor rig
  is recovered from a perturbed calibration on both paths. SmallCity's
  six cameras are not rigid enough for it: its reference poses put
  cameras 3 and 5 about 1° (median, up to 15° in turns) off any fixed
  calibration, ~8 px at its focal length, so a rig model would fight the
  data there.

- **The COLMAP-port mapper self-calibrates the focal length and principal
  point in its bundle adjustments, and `gsplat_photos` turns it on.**
  COLMAP's `ba_refine_focal_length` / `ba_refine_principal_point`
  (`--mapper-opt ba_refine_focal_length=1` etc., recorded in
  `mapper_options.txt`) add the camera to the native BA as one more
  6-dim node of the reduced system (`[focal, cx, cy]` plus three fixed
  slots), so the direct and the iterative (PCG, 1000+ free frames) solves
  take it like a frame; forcing every ETH3D solve onto the iterative path
  reproduces the direct solves' errors to 0.002 cm. One focal step moves
  `fx` and `fy` together (square pixels). Local BA refines a camera only
  while every registered image of it is in the window, as COLMAP does.
  Off by default in the mapper; `gsplat_photos` enables it unless
  `--no-refine-intrinsics`. On the 13 ETH3D DSLR scenes the centred
  principal point prior is ~4 px off, and fixed intrinsics had inflated
  the camera-centre errors 2-5x. Replaying the 8192-detection exports
  with the refinement on (centre RMSE, cm; COLMAP 4.2 from the same prior
  with focal and principal point refined):

  | scene | fixed | refined in the mapper | COLMAP refined |
  |---|---|---|---|
  | facade | 6.688 | **1.673** | |
  | courtyard | 4.431 | **1.420** | 1.680 |
  | electro | 3.506 | **0.523** | |
  | delivery_area | 2.354 | 0.502 | 0.464 |
  | kicker | 2.075 | 0.744 | 0.538 |
  | terrace | 2.049 | **0.380** | 0.550 |
  | office | 0.997 | **0.565** | 0.800 (20/26) |
  | relief | 0.855 | 0.429 | 0.281 |
  | relief_2 | 0.730 | 0.228 | 0.201 |
  | playground | 0.628 | **0.386** | 123.7 |
  | terrains | 0.526 | **0.244** | 0.314 |
  | pipes | 0.699 | **0.326** | 0.392 |
  | meadow | 5.839 | 5.732 | 5.312 |

- **COLMAP 4's structure-less registration fallback in the COLMAP-port
  mapper.** When no candidate image registers from 2D-3D correspondences,
  the mapper now retries the candidates from 2D-2D correspondences to the
  registered images. The registered cameras form one generalized camera
  in world coordinates. GR6P RANSAC between it and the new image gives
  the pose with metric scale. The inlier matches then continue existing
  tracks or triangulate new ones, and a Cauchy-loss BA refines the pose.
  `MapperOptions::structure_less_registration_fallback` turns it on.
  `gsplat_photos` and `gsplat_euroc` enable it, and the port's own controls
  keep it off. Results with it on, from the same features:
  - ETH3D pipes registers 14/14 images instead of 11/14, as COLMAP 4.2's
    mapper does on these matches (COLMAP also registers 11/14 with the
    fallback off).
  - The other 12 ETH3D DSLR training scenes are unchanged.
  - The Courthouse replay with it off is byte-identical.

  Deviations from COLMAP 4: GR6P only, with no 5+1 solver, grouped
  sampler or LO step; a bounded number of trials; and an angular epipolar
  threshold in place of the tangent Sampson error.

- **`gsplat_photos --mask-black N` trains around blacked-out pixels.**
  Near-black regions (people and cars some datasets black out) get alpha 0
  in the training images; `TrainConfig::alpha_mask` leaves pixels with alpha
  below 128 out of the L1 and SSIM losses (no gradient, no window centre),
  and the held-out PSNR skips them (`metrics::psnr_masked`).
- **README hero on a city block: Hierarchical 3DGS SmallCity.** 5,822
  photos, 5,583 registered with 957,418 points, held-out PSNR 24.09 (masked),
  4.7M gaussians and a 5.9M-triangle mesh from one `gsplat_photos` command
  in 2.5 h on a Google Colab A100 (5 h before this cycle's speed-ups;
  earlier in this cycle: Tanks and Temples Courthouse, 553 frames). The orbit
  shows the roads: training leaves the dataset's blacked-out people and cars
  out of the loss, and the hero raises the street splats' opacity
  (`--ground-radius`, `--ground-min-alpha`). `scripts/colab/hero_pipeline.py` runs the whole job on a
  Colab GPU (Tanks and Temples, Mill-19 and Hierarchical 3DGS scenes; from
  the terminal with the Colab CLI or from
  `scripts/colab/readme_hero_courthouse.ipynb`), adding the NVIDIA Vulkan
  user-space driver when the VM lacks it. `scripts/make_readme_hero.py` is
  scene-agnostic (intrinsics from `cameras.txt`, margins relative to the
  camera ring, nadir-aware up axis, orbit / filter flags, Linux + EGL).
- **`gsplat_photos` for large photo sets.** `--retrieval K` (VLAD top-K pairs
  past `--exhaustive-max`), `--exhaustive-max`, `--window`,
  `--max-keypoints`, `--sift-opt key=value` (wide-baseline `affine` /
  `domain_size_pooling` run on the CPU) and `--refine-intrinsics-max`;
  GPU matching splits the descriptor bank into blocks past the ~2 GiB
  binding limit.
- **Faster mapper and BA on large models.** The COLMAP-port mapper filters
  points and completes / merges tracks in parallel (byte-identical
  output), builds BA problems with less overhead, and factors reduced
  camera systems of 64+ frames in a fill-reducing order (nested
  dissection / RCM). The joint pose + intrinsics BA gained a block-sparse,
  parallel solve (`LinearSolver::Sparse`, used by `gsplat_photos`).

- **Multi-camera Basalt VIO for divergent rigs (opt-in, aimed at LaMAria /
  Project Aria).** The VIO used only cam0 for new keypoints, landmark hosting
  and keyframe decisions, and seeded the cam0→cam1 stereo KLT at the same
  pixel, which on Aria's ~75°-divergent cameras found almost no matches. New
  optional keys: `optical_flow_matching_guess_type = "REPROJ_FIX_DEPTH"` with
  `optical_flow_matching_default_depth` (reprojection seed plus a predicted
  patch warp), `optical_flow_detect_all_cameras` (cam1 keypoint
  replenishment), `vio_landmarks_all_cameras` (cam1-hosted landmarks for
  tracks cam0 never sees) and `vio_kf_connectivity_all_cameras`. LaMAria
  variant `configs/basalt/variants/lamaria/euroc_config_big_window_multicam.json`.
  Synthetic Aria-like rig: RMS error 7.1 → 3.3 mm on a textured room, and
  23.5 → 5.1 mm when cam0 faces a blank wall for 3 s. Not yet measured on
  LaMAria; see `docs/lamaria_multicam.md`. Defaults are unchanged.
- **Joint GNSS + visual-odometry fusion** (`visloc_slam::gnss_fusion`,
  `visloc_slam::gnss_pose_graph`). VO relative-pose factors and GNSS
  position factors are optimised together (full batch or a causal sliding
  window) with an estimated GNSS-ENU-to-map alignment (yaw-only or full
  rotation plus translation; scale and per-frame scale drift for monocular
  VO), antenna lever arm, per-fix covariance, GNC outlier rejection with a
  χ² gate, and dropout bridging. Analytic Jacobians with finite-difference
  tests; RANSAC alignment bootstrap. GNSS fixes are matched to frame times by
  `visloc_fusion::interpolate_gnss_fix` / `gnss_fix_covariance` (plus
  `MeasurementBuffer::as_slice`). New `gnss_vo_fusion_demo` example and
  `scripts/check_gnss_fusion_demo_outputs.sh`. Synthetic ~1 km loop: ATE
  5.25 m (VO only) → 0.90 m fused (metric); 3.22 m (Sim(3)-aligned VO) →
  0.37 m (monocular). Loosely coupled: receiver ENU positions only, no raw
  pseudoranges and no IMU. See `docs/gnss_fusion.md`. `visloc-slam` now
  depends directly on `visloc-fusion` (it already did through `visloc-io`).
- **Monocular-inertial VIO on the Basalt port (experimental).**
  `basalt_euroc_vio_demo --mono` replays cam0 + IMU with the stereo
  calibration reduced to camera 0. The estimator needed no change: it already
  triangulates new keyframe landmarks across frames, initialises from IMU
  gravity alignment, and its KLT frontend treats cam1 as optional; the gap was
  the EuRoC reader and rig construction. New `EurocSensorDataset::open_monocular`,
  `BasaltCalibration::retain_cameras` and
  `visloc_basalt::vio_estimator_from_calibration`. The stereo-inertial path is
  unchanged. Synthetic tests (including rendered images through the real KLT
  frontend) recover metric scale within 0.3% at 5–9 mm RMS; EuRoC is still to
  be measured, and position drifts with IMU bias while the platform is static
  before the first motion. See `docs/mono_inertial_vio.md`.
- **Python bindings (`bindings/python`, package `visloc`).** pyo3/maturin
  extension with NumPy interop: `Camera` (COLMAP models + Double Sphere),
  `Pose`/`SE3`, COLMAP text/binary `Reconstruction` read/write, `localize` and
  `estimate_pose_pnp_ransac`, and `evaluate_ate` / `evaluate_rpe` /
  `umeyama_alignment`. Ships `.pyi` stubs and pytest tests; built and tested
  by a new `python-bindings` CI job. The crate is excluded from the root
  workspace (own `Cargo.lock`), so pyo3/numpy do not enter the Rust gates.
- **`visloc_io::colmap::write_colmap_binary_model`** for a `VisualMap`.
- **ROS 2 nodes (`ros2/visloc-ros2`)** over pure-Rust DDS (`ros2-client` /
  RustDDS; no ROS install needed to build). `visloc_vio_node` runs the Basalt
  stereo-inertial VIO on two `sensor_msgs/Image` topics plus
  `sensor_msgs/Imu` and publishes `nav_msgs/Odometry`,
  `geometry_msgs/PoseStamped`, `nav_msgs/Path` and `/tf` (VIO only; the
  online mapper is not attached). `visloc_localize_node` localizes
  `sensor_msgs/Image` frames against a COLMAP map plus landmark descriptors
  and publishes `PoseWithCovarianceStamped`, an inlier count and
  diagnostics. The crate is excluded from the root workspace, has its own
  lockfile and a dedicated CI job, and is tested end to end over RTPS
  (RustDDS to RustDDS), not yet against a ROS 2 install.
- **`Camera::project_with_point_jacobian`** (visloc-core): the pixel from
  `Camera::project` plus its analytic 2×3 Jacobian with respect to the
  camera-frame point, for every lens model.
- **`BaConfig::refine_tangential_distortion` / `--refine-tangential-distortion`**
  (unordered SfM demo): opt-in self-calibration of p1/p2 on top of
  `refine_distortion`. A PINHOLE camera is promoted to OPENCV
  `[fx, fy, cx, cy, k1, k2, p1, p2]` starting from p1 = p2 = 0.
  `refine_intrinsics` now also refines OPENCV cameras (their lens terms stay
  fixed unless `refine_distortion` is set). Default off.

### Changed

- **COLMAP-port BA: cheaper gradient stop and PCG product.**
  - **Gradient stop.** After an accepted step, nearly every solve stops
    on the gradient tolerance at the top of the next iteration. That
    check reads only `Jᵀr`, but it used to come after a full
    re-linearization. A gradient-only pass now decides it first. It
    runs the same code, so the sums and their order are the same. The
    `JᵀJ` blocks are built only when the solve goes on, and not at all
    after the last iteration. Results are bit-identical: ETH3D outputs
    match on all 13 scenes, and the mapper is about 6% faster there
    (37.5 s → 35.3 s). `BA_PHASES` reports the passes as `gradient_ms`.
  - **PCG product.** The edge-list PCG of large global BAs (1,000+
    free frames) computes each off-diagonal block's two products
    (`B x_a` and `Bᵀ x_b`) in one fused pass over the block. It no
    longer converts every block to an f64 matrix first. Each iteration
    is about 25% faster on the 4-thread `large_global_ba` bench.
  - **SmallCity measurement** (#276–#278, before and after on one pod):
    - Mapper time 27:53 → 27:01. Per step, elimination is 3.8× and
      linearization 1.9× faster.
    - The rounding changes moved the reconstruction to a different
      path. One 2,374-frame global BA then ran 25 LM steps instead of
      one, and PCG iterations doubled, eating most of the gain.
    - The median camera-centre error went from 0.640 m to 0.210 m
      (p90 2.19 m → 0.75 m). The previous binary reproduced 0.640 m
      exactly, so the mapper is deterministic and the change comes from
      the code. A single run cannot say whether the gain is systematic.

- **COLMAP-port mapper: fewer allocations and cheaper scoring.** All
  three changes are bit-identical; ETH3D outputs match the previous
  version on all 13 scenes.
  - A linearization keeps every point's `Jpose^T Jpoint` blocks in one
    array instead of one vector per point (hundreds of thousands of
    allocations per global BA), and the LM loop reuses the previous
    array. One linearization of a 3M-observation problem went from
    190 ms to 132 ms; the `large_global_ba` solve from about 8 s to
    6.5 s.
  - PnP RANSAC scores each hypothesis by inlier count and mean error
    only. It used to sort every inlier error per hypothesis for a median
    and maximum that only the final pose reports.
  - Track completion and merging look up their claimed and visited sets
    with the crate's FxHash-style hasher instead of SipHash, about 15%
    faster.

- **Less per-image overhead in the COLMAP-port mapper on large models.**
  The SmallCity replay also spent 435 s in local refinement outside the
  solver core: point filtering 107 s, choosing the local window 63 s,
  and linearizations in every solve. The filter and window changes are
  bit-identical. The linearization rounds differently: its sums are
  grouped by point shard, and the pose Jacobian is formed as
  `[J_point | -J_point [p]x]`. It stays deterministic at any thread count
  (new test).
  - **Linearization** runs in a fixed number of point shards. Each shard
    adds its per-frame blocks into its own accumulators, and the shards
    are summed in order. The old code wrote 336 bytes per observation to
    a buffer and folded it serially, which was a third of the time. A
    point's linearization also no longer copies its frame list. On the
    `large_global_ba_phases` bench (3M observations) one linearization
    went from 350 ms to 190 ms.
  - **Point filters** compute each observing image's pose and
    projection centre once per pass, in one table shared by both
    filters. They used to compose every observation's pose through its
    frame and rig. The triangulation-angle filter also no longer
    collects and sorts every observation's image id.
  - **`find_local_bundle`** counts shared points through a table indexed
    by image id instead of a `BTreeMap` and a `HashMap`, and takes the
    75th-percentile angle with a selection instead of a sort.

  On the 13 ETH3D scenes every centre error is unchanged to the reported
  digit; mapper time is 41.1 s → 39.6 s (facade, the largest, 17.2 s →
  15.6 s).

- **Faster COLMAP-port bundle adjustment.** Profiling a SmallCity replay
  (5,816 images, 18.5 min mapper) put 400 s in global-BA solves and
  76 s in local-BA solves, nearly all of them a single LM step. Three
  changes:
  - A solve that stops on the function tolerance no longer
    re-linearizes at the accepted point first. Ceres does not either. The
    old order spent one of every two linearizations of a one-step solve
    on nothing.
  - The Schur complement's per-pair term reuses the point's
    inverse-Hessian-scaled block and accumulates with fused
    multiply-adds (it recomputed the scaled block per pair).
  - A point's frame pairs are stored frame-major, so the per-row
    elimination of the iterative path reads them contiguously. It also
    adds each point shard's partial sums once at the shard's end instead
    of tracking shards per block.

  Both elimination paths stay bit-identical to each other (new test). On
  a synthetic 1,200-frame / 3M-observation global BA (`large_global_ba`
  bench, 4 threads) the solve went from 11.9 s to 8.1 s: elimination
  6.1 → 3.1 s, linearization 3.0 → 2.0 s. On the 13 ETH3D DSLR scenes
  (intrinsics refined) every centre error is unchanged to the reported
  digit, and mapper wall time fell from 51.5 s to 41.1 s.

- **`gsplat_photos` keeps up to 8192 SIFT detections per image (was
  4000), COLMAP's `max_num_features`.** On textured scenes 4000 kept
  about half of COLMAP's features (playground: 5,934 against 10,000-13,500
  per image), and the photos the port could not register were the ones
  with the fewest verified matches. COLMAP's own mapper on our matches
  dropped the same photos, so the gap was in the features. On the 13
  ETH3D DSLR scenes (1600 px, fixed focal), 8192 registers what COLMAP
  4.2 registers on every scene:

  | scene | 4000 | 8192 | COLMAP 4.2 |
  |---|---|---|---|
  | playground | 35/38, 0.784 cm | 38/38, 0.628 cm | 38/38, 0.619 cm |
  | facade | 75/76, 6.835 cm | 76/76, 6.688 cm | 76/76, 6.541 cm |
  | delivery_area | 2.542 cm | 2.354 cm | 2.567 cm |
  | courtyard | 4.568 cm | 4.431 cm | 4.427 cm |
  | terrains | 0.551 cm | 0.526 cm | 0.580 cm |
  | meadow | 5.878 cm | 5.839 cm | 5.009 cm |
  | terrace | 1.949 cm | 2.049 cm | 1.867 cm |
  | relief | 0.782 cm | 0.855 cm | 0.800 cm |
  | electro | 3.431 cm | 3.506 cm | 3.431 cm |

  Centre RMSE after a Sim(3) fit; office, pipes, relief_2 and kicker
  (under 4000 detections an image) are unchanged. The 13-scene mean RMSE
  goes from 2.45 to 2.41 cm. An earlier SmallCity run with 8000 (before
  the PnP fix) cut the median centre error from 1.66 m to 0.65 m. CPU
  matching takes 2-3.5x longer (facade 317 s -> 1100 s). `--max-keypoints
  4000` restores the old cap.

- **CPU descriptor matching runs on a 16-bit integer kernel for byte
  descriptors, about 2x faster.** `BruteForceMatcher::
  match_descriptors_cross_checked` (the CPU matcher of `gsplat_photos` and
  `gsplat_euroc`) computed every dot product with an f32 matrix product,
  stored the `n x m` result and scanned it for the top two in both
  directions. Descriptors that are bytes over a power-of-two scale
  (`gsplat_photos` keeps COLMAP's bytes / 512) now go through an AVX2
  `pmaddwd` kernel on 4 x 16 tiles with the top-2 scan fused in. Such dot
  products are exact in both f32 and integers, and rows are visited in the
  same order, so the matches are bit-identical; other descriptors keep the
  f32 path. 11,000 x 11,000 descriptors: 1.06 s -> 0.50 s a pair on one
  core (`match_bench` example).

- **COLMAP-port BA builds its problem point by point.** The native solve
  used to list every observation image by image (with a camera id each),
  copy that list twice, and have the solver regroup it by point with a
  stable sort. It now builds each point's observations straight from its
  track, in the order the regrouping produced (in-window elements by
  image and 2D point, then the pulled-in ones), so the problem and the
  results are bit-identical. On the Courthouse replay this takes BA setup
  and problem build from 5.7 s to 4.0 s, and peak memory from 618 MB to
  550 MB. Courthouse output is byte-identical.
- **The correspondence graph looks images up with an integer hash.** It
  used SipHash on the image id, which took most of `find_correspondences`,
  the inner loop of track completion and merging. Courthouse
  complete/merge in global refinement takes 5.9 s instead of 7.0 s.

- **The COLMAP-port BA's PCG is preconditioned with runs of 32 frames.**
  Block-Jacobi dropped every coupling between cameras. Cameras next to
  each other in capture (frame-id) order share most points, so the
  preconditioner is now a dense inverse over each run of 32 consecutive
  free frames. It falls back to block-Jacobi when a run is not positive
  definite. On a SmallCity mapper replay (same machine and inputs), PCG
  took 30,634 iterations instead of 50,393, the solves 405 s instead of
  612 s, and the mapper 904 s instead of 1,088 s, with a comparable result
  (centre error median 0.87 m against 0.96 m). `VISLOC_PORT_PCG_CLUSTER=k`
  changes the run length; `1` restores block-Jacobi. Systems under 1,000
  free frames do not use PCG and are unchanged.
- **VLFeat-compatible SIFT is 2.2x faster on the CPU, bit-identically.** The
  separable blur now runs a kernel tap at a time over whole rows (contiguous
  and vectorizable, same per-sample summation order), and the extremum scan
  stops as soon as a sample fails the threshold or both neighbour tests.
  One 1,600 x 1,066 photo: 5.72 s -> 2.64 s, with the same keypoints and
  descriptors bit for bit (`crates/vision/examples/sift_vlfeat_bench.rs`
  prints a checksum).
- **`gsplat_photos` CPU matching computes one descriptor product per pair**
  instead of two (`match_descriptors_cross_checked` instead of
  `CrossCheckMatcher`): 3.6x faster on 8,000-keypoint photos, identical
  matches.
- **`gsplat_photos` frees SfM inputs it no longer needs.** Byte-identical
  models. Grey images for SIFT are made 256 at a time instead of all at
  once (4 bytes a pixel: 16.5 GB for SmallCity's 5,822 photos). With
  `--no-refine-intrinsics` the SIFT descriptors are freed before the mapper
  (13 GB at 4,000 keypoints). The port's in-memory `pairs.bin` is dropped
  once written. A 553-image Courthouse run (CPU, 800 keypoints) peaks at
  972 MB instead of 1,479 MB. SmallCity with `--retrieval 60` had been
  killed on a 50 GB machine right after matching.
- **`gsplat_photos --retrieval` skips the window's own neighbours.** The
  VLAD top-k used to rank every image. On the SmallCity capture, three
  quarters of the k slots went to images already paired by the sliding
  window, leaving the four passes around the block weakly tied. They drifted
  apart in scale by up to 11%, and passes 2 and 3 bent by metres. Retrieval
  now picks its k among images more than `--window` positions away, which
  adds 126,641 pairs instead of 44,063, with matching time unchanged. Scored
  against the release's own COLMAP model by camera centres after a Sim(3)
  alignment: median error 1.82 m -> 0.88 m, p90 3.49 m -> 2.03 m, max
  8.71 m -> 3.36 m. Within-pass error drops from 1.80 m / 1.11 m to
  0.15 m / 0.05 m on passes 2 and 3. 5,549 instead of 5,581 photos are
  registered. `scripts/score_smallcity_reference.py` fetches that reference
  (about 300 MB of the 3 GB release, by HTTP range) and scores a model.
- **`gsplat_photos` matches COLMAP-style u8 descriptors on the GPU by
  default** (`FeatureBank::upload_u8`: round(512 x) clamped to 255, exact
  integer dot products), 2.2x faster than the previous f32 kernel on an
  NVIDIA L4 (0.92 -> 0.41 ms per cross-checked pair at 4,000 SIFT) with a
  quarter of the descriptor memory. It keeps 99.5% of the f32 matches on
  Tanks and Temples Courthouse; over three SfM settings there, it
  registered 553, 551 and 441 of 553 frames against f32's 550, 439 and
  552 (the mapper's model splits are sensitive to tiny match changes).
  `--match-f32` restores f32 matching.
- **Faster large-scene reconstruction.** Same results: the f32 GPU
  matcher is 1.35x faster (generated by `scripts/gen_match_wgsl.py`), TSDF
  fusion skips blocks outside each view's frustum, VLAD retrieval computes
  its similarities as GEMMs, k-means++ seeding is O(k), and match
  verification overlaps GPU matching. Two-view verification is 2.1x faster:
  the five-point RANSAC's minimal-sample cheirality test is PoseLib's closed
  form instead of one SVD per point (same reports on the benchmarks).
  Approximate: past 1,000 free frames,
  global BA solves the reduced camera system with preconditioned CG
  (relative residual 1e-2; Ceres' `ITERATIVE_SCHUR` default is 0.1);
  smaller problems are unchanged.
- **COLMAP-port bundle adjustment hands the solver arrays, not maps.**
  Byte-identical models. The problem used to be built as a
  `BundleAdjustment`, with a `Camera` clone (a heap-allocated parameter
  vector) per observation and `BTreeMap`s of every pose and landmark, which
  the native solver then converted back into arrays. It is now built
  directly as compact observations plus frame and point arrays. The
  Courthouse mapper takes 24.9 s instead of 29.1 s (mean of two runs, 4
  cores, iterative solve forced on).
- **COLMAP-port mapper 2.2x faster on SmallCity** (5,822 photos: 3,866 ->
  1,782 s). Most of it keeps the model byte-identical. Global BA's Schur
  elimination now runs row by row, in parallel over its outputs, and its
  problem build is 2.4x faster. `CompleteImage`, `TriangulateImage` and the
  negative-depth filter are speculatively parallel. `FindLocalBundle`
  indexes the image's shared points once. Together these bring the mapper
  to 2,477 s. Two BA changes alter results:
  - A failed linear solve is now an invalid step, as in Ceres: the trust
    region shrinks and the solve continues, and five failures in a row end
    it. Before, one failure discarded the whole BA, including every
    accepted step.
  - LM stops once an accepted step lowers the cost by less than 1e-5. This
    deliberately deviates from COLMAP's `function_tolerance=0`; several
    global BAs had crept to the 50-iteration cap at ~1e-6 relative decrease
    per step. `VISLOC_PORT_BA_FUNCTION_TOLERANCE=0` restores COLMAP's
    behaviour.
  - The PCG relative residual tolerance is 1e-2 instead of 1e-3. On a
    SmallCity mapper replay this halves the PCG iterations per global solve
    (356 -> 190) and the mapper takes 1,184 s instead of 1,686 s. The main
    model registers 5,599 images instead of 5,606, with a median
    camera-centre difference of 0.2% of the scene extent, which is within
    the spread of the other solver variants tried.
    `VISLOC_PORT_PCG_TOL=1e-3` restores the previous tolerance.
  - The PCG product reads each camera-pair block once, in f32, instead of
    reading both triangles in f64. On a SmallCity mapper replay a PCG
    iteration takes 5.6 ms instead of 18.5 ms, PCG time drops from 414 s
    to 144 s and the mapper from 1,122 s to 836 s. The main model
    registers 5,581 images instead of 5,599, with a median camera-centre
    difference of 0.19% of the scene extent.
    `VISLOC_PORT_PCG_F64_BLOCKS=1` keeps f64 blocks.

  On SmallCity the main model keeps 5,569 of 5,583 images (scattered
  marginal frames swap in and out) and 956,701 of 957,418 points, with
  camera centres within 0.12 % (median) / 0.96 % (max) of the scene extent.
  On Courthouse the one global BA that used to fail now succeeds: 544
  images registered (was 543).
- **Further mapper speed-ups (byte-identical output).** Courthouse replay:
  47.6 -> ~34 s.
  - The reconstruction's images, frames and 3D points live in id-indexed
    tables instead of `BTreeMap`s. `Reconstruction::{images,frames,points3d}()`
    now return `&IdMap<_>`, with the same read API.
  - Global BA's linearization runs in cache-sized chunks, its edge pattern
    is built per frame, and its elimination computes each row of the
    reduced system in one pass. On a half-scale SmallCity-shaped benchmark
    (`profile_synthetic_city`), linearization is 3-4x faster.
  - The track-completion BFS uses a hashed visited set.
  - A fixed-budget PnP RANSAC scores its hypotheses in parallel, behind
    `visloc-vision`'s new `parallel` feature. `PoseEstimator` now requires
    `Sync`.
- **Large source files split into submodules (no behavior change).**
  `pipelines/slam/src/incremental_sfm.rs` (18.5k lines), `bundle.rs` (17k),
  `examples/unordered_sfm_demo.rs` (22.5k, now
  `examples/unordered_sfm_demo/main.rs` plus modules) and
  `pipelines/basalt/src/vio/aom.rs` (21.5k) are now directories of files
  mostly under 3k lines. Pure code moves: public paths, test names and test
  counts are unchanged, and the Basalt parity arithmetic is untouched.
- **Planning docs archived.** The long `PLAN.md` log and the itemized 0.2.0
  change log moved to `docs/archive/`; `PLAN.md` is now a one-page handoff.

## 0.2.1 - 2026-10-08

Patch release: camera-distortion correctness fixes. Each fix below corrected
results that were silently wrong, not just errors that were reported.

### Fixed

- **COLMAP camera export with self-calibrated distortion.** A `Pinhole`
  camera carrying the refined radial `[k1, k2]` tail
  (`--refine-distortion`) was written as `PINHOLE` with 6 params, which is
  not valid COLMAP (`PINHOLE` has 4); COLMAP rejects it and other readers
  silently drop the distortion. Every COLMAP writer (text and binary) now
  exports it as `OPENCV` with `p1 = p2 = 0` (the identical projection), and
  fallible writers reject any camera whose parameter count does not match its
  COLMAP model. New public helper `visloc_io::colmap::colmap_camera_record`.
- **OPENCV tangential distortion.** `Camera::project` / `normalize_pixel` /
  `unit_ray_from_pixel` ignored `p1, p2` of an `OpenCv` camera, so a COLMAP
  `OPENCV` model with non-zero tangential terms was silently treated as
  radial-only. Such cameras now use the full Brown-Conrady model (shared with
  `FULL_OPENCV`, k3..k6 = 0); results with `p1 = p2 = 0` are bit-identical to
  before. New `Camera::tangential_distortion`.
- **Pose/structure BA ignored self-calibrated radial distortion.** With
  `refine_intrinsics = false` (e.g. the SfM polish / warm-start passes after
  `--refine-distortion`), `optimize_weighted` built its residuals and
  Jacobians with an undistorted pinhole while the cost it accepts steps
  against used the distorted `Camera::project`, so LM optimised a different
  model than it measured and could stall. The monocular residual and Jacobian
  (serial, parallel, and dense-QR assembly, plus the GNC residual vector) now
  apply `1 + k1·r² + k2·r⁴` with an analytic Jacobian; distortion-free cameras
  keep the historical closed form bit-for-bit.

### Added

- **`BaConfig::shared_focal` / `--shared-focal`** (unordered and sequential
  SfM demos): constrain intrinsics self-calibration to `fx == fy`. Previously
  `fx` and `fy` were always refined independently, so weakly observable
  motion (near-pure forward translation) could settle on a spurious aspect
  ratio. Default off; the joint solve is unchanged when it is not set.

## 0.2.0 - 2026-09-15

### Highlights

- **Real-data SfM at scale.** All ten ETH3D low-resolution many-view scenes:
  **9,996/10,008 cameras registered (99.88%)**, with no mapper run above
  3.32 GiB. Same-input ETH3D Electro 1,200 runs **3.46× faster than COLMAP**
  with **25.2% lower** camera-centre RMSE, and the 38-image courtyard control
  reaches **0.5379 cm vs COLMAP's 1.6166 cm**.
- **Visual-Inertial SLAM (faithful Basalt Rust port).** A tightly-coupled
  stereo-inertial VIO estimator plus offline mapper that matches native
  Basalt's ATE within **0.1%** on all 11 EuRoC sequences and, with EuRoC's
  official calibration, beats measured ORB-SLAM3 on **8 of 11** sequences.
- **COLMAP-compatible unordered SfM** with VLAD retrieval, essential-matrix
  verification, incremental mapping, connected-component models, and COLMAP
  text export for downstream 3DGS / MVS.
- **Online stereo SLAM** with loop closure, SE(3)/Sim(3) pose-graph
  optimization, GNC outlier rejection, and bundle adjustment, benchmarked on
  KITTI, EuRoC, and TUM RGB-D.
- **Opt-in deep frontend** (SuperPoint / LightGlue via in-Rust ONNX Runtime,
  CUDA-accelerated), plus public-data map-reuse localization and GNSS-prior
  tracking demos.

### Full change log

The itemized 0.2.0 change log (every Added / Changed / Fixed entry from
2026-05-07 to 2026-09-15, about 4,800 lines) is archived in
[`docs/archive/changelog_0.2.0_full.md`](docs/archive/changelog_0.2.0_full.md).

## 0.1.0 - 2026-05-07

### Added

- Workspace split into core, vision, IO, localization pipeline, and tracking pipeline crates.
- Core visual localization types: `Frame`, `Keyframe`, `VisualMap`, `Landmark`, `Observation`, `Camera`, `Pose`, and `LocalizationResult`.
- `SO3` / `SE3` pose wrappers and reprojection utilities built on `nalgebra`.
- Brute-force descriptor matching with L2 distance, ratio test, optional cross-checking, and match diagnostics.
- Minimal DLT PnP estimator, PnP RANSAC, pose-estimation diagnostics, and optional Gauss-Newton pose refinement.
- COLMAP text and binary map parsers for `cameras`, `images`, and `points3D`.
- Text parsers for landmark descriptors and query features.
- Localization pipeline over query descriptors and visual-map landmarks.
- Map providers, submap selectors, priors, localization quality gates, and map validation reports.
- Tracking skeleton with motion models, state transitions, and sequence examples.
- Local mapping skeleton with keyframe policy, local map windows, landmark candidates, linear triangulation, staged map updates, and local refinement hooks.
- Online SLAM MVP composition that combines tracking and local mapping without loop closure or global optimization.
- COLMAP text model writer for saving reusable sparse maps.
- Sensor-fusion foundation crate with timestamped frames/poses, GNSS/pose/IMU measurements, covariance types, measurement buffers, frame prior sources, and external localization-prior tracking hooks.
- GNSS-prior tracking example showing radius-submap narrowing before localization.
- COLMAP compatibility notes covering supported sparse model inputs, descriptor handling, writer behavior, and current limitations.
- Root crate prelude and top-level re-exports for common application-facing localization APIs.
- Pre-1.0 to v1.0 migration guide covering recommended imports, localization boundaries, COLMAP descriptor handling, tracking priors, and experimental layers.
- Package metadata and crate-content checks in the local quality gate and CI.
- crates.io package metadata now includes project homepage and repository URLs.
- Workspace member crates now use crate-specific descriptions and docs.rs URLs.
- Publishing guide documenting workspace crate publish order and package-check workflow.
- Examples, integration tests, design docs, local check script, and GitHub Actions CI.

### Not Yet Implemented

- Full Visual SLAM.
- Full SfM.
- Loop closure.
- Dense mapping.
- Full bundle adjustment.
- Full tightly-coupled visual-inertial or GNSS/INS fusion.
