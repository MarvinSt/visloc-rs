# ETH3D DSLR: visloc-rs vs COLMAP 4.2 incremental vs GLOMAP

The visloc-rs mapper (`colmap_incremental_mapper`) against COLMAP's two
mappers in pycolmap 4.2.1, on the same inputs:

- the incremental mapper;
- the global mapper, GLOMAP as merged into COLMAP.

The scoring follows the GLOMAP paper.

**Result: visloc-rs is on par with both, and maps about seven times faster.**

- **On COLMAP's matches (the GLOMAP paper's setup):** visloc-rs is ahead of
  COLMAP's incremental mapper at every threshold, by 0.3 to 0.4 AUC points.
  Against the global mapper it is within 0.7 points: lower at 1°, level at
  3°, higher at 5°.
- **On visloc-rs's own matches:** it is ahead of both at every threshold.
  The lead is 1.8 to 4.6 points over their three-run means, and 0.5 to 3.5
  over their best runs.
- **Where it trails:** the global mapper registers more images (454 vs 445)
  and leads at 1° on COLMAP's matches. Single scenes swing more than these
  margins (see Caveats).
- **On a large capture** (SmallCity, 5,822 photos, below): visloc-rs mapped
  it in 22 minutes. COLMAP's global mapper had not finished after 94 minutes.

## Protocol

- **Data:** the 13 ETH3D high-resolution multi-view training scenes,
  undistorted DSLR images, 14 to 76 images each, 454 in all.
- **Ground truth:** ETH3D's laser-scan-aligned poses
  (`dslr_calibration_undistorted`).
- **Camera:** every method gets the same single PINHOLE camera, held fixed.
  The images are resized to 1,600 px. The focal length is the ground truth's,
  scaled. The principal point is the image centre, about 3 px from the ground
  truth's.
- **Two input modes** (`scripts/benchmark_eth3d_auc.py`):
  - `colmap-matches`, the GLOMAP paper's setup: COLMAP's SIFT, exhaustive
    matching and geometric verification feed all three mappers. visloc-rs
    reads COLMAP's keypoints and two-view inliers in its replay format.
  - `visloc-matches`: the keypoints and verified matches of visloc-rs's own
    front end (`gsplat_photos --cpu --max-keypoints 8192`) feed all three.
    COLMAP re-verifies them to build its two-view geometries, so its inliers
    are a subset of visloc-rs's.
- **visloc-rs's mapper options:** the ones `gsplat_photos` writes,
  `abs_pose_min_num_inliers=30` and `structure_less_registration_fallback=1`.
  The fallback is COLMAP 4's 2D–2D registration. The port's mapper leaves it
  off by default, for parity with the COLMAP version it follows, and the
  photo and EuRoC pipelines turn it on.
- **Metrics:**
  - AUC at 1°, 3° and 5° of the relative pose error over every pair of
    ground-truth images. The error is the larger of the rotation angle and
    the translation-direction angle. A pair with an unregistered image counts
    as a failure, and only the largest model is scored.
  - Registered images.
  - Camera-centre RMSE after a Sim(3) fit.
  - Mapper wall time.
- **Machine:** 4 vCPUs (Intel Xeon, 2.1 GHz). Every mapper uses 4 threads.
  The COLMAP runs are CPU-only.

## Results

### COLMAP's features and matches (GLOMAP protocol)

| Mapper | Mean AUC@1° / 3° / 5° | Registered | Mapper time, 13 scenes |
|---|---|---|---|
| visloc-rs | 50.86 / 75.47 / **82.14** | 445 / 454 | **28.4 s** |
| COLMAP incremental | 50.42 / 75.20 / 81.74 | 445 / 454 | 200.7 s |
| COLMAP global (GLOMAP) | **51.55** / **75.53** / 81.70 | **454 / 454** | 184.4 s |

COLMAP's feature extraction took 324 s and its exhaustive matching 427 s over
the 13 scenes. Neither is part of the mapper times above.

| Scene | visloc-rs AUC@1/3/5 (reg.) | COLMAP incremental | COLMAP global |
|---|---|---|---|
| courtyard | 54.4 / 82.7 / 89.6 (38/38) | 53.9 / 82.3 / 89.4 (38/38) | 32.5 / 44.1 / 46.8 (38/38) |
| delivery_area | 53.4 / 82.4 / 89.3 (44/44) | 53.5 / 82.4 / 89.3 (44/44) | 53.0 / 82.2 / 89.1 (44/44) |
| electro | 44.0 / 65.4 / 71.0 (41/45) | 43.8 / 65.4 / 71.0 (41/45) | 47.4 / 71.7 / 78.1 (45/45) |
| facade | 70.5 / 88.5 / 93.0 (76/76) | 70.3 / 88.4 / 92.9 (76/76) | 60.3 / 73.8 / 77.2 (76/76) |
| kicker | 52.1 / 76.3 / 82.6 (30/31) | 52.0 / 76.0 / 82.4 (30/31) | 45.4 / 65.7 / 71.3 (31/31) |
| meadow | 39.2 / 64.5 / 73.4 (14/15) | 27.4 / 53.2 / 60.8 (14/15) | 58.6 / 84.8 / 90.9 (15/15) |
| office | 19.5 / 44.0 / 58.6 (26/26) | 22.9 / 51.0 / 65.5 (26/26) | 27.3 / 53.6 / 65.8 (26/26) |
| pipes | 33.9 / 50.6 / 54.7 (11/14) | 34.7 / 51.0 / 54.9 (11/14) | 47.7 / 78.4 / 86.6 (14/14) |
| playground | 64.5 / 87.2 / 92.2 (38/38) | 65.8 / 87.7 / 92.6 (38/38) | 64.9 / 87.3 / 92.3 (38/38) |
| relief | 57.6 / 84.6 / 90.7 (31/31) | 57.6 / 84.6 / 90.7 (31/31) | 57.9 / 84.7 / 90.8 (31/31) |
| relief_2 | 57.6 / 85.1 / 91.0 (31/31) | 58.8 / 85.5 / 91.2 (31/31) | 58.5 / 85.2 / 91.1 (31/31) |
| terrace | 54.3 / 83.9 / 90.3 (23/23) | 54.7 / 84.1 / 90.5 (23/23) | 52.9 / 83.3 / 90.0 (23/23) |
| terrains | 60.2 / 86.0 / 91.5 (42/42) | 60.2 / 86.0 / 91.5 (42/42) | 63.8 / 87.1 / 92.2 (42/42) |

### visloc-rs's features and matches

This mode needs no image download, so COLMAP was run three times. visloc-rs
is deterministic for given inputs.

| Mapper | Mean AUC@1° / 3° / 5° | Registered | Mapper time, 13 scenes |
|---|---|---|---|
| visloc-rs | **53.09** / **78.85** / **85.69** | 448 / 454 | **32.6 s** |
| COLMAP incremental, mean of 3 runs | 50.23 / 75.12 / 81.96 | 448 / 454 | 215–234 s |
| COLMAP global (GLOMAP), mean of 3 runs | 51.33 / 74.96 / 81.14 | **454 / 454** | 202–219 s |

- **COLMAP incremental:** its run means ranged from 47.82 / 71.51 / 78.28 to
  51.58 / 77.37 / 84.29. In one run it failed on playground (AUC@1° 21.5
  instead of 65).
- **COLMAP global:** its run means ranged from 50.63 / 73.93 / 79.84 to
  52.62 / 76.26 / 82.19.
- **visloc-rs** is ahead of every one of these runs at every threshold.
- **Per-scene table:** the one below is COLMAP's first run.

| Scene | visloc-rs AUC@1/3/5 (reg.) | COLMAP incremental | COLMAP global |
|---|---|---|---|
| courtyard | 54.2 / 82.5 / 89.5 (38/38) | 54.1 / 82.3 / 89.4 (38/38) | 54.9 / 82.7 / 89.6 (38/38) |
| delivery_area | 53.1 / 82.4 / 89.3 (44/44) | 53.5 / 82.5 / 89.3 (44/44) | 53.2 / 82.3 / 89.2 (44/44) |
| electro | 43.5 / 65.0 / 70.8 (41/45) | 43.4 / 65.2 / 70.9 (41/45) | 48.5 / 76.7 / 84.5 (45/45) |
| facade | 71.0 / 88.6 / 93.0 (76/76) | 70.6 / 88.5 / 92.9 (76/76) | 71.0 / 88.7 / 93.0 (76/76) |
| kicker | 49.5 / 74.8 / 81.6 (30/31) | 51.1 / 75.7 / 82.2 (30/31) | 48.8 / 74.3 / 81.2 (31/31) |
| meadow | 39.0 / 63.7 / 72.8 (14/15) | 12.9 / 33.2 / 42.7 (14/15) | 25.9 / 46.2 / 60.5 (15/15) |
| office | 27.8 / 58.5 / 72.1 (26/26) | 28.4 / 57.0 / 70.3 (26/26) | 20.1 / 36.5 / 43.3 (26/26) |
| pipes | 52.7 / 81.6 / 88.5 (14/14) | 52.9 / 81.6 / 88.5 (14/14) | 53.4 / 82.0 / 88.8 (14/14) |
| playground | 65.0 / 87.4 / 92.4 (38/38) | 65.0 / 87.4 / 92.4 (38/38) | 65.6 / 87.6 / 92.5 (38/38) |
| relief | 58.2 / 84.7 / 90.8 (31/31) | 58.4 / 84.8 / 90.8 (31/31) | 58.8 / 85.0 / 91.0 (31/31) |
| relief_2 | 59.6 / 85.8 / 91.4 (31/31) | 60.5 / 86.1 / 91.6 (31/31) | 42.4 / 58.7 / 62.3 (31/31) |
| terrace | 53.8 / 83.4 / 90.0 (23/23) | 53.4 / 83.5 / 90.1 (23/23) | 52.9 / 83.3 / 90.0 (23/23) |
| terrains | 62.7 / 86.7 / 91.8 (42/42) | 62.5 / 86.6 / 91.8 (42/42) | 63.9 / 87.1 / 92.2 (42/42) |

## The structure-less fallback

The port's mapper default leaves out COLMAP 4's structure-less fallback. On
that default, visloc-rs registers fewer images and scores lower:

| Input | Registered | Mean AUC@1° / 3° / 5° |
|---|---|---|
| COLMAP's matches | 438 / 454 | 50.65 / 74.45 / 80.51 |
| visloc-rs's matches | 445 / 454 | 51.67 / 76.46 / 83.07 |

The fallback registers the images that 2D–3D correspondences cannot:

- office on COLMAP's matches, which otherwise splits into models of 6 and
  20 images;
- one image of electro, on COLMAP's matches;
- three images of pipes on visloc-rs's matches.

No other scene changes.

## Where visloc-rs trails

- **Registrations:** the global mapper registers every image. On electro,
  meadow and pipes, with COLMAP's matches, it also scores highest.
- **office (COLMAP's matches):** both COLMAP mappers score higher there.
- **kicker (visloc-rs's matches):** COLMAP's incremental mapper is ahead
  there.

## A large capture: SmallCity

The ETH3D scenes are small. SmallCity is the 5,822 photos of the
Hierarchical 3DGS release: six GoPros on a bicycle, four passes around a
city block. COLMAP's global mapper is set up as in `visloc-matches`:

- **Inputs:** the keypoints and verified matches of one `gsplat_photos` run
  (1,024 px, window and VLAD retrieval matching, 217,844 image pairs) and the
  release's focal length, held fixed. COLMAP re-verifies the matches.
- **Machine:** one RunPod machine. Both mappers use 32 threads and run on
  the CPU.
- **visloc-rs:** mapper seed 0, with the options `gsplat_photos` writes.
- **Accuracy:** camera centres against the release's own COLMAP model, a
  reference rather than ground truth, after a Sim(3) fit
  (`scripts/score_smallcity_reference.py`).

| | visloc-rs | COLMAP global (GLOMAP) |
|---|---|---|
| Mapper wall time | **21.7 min** | unfinished, stopped after 94.3 min |
| Peak memory | **19.3 GiB** | 33.5 GiB up to the stop |
| Registered | 5,816 / 5,822 | — |
| Camera-centre error, median / p90 | 0.15 m / 0.28 m | — |

A 100-minute cap on COLMAP's side stopped it. The cap also covered building
its database (0.9 min) and re-verifying the matches (4.5 min), which come
before the mapper. COLMAP's stages up to the stop:

| Stage | Time |
|---|---|
| Loading, relative poses, connected components | 2.9 min |
| Rotation averaging | 2.2 min |
| Track establishment | 1.9 min |
| Global positioning | 31.6 min |
| Bundle adjustment, three rounds | 36.3 min |
| Retriangulation and refinement | stopped after 19.3 min, merging tracks |

- **CPU only.** The pycolmap wheel from PyPI is built without CUDA, so
  COLMAP's global positioning and bundle adjustment fell back to CPU
  solvers; it logs this. A CUDA build would run them on the GPU, which this
  comparison does not measure.
- **One run each.** COLMAP is not deterministic. visloc-rs is.

The logs are in `benchmarks/smallcity_glomap/`.

## On a GPU: COLMAP with CUDA and Caspar

The runs above are CPU-only. This pass repeats them on one GPU machine
(RunPod: an NVIDIA L4 with 24 GB, 21 CPU threads) against COLMAP 4.2.1
built from source with CUDA and Caspar, its GPU bundle adjuster:

- **COLMAP:** SIFT extraction and exhaustive matching on the GPU; both
  mappers with the Caspar backend (`BundleAdjustmentBackend.CASPAR`).
  - By COLMAP's design, the global mapper's fixed-rotation stage of each
    bundle adjustment stays on Ceres.
  - Ubuntu's Ceres has no CUDA, so that stage and the global positioning
    run on the CPU.
- **visloc-rs:** matching on the GPU (wgpu), SIFT and the mapper on the CPU.
  Its GPU bundle adjustment starts at 1,000 frames, so these scenes do not
  use it.
- **Everything else as above.** Commit `3dd9550`, 21 threads for every step.

### ETH3D

| Input | Mapper | Mean AUC@1° / 3° / 5° | Registered | Mapper time |
|---|---|---|---|---|
| COLMAP's GPU features | visloc-rs | **50.42** / **74.48** / **80.60** | 439 / 454 | **34.1 s** |
| | COLMAP incremental, Caspar | 49.90 / 74.29 / 80.43 | 439 / 454 | 163.0 s |
| | COLMAP global, Caspar | 48.96 / 73.53 / 80.25 | **454 / 454** | 158.6 s |
| visloc-rs's features | visloc-rs | **53.09** / **78.85** / **85.69** | 448 / 454 | **37.4 s** |
| | COLMAP incremental, Caspar | 50.45 / 76.07 / 83.31 | 447 / 454 | 163.8 s |
| | COLMAP global, Caspar | 47.08 / 70.32 / 76.89 | **454 / 454** | 183.7 s |

- **visloc-rs leads both GPU mappers** at every threshold in both modes, and
  maps 4.4 to 4.9 times faster.
- **visloc-rs's GPU matching** reconstructs every scene as its CPU
  matching does: each score equals the CPU run's above.
- **COLMAP with Ceres on the same machine** (visloc-rs's features):

  | Mapper | Mean AUC@1° / 3° / 5° | Mapper time |
  |---|---|---|
  | visloc-rs | 53.09 / **78.85** / **85.69** | **38.0 s** |
  | COLMAP incremental, Ceres | **53.50** / 78.80 / 85.53 | 223.6 s |
  | COLMAP global, Ceres | 51.03 / 75.78 / 82.25 | 226.8 s |

  - In this run COLMAP's incremental mapper is level with visloc-rs: 0.4
    points ahead at 1°, 0.05 and 0.16 behind at 3° and 5°. Its three
    earlier runs on these inputs averaged 50.23 / 75.12 / 81.96.
  - **Caspar trades accuracy for speed.** It made COLMAP's mappers 27%
    (incremental) and 19% (global) faster here. On delivery_area, COLMAP's
    incremental mapper scored 53.5 to 53.7 AUC@1° in all five Ceres runs
    (four above, one here) and 46.7 and 47.2 in the two Caspar runs.
    Caspar solves in f32.
- **Front ends**, 13 scenes, each on its own features:

  | Step | COLMAP | visloc-rs |
  |---|---|---|
  | Feature extraction | **22.2 s** (GPU) | 132.4 s (CPU) |
  | Matching and verification | 44.2 s | **25.9 s** |
  | Mapper | 158.6 s (global) | **37.4 s** |
  | Total | 225.0 s | **195.7 s** |

  - visloc-rs's SIFT is its VLFeat-compatible CPU extractor, the photo
    pipeline's default. The GPU SIFT in `visloc-sift-gpu` does not
    reproduce VLFeat's detector and descriptor, so the pipeline does not
    use it with these settings. This is the largest gap left: COLMAP
    extracts six times faster.
  - COLMAP's extraction time includes decoding its images, resized to
    1,600 px beforehand. visloc-rs's starts after decoding.

### SmallCity

The same machine and settings as the CPU run above: visloc-rs's matches of
the 5,822 photos, 21 threads.

| | visloc-rs | visloc-rs, `--gpu-ba 1` | COLMAP global, Caspar | COLMAP incremental, Caspar |
|---|---|---|---|---|
| Mapper wall time | **19.5 min** | 19.9 min | unfinished, stopped at 90 min | unfinished, stopped at 75 min |
| Peak memory | 18.8 GiB | 16.9 GiB | 31.4 GiB up to the stop | **5.9 GiB** up to the stop |
| Registered | 5,816 / 5,822 | 5,816 / 5,822 | — | 5,352 when stopped |
| Camera-centre error, median / p90 | 0.281 m / 0.731 m | 0.577 m / 1.946 m | — | — |

COLMAP's incremental mapper registered about 1,000 images every 14 minutes
and would have needed about 80 minutes for all of them, before its last
global bundle adjustment.

COLMAP's stages up to the stop, against the CPU-only run above (32
threads). The cap also covered its database (1.4 min) and re-verifying the
matches (6.6 min):

| Stage | GPU machine | CPU run |
|---|---|---|
| Rotation averaging | 2.8 min | 2.2 min |
| Track establishment | 1.8 min | 1.9 min |
| Global positioning (Ceres, CPU) | 54.3 min | 31.6 min |
| Bundle adjustment, three rounds | 13.9 min (Caspar) | 36.3 min |
| Retriangulation and refinement | stopped after 7 min | stopped after 19.3 min |

- **Caspar** ran COLMAP's bundle adjustment 2.6 times faster than Ceres did
  on 32 threads. Global positioning stays on Ceres, which Ubuntu builds
  without CUDA, so it ran on the CPU: 54 minutes on 21 threads.
- **visloc-rs's accuracy differs from the CPU run's** (0.15 m there). The
  two runs register the same images in the same order up to the 1,187th
  image; the 1,188th gets one inlier more (2,475 against 2,474) and from
  there the reconstructions part. The mapper gives byte-identical models
  at any thread count, so the likely cause is the platform's floating
  point (Ubuntu 24.04 here, 22.04 before). On this capture the incremental
  mapper's result spreads like that across mapper seeds too (0.15 m to
  0.65 m).
- **The GPU bundle adjustment** is not faster on this GPU, and its run parts
  from the CPU run after its first solve
  ([GPU bundle adjustment](gpu_bundle_adjustment.md)).
- The ETH3D results are in `benchmarks/eth3d_auc_gpu/`, the SmallCity logs
  in `benchmarks/smallcity_gpu/`.

## Caveats

- **COLMAP is not deterministic.** COLMAP runs with its default random seed
  (−1), so both its verification and its mappers vary between runs.
  - **meadow, rerun with COLMAP's matches:** the verification kept 5,749
    inliers instead of 5,739.
    - COLMAP's incremental mapper then succeeded (AUC@1° 40.7, 5.1 cm) where
      the run above drifted (27.4, 6.7 m).
    - On a third pass its global mapper drifted (7.2 m).
    - visloc-rs moved between 39 and 45 AUC@1° with those inputs.
  - **Read single scenes with care.** The `colmap-matches` numbers come from
    one run, because a pass needs about 6 GB of image downloads.
- **What the times cover.** They are mapper wall times at 4 threads,
  including database or file loading. Front ends are not compared.
- **Scope.** Apart from the one SmallCity run, these are small scenes.
  Learned reconstruction methods are not part of this comparison.

## Reproduce

```bash
cargo build --release --example colmap_incremental_mapper
cargo build --release -p visloc-gsplat-train --features gpu,euroc --example gsplat_photos
pip install "pycolmap>=4.2" numpy pillow py7zr
scripts/benchmark_eth3d_auc.sh /tmp/eth3d_auc   # optionally: scene names
```

The numbers above were measured at commit `bdbd274` and are stored in
`benchmarks/eth3d_auc/`: `colmap_matches.jsonl`, and `visloc_matches.jsonl`
with COLMAP's reruns `visloc_matches_run{2,3}.jsonl`. Print their tables
with `python3 scripts/benchmark_eth3d_auc.py table <file> --markdown`.

The stored `gsplat_photos` runs predate the photo pipeline's fallback
option, so their visloc-rs rows were run with `mapper_options.txt` as the
current `gsplat_photos` writes it. A fresh run of the script writes that
file itself.

SmallCity was measured at commit `bdbd274` with the same two builds, the
front end matching on a GPU (no `--cpu`). The images are in the release's
`small_city.zip`:

```bash
target/release/examples/gsplat_photos --images small_city/camera_calibration/rectified/images \
  --out RUN --max-size 1024 --steps 0 --no-mesh --exhaustive-max 600 --window 20 \
  --retrieval 30 --focal 486.97 --no-refine-intrinsics
RAYON_NUM_THREADS=32 target/release/examples/colmap_incremental_mapper \
  --manifest RUN/port/manifest.txt --features-dir RUN/port/features \
  --pairs-export RUN/port/pairs.bin --out-colmap OURS --random-seed 0
timeout 6000 python3 scripts/benchmark_smallcity_glomap.py RUN/port GLOMAP 32
python3 scripts/score_smallcity_reference.py fetch ref_images.txt
python3 scripts/score_smallcity_reference.py score ref_images.txt OURS/model/0/images.txt
python3 scripts/score_smallcity_reference.py score ref_images.txt GLOMAP/glomap_txt/images.txt
```
