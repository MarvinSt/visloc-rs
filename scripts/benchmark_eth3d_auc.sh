#!/usr/bin/env bash
# ETH3D DSLR (13 training scenes): the visloc-rs mapper vs COLMAP incremental
# vs COLMAP global (GLOMAP), scored like the GLOMAP paper
# (scripts/benchmark_eth3d_auc.py; results in docs/eth3d_auc_benchmark.md).
#
#   scripts/benchmark_eth3d_auc.sh WORK_DIR [scene ...]
#
# Per scene: download and extract the undistorted DSLR images (0.2-1.3 GB),
# run the gsplat_photos front end (CPU, 1,600 px, 8,192 SIFT keypoints, the
# ground-truth focal length held fixed), score both input modes, and delete the
# images. Appends to WORK_DIR/{colmap,visloc}_matches.jsonl. Needs release
# builds of gsplat_photos and colmap_incremental_mapper, and pycolmap >= 4.2,
# numpy, Pillow and py7zr. With VISLOC_BENCH_DEVICE=cuda (benchmark_eth3d_auc.py's
# GPU settings) the front end also extracts and matches on the GPU.
# VISLOC_BENCH_PHOTOS_ARGS adds gsplat_photos flags, e.g. "--cpu-sift
# --no-complete-model" for the setup of the stored runs that predate them.
set -euo pipefail
W=$1
shift
scenes=("$@")
[ ${#scenes[@]} -gt 0 ] || scenes=(courtyard delivery_area electro facade kicker meadow office
  pipes playground relief relief_2 terrace terrains)
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PHOTOS=$ROOT/target/release/examples/gsplat_photos
MAPPER=$ROOT/target/release/examples/colmap_incremental_mapper
BENCH=$ROOT/scripts/benchmark_eth3d_auc.py
front_cpu=--cpu
[ "${VISLOC_BENCH_DEVICE:-cpu}" = cpu ] || front_cpu=
mkdir -p "$W/dl"
for s in "${scenes[@]}"; do
  D=$W/$s
  S=$D/$s
  if [ ! -d "$S/images/dslr_images_undistorted" ]; then
    curl -fsSL -C - --retry 8 --retry-all-errors --retry-delay 10 -o "$W/dl/$s.7z" \
      "https://www.eth3d.net/data/${s}_dslr_undistorted.7z"
    python3 -c "import py7zr, sys; py7zr.SevenZipFile(sys.argv[1]).extractall(sys.argv[2])" \
      "$W/dl/$s.7z" "$D"
    rm -f "$W/dl/$s.7z"
  fi
  R=$D/run
  if [ ! -f "$R/port/pairs.bin" ]; then
    # The ground-truth focal length scaled to 1,600 px.
    read -r width height focal <<<"$(awk '!/^#/{print $3, $4, $5; exit}' \
      "$S/dslr_calibration_undistorted/cameras.txt")"
    f=$(python3 -c "print($focal * 1600 / max($width, $height))")
    "$PHOTOS" --images "$S/images/dslr_images_undistorted" --out "$R" --max-size 1600 \
      --steps 0 --no-mesh $front_cpu --focal "$f" --no-refine-intrinsics --max-keypoints 8192 \
      ${VISLOC_BENCH_PHOTOS_ARGS:-} >"$D/photos.log" 2>&1
  fi
  python3 -I "$BENCH" colmap-matches "$S" "$R" "$D/colmap_matches" "$MAPPER" \
    2>"$D/colmap_matches.log" >>"$W/colmap_matches.jsonl"
  python3 -I "$BENCH" visloc-matches "$S" "$R" "$D/visloc_matches" "$MAPPER" \
    2>"$D/visloc_matches.log" >>"$W/visloc_matches.jsonl"
  rm -rf "$S/images" "$D/colmap_matches/images"
done
python3 -I "$BENCH" table "$W/colmap_matches.jsonl"
python3 -I "$BENCH" table "$W/visloc_matches.jsonl"
