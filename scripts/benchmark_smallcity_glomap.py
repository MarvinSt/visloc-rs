#!/usr/bin/env python3
"""SmallCity: COLMAP's global mapper (GLOMAP, pycolmap) on the keypoints and
verified matches of a ``gsplat_photos`` run, set up as in
``benchmark_eth3d_auc.py visloc-matches`` (docs/eth3d_auc_benchmark.md).

    benchmark_smallcity_glomap.py PORT_DIR WORK_DIR THREADS [global|incremental]

PORT_DIR is the run's replay directory (``port/``). COLMAP gets the run's
single PINHOLE camera, held fixed, and re-verifies the matches. Prints one JSON
line per stage and writes the largest model as text to
``WORK_DIR/glomap_txt`` (``WORK_DIR/incremental_txt`` for ``incremental``,
COLMAP's incremental mapper); ``score_smallcity_reference.py score`` scores its
``images.txt``. A second run in the same WORK_DIR reuses the verified database.
``VISLOC_BENCH_BA_BACKEND=CASPAR`` selects COLMAP's GPU bundle adjuster
(``benchmark_eth3d_auc.py``). Needs pycolmap >= 4.2, numpy and Pillow.
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark_eth3d_auc import global_options, incremental_options, port_database


def main():
    if len(sys.argv) not in (4, 5):
        sys.exit(__doc__)
    port, work, threads = sys.argv[1], sys.argv[2], int(sys.argv[3])
    mapper = sys.argv[4] if len(sys.argv) == 5 else "global"
    if mapper not in ("global", "incremental"):
        sys.exit(__doc__)
    import pycolmap

    os.makedirs(work, exist_ok=True)
    with open(os.path.join(port, "manifest.txt")) as f:
        rows = [line.split() for line in f if line.strip() and not line.startswith("#")]
    srow = next(r for r in rows if r[0] == "S")
    frames = [r[2] for r in rows if r[0] == "F"]
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "PINHOLE"
    reader.camera_params = ",".join(str(float(v)) for v in srow[5:9])
    db = os.path.join(work, "database.db")
    verified = os.path.join(work, "verified")

    def stage(name, seconds, **extra):
        print(json.dumps({"stage": name, "seconds": round(seconds, 1), **extra}), flush=True)

    image_dir = os.path.join(work, "blank")  # port_database's blank images
    if not os.path.exists(verified):
        if os.path.exists(db):
            os.remove(db)
        pycolmap.Database.open(db).close()
        t0 = time.time()
        image_dir, pairs_txt = port_database(pycolmap, port, db, work, reader, frames,
                                             (int(srow[3]), int(srow[4])))
        with open(pairs_txt) as f:
            stage("database", time.time() - t0, images=len(frames), pairs=sum(1 for _ in f))
        t0 = time.time()
        tvg = pycolmap.TwoViewGeometryOptions()
        tvg.compute_relative_pose = True
        pycolmap.verify_matches(db, pairs_txt, options=tvg)
        stage("verify_matches", time.time() - t0)
        open(verified, "w").close()

    out = os.path.join(work, "glomap" if mapper == "global" else "incremental")
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    if mapper == "global":
        recs = pycolmap.global_mapping(db, image_dir, out, global_options(pycolmap, threads))
    else:
        recs = pycolmap.incremental_mapping(db, image_dir, out,
                                            incremental_options(pycolmap, threads))
    seconds = time.time() - t0
    rec = max(recs.values(), key=lambda r: r.num_reg_images())
    txt = os.path.join(work, "glomap_txt" if mapper == "global" else "incremental_txt")
    os.makedirs(txt, exist_ok=True)
    rec.write_text(txt)
    stage(f"{mapper}_mapping", seconds, models=len(recs), registered=rec.num_reg_images(),
          points=rec.num_points3D())


if __name__ == "__main__":
    main()
