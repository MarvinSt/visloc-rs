#!/usr/bin/env python3
"""ETH3D DSLR: the visloc-rs mapper vs COLMAP's incremental and global
(GLOMAP) mappers on the same inputs, scored with the GLOMAP paper's metrics.

    benchmark_eth3d_auc.py colmap-matches SCENE_DIR RUN_DIR WORK_DIR MAPPER
    benchmark_eth3d_auc.py visloc-matches SCENE_DIR RUN_DIR WORK_DIR MAPPER
    benchmark_eth3d_auc.py visloc-only SCENE_DIR RUN_DIR WORK_DIR MAPPER
    benchmark_eth3d_auc.py table RESULTS.jsonl [--markdown]

SCENE_DIR is an extracted ETH3D ``<scene>_dslr_undistorted`` scene (its
``images/dslr_images_undistorted`` and ``dslr_calibration_undistorted``).
RUN_DIR is a ``gsplat_photos --steps 0`` run on it (``port/`` and
``image_names.txt``). MAPPER is the ``colmap_incremental_mapper`` binary.
``scripts/benchmark_eth3d_auc.sh`` runs both modes on every scene.

``colmap-matches`` (the GLOMAP paper's protocol): COLMAP's own SIFT features
and exhaustive matching with geometric verification, on the images resized to
the run's size, feed all three mappers. The visloc-rs mapper reads COLMAP's
keypoints and two-view inliers written in its replay format.

``visloc-matches``: the run's own keypoints and verified matches feed all
three. COLMAP re-verifies them (``verify_matches``) to build the two-view
geometries its mappers need, so its inliers are a subset of the run's.
``visloc-only`` scores just the visloc-rs mapper on them (no pycolmap
needed), e.g. to compare front-end settings.

Every method uses the run's single PINHOLE camera, held fixed. Each mode
prints one JSON line per mapper: registered images, AUC at 1/3/5 degrees of the
pairwise relative pose error (the larger of the rotation and the
translation-direction angle, over every pair of ground-truth images; a pair
with an unregistered image counts as a failure), camera-centre RMSE and median
after a Sim(3) fit (cm), and mapper wall time (s). Needs pycolmap >= 4.2,
numpy and Pillow.

GPU runs: ``VISLOC_BENCH_DEVICE=cuda`` runs COLMAP's SIFT extraction and
matching on the GPU, and ``VISLOC_BENCH_BA_BACKEND=CASPAR`` gives both COLMAP
mappers its GPU bundle adjuster (a pycolmap built from a COLMAP with
``CASPAR_ENABLED``). ``VISLOC_BENCH_THREADS`` overrides the thread count
(default: every CPU).
"""
from __future__ import annotations

import collections
import json
import os
import shutil
import struct
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_smallcity_reference import sim3_errors

THREADS = int(os.environ.get("VISLOC_BENCH_THREADS", os.cpu_count()))
DEVICE = os.environ.get("VISLOC_BENCH_DEVICE", "cpu")
BA_BACKEND = os.environ.get("VISLOC_BENCH_BA_BACKEND", "CERES")
MAPPERS = ["visloc-rs", "colmap-incremental", "colmap-global"]


def quat_to_r(qw, qx, qy, qz):
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])


def read_images_txt(path):
    """Image name -> (R, t), world to camera, from a COLMAP ``images.txt``."""
    with open(path) as f:
        rows = [line.split() for line in f if line.strip() and not line.startswith("#")]
    out = {}
    for r in rows[0::2]:
        if len(r) < 10:
            continue
        q = list(map(float, r[1:5]))
        out[os.path.basename(r[9])] = (quat_to_r(*q), np.array(list(map(float, r[5:8]))))
    return out


def angle_deg(a, b):
    c = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


def rot_err_deg(r):
    return np.degrees(np.arccos(np.clip((np.trace(r) - 1) / 2, -1.0, 1.0)))


def pose_auc(errors, thresholds):
    errors = np.sort(np.asarray(errors, dtype=float))
    recall = (np.arange(len(errors)) + 1) / len(errors)
    errors = np.r_[0.0, errors]
    recall = np.r_[0.0, recall]
    aucs = []
    for t in thresholds:
        last = np.searchsorted(errors, t)
        r = np.r_[recall[:last], recall[last - 1]]
        e = np.r_[errors[:last], t]
        aucs.append(float(np.trapezoid(r, x=e) / t))
    return aucs


def score(gt, est, seconds, method):
    """``gt``/``est``: name -> (R, t); ``est`` from the largest model."""
    names = sorted(gt)
    errs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            if a not in est or b not in est:
                errs.append(np.inf)
                continue
            (ra, ta), (rb, tb) = gt[a], gt[b]
            (ea, sa), (eb, sb) = est[a], est[b]
            r_gt = rb @ ra.T
            r_es = eb @ ea.T
            errs.append(max(rot_err_deg(r_es.T @ r_gt), angle_deg(sb - r_es @ sa, tb - r_gt @ ta)))
    auc = pose_auc(errs, [1, 3, 5])
    common = [n for n in names if n in est]
    rmse = med = float("nan")
    if len(common) >= 3:
        err, _ = sim3_errors(np.array([-gt[n][0].T @ gt[n][1] for n in common]),
                             np.array([-est[n][0].T @ est[n][1] for n in common]))
        rmse, med = 100 * float(np.sqrt((err ** 2).mean())), 100 * float(np.median(err))
    return {"method": method, "registered": len(common), "images": len(names),
            "auc1": round(100 * auc[0], 2), "auc3": round(100 * auc[1], 2),
            "auc5": round(100 * auc[2], 2), "rmse_cm": round(rmse, 3),
            "median_cm": round(med, 3), "seconds": round(seconds, 2)}


def from_recs(recs, name_map):
    if not recs:
        return {}
    rec = max(recs.values(), key=lambda r: r.num_reg_images())
    out = {}
    for im in rec.images.values():
        if im.has_pose:
            p = im.cam_from_world()
            out[name_map(im.name)] = (p.rotation.matrix(), np.array(p.translation))
    return out


def run_visloc(mapper, port, out, orig_of, log):
    """Replays the visloc-rs mapper on a port directory; the largest model."""
    shutil.rmtree(out, ignore_errors=True)
    t0 = time.time()
    with open(log, "w") as stderr:
        subprocess.run([mapper, "--manifest", os.path.join(port, "manifest.txt"),
                        "--features-dir", os.path.join(port, "features"),
                        "--pairs-export", os.path.join(port, "pairs.bin"), "--out-colmap", out],
                       check=True, stdout=subprocess.DEVNULL, stderr=stderr)
    seconds = time.time() - t0
    best = {}
    for m in os.listdir(os.path.join(out, "model")):
        est = read_images_txt(os.path.join(out, "model", m, "images.txt"))
        if len(est) > len(best):
            best = est
    return {orig_of[n]: v for n, v in best.items()}, seconds


def incremental_options(pycolmap, threads):
    """COLMAP's incremental mapper with the camera held fixed."""
    inc = pycolmap.IncrementalPipelineOptions()
    inc.ba_refine_focal_length = False
    inc.ba_refine_principal_point = False
    inc.ba_refine_extra_params = False
    inc.num_threads = threads
    inc.extract_colors = False
    if BA_BACKEND != "CERES":
        inc.ba_local_backend = getattr(pycolmap.BundleAdjustmentBackend, BA_BACKEND)
        inc.ba_global_backend = getattr(pycolmap.BundleAdjustmentBackend, BA_BACKEND)
    return inc


def global_options(pycolmap, threads):
    """COLMAP's global mapper (GLOMAP) with the camera held fixed."""
    glo = pycolmap.GlobalPipelineOptions()
    glo.num_threads = threads
    ba = glo.mapper.bundle_adjustment
    ba.refine_focal_length = False
    ba.refine_principal_point = False
    ba.refine_extra_params = False
    if BA_BACKEND != "CERES":
        ba.backend = getattr(pycolmap.BundleAdjustmentBackend, BA_BACKEND)
    return glo


def colmap_mappers(pycolmap, db, image_dir, work, gt, name_map):
    inc = incremental_options(pycolmap, THREADS)
    out = os.path.join(work, "colmap_incremental")
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    t0 = time.time()
    recs = pycolmap.incremental_mapping(db, image_dir, out, inc)
    rows = [score(gt, from_recs(recs, name_map), time.time() - t0, "colmap-incremental")]

    glo = global_options(pycolmap, THREADS)
    out = os.path.join(work, "colmap_global")
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    t0 = time.time()
    recs = pycolmap.global_mapping(db, image_dir, out, glo)
    rows.append(score(gt, from_recs(recs, name_map), time.time() - t0, "colmap-global"))
    return rows


def read_pairs_export(path):
    """(image names, [(i, j, matches[m, 2])]) from a ``VISLOC-COLMAP-1`` file."""
    with open(path, "rb") as f:
        buf = f.read()
    p = 16
    (n_img,) = struct.unpack_from("<Q", buf, p); p += 8
    names = []
    for _ in range(n_img):
        (length,) = struct.unpack_from("<Q", buf, p); p += 8
        names.append(buf[p:p + length].decode()); p += length + 8
    (n_pairs,) = struct.unpack_from("<Q", buf, p); p += 8
    pairs = []
    for _ in range(n_pairs):
        a, b, m = struct.unpack_from("<QQQ", buf, p); p += 24
        pairs.append((a, b, np.frombuffer(buf, dtype="<u4", count=2 * m, offset=p).reshape(m, 2)))
        p += 8 * m
    return names, pairs


def write_port(port, w, h, k, names, keypoints, pairs, mapper_options):
    """A visloc-rs replay directory: manifest, ``*_features.txt``, ``pairs.bin``."""
    shutil.rmtree(port, ignore_errors=True)
    os.makedirs(os.path.join(port, "features"))
    with open(os.path.join(port, "manifest.txt"), "w") as f:
        f.write("# generalized-rig-manifest-v1\n")
        f.write(f"S 0 1 {w} {h} {k[0]} {k[1]} {k[2]} {k[3]} 1 0 0 0 0 0 0\n")
        f.writelines(f"F {i} {n} 0\n" for i, n in enumerate(names))
    shutil.copy(mapper_options, os.path.join(port, "mapper_options.txt"))
    for n, kp in zip(names, keypoints):
        np.savetxt(os.path.join(port, "features", os.path.splitext(n)[0] + "_features.txt"),
                   kp, fmt="%.6f")
    with open(os.path.join(port, "pairs.bin"), "wb") as f:
        f.write(b"VISLOC-COLMAP-1\0")
        f.write(struct.pack("<Q", len(names)))
        for n, kp in zip(names, keypoints):
            b = n.encode()
            f.write(struct.pack("<Q", len(b)) + b + struct.pack("<Q", len(kp)))
        f.write(struct.pack("<Q", len(pairs)))
        for a, b, m in sorted(pairs, key=lambda p: (p[0], p[1])):
            f.write(struct.pack("<QQQ", a, b, len(m)))
            f.write(np.ascontiguousarray(m, dtype="<u4").tobytes())


def port_database(pycolmap, port, db, work, reader, frames, size):
    """Write a replay directory's keypoints and matches into COLMAP's database,
    under blank images of the run's ``size`` (COLMAP imports images, not
    keypoints). Returns the image directory and the pair list for
    ``verify_matches``."""
    from PIL import Image

    image_dir = os.path.join(work, "blank")
    os.makedirs(image_dir, exist_ok=True)
    blank = Image.new("L", size)
    for fr in frames:
        blank.save(os.path.join(image_dir, fr))
    pycolmap.import_images(db, image_dir, camera_mode=pycolmap.CameraMode.SINGLE,
                           options=reader)
    d = pycolmap.Database.open(db)
    ids = {im.name: im.image_id for im in d.read_all_images()}
    for fr, i in ids.items():
        kp = np.loadtxt(os.path.join(port, "features", os.path.splitext(fr)[0] + "_features.txt"),
                        dtype=np.float32, ndmin=2)
        d.write_keypoints(i, kp[:, :2])
    names, pairs = read_pairs_export(os.path.join(port, "pairs.bin"))
    pair_lines = []
    for a, b, m in pairs:
        na, nb = names[a], names[b]
        m = m.astype(np.uint32)
        if ids[na] > ids[nb]:
            na, nb, m = nb, na, m[:, ::-1].copy()
        d.write_matches(ids[na], ids[nb], m)
        pair_lines.append(f"{na} {nb}\n")
    d.close()
    pairs_txt = os.path.join(work, "pairs.txt")
    with open(pairs_txt, "w") as f:
        f.writelines(pair_lines)
    return image_dir, pairs_txt


def visloc_only(scene, run, work, mapper):
    """The visloc-rs mapper alone on the run's own matches."""
    os.makedirs(work, exist_ok=True)
    name = os.path.basename(os.path.normpath(scene))
    gt = read_images_txt(os.path.join(scene, "dslr_calibration_undistorted", "images.txt"))
    orig_of = {}
    with open(os.path.join(run, "image_names.txt")) as f:
        for line in f:
            if line.strip():
                fr, path = line.split()
                orig_of[fr] = os.path.basename(path)
    est, seconds = run_visloc(mapper, os.path.join(run, "port"), os.path.join(work, "visloc"),
                              orig_of, os.path.join(work, "visloc.log"))
    print(json.dumps({"scene": name, **score(gt, est, seconds, "visloc-rs")}), flush=True)


def bench(mode, scene, run, work, mapper):
    import pycolmap
    from PIL import Image

    os.makedirs(work, exist_ok=True)
    name = os.path.basename(os.path.normpath(scene))
    gt = read_images_txt(os.path.join(scene, "dslr_calibration_undistorted", "images.txt"))
    with open(os.path.join(run, "port", "manifest.txt")) as f:
        srow = next(line.split() for line in f if line.startswith("S "))
    w, h = int(srow[3]), int(srow[4])
    k = tuple(map(float, srow[5:9]))
    frame_of = {}
    with open(os.path.join(run, "image_names.txt")) as f:
        for line in f:
            if line.strip():
                fr, path = line.split()
                frame_of[os.path.basename(path)] = fr

    def emit(row):
        print(json.dumps({"scene": name, **row}), flush=True)
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "PINHOLE"
    reader.camera_params = ",".join(map(str, k))
    db = os.path.join(work, "database.db")
    if os.path.exists(db):
        os.remove(db)
    pycolmap.Database.open(db).close()

    if mode == "visloc-matches":
        orig_of = {fr: n for n, fr in frame_of.items()}
        port = os.path.join(run, "port")
        est, seconds = run_visloc(mapper, port, os.path.join(work, "visloc"), orig_of,
                                  os.path.join(work, "visloc.log"))
        emit(score(gt, est, seconds, "visloc-rs"))
        image_dir, pairs_txt = port_database(pycolmap, port, db, work, reader, orig_of, (w, h))
        tvg = pycolmap.TwoViewGeometryOptions()
        tvg.compute_relative_pose = True
        pycolmap.verify_matches(db, pairs_txt, options=tvg)
        for row in colmap_mappers(pycolmap, db, image_dir, work, gt, lambda n: orig_of[n]):
            emit(row)
        return

    # colmap-matches: COLMAP SIFT + exhaustive matching on resized images.
    src = os.path.join(scene, "images", "dslr_images_undistorted")
    image_dir = os.path.join(work, "images")
    if not os.path.isdir(image_dir):
        os.makedirs(image_dir)
        for n in sorted(os.listdir(src)):
            Image.open(os.path.join(src, n)).convert("RGB").resize((w, h), Image.LANCZOS).save(
                os.path.join(image_dir, n), quality=95)
    ext = pycolmap.FeatureExtractionOptions()
    ext.max_image_size = max(w, h)
    ext.num_threads = THREADS
    t0 = time.time()
    pycolmap.extract_features(db, image_dir, camera_mode=pycolmap.CameraMode.SINGLE,
                              reader_options=reader, extraction_options=ext,
                              device=getattr(pycolmap.Device, DEVICE))
    t_extract = time.time() - t0
    matching = pycolmap.FeatureMatchingOptions()
    matching.num_threads = THREADS
    t0 = time.time()
    pycolmap.match_exhaustive(db, matching_options=matching,
                              device=getattr(pycolmap.Device, DEVICE))
    emit({"method": "colmap-features", "extract_s": round(t_extract, 1),
          "match_s": round(time.time() - t0, 1)})
    for row in colmap_mappers(pycolmap, db, image_dir, work, gt, lambda n: n):
        emit(row)

    d = pycolmap.Database.open(db)
    images = sorted(d.read_all_images(), key=lambda im: im.name)
    index_of = {im.image_id: i for i, im in enumerate(images)}
    frames = [f"frame_{i:05d}.png" for i in range(len(images))]
    keypoints = [d.read_keypoints(im.image_id)[:, :2] for im in images]
    pairs = []
    pair_ids, geometries = d.read_two_view_geometries()
    for pid, tvg in zip(pair_ids, geometries):
        m = np.asarray(tvg.inlier_matches, dtype=np.uint32).reshape(-1, 2)
        if len(m):
            i, j = pycolmap.pair_id_to_image_pair(pid)
            a, b = index_of[i], index_of[j]
            if a > b:
                a, b, m = b, a, m[:, ::-1]
            pairs.append((a, b, m))
    d.close()
    port = os.path.join(work, "port")
    write_port(port, w, h, k, frames, keypoints, pairs,
               os.path.join(run, "port", "mapper_options.txt"))
    orig_of = {fr: im.name for fr, im in zip(frames, images)}
    est, seconds = run_visloc(mapper, port, os.path.join(work, "visloc"), orig_of,
                              os.path.join(work, "visloc.log"))
    emit(score(gt, est, seconds, "visloc-rs"))


def table(path, markdown):
    by = collections.defaultdict(dict)
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if r.get("method") in MAPPERS:
                    by[r["scene"]][r["method"]] = r
    totals = {m: collections.Counter() for m in MAPPERS}
    if markdown:
        print("| Scene | " + " | ".join(f"{m} AUC@1/3/5 (reg.)" for m in MAPPERS) + " |")
        print("|---|" + "---|" * len(MAPPERS))
    for scene in sorted(by):
        cells = []
        for m in MAPPERS:
            r = by[scene].get(m)
            if r is None:
                cells.append("–")
                continue
            for key in ("auc1", "auc3", "auc5", "seconds", "registered", "images"):
                totals[m][key] += r[key]
            totals[m]["n"] += 1
            cells.append(f"{r['auc1']:.1f} / {r['auc3']:.1f} / {r['auc5']:.1f} "
                         f"({r['registered']}/{r['images']})")
        print(("| " + scene + " | " + " | ".join(cells) + " |") if markdown
              else f"{scene:14s} " + "  ".join(f"{c:28s}" for c in cells))
    print()
    for m in MAPPERS:
        t = totals[m]
        n = t["n"]
        if n == 0:
            continue
        print(f"{m}: mean AUC@1/3/5 {t['auc1'] / n:.2f} / {t['auc3'] / n:.2f} / "
              f"{t['auc5'] / n:.2f}, registered {t['registered']}/{t['images']}, "
              f"mapper time {t['seconds']:.1f} s over {t['n']} scenes")


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "table":
        table(sys.argv[2], "--markdown" in sys.argv[3:])
    elif len(sys.argv) == 6 and sys.argv[1] in ("colmap-matches", "visloc-matches"):
        bench(*sys.argv[1:6])
    elif len(sys.argv) == 6 and sys.argv[1] == "visloc-only":
        visloc_only(*sys.argv[2:6])
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
