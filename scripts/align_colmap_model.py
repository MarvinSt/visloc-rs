#!/usr/bin/env python3
"""Move a COLMAP text model into a reference model's frame.

    align_colmap_model.py EST_DIR REF_IMAGES_TXT OUT_DIR [--names IMAGE_NAMES_TXT]

Fits the Sim(3) taking EST_DIR's camera centres onto those of the
reference images with the same file names (compared without directories),
refits once without the cameras more than three times the median error
off, and writes the model with its poses and points transformed to
OUT_DIR. Observations and tracks are kept as they are, so a dense
reconstruction from OUT_DIR lands in the reference frame (ETH3D's laser
scans, for example). With ``--names`` (a ``gsplat_photos`` run's
``image_names.txt``: working name, original path), images are matched and
written under their original file names. Prints the fit: matched cameras, scale and centre
errors in reference units. Needs numpy.
"""
from __future__ import annotations

import os
import shutil
import sys

import numpy as np


def quat_to_r(qw, qx, qy, qz):
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])


def r_to_quat(r):
    w = np.sqrt(max(0.0, 1 + r[0, 0] + r[1, 1] + r[2, 2])) / 2
    x = np.sqrt(max(0.0, 1 + r[0, 0] - r[1, 1] - r[2, 2])) / 2
    y = np.sqrt(max(0.0, 1 - r[0, 0] + r[1, 1] - r[2, 2])) / 2
    z = np.sqrt(max(0.0, 1 - r[0, 0] - r[1, 1] + r[2, 2])) / 2
    x = np.copysign(x, r[2, 1] - r[1, 2])
    y = np.copysign(y, r[0, 2] - r[2, 0])
    z = np.copysign(z, r[1, 0] - r[0, 1])
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def read_images(path):
    """(header tokens, observation line) per image, in file order."""
    images = []
    with open(path) as f:
        lines = [l.rstrip("\n") for l in f if not l.startswith("#")]
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        tokens = lines[i].split()
        obs = lines[i + 1] if i + 1 < len(lines) else ""
        images.append((tokens, obs))
        i += 2
    return images


def centre(tokens):
    r = quat_to_r(*map(float, tokens[1:5]))
    t = np.array(list(map(float, tokens[5:8])))
    return -r.T @ t


def sim3(ref, est):
    """(scale, rotation, translation) with ref ≈ s R est + t."""
    mr, me = ref.mean(0), est.mean(0)
    r0, e0 = ref - mr, est - me
    u, s, vt = np.linalg.svd(r0.T @ e0)
    d = np.eye(3)
    d[2, 2] = np.sign(np.linalg.det(u @ vt))
    rot = u @ d @ vt
    scale = (s * np.diag(d)).sum() / (e0**2).sum()
    return scale, rot, mr - scale * rot @ me


def main():
    args = sys.argv[1:]
    original = {}
    if "--names" in args:
        k = args.index("--names")
        with open(args[k + 1]) as f:
            for line in f:
                if line.strip():
                    work, path = line.rstrip("\n").split(" ", 1)
                    original[work] = os.path.basename(path)
        del args[k:k + 2]
    if len(args) != 3:
        raise SystemExit(__doc__)
    est_dir, ref_path, out_dir = args
    est = read_images(os.path.join(est_dir, "images.txt"))
    for tokens, _ in est:
        name = " ".join(tokens[9:])
        tokens[9:] = [original.get(name, name)]
    ref = {os.path.basename(" ".join(t[9:])): centre(t) for t, _ in read_images(ref_path)}
    names = [os.path.basename(" ".join(t[9:])) for t, _ in est]
    common = [k for k, n in enumerate(names) if n in ref]
    if len(common) < 3:
        raise SystemExit(f"only {len(common)} images in common with the reference")
    a = np.array([ref[names[k]] for k in common])
    b = np.array([centre(est[k][0]) for k in common])
    s, rot, t = sim3(a, b)
    err = np.linalg.norm(a - (s * b @ rot.T + t), axis=1)
    keep = err <= 3 * np.median(err) + 1e-12
    if not keep.all() and keep.sum() >= 3:
        s, rot, t = sim3(a[keep], b[keep])
        err = np.linalg.norm(a - (s * b @ rot.T + t), axis=1)

    os.makedirs(out_dir, exist_ok=True)
    shutil.copy(os.path.join(est_dir, "cameras.txt"), os.path.join(out_dir, "cameras.txt"))
    with open(os.path.join(out_dir, "images.txt"), "w") as f:
        f.write("# Image list (aligned)\n")
        for tokens, obs in est:
            r = quat_to_r(*map(float, tokens[1:5]))
            c = s * rot @ centre(tokens) + t
            r_new = r @ rot.T
            q = r_to_quat(r_new)
            t_new = -r_new @ c
            f.write(" ".join([tokens[0], *(f"{v:.17g}" for v in q), *(f"{v:.17g}" for v in t_new), *tokens[8:]]) + "\n")
            f.write(obs + "\n")
    with open(os.path.join(est_dir, "points3D.txt")) as src, open(os.path.join(out_dir, "points3D.txt"), "w") as f:
        for line in src:
            if line.startswith("#") or not line.strip():
                f.write(line)
                continue
            tokens = line.split()
            x = s * rot @ np.array(list(map(float, tokens[1:4]))) + t
            f.write(" ".join([tokens[0], *(f"{v:.17g}" for v in x), *tokens[4:]]) + "\n")
    print(f"aligned {len(common)} of {len(est)} images to the reference ({len(ref)}): "
          f"scale {s:.6g}, centre error median {np.median(err):.4f}, max {err.max():.4f} "
          f"({int((~keep).sum())} left out of the fit)")


if __name__ == "__main__":
    main()
