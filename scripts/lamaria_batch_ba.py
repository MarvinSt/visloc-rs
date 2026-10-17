#!/usr/bin/env python3
"""Offline visual-inertial bundle adjustment of a LaMAria VIO trajectory.

The Basalt VIO keeps a sliding window of short KLT tracks, so its heading and
scale drift slowly over a sequence. This tool refines the VIO trajectory
afterwards with one batch optimization over the whole sequence:

  features  SIFT on every --kf-step-th VIO frame, both cameras.
  match     epipolar matching between keyframes up to 64 keyframes (32 s at
            the default 2 Hz) apart, guided by the VIO's relative poses.
  solve     union-find tracks -> midpoint triangulation -> Levenberg-Marquardt
            over every keyframe's pose, velocity and IMU biases, every
            landmark, and each camera's IMU-camera rotation, with GTSAM's
            preintegrated IMU factors and Huber reprojection terms. The
            landmarks are eliminated by the Schur complement; because every
            landmark is seen within --band keyframes, the reduced system is
            banded and is factored by LAPACK's banded Cholesky, with the
            camera rotations as a dense border.

The output is the VIO's per-frame trajectory with each frame moved by the
correction of the surrounding keyframes (TUM, seconds, world_from_imu), plus a
JSON summary with the estimated camera rotations. No ground truth is used.

Requirements: numpy, scipy, opencv-python (SIFT), gtsam (IMU factors); numba is optional
and makes the matching about 30 times faster.
The cameras must be pinhole (the ASL release's undistorted images; a Basalt
double-sphere model with xi = alpha = 0).

Revisits (optional): loops matches places seen again; loop-correct turns those
matches into PnP relative poses and closes them in a pose graph over a solved
trajectory; a second solve started from the corrected trajectory then adds the
revisit matches themselves.

usage:
  lamaria_batch_ba.py features --mav0 DIR --vio TUM --work DIR [--kf-step 10]
  lamaria_batch_ba.py match --work DIR --vio TUM --calib JSON
  lamaria_batch_ba.py solve --work DIR --imu CSV --calib JSON --vio TUM --out TUM
with the revisits:
  lamaria_batch_ba.py loops --work DIR --vio TUM --calib JSON
  lamaria_batch_ba.py solve ... --vio TUM --out BA1.tum --save-states
  lamaria_batch_ba.py loop-correct --work DIR --calib JSON --states BA1_states.npz --tum BA1.tum --out LC.tum
  lamaria_batch_ba.py solve ... --vio LC.tum --out TUM --loops
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

DOF = 15  # per keyframe: pose 6 (rotation, translation), velocity 3, bias 6 (accel, gyro)


# ----------------------------------------------------------------------------- io

def quat_to_rot(q):
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def load_calib(path):
    """Pinhole intrinsics, IMU-camera extrinsics and IMU noise of a Basalt calibration."""
    v = json.load(open(path))["value0"]
    cams = []
    for c in range(len(v["T_imu_cam"])):
        e = v["T_imu_cam"][c]
        intr = v["intrinsics"][c]["intrinsics"]
        if abs(intr.get("xi", 0.0)) > 0 or abs(intr.get("alpha", 0.0)) > 0:
            sys.exit("only pinhole cameras (double sphere with xi = alpha = 0) are supported")
        cams.append({
            "R_ic": quat_to_rot(np.array([e["qx"], e["qy"], e["qz"], e["qw"]])),
            "t_ic": np.array([e["px"], e["py"], e["pz"]]),
            "K": np.array([[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1.0]]),
        })
    if len(cams) != 2:
        sys.exit("a two-camera calibration is expected")
    noise = {k: float(v[k][0]) for k in ("accel_noise_std", "gyro_noise_std", "accel_bias_std", "gyro_bias_std")}
    return cams, noise


def load_tum(path):
    d = np.loadtxt(path)
    ns = np.round(d[:, 0] * 1e9).astype(np.int64)
    rot = np.array([quat_to_rot(q) for q in d[:, 4:8]])
    return ns, d[:, 1:4].copy(), rot


def load_features(work):
    f = np.load(os.path.join(work, "features.npz"))
    return f["kf_ns"], f["offsets"], f["xy"], f["desc"]


def keyframe_poses(kf_ns, vio):
    vio_ns, vio_p, vio_r = load_tum(vio)
    idx = np.searchsorted(vio_ns, kf_ns)
    if np.any(idx >= len(vio_ns)) or np.any(vio_ns[np.minimum(idx, len(vio_ns) - 1)] != kf_ns):
        sys.exit("every keyframe must have a VIO pose with the same timestamp")
    return vio_p[idx], vio_r[idx]


# ----------------------------------------------------------------------------- features

def _sift_worker(task):
    import cv2
    path, nfeat, clahe = task
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if clahe:  # low light: local contrast equalization before detection
        im = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(im)
    sift = cv2.SIFT_create(nfeatures=nfeat, contrastThreshold=0.01)
    kp, des = sift.detectAndCompute(im, None)
    xy = np.array([k.pt for k in kp], dtype=np.float32).reshape(-1, 2)
    des = np.zeros((0, 128), np.uint8) if des is None else np.clip(des, 0, 255).astype(np.uint8)
    return xy, des


def cmd_features(a):
    os.makedirs(a.work, exist_ok=True)
    vio_ns, _, _ = load_tum(a.vio)
    frames = []
    for c in range(2):
        names = os.listdir(os.path.join(a.mav0, f"cam{c}", "data"))
        frames.append({int(n[:-4]): n for n in names if n.endswith(".png")})
    vio_set = set(vio_ns.tolist())
    usable = [t for t in sorted(set(frames[0]) & set(frames[1])) if t in vio_set]
    kf_ns = np.array(usable[::a.kf_step], dtype=np.int64)
    print(f"{len(usable)} stereo frames with a VIO pose, {len(kf_ns)} keyframes", flush=True)
    tasks = [(os.path.join(a.mav0, f"cam{c}", "data", frames[c][int(t)]), a.num_features, a.clahe)
             for t in kf_ns for c in range(2)]
    t0 = time.time()
    with mp.Pool(a.jobs) as pool:
        res = pool.map(_sift_worker, tasks, chunksize=8)
    counts = np.array([len(r[0]) for r in res], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(counts)])
    np.savez(os.path.join(a.work, "features.npz"), kf_ns=kf_ns, offsets=offsets,
             xy=np.concatenate([r[0] for r in res]), desc=np.concatenate([r[1] for r in res]))
    print(f"features: {offsets[-1]} keypoints, {counts.mean():.0f}/image, {time.time() - t0:.0f} s", flush=True)


# ----------------------------------------------------------------------------- matching

_G = {}  # read-only matching state shared with the forked workers


def _cam_pose(k, c):
    """world_from_camera of keyframe k, camera c."""
    r_wi, t_wi = _G["kf_r"][k], _G["kf_p"][k]
    cam = _G["cams"][c]
    return r_wi @ cam["R_ic"], r_wi @ cam["t_ic"] + t_wi


def _image(k, c):
    i = 2 * k + c
    s, e = _G["offsets"][i], _G["offsets"][i + 1]
    return _G["xy"][s:e], _G["desc"][s:e]


def _best_two(keys, vals, n):
    """Per key: index (into keys/vals) of the smallest value, and the second smallest value."""
    order = np.lexsort((vals, keys))
    k_sorted = keys[order]
    first = np.flatnonzero(np.r_[True, k_sorted[1:] != k_sorted[:-1]])
    best = np.full(n, -1, np.int64)
    second_val = np.full(n, np.inf, np.float32)
    best[k_sorted[first]] = order[first]
    nxt = np.minimum(first + 1, len(order) - 1)
    same = (first + 1 < len(order)) & (k_sorted[nxt] == k_sorted[first])
    second_val[k_sorted[first[same]]] = vals[order[nxt[same]]]
    return best, second_val



try:  # optional: the same matching rules ~30x faster
    from numba import njit
except ImportError:
    njit = None

if njit is not None:
    @njit(cache=True, nogil=True)
    def _match_kernel(xa, xb, da, db, mat, use_rot, tau_n, ratio, max_desc):
        n, m = xa.shape[0], xb.shape[0]
        inf = np.float32(np.inf)
        best_j = np.full(n, -1, np.int64)
        best_d = np.full(n, inf, np.float32)
        second_d = np.full(n, inf, np.float32)
        bbest_i = np.full(m, -1, np.int64)
        bbest_d = np.full(m, inf, np.float32)
        for i in range(n):
            l0 = mat[0, 0] * xa[i, 0] + mat[0, 1] * xa[i, 1] + mat[0, 2] * xa[i, 2]
            l1 = mat[1, 0] * xa[i, 0] + mat[1, 1] * xa[i, 1] + mat[1, 2] * xa[i, 2]
            l2 = mat[2, 0] * xa[i, 0] + mat[2, 1] * xa[i, 1] + mat[2, 2] * xa[i, 2]
            if not use_rot:  # epipolar line in b
                nrm = np.sqrt(l0 * l0 + l1 * l1)
                l0, l1, l2 = l0 / nrm, l1 / nrm, l2 / nrm
            else:  # rotation-only prediction in b
                l2 = max(l2, np.float32(1e-3))
                l0, l1 = l0 / l2, l1 / l2
            for j in range(m):
                if not use_rot:
                    dist = abs(l0 * xb[j, 0] + l1 * xb[j, 1] + l2 * xb[j, 2])
                else:
                    dx, dy = l0 - xb[j, 0], l1 - xb[j, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                if dist < tau_n:
                    acc = 0
                    for k in range(128):
                        d = np.int32(da[i, k]) - np.int32(db[j, k])
                        acc += d * d
                    dd = np.sqrt(np.float32(acc))
                    if dd < best_d[i]:
                        second_d[i] = best_d[i]
                        best_d[i] = dd
                        best_j[i] = j
                    elif dd < second_d[i]:
                        second_d[i] = dd
                    if dd < bbest_d[j]:
                        bbest_d[j] = dd
                        bbest_i[j] = i
        cnt = 0
        out = np.empty((n, 2), np.int32)
        for i in range(n):
            j = best_j[i]
            if j >= 0 and best_d[i] < ratio * second_d[i] and best_d[i] < max_desc and bbest_i[j] == i:
                out[cnt, 0], out[cnt, 1] = i, j
                cnt += 1
        return out[:cnt]

def _match_worker(pair):
    """Mutual, ratio-tested SIFT matches restricted to the VIO's epipolar band."""
    ka, ca, kb, cb, tau = pair
    xya, da = _image(ka, ca)
    xyb, db = _image(kb, cb)
    empty = np.zeros((0, 2), np.int32)
    if len(xya) == 0 or len(xyb) == 0:
        return pair, empty
    r_wa, t_wa = _cam_pose(ka, ca)
    r_wb, t_wb = _cam_pose(kb, cb)
    r_ba = r_wb.T @ r_wa
    t_ba = r_wb.T @ (t_wa - t_wb)
    fb = _G["cams"][cb]["K"][0, 0]
    xa = (np.c_[xya, np.ones(len(xya))] @ np.linalg.inv(_G["cams"][ca]["K"]).T).astype(np.float32)
    xb = (np.c_[xyb, np.ones(len(xyb))] @ np.linalg.inv(_G["cams"][cb]["K"]).T).astype(np.float32)
    use_rot = np.linalg.norm(t_ba) <= 0.05
    if njit is not None:
        mat = r_ba if use_rot else np.array([[0, -t_ba[2], t_ba[1]], [t_ba[2], 0, -t_ba[0]],
                                             [-t_ba[1], t_ba[0], 0]]) @ r_ba
        return pair, _match_kernel(xa, xb, np.ascontiguousarray(da), np.ascontiguousarray(db),
                                   mat.astype(np.float32), use_rot, np.float32(tau / fb),
                                   np.float32(_G["ratio"]), np.float32(_G["max_desc"]))
    if not use_rot:
        e = np.array([[0, -t_ba[2], t_ba[1]], [t_ba[2], 0, -t_ba[0]], [-t_ba[1], t_ba[0], 0]]) @ r_ba
        lines = xa @ e.T.astype(np.float32)
        lines /= np.hypot(lines[:, 0], lines[:, 1])[:, None]
        dist = np.abs(lines @ xb.T)
    else:  # no baseline: predict with the rotation alone
        pred = xa @ r_ba.T.astype(np.float32)
        pred = pred[:, :2] / np.maximum(pred[:, 2:3], 1e-3)
        dist = np.sqrt(((pred[:, None, :] - xb[None, :, :2]) ** 2).sum(2))
    ia, ib = np.nonzero(dist < tau / fb)
    if len(ia) == 0:
        return pair, empty
    diff = da[ia].astype(np.int32) - db[ib].astype(np.int32)
    dd = np.sqrt((diff * diff).sum(1).astype(np.float32))
    best_a, second_a = _best_two(ia, dd, len(xya))
    best_b, _ = _best_two(ib, dd, len(xyb))
    rows = np.flatnonzero(best_a >= 0)
    c = best_a[rows]
    d1 = dd[c]
    j = ib[c]
    ok = (d1 < _G["ratio"] * second_a[rows]) & (d1 < _G["max_desc"]) & (ia[best_b[j]] == rows)
    return pair, np.c_[rows[ok], j[ok]].astype(np.int32)


def cmd_match(a):
    kf_ns, offsets, xy, desc = load_features(a.work)
    cams, _ = load_calib(a.calib)
    kf_p, kf_r = keyframe_poses(kf_ns, a.vio)
    _G.update(offsets=offsets, xy=xy, desc=desc, cams=cams, kf_p=kf_p, kf_r=kf_r,
              ratio=a.ratio, max_desc=a.max_desc)
    offs = [int(x) for x in a.offsets.split(",")]
    cos_max = np.cos(np.radians(a.max_axis_angle))
    pairs = []
    n = len(kf_ns)
    for ka in range(n):
        for d in offs:
            kb = ka + d
            if kb >= n:
                break
            for ca in range(2):
                ra, ta = _cam_pose(ka, ca)
                for cb in range(2):
                    rb, tb = _cam_pose(kb, cb)
                    if ra[:, 2] @ rb[:, 2] < cos_max or np.linalg.norm(ta - tb) > a.max_dist:
                        continue
                    pairs.append((ka, ca, kb, cb, min(a.tau0 + a.tau_slope * d, a.tau_max)))
    print(f"{n} keyframes, {len(pairs)} image pairs to match", flush=True)
    t0 = time.time()
    out_pairs, out_matches = [], []
    with mp.Pool(a.jobs) as pool:
        for i, (pair, m) in enumerate(pool.imap(_match_worker, pairs, chunksize=16)):
            if len(m):
                out_pairs.append(np.array(pair[:4], np.int32)[None].repeat(len(m), 0))
                out_matches.append(m)
            if i % 20000 == 0:
                print(f"  {i}/{len(pairs)} pairs, {time.time() - t0:.0f} s", flush=True)
    np.savez(os.path.join(a.work, "matches.npz"), pairs=np.concatenate(out_pairs),
             matches=np.concatenate(out_matches))
    print(f"matches: {sum(len(m) for m in out_matches)} over {len(out_pairs)} pairs, "
          f"{time.time() - t0:.0f} s", flush=True)


# ----------------------------------------------------------------------------- revisits

def _unguided_matches(da, db, ratio, max_desc):
    """Mutual nearest neighbours with the ratio test, over all features."""
    if njit is not None:
        ones = np.zeros((len(da), 3), np.float32)
        return _match_kernel(ones, np.zeros((len(db), 3), np.float32), np.ascontiguousarray(da),
                             np.ascontiguousarray(db), np.eye(3, dtype=np.float32), True, np.float32(1e9),
                             np.float32(ratio), np.float32(max_desc))
    fa, fb = da.astype(np.float32), db.astype(np.float32)
    d2 = np.maximum((fa * fa).sum(1)[:, None] + (fb * fb).sum(1)[None, :] - 2 * fa @ fb.T, 0)
    two = np.argpartition(d2, 1, axis=1)[:, :2]
    rows = np.arange(len(fa))
    d_two = d2[rows[:, None], two]
    o = np.argsort(d_two, axis=1)
    j = two[rows, o[:, 0]]
    d1, d2nd = np.sqrt(d_two[rows, o[:, 0]]), np.sqrt(d_two[rows, o[:, 1]])
    ok = (d1 < ratio * d2nd) & (d1 < max_desc) & (np.argmin(d2, axis=0)[j] == rows)
    return np.c_[rows[ok], j[ok]].astype(np.int32)


def _loop_worker(task):
    """A revisit candidate: unguided matching, essential-matrix RANSAC, and a rotation check."""
    import cv2
    ka, ca, kb, cb = task
    xya, da = _image(ka, ca)
    xyb, db = _image(kb, cb)
    empty = np.zeros((0, 2), np.int32)
    if len(xya) < _G["min_inliers"] or len(xyb) < _G["min_inliers"]:
        return task, empty, 0
    m = _unguided_matches(da, db, _G["ratio"], _G["max_desc"])
    if len(m) < _G["min_inliers"]:
        return task, empty, 0
    pa = (np.c_[xya[m[:, 0]], np.ones(len(m))] @ np.linalg.inv(_G["cams"][ca]["K"]).T)[:, :2]
    pb = (np.c_[xyb[m[:, 1]], np.ones(len(m))] @ np.linalg.inv(_G["cams"][cb]["K"]).T)[:, :2]
    e, mask = cv2.findEssentialMat(pa, pb, np.eye(3), method=cv2.RANSAC, prob=0.999,
                                   threshold=1.0 / _G["cams"][cb]["K"][0, 0])
    if e is None or mask is None or e.shape != (3, 3) or mask.sum() < _G["min_inliers"]:
        return task, empty, 0
    mask = mask.ravel().astype(bool)
    _, r_est, _, _ = cv2.recoverPose(e, pa[mask], pb[mask], np.eye(3))
    r_ba = _cam_pose(kb, cb)[0].T @ _cam_pose(ka, ca)[0]
    angle = np.degrees(np.arccos(np.clip((np.trace(r_est.T @ r_ba) - 1) / 2, -1, 1)))
    if angle > _G["max_rot"]:
        return task, empty, -1
    return task, m[mask], int(mask.sum())


def cmd_loops(a):
    kf_ns, offsets, xy, desc = load_features(a.work)
    cams, _ = load_calib(a.calib)
    kf_p, kf_r = keyframe_poses(kf_ns, a.vio)
    _G.update(offsets=offsets, xy=xy, desc=desc, cams=cams, kf_p=kf_p, kf_r=kf_r, ratio=a.ratio,
              max_desc=a.max_desc, min_inliers=a.min_inliers, max_rot=a.max_rot)
    n = len(kf_ns)
    axes = np.array([[_cam_pose(k, c)[0][:, 2] for c in range(2)] for k in range(n)])
    ks = np.arange(0, n, a.step)
    dist = np.linalg.norm(kf_p[ks][:, None, :2] - kf_p[ks][None, :, :2], axis=2)
    ia, ib = np.nonzero((ks[None, :] - ks[:, None] > a.min_gap) & (dist < a.max_dist))
    cos_max = np.cos(np.radians(a.max_axis_angle))
    tasks = [(int(ks[i]), ca, int(ks[j]), cb) for i, j in zip(ia, ib) for ca in range(2) for cb in range(2)
             if axes[ks[i], ca] @ axes[ks[j], cb] >= cos_max]
    print(f"{len(ia)} revisit keyframe pairs, {len(tasks)} image pairs to verify", flush=True)
    t0 = time.time()
    out_pairs, out_matches, by_rotation = [], [], 0
    with mp.Pool(a.jobs) as pool:
        for task, m, ninl in pool.imap(_loop_worker, tasks, chunksize=8):
            by_rotation += ninl < 0
            if len(m):
                out_pairs.append(np.array(task, np.int32)[None].repeat(len(m), 0))
                out_matches.append(m)
    pairs = np.concatenate(out_pairs) if out_pairs else np.zeros((0, 4), np.int32)
    matches = np.concatenate(out_matches) if out_matches else np.zeros((0, 2), np.int32)
    np.savez(os.path.join(a.work, "loops.npz"), pairs=pairs, matches=matches)
    print(f"loops: {len(matches)} matches over {len(out_pairs)} verified image pairs "
          f"({len(np.unique(pairs[:, [0, 2]], axis=0))} keyframe pairs, {by_rotation} rejected by rotation), "
          f"{time.time() - t0:.0f} s", flush=True)


# ----------------------------------------------------------------------------- tracks

def build_tracks(offsets, pairs, matches, max_offset):
    """Connected components of the match graph; components with two features in one image are dropped."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    keep = (pairs[:, 2] - pairs[:, 0]) <= max_offset
    pairs, matches = pairs[keep], matches[keep]
    ga = offsets[2 * pairs[:, 0] + pairs[:, 1]] + matches[:, 0]
    gb = offsets[2 * pairs[:, 2] + pairs[:, 3]] + matches[:, 1]
    n = int(offsets[-1])
    _, roots = connected_components(coo_matrix((np.ones(len(ga), np.int8), (ga, gb)), shape=(n, n)),
                                    directed=False)
    image_of = np.repeat(np.arange(len(offsets) - 1), np.diff(offsets))
    order = np.argsort(roots, kind="stable")
    r_sorted = roots[order]
    starts = np.flatnonzero(np.r_[True, r_sorted[1:] != r_sorted[:-1]])
    ends = np.r_[starts[1:], len(order)]
    tracks, conflicts = [], 0
    for s, e in zip(starts, ends):
        if e - s < 2:
            continue
        feats = order[s:e]
        imgs = image_of[feats]
        if len(np.unique(imgs)) != len(imgs):
            conflicts += 1
            continue
        tracks.append(feats)
    return tracks, image_of, conflicts


def triangulate(tracks, image_of, xy, kf_p, kf_r, cams, max_err, min_angle_deg):
    """Midpoint triangulation with the VIO poses; drops observations off by more than max_err pixels."""
    nimg = len(kf_p) * 2
    centers = np.zeros((nimg, 3))
    rots = np.zeros((nimg, 3, 3))
    for k in range(len(kf_p)):
        for c in range(2):
            rots[2 * k + c] = kf_r[k] @ cams[c]["R_ic"]
            centers[2 * k + c] = kf_r[k] @ cams[c]["t_ic"] + kf_p[k]
    kinv = [np.linalg.inv(cams[c]["K"]) for c in range(2)]
    cos_min = np.cos(np.radians(min_angle_deg))
    pts, kept = [], []
    for feats in tracks:
        ok = False
        for _ in range(3):
            if len(feats) < 2:
                break
            imgs = image_of[feats]
            cams_i = imgs % 2
            uv1 = np.c_[xy[feats], np.ones(len(feats))]
            rays = np.einsum("nij,nj->ni", rots[imgs],
                             np.where(cams_i[:, None] == 0, uv1 @ kinv[0].T, uv1 @ kinv[1].T))
            rays /= np.linalg.norm(rays, axis=1, keepdims=True)
            proj = np.eye(3)[None] - rays[:, :, None] * rays[:, None, :]
            try:
                x = np.linalg.solve(proj.sum(0), np.einsum("nij,nj->i", proj, centers[imgs]))
            except np.linalg.LinAlgError:
                break
            pc = np.einsum("nji,nj->ni", rots[imgs], x[None] - centers[imgs])
            kmat = np.where(cams_i[:, None, None] == 0, cams[0]["K"][None], cams[1]["K"][None])
            proj_px = np.einsum("nij,nj->ni", kmat, pc)
            err = np.linalg.norm(proj_px[:, :2] / np.maximum(proj_px[:, 2:3], 1e-9) - xy[feats], axis=1)
            good = (pc[:, 2] > 0.2) & (err < max_err)
            if good.all():
                ok = True
                break
            feats = feats[good]
        if not ok or (rays @ rays.T).min() > cos_min:
            continue
        pts.append(x)
        kept.append(feats)
    return np.array(pts), kept


# ----------------------------------------------------------------------------- solver

def skew(v):
    o = np.zeros(v.shape[:-1] + (3, 3))
    o[..., 0, 1], o[..., 0, 2] = -v[..., 2], v[..., 1]
    o[..., 1, 0], o[..., 1, 2] = v[..., 2], -v[..., 0]
    o[..., 2, 0], o[..., 2, 1] = -v[..., 1], v[..., 0]
    return o


def so3_exp(w):
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3) + skew(w[None])[0]
    k = skew((w / th)[None])[0]
    return np.eye(3) + np.sin(th) * k + (1 - np.cos(th)) * k @ k


def so3_log_deg(r):
    from scipy.spatial.transform import Rotation
    return np.degrees(Rotation.from_matrix(r).as_rotvec())


class Reprojection:
    """All reprojection observations, sorted by landmark, with the fixed Schur pair structure.

    Unknowns per observation: the keyframe pose (6, GTSAM Pose3 retract: R <- R Exp(w),
    t <- t + R v), the camera's extrinsic rotation (3, R_imu_cam <- R_imu_cam Exp(phi); zero
    columns for a camera that is not estimated) and the landmark (3, world).
    """

    def __init__(self, obs_kf, obs_cam, obs_lm, obs_z, n_lm, cams, est_cams, sigma, huber, band, chunk=400_000):
        order = np.lexsort((obs_kf, obs_lm))
        self.kf, self.cam, self.lm, self.z = obs_kf[order], obs_cam[order], obs_lm[order], obs_z[order]
        self.n_lm, self.sigma, self.huber, self.chunk = n_lm, sigma, huber, chunk
        self.active = np.ones(len(self.kf), bool)  # observations not rejected as outliers
        for name, i, j in (("fx", 0, 0), ("fy", 1, 1), ("cx", 0, 2), ("cy", 1, 2)):
            setattr(self, name, np.array([c["K"][i, j] for c in cams])[self.cam])
        self.est = np.array([c in est_cams for c in range(2)])
        self.ext_slot = np.full(2, -1)
        for i, c in enumerate(est_cams):
            self.ext_slot[c] = i
        self.n_ext = len(est_cams)
        counts = np.bincount(self.lm, minlength=n_lm)
        start = np.r_[0, np.cumsum(counts)[:-1]]
        # every ordered pair (a, b) of observations of the same landmark
        m = counts[self.lm]
        pa = np.repeat(np.arange(len(self.lm)), m)
        pb = start[self.lm[pa]] + np.arange(len(pa)) - np.repeat(np.r_[0, np.cumsum(m)[:-1]], m)
        print(f"  {len(self.lm)} observations, {len(pa)} Schur pairs", flush=True)
        ka, kb = self.kf[pa], self.kf[pb]
        # pose-pose pairs further apart than --band keyframes (revisits) stay outside the banded factor
        off = np.flatnonzero((ka >= kb) & (ka - kb > band))
        self.off_keys, oinv = np.unique(ka[off].astype(np.int64) * 1_000_000 + kb[off], return_inverse=True)
        s = np.argsort(oinv, kind="stable")
        self.off_a, self.off_b, self.off_bid = pa[off][s].astype(np.int32), pb[off][s].astype(np.int32), oinv[s]
        self.off_ka, self.off_kb = self.off_keys // 1_000_000, self.off_keys % 1_000_000
        if len(off):
            print(f"  {len(self.off_keys)} keyframe blocks outside the band (revisits)", flush=True)
        # lower-band pose-pose pairs (kf_a >= kf_b), sorted by their (kf_a, kf_b) block
        low = np.flatnonzero((ka >= kb) & (ka - kb <= band))
        self.band_kf = int((ka - kb)[low].max()) if len(low) else 1
        self.blk_keys, inv = np.unique(ka[low].astype(np.int64) * 1_000_000 + kb[low], return_inverse=True)
        s = np.argsort(inv, kind="stable")
        self.low_a, self.low_b, self.low_bid = pa[low][s].astype(np.int32), pb[low][s].astype(np.int32), inv[s]
        self.blk_a, self.blk_b = self.blk_keys // 1_000_000, self.blk_keys % 1_000_000
        # border pairs (pose of a x extrinsic of b), sorted by (kf_a, slot_b); corner pairs (both extrinsic)
        slot = self.ext_slot[self.cam]
        if self.n_ext:
            hb = np.flatnonzero(slot[pb] >= 0)
            key = self.kf[pa[hb]].astype(np.int64) * 2 + slot[pb[hb]]
            o = np.argsort(key, kind="stable")
            self.bd_a, self.bd_b, self.bd_key = pa[hb][o].astype(np.int32), pb[hb][o].astype(np.int32), key[o]
            both = np.flatnonzero((slot[pa] >= 0) & (slot[pb] >= 0))
            self.cn_a, self.cn_b = pa[both].astype(np.int32), pb[both].astype(np.int32)
        # direct (same-observation) pose-pose terms, reduced per keyframe
        self.obs_sort = np.argsort(self.kf, kind="stable")
        self.obs_starts = np.flatnonzero(np.r_[True, np.diff(self.kf[self.obs_sort]) != 0])
        self.obs_kfs = self.kf[self.obs_sort][self.obs_starts]

    def _project(self, r_wi, t_wi, r_ic, t_ic, pts):
        rw = r_wi[self.kf]
        p_i = np.einsum("nji,nj->ni", rw, pts[self.lm] - t_wi[self.kf])
        rci = np.transpose(r_ic[self.cam], (0, 2, 1))
        p_c = np.einsum("nij,nj->ni", rci, p_i - t_ic[self.cam])
        valid = p_c[:, 2] > 0.05
        zs = np.where(valid, p_c[:, 2], 1.0)
        res = np.stack([self.fx * p_c[:, 0] / zs + self.cx - self.z[:, 0],
                        self.fy * p_c[:, 1] / zs + self.cy - self.z[:, 1]], 1) / self.sigma
        return rw, p_i, rci, p_c, zs, valid, res

    def _robust(self, res, valid):
        e = np.linalg.norm(res, axis=1)
        k = self.huber
        valid = valid & self.active
        w = np.where(valid, np.where(e <= k, 1.0, k / np.maximum(e, 1e-12)), 0.0)
        cost = np.where(valid, np.where(e <= k, 0.5 * e * e, k * (e - 0.5 * k)), 0.0)
        return e, w, cost

    def cost(self, r_wi, t_wi, r_ic, t_ic, pts):
        *_, valid, res = self._project(r_wi, t_wi, r_ic, t_ic, pts)
        e, _, cost = self._robust(res, valid)
        return float(cost.sum()), e

    def linearize(self, r_wi, t_wi, r_ic, t_ic, pts):
        rw, p_i, rci, p_c, zs, valid, res = self._project(r_wi, t_wi, r_ic, t_ic, pts)
        jpi = np.zeros((len(zs), 2, 3))
        jpi[:, 0, 0] = self.fx / zs
        jpi[:, 0, 2] = -self.fx * p_c[:, 0] / zs ** 2
        jpi[:, 1, 1] = self.fy / zs
        jpi[:, 1, 2] = -self.fy * p_c[:, 1] / zs ** 2
        jpi /= self.sigma
        jr = jpi @ rci
        jx = np.zeros((len(zs), 2, 9))
        jx[:, :, 0:3] = jr @ skew(p_i)          # pose rotation
        jx[:, :, 3:6] = -jr                     # pose translation
        jx[:, :, 6:9] = jpi @ skew(p_c)         # extrinsic rotation
        jx[~self.est[self.cam], :, 6:9] = 0.0
        jl = jr @ np.transpose(rw, (0, 2, 1))   # landmark
        _, w, _ = self._robust(res, valid)
        return res, jx, jl, w


def _imu_graph(kf_ns, kf_p, kf_r, vio, imu_path, noise, noise_scale=1.0, accel_rw_scale=1.0, gyro_rw_scale=1.0,
               prior_rot_sigma=1e-4, omega_coriolis=None):
    """GTSAM values and factors for poses, velocities, biases, the IMU and the gauge priors.

    The first keyframe's position is pinned (1e-4 m); its rotation gets prior_rot_sigma (rad). Gravity
    makes roll and pitch observable, so a loose rotation prior lets the solve fix a tilt the VIO got
    wrong at its start instead of pinning it there.
    """
    import gtsam
    from gtsam.symbol_shorthand import B, V, X
    vio_ns, vio_p, _ = vio
    imu = np.loadtxt(imu_path, delimiter=",", comments="#")
    imu_ns = imu[:, 0].astype(np.int64)
    params = gtsam.PreintegrationCombinedParams.MakeSharedU(9.81)
    params.setAccelerometerCovariance(np.eye(3) * (noise_scale * noise["accel_noise_std"]) ** 2)
    params.setGyroscopeCovariance(np.eye(3) * (noise_scale * noise["gyro_noise_std"]) ** 2)
    params.setIntegrationCovariance(np.eye(3) * 1e-8)
    params.setBiasAccCovariance(np.eye(3) * (accel_rw_scale * noise["accel_bias_std"]) ** 2)
    params.setBiasOmegaCovariance(np.eye(3) * (gyro_rw_scale * noise["gyro_bias_std"]) ** 2)
    if omega_coriolis is not None:  # Earth's rotation in the world frame, which the gyroscope also measures
        params.setOmegaCoriolis(np.asarray(omega_coriolis, float))
    graph, values = gtsam.NonlinearFactorGraph(), gtsam.Values()
    imu_factors = []
    zero_bias = gtsam.imuBias.ConstantBias()
    n = len(kf_ns)
    for k in range(n):
        i = np.searchsorted(vio_ns, kf_ns[k])
        i0, i1 = max(i - 1, 0), min(i + 1, len(vio_ns) - 1)
        vel = (vio_p[i1] - vio_p[i0]) / ((vio_ns[i1] - vio_ns[i0]) * 1e-9)
        values.insert(X(k), gtsam.Pose3(gtsam.Rot3(kf_r[k]), kf_p[k]))
        values.insert(V(k), vel)
        values.insert(B(k), zero_bias)
        if k == 0:
            graph.add(gtsam.PriorFactorPose3(X(0), values.atPose3(X(0)), gtsam.noiseModel.Diagonal.Sigmas(
                np.r_[np.full(3, prior_rot_sigma), np.full(3, 1e-4)])))
            graph.add(gtsam.PriorFactorVector(V(0), vel, gtsam.noiseModel.Isotropic.Sigma(3, 0.1)))
            graph.add(gtsam.PriorFactorConstantBias(B(0), zero_bias, gtsam.noiseModel.Diagonal.Sigmas(
                np.array([0.1, 0.1, 0.1, 1e-3, 1e-3, 1e-3]))))
    for k in range(n - 1):
        pim = gtsam.PreintegratedCombinedMeasurements(params, zero_bias)
        ta, tb = kf_ns[k], kf_ns[k + 1]
        s = max(np.searchsorted(imu_ns, ta, side="right") - 1, 0)
        e = np.searchsorted(imu_ns, tb, side="left")
        for i in range(s, min(e, len(imu_ns) - 1)):
            lo, hi = max(imu_ns[i], ta), min(imu_ns[i + 1], tb)
            if hi > lo:
                pim.integrateMeasurement(imu[i, 4:7], imu[i, 1:4], (hi - lo) * 1e-9)
        imu_factors.append(gtsam.CombinedImuFactor(X(k), V(k), X(k + 1), V(k + 1), B(k), B(k + 1), pim))
        graph.add(imu_factors[-1])
    index = {}
    for k in range(n):
        index[X(k)], index[V(k)], index[B(k)] = (k * DOF, 6), (k * DOF + 6, 3), (k * DOF + 9, 6)
    return graph, values, index, imu_factors


def mode_vectors(modes, kf_s, r_wi, t_wi, vel, pts, lm_first):
    """Tangent directions of slow deformations of the whole solution, per unit amplitude at the end.

    yaw: the heading turns by s_k rad at keyframe k (s from 0 at the first keyframe to 1 at the
    last); positions are re-integrated from the turned displacements, velocities turn, landmarks
    move with their first keyframe and the gyro bias takes one offset, the mean body-frame vertical,
    so that the preintegrated rotations still agree. scale: displacements stretch by 1 + s_k.
    Returns V_x (m, n, DOF) in the tangent space of each keyframe and V_l (m, n_lm, 3).
    """
    up = np.array([0.0, 0.0, 1.0])
    dur = max(kf_s[-1] - kf_s[0], 1e-9)
    s = (kf_s - kf_s[0]) / dur
    s_mid = 0.5 * (s[1:] + s[:-1])
    dp = np.diff(t_wi, axis=0)
    out_x, out_l = [], []
    for mode in modes:
        v = np.zeros((len(kf_s), DOF))
        if mode == "yaw":
            shift = np.vstack([np.zeros(3), np.cumsum(np.cross(up, dp) * s_mid[:, None], axis=0)])
            up_body = np.einsum("nji,j->ni", r_wi, up)
            v[:, 0:3] = s[:, None] * up_body
            v[:, 6:9] = s[:, None] * np.cross(up, vel)
            v[:, 12:15] = -up_body.mean(0) / dur
            v_l = shift[lm_first] + s[lm_first, None] * np.cross(up, pts - t_wi[lm_first])
        elif mode == "scale":
            shift = np.vstack([np.zeros(3), np.cumsum(dp * s_mid[:, None], axis=0)])
            v[:, 6:9] = s[:, None] * vel
            v_l = shift[lm_first] + s[lm_first, None] * (pts - t_wi[lm_first])
        else:
            raise ValueError(f"unknown mode {mode}")
        v[:, 3:6] = np.einsum("nji,nj->ni", r_wi, shift)
        out_x.append(v)
        out_l.append(v_l)
    return np.array(out_x), np.array(out_l)


def deform(mode, beta, kf_s, r_wi, t_wi, vel, bias, pts, lm_first):
    """The finite deformation whose tangent mode_vectors gives, applied with amplitude beta.

    Applied exactly instead of through the pose retraction: GTSAM's Pose3 retraction couples
    rotation and translation, and a heading turn of 0.05 rad would move far keyframes by decimetres.
    """
    dur = max(kf_s[-1] - kf_s[0], 1e-9)
    s = (kf_s - kf_s[0]) / dur
    s_mid = 0.5 * (s[1:] + s[:-1])
    dp = np.diff(t_wi, axis=0)
    k = lm_first
    bias = bias.copy()
    if mode == "yaw":
        def rz(angle):
            c, sn, zero, one = np.cos(angle), np.sin(angle), np.zeros_like(angle), np.ones_like(angle)
            return np.stack([np.stack([c, -sn, zero], -1), np.stack([sn, c, zero], -1),
                             np.stack([zero, zero, one], -1)], -2)
        turn, turn_mid = rz(beta * s), rz(beta * s_mid)
        t_new = np.vstack([t_wi[:1], t_wi[:1] + np.cumsum(np.einsum("nij,nj->ni", turn_mid, dp), axis=0)])
        r_new = np.einsum("nij,njk->nik", turn, r_wi)
        vel_new = np.einsum("nij,nj->ni", turn, vel)
        bias[:, 3:6] -= beta * r_wi[:, 2, :].mean(0) / dur  # R^T z: the body-frame vertical
        pts_new = t_new[k] + np.einsum("nij,nj->ni", turn[k], pts - t_wi[k])
    elif mode == "scale":
        t_new = np.vstack([t_wi[:1], t_wi[:1] + np.cumsum(dp * (1 + beta * s_mid[:, None]), axis=0)])
        r_new = r_wi
        vel_new = vel * (1 + beta * s[:, None])
        pts_new = t_new[k] + (1 + beta * s[k, None]) * (pts - t_wi[k])
    else:
        raise ValueError(f"unknown mode {mode}")
    return r_new, t_new, vel_new, bias, pts_new


def _band_matvec(ab, x):
    """Symmetric banded matrix (lower storage) times a vector."""
    nb = len(x)
    y = ab[0] * x
    for d in range(1, ab.shape[0]):
        y[d:] += ab[d, :nb - d] * x[:nb - d]
        y[:nb - d] += ab[d, :nb - d] * x[d:]
    return y


def cmd_solve(a):
    import gtsam
    from gtsam.symbol_shorthand import B, V, X
    from scipy.linalg import cho_solve_banded, cholesky_banded

    t0 = time.time()
    kf_ns, offsets, xy, _ = load_features(a.work)
    cams, noise = load_calib(a.calib)
    vio = load_tum(a.vio)
    kf_p, kf_r = keyframe_poses(kf_ns, a.vio)
    m = np.load(os.path.join(a.work, "matches.npz"))
    k0, k1 = (int(x) for x in a.kf_range.split(":")) if a.kf_range else (0, len(kf_ns))
    k1 = min(k1, len(kf_ns))
    sel = (m["pairs"][:, 0] >= k0) & (m["pairs"][:, 2] < k1) & (m["pairs"][:, 2] - m["pairs"][:, 0] <= a.max_offset)
    pairs, matches = m["pairs"][sel], m["matches"][sel]
    if a.loops:  # verified revisit matches from the loops stage
        lm = np.load(os.path.join(a.work, "loops.npz"))
        lsel = (lm["pairs"][:, 0] >= k0) & (lm["pairs"][:, 2] < k1)
        pairs, matches = np.concatenate([pairs, lm["pairs"][lsel]]), np.concatenate([matches, lm["matches"][lsel]])
        print(f"  + {lsel.sum()} revisit matches", flush=True)
    tracks, image_of, conflicts = build_tracks(offsets, pairs, matches, 1 << 30)
    pts, kept = triangulate(tracks, image_of, xy, kf_p, kf_r, cams, a.max_reproj, a.min_angle)
    kf_ns, kf_p, kf_r = kf_ns[k0:k1], kf_p[k0:k1], kf_r[k0:k1]
    # a landmark keeps the observations within --band keyframes of its first one, a revisit landmark all
    obs_kf, obs_cam, obs_lm, obs_z = [], [], [], []
    for j, feats in enumerate(kept):
        k = image_of[feats] // 2
        keep = (k - k.min() <= a.band) | bool(a.loops)
        if keep.sum() >= 2:
            obs_kf.append(k[keep] - k0)
            obs_cam.append(image_of[feats[keep]] % 2)
            obs_lm.append(np.full(keep.sum(), j))
            obs_z.append(xy[feats[keep]].astype(np.float64))
    obs_lm = np.concatenate(obs_lm)
    used = np.unique(obs_lm)
    remap = np.full(len(kept), -1)
    remap[used] = np.arange(len(used))
    pts = pts[used].copy()
    print(f"tracks {len(tracks)} (dropped {conflicts} with conflicts), landmarks {len(pts)}, "
          f"{time.time() - t0:.0f} s", flush=True)
    est_cams = [int(ch) for ch in a.est_extrinsic]
    rp = Reprojection(np.concatenate(obs_kf), np.concatenate(obs_cam), remap[obs_lm], np.concatenate(obs_z),
                      len(pts), cams, est_cams, a.pixel_sigma, a.huber, a.band)
    omega = [float(v) for v in a.omega_coriolis.split(",")] if a.omega_coriolis else None
    graph, values, index, imu_factors = _imu_graph(kf_ns, kf_p, kf_r, vio, a.imu, noise, a.imu_noise_scale,
                                                   a.accel_rw_scale, a.gyro_rw_scale, a.prior_rot_sigma, omega)
    n = len(kf_ns)
    nb, ne = n * DOF, 3 * rp.n_ext
    modes = [m for m in a.modes.split(",") if m]
    nm = len(modes)
    nz = ne + nm  # border: the camera rotations, then the deformation modes (undamped)
    order = np.argsort(rp.lm, kind="stable")
    lm_first = np.zeros(rp.n_lm, np.int64)
    lm_first[rp.lm[order][::-1]] = rp.kf[order][::-1]  # first keyframe of every landmark
    kf_s = kf_ns.astype(np.float64) * 1e-9
    u = (max(rp.band_kf, 1) + 1) * DOF - 1  # scalar bandwidth of the reduced system
    print(f"  reduced system: {nb} banded dims (bandwidth {u}) + {ne} border dims, {time.time() - t0:.0f} s",
          flush=True)
    r_ic = np.array([c["R_ic"] for c in cams])
    t_ic = np.array([c["t_ic"] for c in cams])

    def poses(vals):
        return (np.array([vals.atPose3(X(k)).rotation().matrix() for k in range(n)]),
                np.array([vals.atPose3(X(k)).translation() for k in range(n)]))

    r_wi, t_wi = poses(values)
    cost = rp.cost(r_wi, t_wi, r_ic, t_ic, pts)[0] + graph.error(values)
    print(f"initial cost {cost:.6g}", flush=True)
    lam = a.lambda0
    mu = 0.0  # damping of the modes relative to their own curvature: none until a step with them fails
    for it in range(a.iterations):
        ti = time.time()
        if a.outlier_px > 0 and it >= a.outlier_start:
            # drop observations that moved far off, then landmarks left with a single observation;
            # from --outlier-start2 on with the tighter --outlier-px2
            thr = a.outlier_px2 if 0 < a.outlier_px2 and it >= a.outlier_start2 else a.outlier_px
            bad = rp.active & (rp.cost(r_wi, t_wi, r_ic, t_ic, pts)[1] * a.pixel_sigma > thr)
            if bad.any():
                rp.active &= ~bad
                lone = rp.active & (np.bincount(rp.lm[rp.active], minlength=rp.n_lm)[rp.lm] < 2)
                rp.active &= ~lone
                cost = rp.cost(r_wi, t_wi, r_ic, t_ic, pts)[0] + graph.error(values)
                print(f"  outliers: {bad.sum()} observations over {thr} px (+{lone.sum()} left alone), "
                      f"{rp.active.sum()} active, cost {cost:.6g}", flush=True)
        res, jx, jl, w = rp.linearize(r_wi, t_wi, r_ic, t_ic, pts)
        wjx, wjl = jx * w[:, None, None], jl * w[:, None, None]
        h_xx = np.einsum("nki,nkj->nij", wjx, jx)
        h_xl = np.einsum("nki,nkj->nij", wjx, jl)
        g_x = np.einsum("nki,nk->ni", wjx, res)
        h_ll_o = np.einsum("nki,nkj->nij", wjl, jl)
        h_ll = np.zeros((rp.n_lm, 3, 3))
        np.add.at(h_ll, rp.lm, h_ll_o)
        g_l = np.zeros((rp.n_lm, 3))
        np.add.at(g_l, rp.lm, np.einsum("nki,nk->ni", wjl, res))
        # IMU and prior factors, linearized by GTSAM: ||A dx - b||^2
        ab_imu, rhs_imu = np.zeros((u + 1, nb)), np.zeros(nb)
        lin = graph.linearize(values)
        for fi in range(lin.size()):
            jf = lin.at(fi)
            A, b = jf.jacobian()
            cols = np.concatenate([np.arange(index[key][0], index[key][0] + index[key][1]) for key in jf.keys()])
            ci, cj = np.meshgrid(cols, cols, indexing="ij")
            low = ci >= cj
            np.add.at(ab_imu, ((ci - cj)[low], cj[low]), (A.T @ A)[low])
            np.add.at(rhs_imu, cols, A.T @ b)
        direct = np.add.reduceat(h_xx[rp.obs_sort][:, 0:6, 0:6], rp.obs_starts, axis=0)
        modes_on = nm > 0 and it >= a.modes_start
        if modes_on:
            # H v for every mode before the landmarks are eliminated (undamped)
            vel = np.array([values.atVector(V(k)) for k in range(n)])
            v_x, v_l = mode_vectors(modes, kf_s, r_wi, t_wi, vel, pts, lm_first)
            vxf = v_x.reshape(nm, nb)
            hv_x, hv_l, hv_e = np.zeros((nm, nb)), np.zeros((nm, rp.n_lm, 3)), np.zeros((nm, ne))
            has_o = rp.ext_slot[rp.cam] >= 0
            for mi in range(nm):
                v6, vlo = v_x[mi][rp.kf, 0:6], v_l[mi][rp.lm]
                hv_x[mi] = _band_matvec(ab_imu, vxf[mi])
                np.add.at(hv_x[mi], (rp.kf * DOF)[:, None] + np.arange(6)[None, :],
                          np.einsum("nij,nj->ni", h_xx[:, 0:6, 0:6], v6) + np.einsum("nij,nj->ni", h_xl[:, 0:6], vlo))
                np.add.at(hv_l[mi], rp.lm, np.einsum("nji,nj->ni", h_xl[:, 0:6], v6) + np.einsum("nij,nj->ni", h_ll_o, vlo))
                if ne:
                    pe = (np.einsum("nji,nj->ni", h_xx[has_o][:, 0:6, 6:9], v6[has_o])
                          + np.einsum("nij,nj->ni", h_xl[has_o, 6:9], vlo[has_o]))
                    for sl in range(rp.n_ext):
                        hv_e[mi, sl * 3:sl * 3 + 3] = pe[rp.ext_slot[rp.cam[has_o]] == sl].sum(0)
            vhv = vxf @ hv_x.T + np.einsum("mlk,nlk->mn", v_l, hv_l)
            b_x = rhs_imu.copy()
            np.add.at(b_x, (rp.kf * DOF)[:, None] + np.arange(6)[None, :], -g_x[:, 0:6])
            b_v = vxf @ b_x - np.einsum("mlk,lk->m", v_l, g_l)
        accepted = False
        # landmark damping: the diagonal, floored relative to the landmark's largest entry, so that a
        # landmark with a poorly observed direction (low parallax) cannot take a huge step
        d_l = np.diagonal(h_ll, axis1=1, axis2=2)
        d_l = np.maximum(d_l, a.lm_damp_floor * d_l.max(axis=1, keepdims=True))[:, :, None] * np.eye(3)[None]
        for _ in range(8):
            h_ll_inv = np.linalg.inv(h_ll + lam * d_l + 1e-9 * np.eye(3)[None])
            m_o = np.einsum("nij,njk->nik", h_xl, h_ll_inv[rp.lm])
            ab, rhs = ab_imu.copy(), np.zeros(nb + nz)
            rhs[:nb] = rhs_imu
            # Schur complement of the landmarks: -M_a X_b^T per (kf_a, kf_b) block, in chunks
            blocks = np.zeros((len(rp.blk_keys), 6, 6))
            for s in range(0, len(rp.low_a), rp.chunk):
                e = min(s + rp.chunk, len(rp.low_a))
                prod = -np.einsum("nij,nkj->nik", m_o[rp.low_a[s:e], 0:6, :], h_xl[rp.low_b[s:e], 0:6, :])
                bid = rp.low_bid[s:e]
                st = np.flatnonzero(np.r_[True, bid[1:] != bid[:-1]])
                blocks[bid[st]] += np.add.reduceat(prod, st, axis=0)
            for rows, cols, vals in ((rp.blk_a * DOF, rp.blk_b * DOF, blocks), (rp.obs_kfs * DOF, rp.obs_kfs * DOF, direct)):
                ii, jj = np.broadcast_arrays(rows[:, None, None] + np.arange(6)[None, :, None],
                                             cols[:, None, None] + np.arange(6)[None, None, :])
                low = ii >= jj
                ab[(ii - jj)[low], jj[low]] += vals[low]
            border, corner = np.zeros((nb, nz)), np.zeros((nz, nz))
            if ne:
                slot = rp.ext_slot[rp.cam]
                has = slot >= 0
                bsum = np.zeros((n * 2, 6, 3))
                for s in range(0, len(rp.bd_a), rp.chunk):
                    e = min(s + rp.chunk, len(rp.bd_a))
                    prod = -np.einsum("nij,nkj->nik", m_o[rp.bd_a[s:e], 0:6, :], h_xl[rp.bd_b[s:e], 6:9, :])
                    key = rp.bd_key[s:e]
                    st = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
                    bsum[key[st]] += np.add.reduceat(prod, st, axis=0)
                np.add.at(bsum, rp.kf[has].astype(np.int64) * 2 + slot[has], h_xx[has][:, 0:6, 6:9])
                for sl in range(rp.n_ext):
                    border.reshape(n, DOF, nz)[:, 0:6, sl * 3:sl * 3 + 3] += bsum[sl::2]
                csum = np.zeros((2, 2, 3, 3))
                for s in range(0, len(rp.cn_a), rp.chunk):
                    e = min(s + rp.chunk, len(rp.cn_a))
                    prod = -np.einsum("nij,nkj->nik", m_o[rp.cn_a[s:e], 6:9, :], h_xl[rp.cn_b[s:e], 6:9, :])
                    np.add.at(csum, (slot[rp.cn_a[s:e]], slot[rp.cn_b[s:e]]), prod)
                np.add.at(csum, (slot[has], slot[has]), h_xx[has][:, 6:9, 6:9])
                for sa in range(rp.n_ext):
                    for sb in range(rp.n_ext):
                        corner[sa * 3:sa * 3 + 3, sb * 3:sb * 3 + 3] = csum[sa, sb]
            red = -g_x + np.einsum("nij,nj->ni", m_o, g_l[rp.lm])
            np.add.at(rhs, (rp.kf * DOF)[:, None] + np.arange(6)[None, :], red[:, 0:6])
            if ne:
                np.add.at(rhs, nb + (slot[has] * 3)[:, None] + np.arange(3)[None, :], red[has, 6:9])
            ab[0] += lam * ab[0] + 1e-9
            if ne:
                corner[np.arange(ne), np.arange(ne)] += lam * np.diag(corner)[:ne] + 1e-9
            if nm and not modes_on:
                corner[ne:, ne:] = np.eye(nm)  # inert until --modes-start
            if modes_on:  # the modes' column after eliminating the landmarks; no damping
                q = np.einsum("lij,mlj->mli", h_ll_inv, hv_l)
                for mi in range(nm):
                    border[:, ne + mi] = hv_x[mi]
                    np.add.at(border[:, ne + mi], (rp.kf * DOF)[:, None] + np.arange(6)[None, :],
                              -np.einsum("nij,nj->ni", h_xl[:, 0:6], q[mi][rp.lm]))
                if ne:
                    pe = np.einsum("nij,mnj->mni", h_xl[has, 6:9], q[:, rp.lm[has]])
                    ce = hv_e.copy()
                    for sl in range(rp.n_ext):
                        ce[:, sl * 3:sl * 3 + 3] -= pe[:, rp.ext_slot[rp.cam[has]] == sl].sum(1)
                    corner[:ne, ne:], corner[ne:, :ne] = ce.T, ce
                cmm = vhv - np.einsum("mlk,nlk->mn", hv_l, q)
                corner[ne:, ne:] = cmm + np.diag((mu + 1e-12) * np.abs(np.diag(cmm)) + 1e-12)
                rhs[nb + ne:] = b_v + np.einsum("mlk,lk->m", hv_l, np.einsum("lij,lj->li", h_ll_inv, g_l))
            try:
                cfac = cholesky_banded(ab, lower=True)
            except np.linalg.LinAlgError:
                lam *= 10
                continue
            if nz:  # block elimination of the border
                yb = cho_solve_banded((cfac, True), border)
                schur = corner - border.T @ yb
                s_inv = np.linalg.inv(schur)
                if modes_on and len(rp.off_keys) == 0:
                    # trust region: damp the modes further until each step is within its cap
                    rz = rhs[nb:] - border.T @ cho_solve_banded((cfac, True), rhs[:nb])
                    cap = np.array([a.mode_cap_yaw if m_ == "yaw" else a.mode_cap_scale for m_ in modes])
                    extra = 0.0
                    for _ in range(30):
                        if np.all(np.abs((s_inv @ rz)[ne:]) <= cap):
                            break
                        extra = 4 * extra if extra else 1e-3
                        damped = schur.copy()
                        damped[ne:, ne:] += np.diag(extra * np.abs(np.diag(cmm)))
                        s_inv = np.linalg.inv(damped)

            def precond(r_):
                y0 = cho_solve_banded((cfac, True), r_[:nb])
                if not nz:
                    return y0
                de0 = s_inv @ (r_[nb:] - border.T @ y0)
                return np.r_[y0 - yb @ de0, de0]

            sol = precond(rhs) if len(rp.off_keys) == 0 else _pcg(rp, m_o, h_xl, ab, u, nb, nz, border, corner,
                                                                    precond, rhs)
            dx, de, dk = sol[:nb], sol[nb:nb + ne], sol[nb + ne:]
            dxo = np.zeros((len(rp.kf), 9))
            dxo[:, 0:6] = dx.reshape(n, DOF)[rp.kf, 0:6]
            if ne:
                dxo[has, 6:9] = de.reshape(-1, 3)[slot[has]]
            back = np.zeros((rp.n_lm, 3))
            np.add.at(back, rp.lm, np.einsum("nji,nj->ni", h_xl, dxo))
            if modes_on:
                back += np.einsum("mlk,m->lk", hv_l, dk)
            dl = -np.einsum("nij,nj->ni", h_ll_inv, g_l + back)
            vv = gtsam.VectorValues()
            for key, (col, dim) in index.items():
                vv.insert(key, dx[col:col + dim])
            new_values = values.retract(vv)
            new_ric = r_ic.copy()
            for c in est_cams:
                new_ric[c] = r_ic[c] @ so3_exp(de[rp.ext_slot[c] * 3:rp.ext_slot[c] * 3 + 3])
            new_pts = pts + dl
            nr, nt = poses(new_values)
            if modes_on:  # the modes' share of the step, as exact deformations
                st = (nr, nt, np.array([new_values.atVector(V(k)) for k in range(n)]),
                      np.array([np.r_[new_values.atConstantBias(B(k)).accelerometer(),
                                      new_values.atConstantBias(B(k)).gyroscope()] for k in range(n)]), new_pts)
                for mo, be in zip(modes, dk):
                    st = deform(mo, be, kf_s, *st, lm_first)
                new_values = gtsam.Values()
                for k in range(n):
                    new_values.insert(X(k), gtsam.Pose3(gtsam.Rot3(st[0][k]), st[1][k]))
                    new_values.insert(V(k), st[2][k])
                    new_values.insert(B(k), gtsam.imuBias.ConstantBias(st[3][k, 0:3], st[3][k, 3:6]))
                nr, nt, new_pts = st[0], st[1], st[4]
            new_cost = rp.cost(nr, nt, new_ric, t_ic, new_pts)[0] + graph.error(new_values)
            if new_cost < cost:
                rel = (cost - new_cost) / cost
                values, r_ic, pts, r_wi, t_wi, cost = new_values, new_ric, new_pts, nr, nt, new_cost
                lam = max(lam / 3, 1e-10)
                mu = mu / 10 if mu > 1e-4 else 0.0
                accepted = True
                break
            lam *= 5
            if modes_on:  # a rejected step also shrinks the modes' share
                mu = max(mu * 10, 1e-3)
        msg = f"iter {it:2d}: cost {cost:.6g} lambda {lam:.1e} {time.time() - ti:.0f} s"
        if modes_on and accepted:
            msg += " | " + " ".join(f"{mo} {d_:+.2e}" for mo, d_ in zip(modes, dk))
        for c in est_cams:
            msg += f" | cam{c} rotation {np.round(so3_log_deg(cams[c]['R_ic'].T @ r_ic[c]), 3)}"
        print(msg, flush=True)
        if not accepted or rel < a.rel_tol:
            break
    reproj, err = rp.cost(r_wi, t_wi, r_ic, t_ic, pts)
    err_px = err[rp.active] * a.pixel_sigma
    comp = np.zeros(5)
    for fct in imu_factors:  # whitened IMU residual per component: rotation, position, velocity, biases
        ew = fct.whitenedError(values)
        comp += [np.sum(ew[i:i + 3] ** 2) for i in (0, 3, 6, 9, 12)]
    comp /= max(len(imu_factors), 1)
    summary = {
        "keyframes": n, "landmarks": int(rp.n_lm), "observations": int(len(rp.kf)),
        "active_observations": int(rp.active.sum()),
        "final_cost": cost, "reprojection_cost": reproj, "imu_prior_cost": graph.error(values),
        "median_reprojection_px": float(np.median(err_px)),
        "reprojection_over_4px_percent": float(np.mean(err_px > 4) * 100),
        "imu_residual_per_factor": dict(zip(("rotation", "position", "velocity", "accel_bias", "gyro_bias"),
                                            comp.round(3).tolist())),
        "camera_rotation_change_deg": {f"cam{c}": so3_log_deg(cams[c]["R_ic"].T @ r_ic[c]).tolist()
                                       for c in est_cams},
        "keyframe_displacement_m": {"median": float(np.median(np.linalg.norm(t_wi - kf_p, axis=1))),
                                    "max": float(np.linalg.norm(t_wi - kf_p, axis=1).max())},
        "biases": {str(k): {"accel": values.atConstantBias(B(k)).accelerometer().tolist(),
                            "gyro": values.atConstantBias(B(k)).gyroscope().tolist()} for k in (0, n // 2, n - 1)},
        "seconds": time.time() - t0,
    }
    print(json.dumps(summary, indent=1), flush=True)
    json.dump(summary, open(os.path.splitext(a.out)[0] + "_summary.json", "w"), indent=1)
    write_corrected(a.out, vio, kf_ns, kf_p, kf_r, t_wi, r_wi)
    if a.save_states:  # what loop-correct needs: keyframe poses, landmarks and their observations
        np.savez(os.path.splitext(a.out)[0] + "_states.npz", k0=k0, kf_ns=kf_ns, r_wi=r_wi, t_wi=t_wi, r_ic=r_ic,
                 pts=pts, obs_kf=rp.kf, obs_cam=rp.cam, obs_lm=rp.lm, obs_z=rp.z, active=rp.active)


def _pcg(rp, m_o, h_xl, ab, u, nb, ne, border, corner, precond, rhs):
    """Conjugate gradients on the full reduced system; the bordered banded factor preconditions it."""
    blocks = np.zeros((len(rp.off_keys), 6, 6))
    for s in range(0, len(rp.off_a), rp.chunk):
        e = min(s + rp.chunk, len(rp.off_a))
        prod = -np.einsum("nij,nkj->nik", m_o[rp.off_a[s:e], 0:6, :], h_xl[rp.off_b[s:e], 0:6, :])
        bid = rp.off_bid[s:e]
        st = np.flatnonzero(np.r_[True, bid[1:] != bid[:-1]])
        blocks[bid[st]] += np.add.reduceat(prod, st, axis=0)
    rows_a = (rp.off_ka * DOF)[:, None] + np.arange(6)
    rows_b = (rp.off_kb * DOF)[:, None] + np.arange(6)

    def operator(x):
        xb = x[:nb]
        y = ab[0] * xb
        for d in range(1, u + 1):
            y[d:] += ab[d, :nb - d] * xb[:nb - d]
            y[:nb - d] += ab[d, :nb - d] * xb[d:]
        np.add.at(y, rows_a, np.einsum("nij,nj->ni", blocks, xb[rows_b]))
        np.add.at(y, rows_b, np.einsum("nji,nj->ni", blocks, xb[rows_a]))
        if not ne:
            return y
        return np.r_[y + border @ x[nb:], border.T @ xb + corner @ x[nb:]]

    x = np.zeros(nb + ne)
    r = rhs.copy()
    z = precond(r)
    p = z.copy()
    rz = r @ z
    bnorm = np.linalg.norm(rhs)
    for it in range(200):
        ap = operator(p)
        alpha = rz / (p @ ap)
        x += alpha * p
        r -= alpha * ap
        if np.linalg.norm(r) < 1e-9 * bnorm:
            break
        z = precond(r)
        rz, rz_old = r @ z, rz
        p = z + (rz / rz_old) * p
    print(f"    pcg: {it + 1} iterations", flush=True)
    return x


def write_corrected(path, vio, kf_ns, kf_p, kf_r, opt_p, opt_r):
    """Each VIO frame moves with the world-frame correction of the surrounding keyframes, interpolated."""
    from scipy.spatial.transform import Rotation, Slerp
    vio_ns, vio_p, vio_r = vio
    corr_r = np.einsum("nij,nkj->nik", opt_r, kf_r)
    corr_t = opt_p - np.einsum("nij,nj->ni", corr_r, kf_p)
    tq = kf_ns.astype(np.float64)
    t = np.clip(vio_ns.astype(np.float64), tq[0], tq[-1])
    rc = Slerp(tq, Rotation.from_matrix(corr_r))(t).as_matrix()
    tc = np.stack([np.interp(t, tq, corr_t[:, i]) for i in range(3)], axis=1)
    out_r = np.einsum("nij,njk->nik", rc, vio_r)
    out_p = np.einsum("nij,nj->ni", rc, vio_p) + tc
    q = Rotation.from_matrix(out_r).as_quat()
    with open(path, "w") as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        for i in range(len(vio_ns)):
            f.write(f"{vio_ns[i] * 1e-9:.9f} {out_p[i, 0]:.6f} {out_p[i, 1]:.6f} {out_p[i, 2]:.6f} "
                    f"{q[i, 0]:.9f} {q[i, 1]:.9f} {q[i, 2]:.9f} {q[i, 3]:.9f}\n")
    print(f"wrote {path}", flush=True)


# ----------------------------------------------------------------------------- loop correction

def pnp_body_pose(p3, p2, kmat, r_ic, t_ic, r_wb0, t_wb0, min_inliers):
    """World pose (R_wb, t_wb) of the body whose camera (R_ic, t_ic in the body) sees the world points p3 at
    the pixels p2, by PnP with RANSAC (3 px) from the guess (R_wb0, t_wb0); None below min_inliers."""
    import cv2
    r_wc = r_wb0 @ r_ic
    c_w = r_wb0 @ t_ic + t_wb0
    rvec, tvec = cv2.Rodrigues(r_wc.T)[0], (-r_wc.T @ c_w).reshape(3, 1)
    found, rvec, tvec, inl = cv2.solvePnPRansac(p3, p2, kmat, None, rvec, tvec, useExtrinsicGuess=True,
                                                iterationsCount=200, reprojectionError=3.0, confidence=0.999)
    if not found or inl is None or len(inl) < min_inliers:
        return None
    rvec, tvec = cv2.solvePnPRefineLM(p3[inl[:, 0]], p2[inl[:, 0]], kmat, None, rvec, tvec)
    r_wc = cv2.Rodrigues(rvec)[0].T
    r_wb = r_wc @ r_ic.T
    return r_wb, -r_wc @ tvec.ravel() - r_wb @ t_ic


def cmd_loop_correct(a):
    """Close the revisits of a finished solve with a pose graph.

    A solve with the revisit matches cannot close a loop the VIO drifted on: the triangulation and the
    outlier rejection drop exactly the revisit observations that disagree by a metre or more. Here
    the revisits become relative poses instead. For each revisit image pair (a, b), a's features that
    are observations of a solved landmark give 3D points of a's visit; PnP with RANSAC places b's
    camera against them (and a's against b's visit). Each such edge between keyframes a and b joins
    a pose graph whose odometry edges are the solve's own consecutive relative poses; an edge whose
    disagreement with the solve is implausible for the time between the visits (--gate) is dropped,
    and the rest are robust (Cauchy). The solved graph's keyframe corrections are spread over every
    frame of the solve's output trajectory.
    """
    import gtsam
    from gtsam.symbol_shorthand import X
    from scipy.spatial.transform import Rotation

    st = np.load(a.states)
    cams, _ = load_calib(a.calib)
    k0, kf_ns, r_wi, t_wi, r_ic, pts = (st[k] for k in ("k0", "kf_ns", "r_wi", "t_wi", "r_ic", "pts"))
    k0 = int(k0)
    t_ic = np.array([c["t_ic"] for c in cams])
    act = st["active"]
    okf, ocam, olm, oz = st["obs_kf"][act], st["obs_cam"][act], st["obs_lm"][act], st["obs_z"][act]
    _, offsets, xy, _ = load_features(a.work)
    lp = np.load(os.path.join(a.work, a.loops_file))
    n = len(kf_ns)
    sel = (lp["pairs"][:, 0] >= k0) & (lp["pairs"][:, 2] < k0 + n)
    pairs, matches = lp["pairs"][sel].astype(np.int64), lp["matches"][sel]
    pairs[:, [0, 2]] -= k0
    # landmark of a feature, found by its pixel among the solve's active observations
    lookup = {(int(2 * k + c), round(float(z[0]), 3), round(float(z[1]), 3)): int(j)
              for k, c, z, j in zip(okf, ocam, oz, olm)}

    def edge(k3, c3, f3, k2, c2, f2):
        """Body pose of keyframe k2 placed by PnP against the landmarks seen by features f3 of (k3, c3)."""
        lms = np.array([lookup.get((2 * k3 + c3, round(float(xy[i, 0]), 3), round(float(xy[i, 1]), 3)), -1)
                        for i in f3])
        ok = lms >= 0
        if ok.sum() < a.min_inliers:
            return None
        placed = pnp_body_pose(pts[lms[ok]], xy[f2[ok]].astype(np.float64), cams[c2]["K"], r_ic[c2], t_ic[c2],
                               r_wi[k2], t_wi[k2], a.min_inliers)
        if placed is None:
            return None
        r_wb, t_wb = placed
        rel_r, rel_t = r_wi[k3].T @ r_wb, r_wi[k3].T @ (t_wb - t_wi[k3])
        dev_t = np.linalg.norm(rel_t - r_wi[k3].T @ (t_wi[k2] - t_wi[k3]))
        dev_r = np.degrees(np.linalg.norm(Rotation.from_matrix((r_wi[k3].T @ r_wi[k2]).T @ rel_r).as_rotvec()))
        return k3, k2, rel_r, rel_t, dev_r, dev_t

    key = pairs[:, 0] * 10_000_000 + pairs[:, 1] * 1_000_000 + pairs[:, 2] * 10 + pairs[:, 3]
    _, inv = np.unique(key, return_inverse=True)
    t_kf = (kf_ns - kf_ns[0]) * 1e-9
    base, rate, max_rot = (float(v) for v in a.gate.split(","))
    edges, gated = [], 0
    for j in range(inv.max() + 1 if len(inv) else 0):
        rows = np.flatnonzero(inv == j)
        ka, ca, kb, cb = pairs[rows[0]]
        fa = offsets[2 * (ka + k0) + ca] + matches[rows, 0]
        fb = offsets[2 * (kb + k0) + cb] + matches[rows, 1]
        for e in (edge(ka, ca, fa, kb, cb, fb), edge(kb, cb, fb, ka, ca, fa)):
            if e is None:
                continue
            if e[5] > base + rate * abs(t_kf[e[1]] - t_kf[e[0]]) or e[4] > max_rot:
                gated += 1
                continue
            edges.append(e)
    print(f"{inv.max() + 1 if len(inv) else 0} revisit image pairs: {len(edges)} loop edges, {gated} more over the "
          f"drift gate", flush=True)
    if not edges:
        sys.exit("no loop edges")
    dev = np.array([e[5] for e in edges])
    print(f"  disagreement with the solve: median {np.median(dev):.2f} m, p90 {np.percentile(dev, 90):.2f} m",
          flush=True)
    graph, vals = gtsam.NonlinearFactorGraph(), gtsam.Values()
    for k in range(n):
        vals.insert(X(k), gtsam.Pose3(gtsam.Rot3(r_wi[k]), t_wi[k]))
    graph.add(gtsam.PriorFactorPose3(X(0), vals.atPose3(X(0)), gtsam.noiseModel.Isotropic.Sigma(6, 1e-6)))
    odo = gtsam.noiseModel.Diagonal.Sigmas(np.r_[np.full(3, a.odo_rot_sigma), np.full(3, a.odo_pos_sigma)])
    for k in range(n - 1):
        graph.add(gtsam.BetweenFactorPose3(X(k), X(k + 1), vals.atPose3(X(k)).between(vals.atPose3(X(k + 1))), odo))
    loop_noise = gtsam.noiseModel.Robust.Create(
        gtsam.noiseModel.mEstimator.Cauchy.Create(1.0),
        gtsam.noiseModel.Diagonal.Sigmas(np.r_[np.full(3, a.loop_rot_sigma), np.full(3, a.loop_pos_sigma)]))
    for k3, k2, rel_r, rel_t, _, _ in edges:
        graph.add(gtsam.BetweenFactorPose3(X(k3), X(k2), gtsam.Pose3(gtsam.Rot3(rel_r), rel_t), loop_noise))
    res = gtsam.LevenbergMarquardtOptimizer(graph, vals, gtsam.LevenbergMarquardtParams()).optimize()
    new_r = np.array([res.atPose3(X(k)).rotation().matrix() for k in range(n)])
    new_t = np.array([res.atPose3(X(k)).translation() for k in range(n)])
    moved = np.linalg.norm(new_t - t_wi, axis=1)
    print(f"pose graph: error {graph.error(vals):.4g} -> {graph.error(res):.4g}; keyframes moved by "
          f"{np.median(moved):.2f} m at the median, {moved.max():.2f} m at most", flush=True)
    # the per-keyframe corrections, interpolated over the solve's per-frame output
    write_corrected(a.out, load_tum(a.tum), kf_ns, t_wi, r_wi, new_t, new_r)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("features", help="SIFT on the keyframes")
    f.add_argument("--mav0", required=True, help="ASL mav0 directory (cam0/data, cam1/data)")
    f.add_argument("--vio", required=True, help="VIO trajectory (TUM, seconds); keyframes need a pose")
    f.add_argument("--work", required=True)
    f.add_argument("--kf-step", type=int, default=10, help="every Nth stereo frame is a keyframe")
    f.add_argument("--num-features", type=int, default=1500)
    f.add_argument("--jobs", type=int, default=2)
    f.add_argument("--clahe", action="store_true", help="contrast-limited histogram equalization (low light)")
    m = sub.add_parser("match", help="VIO-guided epipolar matching between keyframes")
    m.add_argument("--work", required=True)
    m.add_argument("--vio", required=True)
    m.add_argument("--calib", required=True, help="Basalt calibration (pinhole)")
    m.add_argument("--offsets", default="1,2,3,4,6,8,12,16,24,32,48,64", help="keyframe offsets to match")
    m.add_argument("--max-axis-angle", type=float, default=75.0, help="skip pairs whose optical axes differ more")
    m.add_argument("--max-dist", type=float, default=80.0, help="skip pairs whose cameras are further apart (m)")
    m.add_argument("--tau0", type=float, default=2.0, help="epipolar band (px) at offset 0")
    m.add_argument("--tau-slope", type=float, default=0.1, help="band growth per keyframe offset (px)")
    m.add_argument("--tau-max", type=float, default=6.0)
    m.add_argument("--ratio", type=float, default=0.8, help="descriptor ratio test within the band")
    m.add_argument("--max-desc", type=float, default=300.0, help="maximum SIFT L2 distance")
    m.add_argument("--jobs", type=int, default=2)
    lp = sub.add_parser("loops", help="verified matches between revisits of the same place")
    lp.add_argument("--work", required=True)
    lp.add_argument("--vio", required=True)
    lp.add_argument("--calib", required=True)
    lp.add_argument("--min-gap", type=int, default=64, help="keyframes between the two visits")
    lp.add_argument("--max-dist", type=float, default=15.0, help="horizontal distance of the two keyframes (m)")
    lp.add_argument("--max-axis-angle", type=float, default=60.0)
    lp.add_argument("--step", type=int, default=2, help="candidate keyframe stride")
    lp.add_argument("--ratio", type=float, default=0.8)
    lp.add_argument("--max-desc", type=float, default=300.0)
    lp.add_argument("--min-inliers", type=int, default=25, help="essential-matrix inliers to accept a pair")
    lp.add_argument("--max-rot", type=float, default=10.0, help="largest disagreement with the VIO rotation (deg)")
    lp.add_argument("--jobs", type=int, default=2)
    s = sub.add_parser("solve", help="batch VI bundle adjustment")
    s.add_argument("--work", required=True)
    s.add_argument("--imu", required=True, help="IMU csv (ns, gyro, accel) as the VIO used it")
    s.add_argument("--calib", required=True)
    s.add_argument("--vio", required=True)
    s.add_argument("--out", required=True, help="corrected trajectory (TUM)")
    s.add_argument("--est-extrinsic", default="01", help="cameras whose IMU-camera rotation is estimated")
    s.add_argument("--max-offset", type=int, default=64, help="use matches up to this keyframe offset")
    s.add_argument("--band", type=int, default=64, help="landmark observations within this many keyframes")
    s.add_argument("--max-reproj", type=float, default=4.0, help="triangulation inlier threshold (px)")
    s.add_argument("--min-angle", type=float, default=1.0, help="minimum triangulation angle (deg)")
    s.add_argument("--pixel-sigma", type=float, default=1.0)
    s.add_argument("--huber", type=float, default=2.0, help="Huber threshold in sigmas")
    s.add_argument("--iterations", type=int, default=30)
    s.add_argument("--rel-tol", type=float, default=1e-5)
    s.add_argument("--lambda0", type=float, default=1e-4)
    s.add_argument("--outlier-px", type=float, default=10.0,
                   help="each iteration drops observations whose error exceeds this (0: keep all)")
    s.add_argument("--outlier-start", type=int, default=0, help="first iteration that drops outliers")
    s.add_argument("--outlier-px2", type=float, default=3.0,
                   help="tighter threshold (px) from --outlier-start2 on, once the solution has settled (0: off)")
    s.add_argument("--outlier-start2", type=int, default=8)
    s.add_argument("--omega-coriolis", default="", help="x,y,z: Earth's rotation (rad/s) in the VIO's world frame, "
                   "e.g. w_e (0, cos lat, sin lat) turned into it; needs the world frame's north")
    s.add_argument("--prior-rot-sigma", type=float, default=1e-4,
                   help="prior on the first keyframe's rotation (rad); its position stays pinned")
    s.add_argument("--lm-damp-floor", type=float, default=0.01,
                   help="landmark damping floor, relative to the landmark's largest diagonal entry")
    s.add_argument("--kf-range", default="", help="START:END keyframe window (default: all)")
    s.add_argument("--imu-noise-scale", type=float, default=1.0,
                   help="multiplies the calibration's accelerometer and gyroscope noise densities")
    s.add_argument("--accel-rw-scale", type=float, default=1.0, help="multiplies the accel bias random walk")
    s.add_argument("--gyro-rw-scale", type=float, default=1.0, help="multiplies the gyro bias random walk")
    s.add_argument("--loops", action="store_true", help="add the revisit matches of the loops stage")
    s.add_argument("--modes", default="", help="undamped deformation modes of the whole solution: yaw, scale "
                   "(comma-separated); they let LM move along slow drifts the damped steps cannot follow")
    s.add_argument("--modes-start", type=int, default=3, help="first iteration that uses the modes")
    s.add_argument("--mode-cap-yaw", type=float, default=0.01, help="largest yaw-mode step per iteration (rad)")
    s.add_argument("--mode-cap-scale", type=float, default=0.01, help="largest scale-mode step per iteration")
    s.add_argument("--save-states", action="store_true",
                   help="also write <out>_states.npz (keyframes, landmarks, observations) for loop-correct")
    c = sub.add_parser("loop-correct", help="close the revisits of a solve with a pose graph (after solve "
                       "--save-states and loops)")
    c.add_argument("--work", required=True)
    c.add_argument("--calib", required=True)
    c.add_argument("--states", required=True, help="<out>_states.npz of the solve")
    c.add_argument("--tum", required=True, help="the solve's output trajectory")
    c.add_argument("--out", required=True, help="corrected trajectory (TUM)")
    c.add_argument("--loops-file", default="loops.npz", help="revisit matches in --work")
    c.add_argument("--min-inliers", type=int, default=15, help="PnP inliers for a loop edge")
    c.add_argument("--gate", default="0.3,0.01,2.0",
                   help="drop an edge disagreeing with the solve by more than BASE + RATE * (time between the visits)"
                        " metres, or by more than MAX_ROT degrees: BASE,RATE,MAX_ROT")
    c.add_argument("--odo-rot-sigma", type=float, default=2e-4, help="rad, per consecutive keyframe pair")
    c.add_argument("--odo-pos-sigma", type=float, default=0.02, help="m, per consecutive keyframe pair")
    c.add_argument("--loop-rot-sigma", type=float, default=0.005, help="rad")
    c.add_argument("--loop-pos-sigma", type=float, default=0.1, help="m")
    a = p.parse_args()
    {"features": cmd_features, "match": cmd_match, "loops": cmd_loops, "solve": cmd_solve,
     "loop-correct": cmd_loop_correct}[a.cmd](a)


if __name__ == "__main__":
    main()
