"""Resample an ETH3D scene's undistorted images onto its most used camera.

Usage: eth3d_one_camera.py CALIBRATION_DIR IMAGE_DIR OUT_DIR

ETH3D's DSLR scenes with several cameras have slightly different pinhole
intrinsics and image sizes per camera. A pipeline that takes one camera
for every image (gsplat_photos, or COLMAP with fixed intrinsics) then
matches with intrinsics a few pixels off. The images are pinhole, so an
image of camera K_i is exactly an image of camera K by the affine map
K K_i^-1: each image is resampled (bilinearly) onto the most used camera,
which leaves the ground-truth poses valid.

Writes OUT_DIR/images/<name> (the dominant camera's images copied, the
others resampled) and OUT_DIR/calibration/{cameras,images,points3D}.txt
with that one camera, the same poses and the observations moved to it.
"""
import collections
import os
import shutil
import sys

from PIL import Image


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    calib, image_dir, out = sys.argv[1:]
    cams = {}
    for line in open(os.path.join(calib, "cameras.txt")):
        if line.strip() and not line.startswith("#"):
            t = line.split()
            if t[1] != "PINHOLE":
                sys.exit(f"camera {t[0]}: {t[1]}, not PINHOLE")
            cams[t[0]] = (int(t[2]), int(t[3])) + tuple(map(float, t[4:8]))
    lines = [l.rstrip("\n") for l in open(os.path.join(calib, "images.txt"))
             if not l.startswith("#")]
    entries = []
    i = 0
    while i < len(lines):
        t = lines[i].split()
        if len(t) >= 10:
            entries.append((t, lines[i + 1] if i + 1 < len(lines) else ""))
            i += 2
        else:
            i += 1
    dominant = collections.Counter(t[8] for t, _ in entries).most_common(1)[0][0]
    w, h, fx, fy, cx, cy = cams[dominant]
    os.makedirs(os.path.join(out, "images"), exist_ok=True)
    os.makedirs(os.path.join(out, "calibration"), exist_ok=True)
    out_images = []
    resampled = 0
    for t, obs in entries:
        name, cam = t[9], t[8]
        src = os.path.join(image_dir, name)
        dst = os.path.join(out, "images", name)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        _, _, fxi, fyi, cxi, cyi = cams[cam]
        if cam == dominant:
            shutil.copyfile(src, dst)
        else:
            # Output pixel (u, v) of camera K samples (u_i, v_i) =
            # K_i K^-1 (u, v); PIL takes the input coordinates as an affine
            # function of the output's, both with pixel centres at +0.5.
            a, c = fxi / fx, cxi - fxi / fx * cx
            e, f = fyi / fy, cyi - fyi / fy * cy
            img = Image.open(src)
            img.transform((w, h), Image.Transform.AFFINE, (a, 0.0, c, 0.0, e, f),
                          resample=Image.Resampling.BILINEAR).save(dst, quality=95)
            resampled += 1
        # Observations move with the image.
        sx, ox = fx / fxi, cx - fx / fxi * cxi
        sy, oy = fy / fyi, cy - fy / fyi * cyi
        o = obs.split()
        moved = []
        for k in range(0, len(o) - 2, 3):
            moved += [f"{float(o[k]) * sx + ox:.6f}", f"{float(o[k + 1]) * sy + oy:.6f}", o[k + 2]]
        out_images.append(" ".join(t[:8] + [dominant] + t[9:]) + "\n" + " ".join(moved) + "\n")
    with open(os.path.join(out, "calibration", "cameras.txt"), "w") as f:
        f.write(f"# One camera: {dominant} of {calib}\n")
        f.write(f"{dominant} PINHOLE {w} {h} {fx} {fy} {cx} {cy}\n")
    with open(os.path.join(out, "calibration", "images.txt"), "w") as f:
        f.writelines(out_images)
    shutil.copyfile(os.path.join(calib, "points3D.txt"),
                    os.path.join(out, "calibration", "points3D.txt"))
    print(f"{len(entries)} images onto camera {dominant} ({w}x{h}); {resampled} resampled")


if __name__ == "__main__":
    main()
