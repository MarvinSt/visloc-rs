#!/usr/bin/env python3
"""Score a SmallCity reconstruction against the release's own COLMAP model.

The Hierarchical 3DGS SmallCity release ships a COLMAP reconstruction of all
5,822 photos (``camera_calibration/aligned/sparse/0``, metric). It serves as
the reference here; it is not ground truth. ``fetch`` streams just the camera
poses out of its 640 MB ``images.bin`` from the 3 GB release zip, using HTTP
range requests (about 300 MB transferred, nothing large written). ``score``
Sim(3)-aligns a COLMAP text ``images.txt`` to it by camera centres. It
reports the median / p90 / max centre error in metres, plus a separate Sim(3)
fit per capture pass (``pass1``..``pass4``), which separates drift *within*
a pass from misalignment *between* passes. ``gsplat_photos`` names its
working copies ``frame_00000.png``... in sorted input order; such names are
mapped to the sorted release names.

    python3 scripts/score_smallcity_reference.py fetch ref_images.txt
    python3 scripts/score_smallcity_reference.py score ref_images.txt out/sparse/0/images.txt
"""

from __future__ import annotations

import argparse
import io
import struct
import sys
import urllib.request
import zipfile

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover
    raise SystemExit("score_smallcity_reference.py requires numpy") from exc

URL = (
    "https://repo-sam.inria.fr/fungraph/hierarchical-3d-gaussians/"
    "datasets/full_scenes/small_city.zip"
)
MEMBER = "small_city/camera_calibration/aligned/sparse/0/images.bin"


class RangeFile(io.RawIOBase):
    """A seekable read-only view of a remote file over HTTP range requests."""

    def __init__(self, url: str) -> None:
        self.url = url
        head = urllib.request.urlopen(urllib.request.Request(url, method="HEAD"))
        self.size = int(head.headers["Content-Length"])
        self.pos = 0

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        base = {0: 0, 1: self.pos, 2: self.size}[whence]
        self.pos = base + offset
        return self.pos

    def readinto(self, buf) -> int:
        n = min(len(buf), self.size - self.pos)
        if n <= 0:
            return 0
        req = urllib.request.Request(
            self.url, headers={"Range": f"bytes={self.pos}-{self.pos + n - 1}"}
        )
        data = urllib.request.urlopen(req).read()
        buf[: len(data)] = data
        self.pos += len(data)
        return len(data)


def fetch(out_path: str) -> None:
    """Write the reference poses as COLMAP ``images.txt`` lines (no points)."""
    archive = zipfile.ZipFile(io.BufferedReader(RangeFile(URL), 8 << 20))
    f = archive.open(MEMBER)

    def read(n: int) -> bytes:
        data = f.read(n)
        if len(data) != n:
            raise SystemExit("truncated images.bin")
        return data

    (count,) = struct.unpack("<Q", read(8))
    with open(out_path, "w") as out:
        for _ in range(count):
            (image_id,) = struct.unpack("<i", read(4))
            qvec = struct.unpack("<4d", read(32))
            tvec = struct.unpack("<3d", read(24))
            (camera_id,) = struct.unpack("<i", read(4))
            name = b""
            while (c := read(1)) != b"\0":
                name += c
            (num_points,) = struct.unpack("<Q", read(8))
            left = num_points * 24
            while left:
                left -= len(f.read(min(left, 1 << 22)))
            fields = [image_id, *qvec, *tvec, camera_id, name.decode()]
            out.write(" ".join(map(str, fields)) + "\n")
    print(f"{count} reference poses -> {out_path}")


def centres(path: str) -> dict[str, np.ndarray]:
    """Camera centres by image name from a COLMAP text ``images.txt``."""
    out: dict[str, np.ndarray] = {}
    with open(path) as f:
        for line in f:
            e = line.split()
            if line.startswith("#") or len(e) < 10 or "." not in e[9]:
                continue
            qw, qx, qy, qz, tx, ty, tz = map(float, e[1:8])
            r = np.array(
                [
                    [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                    [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                    [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
                ]
            )
            out[e[9]] = -r.T @ np.array([tx, ty, tz])
    return out


def sim3_errors(ref: np.ndarray, est: np.ndarray) -> tuple[np.ndarray, float]:
    """Centre errors after the least-squares Sim(3) taking ``est`` onto ``ref``."""
    mr, me = ref.mean(0), est.mean(0)
    r0, e0 = ref - mr, est - me
    u, s, vt = np.linalg.svd(r0.T @ e0)
    d = np.eye(3)
    d[2, 2] = np.sign(np.linalg.det(u @ vt))
    rot = u @ d @ vt
    scale = (s * np.diag(d)).sum() / (e0**2).sum()
    return np.linalg.norm(r0 - scale * (e0 @ rot.T), axis=1), scale


def score(ref_path: str, est_path: str) -> None:
    ref, est = centres(ref_path), centres(est_path)
    if any(n.startswith("frame_") for n in est):
        sorted_ref = sorted(ref)
        est = {sorted_ref[int(n[6:].split(".")[0])]: c for n, c in est.items()}
    names = sorted(set(ref) & set(est))
    if len(names) < 3:
        raise SystemExit(f"only {len(names)} images in common with the reference")
    a = np.array([ref[n] for n in names])
    b = np.array([est[n] for n in names])
    err, _ = sim3_errors(a, b)
    print(
        f"{est_path}: {len(est)} registered, {len(names)} of {len(ref)} reference images; "
        f"centre error median {np.median(err):.3f} m, p90 {np.percentile(err, 90):.3f} m, "
        f"max {err.max():.2f} m"
    )
    passes = np.array([n.split("_")[0] for n in names])
    for p in sorted(set(passes)):
        m = passes == p
        own, scale = sim3_errors(a[m], b[m])
        print(
            f"  {p}: {m.sum()} images; global fit median {np.median(err[m]):.3f} m; "
            f"own fit median {np.median(own):.3f} m, p90 {np.percentile(own, 90):.3f} m, "
            f"scale {scale:.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="stream the reference poses into a text file")
    f.add_argument("out")
    s = sub.add_parser("score", help="score an images.txt against the reference")
    s.add_argument("reference")
    s.add_argument("images", nargs="+")
    args = parser.parse_args()
    if args.cmd == "fetch":
        fetch(args.out)
    else:
        for path in args.images:
            score(args.reference, path)


if __name__ == "__main__":
    sys.exit(main())
