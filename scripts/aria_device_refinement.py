#!/usr/bin/env python3
"""Per-device refinement of the Aria IMU-camera calibration for LaMAria.

LaMAria's whole dataset comes from two Aria devices. On both, the factory
calibration in the `.vrs` header leaves two errors that partly cancel in the
VIO:

* **Time.** The gyroscope stamps its samples about 4.6 ms after the camera
  image they belong to. The factory `TimeOffsetSec_Device_Gyro` is 4.1 ms.
* **Rotation.** The rotation between the IMU and the camera rig is off by
  0.3-0.7 degrees.

Both were fitted by integrating the rectified gyroscope between LaMAria
training pseudo-GT keyframes and matching the pseudo-GT rotations, with a
free time offset, constant bias and IMU-to-camera rotation
(docs/lamaria_drift_analysis.md). The accelerometer delay is the factory
gyroscope-accelerometer difference applied to the fitted gyroscope delay.

Only training data was used: device 1WM093701G1276 from sequence_2_11,
sequence_3_17 and sequence_4_10, device 1WM09370081264 from sequence_1_19.
"""

import numpy as np

# serial -> gyro delay (ns), accel delay (ns), rig rotation correction
# (rotation vector in degrees, IMU frame: R_imu_cam <- R(delta) * R_imu_cam).
DEVICE_REFINEMENT = {
    "1WM093701G1276": {"gyro_delay_ns": 4_600_000, "accel_delay_ns": 3_580_000,
                       "rig_rotation_deg": [0.276, 0.217, -0.077]},
    "1WM09370081264": {"gyro_delay_ns": 4_600_000, "accel_delay_ns": 3_140_000,
                       "rig_rotation_deg": [0.566, -0.290, 0.369]},
}


def lookup(factory: dict):
    """The refinement for the device that recorded `factory`, or None."""
    return DEVICE_REFINEMENT.get(factory.get("Serial"))


def quat_to_matrix(x, y, z, w):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def matrix_to_quat(m):
    """(x, y, z, w) with w >= 0 (Shepperd's method: divide by the largest component)."""
    trace = np.trace(m)
    k = int(np.argmax([trace, m[0, 0], m[1, 1], m[2, 2]]))
    if k == 0:
        s = 2 * np.sqrt(1 + trace)
        q = [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, s / 4]
    elif k == 1:
        s = 2 * np.sqrt(1 + m[0, 0] - m[1, 1] - m[2, 2])
        q = [s / 4, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s]
    elif k == 2:
        s = 2 * np.sqrt(1 - m[0, 0] + m[1, 1] - m[2, 2])
        q = [(m[0, 1] + m[1, 0]) / s, s / 4, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s]
    else:
        s = 2 * np.sqrt(1 - m[0, 0] - m[1, 1] + m[2, 2])
        q = [(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, s / 4, (m[1, 0] - m[0, 1]) / s]
    q = np.array(q) / np.linalg.norm(q)
    return -q if q[3] < 0 else q


def rotvec_to_matrix(rotvec_deg):
    v = np.radians(np.asarray(rotvec_deg, dtype=float))
    angle = np.linalg.norm(v)
    if angle == 0:
        return np.eye(3)
    k = v / angle
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle) * kx + (1 - np.cos(angle)) * kx @ kx


def rotate_rig(calibration: dict, rotvec_deg) -> dict:
    """Rotate every camera of a Basalt calibration about camera 0 in the IMU frame.

    `R_imu_cam <- delta * R_imu_cam` and `p_cam <- p_cam0 + delta (p_cam - p_cam0)`,
    so the cameras keep their relative poses. Modifies and returns `calibration`.
    """
    delta = rotvec_to_matrix(rotvec_deg)
    cams = calibration["value0"]["T_imu_cam"]
    p0 = np.array([cams[0]["px"], cams[0]["py"], cams[0]["pz"]])
    for cam in cams:
        r = delta @ quat_to_matrix(cam["qx"], cam["qy"], cam["qz"], cam["qw"])
        p = p0 + delta @ (np.array([cam["px"], cam["py"], cam["pz"]]) - p0)
        q = matrix_to_quat(r)
        cam.update(px=float(p[0]), py=float(p[1]), pz=float(p[2]),
                   qx=float(q[0]), qy=float(q[1]), qz=float(q[2]), qw=float(q[3]))
    return calibration
