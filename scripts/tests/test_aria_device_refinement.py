from pathlib import Path
import copy
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aria_device_refinement import (  # noqa: E402
    DEVICE_REFINEMENT, lookup, matrix_to_quat, quat_to_matrix, rotate_rig, rotvec_to_matrix)
from aria_factory_imu_rectify import delay_compensate  # noqa: E402


def calibration():
    # Two cameras: camera 0 rotated 104 deg, camera 1 almost 180 deg (w ~ 0).
    def cam(rotvec_deg, p):
        q = matrix_to_quat(rotvec_to_matrix(rotvec_deg))
        return {"px": p[0], "py": p[1], "pz": p[2],
                "qx": q[0], "qy": q[1], "qz": q[2], "qw": q[3]}
    return {"value0": {"T_imu_cam": [cam([104.2, 11.1, -7.1], [0.01, -0.05, 0.002]),
                                     cam([179.2, 2.3, -13.7], [0.03, 0.06, -0.01])]}}


def pose(cam):
    t = np.eye(4)
    t[:3, :3] = quat_to_matrix(cam["qx"], cam["qy"], cam["qz"], cam["qw"])
    t[:3, 3] = [cam["px"], cam["py"], cam["pz"]]
    return t


class QuaternionTests(unittest.TestCase):
    def test_round_trip_including_half_turns(self):
        rng = np.random.default_rng(0)
        vectors = [rng.normal(size=3) * 60 for _ in range(50)] + [[180, 0, 0], [0, 179.9, 1]]
        for v in vectors:
            m = rotvec_to_matrix(v)
            np.testing.assert_allclose(quat_to_matrix(*matrix_to_quat(m)), m, atol=1e-12)


class RotateRigTests(unittest.TestCase):
    def test_rotates_camera_zero_and_keeps_the_rig(self):
        before = calibration()
        delta = [0.3, 0.2, -0.1]
        after = rotate_rig(copy.deepcopy(before), delta)
        c0b, c1b = (pose(c) for c in before["value0"]["T_imu_cam"])
        c0a, c1a = (pose(c) for c in after["value0"]["T_imu_cam"])
        np.testing.assert_allclose(c0a[:3, :3], rotvec_to_matrix(delta) @ c0b[:3, :3], atol=1e-12)
        np.testing.assert_allclose(c0a[:3, 3], c0b[:3, 3], atol=1e-15)
        np.testing.assert_allclose(np.linalg.inv(c0a) @ c1a, np.linalg.inv(c0b) @ c1b, atol=1e-12)

    def test_zero_rotation_is_identity(self):
        before = calibration()
        after = rotate_rig(copy.deepcopy(before), [0, 0, 0])
        for a, b in zip(after["value0"]["T_imu_cam"], before["value0"]["T_imu_cam"]):
            np.testing.assert_allclose(pose(a), pose(b), atol=1e-12)


class DelayTests(unittest.TestCase):
    def test_row_holds_the_sample_stamped_delay_later(self):
        ts = np.arange(0, 20_000_000, 1_000_000, dtype=np.int64)  # 1 kHz
        values = np.column_stack([ts * 1e-6, -ts * 2e-6, np.sin(ts * 1e-7)])
        out = delay_compensate(ts, values, 3_000_000)
        np.testing.assert_allclose(out[:-3], values[3:], atol=1e-12)
        np.testing.assert_allclose(out[-1], values[-1])  # clamped at the end

    def test_zero_delay_is_a_no_op(self):
        values = np.ones((4, 3))
        self.assertIs(delay_compensate(np.arange(4), values, 0), values)


class LookupTests(unittest.TestCase):
    def test_known_and_unknown_serials(self):
        for serial, refinement in DEVICE_REFINEMENT.items():
            self.assertIs(lookup({"Serial": serial}), refinement)
        self.assertIsNone(lookup({"Serial": "1WM0000000000"}))
        self.assertIsNone(lookup({}))


if __name__ == "__main__":
    unittest.main()
