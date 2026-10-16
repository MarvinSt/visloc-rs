"""Unit tests for scripts/lamaria_batch_ba.py (numpy-only parts)."""
import importlib.util
import pathlib
import unittest

import numpy as np

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "lamaria_batch_ba.py"
spec = importlib.util.spec_from_file_location("lamaria_batch_ba", SCRIPT)
lb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lb)


def rot(rng, scale=1.0):
    return lb.so3_exp(rng.normal(size=3) * scale)


def two_cameras():
    k0 = np.array([[240.0, 0, 390], [0, 240.0, 290], [0, 0, 1]])
    k1 = np.array([[242.0, 0, 385], [0, 242.0, 296], [0, 0, 1]])
    return [{"R_ic": lb.so3_exp(np.array([0.2, -1.3, 0.1])), "t_ic": np.array([0.01, -0.1, 0.07]), "K": k0},
            {"R_ic": lb.so3_exp(np.array([-0.1, 0.4, 0.05])), "t_ic": np.array([0.0, 0.0, -0.006]), "K": k1}]


class ReprojectionJacobianTest(unittest.TestCase):
    def test_jacobians_match_finite_differences(self):
        rng = np.random.default_rng(3)
        cams = two_cameras()
        n_kf, n_lm = 3, 4
        r_wi = np.array([rot(rng, 0.3) for _ in range(n_kf)])
        t_wi = rng.normal(size=(n_kf, 3))
        r_ic = np.array([c["R_ic"] for c in cams])
        t_ic = np.array([c["t_ic"] for c in cams])
        # landmarks in front of every camera: place them along each camera's axis
        obs_kf, obs_cam, obs_lm = [], [], []
        pts = np.zeros((n_lm, 3))
        for j in range(n_lm):
            k, c = j % n_kf, j % 2
            r_wc = r_wi[k] @ r_ic[c]
            center = r_wi[k] @ t_ic[c] + t_wi[k]
            pts[j] = center + r_wc @ np.array([rng.normal() * 0.3, rng.normal() * 0.3, 4.0 + j])
            obs_kf.append(k)
            obs_cam.append(c)
            obs_lm.append(j)
        obs_z = np.zeros((n_lm, 2))
        rp = lb.Reprojection(np.array(obs_kf), np.array(obs_cam), np.array(obs_lm), obs_z, n_lm, cams,
                             [0, 1], 1.0, 1e9, 64)
        res, jx, jl, _ = rp.linearize(r_wi, t_wi, r_ic, t_ic, pts)
        eps = 1e-6

        def residuals(r_wi_, t_wi_, r_ic_, pts_):
            return rp._project(r_wi_, t_wi_, r_ic_, t_ic, pts_)[-1]

        for o in range(len(rp.kf)):
            k, c, j = rp.kf[o], rp.cam[o], rp.lm[o]
            for d in range(9):
                r2, t2, ric2 = r_wi.copy(), t_wi.copy(), r_ic.copy()
                step = np.zeros(3)
                step[d % 3] = eps
                if d < 3:      # pose rotation, right perturbation
                    r2[k] = r_wi[k] @ lb.so3_exp(step)
                elif d < 6:    # pose translation in the body frame
                    t2[k] = t_wi[k] + r_wi[k] @ step
                else:          # extrinsic rotation, right perturbation
                    ric2[c] = r_ic[c] @ lb.so3_exp(step)
                num = (residuals(r2, t2, ric2, pts)[o] - res[o]) / eps
                np.testing.assert_allclose(jx[o, :, d], num, rtol=1e-4, atol=1e-3)
            for d in range(3):
                p2 = pts.copy()
                p2[j, d] += eps
                num = (residuals(r_wi, t_wi, r_ic, p2)[o] - res[o]) / eps
                np.testing.assert_allclose(jl[o, :, d], num, rtol=1e-4, atol=1e-3)


class MatchingTest(unittest.TestCase):
    def test_best_two(self):
        keys = np.array([0, 0, 1, 0, 1])
        vals = np.array([5.0, 2.0, 7.0, 3.0, 1.0], np.float32)
        best, second = lb._best_two(keys, vals, 3)
        self.assertEqual(best.tolist(), [1, 4, -1])
        self.assertEqual(second[:2].tolist(), [3.0, 7.0])
        self.assertTrue(np.isinf(second[2]))

    def test_epipolar_matcher_recovers_true_correspondences(self):
        rng = np.random.default_rng(7)
        cams = two_cameras()
        k = cams[0]["K"]
        # two keyframes 0.8 m apart; points 3-20 m in front of camera 0 of keyframe 0
        kf_r = np.array([np.eye(3), lb.so3_exp(np.array([0.0, 0.05, 0.0]))])
        kf_p = np.array([[0.0, 0.0, 0.0], [0.8, 0.1, 0.0]])
        lb._G.update(cams=[cams[0], cams[0]], kf_p=kf_p, kf_r=kf_r, ratio=0.8, max_desc=300.0)
        r_wc0, c0 = lb._cam_pose(0, 0)
        r_wc1, c1 = lb._cam_pose(1, 0)
        n = 200
        xy0 = rng.uniform([50, 50], [730, 530], size=(n, 2))
        depth = rng.uniform(3, 20, size=n)
        rays = np.c_[xy0, np.ones(n)] @ np.linalg.inv(k).T
        world = c0 + (rays * depth[:, None]) @ r_wc0.T
        cam1 = (world - c1) @ r_wc1
        xy1 = (cam1 @ k.T)[:, :2] / cam1[:, 2:3]
        desc0 = rng.integers(0, 256, size=(n, 128), dtype=np.uint8)
        noise = rng.integers(-3, 4, size=(n, 128))
        desc1 = np.clip(desc0.astype(int) + noise, 0, 255).astype(np.uint8)
        perm = rng.permutation(n)  # shuffle image 1 so a match is not just the identity
        xy = np.concatenate([xy0, xy1[perm]]).astype(np.float32)
        desc = np.concatenate([desc0, desc1[perm]])
        lb._G.update(offsets=np.array([0, n, n, 2 * n, 2 * n]), xy=xy, desc=desc)
        saved = lb.njit
        try:
            lb.njit = None  # numpy path
            _, m = lb._match_worker((0, 0, 1, 0, 2.0))
        finally:
            lb.njit = saved
        self.assertGreater(len(m), 0.9 * n)
        inverse = np.argsort(perm)
        np.testing.assert_array_equal(m[:, 1], inverse[m[:, 0]])

    def test_unguided_matches_numpy_path(self):
        rng = np.random.default_rng(11)
        desc_a = rng.integers(0, 256, size=(80, 128), dtype=np.uint8)
        perm = rng.permutation(80)
        desc_b = desc_a[perm]
        saved = lb.njit
        try:
            lb.njit = None
            m = lb._unguided_matches(desc_a, desc_b, 0.8, 300.0)
        finally:
            lb.njit = saved
        self.assertEqual(len(m), 80)
        np.testing.assert_array_equal(perm[m[:, 1]], m[:, 0])


if __name__ == "__main__":
    unittest.main()
