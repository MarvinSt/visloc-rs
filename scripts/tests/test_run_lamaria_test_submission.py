from pathlib import Path
import argparse
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_lamaria_test_submission as submission


def make_args(**overrides):
    args = argparse.Namespace(vio_exe=Path("vio_demo"), slam_exe=None,
                              config=Path("cfg.json"), threads=12)
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


class EstimatorCommandTests(unittest.TestCase):
    def test_vio_only(self):
        command, trajectories = submission.estimator_command(
            make_args(), Path("seq"), Path("calib.json"), Path("out"))
        self.assertEqual(command[0], Path("vio_demo"))
        for flag in ("--pipeline", "--no-trace", "--no-marg-data"):
            self.assertIn(flag, command)
        self.assertEqual(trajectories, [Path("out/trajectory.tum")])

    def test_online_mapper_takes_precedence(self):
        command, trajectories = submission.estimator_command(
            make_args(slam_exe=Path("slam_demo")), Path("seq"), Path("calib.json"), Path("out"))
        self.assertEqual(command[0], Path("slam_demo"))
        self.assertIn("--no-urgent-keyframes", command)
        self.assertNotIn("--no-marg-data", command)
        self.assertEqual(command[command.index("--euroc-dir") + 1], Path("seq"))
        self.assertEqual(command[command.index("--calibration") + 1], Path("calib.json"))
        self.assertEqual(command[command.index("--threads") + 1], 12)
        self.assertEqual(trajectories, [Path("out/trajectory_online.tum"),
                                        Path("out/trajectory_vio.tum")])


if __name__ == "__main__":
    unittest.main()
