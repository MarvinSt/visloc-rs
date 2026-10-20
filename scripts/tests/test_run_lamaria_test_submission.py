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
        self.assertEqual(command[command.index("--loop-match-max-rot-error") + 1], 5)
        self.assertNotIn("--no-marg-data", command)
        self.assertEqual(command[command.index("--euroc-dir") + 1], Path("seq"))
        self.assertEqual(command[command.index("--calibration") + 1], Path("calib.json"))
        self.assertEqual(command[command.index("--threads") + 1], 12)
        self.assertEqual(trajectories, [Path("out/trajectory_online.tum"),
                                        Path("out/trajectory_vio.tum")])

    def test_vio_track_with_the_online_demo_only_submits_its_vio_trajectory(self):
        command, trajectories = submission.estimator_command(
            make_args(vio_exe=None, slam_exe=Path("slam_demo")), Path("seq"),
            Path("calib.json"), Path("out"), submit_vio=True)
        self.assertEqual(command[0], Path("slam_demo"))
        self.assertEqual(trajectories, [Path("out/trajectory_vio.tum")])

    def test_vio_track_prefers_the_vio_only_build(self):
        command, trajectories = submission.estimator_command(
            make_args(slam_exe=Path("slam_demo")), Path("seq"), Path("calib.json"),
            Path("out"), submit_vio=True)
        self.assertEqual(command[0], Path("vio_demo"))
        self.assertEqual(trajectories, [Path("out/trajectory.tum")])

    def test_per_sequence_config_overrides_the_default(self):
        command, _ = submission.estimator_command(
            make_args(), Path("seq"), Path("calib.json"), Path("out"),
            config=Path("lowlight.json"))
        self.assertEqual(command[command.index("--config") + 1], Path("lowlight.json"))


class TrackTests(unittest.TestCase):
    def test_sequence_track(self):
        self.assertEqual(submission.sequence_track("sequence_4_10"), 4)
        self.assertEqual(submission.sequence_track("sequence_1_19"), 1)

    def test_low_light_config_ships_with_the_repo(self):
        repo = Path(__file__).resolve().parents[2]
        self.assertTrue((repo / submission.DEFAULT_LOW_LIGHT_CONFIG).is_file())

    def test_ba_chain_configs_ship_with_the_repo(self):
        repo = Path(__file__).resolve().parents[2]
        for config in (submission.DEFAULT_CONFIG, submission.BA_CHAIN_CONFIG, submission.BA_CHAIN_LOW_LIGHT_CONFIG):
            self.assertTrue((repo / config).is_file(), config)


if __name__ == "__main__":
    unittest.main()
