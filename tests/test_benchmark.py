"""Check measurement validity and reproducibility, not a desired success score."""
import tempfile
from pathlib import Path
import unittest

import numpy as np

from cleaning_robot.benchmark import (bootstrap_ratio, distribution, make_protocol,
                                     make_trial, run_arm, run_benchmark, wilson_interval)


class BenchmarkTests(unittest.TestCase):
    def test_all_success_interval_still_expresses_sample_uncertainty(self):
        lo, hi = wilson_interval(100, 100)
        self.assertAlmostEqual(lo, .9630065, places=6)
        self.assertAlmostEqual(hi, 1.)
        self.assertIsNone(wilson_interval(0, 0))

    def test_bootstrap_resamples_scenes_and_handles_zero_denominators(self):
        result = bootstrap_ratio([4, 0], [4, 4])
        self.assertEqual(result["value"], .5)
        self.assertEqual(result["ci95_scene_bootstrap"], [0., 1.])
        self.assertIsNone(bootstrap_ratio([0, 0], [0, 0]))

    def test_seeded_layouts_and_assumptions_are_reproducible(self):
        left = make_trial(2026101801, "disturbance")
        right = make_trial(2026101801, "disturbance")
        self.assertEqual(left[3], right[3])
        self.assertEqual([s.radius_m for s in left[2]], [s.radius_m for s in right[2]])
        for a, b in zip(left[1], right[1]):
            np.testing.assert_array_equal(a.position, b.position)
            self.assertEqual(a.yaw_rad, b.yaw_rad)
        self.assertEqual(len(left[1]), 4)
        self.assertEqual(len(left[2]), 2)

    def test_timeout_is_not_success_and_leaves_all_objects_in_denominator(self):
        cfg, litter, stains, assumptions = make_trial(2026101802, "nominal")
        row = run_arm(cfg, litter, stains, assumptions, enabled=True, max_steps=1, unload_delay_s=10)
        self.assertTrue(row["timed_out"])
        self.assertFalse(row["full_success"])
        self.assertEqual(len(row["summary"]["objects"]), 4)
        self.assertAlmostEqual(sum(row["state_time_s"].values()), row["elapsed_s"])
        self.assertAlmostEqual(row["active_s"]+row["unload_wait_s"], row["elapsed_s"])

    def test_one_pair_report_includes_both_timeouts_and_preserves_evidence(self):
        protocol = make_protocol(1, 2026101803, 1, 10)
        with tempfile.TemporaryDirectory() as directory:
            report = run_benchmark(protocol, directory)
            self.assertEqual(report["aggregate"]["objects"], 4)
            self.assertEqual(report["aggregate"]["enhanced_timeouts"], 1)
            self.assertEqual(report["aggregate"]["baseline_timeouts"], 1)
            self.assertEqual(report["aggregate"]["paired_completed_count"], 0)
            self.assertIsNone(report["aggregate"]["paired_elapsed_ratio"])
            self.assertEqual(len(Path(directory, "trials.jsonl").read_text().splitlines()), 1)
            with self.assertRaises(FileExistsError):
                run_benchmark(protocol, directory)

    def test_invalid_protocol_and_empty_distributions(self):
        for args in ((0, 1, 10, 10), (1, -1, 10, 10), (1, 1, 0, 10), (1, 1, 10, float("nan"))):
            with self.assertRaises(ValueError):
                make_protocol(*args)
        self.assertIsNone(distribution([]))


if __name__ == "__main__":
    unittest.main()
