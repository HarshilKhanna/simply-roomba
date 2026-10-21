"""Behavioral checks for the hardware-free collector mission.

Run: python -m unittest discover -s tests -v
"""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from cleaning_robot.collector import (CollectorCommand, CollectorFeedback,
                                      collector_navigation_radius, pickup_rejection)
from cleaning_robot.config import CollectorConfig, PerceptionConfig, RobotConfig
from cleaning_robot.motion_planning import MissionPlanner, MissionState, generate_boustrophedon
from cleaning_robot.navigation import ArenaNavigator
from cleaning_robot.perception import (Detection, DetectionTracker,
    depth_image_to_meters, detection_to_planning_frame, interpolate_planar_pose)
from cleaning_robot.simulation import (CleaningSimulation, D405Camera, Litter,
                                       simulation_config)


def observation(x=1.0, y=0.0, *, kind="debris", object_class="paper_ball", **kwargs):
    return Detection(kind, np.array([x, y, 0.02]), 0.045, (0, 0),
                     object_class=object_class if kind == "debris" else None,
                     size_m=(0.065, 0.065, 0.065), estimated_mass_kg=0.008, **kwargs)


def advance_until(sim, state, limit=5000):
    for _ in range(limit):
        sim.step()
        if sim.mission.state == state:
            return
        if sim.mission.state == MissionState.FAULT:
            raise AssertionError(sim.mission.events[-1])
    raise AssertionError(f"did not reach {state.name}: {sim.summary()}")


class MissionBehaviorTests(unittest.TestCase):
    def test_all_four_litter_classes_and_stains_complete(self):
        sim = CleaningSimulation()
        self.assertIsInstance(sim.mission, MissionPlanner)
        result = sim.run(20000, auto_unload=True)
        self.assertEqual(result["state"], "DONE", sim.mission.events[-5:])
        self.assertEqual(result["delivered"], 4)
        self.assertEqual(result["floor_litter"], 0)
        self.assertEqual(result["stains_cleaned"], 3)
        self.assertEqual(result["safety_stops"], 0)
        self.assertEqual(result["coverage_index"], result["coverage_segments"])
        events = sim.mission.events
        returns = [e for e in events if e["state"] == "RETURNING_HOME"]
        waits = [e for e in events if e["state"] == "WAITING_FOR_UNLOAD"]
        self.assertEqual(len(returns), 4)
        self.assertEqual(len(waits), 4)
        self.assertEqual({x.object_class for x in sim.litter if x.status == "delivered"},
                         {"plastic_bottle", "wrapper", "paper_cup", "paper_ball"})

    def test_manual_wait_retains_payload_and_checkpoint(self):
        sim = CleaningSimulation()
        self.assertFalse(sim.unload())
        advance_until(sim, MissionState.WAITING_FOR_UNLOAD)
        home = sim.mission.home_pose
        self.assertLessEqual(np.linalg.norm(np.array(sim.pose[:2]) - home[:2]),
                             sim.cfg.collector.home_tolerance_m)
        self.assertLessEqual(abs(sim.mission._angle_error(sim.pose[2], home[2])),
                             sim.cfg.collector.heading_tolerance_rad)
        before_pose = sim.pose
        checkpoint = sim.mission._resume_pose
        resume_index = sim.mission.resume_index
        payload = sim.collector.payload
        for _ in range(100):
            self.assertEqual(sim.step(), (0.0, 0.0))
        self.assertEqual(sim.pose, before_pose)
        self.assertIs(sim.collector.payload, payload)
        self.assertEqual(payload.status, "carried")
        self.assertFalse(sim.mission.mop_enabled)
        self.assertEqual(sim.deliveries, 0)
        self.assertTrue(sim.unload())
        self.assertFalse(sim.unload())
        advance_until(sim, MissionState.COVERAGE)
        self.assertEqual(sim.mission.coverage_index, resume_index)
        self.assertLessEqual(np.linalg.norm(np.array(sim.pose[:2]) - checkpoint[:2]),
                             sim.cfg.collector.home_tolerance_m)
        self.assertEqual(payload.status, "delivered")

    def test_failed_pickup_retries_same_object_before_return(self):
        sim = CleaningSimulation()
        sim.force_next_failure()
        advance_until(sim, MissionState.WAITING_FOR_UNLOAD)
        self.assertEqual(sim.litter[0].attempts, 2)
        self.assertEqual(sim.deliveries, 0)
        messages = [e["message"] for e in sim.mission.events]
        self.assertIn("payload verification failed; retry", messages)
        self.assertEqual(sum(e["state"] == "RETURNING_HOME" for e in sim.mission.events), 1)
        self.assertEqual(sim.safety_stops, 0)

    def test_permanent_failure_is_bounded_and_coverage_continues(self):
        item = Litter(1, "plastic_bottle", np.array([0.98, 0.50]),
                      (0.19, 0.065, 0.065), 0.025)
        sim = CleaningSimulation(litter=[item], stains=[])
        sim.collector.success_probability = 0.0
        result = sim.run(20000, auto_unload=True)
        self.assertEqual(result["state"], "DONE", sim.mission.events[-5:])
        self.assertEqual(sim.litter[0].attempts, sim.cfg.collector.max_pickup_attempts)
        self.assertEqual(result["delivered"], 0)
        self.assertEqual(result["floor_litter"], 1)
        self.assertEqual(len(result["failed_tracks"]), 1)
        self.assertFalse(any(e["state"] == "RETURNING_HOME" for e in sim.mission.events))
        self.assertEqual(result["safety_stops"], 0)

    def test_missing_actuator_feedback_never_proves_pickup(self):
        sim = CleaningSimulation()
        advance_until(sim, MissionState.COLLECTING)
        for _ in range(120):
            velocity = sim.mission.update(sim.pose, [], [], dt=0.1, collector_feedback=None)
            self.assertEqual(velocity, (0.0, 0.0))
        self.assertEqual(sim.mission.state, MissionState.FAULT)
        self.assertIsNone(sim.mission.carried_target)
        self.assertFalse(any(e["state"] == "RETURNING_HOME" for e in sim.mission.events))

    def test_lost_payload_stops_transport(self):
        sim = CleaningSimulation()
        advance_until(sim, MissionState.RETURNING_HOME)
        sim.collector.payload = None
        sim.collector.feedback = CollectorFeedback(CollectorCommand.HOLD, True, False)
        pose = sim.pose
        self.assertEqual(sim.step(), (0.0, 0.0))
        self.assertEqual(sim.pose, pose)
        self.assertEqual(sim.mission.state, MissionState.FAULT)
        self.assertIn("payload lost", sim.mission.events[-1]["message"])

    def test_collection_defaults_off_for_existing_callers(self):
        cfg = RobotConfig()
        planner = MissionPlanner(cfg.planning, cfg.geometry, np.array([[0., 0.], [2., 0.]]))
        debris = observation(0.4, track_id=1, confidence=5)
        planner.update((0., 0., 0.), [], [debris])
        self.assertEqual(planner.state, MissionState.COVERAGE)
        self.assertEqual(planner.collector_command, CollectorCommand.STOW)

    def test_missing_payload_telemetry_stops_return(self):
        sim = CleaningSimulation()
        advance_until(sim, MissionState.RETURNING_HOME)
        for _ in range(30):
            velocity = sim.mission.update(sim.pose, [], [], dt=.1, collector_feedback=None)
        self.assertEqual(sim.mission.state, MissionState.FAULT)
        self.assertEqual(velocity, (0., 0.))
        self.assertIn("feedback lost", sim.mission.events[-1]["message"])

    def test_return_timeout_stops_on_the_same_control_tick(self):
        sim = CleaningSimulation()
        advance_until(sim, MissionState.RETURNING_HOME)
        sim.mission.collector_cfg.navigation_timeout_s = .01
        velocity = sim.mission.update(sim.pose, [], [], dt=.1,
                    collector_feedback=CollectorFeedback(CollectorCommand.HOLD, True, True))
        self.assertEqual(sim.mission.state, MissionState.FAULT)
        self.assertEqual(velocity, (0., 0.))


class ObservationTests(unittest.TestCase):
    def test_base_to_fixed_transform_is_explicit_and_does_not_mutate(self):
        det = observation(0.3, 0., orientation_rad=0.)
        fixed = detection_to_planning_frame(det, (1., 2., np.pi/2))
        np.testing.assert_allclose(fixed.position_robot, [1., 2.3, 0.02])
        np.testing.assert_allclose(det.position_robot, [0.3, 0., 0.02])
        self.assertAlmostEqual(fixed.orientation_rad, np.pi/2)
        again = detection_to_planning_frame(fixed, (8., 9., 1.))
        np.testing.assert_allclose(again.position_robot, fixed.position_robot)
        with self.assertRaises(ValueError):
            detection_to_planning_frame(replace(det, coordinate_frame="camera_optical"), (0., 0., 0.))

    def test_robot_motion_does_not_change_fixed_target_identity(self):
        tracker = DetectionTracker(PerceptionConfig())
        for shift in (0., 0.1, 0.2):
            det = observation(0.4-shift)
            confirmed = tracker.update([detection_to_planning_frame(det, (shift, 0., 0.))])
        self.assertEqual(len(confirmed), 1)
        self.assertEqual(confirmed[0].track_id, 0)
        self.assertAlmostEqual(confirmed[0].position_robot[0], 0.4)

    def test_confirmation_requires_consecutive_sensor_frames(self):
        tracker = DetectionTracker(PerceptionConfig())
        for detections in ([observation()], [], [observation()], [], [observation()]):
            self.assertEqual(tracker.update(detections), [])
        tracker.update([observation()])
        self.assertEqual(len(tracker.update([observation()])), 1)

    def test_three_components_cannot_confirm_in_one_frame(self):
        tracker = DetectionTracker(PerceptionConfig())
        self.assertEqual(tracker.update([observation(1.), observation(1.01), observation(1.02)]), [])
        self.assertEqual(len(tracker._tracks), 3)
        self.assertTrue(all(t.confidence == 1 for t in tracker._tracks))

    def test_persistent_static_tracks_reacquire_without_duplicate_ids(self):
        tracker = DetectionTracker(PerceptionConfig(), max_missed_frames=None)
        for _ in range(3):
            confirmed = tracker.update([observation()])
        original = confirmed[0]
        for _ in range(20):
            self.assertEqual(tracker.update([]), [])
        for _ in range(3):
            confirmed = tracker.update([observation()])
        self.assertEqual(confirmed[0].track_id, original.track_id)

    def test_ros_depth_encodings_and_invalid_values(self):
        np.testing.assert_allclose(depth_image_to_meters(np.array([[0, 200, 500]], dtype=np.uint16),
                                                         "16UC1"), [[0., .2, .5]])
        values = np.array([[.2, np.nan, np.inf, -1.]], dtype=np.float32)
        np.testing.assert_allclose(depth_image_to_meters(values, "32FC1"), [[.2, 0., 0., 0.]])
        self.assertTrue(np.isnan(values[0, 1]))
        with self.assertRaises(ValueError):
            depth_image_to_meters(values, "rgb8")

    def test_pose_interpolation_wraps_yaw_and_rejects_stale_history(self):
        samples = [(1., (0., 0., np.deg2rad(179))), (2., (2., 0., np.deg2rad(-179)))]
        pose = interpolate_planar_pose(samples, 1.5)
        self.assertAlmostEqual(pose[0], 1.)
        self.assertAlmostEqual(abs(pose[2]), np.pi)
        self.assertIsNone(interpolate_planar_pose(samples, 3.))

    def test_d405_frustum_limits_and_robot_rotation(self):
        cfg = simulation_config()
        camera = D405Camera(cfg.d405)
        for depth, expected in ((.05, False), (.3, True), (.6, False)):
            optical = np.array([0., 0., depth])
            base = camera.transform[:3, :3] @ optical + camera.transform[:3, 3]
            self.assertEqual(camera.visible(base, (0., 0., 0.)), expected)
        floor = camera.floor_polygon((.5, .5, 0.), 3.2, 2.4)
        self.assertGreater(len(floor), 2)
        base_point = np.array([.32, 0., .002])
        rotated = np.array([1., 1.32, .002])
        self.assertEqual(camera.visible(base_point, (0., 0., 0.)),
                         camera.visible(rotated, (1., 1., np.pi/2)))


class GeometryAndConfigTests(unittest.TestCase):
    def test_coverage_includes_last_boundary_lane(self):
        path = generate_boustrophedon(0., 2., 0., 1., .17, .15)
        self.assertAlmostEqual(path[-1, 1], 1.)

    def test_router_avoids_obstacles_and_detects_blocked_goal(self):
        nav = ArenaNavigator(2., 2., .05)
        nav.observe([observation(1., 1., track_id=1, coordinate_frame="odom")])
        path = nav.route([.3, 1.], [1.7, 1.], .2)
        self.assertIsNotNone(path)
        self.assertGreater(len(path), 2)
        for start, end in zip(path, path[1:]):
            self.assertTrue(nav.segment_clear(start, end, .2))
        self.assertIsNone(nav.route([.3, 1.], [1., 1.], .2))
        self.assertIsNone(nav.route([.3, 1.], [.05, 1.], .2))

    def test_payload_clearance_does_not_change_mop_radius(self):
        cfg = RobotConfig()
        before = cfg.geometry.radius_m
        small = collector_navigation_radius(cfg.collector, cfg.geometry.chassis_radius_m, False)
        carrying = collector_navigation_radius(cfg.collector, cfg.geometry.chassis_radius_m, True)
        self.assertGreater(carrying, small)
        self.assertEqual(cfg.geometry.radius_m, before)

    def test_litter_eligibility_rejects_unknown_heavy_and_oversized_objects(self):
        cfg = CollectorConfig(enabled=True)
        target = observation()
        self.assertIsNone(pickup_rejection(target, cfg))
        for invalid in (replace(target, object_class="phone"), replace(target, estimated_mass_kg=None),
                        replace(target, estimated_mass_kg=1.), replace(target, size_m=(.6, .1, .1)),
                        replace(target, orientation_rad=float("nan"))):
            self.assertIsNotNone(pickup_rejection(invalid, cfg))

    def test_json_overrides_are_strict_and_validated(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.json"
            path.write_text(json.dumps({"collector": {"opening_width_m": .24},
                                        "simulation": {"seed": 12}}), encoding="utf-8")
            cfg = simulation_config(path)
            self.assertEqual(cfg.collector.opening_width_m, .24)
            self.assertEqual(cfg.simulation.seed, 12)
            for invalid in ({"collector": {"typo": 1}}, {"simulation": {"time_step_s": 0}},
                            {"collector": {"enabled": "false"}}, {"d405": {"max_depth_m": .01}}):
                path.write_text(json.dumps(invalid), encoding="utf-8")
                with self.assertRaises(ValueError):
                    simulation_config(path)


if __name__ == "__main__":
    unittest.main()
