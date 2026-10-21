"""Interaction and scenario behavior, independent of a visible desktop."""
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from cleaning_robot.motion_planning import MissionState
from cleaning_robot.simulation import CleaningSimulation, Litter, simulation_config
from cleaning_robot.stress import (ScenarioSpec, StressSession, make_scenario,
                                   scene_snapshot, evaluate_result)
from visual_sim import SimulationApp


class DesktopInteractionTests(unittest.TestCase):
    def test_mouse_buttons_pause_and_step_the_actual_simulation(self):
        app = SimulationApp()
        app.render()
        button = next(b for b in app.renderer.buttons if b.action == "pause")
        x, y, width, height = button.rect
        app.mouse(cv2.EVENT_LBUTTONUP, x+width//2, y+height//2, 0)
        self.assertTrue(app.sim.paused)
        before = app.sim.time_s
        app.advance()
        self.assertEqual(app.sim.time_s, before)
        app.handle("step")
        self.assertAlmostEqual(app.sim.time_s, before+app.cfg.simulation.time_step_s)
        self.assertTrue(app.sim.paused)

    def test_playback_changes_steps_without_changing_physics_dt(self):
        app = SimulationApp(speed=2)
        dt = app.sim.cfg.simulation.time_step_s
        app.advance()
        self.assertAlmostEqual(app.sim.time_s, dt*2)
        app.handle("speed")
        self.assertEqual(app.speed, 4)
        app.advance()
        self.assertAlmostEqual(app.sim.time_s, dt*6)
        self.assertEqual(app.sim.cfg.simulation.time_step_s, dt)

    def test_unload_control_is_disabled_until_home_wait(self):
        app = SimulationApp()
        app.render()
        button = next(b for b in app.renderer.buttons if b.action == "unload")
        self.assertFalse(button.enabled)
        x, y, width, height = button.rect
        self.assertIsNone(app.renderer.hit_test(x+width//2, y+height//2))
        app.handle("unload")
        self.assertEqual(app.sim.deliveries, 0)
        self.assertIn("home", app.notice)

    def test_random_scene_reset_is_reproducible(self):
        app = SimulationApp()
        app.handle("random")
        self.assertEqual(app.spec, ScenarioSpec("random", 8))
        before = scene_snapshot(app.sim, app.spec)
        app.advance()
        app.handle("reset")
        self.assertEqual(scene_snapshot(app.sim, app.spec), before)

    def test_keyboard_focus_uses_the_same_actions_as_mouse(self):
        app = SimulationApp()
        app.render()
        app.key(9)
        self.assertEqual(app.focused_action, "live_view")
        app.focused_action = "pause"
        app.key(13)
        self.assertTrue(app.sim.paused)
        self.assertFalse(app.key(ord("q")))

    def test_opening_stress_view_does_not_restart_live_scene(self):
        app = SimulationApp()
        app.advance()
        before = app.sim.time_s
        app.handle("stress_view")
        app.advance()
        self.assertEqual(app.sim.time_s, before)
        app.handle("live_view")
        self.assertEqual(app.sim.time_s, before)


class StressBehaviorTests(unittest.TestCase):
    def test_lift_uses_the_configured_compact_carrying_pose(self):
        sim, _ = make_scenario()
        for _ in range(1000):
            sim.step()
            if sim.mission.state == MissionState.RETURNING_HOME:
                break
        self.assertIsNotNone(sim.collector.payload)
        offset = np.linalg.norm(sim.collector.payload.position - sim.pose[:2])
        self.assertAlmostEqual(offset, sim.cfg.collector.carry_offset_m)
        self.assertLess(offset, sim.cfg.collector.pickup_offset_m)

    def test_unobserved_near_wall_litter_does_not_count_as_a_safe_skip(self):
        spec = ScenarioSpec("near_wall", 10)
        sim, _ = make_scenario(spec)
        snapshot = scene_snapshot(sim, spec)
        sim.mission.state = MissionState.DONE
        result = evaluate_result(sim, spec, snapshot, 1, 100)
        self.assertFalse(result["passed"])
        self.assertEqual(result["safe_skipped_litter_ids"], [])

    def test_home_arrival_error_is_part_of_the_stress_expectation(self):
        spec = ScenarioSpec()
        sim = CleaningSimulation(litter=[], stains=[])
        snapshot = scene_snapshot(sim, spec)
        sim.mission.state = MissionState.DONE
        sim.home_arrivals.append({"position_error_m": .1, "heading_error_rad": 0.})
        result = evaluate_result(sim, spec, snapshot, 1, 100)
        self.assertFalse(result["passed"])
        self.assertIn("tolerance", result["reason"])

    def test_long_bottle_allows_tool_contact_while_chassis_stays_clear(self):
        sim = CleaningSimulation(litter=[Litter(1, "plastic_bottle", np.array([.98, .5]),
                                              (.24, .08, .08), .03)], stains=[])
        result = sim.run(5000)
        self.assertEqual(result["state"], "WAITING_FOR_UNLOAD", sim.mission.events[-5:])
        self.assertEqual(result["safety_stops"], 0)
        self.assertEqual(result["carrying"], "plastic_bottle")

    def test_random_geometry_and_orientation_repeat_by_seed(self):
        spec = ScenarioSpec("random", 42)
        first, _ = make_scenario(spec)
        second, _ = make_scenario(spec)
        self.assertEqual(scene_snapshot(first, spec), scene_snapshot(second, spec))
        other, _ = make_scenario(ScenarioSpec("random", 43))
        self.assertFalse(np.allclose(first.litter[0].position, other.litter[0].position))

    def test_near_wall_case_places_a_cup_near_the_wall(self):
        sim, _ = make_scenario(ScenarioSpec("near_wall", 10))
        cup = next(item for item in sim.litter if item.object_class == "paper_cup")
        self.assertAlmostEqual(sim.cfg.simulation.arena_width_m-cup.position[0], .17)

    def test_payload_loss_and_blocked_home_are_expected_safe_stops(self):
        specs = [ScenarioSpec("payload_loss", 15), ScenarioSpec("blocked_home", 14)]
        session = StressSession(specs=specs)
        report = session.run()
        self.assertEqual(report["passed"], 2)
        self.assertEqual(report["safety_stops"], 0)
        self.assertTrue(all(r["summary"]["state"] == "FAULT" for r in report["results"]))
        self.assertTrue(all("injected" in str(r["events"]) for r in report["results"]))

    def test_camera_blackout_clears_current_observations(self):
        sim, _ = make_scenario()
        for _ in range(5):
            sim.step()
        self.assertTrue(sim.confirmed)
        sim.blackout_sensor(1.)
        for _ in range(3):
            sim.step()
        self.assertEqual(sim.confirmed, [])
        self.assertEqual(sim.visible_ids, set())

    def test_step_budget_failure_exports_an_exact_replay(self):
        session = StressSession(runs=1, max_steps=1)
        report = session.run()
        self.assertEqual(report["failed"], 1)
        self.assertIn("step budget", report["results"][0]["reason"])
        with tempfile.TemporaryDirectory() as temp:
            path = session.save(temp)
            data = json.loads(path.read_text(encoding="utf-8"))
            replay_path = next(Path(temp).glob("replay_*.json"))
            snapshot = json.loads(replay_path.read_text(encoding="utf-8"))
            sim, _ = make_scenario(ScenarioSpec(**snapshot["scenario"]), snapshot=snapshot)
            self.assertEqual(scene_snapshot(sim, ScenarioSpec(**snapshot["scenario"])),
                             data["results"][0]["replay"])
            self.assertTrue((Path(temp)/"stress_results.csv").exists())

    def test_cancel_stops_a_cooperative_batch(self):
        session = StressSession(runs=2)
        session.tick(max_steps=1, max_wall_s=1.)
        before = session.current.time_s
        session.cancel()
        session.tick(max_steps=100, max_wall_s=1.)
        self.assertEqual(session.current.time_s, before)
        self.assertTrue(session.report()["cancelled"])

    def test_result_row_replays_its_saved_scene(self):
        app = SimulationApp()
        app.stress = StressSession(runs=1, max_steps=1)
        app.stress.run()
        snapshot = app.stress.results[0]["replay"]
        app.handle("replay:0")
        self.assertEqual(app.mode, "live")
        self.assertEqual(scene_snapshot(app.sim, app.spec), snapshot)


if __name__ == "__main__":
    unittest.main()
