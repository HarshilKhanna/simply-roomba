"""Interactive mop + litter retrieval lab, backed by the shared planner.

    python visual_sim.py
    python visual_sim.py --headless --auto-unload --fail-first-pickup

U models manual unloading at home. Headless auto-unloading is opt-in.
"""
import argparse
from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

from cleaning_robot.collector import CollectorCommand, collector_position
from cleaning_robot.motion_planning import MissionState
from cleaning_robot.simulation import CleaningSimulation, simulation_config
from cleaning_robot.stress import (SCENARIOS, ScenarioSpec, StressSession,
                                   make_scenario)


@dataclass
class Button:
    action: str
    rect: tuple
    enabled: bool = True

    def contains(self, x, y):
        left, top, width, height = self.rect
        return left <= x < left+width and top <= y < top+height


class SimulationRenderer:
    SCALE = 230
    HEADER = 80
    FOOTER = 132
    PANEL = 370
    INK = (44, 39, 34)
    MUTED = (151, 160, 163)
    GREEN = (121, 204, 92)
    BLUE = (233, 177, 73)
    PURPLE = (222, 138, 183)
    AMBER = (72, 191, 248)

    def __init__(self, sim, app=None):
        self.sim = sim
        self.app = app
        self.buttons = []
        self.width = round(sim.cfg.simulation.arena_width_m * self.SCALE)
        self.height = round(sim.cfg.simulation.arena_height_m * self.SCALE)
        self.trail_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        self._trail_index = 0

    def button(self, frame, label, action, rect, *, enabled=True, selected=False):
        x, y, width, height = rect
        self.buttons.append(Button(action, rect, enabled))
        hovered = self.app is not None and self.app.hovered_action == action
        focused = self.app is not None and self.app.focused_action == action
        fill = (64, 67, 62) if enabled else (35, 37, 34)
        border = self.AMBER if (hovered or focused or selected) and enabled else (99, 111, 104)
        color = (240, 244, 242) if enabled else (130, 140, 134)
        cv2.rectangle(frame, (x, y), (x+width, y+height), fill, -1)
        cv2.rectangle(frame, (x, y), (x+width, y+height), border, 1)
        font_scale = .40
        while cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0][0] > width-14:
            font_scale -= .02
        self.text(frame, label, x+8, y+height//2+5, color, font_scale)

    def hit_test(self, x, y):
        return next((b.action for b in reversed(self.buttons) if b.enabled and b.contains(x, y)), None)

    def tabs(self, frame, mode="live"):
        x = self.width - 249
        self.button(frame, "Live arena", "live_view", (x, 22, 104, 34), selected=mode == "live")
        self.button(frame, "Stress results", "stress_view", (x+114, 22, 123, 34), selected=mode == "stress")

    def footer(self, frame):
        app, sim = self.app, self.sim
        y, x = self.HEADER+self.height+12, 18
        speed = app.speed if app else 1
        automatic = app.auto_unload if app else False
        controls = [
            ("Resume [Space]" if sim.paused else "Pause [Space]", "pause", 118, True),
            ("Unload + resume [U]", "unload", 157, sim.mission.state == MissionState.WAITING_FOR_UNLOAD),
            ("Step [.]", "step", 86, True),
            (f"Speed {speed}x [+]", "speed", 107, True),
            ("Auto unload: " + ("ON [A]" if automatic else "OFF [A]"), "auto_unload", 165, True),
            ("Random scene [N]", "random", 140, True),
            ("Reset [R]", "reset", 94, True),
            ("Stress test [T]", "stress", 126, True)]
        for label, action, width, enabled in controls:
            self.button(frame, label, action, (x, y, width, 34), enabled=enabled)
            x += width+9
        y += 43
        x = 18
        for label, action, width, enabled in [
                ("Add litter [D]", "litter", 140, True),
                ("Add stain [S]", "stain", 140, True),
                ("Fail pickup [F]", "fail", 140, True),
                ("Blackout [B]", "blackout", 140, True),
                ("Drop load [X]", "drop", 140, sim.collector.payload is not None),
                ("Save frame [P]", "save_frame", 140, True),
                ("Camera view [C]", "camera", 140, True)]:
            self.button(frame, label, action, (x, y, width, 32), enabled=enabled)
            x += width+9
        self.text(frame, "Mock perception and pickup | Tab + Enter operate buttons | Q / Esc closes the window",
                  18, y+56, self.MUTED, .37)

    def pixel(self, point):
        return (round(float(point[0]) * self.SCALE),
                self.HEADER + self.height - round(float(point[1]) * self.SCALE))

    @staticmethod
    def text(frame, value, x, y, color=(225, 231, 234), scale=0.45, weight=1):
        cv2.putText(frame, str(value), (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, weight, cv2.LINE_AA)

    def _poly(self, frame, points, color, *, fill=False, thickness=1):
        points = np.array([self.pixel(p) for p in points], dtype=np.int32)
        if len(points) < 2:
            return
        if fill:
            cv2.fillPoly(frame, [points], color, cv2.LINE_AA)
        else:
            cv2.polylines(frame, [points], True, color, thickness, cv2.LINE_AA)

    def _object(self, frame, item, label=True):
        centre = self.pixel(item.position)
        color = {"plastic_bottle": (162, 166, 58), "wrapper": (103, 171, 230),
                 "paper_cup": (197, 140, 195), "paper_ball": (196, 190, 176)}.get(item.object_class, (119, 126, 137))
        length, width = item.size_m[:2]
        c, s = np.cos(item.yaw_rad), np.sin(item.yaw_rad)
        rotation = np.array([[c, s], [-s, c]])
        if item.object_class == "plastic_bottle":
            points = np.array([[-length*.5, -width*.5], [length*.3, -width*.5],
                               [length*.5, -width*.2], [length*.5, width*.2],
                               [length*.3, width*.5], [-length*.5, width*.5]])
            self._poly(frame, points @ rotation + item.position, color, fill=True)
            self._poly(frame, points @ rotation + item.position, self.INK)
        elif item.object_class == "wrapper":
            points = np.array([[-length*.5, -width*.5], [-length*.15, -width*.3],
                               [length*.5, -width*.5], [length*.4, width*.5],
                               [0, width*.3], [-length*.5, width*.5]])
            self._poly(frame, points @ rotation + item.position, color, fill=True)
            self._poly(frame, points @ rotation + item.position, self.INK)
        elif item.object_class == "obstacle":
            points = np.array([[-length/2, -width/2], [length/2, -width/2],
                               [length/2, width/2], [-length/2, width/2]])
            self._poly(frame, points @ rotation + item.position, color, fill=True)
            self._poly(frame, points @ rotation + item.position, self.INK, thickness=2)
        else:
            radius = max(5, round(width / 2 * self.SCALE))
            cv2.circle(frame, centre, radius, color, -1, cv2.LINE_AA)
            cv2.circle(frame, centre, radius, self.INK, 1, cv2.LINE_AA)
            if item.object_class == "paper_cup":
                cv2.circle(frame, centre, max(2, radius - 3), (235, 229, 236), 1, cv2.LINE_AA)
            else:
                cv2.line(frame, (centre[0]-4, centre[1]-3), (centre[0]+3, centre[1]+4), self.INK, 1, cv2.LINE_AA)
        if item.object_id in self.sim.visible_ids:
            cv2.circle(frame, centre, round(item.radius_m * self.SCALE) + 5, self.BLUE, 1, cv2.LINE_AA)
        if label:
            self.text(frame, item.object_class.replace("_", " "), centre[0] - 30,
                      centre[1] - round(item.radius_m * self.SCALE) - 9, self.INK, 0.34)

    def render(self):
        sim = self.sim
        self.buttons = []
        total_height = self.HEADER + self.height + self.FOOTER
        frame = np.full((total_height, self.width + self.PANEL, 3), (31, 28, 24), dtype=np.uint8)
        floor = frame[self.HEADER:self.HEADER + self.height, :self.width]
        floor[:] = (237, 240, 239)
        for x in range(0, self.width, round(self.SCALE / 4)):
            cv2.line(floor, (x, 0), (x, self.height), (224, 229, 227), 1)
        for y in range(0, self.height, round(self.SCALE / 4)):
            cv2.line(floor, (0, y), (self.width, y), (224, 229, 227), 1)
        for point in sim.trail[self._trail_index:]:
            px, py = self.pixel(point)
            cv2.circle(self.trail_mask, (px, py - self.HEADER),
                       round(sim.cfg.geometry.radius_m * self.SCALE), 255, -1)
        self._trail_index = len(sim.trail)
        floor[self.trail_mask > 0] = (218, 235, 220)
        coverage = sim.mission.coverage_path
        if len(coverage) > 1:
            cv2.polylines(frame, [np.array([self.pixel(p) for p in coverage], dtype=np.int32)],
                          False, (187, 202, 190), 1, cv2.LINE_AA)
        view = sim.camera.floor_polygon(sim.pose, sim.cfg.simulation.arena_width_m,
                                        sim.cfg.simulation.arena_height_m)
        overlay = frame.copy()
        if self.app is None or self.app.show_camera:
            self._poly(overlay, view, self.BLUE, fill=True)
        frame = cv2.addWeighted(overlay, 0.18, frame, 0.82, 0)
        if self.app is None or self.app.show_camera:
            self._poly(frame, view, (203, 160, 70))
        route = sim.mission.active_route
        if route is not None and len(route) > 1:
            cv2.polylines(frame, [np.array([self.pixel(p) for p in route], dtype=np.int32)],
                          False, (146, 118, 88), 2, cv2.LINE_AA)
        home = self.pixel(sim.mission.home_pose[:2])
        cv2.rectangle(frame, (home[0]-18, home[1]-18), (home[0]+18, home[1]+18), self.BLUE, 2, cv2.LINE_AA)
        for stain in sim.stains:
            if not stain.cleaned:
                cv2.circle(frame, self.pixel(stain.position), round(stain.radius_m*self.SCALE),
                           (104, 133, 173), -1, cv2.LINE_AA)
                self.text(frame, "stain", self.pixel(stain.position)[0]-15,
                          self.pixel(stain.position)[1]-round(stain.radius_m*self.SCALE)-6, self.INK, 0.32)
        for item in sim.litter:
            if item.status == "floor":
                self._object(frame, item)
        self._robot(frame)
        self.text(frame, "HOME", home[0]-20,
                  home[1]+round(sim.cfg.geometry.chassis_radius_m*self.SCALE)+22, self.INK, 0.4)
        cv2.rectangle(frame, (1, self.HEADER+1), (self.width-2, self.HEADER+self.height-2),
                       (83, 98, 91), 3)
        self.text(frame, "FLOOR / RETRIEVAL LAB", 22, 29, (236, 241, 240), 0.65, 1)
        self.text(frame, "Shared mission planner  |  virtual concave collector", 23, 51, self.MUTED, 0.4)
        self.tabs(frame)
        self._panel(frame)
        self.footer(frame)
        if sim.paused:
            self.text(frame, "PAUSED", self.width//2-70, self.HEADER+45, self.INK, 0.9, 2)
        return frame

    def _robot(self, frame):
        sim, cfg = self.sim, self.sim.cfg.collector
        centre = self.pixel(sim.pose[:2])
        radius = round(sim.cfg.geometry.chassis_radius_m * self.SCALE)
        cv2.circle(frame, centre, radius+3, self.GREEN if sim.mission.mop_enabled else (154, 167, 161), 3, cv2.LINE_AA)
        cv2.circle(frame, centre, radius-2, (57, 61, 58), -1, cv2.LINE_AA)
        self._poly(frame, sim.collector_polygon(), (208, 185, 199), fill=True)
        self._poly(frame, sim.collector_polygon(), (132, 91, 123), thickness=2)
        extended = sim.mission.collector_command != CollectorCommand.STOW
        closed = sim.mission.collector_command in (CollectorCommand.CLOSE, CollectorCommand.LIFT, CollectorCommand.HOLD)
        length = sim.collector_offset() + cfg.palm_depth_m / 2 if extended else cfg.stowed_front_m
        if sim.mission.collector_command == CollectorCommand.LOWER:
            progress = min(1., sim.collector.elapsed / sim.cfg.simulation.lower_duration_s)
            length = cfg.stowed_front_m + (length-cfg.stowed_front_m)*progress
        jaw_tip = .25 if closed else .65
        if sim.mission.collector_command == CollectorCommand.CLOSE:
            progress = min(1., sim.collector.elapsed / sim.cfg.simulation.close_duration_s)
            jaw_tip = .65 - .4*progress
        half = cfg.opening_width_m / 2 if extended else cfg.opening_width_m / 4
        c, s = np.cos(sim.pose[2]), np.sin(sim.pose[2])
        rotation = np.array([[c, s], [-s, c]])
        for side in (-1, 1):
            jaw = np.array([[0.13, side*half*.65], [length*.7, side*half],
                            [length*.95, side*half*.8], [length, side*half*jaw_tip]])
            points = np.array([self.pixel(p) for p in jaw @ rotation + sim.pose[:2]], dtype=np.int32)
            cv2.polylines(frame, [points], False, (146, 86, 132), 4, cv2.LINE_AA)
        lens = np.array(sim.pose[:2]) + sim.cfg.d405.mount.forward_offset_m * np.array([c, s])
        cv2.circle(frame, self.pixel(lens), 5, self.BLUE, -1, cv2.LINE_AA)
        if sim.collector.payload is not None:
            self._object(frame, sim.collector.payload, label=False)

    def _panel(self, frame):
        sim = self.sim
        x, y = self.width + 22, 30
        self.text(frame, "Mission", x, y, (235, 240, 237), 0.52)
        waiting = sim.mission.state == MissionState.WAITING_FOR_UNLOAD
        faulted = sim.mission.state == MissionState.FAULT
        self.text(frame, sim.mission.state.name.replace("_", " "), x, y+30,
                  (109, 143, 245) if faulted else self.AMBER if waiting else self.GREEN, 0.53, 1)
        y += 77
        if waiting:
            self.text(frame, "Payload retained at home", x, y, self.AMBER, 0.46)
            self.text(frame, "Click Unload + resume, or press U", x, y+23, (239, 241, 242), 0.41)
        elif faulted:
            self.text(frame, "Robot stopped", x, y, (109, 143, 245), 0.47)
            self.text(frame, "Inspect the event log, then Reset", x, y+23, (239, 241, 242), 0.40)
        else:
            seed = self.app.spec.seed if self.app else sim.cfg.simulation.seed
            self.text(frame, f"Scene: {getattr(sim, 'scenario_name', 'baseline').replace('_', ' ')}", x, y, self.MUTED, .41)
            self.text(frame, f"Seed {seed}  |  one item per return trip", x, y+23, self.MUTED, 0.38)
        y += 65
        summary = sim.summary()
        rows = [("Elapsed", f"{sim.time_s:.1f} s"),
                ("Velocity", f"{sim.last_velocity[0]:.2f} m/s"),
                ("Delivered", str(sim.deliveries)),
                ("Pickup attempts", str(summary["pickup_attempts"])),
                ("Carrying", (summary["carrying"] or "empty").replace("_", " ")),
                ("Mop", "active" if sim.mission.mop_enabled else "paused"),
                ("Claw / lift", sim.mission.collector_command.name.lower()),
                ("Coverage segment", f"{sim.mission.coverage_index}/{summary['coverage_segments']}"),
                ("Safety stops", str(sim.safety_stops)),
                ("Travelled", f"{sim.path_length_m:.1f} m")]
        for label, value in rows:
            self.text(frame, label, x, y, self.MUTED, 0.4)
            self.text(frame, value, x+155, y, (229, 235, 234), 0.42)
            y += 25
        y += 16
        self.text(frame, "D405 workspace", x, y, self.BLUE, 0.46)
        self.text(frame, f"Depth {sim.cfg.d405.min_depth_m:.2f}-{sim.cfg.d405.max_depth_m:.2f} m  |  nominal FOV",
                  x, y+23, self.MUTED, 0.39)
        blackout = sim.time_s < sim.sensor_blackout_until
        self.text(frame, "Camera blackout" if blackout else "Mock camera online", x, y+45,
                  self.AMBER if blackout else self.MUTED, .39)
        y += 85
        self.text(frame, "Recent events", x, y, (225, 231, 234), 0.45)
        y += 24
        for event in sim.mission.events[-2:]:
            self.text(frame, f"{event['time_s']:.1f}s", x, y, self.BLUE, .34)
            words, line = event["message"].split(), ""
            lines = []
            for word in words:
                candidate = (line+" "+word).strip()
                if cv2.getTextSize(candidate, cv2.FONT_HERSHEY_SIMPLEX, .35, 1)[0][0] > self.PANEL-100:
                    lines.append(line)
                    line = word
                else:
                    line = candidate
            lines.append(line)
            for line in lines[:2]:
                self.text(frame, line, x+53, y, self.MUTED, .35)
                y += 18
            y += 7
        if self.app and self.app.notice and time.perf_counter() < self.app.notice_until:
            self.text(frame, self.app.notice[:48], x, self.HEADER+self.height-14, self.AMBER, .35)


class SimulationApp:
    """UI actions are independent of HighGUI, so keyboard and mouse share behavior."""
    SPEEDS = (1, 2, 4, 8)

    def __init__(self, cfg=None, *, spec=None, auto_unload=False, speed=2,
                 stress_runs=20, fail_first=False, success_rate=1., snapshot=None):
        self.cfg = deepcopy(cfg) if cfg is not None else simulation_config()
        self.spec = spec or ScenarioSpec("baseline", self.cfg.simulation.seed)
        self.snapshot = snapshot
        self.auto_unload = auto_unload
        self.speed = speed
        self.stress_runs = stress_runs
        self.fail_first = fail_first
        self.success_rate = success_rate
        self.hovered_action = None
        self.focused_action = None
        self.show_camera = True
        self.mode = "live"
        self.stress = None
        self.result_page = 0
        self.failures_only = False
        self.notice = ""
        self.notice_until = 0.
        self._stress_saved = False
        self.reset()

    def reset(self):
        self.sim, self.faults = make_scenario(self.spec, self.cfg, self.snapshot)
        self.sim.collector.success_probability = self.success_rate if self.spec.name != "pickup_exhaustion" else 0.
        if self.fail_first:
            self.sim.force_next_failure()
        self.renderer = SimulationRenderer(self.sim, self)

    def notify(self, message):
        self.notice, self.notice_until = message, time.perf_counter()+5.

    def _live_step(self):
        self.faults.apply(self.sim)
        self.sim.step()
        if (self.auto_unload and self.sim.mission.state == MissionState.WAITING_FOR_UNLOAD
                and self.sim.time_s-self.sim.mission._state_since >= 1.):
            self.sim.unload()

    def advance(self):
        if self.stress is not None and self.stress.running:
            self.stress.tick(max_steps=250, max_wall_s=.020 if self.mode == "stress" else .008)
            if self.stress.complete and not self._stress_saved:
                self.stress.save("artifacts/stress/latest")
                self._stress_saved = True
        if self.mode == "live" and not self.sim.paused:
            for _ in range(self.speed):
                if self.sim.mission.state in (MissionState.FAULT, MissionState.DONE):
                    break
                self._live_step()

    def handle(self, action):
        try:
            if action == "pause":
                self.sim.paused = not self.sim.paused
            elif action == "unload":
                if not self.sim.unload():
                    self.notify("Unloading is available when the robot reaches home")
            elif action == "step":
                self.sim.paused = False
                self._live_step()
                self.sim.paused = True
            elif action == "speed":
                self.speed = self.SPEEDS[(self.SPEEDS.index(self.speed)+1) % len(self.SPEEDS)]
            elif action == "auto_unload":
                self.auto_unload = not self.auto_unload
            elif action == "random":
                self.spec = ScenarioSpec("random", self.spec.seed+1)
                self.snapshot = None
                self.reset()
                self.mode = "live"
            elif action == "reset":
                self.reset()
            elif action == "litter":
                if not self.sim.spawn_litter():
                    self.notify("No clear space for another object")
            elif action == "stain":
                self.sim.spawn_stain()
            elif action == "fail":
                self.sim.force_next_failure()
                self.notify("The next claw pickup will fail")
            elif action == "blackout":
                self.sim.blackout_sensor()
                self.notify("Camera signal disabled for 2 simulated seconds")
            elif action == "drop":
                if not self.sim.drop_payload():
                    self.notify("There is no payload to drop")
            elif action == "camera":
                self.show_camera = not self.show_camera
            elif action == "save_frame":
                directory = Path("artifacts")
                directory.mkdir(exist_ok=True)
                path = directory / f"simulator_{self.spec.seed}_{round(self.sim.time_s*100):06d}.png"
                cv2.imwrite(str(path), self.render())
                self.notify(f"Saved {path.name}")
            elif action == "live_view":
                self.mode = "live"
            elif action == "stress_view":
                self.mode = "stress"
            elif action in ("stress", "start_stress"):
                self.mode = "stress"
                if self.stress is None or not self.stress.running:
                    self.stress = StressSession(self.stress_runs, self.spec.seed, config=self.cfg)
                    self._stress_saved = False
                    self.result_page = 0
            elif action == "stop_stress":
                if self.stress:
                    self.stress.cancel()
            elif action == "save_report":
                if self.stress:
                    self.stress.save("artifacts/stress/latest")
                    cv2.imwrite("artifacts/stress/latest/stress_report.png", self.render_stress())
                    self.notify("Saved report in artifacts/stress/latest")
            elif action == "filter_failures":
                self.failures_only = not self.failures_only
                self.result_page = 0
            elif action == "next_page":
                self.result_page += 1
            elif action == "previous_page":
                self.result_page = max(0, self.result_page-1)
            elif action.startswith("replay:"):
                result = self.stress.results[int(action.split(":")[1])]
                self.snapshot = result["replay"]
                self.spec = ScenarioSpec(**self.snapshot["scenario"])
                self.reset()
                self.mode = "live"
                self.notify("Replaying the exact scene and scheduled faults")
        except (ValueError, OSError) as error:
            self.notify(str(error))

    def key(self, key):
        if key in (27, ord("q"), ord("Q")):
            return False
        if key == 9:
            actions = [b.action for b in self.renderer.buttons if b.enabled]
            if actions:
                index = actions.index(self.focused_action) if self.focused_action in actions else -1
                self.focused_action = actions[(index+1) % len(actions)]
            return True
        if key == 13 and self.focused_action:
            self.handle(self.focused_action)
            return True
        shortcuts = {" ": "pause", "u": "unload", ".": "step", "+": "speed", "=": "speed",
                     "a": "auto_unload", "n": "random", "r": "reset", "d": "litter", "s": "stain",
                     "f": "fail", "b": "blackout", "x": "drop", "c": "camera", "p": "save_frame", "t": "stress"}
        if 0 <= key <= 255 and chr(key).lower() in shortcuts:
            self.handle(shortcuts[chr(key).lower()])
        return True

    def mouse(self, event, x, y, flags, param=None):
        self.hovered_action = self.renderer.hit_test(x, y)
        if event == cv2.EVENT_LBUTTONUP and self.hovered_action:
            self.focused_action = self.hovered_action
            self.handle(self.hovered_action)

    def render(self):
        return self.render_stress() if self.mode == "stress" else self.renderer.render()

    def render_stress(self):
        renderer = self.renderer
        renderer.buttons = []
        width = renderer.width+renderer.PANEL
        height = renderer.HEADER+renderer.height+renderer.FOOTER
        frame = np.full((height, width, 3), (31, 28, 24), dtype=np.uint8)
        frame[renderer.HEADER:renderer.HEADER+renderer.height] = (237, 240, 239)
        text = renderer.text
        text(frame, "STRESS TESTS", 22, 32, (236, 241, 240), .65)
        text(frame, "Seeded scenes, recorded expectations and exact replays", 23, 57, renderer.MUTED, .40)
        renderer.tabs(frame, "stress")
        results = self.stress.results if self.stress else []
        passed = sum(r["passed"] for r in results)
        requested = len(self.stress.specs) if self.stress else self.stress_runs
        state = "RUNNING" if self.stress and self.stress.running else "CANCELLED" if self.stress and self.stress.cancelled else "COMPLETE" if self.stress and self.stress.complete else "READY"
        text(frame, f"{state}   {len(results)}/{requested} cases", 24, 114, renderer.INK, .58)
        text(frame, f"{passed} passed    {len(results)-passed} failed", width-310, 114, renderer.INK, .50)
        cv2.rectangle(frame, (24, 132), (width-24, 138), (209, 217, 212), -1)
        if results:
            cv2.rectangle(frame, (24, 132), (24+round((width-48)*len(results)/requested), 138), (124, 175, 100), -1)
        if self.stress and self.stress.current:
            text(frame, f"Now: {self.stress.specs[len(results)].name.replace('_', ' ')} / seed {self.stress.specs[len(results)].seed}",
                 24, 163, (102, 111, 104), .40)
        else:
            text(frame, "A planned safe stop passes fault-injection cases. Failed expectations stay visible.",
                 24, 163, (102, 111, 104), .40)
        columns = [(24, "SCENARIO"), (260, "SEED"), (345, "DELIVERED"), (465, "FINAL STATE"),
                   (695, "CHECK"), (860, "HOME ERROR")]
        for x, label in columns:
            text(frame, label, x, 195, (91, 105, 96), .38)
        indexed = [(i, result) for i, result in enumerate(results) if not self.failures_only or not result["passed"]]
        rows_per_page = max(4, (renderer.HEADER+renderer.height-230)//40)
        pages = max(1, (len(indexed)+rows_per_page-1)//rows_per_page)
        self.result_page = min(self.result_page, pages-1)
        start = self.result_page*rows_per_page
        for row, (index, result) in enumerate(indexed[start:start+rows_per_page]):
            y = 211+row*40
            color = (72, 124, 65) if result["passed"] else (72, 77, 190)
            hovered = self.hovered_action == f"replay:{index}"
            cv2.rectangle(frame, (18, y-2), (width-18, y+35),
                          (219, 229, 220) if hovered else (229, 235, 231) if row%2 == 0 else (237, 240, 239), -1)
            renderer.buttons.append(Button(f"replay:{index}", (18, y-2, width-36, 37)))
            outcome = "FAIL / replay"
            if result["passed"]:
                outcome = ("PASS / safe stop" if result["summary"]["state"] == "FAULT" else
                           "PASS / bounded" if result["scenario"] == "pickup_exhaustion" else
                           "PASS / skipped" if result["missed_litter_ids"] else "PASS")
            values = [result["scenario"].replace("_", " "), str(result["seed"]),
                      f"{result['summary']['delivered']}/{result['total_litter']}", result["summary"]["state"].replace("_", " "),
                      outcome,
                      f"{result['max_home_error_m']*1000:.1f} mm" if result["max_home_error_m"] is not None else "--"]
            for (x, _), value in zip(columns, values):
                text(frame, value, x, y+21, color if x == 695 else renderer.INK, .41)
        if not indexed:
            text(frame, "No failed cases." if self.failures_only else "Run a batch to inspect varied scenes and fault recovery.",
                 26, 260, renderer.INK, .52)
        y = renderer.HEADER+renderer.height+12
        running = self.stress is not None and self.stress.running
        x = 18
        for label, action, button_width, enabled in [
                (f"Run {self.stress_runs} cases", "start_stress", 142, not running),
                ("Stop batch", "stop_stress", 116, running),
                ("Save report", "save_report", 122, bool(results)),
                ("All cases" if self.failures_only else "Failures only", "filter_failures", 126, True),
                ("Previous", "previous_page", 104, self.result_page > 0),
                ("Next", "next_page", 90, self.result_page+1 < pages),
                ("Back to arena", "live_view", 138, True)]:
            renderer.button(frame, label, action, (x, y, button_width, 35), enabled=enabled)
            x += button_width+10
        text(frame, f"Page {self.result_page+1}/{pages}. Click a result row to replay its scene and fault schedule.",
             20, y+65, renderer.MUTED, .42)
        text(frame, "Reports and failed-case replays: artifacts/stress/latest", 20, y+92, renderer.MUTED, .38)
        return frame


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    display = parser.add_mutually_exclusive_group()
    display.add_argument("--gui", action="store_true", help="open the interactive simulator (default)")
    display.add_argument("--headless", action="store_true", help="batch mode: JSON only, no window")
    parser.add_argument("--auto-unload", action="store_true", help="explicitly simulate operator acknowledgements")
    parser.add_argument("--steps", type=int, default=20000, help="maximum headless control steps")
    parser.add_argument("--config", type=Path, help="JSON overrides for RobotConfig")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--fail-first-pickup", action="store_true")
    parser.add_argument("--pickup-success-rate", type=float, default=1.0)
    parser.add_argument("--report", type=Path, help="write summary and mission events as JSON")
    parser.add_argument("--output", type=Path, help="save final simulator frame as PNG")
    parser.add_argument("--scenario", choices=SCENARIOS, default="baseline")
    parser.add_argument("--random-scene", action="store_true", help="start with a seeded random layout")
    parser.add_argument("--replay", type=Path, help="replay a saved stress case")
    parser.add_argument("--speed", type=int, choices=SimulationApp.SPEEDS, default=2, help="GUI playback speed")
    parser.add_argument("--stress-runs", type=int, default=20)
    parser.add_argument("--stress", action="store_true", help="start the visual stress-test view")
    parser.add_argument("--json", action="store_true", help="also print JSON after closing the GUI")
    parser.add_argument("--frames", type=int, help="close the GUI after this many frames (automated smoke tests)")
    args = parser.parse_args(argv)
    if args.steps <= 0 or args.stress_runs <= 0 or not 0 <= args.pickup_success_rate <= 1:
        parser.error("steps/runs must be positive and pickup success rate must be in [0, 1]")
    if args.frames is not None and args.frames < 1:
        parser.error("frames must be positive")
    if args.headless and args.stress:
        parser.error("use python stress_test.py for headless batches; --stress opens a visual report")
    try:
        cfg = simulation_config(args.config)
        if args.seed is not None:
            cfg.simulation.seed = args.seed
        snapshot = json.loads(args.replay.read_text(encoding="utf-8")) if args.replay else None
        if snapshot is not None and "replay" in snapshot:
            snapshot = snapshot["replay"]
        spec = (ScenarioSpec(**snapshot["scenario"]) if snapshot is not None else
                ScenarioSpec("random" if args.random_scene else args.scenario, cfg.simulation.seed))
    except (ValueError, OSError, TypeError) as error:
        parser.error(str(error))
    renderer = None
    if args.headless:
        sim, faults = make_scenario(spec, cfg, snapshot)
        if spec.name != "pickup_exhaustion":
            sim.collector.success_probability = args.pickup_success_rate
        if args.fail_first_pickup:
            sim.force_next_failure()
        for _ in range(args.steps):
            faults.apply(sim)
            sim.step()
            if sim.mission.state in (MissionState.DONE, MissionState.FAULT):
                break
            if sim.mission.state == MissionState.WAITING_FOR_UNLOAD:
                if not args.auto_unload:
                    break
                if sim.time_s-sim.mission._state_since >= 1.:
                    sim.unload()
    else:
        app = SimulationApp(cfg, spec=spec, snapshot=snapshot, auto_unload=args.auto_unload,
                            speed=args.speed, stress_runs=args.stress_runs,
                            fail_first=args.fail_first_pickup, success_rate=args.pickup_success_rate)
        window = "Cleaning Robot - Litter Retrieval Simulation"
        try:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        except cv2.error:
            print("Could not open the simulator window. Use a desktop session with opencv-python installed.\n"
                  "A headless OpenCV package cannot show windows. Run: python -m pip install --force-reinstall opencv-python",
                  file=sys.stderr)
            return 2
        frame_height = app.renderer.HEADER+app.renderer.height+app.renderer.FOOTER
        frame_width = app.renderer.width+app.renderer.PANEL
        scale = 1.
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes
            work = wintypes.RECT()
            if ctypes.windll.user32.SystemParametersInfoW(48, 0, ctypes.byref(work), 0):
                scale = min(1., (work.right-work.left-60)/frame_width,
                            (work.bottom-work.top-65)/frame_height)
                cv2.moveWindow(window, work.left+20, work.top+15)
        cv2.resizeWindow(window, round(frame_width*scale), round(frame_height*scale))
        cv2.setMouseCallback(window, app.mouse)
        if args.stress:
            app.handle("stress")
        print("Visual simulator opened. Click the on-screen controls.\n"
              "Space: pause | U: unload at home | +: speed | N: random scene | T: stress tests | Q: quit", flush=True)
        frames = 0
        try:
            while True:
                started = time.perf_counter()
                app.advance()
                cv2.imshow(window, app.render())
                delay = max(1, round((cfg.simulation.time_step_s-(time.perf_counter()-started))*1000))
                key = cv2.waitKeyEx(delay)
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if not app.key(key):
                    break
                frames += 1
                if args.frames is not None and frames >= args.frames:
                    break
        finally:
            cv2.destroyAllWindows()
        sim, renderer = app.sim, app.renderer
    summary = sim.summary()
    if args.headless or args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(f"Simulator closed: {summary['state']}; {summary['delivered']} items delivered.")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({**summary, "events": sim.mission.events}, indent=2), encoding="utf-8")
    if args.output:
        renderer = renderer or SimulationRenderer(sim)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        frame = renderer.render() if args.headless else app.render()
        if not cv2.imwrite(str(args.output), frame):
            raise OSError(f"could not write {args.output}")
    if args.headless and sim.mission.state == MissionState.FAULT:
        return 2
    if args.headless and sim.mission.state not in (MissionState.DONE, MissionState.WAITING_FOR_UNLOAD):
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
