"""Deterministic, hardware-free world exercising the real MissionPlanner.

Perception uses labelled scene objects filtered by the nominal D405 frustum.
It is a mock RGB/semantic sensor, NOT a trained detector or a depth renderer.
Pickup is a geometric/feedback model, NOT contact-physics validation.
"""
from copy import deepcopy
from dataclasses import dataclass, field, fields, is_dataclass
import json
from pathlib import Path

import numpy as np

from .collector import (CollectorCommand, CollectorFeedback, collector_position,
                        collector_navigation_radius, LITTER_CLASSES)
from .config import RobotConfig, D405Config
from .motion_planning import MissionPlanner, MissionState, generate_boustrophedon
from .navigation import ArenaNavigator
from .perception import Detection, DetectionTracker, detection_to_planning_frame


def simulation_config(path=None) -> RobotConfig:
    """Load optional strict JSON overrides, preserving the hardware defaults."""
    cfg = RobotConfig()
    cfg.collector.enabled = True
    if path is not None:
        overrides = json.loads(Path(path).read_text(encoding="utf-8"))
        apply_config_overrides(cfg, overrides)
    validate_simulation_config(cfg)
    return cfg


def apply_config_overrides(target, values, prefix=""):
    """Apply strict JSON-compatible overrides to an existing config dataclass."""
    if not isinstance(values, dict):
        raise ValueError(f"{prefix or 'config'} must be an object")
    allowed = {item.name for item in fields(target)}
    for key, value in values.items():
        if key not in allowed:
            raise ValueError(f"unknown configuration key: {prefix}{key}")
        current = getattr(target, key)
        if is_dataclass(current):
            apply_config_overrides(current, value, prefix + key + ".")
        else:
            if isinstance(current, bool):
                valid_type = isinstance(value, bool)
            elif isinstance(current, int):
                valid_type = isinstance(value, int) and not isinstance(value, bool)
            elif isinstance(current, float):
                valid_type = isinstance(value, (float, int)) and not isinstance(value, bool)
            elif isinstance(current, tuple):
                valid_type = isinstance(value, (list, tuple))
                value = tuple(value) if valid_type else value
            else:
                valid_type = isinstance(value, type(current))
            if not valid_type:
                raise ValueError(f"invalid type for {prefix}{key}")
            setattr(target, key, value)


def validate_simulation_config(cfg):
    def finite_numbers(value):
        for item in fields(value):
            member = getattr(value, item.name)
            if is_dataclass(member):
                finite_numbers(member)
            elif isinstance(member, (int, float)) and not np.isfinite(member):
                raise ValueError(f"{item.name} must be finite")
    finite_numbers(cfg)
    sim, tool, camera = cfg.simulation, cfg.collector, cfg.d405
    positives = [sim.arena_width_m, sim.arena_height_m, sim.time_step_s,
                 sim.perception_period_s, sim.grid_resolution_m, sim.lower_duration_s,
                 sim.close_duration_s, sim.lift_duration_s, cfg.geometry.chassis_radius_m,
                 cfg.geometry.radius_m, tool.opening_width_m, tool.palm_depth_m,
                 tool.pickup_offset_m, tool.carry_offset_m, tool.carry_height_m,
                 tool.alignment_tolerance_m, tool.home_tolerance_m,
                 tool.pickup_speed_m_s, tool.return_speed_m_s, tool.max_payload_kg,
                 tool.max_length_m, tool.max_width_m, tool.max_height_m,
                 tool.operation_timeout_s, tool.verification_timeout_s,
                 tool.navigation_timeout_s, tool.heading_tolerance_rad,
                 tool.stowed_front_m, cfg.geometry.max_linear_vel,
                 cfg.geometry.max_angular_vel, cfg.geometry.scrub_linear_vel,
                 cfg.geometry.nav_linear_vel]
    if min(positives) <= 0 or sim.obstacle_margin_m < 0:
        raise ValueError("dimensions, rates and timeouts must be positive")
    if not 0 < camera.min_depth_m < camera.max_depth_m:
        raise ValueError("invalid D405 depth interval")
    if not all(0 < fov < 180 for fov in (camera.horizontal_fov_deg, camera.vertical_fov_deg)):
        raise ValueError("invalid D405 field of view")
    if not 0 < camera.mount.pitch_deg < 90 or camera.mount.height_m <= 0:
        raise ValueError("camera must be above the floor and pitched downward")
    if tool.pickup_offset_m < cfg.geometry.chassis_radius_m + tool.max_length_m / 2 + sim.obstacle_margin_m:
        raise ValueError("pickup offset must clear the chassis and largest accepted item")
    if tool.approach_standoff_m < 0:
        raise ValueError("approach standoff must be nonnegative")
    if tool.max_pickup_attempts < 1:
        raise ValueError("at least one pickup attempt is required")
    if len(sim.home_pose) != 3 or not np.isfinite(sim.home_pose).all():
        raise ValueError("home_pose must contain finite x, y and yaw")
    if not 0 <= cfg.planning.coverage_lane_overlap < 1:
        raise ValueError("coverage overlap must be in [0, 1)")


@dataclass
class Litter:
    object_id: int
    object_class: str
    position: np.ndarray
    size_m: tuple
    mass_kg: float
    yaw_rad: float = 0.0
    status: str = "floor"
    attempts: int = 0

    @property
    def radius_m(self):
        return float(np.hypot(*self.size_m[:2]) / 2)


@dataclass
class Stain:
    object_id: int
    position: np.ndarray
    radius_m: float
    samples: np.ndarray = field(init=False, repr=False)
    remaining: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        axis = np.linspace(-self.radius_m, self.radius_m, 17)
        x, y = np.meshgrid(axis, axis)
        offsets = np.column_stack([x.ravel(), y.ravel()])
        self.samples = offsets[np.linalg.norm(offsets, axis=1) <= self.radius_m] + self.position
        self.remaining = np.ones(len(self.samples), dtype=bool)

    @property
    def cleaned(self):
        return not self.remaining.any()


class D405Camera:
    def __init__(self, cfg: D405Config):
        self.cfg = cfg
        self.transform = cfg.mount.transform_matrix()
        self.tan_h = np.tan(np.deg2rad(cfg.horizontal_fov_deg / 2))
        self.tan_v = np.tan(np.deg2rad(cfg.vertical_fov_deg / 2))

    def optical_point(self, position, pose):
        x, y, theta = pose
        c, s = np.cos(theta), np.sin(theta)
        relative = np.asarray(position, dtype=float).copy()
        relative[:2] = np.array([[c, s], [-s, c]]) @ (relative[:2] - [x, y])
        return self.transform[:3, :3].T @ (relative - self.transform[:3, 3])

    def visible(self, position, pose):
        x, y, z = self.optical_point(position, pose)
        return bool(self.cfg.min_depth_m <= z <= self.cfg.max_depth_m
                    and abs(x) <= z * self.tan_h and abs(y) <= z * self.tan_v)

    def floor_polygon(self, pose, width, height):
        """Clip the arena floor against the six nominal camera half-planes."""
        polygon = [np.array(p, dtype=float) for p in ((0, 0), (width, 0),
                                                      (width, height), (0, height))]
        def distances(point):
            x, y, z = self.optical_point([*point, 0.0], pose)
            return [z - self.cfg.min_depth_m, self.cfg.max_depth_m - z,
                    z * self.tan_h - x, z * self.tan_h + x,
                    z * self.tan_v - y, z * self.tan_v + y]
        for plane in range(6):
            output = []
            if not polygon:
                break
            previous = polygon[-1]
            before = distances(previous)[plane]
            for current in polygon:
                after = distances(current)[plane]
                if (after >= 0) != (before >= 0):
                    output.append(previous + (current - previous) * before / (before - after))
                if after >= 0:
                    output.append(current)
                previous, before = current, after
            polygon = output
        return np.array(polygon)


class SimulatedCollector:
    def __init__(self, cfg, rng):
        self.cfg = cfg
        self.rng = rng
        self.command = CollectorCommand.STOW
        self.elapsed = 0.0
        self.payload = None
        self.forced_failures = 0
        self.success_probability = 1.0
        self.attempts = 0
        self.feedback = CollectorFeedback(CollectorCommand.STOW, True, False)
        self._completed = True

    def update(self, command, dt, capture):
        if command != self.command:
            self.command, self.elapsed, self._completed = command, 0.0, False
        self.elapsed += dt
        durations = {CollectorCommand.STOW: self.cfg.lower_duration_s,
                     CollectorCommand.LOWER: self.cfg.lower_duration_s,
                     CollectorCommand.CLOSE: self.cfg.close_duration_s,
                     CollectorCommand.LIFT: self.cfg.lift_duration_s,
                     CollectorCommand.HOLD: 0.0}
        if not self._completed and self.elapsed >= durations[command]:
            if command == CollectorCommand.CLOSE:
                self.attempts += 1
                candidate = capture()
                if candidate is not None:
                    candidate.attempts += 1
                forced = self.forced_failures > 0
                if forced:
                    self.forced_failures -= 1
                if candidate is not None and not forced and self.rng.random() < self.success_probability:
                    self.payload = candidate
                    candidate.status = "carried"
            self._completed = True
        self.feedback = CollectorFeedback(command, self._completed, self.payload is not None)
        return self.feedback

    def unload(self):
        if self.payload is not None:
            self.payload.status = "delivered"
        self.payload = None
        self.feedback = CollectorFeedback(self.command, self._completed, False)


class CleaningSimulation:
    def __init__(self, config=None, *, litter=None, stains=None):
        self.cfg = deepcopy(config) if config is not None else simulation_config()
        validate_simulation_config(self.cfg)
        sim = self.cfg.simulation
        self.rng = np.random.default_rng(sim.seed)
        self.perception_rng = np.random.default_rng(sim.seed + 1)
        self.pose = tuple(sim.home_pose)
        self.time_s = 0.0
        self.paused = False
        self.last_velocity = (0.0, 0.0)
        self.safety_stops = 0
        self.camera = D405Camera(self.cfg.d405)
        # Fixed world objects retain their IDs when the short-range camera
        # turns away; missed tracks are not emitted as confirmed observations.
        self.tracker = DetectionTracker(self.cfg.perception, max_missed_frames=None)
        self.collector = SimulatedCollector(sim, self.rng)
        self.navigator = ArenaNavigator(sim.arena_width_m, sim.arena_height_m,
                                        sim.grid_resolution_m, sim.obstacle_margin_m)
        self.litter = deepcopy(litter) if litter is not None else [
            Litter(1, "plastic_bottle", np.array([0.98, 0.50]), (0.19, 0.065, 0.065), 0.025),
            Litter(2, "paper_ball", np.array([1.85, 0.80]), (0.065, 0.065, 0.065), 0.008),
            Litter(3, "paper_cup", np.array([2.48, 1.36]), (0.08, 0.08, 0.10), 0.012),
            Litter(4, "wrapper", np.array([1.25, 1.66]), (0.10, 0.07, 0.002), 0.002, 0.2),
        ]
        self.stains = deepcopy(stains) if stains is not None else [
            Stain(101, np.array([1.60, 0.50]), 0.065),
            Stain(102, np.array([2.35, 1.05]), 0.07),
            Stain(103, np.array([0.85, 1.91]), 0.06),
        ]
        for item in self.litter + self.stains:
            if (len(item.position) != 2 or not np.isfinite(item.position).all()
                    or not item.radius_m <= item.position[0] <= sim.arena_width_m - item.radius_m
                    or not item.radius_m <= item.position[1] <= sim.arena_height_m - item.radius_m):
                raise ValueError("scene objects must fit inside the configured arena")
        # The conservative carrying footprint can rotate throughout this inset.
        margin = collector_navigation_radius(self.cfg.collector,
                    self.cfg.geometry.chassis_radius_m, True) + sim.obstacle_margin_m + 0.025
        if 2 * margin >= min(sim.arena_width_m, sim.arena_height_m):
            raise ValueError("arena is too small for the carrying footprint")
        if not self.navigator.point_clear(self.pose[:2], margin - sim.obstacle_margin_m):
            raise ValueError("home must allow the collector's carrying footprint to rotate")
        coverage = generate_boustrophedon(margin, sim.arena_width_m - margin,
                    margin, sim.arena_height_m - margin, self.cfg.geometry.radius_m,
                    self.cfg.planning.coverage_lane_overlap)
        self.mission = MissionPlanner(self.cfg.planning, self.cfg.geometry, coverage,
                                      collector_cfg=self.cfg.collector, navigator=self.navigator,
                                      home_pose=self.pose)
        self.confirmed = []
        self.visible_ids = set()
        self._next_perception = 0.0
        self.trail = []
        self.deliveries = 0
        self.sensor_blackout_until = 0.0
        self.sensor_drop_probability = 0.0
        self.phantom_detections = []
        self.sensor_frames = 0
        self.missed_observations = 0
        self.path_length_m = 0.0
        self.home_arrivals = []
        self.trajectory = [self.pose[:2]]

    def _observe(self):
        observations = []
        self.visible_ids = set()
        self.sensor_frames += 1
        if self.time_s < self.sensor_blackout_until:
            self.missed_observations += 1
            self.confirmed = self.tracker.update([])
            return
        for item in self.litter:
            if item.status != "floor":
                continue
            point = np.array([*item.position, item.size_m[2] / 2])
            if self.camera.visible(point, self.pose):
                if self.sensor_drop_probability and self.perception_rng.random() < self.sensor_drop_probability:
                    self.missed_observations += 1
                    continue
                self.visible_ids.add(item.object_id)
                # The semantic mock can label a flat wrapper without a depth bump.
                observations.append(Detection("debris", point, item.radius_m, (0, 0),
                    coordinate_frame="odom", object_class=item.object_class,
                    size_m=item.size_m, orientation_rad=item.yaw_rad,
                    estimated_mass_kg=item.mass_kg, observed_at_s=self.time_s))
        for stain in self.stains:
            point = np.array([*stain.position, 0.0])
            if not stain.cleaned and self.camera.visible(point, self.pose):
                self.visible_ids.add(stain.object_id)
                observations.append(Detection("stain", point, stain.radius_m, (0, 0),
                    coordinate_frame="odom", observed_at_s=self.time_s))
        for phantom in self.phantom_detections:
            if self.camera.visible(phantom.position_robot, self.pose):
                observations.append(Detection("debris", phantom.position_robot.copy(), phantom.radius_m,
                    (0, 0), coordinate_frame="odom", object_class=phantom.object_class,
                    size_m=phantom.size_m, orientation_rad=phantom.orientation_rad,
                    estimated_mass_kg=phantom.estimated_mass_kg, observed_at_s=self.time_s))
        self.confirmed = self.tracker.update(observations)

    def _capture_candidate(self):
        target = self.mission.current_target
        if target is None or target.kind != "debris":
            return None
        palm = collector_position(self.pose, self.cfg.collector)
        for item in self.litter:
            if (item.status == "floor" and item.object_class == target.object_class
                    and np.linalg.norm(item.position - target.position_robot[:2]) < 0.06
                    and np.linalg.norm(item.position - palm) <= 2 * self.cfg.collector.alignment_tolerance_m):
                if item.size_m[0] > 1.4 * item.size_m[1]:
                    error = abs((item.yaw_rad - self.pose[2] + np.pi / 2) % np.pi - np.pi / 2)
                    if error > 2 * self.cfg.collector.heading_tolerance_rad:
                        continue
                return item
        return None

    def collector_polygon(self, pose=None):
        pose = self.pose if pose is None else pose
        cfg = self.cfg.collector
        extended = self.mission.collector_command != CollectorCommand.STOW or self.collector.payload is not None
        offset = self.collector_offset()
        front = offset + cfg.palm_depth_m / 2 if extended else cfg.stowed_front_m
        half_width = cfg.opening_width_m / 2 if extended else cfg.opening_width_m / 4
        if self.collector.payload is not None:
            front = max(front, offset + self.collector.payload.size_m[0] / 2)
            half_width = max(half_width, self.collector.payload.size_m[1] / 2)
        rear = self.cfg.geometry.chassis_radius_m * 0.6
        points = np.array([[rear, -half_width], [front, -half_width],
                           [front, half_width], [rear, half_width]])
        c, s = np.cos(pose[2]), np.sin(pose[2])
        return points @ np.array([[c, s], [-s, c]]) + pose[:2]

    def collector_offset(self):
        cfg = self.cfg.collector
        if self.mission.collector_command == CollectorCommand.HOLD:
            return cfg.carry_offset_m
        if self.mission.collector_command == CollectorCommand.LIFT:
            progress = min(1., self.collector.elapsed/self.cfg.simulation.lift_duration_s)
            return cfg.pickup_offset_m + (cfg.carry_offset_m-cfg.pickup_offset_m)*progress
        return cfg.pickup_offset_m

    def _collision(self, pose):
        sim, radius = self.cfg.simulation, self.cfg.geometry.chassis_radius_m
        x, y, theta = pose
        if not (radius <= x <= sim.arena_width_m - radius and radius <= y <= sim.arena_height_m - radius):
            return "chassis would cross arena wall"
        polygon = self.collector_polygon(pose)
        if (np.any(polygon < 0) or np.any(polygon[:, 0] > sim.arena_width_m)
                or np.any(polygon[:, 1] > sim.arena_height_m)):
            return "collector would cross arena wall"
        c, s = np.cos(theta), np.sin(theta)
        local_poly = (polygon - [x, y]) @ np.array([[c, -s], [s, c]])
        low, high = local_poly.min(axis=0), local_poly.max(axis=0)
        for item in self.litter:
            if item.status != "floor":
                continue
            if np.linalg.norm(item.position - [x, y]) < radius + item.radius_m:
                return "chassis would touch " + item.object_class
            target = self.mission.current_target
            selected = (target is not None and target.kind == "debris"
                        and item.object_class == target.object_class
                        and np.linalg.norm(item.position - target.position_robot[:2]) < 0.06)
            if selected:
                continue  # only the collector may intentionally contact its target
            local = np.array([[c, s], [-s, c]]) @ (item.position - [x, y])
            closest = np.clip(local, low, high)
            if np.linalg.norm(local - closest) < item.radius_m:
                return "collector would touch other litter"
        return None

    def step(self):
        if self.paused:
            return 0.0, 0.0
        dt = self.cfg.simulation.time_step_s
        if self.time_s + 1e-9 >= self._next_perception:
            self._observe()
            self._next_perception = self.time_s + self.cfg.simulation.perception_period_s
        stains = [d for d in self.confirmed if d.kind == "stain"]
        debris = [d for d in self.confirmed if d.kind == "debris"]
        previous_state = self.mission.state
        v, w = self.mission.update(self.pose, stains, debris, dt=dt,
                                    collector_feedback=self.collector.feedback)
        self.collector.update(self.mission.collector_command, dt, self._capture_candidate)
        x, y, theta = self.pose
        candidate = (x + v * np.cos(theta + w * dt / 2) * dt,
                     y + v * np.sin(theta + w * dt / 2) * dt,
                     (theta + w * dt + np.pi) % (2 * np.pi) - np.pi)
        # Also check stationary extension of the claw, not just base motion.
        collision = self._collision(candidate) if self.mission.state != MissionState.FAULT else None
        if collision:
            self.safety_stops += 1
            self.mission._transition(MissionState.FAULT, "simulation safety stop: " + collision)
            v, w = 0.0, 0.0
        else:
            self.path_length_m += float(np.linalg.norm(np.array(candidate[:2]) - self.pose[:2]))
            self.pose = candidate
        if (previous_state != MissionState.WAITING_FOR_UNLOAD
                and self.mission.state == MissionState.WAITING_FOR_UNLOAD):
            home = self.mission.home_pose
            self.home_arrivals.append({"time_s": round(self.time_s, 3), "pose": list(self.pose),
                "position_error_m": float(np.linalg.norm(np.array(self.pose[:2]) - home[:2])),
                "heading_error_rad": abs(float(self.mission._angle_error(self.pose[2], home[2])))})
        if np.linalg.norm(np.array(self.pose[:2]) - self.trajectory[-1]) >= 0.01:
            self.trajectory.append(self.pose[:2])
        if self.collector.payload is not None:
            self.collector.payload.position = np.array(self.pose[:2]) + self.collector_offset() * np.array(
                [np.cos(self.pose[2]), np.sin(self.pose[2])])
            self.collector.payload.yaw_rad = self.pose[2]
        if self.mission.mop_enabled:
            self.trail.append(self.pose[:2])
            for stain in self.stains:
                stain.remaining &= np.linalg.norm(stain.samples - self.pose[:2], axis=1) > self.cfg.geometry.radius_m
        self.time_s += dt
        self.last_velocity = (v, w)
        return v, w

    def unload(self):
        """Model the operator removing the payload, then pressing resume."""
        if self.mission.state != MissionState.WAITING_FOR_UNLOAD:
            return False
        self.collector.unload()
        self.deliveries += 1
        return self.mission.confirm_unloaded()

    def force_next_failure(self):
        self.collector.forced_failures += 1
        self.mission._record("next pickup will fail (injected)")

    def blackout_sensor(self, duration_s=2.0):
        if not np.isfinite(duration_s) or duration_s <= 0:
            raise ValueError("blackout duration must be positive")
        self.sensor_blackout_until = max(self.sensor_blackout_until, self.time_s + duration_s)
        self.mission._record(f"camera blackout injected for {duration_s:.2f}s")

    def drop_payload(self):
        if self.collector.payload is None:
            return False
        payload = self.collector.payload
        payload.status = "floor"
        payload.position = collector_position(self.pose, self.cfg.collector)
        self.collector.payload = None
        self.collector.feedback = CollectorFeedback(CollectorCommand.HOLD, True, False)
        self.mission._record("payload drop injected")
        return True

    def block_home(self):
        """Fault fixture: a reported obstruction at home, not an RGB inference."""
        home = np.array(self.mission.home_pose[:2])
        if np.linalg.norm(home - self.pose[:2]) < 0.7:
            return False
        obstacle_id = max((x.object_id for x in self.litter), default=0) + 1000
        obstacle = Litter(obstacle_id, "obstacle", home, (.24, .24, .30), 2.0)
        self.litter.append(obstacle)
        self.navigator.observe([Detection("debris", np.array([*home, .15]), obstacle.radius_m,
            (0, 0), track_id=-obstacle_id, coordinate_frame="odom", object_class="obstacle")])
        self.mission._record("home obstruction reported (injected)")
        return True

    def spawn_litter(self, object_class=None):
        classes = ("plastic_bottle", "wrapper", "paper_cup", "paper_ball")
        object_class = object_class or str(self.rng.choice(classes))
        specs = {"plastic_bottle": ((0.19, 0.065, 0.065), 0.025),
                 "wrapper": ((0.10, 0.07, 0.002), 0.002),
                 "paper_cup": ((0.08, 0.08, 0.10), 0.012),
                 "paper_ball": ((0.065, 0.065, 0.065), 0.008)}
        if object_class not in specs:
            raise ValueError("unsupported litter class")
        for _ in range(100):
            point = self.rng.uniform([0.75, 0.65], [self.cfg.simulation.arena_width_m - 0.6,
                                                   self.cfg.simulation.arena_height_m - 0.6])
            if np.linalg.norm(point - self.pose[:2]) < 0.65:
                continue
            if any(item.status == "floor" and np.linalg.norm(point - item.position) < 0.4
                   for item in self.litter):
                continue
            size, mass = specs[object_class]
            self.litter.append(Litter(max((x.object_id for x in self.litter), default=0) + 1,
                                       object_class, point, size, mass))
            self._restart_if_done()
            return True
        return False

    def spawn_stain(self):
        point = self.rng.uniform([0.7, 0.7], [self.cfg.simulation.arena_width_m - 0.7,
                                            self.cfg.simulation.arena_height_m - 0.7])
        self.stains.append(Stain(max((x.object_id for x in self.stains), default=100) + 1,
                                  point, 0.07))
        self._restart_if_done()

    def _restart_if_done(self):
        if self.mission.state == MissionState.DONE:
            self.mission.coverage_index = 0
            self.mission._transition(MissionState.COVERAGE, "new target; restart coverage sweep")

    def summary(self):
        return {"state": self.mission.state.name, "simulation_time_s": round(self.time_s, 2),
                "pose": [round(float(x), 4) for x in self.pose],
                "home_pose": list(self.mission.home_pose), "delivered": self.deliveries,
                "floor_litter": sum(x.status == "floor" and x.object_class in LITTER_CLASSES for x in self.litter),
                "obstacles": sum(x.status == "floor" and x.object_class not in LITTER_CLASSES for x in self.litter),
                "carrying": self.collector.payload.object_class if self.collector.payload else None,
                "pickup_attempts": self.collector.attempts,
                "failed_tracks": sorted(self.mission.failed_pickup_ids),
                "deferred_tracks": sorted(self.mission.deferred_pickup_ids),
                "stains_cleaned": sum(x.cleaned for x in self.stains),
                "safety_stops": self.safety_stops,
                "path_length_m": round(self.path_length_m, 3),
                "sensor_frames": self.sensor_frames,
                "missed_observations": self.missed_observations,
                "home_arrivals": self.home_arrivals,
                "coverage_index": self.mission.coverage_index,
                "coverage_segments": max(0, len(self.mission.coverage_path) - 1),
                "objects": [{"id": x.object_id, "class": x.object_class,
                             "status": x.status, "attempts": x.attempts} for x in self.litter]}

    def run(self, steps=20000, *, auto_unload=False):
        for _ in range(steps):
            self.step()
            state = self.mission.state
            if state in (MissionState.DONE, MissionState.FAULT):
                break
            if state == MissionState.WAITING_FOR_UNLOAD:
                if not auto_unload:
                    break
                if self.time_s - self.mission._state_since >= 1.0:
                    self.unload()
        return self.summary()
