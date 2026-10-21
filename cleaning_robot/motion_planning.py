"""
motion_planning.py
-------------------
Path generation + control for the two operating modes, and the state
machine that switches between them.

Coverage guarantee for the perimeter mop
=========================================
The mop's cleaning footprint at any instant is (approximately) the disk of
radius R centered on the robot. Two different path topologies are used
here, and they need DIFFERENT spacing rules to guarantee full coverage:

- Boustrophedon lanes: each lane independently spans the whole area, so
  a point between two adjacent lanes always has both a "lane above" and a
  "lane below" within reach. Spacing lanes by

      lane_pitch <= 2*R*(1 - overlap_ratio)

  guarantees full coverage (overlap_ratio in ~[0.1, 0.4] as a safety
  margin for odometry drift and pose-tracking error).

- Archimedean spiral: a single continuous curve only revisits a given
  angle once per full revolution, so the "next crossing out" at a given
  angle is a full turn away, not just "the next line over". For angles
  near the start of the final turn, the outermost same-angle crossing is
  a full `pitch` short of the spiral's true tip, so the worst-case
  covered radius is (r_max - pitch + R). Guaranteeing that still reaches
  the target radius requires the stricter

      spiral_pitch <= R*(1 - overlap_ratio)

  i.e. half the lane spacing used for boustrophedon lanes. See
  generate_spot_scrub_path() and the Monte-Carlo coverage check in
  test_pipeline.py for the derivation and a numerical proof.

Spot-scrub geometry
====================
For a stain with radius r_stain (max distance from centroid to its
boundary) detected by the perception stack:
  - If r_stain + margin <= R, a single centered dwell already covers it;
    we still trace a small agitation circle for mechanical scrubbing
    action rather than sitting static.
  - Otherwise, trace an outward Archimedean spiral r(theta) = b*theta from
    the centroid out to r_max = r_stain + margin, with radial pitch b*2*pi
    satisfying the coverage rule above, then spiral back inward
    (double-pass) so every point gets two overlapping mop strokes in
    opposite sweep directions -- this is the "oscillatory scrubbing"
    motion requested, generated as one continuous smooth path rather than
    discrete back-and-forth strokes so the differential-drive controller
    never has to stop-reverse-accelerate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple

import numpy as np

from .config import PlanningConfig, RobotGeometry, CollectorConfig
from .perception import Detection, detection_to_planning_frame
from .collector import (CollectorCommand, CollectorFeedback, pickup_rejection,
                        collector_navigation_radius)
from .navigation import ArenaNavigator


# ----------------------------------------------------------------------------
# Path generators
# ----------------------------------------------------------------------------

def generate_spot_scrub_path(robot_radius_m: float, stain_radius_m: float,
                              overlap_ratio: float, margin_m: float,
                              points_per_rev: int = 24,
                              double_pass: bool = False) -> np.ndarray:
    """Local-frame (centered on the stain centroid) waypoints for spot
    scrubbing. See module docstring for the coverage derivation.

    Returns:
        (N, 2) array of local (x, y) waypoints, first point is (0, 0).
    """
    effective_radius = stain_radius_m + margin_m

    if effective_radius <= robot_radius_m:
        # One footprint already covers the whole stain -- trace a small
        # agitation loop instead of sitting perfectly still, for
        # mechanical scrubbing action.
        agitation_r = 0.25 * robot_radius_m
        thetas = np.linspace(0, 2 * np.pi, points_per_rev)
        loop = np.stack([agitation_r * np.cos(thetas),
                          agitation_r * np.sin(thetas)], axis=1)
        return np.concatenate([[[0.0, 0.0]], loop], axis=0)

    # NOTE on pitch: unlike the boustrophedon lanes below (each lane spans
    # the whole area independently, so 2R spacing between lanes is safe),
    # a single continuous spiral only revisits a given angle once per full
    # turn. For angles near the start of the final turn, the outermost
    # same-angle crossing is a full `pitch` short of the spiral's true tip
    # -- so the worst-case covered radius is (r_max - pitch + R), and
    # guaranteeing that reaches the target requires pitch <= R, not 2R.
    pitch = robot_radius_m * (1.0 - overlap_ratio)
    b = pitch / (2 * np.pi)
    theta_raw = effective_radius / b
    # Round UP to a whole number of revolutions so the final turn is
    # complete (an incomplete last turn would leave part of the boundary
    # with no outer crossing at all -- see coverage test in test_pipeline.py).
    n_turns = max(int(np.ceil(theta_raw / (2 * np.pi))), 1)
    theta_max = n_turns * 2 * np.pi
    n_points = max(int(points_per_rev * n_turns), 16)

    thetas = np.linspace(0.0, theta_max, n_points)
    r = b * thetas
    x = r * np.cos(thetas)
    y = r * np.sin(thetas)
    outward = np.stack([x, y], axis=1)

    path = outward
    if double_pass:
        inward = outward[::-1]
        path = np.concatenate([outward, inward], axis=0)

    return np.concatenate([[[0.0, 0.0]], path], axis=0)


def generate_boustrophedon(x_min: float, x_max: float, y_min: float, y_max: float,
                            robot_radius_m: float, overlap_ratio: float) -> np.ndarray:
    """Lawnmower coverage lanes over an axis-aligned free-space rectangle.
    In production, replace the rectangle with the actual free-space
    boundary from your occupancy map / SLAM output and clip each lane
    against it; this keeps the generator itself map-agnostic.

    Returns (N, 2) array of waypoints, alternating lane direction.
    """
    lane_spacing = 2.0 * robot_radius_m * (1.0 - overlap_ratio)
    lane_spacing = max(lane_spacing, 1e-3)
    lanes_y = np.arange(y_min, y_max + 1e-9, lane_spacing)
    if len(lanes_y) and y_max - lanes_y[-1] > 1e-9:
        lanes_y = np.append(lanes_y, y_max)

    waypoints = []
    for i, y in enumerate(lanes_y):
        if i % 2 == 0:
            waypoints.append((x_min, y))
            waypoints.append((x_max, y))
        else:
            waypoints.append((x_max, y))
            waypoints.append((x_min, y))
    return np.array(waypoints)


# ----------------------------------------------------------------------------
# Pure-pursuit path follower
# ----------------------------------------------------------------------------

class PurePursuitController:
    """Continuous Polyline Lookahead Pure Pursuit. Monotonically tracks
    path segments with smooth in-place rotation for large heading errors."""

    def __init__(self, cfg: PlanningConfig, geometry: RobotGeometry):
        self.cfg = cfg
        self.geometry = geometry

    def step(self, pose_xytheta: Tuple[float, float, float], path_xy: np.ndarray,
             path_index: int, cruise_speed: float
             ) -> Tuple[float, float, int, bool]:
        x, y, theta = pose_xytheta
        pos = np.array([x, y])
        n = len(path_xy)
        if n == 0:
            return 0.0, 0.0, path_index, True
        if n == 1:
            dist = float(np.linalg.norm(path_xy[0] - pos))
            if dist < max(self.cfg.goal_tolerance_m, 0.05):
                return 0.0, 0.0, 0, True
            dx, dy = path_xy[0][0] - x, path_xy[0][1] - y
            ang = np.arctan2(dy, dx)
            err = (ang - theta + np.pi) % (2 * np.pi) - np.pi
            if abs(err) > np.deg2rad(45):
                return 0.0, float(np.clip(2.0 * err, -self.geometry.max_angular_vel, self.geometry.max_angular_vel)), 0, False
            return float(cruise_speed * np.cos(err)), float(np.clip(2.0 * err, -self.geometry.max_angular_vel, self.geometry.max_angular_vel)), 0, False

        lookahead = self.cfg.pure_pursuit_lookahead_m

        # 1. Advance segment index monotonically
        idx = min(path_index, n - 2)
        while idx < n - 2:
            p1 = path_xy[idx]
            p2 = path_xy[idx + 1]
            v_seg = p2 - p1
            len_sq = np.dot(v_seg, v_seg)
            if len_sq < 1e-6:
                idx += 1
                continue
            t = np.dot(pos - p1, v_seg) / len_sq
            if t >= 0.90 or np.linalg.norm(pos - p2) < lookahead * 0.4:
                idx += 1
            else:
                break

        # Check goal reach
        dist_to_final = float(np.linalg.norm(path_xy[-1] - pos))
        if idx >= n - 2 and dist_to_final < max(self.cfg.goal_tolerance_m, 0.06):
            return 0.0, 0.0, idx, True

        # 2. Continuous lookahead along polyline
        p1 = path_xy[idx]
        p2 = path_xy[idx + 1]
        v_seg = p2 - p1
        len_seg = float(np.linalg.norm(v_seg))
        t = float(np.clip(np.dot(pos - p1, v_seg) / max(len_seg ** 2, 1e-6), 0.0, 1.0))
        proj_pt = p1 + t * v_seg

        dist_needed = lookahead
        dist_rem_in_seg = (1.0 - t) * len_seg

        if dist_rem_in_seg >= dist_needed:
            target = proj_pt + (v_seg / max(len_seg, 1e-6)) * dist_needed
        else:
            dist_needed -= dist_rem_in_seg
            target = path_xy[-1]
            for k in range(idx + 1, n - 1):
                s_len = float(np.linalg.norm(path_xy[k + 1] - path_xy[k]))
                if s_len >= dist_needed:
                    target = path_xy[k] + (path_xy[k + 1] - path_xy[k]) * (dist_needed / s_len)
                    break
                dist_needed -= s_len

        # 3. Steering & velocity computation
        dx = target[0] - x
        dy = target[1] - y
        target_angle = np.arctan2(dy, dx)
        heading_error = (target_angle - theta + np.pi) % (2 * np.pi) - np.pi

        if abs(heading_error) > np.deg2rad(45):
            linear = 0.0
            angular = float(np.clip(2.5 * heading_error, -self.geometry.max_angular_vel, self.geometry.max_angular_vel))
            if abs(angular) < 0.4:
                angular = float(np.sign(heading_error) * 0.5)
        else:
            curvature = 2.0 * np.sin(heading_error) / max(lookahead, 0.05)
            linear = cruise_speed * max(0.2, np.cos(heading_error))
            if dist_to_final < 0.3 and idx >= n - 2:
                linear = min(linear, max(0.06, dist_to_final * 1.5))
            angular = float(np.clip(linear * curvature + 0.8 * heading_error,
                                     -self.geometry.max_angular_vel, self.geometry.max_angular_vel))

        return float(linear), float(angular), idx, False

# ----------------------------------------------------------------------------
# Reactive debris avoidance (lightweight; see note below)
# ----------------------------------------------------------------------------

def repulsive_angular_bias(pose_xytheta: Tuple[float, float, float],
                            debris: List[Detection], robot_radius_m: float,
                            safety_margin_m: float = 0.05,
                            lookahead_m: float = 0.6) -> float:
    """Reactive steering bias away from nearby debris in front of the robot.

    NOTE: this is a lightweight local layer suitable for a Roomba-scale bot
    on open room floors, meant to nudge the path follower around small
    obstacles it's about to drive over. It is not a substitute for a real
    costmap-based local planner (e.g. Nav2's controller server with an
    inflation layer) -- if you already have Nav2 in the stack, feed debris
    detections into its costmap as lethal/inflated obstacles instead of
    using this function.
    """
    x, y, theta = pose_xytheta
    bias = 0.0
    for det in debris:
        dx = det.position_robot[0] - x
        dy = det.position_robot[1] - y
        local_x = np.cos(-theta) * dx - np.sin(-theta) * dy
        local_y = np.sin(-theta) * dx + np.cos(-theta) * dy
        danger_radius = robot_radius_m + det.radius_m + safety_margin_m
        dist = np.hypot(local_x, local_y)
        if local_x > 0 and local_x < lookahead_m and dist < (danger_radius + lookahead_m):
            # Push away from the obstacle's side; strength grows as it gets closer.
            closeness = max(0.0, 1.0 - dist / (danger_radius + lookahead_m))
            bias += -np.sign(local_y if abs(local_y) > 1e-3 else 1.0) * closeness * 1.5
    return float(np.clip(bias, -1.0, 1.0))


# ----------------------------------------------------------------------------
# Mission state machine
# ----------------------------------------------------------------------------

class MissionState(Enum):
    COVERAGE = auto()
    DIVERTING = auto()
    SCRUBBING = auto()
    APPROACHING_DEBRIS = auto()
    ALIGNING_PICKUP = auto()
    COLLECTING = auto()
    VERIFYING_PICKUP = auto()
    RECOVERING_PICKUP = auto()
    RETURNING_HOME = auto()
    WAITING_FOR_UNLOAD = auto()
    RESUMING_COVERAGE = auto()
    FAULT = auto()
    DONE = auto()


@dataclass
class _StainMissionPlanner:
    """Owns the coverage path and orchestrates interrupt-divert-scrub-resume.

    Debris detections never generate a divert -- they only bias steering
    (see repulsive_angular_bias) and are skipped by the coverage lane
    generator's exclusion list if you extend it with a costmap. Stains
    generate a divert once confirmed and within `divert_max_distance_m`.
    """
    cfg: PlanningConfig
    geometry: RobotGeometry
    coverage_path: np.ndarray  # (N, 2) in planning/odom frame

    state: MissionState = MissionState.COVERAGE
    coverage_index: int = 0
    resume_index: int = 0
    scrub_path: Optional[np.ndarray] = None
    scrub_index: int = 0
    current_target: Optional[Detection] = None
    pending_stains: List[Detection] = field(default_factory=list)
    handled_track_ids: set = field(default_factory=set)

    def __post_init__(self):
        self._controller = PurePursuitController(self.cfg, self.geometry)

    # -- public API ----------------------------------------------------

    def update(self, pose_xytheta: Tuple[float, float, float],
               confirmed_stains: List[Detection],
               confirmed_debris: List[Detection]) -> Tuple[float, float]:
        """Call once per control tick. Returns (linear_x, angular_z)."""
        self._ingest_new_stains(pose_xytheta, confirmed_stains)

        if self.state == MissionState.COVERAGE:
            linear, angular = self._run_coverage(pose_xytheta)
        elif self.state == MissionState.DIVERTING:
            linear, angular = self._run_diverting(pose_xytheta)
        elif self.state == MissionState.SCRUBBING:
            linear, angular = self._run_scrubbing(pose_xytheta)
        else:  # DONE
            linear, angular = 0.0, 0.0

        # Reactive debris nudge applies in every driving state.
        if self.state != MissionState.DONE and confirmed_debris:
            bias = repulsive_angular_bias(pose_xytheta, confirmed_debris, self.geometry.radius_m)
            angular = float(np.clip(angular + bias, -self.geometry.max_angular_vel,
                                     self.geometry.max_angular_vel))

        return linear, angular

    # -- internals -------------------------------------------------------

    def _ingest_new_stains(self, pose_xytheta, confirmed_stains: List[Detection]) -> None:
        known_ids = {t.track_id for t in self.pending_stains}
        if self.current_target is not None:
            known_ids.add(self.current_target.track_id)
        for stain in confirmed_stains:
            if stain.track_id in self.handled_track_ids:
                continue
            if stain.track_id in known_ids:
                continue
            self.pending_stains.append(stain)
            known_ids.add(stain.track_id)

    def _closest_pending_stain(self, pose_xytheta) -> Tuple[Optional[Detection], Optional[int]]:
        if not self.pending_stains:
            return None, None
        pos = np.array(pose_xytheta[:2])
        dists = [np.linalg.norm(s.position_robot[:2] - pos) for s in self.pending_stains]
        i = int(np.argmin(dists))
        if dists[i] < self.cfg.divert_max_distance_m:
            return self.pending_stains[i], i
        return None, None

    def _run_coverage(self, pose_xytheta) -> Tuple[float, float]:
        target, idx = self._closest_pending_stain(pose_xytheta)
        if target is not None and idx is not None:
            self.pending_stains.pop(idx)
            self.current_target = target
            self.resume_index = self.coverage_index
            self.state = MissionState.DIVERTING
            return self._run_diverting(pose_xytheta)

        linear, angular, idx, finished = self._controller.step(
            pose_xytheta, self.coverage_path, self.coverage_index,
            self.geometry.max_linear_vel)
        self.coverage_index = idx
        if finished:
            self.state = MissionState.DONE
        return linear, angular

    def _run_diverting(self, pose_xytheta) -> Tuple[float, float]:
        assert self.current_target is not None
        target_xy = self.current_target.position_robot[:2].reshape(1, 2)
        # Two-point path: current heading toward the stain centroid. A more
        # elaborate version could route through the occupancy map; a stain
        # target is a soft goal (not lethal to graze) so a direct path is
        # generally fine over the divert_max_distance_m range.
        path = np.concatenate([[pose_xytheta[:2]], target_xy], axis=0)
        linear, angular, _, finished = self._controller.step(
            pose_xytheta, path, 0, self.geometry.nav_linear_vel)
        if finished:
            self._start_scrub()
            return 0.0, 0.0
        return linear, angular

    def _start_scrub(self) -> None:
        assert self.current_target is not None
        local_path = generate_spot_scrub_path(
            self.geometry.radius_m, self.current_target.radius_m,
            self.cfg.spiral_overlap, self.cfg.spiral_margin_m)
        self.scrub_path = local_path + self.current_target.position_robot[:2]
        self.scrub_index = 0
        self.state = MissionState.SCRUBBING

    def _run_scrubbing(self, pose_xytheta) -> Tuple[float, float]:
        assert self.scrub_path is not None
        linear, angular, idx, finished = self._controller.step(
            pose_xytheta, self.scrub_path, self.scrub_index,
            self.geometry.scrub_linear_vel)
        self.scrub_index = idx
        if finished:
            if self.current_target is not None and self.current_target.track_id is not None:
                self.handled_track_ids.add(self.current_target.track_id)
            self.current_target = None
            self.scrub_path = None
            
            # Intelligently resume at the closest forward waypoint along coverage path
            if self.coverage_path is not None and len(self.coverage_path) > 0:
                pos = np.array(pose_xytheta[:2])
                search_end = min(len(self.coverage_path), self.resume_index + 60)
                window = self.coverage_path[self.resume_index:search_end]
                if len(window) > 0:
                    dists = np.linalg.norm(window - pos, axis=1)
                    best_offset = int(np.argmin(dists))
                    self.coverage_index = min(len(self.coverage_path) - 1, self.resume_index + best_offset + 1)
                else:
                    self.coverage_index = self.resume_index
            self.state = MissionState.COVERAGE
            return 0.0, 0.0
        return linear, angular


@dataclass
class MissionPlanner(_StainMissionPlanner):
    """Shared mop/collector mission, independent of ROS and simulation.

    Inputs must be observations in a fixed planning frame, or explicitly
    marked base_link observations accompanied by their acquisition pose.
    Collection defaults off. Enabling it requires actuator feedback.
    """
    collector_cfg: CollectorConfig = field(default_factory=CollectorConfig)
    navigator: Optional[ArenaNavigator] = None
    home_pose: Optional[Tuple[float, float, float]] = None

    def __post_init__(self):
        super().__post_init__()
        self.time_s = 0.0
        self.collector_command = CollectorCommand.STOW
        self.carried_target = None
        self.failed_pickup_ids = set()
        self.deferred_pickup_ids = {}
        self.known_debris = {}
        self.events = []
        self.pickup_attempts = 0
        self._state_since = 0.0
        self._phase_since = 0.0
        self._pickup_phase = CollectorCommand.LOWER
        self._resume_pose = None
        self._pickup_pose = None
        self._approach_pose = None
        self._route = None
        self._route_key = None
        self._route_index = 1
        self._feedback = None
        self._last_payload_evidence = None
        self.last_navigation_error = None
        self._record("mission initialized")

    def _record(self, message):
        self.events.append({"time_s": round(self.time_s, 3), "state": self.state.name,
                            "message": message,
                            "track_id": getattr(self.current_target, "track_id", None)})

    def _transition(self, state, message):
        self.state = state
        self._state_since = self.time_s
        self._route = self._route_key = None
        self._record(message)

    @property
    def mop_enabled(self):
        return self.state in (MissionState.COVERAGE, MissionState.SCRUBBING)

    @property
    def active_route(self):
        return self._route

    def confirm_unloaded(self) -> bool:
        """Explicit operator acknowledgement. Ignored outside the home wait."""
        if self.state != MissionState.WAITING_FOR_UNLOAD:
            return False
        self.carried_target = None
        self.current_target = None
        self.collector_command = CollectorCommand.STOW
        self._transition(MissionState.RESUMING_COVERAGE, "operator confirmed unloading")
        return True

    def update(self, pose_xytheta, confirmed_stains, confirmed_debris, *,
               dt: float = 0.05, collector_feedback: Optional[CollectorFeedback] = None):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be a finite positive number")
        self.time_s += dt
        if self.home_pose is None:
            self.home_pose = tuple(pose_xytheta)
        self._feedback = collector_feedback
        if collector_feedback is not None and collector_feedback.payload_present is True:
            self._last_payload_evidence = self.time_s
        stains = [detection_to_planning_frame(d, pose_xytheta) for d in confirmed_stains]
        debris = [detection_to_planning_frame(d, pose_xytheta) for d in confirmed_debris]
        self._ingest_new_stains(pose_xytheta, stains)
        for det in debris:
            if det.track_id is None or det.track_id in self.handled_track_ids:
                continue
            self.known_debris[det.track_id] = det
            if (self.current_target is not None and self.current_target.kind == "debris"
                    and self.current_target.track_id == det.track_id):
                self.current_target = det
        if self.navigator is not None:
            self.navigator.observe(self.known_debris.values(), self.handled_track_ids)
        if (collector_feedback is not None and collector_feedback.fault
                and self.collector_cfg.enabled and self.state != MissionState.FAULT):
            self.collector_command = CollectorCommand.HOLD
            self._transition(MissionState.FAULT, "collector fault: " + collector_feedback.fault)

        handlers = {
            MissionState.COVERAGE: self._run_coverage,
            MissionState.DIVERTING: self._run_diverting,
            MissionState.SCRUBBING: self._run_scrubbing,
            MissionState.APPROACHING_DEBRIS: self._run_approach,
            MissionState.ALIGNING_PICKUP: self._run_alignment,
            MissionState.COLLECTING: self._run_collection,
            MissionState.VERIFYING_PICKUP: self._run_verification,
            MissionState.RECOVERING_PICKUP: self._run_recovery,
            MissionState.RETURNING_HOME: self._run_return,
            MissionState.RESUMING_COVERAGE: self._run_resume,
        }
        handler = handlers.get(self.state)
        linear, angular = handler(pose_xytheta) if handler is not None else (0.0, 0.0)
        if self.state in (MissionState.FAULT, MissionState.DONE, MissionState.WAITING_FOR_UNLOAD,
                          MissionState.COLLECTING, MissionState.VERIFYING_PICKUP,
                          MissionState.RECOVERING_PICKUP):
            linear, angular = 0.0, 0.0
        # The arena router handles avoidance in simulation. Keep the existing
        # lightweight steering layer for the legacy mop-only ROS integration.
        if self.navigator is None and self.state in (
                MissionState.COVERAGE, MissionState.DIVERTING, MissionState.SCRUBBING):
            obstacles = [d for d in debris if d.track_id not in self.handled_track_ids]
            angular = float(np.clip(angular + repulsive_angular_bias(
                pose_xytheta, obstacles, self.geometry.chassis_radius_m),
                -self.geometry.max_angular_vel, self.geometry.max_angular_vel))
        return float(linear), float(angular)

    def _save_checkpoint(self, pose):
        self.resume_index = self.coverage_index
        self._resume_pose = tuple(pose)

    def _navigation_radius(self, extended=False):
        if not self.collector_cfg.enabled:
            return self.geometry.chassis_radius_m
        return collector_navigation_radius(self.collector_cfg, self.geometry.chassis_radius_m, extended)

    @staticmethod
    def _angle_error(goal, actual):
        return (goal - actual + np.pi) % (2 * np.pi) - np.pi

    def _drive_point(self, pose, goal, speed, tolerance):
        delta = np.asarray(goal) - np.asarray(pose[:2])
        distance = float(np.linalg.norm(delta))
        if distance <= tolerance:
            return 0.0, 0.0, True
        error = self._angle_error(np.arctan2(delta[1], delta[0]), pose[2])
        angular = float(np.clip(3.0 * error, -self.geometry.max_angular_vel,
                                self.geometry.max_angular_vel))
        linear = min(speed, 1.8 * distance) * max(0.0, np.cos(error)) if abs(error) < 0.35 else 0.0
        return linear, angular, False

    def _navigate(self, pose, goal, speed, *, extended=False, tolerance=None, heading=None, ignore_id=None):
        tolerance = self.collector_cfg.home_tolerance_m if tolerance is None else tolerance
        goal = np.asarray(goal, dtype=float)
        radius = self._navigation_radius(extended)
        revision = self.navigator.revision if self.navigator is not None else 0
        key = (tuple(goal), radius, revision, ignore_id)
        if self._route_key != key or self._route is None:
            self._route = (self.navigator.route(pose[:2], goal, radius, ignore_id) if self.navigator is not None
                           else np.array([pose[:2], goal]))
            self._route_key, self._route_index = key, 1
            self.last_navigation_error = None if self._route is not None else "no collision-free route"
        if self._route is None:
            return 0.0, 0.0, False
        while self._route_index < len(self._route) - 1 and np.linalg.norm(
                self._route[self._route_index] - pose[:2]) < 0.02:
            self._route_index += 1
        is_last = self._route_index == len(self._route) - 1
        v, w, arrived = self._drive_point(pose, self._route[self._route_index], speed,
                                         tolerance if is_last else 0.02)
        if arrived and is_last and heading is not None:
            error = self._angle_error(heading, pose[2])
            if abs(error) > self.collector_cfg.heading_tolerance_rad:
                return 0.0, float(np.clip(3 * error, -self.geometry.max_angular_vel,
                                           self.geometry.max_angular_vel)), False
        return v, w, arrived and is_last

    def _select_pickup(self, pose):
        candidates = sorted(self.known_debris.values(),
                            key=lambda d: np.linalg.norm(d.position_robot[:2] - pose[:2]))
        for target in candidates:
            if target.track_id in self.handled_track_ids or target.track_id in self.failed_pickup_ids:
                continue
            revision = self.navigator.revision if self.navigator is not None else 0
            if self.deferred_pickup_ids.get(target.track_id) == revision:
                continue
            if pickup_rejection(target, self.collector_cfg) is not None:
                continue
            if (self.coverage_index < len(self.coverage_path)-1 and
                    np.linalg.norm(target.position_robot[:2] - pose[:2]) > self.cfg.divert_max_distance_m):
                continue
            bearing = np.arctan2(target.position_robot[1] - pose[1], target.position_robot[0] - pose[0])
            if target.orientation_rad is not None and target.size_m[0] > 1.4 * target.size_m[1]:
                headings = [target.orientation_rad, target.orientation_rad + np.pi]
            else:
                headings = [bearing + i * np.pi / 4 for i in range(8)]
            options = []
            for theta in headings:
                direction = np.array([np.cos(theta), np.sin(theta)])
                pickup = target.position_robot[:2] - self.collector_cfg.pickup_offset_m * direction
                approach = pickup - self.collector_cfg.approach_standoff_m * direction
                # A feasible pickup also needs room to rotate with its payload.
                # Reject a pose before closing the claw if it cannot start a
                # return route with the conservative carrying envelope.
                if self.navigator is not None and not self.navigator.point_clear(
                        pickup, self._navigation_radius(extended=True), ignore_id=target.track_id):
                    continue
                route = (self.navigator.route(pose[:2], approach, self._navigation_radius())
                         if self.navigator is not None else np.array([pose[:2], approach]))
                if route is None:
                    continue
                if self.navigator is not None:
                    # Intentional tool contact is allowed only for the selected
                    # item; the chassis must still clear that item completely.
                    if not self.navigator.segment_clear(approach, pickup, self._navigation_radius(),
                                                        ignore_id=target.track_id):
                        continue
                    if not self.navigator.segment_clear(approach, pickup, self.geometry.chassis_radius_m):
                        continue
                cost = np.linalg.norm(np.diff(route, axis=0), axis=1).sum()
                options.append((cost, tuple(pickup) + (theta,), tuple(approach) + (theta,)))
            if options:
                _, self._pickup_pose, self._approach_pose = min(options, key=lambda x: x[0])
                self.current_target = target
                self.pickup_attempts = 1
                self._save_checkpoint(pose)
                self._transition(MissionState.APPROACHING_DEBRIS, "collect " + target.object_class)
                return True
            self.deferred_pickup_ids[target.track_id] = revision
            self._record(f"deferred constrained litter track {target.track_id}")
        return False

    def _run_coverage(self, pose):
        if self.collector_cfg.enabled and self._select_pickup(pose):
            return 0.0, 0.0
        if self.navigator is None:
            self._save_checkpoint(pose)
            return super()._run_coverage(pose)
        target, index = self._closest_pending_stain(pose)
        if target is not None:
            self.pending_stains.pop(index)
            self.current_target = target
            self._save_checkpoint(pose)
            self._transition(MissionState.DIVERTING, "divert to stain")
            return 0.0, 0.0
        if self.coverage_index >= len(self.coverage_path) - 1:
            self._transition(MissionState.DONE, "coverage complete")
            return 0.0, 0.0
        goal = self.coverage_path[self.coverage_index + 1]
        v, w, arrived = self._navigate(pose, goal, self.geometry.max_linear_vel)
        if arrived:
            self.coverage_index += 1
            self._route_key = None
        elif self.last_navigation_error:
            self._transition(MissionState.FAULT, self.last_navigation_error)
        return v, w

    def _run_diverting(self, pose):
        if self.navigator is None:
            return super()._run_diverting(pose)
        v, w, arrived = self._navigate(pose, self.current_target.position_robot[:2],
                                      self.geometry.nav_linear_vel)
        if arrived:
            self._start_scrub()
            self._record("start mop scrub")
        elif self.last_navigation_error:
            self._record("stain currently unreachable")
            self.current_target = None
            self._transition(MissionState.RESUMING_COVERAGE, "resume after unreachable stain")
        return v, w

    def _run_scrubbing(self, pose):
        if self.navigator is None:
            return super()._run_scrubbing(pose)
        if self.scrub_index < len(self.scrub_path):
            v, w, arrived = self._navigate(pose, self.scrub_path[self.scrub_index],
                                          self.geometry.scrub_linear_vel, tolerance=0.018)
            if arrived:
                self.scrub_index += 1
                self._route_key = None
            elif self.last_navigation_error:
                self._transition(MissionState.FAULT, "scrub path blocked")
            return v, w
        if self.current_target.track_id is not None:
            self.handled_track_ids.add(self.current_target.track_id)
        self.current_target, self.scrub_path = None, None
        self._transition(MissionState.RESUMING_COVERAGE, "stain scrub finished")
        return 0.0, 0.0

    def _run_approach(self, pose):
        v, w, arrived = self._navigate(pose, self._approach_pose[:2], self.geometry.nav_linear_vel,
                                      heading=self._approach_pose[2])
        if arrived:
            self._transition(MissionState.ALIGNING_PICKUP, "align palm with litter")
        elif self.last_navigation_error or self.time_s - self._state_since > self.collector_cfg.navigation_timeout_s:
            self._pickup_failed("approach unavailable or timed out")
        return v, w

    def _run_alignment(self, pose):
        seen = self.current_target.observed_at_s
        if seen is not None and self.time_s - seen > 1.0:
            # Allow the consecutive-frame gate to reacquire a target after
            # turning toward it. No collector motion without fresh evidence.
            if self.time_s - self._state_since > 1.0:
                self._pickup_failed("lost visual confirmation during alignment")
            return 0.0, 0.0
        if self.navigator is not None and (not self.navigator.segment_clear(
                pose[:2], self._pickup_pose[:2], self._navigation_radius(),
                ignore_id=self.current_target.track_id) or not self.navigator.segment_clear(
                pose[:2], self._pickup_pose[:2], self.geometry.chassis_radius_m)):
            self.deferred_pickup_ids[self.current_target.track_id] = self.navigator.revision
            self.current_target = None
            self.collector_command = CollectorCommand.STOW
            self._transition(MissionState.RESUMING_COVERAGE, "pickup deferred: approach corridor blocked")
            return 0.0, 0.0
        v, w, arrived = self._navigate(pose, self._pickup_pose[:2], self.collector_cfg.pickup_speed_m_s,
                                      tolerance=self.collector_cfg.alignment_tolerance_m,
                                      heading=self._pickup_pose[2], ignore_id=self.current_target.track_id)
        if arrived:
            if self.navigator is not None and self.navigator.route(
                    pose[:2], self.home_pose[:2], self._navigation_radius(True),
                    ignore_id=self.current_target.track_id) is None:
                self.deferred_pickup_ids[self.current_target.track_id] = self.navigator.revision
                self.current_target = None
                self.collector_command = CollectorCommand.STOW
                self._transition(MissionState.RESUMING_COVERAGE,
                                 "pickup deferred: return clearance unavailable")
                return 0.0, 0.0
            self._pickup_phase = self.collector_command = CollectorCommand.LOWER
            self._phase_since = self.time_s
            self._transition(MissionState.COLLECTING, f"pickup attempt {self.pickup_attempts}")
        elif self.last_navigation_error or self.time_s - self._state_since > self.collector_cfg.navigation_timeout_s:
            self._pickup_failed("alignment unavailable or timed out")
        return v, w

    def _run_collection(self, pose):
        feedback = self._feedback
        if feedback is not None and feedback.command == self._pickup_phase and feedback.completed:
            if self._pickup_phase == CollectorCommand.LOWER:
                self._pickup_phase = self.collector_command = CollectorCommand.CLOSE
            elif self._pickup_phase == CollectorCommand.CLOSE:
                self._pickup_phase = self.collector_command = CollectorCommand.LIFT
            else:
                self.collector_command = CollectorCommand.HOLD
                self._transition(MissionState.VERIFYING_PICKUP, "verify lifted payload")
            self._phase_since = self.time_s
        elif self.time_s - self._phase_since > self.collector_cfg.operation_timeout_s:
            self._pickup_failed("actuator completion timed out")
        return 0.0, 0.0

    def _run_verification(self, pose):
        evidence = None if self._feedback is None else self._feedback.payload_present
        if evidence is True:
            self.carried_target = self.current_target
            self.handled_track_ids.add(self.current_target.track_id)
            self._transition(MissionState.RETURNING_HOME, "pickup verified; return home")
        elif evidence is False or self.time_s - self._state_since > self.collector_cfg.verification_timeout_s:
            self._pickup_failed("payload verification failed")
        return 0.0, 0.0

    def _pickup_failed(self, reason):
        self.collector_command = CollectorCommand.STOW
        if self.pickup_attempts < self.collector_cfg.max_pickup_attempts:
            self.pickup_attempts += 1
            self._transition(MissionState.RECOVERING_PICKUP, reason + "; retry")
        else:
            self.failed_pickup_ids.add(self.current_target.track_id)
            self.current_target = None
            self._transition(MissionState.RESUMING_COVERAGE, reason + "; attempts exhausted")

    def _stowed(self):
        if not self.collector_cfg.enabled:
            return True
        return (self._feedback is not None and self._feedback.command == CollectorCommand.STOW
                and self._feedback.completed and self._feedback.payload_present is False)

    def _run_recovery(self, pose):
        if self._stowed():
            self._transition(MissionState.APPROACHING_DEBRIS, "collector reset for retry")
        elif self.time_s - self._state_since > self.collector_cfg.operation_timeout_s:
            self._transition(MissionState.FAULT, "collector did not stow after failed pickup")
        return 0.0, 0.0

    def _run_return(self, pose):
        self.collector_command = CollectorCommand.HOLD
        if self._feedback is not None and self._feedback.payload_present is False:
            self._transition(MissionState.FAULT, "payload lost during transport")
            return 0.0, 0.0
        if (self._last_payload_evidence is None or self.time_s - self._last_payload_evidence
                > self.collector_cfg.verification_timeout_s):
            self._transition(MissionState.FAULT, "payload feedback lost during transport")
            return 0.0, 0.0
        v, w, arrived = self._navigate(pose, self.home_pose[:2], self.collector_cfg.return_speed_m_s,
                                      extended=True, heading=self.home_pose[2])
        if arrived:
            self._transition(MissionState.WAITING_FOR_UNLOAD, "home reached; waiting for operator")
        elif self.last_navigation_error or self.time_s - self._state_since > self.collector_cfg.navigation_timeout_s:
            self._transition(MissionState.FAULT, "return route unavailable or timed out")
        return v, w

    def _run_resume(self, pose):
        self.collector_command = CollectorCommand.STOW
        if not self._stowed():
            if self.time_s - self._state_since > self.collector_cfg.operation_timeout_s:
                self._transition(MissionState.FAULT, "collector not empty and stowed")
            return 0.0, 0.0
        if self._resume_pose is None:
            self._transition(MissionState.COVERAGE, "resume coverage")
            return 0.0, 0.0
        v, w, arrived = self._navigate(pose, self._resume_pose[:2], self.geometry.nav_linear_vel,
                                      heading=self._resume_pose[2])
        if arrived:
            self.coverage_index = self.resume_index
            self._transition(MissionState.COVERAGE, "saved coverage checkpoint reached")
        elif self.last_navigation_error or self.time_s - self._state_since > self.collector_cfg.navigation_timeout_s:
            self._transition(MissionState.FAULT, "coverage checkpoint unavailable")
        return v, w
