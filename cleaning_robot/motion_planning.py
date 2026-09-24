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

from .config import PlanningConfig, RobotGeometry
from .perception import Detection


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
    DONE = auto()


@dataclass
class MissionPlanner:
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
        known_ids = {id(t) for t in self.pending_stains}
        if self.current_target is not None:
            known_ids.add(id(self.current_target))
        for stain in confirmed_stains:
            if stain.track_id in self.handled_track_ids:
                continue
            if id(stain) in known_ids:
                continue
            self.pending_stains.append(stain)

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
