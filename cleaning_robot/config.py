"""
config.py
---------
Central calibration / tuning parameters for the perception and motion
planning stack. Keeping these in one dataclass-driven module means every
other module takes a `RobotConfig` instance instead of hardcoded numbers,
so the whole stack is testable without ROS and re-tunable without touching
algorithm code.

All units: meters, radians, seconds unless noted.
"""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class CameraIntrinsics:
    """Pinhole intrinsics for the depth-aligned RGB stream (post rs-align or
    equivalent). Get these from /camera/color/camera_info at runtime rather
    than hardcoding in production -- see CleaningRobotNode._on_camera_info."""
    fx: float = 615.0
    fy: float = 615.0
    cx: float = 320.0
    cy: float = 240.0
    width: int = 640
    height: int = 480

    def as_matrix(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx],
                          [0, self.fy, self.cy],
                          [0, 0, 1.0]])


@dataclass
class CameraExtrinsics:
    """
    Static transform from the camera *optical* frame (x-right, y-down,
    z-forward -- REP-103 optical convention) to the robot *base_link* frame
    (x-forward, y-left, z-up).

    In production this should come from the URDF / tf2 static transform tree
    (see node code), not from hand-measured numbers -- but a closed-form
    fallback is provided here for bench testing without tf2.

    pitch_deg: downward tilt of the camera, measured from horizontal
               (positive = tilting down toward the floor). Matches the
               20-30 deg spec.
    height_m:  camera optical center height above the floor.
    forward_offset_m: camera position forward of the robot's geometric
               center (base_link origin), along robot +X.
    """
    pitch_deg: float = 25.0
    height_m: float = 0.12
    forward_offset_m: float = 0.08

    def transform_matrix(self) -> np.ndarray:
        """4x4 homogeneous transform T_base_cam: p_base = T_base_cam @ p_cam_h"""
        pitch = np.deg2rad(self.pitch_deg)

        # Step 1: optical frame (x-right, y-down, z-forward) -> a
        # "camera-body" frame (x-forward, y-left, z-up), i.e. undo the
        # optical axis convention.
        R_opt_to_body = np.array([
            [0, 0, 1],
            [-1, 0, 0],
            [0, -1, 0],
        ], dtype=float)

        # Step 2: apply downward pitch. A positive pitch tilts the camera's
        # forward axis down toward the floor, i.e. rotate about the body Y
        # (left) axis.
        c, s = np.cos(pitch), np.sin(pitch)
        R_pitch = np.array([
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c],
        ])

        R_base_cam = R_pitch @ R_opt_to_body
        t_base_cam = np.array([self.forward_offset_m, 0.0, self.height_m])

        T = np.eye(4)
        T[:3, :3] = R_base_cam
        T[:3, 3] = t_base_cam
        return T


@dataclass
class RobotGeometry:
    radius_m: float = 0.17               # mop cleaning radius (legacy name)
    chassis_radius_m: float = 0.17       # collision geometry, independent of mop
    max_linear_vel: float = 0.25         # m/s, coverage cruise speed
    max_angular_vel: float = 1.5         # rad/s
    scrub_linear_vel: float = 0.16       # m/s, slow speed during spot scrub
    nav_linear_vel: float = 0.18         # m/s, speed while diverting to a target


@dataclass
class PerceptionConfig:
    # --- Depth / plane fitting ---
    plane_ransac_iterations: int = 60
    plane_ransac_inlier_thresh_m: float = 0.004     # 4mm inlier band
    plane_refit_every_n_frames: int = 15            # amortize RANSAC cost
    plane_sample_points: int = 1500                 # subsample for RANSAC speed
    debris_min_height_m: float = 0.005              # 5mm, per spec
    debris_max_height_m: float = 0.15               # reject furniture/walls
    debris_min_pixel_area: int = 40                 # connected-component filter
    debris_max_range_m: float = 2.5
    debris_min_range_m: float = 0.25

    # --- RGB stain detection ---
    stain_flatness_band_m: float = 0.006            # |height| below this = "on floor"
    mahalanobis_threshold: float = 4.5              # chi-square-ish cutoff in Lab
    stain_min_pixel_area: int = 150
    stain_morph_kernel: int = 5
    color_model_ema_alpha: float = 0.02             # background update rate
    color_model_outlier_percentile: float = 92.0    # don't learn from likely-anomalous pixels

    # --- Temporal filtering / detection commit ---
    min_consecutive_detections: int = 3             # frames before a target is "confirmed"
    detection_match_radius_m: float = 0.08           # track-association gate


@dataclass
class PlanningConfig:
    coverage_lane_overlap: float = 0.15             # fraction of 2R overlapped between lanes
    spiral_overlap: float = 0.35                    # fraction of 2R overlapped between spiral turns
    spiral_margin_m: float = 0.03                   # extra radius past stain boundary
    divert_max_distance_m: float = 1.2              # don't divert for something far away
    pure_pursuit_lookahead_m: float = 0.20
    goal_tolerance_m: float = 0.03


@dataclass
class CollectorConfig:
    """Provisional scoop/claw dimensions, not a specification for hardware.

    Collection is opt-in. The ROS node continues to avoid debris unless a
    collector with real feedback is explicitly connected.
    """
    enabled: bool = False
    opening_width_m: float = 0.22
    palm_depth_m: float = 0.18
    pickup_offset_m: float = 0.32       # base centre -> palm centre
    carry_offset_m: float = 0.22        # lift pivots the palm back toward the base
    carry_height_m: float = 0.08        # illustrative raised carrying pose
    stowed_front_m: float = 0.20
    max_length_m: float = 0.24
    max_width_m: float = 0.18
    max_height_m: float = 0.16
    max_payload_kg: float = 0.12
    approach_standoff_m: float = 0.12
    alignment_tolerance_m: float = 0.012
    heading_tolerance_rad: float = 0.06
    pickup_speed_m_s: float = 0.05
    return_speed_m_s: float = 0.14
    home_tolerance_m: float = 0.025
    operation_timeout_s: float = 4.0
    verification_timeout_s: float = 2.0
    navigation_timeout_s: float = 90.0
    max_pickup_attempts: int = 2


@dataclass
class D405Config:
    """Nominal D405 workspace for simulation; verify the real stream profile."""
    min_depth_m: float = 0.07
    max_depth_m: float = 0.50
    horizontal_fov_deg: float = 87.0
    vertical_fov_deg: float = 58.0
    mount: CameraExtrinsics = field(default_factory=lambda: CameraExtrinsics(
        pitch_deg=45.0, height_m=0.22, forward_offset_m=0.08))


@dataclass
class SimulationConfig:
    arena_width_m: float = 3.2
    arena_height_m: float = 2.4
    home_pose: tuple = (0.50, 0.50, 0.0)
    time_step_s: float = 0.05
    perception_period_s: float = 0.10
    grid_resolution_m: float = 0.04
    obstacle_margin_m: float = 0.015
    lower_duration_s: float = 0.45
    close_duration_s: float = 0.55
    lift_duration_s: float = 0.55
    seed: int = 7


@dataclass
class RobotConfig:
    intrinsics: CameraIntrinsics = field(default_factory=CameraIntrinsics)
    extrinsics: CameraExtrinsics = field(default_factory=CameraExtrinsics)
    geometry: RobotGeometry = field(default_factory=RobotGeometry)
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    planning: PlanningConfig = field(default_factory=PlanningConfig)
    collector: CollectorConfig = field(default_factory=CollectorConfig)
    d405: D405Config = field(default_factory=D405Config)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
