#!/usr/bin/env python3
"""
cleaning_robot_node.py
-----------------------
ROS 2 (rclpy) node wiring perception.py + motion_planning.py to real
topics. Two callback groups are used deliberately:

  - Perception (image callbacks) is CPU-bound and can run at a lower rate
    (5-10 Hz is plenty -- the floor doesn't move fast) in a
    MutuallyExclusiveCallbackGroup so RANSAC/segmentation calls don't
    overlap each other.
  - Control (the cmd_vel timer) needs to run fast and smoothly (20 Hz+)
    in its own MutuallyExclusiveCallbackGroup so a slow perception frame
    never causes a control-loop stall. Use a MultiThreadedExecutor so the
    two groups actually run concurrently.

Subscriptions:
  /camera/color/image_raw          (sensor_msgs/Image, rgb8)
  /camera/aligned_depth_to_color/image_raw  (sensor_msgs/Image, 32FC1 meters)
  /camera/color/camera_info        (sensor_msgs/CameraInfo)  -- one-shot
  /odom                             (nav_msgs/Odometry)

Publications:
  /cmd_vel                          (geometry_msgs/Twist)

This file assumes cv_bridge + OpenCV for image conversion and RGB->Lab,
and tf2 for the camera extrinsic (recommended over the hardcoded fallback
in config.CameraExtrinsics). Swap message_filters' sync policy or topic
names to match your driver (e.g. realsense2_camera's default topics).
"""

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

import message_filters
from cv_bridge import CvBridge
import cv2
import tf_transformations  # pip/apt: ros-<distro>-tf-transformations

from cleaning_robot.config import RobotConfig
from cleaning_robot.perception import (
    GroundPlaneEstimator, DebrisDetector, FloorColorModel, StainDetector,
    CoordinateTransformer, DetectionTracker, backproject_depth,
)
from cleaning_robot.motion_planning import MissionPlanner, MissionState, generate_boustrophedon


class CleaningRobotNode(Node):
    def __init__(self):
        super().__init__('cleaning_robot_node')
        self.cfg = RobotConfig()
        self.bridge = CvBridge()

        # ---- Perception state ----
        self.plane_estimator = GroundPlaneEstimator(self.cfg.perception, self.cfg.intrinsics)
        self.debris_detector = DebrisDetector(self.cfg.perception, self.cfg.intrinsics)
        self.color_model = FloorColorModel(self.cfg.perception)
        self.stain_detector = StainDetector(self.cfg.perception, self.color_model)
        self.transformer = CoordinateTransformer(self.cfg.intrinsics, self.cfg.extrinsics)
        self.tracker = DetectionTracker(self.cfg.perception)
        self._calibrated_background = False
        self._latest_confirmed_stains = []
        self._latest_confirmed_debris = []

        # ---- Motion planning state ----
        # In production, replace this fixed rectangle with the actual
        # free-space boundary from your occupancy map / SLAM output.
        coverage_path = generate_boustrophedon(
            0.0, 3.0, 0.0, 2.0, self.cfg.geometry.radius_m,
            self.cfg.planning.coverage_lane_overlap)
        self.mission = MissionPlanner(self.cfg.planning, self.cfg.geometry, coverage_path)
        self.current_pose = (0.0, 0.0, 0.0)

        # ---- Callback groups: perception vs control run concurrently ----
        perception_group = MutuallyExclusiveCallbackGroup()
        control_group = MutuallyExclusiveCallbackGroup()

        # ---- Subscriptions ----
        self.create_subscription(
            CameraInfo, '/camera/color/camera_info', self._on_camera_info, 1,
            callback_group=perception_group)

        rgb_sub = message_filters.Subscriber(
            self, Image, '/camera/color/image_raw', qos_profile=qos_profile_sensor_data)
        depth_sub = message_filters.Subscriber(
            self, Image, '/camera/aligned_depth_to_color/image_raw', qos_profile=qos_profile_sensor_data)
        self._sync = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub], queue_size=5, slop=0.03)
        self._sync.registerCallback(self._on_frame)

        self.create_subscription(
            Odometry, '/odom', self._on_odom, 20, callback_group=control_group)

        # ---- Publisher + control timer ----
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_timer(1.0 / 20.0, self._control_tick, callback_group=control_group)

        self.get_logger().info('CleaningRobotNode started.')

    # ------------------------------------------------------------------
    # Perception callbacks
    # ------------------------------------------------------------------

    def _on_camera_info(self, msg: CameraInfo) -> None:
        # Overwrite the config defaults with the real, driver-reported
        # intrinsics -- never hardcode these in production.
        self.cfg.intrinsics.fx = msg.k[0]
        self.cfg.intrinsics.fy = msg.k[4]
        self.cfg.intrinsics.cx = msg.k[2]
        self.cfg.intrinsics.cy = msg.k[5]
        self.cfg.intrinsics.width = msg.width
        self.cfg.intrinsics.height = msg.height

    def _on_frame(self, rgb_msg: Image, depth_msg: Image) -> None:
        rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='rgb8')
        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='32FC1')
        # Some drivers publish depth in mm as 16UC1 -- convert if needed:
        # depth = depth.astype(np.float32) / 1000.0

        points_cam, valid = backproject_depth(depth, self.cfg.intrinsics)
        if not self.plane_estimator.update(points_cam, valid):
            return  # not enough floor visible yet this frame
        height_map = self.plane_estimator.height_map(points_cam)

        # --- Debris (3D) ---
        debris_dets = self.debris_detector.detect(points_cam, valid, height_map)
        debris_dets = self.transformer.debris_to_robot_frame(debris_dets, points_cam)

        # --- Stains (2D color) ---
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        # OpenCV's 8-bit Lab packing: L in [0,255]->[0,100], a/b in [0,255]->[-128,127].
        lab[..., 0] *= (100.0 / 255.0)
        lab[..., 1] -= 128.0
        lab[..., 2] -= 128.0
        floor_mask = self.stain_detector.floor_plane_mask(height_map, valid)

        if not self._calibrated_background:
            # Simple startup calibration: assume the first few seconds see
            # mostly clean floor. Swap for an explicit calibration routine
            # (e.g. triggered by a service call after visual confirmation)
            # in a real deployment.
            sample = lab[floor_mask]
            if sample.shape[0] > 500:
                self.color_model.initialize(sample)
                self._calibrated_background = True
                self.get_logger().info('Floor color model calibrated.')
            return

        stain_dets = self.stain_detector.detect(lab, floor_mask)
        stain_dets = self.transformer.stain_to_robot_frame(
            stain_dets, self.plane_estimator.normal, self.plane_estimator.d)
        self.stain_detector.update_background_model(lab, floor_mask)

        # --- Temporal confirmation (separately per kind) ---
        confirmed_debris = self.tracker.update(debris_dets)
        confirmed_stains = self.tracker.update(stain_dets)
        self._latest_confirmed_debris = [d for d in confirmed_debris if d.kind == 'debris']
        self._latest_confirmed_stains = [d for d in confirmed_stains if d.kind == 'stain']

    # ------------------------------------------------------------------
    # Odometry + control
    # ------------------------------------------------------------------

    def _on_odom(self, msg: Odometry) -> None:
        q = msg.pose.pose.orientation
        _, _, yaw = tf_transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.current_pose = (msg.pose.pose.position.x, msg.pose.pose.position.y, yaw)

    def _control_tick(self) -> None:
        linear, angular = self.mission.update(
            self.current_pose, self._latest_confirmed_stains, self._latest_confirmed_debris)

        cmd = Twist()
        cmd.linear.x = linear
        cmd.angular.z = angular
        self.cmd_pub.publish(cmd)

        if self.mission.state == MissionState.DONE:
            self.get_logger().info('Coverage complete.', once=True)


def main(args=None):
    rclpy.init(args=args)
    node = CleaningRobotNode()
    from rclpy.executors import MultiThreadedExecutor
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
