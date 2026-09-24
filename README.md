# Cleaning Robot Perception + Motion Planning

Pure-Python core (`cleaning_robot/perception.py`, `cleaning_robot/motion_planning.py`,
`cleaning_robot/config.py`) with no ROS dependency, so it's unit-testable on
a laptop. `cleaning_robot/cleaning_robot_node.py` is the thin ROS 2 (rclpy)
wrapper that feeds it real topics.

## Layout
```
cleaning_robot/
  config.py              # calibration + tuning dataclasses (edit this first)
  perception.py          # plane RANSAC, debris clustering, stain detection, transforms
  motion_planning.py      # spiral/boustrophedon generators, pure pursuit, state machine
  cleaning_robot_node.py  # ROS 2 node (rclpy, cv_bridge, message_filters, tf_transformations)
test_pipeline.py          # synthetic-data validation, no ROS/camera required
```

## Run the tests
```
pip install numpy scipy
python3 test_pipeline.py
```
This builds synthetic RGB-D frames (dense, via analytic ray/plane
intersection) with known debris bumps and stain patches, and checks:
- fitted plane normal vs. ground truth,
- debris cluster localization vs. ground truth,
- stain localization + Mahalanobis detection,
- Monte-Carlo proof that the spiral scrub path's mop-footprint union
  covers >99.5% of the target stain disk,
- a full boustrophedon coverage pass under pure-pursuit control,
- the coverage -> divert -> scrub -> resume state machine end to end.

## Run on the robot
Requires ROS 2, `cv_bridge`, `message_filters`, `tf_transformations`, and a
depth-aligned RGB-D driver (e.g. `realsense2_camera`). Adjust topic names in
`cleaning_robot_node.py` to match your driver, and replace
`config.CameraExtrinsics`'s hardcoded pitch/height with a URDF + tf2 lookup
before trusting it for anything beyond bench testing.

```
ros2 run cleaning_robot cleaning_robot_node
```

## Key design decisions worth knowing before you tune this
- **Debris is treated as an obstacle, not a scrub target.** The only
  actuator is a passive perimeter mop, which can smear/push solid debris
  rather than clean it. If you add suction or a brush later, route
  `DebrisDetector` output to a pickup behavior instead of the avoidance bias.
- **Stains are localized by ray/plane intersection, not raw depth**,
  because liquids are frequently IR-transparent or specular and give
  noisy/absent depth returns exactly where you need them.
- **Spiral pitch uses `R*(1-overlap)`, not `2R*(1-overlap)`.** A
  continuous spiral only revisits a given angle once per full revolution
  (unlike independently-spannable boustrophedon lanes), so the coverage
  math is stricter -- see the docstring in `motion_planning.py` and the
  Monte-Carlo test for the derivation. Getting this wrong leaves a
  thin uncovered wedge right at the stain boundary.
- **RANSAC refit is amortized** (`plane_refit_every_n_frames`, default
  15) since the camera-to-floor geometry is rigid; per-frame work is just
  a cached-plane dot product, not a fresh RANSAC every frame.
- **Detections require N consecutive frames before triggering a divert**
  (`min_consecutive_detections`, default 3) so single-frame noise (a
  glare, a shadow) can't send the robot chasing nothing.
- **The reactive debris-avoidance bias is a lightweight stand-in**, not a
  costmap-based local planner. If Nav2 is already in your stack, feed
  debris detections into its costmap as obstacles instead.

## Things to tune for your hardware
All in `config.py`:
- `CameraExtrinsics.pitch_deg / height_m / forward_offset_m` -- measure or
  calibrate against your actual mount.
- `PerceptionConfig.debris_min_height_m` -- raise above your depth
  sensor's actual noise floor (5mm is aggressive for many consumer RGB-D
  sensors at 1-2m range; check your camera's datasheet).
- `PerceptionConfig.mahalanobis_threshold` -- tune against real floor
  material/lighting; log the mahalanobis map during bring-up to pick a
  threshold with an acceptable false-positive rate.
- `PlanningConfig.spiral_overlap / coverage_lane_overlap` -- increase if
  you see gaps in practice due to odometry drift; the values here already
  include a safety margin over the bare mathematical minimum.
