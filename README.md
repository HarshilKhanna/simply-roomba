# Autonomous Cleaning Robot - Perception & Motion Planning

A complete, high-performance perception and motion planning stack for an autonomous floor-cleaning robot. Features a **Dual-Layer Perception Engine** (Mahalanobis color-anomaly + spatial texture/gradient energy) and an **Opportunistic Motion Planner** (Boustrophedon coverage, dynamic target diversion, Archimedean spiral spot scrubbing, and path resumption).

Built with a pure-Python core (`NumPy`, `SciPy`, `OpenCV`) for instant, standalone testing on a laptop with a regular webcam or RGB-D camera, paired with a thin **ROS 2** wrapper (`cleaning_robot_node.py`) for physical robot deployment.

---

## Repository Layout

```text
cleaning_robot/
|-- cleaning_robot/
|   |-- __init__.py
|   |-- config.py              # Calibration parameters, camera extrinsics & tuning
|   |-- perception.py          # Plane-RANSAC, 3D debris clustering, Mahalanobis stain detector
|   |-- motion_planning.py     # Spiral/boustrophedon path generators, pure pursuit, state machine
|   \-- cleaning_robot_node.py # ROS 2 node (rclpy, cv_bridge, message_filters, tf2)
|
|-- live_stain_test.py         # Real-time dual-layer webcam test with live sliders & classification
|-- test_stain_detector.py     # 4-quadrant diagnostic dashboard (RGB, Heatmap, Raw Mask, Cleaned Mask)
|-- visual_sim.py              # Interactive 2D visual simulator proving the coverage & scrub state machine
|-- test_pipeline.py           # Automated unit tests on synthetic data (ground plane, debris, spiral coverage)
\-- README.md
```

---

## 1. Quick-Start Testing Tools (No ROS Required)

You can test every component of the system directly on your PC before deploying to a physical robot or ROS environment.

### A. Live Webcam Perception Test (`live_stain_test.py`)
Tests the **Dual-Layer Perception Engine** directly using your laptop webcam or external camera.

```powershell
pip install -r requirements.txt
python live_stain_test.py
```

* **How it works:**
  1. Point the camera at a clean, empty patch of floor and press **`c`** to calibrate.
  2. The script learns both your floor's baseline color $(\mu, \Sigma)$ in CIE Lab space and the tile smoothness.
  3. Place objects on the floor:
     * **`[STAIN: SCRUB]` (Green Box):** Flat discoloration, coffee, mud, or liquids triggers the scrubbing pipeline.
     * **`[DEBRIS: AVOID]` (Orange Box):** 3D objects, phones, towels, or crumpled paper triggers obstacle avoidance.
* **On-Screen Sliders (Real-time Tuning):**
  * `Color Cutoff x10`: Lower to catch faint, low-contrast smudges; raise to reject floor discoloration.
  * `Texture Cutoff x10`: Controls sensitivity to fabric wrinkles, paper folds, and edges.
  * `Min Area (px)`: Filters out thin tile grout lines and digital camera noise (default 1200px).
* **Controls:** `[c]` Calibrate | `[t]` Toggle Texture Heatmap | `[s]` Screenshot | `[q]` Quit.

---

### B. 4-Quadrant Diagnostic Tool (`test_stain_detector.py`)
A comprehensive visual analytics dashboard breaking down every stage of the computer vision pipeline into 4 side-by-side quadrants.

```powershell
python test_stain_detector.py                   # Uses webcam or synthetic fallback
python test_stain_detector.py --image floor.jpg # Test against a saved floor image
```

* **Quadrant 1:** Camera feed with centroid crosshairs and bounding radius.
* **Quadrant 2:** Continuous Mahalanobis Distance Heatmap ($D_M = \sqrt{(p-\mu)^T \Sigma^{-1} (p-\mu)}$).
* **Quadrant 3:** Raw binary thresholded candidate mask.
* **Quadrant 4:** Morphologically filtered mask ready for target tracking.

---

### C. 2D Visual Simulator (`visual_sim.py`)
Simulates the autonomous mop robot in a 2.4m x 1.7m arena with dynamic stain spawning and full state machine execution.

```powershell
python visual_sim.py
```

* **Controls:**
  * `[SPACE]` : Pause / Resume
  * `[S]`     : Spawn random stain on the floor
  * `[R]`     : Reset simulation
  * `[Q/ESC]` : Quit
* **Features:**
  * Real-time HUD showing robot pose, mission state (`COVERAGE`, `DIVERTING`, `SCRUBBING`), velocity ($v, \omega$), and clean % progress.
  * Robust Pure Pursuit path tracking with in-place pivot turns for lane changes.
  * Dynamic Archimedean spiral mop pattern covering the full stain area plus safety margin.
  * Wall collision avoidance margins.

---

### D. Automated Math & Pipeline Unit Tests (`test_pipeline.py`)
Validates the mathematical foundations against synthetic ground-truth data.

```powershell
python test_pipeline.py
```

Checks:
1. **Plane-RANSAC:** Fits floor plane equation $ax + by + cz + d = 0$ with normal alignment $\cos\theta > 0.98$.
2. **Debris Segmentation:** Verifies positive signed height thresholding ($5\text{mm} < h < 15\text{cm}$).
3. **Stain Localization:** Ray/plane intersection with known floor geometry.
4. **Monte-Carlo Coverage Proof:** Verifies $>99.5\%$ geometric mop footprint coverage over the target disk during spiral scrubs.
5. **Mission Planner State Machine:** End-to-end verification of `COVERAGE` -> `DIVERTING` -> `SCRUBBING` -> `RESUME`.

---

## 2. Core Architecture & Design Decisions

### Dual-Layer Perception: Why Color Alone Is Not Enough
* **The Camouflage Problem:** A white towel or grey sock on light tile has almost zero color difference (both have near-zero chroma in CIE Lab). A pure color detector is blind to them.
* **The Solution:** 
  * **Layer 1 (Color Anomaly):** Flags color shifts against the clean floor baseline $\mathcal{N}(\mu, \Sigma)$ in CIE Lab space.
  * **Layer 2 (Spatial Gradient Energy / Depth):** Measures surface wrinkles, creases, and physical height. Smooth tile has near-zero gradient energy; crumpled fabric or solid objects have high texture density.

### Why Debris is Treated as an Obstacle (Not Scrubbed)
* The robot's primary cleaning tool is a wet perimeter mop.
* Driving a wet mop over solid debris (paper balls, socks, cables) smears paper pulp or jams the drive motors.
* **Stains (Flat, $|h| < 5\text{mm}$):** Scrubbed via Archimedean spiral.
* **Debris (Raised, $h > 10\text{mm}$):** Avoided via dynamic costmap keep-out zones (or vacuumed if hardware supports suction).

### Stains Localized by Ray/Plane Intersection (Not Raw Depth)
* Liquid puddles and wet stains are frequently IR-transparent or specular, causing consumer depth cameras (RealSense/OAK-D) to return zero or invalid depth right at the spill.
* Our system intersects the 2D pixel ray from camera intrinsics $K$ with the **rigidly estimated ground plane** to compute the metric world coordinate $(X, Y)$ without needing per-pixel depth returns on the liquid itself.

### Camera Mounting Geometry
* **Tilt Angle:** $25^\circ$ downward pitch from horizontal (not straight down).
* **Mounting Height:** $\sim 12\text{cm}$ above the floor.
* **Why:** Avoids the near-field blind spot of RGB-D sensors ($<20\text{cm}$ minimum range) while providing a **0.3m to 1.8m forward lookahead** so the robot can brake and steer smoothly.

---

## 3. Physical Robot Deployment (ROS 2)

### Dependencies
* ROS 2 (Humble / Iron / Jazzy)
* `rclpy`, `sensor_msgs`, `geometry_msgs`, `nav_msgs`
* `cv_bridge`, `message_filters`, `tf2_ros`
* RGB-D camera driver (e.g. `realsense2_camera`)

### Launch
```bash
ros2 run cleaning_robot cleaning_robot_node
```

The node subscribes to:
* `/camera/color/image_raw` (`sensor_msgs/Image`)
* `/camera/aligned_depth_to_color/image_raw` (`sensor_msgs/Image`)
* `/odom` (`nav_msgs/Odometry`)

And publishes:
* `/cmd_vel` (`geometry_msgs/Twist`)
* Dynamic stain & debris markers for RViz visualization.

---

## 4. Tuning Parameters (`config.py`)

* `CameraExtrinsics.pitch_deg`: Calibrate against your physical camera mount angle (default $25.0^\circ$).
* `PerceptionConfig.mahalanobis_threshold`: Color sensitivity cutoff (default $4.2\sigma$).
* `PerceptionConfig.debris_min_height_m`: Minimum height to consider as 3D obstacle (default $0.005\text{m}$ / $5\text{mm}$).
* `PlanningConfig.spiral_overlap`: Radial overlap between successive spiral scrub turns (default $0.35$).
* `PlanningConfig.coverage_lane_overlap`: Overlap between boustrophedon sweep passes (default $0.15$).
