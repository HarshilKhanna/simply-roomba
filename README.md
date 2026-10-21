# Autonomous Cleaning Robot - Perception & Motion Planning

Python perception and motion planning for a floor-cleaning robot: ground-plane
and stain detection, coverage, spot scrubbing, and a simulated litter-retrieval
enhancement using a concave collector and nominal RealSense D405 visibility.

Built with a pure-Python core (`NumPy`, `SciPy`, `OpenCV`) for instant, standalone testing on a laptop with a regular webcam or RGB-D camera, paired with a thin **ROS 2** wrapper (`cleaning_robot_node.py`) for physical robot deployment.

The hardware-free retrieval prototype adds a front concave collector for empty
plastic bottles, wrappers, paper cups and paper balls. It carries one item back
to the recorded starting pose, **waits for manual unloading**, and rejoins the
saved cleaning checkpoint. Stain scrubbing uses the same mission planner.

**Partner handoff:** [update.md](update.md) covers all changes, the file map,
setup commands, recorded results, current limitations and remaining
hardware integration. The [benchmark summary](docs/benchmark_results_20261018.md)
documents the measured simulation results. Collection is not yet connected to
real motors or a real litter classifier in the ROS node.

### Run the litter retrieval simulation

```powershell
pip install -r requirements.txt
python visual_sim.py --gui
```

On Windows you can also **double-click `launch_simulator.cmd`**. It explicitly
starts the graphical interface; no camera or ROS installation is needed.
The default command `python visual_sim.py` also opens the window. A JSON-only
run is the separate `--headless` mode; omit that flag to see the robot move.

The window has clickable buttons for playback, manual unloading, random scenes
and injected faults. Playback speed changes how many fixed physics steps run
between frames, not the robot's motion settings. Window size fits the Windows
desktop work area so the controls remain reachable above the taskbar.

| Key | Action |
| --- | --- |
| Space | Pause/resume |
| U | Manually unload at home and resume cleaning |
| . | Advance one physics step and stay paused |
| + | Cycle playback through 1x, 2x, 4x, 8x |
| A | Explicitly enable/disable simulated automatic unloading |
| N | Generate a new reproducible random scene |
| R | Reset the current scene with the same seed |
| D / S | Add litter / add a stain |
| F | Force the next pickup to fail |
| B | Inject a two-second mock-camera blackout |
| X | Drop the payload to test the transport stop |
| C | Toggle the nominal D405 floor workspace |
| P | Save the current view as a PNG in `artifacts/` |
| T | Start a visual stress-test batch |
| Tab / Enter | Focus and activate on-screen controls |
| Q / Esc | Close the window |

The robot holds its payload and remains stationary until U is pressed at home.
U has no effect during pickup or transport.

### Visual stress tests and exact replays

Click **Stress test** or press **T**. The live arena stays available in its own
tab. A cooperative runner advances cases in bounded chunks so the results view
keeps responding. The batch includes random sizes/positions/orientations,
near-wall litter, pickup retries and exhausted attempts, missed observations,
a transient camera blackout, a false positive, a reported home obstruction,
and transport payload loss. Tests use the same planner as the live arena.

The results show deliveries, final state, home-position error and each case's
expectation. A successful blocked-home/payload-loss test means **a planned safe
stop**, and a near-wall test permits safe rejection of detected, constrained
pickups. An unseen item does not count as a safe rejection.
These are labelled in the table. A passing batch does not mean every object
was collected. Actual delivered counts and missed IDs remain in the report.

Click **any result row** to replay its exact initial scene and scheduled faults.
Use **Failures only**, **Previous/Next**, **Stop batch** and **Save report** to
inspect results. Completed visual batches save JSON/CSV in
`artifacts/stress/latest/`; Save report also exports the visual results as PNG.
Failed cases export individual replay files.

```powershell
python visual_sim.py --gui --random-scene --seed 42
python visual_sim.py --gui --stress --stress-runs 100
python visual_sim.py --gui --replay artifacts/stress/latest/replay_022_28.json
```

For command-line evaluation with JSON, CSV, a PNG report and failed-case replays:

```powershell
python stress_test.py --runs 100 --output-dir artifacts/stress/latest
```

The stress runner exits 1 when an expectation fails. It preserves failures in
the report instead of suppressing them. Replays include the full config and
initial object geometry, not just a seed that might change with future code.

### Randomized efficiency benchmark (100 paired scenes)

The recorded [100-scene benchmark](docs/benchmark_results_20261018.md) delivered
383/400 objects (95.75%), completed 90/100 full missions and covered 196/200
stains. It includes all failures and documents the simulation assumptions.

The efficiency benchmark measures collection and cleaning outcomes separately
from fault-test pass rates. It runs **100 fresh scenes twice**: once with
collection enabled and once with collection disabled on the same simulated
platform. Each scene has one bottle, wrapper, paper cup and paper ball, plus
two stains. The default room is 3.2m by 2.4m. Positions, sizes, orientations
and stain radii vary; the separated-object sampler excludes cluttered piles.

```powershell
pip install matplotlib
python -B efficiency_benchmark.py --trials 100
```

The predefined mix is 50 nominal scenes, 25 disturbed scenes and 25 near-wall
scenes with disturbances. Nominal pickup probability is 1.0; disturbed cases
use assumed probabilities of 0.80-0.95 per geometric pickup, 0.05-0.20 per
observation drop and one 0.30-0.80s alignment blackout. These values are
sensitivity inputs, **not measured camera or gripper characteristics**.
Unloading is a standardized ten-second simulated operator acknowledgement;
active time excludes that wait. Change it with `--unload-delay`.

Reports go in a new timestamped directory under `artifacts/efficiency/`:

- `report.md`: presentation wording, assumptions, failures and denominator definitions.
- `benchmark_summary.png` and `.pdf`: plots suitable for sharing.
- `protocol.json`: seeds, configuration, assumptions and source hashes saved before execution.
- `results.json` and `trials.jsonl`: every paired outcome, initial scene and event log.

Collection efficiency is delivered objects divided by **all initial litter**.
Full mission success additionally requires all stains cleared, no guard stops
and home arrivals within tolerance. Failed or timed-out trials remain in these
denominators. Time/path overhead uses only pairs where both arms finish cleaning;
the report gives that subset size and flags the selection limitation. A
collection-disabled arm leaves litter in place and can encounter blocked routes;
it is an ablation of this simulator, not a measured historical hardware baseline.

The runner refuses to overwrite an existing benchmark directory. Recreate
reports from saved measurements without rerunning missions:

```powershell
python -B efficiency_benchmark.py --render-report artifacts/efficiency/RUN_DIRECTORY/results.json
python -B audit_benchmark.py artifacts/efficiency/RUN_DIRECTORY/results.json
python -B replay_benchmark.py artifacts/efficiency/RUN_DIRECTORY/results.json --trial 19
python -B -m unittest discover -s tests -v
```

These experiments measure mission logic under ideal pose and labelled geometric
perception. They do not establish physical D405 precision/recall, claw grasp
reliability, real dirt removal, electrical energy efficiency or generalization
to arbitrary rooms. Confidence intervals describe the sampled simulator scenes.
The audit independently checks counts, matched scenes, runtime accounting,
retry bounds, origin tolerances, source hashes and agreement between the
streamed journal and final report.

### Headless missions

To run without a window, stop at the first manual-unloading wait:

```powershell
python visual_sim.py --headless --report artifacts/manual_trip.json --output artifacts/home.png
```

For an automated complete mission, explicitly simulate operator acknowledgements:

```powershell
python visual_sim.py --headless --auto-unload --fail-first-pickup --report artifacts/mission.json
```

`--pickup-success-rate 0.0` exercises exhausted retries and obstacle-aware
coverage around uncollected litter. `--seed` controls random outcomes and spawned
objects. `--steps` bounds headless execution. Exit codes are 0 for completion or
a manual-unloading wait, 2 for a fault, and 3 for exhausting the step budget.
Reports include object outcomes, attempts, safety stops, and timestamped mission
events. Auto-unloading is off by default.

### Assumptions you can change

```powershell
python visual_sim.py --config simulation_config.json
```

The JSON file contains overrides; all available fields live in `config.py`.
Unknown keys, invalid types and invalid dimensions are rejected. Initial
assumptions include a 34 cm chassis, a 22 cm collector opening, an 18 cm palm,
a pickup palm centre 32 cm ahead of the base, a 120 g payload limit, and a nominal
D405 at 22 cm height with 45 degrees downward pitch. These are prototype
parameters, not measured hardware specifications. The simulated lift pivots the
palm back to a 22 cm carrying offset and an illustrative 8 cm carrying height;
the shortened horizontal envelope is included in navigation and drawing. This
assumes a suitable lift linkage; real joint limits and interference require
mechanical validation. The default scene has fixed
metric object positions; shrinking the arena must still leave those objects
inside it, or use a custom scene through the Python API.

The simulator applies the nominal D405 **7-50 cm optical-depth interval** and
87 x 58 degree frustum, including the camera's position and downward pitch.
The blue polygon is its intersection with the floor, not a long-range navigation
sensor. Object labels, dimensions and masses come from the simulated scene.
**There is no trained litter detector, rendered RGB-D image, stereo noise model,
occlusion model, or contact-physics simulation.** A flat wrapper can be selected
by the mock semantic sensor even though it is below the legacy height threshold.
Transparent-bottle depth and real wrapper pickup still require bench testing.
Nominal specifications: [RealSense D405](https://www.realsenseai.com/products/d405-series/).

The collector exposes `STOW`, `LOWER`, `CLOSE`, `LIFT` and `HOLD` commands with
completion, payload-presence and fault feedback. Successful motor timing alone
never verifies a pickup. Failed pickups and pickup completion/verification
timeouts retry up to the configured limit. Collector faults, failure to stow
during recovery, blocked return routes, transport payload loss and lost transport
payload telemetry stop the mission. An operator acknowledgement models
physical unloading; no automatic dumping is implemented.

An A* router checks the bounded arena and observed litter using separate
stowed/carrying clearances. The accepted payload length is included in the
carrying envelope. The mop radius still determines cleaning lane spacing.
Before closing, the planner checks the return clearance. During final alignment
only the selected item permits collector contact; the chassis still has to
clear it. Geometrically constrained targets are deferred and reconsidered when
the obstacle map changes, while exhausted physical pickup attempts stay bounded.
Coverage is inset conservatively from arena walls so the collector can rotate;
this is not a full-room coverage or SLAM implementation. Returning from a
retrieval rejoins the saved pose and segment instead of jumping to a later lane.

`RobotConfig().collector.enabled` remains **False** for existing hardware
callers; `simulation_config()` explicitly enables it for the simulator. The ROS
node continues its mop/debris-avoidance behavior. Real collection needs a motor
adapter, payload sensing, calibrated camera transforms, actual litter
classification and a suitable navigation/localization stack. The ROS wrapper
now normalizes standard `16UC1` millimetres / `32FC1` metres, tracks both kinds
once per frame in odometry coordinates using interpolated acquisition poses,
and holds motion when image/odometry updates are absent or stale. Native SDK
Z16 acquisition must use the device-reported depth scale instead.

### Verification without hardware

```powershell
python test_pipeline.py
python -m unittest discover -s tests -v
```

The original synthetic perception/coverage checks remain. Retrieval checks cover
all four litter types, stain cleaning, manual waiting, checkpoint resumption,
failed and exhausted pickups, missing actuator feedback, payload loss, frame
transforms, temporal association, D405 visibility, route clearance and config
validation. ROS callbacks and physical camera/actuator operation require a ROS
environment and hardware; the hardware-free tests do not exercise them.

---

## Repository Layout

```text
cleaning_robot/
|-- cleaning_robot/
|   |-- __init__.py
|   |-- config.py              # Calibration parameters, camera extrinsics & tuning
|   |-- perception.py          # Plane-RANSAC, 3D debris clustering, Mahalanobis stain detector
|   |-- motion_planning.py     # Spiral/boustrophedon path generators, pure pursuit, state machine
|   |-- collector.py           # Claw command/feedback contract and pickup eligibility
|   |-- navigation.py          # Bounded-arena obstacle-aware A* router
|   |-- simulation.py          # Deterministic world, D405 visibility and mock collector
|   |-- stress.py              # Seeded scenarios, fault schedules, replayable stress runner
|   |-- benchmark.py           # Paired efficiency measurements, intervals and source provenance
|   \-- cleaning_robot_node.py # ROS 2 node (rclpy, cv_bridge, message_filters, tf2)
|
|-- live_stain_test.py         # Real-time dual-layer webcam test with live sliders & classification
|-- test_stain_detector.py     # 4-quadrant diagnostic dashboard (RGB, Heatmap, Raw Mask, Cleaned Mask)
|-- visual_sim.py              # Shared-planner retrieval UI and headless CLI
|-- launch_simulator.cmd       # Windows double-click graphical launcher
|-- stress_test.py             # Reproducible batch evaluation and visual report export
|-- efficiency_benchmark.py    # Randomized paired benchmark and Markdown/PNG/PDF reports
|-- audit_benchmark.py         # Saved-data consistency and source-hash audit
|-- replay_benchmark.py        # Replay one benchmark arm and compare every measurement
|-- simulation_config.json     # Editable prototype geometry and camera assumptions
|-- tests/
|   |-- test_retrieval.py       # 22 hardware-free mission, geometry and perception checks
|   |-- test_visual_stress.py   # 17 interaction, scenario and regression checks
|   \-- test_benchmark.py       # 6 benchmark accounting and reproducibility checks
|-- test_pipeline.py           # Automated unit tests on synthetic data (ground plane, debris, spiral coverage)
|-- docs/benchmark_results_20261018.md # Recorded results and evidence links
|-- update.md                  # Partner handoff and complete change summary
|-- requirements.txt           # Core dependencies; install Matplotlib separately for plots
|-- .gitignore                 # Generated artifacts and cache exclusions
\-- README.md
```

The full suite currently contains 45 tests in `tests/`, plus five original
pipeline checks. Generated reports under `artifacts/` are ignored by Git.
Share the recorded benchmark folder separately when handing off its raw data
and figures; artifact links in the docs require those local files.

---

## 1. Quick-Start Testing Tools (No ROS Required)

The simulation, numerical helpers and mission logic can be tested without ROS.
ROS callbacks, physical sensing, motor feedback and real cleaning require their
respective runtime and hardware.

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
Simulates the autonomous mop/collector robot in a configurable 3.2m x 2.4m
arena. It uses `cleaning_robot.motion_planning.MissionPlanner` directly, including
the litter pickup, return-home, manual-unloading wait and coverage-resumption
states described above.

```powershell
python visual_sim.py
```

See the controls and headless examples at the top of this document. The HUD
shows mission state, payload, actuator command, pickup attempts, coverage
segment, safety stops and the nominal D405 floor workspace. The simulator's
collision guard checks the chassis and collector; a prevented collision is
reported as a fault rather than silently clipping the robot back inside walls.

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

### Stain scrubbing and litter handling
* The robot's primary cleaning tool is a wet perimeter mop.
* Driving a wet mop over solid debris (paper balls, socks, cables) smears paper pulp or jams the drive motors.
* **Stains (Flat, $|h| < 5\text{mm}$):** Scrubbed via Archimedean spiral.
* **Debris:** Hardware defaults use reactive steering avoidance. In the
  simulator, recognized litter within the collector limits generates a pickup
  mission; unknown, oversized and failed targets remain navigation obstacles.
  The existing ROS node does not implement a dynamic costmap.

### Stains Localized by Ray/Plane Intersection (Not Raw Depth)
* Liquid puddles and wet stains are frequently IR-transparent or specular, causing consumer depth cameras (RealSense/OAK-D) to return zero or invalid depth right at the spill.
* Our system intersects the 2D pixel ray from camera intrinsics $K$ with the **rigidly estimated ground plane** to compute the metric world coordinate $(X, Y)$ without needing per-pixel depth returns on the liquid itself.

### Camera Mounting Geometry
* The original bench-test fallback in `CameraExtrinsics` is 12 cm high with a
  25 degree downward pitch. It is not a calibrated D405 mounting transform.
* The retrieval simulation instead uses `D405Config.mount`: 22 cm high and
  45 degrees downward, adjustable in `simulation_config.json`.
* The D405's nominal ideal range is 7-50 cm; validate minimum distance for the
  actual stream profile. It is a close-range workspace sensor. Longer-range
  navigation and reliable home localization require additional integration.

---

## 3. Physical Robot Deployment (ROS 2)

### Dependencies
* A compatible ROS 2 Python environment (runtime compatibility is unverified here)
* `rclpy`, `sensor_msgs`, `geometry_msgs`, `nav_msgs`
* `cv_bridge`, `message_filters`, `tf_transformations`, plus the core Python dependencies
* RGB-D camera driver (e.g. `realsense2_camera`)

### Launch

After sourcing an appropriate ROS environment, the wrapper's Python entry
point can be invoked from the repository root:

```bash
python -B -m cleaning_robot.cleaning_robot_node
```

This checkout does not contain ROS package metadata or an installed console
entry point. `ros2 run cleaning_robot cleaning_robot_node` requires separately
packaging/installing it in a ROS workspace. Neither launch path has been tested
against physical hardware here. The node creates `RobotConfig()` with collection
disabled, a fixed rectangular coverage path and the original camera-transform
fallback; the simulator's D405 mount and JSON overrides are not applied to it.
Calibrate/replace the transform and integrate motor feedback, semantic detection
and navigation before enabling real collection.

The node subscribes to:
* `/camera/color/image_raw` (`sensor_msgs/Image`)
* `/camera/aligned_depth_to_color/image_raw` (`sensor_msgs/Image`, standard `16UC1` mm or `32FC1` m)
* `/camera/color/camera_info` (`sensor_msgs/CameraInfo`)
* `/odom` (`nav_msgs/Odometry`)

And publishes:
* `/cmd_vel` (`geometry_msgs/Twist`)

---

## 4. Tuning Parameters (`config.py`)

* `CameraExtrinsics.pitch_deg`: Calibrate against your physical camera mount angle (default $25.0^\circ$).
* `PerceptionConfig.mahalanobis_threshold`: Mahalanobis-distance cutoff (default `4.5`).
* `PerceptionConfig.debris_min_height_m`: Minimum height to consider as 3D obstacle (default $0.005\text{m}$ / $5\text{mm}$).
* `PlanningConfig.spiral_overlap`: Radial overlap between successive spiral scrub turns (default $0.35$).
* `PlanningConfig.coverage_lane_overlap`: Overlap between boustrophedon sweep passes (default $0.15$).
* `RobotGeometry.chassis_radius_m` / `radius_m`: Separate collision and mop radii (both default `0.17m`).
* `CollectorConfig`: Prototype opening, palm dimensions, pickup/carry offsets, payload limits, retries and home tolerances; hardware callers default to disabled.
* `D405Config.mount`: Simulator camera transform, separate from the original `CameraExtrinsics` fallback.
* `SimulationConfig`: Arena, starting pose, fixed control/perception periods, actuator timing and seed.

Use `--config simulation_config.json` with the simulator/stress/efficiency CLIs
to load strict overrides. The current ROS wrapper does not load that JSON file.
