# Partner handoff: mop and litter-collection enhancement

Last reviewed: **21 October 2026**. This documents the changes since the original
stain-cleaning code, including the visual simulator and recorded efficiency
benchmark. The date is a documentation review date, not a new benchmark run.

## Current state

The project now has a hardware-free prototype that combines the existing mop
mission with a concave claw/scoop collector. The intended camera is the
**Intel RealSense D405**. Supported mock litter classes are empty plastic
bottles, wrappers, paper cups and paper balls.

The robot carries **one item at a time back to its recorded starting pose**,
waits there for **manual unloading**, and resumes the interrupted cleaning
route. There is no automatic dumping mechanism or collection tray requirement.
The stain detector and mop/scrub planning remain part of the project.

Collection is implemented and tested in simulation. The ROS node still uses
the stain-cleaning/debris-avoidance configuration; a real claw motor driver,
payload sensor and hardware collection integration have not been implemented.

Start with [README.md](README.md) for operating instructions and
[the recorded benchmark](docs/benchmark_results_20261018.md) for results.

## 1. Collector mission and navigation

The public `MissionPlanner` now supports this sequence alongside coverage and
stain scrubbing:

```text
COVERAGE -> APPROACHING_DEBRIS -> ALIGNING_PICKUP -> COLLECTING
         -> VERIFYING_PICKUP -> RETURNING_HOME -> WAITING_FOR_UNLOAD
         -> RESUMING_COVERAGE -> COVERAGE
```

- Selection requires a supported litter class, finite dimensions and a payload
  estimate inside the configured limits. Unknown or unsuitable objects remain
  obstacles. The simulated scene supplies these labels and estimates.
- Pickup aligns the palm with the object; elongated objects use their
  orientation when choosing a pickup pose.
- The collector command interface is `STOW`, `LOWER`, `CLOSE`, `LIFT`, `HOLD`.
  Feedback reports command completion, payload presence and faults.
- Pickup verification requires payload feedback. A completed timer or motor
  command alone does not establish that an object was collected.
- Failed pickups enter recovery and retry, with two attempts by default.
  Exhausted targets stay bounded rather than being retried indefinitely.
- The planner records the interrupted base pose and coverage segment. After
  unloading, it returns to that checkpoint before continuing coverage.
- Manual acknowledgement is accepted only in `WAITING_FOR_UNLOAD`. The GUI's
  `U` control models the operator removing the object and confirming unloading.
- `FAULT` stops the mission on collector faults, lost payload/telemetry during
  transport or unavailable return routes. Pickup completion/verification
  timeouts use the bounded recovery path; failure to stow during recovery faults.

The new bounded-arena A* navigator uses observed obstacles and separate
stowed/carrying clearances. Payload size contributes to the carrying envelope;
the mop radius separately controls cleaning-lane spacing. Constrained pickups
are deferred and reconsidered when the obstacle map changes. Return clearance
is checked before closing, but later observations can still invalidate a route.

Two geometry corrections are included: the coverage generator includes the
final boundary lane, and final pickup alignment permits intentional **tool**
contact with the selected object while retaining chassis clearance. The latter
resolved long bottles being rejected by the conservative tool envelope.

`RobotConfig().collector.enabled` defaults to **False** for existing callers.
`simulation_config()` explicitly enables collection. Simply switching this flag
on in a hardware caller does not connect motors or sensors.

## 2. Prototype dimensions and D405 assumptions

Values are editable in [simulation_config.json](simulation_config.json) and the
dataclasses in [config.py](cleaning_robot/config.py). JSON overrides are checked
for unknown fields, invalid types and invalid geometry.

| Parameter | Current simulation assumption |
|---|---:|
| Chassis diameter / mop diameter | 34cm / 34cm, configured separately |
| Collector opening / palm depth | 22cm / 18cm |
| Palm centre ahead of base during pickup / carrying | 32cm / 22cm |
| Illustrative raised carrying height | 8cm |
| Maximum accepted object length / width / height | 24cm / 18cm / 16cm |
| Payload limit / pickup attempts | 120g / 2 |
| Home position / heading tolerance | 25mm / 0.06rad (about 3.44 degrees) |
| D405 mount height / downward pitch / forward offset | 22cm / 45 degrees / 8cm |
| Mock optical-depth interval / field of view | 7-50cm / 87 by 58 degrees |
| Arena / starting pose | 3.2m by 2.4m / `(0.5, 0.5, 0)` |
| Simulation control step / perception period | 0.05s / 0.10s |

The carrying pose assumes a lift linkage that folds the palm back toward the
body. It is a geometric assumption, not a validated mechanical design. There
are no CAD files, motor/torque selections or verified linkage/joint limits.

The D405 mock applies the configured frustum and camera transform to labelled
objects; it does not render RGB-D frames. The original `CameraExtrinsics`
fallback remains 12cm high with 25-degree downward pitch and is distinct from
`D405Config.mount`. The ROS node uses the original config transform until a
calibrated replacement is connected; the simulator mount is not applied to ROS
automatically. Actual intrinsics, mounting, depth behavior and visibility need
measurement. A camera cannot directly measure an object's mass.

## 3. Perception and ROS corrections

`Detection` now carries explicit coordinate-frame metadata, optional object
class, dimensions, orientation, estimated mass and observation time. Its old
`position_robot` field name is retained for compatibility; inspect
`coordinate_frame` to determine what those coordinates mean.

- `detection_to_planning_frame()` converts base-relative observations into a
  fixed planning/odometry frame without transforming fixed observations twice.
- The tracker processes stains and debris together **once per sensor frame**.
  A track can receive at most one confirmation hit per frame, and missed frames
  reset consecutive confirmation. Simulator tracks retain static-object IDs
  across longer camera absences; the default tracker still prunes stale tracks.
- ROS depth conversion accepts standard `16UC1` millimetres and `32FC1` metres,
  and rejects invalid/nonpositive samples. Native SDK Z16 data must instead use
  the device-reported depth scale.
- Timestamped odometry is interpolated at image acquisition time, including
  heading wraparound. Confirmed targets are tracked in odometry coordinates.
- The ROS control loop publishes zero velocity before usable data arrives or
  when processed image/odometry updates become more than 0.5s old. Image
  processing also waits for camera intrinsics and a usable acquisition pose.

These helpers have synthetic tests. ROS callbacks, driver/topic compatibility
and physical motion have not been runtime-tested here. The geometric debris
detector produces unclassified obstacle proposals; no trained litter classifier
was added. A real semantic/eligibility adapter is still needed for collection.

## 4. Visual simulator and fault testing

`visual_sim.py` now opens a native OpenCV window by default. It uses the same
public mission planner as the core, with a moving robot, stains, litter,
collector animation, payload, nominal camera footprint, mop trail, status and
event history. JSON-only execution is explicit through `--headless`.

The window has mouse controls and keyboard shortcuts: pause (`Space`), manual
unload (`U`), single step (`.`), playback speed (`+`), simulated automatic
unloading (`A`), seeded random scene (`N`), reset (`R`), spawn litter/stain
(`D`/`S`), fail next pickup (`F`), camera blackout (`B`), payload drop (`X`),
camera overlay (`C`), screenshot (`P`) and stress tests (`T`). `Tab`/`Enter`
operate focused controls. The Windows window fits the desktop work area, and
`launch_simulator.cmd` provides a double-click launcher.

Stress tests have a separate results view with progress, filters, pagination,
per-case deliveries/state/home error, cancellation, report saving and row-based
replay. Cases cover baseline/random layouts, retries, near-wall objects,
observation drops, transient blackout, a false positive, blocked home, payload
loss and exhausted pickups. The runner advances in bounded chunks so the UI
can repaint during tests.

A passing stress case means its **expected behavior** occurred. For example,
a blocked-home test passes on the intended stop. This pass rate must not be
presented as a collection percentage. Reports retain delivered/missed objects
and individual failed-case snapshots. The simulator's collision guard prevents
motion/extension into detected geometric contact and faults; it is not a
measurement of physical collision probability.

## 5. Recorded efficiency benchmark

A separate benchmark ran **100 randomized paired scenes: 200 mission runs**,
with collection enabled and disabled on matched initial scenes. Each scene has
four litter classes and two stains. The final seeds are **2026201800-2026201899**.

| Measurement | Recorded collection-enabled result |
|---|---:|
| Delivered and unloaded at origin | **383/400 = 95.75%** |
| Full mission success | **90/100 = 90%** |
| Full success, 95% Wilson interval | 82.6%-94.5% |
| Stains fully covered | **196/200 = 98%** |
| Verified pickups / close operations | 385/403 = 95.53% |
| Mean full successful mission time / travel | 307.73s / 35.31m |
| Routing faults / timeouts / collision-guard stops | 5 / 0 / 0 |

There were 50 nominal scenes (97% collection), 25 disturbed scenes (96%) and
25 near-wall scenes with disturbances (93%). Disturbed cases use assumed
pickup probabilities of 0.80-0.95, observation-drop probabilities of 0.05-0.20,
and one 0.30-0.80s alignment blackout. These are sensitivity inputs, not measured
hardware probabilities. Unloading uses a standardized ten-second operator wait.

All incomplete trials remain in the headline denominators. Five runs stopped
on route failures and five reached `DONE` with an uncollected item. Means for
completed full missions exclude those ten incomplete runs. The comparison arm
is collection disabled on the same simulated platform, not the historical
physical mop robot. Added time/distance estimates use only the 40 pairs where
both cleaning missions completed: mean +163.50s elapsed and +13.17m travel.

The benchmark saves a protocol/configuration and source hashes before running,
streamed JSONL measurements, final JSON, Markdown and PNG/PDF plots. It uses
separate RNG streams and retains failed outcomes. An early 19-pair run exposed
shared layout/pickup RNG initialization; it was stopped and preserved separately.
The final run uses fresh seeds and excludes those preliminary outcomes.

An independent audit passed **2,822 checks**, including 100 unique layouts,
matched scenes, time/count accounting, retry limits, home tolerances and source
hashes. A successful disturbed trial (2) and failed boundary trial (4) reproduced
every saved measurement exactly. The automated suite recorded **45 passing
tests**, plus the **five original pipeline checks**.

The full evidence and limitations are in
[docs/benchmark_results_20261018.md](docs/benchmark_results_20261018.md). These
results measure simulated mission behavior. They do not establish physical
D405 accuracy, claw grasp reliability, actual stain removal or energy efficiency.

## 6. Changed and added files

| File | Change |
|---|---|
| `cleaning_robot/config.py` | Added collector, D405 and simulation configuration; separated chassis and mop radii. |
| `cleaning_robot/collector.py` | New command/feedback contract, eligibility policy, palm positions and swept-envelope helpers. |
| `cleaning_robot/navigation.py` | New bounded-arena A* navigation and obstacle-clearance checks. |
| `cleaning_robot/motion_planning.py` | Extended public planner with retrieval/recovery/home/unload/resume states, checkpoints, events and clearance fixes. |
| `cleaning_robot/perception.py` | Frame metadata/transforms, depth normalization, pose interpolation and tracker fixes. |
| `cleaning_robot/cleaning_robot_node.py` | ROS acquisition-time transforms, combined tracking and stale-input stop behavior. |
| `cleaning_robot/__init__.py` | Exported new configuration, collector contracts and frame helper. |
| `cleaning_robot/simulation.py` | New deterministic world, D405 visibility mock, simulated collector, stain coverage and fault fixtures. |
| `cleaning_robot/stress.py` | New seeded scenarios, fault schedules, snapshots, expectation evaluation and cooperative runner. |
| `visual_sim.py` | Reworked visual application and explicit headless CLI using the shared planner. |
| `launch_simulator.cmd` | New Windows GUI launcher. |
| `simulation_config.json` | New editable prototype geometry/camera/world overrides. |
| `stress_test.py` | New CLI stress runner with JSON/CSV/PNG outputs and replay snapshots. |
| `cleaning_robot/benchmark.py` | New paired efficiency runner, independent RNG streams, measurements, intervals and provenance. |
| `efficiency_benchmark.py` | New benchmark CLI and Markdown/PNG/PDF reporting. |
| `audit_benchmark.py` | New independent saved-data/source-hash consistency audit. |
| `replay_benchmark.py` | New individual-arm replay and exact saved-measurement comparison. |
| `tests/test_retrieval.py` | New 22 mission, geometry, tracking and perception regression tests. |
| `tests/test_visual_stress.py` | New 17 interaction, scenario and regression tests. |
| `tests/test_benchmark.py` | New six benchmark accounting, timeout, interval and reproducibility tests. |
| `requirements.txt` | Updated optional RealSense driver comment for D405; plotting still needs separate Matplotlib installation. |
| `.gitignore` | Added exclusions for generated artifacts, bytecode and test caches. |
| `README.md` | Updated operating instructions, architecture, assumptions, benchmark, repository map and ROS setup guidance. |
| `docs/benchmark_results_20261018.md` | Added recorded results, failure analysis, evidence links and reproduction commands. |
| `update.md` | This partner handoff. |

The original `live_stain_test.py`, `test_stain_detector.py` and `test_pipeline.py`
remain available. Generated `__pycache__` changes are runtime by-products rather
than source changes; some cache files were already tracked before `.gitignore`.

## 7. Run and reproduce

Run from the repository root with a desktop-capable Python/OpenCV environment:

```powershell
python -m pip install -r requirements.txt
python -m pip install matplotlib
python -B visual_sim.py --gui
```

Wait for a home arrival, remove the simulated payload with `U`, and observe the
return to the saved cleaning checkpoint. No camera or ROS is required for this
demo. Auto-unloading is an explicit simulation option, not a hardware dump.

```powershell
# Existing pipeline and new regression suite
python -B test_pipeline.py
python -B -m unittest discover -s tests -v

# Complete a headless mission with simulated operator acknowledgements
python -B visual_sim.py --headless --auto-unload --report artifacts/mission.json

# Behavior/fault expectations; choose a new output folder for a new run
python -B stress_test.py --runs 100 --output-dir artifacts/stress/partner_run

# Reproduce the recorded efficiency seed set in a NEW directory
python -B efficiency_benchmark.py --trials 100 --seed 2026201800 --output-dir artifacts/efficiency/partner_reproduction

# Inspect the existing evidence without rerunning 200 missions
python -B audit_benchmark.py artifacts/efficiency/random100_independent_20261018/results.json
python -B replay_benchmark.py artifacts/efficiency/random100_independent_20261018/results.json --trial 4
```

The efficiency runner refuses to overwrite existing protocol/results files.
Its exit code 0 means execution/report generation completed, even if measured
missions failed; read the outcome fields. The stress CLI instead exits 1 when
an expectation fails. Benchmark replay exits 1 when saved measurements differ.
An audit compares the current source files with the original protocol hashes;
subsequent source edits can therefore invalidate that source-match check.

**Sharing with the partner:** source changes and new files are not committed
by this handoff. Send/commit the new modules, scripts, tests and docs together.
`artifacts/` is intentionally Git-ignored: share
`artifacts/efficiency/random100_independent_20261018/` separately if the partner
needs the original raw data, figures, protocol and audit. Links into that folder
will be unavailable on a source-only clone. The Markdown summary retains the
headline results when shared without these local artifacts.

## 8. Remaining work and priorities

1. Investigate the five route failures, return routes invalidated by new
   observations and coverage waypoints blocked by retained litter. Investigate
   uncollected wrappers and other missed targets using the saved trial data.
2. Preserve this baseline, make improvements, then evaluate on new held-out
   seeds rather than reporting performance only on the cases used for fixes.
3. Design and validate the claw/scoop, lift linkage, mount, floor clearance,
   motor torque and payload retention for the actual chassis.
4. Build a motor adapter that implements the command/feedback contract and
   supplies credible completion, payload-presence and fault signals.
5. Integrate calibrated D405 streams and a real litter detector/eligibility
   policy. Test wrappers and transparent/reflective bottles with recorded data.
6. Add the actual navigation/localization/obstacle-map integration needed to
   return to the real starting pose. The arena router is not SLAM or Nav2.
7. Exercise the ROS callbacks and physical collection lifecycle in an equipped
   ROS environment, including stale inputs, pickup failures and payload loss.
8. Measure real pickup/delivery rates, home error, elapsed time, operator wait,
   dirt removal and electrical energy before making physical efficiency claims.

This handoff changes documentation only; it does not modify the controller or
replace the recorded benchmark data.
