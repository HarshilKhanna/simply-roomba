# Cleaning and retrieval efficiency: 100 randomized paired trials

For the implementation handoff and file-by-file changes, see
[update.md](../update.md). See [README.md](../README.md) for operating commands.
This document records the original benchmark; the 21 October 2026 documentation
update did not rerun or alter its measurements.

The geometric simulator delivered **383 of 400 litter objects to the starting
location for manual unloading: 95.75% collection efficiency**. It completed
every retrieval and stain-cleaning requirement in **90 of 100 scenes**.
These are simulation results. Physical D405 detection, grasp reliability,
localization and dirt removal have not been measured.

| Metric | Measured result |
|---|---:|
| Randomized scenes / total mission runs | 100 / 200 |
| Objects delivered / all initial litter | 383/400 = 95.75% |
| Collection rate, 95% scene-bootstrap interval | 92.50%-98.25% |
| Full mission success | 90/100 = 90% |
| Full success, 95% Wilson interval | 82.6%-94.5% |
| Stains fully covered | 196/200 = 98% |
| Verified pickups / close operations | 385/403 = 95.53% |
| Mean elapsed time, 90 full successful missions | 307.73s (5.13min) |
| Mean travel, 90 full successful missions | 35.31m |
| Throughput, all trials including faults | 0.77 deliveries/elapsed minute |
| Throughput excluding unloading wait | 0.88 deliveries/active minute |
| Enhanced-arm routing faults / timeouts / collision-guard stops | 5 / 0 / 0 |
| Home arrivals within configured tolerance | 383/383 |
| Maximum simulated home position / heading error | 24.99mm / 3.44 degrees |

Every incomplete mission remains in the collection and mission-success
denominators. Mean completion time and distance apply only to the 90 full
successful missions. Throughput uses the observed runtime of every trial;
it does not estimate how long failed missions would take to finish.

## Conditions tested

| Profile | Scenes | Delivered | Collection | Full missions |
|---|---:|---:|---:|---:|
| Nominal | 50 | 194/200 | 97% | 47/50 |
| Synthetic disturbances | 25 | 96/100 | 96% | 23/25 |
| Near-wall plus disturbances | 25 | 93/100 | 93% | 20/25 |

Each scene contains an empty plastic bottle, wrapper, paper cup and paper ball,
plus two stains. Positions, sizes, orientations and stain radii vary in a
fixed 3.2m by 2.4m room. Objects are separated rather than piled together.
Class-level delivery rates were 96% for bottles, 93% for wrappers, 96% for
cups and 98% for paper balls (100 initial objects per class).

Nominal mock pickup probability is 1. Disturbed scenes assume a per-scene
pickup probability sampled from 0.80-0.95, observation-drop probability from
0.05-0.20, and one alignment blackout lasting 0.30-0.80s. These are assumed
sensitivity inputs, not hardware measurements. The overall rate reflects
this chosen 50/25/25 mix, not an estimated distribution of real rooms.

Manual unloading is represented by a standardized ten-second acknowledgement
at each return. Actual operator response time is unknown. Home error is
against exact simulated pose and does not establish localization accuracy.
Stain cleaning means geometric coverage of sampled stain disks.

## Comparison and cost

Every scene also ran with collection disabled on the same simulated platform,
including its stowed tool. This is an ablation, not a comparison with the
historical hardware robot. The collection-disabled arm leaves litter in place
and its route or footprint can be blocked by that litter.

The disabled arm completed 43/100 cleaning missions and covered 169/200 stains;
57 runs faulted, including 24 collision-guard stops. It delivered zero objects
by design. These results should not be presented as the measured performance
of the original stain-cleaning robot.

Time comparisons use **40 pairs where both arms completed cleaning**:

- Mean added elapsed time: **163.50s** per scene.
- Mean added active time: **123.75s**, excluding unloading.
- Mean added travel: **13.17m** per scene.
- Added elapsed time per delivered object: **41.13s**.

This subset can favor easier scenes, and the disabled arm performs less work.
The enhancement adds retrieval capability at a time/travel cost. The experiment
does not demonstrate faster stain cleaning or lower electrical energy use.

## What failed

Ten scenes did not meet every requirement: five stopped on route failures,
and five reached `DONE` with one object uncollected. Seventeen initial objects
were not delivered. Failure details and event logs are retained in the full
report. The nominal profile also has failures, so the limitation cannot be
attributed entirely to injected pickup uncertainty.

The next engineering priorities are return-route robustness as new obstacles
are observed, handling coverage waypoints blocked by retained litter, and
investigating wrappers that remain uncollected. Improvements should be evaluated
on a new seed set after this recorded baseline is preserved.

## Evidence and reproduction

The final run uses seeds **2026201800-2026201899**. Its protocol, settings and
source hashes were recorded before execution. The controller stayed unchanged
throughout the batch. An independent audit passed **2,822 consistency checks**
and confirmed 100 unique layouts and matching source hashes. Enhanced trial 2
(successful, disturbed) and trial 4 (failed, boundary) reproduced every saved
measurement exactly. The 45 automated checks and five original pipeline checks
also passed.

A preliminary 19-pair instrumentation run exposed shared RNG initialization
between layout and pickup draws. It was stopped and preserved separately.
The final run uses fresh seeds and separate actuator/perception streams; none
of the preliminary outcomes are pooled into these rates.

- [Presentation charts (PDF)](../artifacts/efficiency/random100_independent_20261018/benchmark_summary.pdf)
- [Full report and failure table](../artifacts/efficiency/random100_independent_20261018/report.md)
- [Raw results and initial scenes](../artifacts/efficiency/random100_independent_20261018/results.json)
- [Protocol and source hashes](../artifacts/efficiency/random100_independent_20261018/protocol.json)
- [Independent audit](../artifacts/efficiency/random100_independent_20261018/audit.json)
- [Exact-replay verification](../artifacts/efficiency/random100_independent_20261018/replay_verification.json)

The evidence directory is intentionally excluded by `.gitignore`. A source-only
clone retains this Markdown summary but not the raw data or figures; share
`artifacts/efficiency/random100_independent_20261018/` separately with a partner
who needs to open these links, audit the original data or replay a saved trial.
Keep the measured source revision available: the audit checks current source
hashes against the saved protocol and will reject changed source files.

```powershell
python -B efficiency_benchmark.py --trials 100 --seed 2026201800 --output-dir artifacts/efficiency/NEW_DIRECTORY
python -B audit_benchmark.py artifacts/efficiency/random100_independent_20261018/results.json
python -B replay_benchmark.py artifacts/efficiency/random100_independent_20261018/results.json --trial 4
```

`NEW_DIRECTORY` must be a fresh path. The benchmark CLI returns 0 when execution
and report generation finish, even if individual missions fail. Read the saved
outcomes; an execution exit code is not a mission-success score. The behavior
stress suite in `stress_test.py` has different expectations and must not be
substituted for this collection-efficiency benchmark.

No rendered RGB-D images, depth noise, occlusion, physical contact dynamics,
motor current or battery energy are modeled. The experiment supports a claim
about the simulated mission controller under documented assumptions, and does
not establish physical robot efficiency.

> Across 100 seeded randomized simulated scenes containing 400 lightweight
> litter objects, the mop-plus-collector controller delivered 383 objects to its
> starting location (95.75%) and completed all retrieval and stain-cleaning tasks
> in 90% of scenes. Physical performance remains to be measured.
