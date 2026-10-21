"""Paired, seeded efficiency measurements; never changes the controller.

This measures the labelled geometric simulator, not physical detection/grasp
accuracy. All trial outcomes, including faults and time limits, are retained.
"""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import platform
import sys

import numpy as np

from .motion_planning import MissionState
from .simulation import CleaningSimulation, simulation_config
from .stress import ScenarioSpec, _random_scene, scene_snapshot


def make_trial(seed, profile, config=None):
    """New layout RNG independent of the actuator/perception RNG streams."""
    if profile not in ("nominal", "disturbance", "boundary"):
        raise ValueError("unknown benchmark profile")
    cfg = deepcopy(config) if config is not None else simulation_config()
    cfg.collector.enabled = True
    cfg.simulation.seed = seed
    litter, stains = _random_scene(cfg, seed, near_wall=profile == "boundary")
    rng = np.random.default_rng(seed + 2000000)
    # Change stain extent without changing its position or class count.
    from .simulation import Stain
    stains = [Stain(s.object_id, s.position, float(rng.uniform(.045, .085))) for s in stains]
    challenge = profile != "nominal"
    assumptions = {
        "pickup_success_probability": float(rng.uniform(.80, .95)) if challenge else 1.,
        "observation_drop_probability": float(rng.uniform(.05, .20)) if challenge else 0.,
        "one_alignment_blackout_s": float(rng.uniform(.30, .80)) if challenge else 0.,
    }
    return cfg, litter, stains, assumptions


def run_arm(cfg, litter, stains, assumptions, *, enabled, max_steps, unload_delay_s):
    """Same initial platform, coverage path, mock sensor and scene in both arms.

    Collection-disabled arm retains the stowed tool and uses the planner's
    existing disabled-collector navigation mode. It is an ablation, not a
    reproduction of an older hardware robot. A scheduled alignment blackout
    applies only to an arm that enters alignment; both arms have the same
    stochastic observation-drop parameter, not identical observations.
    """
    cfg = deepcopy(cfg)
    cfg.collector.enabled = enabled
    sim = CleaningSimulation(cfg, litter=litter, stains=stains)
    # Scene placement also uses cfg.simulation.seed. Separate streams avoid
    # correlating object size/position draws with pickup/drop outcomes.
    sim.rng = np.random.default_rng(np.random.SeedSequence([cfg.simulation.seed, 1]))
    sim.collector.rng = sim.rng
    sim.perception_rng = np.random.default_rng(np.random.SeedSequence([cfg.simulation.seed, 2]))
    sim.collector.success_probability = assumptions["pickup_success_probability"]
    sim.sensor_drop_probability = assumptions["observation_drop_probability"]
    initial = scene_snapshot(sim, ScenarioSpec("random", cfg.simulation.seed))
    durations = Counter()
    wait = 0.
    verified = first_attempt_verified = 0
    waiting_since = None
    blackout_triggered = False
    for steps in range(1, max_steps + 1):
        before = sim.mission.state
        if (not blackout_triggered and assumptions["one_alignment_blackout_s"]
                and before == MissionState.ALIGNING_PICKUP):
            sim.blackout_sensor(assumptions["one_alignment_blackout_s"])
            blackout_triggered = True
        old_events = len(sim.mission.events)
        sim.step()
        dt = cfg.simulation.time_step_s
        durations[before.name] += dt
        if before == MissionState.WAITING_FOR_UNLOAD:
            wait += dt
        for event in sim.mission.events[old_events:]:
            if event["message"] == "pickup verified; return home":
                verified += 1
                first_attempt_verified += sim.mission.pickup_attempts == 1
        after = sim.mission.state
        if after == MissionState.WAITING_FOR_UNLOAD:
            if waiting_since is None:
                waiting_since = sim.time_s
            if sim.time_s - waiting_since + 1e-9 >= unload_delay_s:
                sim.unload()
                waiting_since = None
        if after in (MissionState.DONE, MissionState.FAULT):
            break
    summary = sim.summary()
    home_ok = all(a["position_error_m"] <= cfg.collector.home_tolerance_m + 1e-9
                  and a["heading_error_rad"] <= cfg.collector.heading_tolerance_rad + 1e-9
                  for a in sim.home_arrivals)
    all_clean = all(s.cleaned for s in sim.stains)
    all_delivered = all(x.status == "delivered" for x in sim.litter)
    clean_complete = summary["state"] == "DONE" and all_clean and sim.safety_stops == 0
    full_success = clean_complete and (all_delivered and home_ok if enabled else True)
    fraction = sum(np.count_nonzero(~s.remaining) for s in sim.stains)
    sample_count = sum(len(s.remaining) for s in sim.stains)
    timed_out = summary["state"] not in ("DONE", "FAULT")
    issues = []
    if summary["state"] == "FAULT":
        issues.append(sim.mission.events[-1]["message"])
    if timed_out:
        issues.append("mission time budget exhausted")
    missing = [x.object_id for x in sim.litter if x.status != "delivered"]
    if enabled and missing:
        issues.append(f"undelivered litter IDs {missing}")
    if not all_clean:
        issues.append("stains not fully cleaned")
    if not home_ok:
        issues.append("home arrival outside configured tolerance")
    return {
        "collector_enabled": enabled, "full_success": bool(full_success),
        "cleaning_complete": bool(clean_complete), "timed_out": timed_out,
        "issues": issues, "steps": steps, "summary": summary,
        "elapsed_s": sim.time_s, "active_s": sim.time_s - wait,
        "unload_wait_s": wait, "path_m": sim.path_length_m,
        "state_time_s": dict(durations), "verified_pickups": verified,
        "first_attempt_verified": int(first_attempt_verified),
        "stain_samples_removed": int(fraction), "stain_samples_total": sample_count,
        "stain_sample_removal_fraction": fraction / sample_count if sample_count else 1.,
        "home_arrivals_in_tolerance": home_ok,
        "blackout_triggered": blackout_triggered,
        "initial_scene": initial, "events": sim.mission.events,
    }


def wilson_interval(successes, total):
    if total == 0:
        return None
    z = 1.959963984540054
    p, z2 = successes / total, z * z
    center = (p + z2 / (2 * total)) / (1 + z2 / total)
    half = z * math.sqrt(p * (1-p) / total + z2 / (4 * total**2)) / (1 + z2 / total)
    return [max(0., center-half), min(1., center+half)]


def bootstrap_ratio(numerators, denominators, *, seed=501, samples=5000):
    """Resample whole scenes, preserving dependence of objects within scenes."""
    n, d = np.asarray(numerators, float), np.asarray(denominators, float)
    if len(n) == 0 or d.sum() <= 0:
        return None
    picks = np.random.default_rng(seed).integers(0, len(n), (samples, len(n)))
    sums = d[picks].sum(axis=1)
    ratios = n[picks].sum(axis=1)[sums > 0] / sums[sums > 0]
    return {"value": float(n.sum()/d.sum()),
            "ci95_scene_bootstrap": np.quantile(ratios, [.025, .975]).tolist()}


def distribution(values):
    values = np.asarray(values, float)
    if not len(values):
        return None
    return {"n": len(values), "mean": float(values.mean()),
            "median": float(np.median(values)), "p90": float(np.quantile(values, .9)),
            "min": float(values.min()), "max": float(values.max())}


def aggregate(rows):
    if not rows:
        raise ValueError("cannot aggregate zero trials")
    e, b = [r["enhanced"] for r in rows], [r["baseline"] for r in rows]
    deliveries = [a["summary"]["delivered"] for a in e]
    objects = [len(a["summary"]["objects"]) for a in e]
    complete_pairs = [r for r in rows if r["enhanced"]["cleaning_complete"]
                      and r["baseline"]["cleaning_complete"]]
    per_class = {}
    for label in ("plastic_bottle", "wrapper", "paper_cup", "paper_ball"):
        per_class[label] = bootstrap_ratio(
            [sum(x["class"] == label and x["status"] == "delivered"
                 for x in a["summary"]["objects"]) for a in e],
            [sum(x["class"] == label for x in a["summary"]["objects"]) for a in e])
    arrivals = [x for a in e for x in a["summary"]["home_arrivals"]]
    successes = sum(a["full_success"] for a in e)
    return {
        "trials": len(rows), "objects": sum(objects), "delivered": sum(deliveries),
        "mission_successes": successes, "mission_success_rate": successes/len(rows),
        "mission_success_ci95_wilson": wilson_interval(successes, len(rows)),
        "delivery_rate": bootstrap_ratio(deliveries, objects),
        "by_class": per_class,
        "stains_cleaned": sum(a["summary"]["stains_cleaned"] for a in e),
        "stains_total": sum(len(a["initial_scene"]["stains"]) for a in e),
        "stain_sample_removal": bootstrap_ratio([a["stain_samples_removed"] for a in e],
                                               [a["stain_samples_total"] for a in e]),
        "pickup_attempts": sum(a["summary"]["pickup_attempts"] for a in e),
        "verified_pickups": sum(a["verified_pickups"] for a in e),
        "first_attempt_verified": sum(a["first_attempt_verified"] for a in e),
        "pickup_attempt_success": bootstrap_ratio([a["verified_pickups"] for a in e],
                          [a["summary"]["pickup_attempts"] for a in e]),
        "objects_per_active_minute": bootstrap_ratio(deliveries, [a["active_s"]/60 for a in e]),
        "objects_per_elapsed_minute": bootstrap_ratio(deliveries, [a["elapsed_s"]/60 for a in e]),
        "observed_time_all_runs_s": distribution([a["elapsed_s"] for a in e]),
        "completed_time_s": distribution([a["elapsed_s"] for a in e if a["full_success"]]),
        "completed_active_s": distribution([a["active_s"] for a in e if a["full_success"]]),
        "completed_path_m": distribution([a["path_m"] for a in e if a["full_success"]]),
        "enhanced_faults": sum(a["summary"]["state"] == "FAULT" for a in e),
        "enhanced_timeouts": sum(a["timed_out"] for a in e),
        "enhanced_safety_stops": sum(a["summary"]["safety_stops"] for a in e),
        "baseline_cleaning_successes": sum(a["cleaning_complete"] for a in b),
        "baseline_stains_cleaned": sum(a["summary"]["stains_cleaned"] for a in b),
        "baseline_faults": sum(a["summary"]["state"] == "FAULT" for a in b),
        "baseline_timeouts": sum(a["timed_out"] for a in b),
        "baseline_safety_stops": sum(a["summary"]["safety_stops"] for a in b),
        "baseline_delivered": sum(a["summary"]["delivered"] for a in b),
        "paired_completed_count": len(complete_pairs),
        "paired_extra_elapsed_s": distribution([r["enhanced"]["elapsed_s"]-r["baseline"]["elapsed_s"]
                                                for r in complete_pairs]),
        "paired_extra_active_s": distribution([r["enhanced"]["active_s"]-r["baseline"]["active_s"]
                                               for r in complete_pairs]),
        "paired_extra_path_m": distribution([r["enhanced"]["path_m"]-r["baseline"]["path_m"]
                                             for r in complete_pairs]),
        "paired_elapsed_ratio": bootstrap_ratio([r["enhanced"]["elapsed_s"] for r in complete_pairs],
                                                [r["baseline"]["elapsed_s"] for r in complete_pairs]),
        "paired_extra_seconds_per_delivery": bootstrap_ratio(
            [r["enhanced"]["elapsed_s"]-r["baseline"]["elapsed_s"] for r in complete_pairs],
            [r["enhanced"]["summary"]["delivered"] for r in complete_pairs]),
        "home_arrivals": len(arrivals),
        "max_home_error_mm": max((a["position_error_m"]*1000 for a in arrivals), default=None),
        "max_home_heading_error_deg": max((np.rad2deg(a["heading_error_rad"]) for a in arrivals), default=None),
        "out_of_tolerance_arrivals": sum(a["position_error_m"] > r["config"]["collector"]["home_tolerance_m"]+1e-9
                   or a["heading_error_rad"] > r["config"]["collector"]["heading_tolerance_rad"]+1e-9
                   for r in rows for a in r["enhanced"]["summary"]["home_arrivals"]),
        "state_time_s": dict(sum((Counter(a["state_time_s"]) for a in e), Counter())),
        "failure_reasons": dict(Counter(issue for a in e for issue in a["issues"])),
    }


def make_protocol(trials, seed, max_steps, unload_delay_s, config=None):
    if trials < 1 or max_steps < 1 or seed < 0:
        raise ValueError("positive trials/steps and nonnegative seed required")
    if not np.isfinite(unload_delay_s) or unload_delay_s < 0:
        raise ValueError("unload delay must be finite and nonnegative")
    cfg = deepcopy(config) if config is not None else simulation_config()
    root = Path(__file__).resolve().parents[1]
    sources = ["cleaning_robot/benchmark.py", "efficiency_benchmark.py"]
    sources += [str(p.relative_to(root)).replace("\\", "/") for p in sorted((root/"cleaning_robot").glob("*.py"))
                if p.name != "benchmark.py"]
    hashes = {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in sources if (root/name).exists()}
    return {
        "schema_version": 1, "trials": trials, "paired_mission_runs": trials*2,
        "seed_start": seed, "seeds": list(range(seed, seed+trials)),
        "profile_schedule": ["nominal", "disturbance", "nominal", "boundary"],
        "max_steps_per_arm": max_steps, "unload_delay_s": unload_delay_s,
        "max_simulated_seconds_per_arm": max_steps*cfg.simulation.time_step_s,
        "config": json.loads(json.dumps(asdict(cfg))), "source_sha256": hashes,
        "python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
        "sampling": "One of each litter class, randomized position/size/yaw, two random stains. "
                    "Same four-object separated-layout sampler as stress.py; no scene resampling based on outcomes. "
                    "Boundary profile moves the cup centre to arena_width-0.17m. "
                    "Stain radii uniform 0.045-0.085m. Fixed configured room and home.",
        "assumptions": "Nominal: perfect mock semantic observations inside frustum and geometric pickup probability 1. "
                       "Other profiles: per-scene assumed pickup probability uniform 0.80-0.95, observation drop "
                       "probability uniform 0.05-0.20, one 0.30-0.80s blackout on first pickup alignment. "
                       "These are sensitivity inputs, not measured hardware probabilities.",
        "rng_streams": "Layout uses default_rng(seed); disturbance parameters use default_rng(seed+2000000); "
                       "actuator uses SeedSequence([seed,1]); observations use SeedSequence([seed,2]). "
                       "Arms share stream initialization but consume observations differently along their paths.",
        "success_definition": "DONE, all four litter objects physically removed at home by simulated operator, "
                              "all stains fully cleared, no collision-guard stops, every home pose within tolerance. "
                              "Baseline success requires DONE, all stains cleared and no guard stops.",
        "comparison": "Collection-disabled ablation of same simulated platform: stowed tool retained, identical "
                      "initial scene and generated coverage path, existing disabled-collector navigation mode. "
                      "Not a hardware or historical-software comparison. Both arms have the same drop parameter; "
                      "their observations differ because they follow different trajectories. Alignment blackout "
                      "only affects an arm entering pickup alignment.",
        "statistics": "Wilson 95% interval for scene mission success; 5000 seeded scene-cluster bootstrap draws "
                      "for ratio metrics. No hardware confidence inference. Paired time/distance overhead uses "
                      "only pairs where both cleaning missions finish, explicitly counted. Aborted/censored "
                      "times are not completion times. Every trial remains in success/delivery denominators. "
                      "Bootstrap intervals at 100% can collapse; use mission Wilson interval for uncertainty.",
        "limits": "Mock labels, exact poses and dimensions; ideal odometry; no rendered RGB-D, depth noise, "
                  "occlusion, real grasp/contact physics, motor energy or real dirt removal. Stain removal is "
                  "geometric sample coverage. Stops are collision guards, not measured physical collisions. "
                  "Manual unload delay is standardized; real operator availability is unmeasured. "
                  "No controller changes are made during this benchmark.",
    }


def run_benchmark(protocol, output_dir, progress=None):
    """Write protocol before executing; stream each finished pair to JSONL."""
    from .simulation import apply_config_overrides
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    if any((directory/name).exists() for name in ("protocol.json", "trials.jsonl", "results.json")):
        raise FileExistsError("use a fresh output directory to preserve benchmark evidence")
    (directory/"protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    cfg = simulation_config()
    apply_config_overrides(cfg, protocol["config"])
    rows = []
    with (directory/"trials.jsonl").open("w", encoding="utf-8") as stream:
        for index, seed in enumerate(protocol["seeds"]):
            profile = protocol["profile_schedule"][index % len(protocol["profile_schedule"]) ]
            trial_cfg, litter, stains, assumptions = make_trial(seed, profile, cfg)
            row = {"trial": index+1, "seed": seed, "profile": profile,
                   "config": json.loads(json.dumps(asdict(trial_cfg))), "assumptions": assumptions}
            for name, enabled in (("enhanced", True), ("baseline", False)):
                row[name] = run_arm(trial_cfg, litter, stains, assumptions, enabled=enabled,
                         max_steps=protocol["max_steps_per_arm"], unload_delay_s=protocol["unload_delay_s"])
            rows.append(row)
            stream.write(json.dumps(row)+"\n")
            stream.flush()
            if progress:
                progress(row)
    report = {"protocol": protocol, "aggregate": aggregate(rows),
              "profiles": {name: aggregate([r for r in rows if r["profile"] == name])
                           for name in dict.fromkeys(r["profile"] for r in rows)}, "results": rows}
    (directory/"results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
