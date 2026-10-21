"""Seeded scenarios, fault schedules and a cooperative stress-test runner.

The runner advances in bounded chunks so the desktop UI can keep repainting.
Every result contains its initial scene/config for an exact, shareable replay.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
import time

import numpy as np

from .collector import LITTER_CLASSES
from .motion_planning import MissionState
from .perception import Detection
from .simulation import CleaningSimulation, Litter, Stain, simulation_config


SCENARIOS = ("baseline", "random", "pickup_retry", "near_wall", "detection_noise",
             "sensor_blackout", "false_positive", "blocked_home", "payload_loss", "pickup_exhaustion")


@dataclass(frozen=True)
class ScenarioSpec:
    name: str = "baseline"
    seed: int = 7

    def __post_init__(self):
        if self.name not in SCENARIOS:
            raise ValueError(f"unknown scenario {self.name!r}")
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")


def _item_dict(item):
    return {"object_id": item.object_id, "object_class": item.object_class,
            "position": item.position.tolist(), "size_m": list(item.size_m),
            "mass_kg": item.mass_kg, "yaw_rad": item.yaw_rad}


def scene_snapshot(sim, spec):
    snapshot = {"scenario": asdict(spec), "config": asdict(sim.cfg),
            "litter": [_item_dict(item) for item in sim.litter],
            "stains": [{"object_id": item.object_id, "position": item.position.tolist(),
                        "radius_m": item.radius_m} for item in sim.stains]}
    # Canonical JSON shapes (including tuple -> list) make on-disk and live
    # snapshots identical and keep replay comparisons independent of Python.
    return json.loads(json.dumps(snapshot))


def _random_scene(cfg, seed, near_wall=False):
    rng = np.random.default_rng(seed)
    width, height = cfg.simulation.arena_width_m, cfg.simulation.arena_height_m
    home = np.array(cfg.simulation.home_pose[:2])
    definitions = [("plastic_bottle", (.19, .065, .065), .025),
                   ("wrapper", (.10, .07, .002), .002),
                   ("paper_cup", (.08, .08, .10), .012),
                   ("paper_ball", (.065, .065, .065), .008)]
    litter = []
    for index, (label, size, mass) in enumerate(definitions):
        factor = float(rng.uniform(.85, 1.08))
        size = tuple(np.array(size) * factor)
        for attempt in range(1500):
            point = rng.uniform([.8, .7], [width-.55, height-.50])
            if near_wall and index == 2:
                point[0] = width - .17
            if np.linalg.norm(point-home) < .75:
                continue
            if any(np.linalg.norm(point-item.position) < .65 for item in litter):
                continue
            litter.append(Litter(index+1, label, point, size, mass*factor**3,
                                float(rng.uniform(-np.pi, np.pi))))
            break
        else:
            raise ValueError("arena is too crowded to place the random scene; use a larger arena")
    stains = []
    for index in range(2):
        for _ in range(500):
            point = rng.uniform([.7, .65], [width-.65, height-.5])
            if any(np.linalg.norm(point-item.position) < .36 for item in litter):
                continue
            if any(np.linalg.norm(point-item.position) < .3 for item in stains):
                continue
            stains.append(Stain(101+index, point, .06))
            break
    return litter, stains


def make_scenario(spec=None, config=None, snapshot=None):
    spec = spec or ScenarioSpec()
    cfg = deepcopy(config) if config is not None else simulation_config()
    cfg.simulation.seed = spec.seed
    if snapshot is not None:
        # Config snapshots are loaded with the same strict override mechanism.
        from .simulation import apply_config_overrides
        apply_config_overrides(cfg, snapshot["config"])
        litter = [Litter(**{**item, "position": np.array(item["position"]),
                           "size_m": tuple(item["size_m"])}) for item in snapshot["litter"]]
        stains = [Stain(**{**item, "position": np.array(item["position"])}) for item in snapshot["stains"]]
        sim = CleaningSimulation(cfg, litter=litter, stains=stains)
    elif spec.name in ("baseline", "pickup_retry", "pickup_exhaustion", "false_positive"):
        if spec.name == "pickup_exhaustion":
            item = Litter(1, "plastic_bottle", np.array([.98, .5]), (.19, .065, .065), .025)
            sim = CleaningSimulation(cfg, litter=[item], stains=[])
        else:
            sim = CleaningSimulation(cfg)
    elif spec.name in ("blocked_home", "payload_loss"):
        # Start with an item far from home so transport faults are unambiguous.
        sim = CleaningSimulation(cfg, litter=[Litter(1, "paper_ball", np.array([1.8, .9]),
                                                 (.065, .065, .065), .008)], stains=[])
    else:
        litter, stains = _random_scene(cfg, spec.seed, spec.name == "near_wall")
        sim = CleaningSimulation(cfg, litter=litter, stains=stains)
    if spec.name == "pickup_retry":
        sim.force_next_failure()
    elif spec.name == "pickup_exhaustion":
        sim.collector.success_probability = 0.0
    elif spec.name == "detection_noise":
        sim.sensor_drop_probability = .08
    elif spec.name == "false_positive":
        sim.phantom_detections.append(Detection("debris", np.array([1.6, .8, .03]), .04,
            (0, 0), coordinate_frame="odom", object_class="paper_ball",
            size_m=(.06, .06, .06), estimated_mass_kg=.008))
    sim.scenario_name = spec.name
    return sim, ScenarioFaults(spec)


class ScenarioFaults:
    def __init__(self, spec):
        self.spec = spec
        self.triggered = False

    def apply(self, sim):
        if self.triggered:
            return
        state = sim.mission.state
        if self.spec.name == "sensor_blackout" and state == MissionState.ALIGNING_PICKUP:
            sim.blackout_sensor(.45)
            self.triggered = True
        elif self.spec.name == "payload_loss" and state == MissionState.RETURNING_HOME:
            self.triggered = sim.drop_payload()
        elif self.spec.name == "blocked_home" and state == MissionState.RETURNING_HOME:
            self.triggered = sim.block_home()


def evaluate_result(sim, spec, snapshot, steps, limit):
    summary = sim.summary()
    messages = [event["message"] for event in sim.mission.events]
    total = sum(item.object_class in LITTER_CLASSES for item in sim.litter)
    missed = [item.object_id for item in sim.litter if item.object_class in LITTER_CLASSES
              and item.status != "delivered"]
    safe_skips = []
    for item in sim.litter:
        if item.object_id not in missed:
            continue
        if any(det.object_class == item.object_class
               and np.linalg.norm(det.position_robot[:2]-item.position) < .08
               and (track_id in sim.mission.deferred_pickup_ids or track_id in sim.mission.failed_pickup_ids)
               for track_id, det in sim.mission.known_debris.items()):
            safe_skips.append(item.object_id)
    expected = "complete retrieval and cleaning"
    reason = messages[-1]
    passed = summary["state"] == "DONE" and not missed and summary["stains_cleaned"] == len(sim.stains)
    if spec.name == "payload_loss":
        expected = "stop safely when the payload is lost"
        passed = summary["state"] == "FAULT" and "payload lost during transport" in reason
    elif spec.name == "blocked_home":
        expected = "stop safely when the reported home pose is blocked"
        passed = summary["state"] == "FAULT" and "return route unavailable" in reason
    elif spec.name == "pickup_exhaustion":
        expected = "bound pickup attempts and continue coverage"
        passed = (summary["state"] == "DONE" and summary["delivered"] == 0
                  and summary["pickup_attempts"] == sim.cfg.collector.max_pickup_attempts)
    elif spec.name == "near_wall":
        expected = "collect reachable litter; safely reject constrained pickup poses"
        passed = (summary["state"] == "DONE" and not summary["carrying"]
                  and set(missed) == set(safe_skips) and summary["stains_cleaned"] == len(sim.stains))
    if not passed and summary["state"] == "DONE":
        reason = (f"uncollected litter IDs: {missed}" if missed else "stain cleaning incomplete")
    elif passed and safe_skips:
        reason = f"completed with safely rejected litter IDs: {safe_skips}"
    passed = bool(passed and summary["safety_stops"] == 0)
    if steps >= limit and summary["state"] not in ("DONE", "FAULT"):
        reason = "simulation step budget exhausted"
        passed = False
    arrivals = summary["home_arrivals"]
    if any(a["position_error_m"] > sim.cfg.collector.home_tolerance_m+1e-6
           or a["heading_error_rad"] > sim.cfg.collector.heading_tolerance_rad+1e-6 for a in arrivals):
        passed, reason = False, "home arrival exceeded configured position/heading tolerance"
    return {"scenario": spec.name, "seed": spec.seed, "passed": passed,
            "expected": expected, "reason": reason, "steps": steps, "total_litter": total,
            "missed_litter_ids": missed,
            "safe_skipped_litter_ids": safe_skips,
            "max_home_error_m": max((a["position_error_m"] for a in arrivals), default=None),
            "delivery_fraction": summary["delivered"] / total if total else 1.,
            "summary": summary, "events": sim.mission.events, "replay": snapshot}


class StressSession:
    def __init__(self, runs=20, seed=7, max_steps=15000, config=None, specs=None):
        if runs < 1 or max_steps < 1:
            raise ValueError("runs and step budget must be positive")
        self.config = deepcopy(config) if config is not None else simulation_config()
        self.specs = list(specs) if specs is not None else [
            ScenarioSpec(SCENARIOS[i % len(SCENARIOS)], seed+i) for i in range(runs)]
        self.max_steps = max_steps
        self.results = []
        self.current = None
        self.faults = None
        self.snapshot = None
        self.steps = 0
        self.running = True
        self.cancelled = False

    @property
    def complete(self):
        return len(self.results) == len(self.specs)

    def cancel(self):
        self.running = False
        self.cancelled = True

    def tick(self, max_steps=200, max_wall_s=.015):
        deadline = time.perf_counter() + max_wall_s
        for _ in range(max_steps):
            if not self.running or self.complete:
                self.running = False
                return
            if self.current is None:
                spec = self.specs[len(self.results)]
                self.current, self.faults = make_scenario(spec, self.config)
                self.snapshot = scene_snapshot(self.current, spec)
                self.steps = 0
            self.faults.apply(self.current)
            self.current.step()
            self.steps += 1
            if (self.current.mission.state == MissionState.WAITING_FOR_UNLOAD
                    and self.current.time_s - self.current.mission._state_since >= .25):
                self.current.unload()
            if self.current.mission.state in (MissionState.DONE, MissionState.FAULT) or self.steps >= self.max_steps:
                spec = self.specs[len(self.results)]
                self.results.append(evaluate_result(self.current, spec, self.snapshot, self.steps, self.max_steps))
                self.current = None
            if time.perf_counter() >= deadline:
                break

    def run(self, progress=None):
        previous = 0
        while self.running and not self.complete:
            self.tick(max_steps=1000, max_wall_s=1.)
            if progress is not None and len(self.results) != previous:
                for index, result in enumerate(self.results[previous:], start=previous+1):
                    progress(result, index, len(self.specs))
                previous = len(self.results)
        return self.report()

    def report(self):
        return {"runs_requested": len(self.specs), "runs_finished": len(self.results),
                "passed": sum(r["passed"] for r in self.results),
                "failed": sum(not r["passed"] for r in self.results),
                "cancelled": self.cancelled,
                "safety_stops": sum(r["summary"]["safety_stops"] for r in self.results),
                "delivered": sum(r["summary"]["delivered"] for r in self.results),
                "results": self.results}

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        report_path = directory / "stress_report.json"
        report_path.write_text(json.dumps(self.report(), indent=2), encoding="utf-8")
        with (directory / "stress_results.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["scenario", "seed", "passed", "delivered", "total_litter", "state",
                             "home_error_m", "simulation_time_s", "safety_stops", "reason"])
            for result in self.results:
                summary = result["summary"]
                writer.writerow([result["scenario"], result["seed"], result["passed"], summary["delivered"],
                                 result["total_litter"], summary["state"], result["max_home_error_m"],
                                 summary["simulation_time_s"], summary["safety_stops"], result["reason"]])
        for index, result in enumerate(self.results):
            if not result["passed"]:
                (directory / f"replay_{index+1:03d}_{result['seed']}.json").write_text(
                    json.dumps(result["replay"], indent=2), encoding="utf-8")
        return report_path
