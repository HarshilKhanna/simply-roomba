"""Re-execute one recorded benchmark arm and compare every saved measurement.

    python -B replay_benchmark.py artifacts/efficiency/RUN/results.json --trial 19
"""
import argparse
import json
from pathlib import Path

import numpy as np

from cleaning_robot.benchmark import run_arm
from cleaning_robot.simulation import Litter, Stain, apply_config_overrides, simulation_config


def replay(results_path, trial, arm="enhanced"):
    report = json.loads(Path(results_path).read_text(encoding="utf-8"))
    row = next((r for r in report["results"] if r["trial"] == trial), None)
    if row is None:
        raise ValueError(f"trial {trial} not present in report")
    old = row[arm]
    snapshot = old["initial_scene"]
    cfg = simulation_config()
    apply_config_overrides(cfg, snapshot["config"])
    litter = [Litter(**{**x, "position": np.array(x["position"]), "size_m": tuple(x["size_m"])})
              for x in snapshot["litter"]]
    stains = [Stain(x["object_id"], np.array(x["position"]), x["radius_m"]) for x in snapshot["stains"]]
    new = run_arm(cfg, litter, stains, row["assumptions"], enabled=arm == "enhanced",
                  max_steps=report["protocol"]["max_steps_per_arm"],
                  unload_delay_s=report["protocol"]["unload_delay_s"])
    differences = [key for key in old if old[key] != new[key]]
    return {"trial": trial, "seed": row["seed"], "arm": arm, "exact_match": not differences,
            "differing_fields": differences, "summary": new["summary"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--trial", type=int, required=True)
    parser.add_argument("--arm", choices=("enhanced", "baseline"), default="enhanced")
    args = parser.parse_args()
    result = replay(args.results, args.trial, args.arm)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["exact_match"] else 1)
