"""Run repeatable stress cases and export JSON, CSV and a visual report.

    python stress_test.py --runs 100
    python visual_sim.py --gui --stress
"""
import argparse
from pathlib import Path
import sys

import cv2

from cleaning_robot.simulation import simulation_config
from cleaning_robot.stress import StressSession
from visual_sim import SimulationApp


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=15000)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/stress/latest"))
    args = parser.parse_args(argv)
    if args.runs < 1 or args.steps < 1 or args.seed < 0:
        parser.error("runs and steps must be positive; seed must be nonnegative")
    cfg = simulation_config(args.config)
    session = StressSession(args.runs, args.seed, args.steps, cfg)

    def progress(result, index, total):
        outcome = "PASS" if result["passed"] else "FAIL"
        summary = result["summary"]
        print(f"{index:3}/{total}  {outcome}  {result['scenario']:<20} seed={result['seed']:<5} "
              f"delivered={summary['delivered']}/{result['total_litter']}  {summary['state']}", flush=True)

    report = session.run(progress)
    path = session.save(args.output_dir)
    app = SimulationApp(cfg, stress_runs=args.runs)
    app.stress, app.mode = session, "stress"
    cv2.imwrite(str(args.output_dir/"stress_report.png"), app.render())
    print(f"\n{report['passed']}/{report['runs_finished']} expectations passed. "
          f"Safety stops: {report['safety_stops']}\nReport: {path}\n"
          f"Replay a failure: python visual_sim.py --gui --replay {args.output_dir}/replay_NNN_SEED.json")
    return 0 if report["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
