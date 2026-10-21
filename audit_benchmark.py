"""Independently audit saved benchmark evidence and arithmetic.

    python -B audit_benchmark.py artifacts/efficiency/RUN_DIRECTORY/results.json
"""
from collections import Counter
import argparse
import hashlib
import json
from pathlib import Path


def audit(path):
    path = Path(path)
    report = json.loads(path.read_text(encoding="utf-8"))
    protocol, rows, a = report["protocol"], report["results"], report["aggregate"]
    checks = 0
    def require(condition, message):
        nonlocal checks
        checks += 1
        if not condition:
            raise AssertionError(message)

    require(json.loads(path.with_name("protocol.json").read_text(encoding="utf-8")) == protocol,
            "protocol differs from preregistered file")
    journal = [json.loads(line) for line in path.with_name("trials.jsonl").read_text(encoding="utf-8").splitlines()]
    require(journal == rows, "streamed journal differs from final results")
    require(len(rows) == protocol["trials"] == a["trials"], "trial count mismatch")
    require([r["seed"] for r in rows] == protocol["seeds"], "missing/reordered seeds")
    require(len({r["seed"] for r in rows}) == len(rows), "repeated seeds")
    layout_signatures = set()
    for row in rows:
        e, b = row["enhanced"], row["baseline"]
        require(e["initial_scene"]["litter"] == b["initial_scene"]["litter"], "unmatched litter layout")
        require(e["initial_scene"]["stains"] == b["initial_scene"]["stains"], "unmatched stain layout")
        layout_signatures.add(json.dumps(e["initial_scene"]["litter"], sort_keys=True))
        for arm in (e, b):
            s, cfg = arm["summary"], arm["initial_scene"]["config"]
            require(len(s["objects"]) == 4, "scene must contain four litter objects")
            require(len(arm["initial_scene"]["stains"]) == 2, "scene must contain two stains")
            require(s["delivered"] == sum(x["status"] == "delivered" for x in s["objects"]), "delivery count mismatch")
            require(s["delivered"] <= arm["verified_pickups"] <= s["pickup_attempts"], "pickup arithmetic inconsistent")
            require(arm["first_attempt_verified"] <= arm["verified_pickups"], "first-attempt count inconsistent")
            require(s["pickup_attempts"] <= len(s["objects"])*cfg["collector"]["max_pickup_attempts"], "retry budget exceeded")
            require(abs(arm["elapsed_s"] - arm["steps"]*cfg["simulation"]["time_step_s"]) < 1e-6, "step time mismatch")
            require(abs(sum(arm["state_time_s"].values()) - arm["elapsed_s"]) < 1e-6, "state times do not partition elapsed time")
            require(abs(arm["active_s"]+arm["unload_wait_s"]-arm["elapsed_s"]) < 1e-6, "active/wait time mismatch")
            require(0 <= arm["stain_samples_removed"] <= arm["stain_samples_total"], "invalid stain sample count")
            require(arm["elapsed_s"] <= protocol["max_simulated_seconds_per_arm"]+1e-6, "time budget exceeded")
            require(not arm["full_success"] or (s["state"] == "DONE" and s["safety_stops"] == 0
                    and s["stains_cleaned"] == 2 and (not arm["collector_enabled"] or s["delivered"] == 4)),
                    "full success includes an incomplete outcome")
            home_ok = all(x["position_error_m"] <= cfg["collector"]["home_tolerance_m"]+1e-9
                          and x["heading_error_rad"] <= cfg["collector"]["heading_tolerance_rad"]+1e-9
                          for x in s["home_arrivals"])
            require(home_ok == arm["home_arrivals_in_tolerance"], "home tolerance result inconsistent")
    require(len(layout_signatures) == len(rows), "repeated scene geometry")
    require(a["objects"] == sum(len(r["enhanced"]["summary"]["objects"]) for r in rows), "wrong object denominator")
    require(a["delivered"] == sum(r["enhanced"]["summary"]["delivered"] for r in rows), "wrong delivered total")
    require(a["mission_successes"] == sum(r["enhanced"]["full_success"] for r in rows), "wrong mission total")
    require(abs(a["delivery_rate"]["value"]-a["delivered"]/a["objects"]) < 1e-12, "wrong collection rate")
    require(sum(s["trials"] for s in report["profiles"].values()) == len(rows), "profile denominator mismatch")
    root = Path(__file__).resolve().parent
    for filename, digest in protocol["source_sha256"].items():
        require(hashlib.sha256((root/filename).read_bytes()).hexdigest() == digest,
                f"source changed since protocol: {filename}")
    return {"passed": True, "checks": checks, "trials": len(rows), "unique_layouts": len(layout_signatures),
            "profiles": dict(Counter(r["profile"] for r in rows)), "source_hashes_match": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    result = audit(args.results)
    args.results.with_name("audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
