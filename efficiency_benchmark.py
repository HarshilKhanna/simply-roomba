"""Run 100 fresh randomized paired trials and generate a presentation report.

    python -B efficiency_benchmark.py --trials 100

Reported rates describe this simulator and its stated assumptions only.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

from cleaning_robot.benchmark import make_protocol, run_benchmark
from cleaning_robot.simulation import simulation_config


def pct(value):
    return f"{100*value:.1f}%"


def ratio_text(metric, scale=1, suffix=""):
    if metric is None:
        return "not estimable"
    lo, hi = metric["ci95_scene_bootstrap"]
    return f"{scale*metric['value']:.2f}{suffix} (95% interval {scale*lo:.2f}-{scale*hi:.2f}{suffix})"


def mean_text(metric, unit):
    if metric is None:
        return "not estimable"
    return f"mean {metric['mean']:.2f}{unit}; median {metric['median']:.2f}{unit}; p90 {metric['p90']:.2f}{unit}"


def write_report(report, directory):
    a, p = report["aggregate"], report["protocol"]
    low, high = a["mission_success_ci95_wilson"]
    lines = [
        "# Randomized cleaning and retrieval benchmark", "",
        "**Simulation evidence only. These results do not establish physical D405 detection, claw pickup, or dirt-removal efficiency.**", "",
        f"{a['trials']} fresh randomized paired scenes; {p['paired_mission_runs']} mission runs. "
        f"Seeds {p['seeds'][0]}-{p['seeds'][-1]}; {a['objects']} litter objects and {a['stains_total']} stains in the collection-enabled arm. "
        "All outcomes, including incomplete missions, are included in headline denominators.", "",
        "## Measured outcomes", "",
        f"- **Collection efficiency:** {a['delivered']}/{a['objects']} objects delivered and manually unloaded at origin "
        f"= **{pct(a['delivery_rate']['value'])}**. "
        "This is delivered objects / all initial objects, not a fault-scenario pass score.",
        f"- **Full mission success:** {a['mission_successes']}/{a['trials']} = **{pct(a['mission_success_rate'])}** "
        f"(95% Wilson interval {pct(low)}-{pct(high)}). Requires all litter, all stains, DONE, no guard stops and home tolerance.",
        f"- **Stain cleaning:** {a['stains_cleaned']}/{a['stains_total']} fully covered; "
        f"{ratio_text(a['stain_sample_removal'], 100, '%')} stain samples removed geometrically.",
        f"- **Pickup attempt success:** {a['verified_pickups']}/{a['pickup_attempts']} close operations "
        f"({ratio_text(a['pickup_attempt_success'], 100, '%')}). "
        f"{a['first_attempt_verified']} objects were verified on the first attempt. "
        "This is a simulated geometric capture plus assumed Bernoulli pickup model.",
        f"- **Throughput:** {ratio_text(a['objects_per_active_minute'])} objects/active minute; "
        f"{ratio_text(a['objects_per_elapsed_minute'])} objects/elapsed minute with "
        f"{p['unload_delay_s']:g}s standardized operator delay per unload.",
        f"- **Completed full missions:** {mean_text(a['completed_time_s'], 's')} elapsed time; "
        f"{mean_text(a['completed_path_m'], 'm')} travel. These statistics exclude incomplete missions "
        f"and use {a['mission_successes']} completed full missions.",
        f"- **Aborts/time limits:** {a['enhanced_faults']} faults, {a['enhanced_timeouts']} timeouts, "
        f"{a['enhanced_safety_stops']} prevented collisions. Every arm has a "
        f"{p['max_simulated_seconds_per_arm']:g}s simulation-time cap including unload waits.", "",
        "## Nominal performance and sensitivity assumptions", "",
        "| Profile | Scenes | Delivered / initial | Collection | Full missions | Stains cleaned | Guard stops |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, s in report["profiles"].items():
        lines.append(f"| {name} | {s['trials']} | {s['delivered']}/{s['objects']} | "
                     f"{pct(s['delivery_rate']['value'])} | {s['mission_successes']}/{s['trials']} | "
                     f"{s['stains_cleaned']}/{s['stains_total']} | {s['enhanced_safety_stops']} |")
    lines += ["", p["assumptions"], "",
        "The pooled result uses this deliberately chosen 50/25/25 profile mix (for 100 trials); "
        "it does not estimate an unknown real-world mix of conditions. Boundary cases also use the disturbance assumptions.", "",
        "## Object classes", "", "| Class | Delivered / initial | Collection |", "|---|---:|---:|"]
    for label, metric in a["by_class"].items():
        objects = [x for r in report["results"] for x in r["enhanced"]["summary"]["objects"] if x["class"] == label]
        lines.append(f"| {label.replace('_', ' ')} | {sum(x['status']=='delivered' for x in objects)}/{len(objects)} | {pct(metric['value'])} |")
    lines += ["", "## Comparison against collection disabled", "", p["comparison"], "",
        f"The disabled arm delivered {a['baseline_delivered']} objects (expected by design) and fully cleaned "
        f"{a['baseline_stains_cleaned']}/{a['stains_total']} stains. "
        f"It completed {a['baseline_cleaning_successes']}/{a['trials']} cleaning missions, with "
        f"{a['baseline_faults']} faults, {a['baseline_timeouts']} timeouts and {a['baseline_safety_stops']} guard stops.", "",
        f"Time/path comparisons below use **{a['paired_completed_count']}/{a['trials']} pairs where both arms "
        "completed cleaning**. This subset can be selected toward easier scenes; no failed or timed-out arm is treated "
        "as a fast completion. Collection-disabled runs perform less work because they leave litter in place.", "",
        f"- Additional elapsed time: {mean_text(a['paired_extra_elapsed_s'], 's')}.",
        f"- Additional active time excluding unloading: {mean_text(a['paired_extra_active_s'], 's')}.",
        f"- Additional travel: {mean_text(a['paired_extra_path_m'], 'm')}.",
        f"- Elapsed time ratio enhanced / disabled: {ratio_text(a['paired_elapsed_ratio'])}.",
        f"- Additional elapsed seconds per delivered object: {ratio_text(a['paired_extra_seconds_per_delivery'])}.", "",
        "The enhancement adds collection capability and has a measured time/travel cost. "
        "This experiment does not show that it cleans stains faster or uses less energy.", "",
        "## Return to origin", "",
        f"{a['home_arrivals']} measured home arrivals; {a['out_of_tolerance_arrivals']} outside tolerance. "
        f"Maximum position error: {a['max_home_error_mm'] if a['max_home_error_mm'] is not None else 'unavailable'} mm; "
        f"maximum heading error: {a['max_home_heading_error_deg'] if a['max_home_heading_error_deg'] is not None else 'unavailable'} degrees. "
        "These are errors against exact simulated pose, not an odometry/localization accuracy measurement.", "",
        "## Failures retained in the benchmark", "",
        "| Trial | Seed | Profile | Delivered | End state | Reason |", "|---:|---:|---|---:|---|---|"]
    failures = [r for r in report["results"] if not r["enhanced"]["full_success"]]
    for r in failures:
        e = r["enhanced"]
        lines.append(f"| {r['trial']} | {r['seed']} | {r['profile']} | {e['summary']['delivered']}/4 | "
                     f"{e['summary']['state']} | {'; '.join(e['issues'])} |")
    if not failures:
        lines.append("| - | - | - | - | - | No full-mission failures in this sample |")
    lines += ["", "## Method and limits", "", p["sampling"], "", p["statistics"], "", p["limits"], "",
        "The unmodified controller and simulator source SHA-256 hashes, complete configuration, RNG seeds and "
        "protocol were saved before execution in `protocol.json`. Initial scenes and event logs for both arms "
        "are retained in `results.json` and append-only `trials.jsonl`. "
        "The runner refuses to overwrite previous benchmark evidence.", "",
        "Throughput includes every trial's observed runtime, including aborted runs; it is not the expected time "
        "to finish a failed mission. Stain sample percentage is geometric area coverage of discretized stain disks, "
        "not a chemical dirt-removal measurement. No power/energy efficiency is reported because the model lacks motor "
        "loads, currents and a battery model. Scene-cluster bootstrap intervals can collapse at 0% or 100%; "
        "the Wilson interval on full-scene success better communicates finite-sample uncertainty.", "",
        "## Claim suitable for presenting", "",
        f"> Across {a['trials']} seeded randomized simulated scenes containing {a['objects']} lightweight litter objects, "
        f"> the mop-plus-collector controller delivered {a['delivered']} objects to its starting location "
        f"> ({pct(a['delivery_rate']['value'])}) and completed all retrieval and stain-cleaning tasks in "
        f"> {a['mission_successes']}/{a['trials']} scenes. These results use an ideal geometric sensor/actuator model "
        "> with separately documented synthetic pickup and observation disturbances; physical performance remains unmeasured.", "",
        "## Next validation", "",
        "Run independent physical pickup trials for each class and size/orientation, measure classifier precision/recall "
        "from recorded D405 data, compare ground-truth home pose with estimated pose, and log elapsed time, operator "
        "wait, wheel/claw current and failures. Do not relabel the simulated percentages as hardware efficiency.", "",
    ]
    (directory/"report.md").write_text("\n".join(lines), encoding="utf-8")


def plot_report(report, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    rows, a = report["results"], report["aggregate"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.8), layout="constrained")
    fig.suptitle(f"Randomized retrieval benchmark | {a['trials']} paired scenes\n"
                 "SIMULATION ONLY - mock semantic sensor, geometric capture, assumed disturbances", fontsize=16, fontweight="bold")
    colors = ["#14826d", "#d48a23", "#486fa8"]
    ax = axes[0, 0]
    names = list(report["profiles"])
    x = np.arange(len(names))
    delivery = [report["profiles"][n]["delivery_rate"]["value"] for n in names]
    mission = [report["profiles"][n]["mission_success_rate"] for n in names]
    ax.bar(x-.18, delivery, .36, label="Objects delivered", color=colors[0])
    ax.bar(x+.18, mission, .36, label="Full mission success", color=colors[1])
    ax.set(xticks=x, xticklabels=[f"{n}\n(n={report['profiles'][n]['trials']})" for n in names],
           ylim=(0, 1.18), title="Collection and mission completion")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    for i in range(len(x)):
        ax.text(i-.18, delivery[i]+.025, pct(delivery[i]), ha="center", fontsize=9)
        ax.text(i+.18, mission[i]+.025, pct(mission[i]), ha="center", fontsize=9)
    ax.legend(fontsize=8, loc="upper right")
    ax = axes[0, 1]
    labels = list(a["by_class"])
    vals = [a["by_class"][n]["value"] for n in labels]
    ax.barh([n.replace("_", " ") for n in labels], vals, color=colors[0])
    ax.set(xlim=(0, 1.2), title="Delivered / all initial objects by class")
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    for i, v in enumerate(vals):
        ax.text(v+.02, i, pct(v), va="center")
    ax = axes[0, 2]
    for name, color in zip(names, colors):
        selected = [r for r in rows if r["profile"] == name]
        ax.scatter([r["trial"] for r in selected], [r["enhanced"]["summary"]["delivered"] for r in selected],
                   s=25, label=name, alpha=.8, c=color)
    ax.set(xlabel="Trial (seeded; no discarded trials)", ylabel="Objects delivered / four", ylim=(-.2, 4.5),
           yticks=range(5), title="Every trial retained")
    ax.legend(fontsize=8, loc="lower left")
    ax = axes[1, 0]
    pairs = [r for r in rows if r["enhanced"]["cleaning_complete"] and r["baseline"]["cleaning_complete"]]
    for name, color in zip(names, colors):
        selected = [r for r in pairs if r["profile"] == name]
        ax.scatter([r["baseline"]["elapsed_s"]/60 for r in selected],
                   [r["enhanced"]["elapsed_s"]/60 for r in selected], c=color, label=name, alpha=.7, s=22)
    if pairs:
        maximum = max(max(r[k]["elapsed_s"]/60 for k in ("baseline", "enhanced")) for r in pairs)*1.05
        ax.plot([0, maximum], [0, maximum], "--", color="#7b8085", linewidth=1)
    ax.set(xlabel="Collection disabled (minutes)", ylabel="Collection enabled (minutes)",
           title=f"Elapsed time | {len(pairs)} completed cleaning pairs")
    ax.text(.03, .97, "Diagonal = equal time\n10s/unload default; disabled does less work",
            transform=ax.transAxes, va="top", fontsize=8)
    ax = axes[1, 1]
    categories = {"Coverage / stain cleaning": ("COVERAGE", "DIVERTING", "SCRUBBING"),
                  "Approach / align / retries": ("APPROACHING_DEBRIS", "ALIGNING_PICKUP", "RECOVERING_PICKUP"),
                  "Claw / verify": ("COLLECTING", "VERIFYING_PICKUP"),
                  "Return / resume": ("RETURNING_HOME", "RESUMING_COVERAGE"),
                  "Manual unloading wait": ("WAITING_FOR_UNLOAD",)}
    sums = [sum(a["state_time_s"].get(n, 0) for n in states)/60/a["trials"] for states in categories.values()]
    ax.barh(list(categories), sums, color=[colors[0], colors[1], "#9771a8", colors[2], "#939da8"])
    ax.set(xlabel="Mean observed simulated minutes / trial", title="Where time is spent | all trials")
    ax = axes[1, 2]
    ax.axis("off")
    ci = a["mission_success_ci95_wilson"]
    summary = (f"COLLECTION: {a['delivered']}/{a['objects']} ({pct(a['delivery_rate']['value'])})\n\n"
               f"FULL MISSIONS: {a['mission_successes']}/{a['trials']}\n"
               f"95% Wilson interval: {pct(ci[0])} - {pct(ci[1])}\n\n"
               f"STAINS: {a['stains_cleaned']}/{a['stains_total']} fully covered\n"
               f"FAULTS / TIMEOUTS: {a['enhanced_faults']} / {a['enhanced_timeouts']}\n"
               f"COLLISION GUARD STOPS: {a['enhanced_safety_stops']}\n\n"
               "No measured physical detection/grasp accuracy.\n"
               "No motor energy measurement.\n"
               "Controller frozen; protocol and raw scenes saved.")
    ax.text(.02, .97, summary, va="top", fontsize=11, linespacing=1.4)
    for ax in axes.flat:
        if ax.axison:
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", alpha=.15)
            ax.set_axisbelow(True)
            ax.tick_params(labelsize=9)
    fig.savefig(directory/"benchmark_summary.png", dpi=180)
    fig.savefig(directory/"benchmark_summary.pdf")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026101800)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--unload-delay", type=float, default=10.)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--render-report", type=Path, help="regenerate only report/plots from existing results JSON")
    args = parser.parse_args(argv)
    if args.render_report:
        report = json.loads(args.render_report.read_text(encoding="utf-8"))
        directory = args.render_report.parent
    else:
        try:
            protocol = make_protocol(args.trials, args.seed, args.steps, args.unload_delay, simulation_config(args.config))
        except ValueError as error:
            parser.error(str(error))
        directory = args.output_dir or Path("artifacts/efficiency")/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        def progress(row):
            e, b = row["enhanced"], row["baseline"]
            print(f"{row['trial']:3}/{args.trials} {row['profile']:<11} seed={row['seed']} "
                  f"delivered={e['summary']['delivered']}/4 full={'YES' if e['full_success'] else 'NO '} "
                  f"elapsed={e['elapsed_s']:.1f}s baseline={b['summary']['state']} "
                  f"baseline_time={b['elapsed_s']:.1f}s", flush=True)
        report = run_benchmark(protocol, directory, progress)
    write_report(report, directory)
    plot_report(report, directory)
    a = report["aggregate"]
    print(f"\nSIMULATION: {a['delivered']}/{a['objects']} delivered ({pct(a['delivery_rate']['value'])}); "
          f"{a['mission_successes']}/{a['trials']} full missions. Report: {directory/'report.md'}", flush=True)
    # Performance failures are data, not an error in benchmark execution.
    return 0


if __name__ == "__main__":
    sys.exit(main())
