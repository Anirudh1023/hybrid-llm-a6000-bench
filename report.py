"""
Consolidated benchmark report generator.

Reads every JSON file in results/, builds one summary table + narrative
comparison against the T4-session baselines (hardcoded below, from
CONCLUSIONS.md), and writes a single markdown report.

Run this after running whichever experiments/exp*.py scripts you've done --
it's safe to run repeatedly, picks up whatever result files exist so far.
"""

import glob
import json
import os
import statistics as stats

RESULTS_DIR = "results"
OUTPUT_FILE = "BENCHMARK_REPORT.md"

# T4-session reference numbers (2026-07-22/23), for direct comparison.
# See CONCLUSIONS.md for full provenance of each of these.
T4_BASELINES = {
    "untrained_qwen3_0.6b_gemma2_9b_15to1": 1.295,
    "untrained_qwen2.5_0.5b_gemma2_9b_18.2to1": 1.479,
    "same_family_qwen3_0.6b_qwen2.5_7b_11.7to1_UNVALIDATED_confounded": 0.977,
    "truly_native_qwen2.5_0.5b_qwen2.5_3b_6.2to1": 1.339,
    "trained_reverse_kl_adaptive_UNMERGED": 1.042,
    "trained_reverse_kl_fixed4_UNMERGED": 1.210,
    "lora_overhead_isolated_unmerged": 1.443,
    "lora_overhead_isolated_merged": 0.976,
}


def summarize_condition(cond):
    speedups = cond["speedups"]
    return {
        "avg_speedup": sum(speedups) / len(speedups),
        "median_speedup": stats.median(speedups),
        "avg_tpc": cond["avg_tpc"],
        "n": len(speedups),
        "below_1x": sum(1 for s in speedups if s < 1.0),
    }


def render_result_file(path):
    with open(path) as f:
        data = json.load(f)
    lines = [f"### {os.path.basename(path)}", ""]
    meta = data.get("meta", {})
    if meta:
        lines.append(f"Meta: `{json.dumps(meta)}`")
        lines.append("")

    # handle both flat {"conditions": {...}} and nested (e.g. exp03/exp06's
    # {"adaptive": {...}, "fixed": {...}}) result shapes
    def walk(d, prefix=""):
        if isinstance(d, dict) and "conditions" in d:
            lines.append(f"| condition | avg speedup | median speedup | avg tok/call | n below 1.0x |")
            lines.append(f"|---|---|---|---|---|")
            for label, cond in d["conditions"].items():
                s = summarize_condition(cond)
                lines.append(f"| {prefix}{label} | {s['avg_speedup']:.3f}x | {s['median_speedup']:.3f}x | "
                              f"{s['avg_tpc']:.3f} | {s['below_1x']}/{s['n']} |")
        elif isinstance(d, dict):
            for k, v in d.items():
                if k != "meta":
                    walk(v, prefix=f"{k}/")

    walk(data)
    lines.append("")
    return "\n".join(lines)


def main():
    result_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "*.json")))
    if not result_files:
        print(f"No result files found in {RESULTS_DIR}/ yet -- run some experiments first.")
        return

    report = ["# A6000 Benchmark Report", "", f"Generated from {len(result_files)} result file(s).", ""]
    report.append("## T4-session reference numbers (for comparison)")
    report.append("")
    report.append("| label | avg speedup |")
    report.append("|---|---|")
    for label, val in T4_BASELINES.items():
        report.append(f"| {label} | {val:.3f}x |")
    report.append("")
    report.append("## A6000 results")
    report.append("")

    for path in result_files:
        try:
            report.append(render_result_file(path))
        except Exception as e:
            report.append(f"### {os.path.basename(path)}\n\n(failed to parse: {e})\n")

    with open(OUTPUT_FILE, "w") as f:
        f.write("\n".join(report))
    print(f"Wrote {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
