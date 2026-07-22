"""
Master orchestrator -- runs the experiment scripts in a sensible order,
skipping the parts you haven't set up (e.g. exp03/06 need a trained
adapter from reverse_kl_train.py first). Not required -- you can run any
experiments/exp*.py individually -- but this gives one command for
"run everything reasonable given what's already been trained."

Usage:
    python run_all.py                  # run cheap/diagnostic experiments first
    python run_all.py --with-training  # also run reverse_kl_train.py first
    python run_all.py --only exp01,exp04
"""

import argparse
import os
import subprocess
import sys

EXPERIMENTS_DIR = "experiments"
RESULTS_DIR = "results"

# Ordered so cheap/fast experiments (no target model, or smaller loads) run
# first -- fail-fast on config issues before burning time on the big loads.
ORDER = [
    "exp04_lora_overhead_at_scale",       # cheap, no target model, run first
    "exp01_large_target_ratio",
    "exp02_matched_ratio_family_test",
    "exp05_quantization_isolation",
    "exp03_fixed_vs_adaptive_draft_length",  # needs a trained adapter
    "exp06_full_reverse_kl_pipeline",        # needs a trained adapter
]

NEEDS_TRAINED_ADAPTER = {"exp03_fixed_vs_adaptive_draft_length", "exp06_full_reverse_kl_pipeline"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-training", action="store_true",
                         help="run src/reverse_kl_train.py before the experiments that need a trained adapter")
    parser.add_argument("--only", default=None, help="comma-separated experiment names to run, e.g. exp01,exp04")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    to_run = ORDER
    if args.only:
        wanted = set(args.only.split(","))
        to_run = [e for e in ORDER if any(w in e for w in wanted)]

    if args.with_training:
        print("=== Training reverse-KL drafter first (src/reverse_kl_train.py) ===", flush=True)
        subprocess.run([sys.executable, "src/reverse_kl_train.py",
                         "--target-id", "google/gemma-2-27b",
                         "--save-dir", "experiments/trained_drafter"], check=True)

    for exp in to_run:
        if exp in NEEDS_TRAINED_ADAPTER and not args.with_training and not os.path.exists("experiments/trained_drafter"):
            print(f"Skipping {exp}: needs a trained adapter (run with --with-training, "
                  f"or train separately into experiments/trained_drafter/ first)")
            continue
        print(f"\n{'=' * 60}\nRunning {exp}\n{'=' * 60}", flush=True)
        subprocess.run([sys.executable, f"{EXPERIMENTS_DIR}/{exp}.py"], cwd=EXPERIMENTS_DIR, check=True)

    print("\n=== Generating consolidated report ===", flush=True)
    subprocess.run([sys.executable, "report.py"], check=True)


if __name__ == "__main__":
    main()
