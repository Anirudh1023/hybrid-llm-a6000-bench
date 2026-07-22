"""
Experiment 2: same-family vs cross-family, with ratio PROPERLY MATCHED.

Why this is needed: the T4 session ran a same-family test (Qwen3-0.6B +
Qwen2.5-7B-Instruct, ~11.7:1) that came out WORSE (0.977x) than cross-family
Gemma tests at higher ratios (1.295x @ 15:1, 1.479x @ 18.2:1) -- but that
same-family test conflated TWO confounds at once: (a) it silently ran
through the cross-vocab path anyway (config.vocab_size mismatch, see
vocab_utils.py docstring) and (b) it used a SMALLER ratio than the
cross-family comparisons it was set against. Neither the "same-family SD
performed worse" headline NOR a clean "ratio explains everything" claim
survived scrutiny -- this experiment is the properly controlled version
that was never actually run.

Design: pick target_ratio to match a cross-family baseline you already
trust (e.g. ~15-18:1, the T4 Qwen3-0.6B+Gemma-2-9b / Qwen2.5-0.5B+Gemma-2-9b
numbers). Then find a genuinely SAME-FAMILY pair at that SAME ratio, using
vocab_utils.check_native_path() to confirm whether it hits the true native
fast path or still needs the cross-vocab translator -- report BOTH, since
that distinction matters (see exp03).

Example candidate pairs to try (check vocab_size match with vocab_utils
first -- same generation doesn't guarantee it, as the T4 session found):
  - Qwen2.5-1.5B-Instruct (drafter) + Qwen2.5-32B-Instruct (target): ~21:1
  - Qwen2.5-3B-Instruct (drafter) + Qwen2.5-32B-Instruct (target): ~11:1
  - Gemma-2-2B (drafter) + Gemma-2-27B (target): ~13.5:1 -- same family AND
    same generation as our validated T4 Gemma-2 target, if a 2B Gemma-2
    checkpoint is available
"""

import json
import sys

sys.path.insert(0, "../src")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from harness import run_full_comparison
from vocab_utils import check_native_path

# EDIT THESE to a same-family pair at your chosen matched ratio -- pick
# after checking vocab_utils.check_native_path() and computing the actual
# ratio, don't assume from listed sizes alone.
TARGET_ID = "Qwen/Qwen2.5-32B-Instruct"
DRAFT_ID = "Qwen/Qwen2.5-3B-Instruct"
CROSS_FAMILY_BASELINE_RATIO = 15.0  # the T4 number you're trying to match, for reference in the printout

PROMPTS_FILE = "../data/heldout_pairs_groundtruth.json"
RESULTS_FILE = "../results/exp02_matched_ratio_family_test.json"


def main():
    with open(PROMPTS_FILE) as f:
        pairs = json.load(f)
    prompts = [p["prompt"] if isinstance(p, dict) else p for p in pairs]

    vocab_check = check_native_path(TARGET_ID, DRAFT_ID)
    print(f"Vocab check: {vocab_check}", flush=True)
    print(f"native_path={vocab_check['native_path']} -- if False, this pair STILL goes "
          f"through the cross-vocab translator despite being same-family, exactly like "
          f"the T4 session's Qwen3-0.6B+Qwen2.5-7B-Instruct surprise. Report this either way.",
          flush=True)
    same_tokenizer = vocab_check["native_path"]

    target_tokenizer = AutoTokenizer.from_pretrained(TARGET_ID)
    assistant_tokenizer = AutoTokenizer.from_pretrained(DRAFT_ID)

    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    target_model = AutoModelForCausalLM.from_pretrained(TARGET_ID, quantization_config=bnb_config, device_map={"": 0})
    draft_model = AutoModelForCausalLM.from_pretrained(DRAFT_ID, dtype=torch.float16, device_map={"": 0})

    draft_params = sum(p.numel() for p in draft_model.parameters())
    target_params = sum(p.numel() for p in target_model.parameters())
    actual_ratio = target_params / draft_params
    print(f"draft: {draft_params / 1e9:.3f}B  target: {target_params / 1e9:.3f}B  "
          f"ratio: {actual_ratio:.1f}:1  (cross-family baseline was {CROSS_FAMILY_BASELINE_RATIO}:1)",
          flush=True)

    results = run_full_comparison(
        target_model, target_tokenizer,
        conditions=[("same_family_matched_ratio", draft_model, None if same_tokenizer else assistant_tokenizer)],
        prompts=prompts, same_tokenizer=same_tokenizer,
    )
    results["meta"] = {
        "target_id": TARGET_ID, "draft_id": DRAFT_ID, "actual_ratio": actual_ratio,
        "cross_family_baseline_ratio": CROSS_FAMILY_BASELINE_RATIO, "vocab_check": vocab_check,
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {RESULTS_FILE}", flush=True)


if __name__ == "__main__":
    main()
