"""
Experiment 6: the full, corrected reverse-KL pipeline at A6000 scale --
train, merge, fix draft length, THEN measure wall-clock, all in the right
order this time.

This is the "redo everything correctly in one script" experiment -- it
applies every fix/lesson from the T4 session in the right sequence, so you
get a genuinely fair reading of whether reverse-KL distillation helps at
this scale, not another accidental confound pileup:

  1. Train with the FIXED reverse-KL loss (reverse_kl_train.py -- the
     -inf/masked_fill fix, not the version that silently produced NaN on
     100% of steps for an entire session before being caught).
  2. MERGE the LoRA adapter before any inference measurement
     (lora_utils.load_and_merge -- eliminates the confirmed ~44% per-step
     overhead from carrying unmerged LoRA weights).
  3. Measure wall-clock speedup under BOTH adaptive and fixed draft length
     (the adaptive heuristic only partially explains the T4 regression --
     report both so scale-dependence of that effect is visible too).
  4. Compare before/after on tok/call AND wall-clock, always both --
     tok/call alone was proven insufficient and misleading this session
     (the +10.6% tok/call gain masked a real wall-clock regression).

Run reverse_kl_train.py first (or point TRAINED_ADAPTER_DIR at an existing
one), then this script.
"""

import json
import sys

sys.path.insert(0, "../src")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from harness import run_full_comparison, set_fixed_draft_length
from lora_utils import load_and_merge
from vocab_utils import check_native_path

TARGET_ID = "google/gemma-2-27b"   # continues the validated Gemma-2 target family from the T4 session
DRAFT_ID = "Qwen/Qwen3-0.6B"
TRAINED_ADAPTER_DIR = "trained_drafter"
FIXED_DRAFT_LEN = 4

PROMPTS_FILE = "../data/heldout_pairs_groundtruth.json"
RESULTS_FILE = "../results/exp06_full_reverse_kl_pipeline.json"


def main():
    with open(PROMPTS_FILE) as f:
        pairs = json.load(f)
    prompts = [p["prompt"] if isinstance(p, dict) else p for p in pairs]

    vocab_check = check_native_path(TARGET_ID, DRAFT_ID)
    same_tokenizer = vocab_check["native_path"]
    print(f"Vocab check: {vocab_check}", flush=True)

    target_tokenizer = AutoTokenizer.from_pretrained(TARGET_ID)
    assistant_tokenizer = AutoTokenizer.from_pretrained(DRAFT_ID)

    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    target_model = AutoModelForCausalLM.from_pretrained(TARGET_ID, quantization_config=bnb_config, device_map={"": 0})
    base_assistant = AutoModelForCausalLM.from_pretrained(DRAFT_ID, dtype=torch.float16, device_map={"": 1})

    merged_model = load_and_merge(base_assistant, TRAINED_ADAPTER_DIR)

    all_results = {}
    for length_label, fix_len in [("adaptive", None), ("fixed4", FIXED_DRAFT_LEN)]:
        if fix_len is not None:
            set_fixed_draft_length(base_assistant, fix_len)
            set_fixed_draft_length(merged_model, fix_len)
        print(f"=== {length_label} draft length ===", flush=True)
        results = run_full_comparison(
            target_model, target_tokenizer,
            conditions=[
                (f"untrained_{length_label}", base_assistant, None if same_tokenizer else assistant_tokenizer),
                (f"trained_merged_{length_label}", merged_model, None if same_tokenizer else assistant_tokenizer),
            ],
            prompts=prompts, same_tokenizer=same_tokenizer,
        )
        all_results[length_label] = results

    all_results["meta"] = {"target_id": TARGET_ID, "draft_id": DRAFT_ID, "vocab_check": vocab_check}
    with open(RESULTS_FILE, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved to {RESULTS_FILE}", flush=True)


if __name__ == "__main__":
    main()
