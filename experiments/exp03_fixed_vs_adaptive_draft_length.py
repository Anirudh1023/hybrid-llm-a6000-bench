"""
Experiment 3: does the adaptive-draft-length overshoot problem get better
or worse at bigger target scale?

Background: on the T4 (Qwen3-0.6B + Gemma-2-9b), fixing draft length
(disabling HF's adaptive heuristic) recovered a large chunk of the trained
drafter's wall-clock regression (1.042x -> 1.210x), but not all of it.
Separately, we reasoned (not measured) that bigger targets are MORE
memory-bandwidth-bound per forward pass, so a slightly-too-long verification
window should cost proportionally LESS marginal time relative to the fixed
cost of loading a much bigger model's weights -- meaning this failure mode
might matter LESS at 30B+ scale, not more. This experiment tests that directly.

Requires a trained (reverse-KL) adapter for this target -- run
reverse_kl_train.py first, or point TRAINED_ADAPTER_DIR at one you already
have. ALSO merge the adapter first (see lora_utils.py / exp04) -- otherwise
you'll be re-measuring the LoRA-overhead confound, not the draft-length one.
"""

import json
import sys

sys.path.insert(0, "../src")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from harness import run_full_comparison, set_fixed_draft_length
from lora_utils import load_and_merge
from vocab_utils import check_native_path

TARGET_ID = "Qwen/Qwen2.5-32B-Instruct"
DRAFT_ID = "Qwen/Qwen2.5-0.5B-Instruct"
TRAINED_ADAPTER_DIR = "trained_drafter"  # from reverse_kl_train.py, run first
FIXED_DRAFT_LEN = 4

PROMPTS_FILE = "../data/heldout_pairs_groundtruth.json"
RESULTS_FILE = "../results/exp03_fixed_vs_adaptive_draft_length.json"


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
    base_assistant = AutoModelForCausalLM.from_pretrained(DRAFT_ID, dtype=torch.float16, device_map={"": 0})

    # Merged trained model -- avoids re-measuring the LoRA-overhead confound
    trained_model = load_and_merge(base_assistant, TRAINED_ADAPTER_DIR)

    print("=== ADAPTIVE draft length (default) ===", flush=True)
    results_adaptive = run_full_comparison(
        target_model, target_tokenizer,
        conditions=[
            ("untrained_adaptive", base_assistant, None if same_tokenizer else assistant_tokenizer),
            ("trained_adaptive", trained_model, None if same_tokenizer else assistant_tokenizer),
        ],
        prompts=prompts, same_tokenizer=same_tokenizer,
    )

    set_fixed_draft_length(base_assistant, FIXED_DRAFT_LEN)
    set_fixed_draft_length(trained_model, FIXED_DRAFT_LEN)

    print(f"=== FIXED draft length={FIXED_DRAFT_LEN} ===", flush=True)
    results_fixed = run_full_comparison(
        target_model, target_tokenizer,
        conditions=[
            ("untrained_fixed", base_assistant, None if same_tokenizer else assistant_tokenizer),
            ("trained_fixed", trained_model, None if same_tokenizer else assistant_tokenizer),
        ],
        prompts=prompts, same_tokenizer=same_tokenizer,
    )

    combined = {"adaptive": results_adaptive, "fixed": results_fixed,
                "meta": {"target_id": TARGET_ID, "draft_id": DRAFT_ID, "fixed_draft_len": FIXED_DRAFT_LEN}}
    with open(RESULTS_FILE, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"Saved to {RESULTS_FILE}", flush=True)


if __name__ == "__main__":
    main()
