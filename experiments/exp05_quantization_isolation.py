"""
Experiment 5: isolate quantization overhead cleanly -- something the T4 rig
could never do, since 8-bit was mandatory there just to fit a 9B target in
14.56GB usable VRAM.

Background: 8-bit (bitsandbytes LLM.int8()) was flagged repeatedly all T4
session as a real, measurable drag -- slower per-token than fp16 for
autoregressive decode (int8 mainly saves memory, not necessarily speed, and
LLM.int8()'s outlier-decomposition path doesn't get the same tensor-core
speedup fp16 does). But every T4 test HAD to use it, so its exact
contribution to any given slowdown was always a suspect, never isolated.

On a 48GB A6000, a mid-size target (up to ~20B in fp16, ~40GB) can run
UNQUANTIZED. This experiment runs the SAME target model in both fp16 and
8-bit, same drafter, same prompts, to get a clean, direct measurement of
quantization's actual cost on real hardware -- separate from every other
confound this session chased (adaptive draft length, LoRA overhead, n-gram
cache gaps).

Pick a target small enough to fit fp16 comfortably (e.g. Qwen2.5-14B-Instruct,
~28GB fp16) -- NOT the 30B+ target from exp01, which needs quantization to
fit regardless.
"""

import json
import sys

sys.path.insert(0, "../src")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from harness import run_full_comparison
from vocab_utils import check_native_path

TARGET_ID = "Qwen/Qwen2.5-14B-Instruct"  # fits fp16 on a single A6000 (~28GB)
DRAFT_ID = "Qwen/Qwen2.5-0.5B-Instruct"

PROMPTS_FILE = "../data/heldout_pairs_groundtruth.json"
RESULTS_FILE = "../results/exp05_quantization_isolation.json"


def run_condition(target_dtype_or_8bit, label, prompts, same_tokenizer, assistant_tokenizer):
    target_tokenizer = AutoTokenizer.from_pretrained(TARGET_ID)
    if target_dtype_or_8bit == "8bit":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        target_model = AutoModelForCausalLM.from_pretrained(TARGET_ID, quantization_config=bnb_config, device_map={"": 0})
    else:
        target_model = AutoModelForCausalLM.from_pretrained(TARGET_ID, dtype=torch.float16, device_map={"": 0})
    draft_model = AutoModelForCausalLM.from_pretrained(DRAFT_ID, dtype=torch.float16, device_map={"": 1})

    results = run_full_comparison(
        target_model, target_tokenizer,
        conditions=[(label, draft_model, None if same_tokenizer else assistant_tokenizer)],
        prompts=prompts, same_tokenizer=same_tokenizer,
    )
    del target_model, draft_model
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return results


def main():
    with open(PROMPTS_FILE) as f:
        pairs = json.load(f)
    prompts = [p["prompt"] if isinstance(p, dict) else p for p in pairs]

    vocab_check = check_native_path(TARGET_ID, DRAFT_ID)
    same_tokenizer = vocab_check["native_path"]
    assistant_tokenizer = AutoTokenizer.from_pretrained(DRAFT_ID)
    print(f"Vocab check: {vocab_check}", flush=True)

    print("=== fp16 target (unquantized) ===", flush=True)
    results_fp16 = run_condition("fp16", "fp16_sd", prompts, same_tokenizer, assistant_tokenizer)

    print("=== 8-bit target (bitsandbytes) ===", flush=True)
    results_8bit = run_condition("8bit", "8bit_sd", prompts, same_tokenizer, assistant_tokenizer)

    combined = {"fp16": results_fp16, "8bit": results_8bit,
                "meta": {"target_id": TARGET_ID, "draft_id": DRAFT_ID, "vocab_check": vocab_check}}
    with open(RESULTS_FILE, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"Saved to {RESULTS_FILE}", flush=True)


if __name__ == "__main__":
    main()
