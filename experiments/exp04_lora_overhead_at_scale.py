"""
Experiment 4: does LoRA's ~44% per-forward-pass overhead matter less at
scale, in relative terms?

Background: on the T4, a LoRA-wrapped Qwen3-0.6B drafter ran 1.443x the
wall-clock time of the base model, in ISOLATED drafter-only generation (no
SD at all, no target model involved). merge_and_unload() eliminated this
entirely (merged model ran at 0.976x of base, i.e. fully recovered). This
was a big part of why the trained drafter underperformed in the full SD
comparison even after fixing draft length.

Two things worth checking at A6000 scale:
  1. Does the SAME ~44% relative overhead reproduce on a different/bigger
     drafter, or was it specific to Qwen3-0.6B's architecture/LoRA config?
  2. In the FULL SD comparison (not drafter-only), does the drafter's
     absolute contribution to per-round cost shrink as the TARGET gets much
     bigger (32B) -- meaning the same 44% relative drafter-only overhead
     might translate into a much SMALLER fraction of total round cost, and
     therefore matter less for the trained-vs-untrained wall-clock gap?

Always run this drafter-only isolation BEFORE any full SD comparison
involving a trained adapter -- it's cheap (no target model needed) and
tells you immediately whether to expect a LoRA-overhead confound.
"""

import json
import sys

sys.path.insert(0, "../src")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from lora_utils import load_and_merge

DRAFT_ID = "Qwen/Qwen2.5-0.5B-Instruct"
TRAINED_ADAPTER_DIR = "trained_drafter"
PROMPTS_FILE = "../data/heldout_pairs_groundtruth.json"
RESULTS_FILE = "../results/exp04_lora_overhead_at_scale.json"


def measure(model, tokenizer, prompt, max_new=60):
    import gc
    import time
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    elapsed = time.time() - t0
    del out
    gc.collect()
    torch.cuda.empty_cache()
    return elapsed


def main():
    with open(PROMPTS_FILE) as f:
        pairs = json.load(f)
    prompts = [p["prompt"] if isinstance(p, dict) else p for p in pairs]

    tokenizer = AutoTokenizer.from_pretrained(DRAFT_ID)
    base_model = AutoModelForCausalLM.from_pretrained(DRAFT_ID, dtype=torch.float16, device_map={"": 0})

    print("=== BASE (no LoRA), plain generation ===", flush=True)
    base_times = [measure(base_model, tokenizer, p) for p in prompts]
    print(f"base avg: {sum(base_times) / len(base_times):.3f}s", flush=True)

    lora_model = PeftModel.from_pretrained(base_model, TRAINED_ADAPTER_DIR)
    print("=== UNMERGED LoRA, plain generation ===", flush=True)
    lora_times = [measure(lora_model, tokenizer, p) for p in prompts]
    print(f"unmerged avg: {sum(lora_times) / len(lora_times):.3f}s "
          f"({(sum(lora_times) / len(lora_times)) / (sum(base_times) / len(base_times)):.3f}x base)", flush=True)

    merged_model = lora_model.merge_and_unload()
    merged_model.eval()
    print("=== MERGED LoRA, plain generation ===", flush=True)
    merged_times = [measure(merged_model, tokenizer, p) for p in prompts]
    print(f"merged avg: {sum(merged_times) / len(merged_times):.3f}s "
          f"({(sum(merged_times) / len(merged_times)) / (sum(base_times) / len(base_times)):.3f}x base)", flush=True)

    results = {"base_times": base_times, "lora_times": lora_times, "merged_times": merged_times,
               "meta": {"draft_id": DRAFT_ID}}
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {RESULTS_FILE}", flush=True)


if __name__ == "__main__":
    main()
