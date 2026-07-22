"""
Experiment 1: TRUE large-ratio test with a genuinely large (30B+) target.

Why this needs the A6000, not the T4 setup: the T4 rig (2x 15.6GB) could not
load anything bigger than ~9B even in 8-bit. The literature precedent for
cross-family SD (Qwen2-0.5B + Mixtral-8x22B/Llama-3.1-70B/Phi-3-medium,
1.78-1.91x speedup) uses 28:1-280:1 ratios -- our T4 tests only reached
15-18.2:1, and every additional bit of ratio we DID test (11.7 -> 15 -> 18.2)
showed a clean, monotonic improvement. This experiment extends that curve
into the range the literature actually validated.

VRAM budget on a single A6000 (48GB):
  - Qwen2.5-32B-Instruct in 8-bit: ~32GB + small drafter + KV cache -- fits,
    but tight. Recommend 8-bit (default here), not fp16 (32B fp16 = ~64GB,
    will NOT fit on one card).
  - Gemma-2-27B in 8-bit: ~27GB -- more headroom, a good alternative if you
    want to extend the SAME target family (Gemma-2) already validated on T4
    rather than switching families.

Drafter choices, to get a genuinely large ratio (not the marginal 0.5B vs
0.6B non-difference the T4 session got called out for):
  - Qwen2.5-0.5B-Instruct (drafter) + Qwen2.5-32B-Instruct (target): ~64:1
  - Or reuse Llama-68M (drafter) + either target: ~440:1 (32B) or ~400:1
    (27B) -- genuinely at the literature's tested scale.

IMPORTANT: check vocab_utils.check_native_path() before assuming anything
about same-family vs cross-family behavior for whatever pair you pick.
"""

import json
import sys

sys.path.insert(0, "../src")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from harness import run_full_comparison
from vocab_utils import check_native_path

TARGET_ID = "Qwen/Qwen2.5-32B-Instruct"   # or "google/gemma-2-27b" -- see docstring
DRAFT_ID = "JackFram/llama-68m"            # genuinely large ratio; swap for Qwen2.5-0.5B-Instruct for a milder one
PROMPTS_FILE = "../data/heldout_pairs_groundtruth.json"
RESULTS_FILE = "../results/exp01_large_target_ratio.json"


def main():
    with open(PROMPTS_FILE) as f:
        pairs = json.load(f)
    prompts = [p["prompt"] if isinstance(p, dict) else p for p in pairs]

    vocab_check = check_native_path(TARGET_ID, DRAFT_ID)
    print(f"Vocab check: {vocab_check}", flush=True)
    same_tokenizer = vocab_check["native_path"]

    target_tokenizer = AutoTokenizer.from_pretrained(TARGET_ID)
    assistant_tokenizer = AutoTokenizer.from_pretrained(DRAFT_ID)

    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    target_model = AutoModelForCausalLM.from_pretrained(TARGET_ID, quantization_config=bnb_config, device_map={"": 0})
    draft_model = AutoModelForCausalLM.from_pretrained(DRAFT_ID, dtype=torch.float16, device_map={"": 0})

    draft_params = sum(p.numel() for p in draft_model.parameters())
    target_params = sum(p.numel() for p in target_model.parameters())
    print(f"draft: {draft_params / 1e9:.3f}B  target: {target_params / 1e9:.3f}B  "
          f"ratio: {target_params / draft_params:.1f}:1", flush=True)

    results = run_full_comparison(
        target_model, target_tokenizer,
        conditions=[("sd", draft_model, None if same_tokenizer else assistant_tokenizer)],
        prompts=prompts, same_tokenizer=same_tokenizer,
    )
    results["meta"] = {
        "target_id": TARGET_ID, "draft_id": DRAFT_ID,
        "ratio": target_params / draft_params, "vocab_check": vocab_check,
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {RESULTS_FILE}", flush=True)


if __name__ == "__main__":
    main()
