"""
Diagnose the tok/call-up-but-wallclock-down paradox from the ratio-fix run.
Reuses the ALREADY-TRAINED adapter (trained_llama68m_reverse_kl_gemma2_9b) --
no retraining, this is eval-only instrumentation.

Two things the original run never measured, that we log here:
  (A) real average draft length K (tokens PROPOSED per round, not accepted)
  (B) a freshly re-measured baseline at each condition's own time-point,
      instead of reusing one baseline captured only at the start
"""
import gc
import json
import time

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import transformers.generation.candidate_generator as cg_mod

DRAFT_ID = "JackFram/llama-68m"
TARGET_ID = "google/gemma-2-9b"
FIXED_DRAFT_LEN = 4
N_BASELINE_CHECK = 8  # small paired baseline re-check per condition

with open("heldout_pairs_groundtruth.json") as f:
    pairs = json.load(f)
prompts = [p["prompt"] for p in pairs]

target_tokenizer = AutoTokenizer.from_pretrained(TARGET_ID)
assistant_tokenizer = AutoTokenizer.from_pretrained(DRAFT_ID)

bnb_config = BitsAndBytesConfig(load_in_8bit=True)
target_model = AutoModelForCausalLM.from_pretrained(TARGET_ID, quantization_config=bnb_config, device_map={"": 0})

# fresh, untouched copy for "untrained"
untrained_model = AutoModelForCausalLM.from_pretrained(DRAFT_ID, dtype=torch.float16, device_map={"": 1})

# separate fresh copy to load the adapter onto + merge, so untrained_model stays pristine
base_for_trained = AutoModelForCausalLM.from_pretrained(DRAFT_ID, dtype=torch.float16, device_map={"": 1})
trained_peft = PeftModel.from_pretrained(base_for_trained, "trained_llama68m_reverse_kl_gemma2_9b")
merged_model = trained_peft.merge_and_unload()
merged_model.eval()

call_count = {"n": 0}
target_model.register_forward_hook(lambda m, i, o: call_count.update(n=call_count["n"] + 1))

# ---- (A) patch candidate generator to log real proposed K per round ----
captured_Ks = []
patched_classes = []
for cls_name in ["AssistedCandidateGenerator", "AssistedCandidateGeneratorDifferentTokenizers"]:
    cls = getattr(cg_mod, cls_name, None)
    if cls is None:
        continue
    orig_fn = cls.get_candidates

    def make_wrapper(orig_fn):
        def wrapper(self, input_ids, is_first_iteration):
            candidate_ids, candidate_logits = orig_fn(self, input_ids, is_first_iteration)
            k = candidate_ids.shape[1] - input_ids.shape[1]
            captured_Ks.append(k)
            return candidate_ids, candidate_logits
        return wrapper

    cls.get_candidates = make_wrapper(orig_fn)
    patched_classes.append((cls, orig_fn))
print(f"patched {len(patched_classes)} candidate generator classes for K-logging", flush=True)


def measure_baseline(prompt, max_new=60):
    inputs = target_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(target_model.device)
    t0 = time.time()
    with torch.no_grad():
        out = target_model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    elapsed = time.time() - t0
    del out
    gc.collect(); torch.cuda.empty_cache()
    return elapsed


def measure_sd(model, prompt, max_new=60):
    inputs = target_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(target_model.device)
    call_count["n"] = 0
    captured_Ks.clear()
    t0 = time.time()
    with torch.no_grad():
        out = target_model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                                     assistant_model=model, tokenizer=target_tokenizer, assistant_tokenizer=assistant_tokenizer)
    elapsed = time.time() - t0
    n_new = out.shape[1] - inputs["input_ids"].shape[1]
    calls = max(call_count["n"], 1)
    avg_k = sum(captured_Ks) / len(captured_Ks) if captured_Ks else float("nan")
    del out
    gc.collect(); torch.cuda.empty_cache()
    return elapsed, n_new / calls, avg_k


def set_fixed_draft_length(model, n):
    model.generation_config.num_assistant_tokens = n
    model.generation_config.num_assistant_tokens_schedule = "constant"


def run_condition(label, model):
    # (B) fresh paired baseline check at THIS time-point, on a small subset
    fresh_baseline = []
    for i in range(N_BASELINE_CHECK):
        t = measure_baseline(prompts[i])
        fresh_baseline.append(t)
    fresh_baseline_avg = sum(fresh_baseline) / len(fresh_baseline)
    print(f"  [{label}] fresh baseline check ({N_BASELINE_CHECK} prompts) avg: {fresh_baseline_avg:.2f}s", flush=True)

    times, tpcs, ks = [], [], []
    for i, p in enumerate(prompts):
        t, tpc, k = measure_sd(model, p)
        times.append(t); tpcs.append(tpc); ks.append(k)
        print(f"  {label}[{i}]: {t:.2f}s tok/call={tpc:.3f} avg_K={k:.2f}", flush=True)
    avg_time = sum(times) / len(times)
    avg_tpc = sum(tpcs) / len(tpcs)
    avg_k = sum(ks) / len(ks)
    print(f"{label.upper()} avg time: {avg_time:.2f}s  avg tok/call: {avg_tpc:.3f}  avg_K: {avg_k:.3f}  "
          f"fresh_baseline_avg: {fresh_baseline_avg:.2f}s", flush=True)
    return {"times": times, "tpc": tpcs, "ks": ks, "fresh_baseline": fresh_baseline,
            "avg_time": avg_time, "avg_tpc": avg_tpc, "avg_k": avg_k, "fresh_baseline_avg": fresh_baseline_avg}


print("=== [1] ORIGINAL baseline (full 30, for reference), no SD ===", flush=True)
orig_baseline_times = []
for i, p in enumerate(prompts):
    t = measure_baseline(p)
    orig_baseline_times.append(t)
orig_baseline_avg = sum(orig_baseline_times) / len(orig_baseline_times)
print(f"orig baseline avg (30 prompts): {orig_baseline_avg:.2f}s", flush=True)

print("=== [2] UNTRAINED, adaptive ===", flush=True)
r_untrained = run_condition("untrained", untrained_model)

print("=== [3] TRAINED (merged, reused adapter), adaptive ===", flush=True)
r_trained = run_condition("trained", merged_model)

print("=== [4] TRAINED (merged) + fixed draft length=4 ===", flush=True)
set_fixed_draft_length(merged_model, FIXED_DRAFT_LEN)
r_trained_fixed = run_condition("trained_fixed", merged_model)

print("\n=== DIAGNOSTIC SUMMARY ===", flush=True)
print(f"{'condition':16s} {'avg_time':>10s} {'avg_tpc':>10s} {'avg_K':>8s} {'local_baseline':>15s} {'speedup(local)':>15s} {'speedup(orig)':>14s}", flush=True)
for label, r in [("untrained", r_untrained), ("trained", r_trained), ("trained_fixed", r_trained_fixed)]:
    speedup_local = r["fresh_baseline_avg"] / r["avg_time"]
    speedup_orig = orig_baseline_avg / r["avg_time"]
    print(f"{label:16s} {r['avg_time']:10.2f} {r['avg_tpc']:10.3f} {r['avg_k']:8.2f} "
          f"{r['fresh_baseline_avg']:15.2f} {speedup_local:15.3f} {speedup_orig:14.3f}", flush=True)

with open("diagnose_wallclock_results.json", "w") as f:
    json.dump({
        "orig_baseline_times": orig_baseline_times,
        "untrained": r_untrained, "trained": r_trained, "trained_fixed": r_trained_fixed,
    }, f)
print("Saved to diagnose_wallclock_results.json", flush=True)
