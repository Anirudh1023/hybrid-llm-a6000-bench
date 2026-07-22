"""
Core reusable measurement harness for cross-vocab / same-vocab speculative
decoding experiments. Consolidates the pattern rewritten by hand ~7 times
during the T4 session tonight (2026-07-22/23) into one place.

Key lessons baked in from that session, so they don't need rediscovering:
  - Always pin target and assistant to SEPARATE devices (device_map={'':0} /
    {'':1}) if multi-GPU is available -- device_map='auto' cross-GPU layer
    splitting caused a catastrophic 0.2-0.3x slowdown early in that session.
  - torch.cuda.empty_cache() + gc.collect() after every single measurement
    call, not just training steps -- without this, sequential generate()
    calls in one process slowly fragment memory until OOM.
  - Use time.time() wall-clock around the whole generate() call, not
    anything relying on token-count alone -- tok/call and wall-clock time
    can and do diverge (see CONCLUSIONS.md, the reverse-KL wall-clock
    paradox that tok/call alone completely missed).
"""

import gc
import time

import torch


def make_call_counter(target_model):
    """Registers a forward hook that counts target-model forward calls,
    used to compute tokens-per-call (tpc) for a generate() run."""
    call_count = {"n": 0}
    target_model.register_forward_hook(lambda m, i, o: call_count.update(n=call_count["n"] + 1))
    return call_count


def measure_baseline(target_model, target_tokenizer, prompt, max_new=60, max_length=1024):
    """Plain, non-assisted generation -- the speedup denominator."""
    inputs = target_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length).to(target_model.device)
    t0 = time.time()
    with torch.no_grad():
        out = target_model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    elapsed = time.time() - t0
    del out
    gc.collect()
    torch.cuda.empty_cache()
    return elapsed


def measure_sd(
    target_model,
    target_tokenizer,
    assistant_model,
    assistant_tokenizer,
    call_count,
    prompt,
    max_new=60,
    max_length=1024,
    same_tokenizer=None,
):
    """
    Speculative-decoding generation, timed + tokens-per-call measured.

    same_tokenizer: if None, auto-detected via config.vocab_size (see
    vocab_utils.same_tokenizer_path). Pass explicitly to skip the check.
    IMPORTANT lesson from the T4 session: do NOT assume same tokenizer
    CLASS or high vocab-key overlap means HF will use the native path --
    the actual gate is config.vocab_size equality (an embedding-table
    padding artifact), which can differ between checkpoints of the exact
    same tokenizer. Always check via vocab_utils, don't assume.
    """
    inputs = target_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length).to(target_model.device)
    call_count["n"] = 0
    t0 = time.time()
    with torch.no_grad():
        if same_tokenizer:
            out = target_model.generate(
                **inputs, max_new_tokens=max_new, do_sample=False,
                assistant_model=assistant_model,
            )
        else:
            out = target_model.generate(
                **inputs, max_new_tokens=max_new, do_sample=False,
                assistant_model=assistant_model,
                tokenizer=target_tokenizer, assistant_tokenizer=assistant_tokenizer,
            )
    elapsed = time.time() - t0
    n_new = out.shape[1] - inputs["input_ids"].shape[1]
    calls = max(call_count["n"], 1)
    del out
    gc.collect()
    torch.cuda.empty_cache()
    return elapsed, n_new / calls


def set_fixed_draft_length(model, n):
    """Disables HF's adaptive draft-length heuristic (grow +2 on full
    match, shrink -1 otherwise) -- confirmed in the T4 session to
    meaningfully overshoot for a higher-accepting (trained) drafter,
    costing more wall-clock than it saves. Use this to isolate whether
    that heuristic, vs the drafter/training itself, is responsible for a
    given result."""
    model.generation_config.num_assistant_tokens = n
    model.generation_config.num_assistant_tokens_schedule = "constant"


def run_full_comparison(
    target_model, target_tokenizer,
    conditions,   # list of (label, assistant_model, assistant_tokenizer_or_None) tuples
    prompts, max_new=60, same_tokenizer=None,
):
    """
    Runs baseline once, then every (label, model) condition against the
    same prompts, and returns a dict of raw + summary results -- the exact
    shape used by report.py to build the consolidated benchmark report.
    """
    call_count = make_call_counter(target_model)

    print(f"=== BASELINE (no SD), {len(prompts)} prompts ===", flush=True)
    baseline_times = []
    for i, p in enumerate(prompts):
        t = measure_baseline(target_model, target_tokenizer, p, max_new=max_new)
        baseline_times.append(t)
        print(f"  baseline[{i}]: {t:.2f}s", flush=True)
    print(f"baseline avg: {sum(baseline_times) / len(baseline_times):.2f}s", flush=True)

    results = {"baseline_times": baseline_times, "conditions": {}}

    for label, assistant_model, assistant_tokenizer in conditions:
        print(f"=== {label}, {len(prompts)} prompts ===", flush=True)
        times, tpcs = [], []
        for i, p in enumerate(prompts):
            t, tpc = measure_sd(
                target_model, target_tokenizer, assistant_model, assistant_tokenizer,
                call_count, p, max_new=max_new, same_tokenizer=same_tokenizer,
            )
            times.append(t)
            tpcs.append(tpc)
            speedup = baseline_times[i] / t
            print(f"  {label}[{i}]: {t:.2f}s tok/call={tpc:.3f} speedup={speedup:.3f}x", flush=True)
        speedups = [b / t for b, t in zip(baseline_times, times)]
        results["conditions"][label] = {
            "times": times, "tpcs": tpcs, "speedups": speedups,
            "avg_speedup": sum(speedups) / len(speedups),
            "avg_tpc": sum(tpcs) / len(tpcs),
        }
        print(f"{label} avg speedup: {results['conditions'][label]['avg_speedup']:.3f}x, "
              f"avg tok/call: {results['conditions'][label]['avg_tpc']:.3f}", flush=True)

    return results
