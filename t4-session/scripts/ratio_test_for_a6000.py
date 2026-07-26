"""
Ratio test: Llama-68M drafter (~68M params) vs google/gemma-2-9b target (~9B),
ratio ~132:1 -- much closer to the literature-tested regime (28:1-280:1) than
anything reachable on 2x T4 by shrinking the Qwen drafter (0.5B is already
close to the floor for that family).

SETUP (run once):
    pip install -q -U transformers accelerate bitsandbytes peft torchao
    huggingface-cli login   # needs a token with Gemma license accepted

DATA NEEDED: heldout_pairs_groundtruth.json (30 LoCoMo prompts) -- copy this
file alongside this script, or regenerate via:
    git clone --depth 1 https://github.com/snap-research/locomo.git
    # then adapt prep_locomo_data.py's extract_pairs_from_conv() to dump
    # just the 'prompt' field if you don't have the exact file handy --
    # any ~30 real conversational prompts work, exact match to our set
    # only matters if you want directly comparable numbers.

On a single A6000 (48GB), you likely do NOT need 8-bit quantization for the
target at all -- fp16 gemma-2-9b is ~18GB, fits comfortably alongside a tiny
68M drafter. Recommend trying fp16 first (remove BitsAndBytesConfig / load
target in dtype=torch.float16 instead) since 8-bit was a real confound in our
own results tonight (dequant overhead, adaptive-draft-length interaction) --
worth seeing if the ratio effect looks different / cleaner without that
confound stacked on top.

Run:
    python3 ratio_test_for_a6000.py
"""

import torch, json, gc, time, statistics as stats
from transformers import AutoModelForCausalLM, AutoTokenizer

DRAFT_ID  = 'JackFram/llama-68m'
TARGET_ID = 'google/gemma-2-9b'
USE_8BIT  = False  # flip to True to match our T4 setup exactly; False recommended on A6000 (see docstring)

with open('heldout_pairs_groundtruth.json') as f:
    pairs = json.load(f)
prompts = [p['prompt'] if isinstance(p, dict) else p for p in pairs]
print(f'prompts: {len(prompts)}', flush=True)

target_tokenizer    = AutoTokenizer.from_pretrained(TARGET_ID)
assistant_tokenizer = AutoTokenizer.from_pretrained(DRAFT_ID)

if USE_8BIT:
    from transformers import BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    target_model = AutoModelForCausalLM.from_pretrained(TARGET_ID, quantization_config=bnb_config, device_map={'': 0})
else:
    target_model = AutoModelForCausalLM.from_pretrained(TARGET_ID, dtype=torch.float16, device_map={'': 0})

draft_model = AutoModelForCausalLM.from_pretrained(DRAFT_ID, dtype=torch.float16, device_map={'': 0})

draft_params = sum(p.numel() for p in draft_model.parameters())
target_params_approx = 9_000_000_000
print(f'draft params: {draft_params/1e9:.4f}B, approx ratio target:draft = {target_params_approx/draft_params:.1f}:1', flush=True)

call_count = {'n': 0}
target_model.register_forward_hook(lambda m, i, o: call_count.update(n=call_count['n'] + 1))


def measure_baseline(prompt, max_new=60):
    inputs = target_tokenizer(prompt, return_tensors='pt', truncation=True, max_length=1024).to(target_model.device)
    t0 = time.time()
    with torch.no_grad():
        out = target_model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    elapsed = time.time() - t0
    del out
    gc.collect(); torch.cuda.empty_cache()
    return elapsed


def measure_sd(prompt, max_new=60):
    inputs = target_tokenizer(prompt, return_tensors='pt', truncation=True, max_length=1024).to(target_model.device)
    call_count['n'] = 0
    t0 = time.time()
    with torch.no_grad():
        out = target_model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                                     assistant_model=draft_model, tokenizer=target_tokenizer, assistant_tokenizer=assistant_tokenizer)
    elapsed = time.time() - t0
    n_new = out.shape[1] - inputs['input_ids'].shape[1]
    calls = max(call_count['n'], 1)
    del out
    gc.collect(); torch.cuda.empty_cache()
    return elapsed, n_new / calls


print('=== BASELINE (no SD), 30 prompts ===', flush=True)
baseline_times = []
for i, p in enumerate(prompts):
    t = measure_baseline(p)
    baseline_times.append(t)
    print(f'  baseline[{i}]: {t:.2f}s', flush=True)
print(f'baseline avg: {sum(baseline_times)/len(baseline_times):.2f}s', flush=True)

print('=== Llama-68M drafter + SD, 30 prompts ===', flush=True)
sd_times, sd_tpc = [], []
for i, p in enumerate(prompts):
    t, tpc = measure_sd(p)
    sd_times.append(t)
    sd_tpc.append(tpc)
    speedup = baseline_times[i] / t
    print(f'  sd[{i}]: {t:.2f}s tok/call={tpc:.3f} speedup={speedup:.3f}x', flush=True)

speedups = [b / s for b, s in zip(baseline_times, sd_times)]
print(f'\n=== FINAL SUMMARY (Llama-68M drafter, ~{target_params_approx/draft_params:.0f}:1 ratio, 8bit={USE_8BIT}) ===', flush=True)
print(f'avg speedup: {sum(speedups)/len(speedups):.3f}', flush=True)
print(f'median speedup: {stats.median(speedups):.3f}', flush=True)
print(f'avg tok/call: {sum(sd_tpc)/len(sd_tpc):.3f}', flush=True)
below_1 = sum(1 for s in speedups if s < 1.0)
print(f'prompts below 1.0x: {below_1}/{len(speedups)}', flush=True)

with open('ratio_test_results.json', 'w') as f:
    json.dump({'baseline_times': baseline_times, 'sd_times': sd_times, 'sd_tpc': sd_tpc, 'draft_params': draft_params, 'use_8bit': USE_8BIT}, f)
print('Saved to ratio_test_results.json', flush=True)
