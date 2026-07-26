"""
Ratio-fix experiment: Llama-68M (drafter) + Gemma-2-9b (target), ~132:1 ratio
-- close to OmniDraft's own validated regime (they use Llama-68M against
Vicuna-7B/Qwen2-7B/Llama3-8B), vs our earlier Qwen3-0.6B setup (~15:1).

Runs, in order:
  1. Baseline (no SD)
  2. Untrained Llama-68M + Gemma-2-9b, SD (adaptive heuristic)
  3. Train a FRESH reverse-KL LoRA adapter (bug-fixed version: -inf masked_fill
     fix from tonight) for this exact pairing
  4. Trained (merged) Llama-68M + Gemma-2-9b, SD (adaptive heuristic)
  5. Trained (merged) + FIXED draft length=4 (conditional diagnostic, run
     regardless so we have the data if #4 regresses)
"""

import gc
import json
import time

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.generation.candidate_generator import AssistantVocabTranslatorCache

DRAFT_ID = "JackFram/llama-68m"
TARGET_ID = "google/gemma-2-9b"
LAMBDA_NGRAM = 0.2
FIXED_DRAFT_LEN = 4

with open("heldout_pairs_groundtruth.json") as f:
    pairs = json.load(f)
prompts = [p["prompt"] for p in pairs]
with open("train_prompts.json") as f:
    train_prompts = json.load(f)
print(f"prompts: {len(prompts)}  train_prompts: {len(train_prompts)}", flush=True)

target_tokenizer = AutoTokenizer.from_pretrained(TARGET_ID)
assistant_tokenizer = AutoTokenizer.from_pretrained(DRAFT_ID)

bnb_config = BitsAndBytesConfig(load_in_8bit=True)
target_model = AutoModelForCausalLM.from_pretrained(TARGET_ID, quantization_config=bnb_config, device_map={"": 0})
base_assistant = AutoModelForCausalLM.from_pretrained(DRAFT_ID, dtype=torch.float16, device_map={"": 1})

draft_params = sum(p.numel() for p in base_assistant.parameters())
target_params_approx = 9_000_000_000
print(f"draft params: {draft_params/1e9:.4f}B, ratio ~= {target_params_approx/draft_params:.1f}:1", flush=True)

call_count = {"n": 0}
target_model.register_forward_hook(lambda m, i, o: call_count.update(n=call_count["n"] + 1))


def set_fixed_draft_length(model, n):
    model.generation_config.num_assistant_tokens = n
    model.generation_config.num_assistant_tokens_schedule = "constant"


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
    t0 = time.time()
    with torch.no_grad():
        out = target_model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                                     assistant_model=model, tokenizer=target_tokenizer, assistant_tokenizer=assistant_tokenizer)
    elapsed = time.time() - t0
    n_new = out.shape[1] - inputs["input_ids"].shape[1]
    calls = max(call_count["n"], 1)
    del out
    gc.collect(); torch.cuda.empty_cache()
    return elapsed, n_new / calls


def run_condition(label, model, prompts_list, baseline_times=None):
    times, tpcs = [], []
    for i, p in enumerate(prompts_list):
        t, tpc = measure_sd(model, p)
        times.append(t); tpcs.append(tpc)
        if baseline_times is not None:
            print(f"  {label}[{i}]: {t:.2f}s tok/call={tpc:.3f} speedup={baseline_times[i]/t:.3f}x", flush=True)
        else:
            print(f"  {label}[{i}]: {t:.2f}s tok/call={tpc:.3f}", flush=True)
    return times, tpcs


print("=== [1] BASELINE (no SD), 30 prompts ===", flush=True)
baseline_times = []
for i, p in enumerate(prompts):
    t = measure_baseline(p)
    baseline_times.append(t)
    print(f"  baseline[{i}]: {t:.2f}s", flush=True)
print(f"baseline avg: {sum(baseline_times)/len(baseline_times):.2f}s", flush=True)

print("=== [2] UNTRAINED Llama-68M + Gemma-2-9b, SD (adaptive), 30 prompts ===", flush=True)
untrained_times, untrained_tpc = run_condition("untrained", base_assistant, prompts, baseline_times)
untrained_speedups = [b/t for b, t in zip(baseline_times, untrained_times)]
print(f"UNTRAINED avg speedup: {sum(untrained_speedups)/len(untrained_speedups):.3f}x  "
      f"avg tok/call: {sum(untrained_tpc)/len(untrained_tpc):.3f}", flush=True)

# ============ [3] TRAIN FRESH REVERSE-KL LORA ADAPTER ============
print("=== [3] TRAINING fresh reverse-KL LoRA adapter ===", flush=True)
lora_config = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05,
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM")
assistant_model = get_peft_model(base_assistant, lora_config)
assistant_model.print_trainable_parameters()

import bitsandbytes as bnb
optimizer = bnb.optim.AdamW8bit(assistant_model.parameters(), lr=1e-4)

translator = AssistantVocabTranslatorCache.get_translator(
    target_tokenizer, assistant_tokenizer, target_model.config.vocab_size,
    assistant_model=base_assistant, assistant_prune_lm_head=False,
)
target_to_assistant = translator.target_to_assistant_input_ids


def get_target_continuation(prompt, max_new=60):
    inputs = target_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(target_model.device)
    with torch.no_grad():
        out = target_model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    cont_ids = out[0][inputs["input_ids"].shape[1]:]
    del out
    gc.collect(); torch.cuda.empty_cache()
    return cont_ids


def train_step(prompt, target_cont_ids):
    aligned_assistant_ids, aligned_target_positions = [], []
    for i, tid in enumerate(target_cont_ids.tolist()):
        if tid in target_to_assistant:
            aligned_assistant_ids.append(target_to_assistant[tid])
            aligned_target_positions.append(i)
    if not aligned_assistant_ids:
        return None

    target_prompt_ids = target_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).input_ids.to(target_model.device)
    target_full_ids = torch.cat([target_prompt_ids, target_cont_ids.unsqueeze(0).to(target_model.device)], dim=1)
    with torch.no_grad():
        target_out = target_model(input_ids=target_full_ids)
    target_logits_full = target_out.logits[0, target_prompt_ids.shape[1]-1:-1, :]
    del target_out
    gc.collect(); torch.cuda.empty_cache()

    assistant_prompt_ids = assistant_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).input_ids.to(assistant_model.device)
    aligned_cont_ids = torch.tensor(aligned_assistant_ids, device=assistant_model.device).unsqueeze(0)
    assistant_full_ids = torch.cat([assistant_prompt_ids, aligned_cont_ids], dim=1)
    assistant_model.train()
    assistant_out = assistant_model(input_ids=assistant_full_ids)
    assistant_logits_full = assistant_out.logits[0, assistant_prompt_ids.shape[1]-1:-1, :]

    projected = translator.get_target_logits(assistant_logits_full.float().unsqueeze(0))[0]
    target_logits_aligned = target_logits_full[aligned_target_positions].to(assistant_model.device)
    valid_mask = projected[0] != translator.FILTER_VALUE
    projected_safe = projected.masked_fill(projected == translator.FILTER_VALUE, -1e4)
    q = F.log_softmax(projected_safe, dim=-1)
    p = F.softmax(target_logits_aligned.float(), dim=-1)
    kl_per_pos = (q.exp() * (q - torch.log(p.clamp_min(1e-12)))) * valid_mask.unsqueeze(0)
    loss = kl_per_pos.sum(dim=-1).mean()

    if torch.isnan(loss) or torch.isinf(loss):
        assistant_model.eval()
        gc.collect(); torch.cuda.empty_cache()
        return None
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    assistant_model.eval()
    gc.collect(); torch.cuda.empty_cache()
    return loss.item()


losses, skipped = [], 0
for i, p in enumerate(train_prompts):
    cont_ids = get_target_continuation(p)
    loss = train_step(p, cont_ids)
    if loss is None:
        skipped += 1
        continue
    losses.append(loss)
    if i < 3 or i % 20 == 0:
        print(f"  [{i}/{len(train_prompts)}] loss: {loss:.4f}", flush=True)
print(f"Training done, skipped {skipped}/{len(train_prompts)}, "
      f"avg loss: {(sum(losses)/len(losses)) if losses else float('nan'):.4f}", flush=True)
assistant_model.save_pretrained("trained_llama68m_reverse_kl_gemma2_9b")
print("Saved adapter", flush=True)

print("=== MERGING LoRA adapter ===", flush=True)
merged_model = assistant_model.merge_and_unload()
merged_model.eval()

print("=== [4] TRAINED (merged) Llama-68M + Gemma-2-9b, SD (adaptive), 30 prompts ===", flush=True)
trained_times, trained_tpc = run_condition("trained", merged_model, prompts, baseline_times)
trained_speedups = [b/t for b, t in zip(baseline_times, trained_times)]
print(f"TRAINED avg speedup: {sum(trained_speedups)/len(trained_speedups):.3f}x  "
      f"avg tok/call: {sum(trained_tpc)/len(trained_tpc):.3f}", flush=True)

print(f"=== [5] TRAINED (merged) + FIXED draft length={FIXED_DRAFT_LEN}, 30 prompts ===", flush=True)
set_fixed_draft_length(merged_model, FIXED_DRAFT_LEN)
trained_fixed_times, trained_fixed_tpc = run_condition("trained_fixed", merged_model, prompts, baseline_times)
trained_fixed_speedups = [b/t for b, t in zip(baseline_times, trained_fixed_times)]
print(f"TRAINED+FIXED avg speedup: {sum(trained_fixed_speedups)/len(trained_fixed_speedups):.3f}x  "
      f"avg tok/call: {sum(trained_fixed_tpc)/len(trained_fixed_tpc):.3f}", flush=True)

print("\n=== FINAL SUMMARY ===", flush=True)
print(f"{'condition':30s} {'avg speedup':>12s} {'avg tok/call':>14s}", flush=True)
print(f"{'untrained (adaptive)':30s} {sum(untrained_speedups)/len(untrained_speedups):12.3f} {sum(untrained_tpc)/len(untrained_tpc):14.3f}", flush=True)
print(f"{'trained merged (adaptive)':30s} {sum(trained_speedups)/len(trained_speedups):12.3f} {sum(trained_tpc)/len(trained_tpc):14.3f}", flush=True)
print(f"{'trained merged (fixed=4)':30s} {sum(trained_fixed_speedups)/len(trained_fixed_speedups):12.3f} {sum(trained_fixed_tpc)/len(trained_fixed_tpc):14.3f}", flush=True)

with open("ratio_fix_results.json", "w") as f:
    json.dump({
        "baseline_times": baseline_times,
        "untrained_times": untrained_times, "untrained_tpc": untrained_tpc,
        "trained_times": trained_times, "trained_tpc": trained_tpc,
        "trained_fixed_times": trained_fixed_times, "trained_fixed_tpc": trained_fixed_tpc,
        "draft_params": draft_params,
    }, f)
print("Saved to ratio_fix_results.json", flush=True)
