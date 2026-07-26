"""
Acceptance-head test on the CORRECT ratio pairing (Llama-68M / Gemma-2-9b,
~132:1), following tonight's diagnostic finding: reverse-KL-only training
makes the adaptive heuristic inflate avg_K (1.83 -> 3.38 proposed tokens/round)
for almost no gain in tok/call (1.854 -> 1.878), tanking wall-clock despite
looking better on tok/call alone.

This script:
  1. measures a fresh baseline (30 prompts, no SD) for this run
  2. jointly trains reverse-KL + acceptance head (BCE on reject/accept label)
     on Llama-68M, LoRA r=32 q/k/v/o -- same recipe as the earlier Qwen3-0.6B
     run, just the correct drafter this time
  3. merges LoRA
  4. tests v2 gating: the trained head predicts reject-prob per round from
     hidden states captured for FREE during the assistant's own drafting call
     (no extra forward pass -- the v1 mistake), and directly controls
     num_assistant_tokens instead of HF's blind +2/-1 heuristic
  5. logs avg_K (proposed) alongside tok/call and wall-clock, same
     instrumentation as the diagnostic run, so we can see directly whether
     gating brings K back down to where it pays off, instead of just
     reporting speedup numbers with no visibility into the mechanism
"""
import gc
import json
import time
import types

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.generation.candidate_generator import (
    AssistantVocabTranslatorCache, AssistedCandidateGeneratorDifferentTokenizers,
)

DRAFT_ID = "JackFram/llama-68m"
TARGET_ID = "google/gemma-2-9b"
LAMBDA_ACCEPT = 0.3
REJECT_THRESHOLD = 0.5
MIN_DRAFT, MAX_DRAFT = 1, 8

with open("train_prompts.json") as f:
    train_prompts = json.load(f)
with open("heldout_pairs_groundtruth.json") as f:
    heldout_pairs = json.load(f)
prompts = [p["prompt"] for p in heldout_pairs]
print(f"train: {len(train_prompts)}  heldout: {len(prompts)}", flush=True)

target_tokenizer = AutoTokenizer.from_pretrained(TARGET_ID)
assistant_tokenizer = AutoTokenizer.from_pretrained(DRAFT_ID)

bnb_config = BitsAndBytesConfig(load_in_8bit=True)
target_model = AutoModelForCausalLM.from_pretrained(TARGET_ID, quantization_config=bnb_config, device_map={"": 0})
base_assistant = AutoModelForCausalLM.from_pretrained(DRAFT_ID, dtype=torch.float16, device_map={"": 1})

call_count = {"n": 0}
target_model.register_forward_hook(lambda m, i, o: call_count.update(n=call_count["n"] + 1))


def measure_baseline(prompt, max_new=60):
    inputs = target_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(target_model.device)
    t0 = time.time()
    with torch.no_grad():
        out = target_model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    elapsed = time.time() - t0
    del out
    gc.collect(); torch.cuda.empty_cache()
    return elapsed


print("=== [1] FRESH BASELINE (no SD), 30 prompts ===", flush=True)
baseline_times = []
for i, p in enumerate(prompts):
    t = measure_baseline(p)
    baseline_times.append(t)
baseline_avg = sum(baseline_times) / len(baseline_times)
print(f"baseline avg: {baseline_avg:.2f}s", flush=True)

# ============ [2] JOINT REVERSE-KL + ACCEPTANCE HEAD TRAINING ============
print("=== [2] TRAINING (reverse-KL + acceptance head, joint) ===", flush=True)
lora_config = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05,
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM")
assistant_model = get_peft_model(base_assistant, lora_config)
assistant_model.print_trainable_parameters()

hidden_size = base_assistant.config.hidden_size
acceptance_head = nn.Sequential(
    nn.Linear(hidden_size, hidden_size // 4),
    nn.ReLU(),
    nn.Linear(hidden_size // 4, 1),
).to(assistant_model.device).to(torch.float16)
print(f"acceptance head params: {sum(p.numel() for p in acceptance_head.parameters())}", flush=True)

import bitsandbytes as bnb
optimizer = bnb.optim.AdamW8bit(list(assistant_model.parameters()) + list(acceptance_head.parameters()), lr=1e-4)

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
    target_logits_full = target_out.logits[0, target_prompt_ids.shape[1] - 1:-1, :]
    target_argmax_full = target_logits_full.argmax(dim=-1).to(assistant_model.device)
    del target_out
    gc.collect(); torch.cuda.empty_cache()

    assistant_prompt_ids = assistant_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).input_ids.to(assistant_model.device)
    aligned_cont_ids = torch.tensor(aligned_assistant_ids, device=assistant_model.device).unsqueeze(0)
    assistant_full_ids = torch.cat([assistant_prompt_ids, aligned_cont_ids], dim=1)

    assistant_model.train()
    assistant_out = assistant_model(input_ids=assistant_full_ids, output_hidden_states=True)
    assistant_logits_full = assistant_out.logits[0, assistant_prompt_ids.shape[1] - 1:-1, :]
    assistant_hidden_full = assistant_out.hidden_states[-1][0, assistant_prompt_ids.shape[1] - 1:-1, :]

    projected = translator.get_target_logits(assistant_logits_full.float().unsqueeze(0))[0]
    target_logits_aligned = target_logits_full[aligned_target_positions].to(assistant_model.device)
    valid_mask = projected[0] != translator.FILTER_VALUE
    projected_safe = projected.masked_fill(projected == translator.FILTER_VALUE, -1e4)
    q = F.log_softmax(projected_safe, dim=-1)
    p = F.softmax(target_logits_aligned.float(), dim=-1)
    kl_per_pos = (q.exp() * (q - torch.log(p.clamp_min(1e-12)))) * valid_mask.unsqueeze(0)
    loss_dm = kl_per_pos.sum(dim=-1).mean()

    assistant_argmax_in_target_space = projected.argmax(dim=-1)
    target_argmax_aligned = target_argmax_full[aligned_target_positions]
    reject_labels = (assistant_argmax_in_target_space != target_argmax_aligned).float()

    accept_logits = acceptance_head(assistant_hidden_full).squeeze(-1)
    loss_accept = F.binary_cross_entropy_with_logits(accept_logits, reject_labels)

    loss = loss_dm + LAMBDA_ACCEPT * loss_accept
    if torch.isnan(loss) or torch.isinf(loss):
        assistant_model.eval()
        gc.collect(); torch.cuda.empty_cache()
        return None
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    assistant_model.eval()
    gc.collect(); torch.cuda.empty_cache()

    with torch.no_grad():
        pred_reject = (torch.sigmoid(accept_logits) > 0.5).float()
        acc = (pred_reject == reject_labels).float().mean().item()
    return {"loss_dm": loss_dm.item(), "loss_accept": loss_accept.item(), "head_accuracy": acc,
            "n_reject_labels": int(reject_labels.sum().item()), "n_total": len(reject_labels)}


stats_log, skipped = [], 0
for i, p in enumerate(train_prompts):
    cont_ids = get_target_continuation(p)
    stats = train_step(p, cont_ids)
    if stats is None:
        skipped += 1
        continue
    stats_log.append(stats)
    if i < 3 or i % 20 == 0:
        print(f"  [{i}/{len(train_prompts)}] loss_dm={stats['loss_dm']:.4f} "
              f"loss_accept={stats['loss_accept']:.4f} head_acc={stats['head_accuracy']:.3f} "
              f"(reject rate: {stats['n_reject_labels']}/{stats['n_total']})", flush=True)

avg_head_acc = sum(s["head_accuracy"] for s in stats_log) / len(stats_log) if stats_log else float("nan")
print(f"Training done, skipped {skipped}/{len(train_prompts)}, avg head accuracy: {avg_head_acc:.4f}", flush=True)
assistant_model.save_pretrained("trained_llama68m_with_acceptance_head")
torch.save(acceptance_head.state_dict(), "acceptance_head_llama68m.pt")
print("Saved adapter + acceptance_head_llama68m.pt", flush=True)

print("=== MERGING LoRA adapter ===", flush=True)
merged_model = assistant_model.merge_and_unload()
merged_model.eval()
acceptance_head.eval()

# ============ [3] v2 GATED INFERENCE, WITH K-LOGGING ============
captured_Ks = []


class AcceptanceHeadCandidateGeneratorV2(AssistedCandidateGeneratorDifferentTokenizers):
    def get_candidates(self, input_ids, **kwargs):
        max_new_tokens = int(self.num_assistant_tokens)
        if max_new_tokens == 0:
            return input_ids, None
        input_ids = input_ids.to(self.assistant_model.device)
        assistant_input_ids, remove_from_pkv = self._prepare_assistant_input_ids(input_ids)
        self.prev_assistant_ids = assistant_input_ids
        min_new_tokens = max(min(max_new_tokens, self.main_model_min_length - assistant_input_ids.shape[-1]), 0)
        self._update_past_and_masks(assistant_input_ids, remove_from_pkv)
        generation_args = self._prepare_generation_args(assistant_input_ids, min_new_tokens, max_new_tokens)
        self.assistant_kwargs.pop("attention_mask", None)
        self.assistant_kwargs.pop("position_ids", None)

        assistant_output = self.assistant_model.generate(
            **generation_args, **self.assistant_kwargs, output_hidden_states=True, return_dict_in_generate=True,
        )
        if assistant_output.hidden_states:
            last_step_last_layer = assistant_output.hidden_states[-1][-1]
            self._last_hidden_state = last_step_last_layer[0, -1, :]
        else:
            self._last_hidden_state = None

        new_target_ids = self._process_assistant_outputs(input_ids, assistant_output.sequences)
        captured_Ks.append(new_target_ids.shape[1] - input_ids.shape[1])

        self.prev_target_ids_len = input_ids.shape[1]
        self.assistant_kwargs["past_key_values"] = assistant_output.past_key_values
        self.prev_assistant_ids = assistant_output.sequences

        if self.prev_target_ids_len >= new_target_ids.shape[1]:
            return input_ids, None
        return new_target_ids, None

    def update_candidate_strategy(self, input_ids, scores, num_matches):
        try:
            hidden = getattr(self, "_last_hidden_state", None)
            if hidden is None:
                return super().update_candidate_strategy(input_ids, scores, num_matches)
            with torch.no_grad():
                reject_prob = torch.sigmoid(acceptance_head(hidden)).item()
            if reject_prob < REJECT_THRESHOLD:
                self.num_assistant_tokens = min(self.num_assistant_tokens + 2, MAX_DRAFT)
            else:
                self.num_assistant_tokens = max(MIN_DRAFT, self.num_assistant_tokens - 1)
        except Exception as e:
            print(f"  [v2 update failed, falling back to heuristic: {e}]", flush=True)
            super().update_candidate_strategy(input_ids, scores, num_matches)


def patched_get_candidate_generator(self, generation_config, input_ids, inputs_tensor, logits_processor,
                                     model_kwargs, assistant_model=None, target_tokenizer=None, assistant_tokenizer=None):
    return AcceptanceHeadCandidateGeneratorV2(
        input_ids=input_ids, assistant_model=assistant_model, generation_config=generation_config,
        model_kwargs=model_kwargs, inputs_tensor=inputs_tensor, logits_processor=logits_processor,
        target_tokenizer=target_tokenizer, assistant_tokenizer=assistant_tokenizer,
    )


target_model._get_candidate_generator = types.MethodType(patched_get_candidate_generator, target_model)


def measure_sd_gated(prompt, max_new=60):
    inputs = target_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(target_model.device)
    call_count["n"] = 0
    captured_Ks.clear()
    t0 = time.time()
    with torch.no_grad():
        out = target_model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                                     assistant_model=merged_model, tokenizer=target_tokenizer, assistant_tokenizer=assistant_tokenizer)
    elapsed = time.time() - t0
    n_new = out.shape[1] - inputs["input_ids"].shape[1]
    calls = max(call_count["n"], 1)
    avg_k = sum(captured_Ks) / len(captured_Ks) if captured_Ks else float("nan")
    del out
    gc.collect(); torch.cuda.empty_cache()
    return elapsed, n_new / calls, avg_k


print("=== [3] ACCEPTANCE-HEAD-V2 GATED, Llama-68M+Gemma-2-9b, 30 prompts ===", flush=True)
gated_times, gated_tpc, gated_ks = [], [], []
for i, p in enumerate(prompts):
    t, tpc, k = measure_sd_gated(p)
    gated_times.append(t); gated_tpc.append(tpc); gated_ks.append(k)
    speedup = baseline_times[i] / t
    print(f"  gated[{i}]: {t:.2f}s tok/call={tpc:.3f} avg_K={k:.2f} speedup={speedup:.3f}x", flush=True)

speedups = [b / t for b, t in zip(baseline_times, gated_times)]
print("\n=== FINAL SUMMARY (acceptance-head v2 gated, Llama-68M/Gemma-2-9b) ===", flush=True)
print(f"avg speedup: {sum(speedups)/len(speedups):.3f}x", flush=True)
print(f"avg tok/call: {sum(gated_tpc)/len(gated_tpc):.3f}", flush=True)
print(f"avg_K: {sum(gated_ks)/len(gated_ks):.3f}", flush=True)
print(f"baseline avg: {baseline_avg:.2f}s", flush=True)
print("\nFor comparison (from earlier tonight, same pairing, same 30 prompts):", flush=True)
print("  untrained (adaptive heuristic):    avg_time=10.00s tok/call=1.854 avg_K=1.83 speedup~1.51-1.61x", flush=True)
print("  trained, NO gating (adaptive):     avg_time=10.61s tok/call=1.878 avg_K=3.38 speedup~1.42-1.56x", flush=True)
print("  trained, NO gating (fixed K=4):    avg_time=10.35s tok/call=1.891 avg_K=2.55 speedup~1.46-1.58x", flush=True)

with open("acceptance_head_v2_llama68m_results.json", "w") as f:
    json.dump({
        "baseline_times": baseline_times, "gated_times": gated_times,
        "gated_tpc": gated_tpc, "gated_ks": gated_ks,
        "avg_head_acc": avg_head_acc,
    }, f)
print("Saved to acceptance_head_v2_llama68m_results.json", flush=True)
