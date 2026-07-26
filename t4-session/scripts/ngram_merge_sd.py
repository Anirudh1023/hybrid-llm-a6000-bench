"""
OmniDraft's actual n-gram cache mechanism, built as a standalone
draft-verify loop rather than extending HF's internal CandidateGenerator
classes (already spent significant time debugging two real bugs deep in
those internals tonight -- a standalone loop gives full, correct control
over translation logic without fighting more edge cases).

Uses Llama-68M / Qwen2.5-7B-Instruct -- confirmed via diag_suppression_rate.py
that this pairing has REAL tokenizer mismatch (30.4% of drafter tokens have
no 1:1 string match in the target vocab), unlike tonight's Qwen0.5B/Qwen7B
pairing (0% suppression, same tokenizer family -- the n-gram cache would
have literally nothing to do there).

Translation strategy per drafted token run, in priority order:
  1. direct_map: precomputed once, same STRING-overlap 1:1 mapping HF's own
     AssistantToTargetTranslator uses (reused directly, not reimplemented).
  2. ngram_cache: online, updated during generation -- maps a TUPLE of
     drafter token ids (a contiguous run of otherwise-unmapped tokens) to
     the target token ids obtained by decoding that run's text and
     re-encoding with the target tokenizer. Checked before falling back to
     re-doing the (slower, and copy-per-call noisy) decode/re-encode.
  3. Fallback: decode the unmapped run to text, re-encode with target
     tokenizer, cache the result under (2) for future reuse. This is the
     "add to n-gram cache if there exists unseen n-gram instance" step from
     the paper -- first occurrence pays the cost, every later occurrence of
     the same run is a cheap, CONSISTENT cache hit (also fixes the boundary-
     drift-per-call issue that plagued the primitive translator, since the
     same run always translates the same way once cached).

Verification: greedy, byte-identical-to-plain-target guarantee (same math
validated throughout this entire session) -- propose candidates, accept
prefix that matches target's own argmax, substitute target's own token at
first mismatch, discard the rest of that round's candidates.

Reports, per question: draft-side token count, how many were served by
direct_map vs ngram_cache HIT vs ngram_cache MISS-then-cached (a real cache
warm-up signal across a growing conversation), plus wall-clock speedup vs
plain generation -- WITH the cache vs a control run where cache lookups are
disabled (forcing every non-direct-mapped run through the fallback every
time, matching the "no persistent cache" baseline) to isolate what the
cache itself is contributing, not just what direct_map + fallback alone do.
"""
import gc
import json
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.generation.candidate_generator import AssistantToTargetTranslator

CACHE_MODE = sys.argv[1] if len(sys.argv) > 1 else "with_cache"  # "with_cache" or "no_cache"
TARGET_ID = "Qwen/Qwen2.5-7B-Instruct"
DRAFT_ID = "JackFram/llama-68m"
MAX_NEW = 60
DRAFT_WINDOW = 5  # tokens drafted per round

with open("conv_dataset_mixed_complexity.json") as f:
    TOPICS = json.load(f)

target_tokenizer = AutoTokenizer.from_pretrained(TARGET_ID)
assistant_tokenizer = AutoTokenizer.from_pretrained(DRAFT_ID)
bnb_config = BitsAndBytesConfig(load_in_8bit=True)
target_model = AutoModelForCausalLM.from_pretrained(TARGET_ID, quantization_config=bnb_config, device_map={"": 0})
assistant_model = AutoModelForCausalLM.from_pretrained(DRAFT_ID, dtype=torch.float16, device_map={"": 1})

# Precomputed static 1:1 direct mapping, reusing HF's own overlap logic directly.
_translator = AssistantToTargetTranslator(
    target_tokenizer=target_tokenizer, assistant_tokenizer=assistant_tokenizer,
    target_vocab_size=target_model.config.get_text_config().vocab_size,
    assistant_model=None, assistant_prune_lm_head=False,
)
direct_map = {}
_map_tensor = _translator._assistant_to_target_input_ids
for a_id in range(_map_tensor.shape[0]):
    t_id = _map_tensor[a_id].item()
    if t_id != _translator.SUPPRESS_TOKEN_ID:
        direct_map[a_id] = t_id

ngram_cache = {}  # tuple(assistant_ids) -> list[target_ids]
stats = {"direct": 0, "cache_hit": 0, "cache_miss_added": 0, "total": 0}


def translate_run(assistant_ids):
    """Translate a full contiguous list of newly-drafted assistant token ids
    into target token ids, using direct_map first, then ngram_cache for
    unmapped runs, falling back to decode/re-encode (and caching the result,
    unless CACHE_MODE == 'no_cache')."""
    out = []
    i = 0
    n = len(assistant_ids)
    while i < n:
        if assistant_ids[i] in direct_map:
            out.append(direct_map[assistant_ids[i]])
            stats["direct"] += 1
            stats["total"] += 1
            i += 1
            continue
        # start of an unmapped run -- extend it while tokens keep being unmapped
        j = i
        while j < n and assistant_ids[j] not in direct_map:
            j += 1
        run = tuple(assistant_ids[i:j])
        if CACHE_MODE == "with_cache" and run in ngram_cache:
            out.extend(ngram_cache[run])
            stats["cache_hit"] += 1
            stats["total"] += 1
        else:
            text = assistant_tokenizer.decode(list(run), skip_special_tokens=True)
            re_encoded = target_tokenizer(text, add_special_tokens=False)["input_ids"]
            if not re_encoded:
                re_encoded = [target_tokenizer.eos_token_id]
            if CACHE_MODE == "with_cache":
                ngram_cache[run] = re_encoded
            stats["cache_miss_added"] += 1
            stats["total"] += 1
            out.extend(re_encoded)
        i = j
    return out


def gen_sd_ngram(history, question, max_new=MAX_NEW):
    prompt = build_chat_prompt(history, question)
    target_ids = target_tokenizer(prompt, return_tensors="pt").to(target_model.device)["input_ids"]
    n_calls = 0
    generated = []
    t0 = time.time()
    while len(generated) < max_new:
        context_text = target_tokenizer.decode(target_ids[0], skip_special_tokens=False)
        assistant_inputs = assistant_tokenizer(context_text, return_tensors="pt", truncation=True, max_length=2048).to(assistant_model.device)
        with torch.no_grad():
            assistant_out = assistant_model.generate(**assistant_inputs, max_new_tokens=DRAFT_WINDOW, do_sample=False,
                                                       repetition_penalty=1.3, no_repeat_ngram_size=3,
                                                       pad_token_id=assistant_tokenizer.eos_token_id)
        new_assistant_ids = assistant_out[0, assistant_inputs["input_ids"].shape[1]:].tolist()
        if not new_assistant_ids:
            break
        candidate_target_ids = translate_run(new_assistant_ids)
        if not candidate_target_ids:
            candidate_target_ids = [target_tokenizer.eos_token_id]

        cand_tensor = torch.tensor([candidate_target_ids], device=target_model.device)
        full_input = torch.cat([target_ids, cand_tensor], dim=1)
        with torch.no_grad():
            out = target_model(input_ids=full_input)
        n_calls += 1
        logits = out.logits[0, target_ids.shape[1] - 1: full_input.shape[1] - 1]
        target_argmax = logits.argmax(dim=-1).tolist()

        accepted = []
        for k in range(len(candidate_target_ids)):
            if candidate_target_ids[k] == target_argmax[k]:
                accepted.append(candidate_target_ids[k])
            else:
                accepted.append(target_argmax[k])
                break
        else:
            pass
        generated.extend(accepted)
        target_ids = torch.cat([target_ids, torch.tensor([accepted], device=target_model.device)], dim=1)
        if target_tokenizer.eos_token_id in accepted:
            break
    elapsed = time.time() - t0
    text = target_tokenizer.decode(generated[:max_new], skip_special_tokens=True).strip()
    calls = max(n_calls, 1)
    return elapsed, len(generated) / calls, text


def build_chat_prompt(history, new_question):
    messages = []
    for q, a in history:
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": new_question})
    return target_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def gen_target_plain_timed(history, question, max_new=MAX_NEW):
    prompt = build_chat_prompt(history, question)
    inputs = target_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(target_model.device)
    t0 = time.time()
    with torch.no_grad():
        out = target_model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    elapsed = time.time() - t0
    text = target_tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    del out
    gc.collect(); torch.cuda.empty_cache()
    return elapsed, text


print(f"warming up... (cache_mode={CACHE_MODE})", flush=True)
_ = gen_target_plain_timed([], "Hello, how are you today?")
gc.collect(); torch.cuda.empty_cache()

results = []
qi_global = 0
for topic in TOPICS:
    print(f"\n{'='*70}\n[topic: {topic['topic']}]\n", flush=True)
    history = []
    for item in topic["questions"]:
        q, complexity = item["q"], item["complexity"]
        stats_before = dict(stats)
        if qi_global % 2 == 0:
            t_plain, plain_text = gen_target_plain_timed(history, q)
            t_sd, tpc, sd_text = gen_sd_ngram(history, q)
        else:
            t_sd, tpc, sd_text = gen_sd_ngram(history, q)
            t_plain, plain_text = gen_target_plain_timed(history, q)
        speedup = t_plain / t_sd
        q_direct = stats["direct"] - stats_before["direct"]
        q_hit = stats["cache_hit"] - stats_before["cache_hit"]
        q_miss = stats["cache_miss_added"] - stats_before["cache_miss_added"]
        print(f"[{complexity}] Q: {q[:70]}", flush=True)
        print(f"   plain={t_plain:.2f}s sd={t_sd:.2f}s speedup={speedup:.3f}x tok/call={tpc:.3f} "
              f"| direct={q_direct} cache_hit={q_hit} cache_miss={q_miss} cache_size={len(ngram_cache)}", flush=True)
        results.append({"topic": topic["topic"], "question": q, "complexity": complexity,
                         "t_plain": t_plain, "t_sd": t_sd, "speedup": speedup, "tpc": tpc,
                         "direct": q_direct, "cache_hit": q_hit, "cache_miss": q_miss, "cache_size": len(ngram_cache)})
        history.append((q, sd_text))
        qi_global += 1

print(f"\n{'='*70}\n=== SUMMARY (cache_mode={CACHE_MODE}, {len(results)} questions) ===", flush=True)
overall_speedups = [r["speedup"] for r in results]
overall_tpc = [r["tpc"] for r in results]
print(f"OVERALL: avg speedup={sum(overall_speedups)/len(overall_speedups):.3f}x  avg tok/call={sum(overall_tpc)/len(overall_tpc):.3f}", flush=True)
print(f"Final cache size: {len(ngram_cache)} entries", flush=True)
print(f"Cumulative: direct={stats['direct']} cache_hit={stats['cache_hit']} cache_miss_added={stats['cache_miss_added']} total={stats['total']}", flush=True)
if stats["total"] > 0:
    print(f"Direct-map rate: {stats['direct']/stats['total']:.3f}  Cache hit rate (of non-direct): "
          f"{stats['cache_hit']/max(stats['cache_hit']+stats['cache_miss_added'],1):.3f}", flush=True)

with open(f"ngram_merge_results_{CACHE_MODE}.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved to ngram_merge_results_{CACHE_MODE}.json", flush=True)
