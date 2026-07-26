"""
do_sample=True variant -- every prior test tonight (and in every earlier
session) used do_sample=False, which HF's dispatch logic
(_get_candidate_generator in generation/utils.py) routes to
AssistedCandidateGeneratorDifferentTokenizers: a primitive decode-then-
re-encode translator with a longest-common-diagonal-subsequence patch, no
real vocab alignment. do_sample=True routes to a DIFFERENT class,
UniversalSpeculativeDecodingGenerator, which uses AssistantToTargetTranslator
-- a real, precomputed direct vocab-overlap mapping (assistant LM head
pruned to only tokens whose STRING form exists identically in the target
vocab). This is conceptually the "direct mapping" (ELL_DM) baseline from
OmniDraft's own paper -- their ablation starts at 1.59x for direct-mapping
alone, before any n-gram cache is added. We've never triggered this path.

Same mixed-complexity dataset and methodology as mixed_complexity_test.py
(order alternated, warm-up call), but do_sample=True with fixed
temperature/top_p for BOTH plain and SD-handoff calls, so the comparison
stays apples-to-apples. Under sampling, SD's correctness guarantee shifts
from "byte-identical to one deterministic run" (greedy) to "distributionally
matches sampling from the target" -- still lossless, just not checkable
per-example the same way, so this test focuses on tok/call and wall-clock
speedup, not exact-match verification.

Set DRAFTER_MODE via sys.argv[1]: "fp16" or "gptq_exllama2".
"""
import gc
import json
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.generation.candidate_generator import AssistantToTargetTranslator

# Bug fix: AssistantToTargetTranslator only sets `assistant_overlap_token_ids`
# when len(self._suppress_input_ids) > 0 (some assistant token needs
# suppressing). When the assistant vocab is a FULL subset of the target vocab
# (our case: Qwen2.5-0.5B-Instruct / Qwen2.5-7B-Instruct share the same BPE
# merges, differ only in padding-driven vocab_size), nothing needs
# suppressing, so that attribute is never created -- but get_target_ids /
# get_target_logits unconditionally reference it when assistant_prune_lm_head
# is True (which HF's own dispatch always sets for do_sample=True). Patch:
# when nothing was suppressed, the "overlap" mapping is just the identity.
_orig_init = AssistantToTargetTranslator.__init__
_orig_get_target_logits = AssistantToTargetTranslator.get_target_logits


def _patched_init(self, *args, **kwargs):
    _orig_init(self, *args, **kwargs)
    self._no_real_pruning = False
    if self.assistant_prune_lm_head and not hasattr(self, "assistant_overlap_token_ids"):
        # Nothing was suppressed, so the LM head was never actually pruned/
        # reindexed (that only happens inside the len(_suppress_input_ids) > 0
        # branch) -- the identity mapping restores get_target_ids' behavior.
        self.assistant_overlap_token_ids = torch.arange(
            self._assistant_to_target_input_ids.shape[0], device=self._assistant_model_device
        )
        self._no_real_pruning = True


def _patched_get_target_logits(self, assistant_logits):
    if self.assistant_prune_lm_head and self._no_real_pruning:
        # LM head genuinely unpruned (still full assistant vocab_size,
        # e.g. 151936, which is larger than the tokenizer-derived mapping
        # array, e.g. 151665, due to reserved/padding embedding slots) --
        # same slicing HF's own "not pruned" branch already does correctly.
        target_shape = (*assistant_logits.shape[:-1], self.target_vocab_size)
        target_logits = torch.full(target_shape, self.FILTER_VALUE, device=self._assistant_model_device)
        assistant_indices_mask = self._assistant_to_target_input_ids != self.SUPPRESS_TOKEN_ID
        target_logits_supported_indices = self._assistant_to_target_input_ids[assistant_indices_mask]
        valid_assistant_logits = assistant_logits[..., : self._assistant_to_target_input_ids.shape[0]]
        target_logits[..., target_logits_supported_indices] = valid_assistant_logits[..., assistant_indices_mask]
        return target_logits
    return _orig_get_target_logits(self, assistant_logits)


AssistantToTargetTranslator.__init__ = _patched_init
AssistantToTargetTranslator.get_target_logits = _patched_get_target_logits

DRAFTER_MODE = sys.argv[1] if len(sys.argv) > 1 else "fp16"
TARGET_ID = "Qwen/Qwen2.5-7B-Instruct"
MAX_NEW = 60
TEMPERATURE = 0.7
TOP_P = 0.9

with open("conv_dataset_mixed_complexity.json") as f:
    TOPICS = json.load(f)

target_tokenizer = AutoTokenizer.from_pretrained(TARGET_ID)
bnb_config = BitsAndBytesConfig(load_in_8bit=True)
target_model = AutoModelForCausalLM.from_pretrained(TARGET_ID, quantization_config=bnb_config, device_map={"": 0})

if DRAFTER_MODE == "fp16":
    DRAFT_ID = "Qwen/Qwen2.5-0.5B-Instruct"
    assistant_tokenizer = AutoTokenizer.from_pretrained(DRAFT_ID)
    assistant_model = AutoModelForCausalLM.from_pretrained(DRAFT_ID, dtype=torch.float16, device_map={"": 1})
elif DRAFTER_MODE == "gptq_exllama2":
    from gptqmodel import GPTQModel, BACKEND
    DRAFT_ID = "Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4"
    assistant_tokenizer = AutoTokenizer.from_pretrained(DRAFT_ID)
    assistant_model = GPTQModel.load(DRAFT_ID, device="cuda:1", backend=BACKEND.EXLLAMA_V2)
    assistant_model = assistant_model.model if hasattr(assistant_model, "model") else assistant_model
else:
    raise ValueError(f"unknown DRAFTER_MODE {DRAFTER_MODE}")

call_count = {"n": 0}
target_model.register_forward_hook(lambda m, i, o: call_count.update(n=call_count["n"] + 1))


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
        out = target_model.generate(**inputs, max_new_tokens=max_new, do_sample=True,
                                     temperature=TEMPERATURE, top_p=TOP_P)
    elapsed = time.time() - t0
    text = target_tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    del out
    gc.collect(); torch.cuda.empty_cache()
    return elapsed, text


def gen_sd_handoff_timed(history, question, max_new=MAX_NEW):
    prompt = build_chat_prompt(history, question)
    inputs = target_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(target_model.device)
    call_count["n"] = 0
    t0 = time.time()
    with torch.no_grad():
        out = target_model.generate(**inputs, max_new_tokens=max_new, do_sample=True,
                                     temperature=TEMPERATURE, top_p=TOP_P,
                                     assistant_model=assistant_model, tokenizer=target_tokenizer,
                                     assistant_tokenizer=assistant_tokenizer)
    elapsed = time.time() - t0
    n_new = out.shape[1] - inputs["input_ids"].shape[1]
    calls = max(call_count["n"], 1)
    text = target_tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    del out
    gc.collect(); torch.cuda.empty_cache()
    return elapsed, n_new / calls, text


print(f"warming up... (drafter mode={DRAFTER_MODE})", flush=True)
_ = gen_target_plain_timed([], "Hello, how are you today?")
gc.collect(); torch.cuda.empty_cache()

results = []
qi_global = 0
for topic in TOPICS:
    print(f"\n{'='*70}\n[topic: {topic['topic']}]\n", flush=True)
    history = []
    for item in topic["questions"]:
        q, complexity = item["q"], item["complexity"]
        if qi_global % 2 == 0:
            t_plain, plain_text = gen_target_plain_timed(history, q)
            t_sd, tpc, sd_text = gen_sd_handoff_timed(history, q)
        else:
            t_sd, tpc, sd_text = gen_sd_handoff_timed(history, q)
            t_plain, plain_text = gen_target_plain_timed(history, q)
        speedup = t_plain / t_sd
        print(f"[{complexity}] Q: {q[:80]}", flush=True)
        print(f"   plain={t_plain:.2f}s sd={t_sd:.2f}s speedup={speedup:.3f}x tok/call={tpc:.3f}", flush=True)
        results.append({"topic": topic["topic"], "question": q, "complexity": complexity,
                         "t_plain": t_plain, "t_sd": t_sd, "speedup": speedup, "tpc": tpc})
        history.append((q, sd_text))
        qi_global += 1

print(f"\n{'='*70}\n=== SUMMARY (drafter_mode={DRAFTER_MODE}, {len(results)} questions) ===", flush=True)
overall_speedups = [r["speedup"] for r in results]
overall_tpc = [r["tpc"] for r in results]
print(f"OVERALL: avg speedup={sum(overall_speedups)/len(overall_speedups):.3f}x  avg tok/call={sum(overall_tpc)/len(overall_tpc):.3f}", flush=True)
for complexity in ["simple", "moderate", "complex"]:
    subset = [r for r in results if r["complexity"] == complexity]
    sp = [r["speedup"] for r in subset]
    tp = [r["tpc"] for r in subset]
    print(f"{complexity.upper():8s} (n={len(subset)}): avg speedup={sum(sp)/len(sp):.3f}x  avg tok/call={sum(tp)/len(tp):.3f}", flush=True)

with open(f"mixed_complexity_results_sampled_{DRAFTER_MODE}.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved to mixed_complexity_results_sampled_{DRAFTER_MODE}.json", flush=True)
