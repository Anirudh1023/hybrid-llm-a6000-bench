"""
Greedy verification (do_sample=False, byte-identical-to-plain correctness
guarantee preserved) PLUS the better direct-mapping translator
(AssistantToTargetTranslator / UniversalSpeculativeDecodingGenerator) that
HF's own dispatch only wires up for do_sample=True. Confirmed via direct
source read that candidate-generator SELECTION and the do_sample used for
VERIFICATION math are independently read from the same generation_config
inside _assisted_decoding -- so forcing generator selection without
touching do_sample itself should be safe. Tests whether this combination
recovers the tok/call gain seen under do_sample=True (2.366 -> 2.883)
without paying sampling-based rejection verification's per-round overhead
(which made that run's net speedup WORSE despite the tok/call gain:
1.185x -> 1.132x).

Monkey-patches GenerationMixin._get_candidate_generator to always take the
different-tokenizers "do_sample=True" branch's construction logic (build
UniversalSpeculativeDecodingGenerator + AssistantToTargetTranslator) for our
model pair, regardless of the real generation_config.do_sample value -- a
faithful copy of the original branch (transformers 4.57.6,
generation/utils.py lines ~1040-1061), not a simplified reimplementation.
Includes both AssistantToTargetTranslator bug fixes from the do_sample=True
test (missing assistant_overlap_token_ids when suppress list is empty;
get_target_logits shape mismatch between tokenizer-vocab-size and model
embedding-table-size).

Same mixed-complexity dataset and methodology (order alternated, warm-up
call), do_sample=False throughout for both plain and SD-handoff.

Set DRAFTER_MODE via sys.argv[1]: "fp16" or "gptq_exllama2".
"""
import gc
import json
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.generation.candidate_generator import (
    AssistantToTargetTranslator,
    AssistantVocabTranslatorCache,
    UniversalSpeculativeDecodingGenerator,
)
from transformers.generation.utils import GenerationMixin

# --- Bug fixes (same as mixed_complexity_test_sampled.py) ---
_orig_translator_init = AssistantToTargetTranslator.__init__
_orig_get_target_logits = AssistantToTargetTranslator.get_target_logits


def _patched_translator_init(self, *args, **kwargs):
    _orig_translator_init(self, *args, **kwargs)
    self._no_real_pruning = False
    if self.assistant_prune_lm_head and not hasattr(self, "assistant_overlap_token_ids"):
        self.assistant_overlap_token_ids = torch.arange(
            self._assistant_to_target_input_ids.shape[0], device=self._assistant_model_device
        )
        self._no_real_pruning = True


def _patched_get_target_logits(self, assistant_logits):
    if self.assistant_prune_lm_head and self._no_real_pruning:
        target_shape = (*assistant_logits.shape[:-1], self.target_vocab_size)
        target_logits = torch.full(target_shape, self.FILTER_VALUE, device=self._assistant_model_device)
        assistant_indices_mask = self._assistant_to_target_input_ids != self.SUPPRESS_TOKEN_ID
        target_logits_supported_indices = self._assistant_to_target_input_ids[assistant_indices_mask]
        valid_assistant_logits = assistant_logits[..., : self._assistant_to_target_input_ids.shape[0]]
        target_logits[..., target_logits_supported_indices] = valid_assistant_logits[..., assistant_indices_mask]
        return target_logits
    return _orig_get_target_logits(self, assistant_logits)


AssistantToTargetTranslator.__init__ = _patched_translator_init
AssistantToTargetTranslator.get_target_logits = _patched_get_target_logits

# --- Force direct-mapping generator selection regardless of do_sample ---
_orig_get_candidate_generator = GenerationMixin._get_candidate_generator


def _patched_get_candidate_generator(
    self, generation_config, input_ids, inputs_tensor, logits_processor, model_kwargs,
    assistant_model=None, target_tokenizer=None, assistant_tokenizer=None,
):
    different_tokenizers = all(v is not None for v in (assistant_model, target_tokenizer, assistant_tokenizer))
    no_special_case = (
        generation_config.assistant_early_exit is None
        and generation_config.prompt_lookup_num_tokens is None
        and not getattr(generation_config, "use_mtp", False)
        and not (assistant_model is not None and assistant_model.__class__.__name__.startswith(
            ("Gemma4Assistant", "Gemma4UnifiedAssistant")))
    )
    if different_tokenizers and no_special_case:
        atm_translator = AssistantVocabTranslatorCache.get_translator(
            target_tokenizer, assistant_tokenizer, self.config.get_text_config().vocab_size,
            assistant_model=assistant_model, assistant_prune_lm_head=True,
        )
        assistant_model.generation_config.repetition_penalty = None
        return UniversalSpeculativeDecodingGenerator(
            input_ids=input_ids, assistant_model=assistant_model, generation_config=generation_config,
            model_kwargs=model_kwargs, inputs_tensor=inputs_tensor, logits_processor=logits_processor,
            target_tokenizer=target_tokenizer, assistant_tokenizer=assistant_tokenizer,
            atm_translator=atm_translator,
        )
    return _orig_get_candidate_generator(
        self, generation_config, input_ids, inputs_tensor, logits_processor, model_kwargs,
        assistant_model, target_tokenizer, assistant_tokenizer,
    )


GenerationMixin._get_candidate_generator = _patched_get_candidate_generator

DRAFTER_MODE = sys.argv[1] if len(sys.argv) > 1 else "fp16"
TARGET_ID = "Qwen/Qwen2.5-7B-Instruct"
MAX_NEW = 60

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
        out = target_model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
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
        out = target_model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
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

with open(f"mixed_complexity_results_greedy_directmap_{DRAFTER_MODE}.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved to mixed_complexity_results_greedy_directmap_{DRAFTER_MODE}.json", flush=True)
