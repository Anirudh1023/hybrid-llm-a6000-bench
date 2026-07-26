"""
Prompt Lookup Decoding (PLD) test -- the established, already-implemented
n-gram technique (HF Transformers, vLLM), fundamentally simpler than
anything built tonight: no draft model at all, just string-matching the
most recent n-gram against the prompt/generated-so-far text, using whatever
followed that match before as the "free" candidate. Verified the same way
as regular SD (wrong guesses just get rejected), so it's still lossless
under greedy decoding -- same guarantee as everything else this session.

Single model only (target, 8-bit bnb) -- no assistant model, no
cross-vocab translation, no KV-cache reconciliation problem at all, since
there's only ever one model's cache involved.

Same mixed-complexity dataset and methodology as mixed_complexity_test.py
(order alternated, warm-up call, growing multi-turn conversation per
topic) -- this dataset should have real repeated structure to exploit
(recurring topic vocabulary, similar question phrasing, the model's own
repeated scaffolding language across turns).
"""
import gc
import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

TARGET_ID = "Qwen/Qwen2.5-7B-Instruct"
MAX_NEW = 60
PROMPT_LOOKUP_NUM_TOKENS = 10

with open("conv_dataset_mixed_complexity.json") as f:
    TOPICS = json.load(f)

target_tokenizer = AutoTokenizer.from_pretrained(TARGET_ID)
bnb_config = BitsAndBytesConfig(load_in_8bit=True)
target_model = AutoModelForCausalLM.from_pretrained(TARGET_ID, quantization_config=bnb_config, device_map={"": 0})

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


def gen_pld_timed(history, question, max_new=MAX_NEW):
    prompt = build_chat_prompt(history, question)
    inputs = target_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(target_model.device)
    call_count["n"] = 0
    t0 = time.time()
    with torch.no_grad():
        out = target_model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                                     prompt_lookup_num_tokens=PROMPT_LOOKUP_NUM_TOKENS)
    elapsed = time.time() - t0
    n_new = out.shape[1] - inputs["input_ids"].shape[1]
    calls = max(call_count["n"], 1)
    text = target_tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    del out
    gc.collect(); torch.cuda.empty_cache()
    return elapsed, n_new / calls, text


print("warming up...", flush=True)
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
            t_pld, tpc, pld_text = gen_pld_timed(history, q)
        else:
            t_pld, tpc, pld_text = gen_pld_timed(history, q)
            t_plain, plain_text = gen_target_plain_timed(history, q)
        speedup = t_plain / t_pld
        print(f"[{complexity}] Q: {q[:80]}", flush=True)
        print(f"   plain={t_plain:.2f}s pld={t_pld:.2f}s speedup={speedup:.3f}x tok/call={tpc:.3f}", flush=True)
        results.append({"topic": topic["topic"], "question": q, "complexity": complexity,
                         "t_plain": t_plain, "t_pld": t_pld, "speedup": speedup, "tpc": tpc})
        history.append((q, pld_text))
        qi_global += 1

print(f"\n{'='*70}\n=== SUMMARY ({len(results)} questions, prompt_lookup_num_tokens={PROMPT_LOOKUP_NUM_TOKENS}) ===", flush=True)
overall_speedups = [r["speedup"] for r in results]
overall_tpc = [r["tpc"] for r in results]
print(f"OVERALL: avg speedup={sum(overall_speedups)/len(overall_speedups):.3f}x  avg tok/call={sum(overall_tpc)/len(overall_tpc):.3f}", flush=True)
for complexity in ["simple", "moderate", "complex"]:
    subset = [r for r in results if r["complexity"] == complexity]
    sp = [r["speedup"] for r in subset]
    tp = [r["tpc"] for r in subset]
    print(f"{complexity.upper():8s} (n={len(subset)}): avg speedup={sum(sp)/len(sp):.3f}x  avg tok/call={sum(tp)/len(tp):.3f}", flush=True)

with open("pld_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved to pld_results.json", flush=True)
