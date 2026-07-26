import gc
import json
import statistics
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

DRAFT_ID = "Qwen/Qwen2.5-0.5B-Instruct"
TARGET_ID = "Qwen/Qwen2.5-7B-Instruct"
MAX_NEW = 40
SPLIT_FRAC = 0.5

with open("heldout_pairs_groundtruth.json") as f:
    pairs = json.load(f)

target_tokenizer = AutoTokenizer.from_pretrained(TARGET_ID)
assistant_tokenizer = AutoTokenizer.from_pretrained(DRAFT_ID)

bnb_config = BitsAndBytesConfig(load_in_8bit=True)
target_model = AutoModelForCausalLM.from_pretrained(TARGET_ID, quantization_config=bnb_config, device_map={"": 0})
assistant_model = AutoModelForCausalLM.from_pretrained(DRAFT_ID, dtype=torch.float16, device_map={"": 1})

call_count = {"n": 0}
target_model.register_forward_hook(lambda m, i, o: call_count.update(n=call_count["n"] + 1))


def split_history(prompt_text, split_frac):
    lines = prompt_text.strip().split("\n")
    cut = max(1, int(len(lines) * split_frac))
    return "\n".join(lines[:cut]), "\n".join(lines[cut:])


def gen_small(text, max_new=MAX_NEW):
    inputs = assistant_tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(assistant_model.device)
    with torch.no_grad():
        out = assistant_model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                                        repetition_penalty=1.3, no_repeat_ngram_size=3,
                                        pad_token_id=assistant_tokenizer.eos_token_id)
    return assistant_tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def gen_target_plain_timed(text, max_new=MAX_NEW):
    inputs = target_tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(target_model.device)
    t0 = time.time()
    with torch.no_grad():
        out = target_model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    elapsed = time.time() - t0
    del out
    gc.collect(); torch.cuda.empty_cache()
    return elapsed


def gen_sd_handoff_timed(text, max_new=MAX_NEW):
    inputs = target_tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(target_model.device)
    call_count["n"] = 0
    t0 = time.time()
    with torch.no_grad():
        out = target_model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                                     assistant_model=assistant_model, tokenizer=target_tokenizer,
                                     assistant_tokenizer=assistant_tokenizer)
    elapsed = time.time() - t0
    n_new = out.shape[1] - inputs["input_ids"].shape[1]
    calls = max(call_count["n"], 1)
    del out
    gc.collect(); torch.cuda.empty_cache()
    return elapsed, n_new / calls


print("warming up...", flush=True)
_ = gen_target_plain_timed("Hello there, how are you today?")
gc.collect(); torch.cuda.empty_cache()

results = []
for i, pair in enumerate(pairs):
    history_A, history_B = split_history(pair["prompt"], SPLIT_FRAC)
    small_turnA = gen_small(history_A).strip().split("\n")[0]
    handoff_prompt = f"{history_A}\n{small_turnA}\n{history_B}\n"

    if i % 2 == 0:
        t_plain = gen_target_plain_timed(handoff_prompt)
        t_sd, tpc = gen_sd_handoff_timed(handoff_prompt)
    else:
        t_sd, tpc = gen_sd_handoff_timed(handoff_prompt)
        t_plain = gen_target_plain_timed(handoff_prompt)

    speedup = t_plain / t_sd
    print(f"[{i}] plain={t_plain:.2f}s sd={t_sd:.2f}s speedup={speedup:.3f}x tok/call={tpc:.3f}", flush=True)
    results.append({"t_plain": t_plain, "t_sd": t_sd, "speedup": speedup, "tpc": tpc})

speedups = [r["speedup"] for r in results]
tpcs = [r["tpc"] for r in results]
avg_speedup = sum(speedups) / len(speedups)
std_speedup = statistics.stdev(speedups)
print(f"\n=== SUMMARY ({len(pairs)} examples, drafter={DRAFT_ID}) ===", flush=True)
print(f"avg speedup: {avg_speedup:.3f}x  std: {std_speedup:.3f}  min: {min(speedups):.3f}x  max: {max(speedups):.3f}x", flush=True)
print(f"avg tok/call: {sum(tpcs)/len(tpcs):.3f}", flush=True)
print(f"median speedup: {statistics.median(speedups):.3f}x", flush=True)

with open("strengthen_core_speedup_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved to strengthen_core_speedup_results.json", flush=True)
