"""
Control for the confidence-adaptive draft-length sweep: an adaptive
threshold=0.3 run hit +53.9% tok/call vs fixed window=5 (avg_window=4.48,
max_window=8). Before crediting the confidence-based early-stopping
mechanism, need to rule out the obvious confound -- MAX_WINDOW=8 is
simply bigger than FIXED_WINDOW=5, so some or all of the gain could come
from just raising the cap, with the adaptivity doing little real work.
This runs a plain FIXED window=8 (no adaptivity, no confidence check at
all) on the identical held-out eval set to isolate that.
"""
import gc
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.cache_utils import DynamicCache
from transformers.generation.candidate_generator import AssistantToTargetTranslator

TARGET_ID = "google/gemma-2-9b-it"
DRAFT_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_NEW = 60
WINDOW = 8

with open("conv_dataset_mixed_complexity.json") as f:
    TOPICS = json.load(f)
EVAL_TOPICS = TOPICS[3:]

target_tokenizer = AutoTokenizer.from_pretrained(TARGET_ID)
assistant_tokenizer = AutoTokenizer.from_pretrained(DRAFT_ID)
bnb_config = BitsAndBytesConfig(load_in_8bit=True)
target_model = AutoModelForCausalLM.from_pretrained(TARGET_ID, quantization_config=bnb_config, device_map={"": 0})
assistant_model = AutoModelForCausalLM.from_pretrained(DRAFT_ID, dtype=torch.float32, device_map={"": 1})
assistant_model.eval()
for p in assistant_model.parameters():
    p.requires_grad_(False)

target_vocab_size = target_model.config.get_text_config().vocab_size
_translator = AssistantToTargetTranslator(
    target_tokenizer=target_tokenizer, assistant_tokenizer=assistant_tokenizer,
    target_vocab_size=target_vocab_size, assistant_model=None, assistant_prune_lm_head=False,
)
direct_map = {}
_map_tensor = _translator._assistant_to_target_input_ids
for a_id in range(_map_tensor.shape[0]):
    t_id = _map_tensor[a_id].item()
    if t_id != _translator.SUPPRESS_TOKEN_ID:
        direct_map[a_id] = t_id
print(f"direct_map covers {len(direct_map)}/{_map_tensor.shape[0]} assistant tokens", flush=True)

ngram_cache = {}


def build_assistant_prompt(history, question, partial_answer_text):
    messages = []
    for q, a in history:
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": question})
    base = assistant_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return base + partial_answer_text


def build_target_prompt(history, question):
    messages = []
    for q, a in history:
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": question})
    return target_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def translate_run_with_tags(assistant_ids):
    out_tokens, out_tags, out_src_positions = [], [], []
    i, n = 0, len(assistant_ids)
    while i < n:
        if assistant_ids[i] in direct_map:
            out_tokens.append(direct_map[assistant_ids[i]])
            out_tags.append("direct")
            out_src_positions.append([i])
            i += 1
            continue
        j = i
        while j < n and assistant_ids[j] not in direct_map:
            j += 1
        run = tuple(assistant_ids[i:j])
        if run in ngram_cache:
            merged = ngram_cache[run]
        else:
            text = assistant_tokenizer.decode(list(run), skip_special_tokens=True)
            merged = target_tokenizer(text, add_special_tokens=False)["input_ids"]
            if not merged:
                merged = [target_tokenizer.eos_token_id]
            ngram_cache[run] = merged
        out_tokens.extend(merged)
        out_tags.extend(["ngram"] * len(merged))
        out_src_positions.extend([list(range(i, j))] * len(merged))
        i = j
    return out_tokens, out_tags, out_src_positions


def draft_fixed(assistant_prompt, window):
    inputs = assistant_tokenizer(assistant_prompt, return_tensors="pt", truncation=True, max_length=2048).to(assistant_model.device)
    with torch.no_grad():
        out = assistant_model.generate(
            **inputs, max_new_tokens=window, do_sample=False,
            repetition_penalty=1.3, no_repeat_ngram_size=3,
            pad_token_id=assistant_tokenizer.eos_token_id,
        )
    return out[0, inputs["input_ids"].shape[1]:].tolist()


def verify_candidate(new_assistant_ids, target_ids, cache, next_pred_logit):
    if not new_assistant_ids:
        return target_ids, cache, next_pred_logit, True, []
    candidate_target_ids, tags, src_positions = translate_run_with_tags(new_assistant_ids)
    if not candidate_target_ids:
        return target_ids, cache, next_pred_logit, True, []

    cand_tensor = torch.tensor([candidate_target_ids], device=target_model.device)
    prefix_len = cache.get_seq_length()
    with torch.no_grad():
        out = target_model(input_ids=cand_tensor, past_key_values=cache, use_cache=True)
    cache = out.past_key_values
    step_logits = out.logits[0].float().cpu()
    predictions = [next_pred_logit.argmax(dim=-1).item()] + step_logits[:-1].argmax(dim=-1).tolist()

    accepted, rejected_at = [], None
    for k in range(len(candidate_target_ids)):
        if candidate_target_ids[k] == predictions[k]:
            accepted.append(candidate_target_ids[k])
        else:
            accepted.append(predictions[k])
            rejected_at = k
            break

    n_from_cand = rejected_at if rejected_at is not None else len(candidate_target_ids)
    if n_from_cand < len(candidate_target_ids):
        cache.crop(prefix_len + n_from_cand)

    target_ids = torch.cat([target_ids, torch.tensor([accepted], device=target_model.device)], dim=1)
    hit_eos = target_tokenizer.eos_token_id in accepted

    if rejected_at is not None:
        corrected_tensor = torch.tensor([[accepted[-1]]], device=target_model.device)
        with torch.no_grad():
            correction_out = target_model(input_ids=corrected_tensor, past_key_values=cache, use_cache=True)
        cache = correction_out.past_key_values
        next_pred_logit = correction_out.logits[0, -1].float().cpu()
    else:
        next_pred_logit = out.logits[0, -1].float().cpu()

    return target_ids, cache, next_pred_logit, hit_eos, accepted


def run_question(history, q):
    target_prompt = build_target_prompt(history, q)
    target_ids = target_tokenizer(target_prompt, return_tensors="pt").to(target_model.device)["input_ids"]
    prompt_len = target_ids.shape[1]
    cache = DynamicCache(config=target_model.config)
    with torch.no_grad():
        prefill_out = target_model(input_ids=target_ids, past_key_values=cache, use_cache=True)
    cache = prefill_out.past_key_values
    next_pred_logit = prefill_out.logits[0, -1].float().cpu()

    partial_answer_text = ""
    n_calls = 0
    while target_ids.shape[1] - prompt_len < MAX_NEW:
        assistant_prompt = build_assistant_prompt(history, q, partial_answer_text)
        new_assistant_ids = draft_fixed(assistant_prompt, WINDOW)
        target_ids, cache, next_pred_logit, hit_eos, accepted = verify_candidate(
            new_assistant_ids, target_ids, cache, next_pred_logit)
        n_calls += 1
        if not accepted:
            break
        partial_answer_text = partial_answer_text + target_tokenizer.decode(accepted, skip_special_tokens=True)
        if hit_eos:
            break

    tok_per_call = (target_ids.shape[1] - prompt_len) / max(n_calls, 1)
    return tok_per_call, partial_answer_text


results = []
for topic in EVAL_TOPICS:
    history = []
    for item in topic["questions"]:
        q = item["q"]
        tok_per_call, answer = run_question(history, q)
        history.append((q, answer))
        results.append({"complexity": item["complexity"], "tok_per_call": tok_per_call})
        print(f"[{item['complexity']}]: tok/call={tok_per_call:.3f}", flush=True)
        gc.collect(); torch.cuda.empty_cache()

avg = sum(r["tok_per_call"] for r in results) / len(results)
print(f"\nFixed window=8 avg tok/call over {len(results)} questions: {avg:.4f}", flush=True)
with open("fixed8_control_results.json", "w") as f:
    json.dump({"results": results, "avg": avg}, f, indent=2)
print("Fixed window=8 control completed without error.", flush=True)
