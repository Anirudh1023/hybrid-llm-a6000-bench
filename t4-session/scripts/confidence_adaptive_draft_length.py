"""
Adaptive speculation-length controller, take 2 -- training-free this time.

The first attempt (adaptive_draft_length.py) used the acceptance head's
predicted continuous ratio min(1, p/q), gated at a flat 0.5 threshold, as
the stopping signal. Result: -6.3% vs fixed window=5, because the
controller stopped after ~1.7 tokens on average -- the ratio label and the
actual greedy accept event (argmax match) don't line up cleanly at 0.5,
so the threshold was miscalibrated for what we actually wanted.

Literature check before building a second training-based fix: the
established adaptive-draft-length techniques (AdaEDL arxiv 2410.18351,
GammaTune+, CM-ASD, AdaSpec, DSpark, SGLang's own adaptive mode) converge
on using the DRAFTER'S OWN native confidence/entropy at generation time as
the stopping signal -- explicitly training- and parameter-free, avoiding
exactly the label-mismatch problem just hit. Caveat: AdaEDL's own formal
entropy-lower-bound derivation is stated to apply to SAMPLING-based
decoding specifically (built on the general rejection-sampling acceptance
formula) and does not extend to greedy, which is what we use. Not porting
that exact formula for that reason. Using the simpler, decoding-strategy-
agnostic sibling instead: threshold on the drafter's own raw top-token
probability (softmax max), same style as GammaTune+/CM-ASD. Under greedy
verification a bad stopping decision only costs efficiency, never
correctness (accept/reject is still exactly argmax-match either way), so
no formal probabilistic guarantee is needed here, only an empirically
reasonable threshold -- hence a small sweep instead of one guessed value.

No training phase at all this time: drafter frozen (as before), no head,
no optimizer. Both the fixed=5 baseline and every confidence-threshold
condition are measured fresh in this same run on the SAME held-out topics
(3-4, 16 questions) used for the first attempt's eval, so this stays
directly comparable to that -6.3% result.
"""
import gc
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.cache_utils import DynamicCache
from transformers.generation.candidate_generator import AssistantToTargetTranslator
from transformers.generation.logits_process import (
    LogitsProcessorList,
    NoRepeatNGramLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
)

TARGET_ID = "google/gemma-2-9b-it"
DRAFT_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_NEW = 60
FIXED_WINDOW = 5
MIN_WINDOW = 1
MAX_WINDOW = 8
THRESHOLDS_TO_SWEEP = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
# Bracketing the peak: threshold=0.3 was the best of {0.3, 0.5, 0.7} in the
# first sweep, but as threshold -> 0 the confidence gate essentially never
# fires (softmax max prob is ~always > 0), so the method degenerates into
# plain fixed window=8 -- already confirmed to give NO improvement over
# fixed window=5 (1.3879 vs 1.3880). So performance must rise from ~1.388
# near threshold=0, peak somewhere, and 0.3 was only the highest point
# sampled so far, not confirmed as the peak itself. Re-including 0.3 here
# as a continuity check against the first sweep's result.

with open("conv_dataset_mixed_complexity.json") as f:
    TOPICS = json.load(f)
EVAL_TOPICS = TOPICS[3:]  # same held-out split as adaptive_draft_length.py

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


def draft_confidence_adaptive(assistant_prompt, threshold, min_window=MIN_WINDOW, max_window=MAX_WINDOW):
    """Draft one token at a time; stop when the drafter's OWN top-softmax
    probability for the token it just proposed drops below threshold
    (after at least min_window tokens), or max_window is reached. No head,
    no training, no target involvement -- purely the drafter's own signal
    about itself, computed at the same point in the loop a fixed-window
    generate() call would already be doing this forward pass anyway.

    Uses the SAME token-selection policy as draft_fixed (repetition_penalty
    1.3 + no_repeat_ngram_size 3 via HF's own LogitsProcessorList, not a
    reimplementation) and stops on EOS -- draft_fixed's .generate() call
    applies both automatically; this manual loop originally didn't, which
    was a real confound (found via a sanity check: threshold=0.05 forces
    avg_window~=8, which should behave like the fixed window=8 control,
    but gave 2.24 tok/call vs the control's 1.39 -- a gap this size could
    only be the token-selection policies actually differing, not the
    stopping criterion, since the stopping criterion barely fires at
    threshold=0.05). Now the ONLY variable differing between draft_fixed
    and draft_confidence_adaptive is the stopping decision, as intended."""
    inputs = assistant_tokenizer(assistant_prompt, return_tensors="pt", truncation=True, max_length=2048).to(assistant_model.device)
    logits_processor = LogitsProcessorList([
        RepetitionPenaltyLogitsProcessor(1.3),
        NoRepeatNGramLogitsProcessor(3),
    ])
    cache = DynamicCache(config=assistant_model.config)
    with torch.no_grad():
        out = assistant_model(input_ids=inputs["input_ids"], past_key_values=cache, use_cache=True)
    cache = out.past_key_values
    cur_logit = out.logits[0, -1]
    running_ids = inputs["input_ids"]

    drafted_ids, confidences = [], []
    for _ in range(max_window):
        scored_logit = logits_processor(running_ids, cur_logit.unsqueeze(0))[0]
        probs = torch.softmax(scored_logit, dim=-1)
        next_id = probs.argmax(dim=-1).item()
        conf = probs.max().item()
        drafted_ids.append(next_id)
        confidences.append(conf)
        running_ids = torch.cat([running_ids, torch.tensor([[next_id]], device=running_ids.device)], dim=1)
        if next_id == assistant_tokenizer.eos_token_id:
            break
        if len(drafted_ids) >= min_window and conf < threshold:
            break
        if len(drafted_ids) >= max_window:
            break
        with torch.no_grad():
            step_out = assistant_model(
                input_ids=torch.tensor([[next_id]], device=assistant_model.device),
                past_key_values=cache, use_cache=True,
            )
        cache = step_out.past_key_values
        cur_logit = step_out.logits[0, -1]
    return drafted_ids, confidences


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


def run_question(history, q, mode, threshold=None, window_lengths_log=None):
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
        if mode == "fixed":
            new_assistant_ids = draft_fixed(assistant_prompt, FIXED_WINDOW)
        else:
            new_assistant_ids, _ = draft_confidence_adaptive(assistant_prompt, threshold)
            if window_lengths_log is not None:
                window_lengths_log.append(len(new_assistant_ids))

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


def run_condition(mode, threshold=None):
    results, all_windows = [], []
    for topic in EVAL_TOPICS:
        history = []
        for item in topic["questions"]:
            q = item["q"]
            window_lengths = []
            tok_per_call, answer = run_question(history, q, mode, threshold, window_lengths)
            history.append((q, answer))
            results.append({"complexity": item["complexity"], "tok_per_call": tok_per_call})
            all_windows.extend(window_lengths)
            gc.collect(); torch.cuda.empty_cache()
    avg = sum(r["tok_per_call"] for r in results) / len(results)
    avg_window = sum(all_windows) / len(all_windows) if all_windows else None
    return results, avg, avg_window


print("\n=== Fixed window=5 baseline ===", flush=True)
fixed_results, fixed_avg, _ = run_condition("fixed")
print(f"Fixed avg tok/call over {len(fixed_results)} questions: {fixed_avg:.4f}", flush=True)

sweep_summary = {"fixed_avg": fixed_avg}
for thr in THRESHOLDS_TO_SWEEP:
    print(f"\n=== Confidence-adaptive, threshold={thr} ===", flush=True)
    results, avg, avg_window = run_condition("adaptive", thr)
    change = (avg / fixed_avg - 1) * 100
    print(f"threshold={thr} avg tok/call over {len(results)} questions: {avg:.4f} "
          f"(avg_window={avg_window:.2f}) change vs fixed: {change:+.1f}%", flush=True)
    sweep_summary[f"threshold_{thr}"] = {"avg_tok_per_call": avg, "avg_window": avg_window, "change_pct": change}

print("\n=== SWEEP SUMMARY ===", flush=True)
print(json.dumps(sweep_summary, indent=2), flush=True)

with open("confidence_adaptive_draft_length_results.json", "w") as f:
    json.dump(sweep_summary, f, indent=2)
print("\nSaved to confidence_adaptive_draft_length_results.json", flush=True)
print("Confidence-adaptive draft length sweep completed without error.", flush=True)
