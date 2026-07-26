"""
Step 1 of the online-training build: data collection loop.

Qwen2.5-0.5B-Instruct (drafter) + google/gemma-2-9b (target) -- deliberately
NOT the same-family Qwen/Qwen pairing (0% suppression, n-gram merges never
happen there). This pairing has 57.6% suppression (confirmed earlier this
session via diag_suppression_rate.py), so both halves of OmniDraft's hybrid
loss actually get exercised: reverse-KL on the tokens that DO have a direct
1:1 vocab mapping, NLL on the tokens that only exist via n-gram-cache merges.

For each drafted round this records, per token position:
  - the drafter's FULL logit vector (needed for reverse-KL, not just argmax)
  - the target's FULL logit vector at the corresponding verified position
  - whether the token was served by direct_map or by an n-gram merge
  - the acceptance ratio min(1, p_target(token)/q_drafter(token))
  - whether it was actually accepted or corrected

Uses a real DynamicCache on the target side from the start (KV-cache-correct
from the first version this time, not retrofitted after a slow first draft).

This step ONLY collects data -- no gradient steps yet. The hybrid loss gets
built and validated against what this produces, before any online-loop
complexity gets added on top (task #16, separate script).
"""
import gc
import json
import pickle

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.cache_utils import DynamicCache
from transformers.generation.candidate_generator import AssistantToTargetTranslator

TARGET_ID = "google/gemma-2-9b-it"
DRAFT_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_NEW = 60
DRAFT_WINDOW = 5

with open("conv_dataset_mixed_complexity.json") as f:
    TOPICS = json.load(f)

target_tokenizer = AutoTokenizer.from_pretrained(TARGET_ID)
assistant_tokenizer = AutoTokenizer.from_pretrained(DRAFT_ID)
bnb_config = BitsAndBytesConfig(load_in_8bit=True)
target_model = AutoModelForCausalLM.from_pretrained(TARGET_ID, quantization_config=bnb_config, device_map={"": 0})
assistant_model = AutoModelForCausalLM.from_pretrained(DRAFT_ID, dtype=torch.float16, device_map={"": 1})

target_vocab_size = target_model.config.get_text_config().vocab_size

# Static direct_map, same source as every other script tonight -- reused,
# not reimplemented.
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
print(f"direct_map covers {len(direct_map)}/{_map_tensor.shape[0]} assistant tokens "
      f"({100*len(direct_map)/_map_tensor.shape[0]:.1f}%)", flush=True)

ngram_cache = {}  # tuple(assistant_ids) -> list[target_ids], same as ngram_merge_sd.py


def build_chat_prompt(history, new_question):
    messages = []
    for q, a in history:
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": new_question})
    return target_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def build_assistant_prompt(history, new_question, partial_answer_text):
    """The drafter's OWN chat template, not a decode/re-encode of the
    target's formatted prompt -- that was the actual bug (Gemma's
    <start_of_turn>/<end_of_turn> markers ending up as literal, un-parseable
    text once re-tokenized by Qwen's tokenizer, which has never seen them).
    Appends the growing plain-text partial answer as a raw continuation of
    the assistant turn -- chat templates open the assistant turn but don't
    close it when add_generation_prompt=True, so this is a legitimate
    continuation, not a malformed prompt."""
    messages = []
    for q, a in history:
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": new_question})
    base = assistant_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return base + partial_answer_text


def translate_run_with_tags(assistant_ids):
    """Same translation logic as ngram_merge_sd.py, but also returns, per
    emitted target token: its tag ('direct' or 'ngram') AND which assistant
    POSITION(S) (indices into assistant_ids) contributed to it. For 'direct'
    tokens this is always exactly one position (clean 1:1 correspondence --
    the drafter's logit vector there can be directly compared to the
    target's, enabling reverse-KL). For 'ngram' tokens this is a list of
    positions (a merged run) -- no clean per-position distributional
    correspondence exists, which is exactly why these only get NLL against
    the assistant's own predictions, not KL against the target."""
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
        out_src_positions.extend([list(range(i, j))] * len(merged))  # same source run for all merged outputs
        i = j
    return out_tokens, out_tags, out_src_positions


def collect_round(history, question, partial_answer_text, target_ids, cache, next_pred_logit):
    """One draft-verify round, returns collected per-token records plus
    updated (partial_answer_text, target_ids, cache, next_pred_logit) for
    the next round."""
    assistant_prompt = build_assistant_prompt(history, question, partial_answer_text)
    assistant_inputs = assistant_tokenizer(assistant_prompt, return_tensors="pt", truncation=True, max_length=2048).to(assistant_model.device)
    with torch.no_grad():
        assistant_out = assistant_model.generate(
            **assistant_inputs, max_new_tokens=DRAFT_WINDOW, do_sample=False,
            repetition_penalty=1.3, no_repeat_ngram_size=3,
            pad_token_id=assistant_tokenizer.eos_token_id,
            output_scores=True, return_dict_in_generate=True,
        )
    new_assistant_ids = assistant_out.sequences[0, assistant_inputs["input_ids"].shape[1]:].tolist()
    assistant_logits_per_step = [s[0].float().cpu() for s in assistant_out.scores]  # one per drafted assistant token
    assistant_ids_per_step = new_assistant_ids  # same indexing as assistant_logits_per_step
    if not new_assistant_ids:
        return [], target_ids, cache, next_pred_logit, True

    candidate_target_ids, tags, src_positions = translate_run_with_tags(new_assistant_ids)
    if not candidate_target_ids:
        return [], target_ids, cache, next_pred_logit, True

    cand_tensor = torch.tensor([candidate_target_ids], device=target_model.device)
    prefix_len = cache.get_seq_length()
    with torch.no_grad():
        out = target_model(input_ids=cand_tensor, past_key_values=cache, use_cache=True)
    cache = out.past_key_values
    step_logits = out.logits[0].float().cpu()  # (len(candidate_target_ids), vocab)
    predictions = [next_pred_logit.argmax(dim=-1).item()] + step_logits[:-1].argmax(dim=-1).tolist()

    records = []
    accepted, rejected_at = [], None
    for k in range(len(candidate_target_ids)):
        target_logit_k = next_pred_logit if k == 0 else step_logits[k - 1]
        assistant_logits_k = [assistant_logits_per_step[p] for p in src_positions[k]]
        assistant_ids_k = [assistant_ids_per_step[p] for p in src_positions[k]]
        records.append({
            "tag": tags[k],
            "proposed_target_id": candidate_target_ids[k],
            "target_logits": target_logit_k,        # full vocab logit vector, target's own
            "assistant_logits": assistant_logits_k,  # list (len 1 for 'direct', >=1 for 'ngram')
            "assistant_ids": assistant_ids_k,        # the drafter's own token id(s) at those positions
            "matched": candidate_target_ids[k] == predictions[k],
        })
        if candidate_target_ids[k] == predictions[k]:
            accepted.append(candidate_target_ids[k])
        else:
            accepted.append(predictions[k])
            rejected_at = k
            break

    n_from_cand = rejected_at if rejected_at is not None else len(candidate_target_ids)
    if n_from_cand < len(candidate_target_ids):
        cache.crop(prefix_len + n_from_cand)
    # NOTE: records is intentionally NOT truncated here. The KV cache is
    # correctly cropped at n_from_cand (only accepted tokens' K/V entries are
    # real going forward), but that's a separate concern from which RECORDS
    # get kept for training -- the rejected token at position rejected_at
    # (if any) is exactly the negative example the acceptance head needs
    # (matched=False, real low ratio). Discarding it here (an earlier version
    # of this script did exactly that, via records[:n_from_cand]) meant every
    # collected record was, by construction, an accepted one -- the
    # acceptance head could only ever learn "always predict accept."

    target_ids = torch.cat([target_ids, torch.tensor([accepted], device=target_model.device)], dim=1)
    hit_eos = target_tokenizer.eos_token_id in accepted
    partial_answer_text = partial_answer_text + target_tokenizer.decode(accepted, skip_special_tokens=True)

    if rejected_at is not None:
        corrected_tensor = torch.tensor([[accepted[-1]]], device=target_model.device)
        with torch.no_grad():
            correction_out = target_model(input_ids=corrected_tensor, past_key_values=cache, use_cache=True)
        cache = correction_out.past_key_values
        next_pred_logit = correction_out.logits[0, -1].float().cpu()
    else:
        next_pred_logit = out.logits[0, -1].float().cpu()

    return records, partial_answer_text, target_ids, cache, next_pred_logit, hit_eos


all_records = []
qi = 0
for topic in TOPICS[:2]:  # small slice first -- confirm correctness before the full dataset
    print(f"\n[topic: {topic['topic']}]", flush=True)
    history = []
    for item in topic["questions"][:4]:
        q = item["q"]
        prompt = build_chat_prompt(history, q)
        target_ids = target_tokenizer(prompt, return_tensors="pt").to(target_model.device)["input_ids"]
        prompt_len = target_ids.shape[1]

        cache = DynamicCache(config=target_model.config)
        with torch.no_grad():
            prefill_out = target_model(input_ids=target_ids, past_key_values=cache, use_cache=True)
        cache = prefill_out.past_key_values
        next_pred_logit = prefill_out.logits[0, -1].float().cpu()

        partial_answer_text = ""
        while target_ids.shape[1] - prompt_len < MAX_NEW:
            records, partial_answer_text, target_ids, cache, next_pred_logit, hit_eos = collect_round(
                history, q, partial_answer_text, target_ids, cache, next_pred_logit)
            all_records.extend(records)
            if hit_eos:
                break

        history.append((q, partial_answer_text))
        n_direct = sum(1 for r in all_records if r["tag"] == "direct")
        n_ngram = sum(1 for r in all_records if r["tag"] == "ngram")
        print(f"Q{qi}: collected so far -- direct={n_direct} ngram={n_ngram} total={len(all_records)}", flush=True)
        qi += 1
        gc.collect(); torch.cuda.empty_cache()

print(f"\nFinal: {len(all_records)} records, "
      f"direct={sum(1 for r in all_records if r['tag']=='direct')}, "
      f"ngram={sum(1 for r in all_records if r['tag']=='ngram')}", flush=True)

with open("online_train_collected.pkl", "wb") as f:
    pickle.dump(all_records, f)
print("Saved to online_train_collected.pkl", flush=True)
