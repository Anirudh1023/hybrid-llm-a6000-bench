"""
Adaptive speculation-length controller, using the acceptance head as the
live confidence signal -- not another drafter-quality training variant.
Rationale: three separate attempts to improve the drafter itself via
online training (reverse KL, forward KL, all-turns) were real on GSM8K but
consistently null-to-negative on our actual conversational deployment
domain. The acceptance head's own BCE loss, though, reliably improved in
every one of those runs even when tok/call didn't -- it's the one signal
that's clearly learning something real. This uses that head for what it
was actually built for (predicting accept probability from the drafter's
own hidden state, before the target is ever consulted) as a REAL-TIME
controller instead of a training byproduct: shrink the draft window when
confidence drops (saves wasted on-device drafter compute), extend it when
confidence stays high (fewer server round-trips, directly the network/
server-cost levers the joint cost model identified as what actually
matters).

Deliberately decoupled from the online-training thread: the DRAFTER is
frozen throughout (no LoRA, no gradient into the base model at all) in
both phases below. Only the acceptance head has trainable parameters.
This isolates "does smarter use of an already-learnable signal help" from
"does retraining the drafter help" -- the latter is the question the last
three runs already answered for this domain.

Two phases, no overlap, matching the session's train/control rigor
pattern:
  Phase A (priming): topics 0-2 (24 questions) of the conversational
  dataset. Standard fixed-window (5) SD collection, head-only training
  (BCE against the continuous acceptance ratio, direct-mapped tokens
  only -- exactly the existing formula, just with no KL/NLL/drafter
  gradient attached).
  Phase B (eval): topics 3-4 (16 questions), run TWICE with the frozen,
  post-priming head -- once under the fixed window=5 baseline, once under
  the adaptive controller -- same questions, same starting state, no
  further training in either pass. tok/call is the outcome metric.
"""
import gc
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.cache_utils import DynamicCache
from transformers.generation.candidate_generator import AssistantToTargetTranslator

TARGET_ID = "google/gemma-2-9b-it"
DRAFT_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_NEW = 60
FIXED_WINDOW = 5
MIN_WINDOW = 1
MAX_WINDOW = 8
ACCEPT_THRESHOLD = 0.5
HEAD_LR = 1e-3  # head-only: a small randomly-initialized 2-layer MLP, not a
                # pretrained model being nudged -- tolerates a higher LR than
                # the LoRA runs did.

with open("conv_dataset_mixed_complexity.json") as f:
    TOPICS = json.load(f)
PRIME_TOPICS = TOPICS[:3]
EVAL_TOPICS = TOPICS[3:]

target_tokenizer = AutoTokenizer.from_pretrained(TARGET_ID)
assistant_tokenizer = AutoTokenizer.from_pretrained(DRAFT_ID)
bnb_config = BitsAndBytesConfig(load_in_8bit=True)
target_model = AutoModelForCausalLM.from_pretrained(TARGET_ID, quantization_config=bnb_config, device_map={"": 0})
assistant_model = AutoModelForCausalLM.from_pretrained(DRAFT_ID, dtype=torch.float32, device_map={"": 1})
assistant_model.eval()
for p in assistant_model.parameters():
    p.requires_grad_(False)  # frozen throughout -- only the head trains
assistant_hidden_size = assistant_model.config.hidden_size

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
assistant_overlap_ids = torch.tensor(sorted(direct_map.keys()))
target_overlap_ids = torch.tensor([direct_map[a] for a in sorted(direct_map.keys())])
print(f"direct_map covers {len(direct_map)}/{_map_tensor.shape[0]} assistant tokens", flush=True)

ngram_cache = {}


class AcceptanceHead(nn.Module):
    def __init__(self, hidden_size, mlp_dim=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_size, mlp_dim), nn.ReLU(), nn.Linear(mlp_dim, 1))

    def forward(self, hidden_state):
        return self.net(hidden_state).squeeze(-1)


head = AcceptanceHead(assistant_hidden_size).to("cuda:1")
optimizer = torch.optim.AdamW(head.parameters(), lr=HEAD_LR)
print(f"Optimizer covers {sum(p.numel() for p in head.parameters())} head params only "
      f"(drafter frozen, {sum(p.numel() for p in assistant_model.parameters())} params, "
      f"requires_grad=False)", flush=True)


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
    return out[0, inputs["input_ids"].shape[1]:].tolist(), None


def draft_adaptive(assistant_prompt, min_window=MIN_WINDOW, max_window=MAX_WINDOW, threshold=ACCEPT_THRESHOLD):
    """Draft one token at a time; after each token, score it with the
    acceptance head using the SAME hidden state that produced its logit
    (matching train_step's alignment: position p's hidden state predicts
    token p). Stop when confidence drops below threshold (after at least
    min_window tokens) or max_window is reached."""
    inputs = assistant_tokenizer(assistant_prompt, return_tensors="pt", truncation=True, max_length=2048).to(assistant_model.device)
    cache = DynamicCache(config=assistant_model.config)
    with torch.no_grad():
        out = assistant_model(input_ids=inputs["input_ids"], past_key_values=cache, use_cache=True, output_hidden_states=True)
    cache = out.past_key_values
    cur_logit = out.logits[0, -1]
    cur_hidden = out.hidden_states[-1][0, -1]

    drafted_ids, confidences = [], []
    for _ in range(max_window):
        next_id = cur_logit.argmax(dim=-1).item()
        with torch.no_grad():
            conf = torch.sigmoid(head(cur_hidden)).item()
        drafted_ids.append(next_id)
        confidences.append(conf)
        if len(drafted_ids) >= min_window and conf < threshold:
            break
        if len(drafted_ids) >= max_window:
            break
        with torch.no_grad():
            step_out = assistant_model(
                input_ids=torch.tensor([[next_id]], device=assistant_model.device),
                past_key_values=cache, use_cache=True, output_hidden_states=True,
            )
        cache = step_out.past_key_values
        cur_logit = step_out.logits[0, -1]
        cur_hidden = step_out.hidden_states[-1][0, -1]
    return drafted_ids, confidences


def verify_candidate(new_assistant_ids, assistant_prompt, target_ids, cache, next_pred_logit):
    """Shared tail: translate drafted ids to target vocab, verify against
    the target in one batched forward pass, accept up to first mismatch.
    Identical logic to the earlier collect_round's verification half."""
    if not new_assistant_ids:
        return [], target_ids, cache, next_pred_logit, True, []

    candidate_target_ids, tags, src_positions = translate_run_with_tags(new_assistant_ids)
    if not candidate_target_ids:
        return [], target_ids, cache, next_pred_logit, True, []

    cand_tensor = torch.tensor([candidate_target_ids], device=target_model.device)
    prefix_len = cache.get_seq_length()
    with torch.no_grad():
        out = target_model(input_ids=cand_tensor, past_key_values=cache, use_cache=True)
    cache = out.past_key_values
    step_logits = out.logits[0].float().cpu()
    predictions = [next_pred_logit.argmax(dim=-1).item()] + step_logits[:-1].argmax(dim=-1).tolist()

    records = []
    accepted, rejected_at = [], None
    for k in range(len(candidate_target_ids)):
        target_logit_k = next_pred_logit if k == 0 else step_logits[k - 1]
        records.append({
            "tag": tags[k],
            "proposed_target_id": candidate_target_ids[k],
            "target_logits": target_logit_k,
            "assistant_prompt": assistant_prompt,
            "assistant_new_ids": new_assistant_ids,
            "src_positions": src_positions[k],
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

    return records, target_ids, cache, next_pred_logit, hit_eos, accepted


def collect_round(history, question, partial_answer_text, target_ids, cache, next_pred_logit, mode, window_lengths_log=None):
    assistant_prompt = build_assistant_prompt(history, question, partial_answer_text)
    if mode == "fixed":
        new_assistant_ids, _ = draft_fixed(assistant_prompt, FIXED_WINDOW)
    else:
        new_assistant_ids, confidences = draft_adaptive(assistant_prompt)
        if window_lengths_log is not None:
            window_lengths_log.append(len(new_assistant_ids))

    records, target_ids, cache, next_pred_logit, hit_eos, accepted = verify_candidate(
        new_assistant_ids, assistant_prompt, target_ids, cache, next_pred_logit)
    if not records:
        return [], partial_answer_text, target_ids, cache, next_pred_logit, True

    partial_answer_text = partial_answer_text + target_tokenizer.decode(accepted, skip_special_tokens=True)
    return records, partial_answer_text, target_ids, cache, next_pred_logit, hit_eos


def train_step_head(records):
    """Head-only BCE against the continuous acceptance ratio min(1, p/q),
    direct-mapped tokens only -- identical formula to the earlier scripts,
    just with no KL/NLL terms and no gradient into the drafter (it's
    frozen; forward passes below run under no_grad for everything except
    the head itself)."""
    optimizer.zero_grad()

    rounds = {}
    for r in records:
        key = (r["assistant_prompt"], tuple(r["assistant_new_ids"]))
        rounds.setdefault(key, []).append(r)

    n_rounds = len(rounds)
    bce_sum, bce_n = 0.0, 0
    for (prompt, new_ids), round_records in rounds.items():
        inputs = assistant_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(assistant_model.device)
        new_ids_t = torch.tensor([list(new_ids)], device=assistant_model.device)
        full_input = torch.cat([inputs["input_ids"], new_ids_t], dim=1)
        with torch.no_grad():
            out = assistant_model(input_ids=full_input, output_hidden_states=True)
        prompt_len = inputs["input_ids"].shape[1]
        drafted_logits = out.logits[0, prompt_len - 1: prompt_len - 1 + len(new_ids)]
        drafted_hidden = out.hidden_states[-1][0, prompt_len - 1: prompt_len - 1 + len(new_ids)]

        round_losses = []
        for r in round_records:
            if r["tag"] != "direct":
                continue
            p = r["src_positions"][0]
            ratio_q = F.softmax(drafted_logits[p], dim=-1)[new_ids[p]]
            ratio_p = F.softmax(r["target_logits"].to(drafted_logits.device), dim=-1)[r["proposed_target_id"]]
            ratio_label = torch.clamp(ratio_p / (ratio_q + 1e-12), max=1.0)
            bce = F.binary_cross_entropy_with_logits(head(drafted_hidden[p]), ratio_label)
            round_losses.append(bce)
            bce_sum += bce.item(); bce_n += 1

        if round_losses:
            (sum(round_losses) / len(round_losses) / n_rounds).backward()
        del out, drafted_logits, drafted_hidden
        torch.cuda.empty_cache()

    if bce_n == 0:
        return None
    optimizer.step()
    return {"bce": bce_sum / bce_n}


def run_question(history, q, mode, train=False, window_lengths_log=None):
    target_prompt = build_target_prompt(history, q)
    target_ids = target_tokenizer(target_prompt, return_tensors="pt").to(target_model.device)["input_ids"]
    prompt_len = target_ids.shape[1]

    cache = DynamicCache(config=target_model.config)
    with torch.no_grad():
        prefill_out = target_model(input_ids=target_ids, past_key_values=cache, use_cache=True)
    cache = prefill_out.past_key_values
    next_pred_logit = prefill_out.logits[0, -1].float().cpu()

    partial_answer_text = ""
    question_records = []
    n_calls = 0
    while target_ids.shape[1] - prompt_len < MAX_NEW:
        records, partial_answer_text, target_ids, cache, next_pred_logit, hit_eos = collect_round(
            history, q, partial_answer_text, target_ids, cache, next_pred_logit, mode, window_lengths_log)
        question_records.extend(records)
        n_calls += 1
        if hit_eos:
            break

    tok_per_call = (target_ids.shape[1] - prompt_len) / max(n_calls, 1)
    stats = train_step_head(question_records) if (train and question_records) else None
    return tok_per_call, partial_answer_text, stats


print("\n=== Phase A: priming the acceptance head (fixed window=5, drafter frozen) ===", flush=True)
qi = 0
for topic in PRIME_TOPICS:
    print(f"\n[prime topic: {topic['topic']}]", flush=True)
    history = []
    for item in topic["questions"]:
        q = item["q"]
        tok_per_call, answer, stats = run_question(history, q, mode="fixed", train=True)
        history.append((q, answer))
        line = f"P{qi} [{item['complexity']}]: tok/call={tok_per_call:.3f}"
        if stats:
            line += f" bce={stats['bce']}"
        print(line, flush=True)
        qi += 1
        gc.collect(); torch.cuda.empty_cache()

print("\n=== Phase B: eval, fixed window=5 baseline (head frozen, no training) ===", flush=True)
fixed_results = []
for topic in EVAL_TOPICS:
    print(f"\n[eval topic: {topic['topic']}]", flush=True)
    history = []
    for item in topic["questions"]:
        q = item["q"]
        tok_per_call, answer, _ = run_question(history, q, mode="fixed", train=False)
        history.append((q, answer))
        print(f"F [{item['complexity']}]: tok/call={tok_per_call:.3f}", flush=True)
        fixed_results.append({"complexity": item["complexity"], "tok_per_call": tok_per_call})
        gc.collect(); torch.cuda.empty_cache()

print("\n=== Phase B: eval, adaptive window (head-controlled, no training) ===", flush=True)
adaptive_results = []
for topic in EVAL_TOPICS:
    print(f"\n[eval topic: {topic['topic']}]", flush=True)
    history = []
    for item in topic["questions"]:
        q = item["q"]
        window_lengths = []
        tok_per_call, answer, _ = run_question(history, q, mode="adaptive", train=False, window_lengths_log=window_lengths)
        history.append((q, answer))
        avg_window = sum(window_lengths) / len(window_lengths) if window_lengths else 0.0
        print(f"A [{item['complexity']}]: tok/call={tok_per_call:.3f} avg_window={avg_window:.2f}", flush=True)
        adaptive_results.append({"complexity": item["complexity"], "tok_per_call": tok_per_call, "avg_window": avg_window})
        gc.collect(); torch.cuda.empty_cache()

fixed_avg = sum(r["tok_per_call"] for r in fixed_results) / len(fixed_results)
adaptive_avg = sum(r["tok_per_call"] for r in adaptive_results) / len(adaptive_results)
print(f"\n=== RESULT ===", flush=True)
print(f"Fixed window=5 tok/call avg over {len(fixed_results)} eval questions: {fixed_avg:.4f}", flush=True)
print(f"Adaptive window tok/call avg over {len(adaptive_results)} eval questions: {adaptive_avg:.4f}", flush=True)
print(f"Change: {(adaptive_avg / fixed_avg - 1) * 100:+.1f}%", flush=True)

with open("adaptive_draft_length_results.json", "w") as f:
    json.dump({"fixed": fixed_results, "adaptive": adaptive_results,
                "fixed_avg": fixed_avg, "adaptive_avg": adaptive_avg}, f, indent=2)
print("\nSaved to adaptive_draft_length_results.json", flush=True)
print("Adaptive draft length experiment completed without error.", flush=True)
