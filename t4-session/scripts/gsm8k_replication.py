"""
Replication attempt: matching OmniDraft's own model pairing (Llama-68M
drafter) and domain (GSM8K) as closely as practical, rather than our own
conversational Q&A domain -- to isolate "does our implementation actually
work, under conditions closer to what the paper validated" from "does it
help OUR specific use case." If training shows a real, measurable
improvement here (matching the qualitative pattern of their Table 5:
direct-mapping-only < +training), that confirms the mechanism and our
implementation are sound, and the earlier conversational-domain result is
a genuine domain-mismatch finding, not an implementation gap. If it
*doesn't* reproduce here either, that's real evidence pointing at an
implementation or scope gap still to find, not at the paper being wrong.

Target: Qwen2-7B-Instruct, not Llama3-8B -- both are in the paper's tested
pairing set, but Meta's Llama models are typically gated on HF (license
request, not guaranteed instant), a real access risk. Qwen models have been
reliably accessible all session.

Runs a TRAIN condition and a NO-TRAIN control on the SAME sequence of
GSM8K questions (not shuffled differently between the two), so any
tok/call difference between them isolates the training effect specifically
-- addresses the topic-order confound flagged in the conversational-domain
run (GSM8K questions are also more homogeneous in structure than our mixed
topics, which should make ordering effects smaller to begin with).

Drafter trained via LoRA (not full fine-tuning), same reasoning as the
earlier run -- the whole premise of this system is on-device deployment.
"""
import gc
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.cache_utils import DynamicCache
from transformers.generation.candidate_generator import AssistantToTargetTranslator

TARGET_ID = "Qwen/Qwen2-7B-Instruct"
DRAFT_ID = "JackFram/llama-68m"
MAX_NEW = 80  # GSM8K CoT answers tend to run a bit longer than our conversational answers
DRAFT_WINDOW = 5
LR = 1e-4
N_QUESTIONS = 40
import sys
TRAIN_MODE = "--control" not in sys.argv  # flip via CLI flag for the no-train control run, same question sequence

gsm8k = load_dataset("gsm8k", "main", split="test")
questions = [gsm8k[i]["question"] for i in range(N_QUESTIONS)]  # fixed slice, same order both runs

# Few-shot exemplars for the drafter (Llama-68M is a BASE model, no chat
# template -- same plain-text pattern used for it in the earlier n-gram
# cache work this session, standard practice for evaluating base models on
# GSM8K since they don't follow zero-shot instructions well but complete
# patterns fine given examples).
FEWSHOT_PREFIX = (
    "Q: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. "
    "How many clips did Natalia sell altogether in April and May?\n"
    "A: Natalia sold 48/2 = 24 clips in May. Natalia sold 48+24 = 72 clips altogether in April and May. #### 72\n\n"
    "Q: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?\n"
    "A: Weng earns 12/60 = $0.2 per minute. Working 50 minutes, she earned 0.2 x 50 = $10. #### 10\n\n"
)

target_tokenizer = AutoTokenizer.from_pretrained(TARGET_ID)
assistant_tokenizer = AutoTokenizer.from_pretrained(DRAFT_ID)
bnb_config = BitsAndBytesConfig(load_in_8bit=True)
target_model = AutoModelForCausalLM.from_pretrained(TARGET_ID, quantization_config=bnb_config, device_map={"": 0})
assistant_model = AutoModelForCausalLM.from_pretrained(DRAFT_ID, dtype=torch.float32, device_map={"": 1})
assistant_hidden_size = assistant_model.config.hidden_size  # capture before PEFT wrapping

# LoRA, not full fine-tuning: base drafter weights frozen, only small
# low-rank adapters trainable. This is the actual point, not just a memory
# optimization -- the whole premise of this system is the drafter runs
# ON-DEVICE, and full fine-tuning's gradient + AdamW momentum/variance
# buffers for every parameter (~2x model size on top of the model itself)
# is not viable on real phone RAM. LoRA is the standard tool for exactly
# this constraint. Target modules are Qwen2's standard attention + MLP
# projection names.
lora_config = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)
assistant_model = get_peft_model(assistant_model, lora_config)
assistant_model.print_trainable_parameters()

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
trainable_drafter_params = [p for p in assistant_model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(trainable_drafter_params + list(head.parameters()), lr=LR)
print(f"Optimizer covers {sum(p.numel() for p in trainable_drafter_params)} drafter params "
      f"(LoRA adapters only) + {sum(p.numel() for p in head.parameters())} head params", flush=True)


def build_assistant_prompt(question, partial_answer_text):
    # Llama-68M is a BASE model -- plain few-shot text completion, no chat
    # template, same pattern used for it in the earlier n-gram cache work.
    return FEWSHOT_PREFIX + f"Q: {question}\nA: " + partial_answer_text


def build_target_prompt(question):
    messages = [{"role": "user", "content": f"{question}\nSolve step by step, then give the final numeric answer after ####."}]
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


def collect_round(question, partial_answer_text, target_ids, cache, next_pred_logit):
    assistant_prompt = build_assistant_prompt(question, partial_answer_text)
    assistant_inputs = assistant_tokenizer(assistant_prompt, return_tensors="pt", truncation=True, max_length=2048).to(assistant_model.device)
    with torch.no_grad():
        assistant_out = assistant_model.generate(
            **assistant_inputs, max_new_tokens=DRAFT_WINDOW, do_sample=False,
            repetition_penalty=1.3, no_repeat_ngram_size=3,
            pad_token_id=assistant_tokenizer.eos_token_id,
        )
    new_assistant_ids = assistant_out[0, assistant_inputs["input_ids"].shape[1]:].tolist()
    if not new_assistant_ids:
        return [], partial_answer_text, target_ids, cache, next_pred_logit, True

    candidate_target_ids, tags, src_positions = translate_run_with_tags(new_assistant_ids)
    if not candidate_target_ids:
        return [], partial_answer_text, target_ids, cache, next_pred_logit, True

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
            "assistant_prompt": assistant_prompt,       # for replay-with-grad later
            "assistant_new_ids": new_assistant_ids,      # the full drafted run this round
            "src_positions": src_positions[k],           # which position(s) in that run produced this token
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


def reverse_kl_direct(assistant_logits, target_logits):
    q_restricted = assistant_logits[assistant_overlap_ids.to(assistant_logits.device)]
    p_restricted = target_logits.to(assistant_logits.device)[target_overlap_ids.to(assistant_logits.device)]
    log_q = F.log_softmax(q_restricted, dim=-1)
    log_p = F.log_softmax(p_restricted, dim=-1)
    q = log_q.exp()
    kl_terms = q * (log_q - log_p)
    return torch.where(q > 0, kl_terms, torch.zeros_like(kl_terms)).sum()


def train_step(records):
    """Replay each round's (prompt, drafted tokens) WITH gradients enabled,
    one teacher-forced forward pass per unique round (not per token),
    compute hybrid loss + acceptance-head loss.

    Gradient ACCUMULATION, not one combined backward() at the end: the first
    version kept every round's forward pass (with output_hidden_states=True,
    i.e. every layer's activations, needed for backprop) alive simultaneously
    until a single final backward() call -- OOM'd GPU 1 within the second
    question, since peak memory scaled with (number of rounds in a question)
    x (per-round activation memory), not with any single round's cost. Fix:
    call backward() immediately after each round's loss, release that
    round's graph, move to the next -- gradients accumulate correctly in
    .grad across multiple backward() calls (that's what accumulation means),
    only ONE round's computational graph is ever alive at a time, and
    optimizer.step() still only happens once, after the loop."""
    assistant_model.train()
    optimizer.zero_grad()

    rounds = {}
    for r in records:
        key = (r["assistant_prompt"], tuple(r["assistant_new_ids"]))
        rounds.setdefault(key, []).append(r)

    n_rounds = len(rounds)
    kl_sum, nll_sum, bce_sum, kl_n, nll_n, bce_n = 0.0, 0.0, 0.0, 0, 0, 0
    for (prompt, new_ids), round_records in rounds.items():
        inputs = assistant_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(assistant_model.device)
        new_ids_t = torch.tensor([list(new_ids)], device=assistant_model.device)
        full_input = torch.cat([inputs["input_ids"], new_ids_t], dim=1)
        out = assistant_model(input_ids=full_input, output_hidden_states=True)
        prompt_len = inputs["input_ids"].shape[1]
        drafted_logits = out.logits[0, prompt_len - 1: prompt_len - 1 + len(new_ids)]
        drafted_hidden = out.hidden_states[-1][0, prompt_len - 1: prompt_len - 1 + len(new_ids)]

        round_losses = []
        for r in round_records:
            pos = r["src_positions"]
            if r["tag"] == "direct":
                p = pos[0]
                kl = reverse_kl_direct(drafted_logits[p], r["target_logits"])
                round_losses.append(kl)
                kl_sum += kl.item(); kl_n += 1
                ratio_q = F.softmax(drafted_logits[p], dim=-1)[new_ids[p]]
                ratio_p = F.softmax(r["target_logits"].to(drafted_logits.device), dim=-1)[r["proposed_target_id"]]
                ratio_label = torch.clamp(ratio_p / (ratio_q.detach() + 1e-12), max=1.0)
                bce = F.binary_cross_entropy_with_logits(head(drafted_hidden[p]), ratio_label)
                round_losses.append(bce)
                bce_sum += bce.item(); bce_n += 1
            else:
                for p in pos:
                    log_probs = F.log_softmax(drafted_logits[p], dim=-1)
                    nll = -log_probs[new_ids[p]]
                    round_losses.append(nll)
                    nll_sum += nll.item(); nll_n += 1

        if round_losses:
            # scale by 1/n_rounds so accumulated gradients approximate the
            # same mean-over-the-whole-batch objective as a single combined
            # backward() would have given, just computed incrementally
            (sum(round_losses) / len(round_losses) / n_rounds).backward()
        del out, drafted_logits, drafted_hidden
        torch.cuda.empty_cache()

    if kl_n + nll_n + bce_n == 0:
        assistant_model.eval()
        return None

    optimizer.step()
    assistant_model.eval()
    return {
        "kl": kl_sum / kl_n if kl_n else None,
        "nll": nll_sum / nll_n if nll_n else None,
        "bce": bce_sum / bce_n if bce_n else None,
    }


print(f"\n=== Starting GSM8K replication run (TRAIN_MODE={TRAIN_MODE}) ===", flush=True)
metrics_log = []
for qi, q in enumerate(questions):
    target_prompt = build_target_prompt(q)
    target_ids = target_tokenizer(target_prompt, return_tensors="pt").to(target_model.device)["input_ids"]
    prompt_len = target_ids.shape[1]

    cache = DynamicCache(config=target_model.config)
    assistant_model.eval()
    with torch.no_grad():
        prefill_out = target_model(input_ids=target_ids, past_key_values=cache, use_cache=True)
    cache = prefill_out.past_key_values
    next_pred_logit = prefill_out.logits[0, -1].float().cpu()

    partial_answer_text = ""
    question_records = []
    n_calls = 0
    while target_ids.shape[1] - prompt_len < MAX_NEW:
        records, partial_answer_text, target_ids, cache, next_pred_logit, hit_eos = collect_round(
            q, partial_answer_text, target_ids, cache, next_pred_logit)
        question_records.extend(records)
        n_calls += 1
        if hit_eos:
            break

    tok_per_call = (target_ids.shape[1] - prompt_len) / max(n_calls, 1)
    stats = train_step(question_records) if (TRAIN_MODE and question_records) else None

    log_line = f"Q{qi}: tok/call={tok_per_call:.3f} n_records={len(question_records)}"
    if stats:
        log_line += f" kl={stats['kl']} nll={stats['nll']} bce={stats['bce']}"
    print(log_line, flush=True)
    metrics_log.append({"qi": qi, "tok_per_call": tok_per_call,
                         "n_records": len(question_records), **({} if not stats else stats)})
    gc.collect(); torch.cuda.empty_cache()

suffix = "train" if TRAIN_MODE else "control"
with open(f"gsm8k_replication_{suffix}_metrics.json", "w") as f:
    json.dump(metrics_log, f, indent=2)
print(f"\nSaved to gsm8k_replication_{suffix}_metrics.json", flush=True)
print("GSM8K replication run completed without error.", flush=True)
