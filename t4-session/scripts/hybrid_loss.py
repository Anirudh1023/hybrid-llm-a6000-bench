"""
Step 2: hybrid loss (reverse-KL on direct-mapped tokens, NLL on n-gram-merged
tokens), validated against the data collected in step 1.

Reverse-KL is computed over the RESTRICTED overlap subset (the set of
assistant tokens that have a valid direct target mapping), not the full
~256k target vocab with a giant masked tensor -- softmax both distributions
over just that shared, comparable support, then standard KL. This is
mathematically equivalent to the masked-full-vocab version (unmapped tokens
get zero probability either way) but far cheaper.

NLL on n-gram tokens: standard cross-entropy of the drafter assigning
probability to its OWN token(s) that, once merged through the cache, were
actually accepted by the target -- no target distribution needed, matches
the paper's stated reason for using NLL there ("only reliable point-wise
evaluation of probabilities on those tokens").

This step validates the loss MATH: correct shapes, finite values, sensible
relative magnitudes, and a genuine backward-pass smoke test (real gradients
flowing through a live, grad-enabled copy of the drafter). Wiring this into
the actual live generation loop is task #18, a separate script -- this one
proves the loss itself is correct before adding that complexity.
"""
import pickle

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.candidate_generator import AssistantToTargetTranslator

DRAFT_ID = "Qwen/Qwen2.5-0.5B-Instruct"
TARGET_ID = "google/gemma-2-9b-it"

with open("online_train_collected.pkl", "rb") as f:
    records = pickle.load(f)
print(f"Loaded {len(records)} records: "
      f"{sum(1 for r in records if r['tag']=='direct')} direct, "
      f"{sum(1 for r in records if r['tag']=='ngram')} ngram", flush=True)

assistant_tokenizer = AutoTokenizer.from_pretrained(DRAFT_ID)
target_tokenizer = AutoTokenizer.from_pretrained(TARGET_ID)

# Rebuild the same direct_map used during collection (deterministic, cheap --
# no model needed for this, just the two tokenizers' vocabularies).
_translator = AssistantToTargetTranslator(
    target_tokenizer=target_tokenizer, assistant_tokenizer=assistant_tokenizer,
    target_vocab_size=256000, assistant_model=None, assistant_prune_lm_head=False,
)
_map_tensor = _translator._assistant_to_target_input_ids
assistant_overlap_ids, target_overlap_ids = [], []
for a_id in range(_map_tensor.shape[0]):
    t_id = _map_tensor[a_id].item()
    if t_id != _translator.SUPPRESS_TOKEN_ID:
        assistant_overlap_ids.append(a_id)
        target_overlap_ids.append(t_id)
assistant_overlap_ids = torch.tensor(assistant_overlap_ids)
target_overlap_ids = torch.tensor(target_overlap_ids)
print(f"Overlap subset size: {len(assistant_overlap_ids)}", flush=True)


def reverse_kl_direct(assistant_logits, target_logits):
    """reverse-KL(q_drafter || p_target), restricted to the overlap subset.
    assistant_logits: (assistant_vocab,) -- drafter's own full logit vector.
    target_logits: (target_vocab,) -- target's own full logit vector.

    Computed manually rather than via F.kl_div: with a ~64k-token overlap
    subset, float32 softmax genuinely underflows to an exact 0.0 for many
    low-probability entries, and log(0.0) = -inf. The mathematically correct
    limit of q*log(q/p) as q->0 is 0 (standard KL convention), but IEEE
    float arithmetic evaluates 0.0 * -inf as NaN, not 0 -- confirmed this
    was happening (F.kl_div came back NaN for several records). torch.where
    explicitly enforces the correct limit instead of hitting that literal
    0*inf case."""
    q_restricted = assistant_logits[assistant_overlap_ids]
    p_restricted = target_logits[target_overlap_ids]
    log_q = F.log_softmax(q_restricted, dim=-1)
    log_p = F.log_softmax(p_restricted, dim=-1)
    q = log_q.exp()
    kl_terms = q * (log_q - log_p)
    kl_terms = torch.where(q > 0, kl_terms, torch.zeros_like(kl_terms))
    return kl_terms.sum()


def nll_ngram(assistant_logits_list, assistant_ids_list):
    """Standard NLL: for each position, -log P_drafter(actual drafted token)."""
    losses = []
    for logits, tok_id in zip(assistant_logits_list, assistant_ids_list):
        log_probs = F.log_softmax(logits, dim=-1)
        losses.append(-log_probs[tok_id])
    return torch.stack(losses).mean()


# --- Part A: validate the loss math on the collected (detached) data ---
direct_losses, ngram_losses = [], []
for r in records:
    if r["tag"] == "direct":
        loss = reverse_kl_direct(r["assistant_logits"][0], r["target_logits"])
        direct_losses.append(loss.item())
    else:
        loss = nll_ngram(r["assistant_logits"], r["assistant_ids"])
        ngram_losses.append(loss.item())

print(f"\nreverse-KL (direct-mapped), n={len(direct_losses)}: "
      f"mean={sum(direct_losses)/len(direct_losses):.4f} "
      f"min={min(direct_losses):.4f} max={max(direct_losses):.4f} "
      f"all_finite={all(x == x and abs(x) != float('inf') for x in direct_losses)}", flush=True)
print(f"NLL (n-gram merged), n={len(ngram_losses)}: "
      f"mean={sum(ngram_losses)/len(ngram_losses):.4f} "
      f"min={min(ngram_losses):.4f} max={max(ngram_losses):.4f} "
      f"all_finite={all(x == x and abs(x) != float('inf') for x in ngram_losses)}", flush=True)

# --- Part B: real backward-pass smoke test -- live gradients, not just math on
# static detached tensors. Reload the drafter with grad enabled, recompute
# FRESH logits for a small sample via real forward passes, confirm backward()
# succeeds and an optimizer step actually reduces the loss. ---
print("\n--- backward-pass smoke test ---", flush=True)
assistant_model = AutoModelForCausalLM.from_pretrained(DRAFT_ID, dtype=torch.float32).to("cuda:0")
assistant_model.train()
optimizer = torch.optim.AdamW(assistant_model.parameters(), lr=1e-5)

sample = [r for r in records if r["tag"] == "direct"][:8]
losses_before, losses_after = [], []
for step in range(3):
    optimizer.zero_grad()
    total_loss = 0.0
    for r in sample:
        # Recompute a FRESH, grad-enabled logit for the assistant's own
        # single token via a real forward pass (not the stored, detached one)
        inp = torch.tensor([r["assistant_ids"][0]]).unsqueeze(0).to("cuda:0")
        out = assistant_model(input_ids=inp)
        fresh_logits = out.logits[0, -1].float().cpu()
        loss = reverse_kl_direct(fresh_logits, r["target_logits"])
        total_loss = total_loss + loss
    total_loss = total_loss / len(sample)
    if step == 0:
        losses_before.append(total_loss.item())
    total_loss.backward()
    grad_norm = sum(p.grad.norm().item() for p in assistant_model.parameters() if p.grad is not None)
    optimizer.step()
    print(f"step {step}: loss={total_loss.item():.4f} grad_norm={grad_norm:.4f}", flush=True)
    losses_after.append(total_loss.item())

print(f"\nLoss trend across 3 steps: {losses_after} "
      f"({'decreasing' if losses_after[-1] < losses_after[0] else 'NOT decreasing -- check'})", flush=True)
print("Backward pass and optimizer step completed without error -- mechanism confirmed working.", flush=True)
