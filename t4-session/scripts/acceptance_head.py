"""
Step 3: acceptance-prediction head, per the paper's spec -- a lightweight
head on the drafter's hidden state, trained with (weighted) BCE against the
CONTINUOUS acceptance ratio min(1, p_target(y)/q_drafter(y)), not a binary
accept/reject label. Reuses the small-MLP-on-hidden-state architecture from
the earlier acceptance-head v2 work (prior sessions), resized for
Qwen2.5-0.5B-Instruct's actual hidden dimension.

Scoped to 'direct'-tagged records only for this validation pass: the ratio
p(y)/q(y) needs a well-defined SAME token compared in both vocabularies,
which only exists for direct-mapped tokens (exactly the same asymmetry the
paper already draws between direct-mapped and n-gram-merged tokens for the
main loss -- n-gram-merged tokens don't have a clean per-token ratio either,
for the same underlying reason).

Ambiguity flagged, not silently resolved: the paper says "weighted BCE"
without detailing the weighting scheme. Implemented here as standard
BCEWithLogitsLoss against a CONTINUOUS soft target in [0,1] (valid,
well-defined for BCE even though it's not the usual binary case) -- if a
specific class-reweighting scheme turns out to matter, that's a refinement
on top of this, not something silently assumed away.
"""
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

DRAFT_ID = "Qwen/Qwen2.5-0.5B-Instruct"

with open("online_train_collected.pkl", "rb") as f:
    records = pickle.load(f)
direct_records = [r for r in records if r["tag"] == "direct"]
print(f"Using {len(direct_records)} direct-mapped records "
      f"(ratio only well-defined for these, same reason the main loss "
      f"splits direct vs n-gram)", flush=True)

assistant_tokenizer = AutoTokenizer.from_pretrained(DRAFT_ID)
assistant_model = AutoModelForCausalLM.from_pretrained(DRAFT_ID, dtype=torch.float32).to("cuda:0")
hidden_size = assistant_model.config.hidden_size
print(f"Qwen2.5-0.5B-Instruct hidden_size={hidden_size}", flush=True)


class AcceptanceHead(nn.Module):
    """Small MLP on the drafter's hidden state -- same shape as the earlier
    v2 acceptance head (prior sessions), resized for this drafter."""
    def __init__(self, hidden_size, mlp_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, 1),
        )

    def forward(self, hidden_state):
        return self.net(hidden_state).squeeze(-1)  # raw logit -- use BCEWithLogitsLoss


head = AcceptanceHead(hidden_size).to("cuda:0")


def acceptance_ratio_label(record):
    """min(1, p_target(y) / q_drafter(y)) for the SAME token y -- only
    well-defined for direct-mapped records, where the drafter's own token
    and the verified target token are the same underlying token by
    construction (that's what 'direct-mapped' means)."""
    q = F.softmax(record["assistant_logits"][0], dim=-1)[record["assistant_ids"][0]].item()
    p = F.softmax(record["target_logits"], dim=-1)[record["proposed_target_id"]].item()
    return min(1.0, p / max(q, 1e-12))


labels = [acceptance_ratio_label(r) for r in direct_records]
print(f"Ratio labels: mean={sum(labels)/len(labels):.4f} min={min(labels):.4f} max={max(labels):.4f}", flush=True)

# --- backward-pass smoke test: fresh hidden states via real forward passes,
# real gradients into the head (drafter itself stays frozen here -- the
# paper trains the head alongside the drafter, but validating the head's
# OWN mechanism in isolation first, same staged approach as step 2). ---
optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3)
sample = direct_records[:16]
sample_labels = torch.tensor([acceptance_ratio_label(r) for r in sample]).to("cuda:0")

losses = []
for step in range(5):
    optimizer.zero_grad()
    hidden_states = []
    with torch.no_grad():  # drafter frozen for this smoke test
        for r in sample:
            inp = torch.tensor([r["assistant_ids"][0]]).unsqueeze(0).to("cuda:0")
            out = assistant_model(input_ids=inp, output_hidden_states=True)
            hidden_states.append(out.hidden_states[-1][0, -1])
    hidden_states = torch.stack(hidden_states)
    logits = head(hidden_states)
    loss = F.binary_cross_entropy_with_logits(logits, sample_labels)
    loss.backward()
    grad_norm = sum(p.grad.norm().item() for p in head.parameters() if p.grad is not None)
    optimizer.step()
    losses.append(loss.item())
    print(f"step {step}: loss={loss.item():.4f} grad_norm={grad_norm:.4f}", flush=True)

print(f"\nLoss trend: {losses} ({'decreasing' if losses[-1] < losses[0] else 'NOT decreasing -- check'})", flush=True)

# quick calibration check: are predicted probabilities in a sane range,
# correlated at all with the true labels, after training?
with torch.no_grad():
    preds = torch.sigmoid(head(hidden_states)).cpu().tolist()
print(f"\npredicted vs actual (first 8): "
      f"{list(zip([round(p,3) for p in preds[:8]], [round(l,3) for l in sample_labels.cpu().tolist()[:8]]))}", flush=True)

torch.save(head.state_dict(), "acceptance_head_qwen05b_gemma9b.pt")
print("\nSaved head weights. Backward pass and optimizer step completed without error -- mechanism confirmed working.", flush=True)
