# Online joint training — faithful OmniDraft replication plan

This is a handoff plan, written for whichever Claude Code session picks up
the A6000 work next. It's the direct output of a paper-reading session
(2026-07-27/28) that pulled the actual OmniDraft paper (arxiv 2507.02659,
full text in this repo's context, not a summary) and compared it line by
line against what we actually built in `t4-session/scripts/`. Three real,
specific gaps came out of that comparison — this plan is how to close
them, in priority order, cheapest and most informative first.

**Read this alongside**: `PROJECT_README.md` §6.12 (the online-training
results this plan is trying to improve on) and `RESEARCH_LOG.md`'s
2026-07-26 online-training entries (the actual code history, bugs found).

---

## Where we actually stand vs. the paper, precisely

We built the real mechanism — n-gram cache, hybrid distillation loss (KL
on direct-mapped tokens + NLL on n-gram-merged tokens), an acceptance-
prediction head, a live training loop, LoRA. This is NOT the stale stub in
`src/reverse_kl_train.py` (that file's own docstring says its n-gram term
is "STILL STUBBED... contributes 0" — true when it was written, no longer
true). **The real, complete version is `t4-session/scripts/online_train_live.py`**
(plus `online_train_collect.py`, `hybrid_loss.py`, `acceptance_head.py`
for the individual pieces) — start from these, not from `src/`.

Result so far: real, controlled +5.4% tok/call on GSM8K (matching the
paper's own validated domain), null-to-negative on open conversational
Q&A across three loss/data configurations. Three specific implementation
gaps, found by comparing our code against the paper's exact equations and
algorithms, could each independently explain part of the conversational
null — none of them have been tested yet.

### Gap 1 — we never applied the paper's λ weighting

Our `train_step` (in `online_train_live.py`) averages every token's loss
unweighted across a round — KL terms (direct-mapped tokens) and NLL terms
(n-gram tokens) all pooled into one `sum(round_losses) / len(round_losses)`.
That's an implicit λ=1. The paper (§4.3.4): *"we noticed tuning the
scaling factor λ becomes critical... we fix λ=0.2 for all tasks, across
all experiments."* Their loss (Eq. 3/4):

```
L_cross_vocab_distill(θ) = L_DM(θ) + λ · L_N-gram(θ)
```

where `L_DM` is the KL term on direct-mapped tokens and `L_N-gram` is the
NLL term on n-gram-merged tokens. We've been training with λ effectively
uncontrolled, not their tuned 0.2.

### Gap 2 — we only ever built "joint" training, never "interleaved"

The paper defines two variants (§3.3, Algorithms 3 and 4 in Appendix C):

- **Joint** (what we built): one combined update per batch,
  `L_joint = L_distill + L_adapt`, both the drafter and the acceptance
  head trained together on the same online data each step.
- **Interleaved** (not built): the drafter still updates on the same
  cadence, but the acceptance head updates MORE frequently, sampling from
  a larger buffer `R` (max size `N`) that accumulates data across several
  drafter-update cycles instead of just the current batch — deliberately
  giving the head slower-moving labels.

The paper's own diagnosis of joint training's failure mode: *"training
only on incoming data without a stable corrective feedback... the
acceptance prediction head might not have converged well to align with
the current distilled drafter, leading to sub-optimal performance."* Their
Table 2: interleaved beats joint on speedup in 3 of 4 tasks, despite joint
having HIGHER raw acceptance rate — *"the joint variant has higher
acceptance rates... indicating it might be underestimating the acceptance
probability, leading to wrongful early exit."* That "wrongful early exit"
description matches our own acceptance-head result (-6.3% tok/call,
controller stopped after ~1.7 tokens average) almost exactly.

### Gap 3 — the acceptance head has never been validated cross-vocabulary, by anyone, including the paper's own authors

Check the paper's Table 8: Vicuna-7B's vocabulary size is 32000, and its
intersection with Llama-68M is ALSO 32000 — full overlap, same tokenizer
(Vicuna is fine-tuned from Llama). Every acceptance-head/adaptive-drafting
result in the paper (Table 2, Table 11) is measured on Llama-68M +
Vicuna-7B. The paper says so itself, in §4.4 (Limitations): *"we are
working towards incorporating an explicit adaptive head in the
cross-vocabulary setting"* — listed as FUTURE work. We tested the
acceptance head cross-vocabulary (Qwen0.5B/Gemma-9b) directly, with no
same-vocab baseline first. Our null result conflates two separate
questions: is the acceptance-head code correct at all, and does
cross-vocab specifically break it. Never isolated.

### A fourth, smaller gap worth folding in: the streaming update structure

The paper's actual online loop (Algorithm 2, Appendix C) batches examples
into a buffer and updates every `I` examples, not after every single one:

```
while True:
    x ~ S; i += 1
    y = cross_vocab_speculative_decode(x)
    Q.append((x, y))
    if i mod I == 0:
        update θ on Q with L_cross_vocab_distill
        Q = {}
```

Table 10's hyperparameters list `batch size = 8` — almost certainly this
`I`. Our loop updates after every single question/turn (`I=1` in their
notation). Worth testing whether batching ~8 turns per update (more
stable gradient estimate, matches their actual reported setup) changes
anything, though this is lower priority than gaps 1-3 since it's a
hyperparameter, not a structural difference.

---

## The plan, in priority order

### Phase 1 — λ=0.2 retest (cheapest, do this first)

**Change**: in `online_train_live.py`'s `train_step`, weight the n-gram
NLL terms by `λ=0.2` before combining with the KL terms, matching Eq. 3
exactly instead of an unweighted average. One-line-scale change, not a
rebuild.

**Run**: rerun both `gsm8k_replication.py` (the real, controlled
GSM8K win) and `online_train_live.py` (conversational domain, currently
null) with this fix, same train/control structure as before (first-half
vs second-half tok/call, or train-vs-control on identical question
sequences — see `RESEARCH_LOG.md` for the exact comparison method used
each time).

**What the result means**:
- GSM8K gets stronger/more stable → confirms λ tuning matters as the
  paper claims, and our earlier +5.4% was achieved despite a suboptimal
  hyperparameter (makes that result MORE credible, not less).
- Conversational domain's null result changes → the earlier
  "domain-gated" conclusion was partly an artifact of an untuned
  hyperparameter, not a true ceiling. This would be a big finding —
  reopen the conversational-domain thread properly if so.
- Neither changes → strengthens confidence that domain, not λ, is the
  real gate. Proceed to Phase 2/3 regardless either way, since those test
  independent gaps.

**Suggested file**: `experiments/exp07_lambda_weighted_distillation.py`
(stub in this repo, see below) — copy `online_train_live.py` and
`gsm8k_replication.py` from `t4-session/scripts/`, add the λ scale to the
NLL term, keep everything else identical for a clean before/after
comparison.

### Phase 2 — build the interleaved variant (Algorithm 4), retest the acceptance head

**Build**: implement Algorithm 4 precisely — a drafter-update buffer `Q`
(current batch, same cadence as before) and a SEPARATE, larger
acceptance-head buffer `R` (max size `N`, evict oldest when full) that
accumulates across multiple drafter-update cycles. On each step: if it's
a drafter-update step, train the drafter on `Q` as before, then push `Q`
into `R` (evicting oldest if `|R| > N`); otherwise, sample a batch from
`R`, compute fresh labels `l = min(1, p(y)/q_θ(y))` using the CURRENT
drafter (not stale labels), and update only the head.

**Run**: on the same conversational domain and eval set as the original
(null, -6.3%) acceptance-head test, with everything else held constant —
same threshold sweep methodology from `adaptive_draft_length.py`, just
interleaved instead of joint training underneath the head.

**What the result means**: if interleaved closes some or all of the
-6.3% gap, that confirms the paper's own diagnosis (joint training's
"wrongful early exit" from unstable moving labels) is exactly what we
hit, not a domain-specific failure of the acceptance-head idea itself. If
it doesn't help, the acceptance-head problems run deeper than the
joint-vs-interleaved distinction.

**Suggested file**: `experiments/exp08_interleaved_acceptance_head.py`
(stub below) — needs real new code (the buffer logic isn't in any
existing script), budget more time for this one than Phase 1.

### Phase 3 — isolate cross-vocab from implementation correctness

**Run**: the exact same acceptance-head test (`adaptive_draft_length.py`'s
methodology — prime on some questions, eval fixed-window vs
head-controlled-adaptive on held-out questions) on a SAME-VOCAB pair
first. We already have one on hand with zero suppression: Qwen2.5-0.5B +
Qwen2.5-3B-Instruct (matched `vocab_size=151936`, confirmed in earlier
T4-session testing — see `RESEARCH_LOG.md`'s "truly-native control test").
No cross-vocab translation involved at all — this isolates whether the
acceptance-head/adaptive-window CODE itself works correctly before
blaming cross-vocab specifically.

**What the result means**: if it works here (adaptive beats fixed,
head calibrates correctly), the problem is specifically the cross-vocab
extension — matches the paper's own admission that this was never
validated there either, and scopes future work precisely (fixing
cross-vocab acceptance-head calibration is real, unsolved, novel work,
not a bug to hunt for). If it FAILS even same-vocab, the acceptance-head
implementation itself has a bug independent of vocabulary, and that's
what to debug first, before any cross-vocab work resumes.

**Suggested file**: `experiments/exp09_same_vocab_acceptance_head.py`
(stub below) — should be the cheapest of the three builds, mostly a
model-ID swap plus removing the translation-layer machinery.

### Phase 4 (lower priority) — proper streaming batch size

Once phases 1-3 land, if there's still appetite: change the update
cadence in `online_train_live.py` from "every question" to "every I=8
questions" (buffering `(x, y)` pairs and updating once per 8, per
Algorithm 2 and Table 10's `batch size=8`), and see if the more stable
gradient estimate changes anything on the conversational domain. Lowest
priority of the four because it's a hyperparameter difference, not a
structural or diagnosed-failure-mode gap like 1-3.

---

## Order of operations

Phase 1 is cheap and should run first regardless — it's a one-line change
to an existing, already-working script, and its result changes how much
Phase 2/3 even matter (if λ=0.2 alone reopens the conversational domain,
that's the headline finding, and phases 2-3 become refinements rather
than the main story). Phases 2 and 3 are independent of each other and
can run in either order or in parallel if two GPUs are available on the
A6000 box.

## What NOT to do

Don't retrain from the ~205MB of checkpoints already sitting in
`vm-artifacts/` on the original machine (noted, not copied, in
`SCRIPT_INDEX.md`) — they predate the LoRA-vs-full-fine-tuning correction
and none of these three gaps were fixed when they were trained. Start
fresh from `t4-session/scripts/online_train_live.py` and the stub
experiments below.
