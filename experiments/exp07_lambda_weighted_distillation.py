"""
Experiment 7: retest online distillation with the paper's actual lambda=0.2
weighting -- see ONLINE_TRAINING_PLAN.md "Gap 1" for the full context.

Our existing t4-session/scripts/online_train_live.py and gsm8k_replication.py
average every token's loss UNWEIGHTED across a round (KL terms on
direct-mapped tokens and NLL terms on n-gram tokens all pooled together,
implicitly lambda=1). OmniDraft (arxiv 2507.02659, Eq. 3/4) explicitly
weights the NLL term by lambda, and section 4.3.4 states plainly: "we
noticed tuning the scaling factor lambda becomes critical... we fix a
lambda=0.2 for all the tasks, across all experiments." We've never tested
with this weighting.

THIS IS A STUB, not a tested implementation -- the actual change is small
(one scale factor) but needs to be made and run on real GPU hardware,
which this environment doesn't have. Steps to actually run this:

1. Copy t4-session/scripts/online_train_live.py and
   t4-session/scripts/gsm8k_replication.py into this experiments/ directory
   (or import their train_step logic directly -- they're standalone
   scripts, not designed as a package, so copying is simpler than
   importing).
2. In train_step's loss-accumulation loop, find where n-gram NLL losses
   are appended to round_losses (untouched) alongside KL losses. Scale the
   NLL losses by LAMBDA_NGRAM = 0.2 before appending, e.g.:
       nll = -log_probs[new_ids[p]]
       round_losses.append(LAMBDA_NGRAM * nll)   # was: round_losses.append(nll)
   Leave the KL term (direct-mapped tokens) unscaled -- lambda only
   weights the n-gram term per the paper's Eq. 3: L_DM + lambda * L_N-gram.
3. Run BOTH the GSM8K replication (train vs control, same methodology as
   before -- see RESEARCH_LOG.md's 2026-07-26 GSM8K entry for the exact
   comparison: identical 40-question sequence, same starting weights,
   first-half vs second-half tok/call) and the conversational-domain live
   loop (same dataset, same eval split as before), with this one change,
   otherwise identical to the original runs.
4. Compare against the ORIGINAL (lambda=1, implicit) results already
   documented in PROJECT_README.md / RESEARCH_LOG.md:
     - GSM8K: control 1.158->1.164 (flat), train 1.156->1.218 (+5.4%)
     - Conversational (reverse-KL): 1.489->1.431 (-3.9%)

Interpretation guide is in ONLINE_TRAINING_PLAN.md -- briefly: GSM8K result
getting stronger validates the paper's tuning claim; the conversational
null CHANGING would be the headline finding (reopens "domain-gated" as
partly a hyperparameter artifact); neither changing strengthens confidence
that domain, not lambda, is the real gate -- proceed to exp08/exp09 either
way since those test independent gaps.
"""

LAMBDA_NGRAM = 0.2  # OmniDraft section 4.3.4 -- fixed across all their tasks

raise NotImplementedError(
    "This is a plan stub, not tested code -- see the module docstring for "
    "exact steps. Copy online_train_live.py and gsm8k_replication.py from "
    "t4-session/scripts/, apply the LAMBDA_NGRAM scale to the n-gram NLL "
    "term only, and run both on real GPU hardware."
)
