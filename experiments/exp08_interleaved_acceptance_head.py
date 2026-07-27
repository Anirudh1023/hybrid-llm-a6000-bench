"""
Experiment 8: build OmniDraft's INTERLEAVED training variant (Algorithm 4)
for the acceptance-prediction head, not just the JOINT variant we already
tested -- see ONLINE_TRAINING_PLAN.md "Gap 2" for the full context.

Our existing acceptance-head test (t4-session/scripts/adaptive_draft_length.py)
trains the head jointly with the drafter's own hybrid-distillation loss,
matching OmniDraft's Algorithm 3 exactly. Result was -6.3% vs a fixed
window -- the head-controlled adaptive window stopped after ~1.7 tokens on
average, far too conservative.

OmniDraft's own diagnosis of joint training's specific failure mode
matches this almost exactly (section 4.2): "training only on incoming data
without a stable corrective feedback... the acceptance prediction head
might not have converged well" and "the joint variant has higher
acceptance rates... indicating it might be underestimating the acceptance
probability, leading to wrongful early exit." Their fix is Algorithm 4
(Appendix C) -- interleaved training:

    Given target p, drafter q_theta, acceptance head f_phi,
    data stream S, distillation buffer Q, ADAPTIVE-DRAFTING buffer R
    (max size N), batch size B, update interval I.

    while True:
        x ~ S; i += 1
        y = speculative_decode_with_adaptive_drafting(x)
        Q.append((x, y))
        if i mod I == 0:
            update theta on Q with L_distill          # drafter update, same cadence as joint
            push Q into R; evict oldest if |R| > N     # NEW: accumulate into a LARGER buffer
            Q = {}
        else:
            R_B = sample_batch(R, B)                   # NEW: head trains MORE often, on older+newer mixed data
            labels l = min(1, p(y)/q_theta(y)) on R_B   # relabeled fresh each time using the CURRENT drafter
            update phi on (x, y, l) in R_B with L_adapt
            # note: drafter theta is NOT updated in this branch

The key structural difference from joint: the acceptance head updates on
EVERY iteration (using buffer R), but the drafter only updates every I
iterations (using buffer Q) -- giving the head "slowly moving labels"
instead of chasing a drafter that shifts under it every single step. This
is genuinely new code, not a hyperparameter tweak -- budget more time for
this than exp07.

THIS IS A STUB -- the buffer logic below is unimplemented (NotImplementedError
markers at each point real code is needed). Steps to actually build this:

1. Copy t4-session/scripts/adaptive_draft_length.py as the base (it already
   has: fixed-window baseline, the AcceptanceHead class, priming/eval split
   logic, the head-only optimizer). Rename functions as needed.
2. Add a buffer class or plain list `R` with a max size N (try N = a few
   questions' worth of records, e.g. N=200-500 records -- tune empirically,
   the paper doesn't give a specific N).
3. Change the training loop: keep the existing per-question drafter update
   (train_step_head in the current script, or the full hybrid-loss
   train_step if training the drafter jointly with distillation too -- the
   plan's Gap 2 is specifically about the ACCEPTANCE HEAD's training
   cadence, so keeping the drafter frozen as in adaptive_draft_length.py's
   original design, OR extending to a real jointly-trained drafter, is a
   judgment call -- either is a legitimate test of the interleaved-vs-joint
   distinction for the head specifically).
4. After each question's records are collected, push them into R (evict
   oldest if over N). Then, EVERY question (not just every I), sample a
   batch from R and do a head-only update using FRESH labels computed with
   the CURRENT drafter state -- this is the core of "interleaved."
5. Run the SAME eval methodology as the original test (prime on topics 0-2,
   eval fixed-window vs adaptive on topics 3-4, threshold sweep) so the
   result is directly comparable to the original -6.3% (joint) number.
6. Compare tok/call: does interleaved beat -6.3%? Does it beat the fixed
   window entirely, matching the paper's own finding that interleaved beats
   joint on speedup in 3 of 4 of their tasks?
"""

raise NotImplementedError(
    "This is a plan stub, not tested code -- see the module docstring for "
    "the exact Algorithm 4 structure and build steps. Start from "
    "t4-session/scripts/adaptive_draft_length.py and add the R buffer / "
    "interleaved update cadence described above, on real GPU hardware."
)
