# Hybrid LLM speculative decoding — living conclusions

Started 2026-07-23, carried forward from a T4-based diagnostic session
(2026-07-22/23). This is meant to be a living document — update it as new
experiments run, don't just append; correct earlier entries when new data
contradicts them (that happened repeatedly during the T4 session, and
every correction is noted below rather than hidden).

## What's proven, high confidence

- **Gemma-1 is architecturally bad for speculative decoding; Gemma-2 isn't.**
  Global-attention-only, MHA/MQA, no logit soft-capping (Gemma 1) vs GQA +
  alternating sliding-window/global attention (Gemma 2). Confirmed via a
  direct controlled test with the same drafter, only the target generation
  changed (0.852x → 1.196x). If deploying against a Gemma-family server
  model, use Gemma 2 or newer.
- **Bigger drafter:target ratio genuinely helps**, at least across the
  range tested (11.7:1 → 15:1 → 18.2:1 gave 0.977x → 1.295x → 1.479x,
  monotonic, not noise). NOTE the 11.7:1 data point is confounded (see
  below) — treat the 15:1/18.2:1 pair as the trustworthy part of this trend.
- **`config.vocab_size` equality, not tokenizer class or vocab-key overlap,
  gates HF's native assisted-generation path.** Two checkpoints of the
  literal same tokenizer can have different `config.vocab_size` (embedding
  padding) and get silently routed through the cross-vocab translator
  anyway. Always verify with `src/vocab_utils.py` before assuming a pair
  hits the native path.
- **An unmerged LoRA adapter costs real, measurable overhead per forward
  pass** — 1.443x the wall-clock time of the base model, in isolated
  drafter-only generation (no SD, no target model), 30/30 prompts,
  consistent. `merge_and_unload()` recovers essentially all of it (merged
  model measured 0.976x of base — i.e., fully gone, within noise).
- **Big-to-small handoff in a router+SD system is structurally free** —
  not a hypothesis, a necessary consequence of the small model being the
  drafter during every big-model turn (its cache is kept in sync as a
  byproduct of SD working at all, not something built on purpose).
- **LinUCB contextual bandit mechanics work correctly** as implemented
  (`linucb_router.py`, separate repo) — Sherman-Morrison updates, arm
  selection, tested on real data. The router's *calibration* (reward
  function constants) is a separate, still-open problem — see below.

## Corrected conclusions — flagged explicitly so they don't get miscited

- ~~"Same-family SD performed worse than cross-family (0.977x vs 1.295x+)"~~
  **RETRACTED.** The same-family test (Qwen3-0.6B + Qwen2.5-7B-Instruct) was
  never actually running through HF's native path — its `config.vocab_size`
  didn't match, so it silently went through the same cross-vocab translator
  as the Gemma tests, just with much better vocab compatibility. It also
  used a smaller ratio (11.7:1) than the cross-family comparisons it was
  being judged against. Two confounds stacked, neither isolated. **A
  properly controlled same-family-at-matched-ratio test has never been
  run** — that's `exp02` in this repo, specifically because it didn't
  happen on the T4.
- ~~"+10.6% tok/call means reverse-KL training helps"~~ **CORRECTED, three
  times now.** (1) The +10.6% tok/call gain came with a **wall-clock
  regression** (1.313x untrained → 1.042x trained-unmerged-adaptive, ~21-28/30
  prompts worse depending on the exact run) — tok/call and wall-clock
  diverged, only wall-clock is what a user experiences. (2) Fixing draft
  length alone recovered PART of the gap (1.042x → 1.210x) but not all of
  it. (3) Merging the LoRA adapter alone ALSO recovered a similar amount
  (1.042x → 1.208x) but ALSO not all of it — landing at almost the exact
  same point as the fixed-length fix, independently. **This strongly
  suggests the two costs (LoRA per-step overhead, adaptive-heuristic
  overshoot) are separate and additive, and neither fix alone is
  sufficient — you likely need BOTH together.** That combined test
  (merge + fixed length, simultaneously) was never run on the T4 before
  this repo was started — it's exactly what `exp06` does, at real scale.
  Current honest state: reverse-KL training's wall-clock verdict is STILL
  OPEN, narrowed to one well-specified remaining test, not resolved.

## What's real but still unresolved

- **Whether reverse-KL training produces a genuine net wall-clock win once
  BOTH known fixes (merged LoRA + appropriate draft length) are applied
  together** — never cleanly tested end-to-end on the T4 (each fix was
  tested in isolation, sequentially, as it was discovered; each partial
  reversal is real but the FULLY-fixed number was still pending T4-side
  when this repo was started). `exp06` here is that test, done properly.
- **Does the adaptive-draft-length overshoot problem matter less at bigger
  target scale?** Reasoned (bigger targets are more memory-bandwidth-bound
  per forward pass, so a too-long verification window should cost
  proportionally less) but never measured — `exp03`.
- **Router reward calibration** — currently uses placeholder cost constants
  (not real latency/$ numbers), picks the escalate arm ~93% of the time for
  reasons unrelated to real cost. Separate track from everything in this
  repo; tracked in the main project notes, not re-solved here.

## Still missing entirely (structural gaps, not "hasn't been tested yet")

These require actual new engineering, not just a bigger GPU — flagging so
nobody mistakes "we ran a benchmark" for "we built the real thing":

1. **OmniDraft's online n-gram cache.** Our translator only handles direct
   1:1 token correspondences (~25% of Gemma's vocab against Qwen3, measured
   directly). The other ~75% of continuation positions get ZERO training
   signal in the reverse-KL loss, structurally — not because the n-gram
   loss term is unimplemented (though it is, currently stubbed at 0), but
   because you cannot even DEFINE that loss without a mechanism (the
   n-gram cache) that identifies what the correct multi-token
   correspondence even is. This is plausibly a bigger lever than anything
   testable in this repo — worth prioritizing as actual engineering work
   after these benchmarks land, not instead of them.
2. **OmniDraft's trained acceptance-prediction head.** Explicit P(rejection)
   estimation with a tunable early-exit threshold, replacing HF's generic
   heuristic entirely (not just fixing its draft-length parameter). `exp03`
   here tests whether a FIXED length is good enough as a stopgap; it is not
   the same thing as building the real trained head.
3. **Conversational context in the router.** Every routing decision is
   currently treated as independent — no signal about turn count since
   last escalation, cumulative acceptance trend, or recently-established
   facts. Needs wiring the router's feature vector to the same
   conversational-state representation used elsewhere in the project
   (Component C / capsule design). Not attempted in this repo.
4. **Small-to-big handoff via prefill-as-verification.** Designed
   (treating the small model's solo stretch as one giant speculative draft,
   using the divergence point as both a catch-up boundary and a free
   audit signal), reuses the same vocab-translator machinery — but zero
   code, zero test, anywhere.
5. **Continuous-tracking system architecture** (both models always
   running, passive sync via cheap prefill) — designed at length, not built.

## Open questions this repo is specifically trying to close

- [ ] Does a literature-scale ratio (28:1+) reproduce the 1.78-1.91x
      cross-family speedups reported elsewhere, on OUR harness? (`exp01`)
- [ ] Same-family vs cross-family, ratio properly matched — does family
      matter at all once ratio is controlled for? (`exp02`)
- [ ] With merge + draft-length fixes both applied, does reverse-KL
      training show a genuine wall-clock win? (`exp06`)
- [ ] Does 8-bit quantization's cost (never isolated on the T4, which had
      no choice but to use it) turn out to be a meaningful fraction of any
      of the above results? (`exp05`)

Fill in results below as experiments complete:

### exp01 — large target ratio
*(pending)*

### exp02 — matched-ratio family test
*(pending)*

### exp03 — fixed vs adaptive draft length at scale
*(pending)*

### exp04 — LoRA overhead at scale
*(pending)*

### exp05 — quantization isolation
*(pending)*

### exp06 — full corrected reverse-KL pipeline
*(pending — this is the big one)*
