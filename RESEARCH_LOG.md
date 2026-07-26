# Hybrid LLM (edge SLM + cloud LLM) — research notes

## 🔴 SESSION PAUSED (2026-07-21) — VM WILL BE TERMINATED, READ THIS FIRST TO RESUME

**What's saved locally** (in `vm-artifacts/` next to this file):
- `vm-artifacts/trained_drafter_v1/` — the actual trained Llama-68M drafter
  weights (fp16, full fine-tune, 105 LoCoMo prompts, nan-guarded). Downloaded
  successfully, 136MB safetensors file confirmed present.
- `vm-artifacts/scripts/` — every experiment script written this session
  (uag_experiment.py through uag_speedup_one.py, uag_lora_06b.py,
  prep_locomo_data.py, uag_full_pipeline.py, uag_distill.py,
  uag_distill_scaled.py, etc.)
- `vm-artifacts/data/` — train_prompts.json / heldout_prompts.json (the exact
  105/30 LoCoMo split used — reproducible anyway via `prep_locomo_data.py`
  with `random.seed(42)` if these are somehow missing, but saved directly too)
- `vm-artifacts/logs/` — raw experiment logs for reference

**What was IN PROGRESS when the session paused**: the final speedup
comparison (original untrained drafter vs. the newly-trained one, both vs.
plain baseline, on 15 held-out prompts, clean fp16 single-GPU) — split into
two sequential scripts (`uag_speedup_one.py` run twice, with args
`'JackFram/llama-68m' original` then `'trained_drafter_v1' trained`) after
repeated OOM issues from loading 3 models at once (Qwen2-7B fp16 alone eats
~14.4GB of the card's actual 14.56GB usable capacity — confirmed smaller than
the ~15.6GB assumed earlier — leaving only room for ONE drafter at a time,
not two or three simultaneously). This did NOT finish before the session
paused — check `vm-artifacts/logs/uag_speedup_original.log` and
`uag_speedup_trained.log` for whatever partial output was captured.

**To resume tomorrow on a fresh VM**:
1. New VM, new SSH tunnel — get connection details, add to `~/.ssh/config`
   (same `*.trycloudflare.com` wildcard pattern already there should just work
   for any new trycloudflare hostname).
2. Re-clone LoCoMo (`git clone --depth 1 https://github.com/snap-research/locomo.git`)
   or just re-upload the saved `train_prompts.json`/`heldout_prompts.json`
   directly (faster, and identical either way given the fixed seed).
3. Re-upload `vm-artifacts/scripts/*.py` to the new VM (or re-paste as needed).
4. **Re-upload `vm-artifacts/trained_drafter_v1/` if picking up the speedup
   comparison** — this is the one piece of real trained work product, don't
   redo the training from scratch, just re-run Phase 2 (the speedup
   comparison) against these saved weights.
5. Re-install: `transformers`, `accelerate`, `bitsandbytes`, `peft` (all pip
   installs, fast).
6. HF token: Gemma license was already accepted on the account this session —
   should carry over automatically once logged in again with the same account,
   but re-verify with `whoami()` before assuming.
7. **Next planned step**: finish the original-vs-trained-drafter speedup
   comparison (was mid-run), THEN launch the LoRA/Qwen3-0.6B/Gemma-7b-it run
   (`uag_lora_06b.py`, already written and saved, never executed yet —
   GPU was occupied by the speedup comparison when session paused).

**Lesson learned this session, worth remembering**: this VM's actual usable
GPU memory is ~14.56GB per T4, not the ~15.6GB assumed early on — budget
accordingly when loading multiple models simultaneously on a fresh VM (which
may have the same or different actual capacity — verify with
`torch.cuda.get_device_properties(0).total_memory` on the new VM rather than
assuming the same number carries over).

**CRITICAL CONSTRAINT (2026-07-21): only ~30 GPU-hours/week available (Kaggle
weekly quota).** Do NOT leave long unattended runs going, do NOT burn hours on
avoidable retry cycles — be deliberate about what actually needs to run before
launching anything. This should shape experiment design going forward: fewer,
better-planned runs rather than iterate-by-trial-and-error on the GPU clock.

**Final partial result from tonight, before the session was stopped**: the
original-vs-trained speedup comparison partially ran before hitting a NEW
memory issue — fragmentation accumulating over many sequential `generate()`
calls within one long-running process (not the "3 models loaded at once"
problem from before, a different one: repeated calls within a single process
slowly eating the ~150-200MB of slack until OOM). "Original" (untrained)
drafter got through 10/15 held-out prompts before crashing: speedups ranged
1.31x-3.56x, averaging **~2.28x** — notably higher than the earlier clean
baseline (~1.11x), plausibly because these are real diverse LoCoMo
conversational prompts rather than short instruction-style ones tested before.
"Trained" drafter comparison produced ZERO data — crashed immediately, likely
right after loading. **This is a partial, unverified sample — do not treat
2.28x as a trustworthy number, it needs a clean, complete rerun.**

**Fix needed for tomorrow's rerun**: the within-process fragmentation issue
means long sequential loops of `generate()` calls aren't safe as a single
unbroken run on this tight a memory budget. Either (a) call
`torch.cuda.empty_cache()` more aggressively between every single measurement,
not just between train steps, or (b) restart the Python process periodically
(e.g., every 5 prompts) to guarantee a clean CUDA context, or (c) reduce
`max_new_tokens` further to shrink the KV cache footprint per call. Try (a)
first as the cheapest fix before resorting to (b).

**Next session priorities, in order, given the tight time budget**:
1. ✅ DONE (2026-07-21, fresh VM): Rerun the original-vs-trained speedup
   comparison cleanly. See full result below.
2. NEXT: launch the LoRA/Qwen3-0.6B/Gemma-7b-it run (`uag_lora_06b.py`,
   already written, never executed).
3. Do NOT re-run anything already completed successfully (the training run,
   the clean fp16 baseline, the held-out generalization check, the speedup
   comparison below) — those results are captured, re-verifying wastes quota.

## FINAL SPEEDUP RESULT — clean, complete, fully comparable (2026-07-21, fresh VM)

**Fix applied**: separate GPU placement — target model pinned to `cuda:0`
alone (fits, ~14.4GB), drafter pinned to `cuda:1` alone (tiny) — avoids BOTH
earlier failure modes (device_map='auto' cross-GPU layer-splitting from the
very first experiments; single-GPU memory cramming from last night). Added
aggressive `torch.cuda.empty_cache()` + `gc.collect()` after EVERY measurement
call, not just training steps, to fix the within-process fragmentation that
crashed last night's attempt at prompt 11/15.

**Result — full 15/15 held-out prompts, both conditions, zero errors**:

| | Untrained drafter | Trained drafter |
|---|---|---|
| Avg speedup vs baseline | 1.117x | **1.172x** |
| Median speedup | 1.120x | 1.177x |
| Avg tokens/target-call | 1.47 | 1.56 |

**Training improved wall-clock speedup from ~1.12x to ~1.17x** — real,
positive, consistent with the held-out acceptance-rate gain measured
yesterday (+3.93% tok/call on a different metric/method). Per-prompt: trained
drafter won or tied on 13/15 prompts (only prompt 5 regressed, prompt 12 tied)
— a consistent effect, not an average carried by outliers.

**This is the number that closes the loop**: acceptance-rate improvement from
online distillation demonstrably translates into real wall-clock speedup, not
just an abstract intermediate metric. First fully clean, complete, apples-to-
apples comparison of the whole pipeline (plain UAG baseline → trained-drafter
UAG) on genuinely held-out LoCoMo conversational data.

## ⭐⭐ CURRENT PLAN v3 — CONTINUOUS TRACKING (2026-07-20, latest, supersedes
the structured-log-as-primary version below) — read this first

**Core correction from prior version**: switch-triggered catch-up (structured
log + training-free compression) is DEMOTED from primary mechanism to
RECOVERY/FALLBACK path only. The primary mechanism is now continuous
tracking — both models run at all times, so nobody ever goes cold and there is
no catch-up to perform in normal operation.

**Target**: small model on Android device, large model on server. Both cases
now share one architecture, differing only in how the passive-sync step works.

**Component A — Router (unchanged)**: small model's own logprob/entropy
confidence, tunable threshold, no training. Decides who "genuinely" answers
each turn — but does NOT gate whether models are running; both always are.

**The continuous-tracking mechanism**:
1. Small model drafts/generates continuously, every turn, regardless of who
   "officially" answers.
2. **Passive sync** (small model's turns): its tokens keep the large model
   warm, mechanism differs by case —
   - Same-family: REAL cache sharing (DroidSpeak selective layer reuse if
     same size/lineage; LatentMAS-style layer concatenation + terminal
     alignment + trained projection if sizes differ) — large model doesn't
     even need its own forward pass.
   - Heterogeneous: OmniDraft's n-gram cache used for TRANSLATION ONLY (no
     verification) — large model does process the translated tokens as
     prefill, but cheaply and improving via cache hits over time. This is OUR
     extension of OmniDraft, not validated in their paper as-is — flag as
     needs-testing, not assumed.
3. **Real SD switch** (router genuinely wants the large model): cache is
   already warm, no catch-up cost —
   - Same-family: classic lossless SD (BiLD's rollback policy as inspiration
     for correction logic; CS Drafting's horizontal-cascade compute
     allocation as an efficiency layer; speculative cascades' soft deferral
     as a lossy-but-flexible alternative).
   - Heterogeneous: OmniDraft's full draft-verify-reject loop.
4. **Switch back to small model**: free — it was drafting the whole time,
   never went stale.
5. **Recovery/fallback path (NOT primary, kept for when continuous tracking
   breaks — app killed, offline, genuine cold start)**: structured event log
   (ESAA-Conversational pattern) + training-free compression (SnapKV/
   PyramidKV) catch-up, exactly as designed in the prior version below —
   demoted to fallback-only, not deleted.

**Device-side implementation**: Ghidorah-style hetero-core (CPU/GPU/NPU)
scheduling for the small model's now-continuous drafting on Android, so it
doesn't starve other device work. DuoDecoding/Dovetail-style pipelining to
hide device-server round-trip latency (next-round drafting overlaps current-
round verification).

**Network layer**: continuous tracking now means continuous device-server
traffic, not just occasional handoff — WISV-style network-condition
awareness (back off tracking intensity under poor connectivity) and DSSD-
style payload reduction (don't transmit full probability distributions)
matter more than before. Early-exit latency-hiding (device drafts next batch
while cloud still verifying current one) applies to the real-SD switch step.

**Production/maintenance**: FlexSpec's frozen-draft/evolving-target
decoupling keeps the on-device drafter compatible as the server model gets
updated over time. ConfigSpec's profiling methodology for tuning draft
length/switch thresholds/hardware config once built.

**Long-conversation memory (orthogonal now, not catch-up-related)**: SnapKV/
PyramidKV/H2O, low-rank methods (Palu/LoRC/SVDq), MiniCache/xKV — both
caches still grow with conversation length regardless of continuous tracking,
these manage that, independent of the switching problem now.

**Orchestration**: LangGraph/OpenAI Agents SDK for coordinating the continuous
loop and the router's switch-into-SD trigger.

**Eval (LoCoMo-based, reframed around continuous operation)**:
- F1/ROUGE by CUMULATIVE large-model engagement (not raw switch count).
- Acceptance-rate trend: Path 2 (same-family, static cache reuse) expected
  flat-or-degrading; Path 1 (heterogeneous, OmniDraft online-adaptive)
  expected flat-or-IMPROVING — genuine point of contrast, both worth testing.
- Bandwidth/token cost of continuous tracking vs the old switch-only design —
  empirically test whether continuous tracking's recurring cost is actually
  worth it given real router-call frequency, don't assume either way.
- SEPARATE test of the recovery-path fallback: simulate a cold-start/dropped-
  connection scenario, verify structured-log catch-up still works when
  continuous tracking is interrupted.

**Path 2** restated: does CONTINUOUS same-family cache sharing accumulate
drift over a long session with many passive-sync/real-SD mode switches
(not discrete catch-up events anymore).

**Path 1** barely changes — was already implicitly about continuous
engagement, since OmniDraft's online adaptation needs cumulative exposure to
improve. Continuous tracking maximizes that exposure, making the "does it
improve" hypothesis more likely to show a real, measurable effect.

**Honest scope re-estimate**: this is MORE elegant (one unified design instead
of two) but NOT necessarily less work to build — continuous operation must
work reliably on every turn (new reliability burden vs occasional handoff),
OmniDraft's online training loop needs real implementation, AND the recovery
fallback still needs building on top (demoted, not eliminated). Realistic
estimate: back up to ~5-6 weeks, not the ~3-4 of the prior version. Elegance
and reduced-scope are not the same thing — being honest that this iteration
traded some simplicity for a better-justified design.

## EXPERIMENTAL RESULTS — UAG validation on real VM (2026-07-20/21)

**Infra**: Kaggle-style VM via Cloudflare tunnel (`*.trycloudflare.com` SSH
config pattern already in `~/.ssh/config`), 2x Tesla T4 (15.6GB each), Ubuntu
22.04, transformers 5.14.1, bitsandbytes installed for quantization tests. HF
token needs Gemma license accepted separately from having a valid token (gated
per-model, not just per-token). All experiment scripts left on VM
(`uag_experiment.py` through `uag_fp16_singlegpu.py`) as reference/paper trail.

**The journey (in order), because the wrong turns are as informative as the
right one**:
1. First test (Qwen3-0.6B draft / Gemma-7b-it target, huge vocabs 151K/256K):
   0.22-0.29x speedup — badly slower than baseline, despite correct output.
2. Swapped to OmniDraft's own tested pair (Llama-68M / Qwen2-7B, 32K/152K):
   still 0.22-0.33x — **ruled out vocab size as the driver**, since a much
   smaller vocab pairing showed the same magnitude of slowdown.
3. **Major find**: transformers ships TWO candidate generator classes for
   cross-vocab assisted generation — `AssistedCandidateGeneratorDifferentTokenizers`
   (SLEM-style, full decode/retokenize every step, used when `do_sample=False`)
   vs `UniversalSpeculativeDecodingGenerator` (TLI-style, direct token-ID
   lookup via a cached `AssistantToTargetTranslator`, used when `do_sample=True`).
   This is the exact SLEM/TLI distinction from the Timor et al. paper, already
   implemented in the library — selection is controlled entirely by `do_sample`.
4. Tried the fast path (`do_sample=True`) — hit a real CUDA device-side assert
   (scatter/gather index out of bounds) in the assistant's LM-head-pruning +
   top-k logits step. Tried two different top_k values (implicit default ~50,
   explicit top_k=1 via proper GenerationConfig) — both crashed identically.
   Concluded this is a genuine library bug in this version for this model
   pair, not fixable via simple parameter tweaks. Stopped chasing it, not
   blocking since the slow path works and gives a valid baseline.
5. Built our own cache wrapping `convert_source_tokens_to_target_tokens`
   (the expensive decode/retokenize step) — confirmed 100% cache hit rate on
   repeated content (cold pass misses exactly matched warm pass hits), but
   speedup did NOT improve at all (0.22-0.29x cold vs 0.22-0.25x warm). **This
   disproved the "translation step is the bottleneck" hypothesis directly.**
6. Fine-grained timing instrumentation: `get_candidates()` (drafting +
   translation + alignment combined) = only 6% of total time; translate() =
   0.07s; assistant.generate() = ~0s. **94% of time was unaccounted, in target
   verification + loop overhead.**
7. Found the actual cause: target model (Qwen2-7B) was split across BOTH GPUs
   via `device_map='auto'`, assistant model was on a different single GPU —
   cross-GPU communication overhead on every verification round.
8. Confirmed via single-GPU forcing (8-bit quantization to fit): speedup
   jumped to 1.11-1.14x for 2/3 prompts. But all 3 outputs mismatched baseline
   — attributed to 8-bit quantization noise amplifying known greedy-tie-
   breaking sensitivity (see below).
9. **Clean final test**: fp16 (no quantization), single GPU, just barely fit
   (15.37GB/15.6GB) — **speedup 0.95x-1.29x, averaging ~1.11x, ALL THREE
   outputs matched baseline exactly.** Clean, trustworthy, correct baseline
   established.

**Root cause, stated plainly**: the original catastrophic slowdown was caused
by cross-GPU model splitting on shared/ad-hoc infrastructure — NOT vocabulary
size, NOT translation overhead, NOT weak draft-target relationship, NOT
acceptance rate. Every hypothesis chased before finding this was a red herring,
confirmed wrong by direct measurement, not just superseded by a better guess.

**On losslessness**: mismatches only appeared under numerical-noise-inducing
conditions (quantization, or the original cross-GPU fp16 run) — the clean
single-GPU fp16 run matched exactly on all three prompts. Consistent with the
earlier theoretical point: losslessness is a distributional guarantee, not a
bitwise one, and numerical noise from quantization/cross-device computation
paths can trip greedy tie-breaking without indicating a real correctness bug.

**Reprioritization this forces**: since translation/caching is confirmed NOT
the bottleneck once GPU placement is correct, the n-gram cache is not the
highest-leverage next build (contrary to earlier assumption). The real lever
for improving beyond the ~1.11x clean baseline is **acceptance rate** —
exactly what OmniDraft's online hybrid distillation targets, not what the
cache targets. Build distillation next, cache later only if still needed.

**Sanity check**: ~1.11x average lines up closely with (and slightly beats)
OmniDraft's own reported baseline for their comparable "direct mapping only"
condition (0.94-1.04x) — good independent confirmation the measurement
pipeline is now trustworthy.

## DISTILLATION EXPERIMENT — first result was misleading, held-out test caught it
(2026-07-21)

**What was built**: simplified version of OmniDraft's online hybrid
distillation — NOT their exact dual loss (reverse-KL on direct-mapping tokens
+ NLL on n-gram tokens), instead standard NLL fine-tuning of the full 68M
drafter, using their confirmed reverse-translation formula
(`tokenize_drafter(detokenize_target(accepted_output))`) to build training
targets in the drafter's vocabulary, at per-FULL-RESPONSE granularity (not
per-round like their actual method) — justified by the fact that do_sample=False
assisted-generation output was already confirmed to exactly equal target-alone
output, so the final response is a valid proxy for "what the target accepted."

**Infra note**: fp16 single-GPU setup left zero headroom for training (backward
pass OOM'd even with 8-bit AdamW — the problem was the target's own fp16
footprint, not optimizer memory). Fixed by quantizing the TARGET to 8-bit for
training experiments specifically (frees several GB), keeping the drafter in
fp16 for training precision. Acceptable since 8-bit was already shown not to
distort the acceptance-rate signal much (1.11-1.14x vs fp16's 1.11x average),
only affects exact-text-match, not what this experiment measures.

**First result (6 train prompts, 3 epochs, evaluated on the SAME 6 prompts)**:
tok/call improved 1.46 → 1.54 avg (~+5%), 5/6 prompts positive. Looked
promising in isolation.

**Held-out test (same training, evaluated on 6 NEW prompts matched in style/
difficulty, never seen during training)**: tok/call 1.434 → 1.421, **-0.9%,
no real improvement — scattered noise (individual prompts ranged +7.4% to
-6.5%)**. Train-set re-measurement reproduced the first result almost exactly.

**Honest conclusion**: this is the signature of memorization, not
generalization. Training clearly changed the model's behavior (loss decreased
sensibly, train-set numbers reproduced) — the mechanism works technically —
but with only 6 examples over 3 epochs, a 68M-parameter model overfits to the
specific trained continuations rather than acquiring transferable cross-vocab
drafting skill. The first (train-set-only) result would have been genuinely
misleading if reported alone — this is exactly why the held-out check mattered
and should be standard practice for any future training result here.

## CORRECTION (2026-07-21): the RLM-Cascade comparison was apples-to-oranges —
important finding about why, and what the actual fair comparison is

**User correctly caught that comparing RLM-Cascade's 45.8%/1.83x to our SD-only
~1.17x was invalid.** Confirmed by pulling the full paper (arxiv.org/abs/2606.22840):

**RLM-Cascade's real setup**: DeepSeek-V4-Pro (draft) + Claude Opus 4 (verify),
CONFIRMED cross-vendor/cross-architecture — not same-family. Each turn
classified and routed INDEPENDENTLY — no evidence of repeated draft→verify
cycling within one conversation, no state continuity. Direct quote: draft
model "generates a draft using the original system prompt and user messages
UNCHANGED" — full context resent every turn, literally our own B2
("naive full-resend") baseline. Their 45.8%/1.83x comes almost entirely from
NOT calling Opus at all 88.8% of the time ("SKIPPED path dominates").

**Why the comparison is invalid**: their metric is a ROUTING-LAYER number
(savings from avoiding the big model), aggregated across independent turns.
Our ~1.11x→1.17x is a SPECULATIVE-DECODING-LAYER number (per single
engagement, GIVEN the big model is already being called). Different,
COMPOSABLE layers of the same system, not competing/comparable numbers — like
comparing miles saved by not driving vs MPG improvement on a trip you do take.

**The more important finding — why they don't do token-level SD at all**: NOT
a choice, a structural impossibility for their deployment. Direct quote:
token-level SD "inapplicable when draft and verify models are served through
separate HTTP APIs with no access to internal logit distributions." They call
two commercial vendor APIs — zero access to weights/logprobs/internals, ever.

**Why this matters for OUR project specifically**: our entire premise (small
model's weights directly on-device, server side self-hosted or logprob-
exposing) is the OPPOSITE of RLM-Cascade's constraint — we have real model
access, which is exactly the condition that makes token-level SD possible at
all. Their result doesn't show token-level SD isn't worth it; it shows the
ceiling achievable WITHOUT deep model access, which was never our scenario.

**Correct framing going forward**: router savings (avoid calling big model —
RLM-Cascade's whole win) and SD speedup (make it faster when you DO call it —
our whole project) are COMPOSABLE/multiplicative when you have deep model
access, which we do and RLM-Cascade structurally can't. The claim isn't
"our 1.17x beats their 1.83x" — it's "router-layer savings (comparable to any
cascade) PLUS a genuine additional SD-layer bonus only available with real
model access."

**Next steps for an ACTUAL fair comparison**:
1. Measure OUR OWN router's hit-rate on real LoCoMo turns — the thing we've
   never actually computed. RLM-Cascade's headline metric is literally
   "88.8% draft-use rate" — we have a router (logprob threshold) but have
   never measured what fraction of turns it keeps on the small model.
2. Build the real end-to-end aggregate metric: (fraction of turns our router
   avoids the big model on) × (SD speedup on turns that DO reach it, our
   measured ~1.11-1.17x) vs an always-call-big-model baseline — produces a
   number in the SAME SHAPE as RLM-Cascade's 45.8%/1.83x, genuinely comparable.
3. Frame the deliverable honestly: not beating their cascade number, but
   showing router-layer savings (comparable to any cascade) PLUS a genuine
   additional multiplicative bonus only accessible with real model access —
   that's the legitimate, differentiated claim.

## MAJOR FINDING (2026-07-21): router confidence and SD speedup are
ANTI-SYNERGISTIC, not complementary — confirmed empirically

**Setup**: Qwen3-0.6B+LoRA drafter (trained, validated earlier) + Gemma-7b-it
target (8-bit, single-GPU fix after 2 OOM crashes — Gemma-7b-it's 256k-vocab
embedding table doesn't fit fp16 alone on a 14.56GB T4, unlike Qwen2-7B which
barely did). 30 LoCoMo held-out prompts. Confidence = drafter's own avg
per-token logprob on its greedy output. Threshold = median of THIS test set
(flagged as a fairness problem before the run — confirmed below).

**Known flaw, confirmed exactly as predicted**: skip rate landed at EXACTLY
15/30 = 50.0% — a pure artifact of median-splitting the same set being
evaluated, not a real calibrated result. This threshold needs to come from a
SEPARATE calibration set, not the test set itself.

**The much bigger, non-obvious finding**: on the ROUTED (harder/low-confidence)
subset, SD-assisted generation was SLOWER than just calling the target
plainly — avg 0.780x, only 2/15 routed prompts sped up at all (1.04x, 1.11x),
rest ranged 0.29x-0.92x. Direct same-subset comparison: those 15 "hard"
prompts took 201.1s total with NO SD (plain call) vs 283.7s WITH SD applied —
adding speculative decoding to exactly the cases the router flagged as
hardest made the system SLOWER, not faster.

**Why this happens, mechanistically**: confidence (the routing signal) is
inversely correlated with difficulty; SD's speedup depends entirely on
acceptance rate. Low-confidence prompts are low-confidence BECAUSE the
drafter's guesses diverge most from the target — meaning worst acceptance
rate, most wasted draft/rollback overhead, least (or negative) SD benefit.
The router, as wired, sends exactly the wrong cases to the expensive
mechanism — it skims the easy/high-confidence cases (where SD would help
MOST) off the top for free, leaving SD to handle only the leftover hard cases
(where it helps LEAST or actively hurts).

**Overall system number** (misleading given the above): 1.121x vs
always-call-baseline — looks in line with prior results, but this is
dragging down what a corrected design should achieve, not a representative
number for the approach's real potential.

**The actual fix — three-way routing, not a bug patch, a better design**:
1. SKIP (very high confidence) — small model answers alone, big model never
   called.
2. SD-ASSISTED (middle confidence band) — where acceptance rate is plausibly
   decent enough for SD's overhead to actually pay off.
3. PLAIN CALL, NO SD (very low confidence) — don't bother drafting at all;
   overhead isn't recouped when acceptance is expected to be poor.

**Next steps**: (1) identify the actual middle-band boundaries empirically —
needs a genuine held-out calibration set (fixes the self-referential
threshold problem simultaneously), likely by measuring acceptance rate
directly as a function of confidence score across a calibration set, not
just picking thresholds by eye. (2) Re-run the aggregate stat with 3-way
routing and a properly calibrated threshold — that's the real, fair number to
report, not the 1.121x above.

## RESOLVED (2026-07-21): the Gemma-1 vs Gemma-2 finding — corrects an intermediate
wrong conclusion, now well-triangulated with a direct controlled test

**The full diagnostic chain, in order, each ruling something out**:
1. Cross-GPU sync overhead — RULED OUT (same-GPU test: 0.710x, essentially
   identical to cross-GPU's 0.729x).
2. Drafter size/execution time (Qwen3-0.6B ~9x bigger than Llama-68M) — REAL
   contributing factor (Qwen3-0.6B has BETTER tok/call than Llama-68M against
   same target — 2.00 vs 1.47 — yet worse speedup, meaning its own execution
   cost eats the gain) but not sufficient alone, since 0.5B-scale drafters are
   documented working well elsewhere (see below).
3. General cross-family literature guidance found: "same family always
   outperforms cross-family, dramatically" — real, general SD wisdom, and a
   third-party quote found stating "Llama and Qwen both speculate extremely
   well but Gemma...cannot speculate well" — led to an INTERMEDIATE WRONG
   CONCLUSION that Gemma is universally bad at speculation regardless of
   generation.
4. **CORRECTED by direct test**: Qwen3-0.6B (untrained, same drafter that got
   0.852x against Gemma-7b-it/Gemma 1) + Gemma-2-9b (Gemma 2) = **1.196x avg,
   median 1.271x, only 3/15 below 1.0x** — our best cross-vocab result yet.
   Same drafter, same cross-family gap, ONLY the Gemma generation changed, and
   the result flipped from clearly bad to clearly good. This directly
   contradicts the "Gemma is universally bad" conclusion from step 3 — that
   quote was either Gemma-1-specific or about a different method (e.g. TLI/
   intersection rather than SLEM), not a generation-independent truth.

**FINAL, well-triangulated conclusion**: Gemma 1's older architecture (global
attention only, MHA/MQA, no logit soft-capping — confirmed real differences
from Gemma 2's technical writeup) is genuinely disadvantaged for the batched
multi-position verification SD requires. Gemma 2's real architectural
improvements (GQA, alternating sliding-window/global attention) fix this, even
with a drafter (Qwen3-0.6B) that clearly struggled against Gemma 1. This
vindicates the ORIGINAL hypothesis from early in this diagnostic chain, after
an intermediate wrong turn.

**Validated pairing table (size ratios, from literature + our own tests)**:
cross-family successes in the literature (Qwen2-0.5B + Mixtral-8x22B/Llama-3.1-
70B/Phi-3-medium: 1.78-1.91x) all used large size-ratio targets (28:1 to 280:1)
to compensate for cross-family alignment penalty. Our original Gemma-1 test
(Qwen3-0.6B, ~14-15:1 ratio) had neither a large ratio NOR same-family
alignment — the gap nobody else had validated. Switching to Gemma 2 (same
ratio, same drafter) resolved it directly — architecture generation, not
ratio, was the dominant fix for THIS specific case.

**Practical implication for the project**: if deploying against a Gemma-family
server model, use Gemma 2 or newer generations, not Gemma 1 — this is now a
real, evidence-backed, actionable recommendation, not a guess.

## ARCHITECTURAL INSIGHT (2026-07-21): big-to-small handoff needs NO mechanism
at all, if SD is active — a genuine free byproduct, not something to build

**The asymmetry, corrected and sharpened**: small-to-big (escalating) is the
ONLY direction needing a deliberate handoff — full, uncompressed text (per the
earlier correction: compression belongs on the low-stakes direction, never on
escalation, since escalation happens specifically because the query was judged
to need maximum capability — matches RLM-Cascade's validated "unchanged
context" pattern).

**Big-to-small (returning) needs NOTHING — this is the key insight.** If
speculative decoding is active during the big model's turn (which it is,
whenever the big model is engaged in this design), the small model is not a
passive bystander — it's the DRAFTER, continuously proposing tokens every
round. After every SD round, the ACCEPTED (possibly corrected) tokens get
written into the drafter's own KV cache too — not its own guess, the true
verified continuation — because the next drafting round must start from the
real context. This isn't optional, it's structurally required for SD to work
at all. So by the time the big model's turn ends, the small model already has
exact, complete, token-for-token knowledge of everything said — for free, as
an automatic byproduct of a mechanism that exists for a totally different
reason (accelerating the big model), not something we designed or need to
build.

**Why this is a genuine architectural edge, not just a detail**: a pure
text-handoff system with no SD at all (like RLM-Cascade) needs a real catch-up
mechanism in BOTH directions, since neither model is ever continuously
involved in the other's turns. Combining routing + SD gets one of those two
directions entirely for free — a real differentiator from simpler
cascade-only architectures, and not something we set out to design, it falls
out of doing SD correctly.

## SECOND "FREE" BYPRODUCT (2026-07-21): SD's own acceptance rate as
continuous, automatic ROUTER calibration signal — no separate eval needed

Parallel insight, same spirit: every time the router escalates and SD runs,
we get a REAL, MEASURED acceptance rate for that specific turn — not a
predictive signal (which we already showed is unreliable — the confidence-
based router calibration failure earlier), but a POST-HOC, ACTUAL one, safe
from that same failure mode because we're not trying to predict acceptance in
advance, just reading it off after the fact.

**What this buys for free**: if acceptance on an escalated turn turns out
very HIGH, that's direct evidence the small model probably could have handled
this turn alone — the escalation likely wasn't necessary, meaning the
router's threshold is too conservative (escalating more than needed). If
acceptance is consistently LOW, that validates the router correctly
identified a genuinely hard case. This gives a continuous, automatic audit of
router decision quality, and a natural feedback signal for recalibrating the
router's threshold over time — using data the system is already producing as
a byproduct of normal operation, no separate evaluation/labeling pipeline
needed.

**Same underlying principle as the big-to-small insight**: SD, once running,
produces real signal (token-level accept/reject) that's useful for MORE than
its original purpose (accelerating generation) — it simultaneously keeps the
small model synced (previous insight) AND calibrates the router (this one) AND
provides real online training data for the drafter (established earlier in
the session) — three separate benefits, all free, all from the same
mechanism, none requiring additional infrastructure to be built.

## SHARPENED (2026-07-21): concrete router-calibration mechanism, not just the idea

Stop treating "what decides escalation" and "what proves escalation was worth
it" as separate problems — same signal, two different times. A cheap proxy
decides BEFORE the fact; SD's real acceptance rate corrects that proxy AFTER
the fact, continuously.

**Concrete design**:
1. Bootstrap router: start with whatever cheap signal is available before any
   real data exists (even the imperfect drafter-confidence heuristic already
   tried, or simpler — query length, a few keyword/topic features). Doesn't
   need to be good, just needs to get the system running to generate outcomes.
2. Every escalation logs a real, labeled data point: (query features,
   realized acceptance rate) — free, SD produces this on every escalated turn
   regardless.
3. Periodically retrain a LIGHTWEIGHT predictor (start with logistic
   regression/shallow classifier over cheap features — matching the
   don't-over-build discipline used throughout this project) on the
   accumulated (query → realized acceptance rate) pairs. This becomes the
   router going forward, replacing the bootstrap heuristic. Structurally the
   same idea as OmniDraft's own acceptance-prediction head, but trained on
   REAL ground-truth outcomes from actual use, not the drafter's own
   self-confidence (which the calibration experiment already proved doesn't
   correlate well — this is the fix for that exact failure mode: not asking
   the drafter to introspect, asking a separately-trained model to predict a
   real, measured quantity).
4. **The closing loop**: the same SD signal is ALREADY training the drafter
   (online distillation). As the drafter genuinely improves, baseline
   acceptance rate rises — the router predictor, trained on that same rising
   signal, should naturally learn to escalate LESS over time, since escalation
   becomes increasingly less necessary. One signal, three consequences: trains
   the drafter, keeps the small model synced for free, calibrates the router —
   all self-reinforcing, all from data the system produces just by running.

**Escalation path if the simple version proves insufficient**: OmniDraft's
actual acceptance-head architecture, or the retrieval-based (nearest-neighbor
over past query/outcome pairs) middle ground — but start with the shallow
classifier, consistent with every other "start simple, escalate only if
evidence justifies it" decision made throughout this project.

## SETTLED (2026-07-21): the concrete router mechanism — LinUCB contextual
bandit (OrcaRouter pattern), not a hand-wavy "shallow classifier"

Surveyed current router literature specifically for this. **[OrcaRouter](https://arxiv.org/html/2605.30736)**
("Production-Oriented LLM Router with Hybrid Offline-Online Learning") is a
near-exact match, more precise than the classifier idea above — adopt this
mechanism directly, not just as inspiration.

**Direct mapping**:
| OrcaRouter | Our system |
|---|---|
| 10-arm routing (Claude/Gemini/GPT-5-mini/etc) | 2-arm: small-alone vs escalate-with-SD (simpler case) |
| Composite reward: quality+cost+latency+penalties | SD's realized acceptance rate (quality proxy) + measured latency/cost |
| Offline ridge-regression init on 5k labeled prompts | Same pattern on our LoCoMo prompts, real rewards via actual SD runs (pipeline already exists) |
| Online updates via Sherman-Morrison rank-1 updates | Same — cheap, closed-form, real-time, no retrain loop |

**Critically, OrcaRouter explicitly solves our exact blind spot**: "partial-
information feedback — only the selected model's reward is observed" is
literally our problem (SD signal only exists on turns we DID escalate, never
on skipped ones). Not something to invent a fix for — LinUCB's whole design
exists for this. Three concrete, off-the-shelf solutions given: exploration
bonus baked into arm-selection math (√(x_t^T·A^-1·x_t) term), Thompson
sampling, or simple round-robin warmup (force minimum exploration of both arms
before letting confidence dominate).

**Why this beats the earlier "shallow classifier" sketch**: that idea was
directionally right but hand-wavy about the update mechanism and didn't
address partial-feedback at all. LinUCB is decades-established contextual
bandit theory, not a novel invention, and fits almost exactly — binary
decision under uncertainty, partial feedback, needs cheap online updates,
benefits from offline warm start. This is the concrete answer, not a further
direction to explore.

**Complementary signal worth adding, found in the same survey**: chat-based
dissatisfaction detection — "the user's next message indicates whether the
previous response was satisfactory... router can re-route subsequent turns to
a stronger model without needing an explicit reward model." This fills the
OTHER blind spot LinUCB's exploration bonus doesn't fully solve: real signal
specifically on turns we DIDN'T escalate (small model answered alone, user's
follow-up reveals dissatisfaction) — complementary to SD's signal (which only
covers escalated turns), not a replacement for the bandit mechanism.

**Also found, worth knowing but tangential to router specifically**:
Sequential Temperature Scaling (STS, in SGLang's DSpark) — a principled way to
CALIBRATE a raw confidence score to match true acceptance probability, rather
than fully discarding a possibly-informative signal. Could be relevant if we
ever revisit rehabilitating the original drafter-confidence signal rather
than replacing it outright. And "Aurora" — a system continuously training a
speculator directly from live inference traces — closely matches our own
"online distillation" plan for the DRAFTER (not the router), worth checking
if revisiting that thread.

## MAJOR CORRECTION (2026-07-21): calibration data shows the confidence signal
itself is weak/unreliable — bigger finding than "wrong threshold"

**Setup**: 15 genuinely disjoint calibration prompts (seeded shuffle-split of
the 30 held-out set — separate from both the earlier router test and a
held-back final-test-15, fixing the self-referential threshold problem).
Measured confidence, plain baseline, AND full SD time together for every
prompt (not just a skip/route decision) — full raw data in
`calibration_results.json` on the VM, log saved locally at
`/tmp/calibration_full.log`.

**Result**: 14 of 15 prompts showed speedup BELOW 1.0x (only one, at
conf=-0.676, hit 1.080x) — across the ENTIRE confidence range tested, not
concentrated in a "hard half." Splitting this same set at the old threshold
(-0.461): higher-confidence half averaged 0.519x speedup (WORSE), lower-
confidence half averaged 0.860x (less bad) — the OPPOSITE of the assumed
relationship the whole three-way routing plan was built on.

**Why, diagnosed correctly**: tok/call (actual acceptance-rate proxy, measured
AFTER the SD run) correlates with speedup exactly as expected — highest
tok/call (3.33) → best speedup (1.080x); lowest tok/call (1.53) → worst
speedup (0.226x). The underlying SD mechanism behaves sensibly. What's broken
is specifically the PREDICTOR: drafter's own avg logprob (self-confidence,
"how peaked is my own distribution") is conceptually different from cross-
model agreement ("will the TARGET accept what I picked") — a drafter can be
confidently wrong relative to a different, larger, differently-trained model.
This data shows these two things are only weakly, maybe inversely, related
for this pairing/task.

**What this means**: no amount of threshold-tuning on the SAME signal fixes
this — the signal itself doesn't predict what we need it to predict. The
three-way routing ARCHITECTURE isn't wrong; the specific choice of "drafter's
own logprob" as the routing signal is. Correct fix: an actual acceptance-
PREDICTION signal (OmniDraft's/SpecDec++'s approach — trained specifically to
predict target acceptance, not drafter self-confidence), not a better
threshold on self-confidence.

**Reprioritized next step given tight GPU-hour budget**: don't keep chasing a
better threshold on a signal now shown to be weak. Push reverse-KL loss +
more/better training FIRST — this should lift acceptance rate broadly across
the distribution (not just reroute around weak cases). Revisit the router-
signal question once we know how much a properly-trained drafter changes this
picture — possible a better-trained drafter reduces this problem's severity
directly, making the router-signal question easier to solve afterward.

**Also clarified in this session — elegant simplification of the "small model
catches up" direction**: cross-vocab decode/retokenize/align complexity is
ONLY needed for token-level signal comparison (SD's rejection-sampling math,
distillation loss mapping) — confirmed via literature check, every alignment
technique found is for mapping token-level SIGNALS, never for simply handing
one model plain text to read. For "small model absorbs what big model just
said" (no verification happening), it's just ORDINARY context extension — small
model tokenizes the plain text with its own tokenizer, normal forward pass,
no LCS alignment or translation machinery needed at all. This was being
over-complicated earlier by reusing "translation" language from the SD-
verification direction where alignment genuinely is unavoidable.

## HONEST OVERALL ASSESSMENT (2026-07-21): is this a strong idea?

Asked directly, answered without hedging either direction.

**Genuinely strong**: real, still-unaddressed whitespace — nobody published
heterogeneous cross-vendor token-level SD + online adaptation in a genuine
multi-turn switching context. Now know WHY: RLM-Cascade proves production
cascade systems structurally CAN'T do this (zero internals access via
separate commercial APIs) — our core premise (real weight/logit access on
both sides) is exactly the missing condition, not an accident of nobody
trying. Technically validated, not just theorized: real measured speedup
(~1.11-1.17x), genuine held-out generalization (not memorization), and
believable headroom (0.6B/Gemma pairing's +22.6% generalization vs 68M's
+3.93% suggests the ceiling is higher than what's been measured so far).

**More limited than it might sound**: only matters in a specific niche, not
universally. If you control both deployment ends (our whole premise), you
could also just pick same-family models and use Path 2 — literature shows
bigger speedups (2-3x+) with far less complexity. Real reason to choose
heterogeneity anyway: QUALITY — best on-device model and best self-hostable
large model may genuinely come from different labs, and you're not willing
to sacrifice quality for an easier same-family pairing. Legitimate, but
narrower than "generically the right hybrid LLM architecture." Magnitude
right now is modest (~1.17x on the SD layer alone), and composes
multiplicatively with router savings rather than replacing them — meaning
the actual end-to-end contribution may be a meaningful slice, not the
headline number, once folded into a full system.

**Overall verdict**: solid, legitimate, defensible research contribution —
real gap, real proof it works, believable path to a bigger number, clear and
correct scope of when it actually matters. NOT a "this changes how everyone
builds hybrid LLM" claim — that would overclaim given how much value crude
cascading alone already captures per RLM-Cascade. Correct pitch: a genuine
additional lever on top of routing, specifically for the case where deep
model access exists and heterogeneity is a deliberate quality-driven choice.

## HONEST RECONSIDERATION (2026-07-21): continuous tracking vs text-handoff —
which is actually more elegant FOR HYBRID LLM SPECIFICALLY

Asked directly to be honest about this, not just validate the direction just
built toward. Real answer: **continuous tracking is NOT more elegant for
hybrid LLM's actual purpose — it's elegant along one axis (no cold starts,
no catch-up latency) but works against hybrid LLM's core premise along a more
important axis.**

**The core premise of hybrid LLM**: avoid paying for the big model except
when actually needed. Continuous tracking quietly abandons this — the big
model does a real forward pass EVERY turn regardless of router selection.
Cheaper than full regeneration, but not zero, every turn, forever — a
different cost profile than "zero except when needed."

**Bigger problem for the actual deployment target**: continuous tracking
requires constant device-server connectivity — translated tokens must reach
the server every turn even during long stretches where the small model alone
would've sufficed. This is a real assumption for an ANDROID PHONE ON CELLULAR
— intermittent connectivity, battery cost, data cost are the actual operating
conditions, not edge cases. Text-handoff has no such requirement: server can
be genuinely idle, connectivity only needed at actual switch moments.

**Verdict**: text-based handoff/checkpointing is more elegant FOR HYBRID LLM
SPECIFICALLY, because it stays true to the actual premise (big model costs
nothing until deliberately engaged). Continuous tracking answers a narrower,
different question well — "minimize latency when switches are frequent AND
connectivity is cheap/reliable" — which describes a same-datacenter,
same-family deployment far better than phone-to-server. The Android target
leans toward text-handoff's assumptions (rare switches, intermittent
connectivity), not continuous tracking's.

**Synthesis worth pursuing instead of picking one permanently**: make the
choice of mechanism ADAPTIVE — default to text-handoff as baseline, escalate
to continuous tracking only if the router's own observed switching pattern
shows frequent-enough oscillation that repeated catch-up costs would exceed
continuous tracking's steady overhead. Consistent with everything else in
this project being router-driven/adaptive rather than architecturally fixed.

## NEXT PHASE PLAN (2026-07-21, after LoRA/0.6B run completes)

1. Implement OmniDraft's ACTUAL reverse-KL loss (target's full soft
   distribution at direct-mapping positions), replacing our simplified NLL —
   this is the remaining known gap vs their real method, per earlier analysis
   of why it should help (richer signal per example → likely better data
   efficiency, directly relevant given our own overfitting-with-6-examples
   finding).
2. Run a longer/larger "pretrain" pass with the corrected loss before
   considering production readiness — more data, more epochs, same held-out
   generalization check methodology throughout (non-negotiable given what the
   6-prompt memorization scare taught us).
3. Build the actual PRODUCTION EMULATION HARNESS — not isolated before/after
   speedup scripts anymore, the full continuous-tracking loop end to end on a
   real multi-turn LoCoMo conversation with genuine router switching.

**Continuous-tracking design, confirmed understanding**: router selects small
model → small model generates, output translated (decode/re-encode via big
model's tokenizer) and fed to big model as BLIND PREFILL ONLY (no draft/
verify/accept-reject) — keeps big model's KV cache continuously warm/synced.
Router selects big model → cache already warm (never went cold), go straight
into REAL speculative decoding (small model drafts, big model verifies with
actual accept/reject) — no catch-up cost paid at the switch.

**Improvements to add on top of the design as stated**:
- **Drift monitoring during blind-prefill mode**: blind prefill is never
  verified — small translation imperfections could silently accumulate in the
  big model's cache over many consecutive small-model turns before ever being
  checked. Add periodic cheap spot-verification (every N turns, even while
  nominally in small-model mode) to catch/log drift early — same underlying
  idea as the earlier "trust-calibrated adaptive verification" proposal,
  reframed here as protecting cache integrity specifically.
- **Pipeline the blind-prefill step, don't let it block**: the big model's
  forward pass over translated tokens shouldn't run synchronously on every
  small-model turn (adds latency, defeats the point of cheap small-model
  turns) — overlap it with the small model's NEXT generation instead
  (DuoDecoding-style pipelining pattern from earlier).
- **Be honest this isn't free compute**: we proved TRANSLATION is cheap
  (0.07s) — but the big model still does a real forward pass over those
  tokens every single small-model turn. Real, recurring server-side cost, not
  something to assume away. The production emulation should explicitly
  measure this, not just the switch-point savings.

## NEXT: properly-powered training run (scoped, not yet built)

**Data**: stop using 6 hand-picked prompts. Sample many (50-100+) distinct
(context, continuation) pairs from real LoCoMo conversations — different
conversations, different points within them, for genuine diversity. This
reuses infrastructure already planned for the eval harness rather than a
separate data pipeline. **Split into non-overlapping train/held-out sets from
the start** (not reusing prompts across both, unlike the quick test above).

**Training regimen**: 1 epoch (or very few) over the larger, more diverse set
— more data naturally reduces memorization risk without needing many repeated
passes over the same small set, which is what likely drove the overfitting
above.

**Evaluation**: larger held-out set (20-30+ prompts, not 6) — the quick test's
individual-prompt swing (+7.4% to -6.5%) shows 6 samples isn't enough for a
trustworthy average; need enough N that a few noisy prompts don't dominate.

**Loss formulation**: keep the simplified NLL loss for this next test —
cheaper and faster to check whether more data alone fixes generalization
before investing in reproducing OmniDraft's exact dual reverse-KL + n-gram-NLL
loss. Only upgrade the loss if data-scaling alone proves insufficient.

**Honest expectation**: even with this improved setup, given a tiny 68M model
and modest compute, the realistic outcome might be a modest generalization
improvement, not a dramatic one — this is a genuine open question to test, not
a predicted win.

## NEXT STEP (2026-07-20): validate the riskiest assumption first, cheaply,
before building anything else

**The single riskiest, unvalidated piece of the whole v3 design**: using
OmniDraft's n-gram cache for TRANSLATION ONLY (no verification) as passive
sync — this is our own extension, not something OmniDraft's paper tests.
Validate this in isolation before building the orchestration/router/device
layers around it.

**Code availability, checked directly (2026-07-20):**
- **OmniDraft itself: NO code released.** Checked arXiv page, HuggingFace
  papers page, and lead author's (Zhaocong/Justin Yuan, Qualcomm AI Research)
  personal site directly — no GitHub link anywhere, despite their site linking
  code for other papers. Confirmed via direct page fetch, not just search.
- **UAG/TLI (OmniDraft's precursor, simpler mechanism): YES, real production
  code.** Built into Hugging Face Transformers 4.46.0+ (`use_heterogeneous_vocab`
  in vLLM too). Dedicated technical writeup:
  [huggingface.co/blog/jmamou/uag-tli](https://huggingface.co/blog/jmamou/uag-tli).
  This is the actual runnable starting scaffold.
- **Adjacent applied reference**: [Cross-Family Speculative Decoding for Polish
  LMs on Apple Silicon (arXiv 2604.16368)](https://arxiv.org/pdf/2604.16368) —
  real code at `github.com/krzysiekfonal/mlx-lm/tree/feature/slem-with-context-aware`,
  extends SLEM with "context-aware" strategies specifically for on-device
  cross-family SD. MLX (Apple Silicon) only — won't run on Kaggle (CUDA) or
  Android directly, but useful to read and port the LOGIC from, not the code.
- **FlexSpec, Ghidorah**: confirmed no code either, reimplement-from-paper only
  if/when needed for the device-side hetero-core pieces.

**Concrete plan**: build on HF Transformers' UAG as the runnable base
(standard PyTorch/CUDA, works on Kaggle out of the box) — port relevant logic
from the MLX-LM SLEM fork for context-aware extensions — build OmniDraft's
specific improvements (n-gram cache with false-rejection fix, hybrid
distillation, online adaptation) ourselves on top, since nobody has code for
those. Run the isolated validation test (blind-translation-prefill vs full-
verify vs fresh-full-prefill, compare downstream generation quality) on Kaggle
— Llama-68M draft + Vicuna-7B/Llama3-8B target fits comfortably on Kaggle's
dual-T4 (or single T4 with int8 on the target), matches OmniDraft's own tested
pairs so results are comparable to their published numbers.

---

Status as of 2026-07-19: paused before building. Doing a deep-dive pass through
the four components and their source papers before starting the 2-week POC.

## The core question

Route turns of a single conversation across genuinely different-architecture
models (small edge model, large cloud model, different tokenizers/architectures),
and figure out what happens to conversation state/KV cache when you switch
mid-conversation.

## Two established research threads + one bridge

- **Routing/cascades** (crowded, mature): decide which model answers a given
  query. Operates entirely in text space — router never touches either model's
  internals. Easy precisely because of that decoupling.
- **KV cache management** (mature within a model, unsolved across architectures):
  L0 single-turn (solved) → L1 multi-turn same model (solved — CachedAttention)
  → L2 same-architecture-family, different checkpoints (solved — DroidSpeak)
  → L3 different architectures, shared hardware, resource-level sharing only
  (SwiftCache) → L4 different architectures, across turns of one conversation
  (**not solved by anyone** — confirmed by the dedicated 2025 survey not even
  raising it as an open challenge).
- **Speculative decoding**: the one place small/large models are forced into
  真正 synchronized state, token by token. Where the field has actually chipped
  at architecture heterogeneity (vocab mismatch, hardware placement, soft
  deferral) — most relevant bridge to steal mechanisms from.

## Correction: A (routing) vs D (speculative decoding) are NOT the same idea at
different granularity — three distinct paradigms

| Paradigm | What executes | Optimizes | Quality guarantee |
|---|---|---|---|
| Predictive routing (RouteLLM, Hybrid LLM) | Router reads prompt only, exactly one model runs | Cost | None — accepted tradeoff |
| Cascade (FrugalGPT) | Cheap model always runs; expensive model runs too if cheap fails scorer | Cost (w/ latency penalty on escalation) | None |
| Speculative decoding (D) | Both models run, every segment, every time | Latency (wall-clock) | **Exact** — provably identical output distribution to always-large |

Speculative decoding requires shared vocabulary because acceptance is
`min(1, P_target(token)/P_draft(token))` — only well-defined if both models
agree on what a token is. That constraint is exactly what the heterogeneous-
vocab paper (component B) removes via text round-tripping.

Speculative cascades (Google) is the genuine fusion point: SD's mechanism
(parallel batched verification) running a cascade's decision logic (soft,
confidence-based deferral instead of strict likelihood-ratio acceptance).

## The four components of "the system"

| # | Component | What it does | Difficulty | Status |
|---|---|---|---|---|
| A | Turn-level deferral policy (router) | Decide per turn: small model or large model | Low — solved at query granularity, just retarget to turn granularity | Not started |
| B | Cross-architecture handoff medium | Get state across a tokenizer/architecture boundary without tensor incompatibility | Medium — mechanism proven at token-level (heterogeneous-vocab spec decoding), needs lifting to turn-level | Not started |
| C | Cheap reuse of the handoff (avoid O(full history) re-ingestion every switch) | **Correction:** not literal KV-tensor reuse (impossible cross-architecture, out of scope entirely) — this is *context compression*: shrink what has to be re-prefilled on a switch from "full raw transcript" to a small persistent state object, so switch cost stays ~constant instead of growing with conversation length | High — **this is the actual open research gap**, no direct precedent for cross-architecture case | Not started |
| D | Turn-level partial correction (optional, tighter coupling than pure routing) | Let the large model partially correct/reject a small model's output mid-turn, not just accept/reject wholesale | Medium — conceptual precedent exists (BiLD fallback/rollback, speculative cascades' soft deferral) but "unit of rollback" isn't well-defined at turn granularity | Not started |

## Paper map (which paper feeds which component)

**A — Routing/deferral**
- [RouteLLM](https://arxiv.org/abs/2406.18665) (ICLR'25) — preference-trained router (Bradley-Terry / BERT classifier / matrix factorization), trained on Chatbot Arena human pairwise votes. Model pair: GPT-4-Turbo vs Mixtral-8x7B (genuinely different architecture + tokenizer). Tunable threshold trades cost for quality.
- [Hybrid LLM](https://arxiv.org/abs/2404.14618) (MSR, ICLR'24) — router predicts the *quality gap* between small/large outputs, not just difficulty. Up to 40% fewer large-model calls, no quality drop.
- [FrugalGPT](https://arxiv.org/abs/2305.05176) — sequential N-way cascade with a learned "generation scorer" gating escalation. Up to 98% cost reduction matching GPT-4 quality.
- [Speculative cascades](https://research.google/blog/speculative-cascades-a-hybrid-approach-for-smarter-faster-llm-inference/) / [Faster Cascades via Speculative Decoding](https://arxiv.org/abs/2405.19261) (Google, ICLR'25) — soft, confidence-based deferral rule instead of hard cascade threshold; bridges into component D territory.

**B (sub-thread) — hidden-state projection instead of text round-trip: explored, not solved for cross-architecture SD**
- Theory backing it's plausible: [Relative Representations](https://arxiv.org/abs/2209.15430) (Moschella et al., ICLR'23) — independently-trained nets preserve relative (not absolute) latent geometry under similar data/modeling choices; [Platonic Representation Hypothesis](https://proceedings.mlr.press/v235/huh24a.html) (Huh et al., ICML'24) — bigger/better models converge toward shared representational geometry, but follow-up work shows alignment is local (neighbor-preserving) not global, degrades with scale.
- [EAGLE](https://arxiv.org/abs/2401.15077) (Li et al. 2024, ICML) — proves hidden-state-level drafting + frozen LM head reuse works extremely well, but **same-model only** (draft head trained on target's own features, uses target's own head) — sidesteps the cross-architecture translation problem entirely.
- [Patchscopes](https://arxiv.org/abs/2401.06102) (Ghandeharioun et al. 2024) — closest real precedent: patches one model's hidden state into a *different* model's forward pass to decode it. Explicitly reports cross-family patching works but is weaker than same-family — live evidence for the relative-representations degradation. Built for interpretability, not for driving a lossless SD verification loop (no rejection-sampling guarantee, no throughput benchmarking).
- Tuned Lens / Logit Lens / model-stitching literature — trains linear/affine maps from hidden states to vocab space, including some cross-model work, but aimed at interpretability/feature-transfer, not generation.
- **Conclusion**: nobody has published a trained hidden-state→target-vocabulary projection driving lossless cross-architecture speculative decoding. What exists for the cross-architecture SD case stays in discrete/text space (SLEM/SLRS, OmniDraft's n-gram cache) rather than continuous hidden-state space. This is genuinely open, but it's an ML-research project (representation alignment, likely needs paired training data, no easy losslessness proof), not a systems-integration project — different flavor of work than the 2-week POC.

**B — Cross-architecture handoff medium (text round-trip, the established approach)**
- [Lossless Speculative Decoding for Heterogeneous Vocabularies](https://arxiv.org/abs/2502.05202) (ICML'25 Oral) — three algorithms, all proven lossless:
  - **SLEM** (String-Level Exact Match): decode draft tokens to text, re-tokenize into target vocab, run target forward pass, accept up to first divergence + 1 sampled token. Text is the literal bridge.
  - **SLRS** (String-Level Rejection Sampling): generalizes classic rejection sampling to string level — ψ(t) = total draft probability mass over all draft sequences decoding to target token t, accept/reject via p(t) vs ψ(t) (Theorem 3.2). Recovers draft candidates SLEM would throw away on non-exact-match.
  - **TLI** (Token-Level Intersection): restrict draft to only propose tokens in T∩D (vocab intersection), renormalize, run ordinary token-level SD. No text round-trip, but crippled if the two tokenizers barely overlap.
  - Non-injective tokenizer wrinkle: decode(encode(s)) ≠ s in general (normalization/whitespace/accents) — SLEM handles this via "longest matched stretch" realignment.
  - For genuinely heterogeneous edge/cloud pairs (small vocab overlap), SLEM/SLRS are the relevant ones, not TLI — validates "text, not tensors" as the general bridging principle.
  - Speedup numbers (~2.8x, summarization/coding/long-context) reported in an earlier shallow pass, not independently reconfirmed against the paper's actual results table — verify before citing.
  - Shipped in production: [Hugging Face Universal Assisted Generation](https://huggingface.co/blog/universal_assisted_generation) uses essentially the SLEM mechanism (re-encode into target vocab, align via longest-common-subsequence, discard mismatched KV entries), 1.5-2x speedup, in Transformers.
- [OmniDraft](https://arxiv.org/abs/2507.02659) (NeurIPS'25) — real cross-architecture SD system: one Llama-68M draft paired with Vicuna-7B/Qwen2-7B/Llama3-8B targets. Bridges vocab mismatch via an online-adapted n-gram cache + hybrid distillation fine-tuning (discrete token-level translation, not hidden-state projection). 1.5-2x speedup.

**C — Cheap reuse / caching (the open gap)**
- [CachedAttention / AttentionStore](https://arxiv.org/abs/2403.19708) (USENIX ATC'24) — hierarchical KV offload (GPU→CPU→disk), session-keyed persistence, 87% TTFT reduction. Same-model only — steal the *storage architecture pattern*, not the tensors.
- [DroidSpeak](https://arxiv.org/abs/2411.02820) — **correction: requires same *foundation model* (shared weight lineage — fine-tunes of one base), not just same architecture spec.** Two independently-trained models with identical architecture do NOT qualify; paper doesn't evaluate that case. Mechanism: offline O(l²) profiling per model pair finds "critical" layers (~11% avg) that must be recomputed for accuracy; the rest are reused. Critical layers forced into a contiguous block (avoids multiple expensive embedding-cache transitions, which can be 2-4x the size of the KV cache). Recompute (compute-bound) pipelined against reused-layer loading (I/O-bound) on separate CUDA streams for ~2x extra TTFT gain. 4x throughput, 3.1x prefill speedup total. Even narrower scope than previously noted.
- [SwiftCache](https://arxiv.org/abs/2606.16135) — solves hardware/memory sharing, not content reuse, despite the name. Co-located *different* models on one server; low-KV-demand models donate idle GPU memory to high-demand models' overflow, moved over NVLink. Each model's cache content stays entirely its own — nothing computed by one model is ever read by another. Orthogonal to the handoff problem; useful for deployment efficiency only. P99 TTFT -69%, max context 3.98x.
- [KVzip](https://arxiv.org/abs/2505.23416) (NeurIPS'25) — query-agnostic compression of one model's own long-context cache. Decides what to evict by prompting the model itself with "repeat the previous context" on a candidate-compressed cache and checking what's needed for faithful reconstruction — self-supervised importance signal. 3-4x cache size reduction, ~2x decode latency, negligible quality loss (Llama-3.1/Qwen2.5/Gemma3). Same-model only. **Useful trick to steal**: the reconstruction-probe method is a ready-made diagnostic for capsule fidelity — after a switch, ask the new model what it knows from before the switch and check it against ground truth.

**D — Turn-level partial correction / tight coupling**
- [BiLD](https://arxiv.org/abs/2302.07863) (NeurIPS'23) — small model runs by default, large model steps in via fallback/rollback policy. Same-vocab constrained, single generation pass.
- [DuoDecoding](https://arxiv.org/abs/2503.00784) / [Dovetail](https://arxiv.org/abs/2412.18934) — hardware-heterogeneous placement (draft on CPU/edge, target on GPU/cloud), running in parallel.

## What current systems actually do on a mid-conversation model switch (no trick exists)

Most routing papers (RouteLLM, Hybrid LLM, FrugalGPT) don't test true mid-conversation
switching — they route a whole session/query once. When production systems do
support switching (OpenRouter cross-model chat, agent handoffs), the mechanism
is brute force: the **entire prior conversation, as raw text, gets pasted into
the new model's prompt**, and it does a normal fresh prefill. Any competent
model can read a transcript it didn't generate and continue coherently — that
was never the hard part. The only problem is efficiency: full re-prefill cost,
paid again on every switch, growing with conversation length. Component C is
about shrinking that cost, not about making the receiving model "understand"
context it otherwise couldn't — it always gets context, the question is how
much of it has to be raw text.

## Component C — FINAL DESIGN (2026-07-20, supersedes capsule below)

Dropped the incrementally-updated capsule entirely — unnecessary once redesigned
as on-demand extraction. Mechanism:

1. Plain running transcript logged every turn, no processing.
2. On a router-triggered switch: ONE extraction call to the **local small model**
   (always local — free/fast, no reason to ever pay the API model for this),
   given the full raw transcript, prompted for (1) key facts/decisions, (2) open
   threads, (3) last exchange verbatim.
3. That output — not the raw transcript — becomes the handoff payload prepended
   to the new message sent to whichever model takes over.
4. Nothing happens between switches — no incremental state, no schema to
   maintain consistently across turns.

**Why this obsoletes the earlier machinery**: importance scoring, SnapKV-style
windowed voting, and pre-commit reconstruction checks existed to solve "what do
I keep vs summarize incrementally" — that problem doesn't exist once every
extraction starts fresh from the complete history instead of building on a
possibly-already-lossy prior summary. KVzip-style reconstruction probe survives,
but purely as an eval diagnostic now (ask receiving model "what do you know"
post-switch, score against planted facts), not part of the extraction pipeline.

**Explicitly rejected alternative**: literally compressing the small model's KV
cache (H2O/SnapKV/MiniCache-style eviction) before querying it. Real tensor
compression solves a memory/compute-constraint problem this POC doesn't have —
local model, short (~10-20 turn) conversations, no memory pressure forcing
eviction. Would add real engineering (hooking into attention/cache internals)
for no benefit at this scale. Compression and cross-architecture compatibility
are orthogonal axes regardless (compressing a cache changes its size, not which
architecture can read it) — this only ever mattered for the extraction step's
efficiency, and at this scale that step doesn't need it.

Eval conditions unchanged: B0 always-large / B1 always-small / B2 full-transcript
resend / B3 = this on-demand extraction. Cost metric = tokens actually sent
across the switch boundary to the cloud model; the extraction call itself is
local/free and doesn't count.

## Component C — LatentMAS-based redesign (2026-07-20, current)

**[LatentMAS](https://arxiv.org/abs/2511.20639)** ("Latent Collaboration in Multi-Agent Systems", ICML 2026 Spotlight,
code at [github.com/Gen-Verse/LatentMAS](https://github.com/Gen-Verse/LatentMAS)) —
training-free. Agents generate "latent thoughts" autoregressively in last-layer
hidden embedding space instead of text; a shared latent working memory (layer-wise
KV caches) carries representations to the next agent with no text re-encoding —
genuinely lossless transfer, unlike text-mediated handoff. Real numbers: up to
14.6% higher accuracy, 70.8-83.7% fewer output tokens, 4-4.3x faster than
text-based multi-agent systems, across 9 benchmarks.

**Stated requirement**: agents must share the same transformer shape — same
hidden dimension, attention setup, RoPE scale, **layer count**. Authors
explicitly flag transfer across different families, **sizes**, vocabularies, or
positional encodings as unsolved in the base framework.

**CRITICAL CORRECTION (2026-07-20): "same family" does NOT imply compatible
dimensions.** Different sizes within one family (e.g. Llama-3-8B vs
Llama-3-70B) almost always have different hidden dimensions AND different
layer counts — scaling changes both width and depth together. "Size" is
explicitly listed as its own separate axis of incompatibility in the paper's
own limitations discussion, not subsumed by "family." Same issue applies to
DroidSpeak for the identical reason — its "same foundation model" scope means
same-size checkpoints/fine-tunes, not different parameter counts; KV tensor
shapes are tied to (num_layers, hidden_dim, num_heads), which differ across
sizes regardless of shared lineage.

**Practical implication**: the realistic hybrid setup (small edge + large
server, plausibly different sizes even within one family) does NOT get a free
pass from either DroidSpeak or vanilla LatentMAS. A trained adapter is needed
the moment sizes differ, same family or not — same-family narrows the gap
(shared tokenizer/training distribution should make the adapter easier to
train per relative-representations theory) but doesn't eliminate the need for one.

**CORRECTION (2026-07-20, later same day): the "single-point bridge" claim above
was wrong — verified and retracted.** LatentMAS's shared working memory is NOT
a last-layer-only vector. Checked Section 3.2 directly: it extracts KV caches
from ALL L transformer layers and does direct layer-wise concatenation,
prepending each layer's K/V into the corresponding layer of the receiver's
cache — exact layer-by-layer correspondence required, same structural
requirement as DroidSpeak and Cache-to-Cache. Also: all of LatentMAS's actual
experiments use IDENTICAL architectures (same Qwen3 variants as different
"agents") — narrower validated scope than implied; never tested even
same-lineage-different-checkpoint, let alone different sizes.

**Revised Component C plan (corrected)**: same-family-different-size gets NO
shortcut from any of DroidSpeak/LatentMAS/C2C as released — all three require
literal layer-by-layer shape correspondence. The actual fix, structurally
similar to what Cache-to-Cache already does (not simpler than it, as
previously claimed): (1) layer alignment — terminal alignment (match final
layers first, working backward) applied to LatentMAS's layer-wise
concatenation instead of assuming identical shapes; (2) a trained per-layer
(or per-aligned-pair) projection bridging hidden dimensions at each aligned
layer. What same-family plausibly still buys you per relative-representations
theory: easier to train, less data needed, than a genuinely cross-vendor pair
— but the shape of the fix is C2C's, not a simplification of it.

**Also reshapes Path 2**: LatentMAS (training-free, strong same-family/same-size
numbers) may be a stronger base for the repeated-switching reliability question
than DroidSpeak — worth reconsidering which one Path 2 actually tests.

## Component C — ORDER OF OPERATIONS CORRECTED (2026-07-20, later same day)

**Don't default to training KV-Distill compressors — start with training-free
methods and only escalate if they actually fail.** Caught jumping straight to
"train two compressors" without first checking whether SnapKV or PyramidKV
(free, no training, immediately deployable) are just good enough. Two reasons
the case for KV-Distill was weaker than presented:

1. KV-Distill's paper only compares against H2O, not against the stronger
   training-free methods (SnapKV predicts importance from a recent attention
   window instead of needing the future query; PyramidKV allocates budget
   per-layer) — unknown how the gap looks against the good baselines.
2. KV-Distill's headline results (86.6% vs 56.6%) are at EXTREME compression
   (90-99%) on long documents (up to 71k tokens). Our conversations are
   LoCoMo-scale (~16k tokens) — probably don't need extreme compression,
   exactly where training-free methods are strongest relative to trained ones.

**Corrected plan**: use SnapKV or PyramidKV (not basic H2O) as the DEFAULT
catch-up mechanism for both models, zero training cost. Test in the existing
switch-count-sweep harness. Only escalate to training a KV-Distill-style
compressor if that eval shows a real, meaningful quality gap at the switch
depths that matter. Training-free is the baseline to beat, not an
afterthought — KV-Distill is the escalation path, not the default.

## Component C — KV-Distill design (2026-07-20, now the ESCALATION PATH only,
not the default — see correction above): symmetric per-model compression

**Why this replaces cross-model fusion (C2C/LatentMAS/DroidSpeak)**: all three
require solving cross-architecture layer alignment (matching hidden dims and
layer counts between two DIFFERENT models) — the hard, unresolved problem this
whole thread got stuck on. Explored a different literature instead — single-model
trained context compression (gist tokens, AutoCompressor, ICAE,
[KV-Distill](https://arxiv.org/abs/2503.10337), [Cartridges](https://arxiv.org/abs/2506.06266)) —
which sidesteps the alignment problem entirely because there's no cross-model
transfer: you train a compressor **for** whichever model needs to catch up,
using only that model's own shape.

**KV-Distill** (Chari, Qin, Van Durme — JHU/Microsoft): parameter-efficient
adaptor, compresses arbitrary context spans into a much shorter KV cache,
trained via student-teacher KL-divergence (compressed cache vs full cache,
matched on downstream output). "Nearly lossless" per their own claims, up to
99% length reduction preserving downstream task performance. Code released:
[github.com/vnchari/kv-distill](https://github.com/vnchari/kv-distill).

**Cartridges** (Stanford, Eyuboglu et al.) — related but NOT directly
applicable: trains a compressed cache offline per-corpus via "self-study"
(synthetic conversations + context-distillation), meant for static reusable
content queried many times. Our conversation content is live/unique each time,
so we can't pre-train a cartridge for it in advance — KV-Distill's
general-purpose "adaptor trained once, applied to any new context at
inference" framing fits better.

**The design**:
1. Train TWO symmetric compressors — one adaptor for the small (edge) model,
   one for the large (server) model — each only ever matching its own
   architecture's shape. No cross-architecture alignment anywhere.
2. Component A (routing) unchanged — router still decides per turn, small
   model's role in generating turns unchanged.
3. What changes: on a switch, whichever model takes over runs the accumulated
   raw transcript through ITS OWN trained compressor and loads the resulting
   compact cache directly (no re-prefill needed — the compressor's output is
   already a usable cache, unlike a text summary which still needs prefilling).
4. Training recipe: teacher = target model's real output given full context;
   student = same model's output given the compressor's short cache instead;
   loss = KL-divergence between the two, adaptor-only parameters trained
   (base model frozen). Data: reuse LoCoMo conversations again — (context-so-far,
   real answer) pairs at many points, same source now doing triple duty
   (eval + training data for both compressors).
5. Implementation: build on KV-Distill's released code rather than
   reimplementing the adaptor architecture from the paper description alone.

**Eval**: unchanged LoCoMo-based harness (switch-count sweep 2/4/8/16, F1/ROUGE
vs real ground truth sliced by switch depth, failure taxonomy) — B4 redefined
as "compressed cache from trained compressor" instead of prompted text summary.

**Honest scope**: more tractable than the C2C-fusion plan since there's no
cross-architecture layer-alignment problem to solve — closer to 3-4 weeks than
4-6+, especially building on KV-Distill's existing code rather than from scratch.

## Superseded: earlier incremental "state capsule" design (kept for history)

```
{
  "established_facts":     [ short factual/preference statements extracted so far ],
  "open_threads":          [ unresolved questions / promised-but-undelivered items ],
  "last_2_turns_verbatim": [ most recent (user, assistant) pair, word-for-word ],
  "running_summary":       "~150-token prose summary of everything older than that"
}
```

Updated after every turn (cheap call, can run on the local small model). On a
switch, send `running_summary + established_facts + last_2_turns_verbatim +
new message` instead of the full transcript — a few hundred tokens instead of
the whole history, and roughly constant instead of growing with conversation
length. Measure: token count vs. full-transcript baseline (should be much
smaller/flatter), and quality retained via LLM-judge, specifically stress-testing
facts from several turns back that only survive in the compressed summary, not
verbatim — that's the actual failure mode to catch.

Honesty check: **no paper proposes this exact structure — it's my synthesis,
not a cited result.** Provenance of the pieces:
- The one genuinely transferable insight from an actual paper: text-not-tensors
  bridging, ported from component B's token-granularity down... up to turn-granularity.
- Closest real academic precedent for the *shape* (curated structured memory
  vs raw transcript): [MemGPT](https://arxiv.org/abs/2310.08560) (Packer et al.
  2023) — OS-inspired tiered memory (core context vs archival/recall, paged and
  summarized, model self-manages eviction). Solves a different problem though —
  extending one model's own context, not cross-architecture handoff.
- The specific four-field schema and its use as a cross-model handoff payload:
  not validated by anyone. This is the actual hypothesis the POC is testing,
  not an established result to lean on.

## KV cache compression/eviction landscape (added 2026-07-20)

**Eviction (discard tokens permanently):** H2O (cumulative attention score
"heavy hitters"), StreamingLLM (attention-sink tokens + sliding window, no
importance scoring), SnapKV (observation-window attention pooling per head,
clustered selection), PyramidKV (per-layer budget allocation, pyramid-shaped
since crucial-KV count shrinks with depth).

**Low-rank/variance-based:** [Palu](https://aclanthology.org/2024.findings-emnlp.899.pdf),
Eigen-Attention, LoRC, SVDq, ReCalKV — SVD on K/V matrices, keep high-variance
latent directions, discard low-variance ones (literal PCA on the cache). Note:
RoPE increases key vector variance/rank, so low-rank compression works better
on values than keys unless applied pre-RoPE.

**Removing similar tensors:** [MiniCache](https://arxiv.org/abs/2405.14366) —
adjacent layers (mid-to-deep) have highly similar KV states; decompose into
magnitude+direction, merge/share direction across layers, keep only per-layer
magnitude + genuinely distinct outliers. 5x compression, ~5x throughput on
LLaMA-2. xKV — same idea via cross-layer aligned SVD.

**Unifying framing:** RDKV — eviction and quantization as one rate-distortion
problem, eviction = zero-rate boundary of the same curve.

## Speculative decoding × KV cache compression intersection (added 2026-07-20)

**Same-model only (well populated):**
- QuantSpec — self-speculative: draft pass uses 4-bit quantized cache+weights
  of the SAME target model, verify at full precision. >90% acceptance, 2.5x speedup.
- SpecKV — joint optimization of compression level and speculation length (γ).
- VeriCache — draft with a *lossy* compressed cache, verify against the full
  cache; SD's accept/reject recovers losslessness from a lossy starting point.
- KVShot / "When Hidden States Drift" — raw KV cache reuse beats hidden-state-only
  reuse for long-range draft accuracy (same-model) — evidence hidden states are
  a lossier carrier than KV tensors, relevant to the component-B hidden-state thread.

**Cross-architecture (the real intersection, very recent — mostly 2025-2026):**
- Q-KVComm — multi-agent (parallel) KV cache communication, "heterogeneous
  model calibration" = a TRAINED distributional-alignment step (not zero-shot),
  5-6x compression, tested only on small 1.1-1.5B models.
- **[Cache-to-Cache (C2C)](https://arxiv.org/abs/2510.03215)** — **correction to
  earlier claim that no paper does trained cross-architecture hidden-state/cache
  projection.** Trains a neural network to project+fuse one model's KV-cache
  directly into a different model's, with a learnable gating mechanism per
  layer. Tested across genuinely different families/tokenizers: Qwen2.5, Qwen3,
  Llama3.2, Gemma3. Beats text-based communication by 3.1-5.4% accuracy, 2.5x
  faster on average (up to 3.46x). **Real limitations that keep the POC's gap
  intact**: (1) fusion network trained PER Sharer-Receiver pair, ~500k samples,
  no evidence of a general/zero-shot projector; (2) architected for simultaneous
  single-query ensemble consultation during prefill, NOT sequential multi-turn
  conversation handoff — authors flag conversational extension as preliminary/
  future work in an appendix, not solved.
- QKVShare — turns out to be same-architecture handoff (variants of one model
  family) for multi-agent edge systems, not cross-architecture. Useful real
  numbers: handoff vs re-prefill at 8K context, 397ms vs 1029ms.

**Net read**: cross-architecture cache-level communication is now proven
possible (C2C), correcting the earlier "unexplored" framing — but specifically
for single-query, expensively-per-pair-trained ensemble consultation. The
sequential, multi-turn, training-light corner (this POC) remains open even by
the closest paper's own admission.

## Confirmed gaps (why this is real whitespace, not just an empty search)

1. No system does turn-level routing across genuinely different architectures
   while preserving computed state — always pays full re-prefill or drops to a
   lossy text-summary handoff.
2. "Heterogeneous KV cache" in the literature means resource heterogeneity
   (SwiftCache), never representational heterogeneity.
3. The one mechanism that does bridge real vocab/architecture differences
   (heterogeneous-vocab speculative decoding) operates at token granularity
   within a single generation pass — never lifted to conversation-turn
   granularity or combined with a persistent multi-turn router.
4. Apple's Private Cloud Compute made every server request stateless *by
   design* rather than attempt cross-model handoff — evidence this isn't just
   unpublished, it's avoided even at production scale.

## Path 1 completed — now structurally parallel to Path 2, via OmniDraft (2026-07-20)

**[OmniDraft](https://arxiv.org/abs/2507.02659)** is the missing decode-phase piece for
the heterogeneous case, replacing the earlier vaguer SLEM/SLRS mention with a
concrete, validated mechanism. Full methodology documented above (online n-gram
cache with two mapping rules fixing the "false rejection" problem on n-gram
merges; hybrid distillation loss ℒ_DM + λℒ_N-gram, λ=0.2, trained ONLINE during
live inference on accepted/corrected target outputs; optional acceptance-
prediction early-exit head). Tested: Llama-68M drafter with Llama3-8B/Qwen2-7B/
Vicuna-7B targets. Acceptance rate improves 3-4x from distillation alone,
5-6x with full online-adaptive setup; speedups 1.22x-2.20x depending on task.
Theoretically lossless in final output distribution (rejection sampling still
holds); only intermediate n-gram probability handling is approximate.

**Path 1, restructured to match Path 2's shape:**
- Catch-up (prefill): unchanged — training-free SnapKV/PyramidKV.
- Generation (decode): OmniDraft's n-gram cache + hybrid distillation, now
  concrete instead of a placeholder.
- Research question, now genuinely parallel to Path 2's: does this mechanism
  remain reliable under repeated switching — or, since OmniDraft is DESIGNED
  to adapt online, does it actually IMPROVE as the conversation continues?
  Real point of contrast with Path 2: DroidSpeak's cache reuse is static
  (profiled once, fixed after) — can only stay flat or degrade. OmniDraft's
  cache/distillation keep learning from every accepted/corrected output — more
  interaction should mean a richer vocabulary bridge over time. Path 2 asks
  "does it degrade." Path 1 gets to ask "does it improve."

**Eval adjustment required**: OmniDraft's online learning signal only
accumulates while the large model is actively engaged and verifying — not
during small-model-only stretches. The right x-axis for the "does it improve"
hypothesis is CUMULATIVE LARGE-MODEL-ACTIVE TIME (total turns it has verified
across all its activation periods so far), not raw switch count — these
diverge if the large model handles a different number of turns each time it's
active. Track both, but test the online-adaptation hypothesis against
cumulative exposure specifically.

## THE UNIFIED DESIGN — Path 1 and Path 2 are the SAME mechanism, split by case
(2026-07-20, do not lose this connection again)

Core mechanism (user's original proposal): small model runs by default on every
turn; the large model is kept continuously "warm" by whatever cheap mechanism
the architecture allows; real speculative decoding (actual draft+verify+reject
sampling) only fires on turns the router flags for the large model — the large
model is never cold when it's actually needed.

**Same-family case → Path 2**: "keep the large model warm" = REAL tensor-level
KV cache sharing (DroidSpeak selective layer reuse, or LatentMAS layer-wise
concatenation), not token-level prefill — same-family means the shapes are at
least reachable, no re-tokenization needed to sync. Path 2's research question
IS this mechanism, precisely stated: does continuous real cache-sharing survive
repeated switching without degrading, and does classic (lossless) speculative
decoding's acceptance rate hold up once the large model steps in on a
router-flagged turn.

**Heterogeneous case → Path 1**: the same "keep it warm continuously" doesn't
transfer — blind prefill requires the large model to read the small model's
tokens, which needs shared vocabulary; continuous tracking would mean paying
re-tokenization cost every turn, possibly more than switch-only handling if the
router calls the large model rarely (established earlier). So heterogeneous
case = don't stay continuously warm; make each catch-up cheap only when it
happens (KV-Distill compression, current Component C design above). Once
caught up, same "real SD only when router flags it" idea still applies — using
heterogeneous-vocabulary SD (SLEM/SLRS, Component B) instead of classic SD,
since that's the version that survives the vocabulary gap.

**One design, two warm-keeping mechanisms** depending on what the architecture
allows — real cache sharing when free (same-family), compressed catch-up when
not (heterogeneous). Never two separate ideas.

---

## PATH 2 — parked, come back to this after Path 1 (2026-07-20)

**Status**: not started. Documented here so it isn't lost — pick this up once
Path 1 has run, or once the same-family-vs-proprietary-API server decision is
actually made (that decision changes whether this is worth doing at all — see
"why Path 1 first" below).

**Research question**: Does DroidSpeak-style cross-model KV cache reuse remain
reliable under REPEATED back-and-forth switching within a single conversation
(not just the single-hop handoff DroidSpeak was originally evaluated on) — and
does that same repeated reuse degrade speculative decoding's acceptance rate
over the course of a conversation, extending QuantSpec's "cache approximation
hurts SD acceptance rate" finding (established for quantization) to a
different approximation mechanism (cross-model layer-selective reuse) that
hasn't been tested this way by anyone?

**Why it's genuinely novel** (verified via search, not assumed):
- DroidSpeak: only evaluated for single-hop handoffs, never repeated switching
  in one session.
- QuantSpec et al.: established cache approximation hurts SD acceptance rate,
  but specifically for quantization, not for DroidSpeak's layer-reuse mechanism.
- Confirmed by auditing 6 separate paper clusters (routing papers, FrugalGPT's
  cascade, DroidSpeak, Cache-to-Cache, the entire edge-cloud SD cluster,
  speculative cascades) that NONE of them test a model going idle and later
  resuming mid-conversation with state intact — every one avoids the scenario
  by design (route once / escalate one-way / keep both models continuously
  co-active).

**Why it matters beyond novelty**: it's the load-bearing validation question
for the tiered-cascade production architecture (small on-device → same-family
mid-tier on server → rare proprietary-API escalation) — specifically whether
the cheap Tier1↔2 fast path is actually safe to switch on repeatedly, or
whether approximation compounds and periodic full recomputation is needed.
Bidirectionally valuable: degradation found = actionable negative result
(design needs periodic re-profiling); no degradation = validates the fast path
is safe as designed.

**Plan**:
1. Infra: same-family small/large pair (e.g. Llama-3.2/3.1 family). Implement
   DroidSpeak's selective layer-reuse mechanism — check for released code
   first rather than reimplementing from the paper description. Add a
   speculative decoding harness on top.
2. Sanity check: reproduce DroidSpeak's published single-hop numbers before
   trusting the implementation for anything new.
3. Core experiment: multi-turn conversations with controlled switch counts
   (2, 4, 8, 16), measuring at each switch count: (a) generation quality vs a
   full-recompute ground truth, (b) SD acceptance rate — plotted as a function
   of switch count to see whether approximation compounds, plateaus, or stays flat.
4. If degradation found: test whether periodic re-profiling of critical layers
   (redo DroidSpeak's offline search every K switches), or a stricter
   critical-layer threshold, recovers stability — this is what gives the paper
   a full diagnose-then-fix arc rather than just a diagnosis.
5. Write-up frames it against the tiered-cascade motivation.

**Honest scope**: NOT a 2-week project. Requires real model-internals
engineering (KV cache object manipulation, layer selection), not prompt-level
work. Realistically 4-8+ weeks, more if DroidSpeak's code isn't readily
adaptable.

**Why Path 1 first, not this**: Path 2's value is conditional on committing to
the same-family self-hosted server branch of the architecture — undecided as
of this writing. Path 1 (below) directly tests the cross-architecture scenario
that's the actual stated product target regardless of that decision, is faster
(~2 weeks vs 4-8+), and even a negative result is informative given the
confirmed gap. Revisit Path 2 once that server-architecture decision is made,
or once Path 1's result makes the same-family fallback look more attractive.

---

## CURRENT / FINAL 2-week POC plan (2026-07-20 revision — supersedes draft below)

**Scope decisions:**
- Building: Component A (router) + Component C (capsule). That's the whole POC.
- NOT building: Component B's algorithms (SLEM/SLRS/n-gram translation) — not needed.
  Those bridge vocabularies at token granularity inside one speculative-decoding
  generation call; this POC routes whole turns via text, which is already the
  validated bridging medium. No vocabulary-alignment code to write.
- NOT building: Component D (BiLD/CS-Drafting-style per-token tight coupling) —
  different mechanism/problem, excluded deliberately.
- NOT building: a same-family comparison arm (too much extra engineering for 2
  weeks — profiling pass, layer selection, pipelined loading). Instead: cite
  DroidSpeak's published numbers (4x throughput, 3.1x prefill) in the analysis
  writeup as ceiling context, without re-implementing that arm.

**Setup:** small/edge = local open model (Qwen2.5-3B-Instruct or Llama-3.2-3B-Instruct)
via vLLM/Ollama (gives logprob access for the router); large/cloud = hosted API
(Claude or GPT-4o) — guarantees different training lineage from the local model
by construction, satisfying the "meaningfully divergent pair" requirement for
free.

**Four conditions:** B0 always-large (quality ceiling) / B1 always-small (floor)
/ B2 naive switch + full-transcript resend (the baseline to beat) / B3 naive
switch + capsule handoff (the contribution).

**Component A:** local small model drafts an answer, average token logprob/entropy
as confidence signal, threshold τ (tunable, report a cost/quality curve not one
point) decides defer-to-large. No training.

**Component C:** capsule = {established_facts, open_threads, last_2_turns_verbatim,
running_summary}, updated after every turn on the local small model. Sent
instead of full transcript on switch.

**Eval — REVISED 2026-07-20 to test repeated/multiple switches, mirroring
Path 2's "does approximation compound over repeated switches" framing, applied
to the text-extraction mechanism instead of tensor cache reuse.** This makes
Path 1 and Path 2 a deliberately parallel pair: same underlying question
(does the bridging mechanism degrade under repeated back-and-forth switching),
asked once for training-light text extraction (cross-architecture case) and
once for trained tensor reuse (same-family case).

- **Two conversation sets, not one:**
  1. *Natural*: 10-15 conversations, router-driven switching (confidence
     threshold decides each turn naturally) — realistic behavior, but switch
     count/timing isn't controlled, so it alone can't produce a clean
     switch-count sweep.
  2. *Controlled*: a smaller set of longer conversations (20-30 turns) with
     switches **forced** at regular intervals, specifically to sweep switch
     count = 2, 4, 8, 16 within single conversations. This is the set that
     actually answers the compounding-degradation question.
- **Planted facts, staggered relative to switches** — not just "one fact
  needing to survive one switch" as before. Plant facts at different points so
  you can measure whether a fact survives being handed off once vs. multiple
  times as control keeps bouncing back and forth (e.g., a fact planted at turn
  1 has been handed off N times by turn 30; a fact planted right before the
  last switch has only been handed off once — compare retention across both).
- **Metrics, now plotted as a function of switch count/depth**, not just
  aggregated:
  1. Token cost to cloud model — cumulative AND per-switch, to see whether
     extraction's per-switch cost stays flat as switch count grows (it should,
     unlike B2's full-resend, which grows with total history every time).
  2. Latency.
  3. LLM-judge quality vs B0 reference.
  4. KVzip-style reconstruction probe accuracy, plotted against **how many
     switches ago** the probed fact was established — this is the actual
     degradation curve, directly parallel to Path 2's "SD acceptance rate vs
     switch count" plot.

**Day-by-day (revised):** Week 1 — D1 env+orchestrator+B0/B1, D2 B2 (hardcoded
routing), D3 router wired in (supports both natural and forced-switch modes),
D4 extraction mechanism (component C final design), D5 B3 working + write
BOTH conversation sets (natural + controlled switch-count sweep), staggering
planted facts relative to switch points. Week 2 — D6 eval harness runs all 4
conditions across both conversation sets, D7 LLM-judge + reconstruction-probe
scoring (now indexed by switch depth, not just aggregated), D8 analyze
(cost-vs-switch-count and retention-vs-switch-depth curves, not single numbers;
cite DroidSpeak ceiling numbers), D9 thin demo surface (CLI/Streamlit, live
routing + cost counter), D10 polish/README/demo recording/buffer.

**Scope check**: this is a real addition (staggered fact-planting, forced
switch-count sweep, per-depth analysis instead of aggregates) but still
prompt/orchestration-level engineering, not model-internals work — should
still fit the 2-week window. If it doesn't, cut the natural conversation set
first and keep only the controlled switch-count sweep, since that's the set
that actually produces the degradation-curve finding.

**Go/no-go:** router unreliable by D5 → hardcode routing pattern, spend all
remaining time on B2-vs-B3 comparison, that's the actual point.

---

## Earlier draft plan (superseded by above, kept for history)

2-week POC plan (scoped down — component C reduced from "solve cross-architecture
cache portability" to "prove structured text-state beats naive full-transcript resend")

Setup: small model local (Qwen2.5-3B-Instruct or Llama-3.2-3B-Instruct via
vLLM/Ollama) + large model via API (Claude/GPT-4o) — different architecture +
tokenizer for free, and local hosting gives access to raw logprobs for the router.

**Week 1**
- Day 1: env setup, minimal orchestrator loop, always-small/always-large baselines running
- Day 2: Baseline B2 — naive full-transcript resend on switch, manual/hardcoded routing
- Day 3: real router — small model's own logprob/entropy as confidence signal, threshold decides defer-to-cloud
- Day 4: structured state capsule (facts, open questions, last 1-2 turns verbatim, running summary) — the actual contribution
- Day 5: Baseline B3 (capsule handoff) working end to end; write 10-15 test conversations designed to stress cross-turn memory across a switch

**Week 2**
- Day 6: eval harness — run all 4 conditions (always-small, always-large, B2, B3), log tokens/latency
- Day 7: LLM-judge quality scoring against always-large reference
- Day 8: analyze — cost-saved vs quality-retained chart; iterate capsule schema if quality is bad
- Day 9: thin demo surface (CLI/Streamlit) showing live routing + running cost counter
- Day 10: polish, README, demo recording, buffer

**Go/no-go rule:** if router isn't reliable by day 5, hardcode the routing pattern
and put all remaining time into the capsule-vs-full-resend comparison — that's
the part with no existing precedent and the actual point of the POC.

**Explicitly out of scope for 2 weeks:** literal cross-architecture KV tensor
manipulation, training a custom router classifier, supporting more than one
small/large model pair.

---

## 2026-07-21/22 — Reverse-KL (OmniDraft-style) distillation attempt, session paused mid-run

**Goal:** replace the simplified-NLL LoRA training (used in all earlier runs)
with OmniDraft's actual loss: reverse-KL(drafter‖target) on aligned/expressible
vocab positions + λ·n-gram-NLL fallback on non-aligned positions (λ=0.2),
reusing HF's own `AssistantVocabTranslatorCache` / `AssistantToTargetTranslator`
internals for the vocab projection instead of hand-rolling it. Target:
Qwen3-0.6B (LoRA r=32/α=64) + `google/gemma-2-9b` (8-bit), the validated-good
pairing from earlier diagnostics. Script: `~/uag_reverse_kl_distill.py` on VM.

**Bugs hit and fixed this session:**
1. Dtype mismatch inside HF's `get_target_logits()` (`Float` vs `Half`) — fixed
   by casting `assistant_logits_full.float()` before the call.
2. Device mismatch (`cuda:1` vs `cpu`) inside the same call — root cause:
   `AssistantVocabTranslatorCache.get_translator(...)` was called with
   `assistant_model=None`, and HF's `AssistantToTargetTranslator.__init__` sets
   `self._assistant_model_device = "cpu"` whenever `assistant_model is None`,
   so internal tensors (e.g. `_assistant_to_target_input_ids`) got built on CPU
   while the real assistant logits live on `cuda:1`. **Fix: pass the real
   `base_assistant` model object into `get_translator(...)`, not `None`.** This
   fix is applied and confirmed to clear that specific error (training run got
   past model loading and into the forward-pass/eval phase without it recurring).

**n-gram-NLL fallback term is still stubbed as a no-op (contributes 0 loss)** —
not yet implemented faithfully; current loss is reverse-KL-only in practice for
sequences that are (mostly) vocab-aligned. Flagged as a known gap, not yet fixed.

**What happened after the device-mismatch fix:** relaunched successfully, both
models loaded (Gemma-2-9b 8-bit on cuda:0, Qwen3-0.6B fp16+LoRA on cuda:1), got
into the `measure_tpc()` held-out "BEFORE" eval loop (30 held-out prompts) —
but the process was **silently killed (SIGKILL, no Python traceback) after
~25-30 min of active compute**, most likely OOM on system RAM (box only has
**31GB total system RAM** — tight for an 8-bit-quantized 9B model + fp16 0.6B
drafter + repeated `generate()` calls with growing buffers). `dmesg`/syslog
were not accessible in this container to confirm definitively. Because stdout
was **block-buffered** (redirected to a log file via `nohup`, not a tty), none
of the internal print statements had flushed to disk before the kill — so we
lost all visibility into how far the eval loop got, even though it was clearly
making real progress (CPU time climbed steadily, process state was `R`, not
hung).

**Lesson learned (important for next time):** always run long unattended jobs
with unbuffered output (`python3 -u ...` or `print(..., flush=True)` on every
progress line) — otherwise a SIGKILL (OOM or otherwise) erases all progress
evidence even when the run was going fine up to that point. Block-buffering
made an otherwise-healthy-looking run look like a total mystery when it died.

**State at pause (session ended here, explicitly deferred to tomorrow):**
- GPUs confirmed fully free (0.0GB both) after the process died.
- `uag_reverse_kl_distill.py` on the VM has NOT yet been patched for next time
  (a patch was drafted — slice held-out set to 10 prompts instead of 30 for
  speed + lower memory pressure, add flushed per-prompt progress prints in the
  before/after eval loops — but was not applied; user asked to stop for the
  night before it was uploaded).
- No successful reverse-KL training run has completed yet. No before/after
  tok/call numbers exist for this approach.
- VM/Kaggle session itself was left running (could not be remotely stopped —
  no Kaggle API credentials configured locally, no browser/UI access from
  here; only the user can click "Stop Session" or it idles out on its own).

**Next session should:**
1. Apply the drafted patch (10 held-out prompts, `flush=True` on progress
   prints, ideally also relaunch via `python3 -u`) before restarting.
2. Consider also reducing `max_new_tokens` in `measure_tpc`/`get_target_continuation`
   and/or adding explicit `del`/`gc.collect()` more aggressively if OOM recurs —
   31GB system RAM is the binding constraint, not GPU VRAM.
3. If OOM recurs even after these fixes, consider dropping 8-bit target
   quantization overhead as a suspect (dequant buffers) or profiling system RSS
   directly during the eval loop rather than guessing post-hoc.
4. Implement the still-stubbed n-gram-NLL fallback term for full fidelity to
   OmniDraft's actual dual-loss design, once the reverse-KL-only path is
   confirmed working end-to-end.

---

## NEW IDEA (2026-07-22): small-to-big handoff via prefill-as-verification —
the dual of the big-to-small free byproduct, not a separate mechanism

**Core reframe**: don't treat small-to-big escalation as "resend text and
hope." A big model's prefill pass necessarily computes its own predicted
logits at every position anyway — that's just what a forward pass does. So run
the small model's entire solo stretch (everything since the last escalation)
through the big model as **one giant SD-style verification pass**, exactly
like normal per-turn SD verification, just at a coarser grain (whole
uncorrected interval instead of one draft window).

**Cost, stated honestly**: this is NOT cheaper than a plain prefill — a
verification forward pass and a prefill forward pass are the same operation
(forward pass over N tokens, all positions computed in parallel regardless of
where anything diverges). No FLOPs are saved versus naive resend. What's
gained is **information already latent in compute you had to pay for anyway**,
not reduced compute.

**What falls out for free, as a byproduct of that unavoidable prefill:**
1. **The real catch-up boundary** — the first position where the big model's
   own preference stops matching what the small model actually said. Only
   tokens after this point are where the big model's context differs from the
   small model's; everything before is already-agreed history.
2. **A free, retroactive audit of the small model's whole unsupervised
   stretch** — a coarser, higher-value version of the existing "SD acceptance
   rate as router calibration" idea (see the SETTLED/SHARPENED sections
   above), now measured over the full interval since last escalation instead
   of per-turn. Early divergence = router waited too long to escalate (should
   recalibrate toward escalating sooner). Late/no divergence = router timing
   was about right or even slightly premature.
3. **A natural, non-invented correction trigger** — reuse BiLD's rollback
   policy (already cited for token-level SD correction) at the divergence
   point, instead of designing a new repair mechanism from scratch.

**Why this completes the symmetry**: big-to-small is free because the small
model is the drafter during every big-model turn (structural, proven).
Small-to-big becomes free by the same LOGIC (byproduct of computation already
required, not new infrastructure) even though the underlying computation
(prefill) isn't itself free — the free part is the *information extracted*
from a cost you were always going to pay, not the cost itself. Neither
direction needs a dedicated handoff mechanism; both are byproducts of
computation the models were already forced to do.

**Real, un-glossed risk**: raw token-for-token match is too strict for the
heterogeneous-vocab case (our actual target pairing) — the big model may
phrase things differently while agreeing semantically, especially across
tokenizers. Fix: compare in *projected* vocab space, not raw token IDs — which
means this reuses the exact `AssistantVocabTranslatorCache` /
`get_target_logits()` machinery already being built and debugged tonight in
`uag_reverse_kl_distill.py`. Not a new research thread — the same cross-vocab
projection code, applied to a second purpose. Not yet implemented or tested —
flag as a design idea, not a validated mechanism, same status as the
translation-only continuous-prefill idea above.

**Nevertheless, continuous online distillation of the drafter keeps running
regardless of this idea** — this is additive, not a replacement. The
reverse-KL/n-gram training thread (drafter learning to track the target's
distribution better over time) is orthogonal to this handoff idea and should
keep running continuously in the background of the real system either way —
a better-trained drafter makes SD's normal per-turn verification more
efficient AND makes this coarser escalation-time verification pass more
likely to show high agreement (fewer false divergences caused by the drafter
simply being undertrained rather than the small model being "wrong").

---

## 2026-07-22: FIRST REAL REVERSE-KL RESULT — root cause found and fixed,
full run completed, result is real but mixed, not yet a confident verdict

**Root cause of the universal-NaN bug from last session, found and fixed**:
`AssistantToTargetTranslator.FILTER_VALUE` (HF's own constant) is literally
`-float("Inf")`, not a large-finite-negative sentinel. The loss computation
did `q.exp() * (q - log(p))` where `q = log_softmax(projected)` — at any
position the assistant can't express, `q` is real `-inf`, so `exp(-inf)=0`
times `(-inf - finite)` is `0 * -inf = NaN` under IEEE754. The subsequent
`* valid_mask` step meant to zero these out happens too late: `NaN * 0` is
still `NaN`, not `0`. Since the KL sum runs over the full ~256k target vocab,
one such position poisons the entire row — explaining the observed
**105/105 skipped, avg loss nan** from the previous session's completed run.
Confirmed directly: `translator.FILTER_VALUE` really is `-inf` (checked HF
source), and a 3-prompt verbose smoke test showed `projected has -inf: True`
at every position, `q`/`p`/`kl_per_pos` all nan-free only AFTER the fix.

**Fix**: replace `-inf` with a large-but-finite negative (`-1e4`) via
`masked_fill` on `projected` BEFORE `log_softmax`, so `q` itself never
contains real `-inf` — `exp()` of `-1e4`-scale values already underflows
cleanly to `0.0` without ever forming the `0 * -inf` NaN, in both forward and
backward (avoids a known PyTorch gotcha where `torch.where` doesn't reliably
rescue NaN-valued masked branches in the backward pass either).

**Full run, post-fix, on fresh VM (Qwen3-0.6B LoRA r=32 + google/gemma-2-9b
8-bit, 105 train / 10 held-out, same LoCoMo split)**:
- Training: 0/105 skipped (vs 105/105 before the fix) — loss fell from
  ~2.3-2.8 at the start down to ~0.78-1.3 range by the end (avg loss 1.394),
  confirming the objective is real and doing work on the training set.
- **BEFORE avg tok/call: 2.292 → AFTER avg tok/call: 2.357 (+2.80%)**

**Honest read, not oversold**: +2.80% is a real, positive average — but
per-prompt it's genuinely mixed, not a consistent win: 5/10 held-out prompts
improved (1, 5, 7, 8, 9), 5/10 got worse (0, 2, 3, 4, 6), one roughly flat.
Contrast with the earlier plain-NLL training result (different pairing:
Llama-68M drafter, 13/15 held-out prompts won or tied, only 1 clear
regression) — that was a consistent effect; this reverse-KL result is not,
at least not yet, at n=10. With this much per-prompt variance on only 10
held-out prompts, **do not treat +2.80% as a confident "reverse-KL beats
plain NLL" verdict** — it could be noise. Also notable: BEFORE avg here
(2.292 tok/call, untrained Qwen3-0.6B+Gemma-2-9b) is itself much higher than
the untrained Llama-68M+Gemma pairing's earlier numbers (~1.47-1.56) —
consistent with Qwen3-0.6B being a stronger/better-matched drafter for Gemma-2
than Llama-68M, independent of any training.

**What would raise confidence, next session**: re-run with a larger held-out
set (the original 30, now that the fix means training won't silently no-op)
to reduce the per-prompt noise that dominates at n=10, and/or run the
apples-to-apples comparison against plain-NLL training on this SAME
Qwen3-0.6B+Gemma-2-9b pairing (not yet done — all plain-NLL numbers on record
so far used the Llama-68M drafter, a different pairing, so it's not a valid
head-to-head yet).

**Trained adapter saved on VM**: `trained_drafter_reverse_kl_gemma2_9b/` —
not yet downloaded locally, VM may not persist between sessions (Kaggle-style,
same caveat as always — download before assuming it survives).

---

## 2026-07-22 (same day, later): 30-prompt re-eval CONFIRMS the result is
real, not noise — n=10 was underpowered exactly as suspected

Re-ran the already-trained adapter (no retraining, just eval) against the
FULL 30-prompt held-out set instead of the 10-prompt subset, comparing fresh
untrained Qwen3-0.6B vs the reverse-KL-trained adapter, same protocol:

**UNTRAINED avg tok/call: 2.159 → TRAINED avg tok/call: 2.388 → CHANGE: +10.60%**
**Per-prompt: 17/30 improved, 10/30 worse, 3/30 tied**

This resolves the earlier uncertainty cleanly: tripling the sample size both
INCREASED the magnitude (+2.80% at n=10 → +10.60% at n=30) and cleaned up the
win-rate (50/50 split at n=10 → 57%/33%/10% at n=30, a clear majority win).
Confirms the n=10 result was genuinely underpowered/noisy, not evidence of a
weak or unreliable effect — **this is now a confident, real, positive result
for OmniDraft-style reverse-KL training on Qwen3-0.6B + Gemma-2-9b.**

Still outstanding: the actual apples-to-apples comparison against plain-NLL
training on this SAME pairing (not yet run — see above), and the n-gram-NLL
fallback term is still stubbed at 0 (see above) — closing that gap could push
the result higher still, since ~15-25% of continuation tokens per prompt
currently get zero training signal.

---

## 2026-07-22: LinUCB router — first working version built and tested

Built `linucb_router.py` (local, no GPU needed) implementing the router
design settled earlier (see "SETTLED... LinUCB contextual bandit" above):
2-arm (`small_alone` / `escalate_sd`), Sherman-Morrison rank-1 updates (no
full matrix inversion per round), offline ridge-regression warm-start
support, exploration-bonus arm selection. Mechanics verified correct on a
10-round demo using real held-out prompts + real measured tok/call data from
tonight's reverse-KL eval run (not synthetic numbers).

**Reward function for `escalate_sd` arm**: implements the router-calibration
insight directly — HIGH realized tok/call means the small model probably
could have handled the turn alone (escalation likely unnecessary), so reward
is `correction_need - escalate_cost`, NOT naively "high tok/call = good."
This is the first actual code implementing that insight, not just the design
note.

**Reward function for `small_alone` arm — real pivot, worth recording**:
originally planned to use "chat-based dissatisfaction detection" (user's next
message reveals dissatisfaction), but discovered LoCoMo is synthetic
persona-to-persona dialogue (Deborah/Jolene style), not chatbot interaction
logs — there is no "user reacts to a model's answer" structure to mine here.
Pivoted to a different but still-real signal: extracted the ACTUAL recorded
next turn at each truncation point (30 held-out pairs, `heldout_pairs_groundtruth.json`,
via a modified `prep_locomo_data.py` that keeps `turns[end_idx]` instead of
discarding it) and score the small model's own generated continuation against
it via word-level F1 overlap (`uag_small_alone_reward.py`, cheap, no external
LLM-judge API needed). This is a genuine quality-similarity proxy, not a
placeholder — but it is a different signal than dissatisfaction-detection,
and should not be conflated with it in any writeup. Real dissatisfaction
detection would still need actual chatbot interaction logs, which don't exist
in this project yet. AVG F1 across 30 prompts: 0.163 (range 0.000-0.311).

**Data-alignment catch, worth recording as a process lesson**: the freshly
extracted `heldout_pairs_groundtruth.json` (30 prompts) turned out to be a
COMPLETELY DIFFERENT set of 30 prompts than the original `heldout_prompts.json`
used for the tok/call re-eval above — 0/30 exact string matches, despite
identical seed (42) and identical extraction logic. Root cause not fully
confirmed, but most likely explanation: the freshly re-cloned `locomo10.json`
tonight differs slightly in content/ordering from whatever version was used
whenever `heldout_prompts.json` was first generated in an earlier session.
**Lesson: same seed + same code does NOT guarantee same output if the
underlying data file itself might differ between runs — verify data identity
directly (e.g. exact-match check), don't assume determinism from seed alone.**
Fixed by re-measuring tok/call fresh on the SAME `heldout_pairs_groundtruth.json`
prompts (`uag_escalate_reward_aligned.py`) rather than trying to reconcile the
two sets — cost ~15 min GPU time, result: **AVG tok/call (aligned set): 2.395**
— consistent with the 2.388 trained-avg from the other 30-prompt set, good
external-consistency signal that the reverse-KL result generalizes across
different prompt samples, not an artifact of one specific 30-prompt draw.

**Full 30-round LinUCB demo, now fully aligned (real tok/call + real F1, same
30 prompts for both)**: mechanics confirmed working end-to-end — Sherman-
Morrison updates, exploration-bonus arm selection, all behaving correctly on
real data. Result: router picked `escalate_sd` 28/30 rounds, `small_alone`
only 2/30. **This is a calibration artifact, not a validated routing
recommendation** — the reward constants (`escalate_cost=0.3`, small_alone
baseline `-0.1`) are placeholders with no relationship to real-world cost.
Real escalation has real dollar/latency cost (cloud LLM call vs on-device)
that isn't reflected in these constants yet. The bandit math is proven
correct; the reward CALIBRATION between arms is the remaining real gap before
this router recommendation means anything deployable. Next step: replace the
placeholder cost constants with actual measured latency/cost figures for
escalate vs small-alone on the real target hardware, so "which arm wins"
reflects genuine tradeoffs rather than arbitrary numbers.

**Also still placeholder**: the drafter-confidence context feature (one of
three inputs to the router's context vector) still uses synthetic random
values, not a real drafter forward pass — everything else in this demo
(both rewards, the 30 prompts, the bandit algorithm) is real measured data;
this one feature is not yet wired to a live confidence computation.

---

## MAJOR CORRECTION (2026-07-22): full wall-clock comparison reverses the
verdict — +10.6% tok/call does NOT mean reverse-KL training is a real win

Ran the full baseline/untrained-SD/trained-SD wall-clock comparison (matching
yesterday's protocol exactly: separate GPU placement, aggressive cache
clearing, real timed seconds) on the same aligned 30-prompt set
(`uag_full_speedup_comparison.py`). Result:

| | Untrained | Trained (reverse-KL) |
|---|---|---|
| Avg wall-clock speedup | 1.295x | **1.042x** |
| Median speedup | 1.307x | 1.036x |
| Avg tok/call | 2.166 | 2.395 |
| Per-prompt wall-clock | — | **2 improved, 28 worse (of 30)** |

**tok/call improvement is real and now confirmed a third time (+10.6%,
consistent with the two earlier measurements) — but it does NOT translate to
a wall-clock win.** Wall-clock speedup collapsed from 1.295x to 1.042x,
almost entirely erasing the benefit of SD, and got WORSE on 28/30 individual
prompts. **Correcting the earlier "+10.60% confirmed positive result" framing
— that number is real, but it does not mean the trained adapter is a
practical improvement. As deployed, it is not.**

**Same paradox pattern as flagged earlier in the session, but now demonstrated
with drafter SIZE held constant** (untrained vs trained is the SAME Qwen3-0.6B
base model, only a small LoRA delta differs) — ruling out "bigger/slower
drafter" as the explanation this time, which sharpens the diagnosis. Most
likely mechanism: HF's adaptive draft-length heuristic responds to a
higher-accepting drafter by proposing LONGER speculative sequences per round.
Fewer total rounds are needed (hence better tok/call), but each round now
costs more wall-clock — longer sequential drafter rollout, longer target
verification pass over more positions (8-bit dequant overhead scales with
sequence length) — and that per-round cost increase apparently outpaces the
savings from needing fewer rounds.

**Actionable next experiment, not yet run**: cap/fix the draft length (disable
the adaptive heuristic, use a constant small draft length) and re-measure —
if wall-clock recovers under a fixed draft length, this confirms the adaptive
heuristic is the actual culprit, and the fix is either disabling adaptivity
or re-tuning its growth policy, not abandoning reverse-KL training itself.
Until that experiment runs, **do not deploy this trained adapter, and do not
cite "+10.6% tok/call" as evidence reverse-KL training helps** — cite the
wall-clock number as the real verdict, since that's what a user actually
experiences.

---

## 2026-07-22 (later): ratio experiments — bigger ratio helps (confirmed),
same-family vs cross-family is genuinely inconclusive (confounded)

**Motivation**: real deployment target is a 100B+ server model, far bigger
than our 9B testbed — wanted to know if ratio effects (drafter:target size)
seen in the literature actually hold directionally on our own hardware/setup,
within the range reachable on 2x T4.

**Ratio test 1 — Qwen2.5-0.5B-Instruct (0.494B) + Gemma-2-9b, ~18.2:1 ratio**
(vs Qwen3-0.6B's ~15:1 baseline from earlier): avg speedup **1.479x**, median
1.447x, avg tok/call 2.202, 0/30 below 1.0x. A modest ratio bump (15:1→18.2:1)
gave a real, clean **+14% relative speedup improvement** — not noise, this
directly supports the ratio hypothesis even at small ratio deltas.

**Note on Qwen family floor**: Qwen doesn't ship anything smaller than ~0.5B
in its standard lineup, so this is close to the floor for "stay within Qwen
as drafter" — genuinely testing literature-scale ratios (28:1-280:1) needs
either a non-Qwen tiny drafter (we have Llama-68M, ~132:1 against Gemma-2-9b)
or a much bigger target (not reachable on 2x T4, ~9B is near the ceiling for
8-bit on a single T4's usable VRAM). Prepared `ratio_test_for_a6000.py`
(Llama-68M + Gemma-2-9b, ~132:1) for the user to run on their own A6000 —
recommends fp16 target instead of 8-bit there, since 8-bit dequant overhead
was a real confound in our own numbers all session and a 48GB card doesn't
need it for a 9B model.

**Ratio test 2 — same-family Qwen3-0.6B + Qwen2.5-7B-Instruct, ~11.7:1 ratio**:
avg speedup **0.977x** (essentially break-even, slightly negative), median
0.974x, avg tok/call 2.047, **20/30 prompts below 1.0x**. Surprising: this is
WORSE than either Gemma cross-family result (1.295x, 1.479x) despite
near-total vocab overlap (151665 vs 151669 tokens, 100% raw key overlap).

**Real technical surprise along the way**: despite the near-identical vocab,
HF's `_validate_generation_mode` still raised `ValueError: The main and
assistant models have different tokenizers` and forced the cross-vocab UAG
code path (same `tokenizer=`/`assistant_tokenizer=` machinery used for Gemma)
— it does NOT do a simple vocab-content-overlap check, it's stricter
(likely a class/config identity check). So this "same-family" pairing still
paid the UAG code path's overhead, just with near-lossless translation rather
than the ~25% expressible fraction Gemma showed.

**Why the same-family result is NOT a clean refutation of "same family wins"**:
two real confounds stack against it simultaneously — (1) smaller ratio here
(11.7:1) than either Gemma test (15:1, 18.2:1), and per test 1 above, ratio
demonstrably matters; (2) Qwen2.5-7B-Instruct's own baseline is much faster
in absolute terms (~12.6s/prompt) than Gemma-2-9b's (~17-19s/prompt) — a
faster baseline leaves less absolute wall-clock slack for SD to profitably
fill, independent of vocab/family. **Conclusion: this session's data does
NOT establish that cross-family beats same-family — it shows ratio and
absolute-target-speed both matter, and this particular same-family test
wasn't ratio- or speed-matched against the cross-family tests, so it can't
isolate the family variable alone.** A fair test would need a same-family
pair at a MATCHED ratio (e.g., a ~9B same-family target, or a smaller
same-family drafter to hit ~15-18:1) — not yet run.

**Updated missing-pieces list (supersedes/extends earlier ones)**:
1. Fixed-draft-length test (diagnose the reverse-KL wall-clock paradox)
2. A same-family test at a ratio MATCHED to the cross-family tests (isolate
   family from ratio/absolute-speed confounds) — not yet run
3. n-gram-NLL fallback term in reverse-KL loss (still stubbed at 0)
4. Real plain-NLL vs reverse-KL head-to-head on the SAME pairing (never done)
5. Router reward calibration with real cost/latency numbers (currently
   placeholder constants, router picks escalate 28/30 times meaninglessly)
6. Router confidence feature still synthetic (not wired to live drafter)
7. Small-to-big prefill-as-verification — pure design, zero code/test
8. Continuous tracking (v3 architecture) — designed, not built
9. True large-ratio test (100B+ target) — deferred to user's A6000,
   `ratio_test_for_a6000.py` ready to run there

---

## MAJOR METHODOLOGICAL FINDING (2026-07-22): "same family" tests tonight
NEVER actually reached HF's native, zero-translation code path — all of them
were secretly still going through the cross-vocab translator

**User pushback that triggered this**: standard, same-tokenizer speculative
decoding is a well-proven technique in the literature — a same-family pair
performing WORSE than cross-family (the 0.977x same-family Qwen result above)
should be treated as a red flag that something is wrong in OUR harness, not
casually accepted as a "ratio effect" finding. This was the right call.

**Root cause, found by reading HF's actual source**
(`GenerationMixin._validate_generation_mode`): the same-tokenizer vs
cross-tokenizer decision is NOT based on comparing actual tokenizer content
(vocab keys, class, etc.) — it's based purely on whether
`self.config.get_text_config().vocab_size == assistant_model.config.get_text_config().vocab_size`
(the model config's declared embedding-table size). If they differ, HF
REQUIRES `tokenizer=`/`assistant_tokenizer=` and forces the cross-vocab UAG
path regardless of how similar the tokenizers actually are.

**Confirmed empirically**: Qwen2.5-0.5B-Instruct and Qwen2.5-7B-Instruct have
IDENTICAL tokenizer class and IDENTICAL real vocab (151665 tokens each,
verified directly) — yet their model configs declare DIFFERENT `vocab_size`
(151936 vs 152064, different embedding-table padding amounts per model size)
— so even this genuinely same-tokenizer pair was forced through the
cross-vocab translator, exactly like every Gemma test tonight. **Every "same
family" test run this session (Qwen3-0.6B+Qwen2.5-7B-Instruct, and initially
attempted Qwen2.5-0.5B+Qwen2.5-7B) was secretly cross-vocab-path the whole
time** — the earlier "same-family performed worse than cross-family" finding
was NOT comparing native vs cross-vocab SD at all, it was comparing two
DIFFERENT DEGREES of cross-vocab compatibility (near-100% expressible for
Qwen-Qwen vs ~25% for Qwen-Gemma), which is a different and much less
surprising comparison than it first appeared.

**Fix found**: checked `config.vocab_size` across the Qwen2.5 lineup —
0.5B/1.5B/3B all share `vocab_size=151936`; 7B/14B share `152064`. So a
genuinely native, zero-translator test IS possible within the SAME padding
tier: **Qwen2.5-0.5B-Instruct + Qwen2.5-3B-Instruct** (~6:1 ratio, smaller
than any test run tonight). Launched as `uag_truly_native_test.py` — this is
the actual control test that isolates whether our timing harness itself is
sound, independent of the translator/UAG machinery entirely.

**RESULT (completed): avg speedup 1.339x, median 1.326x, avg tok/call 2.220,
0/30 prompts below 1.0x.** Clean, solid, consistently positive across all 30
prompts — exactly what textbook same-tokenizer SD should look like, despite
the smallest ratio tested all session (~6:1). **This confirms the timing
harness itself is sound** — the earlier weak same-family numbers were
entirely an artifact of being forced through the cross-vocab path, not a
harness bug and not a real ratio/family effect at that magnitude.

**Lesson for future sessions**: don't assume "same tokenizer class" or "high
vocab-key overlap" means HF will treat two models as same-tokenizer for
assisted generation — the actual gate is `config.vocab_size` equality, an
embedding-table-size proxy that can differ between checkpoints of the exact
same tokenizer due to padding. Check `AutoConfig.from_pretrained(...).vocab_size`
directly before assuming a pair will hit the native fast path.

---

## MAJOR FINDING (2026-07-22): the reverse-KL wall-clock regression is NOT
evidence against reverse-KL training — we implemented an incomplete OmniDraft,
missing the exact component that prevents this failure mode

**User pushback that triggered this** (correctly): OmniDraft reports genuine
positive wall-clock results with reverse-KL training. If ours regresses, the
default assumption should be "we're missing something from their actual
architecture," not "reverse-KL doesn't work." Went back to the actual paper
([arxiv 2507.02659](https://arxiv.org/abs/2507.02659)) to check precisely
what we borrowed vs. what we skipped.

**What we actually implemented**: only the reverse-KL/n-gram HYBRID LOSS
(and even that incompletely — the n-gram term has been stubbed at zero
contribution all session), grafted onto HF's off-the-shelf generic UAG
pipeline (static `AssistantVocabTranslatorCache` for vocab projection,
generic adaptive draft-length heuristic).

**What OmniDraft's actual system has, that we don't**:
1. **A dedicated, trained lightweight acceptance-PREDICTION head** — explicitly
   estimates P(∃ rejection) and stops drafting early once a threshold γ
   (task-tuned, 0.3-0.7) is crossed. This is the direct, precise explanation
   for our paradox: a better-trained drafter (higher real acceptance) with NO
   dedicated early-exit mechanism just lets a GENERIC heuristic keep extending
   draft length based on recent acceptance streaks — overshooting into longer,
   costlier windows exactly because training worked, with nothing to tell the
   system when to stop. This is the precise failure mode OmniDraft built a
   component to prevent, and we don't have that component.
2. **A genuinely different translation mechanism** — an online n-gram cache
   that MERGES SUB-TOKEN SEQUENCES (e.g. drafter tokens
   `['d','f','la','ke']` → target token `['flake']`), not a static 1:1
   per-token lookup like HF's translator. Different mechanism, not just a
   simplified version of the same idea.
3. **Their own ablation confirms the n-gram loss term matters**: "ℒDM + λℒN-gram
   performs better than training only on ℒDM" — stated directly in the paper.
   Our stubbed-at-zero n-gram term isn't a minor completeness gap, it's a
   component they show measurably helps.

**Why our comparison and theirs were never actually the same experiment**:
OmniDraft's reported 1.70x (vs 0.94x untrained, on GSM8K) is measured with
their FULL system (n-gram cache + acceptance head + LoRA training) against
their own untrained-with-full-system baseline. We measured "training alone,
grafted onto a generic HF pipeline" vs "no training, same generic pipeline" —
a different, narrower ablation than theirs. The discrepancy doesn't
contradict their result; it shows we tested a different (incomplete) system.

**Corrected conclusion, superseding the "MAJOR CORRECTION" entry above**:
reverse-KL training is NOT disproven by tonight's wall-clock regression — we
validated an incomplete implementation missing the specific component
designed to prevent exactly the failure we observed. Do not conclude
"reverse-KL doesn't help" from this session's data. The real next step, before
any further verdict, is implementing the acceptance-prediction head (the
higher-priority missing piece, given it directly explains the observed
failure) — not abandoning the training approach.

---

## ⭐ REVISED PLAN (2026-07-22, supersedes all earlier priority lists in this
file) — read this first when resuming

**Tier 1 — cheap, fast, diagnostic, do these before building anything new:**
1. Native-tokenizer harness sanity test (`uag_truly_native_test.py`,
   Qwen2.5-0.5B+3B, matched `config.vocab_size`) — already running, no
   retraining needed, just confirms our timing harness gives sane numbers
   under textbook same-tokenizer conditions. Check result before anything else.
2. **Fixed/capped draft-length test using the ALREADY-TRAINED reverse-KL
   adapter** (`trained_drafter_reverse_kl_gemma2_9b/`, already downloaded
   locally) — no retraining required, just rerun the wall-clock comparison
   with HF's adaptive draft-length heuristic disabled (fixed small
   `num_assistant_tokens`, no growth). If wall-clock recovers under a fixed
   window, that's cheap, direct confirmation of the adaptive-heuristic-
   overshoot diagnosis before committing to the bigger build below. NOT YET
   RUN — do this first, it's nearly free compared to what follows.

**Tier 2 — the actual fix, bigger build, only after Tier 1 confirms:**
3. **Build OmniDraft's acceptance-prediction head** — the highest-priority
   missing piece, directly explains the observed wall-clock regression.
   Concretely: (a) accept/reject training LABELS can be extracted from data
   already generated inside the existing reverse-KL training loop (compare
   assistant's aligned predictions against the target's actual distribution
   at each position — no new data collection needed); (b) train a small
   head (linear/shallow MLP on drafter hidden states) to predict
   P(∃ rejection); (c) HF's `generate()` has no hook for a custom trained
   early-exit signal, so this needs either a custom draft-verify loop
   (bypassing HF's built-in candidate generator) or monkey-patching the
   adaptive-length logic — needs investigation, not yet designed in detail.
4. Implement the n-gram-NLL fallback term for real (currently stubbed at 0)
   — OmniDraft's own ablation states it measurably helps on top of the KL
   term alone, not just a completeness nice-to-have.

**Tier 3 — parallel track, independent, continue alongside Tier 1/2:**
5. Router reward calibration — replace placeholder cost constants with real
   latency/cost numbers (currently picks escalate 28/30 times meaninglessly).
6. Wire the router's confidence feature to a real drafter forward pass
   (currently synthetic random values).
7. Conversational context integration into the router — currently treats
   every turn as independent; needs to pull from the same conversational-
   state representation as Component C (capsule/continuous-tracking design),
   which has never been connected to the router at all. Bigger lift than the
   other router items, flagged as its own item, not a quick fix.

**Tier 4 — lower priority / deferred, not blocking anything above:**
8. Same-family test at a ratio MATCHED to the cross-family tests (isolate
   family from ratio/absolute-speed confounds properly — today's same-family
   test conflated both).
9. True large-ratio (100B+ target) test — deferred to user's own A6000,
   `ratio_test_for_a6000.py` ready.
10. Small-to-big prefill-as-verification — pure design, zero code/test.
11. Continuous tracking (v3 architecture) — designed, not built at all.

---

## RESULT (2026-07-23): fixed-draft-length test — adaptive heuristic is a
REAL, PARTIAL factor, but does NOT fully explain the regression

Ran `uag_fixed_draft_length.py` (reused the already-trained adapter, no
retraining) with `num_assistant_tokens_schedule='constant'`,
`num_assistant_tokens=4` fixed for both untrained and trained conditions,
same 30-prompt aligned set.

| | Adaptive (original) | Fixed length=4 |
|---|---|---|
| Untrained avg speedup | 1.295x | 1.296x |
| Trained avg speedup | 1.042x | **1.210x** |
| Trained avg tok/call | 2.395 | 2.385 |
| Per-prompt (fixed) | — | 8/30 improved, 22/30 worse |

**Fixing draft length recovered a real, meaningful chunk of the gap**
(1.042x → 1.210x) — confirms the adaptive heuristic genuinely was hurting
the trained drafter, not a red herring. **But trained is STILL worse than
untrained even under matched, fixed conditions** (1.210x vs 1.296x, still
22/30 regressed) — so the adaptive-heuristic theory explains PART of the
problem, not all of it.

**New hypothesis, logically forced by the data**: since tok/call is clearly
higher for trained (2.385) than untrained (2.082) even at fixed draft
length, fewer total rounds should be needed for trained — which should make
it FASTER, not slower, all else equal. The only way this still nets negative
is if the trained drafter's PER-FORWARD-PASS cost itself increased enough to
outweigh needing fewer rounds. Leading candidate: **the LoRA adapter itself
adds real compute overhead to every single drafter forward pass** (low-rank
A/B matmuls on top of frozen base weights, plus potential PEFT wrapper
overhead vs a raw model call) — not a heuristic/architecture issue at all,
a much simpler "the trained model is measurably heavier per step" effect.

**Directly testing this now**: `uag_lora_overhead_test.py` — isolated,
pure drafter-only generation (no SD, no target model at all), base Qwen3-0.6B
vs the same model with the reverse-KL LoRA adapter attached, 30 prompts each.
If the LoRA-wrapped version is measurably slower per token in isolation,
that's the real remaining culprit, independent of anything about draft-length
heuristics or the OmniDraft n-gram-cache/acceptance-head gaps discussed
above.

**RESULT (confirmed): LoRA-wrapped drafter is 1.443x the wall-clock time of
the base drafter — 44.3% slower, 30/30 prompts, completely consistent, not
noise.** This is a real, structural, substantial cost from carrying the LoRA
weights (r=32 adapter on q/k/v/o projections) — not a heuristic issue, not an
architecture gap, just the trained model being measurably heavier per
forward-pass step. This is more than sufficient to explain the residual gap
left after fixing draft length (1.210x trained vs 1.296x untrained, fixed
length): the drafter's own generation time is only part of each SD round's
total cost (target verification is the other, likely dominant, part on a 9B
8-bit target) — a 44% increase in just the drafter's share, diluted across
the full round cost, lines up with the observed ~6-7% net regression.

**The wall-clock puzzle is now essentially resolved, not mysterious**: two
concrete, quantified, real costs fully account for it —
(1) the adaptive draft-length heuristic overshooting on a higher-accepting
    drafter (recovered most of the gap: 1.042x → 1.210x when fixed),
(2) LoRA's genuine ~44% per-forward-pass overhead on the drafter itself
    (accounts for the remaining ~6-7%).
Neither of these means reverse-KL training or the underlying idea is bad —
both are OFFSETTABLE costs that OmniDraft's full system (n-gram cache +
acceptance head) is specifically designed to pay for via savings elsewhere
(fewer, better-targeted rounds). We built the part that INCURS these costs
(the trained LoRA weights) without building the parts that PAY for them.
Building the acceptance-prediction head and n-gram cache remains the correct
next step — not abandoning the approach, and not a new mystery to chase.

**Open sub-question, not yet answered**: how much of the 44% LoRA overhead is
inherent to any r=32 LoRA adapter on this architecture vs. specific to our
PEFT/bitsandbytes stack possibly using unfused/unoptimized kernels compared
to a production deployment (e.g., merged LoRA weights at inference, which
eliminates the low-rank matmul overhead entirely by baking LoRA into the base
weights once, post-training). Worth checking `merge_and_unload()` (PEFT's
weight-merging function) as a cheap potential fix before assuming the 44%
cost is unavoidable — merging would eliminate LoRA's runtime overhead
entirely while keeping the trained behavior, since the low-rank update gets
folded into the base weights as a one-time operation rather than computed on
every forward pass.

**Isolated result (confirmed): merging fully recovers the drafter-only
overhead** — merged model ran at 0.976x of base (fully gone, within noise),
vs the unmerged adapter's 1.443x. Immediately tested whether this recovers
the FULL SD wall-clock comparison too (`uag_merged_sd_comparison.py`, same
30-prompt set, adaptive draft length default):

| | Untrained | Trained, unmerged, adaptive | Trained, merged, adaptive | Trained, unmerged, fixed-4 |
|---|---|---|---|---|
| Avg speedup | 1.313x | 1.042x | **1.208x** | 1.210x |
| Per-prompt | — | mostly worse | 9 improved / 21 worse | 8 improved / 22 worse |

**Merging alone gets real, meaningful improvement (1.042x → 1.208x) but does
NOT fully close the gap to untrained (1.313x)** — still 21/30 regressed.
Notably, merging-alone and fixed-length-alone land in almost the same place
(1.208x vs 1.210x) despite being independent fixes for two different
identified problems (LoRA per-step overhead vs. adaptive-heuristic
overshoot) — suggesting these are two SEPARATE, ADDITIVE costs, and neither
fix alone is sufficient; likely need BOTH together (merged model + fixed
draft length, simultaneously) for a fully fair reading. **That combined test
has not been run yet** — deferred to the new A6000 benchmark repo
(`a6000-bench/experiments/exp06_full_reverse_kl_pipeline.py`), which applies
merge + draft-length-fix together, at genuinely larger target scale, as the
proper "everything fixed, redone correctly" test.

**Closing state for tonight's live debugging chain on reverse-KL training**:
not resolved to a final verdict, but narrowed to a specific, well-understood,
testable next step (merge + fixed-length together) rather than an open
mystery. Two real, quantified, additive costs identified and partially (not
fully) mitigated; the genuine architecture gaps (n-gram cache, acceptance
head) remain the deeper fix if these mitigations still don't fully close it.

---

## 2026-07-23 (new session): starting the two parallel tracks — big
engineering (acceptance-prediction head) + router reward calibration

**Router reward calibration — DONE, real numbers now replace placeholders.**
Computed actual cost constants from wall-clock data already measured this
session: avg escalate (SD, trained drafter) time = 16.50s, avg small_alone
(drafter-only generation) time = 3.21s — a real ~5.14x cost ratio, not the
arbitrary 0.3/0.1 split used before. Normalized against escalate=1.0,
small_alone's real cost is ~0.1946. Updated `linucb_router.py`
(`ESCALATE_COST`, `SMALL_ALONE_COST` constants, `small_alone_reward()`
replacing `small_alone_reward_placeholder()`). **Result: router behavior
flips dramatically** — with real costs, it now picks `small_alone` 28/30
rounds (vs. escalate 28/30 with the old placeholder constants). This makes
sense: escalating really does cost ~5x more in wall-clock terms, so unless
the quality gap (correction_need) is large, staying local wins on the
economics alone. This is the FIRST time the router's arm-selection behavior
has been grounded in real measured costs rather than arbitrary constants —
still not fully validated (the F1-similarity quality signal and the
confidence context-feature are still real-but-imperfect and placeholder
respectively), but a genuine step forward, not another guess.

**Big engineering work — acceptance-prediction head, in progress.** Started
building the OmniDraft acceptance-prediction head (see the earlier "MAJOR
FINDING" entry on what's missing from our implementation). Design:
  - Small 2-layer MLP (hidden_dim -> hidden_dim//4 -> 1, sigmoid) on top of
    the drafter's own last-hidden-state at each position.
  - Label: 0 if the assistant's own top (projected-into-target-space) token
    matches the target's actual greedy argmax at that position (would be
    accepted), 1 if it doesn't (would be rejected) — the real SD acceptance
    criterion for greedy decoding.
  - Trained JOINTLY with the reverse-KL loss (same optimizer step, combined
    loss = loss_dm + 0.3 * loss_accept), reusing the exact same forward
    passes already computed in the training loop — no extra compute needed
    to get the labels.
  - Script: `uag_acceptance_head.py`. Launched on a FRESH Kaggle session
    (previous VM session ended between conversations — different container
    ID confirmed, all prior files gone, re-uploaded data + trained adapter
    from local backup before starting). Training in progress at time of
    writing.
  - **Not yet built**: the inference-time integration (using the trained
    head's predictions to control draft length at generation time). HF's
    generate() has no hook for a custom early-exit signal, so this will need
    either subclassing `AssistedCandidateGenerator.update_candidate_strategy()`
    or a custom round-by-round draft/verify loop. Deferred until the head
    itself is confirmed to train to reasonable accuracy.
  - **Progress so far (in-training)**: head accuracy climbing with noise —
    0.607 (step 0) → 0.673 (step 20) → 0.569 (step 40, noisy dip) → 0.772
    (step 60). `loss_accept` trending down (0.68→0.54). Real signal that
    hidden states carry useful predictive information for this task, not
    yet a final verdict (watch the full-run average, not individual steps).

---

## 2026-07-23: network-latency research — real depth, resolves an
important design question rather than just flagging a risk

**Context**: raised via cross-referencing another LLM conversation — every
wall-clock number measured this entire project has been same-machine
(effectively RTT≈0), and the real deployment is genuinely split
device/cloud. Researched whether this invalidates our numbers.

**Key external finding**: "Speculation at a Distance" (arxiv 2606.25091)
gives the actionable planning number — edge-cloud SD stays competitive
roughly under **RTT < 50ms**; beyond that, their recommendation is to not
split (run verifier locally, or accept pure local/pure cloud instead of a
split). Real mobile RTT (cellular 30-80ms+, WiFi 10-30ms) means this is a
genuinely tight, not-guaranteed bar for the real deployment — a legitimate
risk, not resolved by anything tested so far.

**Orthogonality question, reasoned through (not just asserted)**: is the
network-RTT question orthogonal to same-family vs. cross-vocabulary model
pairs, such that findings from same-family network papers (DSSD, PipeSD,
"Speculation at a Distance") still transfer to our cross-vocab case?
**Conclusion: yes, ORTHOGONAL, CONDITIONAL on one design commitment.** RTT
itself is a pure network property, indifferent to what's inside the packet
— the general "does verification save more than a round-trip costs"
threshold logic doesn't inherently depend on vocab compatibility. The one
place this COULD break: if a cross-vocab design needed to ship full
probability distributions (not just discrete token IDs) across the network
for the other side to do translation, that adds real transmission time
beyond pure RTT, and vocab compatibility would start to matter for network
cost too — this is exactly what DSSD's core innovation (avoiding uplink of
full vocabulary distributions) was solving. **Our design already keeps
translation+verification entirely server-side, shipping only discrete token
IDs across the boundary** — as long as we keep that commitment (a real
constraint to hold onto going forward, not an accident), payload size stays
independent of vocab compatibility and the orthogonality holds cleanly for
us. This should be treated as a hard design requirement, not an assumption.

**DSSD** (arxiv 2507.12000) — checked in depth: uses SAME-FAMILY Llama
variants only, avoids the cross-vocab problem entirely by construction
(compatible vocab from the start). Their innovation (avoiding uplink
transmission of full distributions) is specifically the mechanism that
makes the orthogonality argument above hold — direct validation that this
is a real, known lever, not something we invented.

**PipeSD** (arxiv 2605.13319) — tested on a genuine "real-world cloud-edge
testbed" (not just same-machine), achieved 1.16x-2.16x speedup, 14.3-25.3%
energy reduction — real, encouraging existence-proof that edge-cloud SD
survives real network conditions, at least for whatever pairing they used
(likely same-family, unconfirmed from available extract).

**CE-LSLM** (arxiv 2505.14085) — went deep into full HTML text, not just
abstract, per explicit instruction to actually read papers properly.
Findings:
- Also same-family under the hood (OPT-6.7B cloud + OPT-1.3B edge) —
  "heterogeneous" in their paper means different SIZE, not different
  TOKENIZER. Confirmed no discussion of vocabulary/tokenization differences
  anywhere in the paper — consistent with their mechanism (CKA/RSA
  layer-similarity-based routing + literal KV-cache-tensor reuse with
  channel-dropping for dimension mismatch) fundamentally requiring a shared
  token space to make raw tensors "line up" at all.
- Pure KV-cache/context reuse, NO speculative decoding, no draft-verify loop
  — a different mechanism than ours for the same category of problem
  (cheap conversation-state handoff, our Component C).
- Real numbers: transferring one KV-cache slice (32 layers, 80→64 dim
  reduction) costs **66.9MB I/O, ~6.69s communication latency at 10Mbps**.
  This is a genuinely useful comparison point in OUR favor: it quantifies
  how expensive literal-KV-tensor-transfer approaches are on the network,
  vs. our token-only design (shipping small integers, not large tensors) —
  a real, previously-implicit bandwidth advantage of our approach, now with
  a concrete number to point to.
- **Reconfirms, via a third independent source, that genuine
  cross-architecture (different-tokenizer) KV-cache portability remains
  unaddressed whitespace** — same conclusion reached much earlier in this
  project via the original literature survey, now triangulated a third time
  (after the 2025 edge-SLM/cloud-LLM survey and Apple's Private Cloud
  Compute statelessness-by-design choice).

**Net effect on project risk assessment**: the network-latency question is
real but NOT uniquely fatal to our approach — it's a universal problem for
any edge-cloud SD system (same-family included, as PipeSD's active
engineering work proves), solvable in principle, and our token-only design
choice already avoids the one mechanism (full-distribution transfer) that
would make vocab-compatibility matter for network cost specifically. Still
not empirically validated for OUR system under real RTT — that remains a
real gap — but the risk is now precisely scoped rather than an open worry.

---

## 2026-07-23 (same session): re-verified CE-LSLM under skepticism (held up),
and found the actual round-trip-mitigation mechanism — connects to our own
draft-length work in a way not previously considered

**Re-check of CE-LSLM, more skeptically, per explicit instruction to verify
rather than trust a single pass**: exhaustive re-read confirmed the earlier
finding, did not overturn it. Only one model pair tested (OPT-6.7B/OPT-1.3B),
an explicit passage in the paper itself states structural alignment is only
needed BECAUSE the two are different sizes within the same family ("if SLMs
share the same architecture, shallow-layer KV caches can be directly shared
... without structural alignment or layer matching") — i.e. their own method
description confirms it's a same-family-only mechanism. No cross-family
experiments, no tokenizer/vocab discussion anywhere, no citation of
Cache2Cache or DroidSpeak. Confirmed via a genuinely adversarial second pass,
not just re-asserting the first read.

**How DSD (arxiv 2511.21669) actually solves the multiple-round-trip
problem** — this was the real gap in the earlier summary (mentioned
"pipelining" without explaining the mechanism):
1. **Pipelining**: device never sits idle waiting for a verification result
   — while batch N is in flight/being verified, device immediately starts
   drafting batch N+1 speculatively. Overlaps network wait with useful
   compute instead of pure dead time.
2. **Batching draft tokens per round trip**: accumulate K tokens before
   sending one verification request, instead of paying full RTT per single
   token. Amortizes the FIXED round-trip cost across K tokens — cuts
   round-trips-per-token from ~1 down to ~1/K.
3. **Speculative execution ahead of verification**: device keeps drafting
   before the previous batch's result returns, betting most gets accepted
   (standard SD rollback cost if a rejection lands mid-batch).

DSD-Sim's key finding: speedup is highly sensitive to RTT, and pipelining's
benefit gets MORE critical (not less) as RTT increases — without it, rising
latency is catastrophic; with it, degradation is graceful.

**Direct connection to our own draft-length work, not previously
considered**: on a same-machine setup (everything tested this entire
session), draft/batch length only trades off against COMPUTE cost (longer
verification window = marginally more compute per round, but fewer rounds
needed — this is the whole adaptive-heuristic-overshoot problem debugged
earlier tonight). In a REAL NETWORKED setup, draft length ALSO amortizes the
FIXED RTT tax over more tokens — a second, independent benefit absent from
same-machine testing. **Implication: our "fixed draft length=4" experiments
were implicitly optimized for a zero-RTT world; the real deployment's
optimal draft length is likely LARGER, specifically to amortize network
cost, not just compute cost.** This should inform the acceptance-prediction
head's design once it's built for real deployment — its early-exit
threshold should probably account for network cost too, not just compute
cost, which is all our same-machine testing has ever been able to measure.
Not yet incorporated into the acceptance-head training in progress (which is
still same-machine, correctly scoped to just the compute-side question for
now) — flagged as a needed extension once network-aware testing begins.

---

## RESULT (2026-07-23): acceptance-prediction head — first training run
complete, real signal confirmed, not yet strong

`uag_acceptance_head.py` training finished cleanly: **0/105 skipped (no
NaN issues, unlike the first reverse-KL attempt), final average head
accuracy: 0.6576** across the whole run. Per-checkpoint trajectory: 0.607
(step 0) → 0.673 (step 20) → 0.569 (step 40, noisy dip) → 0.772 (step 60) →
0.673 (step 80) → 0.690 (step 100) — noisy but real, meaningfully above the
0.5 chance baseline throughout after the earliest steps.

**Honest read**: the drafter's hidden states DO carry learnable signal about
accept/reject, confirming the design's core premise — but 66% accuracy is
far from a strong classifier (a coin flip is 50%, so this is real but
modest). Not yet validated whether this is good enough to be USEFUL as an
early-exit signal (a weak-but-better-than-random predictor could still help
if it correctly identifies the clearest over-confident/under-confident
cases, even if it's noisy on the ambiguous middle) — that can only be
checked once the inference-time integration exists and gets compared
against the fixed/adaptive draft-length baselines already measured.

**Artifacts saved on VM**: `trained_drafter_with_acceptance_head/` (LoRA
adapter, trained jointly with the acceptance loss) and `acceptance_head.pt`
(the small MLP's weights) — not yet downloaded to local backup, do that
before this Kaggle session ends (same lesson as every previous session).

**Next step, not yet built**: the inference-time integration. HF's
`generate()` has no hook for a custom early-exit signal, so this needs
either subclassing `AssistedCandidateGenerator.update_candidate_strategy()`
to consult the trained head instead of the generic +2/-1 heuristic, or a
custom round-by-round draft/verify loop written from scratch. Once built,
the real test is comparing head-controlled draft length against both the
adaptive heuristic (1.042x/1.208x from earlier) and the fixed-length
baseline (1.210x) on the same 30-prompt wall-clock comparison — only then
do we know if this modest-but-real 66% accuracy actually translates into a
practical improvement.

**Inference integration: built.** `uag_acceptance_head_inference.py` —
subclassed `AssistedCandidateGeneratorDifferentTokenizers`, overriding ONLY
`update_candidate_strategy()` (everything else — translation, KV cache
management, verification — stays HF's tested code unchanged). Injected via
monkey-patching `target_model._get_candidate_generator` (a regular instance
method, no public kwarg exists for a custom candidate generator). Mechanism:
after each round, one cheap extra forward pass on the assistant's current
context (reusing `prev_assistant_ids`, already maintained by the parent
class) with `output_hidden_states=True`, feed last hidden state through the
trained head to get P(reject), grow/shrink `num_assistant_tokens` (same
+2/-1 shape as the original heuristic, just gated on the head's prediction
instead of raw match count). Test running against the same 30-prompt set;
result pending.

---

## 2026-07-23: C2C codebase deep-dive (real source, not just README) —
confirms it's a substantially more sophisticated mechanism than CE-LSLM

Read `rosetta/model/projector.py` and `rosetta/model/wrapper.py` directly
from [github.com/thu-nics/C2C](https://github.com/thu-nics/C2C) (ICLR'26,
official code).

**The actual fusion mechanism** (`AllInOneProjector.forward`): per-layer
MLP projections (with optional SwiGLU/residual/layer-norm) map source
K/V into target's dimension space, THEN a **learned gate** (Gumbel-sigmoid,
temperature-annealed during training — a soft/discrete "should this
position even use the source's info" decision) and a **learned blending
weight** (both configurable at scalar/token/head/value granularity, and
optionally INPUT-DEPENDENT — computed from live activations, not fixed)
combine as: `output = (1−weight)·target + gate·weight·projected(source)`.
Both models still run their own forward pass to produce native KV caches;
the projector only combines them post-hoc, per layer. This is genuinely
learned, fine-grained, adaptive fusion — not CE-LSLM's fixed
similarity-threshold binary layer-selection.

**Training replication assessment**: LOWER difficulty than expected.
`python script/train/SFT_train.py --config recipe/train_recipe/C2C_0.6+0.5.json`
(single command, single or multi-GPU), and **only the projector trains —
both base/teacher models stay frozen** (cheaper than our own LoRA reverse-KL
runs, which update real model weights). A ready-made config exists for
Qwen3-0.6B+Qwen2.5-0.5B-Instruct. **Pre-trained checkpoints already released
for 7 pairs**, including that exact one — we could test their released
fuser directly with zero training, on our own hardware/prompts, before
deciding whether to extend anything ourselves.

**Where the real gap is for OUR use case**: none of the 7 released pairs
approach our scale (Gemma-2-9b target) or involve a genuinely different
target family at all (their biggest, Qwen3-8B+Qwen2.5-7B, is still
same-family). The framework's own docs claim automatic handling of
different hidden dims/layer counts/attention configs/tokenizers for
arbitrary pairs — real claim, but unverified by us on anything as
divergent as Qwen-drafter + Gemma-target. Extending would mean writing a
new training config (documented, bounded pattern) and empirically testing
whether that heterogeneity claim holds on a harder pair than anything
they've published.

**Clean categorization of same-family-vs-cross-family, trained-vs-not,
resolving the "is CE-LSLM legit" and "no-training same-family sharing"
questions together**:
| | Same-family | Cross-family |
|---|---|---|
| No training | DroidSpeak, CE-LSLM (similarity-metric layer selection) | not found — makes sense, no reason raw tensors reconcile without a learned mapping |
| Trained | (less needed, no-training already works) | C2C (learned per-layer fusion) — the genuinely harder, more validated case |

**Correction, worth flagging**: CE-LSLM's own test pair (OPT-6.7B/OPT-1.3B)
is very likely ALSO independently pretrained at each scale (standard
practice for the OPT release), not derived from one another — meaning it
probably fails DroidSpeak's own stricter "same foundation model / shared
weight lineage" requirement too, while using an even WEAKER justification
(computed CKA/RSA similarity, not verified lineage) to license direct
tensor reuse. Reinforces the earlier skepticism about CE-LSLM specifically.

**CE-LSLM's actual "6G" contribution, precisely stated**: not a technically
6G-specific mechanism — generic compute/communication overlap (layer-wise
pipelining: edge computes shallow layers while deeper layers' cache is
still loading from cloud in the background; per-layer time =
`max(load time, previous layer's compute time)`). Same category of
technique as DSD's round-trip pipelining discussed earlier, just applied to
KV-cache loading instead of draft-token batches. Would work identically
over 5G/WiFi/any sufficiently fast link — "6G" is motivating framing for
publication novelty, not a technical dependency.

---

## RESULT (2026-07-23): acceptance-head inference integration — real,
honest negative, doesn't beat the simpler fixes

Full 30-prompt run of `uag_acceptance_head_inference.py` (merged trained
drafter, acceptance-head-controlled draft length via the monkey-patched
`AcceptanceHeadCandidateGenerator`):

| | Adaptive heuristic | Fixed length=4 | Acceptance-head |
|---|---|---|---|
| Untrained | 1.295-1.313x | 1.296x | — |
| Trained (unmerged) | 1.042x | 1.210x | — |
| Trained (merged) | 1.208x | — | **1.159x, median 1.167x, tok/call 2.468, 5/30 below 1.0x** |

**Does not beat either simpler fix** (merging alone: 1.208x; fixed-length
alone: 1.210x), though it does beat the worst case (unmerged+heuristic,
1.042x). Notably, its tok/call (2.468) is the HIGHEST of any condition
tested — the head genuinely improves draft-length decisions — but the EXTRA
forward pass added every round (to compute the hidden state the head needs)
costs more than that improvement is worth. Real, honest result: the
underlying prediction has value, this specific integration design doesn't
pay for itself. **Likely fix, not yet attempted**: capture hidden states
DURING the assistant's existing draft-generation call (would need deeper
surgery into `get_candidates()`, not just `update_candidate_strategy()`)
rather than a separate bolted-on forward pass afterward — bigger, more
invasive change than tonight's version.

**Standing verdict on the OmniDraft replication overall**: none of tonight's
three interventions (merge, fixed-length, acceptance-head) alone fully
recovers untrained's baseline speedup for the trained drafter. Best result
so far for trained is 1.210x (unmerged+fixed-length) or 1.208x
(merged+adaptive) — both still below untrained's ~1.3x. The real
acceptance-head mechanism (OmniDraft's actual design, hidden states captured
for free during drafting, no extra forward pass) remains untested — what's
been tested tonight is a reasonable but imperfect approximation given HF's
API constraints, not the real thing.

---

## RESULT (2026-07-23, later session): ratio-fix experiment (Llama-68M,
~132:1 ratio) — training the drafter is WORSE than untrained on wall-clock,
even at the correct ratio

Full 6-step run (`uag_ratio_fix_full.py`): baseline → untrained SD (adaptive)
→ fresh reverse-KL LoRA training (r=32, q/k/v/o, bug-fixed -inf mask) →
merge → trained+adaptive → trained+fixed(K=4), all on Llama-68M (drafter) +
google/gemma-2-9b (target), 30 held-out LoCoMo prompts each.

| condition | avg speedup | avg tok/call |
|---|---|---|
| untrained (adaptive) | **1.522x** | 1.854 |
| trained, merged (adaptive) | 1.346x | 1.878 |
| trained, merged (fixed K=4) | 1.411x | 1.891 |

Fixing the ratio to OmniDraft's own validated regime (~132:1 vs our earlier
~15:1 with Qwen3-0.6B) does NOT resolve the wall-clock regression. Trained
is worse than untrained in both variants, despite HIGHER tok/call in both —
confirms the ratio hypothesis was insufficient on its own.

## DIAGNOSIS: the actual mechanism — avg_K (proposed draft length) inflates
much faster than tok/call (accepted), so training costs more compute than
it saves

Root-caused via `uag_diagnose_wallclock.py` — reused the already-saved
adapter (no retraining), added two things the original run never measured:
(A) real average draft length K (tokens PROPOSED per round, via monkeypatching
`AssistedCandidateGeneratorDifferentTokenizers.get_candidates`), and (B) a
freshly re-measured local baseline at each condition's own time-point
(instead of reusing one baseline captured only at the very start of a long
script, in case of GPU-state drift).

| condition | avg_time | avg tok/call | **avg_K (proposed)** | speedup(local baseline) |
|---|---|---|---|---|
| untrained | 10.00s | 1.854 | **1.83** | 1.607x |
| trained (adaptive) | 10.61s | 1.878 | **3.38** | 1.557x |
| trained (fixed K=4) | 10.35s | 1.891 | **2.55** | 1.577x |

Per-token acceptance rate (accepted/proposed): untrained ≈**100%**, trained-
adaptive ≈**56%**, trained-fixed ≈**74%**. **This is the actual mechanism**:
reverse-KL training didn't make the drafter's proposals better, it made the
model more confident (reverse-KL is mode-seeking/peaky), which drives the
adaptive-length heuristic to commit to much longer draft chains per round —
nearly 2x more proposed tokens — while barely improving how many of those
proposals actually get accepted (+1-2% tok/call). Drafting cost scales with
tokens PROPOSED (autoregressive small-model passes), not tokens accepted, so
total compute goes up even though tok/call — which doesn't distinguish "2
accepted out of 2 proposed" from "2 accepted out of 4 proposed" — looks
better. Baseline drift (B) was real (~6-10%, fresh local baselines 16.08-
16.53s vs. the original run's single early baseline of 15.09s) but did NOT
change the ranking — untrained still wins after fully correcting for it.
**Conclusion: nothing was wrong in the harness. This is a genuine property
of reverse-KL-only distillation, and it's exactly why OmniDraft's acceptance
head and n-gram cache aren't incidental extras — they're the necessary gate
against this specific overconfidence failure mode.**

## IN PROGRESS, INTERRUPTED — accept-head v2 on the correct ratio (Llama-68M)

Wrote `uag_acceptance_head_llama68m_full.py`: joint reverse-KL + acceptance-
head training (BCE loss, λ=0.3) on Llama-68M/Gemma-2-9b (the correct ratio,
unlike the earlier Qwen3-0.6B accept-head attempt), then v2 gated inference
(hidden states captured for FREE during the assistant's own drafting call —
fixes v1's flaw of a bolted-on extra forward pass) with K-logging, to test
directly whether the head reins avg_K back down toward the ~1.8-2.5 range
where it pays off. **Launched successfully (PID 3018), but the Cloudflare
tunnel died almost immediately after launch ("websocket: bad handshake"),
and the VM was subsequently stopped completely** — training progress, if
any, was lost (script only saves the adapter + head weights at the END of
the training loop). Script itself is preserved locally at
`/Users/anirudh/Anirudh/research/hybrid/vm-artifacts/` (needs copying in —
was only in the session scratchpad) and should be rerun as-is on the next
VM; no design changes needed, just re-launch.

**Also lost with this VM** (never downloaded locally before it stopped):
`ratio_fix_results.json`, `diagnose_wallclock_results.json`, the
`trained_llama68m_reverse_kl_gemma2_9b` LoRA adapter. None of this is
expensive to regenerate (each run was well under an hour), and the actual
numbers/conclusions above are already captured in this file — only the raw
checkpoints/JSON are gone, not the findings.

## NEXT AGENDA (set 2026-07-23): stop polishing acceleration, prototype
Component C

Standing assessment: the project's genuinely novel claim is using cross-
vocab SD as BOTH the accelerator AND the structurally-free handoff mechanism
between models — none of the papers reviewed this session (OmniDraft, DSD,
PipeSD, DSSD, "Speculation at a Distance", CE-LSLM, DroidSpeak, C2C) combine
those two into one mechanism; they all solve handoff and acceleration
separately. That claim (Component C) has never been prototyped, even
roughly. Meanwhile the acceleration half just got harder to trust (see
above — untrained still beats every trained variant tested so far on wall-
clock). Agreed next step: stop spending more GPU-hours calibrating the
accept-head/n-gram cache until Component C's core claim is checked at all —
build the cheapest possible toy version (two-turn conversation, small model
handles turn 1, cross-vocab SD runs, large model picks up turn 2 using only
what SD naturally exposes — accepted tokens + assistant KV cache — check
qualitatively whether continuity holds), before over-investing further in
the acceleration side.

---

## RESULT (2026-07-23/24): acceptance-head v2, correct ratio (Llama-68M) —
completed on the 3rd VM after two environment blockers; PARTIAL fix, not a
full one — closes out the acceleration-side investigation per the agreed
pivot

Environment notes for next time: this VM ran **transformers 5.0.0** by
default, which ships a rewritten weight-loading path
(`core_model_loading.py`, thread-pool `materialize_tensors`) that OOM'd
loading Gemma-2-9b in 8-bit on a 14.56GB T4 — reproduced identically on TWO
separate clean/fresh boxes, `expandable_segments` did NOT help (confirms
genuine peak-memory usage, not fragmentation). Fixed by pinning
`transformers<5` (landed on 4.57.6), which also reverts `get_candidates`'s
signature back to `(self, input_ids)` (5.0 added a required
`is_first_iteration` positional arg — breaks any code overriding/patching
`get_candidates`, e.g. our v2 gating class and the diagnostic's K-logging
patch; both are written for the **4.x, no `is_first_iteration`** signature
in `vm-artifacts/scripts/`, matching what's needed after pinning `<5`).
Separately, this box's `peft` errored on `get_peft_model` with
`Found an incompatible version of torchao. Found version 0.10.0, but only
versions above 0.16.0 are supported` — a peft footgun where a present-but-
too-old optional dep raises instead of being treated as absent; fixed by
`pip uninstall -y torchao` (we don't use it) rather than upgrading it.
**Recommended setup order for next VM**: `pip install "transformers<5"`
BEFORE anything else, `pip uninstall -y torchao` if present, then
`pip install bitsandbytes`.

**Final result** (joint reverse-KL + acceptance-head training, Llama-68M /
Gemma-2-9b, avg head accuracy 0.6506 — close to the earlier Qwen3-0.6B run's
0.6576), v2 gated inference (hidden states captured free during the
assistant's own drafting call, head's predicted reject-prob directly
controls `num_assistant_tokens` instead of HF's heuristic):

| condition | avg speedup | avg tok/call | avg_K (proposed) |
|---|---|---|---|
| untrained (adaptive) | 1.51-1.61x | 1.854 | 1.83 |
| trained, no gating (adaptive) | 1.42-1.56x | 1.878 | 3.38 |
| trained, no gating (fixed K=4) | 1.46-1.58x | 1.891 | 2.55 |
| **trained, accept-head gated (v2)** | **1.505x** | **1.880** | **3.066** |

**Verdict: partial fix, not a full one.** The head pulled avg_K down from
the unguided-adaptive's 3.38 toward 3.066, and its speedup (1.505x) clearly
beats unguided-adaptive (1.42-1.56x) and is roughly on par with fixed-K=4
(1.46-1.58x) — real, measurable improvement over both no-gating variants.
But it did NOT bring K down anywhere near untrained's 1.83, and its speedup
lands at/just below the low end of untrained's own range (1.51-1.61x) — not
a clear win over simply using the untrained drafter, more like recovering
to rough parity at a much higher engineering cost (joint training + gating
logic vs. nothing). Plausible explanation: a head that's right only ~65% of
the time can only partially correct a heuristic/model combination this
overconfident.

**This closes out the acceleration-side investigation for now**, per the
2026-07-23 pivot decision above — n-gram cache and further accept-head
tuning are parked; next session's priority is the Component C toy prototype.

Artifacts saved locally: `vm-artifacts/trained_llama68m_with_acceptance_head/`,
`vm-artifacts/acceptance_head_llama68m.pt`,
`vm-artifacts/results_2026-07-23/acceptance_head_v2_llama68m_results.json`.

---

## RESULT (2026-07-24): Component C toy prototype — quality equivalence +
real speedup established; warm-cache and pipelining explored, both come up
empty at this scale; multi-turn and big->small direction untested

**Setup for all of this**: LoCoMo conversation split at a fraction
`split_frac` into history_A / history_B. Small model (Llama-68M) generates
its own continuation for history_A (stands in for "small model actually
handled this part on-device"), spliced back in, followed by the real
history_B lines -> `handoff_prompt`/`full_text`. This is ONE simulated
handoff per example, not a real multi-turn conversation with repeated
escalate/de-escalate decisions -- that distinction matters, see caveats below.

**1. Quality test** (`component_c_toy_test.py`, 8 examples): SD-verified
handoff vs plain-target-continuation vs small-alone, scored against real
LoCoMo ground truth via word-overlap F1. Result: small-alone 0.051,
target-plain 0.182, SD-handoff 0.186 -- SD and plain-text are statistically
indistinguishable. **This is expected, not novel**: under greedy decoding,
SD's accepted-token sequence is mathematically guaranteed to equal plain
target decoding on the same input (a proposed token is accepted iff it
matches the target's own argmax; otherwise the target's own token is
substituted). The test confirms nothing is broken, it doesn't test anything
SD uniquely contributes over just handing the target plain text (which was
never a hard problem -- any model can read plain text with zero special
machinery).

**2. Wall-clock grid** (`component_c_wallclock_grid.py`, 3 split_fracs x 2
hygiene settings x 8 examples): real wall-clock, plain-target-continuation
vs SD-handoff, same `handoff_prompt` each time. SD-handoff beat plain in
**every single cell**, 1.43x-1.56x, split=0.5 best, hygiene (repetition
penalty on the small model's degenerate standalone text) very slightly
*lowers* speedup, most visibly at split=0.75. avg_K/tok-call stayed in the
untrained-drafter range (no K-inflation, since this is the untrained
Llama-68M). **This is the real, solid Component C finding**: SD-verified
handoff is measurably faster than plain-text handoff at a single cold-start
handoff point, using nothing but stock cross-vocab SD machinery.

**3. Warm/cold KV cache** (`warm_cold_kv_test.py`) -- three broken attempts,
then a real fix, then a small but real answer:
- Attempt 1 (bare `model(input_ids=..., use_cache=True)` to build cache,
  full sequence into `generate()`): match=False, ~1.0x speedup, on Gemma-2-9b.
- Attempt 2 (only-new-tokens + manual `cache_position`): same divergence,
  byte-identical wrong output to attempt 1 -- proved the bug wasn't in how
  resumption was called.
- Model swap to Qwen2.5-7B-Instruct (ruling out Gemma-2's hybrid/sliding-
  window attention cache as the cause): **same divergence persisted**,
  confirming it was a general implementation bug, not Gemma-2-specific.
- **Actual fix** (found via HF's official docs, not guessing):
  `huggingface.co/docs/transformers/en/kv_cache`'s documented pattern is (a)
  explicitly instantiate `DynamicCache(config=model.config)` and pass it
  INTO the forward call building the cache (not just `use_cache=True` and
  take whatever default gets created), and (b) pass the FULL current
  input_ids into `generate()` every time, never a manually-sliced suffix,
  never a manual `cache_position`. Both of attempts 1-2 violated this.
  Result after the fix: 3/8 exact match, remaining 5/8 small paraphrase-level
  differences (not the wildly-different content from before) -- consistent
  with minor fp/quantization precision differences between two separate
  calls, not a structural bug anymore.
- **The timing answer, which held steady across ALL FOUR attempts regardless
  of correctness**: avg speedup ~1.01-1.05x. Mechanism: prefill over the
  ~250-400 token shared context here is cheap and parallelizes across the
  whole sequence in one GPU pass; the fixed 40-token autoregressive decode
  is strictly sequential and dominates wall-clock cost regardless of whether
  the prefix was cached. **Conclusion: at this conversation scale, warm-cache
  reuse isn't a meaningful lever** (~2-5%, not the free win a "structurally
  free handoff" framing might suggest). Untested but mechanistically
  plausible: this should invert at much longer context (many thousands of
  tokens, e.g. "10 conversations" worth of history) where prefill cost would
  start to dominate over a fixed-length decode -- not measured this session.

**4. Pipelining/concurrency** (`component_c_pipeline_test.py`) -- directly
testing DSD/PipeSD's "speculative execution ahead of verification" idea:
does running the target's cache-build for history_A (which is fully known
before turn-A generation even starts) CONCURRENTLY with the small model's
turn-A generation, on separate GPUs via Python threading, beat doing them
strictly sequentially? Order alternated per example, warm-up call added, to
avoid the same positional-bias risk flagged in the wall-clock grid test.
First run: 100% `used_cache=False` (a reintroduced tokenization-boundary
mismatch -- the bare-forward-cache-building step tokenized `history_A`
standalone while the continuation tokenized `full_text` standalone;
diagnostic logging showed EVERY mismatch was exactly the last token of
history_A, consistently -- LoCoMo splits tend to land right after
turn-ending punctuation, e.g. "?", which BPE-merges differently depending on
what follows). Fixed by caching `history_A` minus its last token (costs
nothing, guarantees exact match by construction). Result after the fix:
**used_cache=True in all 8, but avg speedup 0.984x -- essentially a wash,
slightly negative**. The small compute savings from caching (already only
2-5%, see above) get fully absorbed by Python threading/synchronization
overhead at this scale. Not a bug this time -- a valid, honest negative
result.

**Standing gaps, explicitly not addressed this session**:
- Only small->big handoff tested. Big->small handback: never tested at all.
- Only ONE simulated handoff per example. Real multi-turn conversations with
  repeated escalate/de-escalate decisions over many turns: never tested.
- Warm-cache/pipelining benefit at realistic (long) conversation-history
  scale: never tested, plausible but unmeasured.

**Methodology notes worth remembering** (caught mid-session, not by design):
"speedup" was used loosely across three genuinely different comparisons this
session (SD vs no-SD, SD-handoff vs plain-handoff, warm vs cold plain
generation) -- these are not comparable numbers despite the shared label.
The wall-clock grid test also has an uncontrolled ordering risk (plain
always measured immediately before SD, every single time, for all 48
measurements) that was never checked for a positional warm-up bias, unlike
the pipelining test which explicitly alternated order after this was flagged.

**Overall verdict going into next session**: the core Component C claim
(quality-equivalent, measurably faster SD-verified handoff vs plain-text
handoff, at a single cold handoff point) is solid. Everything explored past
that -- warm caching, concurrency/pipelining -- comes up empty at the scale
tested, with real, mechanistically-understood reasons why, not because the
ideas are wrong but because this session's toy scale (a few hundred tokens
of context) is the wrong regime to see them pay off. Multi-turn dynamics and
bidirectional switching remain completely untested.

---

## RESULT (2026-07-24): prefill-scaling profiler -- resolves WHY warm-cache
showed ~0% benefit tonight, and confirms it should be a real win at
realistic conversation length

Isolated, clean measurement (`prefill_scaling_test.py`): bare forward pass
(`use_cache=True`, no decode, no generation, no threading), `torch.cuda.
synchronize()` around timing, Qwen2.5-7B-Instruct, 5 reps per length, sweep
50 to 2000 tokens:

| tokens | avg time | ms per 100 tokens |
|---|---|---|
| 50 | 204.6ms | 409.1 |
| 100 | 214.5ms | 214.5 |
| 250 | 264.1ms | 105.6 |
| 500 | 428.7ms | 85.7 |
| 1000 | 886.6ms | 88.7 |
| 1500 | 1524.4ms | 101.6 |
| 2000 | 2288.2ms | 114.4 |

**Two clean regimes.** 50-250 tokens: nearly flat -- doubling tokens (50->100)
only costs +5% time, 5x the tokens (50->250) only costs +29%. A fixed floor
of ~180-200ms dominates. This is EXACTLY the range every Component C test
tonight operated in (50-400 new tokens) -- which is why warm-caching and
pipelining both measured ~0% benefit: the tests were entirely inside the
flat part of this curve, where prefilling 50 vs 400 tokens genuinely costs
almost the same. Past ~500 tokens: roughly linear to slightly superlinear
(500->1000, 2x tokens, +107% time; 1000->2000, 2x tokens, +158% time) --
real per-token cost, no longer overhead-dominated.

**Concrete implication**: at a 2000-token conversation history, prefilling
everything from scratch costs ~2288ms; prefilling just 100 new tokens
(reusing a cache for the other 2000) costs ~214ms -- **over 10x faster**,
not the ~2-5% measured tonight at small scale. So the warm-cache idea isn't
wrong, and the earlier "not a meaningful lever" conclusion was scale-
specific, not general: **warm-cache reuse should be a real, substantial win
at realistic multi-turn conversation lengths (1000+ tokens of accumulated
history); it just doesn't matter yet at the few-hundred-token scale every
test tonight happened to use.** Not yet measured end-to-end (prefill +
decode combined, real cold-vs-warm comparison) at this longer scale --
extrapolated from the isolated prefill numbers above, not directly tested.

**Next planned step**: re-run the warm/cold comparison (using the now-
correctly-implemented `DynamicCache`-based mechanism from the earlier fix)
at realistic 1000-2000 token context length, sequential only (no threading
-- explicitly dropped per instruction, since synchronization overhead
dominated at small scale and would only get relatively cheaper, not more
necessary, as the underlying savings grow). This gets the real end-to-end
number instead of an extrapolation from isolated prefill timing.

---

## RESULT (2026-07-24): realistic-scale warm/cold, done properly -- real
~30% end-to-end speedup, confirms the prefill-scaling prediction

`warm_cold_realistic_scale.py`: ~1800 token accumulated history (built from
concatenated real LoCoMo text) + a short new-turn follow-up, sequential
only, 8 examples, Qwen2.5-7B-Instruct, same correctly-implemented
`DynamicCache` mechanism plus the last-token-drop boundary fix from the
pipelining test.

**First run had a real bug, caught before drawing any conclusion from it**:
`t0 = time.time()` was placed BEFORE the cache-building forward pass instead
of after, so "warm" was accidentally timing cache-build + generate() summed
together, instead of just the incremental continuation (the cache-build cost
is supposed to represent work already paid for earlier, same as the small-
scale test's correct design). That bug produced a backwards-looking result
(warm slower than cold, 0.952x) that would have been very easy to mistake
for a real, surprising finding if not caught. Fixed by moving `t0` to after
cache construction and removing a redundant duplicate cache-build.

**Corrected result**: avg speedup **1.307x** (1.261x-1.354x across all 8
examples, tight and consistent, no outliers). This is a real, meaningful,
repeatable win, not the ~2-5% noise-level effect from the small-scale test.
The numbers check out arithmetically against the isolated prefill-scaling
profiler too: cold (~1900ms interpolated prefill at 1800 tokens + ~7300ms
decode ~= 9200ms) matches the measured 9221ms average; warm (~200ms floor
cost for the ~15-20 new tokens + ~7300ms decode ~= 7050-7500ms) matches the
measured 7054ms average almost exactly. Two independent measurements (an
isolated prefill sweep, and this real end-to-end test) agree with each
other, which is real evidence, not coincidence.

**This closes the loop on the whole warm-cache question, cleanly**: the
mechanism works correctly (validated via HF's documented pattern), the
~2-5% small-scale result was real but scale-specific, and at realistic
conversation-history length it delivers a genuine ~30% speedup. The overall
arc of this investigation -- three broken cache-resumption attempts, a
tokenization-boundary bug, an ordering-bias catch, a mislabeled-speedup
audit, a threading-overhead negative result, an isolated-profiler pivot, and
finally a timing-bug catch on the very last run before accepting the
result -- is itself a good demonstration of why verifying every surprising
number before believing it mattered every single time tonight, not just
some of the time.

---

## RESULT (2026-07-24): pipelining retested at realistic scale -- still a
wash, for a DIFFERENT and more fundamental reason than at small scale

Prediction going in: now that warm-caching alone shows ~30% (~2000ms) at
1800-token scale, concurrency's overhead calculus should flip -- the
target's cache-build (~2000ms) is comparable to or longer than the small
model's own generation time, so there should be real overlap to exploit,
unlike the small-scale test where the small model finished almost instantly
and there was nothing to hide behind it.

**Prediction was wrong.** `pipeline_realistic_scale.py` (same alternating-
order, warm-up-call methodology as before, ~1800 token history, 8 examples):
avg sequential 9.276s, avg concurrent 9.322s, avg speedup **0.995x** --
essentially noise (0.976x-1.010x range), no clear direction.

**Why, precisely**: sequential's measured time (9.276s) is nearly identical
to the pure cold-target-alone time from the warm/cold test (9.221s, no small
model involved at all) -- meaning the small model's own contribution to
total sequential time is close to zero. Llama-68M is so much smaller/faster
than the 7B target that generating 40 tokens over 1800 tokens of context
barely registers, regardless of how much context there is. This is a
DIFFERENT failure mode than the small-scale result: that one was "the
savings were tiny and threading overhead ate them"; this one is "there was
never meaningful dead time to hide the target's cache-build behind in the
first place, because the two GPUs are doing wildly imbalanced amounts of
work." Pipelining only pays off when both overlapped tasks take comparable,
substantial time -- a tiny drafter paired with a large target structurally
doesn't have that property, at any context length tested.

**Conclusion: pipelining/concurrency between drafter and target, as
implemented here, does not help** -- not at small scale (overhead-dominated)
and not at realistic scale (nothing to overlap against). This isn't a
methodology gap to fix, it's a structural property of this specific
architecture (tiny drafter, large target). Closing this out; no further
pipelining variants planned unless the drafter/target size ratio changes
significantly (e.g. under the parked quantized-bigger-drafter idea, where a
slower, more capable drafter might actually take long enough to be worth
overlapping against).

---

## RESULT (2026-07-24): first multi-turn, bidirectional Component C test --
mechanism holds up on timing, but real conversational coherence collapse
found, more important than any speedup number

`component_c_multiturn.py`: 4 seeds, each a real growing 5-segment
transcript -- seed dialogue -> small model's own turn -> ESCALATE #1
(small->big, SD) -> DE-ESCALATE (big->small, first time this direction has
ever been tested) -> ESCALATE #2 (small->big, SD again). Plain-generation
baseline added at both escalation points per explicit request, order
alternated per seed to control the positional-bias risk flagged earlier.

**Quantitative (timing) result -- clean and consistent**: escalation #1 avg
speedup 1.217x, escalation #2 (after a full de-escalation cycle, on a grown
multi-author transcript) avg speedup 1.191x -- nearly identical. Whatever
else happens, the SD-handoff speedup doesn't degrade across repeated,
bidirectional switching. avg tok/call also nearly identical (1.526 vs
1.491). This is a real, if modest, positive: the mechanism itself is robust
to repeated switching, at least over 2 escalations.

**Qualitative (coherence) result -- the actually important finding, and a
real problem**: all 4 transcripts show genuine coherence breakdown, with a
consistent, recognizable failure pattern across 3 of 4 seeds -- the TARGET
model (Qwen2.5-7B-Instruct) repeatedly slips into "helpful assistant" mode
instead of staying in character: asking "What were the main topics discussed
in the conversation between Deborah and Jolene?" and listing them (seed 2);
drifting into an unrelated AI/machine-interaction tangent then flagging "a
slight misinterpretation or typo in the last sentence" (seed 0); drifting
into "the following table shows how many people were affected by the
incident" then catching itself with "there's been a bit of a mix-up between
different conversations here" (seed 1). This is not random noise -- it's the
same failure mode recurring: an instruction-tuned model given raw,
un-templated concatenated dialogue text (no explicit "continue this
conversation as [speaker]" framing) defaults to assistant behavior
(summarizing, flagging typos, asking clarifying questions) instead of
staying in character. Separately, the SMALL model (Llama-68M) independently
drifts off-topic too, often on its very first, unescalated turn (seed 1
jumps to "buy a new car or truck," seed 3 jumps to "the 2018-2020 season" --
neither connected to the actual conversation at all) -- the same underlying
weakness flagged earlier tonight for repetition, here showing up as topic
drift instead.

**Important, and worth being precise about: this is NOT an SD-specific
problem.** SD's output is provably identical to plain generation under
greedy decoding (established earlier this session) -- plain-text handoff
would show the exact same coherence collapse, confirmed empirically here
since the plain-generation baseline was measured at every escalation point
and the SD output is what's shown in the transcripts (they're
indistinguishable by construction). This is a prompting/formatting problem
(raw concatenated text vs. proper chat-template role framing) and a small-
model-capability problem, not something SD caused or could fix.

**Implication**: the earlier single-handoff "quality equivalence" result
(SD-handoff F1 vs ground truth ~= plain-handoff F1) was real but didn't
capture this -- it only measured ONE continuation step. Multi-turn,
multi-switch conversations compound whatever coherence weaknesses exist at
each individual step, and with 3 of 4 seeds visibly derailing, that
compounding is a real, serious problem for actual multi-turn viability, not
a minor caveat. Likely fix, not yet attempted: proper chat-template/role
framing (explicit speaker turns via the tokenizer's chat template) instead
of raw string concatenation, which should give the instruction-tuned target
much clearer signal to stay in character rather than defaulting to assistant
mode -- worth testing directly before drawing further conclusions about
multi-turn viability.

---

## RESULT (2026-07-24): proper Q&A-structured multi-turn test -- confirms
the fix, and cleanly isolates a SECOND, more fundamental problem

Root cause of the earlier coherence collapse, sharpened by the user
mid-session: the previous test used raw two-person dialogue continuation
(LoCoMo-style), not the real shape of the problem -- the real system is
user asks a real question, router picks a model, that model answers.
`component_c_qa_multiturn.py` rebuilds the test around that: 3 hand-
constructed topics (learning to cook, home gardening, training for a 5k),
4 topically-related follow-up questions each, same escalation pattern
(Q1->small, Q2->big/SD ESCALATE, Q3->small DE-ESCALATE, Q4->big/SD ESCALATE
again). Formatting deliberately asymmetric: target gets a proper
`apply_chat_template()` prompt (it's instruction-tuned, this is what it
actually expects); drafter gets simple "Q: ... A: ..." plain text (it's a
base model, a chat template would mean nothing to it). Plain-generation
baseline kept at both escalation points, order alternated per topic.

**Big model coherence: completely fixed.** All 6 escalation answers (2 per
topic x 3 topics) are correct, on-topic, properly reference prior context,
zero meta-confusion, zero "let me summarize this conversation" derailment --
e.g. Q2 on knife skills gets a real structured answer about safety and
practice timelines; Q4 on 5k race-readiness gets a real answer referencing
fitness level and training consistency. This fully confirms the diagnosis:
the earlier collapse was a prompt-framing problem (raw dialogue vs. proper
chat template), not anything deeper in the mechanism, and it's now resolved.

**Small model coherence: still broken, but now cleanly isolated as a
DIFFERENT, more fundamental problem.** Even with ideal Q&A framing,
Llama-68M's answers remain largely incoherent: asked about knife skills, it
answers about "a 10-year old son who is currently in the Army"; asked about
knee pain during training, it ends with "This entry was posted on Tuesday,
April 3rd, 2019 by admin" (a leftover blog-post artifact bleeding through).
The fix that completely solved the big model's problem does nothing here --
this isn't a formatting issue, it's a genuine capability ceiling. 68M
parameters is too small to answer open-ended real-world questions
coherently, regardless of prompt structure.

**Real system-design implication, not just a footnote**: if the hybrid
system's premise depends on the small model handling some real user
questions independently, and it's this incoherent under ideal conditions,
that's a genuine viability concern for the on-device model choice itself --
separate from and more fundamental than anything about SD or the handoff
mechanism (both of which now work correctly, confirmed above). Directly
connects to the parked quantized-bigger-drafter idea: a more capable
on-device model may be a requirement, not an optimization, if the small
model is meant to be user-facing rather than purely a speculative-decoding
drafter.

**Timing**: escalation #1 avg speedup 0.998x, escalation #2 avg 1.050x --
both close to breakeven, weaker than the ~1.2x from the raw-dialogue-
continuation version. Noted honestly; secondary to the coherence finding
above.

**Where this leaves Component C overall**: the handoff MECHANISM (SD
verification, escalation, de-escalation, repeated switching) is now
validated as sound -- quality-equivalent, reasonably fast, robust to
repeated bidirectional switching, and (with correct prompt framing)
produces genuinely coherent answers on the big-model side. The open problem
is no longer about SD or handoff at all -- it's whether the specific small
model chosen (Llama-68M) is viable as anything more than a drafter. That's
the real next question for this line of work.

---

## RESULT (2026-07-24): drafter swap to a realistic instruction-tuned
small model -- fully resolves the coherence problem, AND improves timing

Called out directly: Llama-68M (68M params, base model) was an
unrealistically extreme choice for the "user-facing small model" role --
it was picked earlier purely to match OmniDraft's ratio-matching regime for
the (separate, already-closed) acceleration experiments, not because it was
ever a real candidate for an actual on-device assistant. Swapped to
Qwen/Qwen2.5-0.5B-Instruct (real, instruction-tuned, actually-deployable
scale) for `component_c_qa_multiturn_v2.py` -- same 3-topic Q&A structure,
same escalation pattern, but now BOTH models get proper `apply_chat_template()`
framing (Qwen2.5-0.5B-Instruct is instruction-tuned, unlike Llama-68M, so a
chat template is meaningful for it too). Bonus: Qwen2.5-0.5B-Instruct's
vocab_size (151936) differs from Qwen2.5-7B-Instruct's (152064, confirmed
earlier this session), so this pairing genuinely exercises cross-vocab SD.

**Coherence: complete fix, all 12 answers across all 3 topics.** Every
small-model turn (A1, A3) is now correct, on-topic, and specific -- e.g.
asked for a simple recipe to practice knife skills on, it gives an actual
recipe (Garlic Butter Chicken, ingredients, instructions); asked about
common beginner gardening mistakes, it correctly explains overwatering and
root rot. No hallucination, no off-topic drift, no artifacts -- the
complete opposite of Llama-68M's answers to the identical questions.

**Timing: also improved, unexpectedly.** Escalation #1 avg speedup 1.244x
(tok/call 2.305), escalation #2 avg 1.195x (tok/call 2.328) -- both clearly
better than Llama-68M's 0.998x/1.050x on the identical test structure.
tok/call nearly doubled (1.27-1.43 -> 2.3-2.6). Mechanism: a same-family,
more capable drafter agrees with the target far more often, and that
acceptance-rate gain more than offsets the cost of running a ~7x larger
drafter model. This is a clean, real-world confirmation of something the
reverse-KL training experiments were trying (and failing) to achieve
through training earlier tonight -- draft *quality*/alignment with the
target matters more than raw drafter size, and a sensible same-family
choice gets there for free, no training required.

**What's still open**: this was tested on a cloud T4, not real edge
hardware -- 0.5B at fp16 is small enough to likely be fine, but whether it
needs quantization (AWQ/GPTQ, not vanilla bitsandbytes -- see the earlier
caveat this session about bnb's dequant-then-compute overhead not
delivering real inference speedups) for genuinely constrained on-device
deployment is a separate, still-untested question, distinct from anything
about drafting-speed.

**Overall verdict for tonight's Component C + drafter-choice work**: every
open thread is now closed with a real number or a clearly labeled next
question. The hybrid handoff design (SD verification, escalation,
de-escalation, repeated switching) works. The earlier "small model can't
handle real turns" concern was a drafter-choice artifact, not a design flaw
-- fixed by picking a realistic-scale instruction-tuned model instead of an
extreme research-only size. Remaining open items: multi-turn beyond 2
escalations, on-device quantization for real edge hardware, and connecting
all of this to an actual router making real decisions instead of a
hardcoded escalation pattern.

---

## RESULT (2026-07-24): first real router-decision integration + a
self-caught correction on the drafter-swap timing claim

**Router integration** (`component_c_real_router.py`): first time the
router actually DECIDES rather than following a hardcoded escalation
pattern -- signal used was the small model's own confidence (avg per-token
probability on its greedy-decoded answer), escalate if below a threshold
(0.55, picked blind). 2 topics, 6 questions each (extended from 4), 2
deliberately hard/obscure questions per topic to test discrimination.
Mechanically clean (no errors, `output_scores` extraction worked). Result:
**12/12 escalated** -- threshold too high, actual confidence range was
0.44-0.55 even on questions the same drafter answered perfectly in the
prior test. Worse: the signal doesn't reliably discriminate difficulty
either -- topic 0 showed a rough downward trend with harder questions
lowest, but topic 1 did NOT (the deliberately hard Q5/Q6 scored HIGHER
confidence than the easy Q3). Same lesson as OmniDraft's own design choice
(a trained accept-head, not raw model probability) -- naive greedy-decode
token probability isn't a well-calibrated signal for this kind of decision.
**Integration mechanism is proven; the specific heuristic needs real
calibration or a different signal entirely. Router calibration explicitly
parked for a later session** per direct instruction -- this session's
remaining effort goes toward strengthening what's already validated.

**Strengthening pass, and a self-correction**: retested Component C's core
SD-handoff speedup claim (`strengthen_core_speedup.py`) using the full
30-example LoCoMo heldout set (not an 8-example subsample) with
Qwen2.5-0.5B-Instruct as drafter -- the SAME clean single-handoff design as
the original wall-clock grid test, for a genuine apples-to-apples
comparison this time. Result: **avg speedup 1.213x (std 0.144, min 0.924x,
max 1.458x, median 1.228x)** -- LOWER than Llama-68M's original 1.43-1.56x,
despite much better tok/call (2.119 vs ~1.8-1.9).

This corrects a real claim made earlier tonight, worth recording precisely:
the earlier statement "swapping to Qwen2.5-0.5B-Instruct improved both
coherence AND timing" was based on comparing against the WRONG number --
Llama-68M's 0.998x/1.050x from the raw-dialogue multi-turn test, not
Llama-68M's actual flagship 1.43-1.56x from the wall-clock grid (a
different task, different baseline). This retest uses the identical
methodology to the original flagship number, and shows the opposite: the
bigger, more capable drafter gives WORSE wall-clock speedup, not better.
Mechanism: Qwen2.5-0.5B-Instruct (~7x more params than Llama-68M) agrees
with the target far more often (better tok/call, fewer verification
rounds), but each drafting step itself costs proportionally more compute --
the acceptance-rate gain doesn't fully pay back the increased per-token
drafting cost. Same shape as the K-inflation finding from earlier tonight:
tok/call alone doesn't predict wall-clock speedup, because it says nothing
about how expensive each round's drafting actually is.

**Honest, corrected picture: there's a real tradeoff, not a clean win.**
Llama-68M is fast to draft with but too weak to be coherent standalone.
Qwen2.5-0.5B-Instruct is coherent standalone but gives a smaller SD
wall-clock speedup than the tiny model. Coherence (needs a capable enough
model) and drafting-speed (needs a cheap enough model) pull toward
different model sizes -- there is no single drafter size that's
simultaneously best at both, at least not between these two specific
choices. This is a genuinely useful finding for the actual system design:
the "which small model to use" decision may need to be different depending
on whether that turn is escalating (drafting-speed matters) versus being
answered directly by the small model (coherence matters) -- or the system
may need two different small models for two different roles, not one.

---

## RESULT (2026-07-25): quantization closed out with a real fused-kernel
result; n-gram cache mis-scoped and corrected; do_sample dispatch discovery
(two real library bugs found + patched); OmniDraft full-component checklist

**Quantization, finished properly this time.** Last session's GPTQ Int4 test
used gptqmodel's `TORCH` fallback backend (0.958x, worse than fp16) because
the fused Marlin kernel failed to JIT-compile -- diagnosed tonight as a real
hardware-generation gap, not a fixable bug: T4 is compute capability (7,5)
(Turing), Marlin's fused int4 kernels target Ampere+ (8.0+) tensor cores.
Tested all 4 available backends via `GPTQModel.load(..., backend=...)`:
`GPTQ_MARLIN`, `AWQ_GEMM`, `AWQ_MARLIN` all fail to build on this hardware;
**`EXLLAMA_V2` works** -- a genuine fused kernel, not a dequant-fallback.
Reran the timed comparison on a new mixed-complexity conversational dataset
(5 topics x 8 questions each, tagged simple/moderate/complex, built
specifically to span difficulty within natural conversation rather than
isolated escalation points -- `conv_dataset_mixed_complexity.json`):
fp16 baseline **1.185x** (tok/call 2.366) vs GPTQ+EXLLAMA_V2 **1.111x**
(tok/call 2.266) -- real progress from the broken kernel (0.958x -> 1.111x,
~16% relative recovery from fixing the kernel alone) but still below fp16.
**Likely mechanism**: the drafter is tiny (0.5B, batch size 1, single-token
autoregressive decode) -- fused low-bit kernels need enough matmul work to
amortize their own overhead against; fp16 tensor-core matmuls are already
near-optimal at this scale, and quantization adds real dequant/kernel-launch
overhead without enough work to hide it behind, on top of a small real
acceptance-rate cost from int4 precision loss. **Verdict: closed out
properly, with a working fused kernel to back the conclusion -- not
"quantization doesn't work," specifically "quantizing a small single-
sequence drafter on this hardware doesn't pay off, even with the best
available kernel." Might behave differently for a bigger drafter or batched
serving -- neither tested.**

**N-gram cache: first attempt was mis-scoped, caught and corrected.** Built
and ran a feasibility test measuring whether REPEATED PHRASES recur across a
multi-turn conversation (coverage/hit-rate analysis, n=3/4/5-grams) --
result was genuinely low (avg useful rate 4.8% at n=3, dropping to 0.7% at
n=5) and the conclusion drawn was "don't build the real cache, ceiling too
low." **That conclusion was invalid** -- direct WebFetch of the actual paper
(arxiv 2507.02659, full HTML, not the abstract) confirmed OmniDraft's n-gram
cache is NOT a phrase-repetition cache at all. It's a cross-vocabulary
TOKEN-TRANSLATION mechanism: an online-updated cache that merges sequences
of the DRAFTER's sub-tokens into a single TARGET token (their example:
drafter tokens `['f','la','ke']` -> target token `['flake']`), used because
different tokenizers split the same text differently. What got tested was a
different idea that happens to share the name "n-gram cache." Real component
is still unbuilt.

**Paper details confirmed directly from source** (WebFetch, not memory --
this matters, several things below correct earlier assumptions):
- Ablation (Table 5, GSM8K subset): direct mapping alone (ELL_DM) 1.59x ->
  + n-gram postprocessing 1.61x -> + n-gram training loss 1.66x. Small,
  incremental gains from the n-gram machinery specifically, not the primary
  driver of their headline number.
- Table 6: ELL_N-gram (NLL) loss ALONE, no direct-mapping/KL term, actually
  HURTS (0.86x, worse than untrained baseline) -- only helps combined with
  the distillation loss (0.86x -> 1.51x combined). Direct, paper-confirmed
  evidence that components are synergistic, not independently additive --
  this generalizes the lesson from the 2026-07-22 reverse-KL correction
  (incomplete implementation != the idea being wrong) into a documented,
  primary-source-backed pattern, not just an inference from our own results.
- **Speedup is highly domain-dependent, not one number**: GSM8K (math) 1.70x
  best-case; MBPP+HumanEval (code) 1.33x; Alpaca (general instruction-
  following) 1.30x; XSum (summarization) 1.24x. Alpaca is the closest match
  to our own conversational Q&A testing -- **our own 1.209x/1.185x-range
  results are in the same band as their Alpaca number**, not an anomalous
  failure relative to their flashier GSM8K headline.
- Hardware: **NVIDIA A100 (40/80GB)**. All of tonight's and every prior
  session's testing has been on T4s (Turing) -- directly explains why Marlin
  wouldn't compile (see quantization section above), a real generational gap
  not previously accounted for when comparing our numbers to theirs.

**do_sample dispatch discovery -- bigger than the n-gram cache question.**
Read `transformers/generation/utils.py::_get_candidate_generator` directly:
for different-tokenizer pairs, the candidate-generator CLASS is chosen
purely by `do_sample`, unrelated to anything about vocab size or model
pairing. `do_sample=False` (every test this entire research effort, always)
-> `AssistedCandidateGeneratorDifferentTokenizers`: decode drafter tokens to
TEXT, re-encode with the target tokenizer, patch discrepancies with a
longest-common-diagonal-subsequence heuristic. Real per-round text-roundtrip
cost, and boundary-shifting risk (same class of bug as the BPE tokenization
mismatch found earlier this session). `do_sample=True` ->
`UniversalSpeculativeDecodingGenerator` + `AssistantToTargetTranslator`: a
STATIC, precomputed-once, STRING-based 1:1 vocab-overlap mapping (assistant
LM head pruned to the overlapping subset), no text roundtrip per step at
all. This is conceptually the paper's "direct mapping" (ELL_DM) baseline --
we had never once triggered it, in any session, because every test used
greedy decoding for the clean byte-identical correctness guarantee.

**Two real library bugs found and patched** testing this (transformers
4.57.6, not a version-pin issue -- verified via direct source read, not
guessed): (1) `AssistantToTargetTranslator.__init__` only sets
`assistant_overlap_token_ids` when `len(_suppress_input_ids) > 0`; our
Qwen2.5-0.5B-Instruct/Qwen2.5-7B-Instruct pairing has ZERO suppressed tokens
(confirmed via diagnostic script: `len(_suppress_input_ids) == 0`, both
models share the same BPE merges) so the attribute is never created, but
`get_target_ids`/`get_target_logits` unconditionally reference it when
`assistant_prune_lm_head=True` (always true for do_sample=True, hardcoded in
the dispatcher) -> `AttributeError`. Fixed with a monkey-patched `__init__`
setting it to `torch.arange(...)` (semantically correct: nothing to prune,
identity mapping). (2) Second, deeper bug: the mapping array is sized to the
ASSISTANT TOKENIZER's real vocab (151665, confirmed via diagnostic script),
but the assistant MODEL's actual LM head/embedding table is sized to its
configured `vocab_size` (151936 -- includes reserved/padding embedding slots
beyond real tokens), and since no real pruning happens in the zero-suppress
case, `get_target_logits` tries to index a 151936-wide logits tensor with a
151665-wide index array -> shape-mismatch `RuntimeError`. Fixed by patching
`get_target_logits` to fall through to the SAME slicing HF's own
"not-pruned" branch already does correctly (`assistant_logits[..., :151665]`
before indexing) -- both fixes verified via a standalone diagnostic script
(`diag_translator.py`) that printed the actual internal state rather than
guessing from stack traces alone.

**Result once both bugs were fixed**, same mixed-complexity dataset,
identical methodology, only `do_sample`/translator changed: **tok/call
2.366 -> 2.883 (+22% relative)** -- confirms the mechanistic hypothesis
directly, the better translator produces real, substantial acceptance-rate
gains, no boundary-shifting noise. **But overall speedup 1.185x -> 1.132x,
net WORSE** despite the tok/call gain -- the third time this session
tok/call has failed to predict wall-clock speedup (after the drafter-size
tradeoff and the GPTQ quantization result). Mechanism: sampling-based
rejection verification does real extra per-round work (softmax +
probability-ratio computation across every candidate position) that greedy's
single argmax-equality check doesn't -- this overhead sits specifically on
the SD/assisted path, not on plain generation (which samples one token at a
time regardless), so it doesn't cancel out in the ratio.

**Open question, not yet tested**: can `UniversalSpeculativeDecodingGenerator`
+ the fixed `AssistantToTargetTranslator` be manually wired into GREEDY
verification instead of going through `do_sample=True`, to get the
translator's tok/call gain without paying sampling's per-round overhead? If
HF's actual accept/reject loop doesn't intrinsically require sampling to use
this candidate-generator class -- only the convenience dispatcher couples
them -- this could plausibly beat both of tonight's numbers, keeping the
byte-identical-to-plain-greedy correctness guarantee the whole Component C
quality-equivalence claim rests on. Next thing to check.

**OmniDraft full-component checklist, per explicit instruction not to
cherry-pick/fragment again**: four components, confirmed synergistic not
independently additive (see Table 6 above). Status of each, honestly:
(a) **Direct-mapping translator** -- working as of tonight (two bugs fixed),
tok/call improvement confirmed, net wall-clock benefit still unresolved
(see open question above). (b) **Online n-gram cache** (real version: cross-
vocab sub-token merge translation) -- NOT built; tonight's first attempt
tested the wrong mechanism entirely (see correction above), real version is
the next build. (c) **Reverse-KL distillation + n-gram-NLL loss** -- only
the reverse-KL half was ever trained (2026-07-21/22 session), n-gram-NLL
term stubbed at zero throughout; trained on Llama-68M/Gemma-2-9b, a
DIFFERENT pairing than tonight's Qwen2.5-0.5B/7B mechanism work. (d)
**Acceptance-prediction head** -- built and tested (v2, 2026-07-23/24,
same Llama-68M/Gemma-2-9b pairing), 65% accuracy, recovered to rough parity
with the untrained baseline, not a clear standalone win. **None of the four
have ever been combined into the actual full system, and the trained
pieces (c, d) live on a different model pairing than the mechanism work
(a, b).** Any future "does OmniDraft's approach work" conclusion needs to
account for this -- testing one piece alone and drawing a verdict from it
is exactly the mistake already made and corrected multiple times across
this research effort (2026-07-22 reverse-KL correction, tonight's n-gram
mis-scope correction).

---

## RESULT (2026-07-25, continued): greedy verification + direct-mapping
translator -- new flagship number, beats everything measured tonight

Resolved the open question from the previous entry. Confirmed via direct
source read (`_assisted_decoding` in `generation/utils.py`) that candidate-
generator SELECTION and the `do_sample` value used for VERIFICATION math are
independently read from the same `generation_config` object -- HF's
dispatcher couples them by convenience, not mathematical necessity. Monkey-
patched `GenerationMixin._get_candidate_generator` to always build
`UniversalSpeculativeDecodingGenerator` + the (bug-fixed)
`AssistantToTargetTranslator` for our model pair, regardless of the real
`do_sample` value, which stayed `False` throughout -- so verification still
uses the byte-identical-to-plain-greedy equality check, correctness
guarantee fully intact. The patch is a faithful copy of HF's own
`do_sample=True` branch (not a simplified reimplementation), gated to only
apply for the different-tokenizers case with no other special-case generator
active (early-exit, prompt-lookup, MTP, Gemma4 assistant).

**Result, same mixed-complexity dataset, same methodology throughout**:

| configuration | avg speedup | avg tok/call |
|---|---|---|
| greedy + primitive translator (everything before tonight) | 1.185x | 2.366 |
| sampling + direct-mapping translator | 1.132x | 2.883 |
| **greedy + direct-mapping translator** | **1.394x** | **2.797** |

Confirms the hypothesis cleanly: the primitive translator's decode/re-encode
text roundtrip was costing real acceptance rate (boundary-shifting noise),
and sampling-based rejection verification's per-round overhead (softmax +
probability-ratio math across every candidate) was costing more than its
own acceptance-rate gain bought back. Combining greedy's cheap verification
with the better translator's clean candidate quality gets both wins at
once: +17.6% over the old baseline, +23% over the sampled version, tok/call
nearly as high as sampling's (2.797 vs 2.883) without paying sampling's tax.
Breakdown by complexity: simple 1.452x, moderate 1.382x, complex 1.325x --
holds up across the full difficulty spectrum, not just easy questions.

**This is now the flagship Component C speedup number**, superseding the
1.209x/1.213x figures from earlier sessions (same model pair, same core
mechanism, just the corrected translator+verification combination) -- worth
using this one in any future write-up or presentation, not the older
figures. Scripts: `mixed_complexity_test_greedy_directmap.py` (the patch),
`test_load_memory.py`/`diag_translator.py` (diagnostics that found the two
underlying library bugs, documented in the previous entry).

**Not yet tested**: whether this combination also holds up with the GPTQ-
quantized drafter (closed out under the OLD translator; the direct-mapping
translator's cleaner candidates might change that verdict too, since a
chunk of quantization's problem was wasted/rejected candidates from
precision loss -- a cleaner translator doesn't fix precision loss, but it
does mean fewer candidates get lost to translation noise on top of it).
Also not yet tested: this combination as the foundation for the real
n-gram cache build (per the checklist in the previous entry) -- this is the
correct base to build it on, not the old primitive-translator baseline.

---

## RESULT (2026-07-26): the REAL OmniDraft n-gram cache built and tested --
mechanism confirmed working, wall-clock verdict inconclusive (implementation
gap, not a mechanism failure)

**Pairing correction, checked before building anything**: confirmed via
`diag_suppression_rate.py` that tonight's Qwen2.5-0.5B/Qwen2.5-7B-Instruct
pairing has **0% suppressed tokens** (same tokenizer family, every drafter
token already has a 1:1 target match) -- the n-gram cache would have
literally nothing to do on this pairing. `JackFram/llama-68m` /
Qwen2.5-7B-Instruct has **30.4% suppressed** (9726/32000) -- genuinely
heterogeneous, real signal to work with. Switched to this pairing for the
n-gram cache test specifically (the original Component C pairing from
earlier in this research effort, not an arbitrary choice).

**Built as a standalone draft-verify loop** (`ngram_merge_sd.py`), not an
extension of HF's internal CandidateGenerator classes -- already spent
significant time debugging two real library bugs deep in those internals
this session; a standalone loop gives full, correct control over the
translation logic without more of that risk. Translation priority per
drafted run: (1) `direct_map`, the same static 1:1 string-overlap mapping
`AssistantToTargetTranslator` computes, reused directly; (2) `ngram_cache`,
online-updated, maps a tuple of contiguous UNMAPPED drafter token ids to the
target token ids obtained by decoding that run and re-encoding with the
target tokenizer; (3) fallback -- decode/re-encode the unmapped run fresh,
then ADD it to the cache for future reuse ("add to n-gram cache if there
exists unseen n-gram instance", matching the paper's description directly).
Greedy verification throughout -- same byte-identical-to-plain guarantee
validated all session.

**Mechanism result, full mixed-complexity dataset (40 questions)**: cache
grew to just **23 entries** total, yet achieved an **86.6% hit rate** among
the tokens that actually needed cache-level handling (149 hits vs 23
misses-then-cached, out of 10163 total translated positions). Real, correct,
working cache behavior -- a small number of learned mappings captured the
vast majority of repeat occurrences. Direct-map rate was 98.3% -- only
~1.7% of all drafted tokens ever needed the cache/fallback path at all, even
though 30.4% of the *vocabulary* is unmapped (common/frequent tokens skew
toward having direct overlaps; the unmapped 30% skews toward rarer tokens
that don't actually get drafted as often).

**Wall-clock verdict: inconclusive, and the reason why is itself the real
finding.** Ran `with_cache` (0.410x avg speedup) and a `no_cache` control
(0.416x) -- essentially IDENTICAL, confirming the cache's effect is
completely invisible against this implementation's dominant cost: the
standalone loop recomputes the ENTIRE forward pass from scratch every draft
round, with NO KV-cache reuse at all (not even within a single growing
multi-turn conversation) -- worst observed case took 57.6s vs plain's 11.4s
on the longest-accumulated-history question, a pure implementation artifact
unrelated to the n-gram cache itself. Since the cache only ever touches
~1.7% of tokens, even a perfectly-working cache can't move the needle
against an overhead that large. This ISOLATES the finding cleanly (both
conditions share the identical handicap, so the near-zero delta between
them is real signal, not noise from an unfair comparison) but means this
session's implementation cannot answer "is the real wall-clock win worth
the engineering cost" -- that needs a KV-cache-reusing version, a genuinely
separate, bigger build (matching the effort HF's own
`UniversalSpeculativeDecodingGenerator` already has under the hood).

**Expectation-setting for that future build, from two independent sources**:
OmniDraft's own ablation (previous entry) showed the n-gram cache
contributing a SMALL incremental gain on top of direct mapping (1.59x ->
1.61x, ~1.3% relative) -- not their headline number's primary driver. Our
own token-fraction data (~1.7% of tokens ever cache-relevant) points the
same direction independently. Both agree: even a properly-optimized
implementation should be expected to deliver a modest (~1-5%), not
transformative, wall-clock improvement -- useful to know before deciding
whether the KV-cache-reuse engineering investment is worth it.

**Status against the OmniDraft component checklist (previous entry),
updated**: (a) direct-mapping translator -- working, tested, new flagship
number found (previous entry). (b) online n-gram cache -- now genuinely
built and mechanism-validated (this entry), wall-clock verdict pending a
KV-cache-optimized reimplementation. (c) reverse-KL + n-gram-NLL loss --
still only half-implemented, still on the old Llama-68M/Gemma-2-9b pairing.
(d) acceptance-prediction head -- still only tested standalone, same old
pairing. Still nothing has been combined into the actual full system.

---

## RESULT (2026-07-26, continued): n-gram cache thread closed out for real --
Prompt Lookup Decoding confirms, via a completely different mechanism, that
this conversational domain lacks the repeated structure the whole family of
n-gram techniques needs

**Course-correction on scope, worth recording precisely**: was about to sink
another ~2-2.5 hours into properly fixing the assistant-side KV-cache gap in
the custom n-gram merge loop (needed for a fair with/no-cache comparison on
Qwen2.5-0.5B-Instruct / google/gemma-2-9b, confirmed via
`diag_suppression_rate.py` to have a genuinely large 57.6% suppression rate,
87423/151665 -- nearly double Llama-68M's 30.4%, and unlike that pairing,
both models here are real deployment candidates from this project, not a
research-only artifact). **User pushback, correctly**: was this
overcomplicating what the paper actually needed, and had adjacent/simpler
established techniques been checked before building custom machinery?
Answer: no, and they should have been. Checked directly (WebSearch, not
memory): **Prompt Lookup Decoding (PLD)** is the real, widely-used,
already-implemented "n-gram" technique in this space -- built into HF
Transformers (`prompt_lookup_num_tokens`) and vLLM
(`PromptLookupCandidateGenerator` -- a class name literally read past in
this session's own dispatch-code investigation without registering what it
was). PLD is fundamentally simpler than OmniDraft's cross-vocab merge cache:
no draft model at all, just string-match the most recent n-gram against the
prompt/generated-so-far text and guess the continuation is whatever followed
it before -- no cross-vocab translation problem exists in the first place,
since only one model's cache is ever involved. Real-world use case: tasks
where output re-uses input text verbatim (summarization, code editing),
reported 2-4x speedups there.

**Tested directly** (`pld_test.py`, single model, `Qwen2.5-7B-Instruct`,
`prompt_lookup_num_tokens=10`, same mixed-complexity dataset and
alternated-order/warm-up methodology as every other test): **avg speedup
0.998x** (breakeven), **avg tok/call 1.160** -- the n-gram lookup almost
never finds a match that survives verification. Breakdown: simple 0.951x,
moderate 1.020x, complex 1.036x -- flat across the board, no signal
anywhere.

**This closes the loop on the EARLIER whole-phrase-repetition feasibility
test from this same session** (avg useful rate 4.8% at n=3, dropping to
0.7% at n=5) -- that test was, in retrospect, testing almost exactly the
same underlying phenomenon PLD tests (literal n-gram repetition
exploitability), just via a custom coverage/hit-rate analysis instead of
the real established technique. **Two independent methods now agree**: this
specific domain -- open-ended, multi-turn conversational Q&A -- does not
have enough literal text repetition for ANY n-gram-based technique (PLD's
simple prompt-matching, OR OmniDraft's more elaborate cross-vocab merge
cache) to meaningfully help. This is a property of the TASK, not a flaw in
either technique -- PLD's own established use cases (summarization, code
editing) involve genuine verbatim reuse of input text; fresh conversational
answers, even on repeated topics, get generated in new phrasing each time.
Also consistent with OmniDraft's own domain-dependent ablation numbers from
the earlier entry (GSM8K/math best, general instruction-following weakest)
-- conversational Q&A sits even further from the favorable end than their
own weakest reported domain.

**Verdict, with real confidence now**: closing the entire n-gram-cache
thread (component (b) of the OmniDraft checklist) for THIS project's actual
domain -- not because the mechanism doesn't work (it's correctly
implemented and validated, both the custom merge-cache and PLD), but
because the conversational task itself doesn't have the exploitable
repetition either technique needs. The ~2-2.5 hour assistant-side KV-cache
fix was correctly NOT pursued -- PLD's near-zero result on the established,
zero-custom-code technique makes it very unlikely the harder, custom
merge-cache build would have found something PLD missed on the same data.

**Process lesson, worth keeping**: check for adjacent/established
techniques via direct search before building custom machinery for a
research idea, even when the custom machinery seems well-motivated by the
paper being reproduced -- this would have surfaced PLD immediately and
saved real engineering time on the merge-cache KV-cache reconciliation work
that turned out not to be worth completing.

---

## RESULT (2026-07-26, continued): joint cost model -- draft speed, network
latency, and server cost combined into one estimate; user-latency win
survives most conditions but collapses in the worst realistic case

**Context for why this was built**: everything measured all session ran
both models on the same GPU with zero network latency -- a real, flagged
gap once the actual deployment target (drafter on a phone, target on a
remote cloud server) is considered. Literature search (previous entries)
confirmed real edge-cloud SD systems exist and work, but also confirmed
network cost alone can flip a compute-side win negative (DSSD: 0.92x under
realistic conditions). Built a pure arithmetic model (`joint_cost_model.py`,
no GPU/VM needed -- every input is either already-measured, published, or an
explicitly-labeled simulated assumption) to see whether our flagship result
survives once real mobile drafter speed and real network round-trips
replace our same-box test conditions.

**Inputs, each tagged by source to avoid conflating them** (a real mistake
caught and fixed while building this -- see below):
- [MEASURED, ours] `TOK_PER_CALL = 2.797`, `SERVER_TIME_PLAIN = 11.2s` for a
  60-token response -- from tonight's flagship greedy+direct-mapping test.
- [DERIVED, approximation] per-round server verification cost modeled as
  `SERVER_TIME_PLAIN / MAX_NEW` -- justified by the earlier prefill-scaling
  finding (small batched verification passes are memory-bandwidth-bound,
  cost only marginally more than a single token), not a fresh measurement.
- [PUBLISHED] mobile drafter speed: 30-40 tok/s for sub-1B quantized models
  on current-generation phone SoCs (2026 llama.cpp-class benchmarks) --
  explicitly NOT the Lever paper's 1.5-3.4 tok/s figure, which is a
  flash-offloaded 7-14B target model, a different model class and different
  bottleneck (storage I/O) that would have been the wrong number to borrow.
  Used 35 tok/s as "healthy," 15 tok/s as "throttled" (informed by a cited
  real example -- Llama 3.2 3B dropping from 37.58 tok/s under sustained
  thermal load on iPhone -- not a directly published number for our exact
  model, labeled as an inference from a published pattern).
- [ASSUMED] network RTT scenarios modeled the same way DSSD/ConfigSpec did
  it: real compute numbers + injected delay, not a live network -- good
  wifi/5G (20ms), typical 4G (75ms), poor connectivity (150ms).

**A real mistake, caught before presenting results**: the first version of
this model labeled a 0ms-RTT row "same box (this session's actual tests)" --
but it substituted the PUBLISHED mobile tok/s number while keeping zero
network latency, which is a scenario we never actually tested (our real
1.394x used a GPU-speed drafter). That row doesn't reproduce 1.394x and
never could; the label implied it did. Caught on inspection before
reporting, relabeled explicitly as a hypothetical decomposition, not a
sanity check against the real measured result. Worth remembering as a
concrete example of exactly the failure mode to watch for when combining
numbers from different sources -- a mistake in the model-builder's own
work this time, not something a user needed to catch.

**Results**:

| network | drafter state | naive (s) | SD (s) | speedup |
|---|---|---|---|---|
| 0ms (hypothetical, mobile drafter) | healthy | 12.92 | 7.48 | 1.728x |
| 0ms (hypothetical) | throttled | 15.20 | 12.05 | 1.262x |
| good wifi/5G (20ms) | healthy | 12.94 | 7.90 | 1.637x |
| good wifi/5G | throttled | 15.22 | 12.48 | 1.220x |
| typical 4G (75ms) | healthy | 12.99 | 9.08 | 1.430x |
| typical 4G | throttled | 15.28 | 13.66 | 1.119x |
| poor connectivity (150ms) | healthy | 13.07 | 10.69 | 1.222x |
| **poor connectivity** | **throttled** | 15.35 | 15.26 | **1.006x** |

**Server-side compute, independent of network/drafter speed entirely**:
11.2s -> 4.0s of target-GPU work per response, a robust **2.797x**
reduction (matches TOK_PER_CALL almost exactly, as expected -- this part of
the value proposition doesn't depend on anything modeled above).

**Reading**: the user-facing latency win survives across every realistic
network condition as long as the drafter stays healthy (1.2-1.6x range).
It collapses to essentially no benefit (1.006x) only under the worst
realistic combination -- a thermally-throttled phone AND poor connectivity
stacked together. Server-side cost savings hold unconditionally throughout.
This is the two-track story from the literature-review entries made
concrete with actual numbers for our own pairing: "cheaper for the server"
is robust; "faster for the user" is real but fragile, and identifiably
fragile specifically at the intersection of bad hardware conditions and bad
network conditions, not from either alone.

**Caveats carried forward, explicitly**: mobile drafter numbers are
borrowed from published benchmarks for similarly-sized models, not measured
on our actual drafter; the per-round server cost is an approximation, not a
fresh measurement; nothing here has been validated on a real phone or a
real network -- this is a plausibility estimate, not a new measurement, and
the next real step (agenda item 5, real mobile hardware) is what would
close that gap.

**Next planned, per the agenda this unblocks**: model whether pipelining
(overlapping next-round mobile drafting with current-round network
transit) recovers the near-breakeven worst case -- directly motivated by
the DSSD-adjacent Jetson-Nano+A100 paper finding an 11% gain from exactly
this ("preemptive drafting") in a real edge-cloud test, the precondition
(genuine network-driven dead time) our own same-GPU pipelining test lacked.

---

## CORRECTION (2026-07-26): OmniDraft's online training is a specified
mechanism, not an implied one -- understated this earlier, verified against
the actual paper text and corrected

Earlier tonight's entries characterized the drafter's continual/online
adaptation as something "the source paper's design IMPLIES" -- this
understates what's actually in the paper. Verified directly (WebFetch,
arxiv 2507.02659) rather than continuing on the earlier assumption:

**Acceptance-prediction head is trained ONLINE in the paper, not offline.**
Quote: "a lightweight head network f_phi(.) predicts the acceptance rate of
the current proposed token... trained using weighted BCE loss L_adapt with
acceptance ratios min(1, p(y_i)/q(y_i)) as labels." The label is a
continuous acceptance RATIO, not a simple binary accept/reject (a detail
this session's own "scoped-down online head" proposal got wrong -- would
have built the wrong label). Two variants specified: **joint** (head
updates together with the drafter after each batch, same online data) and
**interleaved** (head updates more frequently off a larger buffer, explicitly
to "mimic slowly moving labels to reduce distribution shift").

**The drafter itself has a genuine, explicitly-specified online training
mechanism** -- not something inferred or implied. Quote: framed as
"continuous alignment during speculative decoding, even when the target
model changes due to personalization or task switching." Mechanism,
precisely: after EACH BATCH of SD inference, gradient-update the drafter on
the SAME hybrid loss already discussed in the ablation-table entry --
reverse-KL on direct-mapping tokens (richer signal, real distributions
available), NLL on n-gram-cache-merged tokens (only point-wise probability
estimates available for those) -- using that batch's actual accepted and
corrected tokens as the training signal. "All the tokens will contribute to
the loss calculation including the tokens that were accepted by the target
as they would provide option for improved alignment between the two
models."

**What this means for our own gap, corrected and sharpened**: earlier
framing was "we never built online adaptation of the drafter, the paper
implies we should" -- more precisely, the paper SPECIFIES exactly how, with
a concrete recipe (hybrid loss, per-batch trigger, accepted+corrected
tokens as signal, two head-update variants). Our own gap is now clearly
layered, not singular: (1) even the OFFLINE version of this loss was never
completed here -- the reverse-KL half was trained (2026-07-21/22 session),
but the n-gram-NLL half was stubbed at zero throughout, so the loss itself
was never finished, independent of the online-vs-offline question; (2) the
whole thing was never made to trigger continuously from live batches either
-- it was a one-time offline training script, not a running loop integrated
with inference.

**Revised next step for agenda item 4**: build toward the paper's actual
specified recipe -- complete the hybrid loss first (both terms, not just
reverse-KL), then wire it to trigger after each SD batch using real
accepted/corrected tokens as signal, with the acceptance head updated
alongside (joint or interleaved, per the paper's two variants) using
continuous acceptance-ratio labels, not a simplified binary one. This is a
bigger, more precisely scoped undertaking than the "just update the
acceptance head online" idea floated earlier this session -- that idea
undershot both the label type and the fact that the paper couples the head
and drafter updates together.

---

## RESULT (2026-07-26, continued): online-training build, step 1 (data
collection) done -- two real bugs found and fixed, both worth remembering

**Pairing chosen deliberately**: Qwen2.5-0.5B-Instruct (drafter) /
google/gemma-2-9b-it (target) -- NOT the same-family Qwen/Qwen pairing,
because that has 0% suppression (confirmed earlier) and would make the
n-gram-NLL half of the hybrid loss permanently dead code. Confirmed via
this build: 64242/151665 assistant tokens direct-mapped (42.4%, matches the
57.6% suppression figure from `diag_suppression_rate.py` exactly), so this
is the only validated pairing where both halves of OmniDraft's hybrid loss
actually get exercised.

**Infra note for next time**: `google/gemma-2-9b` download stalled
indefinitely via HF's newer Xet storage backend -- the blob sat at 0 bytes
for 7+ minutes while the download log showed repeated `503 Service
Unavailable` errors from `cas-server.xethub.hf.co` / `us.aws.cdn.hf.co`.
Real external HF infrastructure issue, not our config. Fixed by setting
`HF_HUB_DISABLE_XET=1` before the python call, which falls back to plain
HTTP downloads -- 13GB downloaded in under 30s once switched. Worth trying
this env var first if any future HF download stalls with near-zero
progress for more than a minute or two.

**Model choice correction**: `google/gemma-2-9b` (base, non-instruction-
tuned) has no `chat_template` registered at all -- `apply_chat_template()`
raises immediately. Switched to `google/gemma-2-9b-it`, the correct choice
anyway for an actual conversational target. Vocab/tokenizer is shared
between base and -it variants, so the suppression rate figure carries over
unchanged.

**The real bug, and the one worth remembering most**: first collection run
completed without crashing but collected almost nothing (2 records total
across 8 full questions) -- looked like a systematic near-100% rejection
rate, which briefly raised a wrong hypothesis (Gemma-2's alternating
sliding-window/full-attention layers interacting badly with generic
`DynamicCache.crop()`). Verified that hypothesis wrong FIRST via a targeted
diagnostic (`diag_gemma_predictions.py`) before accepting it -- plain
generation vs DynamicCache-based single-step continuation matched exactly
(same first predicted token), ruling out any cache/architecture corruption.
**The actual bug**: the drafter's prompt was built by decoding the TARGET's
fully chat-templated `target_ids` (Gemma's `<start_of_turn>`/`<end_of_turn>`
markers included, `skip_special_tokens=False`) and re-tokenizing that raw
text with the ASSISTANT's tokenizer -- which has never seen Gemma's special
tokens and just treats them as literal confusing text. Result: the drafter
degraded into generic "Q: ... A: ..." completion instead of a real
continuation (diagnostic showed it literally output "A: Vegetables are..."
for a cooking question), so its proposed tokens essentially never matched
what the target would produce. Same class of mistake as the earlier
raw-dialogue-vs-chat-template coherence collapse from the original
Component C multi-turn work -- prompt formatting mismatches between models
keep being the actual root cause behind results that initially look like
deeper architectural or algorithmic problems.

**Fix**: build the drafter's prompt from its OWN chat template
(`assistant_tokenizer.apply_chat_template`) applied to the same semantic
conversation history, then append the growing plain-text partial answer
(decoded with `skip_special_tokens=True`, so no cross-model special-token
leakage) as a raw continuation -- chat templates with
`add_generation_prompt=True` open the assistant turn without closing it,
so appending raw continuation text is a legitimate pattern, not a hack.

**Secondary bug, same fix pass**: the round loop's termination condition
counted progress from `records` (which excludes the corrected token at a
rejection, by construction) instead of actual `target_ids` growth --
undercounting real progress on every round with a rejection. Fixed by
tracking `target_ids.shape[1] - prompt_len` directly instead of a
separately-maintained counter.

**Result after both fixes**: 160 records collected across the same 8-question
smoke test, 136 direct-mapped + 24 n-gram-merged (~15% n-gram ratio) --
both halves of the loss now have real, healthy data to train against,
confirmed before moving to step 2 (loss implementation).

---

## RESULT (2026-07-26, continued): online-training build, step 2 (hybrid
loss) done -- reverse-KL on direct-mapped tokens, NLL on n-gram-merged
tokens, both validated, one numerical bug found and fixed

**Design**: reverse-KL(q_drafter || p_target) computed over a RESTRICTED
overlap subset (the ~64k assistant tokens with a valid direct target
mapping) rather than a full masked ~256k-vocab tensor -- softmax both
distributions over just that shared support, mathematically equivalent to
the full-vocab masked version (unmapped tokens get zero probability either
way) but far cheaper. NLL on n-gram-merged tokens: standard cross-entropy
of the drafter assigning probability to its OWN token(s) that were actually
drafted and, via the merge cache, accepted -- no target distribution
involved, matching the paper's stated reason for using NLL there (no clean
per-position distributional correspondence exists once tokens are merged).

**Real bug found**: reverse-KL came back NaN for a subset of the 136
direct-mapped records when first computed via `F.kl_div(..., log_target=
True)`. Root cause: with a ~64k-token restricted vocab, float32 softmax
genuinely underflows to an EXACT 0.0 for many low-probability entries, and
log(0.0) = -inf. The mathematically correct limit of q*log(q/p) as q->0 is
0 (standard KL convention), but IEEE float arithmetic evaluates literal
`0.0 * -inf` as NaN, not 0 -- PyTorch's kl_div kernel doesn't appear to
special-case this. Fixed by computing KL manually (`q * (log_q - log_p)`,
`torch.where(q > 0, kl_terms, 0)` to enforce the correct limit explicitly
instead of hitting the literal 0*inf case). After the fix: reverse-KL
mean=2.7994 (min 0.0001, max 13.33, all finite), NLL mean=0.2713 (min
0.0020, max 1.1702, all finite) -- both sensible magnitudes for their
respective loss types.

**Backward-pass mechanism confirmed working**: reloaded the drafter with
gradients enabled, ran 3 real optimizer steps against a small sample,
tracked loss 23.77 -> 18.32 -> 16.31 (monotonically decreasing) with real,
nonzero gradient norms throughout. This confirms the loss is genuinely
differentiable and a standard optimizer can reduce it -- the actual
mechanism, not just the math on static values.

**Explicit scope note, to avoid this being overclaimed later**: this
backward-pass smoke test used single isolated assistant tokens as input
(not the real growing conversational context each token was originally
drafted in) -- it validates that gradients flow and optimization works
mechanically, not that training on real context has been demonstrated yet.
Wiring real, contextual, live forward passes into an actual online training
loop during generation is task #18, not yet done -- this step deliberately
stayed scoped to "is the loss itself correct," per the original build plan
(validate the loss offline before adding online-loop complexity on top).

---

## RESULT (2026-07-26, continued): online-training build, step 3
(acceptance head) done -- a real data-collection bias found and fixed,
worth remembering for any future collection loop of this shape

**Design**: small MLP (896 -> 128 -> 1, matching Qwen2.5-0.5B-Instruct's
actual hidden_size) on the drafter's own hidden state, trained with BCE
against the CONTINUOUS label min(1, p_target(y)/q_drafter(y)) -- not
binary accept/reject, per the paper. Scoped to 'direct'-tagged records only
(same reasoning as the main loss split: the ratio needs a well-defined SAME
token compared in both vocabularies, which n-gram-merged tokens don't have).
"Weighted BCE" in the paper's own description is ambiguous about the exact
weighting scheme -- implemented here as standard BCEWithLogitsLoss against
a continuous soft target, flagged explicitly as an assumption rather than
silently resolved, in case the specific scheme turns out to matter later.

**Real bug found, worth remembering generally**: first training pass showed
labels clustered almost entirely at ~1.0 (mean 0.9907, min 0.7126) --
mechanically the loss decreased and gradients flowed fine, but the
predicted probabilities converged to ~1.0 regardless of the (slightly
lower) true label. Root cause: `collect_round` (step 1's script) discarded
every record past the rejection point via `records[:n_from_cand]` -- correct
for the KV CACHE (only accepted tokens' K/V entries are valid going
forward) but WRONG for which records get kept for training, since it also
silently discarded the one genuinely useful negative example per rejected
round (the rejected token itself, with its real low ratio). Every collected
record was, by construction, an already-accepted one -- the acceptance
head could only ever learn "always predict accept," which is exactly what
it did. **General lesson**: a data-collection loop built primarily to drive
correct GENERATION behavior (correct KV cache state, correct continuation)
can silently produce a biased TRAINING set if the same truncation logic
that's correct for generation gets reused unexamined for what to save --
these are two different concerns that happened to share one line of code.

**Fix**: stopped truncating `records` at all (`n_from_cand` still governs
the KV cache crop, correctly) -- the rejected position's record, when one
exists, now gets kept alongside the accepted ones. Recollected: 484 records
(up from 160), 429 direct + 55 ngram. **After the fix**: ratio labels
mean=0.3554, min=0.0000, max=1.0000 -- genuine spread across the full
range, real positive and negative examples now present. Mechanism smoke
test still passes: loss trend 0.554 -> 1.157 -> 0.562 -> 0.446 -> 0.260 over
5 steps on a 16-example sample (dip at step 1 is just a high-LR/small-batch
artifact on random init, not a real concern -- ends well below where it
started), predictions show real, if imperfect, correlation with true labels
even on this tiny smoke-test scale.

**Status against the build plan**: steps 1-3 done (data collection,
hybrid loss, acceptance head), each validated on real collected data with
a genuine backward-pass mechanism check, two real bugs found and fixed
along the way (both would have silently corrupted results if not caught).
Next: step 4, wire per-batch gradient updates into the actual live
generation loop (not just isolated-token smoke tests) -- this is where the
"online" part of online training actually gets tested for real.

---

## RESULT (2026-07-26, continued): online-training build, step 4 (live
loop) -- real OOM found and fixed, mechanism now running end-to-end during
actual generation

**First integration attempt OOM'd within the second question** (GPU 1,
14.56GB exhausted). Root cause, once diagnosed: `train_step` grouped a
question's records by originating round, ran a forward pass per round, but
deferred a single combined `backward()` call until AFTER looping through
ALL of a question's rounds -- meaning every round's full computational
graph (all layers' activations, since `output_hidden_states=True` was
needed for the acceptance head) stayed alive in memory simultaneously.
Peak memory scaled with (rounds per question) x (per-round activation
memory), not with any single round's cost -- with ~12 rounds per 60-token
question, that's ~12x more memory held open than necessary at any given
moment.

**Fix**: proper gradient accumulation -- call `.backward()` immediately
after each round's loss (scaled by `1/n_rounds` so the accumulated
gradient approximates the same mean-over-the-question objective a single
combined backward() would have given), explicitly `del` the round's
tensors and `torch.cuda.empty_cache()` before moving to the next round,
call `optimizer.step()` only once at the end. This is the standard
memory-efficient accumulation pattern -- only one round's graph is ever
alive at a time, regardless of how many rounds a question has.

**Confirmed working after the fix**: 3+ consecutive questions completed
cleanly with real metrics (tok/call, mean reverse-KL, mean NLL, mean BCE
per question), no OOM. Q0: kl=4.876 nll=0.953 bce=1.034. Q1: kl=3.860
nll=1.458 bce=1.507. Q2: kl=3.401 nll=2.778 bce=1.093 -- kl shows an early
downward trend across just 3 points, nll/bce fluctuate; far too few data
points to draw any conclusion yet, this is purely confirming the mechanism
runs stably, not evidence of learning. Full 40-question run in progress --
the actual per-question trend across the whole set is what task #19 (does
this measurably help without regressing) will assess.

**Pattern worth naming explicitly**: this is the second distinct real bug
in this build alone caused by reusing logic that's correct for ONE purpose
(a single combined backward() is the natural, simplest way to write a
training step) in a context where a DIFFERENT constraint (GPU memory,
scaling with how many rounds a live SD question happens to take) makes it
wrong. The acceptance-head data-collection bug (previous entry) and this
one share the same shape: code that's locally reasonable breaks under a
constraint that isn't visible from reading that code in isolation.

---

## CORRECTION (2026-07-26): full fine-tuning was the wrong methodology --
switched to LoRA, a real, worth-remembering error given the whole system's
own premise

**The mistake**: the live online training loop (previous entry) trained
every parameter of the 0.5B drafter -- full-parameter gradients plus
AdamW's momentum/variance buffers for all of them, ~2x model size in
memory on top of the model itself. **User pushback, correctly**: this
whole project's premise is the drafter runs ON-DEVICE (a phone) -- full
fine-tuning's memory profile is nowhere close to viable on real phone RAM,
regardless of anything measured on a T4. This directly contradicts the
system's own design goal, not just a minor inefficiency.

**Fix**: wrapped the drafter in LoRA (`peft` library, already installed)
instead of full fine-tuning -- base weights frozen, only small low-rank
adapters trainable, `target_modules=["q_proj","k_proj","v_proj","o_proj",
"gate_proj","up_proj","down_proj"]` (Qwen2's standard attention+MLP
projection names), r=8, alpha=16. **Confirmed**: 4,399,104 trainable
params out of 498,431,872 total (0.8826%) -- ~114x fewer trainable
parameters than full fine-tuning, the right order of magnitude for LoRA
and a genuinely on-device-plausible footprint. Optimizer updated to only
cover `requires_grad=True` parameters (LoRA adapters + acceptance head),
not the frozen base weights. First question re-verified working end to
end with the new setup: same-magnitude metrics as the full-fine-tuning run
(kl=4.876, nll=0.953, bce=0.950 vs the earlier run's 4.876/0.953/1.034 --
identical kl/nll since Q0's drafting behavior comes from the same
pretrained base weights either way, only the training methodology
differs).

**Note this doesn't replace the earlier gradient-accumulation memory fix --
it compounds with it.** LoRA reduces optimizer-state and gradient-storage
memory (only need buffers for the small adapter matrices, not all ~500M
base params), but backprop still has to flow THROUGH the frozen base
layers to reach the LoRA adapters, so activation memory during the
backward pass is not reduced by LoRA alone -- the per-round
backward()-then-release pattern from the previous fix is still necessary
and complementary, not made redundant by this change.

---

## RESULT (2026-07-26, continued): replication on OmniDraft's own
conditions REVERSES the earlier verdict -- online training measurably
helps on GSM8K, isolated with a proper train-vs-control comparison

**Why this was run**: the conversational-domain online-training run
(previous entries) showed real, consistent loss decreases but no
improvement in tok/call, the metric that actually matters. Before
concluding the mechanism doesn't help, correctly pushed to isolate "does
our implementation even work under conditions the paper validated" from
"does it help OUR specific domain" -- these are different questions, and
conflating them was the mistake in the earlier verdict.

**Setup**: matched the paper's own model pairing (Llama-68M drafter,
confirmed 30.4% suppression against Qwen2-7B-Instruct target -- Qwen2, not
Llama3-8B, to avoid Meta's HF gating as a real access risk) and domain
(GSM8K, their actual ablation benchmark) as closely as practical. Few-shot
plain-text prompting for the base drafter (same pattern as the earlier
n-gram cache work), chat-template prompting for the instruct target,
independent questions (no multi-turn history-chaining, matching how GSM8K
is normally evaluated). Ran a TRAIN condition and a NO-TRAIN control on
the IDENTICAL 40-question sequence, same order, same starting weights --
isolates the training effect from question-order/difficulty variance,
which the earlier conversational run couldn't rule out.

**Result**:

| condition | first-half avg tok/call | second-half avg tok/call | change |
|---|---|---|---|
| control (no training) | 1.158 | 1.164 | +0.5% (flat, noise) |
| train (online training) | 1.156 | 1.218 | **+5.4% (real)** |

Both conditions start nearly identically (Q0: 1.176 both; Q1-4 within
0.02 of each other), confirming a genuinely matched starting point. The
control shows no trend. The train condition shows a real, consistent
increase the control doesn't have. Reverse-KL also decreased in the train
run (6.533 -> 5.880 first-half-to-second-half, ~10%), consistent with the
conversational-domain run's loss behavior -- but this time the loss
improvement actually shows up in the downstream metric too.

**Verdict, corrected**: the earlier conclusion ("training-based adaptation
doesn't clearly help, matches the session's repeated pattern of trained
modifications losing to smart untrained choices") was too broad. It's
correct for open-ended conversational Q&A specifically -- confirmed not by
assumption but by direct replication showing the SAME mechanism, same
code, same LoRA setup, DOES work on a domain closer to what the paper
validated. This is now the same domain-dependence story already found
twice this session (n-gram cache / PLD both null on conversational Q&A,
both presumably real on repetitive/structured domains) -- online
training-based drafter adaptation belongs in the same bucket: real, but
concentrated in structured, repetitive-pattern domains (math, code,
GSM8K-like tasks), not general open conversation. Not "OmniDraft is wrong
for us" -- "OmniDraft's training benefit, like its n-gram cache benefit,
is domain-gated, and our target deployment domain (conversational
handoff) sits on the wrong side of that gate for THIS specific lever,
while still benefiting fully from the untrained mechanism (direct mapping
+ greedy verification, the validated 1.394x)."

---

## LITERATURE (2026-07-26): "Online Speculative Decoding" (OSD, Liu et
al., ICML 2024, arxiv 2310.07177) -- confirms user's proposed direction,
and exposes a real architecture mistake in what we built

**Trigger**: user proposed using the target model's outputs/distributions
as continuous training signal in BOTH naive-handoff and SD-verified
turns, cross-vocab-adapted, referencing "online speculative decoding"
papers specifically. Researched the actual paper this refers to.

**Confirmed from the paper's own framing and GitHub README**:
- OSD trains the draft model SERVER-SIDE, not on the edge device. Their
  stated justification: LLM serving is memory-bandwidth-bound, so the
  serving cluster has idle compute sitting around during normal serving
  -- that idle compute is repurposed for continuous draft-model training,
  which is why they can call it effectively cost-neutral. The edge/client
  side does no training compute in their design at all.
- Framed as online knowledge distillation: target = teacher, draft =
  student, continually updated on observed live query traffic to track
  distribution shift.
- Reported result: 1.42x-2.17x latency reduction from this continuous
  adaptation vs. a static draft model.
- Cross-vocabulary draft/target pairs: NOT addressed. Confirmed absent
  from available docs (paper text extraction failed on two attempts;
  GitHub README explicitly does not cover it). This is a genuinely open
  gap, not something to route around -- consistent with the edge-cloud SD
  literature gap already noted (DSSD/ConfigSpec/PipeSD all stand in GPUs
  for "edge," nobody in the literature we've found runs real heterogeneous
  edge hardware with cross-vocab pairs and online adaptation together).

**Why this matters -- we likely built the wrong architecture**: the
online-training system built earlier this session (data collection,
hybrid loss, acceptance head, live loop, LoRA) assumed the DRAFTER trains
ITSELF on-device, which is why LoRA was necessary at all (full
fine-tuning memory cost is infeasible on real phone RAM -- the correction
that triggered the LoRA switch). OSD's validated real-world pattern is
different: the CLOUD SERVER does the training (it already has spare
compute, and it already runs the target's full forward pass every turn
regardless of routing decision), then periodically PUSHES updated LoRA
adapter weights DOWN to the device -- the same shape as any ordinary
app/model update, not continuous on-device compute. Under that design the
phone never spends battery or thermal budget on training, only on
occasional small (LoRA-sized) downloads plus normal inference. This is a
strictly better fit for the project's own on-device-deployment premise
than what we built, not just an alternative.

**Answer to the user's specific question (train on naive-handoff turns
too, not just SD-verified ones)**: yes, and it falls out naturally once
training moves server-side. Every turn the target ever handles -- whether
the router chose naive text handoff or full SD verification -- already
requires the server to compute the target's real output. That's free
training signal in both cases. The cross-vocab translation and hybrid-loss
machinery already built (direct_map + n-gram-cache translation,
reverse/forward-KL on directly-mapped tokens, NLL on merged tokens)
carries over unchanged -- what changes is WHERE it runs (server, not
device) and WHICH turns contribute data (all routed turns, not only
SD-verified ones).

**Not yet resolved from available sources**: OSD's exact update trigger/
batching cadence (per-request? per-N-requests? time-windowed?), and
whether they use non-speculative/fallback generations as training signal
at all in their own design (our proposed extension goes beyond their
published scope on this point, not just the cross-vocab dimension).

**Status (superseded below)**: the server-side shadow-copy idea was
retracted the same day -- see CORRECTION entry below. It directly
reintroduced the server compute cost the whole hybrid architecture exists
to avoid (OSD's "train where it's already running" logic only works
because their drafter is colocated with the target; ours deliberately
isn't). Corrected design: training stays on-device (LoRA, as already
built), extended to ALL routed turns (naive-handoff + SD, not just SD
rounds) using whichever signal that turn already carries for free --
target logits on SD-verified turns (already transmitted for the
accept/reject step), target realized text on naive-handoff turns (already
transmitted as the answer; trained via NLL, which is exactly forward KL's
degenerate case when only a single realized sample is observed, not a
different/cruder objective). Zero added network cost, zero added server
compute, in either branch.

---

## RESULT (2026-07-26): forward-KL vs reverse-KL on the conversational
domain -- forward KL does NOT rescue it either; slightly worse if anything

**Setup**: identical to the earlier conversational-domain online-training
run (Qwen2.5-0.5B-Instruct drafter / Gemma-2-9b-it target,
mixed_complexity_dataset, LoRA, live loop, 40 questions) except
`reverse_kl_direct` recomputed as forward KL `KL(p_target || q_drafter)`
instead of reverse KL `KL(q_drafter || p_target)`, per DistillSpec's
finding that greedy decoding is better served by forward KL/JSD than
reverse KL.

**Result** (first-half vs second-half of the 40 questions, n=20 each):

| metric | first-half avg | second-half avg | change |
|---|---|---|---|
| tok/call | 1.438 | 1.263 | **-12.2% (worse)** |
| forward KL loss | 1.902 | 1.544 | -18.8% (loss down) |
| NLL (n-gram tokens) | 2.279 | 1.253 | -45.0% (loss down) |
| BCE (acceptance head) | 0.807 | 0.881 | +9.1% (worse) |

**Comparison to the earlier reverse-KL run on the same domain/pairing**:
reverse KL gave tok/call 1.489 -> 1.431 (-3.9%, roughly flat/slight
decline) with kl/nll/bce all decreasing ~10%/~40%/~9%. Forward KL shows
the same qualitative pattern -- losses decrease cleanly, tok/call does not
follow, in fact declines more than reverse KL did on this run.

**Verdict**: forward KL does not rescue the conversational domain. The
loss-function choice (forward vs reverse KL) is not the variable gating
the conversational-domain null result -- both directions show the same
disconnect between decreasing training loss and flat-or-declining
downstream tok/call. This sharpens the domain-gating hypothesis: the gate
is the DOMAIN (structured/repetitive vs. open conversational), not which
divergence was used to train on it. Reinforces that the next real test is
data volume/coverage (all-routed-turns signal, not SD-only) rather than
further loss-function tuning, before concluding online training is simply
inapplicable to conversational handoff.

---

## RESULT (2026-07-26): all-turns training (naive-handoff NLL + SD hybrid
loss, alternated by question) -- worse than SD-only, and reveals objective
interference, not just insufficient signal

**Setup**: `online_train_live_allturns.py`. Same conversational domain/
pairing, LoRA, on-device, 40 questions. Even qi -> SD-routed (existing
hybrid KL+ngram-NLL+BCE loss). Odd qi -> naive-handoff-routed (target
autoregressively generates the full answer, no drafter involved; drafter
trained via plain NLL against the target's realized text, re-tokenized in
the drafter's own vocab -- no cross-vocab alignment machinery needed for
this branch, since it's sequence-level not token-level distillation).
Both branches already carry this signal for free in the real protocol (SD
transmits target logits for accept/reject; naive handoff transmits the
final text as the answer itself) -- zero added network/server cost.
Training after every question either way, one gradient step per question,
alternating which loss trains the shared LoRA weights.

**Result** (SD-routed questions only, n=20, comparable to earlier SD-only
baselines):

| metric | first-half | second-half | change |
|---|---|---|---|
| tok/call (SD) | 1.585 | 1.460 | **-7.9% (worse)** |
| KL (direct-mapped) | 4.029 | 4.047 | +0.5% (flat) |
| n-gram NLL | 1.994 | 2.403 | **+20.5% (worse)** |
| BCE (acceptance head) | 0.919 | 0.850 | -7.5% (better) |
| naive-handoff NLL | 1.630 | 1.547 | -5.1% (better) |

**Verdict**: does not confirm "more signal helps." The critical new signal
is that KL and n-gram NLL -- which reliably decreased in EVERY earlier
SD-only run (reverse-KL, forward-KL) -- stopped improving here, one even
got worse. That's a different failure mode than before: not "loss goes
down but tok/call doesn't follow" (the earlier domain-gating story), but
"the SD-branch loss itself stopped improving." Most likely cause:
alternating two DIFFERENT training objectives on the SAME shared LoRA
weights via SEPARATE zero_grad()/backward()/step() calls -- each
naive-handoff update likely partially undoes what the previous SD update
just nudged the direct-mapped-token behavior toward. This is standard
multi-task interference from alternating (not jointly combined) gradient
steps on shared parameters, not evidence that naive-handoff data itself
carries no useful signal. A fair test of the original hypothesis would
need to pool both loss streams into ONE joint backward pass per question
(or per batch of questions) rather than alternating separate steps --
not yet built. One clean positive: BCE (acceptance head) improved here
too, consistent with it being the one metric that has now improved in
every online-training variant tried on this domain (SD-only reverse-KL,
SD-only forward-KL, and this alternated run) -- see the adaptive-draft-
length result below, which acts on exactly that signal.

---

## RESULT (2026-07-26): adaptive speculation-length controller (acceptance
head as real-time controller, not a training target) -- worse than fixed
window, but the diagnosis points to a fixable threshold/label-mismatch
bug, not a dead mechanism

**Motivation**: three separate online-training attempts (reverse-KL,
forward-KL, all-turns) all failed to improve tok/call on this
conversational domain despite the drafter's training losses moving in the
expected direction. The one signal that reliably improved across all
three was the acceptance head's own BCE loss. Pivoted from "improve the
drafter" to "use the head's existing signal better" -- specifically, as a
live controller deciding how many tokens to draft per round, instead of a
fixed window=5 every time. Deliberately decoupled from the online-training
thread: drafter frozen throughout (no LoRA, requires_grad=False on all
drafter params), only the acceptance head has trainable parameters, so
this isolates "does smarter use of an already-learnable signal help" from
"does retraining the drafter help" (already answered: no, on this domain).

**Setup**: `adaptive_draft_length.py`. Phase A (priming): topics 0-2 (24
questions), standard fixed-window=5 SD collection, head-only BCE training
against the continuous ratio min(1, p/q) on direct-mapped tokens (same
formula as before, just no KL/NLL/drafter gradient attached). Phase B
(eval): topics 3-4 (16 questions, no overlap with priming), run TWICE with
the frozen post-priming head, no further training in either pass -- once
fixed window=5 (baseline), once adaptive (token-by-token drafting,
scoring each token's hidden state with the head, stopping when predicted
accept-probability < 0.5 after a minimum of 1 token, max 8).

**Result**:

| condition | tok/call (16 eval questions) |
|---|---|
| fixed window=5 | 1.388 |
| adaptive window | 1.301 |
| change | **-6.3% (worse)** |

**Diagnosis**: the adaptive controller chose an average window of only
~1.72 tokens per round, against the fixed 5 -- it's stopping almost
immediately almost every round, which alone caps tok/call well below what
a window of 5 can reach, independent of whether any individual stopping
decision was well-calibrated. Likely root cause: the head was trained (per
OmniDraft's own recipe) on the CONTINUOUS ratio min(1, p/q) via BCE, then
gated with a flat 0.5 threshold on its sigmoid output to decide
continue-vs-stop. Those are not the same target under greedy verification
-- the actual accept event is binary (does the drafter's argmax match the
target's), but the ratio label can sit well below 0.5 even for tokens
whose argmax DOES match (a drafter can get the top token right while being
much less peaked than the target). So 0.5-on-the-ratio is likely much too
conservative a proxy for "will this be a greedy-verification match."
Priming's own BCE trace (0.87 -> ~0.75-0.97 range over 24 questions) also
suggests only partial convergence in this shorter warm-up window.

**Verdict**: does not (yet) demonstrate the adaptive-window idea works,
but the failure mode is a calibration/label-mismatch bug, not evidence
against the mechanism. Two concrete, not-yet-tried fixes: (1) calibrate
the stopping threshold empirically against real match rate instead of
assuming 0.5; (2) retrain the head on the BINARY `matched` label (already
collected in every record) instead of the continuous ratio, which is more
directly aligned with what the controller is actually deciding, at the
cost of diverging from OmniDraft's literal recipe (built for general,
possibly-sampling SD, not specifically calibrated for a greedy-only
stopping decision). Decision pending with user on which to try next.

---

## CORRECTION (2026-07-26): the "+53.9% training-free confidence-adaptive
draft length" result was ENTIRELY a token-selection confound, not a real
effect -- corrected sweep shows a flat null, same as everything else on
this domain

**What happened**: after the acceptance-head controller failed, searched
the literature (AdaEDL, GammaTune+, CM-ASD, AdaSpec, DSpark) and built a
training-free version instead -- threshold the drafter's own top-softmax
probability to decide when to stop drafting, no training, no priming.
First sweep: threshold=0.3 gave 2.137 tok/call vs fixed window=5's 1.388
(+53.9%), and a control (plain fixed window=8, no adaptivity) gave 1.3879
-- ruling out "just a bigger cap" as the explanation. This looked like the
strongest, cleanest result of the whole online-mechanisms exploration and
was reported and persisted to memory as such.

**The bug**: extending the sweep to lower thresholds (0.05-0.25) to
bracket the peak surfaced a red flag: threshold=0.05 (avg_window~=7.95,
i.e. the confidence gate almost never fires) should behave almost
identically to the fixed window=8 control, since both are ~always
drafting the full window. It didn't -- 2.24 tok/call vs the control's
1.39, a huge gap between two conditions that should have been nearly the
same policy. Root cause found on inspection: `draft_fixed` (used for the
fixed=5 baseline AND the fixed=8 control) calls `.generate()` with
`repetition_penalty=1.3, no_repeat_ngram_size=3`, and implicitly stops at
EOS. `draft_confidence_adaptive` (the hand-rolled token-by-token loop)
had NONE of these -- pure unconstrained argmax every step, no EOS check
at all. So "fixed" and "adaptive" were never actually isolating the
stopping-criterion variable; they were two different token-selection
policies entirely, and the entire apparent gain was almost certainly the
adaptive loop's lack of repetition suppression outperforming the fixed
loop's repetition-penalized drafting on this domain, dressed up to look
like a stopping-policy effect.

**Fix**: added the same `RepetitionPenaltyLogitsProcessor(1.3)` +
`NoRepeatNGramLogitsProcessor(3)` (via HF's own `LogitsProcessorList`, not
a reimplementation) plus explicit EOS-stopping to the adaptive loop, so
the ONLY variable differing between conditions is actually the stopping
decision, as originally intended.

**Corrected result** (same held-out eval set, topics 3-4, 16 questions):

| threshold | tok/call | avg window | change vs fixed=5 (1.388) |
|---|---|---|---|
| 0.05 | 1.407 | 7.95 | +1.4% |
| 0.1 | 1.394 | 7.32 | +0.5% |
| 0.15 | 1.387 | 6.22 | -0.1% |
| 0.2 | 1.373 | 4.88 | -1.0% |
| 0.25 | 1.350 | 3.84 | -2.7% |
| 0.3 | 1.364 | 3.40 | -1.8% |

Confirms the fix: threshold=0.05 (avg_window~8) now lands at 1.407, close
to the fixed-window-8 control's 1.3879, as it should if both are ~always
drafting the full window. Every threshold across the whole swept range is
within ~3% of the fixed baseline -- flat, noise-level, no real effect
anywhere.

**Verdict, corrected**: there is no real benefit from confidence-threshold
adaptive draft-length stopping on this domain, once the token-selection
policy is properly matched between conditions. The earlier +53.9%/+62.5%
numbers are RETRACTED -- not "smaller than reported," genuinely zero. This
closes out BOTH adaptive-draft-length attempts (acceptance-head-based:
negative, due to calibration mismatch; confidence-threshold-based: null,
once correctly controlled) as unhelpful on this domain, joining n-gram
cache, PLD, and online training (all configurations) in the same
domain-gated-null bucket. The flagship direct-mapping + greedy-verification
result (1.394x) remains the only lever that has robustly helped on this
specific conversational deployment domain -- every additional mechanism
tried this session (quantization, n-gram cache, PLD, online training in
three configurations, both adaptive-window approaches) has been null or
negative here, though several (online training, n-gram cache) are
confirmed real on other domains (GSM8K, repetitive/structured content).
Also corrects two memory entries written before this fix was found
([[edge-deployment-constraint]], [[tok-per-call-vs-wallclock]]) that
treated +53.9% as a promising lead awaiting wall-clock verification --
there is nothing left to verify; the effect was the bug.

---

## RESULT (2026-07-26): training-free confidence-threshold draft-length
controller -- REAL, SUBSTANTIAL win (+53.9% tok/call), confirmed by a
clean control

**Decision context**: asked to look at adjacent literature before
choosing a next move. Searched specifically for adaptive/dynamic
speculation-length techniques. Finding: the established approaches
(AdaEDL arxiv 2410.18351, GammaTune+, CM-ASD, AdaSpec, DSpark, SGLang's
adaptive mode) converge on using the DRAFTER'S OWN native confidence/
entropy at generation time as the stopping signal -- explicitly
training-and-parameter-free, sidestepping exactly the label/threshold
mismatch that broke the acceptance-head attempt above. Caveat surfaced
and respected: AdaEDL's own formal entropy-lower-bound derivation is
stated to apply to SAMPLING-based decoding specifically (built on the
general rejection-sampling accept formula) and does not extend to greedy,
which is what this system uses. Did not port that exact formula for that
reason. Used the simpler, decoding-strategy-agnostic sibling instead:
threshold on the drafter's own raw top-token softmax probability, same
style as GammaTune+/CM-ASD. Justification for skipping a formal
probabilistic guarantee: under greedy verification a bad stopping
decision only costs efficiency, never correctness (accept/reject is still
exactly argmax-match regardless of window length), so an empirically
reasonable threshold is sufficient -- hence a small sweep instead of a
derived bound.

**Setup**: `confidence_adaptive_draft_length.py`. No training, no priming,
no head at all -- drafter frozen (as in the prior attempt), pure
inference-time heuristic. Same held-out eval set as the failed head-based
attempt (topics 3-4, 16 questions) for direct comparability. Drafts one
token at a time; after each token, checks the drafter's own softmax max
probability; stops (after a minimum of 1 token) once that probability
drops below a threshold, or after MAX_WINDOW=8. Swept threshold in
{0.3, 0.5, 0.7} against a freshly-recomputed fixed window=5 baseline in
the same run (not reusing the number from the earlier script, to avoid
any cross-run confound).

**Result**:

| condition | tok/call (16 eval questions) | avg window | change vs fixed=5 |
|---|---|---|---|
| fixed window=5 (baseline, this run) | 1.388 | 5.0 | -- |
| adaptive, threshold=0.3 | **2.137** | 4.48 | **+53.9%** |
| adaptive, threshold=0.5 | 1.827 | 2.48 | +31.6% |
| adaptive, threshold=0.7 | 1.485 | 1.63 | +7.0% |

Fixed baseline reproduced 1.3880 (identical to the earlier script's
1.3880 on the same eval set), confirming no drift between runs.

**Control run -- ruling out the obvious confound**: MAX_WINDOW=8 is
larger than FIXED_WINDOW=5, so before crediting the adaptive mechanism,
needed to rule out "the gain is just from raising the cap." Ran
`fixed8_control.py`: plain fixed window=8, no adaptivity at all, same
eval set. Result: **1.3879 tok/call -- statistically identical to fixed
window=5's 1.3880.** A bigger cap alone does nothing on this domain. This
confirms the entire +53.9% gain at threshold=0.3 is genuinely coming from
the confidence-based early-stopping mechanism itself, not from window
size.

**Verdict**: the strongest, cleanest result of the entire online-
mechanisms exploration this session -- bigger than every trained
modification tried (drafter retraining in any loss configuration,
acceptance-head-based control), training-free (zero calibration risk,
zero priming cost, deployable immediately), and confirmed via a proper
control rather than taken at face value. Directly fits the session's
repeated pattern: smart untrained choices beat trained modifications on
this domain -- this is now the clearest instance of that pattern, not
just the mildest one. Threshold=0.3 (most lenient -- only stop when
genuinely very uncertain) is the best of the three swept; not yet tested
whether an even more lenient threshold (e.g. 0.2, 0.15) keeps improving
or peaks and reverses. Not yet validated: whether this holds on the
priming topics (0-2) too, on the full 40-question set, or combined with
the flagship cross-vocab direct-mapping result (1.394x) as a multiplier
rather than tested in isolation with a fresh translator/model load.
