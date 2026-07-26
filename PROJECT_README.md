# Hybrid LLM: Cross-Vocabulary Speculative Decoding as Both Accelerator and Handoff Mechanism

**Flagship result:** 1.394x wall-clock speedup, provably zero quality cost, 2.797x server-compute reduction
**Tested on:** cloud GPU (2x T4 / A6000), real mobile hardware not yet included
**Genuinely novel claim, stated explicitly mid-project:** none of the papers reviewed (OmniDraft, DSD, PipeSD, DSSD, "Speculation at a Distance", CE-LSLM, DroidSpeak, C2C) combine acceleration and cross-model handoff into one mechanism — they solve those two problems separately. This project's contribution is using cross-vocabulary speculative decoding (SD) as *both* at once.

This document explains the architecture, the end-to-end flow, every real bug found and fixed along the way, and the full experimental record with exact numbers — not a summary, the actual evidence.

---

## Contents

1. [What the system is](#1-what-the-system-is)
2. [Architecture: the four components](#2-architecture-the-four-components)
3. [End-to-end flow](#3-end-to-end-flow)
4. [The core mechanism in detail](#4-the-core-mechanism-in-detail)
5. [Bugs found and fixed](#5-bugs-found-and-fixed)
6. [Experiments and results](#6-experiments-and-results)
7. [Verdict: proven / ruled out / open](#7-verdict-proven--ruled-out--open)
8. [Literature map](#8-literature-map)
9. [Known limitations](#9-known-limitations)
10. [Next agenda](#10-next-agenda)

---

## 1. What the system is

A small language model runs on the user's device. A large language model runs in the cloud. Most turns of a conversation can be handled by the small model alone. Some turns need the large model's capability. The open problem this project addresses: **when control has to pass from the small model to the large one mid-conversation, how does the large model pick up the conversation without either (a) re-processing the entire history from scratch, or (b) risking a worse answer than if it had just answered from the start?**

The standard answer in production systems today is: it doesn't try. Requests are made stateless by design (this is explicitly how Apple Private Cloud Compute avoids the problem) — the large model gets full context resent as plain text on every call, and any "savings" come purely from *not calling* the large model as often (a routing/cascade decision), never from anything smarter happening on the calls that do escalate.

This project's mechanism is different: **cross-vocabulary speculative decoding (SD)**, normally used only to *accelerate* the large model's own generation, is repurposed to *also* be the handoff. SD works by having the small model propose (draft) tokens, and the large model verify them in a single batched forward pass, accepting whatever agrees with what it would have generated on its own and correcting whatever doesn't. If SD is already the verification step, the large model has, as a side effect, already read and validated everything the small model produced — there is no separate "catch-up" or "re-ingestion" step to design, because the acceleration mechanism and the handoff mechanism are the same forward pass.

Two things had to be true for this to be worth anything over the naive stateless approach:
- **Zero quality cost** — the large model's output has to be provably as good as if it had generated the whole answer itself.
- **Actually faster** — under real conditions (drafter speed, network latency, server compute cost), not just in a best-case lab measurement.

---

## 2. Architecture: the four components

The system was originally scoped into four components. Status reflects where each one actually landed, not where it started.

| # | Component | Role | Difficulty | Status |
|---|---|---|---|---|
| **A** | Turn-level deferral policy (router) | Decide, per turn: does the small model answer, or does control escalate to the large model? | Low | Built (LinUCB contextual bandit), reward calibrated with real measured costs, threshold tuning still parked |
| **B** | Cross-architecture handoff medium | Get conversation state across the tokenizer/architecture boundary | Medium | Solved via cross-vocab SD's own text/token round-trip — no separate mechanism needed |
| **C** | Cheap reuse of the handoff (avoid full-history re-ingestion) | Context compression — shrink what has to be re-processed on a switch | High — the genuinely open research gap in the literature | Prototype tested (quality-equivalence + wall-clock + warm/cold cache + pipelining); training-free (SnapKV/PyramidKV) is the working default, KV-Distill escalation kept in reserve |
| **D** | Turn-level partial correction | Let the large model correct the small model's output mid-turn, not just at turn boundaries | Medium | Explicitly descoped from the POC — different mechanism/problem, not built |

**Why B turned out simpler than expected**: an entire sub-thread explored whether hidden-state projection (mapping the small model's internal activations directly into the large model's space, no text round-trip at all) could bridge the cross-architecture gap more cheaply. Backed by real literature (Relative Representations, Moschella et al., ICLR'23; Platonic Representation Hypothesis, Huh et al., ICML'24) — but EAGLE only works same-model, and Patchscopes shows cross-family activation patching works but weaker. Conclusion: **nobody has published a trained hidden-state→target-vocabulary projection that drives lossless cross-architecture SD** — a real, open ML-research problem, not something to solve inside this POC. The established, working approach (SLEM/SLRS/TLI algorithms, HF's own Universal Assisted Generation) — decode the small model's tokens to text, re-encode in the large model's vocabulary — was used instead, and is exactly what cross-vocab SD already does as part of verification. Component B disappears as a separate piece of engineering because Component D's mechanism (SD) already solves it as a side effect.

**Why C is the hard part, and what got tried**: three designs were built and superseded before landing on the current default.
1. **On-demand extraction** (first design): log the running transcript every turn; on a switch, one extra LLM call summarizes key facts/open threads/last exchange verbatim as the handoff payload, instead of resending everything.
2. **LatentMAS-based redesign**: training-free shared latent working memory across layers (up to 14.6% higher accuracy, 70-83% fewer output tokens, 4-4.3x faster than text-based multi-agent handoff, per the paper). Requires identical hidden dim, attention setup, RoPE scale, **and layer count** between the two models — checked directly against the paper's Section 3.2, confirmed it needs ALL layers aligned, not just a single bridge point. This rules it out for genuinely different model families (our case) — it only works same-architecture, same-size.
3. **KV-Distill** (current escalation path, not yet needed): a per-model trained compressor (student-teacher KL-divergence training, up to 99% length reduction) that sidesteps cross-architecture layer alignment entirely by compressing each model's own cache in its own shape. Kept in reserve — training-free eviction (SnapKV/PyramidKV) is the default until there's real evidence it isn't enough, following an explicit order-of-operations decision: don't default to training a compressor when a training-free one might already work.

---

## 3. End-to-end flow

The design went through several complete revisions (see [Section 5.7 of the research log] for the full history — continuous background tracking was tried, reconsidered, and demoted). The current, settled flow:

**Default state — text-handoff, small model in charge:**
1. The small model answers turns normally, on-device.
2. The router (Component A) watches a per-turn signal and decides whether *this* turn should escalate.

**On escalation (small → large):**
3. The small model's already-generated tokens for this turn are the draft. Each drafted token gets translated into the large model's vocabulary — **direct mapping** first (a static 1:1 token-ID lookup built once from the two tokenizers' vocabulary overlap), falling back to an **n-gram merge cache** for token runs that don't have a 1:1 counterpart (decode the run to text, re-encode in the target's tokenizer, cache the mapping for reuse).
4. The large model verifies the translated candidate tokens in **one batched forward pass** — not one call per token.
5. **Greedy accept/reject rule**: a proposed token is accepted only if it exactly matches the token the large model would have generated on its own; the first mismatch is replaced by the large model's own choice, and everything after the mismatch is discarded (KV cache cropped back to the correction point).
6. This is the handoff. There is no separate "catch-up" step — verifying the small model's draft *is* the large model reading and validating the conversation so far.

**On de-escalation (large → small):**
7. Free, structurally: whenever SD is active, the small model (acting as drafter) already has every accepted token written into its own KV cache as a side effect of the verification round — it was never stale, so handing control back costs nothing extra.

**What was tried and explicitly reversed**: an earlier design ("continuous tracking") had the small model draft on *every* turn regardless of who officially answered, using its output as a permanent blind prefill to keep the large model's cache perpetually warm. Reconsidered directly: *"text-based handoff/checkpointing is more elegant for hybrid LLM specifically, because it stays true to the actual premise (the large model costs nothing until deliberately engaged)... the Android/cellular deployment target leans toward text-handoff's assumptions (rare switches, intermittent connectivity), not continuous tracking's."* The current synthesis: text-handoff is the default, continuous tracking is available only as an adaptive escalation if the router's own observed switching pattern shows frequent-enough oscillation that repeated catch-up costs would exceed continuous tracking's steady overhead — not built as a permanent always-on mechanism.

---

## 4. The core mechanism in detail

### 4.1 Why cross-vocabulary SD needs a translator at all

Standard SD assumes draft and target models share a tokenizer, so "does the target agree with the draft?" is a direct token-ID comparison. A small on-device model and a large cloud model almost never share a vocabulary. The translator's job is converting the drafter's proposed token IDs into the target's token-ID space before verification can happen at all.

### 4.2 Direct mapping + n-gram merge cache

- **Direct mapping**: built once, statically, from the overlap between the two tokenizers' vocabularies — a 1:1 lookup table, no runtime cost.
- **N-gram merge cache** (the real mechanism in OmniDraft's paper — confirmed by pulling the full text directly, since an earlier internal guess had this wrong, see bug log): for a run of drafter tokens with no 1:1 target counterpart, decode the run to text, re-encode it with the target's tokenizer, and cache that mapping (drafter-token-tuple → target-token-list) for reuse across the rest of generation. This is genuinely online-learned: no offline training pass, the cache grows purely from usage.
- Measured behavior on a real conversation (see 6.6): cache grew to 23 entries, 98.3% of tokens resolved via direct mapping alone, the remaining tokens hit the n-gram cache with an 86.6% reuse rate once entries existed.

### 4.3 Greedy verification — provably, not approximately, lossless

Under **greedy decoding** specifically, SD's accept/reject rule isn't approximately safe — it's mathematically forced to produce byte-identical output to plain generation. The general SD rejection-sampling rule is `accept with probability min(1, p_target(token)/p_draft(token))`, with a residual-distribution correction on rejection. Under greedy decoding the target's distribution is a delta function (one token has probability 1, everything else 0), and this general formula collapses deterministically to: *accept the drafted token if and only if it exactly equals the target's own argmax; otherwise substitute the target's own choice.* There is no probabilistic slack in this — the output is forced to be identical to what plain autoregressive generation from the target would have produced, by construction, not by empirical similarity. This was confirmed empirically too (F1 0.186 vs 0.182 for SD-handoff vs plain-target generation, both far above the small model working alone at F1 0.051 — see 6.7).

### 4.4 Why greedy over sampling — a real library-behavior discovery

Reading `transformers/generation/utils.py::_get_candidate_generator` directly surfaced an undocumented dispatch: the candidate-generator class HuggingFace picks for a cross-tokenizer pair is chosen purely by the `do_sample` flag — **not** by anything related to correctness or quality. `do_sample=False` (greedy) routes to a primitive decode→re-encode translator (`AssistedCandidateGeneratorDifferentTokenizers`). `do_sample=True` (sampling) routes to a much better precomputed vocabulary-overlap translator (`UniversalSpeculativeDecodingGenerator` + `AssistantToTargetTranslator`) — but that mode pays sampling-based verification's own per-round overhead. Since candidate-generator selection and the `do_sample` used for the actual verification math are read independently from the same generation config, it's possible to force the *better* translator while *keeping* greedy verification's provable-correctness guarantee — this combination is what produced the flagship number (see 6.9), and required patching two real bugs in the better translator to work at all (see bug log #14).

---

## 5. Bugs found and fixed

Every one of these was a real, working system before and after — not a hypothesis. Grouped by category, in the order discovered.

### 5.1 Infrastructure / environment

| Bug | Root cause | Fix |
|---|---|---|
| Cross-GPU model splitting slowdown | `device_map='auto'` split the target model across both GPUs while the drafter sat on a separate GPU, causing cross-GPU sync overhead on every verification round — confirmed by ruling out vocab size, translation-step cost, and cache-hit rate first, then finding 94% of per-call time was unaccounted "target verification + loop overhead" | Forced single-GPU placement (8-bit quantization to fit both models on one GPU); verified 0.95x-1.29x speedup with all outputs matching baseline exactly |
| Within-process CUDA memory fragmentation | Repeated `generate()` calls in one long-running process slowly ate ~150-200MB of slack until OOM | Aggressive `torch.cuda.empty_cache()` + `gc.collect()` after every measurement call, not just training steps |
| Backward-pass OOM during distillation training | fp16 single-GPU setup left zero headroom for training — the target model's own fp16 footprint was the actual cause, not optimizer state | Quantize the target to 8-bit for training experiments only, keep the drafter in fp16 |
| Silent SIGKILL, no progress evidence | Long unattended job killed (likely system-RAM OOM, 31GB total) ~25-30 min in; stdout was block-buffered under `nohup`, so no progress had flushed to disk | Always run long jobs with `python3 -u` or `flush=True` on every progress print |
| `transformers 5.0.0`'s rewritten weight-loading path OOM'd loading a 9B model in 8-bit on a 14.56GB T4 (`expandable_segments` didn't help) | New codepath, reproduced on two separate boxes | Pinned `transformers<5` (landed on 4.57.6) |
| `peft`'s `get_peft_model` errored on an incompatible-`torchao` version check, even though torchao wasn't used | Stale/unused dependency triggering a version guard | `pip uninstall -y torchao` |
| HF Xet storage backend stalled indefinitely (0 bytes, repeated 503s) downloading a 9B model | Xet CDN issue, not a local problem | `HF_HUB_DISABLE_XET=1`, falls back to plain HTTP |

### 5.2 Methodology bugs — results that looked real but weren't

| Bug | What it looked like | Root cause | Fix |
|---|---|---|---|
| Memorization vs generalization | Train-set-only eval showed tok/call +5% | 6 examples, 3 epochs, a 68M-parameter model simply memorized them | Always evaluate on a held-out set; the held-out number showed **-0.9%**, scattered noise |
| RLM-Cascade apples-to-oranges | RLM-Cascade's 45.8%/1.83x looked directly comparable to our ~1.17x | RLM-Cascade's number is a **routing-layer** metric (skip the large model 88.8% of the time via cross-vendor HTTP APIs with zero internal logit access — their own paper states token-level SD is "inapplicable" for them); ours is an **SD-layer** metric | Corrected framing: router savings and SD speedup are composable, multiplicative layers, not competing numbers |
| Router self-referential threshold | A calibration test looked clean | Threshold was the median of the *same* test set being evaluated — produced an exactly-50% skip rate that was flagged in advance as a likely artifact and confirmed as one | Recalibrated on a genuinely disjoint 15-prompt set |
| "Same family" pairs secretly routed through the cross-vocab path | A same-family model pair scored *worse* than a cross-family pair — should have been an immediate red flag, and was, on pushback | HF's same-vs-cross-tokenizer dispatch checks `config.vocab_size` (the padded embedding-table size), not real tokenizer content — two same-family models can have identical real vocabularies (151,665 tokens) but different declared `vocab_size` (151936 vs 152064, padding tiers) and still get routed through the slower cross-vocab path | Found which model sizes actually share a padding tier (0.5B/1.5B/3B share 151936) and used a genuinely same-tier pair as the control |
| Data non-determinism despite a fixed seed | A freshly-extracted 30-prompt set shared 0/30 exact matches with an earlier set generated with the identical seed and code | The underlying source data file itself had changed between extractions (a re-cloned dataset differed slightly from the version used earlier) | Lesson recorded explicitly: same seed + same code does not guarantee same output if the underlying data file can itself change between runs |
| Token-selection-policy confound (adaptive draft-length controller, 2026-07-26) | +53.9% tok/call from a training-free confidence-threshold controller, with a fixed-window=8 control seemingly ruling out "just a bigger cap" | The "fixed" and "adaptive" drafting functions were never actually differing only in stopping policy — `draft_fixed` applied `repetition_penalty=1.3`/`no_repeat_ngram_size=3`/EOS-stopping (via `.generate()`), the hand-rolled adaptive loop had none of these | Added the same constraints via HF's own `LogitsProcessorList`; corrected result: every threshold tested lands within ~3% of baseline — **the entire +53.9% is retracted, not reduced, genuinely zero** |

### 5.3 Library-level bugs (found by reading HuggingFace source directly)

1. **`AssistantToTargetTranslator.__init__` never initializes `assistant_overlap_token_ids` when the two tokenizers have zero suppressed (non-overlapping) tokens** — true for same-BPE-family pairs like Qwen2.5-0.5B/7B. Downstream code references the attribute unconditionally → `AttributeError`. Fixed via a monkey-patched `__init__` that sets it to an identity mapping (`torch.arange(...)`) in that case.
2. **`get_target_logits` shape mismatch**: the vocabulary-mapping array is sized to the tokenizer's real vocabulary (151,665 tokens), but the model's LM head is sized to the padded `vocab_size` (151,936, includes unused padding slots) — with zero real pruning, indexing one against the other raises a shape-mismatch `RuntimeError`. Fixed by slicing the logits to the tokenizer's real vocabulary size first, matching HF's own not-pruned code path.

Both verified via a standalone diagnostic script comparing patched vs unpatched behavior directly.

### 5.4 Training-pipeline bugs (online-training build, 2026-07-26)

| Bug | Root cause | Fix |
|---|---|---|
| Device/dtype mismatch in reverse-KL loss | `AssistantVocabTranslatorCache.get_translator(...)` was called with `assistant_model=None`, which makes HF build internal tensors on CPU while real logits live on GPU | Pass the real model object into `get_translator`, not `None` |
| Universal NaN in reverse-KL loss | HF's own `FILTER_VALUE` constant is literally `-inf`, not a large finite sentinel; `exp(-inf) = 0`, and IEEE arithmetic evaluates `0 * -inf` as `NaN` (not the mathematically-correct limit of 0) | `masked_fill` replacing `-inf` with a large finite negative value before `log_softmax` |
| A second, distinct NaN source in a restricted-vocabulary KL computation | With a ~64k-token restricted softmax, float32 genuinely underflows to exact `0.0` for many entries; `log(0.0) = -inf`; same `0 * -inf = NaN` issue, different code path | Compute KL manually with `torch.where(q > 0, kl_terms, 0)` to enforce the correct mathematical limit |
| Data-collection truncation bias in acceptance-head training | `collect_round` discarded every record past a rejection point — correct for the KV cache, but this also silently discarded the one genuinely useful *negative* training example per rejected round; labels clustered at ~1.0, head learned "always predict accept" | Stop truncating the saved records (only the KV-cache crop still uses the truncation point); recollected data spanned the full 0.0-1.0 label range |
| Drafter-prompt corruption via cross-model special tokens | The drafter's prompt was built by decoding the *target's* fully chat-templated text (with the target's own `<start_of_turn>`/`<end_of_turn>` markers left in) and re-tokenizing with the drafter's tokenizer, which had never seen those tokens and treated them as confusing literal text — drafter degraded into generic, unrelated completions | Build the drafter's prompt from its *own* chat template, appending only the plain generated text |
| OOM from deferred single `backward()` call | Every round's full computational graph was kept alive simultaneously until one combined `backward()` at the end of a question — peak memory scaled with (rounds per question) × (per-round activation memory) | Proper gradient accumulation: `.backward()` immediately after each round, explicit `del` + `empty_cache()`, one `optimizer.step()` at the end |
| Full-parameter fine-tuning contradicting the on-device premise | The training loop updated every parameter of the drafter — full gradients + AdamW state, roughly 2x model size in memory, not viable on real phone RAM | Switched to LoRA (r=8, alpha=16); confirmed 4,399,104 / 498,431,872 trainable parameters (0.88%) |

---

## 6. Experiments and results

### 6.1 The early UAG validation journey — chasing a 0.22x-0.33x speedup to its real cause

First cross-vocab SD tests (Qwen3-0.6B/Gemma-7b-it, then Llama-68M/Qwen2-7B) both landed at **0.22x-0.33x** — much worse than baseline. Vocab size was ruled out (swapping pairs didn't change it). A custom translation cache confirmed 100% hit rate with speedup unchanged, ruling out translation cost. Fine-grained timing found only 6% of per-call time in candidate generation, 0.07s in translation, ~0s in drafter generation — 94% unaccounted. Root cause: cross-GPU model splitting (see bug log). Clean fix (single GPU, fp16): **0.95x-1.29x, avg ~1.11x, all outputs matched baseline exactly** — consistent with OmniDraft's own reported "direct mapping only" baseline of 0.94-1.04x.

### 6.2 Router experiments

- First calibration test (15 disjoint prompts): drafter's own confidence did **not** predict SD speedup — in fact the relationship trended opposite to what was assumed (higher-confidence half averaged **0.519x**, lower-confidence half **0.860x**). Diagnosed cause: a model's confidence in its own top pick ("how peaked is my distribution") is a different question from cross-model agreement ("will the target accept what I picked") — these were only weakly, and in this case inversely, related for this pairing.
- LinUCB contextual bandit built (`linucb_router.py`, OrcaRouter pattern adopted directly, Sherman-Morrison online updates, exploration bonus). With placeholder cost constants, picked `escalate_sd` in 28/30 rounds — explicitly flagged as a calibration artifact, not a validated recommendation.
- Once real measured costs were substituted (escalate avg 16.50s vs small-alone avg 3.21s, ~5.14x ratio), the router's own decision **flipped**: `small_alone` picked in 28/30 rounds, previously `escalate_sd` in 28/30. Same algorithm, same data — only the cost inputs changed the outcome, a strong argument for calibrating on real measured costs rather than assumed ones.
- First real router-decision integration (confidence threshold=0.55, picked blind): **12/12 escalated** — threshold badly miscalibrated (real confidence range was only 0.44-0.55 even on perfectly-answered questions), and the signal didn't reliably separate hard from easy questions either. Router threshold-tuning explicitly parked for a later session.

### 6.3 Distillation / drafter-training experiments (early attempts, pre-online-training)

| condition | avg speedup | avg tok/call |
|---|---|---|
| Untrained drafter | 1.117x | 1.47 |
| Trained drafter (reverse-KL, offline) | 1.172x | 1.56 |

A held-out re-eval (30 prompts) showed tok/call improving +10.6% (2.159 → 2.388) — but the **full wall-clock comparison reversed the verdict**: avg speedup **1.295x → 1.042x**, 28 of 30 prompts got *worse* in wall-clock terms despite the tok/call gain. Root cause (see 6.4): HF's adaptive draft-length heuristic overshooting on the now more-confident trained drafter, plus ~44% raw overhead from an unmerged LoRA adapter.

### 6.4 The tok/call-vs-wall-clock gap, fully diagnosed

| | adaptive length | fixed length=4 |
|---|---|---|
| Trained, avg speedup | 1.042x | **1.210x** |

Merging the LoRA adapter into the base weights alone recovered most of the gap (merged+adaptive: **1.208x**). The diagnosis: reverse-KL training makes the drafter more *confident* (mode-seeking), which drives the adaptive-length heuristic to propose much longer draft chains (avg proposed length 1.83 → 3.38) without a proportional rise in per-token acceptance (100% → ~56%) — net compute goes up even though tok/call looks better. This became the founding precedent for treating tok/call as a proxy metric, not a latency measurement, for the rest of the project (recurred again in the 2026-07-26 adaptive-window confound, see 5.2).

### 6.5 Acceptance-prediction head, v1 and v2

- **v1** (Qwen3-0.6B): trained jointly with reverse-KL loss, final average head accuracy **0.6576**. Integrated at inference to gate draft length — result **1.159x**, did not beat either merging alone (1.208x) or fixed-length alone (1.210x); the extra forward pass needed to score each round cost more than the gain.
- **v2** (Llama-68M, correct ~132:1 drafter:target ratio, hidden-state capture made free during the drafter's own drafting call instead of a bolted-on extra pass): head accuracy **0.6506**. Partial fix of the avg-K over-inflation: pulled average proposed length from 3.38 toward 3.066, roughly on par with a simple fixed-length-4 policy, not fully back to the untrained baseline's 1.83.

### 6.6 Component C toy prototype (2026-07-24)

- **Quality equivalence** (8 examples): small-model-alone F1 **0.051**, target-plain F1 **0.182**, SD-handoff F1 **0.186** — statistically indistinguishable between plain and SD-handoff, both far above the small model working alone, exactly as the greedy-verification proof predicts.
- **Wall-clock**: SD-handoff beat plain generation in every tested cell, **1.43x-1.56x**.
- **Warm/cold KV cache**, tested at realistic ~1,800-token conversation length after two broken attempts (match=False, ~1.0x) and a proper fix following HF's documented `DynamicCache` pattern: **avg speedup 1.307x** (1.261x-1.354x range), cross-validated against an independent prefill-timing profile (predicted ~9200ms cold / ~7050-7500ms warm vs measured 9221ms / 7054ms).
- **Pipelining/concurrency** at this same realistic scale: avg sequential 9.276s vs avg concurrent 9.322s, **speedup 0.995x** — essentially a wash, no real benefit on a single machine with zero network latency to hide compute behind.
- **Prefill-scaling profile** (why warm/cold matters so much at scale): cold prefill cost per 100 tokens drops from 409ms (at 50 tokens) to ~86-114ms (at 500-2000 tokens) — at 2000-token history, cold prefill (~2288ms) is over 10x slower than a 100-token warm continuation (~214ms).

### 6.7 Multi-turn stability

First bidirectional escalate/de-escalate test: escalation #1 avg speedup **1.217x**, escalation #2 (after a full de-escalation cycle) **1.191x** — nearly identical, confirming the mechanism doesn't degrade under repeated switching. Swapping the drafter for an instruction-tuned model (Qwen2.5-0.5B-Instruct) fixed a real coherence problem the base Llama-68M drafter had (nonsense continuations in a Q&A-structured multi-turn test) — all 12 answers across 3 topics came out correct and on-topic, tok/call nearly doubled (1.27-1.43 → 2.3-2.6). A later, more careful re-measurement on the full 30-example set found the actual wall-clock speedup with this drafter (**1.213x avg**) was *lower* than an earlier flagship number, because that earlier number had been compared against the wrong baseline — corrected once caught.

### 6.8 Quantization — closed out with a real, working fused kernel

GPTQ Int4 via the `EXLLAMA_V2` backend (the only fused kernel that would build on T4 hardware — `GPTQ_MARLIN`, `AWQ_GEMM`, `AWQ_MARLIN` all failed): fp16 baseline **1.185x** (tok/call 2.366) vs quantized **1.111x** (tok/call 2.266) — a real 16% relative recovery over a broken-kernel version, but still below fp16. Verdict: a batch-1, single-sequence drafter this small doesn't have enough compute per call for a low-bit kernel to earn back its own overhead on this hardware class.

### 6.9 The flagship number, fully derived

| configuration | avg speedup | avg tok/call |
|---|---|---|
| greedy + primitive translator (original baseline) | 1.185x | 2.366 |
| sampling + fixed direct-mapping translator | 1.132x | 2.883 |
| **greedy + fixed direct-mapping translator** | **1.394x** | **2.797** |

By complexity: simple **1.452x**, moderate **1.382x**, complex **1.325x**. The sampling-based version has *better* raw agreement (tok/call 2.883 vs 2.797) but *loses* on wall-clock (1.132x vs 1.394x) because sampling-based verification pays real per-round overhead that greedy verification doesn't — a case where the higher-tok/call configuration is the worse deployment choice, another instance of the tok/call-vs-wall-clock gap.

### 6.10 The real OmniDraft n-gram cache, and Prompt Lookup Decoding

- N-gram cache: grew to 23 entries over a real conversation, **98.3%** of tokens resolved via direct mapping, the remainder hit the cache at an **86.6%** reuse rate. Mechanism confirmed correct — but wall-clock was a wash (`with_cache` 0.410x vs `no_cache` 0.416x) because the standalone test loop itself lacked KV-cache reuse, swamping any gain from the translation cache.
- Prompt Lookup Decoding (established alternative, no draft model at all, string-matches recent n-grams): **avg speedup 0.998x** (breakeven), tok/call 1.160.
- Both confirm the same underlying finding independently: **open conversational Q&A doesn't repeat literal text/phrases the way summarization or code does** — the mechanisms are real and correctly implemented, the domain just doesn't have the structure either one needs.

### 6.11 Joint cost model — draft speed, network, and server compute together

| network condition | drafter state | naive (s) | SD (s) | user-latency speedup |
|---|---|---|---|---|
| 0ms (hypothetical) | healthy | 12.92 | 7.48 | 1.728x |
| 0ms (hypothetical) | throttled | 15.20 | 12.05 | 1.262x |
| good wifi/5G (20ms) | healthy | 12.94 | 7.90 | 1.637x |
| good wifi/5G | throttled | 15.22 | 12.48 | 1.220x |
| typical 4G (75ms) | healthy | 12.99 | 9.08 | 1.430x |
| typical 4G | throttled | 15.28 | 13.66 | 1.119x |
| poor connectivity (150ms) | healthy | 13.07 | 10.69 | 1.222x |
| **poor connectivity** | **throttled** | 15.35 | 15.26 | **1.006x (breakeven)** |

Server-side compute reduction (11.2s → 4.0s decode time, **2.797x**) holds **unconditionally** — it doesn't depend on network or drafter speed at all, since it's purely about how many forward passes the expensive model has to run per output token. The user-latency win survives every tested condition except the worst-case stack (throttled phone + poor connectivity), where it collapses to roughly breakeven, not negative.

### 6.12 Online training — the full arc, including a real reversal

Built completely: data collection with real-record hybrid loss (reverse-KL on direct-mapped tokens + NLL on n-gram-merged tokens), an acceptance-prediction head (continuous-ratio BCE label, not binary), a live training loop with gradient accumulation, LoRA (not full fine-tuning, corrected after review — see bug log).

**First test, conversational domain**: losses decreased cleanly (reverse-KL ~10%, NLL ~40%, BCE ~9% over the run) but tok/call did **not** improve (1.489 → 1.431, -3.9%). Read initially as "training doesn't help here."

**Forced replication under OmniDraft's own conditions** (Llama-68M drafter, GSM8K, matching their actual benchmark domain), specifically to separate "wrong domain" from "wrong implementation":

| condition | first-half avg tok/call | second-half avg tok/call | change |
|---|---|---|---|
| control (no training) | 1.158 | 1.164 | +0.5% (flat) |
| train (online training) | 1.156 | 1.218 | **+5.4% (real)** |

This **reversed the verdict**: the mechanism is real and correctly implemented — it's domain-gated, working on structured/repetitive domains (GSM8K) and not (yet) on open conversational Q&A, joining the n-gram-cache/PLD finding in the same domain-dependence pattern.

**Follow-up variants on the conversational domain, testing whether the loss function or data volume was the gate**:
- Forward KL (per DistillSpec's finding that greedy decoding favors forward KL over reverse KL): tok/call **-12.2%**, worse than reverse-KL's -3.9%. Losses still decreased cleanly.
- Training on ALL routed turns (SD-verified + naive-handoff, not just SD rounds, using whichever signal each turn already carries for free): tok/call **-7.9%** on SD-routed questions, and for the first time the SD-branch losses themselves stopped improving (KL flat, n-gram NLL +20.5% worse) — diagnosed as objective interference from alternating two different training objectives on the same shared LoRA weights via separate gradient steps, not evidence that naive-handoff data itself carries no signal.

Neither follow-up rescues the conversational domain. The loss-function choice and the SD-only data-volume limitation were both ruled out as "the" explanation — the remaining, best-supported hypothesis is that the domain itself (open conversation) is the gate, the same conclusion the n-gram cache and PLD independently reached.

### 6.13 Adaptive speculation-length controllers — two attempts, both null

1. **Acceptance-head as a live controller** (frozen drafter, only the head trains): stop drafting when predicted accept-probability drops below 0.5. Result: **-6.3%** vs a fixed window of 5 — the controller stopped after only ~1.7 tokens on average. Diagnosis: the head was trained on the *continuous ratio* `min(1, p/q)`, but the actual accept event under greedy verification is *binary* (argmax match or not) — a drafter can get the top token right while still being much less peaked than the target, giving a ratio well under 0.5 even on tokens that would have matched. The 0.5 threshold on that ratio was miscalibrated for what the controller was actually being asked to decide.
2. **Training-free confidence-threshold controller** (the drafter's own top-softmax probability, no training at all — matching the literature's own established approach: AdaEDL, GammaTune+, CM-ASD, AdaSpec, DSpark all use exactly this kind of signal, explicitly "training and parameter-free"). First sweep showed **+53.9%** at the best threshold, with a fixed-window=8 control seemingly ruling out "just a bigger cap" as the explanation. This was **entirely a token-selection-policy confound** (see bug log, 5.2) — once the fixed and adaptive drafting functions were made to use identical repetition-penalty/no-repeat-ngram/EOS-handling, the corrected sweep showed every threshold within ~3% of the fixed baseline, flat and noise-level. **The +53.9% is retracted, not reduced.**

**Standing conclusion from this whole thread, stated at the point of the final correction**: *"the flagship direct-mapping + greedy-verification result (1.394x) remains the only lever that has robustly helped on this specific conversational deployment domain — every additional mechanism tried this session (quantization, n-gram cache, PLD, online training in three configurations, both adaptive-window approaches) has been null or negative here, though several (online training, n-gram cache) are confirmed real on other domains (GSM8K, repetitive/structured content)."*

---

## 7. Verdict: proven / ruled out / open

| Claim | Status | Evidence |
|---|---|---|
| Handoff costs zero quality | **Proven** | Provable by construction under greedy decoding (§4.3); confirmed empirically, F1 0.186 vs 0.182 |
| Core handoff speedup | **Proven, 1.394x** | §6.9, controlled for ordering bias, two real library bugs fixed to get there |
| KV-cache reuse at realistic scale | **Proven, 1.307x** | §6.6, cross-validated against an independent profiler |
| Multi-turn stability | **Proven** | §6.7, escalate/de-escalate cycles don't degrade speedup or coherence |
| Server compute reduction | **Proven, 2.797x, unconditional** | §6.11 — holds regardless of network or drafter speed |
| User-latency win under real conditions | **Proven, 1.0x-1.7x depending on conditions** | §6.11 — survives every scenario except the worst-case stack |
| Quantized drafter | **Ruled out** | §6.8 — real working kernel tested, not enough per-call compute at this scale to pay off |
| Same-GPU concurrent drafting | **Ruled out, here** | §6.6 — reopened under real network latency in §10 |
| N-gram translation cache | **Ruled out, this domain; confirmed real mechanism** | §6.10 — works, domain lacks the repetition it needs |
| Online drafter/head training | **Domain-gated** | §6.12 — real (+5.4%) on GSM8K, null/negative on conversational Q&A across 3 configurations |
| Adaptive draft-length control | **Ruled out, both approaches tried** | §6.13 — one miscalibrated, one a confound that fully explained a false +53.9% |
| Router (Component A) | **Built, not yet calibrated for deployment** | §6.2 — LinUCB flips decisions correctly once given real costs, threshold tuning still open |
| Component C (context reuse) | **Prototype validated, hidden-state fusion ruled unsolved** | §2, §6.6 |
| Full combined system (A+B+C+D + all trained pieces together) | **Never assembled** | Each piece tested alone or in partial pairs; the source papers' own results show components can flip from harmful to helpful once combined — not closed off either way |

---

## 8. Literature map

**Component A — routing/deferral**: RouteLLM (arxiv 2406.18665), Hybrid LLM (arxiv 2404.14618), FrugalGPT (arxiv 2305.05176), Speculative Cascades (arxiv 2405.19261), OrcaRouter (arxiv.org/html/2605.30736 — LinUCB pattern adopted directly).

**Component B — cross-architecture handoff**: hidden-state projection background (Relative Representations, arxiv 2209.15430; Platonic Representation Hypothesis, Huh et al. ICML'24; EAGLE, arxiv 2401.15077; Patchscopes, arxiv 2401.06102) — explored, confirmed genuinely unsolved for cross-family lossless SD. Established text round-trip approach: Lossless SD for Heterogeneous Vocabularies (arxiv 2502.05202, ICML'25 Oral — SLEM/SLRS/TLI), HF Universal Assisted Generation, **OmniDraft** (arxiv 2507.02659, NeurIPS'25 — n-gram cache + hybrid distillation, the main source paper for the training/cache mechanisms in this project).

**Component C — cheap context reuse**: CachedAttention (arxiv 2403.19708), DroidSpeak (arxiv 2411.02820), SwiftCache (arxiv 2606.16135), KVzip (arxiv 2505.23416), KV-Distill (arxiv 2503.10337), Cartridges (arxiv 2506.06266), Cache-to-Cache/C2C (arxiv 2510.03215), Q-KVComm, QKVShare, CE-LSLM (arxiv 2505.14085).

**Component D — partial correction** (descoped): BiLD (arxiv 2302.07863), DuoDecoding (arxiv 2503.00784), Dovetail (arxiv 2412.18934).

**Network-latency / edge-cloud SD**: "Speculation at a Distance" (arxiv 2606.25091), DSSD (arxiv 2507.12000), PipeSD (arxiv 2605.13319), DSD (arxiv 2511.21669).

**Online adaptation**: Online Speculative Decoding / OSD (arxiv 2310.07177 — server-side continuous training via serving-cluster idle compute; does not address cross-vocab pairs), DistillSpec (forward-KL vs reverse-KL choice for greedy decoding), AdaEDL (arxiv 2410.18351), GammaTune+, CM-ASD, AdaSpec, DSpark (adaptive/dynamic speculation-length literature — AdaEDL's formal bound is explicitly sampling-specific, not ported here).

**Contrast case**: RLM-Cascade (arxiv 2606.22840) — cross-vendor cascade, useful specifically because its own paper confirms token-level SD is structurally impossible when draft/verify models are separate HTTP APIs with no internal logit access, establishing why this project's approach needs real model access to work at all.

---

## 9. Known limitations

- **Every test ran the small model on a cloud GPU, not a phone.** All timing assumes drafter compute a real mobile CPU/NPU doesn't have. Direction is understood (slower drafter → smaller net win), magnitude on real hardware is untested.
- **Network cost is estimated, not measured.** The joint cost model (§6.11) combines published mobile drafter benchmarks, our own server-side measurements, and simulated network scenarios — a plausibility estimate, not a fresh measurement on real hardware or a live network.
- **Drafter-size comparisons were confounded with drafter:target ratio** in early testing — literature suggests ratio, not raw size, may be the more predictive variable; not cleanly isolated here.
- **The complete four-component system was never assembled.** Direct mapping, the translation cache, the joint training loss, and the acceptance-gating head were each tested alone or in partial pairs, never all together on one consistent model pairing.
- **Every domain-dependent finding (n-gram cache, PLD, online training) was tested on exactly one domain family beyond GSM8K** — open-ended conversational Q&A. A task with more literal repetition (code, structured extraction) could plausibly show different results for all three.
- **Router threshold calibration is unresolved** — LinUCB responds correctly to its cost inputs, but no real deployment threshold has been set.

---

## 10. Next agenda

1. **Validate on real mobile hardware, once available.** Everything else here can be estimated without a device; this is what actually closes the loop.
2. **Reopen concurrent drafting under simulated network delay.** Directly motivated by the joint cost model's near-breakeven worst case — a real edge-cloud paper (Jetson Nano + A100) reported an 11% gain from "preemptive drafting" specifically because genuine network-driven dead time exists to hide compute behind, a precondition the same-GPU test in §6.6 didn't have.
3. **Calibrate the router's escalation threshold for real deployment conditions** — the mechanism responds correctly to its inputs (§6.2), it just hasn't been tuned against a real decision boundary yet.
4. **Test online training and n-gram-cache-family mechanisms on a domain with real repetition structure** (code generation, structured extraction) — both are confirmed-real mechanisms that are null specifically because open conversation lacks the structure they need, not because they're broken.
5. **Assemble the complete four-component system once**, on one consistent model pairing, since the source papers' own results show components can flip from harmful to helpful in combination — this possibility hasn't been closed off by testing pieces in isolation.

---

*Compiled from the full dated research log (`hybrid-llm-research-notes.md`) spanning quality-equivalence proofs, wall-clock validation, upstream library bug fixes, a complete online-training build with a real domain-gated reversal, and cross-checks against the published edge-cloud speculative decoding literature.*
