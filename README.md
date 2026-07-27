# Hybrid LLM: cross-vocabulary speculative decoding as accelerator + handoff

This repo is now the full project handoff, not just an A6000 benchmark
extension — it contains the complete T4-session research (architecture,
every bug found and fixed, every experiment run, the full script history)
alongside the A6000-scale experiments this machine exists to run.

**Start here, in this order:**
1. **`PROJECT_README.md`** — the real project overview: architecture, the
   end-to-end flow, the core mechanism, every bug found and fixed, every
   experiment with exact numbers, the current verdict on what's proven vs
   ruled out vs domain-gated, and the literature map. Read this first if
   you're new to the project on this machine.
2. **`SCRIPT_INDEX.md`** — a script-by-script map (with a flowchart) of
   which script produced which result and when, cross-referenced to
   `PROJECT_README.md`. Read this when you need to find "which script
   produced that number" without re-reading the whole narrative.
3. **`RESEARCH_LOG.md`** — the full, dated, append-only raw research log
   `PROJECT_README.md` was written from. Ground truth for anything the
   summary doesn't cover in enough detail.
4. **`t4-session/`** — every script (47 total) and result artifact from
   the T4-based work, organized by phase (see `SCRIPT_INDEX.md`).
5. **`ONLINE_TRAINING_PLAN.md`** — a focused handoff plan for `exp07`-`exp09`,
   written directly from a paper-reading session against the actual
   OmniDraft text (not a summary): three specific implementation gaps
   between our online-training build and the paper's exact recipe (an
   untested λ=0.2 loss weighting, a never-built "interleaved" acceptance-
   head training variant, and an unvalidated cross-vocab assumption the
   paper's own authors never tested either), each with exact equations,
   build steps, and success criteria. Read this before touching exp07-09.
6. **This file + `CONCLUSIONS.md` + `experiments/`** — the A6000-specific
   continuation: the experiments below, motivated directly by what the T4
   session left open (see "What each experiment is FOR").

---

## Why this machine, what it's for

Extends the T4-based diagnostic session (2026-07-22 through 07-26) onto a
48GB A6000, specifically to run the experiments the T4's 2x15.6GB VRAM
couldn't support: genuinely large drafter:target ratios (30B+ targets), a
properly ratio-matched same-family vs cross-family comparison, isolated
quantization-overhead measurement, and a corrected full reverse-KL pipeline
that applies every fix found on the T4 in the right order.

The T4 session's standing verdict, which is what these experiments are
actually trying to move past (full detail in `PROJECT_README.md` §6-7):
> "The flagship direct-mapping + greedy-verification result (1.394x)
> remains the only lever that has robustly helped on this specific
> conversational deployment domain — every additional mechanism tried
> (quantization, n-gram cache, PLD, online training in three
> configurations, both adaptive-window approaches) has been null or
> negative here, though several (online training, n-gram cache) are
> confirmed real on other domains (GSM8K, repetitive/structured content)."

**Read `CONCLUSIONS.md` next** — it's the living summary of what's proven,
what's still open, and what each experiment here is meant to resolve. Update
it as you go; it's meant to accumulate across sessions, not just describe
one machine's work.

## Setup

```bash
pip install -r requirements.txt
huggingface-cli login   # needs gated-repo access for Gemma models
```

VRAM budget (single 48GB A6000) — keep this in mind when picking model IDs
in the experiment scripts:
- 30B+ targets: 8-bit only (fp16 needs ~2x the model size in GB, won't fit)
- Up to ~20B targets: fp16 is viable (~2GB/B param), useful for isolating
  quantization overhead cleanly (see `exp05`)
- Drafters (0.5B–3B): fp16 is always fine, negligible VRAM either way

## Layout

```
PROJECT_README.md    the real project overview — start here
SCRIPT_INDEX.md       script-by-script map + flowchart of the whole project
RESEARCH_LOG.md       full dated raw research log (source of PROJECT_README.md)

t4-session/          everything from the T4-based work (see SCRIPT_INDEX.md)
  scripts/             47 scripts, every experiment actually run so far
  data/                prompt sets, datasets, reward logs used across scripts
  results/             JSON results + small artifacts (acceptance-head checkpoints)

src/                 reusable core (import from experiments/ scripts)
  harness.py          measure_baseline / measure_sd / run_full_comparison
  vocab_utils.py       config.vocab_size native-path check (see below — READ THIS)
  lora_utils.py        merge_and_unload wrapper (see below — READ THIS)
  reverse_kl_train.py  the FIXED OmniDraft-style reverse-KL training script
data/                prompt sets (LoCoMo-derived, same 30-prompt set used on the T4)
experiments/         exp01-exp06, each independently runnable
results/             JSON output from each experiment, read by report.py
run_all.py           orchestrates experiments in a sensible order
report.py            builds BENCHMARK_REPORT.md from results/*.json
CONCLUSIONS.md        the living summary — read/update this every session
```

## Two things to internalize before running anything

**1. `config.vocab_size` equality, not tokenizer class or vocab overlap, gates
HF's native (translator-free) assisted-generation path.** Two checkpoints of
the literal same tokenizer can have different `config.vocab_size` (embedding
table padding) and get silently routed through the slower cross-vocab
translator anyway. Always call `vocab_utils.check_native_path(target_id,
draft_id)` before assuming a "same-family" experiment is testing what you
think it's testing — this exact mistake invalidated a same-family conclusion
on the T4 session.

**2. Always `merge_and_unload()` a LoRA adapter before measuring inference
speed**, unless you specifically need to keep training it further. An
unmerged adapter measured 44% slower per forward pass than the same
weights merged — pure overhead from computing the low-rank update live on
every call, nothing to do with the adapter's actual learned behavior. This
was the single biggest confound in the T4 session's "does reverse-KL
training help" question — see `CONCLUSIONS.md`.

## Running experiments

```bash
cd experiments
python exp04_lora_overhead_at_scale.py     # cheap, no target model, run first
python exp01_large_target_ratio.py
python exp02_matched_ratio_family_test.py
python exp05_quantization_isolation.py

# these two need a trained adapter first:
python ../src/reverse_kl_train.py --target-id google/gemma-2-27b --save-dir trained_drafter
python exp03_fixed_vs_adaptive_draft_length.py
python exp06_full_reverse_kl_pipeline.py

cd ..
python report.py    # -> BENCHMARK_REPORT.md, all results consolidated + compared vs T4 baselines
```

Or just: `python run_all.py --with-training` to run everything in order.

**exp07-exp09 are plan stubs, not runnable yet** — see `ONLINE_TRAINING_PLAN.md`
first. Each raises `NotImplementedError` with the exact build steps in its
own docstring; they build on `t4-session/scripts/` (the real, complete
online-training code), not `src/reverse_kl_train.py` (an earlier,
intentionally incomplete predecessor — its own docstring says so).

## What each experiment is FOR (not just what it does)

| Experiment | Question it answers | Why the T4 couldn't answer it |
|---|---|---|
| exp01 | Does a genuinely large ratio (literature-scale, 28:1-280:1) give the speedup the literature reports, on our own harness? | T4 topped out at 18.2:1 — never reached the range where cross-family SD is actually validated in the literature |
| exp02 | Same-family vs cross-family, ratio held constant | T4's same-family test conflated ratio AND a silent cross-vocab-path fallback — never a clean comparison |
| exp03 | Does the adaptive-draft-length overshoot problem get better or worse at bigger target scale? | T4 could only test at 9B; reasoned (not measured) that bigger targets being more memory-bound should shrink this effect |
| exp04 | Is LoRA's ~44% per-step overhead specific to Qwen3-0.6B, or does it reproduce elsewhere? | Only ever measured on one drafter, one adapter config |
| exp05 | How much does 8-bit quantization itself cost, in isolation? | T4 had no choice but 8-bit for anything past ~7B — never had a fp16 control |
| exp06 | With every known fix applied (merge, fixed-length option, real target scale) — does reverse-KL training finally show a genuine wall-clock win? | The "ultimate" test — nothing this clean was possible on the T4 rig |
| exp07 | Does OmniDraft's actual λ=0.2 loss weighting (never tested — we used an unweighted, implicit λ=1) change the GSM8K win or the conversational-domain null? | Found only by reading the paper's exact equations against our own code, not by running anything on the T4 |
| exp08 | Does OmniDraft's "interleaved" acceptance-head training (never built — we only tested "joint") fix the -6.3% adaptive-window result? | Requires new buffer/training-cadence code, not built during the T4 session |
| exp09 | Does the acceptance-head mechanism work at all on a same-vocab pair, isolating implementation correctness from an untested cross-vocab assumption? | The T4 session tested cross-vocab directly with no same-vocab control — the paper's own authors never validated this mechanism cross-vocab either |

## What's explicitly NOT in scope here (see CONCLUSIONS.md "still missing")

**Update**: the note below was accurate when this repo was created
(2026-07-23) but is now stale — OmniDraft's actual online n-gram cache and
acceptance-prediction head WERE built, in `t4-session/scripts/` (see
`SCRIPT_INDEX.md` phases 6-8). exp01-06 below are about the pre-that-work
question (does the incomplete direct-mapping-only implementation reach a
genuine wall-clock win at real scale); exp07-09 and `ONLINE_TRAINING_PLAN.md`
are the direct continuation of the now-complete online-training work, not
a return to something unbuilt.

Building OmniDraft's actual online n-gram cache and acceptance-prediction
head is a separate, larger engineering task — not something to squeeze into
a benchmarking session. These experiments are about getting the CORRECTLY
MEASURED ceiling of what our current (incomplete) implementation can do at
real scale, and isolating every confound we can, before deciding whether
building those two missing pieces is actually necessary or whether these
fixes (merge, fixed draft length, bigger ratio) already get us most of the
way there.
