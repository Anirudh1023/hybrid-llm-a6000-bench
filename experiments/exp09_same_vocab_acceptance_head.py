"""
Experiment 9: validate the acceptance-head/adaptive-drafting mechanism on a
SAME-VOCAB pair first, before trusting conclusions from the cross-vocab
test -- see ONLINE_TRAINING_PLAN.md "Gap 3" for the full context.

OmniDraft's own published acceptance-head results (Table 2, Table 11) are
ALL measured on Llama-68M + Vicuna-7B. Check their Table 8: Vicuna-7B's
vocab_size is 32000, and its intersection with Llama-68M is ALSO 32000 --
full overlap, same tokenizer (Vicuna is fine-tuned from Llama). The paper
says explicitly, in its own limitations section (4.4): "we are working
towards incorporating an explicit adaptive head in the cross-vocabulary
setting" -- listed as FUTURE work, not something they validated.

We went straight to cross-vocabulary (Qwen0.5B drafter / Gemma-9b target)
for our own acceptance-head test and got a null/negative result (-6.3%,
see adaptive_draft_length.py). That conflates two separate questions:
  1. Is our acceptance-head implementation correct AT ALL?
  2. Does cross-vocabulary specifically break it?
Never isolated. This experiment isolates #1 by removing cross-vocab
entirely.

Pair to use: Qwen2.5-0.5B-Instruct (drafter) + Qwen2.5-3B-Instruct
(target) -- confirmed in the T4 session to share vocab_size=151936 (a
genuinely native, non-cross-vocab pair -- see RESEARCH_LOG.md's
"truly-native control test" and src/vocab_utils.py's check_native_path,
which exists specifically to verify this rather than assume it from model
names). Zero suppression, zero translation layer involved -- this is as
close to OmniDraft's own Llama-68M/Vicuna-7B validation setting as our
existing model roster gets.

THIS IS A STUB but should be the CHEAPEST of the three to actually build --
mostly a model-ID swap plus deleting the translation-layer machinery.
Steps:

1. Copy t4-session/scripts/adaptive_draft_length.py.
2. Change TARGET_ID from "google/gemma-2-9b-it" to
   "Qwen/Qwen2.5-3B-Instruct", DRAFT_ID stays "Qwen/Qwen2.5-0.5B-Instruct".
3. Since this is now a same-vocab pair, the direct_map/translate_run_with_tags
   machinery (AssistantToTargetTranslator, the whole cross-vocab
   translation layer) is not needed at all -- drafted token IDs ARE
   target token IDs directly, no lookup required. Simplify
   collect_round/verify_candidate accordingly (or leave the machinery in
   place with an identity mapping if that's less code to change -- direct_map
   will just be a 1:1 identity over the full shared vocabulary in this
   case, which the existing code should handle correctly as-is without
   modification, since it already handles full-overlap pairs as a special
   case -- verify this rather than assuming).
4. Verify with vocab_utils.check_native_path(TARGET_ID, DRAFT_ID) before
   running anything -- confirm native_path=True and expect ~0 suppression,
   matching the earlier "truly-native control test" finding, before
   trusting the rest of the run.
5. Run the same prime/eval structure (topics 0-2 prime, topics 3-4 eval,
   fixed window=5 vs head-controlled adaptive, same threshold=0.5) as the
   original cross-vocab test.
6. Compare: does adaptive beat fixed here? If yes, the acceptance-head
   code is correct and the -6.3% cross-vocab result is specifically a
   cross-vocab problem (scope future work there, not at re-debugging the
   head). If it ALSO fails same-vocab, there's a bug in the acceptance-head
   implementation itself, independent of vocabulary -- debug that first.
"""

raise NotImplementedError(
    "This is a plan stub, not tested code -- see the module docstring for "
    "exact steps. Copy adaptive_draft_length.py from t4-session/scripts/, "
    "swap in the same-vocab Qwen2.5-0.5B/3B pair, verify native_path via "
    "vocab_utils.check_native_path, and rerun the same eval methodology on "
    "real GPU hardware."
)
