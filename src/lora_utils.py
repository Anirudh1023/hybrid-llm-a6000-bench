"""
LoRA merge utility -- the second major discovery of the T4 session.

An UNMERGED PeftModel (LoRA weights kept as separate low-rank A/B matrices,
computed live on every forward pass) was measured at 1.443x the wall-clock
time of the base model in pure drafter-only generation (no SD at all) --
a genuine, structural 44% overhead per forward pass, not noise (30/30
prompts, consistent).

Calling `merge_and_unload()` bakes the LoRA delta into the base weights as
a ONE-TIME operation. Measured result: merged model ran at 0.976x of base
(i.e., recovered ALL of the overhead, within noise). This single call
plausibly matters as much as the bigger OmniDraft n-gram-cache/acceptance-
head gaps for getting a usable wall-clock win from a trained drafter --
ALWAYS merge before running any inference-speed comparison unless you
specifically need to keep training the same adapter further.
"""

from peft import PeftModel


def load_and_merge(base_model, adapter_dir: str):
    """Load a LoRA adapter onto base_model and immediately merge it,
    returning a plain model with the LoRA delta baked into the base
    weights (no runtime low-rank overhead)."""
    lora_model = PeftModel.from_pretrained(base_model, adapter_dir)
    merged_model = lora_model.merge_and_unload()
    merged_model.eval()
    return merged_model
