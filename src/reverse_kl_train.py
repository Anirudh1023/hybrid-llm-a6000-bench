"""
OmniDraft-style reverse-KL + n-gram distillation training, FIXED version.

Bug fixed here (found the hard way on the T4 session): HF's
AssistantToTargetTranslator.FILTER_VALUE is literal -float('inf'), not a
large-finite-negative sentinel. log_softmax propagates real -inf to filtered
(unmapped) vocab positions; exp(-inf)=0 times (-inf - finite) = 0 * -inf =
NaN under IEEE754 -- and this happens BEFORE any `* valid_mask` masking step,
so masking afterward can't rescue it (NaN * 0 is still NaN). Result before
the fix: 105/105 training steps silently produced NaN loss and were skipped,
every time, on every prompt -- zero actual training happened despite the
script running to completion with no errors.

Fix: replace -inf with a large-but-finite negative (-1e4) via masked_fill on
`projected` BEFORE log_softmax, so log-softmax output never contains literal
-inf -- exp() of a large-enough finite negative already underflows cleanly
to 0.0 without ever forming the 0*-inf NaN, in both forward AND backward
(there's a known PyTorch gotcha where torch.where doesn't reliably rescue
this in the backward pass either -- fixing it upstream of the arithmetic is
the only sound approach).

STILL INCOMPLETE relative to the actual OmniDraft paper (arxiv 2507.02659):
this implements ONLY the KL term over ~25% "expressible" (single-token
aligned) vocab positions. It does NOT implement:
  1. OmniDraft's online n-gram cache (merges sub-token SEQUENCES into
     single target tokens -- structurally required to even construct a loss
     over the ~75% non-aligned positions; without it there's no way to know
     what the "correct" multi-token correspondence even is).
  2. The trained acceptance-prediction head (explicit P(rejection)
     early-exit, replacing HF's generic adaptive draft-length heuristic).
These are real, structural gaps, not just missing polish -- see
CONCLUSIONS.md for the full diagnostic chain. The n-gram cache in particular
is why the reported OmniDraft speedups (1.5-2x) don't need to match what
this script alone can produce.
"""

import gc
import json

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.generation.candidate_generator import AssistantVocabTranslatorCache


def train_reverse_kl(
    draft_id: str,
    target_id: str,
    train_prompts: list[str],
    lambda_ngram: float = 0.2,
    lora_r: int = 32,
    lora_alpha: int = 64,
    lr: float = 1e-4,
    target_8bit: bool = True,
    max_new: int = 60,
    log_every: int = 20,
    save_dir: str = "trained_drafter",
):
    target_tokenizer = AutoTokenizer.from_pretrained(target_id)
    assistant_tokenizer = AutoTokenizer.from_pretrained(draft_id)

    if target_8bit:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        target_model = AutoModelForCausalLM.from_pretrained(target_id, quantization_config=bnb_config, device_map={"": 0})
    else:
        target_model = AutoModelForCausalLM.from_pretrained(target_id, dtype=torch.float16, device_map={"": 0})
    base_assistant = AutoModelForCausalLM.from_pretrained(draft_id, dtype=torch.float16, device_map={"": 1})

    lora_config = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM",
    )
    assistant_model = get_peft_model(base_assistant, lora_config)
    assistant_model.print_trainable_parameters()

    import bitsandbytes as bnb
    optimizer = bnb.optim.AdamW8bit(assistant_model.parameters(), lr=lr)

    translator = AssistantVocabTranslatorCache.get_translator(
        target_tokenizer, assistant_tokenizer, target_model.config.vocab_size,
        assistant_model=base_assistant, assistant_prune_lm_head=False,
    )
    target_to_assistant = translator.target_to_assistant_input_ids

    def get_target_continuation(prompt):
        inputs = target_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(target_model.device)
        with torch.no_grad():
            out = target_model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
        cont_ids = out[0][inputs["input_ids"].shape[1]:]
        del out
        gc.collect()
        torch.cuda.empty_cache()
        return cont_ids

    def train_step(prompt, target_cont_ids):
        aligned_assistant_ids, aligned_target_positions, ngram_positions = [], [], []
        for i, tid in enumerate(target_cont_ids.tolist()):
            if tid in target_to_assistant:
                aligned_assistant_ids.append(target_to_assistant[tid])
                aligned_target_positions.append(i)
            else:
                ngram_positions.append(i)

        if not aligned_assistant_ids and not ngram_positions:
            return None

        target_prompt_ids = target_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).input_ids.to(target_model.device)
        target_full_ids = torch.cat([target_prompt_ids, target_cont_ids.unsqueeze(0).to(target_model.device)], dim=1)
        with torch.no_grad():
            target_out = target_model(input_ids=target_full_ids)
        target_logits_full = target_out.logits[0, target_prompt_ids.shape[1] - 1:-1, :]
        del target_out
        gc.collect()
        torch.cuda.empty_cache()

        assistant_prompt_ids = assistant_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).input_ids.to(assistant_model.device)
        if aligned_assistant_ids:
            aligned_cont_ids = torch.tensor(aligned_assistant_ids, device=assistant_model.device).unsqueeze(0)
            assistant_full_ids = torch.cat([assistant_prompt_ids, aligned_cont_ids], dim=1)
        else:
            assistant_full_ids = assistant_prompt_ids
        assistant_model.train()
        assistant_out = assistant_model(input_ids=assistant_full_ids)
        assistant_logits_full = assistant_out.logits[0, assistant_prompt_ids.shape[1] - 1:-1, :]

        loss_dm = torch.tensor(0.0, device=assistant_model.device)
        if aligned_assistant_ids:
            projected = translator.get_target_logits(assistant_logits_full.float().unsqueeze(0))[0]
            target_logits_aligned = target_logits_full[aligned_target_positions].to(assistant_model.device)
            valid_mask = projected[0] != translator.FILTER_VALUE
            # THE FIX: replace -inf with large-finite-negative BEFORE log_softmax
            projected_safe = projected.masked_fill(projected == translator.FILTER_VALUE, -1e4)
            q = F.log_softmax(projected_safe, dim=-1)
            p = F.softmax(target_logits_aligned.float(), dim=-1)
            kl_per_pos = (q.exp() * (q - torch.log(p.clamp_min(1e-12)))) * valid_mask.unsqueeze(0)
            loss_dm = kl_per_pos.sum(dim=-1).mean()

        # n-gram fallback term -- STILL STUBBED. See module docstring: this
        # requires an actual n-gram cache to even define what the "correct"
        # multi-token target is for these positions. Contributes 0 for now.
        loss_ngram = torch.tensor(0.0, device=assistant_model.device)

        loss = loss_dm + lambda_ngram * loss_ngram
        if torch.isnan(loss) or torch.isinf(loss):
            assistant_model.eval()
            gc.collect()
            torch.cuda.empty_cache()
            return None
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        assistant_model.eval()
        gc.collect()
        torch.cuda.empty_cache()
        return loss.item()

    print("=== TRAINING (reverse-KL + n-gram stub) ===", flush=True)
    losses, skipped = [], 0
    for i, p in enumerate(train_prompts):
        cont_ids = get_target_continuation(p)
        loss = train_step(p, cont_ids)
        if loss is None:
            skipped += 1
            continue
        losses.append(loss)
        if i < 3 or i % log_every == 0:
            print(f"  [{i}/{len(train_prompts)}] loss: {loss:.4f}", flush=True)
    print(f"Training done, skipped {skipped}/{len(train_prompts)}, "
          f"avg loss: {(sum(losses) / len(losses)) if losses else float('nan'):.4f}", flush=True)

    assistant_model.save_pretrained(save_dir)
    print(f"Saved to {save_dir}/", flush=True)
    return assistant_model, target_model, target_tokenizer, assistant_tokenizer


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft-id", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--target-id", required=True)
    parser.add_argument("--train-prompts", default="../data/train_prompts.json")
    parser.add_argument("--save-dir", default="trained_drafter")
    parser.add_argument("--no-8bit", action="store_true")
    args = parser.parse_args()

    with open(args.train_prompts) as f:
        train_prompts = json.load(f)

    train_reverse_kl(
        args.draft_id, args.target_id, train_prompts,
        target_8bit=not args.no_8bit, save_dir=args.save_dir,
    )
