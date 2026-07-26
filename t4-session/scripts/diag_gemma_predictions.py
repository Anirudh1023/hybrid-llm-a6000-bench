"""Diagnostic: print actual predicted vs proposed tokens, decoded, for the
first few rounds -- to see whether disagreement looks like real (but weak)
cross-vocab alignment, or garbage (suggesting cache/architecture corruption).
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.cache_utils import DynamicCache
from transformers.generation.candidate_generator import AssistantToTargetTranslator

TARGET_ID = "google/gemma-2-9b-it"
DRAFT_ID = "Qwen/Qwen2.5-0.5B-Instruct"

target_tokenizer = AutoTokenizer.from_pretrained(TARGET_ID)
assistant_tokenizer = AutoTokenizer.from_pretrained(DRAFT_ID)
bnb_config = BitsAndBytesConfig(load_in_8bit=True)
target_model = AutoModelForCausalLM.from_pretrained(TARGET_ID, quantization_config=bnb_config, device_map={"": 0})
assistant_model = AutoModelForCausalLM.from_pretrained(DRAFT_ID, dtype=torch.float16, device_map={"": 1})
target_vocab_size = target_model.config.get_text_config().vocab_size

_translator = AssistantToTargetTranslator(
    target_tokenizer=target_tokenizer, assistant_tokenizer=assistant_tokenizer,
    target_vocab_size=target_vocab_size, assistant_model=None, assistant_prune_lm_head=False,
)
direct_map = {}
_map_tensor = _translator._assistant_to_target_input_ids
for a_id in range(_map_tensor.shape[0]):
    t_id = _map_tensor[a_id].item()
    if t_id != _translator.SUPPRESS_TOKEN_ID:
        direct_map[a_id] = t_id

messages = [{"role": "user", "content": "What's a good beginner-friendly vegetable to start cooking with?"}]
prompt = target_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
target_ids = target_tokenizer(prompt, return_tensors="pt").to(target_model.device)["input_ids"]

# Sanity check 1: does plain generation even work normally on this model?
with torch.no_grad():
    plain_out = target_model.generate(input_ids=target_ids, max_new_tokens=20, do_sample=False)
print("PLAIN GENERATION (sanity check):", target_tokenizer.decode(plain_out[0, target_ids.shape[1]:], skip_special_tokens=True), flush=True)

# Sanity check 2: DynamicCache prefill + single-token continuation, does it match plain?
cache = DynamicCache(config=target_model.config)
with torch.no_grad():
    prefill_out = target_model(input_ids=target_ids, past_key_values=cache, use_cache=True)
cache = prefill_out.past_key_values
next_pred_logit = prefill_out.logits[0, -1].float().cpu()
first_pred = next_pred_logit.argmax(dim=-1).item()
print(f"\nDynamicCache prefill's first predicted token: {first_pred!r} -> {target_tokenizer.decode([first_pred])!r}", flush=True)
print(f"Plain generation's first token: {plain_out[0, target_ids.shape[1]].item()!r} -> {target_tokenizer.decode([plain_out[0, target_ids.shape[1]].item()])!r}", flush=True)
print(f"MATCH: {first_pred == plain_out[0, target_ids.shape[1]].item()}", flush=True)

# Now draft a round and look at actual candidates vs predictions, decoded.
context_text = target_tokenizer.decode(target_ids[0], skip_special_tokens=False)
assistant_inputs = assistant_tokenizer(context_text, return_tensors="pt", truncation=True, max_length=2048).to(assistant_model.device)
with torch.no_grad():
    assistant_out = assistant_model.generate(
        **assistant_inputs, max_new_tokens=5, do_sample=False,
        repetition_penalty=1.3, no_repeat_ngram_size=3,
        pad_token_id=assistant_tokenizer.eos_token_id,
    )
new_assistant_ids = assistant_out[0, assistant_inputs["input_ids"].shape[1]:].tolist()
print(f"\nDrafter proposed (assistant vocab): {new_assistant_ids}", flush=True)
print(f"Drafter proposed decoded: {assistant_tokenizer.decode(new_assistant_ids)!r}", flush=True)

candidate_target_ids = []
for aid in new_assistant_ids:
    if aid in direct_map:
        candidate_target_ids.append(direct_map[aid])
    else:
        candidate_target_ids.append(None)
print(f"Translated to target vocab (direct_map only, None=unmapped): {candidate_target_ids}", flush=True)
decoded_candidates = [target_tokenizer.decode([c]) if c is not None else "<UNMAPPED>" for c in candidate_target_ids]
print(f"Decoded candidates: {decoded_candidates}", flush=True)
print(f"vs target's own first prediction: {first_pred!r} -> {target_tokenizer.decode([first_pred])!r}", flush=True)
