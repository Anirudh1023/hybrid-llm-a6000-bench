import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.candidate_generator import AssistantToTargetTranslator

TARGET_ID = "Qwen/Qwen2.5-7B-Instruct"
DRAFT_ID = "Qwen/Qwen2.5-0.5B-Instruct"

target_tokenizer = AutoTokenizer.from_pretrained(TARGET_ID)
assistant_tokenizer = AutoTokenizer.from_pretrained(DRAFT_ID)
assistant_model = AutoModelForCausalLM.from_pretrained(DRAFT_ID, dtype=torch.float16, device_map={"": 1})

target_vocab_size = 152064  # Qwen2.5-7B-Instruct's config.get_text_config().vocab_size, hardcoded to avoid loading target

translator = AssistantToTargetTranslator(
    target_tokenizer=target_tokenizer,
    assistant_tokenizer=assistant_tokenizer,
    target_vocab_size=target_vocab_size,
    assistant_model=assistant_model,
    assistant_prune_lm_head=True,
)

print(f"assistant_prune_lm_head: {translator.assistant_prune_lm_head}")
print(f"len(_suppress_input_ids): {len(translator._suppress_input_ids)}")
print(f"assistant vocab size (_assistant_to_target_input_ids len): {translator._assistant_to_target_input_ids.shape[0]}")
print(f"has assistant_overlap_token_ids: {hasattr(translator, 'assistant_overlap_token_ids')}")
if hasattr(translator, "assistant_overlap_token_ids"):
    print(f"assistant_overlap_token_ids shape: {translator.assistant_overlap_token_ids.shape}")
print(f"has logits_processors: {translator.logits_processors is not None}")

# check whether the LM head was actually swapped
lm_head = assistant_model.get_output_embeddings()
print(f"assistant LM head class: {type(lm_head).__name__}")
print(f"assistant LM head out_features (if linear): {getattr(lm_head, 'out_features', 'n/a')}")

# quick forward pass to see actual logits shape produced
dummy = torch.randint(0, 1000, (1, 8)).to(assistant_model.device)
with torch.no_grad():
    out = assistant_model(dummy)
print(f"actual forward-pass logits shape: {out.logits.shape}")
