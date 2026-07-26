import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.generation.candidate_generator import AssistantToTargetTranslator

bnb_config = BitsAndBytesConfig(load_in_8bit=True)

pairs = [
    ("Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct", 152064),
    ("google/gemma-2-9b", "Qwen/Qwen2.5-0.5B-Instruct", None),
]

for target_id, draft_id, known_vocab_size in pairs:
    target_tokenizer = AutoTokenizer.from_pretrained(target_id)
    target_model = AutoModelForCausalLM.from_pretrained(target_id, quantization_config=bnb_config, device_map={"": 0})
    target_vocab_size = known_vocab_size or target_model.config.get_text_config().vocab_size
    assistant_tokenizer = AutoTokenizer.from_pretrained(draft_id)
    assistant_model = AutoModelForCausalLM.from_pretrained(draft_id, dtype=torch.float16, device_map={"": 1})
    translator = AssistantToTargetTranslator(
        target_tokenizer=target_tokenizer,
        assistant_tokenizer=assistant_tokenizer,
        target_vocab_size=target_vocab_size,
        assistant_model=assistant_model,
        assistant_prune_lm_head=True,
    )
    n_suppress = len(translator._suppress_input_ids)
    n_total = translator._assistant_to_target_input_ids.shape[0]
    print(f"target={target_id} draft={draft_id}: {n_suppress}/{n_total} suppressed ({100*n_suppress/n_total:.1f}%)", flush=True)
    del assistant_model, target_model, translator
    torch.cuda.empty_cache()
