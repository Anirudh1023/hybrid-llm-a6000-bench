import torch, time
from transformers import AutoModelForCausalLM, AutoTokenizer

DRAFT_ID  = 'Qwen/Qwen3-0.6B'
TARGET_ID = 'google/gemma-7b-it'

print('Loading tokenizers...')
target_tokenizer    = AutoTokenizer.from_pretrained(TARGET_ID)
assistant_tokenizer = AutoTokenizer.from_pretrained(DRAFT_ID)

print('target vocab size:', target_tokenizer.vocab_size)
print('draft  vocab size:', assistant_tokenizer.vocab_size)

print('Loading models (this will take a while on first run)...')
target_model = AutoModelForCausalLM.from_pretrained(TARGET_ID, dtype='auto', device_map='auto')
assistant_model = AutoModelForCausalLM.from_pretrained(DRAFT_ID, dtype='auto', device_map='auto')

prompts = [
    'Explain why the sky appears blue in simple terms.',
    'Write a short function in Python that reverses a linked list.',
    'Summarize the plot of Romeo and Juliet in three sentences.',
]

def run(prompt, use_assistant):
    inputs = target_tokenizer(prompt, return_tensors='pt').to(target_model.device)
    kwargs = dict(max_new_tokens=100, do_sample=False)
    if use_assistant:
        kwargs.update(assistant_model=assistant_model, tokenizer=target_tokenizer, assistant_tokenizer=assistant_tokenizer)
    torch.cuda.synchronize()
    t0 = time.time()
    out = target_model.generate(**inputs, **kwargs)
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    text = target_tokenizer.decode(out[0], skip_special_tokens=True)
    return text, elapsed

print()
for p in prompts:
    text_base, t_base = run(p, use_assistant=False)
    text_uag, t_uag = run(p, use_assistant=True)
    print(f'PROMPT: {p}')
    print(f'  baseline:     {t_base:.2f}s')
    print(f'  UAG-assisted: {t_uag:.2f}s  (speedup: {t_base/t_uag:.2f}x)')
    print(f'  baseline output: {text_base[:150]!r}')
    print(f'  uag output:      {text_uag[:150]!r}')
    print()
