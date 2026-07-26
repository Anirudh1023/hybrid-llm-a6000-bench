import torch, time
from transformers import AutoModelForCausalLM, AutoTokenizer

DRAFT_ID  = 'JackFram/llama-68m'
TARGET_ID = 'Qwen/Qwen2-7B-Instruct'

print('Loading tokenizers...')
target_tokenizer    = AutoTokenizer.from_pretrained(TARGET_ID)
assistant_tokenizer = AutoTokenizer.from_pretrained(DRAFT_ID)
print('target vocab size:', target_tokenizer.vocab_size)
print('draft  vocab size:', assistant_tokenizer.vocab_size)

print('Loading models...')
target_model = AutoModelForCausalLM.from_pretrained(TARGET_ID, dtype='auto', device_map='auto')
assistant_model = AutoModelForCausalLM.from_pretrained(DRAFT_ID, dtype='auto', device_map='auto')

# Hook to count target model forward calls -> proxy for verification rounds.
# tokens_generated / forward_calls approx = average accepted tokens per round
# (1.0 for plain autoregressive baseline, by definition; higher = better acceptance for assisted gen)
call_count = {'n': 0}
def count_hook(module, inp, out):
    call_count['n'] += 1
target_model.register_forward_hook(count_hook)

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
    call_count['n'] = 0
    torch.cuda.synchronize()
    t0 = time.time()
    out = target_model.generate(**inputs, **kwargs)
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    n_new_tokens = out.shape[1] - inputs['input_ids'].shape[1]
    calls = max(call_count['n'], 1)
    tokens_per_call = n_new_tokens / calls
    text = target_tokenizer.decode(out[0], skip_special_tokens=True)
    return text, elapsed, n_new_tokens, calls, tokens_per_call

print()
for p in prompts:
    text_base, t_base, n_base, calls_base, tpc_base = run(p, use_assistant=False)
    text_uag, t_uag, n_uag, calls_uag, tpc_uag = run(p, use_assistant=True)
    print(f'PROMPT: {p}')
    print(f'  baseline:     {t_base:.2f}s  ({n_base} tokens, {calls_base} target fwd calls, {tpc_base:.2f} tok/call)')
    print(f'  UAG-assisted: {t_uag:.2f}s  ({n_uag} tokens, {calls_uag} target fwd calls, {tpc_uag:.2f} tok/call)  (speedup: {t_base/t_uag:.2f}x)')
    print(f'  outputs match: {text_base == text_uag}')
    print()
