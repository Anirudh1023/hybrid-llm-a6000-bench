import torch, time
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

DRAFT_ID  = 'JackFram/llama-68m'
TARGET_ID = 'Qwen/Qwen2-7B-Instruct'

target_tokenizer    = AutoTokenizer.from_pretrained(TARGET_ID)
assistant_tokenizer = AutoTokenizer.from_pretrained(DRAFT_ID)
target_model = AutoModelForCausalLM.from_pretrained(TARGET_ID, dtype='auto', device_map='auto')
assistant_model = AutoModelForCausalLM.from_pretrained(DRAFT_ID, dtype='auto', device_map='auto')

call_count = {'n': 0}
target_model.register_forward_hook(lambda m,i,o: call_count.update(n=call_count['n']+1))

prompts = [
    'Explain why the sky appears blue in simple terms.',
    'Write a short function in Python that reverses a linked list.',
    'Summarize the plot of Romeo and Juliet in three sentences.',
]

def run(prompt, mode):
    inputs = target_tokenizer(prompt, return_tensors='pt').to(target_model.device)
    common = dict(max_new_tokens=100)
    if mode == 'baseline':
        gc = GenerationConfig(do_sample=False, **common)
        extra = {}
    elif mode == 'uag_greedy':
        gc = GenerationConfig(do_sample=False, **common)
        extra = dict(assistant_model=assistant_model, tokenizer=target_tokenizer, assistant_tokenizer=assistant_tokenizer)
    elif mode == 'uag_sample_fastpath':
        gc = GenerationConfig(do_sample=True, temperature=0.01, top_k=1, top_p=1.0, **common)
        extra = dict(assistant_model=assistant_model, tokenizer=target_tokenizer, assistant_tokenizer=assistant_tokenizer)
    call_count['n'] = 0
    torch.cuda.synchronize()
    t0 = time.time()
    out = target_model.generate(**inputs, generation_config=gc, **extra)
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    n_new = out.shape[1] - inputs['input_ids'].shape[1]
    calls = max(call_count['n'], 1)
    text = target_tokenizer.decode(out[0], skip_special_tokens=True)
    return text, elapsed, n_new, calls, n_new/calls

print()
for p in prompts:
    tb, eb, nb, cb, tpcb = run(p, 'baseline')
    tg, eg, ng, cg, tpcg = run(p, 'uag_greedy')
    try:
        ts, es, ns, cs, tpcs = run(p, 'uag_sample_fastpath')
        fast_result = f'{es:.2f}s  ({ns} tok, {cs} calls, {tpcs:.2f} tok/call)  speedup: {eb/es:.2f}x'
        fast_match = (tb == ts)
    except Exception as e:
        fast_result = f'FAILED: {type(e).__name__}: {str(e)[:200]}'
        fast_match = 'N/A'
    print(f'PROMPT: {p}')
    print(f'  baseline:               {eb:.2f}s  ({nb} tok, {cb} calls, {tpcb:.2f} tok/call)')
    print(f'  UAG greedy (slow path): {eg:.2f}s  ({ng} tok, {cg} calls, {tpcg:.2f} tok/call)  speedup: {eb/eg:.2f}x')
    print(f'  UAG fast path:          {fast_result}')
    print(f'  greedy matches baseline: {tb==tg}   fast matches baseline: {fast_match}')
    print()
