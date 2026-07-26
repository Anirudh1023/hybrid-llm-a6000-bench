import torch, time
from transformers import AutoModelForCausalLM, AutoTokenizer

DRAFT_ID  = 'JackFram/llama-68m'
TARGET_ID = 'Qwen/Qwen2-7B-Instruct'

target_tokenizer    = AutoTokenizer.from_pretrained(TARGET_ID)
assistant_tokenizer = AutoTokenizer.from_pretrained(DRAFT_ID)

try:
    target_model = AutoModelForCausalLM.from_pretrained(TARGET_ID, dtype=torch.float16, device_map={'': 0})
    assistant_model = AutoModelForCausalLM.from_pretrained(DRAFT_ID, dtype=torch.float16, device_map={'': 0})
    print('LOADED_OK_FP16_SINGLE_GPU')
except torch.cuda.OutOfMemoryError as e:
    print('OOM_FP16_SINGLE_GPU:', str(e)[:300])
    raise SystemExit(42)

print('assistant device:', next(assistant_model.parameters()).device)
print('target device(s):', set(p.device for p in target_model.parameters()))
print('target mem allocated (GB):', torch.cuda.memory_allocated(0)/1e9)

call_count = {'n': 0}
target_model.register_forward_hook(lambda m,i,o: call_count.update(n=call_count['n']+1))

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
    n_new = out.shape[1] - inputs['input_ids'].shape[1]
    calls = max(call_count['n'], 1)
    text = target_tokenizer.decode(out[0], skip_special_tokens=True)
    return text, elapsed, n_new, calls

print()
for p in prompts:
    tb, eb, nb, cb = run(p, False)
    tg, eg, ng, cg = run(p, True)
    print(f'PROMPT: {p}')
    print(f'  baseline (single GPU, fp16): {eb:.2f}s  ({nb} tok, {cb} calls, {1000*eb/cb:.0f}ms/tok)')
    print(f'  UAG (single GPU, fp16):      {eg:.2f}s  ({ng} tok, {cg} calls, {1000*eg/cg:.0f}ms/round)  speedup: {eb/eg:.2f}x')
    print(f'  output matches: {tb==tg}')
    print()
