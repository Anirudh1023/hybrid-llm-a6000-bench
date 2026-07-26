import torch, time, json, statistics, sys

DRAFTER_PATH = sys.argv[1]  # 'JackFram/llama-68m' or 'trained_drafter_v1'
LABEL = sys.argv[2]

from transformers import AutoModelForCausalLM, AutoTokenizer
DRAFT_ID  = 'JackFram/llama-68m'
TARGET_ID = 'Qwen/Qwen2-7B-Instruct'

with open('heldout_prompts.json') as f: heldout_prompts = json.load(f)
target_tokenizer    = AutoTokenizer.from_pretrained(TARGET_ID)
assistant_tokenizer = AutoTokenizer.from_pretrained(DRAFT_ID)

target_model = AutoModelForCausalLM.from_pretrained(TARGET_ID, dtype=torch.float16, device_map={'': 0})
drafter = AutoModelForCausalLM.from_pretrained(DRAFTER_PATH, dtype=torch.float16, device_map={'': 0})

call_count = {'n': 0}
target_model.register_forward_hook(lambda m,i,o: call_count.update(n=call_count['n']+1))

def measure(assistant, prompt, max_new=100):
    inputs = target_tokenizer(prompt, return_tensors='pt', truncation=True, max_length=1024).to(target_model.device)
    kwargs = dict(max_new_tokens=max_new, do_sample=False)
    if assistant is not None:
        kwargs.update(assistant_model=assistant, tokenizer=target_tokenizer, assistant_tokenizer=assistant_tokenizer)
    call_count['n'] = 0
    torch.cuda.synchronize(); t0 = time.time()
    out = target_model.generate(**inputs, **kwargs)
    torch.cuda.synchronize(); elapsed = time.time() - t0
    n_new = out.shape[1] - inputs['input_ids'].shape[1]
    calls = max(call_count['n'], 1)
    return elapsed, calls, n_new/calls

eval_prompts = heldout_prompts[:15]
results_base, results_uag = [], []
print(f'=== {LABEL} ===')
for p in eval_prompts:
    eb, cb, tpcb = measure(None, p)
    eu, cu, tpcu = measure(drafter, p)
    results_base.append(eb); results_uag.append(eu)
    print(f'  baseline: {eb:.2f}s | UAG-{LABEL}: {eu:.2f}s (speedup {eb/eu:.2f}x, {tpcu:.2f} tok/call)')

print()
print(f'AVG baseline: {statistics.mean(results_base):.3f}s')
print(f'AVG UAG-{LABEL} speedup: {statistics.mean(results_base)/statistics.mean(results_uag):.4f}x')
