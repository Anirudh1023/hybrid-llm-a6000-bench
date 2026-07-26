import torch, time, json, statistics
from transformers import AutoModelForCausalLM, AutoTokenizer

DRAFT_ID  = 'JackFram/llama-68m'
TARGET_ID = 'Qwen/Qwen2-7B-Instruct'

with open('heldout_prompts.json') as f: heldout_prompts = json.load(f)

target_tokenizer    = AutoTokenizer.from_pretrained(TARGET_ID)
assistant_tokenizer = AutoTokenizer.from_pretrained(DRAFT_ID)

target_model = AutoModelForCausalLM.from_pretrained(TARGET_ID, dtype=torch.float16, device_map={'': 0})
original_drafter = AutoModelForCausalLM.from_pretrained(DRAFT_ID, dtype=torch.float16, device_map={'': 0})
trained_drafter = AutoModelForCausalLM.from_pretrained('trained_drafter_v1', dtype=torch.float16, device_map={'': 0})

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
results_base, results_orig, results_trained = [], [], []
print('prompt-by-prompt:')
for p in eval_prompts:
    eb, cb, tpcb = measure(None, p)
    eo, co, tpco = measure(original_drafter, p)
    et, ct, tpct = measure(trained_drafter, p)
    results_base.append(eb); results_orig.append(eo); results_trained.append(et)
    print(f'  baseline: {eb:.2f}s | UAG-original: {eo:.2f}s (speedup {eb/eo:.2f}x, {tpco:.2f} tok/call) | UAG-trained: {et:.2f}s (speedup {eb/et:.2f}x, {tpct:.2f} tok/call)')

print()
print(f'AVG baseline time: {statistics.mean(results_base):.2f}s')
print(f'AVG UAG-original speedup: {statistics.mean(results_base)/statistics.mean(results_orig):.3f}x')
print(f'AVG UAG-trained  speedup: {statistics.mean(results_base)/statistics.mean(results_trained):.3f}x')
