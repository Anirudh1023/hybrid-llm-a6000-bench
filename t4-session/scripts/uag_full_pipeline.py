import torch, time, json, gc
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import bitsandbytes as bnb

DRAFT_ID  = 'JackFram/llama-68m'
TARGET_ID = 'Qwen/Qwen2-7B-Instruct'
MAX_NEW = 60

with open('train_prompts.json') as f: train_prompts = json.load(f)
with open('heldout_prompts.json') as f: heldout_prompts = json.load(f)

target_tokenizer    = AutoTokenizer.from_pretrained(TARGET_ID)
assistant_tokenizer = AutoTokenizer.from_pretrained(DRAFT_ID)

# ============ PHASE 1: TRAIN (8-bit target for memory headroom) ============
print('=== PHASE 1: TRAINING ===')
bnb_config = BitsAndBytesConfig(load_in_8bit=True)
target_model = AutoModelForCausalLM.from_pretrained(TARGET_ID, quantization_config=bnb_config, device_map={'': 0})
assistant_model = AutoModelForCausalLM.from_pretrained(DRAFT_ID, dtype=torch.float16, device_map={'': 0})
assistant_model.train()
optimizer = bnb.optim.AdamW8bit(assistant_model.parameters(), lr=1e-5)

call_count = {'n': 0}
target_model.register_forward_hook(lambda m,i,o: call_count.update(n=call_count['n']+1))

def measure(prompt, max_new=MAX_NEW):
    inputs = target_tokenizer(prompt, return_tensors='pt', truncation=True, max_length=1024).to(target_model.device)
    call_count['n'] = 0
    torch.cuda.synchronize(); t0 = time.time()
    out = target_model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                                 assistant_model=assistant_model, tokenizer=target_tokenizer, assistant_tokenizer=assistant_tokenizer)
    torch.cuda.synchronize(); elapsed = time.time() - t0
    n_new = out.shape[1] - inputs['input_ids'].shape[1]
    calls = max(call_count['n'], 1)
    cont = target_tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return elapsed, n_new, calls, n_new/calls, cont

def train_step(prompt, continuation_text):
    assistant_model.train()
    prompt_ids = assistant_tokenizer(prompt, return_tensors='pt', truncation=True, max_length=1024).input_ids.to(assistant_model.device)
    cont_ids = assistant_tokenizer(continuation_text, return_tensors='pt', add_special_tokens=False).input_ids.to(assistant_model.device)
    full_ids = torch.cat([prompt_ids, cont_ids], dim=1)
    labels = full_ids.clone()
    labels[:, :prompt_ids.shape[1]] = -100
    outputs = assistant_model(input_ids=full_ids, labels=labels)
    loss = outputs.loss
    torch.cuda.empty_cache()
    if torch.isnan(loss) or torch.isinf(loss):
        assistant_model.eval()
        return None
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    assistant_model.eval()
    return loss.item()

skipped = 0
for i, p in enumerate(train_prompts):
    torch.cuda.empty_cache()
    _, _, _, _, cont = measure(p)
    loss = train_step(p, cont)
    if loss is None: skipped += 1
    if i % 20 == 0: print(f'  [{i}/{len(train_prompts)}]')
print(f'Training done, skipped {skipped}/{len(train_prompts)}')

# ============ SAVE TRAINED DRAFTER ============
assistant_model.save_pretrained('trained_drafter_v1')
assistant_tokenizer.save_pretrained('trained_drafter_v1')
print('Saved trained drafter to trained_drafter_v1/')

# ============ FREE MEMORY: drop 8-bit target + optimizer entirely ============
del target_model, optimizer
gc.collect()
torch.cuda.empty_cache()
torch.cuda.synchronize()
print(f'Memory after cleanup: {torch.cuda.memory_allocated(0)/1e9:.2f} GB allocated')

# ============ PHASE 2: CLEAN FP16 SPEEDUP MEASUREMENT ============
print()
print('=== PHASE 2: CLEAN FP16 SPEEDUP COMPARISON (trained vs untrained drafter) ===')
target_model_clean = AutoModelForCausalLM.from_pretrained(TARGET_ID, dtype=torch.float16, device_map={'': 0})
call_count2 = {'n': 0}
target_model_clean.register_forward_hook(lambda m,i,o: call_count2.update(n=call_count2['n']+1))

def measure_clean(model, assistant, prompt, max_new=100):
    inputs = target_tokenizer(prompt, return_tensors='pt', truncation=True, max_length=1024).to(model.device)
    kwargs = dict(max_new_tokens=max_new, do_sample=False)
    if assistant is not None:
        kwargs.update(assistant_model=assistant, tokenizer=target_tokenizer, assistant_tokenizer=assistant_tokenizer)
    call_count2['n'] = 0
    torch.cuda.synchronize(); t0 = time.time()
    out = model.generate(**inputs, **kwargs)
    torch.cuda.synchronize(); elapsed = time.time() - t0
    n_new = out.shape[1] - inputs['input_ids'].shape[1]
    calls = max(call_count2['n'], 1)
    return elapsed, calls, n_new/calls

# Original untrained drafter
original_drafter = AutoModelForCausalLM.from_pretrained(DRAFT_ID, dtype=torch.float16, device_map={'': 0})
# Trained drafter
trained_drafter = AutoModelForCausalLM.from_pretrained('trained_drafter_v1', dtype=torch.float16, device_map={'': 0})

eval_prompts = heldout_prompts[:15]  # subset for wall-clock time reasons
results_base, results_orig, results_trained = [], [], []
for p in eval_prompts:
    eb, cb, tpcb = measure_clean(target_model_clean, None, p)
    eo, co, tpco = measure_clean(target_model_clean, original_drafter, p)
    et, ct, tpct = measure_clean(target_model_clean, trained_drafter, p)
    results_base.append(eb); results_orig.append(eo); results_trained.append(et)
    print(f'baseline: {eb:.2f}s | UAG-original: {eo:.2f}s (speedup {eb/eo:.2f}x, {tpco:.2f} tok/call) | UAG-trained: {et:.2f}s (speedup {eb/et:.2f}x, {tpct:.2f} tok/call)')

import statistics
print()
print(f'AVG baseline time: {statistics.mean(results_base):.2f}s')
print(f'AVG UAG-original speedup: {statistics.mean(results_base)/statistics.mean(results_orig):.3f}x')
print(f'AVG UAG-trained  speedup: {statistics.mean(results_base)/statistics.mean(results_trained):.3f}x')
