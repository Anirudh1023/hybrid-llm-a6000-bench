import torch, time, json
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import bitsandbytes as bnb

DRAFT_ID  = 'JackFram/llama-68m'
TARGET_ID = 'Qwen/Qwen2-7B-Instruct'
MAX_NEW = 60  # reduced from 100 given the larger N of prompts, to keep wall-clock reasonable

with open('train_prompts.json') as f: train_prompts = json.load(f)
with open('heldout_prompts.json') as f: heldout_prompts = json.load(f)
print(f'train: {len(train_prompts)}  heldout: {len(heldout_prompts)}')

target_tokenizer    = AutoTokenizer.from_pretrained(TARGET_ID)
assistant_tokenizer = AutoTokenizer.from_pretrained(DRAFT_ID)
bnb_config = BitsAndBytesConfig(load_in_8bit=True)
target_model = AutoModelForCausalLM.from_pretrained(TARGET_ID, quantization_config=bnb_config, device_map={'': 0})
assistant_model = AutoModelForCausalLM.from_pretrained(DRAFT_ID, dtype=torch.float16, device_map={'': 0})
assistant_model.train()
optimizer = bnb.optim.AdamW8bit(assistant_model.parameters(), lr=1e-5)

call_count = {'n': 0}
target_model.register_forward_hook(lambda m,i,o: call_count.update(n=call_count['n']+1))

def measure(prompt):
    inputs = target_tokenizer(prompt, return_tensors='pt', truncation=True, max_length=1024).to(target_model.device)
    call_count['n'] = 0
    torch.cuda.synchronize(); t0 = time.time()
    out = target_model.generate(**inputs, max_new_tokens=MAX_NEW, do_sample=False,
                                 assistant_model=assistant_model, tokenizer=target_tokenizer, assistant_tokenizer=assistant_tokenizer)
    torch.cuda.synchronize(); elapsed = time.time() - t0
    n_new = out.shape[1] - inputs['input_ids'].shape[1]
    calls = max(call_count['n'], 1)
    continuation_text = target_tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return n_new/calls, continuation_text

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
        return None  # skip this step entirely, do not touch the optimizer/weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    assistant_model.eval()
    return loss.item()

print('=== HELD-OUT, BEFORE TRAINING ===')
before_tpc = []
for i, p in enumerate(heldout_prompts):
    tpc, _ = measure(p)
    before_tpc.append(tpc)
    if i % 5 == 0: print(f'  [{i}/{len(heldout_prompts)}] running avg: {sum(before_tpc)/len(before_tpc):.3f}')
avg_before = sum(before_tpc)/len(before_tpc)
print(f'HELD-OUT BEFORE avg: {avg_before:.3f}')

print()
print(f'=== TRAINING (1 epoch, {len(train_prompts)} diverse LoCoMo prompts) ===')
losses = []
skipped = 0
for i, p in enumerate(train_prompts):
    torch.cuda.empty_cache()
    _, cont = measure(p)
    loss = train_step(p, cont)
    if loss is None:
        skipped += 1
        if i % 10 == 0: print(f'  [{i}/{len(train_prompts)}] SKIPPED (nan/inf loss)  running avg loss so far: {(sum(losses)/len(losses)) if losses else float("nan"):.4f}')
        continue
    losses.append(loss)
    if i % 10 == 0: print(f'  [{i}/{len(train_prompts)}] loss: {loss:.4f}  running avg loss: {sum(losses)/len(losses):.4f}')
print(f'TRAINING DONE. final avg loss: {(sum(losses)/len(losses)) if losses else float("nan"):.4f}  (skipped {skipped}/{len(train_prompts)} steps)')

print()
print('=== HELD-OUT, AFTER TRAINING ===')
after_tpc = []
for i, p in enumerate(heldout_prompts):
    tpc, _ = measure(p)
    after_tpc.append(tpc)
    if i % 5 == 0: print(f'  [{i}/{len(heldout_prompts)}] running avg: {sum(after_tpc)/len(after_tpc):.3f}')
avg_after = sum(after_tpc)/len(after_tpc)
print(f'HELD-OUT AFTER avg: {avg_after:.3f}')

print()
print(f'FINAL RESULT: before={avg_before:.4f}  after={avg_after:.4f}  change={100*(avg_after-avg_before)/avg_before:+.2f}%')
print(f'before per-prompt: {[round(x,2) for x in before_tpc]}')
print(f'after  per-prompt: {[round(x,2) for x in after_tpc]}')
