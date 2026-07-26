import torch, time, json, gc
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
import bitsandbytes as bnb

DRAFT_ID  = 'Qwen/Qwen3-0.6B'
TARGET_ID = 'google/gemma-7b-it'
MAX_NEW = 60

with open('train_prompts.json') as f: train_prompts = json.load(f)
with open('heldout_prompts.json') as f: heldout_prompts = json.load(f)

target_tokenizer    = AutoTokenizer.from_pretrained(TARGET_ID)
assistant_tokenizer = AutoTokenizer.from_pretrained(DRAFT_ID)

# ============ PHASE 1: TRAIN (8-bit target for memory, LoRA rank 32 on drafter) ============
print('=== PHASE 1: LoRA TRAINING (rank 32, matching OmniDraft on-device config) ===')
bnb_config = BitsAndBytesConfig(load_in_8bit=True)
target_model = AutoModelForCausalLM.from_pretrained(TARGET_ID, quantization_config=bnb_config, device_map={'': 0})
base_assistant = AutoModelForCausalLM.from_pretrained(DRAFT_ID, dtype=torch.float16, device_map={'': 0})

lora_config = LoraConfig(
    r=32, lora_alpha=64, lora_dropout=0.05,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    task_type='CAUSAL_LM',
)
assistant_model = get_peft_model(base_assistant, lora_config)
assistant_model.print_trainable_parameters()
assistant_model.train()
optimizer = bnb.optim.AdamW8bit(assistant_model.parameters(), lr=1e-4)  # higher lr typical for LoRA

call_count = {'n': 0}
target_model.register_forward_hook(lambda m,i,o: call_count.update(n=call_count['n']+1))

def measure(prompt, assistant, max_new=MAX_NEW):
    inputs = target_tokenizer(prompt, return_tensors='pt', truncation=True, max_length=1024).to(target_model.device)
    call_count['n'] = 0
    torch.cuda.synchronize(); t0 = time.time()
    out = target_model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                                 assistant_model=assistant, tokenizer=target_tokenizer, assistant_tokenizer=assistant_tokenizer)
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

print('=== HELD-OUT, BEFORE TRAINING ===')
before_tpc = []
for p in heldout_prompts:
    _, _, _, tpc, _ = measure(p, assistant_model)
    before_tpc.append(tpc)
avg_before = sum(before_tpc)/len(before_tpc)
print(f'BEFORE avg: {avg_before:.3f}')

skipped = 0
for i, p in enumerate(train_prompts):
    torch.cuda.empty_cache()
    _, _, _, _, cont = measure(p, assistant_model)
    loss = train_step(p, cont)
    if loss is None: skipped += 1
    if i % 20 == 0: print(f'  [{i}/{len(train_prompts)}] loss: {loss}')
print(f'Training done, skipped {skipped}/{len(train_prompts)}')

print('=== HELD-OUT, AFTER TRAINING ===')
after_tpc = []
for p in heldout_prompts:
    _, _, _, tpc, _ = measure(p, assistant_model)
    after_tpc.append(tpc)
avg_after = sum(after_tpc)/len(after_tpc)
print(f'AFTER avg: {avg_after:.3f}   change: {100*(avg_after-avg_before)/avg_before:+.2f}%')

# ============ SAVE LoRA ADAPTER (small, just the adapter weights) ============
assistant_model.save_pretrained('trained_drafter_lora_06b')
print('Saved LoRA adapter to trained_drafter_lora_06b/')
