import torch, time
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import bitsandbytes as bnb

DRAFT_ID  = 'JackFram/llama-68m'
TARGET_ID = 'Qwen/Qwen2-7B-Instruct'

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
    inputs = target_tokenizer(prompt, return_tensors='pt').to(target_model.device)
    call_count['n'] = 0
    torch.cuda.synchronize(); t0 = time.time()
    out = target_model.generate(**inputs, max_new_tokens=100, do_sample=False,
                                 assistant_model=assistant_model, tokenizer=target_tokenizer, assistant_tokenizer=assistant_tokenizer)
    torch.cuda.synchronize(); elapsed = time.time() - t0
    n_new = out.shape[1] - inputs['input_ids'].shape[1]
    calls = max(call_count['n'], 1)
    continuation_text = target_tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return elapsed, n_new, calls, n_new/calls, continuation_text

def train_step(prompt, continuation_text):
    assistant_model.train()
    prompt_ids = assistant_tokenizer(prompt, return_tensors='pt').input_ids.to(assistant_model.device)
    cont_ids = assistant_tokenizer(continuation_text, return_tensors='pt', add_special_tokens=False).input_ids.to(assistant_model.device)
    full_ids = torch.cat([prompt_ids, cont_ids], dim=1)
    labels = full_ids.clone()
    labels[:, :prompt_ids.shape[1]] = -100
    outputs = assistant_model(input_ids=full_ids, labels=labels)
    loss = outputs.loss
    torch.cuda.empty_cache()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    assistant_model.eval()
    return loss.item()

train_prompts = [
    'Explain why the sky appears blue in simple terms.',
    'Write a short function in Python that reverses a linked list.',
    'Summarize the plot of Romeo and Juliet in three sentences.',
    'What are the main causes of climate change?',
    'Write a haiku about autumn leaves.',
    'Explain the difference between a list and a tuple in Python.',
]

heldout_prompts = [
    'Explain how vaccines work in simple terms.',
    'Write a Python function that checks if a string is a palindrome.',
    'Summarize the plot of Hamlet in three sentences.',
    'What are the main causes of ocean pollution?',
    'Write a haiku about a rainy day.',
    'Explain the difference between a stack and a queue in programming.',
]

print('=== HELD-OUT SET, BEFORE TRAINING ===')
before_heldout = {}
for p in heldout_prompts:
    e, n, c, tpc, cont = measure(p)
    before_heldout[p] = tpc
    print(f'{p[:45]:45s} {tpc:.2f} tok/call  ({c} calls)')

print()
print('=== TRAINING (3 epochs on TRAIN set only, held-out set never seen) ===')
for epoch in range(3):
    for p in train_prompts:
        torch.cuda.empty_cache()
        _, _, _, _, cont = measure(p)
        loss = train_step(p, cont)
    print(f'epoch {epoch+1} done, last loss: {loss:.4f}')

print()
print('=== HELD-OUT SET, AFTER TRAINING (generalization test) ===')
avg_before, avg_after = 0, 0
for p in heldout_prompts:
    e, n, c, tpc, cont = measure(p)
    tpc_before = before_heldout[p]
    avg_before += tpc_before
    avg_after += tpc
    print(f'{p[:45]:45s} before: {tpc_before:.2f} tok/call   after: {tpc:.2f} tok/call   ({c} calls)')
avg_before /= len(heldout_prompts)
avg_after /= len(heldout_prompts)
print()
print(f'HELD-OUT AVERAGE:  before={avg_before:.3f}  after={avg_after:.3f}  change={100*(avg_after-avg_before)/avg_before:+.1f}%')

print()
print('=== TRAIN SET, AFTER TRAINING (reproduce earlier result for reference) ===')
for p in train_prompts:
    e, n, c, tpc, cont = measure(p)
    print(f'{p[:45]:45s} after: {tpc:.2f} tok/call   ({c} calls)')
