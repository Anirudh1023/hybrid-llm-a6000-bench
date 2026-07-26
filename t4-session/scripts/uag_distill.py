import torch, time
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# NOTE ON SCOPE: this is a simplified, well-justified version of OmniDraft's
# online hybrid distillation (ELDM + lambda*ELNgram, trained per-round mid-generation).
# Simplification: trains per FULL RESPONSE instead of per verification round,
# using the confirmed-correct fact that do_sample=False assisted-generation
# output exactly equals what the target alone would produce (we verified this
# with exact output matches in the clean fp16 single-GPU run). Uses OmniDraft's
# own reverse-translation formula (tokenize_drafter(detokenize_target(...)))
# to build training targets in the drafter's vocabulary, then standard NLL
# fine-tuning (a defensible simplification of their dual reverse-KL + NLL loss).

DRAFT_ID  = 'JackFram/llama-68m'
TARGET_ID = 'Qwen/Qwen2-7B-Instruct'

target_tokenizer    = AutoTokenizer.from_pretrained(TARGET_ID)
assistant_tokenizer = AutoTokenizer.from_pretrained(DRAFT_ID)
bnb_config = BitsAndBytesConfig(load_in_8bit=True)
target_model = AutoModelForCausalLM.from_pretrained(TARGET_ID, quantization_config=bnb_config, device_map={'': 0})
assistant_model = AutoModelForCausalLM.from_pretrained(DRAFT_ID, dtype=torch.float16, device_map={'': 0})
assistant_model.train()
import bitsandbytes as bnb
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
    # OmniDraft's exact reverse-translation formula: tokenize_q(detokenize_p(...))
    # here: detokenize already have as continuation_text (from target), tokenize with drafter's tokenizer
    assistant_model.train()
    prompt_ids = assistant_tokenizer(prompt, return_tensors='pt').input_ids.to(assistant_model.device)
    cont_ids = assistant_tokenizer(continuation_text, return_tensors='pt', add_special_tokens=False).input_ids.to(assistant_model.device)
    full_ids = torch.cat([prompt_ids, cont_ids], dim=1)
    labels = full_ids.clone()
    labels[:, :prompt_ids.shape[1]] = -100  # only train on the continuation, mask the prompt
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

print('=== BEFORE TRAINING ===')
before = {}
for p in train_prompts:
    e, n, c, tpc, cont = measure(p)
    before[p] = (e, n, c, tpc)
    print(f'{p[:40]:40s} {e:.2f}s  {c} calls  {tpc:.2f} tok/call')

print()
print('=== ONLINE TRAINING (3 epochs over the same prompts, simulating repeated interaction) ===')
for epoch in range(3):
    for p in train_prompts:
        torch.cuda.empty_cache()
        _, _, _, _, cont = measure(p)  # generate with current (possibly updated) assistant to get fresh target signal
        loss = train_step(p, cont)
    print(f'epoch {epoch+1} done, last loss: {loss:.4f}')

print()
print('=== AFTER TRAINING ===')
for p in train_prompts:
    e, n, c, tpc, cont = measure(p)
    eb, nb, cb, tpcb = before[p]
    print(f'{p[:40]:40s} before: {eb:.2f}s/{cb}calls/{tpcb:.2f}tpc   after: {e:.2f}s/{c}calls/{tpc:.2f}tpc')
