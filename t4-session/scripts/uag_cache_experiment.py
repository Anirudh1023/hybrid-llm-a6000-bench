import torch, time
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.candidate_generator import AssistedCandidateGeneratorDifferentTokenizers

DRAFT_ID  = 'JackFram/llama-68m'
TARGET_ID = 'Qwen/Qwen2-7B-Instruct'

target_tokenizer    = AutoTokenizer.from_pretrained(TARGET_ID)
assistant_tokenizer = AutoTokenizer.from_pretrained(DRAFT_ID)
target_model = AutoModelForCausalLM.from_pretrained(TARGET_ID, dtype='auto', device_map='auto')
assistant_model = AutoModelForCausalLM.from_pretrained(DRAFT_ID, dtype='auto', device_map='auto')

# --- Cache wrapper around the exact expensive step we identified ---
translation_cache = {}
cache_stats = {'hits': 0, 'misses': 0}
_orig_convert = AssistedCandidateGeneratorDifferentTokenizers.convert_source_tokens_to_target_tokens

def cached_convert(self, input_ids, source_tokenizer, destination_tokenizer):
    key = (id(source_tokenizer), id(destination_tokenizer), tuple(input_ids.flatten().tolist()))
    if key in translation_cache:
        cache_stats['hits'] += 1
        return translation_cache[key].to(input_ids.device)
    cache_stats['misses'] += 1
    result = _orig_convert(self, input_ids, source_tokenizer, destination_tokenizer)
    translation_cache[key] = result.clone()
    return result

AssistedCandidateGeneratorDifferentTokenizers.convert_source_tokens_to_target_tokens = cached_convert

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

print('=== FIRST PASS (cold cache) ===')
for p in prompts:
    tb, eb, nb, cb = run(p, False)
    h0, m0 = cache_stats['hits'], cache_stats['misses']
    tg, eg, ng, cg = run(p, True)
    h1, m1 = cache_stats['hits'], cache_stats['misses']
    print(f'PROMPT: {p}')
    print(f'  baseline: {eb:.2f}s')
    print(f'  cached-UAG: {eg:.2f}s  speedup: {eb/eg:.2f}x  (cache hits this call: {h1-h0}, misses: {m1-m0})')
    print(f'  output matches: {tb==tg}')
    print()

print('=== SECOND PASS (warm cache, same prompts repeated) ===')
for p in prompts:
    tb, eb, nb, cb = run(p, False)
    h0, m0 = cache_stats['hits'], cache_stats['misses']
    tg, eg, ng, cg = run(p, True)
    h1, m1 = cache_stats['hits'], cache_stats['misses']
    print(f'PROMPT: {p}')
    print(f'  baseline: {eb:.2f}s')
    print(f'  cached-UAG: {eg:.2f}s  speedup: {eb/eg:.2f}x  (cache hits this call: {h1-h0}, misses: {m1-m0})')
    print(f'  output matches: {tb==tg}')
    print()

print(f'TOTAL cache stats: {cache_stats}')
