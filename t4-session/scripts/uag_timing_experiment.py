import torch, time
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.candidate_generator import AssistedCandidateGeneratorDifferentTokenizers, AssistedCandidateGenerator

DRAFT_ID  = 'JackFram/llama-68m'
TARGET_ID = 'Qwen/Qwen2-7B-Instruct'

target_tokenizer    = AutoTokenizer.from_pretrained(TARGET_ID)
assistant_tokenizer = AutoTokenizer.from_pretrained(DRAFT_ID)
target_model = AutoModelForCausalLM.from_pretrained(TARGET_ID, dtype='auto', device_map='auto')
assistant_model = AutoModelForCausalLM.from_pretrained(DRAFT_ID, dtype='auto', device_map='auto')
print('assistant device:', next(assistant_model.parameters()).device)
print('target device(s):', set(p.device for p in target_model.parameters()))

timers = {'get_candidates_total': 0.0, 'translate': 0.0, 'assistant_generate': 0.0, 'n_rounds': 0}

_orig_convert = AssistedCandidateGeneratorDifferentTokenizers.convert_source_tokens_to_target_tokens
def timed_convert(self, input_ids, source_tokenizer, destination_tokenizer):
    t0 = time.time()
    r = _orig_convert(self, input_ids, source_tokenizer, destination_tokenizer)
    torch.cuda.synchronize()
    timers['translate'] += time.time() - t0
    return r
AssistedCandidateGeneratorDifferentTokenizers.convert_source_tokens_to_target_tokens = timed_convert

_orig_gen_candidates = AssistedCandidateGenerator._generate_candidates
def timed_generate_candidates(self, generation_args):
    t0 = time.time()
    r = _orig_gen_candidates(self, generation_args)
    torch.cuda.synchronize()
    timers['assistant_generate'] += time.time() - t0
    return r
AssistedCandidateGenerator._generate_candidates = timed_generate_candidates

_orig_get_candidates = AssistedCandidateGeneratorDifferentTokenizers.get_candidates
def timed_get_candidates(self, input_ids, **kwargs):
    t0 = time.time()
    r = _orig_get_candidates(self, input_ids, **kwargs)
    torch.cuda.synchronize()
    timers['get_candidates_total'] += time.time() - t0
    timers['n_rounds'] += 1
    return r
AssistedCandidateGeneratorDifferentTokenizers.get_candidates = timed_get_candidates

prompt = 'Explain why the sky appears blue in simple terms.'
inputs = target_tokenizer(prompt, return_tensors='pt').to(target_model.device)

torch.cuda.synchronize()
t0 = time.time()
out = target_model.generate(**inputs, max_new_tokens=100, do_sample=False,
                             assistant_model=assistant_model, tokenizer=target_tokenizer, assistant_tokenizer=assistant_tokenizer)
torch.cuda.synchronize()
total = time.time() - t0

print()
print(f'TOTAL generate() time:        {total:.2f}s')
print(f'  sum of get_candidates():    {timers["get_candidates_total"]:.2f}s  ({100*timers["get_candidates_total"]/total:.0f}% of total)')
print(f'    of which translate():     {timers["translate"]:.2f}s  ({100*timers["translate"]/total:.0f}% of total)')
print(f'    of which assistant.generate(): {timers["assistant_generate"]:.2f}s  ({100*timers["assistant_generate"]/total:.0f}% of total)')
print(f'  unaccounted (target verification + rest of loop): {total - timers["get_candidates_total"]:.2f}s  ({100*(total-timers["get_candidates_total"])/total:.0f}% of total)')
print(f'  n_rounds: {timers["n_rounds"]}')
