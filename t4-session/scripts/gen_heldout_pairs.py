import json, random

random.seed(42)
with open('locomo/data/locomo10.json') as f:
    data = json.load(f)

def extract_prompts_from_conv(conv, n_samples, window=8):
    conversation = conv['conversation']
    session_keys = [k for k in conversation.keys() if k.startswith('session_') and not k.endswith('date_time')]
    prompts = []
    for sk in session_keys:
        turns = conversation[sk]
        if len(turns) < window + 2:
            continue
        possible_starts = list(range(window, len(turns)))
        random.shuffle(possible_starts)
        for end_idx in possible_starts[:2]:
            segment = turns[max(0, end_idx-window):end_idx]
            text = '\n'.join(f"{t['speaker']}: {t['text']}" for t in segment)
            prompts.append(text)
    random.shuffle(prompts)
    return prompts[:n_samples]

heldout_convs = data[7:10]

heldout_prompts = []
for conv in heldout_convs:
    heldout_prompts.extend(extract_prompts_from_conv(conv, n_samples=10))

pairs = [{"prompt": p} for p in heldout_prompts]

print(f'heldout pairs: {len(pairs)}')
print('SAMPLE:')
print(pairs[0]["prompt"][:300])

with open('heldout_pairs_groundtruth.json', 'w') as f:
    json.dump(pairs, f)
print("Saved to heldout_pairs_groundtruth.json")
