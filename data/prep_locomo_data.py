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
        # sample a few different truncation points within this session
        possible_starts = list(range(window, len(turns)))
        random.shuffle(possible_starts)
        for end_idx in possible_starts[:2]:  # up to 2 samples per session
            segment = turns[max(0, end_idx-window):end_idx]
            text = '\n'.join(f"{t['speaker']}: {t['text']}" for t in segment)
            prompts.append(text)
    random.shuffle(prompts)
    return prompts[:n_samples]

train_convs = data[0:7]
heldout_convs = data[7:10]

train_prompts = []
for conv in train_convs:
    train_prompts.extend(extract_prompts_from_conv(conv, n_samples=15))

heldout_prompts = []
for conv in heldout_convs:
    heldout_prompts.extend(extract_prompts_from_conv(conv, n_samples=10))

print(f'train prompts: {len(train_prompts)}')
print(f'heldout prompts: {len(heldout_prompts)}')
print()
print('SAMPLE TRAIN PROMPT:')
print(train_prompts[0])
print()
print('SAMPLE HELDOUT PROMPT:')
print(heldout_prompts[0])

with open('train_prompts.json', 'w') as f:
    json.dump(train_prompts, f)
with open('heldout_prompts.json', 'w') as f:
    json.dump(heldout_prompts, f)
