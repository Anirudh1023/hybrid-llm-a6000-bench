"""
Feasibility check for OmniDraft's remaining unbuilt piece: the online n-gram
cache. Before investing 3-5 hours building the actual caching/lookup
machinery into a CandidateGenerator, this answers the load-bearing question
directly: in a realistic multi-turn conversation (the shape this whole
session pivoted to), does enough token-level repetition actually occur for
an n-gram cache to matter?

Generates several extended, growing multi-turn Q&A conversations with the
SAME target model used throughout tonight (Qwen2.5-7B-Instruct) -- SD-
verified and plain generation are provably token-identical under greedy
decoding (established earlier this session), so using plain generation here
is representative of what the cache would actually see, without extra
complexity.

For each conversation, tokenizes the full accumulated transcript and, for
each generation position, checks whether the preceding n-token context has
occurred earlier in the SAME conversation:
  - coverage: fraction of positions where that context has a prior
    occurrence at all (how often the cache would even have a candidate)
  - hit_rate: of those, how often the token that followed the earlier
    occurrence matches the actual next token (how often that candidate
    would actually be useful, vs a wasted/rejected guess)
  - coverage * hit_rate: overall fraction of ALL tokens where the cache
    would supply a correct, useful draft -- the real bottom-line number
"""
import gc
import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

TARGET_ID = "Qwen/Qwen2.5-7B-Instruct"
MAX_NEW = 60
NGRAM_LENGTHS = [3, 4, 5]

TOPICS = [
    {
        "name": "learning to cook",
        "questions": [
            "What are some basic knife skills every beginner cook should learn?",
            "How long does it typically take to get comfortable with those skills?",
            "What's a simple recipe I could practice these skills on?",
            "Once I've mastered the basics, what should I learn next?",
            "What are some common mistakes beginners make with seasoning?",
            "How do I know when meat is properly cooked without a thermometer?",
            "What kitchen equipment is actually worth investing in early on?",
            "How can I make my meals healthier without sacrificing flavor?",
        ],
    },
    {
        "name": "starting a home garden",
        "questions": [
            "What are the easiest vegetables for a beginner to grow at home?",
            "How much sunlight do those vegetables actually need each day?",
            "What's the biggest mistake beginners make when starting a garden?",
            "How do I know when it's actually time to harvest them?",
            "What kind of soil works best for a beginner vegetable garden?",
            "How often should I actually be watering my plants?",
            "What are some natural ways to deal with garden pests?",
            "How do I prepare my garden for the changing seasons?",
        ],
    },
    {
        "name": "training for a first 5k run",
        "questions": [
            "How should a complete beginner start training for a 5k race?",
            "How many days a week should I actually be running at first?",
            "What should I do if my knees start hurting during training?",
            "How do I know if I'm ready to race, versus needing more time?",
            "What should I eat before and after a training run?",
            "How important is stretching before and after running?",
            "What kind of shoes should a beginner runner actually get?",
            "How do I stay motivated when training gets repetitive?",
        ],
    },
    {
        "name": "basic personal budgeting",
        "questions": [
            "What's the simplest way for a beginner to start budgeting?",
            "How much of my income should actually go toward savings?",
            "What's the biggest mistake people make when starting a budget?",
            "How do I build an emergency fund if I'm living paycheck to paycheck?",
            "Should I pay off debt or save money first?",
            "What's a realistic way to track my spending without it being a chore?",
            "How do I budget for irregular expenses like car repairs?",
            "How do I know if my budget is actually working?",
        ],
    },
]

target_tokenizer = AutoTokenizer.from_pretrained(TARGET_ID)
bnb_config = BitsAndBytesConfig(load_in_8bit=True)
target_model = AutoModelForCausalLM.from_pretrained(TARGET_ID, quantization_config=bnb_config, device_map={"": 0})


def build_chat_prompt(history, new_question):
    messages = []
    for q, a in history:
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": new_question})
    return target_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def gen_answer(history, question, max_new=MAX_NEW):
    prompt = build_chat_prompt(history, question)
    inputs = target_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(target_model.device)
    with torch.no_grad():
        out = target_model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    text = target_tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return text


def compute_ngram_stats(tokens, n):
    context_to_next = {}
    hits = 0
    checked = 0
    for i in range(n, len(tokens)):
        context = tuple(tokens[i - n:i])
        actual_next = tokens[i]
        if context in context_to_next:
            checked += 1
            if actual_next in context_to_next[context]:
                hits += 1
        context_to_next.setdefault(context, set()).add(actual_next)
    total_positions = len(tokens) - n
    coverage = checked / total_positions if total_positions else 0.0
    hit_rate = hits / checked if checked else 0.0
    return {"n": n, "total_positions": total_positions, "checked": checked, "hits": hits,
            "coverage": coverage, "hit_rate": hit_rate, "overall_useful_rate": coverage * hit_rate}


results = []
for topic in TOPICS:
    print(f"\n{'='*70}\n[topic: {topic['name']}]\n", flush=True)
    history = []
    for q in topic["questions"]:
        a = gen_answer(history, q)
        print(f"Q: {q}\nA: {a[:150]}\n", flush=True)
        history.append((q, a))
        gc.collect(); torch.cuda.empty_cache()

    # Tokenize the full accumulated ANSWER stream (assistant turns concatenated) --
    # this is what a real generation stream looks like token-by-token, the
    # thing the drafter/cache would actually be predicting.
    full_answer_text = "\n".join(a for _, a in history)
    tokens = target_tokenizer.encode(full_answer_text)
    print(f"total answer tokens for topic: {len(tokens)}", flush=True)

    topic_ngram_stats = [compute_ngram_stats(tokens, n) for n in NGRAM_LENGTHS]
    for s in topic_ngram_stats:
        print(f"  n={s['n']}: coverage={s['coverage']:.3f} hit_rate={s['hit_rate']:.3f} "
              f"overall_useful_rate={s['overall_useful_rate']:.3f} (checked {s['checked']}/{s['total_positions']})", flush=True)

    results.append({"topic": topic["name"], "history": history, "n_tokens": len(tokens), "ngram_stats": topic_ngram_stats})

print(f"\n{'='*70}\n=== OVERALL SUMMARY ({len(TOPICS)} topics) ===", flush=True)
for n in NGRAM_LENGTHS:
    covs = [r["ngram_stats"][NGRAM_LENGTHS.index(n)]["coverage"] for r in results]
    hits = [r["ngram_stats"][NGRAM_LENGTHS.index(n)]["hit_rate"] for r in results]
    useful = [r["ngram_stats"][NGRAM_LENGTHS.index(n)]["overall_useful_rate"] for r in results]
    print(f"n={n}: avg coverage={sum(covs)/len(covs):.3f}  avg hit_rate={sum(hits)/len(hits):.3f}  "
          f"avg overall_useful_rate={sum(useful)/len(useful):.3f}", flush=True)

with open("ngram_feasibility_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved to ngram_feasibility_results.json", flush=True)
