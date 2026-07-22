"""
config.vocab_size gating check -- the single most surprising discovery of
the T4 session (2026-07-22/23).

HF's assisted-generation code decides whether to use the fast NATIVE path
(no translator) or the CROSS-VOCAB path (tokenizer=/assistant_tokenizer=
required) based ENTIRELY on:

    self.config.get_text_config().vocab_size == assistant_model.config.get_text_config().vocab_size

NOT on whether the tokenizers are actually the same class, NOT on real
vocab-key overlap. Two checkpoints of the LITERAL SAME tokenizer (e.g.
Qwen2.5-0.5B-Instruct and Qwen2.5-7B-Instruct) can have DIFFERENT
config.vocab_size (151936 vs 152064) purely from different embedding-table
padding -- and HF will force the cross-vocab path for them anyway, even
though the real tokenizer is 100% identical.

This bit us hard: every "same family" comparison attempted on the T4 setup
turned out to secretly be running through the cross-vocab translator, not
the native fast path, invalidating an early conclusion ("same-family SD
performed worse than cross-family") that had to be retracted once this was
found. ALWAYS use this check before assuming a pair will hit the native path.
"""

from transformers import AutoConfig


def check_native_path(target_id: str, assistant_id: str) -> dict:
    """Returns whether (target_id, assistant_id) will use HF's native,
    zero-translator assisted-generation path. Check this BEFORE running any
    'same-family' or 'native SD' experiment -- don't assume from tokenizer
    class or vocab-key overlap alone."""
    t_cfg = AutoConfig.from_pretrained(target_id)
    a_cfg = AutoConfig.from_pretrained(assistant_id)
    t_vocab = t_cfg.get_text_config().vocab_size
    a_vocab = a_cfg.get_text_config().vocab_size
    native = t_vocab == a_vocab
    return {
        "target_vocab_size": t_vocab,
        "assistant_vocab_size": a_vocab,
        "native_path": native,
    }


def find_matching_pair_in_family(model_ids: list[str]) -> dict:
    """Given a list of same-family model IDs (different sizes), groups them
    by config.vocab_size so you can pick a genuinely native-path-eligible
    pair for a same-family control experiment. Print the result and manually
    pick two IDs from the same group."""
    groups: dict[int, list[str]] = {}
    for mid in model_ids:
        cfg = AutoConfig.from_pretrained(mid)
        vs = cfg.get_text_config().vocab_size
        groups.setdefault(vs, []).append(mid)
    return groups


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 3:
        result = check_native_path(sys.argv[1], sys.argv[2])
        print(result)
    else:
        print("Usage: python vocab_utils.py <target_model_id> <assistant_model_id>")
