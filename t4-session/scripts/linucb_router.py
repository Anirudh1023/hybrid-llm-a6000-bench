"""
LinUCB contextual bandit router (OrcaRouter pattern), 2-arm case:
  arm 0 = small_alone   (small model answers, no escalation)
  arm 1 = escalate_sd   (escalate to big model, run under speculative decoding)

Core algorithm is the reusable part -- Sherman-Morrison rank-1 updates avoid
recomputing a full matrix inverse every round. See hybrid-llm-research-notes.md,
"SETTLED (2026-07-21): the concrete router mechanism" for the design rationale
and why this beats a hand-rolled classifier (partial-feedback problem is
handled natively by LinUCB's exploration bonus, not something we invented a
fix for).
"""

import numpy as np


class LinUCBRouter:
    def __init__(self, n_features, n_arms=2, alpha=1.0, ridge_lambda=1.0):
        self.n_features = n_features
        self.n_arms = n_arms
        self.alpha = alpha  # exploration strength; higher = more exploration
        self.A = [ridge_lambda * np.eye(n_features) for _ in range(n_arms)]
        self.A_inv = [np.eye(n_features) / ridge_lambda for _ in range(n_arms)]
        self.b = [np.zeros(n_features) for _ in range(n_arms)]

    def warm_start(self, arm, X, y, ridge_lambda=1.0):
        """Offline ridge-regression init on labeled (context, reward) pairs for one arm."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.A[arm] = ridge_lambda * np.eye(self.n_features) + X.T @ X
        self.A_inv[arm] = np.linalg.inv(self.A[arm])
        self.b[arm] = X.T @ y

    def select_arm(self, x):
        x = np.asarray(x, dtype=float)
        scores = []
        for a in range(self.n_arms):
            theta = self.A_inv[a] @ self.b[a]
            mean = theta @ x
            bonus = self.alpha * np.sqrt(max(x @ self.A_inv[a] @ x, 0.0))
            scores.append(mean + bonus)
        return int(np.argmax(scores)), scores

    def update(self, arm, x, reward):
        """Sherman-Morrison rank-1 update: O(d^2), no matrix inversion each round."""
        x = np.asarray(x, dtype=float)
        Ainv_x = self.A_inv[arm] @ x
        denom = 1.0 + x @ Ainv_x
        self.A_inv[arm] = self.A_inv[arm] - np.outer(Ainv_x, Ainv_x) / denom
        self.A[arm] = self.A[arm] + np.outer(x, x)
        self.b[arm] = self.b[arm] + reward * x


def context_features(prompt, drafter_confidence):
    """
    Kept deliberately simple, matching the project's start-simple discipline:
      [bias, normalized query length, drafter confidence/entropy signal]
    drafter_confidence: the small model's own average token confidence on a
    quick draft of the query -- the heuristic already tried and shown to be
    an unreliable SOLE decision signal, reused here as one input FEATURE
    among several rather than the whole decision.
    """
    query_len_tokens = len(prompt.split())
    return np.array([1.0, min(query_len_tokens / 50.0, 2.0), drafter_confidence])


# Real, measured cost constants (2026-07-23), replacing the earlier arbitrary
# placeholders (escalate_cost=0.3, small_alone baseline=-0.1). Computed from
# actual wall-clock timings collected this session:
#   avg escalate (SD, trained drafter) time: 16.50s
#   avg small_alone (drafter-only generation) time: 3.21s
#   ratio: escalate costs ~5.14x what small_alone costs, in real wall-clock terms
# Normalized against escalate=1.0 (the more expensive option, by construction),
# small_alone's real relative cost is ~0.195 -- NOT the same as the old 0.3/0.1
# split, which had the right rough shape (escalate costs more) but no basis in
# actual measurement. See vm-artifacts/results_2026-07-23/ for the source data.
ESCALATE_COST = 1.0
SMALL_ALONE_COST = 0.1946


def escalation_reward(tok_per_call, max_draft_len=5.0, escalate_cost=ESCALATE_COST):
    """
    Router-calibration insight (see notes, "SECOND FREE BYPRODUCT"): HIGH
    realized acceptance/tok-per-call means the small model probably could
    have handled the turn alone -- escalation was likely unnecessary. LOW
    acceptance validates that escalation was the right call. So reward is
    NOT "high tok/call = good" -- it's "correction actually needed" minus
    the REAL cost of escalating (see ESCALATE_COST derivation above).
      correction_need = 1 - normalized(tok_per_call)   [low tok/call -> high correction_need]
      reward = correction_need - escalate_cost
    """
    normalized = (tok_per_call - 1.0) / max(max_draft_len - 1.0, 1e-6)
    normalized = min(max(normalized, 0.0), 1.0)
    correction_need = 1.0 - normalized
    return correction_need - escalate_cost


def small_alone_reward(f1_similarity, small_alone_cost=SMALL_ALONE_COST):
    """
    Quality signal (F1 similarity to the real next conversational turn, see
    uag_small_alone_reward.py) minus the REAL measured cost of staying local
    (see SMALL_ALONE_COST derivation above) -- replaces the earlier
    drafter-confidence-based placeholder now that we have both a real quality
    proxy AND a real cost number for this arm.
    """
    return f1_similarity - small_alone_cost


if __name__ == '__main__':
    import json

    # All three sources are real, measured on the SAME 30-prompt set (verified
    # aligned -- see notes, 2026-07-22 "data-alignment" entry for why this
    # needed a dedicated re-measurement rather than reusing the original
    # heldout_prompts.json, which turned out to be a different 30 prompts
    # despite same seed/logic, likely due to a locomo10.json content diff
    # between whenever that file was first generated and tonight's fresh clone).
    with open('heldout_pairs_groundtruth.json') as f:
        pairs = json.load(f)
    with open('escalate_rewards_aligned.json') as f:
        escalate_tok_per_call = json.load(f)          # real, trained-adapter tok/call
    with open('small_alone_rewards.json') as f:
        small_alone_f1 = json.load(f)                  # real, F1 vs actual next turn

    prompts = [p['prompt'] for p in pairs]
    small_alone_f1_scores = [r['reward'] for r in small_alone_f1]

    # Confidence feature still needs a live drafter forward pass to be fully
    # real (not yet wired up here) -- everything else in this demo is real,
    # measured data; this one input feature remains a placeholder, flagged
    # explicitly rather than silently treated as real.
    rng = np.random.default_rng(42)
    placeholder_confidences = rng.uniform(0.4, 0.9, size=len(prompts))

    router = LinUCBRouter(n_features=3, n_arms=2, alpha=0.5, ridge_lambda=1.0)

    print(f'=== LinUCB router demo: {len(prompts)} rounds, real aligned data (tok/call + F1), 1 placeholder feature ===')
    escalate_chosen = small_alone_chosen = 0
    for i, (prompt, tpc, f1, conf) in enumerate(zip(prompts, escalate_tok_per_call, small_alone_f1_scores, placeholder_confidences)):
        x = context_features(prompt, conf)
        arm, scores = router.select_arm(x)
        arm_name = 'small_alone' if arm == 0 else 'escalate_sd'
        if arm == 1:
            reward = escalation_reward(tpc)
            escalate_chosen += 1
        else:
            reward = small_alone_reward(f1)
            small_alone_chosen += 1
        router.update(arm, x, reward)
        print(f'[{i:2d}] chose={arm_name:12s} scores={[f"{s:.3f}" for s in scores]} reward={reward:+.3f}')

    print(f'\nArm selection counts: escalate_sd={escalate_chosen} small_alone={small_alone_chosen}')
    print('Final theta (escalate_sd arm):', router.A_inv[1] @ router.b[1])
    print('Final theta (small_alone arm):', router.A_inv[0] @ router.b[0])
