"""
Joint cost model: draft-side (mobile) compute + network latency + server-side
compute, combined into one end-to-end estimate. Pure arithmetic on numbers
from three DISTINCT, EXPLICITLY LABELED sources -- no model execution, no
GPU/VM needed. Every number below is tagged [MEASURED] (ours, this session),
[PUBLISHED] (someone else's real hardware benchmark), or [ASSUMED]
(a simulated scenario, same approach DSSD/ConfigSpec used for network).

Goal: does SD-verified handoff still beat naive text handoff once a real
mobile drafter and real network round-trips replace our same-GPU,
zero-network test setup?
"""

# ---------------------------------------------------------------------------
# [MEASURED] -- ours, this session, greedy + direct-mapping flagship test
# Qwen2.5-0.5B-Instruct drafter / Qwen2.5-7B-Instruct target, 8-bit target,
# mixed-complexity dataset, 40 questions.
# ---------------------------------------------------------------------------
MAX_NEW = 60                    # tokens generated per turn (matches our test's MAX_NEW)
TOK_PER_CALL = 2.797             # avg tokens accepted per verification round (flagship test)
SERVER_TIME_PLAIN = 11.2         # seconds: target-alone autoregressive generation of
                                  # MAX_NEW tokens, GPU, no drafter involved at all --
                                  # representative figure from repeated examples across
                                  # tonight's tests (t_plain consistently ~11-12s at this
                                  # length on the 8-bit Qwen2.5-7B-Instruct target)

N_ROUNDS = MAX_NEW / TOK_PER_CALL                     # ~21.4 verification rounds needed
SERVER_TIME_PER_ROUND = SERVER_TIME_PLAIN / MAX_NEW    # [DERIVED, approximation] --
# treats one verification pass (covering ~TOK_PER_CALL candidate tokens) as costing
# about the same as one single-token autoregressive step. Justified by our OWN
# prefill-scaling finding earlier this session: at these small token counts the GPU
# forward pass is memory-bandwidth-bound, not compute-bound, so batching a handful
# of extra tokens into one pass costs only marginally more than one token alone.
# This is an approximation carried over from a real finding, not a fresh guess --
# flagged explicitly so it doesn't get mistaken for a directly re-measured number.

# ---------------------------------------------------------------------------
# [PUBLISHED] -- real phone benchmarks, sub-1B quantized models, 2026
# (llama.cpp-class CPU inference, Q4 quantization). NOT the Lever paper's
# flash-offloaded 7-14B numbers (1.5-3.4 tok/s) -- that's a different model
# class and a different bottleneck (storage I/O), would be the wrong number
# to borrow here. Qwen2.5-0.5B-Instruct is in the "sub-1B" bracket.
# ---------------------------------------------------------------------------
MOBILE_TOK_S_HEALTHY = 35        # midpoint of the reported 30-40 tok/s range,
                                  # sub-1B models, current-generation phone SoC
MOBILE_TOK_S_THROTTLED = 15      # [PUBLISHED, inferred] -- reflects the cited real
                                  # example (Llama 3.2 3B: 37.58 tok/s initial, degrading
                                  # under sustained thermal load) -- modeling a comparable
                                  # proportional falloff for a smaller, cooler-running model
                                  # under a long sustained SD session. Not a directly
                                  # published number for THIS model -- an assumption
                                  # informed by a published pattern, labeled as such.

# ---------------------------------------------------------------------------
# [ASSUMED] -- simulated network scenarios, same approach as DSSD/ConfigSpec
# (real compute measurements + injected network delay, not a live network)
# ---------------------------------------------------------------------------
NETWORK_SCENARIOS = {
    # NOTE: none of these rows reproduce our measured 1.394x -- that number
    # used a GPU-speed drafter with zero network latency (our actual test
    # setup), which is a DIFFERENT scenario from anything below. Every row
    # here substitutes a mobile-speed drafter, so even the 0ms-RTT row is a
    # hypothetical ("what if network were free but the drafter were phone-
    # speed"), not a sanity check against our real result. Labeled explicitly
    # to avoid exactly the mixing-up-numbers mistake this model needs to avoid.
    "0ms RTT (mobile drafter, hypothetical -- NOT our real test)": 0.000,
    "good wifi / 5G":                                               0.020,
    "typical 4G":                                                    0.075,
    "poor connectivity":                                             0.150,
}
PAYLOAD_TRANSMIT_TIME = 0.002    # [ASSUMED] -- a few hundred bytes (candidate token ids +
                                  # accept/reject response) at typical mobile bandwidth;
                                  # small enough that RTT dominates, included for completeness


def naive_handoff_time(rtt, mobile_tok_s):
    """Small model generates its own turn locally, one upload to the server,
    server generates its entire continuation with zero further edge involvement."""
    drafter_own_turn = MAX_NEW / mobile_tok_s          # mobile compute, once
    one_upload = rtt + PAYLOAD_TRANSMIT_TIME            # single network hop, once
    server_continuation = SERVER_TIME_PLAIN             # server compute, autoregressive, once
    return drafter_own_turn + one_upload + server_continuation


def sd_handoff_time(rtt, mobile_tok_s):
    """Small model generates its own turn locally (same as naive) -- THEN the
    target's continuation requires N_ROUNDS repeated (mobile draft + network
    round-trip + server verify) cycles instead of one uninterrupted server run."""
    drafter_own_turn = MAX_NEW / mobile_tok_s
    per_round_mobile = TOK_PER_CALL / mobile_tok_s
    per_round_network = rtt + PAYLOAD_TRANSMIT_TIME
    per_round_server = SERVER_TIME_PER_ROUND
    continuation = N_ROUNDS * (per_round_mobile + per_round_network + per_round_server)
    return drafter_own_turn + continuation


print(f"Rounds needed for a {MAX_NEW}-token response at tok/call={TOK_PER_CALL}: {N_ROUNDS:.1f}\n")
print(f"{'scenario':32s} {'mobile state':14s} {'naive (s)':>10s} {'SD (s)':>10s} {'speedup':>9s}")
print("-" * 80)
for label, rtt in NETWORK_SCENARIOS.items():
    for mstate, mtoks in [("healthy", MOBILE_TOK_S_HEALTHY), ("throttled", MOBILE_TOK_S_THROTTLED)]:
        t_naive = naive_handoff_time(rtt, mtoks)
        t_sd = sd_handoff_time(rtt, mtoks)
        speedup = t_naive / t_sd
        print(f"{label:32s} {mstate:14s} {t_naive:10.2f} {t_sd:10.2f} {speedup:8.3f}x")
    print()

print("=" * 80)
print("SERVER-SIDE COMPUTE ONLY (independent of mobile speed and network entirely)")
server_only_naive = SERVER_TIME_PLAIN
server_only_sd = N_ROUNDS * SERVER_TIME_PER_ROUND
print(f"naive: {server_only_naive:.2f}s of server compute per response")
print(f"SD:    {server_only_sd:.2f}s of server compute per response")
print(f"server-side compute reduction: {server_only_naive/server_only_sd:.3f}x "
      f"(matches TOK_PER_CALL={TOK_PER_CALL:.3f} almost exactly, as expected)")
