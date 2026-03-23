"""
step0783_transition_hash_set.py — R3_cf: Transition Hash Set.

R3 hypothesis: tracking which transitions occurred (not counts) enables
exploration without visit counts. Graph ban PASS: keyed by (obs, obs_next).
D(s) = {W, seen_transitions, running_mean}. L(s) = ∅.

Kill: if set growth → memory blow-up before 10K steps.
"""
import sys, os
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

print("=" * 70)
print("STEP 783 — TRANSITION HASH SET")
print("=" * 70)

from substrates.step0783 import TransitionHashSet783
from r3cf_runner import run_r3cf

result = run_r3cf(TransitionHashSet783, "Step783_TransitionHashSet", n_actions=4)

print()
print("STEP 783 DONE")
print(f"R3_cf: {'PASS' if result['r3_cf_pass'] else 'FAIL' if result['r3_cf_pass'] is False else 'INCONCLUSIVE'}")
print(f"cold={result['total_cold']}  warm={result['total_warm']}  OR={result['odds_ratio']}  p={result['p_val']}")
