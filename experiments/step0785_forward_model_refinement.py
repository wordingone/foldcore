"""
step0785_forward_model_refinement.py — R3_cf: Forward Model + Refinement.

R3 hypothesis: W (forward model) + ref (encoding adaptation) = two active D
components → stronger R3_cf than either alone.
D(s) = {W, running_mean, ref}. L(s) = ∅.
"""
import sys, os
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

print("=" * 70)
print("STEP 785 — FORWARD MODEL + TRANSITION REFINEMENT")
print("=" * 70)

from substrates.step0785 import ForwardModelRefinement785
from r3cf_runner import run_r3cf

result = run_r3cf(ForwardModelRefinement785, "Step785_ForwardModelRefinement", n_actions=4)

print()
print("STEP 785 DONE")
print(f"R3_cf: {'PASS' if result['r3_cf_pass'] else 'FAIL' if result['r3_cf_pass'] is False else 'INCONCLUSIVE'}")
print(f"cold={result['total_cold']}  warm={result['total_warm']}  OR={result['odds_ratio']}  p={result['p_val']}")
