"""
step0778_global_forward_model.py — R3_cf: Random + Global Forward Model.

R3 hypothesis: global W matrix captures transferable dynamics even under
random action selection. D(s) = {W, running_mean}. L(s) = ∅.

Kill: if R3_cf FAIL (warm <= cold), forward model under random exploration
doesn't encode transferable structure.
"""
import sys, os
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

print("=" * 70)
print("STEP 778 — GLOBAL FORWARD MODEL (RANDOM ACTION)")
print("=" * 70)

from substrates.step0778 import GlobalForwardModel778
from r3cf_runner import run_r3cf

result = run_r3cf(GlobalForwardModel778, "Step778_GlobalForwardModel", n_actions=4,
                  measure_prediction=True)

print()
print("STEP 778 DONE")
print(f"R3_cf: {'PASS' if result['r3_cf_pass'] else 'FAIL' if result['r3_cf_pass'] is False else 'INCONCLUSIVE'}")
print(f"cold={result['total_cold']}  warm={result['total_warm']}  OR={result['odds_ratio']}  p={result['p_val']}")
