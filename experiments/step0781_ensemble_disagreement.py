"""
step0781_ensemble_disagreement.py — R3_cf: Ensemble Disagreement Explorer.

R3 hypothesis: K=3 ensemble variance correlates with novelty without visit counts.
D(s) = {W_1, W_2, W_3, running_mean}. L(s) = ∅.
"""
import sys, os
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

print("=" * 70)
print("STEP 781 — ENSEMBLE DISAGREEMENT EXPLORER (K=3)")
print("=" * 70)

from substrates.step0781 import EnsembleDisagreement781
from r3cf_runner import run_r3cf

result = run_r3cf(EnsembleDisagreement781, "Step781_EnsembleDisagreement", n_actions=4)

print()
print("STEP 781 DONE")
print(f"R3_cf: {'PASS' if result['r3_cf_pass'] else 'FAIL' if result['r3_cf_pass'] is False else 'INCONCLUSIVE'}")
print(f"cold={result['total_cold']}  warm={result['total_warm']}  OR={result['odds_ratio']}  p={result['p_val']}")
