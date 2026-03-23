"""
step0787_reservoir_computing.py — R3_cf: Reservoir Computing with decay.

R3 hypothesis: reservoir (fixed W_res) + Hebbian W_out with decay avoids
convergence. D(s) = {h, W_out}. L(s) = ∅.
h' = tanh(W_res @ h + W_in @ obs). W_out *= decay each step.

Kill: if h converges despite decay (spectral radius check).
"""
import sys, os
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

print("=" * 70)
print("STEP 787 — RESERVOIR COMPUTING (FIXED RES + HEBBIAN DECAY)")
print("=" * 70)

from substrates.step0787 import ReservoirComputing787
from r3cf_runner import run_r3cf

result = run_r3cf(ReservoirComputing787, "Step787_ReservoirComputing", n_actions=4)

print()
print("STEP 787 DONE")
print(f"R3_cf: {'PASS' if result['r3_cf_pass'] else 'FAIL' if result['r3_cf_pass'] is False else 'INCONCLUSIVE'}")
print(f"cold={result['total_cold']}  warm={result['total_warm']}  OR={result['odds_ratio']}  p={result['p_val']}")
