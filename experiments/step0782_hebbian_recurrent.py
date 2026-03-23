"""
step0782_hebbian_recurrent.py — R3_cf: Hebbian Recurrent Network.

R3 hypothesis: recurrent dynamics capture temporal structure → R3_cf > 0.
D(s) = {h, W_x, W_a}. L(s) = ∅. All Hebbian, no graph.

Kill: if h converges to fixed point before 5K steps.
"""
import sys, os
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

print("=" * 70)
print("STEP 782 — HEBBIAN RECURRENT NETWORK")
print("=" * 70)

from substrates.step0782 import HebbianRecurrent782
from r3cf_runner import run_r3cf

result = run_r3cf(HebbianRecurrent782, "Step782_HebbianRecurrent", n_actions=4)

print()
print("STEP 782 DONE")
print(f"R3_cf: {'PASS' if result['r3_cf_pass'] else 'FAIL' if result['r3_cf_pass'] is False else 'INCONCLUSIVE'}")
print(f"cold={result['total_cold']}  warm={result['total_warm']}  OR={result['odds_ratio']}  p={result['p_val']}")
