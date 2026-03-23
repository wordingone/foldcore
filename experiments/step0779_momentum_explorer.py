"""
step0779_momentum_explorer.py — R3_cf: Momentum Explorer + Forward Model.

R3 hypothesis: momentum (70% repeat) creates longer directional trajectories →
W learns richer directional dynamics → R3_cf > 778. D(s) = {W, running_mean}.
"""
import sys, os
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

print("=" * 70)
print("STEP 779 — MOMENTUM EXPLORER + FORWARD MODEL")
print("=" * 70)

from substrates.step0779 import MomentumExplorer779
from r3cf_runner import run_r3cf

result = run_r3cf(MomentumExplorer779, "Step779_MomentumExplorer", n_actions=4)

print()
print("STEP 779 DONE")
print(f"R3_cf: {'PASS' if result['r3_cf_pass'] else 'FAIL' if result['r3_cf_pass'] is False else 'INCONCLUSIVE'}")
print(f"cold={result['total_cold']}  warm={result['total_warm']}  OR={result['odds_ratio']}  p={result['p_val']}")
