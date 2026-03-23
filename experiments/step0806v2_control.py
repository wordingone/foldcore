"""
step0806v2_control.py -- Control for step806v2: different substrate seeds.

Hypothesis: step806v2 L1 PASS (cold=0, warm=390) may be a substrate_seed=0
artifact. Cold W with seed=0 init may systematically prefer action 0 (ties go
to first element), causing L1=0. Warm W with seed=1, 2, 3 may show different
cold baselines.

Tests substrate_seed = 1, 2, 3. For each:
- Cold: fresh W with that seed
- Warm: pretrained W with that seed on LS20 seeds 1-5

If cold > 0 for seed=1,2,3: the seed=0 artifact confirmed.
If cold = 0 for all seeds: the cold=0 result is structural, not accidental.
"""
import sys, time
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from substrates.step0806v2 import EpsilonPrediction806v2
from r3cf_runner import run_r3cf

SUBSTRATE_SEEDS = [1, 2, 3]

print("=" * 70)
print("STEP 806v2 CONTROL — substrate_seed=1,2,3")
print("=" * 70)
print("Checking if cold=0 result is substrate_seed=0 artifact or structural.")
print()

for ss in SUBSTRATE_SEEDS:
    print(f"\n--- substrate_seed={ss} ---")
    result = run_r3cf(EpsilonPrediction806v2, f"Step806v2_ctrl_seed{ss}",
                      measure_prediction=True, substrate_seed=ss)
    l1_status = 'PASS' if result['r3_cf_pass'] else ('FAIL' if result['r3_cf_pass'] is False else 'INCONCLUSIVE')
    print(f"  L1: {l1_status}  cold={result['total_cold']}  warm={result['total_warm']}")
    if result.get('pred_r3_cf_pass') is not None:
        print(f"  Pred: {'PASS' if result['pred_r3_cf_pass'] else 'FAIL'}  cold={result['mean_cold_acc']:.2f}%  warm={result['mean_warm_acc']:.2f}%")

print()
print("STEP 806v2 CONTROL DONE")
