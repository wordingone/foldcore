"""
step0806v2_ls20.py -- EpsilonPrediction806v2 on LS20 (R3_cf).

R3 hypothesis: 80% random provides persistence for LS20 (random gets 164/seed).
20% prediction-contrast uses W to occasionally choose novel actions.
Combined: should preserve LS20 L1 while adding pred accuracy transfer.

Compare: random baseline (164/seed), step780v5 (0/seed, 73% pred improvement).
"""
import sys
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

from substrates.step0806v2 import EpsilonPrediction806v2
from r3cf_runner import run_r3cf

result = run_r3cf(EpsilonPrediction806v2, "Step806v2_EpsilonPrediction_LS20",
                  measure_prediction=True)

print()
print("STEP 806v2 LS20 DONE")
print(f"R3_cf (L1): {'PASS' if result['r3_cf_pass'] else ('FAIL' if result['r3_cf_pass'] is False else 'INCONCLUSIVE')}")
print(f"cold={result['total_cold']}  warm={result['total_warm']}")
print(f"(baseline: ~182 expected from 36.4/seed x 5 seeds)")
if result.get('pred_r3_cf_pass') is not None:
    print(f"R3_cf (pred): {'PASS' if result['pred_r3_cf_pass'] else 'FAIL'}")
    print(f"pred_acc cold={result['mean_cold_acc']:.2f}%  warm={result['mean_warm_acc']:.2f}%")
