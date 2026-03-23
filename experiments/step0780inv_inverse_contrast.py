"""
step0780inv_inverse_contrast.py -- R3_cf: InversePredictionContrast780inv.

Argmin predicted change (familiar regions). Opposite of step780 (argmax=novelty).
Tests if LS20 exits are predictable/familiar states.
Compare to 36.4/seed random baseline.
"""
import sys
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

from substrates.step0780_inv import InversePredictionContrast780inv
from r3cf_runner import run_r3cf

result = run_r3cf(InversePredictionContrast780inv, "Step780inv_InverseContrast",
                  measure_prediction=True)

print()
print("STEP 780inv DONE")
print(f"R3_cf (L1): {'PASS' if result['r3_cf_pass'] else ('FAIL' if result['r3_cf_pass'] is False else 'INCONCLUSIVE')}")
print(f"cold={result['total_cold']}  warm={result['total_warm']}")
print(f"(baseline: ~182 expected from 36.4/seed × 5 seeds)")
if result.get('pred_r3_cf_pass') is not None:
    print(f"R3_cf (pred): {'PASS' if result['pred_r3_cf_pass'] else 'FAIL'}")
    print(f"pred_acc cold={result['mean_cold_acc']:.2f}%  warm={result['mean_warm_acc']:.2f}%")
