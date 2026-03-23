"""
step0809b_cycling_forward_model.py -- R3_cf test: CyclingForwardModel809b.

Action cycling (coverage) + W forward model (dynamics learning) in parallel.
R3_cf: prediction accuracy (warm W predicts better than cold W on test seeds).
L1 metric: does cycling + W navigate better than random (36.4/seed)?
"""
import sys
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

from substrates.step0809b import CyclingForwardModel809b
from r3cf_runner import run_r3cf

result = run_r3cf(CyclingForwardModel809b, "Step809b_CyclingForwardModel",
                  measure_prediction=True)

print()
print("STEP 809b DONE")
print(f"R3_cf (L1): {'PASS' if result['r3_cf_pass'] else ('FAIL' if result['r3_cf_pass'] is False else 'INCONCLUSIVE')}")
print(f"cold={result['total_cold']}  warm={result['total_warm']}")
if result.get('pred_r3_cf_pass') is not None:
    print(f"R3_cf (pred): {'PASS' if result['pred_r3_cf_pass'] else 'FAIL'}")
    print(f"pred_acc cold={result['mean_cold_acc']:.2f}%  warm={result['mean_warm_acc']:.2f}%")
