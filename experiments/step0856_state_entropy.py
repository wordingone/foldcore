"""
step0856_state_entropy.py -- R3_cf: StateEntropy856.

Picks action predicted to go to least-visited obs.
D(s) = {W, obs_histogram, running_mean}. Tests histogram transfer.
"""
import sys
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

from substrates.step0856 import StateEntropy856
from r3cf_runner import run_r3cf

result = run_r3cf(StateEntropy856, "Step856_StateEntropy", measure_prediction=True)

print()
print("STEP 856 DONE")
print(f"R3_cf (L1): {'PASS' if result['r3_cf_pass'] else ('FAIL' if result['r3_cf_pass'] is False else 'INCONCLUSIVE')}")
print(f"cold={result['total_cold']}  warm={result['total_warm']}")
if result.get('pred_r3_cf_pass') is not None:
    print(f"R3_cf (pred): {'PASS' if result['pred_r3_cf_pass'] else 'FAIL'}")
    print(f"pred_acc cold={result['mean_cold_acc']:.2f}%  warm={result['mean_warm_acc']:.2f}%")
