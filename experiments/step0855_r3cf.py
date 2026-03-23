"""
step0855_r3cf.py -- R3_cf protocol for CompressionProgress855.

Tests: does W (forward model) + progress_ema (D(s)) transfer?
Cold = fresh substrate. Warm = pretrained on seeds 1-5.
"""
import sys
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

from substrates.step0855 import CompressionProgress855
from r3cf_runner import run_r3cf

result = run_r3cf(CompressionProgress855, "Step855_CompressionProgress",
                  measure_prediction=True)

print()
print("STEP 855 R3_cf DONE")
print(f"R3_cf (L1): {'PASS' if result['r3_cf_pass'] else 'FAIL'}")
print(f"cold={result['total_cold']}  warm={result['total_warm']}")
if result.get('pred_r3_cf_pass') is not None:
    print(f"R3_cf (pred): {'PASS' if result['pred_r3_cf_pass'] else 'FAIL'}")
    print(f"pred_acc cold={result['mean_cold_acc']:.2f}%  warm={result['mean_warm_acc']:.2f}%")
