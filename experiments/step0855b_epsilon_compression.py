"""
step0855b_epsilon_compression.py -- R3_cf test: EpsilonCompressionProgress855b.

80% compression progress + 20% random. Tests if diversity fix prevents action collapse.
Compare to 807 baseline (36.4 L1/seed). Also measures prediction accuracy R3_cf.
"""
import sys
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

from substrates.step0855b import EpsilonCompressionProgress855b
from r3cf_runner import run_r3cf

result = run_r3cf(EpsilonCompressionProgress855b, "Step855b_EpsilonCompression",
                  measure_prediction=True)

print()
print("STEP 855b DONE")
print(f"R3_cf (L1): {'PASS' if result['r3_cf_pass'] else ('FAIL' if result['r3_cf_pass'] is False else 'INCONCLUSIVE')}")
print(f"cold={result['total_cold']}  warm={result['total_warm']}")
print(f"(baseline: cold ~182 expected from 36.4/seed random)")
if result.get('pred_r3_cf_pass') is not None:
    print(f"R3_cf (pred): {'PASS' if result['pred_r3_cf_pass'] else 'FAIL'}")
    print(f"pred_acc cold={result['mean_cold_acc']:.2f}%  warm={result['mean_warm_acc']:.2f}%")
