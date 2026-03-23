"""
step0780_fam.py -- R3_cf: FamiliarSuccessor780fam on LS20.

argmin_a ||W(enc,a)||^2 (predict toward running mean / zero vector).
"Go home" policy. Leo mail 2582: complement to 780 (novelty) and 780inv.

Compare to:
- step780v5: argmax (novelty) — 0/seed L1, +73% pred
- step780inv: argmin to current enc — 0/seed L1, +6.6% pred
- Random baseline: ~164/seed L1
"""
import sys
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

from substrates.step0780_fam import FamiliarSuccessor780fam
from r3cf_runner import run_r3cf

result = run_r3cf(FamiliarSuccessor780fam, "Step780_fam_FamiliarSuccessor",
                  measure_prediction=True)

print()
print("STEP 780_fam DONE")
print(f"R3_cf (L1): {'PASS' if result['r3_cf_pass'] else ('FAIL' if result['r3_cf_pass'] is False else 'INCONCLUSIVE')}")
print(f"cold={result['total_cold']}  warm={result['total_warm']}")
print(f"(baseline: ~182 expected from 36.4/seed x 5 seeds)")
if result.get('pred_r3_cf_pass') is not None:
    print(f"R3_cf (pred): {'PASS' if result['pred_r3_cf_pass'] else 'FAIL'}")
    print(f"pred_acc cold={result['mean_cold_acc']:.2f}%  warm={result['mean_warm_acc']:.2f}%")
