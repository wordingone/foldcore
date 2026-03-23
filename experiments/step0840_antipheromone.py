"""
step0840_antipheromone.py -- R3_cf test: AntColony840.

Anti-pheromone: per-obs pheromone map (not per-(obs,action)).
Action = predicted successor with lowest pheromone.
Compare to 36.4/seed random baseline.
"""
import sys
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

from substrates.step0840 import AntColony840
from r3cf_runner import run_r3cf

result = run_r3cf(AntColony840, "Step840_AntColony", measure_prediction=True)

print()
print("STEP 840 DONE")
print(f"R3_cf (L1): {'PASS' if result['r3_cf_pass'] else ('FAIL' if result['r3_cf_pass'] is False else 'INCONCLUSIVE')}")
print(f"cold={result['total_cold']}  warm={result['total_warm']}")
if result.get('pred_r3_cf_pass') is not None:
    print(f"R3_cf (pred): {'PASS' if result['pred_r3_cf_pass'] else 'FAIL'}")
