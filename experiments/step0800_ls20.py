"""
step0800_ls20.py -- PerActionChangePursuit800 on LS20 (R3_cf).

R3 hypothesis: On LS20, directional actions produce more observation change
than idle/wrong-direction actions. argmax(delta[a]) should converge on the
productive directions and navigate better than random.

LS20 baseline: 36.4/seed (random, step807). 806v2 warm: 78/seed.
Does per-action tracking beat epsilon-prediction?

Standard R3_cf (substrate_seed=0 for LS20 — not degenerate, game varies).
"""
import sys
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

from substrates.step0800 import PerActionChangePursuit800
from r3cf_runner import run_r3cf

result = run_r3cf(PerActionChangePursuit800, "Step800_PerActionChange_LS20",
                  measure_prediction=False)

print()
print("STEP 800 LS20 DONE")
print(f"R3_cf (L1): {'PASS' if result['r3_cf_pass'] else ('FAIL' if result['r3_cf_pass'] is False else 'INCONCLUSIVE')}")
print(f"cold={result['total_cold']}  warm={result['total_warm']}")
print(f"Baselines: random 182 (36.4/seed), 806v2_warm 390 (78/seed)")
