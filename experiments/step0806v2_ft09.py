"""
step0806v2_ft09.py -- EpsilonPrediction806v2 on FT09 (R3_cf).

R3 hypothesis: 80% random explores FT09's 68 actions (dirs + clicks).
20% prediction-contrast occasionally picks informative (novel) clicks.
Does the epsilon mix help discover productive FT09 click positions?

FT09 floor: L1=0 at 25K steps (step807_ft09 baseline).
"""
import sys
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

from substrates.step0806v2 import EpsilonPrediction806v2
from r3cf_game_runner import run_r3cf_game

N_ACTIONS = 68  # FT09: 4 dirs + 64 clicks

result = run_r3cf_game(EpsilonPrediction806v2, "Step806v2_EpsilonPrediction_FT09",
                       game_name="FT09", n_actions=N_ACTIONS,
                       measure_prediction=True)

print()
print("STEP 806v2 FT09 DONE")
print(f"R3_cf (L1): {'PASS' if result['r3_cf_pass'] else ('FAIL' if result['r3_cf_pass'] is False else 'INCONCLUSIVE')}")
print(f"cold={result['total_cold']}  warm={result['total_warm']}")
if result.get('pred_r3_cf_pass') is not None:
    print(f"R3_cf (pred): {'PASS' if result['pred_r3_cf_pass'] else 'FAIL'}")
    print(f"pred_acc cold={result['mean_cold_acc']:.2f}%  warm={result['mean_warm_acc']:.2f}%")
