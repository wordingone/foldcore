"""
step0780_ft09.py -- R3_cf on FT09: PredictionContrast780 with 68 actions.

Critical test: does prediction-contrast navigate FT09 better than random?
FT09 requires discovering productive click positions (68 actions: 4 dirs + 64 clicks).
Prediction-contrast should help: productive clicks cause large obs changes.

R3_cf: warm W trained on FT09 seeds 1-5. Cold = fresh W. Test seeds 6-10.
Measures: L1 completions AND prediction accuracy.
"""
import sys
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

from substrates.step0780 import PredictionContrast780
from r3cf_game_runner import run_r3cf_game

N_ACTIONS = 68  # FT09: 4 dirs + 64 clicks

result = run_r3cf_game(PredictionContrast780, "Step780_FT09_PredContrast",
                       game_name="FT09", n_actions=N_ACTIONS,
                       measure_prediction=True)

print()
print("STEP 780 FT09 DONE")
print(f"R3_cf (L1): {'PASS' if result['r3_cf_pass'] else ('FAIL' if result['r3_cf_pass'] is False else 'INCONCLUSIVE')}")
print(f"cold={result['total_cold']}  warm={result['total_warm']}")
if result.get('pred_r3_cf_pass') is not None:
    print(f"R3_cf (pred): {'PASS' if result['pred_r3_cf_pass'] else 'FAIL'}")
    print(f"pred_acc cold={result['mean_cold_acc']:.2f}%  warm={result['mean_warm_acc']:.2f}%")
