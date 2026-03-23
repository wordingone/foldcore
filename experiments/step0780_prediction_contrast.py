"""
step0780_prediction_contrast.py — R3_cf: Prediction-Contrast Action Selection.

R3 hypothesis: choosing actions that maximize predicted state change finds novel
states without visit counts. D(s) = {W, running_mean}. L(s) = ∅.
Action: argmax_a ||W(obs,a) - obs||.

Kill: if L1=0/5 AND R3_cf FAIL.
"""
import sys, os
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

print("=" * 70)
print("STEP 780 — PREDICTION-CONTRAST ACTION SELECTION")
print("=" * 70)

from substrates.step0780 import PredictionContrast780
from r3cf_runner import run_r3cf

result = run_r3cf(PredictionContrast780, "Step780_PredictionContrast", n_actions=4,
                  measure_prediction=True)

print()
print("STEP 780 DONE")
print(f"R3_cf: {'PASS' if result['r3_cf_pass'] else 'FAIL' if result['r3_cf_pass'] is False else 'INCONCLUSIVE'}")
print(f"cold={result['total_cold']}  warm={result['total_warm']}  OR={result['odds_ratio']}  p={result['p_val']}")
