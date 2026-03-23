"""
step0809_ft09.py -- ObsHashCycling803 on FT09 (68 actions).

R3 hypothesis: systematic action cycling on FT09 guarantees all 68 actions
are tried at each observed state. FT09 productive clicks are sparse — cycling
eventually discovers them, unlike random exploration.

FT09 floor: L1=0 (random 68 actions, 25K steps).
If cycling > 0: deterministic coverage beats random for sparse-reward games.

Uses step0803 substrate (ObsHashCycling803) with n_actions=68.
"""
import sys
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

from substrates.step0803 import ObsHashCycling803
from r3cf_game_runner import run_r3cf_game

N_ACTIONS = 68  # FT09: 4 dirs + 64 clicks

result = run_r3cf_game(ObsHashCycling803, "Step809_FT09_ActionCycling",
                       game_name="FT09", n_actions=N_ACTIONS,
                       measure_prediction=False)

print()
print("STEP 809 FT09 DONE")
print(f"R3_cf (L1): {'PASS' if result['r3_cf_pass'] else ('FAIL' if result['r3_cf_pass'] is False else 'INCONCLUSIVE')}")
print(f"cold={result['total_cold']}  warm={result['total_warm']}")
print(f"FT09 floor: 0 (step807_ft09)")
