"""
step0803_obs_hash_cycling.py -- R3_cf test: ObsHashCycling803.

R3 hypothesis: per-obs cycling counter (D(s)) transfers.
Cold substrate starts from idx=0 everywhere; warm knows which
actions were already tried at each obs hash.
"""
import sys
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

from substrates.step0803 import ObsHashCycling803
from r3cf_runner import run_r3cf

result = run_r3cf(ObsHashCycling803, "Step803_ObsHashCycling")

print()
print("STEP 803 DONE")
print(f"R3_cf: {'PASS' if result['r3_cf_pass'] else 'FAIL'}")
print(f"cold={result['total_cold']}  warm={result['total_warm']}  OR={result.get('odds_ratio', 'N/A')}  p={result.get('p_val', 'N/A')}")
