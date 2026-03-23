"""
step0788_global_action_balance.py -- R3_cf test: GlobalActionBalance788.

R3 hypothesis: global action count (D(s)) transfers across seeds.
argmin(action_count) balances globally -- not per-state.
"""
import sys
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

from substrates.step0788 import GlobalActionBalance788
from r3cf_runner import run_r3cf

result = run_r3cf(GlobalActionBalance788, "Step788_GlobalActionBalance")

print()
print("STEP 788 DONE")
print(f"R3_cf: {'PASS' if result['r3_cf_pass'] else 'FAIL'}")
print(f"cold={result['total_cold']}  warm={result['total_warm']}  OR={result.get('odds_ratio', 'N/A')}  p={result.get('p_val', 'N/A')}")
