"""
step0848_max_entropy.py -- R3_cf test: MaxEntropy848.

Max-entropy action selection with novelty-based logit updates.
D(s) = {action_logits, obs_histogram, running_mean}.
"""
import sys
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

from substrates.step0848 import MaxEntropy848
from r3cf_runner import run_r3cf

result = run_r3cf(MaxEntropy848, "Step848_MaxEntropy")

print()
print("STEP 848 DONE")
print(f"R3_cf (L1): {'PASS' if result['r3_cf_pass'] else ('FAIL' if result['r3_cf_pass'] is False else 'INCONCLUSIVE')}")
print(f"cold={result['total_cold']}  warm={result['total_warm']}")
