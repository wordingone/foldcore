"""
step0784_encoding_only.py — R3_cf: Encoding-Only (minimal D-only substrate).

R3 hypothesis: running mean adaptation alone produces positive R3_cf.
D(s) = {running_mean}. L(s) = ∅.
Minimal D-only test — cleanest falsification of Proposition 20.
"""
import sys, os
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

print("=" * 70)
print("STEP 784 — ENCODING ONLY (MINIMAL D-ONLY SUBSTRATE)")
print("=" * 70)

from substrates.step0784 import EncodingOnly784
from r3cf_runner import run_r3cf

result = run_r3cf(EncodingOnly784, "Step784_EncodingOnly", n_actions=4)

print()
print("STEP 784 DONE")
print(f"R3_cf: {'PASS' if result['r3_cf_pass'] else 'FAIL' if result['r3_cf_pass'] is False else 'INCONCLUSIVE'}")
print(f"cold={result['total_cold']}  warm={result['total_warm']}  OR={result['odds_ratio']}  p={result['p_val']}")
