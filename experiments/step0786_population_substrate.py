"""
step0786_population_substrate.py — R3_cf: Population Substrate (Price Equation).

R3 hypothesis: population-level selection produces R3 dynamics without individual
self-modification. N=10 substrates, select best every 1K steps.
D(s) = {W_population, running_means}. L(s) = ∅.

Kill: if all 10 converge to same parameters by round 5.
"""
import sys, os
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

print("=" * 70)
print("STEP 786 — POPULATION SUBSTRATE (PRICE EQUATION)")
print("=" * 70)

from substrates.step0786 import PopulationSubstrate786
from r3cf_runner import run_r3cf

result = run_r3cf(PopulationSubstrate786, "Step786_PopulationSubstrate", n_actions=4)

print()
print("STEP 786 DONE")
print(f"R3_cf: {'PASS' if result['r3_cf_pass'] else 'FAIL' if result['r3_cf_pass'] is False else 'INCONCLUSIVE'}")
print(f"cold={result['total_cold']}  warm={result['total_warm']}  OR={result['odds_ratio']}  p={result['p_val']}")
