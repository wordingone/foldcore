"""
step0775_r3_calibration_table1.py — R3 Calibration Baseline Audit (Table 1).

R3 hypothesis: The constitutional judge correctly classifies 4 known system types.
The calibration table establishes what R3 scores mean empirically.

Expected verdicts (pre-registered):
  RandomAgent:      R1=PASS, R2=FAIL (state never changes), R3=FAIL (all U), R3_cf=SKIP
  FixedPolicyAgent: R1=PASS, R2=FAIL (state never changes), R3=FAIL (all U), R3_cf=SKIP
  TabularQLearning: R1=FAIL (uses reward), R3=M elements, R3_cf=PASS
  674 bootloader:   R1=PASS, R2=PASS, R3=FAIL (6 U→now 5 U, K_NAV reclassified to I)

Fixes verified:
  Fix 1 (step budgets) — in judge.py: n_steps budget, not time budget
  Fix 4 (calibration baselines) — this script
  Fix 5 (set_state) — base.py abstract method, all 4 agents implement it

Table 1: System × {R1, R2, R3_static, R3_counterfactual, R3_dynamic, verdict}
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from substrates.judge import ConstitutionalJudge
from substrates.calibration_agents import RandomAgent, FixedPolicyAgent, TabularQLearning
from substrates.step0674 import TransitionTriggered674

print("=" * 70)
print("STEP 775 — R3 CALIBRATION BASELINE AUDIT (TABLE 1)")
print("=" * 70)
print()

judge = ConstitutionalJudge()

agents = [
    ("RandomAgent",         RandomAgent),
    ("FixedPolicyAgent",    FixedPolicyAgent),
    ("TabularQLearning",    TabularQLearning),
    ("TransitionTriggered674", TransitionTriggered674),
]

results = {}
for name, cls in agents:
    print(f"Auditing {name}...")
    t0 = time.time()
    try:
        r = judge.audit(cls)
        elapsed = time.time() - t0
        results[name] = (r, elapsed, None)
        print(f"  Done in {elapsed:.1f}s")
    except Exception as e:
        elapsed = time.time() - t0
        results[name] = (None, elapsed, str(e))
        print(f"  ERROR: {e}")

print()
print("=" * 70)
print("TABLE 1: R3 CALIBRATION BASELINES")
print("=" * 70)
print()

# Header
header = f"{'System':<25} {'R1':<6} {'R2':<6} {'R3':<6} {'R3_cf':<8} {'U':<4} {'M':<4} {'Verdict'}"
print(header)
print("-" * 70)

for name, (r, elapsed, err) in results.items():
    if err is not None:
        print(f"{name:<25} ERROR: {err}")
        continue

    # R1
    r1 = r.get('R1', {})
    r1_str = 'PASS' if r1.get('pass') else 'FAIL'

    # R2
    r2 = r.get('R2', {})
    r2_str = 'PASS' if r2.get('pass') else 'FAIL'

    # R3 static (U count)
    r3 = r.get('R3', {})
    r3_str = 'PASS' if r3.get('pass') else 'FAIL'

    # R3 counterfactual
    r3cf = r.get('R3_counterfactual', {})
    if r3cf.get('pass') is None:
        r3cf_str = 'SKIP'
    elif r3cf.get('pass'):
        r3cf_str = 'PASS'
    else:
        r3cf_str = 'FAIL'

    # U/M counts from frozen_elements
    fe = r.get('frozen_elements', [])
    u_count = sum(1 for e in fe if e.get('class') == 'U')
    m_count = sum(1 for e in fe if e.get('class') == 'M')

    # Summary verdict
    verdict = r.get('summary', {}).get('verdict', '?')

    print(f"{name:<25} {r1_str:<6} {r2_str:<6} {r3_str:<6} {r3cf_str:<8} {u_count:<4} {m_count:<4} {verdict}")

print()
print("=" * 70)
print("DETAILED RESULTS")
print("=" * 70)
for name, (r, elapsed, err) in results.items():
    if err is not None:
        continue
    print(f"\n--- {name} ({elapsed:.1f}s) ---")
    for check in ['R1', 'R2', 'R3', 'R3_counterfactual']:
        cr = r.get(check, {})
        passed = cr.get('pass')
        detail = cr.get('detail', '')[:80]
        if check == 'R3_counterfactual':
            p_cold = cr.get('P_cold', '?')
            p_warm = cr.get('P_warm', '?')
            print(f"  {check}: pass={passed} P_cold={p_cold} P_warm={p_warm} | {detail}")
        else:
            print(f"  {check}: pass={passed} | {detail}")
    summary = r.get('summary', {})
    print(f"  Summary: verdict={summary.get('verdict')} score={summary.get('score')}")

print()
print("=" * 70)
print("PRE-REGISTERED EXPECTATIONS CHECK")
print("=" * 70)
checks = {
    "RandomAgent":      {"R1": True,  "R2": False, "R3": False},
    "FixedPolicyAgent": {"R1": True,  "R2": False, "R3": False},
    "TabularQLearning": {"R1": False, "R3": False},  # R1 FAIL, R3 has M but U too
    "TransitionTriggered674": {"R1": True, "R2": True, "R3": False},
}

all_match = True
for name, expected in checks.items():
    r, elapsed, err = results.get(name, (None, 0, "not run"))
    if err or r is None:
        print(f"  {name}: SKIP (error)")
        continue
    for rule, exp_pass in expected.items():
        actual = r.get(rule, {}).get('pass')
        match = actual == exp_pass
        if not match:
            all_match = False
        status = "OK" if match else f"MISMATCH (expected={exp_pass}, got={actual})"
        print(f"  {name} {rule}: {status}")

print()
if all_match:
    print("ALL EXPECTATIONS MET — Table 1 calibration valid.")
else:
    print("SOME EXPECTATIONS NOT MET — review judge logic.")
print()
print("STEP 775 DONE")
