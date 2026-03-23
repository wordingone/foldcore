"""
step0800b_ft09.py -- EpsilonActionChange800b on FT09 (varied substrate seeds).

R3 hypothesis: per-action EMA change tracking with 68 actions. Does the 80%
argmax(delta) mechanism discover productive clicks on FT09?

FT09 prediction: L1=0 (delta uniform — productive clicks don't produce more
change than non-productive when position-mismatched). Confirms step800 finding.

FIX: substrate_seed=ts for varied internal RNG (Leo directive 2026-03-23).
FT09 floor: L1=0 (step807_ft09, random 68 actions, 25K steps).
"""
import sys, time
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from substrates.step0800b import EpsilonActionChange800b

N_ACTIONS = 68
PRETRAIN_SEEDS = list(range(1, 6))
PRETRAIN_STEPS = 5_000
TEST_SEEDS = list(range(6, 11))
TEST_STEPS = 25_000


def _make_ft09():
    try:
        import arcagi3
        return arcagi3.make("FT09")
    except ImportError:
        import util_arcagi3
        return util_arcagi3.make("FT09")


def run_phase(substrate, env_seed, n_steps):
    env = _make_ft09()
    obs = env.reset(seed=env_seed)
    level_completions = 0
    current_level = 0
    step = 0
    while step < n_steps:
        if obs is None:
            obs = env.reset(seed=env_seed)
            substrate.on_level_transition()
            continue
        obs_arr = np.asarray(obs, dtype=np.float32)
        action = substrate.process(obs_arr) % N_ACTIONS
        obs, reward, done, info = env.step(action)
        step += 1
        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > current_level:
            level_completions += (cl - current_level)
            current_level = cl
            substrate.on_level_transition()
        if done:
            obs = env.reset(seed=env_seed)
            current_level = 0
            substrate.on_level_transition()
    return level_completions


print("=" * 70)
print("STEP 800b FT09 — EPSILON ACTION CHANGE (68 actions)")
print("=" * 70)
print("FIX: substrate_seed=ts (varied per test seed)")
print(f"Pretrain: FT09 seeds {PRETRAIN_SEEDS}, {PRETRAIN_STEPS} steps each")
print(f"Test: FT09 seeds {TEST_SEEDS}, {TEST_STEPS} steps each (cold vs warm)")
print("FT09 floor: 0 (step807_ft09)")
print("=" * 70)

t0 = time.time()

# Pretrain on FT09 seeds 1-5 (substrate_seed=0)
sub_pretrain = EpsilonActionChange800b(n_actions=N_ACTIONS, seed=0)
sub_pretrain.reset(0)
pre_completions = 0
for ps in PRETRAIN_SEEDS:
    sub_pretrain.on_level_transition()
    c = run_phase(sub_pretrain, ps * 1000, PRETRAIN_STEPS)
    pre_completions += c
saved_state = sub_pretrain.get_state()
print(f"\nPretrain done in {time.time()-t0:.1f}s. Completions: {pre_completions}")
print(f"  delta_per_action (first 8): {saved_state['delta_per_action'][:8]}")

cold_totals = []
warm_totals = []
for ts in TEST_SEEDS:
    env_seed = ts * 1000
    sub_cold = EpsilonActionChange800b(n_actions=N_ACTIONS, seed=ts)
    sub_cold.reset(ts)
    c_cold = run_phase(sub_cold, env_seed, TEST_STEPS)
    cold_totals.append(c_cold)

    sub_warm = EpsilonActionChange800b(n_actions=N_ACTIONS, seed=ts)
    sub_warm.reset(ts)
    sub_warm.set_state(saved_state)
    c_warm = run_phase(sub_warm, env_seed, TEST_STEPS)
    warm_totals.append(c_warm)
    print(f"  seed={ts}  cold={c_cold}  warm={c_warm}  delta={c_warm-c_cold:+d}")

total_cold = sum(cold_totals)
total_warm = sum(warm_totals)
print()
print("=" * 70)
print("STEP 800b FT09 RESULTS")
print("=" * 70)
print(f"  Total cold: {total_cold}  |  Total warm: {total_warm}")
print(f"  Mean cold: {np.mean(cold_totals):.2f}  |  Mean warm: {np.mean(warm_totals):.2f}")
print(f"  FT09 floor: 0 (random baseline)")
if total_cold + total_warm == 0:
    print(f"  R3_cf (L1): INCONCLUSIVE — zero completions")
else:
    try:
        from scipy.stats import fisher_exact
        total_steps = len(TEST_SEEDS) * TEST_STEPS
        or_, p = fisher_exact([[total_warm, total_steps-total_warm], [total_cold, total_steps-total_cold]])
        r3_pass = (p < 0.05 and total_warm > total_cold)
        print(f"  Fisher: OR={or_:.3f}  p={p:.4f}")
        print(f"  R3_cf (L1): {'PASS' if r3_pass else 'FAIL'}")
    except ImportError:
        pass
print(f"  Total elapsed: {time.time()-t0:.1f}s")
print()
print("STEP 800b FT09 DONE")
