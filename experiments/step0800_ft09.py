"""
step0800_ft09.py -- PerActionChangePursuit800 on FT09.

R3 hypothesis: EMA per-action change tracking identifies productive clicks.
FT09 has 8/68 productive clicks. argmax(delta[a]) should discover them faster
than random or prediction-contrast.

FIX: use substrate_seed=ts for each test seed (varied internal RNG).
This prevents the FT09 degenerate case (same substrate_seed=0 for all seeds
→ n_effective=1). Leo directive: substrate_seed=env_seed.

FT09 floor: L1=0 (random 68 actions, 25K steps, step807_ft09).
R3_cf: pretrain seeds 1-5, test seeds 6-10.
"""
import sys, time
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from substrates.step0800 import PerActionChangePursuit800

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
    """Run substrate for n_steps. Returns level_completions."""
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
print("STEP 800 FT09 — PER-ACTION CHANGE PURSUIT (68 actions)")
print("=" * 70)
print("FIX: substrate_seed=ts (varied per test seed)")
print(f"Pretrain: FT09 seeds {PRETRAIN_SEEDS}, {PRETRAIN_STEPS} steps each")
print(f"Test: FT09 seeds {TEST_SEEDS}, {TEST_STEPS} steps each (cold vs warm)")
print("=" * 70)

t0 = time.time()

# Phase 1: Pretrain on FT09 seeds 1-5 (substrate_seed=0 for pretrain)
sub_pretrain = PerActionChangePursuit800(n_actions=N_ACTIONS, seed=0)
sub_pretrain.reset(0)
pre_completions = 0
for ps in PRETRAIN_SEEDS:
    sub_pretrain.on_level_transition()
    c = run_phase(sub_pretrain, ps * 1000, PRETRAIN_STEPS)
    pre_completions += c

saved_state = sub_pretrain.get_state()
print(f"\nPretrain done in {time.time()-t0:.1f}s. Completions: {pre_completions}")
print(f"  delta_per_action after pretrain: {saved_state['delta_per_action']}")

# Phase 2+3: Cold vs Warm (substrate_seed=ts for varied RNG)
cold_totals = []
warm_totals = []

for ts in TEST_SEEDS:
    env_seed = ts * 1000

    # Cold: fresh substrate, seed=ts (varied RNG)
    sub_cold = PerActionChangePursuit800(n_actions=N_ACTIONS, seed=ts)
    sub_cold.reset(ts)
    c_cold = run_phase(sub_cold, env_seed, TEST_STEPS)
    cold_totals.append(c_cold)

    # Warm: load pretrain state, seed=ts (varied RNG)
    sub_warm = PerActionChangePursuit800(n_actions=N_ACTIONS, seed=ts)
    sub_warm.reset(ts)
    sub_warm.set_state(saved_state)
    c_warm = run_phase(sub_warm, env_seed, TEST_STEPS)
    warm_totals.append(c_warm)

    print(f"  seed={ts}  cold={c_cold}  warm={c_warm}  delta={c_warm-c_cold:+d}")

total_cold = sum(cold_totals)
total_warm = sum(warm_totals)
total_elapsed = time.time() - t0

print()
print("=" * 70)
print("STEP 800 FT09 RESULTS")
print("=" * 70)
print(f"  Total cold: {total_cold}  |  Total warm: {total_warm}")
print(f"  Mean cold: {np.mean(cold_totals):.2f}  |  Mean warm: {np.mean(warm_totals):.2f}")
print(f"  FT09 floor: 0 (step807_ft09 random baseline)")

try:
    from scipy.stats import fisher_exact
    if total_warm > 0 or total_cold > 0:
        total_steps = len(TEST_SEEDS) * TEST_STEPS
        warm_no = total_steps - total_warm
        cold_no = total_steps - total_cold
        or_, p = fisher_exact([[total_warm, warm_no], [total_cold, cold_no]])
        r3_pass = (p < 0.05 and total_warm > total_cold)
        print(f"  Fisher exact: OR={or_:.3f}  p={p:.4f}")
        print(f"  R3_cf (L1): {'PASS' if r3_pass else 'FAIL'}")
    else:
        print(f"  R3_cf (L1): INCONCLUSIVE — zero completions")
except ImportError:
    pass

print(f"  Total elapsed: {total_elapsed:.1f}s")
print()
print("STEP 800 FT09 DONE")
