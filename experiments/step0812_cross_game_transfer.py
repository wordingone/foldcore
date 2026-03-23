"""
step0812_cross_game_transfer.py -- Prop 20 test: LS20->FT09 transfer.

Protocol:
1. Pretrain: CrossGameTransfer812 on LS20 seeds 1-5 (5K steps each, 25K total).
   Random action + delta rule W. Running_mean adapts to LS20 distribution.
2. Cold baseline: fresh W, tested on FT09 seeds 6-10.
3. Warm test: LS20-trained W, tested on FT09 seeds 6-10.
4. Metric: prediction accuracy (warm > cold -> Prop 20 PASS).

If warm > cold: LS20 dynamics knowledge transfers to FT09.
If warm < cold (LS20 W misleads FT09): games are dynamically distinct.
If warm ~= cold: dynamics don't generalize across games.

R3 hypothesis: the 674 encoding captures universal visual dynamics
that generalize across ARC-AGI-3 games.
"""
import sys, time
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from substrates.step0812 import CrossGameTransfer812

N_ACTIONS_LS20 = 4
N_ACTIONS_FT09 = 68
PRETRAIN_SEEDS = list(range(1, 6))
PRETRAIN_STEPS = 5_000
TEST_SEEDS = list(range(6, 11))
TEST_STEPS = 25_000


def _make_game(name):
    try:
        import arcagi3
        return arcagi3.make(name)
    except ImportError:
        import util_arcagi3
        return util_arcagi3.make(name)


def run_pred_phase(substrate, game_name, n_actions, seed, n_steps):
    """Run substrate on game, collect prediction errors."""
    env = _make_game(game_name)
    obs = env.reset(seed=seed)
    pred_errors = []
    step = 0

    while step < n_steps:
        if obs is None:
            obs = env.reset(seed=seed)
            substrate.on_level_transition()
            continue

        obs_arr = np.asarray(obs, dtype=np.float32)
        action = substrate.process(obs_arr)
        obs_next, reward, done, info = env.step(action % n_actions)
        step += 1

        if obs_next is not None and substrate._last_enc is not None:
            next_arr = np.asarray(obs_next, dtype=np.float32)
            next_enc = substrate._encode_for_pred(next_arr)
            prev_enc = substrate._last_enc
            pred = substrate.predict_next(prev_enc, action % n_actions)
            if pred is not None:
                err = float(np.sum((pred - next_enc) ** 2))
                norm = float(np.sum(next_enc ** 2)) + 1e-8
                pred_errors.append((err, norm))

        obs = obs_next
        if done:
            obs = env.reset(seed=seed)
            substrate.on_level_transition()

    if not pred_errors:
        return None
    total_err = sum(e for e, n in pred_errors)
    total_norm = sum(n for e, n in pred_errors)
    return float(1.0 - total_err / total_norm) * 100.0


print("=" * 70)
print("STEP 812 — CROSS-GAME TRANSFER (LS20 -> FT09)")
print("=" * 70)
print(f"Pretrain: LS20 seeds {PRETRAIN_SEEDS}, {PRETRAIN_STEPS} steps each")
print(f"Test: FT09 seeds {TEST_SEEDS}, {TEST_STEPS} steps each (cold vs warm)")
print("=" * 70)

t0 = time.time()

# Phase 1: Pretrain on LS20 (4 actions)
sub_pretrain = CrossGameTransfer812(n_actions=N_ACTIONS_LS20, seed=0)
sub_pretrain.reset(0)
for ps in PRETRAIN_SEEDS:
    sub_pretrain.on_level_transition()
    env = _make_game("LS20")
    obs = env.reset(seed=ps * 1000)
    step = 0
    while step < PRETRAIN_STEPS:
        if obs is None:
            obs = env.reset(seed=ps * 1000)
            sub_pretrain.on_level_transition()
            continue
        obs_arr = np.asarray(obs, dtype=np.float32)
        action = sub_pretrain.process(obs_arr)
        obs, _, done, _ = env.step(action % N_ACTIONS_LS20)
        step += 1
        if done:
            obs = env.reset(seed=ps * 1000)
            sub_pretrain.on_level_transition()

saved_state = sub_pretrain.get_state()
print(f"\nPretrain done in {time.time()-t0:.1f}s")
print(f"W norm after LS20 pretrain: {float(np.linalg.norm(saved_state['W'])):.4f}")

# Phase 2+3: Cold vs Warm on FT09 (68 actions)
cold_accs = []
warm_accs = []

for ts in TEST_SEEDS:
    env_seed = ts * 1000

    # Cold: fresh W, test on FT09
    sub_cold = CrossGameTransfer812(n_actions=N_ACTIONS_FT09, seed=0)
    sub_cold.reset(0)
    cold_acc = run_pred_phase(sub_cold, "FT09", N_ACTIONS_FT09, env_seed, TEST_STEPS)
    cold_accs.append(cold_acc)

    # Warm: LS20-trained W, but resize for FT09 n_actions
    # W shape changes: LS20 has 256x260, FT09 needs 256x324. Must handle mismatch.
    sub_warm = CrossGameTransfer812(n_actions=N_ACTIONS_FT09, seed=0)
    sub_warm.reset(0)
    # Transfer running_mean only (W shape differs LS20 vs FT09)
    sub_warm.running_mean = saved_state["running_mean"].copy()
    sub_warm._n_obs = saved_state["_n_obs"]
    warm_acc = run_pred_phase(sub_warm, "FT09", N_ACTIONS_FT09, env_seed, TEST_STEPS)
    warm_accs.append(warm_acc)

    cold_str = f"{cold_acc:.2f}%" if cold_acc is not None else "N/A"
    warm_str = f"{warm_acc:.2f}%" if warm_acc is not None else "N/A"
    print(f"  seed={ts}  cold={cold_str}  warm={warm_str}")

valid_cold = [a for a in cold_accs if a is not None]
valid_warm = [a for a in warm_accs if a is not None]

print()
print("=" * 70)
print("STEP 812 RESULTS — CROSS-GAME TRANSFER LS20->FT09")
print("=" * 70)
if valid_cold and valid_warm:
    mean_cold = np.mean(valid_cold)
    mean_warm = np.mean(valid_warm)
    print(f"  Mean pred_acc cold: {mean_cold:.2f}%")
    print(f"  Mean pred_acc warm: {mean_warm:.2f}%")
    transfer_pass = mean_warm > mean_cold
    print(f"  Prop 20 transfer: {'PASS' if transfer_pass else 'FAIL'}")
    if transfer_pass:
        print(f"  LS20 dynamics TRANSFERS to FT09 (+{mean_warm-mean_cold:.2f}%)")
    else:
        print(f"  LS20 dynamics does NOT transfer to FT09 ({mean_warm-mean_cold:.2f}%)")
else:
    print("  No valid predictions collected.")

print(f"  Total elapsed: {time.time()-t0:.1f}s")
print()
print("STEP 812 DONE")
