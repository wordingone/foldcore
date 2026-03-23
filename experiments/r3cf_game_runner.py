"""
r3cf_game_runner.py — Game-agnostic R3_cf protocol.

Same as r3cf_runner but accepts game_name parameter.
Supports LS20 (4 actions), FT09 (68 actions), VC33 (68 actions).
"""
import sys, os, time
sys.path.insert(0, 'B:/M/the-search')

import numpy as np

PRETRAIN_SEEDS = list(range(1, 6))    # seeds 1-5
PRETRAIN_STEPS_PER_SEED = 5_000       # 25K total pretrain
TEST_SEEDS = list(range(6, 11))       # seeds 6-10
TEST_STEPS = 25_000                   # per test seed
DIM = 256


def _make_game(game_name: str):
    try:
        import arcagi3
        return arcagi3.make(game_name)
    except ImportError:
        import util_arcagi3
        return util_arcagi3.make(game_name)


def run_phase_game(env_seed: int, substrate, n_steps: int, n_actions: int,
                   game_name: str, collect_pred_errors: bool = False) -> tuple:
    """Run substrate on game for n_steps. Returns (level_completions, pred_errors_list)."""
    env = _make_game(game_name)
    obs = env.reset(seed=env_seed)
    current_level = 0
    level_completions = 0
    step = 0
    pred_errors = []

    while step < n_steps:
        if obs is None:
            obs = env.reset(seed=env_seed)
            substrate.on_level_transition()
            continue

        obs_arr = np.asarray(obs, dtype=np.float32)
        action = substrate.process(obs_arr)
        obs_next, reward, done, info = env.step(action % n_actions)
        step += 1

        # Prediction accuracy measurement
        if collect_pred_errors and hasattr(substrate, 'predict_next') and hasattr(substrate, '_last_enc'):
            if obs_next is not None and substrate._last_enc is not None:
                next_arr = np.asarray(obs_next, dtype=np.float32)
                if hasattr(substrate, '_encode_for_pred'):
                    next_enc = substrate._encode_for_pred(next_arr)
                    prev_enc = substrate._last_enc
                    pred = substrate.predict_next(prev_enc, action % n_actions)
                    if pred is not None and next_enc is not None:
                        err = float(np.sum((pred - next_enc) ** 2))
                        norm = float(np.sum(next_enc ** 2)) + 1e-8
                        pred_errors.append((err, norm))

        obs = obs_next

        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > current_level:
            level_completions += (cl - current_level)
            current_level = cl
            substrate.on_level_transition()

        if done:
            obs = env.reset(seed=env_seed)
            current_level = 0
            substrate.on_level_transition()

    return level_completions, pred_errors


def pred_accuracy_from_errors(pred_errors: list) -> float:
    if not pred_errors:
        return None
    total_err = sum(e for e, n in pred_errors)
    total_norm = sum(n for e, n in pred_errors)
    if total_norm < 1e-10:
        return None
    return float(1.0 - total_err / total_norm) * 100.0


def run_r3cf_game(SubstrateClass, step_name: str, game_name: str = "LS20",
                  n_actions: int = 4, substrate_seed: int = 0,
                  verbose: bool = True, measure_prediction: bool = False) -> dict:
    """Run full R3_cf protocol on specified game. Returns result dict."""
    if verbose:
        print(f"\n{'='*70}")
        print(f"{step_name} — R3_cf PROTOCOL on {game_name}")
        print(f"Pretrain: seeds {PRETRAIN_SEEDS}, {PRETRAIN_STEPS_PER_SEED} steps each")
        print(f"Test: seeds {TEST_SEEDS}, {TEST_STEPS} steps each (cold vs warm)")
        if measure_prediction:
            print(f"Metrics: level completions + prediction accuracy")
        print(f"{'='*70}")

    t0 = time.time()

    # Phase 1: Pretrain
    sub_pretrain = SubstrateClass(n_actions=n_actions, seed=substrate_seed)
    sub_pretrain.reset(substrate_seed)
    pre_completions = 0
    for ps in PRETRAIN_SEEDS:
        sub_pretrain.on_level_transition()
        c, _ = run_phase_game(ps * 1000, sub_pretrain, PRETRAIN_STEPS_PER_SEED,
                               n_actions, game_name)
        pre_completions += c

    saved_state = sub_pretrain.get_state()
    pretrain_elapsed = time.time() - t0
    if verbose:
        print(f"\nPretrain done in {pretrain_elapsed:.1f}s. Completions: {pre_completions}")

    # Phase 2+3: Cold vs Warm
    cold_totals = []
    warm_totals = []
    cold_pred_accs = []
    warm_pred_accs = []

    for ts in TEST_SEEDS:
        env_seed = ts * 1000

        sub_cold = SubstrateClass(n_actions=n_actions, seed=substrate_seed)
        sub_cold.reset(substrate_seed)
        c_cold, errs_cold = run_phase_game(env_seed, sub_cold, TEST_STEPS,
                                           n_actions, game_name, measure_prediction)
        cold_totals.append(c_cold)
        if errs_cold:
            cold_pred_accs.append(pred_accuracy_from_errors(errs_cold))

        sub_warm = SubstrateClass(n_actions=n_actions, seed=substrate_seed)
        sub_warm.reset(substrate_seed)
        sub_warm.set_state(saved_state)
        c_warm, errs_warm = run_phase_game(env_seed, sub_warm, TEST_STEPS,
                                           n_actions, game_name, measure_prediction)
        warm_totals.append(c_warm)
        if errs_warm:
            warm_pred_accs.append(pred_accuracy_from_errors(errs_warm))

        if verbose:
            pred_str = ""
            if cold_pred_accs and warm_pred_accs:
                pred_str = f"  pred_acc cold={cold_pred_accs[-1]:.1f}% warm={warm_pred_accs[-1]:.1f}%"
            print(f"  seed={ts}  cold={c_cold}  warm={c_warm}  delta={c_warm-c_cold:+d}{pred_str}")

    total_cold = sum(cold_totals)
    total_warm = sum(warm_totals)
    total_elapsed = time.time() - t0

    if verbose:
        print(f"\n{'='*70}")
        print(f"RESULTS — {step_name} on {game_name}")
        print(f"{'='*70}")
        print(f"  Total cold: {total_cold}  |  Total warm: {total_warm}")
        print(f"  Mean cold: {np.mean(cold_totals):.2f}  |  Mean warm: {np.mean(warm_totals):.2f}")

    r3_cf_pass = None
    p_val = None
    odds_ratio = None
    total_steps = len(TEST_SEEDS) * TEST_STEPS

    try:
        from scipy.stats import fisher_exact
        if total_warm > 0 or total_cold > 0:
            warm_no = total_steps - total_warm
            cold_no = total_steps - total_cold
            odds_ratio, p_val = fisher_exact([[total_warm, warm_no], [total_cold, cold_no]])
            r3_cf_pass = (p_val < 0.05 and total_warm > total_cold)
            if verbose:
                print(f"  Fisher exact (L1): OR={odds_ratio:.3f}  p={p_val:.4f}")
                print(f"  R3_cf (L1 metric): {'PASS' if r3_cf_pass else 'FAIL'}")
        else:
            if verbose:
                print("  R3_cf (L1 metric): INCONCLUSIVE — zero level completions")
    except ImportError:
        pass

    pred_r3_cf_pass = None
    mean_cold_acc = None
    mean_warm_acc = None
    if cold_pred_accs and warm_pred_accs:
        mean_cold_acc = float(np.mean(cold_pred_accs))
        mean_warm_acc = float(np.mean(warm_pred_accs))
        pred_r3_cf_pass = (mean_warm_acc > mean_cold_acc)
        if verbose:
            print(f"  Prediction acc cold: {mean_cold_acc:.2f}%  warm: {mean_warm_acc:.2f}%")
            print(f"  R3_cf (pred metric): {'PASS' if pred_r3_cf_pass else 'FAIL'}")

    if verbose:
        print(f"  Total elapsed: {total_elapsed:.1f}s")

    return {
        "step": step_name,
        "game": game_name,
        "cold_totals": cold_totals,
        "warm_totals": warm_totals,
        "total_cold": total_cold,
        "total_warm": total_warm,
        "pretrain_completions": pre_completions,
        "r3_cf_pass": r3_cf_pass,
        "p_val": p_val,
        "odds_ratio": odds_ratio,
        "pred_r3_cf_pass": pred_r3_cf_pass,
        "mean_cold_acc": mean_cold_acc,
        "mean_warm_acc": mean_warm_acc,
        "elapsed": total_elapsed,
    }
