"""
r3cf_runner.py — Shared R3_cf protocol for steps 778-887.

Protocol (Leo, mail 2556/2563/2564):
  Train 25K steps on LS20 seeds 1-5 (5K per seed sequentially, ONE substrate).
  Test on seeds 6-10 (10K steps each): cold vs warm.
  Warm > cold = R3_cf PASS.

TWO R3_cf metrics (mail 2564):
  1. Level completions (warm vs cold)
  2. Prediction accuracy: warm model vs cold model on observed transitions

Correction (mail 2565):
  Forward model input = encoded obs vector (256-dim), not hash integer.
  W ∈ R^{DIM × (DIM + n_actions)} where DIM=256.
"""
import sys, os, time
sys.path.insert(0, 'B:/M/the-search')

import numpy as np

PRETRAIN_SEEDS = list(range(1, 6))    # seeds 1-5
PRETRAIN_STEPS_PER_SEED = 5_000       # 25K total pretrain
TEST_SEEDS = list(range(6, 11))        # seeds 6-10
TEST_STEPS = 25_000                    # per test seed (matches step807 budget)
N_ACTIONS = 4                          # LS20 actions (0-3)
DIM = 256                              # encoding dimension


def _make_ls20():
    try:
        import arcagi3
        return arcagi3.make("LS20")
    except ImportError:
        import util_arcagi3
        return util_arcagi3.make("LS20")


def run_phase(env_seed: int, substrate, n_steps: int,
              collect_pred_errors: bool = False) -> tuple:
    """Run substrate on LS20 for n_steps.

    Returns (level_completions, pred_errors_list).
    pred_errors_list: list of (pred_error, obs_norm) tuples (empty if collect=False).
    """
    env = _make_ls20()
    obs = env.reset(seed=env_seed)
    current_level = 0
    level_completions = 0
    step = 0
    pred_errors = []
    prev_enc = None

    while step < n_steps:
        if obs is None:
            obs = env.reset(seed=env_seed)
            substrate.on_level_transition()
            prev_enc = None
            continue

        obs_arr = np.asarray(obs, dtype=np.float32)
        action = substrate.process(obs_arr)
        obs_next, reward, done, info = env.step(action % N_ACTIONS)
        step += 1

        # Prediction accuracy measurement
        if collect_pred_errors and hasattr(substrate, 'predict_next') and prev_enc is not None:
            if obs_next is not None:
                next_arr = np.asarray(obs_next, dtype=np.float32)
                if hasattr(substrate, '_encode_for_pred'):
                    next_enc = substrate._encode_for_pred(next_arr)
                    pred = substrate.predict_next(prev_enc, action % N_ACTIONS)
                    if pred is not None and next_enc is not None:
                        err = float(np.sum((pred - next_enc) ** 2))
                        norm = float(np.sum(next_enc ** 2)) + 1e-8
                        pred_errors.append((err, norm))
        # Update prev_enc unconditionally (outside the if block to avoid circular dependency)
        if collect_pred_errors and hasattr(substrate, '_last_enc'):
            prev_enc = substrate._last_enc

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
            prev_enc = None

    return level_completions, pred_errors


def pred_accuracy_from_errors(pred_errors: list) -> float:
    """Convert (error, norm) pairs to accuracy percentage."""
    if not pred_errors:
        return None
    total_err = sum(e for e, n in pred_errors)
    total_norm = sum(n for e, n in pred_errors)
    if total_norm < 1e-10:
        return None
    return float(1.0 - total_err / total_norm) * 100.0  # percent


def run_r3cf(SubstrateClass, step_name: str, n_actions: int = N_ACTIONS,
             substrate_seed: int = 0, verbose: bool = True,
             measure_prediction: bool = False) -> dict:
    """Run full R3_cf protocol. Returns result dict with pass/fail."""
    if verbose:
        print(f"\n{'='*70}")
        print(f"{step_name} — R3_cf PROTOCOL")
        print(f"Pretrain: seeds {PRETRAIN_SEEDS}, {PRETRAIN_STEPS_PER_SEED} steps each")
        print(f"Test: seeds {TEST_SEEDS}, {TEST_STEPS} steps each (cold vs warm)")
        if measure_prediction:
            print(f"Metrics: level completions + prediction accuracy")
        print(f"{'='*70}")

    t0 = time.time()

    # --- Phase 1: Pretrain ONE substrate on seeds 1-5 ---
    sub_pretrain = SubstrateClass(n_actions=n_actions, seed=substrate_seed)
    sub_pretrain.reset(substrate_seed)
    pre_completions = 0
    for ps in PRETRAIN_SEEDS:
        sub_pretrain.on_level_transition()
        c, _ = run_phase(ps * 1000, sub_pretrain, PRETRAIN_STEPS_PER_SEED)
        pre_completions += c

    saved_state = sub_pretrain.get_state()
    pretrain_elapsed = time.time() - t0
    if verbose:
        print(f"\nPretrain done in {pretrain_elapsed:.1f}s. Completions: {pre_completions}")

    # --- Phase 2+3: Cold vs Warm on test seeds ---
    cold_totals = []
    warm_totals = []
    cold_pred_accs = []
    warm_pred_accs = []

    for ts in TEST_SEEDS:
        env_seed = ts * 1000

        # Cold
        sub_cold = SubstrateClass(n_actions=n_actions, seed=substrate_seed)
        sub_cold.reset(substrate_seed)
        c_cold, errs_cold = run_phase(env_seed, sub_cold, TEST_STEPS, measure_prediction)
        cold_totals.append(c_cold)
        if errs_cold:
            cold_pred_accs.append(pred_accuracy_from_errors(errs_cold))

        # Warm
        sub_warm = SubstrateClass(n_actions=n_actions, seed=substrate_seed)
        sub_warm.reset(substrate_seed)
        sub_warm.set_state(saved_state)
        c_warm, errs_warm = run_phase(env_seed, sub_warm, TEST_STEPS, measure_prediction)
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

    # --- Results ---
    if verbose:
        print(f"\n{'='*70}")
        print(f"RESULTS — {step_name}")
        print(f"{'='*70}")
        print(f"  Total cold: {total_cold}  |  Total warm: {total_warm}")
        print(f"  Mean cold: {np.mean(cold_totals):.2f}  |  Mean warm: {np.mean(warm_totals):.2f}")
        n_warm_better = sum(1 for c, w in zip(cold_totals, warm_totals) if w > c)
        n_cold_better = sum(1 for c, w in zip(cold_totals, warm_totals) if c > w)
        n_tied = sum(1 for c, w in zip(cold_totals, warm_totals) if c == w)
        print(f"  warm>cold: {n_warm_better}/5  cold>warm: {n_cold_better}/5  tied: {n_tied}/5")

    # Metric 1: Fisher exact on level completions
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
            r3_cf_pass = None
            if verbose:
                print("  R3_cf (L1 metric): INCONCLUSIVE — zero level completions")
    except ImportError:
        pass

    # Metric 2: Prediction accuracy R3_cf
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

        if r3_cf_pass is True:
            print(f"\nFINDING (L1): D(s) TRANSFERS. Pretraining helps on unseen seeds.")
        elif r3_cf_pass is False:
            print(f"\nFINDING (L1): D(s) does NOT transfer. Warm <= cold.")
        else:
            print(f"\nFINDING (L1): INCONCLUSIVE — no level completions.")

        if pred_r3_cf_pass is True:
            print(f"FINDING (pred): Prediction improves after pretraining. Dynamic transfer confirmed.")
        elif pred_r3_cf_pass is False:
            print(f"FINDING (pred): Prediction does NOT improve after pretraining.")

    return {
        "step": step_name,
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
