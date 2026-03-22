"""
Step 714c — VC33 tol sweep diagnostic.

R3 hypothesis: if higher tol prunes VC33 actions, the substrate can self-calibrate
its change-detection threshold from observation statistics (e.g., use median frame
delta as threshold). That's weak R3 — self-derived threshold.

Method: 1 seed, VC33, 500 warmup steps, 4 tol values [0.5, 2.0, 5.0, 10.0].
Report dead count at each tol. If dead rises with tol, VC33 has cosmetic changes
below threshold and calibration is viable. If dead stays 0, something else is wrong.
"""
import numpy as np
import sys

WARMUP_STEPS = 500
DEAD_THRESHOLD = 5
SEED = 0

TOL_VALUES = [0.5, 2.0, 5.0, 10.0]

DIR_ACTIONS = [0, 1, 2, 3]
GRID_ACTIONS = [(gx * 8 + 4) + (gy * 8 + 4) * 64
                for gy in range(8) for gx in range(8)]
UNIVERSAL_ACTIONS = DIR_ACTIONS + GRID_ACTIONS
N_UNIV = len(UNIVERSAL_ACTIONS)  # 68


def obs_changed(frame_before, frame_after, tol):
    try:
        a = np.array(frame_before[0], dtype=np.float32)
        b = np.array(frame_after[0], dtype=np.float32)
        return not np.allclose(a, b, atol=tol)
    except Exception:
        return True


def run_tol(make, tol):
    env = make()
    obs = env.reset(seed=SEED)

    probe_count = [0] * N_UNIV
    change_count = [0] * N_UNIV
    live_actions = set(range(N_UNIV))
    dead_actions = set()
    probe_ptr = 0

    # Track per-action deltas for calibration insight
    action_deltas = {i: [] for i in range(N_UNIV)}

    for step in range(1, WARMUP_STEPS + 1):
        if obs is None:
            obs = env.reset(seed=SEED)
            continue

        live_sorted = sorted(live_actions) or [4]
        idx = live_sorted[probe_ptr % len(live_sorted)]
        probe_ptr += 1
        action_int = UNIVERSAL_ACTIONS[idx]

        try:
            obs_new, reward, done, info = env.step(action_int)
            if obs_new is not None:
                a = np.array(obs[0], dtype=np.float32)
                b = np.array(obs_new[0], dtype=np.float32)
                delta = float(np.max(np.abs(a - b)))
                action_deltas[idx].append(delta)
                changed = delta > tol
            else:
                changed = False
        except Exception:
            obs_new = obs
            done = False
            changed = False

        probe_count[idx] += 1
        if changed:
            change_count[idx] += 1
        if (probe_count[idx] >= DEAD_THRESHOLD
                and change_count[idx] == 0):
            live_actions.discard(idx)
            dead_actions.add(idx)

        obs = obs_new if obs_new is not None else obs
        if done:
            obs = env.reset(seed=SEED)

    dead_dir = len([a for a in dead_actions if a < 4])
    dead_click = len([a for a in dead_actions if a >= 4])

    # Median max-delta across all probed actions
    all_deltas = []
    for deltas in action_deltas.values():
        if deltas:
            all_deltas.append(np.mean(deltas))
    median_delta = float(np.median(all_deltas)) if all_deltas else 0.0

    return dict(tol=tol, dead=len(dead_actions), dead_dir=dead_dir,
                dead_click=dead_click, median_delta=median_delta)


def main():
    try:
        sys.path.insert(0, '.')
        import arcagi3
        mk = lambda: arcagi3.make("VC33")
    except Exception as e:
        print(f"arcagi3: {e}"); return

    print(f"Step 714c: VC33 tol sweep, 1 seed, {WARMUP_STEPS} warmup steps")
    print(f"Tol values: {TOL_VALUES}")
    print(f"Dead threshold: {DEAD_THRESHOLD} probes with 0 changes\n")

    results = []
    for tol in TOL_VALUES:
        r = run_tol(mk, tol)
        results.append(r)
        print(f"  tol={tol:5.1f}: dead={r['dead']}(dir={r['dead_dir']},click={r['dead_click']}) "
              f"median_max_delta={r['median_delta']:.3f}", flush=True)

    print(f"\n{'='*60}")
    print(f"Tol sweep summary:")
    for r in results:
        print(f"  tol={r['tol']:5.1f} -> dead={r['dead']:2d}  median_delta={r['median_delta']:.3f}")
    deltas_rising = results[-1]['dead'] > results[0]['dead']
    if deltas_rising:
        best = max(results, key=lambda r: r['dead'])
        print(f"FINDING: dead count rises with tol. Best: tol={best['tol']} -> dead={best['dead']}")
        print(f"IMPLICATION: self-calibrating threshold is viable (weak R3)")
    else:
        print(f"FINDING: dead=0 at all tol values — VC33 genuinely changes for all actions")
        print(f"IMPLICATION: tol calibration won't help; need outcome-based pruning")


if __name__ == "__main__":
    main()
