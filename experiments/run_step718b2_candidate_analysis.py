"""
Step 718b2 — Enhanced analysis of candidate.c on LS20.

Questions:
1. Action distribution: uniform or structured?
2. Action autocorrelation: does action[t] predict action[t+1]?
3. Episode length distribution vs random baseline (~150 steps)
4. Any L1 events

Kill criterion: uniform distribution + no autocorrelation + episode lengths match random
→ candidate.c = random number generator in this context.

DO NOT MODIFY candidate.c. DO NOT add 674 or any infrastructure.
"""
import subprocess
import numpy as np
import sys
import time

EXE = "B:/M/the-search/substrates/candidate/candidate.exe"

DIR_ACTIONS = [0, 1, 2, 3]
GRID_ACTIONS = [(gx * 8 + 4) + (gy * 8 + 4) * 64
                for gy in range(8) for gx in range(8)]
UNIVERSAL_ACTIONS = DIR_ACTIONS + GRID_ACTIONS
N_UNIV = len(UNIVERSAL_ACTIONS)

N_SEEDS = 5
N_GAME_STEPS = 1000
N_CA_PER_STEP = 4096
N_OUT_PER_STEP = N_CA_PER_STEP // 256  # 16


def run_seed(seed, make):
    env = make()
    obs = env.reset(seed=seed)
    level = 0
    l1 = None
    t_start = time.time()

    action_hist = np.zeros(N_UNIV, dtype=np.int64)
    action_seq = []     # full sequence of action indices
    ep_lengths = []
    ep_step = 0

    n_total_ca = N_GAME_STEPS * N_CA_PER_STEP

    proc = subprocess.Popen(
        [EXE, str(n_total_ca)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL
    )

    action_idx = 0

    for step in range(1, N_GAME_STEPS + 1):
        if obs is None:
            obs = env.reset(seed=seed)
            ep_lengths.append(ep_step)
            ep_step = 0

        frame = np.array(obs[0], dtype=np.uint8).flatten()

        try:
            proc.stdin.write(bytes(frame))
            proc.stdin.flush()
            chunk = proc.stdout.read(N_OUT_PER_STEP)
            if chunk and len(chunk) >= 1:
                action_idx = chunk[-1] % N_UNIV
        except (BrokenPipeError, OSError):
            break

        action_hist[action_idx] += 1
        action_seq.append(action_idx)

        action_int = UNIVERSAL_ACTIONS[action_idx]

        try:
            obs_new, reward, done, info = env.step(action_int)
        except Exception:
            obs_new = obs; done = False; info = {}

        obs = obs_new
        ep_step += 1

        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > level:
            if cl == 1 and l1 is None:
                l1 = step
            level = cl
            ep_lengths.append(ep_step)
            ep_step = 0

        if done:
            ep_lengths.append(ep_step)
            ep_step = 0
            obs = env.reset(seed=seed)

    if ep_step > 0:
        ep_lengths.append(ep_step)  # final partial episode

    proc.stdin.close()
    proc.wait(timeout=5)

    elapsed = time.time() - t_start

    # Action distribution stats
    total = action_hist.sum()
    h_uniform = np.log2(N_UNIV)  # 6.087 bits for 68 actions
    p = action_hist / total
    h_actual = float(-np.sum(p[p > 0] * np.log2(p[p > 0])))
    uniformity = h_actual / h_uniform  # 1.0 = perfectly uniform

    # Action autocorrelation (lag-1)
    if len(action_seq) >= 2:
        a = np.array(action_seq, dtype=np.float64)
        ac1 = float(np.corrcoef(a[:-1], a[1:])[0, 1])
    else:
        ac1 = 0.0

    # Episode length stats
    n_eps = len(ep_lengths)
    mean_ep = float(np.mean(ep_lengths)) if ep_lengths else 0.0
    median_ep = float(np.median(ep_lengths)) if ep_lengths else 0.0

    # Chi-squared for uniformity
    expected = total / N_UNIV
    chi2 = float(np.sum((action_hist - expected) ** 2) / expected)

    print(f"  s{seed:2d}: l1={l1} uniformity={uniformity:.4f} entropy={h_actual:.3f}/{h_uniform:.3f} "
          f"ac1={ac1:.4f} chi2={chi2:.1f} eps={n_eps} ep_mean={mean_ep:.1f} t={elapsed:.1f}s", flush=True)

    return dict(seed=seed, l1=l1, action_hist=action_hist.tolist(),
                uniformity=uniformity, entropy=h_actual, ac1=ac1,
                chi2=chi2, ep_lengths=ep_lengths, n_eps=n_eps, mean_ep=mean_ep)


def main():
    try:
        sys.path.insert(0, '.')
        import arcagi3
        mk = lambda: arcagi3.make("LS20")
    except Exception as e:
        print(f"arcagi3: {e}"); return

    t_start = time.time()
    print(f"Step 718b2: candidate.c enhanced analysis on LS20")
    print(f"N_GAME_STEPS={N_GAME_STEPS}, 5 seeds")
    print(f"Kill: uniform (uniformity~1.0) + no autocorr (ac1~0) + ep_mean~150 (random baseline)")
    print()

    results = []
    for seed in range(N_SEEDS):
        results.append(run_seed(seed, mk))

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    boot_n = sum(1 for r in results if r['l1'])
    print(f"Bootloader: {boot_n}/{N_SEEDS}  total_time={elapsed:.1f}s")
    print()

    # Aggregate action histogram
    total_hist = np.zeros(N_UNIV, dtype=np.float64)
    for r in results:
        total_hist += np.array(r['action_hist'])
    total_hist /= N_SEEDS

    # Uniformity
    avg_uniformity = float(np.mean([r['uniformity'] for r in results]))
    avg_ac1 = float(np.mean([r['ac1'] for r in results]))
    avg_chi2 = float(np.mean([r['chi2'] for r in results]))
    all_ep_lens = []
    for r in results:
        all_ep_lens.extend(r['ep_lengths'])
    avg_ep = float(np.mean(all_ep_lens)) if all_ep_lens else 0.0

    print(f"Action distribution:")
    print(f"  Uniformity (1.0=uniform): {avg_uniformity:.4f}")
    print(f"  Chi-squared ({N_UNIV-1} df, p<0.05 threshold ~82): {avg_chi2:.1f}")
    print(f"  Action autocorrelation lag-1: {avg_ac1:.4f}")
    print()

    # Print full 68-action histogram (normalized to %)
    total_all = sum(r['action_hist'][a] for r in results for a in range(N_UNIV)) / N_SEEDS
    print(f"Full action histogram (% of steps, avg over seeds):")
    print(f"  Expected uniform: {100/N_UNIV:.2f}% per action")
    dir_vals = [total_hist[a] / (N_GAME_STEPS) * 100 for a in range(4)]
    print(f"  Dir actions [0-3]: {[f'{v:.1f}%' for v in dir_vals]}")
    top10 = sorted(range(N_UNIV), key=lambda a: total_hist[a], reverse=True)[:10]
    bot10 = sorted(range(N_UNIV), key=lambda a: total_hist[a])[:10]
    print(f"  Top 10: {[(a, f'{total_hist[a]/N_GAME_STEPS*100:.1f}%') for a in top10]}")
    print(f"  Bot 10: {[(a, f'{total_hist[a]/N_GAME_STEPS*100:.1f}%') for a in bot10]}")
    print()

    print(f"Episode lengths:")
    print(f"  N episodes: {len(all_ep_lens)} total ({len(all_ep_lens)/N_SEEDS:.1f}/seed)")
    print(f"  Mean: {avg_ep:.1f} steps  (random baseline LS20: ~150)")
    if all_ep_lens:
        print(f"  Min/Max: {min(all_ep_lens)}/{max(all_ep_lens)}")
        print(f"  Distribution: {sorted(all_ep_lens)[:10]}... (first 10)")
    print()

    # Kill criterion
    print(f"Kill criterion evaluation:")
    is_uniform = avg_uniformity > 0.995
    is_uncorrelated = abs(avg_ac1) < 0.02
    ep_matches_random = abs(avg_ep - 150) < 50 if all_ep_lens else True

    print(f"  Uniform distribution (>0.995): {is_uniform} ({avg_uniformity:.4f})")
    print(f"  No autocorrelation (|ac1|<0.02): {is_uncorrelated} ({avg_ac1:.4f})")
    print(f"  Episode lengths match random (~150): {ep_matches_random} ({avg_ep:.1f})")

    if is_uniform and is_uncorrelated:
        print(f"\nKILL: candidate.c = random number generator in this context.")
        print(f"  Action distribution: uniform. Autocorrelation: negligible.")
        print(f"  No structure beyond what random produces.")
    elif not is_uniform or not is_uncorrelated:
        print(f"\nSIGNAL: candidate.c shows non-random structure.")
        if not is_uniform:
            print(f"  Action distribution NOT uniform (chi2={avg_chi2:.1f})")
        if not is_uncorrelated:
            print(f"  Action autocorrelation detected (ac1={avg_ac1:.4f})")


if __name__ == "__main__":
    main()
