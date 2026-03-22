"""
Step 656 — Is the graph necessary? N-gram history buffer vs graph.

No graph. No hashing. Just a history buffer of last N actions.
Action selection: argmin over action counts in buffer.

Four variants: N=3, N=5, N=10, N=20.

If N=3 > 0/10: short temporal patterns carry info (Tempest-compatible).
If N=20 > 0/10 but N=3 = 0/10: frequency estimation needs large window.
If all variants fail: graph structure is necessary, not just action history.
"""
import numpy as np
import time
import sys
from collections import deque

N_A = 4
MAX_STEPS = 500_001
PER_SEED_TIME = 7   # 4 variants × 10 seeds × 7s = 280s < 5 min
N_SEEDS = 10
N_VARIANTS = [3, 5, 10, 20]

BASELINE_L1 = [1362, 3270, 48391, 62727, 846, None, None, None, None, None]


def enc_frame(frame):
    a = np.array(frame[0], dtype=np.float32) / 15.0
    x = a.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()
    return x - x.mean()


class NGram:
    """History buffer: count actions in last N steps, argmin select."""

    def __init__(self, n=20):
        self.n = n
        self.buffer = deque(maxlen=n)  # stores actions

    def observe(self, frame):
        pass  # no state needed from obs

    def act(self):
        counts = [0] * N_A
        for a in self.buffer:
            counts[a] += 1
        action = int(np.argmin(counts))
        self.buffer.append(action)
        return action

    def on_reset(self):
        pass  # buffer persists across episodes


def run(seed, n, make):
    env = make()
    sub = NGram(n=n)
    obs = env.reset(seed=seed)
    level = 0
    l1 = l2 = None
    go = 0
    t_start = time.time()

    for step in range(1, MAX_STEPS):
        if obs is None:
            obs = env.reset(seed=seed)
            sub.on_reset()
            continue

        sub.observe(obs)
        action = sub.act()
        obs, reward, done, info = env.step(action)

        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > level:
            if cl == 1 and l1 is None:
                l1 = step
            if cl == 2 and l2 is None:
                l2 = step
            level = cl
            sub.on_reset()

        if done:
            go += 1
            obs = env.reset(seed=seed)
            sub.on_reset()

        if time.time() - t_start > PER_SEED_TIME:
            break

    bsl = BASELINE_L1[seed]
    if l1 and bsl:
        ratio = bsl / l1
        spd = f"{ratio:.1f}x faster" if ratio > 1.0 else f"{1/ratio:.1f}x slower"
    elif l1:
        spd = "no baseline"
    else:
        spd = "NO_L1"

    return dict(seed=seed, n=n, l1=l1, l2=l2, go=go, spd=spd)


def main():
    try:
        sys.path.insert(0, '.')
        import arcagi3
        mk = lambda: arcagi3.make("LS20")
    except (ImportError, Exception) as e:
        print(f"arcagi3: {e}")
        return

    print(f"N-gram buffer: N={N_VARIANTS}, {N_SEEDS} seeds, {PER_SEED_TIME}s cap")

    all_results = {}
    for n in N_VARIANTS:
        print(f"\n--- N={n} ---", flush=True)
        results = []
        for seed in range(N_SEEDS):
            r = run(seed, n, mk)
            print(f"  s{seed}: L1={r['l1']} ({r['spd']})", flush=True)
            results.append(r)
        all_results[n] = results

    print(f"\n{'='*60}")
    print("Summary:")
    for n in N_VARIANTS:
        results = all_results[n]
        l1_n = sum(1 for r in results if r['l1'])
        l1_valid = [(r['l1'], BASELINE_L1[r['seed']]) for r in results
                    if r['l1'] and BASELINE_L1[r['seed']]]
        avg_ratio = np.mean([b / l for l, b in l1_valid]) if l1_valid else 0.0
        print(f"  N={n:2d}: L1={l1_n}/10  avg_speedup={avg_ratio:.2f}x")

    # Key finding
    n3_l1 = sum(1 for r in all_results[3] if r['l1'])
    n20_l1 = sum(1 for r in all_results[20] if r['l1'])
    print()
    if n3_l1 > 0:
        print(f"SIGNAL: N=3 reaches L1 ({n3_l1}/10) — short patterns carry info")
    elif n20_l1 > 0 and n3_l1 == 0:
        print(f"FREQUENCY: N=20 works ({n20_l1}/10) but N=3 fails — needs large window")
    elif n20_l1 == 0:
        print("GRAPH NECESSARY: N-gram buffer fails even at N=20 — graph structure required")


if __name__ == "__main__":
    main()
