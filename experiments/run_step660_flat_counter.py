"""
Step 660 — Graph vs flat counter (same 20 seeds as Step 653).

Replace graph with flat counter: {cell_id: [count_a0, count_a1, count_a2, count_a3]}.
No edges, no transitions, no successors. Just cell + action counts.

Expected: IDENTICAL to argmin since sum_n G[(cell,a,n)] = counts[(cell,a)].
If ANY seed differs: graph contributes through tie-breaking or hash refinement.
"""
import numpy as np
import time
import sys
from collections import defaultdict

K = 12
DIM = 256
N_A = 4
MAX_STEPS = 500_001
PER_SEED_TIME = 5
N_SEEDS = 20


def enc_frame(frame):
    a = np.array(frame[0], dtype=np.float32) / 15.0
    x = a.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()
    return x - x.mean()


class FlatCounter:
    """No graph. Just cell -> action counts. No refinement."""

    def __init__(self, k=K, dim=DIM, seed=0):
        self.H = np.random.RandomState(seed).randn(k, dim).astype(np.float32)
        self.counts = defaultdict(lambda: [0] * N_A)
        self._cn = None

    def _hash(self, x):
        return int(np.packbits(
            (self.H @ x > 0).astype(np.uint8), bitorder='big'
        ).tobytes().hex(), 16)

    def observe(self, frame):
        x = enc_frame(frame)
        self._cn = self._hash(x)
        return self._cn

    def act(self):
        action = int(np.argmin(self.counts[self._cn]))
        self.counts[self._cn][action] += 1
        return action

    def on_reset(self):
        pass


def run(seed, make):
    env = make()
    sub = FlatCounter(seed=seed * 1000)
    obs = env.reset(seed=seed)
    level = 0
    l1 = None
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
            level = cl
            sub.on_reset()
        if done:
            go += 1
            obs = env.reset(seed=seed)
            sub.on_reset()
        if time.time() - t_start > PER_SEED_TIME:
            break

    print(f"  s{seed:2d}: L1={l1}", flush=True)
    return dict(seed=seed, l1=l1)


# Step 653 argmin results for comparison
ARGMIN_653 = {
    0: None, 1: None, 2: None, 3: 220, 4: 132,
    5: None, 6: 950, 7: None, 8: None, 9: None,
    10: 900, 11: None, 12: None, 13: None, 14: None,
    15: None, 16: None, 17: None, 18: None, 19: None,
}


def main():
    try:
        sys.path.insert(0, '.')
        import arcagi3
        mk = lambda: arcagi3.make("LS20")
    except (ImportError, Exception) as e:
        print(f"arcagi3: {e}")
        return

    print(f"Flat counter: {N_SEEDS} seeds, {PER_SEED_TIME}s cap")

    results = []
    for seed in range(N_SEEDS):
        results.append(run(seed, mk))

    print(f"\n{'='*60}")
    from collections import Counter
    tags = Counter()
    diffs = []
    for r in results:
        seed = r['seed']
        fc_l1 = r['l1']
        arg_l1 = ARGMIN_653[seed]
        if fc_l1 == arg_l1:
            tag = "identical"
        elif fc_l1 and not arg_l1:
            tag = "flat_only"
        elif not fc_l1 and arg_l1:
            tag = "argmin_only"
        else:
            tag = f"both_diff(fc={fc_l1},arg={arg_l1})"
            diffs.append(seed)
        tags[tag] += 1
        if tag != "identical":
            print(f"  s{seed}: flat={fc_l1} argmin={arg_l1} -> {tag}", flush=True)

    print(f"\nidentical: {tags['identical']}/{N_SEEDS}")
    print(f"flat_only: {tags['flat_only']}, argmin_only: {tags['argmin_only']}")

    if tags['identical'] == N_SEEDS:
        print("\nCONCLUSION: Graph = counter. Transition structure contributes NOTHING.")
    else:
        print(f"\nCONCLUSION: Graph differs from counter on {N_SEEDS - tags['identical']} seeds.")
        print("Graph contributes through refinement or tie-breaking.")


if __name__ == "__main__":
    main()
