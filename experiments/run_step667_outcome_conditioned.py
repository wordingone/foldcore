"""
Step 667 — Outcome-conditioned action selection.

Standard LSH k=12 graph. BUT action selection key = (current_cell, prev_outcome).
prev_outcome = (prev_cell, prev_action, current_cell) — the transition that brought me here.

This is a 1st-order belief state: "I'm at cell X, arrived via transition Y."
Different arrival paths -> different action selections from the same cell.
Outcome carries hidden-state info that predecessor alone doesn't.

Compare to Step 649 (path-conditioned on prev cell, not prev outcome).
Outcomes carry hidden-state information that predecessors don't.

10 seeds, 25s cap.
"""
import numpy as np
import time
import sys

K = 12
DIM = 256
N_A = 4
REFINE_EVERY = 5000
MIN_OBS = 8
H_SPLIT = 0.05
MAX_STEPS = 500_001
PER_SEED_TIME = 25
N_SEEDS = 10

BASELINE_L1 = {0: 1362, 1: 3270, 2: 48391, 3: 62727, 4: 846}  # steps 653 argmin seeds 0-4


def enc_frame(frame):
    a = np.array(frame[0], dtype=np.float32) / 15.0
    x = a.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()
    return x - x.mean()


class OutcomeConditioned:
    """Graph counts keyed on (current_cell, prev_outcome) instead of current_cell."""

    def __init__(self, k=K, dim=DIM, seed=0):
        self.H = np.random.RandomState(seed).randn(k, dim).astype(np.float32)
        self.ref = {}
        self.G = {}   # {(prev_outcome, a): {successor: count}}
        self.C = {}   # {(prev_outcome, a, n2): (sum, count)}
        self.live = set()
        self._pn = self._pa = self._px = None
        self._prev_outcome = None  # (prev_cell, prev_action, current_cell)
        self.t = 0
        self.dim = dim
        self._cn = None

    def _hash(self, x):
        return int(np.packbits(
            (self.H @ x > 0).astype(np.uint8), bitorder='big'
        ).tobytes().hex(), 16)

    def _node(self, x):
        n = self._hash(x)
        while n in self.ref:
            n = (n, int(self.ref[n] @ x > 0))
        return n

    def observe(self, frame):
        x = enc_frame(frame)
        n = self._node(x)
        self.live.add(n)
        self.t += 1
        if self._pn is not None:
            # The outcome that brought us here
            outcome = (self._pn, self._pa, n)
            # Key for THIS cell's action selection = (current_cell, prev_outcome)
            sel_key = (n, self._prev_outcome)
            # Store edge from prev outcome key
            d = self.G.setdefault((self._prev_outcome, self._pa), {})
            d[n] = d.get(n, 0) + 1
            k = (self._prev_outcome, self._pa, n)
            s, c = self.C.get(k, (np.zeros(self.dim, np.float64), 0))
            self.C[k] = (s + self._px.astype(np.float64), c + 1)
            # Update prev_outcome for next step
            self._prev_outcome = outcome
        else:
            self._prev_outcome = None
        self._px = x
        self._cn = n
        if self.t > 0 and self.t % REFINE_EVERY == 0:
            self._refine()
        return n

    def act(self):
        # Action key = (current_cell, prev_outcome)
        sel_key = (self._cn, self._prev_outcome)
        best_a, best_s = 0, float('inf')
        for a in range(N_A):
            s = sum(self.G.get((sel_key, a), {}).values())
            if s < best_s:
                best_s = s
                best_a = a
        self._pn = self._cn
        self._pa = best_a
        return best_a

    def on_reset(self):
        self._pn = None
        self._prev_outcome = None

    def _h(self, key, a):
        d = self.G.get((key, a))
        if not d or sum(d.values()) < 4:
            return 0.0
        v = np.array(list(d.values()), np.float64)
        p = v / v.sum()
        return float(-np.sum(p * np.log2(np.maximum(p, 1e-15))))

    def _refine(self):
        did = 0
        keys_done = set()
        for (key, a), d in list(self.G.items()):
            # Extract cell from key: key = (cell, prev_outcome) or None
            if isinstance(key, tuple) and len(key) == 2:
                cell = key[0]
            else:
                continue
            if cell not in self.live or cell in self.ref:
                continue
            if cell in keys_done:
                continue
            if len(d) < 2 or sum(d.values()) < MIN_OBS:
                continue
            if self._h(key, a) < H_SPLIT:
                continue
            top = sorted(d, key=d.get, reverse=True)[:2]
            r0 = self.C.get((key, a, top[0]))
            r1 = self.C.get((key, a, top[1]))
            if r0 is None or r1 is None or r0[1] < 3 or r1[1] < 3:
                continue
            diff = (r0[0] / r0[1]) - (r1[0] / r1[1])
            nm = np.linalg.norm(diff)
            if nm < 1e-8:
                continue
            self.ref[cell] = (diff / nm).astype(np.float32)
            self.live.discard(cell)
            keys_done.add(cell)
            did += 1
            if did >= 3:
                break


def run(seed, make):
    env = make()
    sub = OutcomeConditioned(seed=seed * 1000)
    obs = env.reset(seed=seed)
    level = 0
    l1 = l2 = None
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
            obs = env.reset(seed=seed)
            sub.on_reset()
        if time.time() - t_start > PER_SEED_TIME:
            break

    bsl = BASELINE_L1.get(seed)
    if l1 and bsl:
        ratio = bsl / l1
        spd = f"{ratio:.1f}x faster" if ratio > 1.0 else f"{1/ratio:.1f}x slower"
    elif l1:
        spd = "no_baseline"
    else:
        spd = "NO_L1"

    print(f"  s{seed:2d}: L1={l1} ({spd}) L2={l2} unique={len(sub.live)}", flush=True)
    return dict(seed=seed, l1=l1, l2=l2)


def main():
    try:
        sys.path.insert(0, '.')
        import arcagi3
        mk = lambda: arcagi3.make("LS20")
    except (ImportError, Exception) as e:
        print(f"arcagi3: {e}")
        return

    print(f"Outcome-conditioned: {N_SEEDS} seeds, {PER_SEED_TIME}s cap")

    results = []
    for seed in range(N_SEEDS):
        results.append(run(seed, mk))

    print(f"\n{'='*60}")
    l1_n = sum(1 for r in results if r['l1'])
    l2_n = sum(1 for r in results if r['l2'])
    print(f"L1={l1_n}/{N_SEEDS}  L2={l2_n}/{N_SEEDS}")

    # Compare to argmin (653): seeds 3,4,6,10 reached L1
    argmin_seeds = {3, 4, 6, 10}
    oc_seeds = {r['seed'] for r in results if r['l1']}
    new = oc_seeds - argmin_seeds
    lost = argmin_seeds - oc_seeds

    print(f"outcome-conditioned: {sorted(oc_seeds)}")
    print(f"argmin baseline:     {sorted(argmin_seeds)}")
    print(f"new seeds: {sorted(new)}, lost seeds: {sorted(lost)}")

    if l2_n > 0:
        print("\nBREAKTHROUGH: L2 reached — outcome conditioning breaks POMDP")
    elif new:
        print(f"\nFINDING: Outcome conditioning unlocks {len(new)} new seeds — arrival path matters")
    elif lost and not new:
        print(f"\nFINDING: Outcome conditioning LOSES {len(lost)} seeds — path key too sparse")
    else:
        print(f"\nFINDING: Same seeds as argmin — outcomes add nothing over cell identity")


if __name__ == "__main__":
    main()
