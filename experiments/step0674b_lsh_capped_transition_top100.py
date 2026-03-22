"""
Step 674b — Capped transition-triggered refinement (top-100 aliased cells).

Same as Step 674 (transition-triggered dual-hash), but after computing ALL
aliased cells (|successor_set| >= 2, min_visits >= 3), rank by inconsistency:
  inconsistency = max_a |successor_set(cell, a)| / visits(cell)
Keep only top 100 cells. All others use standard k=12 argmin.

10 seeds, 25s cap.
"""
import numpy as np
import time
import sys
from collections import defaultdict

K_NAV = 12
K_FINE = 20
DIM = 256
N_A = 4
REFINE_EVERY = 5000
MIN_OBS = 8
H_SPLIT = 0.05
MAX_STEPS = 500_001
PER_SEED_TIME = 25
N_SEEDS = 10
MIN_VISITS_ALIAS = 3
MAX_ALIASED = 100

BASELINE_L1 = {0: 1362, 1: 3270, 2: 48391, 3: 62727, 4: 846}


def enc_frame(frame):
    a = np.array(frame[0], dtype=np.float32) / 15.0
    x = a.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()
    return x - x.mean()


class CappedTransitionTriggered:
    def __init__(self, seed=0):
        rng = np.random.RandomState(seed)
        self.H_nav = rng.randn(K_NAV, DIM).astype(np.float32)
        self.H_fine = rng.randn(K_FINE, DIM).astype(np.float32)
        self.ref = {}
        self.G = {}   # (cell, action) -> {successor: count}
        self.C = {}
        self.live = set()
        self.G_fine = {}
        self.aliased = set()       # capped set of aliased cells
        self.candidate_alias = {}  # cell -> inconsistency score (all candidates)
        self.visit_count = defaultdict(int)
        self._pn = self._pa = self._px = None
        self._pfn = None
        self.t = 0
        self.dim = DIM
        self._cn = None
        self._fn = None
        self._recompute_interval = 2500  # recompute top-100 periodically

    def _hash_nav(self, x):
        return int(np.packbits(
            (self.H_nav @ x > 0).astype(np.uint8), bitorder='big'
        ).tobytes().hex(), 16)

    def _hash_fine(self, x):
        return int(np.packbits(
            (self.H_fine @ x > 0).astype(np.uint8), bitorder='big'
        ).tobytes().hex(), 16)

    def _node(self, x):
        n = self._hash_nav(x)
        while n in self.ref:
            n = (n, int(self.ref[n] @ x > 0))
        return n

    def _recompute_aliased(self):
        scores = {}
        for cell in self.visit_count:
            vc = self.visit_count[cell]
            if vc < MIN_VISITS_ALIAS:
                continue
            max_successors = 0
            for a in range(N_A):
                d = self.G.get((cell, a), {})
                if sum(d.values()) >= MIN_VISITS_ALIAS and len(d) >= 2:
                    max_successors = max(max_successors, len(d))
            if max_successors >= 2:
                scores[cell] = max_successors / vc
        # Keep top-MAX_ALIASED by inconsistency score
        if len(scores) <= MAX_ALIASED:
            self.aliased = set(scores.keys())
        else:
            top = sorted(scores, key=scores.get, reverse=True)[:MAX_ALIASED]
            self.aliased = set(top)

    def observe(self, frame):
        x = enc_frame(frame)
        n = self._node(x)
        fn = self._hash_fine(x)
        self.live.add(n)
        self.t += 1
        self.visit_count[n] += 1

        if self._pn is not None:
            d = self.G.setdefault((self._pn, self._pa), {})
            d[n] = d.get(n, 0) + 1
            k = (self._pn, self._pa, n)
            s, c = self.C.get(k, (np.zeros(self.dim, np.float64), 0))
            self.C[k] = (s + self._px.astype(np.float64), c + 1)

            if self._pn in self.aliased and self._pfn is not None:
                d_fine = self.G_fine.setdefault((self._pfn, self._pa), {})
                d_fine[fn] = d_fine.get(fn, 0) + 1

        self._px = x
        self._cn = n
        self._fn = fn
        if self.t > 0 and self.t % self._recompute_interval == 0:
            self._recompute_aliased()
        if self.t > 0 and self.t % REFINE_EVERY == 0:
            self._refine()
        return n

    def act(self):
        if self._cn in self.aliased and self._fn is not None:
            best_a, best_s = 0, float('inf')
            for a in range(N_A):
                s = sum(self.G_fine.get((self._fn, a), {}).values())
                if s < best_s:
                    best_s = s
                    best_a = a
            self._pn = self._cn
            self._pfn = self._fn
            self._pa = best_a
            return best_a

        best_a, best_s = 0, float('inf')
        for a in range(N_A):
            s = sum(self.G.get((self._cn, a), {}).values())
            if s < best_s:
                best_s = s
                best_a = a
        self._pn = self._cn
        self._pfn = self._fn
        self._pa = best_a
        return best_a

    def on_reset(self):
        self._pn = None
        self._pfn = None

    def _h(self, n, a):
        d = self.G.get((n, a))
        if not d or sum(d.values()) < 4:
            return 0.0
        v = np.array(list(d.values()), np.float64)
        p = v / v.sum()
        return float(-np.sum(p * np.log2(np.maximum(p, 1e-15))))

    def _refine(self):
        did = 0
        for (n, a), d in list(self.G.items()):
            if n not in self.live or n in self.ref:
                continue
            if len(d) < 2 or sum(d.values()) < MIN_OBS:
                continue
            if self._h(n, a) < H_SPLIT:
                continue
            top = sorted(d, key=d.get, reverse=True)[:2]
            r0 = self.C.get((n, a, top[0]))
            r1 = self.C.get((n, a, top[1]))
            if r0 is None or r1 is None or r0[1] < 3 or r1[1] < 3:
                continue
            diff = (r0[0] / r0[1]) - (r1[0] / r1[1])
            nm = np.linalg.norm(diff)
            if nm < 1e-8:
                continue
            self.ref[n] = (diff / nm).astype(np.float32)
            self.live.discard(n)
            did += 1
            if did >= 3:
                break


def run(seed, make):
    env = make()
    sub = CappedTransitionTriggered(seed=seed * 1000)
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

    n_aliased = len(sub.aliased)
    bsl = BASELINE_L1.get(seed)
    if l1 and bsl:
        ratio = bsl / l1
        spd = f"{ratio:.1f}x faster" if ratio > 1.0 else f"{1/ratio:.1f}x slower"
    elif l1:
        spd = "no_baseline"
    else:
        spd = "NO_L1"

    print(f"  s{seed:2d}: L1={l1} ({spd}) L2={l2} aliased={n_aliased}", flush=True)
    return dict(seed=seed, l1=l1, l2=l2)


def main():
    try:
        sys.path.insert(0, '.')
        import arcagi3
        mk = lambda: arcagi3.make("LS20")
    except (ImportError, Exception) as e:
        print(f"arcagi3: {e}")
        return

    print(f"Capped transition-triggered (top-{MAX_ALIASED}): {N_SEEDS} seeds, {PER_SEED_TIME}s cap")

    results = []
    for seed in range(N_SEEDS):
        results.append(run(seed, mk))

    print(f"\n{'='*60}")
    l1_n = sum(1 for r in results if r['l1'])
    l2_n = sum(1 for r in results if r['l2'])
    l1_valid = [(r['l1'], BASELINE_L1[r['seed']]) for r in results
                if r['l1'] and BASELINE_L1.get(r['seed'])]
    avg_ratio = float(np.mean([b / l for l, b in l1_valid])) if l1_valid else 0.0

    print(f"L1={l1_n}/{N_SEEDS}  L2={l2_n}/{N_SEEDS}  avg_speedup={avg_ratio:.2f}x")

    if l2_n > 0:
        print("BREAKTHROUGH: L2 reached")
    elif l1_n == 10:
        print("PERFECT: 10/10 seeds reach L1")
    elif l1_n >= 8:
        print("STRONG FINDING: Capped aliasing detection works")
    elif l1_n < 3:
        print("KILL: Capped transition-triggered refinement fails")
    else:
        print(f"MARGINAL: L1={l1_n}/{N_SEEDS}")


if __name__ == "__main__":
    main()
