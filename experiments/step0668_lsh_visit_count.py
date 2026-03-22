"""
Step 668 — Multi-visit belief accumulator.

Standard LSH k=12. Add visit_count per cell.
Action selection changes with visit count:
  < 5:   argmin (standard exploration)
  5-20:  argmin over under-explored actions only (count < median)
  >= 20: random (break attractor)

652 showed exit cell visited 152 times. After 20+ visits, argmin has
equalized all actions — random break may expose different hidden states.

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

VISIT_THRESH_FOCUS = 5
VISIT_THRESH_RANDOM = 20

BASELINE_L1 = {0: 1362, 1: 3270, 2: 48391, 3: 62727, 4: 846}


def enc_frame(frame):
    a = np.array(frame[0], dtype=np.float32) / 15.0
    x = a.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()
    return x - x.mean()


class VisitCountRecode:
    def __init__(self, k=K, dim=DIM, seed=0):
        self.H = np.random.RandomState(seed).randn(k, dim).astype(np.float32)
        self.rng = np.random.RandomState(seed * 777 + 99)
        self.ref = {}
        self.G = {}
        self.C = {}
        self.live = set()
        self._pn = self._pa = self._px = None
        self.t = 0
        self.dim = dim
        self._cn = None
        self.visit_count = {}  # cell -> total visit count

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
        self.visit_count[n] = self.visit_count.get(n, 0) + 1
        self.t += 1
        if self._pn is not None:
            d = self.G.setdefault((self._pn, self._pa), {})
            d[n] = d.get(n, 0) + 1
            k = (self._pn, self._pa, n)
            s, c = self.C.get(k, (np.zeros(self.dim, np.float64), 0))
            self.C[k] = (s + self._px.astype(np.float64), c + 1)
        self._px = x
        self._cn = n
        if self.t > 0 and self.t % REFINE_EVERY == 0:
            self._refine()
        return n

    def act(self):
        vc = self.visit_count.get(self._cn, 0)

        if vc >= VISIT_THRESH_RANDOM:
            # Random break — attractor detected
            action = int(self.rng.randint(N_A))
        elif vc >= VISIT_THRESH_FOCUS:
            # Focus on under-explored actions
            counts = [sum(self.G.get((self._cn, a), {}).values()) for a in range(N_A)]
            median_c = float(np.median(counts))
            # argmin over actions below median
            under = [a for a in range(N_A) if counts[a] < median_c]
            if under:
                action = under[int(np.argmin([counts[a] for a in under]))]
            else:
                action = int(np.argmin(counts))
        else:
            # Standard argmin
            best_a, best_s = 0, float('inf')
            for a in range(N_A):
                s = sum(self.G.get((self._cn, a), {}).values())
                if s < best_s:
                    best_s = s
                    best_a = a
            action = best_a

        self._pn = self._cn
        self._pa = action
        return action

    def on_reset(self):
        self._pn = None

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
    sub = VisitCountRecode(seed=seed * 1000)
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

    # High-visit cells
    n_random = sum(1 for v in sub.visit_count.values() if v >= VISIT_THRESH_RANDOM)

    bsl = BASELINE_L1.get(seed)
    if l1 and bsl:
        ratio = bsl / l1
        spd = f"{ratio:.1f}x faster" if ratio > 1.0 else f"{1/ratio:.1f}x slower"
    elif l1:
        spd = "no_baseline"
    else:
        spd = "NO_L1"

    print(f"  s{seed:2d}: L1={l1} ({spd}) L2={l2} n_random_cells={n_random}", flush=True)
    return dict(seed=seed, l1=l1, l2=l2)


def main():
    try:
        sys.path.insert(0, '.')
        import arcagi3
        mk = lambda: arcagi3.make("LS20")
    except (ImportError, Exception) as e:
        print(f"arcagi3: {e}")
        return

    print(f"Visit-count belief accumulator: {N_SEEDS} seeds, {PER_SEED_TIME}s cap")
    print(f"Thresholds: focus at {VISIT_THRESH_FOCUS}, random at {VISIT_THRESH_RANDOM}")

    results = []
    for seed in range(N_SEEDS):
        results.append(run(seed, mk))

    print(f"\n{'='*60}")
    l1_n = sum(1 for r in results if r['l1'])
    l2_n = sum(1 for r in results if r['l2'])
    l1_valid = [(r['l1'], BASELINE_L1[r['seed']]) for r in results
                if r['l1'] and BASELINE_L1.get(r['seed'])]
    avg_ratio = np.mean([b / l for l, b in l1_valid]) if l1_valid else 0.0

    print(f"L1={l1_n}/{N_SEEDS}  L2={l2_n}/{N_SEEDS}  avg_speedup={avg_ratio:.2f}x")

    if l2_n > 0:
        print("BREAKTHROUGH: L2 reached — visit-count belief resolves POMDP")
    elif l1_n >= 6:
        print("FINDING: Visit-count belief improves L1 — repetition detection helps")
    elif l1_n < 3:
        print("KILL: Visit-count belief fails — visit count alone insufficient")
    else:
        print(f"MARGINAL: L1={l1_n}/{N_SEEDS}")


if __name__ == "__main__":
    main()
