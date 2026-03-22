"""
Step 677 — Multi-resolution ensemble (retinal cell-type diversity).

THREE LSH hashes: k=8, k=12, k=16. Three separate graphs.
All three updated on every step.

Action selection: VOTE across resolutions — each resolution's argmin gets 1 vote.
Majority wins. Ties broken by k=12 (middle resolution).

k=8: coarse, fast coverage, few cells.
k=12: standard resolution.
k=16: fine, slow coverage, many cells.

10 seeds, 25s cap.
"""
import numpy as np
import time
import sys

K8 = 8
K12 = 12
K16 = 16
DIM = 256
N_A = 4
REFINE_EVERY = 5000
MIN_OBS = 8
H_SPLIT = 0.05
MAX_STEPS = 500_001
PER_SEED_TIME = 25
N_SEEDS = 10

BASELINE_L1 = {0: 1362, 1: 3270, 2: 48391, 3: 62727, 4: 846}


def enc_frame(frame):
    a = np.array(frame[0], dtype=np.float32) / 15.0
    x = a.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()
    return x - x.mean()


class MultiResRecode:
    def __init__(self, seed=0):
        rng = np.random.RandomState(seed)
        self.H8 = rng.randn(K8, DIM).astype(np.float32)
        self.H12 = rng.randn(K12, DIM).astype(np.float32)
        self.H16 = rng.randn(K16, DIM).astype(np.float32)
        # Three separate graphs
        self.ref8 = {}; self.G8 = {}; self.C8 = {}; self.live8 = set()
        self.ref12 = {}; self.G12 = {}; self.C12 = {}; self.live12 = set()
        self.ref16 = {}; self.G16 = {}; self.C16 = {}; self.live16 = set()
        self._pn8 = self._pn12 = self._pn16 = None
        self._pa = None
        self._px = None
        self.t = 0
        self.dim = DIM
        self._cn8 = self._cn12 = self._cn16 = None

    def _hash(self, H, ref, x):
        n = int(np.packbits((H @ x > 0).astype(np.uint8), bitorder='big').tobytes().hex(), 16)
        while n in ref:
            n = (n, int(ref[n] @ x > 0))
        return n

    def _argmin(self, G, cn, pa_store):
        best_a, best_s = 0, float('inf')
        for a in range(N_A):
            s = sum(G.get((cn, a), {}).values())
            if s < best_s:
                best_s = s
                best_a = a
        return best_a

    def observe(self, frame):
        x = enc_frame(frame)
        n8 = self._hash(self.H8, self.ref8, x)
        n12 = self._hash(self.H12, self.ref12, x)
        n16 = self._hash(self.H16, self.ref16, x)
        self.live8.add(n8); self.live12.add(n12); self.live16.add(n16)
        self.t += 1

        if self._pn8 is not None:
            for G, pn, n, C, live, ref, dim in [
                (self.G8, self._pn8, n8, self.C8, self.live8, self.ref8, DIM),
                (self.G12, self._pn12, n12, self.C12, self.live12, self.ref12, DIM),
                (self.G16, self._pn16, n16, self.C16, self.live16, self.ref16, DIM),
            ]:
                d = G.setdefault((pn, self._pa), {})
                d[n] = d.get(n, 0) + 1
                k = (pn, self._pa, n)
                s, c = C.get(k, (np.zeros(dim, np.float64), 0))
                C[k] = (s + self._px.astype(np.float64), c + 1)

        self._px = x
        self._cn8 = n8; self._cn12 = n12; self._cn16 = n16

        if self.t > 0 and self.t % REFINE_EVERY == 0:
            self._refine_all()
        return n12

    def act(self):
        a8 = self._argmin(self.G8, self._cn8, None)
        a12 = self._argmin(self.G12, self._cn12, None)
        a16 = self._argmin(self.G16, self._cn16, None)

        # Majority vote
        votes = [a8, a12, a16]
        from collections import Counter
        counts = Counter(votes)
        best_count = max(counts.values())
        winners = [a for a, c in counts.items() if c == best_count]
        if len(winners) == 1:
            action = winners[0]
        else:
            # Tiebreak: a12
            action = a12

        self._pn8 = self._cn8
        self._pn12 = self._cn12
        self._pn16 = self._cn16
        self._pa = action
        return action

    def on_reset(self):
        self._pn8 = self._pn12 = self._pn16 = None

    def _refine_one(self, G, C, ref, live):
        did = 0
        for (n, a), d in list(G.items()):
            if n not in live or n in ref:
                continue
            if len(d) < 2 or sum(d.values()) < MIN_OBS:
                continue
            v = np.array(list(d.values()), np.float64)
            p = v / v.sum()
            h = float(-np.sum(p * np.log2(np.maximum(p, 1e-15))))
            if h < H_SPLIT:
                continue
            top = sorted(d, key=d.get, reverse=True)[:2]
            r0 = C.get((n, a, top[0]))
            r1 = C.get((n, a, top[1]))
            if r0 is None or r1 is None or r0[1] < 3 or r1[1] < 3:
                continue
            diff = (r0[0] / r0[1]) - (r1[0] / r1[1])
            nm = np.linalg.norm(diff)
            if nm < 1e-8:
                continue
            ref[n] = (diff / nm).astype(np.float32)
            live.discard(n)
            did += 1
            if did >= 3:
                break

    def _refine_all(self):
        self._refine_one(self.G8, self.C8, self.ref8, self.live8)
        self._refine_one(self.G12, self.C12, self.ref12, self.live12)
        self._refine_one(self.G16, self.C16, self.ref16, self.live16)


def run(seed, make):
    env = make()
    sub = MultiResRecode(seed=seed * 1000)
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

    print(f"  s{seed:2d}: L1={l1} ({spd}) L2={l2}", flush=True)
    return dict(seed=seed, l1=l1, l2=l2)


def main():
    try:
        sys.path.insert(0, '.')
        import arcagi3
        mk = lambda: arcagi3.make("LS20")
    except (ImportError, Exception) as e:
        print(f"arcagi3: {e}")
        return

    print(f"Multi-resolution ensemble (k=8/12/16 vote): {N_SEEDS} seeds, {PER_SEED_TIME}s cap")

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
        print("BREAKTHROUGH: L2 reached — multi-resolution resolves POMDP")
    elif l1_n >= 8:
        print("STRONG FINDING: Multi-resolution ensemble works")
    elif l1_n >= 6 and avg_ratio > 2.0:
        print("FINDING: Multi-resolution improves L1")
    elif l1_n < 3:
        print("KILL: Multi-resolution vote fails")
    else:
        print(f"MARGINAL: L1={l1_n}/{N_SEEDS}")


if __name__ == "__main__":
    main()
