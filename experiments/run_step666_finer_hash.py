"""
Step 666 — Finer hash at high-entropy cells.

Standard LSH k=12 for navigation. Second hash k=20 applied only at cells
where outcome entropy is in top 20% (recomputed every 5K steps).

At high-entropy cells: select action based on k=20 fine graph.
At normal cells: select from k=12 standard graph.

If yes: POMDP breakable by finer hashing at the right cells.
If no: resolution isn't the issue — hidden state isn't in the frame.

10 seeds, 25s cap.
"""
import numpy as np
import time
import sys

K_COARSE = 12
K_FINE = 20
DIM = 256
N_A = 4
REFINE_EVERY = 5000
ENTROPY_UPDATE_EVERY = 5000
MIN_OBS = 8
H_SPLIT = 0.05
MAX_STEPS = 500_001
PER_SEED_TIME = 25
N_SEEDS = 10
FINE_PERCENTILE = 80  # top 20% by entropy

BASELINE_L1 = {0: 1362, 1: 3270, 2: 48391, 3: 62727, 4: 846}


def enc_frame(frame):
    a = np.array(frame[0], dtype=np.float32) / 15.0
    x = a.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()
    return x - x.mean()


class DualHashRecode:
    def __init__(self, k_coarse=K_COARSE, k_fine=K_FINE, dim=DIM, seed=0):
        rng = np.random.RandomState(seed)
        self.H_c = rng.randn(k_coarse, dim).astype(np.float32)
        self.H_f = rng.randn(k_fine, dim).astype(np.float32)
        self.ref = {}
        self.G = {}    # coarse graph: {(n,a): {succ: count}}
        self.G_f = {}  # fine graph: {(n_fine,a): {succ_fine: count}}
        self.C = {}
        self.live = set()
        self._pn = self._pa = self._px = None
        self._pn_f = None  # prev fine node
        self.t = 0
        self.dim = dim
        self._cn = None
        self._cn_f = None
        self.high_entropy_cells = set()  # coarse cells with high outcome entropy
        self._last_entropy_update = 0

    def _hash_coarse(self, x):
        return int(np.packbits(
            (self.H_c @ x > 0).astype(np.uint8), bitorder='big'
        ).tobytes().hex(), 16)

    def _hash_fine(self, x):
        return int(np.packbits(
            (self.H_f @ x > 0).astype(np.uint8), bitorder='big'
        ).tobytes().hex(), 16)

    def _node(self, x):
        n = self._hash_coarse(x)
        while n in self.ref:
            n = (n, int(self.ref[n] @ x > 0))
        return n

    def _node_fine(self, x):
        return self._hash_fine(x)

    def observe(self, frame):
        x = enc_frame(frame)
        n = self._node(x)
        n_f = self._node_fine(x)
        self.live.add(n)
        self.t += 1
        if self._pn is not None:
            # Coarse graph
            d = self.G.setdefault((self._pn, self._pa), {})
            d[n] = d.get(n, 0) + 1
            k = (self._pn, self._pa, n)
            s, c = self.C.get(k, (np.zeros(self.dim, np.float64), 0))
            self.C[k] = (s + self._px.astype(np.float64), c + 1)
            # Fine graph (at high-entropy cells only)
            if self._pn in self.high_entropy_cells:
                d_f = self.G_f.setdefault((self._pn_f, self._pa), {})
                d_f[n_f] = d_f.get(n_f, 0) + 1
        self._px = x
        self._cn = n
        self._cn_f = n_f
        if self.t > 0 and self.t % REFINE_EVERY == 0:
            self._refine()
        if self.t > 0 and self.t % ENTROPY_UPDATE_EVERY == 0:
            self._update_high_entropy()
        return n

    def act(self):
        if self._cn in self.high_entropy_cells:
            # Use fine hash for selection
            best_a, best_s = 0, float('inf')
            for a in range(N_A):
                s = sum(self.G_f.get((self._cn_f, a), {}).values())
                if s < best_s:
                    best_s = s
                    best_a = a
        else:
            # Use coarse hash
            best_a, best_s = 0, float('inf')
            for a in range(N_A):
                s = sum(self.G.get((self._cn, a), {}).values())
                if s < best_s:
                    best_s = s
                    best_a = a
        self._pn = self._cn
        self._pn_f = self._cn_f
        self._pa = best_a
        return best_a

    def on_reset(self):
        self._pn = None
        self._pn_f = None

    def _cell_entropy(self, cell):
        ents = []
        for a in range(N_A):
            d = self.G.get((cell, a), {})
            if sum(d.values()) < 2:
                continue
            v = np.array(list(d.values()), np.float64)
            p = v / v.sum()
            ents.append(float(-np.sum(p * np.log2(np.maximum(p, 1e-15)))))
        return np.mean(ents) if ents else 0.0

    def _update_high_entropy(self):
        if not self.live:
            return
        ents = {n: self._cell_entropy(n) for n in self.live}
        if not ents:
            return
        vals = list(ents.values())
        thresh = np.percentile(vals, FINE_PERCENTILE)
        self.high_entropy_cells = {n for n, e in ents.items() if e >= thresh}

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
    sub = DualHashRecode(seed=seed * 1000)
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

    n_fine = len(sub.high_entropy_cells)
    n_total = len(sub.live)
    print(f"  s{seed:2d}: L1={l1} ({spd}) L2={l2} fine_cells={n_fine}/{n_total}", flush=True)
    return dict(seed=seed, l1=l1, l2=l2)


def main():
    try:
        sys.path.insert(0, '.')
        import arcagi3
        mk = lambda: arcagi3.make("LS20")
    except (ImportError, Exception) as e:
        print(f"arcagi3: {e}")
        return

    print(f"Finer hash at high-entropy cells: {N_SEEDS} seeds, {PER_SEED_TIME}s cap")

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
        print("BREAKTHROUGH: L2 reached — finer resolution breaks POMDP")
    elif l1_n >= 6 and avg_ratio > 1.5:
        print("FINDING: Finer hash significantly faster — resolution reveals hidden state")
    elif l1_n < 3:
        print("KILL: Dual hash fails — hidden state not in frame resolution")
    else:
        print(f"MARGINAL: L1={l1_n}/{N_SEEDS}")


if __name__ == "__main__":
    main()
