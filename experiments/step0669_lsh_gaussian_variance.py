"""
Step 669 — Gaussian-inspired observation (frame variance drives refinement).

Standard LSH k=12. At each cell, maintain running mean/variance of frames.
High-variance pixels = something changes at this cell that hash doesn't capture.

Action selection: standard argmin. BUT at high-variance cells:
  compute fine_hash = sign-based hash of high-variance pixels only.
  If fine_hash is new for (cell, action): prefer this action (novel hidden state).

This is the Gaussian splatting principle: variance identifies uncertainty,
then refine at uncertain locations. The cell is a Gaussian with mean + variance,
variance drives further resolution.

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

VAR_THRESH = 0.01    # min pixel variance to be "high variance"
MIN_VISITS_FOR_VAR = 10   # need this many visits before using variance
TOP_K_VAR_PIXELS = 16     # use top-16 variance pixels for fine hash

BASELINE_L1 = {0: 1362, 1: 3270, 2: 48391, 3: 62727, 4: 846}


def enc_frame(frame):
    a = np.array(frame[0], dtype=np.float32) / 15.0
    x = a.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()
    return x - x.mean()


class GaussianRecode:
    def __init__(self, k=K, dim=DIM, seed=0):
        self.H = np.random.RandomState(seed).randn(k, dim).astype(np.float32)
        self.ref = {}
        self.G = {}
        self.C = {}
        self.live = set()
        self._pn = self._pa = self._px = None
        self.t = 0
        self.dim = dim
        self._cn = None
        # Running stats per cell
        self.cell_sum = {}    # cell -> sum of frames (float64 array)
        self.cell_sum2 = {}   # cell -> sum of frames^2
        self.cell_n = {}      # cell -> visit count
        # Fine hash memory: (cell, action, fine_hash_bits) -> count
        self.fine_seen = {}   # (cell, action) -> set of fine_hash tuples

    def _hash(self, x):
        return int(np.packbits(
            (self.H @ x > 0).astype(np.uint8), bitorder='big'
        ).tobytes().hex(), 16)

    def _node(self, x):
        n = self._hash(x)
        while n in self.ref:
            n = (n, int(self.ref[n] @ x > 0))
        return n

    def _get_fine_hash(self, cell, x):
        """Hash using top-variance pixels at this cell. Returns None if not enough data."""
        n = self.cell_n.get(cell, 0)
        if n < MIN_VISITS_FOR_VAR:
            return None
        s = self.cell_sum[cell]
        s2 = self.cell_sum2[cell]
        var = s2 / n - (s / n) ** 2
        if var.max() < VAR_THRESH:
            return None
        # Top-K high-variance pixels
        top_idx = np.argsort(var)[-TOP_K_VAR_PIXELS:]
        # Fine hash: sign of high-variance pixels
        fine_bits = tuple((x[top_idx] > 0).astype(int))
        return fine_bits

    def observe(self, frame):
        x = enc_frame(frame)
        n = self._node(x)
        self.live.add(n)
        self.t += 1
        # Update cell stats
        if n not in self.cell_sum:
            self.cell_sum[n] = np.zeros(self.dim, np.float64)
            self.cell_sum2[n] = np.zeros(self.dim, np.float64)
            self.cell_n[n] = 0
        self.cell_sum[n] += x.astype(np.float64)
        self.cell_sum2[n] += x.astype(np.float64) ** 2
        self.cell_n[n] += 1
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
        # Get fine hash for current cell
        fine_hash = self._get_fine_hash(self._cn, self._px)

        if fine_hash is not None:
            # Find actions with unseen fine_hash
            novel_actions = []
            for a in range(N_A):
                key = (self._cn, a)
                seen = self.fine_seen.get(key, set())
                if fine_hash not in seen:
                    novel_actions.append(a)

            if novel_actions:
                # Among novel fine-hash actions, prefer least-explored
                best_a = novel_actions[0]
                best_s = float('inf')
                for a in novel_actions:
                    s = sum(self.G.get((self._cn, a), {}).values())
                    if s < best_s:
                        best_s = s
                        best_a = a
                # Record this fine hash as seen for this action
                key = (self._cn, best_a)
                if key not in self.fine_seen:
                    self.fine_seen[key] = set()
                self.fine_seen[key].add(fine_hash)
                self._pn = self._cn
                self._pa = best_a
                return best_a

        # Standard argmin fallback
        best_a, best_s = 0, float('inf')
        for a in range(N_A):
            s = sum(self.G.get((self._cn, a), {}).values())
            if s < best_s:
                best_s = s
                best_a = a

        if fine_hash is not None:
            key = (self._cn, best_a)
            if key not in self.fine_seen:
                self.fine_seen[key] = set()
            self.fine_seen[key].add(fine_hash)

        self._pn = self._cn
        self._pa = best_a
        return best_a

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
    sub = GaussianRecode(seed=seed * 1000)
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

    n_hivar = sum(1 for n, v in sub.cell_n.items() if v >= MIN_VISITS_FOR_VAR)
    print(f"  s{seed:2d}: L1={l1} ({spd}) L2={l2} hi_var_cells={n_hivar}", flush=True)
    return dict(seed=seed, l1=l1, l2=l2)


def main():
    try:
        sys.path.insert(0, '.')
        import arcagi3
        mk = lambda: arcagi3.make("LS20")
    except (ImportError, Exception) as e:
        print(f"arcagi3: {e}")
        return

    print(f"Gaussian variance refinement: {N_SEEDS} seeds, {PER_SEED_TIME}s cap")

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
        print("BREAKTHROUGH: L2 reached — Gaussian variance breaks POMDP")
    elif l1_n >= 6 and avg_ratio > 1.5:
        print("FINDING: Gaussian variance refinement faster — variance IS the signal")
    elif l1_n < 3:
        print("KILL: Variance-driven refinement fails — variance in wrong pixels")
    else:
        print(f"MARGINAL: L1={l1_n}/{N_SEEDS}")


if __name__ == "__main__":
    main()
