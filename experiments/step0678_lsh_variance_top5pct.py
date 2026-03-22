"""
Step 678 — Variance-triggered refinement (fix Step 669).

Step 669 failed: too many high-variance cells (258-286).
Fix: ONLY cells in the TOP 5% by variance (AND 10+ visits) get fine hash.
With ~300 cells, top 5% = ~15 cells. Much more surgical.

10 seeds, 25s cap.
"""
import numpy as np
import time
import sys

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

MIN_VISITS_FOR_VAR = 10
TOP_PCT = 0.05     # top 5% by variance
TOP_K_VAR_PIXELS = 16

BASELINE_L1 = {0: 1362, 1: 3270, 2: 48391, 3: 62727, 4: 846}


def enc_frame(frame):
    a = np.array(frame[0], dtype=np.float32) / 15.0
    x = a.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()
    return x - x.mean()


class VarianceTop5Recode:
    def __init__(self, seed=0):
        rng = np.random.RandomState(seed)
        self.H_nav = rng.randn(K_NAV, DIM).astype(np.float32)
        self.H_fine = rng.randn(K_FINE, DIM).astype(np.float32)
        self.ref = {}
        self.G = {}
        self.C = {}
        self.live = set()
        # Fine graph
        self.G_fine = {}
        # Cell stats
        self.cell_sum = {}
        self.cell_sum2 = {}
        self.cell_n = {}
        # Fine-active cells (top 5%)
        self.fine_active = set()
        self._pn = self._pa = self._px = None
        self._pfn = None
        self.t = 0
        self.dim = DIM
        self._cn = None
        self._fn = None
        self._last_top5_update = 0
        self.UPDATE_INTERVAL = 2000

    def _hash_nav(self, x):
        return int(np.packbits(
            (self.H_nav @ x > 0).astype(np.uint8), bitorder='big'
        ).tobytes().hex(), 16)

    def _hash_fine_cell(self, cell, x):
        n = self.cell_n.get(cell, 0)
        if n < MIN_VISITS_FOR_VAR:
            return None
        s = self.cell_sum[cell]
        s2 = self.cell_sum2[cell]
        var = s2 / n - (s / n) ** 2
        top_idx = np.argsort(var)[-TOP_K_VAR_PIXELS:]
        fine_bits = tuple((x[top_idx] > 0).astype(int))
        return fine_bits

    def _node(self, x):
        n = self._hash_nav(x)
        while n in self.ref:
            n = (n, int(self.ref[n] @ x > 0))
        return n

    def _update_top5(self):
        """Recompute top 5% variance cells."""
        # Compute scalar variance per cell (mean of per-pixel variances)
        cell_var_scalar = {}
        for n, cnt in self.cell_n.items():
            if cnt < MIN_VISITS_FOR_VAR:
                continue
            s = self.cell_sum[n]
            s2 = self.cell_sum2[n]
            var = s2 / cnt - (s / cnt) ** 2
            cell_var_scalar[n] = float(var.mean())

        if not cell_var_scalar:
            return
        # Top 5%
        n_top = max(1, int(len(cell_var_scalar) * TOP_PCT))
        top_cells = sorted(cell_var_scalar, key=cell_var_scalar.get, reverse=True)[:n_top]
        self.fine_active = set(top_cells)

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

        # Get fine hash for current cell
        fn = self._hash_fine_cell(n, x) if n in self.fine_active else None

        if self._pn is not None:
            d = self.G.setdefault((self._pn, self._pa), {})
            d[n] = d.get(n, 0) + 1
            k = (self._pn, self._pa, n)
            s, c = self.C.get(k, (np.zeros(self.dim, np.float64), 0))
            self.C[k] = (s + self._px.astype(np.float64), c + 1)

            if self._pn in self.fine_active and self._pfn is not None and fn is not None:
                d_fine = self.G_fine.setdefault((self._pfn, self._pa), {})
                d_fine[fn] = d_fine.get(fn, 0) + 1

        self._px = x
        self._cn = n
        self._fn = fn

        # Periodically recompute top 5%
        if self.t - self._last_top5_update >= self.UPDATE_INTERVAL:
            self._update_top5()
            self._last_top5_update = self.t

        if self.t > 0 and self.t % REFINE_EVERY == 0:
            self._refine()
        return n

    def act(self):
        # Fine hash if active
        if self._cn in self.fine_active and self._fn is not None:
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
    sub = VarianceTop5Recode(seed=seed * 1000)
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

    print(f"  s{seed:2d}: L1={l1} ({spd}) L2={l2} fine_active={len(sub.fine_active)}", flush=True)
    return dict(seed=seed, l1=l1, l2=l2)


def main():
    try:
        sys.path.insert(0, '.')
        import arcagi3
        mk = lambda: arcagi3.make("LS20")
    except (ImportError, Exception) as e:
        print(f"arcagi3: {e}")
        return

    print(f"Variance top-{int(TOP_PCT*100)}% refinement: {N_SEEDS} seeds, {PER_SEED_TIME}s cap")

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
    elif l1_n >= 8:
        print("STRONG FINDING: Variance top-5% fixes Step 669")
    elif l1_n >= 6 and avg_ratio > 2.0:
        print("FINDING: Tighter variance threshold improves L1")
    elif l1_n < 3:
        print("KILL: Top-5% variance still fails")
    else:
        print(f"MARGINAL: L1={l1_n}/{N_SEEDS}")


if __name__ == "__main__":
    main()
