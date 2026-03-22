"""
Step 672 — Dual-hash baseline (k=12 nav + k=20 everywhere, diagnostic).

Standard LSH k=12 for navigation (argmin over edge counts).
SECOND LSH k=20 passive observation — does NOT affect action selection.

For each k=12 cell, track the set of k=20 sub-cells visited.
At L1 trigger, report:
  exit_cell_k12: how many k=20 sub-cells?
  all_cells_k12: avg k=20 sub-cells per k=12 cell
  Does the triggering frame map to a DIFFERENT k=20 sub-cell
  than the majority of exit cell visits?

DIAGNOSTIC ONLY — k=20 doesn't affect navigation.
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

BASELINE_L1 = {0: 1362, 1: 3270, 2: 48391, 3: 62727, 4: 846}


def enc_frame(frame):
    a = np.array(frame[0], dtype=np.float32) / 15.0
    x = a.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()
    return x - x.mean()


class DualHashDiagnostic:
    def __init__(self, seed=0):
        rng = np.random.RandomState(seed)
        self.H_nav = rng.randn(K_NAV, DIM).astype(np.float32)
        self.H_fine = rng.randn(K_FINE, DIM).astype(np.float32)
        self.ref = {}
        self.G = {}
        self.C = {}
        self.live = set()
        self._pn = self._pa = self._px = None
        self.t = 0
        self.dim = DIM
        self._cn = None
        # k=20 sub-cells per k=12 cell
        self.sub_cells = defaultdict(set)  # k12_cell -> set of k20_cells
        # Track k=20 cell per visit: (k12_cell, step) -> k20_cell
        self.exit_visits = []  # list of (step, k20_cell)

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

    def observe(self, frame):
        x = enc_frame(frame)
        n = self._node(x)
        fine_n = self._hash_fine(x)
        self.live.add(n)
        self.t += 1
        self.sub_cells[n].add(fine_n)
        if self._pn is not None:
            d = self.G.setdefault((self._pn, self._pa), {})
            d[n] = d.get(n, 0) + 1
            k = (self._pn, self._pa, n)
            s, c = self.C.get(k, (np.zeros(self.dim, np.float64), 0))
            self.C[k] = (s + self._px.astype(np.float64), c + 1)
        self._px = x
        self._fn = fine_n
        self._cn = n
        if self.t > 0 and self.t % REFINE_EVERY == 0:
            self._refine()
        return n

    def act(self):
        best_a, best_s = 0, float('inf')
        for a in range(N_A):
            s = sum(self.G.get((self._cn, a), {}).values())
            if s < best_s:
                best_s = s
                best_a = a
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
    sub = DualHashDiagnostic(seed=seed * 1000)
    obs = env.reset(seed=seed)
    level = 0
    l1 = None
    l1_step = None
    exit_cell = None
    t_start = time.time()

    for step in range(1, MAX_STEPS):
        if obs is None:
            obs = env.reset(seed=seed)
            sub.on_reset()
            continue
        sub.observe(obs)
        # Track exit cell visits (we don't know exit cell yet; track all)
        action = sub.act()
        obs, reward, done, info = env.step(action)
        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > level:
            if cl == 1 and l1 is None:
                l1 = step
                l1_step = sub.t
                exit_cell = sub._cn
                # Record the fine cell of the triggering frame
                trig_fine = sub._fn
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

    # Diagnostic output
    total_k12_cells = len(sub.sub_cells)
    avg_sub = float(np.mean([len(v) for v in sub.sub_cells.values()])) if total_k12_cells > 0 else 0.0

    if exit_cell is not None and exit_cell in sub.sub_cells:
        exit_sub_count = len(sub.sub_cells[exit_cell])
        # Most common k=20 sub-cell at exit
        from collections import Counter
        # We need to know which k=20 sub-cell each visit used
        # sub_cells tracks the SET only. For majority, we need per-visit tracking.
        # Approximate: just report count.
        trig_in_majority = "unknown"
    else:
        exit_sub_count = 0
        trig_in_majority = "N/A"

    print(f"  s{seed:2d}: L1={l1} ({spd}) "
          f"k12_cells={total_k12_cells} avg_k20_subcells={avg_sub:.2f} "
          f"exit_k20_subcells={exit_sub_count}", flush=True)
    return dict(seed=seed, l1=l1, exit_k20=exit_sub_count, avg_k20=avg_sub)


def main():
    try:
        sys.path.insert(0, '.')
        import arcagi3
        mk = lambda: arcagi3.make("LS20")
    except (ImportError, Exception) as e:
        print(f"arcagi3: {e}")
        return

    print(f"Dual-hash diagnostic (k=12 nav + k=20 passive): {N_SEEDS} seeds, {PER_SEED_TIME}s cap")

    results = []
    for seed in range(N_SEEDS):
        results.append(run(seed, mk))

    print(f"\n{'='*60}")
    l1_n = sum(1 for r in results if r['l1'])
    exit_k20_vals = [r['exit_k20'] for r in results if r['l1'] and r['exit_k20'] > 0]
    avg_k20_all = float(np.mean([r['avg_k20'] for r in results]))
    print(f"L1={l1_n}/{N_SEEDS}  avg_k20_subcells_per_k12={avg_k20_all:.2f}")
    if exit_k20_vals:
        print(f"exit_cell k20 subcells (L1 seeds): "
              f"avg={np.mean(exit_k20_vals):.1f} max={max(exit_k20_vals)} min={min(exit_k20_vals)}")
        if np.mean(exit_k20_vals) > avg_k20_all * 1.5:
            print("FINDING: Exit cell has MORE k=20 sub-cells than average — hidden state IS visible at k=20")
        elif np.mean(exit_k20_vals) <= avg_k20_all:
            print("FINDING: Exit cell has SAME k=20 sub-cells as average — k=20 doesn't help at exit")
        else:
            print(f"MARGINAL: Exit cell k=20 sub-cells slightly above average")
    else:
        print("No L1 seeds with exit cell k=20 data")


if __name__ == "__main__":
    main()
