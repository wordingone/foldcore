"""
Step 584 — 581d seed expansion: 20 seeds on LS20.

Addresses reviewer gap: 5 seeds insufficient for statistical rigor.
Config identical to 581d (permanent soft penalty, PENALTY=100).
TIME_CAP=280 per seed. Fisher exact test vs argmin baseline.

FAST_MODE=True  → 10K steps (signal check, ~36 min total)
FAST_MODE=False → 50K steps (full validation, ~3 hrs — needs Jun sign-off)

Output: X/20 seeds L1, Y/20 seeds L1, Fisher exact p-value.
"""
import time
import numpy as np
import sys

FAST_MODE = True  # set False for full 50K run (needs Jun approval)

K = 12
DIM = 256
N_A = 4
MAX_STEPS = 10_000 if FAST_MODE else 50_000
TIME_CAP = 60 if FAST_MODE else 280   # per seed
N_SEEDS = 20
PENALTY = 100


# ── LSH hashing ──────────────────────────────────────────────────────────────

def encode(frame, H):
    arr = np.array(frame[0], dtype=np.float32)
    x = arr.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten() / 15.0
    x -= x.mean()
    bits = (H @ x > 0).astype(np.int64)
    return int(np.dot(bits, 1 << np.arange(K)))


# ── Soft penalty substrate (identical to 581d) ────────────────────────────────

class SoftPenaltySub:
    def __init__(self, lsh_seed=0):
        self.H = np.random.RandomState(lsh_seed).randn(K, DIM).astype(np.float32)
        self.G = {}
        self.death_edges = set()
        self._prev_node = None
        self._prev_action = None
        self.cells = set()
        self.total_deaths = 0

    def observe(self, frame):
        node = encode(frame, self.H)
        self.cells.add(node)
        self._curr_node = node
        if self._prev_node is not None:
            d = self.G.setdefault((self._prev_node, self._prev_action), {})
            d[node] = d.get(node, 0) + 1

    def on_death(self):
        if self._prev_node is not None:
            self.death_edges.add((self._prev_node, self._prev_action))
            self.total_deaths += 1

    def act(self):
        node = self._curr_node
        counts = np.array([sum(self.G.get((node, a), {}).values()) for a in range(N_A)],
                          dtype=np.float64)
        penalized = counts.copy()
        for a in range(N_A):
            if (node, a) in self.death_edges:
                penalized[a] += PENALTY
        action = int(np.argmin(penalized))
        self._prev_node = node
        self._prev_action = action
        return action

    def on_reset(self):
        self._prev_node = None
        self._prev_action = None


# ── Argmin baseline ───────────────────────────────────────────────────────────

class ArgminSub:
    def __init__(self, lsh_seed=0):
        self.H = np.random.RandomState(lsh_seed).randn(K, DIM).astype(np.float32)
        self.G = {}
        self._prev_node = None
        self._prev_action = None
        self.cells = set()

    def observe(self, frame):
        node = encode(frame, self.H)
        self.cells.add(node)
        self._curr_node = node

    def act(self):
        node = self._curr_node
        counts = [sum(self.G.get((node, a), {}).values()) for a in range(N_A)]
        action = int(np.argmin(counts))
        if self._prev_node is not None:
            d = self.G.setdefault((self._prev_node, self._prev_action), {})
            d[node] = d.get(node, 0) + 1
        self._prev_node = node
        self._prev_action = action
        return action

    def on_reset(self):
        self._prev_node = None
        self._prev_action = None

    def on_death(self):
        pass


# ── Seed runner ───────────────────────────────────────────────────────────────

def run_seed(mk, seed, SubClass, time_cap=TIME_CAP):
    env = mk()
    sub = SubClass(lsh_seed=seed * 100 + 7)
    obs = env.reset(seed=seed)
    sub.on_reset()

    prev_cl = 0; fresh = True
    l1 = l2 = go = step = 0
    t0 = time.time()

    while step < MAX_STEPS and time.time() - t0 < time_cap:
        if obs is None:
            obs = env.reset(seed=seed)
            sub.on_reset()
            prev_cl = 0; fresh = True; go += 1
            continue

        sub.observe(obs)
        action = sub.act()
        obs, _, done, info = env.step(action)
        step += 1

        if done:
            sub.on_death()
            obs = env.reset(seed=seed)
            sub.on_reset()
            prev_cl = 0; fresh = True; go += 1
            continue

        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if fresh:
            prev_cl = cl; fresh = False
        elif cl >= 1 and prev_cl < 1:
            l1 += 1
            if l1 <= 2:
                print(f"    s{seed} L1@{step}", flush=True)
        elif cl >= 2 and prev_cl < 2:
            l2 += 1
        prev_cl = cl

    elapsed = time.time() - t0
    cells = len(sub.cells)
    deaths = getattr(sub, 'total_deaths', 0)
    print(f"  s{seed}: L1={l1} L2={l2} go={go} step={step} cells={cells} deaths={deaths} {elapsed:.0f}s",
          flush=True)
    return dict(seed=seed, l1=l1, l2=l2, go=go, steps=step, cells=cells)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    try:
        from scipy.stats import fisher_exact
        HAS_SCIPY = True
    except ImportError:
        HAS_SCIPY = False
        print("WARNING: scipy not available, skipping Fisher test", flush=True)

    try:
        sys.path.insert(0, '.')
        import arcagi3
        mk = lambda: arcagi3.make("LS20")
    except Exception as e:
        print(f"arcagi3: {e}"); return

    mode_str = "FAST (10K steps)" if FAST_MODE else "FULL (50K steps)"
    print(f"Step 584: Seed expansion — {N_SEEDS} seeds, LS20 [{mode_str}]", flush=True)
    print(f"  K={K} MAX_STEPS={MAX_STEPS} PENALTY={PENALTY} TIME_CAP={TIME_CAP}s/seed", flush=True)

    sp_results = []
    t_total = time.time()

    print("\n--- SoftPenalty (581d config) ---", flush=True)
    for seed in range(N_SEEDS):
        elapsed_total = time.time() - t_total
        print(f"\nseed {seed} (total {elapsed_total:.0f}s):", flush=True)
        r = run_seed(mk, seed, SoftPenaltySub)
        sp_results.append(r)

    print("\n--- Argmin baseline ---", flush=True)
    am_results = []
    for seed in range(N_SEEDS):
        elapsed_total = time.time() - t_total
        print(f"\nseed {seed} (total {elapsed_total:.0f}s):", flush=True)
        r = run_seed(mk, seed, ArgminSub)
        am_results.append(r)

    sp_l1 = sum(r['l1'] for r in sp_results)
    sp_seeds = sum(1 for r in sp_results if r['l1'] > 0)
    am_l1 = sum(r['l1'] for r in am_results)
    am_seeds = sum(1 for r in am_results if r['l1'] > 0)

    print(f"\n{'='*60}")
    print(f"Step 584: Seed expansion ({N_SEEDS} seeds, {mode_str})")
    print(f"  SoftPenalty: {sp_seeds}/{N_SEEDS} seeds L1, total L1={sp_l1}")
    for r in sp_results:
        print(f"    s{r['seed']:02d}: L1={r['l1']} cells={r['cells']}")
    print(f"  Argmin:      {am_seeds}/{N_SEEDS} seeds L1, total L1={am_l1}")
    for r in am_results:
        print(f"    s{r['seed']:02d}: L1={r['l1']} cells={r['cells']}")

    # Fisher exact test: proportion of seeds reaching L1
    if HAS_SCIPY:
        # 2x2 table: [sp_l1_seeds, sp_not] vs [am_l1_seeds, am_not]
        table = [[sp_seeds, N_SEEDS - sp_seeds],
                 [am_seeds, N_SEEDS - am_seeds]]
        odds, pval = fisher_exact(table, alternative='greater')
        print(f"\n  Fisher exact (SP > argmin): odds={odds:.3f} p={pval:.4f}")
        if pval < 0.05:
            print(f"  SIGNIFICANT: p={pval:.4f} < 0.05")
        elif pval < 0.10:
            print(f"  TREND: p={pval:.4f} < 0.10")
        else:
            print(f"  NOT SIGNIFICANT: p={pval:.4f}")
    else:
        print(f"\n  SoftPenalty: {sp_seeds}/{N_SEEDS}, Argmin: {am_seeds}/{N_SEEDS}")
        print(f"  (install scipy for Fisher exact test)")

    total_elapsed = time.time() - t_total
    print(f"\n  Total elapsed: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
